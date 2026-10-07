import numpy as np
import onnx
import pytest
import torch
from onnx import numpy_helper

from torchlogix import onnx_export
from torchlogix.layers import GroupSum, LogicDense
from torchlogix.utils import set_export_mode

DOMAIN = onnx_export.LOOKUP_TABLE_DOMAIN


def _reference_lookup_table(x, indices, table, input_bits):
    """Spec semantics: addr[..., m] = sum_k x[..., indices[m, k]] * 2**(input_bits*k)."""
    fan_in = x[..., indices].astype(np.int64)
    k = indices.shape[1]
    addr = (fan_in * (2 ** (input_bits * np.arange(k)))).sum(-1)
    m, s = table.shape
    return table.reshape(-1)[addr + np.arange(m) * s]


@pytest.fixture
def mlp():
    torch.manual_seed(0)
    model = torch.nn.Sequential(
        torch.nn.Flatten(),
        LogicDense(64, 40),
        LogicDense(40, 40),
        GroupSum(k=10, tau=8),
    )
    for module in model.modules():
        if isinstance(module, LogicDense):
            with torch.no_grad():
                module.weight.copy_(torch.randn_like(module.weight))
    model.eval()
    return model


def test_export_emits_lookup_table_nodes(mlp, tmp_path):
    x = torch.rand(3, 1, 8, 8) > 0.5
    path = tmp_path / "mlp.onnx"

    onnx_export.export(mlp, (x,), str(path))

    model = onnx.load(str(path))
    onnx.checker.check_model(model)
    lut_nodes = [n for n in model.graph.node if n.domain == DOMAIN]
    assert len(lut_nodes) == 2
    assert all(n.op_type == "LookupTable" for n in lut_nodes)
    assert {(a.name, a.i) for a in lut_nodes[0].attribute} == {
        ("input_bits", 1),
        ("out_bits", 1),
    }
    assert (DOMAIN, onnx_export.LOOKUP_TABLE_OPSET) in {
        (o.domain, o.version) for o in model.opset_import
    }


def test_export_emits_binary_lookup_table_conv_attributes(conv2d_model_wo_group_sum, sample_input_2d, tmp_path):
    path = tmp_path / "conv.onnx"

    onnx_export.export(conv2d_model_wo_group_sum, (sample_input_2d,), str(path))

    model = onnx.load(str(path))
    conv_nodes = [
        node for node in model.graph.node if node.domain == DOMAIN and node.op_type == "LookupTableConv"
    ]
    assert len(conv_nodes) == 1
    assert ("out_bits", 1) in {(attribute.name, attribute.i) for attribute in conv_nodes[0].attribute}


def test_exported_lookup_tables_match_eager_model(mlp, tmp_path):
    x = torch.rand(3, 1, 8, 8) > 0.5
    path = tmp_path / "mlp.onnx"
    set_export_mode(mlp, True)
    expected = mlp(x).numpy()

    onnx_export.export(mlp, (x,), str(path))

    model = onnx.load(str(path))
    init = {i.name: numpy_helper.to_array(i) for i in model.graph.initializer}
    h = x.numpy().reshape(3, 64).astype(np.uint8)
    for prefix in ("1", "2"):
        h = _reference_lookup_table(
            h,
            init[f"{prefix}._export_lut_indices"],
            init[f"{prefix}._export_lut_table"],
            1,
        )
    actual = h.reshape(3, 10, 4).astype(np.int64).sum(-1) / 8.0

    assert np.allclose(actual, expected)


@pytest.mark.parametrize("lut_rank", [2, 4])
def test_exported_lookup_table_conv_matches_eager_model(lut_rank, tmp_path):
    from torchlogix.layers import LogicConv2d

    torch.manual_seed(0)
    conv = LogicConv2d(
        in_dim=8,
        channels=3,
        num_kernels=4,
        receptive_field_size=3,
        tree_depth=2,
        lut_rank=lut_rank,
        parametrization="warp",
    )
    conv.eval()

    x = (torch.rand(2, 3, 8, 8) > 0.5).float()
    expected = conv(x).detach().numpy()

    path = tmp_path / f"conv_rank{lut_rank}.onnx"
    onnx_export.export(conv, (x,), str(path))

    model = onnx.load(str(path))
    onnx.checker.check_model(model)

    init = {i.name: numpy_helper.to_array(i) for i in model.graph.initializer}
    indices = torch.tensor(init["_export_lut_conv_indices"])
    table = torch.tensor(init["_export_lut_conv_table"])

    x_cl = x.movedim(1, -1).to(torch.uint8)
    actual = onnx_export.lookup_table_conv(
        x_cl, indices, table, 2, [3, 3], [1, 1], [0, 0, 0, 0]
    ).movedim(-1, 1).numpy()

    assert np.array_equal(actual, expected)

