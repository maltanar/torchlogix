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
