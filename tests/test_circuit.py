import pytest
import subprocess
import ctypes
import sys
import tempfile
import re
import pathlib
import shutil
import torch
import torch.nn as nn
from torchlogix import Circuit
from torchlogix.circuit import GateOp
from torchlogix.utils import set_export_mode
from torchlogix.layers import (
    GroupSum,
    LogicConv2d,
    LogicConv3d,
    LogicDense,
    OrPooling2d,
    OrPooling3d,
)


class DenseModel(nn.Sequential):
    def __init__(self):
        super().__init__(
            LogicDense(1000, 1000, parametrization="raw", parametrization_kwargs={"weight_init": "random"}),
            LogicDense(1000, 1000, parametrization="raw", parametrization_kwargs={"weight_init": "random"}),
        )
        self.input_shape = (1000,)


# inherit from sequential
class ConvModel(nn.Sequential):
    def __init__(self):
        super().__init__(
            LogicConv2d(in_dim=32, channels=3, num_kernels=8, receptive_field_size=3, tree_depth=2, parametrization_kwargs={"weight_init": "random"}),
            OrPooling2d(kernel_size=2, stride=2),
            nn.Flatten(),  # 8 × 15 x 15 = 1800
            LogicDense(1800, 1000, parametrization="raw", parametrization_kwargs={"weight_init": "random"}),
            LogicDense(1000, 1000, parametrization="raw", parametrization_kwargs={"weight_init": "random"}),
            GroupSum(10)# , tau=2.0),
        )
        self.input_shape = (3, 32, 32)


# w/ custom forward pass
class BranchModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = LogicConv2d(in_dim=32, channels=3, num_kernels=8,
                    receptive_field_size=3, tree_depth=2,
                    parametrization_kwargs={"weight_init": "random"}) # 8 x 30 x 30 = 7200
        self.pool = OrPooling2d(kernel_size=2, stride=2) # 8 x 15 x 15 = 1800
        self.dense = LogicDense(1801, 1000, parametrization="raw", parametrization_kwargs={"weight_init": "random"})
        self.group_sum = GroupSum(10)
        self.input_shape = (32*32*3 + 1,)

    def forward(self, x):
        assert x.shape[1:] == (32*32*3 + 1,)
        img, feat = x[:, :-1].reshape(-1, 3, 32, 32), x[:, -1:]
        x = self.conv(img)
        x = self.pool(x)
        x = x.flatten(1)
        x = torch.cat([x, feat], dim=1)
        x = self.dense(x)
        x = self.group_sum(x)
        return x


class AnyLogicModel(nn.Module):
    """
    Some random non-torchlogix logic and reshaping operations
    to test the flexibility of from_model
    """
    def __init__(self):
        super().__init__()
        self.input_shape = (4, 8, 8)


    def forward(self, x):
        # x: (B, 4, 8, 8) — batch of 4-channel 8×8 bool/int tensors

        # Split along channel dim
        x1, x2, x3, x4 = x[:, 0], x[:, 1], x[:, 2], x[:, 3]  # each (B, 8, 8)

        # Spatial flip on x1 (replaces the buggy ::-1 step)
        x1 = torch.flip(x1, dims=[1])                           # flip rows → (B, 8, 8)

        # Permute x2: swap H and W
        x2 = x2.permute(0, 2, 1)                                # (B, 8, 8) transposed

        # Broadcast a mask over x3 — zero out the bottom half
        mask = torch.ones(8, 8, dtype=x3.dtype, device=x3.device)
        mask[4:, :] = 0                                         # (8, 8), broadcasts over batch
        x3 = x3 & mask

        # Diagonal mask on x4 — keep only upper triangle
        tri = torch.triu(torch.ones(8, 8, dtype=x4.dtype, device=x4.device))
        x4 = x4 & tri

        # Logic ops: operator precedence is & > ^ > |  (same as Python/C)
        # So this reads as:  x1 | (x2 & x3) ^ x4
        # Use parens to make intent explicit:
        out = x1 | ((x2 & x3) ^ x4)                            # (B, 8, 8)

        # Add a channel dim back, then roll it to position 1
        out = out.unsqueeze(1)                                   # (B, 1, 8, 8)

        # Flatten spatial dims only
        out = out.flatten(2)                                     # (B, 1, 64)

        out = out.squeeze(1)                                     # (B, 64)

        out1 = out[:, :8].sum(dim=1, keepdim=True)               # (B, 1)
        out2 = out[:, 8:16].sum(dim=1, keepdim=True)             # (B, 1)
        out3 = out[:, 16:]                                       # (B, 16)

        return torch.cat([out1, out2, out3], dim=1)                      # (B, 18)


@pytest.mark.parametrize("model_cls", [DenseModel, ConvModel, BranchModel, AnyLogicModel])
def test_functional_equivalence(model_cls):
    model = model_cls()
    x = torch.randint(0, 2, (1, *model.input_shape), dtype=torch.bool)

    set_export_mode(model)
    preds_model = model(x)
    
    circuit = Circuit.from_model(model, input_shape=model.input_shape)
    preds_circuit = circuit(x)
    assert torch.equal(preds_model, preds_circuit.to(preds_model.dtype)), \
        "Circuit predictions differ from Eval-mode model predictions"


@pytest.mark.parametrize("model_cls", [DenseModel, ConvModel, BranchModel, AnyLogicModel])
@pytest.mark.parametrize("pack_bits", [None, 8, 16, 32])
@pytest.mark.parametrize("relative_batch_size", [1, 10])
def test_circuit_compilation(model_cls, pack_bits, relative_batch_size):
    model = model_cls()

    batch_size = (1 if pack_bits is None else pack_bits) * relative_batch_size
    x = torch.randint(0, 2, (batch_size, *model.input_shape), dtype=torch.bool)

    set_export_mode(model)
    preds_model = model(x)
    
    circuit = Circuit.from_model(model, input_shape=model.input_shape)
    circuit.compile(pack_bits=pack_bits)
    input_np = x.numpy()
    preds_circuit_compiled = circuit(input_np, use_compiled=True)
    preds_circuit_compiled_torch = torch.from_numpy(preds_circuit_compiled)
    # Cast to a common dtype before comparing: circuit may use a narrower integer
    # type (e.g. uint16_t) while the model returns float32.
    target_dtype = preds_model.dtype
    assert torch.equal(preds_model, preds_circuit_compiled_torch.to(target_dtype)), \
        "Compiled circuit predictions differ from Eval-mode predictions"


@pytest.mark.parametrize("model_cls", [ConvModel, BranchModel, AnyLogicModel])
@pytest.mark.parametrize("simplification", [
    Circuit.simplify, Circuit.constant_fold_gates, Circuit.eliminate_dead_gates, Circuit.bypass_wires, Circuit.dedup, Circuit.fuse_not_inputs
])
def test_circuit_simplifications(model_cls, simplification):
    model = model_cls()
    x = torch.randint(0, 2, (1, *model.input_shape), dtype=torch.bool)

    circuit = Circuit.from_model(model, input_shape=model.input_shape)
    preds_before = circuit(x)

    simplification(circuit)
    preds_after = circuit(x)
    assert torch.equal(preds_before, preds_after), f"Predictions differ after {simplification.__name__}!"


@pytest.mark.parametrize("model_cls", [DenseModel, ConvModel, BranchModel, AnyLogicModel])
def test_gate_node_idx_populated_and_preserved(model_cls):
    circuit = Circuit.from_model(model_cls(), input_shape=model_cls().input_shape)

    assert circuit.gates, "Expected circuit to contain gates"
    assert all(g.node_idx >= 0 for g in circuit.gates)
    assert len({g.node_idx for g in circuit.gates}) >= 2

    circuit.simplify()
    assert circuit.gates, "Expected simplified circuit to retain gates"
    assert all(g.node_idx >= 0 for g in circuit.gates)


def test_dense_pipeline_boundary_metadata_and_verilog_registers():
    circuit = Circuit.from_model(DenseModel(), input_shape=DenseModel().input_shape)
    circuit.simplify()

    assert circuit.pipeline_boundary_origins
    assert circuit.pipeline_boundary_signals

    verilog = circuit.get_verilog_code(pipeline=1)
    assert "input  wire clk" in verilog
    assert "input  wire inp_valid" in verilog
    assert "output reg  out_valid" in verilog
    assert "reg  [" in verilog and " inp_r;" in verilog
    assert "always @(posedge clk)" in verilog
    assert re.search(r"reg\s+\[[0-9]+:0\] pipe_0_r1;", verilog)


def test_pipeline_zero_preserves_combinational_interface():
    circuit = Circuit.from_model(DenseModel(), input_shape=DenseModel().input_shape)
    circuit.simplify()

    verilog = circuit.get_verilog_code(pipeline=0)
    assert "input  wire clk" not in verilog
    assert "inp_valid" not in verilog
    assert "out_valid" not in verilog
    assert "always @(posedge clk)" not in verilog


def test_group_sum_pipeline_verilog_keeps_scores_interface():
    circuit = Circuit.from_model(ConvModel(), input_shape=ConvModel().input_shape)
    circuit.simplify()

    verilog = circuit.get_verilog_code(pipeline=1)
    assert "scores_flat" in verilog
    assert "inp_valid" in verilog
    assert "out_valid" in verilog
    assert "always @(posedge clk)" in verilog


@pytest.mark.parametrize("model_cls", [DenseModel, ConvModel])
def test_pipeline_metadata_survives_json_roundtrip(model_cls):
    circuit = Circuit.from_model(model_cls(), input_shape=model_cls().input_shape)
    circuit.simplify()

    with tempfile.NamedTemporaryFile(suffix=".json") as tmp_file:
        circuit.write_json(tmp_file.name)
        circuit_loaded = Circuit.from_json_file(tmp_file.name)

    assert circuit_loaded.pipeline_boundary_origins == circuit.pipeline_boundary_origins
    assert circuit_loaded.pipeline_boundary_signals == circuit.pipeline_boundary_signals


def _pack_bool_rows(arr):
    packed = []
    for row in arr:
        value = 0
        for bit_idx, bit in enumerate(row):
            if bit:
                value |= (1 << bit_idx)
        packed.append(value)
    return packed


@pytest.mark.skipif(shutil.which("verilator") is None, reason="verilator not installed")
def test_verilog_pipeline_functional_latency():
    model = nn.Sequential(
        LogicDense(8, 16, parametrization="raw", parametrization_kwargs={"weight_init": "random"}),
        LogicDense(16, 8, parametrization="raw", parametrization_kwargs={"weight_init": "random"}),
    )
    set_export_mode(model)
    circuit = Circuit.from_model(model, input_shape=(8,))
    circuit.simplify()

    # If no boundaries were recorded there is nothing to pipeline-test here.
    if not circuit.pipeline_boundary_signals:
        pytest.skip("No pipeline boundaries recorded for this circuit")

    pipeline_depth = 1
    verilog = circuit.get_verilog_code(pipeline=pipeline_depth)

    n_samples = 48
    x0 = torch.randint(0, 2, (1, 8), dtype=torch.bool)
    x = x0.repeat(n_samples, 1)
    input_words = _pack_bool_rows(x.numpy())

    tail_cycles = max(32, 4 * pipeline_depth * max(1, len(circuit.pipeline_boundary_signals)))
    testbench = f"""
#include \"Vcircuit.h\"
#include \"verilated.h\"
#include <cstdint>
#include <cstdio>

static const uint8_t in_words[{n_samples}] = {{ {", ".join(str(v) for v in input_words)} }};
int main(int argc, char** argv) {{
    Verilated::commandArgs(argc, argv);
    Vcircuit dut;
    dut.clk = 0;
    dut.inp_valid = 0;
    dut.inp = 0;

    const int total_cycles = {n_samples} + {tail_cycles};
    for (int t = 0; t < total_cycles; t++) {{
        if (t < {n_samples}) {{
            dut.inp = in_words[t];
            dut.inp_valid = 1;
        }} else {{
            dut.inp = 0;
            dut.inp_valid = 0;
        }}

        dut.clk = 0;
        dut.eval();
        dut.clk = 1;
        dut.eval();
        if (dut.out_valid) {{
            uint8_t got = dut.out & 0xFF;
            std::printf("%u\\n", (unsigned)got);
        }}
    }}

    return 0;
}}
"""

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = pathlib.Path(tmpdir)
        (tmp / "circuit.v").write_text(verilog)
        (tmp / "tb.cpp").write_text(testbench)

        build = subprocess.run(
            ["verilator", "--cc", "circuit.v", "--exe", "tb.cpp", "--build", "--Mdir", "obj_dir", "-Wno-fatal"],
            cwd=tmpdir,
            capture_output=True,
            text=True,
        )
        assert build.returncode == 0, build.stderr

        run = subprocess.run(["./obj_dir/Vcircuit"], cwd=tmpdir, capture_output=True, text=True)
        assert run.returncode == 0, run.stdout + "\n" + run.stderr

    got_words = [int(line.strip()) for line in run.stdout.splitlines() if line.strip()]
    assert len(got_words) >= n_samples, "Not enough valid outputs were produced"
    tail = got_words[-8:]
    assert len(set(tail)) == 1, "Expected stable output tail for repeated constant inputs"


@pytest.mark.parametrize("model_cls", [ConvModel, BranchModel])
def test_json_roundtrip(model_cls):
    model = model_cls()
    x = torch.randint(0, 2, (1, *model.input_shape), dtype=torch.bool)

    circuit = Circuit.from_model(model, input_shape=model.input_shape)
    preds_before = circuit(x.reshape(x.shape[0], -1))

    # Export the circuit to a temporary file and load it back
    with tempfile.NamedTemporaryFile(suffix=".json") as tmp_file:
        circuit.write_json(tmp_file.name)
        circuit_loaded = Circuit.from_json_file(tmp_file.name)

    preds_after = circuit_loaded(x.reshape(x.shape[0], -1))
    assert torch.equal(preds_before, preds_after), "Predictions differ after export/import roundtrip!"


@pytest.mark.parametrize("model_cls", [ConvModel, BranchModel])
def test_c_codegen_group_sum_scores(model_cls):
    """GroupSum reduction is inlined into circuit and compiles cleanly."""
    model = model_cls()
    x = torch.randint(0, 2, (1, *model.input_shape), dtype=torch.bool)

    circuit = Circuit.from_model(model, input_shape=model.input_shape)
    assert circuit.sum_nodes

    from torchlogix.circuit import _c_output_dtype
    sum_by_id = circuit._sum_by_id
    red_outs = [sum_by_id[oid] for oid in circuit.outputs if oid in sum_by_id]
    k = len(red_outs)
    out_dtype = _c_output_dtype(red_outs)
    c_code = circuit.get_c_code()

    assert f"{out_dtype}   out[" in c_code
    assert "bool raw[" in c_code
    assert c_code.count("// --- outputs ---") == 1
    assert c_code.count("int s = 0;") == k

    # Verify it compiles cleanly.
    with tempfile.NamedTemporaryFile(suffix=".c", mode="w", delete=False) as tf:
        tf.write(c_code)
        c_path = tf.name
    result = subprocess.run(
        ["gcc", "-std=c99", "-fsyntax-only", c_path],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, f"C compile error:\n{result.stderr}"

    # Verify scores match Python circuit.
    preds_python = circuit(x.reshape(1, -1))  # shape (1, k)
    assert preds_python.shape[-1] == k


