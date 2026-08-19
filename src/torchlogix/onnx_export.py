"""ONNX export of logic layers as ``qonnx.custom_op.lnn::LookupTable`` nodes.

The ``LookupTable`` op evaluates, for every neuron ``m``, a truth table
``table[m]`` addressed by a fan-in gathered from the last axis of ``X``::

    addr[..., m] = sum_k X[..., indices[m, k]] * 2 ** (input_bits * k)
    Y[..., m]    = table[m, addr[..., m]]

Layers hook into this by calling :func:`lookup_table` while
``torch.onnx.is_in_onnx_export()`` is active; the translation table returned by
:func:`custom_translation_table` maps that op onto the custom ONNX node.
"""

import torch

LOOKUP_TABLE_DOMAIN = "qonnx.custom_op.lnn"
LOOKUP_TABLE_OPSET = 3


@torch.library.custom_op("torchlogix::lookup_table", mutates_args=())
def lookup_table(
    x: torch.Tensor,
    indices: torch.Tensor,
    table: torch.Tensor,
    input_bits: int,
) -> torch.Tensor:
    """Reference (eager) semantics of the ``LookupTable`` op.

    Args:
        x: Unsigned/bool tensor of shape ``(..., C_in)`` with values in
            ``[0, 2 ** input_bits)``.
        indices: ``(M, K)`` int64 tensor, ``indices[m, k]`` selects the input
            feeding slot ``k`` (slot 0 is the least significant) of neuron ``m``.
        table: ``(M, 2 ** (K * input_bits))`` truth table. Its dtype is the
            output dtype.
        input_bits: Number of bits per element of ``x``.

    Returns:
        Tensor of shape ``(..., M)`` with the dtype of ``table``.
    """
    fan_in = x[..., indices].to(torch.int64)  # (..., M, K)
    k = indices.shape[1]
    place_value = 2 ** (input_bits * torch.arange(k, device=x.device, dtype=torch.int64))
    addr = (fan_in * place_value).sum(-1)  # (..., M)
    row_offset = torch.arange(indices.shape[0], device=x.device) * table.shape[1]
    return table.reshape(-1)[addr + row_offset]


@lookup_table.register_fake
def _lookup_table_meta(x, indices, table, input_bits):
    return torch.empty(
        (*x.shape[:-1], indices.shape[0]), dtype=table.dtype, device=x.device
    )


def _register_schema():
    """Register a minimal ONNX schema so the node can be built and checked."""
    import onnx

    global _SCHEMA_REGISTERED
    if _SCHEMA_REGISTERED:
        return
    op_schema = onnx.defs.OpSchema
    x_types = ["tensor(bool)", "tensor(uint8)", "tensor(uint16)", "tensor(uint32)"]
    table_types = x_types + [
        "tensor(int8)", "tensor(int16)", "tensor(int32)",
    ]
    schema = op_schema(
        "LookupTable",
        LOOKUP_TABLE_DOMAIN,
        LOOKUP_TABLE_OPSET,
        inputs=[
            op_schema.FormalParameter("X", "TX"),
            op_schema.FormalParameter("indices", "TI"),
            op_schema.FormalParameter("table", "TT"),
        ],
        outputs=[op_schema.FormalParameter("Y", "TT")],
        attributes=[
            op_schema.Attribute("input_bits", op_schema.AttrType.INT, "", required=False),
            op_schema.Attribute("out_bits", op_schema.AttrType.INT, "", required=False),
        ],
        type_constraints=[
            ("TX", x_types, ""),
            ("TI", ["tensor(int32)", "tensor(int64)"], ""),
            ("TT", table_types, ""),
        ],
    )
    onnx.defs.register_schema(schema)
    _SCHEMA_REGISTERED = True


_SCHEMA_REGISTERED = False


def _lookup_table_onnx(x, indices, table, input_bits):
    import onnxscript

    _register_schema()
    lnn_opset = onnxscript.values.Opset(
        domain=LOOKUP_TABLE_DOMAIN, version=LOOKUP_TABLE_OPSET
    )
    return lnn_opset.LookupTable(
        x, indices, table, input_bits=input_bits, out_bits=1
    )


def custom_translation_table():
    """Translation table mapping torchlogix ops to their ONNX counterparts."""
    return {torch.ops.torchlogix.lookup_table.default: _lookup_table_onnx}


def export(model, args, f, **kwargs):
    """Export a torchlogix model to ONNX with ``LookupTable`` nodes.

    Puts ``model`` into export mode (see :func:`torchlogix.utils.set_export_mode`)
    and forwards everything to :func:`torch.onnx.export`.

    Args:
        model: Module to export.
        args: Example inputs. Elements feeding logic layers must be binary.
        f: Output path or file-like object.
        **kwargs: Extra arguments for :func:`torch.onnx.export`.
    """
    from .utils import set_export_mode

    set_export_mode(model, enabled=True)
    kwargs.setdefault("dynamo", True)
    kwargs.setdefault("external_data", False)
    table = custom_translation_table()
    table.update(kwargs.pop("custom_translation_table", None) or {})
    return torch.onnx.export(
        model, args, f, custom_translation_table=table, **kwargs
    )
