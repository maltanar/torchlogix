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
LOOKUP_TABLE_CONV_OPSET = 1


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


def _im2col_channels_last(x, kernel_shape, strides, pads):
    """x: (N, *D, C) -> patches: (N, *O, prod(kernel_shape) * C), row-major-spatial/channel-minor."""
    n_spatial = len(kernel_shape)
    pad_arg = [0, 0]
    for d in reversed(range(n_spatial)):
        pad_arg += [pads[d], pads[d + n_spatial]]
    x = torch.nn.functional.pad(x, pad_arg, mode="constant", value=0)
    for d in range(n_spatial):
        x = x.unfold(1 + d, kernel_shape[d], strides[d])
    x = x.movedim(n_spatial + 1, -1)  # move channel dim past the newly appended window dims
    return x.flatten(-(n_spatial + 1))


@torch.library.custom_op("torchlogix::lookup_table_conv", mutates_args=())
def lookup_table_conv(
    x: torch.Tensor,
    indices: torch.Tensor,
    table: torch.Tensor,
    tree_depth: int,
    kernel_shape: list[int],
    strides: list[int],
    pads: list[int],
) -> torch.Tensor:
    """Reference (eager) semantics of the ``LookupTableConv`` op.

    Args:
        x: Channels-last tensor of shape ``(N, *D, C)`` with values in {0, 1}.
        indices: ``(M, P, lut_rank)`` int64 tensor, leaf-level receptive-field connectivity.
        table: ``(M, N_nodes, 2 ** lut_rank)`` tensor of per-tree-node truth tables, packed
            level-major starting from the leaves (see the op's spec for ``N_nodes``).
        tree_depth: Number of tree levels; ``0`` means passthrough (no lookup evaluated).
        kernel_shape, strides, pads: Receptive-field geometry, same convention as ONNX ``Conv``.

    Returns:
        ``(N, *O, M)`` tensor with ``table``'s dtype (or ``(N, *O, M, lut_rank)`` with ``x``'s
        dtype when ``tree_depth == 0``).
    """
    lut_rank = indices.shape[-1]
    patches = _im2col_channels_last(x, kernel_shape, strides, pads)  # (N, *O, prod(kernel_shape)*C)
    gathered = patches[..., indices].to(torch.int64)  # (N, *O, M, P, lut_rank)

    if tree_depth == 0:
        return gathered[..., 0, :]

    place_value = 2 ** torch.arange(lut_rank, device=x.device, dtype=torch.int64)
    level = gathered  # (..., M, num_nodes_this_level, lut_rank)
    row_offset = 0
    for l in range(tree_depth):
        addr = (level * place_value).sum(-1)  # (..., M, num_nodes_this_level)
        num_nodes_this_level = addr.shape[-1]
        rows = table[:, row_offset:row_offset + num_nodes_this_level, :]
        rows = rows.expand(*addr.shape[:-2], *rows.shape)
        out = torch.gather(rows, -1, addr.unsqueeze(-1)).squeeze(-1)
        row_offset += num_nodes_this_level
        if l < tree_depth - 1:
            level = out.reshape(*out.shape[:-1], num_nodes_this_level // lut_rank, lut_rank)
        else:
            return out[..., 0]  # root: exactly one node per kernel


@lookup_table_conv.register_fake
def _lookup_table_conv_meta(x, indices, table, tree_depth, kernel_shape, strides, pads):
    n_spatial = len(kernel_shape)
    out_spatial = [
        (x.shape[1 + d] + pads[d] + pads[d + n_spatial] - kernel_shape[d]) // strides[d] + 1
        for d in range(n_spatial)
    ]
    m = indices.shape[0]
    if tree_depth == 0:
        shape = (x.shape[0], *out_spatial, m, indices.shape[-1])
        dtype = x.dtype
    else:
        shape = (x.shape[0], *out_spatial, m)
        dtype = table.dtype
    return torch.empty(shape, dtype=dtype, device=x.device)


def _register_conv_schema():
    """Register a minimal ONNX schema so the node can be built and checked."""
    import onnx

    global _CONV_SCHEMA_REGISTERED
    if _CONV_SCHEMA_REGISTERED:
        return
    op_schema = onnx.defs.OpSchema
    x_types = ["tensor(bool)", "tensor(uint8)", "tensor(uint16)", "tensor(uint32)"]
    table_types = ["tensor(bool)", "tensor(uint8)"]
    schema = op_schema(
        "LookupTableConv",
        LOOKUP_TABLE_DOMAIN,
        LOOKUP_TABLE_CONV_OPSET,
        inputs=[
            op_schema.FormalParameter("X", "TX"),
            op_schema.FormalParameter("indices", "TI"),
            op_schema.FormalParameter("table", "TT"),
        ],
        outputs=[op_schema.FormalParameter("Y", "TT")],
        attributes=[
            op_schema.Attribute("tree_depth", op_schema.AttrType.INT, "", required=False),
            op_schema.Attribute("kernel_shape", op_schema.AttrType.INTS, "", required=True),
            op_schema.Attribute("strides", op_schema.AttrType.INTS, "", required=False),
            op_schema.Attribute("pads", op_schema.AttrType.INTS, "", required=False),
        ],
        type_constraints=[
            ("TX", x_types, ""),
            ("TI", ["tensor(int32)", "tensor(int64)"], ""),
            ("TT", table_types, ""),
        ],
    )
    onnx.defs.register_schema(schema)
    _CONV_SCHEMA_REGISTERED = True


_CONV_SCHEMA_REGISTERED = False


def _lookup_table_conv_onnx(x, indices, table, tree_depth, kernel_shape, strides, pads):
    import onnxscript

    _register_conv_schema()
    lnn_opset = onnxscript.values.Opset(
        domain=LOOKUP_TABLE_DOMAIN, version=LOOKUP_TABLE_CONV_OPSET
    )
    return lnn_opset.LookupTableConv(
        x, indices, table,
        tree_depth=tree_depth, kernel_shape=kernel_shape, strides=strides, pads=pads,
    )


def custom_translation_table():
    """Translation table mapping torchlogix ops to their ONNX counterparts."""
    return {
        torch.ops.torchlogix.lookup_table.default: _lookup_table_onnx,
        torch.ops.torchlogix.lookup_table_conv.default: _lookup_table_conv_onnx,
    }


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
