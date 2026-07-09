#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""IR construction helpers for block ops.

These functions accept IR ``Expr`` objects and return ``Call`` expressions.
Parse handlers registered via ``@op_impl`` dispatch through ``_OP_REGISTRY``
in ``call_parser.parse_op_call``.
"""

from __future__ import annotations

import ast
import os
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Optional

from pypto_pro.ir._utils import _normalize_expr, _to_make_tuple
from pypto.pypto_impl import ir as _ir_core
from pypto.pypto_impl.ir import ConstInt, DataType, Expr, MemorySpace, Span, TensorLayout, TilePad

from pypto.ir import (
    AccToVecMode,
    AtomicType,
    STPhase,
    AccPhase,
    QuantMode,
    ReluPreMode,
    RoundMode,
)
from ._op_registry import OpSpec, op_impl, register_table


def _span() -> Span:
    return Span.unknown()


_BLOCK_OP_NAMESPACE = "block"


def block_ir_op(op_name: str) -> str:
    """Return the IR name for the explicit-output block DSL ops."""
    return f"{_BLOCK_OP_NAMESPACE}.{op_name}"


def _compute_absolute_offsets(
    tile_offsets: _ir_core.MakeTuple,
    tile_shape: list,
    tile_dims: list[int],
    span: Span,
) -> _ir_core.MakeTuple:
    """Convert tile-relative offsets to absolute tensor offsets."""
    offsets = []
    for i, tile_offset in enumerate(tile_offsets.elements):
        if i in tile_dims:
            tile_idx = tile_dims.index(i)
            shape = tile_shape[tile_idx]
            if isinstance(tile_offset, ConstInt) and isinstance(shape, ConstInt):
                offset = ConstInt(tile_offset.value * shape.value, DataType.INT64, span)
            else:
                offset = _ir_core.Mul(tile_offset, shape, DataType.INT64, span)
            offsets.append(offset)
        else:
            offsets.append(tile_offset)
    return _ir_core.MakeTuple(offsets, span)


def _const_int_attr(value: int | Expr, name: str) -> int:
    if isinstance(value, int):
        return value
    const_value = getattr(value, "value", None)
    if isinstance(const_value, int):
        return const_value
    raise TypeError(f"block op requires constant integer {name}")


def _ir_binary_cast(
    op_name: str,
    out: Expr,
    lhs: Expr,
    rhs: Expr,
    target_type: int | DataType,
    *,
    span: Span | None = None,
    mode: RoundMode = RoundMode.CAST_ROUND,
) -> Expr:
    return _ir_core.create_op_call(
        block_ir_op(op_name),
        [out, lhs, rhs],
        {"target_type": target_type, "mode": mode},
        span or _span(),
    )


def _ir_load(
    out: Expr,
    tensor: Expr,
    offsets: Sequence[int | Expr] | _ir_core.MakeTuple,
    *,
    span: Span | None = None,
    is_transpose: bool = False,
    tile_dims: list[int] | None = None,
) -> Expr:
    kwargs = {"is_transpose": is_transpose} if is_transpose else {}
    if tile_dims is not None:
        kwargs["tile_dims"] = tile_dims
    return _ir_core.create_op_call(
        block_ir_op("load"), [out, tensor, _to_make_tuple(offsets, span)], kwargs, span or _span()
    )


def _ir_load_tile(
    out: Expr,
    tensor: Expr,
    tile_offsets: Sequence[int | Expr] | _ir_core.MakeTuple,
    *,
    span: Span | None = None,
    is_transpose: bool = False,
    tile_dims: list[int] | None = None,
) -> Expr:
    actual_span = span or _span()
    offsets_tuple = _to_make_tuple(tile_offsets, actual_span)
    tensor_ndim = len(tensor.type.shape)
    tile_shape = list(out.type.shape)
    tile_ndim = len(tile_shape)
    if tile_dims is None:
        tile_dims = list(range(tensor_ndim - tile_ndim, tensor_ndim))
    abs_offsets = _compute_absolute_offsets(offsets_tuple, tile_shape, tile_dims, actual_span)
    kwargs = {"is_transpose": is_transpose} if is_transpose else {}
    default_tile_dims = list(range(tensor_ndim - tile_ndim, tensor_ndim))
    if tile_dims != default_tile_dims:
        kwargs["tile_dims"] = tile_dims
    return _ir_core.create_op_call(block_ir_op("load"), [out, tensor, abs_offsets], kwargs, actual_span)


def _ir_store(
    out: Expr,
    tile: Expr,
    offsets: Sequence[int | Expr] | _ir_core.MakeTuple,
    *,
    span: Span | None = None,
    relu_pre_mode: ReluPreMode | None = None,
    pre_quant_scalar: int | Expr | None = None,
    fp_tile: Expr | None = None,
    tile_dims: list[int] | None = None,
    atomic: AtomicType = AtomicType.AtomicNone,
    phase: STPhase | None = None,
) -> Expr:
    actual_span = span or _span()
    offsets_tuple = _to_make_tuple(offsets, actual_span)
    if fp_tile is not None and relu_pre_mode is not None:
        raise ValueError("fp_tile cannot be used together with relu_pre_mode")
    if fp_tile is not None and pre_quant_scalar is not None:
        raise ValueError("fp_tile cannot be used together with pre_quant_scalar")
    if fp_tile is not None and phase is not None:
        raise ValueError("fp_tile cannot be combined with phase")
    if fp_tile is not None:
        return _ir_core.create_op_call(block_ir_op("store_fp"), [out, tile, fp_tile, offsets_tuple], {}, actual_span)
    kwargs: dict[str, Any] = {}
    if relu_pre_mode is not None:
        kwargs["relu_pre_mode"] = relu_pre_mode
    if tile_dims is not None:
        kwargs["tile_dims"] = tile_dims
    if atomic != AtomicType.AtomicNone:
        kwargs["atomic"] = atomic
    if phase is not None:
        kwargs["phase"] = phase
    operands: list[Expr] = [out, tile, offsets_tuple]
    if pre_quant_scalar is not None:
        pre_quant_operand = (
            ConstInt(pre_quant_scalar, DataType.UINT64, actual_span)
            if isinstance(pre_quant_scalar, int)
            else pre_quant_scalar
        )
        operands.append(pre_quant_operand)
    return _ir_core.create_op_call(block_ir_op("store"), operands, kwargs, actual_span)


def _ir_store_fp(
    out: Expr,
    tile: Expr,
    fp_tile: Expr,
    offsets: Sequence[int | Expr] | _ir_core.MakeTuple,
    *,
    span: Span | None = None,
) -> Expr:
    actual_span = span or _span()
    return _ir_core.create_op_call(
        block_ir_op("store_fp"),
        [out, tile, fp_tile, _to_make_tuple(offsets, actual_span)],
        {},
        actual_span,
    )


def _ir_store_tile(
    out: Expr,
    tile: Expr,
    tile_offsets: Sequence[int | Expr] | _ir_core.MakeTuple,
    *,
    span: Span | None = None,
    tile_dims: list[int] | None = None,
) -> Expr:
    actual_span = span or _span()
    offsets_tuple = _to_make_tuple(tile_offsets, actual_span)
    tensor_ndim = len(out.type.shape)
    tile_shape = list(tile.type.shape)
    tile_ndim = len(tile_shape)
    if tile_dims is None:
        tile_dims = list(range(tensor_ndim - tile_ndim, tensor_ndim))
    abs_offsets = _compute_absolute_offsets(offsets_tuple, tile_shape, tile_dims, actual_span)
    kwargs = {}
    default_tile_dims = list(range(tensor_ndim - tile_ndim, tensor_ndim))
    if tile_dims != default_tile_dims:
        kwargs["tile_dims"] = tile_dims
    return _ir_core.create_op_call(block_ir_op("store"), [out, tile, abs_offsets], kwargs, actual_span)


def _ir_move(
    out: Expr,
    src: Expr,
    offset: Expr | Sequence[Any] | None = None,
    *,
    span: Span | None = None,
    acc_to_vec_mode: AccToVecMode | None = None,
    relu_pre_mode: ReluPreMode | None = None,
    pre_quant_scalar: int | Expr | None = None,
    fp_tile: Expr | None = None,
) -> Expr:
    actual_span = span or _span()
    if not isinstance(out.type, _ir_core.TileType):
        raise ValueError(f"move: dst must be a Tile, got {type(out.type).__name__}")
    if not isinstance(src.type, _ir_core.TileType):
        raise ValueError(f"move: src must be a Tile, got {type(src.type).__name__}")
    _dst_mem = getattr(getattr(out.type, "memref", None), "memory_space_", None)
    _src_mem = getattr(getattr(src.type, "memref", None), "memory_space_", None)
    _supported_move_paths = {
        (MemorySpace.Mat, MemorySpace.Left),
        (MemorySpace.Mat, MemorySpace.Right),
        (MemorySpace.Acc, MemorySpace.Vec),
        (MemorySpace.Vec, MemorySpace.Vec),
        (MemorySpace.Mat, MemorySpace.Scaling),
    }
    if (_src_mem is not None and _dst_mem is not None
            and (_src_mem, _dst_mem) not in _supported_move_paths):
        raise ValueError(
            f"move: unsupported data path src({_src_mem.name})->dst({_dst_mem.name}), "
            f"supported paths: Mat->Left, Mat->Right, Acc->Vec, Vec->Vec"
        )
    if fp_tile is not None and pre_quant_scalar is not None:
        raise ValueError("fp_tile cannot be used together with pre_quant_scalar")
    if fp_tile is not None and acc_to_vec_mode in {AccToVecMode.DualModeSplitM, AccToVecMode.DualModeSplitN}:
        raise ValueError("fp_tile only supports single-mode acc_to_vec_mode")
    kwargs: dict[str, Any] = {}
    if acc_to_vec_mode is not None:
        kwargs["acc_to_vec_mode"] = acc_to_vec_mode
    if relu_pre_mode is not None:
        kwargs["relu_pre_mode"] = relu_pre_mode
    if fp_tile is not None:
        return _ir_core.create_op_call(block_ir_op("move_fp"), [out, src, fp_tile], kwargs, actual_span)
    # Positional operands: [out, src, (offset tuple)?, (pre_quant_scalar)?].
    # offset is a TupleType, pre_quant_scalar is a ScalarType — the backend disambiguates by type,
    # so pre_quant_scalar can be a runtime scalar value, not just a compile-time constant.
    args = [out, src]
    if offset is not None:
        args.append(_to_make_tuple(offset, actual_span))
    if pre_quant_scalar is not None:
        pre_quant_operand = (
            ConstInt(pre_quant_scalar, DataType.UINT64, actual_span)
            if isinstance(pre_quant_scalar, int)
            else pre_quant_scalar
        )
        args.append(pre_quant_operand)
    return _ir_core.create_op_call(block_ir_op("move"), args, kwargs, actual_span)


def _normalize_2d_sequence(value: Any, parameter: str, span: Span) -> tuple[Expr, Expr]:
    sequence = _to_make_tuple(value, span)
    if len(sequence.elements) != 2:
        raise ValueError(f"{parameter} must contain exactly 2 elements")
    return sequence.elements[0], sequence.elements[1]


def _ir_insert(
    out: Expr,
    src: Expr,
    offset: Sequence[int | Expr] | _ir_core.MakeTuple,
    *,
    span: Span | None = None,
) -> Expr:
    actual_span = span or _span()
    row, col = _normalize_2d_sequence(offset, "offset", actual_span)
    return _ir_core.create_op_call(block_ir_op("insert"), [out, src, row, col], {}, actual_span)


def _ir_sel(out: Expr, mask: Expr, lhs: Expr, rhs: Expr, tmp: Expr, *, span: Span | None = None) -> Expr:
    return _ir_core.create_op_call(block_ir_op("sel"), [out, mask, lhs, rhs, tmp], {}, span or _span())


def _ir_sels(out: Expr, mask: Expr, src: Expr, tmp: Expr, scalar: Expr, *, span: Span | None = None) -> Expr:
    actual_span = span or _span()
    scalar_expr = scalar if isinstance(scalar, Expr) else _normalize_expr(scalar, actual_span)
    return _ir_core.create_op_call(block_ir_op("sels"), [out, mask, src, tmp, scalar_expr], {}, actual_span)



def _ir_set_validshape(
    tile: Expr,
    shape: Sequence[int | Expr] | _ir_core.MakeTuple,
    *,
    span: Span | None = None,
) -> Expr:
    actual_span = span or _span()
    shape_0, shape_1 = _normalize_2d_sequence(shape, "shape", actual_span)
    return _ir_core.create_op_call(block_ir_op("set_validshape"), [tile, shape_0, shape_1], {}, actual_span)


def _ir_set_stride(
    tensor: Expr,
    stride: Sequence[int | Expr] | _ir_core.MakeTuple,
    *,
    span: Span | None = None,
) -> Expr:
    actual_span = span or _span()
    stride_0, stride_1 = _normalize_2d_sequence(stride, "stride", actual_span)
    return _ir_core.create_op_call(block_ir_op("set_stride"), [tensor, stride_0, stride_1], {}, actual_span)


_SCALAR_UNSUPPORTED_DTYPES: tuple[DataType, ...] = (
    DataType.FP4, DataType.FP8E4M3FN, DataType.FP8E5M2,
    DataType.INT4, DataType.UINT4, DataType.HF4, DataType.HF8,
)


def _check_scalar_supported_dtype(op_name: str, container: Expr) -> None:
    dtype = container.type.dtype
    if any(dtype == d for d in _SCALAR_UNSUPPORTED_DTYPES):
        raise TypeError(
            f"{op_name} does not support container dtype {dtype}; "
            "low-precision types (FP4/FP8/INT4/UINT4/HF4/HF8) are storage-only "
            "and cannot be used in scalar expressions"
        )


def _ir_getval(container: Expr, offset: int | Expr, *, span: Span | None = None) -> Expr:
    actual_span = span or _span()
    _check_scalar_supported_dtype("getval", container)
    offset_expr = offset if isinstance(offset, Expr) else _normalize_expr(offset, actual_span, int_dtype=DataType.INDEX)
    return _ir_core.create_op_call("getval", [container, offset_expr], {}, actual_span)


def _ir_setval(container: Expr, offset: int | Expr, value: int | float | Expr, *, span: Span | None = None) -> Expr:
    actual_span = span or _span()
    _check_scalar_supported_dtype("setval", container)
    offset_expr = offset if isinstance(offset, Expr) else _normalize_expr(offset, actual_span, int_dtype=DataType.INDEX)
    if not isinstance(value, Expr):
        container_dtype = container.type.dtype
        value_expr = _normalize_expr(value, actual_span, int_dtype=container_dtype, float_dtype=container_dtype)
    else:
        value_expr = value
    return _ir_core.create_op_call("setval", [container, offset_expr, value_expr], {}, actual_span)


def _ir_transpose(out: Expr, src: Expr, axis1: int | Expr, axis2: int | Expr, *, span: Span | None = None) -> Expr:
    return _ir_core.create_op_call(
        block_ir_op("transpose"),
        [out, src],
        {"axis1": _const_int_attr(axis1, "axis1"), "axis2": _const_int_attr(axis2, "axis2")},
        span or _span(),
    )


def _ir_cast(
    out: Expr,
    src: Expr,
    target_type: int | DataType,
    *,
    span: Span | None = None,
    mode: RoundMode = RoundMode.CAST_ROUND,
) -> Expr:
    return _ir_core.create_op_call(
        block_ir_op("cast"),
        [out, src],
        {"target_type": target_type, "mode": mode},
        span or _span(),
    )


def _ir_add_relu_cast(
    out: Expr,
    lhs: Expr,
    rhs: Expr,
    target_type: int | DataType,
    *,
    span: Span | None = None,
    mode: RoundMode = RoundMode.CAST_ROUND,
) -> Expr:
    return _ir_binary_cast("add_relu_cast", out, lhs, rhs, target_type, span=span, mode=mode)


def _ir_sub_relu_cast(
    out: Expr,
    lhs: Expr,
    rhs: Expr,
    target_type: int | DataType,
    *,
    span: Span | None = None,
    mode: RoundMode = RoundMode.CAST_ROUND,
) -> Expr:
    return _ir_binary_cast("sub_relu_cast", out, lhs, rhs, target_type, span=span, mode=mode)


def _ir_mul_cast(
    out: Expr,
    lhs: Expr,
    rhs: Expr,
    target_type: int | DataType,
    *,
    span: Span | None = None,
    mode: RoundMode = RoundMode.CAST_ROUND,
) -> Expr:
    return _ir_binary_cast("mul_cast", out, lhs, rhs, target_type, span=span, mode=mode)


def _ir_cmp(out: Expr, lhs: Expr, rhs: Expr, *, span: Span | None = None, cmp_type: int | Expr = 0) -> Expr:
    return _ir_core.create_op_call(
        block_ir_op("cmp"),
        [out, lhs, rhs],
        {"cmp_type": _const_int_attr(cmp_type, "cmp_type")},
        span or _span(),
    )


def _ir_cmps(out: Expr, lhs: Expr, rhs: Expr, *, span: Span | None = None, cmp_type: int | Expr = 0) -> Expr:
    return _ir_core.create_op_call(
        block_ir_op("cmps"),
        [out, lhs, rhs],
        {"cmp_type": _const_int_attr(cmp_type, "cmp_type")},
        span or _span(),
    )


def _ir_set_mask_count(*, span: Span | None = None) -> Expr:
    return _ir_core.create_op_call("system.set_mask_count", [], {}, span or _span())


def _ir_set_mask_norm(*, span: Span | None = None) -> Expr:
    return _ir_core.create_op_call("system.set_mask_norm", [], {}, span or _span())


def _ir_set_vec_mask(mask_high: Expr, mask_low: Expr, *, span: Span | None = None) -> Expr:
    return _ir_core.create_op_call("system.set_vec_mask", [mask_high, mask_low], {}, span or _span())


def _ir_reset_mask(*, span: Span | None = None) -> Expr:
    return _ir_core.create_op_call("system.reset_mask", [], {}, span or _span())


def _ir_quant(
    out: Expr,
    src: Expr,
    scale: Expr,
    *,
    span: Span | None = None,
    mode: QuantMode = QuantMode.SYM,
    offset: Expr | None = None,
) -> Expr:
    ins = [out, src, scale]
    if mode == QuantMode.ASYM:
        if offset is None:
            raise ValueError("quant in 'asym' mode requires an offset argument")
        ins.append(offset)
    return _ir_core.create_op_call(block_ir_op("quant"), ins, {"mode": mode}, span or _span())


def _ir_dequant(out: Expr, src: Expr, scale: Expr, offset: Expr, *, span: Span | None = None) -> Expr:
    return _ir_core.create_op_call(block_ir_op("dequant"), [out, src, scale, offset], {}, span or _span())


def _ir_ssbuf_store(*args: Expr, span: Span | None = None) -> Expr:
    return _ir_core.create_op_call(block_ir_op("ssbuf_store"), list(args), {}, span or _span())


def _ir_ssbuf_load(*args: Expr, span: Span | None = None) -> Expr:
    return _ir_core.create_op_call(block_ir_op("ssbuf_load"), list(args), {}, span or _span())


# ---------------------------------------------------------------------------
# TileType descriptor and make_tile_expr
# ---------------------------------------------------------------------------

def _get_current_arch() -> str:
    arch = os.environ.get("PYPTOPRO_JIT_ARCH") or os.environ.get("PYPTOPRO_NPU_ARCH") or "a3"
    arch = arch.strip().lower()
    if arch.startswith("dav-c220") or arch.startswith("dav-2201"):
        return "a3"
    if arch.startswith("dav-c310") or arch.startswith("dav-3510"):
        return "a5"
    if arch in ("a2", "a3", "a5"):
        return arch
    return "a3"


_LAYOUT_TO_BS: dict[TensorLayout, tuple[int, int]] = {
    TensorLayout.ND: (1, 0),
    TensorLayout.DN: (2, 0),
    TensorLayout.NZ: (2, 1),
    TensorLayout.ZN: (1, 2),
    TensorLayout.NN: (2, 2),
    TensorLayout.ZZ: (1, 1),
}

_DEFAULT_LAYOUTS_A3: dict[MemorySpace, TensorLayout] = {
    MemorySpace.Mat: TensorLayout.NZ,
    MemorySpace.Left: TensorLayout.ZZ,
    MemorySpace.Right: TensorLayout.ZN,
    MemorySpace.Scaling: TensorLayout.ND,
    MemorySpace.Acc: TensorLayout.NZ,
}

_DEFAULT_LAYOUTS_A5: dict[MemorySpace, TensorLayout] = {
    MemorySpace.Mat: TensorLayout.NZ,
    MemorySpace.Left: TensorLayout.NZ,
    MemorySpace.Right: TensorLayout.ZN,
    MemorySpace.Scaling: TensorLayout.ND,
    MemorySpace.Acc: TensorLayout.NZ,
}

mem_id: int = 0

_PAD_VALUES = {
    TilePad.null: 0,
    TilePad.zero: 1,
    TilePad.max: 2,
    TilePad.min: 3,
}


def _normalize_tile_pad(pad: "int | TilePad | None") -> "int | None":
    if pad is None:
        return None
    if pad in _PAD_VALUES:
        return _PAD_VALUES[pad]
    if isinstance(pad, int):
        if pad not in _PAD_VALUES.values():
            raise ValueError("TileType.pad must be one of TilePad.null/zero/max/min")
        return pad
    raise TypeError("TileType.pad must be a TilePad or integer 0/1/2/3")


def _apply_default_layout(tt: "TileType") -> None:
    arch = _get_current_arch()
    layout_dict = _DEFAULT_LAYOUTS_A5 if arch == "a5" else _DEFAULT_LAYOUTS_A3
    default_layout = layout_dict.get(tt.target_memory)
    if default_layout is None:
        return

    if tt.layout is None:
        tt.layout = default_layout

    allowed_layouts = {default_layout}
    if tt.target_memory == MemorySpace.Left:
        allowed_layouts = {_DEFAULT_LAYOUTS_A3[MemorySpace.Left], _DEFAULT_LAYOUTS_A5[MemorySpace.Left]}
    elif tt.target_memory == MemorySpace.Mat:
        allowed_layouts.add(TensorLayout.ZN)
        if tt.dtype in (DataType.UINT64, DataType.INT64):
            allowed_layouts.add(TensorLayout.ND)

    if tt.layout not in allowed_layouts:
        space_name = tt.target_memory.name
        allowed_text = ", ".join(f.name for f in sorted(allowed_layouts, key=lambda x: x.name))
        raise ValueError(
            f"{space_name} tiles require layout in {{{allowed_text}}}, "
            f"got {tt.layout.name}. "
            f"Default for '{arch}' is {default_layout.name}."
        )

    if tt.target_memory == MemorySpace.Acc and tt.fractal is None:
        if tt.dtype in (DataType.FP32, DataType.INT32):
            tt.fractal = 1024


@dataclass
class TileType:
    """Tile type descriptor containing shape, dtype, and TileView parameters."""

    shape: "Sequence[int] | _ir_core.MakeTuple"
    dtype: DataType
    target_memory: MemorySpace = MemorySpace.Vec
    valid_shape: Optional[Sequence[int]] = None
    layout: Optional[TensorLayout] = None
    fractal: Optional[int] = None
    pad: Optional[int] = None
    compact: Optional[int] = None

    def __post_init__(self):
        self.pad = _normalize_tile_pad(self.pad)
        _apply_default_layout(self)


def make_tile(
    shape: "Sequence[int] | _ir_core.MakeTuple",
    dtype: DataType,
    target_memory: MemorySpace = MemorySpace.Vec,
    addr: "int | Expr | None" = None,
    size: "int | None" = None,
    valid_shape: "Sequence[int] | _ir_core.MakeTuple | None" = None,
    layout: "TensorLayout | None" = None,
    fractal: "int | None" = None,
    pad: "int | None" = None,
    compact: "int | None" = None,
    span: "Span | None" = None,
) -> Expr:
    """Create the block.make_tile allocation expression used by block buffers."""
    actual_span = span or _span()
    shape_tuple = _to_make_tuple(shape, actual_span)
    valid_shape_tuple = (
        _to_make_tuple(valid_shape, actual_span)
        if valid_shape is not None
        else _ir_core.MakeTuple([], actual_span)
    )
    args = [shape_tuple, valid_shape_tuple]
    blayout: "int | None" = None
    slayout: "int | None" = None
    if layout is not None:
        b, s = _LAYOUT_TO_BS[layout]
        blayout = b
        slayout = s
    kwargs: dict[str, Any] = {
        "dtype": dtype,
        "target_memory": target_memory,
        "blayout": blayout,
        "slayout": slayout,
        "fractal": fractal,
        "pad": pad,
        "compact": compact,
    }
    kwargs = {k: v for k, v in kwargs.items() if v is not None}
    if addr is not None:
        if size is None:
            raise ValueError("When specifying addr for make_tile, size must also be provided.")
        global mem_id
        mem_id += 1
        kwargs["memref_addr"] = addr
        kwargs["memref_size"] = size
        kwargs["memref_id"] = mem_id
    return _ir_core.create_op_call(block_ir_op("make_tile"), args, kwargs, actual_span)


# ---------------------------------------------------------------------------
# Parse handlers — registered into _OP_REGISTRY at import time
# ---------------------------------------------------------------------------

@op_impl("TileType")
def _parse_tile_type_call(self, call: ast.Call):
    kwargs = {}
    for kw in call.keywords:
        # TileType shape / valid_shape must be compile-time constants. A runtime
        # scalar var would leak its name into the generated C++ tile declaration
        # (which is hoisted to the function prologue, above where the var is even
        # defined). Reject it here with a clear error; use pl.set_validshape() for
        # a runtime valid shape.
        if kw.arg in ("shape", "valid_shape") and isinstance(kw.value, ast.List):
            kwargs[kw.arg] = [self.resolve_static_int(elt) for elt in kw.value.elts]
        else:
            kwargs[kw.arg] = self.resolve_single_kwarg(kw.arg, kw.value)
    return TileType(**kwargs)


@op_impl("make_tile")
def _parse_make_tile(self, call: ast.Call) -> Expr:
    span = self.span_tracker.get_span(call)
    args = [self.parse_expression(arg) for arg in call.args]
    kwargs = self.parse_op_kwargs(call)

    if len(args) >= 1 and isinstance(args[0], TileType):
        tile_type = args[0]
        kwargs.setdefault("shape", tile_type.shape)
        kwargs.setdefault("dtype", tile_type.dtype)
        kwargs.setdefault("target_memory", tile_type.target_memory)
        if tile_type.valid_shape is not None:
            kwargs.setdefault("valid_shape", tile_type.valid_shape)
        if tile_type.layout is not None:
            kwargs.setdefault("layout", tile_type.layout)
        if tile_type.fractal is not None:
            kwargs.setdefault("fractal", tile_type.fractal)
        if tile_type.pad is not None:
            kwargs.setdefault("pad", tile_type.pad)
        if tile_type.compact is not None:
            kwargs.setdefault("compact", tile_type.compact)
        args = args[1:]

    return make_tile(*args, **kwargs, span=span)


# tile_dims selects tensor axes at compile time (it drives the tensor-view strides in codegen),
# so it must be a compile-time constant int list. Resolve it via the strict validator, which also
# folds a constant list threaded through an implicit-helper parameter (e.g. tile_dims=tile_dims).
def _resolve_tile_dims_kwarg(self, call: ast.Call, kwargs: dict) -> None:
    tile_dims = self.resolve_const_int_list_kwarg(call, "tile_dims")
    if tile_dims is not None:
        kwargs["tile_dims"] = tile_dims


# ---------------------------------------------------------------------------
# Declarative op registration
# ---------------------------------------------------------------------------

register_table({
    # args + kwargs -> builder
    "store_fp": OpSpec(builder=_ir_store_fp),
    "move": OpSpec(builder=_ir_move),
    "insert": OpSpec(builder=_ir_insert),
    "sel": OpSpec(builder=_ir_sel),
    "sels": OpSpec(builder=_ir_sels),
    "getval": OpSpec(builder=_ir_getval),
    "setval": OpSpec(builder=_ir_setval),
    "transpose": OpSpec(builder=_ir_transpose),
    "cast": OpSpec(builder=_ir_cast),
    "add_relu_cast": OpSpec(builder=_ir_add_relu_cast),
    "sub_relu_cast": OpSpec(builder=_ir_sub_relu_cast),
    "mul_cast": OpSpec(builder=_ir_mul_cast),
    "cmp": OpSpec(builder=_ir_cmp),
    "cmps": OpSpec(builder=_ir_cmps),
    "set_vec_mask": OpSpec(builder=_ir_set_vec_mask),
    "quant": OpSpec(builder=_ir_quant),
    "dequant": OpSpec(builder=_ir_dequant),
    "ssbuf_store": OpSpec(builder=_ir_ssbuf_store),
    "ssbuf_load": OpSpec(builder=_ir_ssbuf_load),
    # args + kwargs + tile_dims hook
    "load": OpSpec(builder=_ir_load, pre_hooks=[_resolve_tile_dims_kwarg]),
    "load_tile": OpSpec(builder=_ir_load_tile, pre_hooks=[_resolve_tile_dims_kwarg]),
    "store": OpSpec(builder=_ir_store, pre_hooks=[_resolve_tile_dims_kwarg]),
    "store_tile": OpSpec(builder=_ir_store_tile, pre_hooks=[_resolve_tile_dims_kwarg]),
    "set_stride": OpSpec(builder=_ir_set_stride),
    # kwargs only
    "set_mask_count": OpSpec(builder=_ir_set_mask_count, parse_args=False),
    "set_mask_norm": OpSpec(builder=_ir_set_mask_norm, parse_args=False),
    "reset_mask": OpSpec(builder=_ir_reset_mask, parse_args=False),
})


@op_impl("set_validshape")
def _parse_set_validshape(self, call: ast.Call) -> Expr:
    span = self.span_tracker.get_span(call)
    args = [self.parse_expression(arg) for arg in call.args]
    kwargs = self.parse_op_kwargs(call)

    if args and self.is_tile_group(args[0]):
        group_var = args[0]
        shape = args[1] if len(args) > 1 else None

        meta = self.tile_group_meta.get(id(group_var), (1, None))
        n_tiles = meta[0]
        tiles = self.lower_attr_access(group_var, "tiles", span)

        for i in range(n_tiles):
            tile_ir = _ir_core.GetItemExpr(tiles, ConstInt(i, DataType.INDEX, span), span)
            vs_call = _ir_set_validshape(tile_ir, shape, span=span)
            self.builder.emit(_ir_core.EvalStmt(vs_call, span))

        return ConstInt(0, DataType.INDEX, span)

    return _ir_set_validshape(*args, **kwargs, span=span)

