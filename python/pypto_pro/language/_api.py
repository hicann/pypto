# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Python API declarations for the PyPTO Pro DSL (``pl.xxx``).

These declarations exist so that:
- IDE "Go to Definition" works for every ``pl.xxx`` call
- Python catches typos at import time (``pl.tensr`` raises ``AttributeError``)
- Type checkers can validate argument types
- Docstrings document the user-facing calling convention

None of these functions are meant to be called at runtime.  Inside a PyPTO
kernel the AST parser intercepts every ``pl.xxx`` call before Python executes
it.  Outside a kernel, calling a declaration raises ``RuntimeError``.

When adding a new op to the parser registry / block-default handler, add a
matching declaration here and re-export it from the package ``__init__.py``.
"""

from __future__ import annotations

import functools
import inspect
import builtins
from contextlib import contextmanager
from typing import Any, List, Optional, Union

from pypto.ir import (
    AccToVecMode,
    AtomicType,
    STPhase,
    AccPhase,
    QuantMode,
    ReluPreMode,
    RoundMode,
)

# ---------------------------------------------------------------------------
# User-facing type aliases (NOT IR types)
# ---------------------------------------------------------------------------
Tile = Any
TileGroup = Any
Tensor = Any
Scalar = Any
DType = Any
Offset = Union[List[int], int]

# ---------------------------------------------------------------------------
# API declaration decorator
# ---------------------------------------------------------------------------

_API_MSG = (
    "This function is a DSL API declaration and must be used inside "
    "a PyPTO kernel"
)


def _api_decl(func):
    sig = inspect.signature(func)

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        sig.bind(*args, **kwargs)
        raise RuntimeError(_API_MSG)

    wrapper.__wrapped__ = func
    return wrapper


# ===================================================================
# Section A: Data-movement block ops
# ===================================================================

@_api_decl
def load(dst_tile: Tile, src_tensor: Tensor, offsets: Offset, *,
         is_transpose: bool = False,
         tile_dims: Optional[List[int]] = None) -> None:
    """Load data from GM Tensor into on-chip Tile by absolute element coordinates.

    Args:
        dst_tile: Destination Tile (L1 or UB only)
        src_tensor: Source Tensor (global memory, from kernel parameter)
        offsets: Element-level offset per axis, e.g. ``[row, col]`` or ``[b, n, sq, sk]``
        is_transpose: Whether to transpose the data during load (default ``False``)
        tile_dims: Optional, which axes of the Tensor the Tile dimensions map to
    """


@_api_decl
def load_tile(dst_tile: Tile, src_tensor: Tensor, tile_offsets: Offset, *,
              is_transpose: bool = False,
              tile_dims: Optional[List[int]] = None) -> None:
    """Load data from GM Tensor into on-chip Tile by tile-block index.

    Offsets are in tile-block units, internally multiplied by tile shape.
    E.g. with tile shape ``[64, 128]``, ``tile_offsets=[2, 2]`` equals
    ``load`` with absolute offsets ``[128, 256]``.

    Args:
        dst_tile: Destination Tile (L1 or UB only)
        src_tensor: Source Tensor (global memory)
        tile_offsets: Tile-block index, e.g. ``[tile_row, tile_col]``
        is_transpose: Whether to transpose the data during load (default ``False``)
        tile_dims: Optional, which axes of the Tensor the Tile dimensions map to
    """


@_api_decl
def store(dst_tensor: Tensor, src_tile: Tile, offsets: Offset, *,
          relu_pre_mode: Optional[ReluPreMode] = None,
          pre_quant_scalar: Optional[int] = None,
          fp_tile: Optional[Tile] = None,
          tile_dims: Optional[List[int]] = None,
          atomic: AtomicType = AtomicType.AtomicNone,
          phase: Optional[STPhase] = None) -> None:
    """Store on-chip Tile back to GM Tensor by absolute element coordinates.

    Args:
        dst_tensor: Destination Tensor (global memory)
        src_tile: Source Tile (on-chip buffer)
        offsets: Element-level offset per axis, e.g. ``[row, col]`` or ``[b, n, sq, sk]``
        relu_pre_mode: Optional ReLU fusion — ``pl.ReluPreMode.NormalRelu``; mutually exclusive with ``fp_tile``
        pre_quant_scalar: Optional pre-quantization scalar (i64 bit pattern); mutually exclusive with ``fp_tile``
        fp_tile: Optional fixpipe quantization Tile; enables ``store_fp`` path,
            mutually exclusive with ``relu_pre_mode``, ``pre_quant_scalar``, and ``phase``
        tile_dims: Optional, which axes of the Tensor the Tile dimensions map to.
            When the Tensor has more dimensions than the Tile, this specifies the mapping.
            E.g. ``tile_dims=[0, 2]`` means Tile dim 0 → Tensor axis 0, Tile dim 1 → Tensor axis 2.
            Default: last N axes of the Tensor (N = Tile ndim)
        atomic: Atomic write mode — ``pl.AtomicType.AtomicNone`` (overwrite) or
            ``pl.AtomicType.AtomicAdd`` (atomic accumulate)
        phase: Fixpipe drain phase — ``pl.STPhase.Partial`` or ``pl.STPhase.Final``; mutually exclusive with ``fp_tile``
    """


@_api_decl
def store_tile(dst_tensor: Tensor, src_tile: Tile, tile_offsets: Offset, *,
                relu_pre_mode: Optional[ReluPreMode] = None,
                pre_quant_scalar: Optional[int] = None,
                fp_tile: Optional[Tile] = None,
                tile_dims: Optional[List[int]] = None,
                atomic: AtomicType = AtomicType.AtomicNone,
                phase: Optional[STPhase] = None) -> None:
    """Store on-chip Tile back to GM Tensor by tile-block index.

    Args:
        dst_tensor: Destination Tensor (global memory)
        src_tile: Source Tile (on-chip buffer)
        tile_offsets: Tile-block index, e.g. ``[tile_row, tile_col]``;
            internally multiplied by tile shape to get element offsets
        relu_pre_mode: Optional ReLU fusion — ``pl.ReluPreMode.NormalRelu``; mutually exclusive with ``fp_tile``
        pre_quant_scalar: Optional pre-quantization scalar (i64 bit pattern); mutually exclusive with ``fp_tile``
        fp_tile: Optional fixpipe quantization Tile; enables ``store_fp`` path,
            mutually exclusive with ``relu_pre_mode``, ``pre_quant_scalar``, and ``phase``
        tile_dims: Optional, which axes of the Tensor the Tile dimensions map to.
            When the Tensor has more dimensions than the Tile, this specifies the mapping.
            E.g. ``tile_dims=[0, 2]`` means Tile dim 0 → Tensor axis 0, Tile dim 1 → Tensor axis 2.
            Default: last N axes of the Tensor (N = Tile ndim)
        atomic: Atomic write mode — ``pl.AtomicType.AtomicNone`` (overwrite) or
            ``pl.AtomicType.AtomicAdd`` (atomic accumulate)
        phase: Fixpipe drain phase — ``pl.STPhase.Partial`` or ``pl.STPhase.Final``; mutually exclusive with ``fp_tile``
    """


@_api_decl
def move(dst_tile: Tile, src_tile: Tile, offset: Optional[Offset] = None, *,
         acc_to_vec_mode: Optional[AccToVecMode] = None,
         relu_pre_mode: Optional[ReluPreMode] = None,
         pre_quant_scalar: Optional[int] = None,
         fp_tile: Optional[Tile] = None) -> None:
    """Move data between on-chip Tiles (tile↔tile, no GM access).

    Supported memory-space paths:

    ============ ============ ========
    src          dst          pipe
    ============ ============ ========
    Acc (L0C)    Vec (UB)     fix
    Mat (L1)     Left (L0A)   mte1
    Mat (L1)     Right (L0B)  mte1
    Mat (L1)     Vec (UB)     v
    Vec (UB)     Mat (L1)     mte3
    others       —            v
    ============ ============ ========

    Supported fusion (side operations):

    - ``acc_to_vec_mode``: Acc→Vec conversion mode (single/dual split M/N)
    - ``relu_pre_mode``: ReLU activation before destination
    - ``pre_quant_scalar``: pre-quantization scalar (mutually exclusive with ``fp_tile``)
    - ``fp_tile``: fixpipe quantization tile, enables ``move_fp`` path
      (mutually exclusive with ``pre_quant_scalar``; single-mode only)

    Args:
        dst_tile: Destination Tile
        src_tile: Source Tile
        offset: Optional ``[offset_m, offset_k]`` to extract a sub-block from a wider source Tile
        acc_to_vec_mode: Acc→Vec mode — ``pl.AccToVecMode.SingleModeVec0``,
            ``pl.AccToVecMode.DualModeSplitM``, ``pl.AccToVecMode.DualModeSplitN``;
            only meaningful when src is Acc and dst is Vec
        relu_pre_mode: Optional ReLU fusion — ``pl.ReluPreMode.NormalRelu``
        pre_quant_scalar: Optional pre-quantization scalar; mutually exclusive with ``fp_tile``
        fp_tile: Optional fixpipe quantization Tile; enables ``move_fp`` path,
            mutually exclusive with ``pre_quant_scalar``
    """


@_api_decl
def insert(dst_tile: Tile, src_tile: Tile, offset: List[int]) -> None:
    """Insert a small Tile into a larger Tile at the given 2-D offset (TINSERT).

    Args:
        dst_tile: Destination (larger) Tile
        src_tile: Source (smaller) Tile
        offset: Insertion coordinates ``[row, col]`` in the destination
    """


@_api_decl
def ssbuf_load(struct_var: Any, offset: int) -> None:
    """Load data from SuperScalar Buffer (SSBUF) into a struct variable.

    Args:
        struct_var: Struct variable (created via ``pl.struct``)
        offset: SSBUF offset
    """


@_api_decl
def ssbuf_store(struct_var: Any, offset: int) -> None:
    """Write a struct variable to SuperScalar Buffer (SSBUF).

    Args:
        struct_var: Struct variable (created via ``pl.struct``)
        offset: SSBUF offset
    """


# ===================================================================
# Section B: Compute block ops
# ===================================================================

# --- B1. Binary element-wise (out, lhs, rhs) ---
#
# Constraints (apply to all ops in this section):
# - No broadcast: ``out``, ``lhs``, ``rhs`` (when Tile) must have identical shape.
# - No implicit type promotion: all operands must have the same dtype.
# - Supported dtypes: FP16, FP32, BF16 (op-dependent; FP8 not supported).

@_api_decl
def add(out: Tile, lhs: Tile, rhs: Union[Tile, Scalar]) -> None:
    """Element-wise addition: ``out = lhs + rhs``

    Supports both tile-tile and tile-scalar operations:
        - ``add(out, tile_a, tile_b)`` -> tile-tile
        - ``add(out, tile_a, scalar)`` -> tile-scalar
    """


@_api_decl
def sub(out: Tile, lhs: Tile, rhs: Union[Tile, Scalar]) -> None:
    """Element-wise subtraction: ``out = lhs - rhs``

    Supports both tile-tile and tile-scalar operations:
        - ``sub(out, tile_a, tile_b)`` -> tile-tile
        - ``sub(out, tile_a, scalar)`` -> tile-scalar
    """


@_api_decl
def mul(out: Tile, lhs: Tile, rhs: Union[Tile, Scalar]) -> None:
    """Element-wise multiplication: ``out = lhs * rhs``

    Supports both tile-tile and tile-scalar operations:
        - ``mul(out, tile_a, tile_b)`` -> tile-tile
        - ``mul(out, tile_a, scalar)`` -> tile-scalar
    """


@_api_decl
def div(out: Tile, lhs: Tile, rhs: Union[Tile, Scalar]) -> None:
    """Element-wise division: ``out = lhs / rhs``

    Supports both tile-tile and tile-scalar operations:
        - ``div(out, tile_a, tile_b)`` -> tile-tile
        - ``div(out, tile_a, scalar)`` -> tile-scalar
    """


# --- B2. Bitwise element-wise (out, lhs, rhs) ---

@_api_decl
def and_(out: Tile, lhs: Tile, rhs: Union[Tile, Scalar]) -> None:
    """Element-wise bitwise AND: ``out = lhs & rhs``

    Supports both tile-tile and tile-scalar operations:
        - ``and_(out, tile_a, tile_b)`` -> tile-tile
        - ``and_(out, tile_a, scalar)`` -> tile-scalar
    """


@_api_decl
def xor(out: Tile, lhs: Tile, rhs: Tile, tmp: Tile) -> None:
    """Element-wise bitwise XOR: ``out = lhs ^ rhs``

    Args:
        out: Destination Tile
        lhs: Left-hand Tile
        rhs: Right-hand Tile
        tmp: Workspace Tile
    """


@_api_decl
def expands(out: Tile, scalar: Scalar) -> None:
    """Fill Tile with a scalar (splat): ``out[i] = scalar``"""


# --- B3. Unary element-wise (out, src) ---

@_api_decl
def neg(out: Tile, src: Tile) -> None:
    """Element-wise negate: ``out = -src``"""


@_api_decl
def abs(out: Tile, src: Tile) -> None:
    """Element-wise absolute value: ``out = |src|``"""


@_api_decl
def exp(out: Tile, src: Tile) -> None:
    """Element-wise exponential: ``out = exp(src)``"""


@_api_decl
def log(out: Tile, src: Tile) -> None:
    """Element-wise natural log: ``out = log(src)``"""


@_api_decl
def sqrt(out: Tile, src: Tile) -> None:
    """Element-wise square root: ``out = sqrt(src)``"""


@_api_decl
def rsqrt(out: Tile, src: Tile) -> None:
    """Element-wise reciprocal square root: ``out = 1/sqrt(src)``"""


@_api_decl
def recip(out: Tile, src: Tile) -> None:
    """Element-wise reciprocal: ``out = 1/src``"""


@_api_decl
def relu(out: Tile, src: Tile) -> None:
    """Element-wise ReLU: ``out = max(0, src)``"""


@_api_decl
def fillpad(out: Tile, src: Tile) -> None:
    """Fill padding region of a Tile."""


@_api_decl
def fillpad_expand(out: Tile, src: Tile) -> None:
    """Fill padding region (expand mode)."""


@_api_decl
def fillpad_inplace(out: Tile, src: Tile) -> None:
    """Fill padding region in-place (dst and src share the same address)."""


# --- B4. Type conversion ---

@_api_decl
def cast(out: Tile, src: Tile, *, mode: RoundMode = RoundMode.CAST_ROUND) -> None:
    """Cast Tile to a different data type.

    The target dtype is inferred from ``out`` tile's dtype.

    Args:
        out: Destination Tile (determines target dtype)
        src: Source Tile
        mode: Rounding mode — ``pl.RoundMode.CAST_NONE``, ``pl.RoundMode.CAST_RINT``, ``pl.RoundMode.CAST_ROUND``,
            ``pl.RoundMode.CAST_FLOOR``, ``pl.RoundMode.CAST_CEIL``,
            ``pl.RoundMode.CAST_TRUNC``, ``pl.RoundMode.CAST_ODD``
    """


@_api_decl
def add_relu_cast(out: Tile, lhs: Tile, rhs: Tile, *,
                  target_type: DType, mode: RoundMode = RoundMode.CAST_ROUND) -> None:
    """Fused add + ReLU + cast: ``out = cast(relu(lhs + rhs), target_type)``"""


@_api_decl
def sub_relu_cast(out: Tile, lhs: Tile, rhs: Tile, *,
                  target_type: DType, mode: RoundMode = RoundMode.CAST_ROUND) -> None:
    """Fused sub + ReLU + cast: ``out = cast(relu(lhs - rhs), target_type)``"""


@_api_decl
def mul_cast(out: Tile, lhs: Tile, rhs: Tile, *,
             target_type: DType, mode: RoundMode = RoundMode.CAST_ROUND) -> None:
    """Fused mul + cast: ``out = cast(lhs * rhs, target_type)``"""


# --- B5. Compare / select ---

@_api_decl
def eq(out: Tile, lhs: Tile, rhs: Union[Tile, Scalar]) -> None:
    """Element-wise equal: ``out = (lhs == rhs)``

    Supports both tile-tile and tile-scalar operations:
        - ``eq(out, tile_a, tile_b)`` -> tile-tile
        - ``eq(out, tile_a, scalar)`` -> tile-scalar
    """


@_api_decl
def ne(out: Tile, lhs: Tile, rhs: Union[Tile, Scalar]) -> None:
    """Element-wise not equal: ``out = (lhs != rhs)``

    Supports both tile-tile and tile-scalar operations:
        - ``ne(out, tile_a, tile_b)`` -> tile-tile
        - ``ne(out, tile_a, scalar)`` -> tile-scalar
    """


@_api_decl
def lt(out: Tile, lhs: Tile, rhs: Union[Tile, Scalar]) -> None:
    """Element-wise less than: ``out = (lhs < rhs)``

    Supports both tile-tile and tile-scalar operations:
        - ``lt(out, tile_a, tile_b)`` -> tile-tile
        - ``lt(out, tile_a, scalar)`` -> tile-scalar
    """


@_api_decl
def le(out: Tile, lhs: Tile, rhs: Union[Tile, Scalar]) -> None:
    """Element-wise less or equal: ``out = (lhs <= rhs)``

    Supports both tile-tile and tile-scalar operations:
        - ``le(out, tile_a, tile_b)`` -> tile-tile
        - ``le(out, tile_a, scalar)`` -> tile-scalar
    """


@_api_decl
def gt(out: Tile, lhs: Tile, rhs: Union[Tile, Scalar]) -> None:
    """Element-wise greater than: ``out = (lhs > rhs)``

    Supports both tile-tile and tile-scalar operations:
        - ``gt(out, tile_a, tile_b)`` -> tile-tile
        - ``gt(out, tile_a, scalar)`` -> tile-scalar
    """


@_api_decl
def ge(out: Tile, lhs: Tile, rhs: Union[Tile, Scalar]) -> None:
    """Element-wise greater or equal: ``out = (lhs >= rhs)``

    Supports both tile-tile and tile-scalar operations:
        - ``ge(out, tile_a, tile_b)`` -> tile-tile
        - ``ge(out, tile_a, scalar)`` -> tile-scalar
    """


@_api_decl
def select(out: Tile, mask: Tile, lhs: Tile, rhs: Union[Tile, Scalar], tmp: Tile) -> None:
    """Select by mask: ``out[i] = lhs[i] if mask[i] else rhs[i]``

    Supports both tile-tile and tile-scalar operations:
        - ``select(out, mask, tile_a, tile_b, tmp)`` -> tile-tile
        - ``select(out, mask, tile_a, scalar, tmp)`` -> tile-scalar

    Args:
        out: Destination Tile
        mask: Mask Tile (from ``eq``/``gt``/...)
        lhs: Tile selected when mask is true
        rhs: Tile or Scalar selected when mask is false
        tmp: Workspace Tile
    """


# --- B6. Fused ops ---

@_api_decl
def add_relu(out: Tile, lhs: Tile, rhs: Tile) -> None:
    """Fused add + ReLU: ``out = relu(lhs + rhs)``"""


@_api_decl
def sub_relu(out: Tile, lhs: Tile, rhs: Tile) -> None:
    """Fused sub + ReLU: ``out = relu(lhs - rhs)``"""


@_api_decl
def addc(out: Tile, a: Tile, b: Tile, c: Tile) -> None:
    """Three-operand add: ``out = a + b + c``"""


@_api_decl
def mul_add_dst(out: Tile, lhs: Tile, rhs: Tile) -> None:
    """Fused multiply-add into out: ``out = lhs * rhs + out``"""


@_api_decl
def fused_mul_add(out: Tile, lhs: Tile, rhs: Tile) -> None:
    """Fused multiply-add in-place: ``out = lhs * out + rhs``"""


@_api_decl
def fused_mul_add_relu(out: Tile, a: Tile, b: Tile) -> None:
    """Fused multiply-add + ReLU in-place: ``out = relu(out * a + b)``"""


@_api_decl
def axpy(out: Tile, src: Tile, alpha: Scalar) -> None:
    """AXPY: ``out[i] = alpha * src[i] + out[i]``

    Args:
        out: Destination Tile (also accumulates)
        src: Source Tile
        alpha: Scalar multiplier
    """


@_api_decl
def partadd(out: Tile, src0: Tile, src1: Tile) -> None:
    """Partial addition: ``out = src0 + src1`` (only src1's valid region)."""


# --- B7. Matrix ops ---

@_api_decl
def matmul(dst_tile: Tile, lhs_tile: Tile, rhs_tile: Tile, *,
           phase: Optional[AccPhase] = None) -> None:
    """Matrix multiply: ``dst = lhs @ rhs`` (L0A × L0B → L0C).

    Args:
        dst_tile: Accumulator Tile (L0C, output)
        lhs_tile: Left matrix Tile (L0A)
        rhs_tile: Right matrix Tile (L0B)
        phase: Optional — ``pl.AccPhase.Partial`` or ``pl.AccPhase.Final``
    """


@_api_decl
def matmul_acc(dst_tile: Tile, acc_tile: Tile, lhs_tile: Tile, rhs_tile: Tile, *,
               phase: Optional[AccPhase] = None) -> None:
    """Accumulating matrix multiply: ``dst = acc + lhs @ rhs`` (K-dim block accumulation).

    Args:
        dst_tile: Destination Tile (L0C, output)
        acc_tile: Accumulator Tile (existing value to add to)
        lhs_tile: Left matrix Tile (L0A)
        rhs_tile: Right matrix Tile (L0B)
        phase: Optional — ``pl.AccPhase.Partial`` or ``pl.AccPhase.Final``
    """


@_api_decl
def transpose(out: Tile, src: Tile) -> None:
    """Transpose Tile by swapping the last two dimensions.

    Args:
        out: Destination Tile
        src: Source Tile
    """

# --- B8. Reductions / expands ---
#
# Unified reduce / expand interfaces.  ``dim=0`` reduces along the last axis
# (row-wise), ``dim=1`` reduces along the first axis (column-wise).
#
# tmp Tile constraints (apply to all reduce ops with ``tmp`` parameter):
# - dtype: must match ``src`` Tile dtype
# - shape: must match ``src`` Tile shape (1:1 workspace, not reduced size)
# - memory: must be in UB (Vec) memory space


@_api_decl
def sum(out: Tile, src: Tile, tmp: Tile, *, dim: int = 0) -> None:
    """Sum reduction along the specified dimension.

    Args:
        out: Destination Tile
        src: Source Tile
        tmp: Workspace Tile (required by hardware)
        dim: Reduction dimension — 0=row (last axis), 1=column (first axis)

    Example::

        tmp = pl.make_tile(src.type.shape, pl.DT_FP32, pl.MemorySpace.Vec, ...)
        pl.sum(out, src, tmp, dim=0)   # row-wise sum, out shape = [src.shape[0], 1]
    """


@_api_decl
def argmax(out: Tile, src: Tile, tmp: Tile, *, dim: int = 0) -> None:
    """Argmax reduction along the specified dimension.

    Args:
        out: Destination Tile
        src: Source Tile
        tmp: Workspace Tile (required by hardware)
        dim: Reduction dimension — 0=row (last axis), 1=column (first axis)
    """


@_api_decl
def argmin(out: Tile, src: Tile, tmp: Tile, *, dim: int = 0) -> None:
    """Argmin reduction along the specified dimension.

    Args:
        out: Destination Tile
        src: Source Tile
        tmp: Workspace Tile (required by hardware)
        dim: Reduction dimension — 0=row (last axis), 1=column (first axis)
    """


@_api_decl
def expand_max(out: Tile, src: Tile, scalar: Tile, *, dim: int = 0) -> None:
    """Max expand (broadcast reduction result back to full shape).

    Args:
        out: Destination Tile
        src: Source Tile (reduction result)
        scalar: Scalar Tile for the expand operation
        dim: Expand dimension — 0=row, 1=column
    """


@_api_decl
def expand_min(out: Tile, src: Tile, scalar: Tile, *, dim: int = 0) -> None:
    """Min expand.

    Args:
        out: Destination Tile
        src: Source Tile (reduction result)
        scalar: Scalar Tile for the expand operation
        dim: Expand dimension — 0=row, 1=column
    """


@_api_decl
def expand_mul(out: Tile, src: Tile, scalar: Tile, *, dim: int = 0) -> None:
    """Multiply expand.

    Args:
        out: Destination Tile
        src: Source Tile (reduction result)
        scalar: Scalar Tile for the expand operation
        dim: Expand dimension — 0=row, 1=column
    """


@_api_decl
def expand_sub(out: Tile, src: Tile, scalar: Tile, *, dim: int = 0) -> None:
    """Subtract expand.

    Args:
        out: Destination Tile
        src: Source Tile (reduction result)
        scalar: Scalar Tile for the expand operation
        dim: Expand dimension — 0=row, 1=column
    """


@_api_decl
def expand_div(out: Tile, src: Tile, scalar: Tile, *, dim: int = 0) -> None:
    """Divide expand.

    Args:
        out: Destination Tile
        src: Source Tile (reduction result)
        scalar: Scalar Tile for the expand operation
        dim: Expand dimension — 0=row, 1=column
    """


# --- B9. Gather / scatter / sort ---

@_api_decl
def gather(out: Tile, src: Tile, idx: Tile, tmp: Tile, *,
           cmp_mode: int = 0, offset: int = 0) -> None:
    """Gather elements by index.

    Args:
        out: Destination Tile
        src: Source Tile
        idx: Index Tile
        tmp: Workspace Tile
        cmp_mode: Comparison mode (default 0)
        offset: Index offset
    """


@_api_decl
def gatherb(out: Tile, src: Tile, offsets: Tile) -> None:
    """Gather elements by 32-byte block byte offset.

    Args:
        out: Destination Tile
        src: Source Tile
        offsets: Byte offset Tile
    """


@_api_decl
def gathermask(out: Tile, src: Tile, *, pattern_mode: int) -> None:
    """Extract columns by bit-pattern mask.

    Args:
        out: Destination Tile
        src: Source Tile
        pattern_mode: Bit-pattern extraction mode
    """


@_api_decl
def scatter(out: Tile, src: Tile, idx: Tile) -> None:
    """Scatter elements by index.

    Args:
        out: Destination Tile
        src: Source Tile
        idx: Index Tile
    """


@_api_decl
def mrgsort(dst: Tile, src: Tile, *, block_len: int) -> None:
    """Merge sort.

    Args:
        dst: Sorted output Tile
        src: Source Tile
        block_len: Sort block length
    """


@_api_decl
def mrgsort2(src0: Tile, src1: Tile, dst: Tile, tmp: Tile, *args,
             exhausted: bool = False) -> None:
    """Two-way (or multi-way) merge sort.

    Args:
        src0: First source Tile (val-idx pairs)
        src1: Second source Tile (val-idx pairs)
        dst: Destination Tile
        tmp: Workspace Tile
        *args: Optional additional source Tiles for multi-way merge
        exhausted: Whether any source is already exhausted
    """


@_api_decl
def sort32(dst: Tile, src: Tile, idx: Tile, tmp: Optional[Tile] = None) -> None:
    """Sort 32 elements with index tracking.

    Args:
        dst: Destination Tile for sorted values
        src: Source Tile (32 elements)
        idx: Index Tile for tracking original positions
        tmp: Optional workspace Tile (for tail-block handling)
    """


@_api_decl
def histogram(dst: Tile, src: Tile, idx: Tile, *, is_msb: bool) -> None:
    """Histogram accumulation for radix sort preprocessing.

    Counts byte-value frequencies in *src* and writes per-row bin counts to *dst*.

    Args:
        dst: Destination Tile (``pl.DT_UINT32``, cols >= 256)
        src: Source Tile (``pl.DT_UINT16``)
        idx: Index Tile (``pl.DT_UINT8``, DN layout); used for filtering when ``is_msb=False``
        is_msb: ``True`` counts high byte (bits 15-8); ``False`` counts low byte (bits 7-0)
            filtered by rows where the high byte matches ``idx``
    """


# --- B10. Quantization / index / misc ---

@_api_decl
def quant(out: Tile, src: Tile, scale: Tile, *,
          mode: QuantMode = QuantMode.SYM, offset: Optional[Tile] = None) -> None:
    """Quantize Tile (high-precision → low-precision integer).

    Args:
        out: Destination Tile (quantized output)
        src: Source Tile (float input)
        scale: Scale Tile
        mode: ``pl.QuantMode.SYM`` (symmetric) or ``pl.QuantMode.ASYM`` (asymmetric, requires *offset*)
        offset: Offset Tile (required for ``mode=pl.QuantMode.ASYM``)
    """


@_api_decl
def dequant(out: Tile, src: Tile, scale: Tile, offset: Tile) -> None:
    """Dequantize Tile (low-precision integer → high-precision).

    Args:
        out: Destination Tile (float output)
        src: Source Tile (quantized input)
        scale: Scale Tile
        offset: Offset Tile
    """



@_api_decl
def getval(container: "Tile | Tensor", offset: int) -> Scalar:
    """Read a scalar value from a Tile or Tensor at the given linear offset."""


@_api_decl
def setval(container: "Tile | Tensor", offset: int, value: Scalar) -> None:
    """Write a scalar value into a Tile or Tensor at the given linear offset."""


@_api_decl
def set_validshape(tile: "Tile | TileGroup", shape: List[int]) -> None:
    """Set the valid shape of a Tile or tile_group (for partial-tile / tail-block operations).

    When a tile_group is passed, valid_shape is set on all tiles in the group.
    """


@_api_decl
def set_stride(tensor: "Tensor", stride: List[int]) -> None:
    """Override the per-dimension stride of a global (GM) tensor in place.

    Rewrites the tensor's stride descriptor so that subsequent ``pl.load`` /
    ``pl.store`` accesses walk the tensor with the given ``stride`` (in elements)
    instead of the default row-major stride. ``stride`` is a 2-element list
    ``[row_stride, col_stride]``; entries may be constants or runtime scalars
    (e.g. read from an input via ``pl.getval``). This lets a single load gather
    rows that are non-contiguous in GM (positive stride).
    """


@_api_decl
def set_mask_count() -> None:
    """Switch mask to counting mode."""


@_api_decl
def set_mask_norm() -> None:
    """Switch mask to per-bit normalization mode."""


@_api_decl
def set_vec_mask(mask_high: int, mask_low: int) -> None:
    """Explicitly set the 128-bit vector mask from two 64-bit integers."""


@_api_decl
def reset_mask() -> None:
    """Reset the mask to all-ones (no masking)."""


@_api_decl
def fill_index(out: Tile, start: Scalar) -> None:
    """Fill target tile with sequential indices starting from *start*."""


# ===================================================================
# Section C: VF namespace (``pl.vf.*``)
# ===================================================================

class Vf:
    """Vector Function unit operations (A5 architecture).

    Used inside ``@pl.vector_function`` decorated functions.  Compute ops
    produce a result and must use the assignment form ``dst = vf.xxx(...)``;
    the destination register is declared implicitly.  Only store/side-effect
    ops are called as bare statements.

    VF op signatures are defined by the C++ backend and may vary;
    use ``*args, **kwargs`` for flexibility.
    """

    @staticmethod
    @_api_decl
    def create_mask(*args, **kwargs):
        """Create and initialize a VF predicate mask register.

        Both kwargs are optional and may be given independently:
            preg = vf.create_mask(dtype=pl.DT_FP16)          # pattern defaults to ALL
            preg = vf.create_mask(pattern=pl.MaskPattern.VL8) # dtype defaults to FP32
            preg = vf.create_mask()                           # both defaults

        Args:
            pattern: Mask pattern, ``pl.MaskPattern.ALL`` (default) or other
                ``MaskPattern`` enum value (VL1..VL128, M3, M4, H, Q, ALLF)
            dtype: Data type that determines mask granularity (default FP32;
                FP16/UINT8 etc.)

        Returns:
            mask_reg: Initialized mask register
        """

    @staticmethod
    @_api_decl
    def update_mask(*args, **kwargs):
        """Update a VF mask register from a scalar value.

        Args:
            scalar: Scalar value whose bits define the new mask pattern

        Kwargs:
            dtype: Data type for mask width selection (default FP32 → b32)
        """

    @staticmethod
    @_api_decl
    def full(*args, **kwargs):
        """Broadcast a scalar or vector-source element into all lanes of a VF register.

        Two modes:

        - **Scalar mode**: ``vf.full(2.5, preg, dtype=pl.DT_FP32)`` — broadcasts a scalar
          value to all register lanes (vbr/vdup instruction).
        - **Tensor mode**: ``vf.full(src_reg, preg)`` — broadcasts the lowest or highest
          element of a source register to all lanes of the destination register
          (vdup instruction). Mask is required in Tensor mode.

        Args:
            scalar_or_src: Scalar value (Scalar mode) or source register (Tensor mode)
            mask: Predicate mask register. Required for Tensor mode; optional for Scalar mode

        Kwargs:
            dtype: Required for Scalar mode (cannot infer from scalar). Auto-inferred
                for Tensor mode from the source register.
            mode: ``pl.MergeMode.ZEROING`` (default) or ``pl.MergeMode.MERGING``
            pos: ``pl.DuplicatePos.LOWEST`` (default) or ``pl.DuplicatePos.HIGHEST`` —
                selects which element to broadcast in Tensor mode
        """

    @staticmethod
    @_api_decl
    def load_align(*args, **kwargs):
        """Load aligned data from a UB Tile into a VF register (vlds instruction).

        Two calling forms:

            vf.load_align(dst, src, offset)     # statement form
            dst = vf.load_align(src, offset)    # assignment form (dst must be pre-declared)

        Args:
            dst: Destination register
            src: Source UB Tile pointer
            offset: Element offset into the tile (or post-update stride)

        Kwargs:
            data_copy_mode: ``pl.DataCopyMode.DATA_BLOCK_COPY`` (AscendC's name for
                the non-contiguous datablock load) selects the vsldb instruction.
                ``DATA_BLOCK_LOAD`` is accepted as an equivalent legacy alias.
            block_stride: Datablock stride in bytes for the datablock-copy mode
            post_mode: ``"POST_UPDATE"`` for post-increment addressing
        """

    @staticmethod
    @_api_decl
    def store_align(*args, **kwargs):
        """Store aligned data from a VF register to a UB Tile (vsts instruction).

        Statement form only (no assignment form — dst is a UB tile, not a register):

            vf.store_align(dst_tile, src, mask)

        Args:
            dst: Destination UB Tile pointer
            src: Source register
            mask: Predicate mask register

        Kwargs:
            dist: ``"FIRST_ELEMENT"`` to store only lane 0; ``"PACK"`` for packed store
            mode: ``pl.MergeMode.ZEROING`` (default) or ``pl.MergeMode.MERGING``
        """

    @staticmethod
    @_api_decl
    def store_unalign(*args, **kwargs):
        """Store unaligned data from a VF register to UB (vstur/vstus instruction).

        Args:
            dst_ptr: Destination UB pointer
            src: Source register
            align_reg: Alignment register (from unalign_reg_for_store)

        When called with 4 args, the 4th arg is stride for strided mode (vstus).
        """

    @staticmethod
    @_api_decl
    def store_unalign_post(*args, **kwargs):
        """Complete an unaligned store sequence (vstar/vstas instruction).

        Args:
            dst_ptr: Destination UB pointer
            align_reg: Alignment register (from unalign_reg_for_store)

        When called with 3 args, the 3rd arg is stride for strided mode (vstas).
        """

    @staticmethod
    @_api_decl
    def unalign_reg_for_store(*args, **kwargs):
        """Declare an unaligned register for store operations.

        Must be called before store_unalign/store_unalign_post to allocate the
        alignment state register.

        Returns:
            Unaligned register handle
        """

    @staticmethod
    @_api_decl
    def mem_bar(*args, **kwargs):
        """Insert a VF memory barrier (maps to AscendC ``LocalMemBar<src,dst>``).

        Orders memory ops of the ``src`` kind before those of the ``dst`` kind.
        Select the pair via the ``mode`` kwarg (default ``VST_VLD``)::

            vf.mem_bar(mode=pl.MemBarMode.VST_VLD)   # vector store → vector load
            vf.mem_bar(mode=pl.MemBarMode.VST_VST)   # vector store → vector store (WAW)
            vf.mem_bar(mode=pl.MemBarMode.VV_ALL)    # all vector ↔ all vector

        Supported modes (12, matching AscendC's legal MemType combinations):
            VST_VLD, VLD_VST, VST_VST, VST_LD, VST_ST, VLD_ST,
            ST_VLD, ST_VST, LD_VST, VV_ALL, VS_ALL, SV_ALL
        where V*=vector, S*/*_LD/*_ST=scalar, *_ALL=full barrier of that unit.

        Kwargs:
            mode: ``pl.MemBarMode`` value selecting the src→dst ordering
        """

    @staticmethod
    @_api_decl
    def max(*args, **kwargs):
        """Element-wise maximum: ``dst[i] = max(src0[i], src1[i])``

        Args:
            dst: Destination register
            src0: First source register
            src1: Second source register
            mask: Predicate mask register

        Kwargs:
            mode: ``pl.MergeMode.ZEROING`` (default) or ``pl.MergeMode.MERGING``
        """

    @staticmethod
    @_api_decl
    def add(*args, **kwargs):
        """Element-wise addition: ``dst[i] = src0[i] + src1[i]``

        Assignment form (the destination register is declared implicitly):

            dst = vf.add(src0, src1, pred)

        Args:
            dst: Destination register
            src0: First source register
            src1: Second source register
            mask: Predicate mask register

        Kwargs:
            mode: ``pl.MergeMode.ZEROING`` (default) or ``pl.MergeMode.MERGING``
        """

    @staticmethod
    @_api_decl
    def sub(*args, **kwargs):
        """Element-wise subtraction: ``dst[i] = src0[i] - src1[i]``

        Args:
            dst: Destination register
            src0: First source register
            src1: Second source register
            mask: Predicate mask register

        Kwargs:
            mode: ``pl.MergeMode.ZEROING`` (default) or ``pl.MergeMode.MERGING``
        """

    @staticmethod
    @_api_decl
    def mul(*args, **kwargs):
        """Element-wise multiplication: ``dst[i] = src0[i] * src1[i]``

        Args:
            dst: Destination register
            src0: First source register
            src1: Second source register
            mask: Predicate mask register

        Kwargs:
            mode: ``pl.MergeMode.ZEROING`` (default) or ``pl.MergeMode.MERGING``
        """

    @staticmethod
    @_api_decl
    def div(*args, **kwargs):
        """Element-wise division: ``dst[i] = src0[i] / src1[i]``

        Args:
            dst: Destination register
            src0: Numerator register
            src1: Denominator register
            mask: Predicate mask register

        Kwargs:
            mode: ``pl.MergeMode.ZEROING`` (default) or ``pl.MergeMode.MERGING``
        """

    @staticmethod
    @_api_decl
    def muls(*args, **kwargs):
        """Multiply all elements by a scalar: ``dst[i] = src[i] * scalar``

        Args:
            dst: Destination register
            src: Source register
            scalar: Scalar multiplier value
            mask: Predicate mask register

        Kwargs:
            mode: ``pl.MergeMode.ZEROING`` (default) or ``pl.MergeMode.MERGING``

        Note:
            Does not support UINT8/INT8 types directly. For 1-byte types,
            the backend automatically expands to vdup+vmul.
        """

    @staticmethod
    @_api_decl
    def mul_add_dst(*args, **kwargs):
        """Fused multiply-add into destination: ``dst = src0 * src1 + dst``

        The destination register is both read (as addend) and written.
        Maps to hardware ``vmula`` instruction.

        Args:
            dst: Destination/accumulator register (read+write)
            src0: First multiplicand register
            src1: Second multiplicand register
            mask: Predicate mask register

        Kwargs:
            mode: ``pl.MergeMode.ZEROING`` (default) or ``pl.MergeMode.MERGING``
        """

    @staticmethod
    @_api_decl
    def and_(*args, **kwargs):
        """Element-wise bitwise AND: ``dst[i] = src0[i] & src1[i]``

        Args:
            dst: Destination register
            src0: First source register
            src1: Second source register
            mask: Predicate mask register

        Kwargs:
            mode: ``pl.MergeMode.ZEROING`` (default) or ``pl.MergeMode.MERGING``
        """

    @staticmethod
    @_api_decl
    def or_(*args, **kwargs):
        """Element-wise bitwise OR: ``dst[i] = src0[i] | src1[i]``

        Args:
            dst: Destination register
            src0: First source register
            src1: Second source register
            mask: Predicate mask register

        Kwargs:
            mode: ``pl.MergeMode.ZEROING`` (default) or ``pl.MergeMode.MERGING``
        """

    @staticmethod
    @_api_decl
    def xor(*args, **kwargs):
        """Element-wise bitwise XOR: ``dst[i] = src0[i] ^ src1[i]``

        Args:
            dst: Destination register
            src0: First source register
            src1: Second source register
            mask: Predicate mask register

        Kwargs:
            mode: ``pl.MergeMode.ZEROING`` (default) or ``pl.MergeMode.MERGING``
        """

    @staticmethod
    @_api_decl
    def reduce_sum(*args, **kwargs):
        """In-register sum reduction across all lanes (vcadd / vcgadd).

        Reduces all active lanes of the source register into the first element
        of the destination register.

        Args:
            src: Source register
            mask: Predicate mask register

        Kwargs:
            datablock: ``True`` to use datablock-granularity reduction (vcgadd)
            merge_mode: ``pl.MergeMode.ZEROING`` (default) or ``pl.MergeMode.MERGING``
        """

    @staticmethod
    @_api_decl
    def reduce_max(*args, **kwargs):
        """In-register max reduction across all lanes (vcmax / vcgmax).

        Reduces all active lanes of the source register into the first element
        of the destination register.

        Args:
            src: Source register
            mask: Predicate mask register

        Kwargs:
            datablock: ``True`` to use datablock-granularity reduction (vcgmax)
            merge_mode: ``pl.MergeMode.ZEROING`` (default) or ``pl.MergeMode.MERGING``
        """

    @staticmethod
    @_api_decl
    def reduce_min(*args, **kwargs):
        """In-register min reduction across all lanes (vcmin / vcgmin).

        Reduces all active lanes of the source register into the first element
        of the destination register.

        Args:
            src: Source register
            mask: Predicate mask register

        Kwargs:
            datablock: ``True`` to use datablock-granularity reduction (vcgmin)
            merge_mode: ``pl.MergeMode.ZEROING`` (default) or ``pl.MergeMode.MERGING``
        """

    @staticmethod
    @_api_decl
    def ln(*args, **kwargs):
        """Natural logarithm: ``dst[i] = ln(src[i])``

        Maps to hardware ``vln`` instruction.

        Args:
            dst: Destination register
            src: Source register (must be positive)
            mask: Predicate mask register

        Kwargs:
            mode: ``pl.MergeMode.ZEROING`` (default) or ``pl.MergeMode.MERGING``
            precision: ``"HIGH"`` to enable subnormal-input compensation
                       (prescale + vln + offset correction)
        """

    @staticmethod
    @_api_decl
    def exp_sub(*args, **kwargs):
        """Fused exp-subtract: ``dst[i] = exp(src[i] - max[i])``

        Commonly used in softmax computation. Subtracts max for numerical
        stability before applying exp.

        Args:
            src: Source register
            max: Max register (subtracted before exp)
            mask: Predicate mask register

        Kwargs:
            layout: ``pl.CastLayout.ZERO`` (even half, default) or
                ``pl.CastLayout.ONE`` (odd half) — for half-width results
        """

    @staticmethod
    @_api_decl
    def astype(*args, **kwargs):
        """Type conversion between register types (vcvt instruction).

        Supports same-width and cross-width conversions: float↔int, int→int
        narrowing/widening, float precision changes.

        Args:
            dst: Destination register (target type)
            src: Source register (source type)
            mask: Predicate mask register

        Kwargs:
            layout: ``pl.CastLayout.ZERO`` (default) / ``ONE`` / ``TWO`` / ``THREE``
            round_mode: ``pl.VFRoundMode.ROUND`` (default) / ``RINT`` / ``FLOOR`` / ``CEIL`` / ``TRUNC`` / ``RNA``
            saturate: ``pl.SaturateMode.OFF`` (default) or ``pl.SaturateMode.ON``
        """

    @staticmethod
    @_api_decl
    def de_interleave(*args, **kwargs):
        """De-interleave: split even/odd elements into two registers.

        ``dst0 = src[0], src[2], src[4], ...`` (even elements)
        ``dst1 = src[1], src[3], src[5], ...`` (odd elements)

        Args:
            dst0: Destination register for even elements
            dst1: Destination register for odd elements
            src0: First source register
            src1: Second source register

        Kwargs:
            dtype: When src operands are MaskReg, specifies the interleave bit-width
                (selects ``pdintlv_b8``/``b16``/``b32``). Inferred from src0 if omitted.
        """

    @staticmethod
    @_api_decl
    def select(*args, **kwargs):
        """Conditional select: ``dst[i] = src_true[i] if mask[i] else src_false[i]``

        Args:
            dst: Destination register
            src_true: Register selected when mask bit is 1
            src_false: Register selected when mask bit is 0
            mask: Predicate mask register

        Kwargs:
            mode: ``pl.MergeMode.ZEROING`` (default) or ``pl.MergeMode.MERGING``
        """

    @staticmethod
    @_api_decl
    def shift_left(*args, **kwargs):
        """Left shift: ``dst[i] = src[i] << shift``

        The shift amount may be a scalar (all lanes shifted by the same amount,
        emits ``vshls``) or a vector register (per-lane shift amount, emits
        ``vshl``). The form is selected automatically from the argument type::

            dst = vf.shift_left(src, 2, mask)          # scalar: uniform shift
            dst = vf.shift_left(src, shift_reg, mask)   # register: per-lane shift

        This unified op replaces the former separate ``vf.shift_lefts`` (scalar)
        and ``vf.shift_left`` (vector) entry points.

        Args:
            src: Source register
            shift: Shift amount — scalar integer (uniform) or a vector register
                (per-lane). Negative values are undefined.
            mask: Predicate mask register

        Kwargs:
            mode: ``pl.MergeMode.ZEROING`` (default) or ``pl.MergeMode.MERGING``
        """

    @staticmethod
    @_api_decl
    def shift_right(*args, **kwargs):
        """Right shift: ``dst[i] = src[i] >> shift``

        The shift amount may be a scalar (all lanes shifted by the same amount,
        emits ``vshrs``) or a vector register (per-lane shift amount, emits
        ``vshr``). The form is selected automatically from the argument type::

            dst = vf.shift_right(src, 24, mask)         # scalar: uniform shift
            dst = vf.shift_right(src, shift_reg, mask)   # register: per-lane shift

        This unified op replaces the former separate ``vf.shift_rights`` (scalar)
        and ``vf.shift_right`` (vector) entry points. Unsigned src does a logical
        shift; signed src does an arithmetic shift.

        Args:
            src: Source register
            shift: Shift amount — scalar integer (uniform) or a vector register
                (per-lane). Negative values are undefined.
            mask: Predicate mask register

        Kwargs:
            mode: ``pl.MergeMode.ZEROING`` (default) or ``pl.MergeMode.MERGING``
        """

    @staticmethod
    @_api_decl
    def histograms(*args, **kwargs):
        """Histogram accumulation (chistv2/dhistv2 instruction).

        Computes histogram on UINT8 data. Supports both cumulative (chistv2)
        and frequency (dhistv2) modes.  The destination register is both read
        and written — ``chistv2`` accumulates into the existing value, so the
        dst must be pre-initialized (e.g. via ``vf.full(0, ...)``) before the
        first call.  Subsequent ``dst = vf.histograms(...)`` calls reuse the
        same register and continue accumulating.

        Args:
            src: Source register (data to bin)
            mask: Predicate mask register

        Kwargs:
            bin_type: ``"BIN0"`` (default) or ``"BIN1"`` — selects bin mapping
            hist_type: ``"ACCUMULATE"`` (default, chistv2) or ``"FREQUENCY"`` (dhistv2)

        Returns:
            Destination histogram register (same register passed as dst,
            with accumulated histogram values)
        """

    @staticmethod
    @_api_decl
    def eq(*args, **kwargs):
        """Element-wise equality comparison.

        Compares two source elements and writes the result to the
        destination mask register.  If the second argument is a scalar
        literal the vector-scalar compare path is used; otherwise the
        vector-vector compare path is used.

        Args:
            a: First source register
            b: Second source register or scalar value
            mask: Source predicate mask

        Returns:
            MaskReg with comparison result (True where a_i == b_i)
        """

    @staticmethod
    @_api_decl
    def ne(*args, **kwargs):
        """Element-wise not-equal comparison.

        Args:
            a: First source register
            b: Second source register or scalar value
            mask: Source predicate mask

        Returns:
            MaskReg with comparison result (True where a_i != b_i)
        """

    @staticmethod
    @_api_decl
    def lt(*args, **kwargs):
        """Element-wise less-than comparison.

        Args:
            a: First source register
            b: Second source register or scalar value
            mask: Source predicate mask

        Returns:
            MaskReg with comparison result (True where a_i < b_i)
        """

    @staticmethod
    @_api_decl
    def gt(*args, **kwargs):
        """Element-wise greater-than comparison.

        Args:
            a: First source register
            b: Second source register or scalar value
            mask: Source predicate mask

        Returns:
            MaskReg with comparison result (True where a_i > b_i)
        """

    @staticmethod
    @_api_decl
    def le(*args, **kwargs):
        """Element-wise less-or-equal comparison.

        Args:
            a: First source register
            b: Second source register or scalar value
            mask: Source predicate mask

        Returns:
            MaskReg with comparison result (True where a_i <= b_i)
        """

    @staticmethod
    @_api_decl
    def ge(*args, **kwargs):
        """Element-wise greater-or-equal comparison.

        Args:
            a: First source register
            b: Second source register or scalar value
            mask: Source predicate mask

        Returns:
            MaskReg with comparison result (True where a_i >= b_i)
        """

    @staticmethod
    @_api_decl
    def squeeze(*args, **kwargs):
        """Squeeze mask to index register (vsqz instruction).

        Converts active mask bits into a packed index sequence in the
        destination register.

        Args:
            dst: Destination register (receives packed indices)
            src: Source register
            mask: Predicate mask register
        """

    @staticmethod
    @_api_decl
    def arange(*args, **kwargs):
        """Generate an index sequence starting from ``start`` (vci instruction).

        Fills each lane with a sequential value. ``index_order`` selects the
        direction::

            dst = vf.arange(start)                                          # dst[i] = start + i (INC)
            dst = vf.arange(start, index_order=pl.IndexOrder.INCREASE_ORDER)  # same as default
            dst = vf.arange(start, index_order=pl.IndexOrder.DECREASE_ORDER)  # dst[i] = start - i (DEC)

        The per-lane step is fixed at ±1 (a hardware property of vci); use a
        following ``vf.muls`` to scale the step if needed.

        Args:
            start: Starting value (scalar)

        Kwargs:
            index_order: ``pl.IndexOrder.INCREASE_ORDER`` (default, dst[i]=start+i)
                or ``pl.IndexOrder.DECREASE_ORDER`` (dst[i]=start-i)
        """

    @staticmethod
    @_api_decl
    def gather(*args, **kwargs):
        """Gather elements by index from UB memory.

        Reads elements from non-contiguous UB locations specified by an index register.
        The ``data_copy_mode`` kwarg selects the gather granularity:

            dst = vf.gather(src_ub, indices, mask)                                    # per-element (vgather2)
            dst = vf.gather(src_ub, indices, mask,
                            data_copy_mode=pl.DataCopyMode.DATA_BLOCK_LOAD)           # per 32B datablock (vgatherb)

        The DATA_BLOCK_LOAD form replaces the former standalone ``vf.gatherb`` op.

        Args:
            src_ub: Source UB pointer (base address)
            indices: Index register. NORM mode: element offsets; DATA_BLOCK_LOAD
                mode: byte offsets, must be 32-byte aligned (one index per datablock)
            mask: Predicate mask register

        Kwargs:
            data_copy_mode: ``pl.DataCopyMode.NORM`` (default, per-element via vgather2)
                or ``pl.DataCopyMode.DATA_BLOCK_LOAD`` (per 32B datablock via vgatherb)
        """

    @staticmethod
    @_api_decl
    def clear_spr(*args, **kwargs):
        """Clear special purpose register (AR register).

        Resets the accumulator register used by certain VF instructions.
        """

    @staticmethod
    @_api_decl
    def log(*args, **kwargs):
        """Natural logarithm (alias for ln): ``dst[i] = ln(src[i])``

        Identical to ``vf.ln()``. Both map to the hardware ``vln`` instruction.

        Args:
            dst: Destination register
            src: Source register (must be positive)
            mask: Predicate mask register

        Kwargs:
            mode: ``pl.MergeMode.ZEROING`` (default) or ``pl.MergeMode.MERGING``
            precision: ``"HIGH"`` for subnormal compensation
        """

    @staticmethod
    @_api_decl
    def min(*args, **kwargs):
        """Element-wise minimum: ``dst[i] = min(src0[i], src1[i])``

        Args:
            dst: Destination register
            src0: First source register
            src1: Second source register
            mask: Predicate mask register

        Kwargs:
            mode: ``pl.MergeMode.ZEROING`` (default) or ``pl.MergeMode.MERGING``
        """

    @staticmethod
    @_api_decl
    def exp(*args, **kwargs):
        """Exponential function: ``dst[i] = exp(src[i])``

        Maps to hardware ``vexp`` instruction.

        Args:
            dst: Destination register
            src: Source register
            mask: Predicate mask register

        Kwargs:
            mode: ``pl.MergeMode.ZEROING`` (default) or ``pl.MergeMode.MERGING``
            precision: ``"HIGH"`` to detect subnormal results and apply
                       exp(x+ln2)*0.5 correction
        """

    @staticmethod
    @_api_decl
    def abs(*args, **kwargs):
        """Element-wise absolute value: ``dst[i] = |src[i]|``

        Args:
            dst: Destination register
            src: Source register
            mask: Predicate mask register

        Kwargs:
            mode: ``pl.MergeMode.ZEROING`` (default) or ``pl.MergeMode.MERGING``
        """

    @staticmethod
    @_api_decl
    def not_(*args, **kwargs):
        """Element-wise bitwise NOT: ``dst[i] = ~src[i]``

        Args:
            dst: Destination register
            src: Source register
            mask: Predicate mask register

        Kwargs:
            mode: ``pl.MergeMode.ZEROING`` (default) or ``pl.MergeMode.MERGING``
        """

    @staticmethod
    @_api_decl
    def sqrt(*args, **kwargs):
        """Square root: ``dst[i] = sqrt(src[i])``

        Maps to hardware ``vsqrt`` instruction.

        Args:
            dst: Destination register
            src: Source register (must be non-negative)
            mask: Predicate mask register

        Kwargs:
            mode: ``pl.MergeMode.ZEROING`` (default) or ``pl.MergeMode.MERGING``
            precision: ``"HIGH"`` for FP16 subnormal precision path
                       (prescale x2^10, vsqrt, then x2^-5)
        """

    @staticmethod
    @_api_decl
    def relu(*args, **kwargs):
        """ReLU activation: ``dst[i] = max(0, src[i])``

        Args:
            dst: Destination register
            src: Source register
            mask: Predicate mask register

        Kwargs:
            mode: ``pl.MergeMode.ZEROING`` (default) or ``pl.MergeMode.MERGING``
        """

    @staticmethod
    @_api_decl
    def neg(*args, **kwargs):
        """Negation: ``dst[i] = -src[i]``

        Args:
            dst: Destination register
            src: Source register
            mask: Predicate mask register

        Kwargs:
            mode: ``pl.MergeMode.ZEROING`` (default) or ``pl.MergeMode.MERGING``
        """

    @staticmethod
    @_api_decl
    def adds(*args, **kwargs):
        """Add scalar to each element: ``dst[i] = src[i] + scalar``

        Args:
            dst: Destination register
            src: Source register
            scalar: Scalar addend value
            mask: Predicate mask register

        Kwargs:
            mode: ``pl.MergeMode.ZEROING`` (default) or ``pl.MergeMode.MERGING``
        """

    @staticmethod
    @_api_decl
    def subs(*args, **kwargs):
        """Subtract scalar from each element: ``dst[i] = src[i] - scalar``

        Implemented as ``vadds(dst, src, -scalar, ...)``.

        Args:
            dst: Destination register
            src: Source register
            scalar: Scalar subtrahend value
            mask: Predicate mask register

        Kwargs:
            mode: ``pl.MergeMode.ZEROING`` (default) or ``pl.MergeMode.MERGING``
        """

    @staticmethod
    @_api_decl
    def mins(*args, **kwargs):
        """Element-wise minimum with scalar: ``dst[i] = min(src[i], scalar)``

        Args:
            dst: Destination register
            src: Source register
            scalar: Scalar value to compare against
            mask: Predicate mask register

        Kwargs:
            mode: ``pl.MergeMode.ZEROING`` (default) or ``pl.MergeMode.MERGING``
        """

    @staticmethod
    @_api_decl
    def maxs(*args, **kwargs):
        """Element-wise maximum with scalar: ``dst[i] = max(src[i], scalar)``

        Args:
            dst: Destination register
            src: Source register
            scalar: Scalar value to compare against
            mask: Predicate mask register

        Kwargs:
            mode: ``pl.MergeMode.ZEROING`` (default) or ``pl.MergeMode.MERGING``
        """

    @staticmethod
    @_api_decl
    def leaky_relu(*args, **kwargs):
        """Leaky ReLU: ``dst[i] = src[i] if src[i] >= 0 else alpha * src[i]``

        Args:
            dst: Destination register
            src: Source register
            alpha: Negative slope scalar (e.g. 0.1)
            mask: Predicate mask register

        Kwargs:
            mode: ``pl.MergeMode.ZEROING`` (default) or ``pl.MergeMode.MERGING``
        """

    @staticmethod
    @_api_decl
    def interleave(*args, **kwargs):
        """Interleave two registers: combine even/odd elements.

        ``dst0 = src0[0], src1[0], src0[1], src1[1], ...``
        ``dst1 = src0[N/2], src1[N/2], ...``

        Two calling forms:

            vf.interleave(dst0, dst1, src0, src1)          # statement form
            dst0, dst1 = vf.interleave(src0, src1)         # tuple assignment form

        Args:
            dst0: First destination register
            dst1: Second destination register
            src0: First source register
            src1: Second source register

        Kwargs:
            dtype: When src operands are MaskReg, specifies the interleave bit-width
                (selects ``pintlv_b8``/``b16``/``b32``). Inferred from src0 if omitted.
        """

    @staticmethod
    @_api_decl
    def pair_reduce_sum(*args, **kwargs):
        """Pairwise reduction sum: ``dst[i] = src[2i] + src[2i+1]``

        Adds adjacent pairs of elements. Maps to hardware ``vcpadd`` instruction.

        Args:
            dst: Destination register
            src: Source register
            mask: Predicate mask register

        Kwargs:
            mode: ``pl.MergeMode.ZEROING`` (default) or ``pl.MergeMode.MERGING``
        """

    @staticmethod
    @_api_decl
    def abs_sub(*args, **kwargs):
        """Absolute difference: ``dst[i] = |src0[i] - src1[i]|``

        Maps to hardware ``vabsdif`` instruction.

        Args:
            dst: Destination register
            src0: First source register
            src1: Second source register
            mask: Predicate mask register

        Kwargs:
            mode: ``pl.MergeMode.ZEROING`` (default) or ``pl.MergeMode.MERGING``
        """

    @staticmethod
    @_api_decl
    def axpy(*args, **kwargs):
        """Fused AXPY: ``dst = src * scalar + dst``

        The destination register is both read (as addend) and written.
        Maps to hardware ``vaxpy`` instruction.

        Args:
            dst: Destination/accumulator register (read+write)
            src: Source register
            scalar: Scalar multiplier
            mask: Predicate mask register

        Kwargs:
            mode: ``pl.MergeMode.ZEROING`` (default) or ``pl.MergeMode.MERGING``
        """

    @staticmethod
    @_api_decl
    def copy(*args, **kwargs):
        """Register copy: ``dst[i] = src[i]`` where mask is active.

        Maps to hardware ``vmov`` instruction with MODE_MERGING.

        Args:
            dst: Destination register
            src: Source register
            mask: Predicate mask register

        Kwargs:
            mode: ``pl.MergeMode.ZEROING`` (default) or ``pl.MergeMode.MERGING``
        """

    @staticmethod
    @_api_decl
    def mul_dst_add(*args, **kwargs):
        """Multiply-dst-add: ``dst = dst * src0 + src1`` (vmadd)

        The destination register is both read (as multiplicand) and written.
        Maps to hardware ``vmadd`` instruction (AscendC MulDstAdd).

        Args:
            dst: Destination/accumulator register (read+write)
            src0: First multiplicand register
            src1: Addend register
            mask: Predicate mask register

        Kwargs:
            mode: ``pl.MergeMode.ZEROING`` (default) or ``pl.MergeMode.MERGING``
        """

    @staticmethod
    @_api_decl
    def pack(*args, **kwargs):
        """Pack/narrow data type (e.g. u32->u16, u16->u8).

        Selects upper or lower half of each wider element.

        Args:
            dst: Destination register (narrower type)
            src: Source register (wider type)

        Kwargs:
            part: ``pl.PackPart.LOWER`` (default) or ``pl.PackPart.UPPER``
        """

    @staticmethod
    @_api_decl
    def unpack(*args, **kwargs):
        """Unpack/widen data type (e.g. u8->u16, u16->u32).

        Zero-extends or sign-extends narrower elements into wider type.

        Args:
            dst: Destination register (wider type)
            src: Source register (narrower type)

        Kwargs:
            part: ``pl.PackPart.LOWER`` (default) or ``pl.PackPart.UPPER`` — which half of src to unpack
        """

    @staticmethod
    @_api_decl
    def prelu(*args, **kwargs):
        """Parametric ReLU with per-element slope register.

        ``dst[i] = src[i] if src[i] >= 0 else src[i] * slope[i]``

        Unlike leaky_relu which uses a scalar alpha, prelu uses a per-element
        slope vector.

        Args:
            dst: Destination register
            src: Source register
            slope: Slope register (per-element negative slope values)
            mask: Predicate mask register

        Kwargs:
            mode: ``pl.MergeMode.ZEROING`` (default) or ``pl.MergeMode.MERGING``
        """

    @staticmethod
    @_api_decl
    def mull(*args, **kwargs):
        """Long multiply: 32x32->64, output split into lo/hi register pair.

        Multiplies two 32-bit registers and produces 64-bit result split
        across two 32-bit destination registers.

        Two calling forms:

            vf.mull(dst_lo, dst_hi, src0, src1, pred)       # statement form
            dst_lo, dst_hi = vf.mull(src0, src1, pred)      # tuple assignment form

        Args:
            dst_lo: Low 32 bits of product
            dst_hi: High 32 bits of product
            src0: First source register (32-bit)
            src1: Second source register (32-bit)
            mask: Predicate mask register

        Kwargs:
            mode: ``pl.MergeMode.ZEROING`` (default) or ``pl.MergeMode.MERGING``
        """

    @staticmethod
    @_api_decl
    def addc(*args, **kwargs):
        """Add with carry (vaddcs): ``carry_out, dst = src0 + src1 + carry_in``

        Used for multi-word (e.g. 64-bit) arithmetic on 32-bit registers.
        Produces two outputs via tuple unpacking — the carry-out flag register
        (declared as a MaskReg) and the sum register (RegTensor)::

            carry_out, dst = vf.addc(src0, src1, carry_in, mask)

        Args:
            src0: First source register
            src1: Second source register
            carry_in: Input carry flag register (MaskReg)
            mask: Predicate mask register

        Returns:
            (carry_out, dst): carry-out flag register and the sum register

        Kwargs:
            mode: ``pl.MergeMode.ZEROING`` (default) or ``pl.MergeMode.MERGING``
        """

    @staticmethod
    @_api_decl
    def subc(*args, **kwargs):
        """Subtract with borrow (vsubcs): ``borrow_out, dst = src0 - src1 - borrow_in``

        Used for multi-word (e.g. 64-bit) arithmetic on 32-bit registers.
        Produces two outputs via tuple unpacking — the borrow-out flag register
        (declared as a MaskReg) and the difference register (RegTensor)::

            borrow_out, dst = vf.subc(src0, src1, borrow_in, mask)

        Args:
            src0: First source register
            src1: Second source register
            borrow_in: Input borrow flag register (MaskReg)
            mask: Predicate mask register

        Returns:
            (borrow_out, dst): borrow-out flag register and the difference register

        Kwargs:
            mode: ``pl.MergeMode.ZEROING`` (default) or ``pl.MergeMode.MERGING``
        """

    @staticmethod
    @_api_decl
    def load_unalign_init(*args, **kwargs):
        """Declare an unaligned register for load operations.

        Must be called before load_unalign_pre/load_unalign to allocate
        the alignment state register.

        Returns:
            Unaligned register handle
        """

    @staticmethod
    @_api_decl
    def load_unalign_pre(*args, **kwargs):
        """Setup unaligned load (vldas instruction).

        Initializes the alignment state for subsequent unaligned loads.

        Args:
            ureg: UnalignRegForLoad register (from load_unalign_init)
            src_ptr: Source UB pointer
        """

    @staticmethod
    @_api_decl
    def load_unalign(*args, **kwargs):
        """Unaligned load from UB to register (vldus instruction).

        Loads data from an unaligned UB address. Supports optional stride
        for POST_UPDATE mode.

        Args:
            dst: Destination register
            ureg: UnalignRegForLoad register
            src_ptr: Source UB pointer
            stride: Optional post-update stride in bytes
        """

    @staticmethod
    @_api_decl
    def scatter(*args, **kwargs):
        """Scatter store by index (vscatter instruction).

        Writes register elements to non-contiguous UB locations specified
        by an index register.

        Args:
            base_ptr: Base UB pointer
            src: Source register (data to scatter)
            index: Index register (destination offsets)
            mask: Predicate mask register
        """

    @staticmethod
    @_api_decl
    def unsqueeze(*args, **kwargs):
        """Unsqueeze mask bits into a register (vusqz instruction).

        Expands each mask bit into the corresponding register lane
        (1 for active, 0 for inactive).

        Args:
            dst: Destination register
            mask: Mask register to unsqueeze
        """

    @staticmethod
    @_api_decl
    def gatherb(*args, **kwargs):
        """Deprecated — merged into :func:`vf.gather`.

        Use ``vf.gather(src_ub, indices, mask,
        data_copy_mode=pl.DataCopyMode.DATA_BLOCK_LOAD)`` for the former
        datablock-granularity gather. This name is no longer accepted by the
        parser and is retained only as a migration pointer.
        """

    @staticmethod
    @_api_decl
    def get_mask_spr(*args, **kwargs):
        """Get mask from special purpose register (movp_b32/movp_b16).

        Reads the {MASK1, MASK0} SPR set by SetVectorMask and converts it
        to a MaskReg. This is the pypto equivalent of AscendC ``MoveMask<T>``.

        - ``width="b32"``: reads 64-bit MASK0, expands each bit to 4 bits (movp_b32)
        - ``width="b16"``: reads full 128-bit {MASK1, MASK0}, expands each bit to 2 bits (movp_b16)

        Kwargs:
            width: ``"b32"`` (default) or ``"b16"`` — selects mask width

        Returns:
            mask_reg with current SPR value
        """

    @staticmethod
    @_api_decl
    def mla(*args, **kwargs):
        """Multiply-add (3 source): ``dst[i] = src0[i] * src1[i] + src2[i]``

        Unlike mul_dst_add which accumulates into dst, mla takes a separate addend register.

        Args:
            dst: Destination register
            src0: First multiplicand register
            src1: Second multiplicand register
            src2: Addend register
            mask: Predicate mask register
        """

    @staticmethod
    @_api_decl
    def avg(*args, **kwargs):
        """Element-wise average: ``dst[i] = (src0[i] + src1[i]) / 2``

        Args:
            dst: Destination register
            src0: First source register
            src1: Second source register
            mask: Predicate mask register
        """

    @staticmethod
    @_api_decl
    def add3(*args, **kwargs):
        """Three-operand add: ``dst[i] = src0[i] + src1[i] + src2[i]``

        Args:
            dst: Destination register
            src0: First source register
            src1: Second source register
            src2: Third source register
            mask: Predicate mask register
        """

    @staticmethod
    @_api_decl
    def gather2(*args, **kwargs):
        """Gather with two-element stride (vgather2 instruction).

        Gathers pairs of elements from UB memory.

        Args:
            dst: Destination register
            src_ub: Source UB pointer
            index: Index register
            mask: Predicate mask register
        """

    @staticmethod
    @_api_decl
    def select_r(*args, **kwargs):
        """Select by register (reversed select): ``dst[i] = src_false[i] if mask[i] else src_true[i]``

        Reversed polarity compared to ``select``.

        Args:
            dst: Destination register
            src_true: Register selected when mask bit is 0
            src_false: Register selected when mask bit is 1
            mask: Predicate mask register
        """

    @staticmethod
    @_api_decl
    def log2(*args, **kwargs):
        """Base-2 logarithm: ``dst[i] = log2(src[i])``

        Synthesized as ``vln(src) * (1/ln(2))``.

        Args:
            dst: Destination register
            src: Source register (must be positive)
            mask: Predicate mask register

        Kwargs:
            mode: ``pl.MergeMode.ZEROING`` (default) or ``pl.MergeMode.MERGING``
        """

    @staticmethod
    @_api_decl
    def log10(*args, **kwargs):
        """Base-10 logarithm: ``dst[i] = log10(src[i])``

        Synthesized as ``vln(src) * (1/ln(10))``.

        Args:
            dst: Destination register
            src: Source register (must be positive)
            mask: Predicate mask register

        Kwargs:
            mode: ``pl.MergeMode.ZEROING`` (default) or ``pl.MergeMode.MERGING``
        """

    @staticmethod
    @_api_decl
    def muls_cast(*args, **kwargs):
        """Multiply by scalar then cast: ``dst(fp16) = cast(src(fp32) * scalar)``

        Fused operation combining vmuls and vcvt.

        Args:
            src: Source register (fp32)
            scalar: Scalar multiplier
            mask: Predicate mask register

        Kwargs:
            layout: ``pl.CastLayout.ZERO`` (even half, default) or
                ``pl.CastLayout.ONE`` (odd half) for the half-width result
        """

    @staticmethod
    @_api_decl
    def load(*args, **kwargs):
        """Unified load (vldas+vldus, matches AscendC Load interface).

        Simple aligned load from UB to register. Supports optional post-update.

        Args:
            dst: Destination register
            src_ptr: Source UB pointer
            stride: Post-update stride (optional, triggers POST_UPDATE mode)

        Kwargs:
            post_update: Enable post-update addressing
            repeat_stride: Stride for repeated loads
            count: Element count
        """

    @staticmethod
    @_api_decl
    def store(*args, **kwargs):
        """Unified store (vstus+vstas, matches AscendC Store interface).

        Simple store from register to UB. Supports optional post-update.

        Args:
            dst_ptr: Destination UB pointer
            src: Source register
            count: Element count (optional, defaults to 256/elem_bytes)

        Kwargs:
            post_update: Enable post-update addressing
            repeat_stride: Stride for repeated stores
        """

    @staticmethod
    @_api_decl
    def mask_and(*args, **kwargs):
        """Mask register bitwise AND: ``dst = src0 & src1`` (pand instruction).

        Args:
            dst: Destination mask register
            src0: First source mask register
            src1: Second source mask register
            mask: Predicate mask register
        """

    @staticmethod
    @_api_decl
    def mask_or(*args, **kwargs):
        """Mask register bitwise OR: ``dst = src0 | src1`` (por instruction).

        Args:
            dst: Destination mask register
            src0: First source mask register
            src1: Second source mask register
            mask: Predicate mask register
        """

    @staticmethod
    @_api_decl
    def mask_xor(*args, **kwargs):
        """Mask register bitwise XOR: ``dst = src0 ^ src1`` (pxor instruction).

        Args:
            dst: Destination mask register
            src0: First source mask register
            src1: Second source mask register
            mask: Predicate mask register
        """

    @staticmethod
    @_api_decl
    def mask_not(*args, **kwargs):
        """Mask register bitwise NOT: ``dst = ~src`` (pnot instruction).

        Args:
            dst: Destination mask register
            src: Source mask register
            mask: Predicate mask register
        """

    @staticmethod
    @_api_decl
    def mask_mov(*args, **kwargs):
        """Mask register copy: ``dst = src`` (pmov instruction).

        Args:
            dst: Destination mask register
            src: Source mask register
        """

    @staticmethod
    @_api_decl
    def mask_sel(*args, **kwargs):
        """Mask select: ``dst = src0 if pred else src1`` (psel instruction).

        Args:
            dst: Destination mask register
            src0: First source mask register
            src1: Second source mask register
            pred: Predicate mask register
        """

    @staticmethod
    @_api_decl
    def mask_pack(*args, **kwargs):
        """Mask pack: narrow mask granularity (ppack instruction).

        Args:
            dst: Destination mask register (narrower granularity)
            src: Source mask register
        """

    @staticmethod
    @_api_decl
    def mask_unpack(*args, **kwargs):
        """Mask unpack: widen mask granularity (punpack instruction).

        Args:
            dst: Destination mask register (wider granularity)
            src: Source mask register
        """

    @staticmethod
    @_api_decl
    def mask_interleave(*args, **kwargs):
        """Mask interleave (pintlv_b8/b16/b32 instruction).

        Interleaves bits from two source masks.

        Args:
            dst0: First destination mask register
            dst1: Second destination mask register
            src0: First source mask register
            src1: Second source mask register

        Kwargs:
            dtype: Data type controlling interleave granularity
        """

    @staticmethod
    @_api_decl
    def mask_deinterleave(*args, **kwargs):
        """Mask de-interleave (pdintlv_b8/b16/b32 instruction).

        Splits interleaved mask bits into even/odd halves.

        Args:
            dst0: First destination mask register (even bits)
            dst1: Second destination mask register (odd bits)
            src0: First source mask register
            src1: Second source mask register

        Kwargs:
            dtype: Data type controlling de-interleave granularity
        """

    @staticmethod
    @_api_decl
    def mask_load(*args, **kwargs):
        """Load a mask register from UB memory (assignment form).

        Fixed-offset form emits ``plds``; passing an AddrReg offset (from
        ``vf.create_addr_reg``) emits ``pld``::

            dst = vf.mask_load(src_ptr)             # plds (offset 0)
            dst = vf.mask_load(src_ptr, addr_reg)   # pld (AddrReg offset)

        Args:
            src_ptr: Source UB pointer
            offset: Optional AddrReg offset (routes to pld); mode ∈ NORM/US/DS

        Kwargs:
            mode: ``pl.MaskLoadDist`` value (NORM/US/DS, default US)
        """

    @staticmethod
    @_api_decl
    def mask_store(*args, **kwargs):
        """Store a mask register to UB memory.

        Fixed-offset form emits ``psts``; passing an AddrReg offset (from
        ``vf.create_addr_reg``) as the 4th arg emits ``pst``::

            vf.mask_store(src_mask, dst_ptr, mask)             # psts (offset 0)
            vf.mask_store(src_mask, dst_ptr, mask, addr_reg)   # pst (AddrReg offset)

        Args:
            src: Source mask register
            dst_ptr: Destination UB pointer
            mask: Predicate mask register
            offset: Optional AddrReg offset (routes to pst); dist ∈ NORM/PK

        Kwargs:
            dist: ``pl.MaskStoreDist.NORM`` or ``pl.MaskStoreDist.PK`` (default PK)
        """

    @staticmethod
    @_api_decl
    def mask_store_unalign(*args, **kwargs):
        """Store mask register to unaligned UB address (pstu instruction).

        Args:
            src: Source mask register
            dst_ptr: Destination UB pointer
            mask: Predicate mask register
        """

    @staticmethod
    @_api_decl
    def truncate(*args, **kwargs):
        """Truncate to integer (round toward zero): ``dst[i] = trunc(src[i])``

        Maps to hardware ``vtrc`` with ROUND_Z mode.

        Args:
            dst: Destination register
            src: Source register
            mask: Predicate mask register

        Kwargs:
            mode: ``pl.MergeMode.ZEROING`` (default) or ``pl.MergeMode.MERGING``
        """

    @staticmethod
    @_api_decl
    def mask_gen_with_reg_tensor(*args, **kwargs):
        """Generate mask from a register tensor bit at a given offset (movvp instruction).

        Converts a bit in a register element into a mask predicate.

        Args:
            src: Source register (uint16 or uint32)

        Kwargs:
            offset: Bit offset within the element

        Returns:
            mask_reg with generated mask
        """

    @staticmethod
    @_api_decl
    def create_addr_reg(*args, **kwargs):
        """Create an address offset register for aligned load/store (CreateAddrReg).

        Computes ``offset = index0 * stride0 + index1 * stride1 + ...`` and
        returns an ``AddrReg`` that can be passed to ``vf.load_align`` /
        ``vf.store_align`` as the offset parameter. Supports 1-4 loop axes
        (index/stride pairs).

        Usage::

            aReg = vf.create_addr_reg(i, 64, dtype=pl.DT_FP32)
            reg = vf.load_align(src_tile, aReg)
            vf.store_align(dst_tile, reg, preg, aReg)

        Args:
            index0: Loop axis 0 index (loop variable)
            stride0: Loop axis 0 stride in elements
            index1: Optional loop axis 1 index
            stride1: Optional loop axis 1 stride in elements
            index2: Optional loop axis 2 index
            stride2: Optional loop axis 2 stride in elements
            index3: Optional loop axis 3 index
            stride3: Optional loop axis 3 stride in elements

        Kwargs:
            dtype: Data type for template instantiation (default ``pl.DT_FP32``).
                Determines the element width: b8/b16/b32/b64.

        Returns:
            AddrReg handle for use as offset in load_align/store_align
        """

    @staticmethod
    @_api_decl
    def move(*args, **kwargs):
        """Move/copy register elements (vmov for RegTensor, pmov for MaskReg).

        For RegTensor: copies valid elements from src to dst; masked-out
        positions retain dst's original value (MODE_MERGING).

        For MaskReg: copies src bits to dst; with mask, only masked bits
        are copied.

        Usage::

            # RegTensor with mask
            vf.move(dst_reg, src_reg, mask)
            # RegTensor without mask
            vf.move(dst_reg, src_reg)
            # MaskReg with mask
            vf.move(dst_mask, src_mask, mask)
            # MaskReg without mask
            vf.move(dst_mask, src_mask)

        Args:
            dst: Destination register (RegTensor or MaskReg)
            src: Source register (same type as dst)
            mask: Optional mask register

        Kwargs:
            mode: ``pl.MergeMode.MERGING`` (default, only supported mode)
        """

# ===================================================================
# Section E: Debug ops
# ===================================================================


@_api_decl
def pto_assert(condition: bool, format_str: Optional[str] = None,
               *args, loc: bool = False) -> None:
    """Runtime assert: abort if condition is false, optionally print error message.

    Args:
        condition: Scalar boolean condition (dtype must be BOOL)
        format_str: Optional compile-time constant format string (printf-style)
        *args: Scalar values to print in the format string
        loc: Show source location in the error message
    """


@_api_decl
def printf(format_str: str, *args, loc: bool = False) -> None:
    """Print scalar values using a compile-time format string (printf-style).

    Supported conversions: ``%d``/``%i`` (signed int), ``%u`` (unsigned int),
    ``%x`` (hex), ``%f`` (float, FP32 only).

    Args:
        format_str: Compile-time constant format string
        *args: Scalar values (int, float, or ``pl.Scalar``)
        loc: Show source location
    """


@_api_decl
def dump_data(data: Union[Tensor, Tile], offsets: Optional[List[int]] = None,
              shapes: Optional[List[int]] = None, *,
              workspace: Optional[Tensor] = None,
              loc: bool = False) -> None:
    """Print Tensor or Tile contents for debugging.

    Automatically dispatches based on input type:
    - TensorType (GM Tensor): prints GM tensor data
    - TileType (on-chip Tile): prints tile data (Acc tiles require workspace)

    Args:
        data: Tensor or Tile to dump
        offsets: Optional window start offsets (per dimension); must pair with *shapes*
        shapes: Optional window shape (per dimension); must pair with *offsets*
        workspace: GM Tensor used as temporary space (only valid for Acc Tile inputs,
                   size >= tile_numel * sizeof(element_type))
        loc: Show source location
    """


@_api_decl
def trap() -> None:
    """Insert a trap instruction to unconditionally abort execution."""


# ===================================================================
# Section F: Scalar ops
# ===================================================================

@_api_decl
def min(lhs: Scalar, rhs: Scalar) -> Scalar:
    """Return the minimum of two scalars.

    Scalar-only operation for loop-bound calculations etc.
    For tile element-wise minimum, use ``pl.minimum``.

    Args:
        lhs: Left operand (scalar)
        rhs: Right operand (scalar)

    Returns:
        Scalar result.
    """


@_api_decl
def max(lhs: Scalar, rhs: Scalar) -> Scalar:
    """Return the maximum of two scalars.

    Scalar-only operation for loop-bound calculations etc.
    For tile element-wise maximum, use ``pl.maximum``.

    Args:
        lhs: Left operand (scalar)
        rhs: Right operand (scalar)

    Returns:
        Scalar result.
    """


# ===================================================================
# Section G: Tile min/max (element-wise + reduce overload)
#
# ``maximum`` / ``minimum`` are overloaded:
#   - Element-wise (no ``dim``): ``out = max(lhs, rhs)``
#       - ``maximum(out, tile_a, tile_b)``  -> tile-tile
#       - ``maximum(out, tile_a, scalar)``  -> tile-scalar
#   - Reduce (with ``dim``): reduction along the specified dimension
#       - ``maximum(out, src, tmp, dim=0)`` -> row-wise max (last axis)
#       - ``maximum(out, src, tmp, dim=1)`` -> column-wise max (first axis)
# ===================================================================

@_api_decl
def minimum(out: Tile, lhs: Tile, rhs: Union[Tile, Scalar], *,
            dim: Optional[int] = None) -> None:
    """Element-wise minimum or dimension-wise min reduction.

    Without ``dim`` (element-wise): ``out = min(lhs, rhs)``
        - ``minimum(out, tile_a, tile_b)`` -> tile-tile
        - ``minimum(out, tile_a, scalar)`` -> tile-scalar

    With ``dim`` (reduce): min reduction along the specified dimension.
        - ``minimum(out, src, tmp, dim=0)`` -> row-wise min (last axis)
        - ``minimum(out, src, tmp, dim=1)`` -> column-wise min (first axis)

    Args:
        out: Destination Tile
        lhs: Source Tile (or left operand in element-wise mode)
        rhs: Right operand — Tile/Scalar in element-wise mode, or workspace
             Tile in reduce mode
        dim: Reduction dimension — None=element-wise, 0=row, 1=column
    """


@_api_decl
def maximum(out: Tile, lhs: Tile, rhs: Union[Tile, Scalar], *,
            dim: Optional[int] = None) -> None:
    """Element-wise maximum or dimension-wise max reduction.

    Without ``dim`` (element-wise): ``out = max(lhs, rhs)``
        - ``maximum(out, tile_a, tile_b)`` -> tile-tile
        - ``maximum(out, tile_a, scalar)`` -> tile-scalar

    With ``dim`` (reduce): max reduction along the specified dimension.
        - ``maximum(out, src, tmp, dim=0)`` -> row-wise max (last axis)
        - ``maximum(out, src, tmp, dim=1)`` -> column-wise max (first axis)

    Args:
        out: Destination Tile
        lhs: Source Tile (or left operand in element-wise mode)
        rhs: Right operand — Tile/Scalar in element-wise mode, or workspace
             Tile in reduce mode
        dim: Reduction dimension — None=element-wise, 0=row, 1=column
    """


@_api_decl
def const(value: Union[int, float], dtype: DType) -> Scalar:
    """Create a typed compile-time constant.

    Args:
        value: Numeric literal (int or float)
        dtype: Data type, e.g. ``pl.DT_INT32``, ``pl.DT_FP16``
    """


# ===================================================================
# Section G: Control flow
# ===================================================================

def range(start: int, stop: Optional[int] = None, step: int = 1):
    """Loop iterator for ``for`` loops.

    Usage::

        for i in pl.range(10):          # 0..9
        for i in pl.range(0, 10, 2):    # 0, 2, 4, 6, 8
    """
    if stop is None:
        start, stop = 0, start
    return builtins.range(start, stop, step)


@contextmanager
def section_vector():
    """Context manager for a Vector-pipe section scope."""
    yield


@contextmanager
def section_cube():
    """Context manager for a Cube-pipe section scope."""
    yield


# ===================================================================
# Section H: Utility / system-level ops
# ===================================================================

@_api_decl
def get_block_idx() -> int:
    """Get the current block (AI Core) index."""


@_api_decl
def get_subblock_idx() -> int:
    """Get the current sub-block index (0 or 1)."""


@_api_decl
def get_block_num() -> int:
    """Get the total number of blocks."""


@_api_decl
def get_spr() -> int:
    """Read a special purpose register value (get_ar instruction).

    Currently only the AR register is supported. The AR register stores
    the total byte count of valid elements produced by Squeeze.

    ``get_ar()`` is an ``__aicore__`` instruction and cannot be used inside
    ``@pl.vector_function``. Call this in the ``@pl.jit`` kernel body.

    Returns:
        int64_t scalar value from the SPR
    """


@_api_decl
def make_tile_group(*, type: Any, addrs: list, mutex_ids: list) -> Any:
    """Create a rotating Tile group for multi-buffering with automatic mutex.

    Args:
        type: ``pl.TileType`` descriptor
        addrs: List of Tile addresses
        mutex_ids: List of mutex IDs for synchronization
    """
