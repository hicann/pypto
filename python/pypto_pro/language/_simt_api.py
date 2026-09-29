#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Python API declarations for the PyPTO Pro SIMT namespace (``pl.simt.*``).

These declarations exist so that:
- IDE "Go to Definition" works for every ``pl.simt.xxx`` call
- Python catches typos at import time
- Type checkers can validate argument types
- Docstrings document the user-facing calling convention

None of these functions are meant to be called at runtime.  Inside a PyPTO
kernel the AST parser intercepts every ``pl.simt.xxx`` call before Python executes
it.  Outside a kernel, calling a declaration raises ``RuntimeError``.
"""

from __future__ import annotations

from typing import Any, Union

from pypto.ir import RoundMode

from ._api import DType, Scalar, _api_decl


class Simt:
    """SIMT namespace (``pl.simt.*``).

    Provides SIMT context queries, synchronization, scalar operations, and atomics.
    """

    @staticmethod
    @_api_decl
    def thread_idx() -> Any:
        """Return the current thread coordinates within its thread block."""

    @staticmethod
    @_api_decl
    def block_dim() -> Any:
        """Return the dimensions of the current thread block."""

    @staticmethod
    @_api_decl
    def block_idx() -> Any:
        """Return the current block coordinates within the outer kernel grid."""

    @staticmethod
    @_api_decl
    def grid_dim() -> Any:
        """Return the dimensions of the outer kernel grid."""

    @staticmethod
    @_api_decl
    def linear_thread_idx() -> Scalar:
        """Return the x-major flattened thread index within the current block."""

    @staticmethod
    @_api_decl
    def warp_size() -> int:
        """Return the target SIMT warp size."""

    @staticmethod
    @_api_decl
    def syncthreads() -> None:
        """Synchronize all threads in the current block.

        The barrier must be reached uniformly by every thread in the block. It
        cannot be placed inside runtime ``if``, ``for``, or ``while`` control flow.
        This no-return operation must be used as a standalone statement.
        """

    @staticmethod
    @_api_decl
    def threadfence_block() -> None:
        """Order this thread's memory operations for threads in the current block.

        This memory fence does not wait for other threads and may be used in
        runtime control flow. It must be used as a standalone statement.
        """

    @staticmethod
    @_api_decl
    def threadfence() -> None:
        """Order this thread's memory operations with device-wide visibility.

        This memory fence does not wait for other threads and may be used in
        runtime control flow. It must be used as a standalone statement.
        """

    @staticmethod
    @_api_decl
    def lane_id() -> int:
        """Return the current lane index in the warp as an INT32 Scalar."""

    @staticmethod
    @_api_decl
    def lanemask_eq() -> int:
        """Return an INT32 mask containing only the current lane bit."""

    @staticmethod
    @_api_decl
    def lanemask_le() -> int:
        """Return an INT32 mask containing lanes up to and including the current lane."""

    @staticmethod
    @_api_decl
    def lanemask_lt() -> int:
        """Return an INT32 mask containing lanes below the current lane."""

    @staticmethod
    @_api_decl
    def lanemask_ge() -> int:
        """Return an INT32 mask containing lanes at or above the current lane."""

    @staticmethod
    @_api_decl
    def lanemask_gt() -> int:
        """Return an INT32 mask containing lanes above the current lane."""

    @staticmethod
    @_api_decl
    def warp_all(predicate: int) -> int:
        """Return INT32 one when predicate is true for every active lane, otherwise zero."""

    @staticmethod
    @_api_decl
    def warp_any(predicate: int) -> int:
        """Return INT32 one when predicate is true for any active lane, otherwise zero."""

    @staticmethod
    @_api_decl
    def warp_ballot(predicate: int) -> int:
        """Return a UINT32 bit mask of active lanes whose predicate is true."""

    @staticmethod
    @_api_decl
    def warp_active_mask() -> int:
        """Return a UINT32 bit mask of the currently active lanes."""

    @staticmethod
    @_api_decl
    def warp_shfl(value: Union[int, float], src_lane: int, width: int = 32) -> Scalar:
        """Read value from src_lane in the current logical warp subgroup, preserving value dtype."""

    @staticmethod
    @_api_decl
    def warp_shfl_up(value: Union[int, float], delta: int, width: int = 32) -> Scalar:
        """Read value from a lower lane by delta within the subgroup, preserving value dtype."""

    @staticmethod
    @_api_decl
    def warp_shfl_down(value: Union[int, float], delta: int, width: int = 32) -> Scalar:
        """Read value from a higher lane by delta within the subgroup, preserving value dtype."""

    @staticmethod
    @_api_decl
    def warp_shfl_xor(value: Union[int, float], lane_mask: int, width: int = 32) -> Scalar:
        """Read value from the lane selected by XOR with lane_mask, preserving value dtype."""

    @staticmethod
    @_api_decl
    def warp_reduce_add(value: Union[int, float]) -> Scalar:
        """Return the same-dtype sum of value across the active lanes in the warp."""

    @staticmethod
    @_api_decl
    def warp_reduce_max(value: Union[int, float]) -> Scalar:
        """Return the same-dtype maximum value across the active lanes in the warp."""

    @staticmethod
    @_api_decl
    def warp_reduce_min(value: Union[int, float]) -> Scalar:
        """Return the same-dtype minimum value across the active lanes in the warp."""

    @staticmethod
    @_api_decl
    def cast(
        value: Union[int, float],
        dtype: DType,
        *,
        mode: RoundMode = RoundMode.CAST_NONE,
    ) -> Scalar:
        """Convert one scalar expression to ``dtype`` inside a SIMT function.

        ``mode`` controls rounding when the target dtype cannot represent ``value``
        exactly. Low-precision floating-point conversions added beyond the default
        paths require an explicit supported mode. ``CAST_ODD`` is supported only
        for FP32-to-FP16 conversion. Saturation mode selection is not yet exposed.
        """

    @staticmethod
    @_api_decl
    def bitcast(value: Union[int, float], dtype: DType) -> Scalar:
        """Reinterpret one scalar bit pattern as a supported equal-width dtype.

        Unlike :meth:`cast`, this operation performs no numerical conversion,
        rounding, or saturation. Only the explicitly supported FP16/BF16 and
        FP32 integer bit-reinterpretation pairs are accepted.
        """

    @staticmethod
    @_api_decl
    def abs(value: Union[int, float]) -> Scalar:
        """Return the absolute value of a FP16, BF16, FP32, or INT64 Scalar."""

    @staticmethod
    @_api_decl
    def min(lhs: Union[int, float], rhs: Union[int, float]) -> Scalar:
        """Return the minimum of two same-dtype floating-point or integer Scalars."""

    @staticmethod
    @_api_decl
    def max(lhs: Union[int, float], rhs: Union[int, float]) -> Scalar:
        """Return the maximum of two same-dtype floating-point or integer Scalars."""

    @staticmethod
    @_api_decl
    def sqrt(value: Union[int, float]) -> Scalar:
        """Return the square root of a FP16, BF16, or FP32 Scalar."""

    @staticmethod
    @_api_decl
    def rsqrt(value: Union[int, float]) -> Scalar:
        """Return the reciprocal square root of a FP16, BF16, or FP32 Scalar."""

    @staticmethod
    @_api_decl
    def exp(value: Union[int, float]) -> Scalar:
        """Return e raised to a FP16, BF16, or FP32 Scalar."""

    @staticmethod
    @_api_decl
    def exp2(value: Union[int, float]) -> Scalar:
        """Return two raised to a FP16, BF16, or FP32 Scalar."""

    @staticmethod
    @_api_decl
    def log(value: Union[int, float]) -> Scalar:
        """Return the natural logarithm of a FP16, BF16, or FP32 Scalar."""

    @staticmethod
    @_api_decl
    def log2(value: Union[int, float]) -> Scalar:
        """Return the base-two logarithm of a FP16, BF16, or FP32 Scalar."""

    @staticmethod
    @_api_decl
    def log1p(value: Union[int, float]) -> Scalar:
        """Return the FP32 natural logarithm of one plus ``value``."""

    @staticmethod
    @_api_decl
    def sin(value: Union[int, float]) -> Scalar:
        """Return the sine of a FP16, BF16, or FP32 Scalar."""

    @staticmethod
    @_api_decl
    def cos(value: Union[int, float]) -> Scalar:
        """Return the cosine of a FP16, BF16, or FP32 Scalar."""

    @staticmethod
    @_api_decl
    def tanh(value: Union[int, float]) -> Scalar:
        """Return the hyperbolic tangent of a FP16, BF16, or FP32 Scalar."""

    @staticmethod
    @_api_decl
    def rint(value: Union[int, float]) -> Scalar:
        """Round a floating-point Scalar to the nearest integer value."""

    @staticmethod
    @_api_decl
    def round(value: Union[int, float]) -> Scalar:
        """Round a floating-point Scalar halfway away from zero."""

    @staticmethod
    @_api_decl
    def floor(value: Union[int, float]) -> Scalar:
        """Round a floating-point Scalar down to an integer value."""

    @staticmethod
    @_api_decl
    def ceil(value: Union[int, float]) -> Scalar:
        """Round a floating-point Scalar up to an integer value."""

    @staticmethod
    @_api_decl
    def trunc(value: Union[int, float]) -> Scalar:
        """Round a floating-point Scalar toward zero to an integer value."""

    @staticmethod
    @_api_decl
    def isnan(value: Union[int, float]) -> Scalar:
        """Return a BOOL Scalar indicating whether a floating-point Scalar is NaN."""

    @staticmethod
    @_api_decl
    def isinf(value: Union[int, float]) -> Scalar:
        """Return a BOOL Scalar indicating whether a floating-point Scalar is infinite."""

    @staticmethod
    @_api_decl
    def isfinite(value: Union[int, float]) -> Scalar:
        """Test whether an FP16 or FP32 Scalar is finite, returning BOOL."""

    @staticmethod
    @_api_decl
    def popcount(value: Union[int, float]) -> Scalar:
        """Count the set bits in a UINT32 or UINT64 Scalar, returning INT32."""

    @staticmethod
    @_api_decl
    def mul_hi(lhs: Union[int, float], rhs: Union[int, float]) -> Scalar:
        """Return the high half of the full product of two same-dtype integers.

        Supports INT32, UINT32, INT64, and UINT64; the result has the input dtype.
        """

    @staticmethod
    @_api_decl
    def fmod(lhs: Union[int, float], rhs: Union[int, float]) -> Scalar:
        """Return the FP32 remainder with a quotient truncated toward zero.

        Both operands must be FP32. The result retains the dividend's sign,
        including signed zero. A zero divisor, infinite dividend, or NaN operand
        produces NaN; a finite dividend modulo infinity returns the dividend.
        """

    @staticmethod
    @_api_decl
    def fma(lhs: Union[int, float], rhs: Union[int, float], addend: Union[int, float]) -> Scalar:
        """Fused-multiply-add three same-dtype FP16, BF16, or FP32 Scalars."""

    @staticmethod
    @_api_decl
    def atomic_add(target: Union[int, float], value: Union[int, float]) -> Scalar | None:
        """Atomically add ``value`` to one Tile or Tensor element.

        ``target`` must be written directly as a subscript expression such as
        ``tile[row, col]`` or ``tensor[index]``. FP16/BF16 targets return no value;
        other supported dtypes return the element value observed before the update.
        """

    @staticmethod
    @_api_decl
    def atomic_sub(target: Union[int, float], value: Union[int, float]) -> Scalar:
        """Atomically subtract ``value`` from one Tile or Tensor element and return its old value."""

    @staticmethod
    @_api_decl
    def atomic_exch(target: Union[int, float], value: Union[int, float]) -> Scalar:
        """Atomically replace one Tile or Tensor element and return its old value."""

    @staticmethod
    @_api_decl
    def atomic_max(target: Union[int, float], value: Union[int, float]) -> Scalar | None:
        """Atomically update an element with its maximum; FP16/BF16 return no value."""

    @staticmethod
    @_api_decl
    def atomic_min(target: Union[int, float], value: Union[int, float]) -> Scalar | None:
        """Atomically update an element with its minimum; FP16/BF16 return no value."""

    @staticmethod
    @_api_decl
    def atomic_inc(target: Union[int, float], limit: Union[int, float]) -> Scalar:
        """Atomically increment and wrap one unsigned counter element, returning its old value."""

    @staticmethod
    @_api_decl
    def atomic_dec(target: Union[int, float], limit: Union[int, float]) -> Scalar:
        """Atomically decrement and wrap one unsigned counter element, returning its old value."""

    @staticmethod
    @_api_decl
    def atomic_cas(target: Union[int, float], compare: Union[int, float], value: Union[int, float]) -> Scalar:
        """Atomically compare and exchange one Tile or Tensor element, returning its old value."""

    @staticmethod
    @_api_decl
    def atomic_and(target: Union[int, float], value: Union[int, float]) -> Scalar:
        """Atomically apply bitwise AND to one Tile or Tensor element and return its old value."""

    @staticmethod
    @_api_decl
    def atomic_or(target: Union[int, float], value: Union[int, float]) -> Scalar:
        """Atomically apply bitwise OR to one Tile or Tensor element and return its old value."""

    @staticmethod
    @_api_decl
    def atomic_xor(target: Union[int, float], value: Union[int, float]) -> Scalar:
        """Atomically apply bitwise XOR to one Tile or Tensor element and return its old value."""
