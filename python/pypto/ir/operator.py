#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) PyPTO Contributors.
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
import inspect
from .pypto_impl import ir as _ir
from .pypto_impl.ir import DataType


def _normalize_expr(
    value: int | float | _ir.Expr,
    span: _ir.Span | None = None,
    int_dtype: DataType = DataType.INDEX,
    float_dtype: DataType = DataType.FP32,
) -> _ir.Expr:
    """Convert Python values to IR expressions.

    Args:
        value: Python int/float or existing Expr
        span: Optional span for created constants
        int_dtype: Data type to use for integer constants (default: INDEX)
        float_dtype: Data type to use for float constants (default: FP32)

    Returns:
        IR expression node

    Raises:
        TypeError: If value is not int, float, or ir.Expr
    """
    if isinstance(value, _ir.Expr):
        return value

    actual_span = span if span is not None else _ir.Span.unknown()

    if isinstance(value, int):
        return _ir.ConstInt(value, int_dtype, actual_span)
    elif isinstance(value, float):
        return _ir.ConstFloat(value, float_dtype, actual_span)
    else:
        raise TypeError(f"Cannot convert {type(value)} to IR expression")


def _capture_call_span() -> _ir.Span:
    """Capture span from the caller's location.

    Goes back through the call stack to find the actual user code that invoked the operator.

    Returns:
        Span: Source location of the caller
    """
    frame = inspect.currentframe()
    if frame is not None and frame.f_back is not None:
        frame = frame.f_back.f_back

    if frame is not None:
        info = inspect.getframeinfo(frame)
        return _ir.Span(info.filename, info.lineno, -1)

    return _ir.Span.unknown()


def _make_binary_op(op_name: str):
    """Create a binary operator wrapper with span capture and normalization.

    Args:
        op_name: Name of the operator (e.g., "add")

    Returns:
        Wrapper function
    """

    def wrapper(self: _ir.Expr, other: int | float | _ir.Expr) -> _ir.Expr:
        span = _capture_call_span()
        other_expr = _normalize_expr(other, span)
        return getattr(_ir, op_name)(self, other_expr, span)

    return wrapper


def _make_reverse_binary_op(op_name: str):
    """Create a binary operator wrapper with span capture and normalization.

    Args:
        op_name: Name of the operator (e.g., "add")

    Returns:
        Wrapper function
    """

    def wrapper(self: _ir.Expr, other: int | float | _ir.Expr) -> _ir.Expr:
        span = _capture_call_span()
        other_expr = _normalize_expr(other, span)
        return getattr(_ir, op_name)(other_expr, self, span)

    return wrapper


def _make_unary_op(op_name: str):
    """Create a unary operator wrapper with span capture.

    Args:
        op_name: Name of the operator (e.g., "neg")

    Returns:
        Wrapper function
    """

    def wrapper(self) -> _ir.Expr:
        span = _capture_call_span()
        return getattr(_ir, op_name)(self, span)

    return wrapper


def _patch_operators():
    """Patch Expr and Var classes with operator overloads."""

    # Binary operators for Expr
    _ir.Expr.__add__ = _make_binary_op("add")
    _ir.Expr.__sub__ = _make_binary_op("sub")
    _ir.Expr.__mul__ = _make_binary_op("mul")
    _ir.Expr.__truediv__ = _make_binary_op("truediv")
    _ir.Expr.__floordiv__ = _make_binary_op("floordiv")
    _ir.Expr.__mod__ = _make_binary_op("mod")
    _ir.Expr.__pow__ = _make_binary_op("pow")

    # Comparison operators for Expr
    _ir.Expr.__eq__ = _make_binary_op("eq")
    _ir.Expr.__ne__ = _make_binary_op("ne")
    _ir.Expr.__lt__ = _make_binary_op("lt")
    _ir.Expr.__le__ = _make_binary_op("le")
    _ir.Expr.__gt__ = _make_binary_op("gt")
    _ir.Expr.__ge__ = _make_binary_op("ge")

    # Bitwise operators for Expr
    _ir.Expr.__and__ = _make_binary_op("bit_and")
    _ir.Expr.__or__ = _make_binary_op("bit_or")
    _ir.Expr.__xor__ = _make_binary_op("bit_xor")
    _ir.Expr.__lshift__ = _make_binary_op("bit_shift_left")
    _ir.Expr.__rshift__ = _make_binary_op("bit_shift_right")

    # Unary operators for Expr
    _ir.Expr.__neg__ = _make_unary_op("neg")
    _ir.Expr.__invert__ = _make_unary_op("bit_not")

    # Reverse operators for Expr (when Expr is on right side)
    _ir.Expr.__radd__ = _make_reverse_binary_op("add")
    _ir.Expr.__rmul__ = _make_reverse_binary_op("mul")
    _ir.Expr.__rsub__ = _make_reverse_binary_op("sub")
    _ir.Expr.__rtruediv__ = _make_reverse_binary_op("truediv")
    _ir.Expr.__rfloordiv__ = _make_reverse_binary_op("floordiv")
    _ir.Expr.__rmod__ = _make_reverse_binary_op("mod")
    _ir.Expr.__rpow__ = _make_reverse_binary_op("pow")
    _ir.Expr.__rand__ = _make_reverse_binary_op("bit_and")
    _ir.Expr.__ror__ = _make_reverse_binary_op("bit_or")
    _ir.Expr.__rxor__ = _make_reverse_binary_op("bit_xor")
    _ir.Expr.__rlshift__ = _make_reverse_binary_op("bit_shift_left")
    _ir.Expr.__rrshift__ = _make_reverse_binary_op("bit_shift_right")


_patch_operators()
