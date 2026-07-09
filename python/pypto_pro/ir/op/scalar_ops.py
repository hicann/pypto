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

"""Scalar op parse handlers (min, max, const) and unified dispatchers."""

from __future__ import annotations

import ast

from pypto.pypto_impl import ir as _ir_core

from ._op_registry import op_impl


@op_impl("min")
def _parse_min(self, call: ast.Call):
    from pypto_pro.language.parser.diagnostics import InvalidOperationError
    call_span = self.span_tracker.get_span(call)
    if call.keywords:
        raise InvalidOperationError(
            "Scalar operation 'min' does not accept keyword arguments", span=call_span
        )
    if len(call.args) != 2:
        raise InvalidOperationError(
            f"Scalar binary operation 'min' requires exactly 2 arguments, got {len(call.args)}",
            span=call_span,
        )
    lhs = self.parse_expression(call.args[0])
    rhs = self.parse_expression(call.args[1])
    return _ir_core.min_(lhs, rhs, call_span)


@op_impl("max")
def _parse_max(self, call: ast.Call):
    from pypto_pro.language.parser.diagnostics import InvalidOperationError
    call_span = self.span_tracker.get_span(call)
    if call.keywords:
        raise InvalidOperationError(
            "Scalar operation 'max' does not accept keyword arguments", span=call_span
        )
    if len(call.args) != 2:
        raise InvalidOperationError(
            f"Scalar binary operation 'max' requires exactly 2 arguments, got {len(call.args)}",
            span=call_span,
        )
    lhs = self.parse_expression(call.args[0])
    rhs = self.parse_expression(call.args[1])
    return _ir_core.max_(lhs, rhs, call_span)


# ---------------------------------------------------------------------------
# Unified dispatch: add / sub / mul / div / minimum / maximum
#   3rd arg is Tile   -> block.add / block.sub / ... / block.minimum / block.maximum
#   3rd arg is Scalar -> block.adds / block.subs / ... / block.mins / block.maxs
# ---------------------------------------------------------------------------
def _make_binary_dispatch(op_name: str, scalar_op_name: str | None = None):
    scalar_op_name = scalar_op_name or (op_name + "s")

    def _dispatch(self, call: ast.Call):
        from pypto_pro.ir.op.block_ops import block_ir_op
        from pypto_pro.language.parser.diagnostics import InvalidOperationError
        call_span = self.span_tracker.get_span(call)
        if len(call.args) != 3:
            raise InvalidOperationError(
                f"Operation '{op_name}' requires exactly 3 arguments (out, lhs, rhs), got {len(call.args)}",
                span=call_span,
            )
        args = [self.parse_expression(arg) for arg in call.args]
        kwargs = self.parse_op_kwargs(call)
        if isinstance(args[2].type, _ir_core.TileType):
            target_op = block_ir_op(op_name)
        else:
            target_op = block_ir_op(scalar_op_name)
        return _ir_core.create_op_call(target_op, args, kwargs, call_span)

    return _dispatch


op_impl("add")(_make_binary_dispatch("add"))
op_impl("sub")(_make_binary_dispatch("sub"))
op_impl("mul")(_make_binary_dispatch("mul"))
op_impl("div")(_make_binary_dispatch("div"))
op_impl("minimum")(_make_binary_dispatch("minimum", "mins"))
op_impl("maximum")(_make_binary_dispatch("maximum", "maxs"))


@op_impl("const")
def _parse_typed_constant(self, call: ast.Call):
    from pypto_pro.language.parser.diagnostics import ParserSyntaxError
    span = self.span_tracker.get_span(call)

    if len(call.args) != 2:
        raise ParserSyntaxError(
            "pl.const() requires exactly 2 arguments: value and dtype",
            span=span,
            hint="Use pl.const(42, pl.DT_INT32) or pl.const(1.0, pl.DT_FP16)",
        )

    value_node = call.args[0]
    negate = False
    if isinstance(value_node, ast.UnaryOp) and isinstance(value_node.op, ast.USub):
        negate = True
        value_node = value_node.operand

    if not isinstance(value_node, ast.Constant) or not isinstance(value_node.value, (int, float)):
        raise ParserSyntaxError(
            "pl.const() first argument must be a numeric literal",
            span=span,
            hint="Use an int or float literal: pl.const(42, pl.DT_INT32)",
        )

    value = value_node.value
    if negate:
        value = -value

    dtype = self.type_resolver.resolve_dtype(call.args[1])

    if isinstance(value, float):
        return _ir_core.ConstFloat(value, dtype, span)
    return _ir_core.ConstInt(value, dtype, span)
