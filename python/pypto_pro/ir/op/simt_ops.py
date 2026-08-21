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

"""IR builders and parse handlers used by the PyPTO Pro SIMT frontend."""

from __future__ import annotations

import ast
from collections.abc import Sequence
from typing import Any

from pypto.pypto_impl import ir as _ir_core
from pypto.pypto_impl.ir import Call, Expr, Function, Span

from .._utils import _get_span_or_capture, _normalize_expr
from ._op_registry import op_impl

_DIM3_FIELDS = ("x", "y", "z")


def _make_dim3_components(op_name: str, span: Span) -> list[Call]:
    return [_ir_core.create_op_call(op_name, [], {"axis": axis}, span) for axis in range(len(_DIM3_FIELDS))]


def _make_dim3_context(op_name: str, span: Span | None = None) -> Expr:
    """Build a three-component SIMT context tuple."""
    actual_span = _get_span_or_capture(span)
    return _ir_core.MakeTuple(_make_dim3_components(op_name, actual_span), actual_span)


def linear_thread_idx(span: Span | None = None) -> Call:
    """Return the x-major flattened thread index within the current block."""
    actual_span = _get_span_or_capture(span)
    return _ir_core.create_op_call("simt.linear_thread_idx", [], {}, actual_span)


def warp_size(span: Span | None = None) -> Call:
    """Return the target SIMT warp size."""
    actual_span = _get_span_or_capture(span)
    return _ir_core.create_op_call("simt.warp_size", [], {}, actual_span)


def launch(
    callee: Function,
    *,
    threads: int | Expr | tuple[int | Expr, ...],
    args: Sequence[Expr],
    span: Span | None = None,
) -> Call:
    """Build a SIMT launch call.

    Normal DSL source is parsed specially so the callee Function is retained in
    the enclosing Program. This builder is primarily useful for direct IR tests.
    """
    actual_span = _get_span_or_capture(span)
    thread_dims = list(threads) if isinstance(threads, tuple) else [threads]
    if not 1 <= len(thread_dims) <= 3:
        raise ValueError("SIMT launch threads must contain one to three dimensions")
    thread_dims.extend([1] * (3 - len(thread_dims)))
    normalized_dims = [_normalize_expr(dim, actual_span) for dim in thread_dims]
    return _ir_core.create_op_call(
        "simt.launch",
        [*normalized_dims, *args],
        {"callee": callee.name, "max_threads": callee.get_attr("max_threads")},
        actual_span,
    )


def _validate_simt_body_op(parser: Any, call: ast.Call, _kwargs: dict[str, Any]) -> None:
    """Validate the common source constraints of a SIMT body operation."""
    from pypto_pro.language.parser.diagnostics import ParserSyntaxError

    span = parser.span_tracker.get_span(call)
    op_name = parser._extract_op_name(call.func)
    if parser._current_func_type not in (_ir_core.FunctionType.SimtVF, _ir_core.FunctionType.SimtCallee):
        raise ParserSyntaxError(
            f"pl.{op_name}() can only be used inside a SIMT function",
            span=span,
            hint="Move SIMT-context-dependent logic into @pl.simt.function.",
        )
    if call.args:
        raise ParserSyntaxError(f"pl.{op_name}() does not accept positional arguments", span=span)
    if call.keywords:
        raise ParserSyntaxError(f"pl.{op_name}() does not accept keyword arguments", span=span)


def _parse_dim3_context(parser: Any, call: ast.Call, op_name: str) -> Expr:
    """Parse a SIMT context query as an IR named tuple."""
    _validate_simt_body_op(parser, call, {})
    span = parser.span_tracker.get_span(call)
    return parser.make_named_tuple(_make_dim3_components(op_name, span), _DIM3_FIELDS, span)


@op_impl("simt.thread_idx")
def _parse_thread_idx(parser: Any, call: ast.Call) -> Expr:
    return _parse_dim3_context(parser, call, "simt.thread_idx")


@op_impl("simt.block_dim")
def _parse_block_dim(parser: Any, call: ast.Call) -> Expr:
    return _parse_dim3_context(parser, call, "simt.block_dim")


@op_impl("simt.block_idx")
def _parse_block_idx(parser: Any, call: ast.Call) -> Expr:
    return _parse_dim3_context(parser, call, "simt.block_idx")


@op_impl("simt.grid_dim")
def _parse_grid_dim(parser: Any, call: ast.Call) -> Expr:
    return _parse_dim3_context(parser, call, "simt.grid_dim")


@op_impl("simt.linear_thread_idx")
def _parse_linear_thread_idx(parser: Any, call: ast.Call) -> Expr:
    _validate_simt_body_op(parser, call, {})
    span = parser.span_tracker.get_span(call)
    return linear_thread_idx(span)


@op_impl("simt.warp_size")
def _parse_warp_size(parser: Any, call: ast.Call) -> Expr:
    _validate_simt_body_op(parser, call, {})
    span = parser.span_tracker.get_span(call)
    return warp_size(span)


@op_impl("simt.launch")
def _parse_simt_launch(parser: Any, call: ast.Call) -> Expr:
    """Parse ``pl.simt.launch(callee, threads=..., args=(...))``."""
    from pypto_pro.language.parser.diagnostics import ParserSyntaxError, ParserTypeError

    span = parser.span_tracker.get_span(call)
    if parser._current_func_type in (_ir_core.FunctionType.SimtVF, _ir_core.FunctionType.SimtCallee):
        raise ParserSyntaxError("Nested pl.simt.launch() is not supported", span=span)
    if parser.target != _ir_core.SectionKind.Vector:
        raise ParserSyntaxError(
            "pl.simt.launch() must appear inside 'with pl.section_vector():'",
            span=span,
            hint="SIMT vector functions execute on AIV and cannot be launched from Cube or shared kernel code.",
        )
    if len(call.args) != 1 or not isinstance(call.args[0], ast.Name):
        raise ParserSyntaxError(
            "pl.simt.launch() requires one SIMT function name as its positional argument",
            span=span,
            hint="Use: pl.simt.launch(my_simt_func, threads=256, args=(out, n))",
        )

    keyword_nodes: dict[str, ast.expr] = {}
    for keyword in call.keywords:
        if keyword.arg is None or keyword.arg not in ("threads", "args") or keyword.arg in keyword_nodes:
            raise ParserSyntaxError(
                "pl.simt.launch() accepts exactly the 'threads' and 'args' keyword arguments",
                span=span,
                hint="Use: pl.simt.launch(my_simt_func, threads=256, args=(out, n))",
            )
        keyword_nodes[keyword.arg] = keyword.value
    if set(keyword_nodes) != {"threads", "args"}:
        raise ParserSyntaxError(
            "pl.simt.launch() requires both threads= and args=",
            span=span,
            hint="Use: pl.simt.launch(my_simt_func, threads=256, args=(out, n))",
        )

    from pypto_pro.language.parser.decorator import get_simt_max_threads, is_simt_function

    local_name = call.args[0].id
    callee_template = parser.expr_evaluator.closure_vars.get(local_name)
    if (
        not callable(callee_template)
        or not is_simt_function(callee_template)
        or get_simt_max_threads(callee_template) is None
    ):
        raise ParserTypeError(
            f"'{local_name}' is not a launchable @pl.simt.function",
            span=parser.span_tracker.get_span(call.args[0]),
        )

    threads_node = keyword_nodes["threads"]
    if isinstance(threads_node, ast.Tuple):
        if not 1 <= len(threads_node.elts) <= 3:
            raise ParserSyntaxError(
                "pl.simt.launch() threads tuple must contain one to three dimensions",
                span=parser.span_tracker.get_span(threads_node),
                hint="Use threads=N, threads=(x, y), or threads=(x, y, z).",
            )
        thread_dims = list(threads_node.elts)
    else:
        thread_dims = [threads_node]
    thread_dims = [parser.parse_expression(dim) for dim in thread_dims]

    args_node = keyword_nodes["args"]
    if not isinstance(args_node, ast.Tuple):
        raise ParserSyntaxError(
            "pl.simt.launch() args must be a tuple",
            span=parser.span_tracker.get_span(args_node),
            hint="Use args=(out, n); remember the trailing comma for a one-element tuple.",
        )
    launch_args = [parser.parse_expression(arg) for arg in args_node.elts]
    callee = parser._instantiate_simt_function(local_name, callee_template, launch_args, args_node.elts, span)
    parser._validate_simt_function_arguments(callee, launch_args, args_node.elts, span)
    try:
        return launch(callee, threads=tuple(thread_dims), args=launch_args, span=span)
    except RuntimeError as error:
        raise ParserTypeError(str(error), span=span) from error
