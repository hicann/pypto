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

"""
Internal kernel representation used by the PyPTO Pro JIT.

Public kernels are defined with @pl.jit. KernelDef stores the captured source
and parser state for one specialization after the JIT frontend has resolved its
compile-time configuration.
"""

from __future__ import annotations

__all__ = ["KernelDef"]


import ast
import inspect
import textwrap
from typing import Any, Callable, TypeVar

from pypto.pypto_impl import ir
from pypto_pro.language.parser._ast_parser import ASTParser
from pypto_pro.language.parser.diagnostics import ParserError, ParserSyntaxError


def _calculate_col_offset(source_lines: list[str]) -> int:
    """Calculate the column offset (indentation) of the first non-empty line.

    This is needed because ast.parse() requires code starting at column 0,
    but we need to report errors at the correct column in the original file.

    Args:
        source_lines: List of source code lines

    Returns:
        Column offset (number of leading spaces/tabs in first non-empty line)
    """
    for line in source_lines:
        if line.strip():  # Skip empty lines
            return len(line) - len(line.lstrip())
    return 0


def _parse_ast_tree(source_code: str, entity_type: str) -> ast.AST:
    """Parse source code into an AST tree with proper error handling.

    Args:
        source_code: Python source code to parse
        entity_type: Type of entity being parsed ("function" or "class") for error messages

    Returns:
        Parsed AST tree

    Raises:
        ParserSyntaxError: If the source code has syntax errors
    """
    try:
        return ast.parse(source_code)
    except SyntaxError as e:
        raise ParserSyntaxError(
            f"Failed to parse {entity_type} source: {e.msg}",
            hint=f"Check for Python syntax errors in your {entity_type}",
        ) from e


TypeASTNode = TypeVar("TypeASTNode", ast.FunctionDef, ast.ClassDef)


def _find_ast_node(tree: ast.AST, node_type: type[TypeASTNode], name: str, entity_type: str) -> TypeASTNode:
    """Find a specific AST node by type and name.

    Args:
        tree: AST tree to search
        node_type: Type of AST node to find (ast.FunctionDef or ast.ClassDef)
        name: Name of the node to find
        entity_type: Type of entity for error messages ("function" or "class")

    Returns:
        Found AST node

    Raises:
        ParserSyntaxError: If the node cannot be found
    """
    for node in ast.walk(tree):
        if isinstance(node, node_type) and node.name == name:
            return node

    raise ParserSyntaxError(
        f"Could not find {entity_type} definition for {name}",
        hint=f"Ensure the {entity_type} is properly defined",
    )


def _attach_source_lines_to_error(error: ParserError, source_file: str, source_lines_raw: list[str]) -> None:
    """Attach source lines to a ParserError if not already present.

    Args:
        error: ParserError to attach source lines to
        source_file: Path to the source file
        source_lines_raw: Raw source lines as fallback
    """
    if error.source_lines is None:
        # Use the span's filename if it differs (e.g., error in an inline function)
        target_file = source_file
        if error.span and isinstance(error.span, dict):
            span_file = error.span.get("filename")
            if span_file and span_file != source_file:
                target_file = span_file
        try:
            with open(target_file, encoding="utf-8") as f:
                error.source_lines = f.read().split("\n")
        except Exception:
            # Fallback to the raw source lines if we can't read the file
            error.source_lines = source_lines_raw


def extract_func_source_info(f: Callable):
    """Extract source file, lines, offsets, and parsed func_def from a function.

    Returns:
        tuple of (source_file, source_lines, source_lines_raw, line_offset,
                  col_offset, func_def)
    """
    source_file = inspect.getfile(f)
    source_lines_raw, starting_line = inspect.getsourcelines(f)
    source_code = "".join(source_lines_raw)
    col_offset = _calculate_col_offset(source_lines_raw)
    source_code = textwrap.dedent(source_code)
    source_lines = source_code.split("\n")
    line_offset = starting_line - 1

    try:
        tree = _parse_ast_tree(source_code, "function")
        func_def = _find_ast_node(tree, ast.FunctionDef, f.__name__, "function")
    except ParserError as e:
        _attach_source_lines_to_error(e, source_file, source_lines_raw)
        raise

    return (source_file, source_lines, source_lines_raw, line_offset, col_offset, func_def)


class KernelDef:
    """Lazy kernel definition — captures source/AST/closure at decoration time,
    defers AST parsing to compile time.

    The JIT codegen path parses the kernel once for each required target.

    Args:
        func: Original Python function.
        source_file: Path to the source file.
        source_lines: Dedented source lines for the parser.
        source_lines_raw: Raw (non-dedented) source lines for error reporting.
        line_offset: Line number offset in the original file.
        col_offset: Column indentation offset.
        func_def: AST FunctionDef node.
        closure_vars: Captured caller scope for name resolution.
        name: Optional program name.
        func_type: IR function type (Opaque, InCore, Helper).
        strict_ssa: Whether to enforce SSA.
        meta_data: Optional metadata.
        auto_mutex: Whether to enable automatic mutex lock/unlock insertion.
    """

    def __init__(
        self,
        func: Callable,
        source_file: str,
        source_lines: list[str],
        source_lines_raw: list[str],
        line_offset: int,
        col_offset: int,
        func_def: ast.FunctionDef,
        closure_vars: dict[str, Any],
        name: str | None,
        func_type: ir.FunctionType,
        strict_ssa: bool,
        meta_data: Any,
        auto_mutex: bool = True,
        pipeline=None,
        tilingkey_consts: dict[str, int] | None = None,
        datatype_consts: dict[str, Any] | None = None,
    ) -> None:
        self._func = func
        self._source_file = source_file
        self._source_lines = source_lines
        self._source_lines_raw = source_lines_raw
        self._line_offset = line_offset
        self._col_offset = col_offset
        self._func_def = func_def
        self._closure_vars = closure_vars
        self._name = name
        self._func_type = func_type
        self._strict_ssa = strict_ssa
        self._auto_mutex = auto_mutex
        self._pipeline = pipeline
        self._meta_data = meta_data
        self._tilingkey_consts = tilingkey_consts
        self._datatype_consts = datatype_consts

    @property
    def func_def(self) -> ast.FunctionDef:
        return self._func_def

    @property
    def closure_vars(self) -> dict[str, Any]:
        return self._closure_vars

    @property
    def func_name(self) -> str:
        return self._func.__name__

    def parse_target_program(
        self,
        target: ir.SectionKind,
        bound_signature=None,
    ) -> tuple[ir.Program, bool]:
        """Parse a fresh target Program and report whether its target section was matched."""
        program_name = self._name if self._name is not None else self._func.__name__

        try:
            # The Program owns one IRDebugInfo; create it here and share it with the parser
            # so field names land in the table the Program carries.
            debug_info = ir.IRDebugInfo()
            parser = ASTParser(
                self._source_file,
                self._source_lines,
                target,
                self._line_offset,
                self._col_offset,
                strict_ssa=self._strict_ssa,
                closure_vars=self._closure_vars,
                auto_mutex=self._auto_mutex,
                debug_info=debug_info,
                tilingkey_consts=self._tilingkey_consts,
                datatype_consts=self._datatype_consts,
                bound_signature=bound_signature,
                # Kernels use a void ABI: they may early-return, but cannot return values.
                void_return_only=True,
                void_return_context="@pl.jit",
                allow_early_return=True,
            )

            try:
                ir_func = parser.parse_function(self._func_def, func_type=self._func_type)
            except ParserError:
                raise
            except Exception as e:
                span = None
                node = getattr(parser, '_current_node', None)
                if node is not None:
                    span = parser.span_tracker.get_span(node)
                if isinstance(e, (AttributeError, TypeError)):
                    hint = (
                        "an internal type check failed while parsing; an argument may "
                        "have an unsupported type — check that kernel arguments match "
                        "the expected Tile/Tensor/scalar types"
                    )
                else:
                    hint = "Check your function definition for errors"
                raise ParserSyntaxError(
                    f"Failed to parse kernel function '{self._func.__name__}': {type(e).__name__}: {e}",
                    span=span,
                    hint=hint,
                ) from e

            external_funcs = list(parser.external_funcs.values())
            starting_line = self._line_offset + 1
            program_span = ir.Span(self._source_file, starting_line, self._col_offset)
            program = ir.Program(
                external_funcs + [ir_func], program_name, program_span, parser.debug_info
            )
            return program, parser.matched_target

        except ParserError as e:
            _attach_source_lines_to_error(e, self._source_file, self._source_lines_raw)
            raise
