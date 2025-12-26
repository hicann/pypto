#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025 CANN community contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""PTO Script Expression Evaluator.

This module provides the expression evaluator for the PTO frontend parser.
The evaluator is responsible for evaluating Python expressions during the
parsing phase, converting them to concrete values or PTO IR constructs.

Key Features:
    - Expression Evaluation: Evaluates Python expressions in the context of
      available variables and the PTO module namespace
    - Symbolic Support: Handles SymbolicScalar objects and converts them to
      concrete values when possible
    - Error Handling: Converts Python evaluation errors to ParserError with
      proper source location information

The evaluator uses Python's built-in compile() and eval() functions to
execute expressions, providing flexibility while maintaining safety through
controlled variable scoping.

Example:
    # During parsing, evaluate an expression like: pypto.Tensor([16], "float32")
    result = ExprEvaluator.eval(expr_node, var_table, diagnostics)
"""

import ast
from typing import Any

import pypto

from . import doc
from .diagnostics import Diagnostics
from .error import ParserError


class ExprEvaluator:
    """Expression evaluator for PTO frontend."""

    var_table: dict[str, Any]
    diag: Diagnostics

    def __init__(self, var_table: dict[str, Any], diag: Diagnostics) -> None:
        self.var_table = var_table
        self.diag = diag

    @staticmethod
    def eval(node: doc.expr, var_table: dict[str, Any], diag: Diagnostics) -> Any:
        """Evaluate an expression node using the provided variable table.

        This is the main entry point for expression evaluation during parsing.
        It creates an ExprEvaluator instance and evaluates the expression using
        Python's compile() and eval() functions.

        Parameters
        ----------
        node : doc.expr
            The expression AST node to evaluate.
        var_table : dict[str, Any]
            Variable table containing available names and their values.
        diag : Diagnostics
            Diagnostics instance for error reporting.

        Returns
        -------
        Any
            The evaluated result of the expression.

        Raises
        ------
        ParserError
            If evaluation fails due to undefined names, type errors, etc.
        """
        self = ExprEvaluator(var_table, diag)
        result = self.visit(node)
        return result

    def visit(self, node: doc.expr) -> Any:
        """Visit and evaluate an expression node.

        Parameters
        ----------
        node : doc.expr
            The expression node to visit.

        Returns
        -------
        Any
            The evaluated result.
        """
        return self._eval_by_python(node, self.var_table)

    def _eval_by_python(self, node: doc.expr, var_table: dict[str, Any]) -> Any:
        node = doc.from_doc(node)
        if isinstance(node, ast.expr):
            # Case 1: a simple expression
            mod = ast.fix_missing_locations(ast.Expression(body=node))
            exe = compile(mod, filename=self.diag.source.source_name, mode="eval")
            dict_locals = var_table.copy()
            # Replace SymbolicScalars with concrete values if available
            for key, value in dict_locals.items():
                if isinstance(value, pypto.SymbolicScalar) and value.is_concrete():
                    dict_locals[key] = value.concrete()
            try:
                return eval(exe, {}, dict_locals)  # pylint: disable=eval-used
            except Exception as e:
                raise ParserError(node, f"{type(e).__name__}: {e}") from e
        elif isinstance(node, ast.Expr):
            # Case 2: a expression in a statement
            mod = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
            exe = compile(mod, filename=self.diag.source.source_name, mode="exec")
            dict_locals = var_table.copy()
            # Replace SymbolicScalars with concrete values if available
            for key, value in dict_locals.items():
                if isinstance(value, pypto.SymbolicScalar) and value.is_concrete():
                    dict_locals[key] = value.concrete()
            try:
                exec(exe, {}, dict_locals)  # pylint: disable=exec-used
            except Exception as e:
                raise ParserError(node, f"{type(e).__name__}: {e}") from e
        else:
            # Other unsupported expression types, raise python native error,
            # which will be caught by the parser and reported as a bug.
            raise NotImplementedError("Unsupported expression type.")
