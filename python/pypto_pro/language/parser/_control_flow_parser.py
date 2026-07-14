# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Control-flow parsing helpers for ASTParser."""

from __future__ import annotations

import ast
from typing import Any

from pypto.pypto_impl import ir
from pypto.pypto_impl.ir import DataType

from .diagnostics import ParserSyntaxError, UnsupportedFeatureError


def _body_without_docstring(func_def: ast.FunctionDef) -> list[ast.stmt]:
    body = list(func_def.body)
    if body and isinstance(body[0], ast.Expr):
        value = body[0].value
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            return body[1:]
    return body


def validate_single_tail_return(func_def: ast.FunctionDef, context: str) -> tuple[ast.Return, str, str] | None:
    """Require at most one return, and only as the top-level final statement."""
    returns = [node for node in ast.walk(func_def) if isinstance(node, ast.Return)]
    if not returns:
        return None

    if len(returns) > 1:
        return (
            returns[1],
            f"{context} only supports a single return statement.",
            "Keep one top-level return at the end of the function. "
            "Only @pl.jit/@pl.kernel supports early return.",
        )

    body = _body_without_docstring(func_def)
    tail_stmt = body[-1] if body else None
    if returns[0] is not tail_stmt:
        return (
            returns[0],
            f"{context} only supports return as the final top-level statement.",
            "Move the return to the end of the function body, or use @pl.jit/@pl.kernel "
            "when early return is required.",
        )

    return None


class ControlFlowParserMixin:
    """Mixin containing loop, branch, with-scope, and return parsing."""

    _VALID_ITERATORS = {"range"}
    _ITERATOR_ERROR = "For loop must use pl.range()"
    _ITERATOR_HINT = "Use pl.range() as the iterator"

    @staticmethod
    def _get_with_context_attr(stmt: ast.With) -> str | None:
        """Return the context manager attribute name for supported with calls."""
        context_expr = stmt.items[0].context_expr
        if not isinstance(context_expr, ast.Call):
            return None
        func = context_expr.func
        return func.attr if isinstance(func, ast.Attribute) else None

    @staticmethod
    def _describe_with_context(stmt: ast.With) -> str:
        """Render the offending context manager(s) for a diagnostic message.

        Returns a source-like description (e.g. ``pl.section_vf()``) so the error
        names exactly what was written, instead of only listing what is allowed.
        """
        try:
            return ", ".join(ast.unparse(item.context_expr) for item in stmt.items)
        except Exception:
            return "<unknown>"

    def parse_for_loop(self, stmt: ast.For) -> None:
        """Parse for loop with pl.range()."""
        iter_call = self._validate_for_loop_iterator(stmt)
        loop_var_name = self._parse_for_loop_target(stmt)
        range_args = self._parse_range_call(iter_call)

        loop_var = self.builder.var(loop_var_name, ir.ScalarType(DataType.INDEX))
        span = self.span_tracker.get_span(stmt)

        with self.builder.for_loop(
            loop_var,
            range_args["start"],
            range_args["stop"],
            range_args["step"],
            span,
        ) as loop:
            self._parse_for_loop_body(stmt, loop, loop_var, loop_var_name)

    def parse_while_loop(self, stmt: ast.While) -> None:
        """Parse natural while loop syntax.

        Natural while syntax: while condition: body

        This creates a WhileStmt without iter_args (non-SSA form).
        The C++ ConvertToSSA pass will convert it to SSA form if needed.

        Args:
            stmt: While AST node
        """
        condition = self.parse_expression(stmt.test)
        span = self.span_tracker.get_span(stmt)

        prev_loop_builder = self.current_loop_builder
        prev_in_while_loop = self.in_while_loop
        with self.builder.while_loop(condition, span) as loop:
            self.current_loop_builder = loop
            self.in_while_loop = True
            self.scope_manager.enter_scope("while")

            for body_stmt in stmt.body:
                self.parse_statement(body_stmt)

            # Variables leak to outer scope (ConvertToSSA will handle)
            self.scope_manager.exit_scope(leak_vars=True)
            self.in_while_loop = prev_in_while_loop
            self.current_loop_builder = prev_loop_builder

    def parse_if_statement(self, stmt: ast.If) -> None:
        """Parse if statement.

        Variables from both branches leak to the outer scope, and the
        C++ ConvertToSSA pass handles creating phi nodes.

        Args:
            stmt: If AST node
        """
        test_node, is_constexpr = self._unwrap_constexpr(stmt.test)
        condition = self.parse_expression(test_node)
        span = self.span_tracker.get_span(stmt)

        condition = self._resolve_constexpr_condition(test_node, condition, is_constexpr, span)

        with self.builder.if_stmt(condition, span) as if_builder:
            self.current_if_builder = if_builder
            self.in_if_stmt = True

            self.scope_manager.enter_scope("if")
            for then_stmt in stmt.body:
                self.parse_statement(then_stmt)
            self.scope_manager.exit_scope(leak_vars=True)

            if stmt.orelse:
                if_builder.else_()
                self.scope_manager.enter_scope("else")
                for else_stmt in stmt.orelse:
                    self.parse_statement(else_stmt)
                self.scope_manager.exit_scope(leak_vars=True)

        self.in_if_stmt = False
        self.current_if_builder = None

    def parse_with_statement(self, stmt: ast.With) -> None:
        """Parse with statement for scope contexts.

        Currently supports:
        - with pl.section_vector(): ... (creates SectionStmt with Vector section)
        - with pl.section_cube(): ... (creates SectionStmt with Cube section)

        Args:
            stmt: With AST node
        """
        # Check that we have exactly one context manager
        if len(stmt.items) != 1:
            raise ParserSyntaxError(
                f"Only a single context manager is supported in a 'with' statement, "
                f"but got {len(stmt.items)}: 'with {self._describe_with_context(stmt)}:'",
                span=self.span_tracker.get_span(stmt),
                hint="Use 'with pl.section_vector():' or 'with pl.section_cube():'",
            )

        attr = self._get_with_context_attr(stmt)
        span = self.span_tracker.get_span(stmt)

        # Check if this is pl.section_vector() or pl.section_cube()
        if attr == "section_vector":
            self._parse_section_with_body(stmt.body, ir.SectionKind.Vector, span, "vec")
            return
        if attr == "section_cube":
            self._parse_section_with_body(stmt.body, ir.SectionKind.Cube, span, "cube")
            return

        # Unsupported context manager
        raise UnsupportedFeatureError(
            f"Unsupported context manager 'with {self._describe_with_context(stmt)}:'"
            + (f" (pl.{attr}() is not a valid section here)" if attr else ""),
            span=self.span_tracker.get_span(stmt),
            hint=(
                "Only 'with pl.section_vector():' or 'with pl.section_cube():' "
                "are currently supported. VF code must be placed in a @pl.vector_function "
                "decorated function."
            ),
        )

    def parse_return(self, stmt: ast.Return) -> None:
        """Parse return statement.

        Args:
            stmt: Return AST node
        """
        span = self.span_tracker.get_span(stmt)

        if stmt.value is None or (
            isinstance(stmt.value, ast.Constant) and stmt.value.value is None
        ):
            # `return None` is identical to a bare `return`; treat it as a void
            # return so a helper can early-exit on a compile-time-dead branch
            # (e.g. `if constInfo.is_tnd: return None`) that inlining later drops.
            self.builder.return_stmt(None, span)
            return

        if self._void_return_only:
            raise ParserSyntaxError(
                f"{self._void_return_context} only supports bare return or return None; "
                "returning values is not supported.",
                span=span,
                hint=(
                    "Do not write `return <value>`; only use `return` or `return None`. "
                    "Pass output Tensor/Tile/buffer parameters for data results."
                ),
            )

        # A tuple return (`return a, b`) is parsed as a single MakeTuple
        # expression via parse_tuple_literal, so the ReturnStmt always carries
        # exactly one value. Downstream inlining lowers it to `target = MakeTuple(...)`
        # and backend codegen resolves the tuple via tuple_var_to_make_tuple_.
        return_expr = self.parse_expression(stmt.value)
        self.builder.return_stmt([return_expr], span)

    def parse_break(self, stmt: ast.Break) -> None:
        """Parse break statement.

        Args:
            stmt: Break AST node
        """
        if not self.in_for_loop and not self.in_while_loop:
            raise ParserSyntaxError(
                "'break' statement outside of a loop",
                span=self.span_tracker.get_span(stmt),
                hint="break can only be used inside a for or while loop",
            )
        span = self.span_tracker.get_span(stmt)
        self.builder.break_stmt(span)

    def parse_continue(self, stmt: ast.Continue) -> None:
        """Parse continue statement.

        Args:
            stmt: Continue AST node
        """
        if not self.in_for_loop and not self.in_while_loop:
            raise ParserSyntaxError(
                "'continue' statement outside of a loop",
                span=self.span_tracker.get_span(stmt),
                hint="continue can only be used inside a for or while loop",
            )
        span = self.span_tracker.get_span(stmt)
        self.builder.continue_stmt(span)

    def _parse_section_with_body(
        self,
        body: list[ast.stmt],
        kind: ir.SectionKind,
        span,
        saved_key: str | None = None,
    ) -> None:
        """Parse a section body and optionally persist its local variables."""
        with self.builder.section(kind, span):
            self.scope_manager.enter_scope("section")
            if saved_key is not None and self._section_saved_vars.get(saved_key):
                for name, value in self._section_saved_vars[saved_key].items():
                    self.scope_manager.define_var(name, value, allow_redef=True)
            for body_stmt in body:
                self.parse_statement(body_stmt)
            saved_vars = self.scope_manager.exit_scope(leak_vars=False)
            if saved_key is not None:
                self._section_saved_vars[saved_key] = saved_vars

    def _validate_for_loop_iterator(self, stmt: ast.For) -> ast.Call:
        """Validate that for loop uses pl.range().

        Returns:
            The call node for pl.range()
        """
        if not isinstance(stmt.iter, ast.Call):
            raise ParserSyntaxError(
                self._ITERATOR_ERROR,
                span=self.span_tracker.get_span(stmt.iter),
                hint=self._ITERATOR_HINT,
            )

        iter_call = stmt.iter
        func = iter_call.func
        if isinstance(func, ast.Attribute) and func.attr in self._VALID_ITERATORS:
            return iter_call

        raise ParserSyntaxError(
            self._ITERATOR_ERROR,
            span=self.span_tracker.get_span(stmt.iter),
            hint=self._ITERATOR_HINT,
        )

    def _parse_for_loop_target(self, stmt: ast.For) -> str:
        """Parse for loop target, returning the loop variable name."""
        if not isinstance(stmt.target, ast.Name):
            raise ParserSyntaxError(
                "For loop target must be a simple name",
                span=self.span_tracker.get_span(stmt.target),
                hint="Use: for i in pl.range(n)",
            )
        return stmt.target.id

    def _parse_for_loop_body(
        self,
        stmt: ast.For,
        loop: Any,
        loop_var: ir.Var,
        loop_var_name: str,
    ) -> None:
        """Parse the body of a for loop inside the loop context."""
        prev_loop_builder = self.current_loop_builder
        prev_in_for_loop = self.in_for_loop
        self.current_loop_builder = loop
        self.in_for_loop = True
        self.scope_manager.enter_scope("for")
        self.scope_manager.define_var(loop_var_name, loop_var, allow_redef=True)

        for body_stmt in stmt.body:
            self.parse_statement(body_stmt)

        self.scope_manager.exit_scope(leak_vars=True)
        self.in_for_loop = prev_in_for_loop
        self.current_loop_builder = prev_loop_builder

    def _parse_range_call(self, call: ast.Call) -> dict[str, Any]:
        """Parse pl.range() call arguments.

        Args:
            call: AST Call node for pl.range()

        Returns:
            Dictionary with start, stop, step
        """
        if call.keywords:
            raise ParserSyntaxError(
                "pl.range() does not support keyword arguments",
                span=self.span_tracker.get_span(call),
                hint="Use: pl.range(stop), pl.range(start, stop), or pl.range(start, stop, step)",
            )

        if len(call.args) < 1:
            raise ParserSyntaxError(
                "pl.range() requires at least 1 argument (stop)",
                span=self.span_tracker.get_span(call),
                hint="Provide at least the stop value: pl.range(10) or pl.range(0, 10)",
            )

        start = 0
        step = 1

        if len(call.args) == 1:
            stop = self.parse_expression(call.args[0])
        elif len(call.args) == 2:
            start = self.parse_expression(call.args[0])
            stop = self.parse_expression(call.args[1])
        else:
            start = self.parse_expression(call.args[0])
            stop = self.parse_expression(call.args[1])
            step = self.parse_expression(call.args[2])

        return {"start": start, "stop": stop, "step": step}
