# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Expression parsing helpers for ASTParser."""

from __future__ import annotations

import ast
import logging
from typing import Any

from pypto_pro.ir import op as ir_op
from pypto_pro.ir._operators import make_binary as _make_binary
from pypto_pro.ir._utils import _normalize_expr
from pypto.pypto_impl import ir
from pypto.pypto_impl.ir import DataType

from .diagnostics import (
    ParserSyntaxError,
    ParserTypeError,
    UndefinedVariableError,
    UnsupportedFeatureError,
)


def _scalar_branches_reconcilable(then_type: Any, else_type: Any) -> bool:
    """Whether two ternary branch types differ only by same-category scalar dtype.

    Such pairs (e.g. ``INT32`` from a tensor read vs ``INDEX`` from a shape access, or
    two float widths) are promoted to a common dtype when the if statement is finalized
    in the builder, mirroring the promotion binary operators already perform. Non-scalar
    or cross-category (int vs float) mismatches are not reconcilable here.
    """
    if not isinstance(then_type, ir.ScalarType) or not isinstance(else_type, ir.ScalarType):
        return False
    then_dtype = then_type.dtype
    else_dtype = else_type.dtype
    both_int = then_dtype.is_int() and else_dtype.is_int()
    both_float = then_dtype.is_float() and else_dtype.is_float()
    return both_int or both_float


class ExpressionParserMixin:

    """Mixin containing expression, attribute, and subscript parsing."""

    @staticmethod
    def _const_int_value(expr: ir.Expr) -> int | None:
        if isinstance(expr, ir.ConstInt):
            return expr.value
        if isinstance(expr, ir.ConstBool):
            return 1 if expr.value else 0
        return None

    @staticmethod
    def _const_dtype(expr: ir.Expr) -> DataType:
        return expr.type.dtype if isinstance(expr.type, ir.ScalarType) else DataType.INDEX

    @staticmethod
    def _check_uniform_tuple_types(value_type: ir.TupleType, span: ir.Span) -> None:
        """Validate that all elements of a tuple share the same type (required for variable indexing)."""
        elem_types = list(value_type.types)
        if not elem_types:
            raise ParserTypeError("Cannot index into empty tuple", span=span)
        first_type = elem_types[0]
        for i, t in enumerate(elem_types[1:], 1):
            if not ir.structural_equal(t, first_type, enable_auto_mapping=False):
                raise ParserTypeError(
                    f"Variable tuple index requires all elements to have the same type, "
                    f"but element 0 has type {first_type} and element {i} has type {t}",
                    span=span,
                    hint="Use a constant index to access elements of different types",
                )

    @staticmethod
    def _resolve_constexpr_condition(
        test_node: ast.expr,
        condition: ir.Expr,
        is_constexpr: bool,
        span: ir.Span,
    ) -> ir.Expr:
        """Validate and normalize a constexpr condition.

        Raises ``ParserSyntaxError`` if ``is_constexpr`` is True but
        ``condition`` is not a compile-time constant.  Converts
        ``ir.ConstBool`` to ``ir.ConstInt`` (value 0/1) so downstream
        consumers only need to handle ``ConstInt``.
        """
        if is_constexpr and not isinstance(condition, (ir.ConstBool, ir.ConstInt)):
            raise ParserSyntaxError(
                "pl.constexpr() can only accept compile-time constants, "
                f"but got a runtime expression: {ast.unparse(test_node)}",
                span=span,
                hint="pl.constexpr() only accepts compile-time constants, "
                     "e.g. 'pl.constexpr(True)' or 'pl.constexpr(False)'",
            )

        if is_constexpr and isinstance(condition, ir.ConstBool):
            condition = ir.ConstInt(1 if condition.value else 0, DataType.BOOL, span)

        return condition

    @staticmethod
    def _is_pl_range_call(call: ast.Call) -> bool:
        """Check if call node is pl.range()."""
        func = call.func
        return isinstance(func, ast.Attribute) and func.attr == "range"

    @staticmethod
    def _try_const_fold_in(left, elements, is_not_in, span):
        """If left and all elements are compile-time constants, eval directly."""
        if not isinstance(left, (ir.ConstInt, ir.ConstFloat, ir.ConstBool)):
            return None
        const_values = []
        for elt in elements:
            if isinstance(elt, ir.ConstInt):
                const_values.append(elt.value)
            elif isinstance(elt, ir.ConstFloat):
                const_values.append(elt.value)
            elif isinstance(elt, ir.ConstBool):
                const_values.append(elt.value)
            else:
                return None
        result = left.value in const_values
        if is_not_in:
            result = not result
        return ir.ConstBool(result, span)

    def parse_evaluation_statement(self, stmt: ast.Expr) -> None:
        """Parse evaluation statement (EvalStmt).

        Evaluation statements represent operations executed for their side effects,
        with the return value discarded (e.g., synchronization barriers).

        Args:
            stmt: Expr AST node
        """
        expr = self.parse_expression(stmt.value, nested=False)
        span = self.span_tracker.get_span(stmt)

        # Void inline functions or python_var method calls (e.g. nbuf.advance())
        # return None or a non-IR sentinel — nothing to emit.
        if expr is None or not isinstance(expr, ir.Expr):
            return

        # Emit EvalStmt using builder method
        self.builder.eval_stmt(expr, span)

        # Auto-mutex: emit deferred mutex_unlock AFTER the op
        if self._auto_mutex:
            self._emit_auto_mutex_unlocks()

    def parse_expression(self, expr: ast.expr, *, nested: bool = True) -> Any:
        """Parse expression and return IR Expr.

        Args:
            expr: AST expression node
            nested: True (default) when this expression should materialize a
                ir.Call result through a let-binding so the outer expression
                only references the temporary Var. Callers at assignment RHS
                positions (``var = expr``, ``struct.field = expr``) and at
                EvalStmt positions should pass ``nested=False`` to avoid
                redundant ``_expr_tmp_N`` indirection.

        Returns:
            IR expression
        """
        if expr in self._parsed_expr_cache:
            return self._parsed_expr_cache[expr]
        self._current_node = expr
        if isinstance(expr, ast.Name):
            result = self.parse_name(expr)
        elif isinstance(expr, ast.Constant):
            result = self.parse_constant(expr)
        elif isinstance(expr, ast.BinOp):
            result = self.parse_binop(expr)
        elif isinstance(expr, ast.Compare):
            result = self.parse_compare(expr)
        elif isinstance(expr, ast.Call):
            result = self.parse_call(expr)
        elif isinstance(expr, ast.Attribute):
            result = self.parse_attribute(expr)
        elif isinstance(expr, ast.UnaryOp):
            result = self.parse_unaryop(expr)
        elif isinstance(expr, ast.List):
            result = self.parse_list(expr)
        elif isinstance(expr, ast.Tuple):
            result = self.parse_tuple_literal(expr)
        elif isinstance(expr, ast.Subscript):
            result = self.parse_subscript(expr)
        elif isinstance(expr, ast.BoolOp):
            result = self.parse_boolop(expr)
        elif isinstance(expr, ast.IfExp):
            result = self.parse_ifexp(expr)
        else:
            raise UnsupportedFeatureError(
                f"Unsupported expression type: {type(expr).__name__}",
                span=self.span_tracker.get_span(expr),
                hint="Use supported expressions like variables, constants, operations, or function calls",
            )

        if nested:
            result = self._materialize_nested_expr(result, self.span_tracker.get_span(expr))
        self._parsed_expr_cache[expr] = result
        return result

    def parse_name(self, name: ast.Name) -> ir.Expr | Any:
        """Parse variable name reference.

        Resolves names by checking the DSL scope first, then falling back
        to closure variables from the enclosing Python scope.

        Args:
            name: Name AST node

        Returns:
            IR expression (Var from scope, or constant/tuple from closure)
        """
        var_name = name.id
        # An inline helper may only resolve its own parameters and locals from
        # the DSL scope.  ``lookup_var_bounded`` behaves like the normal lookup
        # outside an inline scope, but stops at the inline boundary so a helper
        # cannot silently capture a caller IR variable.
        const = self.const_env.get(var_name)
        if const is not None:
            return const

        var = self.scope_manager.lookup_var_bounded(var_name)
        if var is not None:
            return var

        # Fall back to closure variables
        result = self.expr_evaluator.try_eval_as_ir(name)
        if result is not None:
            return result

        raise UndefinedVariableError(
            f"Undefined variable '{var_name}'",
            span=self.span_tracker.get_span(name),
            hint="Check if the variable is defined before using it or is available in the enclosing scope",
        )

    def parse_constant(self, const: ast.Constant) -> ir.Expr:
        """Parse constant value.

        Args:
            const: Constant AST node

        Returns:
            IR constant expression
        """
        span = self.span_tracker.get_span(const)
        value = const.value

        if isinstance(value, bool):
            return ir.ConstBool(value, span)
        elif isinstance(value, int):
            return ir.ConstInt(value, DataType.INDEX, span)
        elif isinstance(value, float):
            return ir.ConstFloat(value, DataType.DEFAULT_CONST_FLOAT, span)
        elif isinstance(value, str):
            return value
        else:
            raise ParserTypeError(
                f"Unsupported constant type: {type(value)}",
                span=self.span_tracker.get_span(const),
                hint="Use int, float, or bool constants",
            )

    def parse_binop(self, binop: ast.BinOp) -> ir.Expr:
        """Parse binary operation.

        Args:
            binop: BinOp AST node

        Returns:
            IR binary expression
        """
        span = self.span_tracker.get_span(binop)
        left = self.parse_expression(binop.left)
        right = self.parse_expression(binop.right)

        # Tile + offset: only supported in VF section as pointer arithmetic.
        # Non-VF sections require slice syntax (tile[r:r+h, c:c+w]) for sub-views.
        if isinstance(binop.op, ast.Add) and isinstance(left.type, ir.TileType):
            if self.inline_vf_depth == 0:
                raise ParserSyntaxError(
                    "Tile + offset is not supported outside VF section; "
                    "use tile[i:i+h, j:j+w] slice syntax for sub-view",
                    span=span,
                    hint="Replace `tile + offset` with `tile[row:row+h, :]` for offset access",
                )
            # VF section: lower to block.subview with original shape (VF codegen
            # only uses the offset as pointer arithmetic, shape is ignored).
            from pypto_pro.ir.op.block_ops import block_ir_op
            shape_tuple = ir.MakeTuple(left.type.shape, span)
            result = ir.create_op_call(
                block_ir_op("subview"),
                [left, right, shape_tuple],
                {},
                span,
            )
            return result

        # Raw pointer arithmetic: only `ptr + offset` is meaningful (→ pl.addptr).
        # Every other operator on a pointer (-, *, /, //, %) is forbidden.
        if isinstance(left.type, ir.PtrType):
            if isinstance(binop.op, ast.Add):
                # Sub-byte element types (INT4/UINT4/FP4/HF4) pack two elements per
                # byte, so an element offset cannot address a half-byte and there is
                # no valid C element type to lower to. Forbid `ptr + offset` on them.
                if left.type.dtype.get_bit() < 8:
                    raise ParserTypeError(
                        f"Pointer arithmetic ('ptr + offset') is not supported on the "
                        f"sub-byte element type '{left.type.dtype.to_string()}'",
                        span=span,
                        hint="Offsetting by elements cannot address a half-byte. "
                        "Reinterpret the pointer as a byte-addressable dtype via "
                        "pl.make_ptr (e.g. pl.DT_UINT8) before pointer arithmetic.",
                    )
                return ir_op.ptr.addptr(left, right, span=span)
            raise UnsupportedFeatureError(
                f"Unsupported operator '{type(binop.op).__name__}' on a pointer (pl.Ptr)",
                span=span,
                hint="Only 'ptr + offset' is supported for pointer arithmetic "
                "(equivalent to pl.addptr(ptr, offset)).",
            )

        # Map AST operators to IR builder names. Everything is routed through
        # ``make_binary`` so mixed int/float operands (e.g. ``1.0 / G`` or ``1.0 + G``
        # with ``G`` an index Var) get Python-consistent promotion to float.
        op_map = {
            ast.Add: "add",
            ast.Sub: "sub",
            ast.Mult: "mul",
            ast.Div: "truediv",
            ast.FloorDiv: "floordiv",
            ast.Mod: "mod",
            ast.BitAnd: "bit_and",
            ast.BitOr: "bit_or",
            ast.BitXor: "bit_xor",
            ast.LShift: "bit_shift_left",
            ast.RShift: "bit_shift_right",
        }

        op_type = type(binop.op)
        if op_type not in op_map:
            raise UnsupportedFeatureError(
                f"Unsupported binary operator: {op_type.__name__}",
                span=self.span_tracker.get_span(binop),
                hint="Use supported operators: +, -, *, /, //, %, &, |, ^, <<, >>",
            )

        op_name = op_map[op_type]
        folded = self._fold_const_binop(op_name, left, right, span)
        return folded if folded is not None else _make_binary(op_name, left, right, span)

    def parse_compare(self, compare: ast.Compare) -> ir.Expr:
        """Parse comparison operation.

        Args:
            compare: Compare AST node

        Returns:
            IR comparison expression
        """
        if len(compare.ops) != 1 or len(compare.comparators) != 1:
            raise ParserSyntaxError(
                "Only simple comparisons supported",
                span=self.span_tracker.get_span(compare),
                hint="Use single comparison operators like: a < b, not chained comparisons",
            )

        span = self.span_tracker.get_span(compare)
        op_type = type(compare.ops[0])

        # Dispatch `in` / `not in` to the desugar path: they are not scalar
        # binary comparisons, so they cannot enter the op_map below.
        if op_type in (ast.In, ast.NotIn):
            return self._parse_in_operator(compare, span)

        # ── Handle 'is None' / 'is not None' for pointer null checks ──
        # Converts to runtime comparison: cast(ptr, UINT64) == 0 / cast(ptr, UINT64) != 0
        # This allows optional pl.Ptr parameters to be checked at runtime without
        # compile-time constant folding or multiple kernel variants.
        if op_type in (ast.Is, ast.IsNot):
            comparator = compare.comparators[0]
            if isinstance(comparator, ast.Constant) and comparator.value is None:
                # Left side must be a simple variable name (typically a pl.Ptr or pl.Tensor parameter)
                if not isinstance(compare.left, ast.Name):
                    raise ParserSyntaxError(
                        "'is None' only supported on simple variable names",
                        span=span,
                        hint="Use 'param_name is None' for null checks on pl.Ptr parameters",
                    )
                # Parse the left operand (must be a pl.Ptr parameter)
                left = self.parse_expression(compare.left)
                # 'is None' only makes sense for pointer parameters. pl.Tensor is a
                # descriptor (ptr + shape); null-checking it has no meaning and the
                # generated code would reference an undeclared identifier. Guide users
                # to pl.Ptr for optional inputs (reviewer guidance: optional inputs
                # should be declared as pl.Ptr since pl.Tensor requires a shape even
                # when the argument is None, making the shape meaningless).
                if not isinstance(left.type, ir.PtrType):
                    raise ParserTypeError(
                        "'is None' / 'is not None' is only supported on pl.Ptr parameters",
                        span=span,
                        hint="Use pl.Ptr[dtype] for optional pointer inputs; "
                             "pl.Tensor requires a shape even when the argument is None",
                    )
                # Cast pointer to UINT64 for comparison (IR requires ScalarType for eq/ne)
                left_as_int = ir.cast(left, ir.DataType.UINT64, span)
                zero = ir.ConstInt(0, ir.DataType.UINT64, span)
                # Generate runtime comparison: cast(ptr, UINT64) == 0 or != 0
                op = "eq" if op_type is ast.Is else "ne"
                return _make_binary(op, left_as_int, zero, span)
            else:
                raise ParserTypeError(
                    "'is' / 'is not' only supported with None",
                    span=span,
                    hint="Use '==' for value comparison, or 'param is None' for pointer null checks",
                )

        # ── Standard comparison operators (==, !=, <, <=, >, >=) ──
        left = self.parse_expression(compare.left)
        right = self.parse_expression(compare.comparators[0])

        # Comparisons also promote mixed int/float operands to float via make_binary.
        op_map = {
            ast.Eq: "eq",
            ast.NotEq: "ne",
            ast.Lt: "lt",
            ast.LtE: "le",
            ast.Gt: "gt",
            ast.GtE: "ge",
        }

        if op_type not in op_map:
            raise UnsupportedFeatureError(
                f"Unsupported comparison: {op_type.__name__}",
                span=span,
                hint="Use supported comparisons: ==, !=, <, <=, >, >=",
            )

        op_name = op_map[op_type]
        folded = self._fold_const_binop(op_name, left, right, span)
        return folded if folded is not None else _make_binary(op_name, left, right, span)

    def parse_unaryop(self, unary: ast.UnaryOp) -> ir.Expr:
        """Parse unary operation.

        Args:
            unary: UnaryOp AST node

        Returns:
            IR unary expression
        """
        span = self.span_tracker.get_span(unary)
        operand = self.parse_expression(unary.operand)

        op_map = {
            ast.USub: ir.neg,
            ast.Not: ir.not_,
            ast.Invert: ir.bit_not,  # ``~x`` bitwise NOT (integer operands)
            ast.UAdd: lambda operand, _span: operand,  # ``+x`` is the identity
        }

        op_type = type(unary.op)
        if op_type not in op_map:
            raise UnsupportedFeatureError(
                f"Unsupported unary operator: {op_type.__name__}",
                span=self.span_tracker.get_span(unary),
                hint="Use supported unary operators: -, not",
            )

        value = self._const_int_value(operand)
        if value is not None:
            if op_type is ast.USub:
                return ir.ConstInt(-value, self._const_dtype(operand), span)
            if op_type is ast.Not:
                return ir.ConstInt(int(not value), DataType.BOOL, span)
            if op_type is ast.Invert:
                return ir.ConstInt(~value, self._const_dtype(operand), span)
            if op_type is ast.UAdd:
                return operand
        return op_map[op_type](operand, span)

    def parse_boolop(self, expr: ast.BoolOp) -> ir.Expr:
        span = self.span_tracker.get_span(expr)
        bool_dtype = DataType.BOOL
        if isinstance(expr.op, ast.And):
            fold_fn = ir.And
        elif isinstance(expr.op, ast.Or):
            fold_fn = ir.Or
        else:
            raise UnsupportedFeatureError(
                f"Unsupported boolean operator: {type(expr.op).__name__}",
                span=span,
            )

        result = self.parse_expression(expr.values[0])
        for value_node in expr.values[1:]:
            result_value = self._const_int_value(result)
            if isinstance(expr.op, ast.And) and result_value == 0:
                return ir.ConstInt(0, bool_dtype, span)
            if isinstance(expr.op, ast.Or) and result_value is not None and result_value != 0:
                return ir.ConstInt(1, bool_dtype, span)
            operand = self.parse_expression(value_node)
            operand_value = self._const_int_value(operand)
            if result_value is not None and operand_value is not None:
                result = ir.ConstInt(
                    int(bool(result_value) and bool(operand_value)) if isinstance(expr.op, ast.And)
                    else int(bool(result_value) or bool(operand_value)),
                    bool_dtype,
                    span,
                )
            else:
                result = fold_fn(result, operand, bool_dtype, span)
        return result

    def parse_ifexp(self, expr: ast.IfExp) -> ir.Expr:
        span = self.span_tracker.get_span(expr)
        test_node, is_constexpr = self._unwrap_constexpr(expr.test)
        condition = self.parse_expression(test_node)

        condition = self._resolve_constexpr_condition(test_node, condition, is_constexpr, span)

        if isinstance(condition, (ir.ConstBool, ir.ConstInt)):
            is_true = (condition.value if isinstance(condition, ir.ConstBool) else condition.value != 0)
            chosen = expr.body if is_true else expr.orelse
            result = self.parse_expression(chosen, nested=False)
            if not isinstance(result, ir.Expr):
                raise ParserTypeError(
                    "Ternary expression branch must return an IR expression",
                    span=span,
                    hint="Ensure the chosen branch of the ternary expression is a valid expression",
                )
            return result

        tmp_name = f"_ifexpr_tmp_{self._ifexpr_tmp_counter}"
        self._ifexpr_tmp_counter += 1

        with self.builder.if_stmt(condition, span) as if_builder:
            then_value = self.parse_expression(expr.body, nested=False)
            if not isinstance(then_value, ir.Expr):
                raise ParserTypeError(
                    "Ternary expression branch must return an IR expression",
                    span=span,
                    hint="Ensure the 'then' branch of the ternary expression is a valid expression",
                )
            if_builder.return_var(tmp_name, then_value.type, span)
            self.builder.emit(ir.YieldStmt([then_value], span))

            if_builder.else_(span)
            else_value = self.parse_expression(expr.orelse, nested=False)
            if not isinstance(else_value, ir.Expr):
                raise ParserTypeError(
                    "Ternary expression branch must return an IR expression",
                    span=span,
                    hint="Ensure the 'else' branch of the ternary expression is a valid expression",
                )
            if not ir.structural_equal(
                then_value.type, else_value.type, enable_auto_mapping=False
            ) and not _scalar_branches_reconcilable(then_value.type, else_value.type):
                raise ParserTypeError(
                    f"Ternary expression branches have mismatched types: "
                    f"then-branch has type {then_value.type}, "
                    f"else-branch has type {else_value.type}",
                    span=span,
                    hint="Ensure both branches of the ternary expression have the same type",
                )
            # Same-category scalar branches with differing dtypes (e.g. INT32 vs INDEX)
            # are promoted to a common dtype when the if statement is finalized.
            self.builder.emit(ir.YieldStmt([else_value], span))

        return_var = if_builder.output(0)

        self.scope_manager.define_var(tmp_name, return_var, span=span)
        return return_var

    def make_named_tuple(self, elements: list, field_names, span: ir.Span) -> ir.Expr:
        """Build a named ``MakeTuple`` and record its type's field names.

        Single chokepoint for named-tuple construction: the field names are not stored
        on the expression or the type; they are registered into the Program's
        ``IRDebugInfo`` keyed by the resulting TupleType pointer, so codegen can recover
        ``a.field`` even when the read does not fold to the MakeTuple.
        """
        field_names = list(field_names)
        mt = ir.MakeTuple(elements, span)
        if field_names and self.debug_info is not None:
            self.debug_info.register_tuple_fields(mt.type, field_names)
        return mt

    def named_fields(self, expr) -> list[str]:
        """Field names of a named tuple / struct expr, from the IRDebugInfo side table.

        Single source of truth: the TupleType no longer carries field names. All parsers
        building one Program share one IRDebugInfo (sub-parsers adopt the parent's), so
        cross-function field access resolves names registered at the construction site.
        Returns [] for non-tuples or unregistered/positional tuples.
        """
        if self.debug_info is None:
            return []
        if not (isinstance(expr, ir.Expr) and isinstance(expr.type, ir.TupleType)):
            return []
        return list(self.debug_info.get_tuple_fields(expr.type) or [])

    def register_struct_fields(self, expr: ir.Expr, field_names) -> ir.Expr:
        """Record the field names of a ``struct.create`` / named-tuple result type."""
        expr_type = expr.type
        if self.debug_info is not None and isinstance(expr_type, ir.TupleType) and field_names:
            self.debug_info.register_tuple_fields(expr_type, list(field_names))
        return expr

    def lower_attr_access(self, base: ir.Expr, field_name: str, span: ir.Span):
        """Lower attribute read ``base.field`` to ``GetItemExpr(base, index)``.

        Resolves the field index from the base's named TupleType and records the
        type's field names in ``IRDebugInfo``. Returns None if ``base`` is not a
        named tuple or has no such field (caller keeps its own error handling).
        """
        fields = self.named_fields(base)
        if not fields or field_name not in fields:
            return None
        idx = fields.index(field_name)
        return ir.GetItemExpr(base, ir.ConstInt(idx, DataType.INDEX, span), span)

    def parse_attribute(self, attr: ast.Attribute) -> ir.Expr:
        """Parse attribute access.

        Args:
            attr: Attribute AST node

        Returns:
            IR expression
        """
        span = self.span_tracker.get_span(attr)

        if attr.attr == "shape":
            base_expr = self.parse_expression(attr.value)
            if isinstance(base_expr, ir.Expr) and isinstance(base_expr.type, ir.TensorType):
                return ir.MakeTuple(list(base_expr.type.shape), span)
            if isinstance(base_expr, ir.Expr):
                lowered = self.lower_attr_access(base_expr, attr.attr, span)
                if lowered is not None:
                    return lowered
            raise ParserTypeError(
                "tensor.shape requires TensorType input",
                span=span,
                hint="Use tensor.shape only on Tensor values",
            )

        if isinstance(attr.value, ast.Name):
            obj_name = attr.value.id
            field_name = attr.attr

            if obj_name == "pl" and field_name.startswith("DT_"):
                dtype = self.type_resolver.resolve_dtype(attr)
                return ir.ConstInt(int(dtype), DataType.INDEX, span)

            # If the scope variable's static IR type is a named tuple
            # (TupleType with dbg_name), lower to GetItemExpr(base, index).
            obj_expr = self.scope_manager.lookup_var_bounded(obj_name)
            if isinstance(obj_expr, ir.Expr):
                const_obj = self.const_env.get(obj_name)
                if isinstance(const_obj, ir.MakeTuple) and not self._is_struct_array_tuple(const_obj):
                    fields = self.named_fields(const_obj)
                    if field_name in fields:
                        return const_obj.elements[fields.index(field_name)]
                lowered = self.lower_attr_access(obj_expr, field_name, span)
                if lowered is not None:
                    return lowered

        # Check for nested attribute access like pl.MemorySpace.Left
        if isinstance(attr.value, ast.Attribute):
            inner_attr = attr.value
            if isinstance(inner_attr.value, ast.Name):
                inner_obj_name = inner_attr.value.id
                inner_field_name = inner_attr.attr
                if inner_obj_name == "pl" and inner_field_name == "MemorySpace":
                    memory_space = self.type_resolver.resolve_memory_space(attr)
                    return ir.ConstInt(memory_space.value, DataType.INT64, span)

        raise UnsupportedFeatureError(
            f"Standalone attribute access not supported: {ast.unparse(attr)}",
            span=span,
            hint="Attribute access is only supported for tiling parameters (e.g., tiling.x) "
                         "or within function calls",
        )

    def parse_list(self, list_node: ast.List) -> ir.MakeTuple:
        """Parse list literal into MakeTuple IR expression.


        Args:
            list_node: List AST node

        Returns:
            MakeTuple IR expression
        """
        span = self.span_tracker.get_span(list_node)
        elements = [self.parse_expression(elt) for elt in list_node.elts]
        return ir.MakeTuple(elements, span)

    def parse_tuple_literal(self, tuple_node: ast.Tuple) -> ir.MakeTuple:
        """Parse tuple literal like (x, y, z).

        Args:
            tuple_node: Tuple AST node

        Returns:
            MakeTuple IR expression
        """
        span = self.span_tracker.get_span(tuple_node)
        elements = [self.parse_expression(elt) for elt in tuple_node.elts]
        return ir.MakeTuple(elements, span)

    def parse_subscript(self, subscript: ast.Subscript) -> ir.Expr:
        span = self.span_tracker.get_span(subscript)

        if isinstance(subscript.value, ast.Attribute) and subscript.value.attr == "shape":
            base_expr = self.parse_expression(subscript.value.value)
            if isinstance(base_expr, ir.Expr) and isinstance(base_expr.type, ir.TensorType):
                return self._parse_tensor_shape_subscript(base_expr, subscript.slice, span)

        value_expr = self.parse_expression(subscript.value)
        value_type = value_expr.type

        # Tile/Tensor subscript: dispatch by index type
        #   - Integer index (no ':') → getval (scalar access, Tile and Tensor)
        #   - Slice index (has ':')  → sub-view (block.subview, Tile only)
        if isinstance(value_type, (ir.TensorType, ir.TileType)):
            has_slice = isinstance(subscript.slice, ast.Slice) or (
                isinstance(subscript.slice, ast.Tuple)
                and any(isinstance(e, ast.Slice) for e in subscript.slice.elts)
            )
            if has_slice:
                return self._parse_slice_subscript(value_expr, subscript.slice, span)
            # All-integer index: A[i, j] → getval(A, i*cols+j)
            index_expr = self._parse_scalar_subscript_index(value_expr, subscript.slice, span)
            meta = self._tile_mutex_meta.get(value_expr) if self._auto_mutex else None
            from pypto_pro.ir.op.block_ops import _ir_getval
            result = _ir_getval(value_expr, index_expr, span=span)
            if meta is not None:
                from ._op_pipeline import get_op_pipe
                from pypto_pro.ir.op.system_ops import mutex_lock, mutex_unlock
                pipe = get_op_pipe("getval")
                buf_id_ir, mutex_ids = meta
                self.builder.emit(ir.EvalStmt(
                    mutex_lock(pipe=pipe, mutex_id=buf_id_ir, mutex_ids=mutex_ids, span=span), span))
                result = self._materialize_nested_expr(result, span)
                self.builder.emit(ir.EvalStmt(
                    mutex_unlock(pipe=pipe, mutex_id=buf_id_ir, mutex_ids=mutex_ids, span=span), span))
            return result

        if not isinstance(value_type, ir.TupleType):
            raise ParserTypeError(
                f"Subscript requires tuple, tile, or tensor type, got {type(value_type).__name__}",
                span=span,
                hint="Subscript access is supported on Tuple, Tile, and Tensor types",
            )

        # TupleType subscript: tuple[i] element access
        if isinstance(subscript.slice, ast.Constant):
            if not isinstance(subscript.slice.value, int):
                raise ParserSyntaxError(
                    "Tuple index must be an integer",
                    span=span,
                    hint="Use integer index like tuple[0]",
                )
        else:
            if isinstance(subscript.slice, ast.Tuple):
                raise ParserSyntaxError(
                    "Multi-dimensional subscript is not supported for tuples",
                    span=span,
                    hint="Use a scalar index like tuple[0]",
                )
            self._check_uniform_tuple_types(value_type, span)

        index_expr = self.parse_expression(subscript.slice)
        if (
            isinstance(value_expr, ir.MakeTuple)
            and not self._is_struct_array_tuple(value_expr)
            and isinstance(index_expr, ir.ConstInt)
        ):
            index = index_expr.value
            if 0 <= index < len(value_expr.elements):
                return value_expr.elements[index]

        # CCE resolves a non-folded GetItem through the underlying MakeTuple's
        # backing array. struct_array stays as a Var here because it is not a
        # parser constant; ordinary tuples may remain folded MakeTuple values.
        item_expr = ir.GetItemExpr(value_expr, index_expr, span)
        # A subscript aliases the base's buffer, so carry the base's mutex
        # metadata onto the item.  auto_mutex then locks accesses to the item on
        # the base's buf_id; it is consumed when the item is bound to a var (see
        # _transfer_tile_sync_metadata).
        meta = self._tile_mutex_meta.get(value_expr)
        if meta is not None:
            self._tile_mutex_meta[item_expr] = meta
        return item_expr

    def _is_const_expr(self, expr: Any) -> bool:
        """Return whether *expr* is a parser-propagatable IR constant."""
        if isinstance(expr, (ir.ConstInt, ir.ConstBool, ir.ConstFloat)):
            return True
        # MakeTuple is immutable.  struct_array is the sole exception because
        # its elements are mutable structs whose static and dynamic accesses
        # must share CCE's backing array.
        return isinstance(expr, ir.MakeTuple) and not self._is_struct_array_tuple(expr)

    def _is_struct_array_tuple(self, tuple_value: ir.MakeTuple) -> bool:
        return id(tuple_value) in self._struct_array_tuple_ids

    def _update_const_env(self, name: str, value: Any) -> None:
        if isinstance(value, ir.Expr) and self._is_const_expr(value):
            self.const_env[name] = value
        else:
            self.const_env.pop(name, None)

    def _fold_const_binop(self, op_name: str, left: ir.Expr, right: ir.Expr, span: ir.Span) -> ir.Expr | None:
        """Fold integer scalar operations without bypassing parser type checks."""
        left_value = self._const_int_value(left)
        right_value = self._const_int_value(right)
        if left_value is None or right_value is None:
            return None
        dtype = self._const_dtype(left)
        binary_ops = {
            "add": lambda: left_value + right_value,
            "sub": lambda: left_value - right_value,
            "mul": lambda: left_value * right_value,
            "floordiv": lambda: left_value // right_value,
            "mod": lambda: left_value % right_value,
            "bit_and": lambda: left_value & right_value,
            "bit_or": lambda: left_value | right_value,
            "bit_xor": lambda: left_value ^ right_value,
            "bit_shift_left": lambda: left_value << right_value,
            "bit_shift_right": lambda: left_value >> right_value,
        }
        if op_name in binary_ops:
            try:
                return ir.ConstInt(binary_ops[op_name](), dtype, span)
            except ZeroDivisionError:
                return None
        comparisons = {
            "eq": left_value == right_value,
            "ne": left_value != right_value,
            "lt": left_value < right_value,
            "le": left_value <= right_value,
            "gt": left_value > right_value,
            "ge": left_value >= right_value,
        }
        if op_name in comparisons:
            return ir.ConstInt(int(comparisons[op_name]), DataType.BOOL, span)
        return None

    def _desugar_in_literal(self, left, elements, is_not_in, span, elements_are_ir=False):
        """Desugar x in (a,b,c) -> Or-chain of eq; x not in -> And-chain of ne."""
        folded = self._try_const_fold_in(left, elements, is_not_in, span)
        if folded is not None:
            return folded
        if len(elements) == 0:
            return ir.ConstBool(is_not_in, span)
        cmp_name = "ne" if is_not_in else "eq"
        fold_fn = ir.And if is_not_in else ir.Or
        bool_dtype = DataType.BOOL

        def get_element(elt):
            return elt if elements_are_ir else self.parse_expression(elt)

        result = _make_binary(cmp_name, left, get_element(elements[0]), span)
        for elt in elements[1:]:
            cmp_expr = _make_binary(cmp_name, left, get_element(elt), span)
            result = fold_fn(result, cmp_expr, bool_dtype, span)
        return result

    def _desugar_in_range(self, left, range_call, is_not_in, span):
        """Desugar x in pl.range(start, stop, step).

        step == 1:  x >= start and x < stop
        step != 1:  x >= start and x < stop and (x - start) % step == 0
        """
        args = self._parse_range_call(range_call)
        start = _normalize_expr(args["start"], span)
        stop = _normalize_expr(args["stop"], span)
        step = _normalize_expr(args["step"], span)
        ge_start = _make_binary("ge", left, start, span)
        lt_stop = _make_binary("lt", left, stop, span)
        result = ir.And(ge_start, lt_stop, DataType.BOOL, span)
        step_is_one = (
            isinstance(step, ir.ConstInt) and step.value == 1
            or isinstance(step, ir.ConstFloat) and step.value == 1.0
        )
        if not step_is_one:
            diff = _make_binary("sub", left, start, span)
            mod_val = _make_binary("mod", diff, step, span)
            zero = ir.ConstInt(0, DataType.INDEX, span)
            result = ir.And(result, _make_binary("eq", mod_val, zero, span),
                            DataType.BOOL, span)
        return ir.not_(result, span) if is_not_in else result

    def _parse_in_operator(self, compare: ast.Compare, span: ir.Span) -> ir.Expr:
        """Parse ``x in (...)`` / ``x not in (...)``.

        Supports three container forms, all desugared to existing IR ops:
          1. Literal tuple/list:  x in (a, b, c)      -> or-chain of eq
          2. pl.range() call:     x in pl.range(s,e,k) -> range bounds + modulo check
          3. Closure variable:    x in my_list          -> eval at compile time, expand to eq-chain
        """
        is_not_in = isinstance(compare.ops[0], ast.NotIn)
        left = self.parse_expression(compare.left)
        container = compare.comparators[0]

        if isinstance(container, (ast.Tuple, ast.List)):
            return self._desugar_in_literal(left, container.elts, is_not_in, span)

        if isinstance(container, ast.Call) and self._is_pl_range_call(container):
            return self._desugar_in_range(left, container, is_not_in, span)

        success, value = self.expr_evaluator.try_eval_expr(container)
        if success and isinstance(value, (list, tuple)):
            ir_elements = [self.expr_evaluator.python_value_to_ir(v, span) for v in value]
            return self._desugar_in_literal(
                left, ir_elements, is_not_in, span, elements_are_ir=True)

        raise ParserSyntaxError(
            f"'{'not in' if is_not_in else 'in'}' only supports tuple/list literals, "
            f"pl.range(), or compile-time list/tuple variables, "
            f"got {ast.unparse(container)}",
            span=span,
            hint="Use: x in (a, b, c), x in pl.range(10), or x in <compile-time-list>",
        )

    def _restore_tuple_var_for_unfolded_getitem(self, value_expr: ir.Expr) -> ir.Expr:
        """Recover a tuple Var when an un-folded GetItem has its MakeTuple value."""
        if not isinstance(value_expr, ir.MakeTuple):
            return value_expr
        for name, const_value in self.const_env.items():
            if const_value is not value_expr:
                continue
            scoped_value = self.scope_manager.lookup_var_bounded(name)
            if isinstance(scoped_value, ir.Expr) and isinstance(scoped_value.type, ir.TupleType):
                return scoped_value
        return value_expr

    def _unwrap_constexpr(self, expr: ast.expr) -> tuple[ast.expr, bool]:
        """Detect pl.constexpr(...) or bare constexpr(...) wrapper.

        Returns (inner_expression, is_constexpr). If *expr* is a call to
        ``pl.constexpr(x)`` or ``constexpr(x)``, returns ``(x, True)``;
        otherwise returns ``(expr, False)``.
        """
        if not isinstance(expr, ast.Call):
            return expr, False
        func = expr.func
        if isinstance(func, ast.Attribute) and func.attr == "constexpr":
            if len(expr.args) != 1:
                raise ParserSyntaxError(
                    "pl.constexpr() expects exactly one argument",
                    span=self.span_tracker.get_span(expr),
                    hint="Use 'pl.constexpr(condition)' with a single compile-time constant.",
                )
            return expr.args[0], True
        if isinstance(func, ast.Name) and func.id == "constexpr":
            if len(expr.args) != 1:
                raise ParserSyntaxError(
                    "constexpr() expects exactly one argument",
                    span=self.span_tracker.get_span(expr),
                    hint="Use 'constexpr(condition)' with a single compile-time constant.",
                )
            return expr.args[0], True
        return expr, False

    def _parse_tensor_shape_subscript(
        self,
        base_expr: ir.Expr,
        index_node: ast.expr,
        span: ir.Span,
    ) -> ir.Expr:
        """Resolve ``tensor.shape[index]`` to the dimension stored in its TensorType.

        TensorType shape expressions are the logical tensor dimensions and are the
        source of truth for both static dimensions and dynamic shape ABI variables.
        No runtime tensor operation is emitted.
        """
        if not isinstance(base_expr.type, ir.TensorType):
            raise ParserTypeError(
                "tensor.shape requires TensorType input",
                span=span,
                hint="Use tensor.shape[index] only on Tensor values",
            )

        success, axis = self.expr_evaluator.try_eval_expr(index_node)
        if not success or type(axis) is not int:
            raise ParserSyntaxError(
                "tensor.shape index must be a compile-time integer",
                span=span,
                hint="Use tensor.shape[0], tensor.shape[-1], or a compile-time integer expression",
            )

        original_axis = axis
        tensor_type = base_expr.type
        rank = len(tensor_type.shape)
        if axis < 0:
            axis += rank
        if axis < 0 or axis >= rank:
            raise ParserTypeError(
                f"shape index {original_axis} out of range for tensor of rank {rank}",
                span=span,
            )
        return tensor_type.shape[axis]

    def _parse_scalar_subscript_index(
        self,
        container_expr: ir.Expr,
        slice_node: ast.expr,
        span: ir.Span,
    ) -> ir.Expr:
        """Parse A[i, j, ...] into a linear offset expression for getval/setval.

        Multi-index: A[i, j] → i * N + j (row-major linearization).
        Single index: A[i] → i (only valid for 1D containers).
        """
        container_type = container_expr.type
        shape = container_type.shape

        # Single index A[x]: only valid for 1D containers (rank 1)
        if not isinstance(slice_node, ast.Tuple):
            if len(shape) != 1:
                if self.inline_vf_depth > 0:
                    raise ParserSyntaxError(
                        f"Tile[x] in VF section is not supported; use `tile + x` for pointer offset",
                        span=span,
                        hint="Replace `tile[x]` with `tile + x`",
                    )
                raise ParserSyntaxError(
                    f"Subscript A[x] requires 1D container, but got rank {len(shape)}; "
                    f"use {len(shape)} indices or add ':' for sub-view (Tile)",
                    span=span,
                    hint=f"Use A[i, j] for scalar access or A[x:, :] for sub-view",
                )
            return self.parse_expression(slice_node)

        # A[i, j, ...] — multi-dimensional coordinate
        elts = slice_node.elts

        if len(elts) != len(shape):
            raise ParserTypeError(
                f"Subscript has {len(elts)} indices but container has rank {len(shape)}",
                span=span,
                hint=f"Use {len(shape)} indices to match the container shape",
            )
        indices = [self.parse_expression(e) for e in elts]
        from pypto_pro.language.parser._utils import _const_int_value

        offset = indices[-1]
        stride = shape[-1]
        for dim in range(len(indices) - 2, -1, -1):
            # Fold constant stride into a single ConstInt to avoid mul IR nodes
            stride_val = _const_int_value(stride)
            if stride_val == 1:
                product = indices[dim]
            else:
                product = indices[dim] * stride
            offset = product + offset
            if dim > 0:
                next_shape_val = _const_int_value(shape[dim])
                cur_stride_val = _const_int_value(stride)
                if next_shape_val is not None and cur_stride_val is not None:
                    stride = ir.ConstInt(next_shape_val * cur_stride_val, DataType.INDEX, span)
                else:
                    stride = shape[dim] * stride
        return offset

    def _parse_slice_subscript(
        self,
        container_expr: ir.Expr,
        slice_node: ast.expr,
        span: ir.Span,
    ) -> ir.Expr:
        """Parse tile[r:, c:] into a block.subview op (Tile only, 2D).

        Result tile preserves the original shape (row_stride unchanged);
        codegen auto-emits SetValidShape with the sub-window dimensions.
        - If slice exceeds tile shape → error.
        - If slice exceeds tile valid_shape → clamp to (valid_shape - start).
        """
        container_type = container_expr.type
        shape = container_type.shape

        if isinstance(container_type, ir.TensorType):
            raise ParserSyntaxError(
                "Tensor slice sub-view is not supported",
                span=span,
                hint="Use pl.load/pl.store with offset lists for tensor access",
            )

        if self.inline_vf_depth > 0:
            logging.warning(
                "Tile slice in VF section is lowered to pointer offset only; "
                "shape/valid_shape are ignored. Consider using `tile + offset` instead."
            )

        if not isinstance(slice_node, ast.Tuple):
            raise ParserSyntaxError(
                "1D tile slice is not supported; use 2D slice like tile[i:i+h, j:j+w]",
                span=span,
                hint="Use tile[i:i+h, j:j+w] for sub-view access",
            )

        slices = slice_node.elts
        if len(slices) != len(shape):
            raise ParserTypeError(
                f"Slice subscript has {len(slices)} dimensions but tile has rank {len(shape)}",
                span=span,
                hint=f"Use {len(shape)} indices to match the tile shape",
            )

        from pypto_pro.language.parser._utils import _const_int_value

        cols = shape[1]
        # valid_shape for clamping: compile-time (TileType.tile_view) or runtime
        # (set_validshape, tracked by tile var name).
        ct_valid = None
        tile_view = getattr(container_type, 'tile_view', None)
        if tile_view is not None:
            ct_valid = getattr(tile_view, 'valid_shape', None)
        rt_valid = None
        if isinstance(container_expr, ir.Var):
            rt_valid = self.get_tile_valid_shape(container_expr)
        dim_starts = []
        new_shape_exprs = []

        for i, s in enumerate(slices):
            if not isinstance(s, ast.Slice):
                raise ParserSyntaxError(
                    f"Tile slice does not support integer index in dimension {i}",
                    span=span,
                    hint="Tile slices must use ':' for all dimensions",
                )
            start = ir.ConstInt(0, DataType.INDEX, span) if s.lower is None else self.parse_expression(s.lower)
            dim_starts.append(start)
            start_val = _const_int_value(start)

            # Compute slice size: upper - start (upper defaults to shape[i]).
            # Clamp upper to shape[i] (Python slice semantics): a[16:77] on a
            # length-64 dim becomes a[16:64], not an error.
            upper = shape[i] if s.upper is None else self.parse_expression(s.upper)
            shape_val = _const_int_value(shape[i])
            upper_val = _const_int_value(upper)
            if shape_val is not None and upper_val is not None and upper_val > shape_val:
                upper = ir.ConstInt(shape_val, DataType.INDEX, span)
                upper_val = shape_val
            size = upper if start_val == 0 else upper - start
            size_val = _const_int_value(size)

            # Check against declared shape: start must not exceed shape
            if start_val is not None and shape_val is not None:
                if start_val > shape_val:
                    raise ParserSyntaxError(
                        f"Tile slice start ({start_val}) exceeds shape dim {i} ({shape_val})",
                        span=span,
                        hint=f"Slice start must not exceed shape[{i}]",
                    )

            for vs in (
                ct_valid[i] if ct_valid is not None and i < len(ct_valid) else None,
                rt_valid[i] if rt_valid is not None and i < len(rt_valid) else None,
            ):
                if vs is None:
                    continue
                vs_val = _const_int_value(vs)
                if vs_val is not None and vs_val >= 0 and start_val is not None:
                    remaining = vs_val - start_val
                    if remaining < 0:
                        raise ParserSyntaxError(
                            f"Tile slice start ({start_val}) exceeds valid_shape dim {i} ({vs_val})",
                            span=span,
                            hint=f"Slice start must not exceed valid_shape[{i}]",
                        )
                    if size_val is not None:
                        if size_val > remaining:
                            size = ir.ConstInt(remaining, DataType.INDEX, span)
                            size_val = remaining
                    else:
                        size = ir.Min(size, ir.ConstInt(remaining, DataType.INDEX, span), DataType.INDEX, span)
                elif vs_val is None and start_val is not None:
                    # Dynamic valid_shape: emit runtime clamp min(size, vs - start)
                    remaining_expr = vs - start
                    size = ir.Min(size, remaining_expr, DataType.INDEX, span)

            new_shape_exprs.append(size)

        # Linear offset: row_start * cols + col_start
        offset = dim_starts[0] * cols + dim_starts[1]

        from pypto_pro.ir.op.block_ops import block_ir_op
        shape_tuple = ir.MakeTuple(new_shape_exprs, span)
        view_expr = ir.create_op_call(
            block_ir_op("subview"),
            [container_expr, offset, shape_tuple],
            {},
            span,
        )

        # Propagate tile mutex metadata for auto_mutex
        self._transfer_tile_sync_metadata(view_expr, container_expr)
        return view_expr


    def _mark_make_tuple_anchor(self, result) -> None:
        """Record that a MakeTuple already has an emitted assignment anchor."""
        if isinstance(result, ir.MakeTuple):
            self._anchored_make_tuples.add(result)


    def _materialize_nested_expr(self, result, span: ir.Span):
        """Materialize nested Calls and anchor nested MakeTuples.

        Calls return their temporary Var as before. A MakeTuple instead keeps its
        expression result: the temporary let is only a CCE backing-array anchor,
        so callers can continue folding the enclosing expression.

        Skips Calls with UnknownType to avoid materializing void/sentinel
        results (e.g. nbuf.advance() void calls).
        """
        if isinstance(result, ir.MakeTuple):
            if result not in self._anchored_make_tuples:
                name = f"_tuple_anchor_{self._expr_tmp_counter}"
                self._expr_tmp_counter += 1
                self.builder.let(name, result, span=span)
                self._mark_make_tuple_anchor(result)
            return result
        if not isinstance(result, ir.Call):
            return result
        if isinstance(result.type, ir.UnknownType):
            return result
        name = f"_expr_tmp_{self._expr_tmp_counter}"
        self._expr_tmp_counter += 1
        value = self.builder.let(name, result, span=span)
        self._transfer_tile_sync_metadata(value, result)
        self._emit_auto_mutex_unlocks()
        return value

    def _route_ir_node_method(self, node: ast.Call):
        """Route ``xxx.f(...)`` to an op func when ``xxx`` is an IR node.

        Tile-group handle: next/current/previous (each returns a bare tile).
        Returns the IR result, or None when not routed.
        """
        if not isinstance(node.func, ast.Attribute):
            return None
        method = node.func.attr
        span = self.span_tracker.get_span(node)
        if not isinstance(node.func.value, ast.Name):
            return None
        obj = self.scope_manager.lookup_var_bounded(node.func.value.id)
        if self.is_tile_group(obj) and method in ("next", "current", "previous"):
            return self._lower_group_accessor(obj, method, span)
        return None
