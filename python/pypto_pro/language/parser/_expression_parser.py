# Copyright (c) PyPTO Contributors.
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
from typing import Any

from pypto_pro.ir import op as ir_op
from pypto_pro.ir._operators import make_binary as _make_binary
from pypto.pypto_impl import ir
from pypto.pypto_impl.ir import DataType

from .diagnostics import (
    ParserSyntaxError,
    ParserTypeError,
    UndefinedVariableError,
    UnsupportedFeatureError,
)


class ExpressionParserMixin:
    """Mixin containing expression, attribute, and subscript parsing."""

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
            result = self._materialize_nested_call(result, self.span_tracker.get_span(expr))
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
        var = self.scope_manager.lookup_var(var_name)
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

        # Tile + offset in VF scope → GetItemExpr (pointer arithmetic)
        if isinstance(binop.op, ast.Add) and isinstance(left.type, ir.TileType):
            return ir.GetItemExpr(left, right, span)

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

        return _make_binary(op_map[op_type], left, right, span)

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

        op_type = type(compare.ops[0])
        if op_type not in op_map:
            raise UnsupportedFeatureError(
                f"Unsupported comparison: {op_type.__name__}",
                span=self.span_tracker.get_span(compare),
                hint="Use supported comparisons: ==, !=, <, <=, >, >=",
            )

        return _make_binary(op_map[op_type], left, right, span)

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

        return op_map[op_type](operand, span)

    def parse_boolop(self, expr: ast.BoolOp) -> ir.Expr:
        span = self.span_tracker.get_span(expr)
        operands = [self.parse_expression(v) for v in expr.values]

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

        result = operands[0]
        for operand in operands[1:]:
            result = fold_fn(result, operand, bool_dtype, span)
        return result

    def parse_ifexp(self, expr: ast.IfExp) -> ir.Expr:
        span = self.span_tracker.get_span(expr)
        test_node, is_constexpr = self._unwrap_constexpr(expr.test)
        condition = self.parse_expression(test_node)

        condition = self._resolve_constexpr_condition(test_node, condition, is_constexpr, span)

        if is_constexpr and isinstance(condition, (ir.ConstBool, ir.ConstInt)):
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
            if not ir.structural_equal(then_value.type, else_value.type, enable_auto_mapping=False):
                raise ParserTypeError(
                    f"Ternary expression branches have mismatched types: "
                    f"then-branch has type {then_value.type}, "
                    f"else-branch has type {else_value.type}",
                    span=span,
                    hint="Ensure both branches of the ternary expression have the same type",
                )
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
            obj_expr = self.scope_manager.lookup_var(obj_name)
            if isinstance(obj_expr, ir.Expr):
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

        if not isinstance(value_type, (ir.TileType, ir.TupleType)):
            raise ParserTypeError(
                f"Subscript requires tuple or tile type, got {type(value_type).__name__}",
                span=span,
                hint="Only tuple types support subscript access in this context",
            )

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
                    "Multi-dimensional subscript is not supported",
                    span=span,
                    hint="Use a scalar index like arr[0]",
                )
            if isinstance(value_type, ir.TupleType):
                self._check_uniform_tuple_types(value_type, span)

        index_expr = self.parse_expression(subscript.slice)
        item_expr = ir.GetItemExpr(value_expr, index_expr, span)
        # A slice of a tile aliases the tile's buffer, so carry the base's mutex
        # metadata onto the slice.  auto_mutex then locks accesses to the slice on
        # the base's buf_id; it is consumed when the slice is bound to a var (see
        # _consume_nbuf_pending).  Nested slices compose: the inner slice is tagged
        # from its root here, and the outer one inherits from the (tagged) inner.
        if isinstance(value_type, ir.TileType):
            meta = self._tile_mutex_meta.get(id(value_expr))
            if meta is not None:
                self._tile_mutex_meta[id(item_expr)] = meta
        return item_expr

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

    def _materialize_nested_call(self, result, span: ir.Span):
        """Let-bind a Call result when it appears as a sub-expression.

        Only materializes ``ir.Call`` results. All other returns (Var, ConstInt,
        GetItemExpr, MakeTuple, BinaryExpr, raw closure values, etc.) pass
        through unchanged so existing call sites that rely on non-Call return
        shapes keep working.

        Skips Calls with UnknownType to avoid materializing void/sentinel
        results (e.g. nbuf.advance() void calls).
        """
        if not isinstance(result, ir.Call):
            return result
        if isinstance(result.type, ir.UnknownType):
            return result
        name = f"_expr_tmp_{self._expr_tmp_counter}"
        self._expr_tmp_counter += 1
        return self.builder.let(name, result, span=span)

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
        obj = self.scope_manager.lookup_var(node.func.value.id)
        if self.is_tile_group(obj) and method in ("next", "current", "previous"):
            return self._lower_group_accessor(obj, method, span)
        return None
