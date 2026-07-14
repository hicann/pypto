# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Assignment parsing helpers for ASTParser."""

from __future__ import annotations

import ast
from typing import Any

from pypto.pypto_impl import ir
from pypto.pypto_impl.ir import DataType

from .diagnostics import ParserSyntaxError, ParserTypeError, UnsupportedFeatureError


def _as_const_int_list(value_expr: Any) -> "list[int] | None":
    """Return the ints of a ``MakeTuple`` of ``ConstInt`` (e.g. ``[1, 3]``), else None."""
    if not isinstance(value_expr, ir.MakeTuple):
        return None
    ints: list[int] = []
    for elt in value_expr.elements:
        if not isinstance(elt, ir.ConstInt):
            return None
        ints.append(elt.value)
    return ints


class AssignmentParserMixin:
    """Mixin containing assignment parsing methods for ``ASTParser``."""

    def parse_annotated_assignment(self, stmt: ast.AnnAssign) -> None:
        """Parse annotated assignment: var: type = value.

        Args:
            stmt: AnnAssign AST node
        """
        if not isinstance(stmt.target, ast.Name):
            raise ParserSyntaxError(
                "Only simple variable assignments supported",
                span=self.span_tracker.get_span(stmt.target),
                hint="Use a simple variable name for assignment targets",
            )

        var_name = stmt.target.id
        span = self.span_tracker.get_span(stmt)

        # Parse value expression
        if stmt.value is None:
            raise UnsupportedFeatureError(
                "Annotated assignment with no value is not supported",
                span=self.span_tracker.get_span(stmt),
                hint="Provide a value for the assignment",
            )
        value_expr = self.parse_expression(stmt.value, nested=False)

        # Use annotation type as override when it carries memref info
        annotation_type = self.type_resolver.resolve_type_if_memref(stmt.annotation)
        var = self.builder.let(var_name, value_expr, var_type=annotation_type, span=span)
        self._consume_nbuf_pending(var, value_expr)

        # Register in scope
        self.scope_manager.define_var(var_name, var, span=span)

    def parse_assignment(self, stmt: ast.Assign) -> None:
        """Parse regular assignment: var = value or tuple unpacking.

        Args:
            stmt: Assign AST node
        """
        if len(stmt.targets) != 1:
            raise ParserSyntaxError(
                f"Unsupported assignment: {ast.unparse(stmt)}",
                span=self.span_tracker.get_span(stmt),
                hint="Use simple variable assignments or tuple unpacking",
            )

        target = stmt.targets[0]
        span = self.span_tracker.get_span(stmt)

        # Intercept VF op assignment form: reg = vf.xxx(...) or reg_lo, reg_hi = vf.xxx(...)
        # Only for compute ops that have dst registers (dst_count > 0).
        # Declaration ops (create_mask, RegTensor, compare, etc.) use normal return-value
        # assignment and must NOT be intercepted.
        vf_op = self._is_vf_op_call(stmt.value)
        if vf_op is not None:
            dst_count = self._get_vf_op_dst_count(vf_op)
            if dst_count is not None and dst_count > 0:
                self._parse_vf_assignment(target, stmt, vf_op, span)
                return

        if isinstance(target, ast.Tuple):
            self._parse_tuple_unpacking(target, stmt)
            return

        if isinstance(target, ast.Name):
            self._parse_name_assignment(target.id, stmt)
            return

        if isinstance(target, ast.Attribute):
            self._parse_struct_field_assignment(target, stmt, span)
            return

        # AssignStmt left-values are restricted to plain Vars. Field writes go through
        # struct.set (above); there is no expression-target / subscript assignment.
        raise ParserSyntaxError(
            f"Unsupported assignment target: {ast.unparse(stmt)}",
            span=span,
            hint="Only `var = ...`, `a.field = ...` (struct), or tuple unpacking are supported",
        )

    # Ops where dst type differs from src type — dtype kwarg is mandatory.
    _TYPE_CHANGING_OPS = frozenset({"astype", "muls_cast", "pack", "unpack"})

    def _parse_vf_assignment(
        self, target: ast.expr, stmt: ast.Assign, vf_op_name: str, span: ir.Span,
    ) -> None:
        """Handle ``reg = vf.xxx(src, ...)`` assignment form.

        Transparently rewrites to the statement form by:
        1. Emitting a ``vf.reg_tensor(dtype=...)`` declaration for the LHS variable
           (if not already declared), so the CCE backend emits ``RegTensor<T> var;``.
        2. Inserting the LHS variable(s) as dst arg(s) at the front of the call.
        3. Emitting the call as an EvalStmt (side-effect).

        The dtype is inferred from the first source-register argument. For ops
        with only scalar sources (e.g. ``vf.full(0.0, preg)``), the user
        must pass ``dtype=pl.DT_FP32`` as a kwarg on the VF call.

        Supports:
            reg = vf.add(a, b, preg)              → RegTensor reg; vf.add(reg, a, b, preg)
            reg_lo, reg_hi = vf.mull(a, b, preg)  → RegTensor reg_lo, reg_hi; vf.mull(reg_lo, reg_hi, a, b, pred)
        """
        dst_count = self._get_vf_op_dst_count(vf_op_name)

        call = stmt.value  # ast.Call

        # Determine dst variable names.
        # Some ops (like load_align with DINTLV mode) can produce 2 outputs even
        # though their static dst_count is 1.  When the LHS is a tuple and
        # dst_count == 1, promote to 2-dst form so the backend receives the
        # correct 4-arg call (dst0, dst1, ptr, offset).
        actual_dst_count = dst_count
        if dst_count == 1 and isinstance(target, ast.Tuple) and len(target.elts) == 2:
            actual_dst_count = 2

        if actual_dst_count == 1:
            if not isinstance(target, ast.Name):
                raise ParserSyntaxError(
                    f"vf.{vf_op_name} assignment target must be a variable name, "
                    f"got {ast.unparse(target)}",
                    span=span,
                )
            dst_names = [target.id]
            dst_targets = [target]
        elif actual_dst_count == 2:
            if not isinstance(target, ast.Tuple) or len(target.elts) != 2:
                raise ParserSyntaxError(
                    f"vf.{vf_op_name} produces 2 outputs, use tuple unpacking: "
                    f"reg0, reg1 = vf.{vf_op_name}(...)",
                    span=span,
                )
            for elt in target.elts:
                if not isinstance(elt, ast.Name):
                    raise ParserSyntaxError(
                        f"Tuple unpacking target must be variable names, got {ast.unparse(elt)}",
                        span=span,
                    )
            dst_names = [target.elts[0].id, target.elts[1].id]
            dst_targets = [target.elts[0], target.elts[1]]

        # Determine dtype: check explicit dtype kwarg first.
        # For type-changing ops (vf.astype), dtype kwarg is mandatory because
        # the dst type differs from src type.
        dtype_val = None
        for kw in call.keywords:
            if kw.arg == "dtype":
                dtype_val = self.resolve_single_kwarg("dtype", kw.value)
                break

        if dtype_val is None and vf_op_name in self._TYPE_CHANGING_OPS:
            raise ParserTypeError(
                f"vf.{vf_op_name} requires explicit dtype kwarg because dst type differs from src. "
                f"Example: vf.{vf_op_name}(src, mask, dtype=pl.DT_FP16)",
                span=span,
            )

        # Step 1: Parse source args FIRST, before any dst manipulation.
        # This is critical for self-referential ops (e.g. max0 = vf.max(max0, src, preg))
        # where the src must resolve to the OLD value, not a newly created register.
        # Also use this pass to infer dtype if not already determined.
        parsed_src_args = [self.parse_expression(src_arg) for src_arg in call.args]

        if dtype_val is None:
            for src_expr in parsed_src_args:
                src_type = getattr(src_expr, 'type', None)
                if src_type is None:
                    continue
                if hasattr(src_type, 'dtype'):
                    dtype_val = src_type.dtype
                    break
                if hasattr(src_type, 'element_dtype'):
                    dtype_val = src_type.element_dtype
                    break

        if dtype_val is None:
            raise ParserTypeError(
                f"Cannot infer dtype for vf.{vf_op_name} assignment. "
                f"Pass dtype=pl.DT_FP32 (or appropriate type) as a kwarg.",
                span=span,
            )

        parsed_kwargs = {k: v for k, v in self.parse_op_kwargs(call).items() if k != "dtype"}

        # Step 2: Declare dst variables via vf.reg_tensor (first definition only).
        # For existing variables, reuse them directly — do NOT create a new
        # reg_tensor, as that would lose the accumulated value in self-referential ops.
        dst_vars = []
        for _, (name, tgt) in enumerate(zip(dst_names, dst_targets)):
            existing = self.scope_manager.lookup_var(name)
            if existing is None:
                self.current_target_name = name
                regtensor_call = ir.create_op_call(
                    "vf.reg_tensor", [], {"dtype": dtype_val}, span,
                )
                var = self.builder.let(name, regtensor_call, span=span)
                self.scope_manager.define_var(name, var, span=span)
                dst_vars.append(var)
            else:
                dst_vars.append(existing)

        # Step 3: Construct the VF op call with dst vars + parsed src args.
        # vf.shift_left / vf.shift_right are unified ops: the backend codegen
        # picks the per-lane (vshl/vshr) vs uniform-scalar (vshls/vshrs) form
        # from the shift-amount argument type (see EmitVFShiftLeft/Right), so no
        # special routing is needed here.
        backend_op_name = self._VF_OP_NAME_MAP.get(vf_op_name, vf_op_name)
        all_args = dst_vars + parsed_src_args
        call_expr = ir.create_op_call(f"vf.{backend_op_name}", all_args, parsed_kwargs, span)
        self.builder.emit(ir.EvalStmt(call_expr, span))

    def _parse_tuple_unpacking(self, target: ast.Tuple, stmt: ast.Assign) -> None:
        span = self.span_tracker.get_span(stmt)
        value_expr = self.parse_expression(stmt.value, nested=False)
        if not isinstance(value_expr, ir.Expr) or not isinstance(value_expr.type, ir.TupleType):
            raise ParserTypeError(
                f"Cannot unpack non-tuple value: {ast.unparse(stmt.value)}",
                span=span,
            )
        expected = len(value_expr.type.types)
        actual = len(target.elts)
        if actual != expected:
            raise ParserTypeError(
                f"Cannot unpack tuple with {expected} items into {actual} targets",
                span=span,
            )
        tuple_var = self.builder.let("_tuple_tmp", value_expr, span=span)
        for i, elt in enumerate(target.elts):
            if not isinstance(elt, ast.Name):
                raise ParserSyntaxError(
                    f"Tuple unpacking target must be a variable name, got {ast.unparse(elt)}",
                    span=self.span_tracker.get_span(elt),
                    hint="Use simple variable names in tuple unpacking: a, b, c = func()",
                )
            if elt.id == "_":
                continue
            item_expr = ir.GetItemExpr(
                tuple_var,
                ir.ConstInt(i, DataType.INDEX, ir.Span.unknown()),
                span,
            )
            var = self.builder.let(elt.id, item_expr, span=span)
            self.scope_manager.define_var(elt.id, var, span=span)

    def _parse_name_assignment(self, var_name: str, stmt: ast.Assign) -> None:
        span = self.span_tracker.get_span(stmt)
        self.current_target_name = var_name
        value_expr = self.parse_expression(stmt.value, nested=False)
        if value_expr is None:
            raise ParserTypeError(
                f"Cannot assign void function result to '{var_name}'",
                span=span,
                hint="Functions used as expressions must return a value",
            )
        if isinstance(value_expr, ir.Expr):
            var = self.builder.let(var_name, value_expr, span=span)
            self._consume_nbuf_pending(var, value_expr)
            self.scope_manager.define_var(var_name, var, span=span)
            if self._auto_mutex:
                self._emit_auto_mutex_unlocks()
            # Track compile-time integer constants so TileType shape/valid_shape can fold them.
            if isinstance(value_expr, ir.ConstInt):
                self._const_scalars[var_name] = value_expr.value
            else:
                self._const_scalars.pop(var_name, None)
            # Track compile-time constant int lists (e.g. ``qkv_tile_dims = [1, 3]``) so a
            # compile-time-only list kwarg such as ``tile_dims`` can fold them, including when the
            # list is threaded through an implicit-helper parameter.
            const_list = _as_const_int_list(value_expr)
            if const_list is not None:
                self._const_lists[var_name] = const_list
            else:
                self._const_lists.pop(var_name, None)
        else:
            self.scope_manager.define_var(var_name, value_expr, span=span)

    def _parse_struct_field_assignment(self, target: ast.Attribute, stmt: ast.Assign, span: ir.Span) -> None:
        """Lower ``base.field = rhs`` to ``EvalStmt(struct.set(base, rhs, field=...))``.

        The field name (not an index) is carried on the call, mirroring struct.create's
        ``fields``; codegen emits ``base.field = rhs;``. This keeps the write off the
        AssignStmt left-value, which is restricted to plain Vars.
        """
        base = self.parse_expression(target.value)
        field_name = target.attr
        fields = self.named_fields(base)
        if not fields or field_name not in fields:
            raise ParserTypeError(
                f"Cannot assign to '{ast.unparse(target)}': base is not a named struct/tuple "
                f"with field '{field_name}'",
                span=span,
            )
        value_expr = self.parse_expression(stmt.value, nested=False)
        if not isinstance(value_expr, ir.Expr):
            raise ParserTypeError(
                f"Right-hand side of '{ast.unparse(target)}' must be an IR expression",
                span=span,
            )
        call = ir.create_op_call("struct.set", [base, value_expr], {"field": field_name}, span)
        self.builder.emit(ir.EvalStmt(call, span))
