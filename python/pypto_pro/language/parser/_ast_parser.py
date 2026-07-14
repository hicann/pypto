# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""AST parsing for converting Python DSL to IR builder calls."""
from __future__ import annotations

__all__ = ["ASTParser"]


import ast
from functools import singledispatchmethod
from typing import Any

from pypto_pro.ir import IRBuilder
from pypto.pypto_impl import ir

from ._assignment_parser import AssignmentParserMixin
from ._buffer_parser import BufferParserMixin
from ._call_parser import CallParserMixin, _infer_return_types_from_body
from ._control_flow_parser import ControlFlowParserMixin, validate_single_tail_return
from ._expression_parser import ExpressionParserMixin
from ._struct_parser import StructParserMixin
from .diagnostics import (
    ParserSyntaxError,
    ParserTypeError,
    UnsupportedFeatureError,
)
from ._expr_evaluator import ExprEvaluator
from ._scope_manager import ScopeManager
from ._span_tracker import SpanTracker
from ._type_resolver import TypeResolver
from ..typing._tiling import ArrayFieldInfo, get_tiling_fields, is_tiling_class
from ..typing.shape import _ShapePolicy


def _snake_visit_name(node: ast.AST) -> str:
    """Return the snake_case visitor handler for an AST node type."""
    chars: list[str] = []
    for char in type(node).__name__:
        if char.isupper() and chars:
            chars.append("_")
        chars.append(char.lower())
    return f"visit_{''.join(chars)}"


class ASTParser(
    AssignmentParserMixin,
    ControlFlowParserMixin,
    ExpressionParserMixin,
    StructParserMixin,
    BufferParserMixin,
    CallParserMixin,
):
    """Parses Python AST and builds IR using IRBuilder."""

    def __init__(
        self,
        source_file: str,
        source_lines: list[str],
        line_offset: int = 0,
        col_offset: int = 0,
        global_vars: set[str] | None = None,
        gvar_to_func: dict[str, ir.Function] | None = None,
        strict_ssa: bool = False,
        closure_vars: dict[str, Any] | None = None,
        auto_mutex: bool = False,
        debug_info: ir.IRDebugInfo | None = None,
        tilingkey_consts: dict[str, int] | None = None,
        datatype_consts: dict[str, Any] | None = None,
        bound_signature=None,
        void_return_only: bool = False,
        void_return_context: str = "this function",
        allow_early_return: bool = False,
    ):
        """Initialize AST parser.

        Args:
            source_file: Path to source file
            source_lines: Lines of source code (dedented for parsing)
            line_offset: Line number offset to add to AST line numbers (for dedented code)
            col_offset: Column offset to add to AST column numbers (for dedented code)
            global_vars: Optional set of function names for cross-function calls
            gvar_to_func: Optional map of function names to parsed Functions for type inference
            strict_ssa: If True, enforce SSA (single assignment). If False (default), allow reassignment.
            closure_vars: Optional variables from the enclosing scope for dynamic shape resolution
            auto_mutex: If True, automatically insert mutex lock/unlock around buffer-managed tile ops.
            void_return_only: If True, reject return values and non-None return annotations.
            void_return_context: User-facing name used only in diagnostics
                (for example, "@pl.jit/@pl.kernel" or "@pl.vector_function").
            allow_early_return: If True, skip the single-tail-return restriction.
                This is intentionally separate from void_return_only: @pl.jit/@pl.kernel
                are void-only but allow early/multiple returns, while @pl.vector_function
                and @pl.pipeline.stage are void-only and still require a single tail return.
        """
        self.span_tracker = SpanTracker(source_file, source_lines, line_offset, col_offset)
        self.scope_manager = ScopeManager(strict_ssa=strict_ssa)
        # Concrete tilingkey field values (per launch key) are injected as closure constants
        # so the parser folds field references (e.g. NeedAttnMask) to ConstInt — no template
        # params reach the IR. Field values win over any same-named closure var.
        merged_closure = {
            **(closure_vars or {}),
            **(tilingkey_consts or {}),
            **(datatype_consts or {}),
        }
        self.expr_evaluator = ExprEvaluator(
            closure_vars=merged_closure,
            span_tracker=self.span_tracker,
        )
        self.type_resolver = TypeResolver(
            expr_evaluator=self.expr_evaluator,
            scope_lookup=self.scope_manager.lookup_var,
            span_tracker=self.span_tracker,
            bound_signature=bound_signature,
        )
        self.builder = IRBuilder()
        # Tuple/struct field-name side table; owned by the Program being built and passed in
        # by the caller (kernel / decorator). Not created here; may be None for parses that
        # do not feed a Program. Sub-parsers adopt the parent's (see call_parser).
        self.debug_info = debug_info
        self.global_vars = global_vars or set()  # Track function names for cross-function calls
        self.gvar_to_func = gvar_to_func or {}  # Track parsed functions for type inference
        self.external_funcs: dict[str, ir.Function] = {}  # Track external functions referenced

        # Track context for handling yields and returns
        self.in_for_loop = False
        self.in_while_loop = False
        self.in_if_stmt = False
        self.current_if_builder = None
        self.current_loop_builder = None

        # Cache for implicitly compiled functions (keyed by id(fn))
        self.implicit_func_cache: dict[int, Any] = {}

        # VF auto_mutex metadata for compiled KernelFunctions (keyed by id(kfunc)).
        # Value:
        #   None       — non-VF function; skip VF auto_mutex entirely.
        #   {names...} — @pl.vector_function: all param names (entire body is VF
        #                scope); mutex_lock/unlock(V) wraps the func.call for
        #                tile-valued params among them.
        self.kfunc_vf_used_params: dict[int, set[str] | None] = {}

        # Counter for anonymous buffer tile variables (auto-named _buf_tile_N).
        self._buf_tile_counter: int = 0
        # Counter for let-bound buf_idx variables in tile-group cursor selection.
        self._tuple_idx_counter: int = 0
        self._expr_tmp_counter: int = 0
        self._ifexpr_tmp_counter: int = 0
        # Maps id(group_var) -> (num_tiles, mutex_ids) for make_tile_group handles.
        self.tile_group_meta: dict[int, tuple] = {}
        # Maps id(tile_var) -> (buf_id_ir, mutex_ids) for tiles returned by
        # group.next()/current()/previous(); consumed by auto_mutex.
        self._tile_mutex_meta: dict[int, tuple] = {}

        # Maps var_name -> int for scalar assignments bound to a compile-time constant
        # (e.g. ``valid_m = 128``). Lets TileType shape/valid_shape fold such vars to constants,
        # since those are compile-time template/constructor params (see resolve_static_int_list).
        self._const_scalars: dict[str, int] = {}

        # Maps var_name -> list[int] for assignments bound to a compile-time constant int list
        # (e.g. ``qkv_tile_dims = [1, 3]``). Lets a compile-time-only list kwarg such as
        # load/store/load_tile/store_tile ``tile_dims`` fold such vars to constants, including when
        # the list is threaded through an implicit-helper parameter (see _inferred_const_lists).
        self._const_lists: dict[str, list[int]] = {}

        # Cache closure-level tuple constants by name/object so repeated uses of
        # the same module constant do not create duplicate MakeTuple IR nodes.
        self._closure_tuple_ir_cache: dict[tuple[str, int], ir.Expr] = {}
        # Cache: (tuple_var_name, index_ssa_var_name) -> phi ir.Var from _build_tuple_index_chain.
        # Applies to all tuple types (tile, tensor, event ID, etc.).
        # Prevents re-emitting an if-else chain when the same buf[idx] expression
        # appears multiple times in the same linear code region.
        self._tuple_select_cache: dict[tuple[str, int], ir.Var] = {}

        self._auto_mutex = auto_mutex
        self._current_func_type = ir.FunctionType.Opaque
        self._void_return_only = void_return_only
        self._void_return_context = void_return_context
        self._allow_early_return = allow_early_return
        # Per-section-kind saved variables: allows same-kind sections to share
        # variables across multiple section blocks (interleaved cube/vector pattern).
        self._section_saved_vars: dict[str, dict] = {"cube": {}, "vec": {}}

        # Current assignment LHS name; set before parse_expression so helpers
        # (_build_tile_group_ir, _parse_struct_array_expr) can name intermediate vars.
        self.current_target_name: str = ""
        # Pending tile-group metadata consumed by _consume_nbuf_pending after builder.let.
        self._tile_group_meta_pending: tuple | None = None
        # Pending tile->mutex metadata for a tile returned by group.next()/current()/previous().
        self._tile_mutex_pending: tuple | None = None

        # Pre-resolved types for helper-call parameters (populated by _implicit_func_call).
        # Maps param_name -> ir.Type inferred from the call-site argument.
        self.inferred_param_types: dict[str, ir.Type] = {}

        self._current_node: ast.AST | None = None

        # Tile-group metadata for helper-call parameters (populated by _implicit_func_call).
        # Maps param_name -> (num_tiles, mutex_ids) so .next()/.current()/.previous() work in helpers.
        self._inferred_tile_group_meta: dict[str, tuple] = {}

        # Tile-mutex metadata for helper-call parameters (populated by _implicit_func_call).
        # Maps param_name -> (buf_id_ir, mutex_ids) so auto_mutex can lock a tile passed into
        # a helper. The buf_id_ir references the caller's scope, which is valid after the helper
        # call is inlined (InlineHelperCalls).
        self._inferred_tile_mutex_meta: dict[str, tuple] = {}

        # Compile-time constant int lists passed as helper-call arguments (populated by
        # _implicit_func_call). Maps param_name -> list[int]. A compile-time-only list kwarg
        # (e.g. ``tile_dims``) must fold to constants at parse time; threading it through a helper
        # param would otherwise leave it an ir.Var. Applied to _const_lists in _register_param.
        self._inferred_const_lists: dict[str, list[int]] = {}

    @property
    def auto_mutex_enabled(self) -> bool:
        """Return whether automatic mutex emission is enabled."""
        return self._auto_mutex

    @staticmethod
    def _attach_ptr_to_tensor_type(name: str, param_type: ir.Type, span: ir.Span) -> ir.Type:
        if not isinstance(param_type, ir.TensorType):
            return param_type
        ptr_var = ir.Var(name + "_ptr", ir.PtrType(param_type.dtype), span)
        tv = ir.TensorView() if param_type.tensor_view is None else ir.TensorView(
            param_type.tensor_view.valid_shape, param_type.tensor_view.stride, param_type.tensor_view.layout)
        tv.ptr = ptr_var
        return ir.TensorType(param_type.shape, param_type.dtype, param_type.memref, tv)

    def set_void_return_mode(self, context: str, allow_early_return: bool = False) -> None:
        """Configure void-only return mode for this parser.

        Args:
            context: User-facing name used in diagnostics (e.g. "@pl.vector_function").
            allow_early_return: If True, skip the single-tail-return restriction.
        """
        self._void_return_only = True
        self._void_return_context = context
        self._allow_early_return = allow_early_return

    def set_auto_mutex_enabled(self, enabled: bool) -> None:
        """Set automatic mutex emission for helper parsing."""
        self._auto_mutex = enabled

    def set_inferred_tile_group_meta(self, param_name: str, meta: tuple) -> None:
        """Attach inferred tile-group metadata to a helper parameter."""
        self._inferred_tile_group_meta[param_name] = meta

    def set_inferred_const_list(self, param_name: str, values: list[int]) -> None:
        """Attach a compile-time constant int list to a helper parameter."""
        self._inferred_const_lists[param_name] = values

    def set_inferred_tile_mutex_meta(self, param_name: str, meta: tuple) -> None:
        """Attach inferred tile-mutex metadata to a helper parameter."""
        self._inferred_tile_mutex_meta[param_name] = meta

    def resolve_tiling_class(self, annotation: ast.expr) -> type | None:
        """Return the tiling class if annotation refers to one in closure_vars, else None.

        Args:
            annotation: AST expression node for the annotation

        Returns:
            The resolved tiling class, or None if the annotation is not a tiling class
        """
        if not isinstance(annotation, ast.Name):
            return None
        cls = self.expr_evaluator.closure_vars.get(annotation.id)
        return cls if is_tiling_class(cls) else None

    def parse_function(
        self,
        func_def: ast.FunctionDef,
        func_type: ir.FunctionType = ir.FunctionType.Opaque,
        is_vector_function: bool = False,
    ) -> ir.Function:
        """Parse function definition and build IR.

        Args:
            func_def: AST FunctionDef node
            func_type: Function type (default: Opaque)
            is_vector_function: If True, wrap the entire function body in an
                implicit ``ir.SectionKind.VF`` section scope (for
                ``@pl.vector_function`` decorated functions).

        Returns:
            IR Function object
        """
        func_name = func_def.name
        func_span = self.span_tracker.get_span(func_def)

        if self._void_return_only and func_def.returns is not None and not (
            isinstance(func_def.returns, ast.Constant) and func_def.returns.value is None
        ):
            raise ParserSyntaxError(
                f"{self._void_return_context} only supports a None return annotation; "
                "returning values is not supported.",
                span=func_span,
                hint=(
                    "Remove the return annotation or use -> None. Do not write `return <value>`; "
                    "only use `return` or `return None`. Pass output Tensor/Tile/buffer parameters "
                    "for data results."
                ),
            )

        if not self._allow_early_return:
            context = self._void_return_context if self._void_return_only else f"Function '{func_name}'"
            return_error = validate_single_tail_return(func_def, context)
            if return_error is not None:
                return_node, message, hint = return_error
                raise ParserSyntaxError(
                    message,
                    span=self.span_tracker.get_span(return_node),
                    hint=hint,
                )

        self._current_func_type = func_type
        self.scope_manager.enter_scope("function")

        # Collect args to process, filtering out bare 'self'
        args_to_process = [
            arg for arg in func_def.args.args
            if not (arg.arg == "self" and arg.annotation is None)
        ]

        self._validate_tiling_params(args_to_process, func_def)

        self._closure_tuple_ir_cache.clear()
        has_policy_return = self.type_resolver.annotation_has_shape_policy(func_def.returns)

        with self.builder.function(func_name, func_span, func_type=func_type) as f:
            for arg in args_to_process:
                self._parse_function_param(arg, f)

            # Parse return type (skip `-> None`)
            if func_def.returns and not has_policy_return and not (
                isinstance(func_def.returns, ast.Constant) and func_def.returns.value is None
            ):
                # tuple[...] resolves to a single TupleType, matching a `return a, b`
                # body (one MakeTuple expr), so a tuple return is one return type.
                f.return_type(self.type_resolver.resolve_type(func_def.returns))

            # Hoist Python closure tuple variables to function entry so they become
            # named IR Vars rather than anonymous inline MakeTuples.
            self._hoist_closure_tuples(func_def)

            # Collect function body statements (skip docstrings)
            body_stmts: list[ast.stmt] = []
            for i, stmt in enumerate(func_def.body):
                if i == 0 and isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant):
                    if isinstance(stmt.value.value, str):
                        continue  # Skip docstring
                body_stmts.append(stmt)

            if is_vector_function:
                # @pl.vector_function: the entire body is a VF section scope.
                with self.builder.section(ir.SectionKind.VF, func_span):
                    self.scope_manager.enter_scope("section")
                    for stmt in body_stmts:
                        self.parse_statement(stmt)
                    self.scope_manager.exit_scope(leak_vars=False)
            else:
                for stmt in body_stmts:
                    self.parse_statement(stmt)

        self.scope_manager.exit_scope()
        result = f.get_result()
        if has_policy_return and func_def.returns is not None:
            inferred = _infer_return_types_from_body(result.body) if result.body else None
            if inferred is None:
                raise ParserTypeError(
                    f"Function '{func_name}' has a shape-policy return annotation but no returned value",
                    span=func_span,
                )
            self.type_resolver.validate_policy_return_types(func_def.returns, inferred)
            result = ir.Function(
                result.name,
                list(result.params),
                inferred,
                result.body,
                result.span,
                result.func_type,
            )
        return result

    def parse_statement(self, stmt: ast.stmt) -> None:
        """Parse a statement node.

        Args:
            stmt: AST statement node
        """
        self._current_node = stmt
        self._dispatch_statement(stmt)

    @singledispatchmethod
    def _dispatch_statement(self, stmt: ast.stmt) -> None:
        raise UnsupportedFeatureError(
            f"Unsupported statement type: {type(stmt).__name__}",
            span=self.span_tracker.get_span(stmt),
            hint="Only assignments, for loops, while loops, if statements, "
            "with statements, returns, break, and continue are supported in DSL functions",
        )

    @_dispatch_statement.register
    def _parse_annotated_assignment_statement(self, stmt: ast.AnnAssign) -> None:
        self.parse_annotated_assignment(stmt)

    @_dispatch_statement.register
    def _parse_assignment_statement(self, stmt: ast.Assign) -> None:
        self.parse_assignment(stmt)

    @_dispatch_statement.register
    def _parse_augmented_assignment_statement(self, stmt: ast.AugAssign) -> None:
        target_load = ast.copy_location(
            type(stmt.target)(
                id=stmt.target.id, ctx=ast.Load(),
            ) if isinstance(stmt.target, ast.Name) else stmt.target,
            stmt.target,
        )
        equivalent = ast.Assign(
            targets=[stmt.target],
            value=ast.BinOp(left=target_load, op=stmt.op, right=stmt.value),
            type_comment=None,
        )
        ast.copy_location(equivalent, stmt)
        ast.fix_missing_locations(equivalent)
        self.parse_assignment(equivalent)

    @_dispatch_statement.register
    def _parse_for_statement(self, stmt: ast.For) -> None:
        self.parse_for_loop(stmt)

    @_dispatch_statement.register
    def _parse_while_statement(self, stmt: ast.While) -> None:
        self.parse_while_loop(stmt)

    @_dispatch_statement.register
    def _parse_if_statement(self, stmt: ast.If) -> None:
        self.parse_if_statement(stmt)

    @_dispatch_statement.register
    def _parse_with_statement(self, stmt: ast.With) -> None:
        self.parse_with_statement(stmt)

    @_dispatch_statement.register
    def _parse_return_statement(self, stmt: ast.Return) -> None:
        self.parse_return(stmt)

    @_dispatch_statement.register
    def _parse_break_statement(self, stmt: ast.Break) -> None:
        self.parse_break(stmt)

    @_dispatch_statement.register
    def _parse_continue_statement(self, stmt: ast.Continue) -> None:
        self.parse_continue(stmt)

    @_dispatch_statement.register
    def _parse_evaluation_statement(self, stmt: ast.Expr) -> None:
        self.parse_evaluation_statement(stmt)

    @_dispatch_statement.register
    def _parse_pass_statement(self, stmt: ast.Pass) -> None:
        pass  # No-op: pass statements are valid in DSL functions

    def _consume_nbuf_pending(self, var: ir.Var, value_expr: ir.Expr | None = None) -> None:
        """Migrate pending tile-group / tile-mutex metadata to id-keyed dicts after builder.let.

        Also propagate tile-mutex metadata through a plain binding ``x = <expr>``
        whose RHS already carries it, so auto_mutex synchronises accesses to
        ``x`` on the SAME buf_id as the tile it aliases (otherwise the access is
        emitted unsynchronised).  Two RHS forms carry it: a tile *slice*
        ``base[off]`` (tagged onto the GetItemExpr in ``parse_subscript``) and a
        direct tile *alias* ``x = tile_a`` (the RHS Var already has meta).
        """
        if self._tile_group_meta_pending is not None:
            self.tile_group_meta[id(var)] = self._tile_group_meta_pending
            self._tile_group_meta_pending = None
        if self._tile_mutex_pending is not None:
            self._tile_mutex_meta[id(var)] = self._tile_mutex_pending
            self._tile_mutex_pending = None
        elif value_expr is not None:
            meta = self._tile_mutex_meta.get(id(value_expr))
            if meta is not None:
                self._tile_mutex_meta[id(var)] = meta

    def _validate_tiling_params(
        self, args_to_process: list[ast.arg], func_def: ast.FunctionDef,
    ) -> None:
        """Pre-validate tiling constraints: at most 1 tiling param, must be last."""
        tiling_param_names = [
            arg.arg for arg in args_to_process
            if arg.annotation is not None and self.resolve_tiling_class(arg.annotation) is not None
        ]
        if len(tiling_param_names) > 1:
            raise ParserSyntaxError(
                f"Function '{func_def.name}' has {len(tiling_param_names)} tiling parameters "
                f"({', '.join(tiling_param_names)}), but at most 1 is allowed",
                span=self.span_tracker.get_span(func_def),
                hint="A kernel may have at most one tiling parameter",
            )
        if len(tiling_param_names) == 1:
            if not args_to_process or args_to_process[-1].arg != tiling_param_names[0]:
                tiling_arg = next(a for a in args_to_process if a.arg == tiling_param_names[0])
                raise ParserSyntaxError(
                    f"Tiling parameter '{tiling_param_names[0]}' must be the last parameter",
                    span=self.span_tracker.get_span(tiling_arg),
                    hint="Move the tiling parameter to the last position",
                )

    def _parse_function_param(self, arg: ast.arg, f: Any) -> None:
        """Parse a single function parameter and register it in scope or tiling registry."""
        param_name = arg.arg
        param_span = self.span_tracker.get_span(arg)

        # 1) tiling-class annotation: always expand via the tiling path (even if a
        # call-site-inferred type exists for this name).
        tiling_cls = self.resolve_tiling_class(arg.annotation) if arg.annotation else None
        if tiling_cls is not None:
            fields = get_tiling_fields(tiling_cls)
            elem_types: list[ir.Type] = []
            for _, field_info in fields.items():
                if isinstance(field_info, ArrayFieldInfo):
                    # T[N] -> a nested TupleType of N homogeneous scalars. Its element
                    # name is NOT registered in IRDebugInfo, so codegen distinguishes array
                    # subscript (tiling.opkind[4]) from struct member access by the missing entry.
                    elem_types.append(
                        ir.TupleType([ir.ScalarType(field_info.dtype)] * field_info.size)
                    )
                else:
                    elem_types.append(ir.ScalarType(field_info.dtype))
            # Single struct parameter: the tiling class is lowered to a named TupleType whose
            # field names live in the IRDebugInfo side table (codegen emits `struct <ClassName>`).
            tuple_type = ir.TupleType(elem_types)
            tiling_var = f.param(param_name, tuple_type, param_span)
            if self.debug_info is not None:
                # Register against the var's actual type object (keyed by pointer in
                # IRDebugInfo), mirroring make_named_tuple's use of the resulting expr type.
                self.debug_info.register_tuple_fields(tiling_var.type, list(fields.keys()))
                # Record the Python tiling class name so codegen emits `struct <ClassName>`
                # (matching the host-side struct) instead of a fixed default.
                self.debug_info.register_tuple_name(tiling_var.type, tiling_cls.__name__)
            self.scope_manager.define_var(param_name, tiling_var, allow_redef=True)
            return

        # 2) Call-site-inferred type (implicit-func path). The inferred type is authoritative;
        # any annotation was already validated for compatibility by _implicit_func_call.
        if param_name in self.inferred_param_types:
            param_var = f.param(param_name, self.inferred_param_types[param_name], param_span)
            self.scope_manager.define_var(param_name, param_var, allow_redef=True)
            if param_name in self._inferred_tile_group_meta:
                self.tile_group_meta[id(param_var)] = self._inferred_tile_group_meta[param_name]
            if param_name in self._inferred_tile_mutex_meta:
                self._tile_mutex_meta[id(param_var)] = self._inferred_tile_mutex_meta[param_name]
            # A compile-time constant int list passed as this argument stays foldable inside the
            # helper body (e.g. ``tile_dims=tile_dims``), so a compile-time-only kwarg resolves to
            # the constant instead of this ir.Var param.
            if param_name in self._inferred_const_lists:
                self._const_lists[param_name] = self._inferred_const_lists[param_name]
            return

        # 3) Annotation only (decorator path: no call site to infer from).
        if arg.annotation is None:
            raise ParserTypeError(
                f"Parameter '{param_name}' missing type annotation",
                span=param_span,
                hint="Add a type annotation like: x: pl.Tensor[[64], pl.DT_FP32]",
            )
        param_type = self.type_resolver.resolve_param_type(arg.annotation, parameter_name=param_name)
        param_type = self._attach_ptr_to_tensor_type(param_name, param_type, param_span)
        param_var = f.param(param_name, param_type, param_span)
        self.scope_manager.define_var(param_name, param_var, allow_redef=True)

    def _hoist_closure_tuples(self, func_def: ast.FunctionDef) -> None:
        """Emit let-bindings for Python closure tuple/list variables at function entry.

        Scans the function AST for Name references that resolve to Python tuples
        or lists in the closure scope. For each unique variable (deduplicated by
        Python object identity), emits an AssignStmt at the current position
        (function entry, before body statements) and caches the resulting IR Var
        in _closure_tuple_ir_cache. Subsequent parse_name() calls return the Var
        instead of an anonymous inline MakeTuple, so the CCE codegen can
        pre-declare the dynamic array at AssignStmt time with correct scoping.
        """
        class _NameCollector(ast.NodeVisitor):
            """Walks AST collecting Name ids, skipping nested FunctionDef bodies."""

            def __init__(self, root_func_def: ast.FunctionDef):
                self.root_func_def = root_func_def
                self.names: list[ast.Name] = []

            def visit(self, node):
                if isinstance(node, ast.FunctionDef) and node is not self.root_func_def:
                    return None
                method = getattr(self, _snake_visit_name(node), None)
                if method is not None:
                    return method(node)
                return super().visit(node)

            def visit_name(self, node: ast.Name) -> None:
                self.names.append(node)

            def visit_function_def(self, node: ast.FunctionDef) -> None:
                self.generic_visit(node)

        collector = _NameCollector(func_def)
        collector.visit(func_def)

        seen: set[tuple[str, int]] = set()
        for name_node in collector.names:
            var_name = name_node.id
            if var_name not in self.expr_evaluator.closure_vars:
                continue
            value = self.expr_evaluator.closure_vars[var_name]
            if not isinstance(value, (tuple, list)):
                continue
            if any(isinstance(v, _ShapePolicy) or v is Ellipsis for v in value):
                continue
            key = (var_name, id(value))
            if key in seen:
                continue
            seen.add(key)
            span = self.span_tracker.get_span(name_node)
            mt_expr = self.expr_evaluator.python_value_to_ir(value, span)
            ir_var = self.builder.let(var_name, mt_expr, span=span)
            self._closure_tuple_ir_cache[key] = ir_var
