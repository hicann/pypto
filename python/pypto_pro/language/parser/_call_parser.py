# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Call and operation parsing helpers for ASTParser."""

from __future__ import annotations

import ast
import logging
from collections import namedtuple
from typing import TYPE_CHECKING, Any

from pypto.pypto_impl import ir
from pypto_pro.ir import op as ir_op
from pypto_pro.ir.op.block_ops import block_ir_op
from pypto_pro.ir.op._op_registry import _OP_REGISTRY

from .diagnostics import (
    InvalidOperationError,
    ParserError,
    ParserSyntaxError,
    ParserTypeError,
    UndefinedVariableError,
    UnsupportedFeatureError,
)

if TYPE_CHECKING:
    from .decorator import KernelFunction


logger = logging.getLogger(__name__)


# Mutex carrier for tile-group tiles (buf_id IR expr, candidate values, memory, dedup id).
_MutexRef = namedtuple("_MutexRef", "buf_id mutex_ids memory slot_id")

# Builtin function names that map to pl.* ops (syntax sugar).
_BUILTIN_TO_OP: dict[str, str] = {
    "min": "min",
    "max": "max",
}


def _dtypes_compatible(a: ir.DataType, b: ir.DataType) -> bool:
    """Lenient dtype compatibility: same dtype, or both integer-like / both float-like.

    Different numeric dtypes of the same family (e.g. INT32 and INT64, FP16 and FP32) are
    interconvertible and treated as compatible.
    """
    return a == b or (a.is_int() and b.is_int()) or (a.is_float() and b.is_float())


def _types_compatible(annotated: ir.Type, actual: ir.Type) -> bool:
    """Lenient type compatibility: same kind + compatible dtype.

    Ignores shape/memref/layout details. ``UnknownType`` is compatible with anything.
    Used to validate a function annotation against the type derived from the actual
    argument/return-value expression.
    """
    if isinstance(annotated, ir.UnknownType) or isinstance(actual, ir.UnknownType):
        return True
    if isinstance(annotated, ir.ScalarType) and isinstance(actual, ir.ScalarType):
        return _dtypes_compatible(annotated.dtype, actual.dtype)
    if isinstance(annotated, ir.PtrType) and isinstance(actual, ir.PtrType):
        return _dtypes_compatible(annotated.dtype, actual.dtype)
    # TensorType / TileType both derive from ShapedType; compare dtype, ignore shape/memref.
    if isinstance(annotated, ir.ShapedType) and isinstance(actual, ir.ShapedType):
        return type(annotated) is type(actual) and _dtypes_compatible(annotated.dtype, actual.dtype)
    if isinstance(annotated, ir.TupleType) and isinstance(actual, ir.TupleType):
        if len(annotated.types) != len(actual.types):
            return False
        return all(_types_compatible(a, b) for a, b in zip(annotated.types, actual.types))
    return False


def _check_type_compatible(annotated: ir.Type, actual: ir.Type, *, what: str, name: str, span) -> None:
    """Raise ParserTypeError if ``annotated`` is not compatible with ``actual``."""
    if not _types_compatible(annotated, actual):
        raise ParserTypeError(
            f"{what} '{name}' annotated as {annotated} but called/returned with {actual}",
            span=span,
            hint="Make the annotation match the actual argument/return type, or remove it.",
        )


def _infer_return_types_from_body(body: ir.Stmt) -> list[ir.Type] | None:
    """Extract return types from the first value-returning ReturnStmt in a function body.

    Recurses into control-flow bodies (if/for) so a helper that returns a value from
    a branch — while an earlier branch does a void `return` (e.g. `return None` on a
    compile-time-dead path) — still resolves its result type.
    """
    if isinstance(body, ir.ReturnStmt):
        if body.value:
            return [v.type for v in body.value]
        return None
    if isinstance(body, ir.SeqStmts):
        for s in body.stmts:
            result = _infer_return_types_from_body(s)
            if result is not None:
                return result
    if isinstance(body, ir.IfStmt):
        for branch in (body.then_body, body.else_body):
            if branch is not None:
                result = _infer_return_types_from_body(branch)
                if result is not None:
                    return result
    if isinstance(body, ir.ForStmt):
        return _infer_return_types_from_body(body.body)
    return None


class CallParserMixin:
    """Mixin containing call and operation parsing methods for ``ASTParser``."""

    @staticmethod
    def _extract_op_name(func: ast.expr) -> str | None:
        """Extract a normalized op name from an attribute-access call node.

        pl.tensor.add  -> "tensor.add"
        pl.add_scalar  -> "add_scalar"
        vf.add         -> "vf.add"  (vf prefix retained)
        bare_name      -> None      (no module prefix)
        """
        attrs: list[str] = []
        node: ast.expr = func
        while isinstance(node, ast.Attribute):
            attrs.insert(0, node.attr)
            node = node.value
        if isinstance(node, ast.Name):
            attrs.insert(0, node.id)
        if len(attrs) < 2:
            return None
        if attrs[0] == "vf":
            return ".".join(attrs)
        return ".".join(attrs[1:])

    @staticmethod
    def _is_docstring(stmt: ast.stmt) -> bool:
        """Check if an AST statement is a docstring (string constant expression)."""
        return (
            isinstance(stmt, ast.Expr)
            and isinstance(stmt.value, ast.Constant)
            and isinstance(stmt.value.value, str)
        )

    @staticmethod
    def _infer_implicit_param_types(
        func_name: str,
        func_params: list[ast.arg],
        arg_exprs: list[ir.Expr],
        sub_parser,
        span,
    ) -> None:
        """Infer helper param types from parsed call-site args and validate annotations."""
        for i, param in enumerate(func_params):
            if param.annotation is not None and sub_parser.resolve_tiling_class(param.annotation):
                continue
            if i >= len(arg_exprs):
                raise ParserTypeError(
                    f"'{func_name}' missing argument for parameter '{param.arg}'",
                    span=span,
                    hint=f"Parameters: {[p.arg for p in func_params]}",
                )
            actual = getattr(arg_exprs[i], "type", None)
            if actual is None or isinstance(actual, ir.UnknownType):
                raise ParserTypeError(
                    f"Cannot infer type of argument for parameter '{param.arg}' of '{func_name}'",
                    span=span,
                    hint="The argument expression must have a resolved DSL type.",
                )
            if param.annotation is not None:
                if sub_parser.type_resolver.annotation_has_shape_policy(param.annotation):
                    sub_parser.type_resolver.validate_policy_parameter_type(
                        param.annotation, param.arg, actual
                    )
                else:
                    annotated = sub_parser.type_resolver.resolve_param_type(param.annotation)
                    _check_type_compatible(annotated, actual, what="Parameter", name=param.arg, span=span)
            sub_parser.inferred_param_types[param.arg] = actual

    @staticmethod
    def _parse_implicit_helper(
        func_name: str, func_def: ast.FunctionDef, sub_parser, span,
        is_vector_function: bool = False,
    ) -> ir.Function:
        """Parse an implicit helper, surfacing the underlying failure with its location."""
        try:
            return sub_parser.parse_function(
                func_def, func_type=ir.FunctionType.Helper,
                is_vector_function=is_vector_function,
            )
        except ParserTypeError as e:
            # Report where in the helper the compile failed. e.span is a normalized dict
            # ({'filename', 'line', ...}); use it for the marker if available.
            inner_span = getattr(e, "span", None)
            loc = ""
            if isinstance(inner_span, dict):
                fname = inner_span.get("filename") or inner_span.get("file")
                line = inner_span.get("line") or inner_span.get("begin_line")
                if line:
                    loc = f" at {fname}:{line}" if fname else f" at line {line}"
            raise UnsupportedFeatureError(
                f"Failed to compile helper '{func_name}' called from the kernel{loc}: {e.message}",
                span=inner_span or span,
                hint="Helper functions are inlined into the kernel, so every statement must be "
                     "valid DSL. Fix the statement above, or give the parameters/return value "
                     f"DSL type annotations (e.g. def {func_name}(x: pl.DT_INT64) -> pl.DT_INT64: ...).",
            ) from e
        except ParserError:
            logger.debug("Re-raising ParserError in _parse_implicit_helper", exc_info=True)
            raise
        except Exception as e:
            inner_span = None
            inner_node = getattr(sub_parser, '_current_node', None)
            if inner_node is not None:
                inner_span = sub_parser.span_tracker.get_span(inner_node)
            raise ParserSyntaxError(
                str(e),
                span=inner_span or span,
                hint="Check your function definition for errors",
            ) from e

    @staticmethod
    def _validate_implicit_returns(func_name: str, ir_func: ir.Function, span) -> ir.Function:
        """Replace annotated helper return types with inferred body return types after validation."""
        inferred = _infer_return_types_from_body(ir_func.body) if ir_func.body else None
        if inferred is None:
            return ir_func
        for annotated, actual in zip(list(ir_func.return_types), inferred):
            _check_type_compatible(annotated, actual, what="Return value of", name=func_name, span=span)
        return ir.Function(
            ir_func.name,
            list(ir_func.params),
            inferred, ir_func.body, ir_func.span, ir_func.func_type)

    @staticmethod
    def _is_vf_op_call(call_node: ast.expr) -> str | None:
        """Check if an AST node is a ``vf.xxx(...)`` call.

        Returns the VF op name (e.g. ``"add"``) if yes, ``None`` otherwise.
        """
        if not isinstance(call_node, ast.Call):
            return None
        op_name = CallParserMixin._extract_op_name(call_node.func)
        if op_name is None or not op_name.startswith("vf."):
            return None
        return op_name[3:]  # strip "vf." prefix

    @classmethod
    def _make_call_with_return_type(
        cls,
        op: ir.Op,
        args: list[ir.Expr],
        return_types: list[ir.Type],
        span: ir.Span,
    ) -> ir.Expr:
        """Create an ir.Call, attaching the return type when known.

        Args:
            op: Op identifying the callee
            args: Parsed argument expressions
            return_types: The callee's return type list (may be empty)
            span: Source span for the call
        """
        if not return_types:
            return ir.Call(op, args, span)
        if len(return_types) == 1:
            return ir.Call(op, args, return_types[0], span)
        return ir.Call(op, args, ir.TupleType(return_types), span)

    @classmethod
    def _retrieve_function_source(
        cls, func_name: str, fn: Callable, span: ir.Span, decorator_hint: str,
    ) -> tuple[str, list[str], int, int, ast.FunctionDef]:
        """Retrieve source, parse AST, and locate FunctionDef for a callable.

        Returns (source_file, source_lines, line_offset, col_offset, func_def).
        """
        import textwrap as _tw

        from .decorator import _get_source_info

        try:
            source_file, source_lines_raw, starting_line = _get_source_info(fn, "function")
        except Exception as e:
            raise UnsupportedFeatureError(
                f"Cannot compile '{func_name}': unable to retrieve source -{e}",
                span=span,
                hint=f"Define '{func_name}' in a .py file, or use {decorator_hint}",
            ) from e

        source_code = _tw.dedent("".join(source_lines_raw))
        col_offset = len(source_lines_raw[0]) - len(source_lines_raw[0].lstrip()) if source_lines_raw else 0
        line_offset = starting_line - 1
        source_lines = source_code.split("\n")

        try:
            tree = ast.parse(source_code)
        except SyntaxError as e:
            raise UnsupportedFeatureError(
                f"Cannot parse '{func_name}': {e}",
                span=span,
                hint=f"Use {decorator_hint} to explicitly mark '{func_name}'",
            ) from e

        func_def = next(
            (n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == fn.__name__),
            None,
        )
        if func_def is None:
            raise UnsupportedFeatureError(
                f"Cannot find function definition for '{func_name}' in source",
                span=span,
                hint=f"Use {decorator_hint} to explicitly mark '{func_name}'",
            )

        return source_file, source_lines, line_offset, col_offset, func_def

    @classmethod
    def _build_function_closure(cls, fn: Callable) -> dict[str, Any]:
        """Build closure dict from a callable's globals and free variables."""
        fn_closure: dict[str, Any] = {**fn.__globals__}
        if fn.__closure__ and fn.__code__.co_freevars:
            fn_closure.update(
                dict(zip(fn.__code__.co_freevars, (c.cell_contents for c in fn.__closure__)))
            )
        return fn_closure

    @classmethod
    def _resolve_auto_mutex_pipe(cls, op_name: str, tilerefs: list):
        """Determine the pipe for auto_mutex from op_name and tile memory spaces."""
        from ._op_pipeline import get_move_pipe, get_op_pipe, get_store_pipe

        if op_name == "move":
            dst_mem = tilerefs[0].memory if tilerefs[0] else None
            src_mem = tilerefs[1].memory if len(tilerefs) > 1 and tilerefs[1] else None
            if src_mem is not None and dst_mem is not None:
                return get_move_pipe(src_mem, dst_mem)
            return None
        if op_name in ("store", "store_tile"):
            src_mem = tilerefs[1].memory if len(tilerefs) > 1 and tilerefs[1] else None
            if src_mem is not None:
                return get_store_pipe(src_mem)
            return None
        return get_op_pipe(op_name)

    @classmethod
    def _init_vf_kwarg_enums(cls):
        if cls._VF_KWARG_ENUMS:
            return
        from pypto.ir import (
            BinType, CastLayout, CompareMode, DataCopyMode, DuplicatePos,
            HistType, IndexOrder, LoadDist, MaskLoadDist, MaskPattern, MaskStoreDist,
            MaskWidth, MemBarMode, MergeMode, PackPart,
            ReduceMode, VFRoundMode, SaturateMode, SqueezeMode, StoreDist,
        )
        cls._VF_KWARG_ENUMS = {
            "pattern": (MaskPattern,),
            "mode": (MergeMode, MemBarMode, MaskLoadDist),
            "merge_mode": (MergeMode,),
            "reduce_mode": (ReduceMode,),
            "reduce_type": (ReduceMode,),
            "cmp_mode": (CompareMode,),
            "pos": (DuplicatePos,),
            "layout": (CastLayout,),
            "round_mode": (VFRoundMode,),
            "saturate": (SaturateMode,),
            "bin_type": (BinType,),
            "hist_type": (HistType,),
            "gather_mode": (SqueezeMode,),
            "half": (PackPart,),
            "part": (PackPart,),
            "width": (MaskWidth,),
            "dist": (LoadDist, StoreDist, MaskStoreDist),
            "data_copy_mode": (DataCopyMode,),
            "index_order": (IndexOrder,),
        }

    @classmethod
    def _get_vf_op_dst_count(cls, op_name: str) -> int | None:
        """Return the number of dst registers for a VF op, or None if unknown."""
        return cls._VF_OP_DST_COUNT.get(op_name)

    def parse_call(self, call: ast.Call) -> ir.Expr:
        """Parse function call.

        Args:
            call: Call AST node

        Returns:
            IR expression from call
        """
        func = call.func

        # Handle cross-function calls via self.method_name() in @pl.program classes
        if isinstance(func, ast.Attribute):
            # Check for self.method_name pattern
            if isinstance(func.value, ast.Name) and func.value.id == "self":
                method_name = func.attr
                if method_name in self.global_vars:
                    func_obj = self.gvar_to_func.get(method_name)
                    args = [self.parse_expression(arg) for arg in call.args]
                    span = self.span_tracker.get_span(call)

                    # Use return type from the parsed function if available
                    return_types = func_obj.return_types if func_obj else []
                    op = ir.Op(method_name)
                    return self._make_call_with_return_type(op, args, return_types, span)
                else:
                    raise UndefinedVariableError(
                        f"Function '{method_name}' not defined in program",
                        span=self.span_tracker.get_span(call),
                        hint=f"Available functions: {sorted(self.global_vars)}",
                    )

            # Handle pl.tensor.*, pl.system.*, and pl.* operation calls
            return self.parse_op_call(call)

        # Handle bare-name calls to external ir.Function, KernelFunction, or implicit func
        if isinstance(func, ast.Name):
            from .decorator import KernelFunction  # circular import

            func_name = func.id

            # Builtin min/max -> route to pl.min/pl.max (scalar_ops.py)
            if func_name in _BUILTIN_TO_OP:
                op_func = _OP_REGISTRY.get(_BUILTIN_TO_OP[func_name])
                if op_func is not None:
                    return op_func(self, call)

            resolved = self.expr_evaluator.closure_vars.get(func_name)
            if isinstance(resolved, ir.Function):
                return self._parse_external_function_call(func_name, resolved, call)
            if isinstance(resolved, KernelFunction):
                return self._parse_func_call(func_name, resolved, call)
            if callable(resolved) and not isinstance(resolved, type):
                return self._implicit_func_call(func_name, resolved, call)

        raise UnsupportedFeatureError(
            f"Unsupported function call: {ast.unparse(call)}",
            span=self.span_tracker.get_span(call),
            hint="Use pl.* operations, self.method() for cross-function calls, "
            "or call an external @pl.function by name",
        )

    def parse_op_call(self, call: ast.Call) -> ir.Expr:
        """Parse operation call like pl.tensor.create_tensor() or pl.add().

        Args:
            call: Call AST node

        Returns:
            IR expression from operation
        """
        result = self._route_ir_node_method(call)
        if result is not None:
            return result

        op_name = self._extract_op_name(call.func)
        span = self.span_tracker.get_span(call)
        if op_name is None:
            raise UnsupportedFeatureError(
                f"Unsupported operation call: {ast.unparse(call)}",
                span=span,
                hint="Use pl.*, pl.tensor.*, or pl.system.* operations",
            )

        if op_name == "constexpr":
            raise ParserSyntaxError(
                "pl.constexpr() can only be used as an 'if' condition or in a ternary expression, "
                "e.g. 'if pl.constexpr(cond):' or 'x = a if pl.constexpr(cond) else b'",
                span=span,
                hint="Use 'pl.constexpr(condition):' in an if statement or ternary expression — "
                     "constexpr is not allowed in for/while/with/break/continue/return or as a standalone statement",
            )

        if self._auto_mutex and not op_name.startswith("vf."):
            self._emit_auto_mutex(op_name, call, span)

        op_func = _OP_REGISTRY.get(op_name)
        if op_func is not None:
            return op_func(self, call)

        return self._default_op_func(op_name, call)

    def parse_op_kwargs(self, call: ast.Call) -> dict[str, Any]:
        """Parse keyword arguments for an operation call."""
        return {kw.arg: self.resolve_single_kwarg(kw.arg, kw.value) for kw in call.keywords}

    # Mapping of VF kwarg names to their expected enum classes (tuple of types).
    # When a kwarg value resolves to an instance of any mapped enum, its .value
    # VF enum kwarg validation: if a kwarg is mapped to enum classes, the parser
    # passes the enum object through to ConvertKwargsDict (which extracts .value as int).
    # If a raw string is passed for a mapped kwarg, the parser raises an error.
    _VF_KWARG_ENUMS: dict[str, tuple] = {}

    def resolve_single_kwarg(self, key: str, value: ast.expr) -> Any:
        """Resolve a single keyword argument value to a Python or IR value."""
        if key == "dtype":
            return self.type_resolver.resolve_dtype(value)
        if isinstance(value, ast.Constant):
            result = value.value
        elif isinstance(value, ast.UnaryOp) and isinstance(value.op, ast.USub):
            result = self._resolve_unary_kwarg(value)
        elif isinstance(value, ast.Name):
            result = self._resolve_name_kwarg(value)
        elif isinstance(value, ast.Attribute):
            result = self._resolve_attribute_kwarg(value)
        elif isinstance(value, ast.List):
            result = self._resolve_list_kwarg(value)
        else:
            result = self.parse_expression(value)

        # If this kwarg expects a VF enum and the resolved value is an enum
        # instance, return it directly — ConvertKwargsDict extracts .value (int).
        # If a raw string is passed for a mapped kwarg, raise an error.
        if not self._VF_KWARG_ENUMS:
            self._init_vf_kwarg_enums()
        enum_classes = self._VF_KWARG_ENUMS.get(key)
        if enum_classes is not None:
            if isinstance(result, enum_classes):
                return result
            if isinstance(result, str):
                all_members = []
                for cls in enum_classes:
                    members = getattr(cls, "__members__", None)
                    if members is not None:
                        all_members.extend(f"pl.{cls.__name__}.{name}" for name in members)
                    else:
                        try:
                            all_members.extend(f"pl.{cls.__name__}.{m.name}" for m in cls)
                        except TypeError:
                            pass
                enum_names = ", ".join(c.__name__ for c in enum_classes)
                hint_str = ", ".join(all_members[:8]) if all_members else enum_names
                if len(all_members) > 8:
                    hint_str += ", ..."
                raise ParserTypeError(
                    f"VF kwarg '{key}' requires an enum value, got string \"{result}\". "
                    f"Use one of: {hint_str}",
                    span=self.span_tracker.get_span(value),
                )
        return result

    def resolve_const_int_list_kwarg(self, call: ast.Call, key: str) -> "list[int] | None":
        """Resolve a compile-time constant int-list kwarg (e.g. ``tile_dims``) to a list of ints.

        Folds a literal list, a var bound to a constant list, or a helper parameter that carries
        one (via _const_lists). Raises ParserTypeError if the kwarg is present but not a
        compile-time constant int list. Returns None if the kwarg is absent.
        """
        for kw in call.keywords:
            if kw.arg != key:
                continue
            value = kw.value
            if isinstance(value, ast.List):
                return [self.resolve_static_int(elt) for elt in value.elts]
            if isinstance(value, ast.Name) and value.id in self._const_lists:
                return list(self._const_lists[value.id])
            success, result = self.expr_evaluator.try_eval_expr(value)
            if (
                success
                and isinstance(result, (list, tuple))
                and all(isinstance(x, int) and not isinstance(x, bool) for x in result)
            ):
                return list(result)
            raise ParserTypeError(
                f"'{key}' must be a compile-time constant integer list, got '{ast.unparse(value)}'",
                span=self.span_tracker.get_span(value),
                hint=f"{key} selects tensor axes at compile time; pass a constant list "
                f"(e.g. {key}=[1, 3]) or a variable bound to one, not a runtime value.",
            )
        return None

    def resolve_static_int(self, elt: ast.expr) -> int:
        """Resolve a single AST node to a compile-time ``int``, folding in-body
        scalar constants (``valid_m = 128``) and closure constants.

        Raises if not a constant.
        """
        if isinstance(elt, ast.Constant) and isinstance(elt.value, int) and not isinstance(elt.value, bool):
            return elt.value
        if isinstance(elt, ast.UnaryOp) and isinstance(elt.op, ast.USub):
            return -self.resolve_static_int(elt.operand)
        if isinstance(elt, ast.Name) and elt.id in self._const_scalars:
            return self._const_scalars[elt.id]
        success, val = self.expr_evaluator.try_eval_expr(elt)
        if success and isinstance(val, int) and not isinstance(val, bool):
            return val
        raise ParserTypeError(
            f"'{ast.unparse(elt)}' is not a compile-time integer constant",
            span=self.span_tracker.get_span(elt),
            hint="A constant list kwarg (e.g. TileType shape/valid_shape) must be compile-time "
            "constants; use pl.set_validshape() for a runtime valid shape.",
        )

    def _default_op_func(self, op_name: str, call: ast.Call) -> ir.Expr:
        if op_name.startswith("vf."):
            return self._parse_vf_op(op_name[3:], call)
        return self._parse_block_default(op_name, call)

    def _parse_external_function_call(
        self, _local_name: str, ext_func: ir.Function, call: ast.Call
    ) -> ir.Expr:
        """Parse a call to an externally-defined ir.Function.

        Args:
            _local_name: The name used in the caller's scope (may be aliased)
            ext_func: The external ir.Function object
            call: The AST Call node
        """
        func_name = ext_func.name
        span = self.span_tracker.get_span(call)

        # Validate no naming conflict with internal program functions
        if func_name in self.global_vars:
            raise ParserSyntaxError(
                f"External function '{func_name}' conflicts with program function '{func_name}'",
                span=span,
                hint="Rename either the external or program function to avoid the name conflict",
            )

        # Check for conflicting externals with same .name but different objects
        if func_name in self.external_funcs and self.external_funcs[func_name] is not ext_func:
            raise ParserSyntaxError(
                f"Conflicting external functions with name '{func_name}'",
                span=span,
                hint="External functions must have unique names; rename one of the functions",
            )

        # Track the external function
        self.external_funcs[func_name] = ext_func

        args = [self.parse_expression(arg) for arg in call.args]
        op = ir.Op(func_name)
        return self._make_call_with_return_type(op, args, ext_func.return_types, span)

    def _make_implicit_sub_parser(self, fn: Callable, source_info: tuple):
        """Create a sub-parser for an implicit helper function."""
        from ._ast_parser import ASTParser

        source_file, source_lines, line_offset, col_offset, _ = source_info
        fn_closure = self._build_function_closure(fn)
        merged_closure = {**fn_closure, **self.expr_evaluator.closure_vars}
        sub_parser = ASTParser(
            source_file,
            source_lines,
            line_offset,
            col_offset,
            closure_vars=merged_closure,
        )
        # Share cache so nested implicit calls are deduplicated.
        sub_parser.implicit_func_cache = self.implicit_func_cache
        sub_parser.kfunc_vf_used_params = self.kfunc_vf_used_params
        # One Program keeps a single IRDebugInfo: the sub-parser registers tuple/struct
        # field names into the parent's side table, so cross-function field access resolves names.
        sub_parser.debug_info = self.debug_info
        return sub_parser

    def _transfer_implicit_tile_metadata(
        self,
        func_params: list[ast.arg],
        arg_exprs: list[ir.Expr],
        sub_parser,
    ) -> None:
        """Transfer tile-group and tile-mutex metadata into an implicit helper parser."""
        for i, param in enumerate(func_params):
            if i >= len(arg_exprs):
                continue
            group_meta = self.tile_group_meta.get(id(arg_exprs[i]))
            if group_meta is not None:
                sub_parser.set_inferred_tile_group_meta(param.arg, group_meta)
            tile_meta = self._tile_mutex_meta.get(id(arg_exprs[i]))
            if tile_meta is not None:
                sub_parser.set_inferred_tile_mutex_meta(param.arg, tile_meta)
        sub_parser.set_auto_mutex_enabled(self.auto_mutex_enabled)

    def _transfer_implicit_const_args(
        self,
        func_params: list[ast.arg],
        call_args: list[ast.expr],
        sub_parser,
    ) -> None:
        """Thread compile-time constant int-list arguments into an implicit helper.

        A compile-time-only list kwarg (e.g. ``tile_dims``) threaded through a helper parameter
        would otherwise resolve to an ir.Var inside the helper body, which the load/store offset
        math cannot consume. Record the constant under the parameter name so _register_param can
        re-expose it via _const_lists (matched positionally; keyword call args are skipped).
        """
        for param, arg in zip(func_params, call_args):
            const_list: list[int] | None = None
            if isinstance(arg, ast.Name) and arg.id in self._const_lists:
                const_list = list(self._const_lists[arg.id])
            elif isinstance(arg, ast.List):
                try:
                    const_list = [self.resolve_static_int(elt) for elt in arg.elts]
                except ParserTypeError:
                    const_list = None
            if const_list is not None:
                sub_parser.set_inferred_const_list(param.arg, const_list)

    def _register_implicit_kernel_function(
        self,
        fn: Callable,
        func_def: ast.FunctionDef,
        ir_func: ir.Function,
        sub_parser,
        is_vector_function: bool = False,
    ):
        """Create and register the KernelFunction for an implicit helper."""
        from .decorator import KernelFunction

        param_names = [a.arg for a in func_def.args.args if a.arg != "self"]
        kfunc = KernelFunction(
            name=fn.__name__,
            ir_function=ir_func,
            op=ir.Op(fn.__name__),
            param_names=param_names,
        )
        self.implicit_func_cache[id(fn)] = kfunc
        if is_vector_function:
            # @pl.vector_function: the entire body is a VF section, so every
            # tile-valued param needs mutex_lock/unlock(V) around the func.call.
            self.kfunc_vf_used_params[id(kfunc)] = set(param_names)
        else:
            self.kfunc_vf_used_params[id(kfunc)] = None
        self.external_funcs.update(sub_parser.external_funcs)
        self.external_funcs[fn.__name__] = ir_func
        return kfunc

    def _implicit_func_call(self, func_name: str, fn: Callable, call: ast.Call) -> ir.Expr:
        """Compile an annotated plain Python function as a KernelFunction and emit func.call.

        Functions with complete DSL type annotations are compiled on first encounter and
        cached by id(fn). Subsequent calls reuse the cached KernelFunction.

        Functions without annotations raise UnsupportedFeatureError with a helpful hint.

        Args:
            func_name: Name used at the call site
            fn: The callable Python function
            call: AST Call node

        Returns:
            IR expression (func.call result)
        """
        span = self.span_tracker.get_span(call)

        fn_id = id(fn)
        if fn_id in self.implicit_func_cache:
            return self._parse_func_call(func_name, self.implicit_func_cache[fn_id], call)

        from .decorator import is_vector_function
        is_vf = is_vector_function(fn)
        is_pipeline_stage = getattr(fn, "pipeline_stage", False)
        decorator_name = (
            "@pl.vector_function" if is_vf
            else "@pl.pipeline.stage" if is_pipeline_stage
            else "helper function"
        )
        source_info = self._retrieve_function_source(func_name, fn, span, decorator_name)
        func_def = source_info[4]
        sub_parser = self._make_implicit_sub_parser(fn, source_info)
        if is_vf or is_pipeline_stage:
            # VF/stage helpers are statement-like: void-only, and no early/multiple return.
            sub_parser.set_void_return_mode("@pl.vector_function" if is_vf else "@pl.pipeline.stage")

        # Parse all call-site args once; derive every param type from the parsed expr.
        # The annotation (if any) is only validated here for compatibility — never used as the
        # param type. Validation lives here (outside parse_function's try/except below) so a
        # clear mismatch error is not reworded into the generic "no annotations" message.
        # tiling-class params keep the annotation-driven expansion path and are skipped here.
        func_params = [a for a in func_def.args.args if a.arg != "self"]
        arg_exprs = [self.parse_expression(a) for a in call.args]
        self._infer_implicit_param_types(func_name, func_params, arg_exprs, sub_parser, span)

        # Transfer tile-group / tile-mutex metadata and auto_mutex so that
        # .next()/.current()/.previous() work on group handles passed into helpers, and
        # auto_mutex can lock bare tiles passed into helpers (valid after inlining).
        self._transfer_implicit_tile_metadata(func_params, arg_exprs, sub_parser)
        self._transfer_implicit_const_args(func_params, call.args, sub_parser)
        ir_func = self._parse_implicit_helper(
            func_name, func_def, sub_parser, span, is_vector_function=is_vf,
        )

        # Return types are derived from the body's return values; the annotated return
        # type (if any) is only validated for compatibility. Both a `tuple[...]` annotation
        # and a `return a, b` body now resolve to a single TupleType, so they line up.
        ir_func = self._validate_implicit_returns(func_name, ir_func, span)
        kfunc = self._register_implicit_kernel_function(
            fn, func_def, ir_func, sub_parser, is_vector_function=is_vf,
        )
        return self._parse_func_call(func_name, kfunc, call, parsed_args=arg_exprs)

    def _parse_func_call(
        self,
        func_name: str,
        kfunc: "KernelFunction",
        call: ast.Call,
        parsed_args: list[ir.Expr] | None = None,
    ) -> ir.Expr:
        """Parse a call to an internal helper function, emitting an ir.Call.

        Args:
            func_name: Name used at the call site
            kfunc: KernelFunction holding the compiled ir.Function
            call: AST Call node
            parsed_args: Already-parsed arg exprs (in positional order) to reuse instead of
                re-parsing ``call.args``

        Returns:
            ir.Call expression with the function's Op and parsed arguments
        """
        span = self.span_tracker.get_span(call)
        expected = len(kfunc.param_names)
        got = len(call.args)
        if got != expected:
            raise ParserTypeError(
                f"Function '{func_name}' expects {expected} argument(s), got {got}",
                span=span,
                hint=f"Parameters: {kfunc.param_names}",
            )

        locked_vf_refs: list = []
        if self._auto_mutex:
            vf_used = self.kfunc_vf_used_params.get(id(kfunc))
            if vf_used is not None:
                arg_tilerefs = [self._try_resolve_tileref(arg) for arg in call.args]
                locked_vf_refs = self._emit_vf_func_mutex_lock(
                    kfunc.param_names, arg_tilerefs, vf_used, span,
                )

        arg_exprs = (
            parsed_args if parsed_args is not None
            else [self.parse_expression(arg) for arg in call.args]
        )
        return_types = list(kfunc.ir_function.return_types)
        call_expr = self._make_call_with_return_type(kfunc.op, arg_exprs, return_types, span)

        if locked_vf_refs:
            self._pending_mutex_unlocks = (locked_vf_refs, ir.PipeType.V, span)

        return call_expr

    # -------------------------------------------------------------------------
    # Keyword argument resolution helpers
    # -------------------------------------------------------------------------

    def _resolve_unary_kwarg(self, value: ast.UnaryOp) -> Any:
        """Resolve a unary op kwarg value (e.g., -1)."""
        if isinstance(value.operand, ast.Constant) and isinstance(value.operand.value, (int, float)):
            return -value.operand.value
        return self.parse_expression(value)

    def _resolve_name_kwarg(self, value: ast.Name) -> Any:
        """Resolve a Name kwarg value via scope lookup or closure eval."""
        if value.id in ["True", "False"]:
            return value.id == "True"
        if self.scope_manager.lookup_var(value.id) is not None:
            return self.parse_expression(value)  # IR var from scope
        # Not in IR scope -evaluate from closure (raises ParserTypeError if undefined)
        return self.expr_evaluator.eval_expr(value)

    def _resolve_attribute_kwarg(self, value: ast.Attribute) -> Any:
        """Resolve an Attribute kwarg value (e.g., pl.DT_FP32, config.field)."""
        try:
            return self.type_resolver.resolve_dtype(value)
        except ParserTypeError:
            return self.expr_evaluator.eval_expr(value)

    def _resolve_list_kwarg(self, value: ast.List) -> Any:
        """Resolve a List kwarg value, trying closure eval first."""
        if any(
            isinstance(elt, ast.Name)
            and self.scope_manager.lookup_var(elt.id) is not None
            for elt in value.elts
        ):
            return self.parse_list(value)
        success, result = self.expr_evaluator.try_eval_expr(value)
        if success and isinstance(result, list):
            return result
        return self.parse_list(value)


    # -------------------------------------------------------------------------
    # VF ops
    # -------------------------------------------------------------------------

    # VF op names are now unified to snake_case across Python API, IR, and C++
    # backend. This map is kept for potential future name aliasing; currently
    # all entries are identity (Python name == IR op name).
    _VF_OP_NAME_MAP: dict[str, str] = {}


    def _parse_vf_op(self, op_name: str, call: ast.Call) -> ir.Expr:
        """Parse a VF API operation call: vf.{op_name}(...).

        VF ops directly emit VF instructions. Arguments and kwargs are passed
        through to ir.create_op_call with the "vf." prefix.

        Note: auto_mutex is NOT applied per-vf-op here, because bisheng requires
        VEC_SCOPE to contain only VF instructions (get_buf/rls_buf are plain CCE
        intrinsics and cause "Do not know how to expand this operator's operand"
        errors if emitted inside the scope). Instead, auto_mutex wraps the whole
        VF inline-function call at its call site -see _parse_inline_call.

        Args:
            op_name: Name of the VF operation (without ``vf.`` prefix).
            call: Call AST node.

        Returns:
            IR expression for the VF op call.
        """
        span = self.span_tracker.get_span(call)

        # Reject the legacy statement form `vf.xxx(dst, ...)` for ops that produce
        # a result. Only store/side-effect ops (dst_count == 0) may be called as a
        # bare statement; all others must use the assignment form `dst = vf.xxx(...)`.
        # (The assignment form is handled in _assignment_parser, which never routes
        # through here — so reaching this point means it's a statement-form call.)
        dst_count = self._VF_OP_DST_COUNT.get(op_name)
        if dst_count is not None and dst_count > 0:
            if dst_count == 1:
                correct = f"dst = vf.{op_name}(...)"
            else:
                dst_list = ", ".join(f"dst{i}" for i in range(dst_count))
                correct = f"{dst_list} = vf.{op_name}(...)"
            raise ParserSyntaxError(
                f"vf.{op_name} produces a result and must use the assignment form. "
                f"The statement form vf.{op_name}(dst, ...) is no longer supported.",
                span=span,
                hint=f"Use: {correct}",
            )

        # Block direct use of vf.reg_tensor — registers are now declared implicitly
        # by the assignment form (dst = vf.xxx(...)); users cannot call it directly.
        if op_name == "reg_tensor":
            raise ParserSyntaxError(
                "vf.reg_tensor cannot be called directly. VF registers are declared "
                "automatically by the assignment form.",
                span=span,
                hint="Use: dst = vf.load_align(...)  # or any VF compute op",
            )

        args = [self.parse_expression(arg) for arg in call.args]
        kwargs = self.parse_op_kwargs(call)
        backend_op_name = self._VF_OP_NAME_MAP.get(op_name, op_name)

        return ir.create_op_call(f"vf.{backend_op_name}", args, kwargs, span)

    # --- VF assignment-form support --------------------------------------------

    # Number of destination registers at the front of the arg list for each VF op.
    # 0 = no dst (store/side-effect ops, or already return-value ops like compare).
    # 1 = single dst at args_[0].
    # 2 = two dsts at args_[0], args_[1].
    _VF_OP_DST_COUNT: dict[str, int] = {
        # 1 dst
        "add": 1, "sub": 1, "mul": 1, "div": 1, "max": 1, "min": 1,
        "and_": 1, "or_": 1, "xor": 1, "abs_sub": 1, "select": 1,
        "shift_left": 1, "shift_right": 1, "prelu": 1,
        "mask_and": 1, "mask_or": 1, "mask_xor": 1, "mask_not": 1,
        "ln": 1, "log": 1, "exp": 1, "abs": 1, "not_": 1, "sqrt": 1,
        "relu": 1, "neg": 1, "copy": 1, "pair_reduce_sum": 1,
        "squeeze": 1, "truncate": 1, "astype": 1, "log2": 1, "log10": 1,
        "reduce_sum": 1, "reduce_max": 1, "reduce_min": 1,
        "muls": 1, "adds": 1, "subs": 1, "mins": 1, "maxs": 1,
        "leaky_relu": 1, "muls_cast": 1,
        "axpy": 1, "exp_sub": 1, "mul_add_dst": 1, "mul_dst_add": 1,
        "pack": 1, "unpack": 1, "mask_pack": 1, "mask_unpack": 1,
        "mask_mov": 1, "arange": 1, "unsqueeze": 1, "full": 1,
        "load_align": 1, "load": 1, "load_unalign": 1,
        "gather": 1,
        "mask_sel": 1, "move": 1,
        # 2 dsts
        "interleave": 2, "de_interleave": 2,
        "mask_interleave": 2, "mask_deinterleave": 2,
        "mull": 2,
        # 0 dst (no assignment form)
        "store_align": 0, "store_unalign": 0, "store_unalign_post": 0,
        "scatter": 0, "store": 0, "mask_store": 0, "mask_store_unalign": 0,
        "mem_bar": 0, "clear_spr": 0,
        "eq": 0, "ne": 0, "lt": 0, "gt": 0, "le": 0, "ge": 0,
        "addc": 0, "subc": 0,
        "load_unalign_pre": 0,
        "store_align_pack": 0, "store_align_intlv": 0,
        "store_align_pack_postupdate": 0,
    }

    # --- auto_mutex helpers ---------------------------------------------------

    def _try_resolve_tileref(self, node: ast.expr):
        """Resolve a bare tile var (from group.next()/current()/previous()) to a mutex ref.

        Returns _MutexRef(buf_id, mutex_ids, memory, tile_id) when ``node`` names a
        tile variable carrying tile-group mutex metadata; otherwise None.

        A tile *slice* argument (``tile[off]``, possibly nested) aliases the base
        tile's buffer, so peel the subscript(s) to the base Name and lock on the
        base's mutex_id -- otherwise auto_mutex skips the access and emits it
        unsynchronised.
        """
        while isinstance(node, ast.Subscript):
            node = node.value
        if not isinstance(node, ast.Name):
            return None
        var = self.scope_manager.lookup_var(node.id)
        if not isinstance(var, ir.Expr):
            return None
        meta = self._tile_mutex_meta.get(id(var))
        if meta is None:
            return None
        buf_id_ir, mutex_ids = meta
        mem = var.type.memref.memory_space_ if isinstance(var.type, ir.TileType) and var.type.memref else None
        return _MutexRef(buf_id_ir, mutex_ids, mem, id(var))

    def _emit_auto_mutex(self, op_name: str, call: ast.Call, span: ir.Span):
        """Emit mutex_lock before and mutex_unlock after a block DSL op.

        Scans call.args for tile-group tiles, determines the op pipe,
        and emits lock/unlock per unique slot.
        Returns None -the caller still parses the op normally.

        Phase-aware skip on Acc tiles: when matmul/matmul_acc/store carries
        phase="partial"/"final", the cube/fixp handshake on the Acc-memory
        accumulator is taken over by the hardware unit_flag bit
        (AccPhase/STPhase). The software mutex on the Acc buf is redundant
        *and* occupies M-pipe / FIX-pipe instruction slots that otherwise
        let cube/fixp run back-to-back. Skip it.

        Non-Acc tiles (L0A / L0B / L1) MUST keep their mutex: unit_flag does
        NOT cover the MTE1 -> cube path. Removing L0A/L0B mutex causes RAW
        between MTE1 finishing the move and cube starting to read
        (verified: device error 507015 on 2026-05-29).
        """
        from pypto_pro.ir.op.system_ops import mutex_lock

        from ._op_pipeline import op_accesses_buffer

        # Descriptor-only ops (e.g. set_validshape) rewrite tile metadata but never
        # touch buffer data, so their accesses cannot race -> no buffer mutex needed.
        if not op_accesses_buffer(op_name):
            return

        # 1. Build unique_refs: scan args for slot.tile mutex refs, dedup by slot,
        #    then drop Acc tiles when a phase-aware matmul/store carries the
        #    unit_flag (the hardware handshake replaces the software mutex there).
        tilerefs = [self._try_resolve_tileref(arg) for arg in call.args]
        unique_refs = []
        seen = set()
        for tref in tilerefs:
            if tref is None or tref.buf_id is None:
                continue
            if tref.slot_id in seen:
                continue
            seen.add(tref.slot_id)
            unique_refs.append(tref)

        if op_name in ("matmul", "matmul_acc", "store"):
            phase = None
            for kw in call.keywords:
                if kw.arg == "phase":
                    resolved = self._resolve_attribute_kwarg(kw.value)
                    if hasattr(resolved, "value"):
                        phase = resolved.value
                    break
            # STPhase/AccPhase enum: Partial/Final indicates multi-step accumulation
            if phase in (ir.STPhase.Partial, ir.STPhase.Final):
                unique_refs = [r for r in unique_refs if r.memory != ir.MemorySpace.Acc]

        if not unique_refs:
            return

        # 2. Determine pipe
        pipe = self._resolve_auto_mutex_pipe(op_name, tilerefs)
        if pipe is None:
            return

        # 3. Emit lock for each unique _TileRef
        for tref in unique_refs:
            lock_expr = mutex_lock(pipe=pipe, mutex_id=tref.buf_id, mutex_ids=tref.mutex_ids, span=span)
            self.builder.emit(ir.EvalStmt(lock_expr, span))

        # Store for post-op unlock emission
        self._pending_mutex_unlocks = (unique_refs, pipe, span)

    def _emit_auto_mutex_unlocks(self):
        """Emit mutex_unlock calls queued by _emit_auto_mutex or _parse_func_call."""
        if not hasattr(self, "_pending_mutex_unlocks") or self._pending_mutex_unlocks is None:
            return
        from pypto_pro.ir.op.system_ops import mutex_unlock

        unique_refs, pipe, span = self._pending_mutex_unlocks
        for tref in unique_refs:
            unlock_expr = mutex_unlock(pipe=pipe, mutex_id=tref.buf_id, mutex_ids=tref.mutex_ids, span=span)
            self.builder.emit(ir.EvalStmt(unlock_expr, span))
        self._pending_mutex_unlocks = None

    def _emit_vf_func_mutex_lock(
        self,
        param_names: list,
        arg_tilerefs: list,
        used_params: set,
        span: ir.Span,
    ) -> list:
        """Emit mutex_lock(V, buf_id) for tile-valued args whose matching
        parameter is referenced inside a ``@pl.vector_function`` body.

        Called before a VF func.call is emitted (i.e., before the VEC_SCOPE
        is generated). Returns the list of locked slot refs, so
        the caller can emit matching unlocks after the call.
        """
        from pypto_pro.ir.op.system_ops import mutex_lock

        unique_refs = []
        seen = set()
        for param_name, tref in zip(param_names, arg_tilerefs):
            if tref is None or tref.buf_id is None:
                continue
            if param_name not in used_params:
                continue
            if tref.slot_id in seen:
                continue
            seen.add(tref.slot_id)
            unique_refs.append(tref)

        for tref in unique_refs:
            lock_expr = mutex_lock(
                pipe=ir.PipeType.V, mutex_id=tref.buf_id, mutex_ids=tref.mutex_ids, span=span
            )
            self.builder.emit(ir.EvalStmt(lock_expr, span))
        return unique_refs

    def _emit_inline_vf_mutex_unlock(self, unique_refs: list, span: ir.Span) -> None:
        """Emit mutex_unlock(V, buf_id) for each ref from _emit_inline_vf_mutex_lock."""
        from pypto_pro.ir.op.system_ops import mutex_unlock

        for tref in unique_refs:
            unlock_expr = mutex_unlock(
                pipe=ir.PipeType.V, mutex_id=tref.buf_id, mutex_ids=tref.mutex_ids, span=span
            )
            self.builder.emit(ir.EvalStmt(unlock_expr, span))

    # -------------------------------------------------------------------------
    # Block default handler and helpers
    # -------------------------------------------------------------------------

    def _parse_block_default(self, op_name: str, call: ast.Call) -> ir.Expr:
        """Handle block DSL ops not registered in _OP_REGISTRY."""
        span = self.span_tracker.get_span(call)

        args = [self.parse_expression(arg) for arg in call.args]
        kwargs = self.parse_op_kwargs(call)

        # first arg is out; keep out as first arg to match pto-isa convention
        return ir.create_op_call(
            block_ir_op(op_name), args, kwargs, span
        )
