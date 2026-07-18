# Copyright (c) 2026 Huawei Technologies Co., Ltd.
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
import copy
import logging
from collections import namedtuple
from dataclasses import dataclass
from typing import Any, Callable

from pypto.pypto_impl import ir
from pypto_pro.ir import op as ir_op
from pypto_pro.ir.op.block_ops import block_ir_op
from pypto_pro.ir.op._op_registry import _OP_REGISTRY

from .diagnostics import (
    InvalidOperationError,
    ParserSyntaxError,
    ParserTypeError,
    UndefinedVariableError,
    UnsupportedFeatureError,
)


logger = logging.getLogger(__name__)


# Mutex carrier for tile-group tiles (buf_id IR expr, candidate values, memory, dedup id).
_MutexRef = namedtuple("_MutexRef", "buf_id mutex_ids memory slot_id")


@dataclass(frozen=True)
class _InlineFunctionTemplate:
    """Immutable source template for one directly-expanded Python callable."""

    func_def: ast.FunctionDef
    closure_vars: dict[str, Any]
    is_vector_function: bool


class _InlineLocalRenamer(ast.NodeTransformer):
    """Rename bindings local to one helper expansion without touching closures."""

    def __init__(self, names: set[str], prefix: str):
        self._names = names
        self._rename = {name: f"{prefix}{name}" for name in names}

    def generic_visit(self, node):
        if isinstance(node, ast.Name) and node.id in self._rename:
            node.id = self._rename[node.id]
        return super().generic_visit(node)

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

    # Mapping of VF kwarg names to their expected enum classes (tuple of types).
    # When a kwarg value resolves to an instance of any mapped enum, its .value
    # VF enum kwarg validation: if a kwarg is mapped to enum classes, the parser
    # passes the enum object through to ConvertKwargsDict (which extracts .value as int).
    # If a raw string is passed for a mapped kwarg, the parser raises an error.
    _VF_KWARG_ENUMS: dict[str, tuple] = {}

    # -------------------------------------------------------------------------
    # VF ops
    # -------------------------------------------------------------------------

    # VF op names are now unified to snake_case across Python API, IR, and C++
    # backend. This map is kept for potential future name aliasing; currently
    # all entries are identity (Python name == IR op name).
    _VF_OP_NAME_MAP: dict[str, str] = {}

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
        "eq": 1, "ne": 1, "lt": 1, "gt": 1, "le": 1, "ge": 1,
        "histograms": 1,
        # 2 dsts
        "interleave": 2, "de_interleave": 2,
        "mask_interleave": 2, "mask_deinterleave": 2,
        "mull": 2,
        "addc": 2, "subc": 2,
        # 0 dst (no assignment form)
        "store_align": 0, "store_unalign": 0, "store_unalign_post": 0,
        "scatter": 0, "store": 0, "mask_store": 0, "mask_store_unalign": 0,
        "mem_bar": 0, "clear_spr": 0,
        "load_unalign_pre": 0,
        "store_align_pack": 0, "store_align_intlv": 0,
        "store_align_pack_postupdate": 0,
    }

    # VF ops whose dst(s) are MaskReg (not RegTensor). The assignment parser
    # declares these via vf.create_mask instead of vf.reg_tensor.
    _VF_MASK_DST_OPS: frozenset[str] = frozenset({
        "eq", "ne", "lt", "gt", "le", "ge",
        "mask_and", "mask_or", "mask_xor", "mask_not",
        "mask_mov", "mask_sel", "mask_pack", "mask_unpack",
        "mask_interleave", "mask_deinterleave",
    })

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
        is_vf: bool = False,
    ) -> ir.Function:
        """Parse an implicit helper, surfacing the underlying failure with its location."""
        try:
            return sub_parser.parse_function(
                func_def, func_type=ir.FunctionType.Helper,
                is_vector_function=is_vf,
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

    @staticmethod
    def _inline_param_list(func_def: ast.FunctionDef) -> list[ast.arg]:
        return list(func_def.args.args)

    @staticmethod
    def _inline_local_names(func_def: ast.FunctionDef, params: list[ast.arg]) -> set[str]:
        """Collect names local to a helper body, including nested block bindings."""
        names = {param.arg for param in params}

        class _Collector(ast.NodeVisitor):
            def generic_visit(self, node):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                    return None
                if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
                    names.add(node.id)
                    return None
                return super().generic_visit(node)

        collector = _Collector()
        for stmt in func_def.body:
            collector.visit(stmt)
        return names

    @staticmethod
    def _inline_reassigned_params(func_def: ast.FunctionDef, params: list[ast.arg]) -> set[str]:
        """Return helper parameters that are rebound by its body.

        A directly inlined parameter normally aliases the caller expression.  If
        the helper assigns to it, however, it needs its own IR definition before
        entering control flow so ConvertToSSA can carry it through branches and
        loops.
        """
        param_names = {param.arg for param in params}
        assigned: set[str] = set()

        class _Collector(ast.NodeVisitor):
            def generic_visit(self, node):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                    return None
                if (
                    isinstance(node, ast.Name)
                    and isinstance(node.ctx, (ast.Store, ast.Del))
                    and node.id in param_names
                ):
                    assigned.add(node.id)
                    return None
                return super().generic_visit(node)

        collector = _Collector()
        for stmt in func_def.body:
            collector.visit(stmt)
        return assigned

    @staticmethod
    def _validate_inline_returns(func_name: str, func_def: ast.FunctionDef, span) -> ast.Return | None:
        """Validate the straight-line return contract and return the final value return."""
        body = [stmt for stmt in func_def.body if not CallParserMixin._is_docstring(stmt)]
        all_returns = [node for node in ast.walk(func_def) if isinstance(node, ast.Return)]
        has_yield = any(isinstance(node, (ast.Yield, ast.YieldFrom)) for node in ast.walk(func_def))
        if has_yield:
            raise ParserSyntaxError(
                f"Inline function '{func_name}' cannot contain yield",
                span=span,
            )
        top_return = body[-1] if body and isinstance(body[-1], ast.Return) else None
        if len(all_returns) != (1 if top_return is not None else 0):
            raise ParserSyntaxError(
                f"Inline function '{func_name}' may only return from its final top-level statement",
                span=span,
            )
        if top_return is None:
            return None
        if top_return.value is None:
            return None
        return top_return

    @staticmethod
    def _has_unsupported_inline_params(args: ast.arguments) -> bool:
        if args.posonlyargs or args.vararg is not None:
            return True
        if args.kwarg is not None or args.kwonlyargs:
            return True
        return bool(args.kw_defaults)

    # -------------------------------------------------------------------------
    # Mutex dedup helpers (shared by _emit_auto_mutex and _emit_vf_func_mutex_lock)
    # -------------------------------------------------------------------------

    @staticmethod
    def _group_refs_by_mutex_overlap(refs: list) -> list:
        """Group tilerefs by mutex_ids overlap (connected components via union-find).

        Two refs whose mutex_ids lists have any common value are in the same group.
        Returns a list of groups, each group is a list of _MutexRef.
        """
        n = len(refs)
        parent = list(range(n))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py

        # Build mutex_id sets for each ref
        id_sets = [set(ref.mutex_ids) if ref.mutex_ids else set() for ref in refs]

        # Union refs that have overlapping mutex_ids
        for i in range(n):
            for j in range(i + 1, n):
                if id_sets[i] & id_sets[j]:
                    union(i, j)

        # Collect groups
        groups: dict[int, list] = {}
        for i in range(n):
            root = find(i)
            groups.setdefault(root, []).append(refs[i])
        return list(groups.values())

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

    @classmethod
    def _is_vf_mask_dst_op(cls, op_name: str) -> bool:
        """Return True if the VF op's dst(s) are MaskReg (not RegTensor)."""
        return op_name in cls._VF_MASK_DST_OPS

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

        # Handle bare-name calls to external IR functions or inline Python callables.
        if isinstance(func, ast.Name):
            func_name = func.id

            # Builtin min/max -> route to pl.min/pl.max (scalar_ops.py)
            if func_name in _BUILTIN_TO_OP:
                op_func = _OP_REGISTRY.get(_BUILTIN_TO_OP[func_name])
                if op_func is not None:
                    return op_func(self, call)

            resolved = self.expr_evaluator.closure_vars.get(func_name)
            if isinstance(resolved, ir.Function):
                return self._parse_external_function_call(func_name, resolved, call)
            if callable(resolved) and not isinstance(resolved, type):
                return self._implicit_func_call(func_name, resolved, call)

        raise UnsupportedFeatureError(
            f"Unsupported function call: {ast.unparse(call)}",
            span=self.span_tracker.get_span(call),
            hint="Use pl.* operations, self.method() for cross-function calls, "
            "or call an inline Python helper by name",
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

    def resolve_single_kwarg(self, key: str, value: ast.expr) -> Any:
        """Resolve a single keyword argument value to a Python or IR value."""
        if key == "dtype":
            return self.resolve_dtype_expr(value)
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

        Validates the parsed expression as a constant tuple of integers.
        """
        for kw in call.keywords:
            if kw.arg != key:
                continue
            value = kw.value
            parsed = self.parse_expression(value)
            if isinstance(parsed, ir.MakeTuple):
                values = []
                for element in parsed.elements:
                    if not isinstance(element, ir.ConstInt) or element.type.dtype == ir.DataType.BOOL:
                        break
                    values.append(element.value)
                else:
                    return values
            raise ParserTypeError(
                f"'{key}' must be a compile-time constant integer list, got '{ast.unparse(value)}'",
                span=self.span_tracker.get_span(value),
                hint=f"{key} selects tensor axes at compile time; pass a constant list "
                f"(e.g. {key}=[1, 3]) or a variable bound to one, not a runtime value.",
            )
        return None

    def resolve_const_bool_kwarg(self, call: ast.Call, key: str) -> bool | None:
        """Resolve a compile-time constant bool kwarg through ``const_env``."""
        for kw in call.keywords:
            if kw.arg != key:
                continue
            value = kw.value
            parsed = self.parse_expression(value)
            if isinstance(parsed, ir.ConstBool):
                return parsed.value
            if isinstance(parsed, ir.ConstInt) and parsed.type.dtype == ir.DataType.BOOL:
                return bool(parsed.value)
            raise ParserTypeError(
                f"'{key}' must be a compile-time constant bool, got '{ast.unparse(value)}'",
                span=self.span_tracker.get_span(value),
                hint=f"{key} is a compile-time attribute; pass a constant bool "
                f"(e.g. {key}=True) or a variable bound to one, not a runtime value.",
            )
        return None

    def resolve_static_int(self, elt: ast.expr) -> int:
        """Resolve a compile-time integer through the normal expression parser."""
        value = self.parse_expression(elt)
        if isinstance(value, ir.ConstInt) and value.type.dtype != ir.DataType.BOOL:
            return value.value
        raise ParserTypeError(
            f"'{ast.unparse(elt)}' is not a compile-time integer constant",
            span=self.span_tracker.get_span(elt),
            hint="A constant list kwarg (e.g. TileType shape/valid_shape) must be compile-time "
            "constants; use pl.set_validshape() for a runtime valid shape.",
        )

    def resolve_dtype_expr(self, value: ast.expr):
        """Resolve a dtype through ``parse_expression`` then validate its enum value."""
        parsed = self.parse_expression(value)
        if isinstance(parsed, ir.ConstInt):
            dtype = self.type_resolver.dtype_from_value(parsed.value)
            if dtype is not None:
                return dtype
            raise ParserTypeError(
                f"'{ast.unparse(value)}' is not a valid dtype value",
                span=self.span_tracker.get_span(value),
            )
        # Preserve the existing closure/diagnostic behavior for non-parser dtype
        # annotations and for unsupported enum values.
        return self.type_resolver.resolve_dtype(value)

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
            group_meta = self.tile_group_meta.get(arg_exprs[i])
            if group_meta is not None:
                sub_parser.set_inferred_tile_group_meta(param.arg, group_meta)
            tile_meta = self._tile_mutex_meta.get(arg_exprs[i])
            if tile_meta is not None:
                sub_parser.set_inferred_tile_mutex_meta(param.arg, tile_meta)
        sub_parser.set_auto_mutex_enabled(self.auto_mutex_enabled)

    def _register_implicit_kernel_function(
        self,
        fn: Callable,
        func_def: ast.FunctionDef,
        ir_func: ir.Function,
        sub_parser,
        is_vf: bool = False,
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
        if is_vf:
            # @pl.vector_function: the entire body is a VF section, so every
            # tile-valued param needs mutex_lock/unlock(V) around the func.call.
            self.kfunc_vf_used_params[id(kfunc)] = set(param_names)
        else:
            self.kfunc_vf_used_params[id(kfunc)] = None
        self.external_funcs.update(sub_parser.external_funcs)
        self.external_funcs[fn.__name__] = ir_func
        return kfunc

    def _inline_template(self, func_name: str, fn: Callable, span) -> _InlineFunctionTemplate:
        template = self.inline_func_cache.get(id(fn))
        if template is not None:
            return template
        from .decorator import is_vector_function

        is_vf = is_vector_function(fn)
        source_info = self._retrieve_function_source(
            func_name, fn, span, "@pl.vector_function" if is_vf else "an annotated Python helper",
        )
        template = _InlineFunctionTemplate(
            func_def=source_info[4],
            closure_vars=self._build_function_closure(fn),
            is_vector_function=is_vf,
        )
        self.inline_func_cache[id(fn)] = template
        return template

    def _bind_inline_arguments(
        self,
        func_name: str,
        func_def: ast.FunctionDef,
        call: ast.Call,
        span,
    ) -> dict[str, tuple[ir.Expr, ast.expr]]:
        """Bind one positional argument to each helper parameter."""
        args = func_def.args
        if self._has_unsupported_inline_params(args):
            raise ParserSyntaxError(
                f"Inline function '{func_name}' only supports positional parameters with optional defaults",
                span=span,
            )
        if call.keywords or any(isinstance(arg, ast.Starred) for arg in call.args):
            raise ParserSyntaxError(
                f"Call to inline function '{func_name}' only supports plain positional arguments",
                span=span,
            )

        params = self._inline_param_list(func_def)
        required_count = len(params) - len(args.defaults)
        if not required_count <= len(call.args) <= len(params):
            raise ParserTypeError(
                f"Function '{func_name}' expects {required_count} to {len(params)} positional argument(s), "
                f"got {len(call.args)}",
                span=span,
            )
        argument_nodes = [*call.args, *args.defaults[len(call.args) - required_count:]]
        bound = {
            param.arg: (self.parse_expression(arg), arg)
            for param, arg in zip(params, argument_nodes)
        }
        for param in params:
            actual = bound[param.arg][0]
            if param.annotation is None or self.resolve_tiling_class(param.annotation):
                continue
            if self.type_resolver.annotation_has_shape_policy(param.annotation):
                self.type_resolver.validate_policy_parameter_type(param.annotation, param.arg, actual.type)
            else:
                annotated = self.type_resolver.resolve_param_type(param.annotation)
                _check_type_compatible(
                    annotated, actual.type, what="Parameter", name=param.arg, span=span,
                )
        return bound

    def _implicit_func_call(self, func_name: str, fn: Callable, call: ast.Call) -> ir.Expr | None:
        """Expand a Python helper body directly into the caller's IR builder."""
        span = self.span_tracker.get_span(call)
        template = self._inline_template(func_name, fn, span)
        if id(fn) in self.inline_call_stack:
            raise ParserSyntaxError(
                f"Recursive inline function call detected for '{func_name}'", span=span,
            )

        old_closure = self.expr_evaluator.closure_vars
        old_const_env = self.const_env
        self.expr_evaluator.closure_vars = {**template.closure_vars, **old_closure}
        try:
            self._validate_inline_returns(func_name, template.func_def, span)
            bound = self._bind_inline_arguments(func_name, template.func_def, call, span)
            params = self._inline_param_list(template.func_def)
            inline_id = self.inline_counter
            self.inline_counter += 1
            prefix = f"__inline_{inline_id}_"
            local_names = self._inline_local_names(template.func_def, params)
            reassigned_params = self._inline_reassigned_params(template.func_def, params)
            func_def = copy.deepcopy(template.func_def)
            renamer = _InlineLocalRenamer(local_names, prefix)
            body = [renamer.visit(stmt) for stmt in func_def.body if not self._is_docstring(stmt)]
            ast.fix_missing_locations(func_def)
            renamed_return = body[-1] if body and isinstance(body[-1], ast.Return) else None

            locked_vf_refs: list = []
            if template.is_vector_function and self._auto_mutex:
                arg_nodes = [bound[param.arg][1] for param in params]
                locked_vf_refs = self._emit_vf_func_mutex_lock(
                    [param.arg for param in params],
                    [self._try_resolve_tileref(arg) for arg in arg_nodes],
                    {param.arg for param in params},
                    span,
                )

            self.inline_call_stack.append(id(fn))
            is_outermost_vf = template.is_vector_function and self.inline_vf_depth == 0
            if template.is_vector_function:
                self.inline_vf_depth += 1
            self.scope_manager.enter_scope("inline")
            self.const_env = dict(self.const_env)
            try:
                for param in params:
                    expr, _arg_node = bound[param.arg]
                    renamed_name = f"{prefix}{param.arg}"
                    # A parameter that the helper rebinds must first become a
                    # real local IR value.  Binding it directly to the caller
                    # expression leaves no version for SSA to merge when the
                    # assignment is nested in an if/while.
                    if param.arg in reassigned_params:
                        value = self.builder.let(renamed_name, expr, span=span)
                        self._transfer_tile_sync_metadata(value, expr)
                    else:
                        value = expr
                    self.scope_manager.define_var(renamed_name, value, allow_redef=True)
                    self._update_const_env(renamed_name, expr)

                statements = body[:-1] if isinstance(renamed_return, ast.Return) else body
                if is_outermost_vf:
                    with self.builder.section(ir.SectionKind.VF, span):
                        self.scope_manager.enter_scope("section")
                        try:
                            for stmt in statements:
                                self.parse_statement(stmt)
                        finally:
                            self.scope_manager.exit_scope(leak_vars=False)
                else:
                    for stmt in statements:
                        self.parse_statement(stmt)
                if isinstance(renamed_return, ast.Return) and renamed_return.value is not None:
                    return self.parse_expression(renamed_return.value)
                return None
            finally:
                self.scope_manager.exit_scope(leak_vars=False)
                self.inline_call_stack.pop()
                if template.is_vector_function:
                    self.inline_vf_depth -= 1
                self.const_env = old_const_env
                if locked_vf_refs:
                    self._emit_inline_vf_mutex_unlock(locked_vf_refs, span)
        finally:
            self.expr_evaluator.closure_vars = old_closure

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
        if self.scope_manager.lookup_var_bounded(value.id) is not None:
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
            and self.scope_manager.lookup_var_bounded(elt.id) is not None
            for elt in value.elts
        ):
            return self.parse_list(value)
        success, result = self.expr_evaluator.try_eval_expr(value)
        if success and isinstance(result, list):
            return result
        return self.parse_list(value)

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

        # Block direct use of vf.reg_tensor / vf.mask_reg — registers are now
        # declared implicitly by the assignment form (dst = vf.xxx(...)); users
        # cannot call them directly.
        if op_name in ("reg_tensor", "mask_reg"):
            raise ParserSyntaxError(
                f"vf.{op_name} cannot be called directly. VF registers are declared "
                "automatically by the assignment form.",
                span=span,
                hint="Use: dst = vf.load_align(...)  # or any VF compute op",
            )

        args = [self.parse_expression(arg) for arg in call.args]
        kwargs = self.parse_op_kwargs(call)
        backend_op_name = self._VF_OP_NAME_MAP.get(op_name, op_name)

        return ir.create_op_call(f"vf.{backend_op_name}", args, kwargs, span)

    # --- auto_mutex helpers ---------------------------------------------------

    def _try_resolve_tileref(self, node: ast.expr):
        """Resolve a tile argument to a mutex ref.

        Call parse_expression to obtain the IR expr, then look up sync metadata
        from _tile_mutex_meta. Works for both inline group accessors
        (acc.next()) and variable-assigned tiles (cur_a).

        Returns _MutexRef(buf_id, mutex_ids, memory, slot_id) or None.
        """
        expr = self.parse_expression(node)
        if not isinstance(expr, ir.Expr):
            return None
        meta = self._tile_mutex_meta.get(expr)
        slot_id = id(expr)
        if meta is None:
            return None
        buf_id_ir, mutex_ids = meta
        mem = expr.type.memref.memory_space_ if isinstance(expr.type, ir.TileType) and expr.type.memref else None
        return _MutexRef(buf_id_ir, mutex_ids, mem, slot_id)

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

        # 3. Emit lock for each unique _TileRef, with dedup for aliasing tiles.
        # Group once here and reuse the grouping at unlock time.
        groups = self._group_refs_by_mutex_overlap(unique_refs)
        self._emit_mutex_ops_with_dedup(groups, pipe, span, is_lock=True)

        # Store grouping for post-op unlock emission (avoids re-grouping)
        self._pending_mutex_unlocks = (groups, pipe, span)

    def _emit_auto_mutex_unlocks(self):
        """Emit mutex_unlock calls queued by _emit_auto_mutex or _parse_func_call."""
        if not hasattr(self, "_pending_mutex_unlocks") or self._pending_mutex_unlocks is None:
            return
        groups, pipe, span = self._pending_mutex_unlocks
        self._emit_mutex_ops_with_dedup(groups, pipe, span, is_lock=False)
        self._pending_mutex_unlocks = None

    def _emit_mutex_ops_with_dedup(self, groups: list, pipe, span: ir.Span, *, is_lock: bool):
        """Emit mutex lock/unlock calls for pre-grouped refs, with dedup if-guards.

        ``groups`` is the output of _group_refs_by_mutex_overlap (computed once at
        lock time and reused at unlock time). Shared by lock (is_lock=True) and unlock
        (is_lock=False) since they only differ in which op is emitted. Per group:
          - single ref, or a group where all buf_ids are the same static int
            (guaranteed equal at compile time) -> one plain mutex_lock/unlock
          - otherwise (dynamic ids) -> one mutex_(un)lock_dyn carrying all mutex_id
            exprs; the CCE codegen emits runtime if-guards so each unique id is only
            locked/unlocked once (avoids hardware hang from double get_buf).
        """
        from pypto_pro.ir.op.system_ops import mutex_lock, mutex_unlock, _create_mutex_dedup_op
        from pypto_pro.ir._utils import _normalize_expr

        emit_plain = mutex_lock if is_lock else mutex_unlock
        op_name = "system.mutex_lock" if is_lock else "system.mutex_unlock"

        for group in groups:
            all_static_equal = (
                all(isinstance(tref.buf_id, int) for tref in group)
                and len(set(tref.buf_id for tref in group)) == 1
            )
            if len(group) == 1 or all_static_equal:
                # Guaranteed a single distinct lock: emit one plain mutex op.
                tref = group[0]
                expr = emit_plain(pipe=pipe, mutex_id=tref.buf_id, mutex_ids=tref.mutex_ids, span=span)
            else:
                # Dynamic ids: emit dedup op with runtime if-guards.
                id_exprs = [_normalize_expr(tref.buf_id, span) for tref in group]
                ids_union = sorted(set().union(*(set(tref.mutex_ids) for tref in group if tref.mutex_ids)))
                expr = _create_mutex_dedup_op(
                    op_name, pipe=pipe, mutex_id_exprs=id_exprs,
                    mutex_ids_union=ids_union, span=span)
            self.builder.emit(ir.EvalStmt(expr, span))

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
        is generated). Returns the ref grouping (from _group_refs_by_mutex_overlap)
        so the caller can emit matching unlocks after the call without re-grouping.
        """
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

        groups = self._group_refs_by_mutex_overlap(unique_refs)
        self._emit_mutex_ops_with_dedup(groups, ir.PipeType.V, span, is_lock=True)
        return groups

    def _emit_inline_vf_mutex_unlock(self, groups: list, span: ir.Span) -> None:
        """Emit mutex_unlock(V, buf_id) for each group from _emit_vf_func_mutex_lock."""
        self._emit_mutex_ops_with_dedup(groups, ir.PipeType.V, span, is_lock=False)

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
