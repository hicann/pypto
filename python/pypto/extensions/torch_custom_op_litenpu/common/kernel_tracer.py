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
"""Discover and pack the helpers, constants and imports a pypto kernel's source references.

A BFS over each function's own globals resolves every reference by object identity, so a helper reached under
several spellings is packed once. Discovered helpers get an injective canonical name, referenced constants are
captured as literals, function-local imports are hoisted, un-emittable helper annotations are stripped, and each
emitted source is rewritten through one span-editor pass preserving comments and formatting.

Anything the tracer cannot pack raises ``_TraceError`` at pack time, never a NameError at deploy.
"""
from __future__ import annotations

import ast
import builtins
from enum import Enum
import functools
import importlib.util
import inspect
import os
import re
import sys
import textwrap
import types
from typing import Any, Callable

import torch

from .authoring import _is_jit_kernel
from .source_utils import (
    _annotation_delete_spans,
    _apply_span_replacements,
    _line_start_offsets,
    _node_char_span,
    _sanitize,
    _scope_classify,
    _stmt_line_span,
)


class _TraceError(ValueError):
    """Export-time error raised by the recursive helper/constant tracer."""


# Pre-imports always available inside the embedded snippet (see ``build_kernel_compile_snippet``'s
# header). Free names resolving to these need no capture.
_EMBEDDED_PREIMPORTS = frozenset({"torch", "pypto"})

# Stdlib modules that are unsafe to emit as a header ``import <mod>``: they have import-time side
# effects or environment requirements, and the snippet header runs at deploy-import time.
# ``antigravity`` launches a web browser on import; ``this`` prints the Zen to stdout;
# ``turtle``/``tkinter``/``idlelib`` require a display. A reference to one of these falls through to
# a clean raise rather than a side-effecting import.
_STDLIB_IMPORT_DENYLIST = frozenset({"antigravity", "this", "turtle", "tkinter", "idlelib"})


def _stdlib_top_module(name: str) -> str | None:
    """The top-level module name of *name* iff it is a safe stdlib module, else ``None``.

    Gates on the top-level name (``os.path`` -> ``os``); ``find_spec`` only scans metadata, never runs it.
    """
    top = (name or "").split(".")[0]
    if not top or top in _STDLIB_IMPORT_DENYLIST:
        return None
    stdlib = getattr(sys, "stdlib_module_names", None)
    if stdlib:
        return top if top in stdlib else None
    try:
        spec = importlib.util.find_spec(top)
    except Exception:  # noqa: BLE001 - third-party meta_path finders may raise arbitrarily
        return None
    if spec is None:
        return None
    origin = getattr(spec, "origin", None)
    if origin in ("built-in", "frozen"):
        return top
    if not origin:
        return None
    norm = os.path.normpath(origin).replace(os.sep, "/")
    stdlib_dir = os.path.normpath(os.path.dirname(os.__file__)).replace(os.sep, "/")
    if norm.startswith(stdlib_dir + "/") and "/site-packages/" not in norm and "/dist-packages/" not in norm:
        return top
    return None


def _is_third_party_path(path: str | None) -> bool:
    """True if *path* lives under site-packages/dist-packages.

    A user's own package is usually editable-installed and so packs; a pip-installed library raises.
    """
    if not path:
        return False
    norm = os.path.normpath(path).replace(os.sep, "/")
    return "/site-packages/" in norm or "/dist-packages/" in norm


def _annotation_referenced_names(func: Callable) -> set[str]:
    """Names referenced from a direct kernel's parameter annotations (not its body).

    A jit kernel's annotations are evaluated in module scope, so the body walker never sees them; the
    caller packs annotation-position constants through this set. Attribute chains yield only their base.
    """
    src = inspect.getsource(func)
    tree = ast.parse(src)
    func_def = next(n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)))
    args = func_def.args
    names: set[str] = set()
    for arg in (*args.posonlyargs, *args.args, *args.kwonlyargs):
        if arg.annotation is None:
            continue
        for node in ast.walk(arg.annotation):
            if isinstance(node, ast.Name):
                names.add(node.id)
    return names


def _is_pypto_dtype(value: Any) -> bool:
    """Duck-type a pypto dtype constant (a ``DataType`` whose ``name`` starts with ``DT_``).

    Duck-typed to avoid a circular top-level pypto import. ``pypto.<name>`` round-trips; ``repr`` does not.
    """
    return (
        type(value).__name__ == "DataType"
        and getattr(value, "name", "").startswith("DT_")
    )


def _is_captureable_const(value: Any) -> bool:
    """True if ``repr(value)`` round-trips to valid Python (scalars and nested tuples/lists/dicts)."""
    if isinstance(value, (bool, int, float, str, type(None))):
        return True
    if isinstance(value, (tuple, list)):
        return all(_is_captureable_const(v) for v in value)
    if isinstance(value, dict):
        return all(_is_captureable_const(k) and _is_captureable_const(v) for k, v in value.items())
    return False


def _is_deep_immutable(value: Any) -> bool:
    """True if *value* is a scalar, or a tuple whose elements are all deep-immutable.

    Two distinct mutable constants must pack under distinct names, so mutating one never reaches the other.
    """
    if isinstance(value, (bool, int, float, str, type(None))):
        return True
    if isinstance(value, tuple):
        return all(_is_deep_immutable(v) for v in value)
    return False


def _same_value(a: Any, b: Any) -> bool:
    """True if two constants may dedup to one emitted name: same type, deeply immutable, and equal.

    A mutable value is never "same" and is always suffixed; an exotic non-comparable falls back to identity.
    """
    try:
        return (
            type(a) is type(b)
            and _is_deep_immutable(a)
            and _is_deep_immutable(b)
            and a == b
        )
    except Exception:  # noqa: BLE001 - exotic non-comparable value
        return a is b


class _CaptureRegistry:
    """The captured constant and dtype channels, plus the emitted-name map they allocate against.

    One registry per build, shared by the body tracer and the annotation channel, so an annotation
    constant colliding by bare name with a body constant gets the same suffix instead of being dropped.
    """

    __slots__ = ("emitted_name", "name_to_value", "consts", "dtype_consts")

    def __init__(self, emitted_name: dict[int, str] | None = None) -> None:
        self.emitted_name: dict[int, str] = {} if emitted_name is None else emitted_name
        self.name_to_value: dict[str, Any] = {}
        self.consts: dict[str, Any] = {}
        self.dtype_consts: dict[str, Any] = {}


def _register_captured_value(value: Any, ref_name: str, registry: _CaptureRegistry) -> None:
    """Record a captureable const or dtype under an emitted name derived from its reference name.

    Keeps the original spelling when free (or already bound to the same value), so the common
    single-reference case stays byte-identical; a same-name-different-value clash gets ``<name>__<n>``.
    Dtypes go to the ``pypto.<DT_*>`` channel, everything else to the literal-repr one. Idempotent by id.
    """
    if id(value) in registry.emitted_name:
        return
    base = _sanitize(ref_name) or "const"
    name = base
    n = 2
    while name in registry.name_to_value and not _same_value(registry.name_to_value[name], value):
        name = f"{base}__{n}"
        n += 1
    if name in registry.name_to_value:  # same value already emitted under this name -> reuse
        registry.emitted_name[id(value)] = name
        return
    registry.name_to_value[name] = value
    if _is_pypto_dtype(value):
        registry.dtype_consts[name] = value
    else:
        registry.consts[name] = value
    registry.emitted_name[id(value)] = name


def _unwrap_traceable_func(value: Any):
    """The plain Python function behind *value* (``__code__`` + ``__globals__``), else ``None``.

    A ``@pypto.frontend.function`` helper and a jit kernel both wrap the real function and expose
    ``_original_func`` without code or globals, so tracing the wrapper would drop transitive helpers.
    """
    raw = getattr(value, "_original_func", value)
    if inspect.isfunction(raw) and hasattr(raw, "__code__") and hasattr(raw, "__globals__"):
        return raw
    return None


def _attr_chain(node: ast.AST) -> list[str] | None:
    """The dotted-name segments of a pure attribute chain rooted at a bare Name, else ``None``."""
    parts: list[str] = []
    cur = node
    while isinstance(cur, ast.Attribute):
        parts.append(cur.attr)
        cur = cur.value
    if not isinstance(cur, ast.Name):
        return None
    parts.append(cur.id)
    parts.reverse()
    return parts


def _resolve_chain(globals_ns: dict, parts: list[str]):
    """Resolve a dotted chain against *globals_ns*, or ``None`` if any hop is missing."""
    if not parts or parts[0] not in globals_ns:
        return None
    value = globals_ns[parts[0]]
    for seg in parts[1:]:
        if not hasattr(value, seg):
            return None
        value = getattr(value, seg)
    return value


@functools.lru_cache(maxsize=1)
def _real_preimport_modules() -> tuple:
    """The module objects the emitted snippet pre-imports (``torch``, and ``pypto`` if importable).

    A reference resolving by identity to one of these, or rooted in one, is already in the snippet and
    is never packed; by identity, not name, so a user module merely named ``torch`` still flattens.
    Resolved lazily because importing pypto at module top is heavy and circular.
    """
    mods = [torch]
    try:
        import pypto as _pypto  # noqa: PLC0415 - lazy to avoid a top-level circular import
        mods.append(_pypto)
    except Exception:  # noqa: BLE001 - pypto may be unavailable in a pure-source context
        pass
    return tuple(mods)


class _CanonicalNamer:
    """Assigns each discovered helper a spelling-independent, injective name by object identity.

    The base is ``_sanitize(__module__) + "__" + _sanitize(__qualname__)``, so a nested helper stays
    distinct. A build-global base-to-object map keeps it injective: a rare clash gets ``base__<n>``, and
    the same object always maps to the same name. The caller pins traversal order, so it is deterministic.
    """

    def __init__(self) -> None:
        self._base_to_obj: dict[str, Any] = {}
        self._by_id: dict[int, str] = {}

    def canonical(self, obj: Any) -> str:
        prior = self._by_id.get(id(obj))
        if prior is not None:
            return prior
        base = _sanitize(getattr(obj, "__module__", "") or "") + "__" + _sanitize(
            getattr(obj, "__qualname__", None) or getattr(obj, "__name__", "helper")
        )
        name = base
        n = 2
        while name in self._base_to_obj and self._base_to_obj[name] is not obj:
            name = f"{base}__{n}"
            n += 1
        self._base_to_obj[name] = obj
        self._by_id[id(obj)] = name
        return name


def _iter_reference_candidates(tree: ast.AST) -> list[tuple[ast.AST, list[str]]]:
    """Every module-rooted reference candidate in *tree*, in AST source order.

    Yields ``(node, parts)`` for each outermost attribute chain rooted at a bare Name load and each bare
    Name load, ordered by ``(lineno, col_offset)``. A chain is reported only at its outermost node, so
    the resolver can pick the longest resolving prefix.
    """
    attr_candidates: list[tuple[ast.AST, list[str]]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Load):
            parts = _attr_chain(node)
            if parts is None or len(parts) < 2:
                continue
            attr_candidates.append((node, parts))
    inner_ids: set[int] = set()
    for node, _parts in attr_candidates:
        cur = node.value
        while isinstance(cur, ast.Attribute):
            inner_ids.add(id(cur))
            cur = cur.value
    chain_root_ids: set[int] = set()
    out: list[tuple[ast.AST, list[str]]] = []
    for node, parts in attr_candidates:
        if id(node) in inner_ids:
            continue
        cur = node.value
        while isinstance(cur, ast.Attribute):
            cur = cur.value
        if isinstance(cur, ast.Name):
            chain_root_ids.add(id(cur))
        out.append((node, parts))
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and id(node) not in chain_root_ids:
            out.append((node, [node.id]))
    out.sort(key=lambda t: (getattr(t[0], "lineno", 0), getattr(t[0], "col_offset", 0)))
    return out


class _RefResolution(Enum):
    """What ``_resolve_reference`` found for a chain, and what the third element of its return holds."""

    PREIMPORTED = "preimported"  # payload None            - chain root OR a prefix IS torch/pypto
    STDLIB = "stdlib"            # payload tuple[str, ...] - the header import statements to emit
    FUNC = "func"                # payload Callable        - the traceable helper to pack
    CONST = "const"              # payload Any             - a captureable literal
    DTYPE = "dtype"              # payload Any             - a pypto dtype
    THIRDPARTY = "thirdparty"    # payload str             - the library name, for the raise
    DENYLISTED = "denylisted"    # payload str             - the module name, for the raise
    UNPACKABLE = "unpackable"    # payload Any             - the offending live object
    UNRESOLVED = "unresolved"    # payload None            - nothing resolves at all


def _resolve_reference(
    parts: list[str], globals_ns: dict, real_modules: tuple
) -> tuple[_RefResolution, int, Any]:
    """Resolve the longest prefix of *parts* that yields a packable object or a captureable value.

    Returns ``(resolution, matched_len, value)``; ``_RefResolution`` documents what ``value`` holds per member.
    """
    root = globals_ns.get(parts[0])
    if root is not None and any(root is m for m in real_modules):
        return (_RefResolution.PREIMPORTED, len(parts), None)
    # Stdlib is classified at the chain ROOT, not in the longest-prefix loop below: for ``math.prod``
    # the longest resolving prefix is the builtin ``prod``, not the ``math`` module. A user module root
    # is not short-circuited; it still resolves to its packable helper in the loop.
    if inspect.ismodule(root):
        root_name = getattr(root, "__name__", "") or ""
        top = _stdlib_top_module(root_name)
        if top is not None:
            # The header import statements that make the chain resolve at deploy. The walk descends only
            # through real submodules, so ``os.path`` (module ``posixpath``) stops at its parent, never renaming it.
            deep_name, cur = root_name, root
            for attr in parts[1:]:
                child = getattr(cur, attr, None)
                if inspect.ismodule(child) and getattr(child, "__name__", "") == f"{deep_name}.{attr}":
                    deep_name, cur = child.__name__, child
                else:
                    break
            if parts[0] == root_name:
                stmts = (f"import {deep_name}",)
            elif deep_name != root_name:
                stmts = (f"import {deep_name}", f"import {root_name} as {parts[0]}")
            else:
                stmts = (f"import {root_name} as {parts[0]}",)
            return (_RefResolution.STDLIB, len(parts), stmts)
        # A denylisted stdlib module gets the specific "not safe to import" resolution, not the
        # generic unpackable raise, and is never imported.
        if root_name.split(".")[0] in _STDLIB_IMPORT_DENYLIST:
            return (_RefResolution.DENYLISTED, len(parts), root_name.split(".")[0])
    resolved_any = None
    resolved_len = 0
    # Longest resolving prefix first.
    for n in range(len(parts), 0, -1):
        value = _resolve_chain(globals_ns, parts[:n])
        if value is None:
            continue
        if resolved_any is None:
            resolved_any, resolved_len = value, n
        if any(value is m for m in real_modules):
            return (_RefResolution.PREIMPORTED, n, None)
        raw = _unwrap_traceable_func(value)
        if raw is not None:
            # A function whose source lives under site/dist-packages is third-party library source.
            if _stdlib_top_module(getattr(raw, "__module__", "") or "") is None and _is_third_party_path(
                inspect.getsourcefile(raw)
            ):
                return (_RefResolution.THIRDPARTY, n, (getattr(raw, "__module__", "") or parts[0]).split(".")[0])
            return (_RefResolution.FUNC, n, raw)
        # A captureable const / dtype must be the FULL chain (attributes of a const are not packed).
        if n == len(parts):
            if _is_pypto_dtype(value):
                return (_RefResolution.DTYPE, n, value)
            if _is_captureable_const(value):
                return (_RefResolution.CONST, n, value)
    if inspect.ismodule(root) and _is_third_party_path(getattr(root, "__file__", None)):
        modname = getattr(root, "__name__", parts[0]).split(".")[0]
        return (_RefResolution.THIRDPARTY, resolved_len or len(parts), modname)
    if resolved_any is not None:
        return (_RefResolution.UNPACKABLE, resolved_len, resolved_any)
    return (_RefResolution.UNRESOLVED, 0, None)


def _thirdparty_raise(modname: str) -> "_TraceError":
    """The library-naming raise for a reference to a non-stdlib, non-torch/pypto module."""
    return _TraceError(
        f"references external library {modname!r}; the packed snippet imports only torch+pypto "
        f"(+ stdlib), so {modname!r} will be unavailable at deploy. Compute the value in Python and "
        "pass it as a constant, or inline the logic using pypto ops."
    )


def _denylisted_stdlib_raise(modname: str) -> "_TraceError":
    """The raise for a stdlib module that is unsafe to import into a compile snippet."""
    return _TraceError(
        f"stdlib module {modname!r} is not safe to import into a compile snippet (import-time side "
        "effects); compute the value in Python and pass it as a constant"
    )


def _annotation_is_tensor_typed(ann: ast.AST, globals_ns: dict) -> bool:
    """True if the annotation expression evaluates live to a ``pypto.Tensor(...)`` object.

    A tensor annotation is load-bearing - the jit parser reads it live - so it is never stripped, only
    its captureable consts are packed. Evaluation failure means it is not a live tensor annotation.
    """
    try:
        value = eval(compile(ast.Expression(ann), "<ann>", "eval"), dict(globals_ns))  # noqa: S307
    except Exception:  # noqa: BLE001 - an un-evaluatable annotation is not a live tensor annotation
        return False
    return hasattr(value, "to_tensor")


def _normalize_helper_annotations(
    raw: Callable, globals_ns: dict, real_modules: tuple, src: str
) -> tuple[dict[str, Any], list[tuple[int, int, str]]]:
    """Scan a packed helper's annotations: capture consts, strip un-emittable non-tensor ones.

    *globals_ns* must be the helper's own ``__globals__``, since a cross-file tensor-annotation const lives in
    the helper's module. A bare Name resolving to an un-emittable class or module raises when the annotation is
    tensor-typed (stripping it would change the kernel signature) and is otherwise deleted whole.

    Returns ``(captured_consts, strip_spans)``, the spans name-end anchored so a default is preserved.
    """
    builtin_names = set(dir(builtins))
    captured: dict[str, Any] = {}
    strip_spans: list[tuple[int, int, str]] = []
    line_starts = _line_start_offsets(src)
    lines = src.splitlines(keepends=True)
    tree = ast.parse(src)
    func_def = next(n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)))
    for ann, del_span in _annotation_delete_spans(func_def, src, line_starts, lines):
        # A PEP 563 string annotation is not evaluated at import, so it cannot NameError.
        if isinstance(ann, ast.Constant) and isinstance(ann.value, str):
            continue
        tensor_typed = _annotation_is_tensor_typed(ann, globals_ns)
        for name_node in ast.walk(ann):
            if not isinstance(name_node, ast.Name):
                continue
            name = name_node.id
            if name in _EMBEDDED_PREIMPORTS or name in builtin_names or name == "typing":
                continue
            value = globals_ns.get(name)
            if value is None:
                continue
            if any(value is m for m in real_modules):
                continue
            if getattr(value, "__name__", None) == "typing":
                continue
            if _is_pypto_dtype(value):
                captured[name] = value
                continue
            if _is_captureable_const(value):
                captured[name] = value
                continue
            if tensor_typed:
                raise _TraceError(
                    f"packed helper {raw.__name__!r} has a TENSOR-typed "
                    f"param/return annotation referencing {name!r} (a {type(value).__name__}) that the "
                    "snippet cannot emit; make it a direct pypto.Tensor(...) expression, or pack the "
                    "referenced name as a module-level constant"
                )
            # One strip span per annotation; break so it is not added twice.
            if del_span[0] >= 0:
                strip_spans.append((del_span[0], del_span[1], ""))
            break
    return captured, strip_spans


def _denylist_guard(modname: str) -> None:
    """Raise, without importing, if *modname*'s top-level is a side-effecting denylisted module."""
    if (modname or "").split(".")[0] in _STDLIB_IMPORT_DENYLIST:
        raise _denylisted_stdlib_raise(modname)


def _safe_import_module(modname: str, ctx: str):
    """Import *modname* at pack time, guarding the denylist first; raise with *ctx* on failure."""
    _denylist_guard(modname)
    try:
        return importlib.import_module(modname)
    except Exception as exc:  # noqa: BLE001
        raise _TraceError(
            f"{ctx} cannot be resolved at pack time ({exc}); the target module is not importable in "
            "the pack environment"
        ) from exc


def _resolve_import_target(node: ast.AST, name: str, asname: str | None, func: Callable) -> tuple[str, Any]:
    """Resolve one binding of a function-local import to ``(local_name, target_object)``.

    ``import lib`` / ``import pkg.sub as z`` give the module object; ``from .x import g [as h]`` gives
    ``getattr(module, g)``, relative specs absolutized against the func's package. Denylist guarded first.
    """
    g = getattr(func, "__globals__", {})
    if isinstance(node, ast.Import):
        # ``import a.b.c`` binds ``a``; ``import a.b.c as z`` binds ``z`` to a.b.c.
        mod = _safe_import_module(name, f"function-local `import {name}`")
        if asname is not None:
            return (asname, mod)
        top = name.split(".", maxsplit=1)[0]
        return (top, _safe_import_module(top, f"function-local `import {top}`"))
    module = node.module or ""
    level = node.level or 0
    if level:
        package = g.get("__package__") or ""
        if not package:
            mod_name = getattr(func, "__module__", "") or ""
            package = mod_name.rsplit(".", 1)[0] if "." in mod_name else mod_name
        try:
            abs_module = importlib.util.resolve_name("." * level + module, package)
        except Exception as exc:  # noqa: BLE001
            raise _TraceError(
                f"function-local relative import `from {'.' * level}{module} import {name}` cannot be "
                f"resolved at pack time ({exc})"
            ) from exc
    else:
        abs_module = module
    mod = _safe_import_module(abs_module, f"function-local `from {abs_module} import {name}`")
    try:
        target = getattr(mod, name)
    except AttributeError as exc:
        raise _TraceError(
            f"function-local `from {abs_module} import {name}` cannot be resolved: {name!r} is not a "
            f"member of {abs_module!r}"
        ) from exc
    return (asname or name, target)


def _hoist_local_imports(
    func: Callable, emitted_src: str, globals_ns: dict, real_modules: tuple
) -> tuple[dict[str, Any], set[str]]:
    """Hoist a packed body's function-local static imports into the resolve and pack path.

    Parses *emitted_src*, collects each function-scope import binding, resolves its target against the module:

    * a stdlib module target becomes a header import reproducing the local binding; the local line is deleted.
    * a third-party module target raises naming the library.
    * a user module target becomes a chain-root binding.
    * a func / const / dtype target becomes a ``local_name -> target`` binding the caller packs.

    Returns ``(local_bindings, stdlib_statements)``.
    """
    local_bindings: dict[str, Any] = {}
    stdlib_stmts: set[str] = set()
    tree = ast.parse(emitted_src)
    # Module-scope imports were already stripped for helpers; any import node here is inside the def.
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Import, ast.ImportFrom)):
            continue
        for alias in node.names:
            local_name, target = _resolve_import_target(node, alias.name, alias.asname, func)
            if inspect.ismodule(target):
                # torch/pypto are pre-imports; an explicit local re-import is redundant, so bind it as
                # a chain root and delete the line.
                if any(target is m for m in real_modules):
                    local_bindings[local_name] = target
                    continue
                top = _stdlib_top_module(getattr(target, "__name__", "") or "")
                if top is not None:
                    # Record the header statement reproducing this binding. The body's chain stays
                    # verbatim, so the name must NOT be bound into the resolution namespace.
                    if isinstance(node, ast.Import) and alias.asname is None:
                        stdlib_stmts.add(f"import {alias.name}")
                    else:
                        stdlib_stmts.add(f"import {target.__name__} as {local_name}")
                    continue
                # Denylisted modules already raised in _resolve_import_target, before import.
                if _is_third_party_path(getattr(target, "__file__", None)):
                    raise _thirdparty_raise(getattr(target, "__name__", local_name).split(".")[0])
                local_bindings[local_name] = target
                continue
            local_bindings[local_name] = target
    return local_bindings, stdlib_stmts


def _local_import_spans_for_source(
    emitted_src: str,
    local_bindings: dict[str, Any],
    globals_ns: dict,
    emitted_name: dict[int, str],
    real_modules: tuple,
) -> list[tuple[int, int, str]]:
    """Span replacements for one emitted source: delete each function-local import statement and
    rewrite each locally-bound reference to its emitted name.

    Imports are deleted whole-line; bound references resolve against ``{**globals_ns, **local_bindings}``
    so a body-local chain root resolves. Stdlib chains are left verbatim.
    """
    tree = ast.parse(emitted_src)
    line_starts = _line_start_offsets(emitted_src)
    lines = emitted_src.splitlines(keepends=True)
    spans: list[tuple[int, int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            s = _stmt_line_span(node, emitted_src, line_starts, lines)
            if s[0] >= 0:
                spans.append((s[0], s[1], ""))
    if not local_bindings:
        return spans
    ns_aug = {**globals_ns, **local_bindings}
    # The bound names are free once the import lines are deleted, but symtable still classifies them
    # as locals against the pre-deletion source, so they are passed in as allowed names.
    spans.extend(
        _global_reference_spans(
            emitted_src, ns_aug, emitted_name, real_modules,
            allowed_names=set(local_bindings.keys()),
        )
    )
    return spans


def _rewrite_source(
    src: str,
    globals_ns: dict,
    emitted_name: dict[int, str],
    real_modules: tuple,
    extra_spans: list[tuple[int, int, str]] | None = None,
) -> str:
    """Rewrite every global reference in *src* to its emitted name, scope-correctly.

    *extra_spans* (import deletes, local rewrites, annotation strips) merge into the one span-editor pass.
    """
    replacements = _global_reference_spans(src, globals_ns, emitted_name, real_modules)
    if extra_spans:
        replacements.extend(extra_spans)
    return _apply_span_replacements(src, replacements)


def _global_reference_spans(
    src: str,
    globals_ns: dict,
    emitted_name: dict[int, str],
    real_modules: tuple,
    *,
    allowed_names: set[str] | None = None,
) -> list[tuple[int, int, str]]:
    """The identity-rewrite spans for *src*: genuine module-global loads mapped to emitted names.

    symtable classifies the names, so a local, parameter, comprehension or lambda variable is never
    rewritten. *allowed_names* unions in names freed by a deleted import. Raises on shadow-and-global-use.
    """
    tree = ast.parse(src)
    global_loads, ambiguous = _scope_classify(src)
    lines = src.splitlines(keepends=True)
    line_starts = _line_start_offsets(src)
    allow = global_loads | (allowed_names or set())
    replacements: list[tuple[int, int, str]] = []
    for node, parts in _iter_reference_candidates(tree):
        if parts[0] not in allow:
            continue
        if parts[0] in ambiguous:
            raise _TraceError(
                f"name {parts[0]!r} is used BOTH as a module-global reference and as a local binding "
                "in the same packed source (shadow-and-global-use); the tracer cannot rewrite it "
                "unambiguously - rename the local or the global helper/constant"
            )
        resolution, matched_len, value = _resolve_reference(parts, globals_ns, real_modules)
        if resolution not in (_RefResolution.FUNC, _RefResolution.CONST, _RefResolution.DTYPE):
            continue  # discovery already raised on any genuine unpackable; here, leave as-is
        name = emitted_name.get(id(value))
        if name is None:
            continue  # a discovered-but-unemitted object; leave as-is
        # Span of the matched prefix: walk inward from the outermost Attribute to the node whose
        # chain length equals matched_len.
        span_node = node
        cur_len = len(parts)
        while cur_len > matched_len and isinstance(span_node, ast.Attribute):
            span_node = span_node.value
            cur_len -= 1
        start, end = _node_char_span(span_node, line_starts, lines)
        replacements.append((start, end, name))
    return replacements


def _trace_referenced(
    entry_funcs: list[Callable],
    *,
    kernel_module: str | None = None,
) -> tuple[dict[int, str], list[str], _CaptureRegistry, set[str], dict[int, dict[str, Any]]]:
    """Discover and pack every helper and constant the kernel and its hooks reference, by identity.

    A BFS worklist seeded with *entry_funcs* resolves each function's references in its own globals, so
    cross-file walks work. One ``emitted_name`` map records what each object is emitted as: entry objects keep
    their original name, discovered helpers get a canonical one. Each emitted source is re-parsed and its
    module-global references rewritten by an absolute-char-offset span editor that preserves comments and
    formatting; the helper's own ``def`` is renamed to match. The same walk captures constants, hoists
    function-local imports and normalizes annotations, and raises precisely on a dynamic import, an
    un-emittable reference, a jit wrapper used as a helper, a sourceless helper, shadow-and-global-use or a
    third-party library, so a pack-time failure never becomes a deploy-time NameError.

    Returns ``(emitted_name, helper_sources, registry, stdlib_imports, local_import_bindings)``: the identity
    map, the packed helper sources in discovery order, the shared capture registry, the ready-to-emit stdlib
    header imports, and the per-func local-import bindings keyed by ``id(func)``.
    """
    real_modules = _real_preimport_modules()
    namer = _CanonicalNamer()

    # emitted_name: id(obj) -> the name it is emitted under. Entry objects keep their original name.
    emitted_name: dict[int, str] = {}
    entry_ids: set[int] = set()
    # name -> the object emitted under it, so two different objects can never share an emitted name.
    name_to_obj: dict[str, Any] = {}
    for f in entry_funcs:
        raw = _unwrap_traceable_func(f) or f
        name = f.__name__
        emitted_name[id(raw)] = name
        emitted_name[id(f)] = name
        name_to_obj[name] = raw
        entry_ids.add(id(raw))
        entry_ids.add(id(f))

    def _helper_emitted_name(raw: Callable) -> str:
        """The emitted name of a discovered helper: its original name only if it lives in the single
        kernel-source module, else the canonical name.

        Restricting the keep-original rule to one module avoids a cross-file collision: infer hooks may
        live elsewhere, and two files each reaching a distinct same-named helper would duplicate a ``def``.
        """
        if kernel_module is not None and getattr(raw, "__module__", None) == kernel_module:
            return raw.__name__
        return namer.canonical(raw)

    def _assign_emitted_name(raw: Callable) -> str:
        """Assign and record ``raw``'s emitted name, raising if a different object already holds it."""
        name = _helper_emitted_name(raw)
        prior = name_to_obj.get(name)
        if prior is not None and prior is not raw:
            raise _TraceError(
                f"two different helper functions would both be emitted as {name!r} "
                f"({getattr(prior, '__module__', '?')}.{getattr(prior, '__qualname__', '?')} vs "
                f"{getattr(raw, '__module__', '?')}.{getattr(raw, '__qualname__', '?')}); rename one. "
                "This happens when same-named helpers in the kernel's own file (or split across the "
                "kernel/infer source files) collide. Move one helper to a separate module (its "
                "module-qualified canonical name will disambiguate it)."
            )
        name_to_obj[name] = raw
        emitted_name[id(raw)] = name
        return name

    seen_func_ids: set[int] = set(entry_ids)
    discovered: list[Callable] = []
    registry = _CaptureRegistry(emitted_name)
    # Per-func local-import bindings, keyed by func identity (helper and entry). Also feeds the entry
    # span channel below.
    local_import_bindings: dict[int, dict[str, Any]] = {}
    stdlib_imports: set[str] = set()

    worklist: list[Callable] = [(_unwrap_traceable_func(f) or f) for f in entry_funcs]

    while worklist:
        func = worklist.pop(0)
        g = getattr(func, "__globals__", {})
        try:
            src = inspect.getsource(func)
        except (OSError, TypeError):
            continue
        # Dedent for discovery parsing only (a nested function's source is indented); the emitted
        # sources are re-derived and rewritten separately, so this never affects output spans.
        src = textwrap.dedent(src)
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue
        # A local bound from ``importlib.import_module(...)``/``__import__(...)`` then used as a module has
        # a runtime-unknowable name, so it can never be packed. Detected by the import-idiom callee, not by
        # "root is a local", which would false-positive on a method call on a parameter. Runs BEFORE the
        # hoist below, which would otherwise raise its own wrong error first on such a body.
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            callee = node.func
            callee_name = None
            if isinstance(callee, ast.Attribute):
                callee_name = callee.attr
            elif isinstance(callee, ast.Name):
                callee_name = callee.id
            if callee_name in ("import_module", "__import__"):
                raise _TraceError(
                    "a packed kernel/helper body performs a DYNAMIC import "
                    f"({callee_name!r}, line {getattr(node, 'lineno', '?')}); the tracer cannot pack a "
                    "runtime-imported module - move the import to module scope, or inline the helper"
                )
        # Resolve every function-local static import into local bindings. Third-party, denylisted or
        # unresolvable targets raise here; stdlib targets go to the header import channel.
        bindings, stmts = _hoist_local_imports(func, src, g, real_modules)
        local_import_bindings[id(func)] = bindings
        stdlib_imports |= stmts
        ns_aug = {**g, **bindings} if bindings else g
        global_loads, _ = _scope_classify(src)
        allow = global_loads | set(bindings.keys())
        # Annotation-position names are stripped or captured at emit time, so the discovery loop must
        # not treat an un-emittable annotation class as a body unpackable raise. The excluded names are
        # separately re-included at emit time by ``_annotation_referenced_names``.
        ann_ids: set[int] = set()
        for fd in ast.walk(tree):
            if not isinstance(fd, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            fd_args = fd.args
            anns = [a.annotation
                    for a in (*fd_args.posonlyargs, *fd_args.args, *fd_args.kwonlyargs) if a.annotation]
            if fd.returns is not None:
                anns.append(fd.returns)
            for ann in anns:
                ann_ids.update(id(n) for n in ast.walk(ann))
        for node, parts in _iter_reference_candidates(tree):
            if parts[0] not in allow:
                continue
            if id(node) in ann_ids:
                continue
            resolution, matched_len, value = _resolve_reference(parts, ns_aug, real_modules)
            if resolution in (_RefResolution.PREIMPORTED, _RefResolution.UNRESOLVED):
                continue
            if resolution is _RefResolution.STDLIB:
                # A module-scope stdlib build-time reference: record its header import statement(s),
                # leave the chain verbatim, capture nothing.
                stdlib_imports.update(value)
                continue
            if resolution in (_RefResolution.CONST, _RefResolution.DTYPE):
                _register_captured_value(value, parts[matched_len - 1], registry)
                continue
            if resolution is _RefResolution.THIRDPARTY:
                raise _thirdparty_raise(value)
            if resolution is _RefResolution.DENYLISTED:
                raise _denylisted_stdlib_raise(value)
            if resolution is _RefResolution.UNPACKABLE:
                raise _TraceError(
                    f"Embedded Compile() cannot pack the reference {'.'.join(parts[:matched_len])!r} "
                    f"(a {type(value).__name__}) reached from the kernel functions - it is a "
                    "module-level reference that will not exist in the generated executor. Make it a "
                    "simple constant (int/float/str/tuple/list/dict), derive it from the tensor "
                    "shapes, move it inside the kernel, or - for a helper - define it as a plain / "
                    "@pypto.frontend.function Python function."
                )
            if resolution is not _RefResolution.FUNC:  # exhaustive: every other member returned above
                raise AssertionError(f"unhandled reference resolution {resolution}")
            raw = value
            # The original global value at the matched prefix. ``raw`` is the unwrapped function; a
            # @pypto.frontend.function helper arrives wrapped in a marker, which is where a decorated
            # helper's own attributes live, so both are consulted below.
            resolved_value = _resolve_chain(ns_aug, parts[:matched_len])
            if matched_len < len(parts):
                # The chain continues PAST the helper, through one of its attributes. Only the helper's own
                # source is packed, never the module-scope statements that assign to it, so such a reference
                # would survive packing and fail at deploy. A function's own dunders and the attributes every
                # function object carries are excluded: the matched-prefix rewrite handles those correctly.
                seg = parts[matched_len]
                if (not (seg.startswith("__") and seg.endswith("__"))
                        and not hasattr(types.FunctionType, seg)
                        and (seg in raw.__dict__ or seg in resolved_value.__dict__)):
                    raise _TraceError(
                        f"Embedded Compile() cannot pack the reference {'.'.join(parts[:matched_len + 1])!r}: "
                        f"{seg!r} is an attribute assigned to the helper {'.'.join(parts[:matched_len])!r} at "
                        "module scope, and only the helper's own source is packed into the snippet - "
                        "the assignment is not, so the reference would fail at deploy with an "
                        "AttributeError. Make the value a module-level constant referenced directly, "
                        "or move it inside the helper."
                    )
            if id(raw) in seen_func_ids or id(raw) in emitted_name:
                continue
            # A jit wrapper used as a helper is invalid (only the top-level kernel may be a jit
            # kernel). A @pypto.frontend.function marker carries ``_func_name``; the raw jit wrapper
            # does not, so the two are distinguished on the original global value.
            if _is_jit_kernel(resolved_value) and not hasattr(resolved_value, "_func_name"):
                raise _TraceError(
                    f"referenced helper {'.'.join(parts[:matched_len])!r} resolves to a @pypto.frontend.jit "
                    "kernel (JitCallableWrapper) - only the top-level kernel may be a jit kernel; "
                    "make it a plain or @pypto.frontend.function helper"
                )
            try:
                inspect.getsource(raw)
            except (OSError, TypeError) as exc:
                raise _TraceError(
                    f"helper {'.'.join(parts[:matched_len])!r} has no retrievable source ({exc}); it must be "
                    "defined in a real .py file so its source can be packed into the snippet"
                ) from exc
            seen_func_ids.add(id(raw))
            _assign_emitted_name(raw)
            discovered.append(raw)
            worklist.append(raw)

    # Two recorded header imports that would bind one name to different modules cannot share the flat
    # snippet namespace: ``import xml.sax`` and ``import xml.etree.ElementTree`` both bind ``xml`` to
    # ``xml`` and coexist, but ``import json as j`` and ``import os as j`` would silently rebind.
    bound_to_module: dict[str, str] = {}
    for stmt in stdlib_imports:
        words = stmt.split()
        if len(words) == 4 and words[2] == "as":
            bound, module = words[3], words[1]
        else:
            bound = module = words[1].split(".")[0]
        prior = bound_to_module.get(bound)
        if prior is not None and prior != module:
            raise _TraceError(
                f"two packed stdlib imports would both bind {bound!r} to different modules "
                f"({prior!r} vs {module!r}); rename one import alias, a flat snippet namespace "
                "cannot hold both"
            )
        bound_to_module[bound] = module

    # Emit each discovered helper: its source, def-renamed and references rewritten, folding in the
    # annotation normalize (capture consts, strip un-emittable non-tensor annotations, resolved
    # against the helper's own globals) and the local-import hoist, delete and rewrite spans.
    helper_sources: list[str] = []
    for raw in discovered:
        g = raw.__globals__
        src = inspect.getsource(raw)
        captured, strip_spans = _normalize_helper_annotations(raw, g, real_modules, src)
        for cname, cval in captured.items():
            _register_captured_value(cval, cname, registry)
        extra = list(strip_spans)
        extra.extend(
            _local_import_spans_for_source(
                src, local_import_bindings.get(id(raw), {}), g, registry.emitted_name, real_modules
            )
        )
        src = _rewrite_source(src, g, registry.emitted_name, real_modules, extra_spans=extra)
        helper_sources.append(_rewrite_def_name(src, raw.__name__, registry.emitted_name[id(raw)]))

    return emitted_name, helper_sources, registry, stdlib_imports, local_import_bindings


def _rewrite_def_name(source: str, orig_name: str, new_name: str) -> str:
    """Rename a packed helper's ``def <orig_name>(`` to ``def <new_name>(``, that line only.

    Matches ``def`` + exact name + open paren, so body occurrences (already span-rewritten) are untouched.
    """
    if orig_name == new_name:
        return source
    pattern = re.compile(r"(\bdef\s+)" + re.escape(orig_name) + r"(\s*\()")
    return pattern.sub(rf"\g<1>{new_name}\g<2>", source, count=1)
