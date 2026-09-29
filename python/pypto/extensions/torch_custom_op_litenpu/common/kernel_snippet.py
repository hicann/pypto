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
"""Pack a pypto kernel's Python source into a self-contained, importable compile snippet.

The emitted module exposes the captured functions (``create_kernel`` / ``infer_shape`` / ``infer_dtype`` /
optional ``kernel_body``, plus any helper they reference), the module-level constants they read inlined as
literals, and a ``__pypto_compile(shapes, dtypes, attrs, soc)`` entry that JIT-builds the kernel for the
given runtime shapes.
"""
from __future__ import annotations

import ast
import inspect
import re
import textwrap
from typing import Callable

from .kernel_tracer import (
    _EMBEDDED_PREIMPORTS,
    _annotation_referenced_names,
    _is_captureable_const,
    _is_pypto_dtype,
    _local_import_spans_for_source,
    _real_preimport_modules,
    _register_captured_value,
    _rewrite_source,
    _trace_referenced,
)
from .return_annotations import infer_shape_output_arity
from .source_utils import _unwrap_decorated_func_source

# The ``run_mode`` value a direct kernel's decorator is rewritten to at snippet emission: the
# deployed ``Compile()``/``__pypto_compile`` must always build the kernel host-side (the ``.so`` only
# compiles; GE does the device launch), so the artifact is SIM regardless of the source run_mode.
_DEPLOY_DECORATOR_RUN_MODE = "pypto.RunMode.SIM"

# Matches a ``run_mode`` keyword on the jit decorator line, in dict form (``"run_mode": ...``) or
# kwarg form (``run_mode=...``). The capture group keeps the original punctuation so the rewrite
# replaces only the ``RunMode.<x>`` value.
_DECORATOR_RUN_MODE_RE = re.compile(
    r'(run_mode["\']?\s*[:=]\s*)(?:pypto\.)?RunMode\.\w+'
)


def _force_sim_in_decorator(kernel_block: str) -> str:
    """Force ``run_mode`` to SIM on the direct kernel's decorator only.

    *kernel_block* is the direct kernel source as captured, so its first
    non-blank lines are the ``@pypto.frontend.jit(...)`` decorator followed by the ``def`` line. The
    edit is scoped to the decorator span, so a kernel body that references ``RunMode`` is untouched:
    an existing ``run_mode`` has its value rewritten in place, and a decorator without one has
    ``runtime_options={"run_mode": pypto.RunMode.SIM}`` injected. Injection parses only the decorator
    span, so the kernel body is never unparsed: a bare ``@...jit`` becomes ``@...jit(runtime_options=...)``,
    while ``runtime_options=<NAME>`` (a non-literal) cannot be mutated safely and raises an authoring error.
    """
    lines = kernel_block.splitlines(keepends=True)
    def_idx = next((i for i, ln in enumerate(lines) if re.match(r"\s*def\s", ln)), len(lines))
    deco_lines = lines[:def_idx]
    deco_text = "".join(deco_lines)

    if _DECORATOR_RUN_MODE_RE.search(deco_text):
        rewritten = _DECORATOR_RUN_MODE_RE.sub(rf"\1{_DEPLOY_DECORATOR_RUN_MODE}", deco_text)
        return rewritten + "".join(lines[def_idx:])

    # No ``run_mode`` on the decorator: inject one. The common leading indent is stripped so
    # ``ast.parse`` accepts the decorator, and restored verbatim after unparse.
    indents = [ln[:len(ln) - len(ln.lstrip())] for ln in deco_text.splitlines() if ln.strip()]
    indent = indents[0] if indents else ""
    for ind in indents[1:]:
        while not ind.startswith(indent):
            indent = indent[:-1]
    dedented = textwrap.dedent(deco_text) if indent else deco_text
    # Attach a trivial function so the decorator is parseable on its own.
    module = ast.parse(dedented + "def __pto_deco_probe__(): pass\n")
    func = module.body[-1]
    if not func.decorator_list:
        raise ValueError("_force_sim_in_decorator: expected a decorator on the DIRECT kernel block")
    deco = func.decorator_list[-1]

    sim_value = ast.parse(_DEPLOY_DECORATOR_RUN_MODE, mode="eval").body

    if isinstance(deco, ast.Call):
        rt_kw = next((k for k in deco.keywords if k.arg == "runtime_options"), None)
        if rt_kw is None:
            deco.keywords.append(ast.keyword(
                arg="runtime_options",
                value=ast.Dict(keys=[ast.Constant(value="run_mode")], values=[sim_value]),
            ))
        elif isinstance(rt_kw.value, ast.Dict):
            rt_kw.value.keys.append(ast.Constant(value="run_mode"))
            rt_kw.value.values.append(sim_value)
        else:
            raise ValueError(
                "DIRECT kernel decorator has runtime_options=<non-literal> and no run_mode: cannot inject "
                "RunMode.SIM into a variable. Pin runtime_options={'run_mode': pypto.RunMode.SIM} inline on "
                "the @pypto.frontend.jit decorator."
            )
    else:
        # Bare ``@...jit`` (no parens): wrap it in a Call carrying runtime_options.
        func.decorator_list[-1] = ast.Call(
            func=deco,
            args=[],
            keywords=[ast.keyword(
                arg="runtime_options",
                value=ast.Dict(keys=[ast.Constant(value="run_mode")], values=[sim_value]),
            )],
        )

    ast.fix_missing_locations(module)
    new_decos = "".join("@" + ast.unparse(d) + "\n" for d in func.decorator_list)
    if indent:
        new_decos = "".join(indent + ln if ln.strip() else ln for ln in new_decos.splitlines(keepends=True))
    return new_decos + "".join(lines[def_idx:])


def build_kernel_compile_snippet(
    *,
    create_kernel_func: Callable | None = None,
    infer_shape_func: Callable,
    infer_dtype_func: Callable,
    n_inputs: int,
    factory_signature: str = "single",
    kernel_body_func: Callable | None = None,
    jit_kernel_func: Callable | None = None,
    declared_annotations: list | None = None,
    mode: str = "factory",
) -> str:
    """Assemble the self-contained Python source that rebuilds one kernel from its captured functions.

    The text must be written to a real ``.py`` and imported, not ``exec``'d -- pypto's jit parser re-reads
    the kernel through ``inspect.getsourcelines``. The emitted ``__pypto_compile(in_shapes, in_dtypes,
    attrs, soc_version)`` takes dtypes as canonical strings (``"float16"``), never enum ints, and returns
    the compiled-kernel path. Functions are emitted decorator-stripped under their ORIGINAL names so
    intra-kernel references resolve, and output arity comes from ``infer_shape``'s return annotation.
    ``mode`` is ``"factory"`` (a ``create_*_kernel`` factory drives ``@jit`` + tiling) or ``"jit_kernel"``
    (an already-built kernel emitted verbatim).
    """
    if mode not in ("factory", "jit_kernel"):
        raise ValueError("build_kernel_compile_snippet: mode must be 'factory' or 'jit_kernel'")
    if mode == "factory":
        # factory_signature applies only to a factory; jit_kernel mode carries None.
        if factory_signature not in ("single", "lists", "full"):
            raise ValueError("build_kernel_compile_snippet: factory_signature must be 'single', 'lists', or 'full'")
        if create_kernel_func is None:
            raise ValueError("build_kernel_compile_snippet: mode='factory' requires create_kernel_func")
    elif jit_kernel_func is None:
        raise ValueError("build_kernel_compile_snippet: mode='jit_kernel' requires jit_kernel_func")

    n_outputs = infer_shape_output_arity(infer_shape_func)
    infer_shape_name = infer_shape_func.__name__
    infer_dtype_name = infer_dtype_func.__name__

    # Functions whose SOURCE is emitted into the snippet.
    source_funcs: list[Callable] = []
    if kernel_body_func is not None:
        source_funcs.append(kernel_body_func)
    source_funcs.extend([infer_shape_func, infer_dtype_func])
    if create_kernel_func is not None:
        source_funcs.append(create_kernel_func)

    if mode == "jit_kernel":
        # DIRECT: emit the jit kernel's source with @pypto.frontend.jit KEPT.
        # The undecorated function object drives constant discovery (its __code__/globals carry the
        # body references: tileshape helper, module constants).
        raw = jit_kernel_func._original_func
        create_name = raw.__name__
        kernel_block = inspect.getsource(raw)
        kernel_block = _force_sim_in_decorator(kernel_block)
        # (source, its own globals, emitting func) for every ENTRY block whose references we must also
        # rewrite. The func object keys the local-import span channel; the DIRECT kernel block is
        # emitted from ``raw``, the object the tracer discovered from.
        entry_blocks = [
            (_unwrap_decorated_func_source(inspect.getsource(fn)), fn.__globals__, fn)
            for fn in source_funcs
        ]
        entry_blocks.append((kernel_block, raw.__globals__, raw))
        discovery_funcs = list(source_funcs) + [raw]
        # The SINGLE module the kernel lives in: the only one whose helpers keep their original name
        # (see ``_trace_referenced``). After regen every func lands here, so it stays a fixed point.
        kernel_module = getattr(raw, "__module__", None)
    else:
        create_name = create_kernel_func.__name__
        entry_blocks = [
            (_unwrap_decorated_func_source(inspect.getsource(fn)), fn.__globals__, fn)
            for fn in source_funcs
        ]
        discovery_funcs = list(source_funcs)
        kernel_module = getattr(create_kernel_func, "__module__", None)

    # Trace every function and constant the kernel and its hooks reference, transitively across files:
    # external helpers packed by source, module constants captured as literals. ``emitted_name`` keys
    # by ``id(obj)``, so the SAME map rewrites the packed helper sources AND the entry/kernel blocks
    # below. Raises on an unpackable reference, a function-local import or an un-emittable annotation.
    (
        emitted_name, helper_sources, registry, stdlib_imports, local_import_bindings,
    ) = _trace_referenced(discovery_funcs, kernel_module=kernel_module)

    # DIRECT-only: parameter ANNOTATIONS evaluate in module scope, so the body tracer never packs a
    # constant used ONLY in an annotation. Register those through the same value-keyed allocator; a
    # name colliding with a differently-valued body constant gets a ``__<n>`` suffix and is rewritten
    # by identity below. No collision keeps the original spelling, so a pinned snippet stays identical.
    if mode == "jit_kernel":
        g = raw.__globals__
        for name in sorted(_annotation_referenced_names(raw)):
            if name in _EMBEDDED_PREIMPORTS:
                continue
            if name not in g:
                continue
            value = g[name]
            # Dtype/captureable-scalar/shape values only. A pypto ``DataType`` is an ``int`` subclass,
            # so ``_register_captured_value`` routes it to the ``pypto.<DT_*>`` channel rather than the
            # literal-repr one -- its ``repr`` does not round-trip. Other names are left verbatim.
            if _is_pypto_dtype(value) or _is_captureable_const(value):
                _register_captured_value(value, name, registry)

    # Rewrite every entry/kernel block's global references through the SAME identity map, then
    # concatenate. Local-import spans ride along as ``extra_spans``, so it is ONE span-editor pass. A
    # block with nothing to rewrite comes back byte-identical, so erased/factory snippets are unchanged.
    real_modules = _real_preimport_modules()

    captured_block = "\n\n".join(
        _rewrite_source(
            src, gns, emitted_name, real_modules,
            extra_spans=_local_import_spans_for_source(
                src, local_import_bindings.get(id(fn), {}), gns, emitted_name, real_modules
            ),
        )
        for src, gns, fn in entry_blocks
    )

    if helper_sources:
        helper_block = "\n\n".join(helper_sources)
        captured_block = f"{helper_block}\n\n{captured_block}"
    consts_block = "".join(f"{name} = {value!r}\n" for name, value in sorted(registry.consts.items()))
    # Dtype-const channel: ``name = pypto.<DT_...>`` (repr of a DataType does not round-trip). Emitted
    # only when non-empty, so erased/factory snippets stay byte-identical.
    dtype_block = "".join(
        f"{name} = pypto.{value.name}\n" for name, value in sorted(registry.dtype_consts.items())
    )
    # Stdlib build-time imports: the tracer's ready-to-emit statements (``import a.b.c`` / ``import
    # <mod> as <alias>``), sorted (byte-stable), AFTER torch/pypto and BEFORE the const/dtype blocks.
    # stdlib SOURCE is never packed; the module exists at deploy. Empty for the common case.
    stdlib_block = "".join(f"{stmt}\n" for stmt in sorted(stdlib_imports))

    header = (
        "import torch\n"
        "import pypto.extensions.torch_custom_op_litenpu\n"
        + stdlib_block
        + (f"\n{consts_block}" if consts_block else "")
        + (f"\n{dtype_block}" if dtype_block else "")
    )

    # The entry returns only the compiled-kernel path; the host runtime reads the sibling launch
    # sidecar (blockDim / kernelName / ...) alongside it. ``factory`` mode emits ``CompileEntry(factory=...)``
    # over ``create_kernel``, ``jit_kernel`` mode ``CompileEntry(jit_kernel=...)`` over the built kernel.
    declared_kw = (
        f"\n    declared_annotations={declared_annotations!r}," if declared_annotations else ""
    )
    mode_kw = (
        f"jit_kernel={create_name}" if mode == "jit_kernel"
        else f"factory={create_name},\n    factory_signature=\"{factory_signature}\""
    )
    entry_expr = (
        f"pypto.extensions.torch_custom_op_litenpu.CompileEntry(\n    {mode_kw}, num_inputs={n_inputs}, "
        f"num_outputs={n_outputs},{declared_kw}\n)"
    )
    glue = f"""_custom_op_compile = {entry_expr}
_custom_op_compile.infer_shape = {infer_shape_name}
_custom_op_compile.infer_dtype = {infer_dtype_name}


def __pypto_compile(in_shapes, in_dtypes, attrs, soc_version):
    return _custom_op_compile(
        tuple(tuple(s) for s in in_shapes), tuple(str(d) for d in in_dtypes),
        dict(attrs), soc_version,
    )
"""

    return f"{header}\n\n{captured_block}\n\n\n{glue}"
