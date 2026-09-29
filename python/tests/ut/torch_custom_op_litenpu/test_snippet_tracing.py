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
"""Unit tests for the snippet-level packing of traced external helpers and constants, under the
uniform canonical-identity mechanism.

Covers the packed helper and constant set for a multi-file direct kernel, the annotation-vs-body
same-name constant split, the absence of spurious defs for a self-contained kernel, and build-time
retrace idempotence. The tracer itself is unit-tested in ``test_multifile_tracer.py``; the exact
emitted text is pinned by the goldens under ``test_compile_snippet_data/``.
"""

import ast
import importlib
import importlib.util
from pathlib import Path
import re
import sys

from pypto.extensions.torch_custom_op_litenpu.common import kernel_snippet

_DIR = Path(__file__).resolve().parent


def _load(name: str, filename: str):
    path = _DIR / filename
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# Ensure the helper package is importable, then load the samples (which import from it).
sys.path.insert(0, str(_DIR))
samples = _load("kernel_snippet_samples_packing_ut", "kernel_snippet_samples.py")


def _direct_snippet(jit_kernel_func) -> str:
    return kernel_snippet.build_kernel_compile_snippet(
        mode="jit_kernel",
        jit_kernel_func=jit_kernel_func,
        infer_shape_func=samples.add_infer_shape,
        infer_dtype_func=samples.add_infer_dtype,
        n_inputs=2,
    )


def test_multifile_snippet_packs_all_helpers_and_constants():
    snippet = _direct_snippet(samples.direct_multifile_kernel)
    compile(snippet, "<snip>", "exec")
    for tok in (
        "__tile_helper(",
        "__add_into(",
        "___bias(",
        "@pypto.frontend.function",
        "TILE = (1, 4, 1, 64)",
        "def direct_multifile_kernel(",
    ):
        assert tok in snippet, tok
    # Self-contained: the helper package name is only ever part of a canonical def name, never a live
    # ``import`` or bare module reference (all such refs were rewritten or stripped).
    assert "import sample_helper_pkg" not in snippet
    assert "sample_helper_pkg.compute." not in snippet


def test_direct_annotation_body_const_collision_kept_distinct():
    """A direct kernel whose param annotation pins ``PIN_SHAPE`` (value B) while
    its body reaches a cross-file same-named ``PIN_SHAPE`` (value A). Both must be packed distinctly
    (``PIN_SHAPE`` / ``PIN_SHAPE__2``) and the annotation rewritten to its own value."""
    snippet = _direct_snippet(samples.direct_annot_collision_kernel)
    compile(snippet, "<snip>", "exec")
    assert "PIN_SHAPE = (1, 1, 4, 64)" in snippet       # body const (value A) kept
    assert "PIN_SHAPE__2 = (1, 1, 8, 64)" in snippet     # annotation const (value B) not dropped
    # the kernel's parameter annotation now references the suffixed (own-value) name
    assert "pypto.Tensor(PIN_SHAPE__2, pypto.DT_FP16)" in snippet
    # value B is never aliased onto value A: no annotation still resolves to the bare body name
    assert "pypto.Tensor(PIN_SHAPE," not in snippet


def test_erased_direct_snippet_has_no_spurious_defs():
    """An erased single-file direct kernel referencing only torch/pypto gains no helper defs."""
    snippet = _direct_snippet(samples.direct_add_kernel_erased)
    compile(snippet, "<snip>", "exec")
    top_defs = re.findall(r"^def (\w+)", snippet, re.M)
    assert set(top_defs) == {
        "add_infer_shape",
        "add_infer_dtype",
        "direct_add_kernel_erased",
        "__pypto_compile",
    }, top_defs


def _normalized_defs_and_consts(snippet: str):
    tree = ast.parse(snippet)
    defs = {n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    consts = {}
    for n in tree.body:
        if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name):
            try:
                consts[n.targets[0].id] = ast.literal_eval(n.value)
            except (ValueError, SyntaxError):
                pass
    return defs, consts


def test_build_time_retrace_is_idempotent(tmp_path):
    """regen(regen(x)) == regen(x): re-importing an already-packed snippet and re-running the builder
    must not add, drop or duplicate defs. The helpers are top-level entry funcs in the reconstructed
    module (kept under their canonical names), so nothing is re-emitted spuriously.
    """
    snippet1 = _direct_snippet(samples.direct_multifile_kernel)
    mod_path = tmp_path / "regen_kmod.py"
    mod_path.write_text(snippet1)
    sys.path.insert(0, str(tmp_path))
    try:
        mod = importlib.import_module("regen_kmod")
        jit_fn = getattr(mod, "direct_multifile_kernel")
        infer_shape = getattr(mod, "add_infer_shape")
        infer_dtype = getattr(mod, "add_infer_dtype")
        snippet2 = kernel_snippet.build_kernel_compile_snippet(
            mode="jit_kernel", jit_kernel_func=jit_fn,
            infer_shape_func=infer_shape, infer_dtype_func=infer_dtype, n_inputs=2,
        )
        compile(snippet2, "<snip2>", "exec")
        assert _normalized_defs_and_consts(snippet1) == _normalized_defs_and_consts(snippet2)
    finally:
        sys.path.remove(str(tmp_path))
        sys.modules.pop("regen_kmod", None)
