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
"""Unit tests for the uniform canonical-identity multi-file kernel tracer.

Three layers: that a reference in every import style (bare / dotted / relative / short / aliased /
re-export / same-object-two-ways / cross-file same-name) is discovered, canonically named and rewritten;
that each ``_TraceError`` guard fires on the input it names; and the symtable scope classifier plus the
byte-to-char span editor as units.

The fixture helper modules live on disk under ``multifile_fixtures/`` so ``inspect.getsource`` works.
"""
import ast
import importlib.util
from pathlib import Path
import re
import sys

import pytest

from pypto.extensions.torch_custom_op_litenpu.common import kernel_tracer
from pypto.extensions.torch_custom_op_litenpu.common.kernel_tracer import (
    _real_preimport_modules,
    _rewrite_source,
    _stdlib_top_module,
    _trace_referenced,
    _TraceError,
)
from pypto.extensions.torch_custom_op_litenpu.common.source_utils import _sanitize

_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_DIR))
sys.path.insert(0, str(_DIR / "multifile_fixtures"))


def _load(name: str, rel: str):
    path = _DIR / rel
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


entries = _load("mf_entries", "multifile_fixtures/entries.py")
guards = _load("mf_guards", "multifile_fixtures/guards.py")
chains = _load("mf_chains", "multifile_fixtures/chains.py")
entry_mod1 = _load("entry_mod1", "multifile_fixtures/entry_mod1.py")
entry_mod2 = _load("entry_mod2", "multifile_fixtures/entry_mod2.py")
guard_collision = _load("guard_collision", "multifile_fixtures/guard_collision.py")
import_hoist_entries = _load("mf_import_hoist_entries", "multifile_fixtures/import_hoist_entries.py")
stdlib_forms = _load("mf_stdlib_forms", "multifile_fixtures/stdlib_forms.py")
pep563_entries = _load("mf_pep563_entries", "multifile_fixtures/pep563_entries.py")


def _canonical(obj) -> str:
    """The expected canonical name for a helper object (mirrors ``_CanonicalNamer.canonical``)."""
    return _sanitize(obj.__module__) + "__" + _sanitize(obj.__qualname__)


def _resolve_dotted(path: str):
    """The object at a dotted ``module.attr`` path, importing the module."""
    mod_name, attr = path.rsplit(".", 1)
    return getattr(importlib.import_module(mod_name), attr)


def _trace(entry_funcs):
    emitted, helpers, registry, _stdlib, _bindings = _trace_referenced(entry_funcs)
    return emitted, helpers, registry.consts, registry.dtype_consts


def _helper_names(helpers):
    return [re.search(r"def (\S+?)\s*\(", s).group(1) for s in helpers]


def _rewritten_entry(fn, emitted):
    import inspect
    return _rewrite_source(inspect.getsource(fn), fn.__globals__, emitted, _real_preimport_modules())


def _no_dup_defs(helpers):
    names = _helper_names(helpers)
    assert len(names) == len(set(names)), names


def _nodoc(src: str) -> str:
    """The full function source (signature and body) with docstrings removed, so substring asserts on
    a packed helper match code rather than the fixture's prose."""
    tree = ast.parse(src.lstrip("\n"))
    for fn in ast.walk(tree):
        if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            b = fn.body
            if b and isinstance(b[0], ast.Expr) and isinstance(getattr(b[0], "value", None), ast.Constant) \
                    and isinstance(b[0].value.value, str):
                fn.body = b[1:] or [ast.Pass()]
    return ast.unparse(tree)


def _nodoc_join(helpers) -> str:
    """All packed helper sources, docstrings removed, joined."""
    return "\n\n".join(_nodoc(h) for h in helpers)


def _code_only(src: str) -> str:
    """The single top-level function's statements, docstring-free, for absence asserts."""
    return "\n".join(ast.unparse(n) for n in ast.parse(_nodoc(src)).body[0].body)


def test_cross_file_same_name_distinct_canonical():
    """Two different same-named ``bias`` (each via its own same-named ``_amount``) get
    distinct canonical names; both packed, no raise."""
    import pkg_a.compute as ca
    import pkg_b.compute as cb
    emitted, helpers, consts, _ = _trace([entries.entry_cross_file_same_name])
    names = _helper_names(helpers)
    assert _canonical(ca.bias) in names and _canonical(cb.bias) in names
    assert _canonical(ca._amount) in names and _canonical(cb._amount) in names
    assert "def bias(" not in "\n".join(helpers)  # old bare spelling gone
    _no_dup_defs(helpers)
    rewritten = _rewritten_entry(entries.entry_cross_file_same_name, emitted)
    code = _code_only(rewritten)
    assert _canonical(ca.bias) in code and _canonical(cb.bias) in code
    assert "pkg_a.compute.bias" not in code and "pkg_b.compute.bias" not in code
    # Same constant NAME across files, different values -> disambiguated.
    assert sorted(consts.values()) == [10, 20]


def test_same_object_two_ways_unified():
    """``tile`` (bare) and ``pkg_a.compute.tile`` (dotted) resolve to one canonical def, both call
    sites rewritten (no orphaned bare call)."""
    import pkg_a.compute as ca
    emitted, helpers, _, _ = _trace([entries.entry_same_object_two_ways])
    assert _helper_names(helpers) == [_canonical(ca.tile)]
    code = _code_only(_rewritten_entry(entries.entry_same_object_two_ways, emitted))
    assert code.count(_canonical(ca.tile)) == 2  # both call sites
    assert "pkg_a.compute.tile" not in code
    assert re.search(r"[^_]tile\(", code) is None  # no bare ``tile(`` left


def test_module_qualified_constant_captured():
    """A module-qualified constant ``pkg_a.compute.constants.CONST`` is captured and rewritten."""
    emitted, _, consts, _ = _trace([entries.entry_module_qualified_const])
    assert consts == {"CONST": 10}
    rewritten = _rewritten_entry(entries.entry_module_qualified_const, emitted)
    assert "pkg_a.compute.constants.CONST" not in rewritten
    assert "+ CONST" in rewritten


def test_helper_named_torch_pypto_renamed_not_dropped():
    """User helpers literally named ``torch`` / ``pypto`` are different objects from the pre-imports,
    so they are flattened canonically rather than dropped by the identity skip."""
    import user_torch
    emitted, helpers, _, _ = _trace([chains.entry_user_torch_helpers])
    names = _helper_names(helpers)
    assert _canonical(user_torch.torch) in names and _canonical(user_torch.pypto) in names
    assert "def torch(" not in "\n".join(helpers)
    rewritten = _rewritten_entry(chains.entry_user_torch_helpers, emitted)
    assert "user_torch.torch()" not in rewritten and "user_torch.pypto()" not in rewritten


def test_alias_from_import_resolved():
    """An aliased from-import (``from pkg_a.compute import bias as bias_a``) resolves by identity
    and is rewritten to ``bias``'s canonical name, with no alias raise."""
    import pkg_a.compute as ca
    emitted, helpers, _, _ = _trace([entries.entry_alias_from_import])
    assert _canonical(ca.bias) in _helper_names(helpers)
    rewritten = _rewritten_entry(entries.entry_alias_from_import, emitted)
    assert re.search(r"[^_]bias_a\(", rewritten) is None  # the alias spelling is gone


def test_short_module_root():
    """A short module root reaches a helper, resolved by identity rather than by spelling."""
    emitted, helpers, _, _ = _trace([entries.entry_short_root])
    assert helpers  # a helper was discovered + packed
    rewritten = _rewritten_entry(entries.entry_short_root, emitted)
    assert "__" in rewritten  # a canonical name is present


def test_reexport_unified_with_deep_path():
    """A package ``__init__`` re-export (``pkg_a.bias``) and the deep path (``pkg_a.compute.bias``)
    are the same object, so they pack once and both references are rewritten."""
    import pkg_a
    import pkg_a.compute as ca
    assert pkg_a.bias is ca.bias
    emitted, helpers, _, _ = _trace([entries.entry_reexport_same_object])
    _no_dup_defs(helpers)
    rewritten = _rewritten_entry(entries.entry_reexport_same_object, emitted)
    assert rewritten.count(_canonical(ca.bias)) == 2
    assert "pkg_a.bias" not in rewritten and "pkg_a.compute.bias" not in rewritten


def test_const_across_files_distinct_values_suffixed():
    """Two same-named module-qualified consts with different values give ``CONST`` + ``CONST__2``."""
    emitted, _, consts, _ = _trace([entries.entry_const_across_files])
    assert consts == {"CONST": 10, "CONST__2": 20}
    rewritten = _rewritten_entry(entries.entry_const_across_files, emitted)
    assert "+ CONST +" in rewritten and "CONST__2" in rewritten


def test_const_across_files_same_value_dedups_to_one_name():
    """Two same-named module-qualified consts with the same value dedup to one emitted ``SHARED``.
    Only a different-value clash suffixes, so the common case stays byte-identical."""
    emitted, _, consts, _ = _trace([entries.entry_const_same_value])
    assert consts == {"SHARED": (8, 8)}
    rewritten = _rewritten_entry(entries.entry_const_same_value, emitted)
    assert "SHARED__2" not in rewritten
    assert rewritten.count("SHARED") == 2  # both cross-file references rewritten to the one name


def _capture_registry():
    """A fresh capture registry for a direct allocator run."""
    return kernel_tracer._CaptureRegistry()


def test_capture_dealiases_distinct_mutables():
    """Two distinct equal-value mutable constants pack as ``CFG`` / ``CFG__2`` rather than aliasing to
    one object, so a build-time mutation of one never touches the other."""
    reg = _capture_registry()
    a, b = {}, {}
    kernel_tracer._register_captured_value(a, "CFG", reg)
    kernel_tracer._register_captured_value(b, "CFG", reg)
    assert set(reg.consts) == {"CFG", "CFG__2"}
    assert reg.emitted_name[id(a)] == "CFG" and reg.emitted_name[id(b)] == "CFG__2"
    a["k"] = 1
    assert b == {}  # distinct objects: mutating CFG left CFG__2 untouched


def test_capture_dedups_equal_immutables():
    """Two distinct but equal immutable constants still dedup to one emitted name each, so the
    byte-identity of the common case is unperturbed."""
    reg = _capture_registry()
    kernel_tracer._register_captured_value(7, "K", reg)
    kernel_tracer._register_captured_value(int("7"), "K", reg)  # distinct object, equal value
    kernel_tracer._register_captured_value((1, 2), "T", reg)
    kernel_tracer._register_captured_value(tuple([1, 2]), "T", reg)
    assert reg.consts == {"K": 7, "T": (1, 2)}  # one name each, no ``__2``


def test_capture_recurses_nested_mutable():
    """Two distinct ``(1, [2])`` tuples are equal and top-level immutable but nest a list, so the
    recursion defeats dedup and they pack as ``T`` / ``T__2``."""
    reg = _capture_registry()
    a, b = (1, [2]), (1, [2])
    kernel_tracer._register_captured_value(a, "T", reg)
    kernel_tracer._register_captured_value(b, "T", reg)
    assert set(reg.consts) == {"T", "T__2"}
    assert reg.emitted_name[id(a)] == "T" and reg.emitted_name[id(b)] == "T__2"


def test_function_base_collision_counter_suffix():
    """Two different helpers whose canonical ``<module>__<qualname>`` bases collide get distinct
    emitted names via the ``__2`` counter: both defs present, no duplicate, both call sites rewritten."""
    import base_collision_pkg.a
    import base_collision_pkg.a__b
    assert (_canonical(base_collision_pkg.a__b.f) == _canonical(base_collision_pkg.a.b__f)
            == "base_collision_pkg__a__b__f")
    emitted, helpers, _, _ = _trace([entries.entry_base_collision])
    names = _helper_names(helpers)
    assert set(names) == {"base_collision_pkg__a__b__f", "base_collision_pkg__a__b__f__2"}
    _no_dup_defs(helpers)
    code = _code_only(_rewritten_entry(entries.entry_base_collision, emitted))
    assert "base_collision_pkg__a__b__f()" in code and "base_collision_pkg__a__b__f__2()" in code
    assert "base_collision_pkg.a__b.f" not in code and "base_collision_pkg.a.b__f" not in code


def test_multi_module_entry_distinct_same_named_helpers_no_dup():
    """Two entry funcs in different files, each reaching a distinct helper sharing the original
    name ``shared``, are both canonicalized to distinct names, so there is no duplicate ``def``."""
    import helpers_x
    import helpers_y
    emitted, helpers, _, _ = _trace([entry_mod1.kernel1, entry_mod2.kernel2])
    names = _helper_names(helpers)
    assert set(names) == {_canonical(helpers_x.shared), _canonical(helpers_y.shared)}
    assert _canonical(helpers_x.shared) != _canonical(helpers_y.shared)
    _no_dup_defs(helpers)


def test_hard_collision_guard_raises_on_two_objects_same_emitted_name():
    """If a same-file (kept-original-name) helper and a cross-file helper would land on the same
    emitted name, the tracer raises rather than emit a duplicate ``def``."""
    with pytest.raises(_TraceError, match=r"both be emitted as 'helpers_x__shared'"):
        _trace_referenced(
            [guard_collision.kernel_collides],
            kernel_module=guard_collision.kernel_collides.__module__,
        )


def test_local_const_import_hoisted():
    """A body-local ``from .constants import TILE_SHAPE`` captures TILE_SHAPE as a const, deletes the
    import line from the packed source, and leaves the body referencing TILE_SHAPE."""
    _, helpers, consts, _ = _trace([import_hoist_entries.entry_hoist_const])
    code = _nodoc_join(helpers)
    assert consts == {"TILE_SHAPE": (1, 4, 1, 64)}
    assert not re.search(r"^\s*(import |from \S+ import )", code, re.M)
    assert "return TILE_SHAPE" in code


@pytest.mark.parametrize(("entry", "helper_paths", "absent"), [
    ("entry_hoist_module", ["import_hoist_lib.g"], ["import import_hoist_lib", "import_hoist_lib.g()"]),
    ("entry_hoist_alias", ["import_hoist_pkg.libmod.g"], ["h()"]),
    ("entry_hoist_multi", ["import_hoist_pkg.libmod.g", "import_hoist_pkg.libmod.k"], []),
    ("entry_hoist_dotted_module", ["import_hoist_pkg.sub.fn"], ["z.fn()"]),
])
def test_local_import_hoist_forms(entry, helper_paths, absent):
    """A body-local import of a module, an alias, several names, or a dotted module is hoisted: the
    target is packed and rewritten under its canonical name, and the import line is gone."""
    _, helpers, _, _ = _trace([getattr(import_hoist_entries, entry)])
    code = _nodoc_join(helpers)
    names = _helper_names(helpers)
    for path in helper_paths:
        canonical = _canonical(_resolve_dotted(path))
        assert canonical in names
        assert canonical + "()" in code
    assert not re.search(r"^\s*(import |from \S+ import )", code, re.M)
    for token in absent:
        assert token not in code


def test_conditional_local_import_hoisted_unconditionally():
    """A conditionally-executed local import is hoisted unconditionally: SCALE captured, import gone."""
    _, helpers, consts, _ = _trace([import_hoist_entries.entry_hoist_conditional])
    assert consts == {"SCALE": 7}
    assert not re.search(r"^\s*(import |from \S+ import )", _nodoc_join(helpers), re.M)


@pytest.mark.parametrize(("entry", "absent", "present"), [
    # ``def h(x: UserClass)`` packs as ``def <canon>(x)``, so the snippet imports with no NameError
    # on UserClass and the parser treats x as a non-tensor param.
    pytest.param("entry_unemittable_annotation", ("UserClass",),
                 (r"def \S+__uses_unemittable_class\(x\):",), id="unemittable"),
    # A mixed subscript ``Optional[UserClass]`` has the whole annotation stripped.
    pytest.param("entry_optional_unemittable", ("UserClass", "Optional"),
                 (r"def \S+__uses_optional_unemittable\(x\):",), id="optional_subscript"),
    # ``def h(x: UserClass = 3)`` keeps the default, because the delete span is name-end anchored
    # rather than a backward ``:`` scan.
    pytest.param("entry_unemittable_with_default", ("UserClass",),
                 (r"def \S+__uses_unemittable_with_default\(x ?= ?3\):",), id="default_preserved"),
    # ``def h(a: int, x: UserClass, y: KeepMe)`` strips only x and y; ``a: int`` is untouched.
    pytest.param("entry_multi_param_one_stripped", ("UserClass", "KeepMe"),
                 (r"def \S+__uses_multi_param_one_stripped\(a: int, x, y\):",), id="multi_param"),
    # Controls: a pre-import root and a builtin are neither stripped nor raised.
    pytest.param("entry_tensor_annotation_ok", (), ("x: torch.Tensor",), id="tensor_kept"),
    pytest.param("entry_int_annotation_ok", (), ("x: int", "-> int"), id="int_kept"),
])
def test_annotation_strip_forms(entry, absent, present):
    """An un-emittable param/return annotation is stripped from the packed def; an emittable one
    survives verbatim."""
    _, helpers, _, _ = _trace([getattr(guards, entry)])
    code = _nodoc_join(helpers)
    for name in absent:
        assert name not in code
    for pattern in present:
        assert re.search(pattern, code), pattern


def _packed_helper_module(helpers, consts, dtype_consts):
    """Import the packed helper sources (with captured consts/dtypes as a header) into a live module,
    so a packed def's live ``__annotations__`` can be inspected."""
    import importlib.util as _u
    import tempfile
    header = "import torch\nimport pypto\n"
    header += "".join(f"{n} = {v!r}\n" for n, v in consts.items())
    header += "".join(f"{n} = pypto.{v.name}\n" for n, v in dtype_consts.items())
    src = header + "\n\n" + "\n\n".join(helpers)
    tf = tempfile.NamedTemporaryFile("w", suffix=".py", delete=False)
    tf.write(src)
    tf.close()
    spec = _u.spec_from_file_location("packed_annot_mod_" + str(abs(hash(src))), tf.name)
    mod = _u.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod, src


def _tensor_param_count(func) -> int:
    """Number of params whose live annotation is a ``pypto.Tensor`` (has ``.to_tensor``)."""
    return sum(1 for a in getattr(func, "__annotations__", {}).values() if hasattr(a, "to_tensor"))


def test_pep563_tensor_helper_packs_live_count1():
    """A PEP 563 helper ``def tensor_helper(x: pypto.Tensor([...]))`` packs, and its packed def's live
    annotation is a pypto.Tensor: the count is the load-bearing assertion, not merely "parses"."""
    _, helpers, consts, dtc = _trace([pep563_entries.entry_uses_pep563])
    code = _nodoc_join(helpers)
    assert "pypto.Tensor([...])" in code and "from __future__" not in code
    mod, _src = _packed_helper_module(helpers, consts, dtc)
    fn = next(getattr(mod, n) for n in dir(mod) if n.endswith("__tensor_helper"))
    assert _tensor_param_count(fn) == 1


def test_pep563_tensor_helper_with_module_consts_count1():
    """``def h(x: pypto.Tensor(SHAPE, DTYPE))`` captures SHAPE (const) and DTYPE (dtype channel)
    against the helper's own globals; the annotation stays live and the tensor-param count is 1."""
    _, helpers, consts, dtc = _trace([pep563_entries.entry_pep563_const_tensor])
    assert consts == {"SHAPE": (1, 4, 1, 64)}
    assert list(dtc) == ["DTYPE"]
    mod, _src = _packed_helper_module(helpers, consts, dtc)
    fn = next(getattr(mod, n) for n in dir(mod) if n.endswith("__const_tensor_helper"))
    assert _tensor_param_count(fn) == 1


def test_stdlib_top_module_classification():
    """``_stdlib_top_module`` gates on the top-level name minus the side-effecting denylist."""
    assert _stdlib_top_module("math") == "math"
    assert _stdlib_top_module("os.path") == "os"
    assert _stdlib_top_module("numpy") is None
    assert _stdlib_top_module("numpy.linalg") is None
    assert _stdlib_top_module("antigravity") is None  # denylisted -> None (falls to raise)
    assert _stdlib_top_module("this") is None


@pytest.mark.parametrize(("entry", "match", "also_in_message"), [
    pytest.param(guards.entry_dynamic_import, r"DYNAMIC import", (), id="dynamic_import"),
    # A tensor-typed annotation whose root (``MyTensor``) is not a pre-import.
    pytest.param(pep563_entries.entry_pep563_alias_tensor,
                 r"TENSOR-typed param/return annotation referencing 'MyTensor'", (),
                 id="tensor_annotation_alias"),
    pytest.param(guards.entry_torch_dtype_value, r"cannot pack the reference", (),
                 id="torch_dtype_value"),
    pytest.param(guards.entry_class_value, r"cannot pack the reference 'UserClass'", (),
                 id="class_value"),
    pytest.param(guards.entry_uses_jit_helper, r"@pypto.frontend.jit kernel", (),
                 id="jit_kernel_as_helper"),
    pytest.param(guards.entry_shadow_and_global_use, r"shadow-and-global-use", (),
                 id="shadow_and_global_use"),
    # A module-scope ``numpy.prod(...)`` build-time value, with the torch+pypto(+stdlib) / constant /
    # inline guidance in the message.
    pytest.param(import_hoist_entries.entry_module_scope_thirdparty, r"external library 'numpy'",
                 ("torch+pypto", "constant"), id="module_scope_thirdparty"),
    # A function-local ``import numpy`` build-time value.
    pytest.param(import_hoist_entries.entry_hoist_numpy, r"external library 'numpy'", (),
                 id="local_numpy"),
    # A body-local reference to a denylisted stdlib module: it is never imported (no browser launch)
    # and never emitted.
    pytest.param(import_hoist_entries.entry_hoist_denylisted, r"antigravity.*not safe to import", (),
                 id="local_denylisted"),
    # A module-scope reference to the same module gets the specific message, not the generic
    # unpackable raise.
    pytest.param(import_hoist_entries.entry_module_scope_denylisted,
                 r"stdlib module 'antigravity' is not safe to import", (),
                 id="module_scope_denylisted"),
    # A body-local ``import import_hoist_lib`` whose bound name is also a genuine module-global load in the same
    # body (via a nested ``global import_hoist_lib``).
    pytest.param(import_hoist_entries.entry_hoist_shadow_and_global_use, r"shadow-and-global-use", (),
                 id="local_import_shadow_and_global_use"),
    # Two helpers binding the same alias to different stdlib modules cannot share the flat snippet
    # namespace.
    pytest.param(stdlib_forms.entry_alias_collision, r"both bind 'j'", (),
                 id="stdlib_alias_collision"),
])
def test_trace_raises_on_unpackable_reference(entry, match, also_in_message):
    """Every ``_TraceError`` guard fires on the input it names, at pack time."""
    with pytest.raises(_TraceError, match=match) as exc:
        _trace_referenced([entry])
    for text in also_in_message:
        assert text in str(exc.value)


def test_local_var_shadowing_helper_not_rewritten():
    """A local named ``tile`` shadows the global helper ``tile``, so the local is not rewritten while
    the module-global dotted ``pkg_a.compute.tile`` still is, and the comment beside it survives."""
    import pkg_a.compute as ca
    emitted, _, _, _ = _trace([entries.entry_local_shadows_helper])
    rewritten = _rewritten_entry(entries.entry_local_shadows_helper, emitted)
    assert "tile = x + 1" in rewritten  # local assignment untouched
    assert "return tile + " + _canonical(ca.tile) in rewritten  # local load untouched; dotted rewritten
    assert "# local binding shadows the global 'tile'" in rewritten


def test_comprehension_var_not_rewritten():
    """A comprehension iteration var ``bias_a`` (PEP 709-inlined on 3.12+) shadows the global helper
    name, so it is not rewritten (asserted on this Python version)."""
    emitted, helpers, _, _ = _trace([entries.entry_comprehension_var])
    rewritten = _rewritten_entry(entries.entry_comprehension_var, emitted)
    assert "[bias_a for bias_a in range(x)]" in rewritten
    assert helpers == []  # nothing packed (the only ``bias_a`` occurrences are the comp var)


def test_lambda_var_shadowing_helper_not_rewritten():
    """A lambda param ``tile`` shadows the global helper ``tile``, so it is not rewritten."""
    emitted, helpers, _, _ = _trace([entries.entry_lambda_var])
    rewritten = _rewritten_entry(entries.entry_lambda_var, emitted)
    assert "lambda tile: tile + 1" in rewritten
    assert helpers == []


def test_entry_equals_helper_same_object_uses_original_name():
    """A reference resolving to an entry object is rewritten to the entry's original name, never a
    canonical ``module__name`` that is not emitted."""
    emitted, helpers, _, _ = _trace([entries.entry_calls_entry, entries.a_hook])
    assert emitted[id(entries.a_hook)] == "a_hook"
    rewritten = _rewritten_entry(entries.entry_calls_entry, emitted)
    assert "return a_hook(x)" in rewritten
    # a_hook is an entry, not a discovered helper, so it is not in the packed helper sources.
    assert "a_hook" not in " ".join(_helper_names(helpers))


def test_overlapping_chain_rewrites_only_resolving_prefix():
    """``pkg_a.compute.tile()[0]`` rewrites only ``pkg_a.compute.tile``; the trailing call and
    subscript are untouched (longest resolving prefix, no nested-span double-edit)."""
    import pkg_a.compute as ca
    emitted, _, _, _ = _trace([chains.entry_overlapping_chain])
    code = _code_only(_rewritten_entry(chains.entry_overlapping_chain, emitted))
    assert _canonical(ca.tile) + "()[0]" in code
    assert "pkg_a.compute.tile" not in code


@pytest.mark.parametrize(("entry", "helper"), [
    pytest.param(guards.entry_helper_attr_unpackable, "tile_helper", id="plain_helper"),
    # A decorated helper is unwrapped by the tracer, so the attribute is reachable only on the marker.
    pytest.param(guards.entry_marker_helper_attr_unpackable, "marker_helper", id="decorated_helper"),
])
def test_helper_module_scope_attribute_raises(entry, helper):
    """A reference walking THROUGH an attribute assigned to a helper at module scope raises at pack
    time: only the helper's ``def`` is packed, so ``<helper>.spec`` would fail at deploy."""
    with pytest.raises(_TraceError, match=rf"cannot pack the reference 'helper_attrs.{helper}.spec'"):
        _trace_referenced([entry])


def test_helper_attribute_forms_that_pack():
    """The attribute forms that ARE packable must not raise and must keep their tail: a function's own
    dunders, a function-valued attribute (resolves at full chain length), and a constant leaf."""
    import helper_attrs
    emitted, helpers, consts, _ = _trace([guards.entry_helper_attr_forms])
    canon, sub_canon = _canonical(helper_attrs.tile_helper), _canonical(helper_attrs.sub_helper)
    assert canon in _helper_names(helpers)
    assert sub_canon in _helper_names(helpers)  # helper.sub, a function-valued attribute
    # A tail resolving to a simple value is captured as a constant, named by its last segment.
    assert consts == {"__name__": "tile_helper", "co_argcount": 1, "n": 8}
    code = _code_only(_rewritten_entry(guards.entry_helper_attr_forms, emitted))
    # A tail that does not (a dict, a code object) keeps the rewritten prefix and its tail verbatim.
    assert f"{canon}.__globals__" in code and f"{canon}.__code__" in code
    assert sub_canon in code
    assert "helper_attrs.tile_helper" not in code


def test_matched_prefix_walk_keeps_the_attribute_tail():
    """``<mod>.helper.__globals__`` rewrites to ``<packed>.__globals__``, not to ``<packed>``: the
    span editor replaces only the matched prefix, so the trailing attribute survives.

    This pins CURRENT behaviour. It does not assert that the ``__globals__`` SEMANTICS are right —
    what a packed helper's ``__globals__`` evaluates to inside the snippet is a separate question.
    """
    import helper_attrs
    emitted, _, _, _ = _trace([guards.entry_helper_attr_forms])
    code = _code_only(_rewritten_entry(guards.entry_helper_attr_forms, emitted))
    canon = _canonical(helper_attrs.tile_helper)
    assert f"{canon}.__globals__" in code
    assert re.search(re.escape(canon) + r"(?!\.)", code) is None  # never rewritten to a bare prefix


def test_line_split_chain_rewritten():
    """A chain whose segments span multiple physical lines (``pkg_a.\\n compute.\\n tile``) rewrites to
    the canonical name, exercising ``_node_char_span`` with ``end_lineno != lineno``."""
    import pkg_a.compute as ca
    emitted, helpers, _, _ = _trace([chains.entry_line_split_chain])
    assert _helper_names(helpers) == [_canonical(ca.tile)]
    code = _code_only(_rewritten_entry(chains.entry_line_split_chain, emitted))
    assert _canonical(ca.tile) + "()[0]" in code
    assert "pkg_a." not in code and "compute." not in code  # the old split dotted text is gone


def test_non_ascii_identifier_span():
    """A reference to a non-ASCII-named helper (``λ_helper``) is canonicalized and rewritten correctly
    (byte-to-char ``col_offset`` conversion), and comments and adjacent tokens are preserved."""
    import nonascii_helper
    emitted, helpers, _, _ = _trace([chains.entry_non_ascii])
    rewritten = _rewritten_entry(chains.entry_non_ascii, emitted)
    assert _canonical(nonascii_helper.λ_helper) + "()" in rewritten
    assert "λ_helper()" not in rewritten
    assert "# noqa: PLC2401" in rewritten  # a comment on the μ line is preserved


def test_determinism_snapshot():
    """Repeated traces of the same entry produce identical emitted names and helper sources (pinned
    traversal: BFS FIFO plus AST source order within a function)."""
    import pkg_a.compute as ca
    import pkg_b.compute as cb
    runs = [_trace_referenced([entries.entry_cross_file_same_name]) for _ in range(3)]
    helper_sets = [tuple(_helper_names(helper_sources)) for _emitted, helper_sources, *_rest in runs]
    assert helper_sets[0] == helper_sets[1] == helper_sets[2]
    # Pin the exact canonical names emitted.
    assert set(helper_sets[0]) == {
        _canonical(ca.bias), _canonical(cb.bias), _canonical(ca._amount), _canonical(cb._amount),
    }


@pytest.mark.parametrize(("entry", "stdlib", "present", "absent", "absent_consts"), [
    # A module-scope ``math.prod(shape)`` build-time value: recorded in the statement channel,
    # captured as no const, chain kept verbatim.
    pytest.param(import_hoist_entries.entry_module_scope_stdlib, {"import math"}, ("math.prod(shape)",), (),
                 ("math",), id="module_scope_plain"),
    # A function-local ``import math; math.prod(s)``: the import line is deleted from the packed body.
    pytest.param(import_hoist_entries.entry_hoist_stdlib, {"import math"}, ("math.prod(shape)",),
                 ("import math",), (), id="local_plain"),
    # Plain ``import os`` with ``os.path.join``: the submodule walk must not rename through
    # ``os.path`` (a module named ``posixpath``).
    pytest.param(stdlib_forms.entry_plain_dotted, {"import os"}, ("os.path.join",), (), (),
                 id="module_scope_plain_dotted_attr"),
    pytest.param(stdlib_forms.entry_aliased_root, {"import os as _os"}, ("_os.path.sep",), (), (),
                 id="module_scope_aliased_root"),
    # ``import xml as x`` with ``x.etree.ElementTree...`` gives two statements: the dotted import
    # loads the submodules, the alias line re-binds exactly what the author bound.
    pytest.param(stdlib_forms.entry_aliased_top_deep_chain,
                 {"import xml.etree.ElementTree", "import xml as x"},
                 ("x.etree.ElementTree.Comment",), (), (), id="module_scope_aliased_top_deep_chain"),
    pytest.param(stdlib_forms.entry_local_aliased, {"import math as m"}, ("m.prod(shape)",),
                 ("import math as m",), (), id="local_aliased"),
    pytest.param(stdlib_forms.entry_local_dotted, {"import xml.etree.ElementTree"},
                 ("xml.etree.ElementTree.Comment",), (), (), id="local_dotted"),
    # ``from os import path`` (a module member): the target's real ``__name__`` reproduces the
    # binding.
    pytest.param(stdlib_forms.entry_local_from_import, {"import posixpath as path"}, ("path.sep",),
                 (), (), id="local_from_import_module"),
])
def test_stdlib_import_statement_forms(entry, stdlib, present, absent, absent_consts):
    """Every stdlib import form reaches the header as the statement that reproduces the author's own
    binding, and the reference chain stays verbatim in the packed body."""
    _emitted, helpers, registry, stdlib_stmts, _bindings = _trace_referenced([entry])
    assert stdlib_stmts == stdlib
    code = _nodoc_join(helpers)
    for text in present:
        assert text in code
    for text in absent:
        assert text not in code
    for name in absent_consts:
        assert name not in registry.consts


def test_stdlib_top_module_without_stdlib_module_names(monkeypatch):
    """On an interpreter that provides no ``sys.stdlib_module_names``, membership is decided by
    import-machinery origin: builtins and stdlib-dir modules classify, denylisted and site-packages
    modules do not."""
    monkeypatch.delattr(sys, "stdlib_module_names", raising=False)
    assert _stdlib_top_module("math") == "math"
    assert _stdlib_top_module("os.path") == "os"
    assert _stdlib_top_module("antigravity") is None
    assert _stdlib_top_module("pytest") is None  # site-packages must never classify as stdlib


def test_denylist_guard_without_stdlib_module_names(monkeypatch):
    """The side-effecting denylist does not depend on ``sys.stdlib_module_names``: the guard must
    raise, never import, regardless of interpreter support."""
    monkeypatch.delattr(sys, "stdlib_module_names", raising=False)
    with pytest.raises(_TraceError, match="not safe to import"):
        kernel_tracer._denylist_guard("antigravity")
