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
"""Entry functions (kernels/helpers) that reference the fixture packages in every supported style.

Loaded by ``test_multifile_tracer.py`` as a real module so ``inspect.getsource`` and ``__globals__``
resolution work. Each entry function is a plain Python function whose references the tracer must
discover and rewrite by canonical identity. The kernels are plain (not @jit) so the pure-python
tracer can walk them without a real jit build; the snippet-level tests use the
``kernel_compile_samples`` jit kernels instead.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# The many import styles the tracer must unify.
import pkg_a  # module-qualified root + re-export
from pkg_a import compute as short_compute  # short root / aliased module
import pkg_a.compute  # module-qualified deep chain
from pkg_a.compute import bias as bias_a  # aliased from-import: def-name 'bias'
from pkg_a.compute import tile  # bare-name reference (same object also reached dotted)
import pkg_b.compute  # sibling with same-named helpers


def entry_cross_file_same_name(x):
    """pkg_a.compute.bias and pkg_b.compute.bias are different objects sharing the name 'bias'
    (each via its own same-named '_amount'), so they get distinct canonical names and both pack."""
    return pkg_a.compute.bias(x) + pkg_b.compute.bias(x)


def entry_same_object_two_ways(x):
    """``tile`` (bare) and ``pkg_a.compute.tile`` (dotted) are the same object, so they pack once and
    both call sites are rewritten to it (no orphaned bare call)."""
    return tile() + pkg_a.compute.tile()[0] + x


# A module-qualified constant (pkg_a.constants.CONST), captured and rewritten.
def entry_module_qualified_const(x):
    return x + pkg_a.compute.constants.CONST


# An aliased from-import; the def-name is 'bias' but the local binding is 'bias_a'.
def entry_alias_from_import(x):
    return bias_a(x)


# A short module root reaches a helper (resolved by identity, spelling-free).
def entry_short_root(x):
    return short_compute.tile()[0] + x


# A package __init__ re-export reached both via pkg_a.bias and pkg_a.compute.bias (same object).
def entry_reexport_same_object(x):
    return pkg_a.bias(x) + pkg_a.compute.bias(x)


# Two same-named module-qualified consts with different values.
def entry_const_across_files(x):
    return x + pkg_a.compute.constants.CONST + pkg_b.compute.CONST


# Two same-named module-qualified consts with the same value, which dedup to one emitted name.
def entry_const_same_value(x):
    return x + pkg_a.compute.constants.SHARED[0] + pkg_b.compute.SHARED[0]


# Scope units: locals / params / comprehension vars / lambdas shadowing a global helper name.
def entry_local_shadows_helper(x):
    """A local named ``tile`` shadows the global helper ``tile``, so the local must not be rewritten
    while the module-global ``pkg_a.compute.tile`` (dotted) still is."""
    tile = x + 1           # local binding shadows the global 'tile'
    return tile + pkg_a.compute.tile()[0]


def entry_comprehension_var(x):
    """A comprehension var ``bias_a`` shadows the global helper ``bias_a`` (PEP 709-inlined on 3.12+);
    the comp var must not be rewritten (it is local to the comprehension)."""
    return [bias_a for bias_a in range(x)]


def entry_lambda_var(x):
    """A lambda param ``tile`` shadows the global helper ``tile``, so it is not rewritten."""
    f = lambda tile: tile + 1  # noqa: E731
    return f(x)


# entry==helper same object: a reference resolving to an entry func keeps the entry's original name.
import pkg_a.compute as _pc  # noqa: E402


def a_hook(x):
    """A second entry func (passed alongside the kernel), referenced both bare and module-qualified
    from the kernel below. As an entry object it is emitted under its original name, and every
    reference to it resolves to that same object, so it is rewritten to ``a_hook`` rather than a
    canonical ``module__a_hook`` that is not emitted."""
    return _pc.bias(x)  # also reaches a real cross-file helper (bias) so the kernel packs something


def entry_calls_entry(x):
    """Kernel that references the entry func ``a_hook`` (bare). Passed as ``[entry_calls_entry,
    a_hook]`` so ``a_hook`` is an entry and the reference keeps its original name."""
    return a_hook(x)


# Two different helpers whose canonical <module>__<qualname> bases collide (a ``__`` inside a
# segment), so the canonical counter must append ``__2`` to give both a distinct name and def.
import base_collision_pkg.a  # noqa: E402
import base_collision_pkg.a__b  # noqa: E402 - module has a ``__`` in its name


def entry_base_collision(x):
    """``base_collision_pkg.a__b.f`` and ``base_collision_pkg.a.b__f`` both sanitize to base
    ``base_collision_pkg__a__b__f`` but are different objects, so one gets the ``__2`` counter suffix."""
    return base_collision_pkg.a__b.f() + base_collision_pkg.a.b__f() + x
