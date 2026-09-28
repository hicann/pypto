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
"""Entry functions for the function-local import hoist and the external-import (stdlib / third-party)
tracer tests. Each references a helper whose body-local import the tracer must hoist, or a
module-scope stdlib or third-party build-time computation.
"""
import math  # module-scope stdlib import for the build-time value cases (external-imports a/b)
import os
import sys
import types

import numpy  # module-scope third-party import (external-imports b: clean RAISE naming numpy)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# A DENYLISTED stdlib module bound at MODULE scope, WITHOUT importing it (a real ``import antigravity``
# would launch a browser). ``types.ModuleType("antigravity")`` is a genuine module object whose
# ``__name__`` is the denylisted stdlib name, so the tracer classifies the reference as "denylisted"
# (the specific "not safe to import" raise) and never imports it.
antigravity = types.ModuleType("antigravity")
antigravity.geohash = None

from import_hoist_pkg.helpers import (  # noqa: E402
    uses_local_alias_import,
    uses_local_conditional_import,
    uses_local_const_import,
    uses_local_denylisted_stdlib,
    uses_local_dotted_module_import,
    uses_local_import_shadow_global,
    uses_local_module_import,
    uses_local_multi_import,
    uses_local_numpy_import,
    uses_local_stdlib_import,
)


# Positive entries: each helper hoists its body-local import.
def entry_hoist_const(x):
    return uses_local_const_import()


def entry_hoist_module(x):
    return uses_local_module_import()


def entry_hoist_alias(x):
    return uses_local_alias_import()


def entry_hoist_multi(x):
    return uses_local_multi_import()


def entry_hoist_dotted_module(x):
    return uses_local_dotted_module_import()


def entry_hoist_stdlib(x):
    return uses_local_stdlib_import((1, 4, 1, 64))


def entry_hoist_conditional(x):
    return uses_local_conditional_import(True)


# external-imports: MODULE-SCOPE stdlib (a) + third-party (b) build-time helpers
def module_scope_stdlib_tile(shape):
    """A helper computing a build-time value with a MODULE-SCOPE ``import math`` -> header import math,
    math.prod(...) verbatim, no ``math`` const captured (external-imports a)."""
    return math.prod(shape)


def entry_module_scope_stdlib(x):
    return module_scope_stdlib_tile((1, 4, 1, 64))


def module_scope_thirdparty_tile(shape):
    """A build-time value via a MODULE-SCOPE ``import numpy`` -> clean RAISE naming numpy (b)."""
    return numpy.prod(shape)


def entry_module_scope_thirdparty(x):
    return module_scope_thirdparty_tile((1, 4, 1, 64))


def module_scope_denylisted_tile(shape):
    """A MODULE-SCOPE reference to a DENYLISTED stdlib module (``antigravity.geohash``) -> the specific
    "not safe to import into a compile snippet" RAISE (never imported / never emitted in the header)."""
    return antigravity.geohash


def entry_module_scope_denylisted(x):
    return module_scope_denylisted_tile((1, 4, 1, 64))


# residual RAISE entries
def entry_hoist_numpy(x):
    return uses_local_numpy_import((1, 4, 1, 64))


def entry_hoist_denylisted(x):
    return uses_local_denylisted_stdlib()


def entry_hoist_shadow_and_global_use(x):
    """A body-local ``import import_hoist_lib`` whose bound name is also used as a module-global load in the
    same body (via a nested ``global import_hoist_lib``), which raises shadow-and-global-use."""
    return uses_local_import_shadow_global()
