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
"""Helpers whose bodies perform static function-local imports. Each is packed by the tracer:
the import target is resolved + packed (func/const) or emitted as a header stdlib import, and the
import statement is DELETED from the packed body.
"""
import os as _os
import sys as _sys

_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import import_hoist_lib  # module-scope global (for the shadow-and-global-use residual below)


def uses_local_const_import():
    """A body-local ``from .constants import TILE_SHAPE``: TILE_SHAPE is captured as a const, the
    import line deleted, the body references TILE_SHAPE (canonical const name)."""
    from .constants import TILE_SHAPE
    return TILE_SHAPE


def uses_local_module_import():
    """A body-local bare ``import import_hoist_lib`` with ``import_hoist_lib.g()``: g is packed,
    import_hoist_lib.g() rewritten, import deleted (exercises the module-root namespace injection)."""
    import import_hoist_lib
    return import_hoist_lib.g()


def uses_local_alias_import():
    """An aliased body-local ``from .libmod import g as h; h()`` resolves by identity to g."""
    from .libmod import g as h
    return h()


def uses_local_multi_import():
    """A multi-name body-local ``from .libmod import g, k``: both packed, both rewritten."""
    from .libmod import g, k
    return g() + k()


def uses_local_dotted_module_import():
    """``import import_hoist_pkg.sub as z; z.fn()``: fn packed, chain rewritten, import deleted."""
    import import_hoist_pkg.sub as z
    return z.fn()


def uses_local_stdlib_import(shape):
    """External-imports (c): a body-local ``import math; math.prod(shape)`` -> math emitted in the
    snippet header, math.prod(...) kept verbatim, import deleted."""
    import math
    return math.prod(shape)


def uses_local_conditional_import(flag):
    """A CONDITIONALLY-executed local import is HOISTED UNCONDITIONALLY (the packed copy is a normalized
    form): SCALE is packed as a const and the ``if`` guard's import is deleted."""
    if flag:
        from .constants import SCALE
        return SCALE
    return 0


def uses_local_numpy_import(shape):
    """External-imports (d): a body-local ``import numpy`` for a build-time value -> clean RAISE naming
    numpy (third-party, not in the torch+pypto+stdlib deploy env)."""
    import numpy
    return numpy.prod(shape)


def uses_local_denylisted_stdlib():
    """A body-local import of a DENYLISTED stdlib module (side-effecting) -> clean RAISE, no header
    ``import antigravity``."""
    import antigravity
    return antigravity.geohash


def uses_local_import_shadow_global():
    """Residual: a body-local ``import import_hoist_lib`` binds ``import_hoist_lib`` locally, but a
    nested function reads the MODULE-GLOBAL ``import_hoist_lib`` (``global import_hoist_lib``) — the
    SAME name is used both as a local-import binding and a genuine module-global load, which is
    un-disambiguable by span -> shadow-and-global-use RAISE."""
    import import_hoist_lib

    def _inner():
        global import_hoist_lib
        return import_hoist_lib.g()

    return import_hoist_lib.g() + _inner()
