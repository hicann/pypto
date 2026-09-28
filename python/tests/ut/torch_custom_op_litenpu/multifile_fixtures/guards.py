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
"""Fixture entries and helpers for the negative (precise-raise) tracer tests.

Each entry or helper here is expected to make the tracer raise ``_TraceError`` with a precise
message, or is a positive control asserting that a legitimate annotation does not raise.
"""
import importlib  # module-scope import (allowed) so the dynamic helper below hits the dynamic path
import os
import sys

import numpy  # a THIRD-PARTY module for the module-value-bare raise (stdlib is supported)
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
# ``kernel_compile_samples`` is a sibling of the test dir, one level above this fixture package.
sys.path.insert(0, os.path.dirname(_HERE))
from annotated_helpers import (  # noqa: E402
    UserClass,
    uses_int_annotation,
    uses_multi_param_one_stripped,
    uses_optional_unemittable,
    uses_tensor_annotation,
    uses_unemittable_class,
    uses_unemittable_with_default,
)
import helper_attrs  # noqa: E402 - a helper carrying module-scope attribute assignments
from kernel_compile_samples import direct_add_kernel_erased  # noqa: E402 - a real @jit kernel object
from pkg_a.compute import tile  # noqa: E402 - a global helper the shadowing helper below also binds


def calls_dynamic_import(x):
    lib = importlib.import_module("math")  # dynamic import call -> raise (import-idiom path)
    return lib.sqrt(x)


def shadowing_helper(x):
    """``tile`` is a global load in the body and a local in the nested lambda, so the name is
    ambiguous and rewriting this helper's source raises."""
    f = lambda tile: tile + 1  # noqa: E731 - ``tile`` LOCAL here
    return f(tile()[0])        # ``tile`` GLOBAL here


# A dynamic import raises. A static function-local import is hoisted instead.
def entry_dynamic_import(x):
    return calls_dynamic_import(x)


# An un-emittable non-tensor helper annotation is stripped (so it packs), vs the positive controls.
def entry_unemittable_annotation(x):
    return uses_unemittable_class(x)


def entry_optional_unemittable(x):
    return uses_optional_unemittable(x)


def entry_unemittable_with_default(x):
    return uses_unemittable_with_default(x)


def entry_multi_param_one_stripped(x):
    return uses_multi_param_one_stripped(x, x, x)


def entry_tensor_annotation_ok(x):
    """Control: a helper annotated ``x: torch.Tensor`` must not raise (pre-import filter)."""
    return uses_tensor_annotation(x)


def entry_int_annotation_ok(x):
    """Control: a helper annotated ``x: int`` must not raise (builtin filter)."""
    return uses_int_annotation(x)


# A module used as a bare value raises.
def entry_module_value_bare(x):
    return numpy  # a THIRD-PARTY module used as a bare value -> raise naming the library


# A torch.dtype or a class used as a bare value raises.
_A_DTYPE = torch.float16  # a torch.dtype OBJECT (not the pre-import module) bound to a module global


def entry_torch_dtype_value(x):
    return _A_DTYPE  # a torch.dtype value -> unpackable


def entry_class_value(x):
    return UserClass  # a class -> unpackable


def entry_uses_jit_helper():
    """References a jit kernel as if it were a plain helper (rejected: only the top-level
    kernel may be jit)."""
    return direct_add_kernel_erased


# Shadow-and-global-use of one name raises.
def entry_shadow_and_global_use(x):
    """References ``shadowing_helper``, whose body uses ``tile`` both as a global load and as a
    nested-lambda local, so packing that helper raises on the ambiguous name."""
    return shadowing_helper(x)


# A reference walking through a helper attribute assigned at MODULE scope raises: the assignment is
# not packed with the helper's source, so the reference would fail at deploy.
def entry_helper_attr_unpackable(x):
    return helper_attrs.tile_helper.spec.width() + helper_attrs.tile_helper(x)


# The same case through a @pypto.frontend.function helper: the tracer unwraps it, so the attribute is
# on the marker and the unwrapped function's own ``__dict__`` is empty.
def entry_marker_helper_attr_unpackable(x):
    return helper_attrs.marker_helper.spec.width() + helper_attrs.marker_helper(x)


# Controls: every attribute form that IS packable - a function's own dunders, a function-valued
# attribute, and a constant leaf reached through an attribute object.
def entry_helper_attr_forms(x):
    g = helper_attrs.tile_helper.__globals__
    code = helper_attrs.tile_helper.__code__
    fname = helper_attrs.tile_helper.__name__
    argc = helper_attrs.tile_helper.__code__.co_argcount
    sub = helper_attrs.tile_helper.sub
    n = helper_attrs.tile_helper.spec.n
    return sub(x) + n + argc + code.co_nlocals + len(g) + len(fname)
