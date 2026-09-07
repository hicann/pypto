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

"""What every pipeline module needs and none of them owns: the generated-code name
vocabulary, and the syntactic forms the transform has to recognise.

This module imports nothing from its siblings, and nothing here knows about
PipelineInfo, the sync graph or the transform. That is the point: the analyzer, the
scanner, the graph and the validator all need these, and before they lived here each
module either kept its own copy or reached sideways for someone else's — which is how
the analyzer's idea of "a call to a stage" came to differ from the validator's, and how
two modules ended up with separate spellings of "is this a slot accessor".

Keeping it a leaf is also what lets ``_validate`` import from here without an import
cycle; add nothing to this module that needs a sibling.
"""

from __future__ import annotations

import ast
import inspect
import textwrap

# ---------------------------------------------------------------------------
# Generated-code names
#
# The ``_pl_`` prefix is reserved for the transform (validate_names refuses a kernel
# that already uses one). Each name below is written by one module and read back by
# another, so it is spelled once, here, and the halves cannot drift apart.
# ---------------------------------------------------------------------------

# Which task the data in a ctx slot belongs to. The analyzer places it in the ctx layout,
# the transformer both assigns it and reads it back as the event-id index.
PL_TASK_ID_FIELD = "_pl_task_id"
# The validity flag. Prefixed like every framework field so a user struct field named
# `is_valid` cannot collide with it.
PL_IS_VALID_FIELD = "_pl_is_valid"
# Marks a stage argument that is a whole struct: the ctx slot is passed in its place, so
# there is no single field to name (its fields are ctx fields under their own names).
PL_STRUCT_ARG = "_pl_struct_arg"


def slot_index_field(group: str) -> str:
    """Ctx field (and variable) name holding a group's current slot index."""
    return f"_pl_idx_{group}"


# ---------------------------------------------------------------------------
# Syntax recognition
# ---------------------------------------------------------------------------


def call_name(call: ast.Call) -> str:
    """The callee of a plainly-called function: ``f(...)`` -> ``"f"``; anything else -> ``""``.

    Deliberately blind to ``obj.f(...)``. Every caller uses this to recognise a call to a
    name bound in the kernel's closure — a ``@stage`` function — and an attribute call is a
    method on some object that merely shares the name.

    This replaced three near-copies that had drifted apart on exactly that point: the
    analyzer's stage extraction was strict while two of the checks accepted the attribute
    form, so a check could count a loop the extraction would then skip.
    """
    if isinstance(call, ast.Call) and isinstance(call.func, ast.Name):
        return call.func.id
    return ""


# The accessors that resolve a tile group handle to one of its tiles. Only ``next``
# advances the group's cursor; the others read it where it stands, which is why the kind
# is reported rather than just the group (see _buffer_parser._lower_group_accessor).
_GROUP_ACCESSORS = ("next", "current", "previous")


def slot_accessor(node: ast.expr) -> tuple[str, str] | None:
    """``(group, kind)`` if ``node`` selects a slot of a tile group, else None.

    Both spellings are recognised: ``group.next()/current()/previous()`` gives that
    accessor as the kind, and ``group[i]`` gives ``"index"``.

    The group name is returned unfiltered — callers that only care about declared groups
    check membership themselves, because they do not all hold the same set (the kernel
    knows every declaration, a stage body only its own parameters).
    """
    if isinstance(node, ast.Call):
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr in _GROUP_ACCESSORS:
            if isinstance(func.value, ast.Name):
                return func.value.id, func.attr
        return None  # any other call is not an accessor, and must not fall through
    if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name):
        return node.value.id, "index"
    return None


def is_vf_function(func_def: ast.FunctionDef) -> bool:
    """True if ``func_def`` carries the ``@pl.vector_function`` decorator.

    Only the decorator counts. A plain function that merely CALLS a vector function is a
    different thing: the access scan treats it as a VF helper for role propagation, but its
    body can still hold block ops that nothing looks at (see
    _validate._check_no_block_ops_in_helpers).
    """
    for dec in func_def.decorator_list:
        if isinstance(dec, ast.Attribute) and dec.attr == "vector_function":
            return True
        if isinstance(dec, ast.Name) and dec.id == "vector_function":
            return True
    return False


def get_funcdef(fn) -> ast.FunctionDef | None:
    """The ``ast.FunctionDef`` for a Python function object, or None.

    Line numbers are shifted to the ones in the real file. Re-parsing a source snippet
    numbers it from 1, and every diagnostic that names a node inside a stage body reports
    whatever this returns — so without the shift those messages point at a line the user
    cannot find.
    """
    if fn is None:
        return None
    try:
        lines, start_lineno = inspect.getsourcelines(fn)
        mod = ast.parse(textwrap.dedent("".join(lines)))
        # getsourcelines is 1-based and so is the fresh parse, hence the -1.
        ast.increment_lineno(mod, start_lineno - 1)
        for node in mod.body:
            if isinstance(node, ast.FunctionDef):
                return node
    except (OSError, TypeError, SyntaxError):
        return None
    return None
