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
"""Fixtures for the span-editor units: an overlapping chain (only the resolving module-prefix
rewrites), a line-split chain, a non-ASCII helper name, and helpers named torch/pypto."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nonascii_helper import λ_helper  # a non-ASCII helper in its OWN module (span byte->char unit)
import pkg_a.compute  # for a deep module-qualified chain
import user_torch  # a user module whose members are named torch / pypto


def entry_non_ascii(x):
    """References a non-ASCII-named helper in another module, so it is canonicalized and rewritten,
    and a non-ASCII char before it (the ``μ`` below) shifts byte vs char columns for the editor."""
    μ = 0  # noqa: PLC2401 - a non-ASCII local before the reference (byte!=char column)
    return λ_helper() + x + μ


def entry_user_torch_helpers(x):
    """References user helpers literally named ``torch`` / ``pypto`` (distinct objects), so both are
    renamed rather than skipped."""
    return user_torch.torch() + user_torch.pypto() + x


def entry_overlapping_chain(x):
    """The resolvable module-chain ``pkg_a.compute.tile`` must rewrite to only ``pkg_a.compute.tile``
    even when followed by a call and subscript ``()[0]``, which stay untouched."""
    return pkg_a.compute.tile()[0] + x


def entry_line_split_chain(x):
    """The chain segments themselves span multiple physical lines (inside parens), exercising
    ``_node_char_span`` with ``end_lineno != lineno``."""
    return (
        pkg_a.
        compute.
        tile()[0]
        + x
    )
