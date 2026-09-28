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
"""Helpers the sample kernel calls, in a different file from the kernel.

* ``tile_helper``: plain helper used as an expression; references the cross-file constant ``TILE``.
* ``add_into``: a ``@pypto.frontend.function`` out-param helper; references the plain ``_bias`` below.
"""

import pypto

from .constants import TILE


def tile_helper():
    """Plain helper returning the tile shape (references the cross-file constant TILE)."""
    return TILE


def _bias(x):
    """Plain transitive helper referenced by add_into."""
    return x


@pypto.frontend.function
def add_into(a, b, out):
    """Out-param helper, inlined by the parser at the call site."""
    out.move(a + _bias(b))
