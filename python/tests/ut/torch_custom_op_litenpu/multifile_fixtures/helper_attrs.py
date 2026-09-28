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
"""A helper carrying MODULE-SCOPE attribute assignments, for the helper-attribute reference cases.

Packing a helper packs its ``def`` and nothing else, so the ``tile_helper.<attr> = ...`` statements
below never reach the snippet. A reference that walks THROUGH such an attribute must therefore raise
at pack time; the attributes every function object already carries (its dunders) and a chain that
resolves all the way to a packable value (a function, a constant) must still pack.

``tile_helper`` is plain and ``marker_helper`` is decorated, which is the same case reached through two
different objects: the tracer unwraps a ``@pypto.frontend.function`` helper, so the attribute lives on
the marker and the unwrapped function's own ``__dict__`` is empty.
"""
import pypto


class Spec:
    """A plain attribute value: ``n`` is a constant leaf, ``width()`` is not packable."""

    def __init__(self, n):
        self.n = n

    def width(self):
        return self.n * 2


def tile_helper(x):
    return x + 1


def sub_helper(x):
    return x - 1


tile_helper.spec = Spec(8)    # an object attribute: only ``.n`` is a packable leaf
tile_helper.sub = sub_helper  # a function-valued attribute (resolves at full chain length)


@pypto.frontend.function
def marker_helper(x):
    return x * 2


marker_helper.spec = Spec(4)  # the attribute lands on the MARKER, not on the unwrapped function
