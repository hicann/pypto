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
"""Unit tests for @pl.inline and @pl.vector_function decorators.

The former @pl.function / @pl.program decorators have been removed.
Use @pl.jit to define kernels.
"""

import pypto_pro.language as pl
import pytest


def test_removed_frontend_apis_are_not_exported():
    removed_apis = (
        "function",
        "program",
        "kernel",
        "KernelDef",
        "parse",
        "loads",
        "parse_program",
        "loads_program",
    )

    for name in removed_apis:
        assert not hasattr(pl, name)
        assert not hasattr(pl.parser, name)

    assert not hasattr(pl.parser.decorator, "function")
    assert not hasattr(pl.parser.decorator, "program")


def test_inline_deprecated_warning():
    """@pl.inline is deprecated and should warn at decoration time."""

    with pytest.warns(DeprecationWarning, match="@pl.inline is deprecated"):

        @pl.inline
        def deprecated_inline(x: pl.Tensor[[64], pl.DT_FP32]) -> pl.Tensor[[64], pl.DT_FP32]:
            return x

    assert deprecated_inline.__name__ == "deprecated_inline"


def test_vector_function_marker():
    """@pl.vector_function sets the internal marker attribute."""

    @pl.vector_function
    def vf_body(in_tile, out_tile):
        pass

    assert vf_body._pypto_vector_function is True


def test_is_vector_function():
    """is_vector_function correctly detects decorated functions."""

    @pl.vector_function
    def vf_func(a, b):
        pass

    assert pl.parser.decorator.is_vector_function(vf_func) is True


def test_is_vector_function_false_for_plain():
    """is_vector_function returns False for undecorated functions."""

    def plain_func(a, b):
        pass

    assert pl.parser.decorator.is_vector_function(plain_func) is False
