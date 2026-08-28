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
"""Tiling class Array index syntax validation test (no NPU required).

Positive cases: arr[i] with literal integer index is valid.
Negative cases: slice, range slice, step slice, multi-dim, non-int index are rejected.
"""


from __future__ import annotations

from dataclasses import dataclass

from pypto_pro import ir
import pypto_pro.language as pl
from pypto_pro.language.parser.diagnostics import (
    ParserSyntaxError,
    ParserTypeError,
    UnsupportedFeatureError,
)
import pytest


@dataclass
class MyTiling:
    arr: int[8]


def test_literal_index_0():
    @pl.jit(auto_mutex=False)
    def kernel(_jit_entry: pl.DT_INT64, tiling: MyTiling):
        result: pl.DT_INT32 = tiling.arr[0]
        _test_result = result

    kernel_program, _ = kernel.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    kernel = kernel_program.get_function(kernel.__name__)

    assert isinstance(kernel, ir.Function)


def test_literal_index_5():
    @pl.jit(auto_mutex=False)
    def kernel(_jit_entry: pl.DT_INT64, tiling: MyTiling):
        result: pl.DT_INT32 = tiling.arr[5]
        _test_result = result

    kernel_program, _ = kernel.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    kernel = kernel_program.get_function(kernel.__name__)

    assert isinstance(kernel, ir.Function)


def test_slice_colon_raises():
    with pytest.raises((ParserSyntaxError, ParserTypeError, TypeError, UnsupportedFeatureError)):
        @pl.jit(auto_mutex=False)
        def kernel(_jit_entry: pl.DT_INT64, tiling: MyTiling):
            result: pl.DT_INT32 = tiling.arr[:]
            _test_result = result

        kernel.to_kernel_def().parse_target_program(ir.SectionKind.Vector)


def test_slice_range_raises():
    with pytest.raises((ParserSyntaxError, ParserTypeError, TypeError, UnsupportedFeatureError)):
        @pl.jit(auto_mutex=False)
        def kernel(_jit_entry: pl.DT_INT64, tiling: MyTiling):
            result: pl.DT_INT32 = tiling.arr[1:3]
            _test_result = result

        kernel.to_kernel_def().parse_target_program(ir.SectionKind.Vector)


def test_slice_step_raises():
    with pytest.raises((ParserSyntaxError, ParserTypeError, TypeError, UnsupportedFeatureError)):
        @pl.jit(auto_mutex=False)
        def kernel(_jit_entry: pl.DT_INT64, tiling: MyTiling):
            result: pl.DT_INT32 = tiling.arr[::2]
            _test_result = result

        kernel.to_kernel_def().parse_target_program(ir.SectionKind.Vector)


def test_multi_dim_index_raises():
    with pytest.raises((ParserSyntaxError, ParserTypeError, TypeError)):
        @pl.jit(auto_mutex=False)
        def kernel(_jit_entry: pl.DT_INT64, tiling: MyTiling):
            result: pl.DT_INT32 = tiling.arr[1, 2]
            _test_result = result

        kernel.to_kernel_def().parse_target_program(ir.SectionKind.Vector)


def test_string_index_raises():
    with pytest.raises((ParserSyntaxError, ParserTypeError, TypeError)):
        @pl.jit(auto_mutex=False)
        def kernel(_jit_entry: pl.DT_INT64, tiling: MyTiling):
            result: pl.DT_INT32 = tiling.arr["x"]
            _test_result = result

        kernel.to_kernel_def().parse_target_program(ir.SectionKind.Vector)


def test_string_element_type_raises():
    with pytest.raises(TypeError):
        str[4]


def test_list_element_type_raises():
    from pypto_pro.language.typing._tiling import get_tiling_fields

    @dataclass
    class BadType:
        arr: list[4]

    with pytest.raises(ValueError):
        get_tiling_fields(BadType)
