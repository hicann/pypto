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

"""Parser tests for the public ``pl.simt.cast`` API.

Operation-level pl.simt.cast tests parse a captured kernel through
``KernelDef.parse_target_program`` so diagnostics and generated IR use the
same parser entry as production kernels.
"""

from pypto_pro import ir
import pypto_pro.language as pl
from pypto_pro.language.parser.diagnostics import ParserSyntaxError, ParserTypeError
import pytest


def test_simt_cast_is_not_exported_as_shared_language_api():
    assert not hasattr(pl, "scalar_cast")


def test_simt_cast_rejects_ordinary_function():
    with pytest.raises(ParserSyntaxError, match="can only be used inside a SIMT function"):

        @pl.jit(auto_mutex=False)
        def unsupported_context(value: pl.DT_INT64):
            _test_result = pl.simt.cast(value, pl.DT_INT32)

        unsupported_context.to_kernel_def().parse_target_program(ir.SectionKind.Vector)


def test_simt_cast_is_available_inside_simt_helper():
    @pl.simt.function(max_threads=1)
    def cast_value(value: pl.DT_FP32):
        _test_result = pl.simt.cast(value, pl.DT_FP16, mode=pl.RoundMode.CAST_RINT)

    @pl.jit(auto_mutex=False)
    def kernel(value: pl.DT_FP32):
        with pl.section_vector():
            pl.simt.launch(cast_value, threads=1, args=(value,))

    program, _ = kernel.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    cast_value = program.get_function(cast_value.__name__)

    function_ir = str(cast_value)
    assert "simt.cast" in function_ir


def test_simt_cast_rejects_unsupported_dtype_pair():
    with pytest.raises(ParserTypeError, match=r"pl\.simt\.cast\(\) does not support"):

        @pl.simt.function(max_threads=1)
        def unsupported(value: pl.DT_FP16):
            _test_result = pl.simt.cast(value, pl.DT_INT32)

        @pl.jit(auto_mutex=False)
        def kernel(value: pl.DT_FP16):
            with pl.section_vector():
                pl.simt.launch(unsupported, threads=1, args=(value,))

        kernel.to_kernel_def().parse_target_program(ir.SectionKind.Vector)


def test_simt_cast_rejects_odd_rounding_for_bfloat16():
    with pytest.raises(ParserTypeError, match=r"pl\.simt\.cast\(\) does not support"):

        @pl.simt.function(max_threads=1)
        def unsupported_mode(value: pl.DT_FP32):
            _test_result = pl.simt.cast(value, pl.DT_BF16, mode=pl.RoundMode.CAST_ODD)

        @pl.jit(auto_mutex=False)
        def kernel(value: pl.DT_FP32):
            with pl.section_vector():
                pl.simt.launch(unsupported_mode, threads=1, args=(value,))

        kernel.to_kernel_def().parse_target_program(ir.SectionKind.Vector)


def test_simt_cast_rejects_tile_operand():
    @pl.simt.function
    def tile_operand(value):
        _ = pl.simt.cast(value, pl.DT_FP16)

    @pl.simt.function(max_threads=1)
    def entry(value):
        tile_operand(value)

    @pl.jit
    def kernel(_jit_entry: pl.DT_INT64):
        tile_type = pl.TileType(shape=[1, 1], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
        value = pl.make_tile(tile_type, addr=0, size=4)
        with pl.section_vector():
            pl.simt.launch(entry, threads=1, args=(value,))

    with pytest.raises(ParserTypeError, match="value must be a scalar expression"):
        kernel.to_kernel_def().parse_target_program(ir.SectionKind.Vector)


def test_simt_cast_rejects_plain_integer_round_mode():
    with pytest.raises(ParserTypeError, match="expects an enum value"):

        @pl.simt.function(max_threads=1)
        def integer_mode(value: pl.DT_FP32):
            _test_result = pl.simt.cast(value, pl.DT_FP16, mode=1)

        @pl.jit(auto_mutex=False)
        def kernel(value: pl.DT_FP32):
            with pl.section_vector():
                pl.simt.launch(integer_mode, threads=1, args=(value,))

        kernel.to_kernel_def().parse_target_program(ir.SectionKind.Vector)


def test_simt_cast_rejects_wrong_positional_arity():
    with pytest.raises(ParserSyntaxError, match="requires exactly 2 positional arguments"):

        @pl.simt.function(max_threads=1)
        def missing_dtype(value: pl.DT_FP32):
            _test_result = pl.simt.cast(value)

        @pl.jit(auto_mutex=False)
        def kernel(value: pl.DT_FP32):
            with pl.section_vector():
                pl.simt.launch(missing_dtype, threads=1, args=(value,))

        kernel.to_kernel_def().parse_target_program(ir.SectionKind.Vector)


def test_simt_cast_rejects_non_dtype_target():
    with pytest.raises(ParserTypeError, match=r"dtype must be a pl\.DT_\* value"):

        @pl.simt.function(max_threads=1)
        def invalid_dtype(value: pl.DT_FP32):
            _test_result = pl.simt.cast(value, 1)

        @pl.jit(auto_mutex=False)
        def kernel(value: pl.DT_FP32):
            with pl.section_vector():
                pl.simt.launch(invalid_dtype, threads=1, args=(value,))

        kernel.to_kernel_def().parse_target_program(ir.SectionKind.Vector)


def test_simt_cast_rejects_unexpected_keyword():
    with pytest.raises(ParserSyntaxError, match="only accepts one optional keyword argument"):

        @pl.simt.function(max_threads=1)
        def unexpected_keyword(value: pl.DT_FP32):
            _test_result = pl.simt.cast(value, pl.DT_FP16, saturate=True)

        @pl.jit(auto_mutex=False)
        def kernel(value: pl.DT_FP32):
            with pl.section_vector():
                pl.simt.launch(unexpected_keyword, threads=1, args=(value,))

        kernel.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
