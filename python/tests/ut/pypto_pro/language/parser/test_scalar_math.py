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

"""Parser tests for public ``pl.simt`` scalar math APIs.

Operation-level tests use eager SIMT IR parsing so diagnostics and generated IR
are observable at function definition time. Delayed ``pl.simt.function`` parsing
is covered by ``test_simt.py``.
"""

import pypto_pro.language as pl
from pypto_pro.language.parser.diagnostics import ParserSyntaxError, ParserTypeError
import pytest


def test_scalar_math_is_available_in_simt_functions():
    @pl.function(type=pl.FunctionType.SimtCallee)
    def scalar_math(value: pl.DT_FP32, rhs: pl.DT_FP32, addend: pl.DT_FP32) -> pl.DT_FP32:
        transformed = pl.simt.max(pl.simt.min(value, rhs), pl.simt.abs(addend))
        transformed = pl.simt.rsqrt(pl.simt.sqrt(transformed))
        transformed = pl.simt.exp2(pl.simt.exp(transformed))
        transformed = pl.simt.log2(pl.simt.log(pl.simt.log1p(transformed)))
        transformed = pl.simt.tanh(pl.simt.cos(pl.simt.sin(transformed)))
        transformed = pl.simt.trunc(
            pl.simt.floor(pl.simt.ceil(pl.simt.round(pl.simt.rint(transformed))))
        )
        classified = pl.simt.isnan(transformed) or pl.simt.isinf(transformed)
        if classified:
            transformed = value
        return pl.simt.fma(transformed, rhs, addend)

    function_ir = str(scalar_math)
    for name in (
        "abs",
        "min",
        "max",
        "sqrt",
        "rsqrt",
        "exp",
        "exp2",
        "log",
        "log2",
        "log1p",
        "sin",
        "cos",
        "tanh",
        "rint",
        "round",
        "floor",
        "ceil",
        "trunc",
        "isnan",
        "isinf",
        "fma",
    ):
        assert f"simt.{name}(" in function_ir


def test_scalar_math_supports_fp16_and_bf16_scalars():
    @pl.function(type=pl.FunctionType.SimtCallee)
    def fp16_math(
        src: pl.Tile[[1, 1], pl.DT_FP16],
        dst: pl.Tile[[1, 1], pl.DT_FP16],
    ):
        value = src[0, 0]
        dst[0, 0] = pl.simt.fma(pl.simt.round(pl.simt.sin(value)), value, value)

    @pl.function(type=pl.FunctionType.SimtCallee)
    def bf16_math(
        src: pl.Tile[[1, 1], pl.DT_BF16],
        dst: pl.Tile[[1, 1], pl.DT_BF16],
        flags: pl.Tile[[1, 1], pl.DT_BOOL],
    ):
        value = src[0, 0]
        dst[0, 0] = pl.simt.tanh(pl.simt.exp2(value))
        flags[0, 0] = pl.simt.isinf(value)

    assert "simt.fma(" in str(fp16_math)
    assert "simt.sin(" in str(fp16_math)
    assert "simt.tanh(" in str(bf16_math)
    assert "simt.isinf(" in str(bf16_math)


def test_scalar_math_supports_int64_abs_and_integer_min_max():
    @pl.function(type=pl.FunctionType.SimtCallee)
    def integer_math(
        src: pl.Tile[[1, 2], pl.DT_INT64],
        dst: pl.Tile[[1, 2], pl.DT_INT64],
    ):
        dst[0, 0] = pl.simt.abs(src[0, 0])
        dst[0, 1] = pl.simt.max(pl.simt.min(src[0, 0], src[0, 1]), src[0, 0])

    function_ir = str(integer_math)
    assert "simt.abs(" in function_ir
    assert "simt.min(" in function_ir
    assert "simt.max(" in function_ir


def test_scalar_math_rejects_ordinary_function():
    with pytest.raises(ParserSyntaxError, match="can only be used inside a SIMT function"):

        @pl.function(type=pl.FunctionType.Orchestration)
        def unsupported(value: pl.DT_FP32) -> pl.DT_FP32:
            return pl.simt.exp(value)


def test_scalar_math_rejects_unsupported_dtype_and_mixed_operands():
    with pytest.raises(ParserTypeError, match="supports only fp32"):

        @pl.function(type=pl.FunctionType.SimtCallee)
        def unsupported_log1p(value: pl.Tile[[1, 1], pl.DT_FP16]):
            value[0, 0] = pl.simt.log1p(value[0, 0])

    with pytest.raises(ParserTypeError, match="supports only fp16, bfloat16, fp32"):

        @pl.function(type=pl.FunctionType.SimtCallee)
        def unsupported_exp(value: pl.Tile[[1, 1], pl.DT_INT32]):
            value[0, 0] = pl.simt.exp(value[0, 0])

    with pytest.raises(ParserTypeError, match="same dtype"):

        @pl.function(type=pl.FunctionType.SimtCallee)
        def mixed_min(
            lhs: pl.Tile[[1, 1], pl.DT_FP16],
            rhs: pl.Tile[[1, 1], pl.DT_FP32],
        ):
            lhs[0, 0] = pl.simt.min(lhs[0, 0], rhs[0, 0])


def test_scalar_math_rejects_tile_operand():
    with pytest.raises(ParserTypeError, match="must be a scalar expression"):

        @pl.function(type=pl.FunctionType.SimtCallee)
        def unsupported(value: pl.Tile[[1, 1], pl.DT_FP32]):
            _ = pl.simt.sqrt(value)


def test_scalar_math_rejects_wrong_arity_and_keywords():
    with pytest.raises(ParserSyntaxError, match="requires exactly 3 positional arguments"):

        @pl.function(type=pl.FunctionType.SimtCallee)
        def missing_addend(value: pl.DT_FP32) -> pl.DT_FP32:
            return pl.simt.fma(value, value)

    with pytest.raises(ParserSyntaxError, match="requires exactly 1 positional argument"):

        @pl.function(type=pl.FunctionType.SimtCallee)
        def keyword_argument(value: pl.DT_FP32) -> pl.DT_FP32:
            return pl.simt.exp(value=value)
