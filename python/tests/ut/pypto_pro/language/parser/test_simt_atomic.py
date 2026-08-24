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

"""Parser tests for PyPTO Pro SIMT atomic operations.

Operation-level tests use eager SIMT IR parsing so diagnostics and generated IR
are observable at function definition time. Delayed ``pl.simt.function`` parsing
is covered by the launch integration cases here and in ``test_simt.py``.
"""

import pypto_pro.language as pl
from pypto_pro.language.parser.diagnostics import ParserSyntaxError, ParserTypeError
import pytest

from pypto.pypto_impl import ir


@pl.simt.function(max_threads=32)
def _atomic_add_tile(
    dst: pl.Tile[[1, 32], pl.DT_INT32],
    old_values: pl.Tile[[1, 32], pl.DT_INT32],
    value: pl.DT_INT32,
):
    tid = pl.simt.linear_thread_idx()
    old_values[0, tid] = pl.simt.atomic_add(dst[0, 0], value)


@pl.simt.function(max_threads=32)
def _atomic_add_tensor(
    dst: pl.Tensor[[1, 32], pl.DT_INT64],
    old_values: pl.Tensor[[1, 32], pl.DT_INT64],
    value: pl.DT_INT64,
):
    tid = pl.simt.linear_thread_idx()
    old_values[0, tid] = pl.simt.atomic_add(dst[0, 0], value)


@pl.function(type=pl.FunctionType.SimtCallee)
def _atomic_rmw_ops(
    numeric: pl.Tile[[1, 1], pl.DT_INT32],
    bitwise: pl.Tile[[1, 1], pl.DT_UINT32],
    counter: pl.Tile[[1, 1], pl.DT_UINT32],
    int_value: pl.DT_INT32,
    uint_value: pl.DT_UINT32,
):
    pl.simt.atomic_sub(numeric[0, 0], int_value)
    pl.simt.atomic_exch(numeric[0, 0], int_value)
    pl.simt.atomic_max(numeric[0, 0], int_value)
    pl.simt.atomic_min(numeric[0, 0], int_value)
    pl.simt.atomic_cas(numeric[0, 0], int_value, int_value)
    pl.simt.atomic_and(bitwise[0, 0], uint_value)
    pl.simt.atomic_or(bitwise[0, 0], uint_value)
    pl.simt.atomic_xor(bitwise[0, 0], uint_value)
    pl.simt.atomic_inc(counter[0, 0], uint_value)
    pl.simt.atomic_dec(counter[0, 0], uint_value)


@pl.kernel
def _atomic_add_tile_kernel(value: pl.DT_INT32):
    tile_type = pl.TileType(shape=[1, 32], dtype=pl.DT_INT32, target_memory=pl.MemorySpace.Vec)
    dst = pl.make_tile(tile_type, addr=0x0000, size=128)
    old_values = pl.make_tile(tile_type, addr=0x0080, size=128)
    with pl.section_vector():
        pl.simt.launch(_atomic_add_tile, threads=32, args=(dst, old_values, value))


@pl.kernel
def _atomic_add_tensor_kernel(
    dst: pl.Tensor[[1, 32], pl.DT_INT64],
    old_values: pl.Tensor[[1, 32], pl.DT_INT64],
    value: pl.DT_INT64,
):
    with pl.section_vector():
        pl.simt.launch(_atomic_add_tensor, threads=32, args=(dst, old_values, value))


def test_atomic_add_preserves_tile_lvalue_and_returns_old_value():
    program, matched = _atomic_add_tile_kernel.parse_target_program(ir.SectionKind.Vector)
    function_ir = str(program.get_function("_atomic_add_tile"))

    assert matched
    assert function_ir.count("simt.atomic_add") == 1
    assert "block.getval" not in function_ir
    assert "block.setval" in function_ir
    assert "simt.launch" in str(program)


def test_atomic_add_supports_gm_int64_tensor():
    program, matched = _atomic_add_tensor_kernel.parse_target_program(ir.SectionKind.Vector)
    function_ir = str(program.get_function("_atomic_add_tensor"))

    assert matched
    assert function_ir.count("simt.atomic_add") == 1
    assert "block.getval" not in function_ir
    assert "block.setval" in function_ir
    assert "simt.launch" in str(program)


def test_atomic_add_contextually_types_numeric_literals():
    @pl.function(type=pl.FunctionType.SimtCallee)
    def literal_values(unsigned: pl.Tile[[1, 1], pl.DT_UINT32], signed: pl.Tile[[1, 1], pl.DT_INT32]):
        old = pl.simt.atomic_add(unsigned[0, 0], 1)
        signed[0, 0] = pl.simt.atomic_add(signed[0, 0], -1)
        unsigned[0, 0] = old

    function_ir = str(literal_values)
    assert function_ir.count("simt.atomic_add") == 2


def test_atomic_add_requires_direct_subscript_target():
    with pytest.raises(ParserSyntaxError, match="direct Tile or Tensor subscript"):

        @pl.function(type=pl.FunctionType.SimtCallee)
        def scalar_alias(dst: pl.Tile[[1, 1], pl.DT_INT32], value: pl.DT_INT32):
            current = dst[0, 0]
            pl.simt.atomic_add(current, value)


def test_atomic_add_rejects_slice_target():
    with pytest.raises(ParserSyntaxError, match="does not support slices"):

        @pl.function(type=pl.FunctionType.SimtCallee)
        def slice_target(dst: pl.Tile[[1, 1], pl.DT_INT32], value: pl.DT_INT32):
            pl.simt.atomic_add(dst[0:1, 0:1], value)


@pytest.mark.parametrize("dtype", [pl.DT_FP16, pl.DT_BF16])
def test_half_precision_atomic_add_max_min_are_void_on_ub_tile(dtype):
    @pl.function(type=pl.FunctionType.SimtCallee)
    def supported_tile(dst: pl.Tile[[1, 3], dtype]):
        pl.simt.atomic_add(dst[0, 0], 1.0)
        pl.simt.atomic_max(dst[0, 1], 2.0)
        pl.simt.atomic_min(dst[0, 2], 3.0)

    function_ir = str(supported_tile)
    for op_name in ("atomic_add", "atomic_max", "atomic_min"):
        assert function_ir.count(f"simt.{op_name}") == 1
    atomic_calls = [
        stmt.expr
        for stmt in supported_tile.body.stmts
        if isinstance(stmt, ir.EvalStmt) and isinstance(stmt.expr, ir.Call)
    ]
    assert len(atomic_calls) == 3
    assert all(isinstance(call.type, ir.NoneType) for call in atomic_calls)


@pytest.mark.parametrize("dtype", [pl.DT_FP16, pl.DT_BF16])
def test_half_precision_atomic_add_max_min_are_void_on_gm_tensor(dtype):
    @pl.function(type=pl.FunctionType.SimtCallee)
    def supported_tensor(dst: pl.Tensor[[1, 3], dtype]):
        pl.simt.atomic_add(dst[0, 0], 1.0)
        pl.simt.atomic_max(dst[0, 1], 2.0)
        pl.simt.atomic_min(dst[0, 2], 3.0)

    function_ir = str(supported_tensor)
    for op_name in ("atomic_add", "atomic_max", "atomic_min"):
        assert function_ir.count(f"simt.{op_name}") == 1
    atomic_calls = [
        stmt.expr
        for stmt in supported_tensor.body.stmts
        if isinstance(stmt, ir.EvalStmt) and isinstance(stmt.expr, ir.Call)
    ]
    assert len(atomic_calls) == 3
    assert all(isinstance(call.type, ir.NoneType) for call in atomic_calls)


@pytest.mark.parametrize("dtype", [pl.DT_FP16, pl.DT_BF16])
def test_half_precision_atomic_result_cannot_be_returned(dtype):
    with pytest.raises(ParserTypeError, match="must return None or one scalar value"):

        @pl.function(type=pl.FunctionType.SimtCallee)
        def add_result(dst: pl.Tile[[1, 1], dtype]) -> None:
            return pl.simt.atomic_add(dst[0, 0], 1.0)


def test_atomic_add_requires_exact_value_dtype():
    with pytest.raises(ParserTypeError, match="operand 0 dtype must match target dtype"):

        @pl.function(type=pl.FunctionType.SimtCallee)
        def mismatched_value(dst: pl.Tile[[1, 1], pl.DT_INT32], value: pl.DT_UINT32):
            pl.simt.atomic_add(dst[0, 0], value)


def test_atomic_add_rejects_float_literal_for_integer_target():
    with pytest.raises(ParserTypeError, match="requires an integer value"):

        @pl.function(type=pl.FunctionType.SimtCallee)
        def float_literal(dst: pl.Tile[[1, 1], pl.DT_INT32]):
            pl.simt.atomic_add(dst[0, 0], 1.0)


def test_atomic_rmw_interfaces_preserve_lvalue_and_build_distinct_ir_ops():
    function_ir = str(_atomic_rmw_ops)

    for op_name in (
        "atomic_sub",
        "atomic_exch",
        "atomic_max",
        "atomic_min",
        "atomic_cas",
        "atomic_and",
        "atomic_or",
        "atomic_xor",
        "atomic_inc",
        "atomic_dec",
    ):
        assert function_ir.count(f"simt.{op_name}") == 1
    assert "block.getval" not in function_ir


def test_atomic_cas_contextually_types_compare_and_value_literals():
    @pl.function(type=pl.FunctionType.SimtCallee)
    def literal_operands(dst: pl.Tile[[1, 1], pl.DT_UINT32]):
        old = pl.simt.atomic_cas(dst[0, 0], 0, 1)
        dst[0, 0] = old

    assert "simt.atomic_cas" in str(literal_operands)


def test_atomic_cas_requires_exact_compare_dtype():
    with pytest.raises(ParserTypeError, match="operand 0 dtype must match target dtype"):

        @pl.function(type=pl.FunctionType.SimtCallee)
        def mismatched_compare(
            dst: pl.Tile[[1, 1], pl.DT_INT32],
            compare: pl.DT_UINT32,
            value: pl.DT_INT32,
        ):
            pl.simt.atomic_cas(dst[0, 0], compare, value)


def test_atomic_bitwise_rejects_fp32_target():
    with pytest.raises(ParserTypeError, match="atomic_or.*does not support dtype.*UB Tile"):

        @pl.function(type=pl.FunctionType.SimtCallee)
        def float_bitwise(dst: pl.Tile[[1, 1], pl.DT_FP32], value: pl.DT_FP32):
            pl.simt.atomic_or(dst[0, 0], value)


def test_atomic_counter_rejects_signed_target():
    with pytest.raises(ParserTypeError, match="atomic_inc.*does not support dtype.*UB Tile"):

        @pl.function(type=pl.FunctionType.SimtCallee)
        def signed_counter(dst: pl.Tile[[1, 1], pl.DT_INT32], limit: pl.DT_INT32):
            pl.simt.atomic_inc(dst[0, 0], limit)


def test_atomic_counter_supports_gm_uint64():
    @pl.function(type=pl.FunctionType.SimtCallee)
    def uint64_counter(dst: pl.Tensor[[1], pl.DT_UINT64], limit: pl.DT_UINT64):
        pl.simt.atomic_dec(dst[0], limit)

    assert "simt.atomic_dec" in str(uint64_counter)
