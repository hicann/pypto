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

"""Parser tests for PyPTO Pro warp-level SIMT operations."""

from pypto_pro._errors import InvalidArgument, InvalidOperation, InvalidType
import pypto_pro.language as pl
import pytest

from pypto.pypto_impl import ir


@pl.vector_function(mode="simt", max_threads=32)
def _warp_ops(value: pl.DT_FP32, predicate_value: pl.DT_INT32):
    lane = pl.simt.lane_id()
    pl.simt.lanemask_eq()
    pl.simt.lanemask_le()
    pl.simt.lanemask_lt()
    pl.simt.lanemask_ge()
    pl.simt.lanemask_gt()
    predicate = lane < 16
    pl.simt.warp_all(predicate)
    pl.simt.warp_any(predicate_value)
    pl.simt.warp_ballot(predicate)
    pl.simt.warp_active_mask()
    warp_width = pl.simt.warp_size()
    subgroup_width = pl.simt.cast(warp_width // 2, pl.DT_INT32)
    pl.simt.warp_shfl(value, 0)
    pl.simt.warp_shfl_up(value, 1, subgroup_width)
    pl.simt.warp_shfl_down(value, 1, width=8)
    pl.simt.warp_shfl_xor(value, 1)
    pl.simt.warp_reduce_add(value)
    pl.simt.warp_reduce_max(value)
    pl.simt.warp_reduce_min(value)


@pl.jit
def _warp_kernel(value: pl.DT_FP32, predicate_value: pl.DT_INT32):
    with pl.section_vector():
        _warp_ops[32](value, predicate_value)


def test_warp_operations_parse_to_registered_ir_calls():
    program, matched = _warp_kernel.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    function_ir = str(program.get_function("_warp_ops"))

    assert matched
    for op_name in (
        "lane_id",
        "lanemask_eq",
        "lanemask_le",
        "lanemask_lt",
        "lanemask_ge",
        "lanemask_gt",
        "warp_all",
        "warp_any",
        "warp_ballot",
        "warp_active_mask",
        "warp_size",
        "warp_shfl",
        "warp_shfl_up",
        "warp_shfl_down",
        "warp_shfl_xor",
        "warp_reduce_add",
        "warp_reduce_max",
        "warp_reduce_min",
    ):
        assert f"simt.{op_name}" in function_ir


def test_shuffle_accepts_runtime_controls_with_asc_dtypes():
    @pl.vector_function(mode="simt", max_threads=32)
    def runtime_shuffle(
        value: pl.DT_FP16,
        src_lane: pl.DT_INT32,
        delta: pl.DT_UINT32,
        width: pl.DT_INT32,
    ):
        pl.simt.warp_shfl(value, src_lane, width)
        pl.simt.warp_shfl_up(value, delta, width)
        pl.simt.warp_shfl_down(value, delta, width)
        pl.simt.warp_shfl_xor(value, src_lane, width)

    @pl.jit
    def kernel(
        value: pl.DT_FP16,
        src_lane: pl.DT_INT32,
        delta: pl.DT_UINT32,
        width: pl.DT_INT32,
    ):
        with pl.section_vector():
            runtime_shuffle[32](value, src_lane, delta, width)

    program, _ = kernel.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    function_ir = str(program.get_function("runtime_shuffle"))
    assert function_ir.count("simt.warp_shfl") == 4


def test_shuffle_rejects_invalid_constant_width():
    @pl.vector_function(mode="simt", max_threads=32)
    def invalid_width(value: pl.DT_FP32):
        pl.simt.warp_shfl(value, 0, 3)

    @pl.jit
    def kernel(value: pl.DT_FP32):
        with pl.section_vector():
            invalid_width[32](value)

    with pytest.raises(InvalidArgument, match="width must be one of"):
        kernel.to_kernel_def().parse_target_program(ir.SectionKind.Vector)


def test_warp_vote_rejects_non_predicate_dtype():
    @pl.vector_function(mode="simt", max_threads=32)
    def invalid_vote(value: pl.DT_FP32):
        pl.simt.warp_ballot(value)

    @pl.jit
    def kernel(value: pl.DT_FP32):
        with pl.section_vector():
            invalid_vote[32](value)

    with pytest.raises(InvalidType, match="predicate must have BOOL or INT32 dtype"):
        kernel.to_kernel_def().parse_target_program(ir.SectionKind.Vector)


def test_shuffle_rejects_wrong_src_lane_dtype():
    @pl.vector_function(mode="simt", max_threads=32)
    def invalid_control(value: pl.DT_FP32, src_lane: pl.DT_UINT32):
        pl.simt.warp_shfl(value, src_lane)

    @pl.jit
    def kernel(value: pl.DT_FP32, src_lane: pl.DT_UINT32):
        with pl.section_vector():
            invalid_control[32](value, src_lane)

    with pytest.raises(InvalidType, match="src_lane must have int32 dtype"):
        kernel.to_kernel_def().parse_target_program(ir.SectionKind.Vector)


def test_warp_operation_rejected_outside_simt_function():
    with pytest.raises(InvalidOperation, match="only be used inside"):

        @pl.jit(auto_mutex=False)
        def bad_lane_id(_jit_entry: pl.DT_INT64):
            pl.simt.lane_id()

        bad_lane_id.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
