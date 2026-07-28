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
"""Tests for system operation DSL parsing and unified printing."""

import pypto_pro
import pypto_pro.language as pl
import pytest


def test_sync_src_print_style():
    """Test unified printing for pl.system.sync_src."""

    @pl.program
    class Before:
        @pl.function
        def main(self, x: pl.Tensor[[64], pl.DT_FP32]) -> pl.Tensor[[64], pl.DT_FP32]:
            pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
            return x

    printed = pypto_pro.ir.python_print(Before)
    assert "ir.call @system.sync_src(" in printed
    assert "set_pipe" in printed
    assert "MTE2" in printed
    assert "wait_pipe" in printed
    assert "PipeType.V" in printed
    assert "event_id" in printed
    assert "0" in printed


def test_bar_all_print_style():
    """Test unified printing for pl.system.bar_all."""

    @pl.program
    class Before:
        @pl.function
        def main(self, x: pl.Tensor[[64], pl.DT_FP32]) -> pl.Tensor[[64], pl.DT_FP32]:
            pl.system.bar_all()
            return x

    printed = pypto_pro.ir.python_print(Before)
    assert "ir.call @system.bar_all()" in printed


def test_multiple_system_ops_print_style():
    """Test unified printing with multiple system ops in a single function."""

    @pl.program
    class Before:
        @pl.function
        def main(self, x: pl.Tensor[[64], pl.DT_FP32]) -> pl.Tensor[[64], pl.DT_FP32]:
            pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
            pl.system.bar_v()
            pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=0)
            pl.system.bar_all()
            return x

    printed = pypto_pro.ir.python_print(Before)
    assert "ir.call @system.sync_src(" in printed
    assert "ir.call @system.bar_v()" in printed
    assert "ir.call @system.sync_dst(" in printed
    assert "ir.call @system.bar_all()" in printed


def test_sync_with_different_pipe_types():
    """Test sync ops with various PipeType enum values."""

    @pl.program
    class Before:
        @pl.function
        def main(self, x: pl.Tensor[[64], pl.DT_FP32]) -> pl.Tensor[[64], pl.DT_FP32]:
            pl.system.sync_src(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.M, event_id=1)
            pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.S, event_id=2)
            return x

    printed = pypto_pro.ir.python_print(Before)
    assert "ir.PipeType.MTE1" in printed
    assert "ir.PipeType.M" in printed
    assert "ir.PipeType.MTE3" in printed
    assert "ir.PipeType.S" in printed


def test_dcci_gm_tensor_with_offset_print_style():
    """Test unified printing for pl.system.dcci with GM tensor and offset."""

    @pl.program
    class Before:
        @pl.function
        def main(self, x: pl.Tensor[[16, 16], pl.DT_FP32]) -> pl.Tensor[[16, 16], pl.DT_FP32]:
            pl.system.dcci(x, [0, 0], cache_line=pl.CacheLine.SINGLE_CACHE_LINE, dst=pl.DcciDst.CACHELINE_OUT)
            return x

    printed = pypto_pro.ir.python_print(Before)
    assert "ir.call @system.dcci(" in printed
    assert "cache_line=" in printed
    assert "dst=" in printed


def test_dcci_gm_tensor_rejects_float_scalar_offset():
    """Test pl.system.dcci rejects float scalar offset for GM tensor."""

    with pytest.raises(RuntimeError, match="scalar integer element offset"):

        @pl.program
        class Before:
            @pl.function
            def main(self, x: pl.Tensor[[16, 16], pl.DT_FP32]) -> pl.Tensor[[16, 16], pl.DT_FP32]:
                pl.system.dcci(x, 1.5)
                return x


def test_dcci_gm_tensor_rejects_float_tuple_offset():
    """Test pl.system.dcci rejects float tuple element offset for GM tensor."""

    with pytest.raises(RuntimeError, match="per-dimension list/tuple"):

        @pl.program
        class Before:
            @pl.function
            def main(self, x: pl.Tensor[[16, 16], pl.DT_FP32]) -> pl.Tensor[[16, 16], pl.DT_FP32]:
                pl.system.dcci(x, [1.5, 0])
                return x


def test_dcci_with_tuple_offset_print_style():
    """Test unified printing for pl.system.dcci with tuple offset."""

    @pl.program
    class Before:
        @pl.function
        def main(self, x: pl.Tensor[[16, 16], pl.DT_FP32]) -> pl.Tensor[[16, 16], pl.DT_FP32]:
            pl.system.dcci(x, (1, 2))
            return x

    printed = pypto_pro.ir.python_print(Before)
    assert "ir.call @system.dcci(" in printed
