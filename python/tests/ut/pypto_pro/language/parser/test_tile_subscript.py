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

"""Parser validation tests for Tile subscript access."""

import pypto_pro.language as pl
from pypto_pro.language.parser.diagnostics import ParserSyntaxError, ParserTypeError
import pytest

from pypto.pypto_impl import ir


def _parse_vector_kernel(kernel) -> None:
    kernel.to_kernel_def().parse_target_program(ir.SectionKind.Vector)


def test_tile_subscript_rejects_negative_constant_index():
    @pl.simt.function(max_threads=32)
    def negative_index(dst):
        dst[-1, 0] = 0

    @pl.jit
    def kernel(_jit_entry: pl.DT_INT64):
        tile_type = pl.TileType(shape=[8, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
        dst = pl.make_tile(tile_type, addr=0, size=2048)
        with pl.section_vector():
            pl.simt.launch(negative_index, threads=32, args=(dst,))

    with pytest.raises(ParserSyntaxError, match="axis 0 must be non-negative, got -1"):
        _parse_vector_kernel(kernel)


def test_tensor_subscript_rejects_constant_index_at_dimension_size():
    @pl.simt.function(max_threads=32)
    def out_of_bounds(dst: pl.Tensor[[8, 64], pl.DT_FP32]):
        dst[8, 0] = 0

    @pl.jit
    def kernel(dst: pl.Tensor[[8, 64], pl.DT_FP32]):
        with pl.section_vector():
            pl.simt.launch(out_of_bounds, threads=32, args=(dst,))

    with pytest.raises(ParserTypeError, match="axis 0 is out of range for dimension size 8"):
        _parse_vector_kernel(kernel)


def test_tile_subscript_read_requires_ub_memory():
    @pl.jit(auto_mutex=False)
    def kernel(_jit_entry: pl.DT_INT64):
        tile_type = pl.TileType(shape=[16, 16], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Mat)
        tile = pl.make_tile(tile_type, addr=0, size=1024)
        with pl.section_vector():
            _value = tile[0, 0]

    with pytest.raises(
        ParserTypeError,
        match=r"getval: Tile element access requires a Vec-memory Tile \(UB\), got Mat",
    ):
        _parse_vector_kernel(kernel)


def test_tile_slice_rejects_negative_static_bound():
    @pl.jit(auto_mutex=False)
    def kernel(_jit_entry: pl.DT_INT64):
        tile_type = pl.TileType(shape=[8, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
        tile = pl.make_tile(tile_type, addr=0, size=2048)
        with pl.section_vector():
            _sub = tile[-1:4, 0:32]

    with pytest.raises(ParserSyntaxError, match="slice start for axis 0 must be non-negative"):
        _parse_vector_kernel(kernel)


def test_tile_slice_rejects_non_integer_static_bound():
    @pl.jit(auto_mutex=False)
    def kernel(_jit_entry: pl.DT_INT64):
        tile_type = pl.TileType(shape=[8, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
        tile = pl.make_tile(tile_type, addr=0, size=2048)
        with pl.section_vector():
            _sub = tile[1.5:4, 0:32]

    with pytest.raises(ParserTypeError, match="slice start for axis 0 must be an integer scalar"):
        _parse_vector_kernel(kernel)


def test_tile_slice_rejects_empty_static_range():
    @pl.jit(auto_mutex=False)
    def kernel(_jit_entry: pl.DT_INT64):
        tile_type = pl.TileType(shape=[8, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
        tile = pl.make_tile(tile_type, addr=0, size=2048)
        with pl.section_vector():
            _sub = tile[4:4, 0:32]

    with pytest.raises(ParserSyntaxError, match=r"axis 0 must satisfy start < min\(end, shape\)"):
        _parse_vector_kernel(kernel)


def test_tile_slice_rejects_non_unit_step():
    @pl.jit(auto_mutex=False)
    def kernel(_jit_entry: pl.DT_INT64):
        tile_type = pl.TileType(shape=[8, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
        tile = pl.make_tile(tile_type, addr=0, size=2048)
        with pl.section_vector():
            _sub = tile[0:8:2, 0:32]

    with pytest.raises(ParserSyntaxError, match="slice step for axis 0 must be the compile-time integer 1"):
        _parse_vector_kernel(kernel)
