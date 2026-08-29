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
"""Unit tests for explicit-output block operations."""

import inspect

from pypto_pro import ir
import pypto_pro.language as pl
from pypto_pro.language.parser.diagnostics import ParserSyntaxError, ParserTypeError
import pytest

from pypto.pypto_impl import ir as _ir


def _program_ir(func: ir.Function) -> str:
    return str(func)


def test_coordinate_apis_use_sequence_parameters():
    assert list(inspect.signature(pl.insert).parameters) == ["dst_tile", "src_tile", "offset"]
    assert list(inspect.signature(pl.set_validshape).parameters) == ["tile", "shape"]


def test_manual_add():
    @pl.jit(auto_mutex=False)
    def main(
        a: pl.Tensor[[128, 128], pl.DT_FP32],
        b: pl.Tensor[[128, 128], pl.DT_FP32],
        output: pl.Tensor[[128, 128], pl.DT_FP32],
    ):
        tile_type = pl.TileType(shape=[32, 32], dtype=pl.DT_FP32)
        tile_a = pl.make_tile(tile_type, addr=0x0000, size=4096)
        tile_b = pl.make_tile(tile_type, addr=0x1000, size=4096)
        tile_out = pl.make_tile(tile_type, addr=0x2000, size=4096)
        pl.load(tile_a, a, [0, 0])
        pl.load(tile_b, b, [0, 0])
        pl.add(tile_out, tile_a, tile_b)
        _test_result = pl.store(output, tile_out, [0, 0])

    main_program, _ = main.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    main = main_program.get_function(main.__name__)

    assert "block.add" in _program_ir(main)


def test_manual_mul_scalar():
    @pl.jit(auto_mutex=False)
    def main(
        a: pl.Tensor[[128, 128], pl.DT_FP32],
        output: pl.Tensor[[128, 128], pl.DT_FP32],
    ):
        tile_type = pl.TileType(shape=[32, 32], dtype=pl.DT_FP32)
        tile_a = pl.make_tile(tile_type, addr=0x3000, size=4096)
        tile_out = pl.make_tile(tile_type, addr=0x4000, size=4096)
        pl.load(tile_a, a, [0, 0])
        pl.mul(tile_out, tile_a, 2.0)
        _test_result = pl.store(output, tile_out, [0, 0])

    main_program, _ = main.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    main = main_program.get_function(main.__name__)

    assert "block.muls" in _program_ir(main)


def test_manual_add_scalar():
    @pl.jit(auto_mutex=False)
    def main(
        a: pl.Tensor[[128, 128], pl.DT_FP32],
        output: pl.Tensor[[128, 128], pl.DT_FP32],
    ):
        tile_type = pl.TileType(shape=[32, 32], dtype=pl.DT_FP32)
        tile_a = pl.make_tile(tile_type, addr=0x3000, size=4096)
        tile_out = pl.make_tile(tile_type, addr=0x4000, size=4096)
        pl.load(tile_a, a, [0, 0])
        pl.add(tile_out, tile_a, 1.0)
        _test_result = pl.store(output, tile_out, [0, 0])

    main_program, _ = main.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    main = main_program.get_function(main.__name__)

    assert "block.adds" in _program_ir(main)


def test_manual_maximum_scalar():
    @pl.jit(auto_mutex=False)
    def main(
        a: pl.Tensor[[128, 128], pl.DT_FP32],
        output: pl.Tensor[[128, 128], pl.DT_FP32],
    ):
        tile_type = pl.TileType(shape=[32, 32], dtype=pl.DT_FP32)
        tile_a = pl.make_tile(tile_type, addr=0x3000, size=4096)
        tile_out = pl.make_tile(tile_type, addr=0x4000, size=4096)
        pl.load(tile_a, a, [0, 0])
        pl.maximum(tile_out, tile_a, 0.0)
        _test_result = pl.store(output, tile_out, [0, 0])

    main_program, _ = main.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    main = main_program.get_function(main.__name__)

    assert "block.maxs" in _program_ir(main)


def test_manual_relu():
    @pl.jit(auto_mutex=False)
    def main(
        a: pl.Tensor[[128, 128], pl.DT_FP32],
        output: pl.Tensor[[128, 128], pl.DT_FP32],
    ):
        tile_type = pl.TileType(shape=[32, 32], dtype=pl.DT_FP32)
        tile_a = pl.make_tile(tile_type, addr=0x5000, size=4096)
        tile_out = pl.make_tile(tile_type, addr=0x6000, size=4096)
        pl.load(tile_a, a, [0, 0])
        pl.relu(tile_out, tile_a)
        _test_result = pl.store(output, tile_out, [0, 0])

    main_program, _ = main.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    main = main_program.get_function(main.__name__)

    assert "block.relu" in _program_ir(main)


def test_manual_cmp():
    @pl.jit(auto_mutex=False)
    def main(
        a: pl.Tensor[[128, 128], pl.DT_FP32],
        b: pl.Tensor[[128, 128], pl.DT_FP32],
        output: pl.Tensor[[128, 128], pl.DT_FP32],
    ):
        tile_type = pl.TileType(shape=[32, 32], dtype=pl.DT_FP32)
        tile_a = pl.make_tile(tile_type, addr=0x7000, size=4096)
        tile_b = pl.make_tile(tile_type, addr=0x8000, size=4096)
        tile_out = pl.make_tile(tile_type, addr=0x9000, size=4096)
        pl.load(tile_a, a, [0, 0])
        pl.load(tile_b, b, [0, 0])
        pl.eq(tile_out, tile_a, tile_b)
        _test_result = pl.store(output, tile_out, [0, 0])

    main_program, _ = main.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    main = main_program.get_function(main.__name__)

    assert "block.cmp" in _program_ir(main)


def test_manual_row_max():
    @pl.jit(auto_mutex=False)
    def main(
        inp: pl.Tensor[[128, 128], pl.DT_FP32],
        output: pl.Tensor[[128, 1], pl.DT_FP32],
    ):
        in_type = pl.TileType(shape=[32, 128], dtype=pl.DT_FP32)
        out_type = pl.TileType(shape=[32, 1], dtype=pl.DT_FP32)
        tile_in = pl.make_tile(in_type, addr=0xA000, size=16384)
        tmp = pl.make_tile(out_type, addr=0xE000, size=128)
        tile_out = pl.make_tile(out_type, addr=0x12000, size=128)
        pl.load(tile_in, inp, [0, 0])
        pl.maximum(tile_out, tile_in, tmp, dim=0)
        _test_result = pl.store(output, tile_out, [0, 0])

    main_program, _ = main.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    main = main_program.get_function(main.__name__)

    assert "block.row_max" in _program_ir(main)


def test_manual_col_expand_mul():
    @pl.jit(auto_mutex=False)
    def main(
        col: pl.Tensor[[128, 128], pl.DT_FP32],
        tile: pl.Tensor[[128, 128], pl.DT_FP32],
        output: pl.Tensor[[128, 128], pl.DT_FP32],
    ):
        tile_type = pl.TileType(shape=[32, 32], dtype=pl.DT_FP32)
        col_type = pl.TileType(shape=[1, 32], dtype=pl.DT_FP32)
        tile_a = pl.make_tile(tile_type, addr=0x13000, size=4096)
        tile_col = pl.make_tile(col_type, addr=0x14000, size=128)
        tile_out = pl.make_tile(tile_type, addr=0x15000, size=4096)
        pl.load(tile_a, tile, [0, 0])
        pl.load(tile_col, col, [0, 0])
        pl.expand_mul(tile_out, tile_a, tile_col, dim=1)
        _test_result = pl.store(output, tile_out, [0, 0])

    main_program, _ = main.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    main = main_program.get_function(main.__name__)

    assert "block.col_expand_mul" in _program_ir(main)


def test_manual_col_expand_div():
    @pl.jit(auto_mutex=False)
    def main(
        col: pl.Tensor[[128, 128], pl.DT_FP32],
        tile: pl.Tensor[[128, 128], pl.DT_FP32],
        output: pl.Tensor[[128, 128], pl.DT_FP32],
    ):
        tile_type = pl.TileType(shape=[32, 32], dtype=pl.DT_FP32)
        col_type = pl.TileType(shape=[1, 32], dtype=pl.DT_FP32)
        tile_a = pl.make_tile(tile_type, addr=0x13000, size=4096)
        tile_col = pl.make_tile(col_type, addr=0x14000, size=128)
        tile_out = pl.make_tile(tile_type, addr=0x15000, size=4096)
        pl.load(tile_a, tile, [0, 0])
        pl.load(tile_col, col, [0, 0])
        pl.expand_div(tile_out, tile_a, tile_col, dim=1)
        _test_result = pl.store(output, tile_out, [0, 0])

    main_program, _ = main.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    main = main_program.get_function(main.__name__)

    assert "block.col_expand_div" in _program_ir(main)


def test_manual_and_scalar():
    @pl.jit(auto_mutex=False)
    def main(
        a: pl.Tensor[[128, 128], pl.DT_INT32],
        output: pl.Tensor[[128, 128], pl.DT_INT32],
    ):
        tile_type = pl.TileType(shape=[32, 32], dtype=pl.DT_INT32)
        tile_a = pl.make_tile(tile_type, addr=0x16000, size=4096)
        tile_out = pl.make_tile(tile_type, addr=0x17000, size=4096)
        pl.load(tile_a, a, [0, 0])
        pl.and_(tile_out, tile_a, 7)
        _test_result = pl.store(output, tile_out, [0, 0])

    main_program, _ = main.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    main = main_program.get_function(main.__name__)

    assert "block.ands" in _program_ir(main)


def test_manual_xor():
    @pl.jit(auto_mutex=False)
    def main(
        a: pl.Tensor[[128, 128], pl.DT_INT32],
        b: pl.Tensor[[128, 128], pl.DT_INT32],
        output: pl.Tensor[[128, 128], pl.DT_INT32],
    ):
        tile_type = pl.TileType(shape=[32, 32], dtype=pl.DT_INT32)
        tile_a = pl.make_tile(tile_type, addr=0x18000, size=4096)
        tile_b = pl.make_tile(tile_type, addr=0x19000, size=4096)
        tmp = pl.make_tile(tile_type, addr=0x1A000, size=4096)
        tile_out = pl.make_tile(tile_type, addr=0x1B000, size=4096)
        pl.load(tile_a, a, [0, 0])
        pl.load(tile_b, b, [0, 0])
        pl.xor(tile_out, tile_a, tile_b, tmp)
        _test_result = pl.store(output, tile_out, [0, 0])

    main_program, _ = main.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    main = main_program.get_function(main.__name__)

    assert "block.xor" in _program_ir(main)


def test_manual_addc():
    @pl.jit(auto_mutex=False)
    def main(
        a: pl.Tensor[[128, 128], pl.DT_FP32],
        b: pl.Tensor[[128, 128], pl.DT_FP32],
        c: pl.Tensor[[128, 128], pl.DT_FP32],
        output: pl.Tensor[[128, 128], pl.DT_FP32],
    ):
        tile_type = pl.TileType(shape=[32, 32], dtype=pl.DT_FP32)
        tile_a = pl.make_tile(tile_type, addr=0x1C000, size=4096)
        tile_b = pl.make_tile(tile_type, addr=0x1D000, size=4096)
        tile_c = pl.make_tile(tile_type, addr=0x1E000, size=4096)
        tile_out = pl.make_tile(tile_type, addr=0x1F000, size=4096)
        pl.load(tile_a, a, [0, 0])
        pl.load(tile_b, b, [0, 0])
        pl.load(tile_c, c, [0, 0])
        pl.addc(tile_out, tile_a, tile_b, tile_c)
        _test_result = pl.store(output, tile_out, [0, 0])

    main_program, _ = main.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    main = main_program.get_function(main.__name__)

    assert "block.addc" in _program_ir(main)


def test_load_with_dynamic_valid_shape():
    @pl.jit(auto_mutex=False)
    def main(
        a: pl.Tensor[[128, 128], pl.DT_FP32],
        rows: pl.DT_INT64,
        cols: pl.DT_INT64,
        output: pl.Tensor[[128, 128], pl.DT_FP32],
    ):
        tile_type = pl.TileType(shape=[128, 128], dtype=pl.DT_FP32, valid_shape=[-1, -1])
        tile = pl.make_tile(tile_type, addr=0x20000, size=65536)
        pl.load(tile, a, [0, 0])
        pl.set_validshape(tile, [rows, cols])
        _test_result = pl.store(output, tile, [0, 0])

    main_program, _ = main.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    main = main_program.get_function(main.__name__)

    ir_str = _program_ir(main)
    assert "block.load" in ir_str
    assert "block.set_validshape" in ir_str


def test_insert_with_offset_sequence():
    @pl.jit(auto_mutex=False)
    def main(
        output: pl.Tensor[[32, 32], pl.DT_FP32],
    ):
        dst_type = pl.TileType(shape=[32, 32], dtype=pl.DT_FP32)
        src_type = pl.TileType(shape=[16, 16], dtype=pl.DT_FP32)
        dst = pl.make_tile(dst_type, addr=0x30200, size=4096)
        src = pl.make_tile(src_type, addr=0x31200, size=1024)
        pl.insert(dst, src, [8, 16])
        _test_result = pl.store(output, dst, [0, 0])

    main_program, _ = main.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    main = main_program.get_function(main.__name__)

    ir_str = _program_ir(main)
    assert "block.insert" in ir_str


def test_manual_transpose():
    @pl.jit(auto_mutex=False)
    def main(
        a: pl.Tensor[[128, 128], pl.DT_FP16],
        output: pl.Tensor[[128, 128], pl.DT_FP16],
    ):
        src_type = pl.TileType(shape=[8, 16], dtype=pl.DT_FP16)
        dst_type = pl.TileType(shape=[16, 8], dtype=pl.DT_FP16)
        src = pl.make_tile(src_type, addr=0x30000, size=256)
        dst = pl.make_tile(dst_type, addr=0x30100, size=256)
        pl.load(src, a, [0, 0])
        pl.transpose(dst, src)
        _test_result = pl.store(output, dst, [0, 0])

    main_program, _ = main.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    main = main_program.get_function(main.__name__)

    assert "block.transpose" in _program_ir(main)


def test_get_block_idx_without_prefix():
    @pl.jit(auto_mutex=False)
    def main(output: pl.Tensor[[1], pl.DT_INT64]):
        _ = pl.get_block_idx()
        _test_result = output

    main_program, _ = main.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    main = main_program.get_function(main.__name__)

    assert "get_block_idx" in _program_ir(main)


def test_set_validshape_tile_group_4buf():
    @pl.jit(auto_mutex=False)
    def main(
        a: pl.Tensor[[128, 128], pl.DT_FP16],
        rows: pl.DT_INT64,
        cols: pl.DT_INT64,
        output: pl.Tensor[[128, 128], pl.DT_FP16],
    ):
        tile_type = pl.TileType(shape=[128, 128], dtype=pl.DT_FP16, valid_shape=[-1, -1])
        a_db = pl.make_tile_group(
            type=tile_type, addrs=0x0000, mutex_ids=[0, 1, 2, 3])
        pl.set_validshape(a_db, [rows, cols])
        tile_a = a_db.next()
        pl.load(tile_a, a, [0, 0])
        _test_result = pl.store(output, tile_a, [0, 0])

    main_program, _ = main.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    main = main_program.get_function(main.__name__)

    ir_str = _program_ir(main)
    assert "block.set_validshape" in ir_str
    vs_pos = ir_str.index("block.set_validshape")
    load_pos = ir_str.index("block.load")
    assert vs_pos < load_pos


def test_set_validshape_tile_group_single_tile():
    @pl.jit(auto_mutex=False)
    def main(
        a: pl.Tensor[[128, 128], pl.DT_FP16],
        cols: pl.DT_INT64,
        output: pl.Tensor[[128, 128], pl.DT_FP16],
    ):
        tile_type = pl.TileType(shape=[128, 128], dtype=pl.DT_FP16, valid_shape=[-1, -1])
        a_db = pl.make_tile_group(
            type=tile_type, addrs=0x0000, mutex_ids=[0])
        pl.set_validshape(a_db, [128, cols])
        tile_a = a_db.next()
        pl.load(tile_a, a, [0, 0])
        _test_result = pl.store(output, tile_a, [0, 0])

    main_program, _ = main.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    main = main_program.get_function(main.__name__)

    ir_str = _program_ir(main)
    assert "block.set_validshape" in ir_str
    vs_pos = ir_str.index("block.set_validshape")
    load_pos = ir_str.index("block.load")
    assert vs_pos < load_pos


def test_move_to_insert_subblock():
    @pl.jit(auto_mutex=False)
    def main(
        output: pl.Tensor[[32, 32], pl.DT_FP32],
    ):
        dst_type = pl.TileType(shape=[32, 32], dtype=pl.DT_FP32)
        src_type = pl.TileType(shape=[16, 16], dtype=pl.DT_FP32)
        dst = pl.make_tile(dst_type, addr=0x30200, size=4096)
        src = pl.make_tile(src_type, addr=0x31200, size=1024)
        pl.move(dst, src, [8, 8])
        _test_result = pl.store(output, dst, [0, 0])

    main_program, _ = main.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    main = main_program.get_function(main.__name__)

    ir_str = _program_ir(main)
    assert "block.insert" in ir_str
    assert "block.move" not in ir_str


def test_move_to_insert_transpose_layout():
    @pl.jit(auto_mutex=False)
    def main(
        output: pl.Tensor[[32, 32], pl.DT_FP32],
    ):
        dst_type = pl.TileType(
            shape=[32, 64], dtype=pl.DT_FP32,
            target_memory=pl.MemorySpace.Vec, layout=pl.ZN)
        src_type = pl.TileType(
            shape=[64, 16], dtype=pl.DT_FP32,
            target_memory=pl.MemorySpace.Vec, layout=pl.NZ)
        dst = pl.make_tile(dst_type, addr=0x30200, size=16384)
        src = pl.make_tile(src_type, addr=0x31200, size=4096)
        pl.move(dst, src, [0, 0])
        _test_result = pl.store(output, dst, [0, 0])

    main_program, _ = main.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    main = main_program.get_function(main.__name__)

    ir_str = _program_ir(main)
    assert "block.insert" in ir_str
    assert "block.move" not in ir_str


def test_move_equal_shape_stays_move():
    @pl.jit(auto_mutex=False)
    def main(
        output: pl.Tensor[[32, 32], pl.DT_FP32],
    ):
        dst_type = pl.TileType(shape=[32, 32], dtype=pl.DT_FP32)
        src_type = pl.TileType(shape=[32, 32], dtype=pl.DT_FP32)
        dst = pl.make_tile(dst_type, addr=0x30200, size=4096)
        src = pl.make_tile(src_type, addr=0x31200, size=4096)
        pl.move(dst, src, [0, 0])
        _test_result = pl.store(output, dst, [0, 0])

    main_program, _ = main.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    main = main_program.get_function(main.__name__)

    ir_str = _program_ir(main)
    assert "block.move" in ir_str


def test_move_src_larger_stays_move():
    @pl.jit(auto_mutex=False)
    def main(
        output: pl.Tensor[[32, 32], pl.DT_FP32],
    ):
        dst_type = pl.TileType(shape=[16, 16], dtype=pl.DT_FP32)
        src_type = pl.TileType(shape=[32, 32], dtype=pl.DT_FP32)
        dst = pl.make_tile(dst_type, addr=0x30200, size=1024)
        src = pl.make_tile(src_type, addr=0x31200, size=4096)
        pl.move(dst, src, [0, 0])
        _test_result = pl.store(output, dst, [0, 0])

    main_program, _ = main.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    main = main_program.get_function(main.__name__)

    ir_str = _program_ir(main)
    assert "block.move" in ir_str


def test_make_tile_missing_addr_rejected():
    with pytest.raises(ParserTypeError, match="missing required keyword 'addr'"):

        @pl.jit(auto_mutex=False)
        def main(
            a: pl.Tensor[[128, 128], pl.DT_FP16],
            output: pl.Tensor[[128, 128], pl.DT_FP16],
        ):
            tile_type = pl.TileType(shape=[128, 128], dtype=pl.DT_FP16)
            tile_a = pl.make_tile(tile_type, size=32768)
            pl.load(tile_a, a, [0, 0])
            _test_result = pl.store(output, tile_a, [0, 0])

        main.to_kernel_def().parse_target_program(ir.SectionKind.Vector)


def test_make_tile_size_defaults_to_tile_footprint():
    @pl.jit(auto_mutex=False)
    def main(
        a: pl.Tensor[[128, 128], pl.DT_FP16],
        output: pl.Tensor[[128, 128], pl.DT_FP16],
    ):
        tile_type = pl.TileType(shape=[128, 128], dtype=pl.DT_FP16)
        tile_a = pl.make_tile(tile_type, addr=0x0000)
        pl.load(tile_a, a, [0, 0])
        _test_result = pl.store(output, tile_a, [0, 0])

    main_program, _ = main.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    main = main_program.get_function(main.__name__)

    # 128 * 128 elements * 2 bytes (FP16)
    assert 'memref_size=32768' in _program_ir(main)


def test_make_tile_explicit_size_overrides_derived():
    """An NZ tile rounded up to whole fractals reserves more than shape * dtype."""

    @pl.jit(auto_mutex=False)
    def main(
        a: pl.Tensor[[128, 128], pl.DT_FP16],
        output: pl.Tensor[[128, 128], pl.DT_FP16],
    ):
        tile_type = pl.TileType(shape=[128, 128], dtype=pl.DT_FP16)
        tile_a = pl.make_tile(tile_type, addr=0x0000, size=40960)
        pl.load(tile_a, a, [0, 0])
        _test_result = pl.store(output, tile_a, [0, 0])

    main_program, _ = main.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    main = main_program.get_function(main.__name__)

    ir_str = _program_ir(main)
    assert 'memref_size=40960' in ir_str
    assert 'memref_size=32768' not in ir_str


def test_make_tile_runtime_size_rejected_with_compile_time_hint():
    with pytest.raises(ParserTypeError) as excinfo:

        @pl.jit(auto_mutex=False)
        def main(
            a: pl.Tensor[[pl.DYNAMIC, 128], pl.DT_FP16],
            output: pl.Tensor[[128, 128], pl.DT_FP16],
        ):
            tile_type = pl.TileType(shape=[128, 128], dtype=pl.DT_FP16)
            tile_a = pl.make_tile(tile_type, addr=0x0000, size=a.shape[0] * 32)
            pl.load(tile_a, a, [0, 0])
            _test_result = pl.store(output, tile_a, [0, 0])

        main.to_kernel_def().parse_target_program(ir.SectionKind.Vector)

    assert "compile-time integer" in str(excinfo.value)


def test_make_tile_runtime_addr_rejected_with_compile_time_hint():
    with pytest.raises(ParserTypeError) as excinfo:

        @pl.jit(auto_mutex=False)
        def main(
            a: pl.Tensor[[pl.DYNAMIC, 128], pl.DT_FP16],
            output: pl.Tensor[[128, 128], pl.DT_FP16],
        ):
            tile_type = pl.TileType(shape=[128, 128], dtype=pl.DT_FP16)
            tile_a = pl.make_tile(tile_type, addr=a.shape[0] * 32, size=32768)
            pl.load(tile_a, a, [0, 0])
            _test_result = pl.store(output, tile_a, [0, 0])

        main.to_kernel_def().parse_target_program(ir.SectionKind.Vector)

    assert "compile-time integer" in str(excinfo.value)
    assert "runtime value" in str(excinfo.value)


def test_make_tile_non_positive_size_rejected():
    with pytest.raises(ParserTypeError, match="positive byte count"):

        @pl.jit(auto_mutex=False)
        def main(
            a: pl.Tensor[[128, 128], pl.DT_FP16],
            output: pl.Tensor[[128, 128], pl.DT_FP16],
        ):
            tile_type = pl.TileType(shape=[128, 128], dtype=pl.DT_FP16)
            tile_a = pl.make_tile(tile_type, addr=0x0000, size=0)
            pl.load(tile_a, a, [0, 0])
            _test_result = pl.store(output, tile_a, [0, 0])

        main.to_kernel_def().parse_target_program(ir.SectionKind.Vector)


def test_make_tile_addr_after_tile_type():
    """The documented prototype: pl.make_tile(tile_type, *, addr, size=None)."""

    @pl.jit(auto_mutex=False)
    def main(
        a: pl.Tensor[[128, 128], pl.DT_FP16],
        output: pl.Tensor[[128, 128], pl.DT_FP16],
    ):
        tile_type = pl.TileType(shape=[128, 128], dtype=pl.DT_FP16)
        tile_a = pl.make_tile(tile_type, addr=0x1000)
        pl.load(tile_a, a, [0, 0])
        _test_result = pl.store(output, tile_a, [0, 0])

    main_program, _ = main.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    main = main_program.get_function(main.__name__)

    ir_str = _program_ir(main)
    assert 'memref_addr=4096' in ir_str
    assert 'memref_size=32768' in ir_str


def test_make_tile_addr_and_size_after_tile_type():
    @pl.jit(auto_mutex=False)
    def main(
        a: pl.Tensor[[128, 128], pl.DT_FP16],
        output: pl.Tensor[[128, 128], pl.DT_FP16],
    ):
        tile_type = pl.TileType(shape=[128, 128], dtype=pl.DT_FP16)
        tile_a = pl.make_tile(tile_type, addr=0x1000, size=40960)
        pl.load(tile_a, a, [0, 0])
        _test_result = pl.store(output, tile_a, [0, 0])

    main_program, _ = main.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    main = main_program.get_function(main.__name__)

    ir_str = _program_ir(main)
    assert 'memref_addr=4096' in ir_str
    assert 'memref_size=40960' in ir_str


def test_make_tile_builder_form_rejected():
    """make_tile(shape, dtype, target_memory, ...) is the IR builder, not the DSL op."""

    with pytest.raises(ParserTypeError, match="takes a pl.TileType as its first argument"):

        @pl.jit(auto_mutex=False)
        def main(
            a: pl.Tensor[[128, 128], pl.DT_FP16],
            output: pl.Tensor[[128, 128], pl.DT_FP16],
        ):
            tile_a = pl.make_tile([128, 128], pl.DT_FP16, pl.MemorySpace.Vec, 0x2000, 32768)
            pl.load(tile_a, a, [0, 0])
            _test_result = pl.store(output, tile_a, [0, 0])

        main.to_kernel_def().parse_target_program(ir.SectionKind.Vector)


def test_make_tile_misaligned_addr_rejected():
    """addr reaches the alignment check as a plain int, whatever expression wrote it."""

    with pytest.raises(ParserSyntaxError, match="32-byte aligned"):

        @pl.jit(auto_mutex=False)
        def main(
            a: pl.Tensor[[128, 128], pl.DT_FP16],
            output: pl.Tensor[[128, 128], pl.DT_FP16],
        ):
            tile_type = pl.TileType(shape=[128, 128], dtype=pl.DT_FP16)
            tile_a = pl.make_tile(tile_type, addr=0x1)
            pl.load(tile_a, a, [0, 0])
            _test_result = pl.store(output, tile_a, [0, 0])

        main.to_kernel_def().parse_target_program(ir.SectionKind.Vector)


def test_make_tile_runtime_addr_rejected_quoting_source():
    with pytest.raises(ParserTypeError) as excinfo:

        @pl.jit(auto_mutex=False)
        def main(
            a: pl.Tensor[[pl.DYNAMIC, 128], pl.DT_FP16],
            output: pl.Tensor[[128, 128], pl.DT_FP16],
        ):
            tile_type = pl.TileType(shape=[128, 128], dtype=pl.DT_FP16)
            tile_a = pl.make_tile(tile_type, addr=a.shape[0] * 32)
            pl.load(tile_a, a, [0, 0])
            _test_result = pl.store(output, tile_a, [0, 0])

        main.to_kernel_def().parse_target_program(ir.SectionKind.Vector)

    assert "compile-time integer" in str(excinfo.value)
    # the offending expression, not a repr of the parsed IR object
    assert "a.shape[0] * 32" in str(excinfo.value)


def test_make_tile_positional_addr_rejected():
    """The tile type is the only positional argument; addr/size are keywords."""

    with pytest.raises(ParserTypeError, match="takes 1 positional argument .* but 2 were given"):

        @pl.jit(auto_mutex=False)
        def main(
            a: pl.Tensor[[128, 128], pl.DT_FP16],
            output: pl.Tensor[[128, 128], pl.DT_FP16],
        ):
            tile_type = pl.TileType(shape=[128, 128], dtype=pl.DT_FP16)
            tile_a = pl.make_tile(tile_type, 0x1000)
            pl.load(tile_a, a, [0, 0])
            _test_result = pl.store(output, tile_a, [0, 0])

        main.to_kernel_def().parse_target_program(ir.SectionKind.Vector)


def test_make_tile_positional_addr_and_size_rejected_with_keyword_hint():
    """The rejection carries the fix, so the call site does not have to guess."""

    with pytest.raises(ParserTypeError) as excinfo:

        @pl.jit(auto_mutex=False)
        def main(
            a: pl.Tensor[[128, 128], pl.DT_FP16],
            output: pl.Tensor[[128, 128], pl.DT_FP16],
        ):
            tile_type = pl.TileType(shape=[128, 128], dtype=pl.DT_FP16)
            tile_a = pl.make_tile(tile_type, 0x0000, 32768)
            pl.load(tile_a, a, [0, 0])
            _test_result = pl.store(output, tile_a, [0, 0])

        main.to_kernel_def().parse_target_program(ir.SectionKind.Vector)

    assert "takes 1 positional argument (the tile type) but 3 were given" in str(excinfo.value)
    # the hint spells out the accepted spelling
    assert "addr=" in str(excinfo.value)


VEC_BASE = 0x1000


def test_make_tile_addr_from_constant_expression():
    """addr goes through the full compile-time path, not just literals."""

    @pl.jit(auto_mutex=False)
    def main(
        a: pl.Tensor[[128, 128], pl.DT_FP16],
        output: pl.Tensor[[128, 128], pl.DT_FP16],
    ):
        tile_type = pl.TileType(shape=[128, 128], dtype=pl.DT_FP16)
        tile_a = pl.make_tile(tile_type, addr=VEC_BASE + 0x20)
        pl.load(tile_a, a, [0, 0])
        _test_result = pl.store(output, tile_a, [0, 0])

    main_program, _ = main.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    main = main_program.get_function(main.__name__)

    assert 'memref_addr=4128' in _program_ir(main)


@pytest.mark.parametrize("memory", [pl.MemorySpace.Left, pl.MemorySpace.Right, pl.MemorySpace.Acc])
@pytest.mark.parametrize("layout", [pl.ZZ, pl.NN])
def test_a5_rejects_zz_nn_for_cube_buffers(monkeypatch, memory, layout):
    monkeypatch.setenv("PYPTOPRO_JIT_ARCH", "a5")

    with pytest.raises(pl.parser.ParserError, match="do not support"):

        @pl.jit(auto_mutex=False)
        def create_tile(_jit_entry: pl.DT_INT64):
            tt = pl.TileType(
                shape=[128, 128],
                dtype=pl.DT_FP16,
                target_memory=memory,
                layout=layout,
            )
            tile = pl.make_tile(tt, addr=0x00000, size=32768)  # noqa: F841

        create_tile.to_kernel_def().parse_target_program(ir.SectionKind.Vector)


def test_a5_allows_regular_cube_buffer_layouts(monkeypatch):
    monkeypatch.setenv("PYPTOPRO_JIT_ARCH", "a5")

    @pl.jit(auto_mutex=False)
    def create_tile(_jit_entry: pl.DT_INT64):
        tt = pl.TileType(
            shape=[128, 128],
            dtype=pl.DT_FP16,
            target_memory=pl.MemorySpace.Left,
            layout=pl.NZ,
        )
        tile = pl.make_tile(tt, addr=0x00000, size=32768)  # noqa: F841

    create_tile_program, _ = create_tile.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    create_tile = create_tile_program.get_function(create_tile.__name__)

    stmt = create_tile.body.stmts[0] if hasattr(create_tile.body, "stmts") else create_tile.body
    assert isinstance(stmt.var.type.hardware_info, _ir.HardwareInfo)


def test_high_dimensional_nz_load_store_use_last_two_axes_by_default():
    @pl.jit(auto_mutex=False)
    def main(
        inp: pl.Tensor[[2, 3, 64, 64], pl.DT_FP16, pl.NZ],
        out: pl.Tensor[[2, 3, 64, 64], pl.DT_FP16, pl.NZ],
    ):
        tile_type = pl.TileType(
            shape=[16, 16], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec, layout=pl.NZ
        )
        tile = pl.make_tile(tile_type, addr=0x1000, size=512)
        pl.load(tile, inp, [1, 2, 16, 16])
        pl.store(out, tile, [1, 2, 16, 16])

    main_program, _ = main.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    main = main_program.get_function(main.__name__)
    ir_text = _program_ir(main)
    assert "block.load" in ir_text
    assert "block.store" in ir_text


def test_high_dimensional_nz_load_rejects_non_final_transfer_axes():
    with pytest.raises(ParserSyntaxError, match="NZ transfer only supports the last two tensor axes"):

        @pl.jit(auto_mutex=False)
        def main(
            inp: pl.Tensor[[2, 64, 64, 64], pl.DT_FP16, pl.NZ],
        ):
            tile_type = pl.TileType(
                shape=[16, 16], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec, layout=pl.NZ
            )
            tile = pl.make_tile(tile_type, addr=0x1000, size=512)
            pl.load(tile, inp, [0, 0, 0, 0], order=[1, 3])

        main.to_kernel_def().parse_target_program(ir.SectionKind.Vector)


def test_high_dimensional_nz_store_rejects_non_final_transfer_axes():
    with pytest.raises(ParserSyntaxError, match="NZ transfer only supports the last two tensor axes"):

        @pl.jit(auto_mutex=False)
        def main(
            out: pl.Tensor[[2, 64, 64, 64], pl.DT_FP16, pl.NZ],
        ):
            tile_type = pl.TileType(
                shape=[16, 16], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec, layout=pl.NZ
            )
            tile = pl.make_tile(tile_type, addr=0x1000, size=512)
            pl.store(out, tile, [0, 0, 0, 0], order=[1, 3])

        main.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
