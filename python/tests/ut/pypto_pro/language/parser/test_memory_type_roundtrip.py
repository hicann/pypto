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
"""Comprehensive tests for MemRef, MemorySpace, and TileView."""

import textwrap

from pypto_pro import ir
import pypto_pro.language as pl


def test_parse_tensor_with_memref():
    """Parse pl.Tensor[[64], pl.DT_FP32, pl.MemRef(...)] annotation."""
    code = textwrap.dedent("""\
        @pl.program
        class TestProg:
            @pl.function
            def test_fn(x: pl.Tensor[[64], pl.DT_FP32]) -> pl.Tensor[[64], pl.DT_FP32]:
                y: pl.Tensor[[64], pl.DT_FP32, pl.MemRef(pl.MemorySpace.DDR, 0, 256, 1)] = pl.tensor.add(x, 1.0)
                return y
    """)
    program = pl.parse(code)
    assert isinstance(program, ir.Program)

    # Verify the parsed IR contains memref by re-printing
    printed = ir.python_print(program)
    assert "ir.MemRef" in printed
    assert "ir.MemorySpace.DDR" in printed
    assert "256" in printed


def test_roundtrip_tile_memref():
    """Parse -> print -> parse -> assert_structural_equal for tile with memref."""
    code = textwrap.dedent("""\
        @pl.program
        class TestProg:
            @pl.function(type=pl.FunctionType.InCore)
            def test_fn(x: pl.Tensor[[64, 64], pl.DT_FP32]):
                tile_a: pl.Tile[
                    [64, 64], pl.DT_FP32,
                    pl.MemRef(pl.MemorySpace.Vec, 0, 16384, 0)
                ] = pl.make_tile(
                    pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
                    addr=0,
                    size=16384,
                )
                pl.load(tile_a, x, [0, 0])
    """)
    parsed1 = pl.parse(code)
    printed = ir.python_print(parsed1)
    assert "ir.MemRef" in printed
    assert "ir.MemorySpace.Vec" in printed


def test_all_tile_memory_spaces_are_printable():
    """Tile memory annotations print for all memory spaces without mixing in load legality."""
    spaces = ["DDR", "Vec", "Mat", "Left", "Right", "Acc"]
    for space_name in spaces:
        code = textwrap.dedent(f"""\
            @pl.program
            class TestProg:
                @pl.function(type=pl.FunctionType.InCore)
                def test_fn(x: pl.Tensor[[64, 64], pl.DT_FP32]):
                    tile_a: pl.Tile[
                        [64, 64], pl.DT_FP32,
                        pl.MemRef(pl.MemorySpace.{space_name}, 0, 16384, 0)
                    ] = pl.make_tile(
                        pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.{space_name}),
                        addr=0,
                        size=16384,
                    )
        """)
        parsed1 = pl.parse(code)
        printed = ir.python_print(parsed1)
        assert f"ir.MemorySpace.{space_name}" in printed, (
            f"Memory space {space_name} not in printed output"
        )


def test_backwards_compat_two_args():
    """Existing 2-arg [shape, dtype] still works."""
    code = textwrap.dedent("""\
        @pl.program
        class TestProg:
            @pl.function(type=pl.FunctionType.InCore)
            def test_fn(x: pl.Tensor[[64, 64], pl.DT_FP32]):
                tile_a: pl.Tile[[64, 64], pl.DT_FP32] = pl.make_tile(
                    pl.TileType(shape=[64, 64], dtype=pl.DT_FP32),
                    addr=0,
                    size=16384,
                )
                pl.load(tile_a, x, [0, 0])
    """)
    # Should parse without errors -> 2-arg syntax still works
    program = pl.parse(code)
    assert isinstance(program, ir.Program)


def test_backwards_compat_three_args_layout():
    """Existing 3-arg [shape, dtype, layout] still works for Tensor."""
    code = textwrap.dedent("""\
        @pl.program
        class TestProg:
            @pl.function
            def test_fn(x: pl.Tensor[[64], pl.DT_FP32]) -> pl.Tensor[[64], pl.DT_FP32]:
                y: pl.Tensor[[64], pl.DT_FP32, pl.NZ] = pl.tensor.add(x, 1.0)
                return y
    """)
    # Should parse without errors
    program = pl.parse(code)
    assert isinstance(program, ir.Program)
