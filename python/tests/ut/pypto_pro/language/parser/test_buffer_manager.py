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
"""Unit tests for the make_tile_group DSL composite (IR named tuple) and auto_mutex.

``pl.make_tile_group(type=, addrs=, mutex_ids=)`` is a DSL composite: declared inside a
kernel, lowered by the parser to an IR named tuple ``{tiles, mutex_ids, cursor}``.
``group.next()/current()/previous()`` each return a bare tile (not a slot); ``next()``
advances the cursor, ``current()/previous()`` do not. The tile <-> mutex_id mapping is
parser metadata consumed by auto_mutex, so the frontend never handles mutex ids or
manual lock/unlock.
"""


import pypto_pro.language as pl

from pypto.pypto_impl import ir


def _ir_to_str(prog: ir.Program) -> str:
    return str(prog)


def _parse_kernel(kernel_def) -> ir.Program:
    return kernel_def.parse_target_program(ir.SectionKind.Vector)[0]


def test_static_mutex_lock_unlock():
    @pl.kernel
    def k(x: pl.Tensor[[64], pl.DT_FP16]):
        tt = pl.TileType(shape=[64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
        t = pl.make_tile(tt, addr=0, size=128)
        pl.system.mutex_lock(pipe=pl.PipeType.MTE2, mutex_id=0)
        pl.load(t, x, [0])
        pl.system.mutex_unlock(pipe=pl.PipeType.MTE2, mutex_id=0)

    ir_str = _ir_to_str(_parse_kernel(k))
    assert "system.mutex_lock" in ir_str
    assert "system.mutex_unlock" in ir_str


def test_keyword_form():
    @pl.kernel
    def k(x: pl.Tensor[[64], pl.DT_FP16]):
        tt = pl.TileType(shape=[64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
        t = pl.make_tile(tt, addr=0, size=128)
        pl.system.mutex_lock(pipe=pl.PipeType.MTE2, mutex_id=1)
        pl.load(t, x, [0])
        pl.system.mutex_unlock(pipe=pl.PipeType.MTE2, mutex_id=1)

    ir_str = _ir_to_str(_parse_kernel(k))
    assert "system.mutex_lock" in ir_str
    assert "system.mutex_unlock" in ir_str


def test_contiguous_addr_auto_offset():
    @pl.kernel(auto_mutex=True)
    def k(a: pl.Tensor[[128, 128], pl.DT_FP16]):
        tt = pl.TileType(shape=[128, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat)
        db = pl.make_tile_group(type=tt, addrs=0, mutex_ids=[0, 1])
        cur0 = db.next()
        pl.load(cur0, a, [0, 0])
        cur1 = db.next()
        pl.load(cur1, a, [0, 0])

    ir_str = _ir_to_str(_parse_kernel(k))
    # 128*128*2 bytes = 32768 -> second tile at base + 32768.
    assert "ir.MemRef" in ir_str
    assert "32768" in ir_str


def test_discrete_addrs_list():
    @pl.kernel(auto_mutex=True)
    def k(a: pl.Tensor[[128, 128], pl.DT_FP16]):
        tt = pl.TileType(shape=[128, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat)
        db = pl.make_tile_group(type=tt, addrs=[0, 0x10000], mutex_ids=[0, 1])
        cur0 = db.next()
        pl.load(cur0, a, [0, 0])
        cur1 = db.next()
        pl.load(cur1, a, [0, 0])

    ir_str = _ir_to_str(_parse_kernel(k))
    assert "ir.MemRef" in ir_str
    assert "65536" in ir_str


def test_auto_mutex_single_tile():
    @pl.kernel(auto_mutex=True)
    def k(x: pl.Tensor[[64], pl.DT_FP16]):
        tt = pl.TileType(shape=[64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
        g = pl.make_tile_group(type=tt, addrs=0, mutex_ids=[3])
        buf = g.next()
        pl.load(buf, x, [0])
        pl.store(x, buf, [0])

    ir_str = _ir_to_str(_parse_kernel(k))
    assert ir_str.count("mutex_lock") >= 2, ir_str
    assert ir_str.count("mutex_unlock") >= 2, ir_str


def test_auto_mutex_group_loop():
    @pl.kernel(auto_mutex=True)
    def k(gm_q: pl.Tensor[[128, 32], pl.DT_FP16]):
        tt = pl.TileType(shape=[32, 32], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat)
        q_l1_db = pl.make_tile_group(type=tt, addrs=0, mutex_ids=[0, 1])
        for i in pl.range(0, 4):
            cur = q_l1_db.next()
            pl.load(cur, gm_q, [i * 32, 0])

    ir_str = _ir_to_str(_parse_kernel(k))
    assert "mutex_lock" in ir_str, ir_str
    assert "mutex_unlock" in ir_str, ir_str
    assert ir_str.count("block.make_tile") == 2, ir_str


def test_next_advances():
    @pl.kernel(auto_mutex=True)
    def k(gm_q: pl.Tensor[[32, 32], pl.DT_FP16]):
        tt = pl.TileType(shape=[32, 32], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat)
        g = pl.make_tile_group(type=tt, addrs=0, mutex_ids=[0, 1])
        a = g.next()
        pl.load(a, gm_q, [0, 0])
        b = g.next()
        pl.load(b, gm_q, [0, 0])

    ir_str = _ir_to_str(_parse_kernel(k))
    assert ir_str.count("block.make_tile") == 2, ir_str
    assert "struct.create" in ir_str
    # Each next() advances the cursor via a struct.set field write.
    assert ir_str.count("struct.set") >= 2


def test_current_does_not_advance():
    @pl.kernel(auto_mutex=True)
    def k(gm_q: pl.Tensor[[32, 32], pl.DT_FP16]):
        tt = pl.TileType(shape=[32, 32], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat)
        g = pl.make_tile_group(type=tt, addrs=0, mutex_ids=[0, 1])
        a = g.current()
        pl.load(a, gm_q, [0, 0])
        b = g.current()
        pl.load(b, gm_q, [0, 0])

    ir_str = _ir_to_str(_parse_kernel(k))
    assert "struct.create" in ir_str
    # current() never advances -> no cursor writes.
    assert "struct.set" not in ir_str


def test_previous_no_advance():
    @pl.kernel(auto_mutex=True)
    def k(gm_q: pl.Tensor[[64, 32], pl.DT_FP16]):
        tt = pl.TileType(shape=[32, 32], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat)
        g = pl.make_tile_group(type=tt, addrs=0, mutex_ids=[0, 1])
        a = g.next()
        pl.load(a, gm_q, [0, 0])
        prev = g.previous()
        pl.load(prev, gm_q, [0, 0])

    ir_str = _ir_to_str(_parse_kernel(k))
    assert ir_str.count("block.make_tile") == 2, ir_str
    # Only the next() advanced the cursor; previous() does not.
    assert ir_str.count("struct.set") == 1


def test_single_tile_group_fast_path():
    @pl.kernel(auto_mutex=True)
    def k(gm_q: pl.Tensor[[32, 32], pl.DT_FP16]):
        tt = pl.TileType(shape=[32, 32], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat)
        g = pl.make_tile_group(type=tt, addrs=0, mutex_ids=[5])
        cur = g.next()
        pl.load(cur, gm_q, [0, 0])

    ir_str = _ir_to_str(_parse_kernel(k))
    assert "block.make_tile" in ir_str
    # Single-tile group: no cursor struct, no rotate, static mutex lock.
    assert "struct.create" not in ir_str
    assert "struct.set" not in ir_str
    assert "mutex_id" in ir_str
    assert "5" in ir_str
