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

"""CCE code-generation tests for the Tile-first A5 SIMT path."""

import re

import pypto_pro.language as pl
import pytest


def _compile_to_cce(kernel_def, arch: str = "a5") -> str:
    from pypto_pro.runtime.jit import _assemble_cv_source, _parse_and_codegen_targets

    cube, vector = _parse_and_codegen_targets(kernel_def, arch, "")
    return _assemble_cv_source(cube, vector).content


@pl.simt.function(max_threads=256)
def _tile_add(
    dst: pl.Tile[[1, 256], pl.DT_FP32],
    src: pl.Tile[[1, 256], pl.DT_FP32],
    n: pl.DT_UINT32,
    delta: pl.DT_FP32,
):
    tid = pl.simt.linear_thread_idx()
    if tid < n:
        dst[0, tid] = src[0, tid] + delta


@pl.simt.function(max_threads=256)
def _gm_add(
    dst: pl.Tensor[[1, 256], pl.DT_FP32],
    src: pl.Tensor[[1, 256], pl.DT_FP32],
    n: pl.DT_UINT32,
    delta: pl.DT_FP32,
):
    tid = pl.simt.linear_thread_idx()
    if tid < n:
        dst[0, tid] = src[0, tid] + delta


@pl.simt.function
def _callee_add(value: pl.DT_INT32, delta: pl.DT_INT32) -> pl.DT_INT32:
    return value + delta


@pl.simt.function
def _callee_load(src: pl.Tensor[[1, 32], pl.DT_INT32], index: pl.DT_UINT32) -> pl.DT_INT32:
    return src[0, index]


@pl.simt.function
def _callee_store(
    dst: pl.Tile[[1, 32], pl.DT_INT32],
    index: pl.DT_UINT32,
    value: pl.DT_INT32,
):
    dst[0, index] = value


@pl.simt.function
def _callee_apply(
    dst: pl.Tile[[1, 32], pl.DT_INT32],
    src: pl.Tensor[[1, 32], pl.DT_INT32],
    index: pl.DT_UINT32,
    delta: pl.DT_INT32,
):
    value = _callee_add(_callee_load(src, index), delta)
    _callee_store(dst, index, value)


@pl.simt.function(max_threads=32)
def _callee_entry(
    dst: pl.Tile[[1, 32], pl.DT_INT32],
    src: pl.Tensor[[1, 32], pl.DT_INT32],
    delta: pl.DT_INT32,
):
    tid = pl.simt.linear_thread_idx()
    _callee_apply(dst, src, tid, delta)


@pl.simt.function(max_threads=256)
def _context_probe(dst: pl.Tile[[1, 256], pl.DT_UINT32]):
    thread = pl.simt.thread_idx()
    block = pl.simt.block_dim()
    block_id = pl.simt.block_idx()
    grid = pl.simt.grid_dim()
    tid = pl.simt.linear_thread_idx()
    value = (
        thread.x
        + thread.y
        + thread.z
        + block.x
        + block.y
        + block.z
        + block_id.x
        + block_id.y
        + block_id.z
        + grid.x
        + grid.y
        + grid.z
        + pl.simt.warp_size()
    )
    dst[0, tid] = value


@pl.simt.function(max_threads=256)
def _tile_valid_shape_access(
    dst: pl.Tile[[8, 64], pl.DT_FP32],
    src: pl.Tile[[8, 64], pl.DT_FP32],
):
    tid = pl.simt.linear_thread_idx()
    rows = src.valid_shape[0]
    cols = src.valid_shape[1]
    row = tid // cols
    col = tid % cols
    if row < rows:
        dst[row, col] = src[row, col]


@pl.kernel
def _simt_tile_codegen_kernel(
    x: pl.Tensor[[1, 256], pl.DT_FP32],
    out: pl.Tensor[[1, 256], pl.DT_FP32],
    n: pl.DT_UINT32,
    delta: pl.DT_FP32,
):
    tile_type = pl.TileType(shape=[1, 256], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    src = pl.make_tile(tile_type, addr=0x0000, size=1024)
    dst = pl.make_tile(tile_type, addr=0x0400, size=1024)
    with pl.section_vector():
        pl.load(src, x, [0, 0])
        pl.load(dst, x, [0, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.simt.launch(_tile_add, threads=256, args=(dst, src, n, delta))
        pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.store(out, dst, [0, 0])


@pl.kernel
def _simt_gm_codegen_kernel(
    x: pl.Tensor[[1, 256], pl.DT_FP32],
    out: pl.Tensor[[1, 256], pl.DT_FP32],
    n: pl.DT_UINT32,
    delta: pl.DT_FP32,
):
    with pl.section_vector():
        pl.simt.launch(_gm_add, threads=256, args=(out, x, n, delta))


@pl.kernel
def _simt_context_codegen_kernel():
    tile_type = pl.TileType(shape=[1, 256], dtype=pl.DT_UINT32, target_memory=pl.MemorySpace.Vec)
    dst = pl.make_tile(tile_type, addr=0x0000, size=1024)
    with pl.section_vector():
        pl.simt.launch(_context_probe, threads=(8, 4, 8), args=(dst,))


@pl.kernel
def _simt_callee_codegen_kernel(src: pl.Tensor[[1, 32], pl.DT_INT32], delta: pl.DT_INT32):
    tile_type = pl.TileType(shape=[1, 32], dtype=pl.DT_INT32, target_memory=pl.MemorySpace.Vec)
    dst = pl.make_tile(tile_type, addr=0x0000, size=128)
    with pl.section_vector():
        pl.simt.launch(_callee_entry, threads=32, args=(dst, src, delta))


@pl.kernel
def _simt_valid_shape_codegen_kernel(valid_rows: pl.DT_UINT32, valid_cols: pl.DT_UINT32):
    tile_type = pl.TileType(
        shape=[8, 64],
        dtype=pl.DT_FP32,
        target_memory=pl.MemorySpace.Vec,
        valid_shape=[-1, -1],
    )
    dst = pl.make_tile(tile_type, addr=0x0000, size=2048)
    src = pl.make_tile(tile_type, addr=0x0800, size=2048)
    with pl.section_vector():
        pl.set_validshape(dst, [valid_rows, valid_cols])
        pl.set_validshape(src, [valid_rows, valid_cols])
        pl.simt.launch(_tile_valid_shape_access, threads=256, args=(dst, src))


def _simt_function_source(cpp: str, name: str) -> str:
    start = cpp.index(f"inline void {name}(")
    end = cpp.index("\n}\n", start)
    return cpp[start:end]


def test_simt_tile_codegen_emits_direct_cce_function_launch_and_ub_accesses():
    cpp = _compile_to_cce(_simt_tile_codegen_kernel)

    signature = next(line for line in cpp.splitlines() if "inline void _tile_add(" in line)
    function_pos = cpp.index(signature)
    kernel_pos = cpp.index("__aicore__ inline void _simt_tile_codegen_kernel_impl_vector")
    launch_line = next(line for line in cpp.splitlines() if "cce::async_invoke<_tile_add>" in line)
    function = cpp[function_pos:kernel_pos]

    assert function_pos < kernel_pos
    assert "__simt_vf__ __launch_bounds__(256)" in signature
    for tile in ("dst", "src"):
        assert re.search(rf"__ubuf__\s+float\s*\*\s*{tile}(?:_\d+)*\b", signature)
        for axis in ("row", "col"):
            assert re.search(rf"\buint32_t\s+{tile}(?:_\d+)*__valid_{axis}\b", signature)
    assert "cce::async_invoke<_tile_add>(cce::dim3{256, 1, 1}" in launch_line
    assert launch_line.count("(__ubuf__ float*)") == 2
    assert launch_line.count(".data()") == 2
    assert launch_line.count(".GetValidRow()") == 2
    assert launch_line.count(".GetValidCol()") == 2
    assert "pipe_barrier(PIPE_ALL);" not in cpp
    assert re.search(r"\bdst(?:_\d+)?\[", function)
    assert re.search(r"\bsrc(?:_\d+)?\[", function)
    assert ".GetValue(" not in function
    assert ".SetValue(" not in function
    assert "PTO_INLINE" not in cpp
    assert "asc_vf_call" not in cpp


def test_simt_context_codegen_uses_native_cce_xyz_context_and_tuple_launch():
    cpp = _compile_to_cce(_simt_context_codegen_kernel)
    launch_line = next(line for line in cpp.splitlines() if "cce::async_invoke<_context_probe>" in line)

    for context in ("threadIdx", "blockDim", "blockIdx", "gridDim"):
        for axis in ("x", "y", "z"):
            assert f"{context}.{axis}" in cpp
    assert "warpSize" in cpp
    assert "threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y" in cpp
    assert "cce::async_invoke<_context_probe>(cce::dim3{8, 4, 8}" in launch_line


def test_simt_gm_tensor_codegen_uses_native_gm_pointers():
    cpp = _compile_to_cce(_simt_gm_codegen_kernel)
    function = _simt_function_source(cpp, "_gm_add")
    launch_line = next(line for line in cpp.splitlines() if "cce::async_invoke<_gm_add>" in line)

    assert re.search(r"__gm__\s+float\s*\*\s*dst", function)
    assert re.search(r"__gm__\s+float\s*\*\s*src", function)
    assert function.count("(__gm__ float*)") >= 2
    assert ".GetValue(" not in function
    assert ".SetValue(" not in function
    assert launch_line.count("(__gm__ float*)") == 2
    assert ".data()" not in launch_line


def test_simt_launch_passes_runtime_tile_valid_shape_to_native_function():
    cpp = _compile_to_cce(_simt_valid_shape_codegen_kernel)
    function = _simt_function_source(cpp, "_tile_valid_shape_access")
    launch_line = next(line for line in cpp.splitlines() if "cce::async_invoke<_tile_valid_shape_access>" in line)

    assert re.search(r"\bsrc(?:_\d+)*__valid_row\b", function)
    assert re.search(r"\bsrc(?:_\d+)*__valid_col\b", function)
    assert launch_line.count(".GetValidRow()") == 2
    assert launch_line.count(".GetValidCol()") == 2


def test_simt_callee_codegen_emits_native_nested_calls_and_tile_abi():
    cpp = _compile_to_cce(_simt_callee_codegen_kernel)

    add_signature = next(line for line in cpp.splitlines() if "_callee_add(" in line and "__simt_callee__" in line)
    load_signature = next(line for line in cpp.splitlines() if "_callee_load(" in line and "__simt_callee__" in line)
    store_signature = next(line for line in cpp.splitlines() if "_callee_store(" in line and "__simt_callee__" in line)
    apply_signature = next(line for line in cpp.splitlines() if "_callee_apply(" in line and "__simt_callee__" in line)
    entry_signature = next(line for line in cpp.splitlines() if "inline void _callee_entry(" in line)

    assert "__simt_callee__ inline int32_t" in add_signature
    assert "__simt_callee__ inline int32_t" in load_signature
    assert "__simt_callee__ inline void" in store_signature
    for signature in (store_signature, apply_signature):
        assert re.search(r"__ubuf__\s+int32_t\s*\*\s*dst(?:_\d+)*\b", signature)
        assert re.search(r"\bdst(?:_\d+)*__valid_row\b", signature)
        assert re.search(r"\bdst(?:_\d+)*__valid_col\b", signature)
    for signature in (load_signature, apply_signature):
        assert re.search(r"__gm__\s+int32_t\s*\*\s*src(?:_\d+)*\b", signature)
        assert not re.search(r"\bsrc(?:_\d+)*__valid_row\b", signature)
        assert not re.search(r"\bsrc(?:_\d+)*__valid_col\b", signature)

    add_pos = cpp.index(add_signature)
    load_pos = cpp.index(load_signature)
    store_pos = cpp.index(store_signature)
    apply_pos = cpp.index(apply_signature)
    entry_pos = cpp.index(entry_signature)
    assert max(add_pos, load_pos, store_pos) < apply_pos < entry_pos
    add_function = cpp[add_pos:cpp.index("\n}\n", add_pos)]
    apply_function = cpp[apply_pos:entry_pos]
    assert re.search(r"\bvalue(?:_\d+)*\s*\+\s*delta(?:_\d+)*\b", add_function)
    assert re.search(r"\breturn\s+[^;]+;", add_function)
    assert re.search(r"_callee_load\(src(?:_\d+)?,\s*index(?:_\d+)?\)", apply_function)
    assert re.search(r"_callee_add\([^;]+\)", apply_function)
    assert re.search(
        r"_callee_store\([^;]+dst(?:_\d+)?__valid_row,\s*dst(?:_\d+)?__valid_col[^;]+\);",
        apply_function,
    )
    assert re.search(
        r"_callee_apply\([^;]+dst(?:_\d+)?__valid_row,\s*dst(?:_\d+)?__valid_col"
        r"[^;]+src(?:_\d+)?[^;]+\);",
        cpp[entry_pos:],
    )


@pl.simt.function(max_threads=256)
def _simt_inplace_add(data: pl.Tile[[1, 256], pl.DT_FP32], delta: pl.DT_FP32):
    tid = pl.simt.linear_thread_idx()
    data[0, tid] = data[0, tid] + delta


@pl.kernel
def _simt_auto_mutex_kernel(
    x: pl.Tensor[[1, 256], pl.DT_FP32],
    out: pl.Tensor[[1, 256], pl.DT_FP32],
    delta: pl.DT_FP32,
):
    tile_type = pl.TileType(shape=[1, 256], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    data = pl.make_tile_group(type=tile_type, addrs=0x0000, mutex_ids=[0])
    with pl.section_vector():
        pl.load(data.current(), x, [0, 0])
        pl.simt.launch(_simt_inplace_add, threads=256, args=(data.current(), delta))
        pl.store(out, data.current(), [0, 0])


def test_simt_auto_mutex_emits_pipe_v_lock_unlock_around_launch():
    cpp = _compile_to_cce(_simt_auto_mutex_kernel)

    launch_line = next(line for line in cpp.splitlines() if "cce::async_invoke<_simt_inplace_add>" in line)
    launch_pos = cpp.index(launch_line)

    lock_match = re.search(r"get_buf\(PIPE_V,\s*\w+,\s*0\);", cpp[:launch_pos])
    assert lock_match is not None, "Expected get_buf(PIPE_V, ...) before cce::async_invoke"

    unlock_match = re.search(r"rls_buf\(PIPE_V,\s*\w+,\s*0\);", cpp[launch_pos + len(launch_line):])
    assert unlock_match is not None, "Expected rls_buf(PIPE_V, ...) after cce::async_invoke"

    assert "pipe_barrier(PIPE_ALL);" not in cpp


def test_simt_codegen_rejects_pre_a5_architecture():
    with pytest.raises(RuntimeError, match="requires arch='a5'"):
        _compile_to_cce(_simt_tile_codegen_kernel, arch="a3")
