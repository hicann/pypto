#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Test use_cache parameter of @jit decorator."""

import os
import time
import logging
import pypto
import torch

logging.basicConfig(level=logging.INFO, format='', force=True)


def create_compute_func_with_cache(shape, tiling):
    """Factory function that creates a new compute function with cache disabled."""
    @pypto.frontend.jit(use_cache=True)
    def compute_add(
        a: pypto.Tensor(shape, pypto.DT_FP32),
        b: pypto.Tensor(shape, pypto.DT_FP32),
    ) -> pypto.Tensor(shape, pypto.DT_FP32):
        pypto.set_cube_tile_shapes([tiling, tiling], [tiling, tiling], [tiling, tiling])
        c = pypto.matmul(a, b, a.dtype)
        return c
    return compute_add


def create_compute_func_without_cache(shape, tiling):
    """Factory function that creates a new compute function with cache disabled."""
    @pypto.frontend.jit(use_cache=False)
    def compute_add(
        a: pypto.Tensor(shape, pypto.DT_FP32),
        b: pypto.Tensor(shape, pypto.DT_FP32),
    ) -> pypto.Tensor(shape, pypto.DT_FP32):
        pypto.set_cube_tile_shapes([tiling, tiling], [tiling, tiling], [tiling, tiling])
        c = pypto.matmul(a, b, a.dtype)
        return c
    return compute_add


def test_use_cache_false_compiles_twice():
    """Test that use_cache=False causes recompilation each time.""" 
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)
    tiling = 32

    logging.info("\n=== Testing use_cache=False (should compile twice) ===")

    shape = (tiling * 16, tiling * 16)
    a = torch.rand(shape, dtype=torch.float32, device=f'npu:{device_id}')
    b = torch.rand(shape, dtype=torch.float32, device=f'npu:{device_id}')

    # First call - will compile
    start_time = time.perf_counter()
    kernel_1 = create_compute_func_without_cache(shape, tiling)
    c = kernel_1(a, b)
    first_call_time = time.perf_counter() - start_time
    logging.info(f"First call time: {first_call_time:.4f}s")

    # Second call - should compile again (cache disabled)
    start_time = time.perf_counter()
    kernel_2 = create_compute_func_without_cache(shape, tiling)
    d = kernel_2(a, b)
    second_call_time = time.perf_counter() - start_time
    logging.info(f"Second call time: {second_call_time:.4f}s")

    # Both calls should take similar time (both compile)
    # Allow some variance, but they should be in the same ballpark
    ratio = second_call_time / first_call_time
    logging.info(f"Time ratio (second/first): {ratio:.2f}")

    # Assert that second call is not significantly faster (compiled both times)
    # If it was cached, second call would be much faster
    assert ratio > 0.5, \
        f"Second call was too fast ({second_call_time:.4f}s vs {first_call_time:.4f}s), " \
        f"suggesting cache was used when it shouldn't be"

    logging.info("✓ Verified: Both calls compiled (use_cache=False working correctly)")


def test_use_cache_true_compiles_once():
    """Test that use_cache=True reuses cached compilation."""
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)
    tiling = 32

    logging.info("\n=== Testing use_cache=True (should compile only once) ===")

    shape = (tiling * 16, tiling * 16)
    a = torch.rand(shape, dtype=torch.float32, device=f'npu:{device_id}')
    b = torch.rand(shape, dtype=torch.float32, device=f'npu:{device_id}')

    # First call - will compile
    start_time = time.perf_counter()
    c = create_compute_func_with_cache(shape, tiling)(a, b)
    first_call_time = time.perf_counter() - start_time
    logging.info(f"First call time: {first_call_time:.4f}s")

    # Second call - should reuse cache (no compilation)
    start_time = time.perf_counter()
    d = create_compute_func_with_cache(shape, tiling)(a, b)
    second_call_time = time.perf_counter() - start_time
    logging.info(f"Second call time: {second_call_time:.4f}s")

    # Second call should be significantly faster (cache hit)
    ratio = second_call_time / first_call_time
    logging.info(f"Time ratio (second/first): {ratio:.2f}")

    # Assert that second call is significantly faster (cached)
    # With cache, the second call should be at least 2x faster
    assert ratio < 0.8, \
        f"Second call was not faster enough ({second_call_time:.4f}s vs {first_call_time:.4f}s), " \
        f"suggesting cache was not used when it should be"

    logging.info(f"✓ Verified: Second call reused cache (speedup: {1/ratio:.1f}x)")


if __name__ == "__main__":
    logging.info("=" * 70)
    logging.info("Testing @pypto.frontend.jit decorator use_cache parameter")
    logging.info("=" * 70)

    test_use_cache_false_compiles_twice()
    test_use_cache_true_compiles_once()

    logging.info("\n" + "=" * 70)
    logging.info("All tests passed!")
    logging.info("=" * 70)
