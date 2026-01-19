#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
Tiling Operation Examples for PyPTO

This file contains all tiling operation examples merged into a single file.
You can run all examples or select specific ones using command-line arguments.

Usage:
    python tiling_ops.py                          # Run all examples
    python tiling_ops.py --list                   # List all available examples
    python tiling_ops.py cube_tile::test_set_cube_tile_shapes_basic    # Run a specific case
"""

import argparse
import os
import sys
import pypto
import torch
import numpy as np
from numpy.testing import assert_allclose
import time


def get_device_id():
    """
    Get and validate TILE_FWK_DEVICE_ID from environment variable.
    
    Returns:
        int: The device ID if valid, None otherwise.
    """
    if 'TILE_FWK_DEVICE_ID' not in os.environ:
        print("If no NPU environment is available, set --run_mode sim to run in simulation mode;")
        print("otherwise, set the environment variable TILE_FWK_DEVICE_ID.")
        print("Please set it before running this example:")
        print("  export TILE_FWK_DEVICE_ID=0")
        return None
    
    try:
        device_id = int(os.environ['TILE_FWK_DEVICE_ID'])
        return device_id
    except ValueError:
        print(f"ERROR: TILE_FWK_DEVICE_ID must be an integer, got: {os.environ['TILE_FWK_DEVICE_ID']}")
        return None


# ============================================================================
# Cube Tile Examples
# ============================================================================
    
def create_cube_tile_kernel(m, k, n, run_mode, set_shapes: list):

    @pypto.frontend.jit(runtime_options={"run_mode": run_mode})
    def compute_with_cube_tile_shapes(
        a: pypto.Tensor((m, k), pypto.DT_FP32),
        b: pypto.Tensor((k, n), pypto.DT_FP32),
    ) -> pypto.Tensor((m, n), pypto.DT_FP32):
        pypto.set_cube_tile_shapes(*set_shapes)
        out = pypto.matmul(a, b, a.dtype)
        return out

    return compute_with_cube_tile_shapes


def compute_with_cube_tile_shapes_op(
    a: torch.Tensor, b: torch.Tensor, set_shapes: list, run_mode: str = "npu"
):
    if run_mode == "npu":
        mode = pypto.RunMode.NPU
    elif run_mode == "sim":
        mode = pypto.RunMode.SIM
    else:
        raise ValueError(f"Invalid run_mode: {run_mode}. Must be 'npu' or 'sim'")
    return create_cube_tile_kernel(a.shape[0], a.shape[1], b.shape[1], mode, set_shapes)(a, b)


def test_set_cube_tile_shapes_basic(device_id: int = None, run_mode: str = "npu"):
    """Test basic usage of set_cube_tile_shapes function"""
    print("=" * 60)
    print("Test: Basic Usage of set_cube_tile_shapes Function")
    print("=" * 60)
    
    device = f'npu:{device_id}' if (run_mode == "npu" and device_id is not None) else 'cpu'
    
    # Test 1: Set and verify tile shapes for cube computation
    dtype = torch.float32
    # shape: (2, 2)
    a = torch.tensor([[1, 2], [3, 4]], dtype=dtype, device=device)
    b = torch.tensor([[5, 6], [7, 8]], dtype=dtype, device=device)
    expected = torch.tensor([[19, 22], [43, 50]], dtype=dtype, device=device)
    set_shapes = [[32, 32], [64, 64], [64, 64]]

    out = compute_with_cube_tile_shapes_op(a, b, set_shapes, run_mode)
    print(f"Output: {out}")
    print(f"Expected: {expected}")
    get_shapes = pypto.get_cube_tile_shapes()
    print(f"set_shapes == get_shapes: {set_shapes == get_shapes}")
    if run_mode == "npu":
        assert_allclose(out.cpu().numpy(), expected.cpu().numpy(), rtol=1e-3, atol=1e-3)

    # Test 2: Set cube_tile_shapes for cube calculations with different shapes
    dtype = torch.float32
    # shape: (4, 6)
    a = torch.randn((4, 6), dtype=dtype, device=device)
    b = torch.randn((6, 4), dtype=dtype, device=device)
    expected = torch.matmul(a, b)

    out = compute_with_cube_tile_shapes_op(a, b, [[32, 32], [64, 64], [64, 64]], run_mode)
    print(f"Output: {out}")
    print(f"Expected: {expected}")
    if run_mode == "npu":
        assert_allclose(out.cpu().numpy(), expected.cpu().numpy(), rtol=1e-3, atol=1e-3)
    print("✓ Basic usage of set_cube_tile_shapes function completed successfully")


def create_different_tile_shapes_kernel(run_mode):
    b, m_batch, k_batch, n_batch = 2, 2, 2, 2

    @pypto.frontend.jit(runtime_options={"run_mode": run_mode})
    def compute_with_different_tile_shapes(
        x: pypto.Tensor((b, m_batch, k_batch), pypto.DT_FP32),
        y: pypto.Tensor((b, k_batch, n_batch), pypto.DT_FP32),
    ) -> (
        pypto.Tensor((b, m_batch, n_batch), pypto.DT_FP32),
        pypto.Tensor((b, m_batch, n_batch), pypto.DT_FP32),
        pypto.Tensor((b, m_batch, n_batch), pypto.DT_FP32),
    ):
        """Compute matmul with three different tile shapes and return all results"""
        print(f"b: {b}, m_batch: {m_batch}, k_batch: {k_batch}, n_batch: {n_batch}")
        out1 = pypto.tensor((b, m_batch, n_batch), pypto.DT_FP32)
        out2 = pypto.tensor((b, m_batch, n_batch), pypto.DT_FP32)
        out3 = pypto.tensor((b, m_batch, n_batch), pypto.DT_FP32)

        pypto.set_cube_tile_shapes([32, 32], [16, 16], [32, 32])
        print(f"pypto.get_cube_tile_shapes(): {pypto.get_cube_tile_shapes()}")
        out1[:] = pypto.matmul(x, y, x.dtype)

        pypto.set_cube_tile_shapes([32, 32], [16, 64], [32, 128])
        print(f"pypto.get_cube_tile_shapes(): {pypto.get_cube_tile_shapes()}")
        out2[:] = pypto.matmul(x, y, x.dtype)

        pypto.set_cube_tile_shapes([64, 64], [128, 128], [128, 128])
        print(f"pypto.get_cube_tile_shapes(): {pypto.get_cube_tile_shapes()}")
        out3[:] = pypto.matmul(x, y, x.dtype)

        return out1, out2, out3

    return compute_with_different_tile_shapes


def compute_with_different_tile_shapes_op(a: torch.Tensor, b: torch.Tensor, run_mode: str = "npu", dynamic: bool = False) -> tuple:
    
    if run_mode == "npu":
        mode = pypto.RunMode.NPU
    elif run_mode == "sim":
        mode = pypto.RunMode.SIM
    else:
        raise ValueError(f"Invalid run_mode: {run_mode}. Must be 'npu' or 'sim'")
    return create_different_tile_shapes_kernel(mode)(a, b)



def test_set_different_tile_shapes_result(device_id: int = None, run_mode: str = "npu"):
    """Test the impact of different tile shape settings on calculation results"""
    print("=" * 60)
    print("Test: Impact of Different Tile Shape Settings on Calculation Results")
    print("=" * 60)
    
    device = f'npu:{device_id}' if (run_mode == "npu" and device_id is not None) else 'cpu'
    
    # Test 1: Different Tile Shape Settings on Calculation Results
    dtype = torch.float32
    # shape:(2, 2, 2)
    a = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=dtype, device=device)
    b = torch.tensor([[[5, 6], [7, 8]], [[1, 2], [3, 4]]], dtype=dtype, device=device)
    expected = torch.tensor([[[19, 22], [43, 50]], [[23, 34], [31, 46]]], dtype=dtype, device=device)

    out1, out2, out3 = compute_with_different_tile_shapes_op(a, b, run_mode)
    print(f"out1 == out2: {torch.equal(out1, out2)}")
    print(f"out2 == out3: {torch.equal(out2, out3)}")
    if run_mode == "npu":
        assert_allclose(out1.cpu().numpy(), expected.cpu().numpy(), rtol=1e-3, atol=1e-3)
        assert_allclose(out2.cpu().numpy(), expected.cpu().numpy(), rtol=1e-3, atol=1e-3)
        assert_allclose(out3.cpu().numpy(), expected.cpu().numpy(), rtol=1e-3, atol=1e-3)
    print("✓ Impact of different tile shape settings on results completed successfully")


def create_tile_32_kernel(run_mode):
    b_rt, m_rt, k_rt, n_rt = 4, 64, 512, 128

    @pypto.frontend.jit(runtime_options={"run_mode": run_mode})
    def compute_with_tile_32(
        a: pypto.Tensor((b_rt, m_rt, k_rt), pypto.DT_FP32),
        b: pypto.Tensor((b_rt, n_rt, k_rt), pypto.DT_FP32),
    ) -> pypto.Tensor((b_rt, m_rt, n_rt), pypto.DT_FP32):
        print(f"b_rt: {b_rt}, m_rt: {m_rt}, k_rt: {k_rt}, n_rt: {n_rt}")
        pypto.set_cube_tile_shapes([32, 32], [32, 32], [32, 32])
        output = pypto.matmul(a, b, a.dtype, b_trans=True)
        return output

    return compute_with_tile_32


def create_tile_64_kernel(run_mode):
    b_rt, m_rt, k_rt, n_rt = 4, 64, 512, 128

    @pypto.frontend.jit(runtime_options={"run_mode": run_mode})
    def compute_with_tile_64(
        a: pypto.Tensor((b_rt, m_rt, k_rt), pypto.DT_FP32),
        b: pypto.Tensor((b_rt, n_rt, k_rt), pypto.DT_FP32),
    ) -> pypto.Tensor((b_rt, m_rt, n_rt), pypto.DT_FP32):
        print(f"b_rt: {b_rt}, m_rt: {m_rt}, k_rt: {k_rt}, n_rt: {n_rt}")
        pypto.set_cube_tile_shapes([64, 64], [128, 128], [128, 128])
        output = pypto.matmul(a, b, a.dtype, b_trans=True)
        return output

    return compute_with_tile_64


def test_set_different_tile_shapes_runtime(device_id: int = None, run_mode: str = "npu"):
    """Test the impact of different tile shape settings on runtime"""
    print("=" * 60)
    print("Test: Impact of Different Tile Shape Settings on Runtime")
    print("=" * 60)
    
    device = f'npu:{device_id}' if (run_mode == "npu" and device_id is not None) else 'cpu'
    
    if run_mode == "npu":
        import torch_npu

        device = f"npu:{device_id}"
        mode = pypto.RunMode.NPU
    else:
        device = "cpu"
        mode = pypto.RunMode.SIM
        
    # Test 1: Different Tile Shape Settings on Runtime
    b_rt, m_rt, k_rt, n_rt = 4, 64, 512, 128
    dtype = torch.float32
    a = torch.randn((b_rt, m_rt, k_rt), dtype=dtype, device=device)
    b = torch.randn((b_rt, n_rt, k_rt), dtype=dtype, device=device)

    compute_tile_32 = create_tile_32_kernel(mode)
    compute_tile_64 = create_tile_64_kernel(mode)

    TEST_TIME = 1
    start = time.perf_counter()
    for _ in range(TEST_TIME):
        out1 = compute_tile_32(a, b)
    runtime_1 = time.perf_counter() - start
    start = time.perf_counter()
    for _ in range(TEST_TIME):
        out2 = compute_tile_64(a, b)
    runtime_2 = time.perf_counter() - start
    print(f"runtime_1(pypto.set_cube_tile_shapes([32, 32], [32, 32], [32, 32])): {runtime_1}")
    print(f"runtime_2(pypto.set_cube_tile_shapes([64, 64], [128, 128], [128, 128])): {runtime_2}")
    print("✓ Impact of different tile shape settings on runtime completed successfully")


# ============================================================================
# Vector Tile Examples
# ============================================================================
    
def create_vec_tile_kernel(shape, run_mode, set_shapes: tuple):
    """Factory function to create vector tile kernel with specific shapes"""

    @pypto.frontend.jit(runtime_options={"run_mode": run_mode})
    def compute_with_vec_tile_shapes(
        a: pypto.Tensor(shape, pypto.DT_FP32),
        b: pypto.Tensor(shape, pypto.DT_FP32),
    ) -> pypto.Tensor(shape, pypto.DT_FP32):
        print(f"shape: {shape}")
        pypto.set_vec_tile_shapes(*set_shapes)
        output = pypto.add(a, b)
        return output

    return compute_with_vec_tile_shapes


def compute_with_vec_tile_shapes_op(a: torch.Tensor, b: torch.Tensor, set_shapes: tuple, run_mode:str = "npu", dynamic: bool = False) -> torch.Tensor:
    if run_mode == "npu":
        mode = pypto.RunMode.NPU
    elif run_mode == "sim":
        mode = pypto.RunMode.SIM
    else:
        raise ValueError(f"Invalid run_mode: {run_mode}. Must be 'npu' or 'sim'")
    return create_vec_tile_kernel(tuple(a.shape), mode, set_shapes)(a, b)


def test_set_vec_tile_shapes_basic(device_id: int = None, run_mode: str = "npu"):
    """Test basic usage of set_vec_tile_shapes function"""
    print("=" * 60)
    print("Test: Basic Usage of set_vec_tile_shapes Function")
    print("=" * 60)
    
    device = f'npu:{device_id}' if (run_mode == "npu" and device_id is not None) else 'cpu'
    
    # Test 1: Set and verify tile shapes for vector computation
    dtype = torch.float32
    # shape: (1, 2, 3)
    a = torch.tensor([[[1, 2, 3],
                       [1, 2, 3]]], dtype=dtype, device=device)
    b = torch.tensor([[[4, 5, 6],
                       [4, 5, 6]]], dtype=dtype, device=device)
    expected = torch.tensor([[[5, 7, 9],
                            [5, 7, 9]]], dtype=dtype, device=device)
    set_shapes = (1, 2, 8)
    print(f"Rule1: len(set_vec_tile_shapes) == len(vec.shape): \
          {len(set_shapes) == len(a.shape)}")
    print("Rule2: valid input dims must in [1, 4]")

    out = compute_with_vec_tile_shapes_op(a, b, set_shapes, run_mode)
    print(f"Output: {out}")
    print(f"Expected: {expected}")
    get_shapes = pypto.get_vec_tile_shapes()
    if run_mode == "npu":
        assert_allclose(out.cpu().numpy(), expected.cpu().numpy(), rtol=1e-3, atol=1e-3)
    print(f"set_shapes == get_shapes: {set_shapes == get_shapes}")
    
    # Test 2: Set vec_tile_shapes for vector calculations with different shapes
    dtype = torch.float32
    # shape: (1, 1, 2, 3)
    a = torch.tensor([[[[1, 2, 3],
                        [4, 5, 6]]]], dtype=dtype, device=device)
    b = torch.tensor([[[[7, 8, 9],
                        [10, 11, 12]]]], dtype=dtype, device=device)
    expected = torch.tensor([[[[8, 10, 12],
                            [14, 16, 18]]]], dtype=dtype, device=device)

    out = compute_with_vec_tile_shapes_op(a, b, (1, 1, 4, 8), run_mode)
    print(f"Output: {out}")
    print(f"Expected: {expected}")
    if run_mode == "npu":
        assert_allclose(out.cpu().numpy(), expected.cpu().numpy(), rtol=1e-3, atol=1e-3)
    print("✓ Basic usage of set_vec_tile_shapes function completed successfully")


def create_vec_different_tile_shapes_kernel(shape, run_mode):
    @pypto.frontend.jit(runtime_options={"run_mode": run_mode})
    def compute_with_vec_different_tile_shapes(
        a: pypto.Tensor(shape, pypto.DT_FP32),
        b: pypto.Tensor(shape, pypto.DT_FP32),
    ) -> (
        pypto.Tensor(shape, pypto.DT_FP32),
        pypto.Tensor(shape, pypto.DT_FP32),
        pypto.Tensor(shape, pypto.DT_FP32),
    ):
        """Compute add with three different vec tile shapes and return all results"""
        print(f"shape: {shape}")
        pypto.set_vec_tile_shapes(1, 2, 8)
        print(f"pypto.get_vec_tile_shapes(): {pypto.get_vec_tile_shapes()}")
        out1 = pypto.add(a, b)

        pypto.set_vec_tile_shapes(2, 6, 32)
        print(f"pypto.get_vec_tile_shapes(): {pypto.get_vec_tile_shapes()}")
        out2 = pypto.add(a, b)

        pypto.set_vec_tile_shapes(5, 3, 16)
        print(f"pypto.get_vec_tile_shapes(): {pypto.get_vec_tile_shapes()}")
        out3 = pypto.add(a, b)

        return out1, out2, out3

    return compute_with_vec_different_tile_shapes



def compute_with_vec_different_tile_shapes_op(a: torch.Tensor, b: torch.Tensor, run_mode:str = "npu", dynamic: bool = False) -> tuple:

    if run_mode == "npu":
        mode = pypto.RunMode.NPU
    elif run_mode == "sim":
        mode = pypto.RunMode.SIM
    else:
        raise ValueError(f"Invalid run_mode: {run_mode}. Must be 'npu' or 'sim'")
    return create_vec_different_tile_shapes_kernel(tuple(a.shape), mode)(a, b)


def test_set_vec_different_tile_shapes_result(device_id: int = None, run_mode: str = "npu"):
    """Test the impact of different tile shape settings on calculation results"""
    print("=" * 60)
    print("Test: Impact of Different Tile Shape Settings on Calculation Results")
    print("=" * 60)
    
    device = f'npu:{device_id}' if (run_mode == "npu" and device_id is not None) else 'cpu'
    
    # Test 1: Different Tile Shape Settings on Calculation Results
    dtype = torch.float32
    # shape: (1, 2, 3)
    a = torch.tensor([[[1, 2, 3],
                       [1, 2, 3]]], dtype=dtype, device=device)
    b = torch.tensor([[[4, 5, 6],
                       [4, 5, 6]]], dtype=dtype, device=device)
    expected = torch.tensor([[[5, 7, 9],
                            [5, 7, 9]]], dtype=dtype, device=device)

    out1, out2, out3 = compute_with_vec_different_tile_shapes_op(a, b, run_mode)
    print(f"out1 == out2: {torch.equal(out1, out2)}")
    print(f"out2 == out3: {torch.equal(out2, out3)}")
    if run_mode == "npu":
        assert_allclose(out1.cpu().numpy(), expected.cpu().numpy(), rtol=1e-3, atol=1e-3)
        assert_allclose(out2.cpu().numpy(), expected.cpu().numpy(), rtol=1e-3, atol=1e-3)
        assert_allclose(out3.cpu().numpy(), expected.cpu().numpy(), rtol=1e-3, atol=1e-3)
    print("✓ Impact of different tile shape settings on results completed successfully")


def create_vec_tile_small_kernel(run_mode):
    vec_rt_shape = (4, 32, 64, 256)

    @pypto.frontend.jit(runtime_options={"run_mode": run_mode})
    def compute_with_vec_tile_small(
        a: pypto.Tensor(vec_rt_shape, pypto.DT_FP32),
        b: pypto.Tensor(vec_rt_shape, pypto.DT_FP32),
    ) -> pypto.Tensor(vec_rt_shape, pypto.DT_FP32):
        print(f"vec_rt_shape: {vec_rt_shape}")
        pypto.set_vec_tile_shapes(1, 2, 4, 128)
        output = pypto.add(a, b)
        return output

    return compute_with_vec_tile_small


def create_vec_tile_large_kernel(run_mode):
    vec_rt_shape = (4, 32, 64, 256)

    @pypto.frontend.jit(runtime_options={"run_mode": run_mode})
    def compute_with_vec_tile_large(
        a: pypto.Tensor(vec_rt_shape, pypto.DT_FP32),
        b: pypto.Tensor(vec_rt_shape, pypto.DT_FP32),
    ) -> pypto.Tensor(vec_rt_shape, pypto.DT_FP32):
        print(f"vec_rt_shape: {vec_rt_shape}")
        pypto.set_vec_tile_shapes(2, 4, 8, 256)
        output = pypto.add(a, b)
        return output

    return compute_with_vec_tile_large


def test_set_vec_different_tile_shapes_runtime(device_id: int = None, run_mode: str = "npu"):
    """Test the impact of different tile shape settings on runtime"""
    print("=" * 60)
    print("Test: Impact of Different Tile Shape Settings on Runtime")
    print("=" * 60)
    
    device = f'npu:{device_id}' if (run_mode == "npu" and device_id is not None) else 'cpu'
    
    if run_mode == "npu":
        import torch_npu

        device = f"npu:{device_id}"
        mode = pypto.RunMode.NPU
    else:
        device = "cpu"
        mode = pypto.RunMode.SIM
        
    # Test 1: Different Tile Shape Settings on Runtime
    vec_rt_shape = (4, 32, 64, 256)
    dtype = torch.float32
    a = torch.randn(vec_rt_shape, dtype=dtype, device=device)
    b = torch.randn(vec_rt_shape, dtype=dtype, device=device)

    compute_vec_small = create_vec_tile_small_kernel(mode)
    compute_vec_large = create_vec_tile_large_kernel(mode)


    TEST_TIME = 1
    start = time.perf_counter()
    for _ in range(TEST_TIME):
        out1 = compute_vec_small(a, b)
    runtime_1 = time.perf_counter() - start
    start = time.perf_counter()
    for _ in range(TEST_TIME):
        out2 = compute_vec_large(a, b)
    runtime_2 = time.perf_counter() - start
    print(f"runtime_1(pypto.set_vec_tile_shapes(1, 2, 4, 128)): {runtime_1}")
    print(f"runtime_2(pypto.set_vec_tile_shapes(2, 4, 8, 256)): {runtime_2}")
    print("✓ Impact of different tile shape settings on runtime completed successfully")


def main():
    """Run tiling examples.
    
    Usage:
        python tiling_ops.py                          # Run all examples
        python tiling_ops.py --list                   # List all available examples
        python tiling_ops.py cube_tile::test_set_cube_tile_shapes_basic    # Run a specific case
    """
    parser = argparse.ArgumentParser(
        description="PyPTO Tiling Operation Examples",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                      Run all examples
  %(prog)s --list               List all available examples
  %(prog)s cube_tile::test_set_cube_tile_shapes_basic    Run a specific case
        """
    )
    parser.add_argument(
        'example_id',
        type=str,
        nargs="?",
        help='Run a specific case (e.g., cube_tile::test_set_cube_tile_shapes_basic). If omitted, all cases run.'
    )
    parser.add_argument(
        '--list',
        action='store_true',
        help='List all available examples and exit'
    )
    parser.add_argument(
        '--run_mode',
        type=str,
        nargs='?',
        default='npu',
        choices=["npu", "sim"],
        help='Run mode, such as npu/sim etc.'
    )
    
    args = parser.parse_args()
    
    # Define available examples
    examples = {
        'cube_tile::test_set_cube_tile_shapes_basic': {
            'name': 'Test basic usage of set_cube_tile_shapes function',
            'description': 'Basic usage of set_cube_tile_shapes function example',
            'function': test_set_cube_tile_shapes_basic,
        },
        'cube_tile::test_set_different_tile_shapes_result': {
            'name': 'Test the impact of different tile shape settings on calculation results',
            'description': 'Impact of different tile shape settings on calculation results example',
            'function': test_set_different_tile_shapes_result,
        },
        'cube_tile::test_set_different_tile_shapes_runtime': {
            'name': 'Test the impact of different tile shape settings on runtime',
            'description': 'Impact of different tile shape settings on runtime example',
            'function': test_set_different_tile_shapes_runtime,
        },
        'vec_tile::test_set_vec_tile_shapes_basic': {
            'name': 'Test basic usage of set_vec_tile_shapes function',
            'description': 'Basic usage of set_vec_tile_shapes function example',
            'function': test_set_vec_tile_shapes_basic,
        },
        'vec_tile::test_set_vec_different_tile_shapes_result': {
            'name': 'Test the impact of different tile shape settings on calculation results',
            'description': 'Impact of different tile shape settings on calculation results example',
            'function': test_set_vec_different_tile_shapes_result,
        },
        'vec_tile::test_set_vec_different_tile_shapes_runtime': {
            'name': 'Test the impact of different tile shape settings on runtime',
            'description': 'Impact of different tile shape settings on runtime example',
            'function': test_set_vec_different_tile_shapes_runtime,
        },
    }
    
    # List examples if requested
    if args.list:
        print("\n" + "=" * 60)
        print("Available Examples")
        print("=" * 60 + "\n")
        for ex_id, ex_info in sorted(examples.items()):
            print(f"  ID: {ex_id}")
            print(f"     name: {ex_info['name']}")
            print(f"     description: {ex_info['description']}\n")
        return
    
    # Validate case if provided
    examples_to_run = []
    device_id = None
    if args.example_id:
        if args.example_id not in examples:
            print(f"ERROR: Invalid case: {args.example_id}")
            print(f"Valid cases are: {', '.join(sorted(examples.keys()))}")
            print("\nUse --list to see all available examples.")
            sys.exit(1)
        examples_to_run = [(args.example_id, examples[args.example_id])]
    else:
        examples_to_run = [(key, info) for key, info in sorted(examples.items())]
    
    print("\n" + "=" * 60)
    print("PyPTO Tiling Operation Examples")
    print("=" * 60 + "\n")
    
    if args.run_mode == "npu":
        device_id = get_device_id()
        if device_id is None:
            return
        import torch_npu
        torch.npu.set_device(device_id)
        print("Running examples that require NPU hardware...")
        print("(Make sure CANN environment is configured and NPU is available)\n")
    
    try:
        for ex_id, ex_info in examples_to_run:
            print(f"Running Example {ex_id}: {ex_info['name']}")
            ex_info['function'](device_id, args.run_mode)
        
        if len(examples_to_run) > 1:
            print("=" * 60)
            print("All tiling tests passed!")
            print("=" * 60)
        
    except Exception as e:
        print(f"\nError: {e}")
        raise


if __name__ == "__main__":
    main()

