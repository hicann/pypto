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
Transform Operation Examples for PyPTO

This file contains all transform operation examples merged into a single file.
You can run all examples or select specific ones using command-line arguments.

Usage:
    python transform_ops.py                          # Run all examples
    python transform_ops.py --list                   # List all available examples
    python transform_ops.py assemble::test_assemble_basic    # Run a specific case
"""

import argparse
import os
import sys
import pypto
import torch
import numpy as np
from numpy.testing import assert_allclose


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
# Assemble Examples
# ============================================================================
    
@pypto.jit
def assemble_kernel_npu(x: pypto.Tensor, out: pypto.Tensor, offsets: list) -> None:
    tile_shapes = [8 for _ in range(len(x.shape))]
    pypto.set_vec_tile_shapes(*tile_shapes)
    pypto.assemble(x, offsets, out)

@pypto.jit(runtime_options={"run_mode": pypto.RunMode.SIM})
def assemble_kernel_sim(x: pypto.Tensor, out: pypto.Tensor, offsets: list) -> None:
    tile_shapes = [8 for _ in range(len(x.shape))]
    pypto.set_vec_tile_shapes(*tile_shapes)
    pypto.assemble(x, offsets, out)


def assemble_op(x: torch.Tensor, out_shape: tuple, offsets: list, dtype: torch.dtype, device: str, run_mode: str = "npu", dynamic: bool = False) -> torch.Tensor:
    out = torch.zeros(out_shape, dtype=dtype, device=device)

    if dynamic:
        x_pto = pypto.from_torch(x, dynamic_axis=[0])
        out_pto = pypto.from_torch(out, dynamic_axis=[0])
    else:
        x_pto = pypto.from_torch(x)
        out_pto = pypto.from_torch(out)

    if run_mode == "npu":
        assemble_kernel_npu(x_pto, out_pto, offsets)
    else:
        assemble_kernel_sim(x_pto, out_pto, offsets)
    return out


def test_assemble_basic(device_id: int = None, run_mode: str = "npu"):
    """Test basic usage of assemble function"""
    print("=" * 60)
    print("Test: Basic Usage of assemble Function")
    print("=" * 60)
    
    device = f'npu:{device_id}' if (run_mode == "npu" and device_id is not None) else 'cpu'
    
    # Test 1: Basic assembly of a small tensor into a larger tensor
    dtype = torch.float32
    x = torch.tensor([[1, 1], [1, 1]], dtype=dtype, device=device)
    offsets = [0, 0]
    expected = torch.tensor([[1, 1, 0, 0],
                             [1, 1, 0, 0],
                             [0, 0, 0, 0],
                             [0, 0, 0, 0]], dtype=dtype, device=device)

    out = assemble_op(x, (4, 4), offsets, dtype, device, run_mode)
    max_diff = np.abs(out.cpu().numpy() - expected.cpu().numpy()).max()
    print(f"Output: {out}")
    print(f"Expected: {expected}")
    print(f"Max difference: {max_diff:.6f}")
    if run_mode == "npu":
        assert_allclose(out.cpu().numpy(), expected.cpu().numpy(), rtol=1e-3, atol=1e-3)


def test_assemble_different_offsets_shapes(device_id: int = None, run_mode: str = "npu"):
    """Test basic usage of assemble function"""
    print("=" * 60)
    print("Test: Basic Usage of assemble Function")
    print("=" * 60)
    
    device = f'npu:{device_id}' if (run_mode == "npu" and device_id is not None) else 'cpu'
    # Test 2: Using different offsets
    dtype = torch.float32
    x = torch.tensor([[2, 2], [2, 2]], dtype=dtype, device=device)
    offsets = [1, 1]
    expected = torch.tensor([[0, 0, 0, 0],
                             [0, 2, 2, 0],
                             [0, 2, 2, 0],
                             [0, 0, 0, 0]], dtype=dtype, device=device)

    out = assemble_op(x, (4, 4), offsets, dtype, device, run_mode)
    max_diff = np.abs(out.cpu().numpy() - expected.cpu().numpy()).max()
    print(f"Output: {out}")
    print(f"Expected: {expected}")
    print(f"Max difference: {max_diff:.6f}")
    if run_mode == "npu":
        assert_allclose(out.cpu().numpy(), expected.cpu().numpy(), rtol=1e-3, atol=1e-3)
    
    # Test 3: Assembly with different shapes
    dtype = torch.float32
    x = torch.tensor([[1, 1, 1], [1, 1, 1]], dtype=dtype, device=device)
    offsets = [1, 1]
    expected = torch.tensor([[0, 0, 0, 0, 0],
                             [0, 1, 1, 1, 0],
                             [0, 1, 1, 1, 0],
                             [0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0]], dtype=dtype, device=device)

    out = assemble_op(x, (5, 5), offsets, dtype, device, run_mode)
    max_diff = np.abs(out.cpu().numpy() - expected.cpu().numpy()).max()
    print(f"Output: {out}")
    print(f"Expected: {expected}")
    print(f"Max difference: {max_diff:.6f}")
    if run_mode == "npu":
        assert_allclose(out.cpu().numpy(), expected.cpu().numpy(), rtol=1e-3, atol=1e-3)
    
    print("✓ Basic usage of assemble function completed successfully")

# ============================================================================
# Gather Examples
# ============================================================================
    
@pypto.jit(
    host_options={"only_codegen": True},
)
def gather_kernel_npu(input_tensor: pypto.Tensor, index_tensor: pypto.Tensor, out: pypto.Tensor, dim: int) -> None:
    tile_shapes = [8 for _ in range(len(input_tensor.shape))]
    pypto.set_vec_tile_shapes(*tile_shapes)
    out[:] = pypto.gather(input_tensor, dim, index_tensor)

@pypto.jit(
    host_options={"only_codegen": True},
    runtime_options={"run_mode": pypto.RunMode.SIM}
)
def gather_kernel_sim(input_tensor: pypto.Tensor, index_tensor: pypto.Tensor, out: pypto.Tensor, dim: int) -> None:
    tile_shapes = [8 for _ in range(len(input_tensor.shape))]
    pypto.set_vec_tile_shapes(*tile_shapes)
    out[:] = pypto.gather(input_tensor, dim, index_tensor)


def gather_op(input_tensor: torch.Tensor, index_tensor: torch.Tensor, dim: int, run_mode: str= "npu", dynamic: bool = False) -> torch.Tensor:
    out = torch.zeros(index_tensor.shape, dtype=input_tensor.dtype, device=input_tensor.device)

    if dynamic:
        input_pto = pypto.from_torch(input_tensor, dynamic_axis=[0])
        index_pto = pypto.from_torch(index_tensor, dynamic_axis=[0])
        out_pto = pypto.from_torch(out, dynamic_axis=[0])
    else:
        input_pto = pypto.from_torch(input_tensor)
        index_pto = pypto.from_torch(index_tensor)
        out_pto = pypto.from_torch(out)
    if run_mode == "npu":
        gather_kernel_npu(input_pto, index_pto, out_pto, dim)
    else:
        gather_kernel_sim(input_pto, index_pto, out_pto, dim)
    return out


def test_gather_basic(device_id: int = None, run_mode: str = "npu"):
    """Test basic usage of gather function"""
    print("=" * 60)
    print("Test: Basic Usage of gather Function")
    print("=" * 60)
    
    device = f'npu:{device_id}' if (run_mode == "npu" and device_id is not None) else 'cpu'
    
    # Test 1: Basic gathering along dimension 0
    dtype = torch.int32
    input_tensor = torch.tensor([[0, 1, 2, 3, 4],
                                [5, 6, 7, 8, 9],
                                [10, 11, 12, 13, 14]], dtype=dtype, device=device)
    index_tensor = torch.tensor([[0, 1, 2, 0],
                                [1, 2, 0, 1],
                                [2, 2, 1, 0]], dtype=dtype, device=device)
    dim = 0
    expected = torch.tensor([[0, 6, 12, 3],
                            [5, 11, 2, 8],
                            [10, 11, 7, 3]], dtype=dtype, device=device)

    out = gather_op(input_tensor, index_tensor, dim, run_mode)
    max_diff = np.abs(out.cpu().numpy() - expected.cpu().numpy()).max()
    print(f"Output: {out}")
    print(f"Expected: {expected}")
    print(f"Max difference: {max_diff:.6f}")
    if run_mode == "npu":
        assert_allclose(out.cpu().numpy(), expected.cpu().numpy(), rtol=1e-3, atol=1e-3)
    print("✓ Basic usage of gather function completed successfully")


def test_gather_different_dimensions(device_id: int = None, run_mode: str = "npu"):
    """Test gathering tensors along different dimensions"""
    print("=" * 60)
    print("Test: Gathering Tensors Along Different Dimensions")
    print("=" * 60)
    
    device = f'npu:{device_id}' if (run_mode == "npu" and device_id is not None) else 'cpu'
    
    # Test: Gatherenating along dimension 2
    dtype = torch.int32
    input_tensor = torch.tensor([[
                    [10, 20, 30, 40],
                    [50, 60, 70, 80],
                    [90, 100, 110, 120]
                ],
                [
                    [1, 2, 3, 4],
                    [5, 6, 7, 8],
                    [9, 10, 11, 12]
                ]], dtype=dtype, device=device)
    index_tensor = torch.tensor([
                        [
                            [0, 3],
                            [2, 1],
                            [3, 3]
                        ],
                        [
                            [1, 2],
                            [0, 3],
                            [2, 0]
                        ]], dtype=dtype, device=device)
    dim = 2
    expected = torch.tensor([
                        [
                            [10., 40.],
                            [70., 60.],
                            [120., 120.]
                        ],
                        [
                            [2., 3.],
                            [5., 8.],
                            [11., 9.]
                        ]], dtype=dtype, device=device)

    out = gather_op(input_tensor, index_tensor, dim, run_mode)
    max_diff = np.abs(out.cpu().numpy() - expected.cpu().numpy()).max()
    print(f"Output (dim=2): {out}")
    print(f"Expected (dim=2): {expected}")
    print(f"Max difference: {max_diff:.6f}")
    if run_mode == "npu":
        assert_allclose(out.cpu().numpy(), expected.cpu().numpy(), rtol=1e-3, atol=1e-3)
    print("✓ Test gatherenating tensors along different dimensions completed successfully")


def test_gather_negative_indexing(device_id: int = None, run_mode: str = "npu"):
    """Test handling negative indexing"""
    print("=" * 60)
    print("Test: Handling Negative Indexing")
    print("=" * 60)
    
    device = f'npu:{device_id}' if (run_mode == "npu" and device_id is not None) else 'cpu'
    
    # Test 1: Gatherenating along dimension -1
    dtype = torch.int32
    input_tensor = torch.tensor([[0, 1, 2, 3, 4],
                                [5, 6, 7, 8, 9],
                                [10, 11, 12, 13, 14]], dtype=dtype, device=device)
    index_tensor = torch.tensor([[0, 1, 2, 0],
                                [1, 2, 0, 1],
                                [2, 2, 1, 0]], dtype=dtype, device=device)
    dim = -1
    expected = torch.tensor([[0, 1, 2, 0],
                            [6, 7, 5, 6],
                            [12, 12, 11, 10]], dtype=dtype, device=device)

    out = gather_op(input_tensor, index_tensor, dim, run_mode)
    max_diff = np.abs(out.cpu().numpy() - expected.cpu().numpy()).max()
    print(f"Output (dim=-1): {out}")
    print(f"Expected (dim=-1): {expected}")
    print(f"Max difference: {max_diff:.6f}")
    if run_mode == "npu":
        assert_allclose(out.cpu().numpy(), expected.cpu().numpy(), rtol=1e-3, atol=1e-3)


# ============================================================================
# Scatter Examples
# ============================================================================

@pypto.jit
def scatter_kernel_npu(x: pypto.Tensor, y: pypto.Tensor, out: pypto.Tensor) -> None:
    tensor_shape = x.shape
    vec_tile_shapes = [8 for _ in range(len(tensor_shape))]
    pypto.set_vec_tile_shapes(*vec_tile_shapes)

    out[:] = pypto.scatter(x, dim_, y, src_)

@pypto.jit(runtime_options={"run_mode": pypto.RunMode.SIM})
def scatter_kernel_sim(x: pypto.Tensor, y: pypto.Tensor, out: pypto.Tensor) -> None:
    tensor_shape = x.shape
    vec_tile_shapes = [8 for _ in range(len(tensor_shape))]
    pypto.set_vec_tile_shapes(*vec_tile_shapes)

    out[:] = pypto.scatter(x, dim_, y, src_)

def scatter(x: torch.Tensor, dim: int, y: torch.Tensor, src: torch.float32, run_mode: str = "npu") -> torch.Tensor:
    out = torch.zeros(x.shape).to(x.device)
    global dim_, src_
    dim_, src_ = dim, src

    x_pto = pypto.from_torch(x)
    y_pto = pypto.from_torch(y)
    out_pto = pypto.from_torch(out)
    if run_mode == "npu":
        scatter_kernel_npu(x_pto, y_pto, out_pto)
    else:
        scatter_kernel_sim(x_pto, y_pto, out_pto)
    return out


def test_scatter(device_id: int = None, run_mode: str = "npu"):
    """Test basic usage of scatter function"""
    print("=" * 60)
    print("Test: Basic Usage of scatter Function")
    print("=" * 60)

    device = f'npu:{device_id}' if (run_mode == "npu" and device_id is not None) else 'cpu'

    x = torch.rand(3, 5, dtype=torch.float32, device=device)
    dim = 0
    y = torch.randint(0, 3, x.shape, dtype=torch.int64, device=device) # BUG: The shape of y must be the same as x.
    src = 0.5

    golden = torch.scatter(x, dim, y, src)
    output = scatter(x, dim, y, src, run_mode)
    max_diff = np.abs(output.cpu().numpy() - golden.cpu().numpy()).max()
    print(f"Max difference: {max_diff:.6f}")
    if run_mode == "npu":
        assert_allclose(np.array(output.cpu()), np.array(golden.cpu()))
    print("✓ Basic usage of scatter function completed successfully")
    print()


# ============================================================================
# Scatter_update Examples
# ============================================================================
    

@pypto.jit
def scatter_update_kernel_npu(x: pypto.Tensor, y: pypto.Tensor, out: pypto.Tensor) -> None:
    tensor_shape = x.shape
    vec_tile_shapes = [8 for _ in range(len(tensor_shape))]
    pypto.set_vec_tile_shapes(*vec_tile_shapes)

    out[:] = pypto.scatter(x, update_dim_, y, update_src_)

@pypto.jit(runtime_options={"run_mode": pypto.RunMode.SIM})
def scatter_update_kernel_sim(x: pypto.Tensor, y: pypto.Tensor, out: pypto.Tensor) -> None:
    tensor_shape = x.shape
    vec_tile_shapes = [8 for _ in range(len(tensor_shape))]
    pypto.set_vec_tile_shapes(*vec_tile_shapes)

    out[:] = pypto.scatter(x, update_dim_, y, update_src_)


def scatter_update(x: torch.Tensor, dim: int, y: torch.Tensor, src: torch.float32, run_mode: str = "npu") -> torch.Tensor:
    out = torch.zeros(x.shape).to(x.device)
    global update_dim_, update_src_
    update_dim_, update_src_ = dim, src

    x_pto = pypto.from_torch(x)
    y_pto = pypto.from_torch(y)
    out_pto = pypto.from_torch(out)
    if run_mode == "npu":
        scatter_update_kernel_npu(x_pto, y_pto, out_pto)
    else:
        scatter_update_kernel_sim(x_pto, y_pto, out_pto)
    return out


def test_scatter_update(device_id: int = None, run_mode: str = "npu") -> None:
    """Test basic usage of scatter_update function"""
    print("=" * 60)
    print("Test: Basic Usage of scatter_update Function")
    print("=" * 60)

    device = f'npu:{device_id}' if (run_mode == "npu" and device_id is not None) else 'cpu'

    x = torch.rand(3, 5, dtype=torch.float32)
    dim = -2
    y = torch.randint(0, 3, x.shape, dtype=torch.int64)
    src = 2.0

    golden = torch.scatter(x, dim, y, src)
    output = scatter_update(x, dim, y, src, run_mode)
    max_diff = np.abs(output.numpy() - golden.numpy()).max()
    print(f"Max difference: {max_diff:.6f}")
    if run_mode == "npu":
        assert_allclose(np.array(output.cpu()), np.array(golden.cpu()))
    print("✓ Scatter_update test passed")
    print()


# ============================================================================
# Concat Examples
# ============================================================================
    
@pypto.jit
def concat_kernel_npu(a: pypto.Tensor, b: pypto.Tensor, out: pypto.Tensor, dim: int) -> None:
    tile_shapes = [8 for _ in range(len(a.shape))]
    pypto.set_vec_tile_shapes(*tile_shapes)
    out[:] = pypto.concat([a, b], dim=dim)

@pypto.jit(runtime_options={"run_mode": pypto.RunMode.SIM})
def concat_kernel_sim(a: pypto.Tensor, b: pypto.Tensor, out: pypto.Tensor, dim: int) -> None:
    tile_shapes = [8 for _ in range(len(a.shape))]
    pypto.set_vec_tile_shapes(*tile_shapes)
    out[:] = pypto.concat([a, b], dim=dim)


@pypto.jit
def concat_multiple_kernel_npu(a: pypto.Tensor, b: pypto.Tensor, c: pypto.Tensor, out: pypto.Tensor, dim: int) -> None:
    tile_shapes = [8 for _ in range(len(a.shape))]
    pypto.set_vec_tile_shapes(*tile_shapes)
    out[:] = pypto.concat([a, b, c], dim=dim)

@pypto.jit(runtime_options={"run_mode": pypto.RunMode.SIM})
def concat_multiple_kernel_sim(a: pypto.Tensor, b: pypto.Tensor, c: pypto.Tensor, out: pypto.Tensor, dim: int) -> None:
    tile_shapes = [8 for _ in range(len(a.shape))]
    pypto.set_vec_tile_shapes(*tile_shapes)
    out[:] = pypto.concat([a, b, c], dim=dim)


def concat_op(a: torch.Tensor, b: torch.Tensor, dim: int, run_mode: str = "npu", dynamic: bool = False) -> torch.Tensor:
    if dim == 0:
        out_shape = (a.shape[0] + b.shape[0], a.shape[1])
    else:
        out_shape = (a.shape[0], a.shape[1] + b.shape[1])
    out = torch.zeros(out_shape, dtype=a.dtype, device=a.device)

    if dynamic:
        a_pto = pypto.from_torch(a, dynamic_axis=[0])
        b_pto = pypto.from_torch(b, dynamic_axis=[0])
        out_pto = pypto.from_torch(out, dynamic_axis=[0])
    else:
        a_pto = pypto.from_torch(a)
        b_pto = pypto.from_torch(b)
        out_pto = pypto.from_torch(out)
    if run_mode == "npu":
        concat_kernel_npu(a_pto, b_pto, out_pto, dim)
    else:
        concat_kernel_sim(a_pto, b_pto, out_pto, dim)
    return out


def concat_multiple_op(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, dim: int, run_mode: str = "npu", dynamic: bool = False) -> torch.Tensor:
    if dim == 0:
        out_shape = (a.shape[0] + b.shape[0] + c.shape[0], a.shape[1])
    else:
        out_shape = (a.shape[0], a.shape[1] + b.shape[1] + c.shape[1])
    out = torch.zeros(out_shape, dtype=a.dtype, device=a.device)

    if dynamic:
        a_pto = pypto.from_torch(a, dynamic_axis=[0])
        b_pto = pypto.from_torch(b, dynamic_axis=[0])
        c_pto = pypto.from_torch(c, dynamic_axis=[0])
        out_pto = pypto.from_torch(out, dynamic_axis=[0])
    else:
        a_pto = pypto.from_torch(a)
        b_pto = pypto.from_torch(b)
        c_pto = pypto.from_torch(c)
        out_pto = pypto.from_torch(out)
    if run_mode == "npu":
        concat_multiple_kernel_npu(a_pto, b_pto, c_pto, out_pto, dim)
    else:
        concat_multiple_kernel_sim(a_pto, b_pto, c_pto, out_pto, dim)
    return out


def test_concat_basic(device_id: int = None, run_mode: str = "npu"):
    """Test basic usage of concat function"""
    print("=" * 60)
    print("Test: Basic Usage of concat Function")
    print("=" * 60)
    
    device = f'npu:{device_id}' if (run_mode == "npu" and device_id is not None) else 'cpu'
    
    # Test 1: Basic concatenating of two tensors
    dtype = torch.float32
    a = torch.tensor([[1, 1], [1, 1]], dtype=dtype, device=device)
    b = torch.tensor([[0, 0], [0, 0]], dtype=dtype, device=device)
    dim = 0
    expected = torch.tensor([[1, 1], [1, 1],
                             [0, 0], [0, 0]], dtype=dtype, device=device)

    out = concat_op(a, b, dim, run_mode)
    max_diff = np.abs(out.cpu().numpy() - expected.cpu().numpy()).max()
    print(f"Output: {out}")
    print(f"Expected: {expected}")
    print(f"Max difference: {max_diff:.6f}")
    if run_mode == "npu":
        assert_allclose(out.cpu().numpy(), expected.cpu().numpy(), rtol=1e-3, atol=1e-3)
    print("✓ Basic usage of concat function completed successfully")


def test_concat_different_dimensions(device_id: int = None, run_mode: str = "npu"):
    """Test concatenating tensors along different dimensions"""
    print("=" * 60)
    print("Test: Concatenating Tensors Along Different Dimensions")
    print("=" * 60)
    
    device = f'npu:{device_id}' if (run_mode == "npu" and device_id is not None) else 'cpu'
    
    # Test 1: Concatenating along dimension 1
    dtype = torch.float32
    a = torch.tensor([[1, 1], [1, 1]], dtype=dtype, device=device)
    b = torch.tensor([[0, 0], [0, 0]], dtype=dtype, device=device)
    dim = 1
    expected = torch.tensor([[1, 1, 0 ,0],
                             [1, 1, 0 ,0]], dtype=dtype, device=device)

    out = concat_op(a, b, dim, run_mode)
    max_diff = np.abs(out.cpu().numpy() - expected.cpu().numpy()).max()
    print(f"Output: {out}")
    print(f"Expected: {expected}")
    print(f"Max difference: {max_diff:.6f}")
    if run_mode == "npu":
        assert_allclose(out.cpu().numpy(), expected.cpu().numpy(), rtol=1e-3, atol=1e-3)
    print("✓ Test concatenating tensors along different dimensions completed successfully")


def test_concat_multiple_tensors(device_id: int = None, run_mode: str = "npu"):
    """Test concatenating multiple tensors"""
    print("=" * 60)
    print("Test: Concatenating Multiple Tensors")
    print("=" * 60)
    
    device = f'npu:{device_id}' if (run_mode == "npu" and device_id is not None) else 'cpu'
    
    # Test 1: Concatenating of three tensors along dimension 0
    dtype = torch.float32
    a = torch.tensor([[1, 1], [1, 1]], dtype=dtype, device=device)
    b = torch.tensor([[0, 0], [0, 0]], dtype=dtype, device=device)
    c = torch.tensor([[2, 2], [2, 2]], dtype=dtype, device=device)
    dim = 0
    expected = torch.tensor([[1, 1], [1, 1],
                             [0, 0], [0, 0],
                             [2, 2], [2, 2]], dtype=dtype, device=device)

    out = concat_multiple_op(a, b, c, dim, run_mode)
    max_diff = np.abs(out.cpu().numpy() - expected.cpu().numpy()).max()
    print(f"Output: {out}")
    print(f"Expected: {expected}")
    print(f"Max difference: {max_diff:.6f}")
    if run_mode == "npu":
        assert_allclose(out.cpu().numpy(), expected.cpu().numpy(), rtol=1e-3, atol=1e-3)
    print("✓ Test concatenating multiple tensors completed successfully")


def test_concat_different_shapes(device_id: int = None, run_mode: str = "npu"):
    """Test concatenating tensors of different shapes"""
    print("=" * 60)
    print("Test: Concatenating Tensors of Different Shapes")
    print("=" * 60)
    
    device = f'npu:{device_id}' if (run_mode == "npu" and device_id is not None) else 'cpu'
    
    # Test 1: Concatenating Tensors of Different Shapes
    dtype = torch.float32
    a = torch.tensor([[1, 1], [1, 1]], dtype=dtype, device=device)
    b = torch.tensor([[0, 0]], dtype=dtype, device=device)
    dim = 0
    expected = torch.tensor([[1, 1], [1, 1],
                             [0, 0]], dtype=dtype, device=device)

    out = concat_op(a, b, dim, run_mode)
    max_diff = np.abs(out.cpu().numpy() - expected.cpu().numpy()).max()
    print(f"Output: {out}")
    print(f"Expected: {expected}")
    print(f"Max difference: {max_diff:.6f}")
    if run_mode == "npu":
        assert_allclose(out.cpu().numpy(), expected.cpu().numpy(), rtol=1e-3, atol=1e-3)
    print("✓ Test concatenating tensors of different shapes completed successfully")


# ============================================================================
# View Examples
# ============================================================================
    
@pypto.jit
def view_kernel_npu(x: pypto.Tensor, out: pypto.Tensor, shape: list, offsets: list) -> None:
    tile_shapes = [8 for _ in range(len(x.shape))]
    pypto.set_vec_tile_shapes(*tile_shapes)
    out[:] = pypto.view(x, shape, offsets)

@pypto.jit(runtime_options={"run_mode": pypto.RunMode.SIM})
def view_kernel_sim(x: pypto.Tensor, out: pypto.Tensor, shape: list, offsets: list) -> None:
    tile_shapes = [8 for _ in range(len(x.shape))]
    pypto.set_vec_tile_shapes(*tile_shapes)
    out[:] = pypto.view(x, shape, offsets)


def view_op(x: torch.Tensor, shape: list, offsets: list, run_mode: str = "npu", dynamic: bool = False) -> torch.Tensor:
    out = torch.zeros(shape, dtype=x.dtype, device=x.device)

    if dynamic:
        x_pto = pypto.from_torch(x, dynamic_axis=[0])
        out_pto = pypto.from_torch(out, dynamic_axis=[0])
    else:
        x_pto = pypto.from_torch(x)
        out_pto = pypto.from_torch(out)
    if run_mode == "npu":
        view_kernel_npu(x_pto, out_pto, shape, offsets)
    else:
        view_kernel_sim(x_pto, out_pto, shape, offsets)
    return out


def test_view_basic(device_id = None, run_mode: str = "npu"):
    """Test basic usage of view function"""
    print("=" * 60)
    print("Test: Basic Usage of view Function")
    print("=" * 60)
    
    device = f'npu:{device_id}' if (run_mode == "npu" and device_id is not None) else 'cpu'
    
    # Test 1: Basic usage of view function
    dtype = torch.float32
    x = torch.tensor([[1, 1, 2, 2, 3, 3, 4, 4],
                      [1, 1, 2, 2, 3, 3, 4, 4],
                      [1, 1, 2, 2, 3, 3, 4, 4],
                      [1, 1, 2, 2, 3, 3, 4, 4]], dtype=dtype, device=device)
    shape = [4, 4]
    offsets = [0, 4]
    expected = torch.tensor([[3, 3, 4, 4],
                             [3, 3, 4, 4],
                             [3, 3, 4, 4],
                             [3, 3, 4, 4]], dtype=dtype, device=device)

    out = view_op(x, shape, offsets, run_mode)
    max_diff = np.abs(out.cpu().numpy() - expected.cpu().numpy()).max()
    print(f"Output: {out}")
    print(f"Expected: {expected}")
    print(f"Max difference: {max_diff:.6f}")
    if run_mode == "npu":
        assert_allclose(out.cpu().numpy(), expected.cpu().numpy(), rtol=1e-3, atol=1e-3)
    print("✓ Basic usage of view function completed successfully")

    
@pypto.jit
def view_with_valid_shape_kernel_npu(x: pypto.Tensor, out: pypto.Tensor, shape: list, offsets: list, valid_shape: list) -> None:
    tile_shapes = [8 for _ in range(len(x.shape))]
    pypto.set_vec_tile_shapes(*tile_shapes)
    out[:] = pypto.view(x, shape, offsets, valid_shape=valid_shape)

@pypto.jit(runtime_options={"run_mode": pypto.RunMode.SIM})
def view_with_valid_shape_kernel_sim(x: pypto.Tensor, out: pypto.Tensor, shape: list, offsets: list, valid_shape: list) -> None:
    tile_shapes = [8 for _ in range(len(x.shape))]
    pypto.set_vec_tile_shapes(*tile_shapes)
    out[:] = pypto.view(x, shape, offsets, valid_shape=valid_shape)


def view_with_valid_shape_op(x: torch.Tensor, shape: list, offsets: list, valid_shape: list, run_mode: str = "npu", dynamic: bool = False) -> torch.Tensor:
    out = torch.zeros(shape, dtype=x.dtype, device=x.device)

    if dynamic:
        x_pto = pypto.from_torch(x, dynamic_axis=[0])
        out_pto = pypto.from_torch(out, dynamic_axis=[0])
    else:
        x_pto = pypto.from_torch(x)
        out_pto = pypto.from_torch(out)
    if run_mode == "npu":
        view_with_valid_shape_kernel_npu(x_pto, out_pto, shape, offsets, valid_shape)
    else:
        view_with_valid_shape_kernel_sim(x_pto, out_pto, shape, offsets, valid_shape)
    return out


def test_view_with_valid_shape(device_id: int = None, run_mode: str = "npu"):
    """Test using the valid_shape parameter"""
    print("=" * 60)
    print("Test: Using the valid_shape Parameter")
    print("=" * 60)
    
    device = f'npu:{device_id}' if (run_mode == "npu" and device_id is not None) else 'cpu'
    
    # Test 1: Using the valid_shape parameter
    dtype = torch.float32
    x = torch.tensor([[1, 1, 2, 2, 3, 3, 4, 4],
                      [1, 1, 2, 2, 3, 3, 4, 4],
                      [1, 1, 2, 2, 5, 5, 6, 6],
                      [1, 1, 2, 2, 5, 5, 6, 6]], dtype=dtype, device=device)
    shape = [4, 4]
    offsets = [2, 4]
    valid_shape = [2, 4]
    expected = torch.tensor([[5, 5, 6, 6],
                             [5, 5, 6, 6],
                             [0, 0, 0, 0],
                             [0, 0, 0, 0]], dtype=dtype, device=device)

    out = view_with_valid_shape_op(x, shape, offsets, valid_shape, run_mode)
    max_diff = np.abs(out.cpu().numpy() - expected.cpu().numpy()).max()
    print(f"Output: {out}")
    print(f"Expected: {expected}")
    print(f"Max difference: {max_diff:.6f}")
    if run_mode == "npu":
        assert_allclose(out.cpu().numpy(), expected.cpu().numpy(), rtol=1e-3, atol=1e-3)
    print("✓ Using the valid_shape parameter completed successfully")


# ============================================================================
# Transpose Examples
# ============================================================================

@pypto.jit
def transpose_kernel_npu(x: pypto.Tensor, y: pypto.Tensor) -> None:
    tensor_shape = x.shape
    vec_tile_shapes = [8 for _ in range(len(tensor_shape))]
    pypto.set_vec_tile_shapes(*vec_tile_shapes)

    y[:] = pypto.transpose(x, 0, 1)

@pypto.jit(runtime_options={"run_mode": pypto.RunMode.SIM})
def transpose_kernel_sim(x: pypto.Tensor, y: pypto.Tensor) -> None:
    tensor_shape = x.shape
    vec_tile_shapes = [8 for _ in range(len(tensor_shape))]
    pypto.set_vec_tile_shapes(*vec_tile_shapes)

    y[:] = pypto.transpose(x, 0, 1)
        

def transpose(x: torch.Tensor, dim0: int, dim1: int, run_mode: str = "npu") -> torch.Tensor:
    y = torch.zeros(3,2).to(x.device)#torch.empty_like(x)
    global dim_0, dim_1
    dim_0, dim_1 = dim0, dim1

    x_pto = pypto.from_torch(x)
    y_pto = pypto.from_torch(y)

    # launch the kernel
    if run_mode == "npu":
        transpose_kernel_npu(x_pto, y_pto)
    else:
        transpose_kernel_sim
    return y


def test_transpose(device_id: int = None, run_mode: str = "npu"):
    """Test basic usage of transpose function"""
    print("=" * 60)
    print("Test: Basic Usage of transpose Function")
    print("=" * 60)
    
    device = f'npu:{device_id}' if (run_mode == "npu" and device_id is not None) else 'cpu'
    
    dtype = torch.float32
    x = torch.tensor([[1.0028, -0.9893, 0.5809],
                        [-0.1669, 0.7299, 0.4942]], dtype=dtype, device=device)

    dim0, dim1 = 0, 1
    y = transpose(x, dim0, dim1, run_mode).cpu()
    golden = torch.tensor([[ 1.0028, -0.1669],
                        [-0.9893, 0.7299],
                        [ 0.5809, 0.4942]], dtype=dtype, device=f'cpu')

    max_diff = np.abs(y.numpy() - golden.numpy()).max()
    print(f"Output: {y}")
    print(f"Expected: {golden}")
    print(f"Max difference: {max_diff:.6f}")
    if run_mode == "npu":
        assert_allclose(np.array(y), np.array(golden), rtol=1e-3, atol=1e-3)
    print("✓ Basic usage of transpose function completed successfully")


# ============================================================================
# Cast Examples
# ============================================================================
data_type = {
    torch.int8:      pypto.DT_INT8,
    torch.int16:     pypto.DT_INT16,
    torch.int32:     pypto.DT_INT32,
    torch.int64:     pypto.DT_INT64,
    torch.float16:   pypto.DT_FP16,
    torch.float32:   pypto.DT_FP32,
    torch.bfloat16:  pypto.DT_BF16,
    torch.uint8:     pypto.DT_UINT8,
    torch.uint16:    pypto.DT_UINT16,
    torch.uint32:    pypto.DT_UINT32,
    torch.uint64:    pypto.DT_UINT64,
    torch.bool:      pypto.DT_BOOL,
}


@pypto.jit
def cast_kernel_npu(x: pypto.Tensor, dtype: pypto.DataType, y: pypto.Tensor) -> None:
    tensor_shape = x.shape
    vec_tile_shapes = [8 for _ in range(len(tensor_shape))]
    pypto.set_vec_tile_shapes(*vec_tile_shapes)
    y[:] = pypto.cast(x, dtype)

@pypto.jit(runtime_options={"run_mode": pypto.RunMode.SIM})
def cast_kernel_sim(x: pypto.Tensor, dtype: pypto.DataType, y: pypto.Tensor) -> None:
    tensor_shape = x.shape
    vec_tile_shapes = [8 for _ in range(len(tensor_shape))]
    pypto.set_vec_tile_shapes(*vec_tile_shapes)
    y[:] = pypto.cast(x, dtype)
   
   
def cast(x: torch.Tensor, dtype: torch.dtype, run_mode: str = "npu") -> torch.Tensor: 
    y = torch.empty_like(x, dtype=dtype)

    x_pto = pypto.from_torch(x)
    y_pto = pypto.from_torch(y)
    
    pto_type = data_type[dtype]
    # launch the kernel
    if run_mode == "npu":
        cast_kernel_npu(x_pto, pto_type, y_pto)
    else:
        cast_kernel_sim(x_pto, pto_type, y_pto)
    return y   


def test_cast(device_id: int = None, run_mode: str = "npu"):
    """Test basic usage of cast function"""
    print("=" * 60)
    print("Test: Basic Usage of cast Function")
    print("=" * 60)
    
    device = f'npu:{device_id}' if (run_mode == "npu" and device_id is not None) else 'cpu'
    
    dtype = torch.float32
    cast_dtype = torch.float16
    x = torch.tensor([2.0, 3.0], dtype=dtype, device=device)
    y = cast(x, cast_dtype, run_mode).cpu()
    golden = torch.tensor([2.0, 3.0], dtype=cast_dtype, device=device).cpu()
    max_diff = np.abs(y.numpy() - golden.numpy()).max()
    print(f"y.dtype == golden.dtype: {y.dtype == golden.dtype}")
    print(f"Max difference: {max_diff:.6f}")
    if run_mode == "npu":
        assert_allclose(np.array(y), np.array(golden), rtol=1e-3, atol=1e-3)
    print("✓ Basic usage of cast function completed successfully")
  
        
def main():
    """Run transform examples.
    
    Usage:
        python transform_ops.py                          # Run all examples
        python transform_ops.py --list                   # List all available examples
        python transform_ops.py assemble::test_assemble_basic    # Run a specific case
    """
    parser = argparse.ArgumentParser(
        description="PyPTO Transform Operation Examples",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                      Run all examples
  %(prog)s --list               List all available examples
  %(prog)s assemble::test_assemble_basic    Run a specific case
        """
    )
    parser.add_argument(
        'example_id',
        type=str,
        nargs="?",
        help='Run a specific case (e.g., assemble::test_assemble_basic). If omitted, all cases run.'
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
        'assemble::test_assemble_basic': {
            'name': 'Test basic usage of assemble function',
            'description': 'Basic usage of assemble function example',
            'function': test_assemble_basic,
        },
        'assemble::test_assemble_basic': {
            'name': 'Test assemble function with different offsets',
            'description': 'Assemble function with different offsets example',
            'function': test_assemble_different_offsets_shapes,
        },
        'gather::test_gather_basic': {
            'name': 'Test basic usage of gather function',
            'description': 'Basic usage of gather function example',
            'function': test_gather_basic,
        },
        'gather::test_gather_different_dimensions': {
            'name': 'Test gathering tensors along different dimensions',
            'description': 'Gathering tensors along different dimensions example',
            'function': test_gather_different_dimensions,
        },
        'gather::test_gather_negative_indexing': {
            'name': 'Test handling negative indexing',
            'description': 'Handling negative indexing example',
            'function': test_gather_negative_indexing,
        },
        'scatter::test_scatter': {
            'name': 'Test scatter',
            'description': 'Scatter example',
            'function': test_scatter,
        },
        'scatter::test_scatter_update': {
            'name': 'Test scatter_update',
            'description': 'Scatter_update example',
            'function': test_scatter_update,
        },
        'concat::test_concat_basic': {
            'name': 'Test basic usage of concat function',
            'description': 'Basic usage of concat function example',
            'function': test_concat_basic,
        },
        'concat::test_concat_different_dimensions': {
            'name': 'Test concatenating tensors along different dimensions',
            'description': 'Concatenating tensors along different dimensions example',
            'function': test_concat_different_dimensions,
        },
        'concat::test_concat_multiple_tensors': {
            'name': 'Test concatenating multiple tensors',
            'description': 'Concatenating multiple tensors example',
            'function': test_concat_multiple_tensors,
        },
        'concat::test_concat_different_shapes': {
            'name': 'Test concatenating tensors of different shapes',
            'description': 'Concatenating tensors of different shapes',
            'function': test_concat_different_shapes,
        },
        'view::test_view_basic': {
            'name': 'Test basic usage of view function',
            'description': 'Basic usage of view function example',
            'function': test_view_basic,
        },
        'view::test_view_with_valid_shape': {
            'name': 'Test using the valid_shape parameter',
            'description': 'Using the valid_shape parameter example',
            'function': test_view_with_valid_shape,
        },
        'transpose::test_transpose': {
            'name': 'Test basic usage of transpose function',
            'description': 'Basic usage of transpose function example',
            'function': test_transpose,
        },
        'cast::test_cast': {
            'name': 'Test basic usage of cast function',
            'description': 'Basic usage of cast function example',
            'function': test_cast,
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
    device_id = None
    examples_to_run = []
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
    print("PyPTO Transform Operation Examples")
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
            print("All transform tests passed!")
            print("=" * 60)
        
    except Exception as e:
        print(f"\nError: {e}")
        raise


if __name__ == "__main__":
    main()

