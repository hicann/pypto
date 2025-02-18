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
Element-wise Operation Examples for PyPTO

This file contains all element-wise operation examples merged into a single file.
You can run all examples or select specific ones using command-line arguments.

Usage:
    python elementwise.py              # Run all examples
    python elementwise.py --list       # List all available examples
    python elementwise.py abs::test_abs_basic    # Run a specific case
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
# ABS Examples
# ============================================================================

@pypto.jit
def abs_kernel(x: pypto.Tensor, out: pypto.Tensor) -> None:
    pypto.set_vec_tile_shapes(2, 8)
    out[:] = pypto.abs(x)


@pypto.jit(runtime_options={"run_mode": pypto.RunMode.SIM})
def abs_kernel_sim(x: pypto.Tensor, out: pypto.Tensor) -> None:
    pypto.set_vec_tile_shapes(2, 8)
    out[:] = pypto.abs(x)


def abs_op(x: torch.Tensor, run_mode: str = "npu", dynamic: bool = False) -> torch.Tensor:
    out = torch.zeros_like(x)

    if dynamic:
        x_pto = pypto.from_torch(x, dynamic_axis=[0])
        out_pto = pypto.from_torch(out, dynamic_axis=[0])
    else:
        x_pto = pypto.from_torch(x)
        out_pto = pypto.from_torch(out)

    if run_mode == "npu":
        abs_kernel(x_pto, out_pto)
    else:
        abs_kernel_sim(x_pto, out_pto)

    return out


def test_abs_basic(device_id: int = None, run_mode: str = "npu"):
    """Test basic usage of abs function"""
    print("=" * 60)
    print("Test: Basic Usage of abs Function")
    print("=" * 60)
    
    device = f'npu:{device_id}' if (run_mode == "npu" and device_id is not None) else 'cpu'

    dtype = torch.float32
    x = torch.tensor([-1, -8, 2], dtype=dtype, device=device)
    expected = torch.tensor([1, 8, 2], dtype=dtype, device=device)

    out = abs_op(x, run_mode)
    if run_mode == "npu":
        assert_allclose(out.cpu().numpy(), expected.cpu().numpy(), rtol=1e-3, atol=1e-3)
    print(f"Output: {out}")
    print(f"Expected: {expected}")
    print("✓ Basic usage of abs function completed successfully")


# ============================================================================
# ADD Examples
# ============================================================================

@pypto.jit
def add_kernel(a: pypto.Tensor, b: pypto.Tensor, out: pypto.Tensor) -> None:
    pypto.set_vec_tile_shapes(2, 8)
    out[:] = pypto.add(a, b)


@pypto.jit(runtime_options={"run_mode": pypto.RunMode.SIM})
def add_kernel_sim(a: pypto.Tensor, b: pypto.Tensor, out: pypto.Tensor) -> None:
    pypto.set_vec_tile_shapes(2, 8)
    out[:] = pypto.add(a, b)


def add_op(a: torch.Tensor, b: torch.Tensor, run_mode: str = "npu", dynamic: bool = False) -> torch.Tensor:
    out = torch.zeros_like(a)

    if dynamic:
        a_pto = pypto.from_torch(a, dynamic_axis=[0])
        b_pto = pypto.from_torch(b, dynamic_axis=[0])
        out_pto = pypto.from_torch(out, dynamic_axis=[0])
    else:
        a_pto = pypto.from_torch(a)
        b_pto = pypto.from_torch(b)
        out_pto = pypto.from_torch(out)

    if run_mode == "npu":
        add_kernel(a_pto, b_pto, out_pto)
    else:
        add_kernel_sim(a_pto, b_pto, out_pto)

    return out


def test_add_basic(device_id: int = None, run_mode: str = "npu"):
    """Test basic usage of add function"""
    print("=" * 60)
    print("Test: Basic Usage of add Function")
    print("=" * 60)
    
    device = f'npu:{device_id}' if (run_mode == "npu" and device_id is not None) else 'cpu'
    
    dtype = torch.float32
    a = torch.tensor([1, 2, 3], dtype=dtype, device=device)
    b = torch.tensor([4, 5, 6], dtype=dtype, device=device)
    expected = torch.tensor([5, 7, 9], dtype=dtype, device=device)

    out = add_op(a, b, run_mode)
    if run_mode == "npu":
        assert_allclose(out.cpu().numpy(), expected.cpu().numpy(), rtol=1e-3, atol=1e-3)
    print(f"Output: {out}")
    print(f"Expected: {expected}")
    print("✓ Basic usage of add function completed successfully")


@pypto.jit
def add_broadcast_kernel(a: pypto.Tensor, b: pypto.Tensor, out: pypto.Tensor) -> None:
    pypto.set_vec_tile_shapes(2, 8)
    out[:] = pypto.add(a, b)


@pypto.jit(runtime_options={"run_mode": pypto.RunMode.SIM})
def add_broadcast_kernel_sim(a: pypto.Tensor, b: pypto.Tensor, out: pypto.Tensor) -> None:
    pypto.set_vec_tile_shapes(2, 8)
    out[:] = pypto.add(a, b)


def add_broadcast_op(a: torch.Tensor, b: torch.Tensor, run_mode: str = "npu", dynamic: bool = False) -> torch.Tensor:
    out = torch.zeros_like(a)

    if dynamic:
        a_pto = pypto.from_torch(a, dynamic_axis=[0])
        b_pto = pypto.from_torch(b, dynamic_axis=[0])
        out_pto = pypto.from_torch(out, dynamic_axis=[0])
    else:
        a_pto = pypto.from_torch(a)
        b_pto = pypto.from_torch(b)
        out_pto = pypto.from_torch(out)

    if run_mode == "npu":
        add_broadcast_kernel(a_pto, b_pto, out_pto)
    else:
        add_broadcast_kernel_sim(a_pto, b_pto, out_pto)

    return out


def test_add_broadcast(device_id: int = None, run_mode: str = "npu"):
    """Test broadcasting between tensors of different shapes"""
    print("=" * 60)
    print("Test: Broadcasting Between Tensors")
    print("=" * 60)
    
    device = f'npu:{device_id}' if (run_mode == "npu" and device_id is not None) else 'cpu'
    
    dtype = torch.float32
    a = torch.tensor([[1, 2], [3, 4]], dtype=dtype, device=device)
    b = torch.tensor([1, 2], dtype=dtype, device=device)
    expected = torch.tensor([[2, 4], [4, 6]], dtype=dtype, device=device)

    out = add_broadcast_op(a, b, run_mode)
    if run_mode == "npu":
        assert_allclose(out.cpu().numpy(), expected.cpu().numpy(), rtol=1e-3, atol=1e-3)
    print(f"Output: {out}")
    print(f"Expected: {expected}")
    print("✓ Test Broadcasting Between Tensors completed successfully")


@pypto.jit
def add_scalar_kernel(a: pypto.Tensor, out: pypto.Tensor, scalar: float) -> None:
    pypto.set_vec_tile_shapes(2, 8)
    out[:] = pypto.add(a, scalar)


@pypto.jit(runtime_options={"run_mode": pypto.RunMode.SIM})
def add_scalar_kernel_sim(a: pypto.Tensor, out: pypto.Tensor, scalar: float) -> None:
    pypto.set_vec_tile_shapes(2, 8)
    out[:] = pypto.add(a, scalar)


def add_scalar_op(a: torch.Tensor, scalar: float, run_mode: str = "npu", dynamic: bool = False) -> torch.Tensor:
    out = torch.zeros_like(a)

    if dynamic:
        a_pto = pypto.from_torch(a, dynamic_axis=[0])
        out_pto = pypto.from_torch(out, dynamic_axis=[0])
    else:
        a_pto = pypto.from_torch(a)
        out_pto = pypto.from_torch(out)

    if run_mode == "npu":
        add_scalar_kernel(a_pto, out_pto, scalar)
    else:
        add_scalar_kernel_sim(a_pto, out_pto, scalar)

    return out


def test_add_scalar(device_id: int = None, run_mode: str = "npu"):
    """Test adding a scalar to a tensor"""
    print("=" * 60)
    print("Test: Adding a scalar to a tensor")
    print("=" * 60)
    
    device = f'npu:{device_id}' if (run_mode == "npu" and device_id is not None) else 'cpu'
    
    dtype = torch.float32
    a = torch.tensor([1, 2, 3], dtype=dtype, device=device)
    scalar = 2.0
    expected = torch.tensor([3, 4, 5], dtype=dtype, device=device)

    out = add_scalar_op(a, scalar, run_mode)
    if run_mode == "npu":
        assert_allclose(out.cpu().numpy(), expected.cpu().numpy(), rtol=1e-3, atol=1e-3)
    print(f"Output: {out}")
    print(f"Expected: {expected}")
    print("✓ Test Adding a scalar to a tensor completed successfully")


@pypto.jit
def add_with_alpha_kernel(a: pypto.Tensor, b: pypto.Tensor, out: pypto.Tensor, alpha: float) -> None:
    pypto.set_vec_tile_shapes(2, 8)
    out[:] = pypto.add(a, b, alpha=alpha)


@pypto.jit(runtime_options={"run_mode": pypto.RunMode.SIM})
def add_with_alpha_kernel_sim(a: pypto.Tensor, b: pypto.Tensor, out: pypto.Tensor, alpha: float) -> None:
    pypto.set_vec_tile_shapes(2, 8)
    out[:] = pypto.add(a, b, alpha=alpha)


def add_with_alpha_op(a: torch.Tensor, b: torch.Tensor, alpha: float, run_mode: str = "npu", dynamic: bool = False) -> torch.Tensor:
    out = torch.zeros_like(a)

    if dynamic:
        a_pto = pypto.from_torch(a, dynamic_axis=[0])
        b_pto = pypto.from_torch(b, dynamic_axis=[0])
        out_pto = pypto.from_torch(out, dynamic_axis=[0])
    else:
        a_pto = pypto.from_torch(a)
        b_pto = pypto.from_torch(b)
        out_pto = pypto.from_torch(out)

    if run_mode == "npu":
        add_with_alpha_kernel(a_pto, b_pto, out_pto, alpha)
    else:
        add_with_alpha_kernel_sim(a_pto, b_pto, out_pto, alpha)

    return out


def test_add_with_alpha(device_id: int = None, run_mode: str = "npu"):
    """Using the alpha parameter to scale the second input"""
    print("=" * 60)
    print("Test: Using the Alpha Parameter")
    print("=" * 60)
    
    device = f'npu:{device_id}' if (run_mode == "npu" and device_id is not None) else 'cpu'
    
    dtype = torch.float32
    a = torch.tensor([1, 2, 3], dtype=dtype, device=device)
    b = torch.tensor([4, 5, 6], dtype=dtype, device=device)
    alpha = 2.0
    expected = torch.tensor([9, 12, 15], dtype=dtype, device=device)

    out = add_with_alpha_op(a, b, alpha, run_mode)
    if run_mode == "npu":
        assert_allclose(out.cpu().numpy(), expected.cpu().numpy(), rtol=1e-3, atol=1e-3)
    print(f"Output: {out}")
    print(f"Expected: {expected}")
    print("✓ Using the alpha parameter to scale the second input completed successfully")


# ============================================================================
# CLIP Examples
# ============================================================================

@pypto.jit
def clip_kernel(a: pypto.Tensor, min_: pypto.Tensor, max_: pypto.Tensor, out: pypto.Tensor) -> None:
    pypto.set_vec_tile_shapes(2, 8)
    out[:] = pypto.clip(a, min_, max_)


@pypto.jit(runtime_options={"run_mode": pypto.RunMode.SIM})
def clip_kernel_sim(a: pypto.Tensor, min_: pypto.Tensor, max_: pypto.Tensor, out: pypto.Tensor) -> None:
    pypto.set_vec_tile_shapes(2, 8)
    out[:] = pypto.clip(a, min_, max_)


def clip_op(a: torch.Tensor, min_: torch.Tensor, max_: torch.Tensor, run_mode: str = "npu", dynamic: bool = False) -> torch.Tensor:
    out = torch.zeros_like(a)

    if dynamic:
        a_pto = pypto.from_torch(a, dynamic_axis=[0])
        min_pto = pypto.from_torch(min_, dynamic_axis=[0])
        max_pto = pypto.from_torch(max_, dynamic_axis=[0])
        out_pto = pypto.from_torch(out, dynamic_axis=[0])
    else:
        a_pto = pypto.from_torch(a)
        min_pto = pypto.from_torch(min_)
        max_pto = pypto.from_torch(max_)
        out_pto = pypto.from_torch(out)

    if run_mode == "npu":
        clip_kernel(a_pto, min_pto, max_pto, out_pto)
    else:
        clip_kernel_sim(a_pto, min_pto, max_pto, out_pto)

    return out


def test_clip_basic(device_id: int = None, run_mode: str = "npu"):
    """Test basic usage of clip function"""
    print("=" * 60)
    print("Test: Basic Usage of clip Function")
    print("=" * 60)
    
    device = f'npu:{device_id}' if (run_mode == "npu" and device_id is not None) else 'cpu'
    
    dtype = torch.float32
    a = torch.tensor([[0, 2, 4], [3, 4, 6]], dtype=dtype, device=device)
    min_ = torch.tensor([[1, 1, 1], [1, 1, 1]], dtype=dtype, device=device)
    max_ = torch.tensor([[3, 3, 3], [3, 3, 3]], dtype=dtype, device=device)
    expected = torch.tensor([[1, 2, 3], [3, 3, 3]], dtype=dtype, device=device)

    out = clip_op(a, min_, max_, run_mode)
    if run_mode == "npu":
        assert_allclose(out.cpu().numpy(), expected.cpu().numpy(), rtol=1e-3, atol=1e-3)
    print(f"Output: {out}")
    print(f"Expected: {expected}")
    print("✓ Basic usage of clip function completed successfully")


@pypto.jit
def clip_broadcast_kernel(a: pypto.Tensor, min_: pypto.Tensor, max_: pypto.Tensor, out: pypto.Tensor) -> None:
    pypto.set_vec_tile_shapes(2, 8)
    out[:] = pypto.clip(a, min_, max_)


@pypto.jit(runtime_options={"run_mode": pypto.RunMode.SIM})
def clip_broadcast_kernel_sim(a: pypto.Tensor, min_: pypto.Tensor, max_: pypto.Tensor, out: pypto.Tensor) -> None:
    pypto.set_vec_tile_shapes(2, 8)
    out[:] = pypto.clip(a, min_, max_)


def clip_broadcast_op(a: torch.Tensor, min_: torch.Tensor, max_: torch.Tensor, run_mode: str = "npu", dynamic: bool = False) -> torch.Tensor:
    out = torch.zeros_like(a)

    if dynamic:
        a_pto = pypto.from_torch(a, dynamic_axis=[0])
        min_pto = pypto.from_torch(min_, dynamic_axis=[0])
        max_pto = pypto.from_torch(max_, dynamic_axis=[0])
        out_pto = pypto.from_torch(out, dynamic_axis=[0])
    else:
        a_pto = pypto.from_torch(a)
        min_pto = pypto.from_torch(min_)
        max_pto = pypto.from_torch(max_)
        out_pto = pypto.from_torch(out)

    if run_mode == "npu":
        clip_broadcast_kernel(a_pto, min_pto, max_pto, out_pto)
    else:
        clip_broadcast_kernel_sim(a_pto, min_pto, max_pto, out_pto)

    return out


def test_clip_broadcast(device_id: int = None, run_mode: str = "npu"):
    """Test broadcasting between tensors of different shapes"""
    print("=" * 60)
    print("Test: Broadcasting Between Tensors")
    print("=" * 60)
    
    device = f'npu:{device_id}' if (run_mode == "npu" and device_id is not None) else 'cpu'
    
    dtype = torch.float32
    a = torch.tensor([[0, 2, 4], [3, 4, 6]], dtype=dtype, device=device)
    min_ = torch.tensor([1, 1, 1], dtype=dtype, device=device)
    max_ = torch.tensor([3, 3, 3], dtype=dtype, device=device)
    expected = torch.tensor([[1, 2, 3], [3, 3, 3]], dtype=dtype, device=device)

    out = clip_broadcast_op(a, min_, max_, run_mode)
    if run_mode == "npu":
        assert_allclose(out.cpu().numpy(), expected.cpu().numpy(), rtol=1e-3, atol=1e-3)
    print(f"Output: {out}")
    print(f"Expected: {expected}")
    print("✓ Test Broadcasting Between Tensors completed successfully")


# ============================================================================
# DIV Examples
# ============================================================================

@pypto.jit
def div_kernel(a: pypto.Tensor, b: pypto.Tensor, out: pypto.Tensor) -> None:
    pypto.set_vec_tile_shapes(2, 8)
    out[:] = pypto.div(a, b)


@pypto.jit(runtime_options={"run_mode": pypto.RunMode.SIM})
def div_kernel_sim(a: pypto.Tensor, b: pypto.Tensor, out: pypto.Tensor) -> None:
    pypto.set_vec_tile_shapes(2, 8)
    out[:] = pypto.div(a, b)


def div_op(a: torch.Tensor, b: torch.Tensor, run_mode: str = "npu", dynamic: bool = False) -> torch.Tensor:
    out = torch.zeros_like(a)

    if dynamic:
        a_pto = pypto.from_torch(a, dynamic_axis=[0])
        b_pto = pypto.from_torch(b, dynamic_axis=[0])
        out_pto = pypto.from_torch(out, dynamic_axis=[0])
    else:
        a_pto = pypto.from_torch(a)
        b_pto = pypto.from_torch(b)
        out_pto = pypto.from_torch(out)

    if run_mode == "npu":
        div_kernel(a_pto, b_pto, out_pto)
    else:
        div_kernel_sim(a_pto, b_pto, out_pto)

    return out


def test_div_basic(device_id: int = None, run_mode: str = "npu"):
    """Test basic usage of div function"""
    print("=" * 60)
    print("Test: Basic Usage of div Function")
    print("=" * 60)
    
    device = f'npu:{device_id}' if (run_mode == "npu" and device_id is not None) else 'cpu'
    
    dtype = torch.float32
    a = torch.tensor([6, 10, 15], dtype=dtype, device=device)
    b = torch.tensor([2, 5, 3], dtype=dtype, device=device)
    expected = torch.tensor([3, 2, 5], dtype=dtype, device=device)

    out = div_op(a, b, run_mode)
    if run_mode == "npu":
        assert_allclose(out.cpu().numpy(), expected.cpu().numpy(), rtol=1e-3, atol=1e-3)
    print(f"Output: {out}")
    print(f"Expected: {expected}")
    print("✓ Basic usage of div function completed successfully")


@pypto.jit
def div_broadcast_kernel(a: pypto.Tensor, b: pypto.Tensor, out: pypto.Tensor) -> None:
    pypto.set_vec_tile_shapes(2, 8)
    out[:] = pypto.div(a, b)


@pypto.jit(runtime_options={"run_mode": pypto.RunMode.SIM})
def div_broadcast_kernel_sim(a: pypto.Tensor, b: pypto.Tensor, out: pypto.Tensor) -> None:
    pypto.set_vec_tile_shapes(2, 8)
    out[:] = pypto.div(a, b)


def div_broadcast_op(a: torch.Tensor, b: torch.Tensor, run_mode: str = "npu", dynamic: bool = False) -> torch.Tensor:
    out = torch.zeros_like(a)

    if dynamic:
        a_pto = pypto.from_torch(a, dynamic_axis=[0])
        b_pto = pypto.from_torch(b, dynamic_axis=[0])
        out_pto = pypto.from_torch(out, dynamic_axis=[0])
    else:
        a_pto = pypto.from_torch(a)
        b_pto = pypto.from_torch(b)
        out_pto = pypto.from_torch(out)

    if run_mode == "npu":
        div_broadcast_kernel(a_pto, b_pto, out_pto)
    else:
        div_broadcast_kernel_sim(a_pto, b_pto, out_pto)

    return out


def test_div_broadcast(device_id: int = None, run_mode: str = "npu"):
    """Test broadcasting between tensors of different shapes"""
    print("=" * 60)
    print("Test: Broadcasting Between Tensors")
    print("=" * 60)
    
    device = f'npu:{device_id}' if (run_mode == "npu" and device_id is not None) else 'cpu'
    
    dtype = torch.float32
    a = torch.tensor([[1, 2], [3, 4]], dtype=dtype, device=device)
    b = torch.tensor([1, 2], dtype=dtype, device=device)
    expected = torch.tensor([[1, 1], [3, 2]], dtype=dtype, device=device)

    out = div_broadcast_op(a, b, run_mode)
    if run_mode == "npu":
        assert_allclose(out.cpu().numpy(), expected.cpu().numpy(), rtol=1e-3, atol=1e-3)
    print(f"Output: {out}")
    print(f"Expected: {expected}")
    print("✓ Test Broadcasting Between Tensors completed successfully")


@pypto.jit
def div_scalar_kernel(a: pypto.Tensor, out: pypto.Tensor, scalar: float) -> None:
    pypto.set_vec_tile_shapes(2, 8)
    out[:] = pypto.div(a, scalar)


@pypto.jit(runtime_options={"run_mode": pypto.RunMode.SIM})
def div_scalar_kernel_sim(a: pypto.Tensor, out: pypto.Tensor, scalar: float) -> None:
    pypto.set_vec_tile_shapes(2, 8)
    out[:] = pypto.div(a, scalar)


def div_scalar_op(a: torch.Tensor, scalar: float, run_mode: str = "npu", dynamic: bool = False) -> torch.Tensor:
    out = torch.zeros_like(a)

    if dynamic:
        a_pto = pypto.from_torch(a, dynamic_axis=[0])
        out_pto = pypto.from_torch(out, dynamic_axis=[0])
    else:
        a_pto = pypto.from_torch(a)
        out_pto = pypto.from_torch(out)

    if run_mode == "npu":
        div_scalar_kernel(a_pto, out_pto, scalar)
    else:
        div_scalar_kernel_sim(a_pto, out_pto, scalar)

    return out


def test_div_scalar(device_id: int = None, run_mode: str = "npu"):
    """Test diving a scalar to a tensor"""
    print("=" * 60)
    print("Test: Diving a scalar to a tensor")
    print("=" * 60)
    
    device = f'npu:{device_id}' if (run_mode == "npu" and device_id is not None) else 'cpu'
    
    dtype = torch.float32
    a = torch.tensor([1, 2, 3], dtype=dtype, device=device)
    scalar = 2.0
    expected = torch.tensor([0.5, 1, 1.5], dtype=dtype, device=device)

    out = div_scalar_op(a, scalar, run_mode)
    if run_mode == "npu":
        assert_allclose(out.cpu().numpy(), expected.cpu().numpy(), rtol=1e-3, atol=1e-3)
    print(f"Output: {out}")
    print(f"Expected: {expected}")
    print("✓ Test Diving a scalar to a tensor completed successfully")


# ============================================================================
# EXP Examples
# ============================================================================

@pypto.jit
def exp_kernel(x: pypto.Tensor, out: pypto.Tensor) -> None:
    pypto.set_vec_tile_shapes(2, 8)
    out[:] = pypto.exp(x)


@pypto.jit(runtime_options={"run_mode": pypto.RunMode.SIM})
def exp_kernel_sim(x: pypto.Tensor, out: pypto.Tensor) -> None:
    pypto.set_vec_tile_shapes(2, 8)
    out[:] = pypto.exp(x)


def exp_op(x: torch.Tensor, run_mode: str = "npu", dynamic: bool = False) -> torch.Tensor:
    out = torch.zeros_like(x)

    if dynamic:
        x_pto = pypto.from_torch(x, dynamic_axis=[0])
        out_pto = pypto.from_torch(out, dynamic_axis=[0])
    else:
        x_pto = pypto.from_torch(x)
        out_pto = pypto.from_torch(out)

    if run_mode == "npu":
        exp_kernel(x_pto, out_pto)
    else:
        exp_kernel_sim(x_pto, out_pto)

    return out


def test_exp_basic(device_id: int = None, run_mode: str = "npu"):
    """Test basic usage of exp function"""
    print("=" * 60)
    print("Test: Basic Usage of exp Function")
    print("=" * 60)
    
    device = f'npu:{device_id}' if (run_mode == "npu" and device_id is not None) else 'cpu'
    
    dtype = torch.float32
    x = torch.tensor([0, 1, 2], dtype=dtype, device=device)
    expected = torch.tensor([1.0000, 2.7183, 7.3891], dtype=dtype, device=device)

    out = exp_op(x, run_mode)
    if run_mode == "npu":
        assert_allclose(out.cpu().numpy(), expected.cpu().numpy(), rtol=1e-3, atol=1e-3)
    print(f"Output: {out}")
    print(f"Expected: {expected}")
    print("✓ Basic usage of exp function completed successfully")


# ============================================================================
# LOG Examples
# ============================================================================

@pypto.jit
def log_kernel(a: pypto.Tensor, out: pypto.Tensor) -> None:
    pypto.set_vec_tile_shapes(2, 8)
    out[:] = pypto.log(a)


@pypto.jit(runtime_options={"run_mode": pypto.RunMode.SIM})
def log_kernel_sim(a: pypto.Tensor, out: pypto.Tensor) -> None:
    pypto.set_vec_tile_shapes(2, 8)
    out[:] = pypto.log(a)


def log_op(a: torch.Tensor, run_mode: str = "npu", dynamic: bool = False) -> torch.Tensor:
    out = torch.zeros_like(a)

    if dynamic:
        a_pto = pypto.from_torch(a, dynamic_axis=[0])
        out_pto = pypto.from_torch(out, dynamic_axis=[0])
    else:
        a_pto = pypto.from_torch(a)
        out_pto = pypto.from_torch(out)

    if run_mode == "npu":
        log_kernel(a_pto, out_pto)
    else:
        log_kernel_sim(a_pto, out_pto)

    return out


def test_log_basic(device_id: int = None, run_mode: str = "npu"):
    """Test basic usage of log function"""
    print("=" * 60)
    print("Test: Basic Usage of log Function")
    print("=" * 60)
    
    device = f'npu:{device_id}' if (run_mode == "npu" and device_id is not None) else 'cpu'
    
    dtype = torch.float32
    a = torch.tensor([1, 2, 3], dtype=dtype, device=device)
    expected = torch.tensor([0, 0.6931, 1.0986], dtype=dtype, device=device)

    out = log_op(a, run_mode)
    if run_mode == "npu":
        assert_allclose(out.cpu().numpy(), expected.cpu().numpy(), rtol=1e-3, atol=1e-3)
    print(f"Output: {out}")
    print(f"Expected: {expected}")
    print("✓ Basic usage of log function completed successfully")


# ============================================================================
# MUL Examples
# ============================================================================

@pypto.jit
def mul_kernel(a: pypto.Tensor, b: pypto.Tensor, out: pypto.Tensor) -> None:
    pypto.set_vec_tile_shapes(2, 8)
    out[:] = pypto.mul(a, b)


@pypto.jit(runtime_options={"run_mode": pypto.RunMode.SIM})
def mul_kernel_sim(a: pypto.Tensor, b: pypto.Tensor, out: pypto.Tensor) -> None:
    pypto.set_vec_tile_shapes(2, 8)
    out[:] = pypto.mul(a, b)


def mul_op(a: torch.Tensor, b: torch.Tensor, run_mode: str = "npu", dynamic: bool = False) -> torch.Tensor:
    out = torch.zeros_like(a)

    if dynamic:
        a_pto = pypto.from_torch(a, dynamic_axis=[0])
        b_pto = pypto.from_torch(b, dynamic_axis=[0])
        out_pto = pypto.from_torch(out, dynamic_axis=[0])
    else:
        a_pto = pypto.from_torch(a)
        b_pto = pypto.from_torch(b)
        out_pto = pypto.from_torch(out)

    if run_mode == "npu":
        mul_kernel(a_pto, b_pto, out_pto)
    else:
        mul_kernel_sim(a_pto, b_pto, out_pto)

    return out


def test_mul_basic(device_id: int = None, run_mode: str = "npu"):
    """Test basic usage of mul function"""
    print("=" * 60)
    print("Test: Basic Usage of mul Function")
    print("=" * 60)
    
    device = f'npu:{device_id}' if (run_mode == "npu" and device_id is not None) else 'cpu'
    
    dtype = torch.float32
    a = torch.tensor([1, 2, 3], dtype=dtype, device=device)
    b = torch.tensor([4, 5, 6], dtype=dtype, device=device)
    expected = torch.tensor([4, 10, 18], dtype=dtype, device=device)

    out = mul_op(a, b, run_mode)
    if run_mode == "npu":
        assert_allclose(out.cpu().numpy(), expected.cpu().numpy(), rtol=1e-3, atol=1e-3)
    print(f"Output: {out}")
    print(f"Expected: {expected}")
    print("✓ Basic usage of mul function completed successfully")


@pypto.jit
def mul_broadcast_kernel(a: pypto.Tensor, b: pypto.Tensor, out: pypto.Tensor) -> None:
    pypto.set_vec_tile_shapes(2, 8)
    out[:] = pypto.mul(a, b)


@pypto.jit(runtime_options={"run_mode": pypto.RunMode.SIM})
def mul_broadcast_kernel_sim(a: pypto.Tensor, b: pypto.Tensor, out: pypto.Tensor) -> None:
    pypto.set_vec_tile_shapes(2, 8)
    out[:] = pypto.mul(a, b)


def mul_broadcast_op(a: torch.Tensor, b: torch.Tensor, run_mode: str = "npu", dynamic: bool = False) -> torch.Tensor:
    out = torch.zeros_like(a)

    if dynamic:
        a_pto = pypto.from_torch(a, dynamic_axis=[0])
        b_pto = pypto.from_torch(b, dynamic_axis=[0])
        out_pto = pypto.from_torch(out, dynamic_axis=[0])
    else:
        a_pto = pypto.from_torch(a)
        b_pto = pypto.from_torch(b)
        out_pto = pypto.from_torch(out)

    if run_mode == "npu":
        mul_broadcast_kernel(a_pto, b_pto, out_pto)
    else:
        mul_broadcast_kernel_sim(a_pto, b_pto, out_pto)

    return out


def test_mul_broadcast(device_id: int = None, run_mode: str = "npu"):
    """Test broadcasting between tensors of different shapes"""
    print("=" * 60)
    print("Test: Broadcasting Between Tensors")
    print("=" * 60)
    
    device = f'npu:{device_id}' if (run_mode == "npu" and device_id is not None) else 'cpu'
    
    dtype = torch.float32
    a = torch.tensor([[1, 2], [3, 4]], dtype=dtype, device=device)
    b = torch.tensor([1, 2], dtype=dtype, device=device)
    expected = torch.tensor([[1, 4], [3, 8]], dtype=dtype, device=device)

    out = mul_broadcast_op(a, b, run_mode)
    if run_mode == "npu":
        assert_allclose(out.cpu().numpy(), expected.cpu().numpy(), rtol=1e-3, atol=1e-3)
    print(f"Output: {out}")
    print(f"Expected: {expected}")
    print("✓ Test Broadcasting Between Tensors completed successfully")


@pypto.jit
def mul_scalar_kernel(a: pypto.Tensor, out: pypto.Tensor, scalar: float) -> None:
    pypto.set_vec_tile_shapes(2, 8)
    out[:] = pypto.mul(a, scalar)


@pypto.jit(runtime_options={"run_mode": pypto.RunMode.SIM})
def mul_scalar_kernel_sim(a: pypto.Tensor, out: pypto.Tensor, scalar: float) -> None:
    pypto.set_vec_tile_shapes(2, 8)
    out[:] = pypto.mul(a, scalar)


def mul_scalar_op(a: torch.Tensor, scalar: float, run_mode: str = "npu", dynamic: bool = False) -> torch.Tensor:
    out = torch.zeros_like(a)

    if dynamic:
        a_pto = pypto.from_torch(a, dynamic_axis=[0])
        out_pto = pypto.from_torch(out, dynamic_axis=[0])
    else:
        a_pto = pypto.from_torch(a)
        out_pto = pypto.from_torch(out)

    if run_mode == "npu":
        mul_scalar_kernel(a_pto, out_pto, scalar)
    else:
        mul_scalar_kernel_sim(a_pto, out_pto, scalar)

    return out


def test_mul_scalar(device_id: int = None, run_mode: str = "npu"):
    """Test muling a scalar to a tensor"""
    print("=" * 60)
    print("Test: Muling a scalar to a tensor")
    print("=" * 60)
    
    device = f'npu:{device_id}' if (run_mode == "npu" and device_id is not None) else 'cpu'
    
    dtype = torch.float32
    a = torch.tensor([1, 2, 3], dtype=dtype, device=device)
    scalar = 2.0
    expected = torch.tensor([2, 4, 6], dtype=dtype, device=device)

    out = mul_scalar_op(a, scalar, run_mode)
    if run_mode == "npu":
        assert_allclose(out.cpu().numpy(), expected.cpu().numpy(), rtol=1e-3, atol=1e-3)
    print(f"Output: {out}")
    print(f"Expected: {expected}")
    print("✓ Test Muling a scalar to a tensor completed successfully")


# ============================================================================
# NEG Examples
# ============================================================================

@pypto.jit
def neg_kernel(a: pypto.Tensor, out: pypto.Tensor) -> None:
    pypto.set_vec_tile_shapes(2, 8)
    out[:] = pypto.neg(a)


@pypto.jit(runtime_options={"run_mode": pypto.RunMode.SIM})
def neg_kernel_sim(a: pypto.Tensor, out: pypto.Tensor) -> None:
    pypto.set_vec_tile_shapes(2, 8)
    out[:] = pypto.neg(a)


def neg_op(a: torch.Tensor, run_mode: str = "npu", dynamic: bool = False) -> torch.Tensor:
    out = torch.zeros_like(a)

    if dynamic:
        a_pto = pypto.from_torch(a, dynamic_axis=[0])
        out_pto = pypto.from_torch(out, dynamic_axis=[0])
    else:
        a_pto = pypto.from_torch(a)
        out_pto = pypto.from_torch(out)

    if run_mode == "npu":
        neg_kernel(a_pto, out_pto)
    else:
        neg_kernel_sim(a_pto, out_pto)

    return out


def test_neg_basic(device_id: int = None, run_mode: str = "npu"):
    """Test basic usage of neg function"""
    print("=" * 60)
    print("Test: Basic Usage of neg Function")
    print("=" * 60)
    
    device = f'npu:{device_id}' if (run_mode == "npu" and device_id is not None) else 'cpu'
    
    dtype = torch.float32
    a = torch.tensor([[1, 4],
                     [16, 9]], dtype=dtype, device=device)
    expected = torch.tensor([[-1, -4],
                             [-16, -9]], dtype=dtype, device=device)

    out = neg_op(a, run_mode)
    if run_mode == "npu":
        assert_allclose(out.cpu().numpy(), expected.cpu().numpy(), rtol=1e-3, atol=1e-3)
    print(f"Output: {out}")
    print(f"Expected: {expected}")
    print("✓ Basic usage of neg function completed successfully")


# ============================================================================
# POW Examples
# ============================================================================

@pypto.jit
def pow_kernel(a: pypto.Tensor, out: pypto.Tensor, b: float) -> None:
    pypto.set_vec_tile_shapes(2, 8)
    out[:] = pypto.pow(a, b)


@pypto.jit(runtime_options={"run_mode": pypto.RunMode.SIM})
def pow_kernel_sim(a: pypto.Tensor, out: pypto.Tensor, b: float) -> None:
    pypto.set_vec_tile_shapes(2, 8)
    out[:] = pypto.pow(a, b)


def pow_op(a: torch.Tensor, b: float, run_mode: str = "npu", dynamic: bool = False) -> torch.Tensor:
    out = torch.zeros_like(a)

    if dynamic:
        a_pto = pypto.from_torch(a, dynamic_axis=[0])
        out_pto = pypto.from_torch(out, dynamic_axis=[0])
    else:
        a_pto = pypto.from_torch(a)
        out_pto = pypto.from_torch(out)

    if run_mode == "npu":
        pow_kernel(a_pto, out_pto, b)
    else:
        pow_kernel_sim(a_pto, out_pto, b)

    return out


def test_pow_basic(device_id: int = None, run_mode: str = "npu"):
    """Test basic usage of pow function"""
    print("=" * 60)
    print("Test: Basic Usage of pow Function")
    print("=" * 60)
    
    device = f'npu:{device_id}' if (run_mode == "npu" and device_id is not None) else 'cpu'
    
    dtype = torch.float32
    a = torch.tensor([3, 3], dtype=dtype, device=device)
    b = 2.0
    expected = torch.tensor([9, 9], dtype=dtype, device=device)

    out = pow_op(a, b, run_mode)
    if run_mode == "npu":
        assert_allclose(out.cpu().numpy(), expected.cpu().numpy(), rtol=1e-3, atol=1e-3)
    print(f"Output: {out}")
    print(f"Expected: {expected}")
    print("✓ Basic usage of pow function completed successfully")


# ============================================================================
# RSQRT Examples
# ============================================================================

@pypto.jit
def rsqrt_kernel(a: pypto.Tensor, out: pypto.Tensor) -> None:
    pypto.set_vec_tile_shapes(2, 8)
    out[:] = pypto.rsqrt(a)


@pypto.jit(runtime_options={"run_mode": pypto.RunMode.SIM})
def rsqrt_kernel_sim(a: pypto.Tensor, out: pypto.Tensor) -> None:
    pypto.set_vec_tile_shapes(2, 8)
    out[:] = pypto.rsqrt(a)


def rsqrt_op(a: torch.Tensor, run_mode: str = "npu", dynamic: bool = False) -> torch.Tensor:
    out = torch.zeros_like(a)

    if dynamic:
        a_pto = pypto.from_torch(a, dynamic_axis=[0])
        out_pto = pypto.from_torch(out, dynamic_axis=[0])
    else:
        a_pto = pypto.from_torch(a)
        out_pto = pypto.from_torch(out)

    if run_mode == "npu":
        rsqrt_kernel(a_pto, out_pto)
    else:
        rsqrt_kernel_sim(a_pto, out_pto)

    return out


def test_rsqrt_basic(device_id: int = None, run_mode: str = "npu"):
    """Test basic usage of rsqrt function"""
    print("=" * 60)
    print("Test: Basic Usage of rsqrt Function")
    print("=" * 60)
    
    device = f'npu:{device_id}' if (run_mode == "npu" and device_id is not None) else 'cpu'
    
    dtype = torch.float32
    a = torch.tensor([[1, 4],
                     [16, 9]], dtype=dtype, device=device)
    expected = torch.tensor([[1, 0.5],
                             [0.25, 0.333333]], dtype=dtype, device=device)

    out = rsqrt_op(a, run_mode)
    if run_mode == "npu":
        assert_allclose(out.cpu().numpy(), expected.cpu().numpy(), rtol=1e-3, atol=1e-3)
    print(f"Output: {out}")
    print(f"Expected: {expected}")
    print("✓ Basic usage of rsqrt function completed successfully")


# ============================================================================
# SQRT Examples
# ============================================================================

@pypto.jit
def sqrt_kernel(a: pypto.Tensor, out: pypto.Tensor) -> None:
    pypto.set_vec_tile_shapes(2, 8)
    out[:] = pypto.sqrt(a)


@pypto.jit(runtime_options={"run_mode": pypto.RunMode.SIM})
def sqrt_kernel_sim(a: pypto.Tensor, out: pypto.Tensor) -> None:
    pypto.set_vec_tile_shapes(2, 8)
    out[:] = pypto.sqrt(a)


def sqrt_op(a: torch.Tensor, run_mode: str = "npu", dynamic: bool = False) -> torch.Tensor:
    out = torch.zeros_like(a)

    if dynamic:
        a_pto = pypto.from_torch(a, dynamic_axis=[0])
        out_pto = pypto.from_torch(out, dynamic_axis=[0])
    else:
        a_pto = pypto.from_torch(a)
        out_pto = pypto.from_torch(out)

    if run_mode == "npu":
        sqrt_kernel(a_pto, out_pto)
    else:
        sqrt_kernel_sim(a_pto, out_pto)

    return out


def test_sqrt_basic(device_id: int = None, run_mode: str = "npu"):
    """Test basic usage of sqrt function"""
    print("=" * 60)
    print("Test: Basic Usage of sqrt Function")
    print("=" * 60)
    
    device = f'npu:{device_id}' if (run_mode == "npu" and device_id is not None) else 'cpu'
    
    dtype = torch.float32
    a = torch.tensor([[1, 4],
                     [16, 9]], dtype=dtype, device=device)
    expected = torch.tensor([[1, 2],
                             [4, 3]], dtype=dtype, device=device)

    out = sqrt_op(a, run_mode)
    if run_mode == "npu":
        assert_allclose(out.cpu().numpy(), expected.cpu().numpy(), rtol=1e-3, atol=1e-3)
    print(f"Output: {out}")
    print(f"Expected: {expected}")
    print("✓ Basic usage of sqrt function completed successfully")


# ============================================================================
# SUB Examples
# ============================================================================

@pypto.jit
def sub_kernel(a: pypto.Tensor, b: pypto.Tensor, out: pypto.Tensor) -> None:
    pypto.set_vec_tile_shapes(2, 8)
    out[:] = pypto.sub(a, b)


@pypto.jit(runtime_options={"run_mode": pypto.RunMode.SIM})
def sub_kernel_sim(a: pypto.Tensor, b: pypto.Tensor, out: pypto.Tensor) -> None:
    pypto.set_vec_tile_shapes(2, 8)
    out[:] = pypto.sub(a, b)


def sub_op(a: torch.Tensor, b: torch.Tensor, run_mode: str = "npu", dynamic: bool = False) -> torch.Tensor:
    out = torch.zeros_like(a)

    if dynamic:
        a_pto = pypto.from_torch(a, dynamic_axis=[0])
        b_pto = pypto.from_torch(b, dynamic_axis=[0])
        out_pto = pypto.from_torch(out, dynamic_axis=[0])
    else:
        a_pto = pypto.from_torch(a)
        b_pto = pypto.from_torch(b)
        out_pto = pypto.from_torch(out)

    if run_mode == "npu":
        sub_kernel(a_pto, b_pto, out_pto)
    else:
        sub_kernel_sim(a_pto, b_pto, out_pto)

    return out


def test_sub_basic(device_id: int = None, run_mode: str = "npu"):
    """Test basic usage of sub function"""
    print("=" * 60)
    print("Test: Basic Usage of sub Function")
    print("=" * 60)
    
    device = f'npu:{device_id}' if (run_mode == "npu" and device_id is not None) else 'cpu'
    
    dtype = torch.float32
    a = torch.tensor([4, 5, 6], dtype=dtype, device=device)
    b = torch.tensor([1, 2, 3], dtype=dtype, device=device)
    expected = torch.tensor([3, 3, 3], dtype=dtype, device=device)

    out = sub_op(a, b, run_mode)
    if run_mode == "npu":
        assert_allclose(out.cpu().numpy(), expected.cpu().numpy(), rtol=1e-3, atol=1e-3)
    print(f"Output: {out}")
    print(f"Expected: {expected}")
    print("✓ Basic usage of sub function completed successfully")


@pypto.jit
def sub_broadcast_kernel(a: pypto.Tensor, b: pypto.Tensor, out: pypto.Tensor) -> None:
    pypto.set_vec_tile_shapes(2, 8)
    out[:] = pypto.sub(a, b)


@pypto.jit(runtime_options={"run_mode": pypto.RunMode.SIM})
def sub_broadcast_kernel_sim(a: pypto.Tensor, b: pypto.Tensor, out: pypto.Tensor) -> None:
    pypto.set_vec_tile_shapes(2, 8)
    out[:] = pypto.sub(a, b)


def sub_broadcast_op(a: torch.Tensor, b: torch.Tensor, run_mode: str = "npu", dynamic: bool = False) -> torch.Tensor:
    out = torch.zeros_like(a)

    if dynamic:
        a_pto = pypto.from_torch(a, dynamic_axis=[0])
        b_pto = pypto.from_torch(b, dynamic_axis=[0])
        out_pto = pypto.from_torch(out, dynamic_axis=[0])
    else:
        a_pto = pypto.from_torch(a)
        b_pto = pypto.from_torch(b)
        out_pto = pypto.from_torch(out)

    if run_mode == "npu":
        sub_broadcast_kernel(a_pto, b_pto, out_pto)
    else:
        sub_broadcast_kernel_sim(a_pto, b_pto, out_pto)

    return out


def test_sub_broadcast(device_id: int = None, run_mode: str = "npu"):
    """Test broadcasting between tensors of different shapes"""
    print("=" * 60)
    print("Test: Broadcasting Between Tensors")
    print("=" * 60)
    
    device = f'npu:{device_id}' if (run_mode == "npu" and device_id is not None) else 'cpu'
    
    dtype = torch.float32
    a = torch.tensor([[1, 2], [3, 4]], dtype=dtype, device=device)
    b = torch.tensor([1, 2], dtype=dtype, device=device)
    expected = torch.tensor([[0, 0], [2, 2]], dtype=dtype, device=device)

    out = sub_broadcast_op(a, b, run_mode)
    if run_mode == "npu":
        assert_allclose(out.cpu().numpy(), expected.cpu().numpy(), rtol=1e-3, atol=1e-3)
    print(f"Output: {out}")
    print(f"Expected: {expected}")
    print("✓ Test Broadcasting Between Tensors completed successfully")


@pypto.jit
def sub_scalar_kernel(a: pypto.Tensor, out: pypto.Tensor, scalar: float) -> None:
    pypto.set_vec_tile_shapes(2, 8)
    out[:] = pypto.sub(a, scalar)


@pypto.jit(runtime_options={"run_mode": pypto.RunMode.SIM})
def sub_scalar_kernel_sim(a: pypto.Tensor, out: pypto.Tensor, scalar: float) -> None:
    pypto.set_vec_tile_shapes(2, 8)
    out[:] = pypto.sub(a, scalar)


def sub_scalar_op(a: torch.Tensor, scalar: float, run_mode: str = "npu", dynamic: bool = False) -> torch.Tensor:
    out = torch.zeros_like(a)

    if dynamic:
        a_pto = pypto.from_torch(a, dynamic_axis=[0])
        out_pto = pypto.from_torch(out, dynamic_axis=[0])
    else:
        a_pto = pypto.from_torch(a)
        out_pto = pypto.from_torch(out)

    if run_mode == "npu":
        sub_scalar_kernel(a_pto, out_pto, scalar)
    else:
        sub_scalar_kernel_sim(a_pto, out_pto, scalar)

    return out


def test_sub_scalar(device_id: int = None, run_mode: str = "npu"):
    """Test subing a scalar to a tensor"""
    print("=" * 60)
    print("Test: Subing a scalar to a tensor")
    print("=" * 60)
    
    device = f'npu:{device_id}' if (run_mode == "npu" and device_id is not None) else 'cpu'
    
    dtype = torch.float32
    a = torch.tensor([1, 2, 3], dtype=dtype, device=device)
    scalar = 2.0
    expected = torch.tensor([-1, 0, 1], dtype=dtype, device=device)

    out = sub_scalar_op(a, scalar, run_mode)
    if run_mode == "npu":
        assert_allclose(out.cpu().numpy(), expected.cpu().numpy(), rtol=1e-3, atol=1e-3)
    print(f"Output: {out}")
    print(f"Expected: {expected}")
    print("✓ Test Subing a scalar to a tensor completed successfully")


@pypto.jit
def sub_with_alpha_kernel(a: pypto.Tensor, b: pypto.Tensor, out: pypto.Tensor, alpha: float) -> None:
    pypto.set_vec_tile_shapes(2, 8)
    out[:] = pypto.sub(a, b, alpha=alpha)


@pypto.jit(runtime_options={"run_mode": pypto.RunMode.SIM})
def sub_with_alpha_kernel_sim(a: pypto.Tensor, b: pypto.Tensor, out: pypto.Tensor, alpha: float) -> None:
    pypto.set_vec_tile_shapes(2, 8)
    out[:] = pypto.sub(a, b, alpha=alpha)


def sub_with_alpha_op(a: torch.Tensor, b: torch.Tensor, alpha: float, run_mode: str = "npu", dynamic: bool = False) -> torch.Tensor:
    out = torch.zeros_like(a)

    if dynamic:
        a_pto = pypto.from_torch(a, dynamic_axis=[0])
        b_pto = pypto.from_torch(b, dynamic_axis=[0])
        out_pto = pypto.from_torch(out, dynamic_axis=[0])
    else:
        a_pto = pypto.from_torch(a)
        b_pto = pypto.from_torch(b)
        out_pto = pypto.from_torch(out)

    if run_mode == "npu":
        sub_with_alpha_kernel(a_pto, b_pto, out_pto, alpha)
    else:
        sub_with_alpha_kernel_sim(a_pto, b_pto, out_pto, alpha)

    return out


def test_sub_with_alpha(device_id: int = None, run_mode: str = "npu"):
    """Using the alpha parameter to scale the second input"""
    print("=" * 60)
    print("Test: Using the Alpha Parameter")
    print("=" * 60)
    
    device = f'npu:{device_id}' if (run_mode == "npu" and device_id is not None) else 'cpu'
    
    dtype = torch.float32
    a = torch.tensor([9, 8, 7], dtype=dtype, device=device)
    b = torch.tensor([1, 2, 3], dtype=dtype, device=device)
    alpha = 2.0
    expected = torch.tensor([7, 4, 1], dtype=dtype, device=device)

    out = sub_with_alpha_op(a, b, alpha, run_mode)
    if run_mode == "npu":
        assert_allclose(out.cpu().numpy(), expected.cpu().numpy(), rtol=1e-3, atol=1e-3)
    print(f"Output: {out}")
    print(f"Expected: {expected}")
    print("✓ Using the alpha parameter to scale the second input completed successfully")


# ============================================================================
# Main Function
# ============================================================================

def main():
    """Run element-wise examples.
    
    Usage:
        python elementwise.py              # Run all examples
        python elementwise.py --list       # List all available examples
        python elementwise.py abs::test_abs_basic    # Run a specific case
    """
    parser = argparse.ArgumentParser(
        description="PyPTO Element-wise Operation Examples",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                      Run all examples
  %(prog)s --list               List all available examples
  %(prog)s abs::test_abs_basic    Run a specific case
        """
    )
    parser.add_argument(
        'example_id',
        type=str,
        nargs="?",
        help='Run a specific case (e.g., abs::test_abs_basic). If omitted, all cases run.'
    )
    parser.add_argument(
        '--list',
        action='store_true',
        help='List all available examples and exit'
    )
    parser.add_argument(
        "--run_mode", "--run-mode",
        nargs="?", type=str, default="npu", choices=["npu", "sim"],
        help="run mode, such as npu/sim etc."
    )
    
    args = parser.parse_args()
    
    # Define available examples
    examples = {
        'abs::test_abs_basic': {
            'name': 'Test basic usage of abs function',
            'description': 'Basic usage of abs function example',
            'function': test_abs_basic
        },
        'add::test_add_basic': {
            'name': 'Test basic usage of add function',
            'description': 'Basic usage of add function example',
            'function': test_add_basic
        },
        'add::test_add_broadcast': {
            'name': 'Test broadcasting between tensors of different shapes',
            'description': 'Broadcasting between tensors example',
            'function': test_add_broadcast
        },
        'add::test_add_scalar': {
            'name': 'Test adding a scalar to a tensor',
            'description': 'Adding a scalar to a tensor example',
            'function': test_add_scalar
        },
        'add::test_add_with_alpha': {
            'name': 'Using the alpha parameter to scale the second input',
            'description': 'Using the alpha parameter example',
            'function': test_add_with_alpha
        },
        'clip::test_clip_basic': {
            'name': 'Test basic usage of clip function',
            'description': 'Basic usage of clip function example',
            'function': test_clip_basic
        },
        'clip::test_clip_broadcast': {
            'name': 'Test broadcasting between tensors of different shapes',
            'description': 'Broadcasting between tensors example',
            'function': test_clip_broadcast
        },
        'div::test_div_basic': {
            'name': 'Test basic usage of div function',
            'description': 'Basic usage of div function example',
            'function': test_div_basic
        },
        'div::test_div_broadcast': {
            'name': 'Test broadcasting between tensors of different shapes',
            'description': 'Broadcasting between tensors example',
            'function': test_div_broadcast
        },
        'div::test_div_scalar': {
            'name': 'Test diving a scalar to a tensor',
            'description': 'Diving a scalar to a tensor example',
            'function': test_div_scalar
        },
        'exp::test_exp_basic': {
            'name': 'Test basic usage of exp function',
            'description': 'Basic usage of exp function example',
            'function': test_exp_basic
        },
        'log::test_log_basic': {
            'name': 'Test basic usage of log function',
            'description': 'Basic usage of log function example',
            'function': test_log_basic
        },
        'mul::test_mul_basic': {
            'name': 'Test basic usage of mul function',
            'description': 'Basic usage of mul function example',
            'function': test_mul_basic
        },
        'mul::test_mul_broadcast': {
            'name': 'Test broadcasting between tensors of different shapes',
            'description': 'Broadcasting between tensors example',
            'function': test_mul_broadcast
        },
        'mul::test_mul_scalar': {
            'name': 'Test muling a scalar to a tensor',
            'description': 'Muling a scalar to a tensor example',
            'function': test_mul_scalar
        },
        'neg::test_neg_basic': {
            'name': 'Test basic usage of neg function',
            'description': 'Basic usage of neg function example',
            'function': test_neg_basic
        },
        'pow::test_pow_basic': {
            'name': 'Test basic usage of pow function',
            'description': 'Basic usage of pow function example',
            'function': test_pow_basic
        },
        'rsqrt::test_rsqrt_basic': {
            'name': 'Test basic usage of rsqrt function',
            'description': 'Basic usage of rsqrt function example',
            'function': test_rsqrt_basic
        },
        'sqrt::test_sqrt_basic': {
            'name': 'Test basic usage of sqrt function',
            'description': 'Basic usage of sqrt function example',
            'function': test_sqrt_basic
        },
        'sub::test_sub_basic': {
            'name': 'Test basic usage of sub function',
            'description': 'Basic usage of sub function example',
            'function': test_sub_basic
        },
        'sub::test_sub_broadcast': {
            'name': 'Test broadcasting between tensors of different shapes',
            'description': 'Broadcasting between tensors example',
            'function': test_sub_broadcast
        },
        'sub::test_sub_scalar': {
            'name': 'Test subing a scalar to a tensor',
            'description': 'Subing a scalar to a tensor example',
            'function': test_sub_scalar
        },
        'sub::test_sub_with_alpha': {
            'name': 'Using the alpha parameter to scale the second input',
            'description': 'Using the alpha parameter example',
            'function': test_sub_with_alpha
        }
    }
    
    # List examples if requested
    if args.list:
        print("\n" + "=" * 60)
        print("Available Examples")
        print("=" * 60 + "\n")
        for case_key, ex_info in sorted(examples.items()):
            print(f"  {case_key}")
            print(f"     {ex_info['name']}")
            print(f"     {ex_info['description']}\n")
        return
    
    # Validate case if provided
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
    print("PyPTO Element-wise Operation Examples")
    print("=" * 60 + "\n")

    device_id = None
    if args.run_mode == "npu":
        device_id = get_device_id()
        if device_id is None:
            return
        import torch_npu
        torch.npu.set_device(device_id)
    
    try:
        for case_key, ex_info in examples_to_run:
            if args.run_mode == "npu" and device_id is None:
                print(f"Skipping {case_key} ({ex_info['name']}): NPU device not configured")
                continue
            
            ex_info['function'](device_id, args.run_mode)
        
        if len(examples_to_run) > 1:
            print("=" * 60)
            print("All element-wise tests passed!")
            print("=" * 60)
        
    except Exception as e:
        print(f"\nError: {e}")
        raise


if __name__ == "__main__":
    main()
