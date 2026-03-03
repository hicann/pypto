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
Basic Operations Quick-Start for PyPTO

A concise overview of core PyPTO capabilities. Each example demonstrates one
key category from the beginner tutorials:

1. Tensor creation and properties
2. Element-wise operations (add, mul)
3. Matrix multiplication (matmul)
4. Reduction operations (sum)
5. Tiling configuration (vec / cube tile shapes)
6. Transform operations (view + assemble)

For more detailed examples of each category, see the corresponding files:
  - basic/tensor_creation.py, basic/symbolic_scalar.py
  - compute/elementwise_ops.py, compute/matmul_ops.py, compute/reduce_ops.py
  - tiling/tiling_config.py
  - transform/transform_ops.py
"""

import os
import sys
import argparse
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


def _get_mode(run_mode: str):
    if run_mode == "npu":
        return pypto.RunMode.NPU
    elif run_mode == "sim":
        return pypto.RunMode.SIM
    raise ValueError(f"Invalid run_mode: {run_mode}. Must be 'npu' or 'sim'")


# ============================================================================
# 1. Tensor Creation
# ============================================================================

def test_tensor_creation(device_id=None, run_mode="npu"):
    """Demonstrate tensor creation and property access."""
    print("=" * 60)
    print("Example 1: Tensor Creation")
    print("=" * 60)

    tensor = pypto.Tensor([4, 4], pypto.DT_FP16, "my_tensor")
    print(f"  name={tensor.name}, shape={tensor.shape}, dtype={tensor.dtype}, "
          f"format={tensor.format}, dim={tensor.dim}")
    print("✓ Tensor creation completed successfully\n")


# ============================================================================
# 2. Element-wise Operations
# ============================================================================

def test_elementwise_ops(device_id=None, run_mode="npu"):
    """Element-wise add + scalar mul: out = (a + b) * 2."""
    print("=" * 60)
    print("Example 2: Element-wise Operations")
    print("=" * 60)

    mode = _get_mode(run_mode)
    shape = (8, 8)

    @pypto.frontend.jit(runtime_options={"run_mode": mode})
    def elementwise_kernel(
        a: pypto.Tensor(shape, pypto.DT_FP16),
        b: pypto.Tensor(shape, pypto.DT_FP16),
    ) -> pypto.Tensor(shape, pypto.DT_FP16):
        pypto.set_vec_tile_shapes(8, 8)
        out = pypto.Tensor(shape, pypto.DT_FP16)
        out = pypto.mul(pypto.add(a, b), 2.0)
        return out

    device = f'npu:{device_id}' if (run_mode == "npu" and device_id is not None) else 'cpu'
    a = torch.randn(shape, dtype=torch.float16, device=device)
    b = torch.randn(shape, dtype=torch.float16, device=device)
    out = elementwise_kernel(a, b)

    if run_mode == "npu":
        expected = (a + b) * 2.0
        max_diff = (out - expected).abs().max().item()
        print(f"  Max difference: {max_diff:.6f}")
        assert max_diff < 1e-2, "Result mismatch!"
    print("✓ Element-wise operations completed successfully\n")


# ============================================================================
# 3. Matrix Multiplication
# ============================================================================

def test_matmul(device_id=None, run_mode="npu"):
    """Basic matrix multiplication: C = A @ B."""
    print("=" * 60)
    print("Example 3: Matrix Multiplication")
    print("=" * 60)

    mode = _get_mode(run_mode)
    m, k, n = 64, 128, 64

    @pypto.frontend.jit(runtime_options={"run_mode": mode})
    def matmul_kernel(
        a: pypto.Tensor((m, k), pypto.DT_BF16),
        b: pypto.Tensor((k, n), pypto.DT_BF16),
    ) -> pypto.Tensor((m, n), pypto.DT_BF16):
        pypto.set_cube_tile_shapes([32, 32], [64, 64], [64, 64])
        out = pypto.Tensor((m, n), pypto.DT_BF16)
        out = pypto.matmul(a, b, a.dtype)
        return out

    device = f'npu:{device_id}' if (run_mode == "npu" and device_id is not None) else 'cpu'
    a = torch.randn(m, k, dtype=torch.bfloat16, device=device)
    b = torch.randn(k, n, dtype=torch.bfloat16, device=device)
    out = matmul_kernel(a, b)

    if run_mode == "npu":
        expected = torch.matmul(a, b)
        max_diff = (out - expected).abs().max().item()
        print(f"  Max difference: {max_diff:.6f}")
        assert max_diff < 1e-1, "Result mismatch!"
    print("✓ Matrix multiplication completed successfully\n")


# ============================================================================
# 4. Reduction Operations
# ============================================================================

def test_reduce_ops(device_id=None, run_mode="npu"):
    """Reduction: sum along last dimension."""
    print("=" * 60)
    print("Example 4: Reduction Operations (sum)")
    print("=" * 60)

    mode = _get_mode(run_mode)
    shape = (2, 3)

    @pypto.frontend.jit(runtime_options={"run_mode": mode})
    def sum_kernel(
        a: pypto.Tensor(shape, pypto.DT_FP32),
    ) -> pypto.Tensor((2,), pypto.DT_FP32):
        pypto.set_vec_tile_shapes(8, 8)
        out = pypto.Tensor((2,), pypto.DT_FP32)
        out = pypto.sum(a, dim=-1, keepdim=False)
        return out

    device = f'npu:{device_id}' if (run_mode == "npu" and device_id is not None) else 'cpu'
    a = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32, device=device)
    out = sum_kernel(a)

    expected = torch.tensor([6, 15], dtype=torch.float32, device=device)
    if run_mode == "npu":
        assert_allclose(out.cpu().numpy(), expected.cpu().numpy(), rtol=1e-3, atol=1e-3)
    print(f"  Input:    {a.tolist()}")
    print(f"  Output:   {out.tolist()}")
    print("✓ Reduction operations completed successfully\n")


# ============================================================================
# 5. Tiling Configuration
# ============================================================================

def test_tiling_config(device_id=None, run_mode="npu"):
    """Show how to set vec and cube tile shapes."""
    print("=" * 60)
    print("Example 5: Tiling Configuration")
    print("=" * 60)

    mode = _get_mode(run_mode)

    @pypto.frontend.jit(runtime_options={"run_mode": mode})
    def tiled_add_kernel(
        a: pypto.Tensor((2, 8), pypto.DT_FP32),
        b: pypto.Tensor((2, 8), pypto.DT_FP32),
    ) -> pypto.Tensor((2, 8), pypto.DT_FP32):
        pypto.set_vec_tile_shapes(2, 8)
        out = pypto.Tensor((2, 8), pypto.DT_FP32)
        out = pypto.add(a, b)
        return out

    device = f'npu:{device_id}' if (run_mode == "npu" and device_id is not None) else 'cpu'
    a = torch.ones((2, 8), dtype=torch.float32, device=device)
    b = torch.ones((2, 8), dtype=torch.float32, device=device)
    out = tiled_add_kernel(a, b)

    if run_mode == "npu":
        expected = a + b
        assert_allclose(out.cpu().numpy(), expected.cpu().numpy(), rtol=1e-3, atol=1e-3)
    print(f"  vec_tile_shapes set to (2, 8)")
    print("✓ Tiling configuration completed successfully\n")


# ============================================================================
# 6. Transform Operations (view + assemble)
# ============================================================================

def test_transform_ops(device_id=None, run_mode="npu"):
    """Loop-based tiling with view and assemble: out = input * 2."""
    print("=" * 60)
    print("Example 6: Transform Operations (view + assemble)")
    print("=" * 60)

    mode = _get_mode(run_mode)
    height, width = 64, 64
    tile_h, tile_w = 32, 32

    @pypto.frontend.jit(runtime_options={"run_mode": mode})
    def view_assemble_kernel(
        x: pypto.Tensor((height, width), pypto.DT_FP16),
    ) -> pypto.Tensor((height, width), pypto.DT_FP16):
        pypto.set_vec_tile_shapes(tile_h, tile_w)
        output = pypto.Tensor((height, width), pypto.DT_FP16)
        h_tiles = height // tile_h
        w_tiles = width // tile_w
        for h_idx in pypto.loop(h_tiles, name="h_loop", idx_name="h_idx"):
            for w_idx in pypto.loop(w_tiles, name="w_loop", idx_name="w_idx"):
                h_off = h_idx * tile_h
                w_off = w_idx * tile_w
                tile = pypto.view(x, [tile_h, tile_w], [h_off, w_off])
                result = pypto.mul(tile, 2.0)
                pypto.assemble(result, [h_off, w_off], output)
        return output

    device = f'npu:{device_id}' if (run_mode == "npu" and device_id is not None) else 'cpu'
    x = torch.randn((height, width), dtype=torch.float16, device=device)
    out = view_assemble_kernel(x)

    if run_mode == "npu":
        expected = x * 2.0
        max_diff = (out - expected).abs().max().item()
        print(f"  Max difference: {max_diff:.6f}")
        assert max_diff < 1e-2, "Result mismatch!"
    print("✓ Transform operations completed successfully\n")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="PyPTO Basic Operations Quick-Start",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        'example_id', type=str, nargs='?',
        help='Run a specific case. If omitted, all cases run.'
    )
    parser.add_argument('--list', action='store_true', help='List available examples')
    parser.add_argument(
        '--run_mode', type=str, nargs='?', default="npu", choices=["npu", "sim"],
        help='Run mode: npu or sim'
    )
    args = parser.parse_args()

    examples = {
        "tensor_creation::test_tensor_creation": {
            'name': 'Tensor Creation',
            'function': test_tensor_creation,
        },
        "elementwise_ops::test_elementwise_ops": {
            'name': 'Element-wise Operations',
            'function': test_elementwise_ops,
        },
        "matmul::test_matmul": {
            'name': 'Matrix Multiplication',
            'function': test_matmul,
        },
        "reduce_ops::test_reduce_ops": {
            'name': 'Reduction Operations',
            'function': test_reduce_ops,
        },
        "tiling_config::test_tiling_config": {
            'name': 'Tiling Configuration',
            'function': test_tiling_config,
        },
        "transform_ops::test_transform_ops": {
            'name': 'Transform Operations',
            'function': test_transform_ops,
        },
    }

    if args.list:
        print("\nAvailable Examples:\n")
        for ex_id, ex_info in examples.items():
            print(f"  {ex_id}: {ex_info['name']}")
        return

    if args.example_id is not None:
        if args.example_id not in examples:
            print(f"ERROR: Invalid example ID: {args.example_id}")
            print(f"Valid IDs: {', '.join(examples.keys())}")
            sys.exit(1)

    device_id = None
    if args.run_mode == "npu":
        device_id = get_device_id()
        if device_id is None:
            return
        import torch_npu
        torch.npu.set_device(device_id)

    examples_to_run = (
        [(args.example_id, examples[args.example_id])]
        if args.example_id else list(examples.items())
    )

    print("\n" + "=" * 60)
    print("PyPTO Basic Operations Quick-Start")
    print("=" * 60 + "\n")

    try:
        for _, ex_info in examples_to_run:
            ex_info['function'](device_id, args.run_mode)

        if len(examples_to_run) > 1:
            print("=" * 60)
            print("All examples completed successfully!")
            print("=" * 60)
    except Exception as e:
        print(f"\nError: {e}")
        raise


if __name__ == "__main__":
    main()
