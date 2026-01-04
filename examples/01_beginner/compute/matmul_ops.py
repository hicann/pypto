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
Matrix Multiplication (matmul) Operation Examples for PyPTO

This file contains all matrix multiplication examples merged into a single file.
You can run all examples or select specific ones using command-line arguments.

Usage:
    python matmul_ops.py              # Run all examples
    python matmul_ops.py --list       # List all available examples
    python matmul_ops.py matmul::test_matmul_basic    # Run a specific case
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
# MATMUL Examples
# ============================================================================


def matmul_op(a: torch.Tensor, b: torch.Tensor, run_mode: str = "npu") -> torch.Tensor:
    a_shape, b_shape = a.shape, b.shape
    out_shape = (a_shape[0], b_shape[1])
    
    if run_mode == "npu":
        mode = pypto.RunMode.NPU
    elif run_mode == "sim":
        mode = pypto.RunMode.SIM
    else:
        raise ValueError(f"Invalid run_mode: {run_mode}. Must be 'npu' or 'sim'")
    
    @pypto.frontend.jit(
    host_options={"only_codegen": True}, runtime_options={"run_mode": mode}
    )
    def matmul_kernel(
        a: pypto.Tensor(a_shape, pypto.DT_FP32),
        b: pypto.Tensor(b_shape, pypto.DT_FP32),
    ) -> pypto.Tensor(out_shape, pypto.DT_FP32):
        pypto.set_cube_tile_shapes([32, 32], [64, 64], [64, 64])
        out = pypto.matmul(a, b, a.dtype)
        return out

    out = matmul_kernel(a, b)
    return out


def test_matmul_basic(device_id: int = None, run_mode: str = "npu"):
    """Test basic matrix multiplication"""
    print("=" * 60)
    print("Test: Basic Matrix Multiplication")
    print("=" * 60)
    
    device = f'npu:{device_id}' if (run_mode == "npu" and device_id is not None) else 'cpu'
    
    dtype = torch.float32
    a = torch.tensor([[1, 2], [3, 4]], dtype=dtype, device=device)
    b = torch.tensor([[5, 6], [7, 8]], dtype=dtype, device=device)
    expected = torch.tensor([[19, 22], [43, 50]], dtype=dtype, device=device)

    out = matmul_op(a, b, run_mode)
    if run_mode == "npu":
        assert_allclose(out.cpu().numpy(), expected.cpu().numpy(), rtol=1e-3, atol=1e-3)
    print(f"Output: {out}")
    print(f"Expected: {expected}")
    print("✓ Basic matrix multiplication completed successfully")


def matmul_batch_op(a: torch.Tensor, b: torch.Tensor, run_mode: str = "npu", dynamic: bool = False) -> torch.Tensor:
    a_shape, b_shape = a.shape, b.shape
    if run_mode == "npu":
        mode = pypto.RunMode.NPU
    elif run_mode == "sim":
        mode = pypto.RunMode.SIM
    else:
        raise ValueError(f"Invalid run_mode: {run_mode}. Must be 'npu' or 'sim'")
        
    @pypto.frontend.jit(
    host_options={"only_codegen": True}, runtime_options={"run_mode": mode}
    )
    def matmul_batch_kernel(
        a: pypto.Tensor(a_shape, pypto.DT_FP32),
        b: pypto.Tensor(b_shape, pypto.DT_FP32),
    ) -> pypto.Tensor(a_shape, pypto.DT_FP32):
        pypto.set_cube_tile_shapes([32, 32], [64, 64], [64, 64])
        out = pypto.matmul(a, b, a.dtype)
        return out
    out = matmul_batch_kernel(a, b)
    return out


def test_matmul_batch(device_id: int = None, run_mode: str = "npu"):
    """Test batch matrix multiplication"""
    print("=" * 60)
    print("Test: Batch Matrix Multiplication")
    print("=" * 60)
    
    device = f'npu:{device_id}' if (run_mode == "npu" and device_id is not None) else 'cpu'
    
    dtype = torch.float32
    a = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=dtype, device=device)
    b = torch.tensor([[[5, 6], [7, 8]], [[1, 2], [3, 4]]], dtype=dtype, device=device)
    expected = torch.tensor([[[19, 22], [43, 50]], [[23, 34], [31, 46]]], dtype=dtype, device=device)

    out = matmul_batch_op(a, b, run_mode)
    if run_mode == "npu":
        assert_allclose(out.cpu().numpy(), expected.cpu().numpy(), rtol=1e-3, atol=1e-3)
    print(f"Output: {out}")
    print(f"Expected: {expected}")
    print("✓ Batch matrix multiplication completed successfully")



def matmul_broadcast_op(a: torch.Tensor, b: torch.Tensor, run_mode: str = "npu", dynamic: bool = False) -> torch.Tensor:
    a_shape, b_shape = a.shape, b.shape

    if run_mode == "npu":
        mode = pypto.RunMode.NPU
    elif run_mode == "sim":
        mode = pypto.RunMode.SIM
    else:
        raise ValueError(f"Invalid run_mode: {run_mode}. Must be 'npu' or 'sim'")
    
    @pypto.frontend.jit(
    host_options={"only_codegen": True}, runtime_options={"run_mode": mode}
    )
    def matmul_broadcast_kernel(
        a: pypto.Tensor(a.shape, pypto.DT_FP32),
        b: pypto.Tensor(b.shape, pypto.DT_FP32),
    ) -> pypto.Tensor(b.shape, pypto.DT_FP32):
        pypto.set_cube_tile_shapes([32, 32], [64, 64], [64, 64])
        out = pypto.matmul(a, b, pypto.DT_FP32)
        return out
    out = matmul_broadcast_kernel(a, b)
    return out


def test_matmul_broadcast(device_id: int = None, run_mode: str = "npu"):
    """Test batch matrix multiplication with broadcasting"""
    print("=" * 60)
    print("Test: Batch Matrix Multiplication with Broadcasting")
    print("=" * 60)
    
    device = f'npu:{device_id}' if (run_mode == "npu" and device_id is not None) else 'cpu'
    
    dtype = torch.float32
    a = torch.tensor([[[1, 2], [3, 4]]], dtype=dtype, device=device)
    b = torch.tensor([[[5, 6], [7, 8]], [[1, 2], [3, 4]]], dtype=dtype, device=device)
    expected = torch.tensor([[[19, 22], [43, 50]], [[7, 10], [15, 22]]], dtype=dtype, device=device)

    out = matmul_broadcast_op(a, b, run_mode)
    if run_mode == "npu":
        assert_allclose(out.cpu().numpy(), expected.cpu().numpy(), rtol=1e-3, atol=1e-3)
    print(f"Output: {out}")
    print(f"Expected: {expected}")
    print("✓ Batch matrix multiplication with broadcasting completed successfully")


def matmul_trans_right_op(a: torch.Tensor, b: torch.Tensor, run_mode: str = "npu", dynamic: bool = False) -> torch.Tensor:
    a_shape, b_shape = a.shape, b.shape
    out_shape = (a_shape[0], b_shape[0])
    if run_mode == "npu":
        mode = pypto.RunMode.NPU
    elif run_mode == "sim":
        mode = pypto.RunMode.SIM
    else:
        raise ValueError(f"Invalid run_mode: {run_mode}. Must be 'npu' or 'sim'")

    @pypto.frontend.jit(
    host_options={"only_codegen": True}, runtime_options={"run_mode": mode}
    )
    def matmul_trans_right_kernel(
        a: pypto.Tensor(a_shape, pypto.DT_FP32),
        b: pypto.Tensor(b_shape, pypto.DT_FP32),
    ) -> (pypto.Tensor(out_shape, pypto.DT_FP32),):
        pypto.set_cube_tile_shapes([32, 32], [64, 64], [64, 64])
        out = pypto.matmul(a, b, a.dtype, b_trans=True)
        return out

    out = matmul_trans_right_kernel(a, b)
    return out


def matmul_trans_left_op(a: torch.Tensor, b: torch.Tensor, run_mode: str = "npu", dynamic: bool = False) -> torch.Tensor:
    a_shape, b_shape = a.shape, b.shape
    out_shape = (a_shape[1], b_shape[1])
    if run_mode == "npu":
        mode = pypto.RunMode.NPU
    elif run_mode == "sim":
        mode = pypto.RunMode.SIM
    else:
        raise ValueError(f"Invalid run_mode: {run_mode}. Must be 'npu' or 'sim'")

    @pypto.frontend.jit(
    host_options={"only_codegen": True}, runtime_options={"run_mode": mode}
    )
    def matmul_trans_left_kernel(
        a: pypto.Tensor(a_shape, pypto.DT_FP32),
        b: pypto.Tensor(b_shape, pypto.DT_FP32),
    ) -> pypto.Tensor(out_shape, pypto.DT_FP32):
        pypto.set_cube_tile_shapes([32, 32], [64, 64], [64, 64])
        out = pypto.matmul(a, b, a.dtype, a_trans=True)
        return out
    
    out = matmul_trans_left_kernel(a, b)

    return out


def test_matmul_trans(device_id: int = None, run_mode: str = "npu"):
    """Test matrix multiplication with transposition"""
    print("=" * 60)
    print("Test: Matrix Multiplication with Transposition")
    print("=" * 60)
    
    device = f'npu:{device_id}' if (run_mode == "npu" and device_id is not None) else 'cpu'
    
    # Test 1: Basic matrix multiplication
    dtype = torch.float32
    a = torch.tensor([[1, 2, 3], 
                      [4, 5, 6]], dtype=dtype, device=device)
    b = torch.tensor([[7, 8],
                      [9, 10],
                      [11, 12]], dtype=dtype, device=device)
    expected = torch.tensor([[58, 64],
                            [139, 154]], dtype=dtype, device=device)

    out = matmul_op(a, b, run_mode)
    if run_mode == "npu":
        assert_allclose(out.cpu().numpy(), expected.cpu().numpy(), rtol=1e-3, atol=1e-3)
    print(f"Output basic: {out}")
    print(f"Expected basic: {expected}")
    
    # Test 2: Matrix multiplication with the right matrix transposed
    dtype = torch.float32
    a = torch.tensor([[1, 2, 3], 
                      [4, 5, 6]], dtype=dtype, device=device)
    b = torch.tensor([[7, 9, 11], 
                      [8, 10, 12]], dtype=dtype, device=device)
    expected = torch.tensor([[58, 64],
                            [139, 154]], dtype=dtype, device=device)

    out = matmul_trans_right_op(a, b, run_mode)
    if run_mode == "npu":
        assert_allclose(out.cpu().numpy(), expected.cpu().numpy(), rtol=1e-3, atol=1e-3)
    print(f"Output trans right: {out}")
    print(f"Expected trans right: {expected}")
    
    # Test 3: Matrix multiplication with the left matrix transposed
    dtype = torch.float32
    a = torch.tensor([[1, 4],
                      [2, 5],
                      [3, 6]], dtype=dtype, device=device)
    b = torch.tensor([[7, 8],
                      [9, 10],
                      [11, 12]], dtype=dtype, device=device)
    expected = torch.tensor([[58, 64],
                            [139, 154]], dtype=dtype, device=device)

    out = matmul_trans_left_op(a, b, run_mode)
    if run_mode == "npu":
        assert_allclose(out.cpu().numpy(), expected.cpu().numpy(), rtol=1e-3, atol=1e-3)
    print(f"Output trans left: {out}")
    print(f"Expected trans left: {expected}")
    
    print("✓ Matrix multiplication with transposition completed successfully")



def matmul_bias_op(a: torch.Tensor, b: torch.Tensor, bias: torch.Tensor, run_mode: str = "npu", dynamic: bool = False) -> torch.Tensor:
    a_shape, b_shape, bias_shape = a.shape, b.shape, bias.shape

    if run_mode == "npu":
        mode = pypto.RunMode.NPU
    elif run_mode == "sim":
        mode = pypto.RunMode.SIM
    else:
        raise ValueError(f"Invalid run_mode: {run_mode}. Must be 'npu' or 'sim'")
    
    @pypto.frontend.jit(
        host_options={"only_codegen": True}, runtime_options={"run_mode": mode}
    )
    def matmul_bias_kernel(
        a: pypto.Tensor(a_shape, pypto.DT_FP32),
        b: pypto.Tensor(b_shape, pypto.DT_FP32),
        bias: pypto.Tensor(bias_shape, pypto.DT_FP32),
    ) -> pypto.Tensor(b_shape, pypto.DT_FP32):
        
        extend_params = {"bias_tensor": bias}
        pypto.set_cube_tile_shapes([32, 32], [64, 64], [64, 64])
        out = pypto.matmul(a, b, a.dtype, extend_params=extend_params)
        return out

    out = matmul_bias_kernel(a, b, bias)
    return out


def test_matmul_bias(device_id: int = None, run_mode: str = "npu"):
    """Test matrix multiplication with bias"""
    print("=" * 60)
    print("Test: Matrix Multiplication with Bias")
    print("=" * 60)

    device = f'npu:{device_id}' if (run_mode == "npu" and device_id is not None) else 'cpu'

    dtype = torch.float32
    a = torch.tensor([[1, 2], [3, 4]], dtype=dtype, device=device)
    b = torch.tensor([[5, 6], [7, 8]], dtype=dtype, device=device)
    bias = torch.tensor([[1, 2]], dtype=dtype, device=device)
    expected = torch.tensor([[20, 24], [44, 52]], dtype=dtype, device=device)

    out = matmul_bias_op(a, b, bias, run_mode)
    if run_mode == "npu":
        assert_allclose(out.cpu().numpy(), expected.cpu().numpy(), rtol=1e-3, atol=1e-3)
    print(f"Output: {out}")
    print(f"Expected: {expected}")
    print("✓ Matrix multiplication with bias completed successfully")


# ============================================================================
# Main Function
# ============================================================================

def main():
    """Run matrix multiplication examples.
    
    Usage:
        python matmul_ops.py              # Run all examples
        python matmul_ops.py --list       # List all available examples
        python matmul_ops.py matmul::test_matmul_basic    # Run a specific case
    """
    parser = argparse.ArgumentParser(
        description="PyPTO Matrix Multiplication (matmul) Operation Examples",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                      Run all examples
  %(prog)s --list               List all available examples
  %(prog)s matmul::test_matmul_basic    Run a specific case
        """
    )
    parser.add_argument(
        'example_id',
        type=str,
        nargs="?",
        help='Run a specific case (e.g., matmul::test_matmul_basic). If omitted, all cases run.'
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
        'matmul::test_matmul_basic': {
            'name': 'Test basic matrix multiplication',
            'description': 'Basic matrix multiplication example',
            'function': test_matmul_basic
        },
        'matmul::test_matmul_batch': {
            'name': 'Test batch matrix multiplication',
            'description': 'Batch matrix multiplication example',
            'function': test_matmul_batch
        },
        'matmul::test_matmul_broadcast': {
            'name': 'Test batch matrix multiplication with broadcasting',
            'description': 'Batch matrix multiplication with broadcasting example',
            'function': test_matmul_broadcast
        },
        'matmul::test_matmul_trans': {
            'name': 'Test matrix multiplication with transposition',
            'description': 'Matrix multiplication with transposition example',
            'function': test_matmul_trans
        },
        'matmul::test_matmul_bias': {
            'name': 'Test matrix multiplication with bias',
            'description': 'Matrix multiplication with bias example',
            'function': test_matmul_bias
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
    print("PyPTO Matrix Multiplication (matmul) Operation Examples")
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
            print("All matmul tests passed!")
            print("=" * 60)
        
    except Exception as e:
        print(f"\nError: {e}")
        raise


if __name__ == "__main__":
    main()
