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
Tensor Creation Operation Examples for PyPTO

This file contains all tensor creation examples merged into a single file.
You can run all examples or select specific ones using command-line arguments.

Usage:
    python creation_ops.py              # Run all examples
    python creation_ops.py --list       # List all available examples
    python creation_ops.py  arange::test_arange_basic    # Run a specific case
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
# ARANGE Examples
# ============================================================================

@pypto.jit(
    host_options={"only_codegen": True}
)
def arange_end_kernel_npu(out: pypto.Tensor, end: float) -> None:
    pypto.set_vec_tile_shapes(8)
    out[:] = pypto.arange(end)

@pypto.jit(
    host_options={"only_codegen": True},
    runtime_options={"run_mode": pypto.RunMode.SIM}
)
def arange_end_kernel_sim(out: pypto.Tensor, end: float) -> None:
    pypto.set_vec_tile_shapes(8)
    out[:] = pypto.arange(end)


@pypto.jit(
    host_options={"only_codegen": True}
)
def arange_start_end_kernel_npu(out: pypto.Tensor, start: float, end: float) -> None:
    pypto.set_vec_tile_shapes(8)
    out[:] = pypto.arange(start, end)

@pypto.jit(
    host_options={"only_codegen": True},
    runtime_options={"run_mode": pypto.RunMode.SIM}
)
def arange_start_end_kernel_sim(out: pypto.Tensor, start: float, end: float) -> None:
    pypto.set_vec_tile_shapes(8)
    out[:] = pypto.arange(start, end)


@pypto.jit(
    host_options={"only_codegen": True}
)
def arange_start_end_step_kernel_npu(out: pypto.Tensor, start: float, end: float, step: float) -> None:
    pypto.set_vec_tile_shapes(8)
    out[:] = pypto.arange(start, end, step)

@pypto.jit(
    host_options={"only_codegen": True},
    runtime_options={"run_mode": pypto.RunMode.SIM}
)
def arange_start_end_step_kernel_sim(out: pypto.Tensor, start: float, end: float, step: float) -> None:
    pypto.set_vec_tile_shapes(8)
    out[:] = pypto.arange(start, end, step)


def arange_op(shape: tuple, dtype: torch.dtype, device: str, end: float = None, run_mode: str= "npu", start: float = None, step: float = None, dynamic: bool = False) -> torch.Tensor:
    out = torch.zeros(shape, dtype=dtype, device=device)

    if dynamic:
        out_pto = pypto.from_torch(out, dynamic_axis=[0])
    else:
        out_pto = pypto.from_torch(out)

    if step is not None:
        if run_mode == "npu":
            arange_start_end_step_kernel_npu(out_pto, start, end, step)
        else:
            arange_start_end_step_kernel_sim(out_pto, start, end, step)
    elif start is not None:
        if run_mode == "npu":
            arange_start_end_kernel_npu(out_pto, start, end)
        else:
            arange_start_end_kernel_sim(out_pto, start, end)
    else:
        if run_mode == "npu":
            arange_end_kernel_npu(out_pto, end)
        else:
            arange_end_kernel_sim(out_pto, end)
    return out


def test_arange_basic(device_id = None, run_mode: str = "npu"):
    """Test basic usage of arange function"""
    print("=" * 60)
    print("Test: Basic Usage of arange Function")
    print("=" * 60)

    device = f'npu:{device_id}' if (run_mode == "npu" and device_id is not None) else 'cpu'
    
    # Test 1: arange(end)
    shape = (4,)
    dtype = torch.int32
    expected_a = torch.tensor([0, 1, 2, 3], dtype=dtype, device=device)

    out_torch = arange_op(shape, dtype, device, end=4, run_mode=run_mode)
    print(f"Output a: {out_torch}")
    print(f"Expected a: {expected_a}")
    if run_mode == "npu":
        assert_allclose(out_torch.cpu().numpy(), expected_a.cpu().numpy(), rtol=1e-3, atol=1e-3)

    # Test 2: arange(start, end)
    shape = (3,)
    dtype = torch.float32
    expected_b = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32, device=device)

    out_torch = arange_op(shape, dtype, device, start=1.0, end=4.0, run_mode=run_mode)
    print(f"Output b: {out_torch}")
    print(f"Expected b: {expected_b}")
    if run_mode == "npu":
        assert_allclose(out_torch.cpu().numpy(), expected_b.cpu().numpy(), rtol=1e-3, atol=1e-3)
    
    # Test 3: arange(start, end, step)
    shape = (6,)
    dtype = torch.float32
    expected_c = torch.tensor([1.0, 1.5, 2.0, 2.5, 3.0, 3.5], dtype=torch.float32, device=device)

    out_torch = arange_op(shape, dtype, device, start=1.0, end=4.0, step=0.5, run_mode=run_mode)
    print(f"Output c: {out_torch}")
    print(f"Expected c: {expected_c}")
    if run_mode == "npu":
        assert_allclose(out_torch.cpu().numpy(), expected_c.cpu().numpy(), rtol=1e-3, atol=1e-3)
    
    print("✓ Basic usage of arange function completed successfully")


# ============================================================================
# DATATYPE Examples
# ============================================================================

def test_tensor_creation_with_datatypes(device_id = None, run_mode: str = "npu"):
    """Test tensor creation with various data types"""
    print("=" * 60)
    print("Test: Tensor Creation with Various Data Types")
    print("=" * 60)
    
    data_types = [
        (pypto.DT_INT4, "DT_INT4"),
        (pypto.DT_INT8, "DT_INT8"),
        (pypto.DT_INT16, "DT_INT16"),
        (pypto.DT_INT32, "DT_INT32"),
        (pypto.DT_INT64, "DT_INT64"),
        (pypto.DT_FP8, "DT_FP8"),
        (pypto.DT_FP16, "DT_FP16"),
        (pypto.DT_FP32, "DT_FP32"),
        (pypto.DT_BF16, "DT_BF16"),
        (pypto.DT_HF4, "DT_HF4"),
        (pypto.DT_HF8, "DT_HF8"),
        (pypto.DT_UINT8, "DT_UINT8"),
        (pypto.DT_UINT16, "DT_UINT16"),
        (pypto.DT_UINT32, "DT_UINT32"),
        (pypto.DT_UINT64, "DT_UINT64"),
        (pypto.DT_BOOL, "DT_BOOL")
    ]
    
    for dtype, dtype_name in data_types:
        print(f"\nCreating tensor with data type: {dtype_name}")

        # Create a tensor with shape [2, 3] and the specified data type
        tensor = pypto.tensor([2, 3], dtype, f"tensor_{dtype_name}")
        
        # Access tensor attributes
        print(f"Name: {tensor.name}") # e.g., tensor_DT_INT8
        print(f"Data Type: {tensor.dtype}") # e.g., DT_INT8
    
    print("✓ Tensor creation with various data types completed successfully")


# ============================================================================
# FULL Examples
# ============================================================================

def test_full_basic(device_id = None, run_mode: str = "npu"):
    """Test basic usage of full function"""
    print("=" * 60)
    print("Test: Basic Usage of full Function")
    print("=" * 60)

    
@pypto.jit
def full_kernel_npu(out: pypto.Tensor, fill_value: float) -> None:
    pypto.set_vec_tile_shapes(2, 8)
    out[:] = pypto.full(out.shape, fill_value, out.dtype)

@pypto.jit(runtime_options={"run_mode": pypto.RunMode.SIM})
def full_kernel_sim(out: pypto.Tensor, fill_value: float) -> None:
    pypto.set_vec_tile_shapes(2, 8)
    out[:] = pypto.full(out.shape, fill_value, out.dtype)


@pypto.jit
def full_symbolic_scalar_kernel_npu(out: pypto.Tensor, fill_value: pypto.SymbolicScalar) -> None:
    pypto.set_vec_tile_shapes(2, 8)
    out[:] = pypto.full(out.shape, fill_value, out.dtype)

@pypto.jit(runtime_options={"run_mode": pypto.RunMode.SIM})
def full_symbolic_scalar_kernel_sim(out: pypto.Tensor, fill_value: pypto.SymbolicScalar) -> None:
    pypto.set_vec_tile_shapes(2, 8)
    out[:] = pypto.full(out.shape, fill_value, out.dtype)


def full_op(shape: list, dtype: torch.dtype, device: str, fill_value, run_mode: str= "npu", dynamic: bool = False) -> torch.Tensor:
    out = torch.zeros(shape, dtype=dtype, device=device)

    if dynamic:
        out_pto = pypto.from_torch(out, dynamic_axis=[0])
    else:
        out_pto = pypto.from_torch(out)

    if isinstance(fill_value, pypto.SymbolicScalar):
        if run_mode == "npu":
            full_symbolic_scalar_kernel_npu(out_pto, fill_value)
        else:
            full_symbolic_scalar_kernel_sim(out_pto, fill_value)
    else:
        if run_mode == "npu":
            full_kernel_npu(out_pto, fill_value)
        else:
            full_kernel_sim(out_pto, fill_value)
    return out


def test_full_basic(device_id = None, run_mode: str = "npu"):
    """Test basic usage of full function"""
    print("=" * 60)
    print("Test: Basic Usage of full Function")
    print("=" * 60)

    device = f'npu:{device_id}' if (run_mode == "npu" and device_id is not None) else 'cpu'
    
    # Test 1: Create a 2x2 tensor filled with 1.0 (float32)
    shape = [2, 2]
    fill_value = 1.0
    dtype = torch.float32
    expected_a = torch.tensor([[1.0, 1.0], [1.0, 1.0]], dtype=dtype, device=device)

    out_torch = full_op(shape, dtype, device, fill_value, run_mode)
    print(f"Output a: {out_torch}")
    print(f"Expected a: {expected_a}")
    if run_mode == "npu":
        assert_allclose(out_torch.cpu().numpy(), expected_a.cpu().numpy(), rtol=1e-3, atol=1e-3)
    
    # Test 2: Create a 2x2 tensor filled with a symbolic scalar (int32)
    shape = [2, 2]
    fill_value = pypto.symbolic_scalar(1)
    dtype = torch.int32
    expected_b = torch.tensor([[1, 1], [1, 1]], dtype=dtype, device=device)

    out_torch = full_op(shape, dtype, device, fill_value, run_mode)
    print(f"Output b: {out_torch}")
    print(f"Expected b: {expected_b}")
    if run_mode == "npu":
        assert_allclose(out_torch.cpu().numpy(), expected_b.cpu().numpy(), rtol=1e-3, atol=1e-3)
    
    print("✓ Basic usage of full function completed successfully")


# ============================================================================
# TENSOR Examples
# ============================================================================

def test_basic_tensor_creation(device_id = None, run_mode: str = "npu"):
    """Test basic tensor creation"""
    print("=" * 60)
    print("Test: Basic Tensor Creation")
    print("=" * 60)
    
    # Create a tensor with shape [2, 3] and FP16 data type
    tensor = pypto.tensor([2, 3], pypto.DT_FP16, "basic_tensor")
    
    # Access tensor attributes
    print(f"Shape: {tensor.shape}") # [2, 3]
    print(f"Data Type: {tensor.dtype}") # DT_FP16
    print(f"Dimensions: {tensor.dim}") # 2 
    print(f"Format: {tensor.format}") # TILEOP_ND
    print(f"Name: {tensor.name}") # basic_tensor
    
    # Rename the tensor
    tensor.name = "new_name"
    print(f"New Name: {tensor.name}") # new_name
    
    print("✓ Basic tensor creation completed successfully")


def test_tensor_creation_with_format(device_id = None, run_mode: str = "npu"):
    """Test tensor creation with specific format"""
    print("=" * 60)
    print("Test: Tensor Creation with Specific Format")
    print("=" * 60)
    
    # Create a tensor using the NZ format
    tensor = pypto.tensor([512, 32], pypto.DT_FP16, "sparse_tensor", pypto.TileOpFormat.TILEOP_NZ)
    
    # Access tensor attributes
    print(f"Shape: {tensor.shape}") # [512, 32]
    print(f"Data Type: {tensor.dtype}") # DT_FP16
    print(f"Dimensions: {tensor.dim}") # 2 
    print(f"Format: {tensor.format}") # TILEOP_NZ
    print(f"Name: {tensor.name}") # sparse_tensor
    
    print("✓ Tensor Creation with Specific Format completed successfully")


# ============================================================================
# Main Function
# ============================================================================

def main():
    """Run tensor creation operation examples.
    
    Usage:
        python creation_ops.py              # Run all examples
        python creation_ops.py --list       # List all available examples
        python creation_ops.py  arange::test_arange_basic    # Run a specific case
    """
    parser = argparse.ArgumentParser(
        description="PyPTO Tensor Creation Operation Examples",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                      Run all examples
  %(prog)s --list               List all available examples
  %(prog)s  arange::test_arange_basic    Run a specific case
        """
    )
    parser.add_argument(
        'example_id',
        type=str,
        nargs="?",
        help='Run a specific case (e.g., arange::test_arange_basic). If omitted, all cases run.'
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
        'arange::test_arange_basic': {
            'name': 'Test basic usage of arange function',
            'description': 'Basic usage of arange function example',
            'function': test_arange_basic,
        },
        'datatype::test_tensor_creation_with_datatypes': {
            'name': 'Test tensor creation with various data types',
            'description': 'Tensor creation with various data types example',
            'function': test_tensor_creation_with_datatypes,
        },
        'full::test_full_basic': {
            'name': 'Test basic usage of full function',
            'description': 'Basic usage of full function example',
            'function': test_full_basic,
        },
        'tensor::test_basic_tensor_creation': {
            'name': 'Test basic tensor creation',
            'description': 'Basic tensor creation example',
            'function': test_basic_tensor_creation,
        },
        'tensor::test_tensor_creation_with_format': {
            'name': 'Test tensor creation with specific format',
            'description': 'Tensor creation with specific format example',
            'function': test_tensor_creation_with_format,
        }
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
    print("PyPTO Tensor Creation Operation Examples")
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
            print("All creation tests passed!")
            print("=" * 60)
        
    except Exception as e:
        print(f"\nError: {e}")
        raise


if __name__ == "__main__":
    main()

