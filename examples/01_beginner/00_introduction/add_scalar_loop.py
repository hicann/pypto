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
add_scalar_loop Example for PyPTO

This example demonstrates how to implement a add_scalar_loop operation using PyPTO, including:
- Manual add_scalar_loop computation from basic operations
- Dynamic axis marking for variable batch sizes
- Tiling configuration for efficient execution
- Loop-based processing for large tensors

add_scalar_loop is a fundamental operation in neural networks, especially for attention mechanisms.
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

@pypto.jit
def add_kernel_npu(x: pypto.Tensor, y: pypto.Tensor, z: pypto.Tensor, val: int) -> None:
    tensor_shape = x.shape
    pypto.set_vec_tile_shapes(1, 4, 1, 64)

    #calculate the loop parameters
    b = tensor_shape[0]
    tile_b = 1
    b_loop = b // tile_b

    for idx in pypto.loop(b_loop):
        b_offset = idx * tile_b
        b_offset_end = (idx + 1) * tile_b
        t0_sub = x[b_offset:b_offset_end, ...]
        t1_sub = y[b_offset:b_offset_end, ...]
        t3_sub = t0_sub + t1_sub
        z[b_offset:b_offset_end, ...] = t3_sub + val

@pypto.jit(runtime_options={"run_mode": 1})
def add_kernel_sim(x: pypto.Tensor, y: pypto.Tensor, z: pypto.Tensor, val: int) -> None:
    tensor_shape = x.shape
    pypto.set_vec_tile_shapes(1, 4, 1, 64)

    #calculate the loop parameters
    b = tensor_shape[0]
    tile_b = 1
    b_loop = b // tile_b

    for idx in pypto.loop(b_loop):
        b_offset = idx * tile_b
        b_offset_end = (idx + 1) * tile_b
        t0_sub = x[b_offset:b_offset_end, ...]
        t1_sub = y[b_offset:b_offset_end, ...]
        t3_sub = t0_sub + t1_sub
        z[b_offset:b_offset_end, ...] = t3_sub + val

def add_scalar_loop(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor, val: int, run_mode: str = "npu", dynamic: bool = True) -> None:
    if dynamic:
        x_pto = pypto.from_torch(x, "IN_0", dynamic_axis=[0])
        y_pto = pypto.from_torch(y, "IN_1", dynamic_axis=[0])
        z_pto = pypto.from_torch(z, "OUT_0", dynamic_axis=[0])
    else:
        x_pto = pypto.from_torch(x, "IN_0")
        y_pto = pypto.from_torch(y, "IN_1")
        z_pto = pypto.from_torch(z, "OUT_0")

    # launch the kernel
    if run_mode == "npu":
        add_kernel_npu(x_pto, y_pto, z_pto, val)
    else:
        add_kernel_sim(x_pto, y_pto, z_pto, val)

def test_add_scalar_loop(device_id = None, run_mode: str = "npu", dynamic: bool = True) -> None:
    device = f'npu:{device_id}' if (run_mode == "npu" and device_id is not None) else 'cpu'

    shape = (32, 32, 1, 256)
    #prepare data
    val = 1
    x = torch.rand(shape, dtype=torch.float, device=device)
    y = torch.rand(shape, dtype=torch.float, device=device)
    z = torch.zeros(shape, dtype=torch.float, device=device)

    add_scalar_loop(x, y, z, val, run_mode, dynamic)
    golden = torch.add(x, y) + val

    max_diff = np.abs(z.cpu().numpy() - golden.cpu().numpy()).max()
    print(f"Input0 shape : {x.shape}")
    print(f"Input1 shape : {y.shape}")
    print(f"Output shape: {z.shape}")
    print(f"Max difference: {max_diff:.6f}")

    if run_mode == "npu":
        assert_allclose(np.array(z.cpu()), np.array(golden.cpu()), rtol=3e-3, atol=3e-3)
    print("✓ add_scalar_loop test passed")
    print()


def main():
    """Run add_scalar_loop example.
    
    Usage:
        python add_scalar_loop.py          # Run example
        python add_scalar_loop.py --list   # List available examples
    """
    parser = argparse.ArgumentParser(
        description="PyPTO Softmax Example",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s add_scalar_loop::test_add_scalar_loop
            Run the add_scalar_loop::test_add_scalar_loop example
  %(prog)s --list       List all available examples
        """
    )
    parser.add_argument(
        'example_id',
        type=str,
        nargs='?',
        help='Example ID to run (1). If not specified, the example will run.'
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
        default="npu",
        choices=["npu", "sim"],
        help='Run mode, such as npu/sim etc.'
    )
    
    args = parser.parse_args()
    
    # Define available examples
    examples = {
        "add_scalar_loop::test_add_scalar_loop": {
            'name': 'add_scalar_loop',
            'description': 'add_scalar_loop implementation with dynamic batch size',
            'function': test_add_scalar_loop
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
    
    # Validate example ID if provided
    if args.example_id is not None:
        if args.example_id not in examples:
            print(f"ERROR: Invalid example ID: {args.example_id}")
            print(f"Valid example IDs are: {', '.join(map(str, sorted(examples.keys())))}")
            print("\nUse --list to see all available examples.")
            sys.exit(1)
    
    print("\n" + "=" * 60)
    print("PyPTO add_scalar_loop Example")
    print("=" * 60 + "\n")
    
    # Get and validate device ID (needed for NPU examples)
    device_id = None
    examples_to_run = []
    
    if args.example_id is not None:
        # Run single example
        examples_to_run = [(args.example_id, examples[args.example_id])]
    else:
        # Run all examples
        examples_to_run = list(examples.items())
    
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
            print("All add_scalar_loop tests passed!")
            print("=" * 60)
        
    except Exception as e:
        print(f"\nError: {e}")
        raise


if __name__ == "__main__":
    main()