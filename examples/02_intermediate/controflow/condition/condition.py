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
Condition Function Example for PyPTO

This example demonstrates:
- Basic usage of if_else and pypto.cond functions
- Nested loops with conditional statements
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


def create_nested_loops_with_conditions_kernel(shape: tuple, dynamic: bool = False, run_mode: str = "npu"):
    if dynamic == True:
        w = pypto.frontend.dynamic("w")
        h = pypto.frontend.dynamic("h")
    else:
        w, h = shape
    
    if run_mode == "npu":
        mode = pypto.RunMode.NPU
    elif run_mode == "sim":
        mode = pypto.RunMode.SIM
    else:
        raise ValueError(f"Invalid run_mode: {run_mode}. Must be 'npu' or 'sim'")
    
    @pypto.frontend.jit(runtime_options={"run_mode": mode})
    def nested_loops_with_conditions_kernel(
        a: pypto.Tensor((w, h), pypto.DT_FP32),
        b: pypto.Tensor((w, h), pypto.DT_FP32),
    ) -> pypto.Tensor((w, h), pypto.DT_FP32):
        pypto.set_vec_tile_shapes(2, 8)
        y = pypto.full((w, h), 0.0, pypto.DT_FP32)
        for i in pypto.loop(2):
            for j in pypto.loop(2):
                a_view = a[i:i + 1, j:j + 1]
                b_view = b[i:i + 1, j:j + 1]
                if i == 0:
                    y[i:i + 1, j:j + 1] = a_view + b_view
                else:
                    y[i:i + 1, j:j + 1] = a_view - b_view
        return y

    return nested_loops_with_conditions_kernel
    

def test_nested_loops_with_conditions(device_id = None, run_mode: str = "npu", dynamic: bool = True) -> None:
    """Test nested loops with conditional statements"""
    print("=" * 60)
    print("Test: Nested Loops with Conditional Statements")
    print("=" * 60)
    
    device = f'npu:{device_id}' if (run_mode == "npu" and device_id is not None) else 'cpu'
    
    shape = (2, 2)
    dtype = torch.float
    a = torch.rand(shape, dtype=dtype, device=device)
    b = torch.rand(shape, dtype=dtype, device=device)
    y = create_nested_loops_with_conditions_kernel(shape, dynamic, run_mode)(a, b)
    golden = torch.zeros(shape, dtype=dtype, device=device)
    golden[0] = a[0] + b[0]
    golden[1] = a[1] - b[1]
    golden = golden.cpu()

    if run_mode == "npu":
        assert_allclose(np.array(y.cpu()), np.array(golden.cpu()), rtol=1e-3, atol=1e-3)
        print(f"Output: {y.cpu()}")
        print(f"Expected: {golden.cpu()}")
    print("✓ Nested loops with conditional statements completed successfully")


def main():
    """Run condition examples.
    
    Usage:
        python condition_example.py          # Run all examples
        python condition_example.py 1         # Run example 1 only
        python condition_example.py --list   # List all available examples
    """
    parser = argparse.ArgumentParser(
        description="PyPTO Condition Function Examples",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s              Run all examples
  %(prog)s nested_loops_with_conditions::test_nested_loops_with_conditions
            Run example nested_loops_with_conditions::test_nested_loops_with_conditions
  %(prog)s --list       List all available examples
        """
    )
    parser.add_argument(
        'example_id',
        type=str,
        nargs='?',
        help='Example ID to run (1-4). If not specified, all examples will run.'
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
        'nested_loops_with_conditions::test_nested_loops_with_conditions': {
            'name': 'Test nested loops with conditional statements',
            'description': 'Nested loops with conditional statements example',
            'function': test_nested_loops_with_conditions,
            'requires_npu': True
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
    
    # Validate example ID if provided
    if args.example_id is not None:
        if args.example_id not in examples:
            print(f"ERROR: Invalid example ID: {args.example_id}")
            print(f"Valid example IDs are: {', '.join(map(str, sorted(examples.keys())))}")
            print("\nUse --list to see all available examples.")
            sys.exit(1)
    
    print("\n" + "=" * 60)
    print("PyPTO Condition Function Examples")
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
            print("All condition tests passed!")
            print("=" * 60)
        
    except Exception as e:
        print(f"\nError: {e}")
        raise


if __name__ == "__main__":
    main()
