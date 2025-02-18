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
add_scalar_loop_multi_jit Example for PyPTO

This example demonstrates how to implement a add_scalar_loop_multi_jit operation using PyPTO, including:
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
        print("ERROR: Environment variable TILE_FWK_DEVICE_ID is not set.")
        print("Please set it before running this example:")
        print("  export TILE_FWK_DEVICE_ID=0")
        return None
    
    try:
        device_id = int(os.environ['TILE_FWK_DEVICE_ID'])
        return device_id
    except ValueError:
        print(f"ERROR: TILE_FWK_DEVICE_ID must be an integer, got: {os.environ['TILE_FWK_DEVICE_ID']}")
        return None


def add_core(input0: pypto.Tensor, input1: pypto.Tensor, output: pypto.Tensor, val: int, add1_flag: bool):
    pypto.set_vec_tile_shapes(1, 4, 1, 64)
    if add1_flag:
        t3 = input0 + input1
        output[:] = t3 + val
    else:
        output[:] = input0 + input1


@pypto.jit
def add_true_kernel_npu(input0: pypto.Tensor, input1: pypto.Tensor, output: pypto.Tensor, val: int):
    add_core(input0, input1, output, val, True)


@pypto.jit(runtime_options={"run_mode": 1})
def add_true_kernel_sim(input0: pypto.Tensor, input1: pypto.Tensor, output: pypto.Tensor, val: int):
    add_core(input0, input1, output, val, True)


@pypto.jit
def add_false_kernel_npu(input0: pypto.Tensor, input1: pypto.Tensor, output: pypto.Tensor, val: int):
    add_core(input0, input1, output, val, False)


@pypto.jit(runtime_options={"run_mode": 1})
def add_false_kernel_sim(input0: pypto.Tensor, input1: pypto.Tensor, output: pypto.Tensor, val: int):
    add_core(input0, input1, output, val, False)


def add_add1flag(input0: torch.Tensor, input1: torch.Tensor, output: torch.Tensor, val: int, add1_flag: bool, run_mode: str = "npu") -> None:
    pto_input0 = pypto.from_torch(input0, "IN_0")
    pto_input1 = pypto.from_torch(input1, "IN_1")
    pto_output = pypto.from_torch(output, "OUT_0")

    if add1_flag:
        # launch the kernel
        if run_mode == "npu":
            add_true_kernel_npu(pto_input0, pto_input1, pto_output, val)
        else:
            add_true_kernel_sim(pto_input0, pto_input1, pto_output, val)
    else:
        if run_mode == "npu":
            add_false_kernel_npu(pto_input0, pto_input1, pto_output, val)
        else:
            add_false_kernel_sim(pto_input0, pto_input1, pto_output, val)


def test_add_scalar_loop_multi_jit(device_id = None, run_mode: str = "npu") -> None:
    device = f'npu:{device_id}' if (run_mode == "npu" and device_id is not None) else 'cpu'

    shape = (32, 32, 1, 256)
    #prepare data
    val = 1
    input_data0 = torch.rand(shape, dtype=torch.float, device=device)
    input_data1 = torch.rand(shape, dtype=torch.float, device=device)
    print(f"Input0 shape: {input_data0.shape}")
    print(f"Input1 shape: {input_data1.shape}")
    golden = torch.add(input_data0, input_data1)
    
    output_data = torch.zeros(shape, dtype=torch.float, device=device)
    add_add1flag(input_data0, input_data1, output_data, val, False, run_mode)
    max_diff = np.abs(output_data.cpu().numpy() - golden.cpu().numpy()).max()
    print(f"Output shape: {output_data.shape}")
    print(f"Max difference: {max_diff:.6f}")
    if run_mode == "npu":
        assert_allclose(np.array(output_data.cpu()), np.array(golden.cpu()), rtol=3e-3, atol=3e-3)

    golden2 = torch.add(input_data0, input_data1) + val
    output_data2 = torch.zeros(shape, dtype=torch.float, device=device)
    add_add1flag(input_data0, input_data1, output_data2, val, True, run_mode)
    max_diff = np.abs(output_data2.cpu().numpy() - golden2.cpu().numpy()).max()
    print(f"Output shape: {output_data2.shape}")
    print(f"Max difference: {max_diff:.6f}")
    if run_mode == "npu":
        assert_allclose(np.array(output_data2.cpu()), np.array(golden2.cpu()), rtol=3e-3, atol=3e-3)

    print("✓ add_scalar_loop_multi_jit test passed")
    print()


def main():
    """Run add_scalar_loop_multi_jit example.
    
    Usage:
        python add_scalar_loop_multi_jit.py          # Run example
        python add_scalar_loop_multi_jit.py --list   # List available examples
    """
    parser = argparse.ArgumentParser(
        description="PyPTO add_scalar_loop_multi_jit Example",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s add_scalar_loop_multi_jit::test_add_scalar_loop_multi_jit
            Run the add_scalar_loop_multi_jit::test_add_scalar_loop_multi_jit example
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
        "add_scalar_loop_multi_jit::test_add_scalar_loop_multi_jit": {
            'name': 'add_scalar_loop_multi_jit',
            'description': 'add_scalar_loop_multi_jit implementation',
            'function': test_add_scalar_loop_multi_jit
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
    print("PyPTO add_scalar_loop_multi_jit Example")
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
            print("All add_scalar_loop_multi_jit tests passed!")
            print("=" * 60)
        
    except Exception as e:
        print(f"\nError: {e}")
        raise


if __name__ == "__main__":
    main()