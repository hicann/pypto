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
Loop Feature Example for PyPTO

This example demonstrates:
- Basic Loop Usage
- Loop Compile Phase Print Feature
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
def loop_basic_kernel_npu(t0: pypto.Tensor, t1: pypto.Tensor, out0: pypto.Tensor, out1: pypto.Tensor) -> None:
    s, n = t0.shape
    pypto.set_vec_tile_shapes(64, 64)
    for bs_idx in pypto.loop(0, n, 1): # start, stop, step
        t0s = t0[bs_idx * s: (bs_idx+1) * s, :]
        t1s = t1[bs_idx * s: (bs_idx+1) * s, :]
        out0[bs_idx * s: (bs_idx+1) * s, :] = pypto.add(t0s, t1s)
    new_step = 2
    for bs_idx in pypto.loop(0, n, new_step): # start, stop, step
        t0s = t0[bs_idx * s: (bs_idx+new_step) * s, :]
        t1s = t1[bs_idx * s: (bs_idx+new_step) * s, :]
        out1[bs_idx * s: (bs_idx+new_step) * s, :] = pypto.add(t0s, t1s)


@pypto.jit(runtime_options={"run_mode": 1})
def loop_basic_kernel_sim(t0: pypto.Tensor, t1: pypto.Tensor, out0: pypto.Tensor, out1: pypto.Tensor) -> None:
    s, n = t0.shape
    pypto.set_vec_tile_shapes(64, 64)
    for bs_idx in pypto.loop(0, n, 1): # start, stop, step
        t0s = t0[bs_idx * s: (bs_idx+1) * s, :]
        t1s = t1[bs_idx * s: (bs_idx+1) * s, :]
        out0[bs_idx * s: (bs_idx+1) * s, :] = pypto.add(t0s, t1s)
    new_step = 2
    for bs_idx in pypto.loop(0, n, new_step): # start, stop, step
        t0s = t0[bs_idx * s: (bs_idx+new_step) * s, :]
        t1s = t1[bs_idx * s: (bs_idx+new_step) * s, :]
        out1[bs_idx * s: (bs_idx+new_step) * s, :] = pypto.add(t0s, t1s)


def loop_basic(t0: pypto.Tensor, t1: pypto.Tensor, run_mode: str = "npu", dynamic: bool = True) -> torch.Tensor:
    y1 = torch.empty_like(t0)
    y2 = torch.empty_like(t0)

    if dynamic:
        t0_pto = pypto.from_torch(t0, dynamic_axis=[0])
        t1_pto = pypto.from_torch(t1, dynamic_axis=[0])
        y1_pto = pypto.from_torch(y1, dynamic_axis=[0])
        y2_pto = pypto.from_torch(y2, dynamic_axis=[0])
    else:
        t0_pto = pypto.from_torch(t0)
        t1_pto = pypto.from_torch(t1)
        y1_pto = pypto.from_torch(y1)
        y2_pto = pypto.from_torch(y2)

    # launch the kernel
    if run_mode == "npu":
        loop_basic_kernel_npu(t0_pto, t1_pto, y1_pto, y2_pto)
    else:
        loop_basic_kernel_sim(t0_pto, t1_pto, y1_pto, y2_pto)

    return y1, y2


def test_loop_basic(device_id = None, run_mode: str = "npu", dynamic: bool = False) -> None:
    """Test basic loop usage."""
    print("=" * 60)
    print("Test: Basic Loop Usage")
    print("=" * 60)
    
    device = f'npu:{device_id}' if (run_mode == "npu" and device_id is not None) else 'cpu'
    
    s, n = 64, 8
    shape = (n * s, s)
    input_t1 = torch.randn(shape, dtype=torch.float16, device=device)
    input_t2 = torch.randn(shape, dtype=torch.float16, device=device)
    output1, output2 = loop_basic(input_t1, input_t2, run_mode, dynamic)

    # Verify
    expected = input_t1 + input_t2
    if run_mode == "npu":
        max_diff1 = (output1 - expected).abs().max().item()
        max_diff2 = (output2 - expected).abs().max().item()
        equal_output_1_2 = (output1 - output2).abs().max().item() < 1e-6
        print(f"Whether output1 equals output2: {equal_output_1_2}")
        assert max_diff1 < 1e-2, "Result mismatch!"
        assert max_diff2 < 1e-2, "Result mismatch!"
    print("✓ Basic loop usage completed successfully")
    print()


@pypto.jit
def loop_compile_phase_print_kernel_npu(in_t0: pypto.Tensor, in_t1: pypto.Tensor, out_t0: pypto.Tensor, out_t1: pypto.Tensor) -> None:
    pypto.set_vec_tile_shapes(64, 64)
    NOTE = '''
    Below are demonstrations of print usage within loops. 
    It executes only during compilation, cannot truly print variable values, 
    and the number of prints is related to the number of subgraphs generated.
    '''
    SEPARATOR = "*" * 60
    print(NOTE)
    print(SEPARATOR)
    cnt_inside_cond = 0
    cnt_outside_cond = 0
    for outside_idx in pypto.loop(5):
        print(f"outside_idx: {outside_idx}")
        for inside_idx in pypto.loop(3):
            print(f"inside_idx: {outside_idx}")
            res = pypto.add(in_t0, in_t0)
            print(f"res: {res}")
            if pypto.cond(outside_idx < 3):
                print(f"pypto.cond(outside_idx < 3)_count: {cnt_outside_cond}")
                cnt_outside_cond += 1
                res = pypto.add(in_t0, in_t0)
            else:
                res = pypto.sub(in_t0, in_t0)
            if pypto.cond(inside_idx < 2):
                print(f"pypto.cond(inside_idx < 2)_count: {cnt_inside_cond}")
                cnt_inside_cond += 1
                res = pypto.div(in_t0, in_t0)
            else:
                res = pypto.add(in_t1, in_t1)
            out_t0[:] = pypto.add(in_t0, in_t0)
            out_t1[:] = pypto.add(in_t1, in_t1)
    print(SEPARATOR)


@pypto.jit(runtime_options={"run_mode": 1})
def loop_compile_phase_print_kernel_sim(in_t0: pypto.Tensor, in_t1: pypto.Tensor, out_t0: pypto.Tensor, out_t1: pypto.Tensor) -> None:
    pypto.set_vec_tile_shapes(64, 64)
    NOTE = '''
    Below are demonstrations of print usage within loops. 
    It executes only during compilation, cannot truly print variable values, 
    and the number of prints is related to the number of subgraphs generated.
    '''
    SEPARATOR = "*" * 60
    print(NOTE)
    print(SEPARATOR)
    cnt_inside_cond = 0
    cnt_outside_cond = 0
    for outside_idx in pypto.loop(5):
        print(f"outside_idx: {outside_idx}")
        for inside_idx in pypto.loop(3):
            print(f"inside_idx: {outside_idx}")
            res = pypto.add(in_t0, in_t0)
            print(f"res: {res}")
            if pypto.cond(outside_idx < 3):
                print(f"pypto.cond(outside_idx < 3)_count: {cnt_outside_cond}")
                cnt_outside_cond += 1
                res = pypto.add(in_t0, in_t0)
            else:
                res = pypto.sub(in_t0, in_t0)
            if pypto.cond(inside_idx < 2):
                print(f"pypto.cond(inside_idx < 2)_count: {cnt_inside_cond}")
                cnt_inside_cond += 1
                res = pypto.div(in_t0, in_t0)
            else:
                res = pypto.add(in_t1, in_t1)
            out_t0[:] = pypto.add(in_t0, in_t0)
            out_t1[:] = pypto.add(in_t1, in_t1)
    print(SEPARATOR)


def loop_compile_phase_print(in_t0: pypto.Tensor, in_t1: pypto.Tensor, run_mode: str = "npu", dynamic: bool = True) -> torch.Tensor:
    y1 = torch.empty_like(in_t0)
    y2 = torch.empty_like(in_t1)

    if dynamic:
        in_t0_pto = pypto.from_torch(in_t0, dynamic_axis=[0])
        in_t1_pto = pypto.from_torch(in_t1, dynamic_axis=[0])
        y1_pto = pypto.from_torch(y1, dynamic_axis=[0])
        y2_pto = pypto.from_torch(y2, dynamic_axis=[0])
    else:
        in_t0_pto = pypto.from_torch(in_t0)
        in_t1_pto = pypto.from_torch(in_t1)
        y1_pto = pypto.from_torch(y1)
        y2_pto = pypto.from_torch(y2)

    # launch the kernel
    if run_mode == "npu":
        loop_compile_phase_print_kernel_npu(in_t0_pto, in_t1_pto, y1_pto, y2_pto)
    else:
        loop_compile_phase_print_kernel_sim(in_t0_pto, in_t1_pto, y1_pto, y2_pto)
    return y1, y2


def test_loop_compile_phase_print(device_id = None, run_mode: str = "npu", dynamic: bool = False) -> None:
    """Test loop compile phase print"""
    print("=" * 60)
    print("Test: Loop Compile Phase Print Feature")
    print("=" * 60)
    
    device = f'npu:{device_id}' if (run_mode == "npu" and device_id is not None) else 'cpu'
    
    m, n = 6, 8
    shape = (m, n)
    input_t1 = torch.randn(shape, dtype=torch.float16, device=device)
    input_t2 = torch.randn(shape, dtype=torch.float16, device=device)
    output_t1, output_t2 = loop_compile_phase_print(input_t1, input_t2, run_mode, dynamic)
    # Verify
    expected_t1 = input_t1 + input_t1
    expected_t2 = input_t2 + input_t2
    if run_mode == "npu":
        max_diff_t1 = (output_t1 - expected_t1).abs().max().item()
        max_diff_t2 = (output_t2 - expected_t2).abs().max().item()
        print(f"Max difference from PyTorch: {max_diff_t1:.6f}")
        print(f"Max difference from PyTorch: {max_diff_t2:.6f}")
        assert max_diff_t1 < 1e-2, "Result mismatch!"
        assert max_diff_t2 < 1e-2, "Result mismatch!"
        print("✓ Test loop compile phase print completed successfully")
        print()


def main():
    """Run loop_feature examples.
    
    Usage:
        python loop_feature.py          # Run all examples
        python loop_feature.py 1         # Run example 1 only
        python loop_feature.py --list   # List all available examples
    """
    parser = argparse.ArgumentParser(
        description="PyPTO Loop Feature Examples",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s              Run all examples
  %(prog)s loop_basic::test_loop_basic
            Run example loop_basic::test_loop_basic
  %(prog)s --list       List all available examples
        """
    )
    parser.add_argument(
        'example_id',
        type=str,
        nargs='?',
        help='Example ID to run (1-2). If not specified, all examples will run.'
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
        'loop_basic::test_loop_basic': {
            'name': 'Test basic loop usage',
            'description': 'Basic loop usages example',
            'function': test_loop_basic,
        },
        'loop_compile_phase_print::test_loop_compile_phase_print': {
            'name': 'Test loop compile phase print',
            'description': 'Loop compile phase print example',
            'function': test_loop_compile_phase_print,
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
    print("PyPTO Loop Examples")
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
            print("All loop tests passed!")
            print("=" * 60)
        
    except Exception as e:
        print(f"\nError: {e}")
        raise


if __name__ == "__main__":
    main()