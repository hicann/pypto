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
add_scalar_loop_dyn_axis_dyn_cond Example for PyPTO

This example demonstrates how to implement a add_scalar_loop_dyn_axis_dyn_cond operation using PyPTO, including:
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


def create_add_scalar_loop_dyn_axis_dyn_cond_kernel(
    shape: tuple, val: int, dynamic_axis: bool = False, run_mode: str = "npu"):
    if dynamic_axis == True:
        w = pypto.frontend.dynamic("w")
        h, c, n = shape[1:]
    else:
        w, h, c, n = shape
    
    if run_mode == "npu":
        mode = pypto.RunMode.NPU
    elif run_mode == "sim":
        mode = pypto.RunMode.SIM
    else:
        raise ValueError(f"Invalid run_mode: {run_mode}. Must be 'npu' or 'sim'")
    
    @pypto.frontend.jit(runtime_options={"run_mode": mode})
    def add_scalar_loop_dyn_axis_dyn_cond_kernel(
        input0: pypto.Tensor((w, h, c, n), pypto.DT_FP32),
        input1: pypto.Tensor((w, h, c, n), pypto.DT_FP32),
    ) -> pypto.Tensor((w, h, c, n), pypto.DT_FP32):
        pypto.set_vec_tile_shapes(1, 4, 1, 64)
        output = pypto.tensor((w, h, c, n), pypto.DT_FP32)
        #calculate the loop parameters
        b = w
        tile_b = 1
        b_loop = b // tile_b

        for idx in pypto.loop(b_loop):
            b_offset = idx * tile_b
            b_offset_end = (idx + 1) * tile_b
            t0_sub = input0[b_offset:b_offset_end, ...]
            t1_sub = input1[b_offset:b_offset_end, ...]
            t3_sub = t0_sub + t1_sub
            if idx < 2:
                output[b_offset:b_offset_end, ...] = t3_sub + val
            else:
                output[b_offset:b_offset_end, ...] = t3_sub
        return output

    return add_scalar_loop_dyn_axis_dyn_cond_kernel


def test_add_scalar_loop_dynamic_axis_dynamic_cond(device_id=None, run_mode: str = "npu") -> None:
    device = f'npu:{device_id}' if (run_mode == "npu" and device_id is not None) else 'cpu'

    shape = (32, 32, 1, 256)
    #prepare data
    val = 1
    input_data0 = torch.rand(shape, dtype=torch.float, device=device)
    input_data1 = torch.rand(shape, dtype=torch.float, device=device)
    output_data = create_add_scalar_loop_dyn_axis_dyn_cond_kernel(shape, val, True, run_mode)(input_data0, input_data1)

    golden = torch.add(input_data0, input_data1)
    golden[0:2, ...] = golden[0:2, ...] + val

    max_diff = np.abs(output_data.cpu().numpy() - golden.cpu().numpy()).max()
    print(f"Input0 shape: {input_data0.shape}")
    print(f"Input1 shape: {input_data1.shape}")
    print(f"Output shape: {output_data.shape}")
    print(f"Max difference: {max_diff:.6f}")

    if run_mode == "npu":
        assert_allclose(np.array(output_data.cpu()), np.array(golden.cpu()), rtol=3e-3, atol=3e-3)
    print("✓ add_scalar_loop_dyn_axis_dyn_cond test passed")
    print()


def main():
    """Run add_scalar_loop_dyn_axis_dyn_cond example.

    Usage:
        python add_scalar_loop_dyn_axis_dyn_cond.py          # Run example
        python add_scalar_loop_dyn_axis_dyn_cond.py --list   # List available examples
    """
    parser = argparse.ArgumentParser(
        description="PyPTO add_scalar_loop_dyn_axis_dyn_cond Example",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s add_scalar_loop_dyn_axis_dyn_cond::test_add_scalar_loop_dynamic_axis_dynamic_cond
            Run the add_scalar_loop_dyn_axis_dyn_cond::test_add_scalar_loop_dynamic_axis_dynamic_cond example
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
        "add_scalar_loop_dyn_axis_dyn_cond::test_add_scalar_loop_dynamic_axis_dynamic_cond": {
            'name': 'add_scalar_loop_dyn_axis_dyn_cond',
            'description': 'add_scalar_loop_dyn_axis_dyn_cond implementation',
            'function': test_add_scalar_loop_dynamic_axis_dynamic_cond
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
    print("PyPTO add_scalar_loop_dyn_axis_dyn_cond Example")
    print("=" * 60 + "\n")

    # Get and validate device ID (needed for NPU examples)
    device_id = None
    examples_to_run = []

    if args.example_id is not None:
        # Run single example
        example = examples.get(args.example_id)
        if example is None:
            raise ValueError(f"Invalid example ID: {args.example_id}")
        examples_to_run = [(args.example_id, example)]
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
            print("All add_scalar_loop_dyn_axis_dyn_cond tests passed!")
            print("=" * 60)

    except Exception as e:
        print(f"\nError: {e}")
        raise


if __name__ == "__main__":
    main()
