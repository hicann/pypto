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
Hello World Example for PyPTO

This example demonstrates the simplest tensor addition.
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
def add_kernel_npu(x: pypto.Tensor, y: pypto.Tensor, z: pypto.Tensor) -> None:
    pypto.set_vec_tile_shapes(1, 4, 1, 64)
    z[:] = x + y


@pypto.jit(runtime_options={"run_mode": 1})
def add_kernel_sim(x: pypto.Tensor, y: pypto.Tensor, z: pypto.Tensor) -> None:
    pypto.set_vec_tile_shapes(1, 4, 1, 64)
    z[:] = x + y


def add_direct(x: torch.Tensor, y: torch.Tensor, run_mode: str = "npu") -> torch.Tensor:
    out = torch.empty_like(x)
    pto_input0 = pypto.from_torch(x)
    pto_input1 = pypto.from_torch(y)
    pto_output = pypto.from_torch(out)

    # launch the kernel
    if run_mode == "npu":
        add_kernel_npu(pto_input0, pto_input1, pto_output)
    else:
        add_kernel_sim(pto_input0, pto_input1, pto_output)
    return out


def test_add_direct(device_id = None, run_mode: str = "npu") -> None:
    device = f'npu:{device_id}' if (run_mode == "npu" and device_id is not None) else 'cpu'
    shape = (1, 4, 1, 64)
    #prepare data
    input_data0 = torch.rand(shape, dtype=torch.float, device=device)
    input_data1 = torch.rand(shape, dtype=torch.float, device=device)
    output_data = add_direct(input_data0, input_data1, run_mode)
    golden = torch.add(input_data0, input_data1)

    max_diff = np.abs(output_data.cpu().numpy() - golden.cpu().numpy()).max()
    print(f"Input0 shape: {input_data0.shape}")
    print(f"Input1 shape: {input_data1.shape}")
    print(f"Output shape: {output_data.shape}")
    
    if run_mode == "npu":
        print(f"Max difference: {max_diff:.6f}")
        assert_allclose(np.array(output_data.cpu()), np.array(golden.cpu()), rtol=3e-3, atol=3e-3)
    print("✓ Hello world example passed")
    print()


def main():
    """Run hello_world example.

    Usage:
        python hello_world.py          # Run example
        python hello_world.py --list   # List available examples
    """
    parser = argparse.ArgumentParser(
        description="PyPTO hello_world Example",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s hello_world::test_add_direct
            Run the hello_world::test_add_direct example
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
        "hello_world::test_add_direct": {
            'name': 'hello_world',
            'description': 'add_direct implementation',
            'function': test_add_direct
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
    print("PyPTO hello_world Example")
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
            print("All hello_world tests passed!")
            print("=" * 60)

    except Exception as e:
        print(f"\nError: {e}")
        raise


if __name__ == "__main__":
    main()