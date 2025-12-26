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
Softmax Example for PyPTO

This example demonstrates how to implement a softmax operation using PyPTO, including:
- Manual softmax computation from basic operations
- Dynamic axis marking for variable batch sizes
- Tiling configuration for efficient execution
- Loop-based processing for large tensors

Softmax is a fundamental operation in neural networks, especially for attention mechanisms.
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


def softmax_core(x: pypto.Tensor) -> pypto.Tensor:
    """
    Core softmax computation: exp(x - max(x)) / sum(exp(x - max(x))).

    Parameters
    ----------
    input_tensor : pypto.Tensor
        Input tensor to apply softmax to

    Returns
    -------
    pypto.tensor
        Softmax normalized tensor
    """
    row_max = pypto.amax(x, dim=-1, keepdim=True)
    sub = x - row_max
    exp = pypto.exp(sub)
    esum = pypto.sum(exp, dim=-1, keepdim=True)
    return exp / esum


@pypto.jit
def softmax_kernel_npu(x: pypto.Tensor, y: pypto.Tensor) -> None:
    # after the dynamic axis of tensor is marked, get the tensor shape accordingly
    tensor_shape = x.shape
    b = tensor_shape[0] # dynamic: symbolic_scalar; static: immediate number
    n1, n2, dim = tensor_shape[1:]
    tile_b = 1
    b_loop = b / tile_b

    # tiling shape setting
    pypto.set_vec_tile_shapes(1, 4, 1, 64)

    for idx in pypto.loop(b_loop):
        b_offset = idx * tile_b
        b_offset_end = (idx + 1) * tile_b
        x_view = x[b_offset:b_offset_end, :n1, :n2, :dim]
        softmax_out = softmax_core(x_view)
        y[b_offset:, ...] = softmax_out


@pypto.jit(runtime_options={"run_mode": 1})
def softmax_kernel_sim(x: pypto.Tensor, y: pypto.Tensor) -> None:
    # after the dynamic axis of tensor is marked, get the tensor shape accordingly
    tensor_shape = x.shape
    b = tensor_shape[0] # dynamic: symbolic_scalar; static: immediate number
    n1, n2, dim = tensor_shape[1:]
    tile_b = 1
    b_loop = b / tile_b

    # tiling shape setting
    pypto.set_vec_tile_shapes(1, 4, 1, 64)

    for idx in pypto.loop(b_loop):
        b_offset = idx * tile_b
        b_offset_end = (idx + 1) * tile_b
        x_view = x[b_offset:b_offset_end, :n1, :n2, :dim]
        softmax_out = softmax_core(x_view)
        y[b_offset:, ...] = softmax_out


def softmax(x: torch.Tensor, run_mode: str = "npu", dynamic: bool = True) -> torch.Tensor:
    y = torch.empty_like(x)

    if dynamic:
        x_pto = pypto.from_torch(x, dynamic_axis=[0])
        y_pto = pypto.from_torch(y, dynamic_axis=[0])
    else:
        x_pto = pypto.from_torch(x)
        y_pto = pypto.from_torch(y)

    # launch the kernel
    if run_mode == "npu":
        softmax_kernel_npu(x_pto, y_pto)
    else:
        softmax_kernel_sim(x_pto, y_pto)
    return y


def test_softmax(device_id = None, run_mode: str = "npu", dynamic: bool = True) -> None:
    device = f'npu:{device_id}' if (run_mode == "npu" and device_id is not None) else 'cpu'

    shape = (32, 32, 1, 256)
    x = torch.rand(shape, dtype=torch.float, device=device)

    y = softmax(x, run_mode, dynamic).cpu() # default dim: -1
    golden = torch.softmax(x, dim=-1).cpu()

    max_diff = np.abs(y.numpy() - golden.numpy()).max()
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Max difference: {max_diff:.6f}")

    if run_mode == "npu":
        assert_allclose(np.array(y), np.array(golden), rtol=3e-3, atol=3e-3)
    print("✓ Softmax test passed")
    print()


def main():
    """Run softmax example.

    Usage:
        python softmax.py          # Run example
        python softmax.py --list   # List available examples
    """
    parser = argparse.ArgumentParser(
        description="PyPTO Softmax Example",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s softmax::test_softmax
            Run the softmax::test_softmax example
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
        "softmax::test_softmax": {
            'name': 'Softmax',
            'description': 'Softmax implementation with dynamic batch size',
            'function': test_softmax
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
    print("PyPTO Softmax Example")
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
            print("All softmax tests passed!")
            print("=" * 60)

    except Exception as e:
        print(f"\nError: {e}")
        raise


if __name__ == "__main__":
    main()
