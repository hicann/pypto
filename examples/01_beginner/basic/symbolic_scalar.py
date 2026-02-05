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
SymbolicScalar Example for PyPTO

This example demonstrates how to use SymbolicScalar in PyPTO, including:
- Immediate (concrete) SymbolicScalar usage
- SymbolicScalar as loop index inside kernel
- Difference between concrete and non-concrete symbolic values

This is a beginner-friendly example focusing on SymbolicScalar semantics
rather than complex numerical computation.
"""

import os
import sys
import argparse
import pypto
import torch
import numpy as np
from numpy.testing import assert_allclose


# ----------------------------------------------------------------------------
# Device Utilities
# ----------------------------------------------------------------------------


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


# ----------------------------------------------------------------------------
# Kernel Definitions
# ----------------------------------------------------------------------------


def create_symbolic_immediate_kernel(shape: tuple, run_mode: str = "npu"):
    if run_mode == "npu":
        mode = pypto.RunMode.NPU
    elif run_mode == "sim":
        mode = pypto.RunMode.SIM
    else:
        raise ValueError(f"Invalid run_mode: {run_mode}. Must be 'npu' or 'sim'")

    @pypto.frontend.jit(runtime_options={"run_mode": mode})
    def symbolic_immediate_kernel(
        x: pypto.Tensor(shape, pypto.DT_FP32),
    ) -> pypto.Tensor(shape, pypto.DT_FP32):
        pypto.set_vec_tile_shapes(2, 8)
        s = pypto.symbolic_scalar(128)
        s = s + 1
        # Assert is not supported yet.
        # Assert whethere s.is_concrete() == True
        # Assert whether int(s) == 129
        y = pypto.add(x, x)
        return y

    return symbolic_immediate_kernel


def create_symbolicscalar_in_loop_kernel(shape: tuple, run_mode: str = "npu"):
    if run_mode == "npu":
        mode = pypto.RunMode.NPU
    elif run_mode == "sim":
        mode = pypto.RunMode.SIM
    else:
        raise ValueError(f"Invalid run_mode: {run_mode}. Must be 'npu' or 'sim'")

    @pypto.frontend.jit(runtime_options={"run_mode": mode})
    def symbolicscalar_in_loop_kernel(
        x: pypto.Tensor(shape, pypto.DT_FP32),
    ) -> pypto.Tensor(shape, pypto.DT_FP32):
        pypto.set_vec_tile_shapes(2, 8)
        y = pypto.zeros(x.shape)
        for _ in pypto.loop(2, name="sym_loop", idx_name="i"):
            # Assert is not supported yet.
            # Assert whether not i.is_concrete()
            # Assert whether i.is_symbol() or i.is_expression()
            # Execute expression: let expr be the result of i + 1
            # Assert whether not expr.is_concrete()
            y = x + y
        return y

    return symbolicscalar_in_loop_kernel


# ----------------------------------------------------------------------------
# Python Wrappers
# ----------------------------------------------------------------------------


def test_symbolicscalar_immediate(device_id: int = None, run_mode: str = "npu") -> None:
    """Immediate (concrete) SymbolicScalar usage"""
    device = f'npu:{device_id}' if (run_mode == "npu" and device_id is not None) else 'cpu'
    x = torch.tensor(
        [1, 2, 3],
        dtype=torch.float32,
        device=device
    )

    y = create_symbolic_immediate_kernel(x.shape, run_mode)(x).cpu()
    golden = (x + x).cpu()

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    if run_mode == "npu":
        assert_allclose(y.numpy(), golden.numpy(), rtol=1e-3, atol=1e-3)
    print("✓ SymbolicScalar immediate test passed")
    print()


def test_symbolicscalar_in_loop(device_id: int = None, run_mode: str = "npu")->None:
    """SymbolicScalar as loop index inside kernel"""
    device = f'npu:{device_id}' if (run_mode == "npu" and device_id is not None) else 'cpu'
    x = torch.tensor(
        [1, 2, 3],
        dtype=torch.float32,
        device=device
    )

    y = create_symbolicscalar_in_loop_kernel(x.shape, run_mode)(x).cpu()
    golden = (x + x).cpu()

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    if run_mode == "npu":
        assert_allclose(y.numpy(), golden.numpy(), rtol=1e-3, atol=1e-3)
    print("✓ SymbolicScalar in loop test passed")
    print()


def test_init_symbolic_scalar_value_arg(device_id: int = None, run_mode: str = "npu")->None:
    """SymbolicScalar Initialization"""
    expected_value = 123

    # Initialize from a concrete value
    scalar = pypto.symbolic_scalar(expected_value)
    assert scalar.is_concrete()
    assert scalar.concrete() == expected_value

    # Initialize from an existing SymbolicScalar
    scalar = pypto.symbolic_scalar(scalar)
    assert scalar.is_concrete()
    assert scalar.concrete() == expected_value

    # Initialize with a name and a concrete value
    named_scalar = pypto.symbolic_scalar("scalar", expected_value)
    assert named_scalar.is_concrete()
    assert named_scalar.concrete() == expected_value

    print("✓ SymbolicScalar Initialization test passed")
    print()


def test_symbolic_scalar_prop(device_id: int = None, run_mode: str = "npu")->None:
    """Inspect core SymbolicScalar properties"""
    scalar = pypto.symbolic_scalar(10)
    assert scalar.is_symbol() == False
    assert scalar.is_expression() == False
    assert scalar.is_immediate() == True
    assert scalar.is_concrete() == True
    assert scalar.concrete() == 10

    scalar2 = pypto.symbolic_scalar("s")
    assert scalar2.is_symbol() == True
    assert scalar2.is_expression() == False
    assert scalar.is_immediate() == True
    assert scalar2.is_concrete() == False

    scalar3 = scalar < 2
    assert isinstance(scalar3, pypto.symbolic_scalar)
    assert scalar3.is_symbol() == False
    assert scalar3.is_expression() == False
    assert scalar3.is_immediate() == True
    assert scalar3.is_concrete() == True
    assert scalar3.concrete() == 0

    scalar4 = scalar2 < 2
    assert isinstance(scalar4, pypto.symbolic_scalar)
    assert scalar4.is_symbol() == False
    assert scalar4.is_expression() == True
    assert scalar4.is_immediate() == False
    assert scalar4.is_concrete() == False

    print("✓ SymbolicScalar properties test passed")
    print()


def test_symbolic_scalar_complex_expr(device_id: int = None, run_mode: str = "npu")->None:
    """SymbolicScalar expression involving multiple comparison operators"""
    b = pypto.symbolic_scalar('b')
    a = (b >= 2) * (b < 8)
    assert str(a) == '((b>=2)*(b<8))'

    print("✓ SymbolicScalar multiple test passed")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="PyPTO SymbolicScalar Example",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s              Run all examples
  %(prog)s --list       List available examples
  %(prog)s symbolicscalar_immediate::test_symbolicscalar_immediate     Run immediate SymbolicScalar example
        """
    )
    parser.add_argument(
        "example_id",
        type=str,
        nargs="?",
        help="Run a specific case (e.g., symbolicscalar_in_loop::test_symbolicscalar_in_loop). If omitted, all cases run."
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all available examples and exit"
    )
    parser.add_argument(
        "--run_mode", "--run-mode",
        nargs="?", type=str, default="npu", choices=["npu", "sim"],
        help="run mode, such as npu/sim etc."
    )

    args = parser.parse_args()


    examples = {
        'symbolicscalar_immediate::test_symbolicscalar_immediate': {
            "name": "SymbolicScalar Immediate in kernel",
            "description": (
                "Demonstrate immediate (concrete) SymbolicScalar usage, including "
                "creation from concrete values and direct evaluation."
            ),
            "function": test_symbolicscalar_immediate
        },
        'symbolicscalar_in_loop::test_symbolicscalar_in_loop': {
            "name": "SymbolicScalar in Loop",
            "description": (
                "Demonstrate SymbolicScalar used as a loop index. "
                "This example verifies that loop indices are symbolic rather than "
                "concrete values, and remain symbolic when used in expressions."
            ),
            "function": test_symbolicscalar_in_loop
        },
        'symbolic_scalar_prop::test_symbolic_scalar_prop': {
            "name": "SymbolicScalar Properties",
            "description": (
                "Inspect core SymbolicScalar properties, including whether a scalar "
                "is symbolic, concrete, immediate, or an expression."
            ),
            "function": test_symbolic_scalar_prop
        },
        'symbolic_scalar_complex_expr::test_symbolic_scalar_complex_expr': {
            "name": "SymbolicScalar Complex Expression (Issue #36)",
            "description": (
                "Demonstrate construction and string representation of a compound "
                "SymbolicScalar expression involving multiple comparison operators."
            ),
            "function": test_symbolic_scalar_complex_expr
        }
    }


    # List examples if requested
    if args.list:
        print("\n" + "=" * 60)
        print("Available Examples")
        print("=" * 60 + "\n")
        for ex_id, ex_info in sorted(examples.items()):
            print(f"  {ex_id}. {ex_info['name']}")
            print(f"     {ex_info['description']}\n")
        return

    # Validate example ID if provided
    if args.example_id is not None:
        if args.example_id not in examples:
            print(f"ERROR: Invalid example ID: {args.example_id}")
            print(f"Valid example IDs are: {', '.join(map(str, sorted(examples.keys())))}")
            print("\nUse --list to see all available examples.")
            sys.exit(1)

    print("\n" + "=" * 60)
    print("PyPTO SymbolicScalar Example")
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
        # Set the device once for all examples
        import torch_npu
        torch.npu.set_device(device_id)

    try:
        for ex_id, ex_info in examples_to_run:
            if args.run_mode == "npu" and device_id is None:
                print(f"Skipping example {ex_id} ({ex_info['name']}): NPU device not configured")
                continue

            print(f"Running Example {ex_id}: {ex_info['name']}")
            ex_info['function'](device_id, args.run_mode)

        if len(examples_to_run) > 1:
            print("=" * 60)
            print("All SymbolicScalar examples passed!")
            print("=" * 60)

    except Exception as e:
        print(f"\nError: {e}")
        raise


if __name__ == "__main__":
    main()
