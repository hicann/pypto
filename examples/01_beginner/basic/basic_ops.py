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
Basic Operations Example for PyPTO

This example demonstrates fundamental PyPTO operations including:
- Tensor creation
- Element-wise operations (add, mul, sub, div)
- Matrix multiplication
- View operations
- Activation functions

This is a beginner-friendly example that shows the core concepts of PyPTO.
"""

import os
import sys
import argparse
import pypto
import torch


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


def test_tensor_creation(device_id: int = None, run_mode: str = "npu", dynamic: bool = False) -> None:
    """Example 1: Creating tensors with different properties."""
    print("=" * 60)
    print("Example 1: Tensor Creation")
    print("=" * 60)

    # Create a tensor with shape [4, 4] and FP16 data type
    tensor = pypto.tensor([4, 4], pypto.DT_FP16, "my_tensor")

    print(f"Tensor name: {tensor.name}")
    print(f"Tensor shape: {tensor.shape}")
    print(f"Tensor dtype: {tensor.dtype}")
    print(f"Tensor format: {tensor.format}")
    print(f"Tensor dimensions: {tensor.dim}")
    print()


def create_element_wise_ops_kernel(shape: tuple, run_mode: str = "npu"):
    if run_mode == "npu":
        mode = pypto.RunMode.NPU
    elif run_mode == "sim":
        mode = pypto.RunMode.SIM
    else:
        raise ValueError(f"Invalid run_mode: {run_mode}. Must be 'npu' or 'sim'")

    @pypto.frontend.jit(runtime_options={"run_mode": mode})
    def element_wise_ops_kernel(
        a: pypto.Tensor(shape, pypto.DT_FP16),
        b: pypto.Tensor(shape, pypto.DT_FP16),
    ) -> pypto.Tensor(shape, pypto.DT_FP16):
        pypto.set_vec_tile_shapes(8, 8)
        add_result = pypto.add(a, b)
        mul_result = pypto.mul(add_result, 2.0)
        return mul_result
    return element_wise_ops_kernel


def test_element_wise_operations(device_id: int = None, run_mode: str = "npu", dynamic: bool = False) -> None:
    """Example 2: Element-wise arithmetic operations."""
    print("=" * 60)
    print("Example 2: Element-wise Operations")
    print("=" * 60)

    device = f'npu:{device_id}' if (run_mode == "npu" and device_id is not None) else 'cpu'
    shape = (8, 8)
    a_torch = torch.randn(shape, dtype=torch.float16, device=device)
    b_torch = torch.randn(shape, dtype=torch.float16, device=device)

    c_torch = create_element_wise_ops_kernel(shape, run_mode)(a_torch, b_torch)

    expected = (a_torch + b_torch) * 2.0
    max_diff = (c_torch - expected).abs().max().item()
    print(f"Input A shape: {a_torch.shape}")
    print(f"Input B shape: {b_torch.shape}")
    print(f"Output shape: {c_torch.shape}")
    if run_mode == "npu":
        print(f"expected: {expected}")
        print(f"c_torch: {c_torch}")
        print(f"Max difference from PyTorch: {max_diff:.6f}")
        assert max_diff < 1e-2, "Result mismatch!"
    print("✓ Element-wise operations completed successfully")
    print()


def create_matrix_multiply_kernel(shape: tuple, run_mode: str = "npu", dynamic: bool = False):
    if dynamic:
        m = pypto.frontend.dynamic("m")
        k = pypto.frontend.dynamic("k")
        n = shape[2]
    else:
        m, k, n = shape

    if run_mode == "npu":
        mode = pypto.RunMode.NPU
    elif run_mode == "sim":
        mode = pypto.RunMode.SIM
    else:
        raise ValueError(f"Invalid run_mode: {run_mode}. Must be 'npu' or 'sim'")

    @pypto.frontend.jit(runtime_options={"run_mode": mode})
    def matrix_multiply_kernel(
        a: pypto.Tensor((m, k), pypto.DT_BF16),
        b: pypto.Tensor((k, n), pypto.DT_BF16),
    ) -> pypto.Tensor((m, n), pypto.DT_BF16):
        pypto.set_cube_tile_shapes([32, 32], [64, 64], [64, 64])
        c = pypto.matmul(a, b, a.dtype)
        return c

    return matrix_multiply_kernel


def test_matrix_multiplication(device_id: int = None, run_mode: str = "npu", dynamic: bool = False) -> None:
    """Example 3: Matrix multiplication."""
    print("=" * 60)
    print("Example 3: Matrix Multiplication")
    print("=" * 60)

    device = f'npu:{device_id}' if (run_mode == "npu" and device_id is not None) else 'cpu'
    m, k, n = 64, 128, 64
    a_torch = torch.randn(m, k, dtype=torch.bfloat16, device=device)
    b_torch = torch.randn(k, n, dtype=torch.bfloat16, device=device)

    c_torch = create_matrix_multiply_kernel((m, k, n), run_mode, dynamic)(a_torch, b_torch)
    
    expected = torch.matmul(a_torch, b_torch)
    max_diff = (c_torch - expected).abs().max().item()
    print(f"Matrix A shape: {a_torch.shape}")
    print(f"Matrix B shape: {b_torch.shape}")
    print(f"Output C shape: {c_torch.shape}")
    if run_mode == "npu":
        print(f"Max difference from PyTorch: {max_diff:.6f}")
        assert max_diff < 1e-1, "Result mismatch!"
    print("✓ Matrix multiplication completed successfully")
    print()


def create_apply_activations_kernel(shape: tuple, run_mode: str = "npu", dynamic: bool = False):
    if dynamic:
        x_shape = pypto.frontend.dynamic("x_shape")
    else:
        x_shape = shape

    if run_mode == "npu":
        mode = pypto.RunMode.NPU
    elif run_mode == "sim":
        mode = pypto.RunMode.SIM
    else:
        raise ValueError(f"Invalid run_mode: {run_mode}. Must be 'npu' or 'sim'")
    
    @pypto.frontend.jit(runtime_options={"run_mode": mode})
    def apply_activations_kernel(
        x: pypto.Tensor(x_shape, pypto.DT_FP16),
    ) -> pypto.Tensor(x_shape, pypto.DT_FP16):
        pypto.set_vec_tile_shapes(32, 64)
        result = pypto.sigmoid(x)
        return result

    return apply_activations_kernel


def test_activation_functions(device_id: int = None, run_mode: str = "npu", dynamic: bool = False) -> None:
    """Example 4: Activation functions."""
    print("=" * 60)
    print("Example 4: Activation Functions")
    print("=" * 60)

    device = f'npu:{device_id}' if (run_mode == "npu" and device_id is not None) else 'cpu'
    shape = (32, 64)
    input_torch = torch.randn(shape, dtype=torch.float16, device=device)

    output_torch = create_apply_activations_kernel(shape, run_mode, dynamic)(input_torch)
    
    expected = torch.sigmoid(input_torch)
    max_diff = (output_torch - expected).abs().max().item()
    print(f"Input shape: {input_torch.shape}")
    print(f"Output shape: {output_torch.shape}")
    print(f"Output range: [{output_torch.min():.4f}, {output_torch.max():.4f}]")
    if run_mode == "npu":
        print(f"Max difference from PyTorch: {max_diff:.6f}")
        assert max_diff < 1e-2, "Result mismatch!"
    print("✓ Activation functions completed successfully")
    print()


def create_view_operations_kernel(shape: tuple, run_mode: str = "npu", dynamic: bool = False):
    if dynamic:
        h = pypto.frontend.dynamic("h")
        w = shape[1]
    else:
        h, w = shape

    if run_mode == "npu":
        mode = pypto.RunMode.NPU
    elif run_mode == "sim":
        mode = pypto.RunMode.SIM
    else:
        raise ValueError(f"Invalid run_mode: {run_mode}. Must be 'npu' or 'sim'")

    @pypto.frontend.jit(runtime_options={"run_mode": mode})
    def view_operations_kernel(
        input_tensor: pypto.Tensor((h, w), pypto.DT_FP16),
    ) -> pypto.Tensor((h, w), pypto.DT_FP16):
        tile_h, tile_w = 32, 32
        pypto.set_vec_tile_shapes(tile_h, tile_w)

        h_tiles = (h + tile_h - 1) // tile_h
        w_tiles = (w + tile_w - 1) // tile_w

        output_tensor = pypto.Tensor((h, w), pypto.DT_FP16)
        for h_idx in pypto.loop(h_tiles, name="h_loop", idx_name="h_idx"):
            for w_idx in pypto.loop(w_tiles, name="w_loop", idx_name="w_idx"):
                h_offset = h_idx * tile_h
                w_offset = w_idx * tile_w
                view = pypto.view(input_tensor, [tile_h, tile_w], [h_offset, w_offset])
                result = pypto.mul(view, 2.0)
                pypto.assemble(result, [h_offset, w_offset], output_tensor)
        return output_tensor

    return view_operations_kernel


def test_view_operations(device_id: int = None, run_mode: str = "npu", dynamic: bool = False) -> None:
    """Example 5: View operations for tiling."""
    print("=" * 60)
    print("Example 5: View Operations")
    print("=" * 60)

    device = f'npu:{device_id}' if (run_mode == "npu" and device_id is not None) else 'cpu'
    shape = (256, 512)
    input_torch = torch.randn(shape, dtype=torch.float16, device=device)

    output_torch = create_view_operations_kernel(shape, run_mode, dynamic)(input_torch)

    # Verify
    expected = input_torch * 2.0
    max_diff = (output_torch - expected).abs().max().item()
    print(f"Input shape: {input_torch.shape}")
    print(f"Tile size: (32, 32)")
    print(f"Number of tiles: ({shape[0]//32 + 1}, {shape[1]//32 + 1})")
    print(f"Output shape: {output_torch.shape}")
    if run_mode == "npu":
        print(f"Max difference from PyTorch: {max_diff:.6f}")
        assert max_diff < 1e-2, "Result mismatch!"
    print("✓ View operations completed successfully")
    print()


def create_linear_layer_with_activation_kernel(shape: tuple, run_mode: str = "npu", dynamic: bool = False):
    if dynamic:
        batch = pypto.frontend.dynamic("batch")
        in_features, out_features = shape[1], shape[2]  
    else:
        batch, in_features, out_features = shape

    if run_mode == "npu":
        mode = pypto.RunMode.NPU
    elif run_mode == "sim":
        mode = pypto.RunMode.SIM
    else:
        raise ValueError(f"Invalid run_mode: {run_mode}. Must be 'npu' or 'sim'")
    
    @pypto.frontend.jit(runtime_options={"run_mode": mode})
    def linear_layer_with_activation_kernel(
        x: pypto.Tensor((batch, in_features), pypto.DT_BF16),
        w: pypto.Tensor((in_features, out_features), pypto.DT_BF16),
        b: pypto.Tensor((out_features,), pypto.DT_BF16),
    ) -> pypto.Tensor((batch, out_features), pypto.DT_BF16):
        pypto.set_vec_tile_shapes(32, 64)
        pypto.set_cube_tile_shapes([32, 32], [64, 64], [64, 64])
        linear = pypto.matmul(x, w, b.dtype)
        biased = pypto.add(linear, b)
        y[:] = pypto.sigmoid(biased)
        return y

    return linear_layer_with_activation_kernel


def test_combined_operations(device_id: int = None, run_mode: str = "npu", dynamic: bool = False) -> None:
    """Example 6: Combining multiple operations."""
    print("=" * 60)
    print("Example 6: Combined Operations")
    print("=" * 60)

    # Get current device ID (set in main)
    device = f'npu:{device_id}' if (run_mode == "npu" and device_id is not None) else 'cpu'

    # Simple neural network layer: y = sigmoid(x @ W + b)
    batch, in_features, out_features = 32, 64, 32
    x_torch = torch.randn(batch, in_features, dtype=torch.bfloat16, device=device)
    W_torch = torch.randn(in_features, out_features, dtype=torch.bfloat16, device=device)
    b_torch = torch.randn(out_features, dtype=torch.bfloat16, device=device)
    y_torch = create_linear_layer_with_activation_kernel(
        (batch, in_features, out_features), run_mode, dynamic)(x_torch, W_torch, b_torch)
    
    expected = torch.sigmoid(torch.matmul(x_torch, W_torch) + b_torch)
    max_diff = (y_torch - expected).abs().max().item()
    print(f"Input x shape: {x_torch.shape}")
    print(f"Weight W shape: {W_torch.shape}")
    print(f"Bias b shape: {b_torch.shape}")
    print(f"Output y shape: {y_torch.shape}")
    if run_mode == "npu":
        print(f"Max difference from PyTorch: {max_diff:.6f}")
        assert max_diff < 1e-1, "Result mismatch!"
    print("✓ Combined operations completed successfully")
    print()


def main():
    """Run basic operation examples.

    Usage:
        python basic_operations.py          # Run all examples
        python basic_operations.py element_wise_operations::test_element_wise_operations
                                # Run example element_wise_operations::test_element_wise_operations only
        python basic_operations.py --list   # List all available examples
    """
    parser = argparse.ArgumentParser(
        description="PyPTO Basic Operations Examples",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s              Run all examples
  %(prog)s element_wise_operations::test_element_wise_operations            Run example 2 (Element-wise operations)
  %(prog)s --list       List all available examples
        """
    )
    parser.add_argument(
        'example_id',
        type=str,
        nargs='?',
        help='Run a specific case (e.g., element_wise_operations::test_element_wise_operations). If omitted, all cases run.'
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
        "tensor_creation::test_tensor_creation": {
            'name': 'Tensor Creation',
            'description': 'Creating tensors with different properties',
            'function': test_tensor_creation,
        },
        "element_wise_operations::test_element_wise_operations": {
            'name': 'Element-wise Operations',
            'description': 'Element-wise arithmetic operations (add, mul)',
            'function': test_element_wise_operations,
        },
        "matrix_multiplication::test_matrix_multiplication": {
            'name': 'Matrix Multiplication',
            'description': 'Matrix multiplication operations',
            'function': test_matrix_multiplication,
        },
        "activation_functions::test_activation_functions": {
            'name': 'Activation Functions',
            'description': 'Activation functions (sigmoid)',
            'function': test_activation_functions,
        },
        "view_operations::test_view_operations": {
            'name': 'View Operations',
            'description': 'View operations for tiling',
            'function': test_view_operations,
        },
        "combined_operations::test_combined_operations": {
            'name': 'Combined Operations',
            'description': 'Combining multiple operations',
            'function': test_combined_operations,
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
    print("PyPTO Basic Operations Examples")
    print("=" * 60 + "\n")

    # Get and validate device ID (needed for NPU examples)
    device_id = None
    examples_to_run = []

    if args.example_id is not None:
        examples_to_run = [(args.example_id, examples[args.example_id])]
    else:
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
            print("All examples completed successfully!")
            print("=" * 60)

    except Exception as e:
        print(f"\nError: {e}")
        print("\nTroubleshooting:")
        print("1. Ensure CANN environment is sourced")
        print("2. Check NPU device is available: npu-smi info")
        print("3. Verify PyTorch and torch_npu are installed")
        raise


if __name__ == "__main__":
    main()