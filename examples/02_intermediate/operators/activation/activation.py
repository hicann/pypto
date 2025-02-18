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
Custom Activation Functions Example for PyPTO

This example demonstrates how to implement custom activation functions by composing
PyPTO operations. It shows:
- SiLU (Swish) activation: x * sigmoid(x)
- GELU activation: x * sigmoid(1.702 * x) approximation
- SwiGLU activation: Swish(gate) * up
- GeGLU activation: GELU(gate) * up
- Custom activation composition patterns

These activations are commonly used in modern transformer architectures.
"""

import os
import sys
import argparse
import pypto
import torch
import numpy as np
from numpy.testing import assert_allclose
from dataclasses import dataclass
from typing import Literal


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


# Constants for element creation
F_1 = 1.0
F_NEGA_1 = -1.0

# Reference implementations for verification
def silu_golden(x: torch.Tensor) -> torch.Tensor:
    """PyTorch reference implementation of SiLU."""
    return x * torch.sigmoid(x)


def gelu_golden(x: torch.Tensor) -> torch.Tensor:
    """PyTorch reference implementation of GELU."""
    return torch.nn.functional.gelu(x)


def swiglu_golden(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """PyTorch reference implementation of SwiGLU."""
    return (gate * torch.sigmoid(gate)) * up


def geglu_golden(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """PyTorch reference implementation of GeGLU."""
    return torch.nn.functional.gelu(gate) * up


@pypto.jit
def silu_activation_kernel_npu(x: pypto.tensor, y: pypto.tensor) -> None:
    """
    SiLU (Swish) activation function: x * sigmoid(x)

    SiLU is a smooth, non-monotonic activation function that has been shown
    to work well in deep networks.

    Formula: SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))

    Parameters
    ----------
    x : pypto.tensor
        Input tensor

    Returns
    -------
    pypto.tensor
        SiLU activated tensor
    """
    # Configure tiling based on input shape
    if len(x.shape) >= 2:
        n_tile = 32
        tile_shapes = [n_tile for _ in range(len(x.shape))]
        pypto.set_vec_tile_shapes(*tile_shapes)
    else:
        pypto.set_vec_tile_shapes(32, 128)

    # Compute sigmoid(x) = 1 / (1 + exp(-x))
    # SiLU(x) = x * sigmoid(x)
    y[:] =  x * pypto.sigmoid(x)


@pypto.jit(runtime_options={"run_mode": 1})
def silu_activation_kernel_sim(x: pypto.tensor, y: pypto.tensor) -> None:
    """
    SiLU (Swish) activation function: x * sigmoid(x)

    SiLU is a smooth, non-monotonic activation function that has been shown
    to work well in deep networks.

    Formula: SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))

    Parameters
    ----------
    x : pypto.tensor
        Input tensor

    Returns
    -------
    pypto.tensor
        SiLU activated tensor
    """
    # Configure tiling based on input shape
    if len(x.shape) >= 2:
        n_tile = 32
        tile_shapes = [n_tile for _ in range(len(x.shape))]
        pypto.set_vec_tile_shapes(*tile_shapes)
    else:
        pypto.set_vec_tile_shapes(32, 128)

    # Compute sigmoid(x) = 1 / (1 + exp(-x))
    # SiLU(x) = x * sigmoid(x)
    y[:] =  x * pypto.sigmoid(x)


def silu_activation(x: torch.Tensor, run_mode: str = "npu", dynamic: bool = False) -> torch.Tensor:
    y = torch.empty_like(x)

    if dynamic:
        x_pto = pypto.from_torch(x, dynamic_axis=[0])
        y_pto = pypto.from_torch(y, dynamic_axis=[0])
    else:
        x_pto = pypto.from_torch(x)
        y_pto = pypto.from_torch(y)

    # launch the kernel
    if run_mode == "npu":
        silu_activation_kernel_npu(x_pto, y_pto)
    else:
        silu_activation_kernel_sim(x_pto, y_pto)

    return y


def test_silu(device_id = None, run_mode: str = "npu", dynamic: bool = False) -> None:
    """Test SiLU activation."""
    print("=" * 60)
    print("Test: SiLU Activation")
    print("=" * 60)

    device = f'npu:{device_id}' if (run_mode == "npu" and device_id is not None) else 'cpu'

    shape = (32, 128)
    x_torch = torch.randn(shape, dtype=torch.bfloat16, device=device)

    # Execute
    out_torch = silu_activation(x_torch, run_mode, dynamic)

    # Verify
    expected = silu_golden(x_torch)
    max_diff = (out_torch - expected).abs().max().item()
    print(f"Input shape: {x_torch.shape}")
    print(f"Output shape: {out_torch.shape}")
    if run_mode == "npu":
        print(f"Max difference: {max_diff:.6f}")
        assert max_diff < 1e-1, "Result mismatch!"
    print("✓ SiLU passed")
    print()


@pypto.jit
def gelu_activation_kernel_npu(x: pypto.tensor, y: pypto.tensor) -> None:
    """
    GELU (Gaussian Error Linear Unit) activation function.

    Uses approximation: x * sigmoid(1.702 * x)
    This is a fast approximation of the full GELU formula.

    Parameters
    ----------
    x : pypto.tensor
        Input tensor

    Returns
    -------
    pypto.tensor
        GELU activated tensor
    """
    # Configure tiling
    if len(x.shape) >= 2:
        n_tile = 32
        tile_shapes = [n_tile for _ in range(len(x.shape))]
        pypto.set_vec_tile_shapes(*tile_shapes)
    else:
        pypto.set_vec_tile_shapes(32, 128)

    # GELU approximation: x * sigmoid(1.702 * x)
    coeff = float(1.702)
    x_scaled = x * coeff

    # GELU(x) = x * sigmoid(1.702 * x)
    y[:] =  x * pypto.sigmoid(x_scaled)


@pypto.jit(runtime_options={"run_mode": 1})
def gelu_activation_kernel_sim(x: pypto.tensor, y: pypto.tensor) -> None:
    """
    GELU (Gaussian Error Linear Unit) activation function.

    Uses approximation: x * sigmoid(1.702 * x)
    This is a fast approximation of the full GELU formula.

    Parameters
    ----------
    x : pypto.tensor
        Input tensor

    Returns
    -------
    pypto.tensor
        GELU activated tensor
    """
    # Configure tiling
    if len(x.shape) >= 2:
        n_tile = 32
        tile_shapes = [n_tile for _ in range(len(x.shape))]
        pypto.set_vec_tile_shapes(*tile_shapes)
    else:
        pypto.set_vec_tile_shapes(32, 128)

    # GELU approximation: x * sigmoid(1.702 * x)
    coeff = float(1.702)
    x_scaled = x * coeff

    # GELU(x) = x * sigmoid(1.702 * x)
    y[:] =  x * pypto.sigmoid(x_scaled)


def gelu_activation(x: torch.Tensor, run_mode: str = "npu", dynamic: bool = False) -> torch.Tensor:
    y = torch.empty_like(x)

    if dynamic:
        x_pto = pypto.from_torch(x, dynamic_axis=[0])
        y_pto = pypto.from_torch(y, dynamic_axis=[0])
    else:
        x_pto = pypto.from_torch(x)
        y_pto = pypto.from_torch(y)

    # launch the kernel
    if run_mode == "npu":
        gelu_activation_kernel_npu(x_pto, y_pto)
    else:
        gelu_activation_kernel_sim(x_pto, y_pto)

    return y


def test_gelu(device_id = None, run_mode: str = "npu", dynamic: bool = False) -> None:
    """Test GELU activation."""
    print("=" * 60)
    print("Test: GELU Activation")
    print("=" * 60)

    device = f'npu:{device_id}' if (run_mode == "npu" and device_id is not None) else 'cpu'

    shape = (32, 128)
    x_torch = torch.randn(shape, dtype=torch.bfloat16, device=device)

    # Execute
    out_torch = gelu_activation(x_torch, run_mode, dynamic)

    # Verify
    expected = gelu_golden(x_torch)
    max_diff = (out_torch - expected).abs().max().item()

    print(f"Input shape: {x_torch.shape}")
    print(f"Output shape: {out_torch.shape}")
    if run_mode == "npu":
        print(f"Max difference: {max_diff:.6f}")
        assert max_diff < 1e-1, "Result mismatch!"
    print("✓ GELU passed")
    print()
    

@pypto.jit
def swiglu_activation_kernel_npu(gate: pypto.tensor, up: pypto.tensor, y: pypto.tensor) -> None:
    """
    SwiGLU activation function: Swish(gate) * up

    SwiGLU is a gated linear unit that uses Swish (SiLU) as the gating function.
    It's commonly used in modern LLMs like PaLM and LLaMA.

    Formula: SwiGLU(gate, up) = Swish(gate) * up = (gate * sigmoid(gate)) * up

    Parameters
    ----------
    gate : pypto.tensor
        Gate tensor
    up : pypto.tensor
        Up projection tensor

    Returns
    -------
    pypto.tensor
        SwiGLU activated tensor
    """
    # Configure tiling
    if len(gate.shape) >= 2:
        pypto.set_vec_tile_shapes(gate.shape[0], gate.shape[1])
    else:
        n_tile = 32
        tile_shapes = [n_tile for _ in range(len(gate.shape))]
        pypto.set_vec_tile_shapes(*tile_shapes)

    # Swish(gate) = gate * sigmoid(gate)
    sigmoid = pypto.sigmoid(gate)
    swish = gate * sigmoid

    # Multiply with up projection
    y[:] =  swish * up


@pypto.jit(runtime_options={"run_mode": 1})
def swiglu_activation_kernel_sim(gate: pypto.tensor, up: pypto.tensor, y: pypto.tensor) -> None:
    """
    SwiGLU activation function: Swish(gate) * up

    SwiGLU is a gated linear unit that uses Swish (SiLU) as the gating function.
    It's commonly used in modern LLMs like PaLM and LLaMA.

    Formula: SwiGLU(gate, up) = Swish(gate) * up = (gate * sigmoid(gate)) * up

    Parameters
    ----------
    gate : pypto.tensor
        Gate tensor
    up : pypto.tensor
        Up projection tensor

    Returns
    -------
    pypto.tensor
        SwiGLU activated tensor
    """
    # Configure tiling
    if len(gate.shape) >= 2:
        pypto.set_vec_tile_shapes(gate.shape[0], gate.shape[1])
    else:
        n_tile = 32
        tile_shapes = [n_tile for _ in range(len(gate.shape))]
        pypto.set_vec_tile_shapes(*tile_shapes)

    # Swish(gate) = gate * sigmoid(gate)
    sigmoid = pypto.sigmoid(gate)
    swish = gate * sigmoid

    # Multiply with up projection
    y[:] =  swish * up


def swiglu_activation(gate: pypto.tensor, up: pypto.tensor, run_mode: str = "npu", dynamic: bool = False) -> torch.Tensor:
    y = torch.empty_like(gate)

    if dynamic:
        gate_pto = pypto.from_torch(gate, dynamic_axis=[0])
        up_pto = pypto.from_torch(up, dynamic_axis=[0])
        y_pto = pypto.from_torch(y, dynamic_axis=[0])
    else:
        gate_pto = pypto.from_torch(gate)
        up_pto = pypto.from_torch(up)
        y_pto = pypto.from_torch(y)

    # launch the kernel
    if run_mode == "npu":
        swiglu_activation_kernel_npu(gate_pto, up_pto, y_pto)
    else:
        swiglu_activation_kernel_sim(gate_pto, up_pto, y_pto)

    return y


def test_swiglu(device_id = None, run_mode: str = "npu", dynamic: bool = False) -> None:
    """Test SwiGLU activation."""
    print("=" * 60)
    print("Test: SwiGLU Activation")
    print("=" * 60)

    device = f'npu:{device_id}' if (run_mode == "npu" and device_id is not None) else 'cpu'

    shape = (32, 128)
    gate_torch = torch.randn(shape, dtype=torch.bfloat16, device=device)
    up_torch = torch.randn(shape, dtype=torch.bfloat16, device=device)

    # Execute
    out_torch = swiglu_activation(gate_torch, up_torch, run_mode, dynamic)

    # Verify
    expected = swiglu_golden(gate_torch, up_torch)
    max_diff = (out_torch - expected).abs().max().item()

    print(f"Gate shape: {gate_torch.shape}")
    print(f"Up shape: {up_torch.shape}")
    print(f"Output shape: {out_torch.shape}")
    if run_mode == "npu":
        print(f"Max difference: {max_diff:.6f}")
        assert max_diff < 1e-1, "Result mismatch!"
    print("✓ SwiGLU passed")
    print()


@pypto.jit
def geglu_activation_kernel_npu(gate: pypto.tensor, up: pypto.tensor, y: pypto.tensor) -> None:
    """
    GeGLU activation function: GELU(gate) * up

    GeGLU is a gated linear unit that uses GELU as the gating function.
    It's an alternative to SwiGLU.

    Formula: GeGLU(gate, up) = GELU(gate) * up

    Parameters
    ----------
    gate : pypto.tensor
        Gate tensor
    up : pypto.tensor
        Up projection tensor

    Returns
    -------
    pypto.tensor
        GeGLU activated tensor
    """
    # Configure tiling
    if len(gate.shape) >= 2:
        n_tile = 32
        tile_shapes = [n_tile for _ in range(len(gate.shape))]
        pypto.set_vec_tile_shapes(*tile_shapes)
    else:
        pypto.set_vec_tile_shapes(32, 128)

    # GELU approximation: x * sigmoid(1.702 * x)
    coeff = float(1.702)
    x_scaled = gate * coeff

    # GELU(x) = x * sigmoid(1.702 * x)
    gelu_gate =  gate * pypto.sigmoid(x_scaled)

    # Multiply with up projection
    y[:] =  gelu_gate * up


@pypto.jit(runtime_options={"run_mode": 1})
def geglu_activation_kernel_sim(gate: pypto.tensor, up: pypto.tensor, y: pypto.tensor) -> None:
    """
    GeGLU activation function: GELU(gate) * up

    GeGLU is a gated linear unit that uses GELU as the gating function.
    It's an alternative to SwiGLU.

    Formula: GeGLU(gate, up) = GELU(gate) * up

    Parameters
    ----------
    gate : pypto.tensor
        Gate tensor
    up : pypto.tensor
        Up projection tensor

    Returns
    -------
    pypto.tensor
        GeGLU activated tensor
    """
    # Configure tiling
    if len(gate.shape) >= 2:
        n_tile = 32
        tile_shapes = [n_tile for _ in range(len(gate.shape))]
        pypto.set_vec_tile_shapes(*tile_shapes)
    else:
        pypto.set_vec_tile_shapes(32, 128)

    # GELU approximation: x * sigmoid(1.702 * x)
    coeff = float(1.702)
    x_scaled = gate * coeff

    # GELU(x) = x * sigmoid(1.702 * x)
    gelu_gate =  gate * pypto.sigmoid(x_scaled)

    # Multiply with up projection
    y[:] =  gelu_gate * up


def geglu_activation(gate: pypto.tensor, up: pypto.tensor, run_mode: str = "npu", dynamic: bool = False) -> torch.Tensor:
    y = torch.empty_like(gate)

    if dynamic:
        gate_pto = pypto.from_torch(gate, dynamic_axis=[0])
        up_pto = pypto.from_torch(up, dynamic_axis=[0])
        y_pto = pypto.from_torch(y, dynamic_axis=[0])
    else:
        gate_pto = pypto.from_torch(gate)
        up_pto = pypto.from_torch(up)
        y_pto = pypto.from_torch(y)

    # launch the kernel
    if run_mode == "npu":
        geglu_activation_kernel_npu(gate_pto, up_pto, y_pto)
    else:
        geglu_activation_kernel_sim(gate_pto, up_pto, y_pto)
    return y


def test_geglu(device_id = None, run_mode: str = "npu", dynamic: bool = False) -> None:
    """Test GeGLU activation."""
    print("=" * 60)
    print("Test: GeGLU Activation")
    print("=" * 60)

    device = f'npu:{device_id}' if (run_mode == "npu" and device_id is not None) else 'cpu'

    shape = (32, 128)
    gate_torch = torch.randn(shape, dtype=torch.bfloat16, device=device)
    up_torch = torch.randn(shape, dtype=torch.bfloat16, device=device)

    # Execute
    out_torch = geglu_activation(gate_torch, up_torch, run_mode, dynamic)

    # Verify
    expected = geglu_golden(gate_torch, up_torch)
    max_diff = (out_torch - expected).abs().max().item()

    print(f"Gate shape: {gate_torch.shape}")
    print(f"Up shape: {up_torch.shape}")
    print(f"Output shape: {out_torch.shape}")
    if run_mode == "npu":
        print(f"Max difference: {max_diff:.6f}")
        assert max_diff < 1e-1, "Result mismatch!"
    print("✓ GeGLU passed")
    print()


def main():
    """Run custom activation examples.

    Usage:
        python custom_activation.py          # Run all examples
        python custom_activation.py 1         # Run example 1 only
        python custom_activation.py --list   # List all available examples
    """
    parser = argparse.ArgumentParser(
        description="PyPTO Custom Activation Functions Examples",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s              Run all examples
  %(prog)s silu::test_silu
            Run example silu::test_silu
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
        'gelu::test_gelu': {
            'name': 'GELU Activation',
            'description': 'Gaussian Error Linear Unit activation',
            'function': test_gelu,
        },
        'silu::test_silu': {
            'name': 'SiLU Activation',
            'description': 'Sigmoid Linear Unit (Swish) activation',
            'function': test_silu,
        },
        'swiglu::test_swiglu': {
            'name': 'SwiGLU Activation',
            'description': 'Swish-Gated Linear Unit activation',
            'function': test_swiglu,
        },
        'geglu::test_geglu': {
            'name': 'GeGLU Activation',
            'description': 'GELU-Gated Linear Unit activation',
            'function': test_geglu,
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
    print("PyPTO Custom Activation Functions Examples")
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
            print("All custom activation tests passed!")
            print("=" * 60)

    except Exception as e:
        print(f"\nError: {e}")
        raise


if __name__ == "__main__":
    main()
