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
Test and Example Usage of FFN Module

This script demonstrates how to use the FFN module with different configurations
and validates the implementation against PyTorch reference.
"""

import os
import sys
import argparse
import math
from dataclasses import dataclass
from typing import Literal
import pypto
import pytest
import torch
import numpy as np
from numpy.testing import assert_allclose




# Constants
F_1 = 1.0
F_NEGA_1 = -1.0
GELU_COEFF = 1.702


@dataclass
class FFNConfig:
    """Configuration for FFN module"""
    batch_size: int
    hidden_size: int
    intermediate_size: int
    activation: Literal["gelu", "swiglu", "relu"] = "gelu"
    dtype: pypto.DataType = pypto.DT_FP16
    use_dynamic_shape: bool = False
    vec_tile_shape: tuple = (64, 128)
    cube_tile_shape: tuple = (64, 128, 128)
    basic_batch: int = 32  # For dynamic batching
    run_mode: pypto.RunMode = pypto.RunMode.NPU


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


def gelu_torch(x):
    """PyTorch reference for GELU"""
    return x * torch.sigmoid(1.702 * x)


def swiglu_torch(gate, up):
    """PyTorch reference for SwiGLU."""
    swish = gate * torch.sigmoid(gate)
    return swish * up   


def ceil_div(a, b):
    """Calculate ceiling division: (a + b - 1) // b"""
    return (a + b - 1) // b


def relu_activation_core(x: pypto.tensor) -> pypto.tensor:
    """
    ReLU activation function: max(0, x)

    Parameters
    ----------
    x : pypto.tensor
        Input tensor

    Returns
    -------
    pypto.tensor
        ReLU activated tensor
    """
    pypto.set_vec_tile_shapes(*x.shape[:2] if len(x.shape) >= 2 else (32, 128))
    zero = pypto.full(x.shape, 0, x.dtype, valid_shape=x.shape)
    return pypto.maximum(x, zero)


def gelu_activation_core(x: pypto.tensor) -> pypto.tensor:
    """
    GELU activation function: x * 0.5 * (1 + erf(x / sqrt(2)))

    Approximated as: x * 0.5 * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))

    Parameters
    ----------
    x : pypto.tensor
        Input tensor

    Returns
    -------
    pypto.tensor
        GELU activated tensor
    """
    pypto.set_vec_tile_shapes(*x.shape[:2] if len(x.shape) >= 2 else (32, 128))
    x_scaled = pypto.mul(x, GELU_COEFF)
    x_neg = pypto.mul(x_scaled, F_NEGA_1)
    exp_neg = pypto.exp(x_neg)
    ones = pypto.full(exp_neg.shape, 1.0, exp_neg.dtype, valid_shape=exp_neg.shape)
    sigmoid = pypto.div(ones, pypto.add(exp_neg, F_1))
    return pypto.mul(x, sigmoid)


def swiglu_activation_core(gate: pypto.tensor, up: pypto.tensor) -> pypto.tensor:
    """
    SwiGLU activation function: Swish(gate) * up
    where Swish(x) = x * sigmoid(x)

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
    pypto.set_vec_tile_shapes(*gate.shape[:2] if len(gate.shape) >= 2 else (32, 128))

    gate_neg = pypto.mul(gate, F_NEGA_1)
    exp_neg = pypto.exp(gate_neg)
    ones = pypto.full(exp_neg.shape, F_1, exp_neg.dtype, valid_shape=exp_neg.shape)
    sigmoid = pypto.div(ones, pypto.add(exp_neg, ones))
    swish = pypto.mul(gate, sigmoid)

    # Multiply with up projection
    return pypto.mul(swish, up)


def dynamic_gelu_activation_core(output: pypto.tensor, hidden_states: pypto.tensor, 
    gate_proj_weight: pypto.tensor, down_proj_weight: pypto.tensor, config: FFNConfig) -> None:
    hidden_size, intermediate_size = config.hidden_size, config.intermediate_size
    basic_batch = config.basic_batch
    if basic_batch == 0:
        raise ValueError("basic_batch must be greater than 0")
    # Calculate number of iterations needed
    batch_size = hidden_states.shape[0]
    num_iterations = ceil_div(batch_size, basic_batch)
    # Process in chunks
    for idx in pypto.loop(0, num_iterations, 1, name="LOOP_FFN_BATCH", idx_name="idx"):
        batch_offset = idx * basic_batch
        # View current batch chunk
        hidden_chunk = pypto.view(
            hidden_states,
            [basic_batch, hidden_size],
            [batch_offset, 0],
            valid_shape=[(batch_size - batch_offset).min(basic_batch), hidden_size]
        )
        # Configure tiling for matrix operations
        pypto.set_matrix_size([basic_batch, hidden_size, intermediate_size])
        # Gate projection
        gate = pypto.matmul(hidden_chunk, gate_proj_weight, config.dtype)
        pypto.set_vec_tile_shapes(*config.vec_tile_shape)
        activated = gelu_activation_core(gate)
        # Down projection
        pypto.set_cube_tile_shapes(
            [config.cube_tile_shape[0], config.cube_tile_shape[0]],
            [config.cube_tile_shape[1], config.cube_tile_shape[1]],
            [config.cube_tile_shape[2], config.cube_tile_shape[2]]
        )
        pypto.set_matrix_size([basic_batch, intermediate_size, hidden_size])
        output_chunk = pypto.matmul(activated, down_proj_weight, config.dtype, b_trans=False)
        # Assemble result back to output
        pypto.assemble(output_chunk, [batch_offset, 0], output)
    return


def ffn(config: FFNConfig):

    batch_size, hidden_size, intermediate_size = config.batch_size, config.hidden_size, config.intermediate_size
    
    @pypto.frontend.jit(runtime_options={"run_mode": config.run_mode})
    def ffn_activation_kernel(
        hidden_states: pypto.tensor((batch_size, hidden_size), config.dtype),
        gate_proj_weight: pypto.tensor((hidden_size, intermediate_size), config.dtype),
        up_proj_weight: pypto.tensor((hidden_size, intermediate_size), config.dtype),
        down_proj_weight: pypto.tensor((intermediate_size, hidden_size), config.dtype),
    ) -> pypto.tensor((batch_size, hidden_size), config.dtype):
        # Configure tiling for matrix operations
        pypto.set_cube_tile_shapes(
            [config.cube_tile_shape[0], config.cube_tile_shape[0]],
            [config.cube_tile_shape[1], config.cube_tile_shape[1]],
            [config.cube_tile_shape[2], config.cube_tile_shape[2]]
        )
        pypto.set_vec_tile_shapes(*config.vec_tile_shape)

        # Gate projection: [batch_size, hidden_size] @ [hidden_size, intermediate_size]
        gate = pypto.matmul(hidden_states, gate_proj_weight, config.dtype)
        
        if config.use_dynamic_shape == True and config.activation == "gelu":
            # Dynamic GELU activation
            output = pypto.tensor(hidden_states.shape, config.dtype)
            dynamic_gelu_activation_core(output, hidden_states, gate_proj_weight, down_proj_weight, config)
        elif config.activation == "gelu":
            # GELU activation
            activated = gelu_activation_core(gate)
        elif config.activation == "swiglu":
            # SwiGLU activation
            up_proj_weight = pypto.matmul(hidden_states, up_proj_weight, config.dtype)
            activated = swiglu_activation_core(gate, up_proj_weight)
        elif config.activation == "relu":
            # ReLU activation
            activated = relu_activation_core(gate) 
        else:
            raise ValueError(f"Unsupported activation: {config.activation}")

        if config.use_dynamic_shape == False:
            output = pypto.matmul(activated, down_proj_weight, config.dtype, b_trans=False)
        return output
    
    return ffn_activation_kernel


@pytest.mark.skip
def test_ffn_static_gelu(device_id=None, run_mode: str = "npu"):
    """Test static FFN with GELU activation."""
    print("=" * 60)
    print("Testing Static FFN with GELU Activation")
    print("=" * 60)

    batch_size = 16
    hidden_size = 128
    intermediate_size = 1024
    dtype = torch.bfloat16
    device = f'npu:{device_id}' if (run_mode == "npu" and device_id is not None) else 'cpu'

    config = FFNConfig(
        batch_size=batch_size,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        activation="gelu",
        dtype=pypto.DT_BF16,
        use_dynamic_shape=False,
        vec_tile_shape=(16, 32),
        cube_tile_shape=(16, 32, 32),
        run_mode=pypto.RunMode.NPU if run_mode == "npu" else pypto.RunMode.SIM
    )

    hidden_states_torch = torch.randn(batch_size, hidden_size, 
                                        dtype=dtype, device=device) / math.sqrt(batch_size)
    gate_proj_weight_torch = torch.randn(hidden_size, intermediate_size, 
                                        dtype=dtype, device=device) / math.sqrt(batch_size)
    up_proj_weight_torch = torch.randn(hidden_size, intermediate_size, 
                                        dtype=dtype, device=device) / math.sqrt(batch_size)
    down_proj_weight_torch = torch.randn(intermediate_size, hidden_size, 
                                        dtype=dtype, device=device) / math.sqrt(batch_size)

    print(f"Input shape: {hidden_states_torch.shape}")
    print(f"Gate weight shape: {gate_proj_weight_torch.shape}")
    print(f"up weight shape: {up_proj_weight_torch.shape}")
    print(f"Down weight shape: {down_proj_weight_torch.shape}")
    gate_torch = torch.matmul(hidden_states_torch, gate_proj_weight_torch)
    gate_activated_torch = gelu_torch(gate_torch.float()).to(dtype)
    output_torch_ref = torch.matmul(gate_activated_torch, down_proj_weight_torch)

    output = ffn(config)(hidden_states_torch, gate_proj_weight_torch, up_proj_weight_torch, down_proj_weight_torch)
    if run_mode == "npu":
        assert_allclose(output.cpu().to(torch.float32), output_torch_ref.cpu().to(torch.float32), rtol=3e-3, atol=3e-3)
    print(f"Output shape: {output_torch_ref.shape}")
    print(f"Output range: [{output_torch_ref.min().item():.4f}, {output_torch_ref.max().item():.4f}]")
    print("✓ Static FFN with GELU test completed")
    print()


def test_ffn_static_swiglu(device_id=None, run_mode: str = "npu"):
    """Test static FFN with SwiGLU activation."""
    print("=" * 60)
    print("Testing Static FFN with SwiGLU Activation")
    print("=" * 60)

    batch_size = 16
    hidden_size = 128
    intermediate_size = 1024
    dtype = torch.bfloat16
    device = f'npu:{device_id}' if (run_mode == "npu" and device_id is not None) else 'cpu'
    config = FFNConfig(
        batch_size=batch_size,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        activation="swiglu",
        dtype=pypto.DT_BF16,
        use_dynamic_shape=False,
        vec_tile_shape=(16, 32),
        cube_tile_shape=(16, 32, 32),
        run_mode=pypto.RunMode.NPU if run_mode == "npu" else pypto.RunMode.SIM
    )
    # Create PyTorch tensors
    hidden_states_torch = torch.randn(batch_size, hidden_size, 
                                        dtype=dtype, device=device) / math.sqrt(batch_size)
    gate_proj_weight_torch = torch.randn(hidden_size, intermediate_size, 
                                        dtype=dtype, device=device) / math.sqrt(batch_size)
    up_proj_weight_torch = torch.randn(hidden_size, intermediate_size, 
                                        dtype=dtype, device=device) / math.sqrt(batch_size)
    down_proj_weight_torch = torch.randn(intermediate_size, hidden_size, 
                                        dtype=dtype, device=device) / math.sqrt(batch_size)
    
    # PyTorch reference computation
    gate_torch = torch.matmul(hidden_states_torch, gate_proj_weight_torch)
    up_torch = torch.matmul(hidden_states_torch, up_proj_weight_torch)
    activated_torch = swiglu_torch(gate_torch.float(), up_torch.float()).to(dtype)
    output_torch_ref = torch.matmul(activated_torch, down_proj_weight_torch)

    output = ffn(config)(hidden_states_torch, gate_proj_weight_torch, up_proj_weight_torch, down_proj_weight_torch)
    print(f"Input shape: {hidden_states_torch.shape}")
    print(f"Gate weight shape: {gate_proj_weight_torch.shape}")
    print(f"Up weight shape: {up_proj_weight_torch.shape}")
    print(f"Down weight shape: {down_proj_weight_torch.shape}")
    print(f"Output shape: {output_torch_ref.shape}")
    print(f"Output range: [{output_torch_ref.min().item():.4f}, {output_torch_ref.max().item():.4f}]")

    if run_mode == "npu":
        assert_allclose(output.cpu().to(torch.float32), output_torch_ref.cpu().to(torch.float32), rtol=3e-3, atol=3e-3)
    print("✓ Static FFN with SwiGLU test completed")
    print()


@pytest.mark.skip
def test_ffn_dynamic_gelu(device_id: int = None, run_mode: str = "npu", dynamic: bool = True):
    """Test dynamic FFN with GELU activation."""
    print("=" * 60)
    print("Testing Dynamic FFN with GELU Activation")
    print("=" * 60)

    batch_size = 32  # Non-power-of-2 to test dynamic handling
    hidden_size = 512
    intermediate_size = 1024
    basic_batch = 16
    dtype = torch.bfloat16
    device = f'npu:{device_id}' if (run_mode == "npu" and device_id is not None) else 'cpu'

    config = FFNConfig(
        batch_size=batch_size,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        activation="gelu",
        dtype=pypto.DT_BF16,
        use_dynamic_shape=True,
        vec_tile_shape=(32, 64),
        cube_tile_shape=(32, 64, 64),
        basic_batch=basic_batch,
        run_mode=pypto.RunMode.NPU if run_mode == "npu" else pypto.RunMode.SIM
    )

    # Create PyTorch tensors
    hidden_states_torch = torch.randn(
        batch_size, hidden_size, dtype=dtype, device=device) / math.sqrt(batch_size)
    gate_proj_weight_torch = torch.randn(
        hidden_size, intermediate_size, dtype=dtype, device=device) / math.sqrt(batch_size)
    up_proj_weight_torch = torch.randn(
        hidden_size, intermediate_size, dtype=dtype, device=device) / math.sqrt(batch_size)
    down_proj_weight_torch = torch.randn(
        intermediate_size, hidden_size, dtype=dtype, device=device) / math.sqrt(batch_size)
    
    # PyTorch reference computation
    gate_torch = torch.matmul(hidden_states_torch, gate_proj_weight_torch)
    gate_activated_torch = gelu_torch(gate_torch.float()).to(dtype)
    output_torch_ref = torch.matmul(gate_activated_torch, down_proj_weight_torch)

    print(f"Input shape: {hidden_states_torch.shape} (dynamic batch size: {batch_size})")
    print(f"Basic batch size: {basic_batch}")
    print(f"Number of iterations: {(batch_size + basic_batch - 1) // basic_batch}")
    print(f"Output shape: {output_torch_ref.shape}")
    print(f"Output range: [{output_torch_ref.min().item():.4f}, {output_torch_ref.max().item():.4f}]")
    
    output = ffn(config)(hidden_states_torch, gate_proj_weight_torch, up_proj_weight_torch, down_proj_weight_torch)
    if run_mode == "npu":
        assert_allclose(output.cpu().to(torch.float32), output_torch_ref.cpu().to(torch.float32), rtol=3e-3, atol=3e-3)
    
    print("✓ Dynamic FFN with GELU test completed")
    print()


@pytest.mark.skip
def test_ffn_static_relu(device_id: int = None, run_mode: str = "npu", dynamic: bool = True):
    """Test static FFN with ReLU activation."""
    print("=" * 60)
    print("Testing Static FFN with ReLU Activation")
    print("=" * 60)

    batch_size = 16
    hidden_size = 128
    intermediate_size = 1024
    dtype = torch.float16
    device = f'npu:{device_id}' if (run_mode == "npu" and device_id is not None) else 'cpu'

    config = FFNConfig(
        batch_size=batch_size,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        activation="relu",
        dtype=pypto.DT_FP16,
        use_dynamic_shape=False,
        vec_tile_shape=(32, 64),
        cube_tile_shape=(32, 64, 64),
        run_mode=pypto.RunMode.NPU if run_mode == "npu" else pypto.RunMode.SIM
    )

    # Create PyTorch tensors
    hidden_states_torch = torch.randn(batch_size, hidden_size,
                                      dtype=dtype, device=device) / math.sqrt(batch_size)
    gate_proj_weight_torch = torch.randn(hidden_size, intermediate_size, 
                                        dtype=dtype, device=device) / math.sqrt(batch_size)
    up_proj_weight_torch = torch.randn(hidden_size, intermediate_size, 
                                        dtype=dtype, device=device) / math.sqrt(batch_size)
    down_proj_weight_torch = torch.randn(intermediate_size, hidden_size, 
                                        dtype=dtype, device=device) / math.sqrt(batch_size)
    
    # PyTorch reference computation
    gate_torch = torch.matmul(hidden_states_torch, gate_proj_weight_torch)
    gate_activated_torch = torch.relu(gate_torch)
    output_torch_ref = torch.matmul(gate_activated_torch, down_proj_weight_torch)
    
    output = ffn(config)(hidden_states_torch, gate_proj_weight_torch, up_proj_weight_torch, down_proj_weight_torch)
    max_diff = np.abs((output.cpu().numpy() - output_torch_ref.cpu().numpy())).max()
    print(f"Input shape: {hidden_states_torch.shape}")
    print(f"Output shape: {output_torch_ref.shape}")
    print(f"Output range: [{output_torch_ref.min().item():.4f}, {output_torch_ref.max().item():.4f}]")
    print(f"Max difference: {max_diff:.6f}")
    if run_mode == "npu":
        assert_allclose(output.cpu().to(torch.float32), output_torch_ref.cpu().to(torch.float32), rtol=3e-3, atol=3e-3)
    print("✓ Static FFN with ReLU test completed")
    print()


def main():
    """Run FFN module examples.

    Usage:
        python ffn_module_example.py          # Run all examples
        python ffn_module_example.py 1         # Run example 1 only
        python ffn_module_example.py --list   # List all available examples
    """
    parser = argparse.ArgumentParser(
        description="FFN Module Test Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s              Run all examples
  %(prog)s ffn_static_gelu::test_ffn_static_gelu
            Run example ffn_static_gelu::test_ffn_static_gelu
  %(prog)s --list       List all available examples
        """
    )
    parser.add_argument(
        'example_id',
        type=str,
        nargs='?',
        help='Example ID to run (1-5). If not specified, all examples will run.'
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
        'ffn_static_gelu::test_ffn_static_gelu': {
            'name': 'Static FFN with GELU',
            'description': 'Static FFN with GELU activation',
            'function': test_ffn_static_gelu
        },
        'ffn_static_swiglu::test_ffn_static_swiglu': {
            'name': 'Static FFN with SwiGLU',
            'description': 'Static FFN with SwiGLU activation',
            'function': test_ffn_static_swiglu
        },
        'ffn_static_relu::test_ffn_static_relu': {
            'name': 'Static FFN with ReLU',
            'description': 'Static FFN with ReLU activation',
            'function': test_ffn_static_relu
        },
        'ffn_dynamic_gelu::test_ffn_dynamic_gelu': {
            'name': 'Dynamic FFN with GELU',
            'description': 'Dynamic FFN with GELU activation',
            'function': test_ffn_dynamic_gelu
        },
    }

    # List examples if requested
    if args.list:
        print("\n" + "=" * 60)
        print("Available Examples")
        print("=" * 60 + "\n")
        for ex_id, ex_info in sorted(examples.items()):
            print(f"  ID: {ex_id}")
            print(f"    name: {ex_info['name']}")
            print(f"    description: {ex_info['description']}\n")
        return

    # Validate example ID if provided
    if args.example_id is not None:
        if args.example_id not in examples:
            print(f"ERROR: Invalid example ID: {args.example_id}")
            print(f"Valid example IDs are: {', '.join(map(str, sorted(examples.keys())))}")
            print("\nUse --list to see all available examples.")
            sys.exit(1)

    print("\n" + "=" * 60)
    print("FFN Module Test Suite")
    print("=" * 60 + "\n")

    # Get and validate device ID (needed for NPU examples)
    device_id = None
    examples_to_run = []

    if args.example_id is not None:
        example = examples.get(args.example_id)
        if example is None:
            raise ValueError(f"Invalid example ID: {args.example_id}")
        examples_to_run = [(args.example_id, example)]
    else:
        examples_to_run = list(examples.items())

    if args.run_mode == "npu":
        device_id = get_device_id()
        if device_id is None:
            return
        import torch_npu
        torch.npu.set_device(device_id)
        print("Running examples that require NPU hardware...")
        print("Make sure CANN environment is configured and NPU is available\n")

    try:
        for ex_id, ex_info in examples_to_run:
            print(f"Running Example {ex_id}: {ex_info['name']}")
            ex_info['function'](device_id, args.run_mode)

        if len(examples_to_run) > 1:
            print("=" * 60)
            print("All tests completed!")
            print("=" * 60)

    except Exception as e:
        print(f"\nError: {e}")
        raise

if __name__ == "__main__":
    main()
