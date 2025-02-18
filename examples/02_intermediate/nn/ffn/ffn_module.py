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
import pypto
import torch
import numpy as np
from numpy.testing import assert_allclose
import math
from ffn_module_impl import (
    FFNConfig,
    ffn_static_gule_kernel_npu,
    ffn_static_gule_kernel_sim,
    ffn_static_relu_kernel_npu,
    ffn_static_relu_kernel_sim,
    ffn_static_swiglu_kernel_npu,
    ffn_static_swiglu_kernel_sim,
    ffn_dynamic_gelu_kernel_npu,
    ffn_dynamic_gelu_kernel_sim
)

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

def ffn(hidden_states_torch: torch.Tensor, gate_proj_weight_torch: torch.Tensor, up_proj_weight_torch: torch.Tensor, down_proj_weight_torch: torch.Tensor, config: FFNConfig, use_dynamic: bool, run_mode: str = "npu") -> torch.Tensor:
    hidden_states = pypto.from_torch(hidden_states_torch)
    gate_proj_weight = pypto.from_torch(gate_proj_weight_torch)
    up_proj_weight = pypto.from_torch(up_proj_weight_torch)
    down_proj_weight = pypto.from_torch(down_proj_weight_torch) 
    output_torch = torch.zeros(hidden_states_torch.shape, dtype=hidden_states_torch.dtype, device=hidden_states_torch.device)
    output = pypto.from_torch(output_torch) 
    
    # ffn_module = create_ffn_module(config, use_dynamic=use_dynamic)
    if use_dynamic == False:
        if config.activation == "gelu":
            if run_mode == "npu":
                ffn_static_gule_kernel_npu(hidden_states, gate_proj_weight,  down_proj_weight, output, config)
            else:
                ffn_static_gule_kernel_sim(hidden_states, gate_proj_weight,  down_proj_weight, output, config)
        if config.activation == "swiglu":
            if run_mode == "npu":
                ffn_static_swiglu_kernel_npu(hidden_states, gate_proj_weight, up_proj_weight, down_proj_weight, output, config)
            else:
                ffn_static_swiglu_kernel_sim(hidden_states, gate_proj_weight, up_proj_weight, down_proj_weight, output, config)
        if config.activation == "relu":
            if run_mode == "npu":
                ffn_static_relu_kernel_npu(hidden_states, gate_proj_weight, down_proj_weight, output, config)
            else:
                ffn_static_relu_kernel_sim(hidden_states, gate_proj_weight, down_proj_weight, output, config)
    elif use_dynamic == True:
        if run_mode == "npu":
            ffn_dynamic_gelu_kernel_npu(hidden_states, gate_proj_weight, down_proj_weight, output, config)
        else:
            ffn_dynamic_gelu_kernel_sim(hidden_states, gate_proj_weight, down_proj_weight, output, config)
    return output_torch    

def test_ffn_static_gelu(device_id = None, run_mode: str = "npu", dynamic: bool = True):
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
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        activation="gelu",
        dtype=pypto.DT_BF16,
        use_dynamic_shape=False,
        vec_tile_shape=(16, 32),
        cube_tile_shape=(16, 32, 32)
    )

    hidden_states_torch = torch.randn(batch_size, hidden_size, dtype=dtype, device=device) / math.sqrt(batch_size)
    gate_proj_weight_torch = torch.randn(hidden_size, intermediate_size, dtype=dtype, device=device) / math.sqrt(batch_size)
    up_proj_weight_torch = torch.randn(hidden_size, intermediate_size, dtype=dtype, device=device) / math.sqrt(batch_size)
    down_proj_weight_torch = torch.randn(intermediate_size, hidden_size, dtype=dtype, device=device) / math.sqrt(batch_size)

    print(f"Input shape: {hidden_states_torch.shape}")
    print(f"Gate weight shape: {gate_proj_weight_torch.shape}")
    print(f"up weight shape: {up_proj_weight_torch.shape}")
    print(f"Down weight shape: {down_proj_weight_torch.shape}")
    gate_torch = torch.matmul(hidden_states_torch, gate_proj_weight_torch)
    gate_activated_torch = gelu_torch(gate_torch.float()).to(dtype)
    output_torch_ref = torch.matmul(gate_activated_torch, down_proj_weight_torch)

    output = ffn(hidden_states_torch, gate_proj_weight_torch, up_proj_weight_torch, down_proj_weight_torch, config, False, run_mode)
    if run_mode == "npu":
        assert_allclose(output.cpu().to(torch.float32), output_torch_ref.cpu().to(torch.float32), rtol=3e-3, atol=3e-3)
    print(f"Output shape: {output_torch_ref.shape}")
    print(f"Output range: [{output_torch_ref.min().item():.4f}, {output_torch_ref.max().item():.4f}]")
    print("✓ Static FFN with GELU test completed")
    print()

def test_ffn_static_swiglu(device_id = None, run_mode: str = "npu", dynamic: bool = True):
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
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        activation="swiglu",
        dtype=pypto.DT_BF16,
        use_dynamic_shape=False,
        vec_tile_shape=(16, 32),
        cube_tile_shape=(16, 32, 32)
    )
    # Create PyTorch tensors
    hidden_states_torch = torch.randn(batch_size, hidden_size, dtype=dtype, device=device) / math.sqrt(batch_size)
    gate_proj_weight_torch = torch.randn(hidden_size, intermediate_size, dtype=dtype, device=device) / math.sqrt(batch_size)
    up_proj_weight_torch = torch.randn(hidden_size, intermediate_size, dtype=dtype, device=device) / math.sqrt(batch_size)
    down_proj_weight_torch = torch.randn(intermediate_size, hidden_size, dtype=dtype, device=device) / math.sqrt(batch_size)
    
    # PyTorch reference computation
    gate_torch = torch.matmul(hidden_states_torch, gate_proj_weight_torch)
    up_torch = torch.matmul(hidden_states_torch, up_proj_weight_torch)
    activated_torch = swiglu_torch(gate_torch.float(), up_torch.float()).to(dtype)
    output_torch_ref = torch.matmul(activated_torch, down_proj_weight_torch)

    output = ffn(hidden_states_torch, gate_proj_weight_torch, up_proj_weight_torch, down_proj_weight_torch, config, False, run_mode)
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

def test_ffn_dynamic_gelu(device_id = None, run_mode: str = "npu", dynamic: bool = True):
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
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        activation="gelu",
        dtype=pypto.DT_BF16,
        use_dynamic_shape=True,
        vec_tile_shape=(32, 64),
        cube_tile_shape=(32, 64, 64),
        basic_batch=basic_batch
    )

    # Create PyTorch tensors
    hidden_states_torch = torch.randn(batch_size, hidden_size, dtype=dtype, device=device) / math.sqrt(batch_size)
    gate_proj_weight_torch = torch.randn(hidden_size, intermediate_size, dtype=dtype, device=device) / math.sqrt(batch_size)
    up_proj_weight_torch = torch.randn(hidden_size, intermediate_size, dtype=dtype, device=device) / math.sqrt(batch_size)
    down_proj_weight_torch = torch.randn(intermediate_size, hidden_size, dtype=dtype, device=device) / math.sqrt(batch_size)
    
    # PyTorch reference computation
    gate_torch = torch.matmul(hidden_states_torch, gate_proj_weight_torch)
    gate_activated_torch = gelu_torch(gate_torch.float()).to(dtype)
    output_torch_ref = torch.matmul(gate_activated_torch, down_proj_weight_torch)

    print(f"Input shape: {hidden_states_torch.shape} (dynamic batch size: {batch_size})")
    print(f"Basic batch size: {basic_batch}")
    print(f"Number of iterations: {(batch_size + basic_batch - 1) // basic_batch}")
    print(f"Output shape: {output_torch_ref.shape}")
    print(f"Output range: [{output_torch_ref.min().item():.4f}, {output_torch_ref.max().item():.4f}]")
    
    output = ffn(hidden_states_torch, gate_proj_weight_torch, up_proj_weight_torch, down_proj_weight_torch, config, True, run_mode)
    if run_mode == "npu":
        assert_allclose(output.cpu().to(torch.float32), output_torch_ref.cpu().to(torch.float32), rtol=3e-3, atol=3e-3)
    
    print("✓ Dynamic FFN with GELU test completed")
    print()

def test_ffn_static_relu(device_id = None, run_mode: str = "npu", dynamic: bool = True):
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
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        activation="relu",
        dtype=pypto.DT_FP16,
        use_dynamic_shape=False,
        vec_tile_shape=(32, 64),
        cube_tile_shape=(32, 64, 64)
    )

    # Create PyTorch tensors
    hidden_states_torch = torch.randn(batch_size, hidden_size, dtype=dtype, device=device) / math.sqrt(batch_size)
    gate_proj_weight_torch = torch.randn(hidden_size, intermediate_size, dtype=dtype, device=device) / math.sqrt(batch_size)
    up_proj_weight_torch = torch.randn(hidden_size, intermediate_size, dtype=dtype, device=device) / math.sqrt(batch_size)
    down_proj_weight_torch = torch.randn(intermediate_size, hidden_size, dtype=dtype, device=device) / math.sqrt(batch_size)
    
    # PyTorch reference computation
    gate_torch = torch.matmul(hidden_states_torch, gate_proj_weight_torch)
    gate_activated_torch = torch.relu(gate_torch)
    output_torch_ref = torch.matmul(gate_activated_torch, down_proj_weight_torch)
    
    output = ffn(hidden_states_torch, gate_proj_weight_torch, up_proj_weight_torch, down_proj_weight_torch, config, False, run_mode)
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

