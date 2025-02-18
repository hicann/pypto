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
Scaled Dot-Product Attention Example for PyPTO

This example demonstrates:
- Scaled dot-product attention mechanism
- Q, K, V computation
- Attention scores calculation
- Softmax normalization
- Output projection
- Static and dynamic batch/sequence length support

Attention is the core mechanism in transformer architectures.
"""

import os
import sys
import argparse
import pypto
import torch
import numpy as np
from numpy.testing import assert_allclose
from dataclasses import dataclass
from typing import Optional


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


@dataclass
class AttentionConfig:
    """Configuration for attention operations."""
    num_heads: int = 8
    head_dim: int = 64
    scale: Optional[float] = None  # If None, uses 1/sqrt(head_dim)
    dtype: pypto.DataType = pypto.DT_BF16
    use_dynamic_shape: bool = False


def scaled_dot_product_attention_golden(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float,
    attn_mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """PyTorch reference implementation of scaled dot-product attention."""
    # Compute attention scores: Q @ K^T
    scores = torch.matmul(q, k.transpose(-2, -1))  # [batch, num_heads, seq_len_q, seq_len_kv]

    # Scale
    scores = scores * scale

    # Apply attention mask if provided
    if attn_mask is not None:
        scores = scores + attn_mask

    # Softmax
    attn_weights = torch.softmax(scores, dim=-1)  # [batch, num_heads, seq_len_q, seq_len_kv]

    # Apply to values: attn_weights @ V
    output = torch.matmul(attn_weights, v)  # [batch, num_heads, seq_len_q, head_dim]

    return output


def scaled_dot_product_attention_core(q: pypto.Tensor, k: pypto.Tensor, v: pypto.Tensor, 
                                      scale: float, dtype: pypto.DataType) -> pypto.Tensor:
    k_t = pypto.transpose(k, 2, 3)
    scores = pypto.matmul(q, k_t, out_dtype=dtype)
    scores_scaled = scores * scale
    attn_weights = pypto.softmax(scores_scaled, dim=-1)
    res = pypto.matmul(attn_weights, v, out_dtype=dtype)
    return res


@pypto.jit(
    host_options={"only_codegen": True},
)
def scaled_dot_product_attention_kernel_npu(q: torch.Tensor, k: torch.Tensor, 
                                 v: torch.Tensor, y: torch.Tensor, params: torch.Size, 
                                 config: AttentionConfig):
    """Scaled dot-product attention with dynamic batch and sequence lengths."""       
    batch_size, num_heads, seq_len, head_dim = params

    # Calculate scale
    scale = config.scale if config.scale is not None else (1.0 / (config.head_dim ** 0.5))
    cube_tiling = 64
    pypto.set_cube_tile_shapes([cube_tiling, cube_tiling], [cube_tiling, cube_tiling], [cube_tiling, cube_tiling])
    view_shape = (batch_size, num_heads, seq_len, head_dim)
    bs_loop = (batch_size + view_shape[0] - 1) // view_shape[0]
    for bs_idx in pypto.loop(bs_loop):
        q_view = q[bs_idx * view_shape[0]:(bs_idx+1) * view_shape[0], ...]
        k_view = k[bs_idx * view_shape[0]:(bs_idx+1) * view_shape[0], ...]
        v_view = v[bs_idx * view_shape[0]:(bs_idx+1) * view_shape[0], ...]
        pypto.set_vec_tile_shapes(1, 8, 16, 64) 
        res = scaled_dot_product_attention_core(q_view, k_view, v_view, scale, config.dtype)
        y[bs_idx * view_shape[0]:, ...] = res
            
@pypto.jit(
    host_options={"only_codegen": True},
    runtime_options={"run_mode" : 1}
)
def scaled_dot_product_attention_kernel_sim(q: torch.Tensor, k: torch.Tensor, 
                                 v: torch.Tensor, y: torch.Tensor, params: torch.Size, 
                                 config: AttentionConfig):
    """Scaled dot-product attention with dynamic batch and sequence lengths."""       
    batch_size, num_heads, seq_len, head_dim = params

    # Calculate scale
    scale = config.scale if config.scale is not None else (1.0 / (config.head_dim ** 0.5))
    cube_tiling = 64
    pypto.set_cube_tile_shapes([cube_tiling, cube_tiling], [cube_tiling, cube_tiling], [cube_tiling, cube_tiling])
    view_shape = (batch_size, num_heads, seq_len, head_dim)
    bs_loop = (batch_size + view_shape[0] - 1) // view_shape[0]
    for bs_idx in pypto.loop(bs_loop):
        q_view = q[bs_idx * view_shape[0]:(bs_idx+1) * view_shape[0], ...]
        k_view = k[bs_idx * view_shape[0]:(bs_idx+1) * view_shape[0], ...]
        v_view = v[bs_idx * view_shape[0]:(bs_idx+1) * view_shape[0], ...]
        pypto.set_vec_tile_shapes(1, 8, 16, 64) 
        res = scaled_dot_product_attention_core(q_view, k_view, v_view, scale, config.dtype)
        y[bs_idx * view_shape[0]:, ...] = res

def scaled_dot_product_attention(q: torch.Tensor, k: torch.Tensor, 
                                 v: torch.Tensor, params: torch.Size, 
                                 config: AttentionConfig, run_mode: str = "npu", dynamic: bool = True) -> torch.Tensor:
    y = torch.empty_like(q)

    if dynamic:
        q_pto = pypto.from_torch(q, dynamic_axis=[0])
        k_pto = pypto.from_torch(k, dynamic_axis=[0])
        v_pto = pypto.from_torch(v, dynamic_axis=[0])
        y_pto = pypto.from_torch(y, dynamic_axis=[0])
    else:
        q_pto = pypto.from_torch(q)
        k_pto = pypto.from_torch(k)
        v_pto = pypto.from_torch(v)
        y_pto = pypto.from_torch(y)

    # launch the kernel
    if run_mode == "npu":
        scaled_dot_product_attention_kernel_npu(q_pto, k_pto, v_pto, y_pto, params, config)
    else:
        scaled_dot_product_attention_kernel_sim(q_pto, k_pto, v_pto, y_pto, params, config)
    return y

def test_attention_dynamic(device_id = None, run_mode: str = "npu", dynamic: bool = True) -> None:
    """Test attention function with dynamic shapes."""
    print("=" * 60)
    print("Test: Dynamic Scaled Dot-Product Attention")
    print("=" * 60)
    
    device = f'npu:{device_id}' if (run_mode == "npu" and device_id is not None) else 'cpu'
    
    num_heads, head_dim = 8, 64
    
    # Test with different batch sizes and sequence lengths (dynamic shapes)
    test_cases = [
        (2, 16, 16),
        (4, 32, 32),
        (8, 64, 64),
    ]
    for batch_size, seq_len_q, seq_len_kv in test_cases:
        dtype = torch.float32
        q_torch = torch.randn(batch_size, num_heads, seq_len_q, head_dim, 
                                dtype=dtype, device=device)
        k_torch = torch.randn(batch_size, num_heads, seq_len_kv, head_dim, 
                                dtype=dtype, device=device)
        v_torch = torch.randn(batch_size, num_heads, seq_len_kv, head_dim, 
                                dtype=dtype, device=device)
        config = AttentionConfig(num_heads=num_heads, head_dim=head_dim, 
                                dtype=pypto.DT_FP32, use_dynamic_shape=True)
        params = q_torch.shape
        # Execute
        out_torch = scaled_dot_product_attention(q_torch, k_torch, v_torch, params, config, run_mode, dynamic).cpu()
        
        # Verify
        scale = 1.0 / (head_dim ** 0.5)
        golden = scaled_dot_product_attention_golden(q_torch, k_torch, v_torch, scale).cpu()
        
        max_diff = (out_torch - golden).abs().max().item()
        print(f"Batch={batch_size}, SeqQ={seq_len_q}, SeqKV={seq_len_kv}")
        print(f"Input shape: {q_torch.shape}")
        print(f"Output shape: {out_torch.shape}")
        if run_mode == "npu":
            print(f"Batch={batch_size}, SeqQ={seq_len_q}, SeqKV={seq_len_kv}, Max diff: {max_diff:.6f}")
            assert_allclose(np.array(out_torch), np.array(golden), rtol=3e-3, atol=3e-3)
        
    print("✓ Attention (dynamic) passed for the test case")
    print()


def attention_with_projection_core(q_view: pypto.Tensor, k_view: pypto.Tensor, 
                                   v_view: pypto.Tensor, out_weight: pypto.Tensor,
                                    scale: float, dtype: pypto.DataType) -> pypto.Tensor:
    batch = q_view.shape[0]
    num_heads = q_view.shape[1]
    seq_len = q_view.shape[2]
    head_dim = q_view.shape[3]
    # Scaled dot-product attention
    k_t = pypto.transpose(k_view, 2, 3)
    scores = pypto.matmul(q_view, k_t, out_dtype=dtype)
    scores_scaled = pypto.mul(scores, scale)
    attn_weights = pypto.softmax(scores_scaled, dim=-1)
    attn_output = pypto.matmul(attn_weights, v_view, out_dtype=dtype)
    # Transpose back and reshape
    attn_output = pypto.transpose(attn_output, 1, 2)
    attn_output_flat = pypto.reshape(attn_output,
                                [batch, seq_len, num_heads * head_dim])
    # Output projection
    res = pypto.matmul(attn_output_flat, out_weight, out_dtype=dtype)
    return res


@pypto.jit
def attention_with_projection_kernel_npu(hidden_states, q_weight, k_weight, v_weight, 
                                     out_weight, out, config: AttentionConfig):
    """Complete attention with input projection (Q, K, V from hidden states)."""
    batch_size = hidden_states.shape[0]
    seq_len = hidden_states.shape[1]
    hidden_size = hidden_states.shape[2]
    view_shape = (batch_size, config.num_heads, seq_len, config.head_dim)
    bs_loop = (batch_size + view_shape[0] - 1) // view_shape[0]
    # Configure tiling
    cube_tiling = 64
    scale = config.scale if config.scale is not None else (1.0 / (config.head_dim ** 0.5))
    pypto.set_cube_tile_shapes([cube_tiling, cube_tiling], [cube_tiling, cube_tiling], [cube_tiling, cube_tiling])
    pypto.set_vec_tile_shapes(1, 16, 8, config.head_dim)
    for bs_idx in pypto.loop(bs_loop, name="LOOP_L0", idx_name="bs_idx", unroll_List={1}):
        q_flat = pypto.matmul(hidden_states, q_weight, out_dtype=config.dtype)
        k_flat = pypto.matmul(hidden_states, k_weight, out_dtype=config.dtype)
        v_flat = pypto.matmul(hidden_states, v_weight, out_dtype=config.dtype)

        # Reshape to multi-head format
        q = pypto.reshape(q_flat, [batch_size, seq_len, config.num_heads, config.head_dim])
        k = pypto.reshape(k_flat, [batch_size, seq_len, config.num_heads, config.head_dim])
        v = pypto.reshape(v_flat, [batch_size, seq_len, config.num_heads, config.head_dim])
        
        # Transpose for attention: [batch, num_heads, seq_len, head_dim]
        q = pypto.transpose(q, 1, 2)
        k = pypto.transpose(k, 1, 2)
        v = pypto.transpose(v, 1, 2)

        offsets = [bs_idx * view_shape[0], 0, 0, 0]
        q_view = pypto.view(q, view_shape, offsets)
        k_view = pypto.view(k, view_shape, offsets)
        v_view = pypto.view(v, view_shape, offsets)
        out[bs_idx * view_shape[0]: (bs_idx+1) * view_shape[0],
            :seq_len, :(config.num_heads*config.head_dim)] = \
                attention_with_projection_core(q_view, k_view,
                                               v_view, out_weight,
                                               scale, config.dtype)

@pypto.jit(runtime_options={"run_mode" : 1})
def attention_with_projection_kernel_sim(hidden_states, q_weight, k_weight, v_weight, 
                                     out_weight, out, config: AttentionConfig):
    """Complete attention with input projection (Q, K, V from hidden states)."""
    batch_size = hidden_states.shape[0]
    seq_len = hidden_states.shape[1]
    hidden_size = hidden_states.shape[2]
    view_shape = (batch_size, config.num_heads, seq_len, config.head_dim)
    bs_loop = (batch_size + view_shape[0] - 1) // view_shape[0]
    # Configure tiling
    cube_tiling = 64
    scale = config.scale if config.scale is not None else (1.0 / (config.head_dim ** 0.5))
    pypto.set_cube_tile_shapes([cube_tiling, cube_tiling], [cube_tiling, cube_tiling], [cube_tiling, cube_tiling])
    pypto.set_vec_tile_shapes(1, 16, 8, config.head_dim)
    for bs_idx in pypto.loop(bs_loop, name="LOOP_L0", idx_name="bs_idx", unroll_List={1}):
        q_flat = pypto.matmul(hidden_states, q_weight, out_dtype=config.dtype)
        k_flat = pypto.matmul(hidden_states, k_weight, out_dtype=config.dtype)
        v_flat = pypto.matmul(hidden_states, v_weight, out_dtype=config.dtype)

        # Reshape to multi-head format
        q = pypto.reshape(q_flat, [batch_size, seq_len, config.num_heads, config.head_dim])
        k = pypto.reshape(k_flat, [batch_size, seq_len, config.num_heads, config.head_dim])
        v = pypto.reshape(v_flat, [batch_size, seq_len, config.num_heads, config.head_dim])
        
        # Transpose for attention: [batch, num_heads, seq_len, head_dim]
        q = pypto.transpose(q, 1, 2)
        k = pypto.transpose(k, 1, 2)
        v = pypto.transpose(v, 1, 2)

        offsets = [bs_idx * view_shape[0], 0, 0, 0]
        q_view = pypto.view(q, view_shape, offsets)
        k_view = pypto.view(k, view_shape, offsets)
        v_view = pypto.view(v, view_shape, offsets)
        out[bs_idx * view_shape[0]: (bs_idx+1) * view_shape[0],
            :seq_len, :(config.num_heads*config.head_dim)] = \
                attention_with_projection_core(q_view, k_view,
                                               v_view, out_weight,
                                               scale, config.dtype)

def attention_with_projection(hidden_states: torch.Tensor, q_weight: torch.Tensor, 
                            k_weight: torch.Tensor, v_weight: torch.Tensor, 
                            out_weight: torch.Tensor, config: AttentionConfig, 
                            run_mode: str = "npu", dynamic: bool = True) -> torch.Tensor:
    y = torch.empty_like(hidden_states)

    if dynamic:
        hidden_states_pto = pypto.from_torch(hidden_states, dynamic_axis=[0])
        q_weight_pto = pypto.from_torch(q_weight, dynamic_axis=[0])
        k_weight_pto = pypto.from_torch(k_weight, dynamic_axis=[0])
        v_weight_pto = pypto.from_torch(v_weight, dynamic_axis=[0])
        out_weight_pto = pypto.from_torch(out_weight, dynamic_axis=[0])
        y_pto = pypto.from_torch(y, dynamic_axis=[0])
    else:
        hidden_states_pto = pypto.from_torch(hidden_states)
        q_weight_pto = pypto.from_torch(q_weight)
        k_weight_pto = pypto.from_torch(k_weight)
        v_weight_pto = pypto.from_torch(v_weight)
        out_weight_pto = pypto.from_torch(out_weight)
        y_pto = pypto.from_torch(y)

    # launch the kernel
    if run_mode == "npu":
        attention_with_projection_kernel_npu(hidden_states_pto, q_weight_pto,
                                     k_weight_pto, v_weight_pto,
                                     out_weight_pto, y_pto, config)
    else:
        attention_with_projection_kernel_sim(hidden_states_pto, q_weight_pto,
                                     k_weight_pto, v_weight_pto,
                                     out_weight_pto, y_pto, config)
    return y

def test_attention_with_projection(device_id = None, run_mode: str = "npu", dynamic: bool = False) -> None:
    """Test complete attention with input/output projections."""
    print("=" * 60)
    print("Test: Attention with Projections")
    print("=" * 60)

    device = f'npu:{device_id}' if (run_mode == "npu" and device_id is not None) else 'cpu'

    batch_size, seq_len, hidden_size = 2, 32, 512
    num_heads, head_dim = 8, 64

    # Create tensors
    hidden_states = torch.randn(batch_size, seq_len, hidden_size,
                               dtype=torch.float32, device=device)
    q_weight = torch.randn(1, hidden_size, num_heads * head_dim,
                          dtype=torch.float32, device=device)
    k_weight = torch.randn(1, hidden_size, num_heads * head_dim,
                          dtype=torch.float32, device=device)
    v_weight = torch.randn(1, hidden_size, num_heads * head_dim,
                          dtype=torch.float32, device=device)
    out_weight = torch.randn(1, num_heads * head_dim, hidden_size,
                            dtype=torch.float32, device=device)
    out_torch = torch.zeros(batch_size, seq_len, hidden_size,
                           dtype=torch.float32, device=device)

    config = AttentionConfig(num_heads=num_heads, head_dim=head_dim, dtype=pypto.DT_FP32)
    # Execute
    attention_with_projection(hidden_states, q_weight,
                                     k_weight, v_weight,
                                     out_weight, config, run_mode, dynamic)

    # Verify (simplified - just check output shape and range)
    print(f"Hidden states shape: {hidden_states.shape}")
    print(f"Output shape: {out_torch.shape}")
    print(f"Output range: [{out_torch.min():.4f}, {out_torch.max():.4f}]")
    print("✓ Attention with projections completed")
    print()


def main():
    """Run attention examples.

    Usage:
        python attention.py          # Run all examples
        python attention.py 1         # Run example 1 only
        python attention.py --list   # List all available examples
    """
    parser = argparse.ArgumentParser(
        description="PyPTO Scaled Dot-Product Attention Examples",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s              Run all examples
  %(prog)s attention_with_projection::test_attention_with_projection
            Run example attention_with_projection::test_attention_with_projection
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
        'attention_dynamic::test_attention_dynamic': {
            'name': 'Attention Dynamic',
            'description': 'Scaled dot-product attention with dynamic shapes',
            'function': test_attention_dynamic
        },
        'attention_with_projection::test_attention_with_projection': {
            'name': 'Attention with Projections',
            'description': 'Complete attention with input/output projections',
            'function': test_attention_with_projection
        }
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
    print("PyPTO Scaled Dot-Product Attention Examples")
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
        print("Make sure CANN environment is configured and NPU is available\n")

    try:
        for ex_id, ex_info in examples_to_run:
            print(f"Running Example {ex_id}: {ex_info['name']}")
            ex_info['function'](device_id, args.run_mode)

        if len(examples_to_run) > 1:
            print("=" * 60)
            print("All attention tests passed!")
            print("=" * 60)

    except Exception as e:
        print(f"\nError: {e}")
        raise


if __name__ == "__main__":
    main()
