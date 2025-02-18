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
kenel_unordered_input Axis Example for PyPTO

This example demonstrates:
- Run attention module with kenel_unordered_input.
"""

import os
import sys
import argparse
import pypto
import torch
import numpy as np
from numpy.testing import assert_allclose
from dataclasses import dataclass
from typing import Optional, Tuple


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
    dtype: pypto.DataType = pypto.DT_FP32
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
    scores = scores * scale
    # Apply attention mask if provided
    if attn_mask is not None:
        scores = scores + attn_mask
    attn_weights = torch.softmax(scores, dim=-1)  # [batch, num_heads, seq_len_q, seq_len_kv]
    output = torch.matmul(attn_weights, v)  # [batch, num_heads, seq_len_q, head_dim]
    return output


def scaled_dot_product_attention_core(q: pypto.Tensor, k: pypto.Tensor, v: pypto.Tensor) -> pypto.Tensor:
    k_t = pypto.transpose(k, 2, 3)
    scores = pypto.matmul(q, k_t, out_dtype=dtype_g)
    scores_scaled = scores * scale_g
    attn_weights = pypto.softmax(scores_scaled, dim=-1)
    res = pypto.matmul(attn_weights, v, out_dtype=dtype_g)
    return res


@pypto.jit(
    host_options={"only_codegen": True},
)
def scaled_dot_product_attention_kernel_npu(q: torch.Tensor, y: torch.Tensor, k: torch.Tensor, 
                                 v: torch.Tensor, params: torch.Size, 
                                 config: AttentionConfig) -> None:
    """Scaled dot-product attention with dynamic batch and sequence lengths."""       
    batch_size, num_heads, seq_len, head_dim = params

    # Calculate scale
    scale = config.scale if config.scale is not None else (1.0 / (config.head_dim ** 0.5))
    global scale_g, dtype_g
    scale_g, dtype_g = scale, config.dtype
    cube_tiling = 64
    pypto.set_cube_tile_shapes([cube_tiling, cube_tiling], [cube_tiling, cube_tiling], [cube_tiling, cube_tiling])
    view_shape = (batch_size, num_heads, seq_len, head_dim)
    bs_loop = (batch_size + view_shape[0] - 1) // view_shape[0]
    for bs_idx in pypto.loop(bs_loop):
        q_view = q[bs_idx * view_shape[0]:(bs_idx+1) * view_shape[0], ...]
        k_view = k[bs_idx * view_shape[0]:(bs_idx+1) * view_shape[0], ...]
        v_view = v[bs_idx * view_shape[0]:(bs_idx+1) * view_shape[0], ...]
        pypto.set_vec_tile_shapes(1, 8, 16, 64) 
        res = scaled_dot_product_attention_core(q_view, k_view, v_view)
        y[bs_idx * view_shape[0]:, ...] = res


@pypto.jit(
    host_options={"only_codegen": True},
    runtime_options={"run_mode": 1}
)
def scaled_dot_product_attention_kernel_sim(q: torch.Tensor, y: torch.Tensor, k: torch.Tensor, 
                                 v: torch.Tensor, params: torch.Size, 
                                 config: AttentionConfig) -> None:
    """Scaled dot-product attention with dynamic batch and sequence lengths."""       
    batch_size, num_heads, seq_len, head_dim = params

    # Calculate scale
    scale = config.scale if config.scale is not None else (1.0 / (config.head_dim ** 0.5))
    global scale_g, dtype_g
    scale_g, dtype_g = scale, config.dtype
    cube_tiling = 64
    pypto.set_cube_tile_shapes([cube_tiling, cube_tiling], [cube_tiling, cube_tiling], [cube_tiling, cube_tiling])
    view_shape = (batch_size, num_heads, seq_len, head_dim)
    bs_loop = (batch_size + view_shape[0] - 1) // view_shape[0]
    for bs_idx in pypto.loop(bs_loop):
        q_view = q[bs_idx * view_shape[0]:(bs_idx+1) * view_shape[0], ...]
        k_view = k[bs_idx * view_shape[0]:(bs_idx+1) * view_shape[0], ...]
        v_view = v[bs_idx * view_shape[0]:(bs_idx+1) * view_shape[0], ...]
        pypto.set_vec_tile_shapes(1, 8, 16, 64) 
        res = scaled_dot_product_attention_core(q_view, k_view, v_view)
        y[bs_idx * view_shape[0]:, ...] = res


def scaled_dot_product_attention(q: torch.Tensor, k: torch.Tensor, 
                                 v: torch.Tensor, params: torch.Size, 
                                 config: AttentionConfig, run_mode: str = "npu",
                                 dynamic: bool = True) -> torch.Tensor:
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
        scaled_dot_product_attention_kernel_npu(q_pto, y_pto, k_pto, v_pto, params, config)
    else:
        scaled_dot_product_attention_kernel_sim(q_pto, y_pto, k_pto, v_pto, params, config)

    return y


def test_unordered_input_attention(device_id = None, run_mode: str = "npu", dynamic: bool = True) -> None:
    """Test attention with kenel_unordered_input."""
    print("=" * 60)
    print("Test: kenel_unordered_input Scaled Dot-Product Attention")
    print("=" * 60)
    
    device = f'npu:{device_id}' if (run_mode == "npu" and device_id is not None) else 'cpu'
    
    num_heads, head_dim = 8, 64
    
    batch_size, seq_len_q, seq_len_kv = 8, 64, 64
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
    
    print(f"Batch={batch_size}, SeqQ={seq_len_q}, SeqKV={seq_len_kv}")
    print(f"Input shape: {q_torch.shape}")
    print(f"Output shape: {out_torch.shape}")
    if run_mode == "npu":
        assert_allclose(np.array(out_torch), np.array(golden), rtol=3e-3, atol=3e-3)
    
    print("✓ Attention (kenel_unordered_input) passed for the test case")
    print()


@pypto.jit
def op_unordered_input_kernel_npu(a: torch.Tensor, y1: torch.Tensor, y2: torch.Tensor,  b: torch.Tensor) -> None:
    pypto.set_vec_tile_shapes(16, 16)
    y1[:] = a + b
    y2[:] = a * b


@pypto.jit(runtime_options={"run_mode": 1})
def op_unordered_input_kernel_sim(a: torch.Tensor, y1: torch.Tensor, y2: torch.Tensor,  b: torch.Tensor) -> None:
    pypto.set_vec_tile_shapes(16, 16)
    y1[:] = a + b
    y2[:] = a * b


def op_unordered_input(a: torch.Tensor, b: torch.Tensor, run_mode: str = "npu", dynamic: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
    y1 = torch.empty_like(a)
    y2 = torch.empty_like(a)

    if dynamic:
        a_pto = pypto.from_torch(a, dynamic_axis=[0])
        b_pto = pypto.from_torch(b, dynamic_axis=[0])
        y1_pto = pypto.from_torch(y1, dynamic_axis=[0])
        y2_pto = pypto.from_torch(y2, dynamic_axis=[0])
    else:
        a_pto = pypto.from_torch(a)
        b_pto = pypto.from_torch(b)
        y1_pto = pypto.from_torch(y1)
        y2_pto = pypto.from_torch(y2)

    # launch the kernel
    if run_mode == "npu":
        op_unordered_input_kernel_npu(a_pto, y1_pto, y2_pto, b_pto)
    else:
        op_unordered_input_kernel_sim(a_pto, y1_pto, y2_pto, b_pto)

    return y1, y2


def test_unordered_input_op(device_id = None, run_mode: str = "npu", dynamic: bool = False) -> None:
    """Test op with kenel_unordered_input"""
    print("=" * 60)
    print("Test: OP with kenel_unordered_input")
    print("=" * 60)
    
    device = f'npu:{device_id}' if (run_mode == "npu" and device_id is not None) else 'cpu'
    
    shape = (3, 2)
    dtype = torch.float
    a = torch.rand(shape, dtype=dtype, device=device)
    b = torch.rand(shape, dtype=dtype, device=device)
    # Execute
    y1, y2 = op_unordered_input(a, b, run_mode, dynamic)
    y1, y2 = y1.cpu(), y2.cpu()
    # Verify
    golden1 = torch.add(a, b).cpu()
    golden2 = torch.mul(a, b).cpu()
    
    if run_mode == "npu":
        assert_allclose(np.array(y1), np.array(golden1), rtol=1e-3, atol=1e-3)
        assert_allclose(np.array(y2), np.array(golden2), rtol=1e-3, atol=1e-3)
        print(f"Output1: {y1}")
        print(f"Expected1: {golden1}")
        print(f"Output2: {y2}")
        print(f"Expected2: {golden2}")  
    
    print("✓ OP with kenel_unordered_input passed for the test case")
    print()


def main():
    """Run dynamic examples.
    
    Usage:
        python dynamic.py          # Run all examples
        python dynamic.py 1         # Run example 1 only
        python dynamic.py --list   # List all available examples
    """
    parser = argparse.ArgumentParser(
        description="PyPTO Full Function Examples",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s              Run all examples
  %(prog)s unordered_input_op::test_unordered_input_op
            Run example unordered_input_op::test_unordered_input_op
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
        'unordered_input_attention::test_unordered_input_attention': {
            'name': 'Test attention with kenel_unordered_input',
            'description': 'Attention with kenel_unordered_input example',
            'function': test_unordered_input_attention,
            'requires_npu': True
        },
        'unordered_input_op::test_unordered_input_op': {
            'name': 'Test op with kenel_unordered_input',
            'description': 'OP with kenel_unordered_input example',
            'function': test_unordered_input_op,
            'requires_npu': True
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
    print("PyPTO Dynamic Function Examples")
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
            print("All kenel_unordered_input tests passed!")
            print("=" * 60)
        
    except Exception as e:
        print(f"\nError: {e}")
        raise


if __name__ == "__main__":
    main()