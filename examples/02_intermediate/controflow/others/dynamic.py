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
Dynamic Axis Example for PyPTO

This example demonstrates:
- Run attention module with dynamic shapes.
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


def scaled_dot_product_attention_core(q: pypto.Tensor, k: pypto.Tensor, v: pypto.Tensor,
                                      scale: float, dtype: pypto.DataType) -> pypto.Tensor:
    k_t = pypto.transpose(k, 2, 3)
    scores = pypto.matmul(q, k_t, out_dtype=dtype)
    scores_scaled = scores * scale
    attn_weights = pypto.softmax(scores_scaled, dim=-1)
    res = pypto.matmul(attn_weights, v, out_dtype=dtype)
    return res


def scaled_dot_product_attention(q_shape: tuple, k_shape: tuple, config: AttentionConfig, run_mode: str = "npu",
                                 dynamic: bool = True):
    if dynamic:
        bs = pypto.frontend.dynamic("bs")
    else:
        bs = q_shape[0]
    head = 8
    dim = 64
    q_len = q_shape[2]
    kv_len = k_shape[2]

    tile = q_shape[0]
    scale = config.scale if config.scale is not None else (1.0 / (dim**0.5))
        
    if run_mode == "npu":
        mode = pypto.RunMode.NPU
    elif run_mode == "sim":
        mode = pypto.RunMode.SIM
    else:
        raise ValueError(f"Invalid run_mode: {run_mode}. Must be 'npu' or 'sim'")
    
    @pypto.frontend.jit(runtime_options={"run_mode": mode}, use_cache=False)
    def scaled_dot_product_attention_kernel(
        q: pypto.Tensor((bs, head, q_len, dim), pypto.DT_FP32),
        k: pypto.Tensor((bs, head, kv_len, dim), pypto.DT_FP32),
        v: pypto.Tensor((bs, head, kv_len, dim), pypto.DT_FP32),
    ) -> pypto.Tensor((bs, head, q_len, dim), pypto.DT_FP32):
        """Scaled dot-product attention with dynamic bsatch size."""
        cubse_tiling = 64
        pypto.set_cube_tile_shapes(
            [cubse_tiling, cubse_tiling],
            [cubse_tiling, cubse_tiling],
            [cubse_tiling, cubse_tiling],
        )

        output_tensor = pypto.tensor((bs, head, q_len, dim), pypto.DT_FP32)
        bs_loop = (bs + tile - 1) // tile

        for bss_idx in pypto.loop(bs_loop):
            bs_offset = bss_idx * tile
            bs_offset_end = pypto.min(bs_offset + tile, bs)
            q_view = pypto.view(q, [tile, head, q_len, dim], [bs_offset, 0, 0, 0], 
                                valid_shape=[bs_offset_end - bs_offset, head, q_len, dim])
            k_view = pypto.view(k, [tile, head, kv_len, dim], [bs_offset, 0, 0, 0], 
                                valid_shape=[bs_offset_end - bs_offset, head, kv_len, dim])
            v_view = pypto.view(v, [tile, head, kv_len, dim], [bs_offset, 0, 0, 0], 
                                valid_shape=[bs_offset_end - bs_offset, head, kv_len, dim])
            pypto.set_vec_tile_shapes(1, 8, 16, 64)
            res = scaled_dot_product_attention_core(
                q_view, k_view, v_view, scale, config.dtype
            )
            pypto.assemble(res, [bs_offset, 0, 0, 0], output_tensor)
        return output_tensor
    
    return scaled_dot_product_attention_kernel


def test_dynamic_shape(device_id: int = None, run_mode: str = "npu", dynamic: bool = True) -> None:
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

        # Execute
        out_torch = scaled_dot_product_attention(
                q_torch.shape, k_torch.shape, config, run_mode, dynamic
            )(q_torch, k_torch, v_torch).cpu()
        
        if run_mode == "npu":
            torch.npu.synchronize()

        # Verify
        scale = 1.0 / (head_dim ** 0.5)
        golden = scaled_dot_product_attention_golden(q_torch, k_torch, v_torch, scale).cpu()

        max_diff = (out_torch - golden).abs().max().item()
        if run_mode == "npu":
            print(f"Batch={batch_size}, SeqQ={seq_len_q}, SeqKV={seq_len_kv}, Max diff: {max_diff:.6f}")
            assert_allclose(np.array(out_torch), np.array(golden), rtol=3e-3, atol=3e-3)
        print(f"Input shape: {q_torch.shape}")
        print(f"Output shape: {out_torch.shape}")

    print("✓ Attention (dynamic) passed for the test case")
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
  %(prog)s dynamic_shape::test_dynamic_shape
            Run example dynamic_shape::test_dynamic_shape
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
        'dynamic_shape::test_dynamic_shape': {
            'name': 'Test dynamic function',
            'description': 'Usage of dynamic function example',
            'function': test_dynamic_shape,
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
            print("All dynamic tests passed!")
            print("=" * 60)

    except Exception as e:
        print(f"\nError: {e}")
        raise


if __name__ == "__main__":
    main()
