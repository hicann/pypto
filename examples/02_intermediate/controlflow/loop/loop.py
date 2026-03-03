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
Loop Feature Examples for PyPTO

This example demonstrates:
- Basic loop usage with start/stop/step
- Loop compile phase print feature
- Loop with scalar addition (add_scalar_loop)
- Loop with dynamic axis (add_scalar_loop_dyn_axis)
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


def _get_mode(run_mode: str):
    if run_mode == "npu":
        return pypto.RunMode.NPU
    elif run_mode == "sim":
        return pypto.RunMode.SIM
    raise ValueError(f"Invalid run_mode: {run_mode}. Must be 'npu' or 'sim'")


# ============================================================================
# 1. Basic Loop Usage
# ============================================================================

def loop_basic(run_mode: str = "npu", dynamic: bool = True):
    if dynamic:
        n = pypto.frontend.dynamic("n")
    else:
        n = 8

    s = 64
    shape = (n * s, s)
    dtype = pypto.DT_FP16
    mode = _get_mode(run_mode)

    @pypto.frontend.jit(runtime_options={"run_mode": mode})
    def loop_basic_kernel(
            t0: pypto.Tensor(shape, dtype),
            t1: pypto.Tensor(shape, dtype),
        ) -> (
            pypto.Tensor(shape, dtype),
            pypto.Tensor(shape, dtype),
        ):
        out0 = pypto.tensor(shape, dtype)
        out1 = pypto.tensor(shape, dtype)
        pypto.set_vec_tile_shapes(64, 64)
        for bs_idx in pypto.loop(0, n, 1):  # start, stop, step
            t0s = t0[bs_idx * s: (bs_idx + 1) * s, :]
            t1s = t1[bs_idx * s: (bs_idx + 1) * s, :]
            out0[bs_idx * s: (bs_idx + 1) * s, :] = pypto.add(t0s, t1s)
        new_step = 2
        for bs_idx in pypto.loop(0, n, new_step):  # start, stop, step
            t0s = t0[bs_idx * s: (bs_idx + new_step) * s, :]
            t1s = t1[bs_idx * s: (bs_idx + new_step) * s, :]
            out1[bs_idx * s: (bs_idx + new_step) * s, :] = pypto.add(t0s, t1s)
        return out0, out1

    return loop_basic_kernel


def test_loop_basic(device_id: int = None, run_mode: str = "npu", dynamic: bool = False) -> None:
    """Test basic loop usage."""
    print("=" * 60)
    print("Test: Basic Loop Usage")
    print("=" * 60)

    device = f'npu:{device_id}' if (run_mode == "npu" and device_id is not None) else 'cpu'

    s, n = 64, 8
    shape = (n * s, s)
    input_t1 = torch.randn(shape, dtype=torch.float16, device=device)
    input_t2 = torch.randn(shape, dtype=torch.float16, device=device)
    output1, output2 = loop_basic(run_mode, dynamic)(input_t1, input_t2)

    expected = input_t1 + input_t2
    if run_mode == "npu":
        max_diff1 = (output1 - expected).abs().max().item()
        max_diff2 = (output2 - expected).abs().max().item()
        equal_output_1_2 = (output1 - output2).abs().max().item() < 1e-6
        print(f"Whether output1 equals output2: {equal_output_1_2}")
        assert max_diff1 < 1e-2, "Result mismatch!"
        assert max_diff2 < 1e-2, "Result mismatch!"
    print("✓ Basic loop usage completed successfully")
    print()


# ============================================================================
# 2. Loop Compile Phase Print
# ============================================================================

def loop_compile_phase_print(shape: tuple, run_mode: str = "npu", dynamic: bool = False):
    if dynamic:
        m = pypto.frontend.dynamic("m")
        _, n = shape
    else:
        m, n = shape

    mode = _get_mode(run_mode)

    @pypto.frontend.jit(runtime_options={"run_mode": mode})
    def loop_compile_phase_print_kernel(
        in_t0: pypto.Tensor((m, n), pypto.DT_FP16),
        in_t1: pypto.Tensor((m, n), pypto.DT_FP16),
    ) -> (
        pypto.Tensor(shape, pypto.DT_FP16),
        pypto.Tensor(shape, pypto.DT_FP16),
    ):
        pypto.set_vec_tile_shapes(64, 64)
        note = '''
        Below are demonstrations of print usage within loops.
        It executes only during compilation, cannot truly print variable values,
        and the number of prints is related to the number of subgraphs generated.
        '''
        separator = "*" * 60
        print(note)
        print(separator)
        cnt_inside_cond = 0
        cnt_outside_cond = 0
        for outside_idx in pypto.loop(5):
            print(f"outside_idx: {outside_idx}")
            for inside_idx in pypto.loop(3):
                print(f"inside_idx: {outside_idx}")
                res = pypto.add(in_t0, in_t0)
                print(f"res: {res}")
                if outside_idx < 3:
                    print(f"(outside_idx < 3)_count: {cnt_outside_cond}")
                    cnt_outside_cond = cnt_outside_cond + 1
                    res = pypto.add(in_t0, in_t0)
                else:
                    res = pypto.sub(in_t0, in_t0)
                if inside_idx < 2:
                    print(f"(inside_idx < 2)_count: {cnt_inside_cond}")
                    cnt_inside_cond = cnt_inside_cond + 1
                    res = pypto.div(in_t0, in_t0)
                else:
                    res = pypto.add(in_t1, in_t1)
                out_t0 = pypto.add(in_t0, in_t0)
                out_t1 = pypto.add(in_t1, in_t1)
        print(separator)
        return out_t0, out_t1

    return loop_compile_phase_print_kernel


def test_loop_compile_phase_print(device_id: int = None, run_mode: str = "npu", dynamic: bool = False) -> None:
    """Test loop compile phase print"""
    print("=" * 60)
    print("Test: Loop Compile Phase Print Feature")
    print("=" * 60)

    device = f'npu:{device_id}' if (run_mode == "npu" and device_id is not None) else 'cpu'

    m, n = 6, 8
    shape = (m, n)
    input_t1 = torch.randn(shape, dtype=torch.float16, device=device)
    input_t2 = torch.randn(shape, dtype=torch.float16, device=device)
    output_t1, output_t2 = loop_compile_phase_print(shape, run_mode, dynamic)(input_t1, input_t2)
    expected_t1 = input_t1 + input_t1
    expected_t2 = input_t2 + input_t2
    if run_mode == "npu":
        max_diff_t1 = (output_t1 - expected_t1).abs().max().item()
        max_diff_t2 = (output_t2 - expected_t2).abs().max().item()
        print(f"Max difference from PyTorch: {max_diff_t1:.6f}")
        print(f"Max difference from PyTorch: {max_diff_t2:.6f}")
        assert max_diff_t1 < 1e-2, "Result mismatch!"
        assert max_diff_t2 < 1e-2, "Result mismatch!"
    print("✓ Test loop compile phase print completed successfully")
    print()


# ============================================================================
# 3. Loop with Scalar Addition (add_scalar_loop)
# ============================================================================

def add_scalar_loop(shape: tuple, val: int, run_mode: str = "npu", dynamic: bool = True):
    if dynamic:
        w = pypto.frontend.dynamic("w")
        _, h, c, n = shape
    else:
        w, h, c, n = shape

    shape = (w, h, c, n)
    mode = _get_mode(run_mode)

    @pypto.frontend.jit(runtime_options={"run_mode": mode})
    def add_kernel(
        input0: pypto.Tensor(shape, pypto.DT_FP32),
        input1: pypto.Tensor(shape, pypto.DT_FP32),
    ) -> pypto.Tensor(shape, pypto.DT_FP32):
        pypto.set_vec_tile_shapes(1, 4, 1, 64)

        b = w
        tile_b = 1
        b_loop = b // tile_b

        output = pypto.tensor(shape, pypto.DT_FP32)
        for idx in pypto.loop(b_loop):
            b_offset = idx * tile_b
            b_offset_end = (idx + 1) * tile_b
            t0_sub = input0[b_offset:b_offset_end, ...]
            t1_sub = input1[b_offset:b_offset_end, ...]
            t3_sub = t0_sub + t1_sub
            t3_sub = t3_sub + val
            pypto.assemble(t3_sub, [b_offset, 0, 0, 0], output)
        return output

    return add_kernel


def test_add_scalar_loop(device_id=None, run_mode: str = "npu", dynamic: bool = True) -> None:
    """Test loop-based scalar addition."""
    print("=" * 60)
    print("Test: Loop with Scalar Addition (add_scalar_loop)")
    print("=" * 60)

    device = f'npu:{device_id}' if (run_mode == "npu" and device_id is not None) else 'cpu'

    shape = (32, 32, 1, 256)
    val = 1
    x = torch.rand(shape, dtype=torch.float32, device=device)
    y = torch.rand(shape, dtype=torch.float32, device=device)

    z = add_scalar_loop(shape, val, run_mode, dynamic)(x, y)
    golden = torch.add(x, y) + val

    max_diff = np.abs(z.cpu().numpy() - golden.cpu().numpy()).max()
    print(f"Input0 shape : {x.shape}")
    print(f"Input1 shape : {y.shape}")
    print(f"Output shape: {z.shape}")
    print(f"Max difference: {max_diff:.6f}")

    if run_mode == "npu":
        assert_allclose(np.array(z.cpu()), np.array(golden.cpu()), rtol=3e-3, atol=3e-3)
    print("✓ add_scalar_loop test passed")
    print()


# ============================================================================
# 4. Loop with Dynamic Axis (add_scalar_loop_dyn_axis)
# ============================================================================

def add_scalar_loop_dynamic_axis(shape: tuple, val: int, run_mode: str = "npu"):
    _, w, n, c = shape
    h = pypto.frontend.dynamic("h")
    mode = _get_mode(run_mode)

    @pypto.frontend.jit(runtime_options={"run_mode": mode})
    def add_scalar_loop_dynamic_axis_kernel(
        input0: pypto.Tensor((h, w, n, c), pypto.DT_FP32),
        input1: pypto.Tensor((h, w, n, c), pypto.DT_FP32),
    ) -> pypto.Tensor((h, w, n, c), pypto.DT_FP32):
        pypto.set_vec_tile_shapes(1, 4, 1, 64)

        b = h
        tile_b = 1
        b_loop = b // tile_b

        output = pypto.tensor((h, w, n, c), pypto.DT_FP32)
        for idx in pypto.loop(b_loop):
            b_offset = idx * tile_b
            b_offset_end = pypto.min((idx + 1) * tile_b, b)

            valid_shape = [b_offset_end - b_offset, w, n, c]

            t0_sub = pypto.view(input0, [tile_b, w, n, c], [b_offset, 0, 0, 0], valid_shape=valid_shape)
            t1_sub = pypto.view(input1, [tile_b, w, n, c], [b_offset, 0, 0, 0], valid_shape=valid_shape)
            t3_sub = t0_sub + t1_sub
            t3_sub = t3_sub + val
            pypto.assemble(t3_sub, [b_offset, 0, 0, 0], output)
        return output

    return add_scalar_loop_dynamic_axis_kernel


def test_add_scalar_loop_dyn_axis(device_id: int = None, run_mode: str = "npu") -> None:
    """Test loop with dynamic axis."""
    print("=" * 60)
    print("Test: Loop with Dynamic Axis (add_scalar_loop_dyn_axis)")
    print("=" * 60)

    device = f'npu:{device_id}' if (run_mode == "npu" and device_id is not None) else 'cpu'

    shape = (32, 32, 1, 256)
    val = 1
    input_data0 = torch.rand(shape, dtype=torch.float, device=device)
    input_data1 = torch.rand(shape, dtype=torch.float, device=device)

    output_data = add_scalar_loop_dynamic_axis(shape, val, run_mode)(input_data0, input_data1)
    golden = torch.add(input_data0, input_data1) + val

    max_diff = np.abs(output_data.cpu().numpy() - golden.cpu().numpy()).max()
    print(f"Input0 shape: {input_data0.shape}")
    print(f"Input1 shape: {input_data1.shape}")
    print(f"Output shape: {output_data.shape}")
    print(f"Max difference: {max_diff:.6f}")

    if run_mode == "npu":
        assert_allclose(np.array(output_data.cpu()), np.array(golden.cpu()), rtol=3e-3, atol=3e-3)
    print("✓ add_scalar_loop_dyn_axis test passed")
    print()


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="PyPTO Loop Feature Examples",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                                              Run all examples
  %(prog)s loop_basic::test_loop_basic                  Run basic loop example
  %(prog)s add_scalar_loop::test_add_scalar_loop        Run scalar loop example
  %(prog)s --list                                       List all available examples
        """
    )
    parser.add_argument(
        'example_id', type=str, nargs='?',
        help='Run a specific case. If omitted, all cases run.'
    )
    parser.add_argument('--list', action='store_true', help='List available examples')
    parser.add_argument(
        '--run_mode', type=str, nargs='?', default="npu", choices=["npu", "sim"],
        help='Run mode: npu or sim'
    )

    args = parser.parse_args()

    examples = {
        'loop_basic::test_loop_basic': {
            'name': 'Test basic loop usage',
            'description': 'Basic loop with start/stop/step',
            'function': test_loop_basic,
        },
        'loop_compile_phase_print::test_loop_compile_phase_print': {
            'name': 'Test loop compile phase print',
            'description': 'Loop compile phase print feature',
            'function': test_loop_compile_phase_print,
        },
        'add_scalar_loop::test_add_scalar_loop': {
            'name': 'Test add_scalar_loop',
            'description': 'Loop-based scalar addition with dynamic batch',
            'function': test_add_scalar_loop,
        },
        'add_scalar_loop_dyn_axis::test_add_scalar_loop_dyn_axis': {
            'name': 'Test add_scalar_loop with dynamic axis',
            'description': 'Loop with dynamic axis using view/assemble and valid_shape',
            'function': test_add_scalar_loop_dyn_axis,
        },
    }

    if args.list:
        print("\n" + "=" * 60)
        print("Available Examples")
        print("=" * 60 + "\n")
        for ex_id, ex_info in sorted(examples.items()):
            print(f"  ID: {ex_id}")
            print(f"     name: {ex_info['name']}")
            print(f"     description: {ex_info['description']}\n")
        return

    if args.example_id is not None:
        if args.example_id not in examples:
            print(f"ERROR: Invalid example ID: {args.example_id}")
            print(f"Valid example IDs are: {', '.join(sorted(examples.keys()))}")
            sys.exit(1)

    print("\n" + "=" * 60)
    print("PyPTO Loop Examples")
    print("=" * 60 + "\n")

    device_id = None
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

    try:
        for ex_id, ex_info in examples_to_run:
            print(f"Running Example {ex_id}: {ex_info['name']}")
            ex_info['function'](device_id, args.run_mode)

        if len(examples_to_run) > 1:
            print("=" * 60)
            print("All loop tests passed!")
            print("=" * 60)

    except Exception as e:
        print(f"\nError: {e}")
        raise


if __name__ == "__main__":
    main()
