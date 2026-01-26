#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025 - 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
arctic lstm Example for PyPTO

This example demonstrates:
- Special lstm provided by arcitc


lstm is the core mechanism in arcitc lstm-based speculators
"""

import os
import sys
import time
import argparse
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Any

import torch
import pypto
from numpy.testing import assert_allclose

BATCH_SIZE = 32
D_GATE = 4096
D_GATE_4 = 16384


def rms_norm_golden(x: torch.Tensor, eps: float) -> torch.Tensor:
    # 转fp32
    x = x.to(torch.float32)
    mean_square = x.pow(2).mean(-1, keepdim=True)
    inv_rms = torch.rsqrt(mean_square + eps)
    return x * inv_rms


def gelu_approx_sigmoid_golden(x: torch.Tensor) -> torch.Tensor:
    """
    GELU approximation using Sigmoid: x * sigmoid(1.702 * x).
    Matches the NPU implementation for alignment.
    """
    return x * torch.sigmoid(1.702 * x)


def sum_lstm_golden(
        states_4d: torch.Tensor,
        z4_4d: torch.Tensor,
        prev_cell: torch.Tensor,
        alpha: float,
        eps_cell: float,
        eps_state: float,
        w_cell: Optional[torch.Tensor] = None,
        b_cell: Optional[torch.Tensor] = None,
        w_state: Optional[torch.Tensor] = None,
        b_state: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Golden reference for Arctic LSTM kernel."""

    # 1. Input Fusion
    fused = states_4d + alpha * z4_4d

    # 2. Chunking: [Forget, Input, Output, Cell_Candidate]
    chunk_size = fused.shape[-1] // 4
    pre_f, pre_i, pre_o, pre_c = torch.split(fused, chunk_size, dim=-1)

    # 3. Gates
    f_gate = torch.sigmoid(pre_f)
    i_gate = torch.sigmoid(pre_i)

    # 4. Pre-Cell Path
    c_cand_norm = rms_norm_golden(pre_c, eps_cell)

    if w_cell is not None:
        c_cand_norm = c_cand_norm * w_cell
    if b_cell is not None:
        c_cand_norm = c_cand_norm + b_cell

    c_act = gelu_approx_sigmoid_golden(c_cand_norm)

    # 5. Cell Update
    c_new = prev_cell * f_gate + c_act * i_gate

    # 6. Post-Cell Path
    h_temp = rms_norm_golden(c_new, eps_state)

    if w_state is not None:
        h_temp = h_temp * w_state
    if b_state is not None:
        h_temp = h_temp + b_state

    h_act = gelu_approx_sigmoid_golden(h_temp)

    # 7. Output Gate & Final Output
    o_gate = torch.sigmoid(pre_o)
    h_new = h_act * o_gate

    return h_new, c_new


@dataclass
class LstmConfig:
    """Hyperparameters for LSTM."""
    alpha: float = 0.1
    eps_cell: float = 1e-6
    eps_state: float = 1e-6


def rms_norm_pure(x: pypto.Tensor, epsilon: float) -> pypto.Tensor:
    """
    Pure RMSNorm without learnable parameters.
    Formula: x * rsqrt(mean(x^2) + eps)
    """
    input_dtype = x.dtype
    x_fp32 = pypto.cast(x, pypto.DT_FP32)

    y = pypto.mul(x_fp32, x_fp32)
    y = pypto.mul(y, 1.0 / x.shape[-1])
    y = pypto.sum(y, -1, keepdim=True)

    y = pypto.add(y, epsilon)
    y = pypto.sqrt(y)

    output = pypto.div(x_fp32, y)
    return pypto.cast(output, input_dtype)


def gelu_activation_core(x: pypto.Tensor) -> pypto.Tensor:
    """
    GELU activation function: x * 0.5 * (1 + erf(x / sqrt(2)))

    Approximated as: x * 0.5 * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))

    Parameters
    ----------
    x : pypto.Tensor
        Input tensor

    Returns
    -------
    pypto.Tensor
        GELU activated tensor
    """
    x_scaled = pypto.mul(x, 1.702)

    x_neg = pypto.mul(x_scaled, -1.0)
    exp_neg = pypto.exp(x_neg)
    ones = pypto.full(exp_neg.shape, 1.0, exp_neg.dtype, valid_shape=exp_neg.shape)
    sigmoid = pypto.div(ones, pypto.add(exp_neg, 1.0))
    return pypto.mul(x, sigmoid)


@pypto.jit(
    runtime_options={"device_sched_mode": 2,
                     "stitch_cfgcache_size": 2700000}
)
def sum_lstm_kernel(states_4d, z4_4d, prev_cell, w_cell, b_cell, w_state, b_state, h_out, c_out, config):
    """Core computation logic for Snowflake Arctic LSTM."""
    # Dimensions
    batch_size = states_4d.shape[0]
    hidden_dim_4 = states_4d.shape[1]  # 4 * H
    hidden_dim = prev_cell.shape[1]  # H

    # Pre-broadcast 1D weights to [1, H] for correct vector multiplication
    if w_cell is not None:
        w_cell_b_half = pypto.reshape(w_cell, [1, hidden_dim], inplace=True)
        b_cell_b_half = pypto.reshape(b_cell, [1, hidden_dim], inplace=True)

    if w_state is not None:
        w_state_b_half = pypto.reshape(w_state, [1, hidden_dim], inplace=True)
        b_state_b_half = pypto.reshape(b_state, [1, hidden_dim], inplace=True)

    # Main Loop over Batch Dimension
    for bs_offset, unroll_length in pypto.loop_unroll(batch_size, name="LSTM_BATCH_LOOP", idx_name="bs_offset",
                                                     unroll_list=[1, 2, 4]):
        current_tile_bs = unroll_length
        output_offset = [bs_offset, 0]
        pypto.set_semantic_label("Input Cast")
        pypto.set_vec_tile_shapes(current_tile_bs, hidden_dim)
        if w_cell is not None:
            w_cell_b = pypto.cast(w_cell_b_half, pypto.DT_FP32)
            b_cell_b = pypto.cast(b_cell_b_half, pypto.DT_FP32)
        if w_state is not None:
            b_state_b = pypto.cast(b_state_b_half, pypto.DT_FP32)
            w_state_b = pypto.cast(w_state_b_half, pypto.DT_FP32)

        # Set vector tile shape for current batch
        pypto.set_vec_tile_shapes(1, hidden_dim_4)

        # === Step 1: Input Fusion (states + alpha * z4) ===
        pypto.set_semantic_label("Input_Fusion")
        states_tile_half = pypto.view(states_4d, [current_tile_bs, hidden_dim_4], [bs_offset, 0])
        z4_tile_half = pypto.view(z4_4d, [current_tile_bs, hidden_dim_4], [bs_offset, 0])
        x_dtype = states_4d.dtype
        states_tile = pypto.cast(states_tile_half, pypto.DT_FP32)
        z4_tile = pypto.cast(z4_tile_half, pypto.DT_FP32)
        z4_scaled = pypto.mul(z4_tile, config.alpha)
        fused = pypto.add(states_tile, z4_scaled)

        # === Step 2: Logical Split ===
        # Reshape [BS, 4H] -> [BS, 4, H] logic handled by stride/view
        pre_f = pypto.view(fused, [current_tile_bs, hidden_dim], [0, 0])
        pre_i = pypto.view(fused, [current_tile_bs, hidden_dim], [0, hidden_dim * 1])
        pre_o = pypto.view(fused, [current_tile_bs, hidden_dim], [0, hidden_dim * 2])
        pre_c = pypto.view(fused, [current_tile_bs, hidden_dim], [0, hidden_dim * 3])

        # === Step 3: Gates ===
        pypto.set_vec_tile_shapes(1, hidden_dim)
        f_gate = pypto.sigmoid(pre_f)
        i_gate = pypto.sigmoid(pre_i)
        o_gate = pypto.sigmoid(pre_o)

        # === Step 4: Pre-Cell Path ===
        c_cand_norm = rms_norm_pure(pre_c, config.eps_cell)

        if w_cell is not None:
            c_cand_norm = pypto.mul(c_cand_norm, w_cell_b)
        if b_cell is not None:
            c_cand_norm = pypto.add(c_cand_norm, b_cell_b)

        c_act = gelu_activation_core(c_cand_norm)

        # === Step 5: Cell Update (c_new = prev * f + act * i) ===
        prev_cell_tile_half = pypto.view(prev_cell, [current_tile_bs, hidden_dim], [bs_offset, 0])
        prev_cell_tile = pypto.cast(prev_cell_tile_half, pypto.DT_FP32)
        term1 = pypto.mul(prev_cell_tile, f_gate)
        term2 = pypto.mul(c_act, i_gate)
        c_new_tile = pypto.add(term1, term2)
        c_new_tile_out = pypto.cast(c_new_tile, x_dtype)
        pypto.assemble(c_new_tile_out, output_offset, c_out)

        # === Step 6: Post-Cell Path ===
        h_temp = rms_norm_pure(c_new_tile, config.eps_state)

        if w_state is not None:
            h_temp = pypto.mul(h_temp, w_state_b)
        if b_state is not None:
            h_temp = pypto.add(h_temp, b_state_b)

        pypto.set_semantic_label("H_out Operation")
        h_act = gelu_activation_core(h_temp)
        # === Step 7: Final Output (h_new = h_act * o_gate) ===
        h_new_tile = pypto.mul(h_act, o_gate)
        h_new_tile_out = pypto.cast(h_new_tile, x_dtype)
        pypto.assemble(h_new_tile_out, output_offset, h_out)


def prepare_test_data(device) -> Dict[str, Any]:
    """Prepare common data for both precision and performance tests."""
    # Data
    states_4d = torch.randn(BATCH_SIZE, D_GATE_4, dtype=torch.float16, device=device)
    z4_4d = torch.randn(BATCH_SIZE, D_GATE_4, dtype=torch.float16, device=device)
    prev_cell = torch.randn(BATCH_SIZE, D_GATE, dtype=torch.float16, device=device)

    # Weights (None for this test case, can be tensors)
    w_c = torch.randn(D_GATE, dtype=torch.float16, device=device)
    b_c = torch.randn(D_GATE, dtype=torch.float16, device=device)
    w_s = torch.randn(D_GATE, dtype=torch.float16, device=device)
    b_s = torch.randn(D_GATE, dtype=torch.float16, device=device)

    # Outputs
    h_out = torch.zeros(BATCH_SIZE, D_GATE, dtype=torch.float16, device=device)
    c_out = torch.zeros(BATCH_SIZE, D_GATE, dtype=torch.float16, device=device)

    config = LstmConfig(alpha=0.1, eps_cell=1e-6, eps_state=1e-6)

    # PyPTO Wrappers
    inputs_torch = [states_4d, z4_4d, prev_cell, w_c, b_c, w_s, b_s]
    outputs_torch = [h_out, c_out]

    return {
        "torch_inputs": inputs_torch,
        "torch_outputs": outputs_torch,
        "config": config,
    }


def test_sum_lstm():
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 4))
    torch.npu.set_device(device_id)

    for _ in range(5):
        # 0. prepare input data
        data = prepare_test_data(device_id)

        # Unpack data
        t_in = data["torch_inputs"]
        states_4d, z4_4d, prev_cell, w_c, b_c, w_s, b_s = data["torch_inputs"]
        h_out, c_out = data["torch_outputs"]
        cfg = data["config"]

        # 1. Run NPU Kernel
        # Reset outputs to 0 to ensure we are reading fresh results
        inputs = {
            states_4d: [0],
            z4_4d: [0],
            prev_cell: [0],
            w_c: [],
            b_c: [],
            w_s: [],
            b_s: []
        }
        outputs = {
            h_out: [0],
            c_out: [0]
        }
        pto_inputs = [pypto.from_torch(tensor, dynamic_axis=axis) for tensor, axis in inputs.items()]
        pto_outputs = [pypto.from_torch(tensor, dynamic_axis=axis) for tensor, axis in outputs.items()]
        sum_lstm_kernel(*pto_inputs, *pto_outputs, cfg)

        # 2. Run Golden
        golden_h, golden_c = sum_lstm_golden(
            t_in[0], t_in[1], t_in[2],
            alpha=cfg.alpha, eps_cell=cfg.eps_cell, eps_state=cfg.eps_state,
            w_cell=t_in[3], b_cell=t_in[4], w_state=t_in[5], b_state=t_in[6]
        )

        # 3. Compare
        assert_allclose(h_out.cpu().numpy(), golden_h.cpu().numpy(), rtol=1e-3, atol=5e-3)
        assert_allclose(c_out.cpu().numpy(), golden_c.cpu().numpy(), rtol=1e-3, atol=5e-3)


def main():
    parser = argparse.ArgumentParser(description="Run Arctic LSTM PyPTO Example")
    parser.add_argument('--run_mode', type=str, default="npu", choices=["npu", "sim"])
    parser.add_argument('--test_type', type=str, default="precision",
                        choices=["precision", "performance", "all"],
                        help="Choose test type: check correctness or measure performance.")
    args = parser.parse_args()
    test_sum_lstm()


if __name__ == "__main__":
    main()
