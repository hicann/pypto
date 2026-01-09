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
"""
import os
import torch
import torch_npu
import pypto
import numpy as np
from numpy.testing import assert_allclose
from torch._subclasses.fake_tensor import FakeTensor
from torch._dynamo import allow_in_graph
from utils.get_format import get_format


def check_args(
    hidden_states,
    residual,
    weight,
    bias,
    eps,
):
    assert hidden_states.dim() == 2
    assert hidden_states.shape[1] == 5120
    assert get_format(hidden_states) == 'ND'
    assert hidden_states.dtype == torch.bfloat16
    
    assert residual.dim() == 2
    assert residual.shape[1] == 5120
    assert get_format(residual) == 'ND'
    assert residual.dtype == torch.bfloat16
    
    assert weight.dim() == 1
    assert weight.shape[0] == 5120
    assert get_format(weight) == 'ND'
    assert weight.dtype == torch.bfloat16
    
    assert bias.dim() == 1
    assert bias.shape[0] == 5120
    assert get_format(bias) == 'ND'
    assert bias.dtype == torch.bfloat16
    
    assert isinstance(eps, float)


def powers_of_2(n: int) -> set[int]:
    assert n > 0, "n must be positive"
    result = set()
    power = 0
    while True:
        current = 1 << power  # 计算2的power次方
        if current > n:
            break
        result.add(current)
        power += 1
    return result


def add_rms_norm_golden(hidden_states, residual, gamma, bias_input, eps):
    x_dtype = residual.dtype
    res_add = residual.to(torch.float32) + hidden_states.to(torch.float32)
    mean_coff = 1.0 / res_add.shape[-1]
    x_f32 = res_add
    square = x_f32 * x_f32
    square = square.sum(dim=-1, keepdim=True)
    mean_res = square * mean_coff
    reduce_sum = mean_res + eps
    reduce_sqrt = torch.sqrt(reduce_sum)
    res_div = x_f32 / reduce_sqrt
    res = res_div * gamma.to(torch.float32)
    res = res + bias_input.to(res.dtype)
    if x_dtype != torch.float32:
        res = res.to(x_dtype)
        x_out = x_f32.to(x_dtype)
    return res, x_out


@pypto.jit(
    runtime_options={
    "stitch_cfgcache_size": 2700000},
    host_options={"only_codegen": True},
)
def add_rms_norm_kernel(x, residual_input, x_gamma, x_bias,
                        hidden_states_out, residual_out, eps):
    # 泳道图使能  pypto.set_option('profile_enable', True)
    # 从入参拿到输入和输出tensor
    calc_dtype = pypto.DT_FP32
    input_dtype = x.dtype
    x_mean_coff = 1.0 / x.shape[-1]

    bs = x.shape[0]
    hidden_size = x.shape[1]
    view_shape = (8, hidden_size)
    bs_loop = (bs + view_shape[0] - 1) // view_shape[0]

    # 实现kernel逻辑， 包在函数中实现变量自动回收
    pypto.set_vec_tile_shapes(hidden_size)
    x_gamma_2d = pypto.reshape(x_gamma, [1, hidden_size], inplace=True)
    x_bias_2d = pypto.reshape(x_bias, [1, hidden_size], inplace=True)

    # 循环展开BS动态轴
    for bs_idx in pypto.loop(bs_loop, name="LOOP_RMS_NORM_L0", idx_name="bs_idx"):
        x_tile = pypto.view(x, view_shape, [bs_idx * view_shape[0], 0],
                            valid_shape=[(bs - bs_idx * view_shape[0]).min(view_shape[0]), hidden_size])
        residual_input_tile = pypto.view(residual_input, view_shape, [bs_idx * view_shape[0], 0],
                                            valid_shape=[(bs - bs_idx * view_shape[0]).min(view_shape[0]),
                                                        hidden_size])

        pypto.set_vec_tile_shapes(1, hidden_size)
        x_tile_fp32 = pypto.cast(x_tile, calc_dtype)

        # add
        residual_input_tile_fp32 = pypto.cast(residual_input_tile, calc_dtype)
        x_f32 = pypto.add(residual_input_tile_fp32, x_tile_fp32)

        # rms norm
        square = pypto.mul(x_f32, x_f32)
        mean_res = pypto.mul(square, x_mean_coff)
        reduce_asum = pypto.sum(mean_res, -1, True)
        reduce_sum = pypto.add(reduce_asum, eps)
        reduce_sqrt = pypto.sqrt(reduce_sum)
        res_div = pypto.div(x_f32, reduce_sqrt)

        hidden_bf16 = pypto.tensor([view_shape[0], hidden_size], pypto.DT_BF16, "hidden_bf16")
        residual_bf16_tmp = pypto.cast(x_f32, input_dtype)
        for tmp_idx in range(view_shape[0]):
            x_gamma_2d_fp32 = pypto.cast(x_gamma_2d, calc_dtype)
            x_bias_2d_fp32 = pypto.cast(x_bias_2d, calc_dtype)
            res_div_single = pypto.view(res_div, [1, hidden_size], [tmp_idx, 0])
            res = pypto.mul(res_div_single, x_gamma_2d_fp32)
            res_add = pypto.add(res, x_bias_2d_fp32)
            x_norm = pypto.cast(res_add, input_dtype)
            hidden_bf16[tmp_idx:tmp_idx + 1, 0:] = x_norm

        residual_out[bs_idx * pypto.symbolic_scalar(view_shape[0]):, 0:] = residual_bf16_tmp
        hidden_states_out[bs_idx * pypto.symbolic_scalar(view_shape[0]):, 0:] = hidden_bf16


def test_rms_norm_main():
    bs = 32
    h_num = 5120

    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)
    eps = 1e-5

    for i in range(0, 2):
        if i == 1:
            bs = 1026
        # 准备测试数据
        residual_tensor = torch.rand(
            (bs, h_num), dtype=torch.bfloat16, device=f'npu:{device_id}')
        hidden_states_tensor = torch.rand(
            (bs, h_num), dtype=torch.bfloat16, device=f'npu:{device_id}')
        weight_tensor = torch.rand(
            (h_num), dtype=torch.bfloat16, device=f'npu:{device_id}')
        bias_input = torch.rand(
            (h_num), dtype=torch.bfloat16, device=f'npu:{device_id}')

        output_hidden_states = torch.empty(
            (bs, h_num), dtype=torch.bfloat16, device=f'npu:{device_id}')
        output_residual = torch.empty(
            (bs, h_num), dtype=torch.bfloat16, device=f'npu:{device_id}')

        inputs = {
            hidden_states_tensor: [0],
            residual_tensor: [0],
            weight_tensor: [],
            bias_input: []
        }
        outputs = {
            output_hidden_states: [0],
            output_residual: [0]
        }

        pto_inputs = [pypto.from_torch(tensor, dynamic_axis=axis) for tensor, axis in inputs.items()]
        pto_outputs = [pypto.from_torch(tensor, dynamic_axis=axis) for tensor, axis in outputs.items()]
        g = torch.npu.NPUGraph()
        with torch.npu.graph(g):
            add_rms_norm_kernel(*pto_inputs, *pto_outputs, eps)
        g.replay()

        golden_hidden_states, golden_residual = add_rms_norm_golden(hidden_states_tensor,
                                                                    residual_tensor, weight_tensor, bias_input, eps)

        assert_allclose(np.array(output_residual.cpu().flatten().tolist()),
                        np.array(golden_residual.flatten().tolist()), rtol=1e-5, atol=1e-5)
        assert_allclose(np.array(output_hidden_states.cpu().flatten().tolist()),
                        np.array(golden_hidden_states.flatten().tolist()), rtol=8e-3, atol=8e-3)


@allow_in_graph
def add_rms_norm(
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
    hidden_states_res: torch.Tensor,
    residual_res: torch.Tensor) -> None:
    if isinstance(hidden_states, FakeTensor):
        return
    check_args(hidden_states, residual, weight, bias, eps)
    inputs = {
        hidden_states: [0],
        residual: [0],
        weight: [],
        bias: []
    }
    outputs = {
        hidden_states_res: [0],
        residual_res: [0]
    }
    pto_inputs = [pypto.from_torch(tensor, dynamic_axis=axis) for tensor, axis in inputs.items()]
    pto_outputs = [pypto.from_torch(tensor, dynamic_axis=axis) for tensor, axis in outputs.items()]
    add_rms_norm_kernel(*pto_inputs, *pto_outputs, eps)


def main():
    test_rms_norm_main()


if __name__ == "__main__":
    main()
