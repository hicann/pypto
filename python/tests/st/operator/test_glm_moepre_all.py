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
import pypto
import pytest
import torch_npu
import numpy as np
from numpy.testing import assert_allclose


def main():
    test_select_experts()


@pypto.jit(
    runtime_options={"stitch_cfgcache_size": 3100000}
)
def select_experts_glm(hidden_states, residual, weight, bias_input, mm_weight, e_score_bias_input,
                       weight_k, ids_k, row_idx, residual_out,
                       renormalize_flag, topk_group, num_expert_group, row_ids_flag, eps):


    # 3. 得到动态tensor的shape
    bs = hidden_states.shape[0]
    h_num = hidden_states.shape[1]
    ne = mm_weight.shape[0]
    idx_k_shape = ids_k.shape
    topk = idx_k_shape[1]
    calc_dtype = pypto.DT_FP32
    input_dtype = hidden_states.dtype
    view_shape = (16, h_num)  # (32, 5120)
    view_first = 16
    tile_shape_rmsnorm = [16, 1024]
    bs_loop = (bs + view_shape[0] - 1) // view_shape[0]

    # 4. 实现kernel逻辑，循环展开BS动态轴
    for bs_idx in pypto.loop(bs_loop, name="LOOP_MOEGATE_L0", idx_name="bs_idx", unroll_List={1}):
        # 5. 通过view得到tile_logits
        tile_residual = pypto.view(residual, view_shape,
                                   [bs_idx * view_shape[0], 0],
                                   valid_shape=[(bs - bs_idx * view_shape[0]).min(view_shape[0]), h_num])

        tile_hidden_states = pypto.view(hidden_states, view_shape,
                                        [bs_idx * view_shape[0], 0],
                                        valid_shape=[(bs - bs_idx * view_shape[0]).min(view_shape[0]),
                                                     h_num])

        pypto.set_vec_tile_shapes(tile_shape_rmsnorm[0], tile_shape_rmsnorm[1])
        mean_coff = 1.0 / tile_hidden_states.shape[-1]
        # cast to calc_dtype
        tile_residual_fp32 = pypto.cast(tile_residual, calc_dtype)
        tile_hidden_states_fp32 = pypto.cast(tile_hidden_states, calc_dtype)
        weight_2d = pypto.unsqueeze(weight, 0)
        tile_weight_fp32 = pypto.cast(weight_2d, calc_dtype)
        x_f32 = pypto.add(tile_residual_fp32, tile_hidden_states_fp32)
        square = pypto.mul(x_f32, x_f32)
        square_sum = pypto.sum(square, -1, True)
        mean_res = pypto.mul(square_sum, mean_coff)
        reduce_sum = pypto.add(mean_res, eps)
        reduce_sqrt = pypto.sqrt(reduce_sum)
        res_div = pypto.div(x_f32, reduce_sqrt)
        res = pypto.mul(res_div, tile_weight_fp32)
        bias_fp32 = pypto.cast(bias_input, res.dtype)
        hidden_states_add = pypto.add(res, bias_fp32)
        residual_out_16 = pypto.cast(x_f32, input_dtype)

        pypto.set_cube_tile_shapes([16, 16], [512, 512], [32, 32])
        res_mm = pypto.matmul(hidden_states_add, mm_weight, hidden_states_add.dtype, b_trans=True)

        # 6. 按照计算图实现运算逻辑，设置set_vec_tile_shapes时应尽可能用满UB，但不要超过UB的大小。
        pypto.set_vec_tile_shapes(view_first, ne)
        # sigmoid
        topk_weights = pypto.sigmoid(res_mm)  # (bs, ne) fp32
        original_topk_weights = topk_weights  # (bs, ne) fp32

        # unsqueeze
        pypto.set_vec_tile_shapes(ne)
        e_score_bias_2d = pypto.unsqueeze(e_score_bias_input, 0)  # (1, ne) fp32
        # add
        pypto.set_vec_tile_shapes(view_first, ne)
        e_score_bias_2d_cast = pypto.cast(e_score_bias_2d, topk_weights.dtype)
        topk_weights_add = pypto.add(topk_weights, e_score_bias_2d_cast)  # (bs, ne) fp32
        # reshape
        group_unit = ne // num_expert_group
        r1 = pypto.reshape(topk_weights_add,
                            [view_shape[0], num_expert_group, group_unit],
                            valid_shape=[(bs - bs_idx * view_shape[0]).min(view_shape[0]), num_expert_group,
                                        group_unit])

        # amax
        pypto.set_vec_tile_shapes(view_first, num_expert_group, group_unit)
        max1 = pypto.amax(r1, -1, False)
        group_weight = max1

        # topk
        pypto.set_vec_tile_shapes(view_first, num_expert_group)
        _, topk_group_indices = pypto.topk(group_weight, topk_group, -1, True)  # (2, topk_group) int32

        # zeros0 -> full(0)
        topk_group_mask = pypto.full([view_shape[0], num_expert_group], 0.0, group_weight.dtype,
                                        valid_shape=[(bs - bs_idx * view_shape[0]).min(view_shape[0]),
                                                    num_expert_group])  # (16, 1)

        # # scatter
        pypto.set_vec_tile_shapes(view_first, num_expert_group)  # 尾轴不能切
        topk_group_mask_scatter_trans = pypto.scatter_(topk_group_mask, 1, topk_group_indices, 1.0)

        # unsqueeze
        pypto.set_vec_tile_shapes(view_first, num_expert_group)
        twm_unsqueeze = pypto.unsqueeze(topk_group_mask_scatter_trans, -1)  # (bs, neg, 1) fp32

        # expand
        pypto.set_vec_tile_shapes(view_first, num_expert_group, 1)
        twm_expand = pypto.expand_clone(twm_unsqueeze, [view_shape[0], num_expert_group, group_unit],
                                        valid_shape=[(bs - bs_idx * view_shape[0]).min(view_shape[0]),
                                                        num_expert_group, group_unit])

        # reshape
        pypto.set_vec_tile_shapes(view_first, num_expert_group, group_unit)
        twm_reshape = pypto.reshape(twm_expand,
                                    [view_shape[0], ne],
                                    valid_shape=[(bs - bs_idx * view_shape[0]).min(view_shape[0]), ne])

        # logical_not
        pypto.set_vec_tile_shapes(view_first, ne)
        twm_not = pypto.logical_not(twm_reshape)

        # where
        pypto.set_vec_tile_shapes(view_first, ne)
        topk_weights_maskfill = pypto.where(twm_not, 0.0, topk_weights_add)

        # topk2
        pypto.set_vec_tile_shapes(view_first, ne)
        _, topk_ids = pypto.topk(topk_weights_maskfill, topk, -1, True)  # (bs, topk) int32

        # tw_gather
        tw_gather = pypto.gather(original_topk_weights, 1, topk_ids)  # (bs, 8)

        # sum & div
        pypto.set_vec_tile_shapes(view_first, topk)
        if pypto.cond(pypto.symbolic_scalar(renormalize_flag)):
            # sum
            denominator = pypto.sum(tw_gather, -1, True)  # (bs, 1)
            # div for shape (b*s, topk) (b*s, 1)
            topk_weight_out = pypto.div(tw_gather, denominator)  # (bs, topk)
        else:
            denominator = tw_gather
            topk_weight_out = denominator

        # row_idx
        if pypto.cond(pypto.symbolic_scalar(row_ids_flag)):
            pypto.set_vec_tile_shapes(topk)
            row_idx_range = pypto.arange(topk)  # 0~topk-1
            for i in range(view_shape[0]):
                offset = bs_idx * view_shape[0] + i
                full_ids_bs = pypto.full([topk], (bs), row_idx.dtype)  # stride = BS
                row_idx_range_mul = pypto.mul(row_idx_range, full_ids_bs)  # base
                full_ids_bs_idx = pypto.full([topk], (offset), row_idx.dtype)  # offset
                row_idx_tmp = pypto.add(full_ids_bs_idx, row_idx_range_mul)
                row_idx_res = pypto.reshape(row_idx_tmp, [1, topk])
                row_idx[offset:, 0:] = row_idx_res

        # 7. 将结果搬运到输出tensor上
        residual_out[bs_idx * view_shape[0]:, 0:] = residual_out_16
        weight_k[bs_idx * view_shape[0]:, 0:] = topk_weight_out
        ids_k[bs_idx * view_shape[0]:, 0:] = topk_ids


def gen_add_rms_norm_golden(hidden_states, residual, gamma, bias_input, eps):
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
        x_out = x_f32.to(x_dtype)
    return res, x_out  # fp32 bf16


def gen_gate_golden(golden_hidden_states, mm_weight):
    result_mm = torch.matmul(golden_hidden_states, mm_weight.t())
    return result_mm


def gen_row_idx_gloden(hidden_states, top_k):
    num_tokens = hidden_states.shape[0]
    row_idx_len = num_tokens * top_k
    row_idx = (torch.arange(0, row_idx_len, dtype=torch.int32,
                            device=hidden_states.device).view(top_k, -1).permute(1, 0).contiguous())
    return row_idx


def gen_select_experts_golden(result_mm, e_score_bias, row_idx, bs,
                              num_expert_group, ne, top_k, topk_group, renormalize, row_ids_flag):
    original_weights = result_mm.sigmoid()
    bias_2d = e_score_bias.unsqueeze(0)
    topk_weights_g_add = original_weights + bias_2d
    tw_view = topk_weights_g_add.view(bs, num_expert_group, -1)
    grouped_weights = tw_view.max(dim=-1).values
    topk_group_indices_g = torch.topk(grouped_weights.to(torch.float32),
                                      k=topk_group,
                                      dim=-1,
                                      sorted=False)[1]
    topk_group_mask = torch.zeros_like(grouped_weights)
    topk_group_mask.scatter_(1, topk_group_indices_g, 1)
    tgm_unsquee = topk_group_mask.unsqueeze(-1)
    tgm_expand = tgm_unsquee.expand(bs, num_expert_group, ne // num_expert_group)
    topk_weight_mask = tgm_expand.reshape(bs, -1)
    logical_not_tmp = ~topk_weight_mask.bool()
    topk_weights_fill = topk_weights_g_add.masked_fill(logical_not_tmp, 0.0)
    topk_ids_int64 = torch.topk(topk_weights_fill.to(torch.float32),
                                k=top_k,
                                dim=-1,
                                sorted=False)[1]
    topk_ids_int32 = topk_ids_int64.to(torch.int32)
    topk_weights_gather = original_weights.gather(1, topk_ids_int64)

    if renormalize:
        topk_weights_out = topk_weights_gather / topk_weights_gather.sum(dim=-1, keepdim=True)
    else:
        topk_weights_out = topk_weights_gather

    if row_ids_flag:
        golden_row_idx = gen_row_idx_gloden(result_mm, top_k)
    else:
        golden_row_idx = row_idx
    return topk_weights_out, topk_ids_int32, golden_row_idx


def select_experts(residual: torch.Tensor,
                   input_norm_eps: float,
                   input_norm_weight: torch.Tensor,  # layer.input_layernorm.weight.data
                   hidden_states: torch.Tensor,  # Hidden states of shape (num_tokens, hidden_size).
                   gate_weight: torch.Tensor,  # gate matmul weights
                   top_k: int,  # number of top k experts.
                   renormalize: bool,  # Whether to renormalize the routing weights.
                   row_ids_flag: bool,  # Whether to calc row_ids
                   topk_group: int,  # Number of expert groups to select from.
                   num_expert_group: int,  # Number of experts in each group.
                   e_score_correction_bias: torch.Tensor,  # Correction bias to apply to expert scores.
                   input_norm_bias: torch.Tensor
                   ) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:  # topk_weights, topk_ids, row_idx, residual
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)
    bs = hidden_states.shape[0]
    h_num = hidden_states.shape[1]
    topk_weights = torch.empty((bs, top_k), dtype=torch.float32, device=f'npu:{device_id}')
    topk_ids = torch.empty((bs, top_k), dtype=torch.int32, device=f'npu:{device_id}')
    row_idx = torch.empty((bs, top_k), dtype=torch.int32, device=f'npu:{device_id}')
    output_residual = torch.empty((bs, h_num), dtype=torch.bfloat16, device=f'npu:{device_id}')

    # 4. 执行kernel并获取结果
    inputs = {
        hidden_states: [0],
        residual: [0],
        input_norm_weight: [],
        input_norm_bias: [],
        gate_weight: [],
        e_score_correction_bias: []
    }
    outputs = {
        topk_weights: [0],
        topk_ids: [0],
        row_idx: [],
        output_residual: [0]
    }
    pto_inputs = [pypto.from_torch(tensor, dynamic_axis=axis) for tensor, axis in inputs.items()]
    pto_outputs = [pypto.from_torch(tensor, dynamic_axis=axis) for tensor, axis in outputs.items()]

    g = torch.npu.NPUGraph()
    with torch.npu.graph(g):
        select_experts_glm(*pto_inputs, *pto_outputs, renormalize, topk_group, num_expert_group,
                           row_ids_flag, input_norm_eps)
    g.replay()
    return topk_weights, topk_ids, row_idx, output_residual


def test_select_experts():
    # 1. 设置参数
    ne = 160  # 160
    h_num = 5120  # 5120
    top_k = 8
    topk_group = 1
    num_expert_group = 1
    eps = 1e-5
    renormalize = True
    row_ids_flag = False
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)

    # 2. 构造多种shape，测试动态case
    for bs in [8, 8192, 16]:
        # 3. 准备测试数据
        torch.manual_seed(0)
        np.random.seed(0)
        # add rms norm
        hidden_states_tensor = torch.rand((bs, h_num), dtype=torch.bfloat16, device=f'npu:{device_id}')
        residual_tensor = torch.rand((bs, h_num), dtype=torch.bfloat16, device=f'npu:{device_id}')
        weight_tensor = torch.rand((h_num), dtype=torch.bfloat16, device=f'npu:{device_id}')
        bias_input = torch.rand((h_num), dtype=torch.bfloat16, device=f'npu:{device_id}')

        # matmul
        mm_weight = torch.rand((ne, h_num), dtype=torch.float32, device=f'npu:{device_id}')

        # select_experts
        e_score_bias = torch.rand((ne), dtype=torch.bfloat16, device=f'npu:{device_id}')

        # 4. 执行kernel并获取结果
        topk_weights, topk_ids, row_idx, output_residual = select_experts(residual_tensor,
                                                                        eps,
                                                                        weight_tensor,
                                                                        hidden_states_tensor,
                                                                        mm_weight,
                                                                        top_k,
                                                                        renormalize,
                                                                        row_ids_flag,
                                                                        topk_group,
                                                                        num_expert_group,
                                                                        e_score_bias,
                                                                        bias_input
                                                                        )

        # 5. 与PyTorch参考实现对比
        # add rms norm
        golden_hidden_states, golden_residual = gen_add_rms_norm_golden(hidden_states_tensor,
                                                                        residual_tensor, weight_tensor,
                                                                        bias_input, eps)

        # mm
        result_mm = gen_gate_golden(golden_hidden_states, mm_weight)

        # select_experts
        topk_weights_out, topk_ids_int32, golden_row_idx = gen_select_experts_golden(result_mm, e_score_bias, row_idx,
                                                                                    bs, num_expert_group, ne, top_k,
                                                                                    topk_group, renormalize,
                                                                                    row_ids_flag)

        topk_weight_2_tensor_list = topk_weights_out.cpu().flatten().tolist()
        topk_ids_tensor_list = topk_ids_int32.cpu().flatten().tolist()
        golden_row_idx_list = golden_row_idx.cpu().flatten().tolist()
        output_residual_list = golden_residual.cpu().flatten().tolist()

        # residual result
        assert_allclose(np.array(output_residual.cpu().flatten().tolist()),
                        np.array(output_residual_list),
                        rtol=1e-5, atol=1e-5)

        # weight result
        assert_allclose(np.array(topk_weights.cpu().flatten().tolist()),
                        np.array(topk_weight_2_tensor_list),
                        rtol=1e-5, atol=1e-5)

        # idx result
        assert_allclose(np.array(topk_ids.cpu().flatten().tolist()),
                        np.array(topk_ids_tensor_list),
                        rtol=1e-5, atol=1e-5)

        # row_idx result
        assert_allclose(np.array(row_idx.cpu().flatten().tolist()),
                        np.array(golden_row_idx_list),
                        rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    main()
