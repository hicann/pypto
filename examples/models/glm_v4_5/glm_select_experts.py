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
GLM-4.5 Expert Selection Module for MoE Architecture

This module implements the expert selection logic for Mixture of Experts (MoE) architecture.
It intelligently assigns input tokens to different expert networks based on router logits,
supporting group-based top-k selection and weight renormalization.

Main Functions:
    - select_experts: Main function for expert selection
    - select_experts_kernel: JIT compiled kernel implementation
    - process_main_loop_interation: Process a single batch iteration
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
    router_logits: torch.Tensor,
    top_k: int,
    renormalize: bool,
    topk_group: int,
    num_expert_group: int,
    e_score_correction_bias: torch.Tensor
) -> None:
    assert router_logits.dim() == 2
    assert router_logits.shape[1] == 160
    assert get_format(router_logits) == 'ND'
    assert router_logits.dtype == torch.float32
    
    assert e_score_correction_bias.dim() == 1
    assert e_score_correction_bias.shape[0] == 160
    assert get_format(e_score_correction_bias) == 'ND'
    assert e_score_correction_bias.dtype == torch.bfloat16
    
    assert isinstance(top_k, int)
    assert isinstance(renormalize, bool)
    assert isinstance(topk_group, int)
    assert isinstance(num_expert_group, int)


def process_main_loop_interation(
    bs_idx,
    logits_input,
    e_score_bias_2d,
    weight_k,
    ids_k,
    bs,
    ne,
    view_shape,
    view_first,
    topk,
    topk_group,
    num_expert_group,
    renormalize_flag
):
    """
    Process a single batch iteration for expert selection.

    This function performs the following operations:
    1. Apply sigmoid to router logits
    2. Add expert score correction bias
    3. Group experts and select top-k groups
    4. Mask non-selected experts
    5. Select top-k experts from masked logits
    6. Optionally renormalize expert weights

    Args:
        bs_idx: Current batch index
        logits_input: Router logits [num_tokens, num_router_experts]
        e_score_bias_2d: Expert score correction bias [1, num_router_experts]
        weight_k: Output tensor for top-k weights [num_tokens, topk]
        ids_k: Output tensor for top-k expert IDs [num_tokens, topk]
        bs: Batch size (number of tokens)
        ne: Number of experts
        view_shape: Shape for view operation
        view_first: First dimension of view shape
        topk: Number of top experts to select
        topk_group: Number of expert groups for top-k selection
        num_expert_group: Number of experts per group
        renormalize_flag: Whether to renormalize expert weights
    """
    # 6. 通过view得到tile_logits
    tile_logits = pypto.view(logits_input, view_shape,
                                [bs_idx * view_shape[0], 0],
                                valid_shape=[(bs - bs_idx * view_shape[0]).min(view_shape[0]), ne])

    # 7. 按照计算图实现运算逻辑，设置set_vec_tile_shapes时应尽可能用满UB，但不要超过UB的大小。
    pypto.set_vec_tile_shapes(view_first, ne)
    tile_logits_fp32 = pypto.cast(tile_logits, pypto.DT_FP32)
    e_score_bias_2d_cast = pypto.cast(e_score_bias_2d, tile_logits_fp32.dtype)

    # sigmoid
    topk_weights = pypto.sigmoid(tile_logits_fp32)  # (bs, ne) fp32

    # add
    topk_weights_add = pypto.add(topk_weights, e_score_bias_2d_cast)  # (8, 160) fp32
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

    # zeros -> full(0)
    topk_group_mask = pypto.full([view_shape[0], num_expert_group], 0.0, group_weight.dtype,
                                    valid_shape=[(bs - bs_idx * view_shape[0]).min(view_shape[0]),
                                                num_expert_group])  # (16, 1)

    # scatter 尾轴不能切
    topk_group_mask_scatter_trans = pypto.scatter_(topk_group_mask, 1, topk_group_indices, 1.0)

    # unsqueeze
    twm_unsqueeze = pypto.unsqueeze(topk_group_mask_scatter_trans, -1)  # (1, 1, 1) fp32

    # expand
    pypto.set_vec_tile_shapes(view_first, num_expert_group, ne)  # ne时 可以切成一块
    twm_expand = pypto.expand_clone(twm_unsqueeze, [view_shape[0], num_expert_group, group_unit],
                                    valid_shape=[(bs - bs_idx * view_shape[0]).min(view_shape[0]),
                                                    num_expert_group, group_unit])

    # reshape
    pypto.set_vec_tile_shapes(view_first, num_expert_group, group_unit)  # (1,1,160)
    twm_reshape = pypto.reshape(twm_expand,
                                [view_shape[0], ne],
                                valid_shape=[(bs - bs_idx * view_shape[0]).min(view_shape[0]), ne])

    # logical_not
    pypto.set_vec_tile_shapes(view_first, ne)
    twm_not = pypto.logical_not(twm_reshape)

    # where
    topk_weights_maskfill = pypto.where(twm_not, 0.0, topk_weights_add)

    # topk2
    _, topk_ids = pypto.topk(topk_weights_maskfill, topk, -1, True)  # (bs, topk) int32

    # tw_gather
    tw_gather = pypto.gather(topk_weights, 1, topk_ids)  # (bs, 8)

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

    # # 8. 将结果搬运到输出tensor上
    weight_k[bs_idx * view_shape[0]:, 0:] = topk_weight_out
    ids_k[bs_idx * view_shape[0]:, 0:] = topk_ids


@pypto.jit(
    runtime_options={"stitch_function_num_initial": 128,
    "stitch_function_outcast_memory": 128,
    "stitch_function_inner_memory": 128,
    "stitch_cfgcache_size": 2500000},
    host_options={"only_codegen": True},
)
def select_experts_kernel(logits_input, e_score_bias_input, weight_k, ids_k,
                          renormalize_flag, topk_group, num_expert_group):
    """
    JIT compiled kernel for expert selection in MoE architecture.

    This kernel implements the expert selection algorithm:
    1. Applies sigmoid to router logits to get expert scores
    2. Adds expert score correction bias
    3. Groups experts and selects top-k groups
    4. Masks non-selected experts
    5. Selects top-k experts from masked logits
    6. Optionally renormalizes expert weights

    Args:
        logits_input: Router logits [num_tokens, num_router_experts]
        e_score_bias_input: Expert score correction bias [num_router_experts]
        weight_k: Output tensor for top-k weights [num_tokens, topk]
        ids_k: Output tensor for top-k expert IDs [num_tokens, topk]
        renormalize_flag: Whether to renormalize expert weights (0 or 1)
        topk_group: Number of expert groups for top-k selection
        num_expert_group: Number of experts per group

    Note:
        This function processes inputs in tiles of size 1 to support dynamic batch sizes.
        The expert selection uses a two-stage approach: first select groups, then select experts.
    """
    # 3. 得到动态tensor的shape
    bs = logits_input.shape[0]
    ne = logits_input.shape[1]
    idx_k_shape = ids_k.shape
    topk = idx_k_shape[1]
    view_shape = (1, ne)
    view_first = 1
    bs_loop = (bs + view_shape[0] - 1) // view_shape[0]

    # 4. 定义动态函数
    for _ in pypto.loop(1, name="LOOP_RESHAPE_INPLACE", idx_name="_"):
        pypto.set_vec_tile_shapes(ne)
        e_score_bias_2d = pypto.reshape(e_score_bias_input, [1, ne], inplace=True)  # (160) -> (1,160)

    # 5. 实现kernel逻辑，循环展开BS动态轴
    for bs_idx in pypto.loop(bs_loop, name="LOOP_MOEGATE_L0", idx_name="bs_idx"):
        process_main_loop_interation(
            bs_idx,
            logits_input,
            e_score_bias_2d,
            weight_k,
            ids_k,
            bs,
            ne,
            view_shape,
            view_first,
            topk,
            topk_group,
            num_expert_group,
            renormalize_flag
        )


def gen_row_idx_gloden(hidden_states, top_k):
    num_tokens = hidden_states.shape[0]
    row_idx_len = num_tokens * top_k
    row_idx = (torch.arange(0, row_idx_len, dtype=torch.int32,
                            device=hidden_states.device).view(top_k, -1).permute(1, 0).contiguous())
    return row_idx


def test_select_experts():
    # 1. 设置参数

    bs = 32
    ne = 160
    top_k = 8
    topk_group = 1
    num_expert_group = 1
    renormalize = True
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)

    # 2. 构造多种shape，测试动态case
    for i in range(0, 2):
        if i == 1:
            bs = 1026
        # 3. 准备测试数据
        torch.manual_seed(0)
        np.random.seed(0)
        router_logits = torch.rand(
            (bs, ne), dtype=torch.float32, device=f'npu:{device_id}')
        e_score_bias = torch.rand(
            (ne), dtype=torch.bfloat16, device=f'npu:{device_id}')
        topk_weights = torch.empty(
            (bs, top_k), dtype=torch.float32, device=f'npu:{device_id}')
        topk_ids = torch.empty(
            (bs, top_k), dtype=torch.int32, device=f'npu:{device_id}')

        # 4. 执行kernel并获取结果
        inputs = {
            router_logits: [0],
            e_score_bias: []
        }
        outputs = {
            topk_weights: [0],
            topk_ids: [0]
        }
        pto_inputs = [pypto.from_torch(tensor, dynamic_axis=axis) for tensor, axis in inputs.items()]
        pto_outputs = [pypto.from_torch(tensor, dynamic_axis=axis) for tensor, axis in outputs.items()]
        g = torch.npu.NPUGraph()
        with torch.npu.graph(g):
            select_experts_kernel(*pto_inputs, *pto_outputs, renormalize, topk_group, num_expert_group)
        g.replay()

        # 5. 与PyTorch参考实现对比
        router_logits_fp32 = router_logits.to(torch.float)
        original_weights = router_logits_fp32.sigmoid()
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
        tgm_expand = tgm_unsquee.expand(
            bs, num_expert_group, ne // num_expert_group)
        topk_weight_mask = tgm_expand.reshape(bs, -1)
        logical_not_tmp = ~topk_weight_mask.bool()
        topk_weights_fill = topk_weights_g_add.masked_fill(
            logical_not_tmp, 0.0)

        topk_ids_int64 = torch.topk(topk_weights_fill.to(torch.float32),
                                    k=top_k,
                                    dim=-1,
                                    sorted=False)[1]
        topk_ids_int32 = topk_ids_int64.to(torch.int32)

        topk_weights_gather = original_weights.gather(1, topk_ids_int64)

        if renormalize:
            topk_weights_out = topk_weights_gather / \
                topk_weights_gather.sum(dim=-1, keepdim=True)
        else:
            topk_weights_out = topk_weights_gather

        topk_weight_2_tensor_list = topk_weights_out.cpu().flatten().tolist()
        topk_ids_tensor_list = topk_ids_int32.cpu().flatten().tolist()

        # weight result
        assert_allclose(np.array(topk_weights.cpu().flatten().tolist()),
                        np.array(topk_weight_2_tensor_list),
                        rtol=5e-3, atol=5e-3)

        # idx result
        assert_allclose(np.array(topk_ids.cpu().flatten().tolist()),
                        np.array(topk_ids_tensor_list),
                        rtol=5e-3, atol=5e-3)


@allow_in_graph
def select_experts(router_logits: torch.Tensor,
                top_k: int,  # number of top k experts.
                # Whether to renormalize the routing weights.
                renormalize: bool,
                # Number of expert groups to select from.
                topk_group: int,
                # Number of experts in each group.
                num_expert_group: int,
                # Correction bias to apply to expert scores.
                e_score_correction_bias: torch.Tensor,
                topk_weights: torch.Tensor,
                topk_ids: torch.Tensor
):
    """
    Select top-k experts for each token based on router logits.

    This function implements the expert selection mechanism for MoE architecture.
    It uses a two-stage selection process:
    1. First selects top-k expert groups
    2. Then selects top-k experts from the selected groups

    Args:
        router_logits: Router logits [num_tokens, num_router_experts]
        top_k: Number of top experts to select per token
        renormalize: Whether to renormalize the routing weights
        topk_group: Number of expert groups to select from
        num_expert_group: Number of experts in each group
        e_score_correction_bias: Correction bias to apply to expert scores [num_router_experts]
        topk_weights: Output tensor for top-k expert weights [num_tokens, topk]
        topk_ids: Output tensor for top-k expert IDs [num_tokens, topk]

    Note:
        This function is decorated with @allow_in_graph to enable integration
        with PyTorch's compilation graph.
    """
    if isinstance(router_logits, FakeTensor):
        return
    check_args(
        router_logits,
        top_k,
        renormalize,
        topk_group,
        num_expert_group,
        e_score_correction_bias
    )
    inputs = {
        router_logits: [0],
        e_score_correction_bias: []
    }
    outputs = {
        topk_weights: [0],
        topk_ids: [0]
    }
    pto_inputs = [pypto.from_torch(tensor, dynamic_axis=axis) for tensor, axis in inputs.items()]
    pto_outputs = [pypto.from_torch(tensor, dynamic_axis=axis) for tensor, axis in outputs.items()]
    select_experts_kernel(*pto_inputs, *pto_outputs, renormalize, topk_group, num_expert_group)


def main():
    test_select_experts()


if __name__ == "__main__":
    main()
