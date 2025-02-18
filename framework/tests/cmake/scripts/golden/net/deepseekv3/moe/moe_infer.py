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

本脚本有 2 种执行模式:
1. CI批跑时, 由 tests/cmake/scripts/golden_ctrl.py 调用, 为避免日志过多, 此时 logging 级别为 logging.INFO;
2. 单独调试时, 本脚本单独被调用, 此时 logging 级别为 logging.DEBUG;
"""
import sys
import logging
from pathlib import Path
from typing import List

import torch
import numpy as np

if __name__ == "__main__":
    """ 单独调试时配置 """
    # 日志级别
    logging.basicConfig(format='%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s: %(message)s',
                        level=logging.DEBUG)
    # 系统 import 路径
    g_src_root: Path = Path(Path(__file__).parent, "../../../../../").resolve()
    logging.debug("SrcRoot: %s", g_src_root)
    g_ctrl_path: Path = Path(g_src_root, "tests/cmake/scripts")
    if str(g_ctrl_path) not in sys.path:
        sys.path.append(str(g_ctrl_path))
    from golden_register import GoldenRegister  # 单独调试 import 失败, 需确认上文中 '系统 import 路径' 配置正确
else:
    from golden_register import GoldenRegister

np.random.seed(0)
# B, N, S, D = [1, 1, 128, 128]
# BSN = B * N * S
# ND = N * D
# BS = B * S
#
# fp16 = np.float16
# fp32 = np.float32


# def moe_infer(self, x, topk_ids, topk_weight):
#     cnts = topk_ids.new_zeros((topk_ids.shape[0], len(self.experts)))
#     cnts.scatter_(1, topk_ids, 1)
#     tokens_per_expert = cnts.sum(dim=0)
#     idxs = topk_ids.view(-1).argsort()
#     sorted_tokens = x[idxs // topk_ids.shape[1]]
#     sorted_tokens_shape = sorted_tokens.shape
#     if self.ep_size > 1:
#         tokens_per_ep_rank = tokens_per_expert.view(self.ep_size, -1).sum(dim=1)
#         tokens_per_expert_group = tokens_per_expert.new_empty(
#             tokens_per_expert.shape[0]
#         )
#         dist.all_to_all_single(tokens_per_expert_group, tokens_per_expert)
#         output_splits = (
#             tokens_per_expert_group.view(self.ep_size, -1)
#             .sum(1)
#             .cpu()
#             .numpy()
#             .tolist()
#         )
#         gathered_tokens = sorted_tokens.new_empty(
#             tokens_per_expert_group.sum(dim=0).cpu().item(), sorted_tokens.shape[1]
#         )
#         input_split_sizes = tokens_per_ep_rank.cpu().numpy().tolist()
#         dist.all_to_all(
#             list(gathered_tokens.split(output_splits)),
#             list(sorted_tokens.split(input_split_sizes)),
#         )
#         tokens_per_expert_post_gather = tokens_per_expert_group.view(
#             self.ep_size, self.experts_per_rank
#         ).sum(dim=0)
#         gatherd_idxs = np.zeros(shape=(gathered_tokens.shape[0],), dtype=np.int32)
#         s = 0
#         for i, k in enumerate(tokens_per_expert_group.cpu().numpy()):
#             gatherd_idxs[s : s + k] = i % self.experts_per_rank
#             s += k
#         gatherd_idxs = gatherd_idxs.argsort()
#         sorted_tokens = gathered_tokens[gatherd_idxs]
#         tokens_per_expert = tokens_per_expert_post_gather
#     tokens_per_expert = tokens_per_expert.cpu().numpy()

#     outputs = []
#     start_idx = 0
#     for i, num_tokens in enumerate(tokens_per_expert):
#         end_idx = start_idx + num_tokens
#         if num_tokens == 0:
#             continue
#         expert = self.experts[i + self.ep_rank * self.experts_per_rank]
#         tokens_for_this_expert = sorted_tokens[start_idx:end_idx]
#         expert_out = expert(tokens_for_this_expert)
#         outputs.append(expert_out)
#         start_idx = end_idx

#     outs = torch.cat(outputs, dim=0) if len(outputs) else sorted_tokens.new_empty(0)
#     if self.ep_size > 1:
#         new_x = torch.empty_like(outs)
#         new_x[gatherd_idxs] = outs
#         gathered_tokens = new_x.new_empty(*sorted_tokens_shape)
#         dist.all_to_all(
#             list(gathered_tokens.split(input_split_sizes)),
#             list(new_x.split(output_splits)),
#         )
#         outs = gathered_tokens

#     new_x = torch.empty_like(outs)
#     new_x[idxs] = outs
#     final_out = (
#         new_x.view(*topk_ids.shape, -1)
#         .type(topk_weight.dtype)
#         .mul_(topk_weight.unsqueeze(dim=-1))
#         .sum(dim=1)
#         .type(new_x.dtype)
#     )
#     return final_out

def expert(hidden_states):
    # 假设 hiddenStates 和 ffnWeight 已经定义，这里给出示例形状的随机初始化
    bs = hidden_states.shape[0]
    h = hidden_states.shape[1]
    ffn_weight1 = torch.randn(h, h * 3).to(torch.float16)  # 示例形状 (20, 30)
    ffn_weight2 = torch.randn(h, h * 3).to(torch.float16)
    ffn_weight3 = torch.randn(h, h * 3).to(torch.float16)

    # 第一步：计算 t_gate
    # t_gate = torch.matmul(hiddenStates, ffnWeight1).to(torch.float32)
    t_gate = torch.matmul(hidden_states.to(torch.float32), ffn_weight1.to(torch.float32))

    # 第二步：计算 t_swish
    t_swish = -t_gate
    t_swish = torch.exp(t_swish)
    t_swish = t_swish + 1
    t_swish = t_gate / t_swish

    # 第三步：计算 t_up
    t_up = torch.matmul(hidden_states.to(torch.float32), ffn_weight2.to(torch.float32)).to(torch.float32)

    # 第四步：更新 t_swish
    t_swish = t_swish * t_up

    # 第五步：计算最终结果 res
    res = torch.matmul(t_swish.to(torch.float32), ffn_weight3.transpose(0, 1).to(torch.float32)).to(torch.float32)

    logging.debug(res)
    return res


def moe_infer():
    b, s, num_experts_per_tok, h, n_routed_experts, is_fake = 6, 1, 8, 7168, 256, False
    b, s, num_experts_per_tok, h, n_routed_experts, is_fake = 6, 1, 8, 1024, 256, False

    topk_ids = get_topk_idx(b * s, num_experts_per_tok)
    x = torch.randn(b * s, h)
    topk_weight = torch.randn(b * s, num_experts_per_tok)

    cnt_s = topk_ids.new_zeros((topk_ids.shape[0], n_routed_experts))
    cnt_s.scatter_(1, topk_ids, 1)
    tokens_per_expert = cnt_s.sum(dim=0)
    idx_s = topk_ids.view(-1).argsort()
    sorted_tokens = x[idx_s // topk_ids.shape[1]]
    sorted_tokens_shape = sorted_tokens.shape
    tokens_per_expert = tokens_per_expert.cpu().numpy()

    outputs = []
    start_idx = 0
    if is_fake:
        for i, num_tokens in enumerate(tokens_per_expert):
            end_idx = start_idx + num_tokens
            if num_tokens == 0:
                continue
            # expert = self.experts[i + self.ep_rank * self.experts_per_rank]
            tokens_for_this_expert = sorted_tokens[start_idx:end_idx]
            expert_out = expert(tokens_for_this_expert)
            # expert_out = tokens_for_this_expert
            outputs.append(expert_out)
            start_idx = end_idx
    else:
        for i in range(8):
            num_tokens = b * s
            end_idx = start_idx + num_tokens
            if num_tokens == 0:
                continue
            # expert = self.experts[i + self.ep_rank * self.experts_per_rank]
            tokens_for_this_expert = sorted_tokens[start_idx:end_idx]
            expert_out = expert(tokens_for_this_expert)
            # expert_out = tokens_for_this_expert
            outputs.append(expert_out)
            start_idx = end_idx

    outs = torch.cat(outputs, dim=0) if len(outputs) else sorted_tokens.new_empty(0)
    # if self.ep_size > 1:
    #     new_x = torch.empty_like(outs)
    #     new_x[gatherd_idxs] = outs
    #     gathered_tokens = new_x.new_empty(*sorted_tokens_shape)
    #     dist.all_to_all(
    #         list(gathered_tokens.split(input_split_sizes)),
    #         list(new_x.split(output_splits)),
    #     )
    #     outs = gathered_tokens

    new_x = torch.empty_like(outs)
    new_x[idx_s] = outs
    final_out = (
        new_x.view(*topk_ids.shape, -1)
        .type(topk_weight.dtype)
        .mul_(topk_weight.unsqueeze(dim=-1))
        .sum(dim=1)
        .type(new_x.dtype)
    )


def get_topk_idx(bs, num_experts_per_tok):
    tensor = torch.randint(0, 256, (bs, num_experts_per_tok), dtype=torch.int64)
    for i in range(tensor.shape[0]):
        shuffled_tensor = tensor[i][torch.randperm(len(tensor[i]))]
        tensor[i] = shuffled_tensor
    # 打印检查结果
    logging.debug(tensor)
    return tensor


def main() -> bool:
    """
    单独调试 入口函数
    """
    # 用例名称
    case_name_list: List[str] = [
        "MlpTest.test_256_512_tileop",
    ]
    # 函数调用
    ret: bool = True
    for cs in case_name_list:
        output: Path = Path(g_src_root, "build/output/bin/golden", cs).resolve()
        output.mkdir(parents=True, exist_ok=True)
        # ret = gen_mlp_golden(case_name=cs, output=output)
        moe_infer()
    return ret


if __name__ == "__main__":
    exit(0 if main() else 1)
