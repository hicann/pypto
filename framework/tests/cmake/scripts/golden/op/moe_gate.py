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
""" topk 相关用例 Golden 生成逻辑.

本脚本有 2 种执行模式:
1. CI批跑时, 由 tests/cmake/scripts/golden_ctrl.py 调用, 为避免日志过多, 此时 logging 级别为 logging.INFO;
2. 单独调试时, 本脚本单独被调用, 此时 logging 级别为 logging.DEBUG;
"""
import sys
import logging
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn.functional as functional

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


@GoldenRegister.reg_golden_func(
    case_names=[
        # moe gate
        "MoEGateOnBoardTest.test_operation_b_4",
        "MoEGateOnBoardTest.test_operation_b_16",
        "MoEGateOnBoardTest.test_operation_b_32",
        "MoEGateOnBoardTest.test_operation_b_128",
    ]
)
def gen_moe_golden(case_name: str, output: Path) -> bool:
    seed = 42
    torch.manual_seed(seed)
    h = 0
    n_routed_experts = 0
    n_group = 0
    topk_group = 0
    num_experts_per_topk = 0
    first_k_dense_replace = 0
    moe_layer_freq = 0
    s = 0
    b = 0

    if case_name == "MoEGateOnBoardTest.test_operation_b_4":
        h = 7168
        n_routed_experts = 256
        n_group = 8
        topk_group = 4
        num_experts_per_topk = 8
        first_k_dense_replace = 3
        moe_layer_freq = 1
        s = 1
        b = 4
    elif case_name == "MoEGateOnBoardTest.test_operation_b_16":
        h = 7168
        n_routed_experts = 256
        n_group = 8
        topk_group = 4
        num_experts_per_topk = 8
        first_k_dense_replace = 3
        moe_layer_freq = 1
        s = 1
        b = 16
    elif case_name == "MoEGateOnBoardTest.test_operation_b_32":
        h = 7168
        n_routed_experts = 256
        n_group = 8
        topk_group = 4
        num_experts_per_topk = 8
        first_k_dense_replace = 3
        moe_layer_freq = 1
        s = 1
        b = 32
    elif case_name == "MoEGateOnBoardTest.test_operation_b_128":
        h = 7168
        n_routed_experts = 256
        n_group = 8
        topk_group = 4
        num_experts_per_topk = 8
        first_k_dense_replace = 3
        moe_layer_freq = 1
        s = 1
        b = 128
    else:
        logging.error("Can't get func to gen golden, Case(%s)", case_name)
        return False

    # dump data path
    e_score_correction_bias_path = Path(output, "e_score_correction_bias.bin")
    hidden_states_path = Path(output, "hidden_states.bin")
    weight_path = Path(output, "weight.bin")
    logits_path = Path(output, "logits.bin")
    scores_path = Path(output, "scores.bin")
    scores_for_choice_path = Path(output, "scores_for_choice.bin")
    group_idx_path = Path(output, 'group_idx.bin')
    group_mask_path = Path(output, 'group_mask.bin')
    score_mask_path = Path(output, 'score_mask.bin')
    tmp_scores_path = Path(output, 'tmp_scores.bin')
    topk_idx_path = Path(output, 'topk_idx.bin')
    topk_weight_path = Path(output, 'topk_weight.bin')

    # ========= part1
    e_score_correction_bias = 0 + (2 - 0) * torch.rand(n_routed_experts, dtype=torch.float32).reshape(n_routed_experts)
    # hidden_states = 0.01 + (0.02 - 0.01) * torch.rand(b*s*h, dtype=torch.float16).reshape(b*s, h)
    # weight = 0.01 + (0.02 - 0.01) * torch.rand(n_routed_experts*h, dtype=torch.float16).reshape(n_routed_experts, h)
    # hidden_states_fp32 = hidden_states.type(torch.float32)
    # weight_fp32 = weight.type(torch.float32)
    # logits = functional.linear(hidden_states_fp32, weight_fp32, None).type(torch.float32)

    hidden_states = 0.01 + (0.02 - 0.01) * torch.rand(b * s * h, dtype=torch.float32).reshape(b * s, h)
    weight = 0.01 + (0.02 - 0.01) * torch.rand(n_routed_experts * h, dtype=torch.float32).reshape(n_routed_experts, h)
    logits = functional.linear(hidden_states, weight, None).type(torch.float32)

    # torch.set_printoptions(threshold=float('inf'), sci_mode=False)
    # scores = logits.sigmoid()
    scores = torch.softmax(logits, dim=1)
    scores_for_choice = scores + e_score_correction_bias.unsqueeze(0)
    logging.debug("scores.dtype %s", scores.dtype)
    logging.debug("scores %s", scores)
    e_score_correction_bias.numpy().tofile(e_score_correction_bias_path)
    hidden_states.numpy().tofile(hidden_states_path)
    logits.numpy().tofile(logits_path)
    weight.numpy().tofile(weight_path)
    scores.numpy().tofile(scores_path)
    scores_for_choice.numpy().tofile(scores_for_choice_path)

    # ========= part2  scores_for_choice <---> group_mask group_idx
    # scores_for_choice [b, 8, 32]
    scores_for_choice = scores_for_choice.reshape(b * s, n_group, 32)
    # val : [b, 8, 2]
    val, idx = scores_for_choice.topk(2, dim=-1)
    # group_scores : [b, 8]
    group_scores = torch.sum(val, dim=-1)
    # output : group_idx [b, 4]
    val1, group_idx = group_scores.topk(topk_group, dim=-1)
    # output : group_mask [b, 4]
    group_mask = torch.zeros_like(group_scores).type(torch.float32)
    logging.debug("group_scores %s", group_scores)
    logging.debug("group_idx %s", group_idx)
    logging.debug("group_mask.dtype %s", group_mask.dtype)
    logging.debug("group_mask %s", group_mask)
    group_idx.numpy().astype(np.int32).tofile(group_idx_path)

    # ========= part3 group_mask group_idx scores_for_choice <---> tmp_scores
    group_mask.scatter_(1, group_idx, 1)
    group_mask.numpy().astype(np.float32).tofile(group_mask_path)
    score_mask = (
        group_mask.unsqueeze(-1)
        .expand(b * s, n_group, n_routed_experts // n_group)
        .reshape(b * s, n_routed_experts)
    ).type(torch.float32)
    scores_for_choice = scores_for_choice.reshape(b * s, n_routed_experts).type(torch.float32)
    tmp_scores = scores_for_choice.masked_fill(~score_mask.bool(), 0)  # -3.4e+38)
    score_mask.numpy().tofile(score_mask_path)
    tmp_scores.numpy().astype(np.float32).tofile(tmp_scores_path)
    logging.debug("score_mask %s", score_mask)
    logging.debug("tmp_scores %s", tmp_scores)

    # ======= part4 scores tmp_scores <---> topk_weight
    _, topk_idx = torch.topk(
        tmp_scores, k=num_experts_per_topk, dim=-1, sorted=True
    )
    topk_weight = scores.gather(1, topk_idx).type(torch.float32)
    denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
    topk_weight = topk_weight / denominator
    logging.debug("topk_idx %s", topk_idx)
    logging.debug("topk_weight %s", topk_weight)
    topk_idx.numpy().astype(np.int32).tofile(topk_idx_path)
    topk_weight.numpy().tofile(topk_weight_path)
    return True


def main() -> bool:
    """
    单独调试 入口函数
    """
    # 用例名称
    case_name_list: List[str] = [
        "MoEGateOnBoardTest.test_operation_b_16",
    ]
    # 函数调用
    ret: bool = True
    for cs in case_name_list:
        output: Path = Path(g_src_root, "build/output/bin/golden", cs).resolve()
        output.mkdir(parents=True, exist_ok=True)
        ret = gen_moe_golden(case_name=cs, output=output)
    return ret


if __name__ == "__main__":
    exit(0 if main() else 1)
