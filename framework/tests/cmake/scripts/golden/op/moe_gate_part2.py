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
1. CI批跑时, 由 cmake/scripts/golden_ctrl.py 调用, 为避免日志过多, 此时 logging 级别为 logging.INFO;
2. 单独调试时, 本脚本单独被调用, 此时 logging 级别为 logging.DEBUG;
"""
import sys
import logging
from pathlib import Path
from typing import List

import numpy as np
import torch

if __name__ == "__main__":
    """ 单独调试时配置 """
    # 日志级别
    logging.basicConfig(format='%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s: %(message)s',
                        level=logging.DEBUG)
    # 系统 import 路径
    g_src_root: Path = Path(Path(__file__).parent, "../../../../../").resolve()
    logging.debug("SrcRoot: %s", g_src_root)
    g_ctrl_path: Path = Path(g_src_root, "cmake/scripts")
    if str(g_ctrl_path) not in sys.path:
        sys.path.append(str(g_ctrl_path))
    from golden_register import GoldenRegister  # 单独调试 import 失败, 需确认上文中 '系统 import 路径' 配置正确
else:
    from golden_register import GoldenRegister


@GoldenRegister.reg_golden_func(
    case_names=[
        # moe_gate part2
        "MoEGatePart2OnBoardTest.test_operation_b_2",
        "MoEGatePart2OnBoardTest.test_operation_b_1024",
    ]
)
def gen_moe_part2_golden(case_name: str, output: Path) -> bool:
    dtype = np.float32
    h = 7168
    n_routed_experts = 256
    n_group = 8
    topk_group = 4
    num_experts_per_topk = 8
    first_k_dense_replace = 3
    moe_layer_freq = 1
    s = 1

    if case_name == "MoEGatePart2OnBoardTest.test_operation_b_2":
        b = 2
        # input: scores_for_choice
        # output: group_idx, group_mask
        scores_for_choice_path = Path(output, 'scores_for_choice.bin')
        group_idx_path = Path(output, 'group_idx.bin')
        group_mask_path = Path(output, 'group_mask.bin')
        complete = scores_for_choice_path.exists() and group_idx_path.exists()
        if complete:
            logging.debug("Case(%s), Golden complete.", case_name)
        else:
            # scores_for_choice [b, 8, 32]
            # scores_for_choice = torch.arange(b*s*n_routed_experts,dtype=torch.float32).reshape(b*s, n_group, 32)
            scores_for_choice = torch.randn(b * s, n_routed_experts, dtype=torch.float32).reshape(b * s, n_group, 32)
            # val : [b, 8, 2]
            val, idx = scores_for_choice.topk(2, dim=-1)
            # group_scores : [b, 8]
            group_scores = torch.sum(val, dim=-1)
            # output : group_idx [b, 4]
            val1, group_idx = group_scores.topk(topk_group, dim=-1)
            group_idx = group_idx.to(torch.int32)
            # output : group_mask [b, 4]
            group_mask = torch.zeros_like(group_idx)
            logging.debug("group_scores %s", group_scores)
            logging.debug("group_idx %s", group_idx)
            logging.debug("group_mask %s", group_mask)

            scores_for_choice.numpy().tofile(scores_for_choice_path)
            group_idx.numpy().tofile(group_idx_path)
            group_mask.numpy().astype(np.int32).tofile(group_mask_path)
    elif case_name == "MoEGatePart2OnBoardTest.test_operation_b_1024":
        b = 1024
        # input: scores_for_choice
        # output: group_idx, group_mask
        scores_for_choice_path = Path(output, 'scores_for_choice.bin')
        group_idx_path = Path(output, 'group_idx.bin')
        group_mask_path = Path(output, 'group_mask.bin')
        complete = scores_for_choice_path.exists() and group_idx_path.exists()
        if complete:
            logging.debug("Case(%s), Golden complete.", case_name)
        else:
            # scores_for_choice [b, 8, 32]
            # scores_for_choice = torch.arange(b*s*n_routed_experts,dtype=torch.float32).reshape(b*s, n_group, 32)
            scores_for_choice = torch.randn(b * s, n_routed_experts, dtype=torch.float32).reshape(b * s, n_group, 32)
            # val : [b, 8, 2]
            val, idx = scores_for_choice.topk(2, dim=-1)
            # group_scores : [b, 8]
            group_scores = torch.sum(val, dim=-1)
            # output : group_idx [b, 4]
            val1, group_idx = group_scores.topk(topk_group, dim=-1)
            group_idx = group_idx.to(torch.int32)
            # output : group_mask [b, 4]
            group_mask = torch.zeros_like(group_idx)
            logging.debug("group_scores %s", group_scores)
            logging.debug("group_idx %s", group_idx)
            logging.debug("group_mask %s", group_mask)

            scores_for_choice.numpy().tofile(scores_for_choice_path)
            group_idx.numpy().tofile(group_idx_path)
            group_mask.numpy().astype(np.int32).tofile(group_mask_path)
    else:
        logging.error("Can't get func to gen golden, Case(%s)", case_name)
        return False
    return True


def main() -> bool:
    """
    单独调试 入口函数
    """
    # 用例名称
    case_name_list: List[str] = [
        "MoEGatePart2OnBoardTest.test_operation_b_1024",
    ]
    # 函数调用
    ret: bool = True
    for cs in case_name_list:
        output: Path = Path(g_src_root, "build/output/bin/golden", cs).resolve()
        output.mkdir(parents=True, exist_ok=True)
        ret = gen_moe_part2_golden(case_name=cs, output=output)
    return ret


if __name__ == "__main__":
    exit(0 if main() else 1)
