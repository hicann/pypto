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


def sigmoid(x):
    return 1 / (1 + torch.exp(-x))


def gated_score_mlp_standard_prefill(x, w1, w2):
    x = torch.from_numpy(x)
    w1 = torch.from_numpy(w1)
    w2 = torch.from_numpy(w2)
    b, s, h = x.shape
    _, n3 = w2.shape
    n = n3 // 3

    act_q_seq_len = [s] * b
    mm2 = torch.zeros(b, s, n * 3)
    for b_idx in range(b):
        valid_q_len = act_q_seq_len[b_idx]
        x_valid = x[b_idx, :valid_q_len, :]
        mm1 = torch.matmul(x_valid, w1)
        mm1_sigmoid = sigmoid(mm1)
        mm2_valid = torch.matmul(mm1_sigmoid, w2)
        mm2[b_idx, :valid_q_len, :] = mm2_valid

    gating_score = mm2.reshape(b, s, 3, n)
    return gating_score.numpy()


def gated_score_mlp_standard_prefill_plus(x, w1, w2):
    x = torch.from_numpy(x).to(torch.float32)
    w1 = torch.from_numpy(w1).to(torch.float32)
    w2 = torch.from_numpy(w2).to(torch.float32)
    b, s, h = x.shape
    _, n3 = w2.shape
    n = n3 // 3
    act_q_seq_len = [s] * b
    gating_score = torch.zeros(b, s, 3, n)

    l = 512

    for b_idx in range(b):
        valid_q_len = act_q_seq_len[b_idx]
        block_num = (valid_q_len + l - 1) // l
        for block_idx in range(block_num):
            block_start = block_idx * l
            block_end = min((block_idx + 1) * l, valid_q_len)
            act_block_size = block_end - block_start

            x_valid = x[b_idx, block_start: block_end, :]
            x_reshape = torch.reshape(x_valid, [act_block_size, h])
            mm1 = torch.matmul(x_reshape, w1)
            mm1_sigmoid = sigmoid(mm1)
            mm2_valid = torch.matmul(mm1_sigmoid, w2)
            mm2_reshape = torch.reshape(mm2_valid, [1, act_block_size, 3, n])
            gating_score[b_idx, block_start: block_end, :, :] = mm2_reshape

    return gating_score.numpy()


def gen_gated_score_entry(dtype, bnsh, output_dir: Path):
    x_path = Path(output_dir, 'x.bin')
    w1_path = Path(output_dir, 'w1.bin')
    w2_path = Path(output_dir, 'w2.bin')
    gating_score_path = Path(output_dir, 'gatingscore.bin')

    b, n, s, h = bnsh
    x_shape = [b, s, h]
    w1_shape = [h, h * 4]
    w2_shape = [h * 4, n * 3]

    x = np.random.uniform(-1, 1, x_shape).astype(dtype)
    w1 = np.random.uniform(-0.1, 0.1, w1_shape).astype(dtype)
    w2 = np.random.uniform(-0.1, 0.1, w2_shape).astype(dtype)
    x.tofile(x_path)
    w1.tofile(w1_path)
    w2.tofile(w2_path)

    gating_score = gated_score_mlp_standard_prefill_plus(x, w1, w2)
    gating_score.astype(dtype).tofile(gating_score_path)


@GoldenRegister.reg_golden_func(
    case_names=[
        "GenGatedScore.gated_score_fp16_s8k_prefill",
    ]
)


def gen_gated_score_func(case_name: str, output: Path) -> bool:
    if case_name == "GenGatedScore.gated_score_fp16_s8k_prefill":
        gen_gated_score_entry(np.float16, (4, 128, 8192, 7168), output)
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
        "GenGatedScore.gated_score_fp16_s8k_prefill",
    ]
    # 函数调用
    ret: bool = True
    for cs in case_name_list:
        output_dir: Path = Path(g_src_root, "build/output/bin/golden", cs).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        ret = gen_gated_score_func(case_name=cs, output=output_dir)
    return ret


if __name__ == "__main__":
    exit(0 if main() else 1)