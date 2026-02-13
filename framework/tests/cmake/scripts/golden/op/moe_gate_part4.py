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
        # moe_gate part4
        "MoEPart4OnBoardTest.test_operation_b_2",
    ]
)
def gen_moe_part4_golden(case_name: str, output: Path) -> bool:
    shape_128_32_i = [128, 32]
    dtype = np.float32
    n_routed_experts = 256
    num_experts_per_topk = 8
    s = 1
    b = 2
    if case_name == "MoEPart4OnBoardTest.test_operation_b_2":
        input_score = Path(output, 'input_score.bin')
        input_tmp_score = Path(output, 'input_tmp_score.bin')

        complete = input_score.exists() and input_tmp_score.exists()
        if complete:
            logging.debug("Case(%s), Golden complete.", case_name)
        else:
            source_score = torch.randn(b * s * n_routed_experts, dtype=torch.float32).reshape(b * s, n_routed_experts)
            tmp_source_score = torch.randn(b * s * n_routed_experts, dtype=torch.float32).reshape(b * s,
                                                                                                  n_routed_experts)
            val, idx = source_score.topk(num_experts_per_topk, dim=-1, largest=True, sorted=True)
            topk_weight = torch.empty((b * s, num_experts_per_topk), dtype=torch.float64)
            for b in range(b * s):
                for k in range(num_experts_per_topk):
                    topk_weight[b][k] = tmp_source_score[b, idx[b][k]]
            # Reduce
            topk_weight_sum = torch.sum(topk_weight, dim=1)
            denominator = topk_weight_sum + np.float64(1e-20)
            denominator_exp = torch.empty((b * s, num_experts_per_topk))

            for b in range(b * s):
                for k in range(num_experts_per_topk):
                    denominator_exp[b][k] = denominator[b]

            output_tensor = topk_weight / denominator_exp

            golden_path = Path(output, 'golden.bin')
            input_score_path = Path(output, 'input_score.bin')
            input_tmp_score_path = Path(output, 'input_tmp_score.bin')
            output_tensor.numpy().astype(np.float32).tofile(golden_path)
            source_score.numpy().tofile(input_score_path)
            tmp_source_score.numpy().tofile(input_tmp_score_path)
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
        "MoEPart4OnBoardTest.test_operation_b_2",
    ]
    # 函数调用
    ret: bool = True
    for cs in case_name_list:
        output: Path = Path(g_src_root, "build/output/bin/golden", cs).resolve()
        output.mkdir(parents=True, exist_ok=True)
        ret = gen_moe_part4_golden(case_name=cs, output=output)
    return ret


if __name__ == "__main__":
    exit(0 if main() else 1)
