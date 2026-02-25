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
import sys
import math
import logging
from pathlib import Path
from typing import List

import torch
import numpy as np
import torch
from ml_dtypes import bfloat16
sys.path.append(str(Path(__file__).parents[4].joinpath("helper")))
from config_gen import TestBase

torch.manual_seed(0)


if __name__ == "__main__":
    # 日志级别
    logging.basicConfig(format='%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s: %(message)s',
                        level=logging.DEBUG)
    # 系统 import 路径
    g_src_root: Path = Path(Path(__file__).parent, "../../../../../").resolve()
    logging.debug("SrcRoot: %s", g_src_root)
    g_ctrl_path: Path = Path(g_src_root, "scripts")
    if str(g_ctrl_path) not in sys.path:
        sys.path.append(str(g_ctrl_path))
    from golden_register import GoldenRegister
else:
    from golden_register import GoldenRegister


class GenAttentionTest(TestBase):
    def __init__(self):
        super().__init__()
        self.name = "GenAttentionTest"

    def define_parameters(self, param_set: tuple):
        self.setup_parameters(
            b = int,
            s1 = int,
            n = int,
            d = int,
            dtype = torch.dtype
        )
        return self.load_parameters(param_set)

    def define_input_tensors(self):
        cmp_atten = torch.rand([self.b, self.s1, self.n, self.d], dtype=self.dtype).uniform_(-1, 1)
        sel_atten = torch.rand([self.b, self.s1, self.n, self.d], dtype=self.dtype).uniform_(-1, 1)
        win_atten = torch.rand([self.b, self.s1, self.n, self.d], dtype=self.dtype).uniform_(-1, 1)
        gating_score = torch.rand([self.b, self.s1, self.n, 3], dtype=self.dtype).uniform_(-1, 1)

        self.setup_input_tensors(locals())
        return cmp_atten, sel_atten, win_atten, gating_score

    def core(self, cmp_atten, sel_atten, win_atten, gating_score):
        cmp_atten_fp32 = cmp_atten.to(torch.float32)
        sel_atten_fp32 = sel_atten.to(torch.float32)
        win_atten_fp32 = win_atten.to(torch.float32)
        gating_score_fp32 = gating_score.to(torch.float32)
        w_cmp, w_slc, w_win = torch.chunk(gating_score_fp32, 3, dim=-1)
        attention_out_fp32 = (w_cmp * cmp_atten_fp32 + w_slc * sel_atten_fp32 + w_win * win_atten_fp32)
        attention_out = attention_out_fp32.to(self.dtype)

        self.setup_output({'attention_out': attention_out})


@GoldenRegister.reg_golden_func(
    case_names=[
        # MLA_prolog v2
        "TestGenAtten.TestDynamicGenAtten_B_16_S1_1_FP16",
        "TestGenAtten.TestDynamicGenAtten_B_16_S1_1_FP32",
        "TestGenAtten.TestDynamicGenAtten_B_16_S1_1_BF16",
        "TestGenAtten.TestDynamicGenAtten_B_16_S1_2_FP16",
        "TestGenAtten.TestDynamicGenAtten_B_16_S1_2_FP32",
        "TestGenAtten.TestDynamicGenAtten_B_16_S1_2_BF16",
        "TestGenAtten.test_mem_check_ok",
        "TestGenAtten.test_mem_check_fail",
    ]
)
def gen_gen_atten_data(case_name: str, output: Path) -> bool:
    cases = {
        "TestGenAtten.TestDynamicGenAtten_B_16_S1_1_FP16": (16, 1, 128, 512, torch.float16),
        "TestGenAtten.TestDynamicGenAtten_B_16_S1_1_FP32": (16, 1, 128, 512, torch.float32),
        "TestGenAtten.TestDynamicGenAtten_B_16_S1_1_BF16": (16, 1, 128, 512, torch.bfloat16),
        "TestGenAtten.TestDynamicGenAtten_B_16_S1_2_FP16": (16, 2, 128, 512, torch.float16),
        "TestGenAtten.TestDynamicGenAtten_B_16_S1_2_FP32": (16, 2, 128, 512, torch.float32),
        "TestGenAtten.TestDynamicGenAtten_B_16_S1_2_BF16": (16, 2, 128, 512, torch.bfloat16),
        "TestGenAtten.test_mem_check_ok": (16, 1, 128, 512, torch.float16),
        "TestGenAtten.test_mem_check_fail": (16, 1, 128, 512, torch.float16),
    }

    test = GenAttentionTest()
    if case_name in cases:
        test.name = case_name
        test.run(cases.get(case_name), output.parent)
        return True

    logging.error("Can't get func to gen golden, Case(%s)", case_name)
    return False


def main() -> bool:
    # 用例名称
    case_name_list: List[str] = [
        "TestGenAtten.TestDynamicGenAtten_B_16_S1_1_FP16",
        "TestGenAtten.TestDynamicGenAtten_B_16_S1_1_FP32",
        "TestGenAtten.TestDynamicGenAtten_B_16_S1_1_BF16",
        "TestGenAtten.TestDynamicGenAtten_B_16_S1_2_FP16",
        "TestGenAtten.TestDynamicGenAtten_B_16_S1_2_FP32",
        "TestGenAtten.TestDynamicGenAtten_B_16_S1_2_BF16",
        "TestGenAtten.test_mem_check_ok",
        "TestGenAtten.test_mem_check_fail",
    ]
    # 函数调用
    ret: bool = True
    g_src_root = Path(__file__).parents[7]
    for cs in case_name_list:
        output: Path = Path(g_src_root, "build/output/bin/golden", cs).resolve()
        output.mkdir(parents=True, exist_ok=True)
        ret = gen_gen_atten_data(case_name=cs, output=output)
    return ret


if __name__ == "__main__":
    exit(0 if main() else 1)
