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
    g_ctrl_path: Path = Path(g_src_root, "tests/cmake/scripts")
    if str(g_ctrl_path) not in sys.path:
        sys.path.append(str(g_ctrl_path))
    from golden_register import GoldenRegister  # 单独调试 import 失败, 需确认上文中 '系统 import 路径' 配置正确
else:
    from golden_register import GoldenRegister

np.random.seed(0)
torch.manual_seed(0)

dtype_bf16 = torch.bfloat16
dtype_fp16 = torch.float16


class QueryStatesData:
    def __init__(
            self,
            b,
            num_heads,
            s,
            qk_nope_head_dim,
            qk_rope_head_dim,
            kv_lora_rank,
            datatype,
    ):
        self.b = b
        self.num_heads = num_heads
        self.s = s
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.kv_lora_rank = kv_lora_rank
        self.datatype = datatype


def tensor_bf16_tofile(t: torch.tensor, filename: Path):
    input_file_bin = open(str(filename), "wb")
    for each in t:
        input_file_bin.write(each.view(torch.int16).numpy().tobytes())
    input_file_bin.close()


def gen_query_states_data_func(query_data: QueryStatesData, case_name: str, output: Path):
    q_path = Path(output, 'query_states_q.bin')
    q_pe_rope_path = Path(output, 'query_states_q_pe_rope.bin')
    kv_b_proj_wk_path = Path(output, 'query_states_kv_b_proj_wk.bin')
    res_path = Path(output, 'query_states_res.bin')
    s1_path = Path(output, 's1.bin')
    r1_path = Path(output, 'r1.bin')
    t1_path = Path(output, 't1.bin')
    bmm1_path = Path(output, 'bmm1.bin')
    t2_path = Path(output, 't2.bin')
    r2_path = Path(output, 'r2.bin')
    t3_path = Path(output, 't3.bin')

    b = query_data.b
    num_heads = query_data.num_heads
    s = query_data.s
    qk_nope_head_dim = query_data.qk_nope_head_dim
    qk_rope_head_dim = query_data.qk_rope_head_dim
    kv_lora_rank = query_data.kv_lora_rank
    datatype = query_data.datatype

    q_head_dim = qk_nope_head_dim + qk_rope_head_dim

    complete = q_path.exists() and q_pe_rope_path.exists() and kv_b_proj_wk_path.exists() and bmm1_path.exists()
    if complete:
        logging.debug("Case(%s), Golden complete.", case_name)

    q = torch.randn([b, s, num_heads, q_head_dim], dtype=datatype)
    q.numpy().tofile(q_path)

    q_pe_rope = torch.randn([b, num_heads, s, qk_rope_head_dim], dtype=datatype)
    q_pe_rope.numpy().tofile(q_pe_rope_path)

    kv_b_proj_wk = torch.randn([num_heads, qk_nope_head_dim, kv_lora_rank], dtype=datatype)
    kv_b_proj_wk.numpy().tofile(kv_b_proj_wk_path)

    q_nope, _ = torch.split(
        q, [qk_nope_head_dim, qk_rope_head_dim], dim=-1
    )
    q_nope.numpy().tofile(s1_path)
    q_nope1 = q_nope.reshape(b * s, num_heads, qk_nope_head_dim)
    q_nope1.numpy().tofile(r1_path)

    q_nope2 = q_nope1.transpose(0, 1)
    q_nope2.numpy().tofile(t1_path)

    q_nope_new = torch.matmul(q_nope2.to(
        torch.float32), kv_b_proj_wk.to(torch.float32))
    if q_nope_new.dtype != datatype:
        q_nope_new = q_nope_new.to(datatype)
    q_nope_new.numpy().tofile(bmm1_path)

    q_nope_new2 = q_nope_new.transpose(0, 1)
    q_nope_new2.numpy().tofile(t2_path)

    q_nope_new3 = q_nope_new2.reshape(b, s, num_heads, kv_lora_rank)
    q_nope_new3.numpy().tofile(r2_path)

    q_nope_new4 = q_nope_new3.transpose(1, 2)
    q_nope_new4.numpy().tofile(t3_path)

    query_states = torch.cat([q_nope_new4, q_pe_rope], -1)
    query_states.numpy().tofile(res_path)


def gen_query_states_data_bf16_func(query_data: QueryStatesData, case_name: str, output: Path):
    q_path = Path(output, 'query_states_q.bin')
    q_pe_rope_path = Path(output, 'query_states_q_pe_rope.bin')
    kv_b_proj_wk_path = Path(output, 'query_states_kv_b_proj_wk.bin')
    res_path = Path(output, 'query_states_res.bin')
    s1_path = Path(output, 's1.bin')
    r1_path = Path(output, 'r1.bin')
    t1_path = Path(output, 't1.bin')
    bmm1_path = Path(output, 'bmm1.bin')
    t2_path = Path(output, 't2.bin')
    r2_path = Path(output, 'r2.bin')
    t3_path = Path(output, 't3.bin')

    b = query_data.b
    num_heads = query_data.num_heads
    s = query_data.s
    qk_nope_head_dim = query_data.qk_nope_head_dim
    qk_rope_head_dim = query_data.qk_rope_head_dim
    kv_lora_rank = query_data.kv_lora_rank
    datatype = query_data.datatype

    q_head_dim = qk_nope_head_dim + qk_rope_head_dim

    complete = q_path.exists() and q_pe_rope_path.exists() and kv_b_proj_wk_path.exists() and bmm1_path.exists()
    if complete:
        logging.debug("Case(%s), Golden complete.", case_name)

    q = torch.randn([b, s, num_heads, q_head_dim], dtype=datatype)
    tensor_bf16_tofile(q, q_path)

    q_pe_rope = torch.randn([b, num_heads, s, qk_rope_head_dim], dtype=datatype)
    tensor_bf16_tofile(q_pe_rope, q_pe_rope_path)

    kv_b_proj_wk = torch.randn([num_heads, qk_nope_head_dim, kv_lora_rank], dtype=datatype)
    tensor_bf16_tofile(kv_b_proj_wk, kv_b_proj_wk_path)

    q_nope, _ = torch.split(
        q, [qk_nope_head_dim, qk_rope_head_dim], dim=-1
    )
    tensor_bf16_tofile(q_nope, s1_path)

    q_nope1 = q_nope.reshape(b * s, num_heads, qk_nope_head_dim)
    tensor_bf16_tofile(q_nope1, r1_path)

    q_nope2 = q_nope1.transpose(0, 1)
    tensor_bf16_tofile(q_nope2, t1_path)

    q_nope_new = torch.matmul(q_nope2.to(torch.float32), kv_b_proj_wk.to(torch.float32))
    if q_nope_new.dtype != datatype:
        q_nope_new = q_nope_new.to(datatype)
    tensor_bf16_tofile(q_nope_new, bmm1_path)

    q_nope_new2 = q_nope_new.transpose(0, 1)
    tensor_bf16_tofile(q_nope_new2, t2_path)

    q_nope_new3 = q_nope_new2.reshape(b, s, num_heads, kv_lora_rank)
    tensor_bf16_tofile(q_nope_new3, r2_path)

    q_nope_new4 = q_nope_new3.transpose(1, 2)
    tensor_bf16_tofile(q_nope_new4, t3_path)

    query_states = torch.cat([q_nope_new4, q_pe_rope], -1)
    tensor_bf16_tofile(query_states, res_path)


@GoldenRegister.reg_golden_func(
    case_names=[
        "OnBoardTest.test_query_states_fp16_b32_n2",
        "OnBoardTest.test_query_states_fp16_b32_n16",
        "OnBoardTest.test_query_states_fp16_b32_n32",
        "OnBoardTest.test_query_states_bf16_b32_n2",
        "OnBoardTest.test_query_states_bf16_b32_n2_nocat",
        "OnBoardTest.test_query_states_bf16_b32_n2_concat",
        "OnBoardTest.test_query_states_bf16_b32_n16",
        "OnBoardTest.test_query_states_bf16_b32_n32",
    ]
)
def query_states_func(case_name: str, output: Path) -> bool:
    if case_name == "OnBoardTest.test_query_states_fp16_b32_n2":
        query_data = QueryStatesData(
            b=32,
            num_heads=2,
            s=1,
            qk_nope_head_dim=128,
            qk_rope_head_dim=64,
            kv_lora_rank=512,
            datatype=torch.float16
        )
        gen_query_states_data_func(query_data, case_name, output)
    elif case_name == "OnBoardTest.test_query_states_fp16_b32_n16":
        query_data = QueryStatesData(
            b=32,
            num_heads=16,
            s=1,
            qk_nope_head_dim=128,
            qk_rope_head_dim=64,
            kv_lora_rank=512,
            datatype=torch.float16
        )
        gen_query_states_data_func(query_data, case_name, output)
    elif case_name == "OnBoardTest.test_query_states_fp16_b32_n32":
        query_data = QueryStatesData(
            b=32,
            num_heads=32,
            s=1,
            qk_nope_head_dim=128,
            qk_rope_head_dim=64,
            kv_lora_rank=512,
            datatype=torch.float16
        )
        gen_query_states_data_func(query_data, case_name, output)
    elif case_name == "OnBoardTest.test_query_states_bf16_b32_n2":
        query_data = QueryStatesData(
            b=32,
            num_heads=2,
            s=1,
            qk_nope_head_dim=128,
            qk_rope_head_dim=64,
            kv_lora_rank=512,
            datatype=torch.bfloat16
        )
        gen_query_states_data_bf16_func(query_data, case_name, output)
    elif case_name == "OnBoardTest.test_query_states_bf16_b32_n2_nocat":
        query_data = QueryStatesData(
            b=32,
            num_heads=2,
            s=1,
            qk_nope_head_dim=128,
            qk_rope_head_dim=64,
            kv_lora_rank=512,
            datatype=torch.bfloat16
        )
        gen_query_states_data_bf16_func(query_data, case_name, output)
    elif case_name == "OnBoardTest.test_query_states_bf16_b32_n2_concat":
        query_data = QueryStatesData(
            b=32,
            num_heads=2,
            s=1,
            qk_nope_head_dim=128,
            qk_rope_head_dim=64,
            kv_lora_rank=512,
            datatype=torch.bfloat16
        )
        gen_query_states_data_bf16_func(query_data, case_name, output)
    elif case_name == "OnBoardTest.test_query_states_bf16_b32_n16":
        query_data = QueryStatesData(
            b=32,
            num_heads=16,
            s=1,
            qk_nope_head_dim=128,
            qk_rope_head_dim=64,
            kv_lora_rank=512,
            datatype=torch.bfloat16
        )
        gen_query_states_data_bf16_func(query_data, case_name, output)
    elif case_name == "OnBoardTest.test_query_states_bf16_b32_n32":
        query_data = QueryStatesData(
            b=32,
            num_heads=32,
            s=1,
            qk_nope_head_dim=128,
            qk_rope_head_dim=64,
            kv_lora_rank=512,
            datatype=torch.bfloat16
        )
        gen_query_states_data_bf16_func(query_data, case_name, output)
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
        "OnBoardTest.test_query_states_fp16_b32_n2",
    ]
    # 函数调用
    ret: bool = True
    for cs in case_name_list:
        output: Path = Path(g_src_root, "build/output/bin/golden", cs).resolve()
        output.mkdir(parents=True, exist_ok=True)
        ret = query_states_func(case_name=cs, output=output)
    return ret


if __name__ == "__main__":
    exit(0 if main() else 1)
