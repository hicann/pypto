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

""" 相关用例 Golden 生成逻辑.
本脚本有 2 种执行模式:
1. CI批跑时, 由 cmake/scripts/golden_ctrl.py 调用, 为避免日志过多, 此时 logging 级别为 logging.INFO;
2. 单独调用时, 本脚本单独被调用, 此时 logging 级别为 logging.DEBUG;
"""

import math
import sys
import logging
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

np.set_printoptions(suppress=True, threshold=np.inf)

if __name__ == "__main__":
    # 日志级别
    logging.basicConfig(format='%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s: %(message)s',
                        level=logging.DEBUG)
    # 系统 import 路径
    g_src_root: Path = Path(Path(__file__).parent, "../../../../../").resolve()
    logging.debug("SrcRoot: %s", g_src_root)
    g_ctrl_path: Path = Path(g_src_root, "cmake/scripts")
    if str(g_ctrl_path) not in sys.path:
        sys.path.append(str(g_ctrl_path))
    from golden_register import GoldenRegister
else:
    from golden_register import GoldenRegister

fp32 = np.float32
fp16 = np.float16
bf16 = bfloat16
int32 = np.int32
int8 = np.int8


def gen_axes_for_transpose(offset, base):
    return [x for x in range(offset)] + [x + offset for x in base]


def ceil_div(a, b):
    return (a + b - 1) // b


def nd_to_nz(data: np.ndarray):
    ori_shape = data.shape
    m_ori, n_ori = ori_shape[-2:]
    batch_ori = ori_shape[:-2]
    batch_num = len(batch_ori)
    batch_padding = ((0, 0),) * batch_num
    if data.dtype == "int8":
        m0, n0 = 16, 32
    else:
        m0, n0 = 16, 16
    m1, n1 = ceil_div(m_ori, m0), ceil_div(n_ori, n0)
    padding_m = m1 * m0 - m_ori
    padding_n = n1 * n0 - n_ori
    data = np.pad(data, (batch_padding + ((0, padding_m), (0, padding_n))), 'constant')
    array_trans = gen_axes_for_transpose(len(data.shape) - 2, [2, 0, 1, 3])
    data = data.reshape(batch_ori + (m1, m0, n1, n0)).transpose(*array_trans)
    return data


class ShapeConfig:
    def __init__(self, b: int, m: int, k: int, n: int, dtype: str, out_dtype: str):
        self.b = b
        self.m = m
        self.k = k
        self.n = n
        self.dtype = dtype
        self.out_dtype = out_dtype


def gen_bmm_data4D(b1, b2, input_config: ShapeConfig, output_dir: Path, is_b_trans=False, is_b_nz=False):
    shape_a = [b1[0], b1[1], input_config.m, input_config.k]
    shape_b = [b2[0], b2[1], input_config.k, input_config.n]
    shape_c = [b1[0], b1[1], input_config.m, input_config.n]

    a_path = Path(output_dir, 'mat_a.bin')
    b_path = Path(output_dir, 'mat_b.bin')
    c_path = Path(output_dir, 'mat_c.bin')

    if input_config.dtype == 'int8':
        a = np.random.randint(-4, 5, shape_a).astype(int8)
        b = np.random.randint(-4, 5, shape_b).astype(int8)
        c = np.matmul(a.astype(int32), b.astype(int32)).astype(int32)
    elif input_config.dtype == 'fp16':
        a = np.random.uniform(-1, 1, shape_a).astype(fp16)
        b = np.random.uniform(-1, 1, shape_b).astype(fp16)
        c = np.matmul(a.astype(fp32), b.astype(fp32))
    else:
        a = np.random.uniform(-1, 1, shape_a).astype(bf16)
        b = np.random.uniform(-1, 1, shape_b).astype(bf16)
        c = np.matmul(a.astype(fp32), b.astype(fp32))
    a.tofile(a_path)
    if is_b_trans:
        b = b.transpose(0, 1, 3, 2)
    if is_b_nz:
        b = nd_to_nz(b)
    b.tofile(b_path)
    c.tofile(c_path)


def gen_bmm_data(input_config: ShapeConfig, output_dir: Path, is_b_trans=False, is_b_nz=False):
    shape_a = [input_config.b, input_config.m, input_config.k]
    shape_b = [input_config.b, input_config.k, input_config.n]
    shape_c = [input_config.b, input_config.m, input_config.n]

    a_path = Path(output_dir, 'mat_a.bin')
    b_path = Path(output_dir, 'mat_b.bin')
    c_path = Path(output_dir, 'mat_c.bin')

    if input_config.dtype == 'fp16':
        a = np.random.uniform(-1, 1, shape_a).astype(fp16)
        b = np.random.uniform(-1, 1, shape_b).astype(fp16)
        c = np.matmul(a.astype(fp32), b.astype(fp32))
    elif input_config.dtype == 'int8':
        a = np.random.randint(-4, 5, shape_a).astype(int8)
        b = np.random.randint(-4, 5, shape_b).astype(int8)
        c = np.matmul(a.astype(int32), b.astype(int32)).astype(int32)
    else:
        a = np.random.uniform(-1, 1, shape_a).astype(bf16)
        b = np.random.uniform(-1, 1, shape_b).astype(bf16)
        c = np.matmul(a.astype(fp32), b.astype(fp32))
    a.tofile(a_path)
    if is_b_trans:
        b = b.transpose(0, 2, 1)
    if is_b_nz:
        b = nd_to_nz(b)
    b.tofile(b_path)
    c.tofile(c_path)


@GoldenRegister.reg_golden_func(
    case_names=[
        "DynamicBatchMatmulInterpreterTest.test_bmm_A_B_ND_bf16",
        "DynamicBatchMatmulInterpreterTest.test_bmm_A_Bt_ND_fp16",
        "DynamicBatchMatmulInterpreterTest.test_bmm_A_B_NZ_bf16",
        "DynamicBatchMatmulInterpreterTest.test_bmm_A_Bt_NZ_fp16",
        "DynamicBatchMatmulInterpreterTest.test_bmm_A_B_ND_bf16_tile1",
        "DynamicBatchMatmulInterpreterTest.bmm4D_A_B_NZ",
    ]
)
def gen_dynamic_bmm_golden(case_name: str, output: Path) -> bool:
    if case_name == "DynamicBatchMatmulInterpreterTest.test_bmm_A_B_ND_bf16":
        input_config = ShapeConfig(2, 64, 128, 384, 'bf16', 'fp32')
        gen_bmm_data(input_config, output, False, False)
        return True
    if case_name == "DynamicBatchMatmulInterpreterTest.test_bmm_A_Bt_ND_fp16":
        input_config = ShapeConfig(2, 2, 320, 512, 'fp16', 'fp32')
        gen_bmm_data(input_config, output, True, False)
        return True
    if case_name == "DynamicBatchMatmulInterpreterTest.test_bmm_A_B_NZ_bf16":
        input_config = ShapeConfig(2, 16, 512, 128, 'bf16', 'fp32')
        gen_bmm_data(input_config, output, False, True)
        return True
    if case_name == "DynamicBatchMatmulInterpreterTest.test_bmm_A_Bt_NZ_fp16":
        input_config = ShapeConfig(2, 96, 128, 256, 'fp16', 'fp32')
        gen_bmm_data(input_config, output, True, True)
        return True
    if case_name == "DynamicBatchMatmulInterpreterTest.test_bmm_A_B_ND_bf16_tile1":
        input_config = ShapeConfig(3, 1, 576, 512, 'bf16', 'int32')
        gen_bmm_data(input_config, output, False, False)
        return True
    if case_name == "DynamicBatchMatmulInterpreterTest.bmm4D_A_B_NZ":
        b, m, k, n, Input, Onput = 1, 16, 64, 32, 'fp16', 'fp32'
        b1 = [4, 5]
        b2 = [4, 5]
        input_config = ShapeConfig(b, m, k, n, Input, Onput)
        gen_bmm_data4D(b1, b2, input_config, output, False, True)
        return True
    else:
        logging.error("Can't get func to gen golden, case(%s)", case_name)
        return False
