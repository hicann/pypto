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
"""
import sys
import math
import logging
from pathlib import Path

from ml_dtypes import bfloat16
import numpy as np

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

FP32 = np.float32
FP16 = np.float16
BF16 = bfloat16
INT32 = np.int32
INT8 = np.int8


def gen_axes_for_transpose(offset, base):
    return [x for x in range(offset)] + [x + offset for x in base]


def ceil_div(a, b):
    return (a + b - 1) // b


def nd_to_fractal_nz(data: np.ndarray):
    ori_shape = data.shape
    m_ori, n_ori = ori_shape[-2:]
    batch_ori = ori_shape[:-2]
    batch_num = len(batch_ori)
    batch_padding = ((0, 0),) * batch_num
    if data.dtype == INT8:
        m0, n0 = 16, 32
    elif data.dtype == FP16 or data.dtype == BF16 or data.dtype == INT32:
        m0, n0 = 16, 16
    elif data.dtype == FP32:
        m0, n0 = 16, 8

    m1, n1 = ceil_div(m_ori, m0), ceil_div(n_ori, n0)
    padding_m = m1 * m0 - m_ori
    padding_n = n1 * n0 - n_ori
    data = np.pad(data, (batch_padding + ((0, padding_m), (0, padding_n))), 'constant')
    array_trans = gen_axes_for_transpose(len(data.shape) - 2, [2, 0, 1, 3])
    data = data.reshape(batch_ori + (m1, m0, n1, n0)).transpose(*array_trans)
    return data


class ShapeConfig:
    def __init__(self, m: int, k: int, n: int, in_dtype: np.dtype, out_dtype: np.dtype, trans_a: bool, trans_b: bool,
    a_nz_flag: bool, b_nz_flag: bool, c_nz_flag: bool):
        self.m = m
        self.k = k
        self.n = n
        self.in_dtype = in_dtype
        self.out_dtype = out_dtype
        self.trans_a = trans_a
        self.trans_b = trans_b
        self.a_nz_flag = a_nz_flag
        self.b_nz_flag = b_nz_flag
        self.c_nz_flag = c_nz_flag


def gen_mm_data(input_config: ShapeConfig, output_dir: Path):
    shape_a = [input_config.m, input_config.k]
    shape_b = [input_config.k, input_config.n]
    shape_c = [input_config.m, input_config.n]

    a_path = Path(output_dir, 'mat_a.bin')
    b_path = Path(output_dir, 'mat_b.bin')
    c_path = Path(output_dir, 'mat_c.bin')

    if input_config.in_dtype == INT8:
        a = np.random.randint(-4, 5, shape_a).astype(INT8)
        b = np.random.randint(-4, 5, shape_b).astype(INT8)
        c = np.matmul(a.astype(INT32), b.astype(INT32)).astype(INT32)
    elif input_config.in_dtype == FP16:
        a = np.random.uniform(-1, 1, shape_a).astype(FP16)
        b = np.random.uniform(-1, 1, shape_b).astype(FP16)
        c = np.matmul(a.astype(FP32), b.astype(FP32))
    elif input_config.in_dtype == BF16:
        a = np.random.uniform(-1, 1, shape_a).astype(BF16)
        b = np.random.uniform(-1, 1, shape_b).astype(BF16)
        c = np.matmul(a.astype(FP32), b.astype(FP32))
    elif input_config.in_dtype == FP32:
        a = np.random.uniform(-1, 1, shape_a).astype(FP32)
        b = np.random.uniform(-1, 1, shape_b).astype(FP32)
        c = np.matmul(a.astype(FP32), b.astype(FP32))

    if input_config.trans_a:
        a = a.transpose(1, 0)
    if input_config.a_nz_flag:
        a = nd_to_fractal_nz(a)

    if input_config.trans_b:
        b = b.transpose(1, 0)
    if input_config.b_nz_flag:
        b = nd_to_fractal_nz(b)
    a.tofile(a_path)
    b.tofile(b_path)

    if input_config.c_nz_flag:
        c = nd_to_fractal_nz(c)
    c.tofile(c_path)


@GoldenRegister.reg_golden_func(
    case_names=[
        #matmul
        "DynamicMatmulInterpreterTest.mm_A_B_ND_bf16",
        "DynamicMatmulInterpreterTest.mm_A_B_NZ_bf16",
        "DynamicMatmulInterpreterTest.mm_A_Bt_ND_fp16",
        "DynamicMatmulInterpreterTest.mm_A_Bt_NZ_fp16",
        "DynamicMatmulInterpreterTest.mm_A_B_NZ_int8",
        "DynamicMatmulInterpreterTest.mm_A_Bt_NZ_int8",
        "DynamicMatmulInterpreterTest.mm_A_B_ND_bf16_tile1",
        "DynamicMatmulInterpreterTest.mm_A_Bt_ND_fp16_tile2",
        "DynamicMatmulInterpreterTest.mm_A_B_NZ_int8_tile3",
        "DynamicMatmulInterpreterTest.mm_A_Bt_NZ_int8_tile4",
        "DynamicMatmulInterpreterTest.mm_A_ND_B_ND_C_NZ",
        "DynamicMatmulInterpreterTest.mm_AT_B_ANZ_BND_bf16",
        "DynamicMatmulInterpreterTest.mm_AT_BT_AND_BND_bf16",
        "DynamicMatmulInterpreterTest.mm_AT_B_ANZ_BND_fp16_UNALIGN",
    ]
)
def gen_dynamic_mm_golden(case_name: str, output: Path) -> bool:
    if case_name == "DynamicMatmulInterpreterTest.mm_A_B_ND_bf16":
        input_config = ShapeConfig(128, 256, 512, BF16, FP32, False, False, False, False, False)
        gen_mm_data(input_config, output)
        return True
    if case_name == "DynamicMatmulInterpreterTest.mm_A_B_NZ_bf16":
        input_config = ShapeConfig(16, 32, 512, BF16, FP32, False, False, False, True, False)
        gen_mm_data(input_config, output)
        return True
    if case_name == "DynamicMatmulInterpreterTest.mm_A_Bt_ND_fp16":
        input_config = ShapeConfig(128, 257, 511, FP16, FP32, False, True, False, False, False)
        gen_mm_data(input_config, output)
        return True
    if case_name == "DynamicMatmulInterpreterTest.mm_A_Bt_NZ_fp16":
        input_config = ShapeConfig(1, 512, 256, FP16, FP32, False, True, False, True, False)
        gen_mm_data(input_config, output)
        return True
    if case_name == "DynamicMatmulInterpreterTest.mm_A_B_NZ_int8":
        input_config = ShapeConfig(16, 32, 512, INT8, INT32, False, False, False, True, False)
        gen_mm_data(input_config, output)
        return True
    if case_name == "DynamicMatmulInterpreterTest.mm_A_Bt_NZ_int8":
        input_config = ShapeConfig(1, 512, 256, INT8, INT32, False, True, False, True, False)
        gen_mm_data(input_config, output)
        return True
    if case_name == "DynamicMatmulInterpreterTest.mm_A_B_ND_bf16_tile1":
        input_config = ShapeConfig(128, 256, 512, BF16, FP32, False, False, False, False, False)
        gen_mm_data(input_config, output)
        return True
    if case_name == "DynamicMatmulInterpreterTest.mm_A_Bt_ND_fp16_tile2":
        input_config = ShapeConfig(16, 512, 512, FP16, FP32, False, True, False, False, False)
        gen_mm_data(input_config, output)
        return True
    if case_name == "DynamicMatmulInterpreterTest.mm_A_B_NZ_int8_tile3":
        input_config = ShapeConfig(16, 32, 512, INT8, INT32, False, False, False, True, False)
        gen_mm_data(input_config, output)
        return True
    if case_name == "DynamicMatmulInterpreterTest.mm_A_Bt_NZ_int8_tile4":
        input_config = ShapeConfig(1, 512, 256, INT8, INT32, False, True, False, True, False)
        gen_mm_data(input_config, output)
        return True
    if case_name == "DynamicMatmulInterpreterTest.mm_A_ND_B_ND_C_NZ":
        input_config = ShapeConfig(16, 192, 128, FP16, FP32, False, False, False, False, True)
        gen_mm_data(input_config, output)
        return True
    else:
        logging.error("Can't get func to gen golden, case(%s)", case_name)
        return False
