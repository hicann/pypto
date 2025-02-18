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
""" BatchMatmul相关用例 Golden 生成逻辑.

本脚本有 2 种执行模式:
1. CI批跑时, 由 tests/cmake/scripts/golden_ctrl.py 调用, 为避免日志过多, 此时 logging 级别为 logging.INFO;
2. 单独调试时, 本脚本单独被调用, 此时 logging 级别为 logging.DEBUG;
"""
import sys
import logging
from pathlib import Path
from typing import List

import numpy as np
from ml_dtypes import bfloat16

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

dtype_f32 = np.float32
dtype_f16 = np.float16
dtype_bf16 = bfloat16
dtype_int32 = np.int32
dtype_int8 = np.int8


def is_exist_and_shape_correct(mat_path: Path, shape, dtype):
    if not mat_path.exists():
        return False
    return mat_path.stat().st_size == np.prod(shape) * np.dtype(dtype).itemsize


@GoldenRegister.reg_golden_func(
    case_names=[
        "OnBoardTest.test_BMM_Simple",
        "OnBoardTest.test_BMM_Simple_FP32_BT",
        "OnBoardTest.test_BMM_Simple_FP32",
        "OnBoardTest.test_BMM_Simple_FP32_256_256",
        "OnBoardTest.test_BMM_Simple_A8W8O32",
        "OnBoardTest.test_BMM_Simple_A8W8O32_4_4_64_64",
        "OnBoardTest.test_BMM_Simple_A8W8O32_1_4_4096_7168",
        "OnBoardTest.test_BMM_UNALIGN_2_1024_32",
        "OnBoardTest.test_BMM_UNALIGN_32_4_128_512",
        "OnBoardTest.test_BMM_BF16",
        "OnBoardTest.test_BMM_UNALIGN_1_32_32",
        "OnBoardTest.test_BMM_UNALIGN_1_1024_32",
        "OnBoardTest.test_BMM_UNALIGN_16_35_32",
        "OnBoardTest.test_BMM_UNALIGN_16_32_35",
        "OnBoardTest.test_BMM_NZ_1_32_32_32",
        "OnBoardTest.test_BMM_NZ_1_32_32_320",
        "OnBoardTest.test_BMM_NZ_1_64_32_16",
        "OnBoardTest.test_BMM_NZ_1_128_128_128",
        "OnBoardTest.test_BMM_NZ_1_32_7168_576",
        "OnBoardTest.test_BMM_NZ_1_32_7168_1536",
        "OnBoardTest.test_BMM_NZ_1_4_4096_7168_ACC",
        "OnBoardTest.test_BMM_NZ_1_128_128_128_ACC",
        "OnBoardTest.test_BMM_NZ_1_32_2048_7168_ACC",
        "OnBoardTest.test_BMM_NZ_1_128_256_128_Batch",
        "OnBoardTest.test_BMMT_NZ_1_128_256_128_Batch",
        "OnBoardTest.test_BMM_NZ_1_64_32_16_A8W8O32",
        "OnBoardTest.test_BMM_NZ_1_128_128_128_A8W8O32",
        "OnBoardTest.test_BMM_FP16",
        "OnBoardTest.test_BMM_post",
        "OnBoardTest.test_BMM_3D_Brc",
        "OnBoardTest.test_BMM_3D_Brc_Transpose",
        "OnBoardTest.test_BMM_4D_Brc",
        "OnBoardTest.test_BMM_4D_Brc_Transpose",
        "OnBoardTest.test_BMM_9_16_7168_2048",
        "OnBoardTest.test_BMM_NZ_Simple",
    ]
)
def gen_batchmatmul_golden(case_name: str, output: Path) -> bool:
    mat_a_path = Path(output, "mat_a.bin")
    mat_b_path = Path(output, "mat_b.bin")
    mat_c_path = Path(output, "mat_c.bin")
    complete = False
    transpose = False
    is_a_nz_input = False
    is_b_nz_input = False
    is_batch_case = False
    batch_num = 1
    m_value = 32
    n_value = 32
    k_value = 32
    mat_a_shape = [0, 0, 0]
    mat_b_shape = [0, 0, 0]
    dtype_in = dtype_int8
    dtype_out = dtype_int32

    if case_name == "OnBoardTest.test_BMM_Simple":
        mat_a_shape = [2, 32, 128]
        mat_b_shape = [2, 128, 512]
        dtype_in = dtype_f16
        dtype_out = dtype_f32
    elif case_name == "OnBoardTest.test_BMM_Simple_FP32_BT":
        mat_a_shape = [1, 32, 32]
        mat_b_shape = [1, 32, 32]
        dtype_in = dtype_f32
        dtype_out = dtype_f32
        transpose = True
    elif case_name == "OnBoardTest.test_BMM_Simple_FP32":
        mat_a_shape = [1, 64, 64]
        mat_b_shape = [1, 64, 64]
        dtype_in = dtype_f32
        dtype_out = dtype_f32
    elif case_name == "OnBoardTest.test_BMM_Simple_FP32_256_256":
        mat_a_shape = [1, 256, 256]
        mat_b_shape = [1, 256, 256]
        dtype_in = dtype_f32
        dtype_out = dtype_f32
    elif case_name == "OnBoardTest.test_BMM_UNALIGN_2_1024_32":
        mat_a_shape = [2, 1, 1024]
        mat_b_shape = [2, 1024, 32]
        dtype_in = dtype_f16
        dtype_out = dtype_f32
    elif case_name == "OnBoardTest.test_BMM_UNALIGN_32_4_128_512":
        mat_a_shape = [32, 4, 128]
        mat_b_shape = [32, 128, 512]
        dtype_in = dtype_f16
        dtype_out = dtype_f16
    elif case_name == "OnBoardTest.test_BMM_NZ_1_32_32_32":
        is_a_nz_input = True
        batch_num = 1
        m_value = int(32)
        n_value = int(32)
        k_value = int(32)
        mat_a_shape = [batch_num, m_value, k_value]
        mat_b_shape = [batch_num, k_value, n_value]
        dtype_in = dtype_f16
        dtype_out = dtype_f32
    elif case_name == "OnBoardTest.test_BMM_NZ_1_32_32_320":
        is_a_nz_input = True
        batch_num = 1
        m_value = int(32)
        n_value = int(320)
        k_value = int(32)
        mat_a_shape = [batch_num, m_value, k_value]
        mat_b_shape = [batch_num, k_value, n_value]
        dtype_in = dtype_f16
        dtype_out = dtype_f32
    elif case_name == "OnBoardTest.test_BMM_NZ_1_64_32_16":
        is_a_nz_input = True
        batch_num = 1
        m_value = int(64)
        n_value = int(32)
        k_value = int(16)
        mat_a_shape = [batch_num, m_value, k_value]
        mat_b_shape = [batch_num, k_value, n_value]
        dtype_in = dtype_f16
        dtype_out = dtype_f32
    elif case_name == "OnBoardTest.test_BMM_NZ_1_64_32_16_A8W8O32":
        is_b_nz_input = True
        batch_num = 1
        m_value = int(64)
        n_value = int(32)
        k_value = int(16)
        mat_a_shape = [batch_num, m_value, k_value]
        mat_b_shape = [batch_num, k_value, n_value]
        dtype_in = dtype_int8
        dtype_out = dtype_int32
    elif case_name == "OnBoardTest.test_BMM_NZ_1_128_128_128":
        is_b_nz_input = True
        batch_num = 1
        m_value = int(128)
        n_value = int(128)
        k_value = int(128)
        mat_a_shape = [batch_num, m_value, k_value]
        mat_b_shape = [batch_num, k_value, n_value]
        dtype_in = dtype_f16
        dtype_out = dtype_f32
    elif case_name == "OnBoardTest.test_BMM_NZ_1_32_7168_576":
        is_b_nz_input = True
        batch_num = 1
        m_value = int(32)
        n_value = int(576)
        k_value = int(7168)
        mat_a_shape = [batch_num, m_value, k_value]
        mat_b_shape = [batch_num, k_value, n_value]
        dtype_in = dtype_f16
        dtype_out = dtype_f32
    elif case_name == "OnBoardTest.test_BMM_NZ_1_32_7168_1536":
        is_b_nz_input = True
        batch_num = 1
        m_value = int(32)
        n_value = int(1536)
        k_value = int(7168)
        mat_a_shape = [batch_num, m_value, k_value]
        mat_b_shape = [batch_num, k_value, n_value]
        dtype_in = dtype_f16
        dtype_out = dtype_f32
    elif case_name == "OnBoardTest.test_BMM_NZ_1_4_4096_7168_ACC":
        is_b_nz_input = True
        batch_num = 1
        m_value = int(4)
        n_value = int(7168)
        k_value = int(4096)
        mat_a_shape = [batch_num, m_value, k_value]
        mat_b_shape = [batch_num, k_value, n_value]
        dtype_in = dtype_f16
        dtype_out = dtype_f32
    elif case_name == "OnBoardTest.test_BMM_NZ_1_128_128_128_ACC":
        is_b_nz_input = True
        batch_num = 1
        m_value = int(128)
        n_value = int(128)
        k_value = int(128)
        mat_a_shape = [batch_num, m_value, k_value]
        mat_b_shape = [batch_num, k_value, n_value]
        dtype_in = dtype_f16
        dtype_out = dtype_f32
    elif case_name == "OnBoardTest.test_BMM_NZ_1_32_2048_7168_ACC":
        is_b_nz_input = True
        batch_num = 1
        m_value = int(32)
        n_value = int(2048)
        k_value = int(7168)
        mat_a_shape = [batch_num, m_value, k_value]
        mat_b_shape = [batch_num, k_value, n_value]
        dtype_in = dtype_f16
        dtype_out = dtype_f32
    elif case_name == "OnBoardTest.test_BMM_NZ_1_128_256_128_Batch":
        is_b_nz_input = True
        # is_batch_case = True
        batch_num = 2
        m_value = int(128)
        n_value = int(256)
        k_value = int(128)
        mat_a_shape = [batch_num, m_value, k_value]
        mat_b_shape = [batch_num, k_value, n_value]
        dtype_in = dtype_f16
        dtype_out = dtype_f32
    elif case_name == "OnBoardTest.test_BMMT_NZ_1_128_256_128_Batch":
        is_b_nz_input = True
        transpose = True
        batch_num = 2
        m_value = int(128)
        n_value = int(256)
        k_value = int(128)
        mat_a_shape = [batch_num, m_value, k_value]
        mat_b_shape = [batch_num, n_value, k_value]
        dtype_in = dtype_f16
        dtype_out = dtype_f32
    elif case_name == "OnBoardTest.test_BMM_NZ_1_128_128_128_A8W8O32":
        is_b_nz_input = True
        batch_num = 1
        m_value = int(128)
        n_value = int(128)
        k_value = int(128)
        mat_a_shape = [batch_num, m_value, k_value]
        mat_b_shape = [batch_num, k_value, n_value]
        dtype_in = dtype_int8
        dtype_out = dtype_int32
    elif case_name == "OnBoardTest.test_BMM_UNALIGN_1_32_32":
        mat_a_shape = [1, 1, 32]
        mat_b_shape = [1, 32, 32]
        dtype_in = dtype_f16
        dtype_out = dtype_f32
    elif case_name == "OnBoardTest.test_BMM_BF16":
        mat_a_shape = [2, 64, 64]
        mat_b_shape = [2, 64, 64]
        dtype_in = dtype_bf16
        dtype_out = dtype_bf16
    elif case_name == "OnBoardTest.test_BMM_post":
        mat_a_shape = [32, 32, 512]
        mat_b_shape = [32, 512, 128]
        dtype_in = dtype_f16
        dtype_out = dtype_f16
    elif case_name == "OnBoardTest.test_BMM_3D_Brc":
        mat_a_shape = [2, 16, 256]
        mat_b_shape = [1, 256, 128]
        dtype_in = dtype_f16
        dtype_out = dtype_f32
    elif case_name == "OnBoardTest.test_BMM_3D_Brc_Transpose":
        mat_a_shape = [2, 64, 256]
        mat_b_shape = [1, 128, 256]
        dtype_in = dtype_f16
        dtype_out = dtype_f32
        transpose = True
    elif case_name == "OnBoardTest.test_BMM_4D_Brc":
        mat_a_shape = [2, 2, 128, 128]
        mat_b_shape = [2, 1, 128, 64]
        dtype_in = dtype_f16
        dtype_out = dtype_f32
    elif case_name == "OnBoardTest.test_BMM_4D_Brc_Transpose":
        mat_a_shape = [2, 2, 128, 128]
        mat_b_shape = [2, 1, 64, 128]
        dtype_in = dtype_f16
        dtype_out = dtype_f32
        transpose = True
    elif case_name == "OnBoardTest.test_BMM_9_16_7168_2048":
        mat_a_shape = [1, 16, 7168]
        mat_b_shape = [9, 7168, 2048]
        dtype_in = dtype_f16
        dtype_out = dtype_f16
    elif case_name == "OnBoardTest.test_BMM_Simple_A8W8O32":
        mat_a_shape = [1, 32, 64]
        mat_b_shape = [1, 64, 64]
        dtype_in = dtype_int8
        dtype_out = dtype_int32
    elif case_name == "OnBoardTest.test_BMM_Simple_A8W8O32_4_4_64_64":
        mat_a_shape = [4, 4, 64]
        mat_b_shape = [4, 64, 64]
        dtype_in = dtype_int8
        dtype_out = dtype_int32
    elif case_name == "OnBoardTest.test_BMM_Simple_A8W8O32_1_4_4096_7168":
        mat_a_shape = [1, 4, 4096]
        mat_b_shape = [1, 4096, 7168]
        dtype_in = dtype_int8
        dtype_out = dtype_int32
    else:
        logging.error("Can't get func to gen golden, Case(%s)", case_name)
        return False

    complete = mat_a_path.exists() and mat_a_path.exists() and mat_a_path.exists()
    if complete:
        logging.debug("Case(%s), Golden already complete.", case_name)

    mat_a = np.random.uniform(-2, 2, mat_a_shape).astype(dtype_in)
    mat_b = np.random.uniform(-2, 2, mat_b_shape).astype(dtype_in)
    mat_b_to_mul = mat_b
    if is_exist_and_shape_correct(mat_a_path, mat_a_shape, dtype_in) and \
            is_exist_and_shape_correct(mat_b_path, mat_b_shape, dtype_in) and \
            mat_c_path.exists():
        logging.debug("Case(%s), Golden already complete.", case_name)
    if transpose:
        # transpose back if is matmul transpose.
        mat_b_to_mul = mat_b.transpose(
            0, 2, 1) if len(mat_b.shape) == 3 else mat_b.transpose(0, 1, 3, 2)

    mat_c = np.matmul(mat_a.astype(np.float32), mat_b_to_mul.astype(np.float32)).astype(dtype_out)
    c0_size = int(16)
    if dtype_in == dtype_int8:
        c0_size = int(32)

    if is_a_nz_input:
        kc0_value = k_value // c0_size
        mat_a = mat_a.reshape((batch_num, m_value, kc0_value, c0_size))
        mat_a = np.transpose(mat_a, (0, 2, 1, 3))
        mat_a.tofile(mat_a_path)
    else:
        mat_a.tofile(mat_a_path)

    if is_b_nz_input and transpose:
        kc0_value = k_value // c0_size
        mat_b = mat_b.reshape((batch_num, n_value, kc0_value, c0_size))
        mat_b = np.transpose(mat_b, (0, 2, 1, 3))
        mat_b.tofile(mat_b_path)
    elif is_b_nz_input:
        nc0_value = n_value // c0_size
        mat_b = mat_b.reshape((batch_num, k_value, nc0_value, c0_size))
        mat_b = np.transpose(mat_b, (0, 2, 1, 3))
        mat_b.tofile(mat_b_path)
    else:
        mat_b.tofile(mat_b_path)

    mat_c.tofile(mat_c_path)
    return True


def main() -> bool:
    """
    单独调试 入口函数
    """
    # 用例名称
    case_name_list: List[str] = [
        "OnBoardTest.test_BMM_Simple",
    ]
    # 函数调用
    ret: bool = True
    for cs in case_name_list:
        output: Path = Path(g_src_root, "build/output/bin/golden", cs).resolve()
        output.mkdir(parents=True, exist_ok=True)
        ret = gen_batchmatmul_golden(case_name=cs, output=output)
    return ret


if __name__ == "__main__":
    exit(0 if main() else 1)
