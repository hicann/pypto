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
""" BinaryOperator 相关用例 Golden 生成逻辑.

本脚本有 2 种执行模式:
1. CI批跑时, 由 tests/cmake/scripts/golden_ctrl.py 调用, 为避免日志过多, 此时 logging 级别为 logging.INFO;
2. 单独调试时, 本脚本单独被调用, 此时 logging 级别为 logging.DEBUG;
"""
import sys
import logging
from pathlib import Path
from typing import List

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


@GoldenRegister.reg_golden_func(
    case_names=[
        "OnBoardTest.test_operation_tensor_8_8_1_1_to_8_8_1_256_tileop_sub",
        "OnBoardTest.test_operation_tensor_2_2_8_8_expand_add",
        "OnBoardTest.test_operation_tensor_4_4_16_16_expand_add",
        "OnBoardTest.test_operation_tensor_1_1_32_to_16_32_32_expand_add",
        "OnBoardTest.test_operation_tensor_8_16_1_to_8_16_16_expand_add",
        "OnBoardTest.test_operation_tensor_8_8_1_to_8_8_7168_expand_Mul_moe",
        "OnBoardTest.test_operation_tensor_8_1_16_to_8_16_16_expand_add",
        "OnBoardTest.test_operation_tensor_1_16_16_to_8_16_16_expand_add",
        "OnBoardTest.test_operation_tensor_8_1_1_to_8_16_16_expand_add",
        "OnBoardTest.test_operation_tensor_1_1_1_to_8_16_16_expand_add",
        "OnBoardTest.test_operation_tensor_16_32_32_to_16_32_1_tileop_mul",
        "OnBoardTest.test_operation_tensor_1_1_1_64_to_1_128_1_64_tileop_mul01",
        "OnBoardTest.test_operation_tensor_1_1_1_64_to_1_128_1_64_tileop_mul02",
        "OnBoardTest.test_operation_tensor_1_1_64_to_32_1_64_tileop_mul03",
        "OnBoardTest.test_operation_tensor_32_32_1_1_to_32_32_1_256_tileop_add",
        "OnBoardTest.test_operation_tensor_32_32_1_1_to_32_32_1_256_tileop_sub",
        "OnBoardTest.test_operation_tensor_32_32_1_1_to_32_32_1_256_tileop_mul",
        "OnBoardTest.test_operation_tensor_32_32_1_1_to_32_32_1_256_tileop_div",
        "OnBoardTest.test_operation_tensor_8_8_1_to_8_8_7168_expand_mul",
        "OnBoardTest.test_operation_tensor_8_8_1_to_8_8_7168_expand_sub",
        "OnBoardTest.test_operation_tensor_dim2_add",
        "OnBoardTest.test_operation_tensor_dim4_add",
        "OnBoardTest.test_operation_tensor_16_16_64_64_tileop_add",
        "OnBoardTest.test_operation_tensor_16_16_64_64_tileop_sub",
        "OnBoardTest.test_operation_tensor_16_16_64_64_tileop_mul",
        "OnBoardTest.test_operation_tensor_16_16_64_64_tileop_div",
        "OnBoardTest.test_operation_tensor_8_80_80_tileop_add",
        "OnBoardTest.test_operation_tensor_1_n_to_m_n_mul",
        "OnBoardTest.test_operation_tensor_8_80_80_tileop_sub",
        "OnBoardTest.test_operation_tensor_8_80_80_tileop_mul",
        "OnBoardTest.test_operation_tensor_8_80_80_tileop_div",
        "OnBoardTest.test_operation_tensor_64_128_tileop_add",
        "OnBoardTest.test_operation_tensor_64_128_tileop_sub",
        "OnBoardTest.test_operation_tensor_64_128_tileop_mul",
        "OnBoardTest.test_operation_tensor_64_128_tileop_div",
        "OnBoardTest.test_operation_scalar_dim2_add",
        "OnBoardTest.test_operation_scalar_dim2_add_FP16",
        "OnBoardTest.test_operation_scalar_dim3_add",
        "OnBoardTest.test_operation_scalar_dim4_add",
        "OnBoardTest.test_operation_scalar_dim2_sub",
        "OnBoardTest.test_operation_scalar_dim3_sub",
        "OnBoardTest.test_operation_scalar_dim4_sub",
        "OnBoardTest.test_operation_scalar_dim2_mul",
        "OnBoardTest.test_operation_scalar_dim3_mul",
        "OnBoardTest.test_operation_scalar_dim4_mul",
        "OnBoardTest.test_operation_scalar_dim2_div",
        "OnBoardTest.test_operation_scalar_dim1_div",
        "OnBoardTest.test_operation_scalar_dim3_div",
        "OnBoardTest.test_operation_scalar_dim4_div",
        "OnBoardTest.test_operation_scalar_32_32_1_256_mul",
        # unalign
        "OnBoardTest.test_operation_tensor_16_16_64_65_tileop_add_unalign",
        "OnBoardTest.test_operation_tensor_16_16_39_65_tileop_add_unalign",
        "OnBoardTest.test_operation_tensor_32_1_tileop_add_unalign",
        "OnBoardTest.test_operation_tensor_32_1_tileop_sub_unalign",
        "OnBoardTest.test_operation_tensor_32_1_tileop_mul_unalign",
        "OnBoardTest.test_operation_add_vs_dim2_unalign",
        "OnBoardTest.test_operation_mul_vs_dim3_unalign",
        "OnBoardTest.test_operation_sub_vs_dim4_unalign",
        "OnBoardTest.test_operation_div_vs_dim1_unalign",
        "OnBoardTest.test_mul_large_row",
        # single tile op test
        "TestTileOpAdd.TestAddDim2",
        # matmul add
        "OnBoardTest.test_matmul_add_dynamic",
    ]
)
def binary_operator_func1(case_name: str, output: Path) -> bool:
    in0 = 2
    in1 = 2
    row = 64
    col = 64
    # row = 16
    # col = 16
    shape_i = [row, col]

    shape_dim3_i = [in0, row, col]

    shape_dim4_i = [in0, in1, row, col]
    shape_16_16_64_64_i = [16, 16, 64, 64]
    shape_8_80_80_i = [8, 80, 80]
    shape_64_128_i = [64, 128]

    shape_2_2_8_8_i = [2, 2, 8, 8]
    shape_2_1_8_8_i = [2, 1, 8, 8]
    shape_4_4_16_16_i = [4, 4, 16, 16]
    shape_4_1_16_16_i = [4, 1, 16, 16]

    shape_16_32_32_i = [16, 32, 32]
    shape_1_1_32_i = [1, 1, 32]

    shape_8_16_1_i = [8, 16, 1]
    shape_8_16_16_i = [8, 16, 16]

    shape_8_8_7168_i = [8, 8, 7168]
    shape_8_8_1_i = [8, 8, 1]

    shape_8_1_16_i = [8, 1, 16]
    shape_1_16_16_i = [1, 16, 16]
    shape_8_1_1_i = [8, 1, 1]
    shape_1_1_1_i = [1, 1, 1]

    dtype = np.float32

    if case_name == "OnBoardTest.test_operation_tensor_dim2_add" or case_name == "TestTileOpAdd.TestAddDim2":
        x_path = Path(output, 'add_x.bin')
        y_path = Path(output, 'add_y.bin')
        o_path = Path(output, 'add_res.bin')
        complete = x_path.exists() and y_path.exists() and o_path.exists()
        if complete:
            logging.debug("Case(%s), Golden complete.", case_name)
        else:
            x = np.random.uniform(-1, 1, shape_i).astype(dtype)
            x.tofile(x_path)
            y = np.random.uniform(-1, 1, shape_i).astype(dtype)
            y.tofile(y_path)
            x = x + y
            x.tofile(o_path)
    elif case_name == "OnBoardTest.test_operation_scalar_dim2_add_FP16":
        x_path = Path(output, 'adds_2d_x.bin')
        o_path = Path(output, 'adds_2d_res.bin')
        complete = x_path.exists() and o_path.exists()
        if complete:
            logging.debug("Case(%s), Golden complete.", case_name)
        else:
            x = np.random.uniform(-1, 1, shape_i).astype(np.float16)
            x.tofile(x_path)
            x = x + 1.5
            x.tofile(o_path)
    elif case_name == "OnBoardTest.test_operation_tensor_dim4_add":
        x_path = Path(output, 'add_dim4_x.bin')
        y_path = Path(output, 'add_dim4_y.bin')
        o_path = Path(output, 'add_dim4_res.bin')
        complete = x_path.exists() and y_path.exists() and o_path.exists()
        if complete:
            logging.debug("Case(%s), Golden complete.", case_name)
        else:
            x = np.random.uniform(-1, 1, shape_dim4_i).astype(dtype)
            x.tofile(x_path)
            y = np.random.uniform(-1, 1, shape_dim4_i).astype(dtype)
            y.tofile(y_path)
            x = x + y
            x.tofile(o_path)
    elif case_name == "OnBoardTest.test_operation_tensor_16_16_64_64_tileop_add":
        x_path = Path(output, 'x.bin')
        y_path = Path(output, 'y.bin')
        o_path = Path(output, 'res.bin')
        complete = x_path.exists() and y_path.exists() and o_path.exists()
        if complete:
            logging.debug("Case(%s), Golden complete.", case_name)
        else:
            x = np.random.uniform(-1, 1, shape_16_16_64_64_i).astype(dtype)
            x.tofile(x_path)
            y = np.random.uniform(-1, 1, shape_16_16_64_64_i).astype(dtype)
            y.tofile(y_path)
            x = x + y
            x.tofile(o_path)
    elif case_name == "OnBoardTest.test_operation_tensor_16_16_64_64_tileop_sub":
        x_path = Path(output, 'x.bin')
        y_path = Path(output, 'y.bin')
        o_path = Path(output, 'res.bin')
        complete = x_path.exists() and y_path.exists() and o_path.exists()
        if complete:
            logging.debug("Case(%s), Golden complete.", case_name)
        else:
            x = np.random.uniform(-1, 1, shape_16_16_64_64_i).astype(dtype)
            x.tofile(x_path)
            y = np.random.uniform(-1, 1, shape_16_16_64_64_i).astype(dtype)
            y.tofile(y_path)
            x = x - y
            x.tofile(o_path)
    elif case_name == "OnBoardTest.test_operation_tensor_16_16_64_64_tileop_mul":
        x_path = Path(output, 'x.bin')
        y_path = Path(output, 'y.bin')
        o_path = Path(output, 'res.bin')
        complete = x_path.exists() and y_path.exists() and o_path.exists()
        if complete:
            logging.debug("Case(%s), Golden complete.", case_name)
        else:
            x = np.random.uniform(-1, 1, shape_16_16_64_64_i).astype(dtype)
            x.tofile(x_path)
            y = np.random.uniform(-1, 1, shape_16_16_64_64_i).astype(dtype)
            y.tofile(y_path)
            x = x * y
            x.tofile(o_path)
    elif case_name == "OnBoardTest.test_operation_tensor_16_16_64_64_tileop_div":
        x_path = Path(output, 'x.bin')
        y_path = Path(output, 'y.bin')
        o_path = Path(output, 'res.bin')
        complete = x_path.exists() and y_path.exists() and o_path.exists()
        if complete:
            logging.debug("Case(%s), Golden complete.", case_name)
        else:
            x = np.random.uniform(-1, 1, shape_16_16_64_64_i).astype(dtype)
            x.tofile(x_path)
            y = np.random.uniform(-1, 1, shape_16_16_64_64_i).astype(dtype)
            y.tofile(y_path)
            x = x / y
            x.tofile(o_path)
    elif case_name == "OnBoardTest.test_operation_tensor_8_80_80_tileop_add":
        x_path = Path(output, 'x.bin')
        y_path = Path(output, 'y.bin')
        o_path = Path(output, 'res.bin')
        complete = x_path.exists() and y_path.exists() and o_path.exists()
        if complete:
            logging.debug("Case(%s), Golden complete.", case_name)
        else:
            x = np.random.uniform(-1, 1, shape_8_80_80_i).astype(dtype)
            x.tofile(x_path)
            y = np.random.uniform(-1, 1, shape_8_80_80_i).astype(dtype)
            y.tofile(y_path)
            x = x + y
            x.tofile(o_path)
    elif case_name == "OnBoardTest.test_operation_tensor_8_80_80_tileop_sub":
        x_path = Path(output, 'x.bin')
        y_path = Path(output, 'y.bin')
        o_path = Path(output, 'res.bin')
        complete = x_path.exists() and y_path.exists() and o_path.exists()
        if complete:
            logging.debug("Case(%s), Golden complete.", case_name)
        else:
            x = np.random.uniform(-1, 1, shape_8_80_80_i).astype(dtype)
            x.tofile(x_path)
            y = np.random.uniform(-1, 1, shape_8_80_80_i).astype(dtype)
            y.tofile(y_path)
            x = x - y
            x.tofile(o_path)
    elif case_name == "OnBoardTest.test_operation_tensor_8_80_80_tileop_mul":
        x_path = Path(output, 'x.bin')
        y_path = Path(output, 'y.bin')
        o_path = Path(output, 'res.bin')
        complete = x_path.exists() and y_path.exists() and o_path.exists()
        if complete:
            logging.debug("Case(%s), Golden complete.", case_name)
        else:
            x = np.random.uniform(-1, 1, shape_8_80_80_i).astype(dtype)
            x.tofile(x_path)
            y = np.random.uniform(-1, 1, shape_8_80_80_i).astype(dtype)
            y.tofile(y_path)
            x = x * y
            x.tofile(o_path)
    elif case_name == "OnBoardTest.test_operation_tensor_8_80_80_tileop_div":
        x_path = Path(output, 'x.bin')
        y_path = Path(output, 'y.bin')
        o_path = Path(output, 'res.bin')
        complete = x_path.exists() and y_path.exists() and o_path.exists()
        if complete:
            logging.debug("Case(%s), Golden complete.", case_name)
        else:
            x = np.random.uniform(-1, 1, shape_8_80_80_i).astype(dtype)
            x.tofile(x_path)
            y = np.random.uniform(-1, 1, shape_8_80_80_i).astype(dtype)
            y.tofile(y_path)
            x = x / y
            x.tofile(o_path)
    elif case_name == "OnBoardTest.test_operation_tensor_64_128_tileop_add":
        x_path = Path(output, 'x.bin')
        y_path = Path(output, 'y.bin')
        o_path = Path(output, 'res.bin')
        complete = x_path.exists() and y_path.exists() and o_path.exists()
        if complete:
            logging.debug("Case(%s), Golden complete.", case_name)
        else:
            x = np.random.uniform(-1, 1, shape_64_128_i).astype(dtype)
            x.tofile(x_path)
            y = np.random.uniform(-1, 1, shape_64_128_i).astype(dtype)
            y.tofile(y_path)
            x = x + y
            x.tofile(o_path)
    elif case_name == "OnBoardTest.test_operation_tensor_64_128_tileop_sub":
        x_path = Path(output, 'x.bin')
        y_path = Path(output, 'y.bin')
        o_path = Path(output, 'res.bin')
        complete = x_path.exists() and y_path.exists() and o_path.exists()
        if complete:
            logging.debug("Case(%s), Golden complete.", case_name)
        else:
            x = np.random.uniform(-1, 1, shape_64_128_i).astype(dtype)
            x.tofile(x_path)
            y = np.random.uniform(-1, 1, shape_64_128_i).astype(dtype)
            y.tofile(y_path)
            x = x - y
            x.tofile(o_path)
    elif case_name == "OnBoardTest.test_operation_tensor_64_128_tileop_mul":
        x_path = Path(output, 'x.bin')
        y_path = Path(output, 'y.bin')
        o_path = Path(output, 'res.bin')
        complete = x_path.exists() and y_path.exists() and o_path.exists()
        if complete:
            logging.debug("Case(%s), Golden complete.", case_name)
        else:
            x = np.random.uniform(-1, 1, shape_64_128_i).astype(dtype)
            x.tofile(x_path)
            y = np.random.uniform(-1, 1, shape_64_128_i).astype(dtype)
            y.tofile(y_path)
            x = x * y
            x.tofile(o_path)
    elif case_name == "OnBoardTest.test_operation_tensor_64_128_tileop_div":
        x_path = Path(output, 'x.bin')
        y_path = Path(output, 'y.bin')
        o_path = Path(output, 'res.bin')
        complete = x_path.exists() and y_path.exists() and o_path.exists()
        if complete:
            logging.debug("Case(%s), Golden complete.", case_name)
        else:
            x = np.random.uniform(-1, 1, shape_64_128_i).astype(dtype)
            x.tofile(x_path)
            y = np.random.uniform(-1, 1, shape_64_128_i).astype(dtype)
            y.tofile(y_path)
            x = x / y
            x.tofile(o_path)
    elif case_name == "OnBoardTest.test_operation_scalar_dim2_add":
        x_path = Path(output, 'adds_2d_x.bin')
        o_path = Path(output, 'adds_2d_res.bin')
        complete = x_path.exists() and o_path.exists()
        if complete:
            logging.debug("Case(%s), Golden complete.", case_name)
        else:
            x = np.random.uniform(-1, 1, shape_i).astype(dtype)
            x.tofile(x_path)
            x = x + 1.5
            x.tofile(o_path)
    elif case_name == "OnBoardTest.test_operation_scalar_dim3_add":
        x_path = Path(output, 'adds_3d_x.bin')
        o_path = Path(output, 'adds_3d_res.bin')
        complete = x_path.exists() and o_path.exists()
        if complete:
            logging.debug("Case(%s), Golden complete.", case_name)
        else:
            x = np.random.uniform(-1, 1, shape_dim3_i).astype(dtype)
            x.tofile(x_path)
            x = x + 1.5
            x.tofile(o_path)
    elif case_name == "OnBoardTest.test_operation_scalar_dim4_add":
        x_path = Path(output, 'adds_4d_x.bin')
        o_path = Path(output, 'adds_4d_res.bin')
        complete = x_path.exists() and o_path.exists()
        if complete:
            logging.debug("Case(%s), Golden complete.", case_name)
        else:
            x = np.random.uniform(-1, 1, shape_dim4_i).astype(dtype)
            x.tofile(x_path)
            x = x + 1.5
            x.tofile(o_path)
    elif case_name == "OnBoardTest.test_operation_scalar_dim2_sub":
        x_path = Path(output, 'subs_2d_x.bin')
        o_path = Path(output, 'subs_2d_res.bin')
        complete = x_path.exists() and o_path.exists()
        if complete:
            logging.debug("Case(%s), Golden complete.", case_name)
        else:
            x = np.random.uniform(-1, 1, shape_i).astype(dtype)
            x.tofile(x_path)
            x = x - 1.5
            x.tofile(o_path)
    elif case_name == "OnBoardTest.test_operation_scalar_dim3_sub":
        x_path = Path(output, 'subs_3d_x.bin')
        o_path = Path(output, 'subs_3d_res.bin')
        complete = x_path.exists() and o_path.exists()
        if complete:
            logging.debug("Case(%s), Golden complete.", case_name)
        else:
            x = np.random.uniform(-1, 1, shape_dim3_i).astype(dtype)
            x.tofile(x_path)
            x = x - 1.5
            x.tofile(o_path)
    elif case_name == "OnBoardTest.test_operation_scalar_dim4_sub":
        x_path = Path(output, 'subs_4d_x.bin')
        o_path = Path(output, 'subs_4d_res.bin')
        complete = x_path.exists() and o_path.exists()
        if complete:
            logging.debug("Case(%s), Golden complete.", case_name)
        else:
            x = np.random.uniform(-1, 1, shape_dim4_i).astype(dtype)
            x.tofile(x_path)
            x = x - 1.5
            x.tofile(o_path)
    elif case_name == "OnBoardTest.test_operation_scalar_dim2_mul":
        x_path = Path(output, 'muls_2d_x.bin')
        o_path = Path(output, 'muls_2d_res.bin')
        complete = x_path.exists() and o_path.exists()
        if complete:
            logging.debug("Case(%s), Golden complete.", case_name)
        else:
            x = np.random.uniform(-1, 1, shape_i).astype(dtype)
            x.tofile(x_path)
            x = x * 1.5
            x.tofile(o_path)
    elif case_name == "OnBoardTest.test_operation_scalar_dim3_mul":
        x_path = Path(output, 'muls_3d_x.bin')
        o_path = Path(output, 'muls_3d_res.bin')
        complete = x_path.exists() and o_path.exists()
        if complete:
            logging.debug("Case(%s), Golden complete.", case_name)
        else:
            x = np.random.uniform(-1, 1, shape_dim3_i).astype(dtype)
            x.tofile(x_path)
            x = x * 1.5
            x.tofile(o_path)
    elif case_name == "OnBoardTest.test_operation_scalar_dim4_mul":
        x_path = Path(output, 'muls_4d_x.bin')
        o_path = Path(output, 'muls_4d_res.bin')
        complete = x_path.exists() and o_path.exists()
        if complete:
            logging.debug("Case(%s), Golden complete.", case_name)
        else:
            x = np.random.uniform(-1, 1, shape_dim4_i).astype(dtype)
            x.tofile(x_path)
            x = x * 1.5
            x.tofile(o_path)
    elif case_name == "OnBoardTest.test_operation_scalar_32_32_1_256_mul":
        x_path = Path(output, 'x.bin')
        o_path = Path(output, 'res.bin')
        complete = x_path.exists() and o_path.exists()
        if complete:
            logging.debug("Case(%s), Golden complete.", case_name)
        else:
            x = np.random.uniform(-1, 1, [32, 32, 1, 256]).astype(dtype)
            x.tofile(x_path)
            x = x * 0.07256
            x.tofile(o_path)
    elif case_name == "OnBoardTest.test_operation_scalar_dim2_div":
        x_path = Path(output, 'divs_2d_x.bin')
        o_path = Path(output, 'divs_2d_res.bin')
        complete = x_path.exists() and o_path.exists()
        if complete:
            logging.debug("Case(%s), Golden complete.", case_name)
        else:
            x = np.random.uniform(-1, 1, shape_i).astype(dtype)
            x.tofile(x_path)
            x = x / 1.5
            x.tofile(o_path)
    elif case_name == "OnBoardTest.test_operation_scalar_dim1_div":
        x_path = Path(output, 'divs_1d_x.bin')
        o_path = Path(output, 'divs_1d_res.bin')
        complete = x_path.exists() and o_path.exists()
        if complete:
            logging.debug("Case(%s), Golden complete.", case_name)
        else:
            x = np.random.uniform(-1, 1, [64]).astype(dtype)
            x.tofile(x_path)
            x = x / 1.5
            x.tofile(o_path)
    elif case_name == "OnBoardTest.test_operation_scalar_dim3_div":
        x_path = Path(output, 'divs_3d_x.bin')
        o_path = Path(output, 'divs_3d_res.bin')
        complete = x_path.exists() and o_path.exists()
        if complete:
            logging.debug("Case(%s), Golden complete.", case_name)
        else:
            x = np.random.uniform(-1, 1, shape_dim3_i).astype(dtype)
            x.tofile(x_path)
            x = x / 1.5
            x.tofile(o_path)
    elif case_name == "OnBoardTest.test_operation_scalar_dim1_div":
        x_path = Path(output, 'divs_1d_x.bin')
        o_path = Path(output, 'divs_1d_res.bin')
        complete = x_path.exists() and o_path.exists()
        if complete:
            logging.debug("Case(%s), Golden complete.", case_name)
        else:
            x = np.random.uniform(-1, 1, [64]).astype(dtype)
            x.tofile(x_path)
            x = x / 1.5
            x.tofile(o_path)
    elif case_name == "OnBoardTest.test_operation_scalar_dim4_div":
        x_path = Path(output, 'divs_4d_x.bin')
        o_path = Path(output, 'divs_4d_res.bin')
        complete = x_path.exists() and o_path.exists()
        if complete:
            logging.debug("Case(%s), Golden complete.", case_name)
        else:
            x = np.random.uniform(-1, 1, shape_dim4_i).astype(dtype)
            x.tofile(x_path)
            x = x / 1.5
            x.tofile(o_path)
    elif case_name == "OnBoardTest.test_operation_tensor_2_2_8_8_expand_add":
        x_path = Path(output, 'x.bin')
        y_path = Path(output, 'y.bin')
        o_path = Path(output, 'res.bin')
        complete = x_path.exists() and y_path.exists() and o_path.exists()
        if complete:
            logging.debug("Case(%s), Golden complete.", case_name)
        else:
            x = np.random.uniform(-1, 1, shape_2_2_8_8_i).astype(dtype)
            x.tofile(x_path)
            y = np.random.uniform(-1, 1, shape_2_1_8_8_i).astype(dtype)
            y.tofile(y_path)
            x = x + y
            x.tofile(o_path)
    elif case_name == "OnBoardTest.test_operation_tensor_4_4_16_16_expand_add":
        x_path = Path(output, 'x.bin')
        y_path = Path(output, 'y.bin')
        o_path = Path(output, 'res.bin')
        complete = x_path.exists() and y_path.exists() and o_path.exists()
        if complete:
            logging.debug("Case(%s), Golden complete.", case_name)
        else:
            x = np.random.uniform(-1, 1, shape_4_4_16_16_i).astype(dtype)
            x.tofile(x_path)
            y = np.random.uniform(-1, 1, shape_4_1_16_16_i).astype(dtype)
            y.tofile(y_path)
            x = x + y
            x.tofile(o_path)
    elif case_name == "OnBoardTest.test_operation_tensor_1_1_32_to_16_32_32_expand_add":
        x_path = Path(output, 'x.bin')
        y_path = Path(output, 'y.bin')
        o_path = Path(output, 'res.bin')
        complete = x_path.exists() and y_path.exists() and o_path.exists()
        if complete:
            logging.debug("Case(%s), Golden complete.", case_name)
        else:
            x = np.random.uniform(-1, 1, shape_16_32_32_i).astype(dtype)
            x.tofile(x_path)
            y = np.random.uniform(-1, 1, shape_1_1_32_i).astype(dtype)
            y.tofile(y_path)
            x = x + y
            x.tofile(o_path)
    elif case_name == "OnBoardTest.test_operation_tensor_8_16_1_to_8_16_16_expand_add":
        x_path = Path(output, 'x.bin')
        y_path = Path(output, 'y.bin')
        o_path = Path(output, 'res.bin')
        complete = x_path.exists() and y_path.exists() and o_path.exists()
        if complete:
            logging.debug("Case(%s), Golden complete.", case_name)
        else:
            x = np.random.uniform(-1, 1, shape_8_16_16_i).astype(dtype)
            x.tofile(x_path)
            y = np.random.uniform(-1, 1, shape_8_16_1_i).astype(dtype)
            y.tofile(y_path)
            x = x + y
            x.tofile(o_path)
    elif case_name == "OnBoardTest.test_operation_tensor_8_8_1_to_8_8_7168_expand_Mul_moe":
        x_path = Path(output, 'x.bin')
        y_path = Path(output, 'y.bin')
        o_path = Path(output, 'res.bin')
        complete = x_path.exists() and y_path.exists() and o_path.exists()
        if complete:
            logging.debug("Case(%s), Golden complete.", case_name)
        else:
            x = np.random.uniform(-1, 1, shape_8_8_7168_i).astype(dtype)
            x.tofile(x_path)
            y = np.random.uniform(-1, 1, shape_8_8_1_i).astype(dtype)
            y.tofile(y_path)
            x = x * y
            x.tofile(o_path)
    elif case_name == "OnBoardTest.test_operation_tensor_8_8_1_to_8_8_7168_expand_sub":
        x_path = Path(output, 'x.bin')
        y_path = Path(output, 'y.bin')
        o_path = Path(output, 'res.bin')
        complete = x_path.exists() and y_path.exists() and o_path.exists()
        if complete:
            logging.debug("Case(%s), Golden complete.", case_name)
        else:
            x = np.random.uniform(-1, 1, shape_8_8_1_i).astype(dtype)
            x.tofile(x_path)
            y = np.random.uniform(-1, 1, shape_8_8_7168_i).astype(dtype)
            y.tofile(y_path)
            x = x - y
            x.tofile(o_path)
    elif case_name == "OnBoardTest.test_operation_tensor_8_1_16_to_8_16_16_expand_add":
        x_path = Path(output, 'x.bin')
        y_path = Path(output, 'y.bin')
        o_path = Path(output, 'res.bin')
        complete = x_path.exists() and y_path.exists() and o_path.exists()
        if complete:
            logging.debug("Case(%s), Golden complete.", case_name)
        else:
            x = np.random.uniform(-1, 1, shape_8_16_16_i).astype(dtype)
            x.tofile(x_path)
            y = np.random.uniform(-1, 1, shape_8_1_16_i).astype(dtype)
            y.tofile(y_path)
            x = x + y
            x.tofile(o_path)
    elif case_name == "OnBoardTest.test_operation_tensor_1_16_16_to_8_16_16_expand_add":
        x_path = Path(output, 'x.bin')
        y_path = Path(output, 'y.bin')
        o_path = Path(output, 'res.bin')
        complete = x_path.exists() and y_path.exists() and o_path.exists()
        if complete:
            logging.debug("Case(%s), Golden complete.", case_name)
        else:
            x = np.random.uniform(-1, 1, shape_8_16_16_i).astype(dtype)
            x.tofile(x_path)
            y = np.random.uniform(-1, 1, shape_1_16_16_i).astype(dtype)
            y.tofile(y_path)
            x = x + y
            x.tofile(o_path)
    elif case_name == "OnBoardTest.test_operation_tensor_8_1_1_to_8_16_16_expand_add":
        x_path = Path(output, 'x.bin')
        y_path = Path(output, 'y.bin')
        o_path = Path(output, 'res.bin')
        complete = x_path.exists() and y_path.exists() and o_path.exists()
        if complete:
            logging.debug("Case(%s), Golden complete.", case_name)
        else:
            x = np.random.uniform(-1, 1, shape_8_16_16_i).astype(dtype)
            x.tofile(x_path)
            y = np.random.uniform(-1, 1, shape_8_1_1_i).astype(dtype)
            y.tofile(y_path)
            x = x + y
            x.tofile(o_path)
    elif case_name == "OnBoardTest.test_operation_tensor_1_1_1_to_8_16_16_expand_add":
        x_path = Path(output, 'x.bin')
        y_path = Path(output, 'y.bin')
        o_path = Path(output, 'res.bin')
        complete = x_path.exists() and y_path.exists() and o_path.exists()
        if complete:
            logging.debug("Case(%s), Golden complete.", case_name)
        else:
            x = np.random.uniform(-1, 1, shape_8_16_16_i).astype(dtype)
            x.tofile(x_path)
            y = np.random.uniform(-1, 1, shape_1_1_1_i).astype(dtype)
            y.tofile(y_path)
            x = x + y
            x.tofile(o_path)
    elif case_name == "OnBoardTest.test_operation_tensor_16_32_32_to_16_32_1_tileop_mul":
        x_path = Path(output, 'x.bin')
        y_path = Path(output, 'y.bin')
        o_path = Path(output, 'res.bin')
        complete = x_path.exists() and y_path.exists() and o_path.exists()
        if complete:
            logging.debug("Case(%s), Golden complete.", case_name)
        else:
            x = np.random.uniform(-1, 1, [16, 32, 32]).astype(dtype)
            x.tofile(x_path)
            y = np.random.uniform(-1, 1, [16, 32, 1]).astype(dtype)
            y.tofile(y_path)
            x = x * y
            x.tofile(o_path)
    elif case_name == "OnBoardTest.test_operation_tensor_8_8_1_1_to_8_8_1_256_tileop_sub":
        x_path = Path(output, 'x.bin')
        y_path = Path(output, 'y.bin')
        o_path = Path(output, 'res.bin')
        complete = x_path.exists() and y_path.exists() and o_path.exists()
        if complete:
            logging.debug("Case(%s), Golden complete.", case_name)
        else:
            x = np.random.uniform(-1, 1, [8, 8, 1, 256]).astype(dtype)
            x.tofile(x_path)
            y = np.random.uniform(-1, 1, [8, 8, 1, 1]).astype(dtype)
            y.tofile(y_path)
            x = x - y
            x.tofile(o_path)
    elif case_name == "OnBoardTest.test_operation_tensor_32_32_1_1_to_32_32_1_256_tileop_add":
        x_path = Path(output, 'x.bin')
        y_path = Path(output, 'y.bin')
        o_path = Path(output, 'res.bin')
        complete = x_path.exists() and y_path.exists() and o_path.exists()
        if complete:
            logging.debug("Case(%s), Golden complete.", case_name)
        else:
            x = np.random.uniform(-1, 1, [32, 32, 1, 256]).astype(dtype)
            x.tofile(x_path)
            y = np.random.uniform(-1, 1, [32, 32, 1, 1]).astype(dtype)
            y.tofile(y_path)
            x = x + y
            x.tofile(o_path)
    elif case_name == "OnBoardTest.test_operation_tensor_32_32_1_1_to_32_32_1_256_tileop_sub":
        x_path = Path(output, 'x.bin')
        y_path = Path(output, 'y.bin')
        o_path = Path(output, 'res.bin')
        complete = x_path.exists() and y_path.exists() and o_path.exists()
        if complete:
            logging.debug("Case(%s), Golden complete.", case_name)
        else:
            x = np.random.uniform(-1, 1, [32, 32, 1, 256]).astype(dtype)
            x.tofile(x_path)
            y = np.random.uniform(-1, 1, [32, 32, 1, 1]).astype(dtype)
            y.tofile(y_path)
            x = x - y
            x.tofile(o_path)
    elif case_name == "OnBoardTest.test_operation_tensor_32_32_1_1_to_32_32_1_256_tileop_mul":
        x_path = Path(output, 'x.bin')
        y_path = Path(output, 'y.bin')
        o_path = Path(output, 'res.bin')
        complete = x_path.exists() and y_path.exists() and o_path.exists()
        if complete:
            logging.debug("Case(%s), Golden complete.", case_name)
        else:
            x = np.random.uniform(-1, 1, [32, 32, 1, 256]).astype(dtype)
            x.tofile(x_path)
            y = np.random.uniform(-1, 1, [32, 32, 1, 1]).astype(dtype)
            y.tofile(y_path)
            x = x * y
            x.tofile(o_path)
    elif case_name == "OnBoardTest.test_operation_tensor_32_32_1_1_to_32_32_1_256_tileop_div":
        x_path = Path(output, 'x.bin')
        y_path = Path(output, 'y.bin')
        o_path = Path(output, 'res.bin')
        complete = x_path.exists() and y_path.exists() and o_path.exists()
        if complete:
            logging.debug("Case(%s), Golden complete.", case_name)
        else:
            x = np.random.uniform(-1, 1, [32, 32, 1, 256]).astype(dtype)
            x.tofile(x_path)
            y = np.random.uniform(-1, 1, [32, 32, 1, 1]).astype(dtype)
            y.tofile(y_path)
            x = x / y
            x.tofile(o_path)
    elif case_name == "OnBoardTest.test_operation_tensor_1_n_to_m_n_mul":
        x_path = Path(output, 'x.bin')
        y_path = Path(output, 'y.bin')
        o_path = Path(output, 'res.bin')
        complete = x_path.exists() and y_path.exists() and o_path.exists()
        if complete:
            logging.debug("Case(%s), Golden complete.", case_name)
        else:
            x = np.random.uniform(-1, 1, [64, 32]).astype(dtype)
            x.tofile(x_path)
            y = np.random.uniform(-1, 1, [1, 32]).astype(dtype)
            y.tofile(y_path)
            x = x * y
            x.tofile(o_path)
    elif case_name == "OnBoardTest.test_operation_tensor_1_1_1_64_to_1_128_1_64_tileop_mul01":
        x_path = Path(output, 'x.bin')
        y_path = Path(output, 'y.bin')
        o_path = Path(output, 'res.bin')
        complete = x_path.exists() and y_path.exists() and o_path.exists()
        if complete:
            logging.debug("Case(%s), Golden complete.", case_name)
        else:
            x = np.random.uniform(-1, 1, [1, 1, 1, 64]).astype(dtype)
            x.tofile(x_path)
            y = np.random.uniform(-1, 1, [1, 128, 1, 64]).astype(dtype)
            y.tofile(y_path)
            x = x * y
            x.tofile(o_path)
    elif case_name == "OnBoardTest.test_operation_tensor_1_1_1_64_to_1_128_1_64_tileop_mul02":
        x_path = Path(output, 'x.bin')
        y_path = Path(output, 'y.bin')
        o_path = Path(output, 'res.bin')
        complete = x_path.exists() and y_path.exists() and o_path.exists()
        if complete:
            logging.debug("Case(%s), Golden complete.", case_name)
        else:
            x = np.random.uniform(-1, 1, [1, 128, 1, 64]).astype(dtype)
            x.tofile(x_path)
            y = np.random.uniform(-1, 1, [1, 1, 1, 64]).astype(dtype)
            y.tofile(y_path)
            x = x * y
            x.tofile(o_path)
    elif case_name == "OnBoardTest.test_operation_tensor_1_1_64_to_32_1_64_tileop_mul03":
        x_path = Path(output, 'x.bin')
        y_path = Path(output, 'y.bin')
        o_path = Path(output, 'res.bin')
        complete = x_path.exists() and y_path.exists() and o_path.exists()
        if complete:
            logging.debug("Case(%s), Golden complete.", case_name)
        else:
            x = np.random.uniform(-1, 1, [32, 1, 64]).astype(dtype)
            x.tofile(x_path)
            y = np.random.uniform(-1, 1, [1, 1, 64]).astype(dtype)
            y.tofile(y_path)
            x = x * y
            x.tofile(o_path)
    elif case_name == "OnBoardTest.test_operation_tensor_8_8_1_to_8_8_7168_expand_mul":
        x_path = Path(output, 'x.bin')
        y_path = Path(output, 'y.bin')
        o_path = Path(output, 'res.bin')
        ccc = 7168
        complete = x_path.exists() and y_path.exists() and o_path.exists()
        complete = False
        if complete:
            logging.debug("Case(%s), Golden complete.", case_name)
        else:
            x = np.random.uniform(-1, 1, [8, 8, ccc]).astype(dtype)
            x.tofile(x_path)
            y = np.random.uniform(-1, 1, [8, 8, 1]).astype(dtype)
            y.tofile(y_path)
            x = x * y
            x.tofile(o_path)
    elif case_name == "OnBoardTest.test_operation_tensor_8_8_1_to_8_8_512_expand_mul":
        x_path = Path(output, 'x.bin')
        y_path = Path(output, 'y.bin')
        o_path = Path(output, 'res.bin')
        ccc = 512
        complete = x_path.exists() and y_path.exists() and o_path.exists()
        complete = False
        if complete:
            logging.debug("Case(%s), Golden complete.", case_name)
        else:
            x = np.random.uniform(-1, 1, [8, 8, ccc]).astype(dtype)
            x.tofile(x_path)
            y = np.random.uniform(-1, 1, [8, 8, 1]).astype(dtype)
            y.tofile(y_path)
            x = x * y
            x.tofile(o_path)
    elif case_name == "OnBoardTest.test_operation_tensor_8_8_512_to_8_8_512_expand_mul":
        x_path = Path(output, 'x.bin')
        y_path = Path(output, 'y.bin')
        o_path = Path(output, 'res.bin')
        ccc = 512
        complete = x_path.exists() and y_path.exists() and o_path.exists()
        complete = False
        if complete:
            logging.debug("Case(%s), Golden complete.", case_name)
        else:
            x = np.random.uniform(-1, 1, [8, 8, ccc]).astype(dtype)
            x.tofile(x_path)
            y = np.random.uniform(-1, 1, [8, 8, ccc]).astype(dtype)
            y.tofile(y_path)
            x = x * y
            x.tofile(o_path)
    elif case_name == "OnBoardTest.test_operation_tensor_16_16_64_65_tileop_add_unalign":
        shape_16_16_64_65_i = [16, 16, 64, 65]
        x_path = Path(output, 'x.bin')
        y_path = Path(output, 'y.bin')
        o_path = Path(output, 'res.bin')
        complete = x_path.exists() and y_path.exists() and o_path.exists()
        if complete:
            logging.debug("Case(%s), Golden complete.", case_name)
        else:
            x = np.random.uniform(-1, 1, shape_16_16_64_65_i).astype(dtype)
            x.tofile(x_path)
            y = np.random.uniform(-1, 1, shape_16_16_64_65_i).astype(dtype)
            y.tofile(y_path)
            x = x + y
            x.tofile(o_path)
    elif case_name == "OnBoardTest.test_operation_tensor_16_16_39_65_tileop_add_unalign":
        shape_16_16_39_65_i = [16, 16, 39, 65]
        x_path = Path(output, 'x.bin')
        y_path = Path(output, 'y.bin')
        o_path = Path(output, 'res.bin')
        complete = x_path.exists() and y_path.exists() and o_path.exists()
        if complete:
            logging.debug("Case(%s), Golden complete.", case_name)
        else:
            x = np.random.uniform(-1, 1, shape_16_16_39_65_i).astype(dtype)
            x.tofile(x_path)
            y = np.random.uniform(-1, 1, shape_16_16_39_65_i).astype(dtype)
            y.tofile(y_path)
            x = x + y
            x.tofile(o_path)
    elif case_name == "OnBoardTest.test_operation_tensor_32_1_tileop_add_unalign":
        shape_32_1_i = [32, 1]
        x_path = Path(output, 'x.bin')
        y_path = Path(output, 'y.bin')
        o_path = Path(output, 'res.bin')
        complete = x_path.exists() and y_path.exists() and o_path.exists()
        if complete:
            logging.debug("Case(%s), Golden complete.", case_name)
        else:
            x = np.random.uniform(-1, 1, shape_32_1_i).astype(dtype)
            x.tofile(x_path)
            y = np.random.uniform(-1, 1, shape_32_1_i).astype(dtype)
            y.tofile(y_path)
            x = x + y
            x.tofile(o_path)
    elif case_name == "OnBoardTest.test_operation_tensor_32_1_tileop_sub_unalign":
        shape_32_1_i = [32, 1]
        x_path = Path(output, 'x.bin')
        y_path = Path(output, 'y.bin')
        o_path = Path(output, 'res.bin')
        complete = x_path.exists() and y_path.exists() and o_path.exists()
        if complete:
            logging.debug("Case(%s), Golden complete.", case_name)
        else:
            x = np.random.uniform(-1, 1, shape_32_1_i).astype(dtype)
            x.tofile(x_path)
            y = np.random.uniform(-1, 1, shape_32_1_i).astype(dtype)
            y.tofile(y_path)
            x = x - y
            x.tofile(o_path)
    elif case_name == "OnBoardTest.test_operation_tensor_32_1_tileop_mul_unalign":
        shape_32_1_i = [32, 1]
        x_path = Path(output, 'x.bin')
        y_path = Path(output, 'y.bin')
        o_path = Path(output, 'res.bin')
        complete = x_path.exists() and y_path.exists() and o_path.exists()
        if complete:
            logging.debug("Case(%s), Golden complete.", case_name)
        else:
            x = np.random.uniform(-1, 1, shape_32_1_i).astype(dtype)
            x.tofile(x_path)
            y = np.random.uniform(-1, 1, shape_32_1_i).astype(dtype)
            y.tofile(y_path)
            x = x * y
            x.tofile(o_path)
    elif case_name == "OnBoardTest.test_operation_add_vs_dim2_unalign":
        shape_test = [79, 85]
        x_path = Path(output, 'x.bin')
        o_path = Path(output, 'res.bin')
        complete = x_path.exists() and o_path.exists()
        if complete:
            logging.debug("Case(%s), Golden complete.", case_name)
        else:
            x = np.random.uniform(-1, 1, shape_test).astype(dtype)
            x.tofile(x_path)
            x = x + 1.5
            x.tofile(o_path)
    elif case_name == "OnBoardTest.test_operation_mul_vs_dim3_unalign":
        shape_test = [2, 79, 85]
        x_path = Path(output, 'x.bin')
        o_path = Path(output, 'res.bin')
        complete = x_path.exists() and o_path.exists()
        if complete:
            logging.debug("Case(%s), Golden complete.", case_name)
        else:
            x = np.random.uniform(-1, 1, shape_test).astype(dtype)
            x.tofile(x_path)
            x = x * 1.5
            x.tofile(o_path)
    elif case_name == "OnBoardTest.test_operation_sub_vs_dim4_unalign":
        shape_test = [2, 2, 67, 125]
        x_path = Path(output, 'x.bin')
        o_path = Path(output, 'res.bin')
        complete = x_path.exists() and o_path.exists()
        if complete:
            logging.debug("Case(%s), Golden complete.", case_name)
        else:
            x = np.random.uniform(-1, 1, shape_test).astype(dtype)
            x.tofile(x_path)
            x = x - 1.5
            x.tofile(o_path)
    elif case_name == "OnBoardTest.test_operation_div_vs_dim1_unalign":
        shape_test = [125]
        x_path = Path(output, 'x.bin')
        o_path = Path(output, 'res.bin')
        complete = x_path.exists() and o_path.exists()
        if complete:
            logging.debug("Case(%s), Golden complete.", case_name)
        else:
            x = np.random.uniform(-1, 1, shape_test).astype(dtype)
            x.tofile(x_path)
            x = x / 1.5
            x.tofile(o_path)
    elif case_name == "OnBoardTest.test_mul_large_row":
        shape_32_1_i = [1, 16384]
        x_path = Path(output, 'x.bin')
        y_path = Path(output, 'y.bin')
        o_path = Path(output, 'res.bin')
        complete = x_path.exists() and y_path.exists() and o_path.exists()
        if complete:
            logging.debug("Case(%s), Golden complete.", case_name)
        else:
            x = np.random.uniform(-1, 1, shape_32_1_i).astype(dtype)
            x.tofile(x_path)
            y = np.random.uniform(-1, 1, shape_32_1_i).astype(dtype)
            y.tofile(y_path)
            x = x * y
            x.tofile(o_path)
    elif case_name == "OnBoardTest.test_matmul_add_dynamic":
        fp16 = np.float16
        m = 128
        k = 256
        n = 512
        matmul_inshape1 = [m, k]
        matmul_inshape2 = [k, n]
        add_inshape = [m, n]
        matmul1_path = Path(output, 'matmulx.bin')
        matmul2_path = Path(output, 'matmuly.bin')
        add1_path = Path(output, 'add1.bin')
        add2_path = Path(output, 'add2.bin')
        o_path = Path(output, 'res.bin')
        complete = matmul1_path.exists() and matmul2_path.exists() and add1_path.exists() and add2_path.exists() and o_path.exists()
        if complete:
            logging.debug("Case(%s), Golden complete.", case_name)
        else:
            x = np.random.uniform(-1, 1, matmul_inshape1).astype(fp16)
            x.tofile(matmul1_path)
            y = np.random.uniform(-1, 1, matmul_inshape2).astype(fp16)
            y.tofile(matmul2_path)
            a1 = np.random.uniform(-1, 1, add_inshape).astype(fp16)
            a1.tofile(add1_path)
            a2 = np.random.uniform(-1, 1, add_inshape).astype(fp16)
            a2.tofile(add2_path)
            z = np.dot(x, y) + a1 + a2
            z.tofile(o_path)
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
        "OnBoardTest.test_operation_tensor_dim2_add",
    ]
    # 函数调用
    ret: bool = True
    for cs in case_name_list:
        output: Path = Path(g_src_root, "build/output/bin/golden", cs).resolve()
        output.mkdir(parents=True, exist_ok=True)
        ret = binary_operator_func1(case_name=cs, output=output)
    return ret


if __name__ == "__main__":
    exit(0 if main() else 1)
