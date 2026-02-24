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
""" UnaryOperator 相关用例 Golden 生成逻辑.

本脚本有 2 种执行模式:
1. CI批跑时, 由 cmake/scripts/golden_ctrl.py 调用, 为避免日志过多, 此时 logging 级别为 logging.INFO;
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
    g_ctrl_path: Path = Path(g_src_root, "cmake/scripts")
    if str(g_ctrl_path) not in sys.path:
        sys.path.append(str(g_ctrl_path))
    from golden_register import GoldenRegister  # 单独调试 import 失败, 需确认上文中 '系统 import 路径' 配置正确
else:
    from golden_register import GoldenRegister


def gen_sin(x):
    return np.sin(x)


def gen_cos(x):
    return np.cos(x)


def gen_logical_not(x, dtype):
    return (x == 0).astype(dtype)


def gen_abs(x):
    return np.abs(x)


@GoldenRegister.reg_golden_func(
    case_names=[
        "AbsOnBoardTest.test_abs_8_4608",
        "AbsOnBoardTest.test_abs_8_4609",
        "AbsOnBoardTest.test_abs_1_16384",
    ]
)
def unary_operator_func_abs(case_name: str, output: Path) -> bool:
    dtype_fp16 = np.float16
    row = 8
    col = 4608
    shape_dim2 = [row, col]
    if case_name == "AbsOnBoardTest.test_abs_8_4608":
        x_path = Path(output, 'abs_x.bin')
        y_path = Path(output, 'abs_golden.bin')
        complete = x_path.exists() and y_path.exists()
        if complete:
            logging.debug("Case(%s), Golden complete.", case_name)
            return True
        else:
            x = np.random.uniform(-1, 1, shape_dim2).astype(dtype_fp16)
            x.tofile(x_path)
            y = gen_abs(x)
            y.tofile(y_path)
            return True
    elif case_name == "AbsOnBoardTest.test_abs_8_4609":
        x_path = Path(output, 'abs_x_not_align.bin')
        y_path = Path(output, 'abs_golden_not_align.bin')
        complete = x_path.exists() and y_path.exists()
        row_not_align = 8
        col_not_align = 4609
        shape_dim2_not_align = [row_not_align, col_not_align]
        if complete:
            logging.debug("Case(%s), Golden complete.", case_name)
            return True
        else:
            x = np.random.uniform(-1, 1, shape_dim2_not_align).astype(dtype_fp16)
            x.tofile(x_path)
            y = gen_abs(x)
            y.tofile(y_path)
            return True
    elif case_name == "AbsOnBoardTest.test_abs_1_16384":
        x_path = Path(output, 'abs_x_not_align.bin')
        y_path = Path(output, 'abs_golden_not_align.bin')
        complete = x_path.exists() and y_path.exists()
        row_not_align = 1
        col_not_align = 16384
        shape_dim2_not_align = [row_not_align, col_not_align]
        if complete:
            logging.debug("Case(%s), Golden complete.", case_name)
            return True
        else:
            x = np.random.uniform(-1, 1, shape_dim2_not_align).astype(dtype_fp16)
            x.tofile(x_path)
            y = gen_abs(x)
            y.tofile(y_path)
            return True
    else:
        logging.error("Can't get func to gen golden, Case(%s)", case_name)
        return False


@GoldenRegister.reg_golden_func(
    case_names=[
        "LogicalNotOnBoardTest.test_logicalnot_16_32_fp32",
        "LogicalNotOnBoardTest.test_logicalnot_16_32_32_fp16",
    ]
)
def unary_operator_func_logicalnot(case_name: str, output: Path) -> bool:
    dtype_fp16 = np.float16
    dtype_fp32 = np.float32
    dim0 = 16
    dim1 = 32
    dim2 = 32
    shape_dim2 = [dim0, dim1]
    shape_dim3 = [dim0, dim1, dim2]
    if case_name == "LogicalNotOnBoardTest.test_logicalnot_16_32_fp32":
        x_path = Path(output, 'logicalnotdim2_x.bin')
        y_path = Path(output, 'logicalnotdim2_golden.bin')
        complete = x_path.exists() and y_path.exists()
        if complete:
            logging.debug("Case(%s), Golden complete.", case_name)
        else:
            x = np.random.choice([0, 1], size=shape_dim2, p=[0.5, 0.5]).astype(dtype_fp32)
            x.tofile(x_path)
            y = gen_logical_not(x, dtype_fp32)
            y.tofile(y_path)

    elif case_name == "LogicalNotOnBoardTest.test_logicalnot_16_32_32_fp16":
        x_path = Path(output, 'logicalnotdim3_x.bin')
        y_path = Path(output, 'logicalnotdim3_golden.bin')
        complete = x_path.exists() and y_path.exists()
        if complete:
            logging.debug("Case(%s), Golden complete.", case_name)
        else:
            x = np.random.choice([0, 1], size=shape_dim3, p=[0.5, 0.5]).astype(dtype_fp16)
            x.tofile(x_path)
            y = gen_logical_not(x, dtype_fp16)
            y.tofile(y_path)

    else:
        logging.error("Can't get func to gen golden, Case(%s)", case_name)
        return False
    return True


@GoldenRegister.reg_golden_func(
    case_names=[
        "OnBoardTest.test_sin_dim2_float32",
        "OnBoardTest.test_cos_dim4_float16",
    ]
)
def unary_operator_func_sin_cos(case_name: str, output: Path) -> bool:
    in0 = 2
    in1 = 2
    row = 64
    col = 64

    shape_dim2 = [row, col]
    shape_dim4 = [in0, in1, row, col]
    dtype_fp16 = np.float16
    dtype_fp32 = np.float32

    if case_name == "OnBoardTest.test_sin_dim2_float32":
        x_path = Path(output, 'x_dim2_fp32.bin')
        y_path = Path(output, 'sin_golden_fp32.bin')
        complete = x_path.exists() and y_path.exists()
        if complete:
            logging.debug("Case(%s), Golden complete.", case_name)
            return True
        else:
            x = np.random.uniform(-1, 1, shape_dim2).astype(dtype_fp32)
            x.tofile(x_path)
            y = gen_sin(x)
            y.tofile(y_path)
            return True
    elif case_name == "OnBoardTest.test_cos_dim4_float16":
        x_path = Path(output, 'x_dim_4_fp16.bin')
        y_path = Path(output, 'cos_golden_fp16.bin')
        complete = x_path.exists() and y_path.exists()
        if complete:
            logging.debug("Case(%s), Golden complete.", case_name)
            return True
        else:
            x = np.random.uniform(-1, 1, shape_dim4).astype(dtype_fp16)
            x.tofile(x_path)
            y = gen_cos(x)
            y.tofile(y_path)
            return True
    else:
        logging.error("Can't get func to gen golden, Case(%s)", case_name)
        return False


@GoldenRegister.reg_golden_func(
    case_names=[
        "OnBoardTest.test_unary_operation_32_32_tileop_exp",
        "OnBoardTest.test_unary_operation_16_32_32_tileop_exp",
        "OnBoardTest.test_unary_operation_16_16_64_64_tileop_exp",
        "OnBoardTest.test_unary_operation_32_32_tileop_sqrt",
        "OnBoardTest.test_unary_operation_16_32_32_tileop_sqrt",
        "OnBoardTest.test_unary_operation_16_16_64_64_tileop_sqrt",
        "OnBoardTest.test_unary_operation_16_16_64_70_tileop_sqrt",
        "OnBoardTest.test_unary_operation_16_16_64_64_tileop_sign",
        "OnBoardTest.test_unary_operation_32_32_tileop_reciprocal",
        "OnBoardTest.test_unary_operation_16_32_32_tileop_reciprocal",
        "OnBoardTest.test_unary_operation_16_16_64_64_tileop_reciprocal",
        "OnBoardTest.test_unary_operation_16_16_64_64_tileop_relu",
    ]
)
def unary_operator_gen_data(case_name: str, output: Path) -> bool:
    shape_16_16_64_64_i = [16, 16, 64, 64]
    shape_16_32_32_i = [16, 32, 32]
    shape_64_128_i = [32, 32]

    dtype = np.float32

    if case_name == "OnBoardTest.test_unary_operation_32_32_tileop_exp":
        x_path = Path(output, 'x.bin')
        o_path = Path(output, 'res.bin')
        complete = x_path.exists() and o_path.exists()
        if complete:
            logging.debug("Case(%s), Golden complete.", case_name)
        else:
            x = np.random.uniform(-1, 1, shape_64_128_i).astype(dtype)
            x.tofile(x_path)
            x = np.exp(x)
            x.tofile(o_path)
    elif case_name == "OnBoardTest.test_unary_operation_16_32_32_tileop_exp":
        x_path = Path(output, 'x.bin')
        o_path = Path(output, 'res.bin')
        complete = x_path.exists() and o_path.exists()
        if complete:
            logging.debug("Case(%s), Golden complete.", case_name)
        else:
            x = np.random.uniform(-1, 1, shape_16_32_32_i).astype(dtype)
            x.tofile(x_path)
            x = np.exp(x)
            x.tofile(o_path)
    elif case_name == "OnBoardTest.test_unary_operation_16_16_64_64_tileop_exp":
        x_path = Path(output, 'x.bin')
        o_path = Path(output, 'res.bin')
        complete = x_path.exists() and o_path.exists()
        if complete:
            logging.debug("Case(%s), Golden complete.", case_name)
        else:
            x = np.random.uniform(-1, 1, shape_16_16_64_64_i).astype(dtype)
            x.tofile(x_path)
            x = np.exp(x)
            x.tofile(o_path)
    elif case_name == "OnBoardTest.test_unary_operation_32_32_tileop_sqrt":
        x_path = Path(output, 'x.bin')
        o_path = Path(output, 'res.bin')
        complete = x_path.exists() and o_path.exists()
        if complete:
            logging.debug("Case(%s), Golden complete.", case_name)
        else:
            x = np.random.uniform(0, 1, shape_64_128_i).astype(dtype)
            x.tofile(x_path)
            x = np.sqrt(x)
            x.tofile(o_path)
    elif case_name == "OnBoardTest.test_unary_operation_16_32_32_tileop_sqrt":
        x_path = Path(output, 'x.bin')
        o_path = Path(output, 'res.bin')
        complete = x_path.exists() and o_path.exists()
        if complete:
            logging.debug("Case(%s), Golden complete.", case_name)
        else:
            x = np.random.uniform(0, 1, shape_16_32_32_i).astype(dtype)
            x.tofile(x_path)
            x = np.sqrt(x)
            x.tofile(o_path)
    elif case_name == "OnBoardTest.test_unary_operation_16_16_64_64_tileop_sqrt":
        x_path = Path(output, 'x.bin')
        o_path = Path(output, 'res.bin')
        complete = x_path.exists() and o_path.exists()
        if complete:
            logging.debug("Case(%s), Golden complete.", case_name)
        else:
            x = np.random.uniform(0, 1, shape_16_16_64_64_i).astype(dtype)
            x.tofile(x_path)
            x = np.sqrt(x)
            x.tofile(o_path)
    elif case_name == "OnBoardTest.test_unary_operation_16_16_64_70_tileop_sqrt":
        x_path = Path(output, 'x.bin')
        o_path = Path(output, 'res.bin')
        complete = x_path.exists() and o_path.exists()
        shape_16_16_64_70_i = [16, 16, 64, 70]
        if complete:
            logging.debug("Case(%s), Golden complete.", case_name)
        else:
            x = np.random.uniform(0, 1, shape_16_16_64_70_i).astype(dtype)
            x.tofile(x_path)
            x = np.sqrt(x)
            x.tofile(o_path)
    elif case_name == "OnBoardTest.test_unary_operation_16_16_64_64_tileop_sign":
        x_path = Path(output, 'x.bin')
        o_path = Path(output, 'res.bin')
        complete = x_path.exists() and o_path.exists()
        if complete:
            logging.debug("Case(%s), Golden complete.", case_name)
        else:
            x = np.random.uniform(-1, 1, shape_16_16_64_64_i).astype(dtype)
            x.tofile(x_path)
            x = np.sign(x)
            x.tofile(o_path)
    elif case_name == "OnBoardTest.test_unary_operation_32_32_tileop_reciprocal":
        x_path = Path(output, 'x.bin')
        o_path = Path(output, 'res.bin')
        complete = x_path.exists() and o_path.exists()
        if complete:
            logging.debug("Case(%s), Golden complete.", case_name)
        else:
            x = np.random.uniform(-1, 1, shape_64_128_i).astype(dtype)
            x.tofile(x_path)
            x = np.reciprocal(x)
            x.tofile(o_path)
    elif case_name == "OnBoardTest.test_unary_operation_16_32_32_tileop_reciprocal":
        x_path = Path(output, 'x.bin')
        o_path = Path(output, 'res.bin')
        complete = x_path.exists() and o_path.exists()
        if complete:
            logging.debug("Case(%s), Golden complete.", case_name)
        else:
            x = np.random.uniform(-1, 1, shape_16_32_32_i).astype(dtype)
            x.tofile(x_path)
            x = np.reciprocal(x)
            x.tofile(o_path)
    elif case_name == "OnBoardTest.test_unary_operation_16_16_64_64_tileop_reciprocal":
        x_path = Path(output, 'x.bin')
        o_path = Path(output, 'res.bin')
        complete = x_path.exists() and o_path.exists()
        if complete:
            logging.debug("Case(%s), Golden complete.", case_name)
        else:
            x = np.random.uniform(-1, 1, shape_16_16_64_64_i).astype(dtype)
            x.tofile(x_path)
            x = np.reciprocal(x)
            x.tofile(o_path)
    elif case_name == "OnBoardTest.test_unary_operation_16_16_64_64_tileop_relu":
        x_path = Path(output, 'x.bin')
        o_path = Path(output, 'res.bin')
        complete = x_path.exists() and o_path.exists()
        if complete:
            logging.debug("Case(%s), Golden complete.", case_name)
        else:
            x = np.random.uniform(-1, 1, shape_16_16_64_64_i).astype(dtype)
            x.tofile(x_path)
            x = np.relu(x)
            x.tofile(o_path)
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
        "OnBoardTest.test_unary_operation_32_32_tileop_exp",
    ]
    # 函数调用
    ret: bool = True
    for cs in case_name_list:
        output: Path = Path(g_src_root, "build/output/bin/golden", cs).resolve()
        output.mkdir(parents=True, exist_ok=True)
        ret = unary_operator_gen_data(case_name=cs, output=output)
    return ret


if __name__ == "__main__":
    exit(0 if main() else 1)
