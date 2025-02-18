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
""" Gather Operator 相关用例 Golden 生成逻辑.

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


def reshape_with_matmul(output_dir: Path):
    a_r_path = Path(output_dir, 'a_r.bin')
    a_path = Path(output_dir, 'a.bin')
    b_path = Path(output_dir, 'b.bin')
    res_path = Path(output_dir, 'res.bin')

    m, k, n = 64, 64, 64
    shape_a_r = [8, 8, 64]
    shape_a = [m, k]
    shape_b = [k, n]
    shape_res = [m, n]
    dtype = np.float16
    np.random.seed(42)
    a_r = np.random.uniform(-1, 1, shape_a_r).astype(dtype)
    a_r.tofile(a_r_path)
    a = a_r.reshape(shape_a)
    a.tofile(a_path)
    b = np.random.uniform(-1, 1, shape_b).astype(dtype)
    b.tofile(b_path)
    res = np.matmul(a.astype(np.float32), b.astype(np.float32))
    res.tofile(res_path)


def reshape_matmul_mul(output_dir: Path):
    a_r_path = Path(output_dir, 'a_r.bin')
    a_path = Path(output_dir, 'a.bin')
    b_path = Path(output_dir, 'b.bin')
    c_path = Path(output_dir, 'c.bin')
    d_path = Path(output_dir, 'd.bin')
    e_path = Path(output_dir, 'e.bin')
    res_path = Path(output_dir, 'res.bin')

    m, k, n = 64, 64, 64
    shape_a_r = [8, 8, 64]
    shape_a = [m, k]
    shape_b = [k, n]
    shape_res = [m, n]
    dtype = np.float16
    np.random.seed(42)
    a_r = np.random.uniform(-1, 1, shape_a_r).astype(dtype)
    a_r.tofile(a_r_path)
    a = a_r.reshape(shape_a)
    a.tofile(a_path)
    b = np.random.uniform(-1, 1, shape_b).astype(dtype)
    b.tofile(b_path)
    c = np.matmul(a.astype(np.float32), b.astype(np.float32))
    c.tofile(c_path)
    d = c.reshape([8, 8, 64])
    d.tofile(d_path)
    e = np.random.uniform(-1, 1, [8, 8, 64]).astype(np.float32)
    e.tofile(e_path)
    res = d * e
    res.tofile(res_path)


def operation_gm_reshape(output_dir: Path):
    x_path = Path(output_dir, 'reshapegm_x.bin')
    x_r_path = Path(output_dir, 'reshapegm_x_r.bin')
    y_path = Path(output_dir, 'reshapegm_y.bin')
    o_path = Path(output_dir, 'reshapegm_res.bin')
    dtype = np.float32

    x_r = np.random.uniform(-1, 1, [64, 8]).astype(dtype)
    x_r.tofile(x_r_path)
    x = x_r.reshape(8, 8, 8)
    x.tofile(x_path)
    y = np.random.uniform(-1, 1, [8, 8, 8]).astype(dtype)
    y.tofile(y_path)
    x = x / y
    x.tofile(o_path)


def operation_ub_reshape(output_dir: Path):
    x_path = Path(output_dir, 'reshapeub_x.bin')
    x_r_path = Path(output_dir, 'reshapeub_x_r.bin')
    x_r_exp_path = Path(output_dir, 'reshapeub_x_r_exp.bin')
    y_path = Path(output_dir, 'reshapeub_y.bin')
    o_path = Path(output_dir, 'reshapeub_res.bin')
    dtype = np.float32

    x_r = np.random.uniform(-1, 1, [64, 16]).astype(dtype)
    x_r.tofile(x_r_path)
    x_r_exp = np.exp(x_r)
    x_r_exp.tofile(x_r_exp_path)
    x = x_r_exp.reshape(8, 8, 16)
    x.tofile(x_path)
    y = np.random.uniform(-1, 1, [8, 8, 16]).astype(dtype)
    y.tofile(y_path)
    x = x / y
    x.tofile(o_path)


def operation_ub_reshape_3dimto2dim(output_dir: Path):
    x_path = Path(output_dir, 'reshapeub_x.bin')
    x_r_path = Path(output_dir, 'reshapeub_x_r.bin')
    x_r_exp_path = Path(output_dir, 'reshapeub_x_r_exp.bin')
    y_path = Path(output_dir, 'reshapeub_y.bin')
    o_path = Path(output_dir, 'reshapeub_res.bin')
    dtype = np.float32

    np.random.seed(42)
    x_r = np.random.uniform(-1, 1, [32, 8, 8]).astype(dtype)
    x_r.tofile(x_r_path)
    x_r_exp = np.exp(x_r)
    x_r_exp.tofile(x_r_exp_path)
    x = x_r_exp.reshape(32, 64)
    x.tofile(x_path)
    y = np.random.uniform(-1, 1, [32, 64]).astype(dtype)
    y.tofile(y_path)
    x = x / y
    x.tofile(o_path)


def operation_gm_reshape_2dimto3dim(output_dir: Path):
    x_path = Path(output_dir, 'reshapeub_x.bin')
    x_r_path = Path(output_dir, 'reshapeub_x_r.bin')
    x_r_exp_path = Path(output_dir, 'reshapeub_x_r_exp.bin')
    y_path = Path(output_dir, 'reshapeub_y.bin')
    o_path = Path(output_dir, 'reshapeub_res.bin')
    dtype = np.float32

    np.random.seed(42)
    x_r = np.random.uniform(-1, 1, [16, 64]).astype(dtype)
    x_r.tofile(x_r_path)
    x_r_exp = np.exp(x_r)
    x_r_exp.tofile(x_r_exp_path)
    x = x_r_exp.reshape(16, 8, 8)
    x.tofile(x_path)
    y = np.random.uniform(-1, 1, [16, 8, 8]).astype(dtype)
    y.tofile(y_path)
    x = x / y
    x.tofile(o_path)


def operation_ub_withoutreshape_3dimto2dim(output_dir: Path):
    x_path = Path(output_dir, 'reshapeub_x.bin')
    x_r_path = Path(output_dir, 'reshapeub_x_r.bin')
    x_r_exp_path = Path(output_dir, 'reshapeub_x_r_exp.bin')
    y_path = Path(output_dir, 'reshapeub_y.bin')
    o_path = Path(output_dir, 'reshapeub_res.bin')
    dtype = np.float32

    np.random.seed(42)
    x_r = np.random.uniform(-1, 1, [16, 64]).astype(dtype)
    x_r.tofile(x_r_path)
    x_r_exp = np.exp(x_r)
    x_r_exp.tofile(x_r_exp_path)
    # x = x_r_exp.reshape(16, 64)
    # x.tofile(x_path)
    y = np.random.uniform(-1, 1, [16, 64]).astype(dtype)
    y.tofile(y_path)
    x = x_r_exp / y
    x.tofile(o_path)


@GoldenRegister.reg_golden_func(
    case_names=[
        "OnBoardTest.test_operation_ub_reshape",
        "OnBoardTest.test_operation_gm_reshape",
        "OnBoardTest.test_operation_ub_reshape_3dimto2dim",
        "OnBoardTest.test_operation_ub_withoutreshape_3dimto2dim",
        "OnBoardTest.test_operation_gm_reshape_2dimto3dim",
        "OnBoardTest.test_reshape_with_matmul",
        "OnBoardTest.test_reshape_matmul_mul",
    ]
)
def reshape_operator_func1(case_name: str, output: Path) -> bool:
    a_path = Path(output, 'a.bin')
    res_path = Path(output, 'res.bin')
    b_path = Path(output, 'b.bin')

    #    complete = a_path.exists() and res_path.exists() and b_path.exists()

    #    if complete:
    #        logging.debug("Case(%s), Golden complete.", case_name)
    #    else:
    if case_name == "OnBoardTest.test_reshape_with_matmul":
        reshape_with_matmul(output)
    elif case_name == "OnBoardTest.test_reshape_matmul_mul":
        reshape_matmul_mul(output)
    elif case_name == "OnBoardTest.test_operation_gm_reshape":
        operation_gm_reshape(output)
    elif case_name == "OnBoardTest.test_operation_ub_reshape":
        operation_ub_reshape(output)
    elif case_name == "OnBoardTest.test_operation_ub_reshape_3dimto2dim":
        operation_ub_reshape_3dimto2dim(output)
    elif case_name == "OnBoardTest.test_operation_gm_reshape_2dimto3dim":
        operation_gm_reshape_2dimto3dim(output)
    elif case_name == "OnBoardTest.test_operation_ub_withoutreshape_3dimto2dim":
        operation_ub_withoutreshape_3dimto2dim(output)
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
        "OnBoardTest.test_reshape_with_matmul",
    ]
    # 函数调用
    ret: bool = True
    for cs in case_name_list:
        output: Path = Path(g_src_root, "build/output/bin/golden", cs).resolve()
        output.mkdir(parents=True, exist_ok=True)
        ret = reshape_operator_func1(case_name=cs, output=output)
    return ret


if __name__ == "__main__":
    exit(0 if main() else 1)
