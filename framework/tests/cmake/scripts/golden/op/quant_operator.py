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
1. CI批跑时, 由 cmake/scripts/golden_ctrl.py 调用, 为避免日志过多, 此时 logging 级别为 logging.INFO;
2. 单独调试时, 本脚本单独被调用, 此时 logging 级别为 logging.DEBUG;
"""
import time
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

np.random.seed(0)

dtype_f16 = np.float16
dtype_f32 = np.float32


def test_scalar_div_s(b, s, scalar, reverse_operand, output_dir: Path):
    qkv_shape = [b * s, 1]
    logging.debug(f'shape --------> b {b} s {s} dir {output_dir}\n')
    input_path = Path(output_dir, 'input.bin')
    res_path = Path(output_dir, 'res.bin')
    input_t = np.arange(1, b * s + 1, 1).reshape(qkv_shape).astype(dtype_f32)
    logging.debug(f'input:\n{input_t}')

    if reverse_operand:
        res = scalar / input_t
    else:
        res = input_t / scalar
    logging.debug(f'max {res.max()} res:\n {res}')

    input_t.tofile(input_path)
    res.tofile(res_path)


def test_scalar_add_s(b, s, scalar, reverse_operand, output_dir: Path):
    qkv_shape = [b * s, 1]
    logging.debug(f'shape --------> b {b} s {s} dir {output_dir}\n')
    input_path = Path(output_dir, 'input.bin')
    res_path = Path(output_dir, 'res.bin')
    input_t = np.arange(1, b * s + 1, 1).reshape(qkv_shape).astype(dtype_f32)
    logging.debug(f'input:\n{input_t}')

    if reverse_operand:
        res = scalar + input_t
    else:
        res = input_t + scalar
    logging.debug(f'max {res.max()} res:\n {res}')

    input_t.tofile(input_path)
    res.tofile(res_path)


def test_scalar_sub_s(b, s, scalar, reverse_operand, output_dir: Path):
    qkv_shape = [b * s, 1]
    logging.debug(f'shape --------> b {b} s {s} dir {output_dir}\n')
    input_path = Path(output_dir, 'input.bin')
    res_path = Path(output_dir, 'res.bin')
    input_t = np.arange(1, b * s + 1, 1).reshape(qkv_shape).astype(dtype_f32)
    logging.debug(f'input:\n{input_t}')

    if reverse_operand:
        res = scalar - input_t
    else:
        res = input_t - scalar
    logging.debug(f'max {res.max()} res:\n {res}')

    input_t.tofile(input_path)
    res.tofile(res_path)


def test_scalar_mul_s(b, s, scalar, reverse_operand, output_dir: Path):
    qkv_shape = [b * s, 1]
    logging.debug(f'shape --------> b {b} s {s} dir {output_dir}\n')
    input_path = Path(output_dir, 'input.bin')
    res_path = Path(output_dir, 'res.bin')
    input_t = np.arange(1, b * s + 1, 1).reshape(qkv_shape).astype(dtype_f32)
    logging.debug(f'input:\n{input_t}')

    if reverse_operand:
        res = scalar * input_t
    else:
        res = input_t * scalar
    logging.debug(f'max {res.max()} res:\n {res}')

    input_t.tofile(input_path)
    res.tofile(res_path)


def test_scalar_max_s(b, s, scalar, reverse_operand, output_dir: Path):
    qkv_shape = [b * s, 1]
    logging.debug(f'shape --------> b {b} s {s} dir {output_dir}\n')
    input_path = Path(output_dir, 'input.bin')
    res_path = Path(output_dir, 'res.bin')
    input_t = np.arange(126, b * s + 126, 1).reshape(qkv_shape).astype(dtype_f32)
    logging.debug(f'input:\n{input_t}')

    res = np.maximum(input_t, scalar)

    logging.debug(f'max {res.max()} res:\n {res}')

    input_t.tofile(input_path)
    res.tofile(res_path)


def test_scalar_op_s(scalar, reverse_operand, output_dir: Path):
    qkv_shape = [128, 35]
    logging.debug(f'shape --------> 128 dir {output_dir}\n')
    input_path = Path(output_dir, 'input.bin')
    res_path = Path(output_dir, 'res.bin')
    input_t = np.arange(1, 128 * 35 + 1, 1).reshape(qkv_shape).astype(dtype_f32)
    logging.debug(f'input_t:\n{input_t}')

    res = np.maximum(input_t, scalar)

    logging.debug(f'max {res.max()} res:\n {res}')

    input_t.tofile(input_path)
    res.tofile(res_path)


@GoldenRegister.reg_golden_func(
    case_names=[
        "QuantTest.Test_ScalarDivS",
        "QuantTest.Test_ScalarAddS",
        "QuantTest.Test_ScalarSubS",
        "QuantTest.Test_ScalarMulS",
        "QuantTest.Test_ScalarMaxS",
        "QuantTest.Test_ScalarOp",
    ]
)
def get_quant_golden(case_name: str, output: Path) -> bool:
    dtype = np.float32
    indices_dtype = np.int32

    input_path = Path(output, 'input.bin')
    res_path = Path(output, 'res.bin')
    complete = input_path.exists() and res_path.exists()
    if complete:
        file_mod_time = input_path.stat().st_mtime
        # 获取当前时间（Unix 时间戳）
        current_time = time.time()
        # 判断文件的修改时间是否超过1小时（3600秒）
        if current_time - file_mod_time > 3600:
            logging.debug("文件的修改时间超过1小时，重新生成文件...")
            complete = False
        else:
            logging.debug("文件的修改时间在1小时内，无需重新生成。")

    if complete:
        logging.debug("Case(%s), Golden data exits. cache catch", case_name)
    else:
        if case_name == "QuantTest.Test_ScalarDivS":
            b, s, scalar = 2, 1, 127
            test_scalar_div_s(b, s, scalar, True, output)

        elif case_name == "QuantTest.Test_ScalarAddS":
            b, s, scalar = 2, 1, 127
            test_scalar_add_s(b, s, scalar, True, output)

        elif case_name == "QuantTest.Test_ScalarSubS":
            b, s, scalar = 2, 1, 127
            test_scalar_sub_s(b, s, scalar, True, output)

        elif case_name == "QuantTest.Test_ScalarMulS":
            b, s, scalar = 2, 1, 127
            test_scalar_mul_s(b, s, scalar, True, output)

        elif case_name == "QuantTest.Test_ScalarMaxS":
            b, s, scalar = 2, 1, 126
            test_scalar_max_s(b, s, scalar, True, output)

        elif case_name == "QuantTest.Test_ScalarOp":
            scalar = 127
            test_scalar_op_s(scalar, True, output)

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
        "QuantTest.Test_ScalarSubS",
    ]
    # 函数调用
    ret: bool = True
    for cs in case_name_list:
        output: Path = Path(g_src_root, "build/output/bin/golden", cs).resolve()
        output.mkdir(parents=True, exist_ok=True)
        ret = get_quant_golden(case_name=cs, output=output)
    return ret


if __name__ == "__main__":
    exit(0 if main() else 1)
