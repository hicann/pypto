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

fp32 = np.float32


def gen_quant_data(input_shape, dtype, output_dir: Path, is_symmetry=True):
    logging.debug(f"gen_quant_data  dtype:{dtype}")

    input_path = Path(output_dir, 'input.bin')
    output_golden_path = Path(output_dir, 'output_golden.bin')
    scale_dequant_golden_path = Path(output_dir, 'scale_dequant_golden.bin')

    input_x = np.random.uniform(-1, 1, input_shape).astype(dtype)
    input_x.tofile(input_path)

    """ numpy """
    logging.debug("================ numpy ================")
    input_fp32 = input_x.astype(fp32)
    if is_symmetry:
        abs_res = np.abs(input_fp32)
        max_value = np.max(abs_res, axis=-1, keepdims=True)
        scale_quant = 127 / max_value
        out_fp32 = input_fp32 * scale_quant
        out_int32 = np.rint(out_fp32).astype(np.int32)
        out_fp16 = out_int32.astype(np.float16)
        out_int8 = np.trunc(out_fp16).astype(np.int8)
        scale_dequant = 1 / scale_quant
    """ golden output """
    logging.debug("out_int8 shape %s", out_int8.shape)
    out_int8.tofile(output_golden_path)
    scale_dequant.tofile(scale_dequant_golden_path)


@GoldenRegister.reg_golden_func(
    case_names=[
        # Quant
        "QuantOnBoardTest.test_Quant_32_1_7168",
        "QuantOnBoardTest.test_Quant_32_7168",
    ]
)
def gen_quant_op_date(case_name: str, output: Path) -> bool:
    input_path = Path(output, 'input.bin')
    output_golden_path = Path(output, 'output_golden.bin')
    scale_dequant_golden_path = Path(output, 'scale_dequant_golden.bin')
    logging.debug("input_path is : %s", input_path)

    complete = (input_path.exists() and output_golden_path.exists() and scale_dequant_golden_path.exists())
    if complete:
        logging.debug("Case(%s), Golden complete.", case_name)
    else:
        if case_name == "QuantOnBoardTest.test_Quant_32_1_7168":
            gen_quant_data([32, 7168], np.float16, output)
        elif case_name == "QuantOnBoardTest.test_Quant_32_7168":
            gen_quant_data([32, 7168], np.float16, output)
        else:
            logging.error("Can't get func to gen golden, Case(%s)", case_name)
            return False
    return True


def gen_quant_data_with_smooth_factor(input_shape, dtype, output_dir: Path, is_symmetry=True):
    logging.debug(f"gen_quant_data  dtype:{dtype}")

    input_path = Path(output_dir, 'input.bin')
    smooth_factor_path = Path(output_dir, 'smooth_factor.bin')
    output_golden_path = Path(output_dir, 'output_golden.bin')
    scale_dequant_golden_path = Path(output_dir, 'scale_dequant_golden.bin')

    input_x = np.random.uniform(-1, 1, input_shape).astype(dtype)
    input_x.tofile(input_path)

    smooth_factor = np.random.uniform(-1, 1, size=(1, input_shape[-1])).astype(np.float32)
    smooth_factor.tofile(smooth_factor_path)

    """ numpy """
    logging.debug("================ numpy ================")
    input_fp32_without_smooth = input_x.astype(fp32)
    input_fp32 = input_fp32_without_smooth * smooth_factor
    if is_symmetry:
        abs_res = np.abs(input_fp32)
        max_value = np.max(abs_res, axis=-1, keepdims=True)
        scale_quant = 127 / max_value
        out_fp32 = input_fp32 * scale_quant
        out_int32 = np.rint(out_fp32).astype(np.int32)
        out_fp16 = out_int32.astype(np.float16)
        out_int8 = np.trunc(out_fp16).astype(np.int8)
        scale_dequant = 1 / scale_quant
    """ golden output """
    logging.debug("out_int8 shape %s", out_int8.shape)
    out_int8.tofile(output_golden_path)
    scale_dequant.tofile(scale_dequant_golden_path)


@GoldenRegister.reg_golden_func(
    case_names=[
        "QuantOnBoardTest.test_Quant_Smooth_32_7168",
        "QuantOnBoardTest.test_Quant_Smooth_32_4_128",
    ]
)
def gen_quant_op_date_with_smooth_factor(case_name: str, output: Path) -> bool:
    input_path = Path(output, 'input.bin')
    smooth_factor_path = Path(output, 'smooth_factor.bin')
    output_golden_path = Path(output, 'output_golden.bin')
    scale_dequant_golden_path = Path(output, 'scale_dequant_golden.bin')
    logging.debug("input_path is : %s", input_path)

    complete = (input_path.exists() and smooth_factor_path.exists()
                and output_golden_path.exists() and scale_dequant_golden_path.exists())
    if complete:
        logging.debug("Case(%s), Golden complete.", case_name)
    else:
        if case_name == "QuantOnBoardTest.test_Quant_Smooth_32_4_128":
            gen_quant_data_with_smooth_factor([32, 4, 128], np.float16, output)

        elif case_name == "QuantOnBoardTest.test_Quant_Smooth_32_7168":
            gen_quant_data_with_smooth_factor([32, 7168], np.float16, output)

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
        "QuantOnBoardTest.test_Quant_32_7168",
    ]
    # 函数调用
    ret: bool = True
    for cs in case_name_list:
        output: Path = Path(g_src_root, "build/output/bin/golden", cs).resolve()
        output.mkdir(parents=True, exist_ok=True)
        ret = gen_quant_op_date(case_name=cs, output=output)
    return ret


if __name__ == "__main__":
    exit(0 if main() else 1)
