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

""" NormOperator 相关用例 Golden 生成逻辑.

本脚本有 2 种执行模式:
1. CI批跑时, 由 cmake/scripts/golden_ctrl.py 调用, 为避免日志过多, 此时 logging 级别为 logging.INFO;
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
    g_ctrl_path: Path = Path(g_src_root, "cmake/scripts")
    if str(g_ctrl_path) not in sys.path:
        sys.path.append(str(g_ctrl_path))
    from golden_register import GoldenRegister  # 单独调试 import 失败, 需确认上文中 '系统 import 路径' 配置正确
else:
    from golden_register import GoldenRegister


def gen_softmax(x, dim=-1, half_to_float=False):
    ori_dtype = x.dtype
    return torch.ops.aten._softmax(x.to(torch.float32), dim, half_to_float).to(ori_dtype)


def gen_sum(x, dim=-1):
    ori_dtype = x.dtype
    return x.astype(np.float32).sum(axis=dim, keepdims=True).astype(ori_dtype)


def gen_max(x, dim=-1):
    ori_dtype = x.dtype
    return x.astype(np.float32).max(axis=dim, keepdims=True).astype(ori_dtype)


def gen_exp(x):
    return np.exp(x)


@GoldenRegister.reg_golden_func(
    case_names=[
        # Softmax
        "SoftmaxOnBoard.test_softmax_cast_in",
        "SoftmaxOnBoard.test_softmax_cast_out",
        "SoftmaxOnBoard.test_softmax_sum_single",
        "SoftmaxOnBoard.test_softmax_max_single",
        "SoftmaxOnBoard.test_softmax_exp",
        "SoftmaxOnBoard.test_softmax_div",
        "SoftmaxOnBoard.test_softmax_sum_all",
        "SoftmaxOnBoard.test_softmax_full_inference",
        "SoftmaxOnBoard.test_softmax_deepseek",
        "SoftmaxOnBoard.test_softmax_flash_attention",
        "SoftmaxOnBoard.test_softmax_dyn",
    ]
)
def norm_operator_func(case_name: str, output: Path) -> bool:
    if case_name == "SoftmaxOnBoard.test_softmax_cast_in":
        shape = [2, 2, 1, 128]
        stype = torch.float16
        dtype = torch.float32
        x_path = Path(output, 'x_softmax_cast_in.bin')
        y_path = Path(output, 'softmax_cast_in.bin')
        complete = x_path.exists() and y_path.exists()
        if complete:
            logging.debug("Case(%s), Golden complete.", case_name)
        else:
            x = torch.randn(shape).to(stype)
            x.numpy().tofile(x_path)
            y = x.to(dtype)
            y.numpy().tofile(y_path)
    elif case_name == "SoftmaxOnBoard.test_softmax_cast_out":
        shape = [2, 2, 1, 128]
        stype = torch.float32
        dtype = torch.float16
        x_path = Path(output, 'x_softmax_cast_out.bin')
        y_path = Path(output, 'softmax_cast_out.bin')
        complete = x_path.exists() and y_path.exists()
        if complete:
            logging.debug("Case(%s), Golden complete.", case_name)
        else:
            x = torch.randn(shape).to(stype)
            x.numpy().tofile(x_path)
            y = x.to(dtype)
            y.numpy().tofile(y_path)
    elif case_name == "SoftmaxOnBoard.test_softmax_sum_single":
        shape = [2, 2, 1, 128]
        dtype = np.float32
        x_path = Path(output, 'x_sum.bin')
        y_path = Path(output, 'softmax_sum.bin')
        complete = x_path.exists() and y_path.exists()
        if complete:
            logging.debug("Case(%s), Golden complete.", case_name)
        else:
            x = np.random.uniform(-1, 1, shape).astype(dtype)
            x.tofile(x_path)
            y = gen_sum(x, -1)

            y = y.astype(dtype)
            y.tofile(y_path)
    elif case_name == "SoftmaxOnBoard.test_softmax_max_single":
        shape = [2, 2, 1, 128]
        dtype = np.float32
        x_path = Path(output, 'x_max.bin')
        y_path = Path(output, 'softmax_max.bin')
        complete = x_path.exists() and y_path.exists()
        if complete:
            logging.debug("Case(%s), Golden complete.", case_name)
        else:
            x = np.random.uniform(-1, 1, shape).astype(dtype)
            x.tofile(x_path)
            y = gen_max(x, -1)
            y = y.astype(dtype)
            y.tofile(y_path)
    elif case_name == "SoftmaxOnBoard.test_softmax_exp":
        shape = [2, 2, 1, 128]
        dtype = np.float32
        x_path = Path(output, 'x_exp.bin')
        y_path = Path(output, "softmax_exp.bin")
        complete = x_path.exists() and y_path.exists()
        if complete:
            logging.debug("Case(%s), Golden complete.", case_name)
        else:
            x = np.random.uniform(-1, 1, shape).astype(dtype)
            x.tofile(x_path)
            y = gen_exp(x)
            y.tofile(y_path)
    elif case_name == "SoftmaxOnBoard.test_softmax_div":
        l_shape = [2, 2, 1, 128]
        r_shape = [2, 2, 1, 1]
        o_shape = [2, 2, 1, 128]
        dtype = np.float32
        x_path = Path(output, 'x_div.bin')
        y_path = Path(output, 'y_div.bin')
        z_path = Path(output, "softmax_div.bin")
        complete = x_path.exists() and y_path.exists() and z_path.exists()
        if complete:
            logging.debug("Case(%s), Golden complete.", case_name)
        else:
            x = np.random.uniform(-1, 1, l_shape).astype(dtype)
            y = np.random.uniform(-1, 1, r_shape).astype(dtype)
            x.tofile(x_path)
            y.tofile(y_path)
            z = x / y
            z.tofile(z_path)
    elif case_name == "SoftmaxOnBoard.test_softmax_sum_all":
        shape = [2, 2, 1, 128]
        dtype = torch.float16
        x_path = Path(output, 'x_sum_all.bin')
        y_path = Path(output, 'softmax_sum_all.bin')
        complete = x_path.exists() and y_path.exists()
        if complete:
            logging.debug("Case(%s), Golden complete.", case_name)
        else:
            x = torch.randn(shape).to(dtype)
            x.numpy().tofile(x_path)
            y = gen_softmax(x)
            y.numpy().tofile(y_path)
    elif case_name == "SoftmaxOnBoard.test_softmax_full_inference":
        shape = [2, 2, 32, 256]
        dtype = torch.float16
        x_path = Path(output, 'x_full.bin')
        y_path = Path(output, 'softmax_full_inference.bin')
        complete = x_path.exists() and y_path.exists()
        if complete:
            logging.debug("Case(%s), Golden complete.", case_name)
        else:
            x = torch.randn(shape).to(dtype)
            x.numpy().tofile(x_path)
            y = gen_softmax(x)
            y.numpy().tofile(y_path)
    elif case_name == "SoftmaxOnBoard.test_softmax_deepseek":
        shape = [4, 8, 1, 512]
        dtype = torch.float16
        x_path = Path(output, 'x_deepseek.bin')
        y_path = Path(output, 'softmax_deepseek.bin')
        complete = x_path.exists() and y_path.exists()
        if complete:
            logging.debug("Case(%s), Golden complete.", case_name)
        else:
            x = torch.randn(shape).to(dtype)
            x.numpy().tofile(x_path)
            y = gen_softmax(x)
            y.numpy().tofile(y_path)
    elif case_name == "SoftmaxOnBoard.test_softmax_flash_attention":
        shape = [32, 32, 1, 256]
        dtype = torch.float32
        x_path = Path(output, 'x.bin')
        y_path = Path(output, 'softmax.bin')
        complete = x_path.exists() and y_path.exists()
        if complete:
            logging.debug("Case(%s), Golden complete.", case_name)
        else:
            x = torch.randn(shape).to(dtype)
            x.numpy().tofile(x_path)
            y = gen_softmax(x)
            y.numpy().tofile(y_path)
    elif case_name == "SoftmaxOnBoard.test_softmax_dyn":
        shape = [32, 32, 1, 256]
        dtype = torch.float32
        x_path = Path(output, 'x.bin')
        y_path = Path(output, 'softmax.bin')
        complete = x_path.exists() and y_path.exists()
        if complete:
            logging.debug("Case(%s), Golden complete.", case_name)
        else:
            x = torch.randn(shape).to(dtype)
            x.numpy().tofile(x_path)
            y = gen_softmax(x)
            y.numpy().tofile(y_path)
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
        "SoftmaxOnBoard.test_softmax_cast_in",
    ]
    # 函数调用
    ret: bool = True
    for cs in case_name_list:
        output: Path = Path(g_src_root, "build/output/bin/golden", cs).resolve()
        output.mkdir(parents=True, exist_ok=True)
        ret = norm_operator_func(case_name=cs, output=output)
    return ret


if __name__ == "__main__":
    exit(0 if main() else 1)
