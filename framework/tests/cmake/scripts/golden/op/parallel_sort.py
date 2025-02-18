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
""" parallel sort 相关用例 Golden 生成逻辑.

本脚本有 2 种执行模式:
1. CI批跑时, 由 tests/cmake/scripts/golden_ctrl.py 调用, 为避免日志过多, 此时 logging 级别为 logging.INFO;
2. 单独调试时, 本脚本单独被调用, 此时 logging 级别为 logging.DEBUG;
"""
import sys
import logging
from pathlib import Path
from typing import List

import torch
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


def dump_sort_params(params: List, output: Path):
    np.array(params).astype(np.int32).tofile(Path(output, 'params.bin'))


def gen_op_sort_golden(length: int, descending: bool, dtype: torch.dtype, output: Path):
    dump_sort_params([length, descending], output)
    x_path, idx_path = Path(output, 'x.bin'), Path(output, 'idx.bin')
    y_path, yidx_path = Path(output, 'y.bin'), Path(output, 'yidx.bin')
    x = torch.randperm(length, dtype=dtype).reshape(1, length)
    idx = torch.arange(length, dtype=torch.int32).reshape(1, length)
    y, sorted_idx = torch.sort(x, dim=-1, descending=descending)
    yidx = torch.gather(idx, 1, sorted_idx)
    x.numpy().tofile(x_path)
    idx.numpy().tofile(idx_path)
    y.numpy().tofile(y_path)
    yidx.numpy().tofile(yidx_path)


def gen_op_compswap_golden(length: int, descending: bool, dtype: torch.dtype, output: Path):
    dump_sort_params([length, descending], output)
    x0_path, idx0_path = Path(output, 'x0.bin'), Path(output, 'idx0.bin')
    x1_path, idx1_path = Path(output, 'x1.bin'), Path(output, 'idx1.bin')
    y0_path, yidx0_path = Path(output, 'y0.bin'), Path(output, 'yidx0.bin')
    y1_path, yidx1_path = Path(output, 'y1.bin'), Path(output, 'yidx1.bin')

    halflength = length // 2
    x0 = torch.randperm(halflength, dtype=dtype).reshape(1, halflength)
    x1 = torch.randperm(halflength, dtype=dtype).reshape(1, halflength)
    idx0 = torch.arange(0, halflength, dtype=torch.int32).reshape(1, halflength)
    idx1 = torch.arange(halflength, length, dtype=torch.int32).reshape(1, halflength)
    x = torch.cat((x0, x1), dim=0)
    idx = torch.cat((idx0, idx1), dim=0)
    if descending:
        y0, max_idx = x.max(dim=0, keepdims=True)
        y1, min_idx = x.min(dim=0, keepdims=True)
        min_idx = 1 - max_idx   # avoid index duplicate due to same value
        yidx0 = torch.gather(idx, 0, max_idx)
        yidx1 = torch.gather(idx, 0, min_idx)
    else:
        y0, min_idx = x.min(dim=0, keepdims=True)
        y1, max_idx = x.max(dim=0, keepdims=True)
        max_idx = 1 - min_idx   # avoid index duplicate due to same value
        yidx0 = torch.gather(idx, 0, min_idx)
        yidx1 = torch.gather(idx, 0, max_idx)
    x0.numpy().tofile(x0_path)
    x1.numpy().tofile(x1_path)
    idx0.numpy().tofile(idx0_path)
    idx1.numpy().tofile(idx1_path)
    y0.numpy().tofile(y0_path)
    y1.numpy().tofile(y1_path)
    yidx0.numpy().tofile(yidx0_path)
    yidx1.numpy().tofile(yidx1_path)


def gen_op_merge_golden(length: int, descending: bool, dtype: torch.dtype, output: Path):
    dump_sort_params([length, descending], output)
    x_path, idx_path = Path(output, 'x.bin'), Path(output, 'idx.bin')
    y_path, yidx_path = Path(output, 'y.bin'), Path(output, 'yidx.bin')
    x = torch.randperm(length, dtype=dtype)
    half = length // 2
    x1 = x[:half].sort(descending=False).values
    x2 = x[half:].sort(descending=True).values
    x = torch.cat((x1, x2), dim=0).reshape(1, length)
    idx = torch.arange(0, length, dtype=torch.int32).reshape(1, length)
    y, sorted_idx = torch.sort(x, dim=-1, descending=descending)
    yidx = torch.gather(idx, 1, sorted_idx)
    x.numpy().tofile(x_path)
    idx.numpy().tofile(idx_path)
    y.numpy().tofile(y_path)
    yidx.numpy().tofile(yidx_path)


def gen_sort_golden(length: int, descending: bool, dtype: torch.dtype, output: Path):
    dump_sort_params([length, descending], output)
    x_path, idx_path = Path(output, 'x.bin'), Path(output, 'idx.bin')
    y_path, yidx_path = Path(output, 'y.bin'), Path(output, 'yidx.bin')
    x = torch.randperm(length, dtype=dtype).reshape(1, length)
    idx = torch.arange(0, length, dtype=torch.int32).reshape(1, length)
    y, sorted_idx = torch.sort(x, dim=-1, descending=descending)
    yidx = torch.gather(idx, 1, sorted_idx)
    x.numpy().tofile(x_path)
    idx.numpy().tofile(idx_path)
    y.numpy().tofile(y_path)
    yidx.numpy().tofile(yidx_path)


def gen_topk_golden(length: int, k: int, descending: bool, dtype: torch.dtype, output: Path):
    dump_sort_params([length, descending, k], output)
    x_path, idx_path = Path(output, 'x.bin'), Path(output, 'idx.bin')
    y_path, yidx_path = Path(output, 'y.bin'), Path(output, 'yidx.bin')
    x = torch.randperm(length, dtype=dtype).reshape(1, length)
    idx = torch.arange(0, length, dtype=torch.int32).reshape(1, length)
    y, sorted_idx = torch.sort(x, dim=-1, descending=descending)
    yidx = torch.gather(idx, 1, sorted_idx)
    x.numpy().tofile(x_path)
    idx.numpy().tofile(idx_path)
    y[:, :k].numpy().tofile(y_path)
    yidx[:, :k].numpy().tofile(yidx_path)


@GoldenRegister.reg_golden_func(
    case_names=[
        "ParallelSortSTest.op_sort",
        "ParallelSortSTest.op_compswap",
        "ParallelSortSTest.op_merge",
        "ParallelSortSTest.sort_static",
        "ParallelSortSTest.sort",
        "ParallelSortSTest.sort_index",
        "ParallelSortSTest.topk",
        "ParallelSortSTest.fp32_64k",
        "ParallelSortSTest.fp32_128k",
        "ParallelSortSTest.topk_128k_2k",
    ]
)
def gen_golden(case_name: str, output: Path) -> bool:
    if case_name == "ParallelSortSTest.op_sort":
        gen_op_sort_golden(8192, False, torch.float32, output)
    elif case_name == "ParallelSortSTest.op_compswap":
        gen_op_compswap_golden(8192, False, torch.float32, output)
    elif case_name == "ParallelSortSTest.op_merge":
        gen_op_merge_golden(8192, False, torch.float32, output)
    elif case_name == "ParallelSortSTest.sort_static":
        gen_sort_golden(2048, False, torch.float32, output)
    elif case_name == "ParallelSortSTest.sort":
        gen_sort_golden(2048, False, torch.float32, output)
    elif case_name == "ParallelSortSTest.sort_index":
        gen_sort_golden(2048, False, torch.float32, output)
    elif case_name == "ParallelSortSTest.topk":
        gen_topk_golden(2048, 512, True, torch.float32, output)
    elif case_name == "ParallelSortSTest.fp32_64k":
        gen_sort_golden(1024 * 64, True, torch.float32, output)
    elif case_name == "ParallelSortSTest.fp32_128k":
        gen_sort_golden(1024 * 128, True, torch.float32, output)
    elif case_name == "ParallelSortSTest.topk_128k_2k":
        gen_topk_golden(1024 * 128, 1024 * 2, True, torch.float32, output)
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
        "ParallelSortSTest.sort",
    ]
    # 函数调用
    ret: bool = True
    for cs in case_name_list:
        output: Path = Path(g_src_root, "build/output/bin/golden", cs).resolve()
        output.mkdir(parents=True, exist_ok=True)
        ret = gen_golden(case_name=cs, output=output)
    return ret


if __name__ == "__main__":
    exit(0 if main() else 1)
