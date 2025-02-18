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

import torch
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

dtype_f32 = torch.float32
dtype_bf16 = torch.bfloat16
dtype_int32 = torch.int32
dtype_int8 = torch.int8


def tensor_tofile(t: torch.Tensor, output: Path):
    input_file_bin = open(str(output), "wb")
    for each in t:
        if t.dtype == torch.bfloat16:
            input_file_bin.write(each.view(torch.int16).numpy().tobytes())
        elif t.dtype == torch.float32:
            input_file_bin.write(each.view(torch.int32).numpy().tobytes())
        elif t.dtype == torch.int32:
            input_file_bin.write(each.numpy().tobytes())
        elif t.dtype == torch.int8:
            input_file_bin.write(each.numpy().tobytes())
        else:
            raise ValueError(f"Unsupported dtype: {t.dtype}")
    input_file_bin.close()


def quantize_torch(input_fp32):
    abs_res = torch.abs(input_fp32)
    max_value, _ = torch.max(abs_res, dim=-1, keepdim=True)
    scale_quant = 127.0 / max_value
    out_fp32 = input_fp32 * scale_quant
    out_int32 = torch.round(out_fp32).to(torch.int32)
    out_int8 = torch.clamp(out_int32, -128, 127).to(torch.int8)
    scale_dequant = 1.0 / scale_quant
    return out_int8, scale_dequant


def gen_quant_mm_torch(a, w, scale_w):
    a_fp32 = a.to(torch.float32)
    quantized_a, scale_dequant_a = quantize_torch(a_fp32)
    a_int32 = quantized_a.to(torch.int32)
    w_int32 = w.to(torch.int32)
    res_int32 = torch.matmul(a_int32, w_int32)
    res = res_int32.to(torch.float32)
    res = res * scale_dequant_a
    res = res * scale_w
    return res.to(a.dtype)


def quantize(input_fp32):
    abs_res = np.abs(input_fp32)
    max_value = np.max(abs_res, axis=-1, keepdims=True)
    scale_quant = 127 / max_value
    out_fp32 = input_fp32 * scale_quant
    out_int32 = np.rint(out_fp32).astype(np.int32)
    out_fp16 = out_int32.astype(np.float16)
    out_int8 = np.trunc(out_fp16).astype(np.int8)
    scale_dequant = 1 / scale_quant
    return out_int8, scale_dequant


def gen_quant_mm(a, w, scale_w):
    a_fp32 = a.astype(np.float32)
    quantized_a, scale_dequant_a = quantize(a_fp32)
    a_int32 = quantized_a.astype(np.int32)
    w_int32 = w.astype(np.int32)
    res_int32 = np.dot(a_int32, w_int32)
    res = res_int32.astype(np.float32)
    res = res * scale_dequant_a
    res = res * scale_w
    res = res.astype(bfloat16)
    return res


@GoldenRegister.reg_golden_func(
    case_names=[
        # QuantMM
        "QuantMMOnBoardTest.test_QuantMM_128_32_512_times_128_512_128_torch",
    ]
)
def quant_mm_func_torch(case_name: str, output: Path) -> bool:
    if case_name == "QuantMMOnBoardTest.test_QuantMM_128_32_512_times_128_512_128_torch":
        shape_a = (128, 32, 512)
        shape_w = (128, 512, 128)
        shape_scale_w = (128, 1, 128)
        a_dtype = torch.bfloat16
        w_dtype = torch.int8
        scale_w_dtype = torch.float32
        a_path = Path(output, 'quant_mm_a.bin')
        w_path = Path(output, 'quant_mm_w.bin')
        scale_w_path = Path(output, 'quant_mm_scale_w.bin')
        golden_path = Path(output, 'quant_mm_golden.bin')
        complete = a_path.exists() and w_path.exists() and golden_path.exists()
        if complete:
            logging.debug("Case(%s), Golden complete.", case_name)
        else:
            a = torch.rand(shape_a, dtype=a_dtype)
            tensor_tofile(a, a_path)
            w = torch.randint(size=shape_w, low=-128, high=128, dtype=w_dtype)
            tensor_tofile(w, w_path)
            scale_w = torch.randn(shape_scale_w, dtype=scale_w_dtype)
            tensor_tofile(scale_w, scale_w_path)
            golden = gen_quant_mm_torch(a, w, scale_w)
            tensor_tofile(golden, golden_path)

    else:
        logging.error("Can't get func to gen golden, Case(%s)", case_name)
        return False
    return True


@GoldenRegister.reg_golden_func(
    case_names=[
        # QuantMM
        "QuantMMOnBoardTest.test_QuantMM_32_16384_times_16384_7168_np",
    ]
)
def quant_mm_func(case_name: str, output: Path) -> bool:
    if case_name == "QuantMMOnBoardTest.test_QuantMM_32_16384_times_16384_7168_np":
        shape_a = (32, 16384)
        shape_w = (16384, 7168)
        shape_scale_w = (1, 7168)
        a_dtype = bfloat16
        w_dtype = np.int8
        scale_w_dtype = np.float32
        a_path = Path(output, 'quant_mm_a.bin')
        w_path = Path(output, 'quant_mm_w.bin')
        scale_w_path = Path(output, 'quant_mm_scale_w.bin')
        golden_path = Path(output, 'quant_mm_golden.bin')
        complete = a_path.exists() and w_path.exists() and golden_path.exists()
        if complete:
            logging.debug("Case(%s), Golden complete.", case_name)
        else:
            a = np.random.uniform(size=shape_a).astype(a_dtype)
            a.tofile(a_path)
            w = np.random.randint(low=-128, high=128, size=shape_w).astype(w_dtype)
            w.tofile(w_path)
            scale_w = np.random.uniform(-1, 1, shape_scale_w).astype(scale_w_dtype)
            scale_w.tofile(scale_w_path)
            golden = gen_quant_mm(a, w, scale_w)
            golden.tofile(golden_path)

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
        "QuantMMOnBoardTest.test_QuantMM_32_16384_times_16384_7168_np",
    ]
    # 函数调用
    ret: bool = True
    for cs in case_name_list:
        output: Path = Path(g_src_root, "build/output/bin/golden", cs).resolve()
        output.mkdir(parents=True, exist_ok=True)
        ret = quant_mm_func(case_name=cs, output=output)
    return ret


if __name__ == "__main__":
    exit(0 if main() else 1)
