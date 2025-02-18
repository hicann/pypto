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
    g_ctrl_path: Path = Path(g_src_root, "tests/cmake/scripts")
    if str(g_ctrl_path) not in sys.path:
        sys.path.append(str(g_ctrl_path))
    from golden_register import GoldenRegister  # 单独调试 import 失败, 需确认上文中 '系统 import 路径' 配置正确
else:
    from golden_register import GoldenRegister

np.random.seed(0)

dtype_f16 = np.float16
dtype_f32 = np.float32


def test_transpose_bnsd_bsnd(b, n, s, d, output_dir: Path):
    qkv_shape = [b, n, s, d]
    logging.debug(f'shape --------> b {b} n {n} s {s} d {d} dir {output_dir}\n')
    input_path = Path(output_dir, 'input.bin')
    res_path = Path(output_dir, 'res.bin')
    input_t = np.arange(0, b * n * s * d, 1).reshape(qkv_shape).astype(dtype_f32)
    logging.debug(f'input:\n{input_t[0, 0, 0, :]}')
    res = input_t.transpose(0, 2, 1, 3)

    res0 = input_t - res
    logging.debug(f'max {res0.max()} res0:\n {res0}')

    input_t.tofile(input_path)
    res.tofile(res_path)
    logging.debug(f'res:\n{res[0, 0, 0, :]}')


def test_transpose_abc_bac(bs, n, d, output_dir: Path):
    qkv_shape = [bs, n, d]
    logging.debug(f'shape --------> bs {bs} n {n} d {d} dir {output_dir}\n')
    input_path = Path(output_dir, 'input.bin')
    res_path = Path(output_dir, 'res.bin')
    input_t = np.arange(0, bs * n * d, 1).reshape(qkv_shape).astype(dtype_f32)
    logging.debug(f'input:\n{input_t[0, 0, :]}')
    res = input_t.transpose(1, 0, 2)

    input_t.tofile(input_path)
    res.tofile(res_path)
    logging.debug(f'res:\n{res[0, 0, :]}')


def test_transpose_bnsd2_bns2_d_small(b, n, s, d, output_dir: Path):
    qkv_shape = [b, n, s, d // 2, 2]
    logging.debug(f'shape --------> b {b} n {n} s {s} d/2 {d // 2} 2 dir {output_dir}\n')
    input_path = Path(output_dir, 'input.bin')
    res_path = Path(output_dir, 'res.bin')

    input_t = np.arange(0, b * n * s * d, 1).reshape(qkv_shape).astype(dtype_f32)
    res = input_t.transpose(0, 1, 2, 4, 3)

    input_t.tofile(input_path)
    res.tofile(res_path)
    logging.debug(f'input:\n{input_t[0, 0, 0, :, :]} res:\n{res[0, 0, 0, :, :]}')


def test_transpose_bnds_bnsd(b, n, s, d, output_dir: Path):
    qkv_shape = [b, n, s, d]
    logging.debug(f'shape --------> b {b} n {n} s {s} d {d} dir {output_dir}\n')
    input_path = Path(output_dir, 'input.bin')
    res_path = Path(output_dir, 'res.bin')
    input_t = np.arange(0, b * n * s * d, 1).reshape(qkv_shape).astype(dtype_f32)
    logging.debug(f'input:\n{input_t[0, 0, 0, :]}')
    res = input_t.transpose(0, 1, 3, 2)

    input_t.tofile(input_path)
    res.tofile(res_path)
    logging.debug(f'res:\n{res[0, 0, 0, :]}')


def test_transpose_add_bnsd_bsnd(b, n, s, d, output_dir: Path):
    qkv_shape = [b, n, s, d]
    logging.debug(f'shape --------> b {b} n {n} s {s} d {d} dir {output_dir}\n')
    input_path = Path(output_dir, 'input.bin')
    res_path = Path(output_dir, 'res.bin')
    input_t = np.arange(0, b * n * s * d, 1).reshape(qkv_shape).astype(dtype_f32)
    logging.debug(f'input:\n{input_t[0, 0, 0, :]}')
    temp = input_t + input_t
    res = temp.transpose(0, 2, 1, 3)

    res0 = input_t - res
    logging.debug(f'max {res0.max()} res0:\n {res0}')

    input_t.tofile(input_path)
    res.tofile(res_path)
    logging.debug(f'res:\n{res[0, 0, 0, :]}')


def test_transpose_abcd_bacd(b, n, s, d, output_dir: Path):
    qkv_shape = [b, n, s, d]
    logging.debug(f'shape --------> b {b} n {n} s {s} d {d} dir {output_dir}\n')
    input_path = Path(output_dir, 'input.bin')
    res_path = Path(output_dir, 'res.bin')
    input_t = np.arange(0, b * n * s * d, 1).reshape(qkv_shape).astype(dtype_f32)
    logging.debug(f'input:\n{input_t[0, 0, 0, :]}')
    res = input_t.transpose(1, 0, 2, 3)

    # res0 = input_t - res
    # logging.debug(f'max {res0.max()} res0:\n {res0}')

    input_t.tofile(input_path)
    res.tofile(res_path)
    logging.debug(f'res:\n{res[0, 0, 0, :]}')


def test_transpose_debug_bnsd_bsnd(b, n, s, d):
    test_transpose_bnsd_bsnd(b, n, s, d, Path('./'))


def test_transpose_debug_abc_bac(bs, n, d):
    test_transpose_abc_bac(bs, n, d, Path('./'))


def test_transpose_debug_bnsd2_bns2_d_small(b, n, s, d):
    test_transpose_bnsd2_bns2_d_small(b, n, s, d, Path('./'))


def test_transpose_debug_bnds_bnsd(b, n, s, d):
    test_transpose_bnds_bnsd(b, n, s, d, Path('./'))


def test_transpose_add_debug_bnsd_bsnd(b, n, s, d):
    test_transpose_add_bnsd_bsnd(b, n, s, d, Path('./'))


def test_transpose_abcd_abdc(x, bs, n, d, output_dir: Path):
    qkv_shape = [x, bs, n, d]
    logging.debug(f'ABCD->ABDC shape --------> x {x} bs {bs} n {n} d {d} dir {output_dir}\n')
    input_path = Path(output_dir, 'input.bin')
    res_path = Path(output_dir, 'res.bin')
    input_t = np.arange(0, x * bs * n * d, 1).reshape(qkv_shape).astype(dtype_f32)
    res = input_t.transpose(0, 1, 3, 2)
    logging.debug("=== input shape:", input_t.shape, "res shape:", res.shape)
    input_t.tofile(input_path)
    res.tofile(res_path)

def test_transpose_ab_ba(a, b, output_dir: Path):
    qkv_shape = [a,b]
    logging.debug(f'AB -> BA shape --------> a {a} b {b} dir {output_dir}\n')
    input_path = Path(output_dir, 'input.bin')
    res_path = Path(output_dir, 'res.bin')
    input_t = np.arange(0, a*b, 1).reshape(qkv_shape).astype(dtype_f32)
    res = input_t.transpose(1,0)
    logging.debug("=== input shape:", input_t.shape, "res shape:", res.shape)
    input_t.tofile(input_path)
    res.tofile(res_path)


@GoldenRegister.reg_golden_func(
    case_names=[
        "TransposeTest.TestTranspose_ROPE_5D",
        "TransposeTest.TestTranspose_MLA_3D_0",
        "TransposeTest.TestTranspose_MLA_3D_1",
        "TransposeTest.TestTranspose_MLA_4D_0",
        "TransposeTest.TestTranspose_MLA_4D_1",
        "TransposeTest.TestTranspose_MLA_4D_2",
        "TransposeTest.TestTranspose_MLA_4D_3",
        "TransposeTest.TestTranspose_MLA_4D_4",
        "TransposeTest.TestTranspose_MLA_4D_5",
        "TransposeTest.TestTranspose_MLA_4D_50",
        "TransposeTest.TestTranspose_MLA_4D_6",
        "TransposeTest.TestTranspose_MLA_4D_7",
        "TransposeTest.TestTranspose_MLA_3D_2",
        "TransposeTest.TestTranspose_AB_BA",
        "TransposeTest.TestTranspose_ABCD_ABDC_1_2_16_31",
        "TransposeTest.Test_Datamove_Nonalign_Dim4",
        "TransposeTest.Test_Datamove_Nonalign_Dim3",
        "TransposeTest.Test_AB_BA_32_768",
        "TransposeTest.Test_AB_BA_768_32",
        "TransposeTest.Test_AB_BA_128_511",
        "TransposeTest.Test_AB_BA_128_255",
        "TransposeTest.Test_AB_BA_128_63",
        "TransposeTest.Test_AB_BA_1_128_511",
        "TransposeTest.Test_AB_BA_1_128_255",
        "TransposeTest.Test_AB_BA_1_128_63",
        "TransposeTest.TestTranspose_abcd_bacd_2_128_3_32"
    ]
)
def get_transpose_golden(case_name: str, output: Path) -> bool:
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
        if case_name == "TransposeTest.TestTranspose_BNSD_BSND":
            b, n, s, d = 2, 32, 16, 16
            test_transpose_bnsd_bsnd(b, n, s, d, output)
        elif case_name == "TransposeTest.TestTranspose_ABC_BAC":
            bs, n, d = 128, 2, 128
            test_transpose_abc_bac(bs, n, d, output)
        elif case_name == "TransposeTest.TestTranspose_BNSD2_BNS2D_small":
            b, n, s, d = 2, 4, 32, 64
            test_transpose_bnsd2_bns2_d_small(b, n, s, d, output)
        elif case_name == "TransposeTest.TestTranspose_BNDS_BNSD":
            b, n, s, d = 2, 32, 32, 32
            test_transpose_bnds_bnsd(b, n, s, d, output)
        elif case_name == "TransposeTest.TestTranspose_AB_BA":
            b, n, s, d = 1, 1, 32, 64
            test_transpose_bnds_bnsd(b, n, s, d, output)
        elif case_name == "TransposeTest.TestTranspose_ROPE_5D":
            b, n, s, d = 4, 64, 1, 64
            test_transpose_bnsd2_bns2_d_small(b, n, s, d, output)
        elif case_name == "TransposeTest.TestTranspose_MLA_3D_0":
            bs, n, d = 32, 32, 64
            test_transpose_abc_bac(bs, n, d, output)
        elif case_name == "TransposeTest.TestTranspose_MLA_3D_1":
            bs, n, d = 32, 32, 512
            test_transpose_abc_bac(bs, n, d, output)
        elif case_name == "TransposeTest.TestTranspose_MLA_4D_0":
            b, n, s, d = 32, 1, 32, 512
            test_transpose_bnsd_bsnd(b, n, s, d, output)
        elif case_name == "TransposeTest.TestTranspose_MLA_4D_1":
            b, n, s, d = 32, 1, 32, 64
            test_transpose_bnsd_bsnd(b, n, s, d, output)
        elif case_name == "TransposeTest.TestTranspose_MLA_4D_2":
            b, n, s, d = 32, 1, 32, 64
            test_transpose_bnsd_bsnd(b, n, s, d, output)
        elif case_name == "TransposeTest.TestTranspose_MLA_4D_3":
            b, n, s, d = 32, 1, 1, 64
            test_transpose_add_bnsd_bsnd(b, n, s, d, output)
        elif case_name == "TransposeTest.TestTranspose_MLA_4D_4":
            b, n, s, d = 32, 1, 1, 512
            test_transpose_add_bnsd_bsnd(b, n, s, d, output)
        elif case_name == "TransposeTest.TestTranspose_MLA_4D_5":
            b, n, s, d = 32, 1, 256, 512 + 64
            test_transpose_bnds_bnsd(b, n, s, d, output)
        elif case_name == "TransposeTest.TestTranspose_MLA_4D_50":
            b, n, s, d = 1, 1, 128, 128
            test_transpose_bnds_bnsd(b, n, s, d, output)
        elif case_name == "TransposeTest.TestTranspose_MLA_4D_6":
            b, n, s, d = 32, 32, 1, 512
            test_transpose_bnsd_bsnd(b, n, s, d, output)
        elif case_name == "TransposeTest.TestTranspose_MLA_4D_7":
            b, n, s, d = 32, 128, 1, 512
            test_transpose_abcd_bacd(b, n, s, d, output)
        elif case_name == "TransposeTest.TestTranspose_MLA_3D_2":
            bs, n, d = 32, 32, 128
            test_transpose_abc_bac(bs, n, d, output)
        elif case_name == "TransposeTest.Test_Datamove_Nonalign_Dim4":
            b, n, s, d = 32, 1, 32, 437
            test_transpose_bnsd_bsnd(b, n, s, d, output)
        elif case_name == "TransposeTest.Test_Datamove_Nonalign_Dim3":
            n, s, d = 1, 32, 437
            test_transpose_abc_bac(n, s, d, output)
        elif case_name == "TransposeTest.TestTranspose_ABCD_ABDC_1_2_16_31":
            b, s, d = 2, 16, 31
            test_transpose_abcd_abdc(1, b, s, d, output)
        elif case_name == "TransposeTest.Test_AB_BA_32_768":
            a,b = 32, 768
            test_transpose_ab_ba(a,b, output)
        elif case_name == "TransposeTest.Test_AB_BA_768_32":
            a,b = 768, 32
            test_transpose_ab_ba(a,b, output)
        elif case_name == "TransposeTest.Test_AB_BA_128_511":
            a,b = 128, 511
            test_transpose_ab_ba(a,b, output)
        elif case_name == "TransposeTest.Test_AB_BA_128_255":
            a,b = 128, 255
            test_transpose_ab_ba(a,b, output)
        elif case_name == "TransposeTest.Test_AB_BA_128_63":
            a,b = 128, 63
            test_transpose_ab_ba(a,b, output)
        elif case_name == "TransposeTest.Test_AB_BA_1_128_511":
            a,b = 128, 511
            test_transpose_ab_ba(a,b, output)
        elif case_name == "TransposeTest.Test_AB_BA_1_128_255":
            a,b = 128, 255
            test_transpose_ab_ba(a,b, output)
        elif case_name == "TransposeTest.Test_AB_BA_1_128_63":
            a,b = 128, 63
            test_transpose_ab_ba(a,b, output)
        elif case_name == "TransposeTest.TestTranspose_abcd_bacd_2_128_3_32":
            b, n, s, d = 2, 128,3, 32
            test_transpose_abcd_bacd(b, n, s, d, output)
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
        "TransposeTest.TestTranspose_BNSD_BSND",
    ]
    # 函数调用
    ret: bool = True
    for cs in case_name_list:
        output: Path = Path(g_src_root, "build/output/bin/golden", cs).resolve()
        output.mkdir(parents=True, exist_ok=True)
        ret = get_transpose_golden(case_name=cs, output=output)
    return ret


if __name__ == "__main__":
    exit(0 if main() else 1)
