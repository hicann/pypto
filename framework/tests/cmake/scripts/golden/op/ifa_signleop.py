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
"""ifa op 相关用例 Golden 生成逻辑.

本脚本有 2 种执行模式:
1. CI批跑时, 由 cmake/scripts/golden_ctrl.py 调用, 为避免日志过多, 此时 logging 级别为 logging.INFO;
2. 单独调试时, 本脚本单独被调用, 此时 logging 级别为 logging.DEBUG;
"""

import logging
from pathlib import Path
import sys
from typing import List

from ml_dtypes import bfloat16
import numpy as np

if __name__ == "__main__":
    """ 单独调试时配置 """
    # 日志级别
    logging.basicConfig(
        format='%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s: %(message)s', level=logging.DEBUG
    )
    # 系统 import 路径
    g_src_root: Path = Path(Path(__file__).parent, "../../../../../").resolve()
    logging.debug("SrcRoot: %s", g_src_root)
    g_ctrl_path: Path = Path(g_src_root, "cmake/scripts")
    if str(g_ctrl_path) not in sys.path:
        sys.path.append(str(g_ctrl_path))
    # 单独调试 import 失败, 需确认上文中 '系统 import 路径' 配置正确
    from golden_register import GoldenRegister
else:
    from golden_register import GoldenRegister


def dump_file(data_pool, data_path, type_str):
    if type_str.lower() == 'fp16':
        np.array(data_pool).astype(np.float16).tofile(data_path)
    elif type_str.lower() == 'fp32':
        np.array(data_pool).astype(np.float32).tofile(data_path)
    elif type_str.lower() == 'fp64':
        np.array(data_pool).astype(np.float64).tofile(data_path)
    elif type_str.lower() == 'int8':
        np.array(data_pool).astype(np.int8).tofile(data_path)
    elif type_str.lower() == 'int16':
        np.array(data_pool).astype(np.int16).tofile(data_path)
    elif type_str.lower() == 'int32':
        np.array(data_pool).astype(np.int32).tofile(data_path)
    elif type_str.lower() == 'int64':
        np.array(data_pool).astype(np.int64).tofile(data_path)
    elif type_str.lower() == 'uint8':
        np.array(data_pool).astype(np.uint8).tofile(data_path)
    elif type_str.lower() == 'uint16':
        np.array(data_pool).astype(np.uint16).tofile(data_path)
    elif type_str.lower() == 'uint32':
        np.array(data_pool).astype(np.uint32).tofile(data_path)
    elif type_str.lower() == 'uint64':
        np.array(data_pool).astype(np.uint64).tofile(data_path)
    elif type_str.lower() == 'complex64':
        np.array(data_pool).astype(np.complex64).tofile(data_path)
    elif type_str.lower() == 'complex128':
        np.array(data_pool).astype(np.complex128).tofile(data_path)
    elif type_str.lower() == 'bool':
        np.array(data_pool).astype(np.bool_).tofile(data_path)
    elif type_str.lower() == 'bf16':
        np.array(data_pool).astype(bfloat16).tofile(data_path)


def gen_scalardivs_golden(batch, sq, d, scalar, actual_seq, reverse_operand, output_dir: Path):
    dtype = np.float32

    shape_q = [batch * sq, d]
    out_shape = [batch * sq, d]
    q = np.ones(shape_q).astype(dtype)
    out = np.zeros(out_shape).astype(dtype)
    logging.debug(f'shape --------> batch {batch} sq {sq} dir {output_dir}\n')

    if reverse_operand:
        res = scalar / q
    else:
        res = q / scalar
    for bid in range(batch):
        seq = actual_seq[bid]
        out[bid * sq:bid * sq + seq, :] = res[bid * sq:bid * sq + seq, :]

    dump_file(q, Path(output_dir, 'q.bin'), "fp32")
    dump_file(out, Path(output_dir, 'out.bin'), "fp32")
    dump_file(actual_seq, Path(output_dir, 'actual_seq_len.bin'), "int32")


@GoldenRegister.reg_golden_func(
    case_names=[
        # ifa op
        "OnBoardIFATest.test_32_128_sub_32_1",
        "OnBoardIFATest.test_32_1_sub_32_1",
        "OnBoardIFATest.test_32_512_add_32_1",
        "OnBoardIFATest.test_32_1_mul_32_1",
        "OnBoardIFATest.test_32_512_mul_32_1",
        "OnBoardIFATest.test_32_128_tileop_exp",
        "OnBoardIFATest.test_32_1_tileop_exp",
        "OnBoardIFATest.test_32_1_maximum",
        "OnBoardIFATest.test_32_1_tileop_log1p",
        "OnBoardIFATest.test_32_1_reciprocal",
        "OnBoardIFATest.test_operation_32_128_row_max_single",
        "OnBoardIFATest.test_operation_32_128_row_sum_single",
        "OnBoardIFATest.test_concat_32_512_32_64",
        "OnBoardIFATest.test_concat_32_tensor",
        "DynamicBinTest.testDynMulsUnalign",
        "DynamicBinTest.TestDynamicAddUnalign",
        "DynamicBinTest.testScalarDivsUnalign",
        "DynamicBrcTest.TestDynamicMulBrcUnalign",
    ]
)
def gen_ifa_op_golden(case_name: str, output: Path) -> bool:
    if case_name == "DynamicBrcTest.TestDynamicMulBrcUnalign":
        return True
    elif case_name == "DynamicBinTest.testDynMulsUnalign":
        return True
    elif case_name == "DynamicBinTest.testScalarDivsUnalign":
        batch = 1
        sq = 128
        d = 64
        scalar = 1
        actual_seq = [100] * batch
        gen_scalardivs_golden(batch, sq, d, scalar, actual_seq, 0, output)
    elif case_name == "OnBoardIFATest.test_32_128_sub_32_1":
        dtype = np.float32
        shape_x = [32, 128]
        shape_y = [32, 1]
        x_path = Path(output, 'x.bin')
        y_path = Path(output, 'y.bin')
        o_path = Path(output, 'res.bin')
        complete = x_path.exists() and y_path.exists() and o_path.exists()
        if complete:
            logging.debug("Case(%s), Golden complete.", case_name)
            return True
        else:
            x = np.random.uniform(-1, 1, shape_x).astype(dtype)
            x.tofile(x_path)
            y = np.random.uniform(-1, 1, shape_y).astype(dtype)
            y.tofile(y_path)
            x = x - y
            x.tofile(o_path)
            return True
    elif case_name == "OnBoardIFATest.test_32_1_sub_32_1":
        dtype = np.float32
        shape_x = [32, 1]
        shape_y = [32, 1]
        x_path = Path(output, 'x.bin')
        y_path = Path(output, 'y.bin')
        o_path = Path(output, 'res.bin')
        complete = x_path.exists() and y_path.exists() and o_path.exists()
        if complete:
            logging.debug("Case(%s), Golden complete.", case_name)
            return True
        else:
            x = np.random.uniform(-1, 1, shape_x).astype(dtype)
            x.tofile(x_path)
            y = np.random.uniform(-1, 1, shape_y).astype(dtype)
            y.tofile(y_path)
            x = x - y
            x.tofile(o_path)
            return True
    elif case_name == "OnBoardIFATest.test_32_512_add_32_1":
        dtype = np.float32
        shape_x = [32, 512]
        shape_y = [32, 1]
        x_path = Path(output, 'x.bin')
        y_path = Path(output, 'y.bin')
        o_path = Path(output, 'res.bin')
        complete = x_path.exists() and y_path.exists() and o_path.exists()
        if complete:
            logging.debug("Case(%s), Golden complete.", case_name)
            return True
        else:
            x = np.random.uniform(-1, 1, shape_x).astype(dtype)
            x.tofile(x_path)
            y = np.random.uniform(-1, 1, shape_y).astype(dtype)
            y.tofile(y_path)
            x = x + y
            x.tofile(o_path)
            return True
    elif case_name == "OnBoardIFATest.test_32_1_mul_32_1":
        dtype = np.float32
        shape_x = [32, 1]
        shape_y = [32, 1]
        x_path = Path(output, 'x.bin')
        y_path = Path(output, 'y.bin')
        o_path = Path(output, 'res.bin')
        complete = x_path.exists() and y_path.exists() and o_path.exists()
        if complete:
            logging.debug("Case(%s), Golden complete.", case_name)
            return True
        else:
            x = np.random.uniform(-1, 1, shape_x).astype(dtype)
            x.tofile(x_path)
            y = np.random.uniform(-1, 1, shape_y).astype(dtype)
            y.tofile(y_path)
            x = x * y
            x.tofile(o_path)
            return True
    elif case_name == "OnBoardIFATest.test_32_512_mul_32_1":
        dtype = np.float32
        shape_x = [32, 512]
        shape_y = [32, 512]
        x_path = Path(output, 'x.bin')
        y_path = Path(output, 'y.bin')
        o_path = Path(output, 'res.bin')
        complete = x_path.exists() and y_path.exists() and o_path.exists()
        if complete:
            logging.debug("Case(%s), Golden complete.", case_name)
            return True
        else:
            x = np.random.uniform(-1, 1, shape_x).astype(dtype)
            x.tofile(x_path)
            y = np.random.uniform(-1, 1, shape_y).astype(dtype)
            y.tofile(y_path)
            y_sum = y.sum(axis=-1, keepdims=True)
            x = x * y_sum
            x.tofile(o_path)
            return True
    elif case_name == "OnBoardIFATest.test_32_128_tileop_exp":
        dtype = np.float32
        shape = [32, 128]
        x_path = Path(output, 'x.bin')
        y_path = Path(output, 'res.bin')
        complete = x_path.exists() and y_path.exists()
        if complete:
            logging.debug("Case(%s), Golden complete.", case_name)
            return True
        else:
            x = np.random.uniform(-1, 1, shape).astype(dtype)
            x.tofile(x_path)
            y = np.exp(x)
            y.tofile(y_path)
            return True
    elif case_name == "OnBoardIFATest.test_32_1_tileop_exp":
        dtype = np.float32
        shape = [32, 1]
        x_path = Path(output, 'x.bin')
        y_path = Path(output, 'res.bin')
        complete = x_path.exists() and y_path.exists()
        if complete:
            logging.debug("Case(%s), Golden complete.", case_name)
            return True
        else:
            x = np.random.uniform(-1, 1, shape).astype(dtype)
            x.tofile(x_path)
            y = np.exp(x)
            y.tofile(y_path)
            return True
    elif case_name == "OnBoardIFATest.test_32_1_maximum":
        dtype = np.float32
        shape_x = [32, 1]
        shape_y = [32, 1]
        x_path = Path(output, 'x.bin')
        y_path = Path(output, 'y.bin')
        o_path = Path(output, 'res.bin')
        complete = x_path.exists() and y_path.exists() and o_path.exists()
        if complete:
            logging.debug("Case(%s), Golden complete.", case_name)
            return True
        else:
            x = np.random.uniform(-1, 1, shape_x).astype(dtype)
            x.tofile(x_path)
            y = np.random.uniform(-1, 1, shape_y).astype(dtype)
            y.tofile(y_path)
            x = np.maximum(x, y)
            x.tofile(o_path)
            return True
    elif case_name == "OnBoardIFATest.test_32_1_reciprocal":
        dtype = np.float32
        shape = [32, 1]
        x_path = Path(output, 'x.bin')
        o_path = Path(output, 'res.bin')
        complete = x_path.exists() and o_path.exists()
        if complete:
            logging.debug("Case(%s), Golden complete.", case_name)
            return True
        else:
            x = np.random.uniform(-1, 1, shape).astype(dtype)
            x.tofile(x_path)
            x = np.reciprocal(x)
            x.tofile(o_path)
            return True
    elif case_name == "OnBoardIFATest.test_32_1_tileop_log1p":
        dtype = np.float32
        shape = [32, 1]
        x_path = Path(output, 'x.bin')
        o_path = Path(output, 'res.bin')
        complete = x_path.exists() and o_path.exists()
        if complete:
            logging.debug("Case(%s), Golden complete.", case_name)
            return True
        else:
            x = np.random.uniform(-1, 1, shape).astype(dtype)
            x.tofile(x_path)
            x = np.log1p(x)
            x.tofile(o_path)
            return True
    elif case_name == "OnBoardIFATest.test_operation_32_128_row_max_single":
        dtype = np.float32
        shape = [32, 128]
        x_path = Path(output, 'x.bin')
        o_path = Path(output, 'res.bin')
        complete = x_path.exists() and o_path.exists()
        if complete:
            logging.debug("Case(%s), Golden complete.", case_name)
            return True
        else:
            x = np.random.uniform(-1, 1, shape).astype(dtype)
            x.tofile(x_path)
            x_max = x.max(axis=-1, keepdims=True)
            x_max.tofile(o_path)
            return True
    elif case_name == "OnBoardIFATest.test_operation_32_128_row_sum_single":
        dtype = np.float32
        shape = [32, 128]
        x_path = Path(output, 'x.bin')
        o_path = Path(output, 'res.bin')
        complete = x_path.exists() and o_path.exists()
        if complete:
            logging.debug("Case(%s), Golden complete.", case_name)
        else:
            x = np.random.uniform(-1, 1, shape).astype(dtype)
            x.tofile(x_path)
            x_max = x.sum(axis=-1, keepdims=True)
            x_max.tofile(o_path)
            return True
    elif case_name == "OnBoardIFATest.test_concat_32_512_32_64":
        dtype = np.float32
        shape1 = [32, 512]
        shape2 = [32, 64]
        x_path = Path(output, 'x.bin')
        y_path = Path(output, 'y.bin')
        o_path = Path(output, 'res.bin')
        complete = x_path.exists() and o_path.exists()
        if complete:
            logging.debug("Case(%s), Golden complete.", case_name)
            return True
        else:
            x = np.random.uniform(-1, 1, shape1).astype(dtype)
            y = np.random.uniform(-1, 1, shape2).astype(dtype)
            x.tofile(x_path)
            y.tofile(y_path)
            out = np.concatenate((x, y), axis=-1)
            out.tofile(o_path)
            return True
    elif case_name == "OnBoardIFATest.test_concat_32_tensor":
        dtype = np.float32
        shape1 = [32, 512]
        tensor_num = 32
        x_path = []
        for i in range(tensor_num):
            x_path.append(Path(output, 'x' + str(i) + '.bin'))

        o_path = Path(output, 'res.bin')
        complete = o_path.exists()
        if False:
            logging.debug("Case(%s), Golden complete.", case_name)
            return True
        else:
            tensor = []
            for i in range(tensor_num):
                x = np.random.uniform(-1, 1, shape1).astype(dtype)
                tensor.append(x)
                x.tofile(x_path[i])
            out = np.concatenate(tensor, axis=0)
            out.tofile(o_path)
            return True
    elif case_name == "DynamicBinTest.TestDynamicAddUnalign":
        return True
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
        "DynamicPATest.test_mm_unalign",
    ]
    # 函数调用
    ret: bool = True
    for cs in case_name_list:
        output: Path = Path(g_src_root, "build/output/bin/golden", cs).resolve()
        output.mkdir(parents=True, exist_ok=True)
        ret = gen_ifa_op_golden(case_name=cs, output=output)
    return ret


if __name__ == "__main__":
    exit(0 if main() else 1)
