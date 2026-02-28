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
""" ifa op 相关用例 Golden 生成逻辑.

本脚本有 2 种执行模式:
1. CI批跑时, 由 cmake/scripts/golden_ctrl.py 调用, 为避免日志过多, 此时 logging 级别为 logging.INFO;
2. 单独调试时, 本脚本单独被调用, 此时 logging 级别为 logging.DEBUG;
"""
import sys
import logging
from pathlib import Path
from typing import List

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


def gen_uniform_data(data_shape, min_value, max_value, dtype):
    if min_value == 0 and max_value == 0:
        return np.zeros(data_shape, dtype=dtype)
    if dtype == np.bool_:
        return np.random.choice([True, False], size=data_shape)
    return np.random.uniform(low=min_value, high=max_value, size=data_shape).astype(
        dtype
    )


def gen_unalign_mm2_golden(batch, nq, block_size, d, actual_seq, output_dir):
    dtype = bfloat16
    sq = 1
    nkv = 1
    shape_qk = [batch * nq * sq, nkv * block_size]
    shape_v = [batch * nkv * block_size, d]
    out_shape = [batch * nq * sq, d]

    # gen data
    qk = np.zeros(shape_qk).astype(dtype)
    v = np.zeros(shape_v).astype(dtype)
    out = np.zeros(out_shape).astype(dtype)

    for bid in range(batch):
        seq = actual_seq[bid]
        signle_actual_shape_qk = [nq * sq, nkv * seq]
        signle_actual_shape_v = [nkv * seq, d]

        qk_i = gen_uniform_data(signle_actual_shape_qk, -1, 1, dtype)
        v_i = gen_uniform_data(signle_actual_shape_v, -1, 1, dtype)

        qkv_bmm_signle = np.matmul(qk_i, v_i, dtype=np.float32)

        qk[bid * nq * sq: (bid + 1) * nq * sq, 0: nkv * seq] = qk_i

        v[bid * nq * sq: bid * nq * sq + nkv * seq, :] = v_i

        out[bid * nq * sq: (bid + 1) * nq * sq, :] = qkv_bmm_signle  # assemble
    # dump golden file
    dump_file(qk, Path(output_dir, 'qk.bin'), "bf16")
    dump_file(v, Path(output_dir, 'v.bin'), "bf16")
    dump_file(out, Path(output_dir, 'out.bin'), "fp32")
    dump_file(actual_seq, Path(output_dir, 'actual_seq_len.bin'), "int32")


def gen_unalign_mm_golden(batch, nq, block_size, dR, dN, actual_seq, output_dir):
    dtype = bfloat16
    sq = 1
    nk = 1
    d = dR + dN

    shape_q = [batch * nq * sq, dR + dN]

    shape_k = [batch * nk * block_size, dR + dN]

    out_shape = [batch * nq * sq, nk * block_size]

    # gen data
    q = gen_uniform_data(shape_q, -1, 1, dtype)
    k = np.zeros(shape_k).astype(dtype)
    out = np.zeros(out_shape).astype(dtype)

    for bid in range(batch):
        seq = actual_seq[bid]
        ki = gen_uniform_data([nk * seq, d], -1, 1, dtype)
        begin = bid * nk * block_size
        k[begin: begin + seq * nk, :] = ki

        q_cur = q[bid * nq * sq: (bid + 1) * nq * sq, :]
        # k_cur = k[begin : begin + block_size * nk, :]

        qk_bmm_signle = np.matmul(q_cur, ki.transpose(1, 0), dtype=np.float32)

        out[bid * nq * sq: (bid + 1) * nq * sq,
            0: seq] = qk_bmm_signle  # assemble

    q_nope = q[:, 0: dN]
    q_rope = q[:, dN:]
    k_nope = k[:, 0: dN]
    k_rope = k[:, dN:]

    # dump golden file
    dump_file(q, Path(output_dir, 'q.bin'), "bf16")
    dump_file(k, Path(output_dir, 'k.bin'), "bf16")
    dump_file(out, Path(output_dir, 'out.bin'), "fp32")
    dump_file(actual_seq, Path(output_dir, 'actual_seq_len.bin'), "int32")

    dump_file(q_nope, Path(output_dir, 'q_nope.bin'), "bf16")
    dump_file(q_rope, Path(output_dir, 'q_rope.bin'), "bf16")
    dump_file(k_nope, Path(output_dir, 'k_nope.bin'), "bf16")
    dump_file(k_rope, Path(output_dir, 'k_rope.bin'), "bf16")


def gen_unalign_reduce_golden(batch, nTile, block_size, actual_seq, output_dir, reduce_axis=-1, reduce_type="Max"):
    dtype = np.float32

    shape_q = [batch * nTile, block_size]
    out_shape = [batch * nTile, 1]

    q = np.ones(shape_q).astype(dtype)
    out = np.ones(out_shape).astype(dtype)

    for bid in range(batch):
        seq = actual_seq[bid]
        q_tmp = gen_uniform_data([nTile, seq], -5, -2, dtype)
        # out_tmp = q_tmp.max(axis=-1, keepdims=True)
        if reduce_type.lower() == "max":
            logging.debug(
                "======================= max golden =======================")
            out_tmp = q_tmp.max(axis=reduce_axis, keepdims=True)
        elif reduce_type.lower() == "sum":
            logging.debug(
                "======================= sum golden =======================")
            out_tmp = q_tmp.sum(axis=reduce_axis, keepdims=True)
        else:
            raise KeyError(f"Unknown Reduce Type {reduce_type}")
        q[bid * nTile: (bid + 1) * nTile, 0: seq] = q_tmp
        out[bid * nTile: (bid + 1) * nTile, :] = out_tmp
    # print(q)
    # print(out)
    dump_file(q, Path(output_dir, 'q.bin'), "fp32")
    dump_file(out, Path(output_dir, 'out.bin'), "fp32")
    dump_file(actual_seq, Path(output_dir, 'actual_seq_len.bin'), "int32")


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
        out[bid * sq: bid * sq + seq, :] = res[bid * sq: bid * sq + seq, :]

    dump_file(q, Path(output_dir, 'q.bin'), "fp32")
    dump_file(out, Path(output_dir, 'out.bin'), "fp32")
    dump_file(actual_seq, Path(output_dir, 'actual_seq_len.bin'), "int32")


def gen_gather_data_2d(
    axis, b, sq, s1, d, dtype, indices_dtype, output_dir: Path, valid_len=None
):
    if valid_len is None:
        valid_len = sq

    shape_params = [s1, d]
    shape_indices = [b * sq]
    shape_res = [b * sq, d]
    logging.debug("shape params is %s", shape_params)
    logging.debug("shape indices is %s", shape_indices)
    logging.debug("shape res is %s", shape_res)
    x_path = Path(output_dir, "x.bin")
    indices_path = Path(output_dir, "indices.bin")
    y_path = Path(output_dir, "out.bin")

    x = np.random.uniform(-10, 10, shape_params).astype(dtype)
    x.tofile(x_path)
    indices = np.zeros(shape_indices, dtype=indices_dtype)
    for bidx in range(b):
        batch_start = bidx * sq
        valid_indices = np.random.randint(0, shape_params[axis], size=valid_len).astype(
            indices_dtype
        )
        indices[batch_start: batch_start + valid_len] = valid_indices
    indices.tofile(indices_path)

    # numpy
    y = np.zeros(shape_res).astype(dtype)
    for bidx in range(b):
        for _s in range(valid_len):
            index = indices[bidx * sq + _s]
            y[bidx * sq + _s][:] = x[index][:]
    y.tofile(y_path)


def gen_gather_data_3d(
    axis, b, sq, s1, d, s2, dtype, indices_dtype, output_dir: Path, valid_len=None
):
    if valid_len is None:
        valid_len = sq

    shape_params = [s1, d]
    shape_indices = [b * sq, s2]
    shape_res = [b * sq, s2, d]
    logging.debug("shape params is %s", shape_params)
    logging.debug("shape indices is %s", shape_indices)
    logging.debug("shape res is %s", shape_res)
    x_path = Path(output_dir, "x.bin")
    indices_path = Path(output_dir, "indices.bin")
    y_path = Path(output_dir, "out.bin")

    x = np.random.uniform(-10, 10, shape_params).astype(dtype)
    x.tofile(x_path)
    indices = np.zeros(shape_indices, dtype=indices_dtype)
    for bidx in range(b):
        batch_start = bidx * sq
        for seq_idx in range(valid_len):
            valid_indices = np.random.randint(
                0, shape_params[axis], size=s2).astype(indices_dtype)
            indices[batch_start + seq_idx, :] = valid_indices
    indices.tofile(indices_path)

    # numpy
    y = np.zeros(shape_res).astype(dtype)
    for bidx in range(b):
        for seq_idx in range(valid_len):
            for s2_idx in range(s2):
                index = indices[bidx * sq + seq_idx, s2_idx]
                y[bidx * sq + seq_idx][s2_idx][:] = x[index][:]
    y.tofile(y_path)


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
        "DynamicUnalignTest.test_unary_unalign",
        "DynamicUnalignTest.test_mm_unalign",
        "DynamicUnalignTest.test_mm2_unalign",
        "DynamicUnalignTest.test_rowmaxsingle_unalign",
        "DynamicUnalignTest.test_rowsumsingle_unalign",
        "DynamicExpandTest.TestDynamicExpandUnalign",
        "DynamicBinTest.testDynMulsUnalign",
        "DynamicCastTest.testDynCastUnalign",
        "DynamicTransposeTest.TestDynamicVnchwconv",
        "DynamicBinTest.TestDynamicAddUnalign",
        "DynamicBinTest.testScalarDivsUnalign",
        "DynamicGatherTest.TestDynamicGatherDim2",
        "DynamicGatherTest.TestDynamicGatherDim3",
        "DynamicDatamoveTest.TestDynamicDatamove",
        "DynamicBrcTest.TestDynamicMulBrcUnalign",

    ]
)
def gen_ifa_op_golden(case_name: str, output: Path) -> bool:
    if case_name == "DynamicTransposeTest.TestDynamicVnchwconv":
        dtype = np.float32
        b = 1
        sq = 128
        d = 64
        shape_q = [b * sq, d]
        shape_out = [b * d, sq]

        q = np.zeros(shape_q).astype(dtype)
        out = np.zeros(shape_out).astype(dtype)
        for bid in range(b):
            seq = 100
            qi = gen_uniform_data([seq, d], -1, 1, dtype)
            begin = bid * sq
            q[begin: begin + seq, :] = qi
            temp = qi.transpose(1, 0)
            out[bid * d: (bid + 1) * d, 0: seq] = temp
        input_path = Path(output, 'q.bin')
        output_path = Path(output, 'out.bin')
        q.tofile(input_path)
        out.tofile(output_path)
    elif case_name == "DynamicGatherTest.TestDynamicGatherDim2":
        b, sq, s1, d = 2, 128, 128, 64
        axis = 0
        dtype = np.float32
        indices_dtype = np.int32
        gen_gather_data_2d(axis, b, sq, s1, d, dtype,
                           indices_dtype, output, 100)
    elif case_name == "DynamicGatherTest.TestDynamicGatherDim3":
        b, sq, s1, d, s2 = 2, 32, 32, 64, 2
        axis = 0
        dtype = np.float32
        indices_dtype = np.int32
        gen_gather_data_3d(axis, b, sq, s1, d, s2, dtype,
                           indices_dtype, output, 30)
    elif case_name == "DynamicDatamoveTest.TestDynamicDatamove":
        dtype = np.float32
        b = 1
        n = 1
        sq = 32
        d = 64
        shape_q = [b * n, sq, d]
        shape_out = [b * sq, n, d]

        q = np.zeros(shape_q).astype(dtype)
        out = np.zeros(shape_out).astype(dtype)
        for bid in range(b):
            for nid in range(n):
                seq = 30
                qi = gen_uniform_data([seq, d], -1, 1, dtype)
                q[bid * n + nid, 0:seq, :] = qi
                temp = qi.transpose(1, 0)
                for i in range(seq):
                    out[bid * sq + i, nid, :] = temp[:, i]
        input_path = Path(output, "q.bin")
        output_path = Path(output, "out.bin")
        q.tofile(input_path)
        out.tofile(output_path)
    elif case_name == "DynamicBrcTest.TestDynamicMulBrcUnalign":
        return True
    elif case_name == "DynamicExpandTest.TestDynamicExpandUnalign":
        return True
    elif case_name == "DynamicCastTest.testDynCastUnalign":
        return True
    elif case_name == "DynamicBinTest.testDynMulsUnalign":
        return True
    elif case_name == "DynamicUnalignTest.test_unary_unalign":
        return True
    elif case_name == "DynamicBinTest.testScalarDivsUnalign":
        batch = 1
        sq = 128
        d = 64
        scalar = 1
        actual_seq = [100] * batch
        gen_scalardivs_golden(batch, sq, d, scalar, actual_seq, 0, output)
    elif case_name == "DynamicUnalignTest.test_rowsumsingle_unalign":
        batch = 2
        nTile = 32
        block_size = 256
        actual_seq = [248] * batch
        gen_unalign_reduce_golden(
            batch, nTile, block_size, actual_seq, output, -1, reduce_type="Sum")
    elif case_name == "DynamicUnalignTest.test_mm_unalign":
        batch = 1
        nq = 32
        block_size = 256
        dR = 64
        dN = 512
        actual_seq = [248] * batch
        gen_unalign_mm_golden(batch, nq, block_size, dR,
                              dN, actual_seq, output)
    elif case_name == "DynamicUnalignTest.test_mm2_unalign":
        batch = 1
        nq = 32
        block_size = 128
        d = 64
        actual_seq = [120] * batch
        gen_unalign_mm2_golden(batch, nq, block_size, d, actual_seq, output)
    elif case_name == "DynamicUnalignTest.test_rowmaxsingle_unalign":
        batch = 1
        nTile = 32
        block_size = 256
        actual_seq = [248] * batch
        gen_unalign_reduce_golden(batch, nTile, block_size, actual_seq, output)
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
