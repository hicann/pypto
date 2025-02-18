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

""" qkvPre子图 相关用例 Golden 生成逻辑.

本脚本有 2 种执行模式:
1. CI批跑时, 由 tests/cmake/scripts/golden_ctrl.py 调用, 为避免日志过多, 此时 logging 级别为 logging.INFO;
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
    g_ctrl_path: Path = Path(g_src_root, "tests/cmake/scripts")
    if str(g_ctrl_path) not in sys.path:
        sys.path.append(str(g_ctrl_path))
    from golden_register import GoldenRegister  # 单独调试 import 失败, 需确认上文中 '系统 import 路径' 配置正确
else:
    from golden_register import GoldenRegister

fp32 = np.float32


def rms_norm_torch(x):
    import torch
    x_dtype = x.dtype
    eps = 1e-6
    mean_coff = 1.0 / x.shape[-1]

    x_f32 = x.to(torch.float32)
    square = x_f32 * x_f32
    mean_res = square * mean_coff

    red_sum = torch.sum(mean_res, dim=-1, keepdim=True)
    red_brc = red_sum.expand_as(x)
    red_sqrt = torch.sqrt(red_brc + eps)
    res = x_f32 / red_sqrt

    if x_dtype != torch.float32:
        res = res.to(x_dtype)
    return res


def rms_norm(x):
    x_dtype = x.dtype
    eps = 1e-6
    mean_coff = 1.0 / x.shape[-1]

    x_f32 = x.astype(fp32)
    square = x_f32 * x_f32
    mean_res = square * mean_coff

    reduce_sum = np.sum(mean_res, axis=-1, keepdims=True) + eps
    reduce_sqrt = np.sqrt(reduce_sum)
    res = x_f32 / reduce_sqrt

    if x_dtype != fp32:
        res = res.astype(x_dtype)
    return res


def gen_qkv_pre_data(params, dtype, w_dtype, output_dir: Path, mm_out_dtype=fp32):
    logging.debug(f"gen_qkv_pre_data  dtype:{dtype}, w_dtype:{w_dtype}, mm_out_dtype:{mm_out_dtype}")
    b = params.get("b")
    s = params.get("s")
    s2 = params.get("s2")
    h = params.get("h")
    n = params.get("num_heads")
    q_lora_rank = params.get("q_lora_rank")
    qk_nope_head_dim = params.get("qk_nope_head_dim")
    qk_rope_head_dim = params.get("qk_rope_head_dim")
    kv_lora_rank = params.get("kv_lora_rank")
    v_head_dim = params.get("v_head_dim")
    q_head_dim = qk_nope_head_dim + qk_rope_head_dim

    x_shape = [b, s, h]
    w_qa_shape = [h, q_lora_rank]
    w_qb_shape = [q_lora_rank, n * q_head_dim]
    w_kv_a_shape = [h, kv_lora_rank + qk_rope_head_dim]
    logging.debug("x_shape is %s", x_shape)
    logging.debug("w_qa_shape is %s", w_qa_shape)
    logging.debug("w_qb_shape is %s", w_qb_shape)
    logging.debug("w_kv_a_shape is %s", w_kv_a_shape)

    x_path = Path(output_dir, 'x.bin')
    w_qa_path = Path(output_dir, 'w_qa.bin')
    w_qb_path = Path(output_dir, 'w_qb.bin')
    w_kv_a_path = Path(output_dir, 'w_kv_a.bin')
    q_path = Path(output_dir, 'q_golden.bin')
    kv_path = Path(output_dir, 'kv_golden.bin')

    x = np.random.uniform(-1, 1, x_shape).astype(dtype)
    x.tofile(x_path)
    w_qa = np.random.uniform(-0.1, 0.1, w_qa_shape).astype(w_dtype)
    w_qa.tofile(w_qa_path)
    w_qb = np.random.uniform(-0.1, 0.1, w_qb_shape).astype(w_dtype)
    w_qb.tofile(w_qb_path)
    w_kv_a = np.random.uniform(-0.1, 0.1, w_kv_a_shape).astype(w_dtype)
    w_kv_a.tofile(w_kv_a_path)

    # numpy
    logging.debug("================ numpy ================")
    x_2d = x.reshape(b * s, h)
    # [b * s, h] * [h, q_lora_rank] = [b * s, q_lora_rank]
    q_a_proj = np.matmul(x_2d.astype(fp32), w_qa.astype(fp32))  # q_a_proj
    if mm_out_dtype != fp32:
        q_a_proj = q_a_proj.astype(dtype)
    q_a_layernorm = rms_norm(q_a_proj)
    logging.debug("q_a_layernorm.shape: %s %s", q_a_layernorm.shape, q_a_layernorm.dtype)

    # [b * s, q_lora_rank] * [q_lora_rank, n * q_head_dim] = [b * s, n * q_head_dim]
    q_b_proj = np.matmul(q_a_layernorm.astype(fp32), w_qb.astype(fp32))  # q_b_proj
    if mm_out_dtype != fp32:
        q_b_proj = q_b_proj.astype(dtype)
    logging.debug("q_b_proj.shape: %s %s", q_b_proj.shape, q_b_proj.dtype)

    q_reshape = q_b_proj.reshape(b, s, n, q_head_dim)
    logging.debug("q_reshape.shape: %s %s", q_reshape.shape, q_reshape.dtype)

    # [b * s, h] * [h, kv_lora_rank + qk_rope_head_dim] = [b * s, kv_lora_rank + qk_rope_head_dim]
    logging.debug("%s, %s", type(x_2d), type(w_kv_a_shape))
    kv_a_proj = np.matmul(x_2d.astype(fp32), w_kv_a.astype(fp32))  # kv_a_proj
    if mm_out_dtype != fp32:
        kv_a_proj = kv_a_proj.astype(dtype)
    logging.debug("kv_a_proj.shape: %s %s", kv_a_proj.shape, kv_a_proj.dtype)
    kv_reshape = kv_a_proj.reshape(b, s, kv_lora_rank + qk_rope_head_dim)
    logging.debug("kv_reshape.shape: %s %s", kv_reshape.shape, kv_reshape.dtype)

    # golden: output
    q_reshape.tofile(q_path)
    kv_reshape.tofile(kv_path)

    """
    ###### torch ######
    logging.debug("================ torch ================")
    import torch
    x_torch = torch.from_numpy(x)
    w_qa_torch = torch.from_numpy(w_qa)
    w_qb_torch = torch.from_numpy(w_qb)
    w_kv_a_torch = torch.from_numpy(w_kv_a).to(torch.float32)
    x_dtype_torch = x_torch.dtype

    # [b, s, h] * [h, q_lora_rank] = [b, s, q_lora_rank]
    q_a_proj_torch = torch.matmul(x_torch.to(torch.float32), w_qa_torch.to(torch.float32))
    q_a_proj_torch = q_a_proj_torch.to(x_dtype_torch)
    q_a_layernorm_torch = rms_norm_torch(q_a_proj_torch)
    logging.debug("q_a_layernorm_torch.shape: %s %s", q_a_layernorm_torch.shape, q_a_layernorm_torch.dtype)

    # [b, s, q_lora_rank] * [q_lora_rank, n * q_head_dim] = [b, s, n * q_head_dim]
    q_b_proj_torch = np.matmul(q_a_layernorm_torch.to(torch.float32), w_qb_torch.to(torch.float32))  # q_b_proj
    q_b_proj_torch = q_b_proj_torch.to(x_dtype_torch)
    logging.debug("q_b_proj_torch.shape: %s %s", q_b_proj_torch.shape, q_b_proj_torch.dtype)

    q_reshape_torch = q_b_proj_torch.reshape(b, s, n, q_head_dim)
    logging.debug("q_reshape_torch.shape: %s %s", q_reshape_torch.shape, q_reshape_torch.dtype)

    # [b, s, h] * [h, kv_lora_rank + qk_rope_head_dim] = [b, s, kv_lora_rank + qk_rope_head_dim]
    kv_a_proj_torch = np.matmul(x_torch.to(torch.float32), w_kv_a_torch.to(torch.float32))  # kv_a_proj
    kv_a_proj_torch = kv_a_proj_torch.to(x_dtype_torch)
    logging.debug("kv_a_proj_torch.shape: %s %s", kv_a_proj_torch.shape, kv_a_proj_torch.dtype)
    kv_reshape_torch = kv_a_proj_torch.reshape(b, s, kv_lora_rank + qk_rope_head_dim)
    logging.debug("kv_reshape_torch.shape: %s %s", kv_reshape_torch.shape, kv_reshape_torch.dtype)

    result_q = np.allclose(q_reshape, q_reshape_torch.numpy(), rtol=1e-3, atol=1e-3)
    result_kv = np.allclose(kv_reshape, kv_reshape_torch.numpy(), rtol=1e-3, atol=1e-3)
    logging.debug(f"======== q, golden precise: {result_q}")
    logging.debug(f"======== kv, golden precise: {result_kv}")
    """


def gen_qkv_pre_test1(dtype, w_dtype, output_dir: Path, mm_out_dtype=fp32):
    params = {
        "b": 4,  # 32
        "s": 1,
        "s2": 256,
        "h": 256,  # 7168
        "num_heads": 2,  # 32
        "q_lora_rank": 512,  # 1536
        "qk_nope_head_dim": 128,
        "qk_rope_head_dim": 64,
        "kv_lora_rank": 512,
        "v_head_dim": 128,
    }
    gen_qkv_pre_data(params, dtype, w_dtype, output_dir, mm_out_dtype)


def gen_qkv_pre_test2(dtype, w_dtype, output_dir: Path, mm_out_dtype=fp32):
    params = {
        "b": 32,
        "s": 1,
        "s2": 256,
        "h": 256,  # 7168
        "num_heads": 2,  # 32
        "q_lora_rank": 512,  # 1536
        "qk_nope_head_dim": 128,
        "qk_rope_head_dim": 64,
        "kv_lora_rank": 512,
        "v_head_dim": 128,
    }
    gen_qkv_pre_data(params, dtype, w_dtype, output_dir, mm_out_dtype)


def gen_qkv_pre_test3(dtype, w_dtype, output_dir: Path, mm_out_dtype=fp32):
    params = {
        "b": 32,
        "s": 1,
        "s2": 256,
        "h": 256,  # 7168
        "num_heads": 32,
        "q_lora_rank": 512,  # 1536
        "qk_nope_head_dim": 128,
        "qk_rope_head_dim": 64,
        "kv_lora_rank": 512,
        "v_head_dim": 128,
    }
    gen_qkv_pre_data(params, dtype, w_dtype, output_dir, mm_out_dtype)


def gen_qkv_pre_test4(dtype, w_dtype, output_dir: Path, mm_out_dtype=fp32):
    params = {
        "b": 32,
        "s": 1,
        "s2": 256,
        "h": 256,  # 7168
        "num_heads": 32,
        "q_lora_rank": 1536,
        "qk_nope_head_dim": 128,
        "qk_rope_head_dim": 64,
        "kv_lora_rank": 512,
        "v_head_dim": 128,
    }
    gen_qkv_pre_data(params, dtype, w_dtype, output_dir, mm_out_dtype)


def gen_qkv_pre_test5(dtype, w_dtype, output_dir: Path, mm_out_dtype=fp32):
    params = {
        "b": 32,
        "s": 1,
        "s2": 256,
        "h": 1024,  # 7168
        "num_heads": 32,
        "q_lora_rank": 1536,
        "qk_nope_head_dim": 128,
        "qk_rope_head_dim": 64,
        "kv_lora_rank": 512,
        "v_head_dim": 128,
    }
    gen_qkv_pre_data(params, dtype, w_dtype, output_dir, mm_out_dtype)


def gen_qkv_pre_test_net(dtype, w_dtype, output_dir: Path, mm_out_dtype=fp32):
    params = {
        "b": 32,
        "s": 1,
        "s2": 256,
        "h": 7168,
        "num_heads": 32,
        "q_lora_rank": 1536,
        "qk_nope_head_dim": 128,
        "qk_rope_head_dim": 64,
        "kv_lora_rank": 512,
        "v_head_dim": 128,
    }
    gen_qkv_pre_data(params, dtype, w_dtype, output_dir, mm_out_dtype)


def gen_qkv_pre_test_net_b4(dtype, w_dtype, output_dir: Path, mm_out_dtype=fp32):
    params = {
        "b": 4,
        "s": 1,
        "s2": 256,
        "h": 7168,
        "num_heads": 32,
        "q_lora_rank": 1536,
        "qk_nope_head_dim": 128,
        "qk_rope_head_dim": 64,
        "kv_lora_rank": 512,
        "v_head_dim": 128,
    }
    gen_qkv_pre_data(params, dtype, w_dtype, output_dir, mm_out_dtype)


@GoldenRegister.reg_golden_func(
    case_names=[
        # qkvPre
        "QkvPreOnBoardTest.test_qkvPre_float16_4_2_1_256_256_512",
        "QkvPreOnBoardTest.test_qkvPre_float16_32_2_1_256_256_512",
        "QkvPreOnBoardTest.test_qkvPre_bfloat16_32_2_1_256_256_512",
        "QkvPreOnBoardTest.test_qkvPre_float16_32_32_1_256_256_512",
        "QkvPreOnBoardTest.test_qkvPre_float16_32_32_1_256_256_1536",
        "QkvPreOnBoardTest.test_qkvPre_float16_32_32_1_256_1024_1536",
        "QkvPreOnBoardTest.test_qkvPre_float16_32_32_1_256_7168_1536",
        "QkvPreOnBoardTest.test_qkvPre_bfloat16_32_32_1_256_7168_1536",
        "QkvPreOnBoardTest.test_qkvPre_float16_4_32_1_256_7168_1536",
        "QkvPreOnBoardTest.test_qkvPre_bfloat16_4_32_1_256_7168_1536",
        "QkvPreOnBoardTest.test_qkvPre_float16_32_2_1_256_256_512_fp32",
        "QkvPreOnBoardTest.test_qkvPre_bfloat16_32_2_1_256_256_512_fp32",
        "QkvPreOnBoardTest.test_qkvPre_bfloat16_32_32_1_256_7168_1536_fp32",
    ]
)
def gen_qkv_pre_op_date(case_name: str, output: Path) -> bool:
    x_path = Path(output, 'x.bin')
    w_qa_path = Path(output, 'w_qa.bin')
    w_qb_path = Path(output, 'w_qb.bin')
    w_kv_a_path = Path(output, 'w_kv_a.bin')
    q_path = Path(output, 'q_golden.bin')
    kv_path = Path(output, 'kv_golden.bin')

    complete = (x_path.exists() and w_qa_path.exists() and w_qb_path.exists() and w_kv_a_path.exists() and
                q_path.exists() and kv_path.exists())

    if complete:
        logging.debug("Case(%s), Golden complete.", case_name)
        return True
    else:
        if case_name == "QkvPreOnBoardTest.test_qkvPre_float16_4_2_1_256_256_512":
            gen_qkv_pre_test1(np.float16, np.float16, output, np.float16)
        elif case_name == "QkvPreOnBoardTest.test_qkvPre_float16_32_2_1_256_256_512":  # b_n_s_s2_h_q_lora_rank
            gen_qkv_pre_test2(np.float16, np.float16, output, np.float16)
        elif case_name == "QkvPreOnBoardTest.test_qkvPre_bfloat16_32_2_1_256_256_512":  # bfloat16
            gen_qkv_pre_test2(bfloat16, bfloat16, output, bfloat16)
        elif case_name == "QkvPreOnBoardTest.test_qkvPre_float16_32_32_1_256_256_512":
            gen_qkv_pre_test3(np.float16, np.float16, output, np.float16)
        elif case_name == "QkvPreOnBoardTest.test_qkvPre_float16_32_32_1_256_256_1536":
            gen_qkv_pre_test4(np.float16, np.float16, output, np.float16)
        elif case_name == "QkvPreOnBoardTest.test_qkvPre_float16_32_32_1_256_1024_1536":
            gen_qkv_pre_test5(np.float16, np.float16, output, np.float16)
        elif case_name == "QkvPreOnBoardTest.test_qkvPre_float16_32_32_1_256_7168_1536":
            gen_qkv_pre_test_net(np.float16, np.float16, output, np.float16)
        elif case_name == "QkvPreOnBoardTest.test_qkvPre_bfloat16_32_32_1_256_7168_1536":  # bfloat16
            gen_qkv_pre_test_net(bfloat16, bfloat16, output, bfloat16)
        elif case_name == "QkvPreOnBoardTest.test_qkvPre_float16_4_32_1_256_7168_1536":
            gen_qkv_pre_test_net_b4(np.float16, np.float16, output, np.float16)
        elif case_name == "QkvPreOnBoardTest.test_qkvPre_bfloat16_4_32_1_256_7168_1536":  # bfloat16
            gen_qkv_pre_test_net_b4(bfloat16, bfloat16, output, bfloat16)
        elif case_name == "QkvPreOnBoardTest.test_qkvPre_float16_32_2_1_256_256_512_fp32":  # bfloat16
            gen_qkv_pre_test2(np.float16, np.float16, output, fp32)
        elif case_name == "QkvPreOnBoardTest.test_qkvPre_bfloat16_32_2_1_256_256_512_fp32":  # bfloat16
            gen_qkv_pre_test2(bfloat16, bfloat16, output, fp32)
        elif case_name == "QkvPreOnBoardTest.test_qkvPre_bfloat16_32_32_1_256_7168_1536_fp32":  # bfloat16
            gen_qkv_pre_test_net(bfloat16, bfloat16, output, fp32)
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
        "QkvPreOnBoardTest.test_qkvPre_float16_4_2_1_256_256_512",
    ]
    # 函数调用
    ret: bool = True
    for cs in case_name_list:
        output: Path = Path(g_src_root, "build/output/bin/golden", cs).resolve()
        output.mkdir(parents=True, exist_ok=True)
        ret = gen_qkv_pre_op_date(case_name=cs, output=output)
    return ret


if __name__ == "__main__":
    exit(0 if main() else 1)
