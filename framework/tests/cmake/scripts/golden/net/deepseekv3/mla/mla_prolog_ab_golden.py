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

""" MLA_prolog 子图a+b 相关用例 Golden 生成逻辑.

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


def quant(input_t, is_pertoken: bool = True):
    input_fp32 = input_t.astype(fp32)
    abs_res = np.abs(input_fp32)
    reduce_idx = -1
    if not is_pertoken:
        reduce_idx = -2
        logging.debug("This PerChannel Quant!!")

    max_value = np.max(abs_res, axis=reduce_idx, keepdims=True)
    scale_quant = 127 / max_value
    out_fp32 = input_fp32 * scale_quant
    out_int32 = np.rint(out_fp32).astype(np.int32)
    out_fp16 = out_int32.astype(np.float16)
    out_int8 = np.trunc(out_fp16).astype(np.int8)
    scale_dequant = 1 / scale_quant

    return out_int8, scale_dequant


def gen_mla_prolog_ab_data(params, dtype, w_dtype, output_dir: Path, is_quant=False):
    logging.debug("gen_mla_prolog_ab_data  dtype:%s, w_dtype:%s", dtype, w_dtype)
    b = params.get("b")
    s = params.get("s")  # s=1
    s2 = params.get("s2")  # s2=4k
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
    w_kv_b_k_shape = [n, qk_nope_head_dim, kv_lora_rank]
    q_pe_new_shape = [b, n, s, qk_rope_head_dim]
    logging.debug("x shape is %s", x_shape)
    logging.debug("w_qa shape is %s", w_qa_shape)
    logging.debug("w_qb shape is %s", w_qb_shape)
    logging.debug("w_kv_a shape is %s", w_kv_a_shape)
    logging.debug("w_kv_b_k shape is %s", w_kv_b_k_shape)
    logging.debug("q_pe_new shape is %s", q_pe_new_shape)

    x_path = Path(output_dir, 'x.bin')
    w_qa_path = Path(output_dir, 'w_qa.bin')
    w_qb_path = Path(output_dir, 'w_qb.bin')
    w_qb_scale_path = Path(output_dir, 'w_qb_scale.bin')
    w_kv_a_path = Path(output_dir, 'w_kv_a.bin')
    w_kv_b_k_path = Path(output_dir, 'w_kv_b_k.bin')  # kv_b_proj_w_k
    q_pe_new_path = Path(output_dir, 'q_pe_new.bin')  # q_pe_new
    q_golden_path = Path(output_dir, 'q_golden.bin')
    kv_golden_path = Path(output_dir, 'kv_golden.bin')

    x = np.random.uniform(-1, 1, x_shape).astype(dtype)
    x.tofile(x_path)
    w_qa = np.random.uniform(-0.1, 0.1, w_qa_shape).astype(w_dtype)
    w_qa.tofile(w_qa_path)
    w_qb = np.random.uniform(-0.1, 0.1, w_qb_shape).astype(w_dtype)
    w_qb_quant, w_qb_scale = w_qb, w_qb
    if is_quant:
        w_qb_quant, w_qb_scale = quant(w_qb, False)
        w_qb_quant.tofile(w_qb_path)
        w_qb_scale.tofile(w_qb_scale_path)
        logging.debug("w_qb_scale shape is %s", w_qb_scale.shape)
        logging.debug(w_qb_scale)
    else:
        w_qb.tofile(w_qb_path)

    w_kv_a = np.random.uniform(-0.1, 0.1, w_kv_a_shape).astype(w_dtype)
    w_kv_a.tofile(w_kv_a_path)
    w_kv_b_k = np.random.uniform(-0.1, 0.1, w_kv_b_k_shape).astype(w_dtype)
    w_kv_b_k.tofile(w_kv_b_k_path)
    q_pe_new = np.random.uniform(-1, 1, q_pe_new_shape).astype(dtype)
    q_pe_new.tofile(q_pe_new_path)

    # numpy
    # q
    logging.debug("================ numpy ================")
    x_2d = x.reshape(b * s, h)
    # shape is: [b * s, h] * [h, q_lora_rank] = [b * s, q_lora_rank]
    q_a_proj = np.matmul(x_2d.astype(fp32), w_qa.astype(fp32))  # q_a_proj
    q_a_proj = q_a_proj.astype(dtype)
    q_a_layernorm = rms_norm(q_a_proj)
    logging.debug("q_a_layernorm.shape: %s %s", q_a_layernorm.shape, q_a_layernorm.dtype)

    # shape is: [b * s, q_lora_rank] * [q_lora_rank, n * q_head_dim] = [b * s, n * q_head_dim]
    if is_quant:
        q_a_layernorm, q_a_layernorm_scale_dequant = quant(q_a_layernorm)
        q_b_proj = np.matmul(q_a_layernorm.astype(np.int32), w_qb_quant.astype(np.int32))  # q_b_proj
        # dequant
        q_b_proj_fp32 = q_b_proj.astype(fp32)
        q_b_proj_fp32_dequant = q_b_proj_fp32 * q_a_layernorm_scale_dequant
        q_b_proj = q_b_proj_fp32_dequant * w_qb_scale
    else:
        q_b_proj = np.matmul(q_a_layernorm.astype(fp32), w_qb.astype(fp32))  # q_b_proj

    q_b_proj = q_b_proj.astype(dtype)
    logging.debug("q_b_proj.shape: %s %s", q_b_proj.shape, q_b_proj.dtype)

    q_reshape = q_b_proj.reshape(b, s, n, q_head_dim)
    logging.debug("q_reshape.shape: %s %s", q_reshape.shape, q_reshape.dtype)

    q_nope = q_reshape[:, :, :, 0:qk_nope_head_dim]  # [b, s, n, qk_nope_head_dim]

    q_nope_r = q_nope.reshape(b * s, n, qk_nope_head_dim)
    q_nope_t = q_nope_r.transpose(1, 0, 2)  # [n, b*s, qk_nope_head_dim]
    # shape is: [n, b*s, qk_nope_head_dim] * [n, qk_nope_head_dim, kv_lora_rank] = [n, b*s, kv_lora_rank]
    q_nope_new = np.matmul(q_nope_t.astype(fp32), w_kv_b_k.astype(fp32))
    q_nope_new = q_nope_new.astype(dtype)
    q_nope_new_t = q_nope_new.transpose(1, 0, 2)  # [b*s, n, kv_lora_rank]
    q_nope_new_r = q_nope_new_t.reshape(b, s, n, kv_lora_rank)  # [b, s, n, kv_lora_rank]
    q_nope_new_t2 = q_nope_new_r.transpose(0, 2, 1, 3)  # [b, n, s, kv_lora_rank]

    q_res = np.concatenate((q_nope_new_t2, q_pe_new), axis=-1)  # [b, n, s, kv_lora_rank + qk_rope_head_dim]

    """ q output """
    q_res.tofile(q_golden_path)

    # kv
    # shape is: [b * s, h] * [h, kv_lora_rank + qk_rope_head_dim] = [b * s, kv_lora_rank + qk_rope_head_dim]
    kv_a_proj = np.matmul(x_2d.astype(fp32), w_kv_a.astype(fp32))  # kv_a_proj
    kv_a_proj = kv_a_proj.astype(dtype)
    logging.debug("kv_a_proj.shape: %s %s", kv_a_proj.shape, kv_a_proj.dtype)
    kv_reshape = kv_a_proj.reshape(b, s, kv_lora_rank + qk_rope_head_dim)
    logging.debug("kv_reshape.shape: %s %s", kv_reshape.shape, kv_reshape.dtype)

    """ kv output """
    kv_reshape.tofile(kv_golden_path)


def gen_mla_prolog_ab_test1(dtype, w_dtype, output_dir: Path, is_quant=False):
    params = {
        "b": 32,
        "s": 1,
        "s2": 256,
        "h": 256,  # 7168
        "num_heads": 2,  #
        "q_lora_rank": 512,  #
        "qk_nope_head_dim": 128,
        "qk_rope_head_dim": 64,
        "kv_lora_rank": 512,
        "v_head_dim": 128,
    }
    gen_mla_prolog_ab_data(params, dtype, w_dtype, output_dir, is_quant)


def gen_mla_prolog_ab_test2(dtype, w_dtype, output_dir: Path):
    params = {
        "b": 4,
        "s": 1,
        "s2": 256,
        "h": 256,  # 7168
        "num_heads": 2,  #
        "q_lora_rank": 512,  #
        "qk_nope_head_dim": 128,
        "qk_rope_head_dim": 64,
        "kv_lora_rank": 512,
        "v_head_dim": 128,
    }
    gen_mla_prolog_ab_data(params, dtype, w_dtype, output_dir)


def gen_mla_prolog_ab_test3(dtype, w_dtype, output_dir: Path):
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
    gen_mla_prolog_ab_data(params, dtype, w_dtype, output_dir)


def gen_mla_prolog_ab_test_net1(dtype, w_dtype, output_dir: Path, is_quant=False):
    params = {
        "b": 32,
        "s": 1,
        "s2": 256,  # 4096
        "h": 7168,
        "num_heads": 32,  # 128
        "q_lora_rank": 1536,
        "qk_nope_head_dim": 128,
        "qk_rope_head_dim": 64,
        "kv_lora_rank": 512,
        "v_head_dim": 128,
    }
    gen_mla_prolog_ab_data(params, dtype, w_dtype, output_dir, is_quant)


def gen_mla_prolog_ab_test_net_b4(dtype, w_dtype, output_dir: Path):
    params = {
        "b": 4,
        "s": 1,
        "s2": 256,  # 4096
        "h": 7168,
        "num_heads": 32,  # 128
        "q_lora_rank": 1536,
        "qk_nope_head_dim": 128,
        "qk_rope_head_dim": 64,
        "kv_lora_rank": 512,
        "v_head_dim": 128,
    }
    gen_mla_prolog_ab_data(params, dtype, w_dtype, output_dir)


@GoldenRegister.reg_golden_func(
    case_names=[
        # MLA prolog ab
        "MlaPrologAbOnBoardTest.test_MlaPrologAb_float16_4_2_1_256_256_512",
        "MlaPrologAbOnBoardTest.test_MlaPrologAb_float16_32_2_1_256_256_512",
        "MlaPrologAbOnBoardTest.test_MlaPrologAb_float16_32_32_1_256_1024_1536",
        "MlaPrologAbOnBoardTest.test_MlaPrologAb_bfloat16_32_32_1_256_1024_1536",
        "MlaPrologAbOnBoardTest.test_MlaPrologAb_float16_32_32_1_256_7168_1536",
        "MlaPrologAbOnBoardTest.test_MlaPrologAb_bfloat16_32_32_1_256_7168_1536",
        "MlaPrologAbOnBoardTest.test_MlaPrologAb_bfloat16_4_32_1_256_7168_1536",
        "MlaPrologAbOnBoardTest.test_MlaPrologAb_float16_32_2_1_256_256_512_quant",
        "MlaPrologAbOnBoardTest.test_MlaPrologAb_float16_32_32_1_256_7168_1536_quant",
    ]
)
def gen_mla_prolog_ab_date(case_name: str, output: Path) -> bool:
    x_path = Path(output, 'x.bin')
    w_qa_path = Path(output, 'w_qa.bin')  # q_a_proj_w
    w_qb_path = Path(output, 'w_qb.bin')  # q_b_proj_w
    w_kv_a_path = Path(output, 'w_kv_a.bin')  # kv_a_proj_with_mqa_w
    w_kv_b_k_path = Path(output, 'w_kv_b_k.bin')  # kv_b_proj_w_k
    q_pe_new_path = Path(output, 'q_pe_new.bin')  # q_pe_new
    q_golden_path = Path(output, 'q_golden.bin')
    kv_golden_path = Path(output, 'kv_golden.bin')

    complete = (x_path.exists() and w_qa_path.exists() and w_qb_path.exists() and w_kv_a_path.exists()
                and w_kv_b_k_path.exists() and q_pe_new_path.exists() and q_golden_path.exists()
                and kv_golden_path.exists())

    if complete:
        logging.debug("Case(%s), Golden complete.", case_name)
        return True
    else:
        # b_n_s_s2_h_q_lora_rank
        if case_name == "MlaPrologAbOnBoardTest.test_MlaPrologAb_float16_32_2_1_256_256_512":
            gen_mla_prolog_ab_test1(np.float16, np.float16, output)
        elif case_name == "MlaPrologAbOnBoardTest.test_MlaPrologAb_float16_4_2_1_256_256_512":
            gen_mla_prolog_ab_test2(np.float16, np.float16, output)
        elif case_name == "MlaPrologAbOnBoardTest.test_MlaPrologAb_float16_32_32_1_256_1024_1536":
            gen_mla_prolog_ab_test3(np.float16, np.float16, output)
        elif case_name == "MlaPrologAbOnBoardTest.test_MlaPrologAb_bfloat16_32_32_1_256_1024_1536":
            gen_mla_prolog_ab_test3(bfloat16, bfloat16, output)
        elif case_name == "MlaPrologAbOnBoardTest.test_MlaPrologAb_float16_32_32_1_256_7168_1536":
            gen_mla_prolog_ab_test_net1(np.float16, np.float16, output)
        elif case_name == "MlaPrologAbOnBoardTest.test_MlaPrologAb_bfloat16_32_32_1_256_7168_1536":
            gen_mla_prolog_ab_test_net1(bfloat16, bfloat16, output)
        elif case_name == "MlaPrologAbOnBoardTest.test_MlaPrologAb_bfloat16_4_32_1_256_7168_1536":
            gen_mla_prolog_ab_test_net_b4(bfloat16, bfloat16, output)
        elif case_name == "MlaPrologAbOnBoardTest.test_MlaPrologAb_float16_32_2_1_256_256_512_quant":
            gen_mla_prolog_ab_test1(np.float16, np.float16, output, True)
        elif case_name == "MlaPrologAbOnBoardTest.test_MlaPrologAb_float16_32_32_1_256_7168_1536_quant":
            gen_mla_prolog_ab_test_net1(np.float16, np.float16, output, True)
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
        "MlaPrologAbOnBoardTest.test_MlaPrologAb_float16_32_2_1_256_256_512_quant",
    ]
    # 函数调用
    ret: bool = True
    for cs in case_name_list:
        output: Path = Path(g_src_root, "build/output/bin/golden", cs).resolve()
        output.mkdir(parents=True, exist_ok=True)
        ret = gen_mla_prolog_ab_date(case_name=cs, output=output)
    return ret


if __name__ == "__main__":
    exit(0 if main() else 1)
