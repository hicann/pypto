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
""" rope 相关用例 Golden 生成逻辑.

本脚本有 2 种执行模式:
1. CI批跑时, 由 cmake/scripts/golden_ctrl.py 调用, 为避免日志过多, 此时 logging 级别为 logging.INFO;
2. 单独调试时, 本脚本单独被调用, 此时 logging 级别为 logging.DEBUG;
"""
import os
import sys
import logging
from pathlib import Path
from typing import List

import numpy as np
from pathlib import Path
from collections import namedtuple
from ml_dtypes import bfloat16

# from scatterupdate_operator import gen_graph_D_data_bf16
# from scatterupdate_operator import scatter_update

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

np.random.seed(10)

InputArgs = namedtuple('input_args', 'q_shape k_shape cos_shape pos_ids_shape dtype pos_ids_dtype')
InputArgsV2 = namedtuple('input_args', 'q_shape k_shape cos_shape dtype')
InputPathArgs = namedtuple('input_path_args', 'q_path, k_path, cos_path, sin_path, pos_ids_path')
InputPathArgsV2 = namedtuple('input_path_args', 'q_path, k_path, cos_path, sin_path')
InputReturn = namedtuple('input_return', 'q k cos sin position_ids save_dtype')
InputReturnV2 = namedtuple('input_return', 'q k cos sin save_dtype')


def rms_norm_bf16(x):
    x = x.astype(np.float32)
    res = x / np.sqrt(np.mean(np.square(x), axis=-1, keepdims=True) + 1e-6)
    res = res.astype(bfloat16)

    return res


def scatter_update(past_key_states, key_states, indices, B, S, S2, kv_lora_rank, qk_rope_head_dim, axis):
    z = past_key_states
    if axis == -2:
        for b in range(B):
            for s in range(S2):
                index = indices[0]
                if s == index:
                    logging.debug("find the index value and to replace!")
                    z[b][0][index][:] = key_states[b][0][0][:]
    return z


def gen_graph_D_data_bf16(B, S, S2, kv_lora_rank, qk_rope_head_dim, axis, output_dir: Path, k_pe, is_cd=False):
    dtype = bfloat16
    indices_dtype = np.int32
    shape_params = [B, 1, S2, kv_lora_rank + qk_rope_head_dim]
    shape_indices = [1]  # default support index dim=1

    src1_shape = [B, 1, S, kv_lora_rank + qk_rope_head_dim]

    shape_res = [B, 1, S2, kv_lora_rank + qk_rope_head_dim]

    shape_kv_len = [1]
    shape_compressed_kv = [B, S, kv_lora_rank]
    shape_k_pe_rope = [B, 1, S, qk_rope_head_dim]
    shape_past_key_states = [B, 1, S2, kv_lora_rank + qk_rope_head_dim]

    logging.debug("shape params0 is %s", shape_params)
    logging.debug("shape params1 is %s", src1_shape)
    logging.debug("shape indices is %s", shape_indices)
    logging.debug("shape res is %s", shape_past_key_states)

    x_path = Path(output_dir, 'x.bin')  # last kery_states
    past_key_states = np.random.uniform(1, 1, shape_past_key_states).astype(
        dtype)  # [B, 1, S2, kv_lora_rank+qk_rope_head_dim]
    past_key_states.tofile(x_path)

    compressed_kv_path = Path(output_dir, 'compressed_kv.bin')
    k_pe_rope_path = Path(output_dir, 'y.bin')

    indices_path = Path(output_dir, 'indices.bin')
    z_path = Path(output_dir, 'z_golden.bin')

    indices = np.random.randint(0, shape_params[axis], size=shape_kv_len).astype(indices_dtype)
    indices.tofile(indices_path)
    logging.debug("====indices===== %s", indices)

    compressed_kv = np.random.uniform(-4, 4, shape_compressed_kv).astype(dtype)  # [B, S, kv_lora_rank]
    k_pe_rope = np.random.uniform(2, 2, shape_k_pe_rope).astype(dtype)  # [B, 1, S, qk_rope_head_dim]
    if is_cd:
        k_pe_rope = k_pe.astype(dtype)
    compressed_kv.tofile(compressed_kv_path)
    k_pe_rope.tofile(k_pe_rope_path)
    logging.debug("=======k_pe_rope=== %s", k_pe_rope)
    logging.debug("=======compressed_kv=== %s", compressed_kv)
    k_nope = rms_norm_bf16(compressed_kv)  # [B, S, kv_lora_rank]
    k_nope_new = k_nope.reshape(B, S, 1, kv_lora_rank).transpose(0, 2, 1, 3)  # [B, 1, S, kv_lora_rank]

    key_states = np.concatenate((k_nope_new, k_pe_rope), axis=-1)  # [B, 1, S, kv_lora_rank + qk_rope_head_dim]

    past_key_states_new = scatter_update(past_key_states, key_states, indices, B, S, S2, kv_lora_rank, qk_rope_head_dim,
                                         -2)
    logging.debug("=======past_key_states_new=== %s", compressed_kv)
    past_key_states_new.tofile(z_path)


def rope_middle_res(x_shape_dir_path, q, k, cos, sin, position_ids, calc_dtype, save_dtype=np.float32):
    cos2 = np.expand_dims(cos[position_ids], axis=1)  # [b,1,s,qk_d]
    sin2 = np.expand_dims(sin[position_ids], axis=1)  # [b,1,s,qk_d]
    cos2.tofile(Path(x_shape_dir_path, "cos_unsqueeze.bin"))
    sin2.tofile(Path(x_shape_dir_path, "sin_unsqueeze.bin"))
    cos_sin_mul = sin2 * cos2
    cos_sin_mul.tofile(Path(x_shape_dir_path, "cos_sin_T_U_mul.bin"))

    q_brc = q * k + q * k
    q_brc.tofile(Path(x_shape_dir_path, "q_brc.bin"))
    q_brc2 = q * k
    q_brc2.tofile(Path(x_shape_dir_path, "q_brc_2.bin"))

    q_rotate_half = rotate_half(q)
    q_rotate_half.tofile(Path(x_shape_dir_path, "q_rotate_half.bin"))
    q_rotate_mul = q_rotate_half * k
    q_rotate_mul.tofile(Path(x_shape_dir_path, "q_rotate_brc.bin"))

    b, n, s, d = q.shape
    arr_2 = np.zeros((b, 64, s, d)).astype(calc_dtype)
    k_brc = k + arr_2
    k_brc.tofile(Path(x_shape_dir_path, "k_brc.bin"))

    b, h, s, d = q.shape
    q_trs = q.reshape(b, h, s, d // 2, 2).transpose(0, 1, 2, 4, 3).reshape(b, h, s, d)  # [b,n,s,qk_d]
    b, h, s, d = k.shape
    k_trs = k.reshape(b, h, s, d // 2, 2).transpose(0, 1, 2, 4, 3).reshape(b, h, s, d)  # [b,n,s,qk_d]

    q_trs.astype(save_dtype).tofile(Path(x_shape_dir_path, "q_R_T_R.bin"))
    k_trs.astype(save_dtype).tofile(Path(x_shape_dir_path, "k_R_T_R.bin"))

    q_mul_cos = q_trs * cos
    q_mul_cos.tofile(Path(x_shape_dir_path, "q_R_T_R_mul.bin"))

    q1 = q_trs[..., : q_trs.shape[-1] // 2]  # 左一半
    q2 = q_trs[..., q_trs.shape[-1] // 2:]  # 右一半
    neg_q2 = q2 * (-1)
    neg_q2.tofile(Path(x_shape_dir_path, "q_R_T_R_View_Muls.bin"))
    concat_q = np.concatenate((neg_q2, q1), axis=-1)

    q_rotate_half = rotate_half(q_trs)
    q_rotate_half.tofile(Path(x_shape_dir_path, "q_RTR_rotate_half.bin"))
    k_rotate_half = rotate_half(k_trs)
    k_rotate_half.tofile(Path(x_shape_dir_path, "k_RTR_rotate_half.bin"))

    q_rotate_half2 = q_rotate_half * k
    q_rotate_half2.tofile(Path(x_shape_dir_path, "q_RTR_rotate_half_mulk.bin"))

    q_rotate_half1 = q_rotate_half * sin2
    q_rotate_half1.tofile(Path(x_shape_dir_path, "q_RTR_rotate_half_mul_sin.bin"))

    q_e = q_mul_cos + q_rotate_half1
    q_e.tofile(Path(x_shape_dir_path, "qE.bin"))


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return np.concatenate((-x2, x1), axis=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    cos = np.expand_dims(cos[position_ids], axis=unsqueeze_dim)  # [b,1,s,qk_d]
    sin = np.expand_dims(sin[position_ids], axis=unsqueeze_dim)  # [b,1,s,qk_d]
    logging.debug("====expand sin.shape===== %s", sin.shape)
    logging.debug("====expand cos.shape===== %s", cos.shape)

    b, h, s, d = q.shape
    q = q.reshape(b, h, s, d // 2, 2).transpose(0, 1, 2, 4, 3).reshape(b, h, s, d)  # [b,n,s,qk_d]

    b, h, s, d = k.shape
    k = k.reshape(b, h, s, d // 2, 2).transpose(0, 1, 2, 4, 3).reshape(b, h, s, d)  # [b,n,s,qk_d]

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def gen_input_data(input_args, input_path_args):
    q_shape, k_shape, cos_shape, pos_ids_shape, dtype, pos_ids_dtype = input_args.q_shape, input_args.k_shape, \
        input_args.cos_shape, input_args.pos_ids_shape, input_args.dtype, input_args.pos_ids_dtype
    q_path, k_path, cos_path, sin_path, pos_ids_path = input_path_args.q_path, input_path_args.k_path, \
        input_path_args.cos_path, input_path_args.sin_path, input_path_args.pos_ids_path,

    if dtype == "bf16":
        logging.debug("================= gen bfloat16 data =====================")
        # 需要借助tensorflow保存
        save_dtype = bfloat16
    elif dtype == "fp16":
        logging.debug("================= gen float16 data =====================")
        save_dtype = np.float16
    elif dtype == "fp32":
        logging.debug("================= gen float32 data =====================")
        save_dtype = np.float32
    else:
        raise RuntimeError("数据类型暂不支持")

    q = np.random.uniform(-0.1, 0.1, q_shape).astype(save_dtype)  # [b, n, s, qk_d]
    k = np.random.uniform(-0.1, 0.1, k_shape).astype(save_dtype)  # [b, n, s, qk_d]
    cos = np.random.uniform(-0.1, 0.1, cos_shape).astype(save_dtype)  # [s, qk_d]
    sin = np.random.uniform(-0.1, 0.1, cos_shape).astype(save_dtype)  # [s, qk_d]
    q.tofile(q_path)
    k.tofile(k_path)
    cos.tofile(cos_path)
    sin.tofile(sin_path)

    position_ids = np.random.randint(0, cos_shape[0], size=pos_ids_shape).astype(pos_ids_dtype)
    position_ids.tofile(pos_ids_path)

    input_return = InputReturn(q, k, cos, sin, position_ids, save_dtype)
    return input_return


def gen_input_data_v2(input_args, input_path_args):
    q_shape, k_shape, cos_shape, dtype = input_args.q_shape, input_args.k_shape, \
        input_args.cos_shape, input_args.dtype
    q_path, k_path, cos_path, sin_path = input_path_args.q_path, input_path_args.k_path, \
        input_path_args.cos_path, input_path_args.sin_path

    if dtype == "bf16":
        logging.debug("================= gen bfloat16 data =====================")
        # 需要借助tensorflow保存
        save_dtype = bfloat16
    elif dtype == "fp16":
        logging.debug("================= gen float16 data =====================")
        save_dtype = np.float16
    elif dtype == "fp32":
        logging.debug("================= gen float32 data =====================")
        save_dtype = np.float32
    else:
        raise RuntimeError("数据类型暂不支持")

    q = np.random.uniform(-0.1, 0.1, q_shape).astype(save_dtype)  # [b, n, s, qk_d]
    k = np.random.uniform(-0.1, 0.1, k_shape).astype(save_dtype)  # [b, n, s, qk_d]
    cos = np.random.uniform(-0.1, 0.1, cos_shape).astype(save_dtype)  # [s, qk_d]
    sin = np.random.uniform(-0.1, 0.1, cos_shape).astype(save_dtype)  # [s, qk_d]
    q.tofile(q_path)
    k.tofile(k_path)
    cos.tofile(cos_path)
    sin.tofile(sin_path)

    input_return = InputReturnV2(q, k, cos, sin, save_dtype)
    return input_return


def apply_rotary_pos_emb_v2(q, k, cos, sin, unsqueeze_dim=2):
    cos = np.expand_dims(cos, axis=unsqueeze_dim)
    sin = np.expand_dims(sin, axis=unsqueeze_dim)
    logging.debug("====expand sin.shape===== %s", sin.shape)
    logging.debug("====expand cos.shape===== %s", cos.shape)

    b, s, h, d = q.shape
    q = q.reshape(b, s, h, d // 2, 2).transpose(0, 1, 2, 4, 3).reshape(b, s, h, d)  # [b,n,s,qk_d]

    b, s, h, d = k.shape
    k = k.reshape(b, s, h, d // 2, 2).transpose(0, 1, 2, 4, 3).reshape(b, s, h, d)  # [b,n,s,qk_d]

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def rope_golden_generator_v2(dir_path, case_name, q_shape, k_shape, cos_shape, dtype):
    x_shape_dir_path = Path(dir_path, '_'.join([str(i) for i in q_shape]))
    if not os.path.exists(x_shape_dir_path):
        os.makedirs(x_shape_dir_path)

    q_path = Path(x_shape_dir_path, "q.bin")
    k_path = Path(x_shape_dir_path, "k.bin")
    cos_path = Path(x_shape_dir_path, "cos.bin")
    sin_path = Path(x_shape_dir_path, "sin.bin")
    q_embed_path = Path(x_shape_dir_path, "qEmbed_res.bin")
    k_embed_path = Path(x_shape_dir_path, "kEmbed_res.bin")

    input_exist = q_path.exists() and k_path.exists() and cos_path.exists() and sin_path.exists()
    output_exist = q_embed_path.exists() and k_embed_path.exists()
    if input_exist and output_exist:
        logging.debug("Case(%s), Golden hit cache.", case_name)
    else:
        calc_dtype = np.float32
        input_args = InputArgsV2(q_shape, k_shape, cos_shape, dtype)
        input_path_args = InputPathArgsV2(q_path, k_path, cos_path, sin_path)
        input_return = gen_input_data_v2(input_args, input_path_args)
        q, k, cos, sin, save_dtype = input_return.q, input_return.k, input_return.cos, input_return.sin, \
            input_return.save_dtype
        # 转成fp32，进行计算
        q = q.astype(calc_dtype)
        k = k.astype(calc_dtype)
        cos = cos.astype(calc_dtype)
        sin = sin.astype(calc_dtype)

        # 记录rope的中间结果，待正式版本rope调通后，删除，辅助调测使用
        logging.debug("====q.shape===== %s", q.shape)
        logging.debug("====k.shape===== %s", k.shape)
        logging.debug("====sin.shape===== %s", sin.shape)
        logging.debug("====cos.shape===== %s", cos.shape)

        q_embed, k_embed = apply_rotary_pos_emb_v2(q, k, cos, sin)
        logging.debug("====q_embed.shape===== %s", q_embed.shape)
        logging.debug("====k_embed.shape===== %s", k_embed.shape)

        q_embed = q_embed.astype(save_dtype)
        k_embed = k_embed.astype(save_dtype)
        q_embed.tofile(q_embed_path)
        k_embed.tofile(k_embed_path)
    return True


def rope_golden_generator(dir_path, case_name, q_shape, k_shape, cos_shape, pos_ids_shape, dtype, pos_ids_dtype):
    x_shape_dir_path = Path(dir_path, '_'.join([str(i) for i in q_shape]))
    if not os.path.exists(x_shape_dir_path):
        os.makedirs(x_shape_dir_path)

    q_path = Path(x_shape_dir_path, "q.bin")
    k_path = Path(x_shape_dir_path, "k.bin")
    cos_path = Path(x_shape_dir_path, "cos.bin")
    sin_path = Path(x_shape_dir_path, "sin.bin")
    pos_ids_path = Path(x_shape_dir_path, "pos_ids.bin")
    q_embed_path = Path(x_shape_dir_path, "qEmbed_res.bin")
    k_embed_path = Path(x_shape_dir_path, "kEmbed_res.bin")

    input_exist = (q_path.exists() and k_path.exists() and cos_path.exists() and sin_path.exists() and
                   pos_ids_path.exists())
    output_exist = q_embed_path.exists() and k_embed_path.exists()
    if input_exist and output_exist:
        logging.debug("Case(%s), Golden hit cache.", case_name)
    else:
        calc_dtype = np.float32
        input_args = InputArgs(q_shape, k_shape, cos_shape, pos_ids_shape, dtype, pos_ids_dtype)
        input_path_args = InputPathArgs(q_path, k_path, cos_path, sin_path, pos_ids_path)
        input_return = gen_input_data(input_args, input_path_args)
        q, k, cos, sin, position_ids, save_dtype = input_return.q, input_return.k, input_return.cos, input_return.sin, \
            input_return.position_ids, input_return.save_dtype
        # 转成fp32，进行计算
        q = q.astype(calc_dtype)
        k = k.astype(calc_dtype)
        cos = cos.astype(calc_dtype)
        sin = sin.astype(calc_dtype)

        # 记录rope的中间结果，待正式版本rope调通后，删除，辅助调测使用
        rope_middle_res(x_shape_dir_path, q, k, cos, sin, position_ids, calc_dtype)

        logging.debug("====q.shape===== %s", q.shape)
        logging.debug("====k.shape===== %s", k.shape)
        logging.debug("====sin.shape===== %s", sin.shape)
        logging.debug("====cos.shape===== %s", cos.shape)
        logging.debug("====position_ids.shape===== %s", position_ids.shape)
        logging.debug("position_ids: \n%s", position_ids)
        q.tofile(q_path)
        k.tofile(k_path)
        cos.tofile(cos_path)
        sin.tofile(sin_path)
        position_ids.tofile(pos_ids_path)

        q_embed, k_embed = apply_rotary_pos_emb(q, k, cos, sin, position_ids)
        logging.debug("====q_embed.shape===== %s", q_embed.shape)
        logging.debug("====k_embed.shape===== %s", k_embed.shape)

        q_embed = q_embed.astype(save_dtype)
        k_embed = k_embed.astype(save_dtype)
        q_embed.tofile(q_embed_path)
        k_embed.tofile(k_embed_path)
    return True


def rope_subgraph_golden_generator(dir_path, case_name, q_shape, k_shape, cos_shape, pos_ids_shape, dtype,
                                   pos_ids_dtype, is_cd=False, is_4k=False):
    x_shape_dir_path = Path(dir_path, '_'.join([str(i) for i in q_shape]))
    if not os.path.exists(x_shape_dir_path):
        os.makedirs(x_shape_dir_path)

    q_path = Path(x_shape_dir_path, "qPe.bin")
    k_path = Path(x_shape_dir_path, "kPe.bin")
    cos_path = Path(x_shape_dir_path, "cos.bin")
    sin_path = Path(x_shape_dir_path, "sin.bin")
    pos_ids_path = Path(x_shape_dir_path, "pos_ids.bin")
    q_embed_path = Path(x_shape_dir_path, "qEmbed_res.bin")
    k_embed_path = Path(x_shape_dir_path, "kEmbed_res.bin")

    input_exist = (q_path.exists() and k_path.exists() and cos_path.exists() and sin_path.exists() and
                   pos_ids_path.exists())
    output_exist = q_embed_path.exists() and k_embed_path.exists()
    if input_exist and output_exist:
        logging.debug("Case(%s), Golden hit cache.", case_name)
    else:
        calc_dtype = np.float32
        input_args = InputArgs(q_shape, k_shape, cos_shape, pos_ids_shape, dtype, pos_ids_dtype)
        input_path_args = InputPathArgs(q_path, k_path, cos_path, sin_path, pos_ids_path)
        input_return = gen_input_data(input_args, input_path_args)
        q, k, cos, sin, position_ids, save_dtype = input_return.q, input_return.k, input_return.cos, input_return.sin, \
            input_return.position_ids, input_return.save_dtype
        # 转成fp32，进行计算
        q_trs = q.transpose(0, 2, 1, 3)
        b, s, d = k.shape
        k_reshape = k.reshape(b, 1, s, d)

        q_trs = q_trs.astype(calc_dtype)
        k_reshape = k_reshape.astype(calc_dtype)
        cos = cos.astype(calc_dtype)
        sin = sin.astype(calc_dtype)

        logging.debug("====q_trs.shape===== %s", q_trs.shape)
        logging.debug("====k_reshape.shape===== %s", k_reshape.shape)
        logging.debug("====sin.shape===== %s", sin.shape)
        logging.debug("====cos.shape===== %s", cos.shape)
        logging.debug("====position_ids.shape===== %s", position_ids.shape)
        logging.debug("position_ids: \n%s", position_ids)

        # 记录rope的中间结果，待正式版本rope调通后，删除，辅助调测使用
        rope_middle_res(x_shape_dir_path, q_trs, k_reshape, cos, sin, position_ids, calc_dtype, save_dtype)

        q_embed, k_embed = apply_rotary_pos_emb(q_trs, k_reshape, cos, sin, position_ids)
        if is_cd:
            b, s, s2, kv_lora_rank, qk_rope_head_dim = 32, 1, 512, 512, 64
            if is_4k:
                s2 = 4096
            gen_graph_D_data_bf16(b, s, s2, kv_lora_rank, qk_rope_head_dim, -2, x_shape_dir_path, k_embed, True)
        logging.debug("====q_embed.shape===== %s", q_embed.shape)
        logging.debug("====k_embed.shape===== %s", k_embed.shape)

        q_embed = q_embed.astype(save_dtype)
        k_embed = k_embed.astype(save_dtype)
        q_embed.tofile(q_embed_path)
        k_embed.tofile(k_embed_path)
    return True

@GoldenRegister.reg_golden_func(
    case_names=[
        # rope sub graph
        "RoPEOnBoardTest.test_operation_rope_reshape_transpose_reshape_muls",
        "RoPEOnBoardTest.test_operation_rope_tensorIndex_unsqueeze_mul",
        "RoPEOnBoardTest.test_operation_rope_reshape_view_muls",
        "RoPEOnBoardTest.test_operation_rope_reshape_view_muls_concat",
        "RoPEOnBoardTest.test_operation_rope_deepseekv3",
        "RoPEOnBoardTest.test_operation_rope_v2_deepseekv3",
        "RoPEOnBoardTest.test_operation_rope_v2_deepseekv3_b32",
    ]
)
def gen_rope_golden(case_name: str, output: Path) -> bool:
    if case_name == "RoPEOnBoardTest.test_operation_rope_reshape_transpose_reshape_muls":
        b, n, s, qk_d = 1, 128, 1, 64
        dtype = "fp32"
        pos_ids_dtype = np.int32  # int64 actually
        rope_golden_generator(output, case_name, [b, n, s, qk_d], [b, 1, s, qk_d], [s, qk_d], [b, s], dtype,
                              pos_ids_dtype)
    elif case_name == "RoPEOnBoardTest.test_operation_rope_tensorIndex_unsqueeze_mul":
        b, n, s, qk_d = 32, 128, 1, 64
        dtype = "fp32"
        pos_ids_dtype = np.int32  # int64 actually
        rope_golden_generator(output, case_name, [b, n, s, qk_d], [b, 1, s, qk_d], [s, qk_d], [b, s], dtype,
                              pos_ids_dtype)
    elif case_name == "RoPEOnBoardTest.test_operation_rope_reshape_view_muls":
        b, n, s, qk_d = 1, 128, 1, 64
        dtype = "fp32"
        pos_ids_dtype = np.int32  # int64 actually
        rope_golden_generator(output, case_name, [b, n, s, qk_d], [b, 1, s, qk_d], [s, qk_d], [b, s], dtype,
                              pos_ids_dtype)
    elif case_name == "RoPEOnBoardTest.test_operation_rope_reshape_view_muls_concat":
        b, n, s, qk_d = 1, 128, 1, 64
        dtype = "fp32"
        pos_ids_dtype = np.int32  # int64 actually
        rope_golden_generator(output, case_name, [b, n, s, qk_d], [b, 1, s, qk_d], [s, qk_d], [b, s], dtype,
                              pos_ids_dtype)
    elif case_name == "RoPEOnBoardTest.test_operation_rope_deepseekv3":
        b, n, s, qk_d = 1, 128, 1, 64
        dtype = "fp32"
        pos_ids_dtype = np.int32  # int64 actually
        rope_golden_generator(output, case_name, [b, n, s, qk_d], [b, 1, s, qk_d], [s, qk_d], [b, s], dtype,
                              pos_ids_dtype)
    elif case_name == "RoPEOnBoardTest.test_operation_rope_v2_deepseekv3":
        b, s, n, qk_d = 1, 1, 128, 64
        dtype = "fp32"
        rope_golden_generator_v2(output, case_name, [b, s, n, qk_d], [b, s, 1, qk_d], [b, s, qk_d], dtype)
    elif case_name == "RoPEOnBoardTest.test_operation_rope_v2_deepseekv3_b32":
        b, s, n, qk_d = 32, 1, 128, 64
        dtype = "fp32"
        rope_golden_generator_v2(output, case_name, [b, s, n, qk_d], [b, s, 1, qk_d], [b, s, qk_d], dtype)
    else:
        logging.error("Can't get func to gen golden, Case(%s)", case_name)
        return False
    return True


@GoldenRegister.reg_golden_func(
    case_names=[
        "RoPESubGraphOnBoardTest.test_operation_rope_subgraph_deepseekv3",
        "RoPESubGraphOnBoardTest.test_operation_rope_subgraph_deepseekv3_fp16",
        "RoPESubGraphOnBoardTest.test_operation_rope_subgraph_deepseekv3_fp16_2batch",
        "RoPESubGraphOnBoardTest.test_operation_rope_subgraph_deepseekv3_bf16",
        "RoPESubGraphOnBoardTest.test_operation_rope_subgraph_deepseekv3_bf16_2batch",
        "RoPESubGraphOnBoardTest.test_operation_rope_subgraph_deepseekv3_bf16_32batch",
        "RoPESubGraphOnBoardTest.test_CD_bf16_32batch",
        "RoPESubGraphOnBoardTest.test_CD_bf16_32batch_4k",
    ]
)
def gen_rope_subgraph_golden(case_name: str, output: Path) -> bool:
    if case_name == "RoPESubGraphOnBoardTest.test_operation_rope_subgraph_deepseekv3":
        b, n, s, qk_d = 1, 32, 1, 64
        dtype = "fp32"
        pos_ids_dtype = np.int32  # int64 actually
        rope_subgraph_golden_generator(output, case_name, [b, s, n, qk_d], [b, s, qk_d], [s, qk_d], [b, s],
                                       dtype, pos_ids_dtype)
    elif case_name == "RoPESubGraphOnBoardTest.test_operation_rope_subgraph_deepseekv3_fp16":
        b, n, s, qk_d = 1, 32, 1, 64
        dtype = "fp16"
        pos_ids_dtype = np.int32  # int64 actually
        rope_subgraph_golden_generator(output, case_name, [b, s, n, qk_d], [b, s, qk_d], [s, qk_d], [b, s],
                                       dtype, pos_ids_dtype)
    elif case_name == "RoPESubGraphOnBoardTest.test_operation_rope_subgraph_deepseekv3_fp16_2batch":
        b, n, s, qk_d = 2, 32, 1, 64
        dtype = "fp16"
        pos_ids_dtype = np.int32  # int64 actually
        rope_subgraph_golden_generator(output, case_name, [b, s, n, qk_d], [b, s, qk_d], [s, qk_d], [b, s],
                                       dtype, pos_ids_dtype)
    elif case_name == "RoPESubGraphOnBoardTest.test_operation_rope_subgraph_deepseekv3_bf16":
        b, n, s, qk_d = 1, 32, 1, 64
        dtype = "bf16"
        pos_ids_dtype = np.int32  # int64 actually
        rope_subgraph_golden_generator(output, case_name, [b, s, n, qk_d], [b, s, qk_d], [s, qk_d], [b, s],
                                       dtype, pos_ids_dtype)
    elif case_name == "RoPESubGraphOnBoardTest.test_operation_rope_subgraph_deepseekv3_bf16_2batch":
        b, n, s, qk_d = 2, 32, 1, 64
        dtype = "bf16"
        pos_ids_dtype = np.int32  # int64 actually
        rope_subgraph_golden_generator(output, case_name, [b, s, n, qk_d], [b, s, qk_d], [s, qk_d], [b, s],
                                       dtype, pos_ids_dtype)
    elif case_name == "RoPESubGraphOnBoardTest.test_operation_rope_subgraph_deepseekv3_bf16_32batch":
        b, n, s, qk_d = 32, 32, 1, 64
        dtype = "bf16"
        pos_ids_dtype = np.int32  # int64 actually
        rope_subgraph_golden_generator(output, case_name, [b, s, n, qk_d], [b, s, qk_d], [s, qk_d], [b, s],
                                       dtype, pos_ids_dtype)
    elif case_name == "RoPESubGraphOnBoardTest.test_CD_bf16_32batch":
        b, n, s, qk_d, kv_d, s2 = 32, 32, 1, 64, 512, 512
        dtype = "bf16"
        pos_ids_dtype = np.int32  # int64 actually
        rope_subgraph_golden_generator(output, case_name, [b, s, n, qk_d], [b, s, qk_d], [s, qk_d], [b, s],
                                       dtype, pos_ids_dtype, True)
    elif case_name == "RoPESubGraphOnBoardTest.test_CD_bf16_32batch_4k":
        b, n, s, qk_d, kv_d, s2 = 32, 32, 1, 64, 512, 4096
        dtype = "bf16"
        pos_ids_dtype = np.int32  # int64 actually
        rope_subgraph_golden_generator(output, case_name, [b, s, n, qk_d], [b, s, qk_d], [s, qk_d], [b, s],
                                       dtype, pos_ids_dtype, True, True)
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
        "RoPEOnBoardTest.test_operation_rope_deepseekv3",
        "RoPEOnBoardTest.test_operation_rope_v2_deepseekv3",
        "RoPEOnBoardTest.test_operation_rope_v2_deepseekv3_b32",
        "RoPEOnBoardTest.test_operation_rope_reshape_transpose_reshape_muls",
        "RoPEOnBoardTest.test_operation_rope_tensorIndex_unsqueeze_mul",
        "RoPEOnBoardTest.test_operation_rope_reshape_view_muls",
        "RoPEOnBoardTest.test_operation_rope_reshape_view_muls_concat",
    ]
    # 函数调用
    ret: bool = True
    for cs in case_name_list:
        output: Path = Path(g_src_root, "build/output/bin/golden", cs).resolve()
        output.mkdir(parents=True, exist_ok=True)
        ret = gen_rope_golden(case_name=cs, output=output)
    if not ret:
        return ret

    # 用例名称
    case_name_list: List[str] = [
        "RoPESubGraphOnBoardTest.test_operation_rope_subgraph_deepseekv3",
        "RoPESubGraphOnBoardTest.test_operation_rope_subgraph_deepseekv3_bf16",
        "RoPESubGraphOnBoardTest.test_operation_rope_subgraph_deepseekv3_fp16",
        "RoPESubGraphOnBoardTest.test_operation_rope_subgraph_deepseekv3_fp16_2batch",
        "RoPESubGraphOnBoardTest.test_operation_rope_subgraph_deepseekv3_bf16_2batch",
        "RoPESubGraphOnBoardTest.test_operation_rope_subgraph_deepseekv3_bf16_32batch",
        "RoPESubGraphOnBoardTest.test_CD_bf16_32batch",
        "RoPESubGraphOnBoardTest.test_CD_bf16_32batch_4K",
    ]
    # 函数调用
    ret: bool = True
    for cs in case_name_list:
        output: Path = Path(g_src_root, "build/output/bin/golden", cs).resolve()
        output.mkdir(parents=True, exist_ok=True)
        ret = gen_rope_subgraph_golden(case_name=cs, output=output)
    return ret


if __name__ == "__main__":
    exit(0 if main() else 1)
