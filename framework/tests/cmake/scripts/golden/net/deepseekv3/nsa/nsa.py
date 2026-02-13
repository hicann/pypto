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

""" MLA_prolog 子图 相关用例 Golden 生成逻辑.

本脚本有 2 种执行模式:
1. CI批跑时, 由 cmake/scripts/golden_ctrl.py 调用, 为避免日志过多, 此时 logging 级别为 logging.INFO;
2. 单独调试时, 本脚本单独被调用, 此时 logging 级别为 logging.DEBUG;
"""
import sys
import math
import logging
from pathlib import Path
from typing import List
import time

import torch
import numpy as np
from ml_dtypes import bfloat16
import os

project_root = os.path.dirname(os.path.abspath(__file__))  # 当前脚本目录
golden_parent = os.path.join(project_root, "../../../../")  # 假设 golden 在上级目录
sys.path.insert(0, golden_parent)

np.random.seed(0)
if __name__ == "__main__":
    """ 单独调试时配置 """
    # 日志级别
    logging.basicConfig(format='%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s: %(message)s',
                        level=logging.DEBUG)
    # 系统 import 路径
    g_src_root: Path = Path(Path(__file__).parent, "../../../../../").resolve()
    logging.debug("SrcRoot: %s", g_src_root)
    g_ctrl_path: Path = Path(g_src_root, "scripts")
    if str(g_ctrl_path) not in sys.path:
        sys.path.append(str(g_ctrl_path))
    from golden_register import GoldenRegister  # 单独调试 import 失败, 需确认上文中 '系统 import 路径' 配置正确
else:
    from golden_register import GoldenRegister

fp32 = np.float32


def sigmoid(x):
    return 1 / (1 + torch.exp(-x))


def softmax(x, axis=-1):
    x_max = np.max(x, axis=axis, keepdims=True)
    x_exp = np.exp(x - x_max)
    x_sum = np.sum(x_exp, axis=axis, keepdims=True)
    return x_exp / x_sum


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return np.concatenate((-x2, x1), axis=-1)


def apply_rope(x, cos, sin, pos_ids, unsqueeze_dim=1):
    cos = np.expand_dims(cos[pos_ids], axis=unsqueeze_dim)
    sin = np.expand_dims(sin[pos_ids], axis=unsqueeze_dim)
    d = cos.shape[-1]
    x_d = x.shape[-1]
    x_rope = x[..., x_d - d:].copy()
    b, n2, s_len, _ = x_rope.shape
    x_rope = x_rope.reshape(b, n2, s_len, d // 2, 2).transpose(0, 1, 2, 4, 3).reshape(b, n2, s_len, d)
    x_embed = (x_rope * cos) + (rotate_half(x_rope) * sin)
    x[..., x_d - d:] = x_embed
    return x


# mlp cmp: rope + matmul + sigmoid + matmul
def mlp_compression(kv_local, w_1, w_2):
    b, n2, _, _ = kv_local.shape
    kv_local_reshape = kv_local.reshape(b, n2, -1)
    mm1 = np.matmul(kv_local_reshape, w_1)
    sigmoid1 = sigmoid(mm1)
    mm2 = np.matmul(sigmoid1, w_2)
    mm2reshape = mm2.reshape(b, n2, 1, -1)
    return mm2reshape


# avg pool cmp: reduce mean
def avg_pool_compression(kv_local, w_1):
    vmul1 = kv_local * w_1.reshape(1, 1, 1, -1)
    reduce1 = np.mean(vmul1, axis=2, keepdims=True)
    return reduce1


# kv_cmp compress s2 to (s2-l)//d+1
def kv_compression(k, v, avg_wk, avg_wv, mlp_wk1, mlp_wk2, mlp_wv1, mlp_wv2, cos, sin, pos_ids, params, l=32, d=16,
                   mode='avg'):
    b = params.get("b")
    n = params.get("n")  # s=1
    _, _, s2, _ = k.shape
    kv_cmp_len = (s2 - l) // d + 1
    k_cmp = []
    v_cmp = []
    k_cmp_block = np.random.rand(b, n)
    v_cmp_block = np.random.rand(b, n)
    for i in range(kv_cmp_len):
        k_block = k[:, :, i * d: i * d + l, :]
        v_block = v[:, :, i * d: i * d + l, :]

        if k_cmp_block.size == b * n or v_cmp_block.size == b * n:
            if mode == 'avg':
                k_cmp_block = avg_pool_compression(k_block, avg_wk)
                v_cmp_block = avg_pool_compression(v_block, avg_wv)
            else:
                k_block = apply_rope(k_block, cos, sin, pos_ids)
                k_cmp_block = mlp_compression(k_block, mlp_wk1, mlp_wk2)
                v_cmp_block = mlp_compression(v_block, mlp_wv1, mlp_wv2)
        k_cmp.append(k_cmp_block)
        v_cmp.append(v_cmp_block)

    k_cmp = np.concatenate(k_cmp, axis=2)
    v_cmp = np.concatenate(v_cmp, axis=2)

    return k_cmp, v_cmp


def gen_cmp_attn(q, k_cmp, v_cmp):
    _, _, _, q_dim = q.shape
    scores = np.matmul(q, k_cmp.transpose(0, 1, 3, 2))
    scores = scores / np.sqrt(q_dim)
    p_cmp = softmax(scores)
    cmp_attn = np.matmul(p_cmp, v_cmp)

    return p_cmp, cmp_attn


def gen_p_slc(p_cmp, l_prime=64, l=32, d=16):
    b, n, s, kv_cmp_len = p_cmp.shape
    out_loop = l_prime // d
    inner_loop = l // d
    reduce_len = kv_cmp_len // out_loop + 1
    p_cmp_reduce = np.zeros((b, n, s, reduce_len))
    for i in range(reduce_len):
        for j in range(out_loop):
            start_idx = i * out_loop + j
            p_cmp_reduce[:, :, :, i] += np.sum(p_cmp[:, :, :, start_idx: start_idx + inner_loop], axis=-1)
    p_slc = np.sum(p_cmp_reduce, axis=1)

    return p_slc


def gen_p_slc_ast(p_cmp, l_prime=64, l=32, d=16):
    b, n, s, kv_cmp_len = p_cmp.shape
    out_loop = l_prime // d
    inner_loop = l // d
    reduce_len = kv_cmp_len // out_loop + 1
    s_slc = (kv_cmp_len + 3) // 4
    p_cmp = p_cmp.reshape(b * n * s, kv_cmp_len)
    trans0 = p_cmp.transpose(1, 0)
    reduce0 = np.zeros((s_slc, b * n * s))
    for i in range(s_slc):
        part0 = trans0[i * out_loop:i * out_loop + out_loop, :]
        part1 = trans0[1 + i * out_loop:1 + i * out_loop + out_loop, :]
        part0Sum = np.sum(part0, axis=0)
        part1Sum = np.sum(part1, axis=0)
        reduce0[i, :] = part0Sum + part1Sum
    trans1 = reduce0.transpose(1, 0)
    reduce1 = np.sum(trans1, axis=0)
    return trans0, reduce0, trans1, reduce1


def gen_topk_indices(p_slc, front=1, near=2, topk=16, actual_len=0):
    b, s, reduce_len = p_slc.shape
    front_indices = np.arange(front)
    near_indices = np.arange((reduce_len if actual_len == 0 else actual_len) - near, reduce_len)
    required_indices = np.concatenate([front_indices, near_indices])

    mask = np.zeros_like(p_slc, dtype=bool)
    mask[:, :, required_indices] = True
    x_masked = np.where(mask, -np.inf, p_slc)

    k = topk - front - near
    additional_indices = np.argpartition(x_masked, -k, axis=-1)[:, :, -k:]
    additional_indices = np.sort(additional_indices)
    topk_indices = np.concatenate([
        np.tile(front_indices, (b, s, 1)),
        additional_indices,
        np.tile(near_indices, (b, s, 1)),
    ], axis=-1)

    return additional_indices, topk_indices


def gen_kv_slc(x, topk_indices, l_prime=64):
    b, s, topk = topk_indices.shape
    x_slc = []

    for i in range(topk):
        # Get the indices for this top-k position
        positions = topk_indices[:, :, i] * l_prime
        # Create a list to store blocks for this top-k position
        blocks = np.zeros((b, s, l_prime, x.shape[-1]), dtype=x.dtype)

        for bi in range(b):
            for si in range(s):
                start = int(positions[bi, si].item())
                end = start + l_prime
                blocks[bi, si] = x[bi, si, start: end, :]

        x_slc.append(blocks)

    x_slc = np.concatenate(x_slc, axis=2)
    return x_slc


def gen_slc_attn(q, p_cmp, k, v, l_prime=64, l=32, d=16, front=1, near=2, topk=16):
    _, _, _, q_dim = q.shape
    p_slc = gen_p_slc(p_cmp, l_prime=l_prime, l=l, d=d)
    topk_indices = gen_topk_indices(p_slc, front=front, near=near, topk=topk)
    k_slc = gen_kv_slc(k, topk_indices, l_prime=l_prime)
    v_slc = gen_kv_slc(v, topk_indices, l_prime=l_prime)
    scores = np.matmul(q, k_slc.transpose(0, 1, 3, 2))
    scores = scores / np.sqrt(q_dim)
    scores = softmax(scores)
    slc_attn = np.matmul(scores, v_slc)

    return slc_attn


def gen_win_attn(q, k, v, win=512):
    _, _, _, q_dim = q.shape
    _, _, s2, _ = k.shape
    k_win = k[:, :, s2 - win:, :]
    v_win = v[:, :, s2 - win:, :]
    scores = np.matmul(q, k_win.transpose(0, 1, 3, 2))
    scores = scores / np.sqrt(q_dim)
    scores = softmax(scores)
    win_attn = np.matmul(scores, v_win)

    return win_attn


def gated_score_mlp_standard(x, w_1, w_2, output: Path):
    b, s, h = x.shape
    _, n3 = w_2.shape
    n = n3 // 3
    print(f'b {b} s {s} h {h} n {n}\n')

    x_2d = x.reshape(-1, h)
    print(f'torch version {torch.__version__}')
    print(f'矩阵 x_2d:\n {x_2d} \n  {x_2d.shape} \n w_1 \n{w_1} \n {w_1.shape}\n')
    mm1 = torch.matmul(x_2d.to(torch.float32), w_1.to(torch.float32))
    mm1_sigmoid = torch.sigmoid(mm1)
    mm2 = torch.matmul(mm1_sigmoid.to(w_2.dtype).to(torch.float32), w_2.to(torch.float32))
    gating_score = mm2.view(b, s, 3, n)  # 使用 view 替代 reshape
    return gating_score


def gated_score_mlp_simple(x, w_1, output: Path):
    b, s, h = x.shape
    _, n_heads = w_1.shape
    n = n_heads // 3
    x_2d = x.reshape(-1, h)
    mm1 = np.matmul(x_2d, w_1)
    mm1_sigmoid = sigmoid(mm1)
    gating_score = mm1_sigmoid.reshape(b, s, n, 3)

    return gating_score


def gen_gated_score(x, gate_sim_w1, gate_w1, gate_w2, output: Path, mode='standard'):
    if mode == 'standard':
        gating_score = gated_score_mlp_standard(x, gate_w1, gate_w2, output)
    gating_score = gating_score.permute(0, 1, 3, 2)  # 使用 permute 替代 transpose
    h, n = gate_w1.shape[0], gate_w2.shape[1] // 3
    inputDtype = bfloat16 if x.dtype == torch.bfloat16 else np.float16
    # 保存输入和权重
    x_path = output / 'x.bin'
    x.to(torch.float32).cpu().numpy().astype(inputDtype).tofile(x_path)
    gate_sim_w1_path = output / 'gate_sim_w1.bin'
    gate_sim_w1.to(torch.float32).cpu().numpy().astype(inputDtype).tofile(gate_sim_w1_path)
    gate_w1_path = output / 'gate_w1.bin'
    gate_w1.to(torch.float32).cpu().numpy().astype(inputDtype).tofile(gate_w1_path)
    gate_w2_path = output / 'gate_w2.bin'
    gate_w2.to(torch.float32).cpu().numpy().astype(inputDtype).tofile(gate_w2_path)

    gate_w1_nz_path = output / 'gate_w1_nz.bin'
    gate_w1.to(torch.float32).cpu().numpy().reshape(h, 4 * h // 16, 16).transpose(1, 0, 2).astype(inputDtype).tofile(
        gate_w1_nz_path)
    gate_w2_nz_path = output / 'gate_w2_nz.bin'
    gate_w2.to(torch.float32).cpu().numpy().reshape(4 * h, 3 * n // 16, 16).transpose(1, 0, 2).astype(
        inputDtype).tofile(
        gate_w2_nz_path)

    gating_score_fp32_path = output / 'gating_score_fp32.bin'
    gating_score.to(torch.float32).cpu().numpy().astype(np.float32).tofile(gating_score_fp32_path)

    gating_score_path = output / 'gating_score.bin'
    gating_score.to(torch.float32).cpu().numpy().astype(inputDtype).tofile(gating_score_path)
    print(f'gating_score_path: {gating_score_path}')
    return gating_score


def gen_attn(cmp_attn, slc_attn, win_attn, gating_score):
    w_cmp = gating_score[..., 0]
    w_slc = gating_score[..., 1]
    w_win = gating_score[..., 2]

    attention = (
            w_cmp[..., np.newaxis] * cmp_attn +
            w_slc[..., np.newaxis] * slc_attn +
            w_win[..., np.newaxis] * win_attn
    )

    return attention


def nsa(x, q, k, v, avg_wk, avg_wv, mlp_wk1, mlp_wk2, mlp_wv1, mlp_wv2, cos, sin, pos_ids, gate_sim_w1, gate_w1,
        gate_w2, output: Path, params,
        cmp_mode='avg', l_prime=64, l=32, d=16, front=1, near=2, topk=16, win=512, gate_mode='standard'):
    gating_score = gen_gated_score(x, gate_sim_w1, gate_w1, gate_w2, output, mode=gate_mode)

    gating_score_path = Path(output, 'gating_score.bin')
    gating_score.astype(np.float16).tofile(gating_score_path)

    temp_path = Path(output, 'temp.bin')

    mm1_path = Path(output, 'mm1.bin')
    print(f"{mm1_path} {temp_path} {gating_score_path}")

    x_path = Path(output, 'x.bin')
    x.astype(np.float16).tofile(x_path)
    gate_sim_w1_path = Path(output, 'gate_sim_w1.bin')
    gate_sim_w1.astype(np.float16).tofile(gate_sim_w1_path)
    gate_w1_path = Path(output, 'gate_w1.bin')
    gate_w1.astype(np.float16).tofile(gate_w1_path)
    gate_w2_path = Path(output, 'gate_w2.bin')
    gate_w2.astype(np.float16).tofile(gate_w2_path)

    k_cmp, v_cmp = kv_compression(k, v, avg_wk, avg_wv, mlp_wk1, mlp_wk2, mlp_wv1, mlp_wv2,
                                  cos=cos, sin=sin, pos_ids=pos_ids, params=params, l=l, d=d, mode=cmp_mode)
    p_cmp, cmp_attn = gen_cmp_attn(q, k_cmp, v_cmp)
    slc_attn = gen_slc_attn(q, p_cmp, k, v, l_prime=l_prime, l=l, d=d, front=front, near=near, topk=topk)
    win_attn = gen_win_attn(q, k, v, win)

    b, s, h = x.shape
    _, n3 = gate_w2.shape
    n = n3 // 3
    if True:
        gate_w1 = np.random.rand(h, 4 * h)
        gate_w2 = np.random.rand(4 * h, 3 * n)

    attention = gen_attn(cmp_attn, slc_attn, win_attn, gating_score)

    # origin input output

    q_path = Path(output, 'q.bin')
    q.tofile(q_path)
    k_path = Path(output, 'k.bin')
    k.tofile(k_path)
    v_path = Path(output, 'v.bin')
    v.tofile(v_path)
    avg_wk_path = Path(output, 'avg_wk.bin')
    avg_wk.tofile(avg_wk_path)
    avg_wv_path = Path(output, 'avg_wv.bin')
    avg_wv.tofile(avg_wv_path)
    mlp_wk1_path = Path(output, 'mlp_wk1.bin')
    mlp_wk1.tofile(mlp_wk1_path)
    mlp_wk2_path = Path(output, 'mlp_wk2.bin')
    mlp_wk2.tofile(mlp_wk2_path)
    mlp_wv1_path = Path(output, 'mlp_wv1.bin')
    mlp_wv1.tofile(mlp_wv1_path)
    mlp_wv2_path = Path(output, 'mlp_wv2.bin')
    mlp_wv2.tofile(mlp_wv2_path)
    mlp_wv2_path = Path(output, 'mlp_wv2.bin')
    mlp_wv2.tofile(mlp_wv2_path)
    cos_path = Path(output, 'cos.bin')
    cos.tofile(cos_path)
    sin_path = Path(output, 'sin.bin')
    sin.tofile(sin_path)
    pos_ids_path = Path(output, 'pos_ids.bin')
    pos_ids.tofile(pos_ids_path)

    # temp output
    k_cmp_path = Path(output, 'k_cmp.bin')
    k_cmp.tofile(k_cmp_path)
    v_cmp_path = Path(output, 'v_cmp.bin')
    v_cmp.tofile(v_cmp_path)
    p_cmp_path = Path(output, 'p_cmp.bin')
    p_cmp.tofile(p_cmp_path)

    cmp_attn_path = Path(output, 'cmp_attn.bin')
    cmp_attn.tofile(cmp_attn_path)
    slc_attn_path = Path(output, 'slc_attn.bin')
    slc_attn.tofile(slc_attn_path)
    win_attn_path = Path(output, 'win_attn.bin')
    win_attn.tofile(win_attn_path)

    return attention


def gen_uniform_data(shape, low, high, dtype):
    return (high - low) * torch.rand(shape, dtype=dtype) + low


def gen_gate_score_golden(params, dtype):
    b = params.get("b")
    s = params.get("s")
    h = params.get("h")
    n = params.get("n")
    x = gen_uniform_data((b, s, h), -0.1, 0.1, dtype)
    gate_sim_w1 = gen_uniform_data((h, n * 3), -0.1, 0.1, dtype)
    gate_w1 = gen_uniform_data((h, h * 4), -0.1, 0.1, dtype)
    gate_w2 = gen_uniform_data((h * 4, n * 3), -0.1, 0.1, dtype)
    return {
        "x": x,
        "gate_sim_w1": gate_sim_w1,
        "gate_w1": gate_w1,
        "gate_w2": gate_w2
    }


def gen_nsa_golden(params, dtypes, output: Path):
    dtype, w_dtype = dtypes
    logging.debug(f"gen_nsa_golden  dtype:{dtype}, w_dtype:{w_dtype}")
    b = params.get("b")
    s = params.get("s")  # s=1
    s2 = params.get("s2")  # s2=4k
    n = params.get("n")

    l = params.get("l")
    l_prime = params.get("l_prime")
    d = params.get("d")
    front = params.get("front")
    near = params.get("near")
    topk = params.get("topk")
    op = params.get("op")
    gen_topk_actual_len = params.get("gen_topk_actual_len")

    gate_score_input = gen_gate_score_golden(params, dtype)
    x = gate_score_input.get("x")
    gate_sim_w1 = gate_score_input.get("gate_sim_w1")
    gate_w1 = gate_score_input.get("gate_w1")
    gate_w2 = gate_score_input.get("gate_w2")

    if op == "GatingScore":
        gen_gated_score(x, gate_sim_w1, gate_w1, gate_w2, output, 'standard')
    elif op == "GenSlc" or op == "GenTop":
        s_cmp_len = (s2 - l) // d + 1
        p_cmp = np.random.rand(b, n, s, s_cmp_len)
        s_slc = (s_cmp_len + 3) // 4
        p_slc = gen_p_slc(p_cmp, l_prime=l_prime, l=l, d=d)
        trans0, reduce0, trans1, reduce1 = gen_p_slc_ast(p_cmp, l_prime=l_prime, l=l, d=d)
        reduce1 = reduce1.reshape(1, 1, s_slc)
        tmp_s_smp = (gen_topk_actual_len - 32) // 16 + 1
        tmp_s_slc = (tmp_s_smp + 3) // 4

        topk_indices, _ = gen_topk_indices(reduce1, front=front, near=near, topk=topk, actual_len=tmp_s_slc)
        p_cmp_path = Path(output, 'p_cmp.bin')
        p_cmp.astype(dtype).tofile(p_cmp_path)
        topk_indices_path = Path(output, 'topk_indices.bin')
        topk_indices.astype(np.float32).tofile(topk_indices_path)
        trans0_path = Path(output, 'trans0.bin')
        trans0.astype(dtype).tofile(trans0_path)
        reduce0_path = Path(output, 'reduce0.bin')
        reduce0.astype(dtype).tofile(reduce0_path)
        trans1_path = Path(output, 'trans1.bin')
        trans1.astype(dtype).tofile(trans1_path)
        reduce1_path = Path(output, 'reduce1.bin')
        reduce1.astype(dtype).tofile(reduce1_path)


def nsa_entry(dtypes, bs1s2, op, output_dir: Path, gen_topk_actual_len=0):
    b, s1, s2, h = bs1s2
    kv_lora_rank = 512
    rope_dim = 64
    q_dim = kv_lora_rank + rope_dim
    k_dim = kv_lora_rank + rope_dim
    v_dim = kv_lora_rank
    params = {
        "b": b,
        "s": s1,
        "s2": s2,
        "h": h,
        "n": 128,
        "q_lora_rank": 1536,
        "qk_nope_head_dim": 128,
        "qk_rope_head_dim": 64,
        "kv_lora_rank": 512,
        "v_head_dim": v_dim,
        "l": 32,
        "d": 16,
        "l_prime": 64,
        "rope_dim": 64,
        "q_dim": q_dim,
        "k_dim": k_dim,
        "v_dim": v_dim,
        "st_test_flag": True,
        "op": op,
        "front": 1,
        "near": 2,
        "topk": 16,
        "actual_seq": s2,
        "gen_topk_actual_len": gen_topk_actual_len,
    }
    gen_nsa_golden(params, dtypes, output_dir)


def dviewPad(output_dir: Path):
    shape0, shape1 = 1, 128
    input = np.arange(0, shape0 * shape1, 1).reshape(shape0, shape1).astype(np.float32)
    output = input[:, 1:14]
    input_path = Path(output_dir, 'input.bin')
    input.tofile(input_path)
    output_path = Path(output_dir, 'output.bin')
    output.tofile(output_path)


@GoldenRegister.reg_golden_func(
    case_names=[
        "DyNsa.gateScore_mini",
        "DyNsa.gateScore_mini_mtp",
        "DyNsa.gateScore_mini_mtp_bf16",
        "DyNsa.GateScore_b16_s1_fp",
        "DyNsa.GateScore_b16_s1_bf",
        "DyNsa.GateScore_b32_s1_fp",
        "DyNsa.GateScore_b32_s2_fp",
        "DyNsa.GateScore_b24_s1_fp",
        "DyNsa.GateScore_b48_s2_fp",
        "DyNsa.GateScore_b32_s2_bf",
        "DyNsa.GateScore_b48_s1_fp",

        "DyNsa.GenSlc_b1_s1_fp_4k",
        "DyNsa.GenSlc_b1_s1_fp_6k1",
        "DyNsa.GenSlc_b1_s1_fp_4k1",
        "DyNsa.GenSlc_b1_s1_fp_8k",
        "DyNsa.TestView",
        "DyNsa.TestAlignRead",
        "DyNsa.TestUnAlignRead",
        "DyNsa.TestMultiLoopAlignRead",
        "DyNsa.GenTopk_b1_s1_fp_8k",
        "DyNsa.GenTopk_b1_s1_fp_4k",
        "DyNsa.GenTopk_b1_s1_fp_4k1",
        "DyNsa.GenTopk_b1_s1_fp_6k1",
        "DyNsa.GenTopk_b1_s1_fp_8k_dyn",
        "DyNsa.GenTopk_b1_s1_fp_4k_dyn",
        "DyNsa.GenTopk_b1_s1_fp_4k1_dyn",
        "DyNsa.GenTopk_b1_s1_fp_6k1_dyn",
        "DyNsa.GenSlc_b1_s1_bf_1k1"

    ]
)
def gen_mla_prolog_date_v2(case_name: str, output: Path) -> bool:
    p_cmp_path = Path(output, 'p_cmp.bin')
    topk_indices_path = Path(output, 'topk_indices.bin')
    trans0_path = Path(output, 'trans0.bin')
    reduce0_path = Path(output, 'reduce0.bin')
    trans1_path = Path(output, 'trans1.bin')
    reduce1_path = Path(output, 'reduce1.bin')
    complete = (p_cmp_path.exists() and topk_indices_path.exists() and trans0_path.exists() and
        reduce0_path.exists() and trans1_path.exists() and reduce1_path.exists())

    if complete:
        logging.info("Case(%s), Golden data exits. cache catch", case_name)
    else:
        if case_name == "DyNsa.GateScore_b16_s1_fp":
            nsa_entry((np.float16, np.float16), (16, 1, 65536, 7168), "GatingScore", output)
        elif case_name == "DyNsa.GateScore_b16_s1_bf":
            nsa_entry((bfloat16, bfloat16), (16, 1, 65536, 7168), "GatingScore", output)
        elif case_name == "DyNsa.GateScore_b32_s1_fp":
            nsa_entry((np.float16, np.float16), (32, 1, 65536, 7168), "GatingScore", output)
        elif case_name == "DyNsa.GateScore_b32_s2_fp":
            nsa_entry((np.float16, np.float16), (32, 2, 65536, 7168), "GatingScore", output)
        elif case_name == "DyNsa.GateScore_b24_s1_fp":
            nsa_entry((np.float16, np.float16), (24, 1, 65536, 7168), "GatingScore", output)
        elif case_name == "DyNsa.GateScore_b48_s2_fp":
            nsa_entry((np.float16, np.float16), (48, 2, 65536, 7168), "GatingScore", output)
        elif case_name == "DyNsa.GateScore_b32_s2_bf":
            nsa_entry((torch.bfloat16, torch.bfloat16), (32, 2, 65536, 7168), "GatingScore", output)
        elif case_name == "DyNsa.GateScore_b48_s1_fp":
            nsa_entry((torch.float16, torch.float16), (48, 1, 65536, 7168), "GatingScore", output)
        elif case_name == "DyNsa.gateScore_mini":
            nsa_entry((np.float16, np.float16), (32, 1, 65536, 128), "GatingScore", output)
        elif case_name == "DyNsa.gateScore_mini_16":
            nsa_entry((np.float16, np.float16), (16, 1, 65536, 128), "GatingScore", output)
        elif case_name == "DyNsa.gateScore_mini_mtp":
            nsa_entry((np.float16, np.float16), (32, 2, 65536, 128), "GatingScore", output)
        elif case_name == "DyNsa.gateScore_mini_mtp_bf16":
            nsa_entry((bfloat16, bfloat16), (32, 2, 65536, 128), "GatingScore", output)
        elif case_name == "DyNsa.GenSlc_b1_s1_fp_8k":
            nsa_entry((np.float16, np.float16), (1, 1, 8192, 128), "GenSlc", output, 8192)
        elif case_name == "DyNsa.GenSlc_b1_s1_fp_4k":
            nsa_entry((np.float16, np.float16), (1, 1, 8192, 128), "GenSlc", output, 4096)
        elif case_name == "DyNsa.GenSlc_b1_s1_fp_6k1":
            nsa_entry((np.float16, np.float16), (1, 1, 8192, 128), "GenSlc", output, 6145)
        elif case_name == "DyNsa.GenSlc_b1_s1_fp_4k1":
            nsa_entry((np.float16, np.float16), (1, 1, 8192, 128), "GenSlc", output, 4097)
        elif case_name == "DyNsa.GenSlc_b1_s1_bf_1k1":
            nsa_entry((bfloat16, bfloat16), (1, 1, 8192, 128), "GenSlc", output, 1025)


        elif case_name == "DyNsa.GenTopk_b1_s1_fp_8k" or case_name == "DyNsa.GenTopk_b1_s1_fp_8k_dyn":
            nsa_entry((np.float16, np.float16), (1, 1, 8192, 128), "GenTop", output, 8192)
        elif case_name == "DyNsa.GenTopk_b1_s1_fp_4k" or case_name == "DyNsa.GenTopk_b1_s1_fp_4k_dyn":
            nsa_entry((np.float16, np.float16), (1, 1, 8192, 128), "GenTop", output, 4096)
        elif case_name == "DyNsa.GenTopk_b1_s1_fp_4k1" or case_name == "DyNsa.GenTopk_b1_s1_fp_4k1_dyn":
            nsa_entry((np.float16, np.float16), (1, 1, 8192, 128), "GenTop", output, 4097)
        elif case_name == "DyNsa.GenTopk_b1_s1_fp_6k1" or case_name == "DyNsa.GenTopk_b1_s1_fp_6k1_dyn":
            nsa_entry((np.float16, np.float16), (1, 1, 8192, 128), "GenTop", output, 6145)
        elif case_name == "DyNsa.TestView" or case_name == "DyNsa.TestAlignRead" or case_name == "DyNsa.TestUnAlignRead" or case_name == "DyNsa.TestMultiLoopAlignRead":
            dviewPad(output)
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
        "DyNsa.GateScore_b32_s2_bf",
    ]
    # 函数调用
    ret: bool = True
    for cs in case_name_list:
        output_dir: Path = Path(g_src_root, "build/output/bin/golden", cs).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        ret = gen_mla_prolog_date_v2(case_name=cs, output=output_dir)
    return ret


if __name__ == "__main__":
    exit(0 if main() else 1)
