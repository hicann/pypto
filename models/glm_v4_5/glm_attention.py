#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
GLM-4.5 Attention Module

This module implements the Attention mechanism for GLM-4.5 model, which uses
a paged memory management approach similar to operating systems to efficiently
handle variable-length sequences and dynamic batch sizes in attention computation.

Main Functions:
    - attention: Main attention function with Attention support
    - ifa_func: JIT compiled kernel implementing Flash Attention with paged KV cache
    - gen_block_table: Generate block mapping table for Attention
    - kv_cache_concat_bsnd: Convert paged KV cache to BSND format
"""
import os
import math
from dataclasses import dataclass
import torch
import torch_npu
import pytest
import numpy as np
from torch._subclasses.fake_tensor import FakeTensor
from torch._dynamo import allow_in_graph
import pypto
from utils.get_format import get_format
from utils.np_compare import detailed_allclose_manual as compare

np.random.seed(0)
torch.manual_seed(0)
np.set_printoptions(formatter={'float': '{:.6f}'.format})


def check_args(
    query,
    key_cache,
    value_cache,
    block_tables,
    actual_seqs,
    attn_res
):
    assert query.dim() == 3
    assert get_format(query) == 'ND'
    assert query.dtype == torch.bfloat16
    assert key_cache.dim() == 4
    assert get_format(key_cache) == 'ND'
    assert key_cache.dtype == torch.bfloat16
    assert value_cache.dim() == 4
    assert get_format(value_cache) == 'ND'
    assert value_cache.dtype == torch.bfloat16
    assert block_tables.dim() == 2
    assert get_format(block_tables) == 'ND'
    assert block_tables.dtype == torch.int32
    assert actual_seqs.dim() == 1
    assert get_format(actual_seqs) == 'ND'
    assert actual_seqs.dtype == torch.int32
    assert attn_res.dim() == 3
    assert get_format(attn_res) == 'ND'
    assert attn_res.dtype == torch.bfloat16


@dataclass
class AttentionTileConfig:
    g_tile: int = 12
    s2_tile: int = 512
    c1_tile_shape: list = None
    v1_tile_shape: list = None
    c2_tile_shape: list = None
    v2_tile_shape: list = None


global_tile_config = AttentionTileConfig()


@dataclass
class AttentionConfig:
    b: int = 8
    s1: int = 1
    s2: int = 16384
    n1: int = 12
    n2: int = 1
    q_d: int = 128
    kv_d: int = 128
    block_size: int = 128
    max_num_blocks_per_query: int = 0
    softmax_scale: float = 1.0
    kv_layout: str = "PA_BSND"
    actual_seq: torch.Tensor = None  # 改为 torch.Tensor 类型
    block_table_batch: int = 0
    kv_num_blocks: int = 0


global_config = AttentionConfig()


def set_qwen_common_config(case_950=0, b=8, s1=1, s2=16384):
    global global_tile_config
    global global_config
    device_id = os.environ.get('TILE_FWK_DEVICE_ID', 0)
    device = f'npu:{device_id}'
    b = b
    s1 = s1
    s2 = s2
    q_d = 128
    nq = 12
    nkv = 1
    kv_layout = "PA_BSND"
    softmax_scale = q_d ** -0.5
    block_table_batch = b
    block_size = 128
    kv_num_blocks = b * ((s2 + block_size - 1) // block_size)
    cube_tile = 128
    m_tile = 128
    s2_tile = 512
    if case_950 == 1:
        b = b
        s1 = s1
        s2 = s2
        q_d = 128
        nq = 12
        nkv = 1
        kv_layout = "PA_BSND"
        softmax_scale = q_d ** -0.5
        block_table_batch = b
        block_size = 128
        kv_num_blocks = b * ((s2 + block_size - 1) // block_size)
        cube_tile = 128
        m_tile = 128
        s2_tile = 1024

    # 创建 torch tensor 类型的 actual_seq
    actual_seq_values = [s2] * b
    actual_seq_tensor = torch.tensor(actual_seq_values, dtype=torch.int32, device=device)

    atten_cfg = AttentionConfig(b=b, s1=s1, s2=s2, n1=nq, n2=nkv, softmax_scale=softmax_scale, kv_layout=kv_layout,
                                q_d=q_d, kv_d=q_d, block_size=block_size, block_table_batch=block_table_batch,
                                kv_num_blocks=kv_num_blocks, actual_seq=actual_seq_tensor)  # 传入 tensor
    atten_cfg.max_num_blocks_per_query = (s2 + block_size - 1) // block_size

    tile_cfg = AttentionTileConfig(
        nq,
        s2_tile,
        [[m_tile, m_tile], [cube_tile, cube_tile], [cube_tile, cube_tile]],
        [m_tile, s2_tile],
        [[m_tile, m_tile], [cube_tile, cube_tile], [cube_tile, cube_tile]],
        [m_tile, cube_tile])
    global_config = atten_cfg
    global_tile_config = tile_cfg


def get_common_config():
    return global_config, global_tile_config


def gen_block_table(actual_seq_len, block_size, block_table_shape):
    block_num_per_batch = []
    block_num = 0

    # 处理 torch tensor 类型的 actual_seq_len
    if isinstance(actual_seq_len, torch.Tensor):
        # 如果 tensor 在 GPU/NPU 上，先移动到 CPU
        if actual_seq_len.device.type != 'cpu':
            actual_seq_len_cpu = actual_seq_len.cpu()
        else:
            actual_seq_len_cpu = actual_seq_len

        # 转换为 numpy 数组进行处理，或者直接使用 torch 操作
        for actual_seq in actual_seq_len_cpu:
            block_num_per_batch.append(math.ceil(actual_seq.item() / block_size))
            block_num += math.ceil(actual_seq.item() / block_size)
    else:
        # 保持对 list 的兼容
        for actual_seq in actual_seq_len:
            block_num_per_batch.append(math.ceil(actual_seq / block_size))
            block_num += math.ceil(actual_seq / block_size)

    # 使用 torch 替换 numpy
    block_idx_list = torch.arange(0, block_num, dtype=torch.int32)
    block_idx_list = block_idx_list[torch.randperm(block_idx_list.size(0))]  # 随机排列

    # 创建 block_table 张量
    block_table = torch.full(block_table_shape, -1, dtype=torch.int32)
    block_idx = 0
    block_table_batch_idx = 0
    for idx in block_num_per_batch:
        for j in range(idx):
            block_table[block_table_batch_idx][j] = block_idx_list[block_idx]
            block_idx += 1
        block_table_batch_idx += 1
    return block_table


def kv_cache_concat_bsnd(kr_cache_out, kv_cache_out, block_table, atten_config):
    b = atten_config.b
    n2 = atten_config.n2
    kv_lora_rank = atten_config.q_d
    rope_dim = atten_config.kv_d
    block_size = atten_config.block_size
    kv_cache_actual_seq = atten_config.actual_seq
    dtype = kv_cache_out.dtype

    # 处理 torch tensor 类型的 kv_cache_actual_seq
    if isinstance(kv_cache_actual_seq, torch.Tensor):
        if kv_cache_actual_seq.device.type != 'cpu':
            kv_cache_actual_seq_cpu = kv_cache_actual_seq.cpu()
        else:
            kv_cache_actual_seq_cpu = kv_cache_actual_seq
        kv_max = (torch.max(kv_cache_actual_seq_cpu).item() + block_size - 1) // block_size * block_size
    else:
        kv_max = (max(kv_cache_actual_seq) + block_size - 1) // block_size * block_size

    # 使用 torch 创建张量，保持在同一设备上
    device = kr_cache_out.device
    k_cache = torch.zeros([b, kv_max, n2, kv_lora_rank], dtype=dtype, device=device)
    v_cache = torch.zeros([b, kv_max, n2, rope_dim], dtype=dtype, device=device)

    for b_idx in range(b):
        block_list = block_table[b_idx]
        kv_nope_temp_tensor = torch.zeros([1, kv_max, n2, kv_lora_rank], dtype=dtype, device=device)
        kv_rope_temp_tensor = torch.zeros([1, kv_max, n2, rope_dim], dtype=dtype, device=device)
        s_idx = 0

        for _, block_idx in enumerate(block_list):
            if block_idx == -1:
                break
            # 使用 torch 的切片操作
            start_idx = s_idx * block_size
            end_idx = (s_idx + 1) * block_size

            kv_nope_temp_tensor[:, start_idx:end_idx, :, :] = kv_cache_out[block_idx:block_idx + 1, :, :, :]
            kv_rope_temp_tensor[:, start_idx:end_idx, :, :] = kr_cache_out[block_idx:block_idx + 1, :, :, :]
            s_idx += 1

        v_cache[b_idx:b_idx + 1, :, :, :] = kv_nope_temp_tensor
        k_cache[b_idx:b_idx + 1, :, :, :] = kv_rope_temp_tensor

    return k_cache, v_cache


def get_special_array(m, n):
    q_shape = [m, n]

    # 生成递增的行值
    base = np.arange(1, m + 1)  # 生成 [1, 2, ..., m]

    # 将 base 扩展到二维形状 [m, n]
    q = base[:, np.newaxis]  # 增加一个新维度，形状变为 [m, 1]
    q = np.broadcast_to(q, q_shape)  # 广播到目标形状 [m, n]

    # 转换为 float16 类型
    q = q.astype(np.float16)
    return q


def softmax(x, is_fp16=False):
    # 使用 torch 的 softmax 实现
    if is_fp16:
        original_dtype = x.dtype
        x = x.float()
    x_max = x.max(dim=-1, keepdim=True).values
    x_sub = x - x_max
    y = torch.exp(x_sub)
    x_sum = y.sum(dim=-1, keepdim=True)
    ans = y / x_sum
    if is_fp16:
        ans = ans.to(original_dtype)
        x_max = x_max.to(original_dtype)
        x_sum = x_sum.to(original_dtype)

    return ans, x_max, x_sum


@pypto.frontend.jit(
    runtime_options={"stitch_function_max_num": 128},
    # 当子图大小达到上界不允许与其他子图合并
    pass_options={
    # Q常驻，0代表第一组mmad，4代表4次matmul合并
    "cube_l1_reuse_setting": {0: 4}},
    host_options={"compile_monitor_enable": True,
        "compile_timeout": 22,
        "compile_timeout_stage": 5,
        "compile_monitor_print_interval": 60},
    debug_options={"runtime_debug_mode": 1, "compile_debug_mode": 0}
)
def ifa_func_kernel(
    q: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_BF16),
    k: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_BF16),
    v: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_BF16),
    block_table: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_INT32),
    kv_act_seqs: pypto.Tensor([pypto.DYNAMIC], pypto.DT_INT32),
    atten_out: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_BF16)
):
    # 1. 添加支持动态的config
    pypto.experimental.set_operation_options(combine_axis=True)
    atten_cfg, tile_cfg = get_common_config()
    if tile_cfg.c1_tile_shape is None:
        set_qwen_common_config()
        atten_cfg, tile_cfg = get_common_config()
    softmax_scale = atten_cfg.softmax_scale

    # 2. 从入参拿到输入和输出tensor
    shape_q = q.shape
    shape_k = k.shape
    bs_scalar = shape_q[0]
    nq = shape_q[1]
    block_num_scalar = shape_k[0]
    block_size = shape_k[1]
    nkv = shape_k[2]
    dn = shape_k[3]
    b_scalar = kv_act_seqs.shape[0]

    dtype = q.dtype
    group = nq // nkv
    n2_sym = nkv

    g_tile = tile_cfg.g_tile
    s2_tile = tile_cfg.s2_tile
    c1_tile = tile_cfg.c1_tile_shape
    v1_tile = tile_cfg.v1_tile_shape
    c2_tile = tile_cfg.c2_tile_shape
    v2_tile = tile_cfg.v2_tile_shape

    # 3. 得到动态tensor的shape
    s1_scalar = bs_scalar // b_scalar
    g = nq // nkv
    g_loop = g // g_tile

    k_2d_shape = (block_num_scalar * block_size, n2_sym * dn)
    q_2d_shape = (b_scalar * s1_scalar * nq, dn)

    k_2d = pypto.reshape(k, k_2d_shape, inplace=True)
    v_2d = pypto.reshape(v, k_2d_shape, inplace=True)
    q_2d = pypto.reshape(q, q_2d_shape, inplace=True)
    # 4. 实现kernel逻辑，循环展开B动态轴
    for b_idx in pypto.loop(b_scalar, name="LOOP_b", idx_name="b_idx"):
        for s1_idx in pypto.loop(s1_scalar, name="LOOP_s1", idx_name="s1_idx"):
            cur_seq = kv_act_seqs[b_idx] - (s1_scalar - 1 - s1_idx)
            s2_loop = (cur_seq + s2_tile - 1) // s2_tile
            for n2_idx in pypto.loop(n2_sym, name="LOOP_n2", idx_name="n2_idx"):
                for g_idx in pypto.loop(g_loop, name="LOOP_g", idx_name="g_idx"):
                    oi_update = pypto.tensor([g_tile, dn], pypto.DT_FP32, "oi_update")
                    sum_update = pypto.tensor([g_tile, 1], pypto.DT_FP32, "sum_update")
                    max_update = pypto.tensor([g_tile, 1], pypto.DT_FP32, "max_update")
                    for s2_idx in pypto.loop(s2_loop, name="LOOP_s2", idx_name="s2_idx", unroll_list=[8, 4, 2, 1]):
                        block_num = s2_tile // block_size
                        idx = s2_idx * block_num
                        bs_ofs = b_idx * s1_scalar + s1_idx
                        n1g_ofs = n2_idx * group + g_idx * g_tile
                        actual_s2_tile = (cur_seq - s2_idx * s2_tile).min(s2_tile)
                        oi_ofs = [bs_ofs, n1g_ofs, 0]
                        # 5. 按照计算图实现运算逻辑，设置set_vec_tile_shapes时应尽可能用满UB，但不要超过UB的大小。
                        pypto.set_vec_tile_shapes(v1_tile[0], v1_tile[1])
                        qi = pypto.view(q_2d, [g_tile, dn], [bs_ofs * nq + n1g_ofs, 0])

                        kj_assemble = pypto.tensor([s2_tile, dn], k_2d.dtype, "kj_assemble")
                        for i in range(block_num):
                            block_idx = block_table[b_idx, idx + i]
                            block_idx_valid = block_idx.max(0)
                            kj_assemble[i * block_size:(i + 1) * block_size, 0:] = \
                                pypto.view(k_2d, [block_size, dn], [block_idx_valid * block_size, 0])
                        kj_assemble = pypto.view(kj_assemble, [s2_tile, dn], [0, 0], valid_shape=[s2_tile, dn])

                        # c1
                        # 6. 下面是flash attention的计算逻辑
                        pypto.set_cube_tile_shapes(c1_tile[0], c1_tile[1], c1_tile[2])
                        sij = pypto.matmul(qi, kj_assemble, pypto.DT_FP32, a_trans=False,
                                            b_trans=True)
                        sij = pypto.view(sij, [g_tile, s2_tile], [0, 0],
                                            valid_shape=[g_tile, actual_s2_tile])
                        # v1
                        pypto.set_vec_tile_shapes(v1_tile[0], v1_tile[1])
                        if pypto.is_loop_begin(s2_idx):
                            sij_scale = pypto.mul(sij, softmax_scale)
                            tilda_mij = pypto.amax(sij_scale, dim=-1, keepdim=True)

                            tsub = pypto.sub(sij_scale, tilda_mij)
                            tilda_pij = pypto.exp(tsub)
                            tilda_pij_fp16 = pypto.cast(tilda_pij, dtype)
                            sum_update[:] = pypto.sum(tilda_pij, dim=-1, keepdim=True)
                            max_update[:] = tilda_mij

                            # c2
                            vj_assemble = pypto.tensor([s2_tile, dn], v_2d.dtype, "vj_assemble")
                            for i in range(block_num):
                                block_idx = block_table[b_idx, idx + i]
                                block_idx_valid = block_idx.max(0)
                                vj_assemble[i * block_size:(i + 1) * block_size, 0:] = \
                                    pypto.view(v_2d, [block_size, dn], [block_idx_valid * block_size, 0])
                            vj_assemble = pypto.view(vj_assemble, [s2_tile, dn],
                                                        [0, 0], valid_shape=[actual_s2_tile, dn])
                            pypto.set_cube_tile_shapes(c2_tile[0], c2_tile[1], c2_tile[2])
                            oi_tmp = pypto.matmul(tilda_pij_fp16, vj_assemble, pypto.DT_FP32)

                            pypto.set_vec_tile_shapes(v2_tile[0], v2_tile[1])
                            oi_update[:] = oi_tmp
                        else:
                            pypto.set_pass_options(sg_set_scope=1)
                            sij_scale = pypto.mul(sij, softmax_scale)
                            tilda_mij = pypto.amax(sij_scale, dim=-1, keepdim=True)
                            max_new = pypto.maximum(max_update, tilda_mij)
                            tsub = pypto.sub(sij_scale, max_new)
                            tilda_pij = pypto.exp(tsub)
                            tilda_pij_fp16 = pypto.cast(tilda_pij, dtype)
                            sum_local = pypto.sum(tilda_pij, dim=-1, keepdim=True)
                            pypto.set_pass_options(sg_set_scope=-1)

                            pypto.set_pass_options(sg_set_scope=2)
                            tsub2 = pypto.sub(max_update, max_new)
                            max_update[:] = max_new
                            update_mul = pypto.exp(tsub2)
                            sum_update[:] = sum_update * update_mul + sum_local
                            pypto.set_pass_options(sg_set_scope=-1)

                            # c2
                            vj_assemble = pypto.tensor([s2_tile, dn], v_2d.dtype, "vj_assemble")
                            for i in range(block_num):
                                block_idx = block_table[b_idx, idx + i]
                                block_idx_valid = block_idx.max(0)
                                vj_assemble[i * block_size:(i + 1) * block_size, 0:] = \
                                    pypto.view(v_2d, [block_size, dn], [block_idx_valid * block_size, 0])
                            vj_assemble = pypto.view(vj_assemble, [s2_tile, dn],
                                                        [0, 0], valid_shape=[actual_s2_tile, dn])
                            pypto.set_cube_tile_shapes(c2_tile[0], c2_tile[1], c2_tile[2])
                            oi_tmp = pypto.matmul(tilda_pij_fp16, vj_assemble, pypto.DT_FP32)

                            # v2
                            pypto.set_vec_tile_shapes(v2_tile[0], v2_tile[1])
                            oi_update[:] = oi_update * update_mul + oi_tmp
                        if pypto.is_loop_end(s2_idx):
                            oi_final = pypto.div(oi_update, sum_update, precision_type=pypto.DivAlgorithm.INTRINSIC)
                            pypto.set_vec_tile_shapes(16, v2_tile[0], v2_tile[1])
                            oi_final_3d = pypto.cast(
                                pypto.reshape(oi_final, [1, g_tile, dn]),
                                dtype)
                            # 7. 将结果搬运到输出tensor上
                            pypto.assemble(oi_final_3d, oi_ofs, atten_out)


@pypto.frontend.jit(
    runtime_options={"stitch_function_max_num": 1024, "device_sched_mode": 1},
    # 当子图大小达到上界不允许与其他子图合并
    pass_options={
    # Q常驻，0代表第一组mmad，4代表4次matmul合并
    "cube_l1_reuse_setting": {-1: 8},
    "cube_nbuffer_setting": {-1: 4}
    },
    host_options={"compile_monitor_enable": True,
        "compile_timeout": 75,
        "compile_timeout_stage": 30,
        "compile_monitor_print_interval": 60},
    debug_options={"runtime_debug_mode": 1, "compile_debug_mode": 0}
)
def ifa_func_kernel_for_950(
    q: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_BF16),
    k: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_BF16),
    v: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_BF16),
    block_table: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_INT32),
    kv_act_seqs: pypto.Tensor([pypto.DYNAMIC], pypto.DT_INT32),
    atten_out: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_BF16)
):
    pypto.experimental.set_operation_options(combine_axis=True)
    atten_cfg, tile_cfg = get_common_config()
    if tile_cfg.c1_tile_shape is None:
        set_qwen_common_config(case_950=1, b=16, s1=1, s2=8192)
        atten_cfg, tile_cfg = get_common_config()
    softmax_scale = atten_cfg.softmax_scale

     # 2. 从入参拿到输入和输出tensor
    shape_q = q.shape
    shape_k = k.shape
    shape_act_seqs = kv_act_seqs.shape
    bs_scalar = shape_q[0]
    nq = shape_q[1]
    block_num_scalar = shape_k[0]
    block_size = shape_k[1]
    nkv = shape_k[2]
    dn = shape_k[3]
    b_scalar = shape_act_seqs[0]

    dtype = q.dtype
    group = nq // nkv
    n2_sym = nkv

    g_tile = tile_cfg.g_tile
    s2_tile = tile_cfg.s2_tile
    c1_tile = tile_cfg.c1_tile_shape
    v1_tile = tile_cfg.v1_tile_shape
    c2_tile = tile_cfg.c2_tile_shape
    v2_tile = tile_cfg.v2_tile_shape

    # 3. 得到动态tensor的shape
    s1_scalar = bs_scalar // b_scalar
    g = nq // nkv
    g_loop = g // g_tile

    k_2d_shape = (block_num_scalar * block_size, n2_sym * dn)
    q_2d_shape = (b_scalar * s1_scalar * nq, dn)

    k_2d = pypto.reshape(k, k_2d_shape, inplace=True)
    v_2d = pypto.reshape(v, k_2d_shape, inplace=True)
    q_2d = pypto.reshape(q, q_2d_shape, inplace=True)
    # 6. 实现kernel逻辑，循环展开B动态轴
    for b_idx in pypto.loop(b_scalar, name="LOOP_b", idx_name="b_idx"):
        for s1_idx in pypto.loop(s1_scalar, name="LOOP_s1", idx_name="s1_idx"):
            cur_seq = kv_act_seqs[b_idx] - (s1_scalar - 1 - s1_idx)
            s2_loop = (cur_seq + s2_tile - 1) // s2_tile
            for n2_idx in pypto.loop(n2_sym, name="LOOP_n2", idx_name="n2_idx"):
                for g_idx in pypto.loop(g_loop, name="LOOP_g", idx_name="g_idx"):
                    oi_update = pypto.tensor([g_tile, dn], pypto.DT_FP32, "oi_update")
                    sum_update = pypto.tensor([g_tile, 1], pypto.DT_FP32, "sum_update")
                    max_update = pypto.tensor([g_tile, 1], pypto.DT_FP32, "max_update")
                    for s2_idx in pypto.loop(s2_loop, name="LOOP_s2", idx_name="s2_idx",
                                              unroll_list=[8, 4, 2, 1]):
                        block_num = s2_tile // block_size
                        idx = s2_idx * block_num
                        bs_ofs = b_idx * s1_scalar + s1_idx
                        n1g_ofs = n2_idx * group + g_idx * g_tile
                        actual_s2_tile = (cur_seq - s2_idx * s2_tile).min(s2_tile)
                        oi_ofs = [bs_ofs, n1g_ofs, 0]
                        # 7. 按照计算图实现运算逻辑，设置set_vec_tile_shapes时应尽可能用满UB，但不要超过UB的大小。
                        pypto.set_vec_tile_shapes(v1_tile[0], v1_tile[1])
                        # 8. 通过view得到tile_q
                        qi = pypto.view(q_2d, [g_tile, dn], [bs_ofs * nq + n1g_ofs, 0])
                        kj_assemble = pypto.tensor([s2_tile, dn], k_2d.dtype, "kj_assemble")
                        for i in range(block_num):
                            block_idx = block_table[b_idx, idx + i]
                            block_idx_vaild = block_idx.max(0)
                            kj_assemble[i * block_size: (i + 1) * block_size, 0:] = pypto.view(k_2d,
                                [block_size, dn], [block_idx_vaild * block_size, 0])
                        kj_assemble = pypto.view(kj_assemble, [s2_tile, dn], [0, 0],
                                                valid_shape=[s2_tile, dn])

                        # c1
                        # 9. 下面是flash attention的计算逻辑  m 128  k=128  n=128
                        pypto.set_cube_tile_shapes(c1_tile[0], c1_tile[1], c1_tile[2])
                        pypto.set_pass_options(sg_set_scope=5001)
                        sij = pypto.matmul(qi, kj_assemble, pypto.DT_FP32, a_trans=False, b_trans=True)
                        # 后续开启 pypto.set_pass_options(sg_set_scope=-1)
                        sij = pypto.view(sij, [g_tile, s2_tile], [0, 0],
                                valid_shape=[g_tile, actual_s2_tile])
                        # v1
                        pypto.set_vec_tile_shapes(v1_tile[0], v1_tile[1])
                        # 后续开启 pypto.set_pass_options(sg_set_scope=5001)
                        sij_scale = pypto.mul(sij, softmax_scale)
                        amax_ij = pypto.amax(sij_scale, dim=-1, keepdim=True)
                        tsub = pypto.sub(sij_scale, amax_ij)
                        vec1_res = pypto.exp(tsub)
                        vec1_res_fp16 = pypto.cast(vec1_res, dtype)
                        sum_local = pypto.sum(vec1_res, dim=-1, keepdim=True)
                        # 后续开启 pypto.set_pass_options(sg_set_scope=-1)

                        #c2
                        vj_assemble = pypto.tensor([s2_tile, dn], v_2d.dtype, "vj_assemble")
                        for i in range(block_num):
                            block_idx = block_table[b_idx, idx + i]
                            block_idx_vaild = block_idx.max(0)
                            vj_assemble[i * block_size: (i + 1) * block_size, 0:] = pypto.view(v_2d,
                                [block_size, dn], [block_idx_vaild * block_size, 0])
                        vj_assemble = pypto.view(vj_assemble, [s2_tile, dn], [0, 0],
                                        valid_shape=[actual_s2_tile, dn])
                        # 后续开启 pypto.set_pass_options(sg_set_scope=5001)
                        pypto.set_cube_tile_shapes(c2_tile[0], c2_tile[1], c2_tile[2])
                        mm2_res = pypto.matmul(vec1_res_fp16, vj_assemble, pypto.DT_FP32)
                        pypto.set_pass_options(sg_set_scope=-1)

                        # # v2
                        if pypto.is_loop_begin(s2_idx):
                            pypto.set_pass_options(sg_set_scope=2)
                            pypto.set_vec_tile_shapes(v2_tile[0], v2_tile[1])
                            oi_tmp = mm2_res
                            oi_update[:] = pypto.tensor(oi_tmp.shape, pypto.DT_FP32, "oi_update")
                            if pypto.is_loop_end(s2_idx):
                                oi_update[:] = pypto.div(oi_tmp, sum_local, precision_type=pypto.DivAlgorithm.INTRINSIC)
                                pypto.set_vec_tile_shapes(16, v2_tile[0], v2_tile[1])
                                oi_update_3d = pypto.cast(pypto.reshape(oi_update, [1, g_tile, dn]),
                                                        dtype)
                                # 10. 将结果搬运到输出tensor上
                                pypto.assemble(oi_update_3d, oi_ofs, atten_out)
                            else:
                                oi_update[:] = oi_tmp
                                sum_update[:] = sum_local
                                max_update[:] = amax_ij
                            pypto.set_pass_options(sg_set_scope=-1)
                        else:
                            pypto.set_pass_options(sg_set_scope=1)
                            pypto.set_vec_tile_shapes(v2_tile[0], 128)
                            max_new = pypto.maximum(max_update, amax_ij)
                            t1 = pypto.sub(max_update, max_new)
                            t2 = pypto.exp(t1)
                            t6 = pypto.mul(t2, sum_update)
                            t3 = pypto.sub(amax_ij, max_new)
                            t4 = pypto.exp(t3)
                            t5 = pypto.mul(t4, sum_local)
                            sum_new = pypto.add(t6, t5)
                            sum_update[:] = sum_new
                            max_update[:] = max_new

                            pypto.set_vec_tile_shapes(v2_tile[0], 128)
                            oi_last = pypto.mul(oi_update, t2)
                            oi_flash = pypto.mul(mm2_res, t4)
                            oi_tmp = pypto.add(oi_last, oi_flash)
                            if pypto.is_loop_end(s2_idx):
                                pypto.set_vec_tile_shapes(16, v2_tile[0], v2_tile[1])
                                oi_update_tmp = pypto.div(oi_tmp, sum_update,
                                                          precision_type=pypto.DivAlgorithm.INTRINSIC)
                                oi_update_3d = pypto.cast(pypto.reshape(oi_update_tmp, [1, g_tile, dn]), dtype)
                                # 11. 将结果搬运到输出tensor上
                                pypto.assemble(oi_update_3d, oi_ofs, atten_out)
                            else:
                                oi_update[:] = oi_tmp
                            pypto.set_pass_options(sg_set_scope=-1)


def ifa(atten_cfg, case_950=0, is_high_precision=True):
    device_id = os.environ.get('TILE_FWK_DEVICE_ID', 0)
    torch_dtype = torch.bfloat16
    torch.npu.set_device(int(device_id))

    b = atten_cfg.b
    s1 = atten_cfg.s1
    d = atten_cfg.q_d
    nq = atten_cfg.n1
    nkv = atten_cfg.n2

    block_size = atten_cfg.block_size
    max_num_blocks_per_query = atten_cfg.max_num_blocks_per_query

    # 获取 torch tensor 类型的 actual_seq
    kv_cache_actual_seq = atten_cfg.actual_seq

    q_shape = [b * s1, nq, d]
    kv_shape = [atten_cfg.kv_num_blocks, block_size, nkv, d]
    block_table_shape = [atten_cfg.block_table_batch, max_num_blocks_per_query]

    # 使用 torch 生成数据
    device = f'npu:{device_id}'
    q = torch.empty(q_shape, dtype=torch_dtype).uniform_(-1, 1).to(device=device)
    k = torch.empty(kv_shape, dtype=torch_dtype).uniform_(-1, 1).to(device=device)
    v = torch.empty(kv_shape, dtype=torch_dtype).uniform_(-1, 1).to(device=device)
    attention_output = torch.zeros(q_shape, dtype=torch_dtype).to(device=device)

    # 2. 生成block table - 传入 torch tensor
    block_table = gen_block_table(kv_cache_actual_seq, block_size, block_table_shape)

    # 3. 根据block table 将pa格式的数据转换成
    k_cache_bsnd, v_cache_bsnd = kv_cache_concat_bsnd(k, v, block_table, atten_cfg)

    # 4. 准备测试数据 - 直接使用 torch 张量
    block_table_torch = block_table.to(dtype=torch.int32, device=device)
    act_seq_torch = kv_cache_actual_seq.to(dtype=torch.int32, device=device)  # 直接使用已有的 tensor

    out_torch = torch.zeros(q_shape, dtype=torch_dtype).to(device=device)

    if is_high_precision:
        for i in range(b):
            for j in range(s1):
                for n2_idx in range(nkv):
                    # 从 torch tensor 获取值
                    kv_seq_len = kv_cache_actual_seq[i].item()  # 使用 .item() 获取标量值
                    seq_len = kv_seq_len - s1 + 1 + j
                    q_bs = q[i * s1 + j]
                    k_bs = k_cache_bsnd[i, :seq_len, n2_idx:n2_idx + 1].reshape(seq_len, d)
                    v_bs = v_cache_bsnd[i, :seq_len, n2_idx:n2_idx + 1].reshape(seq_len, d)
                    # MM1: 矩阵乘法
                    qk_bmm_res = torch.matmul(q_bs, k_bs.transpose(1, 0))  # 1,nq, d  -> n_q,d @ d, s2_actual_len
                    qk_ele_res = qk_bmm_res * atten_cfg.softmax_scale
                    # Softmax计算
                    softmax_res, _, _ = softmax(qk_ele_res, True)

                    # MM2: 矩阵乘法
                    bmm2_res = torch.matmul(softmax_res, v_bs)

                    # 存储结果
                    attention_output[i * s1 + j] = bmm2_res
    else:
        ifa_flash_torch(q=q, k=k, v=v, block_table=block_table_torch, kv_act_seqs=act_seq_torch, out=attention_output)

    inputs = [
        q,
        k,
        v,
        block_table_torch,
        act_seq_torch,
        out_torch
    ]
    # 5. 执行kernel并获取结果
    if case_950 == 1:
        attention_for_950(*inputs)
    else:
        attention(*inputs)

    # 6. 与PyTorch参考实现对比
    compare(np.array(attention_output.cpu().flatten().tolist()), np.array(out_torch.cpu().flatten().tolist()),
            "out_torch", rtol=0.0078125, atol=0.0001)


def matmul_proxy(left, right):
    torch_fp32 = torch.float32
    return torch.matmul(left.to(torch_fp32), right.to(torch_fp32))


def ifa_flash_torch(q, k, v, block_table, kv_act_seqs, out, is_fp32=False):
    """
    PyTorch版本的ifa_flash_torch（修正原NumPy代码的关键bug，适配张量操作）
    参数说明：
        q: torch.Tensor, shape [b*s1, n1, d]  # b: batch数, s1: query序列长度, n1: query头数, d: 头维度
        k: torch.Tensor, shape [block_num, block_size, n2, d]
        # block_num: block总数, block_size: 每个block的长度, n2: key/value头数
        v: torch.Tensor, shape [block_num, block_size, n2, d]
        block_table: torch.Tensor, shape [b, max_block]  # 每个样本的block索引表
        kv_act_seqs: torch.Tensor, shape [b]  # 每个样本的kv有效序列长度
        out: torch.Tensor, shape [b*s1, n1, d]  # 输出注意力结果（需预先分配空间）
    """
    torch_fp32 = torch.float32
    if is_fp32:
        q = q.to(torch_fp32)
        k = k.to(torch_fp32)
        v = v.to(torch_fp32)

    # ========== 1. 提取维度信息（与原代码一致） ==========
    q_shape = q.shape
    bs1, n1, d = q_shape[0], q_shape[1], q_shape[2]
    b = kv_act_seqs.shape[0]
    s1 = bs1 // b  # 每个样本的query序列长度
    k_shape = k.shape
    block_num, block_size, n2, _ = k_shape  # 补充block_num维度（原代码遗漏）
    g = n1 // n2  # 头数比例（n1必须是n2的整数倍）
    g_tile = g  # 与g一致，保留原变量名

    # ========== 2. 张量重塑（修正原代码的reshape维度错误，解决矩阵乘法维度不匹配问题） ==========
    # 原代码错误：k.reshape(-1, n2*d) 导致后续matmul维度不匹配
    # 修正：将k/v的[block_num, block_size, n2, d]重塑为[block_num*block_size*n2, d]
    # 目的：让k/v的最后一维为d，与q的d维度匹配，支持矩阵乘法
    k_2d = k.reshape(-1, d)  # shape: [block_num*block_size*n2, d]
    v_2d = v.reshape(-1, d)  # shape: [block_num*block_size*n2, d]
    # q的重塑：将[b*s1, n1, d]重塑为[b*s1*n1, d]（与原代码一致）
    q_2d = q.reshape(-1, d)  # shape: [bs1*n1, d]

    # ========== 3. 循环处理每个样本、每个位置（保留原代码的循环逻辑） ==========
    # 遍历batch
    for b_idx in range(b):
        # 遍历每个query的位置
        for s1_idx in range(s1):
            # 计算当前kv的有效序列长度（原代码逻辑）
            cur_seq = kv_act_seqs[b_idx] - (s1 - 1 - s1_idx)
            cur_seq = max(cur_seq.item(), 0)  # 防止负数（PyTorch标量需用.item()取数值）
            s2_loop = math.ceil(cur_seq / block_size)  # 需要遍历的block数

            # 遍历每个key/value头
            for n2_idx in range(n2):
                # 遍历头数比例g
                for g_idx in range(g // g_tile):
                    # ========== 4. 初始化中间变量（修正原代码的初始化错误） ==========
                    # 原代码错误：np.array([g_tile, d]) 生成的是[g_tile, d]的一维数组，形状错误
                    # 修正：初始化对应形状的零张量，与q同设备、同数据类型
                    device = q.device
                    dtype = q.dtype
                    oi_upd = torch.zeros((g_tile, d), device=device, dtype=torch_fp32)  # shape: [g_tile, d]
                    li_upd = torch.zeros(g_tile, device=device, dtype=torch_fp32)  # shape: [g_tile]
                    mi_upd = torch.zeros(g_tile, device=device, dtype=torch_fp32)  # shape: [g_tile]

                    # 遍历每个kv block
                    for s2_idx in range(s2_loop):
                        # 获取当前block的索引（需确保block_idx是有效标量）
                        block_idx = block_table[b_idx][s2_idx].item()
                        # 防止block_idx超出范围

                        # 计算偏移量（原代码逻辑）
                        bs_ofs = b_idx * s1 + s1_idx  # batch+seq的偏移
                        n2g_ofs = n2_idx * g + g_idx * g_tile  # 头数的偏移
                        # 计算当前block的有效长度（防止超出cur_seq）
                        actual_s2_tile = min(block_size, cur_seq - s2_idx * block_size)

                        # ========== 5. 提取当前的q、k、v切片（修正索引范围，防止越界） ==========
                        # 提取qi: shape [g_tile, d]
                        qi_start = bs_ofs * n1 + n2g_ofs
                        qi_end = qi_start + g_tile
                        # 防止索引越界
                        qi = q_2d[qi_start:qi_end, :]  # shape: [g_tile, d]

                        # 提取kj: shape [actual_s2_tile*n2, d]（对应block内的所有key头）
                        kj_start = block_idx * block_size
                        kj_end = kj_start + actual_s2_tile
                        # 防止索引越界
                        kj = k_2d[kj_start:kj_end, :]  # shape: [actual_s2_tile*n2, d]

                        # 提取vj: shape [actual_s2_tile*n2, d]（与kj对应）
                        vj = v_2d[kj_start:kj_end, :]  # shape: [actual_s2_tile*n2, d]

                        # ========== 6. 注意力计算（修正原代码的聚合维度，匹配PyTorch操作） ==========
                        # 第一步：q @ k.T (g_tile, d) @ (d, actual_s2_tile*n2) → (g_tile, actual_s2_tile*n2)
                        mm1 = matmul_proxy(qi, kj.t()).to(torch_fp32)
                        # 缩放因子：d^-0.5
                        muls_res = mm1 * (d ** -0.5)
                        # 第二步：计算max(muls_res) → 按最后一维取max（原代码全局max是错误的），保留维度便于广播
                        tilda_mij, _ = torch.max(muls_res, dim=-1, keepdim=True)  # shape: [g_tile, 1]
                        # 第三步：exp(muls_res - max) 防止数值溢出
                        tsub = muls_res - tilda_mij
                        tilda_pij = torch.exp(tsub)  # shape: [g_tile, actual_s2_tile*n2]
                        # 第四步：sum(tilda_pij) → 按最后一维求和
                        tilda_lij = torch.sum(tilda_pij, dim=-1, keepdim=True)  # shape: [g_tile, 1]

                        # ========== 7. 累积更新oi、li、mi（原代码逻辑，适配PyTorch） ==========
                        if s2_idx == 0:
                            # 首次迭代：初始化累积值
                            oi_tmp = matmul_proxy(tilda_pij.to(dtype), vj).to(torch_fp32)
                            if s2_idx == s2_loop - 1:
                                # 最后一个block：归一化后赋值到输出
                                oi_upd = oi_tmp / tilda_lij  # 原代码//是整数除法，PyTorch中用/（浮点数）
                                # 赋值到attn_out：shape [1, g_tile, d]
                                out[bs_ofs:bs_ofs + 1, n2g_ofs:n2g_ofs + g_tile, :] = oi_upd.unsqueeze(0).to(dtype)
                            else:
                                oi_upd = oi_tmp
                            li_upd = tilda_lij.squeeze(-1)  # 去掉最后一维，shape [g_tile]
                            mi_upd = tilda_mij.squeeze(-1)  # 去掉最后一维，shape [g_tile]
                        else:
                            # 后续迭代：累积更新
                            oi = oi_upd
                            li = li_upd.unsqueeze(-1)  # 恢复维度便于广播
                            mi = mi_upd.unsqueeze(-1)  # 恢复维度便于广播

                            # 计算新的max
                            mi_new, _ = torch.max(torch.cat([mi, tilda_mij], dim=-1), dim=-1,
                                                  keepdim=True)  # shape: [g_tile, 1]
                            # 计算指数项
                            t1 = mi - mi_new
                            t2 = torch.exp(t1)
                            t3 = tilda_mij - mi_new
                            t4 = torch.exp(t3)
                            # 累积li
                            t5 = t4 * tilda_lij
                            t6 = t2 * li
                            li_new = t6 + t5  # shape: [g_tile, 1]
                            # 累积oi
                            q3 = oi * t2  # shape: [g_tile, d]
                            q1 = matmul_proxy(tilda_pij.to(dtype), vj).to(torch_fp32)
                            q2 = q1 * t4  # shape: [g_tile, d]
                            oi_tmp = q3 + q2  # shape: [g_tile, d]

                            if s2_idx == s2_loop - 1:
                                # 最后一个block：归一化后赋值到输出
                                oi_upd = oi_tmp / li_new  # 归一化
                                oi_upd_3d = oi_upd.unsqueeze(0)  # shape: [1, g_tile, d]
                                # 赋值到attn_out（防止索引越界）
                                attn_out_start_col = n2g_ofs
                                attn_out_end_col = n2g_ofs + g_tile
                                if attn_out_end_col > out.shape[1]:
                                    attn_out_end_col = out.shape[1]
                                    attn_out_start_col = attn_out_end_col - g_tile
                                out[bs_ofs:bs_ofs + 1, attn_out_start_col:attn_out_end_col, :] = oi_upd_3d.to(dtype)
                            else:
                                oi_upd = oi_tmp
                            li_upd = li_new.squeeze(-1)  # 更新li
                            mi_upd = mi_new.squeeze(-1)  # 更新mi
    return out  # 返回输出张量（可选，因为attn_out是原地修改）


@pytest.mark.soc("950")
def test_ifa_for_950():
    # 1. 设置参数
    for case_i in range(4):
        set_qwen_common_config(case_950=1, b=16, s1=1, s2=8192)
        if case_i == 1:
            set_qwen_common_config(case_950=1, b=64, s1=1, s2=8192)
        if case_i == 2:
            set_qwen_common_config(case_950=1, b=64, s1=2, s2=8192)
        if case_i == 3:
            # 测试同一个进程中连跑两种case(静态轴发生变化)
            set_qwen_common_config(case_950=1, b=16, s1=1, s2=16384)
        atten_cfg, _ = get_common_config()

        # 检查 B 的大小和 actual_seq 长度是否相等
        assert atten_cfg.b == len(
            atten_cfg.actual_seq), f'{atten_cfg.b} {atten_cfg.actual_seq} B的大小必须和actual_seq长度相等'

        # 检查所有值是否都小于 s2
        if atten_cfg.actual_seq.device.type != 'cpu':
            actual_seq_cpu = atten_cfg.actual_seq.cpu()
        else:
            actual_seq_cpu = atten_cfg.actual_seq

        assert all(x <= atten_cfg.s2 for x in actual_seq_cpu), "所有值都必须小于s2"
        ifa(atten_cfg, case_950=1, is_high_precision=False)


@pytest.mark.soc("950", "910")
def test_ifa():
    for case_i in range(2):
        # 1. 设置参数
        set_qwen_common_config(case_950=0, b=8, s1=1, s2=16384)
        if case_i == 1:
            set_qwen_common_config(case_950=0, b=16, s1=1, s2=16384)
        atten_cfg, _ = get_common_config()
        # 检查 B 的大小和 actual_seq 长度是否相等
        assert atten_cfg.b == len(
            atten_cfg.actual_seq), f'{atten_cfg.b} {atten_cfg.actual_seq} B的大小必须和actual_seq长度相等'

        # 检查所有值是否都小于 s2
        if atten_cfg.actual_seq.device.type != 'cpu':
            actual_seq_cpu = atten_cfg.actual_seq.cpu()
        else:
            actual_seq_cpu = atten_cfg.actual_seq

        assert all(x <= atten_cfg.s2 for x in actual_seq_cpu), "所有值都必须小于s2"
        ifa(atten_cfg, case_950=0, is_high_precision=False)


@allow_in_graph
def attention(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_tables: torch.Tensor,
    actual_seqs: torch.Tensor,
    attn_res: torch.Tensor
) -> None:
    """
    Main attention function with Attention support.

    This function implements scaled dot-product attention using Attention
    mechanism, which efficiently handles variable-length sequences and dynamic
    batch sizes by managing KV cache in non-contiguous blocks.

    Args:
        query: Query tensor with shape [num_tokens, num_head, head_size]
        key_cache: Key cache tensor with shape [num_blocks, block_size, kv_head_num, head_size]
        value_cache: Value cache tensor with shape [num_blocks, block_size, kv_head_num, head_size]
        block_tables: Block mapping table with shape [batch_size, max_num_blocks_per_query]
        actual_seqs: Actual sequence lengths with shape [batch_size]
        attn_res: Output attention tensor with shape [num_tokens, num_head, head_size]

    Note:
        This function is decorated with @allow_in_graph to enable integration
        with PyTorch's compilation graph.
    """
    if isinstance(query, FakeTensor):
        return
    check_args(
        query,
        key_cache,
        value_cache,
        block_tables,
        actual_seqs,
        attn_res
    )

    inputs = [query, key_cache, value_cache, block_tables, actual_seqs, attn_res]
    for _ in range(1):
        ifa_func_kernel(*inputs)


@allow_in_graph
def attention_for_950(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_tables: torch.Tensor,
    actual_seqs: torch.Tensor,
    attn_res: torch.Tensor
) -> None:
    """
    Main attention function with Attention support.

    This function implements scaled dot-product attention using Attention
    mechanism, which efficiently handles variable-length sequences and dynamic
    batch sizes by managing KV cache in non-contiguous blocks.

    Args:
        query: Query tensor with shape [num_tokens, num_head, head_size]
        key_cache: Key cache tensor with shape [num_blocks, block_size, kv_head_num, head_size]
        value_cache: Value cache tensor with shape [num_blocks, block_size, kv_head_num, head_size]
        block_tables: Block mapping table with shape [batch_size, max_num_blocks_per_query]
        actual_seqs: Actual sequence lengths with shape [batch_size]
        attn_res: Output attention tensor with shape [num_tokens, num_head, head_size]

    Note:
        This function is decorated with @allow_in_graph to enable integration
        with PyTorch's compilation graph.
    """
    if isinstance(query, FakeTensor):
        return
    check_args(
        query,
        key_cache,
        value_cache,
        block_tables,
        actual_seqs,
        attn_res
    )

    inputs = [query, key_cache, value_cache, block_tables, actual_seqs, attn_res]
    for _ in range(1):
        ifa_func_kernel_for_950(*inputs)

if __name__ == "__main__":
    test_ifa()
    if pypto.platform.npuarch == 'DAV_3510':
        # 950上板
        test_ifa_for_950()
