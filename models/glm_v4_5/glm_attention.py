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
from numpy.testing import assert_allclose
from torch._subclasses.fake_tensor import FakeTensor
from torch._dynamo import allow_in_graph
import pypto
from utils.get_format import get_format

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
    g_tile: int
    s2_tile: int
    c1_tile_shape: list
    v1_tile_shape: list
    c2_tile_shape: list
    v2_tile_shape: list


@dataclass
class AttentionConfig:
    b: int
    s1: int
    s2: int
    n1: int
    n2: int
    q_d: int
    kv_d: int
    block_size: int = 128
    max_num_blocks_per_query: int = 0
    softmax_scale: float = 1.0
    kv_layout: str = "PA_BSND"
    actual_seq: torch.Tensor = None  # 改为 torch.Tensor 类型
    block_table_batch: int = 0
    kv_num_blocks: int = 0


def get_qwen_common_config(device="cpu"):
    b = 8
    s1 = 1
    s2 = 16384
    q_d = 128
    nq = 12
    nkv = 1
    kv_layout = "PA_BSND"
    softmax_scale = q_d ** -0.5
    block_table_batch = b
    block_size = 128
    kv_num_blocks = b * ((s2 + block_size - 1) // block_size)

    # 创建 torch tensor 类型的 actual_seq
    actual_seq_values = [s2] * b
    actual_seq_tensor = torch.tensor(actual_seq_values, dtype=torch.int32, device=device)

    atten_cfg = AttentionConfig(b=b, s1=s1, s2=s2, n1=nq, n2=nkv, softmax_scale=softmax_scale, kv_layout=kv_layout,
                                q_d=q_d, kv_d=q_d, block_size=block_size, block_table_batch=block_table_batch,
                                kv_num_blocks=kv_num_blocks, actual_seq=actual_seq_tensor)  # 传入 tensor
    atten_cfg.max_num_blocks_per_query = (s2 + block_size - 1) // block_size
    cube_tile = 128
    m_tile = 128
    s2_tile = 512
    tile_cfg = AttentionTileConfig(
        nq,
        s2_tile,
        [[m_tile, m_tile], [cube_tile, cube_tile], [cube_tile, cube_tile]],
        [m_tile, s2_tile],
        [[m_tile, m_tile], [cube_tile, cube_tile], [cube_tile, cube_tile]],
        [m_tile, cube_tile])
    return atten_cfg, tile_cfg


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


def ifa_func(q_shape, kv_shape, block_table_shape):
    """
    JIT compiled kernel implementing Incremental Flash Attention (IFA) with Attention.

    This function implements the Flash Attention algorithm optimized for Attention,
    which processes attention computation in tiles to reduce memory usage. It supports
    dynamic batch sizes and variable sequence lengths through block-based KV cache management.

    The algorithm:
    1. Reshapes Q, K, V tensors to 2D for efficient computation
    2. Iterates over batch, sequence, and head dimensions
    3. Assembles KV cache blocks according to block_table
    4. Computes attention scores using Flash Attention algorithm with online softmax
    5. Accumulates attention output incrementally

    Args:
        q: Query tensor [num_tokens, num_head, head_size]
        k: Key cache tensor [num_blocks, block_size, kv_head_num, head_size]
        v: Value cache tensor [num_blocks, block_size, kv_head_num, head_size]
        block_table: Block mapping table [batch_size, max_num_blocks_per_query]
        kv_act_seqs: Actual sequence lengths [batch_size]
        atten_out: Output attention tensor [num_tokens, num_head, head_size]

    Note:
        This function uses Flash Attention's online softmax algorithm to avoid storing
        the full attention matrix, significantly reducing memory requirements.
    """
    q_shape = (pypto.frontend.dynamic("qshape"), q_shape[1], q_shape[2])
    kv_shape = (pypto.frontend.dynamic("kvshape"), kv_shape[1], kv_shape[2], kv_shape[3])

    bs = pypto.frontend.dynamic("bs")

    @pypto.frontend.jit(
        runtime_options={"stitch_function_num_initial": 128,
        "stitch_function_outcast_memory": 1024,
        "stitch_function_inner_memory": 1024},
        # 当子图大小达到上界不允许与其他子图合并
        pass_options={"pg_upper_bound": 1536,
        # Q常驻，0代表第一组mmad，4代表4次matmul合并
        "cube_l1_reuse_setting": {0: 4}}
    )
    def ifa_func_kernel(
        q: pypto.Tensor(q_shape, pypto.DT_BF16),
        k: pypto.Tensor(kv_shape, pypto.DT_BF16),
        v: pypto.Tensor(kv_shape, pypto.DT_BF16),
        block_table: pypto.Tensor(block_table_shape, pypto.DT_INT32),
        kv_act_seqs: pypto.Tensor((bs, ), pypto.DT_INT32),
        atten_out: pypto.Tensor(q_shape, pypto.DT_BF16)
    ):

        # 1. 添加支持动态的config
        pypto.experimental.set_operation_options(combine_axis=True)

        atten_cfg, tile_cfg = get_qwen_common_config()
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

        block_num = s2_tile // block_size


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

                            idx = s2_idx * block_num
                            bs_ofs = b_idx * s1_scalar + s1_idx
                            n1g_ofs = n2_idx * group + g_idx * g_tile
                            actual_s2_tile = (cur_seq - s2_idx * s2_tile).min(s2_tile)
                            oi_ofs = [bs_ofs, n1g_ofs, 0]
                            # 5. 按照计算图实现运算逻辑，设置set_vec_tile_shapes时应尽可能用满UB，但不要超过UB的大小。
                            pypto.set_vec_tile_shapes(v1_tile[0], v1_tile[1])
                            qi = pypto.view(q_2d, [g_tile, dn], [bs_ofs * nq + n1g_ofs, 0])
                            tmp_k_2d_shape = (k.shape[0] * k.shape[1], k.shape[2] * k.shape[3])
                            tmp_k_2d = pypto.reshape(k, tmp_k_2d_shape, inplace=True)
                            kj_assemble = pypto.tensor([s2_tile, dn], tmp_k_2d.dtype, "kj_assemble")

                            for i in pypto.loop(block_num):
                                block_idx = block_table[b_idx, idx + i]
                                block_idx_valid = block_idx.max(0)
                                kj_assemble[i * k.shape[1]:(i + 1) * k.shape[1], 0:] = \
                                    pypto.view(tmp_k_2d, [k.shape[1], dn], [block_idx_valid * k.shape[1], 0])
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
                                tmp_v_2d = pypto.reshape(v, tmp_k_2d_shape, inplace=True)
                                vj_assemble = pypto.tensor([s2_tile, dn], tmp_v_2d.dtype, "vj_assemble")
                                for i in pypto.loop(block_num):
                                    block_idx = block_table[b_idx, idx + i]
                                    block_idx_valid = block_idx.max(0)
                                    vj_assemble[i * k.shape[1]:(i + 1) * k.shape[1], 0:] = \
                                        pypto.view(tmp_v_2d, [k.shape[1], dn], [block_idx_valid * k.shape[1], 0])
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
                                tmp_v_2d = pypto.reshape(v, tmp_k_2d_shape, inplace=True)
                                vj_assemble = pypto.tensor([s2_tile, dn], tmp_v_2d.dtype, "vj_assemble")
                                for i in pypto.loop(block_num):
                                    block_idx = block_table[b_idx, idx + i]
                                    block_idx_valid = block_idx.max(0)
                                    vj_assemble[i * k.shape[1]:(i + 1) * k.shape[1], 0:] = \
                                        pypto.view(tmp_v_2d, [k.shape[1], dn], [block_idx_valid * k.shape[1], 0])
                                vj_assemble = pypto.view(vj_assemble, [s2_tile, dn],
                                                        [0, 0], valid_shape=[actual_s2_tile, dn])
                                pypto.set_cube_tile_shapes(c2_tile[0], c2_tile[1], c2_tile[2])
                                oi_tmp = pypto.matmul(tilda_pij_fp16, vj_assemble, pypto.DT_FP32)

                                # v2
                                pypto.set_vec_tile_shapes(v2_tile[0], v2_tile[1])
                                oi_update[:] = oi_update * update_mul + oi_tmp
                            if pypto.is_loop_end(s2_idx):
                                oi_final = pypto.div(oi_update, sum_update)
                                pypto.set_vec_tile_shapes(16, v2_tile[0], v2_tile[1])
                                oi_final_3d = pypto.cast(
                                    pypto.reshape(oi_final, [1, g_tile, dn]),
                                    dtype)
                                # 7. 将结果搬运到输出tensor上
                                pypto.assemble(oi_final_3d, oi_ofs, atten_out)

    return ifa_func_kernel


def IFA(atten_cfg):
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

    # 4. 准备测试数据 - 直接使用 torch 张量
    block_table_torch = block_table.to(dtype=torch.int32, device=device)
    act_seq_torch = kv_cache_actual_seq.to(dtype=torch.int32, device=device)  # 直接使用已有的 tensor

    out_torch = torch.zeros(q_shape, dtype=torch_dtype).to(device=device)

    inputs = [
        q,
        k,
        v,
        block_table_torch,
        act_seq_torch,
        out_torch
    ]
    # 5. 执行kernel并获取结果
    attention(*inputs)

    # 6. 与PyTorch参考实现对比
    assert_allclose(np.array(attention_output.cpu().flatten().tolist()),
                    np.array(out_torch.cpu().flatten().tolist()),
                    rtol=0.0078125, atol=0.0001)


@pytest.mark.skip(reason="large test case")
def test_ifa():
    # 1. 设置参数
    device_id = os.environ.get('TILE_FWK_DEVICE_ID', 0)
    device = f'npu:{device_id}'
    atten_cfg, _ = get_qwen_common_config(device=device)

    # 检查 B 的大小和 actual_seq 长度是否相等
    assert atten_cfg.b == len(
        atten_cfg.actual_seq), f'{atten_cfg.b} {atten_cfg.actual_seq} B的大小必须和actual_seq长度相等'

    # 检查所有值是否都小于 s2
    if atten_cfg.actual_seq.device.type != 'cpu':
        actual_seq_cpu = atten_cfg.actual_seq.cpu()
    else:
        actual_seq_cpu = atten_cfg.actual_seq

    assert all(x <= atten_cfg.s2 for x in actual_seq_cpu), "所有值都必须小于s2"
    IFA(atten_cfg)


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

    q_shape = query.shape
    kv_shape = key_cache.shape
    block_table_shape = block_tables.shape
    shapes = [q_shape, kv_shape, block_table_shape]
    inputs = [query, key_cache, value_cache, block_tables, actual_seqs, attn_res]
    ifa_func(*shapes)(*inputs)

if __name__ == "__main__":
    test_ifa()