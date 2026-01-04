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
"""

import os
from typing import List
import torch
import pytest

import pypto


def gather_after_prolog_graph(
    topk_indices: pypto.Tensor,
    k_nope_cache: pypto.Tensor,
    k_rope_cache: pypto.Tensor,
    block_table: pypto.Tensor,
    act_seqs: pypto.Tensor,
    gather_res: pypto.Tensor,
    b: int,
    s1: int,
    block_size: int,
    topk: int,
):
    dn = k_nope_cache.shape[-1]
    dr = k_rope_cache.shape[-1]
    n2 = topk_indices.shape[2]
    unroll_list = {64, 32, 16, 8, 4, 2, 1}
    for b_idx in pypto.loop(0, b, 1, name="loop_b_gather", idx_name="bIdx", submit_before_loop=True):
        for s1_idx in pypto.loop(0, s1, 1, name="loop_s1_gather", idx_name="s1Idx"):
            for n2_idx in pypto.loop(0, n2, 1, name="loop_n2_gather", idx_name="n2Idx"):
                pypto.set_semantic_label("gather0")
                cur_kv_seq = act_seqs[b_idx]
                topk_loop = (cur_kv_seq - s1 + 1 + s1_idx).max(0).min(topk)
                for topk_idx in pypto.loop(
                    0, topk_loop, 1, name="loop_k_gather", idx_name="topKIdx", unroll_list=unroll_list
                ):

                    def inside_topk_idx_loop(b_idx, s1_idx, n2_idx, topk_idx):
                        pypto.set_vec_tile_shapes(1, 1, 1, 16)
                        topk_index = topk_indices[b_idx, s1_idx, n2_idx, topk_idx]

                        block_idx_in_batch = topk_index // block_size
                        tail = topk_index % block_size
                        slc_block_idx = block_table[b_idx, block_idx_in_batch]

                        pypto.set_vec_tile_shapes(1, dn)
                        kv_slc_block = pypto.view(k_nope_cache, [1, dn], [slc_block_idx * block_size + tail, 0])
                        kr_slc_block = pypto.view(k_rope_cache, [1, dr], [slc_block_idx * block_size + tail, 0])

                        pypto.set_semantic_label("gather1")
                        kv_slc_block_fp32 = pypto.cast(kv_slc_block, pypto.DT_FP32)
                        kr_slc_block_fp32 = pypto.cast(kr_slc_block, pypto.DT_FP32)

                        pypto.set_semantic_label("gather2")
                        kv_slc_block_fp16 = pypto.cast(kv_slc_block_fp32, gather_res.dtype)
                        kr_slc_block_fp16 = pypto.cast(kr_slc_block_fp32, gather_res.dtype)

                        ofs = b_idx * s1 * n2 * topk + s1_idx * n2 * topk + n2_idx * topk + topk_idx
                        pypto.assemble(kv_slc_block_fp16, [ofs, 0], gather_res)
                        pypto.assemble(kr_slc_block_fp16, [ofs, dn], gather_res)

                    inside_topk_idx_loop(b_idx=b_idx, s1_idx=s1_idx, n2_idx=n2_idx, topk_idx=topk_idx)


def generate_in_out(
    seq_lens: List,
    topk: int,
    block_size: int,
    b: int,
    s1: int,
    n2: int,
    dn: int,
    dr: int,
):
    block_nums = [(s + block_size - 1) // block_size for s in seq_lens]
    block_num = sum(block_nums)
    seq_max_block_num = max(block_nums)
    dt = torch.bfloat16
    i32 = torch.int32

    topk_indices = torch.zeros((b, s1, n2, topk), dtype=i32)
    for bi, s in enumerate(seq_lens):
        topk_indices[bi][0][0][:s] = torch.randperm(s)[:topk]

    block_table = -1 * torch.ones(b, seq_max_block_num, dtype=i32)
    mapping = torch.randperm(block_num)
    offset = 0
    for bi, blk_num in enumerate(block_nums):
        block_table[bi, :blk_num] = mapping[offset:offset + blk_num]
        offset += blk_num

    # go away from the batch space and have a blocks instead
    k_nope_cache = torch.randn((block_num, block_size, dn)).to(dt)
    k_rope_cache = torch.randn((block_num, block_size, dr)).to(dt)
    act_seqs = torch.tensor(seq_lens, dtype=i32)

    gather_res = generate_golden(
        topk_indices=topk_indices,
        k_nope_cache=k_nope_cache,
        k_rope_cache=k_rope_cache,
        block_table=block_table,
        act_seqs=act_seqs,
    ).to(dt)
    inps = [topk_indices, k_nope_cache.reshape(-1, dn), k_rope_cache.reshape(-1, dr), block_table, act_seqs]
    outs = [gather_res.reshape(-1, dn + dr)]
    return inps, outs


def generate_golden(
    topk_indices: torch.Tensor,
    k_nope_cache: torch.Tensor,
    k_rope_cache: torch.Tensor,
    block_table: torch.Tensor,
    act_seqs: torch.Tensor,
):
    b, s1, n2, topk = topk_indices.shape
    _, block_size, dn = k_nope_cache.shape
    _, _, dr = k_rope_cache.shape

    # block_num * blk_size x dn+dr
    kv_cache = torch.cat([k_nope_cache, k_rope_cache], dim=-1).reshape(-1, dn + dr)
    gather_res = torch.zeros((b, s1, topk, dn + dr))
    for b_idx in range(b):
        for s1_idx in range(s1):
            for n2_idx in range(n2):
                act_topk = min(max(act_seqs[b_idx] - s1 + 1 + s1_idx, 0), topk)
                for k_idx in range(act_topk):
                    topk_val = topk_indices[b_idx, s1_idx, n2_idx, k_idx]
                    blk_idx = topk_val // block_size
                    blk_off = topk_val % block_size
                    phy_blk_idx = block_table[b_idx, blk_idx]
                    gather_res[b_idx, s1_idx, k_idx] = kv_cache[phy_blk_idx * block_size + blk_off, :]
    return gather_res


def gather_after_prolog_compute(block_size, b, s1, n2, topk, dn, dr, seq_lens):
    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
    torch.npu.set_device(device_id)

    @pypto.jit(
    )
    def gather_fwd(topk_indices, k_nope_cache, k_rope_cache, block_table, act_seqs, gather_res):
        gather_after_prolog_graph(
            topk_indices,
            k_nope_cache,
            k_rope_cache,
            block_table,
            act_seqs,
            gather_res,
            b,
            s1,
            block_size=block_size,
            topk=topk,
        )

    input_data, output_golden = generate_in_out(
        seq_lens=seq_lens, block_size=block_size, b=b, s1=s1, n2=n2, topk=topk, dn=dn, dr=dr
    )
    input_data = [a.npu() for a in input_data]
    output_data = [a.npu() for a in map(torch.zeros_like, output_golden)]
    pto_inputs = [pypto.from_torch(tensor, f"IN_{idx}") for idx, tensor in enumerate(input_data)]
    pto_outputs = [pypto.from_torch(tensor, f"OUT_{idx}") for idx, tensor in enumerate(output_data)]
    gather_fwd(*pto_inputs, *pto_outputs)
    compare(output_data[0].cpu(), output_golden[0])
    pypto.runtime._device_fini()


def compare(t: torch.Tensor, t_ref: torch.Tensor):
    assert t.shape == t_ref.shape
    assert t.dtype == t_ref.dtype
    assert t.device == t_ref.device
    # Exact since kernel is purely indexing ops
    torch.testing.assert_close(t, t_ref, rtol=0.0, atol=0.0)


def test_gather():
    topk = 2048
    block_size = 128
    b = 4
    s1 = 1
    n2 = 1
    dn = 512
    dr = 64
    # 4 sequences with different context lengths
    kv_act_seq = [768, 4097, 8192, 131071]

    gather_after_prolog_compute(block_size=block_size, b=b, s1=s1, n2=n2, topk=topk, dn=dn, dr=dr, seq_lens=kv_act_seq)


if __name__ == "__main__":
    test_gather()
