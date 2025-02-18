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

本脚本有 2 种执行模式:
1. CI批跑时, 由 tests/cmake/scripts/golden_ctrl.py 调用, 为避免日志过多, 此时 logging 级别为 logging.INFO;
2. 单独调试时, 本脚本单独被调用, 此时 logging 级别为 logging.DEBUG;
"""
import math
import sys
import logging
from pathlib import Path
from typing import List

import numpy as np
import torch

if __name__ == "__main__":
    """单独调试时配置"""
    # 日志级别
    logging.basicConfig(
        format="%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s: %(message)s",
        level=logging.DEBUG,
    )
    # 系统 import 路径
    g_src_root: Path = Path(Path(__file__).parent, "../../../../").resolve()
    logging.debug("SrcRoot: %s", g_src_root)
    g_ctrl_path: Path = Path(g_src_root, "scripts")
    if str(g_ctrl_path) not in sys.path:
        sys.path.append(str(g_ctrl_path))
    from golden_register import (
        GoldenRegister,
    )  # 单独调试 import 失败, 需确认上文中 '系统 import 路径' 配置正确
else:
    from golden_register import GoldenRegister


def tensor_tofile(t: torch.Tensor, output: Path, dtype: torch.dtype):
    with open(str(output), "wb") as f:
        if dtype == torch.bfloat16:
            dtype = torch.int16
        for each in t:
            f.write(each.view(dtype).numpy().tobytes())


def gen_block_table(act_seq, block_size, s1, need_indices=False):
    b = act_seq.shape[0]
    block_num = 0
    block_num_each = []
    max_kv = max(act_seq)
    for cur_s in act_seq:
        cur_block_num = math.ceil(cur_s / block_size)
        block_num_each.append(cur_block_num)
        block_num += cur_block_num
    block_table_shape = [b, math.ceil(max_kv / block_size)]
    block_idx_list = torch.arange(0, block_num, 1)
    block_idx_list = block_idx_list[torch.randperm(block_idx_list.size(0))].to(
        torch.int32
    )

    block_idx = 0
    block_table = -torch.ones(block_table_shape, dtype=torch.int32)

    block_table_bidx = 0
    for cur_block in block_num_each:
        for j in range(cur_block):
            block_table[block_table_bidx, j] = block_idx_list[block_idx]
            block_idx += 1
        block_table_bidx += 1

    if need_indices:
        cache_index = -torch.ones((b, s1), dtype=torch.int32)
        for i in range(b):
            cur_act = act_seq[i]
            for j in range(s1):
                pos = cur_act - s1 + j
                block_idx_in_seq = pos // block_size
                global_block_id = block_table[i, block_idx_in_seq]

                offset_in_block = pos % block_size
                global_index = global_block_id * block_size + offset_in_block
                cache_index[i, j] = global_index
    else:
        cache_index = None

    if need_indices:
        return block_num, block_table, cache_index
    else:
        return block_num, block_table


def gen_cache_tensor(
    k_bsnd, block_table, block_num, block_size, dtype, k_dim, kv_lora_rank
):
    b = block_table.shape[0]
    n2 = 1
    k_cache = torch.zeros([block_num, block_size, n2, k_dim], dtype=dtype)

    k_tensor_bsnd = torch.zeros(
        (b, block_table.shape[1] * block_size, n2, k_dim), dtype=dtype
    )
    k_tensor_bsnd[:, : k_bsnd.shape[1], :, :] = k_bsnd[:, :, :, :]

    for b_idx in range(b):
        for block_idx, cache_block_idx in enumerate(block_table[b_idx]):
            block_offset = block_idx * block_size
            if cache_block_idx == -1:
                continue
            else:
                k_cache[cache_block_idx, :, :, :] = k_tensor_bsnd[
                    b_idx : b_idx + 1, block_offset : (block_offset + block_size), :, :
                ]

    cmp_kv_cache = k_cache[:, :, :, :kv_lora_rank]
    cmp_kr_cache = k_cache[:, :, :, kv_lora_rank:]
    return cmp_kv_cache, cmp_kr_cache


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(x, cos, sin):
    cmp_block_size, rope_dim = x.shape
    x_rope = (
        x.reshape(cmp_block_size, rope_dim // 2, 2)
        .permute(0, 2, 1)
        .reshape(cmp_block_size, rope_dim)
    )
    x_embed = (x_rope * cos) + (rotate_half(x_rope) * sin)
    return x_embed


def safe_sigmoid(x):
    return torch.where(
        x >= 0, 1.0 / (1.0 + torch.exp(-x)), torch.exp(x) / (1.0 + torch.exp(x))
    )


def mlp_compression(kv_local, w1, w2):
    dtype = w1.dtype
    kv_local_matmul_w1 = torch.matmul(kv_local.to(torch.float32), w1.to(torch.float32))
    sigmoid_res = safe_sigmoid(kv_local_matmul_w1)
    sigmoid_res_matmul_w2 = torch.matmul(sigmoid_res, w2.to(torch.float32)).to(dtype)
    return sigmoid_res_matmul_w2


def scatter_update_pa_bsnd(cache, key_states, index_values, invalid_b):
    block_number, block_size, n2, d = cache.shape
    b = index_values.shape[0]
    res = cache.reshape(block_number * block_size * n2, d)
    for bi in range(b):
        if bi in invalid_b:
            continue
        index_value = index_values[bi]
        res[index_value: index_value + 1, :] = key_states[bi]
    return res


def kv_compression_compute(
    input_params,
    kv_cache,
    kr_cache,
    cmp_kv_cache,
    cmp_kr_cache,
    act_seq,
    block_table,
    cmp_cache_index,
    mlp_cos,
    mlp_sin,
    mlp_wk1,
    mlp_wk2,
):
    block_size = input_params[0]
    b = input_params[1]
    s1 = input_params[2]
    kv_lora_rank = input_params[3]
    rope_dim = input_params[4]
    cmp_block_size = input_params[5]
    stride = input_params[6]
    n2 = input_params[7]
    datatype = input_params[8]
    slc_block_size = input_params[11]
    dtype = torch.bfloat16 if datatype == 0 else torch.float16

    cmp_block_num = cmp_kv_cache.shape[0]
    cmp_kv_cache_out = cmp_kv_cache.reshape(
        cmp_block_num * block_size, n2 * kv_lora_rank
    )
    cmp_kr_cache_out = cmp_kr_cache.reshape(cmp_block_num * block_size, n2 * rope_dim)
    k_to_cmp = torch.zeros(b, cmp_block_size * (kv_lora_rank + rope_dim), dtype=dtype)
    invalid_b = []
    for bi in range(b):
        block_start_idx = (act_seq[bi] - cmp_block_size) // block_size
        block_end_idx = (act_seq[bi] - 1) // block_size
        block_start_offset = (act_seq[bi] - cmp_block_size) % block_size
        table_loop = block_end_idx - block_start_idx + 1
        if (act_seq[bi] % stride != 0) or (act_seq[bi] < cmp_block_size):
            block_start_idx = 0
            block_end_idx = 0
            block_start_offset = 0
            table_loop = 1
            invalid_b.append(bi)
        k_nope_local = torch.zeros(cmp_block_size, kv_lora_rank, dtype=dtype)
        k_rope_local = torch.zeros(cmp_block_size, rope_dim, dtype=dtype)
        k_nope_block = torch.zeros(table_loop * block_size, kv_lora_rank, dtype=dtype)
        k_rope_block = torch.zeros(table_loop * block_size, rope_dim, dtype=dtype)
        for tidx in range(table_loop):
            block_idx = block_table[bi, block_start_idx + tidx]
            k_nope_block[tidx * block_size: (tidx + 1) * block_size, :] = kv_cache[
                (block_idx) * block_size: (block_idx + 1) * block_size, :
            ]
            k_rope_block[tidx * block_size: (tidx + 1) * block_size, :] = kr_cache[
                (block_idx) * block_size: (block_idx + 1) * block_size, :
            ]
        k_nope_local = k_nope_block[
            block_start_offset: block_start_offset + cmp_block_size, :
        ]
        k_rope_local = k_rope_block[
            block_start_offset: block_start_offset + cmp_block_size, :
        ]
        mlp_cos_local = mlp_cos[bi, :, :]
        mlp_sin_local = mlp_sin[bi, :, :]
        k_rope_embed = apply_rope(
            k_rope_local.to(torch.float32),
            mlp_cos_local.to(torch.float32),
            mlp_sin_local.to(torch.float32),
        )
        k_local = torch.cat((k_nope_local, k_rope_embed), dim=1)  # [32, 576]
        k_to_cmp[bi, :] = k_local.reshape(1, cmp_block_size * (kv_lora_rank + rope_dim))

    k_cmp = mlp_compression(k_to_cmp, mlp_wk1, mlp_wk2)  # [b, 576]
    k_cmp_nope = k_cmp[:, :kv_lora_rank]
    k_cmp_rope = k_cmp[:, kv_lora_rank:]
    cmp_kv_cache_out = scatter_update_pa_bsnd(
        cmp_kv_cache, k_cmp_nope, cmp_cache_index[:, (s1 - 1):], invalid_b
    )
    cmp_kr_cache_out = scatter_update_pa_bsnd(
        cmp_kr_cache, k_cmp_rope, cmp_cache_index[:, (s1 - 1):], invalid_b
    )

    rs = slc_block_size // stride
    rc = cmp_block_size // stride

    aux_tensor = torch.zeros(rs + rc - 1, block_size)
    aux_tensor_local = torch.ones(rs, block_size)

    for j in range(rc):
        aux_tensor[j: j + rs, :] += aux_tensor_local

    return cmp_kv_cache_out, cmp_kr_cache_out, aux_tensor


def gen_kv_compress_data(params, output):
    in_params_path = Path(output, "input_param.bin")
    kv_cache_path = Path(output, "kv_cache.bin")
    kr_cache_path = Path(output, "kr_cache.bin")
    cmp_kv_cache_path = Path(output, "kv_cache_compress.bin")
    cmp_kr_cache_path = Path(output, "kr_cache_compress.bin")
    block_table_path = Path(output, "block_table.bin")
    cmp_cache_index_path = Path(output, "cache_index_compress.bin")
    act_seq_path = Path(output, "act_seq_compress.bin")
    act_cmp_seq_path = Path(output, "act_cmp_seq_compress.bin")
    mlp_wk1_path = Path(output, "mlp_wk1.bin")
    mlp_wk2_path = Path(output, "mlp_wk2.bin")
    mlp_wk1_nz_path = Path(output, "mlp_wk1_nz.bin")
    mlp_wk2_nz_path = Path(output, "mlp_wk2_nz.bin")
    mlp_cos_path = Path(output, "mlp_cos.bin")
    mlp_sin_path = Path(output, "mlp_sin.bin")
    cmp_kv_cache_output_path = Path(output, "kv_cache_out_compress.bin")
    cmp_kr_cache_output_path = Path(output, "kr_cache_out_compress.bin")
    aux_tensor_path = Path(output, "aux_tensor.bin")

    # construct input tensor
    block_size = params[0]
    b = params[1]
    s1 = params[2]
    kv_lora_rank = params[3]
    rope_dim = params[4]
    cmp_block_size = params[5]
    stride = params[6]
    k_dim = kv_lora_rank + rope_dim
    n2 = params[7]
    datatype = params[8]
    slc_block_size = params[9]
    dtype = torch.bfloat16 if datatype == 0 else torch.float16

    need_update = False
    while True:
        act_seq = torch.randint(low=s1, high=131088, size=(b,), dtype=torch.int32)
        for bi in range(b):
            if act_seq[bi] % stride == 0:
                need_update = True
        if need_update:
            break

    act_cmp_seq = (
        torch.maximum(
            (act_seq - cmp_block_size + stride - 1) // stride,
            torch.tensor(0, dtype=torch.int32),
        )
        + 1
    )
    block_num, block_table = gen_block_table(act_seq, block_size, s1)
    max_block_num = block_table.shape[1]
    cmp_block_num, cmp_block_table, cmp_cache_index = gen_block_table(
        act_cmp_seq, block_size, s1, True
    )
    max_act_seq = max(act_seq)
    max_act_cmp_seq = max(act_cmp_seq)
    k_bsnd_shape = (b, max_act_seq, n2, k_dim)
    k_cmp_bsnd_shape = (b, max_act_cmp_seq, n2, k_dim)
    mlp_wk1_shape = (cmp_block_size * k_dim, 2 * cmp_block_size * k_dim)
    mlp_wk2_shape = (2 * cmp_block_size * k_dim, k_dim)
    mlp_cos_shape = (b, cmp_block_size, rope_dim)
    mlp_sin_shape = (b, cmp_block_size, rope_dim)
    k_bsnd = 2 * torch.rand(k_bsnd_shape, dtype=dtype) - 1
    k_cmp_bsnd = 2 * torch.rand(k_cmp_bsnd_shape, dtype=dtype) - 1
    mlp_wk1 = 2 * torch.rand(mlp_wk1_shape, dtype=dtype) - 1
    mlp_wk2 = 2 * torch.rand(mlp_wk2_shape, dtype=dtype) - 1
    mlp_cos = 2 * torch.rand(mlp_cos_shape, dtype=dtype) - 1
    mlp_sin = 2 * torch.rand(mlp_sin_shape, dtype=dtype) - 1

    kv_cache, kr_cache = gen_cache_tensor(
        k_bsnd, block_table, block_num, block_size, dtype, k_dim, kv_lora_rank
    )
    kv_cache = kv_cache.reshape(block_num * block_size, n2 * kv_lora_rank)
    kr_cache = kr_cache.reshape(block_num * block_size, n2 * rope_dim)

    cmp_kv_cache, cmp_kr_cache = gen_cache_tensor(
        k_cmp_bsnd,
        cmp_block_table,
        cmp_block_num,
        block_size,
        dtype,
        k_dim,
        kv_lora_rank,
    )

    input_params = [
        block_size,
        b,
        s1,
        kv_lora_rank,
        rope_dim,
        cmp_block_size,
        stride,
        n2,
        datatype,  # datatype = 0 implies torch.bfloat16, datatype = 1 implies torch.float16
        block_num,
        cmp_block_num,
        max_block_num,
        slc_block_size,
    ]

    tensor_tofile(
        torch.tensor([input_params], dtype=torch.int32), in_params_path, torch.int32
    )
    tensor_tofile(kv_cache, kv_cache_path, dtype)
    tensor_tofile(kr_cache, kr_cache_path, dtype)
    tensor_tofile(cmp_kv_cache, cmp_kv_cache_path, dtype)
    tensor_tofile(cmp_kr_cache, cmp_kr_cache_path, dtype)
    tensor_tofile(block_table, block_table_path, torch.int32)
    tensor_tofile(mlp_wk1, mlp_wk1_path, dtype)
    tensor_tofile(mlp_wk2, mlp_wk2_path, dtype)
    mlp_wk1_nz = mlp_wk1.reshape(
        cmp_block_size * k_dim, 2 * cmp_block_size * k_dim // 16, 16
    ).permute(1, 0, 2)
    mlp_wk2_nz = mlp_wk2.reshape(2 * cmp_block_size * k_dim, k_dim // 16, 16).permute(
        1, 0, 2
    )
    tensor_tofile(mlp_wk1_nz, mlp_wk1_nz_path, dtype)
    tensor_tofile(mlp_wk2_nz, mlp_wk2_nz_path, dtype)
    tensor_tofile(mlp_cos, mlp_cos_path, dtype)
    tensor_tofile(mlp_sin, mlp_sin_path, dtype)
    tensor_tofile(act_seq, act_seq_path, torch.int32)
    tensor_tofile(act_cmp_seq, act_cmp_seq_path, torch.int32)
    tensor_tofile(cmp_cache_index, cmp_cache_index_path, torch.int32)

    cmp_kv_cache_out, cmp_kr_cache_out, aux_tensor = kv_compression_compute(
        input_params,
        kv_cache,
        kr_cache,
        cmp_kv_cache,
        cmp_kr_cache,
        act_seq,
        block_table,
        cmp_cache_index,
        mlp_cos,
        mlp_sin,
        mlp_wk1,
        mlp_wk2,
    )

    dtype = torch.bfloat16 if datatype == 0 else torch.float16
    tensor_tofile(cmp_kv_cache_out, cmp_kv_cache_output_path, dtype)
    tensor_tofile(cmp_kr_cache_out, cmp_kr_cache_output_path, dtype)
    tensor_tofile(aux_tensor, aux_tensor_path, torch.float32)

    return 0


@GoldenRegister.reg_golden_func(
    case_names=[
        "DynKVCmp.KVCmpBatch32bf16",
        "DynKVCmp.KVCmpBatch48float16",
        "DynKVCmp.AuxVectorBuildFloat32",
    ]
)
def kv_compress(case_name: str, output: Path) -> bool:
    in_params_path = Path(output, "input_param.bin")
    kv_cache_path = Path(output, "kv_cache.bin")
    kr_cache_path = Path(output, "kr_cache.bin")
    cmp_kv_cache_path = Path(output, "kv_cache_compress.bin")
    cmp_kr_cache_path = Path(output, "kr_cache_compress.bin")
    block_table_path = Path(output, "block_table.bin")
    cmp_cache_index_path = Path(output, "cache_index_compress.bin")
    act_seq_path = Path(output, "act_seq_compress.bin")
    act_cmp_seq_path = Path(output, "act_cmp_seq_compress.bin")
    mlp_wk1_path = Path(output, "mlp_wk1.bin")
    mlp_wk2_path = Path(output, "mlp_wk2.bin")
    mlp_wk1_nz_path = Path(output, "mlp_wk1_nz.bin")
    mlp_wk2_nz_path = Path(output, "mlp_wk2_nz.bin")
    mlp_cos_path = Path(output, "mlp_cos.bin")
    mlp_sin_path = Path(output, "mlp_sin.bin")
    cmp_kv_cache_output_path = Path(output, "kv_cache_out_compress.bin")
    cmp_kr_cache_output_path = Path(output, "kr_cache_out_compress.bin")
    aux_tensor_path = Path(output, "aux_tensor.bin")

    complete = (in_params_path.exists() and kv_cache_path.exists() and kr_cache_path.exists() and
        cmp_kv_cache_path.exists() and cmp_kr_cache_path.exists() and block_table_path.exists() and
        cmp_cache_index_path.exists() and act_seq_path.exists() and act_cmp_seq_path.exists() and
        mlp_wk1_path.exists() and mlp_wk2_path.exists() and mlp_wk1_nz_path.exists() and
        mlp_wk2_nz_path.exists() and mlp_cos_path.exists() and mlp_sin_path.exists() and
        cmp_kv_cache_output_path.exists() and cmp_kr_cache_output_path.exists() and
        aux_tensor_path.exists())
    # complete = False

    if complete:
        logging.debug("Case(%s), Golden complete.", case_name)
        return True

    block_size = 128
    kv_lora_rank, rope_dim = 512, 64
    cmp_block_size, stride, slc_block_size = 32, 16, 64
    n2 = 1
    datatype = 0
    if case_name.startswith("DynKVCmp.KVCmpBatch32bf16"):
        b, s1 = 32, 2
        datatype = 0
        params = [
            block_size,
            b,
            s1,
            kv_lora_rank,
            rope_dim,
            cmp_block_size,
            stride,
            n2,
            datatype,
            slc_block_size,
        ]
        gen_kv_compress_data(params, output)
    elif case_name.startswith("DynKVCmp.KVCmpBatch48float16"):
        b, s1 = 48, 1
        datatype = 1
        params = [
            block_size,
            b,
            s1,
            kv_lora_rank,
            rope_dim,
            cmp_block_size,
            stride,
            n2,
            datatype,
            slc_block_size,
        ]
        gen_kv_compress_data(params, output)
    elif case_name.startswith("DynKVCmp.AuxVectorBuildFloat32"):
        b, s1 = 32, 2
        datatype = 1
        params = [
            block_size,
            b,
            s1,
            kv_lora_rank,
            rope_dim,
            cmp_block_size,
            stride,
            n2,
            datatype,
            slc_block_size,
        ]
        gen_kv_compress_data(params, output)
    else:
        logging.error("Can't get func to gen golde, Case(%s)", case_name)
        return False
    return True


def main() -> bool:
    """
    单独调试 入口函数
    """
    # 用例名称
    case_name_list: List[str] = [
        "DynKVCmp.KVCmpBatch16float16",
        "DynKVCmp.AuxVectorBuildFloat32",
    ]
    # 函数调用
    ret: bool = True
    for cs in case_name_list:
        output: Path = Path(g_src_root, "build/output/bin/golden", cs).resolve()
        output.mkdir(parents=True, exist_ok=True)
        ret = kv_compress(case_name=cs, output=output)
    return ret


if __name__ == "__main__":
    exit(0 if main() else 1)
