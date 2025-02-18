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

import math
import sys
import logging
from pathlib import Path
from typing import List
import torch
import numpy as np
from ml_dtypes import bfloat16

if __name__ == "__main__":

    # 日志级别
    logging.basicConfig(format='%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s: %(message)s',
                        level=logging.DEBUG)
    # 系统 import 路径
    g_src_root: Path = Path(Path(__file__).parent, "../../../../../").resolve()
    logging.debug("SrcRoot: %s", g_src_root)
    g_ctrl_path: Path = Path(g_src_root, "tests/cmake/scripts")
    if str(g_ctrl_path) not in sys.path:
        sys.path.append(str(g_ctrl_path))
    from golden_register import GoldenRegister
else:
    from golden_register import GoldenRegister


def dump_file(data_pool, data_path, type_str):
    np.array(data_pool).astype(type_str).tofile(data_path)


def gen_uniform_data(data_shape, min_value, max_value, dtypes):
    if isinstance(data_shape, list):
        data_shape = tuple(data_shape)
    if min_value == 0 and max_value == 0:
        return torch.zeros(data_shape, dtype=dtypes)
    if dtypes == torch.bool:
        return torch.randint(0, 2, size=data_shape, dtype=torch.bool)
    return torch.rand(data_shape, dtype=dtypes).uniform_(min_value, max_value)


def softmax(x):
    # this func is only used by quant_dequant
    if (isinstance(x, np.ndarray)):
        x = x.astype(np.float32)
        x_max = x.max(axis=-1, keepdims=True)
        x_sub = x - x_max
        y = np.exp(x_sub)
        x_sum = y.sum(axis=-1, keepdims=True)
        ans = y
    else:
        x = x.to(torch.float32)
        x_max = torch.max(x, dim=-1, keepdims=True)[0]
        x_sub = x - x_max
        y = torch.exp(x_sub)
        x_sum = torch.sum(y, dim=-1, keepdims=True)
        ans = y
    return ans, x_sum


def win_attn_calc(input_params_win_attn, actual_seq_list, q_bsnd, k_bsnd, v_bsnd, dtypes, atten_out):
    flag = False
    if 'numpy' in str(dtypes):
        flag = True

    b = input_params_win_attn[0]
    s_q = input_params_win_attn[1]
    n_kv = input_params_win_attn[2]
    n_q = input_params_win_attn[3]
    d_q = input_params_win_attn[4]
    win = input_params_win_attn[5]
    d_k = input_params_win_attn[6]
    d_v = input_params_win_attn[7]
    scalar = input_params_win_attn[8]

    for b_index in range(b):
        for s1_index in range(s_q):
            for n_kv_index in range(n_kv):

                act_seq = actual_seq_list[b_index]

                q_tensor_cur = q_bsnd[b_index:(b_index + 1), s1_index:(s1_index + 1), :, :].reshape(n_q, d_q)

                cur_loc = act_seq - s_q + s1_index + 1
                valid_len = min(cur_loc, win)

                k_cur = k_bsnd[b_index:(b_index + 1), cur_loc - valid_len: cur_loc, n_kv_index:(n_kv_index + 1), :].reshape(valid_len, d_k)
                v_cur = v_bsnd[b_index:(b_index + 1), cur_loc - valid_len: cur_loc, n_kv_index:(n_kv_index + 1), :].reshape(valid_len, d_v)

                if flag:
                    qk_mm_res = np.matmul(q_tensor_cur.astype(dtypes), k_cur.astype(dtypes).transpose(1, 0))
                    qk_mm_fp32 = qk_mm_res.astype(np.float32)
                else:
                    qk_mm_res = torch.matmul(q_tensor_cur.to(dtypes), k_cur.T.to(dtypes))
                    qk_mm_fp32 = qk_mm_res.to(torch.float32)
                qk_ele_res = qk_mm_fp32 * scalar
                softmax_res, softmax_sum = softmax(qk_ele_res)
                softmax_out = softmax_res / softmax_sum
                if flag:
                    mm2_res = np.matmul(softmax_out.astype(dtypes), v_cur.astype(dtypes))
                else:
                    mm2_res = torch.matmul(softmax_out.to(dtypes), v_cur.to(dtypes))
                atten_out[b_index:(b_index + 1), s1_index:(s1_index + 1), :, :] = mm2_res
    if flag:
        atten_out = atten_out.astype(np.float32)
    else:
        atten_out = atten_out.to(torch.float32)
    return atten_out


def gen_win_attn_data(win, b, s_q, n_q, skv, block_size, n_kv, dtypes, output):
    torch.seed()

    # output path
    q_nope_path = Path(output, 'q_nope.bin')
    q_rope_path = Path(output, 'q_rope.bin')

    k_cache_nope_path = Path(output, 'k_cache_nope.bin')
    k_cache_rope_path = Path(output, 'k_cache_rope.bin')

    block_table_path = Path(output, 'block_table.bin')
    actual_seq_len_path = Path(output, 'actual_seq_list.bin')
    attent_out_path = Path(output, 'atten_out.bin')
    input_param_path = Path(output, 'input_param.bin')

    if 'bfloat' in str(dtypes):
        new_dtype = torch.bfloat16
    elif 'float16' in str(dtypes):
        new_dtype = torch.float16
    elif 'float32' in str(dtypes):
        new_dtype = torch.float32
    else:
        raise ValueError(f"Unsupposed dtype: {dtypes}. Supported dtypes are bloat16, float16 and float32")

    kv_lora_rank = 512
    qk_rope_dim = 64
    # q head dim
    d_q = kv_lora_rank + qk_rope_dim
    # k head dim
    d_k = kv_lora_rank + qk_rope_dim
    # v head dim
    d_v = kv_lora_rank
    scalar = d_q ** -0.5

    if isinstance(skv, int):
        actual_seq_list = [skv] * b
    elif isinstance(skv, list):
        if len(skv) == b:
            actual_seq_list = skv
        else:
            raise RuntimeError("unsupported skv list length")
    else:
        raise RuntimeError("unsupported skv data type")

    skv_max = max(actual_seq_list)

    shape_q = [b * s_q * n_q, d_q]
    shape_k = [b, skv_max, n_kv, d_k]

    atten_out_shape = [b, s_q, n_q, d_v]

    block_num_per_batch = []
    block_num_min = 0
    block_num = 0

    # gen q k v data
    q = gen_uniform_data(shape_q, -1, 1, new_dtype)
    q_bsnd = q.reshape(b, s_q, n_q, d_q)
    k_bsnd = gen_uniform_data(shape_k, -1, 1, new_dtype)
    v_bsnd = k_bsnd[:, :, :, : kv_lora_rank]

    for actual_seq in actual_seq_list:
        block_num_per_batch.append(math.ceil(actual_seq / block_size))
        block_num_min += math.ceil(actual_seq / block_size)

    # 处理pageatten场景（block table, kv cache处理不涉及cpu、真值计算，仅为npu生成输入）：
    # 1、生成随机的block_table，并覆写原有bin文件
    # 2、将kv shape 统一转换成bsh后处理
    # 3、生成kv cache
    # 4、将kv cache dump成新的bin文件，供aclnn接口调用

    # gen block table [b, skv_max/block_size]
    block_table_shape = [b, math.ceil(skv_max / block_size)]
    block_num = block_num_min
    block_idx_list = torch.randperm(block_num, dtype=torch.int32)

    block_idx = 0
    # invalid block_id set as -1
    block_table = [-1] * block_table_shape[1]

    block_table = torch.tile(torch.tensor(block_table).to(torch.int32), (block_table_shape[0], 1))
    block_table_batch_idx = 0
    for idx in block_num_per_batch:
        for j in range(idx):
            block_table[block_table_batch_idx][j] = (block_idx_list[block_idx])
            block_idx += 1
        block_table_batch_idx += 1
    logging.debug("block_table %s", block_table)

    # gen kv cache. [block_num , block_size, H]
    k_cache = torch.zeros([block_num, block_size, n_kv, d_k], dtype=new_dtype)
    v_cache = torch.zeros([block_num, block_size, n_kv, d_v], dtype=new_dtype)

    # kv paddIng
    k_tensor_bsnd = torch.zeros((b, block_table_shape[1] * block_size, n_kv, d_k), dtype=new_dtype)
    v_tensor_bsnd = torch.zeros((b, block_table_shape[1] * block_size, n_kv, d_v), dtype=new_dtype)

    k_tensor_bsnd[:, :k_bsnd.shape[1], :, :] = k_bsnd[:, :, :, :]
    v_tensor_bsnd[:, :v_bsnd.shape[1], :, :] = v_bsnd[:, :, :, :]

    for b_idx in range(b):
        for block_i, kv_cache_blk_id in enumerate(block_table[b_idx]):
            block_offset = block_i * block_size
            if kv_cache_blk_id == -1:
                continue
            else:
                k_cache[kv_cache_blk_id, 0:block_size, :, :] = k_tensor_bsnd[
                                                            b_idx, block_offset:(block_offset + block_size), :, :]
                v_cache[kv_cache_blk_id, 0:block_size, :, :] = v_tensor_bsnd[
                                                            b_idx, block_offset:(block_offset + block_size), :, :]

    atten_out = torch.zeros(atten_out_shape, dtype=torch.float32)
    input_params_win_attn = [b, s_q, n_kv, n_q, d_q, win, d_k, d_v, scalar]
    atten_out = win_attn_calc(input_params_win_attn, actual_seq_list, q_bsnd, k_bsnd, v_bsnd, new_dtype, atten_out)

    q_nope = q[:, : kv_lora_rank]
    q_rope = q[:, kv_lora_rank:]

    k_cache_nope = k_cache[:, :, :, : kv_lora_rank]
    k_cache_rope = k_cache[:, :, :, kv_lora_rank:]
    input_params = [b, s_q, n_q, n_kv, skv_max, kv_lora_rank, qk_rope_dim, block_size, win]

    #output
    if 'bfloat' in str(dtypes):
        q_nope = q_nope.to(torch.float32)
        q_rope = q_rope.to(torch.float32)
        k_cache_nope = k_cache_nope.to(torch.float32)
        k_cache_rope = k_cache_rope.to(torch.float32)
        atten_out = atten_out.to(torch.float32)

    # dump golden file
    dump_file(q_nope, q_nope_path, dtypes)
    dump_file(q_rope, q_rope_path, dtypes)
    dump_file(k_cache_nope, k_cache_nope_path, dtypes)
    dump_file(k_cache_rope, k_cache_rope_path, dtypes)

    dump_file(block_table, block_table_path, np.int32)
    dump_file(actual_seq_list, actual_seq_len_path, np.int32)
    dump_file(atten_out, attent_out_path, np.float32)
    dump_file(input_params, input_param_path, np.int32)
    return 0


@GoldenRegister.reg_golden_func(
    case_names=[
        # ifa
        "DynamicWinAttenTest.test_DynAttn_nas_win_attn_s1_2_actseqlen_1024_mla_fp16_v1",
        "DynamicWinAttenTest.test_DynAttn_nas_win_attn_s1_2_actseqlen_1023_mla_fp16_unalign",
        "DynamicWinAttenTest.test_DynAttn_nas_win_attn_s1_2_actseqlen_1024_mla_bf16",
        "DynamicWinAttenTest.test_DynAttn_nas_win_attn_s1_2_actseqlen_1023_mla_bf16_unalign",

    ]
)
def win_attn_func(case_name: str, output: Path) -> bool:
    # output path
    q_nope_path = Path(output, 'q_nope.bin')
    q_rope_path = Path(output, 'q_rope.bin')

    k_cache_nope_path = Path(output, 'k_cache_nope.bin')
    k_cache_rope_path = Path(output, 'k_cache_rope.bin')

    block_table_path = Path(output, 'block_table.bin')
    actual_seq_len_path = Path(output, 'actual_seq_list.bin')
    attent_out_path = Path(output, 'atten_out.bin')
    input_param_path = Path(output, 'input_param.bin')

    complete = (q_nope_path.exists() and q_rope_path.exists() and k_cache_nope_path.exists() and
        k_cache_rope_path.exists() and block_table_path.exists() and actual_seq_len_path.exists()
        and attent_out_path.exists() and input_param_path.exists())
    complete = False

    if complete:
        logging.debug("Case(%s), Golden complete.", case_name)
        return True

    win = 1024
    if case_name.startswith('DynamicWinAttenTest.test_DynAttn_nas_win_attn_s1_2_actseqlen_1024_mla_fp16_v1'):
        b = 16
        s_q = 2
        n_q = 128
        skv = 1024
        block_size = 128
        n_kv = 1
        dtypes = np.float16
        gen_win_attn_data(win, b, s_q, n_q, skv, block_size, n_kv, dtypes, output)
    elif case_name.startswith('DynamicWinAttenTest.test_DynAttn_nas_win_attn_s1_2_actseqlen_1023_mla_fp16_unalign'):
        b = 16
        s_q = 2
        n_q = 128
        skv = 1023
        block_size = 128
        n_kv = 1
        dtypes = np.float16
        gen_win_attn_data(win, b, s_q, n_q, skv, block_size, n_kv, dtypes, output)
    elif case_name.startswith('DynamicWinAttenTest.test_DynAttn_nas_win_attn_s1_2_actseqlen_1024_mla_bf16'):
        b = 16
        s_q = 2
        n_q = 128
        skv = 1024
        block_size = 128
        n_kv = 1
        dtypes = bfloat16
        gen_win_attn_data(win, b, s_q, n_q, skv, block_size, n_kv, dtypes, output)
    elif case_name.startswith('DynamicWinAttenTest.test_DynAttn_nas_win_attn_s1_2_actseqlen_1023_mla_bf16_unalign'):
        b = 16
        s_q = 2
        n_q = 128
        skv = 1023
        block_size = 128
        n_kv = 1
        dtypes = bfloat16
        gen_win_attn_data(win, b, s_q, n_q, skv, block_size, n_kv, dtypes, output)
    else:
        logging.error("Can't get func to gen golden, Case(%s)", case_name)
    return True


def main() -> bool:
    # 用例名称
    case_name_list: List[str] = [
        "AscendOnBoardPaCostTest.test_page_attention_low_latency_cost",
    ]
    # 函数调用
    ret: bool = True
    for cs in case_name_list:
        output: Path = Path(g_src_root, "build/output/bin/golden", cs).resolve()
        output.mkdir(parents=True, exist_ok=True)
        ret = win_attn_func(case_name=cs, output=output)
    return ret


if __name__ == "__main__":
    exit(0 if main() else 1)
