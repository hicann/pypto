#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
GLM-4.5 MatMul AllReduce Add RmsNorm Module

This module implements a fused matmul, all-reduce, add, and RMSNorm operation for large-scale distributed models.
It efficiently combines computation and communication, reducing memory overhead and accelerating training and inference.

Main Functions:
    - matmul_allreduce_add_rmsnorm: Main function for fused matmul, all-reduce, add, and RMSNorm computation
"""

import multiprocessing as mp

import numpy as np
import torch
from torch._dynamo import allow_in_graph
from torch._subclasses import fake_tensor
import torch.distributed as dist
import torch_npu

import pypto

# 通信域初始化相关参数
MASTER_IP = "127.0.0.1"
MASTER_PORT = "50001"
WORLD_SIZE = 2
PHYSICAL_START_DEVICE_ID = 0
LOGICAL_RANKS = list(range(WORLD_SIZE))


def init_hccl_comm(logical_rank):
    physical_device_id = PHYSICAL_START_DEVICE_ID + logical_rank
    torch_npu.npu.set_device(physical_device_id)
    dist.init_process_group(
        backend="hccl",
        rank=logical_rank,
        world_size=WORLD_SIZE,
        init_method=f"tcp://{MASTER_IP}:{MASTER_PORT}",
    )
    group_handle = dist.new_group(backend="hccl", ranks=LOGICAL_RANKS)
    group_name = group_handle._get_backend(torch.device("npu")).get_hccl_comm_name(logical_rank)
    return [group_name]


def matmul_allreduce_add_rmsnorm_kernel(batch_size, attn_dim_per_tp, hidden_size, eps, group_name):
    batch_size = pypto.frontend.dynamic("batch_size")
    
    @pypto.frontend.jit()
    def kernel(
        in_tensor: pypto.Tensor((batch_size, attn_dim_per_tp), pypto.DT_BF16),
        matmul_weight: pypto.Tensor((hidden_size, attn_dim_per_tp), pypto.DT_BF16),
        residual: pypto.Tensor((batch_size, hidden_size), pypto.DT_BF16),
        gamma: pypto.Tensor((hidden_size,), pypto.DT_BF16),
        bias: pypto.Tensor((hidden_size,), pypto.DT_BF16),
        out_tensor: pypto.Tensor((batch_size, hidden_size), pypto.DT_BF16),
        residual_out: pypto.Tensor((batch_size, hidden_size), pypto.DT_BF16),
    ):
        in_tensor_mean_coff = 1.0 / hidden_size
        view_row_shape = 8
        bs_loop = (batch_size + view_row_shape - 1) // view_row_shape

        pypto.set_vec_tile_shapes(hidden_size)
        gamma_2d = pypto.reshape(gamma, [1, hidden_size], inplace=True)
        bias_2d = pypto.reshape(bias, [1, hidden_size], inplace=True)

        for bs_idx in pypto.loop(bs_loop, name="LOOP_MM_ALLREDUCE_ADD_RMSNORM", idx_name="bs_idx"):
            # 1. create shmem tesnor
            shmem_shape = [1, view_row_shape, hidden_size]
            shmem_data, shmem_signal = pypto.distributed.create_shmem_tensor(
                group_name, WORLD_SIZE, pypto.DT_FP32, shmem_shape)
            shmem_barrier_signal = pypto.distributed.create_shmem_signal(group_name, WORLD_SIZE)
            my_pe = pypto.distributed.my_symbolic_pe(group_name)
            for _ in pypto.loop(1, name="LOOP_MM_AR_ARMS_L0", idx_name="_"):
                tile_in_tensor = pypto.view(
                    in_tensor, (view_row_shape, in_tensor.shape[1]), [bs_idx * view_row_shape, 0],
                    valid_shape=[(batch_size - bs_idx * view_row_shape).min(view_row_shape), in_tensor.shape[1]])
                
                # 2. clear data
                pypto.set_vec_tile_shapes(view_row_shape, hidden_size)
                data_clear_dummy = pypto.distributed.shmem_clear(
                    shmem_data, shmem_shape, [0, 0, 0], pred=[tile_in_tensor], is_signal=False)
                signal_clear_dummy = pypto.distributed.shmem_clear(
                    shmem_signal, shmem_shape, [0, 0, 0], pred=[tile_in_tensor], is_signal=True)
                pypto.set_vec_tile_shapes(1, 8)
                barrier_dummy = pypto.distributed.shmem_barrier_all(
                    shmem_barrier_signal, [data_clear_dummy, signal_clear_dummy])

                # 3. matmul
                pypto.set_cube_tile_shapes([8, 8], [128, 256], [256, 512], True)
                matmul_result = pypto.matmul(tile_in_tensor, matmul_weight, in_tensor.dtype, b_trans=True)

                # 4. allreduce
                pypto.set_vec_tile_shapes(view_row_shape, hidden_size)
                for dyn_idx in range(WORLD_SIZE):
                    put_dummy = pypto.distributed.shmem_put(matmul_result, [0, 0, 0], shmem_data, dyn_idx,
                        put_op=pypto.AtomicType.ADD, pred=[barrier_dummy])
                    pypto.distributed.shmem_signal(shmem_signal, dyn_idx, 1, [1, 1] + shmem_shape, 
                        [dyn_idx, dyn_idx, 0, 0, 0], sig_op=pypto.AtomicType.ADD, pred=[put_dummy])
                wait_dummy = pypto.distributed.shmem_wait_until(shmem_signal, 1, WORLD_SIZE, [1, 1] + shmem_shape, 
                    [my_pe, my_pe, 0, 0, 0], clear_signal=True, pred=[tile_in_tensor])
                pypto.set_vec_tile_shapes(1, hidden_size)
                reduce_out = pypto.experimental.shmem_load(
                    shmem_data, my_pe, shmem_shape, [0, 0, 0], pred=[wait_dummy]
                )
                in_tensor_tile = pypto.cast(reduce_out, pypto.DT_BF16)

                # 5. Add RmsNorm
                residual_tile = pypto.view(
                    residual, (view_row_shape, hidden_size), [bs_idx * view_row_shape, 0],
                    valid_shape=[(batch_size - bs_idx * view_row_shape).min(view_row_shape), hidden_size])
                in_tensor_tile_fp32 = pypto.cast(in_tensor_tile, pypto.DT_FP32)

                # add
                residual_tile_fp32 = pypto.cast(residual_tile, pypto.DT_FP32)
                in_tensor_f32 = pypto.add(residual_tile_fp32, in_tensor_tile_fp32)

                # rms norm
                square = pypto.mul(in_tensor_f32, in_tensor_f32)
                mean_res = pypto.mul(square, in_tensor_mean_coff)
                reduce_asum = pypto.sum(mean_res, -1, True)
                reduce_sum = pypto.add(reduce_asum, eps)
                reduce_sqrt = pypto.sqrt(reduce_sum)
                res_div = pypto.div(in_tensor_f32, reduce_sqrt)

                hidden_bf16 = pypto.tensor([view_row_shape, hidden_size], pypto.DT_BF16, "hidden_bf16")
                residual_bf16_tmp = pypto.cast(in_tensor_f32, in_tensor.dtype)
                for tmp_idx in range(view_row_shape):
                    gamma_2d_fp32 = pypto.cast(gamma_2d, pypto.DT_FP32)
                    bias_2d_fp32 = pypto.cast(bias_2d, pypto.DT_FP32)
                    res_div_single = pypto.view(res_div, [1, hidden_size], [tmp_idx, 0])
                    res = pypto.mul(res_div_single, gamma_2d_fp32)
                    res_add = pypto.add(res, bias_2d_fp32)
                    in_tensor_norm = pypto.cast(res_add, in_tensor.dtype)
                    hidden_bf16[tmp_idx:tmp_idx + 1, 0:] = in_tensor_norm

                residual_out[bs_idx * pypto.symbolic_scalar(view_row_shape):, 0:] = residual_bf16_tmp
                out_tensor[bs_idx * pypto.symbolic_scalar(view_row_shape):, 0:] = hidden_bf16
    return kernel


def generate_golden_data():
    # 设置参数
    batch_size = 8
    attn_dim_per_tp = 1536
    hidden_size = 5120
    torch.manual_seed(42)

    #构造每张卡上需要的数据
    input_datas = []
    for _ in range(WORLD_SIZE):
        in_tensor = torch.randn((batch_size, attn_dim_per_tp), dtype=torch.bfloat16).share_memory_()
        matmul_weight = torch.randn((hidden_size, attn_dim_per_tp), dtype=torch.bfloat16).share_memory_()
        residual = torch.randn((batch_size, hidden_size), dtype=torch.bfloat16).share_memory_()
        gamma = torch.randn((hidden_size), dtype=torch.bfloat16).share_memory_()
        bias = torch.randn((hidden_size), dtype=torch.bfloat16).share_memory_()
        eps = 1e-5
        input_data = [in_tensor, matmul_weight, residual, gamma, bias, eps]
        input_datas.append(input_data)
    output_datas = matmul_allreduce_add_rmsnorm_result_golden(batch_size, hidden_size, input_datas)
    return input_datas, output_datas


def matmul_allreduce_add_rmsnorm_result_golden(batch_size, num, input_datas):
    output_datas = []
    # 计算 matmul & allreduce 结果， 该结果所有卡上一致
    matmul_allreduce_result_fp32 = torch.zeros((batch_size, num), dtype=torch.float32)
    for input_data in input_datas:
        in_tensor, matmul_weight = input_data[:2]
        matmul_result = torch.matmul(in_tensor, matmul_weight.T)
        matmul_allreduce_result_fp32 += matmul_result.to(torch.float32)

    # 计算各卡上add_rmsnorm之后的结果
    for input_data in input_datas:
        residual, gamma, bias, eps = input_data[-4:]
        res_add = residual.to(torch.float32) + matmul_allreduce_result_fp32
        mean_coff = 1.0 / res_add.shape[-1]
        in_tensor_f32 = res_add
        square = in_tensor_f32 * in_tensor_f32
        square = square.sum(dim=-1, keepdim=True)
        mean_res = square * mean_coff
        reduce_sum = mean_res + eps
        reduce_sqrt = torch.sqrt(reduce_sum)
        res_div = in_tensor_f32 / reduce_sqrt
        res = res_div * gamma.to(torch.float32)
        res = res + bias.to(res.dtype)
        output_data = [res.to(torch.bfloat16), in_tensor_f32.to(torch.bfloat16)]
        output_datas.append(output_data)
    return output_datas


def matmul_allreduce_add_rmsnorm_worker(intput_data, output_data, rank):
    groups = init_hccl_comm(rank)
    device = f'npu:{rank + PHYSICAL_START_DEVICE_ID}'
    in_tensor, matmul_weight, residual, gamma, bias, eps = intput_data
    golden_out_tensor, golden_residual = output_data

    out_tensor = torch.empty(residual.shape, dtype=torch.bfloat16, device=device)
    residual_out = torch.empty(residual.shape, dtype=torch.bfloat16, device=device)

    inputs = [in_tensor.to(device), matmul_weight.to(device), residual.to(device), gamma.to(device), 
        bias.to(device), out_tensor, residual_out]

    batch_size, attn_dim_per_tp = in_tensor.shape
    hidden_size = out_tensor.shape[1]

    matmul_allreduce_add_rmsnorm_kernel(batch_size, attn_dim_per_tp, hidden_size, eps, groups[0])(*inputs)

    np.testing.assert_allclose(
        np.array(out_tensor.cpu().flatten().tolist()), 
        np.array(golden_out_tensor.cpu().flatten().tolist()), 
        rtol=8e-3, 
        atol=7e-2,
    )

    np.testing.assert_allclose(
        np.array(residual_out.cpu().flatten().tolist()), 
        np.array(golden_residual.cpu().flatten().tolist()), 
        rtol=7e-2, 
        atol=8e-3,
    )


@allow_in_graph
def matmul_allreduce_add_rmsnorm(
    hidden_states: torch.Tensor,
    matmul_weight: torch.Tensor,
    residual: torch.Tensor,
    gamma: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
    group_name: str,
):
    if isinstance(hidden_states, fake_tensor.FakeTensor):
        return None, None
    
    out_tensor = torch.empty(residual.shape, dtype=torch.bfloat16, device=residual.device)
    residual_out = torch.empty(residual.shape, dtype=torch.bfloat16, device=residual.device)
    
    inputs = [hidden_states, matmul_weight, residual, gamma, bias, out_tensor, residual_out]

    batch_size, attn_dim_per_tp = hidden_states.shape
    hidden_size = out_tensor.shape[1]

    matmul_allreduce_add_rmsnorm_kernel(batch_size, attn_dim_per_tp, hidden_size, eps, group_name)(*inputs)

    return out_tensor, residual_out


def run_matmul_allreduce_add_rmsnorm():
    mp.set_start_method('spawn', force=True)
    processes = []
    input_datas, output_datas = generate_golden_data()
    for i in range(WORLD_SIZE):
        p = mp.Process(target=matmul_allreduce_add_rmsnorm_worker, args=(input_datas[i], output_datas[i], i))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()


def main():
    run_matmul_allreduce_add_rmsnorm()


if __name__ == '__main__':
    main()