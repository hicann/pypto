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
GLM-4.5 Distributed MoE Combine Module

This module implements the distributed token combine stage for
MoE in an expert parallel setting, based on open shared memory.

Main Functions:
    - moe_distributed_combine_kernel: JIT compiled combine kernel
"""

import dataclasses
from typing import Callable

import multiprocessing as mp
import numpy as np
import torch
import torch.distributed as dist
import torch_npu

import pypto

TensorList = list[torch.Tensor, ...]

np.random.seed(0)
torch.manual_seed(0)

MASTER_IP = '127.0.0.1'
MASTER_PORT = '50001'
WORLD_SIZE = 2
PHYSICAL_START_DEVICE_ID = 0
LOGICAL_RANK_IDS = list(range(WORLD_SIZE))


def check_cond(cond: bool, msg: str) -> None:
    if not cond:
        raise ValueError(msg)


def align_up(value: int, alignment: int) -> int:
    check_cond(
        alignment > 0 and (alignment & (alignment - 1)) == 0,
        f'alignment must be a power of two, but got {alignment}',
    )
    return (value + alignment - 1) & ~(alignment - 1)


def assert_allclose_with_eps(expected: torch.Tensor, actual: torch.Tensor, eps: float = 0.001) -> None:
    if expected.shape != actual.shape:
        raise ValueError(f'Shape mismatch: {expected.shape=}, {actual.shape=}')
    if expected.dtype != actual.dtype:
        raise ValueError(f'Dtype mismatch: {expected.dtype=}, {actual.dtype=}')

    numel = expected.numel()

    abs_err = torch.abs(expected - actual)
    rel_err = abs_err / torch.clamp(torch.abs(expected), min=1e-12)
    err_mask = (abs_err > eps) | (rel_err > eps)
    err_count = err_mask.sum().item()

    zero_mask = (torch.abs(expected) > 1e-6) & (torch.abs(actual) <= 1e-6)
    zero_count = zero_mask.sum().item()

    check_cond(err_count <= numel * eps and zero_count <= 1000, 'Allclose failed')


def init_hccl_comm(logical_rank_id: int) -> list[str, ...]:
    physical_device_id = PHYSICAL_START_DEVICE_ID + logical_rank_id
    torch_npu.npu.set_device(physical_device_id)
    dist.init_process_group(
        backend='hccl',
        rank=logical_rank_id,
        world_size=WORLD_SIZE,
        init_method=f'tcp://{MASTER_IP}:{MASTER_PORT}',
    )
    group_handle = dist.new_group(backend='hccl', ranks=LOGICAL_RANK_IDS)
    group_name = group_handle._get_backend(torch.device('npu')).get_hccl_comm_name(logical_rank_id)
    return [group_name]


@dataclasses.dataclass(frozen=True)
class MoeCase:
    batch_size: int
    hidden_size: int
    moe_expert_num: int
    topk: int
    pypto_data_type: pypto.DataType
    torch_data_type: torch.dtype
    world_size: int

    def __post_init__(self):
        check_cond(
            self.topk <= self.moe_expert_num,
            f'topk ({self.topk}) must be <= moe_expert_num ({self.moe_expert_num})',
        )


@dataclasses.dataclass(frozen=True)
class MoeDispatchGolden:
    x_list: TensorList
    expert_ids_list: TensorList
    expand_x_list: TensorList
    assist_info_for_combine_list: TensorList
    expert_token_nums_list: TensorList
    recv_counts_list: TensorList


@dataclasses.dataclass(frozen=True)
class MoeCombineInputOperands:
    expand_x: torch.Tensor
    assist_info_for_combine: torch.Tensor
    recv_counts: torch.Tensor
    expert_scales: torch.Tensor


def generate_random_tensor(shape: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
    float_dtypes = (torch.float16, torch.float32, torch.float64, torch.bfloat16)
    int_dtypes = (torch.int8, torch.int16, torch.int32, torch.int64)
    if dtype in float_dtypes:
        return torch.randn(shape, dtype=dtype)
    elif dtype in int_dtypes:
        return torch.randint(-10, 10, size=shape, dtype=dtype)
    else:
        raise ValueError(f'Unsupported dtype: {dtype}. Supported: {float_dtypes + int_dtypes}')


def generate_inputs(moe_case: MoeCase) -> tuple[TensorList, TensorList, TensorList]:
    x_list = [
        generate_random_tensor((moe_case.batch_size, moe_case.hidden_size), moe_case.torch_data_type)
        for _ in range(moe_case.world_size)
    ]
    moe_expert_ids_list = []
    topk_expert_scales_list = []

    for _ in range(moe_case.world_size):
        expert_scores = generate_random_tensor((moe_case.batch_size, moe_case.moe_expert_num), torch.float32)
        topk_expert_scores, moe_expert_ids = expert_scores.topk(k=moe_case.topk)
        topk_expert_scales = topk_expert_scores.softmax(dim=-1)
        topk_expert_scales_list.append(topk_expert_scales)
        moe_expert_ids = moe_expert_ids.to(dtype=torch.int32)
        moe_expert_ids_list.append(moe_expert_ids)

    return x_list, moe_expert_ids_list, topk_expert_scales_list


def get_moe_expert_num_per_rank(moe_case: MoeCase) -> int:
    return (moe_case.moe_expert_num + moe_case.world_size - 1) // moe_case.world_size


def get_moe_expert_rank_id_and_expert_offset(expert_id: int, moe_expert_num_per_rank: int) -> tuple[int, int]:
    return divmod(expert_id, moe_expert_num_per_rank)


def get_dispatch_output_row(moe_case: MoeCase) -> int:
    max_send_token_num = moe_case.topk * moe_case.batch_size * moe_case.world_size
    max_receive_token_num = moe_case.batch_size * moe_case.moe_expert_num
    return min(max_send_token_num, max_receive_token_num)


def dispatch_tokens(
    moe_case: MoeCase,
    x_list: TensorList,
    moe_expert_ids_list: TensorList,
    generate_for_dispatch: bool,
) -> tuple[TensorList, TensorList, TensorList, TensorList]:
    # 初始化变量
    moe_expert_num_per_rank = get_moe_expert_num_per_rank(moe_case)
    expand_x_per_expert = [[[] for _ in range(moe_expert_num_per_rank)] for _ in range(moe_case.world_size)]
    assist_info_for_combine_per_expert = [
        [[] for _ in range(moe_expert_num_per_rank)]
        for _ in range(moe_case.world_size)
    ]

    # 发送 token
    for sending_rank_id, (x, moe_expert_ids) in enumerate(zip(x_list, moe_expert_ids_list)):
        for token_id, (token, topk_moe_expert_ids) in enumerate(zip(x, moe_expert_ids)):
            for k_offset, moe_expert_id in enumerate(topk_moe_expert_ids):
                receiving_expert_id = moe_expert_id.item()
                receiving_rank_id, expert_offset = get_moe_expert_rank_id_and_expert_offset(
                    receiving_expert_id, moe_expert_num_per_rank)
                expand_x_per_expert[receiving_rank_id][expert_offset].append(token)
                assist_info_for_combine_per_expert[receiving_rank_id][expert_offset].append(
                    (sending_rank_id, token_id, k_offset))

    # 接收 token
    row = get_dispatch_output_row(moe_case)
    combine_info_col = 64 if generate_for_dispatch else 3
    expand_x_per_rank = []
    assist_info_for_combine_per_rank = []
    expert_token_nums_per_rank = []
    recv_counts_per_rank = []
    for logical_rank_id in range(moe_case.world_size):
        fixed_shape_expand_x = torch.zeros((row, moe_case.hidden_size), dtype=moe_case.torch_data_type)
        fixed_shape_assist_info_for_combine = torch.zeros((row, combine_info_col), dtype=torch.int32)
        expert_token_nums = torch.zeros([moe_expert_num_per_rank], dtype=torch.int32)
        offset = 0

        for expert_offset in range(moe_expert_num_per_rank):
            tokens = expand_x_per_expert[logical_rank_id][expert_offset]
            if tokens:
                actual_expand_x = torch.stack(tokens, dim=0)
                end = offset + actual_expand_x.size(0)
                fixed_shape_expand_x[offset:end] = actual_expand_x
                actual_assist_info_for_combine = torch.tensor(
                    assist_info_for_combine_per_expert[logical_rank_id][expert_offset],
                    dtype=torch.int32,
                )
                fixed_shape_assist_info_for_combine[offset:end, -3:] = actual_assist_info_for_combine
                expert_token_nums[expert_offset] = actual_expand_x.size(0)
                offset = end

        expand_x_per_rank.append(fixed_shape_expand_x)
        assist_info_for_combine_per_rank.append(fixed_shape_assist_info_for_combine)
        expert_token_nums_per_rank.append(expert_token_nums)
        recv_counts_per_rank.append(expert_token_nums.sum().unsqueeze(0))

    return expand_x_per_rank, assist_info_for_combine_per_rank, expert_token_nums_per_rank, recv_counts_per_rank


def combine_tokens(
    moe_case: MoeCase,
    expand_x_list: TensorList,
    assist_info_for_combine_list: TensorList,
    recv_counts_list: TensorList,
    expert_scales_list: TensorList,
) -> TensorList:
    # 初始化变量
    out_golden_list = []
    moe_expert_tokens_list = [
        torch.zeros([moe_case.batch_size, moe_case.topk, moe_case.hidden_size], dtype=moe_case.torch_data_type)
        for _ in range(moe_case.world_size)
    ]

    # 发送 token
    for expand_x, assist_info_for_combine, recv_counts in zip(
        expand_x_list, assist_info_for_combine_list, recv_counts_list
    ):
        for row_index in range(recv_counts.item()):
            token = expand_x[row_index]
            dispatch_sending_rank_id, token_id, k_offset = assist_info_for_combine[row_index]
            moe_expert_tokens_list[dispatch_sending_rank_id][token_id, k_offset] = token

    # 接收 token
    for moe_expert_tokens, expert_scales in zip(moe_expert_tokens_list, expert_scales_list):
        out_golden = (
            expert_scales.unsqueeze(1).
            matmul(moe_expert_tokens.to(torch.float32))
            .squeeze(1)
            .to(torch.bfloat16)
        )
        out_golden_list.append(out_golden)

    return out_golden_list


def generate_dispatch_golden(moe_case: MoeCase) -> MoeDispatchGolden:
    x_list, moe_expert_ids_list, _ = generate_inputs(moe_case)
    (
        expand_x_list,
        assist_info_for_combine_list,
        expert_token_nums_list,
        recv_counts_list,
    ) = dispatch_tokens(moe_case, x_list, moe_expert_ids_list, True)
    return MoeDispatchGolden(
        x_list,
        moe_expert_ids_list,
        expand_x_list,
        assist_info_for_combine_list,
        expert_token_nums_list,
        recv_counts_list,
    )


def generate_combine_golden(
    moe_case: MoeCase,
) -> tuple[TensorList, TensorList, TensorList, TensorList, TensorList]:
    x_list, moe_expert_ids_list, expert_scales_list = generate_inputs(moe_case)
    (
        expand_x_list,
        assist_info_for_combine_list,
        _,
        recv_counts_list,
    ) = dispatch_tokens(moe_case, x_list, moe_expert_ids_list, False)
    out_golden_list = combine_tokens(
        moe_case,
        expand_x_list,
        assist_info_for_combine_list,
        recv_counts_list,
        expert_scales_list,
    )
    return expand_x_list, assist_info_for_combine_list, recv_counts_list, expert_scales_list, out_golden_list


def moe_distributed_combine_kernel(
    moe_case: MoeCase,
    group_name: str,
) -> Callable[[pypto.Tensor, pypto.Tensor, pypto.Tensor, pypto.Tensor], pypto.Tensor]:
    batch_size = moe_case.batch_size
    hidden_size = moe_case.hidden_size
    moe_expert_num = moe_case.moe_expert_num
    topk = moe_case.topk
    data_type = moe_case.pypto_data_type
    world_size = moe_case.world_size
    row = min(topk * batch_size * world_size, batch_size * moe_expert_num)

    check_cond(batch_size == 8 or batch_size == 256, f'batch_size must be 8 or 256, but got {batch_size}')
    check_cond(hidden_size == 5120, f'hidden_size must be 5120, but got {hidden_size}')
    check_cond(moe_expert_num == 160, f'moe_expert_num must be 160, but got {moe_expert_num}')
    check_cond(topk == 8, f'topk must be 8, but got {topk}')
    check_cond(data_type == pypto.DT_BF16, f'data_type must be pypto.DT_BF16, but got {data_type}')
    check_cond(world_size == 4 or world_size == 8, f'world_size must be 4 or 8, but got {world_size}')
    check_cond(isinstance(group_name, str), f'type of group_name must be str, but got {type(group_name)}')
    check_cond(group_name.strip(), f"group_name can't be empty string")
    check_cond(
        1 <= len(group_name) < 128,
        f'The length of group_name only supports [1, 128), but got {len(group_name)}',
    )

    @pypto.frontend.jit()
    def kernel(
        expand_x: pypto.Tensor([row, hidden_size], data_type, format=pypto.TileOpFormat.TILEOP_ND),
        assist_info_for_combine: pypto.Tensor([row, 3], pypto.DT_INT32, format=pypto.TileOpFormat.TILEOP_ND),
        recv_counts: pypto.Tensor([1], pypto.DT_INT32, format=pypto.TileOpFormat.TILEOP_ND),
        expert_scales: pypto.Tensor([batch_size, topk], pypto.DT_FP32, format=pypto.TileOpFormat.TILEOP_ND),
    ) -> pypto.Tensor([batch_size, hidden_size], data_type, format=pypto.TileOpFormat.TILEOP_ND):
        # 创建 shmem_data 和 shmem_signal
        shmem_data, shmem_signal = pypto.distributed.create_shmem_tensor(
            group_name,
            world_size,
            expand_x.dtype,
            [1, topk * batch_size, hidden_size],
        )

        # 发送 token
        recv_counts_scalar = recv_counts[0]
        for row_index in pypto.loop(recv_counts_scalar, name='MOE_DISTRIBUTED_SEND', idx_name='row_index'):
            logical_rank_id = assist_info_for_combine[row_index, 0]
            token_id = assist_info_for_combine[row_index, 1]
            k_offset = assist_info_for_combine[row_index, 2]

            pypto.set_vec_tile_shapes(1, hidden_size)
            expand_x_tile = expand_x[row_index:row_index + 1, :hidden_size]
            pred_token = pypto.tensor([1, 1], pypto.DT_INT32)
            shmem_put_out = pypto.distributed.shmem_put(
                expand_x_tile,
                [0, topk * token_id + k_offset, 0],
                shmem_data,
                logical_rank_id,
                pred=[pred_token],
            )

            pypto.distributed.shmem_signal(
                shmem_signal,
                0,
                1,
                [1, 1, 1, 1, hidden_size],
                [logical_rank_id, 0, 0, token_id, 0],
                sig_op=pypto.AtomicType.ADD,
                pred=[shmem_put_out],
            )

        # 接收 token
        out = pypto.tensor([batch_size, hidden_size], expand_x.dtype)
        my_pe = pypto.distributed.my_symbolic_pe(group_name)
        for token_id in range(batch_size):
            pypto.set_vec_tile_shapes(1, hidden_size)
            pred_token = pypto.tensor([1, 1], pypto.DT_INT32)
            wait_until_out = pypto.distributed.shmem_wait_until(
                shmem_signal,
                pypto.OpType.EQ,
                topk,
                [1, 1, 1, 1, hidden_size],
                [my_pe, 0, 0, token_id, 0],
                pred=[pred_token],
            )

            pypto.set_vec_tile_shapes(topk, hidden_size)
            shmem_get_out = pypto.distributed.shmem_get(
                shmem_data,
                my_pe,
                [1, topk, hidden_size],
                [0, topk * token_id, 0],
                pred=[wait_until_out],
            )
            shmem_get_out = shmem_get_out.view([topk, hidden_size], [0, 0], valid_shape=[topk, hidden_size])

            pypto.set_vec_tile_shapes(topk // 2, hidden_size)
            shmem_get_out_fp32 = pypto.cast(shmem_get_out, pypto.DT_FP32)

            k_tile_shape = align_up(topk, 16)
            l0b_size = 65536
            n_tile_shape = l0b_size // pypto.bytes_of(pypto.DT_FP32) // k_tile_shape
            pypto.set_cube_tile_shapes([1, 1], [k_tile_shape, k_tile_shape], [n_tile_shape, n_tile_shape])
            expert_scales_tile = expert_scales[token_id:token_id + 1, :topk]
            matmul_out_fp32 = expert_scales_tile.matmul(shmem_get_out_fp32, pypto.DT_FP32)

            matmul_out_fp16 = pypto.cast(matmul_out_fp32, expand_x.dtype)

            out[token_id:, :] = matmul_out_fp16

        return out

    return kernel


def moe_distributed_combine(
    moe_case: MoeCase,
    input_operands: MoeCombineInputOperands,
    out_golden: torch.Tensor,
    logical_rank_id: int,
) -> None:
    groups = init_hccl_comm(logical_rank_id)

    expand_x = input_operands.expand_x
    assist_info_for_combine = input_operands.assist_info_for_combine
    recv_counts = input_operands.recv_counts
    expert_scales = input_operands.expert_scales

    expand_x.share_memory_()
    assist_info_for_combine.share_memory_()
    recv_counts.share_memory_()
    expert_scales.share_memory_()
    out_golden.share_memory_()

    physical_device_id = PHYSICAL_START_DEVICE_ID + logical_rank_id
    expand_x = expand_x.to(f'npu:{physical_device_id}')
    assist_info_for_combine = assist_info_for_combine.to(f'npu:{physical_device_id}')
    recv_counts = recv_counts.to(f'npu:{physical_device_id}')
    expert_scales = expert_scales.to(f'npu:{physical_device_id}')
    out_golden = out_golden.to(f'npu:{physical_device_id}')

    kernel = moe_distributed_combine_kernel(moe_case=moe_case, group_name=groups[0])
    out_actual = kernel(expand_x, assist_info_for_combine, recv_counts, expert_scales)

    assert_allclose_with_eps(out_golden.cpu(), out_actual.cpu())


def test_moe_distributed_combine() -> None:
    mp.set_start_method('spawn', force=True)
    processes = []

    moe_case = MoeCase(
        batch_size=8,
        hidden_size=5120,
        moe_expert_num=160,
        topk=8,
        pypto_data_type=pypto.DT_BF16,
        torch_data_type=torch.bfloat16,
        world_size=WORLD_SIZE,
    )

    (
        expand_x_list,
        assist_info_for_combine_list,
        recv_counts_list,
        expert_scales_list,
        out_golden_list,
    ) = generate_combine_golden(moe_case)
    for expand_x, assist_info_for_combine, recv_counts, expert_scales, out_actual, logical_rank_id in zip(
        expand_x_list,
        assist_info_for_combine_list,
        recv_counts_list,
        expert_scales_list,
        out_golden_list,
        LOGICAL_RANK_IDS,
    ):
        input_operands = MoeCombineInputOperands(expand_x, assist_info_for_combine, recv_counts, expert_scales)
        p = mp.Process(
            target=moe_distributed_combine,
            args=(moe_case, input_operands, out_actual, logical_rank_id),
        )
        p.start()
        processes.append(p)
    for p in processes:
        p.join()


if __name__ == '__main__':
    test_moe_distributed_combine()
