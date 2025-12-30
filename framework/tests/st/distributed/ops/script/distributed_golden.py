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

import dataclasses
import logging
import pathlib
from typing import List, Tuple

import math
import numpy as np
import torch

from golden_register import GoldenRegister

np.random.seed(0)
torch.manual_seed(0)

DTYPE_STR_TO_TORCH = {
    'bool': torch.bool,
    'uint8': torch.uint8,
    'int8': torch.int8,
    'int16': torch.int16,
    'int32': torch.int32,
    'int64': torch.int64,
    'float16': torch.float16,
    'float32': torch.float,
    'bfloat16': torch.bfloat16,
}

TORCH_DTYPE_TO_NUM = {
    torch.bool: 15,
    torch.uint8: 11,
    torch.int8: 1,
    torch.int16: 2,
    torch.int32: 3,
    torch.int64: 4,
    torch.float16: 6,
    torch.float: 7,
    torch.bfloat16: 8,
}


@dataclasses.dataclass(frozen=True)
class BaseCase:
    dtype: torch.dtype
    shape: Tuple[int, ...]
    rank_size: int


@dataclasses.dataclass
class MoeCase:
    dtype: torch.dtype
    batch_size: int
    hidden_size: int
    shared_expert_num: int
    routed_expert_num: int
    top_k: int
    rank_size: int

    def __post_init__(self):
        if self.top_k > self.routed_expert_num:
            raise ValueError(f'top_k ({self.top_k}) cannot exceed routed_expert_num ({self.routed_expert_num})')


@dataclasses.dataclass
class AllGatherAttnPostReducescatterCase:
    dtype: torch.dtype
    batch_size: int
    seq_len: int
    num_heads: int
    kv_lora_rank: int
    value_head_dim: int
    output_hidden_size: int
    rank_size: int


def get_dtype(dtype_str: str) -> torch.dtype:
    if dtype_str not in DTYPE_STR_TO_TORCH:
        raise ValueError(f'Unsupported dtype: {dtype_str}')
    return DTYPE_STR_TO_TORCH[dtype_str]


def get_dtype_num(dtype: torch.dtype) -> int:
    if dtype not in TORCH_DTYPE_TO_NUM:
        raise ValueError(f'Unsupported dtype: {dtype}')
    return TORCH_DTYPE_TO_NUM[dtype]


def parse_base_case(case_name: str, dim: int) -> BaseCase:
    parts = case_name.split('_')
    if len(parts) < dim + 2:
        raise ValueError(f'case_name {case_name} format is error.')
    rank_size = int(parts[-1])
    shape = tuple(map(int, parts[-(dim + 1):-1]))
    dtype = get_dtype(parts[-(dim + 2)])

    case = BaseCase(dtype=dtype, shape=shape, rank_size=rank_size)
    logging.info(f'Case {case_name}, case info: {case}')
    return case


def validate_rank_size(rank_size: int) -> None:
    if rank_size <= 1:
        raise ValueError(f'rank_size must be greater than 1, got {rank_size}')


def save_params(params: Tuple[int, ...], save_dir: pathlib.Path) -> None:
    params_tensor = torch.tensor(params, dtype=torch.int64)
    params_ndarray = params_tensor.numpy()
    params_ndarray.tofile(save_dir / 'params.bin')


def save_tensor(tensor: torch.Tensor, save_path: pathlib.Path) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    if tensor.dtype == torch.bfloat16:
        tensor = tensor.view(torch.int16)  # 仅改变 tensor 的 dtype 解释方式，内存布局不变
    tensor.numpy().tofile(save_path)


def save_tensor_list(tensors: List[torch.Tensor], save_dir: pathlib.Path, filename_prefix: str) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    for rank, tensor in enumerate(tensors):
        save_tensor(tensor, save_dir / f'{filename_prefix}_rank_{rank}.bin')


def generate_random_tensor(shape: Tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
    if dtype == torch.int32 or dtype == torch.int16 or dtype == torch.int8:
        return torch.randint(-10, 10, shape, dtype=dtype)
    else:
        return torch.randn(shape, dtype=dtype)


def generate_random_tensor_list(shape: Tuple[int, ...], dtype: torch.dtype, rank_size: int) -> List[torch.Tensor]:
    return [generate_random_tensor(shape, dtype) for rank in range(rank_size)]


def generate_random_tensor_list_and_save(
    shape: Tuple[int, ...], dtype: torch.dtype, rank_size: int, save_dir: pathlib.Path, filename_prefix: str,
) -> List[torch.Tensor]:
    tensor_list = generate_random_tensor_list(shape, dtype, rank_size)
    save_tensor_list(tensor_list, save_dir, filename_prefix)
    return tensor_list


def all_gather_and_save(
    inputs: List[torch.Tensor], rank_size: int, save_dir: pathlib.Path, filename_prefix: str,
) -> torch.Tensor:
    gathered_output = torch.cat(inputs, dim=0)
    outputs = [gathered_output] * rank_size
    save_tensor_list(outputs, save_dir, filename_prefix)
    return outputs


def reduce_scatter_and_save(
    inputs: List[torch.Tensor], row: int, rank_size: int, save_dir: pathlib.Path, filename_prefix: str,
) -> torch.Tensor:
    stacked_output = torch.stack(inputs, dim=0)
    reduced_output = torch.sum(stacked_output, dim=0).to(inputs[0].dtype)
    row_per_rank = row // rank_size
    outputs = [reduced_output[rank * row_per_rank: (rank + 1) * row_per_rank] for rank in range(rank_size)]
    save_tensor_list(outputs, save_dir, filename_prefix)
    return outputs


def all_reduce_and_save(
    inputs: List[torch.Tensor], rank_size: int, save_dir: pathlib.Path, filename_prefix: str,
) -> torch.Tensor:
    stacked_output = torch.stack(inputs, dim=0)
    reduced_output = torch.sum(stacked_output, dim=0).to(inputs[0].dtype)
    outputs = [reduced_output for _ in range(rank_size)]
    save_tensor_list(outputs, save_dir, filename_prefix)
    return outputs


def generate_all_gather_golden(case_name: str, save_dir: pathlib.Path):
    dim = 2
    case = parse_base_case(case_name, dim)
    row, col = case.shape
    rank_size, dtype = case.rank_size, case.dtype

    validate_rank_size(rank_size)

    params = (row, col, get_dtype_num(dtype))
    save_params(params, save_dir)

    inputs = generate_random_tensor_list_and_save((row, col), dtype, rank_size, save_dir, 'input')

    all_gather_and_save(inputs, rank_size, save_dir, 'output')


def generate_reduce_scatter_golden(case_name: str, save_dir: pathlib.Path):
    dim = 2
    case = parse_base_case(case_name, dim)
    row, col = case.shape
    rank_size, dtype = case.rank_size, case.dtype

    validate_rank_size(rank_size)
    if row % rank_size != 0:
        raise ValueError(
            'The first dimension of the input tensor must be an integer multiple of the rank size, '
            f'got row={row}, rank_size={rank_size}'
        )
    params = (row, col, get_dtype_num(dtype))
    save_params(params, save_dir)
    inputs = generate_random_tensor_list_and_save((row, col), dtype, rank_size, save_dir, 'input')
    reduce_scatter_and_save(inputs, row, rank_size, save_dir, 'output')


def generate_all_reduce_golden(case_name: str, save_dir: pathlib.Path):
    dim = 2
    case = parse_base_case(case_name, dim)
    row, col = case.shape
    rank_size, dtype = case.rank_size, case.dtype

    validate_rank_size(rank_size)
    if row == 0:
        raise ValueError(
            'The first dimension of the input tensor must not be zero, '
            f'got row={row}, rank_size={rank_size}'
        )

    params = (row, col, get_dtype_num(dtype))
    save_params(params, save_dir)
    inputs = generate_random_tensor_list_and_save((row, col), dtype, rank_size, save_dir, 'input')
    all_reduce_and_save(inputs, rank_size, save_dir, 'output')


def parse_moe_case(case_name: str) -> MoeCase:
    parts = case_name.split('_')
    return MoeCase(
        dtype=get_dtype(parts[-7]),
        batch_size=int(parts[-6]),
        hidden_size=int(parts[-5]),
        shared_expert_num=int(parts[-4]),
        routed_expert_num=int(parts[-3]),
        top_k=int(parts[-2]),
        rank_size=int(parts[-1]),
    )


def generate_moe_dispatch_input_data(case: MoeCase, save_dir: pathlib.Path) \
    -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    x_list = generate_random_tensor_list_and_save(
        (case.batch_size, case.hidden_size), case.dtype, case.rank_size, save_dir, 'x',
    )

    scores_list = generate_random_tensor_list(
        (case.batch_size, case.routed_expert_num), torch.float32, case.rank_size,
    )
    scores_list = [scores.sigmoid() for scores in scores_list]

    routed_expert_ids_list = []
    for rank in range(case.rank_size):
        scores = scores_list[rank]
        _, routed_expert_ids = torch.topk(scores, k=case.top_k)

        scales = scores.gather(1, routed_expert_ids)
        save_tensor(scales, save_dir / f'scale_rank_{rank}.bin')

        routed_expert_ids += case.shared_expert_num
        routed_expert_ids_list.append(routed_expert_ids)
        routed_expert_ids = routed_expert_ids.to(dtype=torch.int32)
        save_tensor(routed_expert_ids, save_dir / f'expert_ids_rank_{rank}.bin')

    return x_list, routed_expert_ids_list


def generate_combine_info_tensor(rank_id: int, token_id: int, k_offset: int) -> torch.Tensor:
    return torch.tensor([rank_id, token_id, k_offset], dtype=torch.int32).unsqueeze(0)


def get_shared_expert_rank_id(case: MoeCase, rank_id: int) -> int:
    if rank_id < case.shared_expert_num:
        return rank_id
    shared_expert_capacity = case.routed_expert_num // case.shared_expert_num
    return (rank_id - case.shared_expert_num) // shared_expert_capacity


def send_to_shared_experts(
    case: MoeCase,
    x_list: List[torch.Tensor],
    y_list: List[List[List[torch.Tensor]]],
    combine_info_list: List[List[List[torch.Tensor]]],
) -> None:
    expert_offset = 0
    for rank_id in range(case.rank_size):
        x = x_list[rank_id]
        target_shared_expert_rank_id = get_shared_expert_rank_id(case, rank_id)
        for token_id in range(case.batch_size):
            token = x[token_id].unsqueeze(0)
            y_list[target_shared_expert_rank_id][expert_offset].append(token)
            combine_info_list[target_shared_expert_rank_id][expert_offset].append(
                generate_combine_info_tensor(rank_id, token_id, case.top_k),
            )


def get_routed_expert_capacity(case: MoeCase) -> int:
    if case.shared_expert_num > 0:
        return 1
    return math.ceil(case.routed_expert_num / case.rank_size)


def get_routed_expert_rank_id_and_expert_offset(case: MoeCase, expert_id: int) -> Tuple[int, int]:
    if case.shared_expert_num > 0:
        return expert_id, 0
    routed_expert_capacity = get_routed_expert_capacity(case)
    return divmod(expert_id, routed_expert_capacity)


def send_to_routed_experts(
    case: MoeCase,
    x_list: List[torch.Tensor],
    routed_expert_ids_list: List[torch.Tensor],
    y_list: List[List[List[torch.Tensor]]],
    combine_info_list: List[List[List[torch.Tensor]]],
) -> None:
    for source_rank_id in range(case.rank_size):
        x = x_list[source_rank_id]
        routed_expert_ids = routed_expert_ids_list[source_rank_id]
        for token_id in range(case.batch_size):
            token = x[token_id].unsqueeze(0)
            for k_offset in range(case.top_k):
                target_routed_expert_id = routed_expert_ids[token_id][k_offset].item()
                target_routed_expert_rank_id, expert_offset = \
                    get_routed_expert_rank_id_and_expert_offset(case, target_routed_expert_id)
                y_list[target_routed_expert_rank_id][expert_offset].append(token)
                combine_info_list[target_routed_expert_rank_id][expert_offset].append(
                    generate_combine_info_tensor(source_rank_id, token_id, k_offset),
                )


def get_dispatch_output_row(case: MoeCase) -> int:
    if case.shared_expert_num > 0:
        return case.batch_size * case.rank_size
    else:
        max_send_token_num = case.batch_size * case.top_k * case.rank_size
        max_receive_token_num = case.batch_size * case.routed_expert_num
        return min(max_send_token_num, max_receive_token_num)


def collect_and_save(
    case: MoeCase,
    y_list: List[List[List[torch.Tensor]]],
    combine_info_list: List[List[List[torch.Tensor]]],
    save_dir: pathlib.Path,
) -> None:
    row = get_dispatch_output_row(case)
    routed_expert_capacity = get_routed_expert_capacity(case)
    for rank_id in range(case.rank_size):
        fixed_shape_y = torch.zeros((row, case.hidden_size), dtype=case.dtype)
        fixed_shape_combine_info = torch.full((row, 3), -1, dtype=torch.int32)
        valid_count = torch.zeros([routed_expert_capacity], dtype=torch.int32)
        y_offset, combine_info_offset = 0, 0
        for expert_offset in range(routed_expert_capacity):
            if y_list[rank_id][expert_offset]:
                actual_y = torch.cat(y_list[rank_id][expert_offset], dim=0)
                fixed_shape_y[y_offset:y_offset + actual_y.size(0)] = actual_y
                y_offset += actual_y.size(0)
                actual_combine_info = torch.cat(combine_info_list[rank_id][expert_offset], dim=0)
                fixed_shape_combine_info[combine_info_offset:combine_info_offset + actual_combine_info.size(0)] \
                    = actual_combine_info
                combine_info_offset += actual_combine_info.size(0)
                valid_count[expert_offset] = actual_y.size(0)
        save_tensor(fixed_shape_y, save_dir / f'y_rank_{rank_id}.bin')
        save_tensor(fixed_shape_combine_info, save_dir / f'combine_info_rank_{rank_id}.bin')
        save_tensor(valid_count, save_dir / f'valid_count_rank_{rank_id}.bin')


def generate_moe_dispatch_case(case: MoeCase, save_dir: pathlib.Path) -> None:
    params = (case.batch_size, case.hidden_size, case.routed_expert_num, case.top_k, get_dtype_num(case.dtype))
    save_params(params, save_dir)

    x_list, routed_expert_ids_list = generate_moe_dispatch_input_data(case, save_dir)

    routed_expert_capacity = get_routed_expert_capacity(case)
    y_list = [[[] for _ in range(routed_expert_capacity)] for _ in range(case.rank_size)]
    combine_info_list = [[[] for _ in range(routed_expert_capacity)] for _ in range(case.rank_size)]
    if case.shared_expert_num > 0:
        send_to_shared_experts(case, x_list, y_list, combine_info_list)
    send_to_routed_experts(case, x_list, routed_expert_ids_list, y_list, combine_info_list)
    collect_and_save(case, y_list, combine_info_list, save_dir)


def generate_moe_dispatch_golden(case_name: str, save_dir: pathlib.Path):
    case = parse_moe_case(case_name)
    generate_moe_dispatch_case(case, save_dir)


def get_moe_combine_input_data(dispatch_save_dir: pathlib.Path, case: MoeCase) \
    -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
    x_list = []
    combine_info_list = []
    scale_list = []
    row = get_dispatch_output_row(case)
    for rank in range(case.rank_size):
        x = torch.from_numpy(np.fromfile(dispatch_save_dir / f'y_rank_{rank}.bin'))
        x = x.view(dtype=case.dtype).view([row, case.hidden_size])
        x_list.append(x)

        combine_info = torch.from_numpy(
            np.fromfile(dispatch_save_dir / f'combine_info_rank_{rank}.bin', dtype=np.int32),
        )
        combine_info = combine_info.view([row, 3])
        combine_info_list.append(combine_info)

        scale = torch.from_numpy(np.fromfile(dispatch_save_dir / f'scale_rank_{rank}.bin', dtype=np.float32))
        scale = scale.view([case.batch_size, case.top_k, 1])
        scale_list.append(scale)

    return x_list, combine_info_list, scale_list


def get_shared_y_and_save(
    case: MoeCase,
    x_list: List[torch.Tensor],
    combine_info_list: List[torch.Tensor],
    save_dir: pathlib.Path,
) -> List[torch.Tensor]:
    shared_y_list = []
    for rank_id in range(case.rank_size):
        source_shared_expert_rank_id = get_shared_expert_rank_id(case, rank_id)
        x = x_list[source_shared_expert_rank_id]
        combine_info = combine_info_list[source_shared_expert_rank_id]
        mask = combine_info[:, 0] == rank_id
        shared_y = x[mask]
        shared_y_list.append(shared_y)
        save_tensor(shared_y, save_dir / f'share_y_rank_{rank_id}.bin')
    return shared_y_list


def get_routed_y_and_save(
    case: MoeCase,
    x_list: List[torch.Tensor],
    combine_info_list: List[torch.Tensor],
    scale_list: List[torch.Tensor],
    save_dir: pathlib.Path,
) -> List[torch.Tensor]:
    routed_y_list = [
        torch.zeros([case.batch_size, case.top_k, case.hidden_size], dtype=case.dtype)
        for _ in range(case.rank_size)
    ]
    for source_rank_id in range(case.shared_expert_num, case.rank_size):
        x = x_list[source_rank_id]
        combine_info = combine_info_list[source_rank_id]
        for token, (target_rank_id, token_id, k_offset) in zip(x, combine_info):
            if target_rank_id != -1:
                routed_y_list[target_rank_id][token_id, k_offset] = token
    for source_rank_id in range(case.rank_size):
        routed_y = routed_y_list[source_rank_id]
        save_tensor(routed_y, save_dir / f'moe_y_rank_{source_rank_id}.bin')
        scale = scale_list[source_rank_id]
        routed_y = routed_y.to(dtype=torch.float32)
        routed_y = routed_y * scale
        save_tensor(routed_y, save_dir / f'scaled_moe_y_rank_{source_rank_id}.bin')
        routed_y_list[source_rank_id] = torch.sum(routed_y, dim=1)
    return routed_y_list


def generate_combine_case(case: MoeCase, save_dir: pathlib.Path, dispatch_save_dir: pathlib.Path) -> None:
    x_list, combine_info_list, scale_list = get_moe_combine_input_data(dispatch_save_dir, case)

    if case.shared_expert_num > 0:
        shared_y_list = get_shared_y_and_save(case, x_list, combine_info_list, save_dir)
    routed_y_list = get_routed_y_and_save(case, x_list, combine_info_list, scale_list, save_dir)

    for rank_id in range(case.rank_size):
        routed_y = routed_y_list[rank_id]
        y = routed_y.to(dtype=torch.float32)
        if case.shared_expert_num > 0:
            y += shared_y_list[rank_id]
        save_tensor(y.to(dtype=case.dtype), save_dir / f'y_rank_{rank_id}.bin')


def generate_moe_combine_golden(case_name: str, save_dir: pathlib.Path):
    case = parse_moe_case(case_name)

    params = (case.batch_size, case.hidden_size, case.routed_expert_num, case.top_k, get_dtype_num(case.dtype))
    save_params(params, save_dir)

    dispatch_save_dir = save_dir / 'dispatch'
    dispatch_save_dir.mkdir(parents=True, exist_ok=True)
    generate_moe_dispatch_case(case, dispatch_save_dir)

    generate_combine_case(case, save_dir, dispatch_save_dir)


def generate_allgather_matmul_reducescatter_golden(case_name: str, save_dir: pathlib.Path):
    dim = 2
    case = parse_base_case(case_name, dim)
    row, col = case.shape
    rank_size, dtype = case.rank_size, case.dtype

    validate_rank_size(rank_size)

    params = (row, col, get_dtype_num(dtype))
    save_params(params, save_dir)
    
    all_gather_inputs = generate_random_tensor_list_and_save((row, col), dtype, rank_size, save_dir, 'input')

    all_gather_outputs = all_gather_and_save(all_gather_inputs, rank_size, save_dir, 'allgather')

    generate_random_tensor_list_and_save((col, col), dtype, rank_size, save_dir, 'matmul')  # 暂时没用到

    add_output = all_gather_outputs[0] + all_gather_outputs[0]
    add_outputs = []
    for rank in range(rank_size):
        save_tensor(add_output, save_dir / f'ag_add_rank_{rank}.bin')
        add_outputs.append(add_output)

    reduce_scatter_outputs = reduce_scatter_and_save(add_outputs, row * rank_size, rank_size, save_dir, 'rs')

    all_gather_and_save(reduce_scatter_outputs, rank_size, save_dir, 'double_allgather')


def gen_allgather_attnpost_reducescatter_case(case: AllGatherAttnPostReducescatterCase, save_dir: pathlib.Path) -> None:
    batch_size = case.batch_size
    seq_len = case.seq_len
    num_heads = case.num_heads
    kv_lora_rank = case.kv_lora_rank
    value_head_dim = case.value_head_dim
    output_hidden_size = case.output_hidden_size
    rank_size = case.rank_size
    dtype = case.dtype
    params = (
        batch_size,
        seq_len,
        num_heads,
        kv_lora_rank,
        value_head_dim,
        output_hidden_size,
        get_dtype_num(dtype),
    )
    save_params(params, save_dir)

    all_gather_input_shape = (batch_size * seq_len * num_heads // rank_size, kv_lora_rank)
    all_gather_inputs = generate_random_tensor_list_and_save(
        all_gather_input_shape, dtype, rank_size, save_dir, 'ag_in',
    )

    attention_input = torch.cat(all_gather_inputs, dim=0)
    attention_input = attention_input.reshape([batch_size, num_heads, seq_len, kv_lora_rank])
    attention_input = torch.transpose(attention_input, 1, 2)
    attention_input = torch.reshape(attention_input, [batch_size * seq_len, num_heads, kv_lora_rank])
    attention_input = torch.transpose(attention_input, 0, 1)

    reduce_scatter_inputs = []
    for rank in range(rank_size):
        lora_weight = generate_random_tensor((num_heads, kv_lora_rank, value_head_dim), dtype)
        save_tensor(lora_weight, save_dir / f'w_lora_rank_{rank}.bin')

        attention_output = torch.bmm(attention_input.to(torch.float32), lora_weight.to(torch.float32)).to(dtype=dtype)
        attention_output = torch.transpose(attention_output, 0, 1)
        attention_output = torch.reshape(attention_output, [batch_size * seq_len, num_heads * value_head_dim])

        output_weight = generate_random_tensor((num_heads * value_head_dim, output_hidden_size), dtype)
        save_tensor(output_weight, save_dir / f'w_out_rank_{rank}.bin')

        attention_output = torch.matmul(
            attention_output.to(dtype=torch.float32), output_weight.to(dtype=torch.float32)
        ).to(dtype=dtype)
        reduce_scatter_inputs.append(attention_output)

    reduce_scatter_and_save(reduce_scatter_inputs, batch_size * seq_len, rank_size, save_dir, 'rs_out')


def generate_allgather_attn_post_reducescatter_golden(case_name: str, save_dir: pathlib.Path) -> None:
    parts = case_name.split('_')
    if len(parts) < 8:
        raise ValueError(f'case_name {case_name} format is error.')
    case = AllGatherAttnPostReducescatterCase(
        dtype=get_dtype(parts[-8]),
        batch_size=int(parts[-7]),
        seq_len=int(parts[-6]),
        num_heads=int(parts[-5]),
        kv_lora_rank=int(parts[-4]),
        value_head_dim=int(parts[-3]),
        output_hidden_size=int(parts[-2]),
        rank_size=int(parts[-1]),
    )
    gen_allgather_attnpost_reducescatter_case(case, save_dir)


def generate_allreduce_add_allreduce_golden(case_name: str, save_dir: pathlib.Path) -> None:
    dim = 2
    case = parse_base_case(case_name, dim)
    row, col = case.shape
    rank_size, dtype = case.rank_size, case.dtype

    validate_rank_size(rank_size)

    params = (row, col, get_dtype_num(dtype))
    save_params(params, save_dir)
    
    inputs = generate_random_tensor_list_and_save((row, col), dtype, rank_size, save_dir, 'input')
    all_reduce_outs = all_reduce_and_save(inputs, rank_size, save_dir, 'all_reduce_out')
    add_outs = [all_reduce_outs[0] + all_reduce_outs[0] for _ in range(rank_size)]
    save_tensor_list(add_outs, save_dir, 'add_out')
    all_reduce_and_save(add_outs, rank_size, save_dir, 'out')


OPERATOR_DISPATCHERS = [
    ('all_gather', generate_all_gather_golden),
    ('reduce_scatter', generate_reduce_scatter_golden),
    ('moe_dispatch', generate_moe_dispatch_golden),
    ('moe_combine', generate_moe_combine_golden),
    ('allgather_matmul_reducescatter', generate_allgather_matmul_reducescatter_golden),
    ('allgather_attn_post_reducescatter', generate_allgather_attn_post_reducescatter_golden),
    ('all_reduce', generate_all_reduce_golden),
    ('allreduce_add_allreduce', generate_allreduce_add_allreduce_golden),
]


@GoldenRegister.reg_golden_func(
    case_names=[
        'DistributedTest.shmem_all_gather_int32_128_256_4',
        'DistributedTest.shmem_reduce_scatter_int32_128_256_4',
        'DistributedTest.shmem_allgather_attn_post_reducescatter_bfloat16_64_1_32_256_128_128_4',
        'DistributedTest.shmem_reduce_scatter_float16_128_256_4',
        'DistributedTest.shmem_reduce_scatter_bfloat16_32_32_4',
        'DistributedTest.shmem_all_reduce_int32_64_256_4',
        'DistributedTest.shmem_all_reduce_bfloat16_50_256_4',
        'DistributedTest.shmem_moe_combine_bfloat16_8_5120_0_160_8_4',
        'DistributedTest.shmem_moe_combine_bfloat16_256_5120_0_160_8_4',
        'DistributedTest.shmem_moe_combine_bfloat16_8_5120_0_160_8_8',
        'DistributedTest.shmem_moe_combine_bfloat16_256_5120_0_160_8_8',
        'DistributedTest.shmem_moe_dispatch_bfloat16_8_5120_0_160_8_4',
        'DistributedTest.shmem_moe_dispatch_bfloat16_8_5120_0_160_8_8',
        'DistributedTest.shmem_allreduce_add_allreduce_bfloat16_256_102400_4',
    ]
)
def generate_golden_case(case_name: str, output: pathlib.Path) -> bool:
    handler = None
    for keyword, func in OPERATOR_DISPATCHERS:
        if keyword in case_name:
            handler = func
            break

    if handler is None:
        raise ValueError(f"Can't find handler for case {case_name}")

    handler(case_name, output)
    logging.info('Generate golden success for %s', case_name)
    return True
