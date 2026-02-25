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
import os
import sys
import json
from pathlib import Path
from typing import List, Tuple

import math
import numpy as np
import torch

root_path: Path = Path(Path(__file__).parent, "../../../../../").resolve()
scripts_path: Path = Path(root_path, 'tests/cmake/scripts')
if str(scripts_path) not in sys.path:
    sys.path.append(str(scripts_path))
from golden_register import GoldenRegister

helper_path: Path = Path(scripts_path, 'helper')
if str(helper_path) not in sys.path:
    sys.path.append(str(helper_path))
from test_case_loader import TestCaseLoader


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
class ValueRange:
    min_val: int
    max_val: int


@dataclasses.dataclass(frozen=True)
class BaseCase:
    dtype: torch.dtype
    shape: Tuple[int, ...]
    world_size: int
    tile_shape: Tuple[int, ...]
    value_range: ValueRange


@dataclasses.dataclass(frozen=True)
class GenTensorCase:
    dtype: torch.dtype
    shape: Tuple[int, ...]
    world_size: int
    value_range: ValueRange


@dataclasses.dataclass
class MoeCase:
    dtype: torch.dtype
    batch_size: int
    hidden_size: int
    shared_expert_num: int
    routed_expert_num: int
    top_k: int
    world_size: int
    value_range: ValueRange

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
    world_size: int
    value_range: ValueRange


@dataclasses.dataclass
class SendToRoutedExpertsArgs:
    case: MoeCase
    x_list: List[torch.Tensor]
    routed_expert_ids_list: List[torch.Tensor]
    y_list: List[List[List[torch.Tensor]]]
    combine_info_list: List[List[List[torch.Tensor]]]
    generate_combine: bool


@dataclasses.dataclass
class GetRoutedOutAndSaveArgs:
    case: MoeCase
    expand_x_list: List[torch.Tensor]
    assist_info_for_combine: List[torch.Tensor]
    expert_scales_list: List[torch.Tensor]
    recv_counts_list: List[torch.Tensor]
    save_dir: Path


def get_dtype(dtype_str: str) -> torch.dtype:
    if dtype_str not in DTYPE_STR_TO_TORCH:
        raise ValueError(f'Unsupported dtype: {dtype_str}')
    return DTYPE_STR_TO_TORCH[dtype_str]


def get_dtype_num(dtype: torch.dtype) -> int:
    if dtype not in TORCH_DTYPE_TO_NUM:
        raise ValueError(f'Unsupported dtype: {dtype}')
    return TORCH_DTYPE_TO_NUM[dtype]


def parse_base_case(config: dict) -> BaseCase:
    params = config['params']
    world_size = params['world_size']
    input_tensor = config['input_tensors'][0]
    shape = tuple(input_tensor['shape'])
    dtype = get_dtype(input_tensor['dtype'])
    min_val, max_val = input_tensor['data_range']['min'], input_tensor['data_range']['max']
    tile_shape = tuple(config['tile_shape'])
    value_range = ValueRange(min_val=min_val, max_val=max_val)
    case = BaseCase(dtype=dtype, shape=shape, world_size=world_size, tile_shape=tile_shape, 
        value_range=value_range)
    return case


def validate_world_size(world_size: int) -> None:
    if world_size <= 1:
        raise ValueError(f'world_size must be greater than 1, got {world_size}')


def save_params(params: Tuple[int, ...], save_dir: Path) -> None:
    params_tensor = torch.tensor(params, dtype=torch.int64)
    params_ndarray = params_tensor.numpy()
    params_ndarray.tofile(save_dir / 'params.bin')


def save_tensor(tensor: torch.Tensor, save_path: Path) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    if tensor.dtype == torch.bfloat16:
        tensor = tensor.view(torch.int16)  # 仅改变 tensor 的 dtype 解释方式，内存布局不变
    tensor.numpy().tofile(save_path)


def save_tensor_list(tensors: List[torch.Tensor], save_dir: Path, filename_prefix: str) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    for rank, tensor in enumerate(tensors):
        save_tensor(tensor, save_dir / f'{filename_prefix}_rank_{rank}.bin')


def generate_random_tensor(
    shape: Tuple[int, ...], dtype: torch.dtype, value_range: ValueRange
) -> torch.Tensor:
    spec_value_map = {
        'nan': np.nan,
        'inf': np.inf,
        '-inf': -np.inf
    }
    if value_range.min_val in spec_value_map:
        return torch.full(
            shape, 
            spec_value_map[value_range.min_val],
            dtype=dtype)
    if dtype in (torch.int32, torch.int16, torch.int8):
        return torch.randint(
            low=int(value_range.min_val), 
            high=int(value_range.max_val), 
            size=shape, 
            dtype=dtype
        )
    else:
        return torch.randn(shape, dtype=dtype)


def generate_random_tensor_list(gen_tensor_case: GenTensorCase) -> List[torch.Tensor]:
    return [generate_random_tensor(
        gen_tensor_case.shape, gen_tensor_case.dtype, gen_tensor_case.value_range
    ) for _ in range(gen_tensor_case.world_size)]


def generate_random_tensor_list_and_save(
    gen_tensor_case: GenTensorCase, save_dir: Path, filename_prefix: str,
) -> List[torch.Tensor]:
    tensor_list = generate_random_tensor_list(gen_tensor_case)
    save_tensor_list(tensor_list, save_dir, filename_prefix)
    return tensor_list


def load_test_cases_from_json(json_file: str) -> list:
    with open(json_file, 'r') as data_file:
        json_data = json.load(data_file)
    if json_data is None:
        raise ValueError(f'Json file {json_file} is invalid.')
    file_name = json_file.stem
    if 'test_cases' in json_data:
        test_cases = json_data['test_cases']
    else:
        test_cases = [json_data]
    for tc in test_cases:
        tc['file_name'] = file_name
    test_cases.sort(key=lambda x: x['case_index'])
    return test_cases


def all_gather_and_save(
    inputs: List[torch.Tensor], world_size: int, save_dir: Path, filename_prefix: str,
) -> torch.Tensor:
    gathered_output = torch.cat(inputs, dim=0)
    outputs = [gathered_output] * world_size
    save_tensor_list(outputs, save_dir, filename_prefix)
    return outputs


def reduce_scatter_and_save(
    inputs: List[torch.Tensor], row: int, world_size: int, save_dir: Path, filename_prefix: str,
) -> torch.Tensor:
    stacked_output = torch.stack(inputs, dim=0)
    reduced_output = torch.sum(stacked_output, dim=0).to(inputs[0].dtype)
    row_per_rank = row // world_size
    outputs = [reduced_output[rank * row_per_rank: (rank + 1) * row_per_rank] for rank in range(world_size)]
    save_tensor_list(outputs, save_dir, filename_prefix)
    return outputs


def all_reduce_and_save(
    inputs: List[torch.Tensor], world_size: int, save_dir: Path, filename_prefix: str,
) -> torch.Tensor:
    stacked_output = torch.stack(inputs, dim=0)
    reduced_output = torch.sum(stacked_output, dim=0).to(inputs[0].dtype)
    outputs = [reduced_output for _ in range(world_size)]
    save_tensor_list(outputs, save_dir, filename_prefix)
    return outputs


def parse_moe_case(config: dict) -> MoeCase:
    params = config['params']
    input_tensor = config['input_tensors'][0]
    dtype = get_dtype(input_tensor['dtype'])
    batch_size = params['batch_size']
    hidden_size = params['hidden_size']
    shared_expert_num = params['shared_expert_num']
    routed_expert_num = params['routed_expert_num']
    top_k = params['top_k']
    world_size = params['world_size']
    min_val, max_val = input_tensor['data_range']['min'], input_tensor['data_range']['max']
    value_range = ValueRange(min_val=min_val, max_val=max_val)
    case = MoeCase(dtype=dtype, batch_size=batch_size, hidden_size=hidden_size, shared_expert_num=shared_expert_num,
        routed_expert_num=routed_expert_num, top_k=top_k, world_size=world_size, value_range=value_range)
    return case


def generate_moe_dispatch_input_data(case: MoeCase, save_dir: Path) \
    -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    gen_tensor_case = GenTensorCase(
        dtype=case.dtype, shape=(case.batch_size, case.hidden_size),
        world_size=case.world_size, value_range=case.value_range
    )
    x_list = generate_random_tensor_list_and_save(gen_tensor_case, save_dir, 'x',)
    gen_tensor_case = GenTensorCase(
        dtype=torch.float32, shape=(case.batch_size, case.routed_expert_num),
        world_size=case.world_size, value_range=case.value_range
    )
    scores_list = generate_random_tensor_list(gen_tensor_case)
    scores_list = [scores.sigmoid() for scores in scores_list]

    routed_expert_ids_list = []
    for rank in range(case.world_size):
        scores = scores_list[rank]
        _, routed_expert_ids = torch.topk(scores, k=case.top_k)

        scales = scores.gather(1, routed_expert_ids)
        save_tensor(scales, save_dir / f'scale_rank_{rank}.bin')

        routed_expert_ids += case.shared_expert_num
        routed_expert_ids_list.append(routed_expert_ids)
        routed_expert_ids = routed_expert_ids.to(dtype=torch.int32)
        save_tensor(routed_expert_ids, save_dir / f'expert_ids_rank_{rank}.bin')

    return x_list, routed_expert_ids_list


def generate_combine_info_tensor(rank_id: int, token_id: int, k_offset: int, generate_combine: bool) -> torch.Tensor:
    dispatch_info_row = 64
    combine_cols = 3 if generate_combine else dispatch_info_row
    combine_info = torch.zeros(combine_cols, dtype=torch.int32)
    combine_info[combine_cols - 3] = rank_id
    combine_info[combine_cols - 2] = token_id
    combine_info[combine_cols - 1] = k_offset
    return combine_info.unsqueeze(0)


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
    generate_combine: bool,
) -> None:
    expert_offset = 0
    for rank_id in range(case.world_size):
        x = x_list[rank_id]
        target_shared_expert_rank_id = get_shared_expert_rank_id(case, rank_id)
        for token_id in range(case.batch_size):
            token = x[token_id].unsqueeze(0)
            y_list[target_shared_expert_rank_id][expert_offset].append(token)
            combine_info_list[target_shared_expert_rank_id][expert_offset].append(
                generate_combine_info_tensor(rank_id, token_id, case.top_k, generate_combine),
            )


def get_routed_expert_capacity(case: MoeCase) -> int:
    if case.shared_expert_num > 0:
        return 1
    return math.ceil(case.routed_expert_num / case.world_size)


def get_routed_expert_rank_id_and_expert_offset(case: MoeCase, expert_id: int) -> Tuple[int, int]:
    if case.shared_expert_num > 0:
        return expert_id, 0
    routed_expert_capacity = get_routed_expert_capacity(case)
    return divmod(expert_id, routed_expert_capacity)


def send_to_routed_experts(args: SendToRoutedExpertsArgs) -> None:
    case = args.case
    x_list = args.x_list
    routed_expert_ids_list = args.routed_expert_ids_list
    y_list = args.y_list
    combine_info_list = args.combine_info_list
    generate_combine = args.generate_combine
    for source_rank_id in range(case.world_size):
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
                    generate_combine_info_tensor(source_rank_id, token_id, k_offset, generate_combine),
                )


def get_dispatch_output_row(case: MoeCase) -> int:
    if case.shared_expert_num > 0:
        return case.batch_size * case.world_size
    else:
        max_send_token_num = case.batch_size * case.top_k * case.world_size
        max_receive_token_num = case.batch_size * case.routed_expert_num
        return min(max_send_token_num, max_receive_token_num)


def collect_and_save(
    case: MoeCase,
    y_list: List[List[List[torch.Tensor]]],
    combine_info_list: List[List[List[torch.Tensor]]],
    save_dir: Path,
    combine_generate: bool,
) -> None:
    row = get_dispatch_output_row(case)
    routed_expert_capacity = get_routed_expert_capacity(case)
    combine_cols = 3 if combine_generate else 64
    for rank_id in range(case.world_size):
        fixed_shape_y = torch.zeros((row, case.hidden_size), dtype=case.dtype)
        fixed_shape_combine_info = torch.zeros((row, combine_cols), dtype=torch.int32)
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
        recv_counts = torch.sum(valid_count).item()
        recv_counts_tensor = torch.tensor(recv_counts, dtype=torch.int32)
        save_tensor(recv_counts_tensor, save_dir / f'recv_counts_rank_{rank_id}.bin')


def generate_moe_dispatch_case(case: MoeCase, save_dir: Path, combine_generate: bool) -> None:
    params = (case.batch_size, case.hidden_size, case.routed_expert_num, case.top_k, get_dtype_num(case.dtype))
    save_params(params, save_dir)

    x_list, routed_expert_ids_list = generate_moe_dispatch_input_data(case, save_dir)

    routed_expert_capacity = get_routed_expert_capacity(case)
    y_list = [[[] for _ in range(routed_expert_capacity)] for _ in range(case.world_size)]
    combine_info_list = [[[] for _ in range(routed_expert_capacity)] for _ in range(case.world_size)]
    if case.shared_expert_num > 0:
        send_to_shared_experts(case, x_list, y_list, combine_info_list, combine_generate)
    args = SendToRoutedExpertsArgs(case=case, x_list=x_list, routed_expert_ids_list=routed_expert_ids_list,
    y_list=y_list, combine_info_list=combine_info_list, generate_combine=combine_generate)
    send_to_routed_experts(args)
    collect_and_save(case, y_list, combine_info_list, save_dir, combine_generate)


def get_moe_distributed_combine_input_data(dispatch_save_dir: Path, case: MoeCase) \
    -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
    expand_x_list = []
    assist_info_for_combine_list = []
    expert_scales_list = []
    recv_counts_list = []
    row = get_dispatch_output_row(case)
    for rank in range(case.world_size):
        expand_x = torch.from_numpy(np.fromfile(dispatch_save_dir / f'y_rank_{rank}.bin'))
        expand_x = expand_x.view(dtype=case.dtype).view([row, case.hidden_size])
        expand_x_list.append(expand_x)

        assist_info_for_combine = torch.from_numpy(
            np.fromfile(dispatch_save_dir / f'combine_info_rank_{rank}.bin', dtype=np.int32),
        )
        assist_info_for_combine = assist_info_for_combine.view([row, 3])
        assist_info_for_combine_list.append(assist_info_for_combine)

        expert_scales = torch.from_numpy(np.fromfile(dispatch_save_dir / f'scale_rank_{rank}.bin', dtype=np.float32))
        expert_scales = expert_scales.view([case.batch_size, case.top_k, 1])
        expert_scales_list.append(expert_scales)

        recvcounts = torch.from_numpy(np.fromfile(dispatch_save_dir / f'recv_counts_rank_{rank}.bin', dtype=np.int32))
        recv_counts_list.append(recvcounts)
    return expand_x_list, assist_info_for_combine_list, expert_scales_list, recv_counts_list


def get_shared_out_and_save(
    case: MoeCase,
    expand_x_list: List[torch.Tensor],
    assist_info_for_combine_list: List[torch.Tensor],
    save_dir: Path,
) -> List[torch.Tensor]:
    shared_out_list = []
    for rank_id in range(case.world_size):
        source_shared_expert_rank_id = get_shared_expert_rank_id(case, rank_id)
        expand_x = expand_x_list[source_shared_expert_rank_id]
        assist_info_for_combine = assist_info_for_combine_list[source_shared_expert_rank_id]
        mask = assist_info_for_combine[:, 0] == rank_id
        shared_out = expand_x[mask]
        shared_out_list.append(shared_out)
        save_tensor(shared_out, save_dir / f'share_y_rank_{rank_id}.bin')
    return shared_out_list


def get_routed_out_and_save(args: GetRoutedOutAndSaveArgs) -> List[torch.Tensor]:
    case = args.case
    expand_x_list = args.expand_x_list
    assist_info_for_combine_list = args.assist_info_for_combine
    expert_scales_list = args.expert_scales_list
    recv_counts_list = args.recv_counts_list
    save_dir = args.save_dir
    routed_out_list = [
        torch.zeros([case.batch_size, case.top_k, case.hidden_size], dtype=case.dtype)
        for _ in range(case.world_size)
    ]
    for source_rank_id in range(case.shared_expert_num, case.world_size):
        expand_x = expand_x_list[source_rank_id]
        assist_info_for_combine = assist_info_for_combine_list[source_rank_id]
        valid_row_shape = recv_counts_list[source_rank_id]
        for i, (token, (target_rank_id, token_id, k_offset)) in enumerate(zip(expand_x, assist_info_for_combine)):
            if i < valid_row_shape:
                routed_out_list[target_rank_id][token_id, k_offset] = token
    for source_rank_id in range(case.world_size):
        routed_out = routed_out_list[source_rank_id]
        save_tensor(routed_out, save_dir / f'moe_y_rank_{source_rank_id}.bin')
        expert_scales = expert_scales_list[source_rank_id]
        routed_out = routed_out.to(dtype=torch.float32)
        routed_out = routed_out * expert_scales
        save_tensor(routed_out, save_dir / f'scaled_moe_y_rank_{source_rank_id}.bin')
        routed_out_list[source_rank_id] = torch.sum(routed_out, dim=1)
    return routed_out_list


def generate_moe_distributed_combine_case(case: MoeCase, save_dir: Path, dispatch_save_dir: Path) \
    -> None:
    expand_x_list, assist_info_for_combine_list, expert_scales_list, recv_counts_list \
        = get_moe_distributed_combine_input_data(dispatch_save_dir, case)

    if case.shared_expert_num > 0:
        shared_out_list = get_shared_out_and_save(case, expand_x_list, assist_info_for_combine_list, save_dir)
    args = GetRoutedOutAndSaveArgs(case=case, expand_x_list=expand_x_list,
        assist_info_for_combine=assist_info_for_combine_list,
        expert_scales_list=expert_scales_list, recv_counts_list=recv_counts_list, save_dir=save_dir)
    routed_out_list = get_routed_out_and_save(args)

    for rank_id in range(case.world_size):
        routed_out = routed_out_list[rank_id]
        out = routed_out.to(dtype=torch.float32)
        if case.shared_expert_num > 0:
            out += shared_out_list[rank_id]
        save_tensor(out.to(dtype=case.dtype), save_dir / f'out_rank_{rank_id}.bin')


def generate_allgather_attn_post_reducescatter_case(config: dict) -> AllGatherAttnPostReducescatterCase:
    params = config['params']
    input_tensor = config['input_tensors'][0]
    dtype = get_dtype(input_tensor['dtype'])
    batch_size = params['batch_size']
    seq_len = params['seq_len']
    num_heads = params['num_heads']
    kv_lora_rank = params['kv_lora_rank']
    value_head_dim = params['value_head_dim']
    output_hidden_size = params['output_hidden_size']
    world_size = params['world_size']
    min_val, max_val = input_tensor['data_range']['min'], input_tensor['data_range']['max']
    value_range = ValueRange(min_val=min_val, max_val=max_val)
    case = AllGatherAttnPostReducescatterCase(dtype=dtype, batch_size=batch_size, seq_len=seq_len, 
        num_heads=num_heads, kv_lora_rank=kv_lora_rank, value_head_dim=value_head_dim, 
        output_hidden_size=output_hidden_size, world_size=world_size, value_range=value_range)
    return case


def generate_all_gather_golden(config: dict, output: Path) -> bool:
    case = parse_base_case(config)
    validate_world_size(case.world_size)
    params = (*case.shape, get_dtype_num(case.dtype), *case.tile_shape)
    save_params(params, output)
    gen_tensor_case = GenTensorCase(
        dtype=case.dtype, shape=case.shape, world_size=case.world_size, value_range=case.value_range
    )
    inputs = generate_random_tensor_list_and_save(gen_tensor_case, output, 'input')
    all_gather_and_save(inputs, case.world_size, output, 'output')


def generate_reduce_scatter_golden(config: dict, output: Path) -> bool:
    case = parse_base_case(config)
    validate_world_size(case.world_size)
    row = case.shape[0]
    if row % case.world_size != 0:
        raise ValueError(
            'The first dimension of the input tensor must be an integer multiple of the world size, '
            f'got row={row}, world_size={case.world_size}'
        )
    params = (*case.shape, get_dtype_num(case.dtype), *case.tile_shape)
    save_params(params, output)
    gen_tensor_case = GenTensorCase(
        dtype=case.dtype, shape=case.shape, world_size=case.world_size, value_range=case.value_range
    )
    inputs = generate_random_tensor_list_and_save(gen_tensor_case, output, 'input')
    reduce_scatter_and_save(inputs, row, case.world_size, output, 'output')


def generate_all_reduce_golden(config: dict, output: Path) -> bool:
    case = parse_base_case(config)
    validate_world_size(case.world_size)
    row = case.shape[0]
    if row == 0:
        raise ValueError(
            'The first dimension of the input tensor must not be zero, '
            f'got row={row}, world_size={case.world_size}'
        )
    params = config['params']
    use_two_shot = params['use_two_shot']
    params = (*case.shape, get_dtype_num(case.dtype), *case.tile_shape, use_two_shot)
    save_params(params, output)
    gen_tensor_case = GenTensorCase(
        dtype=case.dtype, shape=case.shape, world_size=case.world_size, value_range=case.value_range
    )
    inputs = generate_random_tensor_list_and_save(gen_tensor_case, output, 'input')
    all_reduce_and_save(inputs, case.world_size, output, 'output')


def generate_allreduce_add_allreduce_golden(config: dict, output: Path) -> bool:
    case = parse_base_case(config)
    validate_world_size(case.world_size)
    params = (*case.shape, get_dtype_num(case.dtype))
    save_params(params, output)
    gen_tensor_case = GenTensorCase(
        dtype=case.dtype, shape=case.shape, world_size=case.world_size, value_range=case.value_range
    )
    inputs = generate_random_tensor_list_and_save(gen_tensor_case, output, 'input')
    all_reduce_outs = all_reduce_and_save(inputs, case.world_size, output, 'all_reduce_out')
    add_outs = [all_reduce_outs[0] + all_reduce_outs[0] for _ in range(case.world_size)]
    save_tensor_list(add_outs, output, 'add_out')
    all_reduce_and_save(add_outs, case.world_size, output, 'out')


def generate_moe_dispatch_golden(config: dict, output: Path) -> bool:
    case = parse_moe_case(config)
    generate_moe_dispatch_case(case, output, False)


def generate_moe_distributed_combine_golden(config: dict, output: Path) -> bool:
    case = parse_moe_case(config)
    params = config['params']   # 整理一下
    use_v2 = params['use_v2']
    params = (case.batch_size, case.hidden_size, case.routed_expert_num, case.top_k, get_dtype_num(case.dtype),
        use_v2)
    save_params(params, output)
    dispatch_save_dir = output / 'dispatch'
    dispatch_save_dir.mkdir(parents=True, exist_ok=True)
    generate_moe_dispatch_case(case, dispatch_save_dir, True)
    generate_moe_distributed_combine_case(case, output, dispatch_save_dir)


def prepare_attention_input(case, output: Path):
    all_gather_input_shape = (
        case.batch_size * case.seq_len * case.num_heads // case.world_size,
        case.kv_lora_rank,
    )
    gen_tensor_case = GenTensorCase(
        dtype=case.dtype, shape=all_gather_input_shape, world_size=case.world_size, value_range=case.value_range
    )
    all_gather_inputs = generate_random_tensor_list_and_save(gen_tensor_case, output, 'ag_in')
    attention_input = torch.cat(all_gather_inputs, dim=0)
    attention_input = attention_input.reshape([case.batch_size, case.num_heads, case.seq_len, case.kv_lora_rank])
    attention_input = torch.transpose(attention_input, 1, 2)
    attention_input = attention_input.reshape([case.batch_size * case.seq_len, case.num_heads, case.kv_lora_rank])
    attention_input = torch.transpose(attention_input, 0, 1)
    return attention_input


def compute_attention_outputs(attention_input, case, output: Path):
    reduce_scatter_inputs = []
    for rank in range(case.world_size):
        lora_weight = generate_random_tensor(
            (case.num_heads, case.kv_lora_rank, case.value_head_dim), case.dtype, case.value_range
        )
        save_tensor(lora_weight, output / f'w_lora_rank_{rank}.bin')
        attention_output = torch.bmm(
            attention_input.to(torch.float32), lora_weight.to(torch.float32)).to(dtype=case.dtype
        )
        attention_output = torch.transpose(attention_output, 0, 1)
        attention_output = torch.reshape(
            attention_output, [case.batch_size * case.seq_len, case.num_heads * case.value_head_dim]
        )
        output_weight = generate_random_tensor(
            (case.num_heads * case.value_head_dim, case.output_hidden_size), case.dtype, case.value_range
        )
        save_tensor(output_weight, output / f'w_out_rank_{rank}.bin')
        attention_output = torch.matmul(
            attention_output.to(dtype=torch.float32), output_weight.to(dtype=torch.float32)
        ).to(dtype=case.dtype)
        reduce_scatter_inputs.append(attention_output)
    return reduce_scatter_inputs


def gen_allgather_attnpost_reducescatter_case(config: dict, output: Path) -> bool:
    case = generate_allgather_attn_post_reducescatter_case(config)
    params = (
        case.batch_size,
        case.seq_len,
        case.num_heads,
        case.kv_lora_rank,
        case.value_head_dim,
        case.output_hidden_size,
        get_dtype_num(case.dtype),
    )
    save_params(params, output)
    attention_input = prepare_attention_input(case, output)
    reduce_scatter_inputs = compute_attention_outputs(attention_input, case, output)
    reduce_scatter_and_save(reduce_scatter_inputs, case.batch_size * case.seq_len, case.world_size, output, 'rs_out')


def get_case_files() -> list[Path]:
    case_file = os.environ.get('JSON_PATH')
    case_path = Path(case_file) if case_file else Path(Path(__file__).parent.parent, 'test_case').resolve()
    if case_path.is_file():
        logging.info('loading single JSON file: %s', case_path)
        return [case_path]
    if case_path.is_dir():
        logging.info('loading all JSON files form directory: %s', case_path)
        files = list(case_path.glob("*.json"))
        files.sort(key=lambda x: x.name.lower())
        if not files:
            raise ValueError(f'JSON files found in the directory: %s', case_path)
        return files
    raise ValueError(f'Invalid path: %s. It must be either a valid file or a directory.', case_path)


def load_all_test_configs(case_files: list[Path]) -> list[dict]:
    all_test_configs = []
    for json_file in case_files:
        test_configs = load_test_cases_from_json(json_file)
        if test_configs:
            all_test_configs.extend(test_configs)
    if not all_test_configs:
        raise ValueError('No test cases loaded.')
    return all_test_configs


def generate_output_path(output: Path, test_config: dict, index: int = None) -> Path:
    case_str = f"{test_config['case_index']}_{test_config['case_name']}"
    operation = test_config['operation']
    file_name = test_config['file_name']
    if index is None:
        output_path = output.parent / operation / file_name / case_str 
    else:
        output_path = Path(*output.parts[:-2]) / operation / file_name / case_str 
    return output_path


OPERATOR_DISPATCHERS = {
    'AllGather': generate_all_gather_golden,
    'ReduceScatter': generate_reduce_scatter_golden,
    'AllReduce': generate_all_reduce_golden,
    'MoeDispatch': generate_moe_dispatch_golden,
    'MoeDistributedCombine': generate_moe_distributed_combine_golden,
    'AllReduceAddAllReduce': generate_allreduce_add_allreduce_golden,
    'AllGatherAttnPostReduceScatter': gen_allgather_attnpost_reducescatter_case,
}


def generate_single_golden(config: dict, output: Path):
    op_name = config['operation']
    if not op_name:
        raise ValueError(f'No operation field: {config}')
    handler = OPERATOR_DISPATCHERS[op_name]
    if handler is None:
        raise ValueError(f"Unsupported operation: {op_name}")
    handler(config, output)
    logging.info('Generate golden for success op: %s (case_name: %s)', op_name, config['case_name'])


@GoldenRegister.reg_golden_func(
    case_names=[
        'TestDistributedOps/DistributedTest.TestOps',
    ]
)
def generate_golden_case(case_name: str, output: Path, case_index: int = None) -> bool:
    case_files = get_case_files()
    all_test_configs = load_all_test_configs(case_files)
    if case_index is None:
        for test_config in all_test_configs:
            output_path1 = generate_output_path(output, test_config)
            output_path1.mkdir(parents=True, exist_ok=True)
            generate_single_golden(test_config, output_path1)
    else:
        if case_index >= len(all_test_configs):
            raise IndexError(f'case_index {case_index} out of range')
        test_config = all_test_configs[case_index]
        output = generate_output_path(output, test_config, case_index)
        output.mkdir(parents=True, exist_ok=True)
        generate_single_golden(test_config, output)
    return True