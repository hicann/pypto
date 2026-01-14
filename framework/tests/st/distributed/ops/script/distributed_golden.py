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
import sys
import json
from pathlib import Path
from typing import List, Tuple

import math
import numpy as np
import torch

root_path: Path = Path(__file__).parent.parent.parent.parent.parent.parent.resolve()
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
    rank_size: int
    tile_row_shape: int
    tile_col_shape: int
    value_range: ValueRange


@dataclasses.dataclass(frozen=True)
class GenTensorCase:
    dtype: torch.dtype
    shape: Tuple[int, ...]
    rank_size: int
    value_range: ValueRange


@dataclasses.dataclass
class MoeCase:
    dtype: torch.dtype
    batch_size: int
    hidden_size: int
    shared_expert_num: int
    routed_expert_num: int
    top_k: int
    rank_size: int
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
    rank_size: int
    value_range: ValueRange


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
    rank_size = params['rank_size']
    input_tensor = config['input_tensors'][0]
    shape = tuple(input_tensor['shape'])
    dtype = get_dtype(input_tensor['dtype'])
    min_val, max_val = input_tensor['data_range']['min'], input_tensor['data_range']['max']
    tile_row_shape, tile_col_shape = params['tile_row_shape'], params['tile_col_shape']
    value_range = ValueRange(min_val=min_val, max_val=max_val)
    case = BaseCase(dtype=dtype, shape=shape, rank_size=rank_size, 
    tile_row_shape=tile_row_shape, tile_col_shape=tile_col_shape, value_range=value_range)
    return case


def validate_rank_size(rank_size: int) -> None:
    if rank_size <= 1:
        raise ValueError(f'rank_size must be greater than 1, got {rank_size}')


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
    if dtype in (torch.int32, torch.int16, torch.int8):
        return torch.randint(
            low=value_range.min_val, 
            high=value_range.max_val, 
            size=shape, 
            dtype=dtype
        )
    else:
        return torch.randn(shape, dtype=dtype)


def generate_random_tensor_list(gen_tensor_case: GenTensorCase) -> List[torch.Tensor]:
    return [generate_random_tensor(
        gen_tensor_case.shape, gen_tensor_case.dtype, gen_tensor_case.value_range
    ) for _ in range(gen_tensor_case.rank_size)]


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
    if 'test_cases' in json_data:
        test_cases = json_data['test_cases']
    else:
        test_cases = [json_data]
    test_cases.sort(key=lambda x: x['case_index'])
    return test_cases


def all_gather_and_save(
    inputs: List[torch.Tensor], rank_size: int, save_dir: Path, filename_prefix: str,
) -> torch.Tensor:
    gathered_output = torch.cat(inputs, dim=0)
    outputs = [gathered_output] * rank_size
    save_tensor_list(outputs, save_dir, filename_prefix)
    return outputs


def reduce_scatter_and_save(
    inputs: List[torch.Tensor], row: int, rank_size: int, save_dir: Path, filename_prefix: str,
) -> torch.Tensor:
    stacked_output = torch.stack(inputs, dim=0)
    reduced_output = torch.sum(stacked_output, dim=0).to(inputs[0].dtype)
    row_per_rank = row // rank_size
    outputs = [reduced_output[rank * row_per_rank: (rank + 1) * row_per_rank] for rank in range(rank_size)]
    save_tensor_list(outputs, save_dir, filename_prefix)
    return outputs


def all_reduce_and_save(
    inputs: List[torch.Tensor], rank_size: int, save_dir: Path, filename_prefix: str,
) -> torch.Tensor:
    stacked_output = torch.stack(inputs, dim=0)
    reduced_output = torch.sum(stacked_output, dim=0).to(inputs[0].dtype)
    outputs = [reduced_output for _ in range(rank_size)]
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
    rank_size = params['rank_size']
    min_val, max_val = input_tensor['data_range']['min'], input_tensor['data_range']['max']
    value_range = ValueRange(min_val=min_val, max_val=max_val)
    case = MoeCase(dtype=dtype, batch_size=batch_size, hidden_size=hidden_size, shared_expert_num=shared_expert_num,
        routed_expert_num=routed_expert_num, top_k=top_k, rank_size=rank_size, value_range=value_range)
    return case


def generate_moe_dispatch_input_data(case: MoeCase, save_dir: Path) \
    -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    gen_tensor_case = GenTensorCase(
        dtype=case.dtype, shape=(case.batch_size, case.hidden_size),
        rank_size=case.rank_size, value_range=case.value_range
    )
    x_list = generate_random_tensor_list_and_save(gen_tensor_case, save_dir, 'x',)
    gen_tensor_case = GenTensorCase(
        dtype=torch.float32, shape=(case.batch_size, case.routed_expert_num),
        rank_size=case.rank_size, value_range=case.value_range
    )
    scores_list = generate_random_tensor_list(gen_tensor_case)
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
    save_dir: Path,
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
        recv_counts = torch.sum(valid_count).item()
        recv_counts_tensor = torch.tensor(recv_counts, dtype=torch.int32)
        save_tensor(recv_counts_tensor, save_dir / f'recv_counts_rank_{rank_id}.bin')


def generate_moe_dispatch_case(case: MoeCase, save_dir: Path) -> None:
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


def get_moe_distributed_combine_input_data(dispatch_save_dir: Path, case: MoeCase) \
    -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
    expand_x_list = []
    assist_info_for_combine_list = []
    expert_scales_list = []
    row = get_dispatch_output_row(case)
    for rank in range(case.rank_size):
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

    return expand_x_list, assist_info_for_combine_list, expert_scales_list


def get_shared_out_and_save(
    case: MoeCase,
    expand_x_list: List[torch.Tensor],
    assist_info_for_combine_list: List[torch.Tensor],
    save_dir: Path,
) -> List[torch.Tensor]:
    shared_out_list = []
    for rank_id in range(case.rank_size):
        source_shared_expert_rank_id = get_shared_expert_rank_id(case, rank_id)
        expand_x = expand_x_list[source_shared_expert_rank_id]
        assist_info_for_combine = assist_info_for_combine_list[source_shared_expert_rank_id]
        mask = assist_info_for_combine[:, 0] == rank_id
        shared_out = expand_x[mask]
        shared_out_list.append(shared_out)
        save_tensor(shared_out, save_dir / f'share_y_rank_{rank_id}.bin')
    return shared_out_list


def get_routed_out_and_save(
    case: MoeCase,
    expand_x_list: List[torch.Tensor],
    assist_info_for_combine_list: List[torch.Tensor],
    expert_scales_list: List[torch.Tensor],
    save_dir: Path,
) -> List[torch.Tensor]:
    routed_out_list = [
        torch.zeros([case.batch_size, case.top_k, case.hidden_size], dtype=case.dtype)
        for _ in range(case.rank_size)
    ]
    for source_rank_id in range(case.shared_expert_num, case.rank_size):
        expand_x = expand_x_list[source_rank_id]
        assist_info_for_combine = assist_info_for_combine_list[source_rank_id]
        for token, (target_rank_id, token_id, k_offset) in zip(expand_x, assist_info_for_combine):
            if target_rank_id != -1:
                routed_out_list[target_rank_id][token_id, k_offset] = token
    for source_rank_id in range(case.rank_size):
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
    expand_x_list, assist_info_for_combine_list, expert_scales_list \
        = get_moe_distributed_combine_input_data(dispatch_save_dir, case)

    if case.shared_expert_num > 0:
        shared_out_list = get_shared_out_and_save(case, expand_x_list, assist_info_for_combine_list, save_dir)
    routed_out_list = get_routed_out_and_save(
        case,
        expand_x_list,
        assist_info_for_combine_list,
        expert_scales_list,
        save_dir,
    )

    for rank_id in range(case.rank_size):
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
    rank_size = params['rank_size']
    min_val, max_val = input_tensor['data_range']['min'], input_tensor['data_range']['max']
    value_range = ValueRange(min_val=min_val, max_val=max_val)
    case = AllGatherAttnPostReducescatterCase(dtype=dtype, batch_size=batch_size, seq_len=seq_len, 
        num_heads=num_heads, kv_lora_rank=kv_lora_rank, value_head_dim=value_head_dim, 
        output_hidden_size=output_hidden_size, rank_size=rank_size, value_range=value_range)
    return case


def gen_op_golden(op: str, golden_func, output_path: Path, case_index: int = None) -> bool:
    case_path: Path = Path(Path(__file__).parent.parent, 'test_case').resolve()
    case_file: Path = Path(case_path, op + '_st_test_cases.json').resolve()
    test_configs = load_test_cases_from_json(str(case_file))
    if len(test_configs) == 0:
        raise ValueError('Not find test cases, please check.')

    if case_index is None:
        for index, test_config in enumerate(test_configs):
            output_path1 = Path(output_path, str(index))
            output_path1.mkdir(parents=True, exist_ok=True)
            golden_func(test_config)
    else:
        golden_func(test_configs[case_index])
    return True


@GoldenRegister.reg_golden_func(
    case_names=[
        'TestAllgather/DistributedTest.TestAllgather',
    ]
)
def generate_all_gather_golden(case_name: str, output: Path, case_index: int = None) -> bool:
    def golden_func(config: dict):
        case = parse_base_case(config)
        validate_rank_size(case.rank_size)
        params = (*case.shape, get_dtype_num(case.dtype), case.tile_row_shape, case.tile_col_shape)
        save_params(params, output)
        gen_tensor_case = GenTensorCase(
            dtype=case.dtype, shape=case.shape, rank_size=case.rank_size, value_range=case.value_range
        )
        inputs = generate_random_tensor_list_and_save(gen_tensor_case, output, 'input')
        all_gather_and_save(inputs, case.rank_size, output, 'output')
    logging.debug('Case(%s), Golden creating...', case_name)
    return gen_op_golden('Allgather', golden_func, output, case_index)


@GoldenRegister.reg_golden_func(
    case_names=[
        'TestReducescatter/DistributedTest.TestReducescatter',
    ]
)
def generate_reduce_scatter_golden(case_name: str, output: Path, case_index: int = None) -> bool:
    def golden_func(config: dict):
        case = parse_base_case(config)
        validate_rank_size(case.rank_size)
        row = case.shape[0]
        if row % case.rank_size != 0:
            raise ValueError(
                'The first dimension of the input tensor must be an integer multiple of the rank size, '
                f'got row={row}, rank_size={case.rank_size}'
            )
        params = (*case.shape, get_dtype_num(case.dtype), case.tile_row_shape, case.tile_col_shape)
        save_params(params, output)
        gen_tensor_case = GenTensorCase(
            dtype=case.dtype, shape=case.shape, rank_size=case.rank_size, value_range=case.value_range
        )
        inputs = generate_random_tensor_list_and_save(gen_tensor_case, output, 'input')
        reduce_scatter_and_save(inputs, row, case.rank_size, output, 'output')
    logging.debug('Case(%s), Golden creating...', case_name)
    return gen_op_golden('Reducescatter', golden_func, output, case_index)


@GoldenRegister.reg_golden_func(
    case_names=[
        'TestAllreduce/DistributedTest.TestAllreduce',
    ]
)
def generate_all_reduce_golden(case_name: str, output: Path, case_index: int = None) -> bool:
    def golden_func(config: dict):
        case = parse_base_case(config)
        validate_rank_size(case.rank_size)
        row = case.shape[0]
        if row == 0:
            raise ValueError(
                'The first dimension of the input tensor must not be zero, '
                f'got row={row}, rank_size={case.rank_size}'
            )
        params = config['params']
        use_two_shot = params['use_two_shot']
        params = (*case.shape, get_dtype_num(case.dtype), case.tile_row_shape, case.tile_col_shape, use_two_shot)
        save_params(params, output)
        gen_tensor_case = GenTensorCase(
            dtype=case.dtype, shape=case.shape, rank_size=case.rank_size, value_range=case.value_range
        )
        inputs = generate_random_tensor_list_and_save(gen_tensor_case, output, 'input')
        all_reduce_and_save(inputs, case.rank_size, output, 'output')
    logging.debug('Case(%s), Golden creating...', case_name)
    return gen_op_golden('Allreduce', golden_func, output, case_index)


@GoldenRegister.reg_golden_func(
    case_names=[
        'TestAllreduce_Add_Allreduce/DistributedTest.TestAllreduce_Add_Allreduce',
    ]
)
def generate_allreduce_add_allreduce_golden(case_name: str, output: Path, case_index: int = None) -> bool:
    def golden_func(config: dict):
        case = parse_base_case(config)
        validate_rank_size(case.rank_size)
        params = (*case.shape, get_dtype_num(case.dtype))
        save_params(params, output)
        gen_tensor_case = GenTensorCase(
            dtype=case.dtype, shape=case.shape, rank_size=case.rank_size, value_range=case.value_range
        )
        inputs = generate_random_tensor_list_and_save(gen_tensor_case, output, 'input')
        all_reduce_outs = all_reduce_and_save(inputs, case.rank_size, output, 'all_reduce_out')
        add_outs = [all_reduce_outs[0] + all_reduce_outs[0] for _ in range(case.rank_size)]
        save_tensor_list(add_outs, output, 'add_out')
        all_reduce_and_save(add_outs, case.rank_size, output, 'out')
    logging.debug('Case(%s), Golden creating...', case_name)
    return gen_op_golden('Allreduce_Add_Allreduce', golden_func, output, case_index)


def generate_moe_dispatch_golden(case_name: str, output: Path, case_index: int = None) -> bool:
    def golden_func(config: dict):
        case = parse_moe_case(config)
        generate_moe_dispatch_case(case, output)
    logging.debug('Case(%s), Golden creating...', case_name)
    return gen_op_golden('MoeDispatch', golden_func, output, case_index)


@GoldenRegister.reg_golden_func(
    case_names=[
        'TestMoeDistributedCombine/DistributedTest.TestMoeDistributedCombine',
    ]
)
def generate_moe_distributed_combine_golden(case_name: str, output: Path, case_index: int = None) -> bool:
    def golden_func(config: dict):
        case = parse_moe_case(config)
        params = (case.batch_size, case.hidden_size, case.routed_expert_num, case.top_k, get_dtype_num(case.dtype))
        save_params(params, output)
        dispatch_save_dir = output / 'dispatch'
        dispatch_save_dir.mkdir(parents=True, exist_ok=True)
        generate_moe_dispatch_case(case, dispatch_save_dir)
        generate_moe_distributed_combine_case(case, output, dispatch_save_dir)
    logging.debug('Case(%s), Golden creating...', case_name)
    return gen_op_golden('MoeDistributedCombine', golden_func, output, case_index)


@GoldenRegister.reg_golden_func(
    case_names=[
        'TestAllgather_AttnPost_Reducescatter/DistributedTest.TestAllgather_AttnPost_Reducescatter',
    ]
)
def gen_allgather_attnpost_reducescatter_case(case_name: str, output: Path, case_index: int = None) -> bool:
    def golden_func(config: dict):
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
        all_gather_input_shape = (case.batch_size * case.seq_len * case.num_heads // case.rank_size, case.kv_lora_rank)
        gen_tensor_case = GenTensorCase(
            dtype=case.dtype, shape=all_gather_input_shape, rank_size=case.rank_size, value_range=case.value_range
        )
        all_gather_inputs = generate_random_tensor_list_and_save(gen_tensor_case, output, 'ag_in')
        attention_input = torch.cat(all_gather_inputs, dim=0)
        attention_input = attention_input.reshape([case.batch_size, case.num_heads, case.seq_len, case.kv_lora_rank])
        attention_input = torch.transpose(attention_input, 1, 2)
        attention_input = torch.reshape(
            attention_input, [case.batch_size * case.seq_len, case.num_heads, case.kv_lora_rank]
        )
        attention_input = torch.transpose(attention_input, 0, 1)

        reduce_scatter_inputs = []
        for rank in range(case.rank_size):
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
        reduce_scatter_and_save(reduce_scatter_inputs, case.batch_size * case.seq_len, case.rank_size, output, 'rs_out') 
    logging.debug('Case(%s), Golden creating...', case_name)
    return gen_op_golden('Allgather_AttnPost_Reducescatter', golden_func, output, case_index)