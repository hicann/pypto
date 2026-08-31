#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Input generation and result comparison shared by vector system tests."""

from itertools import islice, product
import math

import torch
from vector_testcase.vector_test_case import TORCH_DTYPES, VectorConfig


def make_inputs(config: VectorConfig, *, unique_indices: bool = False) -> list[torch.Tensor]:
    inputs = []
    for tensor in config.input_tensors:
        dtype = TORCH_DTYPES[tensor.dtype]
        minimum, maximum = tensor.data_min, tensor.data_max
        if minimum == "max":
            minimum = torch.finfo(dtype).max
        if maximum == "max":
            maximum = torch.finfo(dtype).max
        if minimum == maximum:
            data = torch.full(tensor.shape, maximum, dtype=dtype)
        elif dtype.is_floating_point:
            data = torch.empty(tensor.shape, dtype=dtype).uniform_(minimum, maximum)
        elif dtype == torch.bool:
            data = torch.randint(0, 2, tensor.shape, dtype=dtype)
        elif dtype in {torch.uint16, torch.uint32}:
            data = torch.randint(math.ceil(minimum), math.ceil(maximum), tensor.shape, dtype=torch.int64).to(dtype)
        else:
            data = torch.randint(math.ceil(minimum), math.ceil(maximum), tensor.shape, dtype=dtype)
        inputs.append(data)

    if unique_indices:
        ranges = [range(int(tensor.data_min), int(tensor.data_max)) for tensor in config.input_tensors[2:]]
        coordinates = list(islice(product(*ranges), config.input_tensors[1].shape[0]))
        for axis, tensor in enumerate(config.input_tensors[2:]):
            inputs[axis + 2] = torch.tensor(
                [coordinate[axis] for coordinate in coordinates], dtype=TORCH_DTYPES[tensor.dtype]
            )
    return inputs


def make_outputs(config: VectorConfig, device: str) -> list[torch.Tensor]:
    return [
        torch.empty(tensor.shape, dtype=TORCH_DTYPES[tensor.dtype], device=device) for tensor in config.output_tensors
    ]


def assert_outputs(actual: list[torch.Tensor], expected: list[torch.Tensor]) -> None:
    assert len(actual) == len(expected)
    for actual_tensor, expected_tensor in zip(actual, expected):
        actual_cpu = actual_tensor.cpu()
        expected_cpu = expected_tensor.to(actual_cpu.dtype).cpu()
        if not actual_cpu.is_floating_point():
            assert torch.equal(actual_cpu, expected_cpu)
            continue
        tolerance = 1e-2 if actual_cpu.dtype in {torch.float16, torch.bfloat16} else 5e-3
        assert torch.allclose(actual_cpu, expected_cpu, rtol=tolerance, atol=tolerance, equal_nan=True)
