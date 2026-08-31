#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Testcase configuration for pypto.add vector ST."""

from dataclasses import dataclass

from vector_testcase.vector_test_case import TensorConfig


@dataclass(frozen=True)
class AddConfig:
    case_index: int
    case_name: str
    input_tensors: tuple[TensorConfig, ...]
    output_tensors: tuple[TensorConfig, ...]
    a_shape: tuple[int, ...]
    b_shape: tuple[int, ...]
    output_shape: tuple[int, ...]
    view_shape: tuple[int, ...]
    tile_shape: tuple[int, ...]

    @classmethod
    def from_test_case(cls, case: dict) -> "AddConfig":
        return cls(
            case_index=case["case_index"],
            case_name=case["case_name"],
            input_tensors=tuple(TensorConfig.from_test_case(tensor) for tensor in case["input_tensors"]),
            output_tensors=tuple(TensorConfig.from_test_case(tensor) for tensor in case["output_tensors"]),
            a_shape=tuple(case["input_tensors"][0]["shape"]),
            b_shape=tuple(case["input_tensors"][1]["shape"]),
            output_shape=tuple(case["output_tensors"][0]["shape"]),
            view_shape=tuple(case["view_shape"]),
            tile_shape=tuple(case["tile_shape"]),
        )


ADD_TESTS = [
    {
        'case_index': 10,
        'case_name': 'Add_test_11',
        'operation': 'Add',
        'input_tensors': [
            {
                'name': 'input0',
                'shape': (1024, 256),
                'dtype': 'fp32',
                'format': 'ND',
                'need_trans': False,
                'data_range': {'min': -100, 'max': 100},
            },
            {
                'name': 'input1',
                'shape': (1024, 256),
                'dtype': 'fp32',
                'format': 'ND',
                'need_trans': False,
                'data_range': {'min': -1, 'max': 1},
            },
        ],
        'output_tensors': [
            {'name': 'output0', 'shape': (1024, 256), 'dtype': 'fp32', 'format': 'ND', 'need_trans': False}
        ],
        'view_shape': (512, 128),
        'tile_shape': (64, 128),
        'params': {},
        'source_case_index': 10,
    },
    {
        'case_index': 11,
        'case_name': 'Add_test_12',
        'operation': 'Add',
        'input_tensors': [
            {
                'name': 'input0',
                'shape': (32, 128, 1, 1),
                'dtype': 'fp32',
                'format': 'ND',
                'need_trans': False,
                'data_range': {'min': -10, 'max': 10},
            },
            {
                'name': 'input1',
                'shape': (32, 128, 1, 1),
                'dtype': 'fp32',
                'format': 'ND',
                'need_trans': False,
                'data_range': {'min': -10, 'max': 10},
            },
        ],
        'output_tensors': [
            {'name': 'output0', 'shape': (32, 128, 1, 1), 'dtype': 'fp32', 'format': 'ND', 'need_trans': False}
        ],
        'view_shape': (1, 49, 2, 2),
        'tile_shape': (31, 2, 3, 40),
        'params': {},
        'source_case_index': 11,
    },
    {
        'case_index': 18,
        'case_name': 'Add_test_19',
        'operation': 'Add',
        'input_tensors': [
            {
                'name': 'input0',
                'shape': (2, 4, 8, 16),
                'dtype': 'fp32',
                'format': 'ND',
                'need_trans': False,
                'data_range': {'min': -10, 'max': 10},
            },
            {
                'name': 'input1',
                'shape': (1, 1, 8, 1),
                'dtype': 'fp32',
                'format': 'ND',
                'need_trans': False,
                'data_range': {'min': -10, 'max': 10},
            },
        ],
        'output_tensors': [
            {'name': 'output0', 'shape': (2, 4, 8, 16), 'dtype': 'fp32', 'format': 'ND', 'need_trans': False}
        ],
        'view_shape': (2, 4, 8, 16),
        'tile_shape': (2, 4, 8, 16),
        'params': {},
        'source_case_index': 18,
    },
]
