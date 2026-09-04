#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Testcase configuration for Round vector ST."""

from dataclasses import dataclass

from vector_testcase.vector_test_case import VectorConfig


@dataclass(frozen=True)
class RoundConfig(VectorConfig):
    OPERATION = 'Round'


ROUND_TESTS = [
    {
        'case_index': 0,
        'case_name': 'Round_test_0',
        'operation': 'Round',
        'input_tensors': [
            {'name': 'input0', 'shape': (16, 32), 'dtype': 'fp16', 'data_range': {'min': 0, 'max': 1}, 'format': 'ND'}
        ],
        'output_tensors': [{'name': 'output0', 'shape': (16, 32), 'dtype': 'fp16', 'format': 'ND'}],
        'view_shape': (16, 32),
        'tile_shape': (16, 32),
        'params': {'decimals': 0},
        'source_case_index': 0,
    },
    {
        'case_index': 1,
        'case_name': 'Round_test_1',
        'operation': 'Round',
        'input_tensors': [
            {'name': 'input0', 'shape': (16, 32), 'dtype': 'fp32', 'data_range': {'min': 0, 'max': 1}, 'format': 'ND'}
        ],
        'output_tensors': [{'name': 'output0', 'shape': (16, 32), 'dtype': 'fp32', 'format': 'ND'}],
        'view_shape': (16, 32),
        'tile_shape': (16, 32),
        'params': {'decimals': 1},
        'source_case_index': 1,
    },
    {
        'case_index': 2,
        'case_name': 'Round_test_2',
        'operation': 'Round',
        'input_tensors': [
            {
                'name': 'input0',
                'shape': (1024, 2048),
                'dtype': 'bf16',
                'data_range': {'min': 0, 'max': 1},
                'format': 'ND',
            }
        ],
        'output_tensors': [{'name': 'output0', 'shape': (1024, 2048), 'dtype': 'bf16', 'format': 'ND'}],
        'view_shape': (128, 128),
        'tile_shape': (32, 32),
        'params': {'decimals': 2},
        'source_case_index': 2,
    },
    {
        'case_index': 13,
        'case_name': 'Round_test_13',
        'operation': 'Round',
        'input_tensors': [
            {
                'name': 'input0',
                'shape': (128, 128),
                'dtype': 'int32',
                'data_range': {'min': -2147483648, 'max': 2147483647},
                'format': 'ND',
            }
        ],
        'output_tensors': [{'name': 'output0', 'shape': (128, 128), 'dtype': 'int32', 'format': 'ND'}],
        'view_shape': (40, 40),
        'tile_shape': (32, 32),
        'params': {'decimals': 0},
        'source_case_index': 13,
    },
]
