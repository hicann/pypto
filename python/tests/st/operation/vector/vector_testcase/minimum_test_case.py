#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Testcase configuration for Minimum vector ST."""

from dataclasses import dataclass

from vector_testcase.vector_test_case import VectorConfig


@dataclass(frozen=True)
class MinimumConfig(VectorConfig):
    OPERATION = 'Minimum'


MINIMUM_TESTS = [
    {
        'case_index': 12,
        'case_name': 'Minimum_test_13',
        'operation': 'Minimum',
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
                'shape': (1, 128, 1, 1),
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
        'params': {'scalar': None, 'scalar_type': None},
        'source_case_index': 12,
    },
    {
        'case_index': 19,
        'case_name': 'MinS_test_4',
        'operation': 'Minimum',
        'input_tensors': [
            {
                'name': 'input0',
                'shape': (32, 128, 64),
                'dtype': 'fp16',
                'format': 'ND',
                'need_trans': False,
                'data_range': {'min': -100, 'max': 100},
            }
        ],
        'output_tensors': [
            {'name': 'output0', 'shape': (32, 128, 64), 'dtype': 'fp16', 'format': 'ND', 'need_trans': False}
        ],
        'view_shape': (16, 16, 16),
        'tile_shape': (8, 8, 8),
        'params': {'scalar': 50.000195313, 'scalar_type': 'fp16'},
        'source_case_index': 19,
    },
]
