#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Testcase configuration for Full vector ST."""

from dataclasses import dataclass

from vector_testcase.vector_test_case import VectorConfig


@dataclass(frozen=True)
class FullConfig(VectorConfig):
    OPERATION = 'Full'


FULL_TESTS = [
    {
        'case_index': 0,
        'case_name': 'Full_test_0',
        'operation': 'Full',
        'input_tensors': [
            {'name': 'input0', 'shape': (3072, 1), 'dtype': 'fp32', 'data_range': {'min': -1, 'max': 1}, 'format': 'ND'}
        ],
        'output_tensors': [{'name': 'output0', 'shape': (3072, 1), 'dtype': 'fp32', 'format': 'ND'}],
        'view_shape': (128, 1),
        'tile_shape': (32, 32),
        'params': {'scalar': 1e-05, 'scalar_type': 'fp32'},
        'source_case_index': 0,
    },
    {
        'case_index': 5,
        'case_name': 'Full_test_5',
        'operation': 'Full',
        'input_tensors': [
            {
                'name': 'input0',
                'shape': (6144, 64, 64),
                'dtype': 'fp32',
                'data_range': {'min': -100, 'max': 100},
                'format': 'ND',
            }
        ],
        'output_tensors': [{'name': 'output0', 'shape': (6144, 64, 64), 'dtype': 'fp32', 'format': 'ND'}],
        'view_shape': (64, 64, 64),
        'tile_shape': (32, 32, 32),
        'params': {'scalar': 2.0, 'scalar_type': 'fp32'},
        'source_case_index': 5,
    },
    {
        'case_index': 9,
        'case_name': 'Full_test_9',
        'operation': 'Full',
        'input_tensors': [
            {
                'name': 'input0',
                'shape': (48, 64, 128, 32),
                'dtype': 'fp32',
                'data_range': {'min': -100, 'max': 100},
                'format': 'ND',
            }
        ],
        'output_tensors': [{'name': 'output0', 'shape': (48, 64, 128, 32), 'dtype': 'fp32', 'format': 'ND'}],
        'view_shape': (48, 64, 32, 16),
        'tile_shape': (16, 16, 16, 8),
        'params': {'scalar': 2.0, 'scalar_type': 'fp32'},
        'source_case_index': 9,
    },
]
