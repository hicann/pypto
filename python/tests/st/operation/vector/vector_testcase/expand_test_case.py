#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Testcase configuration for Expand vector ST."""

from dataclasses import dataclass

from vector_testcase.vector_test_case import VectorConfig


@dataclass(frozen=True)
class ExpandConfig(VectorConfig):
    OPERATION = 'Expand'


EXPAND_TESTS = [
    {
        'case_index': 4,
        'case_name': 'Expand_test_004',
        'operation': 'Expand',
        'input_tensors': [
            {
                'name': 'input0',
                'shape': (1, 16),
                'dtype': 'uint8',
                'format': 'ND',
                'need_trans': False,
                'data_range': {'min': 1, 'max': 1},
            }
        ],
        'output_tensors': [
            {'name': 'output0', 'shape': (4, 16), 'dtype': 'uint8', 'format': 'ND', 'need_trans': False}
        ],
        'view_shape': (4, 8),
        'tile_shape': (1, 4),
        'params': {},
        'source_case_index': 4,
    },
    {
        'case_index': 20,
        'case_name': 'Expand_test_020',
        'operation': 'Expand',
        'input_tensors': [
            {
                'name': 'input0',
                'shape': (1, 16),
                'dtype': 'fp32',
                'format': 'ND',
                'need_trans': False,
                'data_range': {'min': 1, 'max': 1},
            }
        ],
        'output_tensors': [{'name': 'output0', 'shape': (4, 16), 'dtype': 'fp32', 'format': 'ND', 'need_trans': False}],
        'view_shape': (4, 16),
        'tile_shape': (5, 8),
        'params': {},
        'source_case_index': 20,
    },
    {
        'case_index': 42,
        'case_name': 'Expand_test_042',
        'operation': 'Expand',
        'input_tensors': [
            {
                'name': 'input0',
                'shape': (16, 1, 1, 1),
                'dtype': 'fp32',
                'format': 'ND',
                'need_trans': False,
                'data_range': {'min': 1, 'max': 1},
            }
        ],
        'output_tensors': [
            {'name': 'output0', 'shape': (16, 4, 4, 4), 'dtype': 'fp32', 'format': 'ND', 'need_trans': False}
        ],
        'view_shape': (7, 3, 3, 3),
        'tile_shape': (3, 2, 2, 2),
        'params': {},
        'source_case_index': 42,
    },
]
