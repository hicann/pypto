#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Testcase configuration for Muls vector ST."""

from dataclasses import dataclass

from vector_testcase.vector_test_case import VectorConfig


@dataclass(frozen=True)
class MulsOnboardConfig(VectorConfig):
    OPERATION = 'Muls'


MULS_ONBOARD_TESTS = [
    {
        'case_index': 13,
        'case_name': 'Muls_test_13',
        'operation': 'Muls',
        'input_tensors': [
            {
                'name': 'input0',
                'shape': (16, 512),
                'dtype': 'fp32',
                'data_range': {'min': -100, 'max': 100},
                'format': 'ND',
            }
        ],
        'output_tensors': [{'name': 'output0', 'shape': (16, 512), 'dtype': 'fp32', 'format': 'ND'}],
        'view_shape': (2, 117),
        'tile_shape': (590, 16),
        'params': {'scalar': 0.044194, 'scalar_type': 'fp32'},
        'source_case_index': 13,
    },
    {
        'case_index': 14,
        'case_name': 'Muls_test_14',
        'operation': 'Muls',
        'input_tensors': [
            {
                'name': 'input0',
                'shape': (2048, 1, 127),
                'dtype': 'fp32',
                'data_range': {'min': -1, 'max': 1},
                'format': 'ND',
            }
        ],
        'output_tensors': [{'name': 'output0', 'shape': (2048, 1, 127), 'dtype': 'fp32', 'format': 'ND'}],
        'view_shape': (638, 2, 59),
        'tile_shape': (3, 83, 56),
        'params': {'scalar': 0.41667, 'scalar_type': 'fp32'},
        'source_case_index': 14,
    },
    {
        'case_index': 15,
        'case_name': 'Muls_test_15',
        'operation': 'Muls',
        'input_tensors': [
            {
                'name': 'input0',
                'shape': (16, 127, 128, 32),
                'dtype': 'fp32',
                'data_range': {'min': -10, 'max': 10},
                'format': 'ND',
            }
        ],
        'output_tensors': [{'name': 'output0', 'shape': (16, 127, 128, 32), 'dtype': 'fp32', 'format': 'ND'}],
        'view_shape': (7, 54, 6, 14),
        'tile_shape': (6, 12, 20, 8),
        'params': {'scalar': -1.0, 'scalar_type': 'fp32'},
        'source_case_index': 15,
    },
]
