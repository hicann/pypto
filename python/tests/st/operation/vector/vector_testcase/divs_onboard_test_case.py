#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Testcase configuration for Divs vector ST."""

from dataclasses import dataclass

from vector_testcase.vector_test_case import VectorConfig


@dataclass(frozen=True)
class DivsOnboardConfig(VectorConfig):
    OPERATION = 'Divs'


DIVS_ONBOARD_TESTS = [
    {
        'case_index': 0,
        'case_name': 'Divs_test_1',
        'operation': 'Divs',
        'input_tensors': [
            {
                'name': 'input0',
                'shape': (64, 128),
                'dtype': 'fp32',
                'data_range': {'min': -10, 'max': 10},
                'format': 'ND',
            }
        ],
        'output_tensors': [{'name': 'output0', 'shape': (64, 128), 'dtype': 'fp32', 'format': 'ND'}],
        'view_shape': (7, 43),
        'tile_shape': (192, 120),
        'params': {'scalar': 1.5, 'scalar_type': 'fp32'},
        'source_case_index': 0,
    },
    {
        'case_index': 1,
        'case_name': 'Divs_test_2',
        'operation': 'Divs',
        'input_tensors': [
            {
                'name': 'input0',
                'shape': (16, 128, 64),
                'dtype': 'fp32',
                'data_range': {'min': -10, 'max': 10},
                'format': 'ND',
            }
        ],
        'output_tensors': [{'name': 'output0', 'shape': (16, 128, 64), 'dtype': 'fp32', 'format': 'ND'}],
        'view_shape': (3, 4, 8),
        'tile_shape': (5, 142, 8),
        'params': {'scalar': 10.0, 'scalar_type': 'fp32'},
        'source_case_index': 1,
    },
    {
        'case_index': 2,
        'case_name': 'Divs_test_3',
        'operation': 'Divs',
        'input_tensors': [
            {
                'name': 'input0',
                'shape': (64, 1024, 16, 128),
                'dtype': 'fp32',
                'data_range': {'min': -100, 'max': 100},
                'format': 'ND',
            }
        ],
        'output_tensors': [{'name': 'output0', 'shape': (64, 1024, 16, 128), 'dtype': 'fp32', 'format': 'ND'}],
        'view_shape': (25, 477, 6, 7),
        'tile_shape': (16, 1, 169, 8),
        'params': {'scalar': 0.125, 'scalar_type': 'fp32'},
        'source_case_index': 2,
    },
]
