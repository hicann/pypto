#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Testcase configuration for LReLU vector ST."""

from dataclasses import dataclass

from vector_testcase.vector_test_case import VectorConfig


@dataclass(frozen=True)
class LreluConfig(VectorConfig):
    OPERATION = 'LReLU'


LRELU_TESTS = [
    {
        'case_index': 0,
        'case_name': 'LReLU_test_0',
        'operation': 'LReLU',
        'input_tensors': [
            {
                'name': 'input0',
                'shape': (96, 3318),
                'dtype': 'fp32',
                'format': 'ND',
                'need_trans': False,
                'data_range': {'min': -5, 'max': 5},
            }
        ],
        'output_tensors': [
            {'name': 'output0', 'shape': (96, 3318), 'dtype': 'fp32', 'format': 'ND', 'need_trans': False}
        ],
        'view_shape': (16, 256),
        'tile_shape': (4, 128),
        'params': {'scalar': 0.008, 'scalar_type': 'fp32'},
        'source_case_index': 0,
    },
    {
        'case_index': 1,
        'case_name': 'LReLU_test_1',
        'operation': 'LReLU',
        'input_tensors': [
            {
                'name': 'input0',
                'shape': (128, 1, 8192),
                'dtype': 'fp32',
                'format': 'ND',
                'need_trans': False,
                'data_range': {'min': -23.32, 'max': 19.82},
            }
        ],
        'output_tensors': [
            {'name': 'output0', 'shape': (128, 1, 8192), 'dtype': 'fp32', 'format': 'ND', 'need_trans': False}
        ],
        'view_shape': (46, 2, 3511),
        'tile_shape': (32, 2, 32),
        'params': {'scalar': 0.008, 'scalar_type': 'fp32'},
        'source_case_index': 1,
    },
]
