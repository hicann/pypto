#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Testcase configuration for PReLU vector ST."""

from dataclasses import dataclass

from vector_testcase.vector_test_case import VectorConfig


@dataclass(frozen=True)
class PreluConfig(VectorConfig):
    OPERATION = 'PReLU'


PRELU_TESTS = [
    {
        'case_index': 4,
        'case_name': 'PReLU_test_4',
        'operation': 'PReLU',
        'input_tensors': [
            {
                'name': 'input0',
                'shape': (32, 32),
                'dtype': 'bf16',
                'format': 'ND',
                'need_trans': False,
                'data_range': {'min': -100, 'max': 100},
            },
            {
                'name': 'weight',
                'shape': (32,),
                'dtype': 'bf16',
                'format': 'ND',
                'need_trans': False,
                'data_range': {'min': 0.01, 'max': 0.3},
            },
        ],
        'output_tensors': [
            {'name': 'output0', 'shape': (32, 32), 'dtype': 'bf16', 'format': 'ND', 'need_trans': False}
        ],
        'view_shape': (32, 32),
        'tile_shape': (32, 32),
        'params': {},
        'source_case_index': 4,
    },
    {
        'case_index': 16,
        'case_name': 'PReLU_test_16',
        'operation': 'PReLU',
        'input_tensors': [
            {
                'name': 'input0',
                'shape': (1, 1, 16, 16),
                'dtype': 'fp32',
                'format': 'ND',
                'need_trans': False,
                'data_range': {'min': -10, 'max': 10},
            },
            {
                'name': 'weight',
                'shape': (1,),
                'dtype': 'fp32',
                'format': 'ND',
                'need_trans': False,
                'data_range': {'min': 0.01, 'max': 0.3},
            },
        ],
        'output_tensors': [
            {'name': 'output0', 'shape': (1, 1, 16, 16), 'dtype': 'fp32', 'format': 'ND', 'need_trans': False}
        ],
        'view_shape': (1, 1, 16, 16),
        'tile_shape': (1, 1, 16, 16),
        'params': {},
        'source_case_index': 16,
    },
]
