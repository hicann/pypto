#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Testcase configuration for Pow vector ST."""

from dataclasses import dataclass

from vector_testcase.vector_test_case import VectorConfig


@dataclass(frozen=True)
class PowOnboardConfig(VectorConfig):
    OPERATION = 'Pow'


POW_ONBOARD_TESTS = [
    {
        'case_index': 0,
        'case_name': 'Pow_test_0',
        'operation': 'Pow',
        'input_tensors': [
            {
                'name': 'input0',
                'shape': (32, 32),
                'dtype': 'fp32',
                'format': 'ND',
                'need_trans': False,
                'data_range': {'min': 0, 'max': 10},
            },
            {
                'name': 'input1',
                'shape': (32, 32),
                'dtype': 'fp32',
                'format': 'ND',
                'need_trans': False,
                'data_range': {'min': -10, 'max': 10},
            },
        ],
        'output_tensors': [
            {'name': 'output0', 'shape': (32, 32), 'dtype': 'fp32', 'format': 'ND', 'need_trans': False}
        ],
        'view_shape': (16, 16),
        'tile_shape': (16, 16),
        'params': {},
        'source_case_index': 0,
    },
    {
        'case_index': 1,
        'case_name': 'Pow_test_1',
        'operation': 'Pow',
        'input_tensors': [
            {
                'name': 'input0',
                'shape': (32, 32, 32),
                'dtype': 'fp32',
                'format': 'ND',
                'need_trans': False,
                'data_range': {'min': 0, 'max': 10},
            },
            {
                'name': 'input1',
                'shape': (32, 32, 32),
                'dtype': 'fp32',
                'format': 'ND',
                'need_trans': False,
                'data_range': {'min': -10, 'max': 10},
            },
        ],
        'output_tensors': [
            {'name': 'output0', 'shape': (32, 32, 32), 'dtype': 'fp32', 'format': 'ND', 'need_trans': False}
        ],
        'view_shape': (2, 16, 16),
        'tile_shape': (2, 16, 16),
        'params': {},
        'source_case_index': 1,
    },
]
