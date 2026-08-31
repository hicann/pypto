#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Testcase configuration for Gcd vector ST."""

from dataclasses import dataclass

from vector_testcase.vector_test_case import VectorConfig


@dataclass(frozen=True)
class GcdOnboardConfig(VectorConfig):
    OPERATION = 'Gcd'


GCD_ONBOARD_TESTS = [
    {
        'case_index': 3,
        'case_name': 'Gcd_test_3',
        'operation': 'Gcd',
        'input_tensors': [
            {
                'name': 'input0',
                'shape': (257, 129),
                'dtype': 'int32',
                'format': 'ND',
                'need_trans': False,
                'data_range': {'min': -100000000, 'max': 100000000},
            },
            {
                'name': 'input1',
                'shape': (257, 129),
                'dtype': 'int32',
                'format': 'ND',
                'need_trans': False,
                'data_range': {'min': -100000000, 'max': 100000000},
            },
        ],
        'output_tensors': [
            {'name': 'output0', 'shape': (257, 129), 'dtype': 'int32', 'format': 'ND', 'need_trans': False}
        ],
        'view_shape': (64, 32),
        'tile_shape': (32, 16),
        'params': {},
        'source_case_index': 3,
    },
    {
        'case_index': 4,
        'case_name': 'Gcd_test_4',
        'operation': 'Gcd',
        'input_tensors': [
            {
                'name': 'input0',
                'shape': (129, 257),
                'dtype': 'int32',
                'format': 'ND',
                'need_trans': False,
                'data_range': {'min': -100000000, 'max': 100000000},
            },
            {
                'name': 'input1',
                'shape': (129, 1),
                'dtype': 'int32',
                'format': 'ND',
                'need_trans': False,
                'data_range': {'min': -100000000, 'max': 100000000},
            },
        ],
        'output_tensors': [
            {'name': 'output0', 'shape': (129, 257), 'dtype': 'int32', 'format': 'ND', 'need_trans': False}
        ],
        'view_shape': (32, 64),
        'tile_shape': (16, 32),
        'params': {},
        'source_case_index': 4,
    },
]
