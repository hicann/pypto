#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Testcase configuration for Hypot vector ST."""

from dataclasses import dataclass

from vector_testcase.vector_test_case import VectorConfig


@dataclass(frozen=True)
class HypotConfig(VectorConfig):
    OPERATION = 'Hypot'


HYPOT_TESTS = [
    {
        'case_index': 0,
        'case_name': 'Hypot_test_1',
        'operation': 'Hypot',
        'input_tensors': [
            {
                'name': 'input0',
                'shape': (1, 1),
                'dtype': 'fp32',
                'format': 'ND',
                'need_trans': False,
                'data_range': {'min': -10, 'max': 10},
            },
            {
                'name': 'input1',
                'shape': (1, 1),
                'dtype': 'fp32',
                'format': 'ND',
                'need_trans': False,
                'data_range': {'min': -10, 'max': 10},
            },
        ],
        'output_tensors': [{'name': 'output0', 'shape': (1, 1), 'dtype': 'fp32', 'format': 'ND', 'need_trans': False}],
        'view_shape': (32, 32),
        'tile_shape': (16, 16),
        'params': {},
        'source_case_index': 0,
    },
    {
        'case_index': 3,
        'case_name': 'Hypot_test_4',
        'operation': 'Hypot',
        'input_tensors': [
            {
                'name': 'input0',
                'shape': (32, 32),
                'dtype': 'fp32',
                'format': 'ND',
                'need_trans': False,
                'data_range': {'min': -100, 'max': 100},
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
        'view_shape': (32, 32),
        'tile_shape': (16, 16),
        'params': {},
        'source_case_index': 3,
    },
]
