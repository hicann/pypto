#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Testcase configuration for Remainder vector ST."""

from dataclasses import dataclass

from vector_testcase.vector_test_case import VectorConfig


@dataclass(frozen=True)
class RemainderOnboardConfig(VectorConfig):
    OPERATION = 'Remainder'


REMAINDER_ONBOARD_TESTS = [
    {
        'case_index': 3,
        'case_name': 'Remainder_2D_2',
        'operation': 'Remainder',
        'input_tensors': [
            {
                'name': 'input0',
                'shape': (32, 32),
                'dtype': 'bf16',
                'format': 'ND',
                'need_trans': False,
                'data_range': {'min': 1, 'max': 100},
            },
            {
                'name': 'input1',
                'shape': (32, 32),
                'dtype': 'bf16',
                'format': 'ND',
                'need_trans': False,
                'data_range': {'min': -100, 'max': -1},
            },
        ],
        'output_tensors': [
            {'name': 'output0', 'shape': (32, 32), 'dtype': 'bf16', 'format': 'ND', 'need_trans': False}
        ],
        'view_shape': (16, 16),
        'tile_shape': (3, 16),
        'params': {},
        'source_case_index': 3,
    },
    {
        'case_index': 12,
        'case_name': 'Remainder_2D_11',
        'operation': 'Remainder',
        'input_tensors': [
            {
                'name': 'input0',
                'shape': (1, 10),
                'dtype': 'int32',
                'format': 'ND',
                'need_trans': False,
                'data_range': {'min': -100, 'max': -1},
            },
            {
                'name': 'input1',
                'shape': (10, 1),
                'dtype': 'int32',
                'format': 'ND',
                'need_trans': False,
                'data_range': {'min': 1, 'max': 100},
            },
        ],
        'output_tensors': [
            {'name': 'output0', 'shape': (10, 10), 'dtype': 'int32', 'format': 'ND', 'need_trans': False}
        ],
        'view_shape': (8, 16),
        'tile_shape': (8, 16),
        'params': {},
        'source_case_index': 12,
    },
]
