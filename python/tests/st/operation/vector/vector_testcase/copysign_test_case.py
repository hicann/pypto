#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Testcase configuration for CopySign vector ST."""

from dataclasses import dataclass

from vector_testcase.vector_test_case import VectorConfig


@dataclass(frozen=True)
class CopysignConfig(VectorConfig):
    OPERATION = 'CopySign'


COPYSIGN_TESTS = [
    {
        'case_index': 1,
        'case_name': 'CopySign_test_1',
        'operation': 'CopySign',
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
        'view_shape': (16, 16),
        'tile_shape': (16, 16),
        'params': {},
        'source_case_index': 1,
    },
    {
        'case_index': 4,
        'case_name': 'CopySign_test_4',
        'operation': 'CopySign',
        'input_tensors': [
            {
                'name': 'input0',
                'shape': (32, 32),
                'dtype': 'int16',
                'format': 'ND',
                'need_trans': False,
                'data_range': {'min': -100, 'max': 100},
            },
            {
                'name': 'input1',
                'shape': (32, 32),
                'dtype': 'int16',
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
        'source_case_index': 4,
    },
    {
        'case_index': 5,
        'case_name': 'CopySign_test_5',
        'operation': 'CopySign',
        'input_tensors': [
            {
                'name': 'input0',
                'shape': (32, 32),
                'dtype': 'int32',
                'format': 'ND',
                'need_trans': False,
                'data_range': {'min': -100, 'max': 100},
            },
            {
                'name': 'input1',
                'shape': (32, 32),
                'dtype': 'int32',
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
        'source_case_index': 5,
    },
]
