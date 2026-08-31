#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Testcase configuration for GatherElement vector ST."""

from dataclasses import dataclass

from vector_testcase.vector_test_case import VectorConfig


@dataclass(frozen=True)
class GatherelementConfig(VectorConfig):
    OPERATION = 'GatherElement'


GATHERELEMENT_TESTS = [
    {
        'case_index': 0,
        'case_name': 'GatherElement_test_0',
        'operation': 'GatherElement',
        'input_tensors': [
            {
                'name': 'input0',
                'shape': (10, 10),
                'dtype': 'fp32',
                'format': 'ND',
                'need_trans': False,
                'data_range': {'min': -1, 'max': 1},
            },
            {
                'name': 'input1',
                'shape': (5, 9),
                'dtype': 'int64',
                'format': 'ND',
                'need_trans': False,
                'data_range': {'min': 0, 'max': 10},
            },
        ],
        'output_tensors': [{'name': 'output0', 'shape': (5, 9), 'dtype': 'fp32', 'format': 'ND', 'need_trans': False}],
        'view_shape': (10, 8),
        'tile_shape': (3, 8),
        'params': {'axis': 0},
        'source_case_index': 0,
    },
    {
        'case_index': 4,
        'case_name': 'GatherElement_test_4',
        'operation': 'GatherElement',
        'input_tensors': [
            {
                'name': 'input0',
                'shape': (16, 16, 16),
                'dtype': 'fp32',
                'format': 'ND',
                'need_trans': False,
                'data_range': {'min': -1, 'max': 1},
            },
            {
                'name': 'input1',
                'shape': (8, 8, 8),
                'dtype': 'int64',
                'format': 'ND',
                'need_trans': False,
                'data_range': {'min': 0, 'max': 16},
            },
        ],
        'output_tensors': [
            {'name': 'output0', 'shape': (8, 8, 8), 'dtype': 'fp32', 'format': 'ND', 'need_trans': False}
        ],
        'view_shape': (16, 8, 8),
        'tile_shape': (16, 8, 8),
        'params': {'axis': 0},
        'source_case_index': 4,
    },
    {
        'case_index': 14,
        'case_name': 'GatherElement_test_14',
        'operation': 'GatherElement',
        'input_tensors': [
            {
                'name': 'input0',
                'shape': (32, 32),
                'dtype': 'fp32',
                'format': 'ND',
                'need_trans': False,
                'data_range': {'min': -10, 'max': 10},
            },
            {
                'name': 'input1',
                'shape': (32, 32),
                'dtype': 'int64',
                'format': 'ND',
                'need_trans': False,
                'data_range': {'min': 0, 'max': 32},
            },
        ],
        'output_tensors': [
            {'name': 'output0', 'shape': (32, 32), 'dtype': 'fp32', 'format': 'ND', 'need_trans': False}
        ],
        'view_shape': (32, 32),
        'tile_shape': (16, 32),
        'params': {'axis': 1},
        'source_case_index': 14,
    },
]
