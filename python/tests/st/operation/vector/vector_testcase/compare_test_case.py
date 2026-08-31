#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Testcase configuration for Compare vector ST."""

from dataclasses import dataclass

from vector_testcase.vector_test_case import VectorConfig


@dataclass(frozen=True)
class CompareConfig(VectorConfig):
    OPERATION = 'Compare'


COMPARE_TESTS = [
    {
        'case_index': 1,
        'case_name': 'Compare_test_1',
        'operation': 'Compare',
        'input_tensors': [
            {'name': 'input0', 'shape': (1, 8), 'dtype': 'fp32', 'data_range': {'min': -1, 'max': 1}, 'format': 'ND'},
            {
                'name': 'input1',
                'shape': (115, 8),
                'dtype': 'fp32',
                'data_range': {'min': -10, 'max': 10},
                'format': 'ND',
            },
        ],
        'output_tensors': [{'name': 'output0', 'shape': (115, 8), 'dtype': 'bool', 'format': 'ND'}],
        'view_shape': (20, 1),
        'tile_shape': (638, 24),
        'params': {'compare_op': 'eq', 'mode': 'bool', 'eq_num': 280, 'tile_block': 1, 'total_size': 6.88e-06},
        'source_case_index': 1,
    },
    {
        'case_index': 13,
        'case_name': 'Compare_test_13',
        'operation': 'Compare',
        'input_tensors': [
            {
                'name': 'input0',
                'shape': (50, 5, 1, 3),
                'dtype': 'fp16',
                'data_range': {'min': -1, 'max': 1},
                'format': 'ND',
            },
            {
                'name': 'input1',
                'shape': (50, 5, 1, 3),
                'dtype': 'fp16',
                'data_range': {'min': -1, 'max': 1},
                'format': 'ND',
            },
        ],
        'output_tensors': [{'name': 'output0', 'shape': (50, 5, 1, 3), 'dtype': 'bool', 'format': 'ND'}],
        'view_shape': (21, 1, 2, 1),
        'tile_shape': (2, 6, 142, 16),
        'params': {'compare_op': 'eq', 'mode': 'bool', 'eq_num': 23, 'tile_block': 11, 'total_size': 4.19e-06},
        'source_case_index': 13,
    },
]
