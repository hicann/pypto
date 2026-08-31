#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Testcase configuration for TriU vector ST."""

from dataclasses import dataclass

from vector_testcase.vector_test_case import VectorConfig


@dataclass(frozen=True)
class TriuConfig(VectorConfig):
    OPERATION = 'TriU'


TRIU_TESTS = [
    {
        'case_index': 8,
        'case_name': 'TriU_test_8',
        'operation': 'TriU',
        'input_tensors': [
            {
                'name': 'input0',
                'shape': (20, 32),
                'dtype': 'bf16',
                'format': 'ND',
                'need_trans': False,
                'data_range': {'min': -10, 'max': 10},
            }
        ],
        'output_tensors': [
            {'name': 'output0', 'shape': (20, 32), 'dtype': 'bf16', 'format': 'ND', 'need_trans': False}
        ],
        'view_shape': (8, 20),
        'tile_shape': (3, 16),
        'params': {'diagonal': 0},
        'source_case_index': 8,
    },
    {
        'case_index': 10,
        'case_name': 'TriU_test_10',
        'operation': 'TriU',
        'input_tensors': [
            {
                'name': 'input0',
                'shape': (20, 56),
                'dtype': 'int8',
                'format': 'ND',
                'need_trans': False,
                'data_range': {'min': -10, 'max': 10},
            }
        ],
        'output_tensors': [
            {'name': 'output0', 'shape': (20, 56), 'dtype': 'int8', 'format': 'ND', 'need_trans': False}
        ],
        'view_shape': (8, 46),
        'tile_shape': (3, 32),
        'params': {'diagonal': 0},
        'source_case_index': 10,
    },
]
