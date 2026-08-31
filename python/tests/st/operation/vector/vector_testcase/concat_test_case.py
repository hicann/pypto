#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Testcase configuration for Concat vector ST."""

from dataclasses import dataclass

from vector_testcase.vector_test_case import VectorConfig


@dataclass(frozen=True)
class ConcatConfig(VectorConfig):
    OPERATION = 'Concat'


CONCAT_TESTS = [
    {
        'case_index': 0,
        'case_name': 'Concat_test_0',
        'operation': 'Concat',
        'input_tensors': [
            {
                'name': 'input0',
                'shape': (64, 64),
                'dtype': 'fp32',
                'data_range': {'min': -10, 'max': 10},
                'format': 'ND',
            },
            {
                'name': 'input1',
                'shape': (64, 64),
                'dtype': 'fp32',
                'data_range': {'min': -10, 'max': 10},
                'format': 'ND',
            },
        ],
        'output_tensors': [{'name': 'output0', 'shape': (64, 128), 'dtype': 'fp32', 'format': 'ND'}],
        'view_shape': (32, 32),
        'tile_shape': (16, 32),
        'params': {'axis': 1},
        'source_case_index': 0,
    },
    {
        'case_index': 1,
        'case_name': 'Concat_test_1',
        'operation': 'Concat',
        'input_tensors': [
            {
                'name': 'input0',
                'shape': (64, 64),
                'dtype': 'fp32',
                'data_range': {'min': -10, 'max': 10},
                'format': 'ND',
            },
            {
                'name': 'input1',
                'shape': (64, 64),
                'dtype': 'fp32',
                'data_range': {'min': -10, 'max': 10},
                'format': 'ND',
            },
        ],
        'output_tensors': [{'name': 'output0', 'shape': (128, 64), 'dtype': 'fp32', 'format': 'ND'}],
        'view_shape': (32, 32),
        'tile_shape': (16, 32),
        'params': {'axis': 0},
        'source_case_index': 1,
    },
]
