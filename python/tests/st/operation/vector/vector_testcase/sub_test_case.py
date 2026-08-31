#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Testcase configuration for Sub vector ST."""

from dataclasses import dataclass

from vector_testcase.vector_test_case import VectorConfig


@dataclass(frozen=True)
class SubConfig(VectorConfig):
    OPERATION = 'Sub'


SUB_TESTS = [
    {
        'case_index': 2,
        'case_name': 'Sub_test_3',
        'operation': 'Sub',
        'input_tensors': [
            {
                'name': 'input0',
                'shape': (32, 36864),
                'dtype': 'fp32',
                'data_range': {'min': -100, 'max': 100},
                'format': 'ND',
            },
            {
                'name': 'input1',
                'shape': (32, 1),
                'dtype': 'fp32',
                'data_range': {'min': -100, 'max': 100},
                'format': 'ND',
            },
        ],
        'output_tensors': [{'name': 'output0', 'shape': (32, 36864), 'dtype': 'fp32', 'format': 'ND'}],
        'view_shape': (2, 13030),
        'tile_shape': (336, 40),
        'params': {},
        'source_case_index': 2,
    },
    {
        'case_index': 3,
        'case_name': 'Sub_test_4',
        'operation': 'Sub',
        'input_tensors': [
            {
                'name': 'input0',
                'shape': (2048, 1, 1),
                'dtype': 'fp32',
                'data_range': {'min': -10, 'max': 10},
                'format': 'ND',
            },
            {
                'name': 'input1',
                'shape': (2048, 1, 1),
                'dtype': 'fp32',
                'data_range': {'min': -100, 'max': 100},
                'format': 'ND',
            },
        ],
        'output_tensors': [{'name': 'output0', 'shape': (2048, 1, 1), 'dtype': 'fp32', 'format': 'ND'}],
        'view_shape': (550, 2, 2),
        'tile_shape': (9, 7, 232),
        'params': {},
        'source_case_index': 3,
    },
]
