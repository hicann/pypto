#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Testcase configuration for Pows vector ST."""

from dataclasses import dataclass

from vector_testcase.vector_test_case import VectorConfig


@dataclass(frozen=True)
class PowsConfig(VectorConfig):
    OPERATION = 'Pows'


POWS_TESTS = [
    {
        'case_index': 12,
        'case_name': 'Pows_test_12',
        'operation': 'Pows',
        'input_tensors': [
            {
                'name': 'input0',
                'shape': (32, 32),
                'dtype': 'fp32',
                'data_range': {'min': -10, 'max': 10},
                'format': 'ND',
            }
        ],
        'output_tensors': [{'name': 'output0', 'shape': (32, 32), 'dtype': 'fp32', 'format': 'ND'}],
        'view_shape': (32, 32),
        'tile_shape': (32, 32),
        'params': {'scalar': 6.0, 'scalar_type': 'fp32'},
        'source_case_index': 12,
    },
    {
        'case_index': 13,
        'case_name': 'Pows_test_13',
        'operation': 'Pows',
        'input_tensors': [
            {
                'name': 'input0',
                'shape': (32, 32),
                'dtype': 'fp32',
                'data_range': {'min': -10, 'max': 10},
                'format': 'ND',
            }
        ],
        'output_tensors': [{'name': 'output0', 'shape': (32, 32), 'dtype': 'fp32', 'format': 'ND'}],
        'view_shape': (32, 32),
        'tile_shape': (32, 32),
        'params': {'scalar': -5.0, 'scalar_type': 'fp32'},
        'source_case_index': 13,
    },
    {
        'case_index': 14,
        'case_name': 'Pows_test_14',
        'operation': 'Pows',
        'input_tensors': [
            {
                'name': 'input0',
                'shape': (32, 32),
                'dtype': 'fp32',
                'data_range': {'min': -10, 'max': 10},
                'format': 'ND',
            }
        ],
        'output_tensors': [{'name': 'output0', 'shape': (32, 32), 'dtype': 'fp32', 'format': 'ND'}],
        'view_shape': (32, 32),
        'tile_shape': (32, 32),
        'params': {'scalar': 5.7, 'scalar_type': 'fp32'},
        'source_case_index': 14,
    },
    {
        'case_index': 15,
        'case_name': 'Pows_test_15',
        'operation': 'Pows',
        'input_tensors': [
            {
                'name': 'input0',
                'shape': (32, 32),
                'dtype': 'fp32',
                'data_range': {'min': -10, 'max': 10},
                'format': 'ND',
            }
        ],
        'output_tensors': [{'name': 'output0', 'shape': (32, 32), 'dtype': 'fp32', 'format': 'ND'}],
        'view_shape': (32, 32),
        'tile_shape': (32, 32),
        'params': {'scalar': -5.7, 'scalar_type': 'fp32'},
        'source_case_index': 15,
    },
]
