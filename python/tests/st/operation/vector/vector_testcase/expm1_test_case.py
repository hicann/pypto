#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Testcase configuration for Expm1 vector ST."""

from dataclasses import dataclass

from vector_testcase.vector_test_case import VectorConfig


@dataclass(frozen=True)
class Expm1Config(VectorConfig):
    OPERATION = 'Expm1'


EXPM1_TESTS = [
    {
        'case_index': 19,
        'case_name': 'Expm1_test_19',
        'operation': 'Expm1',
        'input_tensors': [
            {
                'name': 'input0',
                'shape': (512, 256, 256),
                'dtype': 'fp32',
                'data_range': {'min': -10, 'max': 10},
                'format': 'ND',
            }
        ],
        'output_tensors': [{'name': 'output0', 'shape': (512, 256, 256), 'dtype': 'fp32', 'format': 'ND'}],
        'view_shape': (71, 8, 214),
        'tile_shape': (167, 13, 8),
        'params': {},
        'source_case_index': 19,
    },
    {
        'case_index': 20,
        'case_name': 'Expm1_test_20',
        'operation': 'Expm1',
        'input_tensors': [
            {'name': 'input0', 'shape': (32, 32), 'dtype': 'fp16', 'data_range': {'min': -1, 'max': 1}, 'format': 'ND'}
        ],
        'output_tensors': [{'name': 'output0', 'shape': (32, 32), 'dtype': 'fp16', 'format': 'ND'}],
        'view_shape': (32, 32),
        'tile_shape': (32, 32),
        'params': {},
        'source_case_index': 20,
    },
    {
        'case_index': 21,
        'case_name': 'Expm1_test_21',
        'operation': 'Expm1',
        'input_tensors': [
            {'name': 'input0', 'shape': (32, 32), 'dtype': 'bf16', 'data_range': {'min': -1, 'max': 1}, 'format': 'ND'}
        ],
        'output_tensors': [{'name': 'output0', 'shape': (32, 32), 'dtype': 'bf16', 'format': 'ND'}],
        'view_shape': (32, 32),
        'tile_shape': (32, 32),
        'params': {},
        'source_case_index': 21,
    },
    {
        'case_index': 22,
        'case_name': 'Expm1_test_22',
        'operation': 'Expm1',
        'input_tensors': [
            {
                'name': 'input0',
                'shape': (128, 128),
                'dtype': 'int32',
                'data_range': {'min': 0, 'max': 1},
                'format': 'ND',
            }
        ],
        'output_tensors': [{'name': 'output0', 'shape': (128, 128), 'dtype': 'fp32', 'format': 'ND'}],
        'view_shape': (32, 32),
        'tile_shape': (64, 64),
        'params': {},
        'source_case_index': 22,
    },
    {
        'case_index': 23,
        'case_name': 'Expm1_test_23',
        'operation': 'Expm1',
        'input_tensors': [
            {
                'name': 'input0',
                'shape': (16, 1024),
                'dtype': 'int16',
                'data_range': {'min': -1, 'max': 1},
                'format': 'ND',
            }
        ],
        'output_tensors': [{'name': 'output0', 'shape': (16, 1024), 'dtype': 'fp32', 'format': 'ND'}],
        'view_shape': (7, 127),
        'tile_shape': (23, 328),
        'params': {},
        'source_case_index': 23,
    },
]
