#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Testcase configuration for Exp2 vector ST."""

from dataclasses import dataclass

from vector_testcase.vector_test_case import VectorConfig


@dataclass(frozen=True)
class Exp2Config(VectorConfig):
    OPERATION = 'Exp2'


EXP2_TESTS = [
    {
        'case_index': 17,
        'case_name': 'Exp2_test_17',
        'operation': 'Exp2',
        'input_tensors': [
            {
                'name': 'input0',
                'shape': (96, 8192),
                'dtype': 'int16',
                'data_range': {'min': 0, 'max': 1},
                'format': 'ND',
            }
        ],
        'output_tensors': [{'name': 'output0', 'shape': (96, 8192), 'dtype': 'fp32', 'format': 'ND'}],
        'view_shape': (128, 128),
        'tile_shape': (32, 32),
        'params': {},
        'source_case_index': 17,
    },
    {
        'case_index': 18,
        'case_name': 'Exp2_test_18',
        'operation': 'Exp2',
        'input_tensors': [
            {
                'name': 'input0',
                'shape': (16, 1024),
                'dtype': 'int32',
                'data_range': {'min': -1, 'max': 1},
                'format': 'ND',
            }
        ],
        'output_tensors': [{'name': 'output0', 'shape': (16, 1024), 'dtype': 'fp32', 'format': 'ND'}],
        'view_shape': (7, 127),
        'tile_shape': (23, 328),
        'params': {},
        'source_case_index': 18,
    },
    {
        'case_index': 19,
        'case_name': 'Exp2_test_19',
        'operation': 'Exp2',
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
]
