#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Testcase configuration for Sign vector ST."""

from dataclasses import dataclass

from vector_testcase.vector_test_case import VectorConfig


@dataclass(frozen=True)
class SignConfig(VectorConfig):
    OPERATION = 'Sign'


SIGN_TESTS = [
    {
        'case_index': 3,
        'case_name': 'Sign_test_3',
        'operation': 'Sign',
        'input_tensors': [
            {
                'name': 'input0',
                'shape': (4096, 8),
                'dtype': 'int8',
                'format': 'ND',
                'need_trans': False,
                'data_range': {'min': -128, 'max': 127},
            }
        ],
        'output_tensors': [
            {'name': 'output0', 'shape': (4096, 8), 'dtype': 'int8', 'format': 'ND', 'need_trans': False}
        ],
        'view_shape': (64, 64),
        'tile_shape': (32, 16),
        'params': {},
        'source_case_index': 3,
    },
    {
        'case_index': 18,
        'case_name': 'Sign_test_18',
        'operation': 'Sign',
        'input_tensors': [
            {
                'name': 'input0',
                'shape': (64, 16),
                'dtype': 'fp32',
                'format': 'ND',
                'need_trans': False,
                'data_range': {'min': -100.0, 'max': 100.0},
            }
        ],
        'output_tensors': [
            {'name': 'output0', 'shape': (64, 16), 'dtype': 'fp32', 'format': 'ND', 'need_trans': False}
        ],
        'view_shape': (64, 16),
        'tile_shape': (32, 16),
        'params': {},
        'source_case_index': 18,
    },
]
