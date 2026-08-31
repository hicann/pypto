#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Testcase configuration for Fmod vector ST."""

from dataclasses import dataclass

from vector_testcase.vector_test_case import VectorConfig


@dataclass(frozen=True)
class FmodConfig(VectorConfig):
    OPERATION = 'Fmod'


FMOD_TESTS = [
    {
        'case_index': 0,
        'case_name': 'Fmod_test_0',
        'operation': 'Fmod',
        'input_tensors': [
            {
                'name': 'input0',
                'shape': (96, 3318),
                'dtype': 'fp32',
                'data_range': {'min': -5, 'max': 5},
                'format': 'ND',
            },
            {
                'name': 'input1',
                'shape': (96, 3318),
                'dtype': 'fp32',
                'data_range': {'min': -0.001, 'max': 1},
                'format': 'ND',
            },
        ],
        'output_tensors': [{'name': 'output0', 'shape': (96, 3318), 'dtype': 'fp32', 'format': 'ND'}],
        'view_shape': (16, 256),
        'tile_shape': (4, 128),
        'params': {},
        'source_case_index': 0,
    },
    {
        'case_index': 1,
        'case_name': 'Fmod_test_1',
        'operation': 'Fmod',
        'input_tensors': [
            {
                'name': 'input0',
                'shape': (6, 16, 48, 64),
                'dtype': 'fp32',
                'data_range': {'min': -0.001, 'max': 0.001},
                'format': 'ND',
            },
            {
                'name': 'input1',
                'shape': (6, 16, 48, 64),
                'dtype': 'fp32',
                'data_range': {'min': 0, 'max': 0.001},
                'format': 'ND',
            },
        ],
        'output_tensors': [{'name': 'output0', 'shape': (6, 16, 48, 64), 'dtype': 'fp32', 'format': 'ND'}],
        'view_shape': (2, 8, 16, 16),
        'tile_shape': (2, 8, 8, 16),
        'params': {},
        'source_case_index': 1,
    },
]
