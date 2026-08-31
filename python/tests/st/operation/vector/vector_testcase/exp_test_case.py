#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Testcase configuration for Exp vector ST."""

from dataclasses import dataclass

from vector_testcase.vector_test_case import VectorConfig


@dataclass(frozen=True)
class ExpConfig(VectorConfig):
    OPERATION = 'Exp'


EXP_TESTS = [
    {
        'case_index': 18,
        'case_name': 'Exp_test_18',
        'operation': 'Exp',
        'input_tensors': [
            {
                'name': 'input0',
                'shape': (16, 1024),
                'dtype': 'fp32',
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
        'case_name': 'Exp_test_19',
        'operation': 'Exp',
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
