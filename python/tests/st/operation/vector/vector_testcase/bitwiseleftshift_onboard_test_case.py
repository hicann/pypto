#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Testcase configuration for BitwiseLeftShift vector ST."""

from dataclasses import dataclass

from vector_testcase.vector_test_case import VectorConfig


@dataclass(frozen=True)
class BitwiseleftshiftOnboardConfig(VectorConfig):
    OPERATION = 'BitwiseLeftShift'


BITWISELEFTSHIFT_ONBOARD_TESTS = [
    {
        'case_index': 0,
        'case_name': 'BitwiseLeftShift_test_0',
        'operation': 'BitwiseLeftShift',
        'input_tensors': [
            {
                'name': 'input0',
                'shape': (32, 32),
                'dtype': 'int16',
                'format': 'ND',
                'need_trans': False,
                'data_range': {'min': -10, 'max': 10},
            },
            {
                'name': 'input1',
                'shape': (32, 32),
                'dtype': 'int16',
                'format': 'ND',
                'need_trans': False,
                'data_range': {'min': 0, 'max': 7},
            },
        ],
        'output_tensors': [
            {'name': 'output0', 'shape': (32, 32), 'dtype': 'int16', 'format': 'ND', 'need_trans': False}
        ],
        'view_shape': (10, 10),
        'tile_shape': (3, 3),
        'params': {},
        'source_case_index': 0,
    }
]
