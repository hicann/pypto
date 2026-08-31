#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Testcase configuration for Atanh vector ST."""

from dataclasses import dataclass

from vector_testcase.vector_test_case import VectorConfig


@dataclass(frozen=True)
class AtanhOnboardConfig(VectorConfig):
    OPERATION = 'Atanh'


ATANH_ONBOARD_TESTS = [
    {
        'case_index': 0,
        'case_name': 'Atanh_test_0',
        'operation': 'Atanh',
        'input_tensors': [
            {
                'name': 'input0',
                'shape': (1024,),
                'dtype': 'fp32',
                'format': 'ND',
                'need_trans': False,
                'data_range': {'min': -1, 'max': 1},
            }
        ],
        'output_tensors': [{'name': 'output0', 'shape': (1024,), 'dtype': 'fp32', 'format': 'ND', 'need_trans': False}],
        'view_shape': (257,),
        'tile_shape': (128,),
        'params': {},
        'source_case_index': 0,
    },
    {
        'case_index': 2,
        'case_name': 'Atanh_test_2',
        'operation': 'Atanh',
        'input_tensors': [
            {
                'name': 'input0',
                'shape': (4096, 16),
                'dtype': 'fp16',
                'format': 'ND',
                'need_trans': False,
                'data_range': {'min': -1, 'max': 1},
            }
        ],
        'output_tensors': [
            {'name': 'output0', 'shape': (4096, 16), 'dtype': 'fp16', 'format': 'ND', 'need_trans': False}
        ],
        'view_shape': (512, 4),
        'tile_shape': (128, 6),
        'params': {},
        'source_case_index': 2,
    },
]
