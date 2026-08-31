#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Testcase configuration for Erfc vector ST."""

from dataclasses import dataclass

from vector_testcase.vector_test_case import VectorConfig


@dataclass(frozen=True)
class ErfcOnboardConfig(VectorConfig):
    OPERATION = 'Erfc'


ERFC_ONBOARD_TESTS = [
    {
        'case_index': 0,
        'case_name': 'Erfc_test_0',
        'operation': 'Erfc',
        'input_tensors': [
            {
                'name': 'input0',
                'shape': (3072, 1),
                'dtype': 'fp16',
                'format': 'ND',
                'need_trans': False,
                'data_range': {'min': 0, 'max': 1},
            }
        ],
        'output_tensors': [
            {'name': 'output0', 'shape': (3072, 1), 'dtype': 'fp16', 'format': 'ND', 'need_trans': False}
        ],
        'view_shape': (128, 1),
        'tile_shape': (32, 1),
        'params': {},
        'source_case_index': 0,
    },
    {
        'case_index': 1,
        'case_name': 'Erfc_test_1',
        'operation': 'Erfc',
        'input_tensors': [
            {
                'name': 'input0',
                'shape': (3072, 1),
                'dtype': 'fp32',
                'format': 'ND',
                'need_trans': False,
                'data_range': {'min': 0, 'max': 1},
            }
        ],
        'output_tensors': [
            {'name': 'output0', 'shape': (3072, 1), 'dtype': 'fp32', 'format': 'ND', 'need_trans': False}
        ],
        'view_shape': (128, 1),
        'tile_shape': (16, 1),
        'params': {},
        'source_case_index': 1,
    },
    {
        'case_index': 2,
        'case_name': 'Erfc_test_2',
        'operation': 'Erfc',
        'input_tensors': [
            {
                'name': 'input0',
                'shape': (1001, 2),
                'dtype': 'fp16',
                'format': 'ND',
                'need_trans': False,
                'data_range': {'min': -1, 'max': 1},
            }
        ],
        'output_tensors': [
            {'name': 'output0', 'shape': (1001, 2), 'dtype': 'fp16', 'format': 'ND', 'need_trans': False}
        ],
        'view_shape': (100, 2),
        'tile_shape': (33, 1),
        'params': {},
        'source_case_index': 2,
    },
]
