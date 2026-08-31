#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Testcase configuration for Div vector ST."""

from dataclasses import dataclass

from vector_testcase.vector_test_case import VectorConfig


@dataclass(frozen=True)
class DivConfig(VectorConfig):
    OPERATION = 'Div'


DIV_TESTS = [
    {
        'case_index': 13,
        'case_name': 'Div_test_14',
        'operation': 'Div',
        'input_tensors': [
            {
                'name': 'input0',
                'shape': (16, 512),
                'dtype': 'fp32',
                'format': 'ND',
                'need_trans': False,
                'data_range': {'min': -1, 'max': 1},
            },
            {
                'name': 'input1',
                'shape': (16, 1),
                'dtype': 'fp32',
                'format': 'ND',
                'need_trans': False,
                'data_range': {'min': -10, 'max': 10},
            },
        ],
        'output_tensors': [
            {'name': 'output0', 'shape': (16, 512), 'dtype': 'fp32', 'format': 'ND', 'need_trans': False}
        ],
        'view_shape': (5, 152),
        'tile_shape': (33, 32),
        'params': {},
        'source_case_index': 13,
    },
    {
        'case_index': 14,
        'case_name': 'Div_test_15',
        'operation': 'Div',
        'input_tensors': [
            {
                'name': 'input0',
                'shape': (2048, 127, 1),
                'dtype': 'fp32',
                'format': 'ND',
                'need_trans': False,
                'data_range': {'min': -1, 'max': 1},
            },
            {
                'name': 'input1',
                'shape': (2048, 127, 1),
                'dtype': 'fp32',
                'format': 'ND',
                'need_trans': False,
                'data_range': {'min': -10, 'max': 10},
            },
        ],
        'output_tensors': [
            {'name': 'output0', 'shape': (2048, 127, 1), 'dtype': 'fp32', 'format': 'ND', 'need_trans': False}
        ],
        'view_shape': (460, 32, 2),
        'tile_shape': (34, 8, 2),
        'params': {},
        'source_case_index': 14,
    },
]
