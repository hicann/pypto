#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Testcase configuration for CumSum vector ST."""

from dataclasses import dataclass

from vector_testcase.vector_test_case import VectorConfig


@dataclass(frozen=True)
class CumsumOnboardConfig(VectorConfig):
    OPERATION = 'CumSum'


CUMSUM_ONBOARD_TESTS = [
    {
        'case_index': 2,
        'case_name': 'CumSum_test_2',
        'operation': 'CumSum',
        'input_tensors': [
            {
                'name': 'input0',
                'shape': (4, 10),
                'dtype': 'fp32',
                'format': 'ND',
                'need_trans': False,
                'data_range': {'min': -10, 'max': 10},
            }
        ],
        'output_tensors': [{'name': 'output0', 'shape': (4, 10), 'dtype': 'fp32', 'format': 'ND', 'need_trans': False}],
        'view_shape': (2, 2),
        'tile_shape': (2, 1),
        'params': {'axis': 1},
        'source_case_index': 2,
    },
    {
        'case_index': 3,
        'case_name': 'CumSum_test_3',
        'operation': 'CumSum',
        'input_tensors': [
            {
                'name': 'input0',
                'shape': (4, 10),
                'dtype': 'fp32',
                'format': 'ND',
                'need_trans': False,
                'data_range': {'min': -10, 'max': 10},
            }
        ],
        'output_tensors': [{'name': 'output0', 'shape': (4, 10), 'dtype': 'fp32', 'format': 'ND', 'need_trans': False}],
        'view_shape': (2, 2),
        'tile_shape': (2, 2),
        'params': {'axis': 0},
        'source_case_index': 3,
    },
]
