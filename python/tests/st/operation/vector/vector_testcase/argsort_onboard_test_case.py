#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Testcase configuration for ArgSort vector ST."""

from dataclasses import dataclass

from vector_testcase.vector_test_case import VectorConfig


@dataclass(frozen=True)
class ArgsortOnboardConfig(VectorConfig):
    OPERATION = 'ArgSort'


ARGSORT_ONBOARD_TESTS = [
    {
        'case_index': 0,
        'case_name': 'ArgSort_test_0',
        'operation': 'ArgSort',
        'input_tensors': [
            {
                'name': 'input0',
                'shape': (128,),
                'dtype': 'fp32',
                'format': 'ND',
                'need_trans': False,
                'data_range': {'min': -1000, 'max': 1000},
            }
        ],
        'output_tensors': [{'name': 'output0', 'shape': (128,), 'dtype': 'int32', 'format': 'ND', 'need_trans': False}],
        'view_shape': (128,),
        'tile_shape': (32,),
        'params': {'dims': [0], 'descending': [True]},
        'source_case_index': 0,
    },
    {
        'case_index': 2,
        'case_name': 'ArgSort_test_2',
        'operation': 'ArgSort',
        'input_tensors': [
            {
                'name': 'input0',
                'shape': (16, 128),
                'dtype': 'fp32',
                'format': 'ND',
                'need_trans': False,
                'data_range': {'min': -1000, 'max': 1000},
            }
        ],
        'output_tensors': [
            {'name': 'output0', 'shape': (16, 128), 'dtype': 'int32', 'format': 'ND', 'need_trans': False}
        ],
        'view_shape': (8, 128),
        'tile_shape': (8, 32),
        'params': {'dims': [1], 'descending': [True]},
        'source_case_index': 2,
    },
]
