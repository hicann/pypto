#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Testcase configuration for Relu vector ST."""

from dataclasses import dataclass

from vector_testcase.vector_test_case import VectorConfig


@dataclass(frozen=True)
class ReluConfig(VectorConfig):
    OPERATION = 'Relu'


RELU_TESTS = [
    {
        'case_index': 0,
        'case_name': 'ReLU_test_0',
        'operation': 'Relu',
        'input_tensors': [
            {
                'name': 'input0',
                'shape': (16, 32),
                'dtype': 'fp16',
                'format': 'ND',
                'need_trans': False,
                'data_range': {'min': -10, 'max': 10},
            }
        ],
        'output_tensors': [
            {'name': 'output0', 'shape': (16, 32), 'dtype': 'fp16', 'format': 'ND', 'need_trans': False}
        ],
        'view_shape': (16, 32),
        'tile_shape': (16, 32),
        'params': {},
        'source_case_index': 0,
    },
    {
        'case_index': 3,
        'case_name': 'ReLU_test_3',
        'operation': 'Relu',
        'input_tensors': [
            {
                'name': 'input0',
                'shape': (1024, 128, 64),
                'dtype': 'fp32',
                'format': 'ND',
                'need_trans': False,
                'data_range': {'min': -1, 'max': 1},
            }
        ],
        'output_tensors': [
            {'name': 'output0', 'shape': (1024, 128, 64), 'dtype': 'fp32', 'format': 'ND', 'need_trans': False}
        ],
        'view_shape': (64, 64, 64),
        'tile_shape': (16, 16, 32),
        'params': {},
        'source_case_index': 3,
    },
]
