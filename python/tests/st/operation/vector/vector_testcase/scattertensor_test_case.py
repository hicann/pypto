#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Testcase configuration for ScatterTensor vector ST."""

from dataclasses import dataclass

from vector_testcase.vector_test_case import VectorConfig


@dataclass(frozen=True)
class ScattertensorConfig(VectorConfig):
    OPERATION = 'ScatterTensor'


SCATTERTENSOR_TESTS = [
    {
        'case_index': 8,
        'case_name': 'ScatterNonInplace_test_8',
        'operation': 'ScatterTensor',
        'input_tensors': [
            {
                'name': 'input0',
                'shape': (3, 10),
                'dtype': 'fp16',
                'format': 'ND',
                'need_trans': False,
                'data_range': {'min': -1, 'max': 1},
            },
            {
                'name': 'input1',
                'shape': (1, 9),
                'dtype': 'int64',
                'format': 'ND',
                'need_trans': False,
                'data_range': {'min': 0, 'max': 3},
            },
            {
                'name': 'input2',
                'shape': (2, 9),
                'dtype': 'fp16',
                'format': 'ND',
                'need_trans': False,
                'data_range': {'min': -10, 'max': 10},
            },
        ],
        'output_tensors': [{'name': 'output0', 'shape': (3, 10), 'dtype': 'fp16', 'format': 'ND', 'need_trans': False}],
        'view_shape': (3, 8),
        'tile_shape': (3, 16),
        'params': {'axis': 0, 'reduce': 'add'},
        'source_case_index': 8,
    },
    {
        'case_index': 16,
        'case_name': 'ScatterNonInplace_test_16',
        'operation': 'ScatterTensor',
        'input_tensors': [
            {
                'name': 'input0',
                'shape': (1628, 170),
                'dtype': 'fp32',
                'format': 'ND',
                'need_trans': False,
                'data_range': {'min': -10, 'max': 10},
            },
            {
                'name': 'input1',
                'shape': (1471, 94),
                'dtype': 'int32',
                'format': 'ND',
                'need_trans': False,
                'data_range': {'min': 0, 'max': 170},
            },
            {
                'name': 'input2',
                'shape': (1488, 116),
                'dtype': 'fp32',
                'format': 'ND',
                'need_trans': False,
                'data_range': {'min': -10, 'max': 10},
            },
        ],
        'output_tensors': [
            {'name': 'output0', 'shape': (1628, 170), 'dtype': 'fp32', 'format': 'ND', 'need_trans': False}
        ],
        'view_shape': (294, 170),
        'tile_shape': (45, 176),
        'params': {'axis': 1, 'reduce': ''},
        'source_case_index': 16,
    },
]
