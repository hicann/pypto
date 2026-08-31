#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Testcase configuration for OneHot vector ST."""

from dataclasses import dataclass

from vector_testcase.vector_test_case import VectorConfig


@dataclass(frozen=True)
class OnehotConfig(VectorConfig):
    OPERATION = 'OneHot'


ONEHOT_TESTS = [
    {
        'case_index': 7,
        'case_name': 'OneHot_test_7',
        'operation': 'OneHot',
        'input_tensors': [
            {'name': 'input0', 'shape': (32, 32), 'dtype': 'int32', 'data_range': {'min': 0, 'max': 31}, 'format': 'ND'}
        ],
        'output_tensors': [{'name': 'output0', 'shape': (32, 32, 32), 'dtype': 'int64', 'format': 'ND'}],
        'view_shape': (16, 8, 32),
        'tile_shape': (16, 8, 32),
        'params': {'num_classes': 32},
        'source_case_index': 7,
    },
    {
        'case_index': 14,
        'case_name': 'OneHot_test_14',
        'operation': 'OneHot',
        'input_tensors': [
            {'name': 'input0', 'shape': (120,), 'dtype': 'int32', 'data_range': {'min': 0, 'max': 63}, 'format': 'ND'}
        ],
        'output_tensors': [{'name': 'output0', 'shape': (120, 64), 'dtype': 'int64', 'format': 'ND'}],
        'view_shape': (32, 64),
        'tile_shape': (32, 64),
        'params': {'num_classes': 64},
        'source_case_index': 14,
    },
    {
        'case_index': 16,
        'case_name': 'OneHot_test_16',
        'operation': 'OneHot',
        'input_tensors': [
            {
                'name': 'input0',
                'shape': (128,),
                'dtype': 'int32',
                'data_range': {'min': 451, 'max': 975},
                'format': 'ND',
            }
        ],
        'output_tensors': [{'name': 'output0', 'shape': (128, 1007), 'dtype': 'int64', 'format': 'ND'}],
        'view_shape': (32, 1007),
        'tile_shape': (8, 1007),
        'params': {'num_classes': 1007},
        'source_case_index': 16,
    },
]
