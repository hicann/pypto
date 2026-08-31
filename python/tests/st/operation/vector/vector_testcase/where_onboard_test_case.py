#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Testcase configuration for Where vector ST."""

from dataclasses import dataclass

from vector_testcase.vector_test_case import VectorConfig


@dataclass(frozen=True)
class WhereOnboardConfig(VectorConfig):
    OPERATION = 'Where'


WHERE_ONBOARD_TESTS = [
    {
        'case_index': 0,
        'case_name': 'Where_test_0',
        'operation': 'Where',
        'input_tensors': [
            {'name': 'input0', 'shape': (10, 54), 'dtype': 'bool', 'data_range': {'min': 0, 'max': 0}, 'format': 'ND'},
            {
                'name': 'input1',
                'shape': (10, 54),
                'dtype': 'fp32',
                'data_range': {'min': 64, 'max': 64},
                'format': 'ND',
            },
            {
                'name': 'input2',
                'shape': (10, 54),
                'dtype': 'fp32',
                'data_range': {'min': 32, 'max': 32},
                'format': 'ND',
            },
        ],
        'output_tensors': [{'name': 'output0', 'shape': (10, 54), 'dtype': 'fp32', 'format': 'ND'}],
        'view_shape': (42, 14),
        'tile_shape': (7, 560),
        'params': {'flag': 0, 'flag_dtype': 'int32', 'x_scalar': 64, 'y_scalar': 32, 'scalar_dtype': 'fp32'},
        'source_case_index': 0,
    },
    {
        'case_index': 8,
        'case_name': 'Where_test_8',
        'operation': 'Where',
        'input_tensors': [
            {'name': 'input0', 'shape': (2, 140), 'dtype': 'bool', 'data_range': {'min': 0, 'max': 0}, 'format': 'ND'},
            {
                'name': 'input1',
                'shape': (2, 140),
                'dtype': 'fp32',
                'data_range': {'min': 64, 'max': 64},
                'format': 'ND',
            },
            {
                'name': 'input2',
                'shape': (2, 140),
                'dtype': 'fp32',
                'data_range': {'min': 32, 'max': 32},
                'format': 'ND',
            },
        ],
        'output_tensors': [{'name': 'output0', 'shape': (2, 140), 'dtype': 'fp32', 'format': 'ND'}],
        'view_shape': (2, 140),
        'tile_shape': (1, 64),
        'params': {'flag': 1, 'flag_dtype': 'int32', 'x_scalar': 64, 'y_scalar': 32, 'scalar_dtype': 'fp32'},
        'source_case_index': 8,
    },
    {
        'case_index': 15,
        'case_name': 'Where_test_15',
        'operation': 'Where',
        'input_tensors': [
            {'name': 'input0', 'shape': (10, 54), 'dtype': 'bool', 'data_range': {'min': 0, 'max': 0}, 'format': 'ND'},
            {
                'name': 'input1',
                'shape': (10, 54),
                'dtype': 'fp32',
                'data_range': {'min': 64, 'max': 64},
                'format': 'ND',
            },
            {
                'name': 'input2',
                'shape': (10, 54),
                'dtype': 'fp32',
                'data_range': {'min': 32, 'max': 32},
                'format': 'ND',
            },
        ],
        'output_tensors': [{'name': 'output0', 'shape': (10, 54), 'dtype': 'fp32', 'format': 'ND'}],
        'view_shape': (42, 14),
        'tile_shape': (7, 560),
        'params': {'flag': 3, 'flag_dtype': 'int32', 'x_scalar': 64, 'y_scalar': 32, 'scalar_dtype': 'fp32'},
        'source_case_index': 15,
    },
    {
        'case_index': 22,
        'case_name': 'Where_test_22',
        'operation': 'Where',
        'input_tensors': [
            {
                'name': 'input0',
                'shape': (10, 9),
                'dtype': 'uint8',
                'data_range': {'min': 15, 'max': 15},
                'format': 'ND',
            },
            {
                'name': 'input1',
                'shape': (10, 72),
                'dtype': 'fp32',
                'data_range': {'min': 64, 'max': 64},
                'format': 'ND',
            },
            {
                'name': 'input2',
                'shape': (10, 72),
                'dtype': 'fp32',
                'data_range': {'min': 32, 'max': 32},
                'format': 'ND',
            },
        ],
        'output_tensors': [{'name': 'output0', 'shape': (10, 72), 'dtype': 'fp32', 'format': 'ND'}],
        'view_shape': (7, 56),
        'tile_shape': (5, 24),
        'params': {'flag': 0, 'flag_dtype': 'int32', 'x_scalar': 64, 'y_scalar': 32, 'scalar_dtype': 'fp32'},
        'source_case_index': 22,
    },
    {
        'case_index': 31,
        'case_name': 'Where_test_31',
        'operation': 'Where',
        'input_tensors': [
            {
                'name': 'input0',
                'shape': (10, 9),
                'dtype': 'uint8',
                'data_range': {'min': 15, 'max': 15},
                'format': 'ND',
            },
            {
                'name': 'input1',
                'shape': (10, 72),
                'dtype': 'fp32',
                'data_range': {'min': 64, 'max': 64},
                'format': 'ND',
            },
            {
                'name': 'input2',
                'shape': (10, 72),
                'dtype': 'fp32',
                'data_range': {'min': 32, 'max': 32},
                'format': 'ND',
            },
        ],
        'output_tensors': [{'name': 'output0', 'shape': (10, 72), 'dtype': 'fp32', 'format': 'ND'}],
        'view_shape': (7, 56),
        'tile_shape': (5, 24),
        'params': {'flag': 3, 'flag_dtype': 'int32', 'x_scalar': 64, 'y_scalar': 32, 'scalar_dtype': 'fp32'},
        'source_case_index': 31,
    },
]
