#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Testcase configuration for Dequantize vector ST."""

from dataclasses import dataclass

from vector_testcase.vector_test_case import VectorConfig


@dataclass(frozen=True)
class DequantizeOnboardConfig(VectorConfig):
    OPERATION = 'Dequantize'


DEQUANTIZE_ONBOARD_TESTS = [
    {
        'case_index': 0,
        'case_name': 'Dequantize_symmetric_int8_2d_basic',
        'operation': 'Dequantize',
        'input_tensors': [
            {
                'name': 'input0',
                'shape': (32, 64),
                'dtype': 'int8',
                'format': 'ND',
                'need_trans': False,
                'data_range': {'min': -100, 'max': 100},
            },
            {
                'name': 'input1',
                'shape': (32,),
                'dtype': 'fp32',
                'format': 'ND',
                'need_trans': False,
                'data_range': {'min': 0.001, 'max': 0.01},
            },
        ],
        'output_tensors': [
            {'name': 'output0', 'shape': (32, 64), 'dtype': 'fp32', 'format': 'ND', 'need_trans': False}
        ],
        'view_shape': (32, 64),
        'tile_shape': (32, 64),
        'params': {'otype': 0, 'axis': -1, 'use_zero_points': False, 'param_input_dtype': 1},
        'source_case_index': 0,
    },
    {
        'case_index': 4,
        'case_name': 'Dequantize_asymmetric_int16_2d',
        'operation': 'Dequantize',
        'input_tensors': [
            {
                'name': 'input0',
                'shape': (32, 64),
                'dtype': 'int16',
                'format': 'ND',
                'need_trans': False,
                'data_range': {'min': -20000, 'max': 20000},
            },
            {
                'name': 'input1',
                'shape': (32,),
                'dtype': 'fp32',
                'format': 'ND',
                'need_trans': False,
                'data_range': {'min': 1e-05, 'max': 0.0001},
            },
            {
                'name': 'input2',
                'shape': (32,),
                'dtype': 'fp32',
                'format': 'ND',
                'need_trans': False,
                'data_range': {'min': -100.0, 'max': 100.0},
            },
        ],
        'output_tensors': [
            {'name': 'output0', 'shape': (32, 64), 'dtype': 'fp32', 'format': 'ND', 'need_trans': False}
        ],
        'view_shape': (32, 64),
        'tile_shape': (32, 64),
        'params': {'otype': 0, 'axis': -1, 'use_zero_points': True, 'param_input_dtype': 2},
        'source_case_index': 4,
    },
]
