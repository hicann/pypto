#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Testcase configuration for Quantize vector ST."""

from dataclasses import dataclass

from vector_testcase.vector_test_case import VectorConfig


@dataclass(frozen=True)
class QuantizeOnboardConfig(VectorConfig):
    OPERATION = 'Quantize'


QUANTIZE_ONBOARD_TESTS = [
    {
        'case_index': 0,
        'case_name': 'quant_2d_sym_axis2_unalign_01',
        'operation': 'Quantize',
        'input_tensors': [
            {
                'name': 'input0',
                'shape': (4, 16),
                'dtype': 'fp32',
                'format': 'ND',
                'need_trans': False,
                'data_range': {'min': -10.0, 'max': 10.0},
            },
            {
                'name': 'input1',
                'shape': (4,),
                'dtype': 'fp32',
                'format': 'ND',
                'need_trans': False,
                'data_range': {'min': 0.01, 'max': 0.15},
            },
        ],
        'output_tensors': [{'name': 'output0', 'shape': (4, 16), 'dtype': 'int8', 'format': 'ND', 'need_trans': False}],
        'view_shape': (4, 16),
        'tile_shape': (4, 16),
        'params': {'dtype': 1, 'axis': -1, 'use_zero_points': False},
        'source_case_index': 0,
    },
    {
        'case_index': 4,
        'case_name': 'quant_2d_sym_axis2_unalign_05',
        'operation': 'Quantize',
        'input_tensors': [
            {
                'name': 'input0',
                'shape': (4, 4),
                'dtype': 'fp32',
                'format': 'ND',
                'need_trans': False,
                'data_range': {'min': -10.0, 'max': 10.0},
            },
            {
                'name': 'input1',
                'shape': (4,),
                'dtype': 'fp32',
                'format': 'ND',
                'need_trans': False,
                'data_range': {'min': 0.0, 'max': 0.5},
            },
        ],
        'output_tensors': [{'name': 'output0', 'shape': (4, 4), 'dtype': 'int8', 'format': 'ND', 'need_trans': False}],
        'view_shape': (4, 4),
        'tile_shape': (4, 4),
        'params': {'dtype': 1, 'axis': -1, 'use_zero_points': False},
        'source_case_index': 4,
    },
]
