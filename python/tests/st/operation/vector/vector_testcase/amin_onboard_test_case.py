#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Testcase configuration for Amin vector ST."""

from dataclasses import dataclass

from vector_testcase.vector_test_case import VectorConfig


@dataclass(frozen=True)
class AminOnboardConfig(VectorConfig):
    OPERATION = 'Amin'


AMIN_ONBOARD_TESTS = [
    {
        'case_index': 0,
        'case_name': 'RowMinSingle_test_00',
        'operation': 'Amin',
        'input_tensors': [
            {
                'name': 'input0',
                'shape': (64, 64),
                'dtype': 'fp32',
                'format': 'ND',
                'need_trans': False,
                'data_range': {'min': -100, 'max': 100},
            }
        ],
        'output_tensors': [{'name': 'output0', 'shape': (64, 1), 'dtype': 'fp32', 'format': 'ND', 'need_trans': False}],
        'view_shape': (128, 128),
        'tile_shape': (32, 32),
        'params': {'dims': [1], 'keepDim': True, 'skip': None},
        'source_case_index': 0,
    },
    {
        'case_index': 1,
        'case_name': 'RowMinSingle_test_01',
        'operation': 'Amin',
        'input_tensors': [
            {
                'name': 'input0',
                'shape': (64, 63),
                'dtype': 'fp32',
                'format': 'ND',
                'need_trans': False,
                'data_range': {'min': -100, 'max': 100},
            }
        ],
        'output_tensors': [{'name': 'output0', 'shape': (64, 1), 'dtype': 'fp32', 'format': 'ND', 'need_trans': False}],
        'view_shape': (64, 63),
        'tile_shape': (32, 32),
        'params': {'dims': [1], 'keepDim': True, 'skip': None},
        'source_case_index': 1,
    },
    {
        'case_index': 33,
        'case_name': 'RowMin_int32_4D_dim1_32align_keepdim',
        'operation': 'Amin',
        'input_tensors': [
            {
                'name': 'input0',
                'shape': (5, 127, 9, 15),
                'dtype': 'int32',
                'format': 'ND',
                'need_trans': False,
                'data_range': {'min': -1000, 'max': 1000},
            }
        ],
        'output_tensors': [
            {'name': 'output0', 'shape': (5, 1, 9, 15), 'dtype': 'int32', 'format': 'ND', 'need_trans': False}
        ],
        'view_shape': (5, 127, 9, 15),
        'tile_shape': (1, 32, 9, 16),
        'params': {'dims': [1], 'keepDim': True, 'skip': None},
        'source_case_index': 32,
    },
]
