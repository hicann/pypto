#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Testcase configuration for Sum vector ST."""

from dataclasses import dataclass

from vector_testcase.vector_test_case import VectorConfig


@dataclass(frozen=True)
class SumOnboardConfig(VectorConfig):
    OPERATION = 'Sum'


SUM_ONBOARD_TESTS = [
    {
        'case_index': 61,
        'case_name': 'RowSumSingle_redline_iter_3_fp32_104_15_0_dense',
        'operation': 'Sum',
        'input_tensors': [
            {
                'name': 'input0',
                'shape': (104, 15),
                'dtype': 'fp32',
                'format': 'ND',
                'need_trans': False,
                'data_range': {'min': -1, 'max': 1},
            }
        ],
        'output_tensors': [{'name': 'output0', 'shape': (1, 15), 'dtype': 'fp32', 'format': 'ND', 'need_trans': False}],
        'view_shape': (104, 6),
        'tile_shape': (88, 40),
        'params': {'dims': [0], 'keepDim': True},
        'source_case_index': 61,
    },
    {
        'case_index': 62,
        'case_name': 'RowSumSingle_redline_iter_3_fp32_37_94_74_1_dense',
        'operation': 'Sum',
        'input_tensors': [
            {
                'name': 'input0',
                'shape': (37, 94, 74),
                'dtype': 'fp32',
                'format': 'ND',
                'need_trans': False,
                'data_range': {'min': -10, 'max': 10},
            }
        ],
        'output_tensors': [
            {'name': 'output0', 'shape': (37, 1, 74), 'dtype': 'fp32', 'format': 'ND', 'need_trans': False}
        ],
        'view_shape': (18, 94, 10),
        'tile_shape': (4, 15, 232),
        'params': {'dims': [1], 'keepDim': True},
        'source_case_index': 62,
    },
    {
        'case_index': 75,
        'case_name': 'RowSum_int16_3D_dim0_non32_keepdim',
        'operation': 'Sum',
        'input_tensors': [
            {
                'name': 'input0',
                'shape': (31, 65, 63),
                'dtype': 'int16',
                'format': 'ND',
                'need_trans': False,
                'data_range': {'min': -100, 'max': 100},
            }
        ],
        'output_tensors': [
            {'name': 'output0', 'shape': (1, 65, 63), 'dtype': 'int16', 'format': 'ND', 'need_trans': False}
        ],
        'view_shape': (31, 65, 63),
        'tile_shape': (16, 32, 64),
        'params': {'dims': [0], 'keepDim': True},
        'source_case_index': 75,
    },
]
