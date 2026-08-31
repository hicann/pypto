#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Testcase configuration for Amax vector ST."""

from dataclasses import dataclass

from vector_testcase.vector_test_case import VectorConfig


@dataclass(frozen=True)
class AmaxOnboardConfig(VectorConfig):
    OPERATION = 'Amax'


AMAX_ONBOARD_TESTS = [
    {
        'case_index': 65,
        'case_name': 'RowMaxSingle_test1',
        'operation': 'Amax',
        'input_tensors': [
            {
                'name': 'input0',
                'shape': (1, 1, 47, 32),
                'dtype': 'fp32',
                'format': 'ND',
                'need_trans': False,
                'data_range': {'min': -1, 'max': 1},
            }
        ],
        'output_tensors': [
            {'name': 'output0', 'shape': (1, 1, 1, 32), 'dtype': 'fp32', 'format': 'ND', 'need_trans': False}
        ],
        'view_shape': (1, 1184, 47, 32),
        'tile_shape': (1, 256, 2, 32),
        'params': {'dims': [2], 'keepDim': True},
        'source_case_index': 65,
    },
    {
        'case_index': 64,
        'case_name': 'RowMaxSingle_redline_iter_3_fp32_37_94_74_1_dense',
        'operation': 'Amax',
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
        'source_case_index': 64,
    },
    {
        'case_index': 63,
        'case_name': 'RowMaxSingle_redline_iter_3_fp32_104_15_0_dense',
        'operation': 'Amax',
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
        'source_case_index': 63,
    },
    {
        'case_index': 74,
        'case_name': 'RowMax_int32_3D_dim1_non32_nokeepdim',
        'operation': 'Amax',
        'input_tensors': [
            {
                'name': 'input0',
                'shape': (31, 99, 63),
                'dtype': 'int32',
                'format': 'ND',
                'need_trans': False,
                'data_range': {'min': -9999, 'max': 9999},
            }
        ],
        'output_tensors': [
            {'name': 'output0', 'shape': (31, 63), 'dtype': 'int32', 'format': 'ND', 'need_trans': False}
        ],
        'view_shape': (31, 99, 63),
        'tile_shape': (8, 32, 64),
        'params': {'dims': [-2], 'keepDim': False},
        'source_case_index': 74,
    },
]
