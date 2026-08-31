#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Testcase configuration for Prod vector ST."""

from dataclasses import dataclass

from vector_testcase.vector_test_case import VectorConfig


@dataclass(frozen=True)
class ProdOnboardConfig(VectorConfig):
    OPERATION = 'Prod'


PROD_ONBOARD_TESTS = [
    {
        'case_index': 1,
        'case_name': 'RowProd_fp32_3D_dim0_non32_nokeepdim',
        'operation': 'Prod',
        'input_tensors': [
            {
                'name': 'input0',
                'shape': (32, 32, 16),
                'dtype': 'fp32',
                'format': 'ND',
                'need_trans': False,
                'data_range': {'min': -10.0, 'max': 10.0},
            }
        ],
        'output_tensors': [
            {'name': 'output0', 'shape': (32, 16), 'dtype': 'fp32', 'format': 'ND', 'need_trans': False}
        ],
        'view_shape': (32, 16, 32),
        'tile_shape': (32, 16, 32),
        'params': {'dims': [0], 'keepDim': False},
        'source_case_index': 1,
    },
    {
        'case_index': 9,
        'case_name': 'RowProd_fp32_2D_tail_non32_keepdim',
        'operation': 'Prod',
        'input_tensors': [
            {
                'name': 'input0',
                'shape': (40, 400),
                'dtype': 'fp32',
                'format': 'ND',
                'need_trans': False,
                'data_range': {'min': -10.0, 'max': 10.0},
            }
        ],
        'output_tensors': [{'name': 'output0', 'shape': (40, 1), 'dtype': 'fp32', 'format': 'ND', 'need_trans': False}],
        'view_shape': (20, 800),
        'tile_shape': (20, 800),
        'params': {'dims': [-1], 'keepDim': True},
        'source_case_index': 9,
    },
    {
        'case_index': 30,
        'case_name': 'RowProd_int16_64_256_safe_keepdim_true',
        'operation': 'Prod',
        'input_tensors': [
            {
                'name': 'input0',
                'shape': (64, 256),
                'dtype': 'int16',
                'format': 'ND',
                'need_trans': False,
                'data_range': {'min': 5, 'max': 5},
            }
        ],
        'output_tensors': [
            {'name': 'output0', 'shape': (64, 1), 'dtype': 'int16', 'format': 'ND', 'need_trans': False}
        ],
        'view_shape': (64, 256),
        'tile_shape': (64, 256),
        'params': {'dims': [-1], 'keepDim': True},
        'source_case_index': 30,
    },
]
