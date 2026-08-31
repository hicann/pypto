#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Testcase configuration for Permute vector ST."""

from dataclasses import dataclass

from vector_testcase.vector_test_case import VectorConfig


@dataclass(frozen=True)
class PermuteConfig(VectorConfig):
    OPERATION = 'Permute'


PERMUTE_TESTS = [
    {
        'case_index': 0,
        'case_name': 'permute_0_tail_3D_S',
        'operation': 'Permute',
        'input_tensors': [
            {
                'name': 'input0',
                'shape': (2, 4, 8),
                'dtype': 'int16',
                'format': 'ND',
                'need_trans': False,
                'data_range': {'min': 0, 'max': 1},
            }
        ],
        'output_tensors': [
            {'name': 'output0', 'shape': (2, 8, 4), 'dtype': 'int16', 'format': 'ND', 'need_trans': False}
        ],
        'view_shape': (2, 4, 8),
        'tile_shape': (1, 4, 8),
        'params': {'perm': [0, 2, 1]},
        'source_case_index': 0,
    },
    {
        'case_index': 1,
        'case_name': 'permute_9_tail_4D_S',
        'operation': 'Permute',
        'input_tensors': [
            {
                'name': 'input0',
                'shape': (1, 2, 4, 8),
                'dtype': 'fp32',
                'format': 'ND',
                'need_trans': False,
                'data_range': {'min': 0, 'max': 1},
            }
        ],
        'output_tensors': [
            {'name': 'output0', 'shape': (2, 4, 8, 1), 'dtype': 'fp32', 'format': 'ND', 'need_trans': False}
        ],
        'view_shape': (1, 2, 2, 8),
        'tile_shape': (1, 1, 2, 8),
        'params': {'perm': [1, 2, 3, 0]},
        'source_case_index': 1,
    },
    {
        'case_index': 2,
        'case_name': 'permute_25_tail_5D_N',
        'operation': 'Permute',
        'input_tensors': [
            {
                'name': 'input0',
                'shape': (1, 1, 16, 56, 56),
                'dtype': 'fp32',
                'format': 'ND',
                'need_trans': False,
                'data_range': {'min': 0, 'max': 1},
            }
        ],
        'output_tensors': [
            {'name': 'output0', 'shape': (1, 1, 56, 56, 16), 'dtype': 'fp32', 'format': 'ND', 'need_trans': False}
        ],
        'view_shape': (1, 1, 9, 56, 56),
        'tile_shape': (1, 1, 9, 56, 32),
        'params': {'perm': [0, 1, 4, 3, 2]},
        'source_case_index': 2,
    },
    {
        'case_index': 3,
        'case_name': 'permute_107_tail_5D_S',
        'operation': 'Permute',
        'input_tensors': [
            {
                'name': 'input0',
                'shape': (1, 1, 2, 4, 8),
                'dtype': 'fp32',
                'format': 'ND',
                'need_trans': False,
                'data_range': {'min': 0, 'max': 1},
            }
        ],
        'output_tensors': [
            {'name': 'output0', 'shape': (8, 2, 1, 4, 1), 'dtype': 'fp32', 'format': 'ND', 'need_trans': False}
        ],
        'view_shape': (1, 1, 2, 4, 8),
        'tile_shape': (1, 1, 1, 4, 8),
        'params': {'perm': [4, 2, 0, 3, 1]},
        'source_case_index': 3,
    },
    {
        'case_index': 7,
        'case_name': 'permute_118_ntail_3D_S',
        'operation': 'Permute',
        'input_tensors': [
            {
                'name': 'input0',
                'shape': (2, 4, 8),
                'dtype': 'fp32',
                'format': 'ND',
                'need_trans': False,
                'data_range': {'min': 0, 'max': 1},
            }
        ],
        'output_tensors': [
            {'name': 'output0', 'shape': (2, 4, 8), 'dtype': 'fp32', 'format': 'ND', 'need_trans': False}
        ],
        'view_shape': (2, 4, 8),
        'tile_shape': (1, 4, 8),
        'params': {'perm': [0, 1, 2]},
        'source_case_index': 7,
    },
    {
        'case_index': 8,
        'case_name': 'permute_122_ntail_4D_L',
        'operation': 'Permute',
        'input_tensors': [
            {
                'name': 'input0',
                'shape': (16, 32, 64, 64),
                'dtype': 'fp32',
                'format': 'ND',
                'need_trans': False,
                'data_range': {'min': 0, 'max': 1},
            }
        ],
        'output_tensors': [
            {'name': 'output0', 'shape': (32, 16, 64, 64), 'dtype': 'fp32', 'format': 'ND', 'need_trans': False}
        ],
        'view_shape': (16, 32, 37, 42),
        'tile_shape': (16, 32, 1, 32),
        'params': {'perm': [1, 0, 2, 3]},
        'source_case_index': 8,
    },
    {
        'case_index': 9,
        'case_name': 'permute_131_ntail_5D_S',
        'operation': 'Permute',
        'input_tensors': [
            {
                'name': 'input0',
                'shape': (1, 1, 2, 4, 8),
                'dtype': 'fp32',
                'format': 'ND',
                'need_trans': False,
                'data_range': {'min': 0, 'max': 1},
            }
        ],
        'output_tensors': [
            {'name': 'output0', 'shape': (1, 4, 2, 1, 8), 'dtype': 'fp32', 'format': 'ND', 'need_trans': False}
        ],
        'view_shape': (1, 1, 2, 4, 8),
        'tile_shape': (1, 1, 1, 4, 8),
        'params': {'perm': [0, 3, 2, 1, 4]},
        'source_case_index': 9,
    },
    {
        'case_index': 10,
        'case_name': 'permute_148_ntail_5D_L',
        'operation': 'Permute',
        'input_tensors': [
            {
                'name': 'input0',
                'shape': (4, 8, 8, 16, 32),
                'dtype': 'fp32',
                'format': 'ND',
                'need_trans': False,
                'data_range': {'min': 0, 'max': 1},
            }
        ],
        'output_tensors': [
            {'name': 'output0', 'shape': (16, 8, 4, 8, 32), 'dtype': 'fp32', 'format': 'ND', 'need_trans': False}
        ],
        'view_shape': (4, 8, 8, 12, 32),
        'tile_shape': (4, 8, 8, 2, 32),
        'params': {'perm': [3, 2, 0, 1, 4]},
        'source_case_index': 10,
    },
]
