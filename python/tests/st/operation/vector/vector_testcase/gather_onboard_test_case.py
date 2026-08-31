#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Testcase configuration for Gather vector ST."""

from dataclasses import dataclass

from vector_testcase.vector_test_case import VectorConfig


@dataclass(frozen=True)
class GatherOnboardConfig(VectorConfig):
    OPERATION = 'Gather'


GATHER_ONBOARD_TESTS = [
    {
        'case_index': 6,
        'case_name': 'Gather_test_6',
        'operation': 'Gather',
        'input_tensors': [
            {
                'name': 'input0',
                'shape': (32, 32, 64),
                'dtype': 'fp32',
                'format': 'ND',
                'need_trans': False,
                'data_range': {'min': -1, 'max': 1},
            },
            {
                'name': 'input1',
                'shape': (64,),
                'dtype': 'int64',
                'format': 'ND',
                'need_trans': False,
                'data_range': {'min': 0, 'max': 31},
            },
        ],
        'output_tensors': [
            {'name': 'output0', 'shape': (32, 32, 64), 'dtype': 'fp32', 'format': 'ND', 'need_trans': False}
        ],
        'view_shape': (16, 32, 32),
        'tile_shape': (8, 16, 8),
        'params': {'axis': 2},
        'source_case_index': 6,
    },
    {
        'case_index': 0,
        'case_name': 'Gather_test_0',
        'operation': 'Gather',
        'input_tensors': [
            {
                'name': 'input0',
                'shape': (32, 128),
                'dtype': 'fp32',
                'format': 'ND',
                'need_trans': False,
                'data_range': {'min': -1, 'max': 1},
            },
            {
                'name': 'input1',
                'shape': (64,),
                'dtype': 'int64',
                'format': 'ND',
                'need_trans': False,
                'data_range': {'min': 0, 'max': 31},
            },
        ],
        'output_tensors': [
            {'name': 'output0', 'shape': (64, 128), 'dtype': 'fp32', 'format': 'ND', 'need_trans': False}
        ],
        'view_shape': (32, 16),
        'tile_shape': (16, 16),
        'params': {'axis': 0},
        'source_case_index': 0,
    },
    {
        'case_index': 22,
        'case_name': 'Gather_test_22',
        'operation': 'Gather',
        'input_tensors': [
            {
                'name': 'input0',
                'shape': (32, 128, 64),
                'dtype': 'fp32',
                'format': 'ND',
                'need_trans': False,
                'data_range': {'min': -1, 'max': 1},
            },
            {
                'name': 'input1',
                'shape': (4, 16),
                'dtype': 'int64',
                'format': 'ND',
                'need_trans': False,
                'data_range': {'min': 0, 'max': 15},
            },
        ],
        'output_tensors': [
            {'name': 'output0', 'shape': (32, 128, 4, 16), 'dtype': 'fp32', 'format': 'ND', 'need_trans': False}
        ],
        'view_shape': (8, 16, 32, 16),
        'tile_shape': (8, 8, 16, 8),
        'params': {'axis': -1},
        'source_case_index': 22,
    },
]
