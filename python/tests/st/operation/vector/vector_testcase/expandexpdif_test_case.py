#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Testcase configuration for ExpandExpDif vector ST."""

from dataclasses import dataclass

from vector_testcase.vector_test_case import VectorConfig


@dataclass(frozen=True)
class ExpandexpdifConfig(VectorConfig):
    OPERATION = 'ExpandExpDif'


EXPANDEXPDIF_TESTS = [
    {
        'case_index': 12,
        'case_name': 'ExpandExpDif_test_13',
        'operation': 'ExpandExpDif',
        'input_tensors': [
            {
                'name': 'input0',
                'shape': (16, 128),
                'dtype': 'fp32',
                'format': 'ND',
                'need_trans': False,
                'data_range': {'min': -10, 'max': 10},
            },
            {
                'name': 'input1',
                'shape': (1, 128),
                'dtype': 'fp32',
                'format': 'ND',
                'need_trans': False,
                'data_range': {'min': -10, 'max': 10},
            },
        ],
        'output_tensors': [
            {'name': 'output0', 'shape': (16, 128), 'dtype': 'fp32', 'format': 'ND', 'need_trans': False}
        ],
        'view_shape': (16, 128),
        'tile_shape': (16, 32),
        'params': {},
        'source_case_index': 12,
    },
    {
        'case_index': 13,
        'case_name': 'ExpandExpDif_test_14',
        'operation': 'ExpandExpDif',
        'input_tensors': [
            {
                'name': 'input0',
                'shape': (16, 128),
                'dtype': 'fp32',
                'format': 'ND',
                'need_trans': False,
                'data_range': {'min': -10, 'max': 10},
            },
            {
                'name': 'input1',
                'shape': (16, 1),
                'dtype': 'fp32',
                'format': 'ND',
                'need_trans': False,
                'data_range': {'min': -10, 'max': 10},
            },
        ],
        'output_tensors': [
            {'name': 'output0', 'shape': (16, 128), 'dtype': 'fp32', 'format': 'ND', 'need_trans': False}
        ],
        'view_shape': (16, 128),
        'tile_shape': (16, 32),
        'params': {},
        'source_case_index': 13,
    },
]
