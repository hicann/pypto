#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Testcase configuration for GatherMask vector ST."""

from dataclasses import dataclass

from vector_testcase.vector_test_case import VectorConfig


@dataclass(frozen=True)
class GathermaskConfig(VectorConfig):
    OPERATION = 'GatherMask'


GATHERMASK_TESTS = [
    {
        'case_index': 19,
        'case_name': 'GatherMask_test_19',
        'operation': 'GatherMask',
        'input_tensors': [
            {
                'name': 'input0',
                'shape': (4, 8, 8, 780),
                'dtype': 'uint32',
                'format': 'ND',
                'need_trans': False,
                'data_range': {'min': -1, 'max': 1},
            }
        ],
        'output_tensors': [
            {'name': 'output0', 'shape': (4, 8, 8, 195), 'dtype': 'uint32', 'format': 'ND', 'need_trans': False}
        ],
        'view_shape': (3, 5, 5, 780),
        'tile_shape': (2, 3, 5, 136),
        'params': {'patternMode': 3},
        'source_case_index': 19,
    },
    {
        'case_index': 18,
        'case_name': 'GatherMask_test_18',
        'operation': 'GatherMask',
        'input_tensors': [
            {
                'name': 'input0',
                'shape': (8, 3, 112),
                'dtype': 'uint32',
                'format': 'ND',
                'need_trans': False,
                'data_range': {'min': 50, 'max': 100},
            }
        ],
        'output_tensors': [
            {'name': 'output0', 'shape': (8, 3, 56), 'dtype': 'uint32', 'format': 'ND', 'need_trans': False}
        ],
        'view_shape': (2, 1, 112),
        'tile_shape': (1, 1, 8),
        'params': {'patternMode': 2},
        'source_case_index': 18,
    },
]
