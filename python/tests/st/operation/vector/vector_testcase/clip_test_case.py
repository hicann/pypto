#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Testcase configuration for Clip vector ST."""

from dataclasses import dataclass

from vector_testcase.vector_test_case import VectorConfig


@dataclass(frozen=True)
class ClipConfig(VectorConfig):
    OPERATION = 'Clip'


CLIP_TESTS = [
    {
        'case_index': 29,
        'case_name': 'Clip_test_30',
        'operation': 'Clip',
        'input_tensors': [
            {
                'name': 'input0',
                'shape': (128, 128, 128),
                'dtype': 'fp32',
                'tensor_format': 'ND',
                'need_trans': False,
                'data_range': {'min': 0, 'max': 1},
            },
            {
                'name': 'input1',
                'shape': (128, 1, 128),
                'dtype': 'fp32',
                'tensor_format': 'ND',
                'need_trans': False,
                'data_range': {'min': 0, 'max': 1},
            },
            {
                'name': 'input2',
                'shape': (128, 128, 1),
                'dtype': 'fp32',
                'tensor_format': 'ND',
                'need_trans': False,
                'data_range': {'min': 0, 'max': 1},
            },
        ],
        'output_tensors': [
            {'name': 'output0', 'shape': (128, 128, 128), 'dtype': 'fp32', 'tensor_format': 'ND', 'need_trans': False}
        ],
        'view_shape': (32, 32, 32),
        'tile_shape': (16, 16, 16),
        'params': {'min_value': '', 'min_dtype': '', 'max_value': '', 'max_dtype': '', 'test_type': 3},
        'source_case_index': 29,
    }
]
