#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Testcase configuration for Scatter vector ST."""

from dataclasses import dataclass

from vector_testcase.vector_test_case import VectorConfig


@dataclass(frozen=True)
class ScatterOnboardConfig(VectorConfig):
    OPERATION = 'Scatter'


SCATTER_ONBOARD_TESTS = [
    {
        'case_index': 0,
        'case_name': 'ScatterNonInplace_test_0',
        'operation': 'Scatter',
        'input_tensors': [
            {
                'name': 'input0',
                'shape': (128, 512),
                'dtype': 'fp32',
                'format': 'ND',
                'need_trans': False,
                'data_range': {'min': -1, 'max': 1},
            },
            {
                'name': 'input1',
                'shape': (128, 512),
                'dtype': 'int64',
                'format': 'ND',
                'need_trans': False,
                'data_range': {'min': 0, 'max': 128},
            },
        ],
        'output_tensors': [
            {'name': 'output0', 'shape': (128, 512), 'dtype': 'fp32', 'format': 'ND', 'need_trans': False}
        ],
        'view_shape': (128, 128),
        'tile_shape': (128, 64),
        'params': {'axis': 0, 'src': 1.0, 'reduce': ''},
        'source_case_index': 0,
    },
    {
        'case_index': 2,
        'case_name': 'ScatterNonInplace_test_2',
        'operation': 'Scatter',
        'input_tensors': [
            {
                'name': 'input0',
                'shape': (128, 512),
                'dtype': 'fp16',
                'format': 'ND',
                'need_trans': False,
                'data_range': {'min': -1, 'max': 1},
            },
            {
                'name': 'input1',
                'shape': (128, 512),
                'dtype': 'int64',
                'format': 'ND',
                'need_trans': False,
                'data_range': {'min': 0, 'max': 128},
            },
        ],
        'output_tensors': [
            {'name': 'output0', 'shape': (128, 512), 'dtype': 'fp16', 'format': 'ND', 'need_trans': False}
        ],
        'view_shape': (128, 128),
        'tile_shape': (128, 64),
        'params': {'axis': 0, 'src': 0.015005, 'reduce': ''},
        'source_case_index': 2,
    },
]
