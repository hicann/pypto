#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Testcase configuration for TopK vector ST."""

from dataclasses import dataclass

from vector_testcase.vector_test_case import VectorConfig


@dataclass(frozen=True)
class TopkConfig(VectorConfig):
    OPERATION = 'TopK'


TOPK_TESTS = [
    {
        'case_index': 0,
        'case_name': 'Topk_16_128_tail_10',
        'operation': 'TopK',
        'input_tensors': [
            {
                'name': 'input0',
                'shape': (16, 128),
                'dtype': 'fp32',
                'data_range': {'min': -100, 'max': 100},
                'format': 'ND',
            }
        ],
        'output_tensors': [
            {'name': 'output0', 'shape': (16, 10), 'dtype': 'fp32', 'format': 'ND'},
            {'name': 'output1', 'shape': (16, 10), 'dtype': 'int32', 'format': 'ND'},
        ],
        'view_shape': (64, 128),
        'tile_shape': (32, 128),
        'params': {'dims': [-1], 'count': [10], 'islargest': [False]},
        'source_case_index': 0,
    },
    {
        'case_index': 4,
        'case_name': 'Topk_8_8_128_tail_10',
        'operation': 'TopK',
        'input_tensors': [
            {
                'name': 'input0',
                'shape': (8, 8, 128),
                'dtype': 'fp32',
                'data_range': {'min': -100, 'max': 100},
                'format': 'ND',
            }
        ],
        'output_tensors': [
            {'name': 'output0', 'shape': (8, 8, 10), 'dtype': 'fp32', 'format': 'ND'},
            {'name': 'output1', 'shape': (8, 8, 10), 'dtype': 'int32', 'format': 'ND'},
        ],
        'view_shape': (2, 2, 128),
        'tile_shape': (2, 2, 128),
        'params': {'dims': [-1], 'count': [10], 'islargest': [False]},
        'source_case_index': 4,
    },
    {
        'case_index': 6,
        'case_name': 'Topk_8_8_128_tail_10',
        'operation': 'TopK',
        'input_tensors': [
            {
                'name': 'input0',
                'shape': (2, 8, 8, 128),
                'dtype': 'fp32',
                'data_range': {'min': -100, 'max': 100},
                'format': 'ND',
            }
        ],
        'output_tensors': [
            {'name': 'output0', 'shape': (2, 8, 8, 10), 'dtype': 'fp32', 'format': 'ND'},
            {'name': 'output1', 'shape': (2, 8, 8, 10), 'dtype': 'int32', 'format': 'ND'},
        ],
        'view_shape': (2, 2, 2, 128),
        'tile_shape': (2, 2, 2, 128),
        'params': {'dims': [-1], 'count': [10], 'islargest': [False]},
        'source_case_index': 6,
    },
]
