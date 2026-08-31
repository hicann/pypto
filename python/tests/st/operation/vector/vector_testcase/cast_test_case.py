#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Testcase configuration for Cast vector ST."""

from dataclasses import dataclass

from vector_testcase.vector_test_case import VectorConfig


@dataclass(frozen=True)
class CastConfig(VectorConfig):
    OPERATION = 'Cast'


CAST_TESTS = [
    {
        'case_index': 10,
        'case_name': 'Cast_test_10',
        'operation': 'Cast',
        'input_tensors': [
            {
                'name': 'input0',
                'shape': (16, 384),
                'dtype': 'int32',
                'data_range': {'min': -1, 'max': 1},
                'format': 'ND',
            }
        ],
        'output_tensors': [{'name': 'output0', 'shape': (16, 384), 'dtype': 'fp32', 'format': 'ND'}],
        'view_shape': (7, 169),
        'tile_shape': (215, 48),
        'params': {'mode': 0},
        'source_case_index': 10,
    },
    {
        'case_index': 11,
        'case_name': 'Cast_test_11',
        'operation': 'Cast',
        'input_tensors': [
            {
                'name': 'input0',
                'shape': (16, 1, 64),
                'dtype': 'bf16',
                'data_range': {'min': -1, 'max': 1},
                'format': 'ND',
            }
        ],
        'output_tensors': [{'name': 'output0', 'shape': (16, 1, 64), 'dtype': 'fp32', 'format': 'ND'}],
        'view_shape': (5, 2, 18),
        'tile_shape': (310, 6, 16),
        'params': {'mode': 0},
        'source_case_index': 11,
    },
    {
        'case_index': 12,
        'case_name': 'Cast_test_12',
        'operation': 'Cast',
        'input_tensors': [
            {
                'name': 'input0',
                'shape': (32, 127, 128, 64),
                'dtype': 'fp16',
                'data_range': {'min': -100, 'max': 100},
                'format': 'ND',
            }
        ],
        'output_tensors': [{'name': 'output0', 'shape': (32, 127, 128, 64), 'dtype': 'fp32', 'format': 'ND'}],
        'view_shape': (2, 42, 44, 29),
        'tile_shape': (1, 28, 8, 112),
        'params': {'mode': 0},
        'source_case_index': 12,
    },
]
