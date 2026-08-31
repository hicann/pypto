#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Testcase configuration for Mul vector ST."""

from dataclasses import dataclass

from vector_testcase.vector_test_case import VectorConfig


@dataclass(frozen=True)
class MulConfig(VectorConfig):
    OPERATION = 'Mul'


MUL_TESTS = [
    {
        'case_index': 21,
        'case_name': 'Mul_test_22',
        'operation': 'Mul',
        'input_tensors': [
            {
                'name': 'input0',
                'shape': (16, 1536),
                'dtype': 'fp32',
                'data_range': {'min': -100, 'max': 100},
                'format': 'ND',
            },
            {
                'name': 'input1',
                'shape': (16, 1536),
                'dtype': 'fp32',
                'data_range': {'min': -1, 'max': 1},
                'format': 'ND',
            },
        ],
        'output_tensors': [{'name': 'output0', 'shape': (16, 1536), 'dtype': 'fp32', 'format': 'ND'}],
        'view_shape': (3, 437),
        'tile_shape': (7, 288),
        'params': {},
        'source_case_index': 21,
    },
    {
        'case_index': 22,
        'case_name': 'Mul_test_23',
        'operation': 'Mul',
        'input_tensors': [
            {
                'name': 'input0',
                'shape': (16, 1536),
                'dtype': 'fp32',
                'data_range': {'min': -10, 'max': 10},
                'format': 'ND',
            },
            {
                'name': 'input1',
                'shape': (1, 1536),
                'dtype': 'fp32',
                'data_range': {'min': -100, 'max': 100},
                'format': 'ND',
            },
        ],
        'output_tensors': [{'name': 'output0', 'shape': (16, 1536), 'dtype': 'fp32', 'format': 'ND'}],
        'view_shape': (1, 625),
        'tile_shape': (38, 192),
        'params': {},
        'source_case_index': 22,
    },
    {
        'case_index': 25,
        'case_name': 'Mul_test_26',
        'operation': 'Mul',
        'input_tensors': [
            {
                'name': 'input0',
                'shape': (48, 5, 128, 64),
                'dtype': 'fp32',
                'data_range': {'min': -10, 'max': 10},
                'format': 'ND',
            },
            {
                'name': 'input1',
                'shape': (48, 1, 128, 64),
                'dtype': 'fp32',
                'data_range': {'min': -1, 'max': 1},
                'format': 'ND',
            },
        ],
        'output_tensors': [{'name': 'output0', 'shape': (48, 5, 128, 64), 'dtype': 'fp32', 'format': 'ND'}],
        'view_shape': (10, 2, 57, 9),
        'tile_shape': (30, 10, 20, 8),
        'params': {},
        'source_case_index': 25,
    },
    {
        'case_index': 26,
        'case_name': 'Mul_test_27',
        'operation': 'Mul',
        'input_tensors': [
            {
                'name': 'input0',
                'shape': (48, 5, 128, 64),
                'dtype': 'fp32',
                'data_range': {'min': -10, 'max': 10},
                'format': 'ND',
            },
            {
                'name': 'input1',
                'shape': (1, 5, 128, 64),
                'dtype': 'fp32',
                'data_range': {'min': -1, 'max': 1},
                'format': 'ND',
            },
        ],
        'output_tensors': [{'name': 'output0', 'shape': (48, 5, 128, 64), 'dtype': 'fp32', 'format': 'ND'}],
        'view_shape': (10, 2, 57, 9),
        'tile_shape': (30, 10, 20, 8),
        'params': {},
        'source_case_index': 26,
    },
    {
        'case_index': 27,
        'case_name': 'Mul_test_28',
        'operation': 'Mul',
        'input_tensors': [
            {
                'name': 'input0',
                'shape': (48, 1, 128, 64),
                'dtype': 'fp32',
                'data_range': {'min': -10, 'max': 10},
                'format': 'ND',
            },
            {
                'name': 'input1',
                'shape': (48, 5, 128, 64),
                'dtype': 'fp32',
                'data_range': {'min': -1, 'max': 1},
                'format': 'ND',
            },
        ],
        'output_tensors': [{'name': 'output0', 'shape': (48, 5, 128, 64), 'dtype': 'fp32', 'format': 'ND'}],
        'view_shape': (10, 2, 57, 9),
        'tile_shape': (30, 10, 20, 8),
        'params': {},
        'source_case_index': 27,
    },
]
