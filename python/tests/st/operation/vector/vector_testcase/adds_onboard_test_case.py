#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Testcase configuration for Adds vector ST."""

from dataclasses import dataclass

from vector_testcase.vector_test_case import VectorConfig


@dataclass(frozen=True)
class AddsOnboardConfig(VectorConfig):
    OPERATION = 'Adds'


ADDS_ONBOARD_TESTS = [
    {
        'case_index': 20,
        'case_name': 'Adds_test_21',
        'operation': 'Adds',
        'input_tensors': [
            {
                'name': 'input0',
                'shape': (16, 384),
                'dtype': 'fp32',
                'data_range': {'min': -10, 'max': 10},
                'format': 'ND',
            }
        ],
        'output_tensors': [{'name': 'output0', 'shape': (16, 384), 'dtype': 'fp32', 'format': 'ND'}],
        'view_shape': (5, 126),
        'tile_shape': (351, 8),
        'params': {'scalar': 1.0, 'scalar_type': 'fp32'},
        'source_case_index': 20,
    },
    {
        'case_index': 21,
        'case_name': 'Adds_test_22',
        'operation': 'Adds',
        'input_tensors': [
            {
                'name': 'input0',
                'shape': (16, 127, 128, 32),
                'dtype': 'fp32',
                'data_range': {'min': -10, 'max': 10},
                'format': 'ND',
            }
        ],
        'output_tensors': [{'name': 'output0', 'shape': (16, 127, 128, 32), 'dtype': 'fp32', 'format': 'ND'}],
        'view_shape': (2, 29, 37, 1),
        'tile_shape': (13, 149, 1, 8),
        'params': {'scalar': 0.0, 'scalar_type': 'fp32'},
        'source_case_index': 21,
    },
    {
        'case_index': 22,
        'case_name': 'Adds_test_23',
        'operation': 'Adds',
        'input_tensors': [
            {'name': 'input0', 'shape': (1, 2048), 'dtype': 'fp32', 'data_range': {'min': -1, 'max': 1}, 'format': 'ND'}
        ],
        'output_tensors': [{'name': 'output0', 'shape': (1, 2048), 'dtype': 'fp32', 'format': 'ND'}],
        'view_shape': (1, 2047),
        'tile_shape': (1, 2048),
        'params': {'scalar': 1e-05, 'scalar_type': 'fp32'},
        'source_case_index': 22,
    },
]
