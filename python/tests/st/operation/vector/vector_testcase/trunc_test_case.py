#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Testcase configuration for Trunc vector ST."""

from dataclasses import dataclass

from vector_testcase.vector_test_case import VectorConfig


@dataclass(frozen=True)
class TruncConfig(VectorConfig):
    OPERATION = 'Trunc'


TRUNC_TESTS = [
    {
        'case_index': 0,
        'case_name': 'Trunc_test_0',
        'operation': 'Trunc',
        'input_tensors': [
            {'name': 'input0', 'shape': (16, 32), 'dtype': 'fp16', 'data_range': {'min': 0, 'max': 1}, 'format': 'ND'}
        ],
        'output_tensors': [{'name': 'output0', 'shape': (16, 32), 'dtype': 'fp16', 'format': 'ND'}],
        'view_shape': (16, 32),
        'tile_shape': (16, 32),
        'params': {},
        'source_case_index': 0,
    },
    {
        'case_index': 1,
        'case_name': 'Trunc_test_1',
        'operation': 'Trunc',
        'input_tensors': [
            {'name': 'input0', 'shape': (16, 32), 'dtype': 'fp32', 'data_range': {'min': 0, 'max': 1}, 'format': 'ND'}
        ],
        'output_tensors': [{'name': 'output0', 'shape': (16, 32), 'dtype': 'fp32', 'format': 'ND'}],
        'view_shape': (16, 32),
        'tile_shape': (16, 32),
        'params': {},
        'source_case_index': 1,
    },
]
