#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Testcase configuration for LogicalNot vector ST."""

from dataclasses import dataclass

from vector_testcase.vector_test_case import VectorConfig


@dataclass(frozen=True)
class LogicalnotConfig(VectorConfig):
    OPERATION = 'LogicalNot'


LOGICALNOT_TESTS = [
    {
        'case_index': 1,
        'case_name': 'LogicalNot_test_1',
        'operation': 'LogicalNot',
        'input_tensors': [
            {'name': 'input0', 'shape': (27, 18), 'dtype': 'fp32', 'data_range': {'min': -3, 'max': 3}, 'format': 'ND'}
        ],
        'output_tensors': [{'name': 'output0', 'shape': (27, 18), 'dtype': 'bool', 'format': 'ND'}],
        'view_shape': (6, 12),
        'tile_shape': (14, 8),
        'params': {},
        'source_case_index': 1,
    },
    {
        'case_index': 4,
        'case_name': 'LogicalNot_test_4',
        'operation': 'LogicalNot',
        'input_tensors': [
            {'name': 'input0', 'shape': (4, 68), 'dtype': 'uint8', 'data_range': {'min': 0, 'max': 3}, 'format': 'ND'}
        ],
        'output_tensors': [{'name': 'output0', 'shape': (4, 68), 'dtype': 'bool', 'format': 'ND'}],
        'view_shape': (12, 6),
        'tile_shape': (256, 128),
        'params': {},
        'source_case_index': 4,
    },
]
