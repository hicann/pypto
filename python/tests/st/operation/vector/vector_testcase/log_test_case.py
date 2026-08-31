#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Testcase configuration for Log vector ST."""

from dataclasses import dataclass

from vector_testcase.vector_test_case import VectorConfig


@dataclass(frozen=True)
class LogConfig(VectorConfig):
    OPERATION = 'Log'


LOG_TESTS = [
    {
        'case_index': 2,
        'case_name': 'Log_test_2',
        'operation': 'Log',
        'input_tensors': [
            {'name': 'input0', 'shape': (3072, 48), 'dtype': 'fp32', 'data_range': {'min': 0, 'max': 1}, 'format': 'ND'}
        ],
        'output_tensors': [{'name': 'output0', 'shape': (3072, 48), 'dtype': 'fp32', 'format': 'ND'}],
        'view_shape': (128, 128),
        'tile_shape': (32, 32),
        'params': {'base': '10'},
        'source_case_index': 2,
    },
    {
        'case_index': 5,
        'case_name': 'Log_test_5',
        'operation': 'Log',
        'input_tensors': [
            {
                'name': 'input0',
                'shape': (3072, 384),
                'dtype': 'fp16',
                'data_range': {'min': 0, 'max': 1},
                'format': 'ND',
            }
        ],
        'output_tensors': [{'name': 'output0', 'shape': (3072, 384), 'dtype': 'fp16', 'format': 'ND'}],
        'view_shape': (128, 128),
        'tile_shape': (32, 32),
        'params': {'base': '10'},
        'source_case_index': 5,
    },
]
