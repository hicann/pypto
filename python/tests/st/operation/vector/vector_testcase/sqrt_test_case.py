#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Testcase configuration for Sqrt vector ST."""

from dataclasses import dataclass

from vector_testcase.vector_test_case import VectorConfig


@dataclass(frozen=True)
class SqrtConfig(VectorConfig):
    OPERATION = 'Sqrt'


SQRT_TESTS = [
    {
        'case_index': 0,
        'case_name': 'Sqrt_test_1',
        'operation': 'Sqrt',
        'input_tensors': [
            {'name': 'input0', 'shape': (16384, 1), 'dtype': 'fp32', 'data_range': {'min': 0, 'max': 1}, 'format': 'ND'}
        ],
        'output_tensors': [{'name': 'output0', 'shape': (16384, 1), 'dtype': 'fp32', 'format': 'ND'}],
        'view_shape': (2352, 2),
        'tile_shape': (121, 144),
        'params': {},
        'source_case_index': 0,
    }
]
