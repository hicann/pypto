#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Testcase configuration for Subs vector ST."""

from dataclasses import dataclass

from vector_testcase.vector_test_case import VectorConfig


@dataclass(frozen=True)
class SubsOnboardConfig(VectorConfig):
    OPERATION = 'Subs'


SUBS_ONBOARD_TESTS = [
    {
        'case_index': 0,
        'case_name': 'Subs_test_1',
        'operation': 'Subs',
        'input_tensors': [
            {'name': 'input0', 'shape': (3072, 1), 'dtype': 'fp32', 'data_range': {'min': -1, 'max': 1}, 'format': 'ND'}
        ],
        'output_tensors': [{'name': 'output0', 'shape': (3072, 1), 'dtype': 'fp32', 'format': 'ND'}],
        'view_shape': (128, 1),
        'tile_shape': (32, 32),
        'params': {'scalar': 1e-05, 'scalar_type': 'fp32'},
        'source_case_index': 0,
    },
    {
        'case_index': 1,
        'case_name': 'Subs_test_2',
        'operation': 'Subs',
        'input_tensors': [
            {
                'name': 'input0',
                'shape': (3072, 384),
                'dtype': 'fp32',
                'data_range': {'min': -1, 'max': 1},
                'format': 'ND',
            }
        ],
        'output_tensors': [{'name': 'output0', 'shape': (3072, 384), 'dtype': 'fp32', 'format': 'ND'}],
        'view_shape': (128, 128),
        'tile_shape': (32, 32),
        'params': {'scalar': 1.0, 'scalar_type': 'fp32'},
        'source_case_index': 1,
    },
]
