#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Testcase configuration for BitwiseNot vector ST."""

from dataclasses import dataclass

from vector_testcase.vector_test_case import VectorConfig


@dataclass(frozen=True)
class BitwisenotOnboardConfig(VectorConfig):
    OPERATION = 'BitwiseNot'


BITWISENOT_ONBOARD_TESTS = [
    {
        'case_index': 2,
        'case_name': 'BitwiseNot_test_3',
        'operation': 'BitwiseNot',
        'input_tensors': [
            {
                'name': 'input0',
                'shape': (33, 31),
                'dtype': 'bool',
                'format': 'ND',
                'need_trans': False,
                'data_range': {'min': 0, 'max': 1},
            }
        ],
        'output_tensors': [
            {'name': 'output0', 'shape': (33, 31), 'dtype': 'bool', 'format': 'ND', 'need_trans': False}
        ],
        'view_shape': (17, 16),
        'tile_shape': (17, 16),
        'params': {},
        'source_case_index': 2,
    }
]
