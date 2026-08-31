#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Testcase configuration for RemainderS vector ST."""

from dataclasses import dataclass

from vector_testcase.vector_test_case import VectorConfig


@dataclass(frozen=True)
class RemaindersConfig(VectorConfig):
    OPERATION = 'RemainderS'


REMAINDERS_TESTS = [
    {
        'case_index': 3,
        'case_name': 'RemainderS_2D_align_fp16_int32',
        'operation': 'RemainderS',
        'input_tensors': [
            {
                'name': 'input0',
                'shape': (32, 56),
                'dtype': 'fp16',
                'format': 'ND',
                'need_trans': False,
                'data_range': {'min': -100, 'max': 100},
            }
        ],
        'output_tensors': [
            {'name': 'output0', 'shape': (32, 56), 'dtype': 'fp16', 'format': 'ND', 'need_trans': False}
        ],
        'view_shape': (16, 16),
        'tile_shape': (8, 16),
        'params': {'scalar': 3.0, 'scalar_type': 'int32'},
        'source_case_index': 3,
    }
]
