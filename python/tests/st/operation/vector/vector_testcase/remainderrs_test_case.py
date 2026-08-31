#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Testcase configuration for RemainderRS vector ST."""

from dataclasses import dataclass

from vector_testcase.vector_test_case import VectorConfig


@dataclass(frozen=True)
class RemainderrsConfig(VectorConfig):
    OPERATION = 'RemainderRS'


REMAINDERRS_TESTS = [
    {
        'case_index': 9,
        'case_name': 'RemainderRS_2D_unalign_int16_int32',
        'operation': 'RemainderRS',
        'input_tensors': [
            {
                'name': 'input0',
                'shape': (28, 44),
                'dtype': 'int16',
                'format': 'ND',
                'need_trans': False,
                'data_range': {'min': 1, 'max': 10},
            }
        ],
        'output_tensors': [
            {'name': 'output0', 'shape': (28, 44), 'dtype': 'int16', 'format': 'ND', 'need_trans': False}
        ],
        'view_shape': (14, 20),
        'tile_shape': (7, 16),
        'params': {'scalar': 34.0, 'scalar_type': 'int32'},
        'source_case_index': 9,
    }
]
