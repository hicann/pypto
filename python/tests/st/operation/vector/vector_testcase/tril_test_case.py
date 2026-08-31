#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Testcase configuration for TriL vector ST."""

from dataclasses import dataclass

from vector_testcase.vector_test_case import VectorConfig


@dataclass(frozen=True)
class TrilConfig(VectorConfig):
    OPERATION = 'TriL'


TRIL_TESTS = [
    {
        'case_index': 1,
        'case_name': 'TriL_test_1',
        'operation': 'TriL',
        'input_tensors': [
            {
                'name': 'input0',
                'shape': (8, 9),
                'dtype': 'int32',
                'format': 'ND',
                'need_trans': False,
                'data_range': {'min': -10, 'max': 10},
            }
        ],
        'output_tensors': [{'name': 'output0', 'shape': (8, 9), 'dtype': 'int32', 'format': 'ND', 'need_trans': False}],
        'view_shape': (4, 16),
        'tile_shape': (4, 16),
        'params': {'diagonal': 1},
        'source_case_index': 1,
    }
]
