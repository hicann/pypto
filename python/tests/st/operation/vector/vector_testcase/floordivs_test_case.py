#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Testcase configuration for FloorDivs vector ST."""

from dataclasses import dataclass

from vector_testcase.vector_test_case import VectorConfig


@dataclass(frozen=True)
class FloordivsConfig(VectorConfig):
    OPERATION = 'FloorDivs'


FLOORDIVS_TESTS = [
    {
        'case_index': 0,
        'case_name': 'FloorDivs_test_1',
        'operation': 'FloorDivs',
        'input_tensors': [
            {
                'name': 'input0',
                'shape': (64, 128),
                'dtype': 'int32',
                'format': 'ND',
                'need_trans': False,
                'data_range': {'min': -10, 'max': 10},
            }
        ],
        'output_tensors': [
            {'name': 'output0', 'shape': (64, 128), 'dtype': 'int32', 'format': 'ND', 'need_trans': False}
        ],
        'view_shape': (7, 43),
        'tile_shape': (192, 120),
        'params': {'scalar': 1.0, 'scalar_type': 'int32'},
        'source_case_index': 0,
    }
]
