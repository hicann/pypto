#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Testcase configuration for FloorDiv vector ST."""

from dataclasses import dataclass

from vector_testcase.vector_test_case import VectorConfig


@dataclass(frozen=True)
class FloordivConfig(VectorConfig):
    OPERATION = 'FloorDiv'


FLOORDIV_TESTS = [
    {
        'case_index': 0,
        'case_name': 'FloorDiv_test_1',
        'operation': 'FloorDiv',
        'input_tensors': [
            {
                'name': 'input0',
                'shape': (1024, 128),
                'dtype': 'int32',
                'format': 'ND',
                'need_trans': False,
                'data_range': {'min': 1, 'max': 1000},
            },
            {
                'name': 'input1',
                'shape': (1024, 128),
                'dtype': 'int32',
                'format': 'ND',
                'need_trans': False,
                'data_range': {'min': 1, 'max': 1000},
            },
        ],
        'output_tensors': [
            {'name': 'output0', 'shape': (1024, 128), 'dtype': 'int32', 'format': 'ND', 'need_trans': False}
        ],
        'view_shape': (128, 512),
        'tile_shape': (64, 64),
        'params': {},
        'source_case_index': 0,
    }
]
