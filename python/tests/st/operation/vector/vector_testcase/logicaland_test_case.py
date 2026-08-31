#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Testcase configuration for LogicalAnd vector ST."""

from dataclasses import dataclass

from vector_testcase.vector_test_case import VectorConfig


@dataclass(frozen=True)
class LogicalandConfig(VectorConfig):
    OPERATION = 'LogicalAnd'


LOGICALAND_TESTS = [
    {
        'case_index': 0,
        'case_name': 'LogicalAnd_test_0',
        'operation': 'LogicalAnd',
        'input_tensors': [
            {
                'name': 'input0',
                'shape': (4, 28),
                'dtype': 'fp32',
                'tensor_format': 'ND',
                'need_trans': False,
                'data_range': {'min': 1, 'max': 2},
            },
            {
                'name': 'input1',
                'shape': (4, 28),
                'dtype': 'fp32',
                'tensor_format': 'ND',
                'need_trans': False,
                'data_range': {'min': 1, 'max': 2},
            },
        ],
        'output_tensors': [
            {'name': 'output0', 'shape': (4, 28), 'dtype': 'bool', 'tensor_format': 'ND', 'need_trans': False}
        ],
        'view_shape': (3, 15),
        'tile_shape': (2, 3),
        'params': {},
        'source_case_index': 0,
    }
]
