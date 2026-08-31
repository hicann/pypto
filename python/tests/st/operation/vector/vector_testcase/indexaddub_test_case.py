#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Testcase configuration for IndexAddUB vector ST."""

from dataclasses import dataclass

from vector_testcase.vector_test_case import VectorConfig


@dataclass(frozen=True)
class IndexaddubConfig(VectorConfig):
    OPERATION = 'IndexAddUB'


INDEXADDUB_TESTS = [
    {
        'case_index': 20,
        'case_name': 'IndexAddUBNotInplace_test_20',
        'operation': 'IndexAddUB',
        'input_tensors': [
            {
                'name': 'input0',
                'shape': (10, 10),
                'dtype': 'bf16',
                'format': 'ND',
                'need_trans': False,
                'data_range': {'min': -1, 'max': 1},
            },
            {
                'name': 'input1',
                'shape': (10, 5),
                'dtype': 'bf16',
                'format': 'ND',
                'need_trans': False,
                'data_range': {'min': -10, 'max': 10},
            },
            {
                'name': 'input2',
                'shape': (5,),
                'dtype': 'int32',
                'format': 'ND',
                'need_trans': False,
                'data_range': {'min': 0, 'max': 10},
            },
        ],
        'output_tensors': [
            {'name': 'output0', 'shape': (10, 10), 'dtype': 'bf16', 'format': 'ND', 'need_trans': False}
        ],
        'view_shape': (10, 10),
        'tile_shape': (10, 16),
        'params': {'axis': 1, 'alpha': 1.0},
        'source_case_index': 20,
    },
    {
        'case_index': 21,
        'case_name': 'IndexAddUBNotInplace_test_21',
        'operation': 'IndexAddUB',
        'input_tensors': [
            {
                'name': 'input0',
                'shape': (10, 30),
                'dtype': 'bf16',
                'format': 'ND',
                'need_trans': False,
                'data_range': {'min': -10, 'max': 10},
            },
            {
                'name': 'input1',
                'shape': (27, 30),
                'dtype': 'bf16',
                'format': 'ND',
                'need_trans': False,
                'data_range': {'min': -10, 'max': 10},
            },
            {
                'name': 'input2',
                'shape': (27,),
                'dtype': 'int32',
                'format': 'ND',
                'need_trans': False,
                'data_range': {'min': 0, 'max': 10},
            },
        ],
        'output_tensors': [
            {'name': 'output0', 'shape': (10, 30), 'dtype': 'bf16', 'format': 'ND', 'need_trans': False}
        ],
        'view_shape': (27, 10),
        'tile_shape': (10, 16),
        'params': {'axis': 0, 'alpha': 1.2},
        'source_case_index': 21,
    },
]
