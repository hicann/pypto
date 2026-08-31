#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Testcase configuration for IndexAdd_ vector ST."""

from dataclasses import dataclass

from vector_testcase.vector_test_case import VectorConfig


@dataclass(frozen=True)
class IndexaddOnboardConfig(VectorConfig):
    OPERATION = 'IndexAdd_'


INDEXADD_ONBOARD_TESTS = [
    {
        'case_index': 7,
        'case_name': 'IndexAddInplace_test_7',
        'operation': 'IndexAdd_',
        'input_tensors': [
            {
                'name': 'input0',
                'shape': (10, 10),
                'dtype': 'fp32',
                'format': 'ND',
                'need_trans': False,
                'data_range': {'min': -10, 'max': 10},
            },
            {
                'name': 'input1',
                'shape': (13, 10),
                'dtype': 'fp32',
                'format': 'ND',
                'need_trans': False,
                'data_range': {'min': -10, 'max': 10},
            },
            {
                'name': 'input2',
                'shape': (13,),
                'dtype': 'int32',
                'format': 'ND',
                'need_trans': False,
                'data_range': {'min': 0, 'max': 10},
            },
        ],
        'output_tensors': [
            {'name': 'output0', 'shape': (10, 10), 'dtype': 'fp32', 'format': 'ND', 'need_trans': False}
        ],
        'view_shape': (13, 10),
        'tile_shape': (3, 8),
        'params': {'axis': 0, 'alpha': 1.2},
        'source_case_index': 7,
    },
    {
        'case_index': 18,
        'case_name': 'IndexAddInplace_test_18',
        'operation': 'IndexAdd_',
        'input_tensors': [
            {
                'name': 'input0',
                'shape': (10, 10),
                'dtype': 'int8',
                'format': 'ND',
                'need_trans': False,
                'data_range': {'min': -100, 'max': 100},
            },
            {
                'name': 'input1',
                'shape': (10, 5),
                'dtype': 'int8',
                'format': 'ND',
                'need_trans': False,
                'data_range': {'min': -100, 'max': 100},
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
            {'name': 'output0', 'shape': (10, 10), 'dtype': 'int8', 'format': 'ND', 'need_trans': False}
        ],
        'view_shape': (10, 10),
        'tile_shape': (10, 32),
        'params': {'axis': 1, 'alpha': 127.0},
        'source_case_index': 18,
    },
]
