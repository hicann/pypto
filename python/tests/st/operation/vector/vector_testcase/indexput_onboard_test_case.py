#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Testcase configuration for IndexPut_ vector ST."""

from dataclasses import dataclass

from vector_testcase.vector_test_case import VectorConfig


@dataclass(frozen=True)
class IndexputOnboardConfig(VectorConfig):
    OPERATION = 'IndexPut_'


INDEXPUT_ONBOARD_TESTS = [
    {
        'case_index': 37,
        'case_name': 'IndexPutInplace_test_37',
        'operation': 'IndexPut_',
        'input_tensors': [
            {
                'name': 'input0',
                'shape': (5, 11, 13, 31),
                'dtype': 'fp32',
                'format': 'ND',
                'need_trans': False,
                'data_range': {'min': -1, 'max': 1},
            },
            {
                'name': 'input1',
                'shape': (3, 11, 13, 31),
                'dtype': 'fp32',
                'format': 'ND',
                'need_trans': False,
                'data_range': {'min': -10, 'max': 10},
            },
            {
                'name': 'input2',
                'shape': (3,),
                'dtype': 'int32',
                'format': 'ND',
                'need_trans': False,
                'data_range': {'min': 0, 'max': 5},
            },
        ],
        'output_tensors': [
            {'name': 'output0', 'shape': (5, 11, 13, 31), 'dtype': 'fp32', 'format': 'ND', 'need_trans': False}
        ],
        'view_shape': (7,),
        'tile_shape': (3, 5, 7, 9),
        'params': {'accumulate': False},
        'source_case_index': 37,
    },
    {
        'case_index': 42,
        'case_name': 'IndexPutInplace_test_42',
        'operation': 'IndexPut_',
        'input_tensors': [
            {
                'name': 'input0',
                'shape': (5, 11, 13, 31),
                'dtype': 'fp32',
                'format': 'ND',
                'need_trans': False,
                'data_range': {'min': -1, 'max': 1},
            },
            {
                'name': 'input1',
                'shape': (10, 31),
                'dtype': 'fp32',
                'format': 'ND',
                'need_trans': False,
                'data_range': {'min': -10, 'max': 10},
            },
            {
                'name': 'input2',
                'shape': (10,),
                'dtype': 'int32',
                'format': 'ND',
                'need_trans': False,
                'data_range': {'min': 0, 'max': 5},
            },
            {
                'name': 'input3',
                'shape': (10,),
                'dtype': 'int32',
                'format': 'ND',
                'need_trans': False,
                'data_range': {'min': 0, 'max': 11},
            },
            {
                'name': 'input4',
                'shape': (10,),
                'dtype': 'int32',
                'format': 'ND',
                'need_trans': False,
                'data_range': {'min': 0, 'max': 13},
            },
        ],
        'output_tensors': [
            {'name': 'output0', 'shape': (5, 11, 13, 31), 'dtype': 'fp32', 'format': 'ND', 'need_trans': False}
        ],
        'view_shape': (7,),
        'tile_shape': (3, 5),
        'params': {'accumulate': False},
        'source_case_index': 42,
    },
]
