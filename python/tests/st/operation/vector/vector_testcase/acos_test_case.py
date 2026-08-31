#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Testcase configuration for Acos vector ST."""

from dataclasses import dataclass

from vector_testcase.vector_test_case import VectorConfig


@dataclass(frozen=True)
class AcosConfig(VectorConfig):
    OPERATION = 'Acos'


ACOS_TESTS = [
    {
        'case_index': 0,
        'case_name': 'Acos_test_1',
        'operation': 'Acos',
        'input_tensors': [
            {
                'name': 'input0',
                'shape': (8, 64),
                'dtype': 'fp32',
                'format': 'ND',
                'need_trans': False,
                'data_range': {'min': -1, 'max': 1},
            }
        ],
        'output_tensors': [{'name': 'output0', 'shape': (8, 64), 'dtype': 'fp32', 'format': 'ND', 'need_trans': False}],
        'view_shape': (4, 32),
        'tile_shape': (4, 16),
        'params': {},
        'source_case_index': 0,
    }
]
