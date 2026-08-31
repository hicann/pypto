#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Testcase configuration for Log1p vector ST."""

from dataclasses import dataclass

from vector_testcase.vector_test_case import VectorConfig


@dataclass(frozen=True)
class Log1pConfig(VectorConfig):
    OPERATION = 'Log1p'


LOG1P_TESTS = [
    {
        'case_index': 0,
        'case_name': 'Log1p_test_0',
        'operation': 'Log1p',
        'input_tensors': [
            {
                'name': 'input0',
                'shape': (3072, 1),
                'dtype': 'fp32',
                'format': 'ND',
                'need_trans': False,
                'data_range': {'min': 1.1e-14, 'max': 1.1e-14},
            }
        ],
        'output_tensors': [
            {'name': 'output0', 'shape': (3072, 1), 'dtype': 'fp32', 'format': 'ND', 'need_trans': False}
        ],
        'view_shape': (128, 1),
        'tile_shape': (32, 32),
        'params': {'base': 'e'},
        'source_case_index': 0,
    },
    {
        'case_index': 7,
        'case_name': 'Log1p_test_7',
        'operation': 'Log1p',
        'input_tensors': [
            {
                'name': 'input0',
                'shape': (3072, 384),
                'dtype': 'fp32',
                'format': 'ND',
                'need_trans': False,
                'data_range': {'min': 'max', 'max': 'max'},
            }
        ],
        'output_tensors': [
            {'name': 'output0', 'shape': (3072, 384), 'dtype': 'fp32', 'format': 'ND', 'need_trans': False}
        ],
        'view_shape': (128, 128),
        'tile_shape': (32, 32),
        'params': {'base': '10'},
        'source_case_index': 7,
    },
]
