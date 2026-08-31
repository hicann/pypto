#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Testcase configuration for Var vector ST."""

from dataclasses import dataclass

from vector_testcase.vector_test_case import VectorConfig


@dataclass(frozen=True)
class VarOnboardConfig(VectorConfig):
    OPERATION = 'Var'


VAR_ONBOARD_TESTS = [
    {
        'case_index': 0,
        'case_name': 'Var_0',
        'operation': 'Var',
        'input_tensors': [
            {
                'name': 'input0',
                'shape': (128, 96),
                'dtype': 'fp32',
                'format': 'ND',
                'need_trans': False,
                'data_range': {'min': -10, 'max': 10},
            }
        ],
        'output_tensors': [{'name': 'output0', 'shape': (96,), 'dtype': 'fp32', 'format': 'ND', 'need_trans': False}],
        'view_shape': (128, 48),
        'tile_shape': (64, 48),
        'params': {'dim': [0], 'correction': 0.0, 'keepDim': False, 'skip': None},
        'source_case_index': 0,
    },
    {
        'case_index': 1,
        'case_name': 'Var_1',
        'operation': 'Var',
        'input_tensors': [
            {
                'name': 'input0',
                'shape': (128, 96),
                'dtype': 'fp16',
                'format': 'ND',
                'need_trans': False,
                'data_range': {'min': -100, 'max': 100},
            }
        ],
        'output_tensors': [{'name': 'output0', 'shape': (96,), 'dtype': 'fp16', 'format': 'ND', 'need_trans': False}],
        'view_shape': (128, 48),
        'tile_shape': (64, 48),
        'params': {'dim': [0], 'correction': 0.0, 'keepDim': False, 'skip': None},
        'source_case_index': 1,
    },
]
