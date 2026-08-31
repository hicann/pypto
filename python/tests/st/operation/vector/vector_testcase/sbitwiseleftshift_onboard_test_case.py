#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Testcase configuration for SBitwiseLeftShift vector ST."""

from dataclasses import dataclass

from vector_testcase.vector_test_case import VectorConfig


@dataclass(frozen=True)
class SbitwiseleftshiftOnboardConfig(VectorConfig):
    OPERATION = 'SBitwiseLeftShift'


SBITWISELEFTSHIFT_ONBOARD_TESTS = [
    {
        'case_index': 0,
        'case_name': 'SBitwiseLeftShift_test_0',
        'operation': 'SBitwiseLeftShift',
        'input_tensors': [
            {
                'name': 'input0',
                'shape': (3072, 384),
                'dtype': 'int16',
                'format': 'ND',
                'need_trans': False,
                'data_range': {'min': -10, 'max': 10},
            }
        ],
        'output_tensors': [
            {'name': 'output0', 'shape': (3072, 384), 'dtype': 'int16', 'format': 'ND', 'need_trans': False}
        ],
        'view_shape': (128, 128),
        'tile_shape': (32, 32),
        'params': {'scalar': 30, 'scalar_type': 'int16'},
        'source_case_index': 0,
    }
]
