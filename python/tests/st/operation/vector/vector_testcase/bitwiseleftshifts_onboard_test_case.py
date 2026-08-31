#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Testcase configuration for BitwiseLeftShifts vector ST."""

from dataclasses import dataclass

from vector_testcase.vector_test_case import VectorConfig


@dataclass(frozen=True)
class BitwiseleftshiftsOnboardConfig(VectorConfig):
    OPERATION = 'BitwiseLeftShifts'


BITWISELEFTSHIFTS_ONBOARD_TESTS = [
    {
        'case_index': 0,
        'case_name': 'BitwiseLeftShifts_test_0',
        'operation': 'BitwiseLeftShifts',
        'input_tensors': [
            {
                'name': 'input0',
                'shape': (3072, 384),
                'dtype': 'int16',
                'format': 'ND',
                'need_trans': False,
                'data_range': {'min': -1, 'max': 1},
            }
        ],
        'output_tensors': [
            {'name': 'output0', 'shape': (3072, 384), 'dtype': 'int16', 'format': 'ND', 'need_trans': False}
        ],
        'view_shape': (128, 128),
        'tile_shape': (32, 32),
        'params': {'scalar': 1, 'scalar_type': 'int16'},
        'source_case_index': 0,
    }
]
