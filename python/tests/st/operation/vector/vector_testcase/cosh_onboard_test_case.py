#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Testcase configuration for Cosh vector ST."""

from dataclasses import dataclass

from vector_testcase.vector_test_case import VectorConfig


@dataclass(frozen=True)
class CoshOnboardConfig(VectorConfig):
    OPERATION = 'Cosh'


COSH_ONBOARD_TESTS = [
    {
        'case_index': 2,
        'case_name': 'Cosh_test_3',
        'operation': 'Cosh',
        'input_tensors': [
            {
                'name': 'input0',
                'shape': (2048, 31),
                'dtype': 'fp32',
                'format': 'ND',
                'need_trans': False,
                'data_range': {'min': -22, 'max': 22},
            }
        ],
        'output_tensors': [
            {'name': 'output0', 'shape': (2048, 31), 'dtype': 'fp32', 'format': 'ND', 'need_trans': False}
        ],
        'view_shape': (256, 8),
        'tile_shape': (128, 8),
        'params': {},
        'source_case_index': 2,
    }
]
