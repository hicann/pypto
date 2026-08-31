#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Testcase configuration for ACosh vector ST."""

from dataclasses import dataclass

from vector_testcase.vector_test_case import VectorConfig


@dataclass(frozen=True)
class AcoshOnboardConfig(VectorConfig):
    OPERATION = 'ACosh'


ACOSH_ONBOARD_TESTS = [
    {
        'case_index': 2,
        'case_name': 'ACosh_test_3',
        'operation': 'ACosh',
        'input_tensors': [
            {
                'name': 'input0',
                'shape': (1984, 27),
                'dtype': 'fp32',
                'format': 'ND',
                'need_trans': False,
                'data_range': {'min': 1.0001, 'max': 24},
            }
        ],
        'output_tensors': [
            {'name': 'output0', 'shape': (1984, 27), 'dtype': 'fp32', 'format': 'ND', 'need_trans': False}
        ],
        'view_shape': (248, 7),
        'tile_shape': (124, 8),
        'params': {},
        'source_case_index': 2,
    }
]
