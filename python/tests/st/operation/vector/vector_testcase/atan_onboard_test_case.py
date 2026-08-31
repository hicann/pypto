#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Testcase configuration for Atan vector ST."""

from dataclasses import dataclass

from vector_testcase.vector_test_case import VectorConfig


@dataclass(frozen=True)
class AtanOnboardConfig(VectorConfig):
    OPERATION = 'Atan'


ATAN_ONBOARD_TESTS = [
    {
        'case_index': 5,
        'case_name': 'Atan_test_5',
        'operation': 'Atan',
        'input_tensors': [
            {
                'name': 'input0',
                'shape': (61, 37),
                'dtype': 'fp32',
                'format': 'ND',
                'need_trans': False,
                'data_range': {'min': -2, 'max': 2},
            }
        ],
        'output_tensors': [
            {'name': 'output0', 'shape': (61, 37), 'dtype': 'fp32', 'format': 'ND', 'need_trans': False}
        ],
        'view_shape': (7, 23),
        'tile_shape': (5, 17),
        'params': {},
        'source_case_index': 5,
    }
]
