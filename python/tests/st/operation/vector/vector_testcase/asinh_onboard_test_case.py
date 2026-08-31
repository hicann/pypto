#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Testcase configuration for ASinh vector ST."""

from dataclasses import dataclass

from vector_testcase.vector_test_case import VectorConfig


@dataclass(frozen=True)
class AsinhOnboardConfig(VectorConfig):
    OPERATION = 'ASinh'


ASINH_ONBOARD_TESTS = [
    {
        'case_index': 2,
        'case_name': 'ASinh_test_3',
        'operation': 'ASinh',
        'input_tensors': [
            {
                'name': 'input0',
                'shape': (3585, 21),
                'dtype': 'fp16',
                'format': 'ND',
                'need_trans': False,
                'data_range': {'min': -6, 'max': 6},
            }
        ],
        'output_tensors': [
            {'name': 'output0', 'shape': (3585, 21), 'dtype': 'fp16', 'format': 'ND', 'need_trans': False}
        ],
        'view_shape': (449, 7),
        'tile_shape': (120, 8),
        'params': {},
        'source_case_index': 2,
    }
]
