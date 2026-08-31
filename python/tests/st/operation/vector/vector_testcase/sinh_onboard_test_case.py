#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Testcase configuration for Sinh vector ST."""

from dataclasses import dataclass

from vector_testcase.vector_test_case import VectorConfig


@dataclass(frozen=True)
class SinhOnboardConfig(VectorConfig):
    OPERATION = 'Sinh'


SINH_ONBOARD_TESTS = [
    {
        'case_index': 2,
        'case_name': 'Sinh_test_3',
        'operation': 'Sinh',
        'input_tensors': [
            {
                'name': 'input0',
                'shape': (4096, 16),
                'dtype': 'fp16',
                'format': 'ND',
                'need_trans': False,
                'data_range': {'min': -8, 'max': 8},
            }
        ],
        'output_tensors': [
            {'name': 'output0', 'shape': (4096, 16), 'dtype': 'fp16', 'format': 'ND', 'need_trans': False}
        ],
        'view_shape': (512, 4),
        'tile_shape': (128, 6),
        'params': {},
        'source_case_index': 2,
    }
]
