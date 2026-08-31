#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Testcase configuration for Pack vector ST."""

from dataclasses import dataclass

from vector_testcase.vector_test_case import VectorConfig


@dataclass(frozen=True)
class PackOnboardConfig(VectorConfig):
    OPERATION = 'Pack'


PACK_ONBOARD_TESTS = [
    {
        'case_index': 1,
        'case_name': 'Pack_test_1',
        'operation': 'Pack',
        'input_tensors': [
            {
                'name': 'input0',
                'shape': (32, 32),
                'dtype': 'int32',
                'format': 'ND',
                'need_trans': False,
                'data_range': {'min': -2147483648, 'max': 2147483647},
            }
        ],
        'output_tensors': [
            {'name': 'output0', 'shape': (4096,), 'dtype': 'uint8', 'format': 'ND', 'need_trans': False}
        ],
        'view_shape': (32, 32),
        'tile_shape': (32, 32),
        'params': {},
        'source_case_index': 1,
    }
]
