#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Testcase configuration for BitwiseAnd vector ST."""

from dataclasses import dataclass

from vector_testcase.vector_test_case import VectorConfig


@dataclass(frozen=True)
class BitwiseandOnboardConfig(VectorConfig):
    OPERATION = 'BitwiseAnd'


BITWISEAND_ONBOARD_TESTS = [
    {
        'case_index': 1,
        'case_name': 'BitwiseAnd_2D_align_1',
        'operation': 'BitwiseAnd',
        'input_tensors': [
            {
                'name': 'input0',
                'shape': (1, 32),
                'dtype': 'uint16',
                'format': 'ND',
                'need_trans': False,
                'data_range': {'min': 0, 'max': 65535},
            },
            {
                'name': 'input1',
                'shape': (32, 32),
                'dtype': 'uint16',
                'format': 'ND',
                'need_trans': False,
                'data_range': {'min': 0, 'max': 65535},
            },
        ],
        'output_tensors': [
            {'name': 'output0', 'shape': (32, 32), 'dtype': 'uint16', 'format': 'ND', 'need_trans': False}
        ],
        'view_shape': (16, 16),
        'tile_shape': (16, 16),
        'params': {},
        'source_case_index': 1,
    }
]
