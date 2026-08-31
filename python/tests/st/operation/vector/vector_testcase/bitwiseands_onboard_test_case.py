#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Testcase configuration for BitwiseAnds vector ST."""

from dataclasses import dataclass

from vector_testcase.vector_test_case import VectorConfig


@dataclass(frozen=True)
class BitwiseandsOnboardConfig(VectorConfig):
    OPERATION = 'BitwiseAnds'


BITWISEANDS_ONBOARD_TESTS = [
    {
        'case_index': 2,
        'case_name': 'BitwiseAnd_scalar_2D_unalign_0',
        'operation': 'BitwiseAnds',
        'input_tensors': [
            {
                'name': 'input0',
                'shape': (17, 31),
                'dtype': 'int16',
                'format': 'ND',
                'need_trans': False,
                'data_range': {'min': 0, 'max': 255},
            }
        ],
        'output_tensors': [
            {'name': 'output0', 'shape': (17, 31), 'dtype': 'int16', 'format': 'ND', 'need_trans': False}
        ],
        'view_shape': (8, 16),
        'tile_shape': (16, 16),
        'params': {'scalar': 127, 'scalar_type': 'int16'},
        'source_case_index': 2,
    }
]
