#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Testcase configuration for UnPack vector ST."""

from dataclasses import dataclass

from vector_testcase.vector_test_case import VectorConfig


@dataclass(frozen=True)
class UnpackOnboardConfig(VectorConfig):
    OPERATION = 'UnPack'


UNPACK_ONBOARD_TESTS = [
    {
        'case_index': 22,
        'case_name': 'UnPack_test_22',
        'operation': 'UnPack',
        'input_tensors': [
            {
                'name': 'input0',
                'shape': (34,),
                'dtype': 'uint8',
                'format': 'ND',
                'need_trans': False,
                'data_range': {'min': 0, 'max': 255},
            }
        ],
        'output_tensors': [{'name': 'output0', 'shape': (17,), 'dtype': 'fp16', 'format': 'ND', 'need_trans': False}],
        'view_shape': (23,),
        'tile_shape': (16,),
        'params': {},
        'source_case_index': 22,
    }
]
