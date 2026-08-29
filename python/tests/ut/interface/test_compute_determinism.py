#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Prove torch_npu deterministic level 0/1/2 is synced to global config."""

import pytest
import torch_npu

import pypto
from pypto._utils import get_torch_npu_compute_determinism_level
from pypto.frontend.parser.entry import JitCallableWrapper


def _dummy_kernel(x):
    return x


@pytest.fixture
def restore_determinism():
    prev_level = int(torch_npu.npu._get_deterministic_level())
    try:
        yield
    finally:
        torch_npu.npu.set_deterministic_level(prev_level)
        pypto.set_global_config("compute_determinism_level", 0)


def test_torch_npu_determinism_level_synced_to_global_config(restore_determinism):
    """torch_npu.set_deterministic_level -> set_global_config -> get_global_config."""
    wrapper = JitCallableWrapper(None, _dummy_kernel, None)

    for level in (0, 1, 2):
        torch_npu.npu.set_deterministic_level(level)
        assert get_torch_npu_compute_determinism_level() == level
        wrapper._set_config_option()
        assert pypto.get_global_config("compute_determinism_level") == level
