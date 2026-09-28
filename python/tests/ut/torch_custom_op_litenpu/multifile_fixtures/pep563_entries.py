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
"""Entry functions for the live PEP 563 tensor-param helper annotation tracer tests."""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
# ``sample_helper_pkg`` is a sibling of the test dir, one level above this fixture package.
sys.path.insert(0, os.path.dirname(_HERE))

from pep563_consts import alias_tensor_helper, const_tensor_helper  # noqa: E402
from sample_helper_pkg.pep563_helper import tensor_helper  # noqa: E402


def entry_uses_pep563():
    """References the PEP-563 tensor-param helper; it packs with a live ``pypto.Tensor(...)``
    annotation."""
    return tensor_helper


def entry_pep563_const_tensor(x):
    """References a PEP 563 tensor helper whose annotation uses module consts SHAPE/DTYPE."""
    return const_tensor_helper(x)


def entry_pep563_alias_tensor(x):
    """References a PEP-563 tensor helper whose annotation root is an un-packable alias -> RAISE."""
    return alias_tensor_helper(x)
