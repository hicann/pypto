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
"""Package B compute helpers — same NAMES as package A (bias, _amount, CONST) but distinct objects."""

CONST = 20  # same NAME as pkg_a.constants.CONST, DIFFERENT value
SHARED = (8, 8)  # same NAME + same VALUE as pkg_a.constants.SHARED -> deduped to ONE emitted name


def _amount():
    """Package B's OWN ``_amount`` (distinct object from pkg_a.compute._amount)."""
    return 2


def bias(x):
    """Same NAME as pkg_a.compute.bias, DIFFERENT object -> distinct canonical name."""
    return x + _amount() + CONST
