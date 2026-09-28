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
"""A kernel whose OWN module hand-defines a helper literally named ``helpers_x__shared`` AND also
reaches the cross-file ``helpers_x.shared`` (whose canonical name IS ``helpers_x__shared``) — two
DIFFERENT objects that would land on the same emitted name. The hard collision guard must RAISE."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import helpers_x  # cross-file: helpers_x.shared -> canonical ``helpers_x__shared``


def helpers_x__shared():  # same-file helper whose ORIGINAL name == the cross-file helper's canonical
    return "collision"


def kernel_collides(x):
    return helpers_x__shared() + helpers_x.shared() + x
