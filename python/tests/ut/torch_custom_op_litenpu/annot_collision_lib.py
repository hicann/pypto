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
"""Fixture for the annotation-vs-body constant-collision test.

Exports ``PIN_SHAPE`` (value A), read by ``body_shape``; a kernel in ``kernel_snippet_samples`` pins a
same-named, differently-valued ``PIN_SHAPE`` in its parameter annotation.
"""

PIN_SHAPE = (1, 1, 4, 64)


def body_shape():
    """Reads this module's ``PIN_SHAPE``."""
    return PIN_SHAPE
