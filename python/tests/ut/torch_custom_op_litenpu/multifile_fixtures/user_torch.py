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
"""A user module whose top-level names literally include ``torch`` / ``pypto`` (DIFFERENT objects
from the real pre-imports) — the tracer's IDENTITY skip must rename these, never drop them."""


def torch():  # noqa: A001 - deliberately shadows the pre-import NAME (a different object)
    return 100


def pypto():  # noqa: A001 - deliberately shadows the pre-import NAME
    return 200
