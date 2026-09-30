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
"""Case-conversion for the C++ codegen: map an op_type to the snake_case stem of its generated files.

The generated C++ derives its file stems from an op_type like ``PyptoCustomOpAdd``. The conversion is a
codegen concern, so it lives in ``exported_custom_op_litenpu.deploy`` with the C++ codegen it serves.
"""
import re

__all__ = ("camel_case_to_snake_case",)


def camel_case_to_snake_case(name: str) -> str:
    """Convert PascalCase or camelCase (e.g. ``Add``, ``PyptoCustomOpAdd``) to snake_case for paths."""
    step1 = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", name)
    step2 = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", step1)
    return step2.lower()
