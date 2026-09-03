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

"""Parser diagnostics and error reporting."""

__all__ = [
    # Exceptions
    "ParserError",
    "ParserSyntaxError",
    "ParserTypeError",
    "FinalRejectionError",
    "UndefinedVariableError",
    "SSAViolationError",
    "UnsupportedFeatureError",
    "InvalidOperationError",
    "ScopeIsolationError",
    # Error codes
    "ErrorCode",
    "get_error_code",
    # Range checks
    "check_const_expr_fits_dtype",
    "check_fits_dtype",
    "check_in_range",
    "check_ir_int",
    "make_const_int",
    "range_message",
]


from ._error_codes import ErrorCode, get_error_code
from ._exceptions import (
    FinalRejectionError,
    InvalidOperationError,
    ParserError,
    ParserSyntaxError,
    ParserTypeError,
    ScopeIsolationError,
    SSAViolationError,
    UndefinedVariableError,
    UnsupportedFeatureError,
)
from ._range import (
    check_const_expr_fits_dtype,
    check_fits_dtype,
    check_in_range,
    check_ir_int,
    make_const_int,
    range_message,
)
