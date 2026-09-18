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

"""PyPTO error codes used by parser diagnostics."""

from __future__ import annotations

__all__ = ["ErrorCode", "get_error_code"]


from enum import IntEnum

from pypto.pypto_impl import ExternalError, InternalErrorCode

_PYPTO_ERROR_CODE_PREFIX = 0xF00000


def _to_pypto_error_code(error: ExternalError | InternalErrorCode) -> int:
    """Convert a centrally allocated error to its PyPTO error number."""
    return _PYPTO_ERROR_CODE_PREFIX | int(error)


class ErrorCode(IntEnum):
    """PyPTO error codes, backed by the centrally allocated codes in error_code.h.

    The values are shared with pypto.
    """

    # External: the user wrote something the framework cannot accept.
    INVALID_TYPE = _to_pypto_error_code(ExternalError.INVALID_TYPE)
    INVALID_VAL = _to_pypto_error_code(ExternalError.INVALID_VAL)
    RUNTIME_ERROR = _to_pypto_error_code(ExternalError.RUNTIME_ERROR)
    NAME_ERROR = _to_pypto_error_code(ExternalError.NAME_ERROR)
    NOT_IMPLEMENTED_ERROR = _to_pypto_error_code(ExternalError.NOT_IMPLEMENTED_ERROR)
    KEY_ERROR = _to_pypto_error_code(ExternalError.KEY_ERROR)
    INVALID_OPERATION = _to_pypto_error_code(ExternalError.INVALID_OPERATION)
    OUT_OF_RANGE = _to_pypto_error_code(ExternalError.OUT_OF_RANGE)
    BAD_FD = _to_pypto_error_code(ExternalError.BAD_FD)
    DYNAMIC_SHAPE_COMPUTE_UNSUPPORTED = _to_pypto_error_code(ExternalError.DYNAMIC_SHAPE_COMPUTE_UNSUPPORTED)
    INVALID_SHAPE = _to_pypto_error_code(ExternalError.INVALID_SHAPE)
    INVALID_TILE = _to_pypto_error_code(ExternalError.INVALID_TILE)
    INVALID_FORMAT = _to_pypto_error_code(ExternalError.INVALID_FORMAT)
    INVALID_ARGUMENT = _to_pypto_error_code(ExternalError.INVALID_ARGUMENT)
    COMMON_EXTERNAL_ERROR = _to_pypto_error_code(ExternalError.COMMON_EXTERNAL_ERROR)

    # Internal: the framework broke its own invariant. Only three stages exist
    # in pypto_pro; everything outside pass and codegen uses COMMON.
    COMMON_INNER_ERROR = _to_pypto_error_code(InternalErrorCode.COMMON_INNER_ERROR)
    PASS_INNER_ERROR = _to_pypto_error_code(InternalErrorCode.PASS_INNER_ERROR)
    CODEGEN_INNER_ERROR = _to_pypto_error_code(InternalErrorCode.CODEGEN_INNER_ERROR)

    UNKNOWN = _to_pypto_error_code(ExternalError.UNKNOWN)


def get_error_code(exception_type: type) -> ErrorCode | None:
    """Get error code for exception type.

    Args:
        exception_type: Exception class

    Returns:
        Corresponding error code or None
    """

    error_code = getattr(exception_type, "error_code", None)
    return error_code if isinstance(error_code, ErrorCode) else None
