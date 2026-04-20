#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025 CANN community contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Error classes and exception handling for PyPTO.

This module defines custom exception classes and error handling utilities
for the PyPTO framework. It provides a consistent error interface with
error codes for different error categories.

Classes:
    PyptoError: Base exception class for all PyPTO errors.
    ParserError: Parser exception with AST node association and error codes.
    RenderedParserError: A special error class indicating that the error.
"""
import ast
from typing import Union, Optional


_ERROR_CODE_UNKNOWN = 0xF1FFFF

_error_mapping = {
    TypeError: 0xF00001,
    ValueError: 0xF00002,
    RuntimeError: 0xF00003,
    NameError: 0xF00004,
    NotImplementedError: 0xF00005,
}


def _get_err_code(msg: Exception) -> int:
    """Get error code based on exception type."""
    return _error_mapping.get(type(msg), _ERROR_CODE_UNKNOWN)


class PyptoError(Exception):
    """Base exception class for all PyPTO errors.
    
    Args:
        msg: Exception object or error message string.
        err_code: Error code (int). Defaults to _ERROR_CODE_UNKNOWN.
    """

    def __init__(self, msg: Union[str, Exception], err_code: Optional[int] = None):
        if err_code is None:
            err_code = _ERROR_CODE_UNKNOWN
        if isinstance(msg, Exception):
            msg = f"ErrCode: {err_code:X}, {type(msg).__name__}: {msg}"
        else:
            msg = f"ErrCode: {err_code:X}, {msg}"
        super().__init__(msg)
        self.node: Optional[ast.AST] = None


class ParserError(PyptoError):
    """ParserError class for diagnostics.

    Args:
        node: AST node where the error occurred, used for source location
              reporting and error message formatting.
        msg: Exception object or error message string.

    Attributes:
        node: The AST node where the error occurred.
    """

    def __init__(self, node: ast.AST, msg: Union[str, Exception]):
        err_code = _get_err_code(msg) if isinstance(msg, Exception) else None
        super().__init__(msg, err_code=err_code)
        self.node = node


class RenderedParserError(ParserError):
    """Error class for diagnostics with rendered message."""