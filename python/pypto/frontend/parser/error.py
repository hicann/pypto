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

"""Error classes and exception handling for PTO Script Parser.

This module defines custom exception classes and error handling utilities
for the PTO Script Parser. It provides enhanced error reporting with optional
backtraces controlled via environment variables.

Key Components:
    - ParserError: Base exception class for parser errors, associates errors
      with specific AST nodes for better source location reporting
    - RenderedParserError: A special error class indicating that the error
      message has already been formatted and displayed
    - Exception hook wrapper: Customizes Python's exception handling to
      provide cleaner error output and cleanup multiprocessing resources

Environment Variables:
    - PTO_BACKTRACE: Set to 1 to enable full Python backtraces for parser errors.
      By default (0), only the user-friendly error message is shown without
      internal stack traces, making errors more readable for end users.

The exception hook automatically cleans up multiprocessing child processes
when the parser is interrupted, preventing orphaned processes.
"""
import logging
import multiprocessing
import os
import sys
from typing import Union

from . import doc

PTO_BACKTRACE_ENV_VAR = "PTO_BACKTRACE"


class ParserError(Exception):
    """Error class for diagnostics."""

    def __init__(self, node: doc.AST, msg: Union[str, Exception]):
        if isinstance(msg, Exception):
            msg = f"{type(msg).__name__}: {msg}"
        super().__init__(msg)
        self.node = node


class RenderedParserError(ParserError):
    """Error class for diagnostics with rendered message."""


def _should_print_backtrace():
    pto_backtrace = os.environ.get(PTO_BACKTRACE_ENV_VAR, "0")

    try:
        pto_backtrace = bool(int(pto_backtrace))
    except ValueError as exc:
        raise ValueError(
            f"invalid value for {PTO_BACKTRACE_ENV_VAR} {pto_backtrace}, please set to 0 or 1."
        ) from exc

    return pto_backtrace


def pto_wrap_excepthook(exception_hook):
    """Wrap given excepthook with PTO additional work."""

    def wrapper(exc_type, value, trbk):
        """Clean subprocesses when PTO is interrupted."""

        if exc_type is RenderedParserError:
            if not _should_print_backtrace():
                logging.info(
                    f"note: run with `{PTO_BACKTRACE_ENV_VAR}=1` environment variable to display a backtrace."
                )
            else:
                exception_hook(exc_type, value, trbk)
        else:
            exception_hook(exc_type, value, trbk)

        if hasattr(multiprocessing, "active_children"):
            for p in multiprocessing.active_children():
                p.terminate()

    return wrapper


sys.excepthook = pto_wrap_excepthook(sys.excepthook)
