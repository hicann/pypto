#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Python API declarations for the PyPTO Pro SIMT namespace (``pl.simt.*``).

These declarations exist so that:
- IDE "Go to Definition" works for every ``pl.simt.xxx`` call
- Python catches typos at import time
- Type checkers can validate argument types
- Docstrings document the user-facing calling convention

None of these functions are meant to be called at runtime.  Inside a PyPTO
kernel the AST parser intercepts every ``pl.simt.xxx`` call before Python executes
it.  Outside a kernel, calling a declaration raises ``RuntimeError``.
"""

from __future__ import annotations

from typing import Any, Callable

from ._api import Scalar, _api_decl


class Simt:
    """SIMT namespace (``pl.simt.*``).

    Provides SIMT context queries (thread_idx, block_dim, ...), launch, and
    the ``function`` decorator for defining SIMT VF functions.
    """

    @staticmethod
    def function(fn: Callable | None = None, *, max_threads: int | None = None) -> Callable:
        """Mark a callable for delayed SIMT parsing at its call site.

        Args:
            fn: Python function to decorate (when used without parentheses).
            max_threads: Launch bound for SIMT VF functions. When provided, the
                function is launchable via ``pl.simt.launch()``. When omitted,
                the function is a SIMT callee helper.
        """
        from .parser.decorator import simt_function

        return simt_function(fn, max_threads=max_threads)

    @staticmethod
    @_api_decl
    def thread_idx() -> Any:
        """Return the current thread coordinates within its thread block."""

    @staticmethod
    @_api_decl
    def block_dim() -> Any:
        """Return the dimensions of the current thread block."""

    @staticmethod
    @_api_decl
    def block_idx() -> Any:
        """Return the current block coordinates within the outer kernel grid."""

    @staticmethod
    @_api_decl
    def grid_dim() -> Any:
        """Return the dimensions of the outer kernel grid."""

    @staticmethod
    @_api_decl
    def linear_thread_idx() -> Scalar:
        """Return the x-major flattened thread index within the current block."""

    @staticmethod
    @_api_decl
    def warp_size() -> Scalar:
        """Return the target SIMT warp size."""

    @staticmethod
    @_api_decl
    def launch(callee: Any, *, threads: int | tuple[int, ...], args: tuple[Any, ...]) -> None:
        """Launch a SIMT function from a Vector section.

        Args:
            callee: Launchable function defined with ``@pl.simt.function(max_threads=...)``.
            threads: One- to three-dimensional compile-time thread configuration.
            args: Scalar, Tensor, and Tile arguments passed to ``callee``.
        """
