# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Decorators and markers supported by the PyPTO Pro parser."""

from __future__ import annotations

__all__ = ["inline", "vector_function"]


from collections.abc import Callable

_SIMT_FUNCTION_MARKER = "_pypto_simt_function"
_SIMT_MAX_THREADS_ATTR = "_pypto_simt_max_threads"


def simt_function(fn: Callable | None = None, *, max_threads: int | None = None) -> Callable:
    """Mark a callable for delayed SIMT parsing at its call site."""
    if fn is None and max_threads is None:
        raise TypeError("@pl.simt.function requires max_threads or a directly decorated function")

    def decorate(func: Callable) -> Callable:
        setattr(func, _SIMT_FUNCTION_MARKER, True)
        setattr(func, _SIMT_MAX_THREADS_ATTR, max_threads)
        return func

    return decorate(fn) if fn is not None else decorate


def is_simt_function(fn: Callable) -> bool:
    """Return whether *fn* is marked as a SIMT function template."""
    return bool(getattr(fn, _SIMT_FUNCTION_MARKER, False))


def get_simt_max_threads(fn: Callable) -> int | None:
    """Return the launch bound recorded by @pl.simt.function."""
    return getattr(fn, _SIMT_MAX_THREADS_ATTR, None)


def inline(fn: Callable) -> Callable:
    """Deprecated compatibility marker for inline callables."""
    import warnings

    warnings.warn(
        "@pl.inline is deprecated and will be removed. "
        "Use @pl.fn with type annotations, or pass the function as an annotated callable.",
        DeprecationWarning,
        stacklevel=2,
    )
    return fn


def vector_function(fn: Callable) -> Callable:
    """Mark a callable as a vector-function body expanded at its call site."""
    _mark_vector_function(fn)
    return fn


def _mark_vector_function(fn: Callable) -> None:
    """Set the internal marker used for vector-function call-site expansion."""
    setattr(fn, "_pypto_vector_function", True)


def is_vector_function(fn: Callable) -> bool:
    """Return whether *fn* is marked as a vector-function body."""
    return bool(getattr(fn, "_pypto_vector_function", False))
