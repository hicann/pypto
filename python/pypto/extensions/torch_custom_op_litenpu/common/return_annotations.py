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
"""Read the output arity a return annotation declares.

The annotation can be passed directly, or resolved from a function first.
"""
import inspect
from typing import Any, Callable, get_args, get_origin, get_type_hints

import torch

__all__ = ("parse_output_arity", "infer_shape_output_arity")


def _item_repr(item: Any) -> str:
    """Pretty name for diagnostic messages (``torch.Size``, ``torch.dtype``, etc.)."""
    if item is torch.Size:
        return "torch.Size"
    if item is torch.dtype:
        return "torch.dtype"
    return getattr(item, "__qualname__", None) or getattr(item, "__name__", repr(item))


def parse_output_arity(ret_ann: Any, expected_item: Any, ctx: str) -> int:
    """Return the output arity declared by *ret_ann*.

    Single output is *expected_item* (e.g. ``torch.Size`` or ``torch.dtype``).
    Multi-output is ``tuple[<item>, <item>, ...]`` with an explicit arity
    (no ``Ellipsis``). The variadic form is rejected because the output arity must
    be known at export time.
    """
    item_name = _item_repr(expected_item)
    if ret_ann is expected_item:
        return 1
    if get_origin(ret_ann) is tuple:
        targs = get_args(ret_ann)
        if not targs:
            raise TypeError(f"{ctx}: tuple annotation must have at least one element")
        if len(targs) == 2 and targs[1] is Ellipsis:
            raise TypeError(
                f"{ctx}: variadic-arity multi-output (tuple[{item_name}, ...]) is not "
                f"supported; use tuple[{item_name}, {item_name}] with an explicit arity"
            )
        for i, a in enumerate(targs):
            if a is not expected_item:
                raise TypeError(f"{ctx} output[{i}]: expected {item_name}, got {a!r}")
        return len(targs)
    raise TypeError(f"{ctx}: expected {item_name} or tuple[{item_name}, {item_name}], got {ret_ann!r}")


def infer_shape_output_arity(func: Callable) -> int:
    """The output arity ``infer_shape`` declares through its return annotation.

    Resolution goes through ``get_type_hints``, so a module using postponed evaluation (PEP 563)
    reads correctly; a missing or malformed annotation raises ``TypeError``.
    """
    sig = inspect.signature(func)
    try:
        hints = get_type_hints(func, include_extras=True)
    except Exception as exc:
        raise TypeError(f"infer_shape requires type annotations resolvable by get_type_hints: {exc}") from exc
    if sig.return_annotation is inspect.Signature.empty:
        raise TypeError(
            "infer_shape must have a return annotation (torch.Size or tuple[torch.Size, torch.Size])"
        )
    ret_ann = hints.get("return", sig.return_annotation)
    return parse_output_arity(ret_ann, torch.Size, "infer_shape return")
