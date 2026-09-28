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
"""Helpers with param/return annotations for the annotation-strip cases:
an un-emittable class annotation is STRIPPED from the packed source (the AST parser ignores it), while
``torch.Tensor`` / ``int`` are kept (pre-import / builtin filters). A TENSOR-typed un-emittable
annotation still raises (not exercised here; see the live-annotation residual)."""
import typing

import torch


class UserClass:
    """A user class that cannot be packed into the snippet (would NameError at import)."""


class KeepMe:
    """A second user class in a multi-param helper — its annotation is also stripped, independently."""


def uses_unemittable_class(x: UserClass):
    """A non-tensor annotation referencing an un-emittable class: the annotation is stripped
    from the packed source (``def <canon>(x)``); the snippet imports (no NameError on ``UserClass``)."""
    return x


def uses_optional_unemittable(x: typing.Optional[UserClass]):
    """A mixed subscript ``Optional[UserClass]``: the whole non-tensor annotation is stripped."""
    return x


def uses_unemittable_with_default(x: UserClass = 3):
    """``x: UserClass = 3`` strips to ``def <canon>(x=3)``: the default is preserved because
    the delete span is name-end anchored, not a backward ``:`` scan)."""
    return x


def uses_multi_param_one_stripped(a: int, x: UserClass, y: KeepMe):
    """Only ``x``/``y``'s un-emittable annotations are stripped; ``a: int`` (builtin) is kept."""
    return a


def uses_tensor_annotation(x: torch.Tensor):
    """``torch.Tensor`` is a pre-import attribute -> filtered, NO strip, NO raise."""
    return x


def uses_int_annotation(x: int) -> int:
    """``int`` is a builtin -> filtered, NO strip, NO raise."""
    return x
