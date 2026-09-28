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
"""Unit tests for the return-annotation output-arity helpers.

Covers both axes the parser is generic over (``torch.Size`` and ``torch.dtype``), every error
branch it raises, and the PEP 563 string-annotation path that only ``get_type_hints`` resolves.
"""
import pytest
import torch

from pypto.extensions.torch_custom_op_litenpu.common.return_annotations import (
    _item_repr,
    infer_shape_output_arity,
    parse_output_arity,
)


def test_single_item_annotation_is_one_output():
    assert parse_output_arity(torch.Size, torch.Size, "ctx") == 1


def test_explicit_tuple_arity():
    assert parse_output_arity(tuple[torch.Size, torch.Size], torch.Size, "ctx") == 2
    assert parse_output_arity(tuple[torch.Size, torch.Size, torch.Size], torch.Size, "ctx") == 3


def test_variadic_tuple_is_rejected():
    with pytest.raises(TypeError, match="variadic-arity"):
        parse_output_arity(tuple[torch.Size, ...], torch.Size, "ctx")


def test_empty_tuple_is_rejected():
    with pytest.raises(TypeError, match="tuple annotation must have at least one element"):
        parse_output_arity(tuple[()], torch.Size, "ctx")


def test_wrong_item_type_names_the_index():
    with pytest.raises(TypeError, match=r"output\[1\]"):
        parse_output_arity(tuple[torch.Size, int], torch.Size, "ctx")


def test_non_tuple_non_item_annotation_is_rejected():
    with pytest.raises(TypeError, match="expected torch.Size or tuple"):
        parse_output_arity(int, torch.Size, "ctx")


def test_dtype_axis():
    assert parse_output_arity(torch.dtype, torch.dtype, "c") == 1
    assert parse_output_arity(tuple[torch.dtype, torch.dtype], torch.dtype, "c") == 2
    with pytest.raises(TypeError, match="expected torch.dtype"):
        parse_output_arity(torch.Size, torch.dtype, "c")


def test_missing_return_annotation_is_rejected():
    def no_ann(a):
        return a

    with pytest.raises(TypeError, match="must have a return annotation"):
        infer_shape_output_arity(no_ann)


def test_unresolvable_annotation_is_rejected():
    def bad_ann(a):
        return a

    bad_ann.__annotations__["return"] = "NoSuchName"
    with pytest.raises(TypeError, match="resolvable by get_type_hints"):
        infer_shape_output_arity(bad_ann)


def test_annotated_functions_report_their_output_arity():
    def one(a) -> torch.Size:
        return a

    def three(a) -> tuple[torch.Size, torch.Size, torch.Size]:
        return a

    assert infer_shape_output_arity(one) == 1
    assert infer_shape_output_arity(three) == 3


def test_item_repr_names():
    class Plain:
        pass

    assert _item_repr(torch.Size) == "torch.Size"
    assert _item_repr(torch.dtype) == "torch.dtype"
    assert _item_repr(Plain) == Plain.__qualname__


def test_string_annotations_are_resolved():
    def one(a) -> "torch.Size":
        return a

    def two(a) -> "tuple[torch.Size, torch.Size]":
        return a

    assert infer_shape_output_arity(one) == 1
    assert infer_shape_output_arity(two) == 2
