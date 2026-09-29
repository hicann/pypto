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
"""Numerical guards for the three declaration shapes the demos introduce: attrs, multi-output, factory.

Everything here runs on CPU through the torch-reference path (``_impl`` calls the op's ``torch_defn``), so it
needs torch and nothing else and is a real gate everywhere the package imports.

Operands are exact binary fractions (multiples of 1/64, all strictly positive), so every comparison is
``rtol=0, atol=0``: no tolerance is available to absorb a wrong result, and no elementwise difference can be
lost to rounding.
"""
import pytest
import torch

import pypto
from pypto.extensions.torch_custom_op_litenpu import AttrSpec, ExportedCustomOp
from pypto.extensions.torch_custom_op_litenpu.common import node_meta
from pypto.extensions.torch_custom_op_litenpu.common.authoring import _detect_factory_signature

SHAPE = (1, 8, 1, 64)          # 512 elements, tile (1, 4, 1, 64)
SWAP_SHAPE = (1, 4, 1, 32)     # deliberately different from SHAPE — see the multi-output guard
FACTORY_DTYPES = (torch.float16, torch.float32, torch.bfloat16)


def _exact_operands(shape, dtype, offset):
    """Operands that are exact in fp16/bf16/fp32 alike: strictly positive multiples of 1/64."""
    numel = 1
    for dim in shape:
        numel *= dim
    steps = (torch.arange(numel) + offset) % 64 + 1     # 1 .. 64
    return (steps.to(torch.float64) / 64.0).reshape(shape).to(dtype)


# ── the Float-attr declaration ────────────────────────────────────────────────────────────────────────
def _scale_factory(shapes, dtypes, attrs, soc_version, run_mode=pypto.RunMode.SIM):
    dtype = dtypes[0]
    scale = float(attrs["scale"])   # attr values reach the factory as strings

    @pypto.frontend.jit(codegen_options={"soc_version": soc_version},
                        runtime_options={"run_mode": run_mode})
    def attrs_scale_inner(a: pypto.Tensor([...], dtype), b: pypto.Tensor([...], dtype),
                        out: pypto.Tensor([...], dtype)):
        pypto.set_vec_tile_shapes(1, 4, 1, 64)
        out.move((a + b) * scale)

    return attrs_scale_inner


def _scale_infer_shape(a: torch.Size, b: torch.Size) -> torch.Size:
    return a


def _scale_infer_dtype(a: torch.dtype, b: torch.dtype) -> torch.dtype:
    return a


def _scale_torch(a, b, scale):
    return (a + b) * scale


SCALE_OP = ExportedCustomOp(
    kernel=_scale_factory,
    infer_shape=_scale_infer_shape,
    infer_dtype=_scale_infer_dtype,
    torch_defn=_scale_torch,
    torch_op_qualname="pypto::attrs_scale",
    attrs=[AttrSpec("scale", "Float", 1.0)],
)


# ── the multi-output declaration (a swap: out0 = b, out1 = a) ─────────────────────────────────────────
@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.SIM})
def _attrs_swap_kernel(a: pypto.Tensor([...]), b: pypto.Tensor([...]),
                     out0: pypto.Tensor([...]), out1: pypto.Tensor([...])):
    pypto.set_vec_tile_shapes(1, 4, 1, 32)
    out0.move(b)
    out1.move(a)


def _swap_infer_shape(a: torch.Size, b: torch.Size) -> tuple[torch.Size, torch.Size]:
    return (b, a)


def _swap_infer_dtype(a: torch.dtype, b: torch.dtype) -> tuple[torch.dtype, torch.dtype]:
    return (b, a)


def _swap_torch(a, b):
    # clone: a torch custom op may not return one of its own inputs
    return b.clone(), a.clone()


SWAP_OP = ExportedCustomOp(
    kernel=_attrs_swap_kernel,
    infer_shape=_swap_infer_shape,
    infer_dtype=_swap_infer_dtype,
    torch_defn=_swap_torch,
    torch_op_qualname="pypto::attrs_swap",
)


# ── the factory declaration (dtype-polymorphic add) ───────────────────────────────────────────────────
def _add_factory(shape, dtype, soc_version, run_mode=pypto.RunMode.SIM):
    @pypto.frontend.jit(codegen_options={"soc_version": soc_version},
                        runtime_options={"run_mode": run_mode})
    def attrs_add_inner(a: pypto.Tensor([...], dtype), b: pypto.Tensor([...], dtype),
                      out: pypto.Tensor([...])):
        pypto.set_vec_tile_shapes(1, 4, 1, 64)
        out.move(a + b)

    return attrs_add_inner


def _add_infer_shape(a: torch.Size, b: torch.Size) -> torch.Size:
    return a


def _add_infer_dtype(a: torch.dtype, b: torch.dtype) -> torch.dtype:
    return a


def _add_torch(a, b):
    return a + b


ADD_FACTORY_OP = ExportedCustomOp(
    kernel=_add_factory,
    infer_shape=_add_infer_shape,
    infer_dtype=_add_infer_dtype,
    torch_defn=_add_torch,
    torch_op_qualname="pypto::attrs_add_factory",
)


# ── Float attribute ───────────────────────────────────────────────────────────────────────────────────
def test_float_attr_is_declared_as_float():
    # The value under test travels through a declared Float channel, not an untyped extra operand.
    (spec,) = SCALE_OP._attr_specs
    assert (spec.name, spec.type) == ("scale", "Float")


def test_float_attr_of_two_scales_every_element():
    # Same operands, three scales: the attribute must CHANGE the numbers, and by exactly the declared
    # factor. Operands are strictly positive, so a scale of 2.0 has to move every single element — an
    # attribute that arrived but was ignored would leave `scaled` equal to `unit`.
    a = _exact_operands(SHAPE, torch.float16, 0)
    b = _exact_operands(SHAPE, torch.float16, 17)
    base = a + b
    assert bool((base > 0).all()), "operands must be strictly positive for 'scaling changes it' to be provable"

    unit = torch.ops.pypto.attrs_scale(a, b, scale=1.0)
    scaled = torch.ops.pypto.attrs_scale(a, b, scale=2.0)
    halved = torch.ops.pypto.attrs_scale(a, b, scale=0.5)

    # exact: the operands are binary fractions and 2.0 / 0.5 are powers of two, so no rounding occurs
    torch.testing.assert_close(unit, base, rtol=0, atol=0)
    torch.testing.assert_close(scaled, base * 2.0, rtol=0, atol=0)
    torch.testing.assert_close(halved, base * 0.5, rtol=0, atol=0)

    # the effect itself: EVERY element moved, in both directions
    assert bool(((scaled - unit).abs() > 0).all()), "scale=2.0 left at least one element unchanged"
    assert bool(((halved - unit).abs() > 0).all()), "scale=0.5 left at least one element unchanged"
    assert bool((scaled > unit).all()) and bool((halved < unit).all())
    # 0.5 is not an integer: had the Float channel degraded to an int, halved would equal scaled/4 or unit
    assert not torch.equal(halved, unit)
    assert not torch.equal(halved, scaled)


def test_float_attr_preserves_shape_and_dtype():
    a = _exact_operands(SHAPE, torch.float16, 0)
    b = _exact_operands(SHAPE, torch.float16, 17)
    out = torch.ops.pypto.attrs_scale(a, b, scale=2.0)
    assert out.shape == torch.Size(SHAPE)
    assert out.dtype == torch.float16


# ── multi-output ──────────────────────────────────────────────────────────────────────────────────────
def test_multi_output_arity_shapes_and_dtypes():
    # The two outputs differ in BOTH shape and dtype, so a swapped or collapsed output mapping cannot
    # accidentally satisfy this.
    a = _exact_operands(SHAPE, torch.float16, 0)
    b = _exact_operands(SWAP_SHAPE, torch.float32, 5)
    assert a.shape != b.shape and a.dtype != b.dtype

    out = torch.ops.pypto.attrs_swap(a, b)

    assert isinstance(out, (tuple, list)), f"multi-output op returned a {type(out).__name__}"
    assert len(out) == 2
    out0, out1 = out
    assert out0.shape == torch.Size(SWAP_SHAPE) == b.shape
    assert out1.shape == torch.Size(SHAPE) == a.shape
    assert out0.dtype == torch.float32 == b.dtype
    assert out1.dtype == torch.float16 == a.dtype
    assert out0.shape != out1.shape and out0.dtype != out1.dtype

    # The tensors above come from torch_defn, so they alone say nothing about the DECLARATION — and
    # ExportedCustomOp cross-checks the two only on an op's first call. The declared pair is what the deployed
    # kernel's outputs are allocated from, so it is asserted here in its own right.
    assert SWAP_OP._infer_shape_fn(a.shape, b.shape) == (b.shape, a.shape)
    assert SWAP_OP._infer_dtype_fn(a.dtype, b.dtype) == (b.dtype, a.dtype)
    assert (SWAP_OP._infer_shape_fn(a.shape, b.shape), SWAP_OP._infer_dtype_fn(a.dtype, b.dtype)) == (
        (out0.shape, out1.shape), (out0.dtype, out1.dtype)
    )


def test_multi_output_values_follow_the_declared_order():
    # Values, not just metadata: out0 IS b and out1 IS a, elementwise and exactly.
    a = _exact_operands(SHAPE, torch.float16, 0)
    b = _exact_operands(SWAP_SHAPE, torch.float32, 5)
    out0, out1 = torch.ops.pypto.attrs_swap(a, b)
    torch.testing.assert_close(out0, b, rtol=0, atol=0)
    torch.testing.assert_close(out1, a, rtol=0, atol=0)
    assert out0.data_ptr() != b.data_ptr() and out1.data_ptr() != a.data_ptr()


# ── factory over several dtypes ───────────────────────────────────────────────────────────────────────
def test_factory_op_is_declared_in_the_factory_form():
    # Guards the guard: the dtype sweep below is only meaningful if this op really is the factory form.
    assert ADD_FACTORY_OP._create_kernel_fn is _add_factory
    assert ADD_FACTORY_OP._bare_kernel_fn is None
    assert _detect_factory_signature(_add_factory) == "single"


def test_factory_op_is_correct_for_every_dtype():
    # One op declaration, several dtypes: each result must be right AND come back in the dtype it was
    # computed in. Asserting on the covered set keeps "more than one dtype" a property of the run.
    # The declaration is re-checked per dtype because ExportedCustomOp cross-checks torch_defn against
    # infer_shape/infer_dtype only on an op's FIRST call, and it is what the deployed kernel is built from.
    covered = set()
    for dtype in FACTORY_DTYPES:
        a = _exact_operands(SHAPE, dtype, 0)
        b = _exact_operands(SHAPE, dtype, 17)
        out = torch.ops.pypto.attrs_add_factory(a, b)
        assert out.dtype == dtype, f"{dtype}: the op returned {out.dtype}"
        assert out.shape == torch.Size(SHAPE)
        torch.testing.assert_close(out, a + b, rtol=0, atol=0)
        declared_dtype = ADD_FACTORY_OP._infer_dtype_fn(dtype, dtype)
        declared_shape = ADD_FACTORY_OP._infer_shape_fn(torch.Size(SHAPE), torch.Size(SHAPE))
        assert declared_dtype == dtype, f"{dtype}: infer_dtype declares {declared_dtype}"
        assert declared_shape == out.shape
        covered.add(dtype)
    assert covered == set(FACTORY_DTYPES)
    assert len(covered) > 1


def test_attr_name_colliding_with_meta_key_raises():
    """An attr named like a node-meta key ("op_type") would collide on the exported node's attr
    namespace -> rejected at construction, before any registration."""
    with pytest.raises(ValueError, match="collide with the reserved node-meta key"):
        ExportedCustomOp(
            kernel=_scale_factory, infer_shape=_scale_infer_shape, infer_dtype=_scale_infer_dtype,
            attrs=[AttrSpec("op_type", "Float", 1.0)],
        )


def test_infer_shape_trailing_param_must_match_declared_attr():
    """A trailing infer_shape param naming no declared attr is a construction-time error (it would
    otherwise KeyError deep inside torch fake-tensor tracing)."""
    def bad_infer_shape(a_shape: torch.Size, b_shape: torch.Size, scal: float) -> torch.Size:
        return a_shape
    with pytest.raises(ValueError, match=r"match no\s+declared attr name"):
        ExportedCustomOp(
            kernel=_scale_factory, infer_shape=bad_infer_shape, infer_dtype=_scale_infer_dtype,
            attrs=[AttrSpec("scale", "Float", 1.0)],
        )


def test_every_declared_meta_key_is_in_the_reserved_set():
    """Every ``_META_KEY__`` string constant is a member of ``_RESERVED_META_KEYS``.

    The frozenset IS the control: an empty scan (a renamed prefix, a moved constant) fails against a
    non-empty expected set. ``isinstance(v, str)`` keeps a non-string module global from scanning in.
    """
    declared = {v for k, v in vars(node_meta).items() if k.startswith("_META_KEY__") and isinstance(v, str)}
    assert declared == node_meta._RESERVED_META_KEYS
