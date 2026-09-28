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
"""Unit tests for the ``torch.library`` registration machinery in ``common/torch_op.py``.

``torch_op`` declares its caller as a duck-typed protocol -- ``_torch_op_qualname``, ``_infer_shape_fn``,
``_infer_dtype_fn``, ``_torch_defn_fn``, ``_attr_specs``, ``_assert_torch_defn_output`` -- so these tests
drive :func:`_synthesize_torch_op_qualname` with a stub satisfying it, exercising registration, the
synthesized schema and the call-time impl/fake dispatch through the real ``torch.library``.

A declared attr is a local duck carrying only ``.name``/``.type``: those are the two fields ``torch_op`` reads.

``torch.library`` ops are process-global and cannot be deregistered, so every qualname comes from
:func:`_uniq` and none is reused across tests except where a test is about reuse.
"""
import itertools

import pytest
import torch

from pypto.extensions.torch_custom_op_litenpu.common.torch_op import (
    _QUALNAME_TO_OP,
    _op_ns_name,
    _synthesize_torch_op_qualname,
    exporting_scope,
)

_counter = itertools.count()


def _uniq(prefix="reg"):
    """A fresh torch_op_qualname (torch ops can't be deregistered -- never reuse a name across tests)."""
    return f"pypto::{prefix}_{next(_counter)}"


def _ns_name(qualname):
    ns, _, name = qualname.partition("::")
    return ns, name


def _torch_op(qualname):
    """The live ``torch.ops.<ns>.<op_name>`` overload packet for *qualname*."""
    ns, name = _ns_name(qualname)
    return getattr(getattr(torch.ops, ns), name)


class _Attr:
    """A declared compute attr, duck-typed: ``torch_op`` reads only ``.name`` and ``.type``."""

    def __init__(self, name: str, attr_type: str):
        self.name = name
        self.type = attr_type


class _StubOp:
    """The minimal object satisfying ``torch_op``'s duck-typed caller protocol."""

    def __init__(self, qualname, infer_shape, infer_dtype, *, torch_defn=None, attrs=()):
        self._torch_op_qualname = qualname
        self._infer_shape_fn = infer_shape
        self._infer_dtype_fn = infer_dtype
        self._torch_defn_fn = torch_defn
        self._attr_specs = tuple(attrs)
        self.asserted = []  # (out, inputs, attr_values) per _assert_torch_defn_output call, in order

    def _assert_torch_defn_output(self, out, inputs, attr_values=()):
        self.asserted.append((out, tuple(inputs), tuple(attr_values)))


def _infer_shape(a: torch.Size, b: torch.Size) -> torch.Size:
    return a


def _infer_dtype(a: torch.dtype, b: torch.dtype) -> torch.dtype:
    return a


def _infer_shape1(a: torch.Size) -> torch.Size:
    return a


def _infer_dtype1(a: torch.dtype) -> torch.dtype:
    return a


def _crop_infer_shape(x: torch.Size, crop: int) -> torch.Size:
    return torch.Size([x[0], crop])


def _crop_infer_dtype(x: torch.dtype) -> torch.dtype:
    return x


def _swap_infer_shape(a: torch.Size, b: torch.Size) -> tuple[torch.Size, torch.Size]:
    return (b, a)


def _swap_infer_dtype(a: torch.dtype, b: torch.dtype) -> tuple[torch.dtype, torch.dtype]:
    return (b, a)


def test_registers_and_dispatches_through_torch_ops():
    name = _uniq()
    seen = []

    def defn(a, b):
        seen.append((a, b))
        return a + b

    stub = _StubOp(name, _infer_shape, _infer_dtype, torch_defn=defn)
    _synthesize_torch_op_qualname(stub)

    assert _QUALNAME_TO_OP[name] is stub
    x = torch.rand(2, 3)
    y = torch.rand(2, 3)
    out = _torch_op(name)(x, y)

    assert torch.equal(out, x + y)  # the real torch op dispatched into torch_defn
    assert len(seen) == 1
    assert len(stub.asserted) == 1  # ... and routed the result through the output assert
    got_out, got_inputs, got_attrs = stub.asserted[0]
    assert torch.equal(got_out, x + y) and len(got_inputs) == 2 and not got_attrs


def test_no_torch_defn_returns_a_shaped_stub_while_exporting():
    name = _uniq()
    stub = _StubOp(name, _infer_shape, _infer_dtype)  # no torch_defn
    _synthesize_torch_op_qualname(stub)

    x = torch.rand(2, 3, dtype=torch.float16)
    y = torch.rand(4, 5, dtype=torch.float32)  # infer_shape/infer_dtype report a's, never b's
    with exporting_scope():
        out = _torch_op(name)(x, y)

    assert tuple(out.shape) == (2, 3) and out.dtype == torch.float16
    assert not stub.asserted  # no torch_defn ran, so there was no output to check


def test_no_torch_defn_raises_off_the_export_path():
    name = _uniq()
    _synthesize_torch_op_qualname(_StubOp(name, _infer_shape, _infer_dtype))

    x = torch.rand(2, 3)
    y = torch.rand(2, 3)
    with pytest.raises(RuntimeError, match="no torch_defn"):
        _torch_op(name)(x, y)


def test_fake_tensor_mode_uses_the_registered_fake():
    from torch._subclasses.fake_tensor import FakeTensorMode

    name = _uniq()
    # No torch_defn: the impl would raise on a real call, so a shaped result proves the FAKE ran.
    _synthesize_torch_op_qualname(_StubOp(name, _infer_shape, _infer_dtype))

    with FakeTensorMode() as fm:
        x = fm.from_tensor(torch.empty(2, 3, dtype=torch.float16))
        y = fm.from_tensor(torch.empty(2, 3, dtype=torch.float16))
        out = _torch_op(name)(x, y)

    assert tuple(out.shape) == (2, 3) and out.dtype == torch.float16


@pytest.mark.parametrize("attr_type,schema_token,value", [
    ("Int", "int", 3),
    ("Float", "float", 1.5),
    ("String", "str", "sum"),
    ("ListInt", "int[]", [1, 2]),
], ids=["Int", "Float", "String", "ListInt"])
def test_declared_attr_becomes_a_trailing_scalar_operand(attr_type, schema_token, value):
    name = _uniq()
    seen = []

    def defn(a, attr):
        seen.append(attr)
        return a * 2

    stub = _StubOp(name, _infer_shape1, _infer_dtype1, torch_defn=defn, attrs=[_Attr("bias", attr_type)])
    _synthesize_torch_op_qualname(stub)

    # A declared attr is a TRAILING operand of the synthesized schema, after the tensor inputs, typed per
    # _ATTR_SCHEMA_TYPE -- infer_shape/infer_dtype still take only the tensors.
    assert f"(Tensor x0, {schema_token} bias)" in str(_torch_op(name).default._schema)

    x = torch.rand(2, 3)
    out = _torch_op(name)(x, value)

    assert torch.equal(out, x * 2)
    assert len(seen) == 1  # torch_defn received the attr as its trailing argument
    assert len(stub.asserted[0][1]) == 1 and len(stub.asserted[0][2]) == 1  # impl split tensors from attrs


def test_shape_affecting_attr_is_routed_into_infer_shape_by_name():
    name = _uniq()
    # Two declared attrs, only the SECOND of which is a trailing infer_shape param. A positional hand-off
    # would feed infer_shape `bias`; the by-name lookup feeds it `crop`.
    stub = _StubOp(name, _crop_infer_shape, _crop_infer_dtype,
                   attrs=[_Attr("bias", "Int"), _Attr("crop", "Int")])
    _synthesize_torch_op_qualname(stub)

    x = torch.rand(4, 9)
    with exporting_scope():
        out = _torch_op(name)(x, 5, 2)  # bias=5 (shape-invariant), crop=2 (shape-affecting)

    assert tuple(out.shape) == (4, 2)  # crop, not bias, reached infer_shape


def test_multi_output_schema_returns_a_tuple():
    name = _uniq()

    def defn(a, b):
        return (b.clone(), a.clone())

    _synthesize_torch_op_qualname(_StubOp(name, _swap_infer_shape, _swap_infer_dtype, torch_defn=defn))

    assert "-> (Tensor, Tensor)" in str(_torch_op(name).default._schema)
    x = torch.rand(2, 3, dtype=torch.float32)
    y = torch.rand(4, 5, dtype=torch.float16)
    out = _torch_op(name)(x, y)

    assert len(out) == 2
    assert tuple(out[0].shape) == (4, 5) and out[0].dtype == torch.float16
    assert tuple(out[1].shape) == (2, 3) and out[1].dtype == torch.float32


def test_qualname_reuse_warns_and_the_last_declaration_wins(caplog):
    name = _uniq()
    first = _StubOp(name, _infer_shape, _infer_dtype, torch_defn=lambda a, b: a + b)
    _synthesize_torch_op_qualname(first)

    x = torch.rand(2, 3)
    y = torch.rand(2, 3)
    assert torch.equal(_torch_op(name)(x, y), x + y)
    assert _QUALNAME_TO_OP[name] is first

    second = _StubOp(name, _infer_shape, _infer_dtype, torch_defn=lambda a, b: a - b)
    with caplog.at_level("WARNING"):
        _synthesize_torch_op_qualname(second)

    # Match the stable prefix of the warning, not its full text.
    assert "re-declared by a different op" in caplog.text
    assert _QUALNAME_TO_OP[name] is second
    assert torch.equal(_torch_op(name)(x, y), x - y)  # resolved at CALL time, not captured at registration


def test_an_escape_hatch_qualname_is_left_alone():
    name = _uniq()

    @torch.library.custom_op(name, mutates_args=())
    def _manual(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return a * 2

    @torch.library.register_fake(name)
    def _manual_fake(a, b):
        return torch.empty_like(a)

    _synthesize_torch_op_qualname(_StubOp(name, _infer_shape, _infer_dtype, torch_defn=lambda a, b: a + b))

    assert name not in _QUALNAME_TO_OP
    x = torch.rand(2, 3)
    y = torch.rand(2, 3)
    assert torch.equal(_torch_op(name)(x, y), x * 2)  # the hand-written impl, not the stub's torch_defn


@pytest.mark.parametrize("bad", ["no_separator", "::op_name", "namespace::", ""],
                         ids=["no_separator", "empty_namespace", "empty_name", "empty_string"])
def test_op_ns_name_rejects_a_bad_qualname(bad):
    with pytest.raises(ValueError, match="namespace"):
        _op_ns_name(bad)
    # The same guard runs at registration, ahead of any bookkeeping.
    with pytest.raises(ValueError, match="namespace"):
        _synthesize_torch_op_qualname(_StubOp(bad, _infer_shape, _infer_dtype))
    assert bad not in _QUALNAME_TO_OP


def test_infer_dtype_must_declare_at_least_one_input():
    name = _uniq()
    ns, short = _ns_name(name)

    def _no_inputs_infer_shape() -> torch.Size:
        return torch.Size([1])

    def _no_inputs_infer_dtype() -> torch.dtype:
        return torch.float32

    # infer_dtype's parameter count IS the synthesized op's tensor-input arity, so a zero-param one would
    # register an op with no inputs and leave _shape_stub indexing inputs[0].
    with pytest.raises(ValueError, match=">=1 input"):
        _synthesize_torch_op_qualname(_StubOp(name, _no_inputs_infer_shape, _no_inputs_infer_dtype))
    assert not hasattr(getattr(torch.ops, ns, None), short)  # nothing was registered
