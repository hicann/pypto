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
"""Unit tests for torch_defn + auto-registration ("register when ready") + the run/export impl branches.

The torch op is REALLY registered (auto, once the initializer attaches qualname + both infers) so the
schema synthesis + call-time dispatch are exercised for real. torch.library ops are process-global and
can't be unregistered, so every test uses a UNIQUE torch_op_qualname (never collides in this one process).
"""
import itertools

import pytest
import torch

import pypto
import pypto.extensions.torch_custom_op_litenpu as custom_op_pkg
from pypto.extensions.torch_custom_op_litenpu import ExportedCustomOp
from pypto.extensions.torch_custom_op_litenpu.common.torch_op import (
    _QUALNAME_TO_OP,
    _is_exporting,
    exporting_scope,
)

_counter = itertools.count()


def _uniq(prefix="td"):
    """A fresh torch_op_qualname (torch ops can't be deregistered — never reuse a name across tests)."""
    return f"pypto::{prefix}_{next(_counter)}"


@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.SIM})
def _kernel(a: pypto.Tensor([...]), b: pypto.Tensor([...]), out: pypto.Tensor([...])):
    pypto.set_vec_tile_shapes(1, 4, 1, 64)
    out.move(a + b)


def _infer_shape(a: torch.Size, b: torch.Size) -> torch.Size:
    return a


def _infer_dtype(a: torch.dtype, b: torch.dtype) -> torch.dtype:
    return a


def _make_op(name, *, defn=None):
    return ExportedCustomOp(
        torch_op_qualname=name,
        kernel=_kernel, infer_shape=_infer_shape, infer_dtype=_infer_dtype,
        torch_defn=defn,
    )


def _ns_name(qualname):
    ns, _, name = qualname.partition("::")
    return ns, name


def test_torch_defn_attach_stores():
    name = _uniq()

    def defn(a, b):
        return a + b

    op = _make_op(name, defn=defn)
    assert op._torch_defn_fn is defn


def test_torch_defn_arity_mismatch_raises():
    name = _uniq()
    with pytest.raises(ValueError, match="params but infer_dtype declares"):
        ExportedCustomOp(
            torch_op_qualname=name,
            kernel=_kernel, infer_shape=_infer_shape, infer_dtype=_infer_dtype,
            torch_defn=lambda a: a,  # 1 param vs infer_shape's 2 inputs
        )


def test_auto_register_when_ready():
    name = _uniq()
    ns, short = _ns_name(name)

    def defn(a, b):
        return a + b

    op = _make_op(name, defn=defn)
    # registered at construction (qualname + both infers present), no finalize/session needed
    assert hasattr(getattr(torch.ops, ns), short)
    assert _QUALNAME_TO_OP[name] is op


def test_impl_torch_defn_runs_real_compute():
    name = _uniq()
    ns, short = _ns_name(name)

    def defn(a, b):
        return a + b

    _make_op(name, defn=defn)
    x = torch.rand((1, 8, 1, 64), dtype=torch.float16)
    y = torch.rand((1, 8, 1, 64), dtype=torch.float16)
    out = getattr(getattr(torch.ops, ns), short)(x, y)
    assert torch.equal(out, x + y)


def test_impl_no_torch_defn_exporting_returns_shaped_empty():
    name = _uniq()
    ns, short = _ns_name(name)
    _make_op(name, defn=None)  # no torch_defn
    x = torch.rand((1, 8, 1, 64), dtype=torch.float16)
    y = torch.rand((1, 8, 1, 64), dtype=torch.float16)
    with exporting_scope():
        out = getattr(getattr(torch.ops, ns), short)(x, y)
    assert tuple(out.shape) == tuple(x.shape) and out.dtype == x.dtype


def test_impl_no_torch_defn_run_raises():
    name = _uniq()
    ns, short = _ns_name(name)
    _make_op(name, defn=None)
    x = torch.rand((1, 8, 1, 64), dtype=torch.float16)
    y = torch.rand((1, 8, 1, 64), dtype=torch.float16)
    assert not _is_exporting()
    with pytest.raises(RuntimeError, match="no torch_defn"):
        getattr(getattr(torch.ops, ns), short)(x, y)


def test_fake_stays_shaped_empty_and_never_raises():
    name = _uniq()
    ns, short = _ns_name(name)
    _make_op(name, defn=None)  # no torch_defn -> impl would raise on a real run, but the fake must not
    # Register-fake path: run under FakeTensorMode so the fake (not the impl) is exercised.
    from torch._subclasses.fake_tensor import FakeTensorMode

    with FakeTensorMode() as fm:
        x = fm.from_tensor(torch.empty((1, 8, 1, 64), dtype=torch.float16))
        y = fm.from_tensor(torch.empty((1, 8, 1, 64), dtype=torch.float16))
        out = getattr(getattr(torch.ops, ns), short)(x, y)
    assert tuple(out.shape) == (1, 8, 1, 64) and out.dtype == torch.float16


def test_output_shape_dtype_assert_catches_author_error():
    name = _uniq()
    ns, short = _ns_name(name)

    def bad_defn(a, b):
        return (a + b).to(torch.float32)  # wrong dtype vs infer_dtype (returns a's dtype)

    _make_op(name, defn=bad_defn)
    x = torch.rand((1, 8, 1, 64), dtype=torch.float16)
    y = torch.rand((1, 8, 1, 64), dtype=torch.float16)
    with pytest.raises(RuntimeError, match="torch_defn output"):
        getattr(getattr(torch.ops, ns), short)(x, y)


def test_qualname_reuse_resolves_to_last_declared_and_warns(caplog):
    name = _uniq()
    ns, short = _ns_name(name)

    def defn_a(a, b):
        return a + b

    def defn_b(a, b):
        return a - b

    _op_a = _make_op(name, defn=defn_a)
    x = torch.rand((1, 8, 1, 64), dtype=torch.float16)
    y = torch.rand((1, 8, 1, 64), dtype=torch.float16)
    out_a = getattr(getattr(torch.ops, ns), short)(x, y)
    assert torch.equal(out_a, x + y)

    # A DIFFERENT op reusing the same qualname -> warning, last-declared wins for run dispatch.
    with caplog.at_level("WARNING"):
        op_b = _make_op(name, defn=defn_b)
    assert any("re-declared by a different op" in r.message for r in caplog.records)
    assert _QUALNAME_TO_OP[name] is op_b
    out_b = getattr(getattr(torch.ops, ns), short)(x, y)
    assert torch.equal(out_b, x - y)  # resolves to the last-declared op, not silently shadowed by op_a


def test_exporting_scope_resolves_from_public_package():
    """The public path stays: ``pypto.extensions.torch_custom_op_litenpu.exporting_scope``
    resolves + is a context manager."""
    assert hasattr(custom_op_pkg, "exporting_scope")
    with custom_op_pkg.exporting_scope():
        assert _is_exporting()
    assert not _is_exporting()
