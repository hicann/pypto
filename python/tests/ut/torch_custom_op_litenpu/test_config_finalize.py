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
"""Unit tests for config-driven op finalization: ExportedCustomOp(torch_op_qualname=,
onnx_spec=OnnxSymbolicSpec(...)) + finalize_pending_ops().

Construction registers the torch.library op stub (via an EXPLICIT schema built from the derived arity);
finalize_pending_ops() synthesizes + registers the ONNX symbolic. torch.onnx registration is
monkeypatched (spy), but the torch.library op is REALLY registered so the schema synthesis is exercised
for real. Each test uses a UNIQUE torch op name (torch.library ops are process-global and can't be
unregistered).
"""
import pytest
import torch
import torch.library
import torch.onnx

import pypto
from pypto.extensions.torch_custom_op_litenpu import (
    ExportedCustomOp,
    OnnxSymbolicSpec,
    finalize_pending_ops,
    recorded_onnx_opset_floor,
)
from pypto.extensions.torch_custom_op_litenpu.common.exported_custom_op import exporting_scope
from pypto.extensions.torch_custom_op_litenpu.common.finalize import _PENDING_OPS, _reset_export_state
from pypto.extensions.torch_custom_op_litenpu.onnx.export import (
    _ONNX_OPSET_FLOORS,
    _reset_onnx_opset_floors,
)


@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.SIM})
def _kernel(a: pypto.Tensor([...]), b: pypto.Tensor([...]), out: pypto.Tensor([...])):
    pypto.set_vec_tile_shapes(1, 4, 1, 64)
    out.move(a + b)


@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.SIM})
def _kernel2(a: pypto.Tensor([...]), b: pypto.Tensor([...]),
             out0: pypto.Tensor([...]), out1: pypto.Tensor([...])):
    pypto.set_vec_tile_shapes(1, 4, 1, 64)
    out0.move(a + b)
    out1.move(a + b)


def _infer_shape(a: torch.Size, b: torch.Size) -> torch.Size:
    return a


def _infer_dtype(a: torch.dtype, b: torch.dtype) -> torch.dtype:
    return a


def _infer_shape2(a: torch.Size, b: torch.Size) -> tuple[torch.Size, torch.Size]:
    return a, a


def _infer_dtype2(a: torch.dtype, b: torch.dtype) -> tuple[torch.dtype, torch.dtype]:
    return a, a


@pytest.fixture
def spy(monkeypatch):
    _reset_export_state()
    calls = []
    monkeypatch.setattr(torch.onnx, "register_custom_op_symbolic",
                        lambda name, fn, opset: calls.append((name, fn, opset)))
    yield calls
    _reset_export_state()


def _cfg_op(name, opset=12, infer_shape=_infer_shape, infer_dtype=_infer_dtype, kernel=_kernel):
    return ExportedCustomOp(torch_op_qualname=f"pypto::{name}",
                            onnx_spec=OnnxSymbolicSpec(op_type="XCustomOp", opset_version=opset),
                            kernel=kernel, infer_shape=infer_shape, infer_dtype=infer_dtype)


def test_declared_op_synthesizes_torch_op_symbolic_and_floor(spy):
    op = _cfg_op("cfg_add1")
    assert op in _PENDING_OPS and not op._finalized
    finalize_pending_ops()
    assert op._finalized
    # torch op really registered + produces the infer-driven shape/dtype. No torch_defn here, so
    # the impl uses the export-safe shaped-empty stub (exporting_scope); off the export path it would raise.
    assert hasattr(torch.ops.pypto, "cfg_add1")
    x = torch.empty((1, 8, 1, 64), dtype=torch.float16)
    with exporting_scope():
        out = torch.ops.pypto.cfg_add1(x, torch.empty_like(x))
    assert out.shape == x.shape and out.dtype == x.dtype
    # onnx symbolic registered (spy) + floor recorded
    assert "pypto::cfg_add1" in [c[0] for c in spy]
    assert recorded_onnx_opset_floor(["pypto::cfg_add1"]) == 12


def test_finalize_is_idempotent(spy):
    _cfg_op("cfg_add2")
    finalize_pending_ops()
    n = len(spy)
    finalize_pending_ops()
    assert len(spy) == n  # no re-registration on the second finalize


def test_multi_output_synth_returns_tuple(spy):
    # A 2-in/2-out op needs a 4-tensor DIRECT kernel: kernel(*inputs, *outputs).
    _cfg_op("cfg_dual1", infer_shape=_infer_shape2, infer_dtype=_infer_dtype2, kernel=_kernel2)
    finalize_pending_ops()
    x = torch.empty((1, 8, 1, 64), dtype=torch.float16)
    # No torch_defn -> export-safe stub path (exporting_scope); returns the multi-output tuple.
    with exporting_scope():
        out = torch.ops.pypto.cfg_dual1(x, torch.empty_like(x))
    assert isinstance(out, (list, tuple)) and len(out) == 2


def test_onnx_spec_requires_torch_op_qualname():
    with pytest.raises(ValueError):
        ExportedCustomOp(onnx_spec=OnnxSymbolicSpec(op_type="X", opset_version=12),
                         kernel=_kernel, infer_shape=_infer_shape, infer_dtype=_infer_dtype)


def test_kernel_name_cannot_be_set_manually(spy):
    # The constructor takes no free-form node-meta overrides: an author-supplied name is rejected by the
    # keyword-only signature itself.
    with pytest.raises(TypeError):
        ExportedCustomOp(kernel_name="pinned", torch_op_qualname="pypto::cfg_kn2",
                onnx_spec=OnnxSymbolicSpec(op_type="X", opset_version=12),
                kernel=_kernel, infer_shape=_infer_shape, infer_dtype=_infer_dtype)


def test_manual_torch_op_qualname_is_not_overwritten(spy):
    # A hand-written @torch.library op (escape hatch) — construction must skip synthesizing the torch op.
    @torch.library.custom_op("pypto::cfg_manual1", mutates_args=())
    def _manual(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.empty_like(a)

    @torch.library.register_fake("pypto::cfg_manual1")
    def _manual_fake(a, b):
        return torch.empty_like(a)

    # Construction must NOT raise "already registered": the existence probe skips the torch-op synth.
    _cfg_op("cfg_manual1")
    finalize_pending_ops()
    assert hasattr(torch.ops.pypto, "cfg_manual1")


def test_opset_floor_records_and_resets():
    _reset_onnx_opset_floors()
    assert recorded_onnx_opset_floor(["pypto::a"]) is None
    _ONNX_OPSET_FLOORS["pypto::a"] = 12
    _ONNX_OPSET_FLOORS["pypto::b"] = 17
    assert recorded_onnx_opset_floor(["pypto::a", "pypto::b"]) == 17
    _reset_onnx_opset_floors()
    assert recorded_onnx_opset_floor(["pypto::a"]) is None
