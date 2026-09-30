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
"""Tests for the ``check_model`` gate on the export-helpers ``onnx_export_session``.

A graph may carry a non-pypto custom op registered in the **standard** ai.onnx
domain (e.g. a hand-written AscendC op whose plugin uses
``OriginOpType("ai.onnx::N::Op")``). ``onnx.checker`` validates
standard-domain ops against the ONNX schema registry and rejects such an op as
unregistered, so ``onnx_export_session`` must let the caller skip the structural
check while still running the pypto-node consistency check.

These tests drive the post-trace half of ``onnx_export_session`` over a hand-built
model (the torch trace and file I/O are stubbed) so the ``check_model`` branch
is exercised directly, without the heavy torch.onnx tracing machinery. The model
carries no pypto marker; the pypto-node finalize is a no-op on it.
"""
from __future__ import annotations

import exported_custom_op_litenpu.export.onnx_export as export_mod  # export helper (conftest puts tools on sys.path)
import onnx
from onnx import TensorProto, helper
import pytest


def _model_with_unregistered_standard_op() -> onnx.ModelProto:
    # "AddCustom" in the empty (== ai.onnx) domain is NOT a registered standard op, so
    # onnx.checker.check_model rejects it. No pypto marker, so the pypto checks are no-ops.
    inp = helper.make_tensor_value_info("in0", TensorProto.FLOAT, [4])
    out = helper.make_tensor_value_info("out0", TensorProto.FLOAT, [4])
    node = helper.make_node("AddCustom", ["in0", "in0"], ["out0"], name="ac", bias_i=1)
    g = helper.make_graph([node], "g", [inp], [out])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 11)])
    m.ir_version = 7
    return m


def _stub_io(monkeypatch, model: onnx.ModelProto) -> dict:
    """Stub onnx.load / onnx.save so onnx_export_session's post-trace pass runs on *model*."""
    saved: dict = {}
    monkeypatch.setattr(export_mod.onnx, "load", lambda _p: model)
    monkeypatch.setattr(export_mod.onnx, "save", lambda mm, p: saved.update(model=mm, path=p))
    return saved


def test_check_model_true_rejects_unregistered_standard_op(monkeypatch, tmp_path):
    m = _model_with_unregistered_standard_op()
    _stub_io(monkeypatch, m)
    # checker runs on the session's clean __exit__ (the empty body stands in for torch.onnx.export).
    with pytest.raises(onnx.checker.ValidationError):
        with export_mod.onnx_export_session(
            str(tmp_path / "m.onnx"), check_model=True,
        ):
            pass


def test_check_model_false_skips_checker_and_saves(monkeypatch, tmp_path):
    m = _model_with_unregistered_standard_op()
    saved = _stub_io(monkeypatch, m)
    with export_mod.onnx_export_session(
        str(tmp_path / "m.onnx"), check_model=False,
    ):
        pass
    assert saved.get("model") is m  # reached onnx.save -> checker did not abort the export


def test_check_model_false_still_runs_ge_standard_domain_rewrite(monkeypatch, tmp_path):
    # With the checker skipped, the experimental ge_standard_domain rewrite still applies: the
    # empty-domain AddCustom node gets the explicit domain + a matching opset_import (this is the
    # combination a mixed pypto + AscendC graph relies on).
    m = _model_with_unregistered_standard_op()
    saved = _stub_io(monkeypatch, m)
    with export_mod.onnx_export_session(
        str(tmp_path / "m.onnx"), check_model=False,
        ge_standard_domain="ai.onnx",
    ):
        pass
    out = saved["model"]
    domains = {n.op_type: n.domain for n in out.graph.node}
    assert domains["AddCustom"] == "ai.onnx"
    opsets = {imp.domain: imp.version for imp in out.opset_import}
    assert opsets["ai.onnx"] == 11
