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
"""Tests for the run-context dispatch of an op declaring compute attributes.

* ``torch_op._impl`` under a run context hands ``run_op`` only the tensor inputs and the attrs as the
  stringified ``{name: str(value)}`` dict the deployed kernel receives.
* ``_build_compile_entry`` sizes a shape-affecting-attr op's ``CompileEntry`` by its TENSOR count (from
  ``infer_dtype``), not by ``infer_shape``'s parameter count.
"""
import torch

import pypto
from pypto.extensions.torch_custom_op_litenpu import AttrSpec, ExportedCustomOp


# ── module-scope authoring fns (kernel_snippet inspects their source, so they must be importable) ──
def _bias_factory(shapes, dtypes, attrs, soc_version, run_mode=pypto.RunMode.SIM):
    dtype = dtypes[0]
    bias = int(attrs["bias"])  # attrs arrive as strings

    @pypto.frontend.jit(codegen_options={"soc_version": soc_version},
                        runtime_options={"run_mode": run_mode})
    def add_bias_inner(a: pypto.Tensor([...], dtype), b: pypto.Tensor([...], dtype), out: pypto.Tensor([...])):
        pypto.set_vec_tile_shapes(1, 4, 1, 64)
        out.move(a + b + bias)
    return add_bias_inner


def _bias_infer_shape(a: torch.Size, b: torch.Size) -> torch.Size:
    return a


def _bias_infer_dtype(a: torch.dtype, b: torch.dtype) -> torch.dtype:
    return a


def _bias_torch(a, b, bias):
    return a + b + bias


def _crop_factory(shapes, dtypes, attrs, soc_version, run_mode=pypto.RunMode.SIM):
    import json as _json
    dtype = dtypes[0]
    shape = shapes[0]
    h, w = _json.loads(attrs["crop_size"])

    @pypto.frontend.jit(codegen_options={"soc_version": soc_version},
                        runtime_options={"run_mode": run_mode})
    def crop_inner(x: pypto.Tensor([...], dtype), out: pypto.Tensor([...], dtype)):
        pypto.set_vec_tile_shapes(shape[0], shape[1], h, w)
        out[:] = x[:, :, :h, :w] * 2.0  # step-1 slice (supported) + real arithmetic
    return crop_inner


def _single_infer_dtype(x: torch.dtype) -> torch.dtype:
    return x


def _crop_infer_shape(x: torch.Size, crop_size: list[int]) -> torch.Size:
    return torch.Size([x[0], x[1], crop_size[0], crop_size[1]])


def _crop_torch(x, crop_size):
    return x[:, :, :crop_size[0], :crop_size[1]] * 2.0


def test_impl_run_context_builds_stringified_attr_dict_and_passes_tensors_only(monkeypatch):
    # The run-context branch of torch_op._impl must (a) build the compile attrs dict as
    # {spec.name: str(value)} (mirrors the deployed C++ BuildAttrsDict STRING contract) and (b) hand
    # run_op only the tensor inputs (the trailing attr scalar is split off). Stub run_op to capture the
    # call so the node->kernel string round-trip is covered off-box.
    from pypto.extensions.torch_custom_op_litenpu.common import run as run_mod

    op = ExportedCustomOp(kernel=_bias_factory, infer_shape=_bias_infer_shape, infer_dtype=_bias_infer_dtype,
                          torch_defn=_bias_torch, torch_op_qualname="pypto::attr_impl_run1",
                          attrs=[AttrSpec("bias", "Int", 0)])
    assert op._attr_specs == (AttrSpec("bias", "Int", 0),)

    captured = {}

    def _fake_run_op(cur, inputs, *, run_mode, soc_version=None, attrs=None):
        captured["inputs"] = inputs
        captured["attrs"] = attrs
        return torch.empty((1, 8, 1, 64), dtype=torch.float16)

    monkeypatch.setattr(run_mod, "run_op", _fake_run_op)

    a = torch.rand((1, 8, 1, 64), dtype=torch.float16)
    b = torch.rand((1, 8, 1, 64), dtype=torch.float16)
    with run_mod.pypto_run_context(run_mode=pypto.RunMode.SIM, soc_version="Ascend910B3"):
        torch.ops.pypto.attr_impl_run1(a, b, bias=3)

    assert captured["attrs"] == {"bias": "3"}  # int 3 -> string "3", keyed by the AttrSpec name
    # Only the two tensor inputs reach run_op — the trailing attr scalar is NOT in the operand list.
    assert len(captured["inputs"]) == 2
    assert all(isinstance(t, torch.Tensor) for t in captured["inputs"])


def test_run_path_compile_entry_num_inputs_is_tensor_count_not_infer_shape_params():
    # Run-path regression: a shape-affecting-attr op's CompileEntry must use the TENSOR count (from
    # infer_dtype), NOT infer_shape's param count. crop infer_shape has 2 params (x_shape, crop_size) but
    # only 1 tensor input, so _build_compile_entry must yield _num_inputs == 1 (else the run-path build_jit
    # "got 1 shapes / expected num_inputs=2" failure the NPU run hit).
    from pypto.extensions.torch_custom_op_litenpu.common.run import _build_compile_entry

    op = ExportedCustomOp(kernel=_crop_factory, infer_shape=_crop_infer_shape, infer_dtype=_single_infer_dtype,
                          torch_defn=_crop_torch, torch_op_qualname="pypto::attr_run_entry_shape",
                          attrs=[AttrSpec("crop_size", "ListInt", [4, 4])])
    assert _build_compile_entry(op)._num_inputs == 1  # tensor count, NOT the 2 infer_shape params

    # Shape-invariant op is unaffected: infer_dtype-count == infer_shape-count == tensor count.
    op2 = ExportedCustomOp(kernel=_bias_factory, infer_shape=_bias_infer_shape, infer_dtype=_bias_infer_dtype,
                           torch_defn=_bias_torch, torch_op_qualname="pypto::attr_run_entry_invariant",
                           attrs=[AttrSpec("bias", "Int", 0)])
    assert _build_compile_entry(op2)._num_inputs == 2
