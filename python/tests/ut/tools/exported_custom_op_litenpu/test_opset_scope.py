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
"""Example-library tests for the ONNX base-opset floor resolved by resolve_pypto_export_opset.

The floor is discovered from the model itself: one pre-trace forward records the pypto ops the model
invokes and only those ops' recorded floors are maxed, so a higher-opset op elsewhere in the process
cannot bleed in. An explicit opset_version pin overrides the discovery.

Each op uses a UNIQUE op_type + torch_op_qualname (torch.library ops are process-global and cannot be
unregistered). The conftest in this tree puts ``tools`` on sys.path so ``exported_custom_op_litenpu.export`` resolves.
"""

from exported_custom_op_litenpu.export.onnx_export import (
    resolve_pypto_export_opset,  # conftest puts the tools root on sys.path
)
import pytest
import torch
import torch.nn as nn

import pypto
from pypto.extensions.torch_custom_op_litenpu import ExportedCustomOp, OnnxSymbolicSpec, finalize_pending_ops
from pypto.extensions.torch_custom_op_litenpu.common.finalize import _reset_export_state


@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.SIM})
def _kernel(a: pypto.Tensor([...]), out: pypto.Tensor([...])):
    pypto.set_vec_tile_shapes(1, 8, 1, 64)
    out.move(a)


def _infer_shape(a: torch.Size) -> torch.Size:
    return a


def _infer_dtype(a: torch.dtype) -> torch.dtype:
    return a


def _torch_defn(a):
    return a.clone()


def _op(name, opset):
    return ExportedCustomOp(torch_op_qualname=f"pypto::{name}",
                            onnx_spec=OnnxSymbolicSpec(op_type=name, opset_version=opset),
                            kernel=_kernel, infer_shape=_infer_shape, infer_dtype=_infer_dtype,
                            torch_defn=_torch_defn)


@pytest.fixture(autouse=True)
def _clean():
    _reset_export_state()
    yield
    _reset_export_state()


def test_explicit_pin_still_wins():
    _op("scope_pin", 13)
    assert resolve_pypto_export_opset(11) == 11  # the pin overrides discovery (ascendc pattern)


def test_default_base_when_no_onnx_op():
    """A model invoking no pypto op has no floor to max, so the default base opset is returned."""
    _op("scope_default_unused", 17)  # a recorded floor this model does not invoke
    finalize_pending_ops()

    class PlainModel(nn.Module):
        def forward(self, x):
            return x + 1

    inp = torch.rand(1, 8, 1, 64, dtype=torch.float16)
    assert resolve_pypto_export_opset(model=PlainModel(), example_inputs=(inp,)) == 12  # _DEFAULT_BASE_OPSET


def test_auto_scoped_opset_ignores_other_models_floor():
    """Discovery (model + example_inputs) maxes ONLY the floors of the ops the model invokes.

    Two models in ONE process: model A calls a @11 op, model B a DIFFERENT-qualname @13 op. Scoping
    each model reads only its own op's floor — B's 13 does NOT bleed into A. A third model calling TWO ops
    (@11 and @17) returns 17 (the max WITHIN that model).
    """
    _op("auto_model_a", 11)
    _op("auto_model_b", 13)
    _op("auto_model_c1", 11)
    _op("auto_model_c2", 17)
    finalize_pending_ops()  # register the torch ops so the forwards dispatch

    inp = torch.rand(1, 8, 1, 64, dtype=torch.float16)

    class ModelA(nn.Module):
        def forward(self, x):
            return torch.ops.pypto.auto_model_a(x)

    class ModelB(nn.Module):
        def forward(self, x):
            return torch.ops.pypto.auto_model_b(x)

    class ModelC(nn.Module):
        def forward(self, x):
            return torch.ops.pypto.auto_model_c2(torch.ops.pypto.auto_model_c1(x))

    assert resolve_pypto_export_opset(model=ModelA(), example_inputs=(inp,)) == 11  # B's 13 does NOT bleed
    assert resolve_pypto_export_opset(model=ModelB(), example_inputs=(inp,)) == 13
    assert resolve_pypto_export_opset(model=ModelC(), example_inputs=(inp,)) == 17  # max WITHIN the model
