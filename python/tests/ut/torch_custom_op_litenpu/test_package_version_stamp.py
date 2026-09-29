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
"""Tests that ``ExportedCustomOp`` stamps the exporting pypto release onto the node meta."""
import importlib.metadata

import torch

import pypto
from pypto.extensions.torch_custom_op_litenpu import ExportedCustomOp
from pypto.extensions.torch_custom_op_litenpu.common.node_meta import (
    _META_KEY__PYPTO_PACKAGE_VERSION,
    _installed_pypto_version,
)


@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.SIM})
def _kernel(a: pypto.Tensor([...]), b: pypto.Tensor([...]), out: pypto.Tensor([...])):
    pypto.set_vec_tile_shapes(1, 4, 1, 64)
    out.move(a + b)


def _infer_shape(a: torch.Size, b: torch.Size) -> torch.Size:
    return a


def _infer_dtype(a: torch.dtype, b: torch.dtype) -> torch.dtype:
    return a


_FNS = dict(kernel=_kernel, infer_shape=_infer_shape, infer_dtype=_infer_dtype)


def test_stamped_package_version_matches_the_installed_distribution():
    # Duplicated on purpose: _LOCAL_PYPTO_VERSION IS _installed_pypto_version()'s return value, so
    # calling the helper here would compare it with itself and pass even when the helper is wrong.
    op = ExportedCustomOp(**_FNS)
    try:
        expected = importlib.metadata.version("pypto")
    except importlib.metadata.PackageNotFoundError:
        expected = "unknown"
    assert op._node_meta[_META_KEY__PYPTO_PACKAGE_VERSION] == expected


def test_installed_pypto_version_falls_back_to_unknown(monkeypatch):
    def _raise(_name):
        raise importlib.metadata.PackageNotFoundError("pypto")

    monkeypatch.setattr(importlib.metadata, "version", _raise)
    assert _installed_pypto_version() == "unknown"
