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
"""Unit tests for the ONNX symbolic spec module (``onnx.spec``).

Covers the spec defaults, the op-type qualification, the node-meta encoding and the recorded
opset floor. The module under test imports only ``dataclasses``, ``typing`` and ``common.authoring``,
so nothing here declares an op or packs a kernel.
"""
from pypto.extensions.torch_custom_op_litenpu.onnx.spec import OnnxSymbolicSpec


def test_spec_defaults_come_from_the_pypto_domain():
    s = OnnxSymbolicSpec(op_type="Add")
    assert (s.op_type, s.opset_version, s.domain, s.domain_opset_version) == ("Add", 12, "pypto", 1)


def test_spec_fields_are_overridable():
    s = OnnxSymbolicSpec(op_type="X", opset_version=17, domain="d", domain_opset_version=3)
    assert (s.opset_version, s.domain, s.domain_opset_version) == (17, "d", 3)
