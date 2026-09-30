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
"""The add op: its shape/dtype inference, its CPU reference, and its export config.

Takes the kernel from ``kernel.py`` and declares the op. Importing this module is what registers
``torch.ops.pypto.add`` and enqueues the ONNX wiring for ``finalize_pending_ops()``.
"""
from kernel import add_kernel
import torch

import pypto.extensions.torch_custom_op_litenpu

ONNX_OPSET_VERSION = 12


# the CPU compute for this op, also used as the reference (made available to torch when
# this module is imported).
def add_torch(input0, input1):
    return input0 + input1


# output shape+dtype: drives inference both while tracing and in the deployed kernel.
def add_infer_shape(input0_shape: torch.Size, input1_shape: torch.Size) -> torch.Size:
    return input0_shape


def add_infer_dtype(input0_dtype: torch.dtype, input1_dtype: torch.dtype) -> torch.dtype:
    return input0_dtype


# declare the op and its ONNX wiring as config; pypto generates the torch op + node.
pypto.extensions.torch_custom_op_litenpu.ExportedCustomOp(
    kernel=add_kernel,
    infer_shape=add_infer_shape,
    infer_dtype=add_infer_dtype,
    torch_defn=add_torch,
    # op_type names the COMPUTE in the ONNX graph (may repeat across demos); torch_op_qualname is
    # the unique per-op torch-registry key.
    torch_op_qualname="pypto::add",
    onnx_spec=pypto.extensions.torch_custom_op_litenpu.OnnxSymbolicSpec(
        op_type="Add", opset_version=ONNX_OPSET_VERSION
    ),
)
