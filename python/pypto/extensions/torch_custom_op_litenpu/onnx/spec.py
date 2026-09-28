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
"""The ONNX symbolic config an author declares for a pypto op; the sibling ``export`` module applies it."""
from dataclasses import dataclass

from ..common.authoring import _DEFAULT_DOMAIN, _DEFAULT_DOMAIN_OPSET_VERSION

__all__ = ("OnnxSymbolicSpec",)


@dataclass
class OnnxSymbolicSpec:
    """Config for a pypto op's ONNX symbolic, from which the symbolic is synthesized and registered.

    ``op_type`` is the graph node type (GE lookup key ``<domain>::<domain_opset_version>::<op_type>``);
    ``opset_version`` is the base ai.onnx floor recorded for the op (see ``recorded_onnx_opset_floor``).
    ``domain``/``domain_opset_version`` default to the pypto domain.
    """
    op_type: str
    opset_version: int = 12
    domain: str = _DEFAULT_DOMAIN
    domain_opset_version: int = _DEFAULT_DOMAIN_OPSET_VERSION
