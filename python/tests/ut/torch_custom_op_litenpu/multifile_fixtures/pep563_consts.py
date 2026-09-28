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
"""PEP-563 (``from __future__ import annotations``) tensor-param helpers whose tensor annotation
references module constants, so the annotation consts must be captured against this module's
globals and rewritten, and the annotation emitted LIVE (de-stringized by construction).

Also holds the residual case: a tensor annotation using an un-packable alias raises.
"""
from __future__ import annotations

import pypto

SHAPE = (1, 4, 1, 64)
DTYPE = pypto.DT_FP16

# An alias to pypto.Tensor bound under a NON-pre-import name: a tensor annotation using it is
# load-bearing and un-emittable, so it raises (the residual annotation guard).
MyTensor = pypto.Tensor


def const_tensor_helper(x: pypto.Tensor(SHAPE, DTYPE)):
    """SHAPE captured (const), DTYPE captured (dtype channel), annotation rewritten to the
    emitted names, live. Tensor-param count == 1."""
    return x


def alias_tensor_helper(x: MyTensor(SHAPE, DTYPE)):
    """Residual: a tensor annotation whose root ``MyTensor`` is not a pre-import raises."""
    return x
