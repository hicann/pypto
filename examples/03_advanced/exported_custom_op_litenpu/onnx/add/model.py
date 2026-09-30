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
"""The graph the add demo exports and runs: the nn.Module plus the shape its tensors carry.

``forward`` calls ``torch.ops.pypto.add``, so this module imports ``op`` — that import is what
registers the torch op. Shared by ``run_demo.py`` and ``export_demo.py``.
"""
import op  # noqa: F401 -- imported for its side effect: declaring the op registers its torch.ops entry
import torch
import torch.nn as nn

SHAPE = (1, 8, 1, 64)  # 512 elems — kept small so the SIM run stays fast; tile (1, 4, 1, 64)


# the graph traced for export and executed by run_demo.py.
class CustomModel(nn.Module):
    def forward(self, input0, input1):
        return torch.ops.pypto.add(input0, input1)
