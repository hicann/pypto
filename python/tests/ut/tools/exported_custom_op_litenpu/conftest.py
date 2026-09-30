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
"""Put the repo's ``tools`` dir on ``sys.path`` so these tests import ``exported_custom_op_litenpu.common.* /
exported_custom_op_litenpu.export.* / exported_custom_op_litenpu.deploy.*``.

Sitting under ``python/tests/ut`` also satisfies the repo-root conftest's ``"/tests/ut/" in path``
check, so these tests run without an NPU.
"""
from pathlib import Path
import sys

_TOOLS_ROOT = Path(__file__).resolve().parents[5] / "tools"
if str(_TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(_TOOLS_ROOT))
