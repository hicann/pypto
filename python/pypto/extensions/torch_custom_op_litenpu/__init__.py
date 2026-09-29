# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""pypto.extensions.torch_custom_op_litenpu -- pack a pypto kernel into a framework custom-op node.

The ``_litenpu`` suffix names the deploy path, not a soc check: nothing here inspects ``soc_version``,
which is forwarded verbatim to ``pypto.set_codegen_options``. pypto core's ``frontend/parser/entry.py``
requires exactly one compiled kernel object, which LiteNPU guarantees and a partitioned kernel violates.

Held here is ONLY what concerns the pypto kernel SOURCE -- ``common`` maps the modules. Whole-graph
orchestration and the build tooling that makes a deployable op library live with the consuming application.
"""
# Load order matters: common.compile and common.exported_custom_op carry no subpackage dependency, so they
# load fully first.
from .common.compile import *  # noqa: F403, I001 - keep the load order described above
from .common.exported_custom_op import *  # noqa: F403
from .common.attr_spec import AttrSpec  # noqa: F401 - re-export
