# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Framework-agnostic kernel-source machinery.

Holds the op authoring surface (``exported_custom_op``: the ``ExportedCustomOp`` class and the shared
export closures), the declare-then-finalize registry (``finalize``), the node-meta schema (``node_meta``), the
self-contained kernel compile-snippet packing (``kernel_snippet``), authoring constants (``authoring``),
source-manipulation utils (``source_utils``), and the runtime compile contract (``compile``).
Nothing here depends on a specific export framework.
"""
