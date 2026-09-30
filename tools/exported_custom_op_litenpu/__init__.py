# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Shared glue library the custom-op export examples import.

Holds the non-runnable module folders: ``common`` (pypto custom-op node discovery), ``export``
(per-framework export sessions that produce the artifact), ``run`` (the shared demo run/CLI helpers),
and ``deploy`` (build + place the ``.so``). Not part of the ``pypto`` package — it lives under
``tools`` so the demos own the export call while reusing this glue.
"""
