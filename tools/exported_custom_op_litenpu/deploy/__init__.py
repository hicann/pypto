# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Deploy step: build a compiled ``libcust_opapi.so`` op library from a pypto custom-op artifact.

The python->C++ codegen + CMake build plugin: it reconstructs each op's kernel from the embedded compile
snippet and compiles the generated executor/plugin TUs (with the bundled C++ vendors under ``vendors/``)
into a placeable ``.so``. The demos reach ``build_so_from_model`` and the op-package install step straight
off this package.
"""
import logging
import sys

from .build import *  # noqa: F403
from .setup import *  # noqa: F403

# Console output for this package: its own stdout handler at INFO, no root propagation.
_logger = logging.getLogger(__name__)
if not _logger.handlers:
    _console = logging.StreamHandler(sys.stdout)
    _console.setFormatter(logging.Formatter("%(message)s"))
    _logger.addHandler(_console)
_logger.setLevel(logging.INFO)
_logger.propagate = False
