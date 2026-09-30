# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""ONNX export session that produces a pypto custom-op artifact from a traced model.

Holds the ONNX export driver (:mod:`export.onnx_export`): it wraps the framework's own export call
(``torch.onnx.export``), finalizes the config-declared ops, and finalizes the saved artifact. Composed
by the demos; not part of ``pypto.extensions.torch_custom_op_litenpu``, which packs only the kernel source.
"""

import logging
import sys

# Console output for this package: its own stdout handler at INFO, no root propagation.
_logger = logging.getLogger(__name__)
if not _logger.handlers:
    _console = logging.StreamHandler(sys.stdout)
    _console.setFormatter(logging.Formatter("%(message)s"))
    _logger.addHandler(_console)
_logger.setLevel(logging.INFO)
_logger.propagate = False
