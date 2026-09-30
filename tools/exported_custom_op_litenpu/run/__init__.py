# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Shared run CLI + thin whole-model run helpers for the custom-op export demos.

Holds :mod:`run.demo_run` — ``add_run_args(parser)`` adds the ``--device`` / ``--soc_version``
flags to a demo's ``run_demo.py``, ``validate_args(args)`` enforces the valid CLI combos (resolving
nothing), and thin helpers
(``scenario_label``, ``npu_device_id``, ``move_inputs_to_npu``, ``enable_eslmodel``, ``print_outputs``, plus
the re-exported ``pypto_run_context``) keep the per-demo run path DRY. Each demo applies the args→run mapping
VISIBLY in its own ``run_demo`` (``--device=cpu`` runs each pypto op's ``torch_defn`` — or, with
``--soc_version``, the env-level NPU simulator; ``--device=npu`` launches each op's real pypto kernel on
device). Kept in this shared glue (not in pypto) so per-demo files stay thin and don't duplicate argparse.
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
