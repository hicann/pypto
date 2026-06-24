#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
atexit cleanup for --forked subprocess: flush NPU device queues + release cached memory
before the process exits, preventing residual device state from leaking into the next test case.
"""
import atexit
import logging

logger = logging.getLogger(__name__)


def _cleanup_npu_device():
    """Wait for all pending asynchronous operations and release cached device memory."""
    synchronized_ok = False
    cache_emptied_ok = False
    try:
        import torch_npu
        torch_npu.npu.synchronize()
        synchronized_ok = True
    except Exception as e:
        logger.warning("NPU synchronize failed: %s", e)
    try:
        import torch
        torch.npu.empty_cache()
        cache_emptied_ok = True
    except Exception as e:
        logger.warning("NPU empty_cache failed: %s", e)

    if synchronized_ok and cache_emptied_ok:
        logger.debug("NPU device cleanup: synchronize + empty_cache completed.")
    elif synchronized_ok:
        logger.debug("NPU device cleanup: synchronize completed (empty_cache skipped).")
    elif cache_emptied_ok:
        logger.debug("NPU device cleanup: empty_cache completed (synchronize skipped).")


atexit.register(_cleanup_npu_device)
