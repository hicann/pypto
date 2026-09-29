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
"""The declare-then-finalize registry: defers an op's framework-symbolic synthesis.

An op that carries ``torch_op_qualname`` plus an ``*_spec`` enqueues itself here from
``ExportedCustomOp.__init__``, so its symbolic is synthesized later rather than at import time -- importing
an op module must not pull in ``torch.onnx``. Pending ops are handled duck-typed (``_finalized`` /
``_onnx_symbolic_attached``); this module never imports ``exported_custom_op``.
"""
__all__ = ()


# Config-declared ops pending finalization. A LIST (not a dict keyed by torch_op_qualname) because
# different demos legitimately reuse a name like "pypto::add_pypto".
_PENDING_OPS: list = []


def _register_pending_op(op) -> None:
    """Enqueue a config-declared op for finalization (called from ``ExportedCustomOp.__init__``)."""
    _PENDING_OPS.append(op)


def _reset_pending_ops() -> None:
    """Clear the pending-op registry and each op's finalize state. Test-only isolation hook.

    Clearing ``_onnx_symbolic_attached`` too is what makes a re-finalize of the same instance
    re-synthesize its symbolic instead of short-circuiting."""
    for op in _PENDING_OPS:
        op._finalized = False
        op._onnx_symbolic_attached = False
    _PENDING_OPS.clear()
