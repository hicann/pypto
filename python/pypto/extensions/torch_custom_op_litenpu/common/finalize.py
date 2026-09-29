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
"""The declare-then-finalize registry: enqueue config-declared ops and synthesize their framework wiring.

An op that carries ``torch_op_qualname`` plus an ``onnx_spec`` enqueues itself here from
``ExportedCustomOp.__init__``; ``finalize_pending_ops()`` then synthesizes and registers its onnx
symbolic. Deferring it keeps ``torch.onnx`` out of import time. Pending ops are handled duck-typed (reads
``_onnx_spec``, writes ``_finalized`` / ``_onnx_symbolic_attached``); the onnx synthesizer and the
opset-floor reset are reached by function-scope imports, so this module carries no module-scope edge to
``onnx``.
"""
__all__ = ("finalize_pending_ops",)


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


def _reset_export_state() -> None:
    """Reset ALL process-global export registration state (pending ops + onnx opset floors). Test-only."""
    _reset_pending_ops()
    # layering: this module carries no module-scope edge to onnx
    from ..onnx.export import _reset_onnx_opset_floors  # noqa: PLC0415
    _reset_onnx_opset_floors()


def finalize_pending_ops() -> None:
    """Synthesize + register the framework symbolic of every pending config-declared op.

    Idempotent (per-op ``_finalized`` flag), so every export-time caller may call it and whichever runs
    first does the work. A no-op when no op used config (hand-written demos are never in the registry).
    An op declaring no framework spec has nothing to synthesize; its torch op was registered at
    construction.
    """
    for op in _PENDING_OPS:
        if getattr(op, "_finalized", False):
            continue
        if op._onnx_spec is not None:
            # layering: this module carries no module-scope edge to onnx
            from ..onnx.export import _synthesize_onnx  # noqa: PLC0415
            _synthesize_onnx(op, op._onnx_spec)
        op._finalized = True
