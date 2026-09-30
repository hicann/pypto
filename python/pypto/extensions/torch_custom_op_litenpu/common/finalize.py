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
symbolic. The onnx layer binds its synthesizer and its opset-floor reset into the registry below at
import time, so every import edge runs framework -> common. Pending ops are handled duck-typed (reads
``_onnx_spec``, writes ``_finalized`` / ``_onnx_symbolic_attached``).
"""
__all__ = ("finalize_pending_ops",)


# Config-declared ops pending finalization. A LIST (not a dict keyed by torch_op_qualname) because
# different demos legitimately reuse a name like "pypto::add_pypto".
_PENDING_OPS: list = []

# Framework wiring bound in by the framework layers at import time, keyed by framework name.
_SYNTHESIZERS: dict = {}
_RESET_HOOKS: list = []


def _register_synthesizer(framework: str, fn) -> None:
    """Bind *framework* to the callable that synthesizes an op's wiring from its declared spec."""
    _SYNTHESIZERS[framework] = fn


def _register_reset_hook(fn) -> None:
    """Add *fn* to the process-global state that ``_reset_export_state`` clears."""
    _RESET_HOOKS.append(fn)


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
    """Reset ALL process-global export registration state (pending ops + every reset hook). Test-only."""
    _reset_pending_ops()
    for reset_hook in _RESET_HOOKS:
        reset_hook()


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
            synthesize = _SYNTHESIZERS.get("onnx")
            if synthesize is None:
                raise RuntimeError(f"{op._torch_op_qualname!r} declares an onnx spec but no onnx synthesizer "
                                   "is registered; import torch_custom_op_litenpu.onnx.export before finalizing.")
            synthesize(op, op._onnx_spec)
        op._finalized = True
