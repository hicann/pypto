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
"""Run-path helpers: drive the real pypto kernel (NPU device launch or env-level NPU simulator).

A thread-local run context (:func:`pypto_run_context`) selects each op's run_mode; with none, an op
runs its eager ``torch_defn`` on CPU (there is no CPU run_mode). With a context, :func:`run_op` builds
the kernel, allocates outputs from ``infer_shape``/``infer_dtype``, and launches: ``RunMode.NPU`` syncs
the device after launch and auto-detects ``soc_version`` when omitted; ``RunMode.SIM`` needs an explicit
soc plus ``ENABLE_ESLMODEL=TRUE`` in the env, routing through the ESL model instead of the cost model.
"""
from __future__ import annotations

import contextlib
from dataclasses import dataclass
import inspect
import threading

import pypto

from . import return_annotations
from .authoring import _detect_factory_signature, _direct_declared_annotations
from .compile import CompileEntry


class RunEnvError(RuntimeError):
    """The run environment can't satisfy the requested scenario (e.g. no NPU to auto-detect a soc).

    Distinct from a kernel/compute failure so callers (the demo CLI) can print it cleanly instead of a
    stack trace: it signals "run this elsewhere / pass an explicit soc", not a bug.
    """


@dataclass(frozen=True)
class _RunContext:
    """A resolved whole-model run target, read by ``torch_op._impl`` for every pypto op in the forward.

    Carries a ``pypto.RunMode`` and an optional ``soc_version``; see the module docstring for what each
    mode means and when the soc is required.
    """
    run_mode: "pypto.RunMode"
    soc_version: str | None = None


# The active run context, set only around an explicit demo ``model(*inputs)`` run. Thread-local so a run
# on one thread and an export on another do not cross-contaminate.
_RUN_CONTEXT = threading.local()


def get_run_context():
    """Return the active :class:`_RunContext`, or ``None`` when no run context is set (cpu / export)."""
    return getattr(_RUN_CONTEXT, "ctx", None)


@contextlib.contextmanager
def pypto_run_context(*, run_mode, soc_version=None):
    """Set the run context for the duration of the block (restored on exit).

    The demo CLI wraps ``model(*inputs)`` in this so every pypto op's ``_impl`` reads the same target;
    see the module docstring for the per-mode semantics. Nesting-safe.
    """
    prev = getattr(_RUN_CONTEXT, "ctx", None)
    _RUN_CONTEXT.ctx = _RunContext(run_mode, soc_version=soc_version)
    try:
        yield _RUN_CONTEXT.ctx
    finally:
        _RUN_CONTEXT.ctx = prev


def _build_compile_entry(op):
    """Build a :class:`CompileEntry` for *op*'s kernel (DIRECT or factory), infer hooks attached.

    Reuses the same classifier the export path uses (DIRECT jit vs factory + factory_signature) so the run
    path and the deployed compile snippet drive the kernel identically.
    """
    # Tensor-input count from infer_dtype (never takes attrs), NOT infer_shape: a shape-affecting attr
    # adds a trailing param to infer_shape, so its param count would over-count the tensor inputs.
    # Same drift-free rationale as the torch_op n_in derivation.
    n_inputs = len(inspect.signature(op._infer_dtype_fn).parameters)
    n_outputs = return_annotations.infer_shape_output_arity(op._infer_shape_fn)
    if op._bare_kernel_fn is not None:
        entry = CompileEntry(
            num_inputs=n_inputs,
            num_outputs=n_outputs,
            jit_kernel=op._bare_kernel_fn,
            declared_annotations=_direct_declared_annotations(op),
        )
    else:
        entry = CompileEntry(
            num_inputs=n_inputs,
            num_outputs=n_outputs,
            factory=op._create_kernel_fn,
            factory_signature=_detect_factory_signature(op._create_kernel_fn),
        )
    entry.infer_shape = op._infer_shape_fn
    entry.infer_dtype = op._infer_dtype_fn
    return entry


# pypto.frontend.jit() parameter -> the attribute JitCallableWrapper stores it under. Anything not listed
# uses entry.py's "_<param>" convention. Derived from jit()'s live signature rather than hand-listed, so
# an upstream add, rename or removal is either carried through or fails loudly here.
_JIT_PARAM_ATTR_OVERRIDES = {"new_ir": "_use_new_ir"}


def _collect_decorator_options(wrapper):
    """Read every ``pypto.frontend.jit()`` option back off *wrapper*, keyed by jit()'s parameter names.

    Returns a kwargs dict ready for ``pypto.frontend.jit(**options)``. Raises ``RuntimeError`` (not
    ``RunEnvError``, which the demo CLI turns into a clean skip) naming any parameter whose value cannot
    be recovered, so an upstream signature change surfaces here rather than as a dropped option.
    """
    options = {}
    missing = []
    for name, param in inspect.signature(pypto.frontend.jit).parameters.items():
        if name == "func" or param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            continue
        attr = _JIT_PARAM_ATTR_OVERRIDES.get(name, f"_{name}")
        if hasattr(wrapper, attr):
            options[name] = getattr(wrapper, attr)
        else:
            missing.append(f"{name} (looked for {type(wrapper).__name__}.{attr})")
    if missing:
        raise RuntimeError(
            "pypto.frontend.jit() has option(s) this DIRECT re-wrap cannot carry through: "
            + ", ".join(missing)
            + "; add the jit-param to wrapper-attribute mapping to _JIT_PARAM_ATTR_OVERRIDES."
        )
    return options


def _rewrap_direct(_op, _soc_version, _run_mode):
    """Re-wrap a DIRECT kernel's underlying function in a fresh jit for *_run_mode* (soc pinned).

    A DIRECT ``@pypto.frontend.jit`` kernel pins its run_mode at decoration, so launching it for a
    different target means rebuilding the same function with the requested ``run_mode`` and soc, carrying
    every other author-set option through unchanged (:func:`_collect_decorator_options`) so this run build
    matches the deployed ``.so``. ``soc_version`` is injected only when explicitly given -- ``None`` leaves
    no soc key pinned, so the framework auto-detects it on-device. The DIRECT decorator itself is forbidden
    from pinning soc (see ``_validate_direct_decorator_no_soc``).
    """
    _wrapper = _op._bare_kernel_fn
    _options = _collect_decorator_options(_wrapper)
    # Copy both dicts before overriding: getattr hands back the wrapper's OWN stored dicts
    # (entry.py stores runtime_options by reference), so mutating them in place would corrupt the
    # author's decorated kernel and make soc_version sticky across run_op calls.
    _codegen_options = dict(_options.get("codegen_options") or {})
    if _soc_version is not None:
        _codegen_options["soc_version"] = _soc_version
    _options["codegen_options"] = _codegen_options
    _options["runtime_options"] = {**(_options.get("runtime_options") or {}), "run_mode": _run_mode}
    # Every name in this frame, the parameters included, is underscore-prefixed: jit() snapshots this
    # frame's locals (entry.py decorator_wrapper -> frame.f_back.f_locals) to override the author's
    # module globals during parsing, so a plain `run_mode`/`soc_version` name would shadow an
    # identically-named author global. Callers pass positionally, so the underscores stay invisible.
    return pypto.frontend.jit(**_options)(_wrapper._original_func)


def run_op(op, inputs, *, run_mode, soc_version=None, attrs=None):
    """Launch *op*'s real pypto kernel and return the output(s): the run primitive.

    Called by ``torch_op._impl`` only when a run context is set. Builds the op's jit kernel via
    :class:`CompileEntry`, allocates outputs from ``infer_shape``/``infer_dtype``, launches (mutating the
    outputs in place), and returns a bare tensor for a single output else a tuple.

    *attrs* is the op's compute attributes as a ``{name: value-string}`` dict, pre-stringified to mirror
    the deployed C++ ``BuildAttrsDict`` contract; ``None`` means ``{}``.
    """
    import torch  # noqa: PLC0415 - optional dependency: the run path is only entered with a run context

    attrs = {} if attrs is None else dict(attrs)

    if run_mode == pypto.RunMode.NPU:
        try:
            # optional dependency: auto-loads torch.npu on setups w/o TORCH_DEVICE_BACKEND_AUTOLOAD,
            # and must be free to fail on a host without it
            import torch_npu  # noqa: F401,PLC0415
        except Exception:
            pass
        if not (hasattr(torch, "npu") and torch.npu.is_available()):
            raise RunEnvError(
                "run_op(run_mode=RunMode.NPU): no NPU available (torch_npu missing or no device). "
                "Run on an NPU box, or use --device=cpu --soc_version=<soc> for the env simulator."
            )
        # soc_version stays as passed (may be None): when unpinned the framework auto-detects the soc
        # on-device (Platform::ObtainPlatformInfo -> CannHostRuntime::GetSocVersion).
    elif run_mode == pypto.RunMode.SIM:
        if soc_version is None:
            raise RunEnvError(
                "run_op(run_mode=RunMode.SIM): soc_version is required (the soc the env-level NPU simulator emulates)"
            )
    else:
        raise ValueError(f"run_op: unknown run_mode {run_mode!r} (expected pypto.RunMode.NPU or pypto.RunMode.SIM)")

    entry = _build_compile_entry(op)
    shapes = [tuple(t.shape) for t in inputs]
    dtype_strs = [str(t.dtype).split(".")[-1] for t in inputs]

    # npu launches on the current device (selected by the caller's torch.npu.set_device()); sim keeps CPU
    # tensors (the env-level NPU simulator runs the RunMode.SIM kernel against them).
    out_device = "npu" if run_mode == pypto.RunMode.NPU else "cpu"

    if op._create_kernel_fn is not None:
        jit_obj, out_shapes, out_dtypes = entry.build_jit(
            shapes, dtype_strs, soc_version=soc_version, run_mode=run_mode, attrs=attrs,
        )
    else:
        # DIRECT kernel: the decoration-pinned jit can't launch for this target; re-wrap the underlying
        # function for run_mode (soc injected only when given), and infer the output shapes/dtypes without a
        # jit build so no process-global soc is touched.
        jit_obj = _rewrap_direct(op, soc_version, run_mode)
        out_shapes, out_dtypes = entry.infer_outputs(shapes, dtype_strs, attrs=attrs)

    run_inputs = [t.to(out_device) for t in inputs]
    outputs = [torch.empty(s, dtype=d, device=out_device) for s, d in zip(out_shapes, out_dtypes)]
    jit_obj(*run_inputs, *outputs)  # __call__ returns None; outputs are mutated in place
    if run_mode == pypto.RunMode.NPU:
        pypto.runtime._device_synchronize()  # sync every real-device launch

    return outputs[0] if len(outputs) == 1 else tuple(outputs)
