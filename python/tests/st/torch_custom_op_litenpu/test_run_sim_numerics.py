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
"""Run-stage guard: build and launch a declared op's REAL pypto kernel on the environment-level NPU simulator.

Real compute is gated by ``simulation.accuracy_level == 2``, not ``ENABLE_ESLMODEL`` (set only to match
the normal environment; it routes nothing). Skips only when CANN is absent, a subprocess capability probe
proves the host can't run SIM kernels, or the platform ini lacks ``CCEC_AIV_version`` (a known gap); any
other failure is real and reported. Output buffers carry a sentinel so an unexecuted launch can't pass as
a real one. A simulator that aborts at ACL teardown takes the pytest session down (exit 134) even with
green assertions; SIM and real-NPU builds are mutually exclusive per process.
"""
import contextlib
import glob
import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest
import torch

import pypto
from pypto.extensions.torch_custom_op_litenpu import AttrSpec, ExportedCustomOp
from pypto.extensions.torch_custom_op_litenpu.common.finalize import _reset_pending_ops
from pypto.extensions.torch_custom_op_litenpu.common.run import run_op

_SOC_VERSION = "Kirin9030"
_QUALNAME = "pypto::scale_add"
_ATTRS = (AttrSpec("scale", "Float", 1.0),)
_SCALE_VALUE = 2.0
_SHAPE = (1, 8, 1, 64)
_DTYPE = torch.float16
# float16 carries 11 significand bits, so a sum-then-scale of operands in [0, 1) agrees with the CPU
# reference to a few ulps; the tolerance is that rounding margin and nothing looser.
_RTOL = 2e-3
_ATOL = 2e-3
# Exactly representable in float16 (and float32), and far outside the range the kernel can produce from
# operands in [0, 1) — so "the buffer still holds the sentinel" can only mean "nothing was written".
_POISON = -1024.0

# Seconds allowed for the capability probe: it compiles and launches one small kernel from a cold start.
_CONTROL_TIMEOUT_S = 900

# The capability probe runs as a script in its own process, using the framework directly (no
# ExportedCustomOp, no run_op) so its verdict is about the HOST, not this PR. A real file is required
# because pypto's jit parser reads the kernel body via inspect.getsource. The result line is flushed
# immediately so an abort during interpreter finalization can't swallow it.
_CONTROL_KERNEL_SCRIPT = '''\
"""Capability probe: can this host execute a pypto kernel through the environment-level NPU simulator?"""
import torch

import pypto

pypto.set_global_config("simulation.accuracy_level", 2)

_POISON = -1024.0
_SHAPE = (1, 8, 1, 64)


@pypto.frontend.jit(codegen_options={"soc_version": "Kirin9030"},
                    runtime_options={"run_mode": pypto.RunMode.SIM})
def control_kernel(a: pypto.Tensor([...], pypto.DT_FP16), out: pypto.Tensor([...], pypto.DT_FP16)):
    pypto.set_vec_tile_shapes(1, 4, 1, 64)
    out.move(a + a)


_a = torch.full(_SHAPE, 1.5, dtype=torch.float16)
_out = torch.empty(_SHAPE, dtype=torch.float16)
_out.fill_(_POISON)
control_kernel(_a, _out)
print("PYPTO_SIM_CONTROL", int(torch.sum(_out == _POISON)), _out.numel(), flush=True)
'''
_CONTROL_RESULT_PREFIX = "PYPTO_SIM_CONTROL"


def _missing_sim_requirement():
    """The reason the environment-level NPU simulator cannot run here, or ``None`` when it can."""
    if not os.environ.get("ASCEND_HOME_PATH"):
        return (
            "ASCEND_HOME_PATH is unset: no CANN toolkit, so the RunMode.SIM kernel can be neither built "
            "nor emulated (the environment-level NPU simulator ships with CANN)"
        )
    return None


def _missing_aiv_version_requirement(soc_version=_SOC_VERSION):
    """Whether *soc_version*'s platform config lacks ``CCEC_AIV_version``, or ``None`` when it is present.

    Without this key in the ``[version]`` tab, a ``RunMode.SIM`` launch cannot reach the ESL kernel at all
    (``F21003 FeError::INVALID_VAL``) — see the module docstring. This reads the shipped platform ini
    directly (searched under ``ASCEND_HOME_PATH``, whose exact subpath varies by CANN version), so a host
    already known to lack the key is skipped by name before the control-kernel probe spends a subprocess
    launch proving the same thing indirectly. A host whose ini cannot be located this way falls through to
    that probe instead of being skipped on an inconclusive search.
    """
    home = os.environ.get("ASCEND_HOME_PATH")
    if not home:
        return None  # _missing_sim_requirement() already reports the ASCEND_HOME_PATH gap
    candidates = sorted(glob.glob(os.path.join(home, "**", "platform_config", f"{soc_version}.ini"), recursive=True))
    if not candidates:
        return None  # layout not found under this install; let the control-kernel probe decide
    ini_path = candidates[0]
    text = Path(ini_path).read_text(encoding="utf-8", errors="ignore")
    if "CCEC_AIV_version" not in text:
        return (
            f"{ini_path} has no CCEC_AIV_version key in its [version] tab: a RunMode.SIM launch cannot "
            "reach the ESL kernel here (F21003 FeError::INVALID_VAL) and would silently fall through to "
            "the cost model instead of failing, making a numeric assertion on this path vacuous"
        )
    return None


# The frontend's SIM branch runs the built kernel only at this accuracy level; anything else is the cost
# model (see the module docstring).
_ACCURACY_LEVEL_KEY = "simulation.accuracy_level"
_ACCURACY_LEVEL_FUNCTIONAL = 2


# ── authoring functions (module scope: the tracer reads their source off disk) ────────────────────────
def _scale_add_factory(shapes, dtypes, attrs, soc_version, run_mode=pypto.RunMode.SIM):
    dtype = dtypes[0]
    scale = float(attrs["scale"])   # attr values reach the factory as strings

    @pypto.frontend.jit(codegen_options={"soc_version": soc_version},
                        runtime_options={"run_mode": run_mode})
    def scale_add_inner(a: pypto.Tensor([...], dtype), b: pypto.Tensor([...], dtype),
                            out: pypto.Tensor([...], dtype)):
        pypto.set_vec_tile_shapes(1, 4, 1, 64)
        out.move((a + b) * scale)

    return scale_add_inner


def _scale_add_infer_shape(a: torch.Size, b: torch.Size) -> torch.Size:
    return a


def _scale_add_infer_dtype(a: torch.dtype, b: torch.dtype) -> torch.dtype:
    return a


def _scale_add_torch(a, b, scale):
    return (a + b) * scale


@contextlib.contextmanager
def _poisoned_output_buffers():
    """Fill every device-and-dtype-allocated float buffer with :data:`_POISON` for the duration of the block.

    ``run_op`` allocates the op's outputs as ``torch.empty(shape, dtype=..., device=...)``; poisoning those
    allocations turns "the kernel never wrote its output" from an undetectable read of uninitialized memory
    into a hard, checkable failure.
    """
    real_empty = torch.empty
    poisoned = []

    def _empty(*args, **kwargs):
        buf = real_empty(*args, **kwargs)
        if "dtype" in kwargs and "device" in kwargs and buf.is_floating_point():
            buf.fill_(_POISON)
            poisoned.append(buf)
        return buf

    torch.empty = _empty
    try:
        yield poisoned
    finally:
        torch.empty = real_empty


def _control_kernel_verdict(script_path):
    """Run the capability probe; return ``None`` when it computed, else the evidence that it could not.

    The probe is an independent framework kernel, so a poisoned or failed result is a statement about the
    host's simulator, never about the code under test — which is what makes skipping on it honest.
    """
    script_path.write_text(_CONTROL_KERNEL_SCRIPT)
    try:
        proc = subprocess.run(
            [sys.executable, str(script_path)], capture_output=True, text=True, timeout=_CONTROL_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired:
        return f"control kernel did not finish within {_CONTROL_TIMEOUT_S}s: this host cannot run SIM kernels"

    # The result line is the verdict whenever the probe got far enough to print it — the launch outcome
    # outranks the exit status, which also catches teardown aborts that say nothing about compute.
    line = next((ln for ln in proc.stdout.splitlines() if ln.startswith(_CONTROL_RESULT_PREFIX)), None)
    if line is not None:
        parts = line.split()
        if len(parts) != 3:
            # The verdict is matched by prefix out of the child's whole stdout, and a line carrying it can
            # still arrive incomplete; only the three fields the probe prints are a verdict.
            return f"control kernel printed a malformed result line {line!r}: no verdict on this host"
        _, unwritten, total = parts
        if int(unwritten):
            return (
                f"control kernel left {unwritten}/{total} output elements unwritten: this environment's ESL "
                "simulator executes no kernels (the launch resolves to the no-op Stub* fallback in "
                "framework/src/adapter/api/runtime_api.cpp, which returns success without running anything)"
            )
        return None

    tail = (proc.stderr.strip().splitlines() or ["<no stderr>"])[-1]
    return (
        f"control kernel exited {proc.returncode} without a result, building or launching a bare "
        f"@pypto.frontend.jit RunMode.SIM kernel for {_SOC_VERSION}: this host cannot run SIM kernels "
        f"-- {tail}"
    )


@pytest.fixture()
def eslmodel_simulation(monkeypatch):
    """Select the environment-level simulator's ESL kernel over the cost model, restoring both switches."""
    monkeypatch.setenv("ENABLE_ESLMODEL", "TRUE")
    prior = pypto.get_global_config(_ACCURACY_LEVEL_KEY)
    pypto.set_global_config(_ACCURACY_LEVEL_KEY, _ACCURACY_LEVEL_FUNCTIONAL)
    try:
        yield
    finally:
        pypto.set_global_config(_ACCURACY_LEVEL_KEY, prior)


@pytest.fixture()
def sim_executes_kernels(eslmodel_simulation, tmp_path):
    """Skip only if SIM provably cannot run here; otherwise let the test bite.

    Two gates, in order: the explicit ``CCEC_AIV_version`` precondition (a known, named gap — see
    ``_missing_aiv_version_requirement``), then the capability probe (a general "does this host execute
    SIM kernels at all" control) for every other way the environment could fail to run one.
    """
    reason = _missing_sim_requirement() or _missing_aiv_version_requirement()
    if reason is not None:
        pytest.skip(reason)
    evidence = _control_kernel_verdict(tmp_path / "sim_capability_control.py")
    if evidence is not None:
        pytest.skip(evidence)


@pytest.fixture()
def declared_op():
    """The op under test, with the export/registration state cleared around it so it leaks into no other test."""
    _reset_pending_ops()
    yield ExportedCustomOp(
        kernel=_scale_add_factory,
        infer_shape=_scale_add_infer_shape,
        infer_dtype=_scale_add_infer_dtype,
        torch_defn=_scale_add_torch,
        torch_op_qualname=_QUALNAME,
        attrs=list(_ATTRS),
    )
    _reset_pending_ops()


def test_run_sim_matches_torch_defn(sim_executes_kernels, declared_op):
    op = declared_op

    torch.manual_seed(0)
    a = torch.rand(_SHAPE, dtype=_DTYPE)
    b = torch.rand(_SHAPE, dtype=_DTYPE)

    # The level is the switch the launch actually keys on; it is asserted here so a build that silently
    # reset it shows up as this test's failure rather than as an unexplained cost-model result.
    assert pypto.get_global_config(_ACCURACY_LEVEL_KEY) == _ACCURACY_LEVEL_FUNCTIONAL
    with _poisoned_output_buffers() as poisoned:
        out = run_op(
            op, [a, b], run_mode=pypto.RunMode.SIM, soc_version=_SOC_VERSION,
            attrs={"scale": str(_SCALE_VALUE)},
        )

    # The returned tensor must BE one of the poisoned allocations, otherwise the sentinel check below
    # would be checking a buffer the kernel was never asked to fill.
    assert any(out is buf for buf in poisoned), (
        f"run_op returned a tensor that was not one of the {len(poisoned)} poisoned output allocations"
    )
    # Real compute: every element of the output was written by the kernel. A buffer that comes back
    # untouched means no kernel ran — the cost model instead of the ESL kernel, or a launch that resolved
    # to the runtime adapter's no-op stubs — and neither is something to compare numbers against.
    assert not torch.any(out == _POISON), (
        f"{int(torch.sum(out == _POISON))} of {out.numel()} output elements still hold the poison sentinel "
        f"{_POISON}: the environment-level simulator ran no kernel on this host"
    )

    golden = _scale_add_torch(a, b, _SCALE_VALUE)
    assert out.shape == golden.shape
    assert out.dtype == golden.dtype
    max_abs_err = (out.float() - golden.float()).abs().max().item()
    assert max_abs_err <= _ATOL + _RTOL * golden.float().abs().max().item(), (
        f"simulator output disagrees with the torch_defn golden: max abs err {max_abs_err}"
    )
    torch.testing.assert_close(out.float(), golden.float(), rtol=_RTOL, atol=_ATOL)


# ── the NPU path (real device, own subprocess — see the module docstring) ───────────────────────────────
# Runs in a FRESH interpreter for the same reason test_run_context_and_deploy_sim.py's current-device check does: the
# backend's platform info is process-global and first-touch-wins, so a Kirin9030/LiteNPU SIM build (this
# file's own test above builds one) and a later real-NPU launch in the SAME process are mutually exclusive.

_NPU_RUN_TIMEOUT_S = 900
_NPU_RESULT_MARKER = "PYPTO_NPU_RUN_OK"

# Declares the identical op fresh and compares run_op(run_mode=NPU) against torch_defn. Explicitly imports
# torch_npu (rather than relying on autoload) so the child works even when the parent process disabled
# TORCH_DEVICE_BACKEND_AUTOLOAD for the SIM test above — the same pattern test_demo_sweep.py's own NPU
# capability control uses.
_NPU_CHILD_SCRIPT = '''\
"""Real-device run: build and launch the op via RunMode.NPU, compare against torch_defn."""
import os

import torch
import torch_npu  # noqa: F401  - registers the npu device backend with torch

import pypto
from pypto.extensions.torch_custom_op_litenpu import AttrSpec, ExportedCustomOp
from pypto.extensions.torch_custom_op_litenpu.common.run import run_op

_QUALNAME = "pypto::scale_add"
_SCALE_VALUE = 2.0
_SHAPE = (1, 8, 1, 64)
_DTYPE = torch.float16
_RTOL = 2e-3
_ATOL = 2e-3


def _scale_add_factory(shapes, dtypes, attrs, soc_version, run_mode=pypto.RunMode.NPU):
    dtype = dtypes[0]
    scale = float(attrs["scale"])   # attr values reach the factory as strings

    @pypto.frontend.jit(codegen_options={"soc_version": soc_version},
                        runtime_options={"run_mode": run_mode})
    def scale_add_inner(a: pypto.Tensor([...], dtype), b: pypto.Tensor([...], dtype),
                            out: pypto.Tensor([...], dtype)):
        pypto.set_vec_tile_shapes(1, 4, 1, 64)
        out.move((a + b) * scale)

    return scale_add_inner


def _scale_add_infer_shape(a: torch.Size, b: torch.Size) -> torch.Size:
    return a


def _scale_add_infer_dtype(a: torch.dtype, b: torch.dtype) -> torch.dtype:
    return a


def _scale_add_torch(a, b, scale):
    return (a + b) * scale


op = ExportedCustomOp(
    kernel=_scale_add_factory,
    infer_shape=_scale_add_infer_shape,
    infer_dtype=_scale_add_infer_dtype,
    torch_defn=_scale_add_torch,
    torch_op_qualname=_QUALNAME,
    attrs=[AttrSpec("scale", "Float", 1.0)],
)

torch.npu.set_device(int(os.environ["TILE_FWK_DEVICE_ID"]))

torch.manual_seed(0)
a = torch.rand(_SHAPE, dtype=_DTYPE)
b = torch.rand(_SHAPE, dtype=_DTYPE)

# run_op moves CPU inputs to the current NPU device itself; soc_version=None lets the framework
# auto-detect on-device, mirroring test_run_context_and_deploy_sim.py's current-device check.
out = run_op(op, [a, b], run_mode=pypto.RunMode.NPU, soc_version=None, attrs={"scale": str(_SCALE_VALUE)})
out = out.to("cpu")

golden = _scale_add_torch(a, b, _SCALE_VALUE)
torch.testing.assert_close(out.float(), golden.float(), rtol=_RTOL, atol=_ATOL)
print("PYPTO_NPU_RUN_OK", flush=True)
'''


def _missing_npu_requirement():
    """The reason a real-device run cannot happen here, or ``None`` when it can.

    Uses ``importlib.util.find_spec`` rather than ``import torch_npu``: an eager import in THIS (parent)
    process would initialize the NPU device as a side effect of torch's device-backend autoload, and on
    this box that silently poisons a LATER-SPAWNED CHILD's own ``import torch_npu`` for the same device —
    the child's ``torch`` module ends up with no ``.npu`` attribute at all, no exception raised anywhere.
    ``test_demo_run_numerics.py`` and ``test_golden_gate.py`` document and rely on the same mechanism.
    """
    if importlib.util.find_spec("torch_npu") is None:
        return "torch_npu is not importable: no NPU device backend on this host"
    if shutil.which("npu-smi") is None:
        return "npu-smi is not on PATH, so there is no Ascend device tooling here"
    if "TILE_FWK_DEVICE_ID" not in os.environ:
        return "TILE_FWK_DEVICE_ID is unset: no NPU device id to target"
    return None


def test_run_npu_matches_torch_defn(tmp_path):
    reason = _missing_npu_requirement()
    if reason is not None:
        pytest.skip(reason)

    script_path = tmp_path / "npu_scale_add_child.py"
    script_path.write_text(_NPU_CHILD_SCRIPT)
    proc = subprocess.run(
        [sys.executable, str(script_path)], capture_output=True, text=True, timeout=_NPU_RUN_TIMEOUT_S,
    )
    tail = "\n".join((proc.stdout + proc.stderr).strip().splitlines()[-60:])
    assert proc.returncode == 0, f"NPU run child exited {proc.returncode}\n--- output (last 60 lines) ---\n{tail}"
    assert _NPU_RESULT_MARKER in proc.stdout, f"no success marker from the NPU run child\n{tail}"
