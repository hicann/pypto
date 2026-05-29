# Copyright (c) Huawei Technologies Co., Ltd. 2024-2026. All rights reserved.

# Template: SIM / NPU 2-way dispatcher for adversarial_runner.py
#
# Implements the contract from verification.md §B.3.1. Drop this into
# custom/<op>/eval/adversarial_runner.py (or import the helpers). It wraps the
# existing _compare() + build_hybrid() functions and produces the new fields
# required in evaluation_report.json (per_mode_status, divergence_fingerprint).
#
# Modes supported: "sim" and "npu". Torch mode was removed — pypto.RunMode
# only defines NPU and SIM (see python/pypto/runtime.py); there is no
# pypto.torch_backend module.
#
# DO NOT rename run_modes, compute_divergence_fingerprint, or aggregate_per_mode_status.
# @pypto-op-debugger binds to these names.

from __future__ import annotations

import io
import os
import sys
import time
import traceback
from contextlib import redirect_stdout, redirect_stderr
from typing import Any, Dict, List, Tuple

import torch

# -----------------------------------------------------------------------------
# Mode availability probes
# -----------------------------------------------------------------------------


def _mode_available(mode: str) -> Tuple[bool, str]:
    """Return (available, reason_if_not). Supported modes: "sim", "npu"."""
    if mode == "sim":
        return True, ""
    if mode == "npu":
        if not os.environ.get("ASCEND_HOME_PATH"):
            return False, "ASCEND_HOME_PATH not set"
        try:
            import torch_npu  # noqa: F401
            if not torch.npu.is_available():
                return False, "torch.npu.is_available() == False"
            return True, ""
        except ImportError as e:
            return False, f"torch_npu import failed: {e}"
    return False, f"unknown mode {mode!r} (supported: sim, npu)"


# -----------------------------------------------------------------------------
# Per-mode dispatch
# -----------------------------------------------------------------------------

def _run_one_mode(
    impl_module,
    case: dict,
    mode: str,
    make_inputs_fn,
    primary_input_order: list,
    truth_tuple: tuple,
    compare_fn,
) -> Dict[str, Any]:
    """
    Run the impl under one mode and compare to the truth tuple.
    Returns: {status, per_tensor, max_abs_diff, max_rel_diff, stdout, stderr, runtime_s}
    """
    import pypto   # late import so the test process can pick up the backend
    ok, reason = _mode_available(mode)
    if not ok:
        return {
            "status": "SKIPPED", "reason": reason,
            "per_tensor": [], "max_abs_diff": None, "max_rel_diff": None,
            "stdout": "", "stderr": "", "runtime_s": 0.0,
        }

    stdout_buf, stderr_buf = io.StringIO(), io.StringIO()
    t0 = time.time()
    status = "ERROR"
    per_tensor: List[Dict[str, Any]] = []
    max_abs_diff, max_rel_diff = None, None

    try:
        # Build inputs once per mode (shapes/dtypes live in case).
        inputs = make_inputs_fn(case)
        positional = [inputs[name] for name in primary_input_order]

        # Override run_mode on the impl's JIT wrapper.
        run_mode_map = {"sim": pypto.RunMode.SIM, "npu": pypto.RunMode.NPU}
        try:
            impl_module.__pypto_run_mode__ = run_mode_map[mode]
        except KeyError as e:
            raise KeyError(
                f"{mode!r} is not a valid mode; available: {sorted(run_mode_map)}"
            ) from e

        with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
            candidate = impl_module.host_wrapper(*positional)
            if not isinstance(candidate, tuple):
                candidate = (candidate,)

        cmp = compare_fn(candidate, truth_tuple, atol=case["atol"], rtol=case["rtol"])
        per_tensor = cmp["per_tensor"]
        max_abs_diff = max((t["max_abs_diff"] for t in per_tensor), default=0.0)
        max_rel_diff = max((t["max_rel_diff"] for t in per_tensor), default=0.0)
        status = "PASS" if cmp["all_close"] else "FAIL"

    except Exception as e:
        status = "ERROR"
        stderr_buf.write(f"\n[dispatcher] {mode} raised: {e}\n{traceback.format_exc()}")

    return {
        "status": status,
        "per_tensor": per_tensor,
        "max_abs_diff": max_abs_diff,
        "max_rel_diff": max_rel_diff,
        "stdout": stdout_buf.getvalue(),
        "stderr": stderr_buf.getvalue(),
        "runtime_s": time.time() - t0,
    }


# -----------------------------------------------------------------------------
# Top-level dispatcher — DO NOT RENAME
# -----------------------------------------------------------------------------

def run_modes(
    impl_module,
    case: dict,
    modes: List[str],
    *,
    make_inputs_fn,
    primary_input_order: list,
    truth_tuple: tuple,
    compare_fn,
) -> Dict[str, Dict[str, Any]]:
    """
    Execute the impl under every mode and return per-mode verdicts.
    Never short-circuits — the cross-mode verdict pattern is itself the signal.
    """
    out: Dict[str, Dict[str, Any]] = {}
    for mode in modes:
        out[mode] = _run_one_mode(
            impl_module, case, mode,
            make_inputs_fn=make_inputs_fn,
            primary_input_order=primary_input_order,
            truth_tuple=truth_tuple,
            compare_fn=compare_fn,
        )
    return out


# -----------------------------------------------------------------------------
# Aggregation helpers — DO NOT RENAME
# -----------------------------------------------------------------------------

def aggregate_per_mode_status(per_case_results: List[Dict[str, Dict[str, Any]]]) -> Dict[str, str]:
    """
    Fold per-case per-mode verdicts into an aggregate per-mode status.
    Rule: worst-case across cases. ERROR > FAIL > PASS > SKIPPED.
    """
    order = {"ERROR": 3, "FAIL": 2, "PASS": 1, "SKIPPED": 0}
    reverse = {v: k for k, v in order.items()}
    agg: Dict[str, int] = {}
    for case_result in per_case_results:
        for mode, verdict in case_result.items():
            agg[mode] = max(agg.get(mode, 0), order.get(verdict["status"], 0))
    return {m: reverse[v] for m, v in agg.items()}


def compute_divergence_fingerprint(failing_case_result: Dict[str, Dict[str, Any]]) -> str:
    """
    Classify the cross-mode verdict pattern for a single failing case.
    Returns one of: kernel_ok_npu_only, ir_divergence, all_fail, ambiguous.

    sim PASS + npu FAIL  → kernel_ok_npu_only (NPU codegen / tile / alignment issue)
    sim FAIL + npu PASS  → ir_divergence (simulator-only divergence; rare)
    sim FAIL + npu FAIL  → all_fail
    """
    def ok(m: str) -> bool:
        return failing_case_result.get(m, {}).get("status") == "PASS"

    def bad(m: str) -> bool:
        return failing_case_result.get(m, {}).get("status") in ("FAIL", "ERROR")

    if ok("sim") and bad("npu"):
        return "kernel_ok_npu_only"
    if bad("sim") and ok("npu"):
        return "ir_divergence"
    if bad("sim") and bad("npu"):
        return "all_fail"
    return "ambiguous"


_RUNNER_INTEGRATION_NOTE = """\
-----------------------------------------------------------------------------
Wiring into adversarial_runner.py main()
-----------------------------------------------------------------------------

In main(), replace the single-mode call:

    candidate = impl_module.host_wrapper(*positional)
    cmp = _compare(candidate, truth_tuple, atol, rtol)

with:

    per_mode = run_modes(
        impl_module, case, modes,
        make_inputs_fn=make_inputs,
        primary_input_order=PRIMARY_INPUT_ORDER,
        truth_tuple=truth_tuple,
        compare_fn=_compare,
    )
    case_result = {
        "id": case["id"], "level": case["level"],
        "precision_checked": case.get("precision", True),
        "all_close": all(v["status"] in ("PASS", "SKIPPED") for v in per_mode.values()),
        "per_mode_status": {m: v["status"] for m, v in per_mode.items()},
        "max_abs_diff": max((v["max_abs_diff"] or 0.0) for v in per_mode.values()),
        "max_rel_diff": max((v["max_rel_diff"] or 0.0) for v in per_mode.values()),
    }
    all_case_results.append(case_result)
    if not case_result["all_close"] and report["first_failure"] is None:
        report["first_failure"] = {
            "case_id": case["id"],
            "failing_module_boundary": up_to_module,
            "failure_category": "precision",   # refined post-hoc — see DEBUG_GUIDEBOOK.md 2f
            "divergence_fingerprint": compute_divergence_fingerprint(per_mode),
            "summary": (
                "failed under modes: "
                f"{[m for m, v in per_mode.items() if v['status'] == 'FAIL']}"
            ),
        }

Then at the end:

    report["per_mode_status"] = aggregate_per_mode_status(
        [{mode: {"status": cr["per_mode_status"][mode]} for mode in cr["per_mode_status"]}
         for cr in all_case_results]
    )
    report = _sanitize(report)
    json.dump(report, open(args.report, "w"), indent=2)
"""
