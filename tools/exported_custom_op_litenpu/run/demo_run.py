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
"""Shared run-flag CLI + thin whole-model run helpers for the export demos.

Each demo's ``run_demo.py`` runs the WHOLE model (``model(*inputs)``, every pypto op in the graph) on the
target selected by ``--device`` plus the optional ``--soc_version``: the demo adds the flags with
:func:`add_run_args`, validates the combo with :func:`validate_args`, and then applies the args-to-run
mapping ITSELF, visibly, in its own ``run_demo``. The scenario is always printed before the run. Exporting
is the separate ``export_demo.py`` entry point and takes none of these flags.
"""


import contextlib
import logging
import os

from pypto.extensions.torch_custom_op_litenpu.common.run import (  # noqa: F401 - demo re-export
    RunEnvError,
    pypto_run_context,
)

logger = logging.getLogger(__name__)


class RunCliError(Exception):
    """A misuse of the run flags (one of the invalid combos). The demo prints the message + exits non-zero."""


def npu_device_id():
    """Read the NPU device id from ``TILE_FWK_DEVICE_ID`` (mirrors examples/00|01|02 ``get_device_id``).

    Raises :class:`RunEnvError`, not a print+None, when ``torch.npu`` is missing or the var is unset or
    non-integer, so the demo exits with a clean message instead of a traceback. cpu/sim never call this.
    """
    import torch
    # No torch_npu on this host -> a clean RunEnvError (the demo exits with a message), not an AttributeError.
    if not hasattr(torch, "npu"):
        raise RunEnvError(
            "--device=npu: torch_npu is not available on this host (no torch.npu). Run on an NPU box, "
            "or use --soc_version=<soc> for the env-level NPU simulator."
        )
    if "TILE_FWK_DEVICE_ID" not in os.environ:
        raise RunEnvError(
            "--device=npu requires the NPU device id: export TILE_FWK_DEVICE_ID=<id> "
            "(or use --device=cpu --soc_version=<soc> for the env-level NPU simulator)."
        )
    raw = os.environ["TILE_FWK_DEVICE_ID"]
    try:
        return int(raw)
    except ValueError as exc:
        raise RunEnvError(f"TILE_FWK_DEVICE_ID must be an integer, got: {raw}") from exc


def move_inputs_to_npu(inputs):
    """Move ALL *inputs* to the current NPU device and return the moved tuple.

    A model may mix pypto ops (which output on-device) with standard torch ops that would otherwise keep
    CPU operands -> "Expected all tensors to be on the same device". Bare ``"npu"`` targets the CURRENT
    device, which the demo selects with ``torch.npu.set_device(npu_device_id())`` before calling this.
    """
    return tuple(t.to("npu") for t in inputs)


@contextlib.contextmanager
def enable_eslmodel():
    """Set ENABLE_ESLMODEL=TRUE for the block, restoring the prior value (or unset) on exit.

    A demo-side env toggle, composed alongside ``pypto_run_context(...)`` at the demo call site and
    deliberately kept OUT of the pypto run-context primitive.
    """
    prev = os.environ.get("ENABLE_ESLMODEL")
    os.environ["ENABLE_ESLMODEL"] = "TRUE"
    try:
        yield
    finally:
        if prev is None:
            os.environ.pop("ENABLE_ESLMODEL", None)
        else:
            os.environ["ENABLE_ESLMODEL"] = prev


def add_run_args(parser):
    """Add ``--device`` / ``--soc_version`` to *parser* (the valid whole-model run forms)."""
    parser.add_argument(
        "--device", choices=["cpu", "npu"], default=None,
        help="cpu (default; torch_defn — or, with --soc_version, the env-level NPU simulator) | "
             "npu -> real NPU (RunMode.NPU, soc auto-detected, device from TILE_FWK_DEVICE_ID)",
    )
    parser.add_argument(
        "--soc_version", default=None,
        help="run the env-level NPU simulator (RunMode.SIM kernel + CPU tensors) for this soc; valid with "
             "--device=cpu or with --device omitted, never with --device=npu",
    )


def validate_args(args):
    """Raise :class:`RunCliError` on any invalid ``--device``/``--soc_version`` combo.

    ``--soc_version`` selects the env-level NPU simulator for the cpu run, so it pairs with ``--device=cpu``
    or with ``--device`` omitted, never with ``--device=npu``, which auto-detects the soc on-device.
    """
    if args.device == "npu" and args.soc_version is not None:
        raise RunCliError(
            "--soc_version is only for --device=cpu (env-level NPU simulator); --device=npu auto-detects the soc"
        )


def scenario_label(args):
    """The human-readable scenario label for *args* (label text only, no control flow).

    Kept here so the printed label stays consistent across demos while the mapping ``if`` stays visible at
    each call site.
    """
    if args.device == "npu":
        return "npu (RunMode.NPU, device from TILE_FWK_DEVICE_ID, auto-soc)"
    if args.soc_version:
        return f"cpu + soc={args.soc_version} (env-level NPU simulator, RunMode.SIM)"
    return "cpu (torch_defn)"


def print_outputs(out):
    """Print each output's shape/dtype/sample (single tensor or tuple of tensors)."""
    outs = out if isinstance(out, tuple) else (out,)
    for i, o in enumerate(outs):
        tag = f"out{i}" if len(outs) > 1 else "output"
        o_cpu = o.to("cpu")
        logger.info(
            "run: %s shape=%s dtype=%s sample=%s",
            tag, tuple(o_cpu.shape), o_cpu.dtype, o_cpu.flatten()[:4].tolist(),
        )
