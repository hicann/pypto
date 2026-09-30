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
"""Drive every demo in ``onnx/`` through its two shipping stages: export, then ``build_so``.

Demos are discovered, never listed: the parametrization globs ``onnx/*/export_demo.py``, so a demo
folder added to the tree is swept the moment it lands.

Each stage is a subprocess, exactly as a user runs it. Demo modules register process-global
``torch.ops`` entries under fixed qualnames and build kernels through process-global framework state,
so one process per demo per stage is the only regime in which the sweep measures the demos rather than
their collisions.

Asserted per demo: export exits 0 and leaves a non-empty ``.onnx``; ``build_so`` on that model exits 0
and leaves a non-empty ``libcust_opapi.so``. Any non-zero exit is a failure with the captured output
attached.
"""
import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[5]
DEMOS_DIR = _REPO_ROOT / "examples/03_advanced/exported_custom_op_litenpu/onnx"
BUILD_SO_SCRIPT = _REPO_ROOT / "tools/exported_custom_op_litenpu/deploy/build_so_and_setup.py"

EXPORT_TIMEOUT_S = 900
BUILD_TIMEOUT_S = 2400

# Python packages the export stage needs.
EXPORT_REQUIREMENTS = ("torch", "onnx", "pypto")
# Accepted C++ compiler drivers for the build stage.
CXX_DRIVERS = ("g++", "c++", "clang++")

# The GE header the generated op host and the shared ``PtoCustomOp`` base ``#include``. CANN 9.0.0 ships
# ``exe_graph/runtime/`` but not this header; CANN 9.2.0 does. Its absence must surface as a proven skip
# here, not as a ``fatal error: ... No such file or directory`` from every demo's build subprocess.
REQUIRED_GE_HEADER = "exe_graph/runtime/annotated_args_context.h"


def discover_demos():
    """Every demo in the tree, by its entry point. Sorted so the parametrization ids are stable."""
    return sorted(DEMOS_DIR.glob("*/export_demo.py"))


DEMOS = discover_demos()
DEMO_IDS = [p.parent.name for p in DEMOS]


def _missing_export_requirements():
    return [name for name in EXPORT_REQUIREMENTS if importlib.util.find_spec(name) is None]


def _missing_build_requirements():
    missing = []
    ascend_home = os.environ.get("ASCEND_HOME_PATH")
    if not ascend_home or not Path(ascend_home).is_dir():
        missing.append("ASCEND_HOME_PATH pointing at an Ascend install")
    if shutil.which("cmake") is None:
        missing.append("cmake")
    if not any(shutil.which(drv) for drv in CXX_DRIVERS):
        missing.append("a C++ compiler (%s)" % "/".join(CXX_DRIVERS))

    # Plausible include roots for the header: the Ascend install (plain ``include/`` plus the
    # arch-specific subtrees real installs use, e.g. ``aarch64-linux/include/``) and any directory
    # listed in $PYPTO_EXTRA_INCLUDE_DIRS (colon-separated, same convention as $CPATH) — the documented
    # way to point the build at GE headers a CANN install doesn't ship.
    include_roots = []
    if ascend_home:
        home = Path(ascend_home)
        include_roots += [
            home / "include",
            home / "aarch64-linux" / "include",
            home / "x86_64-linux" / "include",
        ]
    include_roots += [Path(d) for d in os.environ.get("PYPTO_EXTRA_INCLUDE_DIRS", "").split(os.pathsep) if d]
    if not any((root / REQUIRED_GE_HEADER).is_file() for root in include_roots):
        missing.append(
            "%s (CANN 9.0.0 does not ship it, 9.2.0 does; point $ASCEND_HOME_PATH at a CANN "
            "install that has it, or add its directory to $PYPTO_EXTRA_INCLUDE_DIRS)" % REQUIRED_GE_HEADER
        )
    return missing


def _run(argv, cwd, timeout):
    return subprocess.run(
        [sys.executable, *[str(a) for a in argv]],
        cwd=str(cwd), capture_output=True, text=True, timeout=timeout, check=False,
    )


def _report(stage, demo, proc):
    tail = "\n".join((proc.stdout + proc.stderr).strip().splitlines()[-40:])
    return f"{demo.parent.name}: {stage} exited {proc.returncode}\n--- output (last 40 lines) ---\n{tail}"


def test_demos_are_discovered():
    # An empty parametrization would report a clean run having tested nothing.
    assert DEMOS, f"no demos discovered under {DEMOS_DIR}"
    assert BUILD_SO_SCRIPT.is_file(), f"missing the build_so entry point: {BUILD_SO_SCRIPT}"


@pytest.fixture(scope="module")
def sweep_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("demo_sweep")


@pytest.fixture(scope="module")
def exported(sweep_dir):
    """``exported(demo)`` -> the demo's ``.onnx``, exporting it once per module run and caching the result."""
    missing = _missing_export_requirements()
    if missing:
        pytest.skip(f"export needs {', '.join(missing)}")
    # name -> the exported model path, so every consumer of this fixture reuses one export subprocess.
    cache = {}

    def _export_once(demo, name):
        out_dir = sweep_dir / name
        out_dir.mkdir(parents=True, exist_ok=True)
        model_path = out_dir / f"{name}.onnx"
        proc = _run([demo, model_path], cwd=out_dir, timeout=EXPORT_TIMEOUT_S)
        assert proc.returncode == 0, _report("export", demo, proc)
        assert model_path.is_file(), _report("export", demo, proc) + f"\n(no model at {model_path})"
        assert model_path.stat().st_size > 0, f"{name}: exported an EMPTY model at {model_path}"
        return model_path

    def _export(demo):
        name = demo.parent.name
        if name not in cache:
            cache[name] = _export_once(demo, name)
        return cache[name]

    return _export


@pytest.mark.parametrize("demo", DEMOS, ids=DEMO_IDS)
def test_demo_exports_onnx(demo, exported):
    exported(demo)


@pytest.mark.parametrize("demo", DEMOS, ids=DEMO_IDS)
def test_demo_builds_so(demo, exported, sweep_dir):
    missing = _missing_build_requirements()
    if missing:
        pytest.skip(f"build_so needs {', '.join(missing)}")
    model_path = exported(demo)          # skips/fails on the export stage before the build is attempted

    out_dir = sweep_dir / demo.parent.name / "build_op_kernel_lib"
    proc = _run([BUILD_SO_SCRIPT, model_path, "--out-dir", out_dir], cwd=sweep_dir, timeout=BUILD_TIMEOUT_S)
    assert proc.returncode == 0, _report("build_so", demo, proc)

    built = sorted(out_dir.rglob("libcust_opapi.so"))
    assert built, _report("build_so", demo, proc) + f"\n(no libcust_opapi.so under {out_dir})"
    for so_path in built:
        assert so_path.stat().st_size > 0, f"{demo.parent.name}: built an EMPTY library at {so_path}"
