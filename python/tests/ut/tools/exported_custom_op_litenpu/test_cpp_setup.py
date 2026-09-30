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
"""Tests for the deploy helper ``setup_onnx_custom_op_so`` (placement layout) and for the op package's
``pypto_version.info`` — the file the build writes beside the ``.so`` and the install refuses to place
a package without."""

from __future__ import annotations

# Running this file directly, with no conftest, needs the tools root on sys.path
# for the `exported_custom_op_litenpu.*` imports below. Redundant under pytest: the sibling conftest.py
# already does this, and the guard there makes it a no-op.
import pathlib as _pl
import sys as _sys

_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[5] / "tools"))

import inspect
from pathlib import Path

from exported_custom_op_litenpu.deploy.build import _build_so_from_nodes, _write_pypto_version_info
from exported_custom_op_litenpu.deploy.embed import installed_pypto_version, load_embedded_compile_module
from exported_custom_op_litenpu.deploy.setup import setup_onnx_custom_op_so
import pytest


def _write_version_info(opp_root: Path, version=None) -> Path:
    """Write the package's ``pypto_version.info``, defaulting to the version this pypto reports."""
    recorded = installed_pypto_version() if version is None else version
    info = opp_root / "pypto_version.info"
    info.write_text(f"Version={recorded}\n", encoding="utf-8")
    return info


def _make_so(tmp_path: Path) -> Path:
    so = tmp_path / "libcust_opapi.so"
    so.write_bytes(b"\x7fELF fake")
    # A real build writes this beside the .so, and the install refuses a package without it — so every
    # fixture package here is a complete one.
    _write_version_info(tmp_path)
    return so


def test_setup_places_so_in_op_proto_framework_and_root(tmp_path):
    """The combined .so must land in op_proto/ (compile-phase InferShape) and
    framework/onnx/ (parse phase), while the original stays at the root for
    OpLibRegistry. The op_proto/ copy is the fix for the GE
    "output is unknown shape" CheckStaticShape failure."""
    so = _make_so(tmp_path)
    dest = setup_onnx_custom_op_so(so, kernel_py_paths=[])

    # parse-phase plugin location is the documented return value
    assert dest == tmp_path / "framework" / "onnx" / "libcust_opapi.so"
    assert dest.is_file()
    # compile-phase op-proto location, OpsProtoManager scans <entry>/op_proto/
    assert (tmp_path / "op_proto" / "libcust_opapi.so").is_file()
    # root copy preserved for OpLibRegistry (copy, not move, by default)
    assert so.is_file()


def test_setup_missing_so_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        setup_onnx_custom_op_so(tmp_path / "does_not_exist.so", kernel_py_paths=[])


def test_setup_deploys_kernel_py_to_op_kernel_dir(tmp_path):
    """Each op's dev-editable snippet is copied to <opp_root>/op_kernel/<stem>.py (NOT op_host/),
    where PtoCustomOp::Compile resolves it via the ASCEND_CUSTOM_OPP_PATH entries."""
    so = _make_so(tmp_path)
    src_py = tmp_path / "pypto_custom_op_add.py"
    src_py.write_text("def __pypto_compile(*a):\n    return '/tmp/x.o'\n", encoding="utf-8")

    setup_onnx_custom_op_so(
        so, kernel_py_paths=[("Add", "pypto_custom_op_add", src_py)]
    )

    deployed = tmp_path / "op_kernel" / "pypto_custom_op_add.py"
    assert deployed.is_file()
    assert deployed.read_text(encoding="utf-8") == src_py.read_text(encoding="utf-8")
    # NOT the old op_host/ location.
    assert not (tmp_path / "op_host" / "pypto_custom_op_add_kernel.py").exists()
    assert not (tmp_path / "op_host").exists()
    # src is copied, not moved.
    assert src_py.is_file()


def test_setup_verifies_kernel_py_importable(tmp_path):
    """A valid snippet (callable __pypto_compile) passes the install-time import smoke (verify=True)."""
    so = _make_so(tmp_path)
    src_py = tmp_path / "add.py"
    src_py.write_text("def __pypto_compile(*a):\n    return '/tmp/x.o'\n", encoding="utf-8")

    setup_onnx_custom_op_so(so, kernel_py_paths=[("Add", "add", src_py)], verify=True)

    deployed = tmp_path / "op_kernel" / "add.py"
    assert deployed.is_file()


def test_setup_install_check_rejects_bad_snippet(tmp_path):
    """A shipped .py that does not expose a callable __pypto_compile fails at INSTALL, naming the path."""
    so = _make_so(tmp_path)
    src_py = tmp_path / "add.py"
    src_py.write_text("def not_the_entry():\n    return 1\n", encoding="utf-8")

    with pytest.raises(RuntimeError) as ei:
        setup_onnx_custom_op_so(so, kernel_py_paths=[("Add", "add", src_py)], verify=True)
    assert str(tmp_path / "op_kernel" / "add.py") in str(ei.value)


def test_setup_verify_false_skips_import_smoke(tmp_path):
    """verify=False copies + existence-checks but does NOT import (so an import-time raise is not hit),
    while verify=True DOES import and surfaces the failure as a RuntimeError."""
    so = _make_so(tmp_path)
    src_py = tmp_path / "add.py"
    # Raises at module import scope -> only the verify=True path (which imports it) fails.
    src_py.write_text("raise RuntimeError('boom at import')\n", encoding="utf-8")

    # verify=False: copied + existence-checked, never imported -> no raise.
    setup_onnx_custom_op_so(so, kernel_py_paths=[("Add", "add", src_py)], verify=False)
    assert (tmp_path / "op_kernel" / "add.py").is_file()

    # verify=True: import smoke runs and surfaces the import-time failure.
    with pytest.raises(RuntimeError):
        setup_onnx_custom_op_so(so, kernel_py_paths=[("Add", "add", src_py)], verify=True)


# ── the op package's recorded pypto version ──────────────────────────────────────────────────────────
# The install side of the same comparator the .so carries for atc, plus the writer that produces the
# file: build_so_from_model itself needs cmake + CANN, which is why the writer is a callable helper.


def test_build_writes_the_version_info_the_install_reads(tmp_path):
    # Producer and consumer, end to end and card-free: what the build writes is what the install accepts.
    build_dir = tmp_path / "build"
    build_dir.mkdir()
    info = _write_pypto_version_info(build_dir)
    assert info == build_dir / "pypto_version.info"
    assert info.read_text(encoding="utf-8").strip() == f"Version={installed_pypto_version()}"

    so = build_dir / "libcust_opapi.so"
    so.write_bytes(b"\x7fELF fake")
    setup_onnx_custom_op_so(so, kernel_py_paths=[])   # the file the writer produced satisfies the gate


def test_build_calls_the_version_info_writer():
    # The writer's CALL ships inside _build_so_from_nodes, which no card-free test can run, so a deleted
    # call line would pass every other test in this file.
    assert "_write_pypto_version_info(build_dir)" in inspect.getsource(_build_so_from_nodes)


def test_setup_refuses_a_package_without_version_info(tmp_path):
    # The install ENTRY POINT accepts any directory, so the refusal lives here, not in build.py.
    so = _make_so(tmp_path)
    (tmp_path / "pypto_version.info").unlink()
    with pytest.raises(RuntimeError, match="pypto_version.info"):
        setup_onnx_custom_op_so(so, kernel_py_paths=[])


def test_installed_package_is_the_root_the_atc_loader_reads(tmp_path):
    # The install -> atc seam, in one place: the directory setup treats as the package root (so_path's
    # parent) is the one the .so's loader derives from the SHIPPED .py as dirname(dirname(py_path)). If
    # either side moved by one level, the version file would be searched somewhere nothing writes it.
    so = _make_so(tmp_path)
    src_py = tmp_path / "add.py"
    src_py.write_text("def __pypto_compile(*a):\n    return '/tmp/x.o'\n", encoding="utf-8")
    setup_onnx_custom_op_so(so, kernel_py_paths=[("Add", "add", src_py)])
    deployed = tmp_path / "op_kernel" / "add.py"
    assert load_embedded_compile_module("Add", str(deployed)) is not None


def test_setup_refuses_a_newer_package_and_places_nothing(tmp_path):
    # The refusal names both versions and tells the reader what to do, in the wording the atc site also
    # uses. It joins the PRE-placement verify block, so it leaves the deploy tree untouched, not
    # half-copied.
    so = _make_so(tmp_path)
    _write_version_info(tmp_path, "999.0.0")
    with pytest.raises(RuntimeError) as excinfo:
        setup_onnx_custom_op_so(so, kernel_py_paths=[])
    message = str(excinfo.value)
    assert message.startswith("[SETUP] ")     # install-only prefix, the one per-site difference
    assert "999.0.0" in message
    assert "this pypto is" in message
    assert not (tmp_path / "op_proto").exists()
    assert not (tmp_path / "framework").exists()


def test_setup_refuses_a_package_carrying_a_plain_version_info(tmp_path):
    # GE reads a plain version.info on the opp entry and silently drops the whole package when it is out
    # of its ABI range. At install pypto owns the directory, so this is fatal rather than a warning.
    so = _make_so(tmp_path)
    (tmp_path / "version.info").write_text("Version=1.0\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="plain version.info"):
        setup_onnx_custom_op_so(so, kernel_py_paths=[])
    assert not (tmp_path / "op_proto").exists()


def test_so_existence_check_runs_before_the_version_gate(tmp_path):
    # Ordering guard: a wrong .so path is diagnosed as a missing .so. tmp_path carries no version file,
    # so a gate that moved above this check would raise RuntimeError about that instead.
    with pytest.raises(FileNotFoundError, match="custom-op .so not found"):
        setup_onnx_custom_op_so(tmp_path / "does_not_exist.so", kernel_py_paths=[])


def test_setup_warns_but_installs_on_an_uncomparable_version(tmp_path):
    # Uncomparable warns and continues at every site; only a MISSING file refuses.
    so = _make_so(tmp_path)
    _write_version_info(tmp_path, "unknown")
    with pytest.warns(UserWarning, match="records pypto_version.info"):
        setup_onnx_custom_op_so(so, kernel_py_paths=[])
    assert (tmp_path / "op_proto" / "libcust_opapi.so").is_file()
