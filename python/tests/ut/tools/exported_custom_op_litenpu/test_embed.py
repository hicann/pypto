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
"""Tests for the kernel-module loader (exported_custom_op_litenpu.deploy.embed).

The shipped ``op_kernel/<stem>.py`` is the single source of truth: the loader imports it BY PATH, so
pypto's jit parser (``inspect.getsourcelines``) can read the source of a NESTED function (the @jit
kernel inside the factory) straight off disk. Loading is strict, an empty or absent path raises.
"""

# Running this file directly, with no conftest, needs the tools root on sys.path
# for the `exported_custom_op_litenpu.*` imports below. Redundant under pytest: the sibling conftest.py
# already does this, and the guard there makes it a no-op.
import pathlib as _pl
import sys as _sys

_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[5] / "tools"))

import inspect
import os
import textwrap

from exported_custom_op_litenpu.deploy import embed
import pytest

_SRC = textwrap.dedent(
    """
    def add_kernel_body(a, b):
        return a + b

    def create_add_kernel(shape, dtype, soc):
        def add_kernel(input0, input1, output):
            output.move(add_kernel_body(input0, input1))
        return add_kernel

    def __pypto_compile(in_shapes, in_dtypes, attrs, soc_version):
        return "/tmp/fake_kernel.o"
    """
).strip("\n")


@pytest.fixture(autouse=True)
def _reset():
    # Isolate the module-level memo between tests.
    embed._loaded.clear()
    yield
    embed._loaded.clear()


def _nested_kernel(mod):
    """Return the nested @jit-style kernel the parser would re-read via getsourcelines."""
    return mod.create_add_kernel((8, 64), "float16", "Kirin9030")


def _shipped(tmp_path, *, version=None, write_version=True):
    """Lay out a real op package under *tmp_path* and return its shipped ``op_kernel/add_kernel.py``.

    The root is the ``.py``'s ``parent.parent``, so a flat layout would share one version file across tests.
    """
    op_kernel = tmp_path / "op_kernel"
    op_kernel.mkdir(exist_ok=True)
    shipped = op_kernel / "add_kernel.py"
    shipped.write_text(_SRC, encoding="utf-8")
    if write_version:
        recorded = embed.installed_pypto_version() if version is None else version
        (tmp_path / "pypto_version.info").write_text(f"Version={recorded}\n", encoding="utf-8")
    return shipped


def test_imports_shipped_file(tmp_path):
    # The shipped .py is imported BY PATH (dev-editable snippet).
    shipped = _shipped(tmp_path)
    mod = embed.load_embedded_compile_module("AddOp", str(shipped))
    assert mod.__file__ == str(shipped)
    assert mod.__pypto_compile((), (), {}, "Kirin9030") == "/tmp/fake_kernel.o"
    # A real file on disk => getsourcelines reads the nested kernel directly.
    assert "def add_kernel(" in "".join(inspect.getsourcelines(_nested_kernel(mod))[0])


def test_missing_path_raises(tmp_path):
    # Strict: an ABSENT resolved path is a FATAL FileNotFoundError, the shipped .py is REQUIRED.
    missing = tmp_path / "does_not_exist_kernel.py"
    with pytest.raises(FileNotFoundError):
        embed.load_embedded_compile_module("AddOp", str(missing))


def test_empty_path_raises():
    # No path passed (the C++ resolved nothing) -> FATAL, not a silent fallback.
    with pytest.raises(FileNotFoundError):
        embed.load_embedded_compile_module("AddOp", "")


def test_memoized_per_stem_and_mtime(tmp_path):
    # One load per (stem, mtime), reused across compiles for different runtime shapes.
    shipped = _shipped(tmp_path)
    a = embed.load_embedded_compile_module("AddOp", str(shipped))
    b = embed.load_embedded_compile_module("AddOp", str(shipped))
    assert a is b


def test_mtime_reload_picks_up_edit(tmp_path):
    # The dev-editability contract: an EDITED shipped .py re-imports on the next call (mtime-keyed memo).
    shipped = _shipped(tmp_path)
    os.utime(shipped, (1_000_000, 1_000_000))
    first = embed.load_embedded_compile_module("AddOp", str(shipped))

    # Edit the file (change the compile return) and bump its mtime -> a fresh key -> fresh import.
    edited = _SRC.replace("/tmp/fake_kernel.o", "/tmp/edited_kernel.o")
    shipped.write_text(edited, encoding="utf-8")
    os.utime(shipped, (2_000_000, 2_000_000))
    reloaded = embed.load_embedded_compile_module("AddOp", str(shipped))
    assert reloaded is not first
    assert reloaded.__pypto_compile((), (), {}, "Kirin9030") == "/tmp/edited_kernel.o"


# ── the op package's recorded pypto version ──────────────────────────────────────────────────────────
# The atc-side gate: this same code, spliced into the .so, runs inside GE.


def test_missing_version_info_refuses(tmp_path):
    # Every build/ a pypto build produces carries the file, so its absence means the package was not
    # built by this vintage. "[SETUP] " is an install-time prefix and must not reach a GE log.
    shipped = _shipped(tmp_path, write_version=False)
    with pytest.raises(RuntimeError, match="pypto_version.info") as excinfo:
        embed.load_embedded_compile_module("AddOp", str(shipped))
    assert "[SETUP]" not in str(excinfo.value)


def test_newer_recorded_version_refuses_on_every_call(tmp_path):
    # An older pypto cannot build a newer package's kernels. Nothing caches the refusal: the gate runs
    # above the module memo, so the second compile of the same package is refused again. The wording is
    # the shared one, and the install-only "[SETUP] " prefix must never reach a GE log.
    shipped = _shipped(tmp_path, version="999.0.0")
    for _ in range(2):
        with pytest.raises(RuntimeError) as excinfo:
            embed.load_embedded_compile_module("AddOp", str(shipped))
    message = str(excinfo.value)
    assert "999.0.0" in message
    assert "this pypto is" in message
    assert "[SETUP]" not in message


def test_version_check_runs_before_the_memo(tmp_path):
    # The gate sits above the (stem, mtime) memo, so a module cached by an earlier compile cannot carry
    # a package past a version file that has since become unacceptable.
    shipped = _shipped(tmp_path)
    embed.load_embedded_compile_module("AddOp", str(shipped))
    (tmp_path / "pypto_version.info").write_text("Version=999.0.0\n", encoding="utf-8")
    with pytest.raises(RuntimeError):
        embed.load_embedded_compile_module("AddOp", str(shipped))


def test_uncomparable_recorded_version_warns_and_loads(tmp_path):
    # Uncomparable is not missing: it warns and the compile continues, one behaviour at every site.
    shipped = _shipped(tmp_path, version="nightly")
    with pytest.warns(UserWarning, match="records pypto_version.info"):
        assert embed.load_embedded_compile_module("AddOp", str(shipped)) is not None


def test_stray_plain_version_info_warns_at_atc(tmp_path):
    # At atc the root is user-pointed and a plain version.info may be legitimate, so this warns and the
    # compile continues; the install leg refuses (test_cpp_setup.py). Rationale: VERSION_INFO_BASENAME.
    shipped = _shipped(tmp_path)
    (tmp_path / "version.info").write_text("Version=1.0\n", encoding="utf-8")
    with pytest.warns(UserWarning, match="plain version.info"):
        assert embed.load_embedded_compile_module("AddOp", str(shipped)) is not None


def test_stray_version_info_does_not_mask_the_missing_file_diagnosis(tmp_path):
    # Ordering guard: the guard sits BELOW the missing-pypto_version.info raise, so a root carrying only
    # the stray file is still diagnosed as the stale package it is. Both messages mention
    # "pypto_version.info", so the distinguishing substring is what this asserts.
    shipped = _shipped(tmp_path, write_version=False)
    (tmp_path / "version.info").write_text("Version=1.0\n", encoding="utf-8")
    with pytest.raises(RuntimeError) as excinfo:
        embed.load_embedded_compile_module("AddOp", str(shipped))
    assert "was not produced by a pypto build" in str(excinfo.value)
    assert "plain version.info" not in str(excinfo.value)


def test_uncomparable_local_version_warns(tmp_path, monkeypatch):
    # The LOCAL side is uncomparable on a pypto-less atc box, which setup explicitly supports.
    monkeypatch.setattr(embed, "installed_pypto_version", lambda: "unknown")
    shipped = _shipped(tmp_path, version="0.2.1")
    with pytest.warns(UserWarning, match="this pypto reports version"):
        embed.load_embedded_compile_module("AddOp", str(shipped))


# ── the comparator itself ────────────────────────────────────────────────────────────────────────────

_VERSION_TABLE = {
    "0.2.1": (0, 2, 1),
    "1": (1,),
    "1.0": (1, 0),
    "0.02": (0, 2),
    "0.2": (0, 2),
    "0.2.1.dev0+g1234": (0, 2, 1),
    "0.3.0rc1": (0, 3, 0),
    "1.0-beta": (1, 0),         # COMPARABLE: the leading release segment parses, the suffix is ignored
    "1.0.post2": (1, 0),
    "2!1.0": (2,),              # an epoch is not a release segment, so only the leading 2 is read
    "unknown": None,
    "nightly": None,
    "v1": None,
    "": None,
}


@pytest.mark.parametrize("raw,expected", sorted(_VERSION_TABLE.items()))
def test_parses_the_leading_release_segment(raw, expected):
    # The parse table: only the leading release segment is read, so a suffix or an epoch never makes a
    # version uncomparable, and a value with no leading number is None.
    assert embed.parse_version_tuple(raw) == expected


@pytest.mark.parametrize("shorter,longer", [("1", "1.0"), ("1.0", "1.0.0"), ("0.2", "0.02")])
def test_shorter_version_is_zero_padded_not_older(shorter, longer):
    # A raw tuple compare would make (1,) older than (1, 0) and refuse a package built by an identical
    # pypto that spelled its version with fewer components.
    a = embed.parse_version_tuple(shorter)
    b = embed.parse_version_tuple(longer)
    assert not embed.is_newer(recorded=a, local=b)
    assert not embed.is_newer(recorded=b, local=a)
