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
"""Unit tests for the ``debug`` knob on ``generate_cmakelists``.

These are pure text-emission checks (no cmake / no compiler), so they run
everywhere.
"""

from __future__ import annotations

# Running this file directly, with no conftest, needs the tools root on sys.path
# for the `exported_custom_op_litenpu.*` imports below. Redundant under pytest: the sibling conftest.py
# already does this, and the guard there makes it a no-op.
import pathlib as _pl
import sys as _sys

_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[5] / "tools"))

from pathlib import Path

from exported_custom_op_litenpu.deploy.build import DEFAULT_SO_NAME, generate_cmakelists


def _emit(tmp_path: Path, *, debug: bool = False, so_name: str | None = None) -> str:
    dst = tmp_path / "CMakeLists.txt"
    kwargs = {} if so_name is None else {"so_name": so_name}
    generate_cmakelists(
        dst,
        "op_kernel_lib",
        cpp_source_names=["src/PyptoCustomOpFoo/op_host/pypto_custom_op_foo.cpp", "src/common/pto_custom_op.cpp"],
        ascend_home=Path("/opt/ascend/x86_64-linux"),
        debug=debug,
        **kwargs,
    )
    return dst.read_text(encoding="utf-8")


def test_cmakelists_default_so_name_is_libcust_opapi(tmp_path):
    # The produced filename defaults to libcust_opapi.so (GE's fixed lookup name),
    # forced via PREFIX""/OUTPUT_NAME/SUFFIX so it does not track the CMake target name.
    assert DEFAULT_SO_NAME == "libcust_opapi.so"
    txt = _emit(tmp_path)
    assert 'PREFIX ""' in txt
    assert 'OUTPUT_NAME "libcust_opapi"' in txt
    assert 'SUFFIX ".so"' in txt
    # The CMake target identifier is still op_kernel_lib (decoupled from the filename).
    assert "add_library(op_kernel_lib SHARED" in txt


def test_cmakelists_so_name_override_is_honored(tmp_path):
    txt = _emit(tmp_path, so_name="libfoo.so")
    assert 'OUTPUT_NAME "libfoo"' in txt
    assert 'SUFFIX ".so"' in txt
    assert 'OUTPUT_NAME "libcust_opapi"' not in txt


def test_cmakelists_always_declares_debug_option(tmp_path):
    # The option exists regardless of the default so callers can flip it at
    # configure time with -DPYPTO_DEBUG_SYMBOLS=ON/OFF.
    assert "option(PYPTO_DEBUG_SYMBOLS" in _emit(tmp_path, debug=False)
    assert "option(PYPTO_DEBUG_SYMBOLS" in _emit(tmp_path, debug=True)


def test_cmakelists_debug_default_tracks_flag(tmp_path):
    off = _emit(tmp_path, debug=False)
    on = _emit(tmp_path, debug=True)
    # The option's default literal is the only thing the flag changes.
    assert "    OFF)" in off and "    ON)" not in off
    assert "    ON)" in on and "    OFF)" not in on


def test_cmakelists_debug_block_has_symbol_and_dwarf_flags(tmp_path):
    on = _emit(tmp_path, debug=True)
    # Guarded so production (default OFF) builds stay optimized + stripped.
    assert "if(PYPTO_DEBUG_SYMBOLS)" in on
    # DWARF + walkable stack.
    assert "-g" in on
    assert "-fno-omit-frame-pointer" in on
    # Undo pybind11's -fvisibility=hidden so symbol names reach .dynsym; this
    # is what turns backtrace_symbols() output from "+0xoffset" into names.
    assert "-fvisibility=default" in on
    assert "-rdynamic" in on or "--export-dynamic" in on
    # Avoid pybind11's post-build strip (which would drop .debug_*/.symtab) by
    # forcing a debug build type before the module is declared.
    assert "set(CMAKE_BUILD_TYPE Debug" in on


def test_cmakelists_debug_flags_are_gated_and_default_off(tmp_path):
    # The debug flags live inside an ``if(PYPTO_DEBUG_SYMBOLS)`` block, so with
    # the default OFF (debug=False) cmake never applies them, production builds
    # stay optimized + stripped, yet -DPYPTO_DEBUG_SYMBOLS=ON still flips them.
    off = _emit(tmp_path, debug=False)
    idx_if = off.index("if(PYPTO_DEBUG_SYMBOLS)")
    idx_opt = off.index("option(PYPTO_DEBUG_SYMBOLS")
    # Both the gating and the OFF default are present.
    assert idx_opt < idx_if
    assert "    OFF)" in off
    # The CMAKE_BUILD_TYPE force is itself guarded by the option, not unconditional.
    assert "if(PYPTO_DEBUG_SYMBOLS AND NOT CMAKE_BUILD_TYPE)" in off
