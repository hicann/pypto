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
"""Helpers for generating a buildable cpp_sources project and compiling it into a ``.so``.

The codegen writes each op's TUs straight into the project's ``src/<OpType>/`` tree (one merged
``op_host/<stem>.cpp`` per op + an optional ``framework/onnx_plugin/<stem>_plugin.cpp``), with the shared
base class (``pto_custom_op.{h,cpp}``) written once under ``src/common/``. Op/plugin TUs reach nlohmann via
``#include <nlohmann/json.hpp>`` resolved off an include root the build places on the ``-I`` path. The low-level
primitives (``generate_bindings_cpp``, ``generate_cmakelists``) do pure template emission and are useful on
their own for callers that want a different build system or to post-process the generated CMakeLists. The
one-shot ``build_so_from_model`` composes everything end-to-end and shells out to ``cmake``.

``build_so_from_model`` discovers every pypto custom-op node in a model and builds a single combined ``.so``
containing all of their TUs. Pass ``op_types=["X"]`` to restrict the build to a subset (use this for
single-op builds, there is no separate single-node entry point).
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from typing import List, Optional, Sequence

from .embed import VERSION_INFO_BASENAME, installed_pypto_version

logger = logging.getLogger(__name__)

__all__ = (
    "generate_bindings_cpp",
    "generate_cmakelists",
    "default_ascend_home",
    "build_so_from_model",
    "DEFAULT_SO_NAME",
)


# GE loads a custom-op library by the fixed filename ``libcust_opapi.so`` from each
# ``ASCEND_CUSTOM_OPP_PATH`` entry, so that is the default, independent of the CMake target name.
DEFAULT_SO_NAME = "libcust_opapi.so"


# The ``domi::FrameworkType`` token bound into the ``framework/onnx_plugin/<stem>_plugin.cpp``
# REGISTER_CUSTOM_OP, keyed by the ``framework_kind`` a pypto node records at export. This mapping is a
# codegen concern and lives here in the export helpers, not in the pypto.extensions.torch_custom_op_litenpu core.
_FRAMEWORK_TYPE__ONNX = "ONNX"

_FRAMEWORK_KIND_TO_FRAMEWORK_TYPE = {
    "onnx": _FRAMEWORK_TYPE__ONNX,
}


# The pybind module name is token-pasted into ``PYBIND11_MODULE(<name>, m)`` (yielding
# ``PyInit_<name>``) and into the CMake project/target names, so it must be a plain C identifier.
_MODULE_NAME_ID_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _validate_module_name(module_name) -> None:
    """Ensure *module_name* is a safe program identifier (letters, digits, underscore; not digit-leading)."""
    if not isinstance(module_name, str) or not _MODULE_NAME_ID_RE.match(module_name):
        raise ValueError(
            "module_name must be a non-empty identifier (letters, digits, underscore; "
            f"must not start with a digit), got {module_name!r}"
        )


# bindings.cpp + CMakeLists.txt emission.


def generate_bindings_cpp(dst: Path, module_name: str) -> None:
    """Emit a stub pybind module so the generated cpp tree has a
    ``PYBIND11_MODULE`` entry point and can be linked as a Python extension.

    The generated TUs' top-level functions (``inferShape`` etc.) live in
    anonymous namespaces so they cannot be forward-declared or re-exposed
    from here. The .so's value is end-to-end compile + link validation
    against Ascend libs; the Python-visible surface is intentionally empty.
    """
    content = f'''#include <pybind11/pybind11.h>

namespace py = pybind11;

PYBIND11_MODULE({module_name}, m) {{
    m.doc() = "Generated PyPTO operator runtime module (stub bindings)";
}}
'''
    Path(dst).write_text(content, encoding="utf-8")
    logger.info("[GEN] %s", dst)


def generate_cmakelists(
    dst: Path,
    module_name: str,
    *,
    cpp_source_names: List[str],
    ascend_home: Path,
    debug: bool = False,
    so_name: str = DEFAULT_SO_NAME,
    extra_include_dirs: Sequence[str] = (),
) -> None:
    """Emit CMakeLists.txt wiring all generated cpp sources + Ascend includes/libs.

    *extra_include_dirs* are emitted as the FIRST entries of ``target_include_directories`` (so they
    shadow the CANN install's headers). Use this when the build needs newer GE headers than the CANN
    install ships, e.g. a CANN that predates the ge 20260717 annotated-args launch header
    (``annotated_args_context.h``; point at
    the GE repo's ``inc/graph_metadef/external``). Sourced from the
    ``PYPTO_EXTRA_INCLUDE_DIRS`` env var by ``build_so_from_model``; empty (the common case) is a no-op.

    *debug* sets the default of the ``PYPTO_DEBUG_SYMBOLS`` CMake option. When
    that option is ON the module is built so a crash inside it (e.g. a fault
    during ``dlopen``) yields a *resolvable* backtrace:

    - ``-g`` emits DWARF debug info (``addr2line`` / gdb give file:line);
    - ``-O0 -fno-omit-frame-pointer -fasynchronous-unwind-tables`` keep the call
      stack walkable so ``backtrace()`` recovers every frame;
    - ``-fvisibility=default`` keeps symbol names in ``.dynsym`` so
      ``backtrace_symbols()`` resolves frames to names instead of bare
      ``+0xoffset`` (explicit here; ``add_library(SHARED)`` already defaults to
      visible symbols);
    - ``-rdynamic`` / ``--export-dynamic`` keep those symbols dynamically
      exported (and the library unstripped).

    The flag defaults to OFF so production builds omit the debug info and the
    extra exported-symbol surface; pass ``-DPYPTO_DEBUG_SYMBOLS=ON`` at
    configure time to flip it per-build.
    """
    # *cpp_source_names* are project-relative paths (e.g. ``src/PyptoCustomOpAdd/op_host/pypto_custom_op_add.cpp``,
    # ``src/common/pto_custom_op.cpp``, ``src/common/bindings.cpp``), emitted verbatim into add_library.
    sources_block = "\n".join(f"    {name}" for name in cpp_source_names)

    # Extra include dirs (prepended so they shadow the CANN install), e.g. newer GE headers.
    extra_includes_block = "".join(f'    "{d}"\n' for d in extra_include_dirs)

    ascend_lib_dir = Path(ascend_home) / "lib64"
    ascend_inc_dir = Path(ascend_home) / "include"

    debug_default = "ON" if debug else "OFF"

    # Force the produced library filename to *so_name* exactly (default libcust_opapi.so),
    # regardless of the CMake target name. CMake would otherwise emit ``lib<target>.so``; we
    # clear the prefix and set OUTPUT_NAME/SUFFIX so e.g. "libcust_opapi.so" comes out verbatim.
    so_output_name, so_suffix = so_name[:-len(".so")], ".so"

    # Runtime rpath for libpython (baked as INSTALL_RPATH below). Derive it from the running
    # interpreter (sys.executable) rather than sysconfig's LIBDIR: in a multi-env conda setup
    # LIBDIR can resolve to a *different* env's lib (same 3.x, wrong prefix), so the .so would
    # rpath-load the wrong libpython3.x at runtime, an env mismatch that crashes inside CPython
    # (e.g. deep in _ctypes / PyModule_Create). sys.executable's own <prefix>/lib matches the
    # interpreter whose headers + Python3_EXECUTABLE configure this build.
    python3_lib_dir = str(Path(sys.executable).parent.parent / "lib")

    content = f'''cmake_minimum_required(VERSION 3.15)
project({module_name} LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(PYPTO_DEBUG_SYMBOLS
    "Build with debug symbols + exported symbols so crashes give resolvable backtraces"
    {debug_default})

# When debugging, default the whole project to a Debug build type (unless the
# caller set one explicitly) so every TU is compiled -g -O0 and the resulting
# library is left unstripped, addr2line/gdb then have DWARF + .symtab to use.
if(PYPTO_DEBUG_SYMBOLS AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug CACHE STRING "Debug build for resolvable backtraces" FORCE)
endif()

find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

set(ASCEND_HOME "{ascend_home}")
set(ASCEND_INCLUDE_DIR "{ascend_inc_dir}")
set(ASCEND_LIB_DIR "{ascend_lib_dir}")

add_library({module_name} SHARED
{sources_block}
)

target_include_directories({module_name} PRIVATE
{extra_includes_block}    ${{CMAKE_SOURCE_DIR}}/src/common
    ${{ASCEND_INCLUDE_DIR}}
    ${{Python3_INCLUDE_DIRS}}
    ${{pybind11_INCLUDE_DIRS}}
)
# Make pybind11 manage the GIL via CPython's official PyGILState API
# (PyGILState_Ensure/Release) instead of its default private PyThreadState.
# The embedded glue may run in a non-Python host where the interpreter is
# bootstrapped through the raw C-API (Py_InitializeEx + PyEval_SaveThread);
# pybind's default mode would create a *second*, private thread state per
# thread, inconsistent with CPython's official per-thread state, corrupting
# the import machinery (crash in PyList_New on the first import). Must be set
# for every pybind TU in the target (it changes gil_scoped_acquire's layout).
target_compile_definitions({module_name} PRIVATE PYBIND11_SIMPLE_GIL_MANAGEMENT)
# Match CANN's libstdc++ string ABI. The CANN runtime libs (libgraph.so etc.)
# are built with the old/COW std::string ABI (_GLIBCXX_USE_CXX11_ABI=0), so they
# export e.g. SinkableOpExecutionContext::SetCustomKernelName(std::basic_string)
# mangled as ...ESs. Modern gcc/conda default to the new SSO ABI (=1), which
# would mangle our *references* as ...NSt7__cxx11... and fail to resolve those
# std::string-taking methods at dlopen ("undefined symbol: ...__cxx11..."). The
# ascendc op CMake (npu_op_library) sets this for the same reason; we must too.
target_compile_definitions({module_name} PRIVATE _GLIBCXX_USE_CXX11_ABI=0)
target_link_directories({module_name} PRIVATE ${{ASCEND_LIB_DIR}})
target_link_libraries({module_name} PRIVATE
    -Wl,--whole-archive
    rt2_registry
    -Wl,--no-whole-archive
    graph
    register
    exe_graph
    lowering
    ascendcl
    unified_dlog
    ${{Python3_LIBRARIES}}
    ${{CMAKE_DL_LIBS}}
)
set_target_properties({module_name} PROPERTIES
    PREFIX ""
    OUTPUT_NAME "{so_output_name}"
    SUFFIX "{so_suffix}"
    INSTALL_RPATH "${{ASCEND_LIB_DIR}};{python3_lib_dir}"
    BUILD_WITH_INSTALL_RPATH TRUE
)

if(PYPTO_DEBUG_SYMBOLS)
    # Emit DWARF for addr2line/gdb and keep the stack walkable so backtrace()
    # recovers every frame. add_library(SHARED) already builds with default
    # symbol visibility (no pybind11 -fvisibility=hidden), so symbol names reach
    # .dynsym for backtrace_symbols() to resolve; -fvisibility=default keeps
    # that explicit.
    target_compile_options({module_name} PRIVATE
        -g -O0 -fno-omit-frame-pointer -fasynchronous-unwind-tables
        -fvisibility=default)
    target_link_options({module_name} PRIVATE -rdynamic -Wl,--export-dynamic)
    message(STATUS "{module_name}: PYPTO_DEBUG_SYMBOLS=ON (debuggable backtraces)")
endif()
'''
    Path(dst).write_text(content, encoding="utf-8")
    logger.info("[GEN] %s", dst)


# Ascend install discovery.


def default_ascend_home() -> Path:
    """Return the arch-specific Ascend install subtree containing ``include/`` + ``lib64/``.

    Honors ``$ASCEND_HOME_PATH``; falls back to ``/usr/local/Ascend/cann``. If the
    value is already arch-specific it is returned unchanged.
    """
    home = os.environ.get("ASCEND_HOME_PATH", "/usr/local/Ascend/cann")
    home_path = Path(home)
    for arch in ("aarch64-linux", "x86_64-linux"):
        sub = home_path / arch
        if (sub / "include").is_dir():
            return sub
    return home_path


# End-to-end build.


def _run_cmd(cmd, cwd=None, env=None):
    """Thin ``subprocess.run`` wrapper with check=True + a ``[RUN]`` log line."""
    logger.info("\n[RUN] %s", " ".join(map(str, cmd)))
    subprocess.run(cmd, cwd=cwd, env=env, check=True)


# Every key ``op_export_record`` carries. The exporter writes all of them unconditionally (both
# kernel-name keys, one of which is None for the mode that does not use it); only ``attrs`` is optional,
# and is added just for an op that declares attrs.
_REQUIRED_OP_EXPORT_RECORD_KEYS = (
    "infer_shape_name",
    "infer_dtype_name",
    "dtypes",
    "domain",
    "domain_opset_version",
    "framework_kind",
    "mode",
    "factory_signature",
    "declared_annotations",
    "create_kernel_name",
    "kernel_name",
)


def _validate_op_export_record(op_type: str, record) -> None:
    """Reject a malformed ``op_export_record`` payload with an error naming the op.

    Checks key PRESENCE only: the infer-hook names are in-contract when present and empty (an op
    without that hook records an empty name), so truthiness is not the criterion.
    """
    if not isinstance(record, dict):
        raise ValueError(
            f"op {op_type}: op_export_record must decode to a dict, "
            f"got {type(record).__name__}"
        )
    missing = [key for key in _REQUIRED_OP_EXPORT_RECORD_KEYS if key not in record]
    if missing:
        raise ValueError(
            f"op {op_type}: op_export_record is missing {missing}; re-export with a matching pypto"
        )


def _codegen_cpp_sources_into(node, op_dir):
    """Reconstruct the op's live functions from the node's ``kernel_compile_snippet`` +
    ``op_export_record``, run the C++ codegen, and write this op's files into *op_dir*
    (``src/<OpType>/``): the merged ``op_host/<stem>.cpp``, the onnx ``framework/onnx_plugin`` TU, and
    the dev-editable ``op_kernel/<stem>.py``.

    Returns ``(compiled_tu_paths, (op_type, stem, kernel_py_path))``. The ``.py`` is the single source
    of this op's Python, written verbatim and never compiled; setup ships it beside the deployed ``.so``.
    The shared base class is written once under ``src/common/`` by ``_write_shared_base_into``.
    """
    import importlib.util
    import json
    import sys
    import tempfile

    import torch

    from pypto.extensions.torch_custom_op_litenpu import (
        extract_kernel_compile_snippet,
        extract_op_export_record,
        extract_op_type,
    )

    from . import codegen
    from .cpp_naming import camel_case_to_snake_case

    snippet = extract_kernel_compile_snippet(node)
    record = json.loads(extract_op_export_record(node))
    op_type = extract_op_type(node)
    _validate_op_export_record(op_type, record)

    # Reconstruct the live capture-set functions by importing the self-contained snippet from a temp
    # .py file (import, NOT exec: pypto's jit re-reads kernel source via inspect.getsourcelines).
    tmp = Path(tempfile.mkdtemp(prefix="pypto_codegen_"))
    mod_name = f"_pypto_ksnip_{op_type}"
    mod_path = tmp / f"{mod_name}.py"
    mod_path.write_text(snippet, encoding="utf-8")
    sys.path.insert(0, str(tmp))
    try:
        spec = importlib.util.spec_from_file_location(mod_name, mod_path)
        ksnip = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = ksnip
        spec.loader.exec_module(ksnip)

        def _fn(role):
            name = record[role]
            if not name:
                return None
            fn = getattr(ksnip, name, None)
            if fn is None:
                raise ValueError(
                    f"op {op_type}: op_export_record names {role}={name!r}, which the embedded "
                    "kernel-compile snippet does not define"
                )
            return fn

        infer_shape_fn = _fn("infer_shape_name")
        infer_dtype_fn = _fn("infer_dtype_name")
        # Declared compute attrs (ordered list of {"name","type","default"} dicts), the ONLY carrier of
        # the attr contract from author to build-time codegen. Absent/empty for attr-free ops. Threaded
        # (as plain dicts) into both codegen entry points; the C++ emission derives REG_OP order,
        # ParseParam blocks, and the ExtractAttrs positional index from this one ordered list.
        attr_specs = record.get("attrs", [])
        # Compile-entry mode: "factory" (create_*_kernel drives @jit) or "jit_kernel" (a DIRECT,
        # already-built @pypto.frontend.jit kernel). The factory call shape is None in jit_kernel mode.
        mode = record["mode"]
        factory_signature = record["factory_signature"]
        declared_annotations = record["declared_annotations"]
        # DIRECT stores the top-level jit kernel under the distinct "kernel_name" key (the emitted def
        # name; "create_kernel_name" is None in this mode); factory stores the create_*_kernel under
        # "create_kernel_name". Resolve the right live handle per mode.
        if mode == "jit_kernel":
            create_fn = None
            jit_kernel_fn = _fn("kernel_name")
        else:
            create_fn = _fn("create_kernel_name")
            jit_kernel_fn = None
        dtypes = []
        for token in record["dtypes"]:
            dtype = getattr(torch, str(token).split(".")[-1], None)
            if not isinstance(dtype, torch.dtype):
                raise ValueError(
                    f"op {op_type}: op_export_record dtype token {token!r} does not name a "
                    "torch dtype"
                )
            dtypes.append(dtype)
        domain = record["domain"]
        domain_opset_version = record["domain_opset_version"]
        framework_kind = record["framework_kind"]
        try:
            framework_type = _FRAMEWORK_KIND_TO_FRAMEWORK_TYPE[framework_kind]
        except KeyError as exc:
            raise ValueError(f"Unknown framework_kind: {framework_kind!r}") from exc

        stem = camel_case_to_snake_case(op_type)
        # One merged TU per op (the thin executor + the GE OpDef REG_OP prototype) PLUS the op's
        # kernel-compile snippet, returned verbatim so the shipped .py is written with no
        # re-derivation / drift. The TU itself carries no Python source.
        tu_cpp, embedded_py_src = codegen._generate_op_tu_cpp_with_snippet(
            op_type,
            create_kernel_func=create_fn,
            jit_kernel_func=jit_kernel_fn,
            infer_shape_func=infer_shape_fn,
            infer_dtype_func=infer_dtype_fn,
            dtypes=dtypes,
            factory_signature=factory_signature,
            declared_annotations=declared_annotations,
            mode=mode,
            attr_specs=attr_specs,
        )
        plugin_cpp = codegen._generate_op_custom_plugin_cpp(
            op_type=op_type, framework_type=framework_type,
            domain=domain, domain_opset_version=domain_opset_version,
            attr_specs=attr_specs,
        )
        # The op's three files, written in place. The dev-editable snippet lands in a SEPARATE
        # op_kernel/ dir (matching the reference's op_kernel role convention; op_host/ stays C++-only)
        # and is written verbatim; it is NOT in the compiled ``sources`` list, so CMake never compiles it.
        tu_path = op_dir / "op_host" / f"{stem}.cpp"
        plugin_path = op_dir / "framework" / "onnx_plugin" / f"{stem}_plugin.cpp"
        kernel_py_path = op_dir / "op_kernel" / f"{stem}.py"
        for path, content in (
            (tu_path, tu_cpp), (plugin_path, plugin_cpp), (kernel_py_path, embedded_py_src),
        ):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
        # The build lists these TUs directly by their deterministic in-place paths, so no on-disk
        # manifest is emitted.
        sources = [tu_path, plugin_path]
        # The dev-editable snippet path (written above, NOT compiled), threaded out so deploy can ship it.
        kernel_py = (op_type, stem, kernel_py_path)
        return sources, kernel_py
    finally:
        sys.path.remove(str(tmp))
        sys.modules.pop(mod_name, None)
        shutil.rmtree(tmp, ignore_errors=True)


def _write_shared_base_into(common_dir: Path) -> None:
    """Write the shared base class + header under *common_dir*, one copy serves every op.

    ``pto_custom_op.h`` is the header reached by every op TU via the ``src/common`` include dir;
    ``pto_custom_op.cpp`` is a project-wide TU the caller adds to the CMake source list exactly once.
    """
    from . import codegen
    common_dir.mkdir(parents=True, exist_ok=True)
    (common_dir / "pto_custom_op.h").write_text(
        codegen._bundled_pto_custom_op_h_source(), encoding="utf-8"
    )
    (common_dir / "pto_custom_op.cpp").write_text(
        codegen._bundled_pto_custom_op_cpp_source(), encoding="utf-8"
    )


def _write_pypto_version_info(build_dir: Path) -> Path:
    """Record the BUILDING pypto's version in ``<build_dir>/pypto_version.info`` and return that path.

    *build_dir* doubles as the ``ASCEND_CUSTOM_OPP_PATH`` entry root (``so_path.parent``), so the stamp lands here.
    """
    info_path = build_dir / VERSION_INFO_BASENAME
    info_path.write_text(f"Version={installed_pypto_version()}\n", encoding="utf-8")
    return info_path


def _build_so_from_nodes(
    nodes: list,
    *,
    out_dir,
    module_name: str,
    ascend_home,
    clean: bool,
    debug: bool = False,
    so_name: str = DEFAULT_SO_NAME,
    extra_include_dirs: Sequence[str] = (),
) -> tuple:
    """Codegen every node's cpp TUs into ``<out_dir>/src/<OpType>/`` and build one combined ``.so``.

    Each node's merged op TU (+ optional onnx plugin) goes under that op's own ``src/<OpType>/`` subdir
    (the codegen returns the TU paths directly); the shared base class (``pto_custom_op.{h,cpp}``) is
    written once under ``src/common/`` and its ``.cpp`` compiled exactly once (avoiding duplicate-symbol
    link errors). Op TUs resolve ``#include "pto_custom_op.h"`` via the ``src/common`` include dir and
    ``#include <nlohmann/json.hpp>`` via the nlohmann include root prepended to the build ``-I`` path.

    Pre-conditions: *nodes* is non-empty, every node carries an ``op_type`` node-meta entry, and op_types
    are pairwise distinct (``find_pypto_nodes`` already dedupes).

    Returns ``(so_path, kernel_py_paths)``, the built ``.so`` and a list of ``(op_type, stem, src_py)``
    for each op's shipped-editable ``op_kernel/<stem>.py`` (for ``setup_onnx_custom_op_so``).
    """
    from pypto.extensions.torch_custom_op_litenpu import extract_op_type
    from pypto.extensions.torch_custom_op_litenpu.common.authoring import validate_op_type_identifier

    if not nodes:
        raise ValueError("_build_so_from_nodes called with empty nodes list")

    # The module name is token-pasted into the generated PYBIND11_MODULE and the CMake project/target
    # names, so it is validated here, before any file is written, rather than surfacing as a C++ or
    # CMake syntax error in a generated file the caller never wrote.
    _validate_module_name(module_name)

    resolved_ascend_home = (
        Path(ascend_home).resolve() if ascend_home else default_ascend_home()
    )
    if not (resolved_ascend_home / "include").is_dir():
        raise FileNotFoundError(
            f"Expected Ascend include dir under {resolved_ascend_home}/include; pass "
            "ascend_home pointing at the arch-specific subtree (e.g. $ASCEND_HOME_PATH/aarch64-linux)."
        )
    logger.info("[INFO] Ascend home: %s", resolved_ascend_home)

    project_dir = Path(out_dir).resolve()
    if clean and project_dir.exists():
        logger.info("[CLEAN] Removing %s", project_dir)
        shutil.rmtree(project_dir)

    src_dir = project_dir / "src"
    common_dir = src_dir / "common"
    project_dir.mkdir(parents=True, exist_ok=True)
    src_dir.mkdir(parents=True, exist_ok=True)

    # Shared base class + headers, one copy under src/common/ serves every op.
    _write_shared_base_into(common_dir)

    # Per-op codegen straight into src/<OpType>/; collect each op's TU(s) as project-relative sources
    # and each op's dev-editable kernel .py path (written but NOT compiled, shipped by setup).
    cpp_source_names: list[str] = []
    kernel_py_paths: list = []
    for node in nodes:
        op_type = extract_op_type(node)
        # The op_type is a path component here, so it is validated before the directory is created.
        validate_op_type_identifier(op_type)
        op_dir = src_dir / op_type
        op_dir.mkdir(parents=True, exist_ok=True)
        sources, kernel_py = _codegen_cpp_sources_into(node, op_dir)
        for p in sources:
            cpp_source_names.append(str(p.relative_to(project_dir)))
        kernel_py_paths.append(kernel_py)

    # Everything shipped is keyed by the snake_case stem, the single flat ``<opp_root>/op_kernel/
    # <stem>.py`` deploy slot and the executor's own by-stem resolution, so distinct op_types that
    # collapse onto one stem would cross-wire their kernels.
    stem_owners: dict = {}
    for op_type, stem, _src_py in kernel_py_paths:
        stem_owners.setdefault(stem, []).append(op_type)
    colliding = sorted(
        (stem, sorted(ops)) for stem, ops in stem_owners.items() if len(ops) > 1
    )
    if colliding:
        detail = "; ".join(f"{stem}: {ops}" for stem, ops in colliding)
        raise ValueError(
            f"op_types collide on their snake_case kernel-module stem ({detail}), rename one of "
            "each colliding pair so every op maps to a unique op_kernel/<stem>.py"
        )

    # The shared base-class TU is project-wide, compiled exactly once.
    cpp_source_names.append(str((common_dir / "pto_custom_op.cpp").relative_to(project_dir)))

    # The pybind module entry point is project-wide (one per .so), so it lives under src/common/
    # alongside the shared base TU, keeping the top level to per-op subdirs + common/.
    generate_bindings_cpp(common_dir / "bindings.cpp", module_name)
    cpp_source_names.append(str((common_dir / "bindings.cpp").relative_to(project_dir)))

    # Prepend the nlohmann include root so op/plugin TUs resolve ``<nlohmann/json.hpp>`` off the ``-I``
    # with no env set; any caller-supplied dirs follow right behind it.
    from . import codegen
    nlohmann_root = str(codegen._resolve_nlohmann_include_dir())
    extra_include_dirs = (nlohmann_root, *tuple(extra_include_dirs))

    generate_cmakelists(
        project_dir / "CMakeLists.txt",
        module_name,
        cpp_source_names=cpp_source_names,
        ascend_home=resolved_ascend_home,
        debug=debug,
        so_name=so_name,
        extra_include_dirs=extra_include_dirs,
    )

    build_dir = project_dir / "build"
    # PYPTO_DEBUG_SYMBOLS is passed on every configure: ``option()`` only seeds a cache entry that is
    # not already set, so over a pre-existing build dir the command-line ``-D`` is what makes *debug*
    # take effect.
    _run_cmd([
        "cmake",
        "-S", str(project_dir),
        "-B", str(build_dir),
        f"-DPython3_EXECUTABLE={sys.executable}",
        # pybind11's cmake config dir, for the current interpreter.
        "-Dpybind11_DIR=" + subprocess.check_output(
            [sys.executable, "-m", "pybind11", "--cmakedir"], text=True).strip(),
        f"-DPYPTO_DEBUG_SYMBOLS={'ON' if debug else 'OFF'}",
    ])
    _run_cmd(["cmake", "--build", str(build_dir), "-j"])

    # The exact so_name, which OUTPUT_NAME/SUFFIX above forced.
    so_files = sorted(build_dir.glob(so_name))
    if not so_files:
        raise FileNotFoundError(f"Build finished but no .so found in {build_dir}")

    # The stamp names the pypto that GENERATED these sources, not whoever later recompiles them: a bare
    # ``cmake --build`` over this tree re-compiles that same generated code, so it correctly leaves the
    # file alone.
    _write_pypto_version_info(build_dir)

    logger.info("\n[OK] Built shared object(s):")
    for so in so_files:
        logger.info("  %s", so)
    return so_files[0], kernel_py_paths


def build_so_from_model(
    model,
    *,
    out_dir,
    module_name: str = "op_kernel_lib",
    ascend_home=None,
    clean: bool = False,
    op_types: Optional[Sequence[str]] = None,
    debug: bool = False,
    so_name: str = DEFAULT_SO_NAME,
    extra_include_dirs: Optional[Sequence[str]] = None,
) -> tuple:
    """Discover every pypto custom-op node in *model* and build one combined ``.so``.

    *model* may be a loaded model object (``onnx.ModelProto``) OR a path-like to
    a ``.onnx`` file (in which case ``exported_custom_op_litenpu.common.discovery.load_model`` is
    called internally).

    *op_types* restricts the build to a subset: pass
    ``op_types=["MyOp"]`` to build a single-op ``.so``. Omit to include
    every custom-op node found in the model (deduped by ``op_type``).

    Composes the lower-level steps: ``find_pypto_nodes`` → the per-node package-version gate
    (``node_gate.check_pypto_node_package_version``, refusing a node exported by a newer pypto than this
    one) → per-op codegen into ``src/<OpType>/`` (``_codegen_cpp_sources_into``) → shared base into
    ``src/common/`` → ``generate_bindings_cpp`` → ``generate_cmakelists`` → ``cmake``.

    Parameters
    ----------
    model
        Loaded model object or path to a ``.onnx`` file.
    out_dir
        Project directory for the build tree. Created if missing.
    module_name
        Python extension module / CMake project + pybind11 target name. NOT the
        produced filename, see *so_name*.
    so_name
        Filename of the produced library. Defaults to ``libcust_opapi.so``, the
        fixed name GE looks for under ``ASCEND_CUSTOM_OPP_PATH``. Override only if
        the consumer expects a different name.
    ascend_home
        Override the Ascend arch-subdir (containing ``include/`` + ``lib64/``).
        Defaults to ``default_ascend_home``.
    clean
        If True, remove *out_dir* before regenerating.
    op_types
        Optional subset of custom-op type names to include in the build.
    debug
        If True, set ``PYPTO_DEBUG_SYMBOLS`` ON so the produced ``.so``
        carries debug info and exported symbols and a crash inside it yields a
        resolvable backtrace. Defaults to False (optimized, stripped builds).
        The value is passed to every ``cmake`` configure, so it also applies over
        an existing *out_dir*; a True -> False flip additionally needs
        ``clean=True`` because the Debug build type is cached.

    Returns
    -------
    tuple
        ``(so_path, kernel_py_paths)``, the produced combined library file in the build directory
        (e.g. ``libcust_opapi.so``) and a list of ``(op_type, stem, src_py)`` for each op's shipped
        dev-editable ``op_kernel/<stem>.py`` (forward to ``setup_onnx_custom_op_so`` so it is
        placed beside the deployed ``.so``).
    """
    from exported_custom_op_litenpu.common.discovery import find_pypto_nodes, load_model
    from pypto.extensions.torch_custom_op_litenpu import extract_op_type

    from .node_gate import check_pypto_node_package_version

    if isinstance(model, (str, Path)):
        model = load_model(model)
    nodes = find_pypto_nodes(model, op_types=op_types)
    if not nodes:
        raise ValueError(
            f"No pypto custom-op nodes found in model "
            f"(filter op_types={op_types!r})"
        )
    for node in nodes:
        check_pypto_node_package_version(node)
    discovered = [extract_op_type(n) for n in nodes]
    logger.info("[INFO] Building combined .so for ops: %s", discovered)
    # Default extra includes from PYPTO_EXTRA_INCLUDE_DIRS (colon-separated), usually empty/unset.
    if extra_include_dirs is None:
        _env = os.environ.get("PYPTO_EXTRA_INCLUDE_DIRS", "")
        extra_include_dirs = tuple(d for d in _env.split(os.pathsep) if d)
    return _build_so_from_nodes(
        nodes,
        out_dir=out_dir,
        module_name=module_name,
        ascend_home=ascend_home,
        clean=clean,
        debug=debug,
        so_name=so_name,
        extra_include_dirs=extra_include_dirs,
    )
