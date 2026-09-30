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
"""Build_so-stage guard: declare an op, export it, build its ``.so`` and prove the library is real.

The stage under test is ``exported_custom_op_litenpu.deploy.build.build_so_from_model``, the python->C++ codegen
plus the cmake build that turns a declared ``ExportedCustomOp`` into the ``libcust_opapi.so`` GE dlopens.
The three assertions are the three ways that library can be hollow:

* the file exists and is non-empty, a cmake run that emits nothing still returns a path;
* ``ctypes.CDLL`` loads it, the load is ``RTLD_NOW``, so every undefined reference the generated
  executor makes into the CANN libraries (GE's annotated-args launch ABI in particular) must resolve.
  A library that links but cannot be dlopen'd is exactly what GE would reject at ATC time;
* the generated per-op executor symbol is exported, proof that the class codegen wrote is IN this
  ``.so``, not merely that some ``.so`` came out of the build directory. A control lookup of a name that
  was never generated must fail, so a symbol probe that answers "yes" to everything cannot pass this.

Needs cmake, a C++ compiler, the CANN GE headers + libraries, pybind11 and the Python development
headers. Every one of those is checked before the build runs, so a skip names the missing piece instead
of surfacing as a cmake error.
"""
from __future__ import annotations

import ctypes
import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import sys
import sysconfig

import onnx
import pytest
import torch
import torch.nn as nn

import pypto
from pypto.extensions.torch_custom_op_litenpu import ExportedCustomOp, OnnxSymbolicSpec, finalize_pending_ops
from pypto.extensions.torch_custom_op_litenpu.common.finalize import _reset_export_state
from pypto.extensions.torch_custom_op_litenpu.common.torch_op import exporting_scope
from pypto.extensions.torch_custom_op_litenpu.onnx.export import _qualify_op_type

# The repo-root conftest gates every collected item whose path carries no ``ut`` component on an NPU soc
# probe: it resolves the soc through ``torch_npu`` and ends the whole session when that import fails, then
# deselects items whose ``soc`` marker does not name the box's tag. This build test needs no NPU, so under
# pytest it skips at module scope wherever that probe cannot answer, a skip, never a session abort, and
# it carries both soc tags so a box that does have ``torch_npu`` keeps the tests selected. The ``__name__``
# guard keeps the module-level skip inside pytest: run bare, this file is ``__main__``, and there
# ``pytest.skip`` has no session to skip.
if __name__ != "__main__" and importlib.util.find_spec("torch_npu") is None:
    pytest.skip(
        "torch_npu is unavailable, so the soc probe the repo-root conftest applies to this path cannot "
        "resolve a soc version",
        allow_module_level=True,
    )

# ``cpp_codegen`` is the compile-cost marker: this module runs a cmake configure + build, a CANN link and a
# dlopen, so ``-m cpp_codegen`` selects it and ``-m "not cpp_codegen"`` excludes it. The prerequisite gate
# below stays the within-selection guard that skips on a missing toolchain piece.
pytestmark = [pytest.mark.cpp_codegen, pytest.mark.soc("910", "950")]

# <repo root>/tools, the ``exported_custom_op_litenpu.*`` import root. This file's own directory (``tests/st/``)
# carries no ``conftest.py`` of its own, so every ``exported_custom_op_litenpu.*`` import goes through
# ``_deploy_modules`` and nothing heavy runs at import time.
_TOOLS_ROOT = Path(__file__).resolve().parents[5] / "tools"

# The declaration under test: a two-input, one-output float16 op built by a factory kernel.
_QUALNAME = "pypto::so_load_add"
_SPEC = OnnxSymbolicSpec(op_type="SoLoadAdd", opset_version=12)
_OP_TYPE = _qualify_op_type(_SPEC.op_type)
_SHAPE = (1, 8, 1, 64)
_DTYPE = torch.float16

# GE headers the generated executor and the shared ``PtoCustomOp`` base include. ``annotated_args_context.h``
# is the discriminating one: CANN installs older than the 20260717-era GE do not ship it.
_REQUIRED_GE_HEADERS = (
    "graph/custom_op.h",
    "exe_graph/runtime/annotated_args_context.h",
    "exe_graph/runtime/op_compile_context.h",
    "exe_graph/runtime/runtime_attrs.h",
    "register/register.h",
)


# authoring functions (module scope: the tracer reads their source off disk)
def _so_load_add_factory(shapes, dtypes, attrs, soc_version, run_mode=pypto.RunMode.SIM):
    dtype = dtypes[0]

    @pypto.frontend.jit(codegen_options={"soc_version": soc_version},
                        runtime_options={"run_mode": run_mode})
    def so_load_add_inner(a: pypto.Tensor([...], dtype), b: pypto.Tensor([...], dtype),
                      out: pypto.Tensor([...], dtype)):
        pypto.set_vec_tile_shapes(1, 4, 1, 64)
        out.move(a + b)

    return so_load_add_inner


def _so_load_add_infer_shape(a: torch.Size, b: torch.Size) -> torch.Size:
    return a


def _so_load_add_infer_dtype(a: torch.dtype, b: torch.dtype) -> torch.dtype:
    return a


def _so_load_add_torch(a, b):
    return a + b


class _SoLoadAddModel(nn.Module):
    def forward(self, a, b):
        return torch.ops.pypto.so_load_add(a, b)


# environment gates
def _deploy_modules():
    """Return the ``(build, codegen)`` deploy modules, putting the repo-root ``tools`` dir on sys.path."""
    if str(_TOOLS_ROOT) not in sys.path:
        sys.path.insert(0, str(_TOOLS_ROOT))
    from exported_custom_op_litenpu.deploy import build as build_mod
    from exported_custom_op_litenpu.deploy import codegen as codegen_mod
    return build_mod, codegen_mod


def _missing_build_prerequisite() -> str | None:
    """Name the first missing build prerequisite, or None when the ``.so`` can be built here.

    Each check is a positive probe for a tool or file the build needs, so the caller skips on a proven
    gap rather than on a swallowed cmake failure.
    """
    build_mod, codegen_mod = _deploy_modules()

    if shutil.which("cmake") is None:
        return "cmake is not on PATH"

    cxx = (os.environ.get("CXX") or "g++").split()[0]
    if shutil.which(cxx) is None:
        return f"no C++ compiler ({cxx!r} is not on PATH; set $CXX)"

    ascend_home = Path(build_mod.default_ascend_home())
    include_dirs = [Path(d) for d in _extra_include_dirs()] + [ascend_home / "include"]
    missing_headers = [
        header for header in _REQUIRED_GE_HEADERS
        if not any((inc / header).is_file() for inc in include_dirs)
    ]
    if missing_headers:
        return (
            f"CANN GE headers missing from {[str(d) for d in include_dirs]}: "
            f"{', '.join(missing_headers)} (set $ASCEND_HOME_PATH to a CANN that ships the "
            "20260717-era GE, or point $PYPTO_EXTRA_INCLUDE_DIRS at those headers)"
        )

    lib_dir = ascend_home / "lib64"
    if not lib_dir.is_dir():
        return f"CANN link directory missing: {lib_dir}"

    if not (Path(sysconfig.get_paths()["include"]) / "Python.h").is_file():
        return f"Python development headers missing ({sysconfig.get_paths()['include']}/Python.h)"

    try:
        subprocess.run(
            [sys.executable, "-m", "pybind11", "--cmakedir"],
            check=True, capture_output=True, text=True,
        )
    except (subprocess.CalledProcessError, OSError):
        return f"pybind11 is not importable by {sys.executable} (cmake find_package(pybind11) would fail)"

    try:
        codegen_mod._resolve_nlohmann_include_dir()
    except FileNotFoundError as exc:
        # Third-party payload, so its absence is normally an environment gap, but the same resolve
        # also fails if the repo's own vendoring regresses, which is a defect, not an environment.
        return (
            f"nlohmann/json.hpp include root unavailable: {exc} (populate third_party_path or set "
            "$PYPTO_THIRD_PARTY_PATH; if both are in place this is a vendoring regression, not an "
            "environment gap)"
        )

    return None


def _extra_include_dirs() -> tuple[str, ...]:
    """The ``PYPTO_EXTRA_INCLUDE_DIRS`` entries the build prepends to the -I path (usually empty)."""
    return tuple(d for d in os.environ.get("PYPTO_EXTRA_INCLUDE_DIRS", "").split(os.pathsep) if d)


# the built artifact
def _export_fixture_model(work_dir: Path) -> Path:
    """Declare the op and ``torch.onnx.export`` it, returning the written ``.onnx`` path."""
    _reset_export_state()
    ExportedCustomOp(
        kernel=_so_load_add_factory,
        infer_shape=_so_load_add_infer_shape,
        infer_dtype=_so_load_add_infer_dtype,
        torch_defn=_so_load_add_torch,
        torch_op_qualname=_QUALNAME,
        onnx_spec=_SPEC,
    )
    finalize_pending_ops()
    a = torch.rand(_SHAPE, dtype=_DTYPE)
    b = torch.rand(_SHAPE, dtype=_DTYPE)
    path = work_dir / "so_load_add.onnx"
    with exporting_scope():
        torch.onnx.export(
            _SoLoadAddModel(), (a, b), str(path),
            input_names=["a", "b"], output_names=["y"],
            opset_version=_SPEC.opset_version, do_constant_folding=False, dynamo=False,
            custom_opsets={_SPEC.domain: _SPEC.domain_opset_version},
        )
    model = onnx.load(str(path))
    assert [n.op_type for n in model.graph.node] == [_OP_TYPE], (
        f"the fixture model must carry exactly the {_OP_TYPE} node, got "
        f"{[n.op_type for n in model.graph.node]}"
    )
    return path


@pytest.fixture(scope="module")
def built_so(tmp_path_factory):
    """The ``.so`` produced by the real ``build_so_from_model`` for the declared op."""
    reason = _missing_build_prerequisite()
    if reason is not None:
        pytest.skip(f"cannot build the operator .so here: {reason}")

    build_mod, _codegen_mod = _deploy_modules()
    work = tmp_path_factory.mktemp("so_load_build_so")
    try:
        model_path = _export_fixture_model(work)
        so_path, _kernel_py_paths = build_mod.build_so_from_model(
            model_path, out_dir=work / "op_lib", op_types=[_OP_TYPE],
        )
    finally:
        _reset_export_state()
    return Path(so_path)


@pytest.fixture(scope="module")
def loaded_so(built_so):
    """``built_so`` opened with ctypes. ``CDLL`` uses RTLD_NOW, so this resolves every CANN reference."""
    return ctypes.CDLL(str(built_so))


def _mangled_const_method(class_name: str, method_name: str) -> str:
    """Itanium C++ ABI mangling of ``<class_name>::<method_name>() const`` (no arguments)."""
    return f"_ZNK{len(class_name)}{class_name}{len(method_name)}{method_name}Ev"


def _exports(lib: ctypes.CDLL, symbol: str) -> bool:
    """Whether ``dlsym`` resolves *symbol* in *lib*."""
    try:
        lib[symbol]
    except AttributeError:
        return False
    return True


def test_built_so_exists_and_is_non_empty(built_so):
    # The build returns a path; that path has to be a real library file, not a stale or truncated stub.
    assert built_so.is_file(), f"build_so_from_model returned {built_so}, which is not a file"
    assert built_so.name == "libcust_opapi.so", (
        f"GE looks up the fixed filename libcust_opapi.so under ASCEND_CUSTOM_OPP_PATH, got {built_so.name}"
    )
    size = built_so.stat().st_size
    assert size > 0, f"{built_so} is empty"
    # An ELF header alone is 64 bytes; a library carrying an executor, the shared base and the pybind
    # infer wrappers is orders of magnitude larger, so a few-KB floor separates "built" from "emitted".
    assert size > 4096, f"{built_so} is only {size} bytes, no compiled executor can be in it"


def test_ctypes_cdll_loads_the_built_so(built_so):
    # RTLD_NOW: a successful load means every symbol the generated executor references (GE's
    # annotated-args launch ABI, the CANN registries, libpython) resolved against the CANN install.
    # A library that links but cannot be dlopen'd is exactly what GE rejects at ATC time.
    try:
        lib = ctypes.CDLL(str(built_so))
    except OSError as exc:
        raise AssertionError(f"dlopen({built_so}) failed: {exc}") from exc
    assert isinstance(lib, ctypes.CDLL)


def test_generated_executor_symbols_are_exported(loaded_so):
    # The per-op hooks codegen emits on the executor subclass. Their presence proves THIS op's generated
    # class is compiled into the library, a build that produced a library without the op would link and
    # load, and only this check would catch it.
    assert _OP_TYPE == "PyptoCustomOpSoLoadAdd"
    for method in ("GetKernelPyBasename", "GetCompileModuleStem"):
        symbol = _mangled_const_method(_OP_TYPE, method)
        assert _exports(loaded_so, symbol), (
            f"{_OP_TYPE}::{method}() is not exported by the built .so (looked up {symbol}); "
            "the generated executor class is not in this library"
        )


@pytest.mark.parametrize(
    "absent_class",
    [
        # A name nothing could ever emit: catches a dlsym probe that answers yes to everything.
        _OP_TYPE + "NeverGenerated",
        # A plausible sibling op, same namespace and same length family: catches a probe that matches
        # on the qualifier prefix or ignores the mangled length field instead of the whole symbol.
        "PyptoCustomOpSoLoadSub",
    ],
)
def test_symbol_probe_rejects_names_that_were_never_generated(loaded_so, absent_class):
    # Control for the check above. Without it, the symbol assertions would be vacuous.
    absent = _mangled_const_method(absent_class, "GetKernelPyBasename")
    assert not _exports(loaded_so, absent), (
        f"the built .so resolved {absent}, a symbol no codegen emitted for this model, the symbol "
        "probe does not discriminate, so it cannot witness the generated executor either"
    )
