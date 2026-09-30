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
"""Optional compile-and-run tests for export C++ codegen.

Uses a single ``_compile_and_run`` with ``CppCompileOptions``:

- ``embed_python=True``: infer_shape TU (pybind11 + Python embed).
- default (``embed_python=False``): TU fragments that do not use pybind / embedded Python.
- custom-executor compile tests use ``embed_python=True`` because the generated executor embeds
  pybind and runs Python for infer_shape/infer_dtype and JIT kernel compile.
"""

from __future__ import annotations

# Running this file directly, with no conftest, needs the tools root on sys.path
# for the `exported_custom_op_litenpu.*` imports below. Redundant under pytest: the sibling conftest.py
# already does this, and the guard there makes it a no-op.
import pathlib as _pl
import sys as _sys

_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[5] / "tools"))

from dataclasses import dataclass
import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Callable, Sequence, Tuple

from exported_custom_op_litenpu.deploy import codegen as cpp_mod
import pytest
import torch


@dataclass(frozen=True)
class CppCompileOptions:
    """Options for compiling a generated C++ TU in tests."""

    embed_python: bool = False
    """If True, add pybind11 + Python embed link flags (infer_shape TU)."""
    extra_includes: tuple[str, ...] = ()
    """Extra ``-I`` directories (after fixtures include)."""


_EXPORT_TEST_DIR = Path(__file__).resolve().parent
_FIXTURES_DIR = _EXPORT_TEST_DIR / "fixtures"

# The ``domi::FrameworkType`` token the generated onnx plugin registers under. Held locally so
# these checks collect without importing the build module.
_FRAMEWORK_TYPE__ONNX = "ONNX"


def _load_samples_module(unique_name: str, filename: str):
    """Load a sibling ``*.py`` sample module by path (works without a parent package)."""
    path = _EXPORT_TEST_DIR / filename
    spec = importlib.util.spec_from_file_location(unique_name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


infer_shape_samples = _load_samples_module("infer_shape_samples_ut_compile", "infer_shape_samples.py")
kernel_compile_samples = _load_samples_module("kernel_compile_samples_ut_compile", "cpp_kernel_compile_samples.py")
codegen_test_helpers = _load_samples_module("codegen_test_helpers_ut_compile", "codegen_test_helpers.py")


def _executor_kwargs(dtypes=None):
    """Kwargs for the new ``_generate_custom_executor_cpp`` signature using the real kernel samples."""
    return dict(
        create_kernel_func=kernel_compile_samples.create_add_kernel,
        infer_shape_func=kernel_compile_samples.add_infer_shape,
        infer_dtype_func=kernel_compile_samples.add_infer_dtype,
        kernel_body_func=kernel_compile_samples.add_kernel_body,
        dtypes=dtypes if dtypes is not None else [torch.float16, torch.float16],
    )


def _find_cxx() -> str | None:
    for key in (os.environ.get("CXX"), "g++", "clang++"):
        if key and shutil.which(key.split()[0]):
            return key
    return None


def _pybind_include() -> str | None:
    try:
        import pybind11

        return str(pybind11.get_include())
    except ImportError:
        return None


def _python_config_cmd() -> str | None:
    """Locate the ``python*-config`` matching the running interpreter.

    Prefer the one beside ``sys.executable`` so the embedded binary uses the same site-packages (which
    has ``torch``, which the embedded pybind wrapper imports to construct ``torch.Size`` instances);
    fall back to whatever ``python3-config`` is on PATH.
    """
    py_ver = sys.version_info
    name = f"python{py_ver.major}.{py_ver.minor}-config"
    beside = Path(sys.executable).parent / name
    if beside.is_file():
        return str(beside)
    return shutil.which(name) or shutil.which("python3-config")


def _python_embed_link_flags() -> list[str] | None:
    cfg = _python_config_cmd()
    if not cfg:
        return None
    try:
        out = subprocess.check_output(
            [cfg, "--ldflags", "--embed"], stderr=subprocess.DEVNULL, text=True,
        )
        return out.split()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _python_cflags() -> list[str]:
    cfg = _python_config_cmd()
    if not cfg:
        return []
    try:
        out = subprocess.check_output([cfg, "--cflags"], stderr=subprocess.DEVNULL, text=True)
        return out.split()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return []


def _conda_link_flags() -> list[str]:
    """Return extra linker flags for conda Python lib directory, if present."""
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if not conda_prefix:
        return []
    lib_dir = Path(conda_prefix) / "lib"
    if not lib_dir.is_dir():
        return []
    return [f"-L{lib_dir}", f"-Wl,-rpath,{lib_dir}"]


def _cxx_or_skip(opts: CppCompileOptions):
    """The compiler plus the pybind include, or a skip when either prerequisite is missing."""
    cxx = _find_cxx()
    if not cxx:
        pytest.skip("No C++ compiler")
    pybind_inc = None
    if opts.embed_python:
        pybind_inc = _pybind_include()
        if not pybind_inc:
            pytest.skip("embed_python=True requires pybind11 include path")
    return cxx, pybind_inc


def _compile_cmd(cxx, src: Path, opts: CppCompileOptions, pybind_inc, *, syntax_only: bool) -> list:
    """The shared g++ command: standard flags, the fixtures include, the caller's includes, and the
    pybind include + python cflags when embedding."""
    cmd = [cxx, "-std=c++17", "-O0"]
    if syntax_only:
        cmd.append("-fsyntax-only")
    cmd.append(f"-I{_FIXTURES_DIR}")
    cmd.extend(f"-I{inc}" for inc in opts.extra_includes)
    if opts.embed_python:
        cmd.append(f"-I{pybind_inc}")
        cmd.extend(_python_cflags())
    cmd.append(str(src))
    return cmd


def _run_compile(cmd: list, label: str) -> None:
    """Run the compile, failing the test with the full command line and the compiler output."""
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        pytest.fail(
            f"Compile failed ({label}):\n" + " ".join(cmd) + "\n" + (e.stderr or "") + "\n" + (e.stdout or "")
        )


def _compile_and_run(cpp_source: str, *, options: CppCompileOptions | None = None) -> None:
    opts = options if options is not None else CppCompileOptions()
    cxx, pybind_inc = _cxx_or_skip(opts)
    ldflags = _python_embed_link_flags() if opts.embed_python else []
    if opts.embed_python and not ldflags:
        pytest.skip("embed_python=True requires python-config --ldflags")

    with tempfile.TemporaryDirectory() as tmp:
        tdir = Path(tmp)
        src = tdir / "export_codegen_test.cpp"
        exe = tdir / "export_codegen_test_bin"
        src.write_text(cpp_source, encoding="utf-8")

        cmd = _compile_cmd(cxx, src, opts, pybind_inc, syntax_only=False)
        if opts.embed_python:
            cmd.extend(_conda_link_flags())
            cmd.extend(ldflags)
        cmd.extend(["-o", str(exe)])

        label = "embed_python" if opts.embed_python else "native"
        _run_compile(cmd, label)

        run_env = dict(os.environ)
        if opts.embed_python:
            conda_prefix = run_env.get("CONDA_PREFIX")
            if conda_prefix:
                conda_lib = str(Path(conda_prefix) / "lib")
                old_ld = run_env.get("LD_LIBRARY_PATH", "")
                run_env["LD_LIBRARY_PATH"] = f"{conda_lib}:{old_ld}" if old_ld else conda_lib

        proc = subprocess.run([str(exe)], capture_output=True, text=True, env=run_env)
        if proc.returncode != 0:
            pytest.fail(
                f"Binary exited {proc.returncode} ({label})\n"
                f"stdout={proc.stdout!r}\nstderr={proc.stderr!r}"
            )


def _compile_only(cpp_source: str, *, options: CppCompileOptions | None = None) -> None:
    """Compile *cpp_source* with ``-fsyntax-only`` (no link, no run).

    Used for the new custom-executor TU: its ``Compile()`` JIT-builds a real kernel and
    ``DeclareLaunchArgs()`` reads a launch sidecar, so the TU cannot be *run* in a unit test.
    We only assert it compiles cleanly against the fixture mocks + pybind headers.
    """
    opts = options if options is not None else CppCompileOptions()
    cxx, pybind_inc = _cxx_or_skip(opts)

    with tempfile.TemporaryDirectory() as tmp:
        tdir = Path(tmp)
        src = tdir / "export_codegen_test.cpp"
        src.write_text(cpp_source, encoding="utf-8")
        _run_compile(_compile_cmd(cxx, src, opts, pybind_inc, syntax_only=True), "-fsyntax-only")


def _full_custom_executor_test_tu(*, op_type: str, dtypes) -> str:
    """Assemble the thin custom-executor TU for a *compile-only* test (no main, no run).

    The thin executor only ``#include "pto_custom_op.h"`` (the shared base owns
    Compile/DeclareLaunchArgs); for the test that resolves to the fixture mock
    (fixtures/pto_custom_op.h) so we type-check the generated subclass's hooks (module-stem,
    kernel-.py basename, InferShape/InferDataType) without the GE/Python/json deps.
    """
    return cpp_mod._generate_custom_executor_cpp(
        op_type, **_executor_kwargs(dtypes=dtypes)
    )


def _normalize_infer_shape_outputs(out: tuple) -> list[tuple]:
    if not out:
        return []
    if isinstance(out[0], int):
        return [tuple(out)]
    return [tuple(x) for x in out]


def _build_main(
    fn: Callable[..., tuple],
    input_shapes: Sequence[Tuple[int, ...]],
) -> str:
    out = fn(*input_shapes)
    outputs = _normalize_infer_shape_outputs(out)
    lines = [
        "int main() {",
        "  pybind11::scoped_interpreter guard{};",
        "  gert::InferShapeContext ctx;",
    ]
    for i, shp in enumerate(input_shapes):
        br = ", ".join(str(int(x)) for x in shp)
        lines.append(f"  ctx.SetInput({i}, {{{br}}});")
    lines.extend(
        [
            "  ge::graphStatus st = ge::InferShapeGeImpl(&ctx);",
            "  if (st != ge::GRAPH_SUCCESS) return 1;",
        ]
    )
    ret_code = 2
    for oi, dims in enumerate(outputs):
        lines.append(f"  const auto& o{oi} = ctx.OutputShape({oi}u);")
        lines.append(f"  if (o{oi}.GetDimNum() != {len(dims)}u) return {ret_code};")
        ret_code += 1
        for j, v in enumerate(dims):
            lines.append(f"  if (o{oi}[{j}] != {int(v)}ll) return {ret_code};")
            ret_code += 1
    lines.append("  return 0;")
    lines.append("}")
    return "\n".join(lines)


def _write_infer_py(tmp_path, *funcs: Callable) -> str:
    """Write the ``.py`` the path-import wrapper loads at run time, and return its absolute path.

    Its content comes from ``codegen_test_helpers`` (``_infer_py_module_source``), so the file the generated TU
    imports always matches what the wrapper expects.
    """
    py_file = tmp_path / "infer_defs.py"
    py_file.write_text(codegen_test_helpers._infer_py_module_source(*funcs), encoding="utf-8")
    return str(py_file)


def _full_infer_shape_translation_unit(
    fn: Callable[..., tuple],
    input_shapes: Sequence[Tuple[int, ...]],
    infer_py_path: str,
) -> str:
    tu = codegen_test_helpers._generate_infer_shape_host_tu_for_test(fn, infer_py_path)
    main = _build_main(fn, input_shapes)
    return f"""#include "gert_ge_minimal.hpp"
#include <pybind11/embed.h>
{tu}

{main}
"""


def _opdef_infer_pair_for_arity(n: int) -> tuple[Callable, Callable]:
    mapping: dict[int, tuple[Callable, Callable]] = {
        1: (infer_shape_samples.infer_shape_one_2d, infer_shape_samples.infer_dtype_one),
        2: (infer_shape_samples.infer_shape_two_by_two, infer_shape_samples.infer_dtype_two),
        3: (infer_shape_samples.infer_shape_three_4d, infer_shape_samples.infer_dtype_three),
    }
    return mapping[n]


def _op_custom_def_compile_tu(*, op_type: str, dtypes: Sequence[torch.dtype]) -> str:
    """The op-def TU is now a bare ``REG_OP`` prototype (infer moved to the executor as
    ``ge::ShapeInferOp`` member methods). Nothing to *run*, just verify it compiles against the
    ``graph/operator_reg.h`` mock."""
    n = len(dtypes)
    infer_shape_fn, infer_dtype_fn = _opdef_infer_pair_for_arity(n)
    return codegen_test_helpers._generate_op_custom_def_cpp(
        infer_shape_fn, infer_dtype_fn, op_type=op_type, dtypes=list(dtypes)
    )


def _opdef_infer_run_tu(*, dtypes: Sequence[torch.dtype], infer_py_path: str) -> str:
    """Run-test harness for the relocated pybind infer wrappers (the ones the executor's
    ``InferShape``/``InferDataType`` member methods call). Emits both wrappers in their anonymous
    namespace (callable from ``main`` in the same TU) and exercises them with real Python, preserving
    the infer_shape + infer_dtype *execution* coverage that used to live in the op-def run test."""
    n = len(dtypes)
    infer_shape_fn, infer_dtype_fn = _opdef_infer_pair_for_arity(n)
    shape_meta = cpp_mod._parse_infer_shape_for_codegen(infer_shape_fn)
    dtype_meta = cpp_mod._parse_infer_dtype_for_codegen(infer_dtype_fn)
    # Both wrappers import from the SAME .py (it defines both funcs), each by path.
    shape_wrapper = cpp_mod._generate_pybind_wrapper(
        infer_shape_fn, cpp_mod._CPP_BIND__INFER_SHAPE, import_via="path", py_path=infer_py_path
    )
    dtype_wrapper = cpp_mod._generate_pybind_wrapper(
        infer_dtype_fn, cpp_mod._CPP_BIND__INFER_DTYPE, import_via="path", py_path=infer_py_path
    )
    shape_args = ", ".join(["std::vector<int64_t>{2, 3}"] * n)
    dtype_args = ", ".join(["int64_t(1)"] * n)  # 1 == DT_FLOAT16 enum value
    return f"""#include <cstdint>
#include <cstdio>
#include <vector>
#include <tuple>
#include <pybind11/pybind11.h>
#include <pybind11/eval.h>
#include <pybind11/stl.h>
#include <pybind11/embed.h>

#ifndef PTO_CUSTOM_LOGD
#define PTO_CUSTOM_LOGD(fmt, ...) ((void)0)
#endif

namespace py = pybind11;
using namespace py::literals;

{shape_wrapper}
{dtype_wrapper}

int main() {{
  pybind11::scoped_interpreter guard{{}};
  auto s = {shape_meta.cpp_bind_name}({shape_args});
  if (s.empty()) return 1;
  auto d = {dtype_meta.cpp_bind_name}({dtype_args});
  (void)d;
  return 0;
}}
"""


@pytest.mark.cpp_codegen
@pytest.mark.parametrize(
    ("fn", "inputs"),
    [
        (infer_shape_samples.infer_shape_two_by_two, ((32, 64), (10, 20))),
        (infer_shape_samples.infer_shape_4d_broadcast, ((1, 2, 3, 4), (1, 2, 3, 4))),
        (infer_shape_samples.infer_shape_sum_last, ((2, 3, 5), (2, 3, 7))),
        (infer_shape_samples.infer_shape_nd_identity, ((1, 2, 3, 4),)),
        (infer_shape_samples.infer_shape_two_outputs, ((5, 6), (7, 8))),
    ],
)
def test_compile_and_run_infer_shape_host(fn, inputs, tmp_path):
    # The infer func lives in a real .py; its path is baked into the TU and imported at run time.
    infer_py_path = _write_infer_py(tmp_path, fn)
    src = _full_infer_shape_translation_unit(fn, inputs, infer_py_path)
    _compile_and_run(src, options=CppCompileOptions(embed_python=True))


@pytest.mark.cpp_codegen
@pytest.mark.parametrize(
    "dtypes",
    [
        (torch.float16,),
        (torch.float16, torch.float16),
        (torch.float16, torch.float16, torch.float16),
    ],
)
@pytest.mark.parametrize("op_type", ["Add", "MyKernel"])
def test_compile_custom_executor_launch(op_type: str, dtypes):
    """The thin executor TU is compile-only (its base JIT-builds a real kernel + launches at run
    time). This ``-fsyntax-only`` check resolves ``pto_custom_op.h`` to the fixture mock. It now
    embeds pybind (the ``ge::ShapeInferOp`` InferShape/InferDataType member methods call the embedded
    pybind infer wrappers), so ``embed_python=True`` to supply pybind + Python headers."""
    src = _full_custom_executor_test_tu(op_type=op_type, dtypes=list(dtypes))
    _compile_only(src, options=CppCompileOptions(embed_python=True))


@pytest.mark.cpp_codegen
@pytest.mark.parametrize("op_type", ["Add", "MyKernel"])
@pytest.mark.parametrize(
    "dtypes",
    [
        (torch.float16,),
        (torch.float16, torch.float32),
        (torch.float16, torch.float32, torch.bfloat16),
    ],
)
def test_compile_op_custom_def_cpp(op_type: str, dtypes: Sequence[torch.dtype]):
    """The bare ``REG_OP`` op-def TU compiles against the ``graph/operator_reg.h`` mock (no infer,
    no pybind, inference is now on the executor)."""
    src = _op_custom_def_compile_tu(op_type=op_type, dtypes=dtypes)
    _compile_only(src, options=CppCompileOptions())


# NOTE: the merged op TU (executor + REG_OP prototype) is compile-validated as one unit by the real
# ``build_so`` demos (cmake against the actual GE headers), not here: the two halves' fixture mocks
# (``pto_custom_op.h`` + ``graph/operator_reg.h``) each independently define the ``gert`` context classes,
# so combining them in one TU trips a mock-only redefinition that real (include-guarded) GE headers don't.
# The standalone executor + op-def compile tests above cover each half against its own coherent mock;
# ``test_generate_op_tu_cpp_merges_...`` (structure) covers the merge composition.


@pytest.mark.cpp_codegen
@pytest.mark.parametrize(
    "dtypes",
    [
        (torch.float16,),
        (torch.float16, torch.float32),
        (torch.float16, torch.float32, torch.bfloat16),
    ],
)
def test_compile_and_run_opdef_infer(dtypes: Sequence[torch.dtype], tmp_path):
    """The relocated pybind infer wrappers (called by the executor's ShapeInferOp member methods)
    execute under real Python, preserves infer_shape + infer_dtype run coverage."""
    infer_shape_fn, infer_dtype_fn = _opdef_infer_pair_for_arity(len(dtypes))
    infer_py_path = _write_infer_py(tmp_path, infer_shape_fn, infer_dtype_fn)
    src = _opdef_infer_run_tu(dtypes=dtypes, infer_py_path=infer_py_path)
    _compile_and_run(src, options=CppCompileOptions(embed_python=True))


# Executors with Int/Float/String/ListInt attrs, and a shape-affecting ListInt, must type-check
# against the fixture (RuntimeAttrs GetFloat/GetStr/GetListInt + ListIntToJsonStr stubs).

@pytest.mark.cpp_codegen
@pytest.mark.parametrize(
    ("attr_type", "default", "accessor"),
    [
        ("Int", 0, "GetInt"),
        ("Float", 0.01, "GetFloat"),
        ("String", "sum", "GetStr"),
        ("ListInt", [2, 2], "GetListInt"),
    ],
)
def test_compile_custom_executor_with_typed_attr(attr_type, default, accessor):
    src = cpp_mod._generate_custom_executor_cpp(
        "TypedAttrOp",
        attr_specs=[{"name": "a", "type": attr_type, "default": default}],
        **_executor_kwargs(),
    )
    assert f"attrs->{accessor}(0)" in src
    _compile_only(src, options=CppCompileOptions(embed_python=True))


def _crop_infer_shape_c(x_shape: torch.Size, crop_size: list[int]) -> torch.Size:
    out = list(x_shape)
    out[-1] = out[-1] // crop_size[0]
    return torch.Size(out)


def _crop_infer_dtype_c(x_dtype: torch.dtype) -> torch.dtype:
    return x_dtype


@pytest.mark.cpp_codegen
def test_compile_custom_executor_shape_affecting_listint_infershape():
    """A shape-affecting ListInt attr: the executor's InferShape reads context->GetAttrs()->
    GetListInt(0) and threads it into the pybind wrapper call, must compile against the fixture."""
    src = cpp_mod._generate_custom_executor_cpp(
        "Crop",
        create_kernel_func=kernel_compile_samples.create_add_kernel,
        infer_shape_func=_crop_infer_shape_c,
        infer_dtype_func=_crop_infer_dtype_c,
        dtypes=[torch.float16],
        attr_specs=[{"name": "crop_size", "type": "ListInt", "default": [2, 2]}],
    )
    assert "context->GetAttrs()" in src
    assert "GetListInt(0)" in src
    _compile_only(src, options=CppCompileOptions(embed_python=True))


# Compile the onnx-plugin ParseParam TU per attr type (it was string-asserted only).
# The plugin ParseParam body (nlohmann/json parse + Int null-guard + Float std::stof/is_string + String
# .c_str + ListInt ``for(auto e:attr["ints"])``) was only text-checked in test_cpp_codegen_structure.
# Here we g++-compile it so a C++ regression in ``_attr_parse_param_block`` fails CI, not only on-box ATC.
# The TU ``#include``s register/register.h (fixture: ge::Operator + the domi REGISTER_CUSTOM_OP surface),
# <nlohmann/json.hpp> (resolved from the nlohmann include root via ``codegen._resolve_nlohmann_include_dir()``,
# added to -I via extra_includes), and the slog log preamble (toolchain/slog.h + base/log_types.h fixtures).
# No pybind, the plugin runs no embedded Python.

def _compile_plugin(op_type: str, *, attr_specs) -> None:
    """g++ ``-fsyntax-only`` the domi onnx-plugin TU with the nlohmann include root on the -I path."""
    src = cpp_mod._generate_op_custom_plugin_cpp(
        op_type, framework_type=_FRAMEWORK_TYPE__ONNX, attr_specs=attr_specs,
    )
    try:
        json_inc_dir = cpp_mod._resolve_nlohmann_include_dir()
    except FileNotFoundError:
        pytest.skip("nlohmann include root not available (populate third_party_path or set PYPTO_THIRD_PARTY_PATH)")
    _compile_only(src, options=CppCompileOptions(extra_includes=(str(json_inc_dir),)))


@pytest.mark.cpp_codegen
@pytest.mark.parametrize(
    ("attr_type", "default"),
    [
        ("Int", 0),
        ("Float", 0.01),
        ("String", "sum"),
        ("ListInt", [2, 2]),
    ],
)
def test_compile_onnx_plugin_parse_param_per_attr_type(attr_type, default):
    _compile_plugin("TypedAttrOp", attr_specs=[{"name": "a", "type": attr_type, "default": default}])


@pytest.mark.cpp_codegen
def test_compile_onnx_plugin_parse_param_no_attrs_stub():
    """The attr-free ParseParam is the no-op stub (no `<nlohmann/json.hpp>` include), it must still
    compile against register/register.h + the slog preamble alone."""
    _compile_plugin("PlainOp", attr_specs=[])


# A mixed Int+Float+String executor (the add_sub_scale_bias contract), compiled as one TU.
@pytest.mark.cpp_codegen
def test_compile_custom_executor_three_mixed_attrs():
    """MANY attrs on ONE op (Int + Float + String + a SECOND Float + ListInt) compiled as one TU:
    proves the concatenated per-attr ExtractAttrs read blocks compile together with correct per-block
    ``{}`` scoping, the TWO Float blocks each declare a local ``_buf`` and the ListInt block a ``_v``,
    so a scoping regression (same local redeclared in one scope) would fail to compile here. Extends the
    add_sub_scale_bias Int+Float+String contract with the repeated-Float + ListInt to exercise that."""
    src = cpp_mod._generate_custom_executor_cpp(
        "AddSubScaleBias",
        attr_specs=[
            {"name": "bias", "type": "Int", "default": 0},
            {"name": "scale", "type": "Float", "default": 1.0},
            {"name": "mode", "type": "String", "default": "sum"},
            {"name": "extra", "type": "Float", "default": 2.0},        # 2nd Float -> 2nd _buf block
            {"name": "sizes", "type": "ListInt", "default": [2, 2]},   # ListInt -> a _v block
        ],
        **_executor_kwargs(),
    )
    assert "attrs->GetInt(0)" in src        # bias  = declaration-index 0
    assert "attrs->GetFloat(1)" in src      # scale = declaration-index 1
    assert "attrs->GetStr(2)" in src        # mode  = declaration-index 2
    assert "attrs->GetFloat(3)" in src      # extra = declaration-index 3 (2nd Float)
    assert "attrs->GetListInt(4)" in src    # sizes = declaration-index 4
    _compile_only(src, options=CppCompileOptions(embed_python=True))


def test_local_framework_type_onnx_matches_the_build_module():
    """The locally held token must equal ``build``'s, the value the generated plugin registers under."""
    build = pytest.importorskip("exported_custom_op_litenpu.deploy.build")
    assert _FRAMEWORK_TYPE__ONNX == build._FRAMEWORK_TYPE__ONNX
