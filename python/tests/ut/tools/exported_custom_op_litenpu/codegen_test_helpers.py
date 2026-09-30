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
"""C++ translation units that only the codegen unit tests generate.

The deployed pipeline never emits these: they exist so the structure tests can assert on their text and
the compile tests can build and run them against the mock GE headers. Each one is a thin assembly of the
shipped generators in ``exported_custom_op_litenpu.deploy.codegen``, so keeping them beside their only
callers leaves that module holding shipped codegen only.
"""

from __future__ import annotations

import inspect
from typing import Any, Callable

from exported_custom_op_litenpu.deploy import codegen as cpp_mod

from pypto.extensions.torch_custom_op_litenpu.common.source_utils import _unwrap_decorated_func_source


def _infer_py_module_source(*funcs: Callable) -> str:
    """Return the ``.py`` module content an ``import_via="path"`` wrapper expects for *funcs*.

    ``import torch`` is prepended once so each function's ``torch.Size`` / ``torch.dtype`` annotations
    resolve at def-time. The torch↔GE dtype conversions are NOT included: the wrapper applies those
    itself from its own ``globals``, so the module only has to define the functions.

    Only the function ``def`` bodies are packed, so a function that reads a module-level constant from
    its defining module raises ``NameError`` when called, give such a function everything it needs
    inside its own body.
    """
    bodies = "\n\n".join(_unwrap_decorated_func_source(inspect.getsource(f)) for f in funcs)
    return f"import torch\n\n{bodies}\n"


def _generate_infer_shape_host_tu_for_test(func: Callable, infer_py_path: str) -> str:
    """Assemble infer_shape host slice for compile tests: preamble, embed pybind, InferShapeGeImpl only.

    Callers must include a mock header (e.g. ``gert_ge_minimal.hpp``) before this fragment
    so ``gert::`` / ``ge::`` symbols resolve. Omits GE register headers and OpDef.

    *infer_py_path* is the ``.py`` defining *func*, baked into the wrapper as a compile-time literal and
    imported by path at call time (``import_via="path"``, this TU is standalone, with no
    ``PtoCustomOp``). Write it with ``_infer_py_module_source`` so the module matches what the
    wrapper expects.
    """
    meta = cpp_mod._parse_infer_shape_for_codegen(func)
    pybind_block = cpp_mod._generate_pybind_wrapper(
        func, cpp_mod._CPP_BIND__INFER_SHAPE, import_via="path", py_path=infer_py_path
    ) + "\n\n"
    infer_shape_ge_body = cpp_mod._infer_shape_ge_impl_body(meta)
    tuple_inc = "#include <tuple>\n" if meta.n_outputs > 1 else ""
    return f"""// Auto-generated test TU fragment (include mock gert/ge header first)

#include <cstdint>
#include <cstdio>
{cpp_mod._LOG_PREAMBLE}{tuple_inc}#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/eval.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace py::literals;

{pybind_block}namespace ge {{

static ge::graphStatus InferShapeGeImpl(gert::InferShapeContext* context) {{
{infer_shape_ge_body}
}}

}}
"""


def _generate_op_custom_def_cpp(
    infer_shape_func: Callable,
    infer_dtype_func: Callable,
    *,
    op_type: str,
    dtypes: list[Any],
) -> str:
    """Standalone GE OpDef TU under ``op_host``: ``graph/operator_reg.h`` + the bare REG_OP prototype."""
    return (
        "// Auto-generated\n\n"
        '#include "graph/operator_reg.h"\n\n'
        + cpp_mod._reg_op_block(infer_shape_func, infer_dtype_func, op_type=op_type, dtypes=dtypes)
    )
