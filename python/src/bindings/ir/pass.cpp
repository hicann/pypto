/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>


#include "bindings/ir/bindings.h"
#include "ir/transforms/passes.h"

namespace pypto {
namespace ir {

void BindPasses(py::module_& m)
{
    py::class_<Pass>(m, "Pass", "Opaque pass object. Do not instantiate directly - use factory functions.")
        .def("__call__", &Pass::operator(), py::arg("program"), "Execute pass on program")
        .def_static("convert_to_ssa", &pass::ConvertToSSA, "Create an SSA conversion pass")
        .def_static("init_mem_ref", &pass::InitMemRef, "Create a memory reuse pass")
        .def_static("aggressive_dce", &pass::AggressiveDCE, "Eliminate dead code")
        .def_static("canonicalize", &pass::Canonicalize, "Canonicalize IR")
        .def_static("token_pass", &pass::TokenPass, "Add WAR/WAW token dependencies")
        .def_static("merge_stmts_into_if", &pass::MergeStmtsIntoIf, "Merge stmts into if branches")
        .def_static("create_path_funcs", &pass::CreatePathFuncs, "Create path functions from IR");

}
} // namespace ir
} // namespace pypto
