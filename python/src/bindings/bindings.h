/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file bindings.h
 * \brief
 */

#pragma once

#include <pybind11/pybind11.h>

namespace py = pybind11;
namespace pypto {
void bind_enum(py::module &m);
void BindElement(py::module &m);
void BindTensor(py::module &m);
void BindSymbolicScalar(py::module &m);
void bind_controller(py::module &m);
void bind_operation(py::module &m);
void BindRuntime(py::module &m);
void BindCostModelRuntime(py::module &m);
void bind_pass(py::module &m);
void BindFunction(py::module &m);
} // namespace pypto
