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
 * \file pybind11.cpp
 * \brief
 */

#include "pybind_common.h"
#include "bindings/bindings.h"

using namespace npu::tile_fwk;

namespace pypto {
PYBIND11_MODULE(pypto_impl, m) {
    m.doc() = "PyPTO";
    bind_enum(m);
    BindElement(m);
    BindTensor(m);
    BindSymbolicScalar(m);
    bind_controller(m);
    bind_operation(m);
    BindRuntime(m);
    BindCostModelRuntime(m);
    bind_pass(m);
    BindFunction(m);
    BindDistributed(m);
};
} // namespace pypto
