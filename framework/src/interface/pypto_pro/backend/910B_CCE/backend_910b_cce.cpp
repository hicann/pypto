/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "backend/910B_CCE/backend_910b_cce.h"

#include "backend/common/backend.h"
#include "backend/common/soc.h"

namespace pypto {
namespace backend {

Backend910B_CCE::Backend910B_CCE() : Backend(Create910BSoC())
{
    // Operators are registered via REGISTER_BACKEND_OP macro
    // in backend_910b_cce_ops.cpp during static initialization
}

Backend910B_CCE& Backend910B_CCE::Instance()
{
    static Backend910B_CCE instance;
    return instance;
}

} // namespace backend
} // namespace pypto
