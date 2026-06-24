/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ir.h"

#include "symbolic_scalar.h"
#include "logical_tensor.h"

namespace pypto::ir {
std::string DumpScalarExpr(const ScalarExprPtr& op)
{
    auto p = std::dynamic_pointer_cast<const npu::tile_fwk::RawSymbolicExpression>(op);
    ASSERT(p) << "not a RawSymbolicExpression";
    return p->Dump();
}

std::string DumpTensorVar(const VarPtr& var)
{
    auto t = std::dynamic_pointer_cast<const npu::tile_fwk::LogicalTensor>(var);
    ASSERT(t) << "not a logical tensor";
    return var->name_ + ": " + t->DumpType();
}
} // namespace pypto::ir
