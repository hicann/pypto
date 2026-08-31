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
 * \file unary.cpp
 * \brief
 */

#include "binary.h"
#include "unary.h"
#include "tensor_transformation.h"
#include "interface/utils/operator_tracer.h"
#include "tilefwk/error_code.h"

namespace npu::tile_fwk {

void UnaryOperationOperandCheck(const std::vector<LogicalTensorPtr>& iOperand,
                                const std::vector<LogicalTensorPtr>& oOperand)
{
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, iOperand.size() == 1) << "The input operand size should be 1";
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, oOperand.size() == 1) << "The output operand size should be 1";
}

} // namespace npu::tile_fwk
