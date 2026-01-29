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
 * \file unary.h
 * \brief
 */

#pragma once
#include <string>
#include "interface/utils/common.h"
#include "interface/operation/opcode.h"
#include "interface/operation/operation_common.h"
#include "interface/function/function.h"
#include "interface/program/program.h"

namespace npu::tile_fwk {

enum class UnaryOpType {
    EXP,
    RSQRT,
    SQRT,
    CEIL,
    FLOOR,
    TRUNC,
    RECIPROCAL,
    DUPLICATE,
    ABS,
    LN,
    HUB,
};

template <UnaryOpType T>
std::string GetUnaryOpName() {
    switch (T) {
        case UnaryOpType::EXP: return "EXP";
        case UnaryOpType::RSQRT: return "RSQRT";
        case UnaryOpType::SQRT: return "SQRT";
        case UnaryOpType::CEIL: return "CEIL";
        case UnaryOpType::FLOOR: return "FLOOR";
        case UnaryOpType::TRUNC: return "TRUNC";
        case UnaryOpType::RECIPROCAL: return "RECIPROCAL";
        case UnaryOpType::DUPLICATE: return "DUPLICATE";
        case UnaryOpType::ABS: return "ABS";
        case UnaryOpType::LN: return "LN";
        case UnaryOpType::HUB: return "HUB";
        default: ASSERT(false && "unknown unary op type"); return "";
    }
}

template <UnaryOpType T>
Opcode GetUnaryOpNameCode() {
#define CASE(X) \
    case UnaryOpType::X: return Opcode::OP_##X
    switch (T) {
        CASE(EXP);
        CASE(RSQRT);
        CASE(SQRT);
        CASE(CEIL);
        CASE(FLOOR);
        CASE(TRUNC);
        CASE(RECIPROCAL);
        CASE(DUPLICATE);
        CASE(ABS);
        CASE(LN);
        CASE(HUB);
        default: ASSERT(false && "unknown unary op type");
    }
#undef CASE
}

void UnaryOperationOperandCheck(
    const std::vector<LogicalTensorPtr> &iOperand, const std::vector<LogicalTensorPtr> &oOperand);

template <UnaryOpType T>
LogicalTensorPtr TensorUnaryOperation(Function &function, LogicalTensorPtr operand) {
    auto opName = GetUnaryOpName<T>();
    CheckTensorShape(operand, opName);
    auto result = std::make_shared<LogicalTensor>(
        function, operand->tensor->datatype, operand->shape, operand->GetDynValidShape(), operand->Format());
    function.AddOperation(GetUnaryOpNameCode<T>(), {operand}, {result});
    return result;
}

} // namespace npu::tile_fwk
