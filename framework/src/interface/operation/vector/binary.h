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

enum class BinaryOpType {
    ADD,
    SUB,
    MUL,
    DIV,
    MAX,
    MIN,
    POW,
    ADD_BRC,
    SUB_BRC,
    MUL_BRC,
    DIV_BRC,
    MAX_BRC,
    MIN_BRC,
    S_ADD,
    S_SUB,
    S_MUL,
    S_DIV,
    S_MAX,
    S_MIN,
    MAXIMUM,
    MINIMUM,
    CMP,
};

template <BinaryOpType T>
std::string GetBinaryOpName() {
    switch (T) {
        case BinaryOpType::ADD: return "ADD";
        case BinaryOpType::SUB: return "SUB";
        case BinaryOpType::MUL: return "MUL";
        case BinaryOpType::DIV: return "DIV";
        case BinaryOpType::MAX: return "MAX";
        case BinaryOpType::MIN: return "MIN";
        case BinaryOpType::MAXIMUM: return "MAXIMUM";
        case BinaryOpType::MINIMUM: return "MINIMUM";
        case BinaryOpType::POW: return "POW";
        default: ASSERT(false && "unknown binary op type"); return "";
    }
}

template <BinaryOpType T, bool WithElement = false, bool WithBrc = false>
Opcode GetBinaryOpNameCode() {
    if constexpr (WithElement) {
#define CASE(X) \
    case BinaryOpType::X: return Opcode::OP_##X##S
        switch (T) {
            CASE(ADD);
            CASE(SUB);
            CASE(MUL);
            CASE(DIV);
            CASE(MAX);
            CASE(MIN);
            CASE(S_ADD);
            CASE(S_SUB);
            CASE(S_MUL);
            CASE(S_DIV);
            CASE(S_MAX);
            CASE(S_MIN);
            default: ASSERT(false && "unknown binary op type");
        }
#undef CASE
    }

    if constexpr (WithBrc) {
#define CASE(X) \
    case BinaryOpType::X: return Opcode::OP_##X##_BRC
        switch (T) {
            CASE(ADD);
            CASE(SUB);
            CASE(MUL);
            CASE(DIV);
            CASE(MAX);
            CASE(MIN);
            default: ASSERT(false && "unknown binary op type");
        }
#undef CASE
    }

#define CASE(X) \
    case BinaryOpType::X: return Opcode::OP_##X
    switch (T) {
        CASE(ADD);
        CASE(SUB);
        CASE(MUL);
        CASE(DIV);
        CASE(S_ADD);
        CASE(S_SUB);
        CASE(S_MUL);
        CASE(S_DIV);
        CASE(S_MAX);
        CASE(S_MIN);
        CASE(MAXIMUM);
        CASE(MINIMUM);
        CASE(POW);
        default: ASSERT(false && "unknown binary op type");
    }
#undef CASE
}

std::vector<int64_t> BinaryOperationResultShape(LogicalTensorPtr operand1, LogicalTensorPtr operand2);
LogicalTensorPtr BinaryOperationBroadCast(const LogicalTensorPtr &operand, const std::vector<int> &broadCastShape);
void CheckBinOpOperandsValid(const LogicalTensorPtr &operand1, const LogicalTensorPtr &operand2);
void BinaryOperationOperandCheck(
    const std::vector<LogicalTensorPtr> &iOperand, const std::vector<LogicalTensorPtr> &oOperand);
void CheckBinaryInputTensors(const LogicalTensorPtr &tensor1, const LogicalTensorPtr &tensor2, std::string &op);

// OP_ADD OP_SUB OP_MUL OP_DIV OP_MAX
template <BinaryOpType T>
LogicalTensorPtr TensorBinaryOperation(Function &function, const Tensor &operand1, const Tensor &operand2) {
    auto oprandT1 = operand1.GetStorage();
    auto oprandT2 = operand2.GetStorage();
    if (oprandT1->shape.size() != oprandT2->shape.size()) {
        std::vector<int> broadCastShape = GetBroadCastShape(oprandT1, oprandT2);
        oprandT1 = BinaryOperationBroadCast(oprandT1, broadCastShape);
        oprandT2 = BinaryOperationBroadCast(oprandT2, broadCastShape);
    }
    auto opName = GetBinaryOpName<T>();
    CheckBinaryInputTensors(oprandT1, oprandT2, opName);

    std::vector<SymbolicScalar> resultValidShape;
    std::vector<int64_t> resultShape = BinaryOperationResultShape(oprandT1, oprandT2);
    if ((!oprandT1->GetDynValidShape().empty()) && (!oprandT2->GetDynValidShape().empty())) {
        for (size_t i = 0; i < resultShape.size(); ++i) {
            if (resultShape[i] == oprandT1->shape[i]) {
                resultValidShape.push_back(operand1.GetStorage()->GetDynValidShape()[i]);
            } else {
                resultValidShape.push_back(operand2.GetStorage()->GetDynValidShape()[i]);
            }
        }
    }
    auto result = std::make_shared<LogicalTensor>(
        function, oprandT1->Datatype(), resultShape, resultValidShape, oprandT1->Format());
    function.AddOperation(GetBinaryOpNameCode<T>(), {oprandT1, oprandT2}, {result});
    return result;
}

// OP_ADDS OP_SUBS OP_MULS OP_DIVS OP_MAXS OP_MINS
template <BinaryOpType T>
LogicalTensorPtr TensorBinaryOperationScalar(Function &function, LogicalTensorPtr operand1, const Element &value) {
    auto opName = GetBinaryOpName<T>();
    CheckTensorShape(operand1, opName);
    auto result =
        std::make_shared<LogicalTensor>(function, operand1->Datatype(), operand1->shape, operand1->GetDynValidShape());
    auto &op = function.AddOperation(GetBinaryOpNameCode<T, true>(), {operand1}, {result});
    op.SetAttribute(OpAttributeKey::scalar, value);
    return result;
}

// OP_S_ADDS OP_S_SUBS OP_S_MULS OP_S_DIVS OP_S_MAXS
template <BinaryOpType T>
LogicalTensorPtr TensorBinaryOperationAllScalar(
    Function &function, const Tensor &operand1, const Element &value, bool reverseOperand) {
    auto result = std::make_shared<LogicalTensor>(
        function, operand1.GetStorage()->Datatype(), operand1.GetShape(), operand1.GetStorage()->GetDynValidShape());
    auto &op = function.AddOperation(GetBinaryOpNameCode<T, true>(), {operand1.GetStorage()}, {result});
    op.SetAttribute(OpAttributeKey::scalar, value);
    op.SetAttribute(OP_ATTR_PREFIX + "reverseOperand", reverseOperand);
    return result;
}

// OP_S_ADD OP_S_SUB OP_S_MUL OP_S_DIV OP_S_MAX
template <BinaryOpType T>
LogicalTensorPtr TensorBinaryOperationAllScalar(Function &function, const Tensor &operand1, const Tensor &operand2) {
    auto opName = GetBinaryOpName<T>();
    CheckBinaryInputTensors(operand1.GetStorage(), operand2.GetStorage(), opName);
    auto result = std::make_shared<LogicalTensor>(
        function, operand1.GetStorage()->Datatype(), operand1.GetShape(), operand1.GetStorage()->GetDynValidShape());
    function.AddOperation(GetBinaryOpNameCode<T, false>(), {operand1.GetStorage(), operand2.GetStorage()}, {result});
    return result;
}

} // namespace npu::tile_fwk
