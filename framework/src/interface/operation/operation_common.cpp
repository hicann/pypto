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
 * \file operation_common.cpp
 * \brief
 */

#include "operation_common.h"
#include "interface/utils/vector_error.h"

namespace npu::tile_fwk {

inline const std::vector<size_t>& GetShapeLenLimit(const std::string& op)
{
    // if the limit of op is not [1, 4], should add here
    static std::unordered_map<std::string, const std::vector<size_t>> op_shape_len_limit = {
        {"ADD", {1, 4}}, {"CAST", {1, 4}}, {"ISFINITE", {1, 5}}, {"REMR", {1, 5}}, {"DEFAULT", {1, 4}}};
    if (op_shape_len_limit.find(op) == op_shape_len_limit.end()) {
        return op_shape_len_limit.at("DEFAULT");
    }
    return op_shape_len_limit.at(op);
}

void CheckTensorShape(const LogicalTensorPtr& tensor, const std::string& op)
{
    auto shape = tensor->shape;
    // valid input dims must in [1, 4]
    auto shape_len_limit = GetShapeLenLimit(op);
    ASSERT(VectorErrorCode::ERR_PARAM_SHAPE_DIM_UNSUPPORTED,
           shape.size() >= shape_len_limit[0] && shape.size() <= shape_len_limit[1])
        << "The dims of tensor out of range.";
    int64_t shapeSize = 1;
    for (const auto& value : shape) {
        ASSERT(VectorErrorCode::ERR_PARAM_INVALID, value <= INT32_MAX)
            << "The dim value of tensor must less than or equal to INT32_MAX(2,147,483,647)";
        shapeSize *= value;
        ASSERT(VectorErrorCode::ERR_PARAM_INVALID, shapeSize <= INT32_MAX)
            << "The shape size of tensor must less than or equal to INT32_MAX(2,147,483,647)";
    }
}

void CheckTensorDynamicShape(const LogicalTensors iOperands, const Opcode opCode)
{
    const std::string opName = OpcodeManager::Inst().GetOpcodeStr(opCode);
    const auto& inputMemType = OpcodeManager::Inst().GetInputsMemType(opCode);
    if (inputMemType.size() != iOperands.size()) {
        return;
    }
    for (size_t i = 0; i < iOperands.size(); i++) {
        if (inputMemType[i] == MemoryType::MEM_DEVICE_DDR) {
            continue;
        }
        for (size_t dimIdx = 0; dimIdx < iOperands[i]->shape.size(); ++dimIdx) {
            ASSERT(VectorErrorCode::ERR_PARAM_INVALID, iOperands[i]->shape[dimIdx] > 0)
                << (!opName.empty() ? "Operation: " + opName : "")
                << " Input operand (name: " << iOperands[i]->tensor->GetSymbol() << ") "
                << " at dimension[" << dimIdx << "] has invalid shape value: " << iOperands[i]->shape[dimIdx];
        }
    }
}

std::vector<int> GetBroadCastShape(LogicalTensorPtr& operand1, LogicalTensorPtr& operand2)
{
    std::vector<int64_t> opShape1(operand1->shape);
    std::vector<int64_t> opShape2(operand2->shape);
    auto maxShapeSize = std::max(opShape1.size(), opShape2.size());
    if (opShape1.size() != maxShapeSize) {
        opShape1.insert(opShape1.begin(), maxShapeSize - opShape1.size(), 1);
    }
    if (opShape2.size() != maxShapeSize) {
        opShape2.insert(opShape2.begin(), maxShapeSize - opShape2.size(), 1);
    }
    std::vector<int> broadCastShape(maxShapeSize, 0);
    for (size_t i = 0; i < maxShapeSize; i++) {
        broadCastShape[i] = std::max(opShape1[i], opShape2[i]);
    }
    return broadCastShape;
}

std::vector<int> GetBroadcastAxes(const Shape& shape1, const Shape& shape2)
{
    Shape shape1_(shape1), shape2_(shape2);
    std::vector<int> result = {};
    auto maxShapeSize = std::max(shape1_.size(), shape2_.size());
    if (shape1_.size() != maxShapeSize) {
        shape1_.insert(shape1_.begin(), maxShapeSize - shape1_.size(), 1);
    }
    if (shape2_.size() != maxShapeSize) {
        shape2_.insert(shape2_.begin(), maxShapeSize - shape2_.size(), 1);
    }
    for (size_t i = 0; i < shape1_.size(); i++) {
        if (shape1_[i] != shape2_[i] && (shape1_[i] == 1 || shape2_[i] == 1)) {
            result.push_back(i);
        }
    }
    return result;
}

void CheckAxisRange(const Tensor& tensor, int& axis)
{
    int shapeSize = tensor.GetShape().size();
    if (axis < 0) {
        axis += shapeSize;
    }
    ASSERT(VectorErrorCode::ERR_PARAM_INVALID, axis >= 0 && axis < shapeSize)
        << "Axis is not in the reasonable range!";
}
} // namespace npu::tile_fwk
