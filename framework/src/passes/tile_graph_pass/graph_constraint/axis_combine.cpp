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
 * \file axis_combine.cpp
 * \brief
 */

#include "axis_combine.h"
#include "passes/pass_utils/dead_operation_eliminate.h"
#include "passes/pass_log/pass_log.h"

#define MODULE_NAME "AxisCombine"

namespace npu {
namespace tile_fwk {
constexpr size_t INPUT_SIZE = 2;
const std::unordered_set<Opcode> NEED_BRC_OPS{
    Opcode::OP_ADD,
    Opcode::OP_SUB,
    Opcode::OP_MUL,
    Opcode::OP_DIV,
    Opcode::OP_MAXIMUM,
    Opcode::OP_MINIMUM,
};

bool InsertCondition(const Opcode &code) {
    if (NEED_BRC_OPS.count(code) > 0) {
        return true;
    }
    return false;
}

void AlignedIfNeed(int64_t &currentDim, int64_t &padValue) {
    if (padValue == 0) {
        APASS_LOG_ERROR_F(Elements::Config, "invalid pad base %d.", padValue);
        return;
    }
    if (currentDim % padValue != 0) {
        currentDim = (currentDim + padValue - 1) / padValue * padValue;
    }
}

Status GetPaddingValue(const LogicalTensorPtr &tensor, int64_t &padValue) {
    auto bytes = BytesOf(tensor->Datatype());
    auto paddingIter = BLOCK_PADDING_DIM.find(bytes);
    if (paddingIter == BLOCK_PADDING_DIM.end()) {
        APASS_LOG_ERROR_F(Elements::Tensor, "tensor %d's datatype is not supported.", tensor->GetMagic());
        return FAILED;
    }
    padValue = paddingIter->second;
    return SUCCESS;
}

Status AlignBroadCastOpInputs(Function &function, Operation &op) {
    auto inputTensor = op.GetIOperands();
    auto inTensor0 = inputTensor[0];
    auto inTensor1 = inputTensor[1];
    if (inTensor0->GetShape() == inTensor1->GetShape()) {
        return SUCCESS;
    }
    for (size_t idx = 0; idx < inputTensor.size(); ++idx) {
        auto srcTensor = inputTensor[idx];
        auto alignedShape = srcTensor->GetShape();
        if (alignedShape.back() == 1) {
            int64_t padValue = 0;
            if (GetPaddingValue(srcTensor, padValue) != SUCCESS) {
                return FAILED;
            }
            AlignedIfNeed(alignedShape.back(), padValue);
            auto alignedTensor = std::make_shared<LogicalTensor>(function, srcTensor->Datatype(), alignedShape, srcTensor->Format());
            alignedTensor->SetMemoryTypeBoth(MemoryType::MEM_UB, true);
            auto &brcb = function.AddRawOperation(Opcode::OP_BRCB, {srcTensor}, {alignedTensor});
            brcb.UpdateSubgraphID(op.GetSubgraphID());
            srcTensor->RemoveConsumer(op);
            op.ReplaceIOperand(idx, alignedTensor);
            op.SetAttribute(OpAttributeKey::brcbIdx, static_cast<int64_t>(idx + 1));
            inputTensor[idx] = alignedTensor;
        }
    }
    return SUCCESS;
}

Status AxisCombine::Process(Function &function) {   
    for (auto &op : function.Operations()) {
        if (InsertCondition(op.GetOpcode()) && op.GetIOperands().size() == INPUT_SIZE) {
            if (AlignBroadCastOpInputs(function, op) != SUCCESS) {
                    APASS_LOG_ERROR_F(Elements::Operation, "operation %d's aligned faild. %s", op.GetOpMagic(), op.GetOpcodeStr().c_str());
                    return FAILED;
            } 
        }
    }
    return SUCCESS;
}

Status AxisCombine::RunOnFunction(Function &function) {
    APASS_LOG_INFO_F(Elements::Function, "===> Start AxisCombine.");
    if (!ConfigManager::Instance().GetOperationConfig("COMBINE_AXIS", false)) {
        APASS_LOG_INFO_F(Elements::Operation, "AxisCombine is skipped.");
        return SUCCESS;
    }
    if (Process(function) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Function, "AxisCombine process failed.");
        return FAILED;
    }
    APASS_LOG_INFO_F(Elements::Function, "===> End AxisCombine.");
    return SUCCESS;
}
} // namespace tile_fwk
} // namespace npu