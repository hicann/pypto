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
 * \file split_raw.cpp
 * \brief
 */

#include "split_raw.h"
#include "passes/pass_log/pass_log.h"

#define MODULE_NAME "SplitRawTensor"

namespace npu {
namespace tile_fwk {
std::vector<int64_t> SplitRawTensor::UpdateOffset(std::vector<int64_t> &offset, const std::vector<int64_t> &diff) const {
    std::vector<int64_t> result = offset;
    for (size_t i = 0; i < offset.size(); i++) {
        if (offset[i] >= diff[i]) {
            result[i] = offset[i] - diff[i];
        }
    }
    return result;
}

std::vector<SymbolicScalar> SplitRawTensor::UpdateDynOffset(
    std::vector<SymbolicScalar> &offset, const std::vector<SymbolicScalar> &diff) const {
    std::vector<SymbolicScalar> result = offset;
    for (size_t i = 0; i < offset.size(); i++) {
        if (offset[i] >= diff[i]) {
            result[i] = offset[i] - diff[i];
        }
    }
    return result;
}

void SplitRawTensor::UpdateConsumerView(
    Function &function, const LogicalTensorPtr &logicalTensor) const {
    /* All the consumer View op's attr offset should be corret */
    /* 1. 更新View相关的属性 */
    TensorOffset tensorOffset = logicalTensor->GetTensorOffset();
    for (const auto &viewOp : logicalTensor->GetConsumers()) {
        if (viewOp->GetOpcode() != Opcode::OP_VIEW) {
            continue;
        }
        auto &output = viewOp->oOperand[0];
        if (function.IsFromOutCast(output)) {
            APASS_LOG_WARN_F(Elements::Tensor, "OP_VIEW oOperand tensor[%d] is outCast; Please check if it is an external output.", output->GetMagic());
            continue;
        }
        auto viewOpAttribute = dynamic_cast<ViewOpAttribute *>(viewOp->GetOpAttribute().get());
        if (viewOpAttribute != nullptr) {
            // VIEW操作的offset要相应被修改, 要减去被拆分LogicalTensor的offset
            auto &fromOffset = viewOpAttribute->GetFromOffset();
            fromOffset = UpdateOffset(fromOffset, tensorOffset.GetOffset());
            if (!tensorOffset.GetDynOffset().empty() && !viewOpAttribute->GetFromDynOffset().empty()) {
                auto &fromDynOffset = viewOpAttribute->GetFromDynOffset();
                fromDynOffset = UpdateDynOffset(fromDynOffset, tensorOffset.GetDynOffset());
            }
        }
        APASS_LOG_DEBUG_F(Elements::Operation, "Update View op needs fromOffset: %d.", viewOp->GetOpMagic());
    }
}

void SplitRawTensor::UpdateProducerAssemble(
    Function &function, const LogicalTensorPtr &logicalTensor) const {
    /* 1. 更新Assemble相关的属性 */
    // Assemble1 ->
    //              logicalTensor(UB) -> Reshape -> UB
    // Assemble2 ->
    TensorOffset tensorOffset = logicalTensor->GetTensorOffset();
    for (const auto &assembleOp : logicalTensor->GetProducers()) {
        if (assembleOp->GetOpcode() != Opcode::OP_ASSEMBLE) {
            continue;
        }
        auto &input = assembleOp->iOperand[0];
        if (function.IsFromInCast(input)) {
            APASS_LOG_WARN_F(Elements::Operation, "OP_ASSEMBLE iOperand tensor[%d] is inCast; Please check if it is an external input.",
                input->GetMagic());
            continue;
        }
        auto assembleOpAttribute = dynamic_cast<AssembleOpAttribute *>(assembleOp->GetOpAttribute().get());
        if (assembleOpAttribute != nullptr) {
            // Assemble操作的offset要相应被修改, 要减去被拆分LogicalTensor的offset
            auto &toOffset = assembleOpAttribute->GetToOffset();
            toOffset = UpdateOffset(toOffset, tensorOffset.GetOffset());
            if (!tensorOffset.GetDynOffset().empty() && !assembleOpAttribute->GetToDynOffset().empty()) {
                auto &toDynOffset = assembleOpAttribute->GetToDynOffset();
                toDynOffset = UpdateDynOffset(toDynOffset, tensorOffset.GetDynOffset());
            }
        }
        APASS_LOG_DEBUG_F(Elements::Operation, "Update Assemble op needs toOffset: %d.", assembleOp->GetOpMagic());
    }
}

bool SplitRawTensor::ShouldProcessTensor(Function &function, const LogicalTensorPtr &singleTensor) const {
    // 检查raw shape 和 shape 是否相等
    if ((singleTensor->GetShape() == singleTensor->tensor->GetRawShape())) {
        return false;
    }
    // 检查是否为InCast或OutCast
    if (function.IsFromOutCast(singleTensor) || function.IsFromInCast(singleTensor)) {
        APASS_LOG_WARN_F(Elements::Tensor, "Tensor[%d] is inCast or outCast; Please check if it is an external input/output", singleTensor->GetMagic());
        return false;
    }
    return true;
}

/*
遍历所有tensor，如果rawTensor的shape大于tensor的shape，
且rawTensor不是InCast和OutCast(当前Incast、OutCast的Symbol不在tensormap里)、且tensor不是ddr，
那么需要将重新创建一个shape和当前tensor相同的rawTensor
*/
void SplitRawTensor::SplitRaw(Function &function) const {
    std::vector<int> rawIdNeedDelete;
    std::vector<std::pair<int, std::set<std::shared_ptr<LogicalTensor>, TensorPtrComparator>>> newRawVec;
    // 为了按序访问tensormap, 将tensormap转化为有序map
    auto &tensorMap = function.GetTensorMap().tensorMap_;
    std::map<int, std::set<std::shared_ptr<LogicalTensor>, TensorPtrComparator>> omap;
    for (const auto &kv : tensorMap) {
        omap.insert(kv);
    }
    for (const auto &ele : omap) {
        bool needDelete = false;
        std::unordered_set<LogicalTensorPtr> relatedViewOutput;
        for (const auto &singleLogicalTensor : ele.second) {
            auto rawShape = singleLogicalTensor->tensor->GetRawShape();
            if (!ShouldProcessTensor(function, singleLogicalTensor)) {
                continue;
            }
            /* 创建新的rawTensor，并将其后接的View以及view的Consumer的rawtensor刷新为新的rawTensor */
            singleLogicalTensor->tensor = std::make_shared<RawTensor>(
                singleLogicalTensor->tensor->datatype, singleLogicalTensor->GetShape(),
                singleLogicalTensor->Format(), singleLogicalTensor->Symbol());
            APASS_LOG_DEBUG_F(Elements::Operation, "SplitRawTensor::SplitRaw: tensor[%d] updated new raw tensor[%d] with the same raw shape.",
                singleLogicalTensor->GetMagic(), singleLogicalTensor->GetRawMagic());
            if (singleLogicalTensor->tensor == nullptr) {
                continue;
            }
            needDelete = true;
            std::set<std::shared_ptr<LogicalTensor>, TensorPtrComparator> newSet;
            newSet.emplace(singleLogicalTensor);
            UpdateConsumerView(function, singleLogicalTensor);
            UpdateProducerAssemble(function, singleLogicalTensor);
            newRawVec.emplace_back(std::make_pair(singleLogicalTensor->tensor->GetRawMagic(), newSet));
            for (auto &offset : singleLogicalTensor->offset) {
                offset = 0;
            }
        }
        if (needDelete) {
            rawIdNeedDelete.emplace_back(ele.first);
        }
    }
    // 将tensormap更新回来
    tensorMap.clear();
    for (const auto &kv : omap) {
        tensorMap.insert(kv);
    }
    for (const auto &ele : newRawVec) {
        function.GetTensorMap().tensorMap_.emplace(ele);
    }
    for (const auto &id : rawIdNeedDelete) {
        function.GetTensorMap().tensorMap_.erase(id);
    }
}

Status SplitRawTensor::RunOnFunction(Function &function) {
    APASS_LOG_INFO_F(Elements::Function, "===> Start SplitRaw.");
    SplitRaw(function);
    APASS_LOG_INFO_F(Elements::Function, "===> End SplitRaw.");
    return SUCCESS;
}
} // namespace tile_fwk
} // namespace npu
