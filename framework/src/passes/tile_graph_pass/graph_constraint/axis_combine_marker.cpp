/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file axis_combine_marker.cpp
 * \brief
 */

#include "axis_combine_marker.h"
namespace npu {
namespace tile_fwk {
void AxisCombineMarker::Run(Function &function) {
    Init(function);
    ForwardVisit();
    BackwardVisit();
}

bool AxisCombineMarker::IsTensorEnableAxisCombine(LogicalTensorPtr tensor) {
    if (tensorStatus_.find(tensor) != tensorStatus_.end() && tensorStatus_[tensor] == AxisReorderStatus::ENABLE) {
        return true;
    }
    return false;
}

void AxisCombineMarker::Init(Function &function) {
    size_t i = 0U;
    std::map<int, size_t> opMagic2Idx;
    opList_ = function.Operations().DuplicatedOpList();
    for (const auto op : opList_) {
        opMagic2Idx[op->GetOpMagic()] = i;
        i++;
    }
    opInGraph_.resize(opList_.size());
    opOutGraph_.resize(opList_.size());
    for (size_t opIdx = 0; opIdx < opList_.size(); opIdx++) {
        const auto& op = opList_[opIdx];
        for (const auto producer : op->ProducerOpsOrdered()) {
            opInGraph_[opMagic2Idx[op->GetOpMagic()]].push_back(opMagic2Idx[producer->GetOpMagic()]);
        }
        for (const auto consumer : op->ConsumerOpsOrdered()) {
            opOutGraph_[opMagic2Idx[op->GetOpMagic()]].push_back(opMagic2Idx[consumer->GetOpMagic()]);
        }
    }
}

void UpdateCopyinStatus(Operation *op, std::unordered_map<LogicalTensorPtr, AxisReorderStatus> &tensorStatus) {
    auto inputTensor = op->GetIOperands()[0];
    auto outputTensor = op->GetOOperands()[0];
    if (outputTensor->GetShape().back() != 1) {
        tensorStatus[outputTensor] = AxisReorderStatus::UNKNOWN;
        return;
    }
    if (outputTensor->GetShape().back() == inputTensor->GetShape().back()) {
        tensorStatus[outputTensor] = AxisReorderStatus::ENABLE;
        return;
    }
    tensorStatus[outputTensor] = AxisReorderStatus::DISABLE;
    return;
}

void UpdateViewStatus(Operation *op, std::unordered_map<LogicalTensorPtr, AxisReorderStatus> &tensorStatus) {
    auto inputTensor = op->GetIOperands()[0];
    auto outputTensor = op->GetOOperands()[0];
    if (inputTensor->GetShape().back() != outputTensor->GetShape().back()) {
        tensorStatus[outputTensor] = AxisReorderStatus::DISABLE;
        return;
    }
    if (outputTensor->GetShape().back() != 1) {
        tensorStatus[outputTensor] = AxisReorderStatus::UNKNOWN;
        return;
    }
    if (tensorStatus.find(inputTensor) != tensorStatus.end()) {
        tensorStatus[outputTensor] = tensorStatus[inputTensor];
        return;
    }
    if (inputTensor->GetShape().back() == 1 && outputTensor->GetShape().back() == 1) {
        tensorStatus[inputTensor] = AxisReorderStatus::ENABLE;
        tensorStatus[outputTensor] = AxisReorderStatus::ENABLE;
        return;
    }
    tensorStatus[inputTensor] = AxisReorderStatus::DISABLE;
    tensorStatus[outputTensor] = AxisReorderStatus::DISABLE;  // DDR场景，不涉及。
}

void UpdateAssembleStatus(Operation *op, std::unordered_map<LogicalTensorPtr, AxisReorderStatus> &tensorStatus) {
    auto inputTensor = op->GetIOperands()[0];
    auto outputTensor = op->GetOOperands()[0];
    if (tensorStatus.find(inputTensor) != tensorStatus.end()) {
        if (tensorStatus[inputTensor] == AxisReorderStatus::ENABLE) {
            if (inputTensor->GetShape().back() != outputTensor->GetShape().back()) {
                tensorStatus[outputTensor] = AxisReorderStatus::DISABLE;
                tensorStatus[inputTensor] = AxisReorderStatus::DISABLE;  // 如果尾轴有assemble，那么不能支持合轴
            } else {
                tensorStatus[outputTensor] = AxisReorderStatus::ENABLE;
            }
            return;
        }
        tensorStatus[outputTensor] = AxisReorderStatus::DISABLE;
        return;
    }
    // 正向推导不应该存在assemble输入没被访问过的场景
}

void UpdateExpandStatus(Operation *op, std::unordered_map<LogicalTensorPtr, AxisReorderStatus> &tensorStatus) {
    auto inputTensor = op->GetIOperands()[0];
    auto outputTensor = op->GetOOperands()[0];
    if (tensorStatus[inputTensor] == AxisReorderStatus::ENABLE) {
        auto dimSize = static_cast<int>(inputTensor->GetShape().size());
        int axis = op->GetIntAttribute(OP_ATTR_PREFIX + "EXPANDDIM");
        // 在尾轴为1的条件下，要求尾轴没有发生broadcast。[n, 1, 1]->expand->[n, 8, 1]??
        if (axis < dimSize - 2) {
            tensorStatus[outputTensor] = AxisReorderStatus::ENABLE;
            return;
        } else {
            // 如果是尾轴broadcast，不支持交换轴
            tensorStatus[inputTensor] = AxisReorderStatus::DISABLE;
            tensorStatus[outputTensor] = AxisReorderStatus::UNKNOWN;
            if (outputTensor->GetShape().back() == 1) {
                tensorStatus[outputTensor] = AxisReorderStatus::DISABLE;
                return;
            }
        }
        return;
    }
    // 如果expand输出尾轴为1，并且输入就不支持合轴，那么输出也不支持合轴
    if (outputTensor->GetShape().back() == 1) {
        tensorStatus[outputTensor] = AxisReorderStatus::DISABLE;
        return;
    }
    // 如果expand的输出尾轴不为1，那么不涉及到合轴优化。
    tensorStatus[outputTensor] = AxisReorderStatus::UNKNOWN;
}

void UpdateReduceStatus(Operation *op, std::unordered_map<LogicalTensorPtr, AxisReorderStatus> &tensorStatus) {
    // 最后两根轴不发生reduce，并且尾轴为1。那么支持交换轴，如果倒数第二根轴发生reduce，不支持。尾轴reduce，需不需要交换轴要看后继节点
    auto inputTensor = op->GetIOperands()[0];
    auto outputTensor = op->GetOOperands()[0];
    auto dimSize = static_cast<int>(inputTensor->GetShape().size());
    int axis = op->GetIntAttribute(OP_ATTR_PREFIX + "AXIS");
    if (dimSize > 1 && axis < dimSize - 2) {
        tensorStatus[outputTensor] = tensorStatus[inputTensor];
        return;
    }
    if (axis == dimSize - 2) {
        // reduce倒数第二轴，当前不支持合轴优化
        tensorStatus[outputTensor] = AxisReorderStatus::DISABLE;
        return;
    }
    // Reduce尾轴，默认可以
    tensorStatus[outputTensor] = AxisReorderStatus::ENABLE;
}

void UpdateElewiseStatus(Operation *op, std::unordered_map<LogicalTensorPtr, AxisReorderStatus> &tensorStatus) {
    auto outputTensor = op->GetOOperands()[0];
    for (auto inputTensor : op->GetIOperands()) {
        if (tensorStatus[inputTensor] == AxisReorderStatus::UNKNOWN && inputTensor->GetShape().back() == 1) {
            tensorStatus[inputTensor] = AxisReorderStatus::ENABLE;
        }
    }
    if (outputTensor->GetShape().back() != 1) {
        tensorStatus[outputTensor] = AxisReorderStatus::UNKNOWN;
        return;
    }
    for (auto inputTensor : op->GetIOperands()) {
        if (tensorStatus.find(inputTensor) != tensorStatus.end() && tensorStatus[inputTensor] == AxisReorderStatus::DISABLE) {
            tensorStatus[outputTensor] = AxisReorderStatus::DISABLE;
            return;
        }
    }
    tensorStatus[outputTensor] = AxisReorderStatus::ENABLE;
}

void AxisCombineMarker::UpdateOpACEnableForward(uint16_t opIdx) {
    auto op = opList_[opIdx];
    auto outputTensor = op->GetOOperands()[0];
    if (outputTensor->GetShape().back() != outputTensor->GetRawTensor()->GetRawShape().back()) {
        tensorStatus_[outputTensor] = AxisReorderStatus::DISABLE;
        return;
    }
    if (op->GetOpcode() == Opcode::OP_COPY_IN) {
        UpdateCopyinStatus(op, tensorStatus_);
        return;
    }
    if (op->GetOpcode() == Opcode::OP_VIEW) {
        UpdateViewStatus(op, tensorStatus_);
        return;
    }
    if (op->GetOpcode() == Opcode::OP_ASSEMBLE) {
        UpdateAssembleStatus(op, tensorStatus_);
        return;
    }
    if (op->GetOpcode() == Opcode::OP_EXPAND) {
        UpdateExpandStatus(op, tensorStatus_);
        return;
    }
    if (OpcodeManager::Inst().GetOpCalcType(op->GetOpcode()) == OpCalcType::REDUCE) {
        UpdateReduceStatus(op, tensorStatus_);
        return;
    }
    if (OpcodeManager::Inst().GetOpCalcType(op->GetOpcode()) == OpCalcType::ELMWISE ||
        OpcodeManager::Inst().GetOpCalcType(op->GetOpcode()) == OpCalcType::BROADCAST) {
        UpdateElewiseStatus(op, tensorStatus_);
        return;
    }
    tensorStatus_[outputTensor] = AxisReorderStatus::UNKNOWN;
}

void AxisCombineMarker::UpdateOpACEnableBackward(uint16_t opIdx) {
    auto op = opList_[opIdx];
    auto outputTensor = op->GetOOperands()[0];
    if (OpcodeManager::Inst().GetOpCalcType(op->GetOpcode()) == OpCalcType::ELMWISE ||
        OpcodeManager::Inst().GetOpCalcType(op->GetOpcode()) == OpCalcType::BROADCAST ||
        ((op->GetOpcode() == Opcode::OP_VIEW || op->GetOpcode() == Opcode::OP_ASSEMBLE) &&
          outputTensor->GetShape().back() == op->GetIOperands()[0]->GetShape().back())) {
        if (tensorStatus_[outputTensor] == AxisReorderStatus::DISABLE) {
            for (auto inputTensor : op->GetIOperands()) {
                tensorStatus_[inputTensor] = AxisReorderStatus::DISABLE;
            }
            return;
        }
        bool disable{false};
        for (auto inputTensor : op->GetIOperands()) {
            if (tensorStatus_[inputTensor] == AxisReorderStatus::DISABLE) {
                disable = true;
            }
        }
        if (disable) {
            for (auto inputTensor : op->GetIOperands()) {
                tensorStatus_[inputTensor] = AxisReorderStatus::DISABLE;
            }
            return;
        }
        for (auto inputTensor : op->GetIOperands()) {
            if (tensorStatus_[inputTensor] == AxisReorderStatus::UNKNOWN) {
                tensorStatus_[inputTensor] = tensorStatus_[outputTensor];
            }
        }
    }
}

void AxisCombineMarker::ForwardVisit()
{
    std::queue<size_t> procOpQueue;
    std::vector<size_t> inDegree(opList_.size(), 0);
    for (size_t j = 0; j < opInGraph_.size(); ++j) {
        if (opInGraph_[j].empty()) {
            procOpQueue.push(j);
            UpdateOpACEnableForward(j);
        }
        inDegree[j] = opInGraph_[j].size();
    }
    while (!procOpQueue.empty()) {
        auto opIdx = procOpQueue.front();
        procOpQueue.pop();
        for (auto outIdx : opOutGraph_[opIdx]) {
            inDegree[outIdx]--;
            if (inDegree[outIdx] == 0) {
                procOpQueue.push(outIdx);
                UpdateOpACEnableForward(outIdx);
            }
        }
    }
}

void AxisCombineMarker::BackwardVisit()
{
    std::queue<size_t> procOpQueue;
    std::vector<size_t> outDegree(opList_.size(), 0);
    for (size_t j = 0; j < opOutGraph_.size(); ++j) {
        if (opOutGraph_[j].empty()) {
            procOpQueue.push(j);
            UpdateOpACEnableBackward(j);
        }
        outDegree[j] = opOutGraph_[j].size();
    }
    while (!procOpQueue.empty()) {
        auto opIdx = procOpQueue.front();
        procOpQueue.pop();
        for (auto outIdx : opInGraph_[opIdx]) {
            outDegree[outIdx]--;
            if (outDegree[outIdx] == 0) {
                procOpQueue.push(outIdx);
                UpdateOpACEnableBackward(outIdx);
            }
        }
    }
}
}
}