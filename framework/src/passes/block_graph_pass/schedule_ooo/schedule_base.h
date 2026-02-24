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
 * \file schedule_base.h
 * \brief
 */

#ifndef PASS_SCHEDULE_BASE_H
#define PASS_SCHEDULE_BASE_H

#include <vector>
#include <map>
#include <unordered_map>
#include <cstdint>
#include <climits>
#include "interface/operation/operation.h"
#include "interface/tensor/logical_tensor.h"
#include "interface/utils/common.h"
#include "passes/pass_log/pass_log.h"
#include "passes/pass_interface/pass.h"
#include "passes/statistics/ooo_schedule_statistic.h"
#include "passes/block_graph_pass/schedule_ooo/buffer_pool.h"
#include "passes/pass_utils/reschedule_utils.h"

#ifndef MODULE_NAME
#define MODULE_NAME "OoOScheduleBase"
#endif

namespace npu::tile_fwk {

constexpr int64_t MAX_L0A_SIZE = 64 * 1024;
constexpr int64_t MAX_L0C_SIZE = 128 * 1024;
constexpr int64_t MAX_BT_SIZE = 1 * 1024;
constexpr int64_t MAX_FIX_SIZE = 1 * 1024;
constexpr int64_t MAX_FIX_QUANT_PRE_SIZE = 1 * 2048;
constexpr int32_t DIM_FIVE = 5;
constexpr int32_t LAST_TWO_DIM = 2;
constexpr int32_t UB_BLOCK_SIZE = 32;

const std::unordered_set<Opcode> USE_LESS_OPS2 = {
    Opcode::OP_NOP,
    Opcode::OP_RESHAPE,
    Opcode::OP_SHMEM_WAIT_UNTIL,
    Opcode::OP_VIEW,
    Opcode::OP_ASSEMBLE,
    Opcode::OP_BIND_TENSOR,
    Opcode::OP_VIEW_TYPE,
    Opcode::OP_HUB
};

inline int BytesPerElement2(DataType dataType) {
    return BytesOf(dataType);
}

inline uint64_t CeilAlign2(uint64_t a, int b) {
    return ((a + b - 1) / b) * b;
}

class ScheduleBase {
public:
    ScheduleBase() {}
    ~ScheduleBase() {}

    std::unordered_map<Operation*, std::unordered_set<Operation*>> inGraph;
    std::unordered_map<Operation*, std::unordered_set<Operation*>> outGraph;
    std::unordered_map<int, int> bufRefCount_;
    std::unordered_map<MemoryType, int64_t> localMemSize; //内存剩余情况
    std::unordered_map<MemoryType, int64_t> localMemoryCurrentSize;
    std::unordered_map<int, LocalBufferPtr> localBufferMap_; //memid:local
    std::unordered_map<Operation*, std::unordered_set<Operation*>> opConsumers;
    std::unordered_map<Operation*, std::unordered_set<Operation*>> opProducers;
    std::unordered_map<Operation*, LogicalTensors> inOutOperandsCache_;

    //  初始依赖的list序列
    std::vector<Operation*> operations;

    const LogicalTensors& GetInOutOperandCached(Operation* op) {
        auto it = inOutOperandsCache_.find(op);
        if (it != inOutOperandsCache_.end()) return it->second;

        LogicalTensors inOutOperand;
        inOutOperand.reserve(op->GetOOperands().size() + op->GetIOperands().size());
        for (auto o : op->GetOOperands()) {
            if (o->GetMemoryTypeOriginal() < MemoryType::MEM_DEVICE_DDR) {
                inOutOperand.push_back(o);
            }
        }
        for (auto i : op->GetIOperands()) {
            if (i->GetMemoryTypeOriginal() < MemoryType::MEM_DEVICE_DDR) {
                inOutOperand.push_back(i);
            }
        }
        auto cacheIt = inOutOperandsCache_.emplace(op, std::move(inOutOperand)).first;
        return cacheIt->second;
    }

    void InitMemorySize() {
        localMemSize = {
            {MemoryType::MEM_L0A, MAX_L0A_SIZE}, {MemoryType::MEM_L0C, MAX_L0C_SIZE},
            {MemoryType::MEM_BT, MAX_BT_SIZE}, {MemoryType::MEM_FIX, MAX_FIX_SIZE},
            {MemoryType::MEM_FIX_QUANT_PRE, MAX_FIX_QUANT_PRE_SIZE},
        };
        localMemSize.insert({MemoryType::MEM_UB,
            Platform::Instance().GetDie().GetMemoryLimit(MemoryType::MEM_UB)});
        localMemSize.insert({MemoryType::MEM_L1,
            Platform::Instance().GetDie().GetMemoryLimit(MemoryType::MEM_L1)});
        localMemSize.insert({MemoryType::MEM_L0B,
            Platform::Instance().GetDie().GetMemoryLimit(MemoryType::MEM_L0B)});
        localMemSize.insert({MemoryType::MEM_FIX_QUANT_PRE,
            Platform::Instance().GetDie().GetMemoryLimit(MemoryType::MEM_FIX_QUANT_PRE)});
        localMemoryCurrentSize = localMemSize;
    }

    void InitOpConsumerAndProducer() {
        std::unordered_set<Operation*> operationList;
        for (auto op : operations) {
            operationList.insert(op);
        }
        for (auto op : operations) {
            for (auto consumer : op->ConsumerOps()) {
                if (operationList.find(consumer) != operationList.end()) {
                    opConsumers[op].insert(consumer);
                }
            }
            for (auto producer : op->ProducerOps()) {
                if (operationList.find(producer) != operationList.end()) {
                    opProducers[op].insert(producer);
                }
            }
        }
    }

    Status InitLocalBuffer(LogicalTensorPtr oOperand, int memId) {
        if (oOperand->GetMemoryTypeOriginal() >= MemoryType::MEM_DEVICE_DDR) {
            return SUCCESS;
        }
        if (static_cast<uint64_t>(oOperand->tensor->GetRawDataSize()) != ShapeCeilAlign(oOperand->tensor->rawshape, oOperand->tensor->datatype)) {
            APASS_LOG_WARN_F(Elements::Tensor, "InitLocalBuffer Failed at ShapeCeilAlign! "
                "Please ensure that the rawTensor[%d] shapes are aligned.", oOperand->GetRawMagic());
        }
        if (localBufferMap_.find(memId) == localBufferMap_.end()) {
            localBufferMap_[memId] = std::make_shared<LocalBuffer>(
                memId, oOperand->tensor->GetRawDataSize(), oOperand->GetMemoryTypeOriginal());
        } else {
            localBufferMap_[memId]->size = 
                std::max(localBufferMap_[memId]->size, static_cast<uint64_t>(oOperand->tensor->GetRawDataSize()));
        }
        return SUCCESS;
    }

    std::string GetOpInfo(Operation* op) {
        return op->GetOpcodeStr() + "[" + std::to_string(op->GetOpMagic()) + "]";
    }

    Status DelBufRefCount(const int memId) {
        if (bufRefCount_.find(memId) == bufRefCount_.end()) {
            APASS_LOG_ERROR_F(Elements::Tensor, "bufRefCount cannot find Tensor[%d].", memId);
            return FAILED;
        }
        bufRefCount_[memId]--;
        APASS_LOG_DEBUG_F(Elements::Tensor, "DelBufRefCount: memId [%d], refcount [%d].", memId, bufRefCount_[memId]);
        if (bufRefCount_[memId] < 0) {
            APASS_LOG_ERROR_F(Elements::Tensor, "Tensor[%d] bufRefCount cannot less than 0.", memId);
            return FAILED;
        }
        return SUCCESS;
    }

    uint64_t ShapeCeilAlign(std::vector<int64_t> shape, DataType dtype) {
        uint64_t bytes = 0;
        if (shape.size() == DIM_FIVE) {
            bytes = BytesPerElement2(dtype) * std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
            bytes = CeilAlign2(bytes, UB_BLOCK_SIZE);
        } else {
            uint64_t preDimSize = 1;
            uint64_t lastTwoDimSize = 1;
            for (size_t i = 0; i < shape.size(); i++) {
                if ((shape.size() != 1) && (i < (shape.size() - LAST_TWO_DIM))) {
                    preDimSize *= shape[i];
                } else {
                    lastTwoDimSize *= shape[i];
                }
            }
            bytes = preDimSize * CeilAlign2(lastTwoDimSize * BytesPerElement2(dtype), UB_BLOCK_SIZE);
        }
        return bytes;
    }

    void UpdateBufRefCount(LogicalTensorPtr tensor) {
        int memId = tensor->memoryrange.memId;
        if (tensor->GetMemoryTypeOriginal() < MemoryType::MEM_DEVICE_DDR) {
            bufRefCount_[memId]++;
        }
    }

    Status InitBufRefCount() {
        bufRefCount_.clear();
        for (const auto &op : operations) {
            inGraph[op].clear();
            outGraph[op].clear();
            for (auto &tensor : op->GetIOperands()) {
                UpdateBufRefCount(tensor);
                int memId = tensor->memoryrange.memId;
                if (InitLocalBuffer(tensor, memId) == FAILED) {
                    APASS_LOG_ERROR_F(Elements::Operation, "InitLocalBuffer failed at InitBufRefCount!");
                    return FAILED;
                }
            }
            for (auto &tensor : op->GetOOperands()) {
                UpdateBufRefCount(tensor);
                int memId = tensor->memoryrange.memId;
                if (InitLocalBuffer(tensor, memId) == FAILED) {
                    APASS_LOG_ERROR_F(Elements::Operation, "InitLocalBuffer failed at InitBufRefCount!");
                    return FAILED;
                }
            }
        }
        return SUCCESS;
    }

    void PrintDependencies() {
        for (const auto &op : operations) {
            APASS_LOG_DEBUG_F(Elements::Operation, "op: %s", GetOpInfo(op).c_str());
            for (const auto &preOp : inGraph[op]) {
                APASS_LOG_DEBUG_F(Elements::Operation, "    |--- Predecessors:");
                APASS_LOG_DEBUG_F(Elements::Operation, "        |--- %s", GetOpInfo(preOp).c_str());
            }
            for (const auto &succOp : outGraph[op]) {
                APASS_LOG_DEBUG_F(Elements::Operation, "    |--- Successors:");
                APASS_LOG_DEBUG_F(Elements::Operation, "        |--- %s", GetOpInfo(succOp).c_str());
            }
            APASS_LOG_DEBUG_F(Elements::Operation, "\n");
        }
    }

    Status InitAllocDependencies(Operation* op, std::unordered_map<int, Operation*> &tensor2AllocMap) {
        for (auto &tensor : op->GetOOperands()) {
            int memId = tensor->memoryrange.memId;
            if (tensor->GetMemoryTypeOriginal() < MemoryType::MEM_DEVICE_DDR) {
                if (tensor2AllocMap.find(memId) == tensor2AllocMap.end()) {
                    APASS_LOG_ERROR_F(Elements::Operation, "Tensor[%d] must have alloc. magic: %d, op: %s", memId, tensor->GetMagic(), GetOpInfo(op).c_str());
                    return FAILED;
                }
                AddDependency(tensor2AllocMap[memId], op, true);
            }
        }
        return SUCCESS;
    }

    bool IsOpAlloc(Operation* op) {
        if (op->GetOpcodeStr().find("ALLOC") != std::string::npos) {
            return true;
        }
        return false;
    }

    void AddDependency(Operation* preOp, Operation* postOp, bool isAlloc) {
        if (isAlloc || (!IsOpAlloc(preOp) && !IsOpAlloc(postOp))) {
            outGraph[preOp].insert(postOp);
            inGraph[postOp].insert(preOp);
        }
    }

    void FindDependencies(Operation* op) {
        for (auto &producer : opProducers[op]) {
            AddDependency(producer, op, false);
        }
        for (auto &consumer : opConsumers[op]) {
            AddDependency(op, consumer, false);
        }
    }

    Status InitDependencies() {
        std::unordered_map<int, Operation*> tensor2AllocMap;
        for (const auto &op : operations) {
            inGraph[op].clear();
            outGraph[op].clear();
            if (IsOpAlloc(op)) {
                if (op->GetOOperands().size() != 1) {
                    APASS_LOG_ERROR_F(Elements::Operation, "Alloc[%d] oOperand must be 1.", op->GetOpMagic());
                    return FAILED;
                }
                int memId = op->GetOutputOperand(0)->memoryrange.memId;
                tensor2AllocMap[memId] = op;
                continue;
            }
        }
        for (const auto &op : operations) {
            if (!IsOpAlloc(op)) {
                FindDependencies(op);
                if (InitAllocDependencies(op, tensor2AllocMap) != SUCCESS) {
                    APASS_LOG_ERROR_F(Elements::Operation, "InitAllocDependencies failed.");
                    return FAILED;
                }
            }
        }
        PrintDependencies();
        return SUCCESS;
    }

    Status CalcBufferSize(LogicalTensors tensors, std::map<MemoryType, int64_t> &bufferSize, std::set<int> &memIdMap) {
        for (auto tensor : tensors) {
            if (tensor->GetMemoryTypeOriginal() == MemoryType::MEM_DEVICE_DDR) {
                continue;
            }
            const auto &shape = tensor->tensor->GetRawShape();
            if (std::any_of(shape.begin(), shape.end(), [](int64_t d) {return d <= 0;})) {
                APASS_LOG_ERROR_F(Elements::Tensor, "Dynamic axis detected in %s, "
                    "OoOSchedule requires static rawShape!", tensor->Dump().c_str());
                return FAILED;
            }
            if (memIdMap.find(tensor->memoryrange.memId) == memIdMap.end()) {
                bufferSize[tensor->GetMemoryTypeOriginal()] += tensor->tensor->GetRawDataSize();
                memIdMap.insert(tensor->memoryrange.memId);
            }
        }
        return SUCCESS;
    }

    std::string DumpOpInfo(Operation &op) {
        std::ostringstream os;
        os << "op: " << op.GetOpcodeStr().c_str() << "[" << op.GetOpMagic() << "] | ";
        os << "inputs: { ";
        for (size_t i = 0; i < op.iOperand.size(); i++) {
            os << "RawTensor [" << op.GetInputOperand(i)->tensor->GetRawMagic() << "] ";
            os << op.iOperand[i]->tensor->DumpSSA(true, true);
            if (i != op.iOperand.size() - 1) {
                os << ", ";
            }
        }
        os << " }" << " | ";
        os << "outputs: { ";
        for (size_t i = 0; i < op.oOperand.size(); i++) {
            os << "RawTensor [" << op.GetOutputOperand(i)->tensor->GetRawMagic() << "] ";
            os << op.oOperand[i]->tensor->DumpSSA(true, true);
            if (i != op.oOperand.size() - 1) {
                os << ", ";
            }
        }
        os << "} ";
        return os.str();
    }

    Status CheckOpBufferSize(Operation *op) {
        std::map<MemoryType, int64_t> bufferSizeMap;
        std::set<int> memIdMap;
        if (CalcBufferSize(op->GetIOperands(), bufferSizeMap, memIdMap) != SUCCESS || CalcBufferSize(op->GetOOperands(), bufferSizeMap, memIdMap) != SUCCESS) {
            return FAILED;
        }
        for (auto &bufferPair : bufferSizeMap) {
            if (localMemSize.find(bufferPair.first) == localMemSize.end()) {
                continue;
            }
            if (bufferPair.second <= localMemSize[bufferPair.first]) {
                continue;
            }
            if (op->GetOpcodeStr().find("ALLOC") != std::string::npos) {
                APASS_LOG_ERROR_F(Elements::Operation, "Alloc tensor [%d] size [%d] exceeds %s size [%d]! %s",
                    op->GetOutputOperand(0)->GetMagic(), bufferPair.second, MemoryTypeToString(bufferPair.first).c_str(),
                    localMemSize[bufferPair.first], GetFormatBacktrace(*op).c_str());
                APASS_LOG_ERROR_F(Elements::Operation, "Tensor [%d] producer info:", op->GetOutputOperand(0)->GetMagic());
                for (auto producer : op->GetOutputOperand(0)->GetProducers()) {
                    if (producer == op) {
                        continue;
                    }
                    APASS_LOG_ERROR_F(Elements::Operation, "      %s.", DumpOpInfo(*producer).c_str());
                }
            } else {
                APASS_LOG_ERROR_F(Elements::Operation, "OP %s[%d] in/output total size [%ld] exceeds %s size [%d]!",
                    op->GetOpcodeStr().c_str(), op->GetOpMagic(), bufferPair.second, MemoryTypeToString(bufferPair.first).c_str(),
                    localMemSize[bufferPair.first]);
                APASS_LOG_ERROR_F(Elements::Operation, " %s.", DumpOpInfo(*op).c_str());
            }
            return FAILED;
        }
        return SUCCESS;
    }

    void UpdateAllocMap(Operation* op, std::map<int, Operation*> &tensorAllocMap) {
        for (auto outTensor : op->GetOOperands()) {
            if (outTensor->GetMemoryTypeOriginal() >= MemoryType::MEM_DEVICE_DDR) {
                continue;
            }
            int memId = outTensor->memoryrange.memId;
            if (tensorAllocMap.find(memId) == tensorAllocMap.end()) {
                tensorAllocMap[memId] = op;
            }
        }
        for (auto inTensor : op->GetIOperands()) {
            if (inTensor->GetMemoryTypeOriginal() >= MemoryType::MEM_DEVICE_DDR) {
                continue;
            }
            int memId = inTensor->memoryrange.memId;
            if (tensorAllocMap.find(memId) == tensorAllocMap.end()) {
                tensorAllocMap[memId] = op;
            }
        }
    }

    Status CheckAllocOp(std::vector<Operation*> list) {
        std::map<int, Operation*> tensorAllocMap;
        for (const auto &op : list) {
            if (IsOpAlloc(op)) {
                if (GetInOutOperandCached(op).size() != 1) {
                    APASS_LOG_ERROR_F(Elements::Operation, "%s InOutOperand size not equal to 1.",
                        GetOpInfo(op).c_str());
                    return FAILED;
                }
                UpdateAllocMap(op, tensorAllocMap);
            }
        }
        for (const auto &op : list) {
            if (!IsOpAlloc(op)) {
                UpdateAllocMap(op, tensorAllocMap);
            }
        }
        for (auto tensorAlloc : tensorAllocMap) {
            if (!IsOpAlloc(tensorAlloc.second)) {
                APASS_LOG_ERROR_F(Elements::Tensor, "%s Tensor[%d] is missing Alloc.",
                    GetOpInfo(tensorAlloc.second).c_str(), tensorAlloc.first);
                return FAILED;
            }
        }
        return SUCCESS;
    }

    Status Init(std::vector<Operation*> &opList) {
        // 初始化芯片各buffer大小
        InitMemorySize();
        operations = opList;
        InitOpConsumerAndProducer();
        for (auto& op : operations) {
            if (CheckOpBufferSize(op) != SUCCESS) {
                APASS_LOG_ERROR_F(Elements::Operation, "%s[%d] checkOpBufferSize failed! %s",
                    op->GetOpcodeStr().c_str(), op->GetOpMagic(), GetFormatBacktrace(*op).c_str());
                return FAILED;
            }
        }
        InitBufRefCount();
        // 构建依赖关系
        if (InitDependencies() != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "InitDependencies failed!");
            return FAILED;
        }

        opList = operations;
        return SUCCESS;
    }

};

struct OpQueue {
    bool busy{false};
    Operation* curOp = nullptr;
    int curOpRetireCycle{-1};
    std::deque<Operation*> queue;

    OpQueue() {}
    ~OpQueue() {}

    void Insert(Operation* op) {
        queue.push_back(op);
    }

    bool Empty() {
        return queue.empty();
    }

    Operation* Front() {
        return queue[0];
    }

    Operation* PopFront() {
        Operation* op = queue.front();
        queue.pop_front();
        return op;
    }
};
} // namespace npu::tile_fwk
#endif // PASS_SCHEDULE_BASE_H