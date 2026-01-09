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

    std::map<Operation*, std::set<Operation*>> inGraph;
    std::map<Operation*, std::set<Operation*>> outGraph;
    std::unordered_map<int, int> bufRefCount;
    std::unordered_map<MemoryType, int64_t> localMemSize; //内存剩余情况
    std::unordered_map<MemoryType, int64_t> localMemoryCurrentSize;
    std::unordered_map<int, LocalBufferPtr> localBufferMap; //memid:local

    //  初始依赖的list序列
    std::vector<Operation*> operations;

    LogicalTensors GetInOutOperand(Operation* op) {
        LogicalTensors inOutOperand;
        for (auto o : op->GetOOperands()) {
            if (o->GetMemoryTypeOriginal() != MemoryType::MEM_DEVICE_DDR) {
                inOutOperand.push_back(o);
            }
        }
        for (auto i : op->GetIOperands()) {
            if (i->GetMemoryTypeOriginal() != MemoryType::MEM_DEVICE_DDR) {
                inOutOperand.push_back(i);
            }
        }
        return inOutOperand;
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

    void InitLocalBuffer(LogicalTensorPtr operand, int memId) {
        if (operand->GetMemoryTypeOriginal() >= MemoryType::MEM_DEVICE_DDR) {
            return;
        }
        if (localBufferMap.find(memId) == localBufferMap.end()) {
            localBufferMap[memId] = std::make_shared<LocalBuffer>(
                memId, ShapeCeilAlign(operand->GetShape(), operand->Datatype()), operand->GetMemoryTypeOriginal());
        } else {
            localBufferMap[memId]->size =
                std::max(localBufferMap[memId]->size, ShapeCeilAlign(operand->GetShape(), operand->Datatype()));
        }
    }

    std::string GetOpInfo(Operation* op) {
        return op->GetOpcodeStr() + "[" + std::to_string(op->GetOpMagic()) + "]";
    }

    Status DelBufRefCount(const int memId) {
        if (bufRefCount.find(memId) == bufRefCount.end()) {
            APASS_LOG_ERROR_F(Elements::Tensor, "bufRefCount cannot find Tensor[%d].", memId);
            return FAILED;
        }
        bufRefCount[memId]--;
        APASS_LOG_DEBUG_F(Elements::Tensor, "DelBufRefCount: memId [%d], refcount [%d].", memId, bufRefCount[memId]);
        if (bufRefCount[memId] < 0) {
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
                if ((i < (shape.size() - LAST_TWO_DIM)) && (shape.size() != 1)) {
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
        if (tensor->GetMemoryTypeOriginal() != MemoryType::MEM_DEVICE_DDR) {
            bufRefCount[memId]++;
        }
    }

    void InitBufRefCount() {
        bufRefCount.clear();
        for (const auto &op : operations) {
            inGraph[op].clear();
            outGraph[op].clear();
            for (auto &tensor : op->GetIOperands()) {
                UpdateBufRefCount(tensor);
                int memId = tensor->memoryrange.memId;
                InitLocalBuffer(tensor, memId);
            }
            for (auto &tensor : op->GetOOperands()) {
                UpdateBufRefCount(tensor);
                int memId = tensor->memoryrange.memId;
                InitLocalBuffer(tensor, memId);
            }
        }
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

    Status InitAllocDependencies(Operation* op, std::map<int, Operation*> tensor2AllocMap) {
        for (auto &tensor : op->GetOOperands()) {
            int memId = tensor->memoryrange.memId;
            if (tensor->GetMemoryTypeOriginal() != MemoryType::MEM_DEVICE_DDR) {
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
        if (std::find(operations.begin(), operations.end(), preOp) == operations.end()) {
            return;
        }
        if (std::find(operations.begin(), operations.end(), postOp) == operations.end()) {
            return;
        }
        if (isAlloc || (!IsOpAlloc(preOp) && !IsOpAlloc(postOp))) {
            outGraph[preOp].insert(postOp);
            inGraph[postOp].insert(preOp);
        }
    }

    void FindDependencies(Operation* op) {
        for (auto &producer : op->ProducerOps()) {
            if (producer->GetOpcode() == Opcode::OP_VIEW) {
                for (auto viewProducer : producer->ProducerOps()) {
                    AddDependency(viewProducer, op, false);
                }
                continue;
            }
            AddDependency(producer, op, false);
        }
        for (auto &consumer : op->ConsumerOps()) {
            if (consumer->GetOpcode() == Opcode::OP_VIEW) {
                for (auto viewConsumer : consumer->ConsumerOps()) {
                    AddDependency(op, viewConsumer, false);
                }
            }
        }
    }

    Status InitDependencies() {
        for (const auto &op : operations) {
            inGraph[op].clear();
            outGraph[op].clear();
        }
        std::map<int, Operation*> tensor2AllocMap;
        for (const auto &op : operations) {
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

    void UpdateAllocMap(Operation* op, std::map<int, Operation*> tensorAllocMap) {
        for (auto outTensor : op->GetOOperands()) {
            if (outTensor->GetMemoryTypeOriginal() == MemoryType::MEM_DEVICE_DDR) {
                continue;
            }
            int memId = outTensor->memoryrange.memId;
            if (tensorAllocMap.find(memId) == tensorAllocMap.end()) {
                tensorAllocMap[memId] = op;
            }
        }
        for (auto inTensor : op->GetIOperands()) {
            if (inTensor->GetMemoryTypeOriginal() == MemoryType::MEM_DEVICE_DDR) {
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
                if (GetInOutOperand(op).size() != 1) {
                    APASS_LOG_ERROR_F(Elements::Operation, "%s InOutOperand size not equal to 1.",
                        GetOpInfo(op).c_str());
                    return FAILED;
                }
            }
            UpdateAllocMap(op, tensorAllocMap);
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

    Status Init(std::vector<Operation*> opList) {
        // 初始化芯片各buffer大小
        InitMemorySize();
        operations = opList;
        std::vector<Operation *> newOperations;
        for (auto& op : operations) {
            if (op->GetOpcodeStr().find("ALLOC") != std::string::npos) {
                newOperations.insert(newOperations.begin(), op);
                continue;
            }
            newOperations.push_back(op);
        }
        operations = newOperations;
        InitBufRefCount();
        // 构建依赖关系
        if (InitDependencies() != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "InitDependencies failed!");
            return FAILED;
        }

        if (CheckAllocOp(operations) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "CheckAllocOp failed!");
            return FAILED;
        }
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