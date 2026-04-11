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
#include "passes/block_graph_pass/schedule_ooo/dep_manager.h"
#include "passes/pass_utils/reschedule_utils.h"
#include "passes/pass_utils/pass_utils.h"

#ifdef MODULE_NAME
#undef MODULE_NAME
#endif
#define MODULE_NAME "OoOScheduleBase"

namespace npu::tile_fwk {

constexpr int32_t DIM_FIVE = 5;
constexpr int32_t LAST_TWO_DIM = 2;
constexpr int32_t UB_BLOCK_SIZE = 32;

const std::unordered_set<Opcode> USE_LESS_OPS2 = {
    Opcode::OP_NOP,      Opcode::OP_RESHAPE,     Opcode::OP_SHMEM_WAIT_UNTIL, Opcode::OP_VIEW,
    Opcode::OP_ASSEMBLE, Opcode::OP_BIND_TENSOR, Opcode::OP_VIEW_TYPE,        Opcode::OP_HUB};

inline int BytesPerElement2(DataType dataType) { return BytesOf(dataType); }

inline uint64_t CeilAlign2(uint64_t a, int b) { return ((a + b - 1) / b) * b; }

class ScheduleBase {
public:
    ScheduleBase() {}
    ~ScheduleBase() {}

    std::unordered_map<int, int> bufRefCount_;
    std::unordered_map<MemoryType, int64_t> localMemSize; //内存剩余情况
    std::unordered_map<MemoryType, int64_t> localMemoryCurrentSize;
    std::unordered_map<int, LocalBufferPtr> localBufferMap_; //memid:local
    std::unordered_map<Operation*, LogicalTensors> inOutOperandsCache_;
    std::unordered_map<Operation*, std::vector<int>> opReqMemIdsMap;

    //  初始依赖的list序列
    std::vector<Operation*> operations;

protected:
    DependencyManager depManager_;

public:
    std::vector<int>& GetOpMemIds(Operation* op)
    {
        auto it = opReqMemIdsMap.find(op);
        if (it != opReqMemIdsMap.end()) {
            return it->second;
        }
        std::vector<int> memIds;
        for (auto tensor : GetInOutOperandCached(op)) {
            memIds.push_back(tensor->memoryrange.memId);
        }
        auto inserted = opReqMemIdsMap.emplace(op, std::move(memIds));
        return inserted.first->second;
    }

    void SetOpMemIds(Operation* op, const std::vector<int>& memIds)
    {
        opReqMemIdsMap[op] = memIds;
    }

    void ClearOpMemIds(Operation* op)
    {
        opReqMemIdsMap[op].clear();
    }

    void AddOpMemId(Operation* op, int memId)
    {
        opReqMemIdsMap[op].push_back(memId);
    }

    void ClearAllOpMemIds()
    {
        opReqMemIdsMap.clear();
    }

    bool ReplaceOpMemId(Operation* op, int oldMemId, int newMemId)
    {
        auto& memIds = opReqMemIdsMap[op];
        bool replaced = false;
        for (auto& memId : memIds) {
            if (memId == oldMemId) {
                memId = newMemId;
                replaced = true;
            }
        }
        return replaced;
    }

    Status InitLocalBuffer(LogicalTensorPtr oOperand, int memId) {
        if (oOperand->GetMemoryTypeOriginal() >= MemoryType::MEM_DEVICE_DDR) {
            return SUCCESS;
        }
        if (static_cast<uint64_t>(oOperand->tensor->GetRawDataSize()) !=
            ShapeCeilAlign(oOperand->tensor->rawshape, oOperand->tensor->datatype)) {
            APASS_LOG_WARN_F(
                Elements::Tensor,
                "InitLocalBuffer Failed at ShapeCeilAlign! "
                "Please ensure that the rawTensor[%d] shapes are aligned.",
                oOperand->GetRawMagic());
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

    std::string GetOpInfo(Operation* op) { return op->GetOpcodeStr() + "[" + std::to_string(op->GetOpMagic()) + "]"; }

    Status DelBufRefCount(const int memId)
    {
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

    uint64_t ShapeCeilAlign(std::vector<int64_t> shape, DataType dtype)
    {
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

    const LogicalTensors& GetInOutOperandCached(Operation* op) {
        auto it = inOutOperandsCache_.find(op);
        if (it != inOutOperandsCache_.end())
            return it->second;
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

    void UpdateBufRefCount(Operation* op, LogicalTensorPtr tensor)
    {
        int memId = tensor->memoryrange.memId;
        if (tensor->GetMemoryTypeOriginal() < MemoryType::MEM_DEVICE_DDR) {
            bufRefCount_[memId]++;
            opReqMemIdsMap[op].push_back(memId);
        }
    }

    Status InitBufRefCount(std::vector<Operation*> &list)
    {
        bufRefCount_.clear();
        depManager_.ClearDependencies();
        localBufferMap_.clear();
        inOutOperandsCache_.clear();
        opReqMemIdsMap.clear();
        for (const auto &op : list) {
            for (auto &tensor : op->GetIOperands()) {
                UpdateBufRefCount(op, tensor);
                int memId = tensor->memoryrange.memId;
                if (InitLocalBuffer(tensor, memId) == FAILED) {
                    APASS_LOG_ERROR_F(Elements::Operation, "InitLocalBuffer failed at InitBufRefCount!");
                    return FAILED;
                }
            }
            for (auto& tensor : op->GetOOperands()) {
                UpdateBufRefCount(op, tensor);
                int memId = tensor->memoryrange.memId;
                if (InitLocalBuffer(tensor, memId) == FAILED) {
                    APASS_LOG_ERROR_F(Elements::Operation, "InitLocalBuffer failed at InitBufRefCount!");
                    return FAILED;
                }
            }
        }
        return SUCCESS;
    }

    bool IsOpAlloc(Operation *op) {
        if (op == nullptr) {
            return false;
        }
        return op->GetOpcodeStr().find("ALLOC") != std::string::npos;
    }

    Status CalcBufferSize(LogicalTensors tensors, std::map<MemoryType, int64_t>& bufferSize, std::set<int>& memIdMap)
    {
        for (auto tensor : tensors) {
            if (tensor->GetMemoryTypeOriginal() == MemoryType::MEM_DEVICE_DDR) {
                continue;
            }
            const auto& shape = tensor->tensor->GetRawShape();
            if (std::any_of(shape.begin(), shape.end(), [](int64_t d) { return d <= 0; })) {
                APASS_LOG_ERROR_F(
                    Elements::Tensor,
                    "Dynamic axis detected in %s, "
                    "OoOSchedule requires static rawShape!",
                    tensor->Dump().c_str());
                return FAILED;
            }
            if (memIdMap.find(tensor->memoryrange.memId) == memIdMap.end()) {
                bufferSize[tensor->GetMemoryTypeOriginal()] += tensor->tensor->GetRawDataSize();
                memIdMap.insert(tensor->memoryrange.memId);
            }
        }
        return SUCCESS;
    }

    std::string DumpOpInfo(Operation& op)
    {
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
        os << " }"
           << " | ";
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

    Status CheckOpBufferSize(Operation* op)
    {
        std::map<MemoryType, int64_t> bufferSizeMap;
        std::set<int> memIdMap;
        if (CalcBufferSize(op->GetIOperands(), bufferSizeMap, memIdMap) != SUCCESS ||
            CalcBufferSize(op->GetOOperands(), bufferSizeMap, memIdMap) != SUCCESS) {
            return FAILED;
        }
        for (auto& bufferPair : bufferSizeMap) {
            if (localMemSize.find(bufferPair.first) == localMemSize.end()) {
                continue;
            }
            if (bufferPair.second <= localMemSize[bufferPair.first]) {
                continue;
            }
            if (op->GetOpcodeStr().find("ALLOC") != std::string::npos) {
                APASS_LOG_ERROR_F(
                    Elements::Operation, "Alloc tensor [%d] size [%ld] exceeds %s size [%ld]! %s",
                    op->GetOutputOperand(0)->GetMagic(), bufferPair.second,
                    MemoryTypeToString(bufferPair.first).c_str(), localMemSize[bufferPair.first],
                    GetFormatBacktrace(*op).c_str());
                APASS_LOG_ERROR_F(
                    Elements::Operation, "Tensor [%d] producer info:", op->GetOutputOperand(0)->GetMagic());
                for (auto producer : op->GetOutputOperand(0)->GetProducers()) {
                    if (producer == op) {
                        continue;
                    }
                    APASS_LOG_ERROR_F(Elements::Operation, "      %s.", DumpOpInfo(*producer).c_str());
                }
            } else {
                APASS_LOG_ERROR_F(
                    Elements::Operation, "OP %s[%d] in/output total size [%ld] exceeds %s size [%ld]!",
                    op->GetOpcodeStr().c_str(), op->GetOpMagic(), bufferPair.second,
                    MemoryTypeToString(bufferPair.first).c_str(), localMemSize[bufferPair.first]);
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

    Status CheckAllocOp(std::vector<Operation*> list)
    {
        std::map<int, Operation*> tensorAllocMap;
        for (const auto& op : list) {
            if (IsOpAlloc(op)) {
                if (GetInOutOperandCached(op).size() != 1) {
                    APASS_LOG_ERROR_F(
                        Elements::Operation, "%s InOutOperand size not equal to 1.", GetOpInfo(op).c_str());
                    return FAILED;
                }
                UpdateAllocMap(op, tensorAllocMap);
            }
        }
        for (const auto& op : list) {
            if (!IsOpAlloc(op)) {
                UpdateAllocMap(op, tensorAllocMap);
            }
        }
        for (auto tensorAlloc : tensorAllocMap) {
            if (!IsOpAlloc(tensorAlloc.second)) {
                APASS_LOG_ERROR_F(
                    Elements::Tensor, "%s Tensor[%d] is missing Alloc.", GetOpInfo(tensorAlloc.second).c_str(),
                    tensorAlloc.first);
                return FAILED;
            }
        }
        return SUCCESS;
    }

    Status Init(std::vector<Operation*> &opList) {
        // 初始化芯片各buffer大小
        localMemSize = CommonUtils::GetLocalMemorySize();
 	    localMemoryCurrentSize = localMemSize;
        operations = opList;
        for (auto& op : operations) {
            if (CheckOpBufferSize(op) != SUCCESS) {
                APASS_LOG_ERROR_F(
                    Elements::Operation, "%s[%d] checkOpBufferSize failed! %s", op->GetOpcodeStr().c_str(),
                    op->GetOpMagic(), GetFormatBacktrace(*op).c_str());
                return FAILED;
            }
        }
        InitBufRefCount(operations);
        // 构建依赖关系
        if (depManager_.InitDependencies(operations, true) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "InitDependencies failed!");
            return FAILED;
        }
        depManager_.PrintDependencies(operations);

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

    Operation* PopFront()
    {
        Operation* op = queue.front();
        queue.pop_front();
        return op;
    }
};
} // namespace npu::tile_fwk
#endif // PASS_SCHEDULE_BASE_H
