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
 * \brief Proxy layer over ScheduleState. Public interface unchanged for backward compatibility.
 */

#ifndef PASS_SCHEDULE_BASE_H
#define PASS_SCHEDULE_BASE_H

#include "passes/block_graph_pass/schedule_ooo/common/schedule_state.h"

#ifdef MODULE_NAME
#undef MODULE_NAME
#endif
#define MODULE_NAME "OoOScheduleBase"

namespace npu::tile_fwk {

class ScheduleBase {
public:
    ScheduleBase() {}
    ~ScheduleBase() {}

    ScheduleState state_;

    // === Proxy fields: redirect to ScheduleState ===
    std::unordered_map<int, int>& bufRefCount_ = state_.bufRefCount;
    std::unordered_map<MemoryType, int64_t>& localMemSize = state_.localMemSize;
    std::unordered_map<MemoryType, int64_t>& localMemoryCurrentSize = state_.localMemoryCurrentSize;
    std::unordered_map<int, LocalBufferPtr>& localBufferMap_ = state_.localBufferMap;
    std::unordered_map<Operation*, LogicalTensors>& inOutOperandsCache_ = state_.inOutOperandsCache;
    std::unordered_map<Operation*, std::vector<int>>& opReqMemIdsMap = state_.opReqMemIdsMap;
    std::vector<Operation*>& operations = state_.operations;

    // === Proxy fields: OoO-specific (redirect to ScheduleState) ===
    std::unordered_map<Operation*, OpSchedInfo>& schedInfoMap_ = state_.schedInfoMap;
    std::unordered_map<int, Operation*>& tensorOccupyMap = state_.tensorOccupyMap;
    std::unordered_map<int, Operation*>& tensorAllocMap = state_.tensorAllocMap;
    std::unordered_map<CoreLocationType, std::map<MemoryType, BufferPool>>& bufferManagerMap = state_.bufferManagerMap;
    std::vector<Operation*>& newOperations_ = state_.newOperations;
    std::unordered_set<CoreLocationType>& CORE_INIT_CONFIGS = state_.coreInitConfigs;
    std::unordered_map<int, CoreLocationType>& dualDstMemIdCoreOverride_ = state_.dualDstMemIdCoreOverride;
    int64_t& workspaceOffset = state_.workspaceOffset;
    std::unordered_map<PipeType, int>& pipeEndTime = state_.pipeEndTime;
    int& workspaceMemId = state_.workspaceMemId;

    // === Proxy fields: OoO scheduling state (redirect to ScheduleState) ===
    std::vector<Operation*>& orderedOps = state_.orderedOps;
    int& clock = state_.clock;
    uint64_t& numTotalIssues = state_.numTotalIssues;
    std::unordered_map<CoreLocationType, std::map<MemoryType, OpQueue>>& allocIssueQueue = state_.allocIssueQueue;

protected:
    DependencyManager& depManager_ = state_.depManager;

public:
    // === Proxy methods: redirect to ScheduleState ===

    std::vector<int>& GetOpMemIds(Operation* op)
    {
        return state_.GetOpMemIds(op);
    }

    void SetOpMemIds(Operation* op, const std::vector<int>& memIds)
    {
        state_.SetOpMemIds(op, memIds);
    }

    void ClearOpMemIds(Operation* op)
    {
        state_.ClearOpMemIds(op);
    }

    void AddOpMemId(Operation* op, int memId)
    {
        state_.AddOpMemId(op, memId);
    }

    void ClearAllOpMemIds()
    {
        state_.ClearAllOpMemIds();
    }

    bool ReplaceOpMemId(Operation* op, int oldMemId, int newMemId)
    {
        return state_.ReplaceOpMemId(op, oldMemId, newMemId);
    }

    Status InitLocalBuffer(LogicalTensorPtr oOperand, int memId) {
        return state_.InitLocalBuffer(oOperand, memId);
    }

    std::string GetOpInfo(Operation* op) const {
        return state_.GetOpInfo(op);
    }

    Status DelBufRefCount(const int memId)
    {
        return state_.DelBufRefCount(memId);
    }

    uint64_t ShapeCeilAlign(std::vector<int64_t> shape, DataType dtype)
    {
        return state_.ShapeCeilAlign(shape, dtype);
    }

    const LogicalTensors& GetInOutOperandCached(Operation* op) {
        return state_.GetInOutOperandCached(op);
    }

    void UpdateBufRefCount(Operation* op, LogicalTensorPtr tensor)
    {
        state_.UpdateBufRefCount(op, tensor);
    }

    Status InitBufRefCount(std::vector<Operation*> &list)
    {
        return state_.InitBufRefCount(list);
    }

    bool IsOpAlloc(Operation *op) {
        return state_.IsOpAlloc(op);
    }

    Status CalcBufferSize(LogicalTensors tensors, std::map<MemoryType, int64_t>& bufferSize, std::set<int>& memIdMap)
    {
        return state_.CalcBufferSize(tensors, bufferSize, memIdMap);
    }

    std::string DumpOpInfo(Operation& op)
    {
        return state_.DumpOpInfo(op);
    }

    Status CheckOpBufferSize(Operation* op)
    {
        return state_.CheckOpBufferSize(op);
    }

    void UpdateAllocMap(Operation* op, std::map<int, Operation*> &allocMap) {
        state_.UpdateAllocMap(op, allocMap);
    }

    Status CheckAllocOp(std::vector<Operation*> list)
    {
        return state_.CheckAllocOp(list);
    }

    Status Init(std::vector<Operation*> &opList) {
        return state_.Init(opList);
    }

    void InsertOrdered(Operation* insertOp) {
        state_.InsertOrdered(insertOp);
    }
};

} // namespace npu::tile_fwk
#endif // PASS_SCHEDULE_BASE_H
