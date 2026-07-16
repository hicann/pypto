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
 * \file dev_encode_function_stitch.h
 * \brief
 */

#pragma once

#include "machine/utils/dynamic/dev_encode_types.h"
#include "machine/utils/dynamic/dev_encode_function.h"
#include "machine/utils/dynamic/dev_cell_match_mem_layout.h"
#include "tilefwk/aicpu_common.h"
#include "machine/utils/dynamic/dev_callop_attribute.h"
namespace npu::tile_fwk::dynamic {
// Max cell-match table entry count (desc stride[0]); checked at encode and in host GetWorkspaceSize.
constexpr int64_t MAX_CELLMATCHSSTRIDE = 20000000;
constexpr uint32_t DUPPED_STITCH_SIZE = 0x10 - (sizeof(void*) / sizeof(uint32_t)) - 0x1;
struct DevAscendFunctionDuppedStitch {
    void InitWithNext(DevAscendFunctionDuppedStitch* next)
    {
        next_ = next;
        size_ = 0;
    }

    void PushBack(uint32_t taskId)
    {
        DEV_ASSERT_MSG(ProgEncodeErr::STITCH_LIST_TOO_LARGE, size_ < DUPPED_STITCH_SIZE,
                       "Exceed maximum stitch size %u.", DUPPED_STITCH_SIZE);
        taskList_[size_++] = taskId;
    }

    uint32_t Size() const { return size_; }
    DevAscendFunctionDuppedStitch* const& Next() const { return next_; }
    DevAscendFunctionDuppedStitch*& Next() { return next_; }

    // 函数在核心流程，已在Size()内循环，校验会影响性能
    uint32_t At(uint32_t idx) const { return taskList_[idx]; }

    void ForEach(const std::function<void(uint32_t id)>& callback) const
    {
        for (uint32_t i = 0; i < size_; i++) {
            callback(taskList_[i]);
        }
    }

private:
    DevAscendFunctionDuppedStitch* next_;
    uint32_t size_;
    uint32_t taskList_[DUPPED_STITCH_SIZE];
};

struct DevAscendFunctionDuppedStitchList {
    DevAscendFunctionDuppedStitchList() = default;

    bool IsNull() const { return head_ == nullptr; }

    DevAscendFunctionDuppedStitch* const& Head() const { return head_; }
    DevAscendFunctionDuppedStitch*& Head() { return head_; }

    // Low performance, only used in debug
    void ForEach(const std::function<void(uint32_t id)>& callback) const
    {
        for (auto* p = head_; p != nullptr; p = p->Next()) {
            p->ForEach(callback);
        }
    }

    void PushBack(uint32_t taskId, std::function<DevAscendFunctionDuppedStitch*()> allocate)
    {
        if (head_ == nullptr || head_->Size() == DUPPED_STITCH_SIZE) {
            auto* newNode = allocate();
            DEV_VERBOSE_DEBUG("New node %p", newNode);
            newNode->InitWithNext(head_);
            head_ = newNode;
        }
        head_->PushBack(taskId);
    }

    template <typename T = uint32_t>
    static std::string DumpTask(T id)
    {
        std::ostringstream oss;
        if constexpr (std::is_same<T, uint64_t>::value) {
            oss << (id >> CELL_MATCH_META_TAGID_SHIFT32) << "!"; // devicetaskid
        }
        oss << FuncID(static_cast<uint32_t>(id)) << "!" << TaskID(static_cast<uint32_t>(id));
        return oss.str();
    }

    template <typename T = uint32_t>
    static std::string DumpTask(T* idx, int size)
    {
        std::ostringstream oss;
        oss << "{";
        oss << "size = " << size << " -> ";
        for (int i = 0; i < size; i++) {
            if (idx[i] != AICORE_TASK_INIT) {
                oss << Delim(i != 0, ",");
                oss << "[" << std::dec << i << "]=" << DumpTask<T>(idx[i]);
            }
        }
        oss << "}";
        return oss.str();
    }

    std::string Dump() const
    {
        std::ostringstream oss;

        uint32_t index = 0;
        oss << "[";
        for (auto p = head_; p != nullptr; p = p->Next()) {
            oss << Delim(p != head_, ";");
            for (uint32_t i = 0; i < p->Size(); i++) {
                oss << Delim(i != 0, ",");
                oss << "[" << index++ << "]=" << DumpTask(p->At(i));
            }
        }
        oss << "]";
        return oss.str();
    }

private:
    DevAscendFunctionDuppedStitch* head_{nullptr};
};
static_assert(sizeof(DevAscendFunctionDuppedStitchList) == sizeof(void*));

struct DevAscendProgramPartialUpdate {
    int slotIndex;
    bool isOutputTensorStitchSlot{false};

    DevCellMatchTableDesc cellMatchTableDesc;
    DevRelocVector<uint64_t> cellMatchRuntimePartialUpdateTable;

    bool Empty() const { return cellMatchRuntimePartialUpdateTable.size() == 0; }
};

template <typename HandleType, typename... TyArgs>
static uint32_t CellMatch5Dimension(const DevCellMatchTableDesc& cellMatchTableDesc, uint64_t* rangeBegin,
                                    uint64_t* rangeEnd, TyArgs... args)
{
    uint32_t errCode = 0;
    int s0 = cellMatchTableDesc.GetStride(1), s1 = cellMatchTableDesc.GetStride(2);
    int s2 = cellMatchTableDesc.GetStride(3), s3 = cellMatchTableDesc.GetStride(4), s4 = 1;
    for (int d0 = 0 + rangeBegin[0] * s0, e0 = 0 + rangeEnd[0] * s0; d0 <= e0; d0 += s0) {
        for (int d1 = d0 + rangeBegin[1] * s1, e1 = d0 + rangeEnd[1] * s1; d1 <= e1; d1 += s1) {
            for (int d2 = d1 + rangeBegin[2] * s2, e2 = d1 + rangeEnd[2] * s2; d2 <= e2; d2 += s2) {
                for (int d3 = d2 + rangeBegin[3] * s3, e3 = d2 + rangeEnd[3] * s3; d3 <= e3; d3 += s3) {
                    for (int d4 = d3 + rangeBegin[4] * s4, e4 = d3 + rangeEnd[4] * s4; d4 <= e4; d4 += s4) {
                        errCode = HandleType::Process(d4, args...);
                        if (errCode != 0) {
                            return errCode;
                        }
                    }
                }
            }
        }
    }
    return errCode;
}

template <typename HandleType, typename... TyArgs>
static uint32_t CellMatch4Dimension(const DevCellMatchTableDesc& cellMatchTableDesc, uint64_t* rangeBegin,
                                    uint64_t* rangeEnd, TyArgs... args)
{
    uint32_t errCode = 0;
    int s0 = cellMatchTableDesc.GetStride(1), s1 = cellMatchTableDesc.GetStride(2);
    int s2 = cellMatchTableDesc.GetStride(3), s3 = 1;
    for (int d0 = 0 + rangeBegin[0] * s0, e0 = 0 + rangeEnd[0] * s0; d0 <= e0; d0 += s0) {
        for (int d1 = d0 + rangeBegin[1] * s1, e1 = d0 + rangeEnd[1] * s1; d1 <= e1; d1 += s1) {
            for (int d2 = d1 + rangeBegin[2] * s2, e2 = d1 + rangeEnd[2] * s2; d2 <= e2; d2 += s2) {
                for (int d3 = d2 + rangeBegin[3] * s3, e3 = d2 + rangeEnd[3] * s3; d3 <= e3; d3 += s3) {
                    errCode = HandleType::Process(d3, args...);
                    if (errCode != 0) {
                        return errCode;
                    }
                }
            }
        }
    }
    return errCode;
}

template <typename HandleType, typename... TyArgs>
static uint32_t CellMatchProcessByDim(const DevCellMatchTableDesc& cellMatchTableDesc, uint64_t* rangeBegin,
                                      uint64_t* rangeEnd, TyArgs... args)
{
    uint32_t errCode = 0;
    switch (cellMatchTableDesc.cellShape.dimSize) {
        case 1: {
            int s0 = 1;
            for (int d0 = 0 + rangeBegin[0] * s0, e0 = 0 + rangeEnd[0] * s0; d0 <= e0; d0 += s0) {
                errCode = HandleType::Process(d0, args...);
                if (errCode != 0) {
                    return errCode;
                }
            }
        } break;
        case DEV_SHAPE_DIM_NUM_2: {
            int s0 = cellMatchTableDesc.GetStride(1), s1 = 1;
            for (int d0 = 0 + rangeBegin[0] * s0, e0 = 0 + rangeEnd[0] * s0; d0 <= e0; d0 += s0)
                for (int d1 = d0 + rangeBegin[1] * s1, e1 = d0 + rangeEnd[1] * s1; d1 <= e1; d1 += s1) {
                    errCode = HandleType::Process(d1, args...);
                    if (errCode != 0) {
                        return errCode;
                    }
                }
        } break;
        case DEV_SHAPE_DIM_NUM_3: {
            int s0 = cellMatchTableDesc.GetStride(1), s1 = cellMatchTableDesc.GetStride(2), s2 = 1;
            for (int d0 = 0 + rangeBegin[0] * s0, e0 = 0 + rangeEnd[0] * s0; d0 <= e0; d0 += s0)
                for (int d1 = d0 + rangeBegin[1] * s1, e1 = d0 + rangeEnd[1] * s1; d1 <= e1; d1 += s1)
                    for (int d2 = d1 + rangeBegin[2] * s2, e2 = d1 + rangeEnd[2] * s2; d2 <= e2; d2 += s2) {
                        errCode = HandleType::Process(d2, args...);
                        if (errCode != 0) {
                            return errCode;
                        }
                    }
        } break;
        case DEV_SHAPE_DIM_NUM_4: {
            errCode = CellMatch4Dimension<HandleType>(cellMatchTableDesc, rangeBegin, rangeEnd, args...);
            if (errCode != 0) {
                return errCode;
            }
        } break;
        case DEV_SHAPE_DIM_NUM_5: {
            errCode = CellMatch5Dimension<HandleType>(cellMatchTableDesc, rangeBegin, rangeEnd, args...);
            if (errCode != 0) {
                return errCode;
            }
        } break;
        default:
            DEV_ERROR(ProgEncodeErr::CELL_MATCH_PARAM_INVALID,
                      "#ctrl.encode.stitch.dim: [Stitch] Too many dimensions: dimSize=%d\n",
                      (int)cellMatchTableDesc.GetDimensionSize());
            break;
    }
    return errCode;
}

template <typename HandleType, typename... TyArgs>
static uint32_t CellMatchHandle(const uint64_t offset[DEV_SHAPE_DIM_MAX], const uint64_t shape[DEV_SHAPE_DIM_MAX],
                                const DevCellMatchTableDesc& cellMatchTableDesc, TyArgs... args)
{
    uint64_t rangeBegin[DEV_SHAPE_DIM_MAX];
    uint64_t rangeEnd[DEV_SHAPE_DIM_MAX];
    for (int i = 0; i < cellMatchTableDesc.GetDimensionSize(); ++i) {
        auto cellMatchShapeDim = cellMatchTableDesc.GetCellShape(i);
        if (cellMatchShapeDim != 0) {
            rangeBegin[i] = offset[i] / cellMatchShapeDim;
            if (shape[i] == 0) {
                return 0;
            }
            rangeEnd[i] = (offset[i] + shape[i] - 1) / cellMatchShapeDim;
        } else {
            DEV_ERROR(ProgEncodeErr::CELL_MATCH_DIM_ZERO,
                      "#ctrl.encode.cell_match: CellMatchGetIndexRange: cellMatchShapeDim is zero for dimension=%d", i);
            DEV_ASSERT(ProgEncodeErr::CELL_MATCH_DIM_ZERO, 0);
        }
    }
    return CellMatchProcessByDim<HandleType>(cellMatchTableDesc, rangeBegin, rangeEnd, args...);
}

template <typename... TyArgs>
static uint32_t CellMatchFill(const uint64_t offset[DEV_SHAPE_DIM_MAX], const uint64_t shape[DEV_SHAPE_DIM_MAX],
                              uint32_t operationIdx, const DevCellMatchTableDesc& cellMatchTableDesc, TyArgs... args)
{
    if constexpr (sizeof...(args) == 1) {
        auto argsTuple = std::make_tuple(args...);
        uint32_t* cellMatchTableData = std::get<0>(argsTuple);
        struct HandleFill {
            static inline uint32_t Process(int index, uint32_t* cellMatchTableData, uint32_t operationIdx)
            {
                cellMatchTableData[index] = operationIdx;
                DEV_VERBOSE_DEBUG("cell match fill, operation %u , cellindex[%d] = operationindex(%u)", operationIdx,
                                  index, operationIdx);
                return 0;
            }
        };
        return CellMatchHandle<HandleFill>(offset, shape, cellMatchTableDesc, cellMatchTableData, operationIdx);
    }
    if constexpr (sizeof...(args) == 3) {
        auto argsTuple = std::make_tuple(args...);
        uint64_t* cellMatchTableData = std::get<0>(argsTuple);
        uint32_t tagId = std::get<1>(argsTuple);
        uint32_t funcIdx = std::get<2>(argsTuple);
        struct HandleFill {
            static inline uint32_t Process(int index, uint64_t* cellMatchTableData, uint32_t tagId, uint32_t funcIdx,
                                           uint32_t operationIdx)
            {
                cellMatchTableData[index] = (static_cast<uint64_t>(tagId) << CELL_MATCH_META_TAGID_SHIFT32) |
                                            MakeTaskID(funcIdx, operationIdx);
                DEV_VERBOSE_DEBUG("cell match fill, tagid:%u funcIdx %u operation %u , cellindex[%d] = taskid(%lx)",
                                  tagId, funcIdx, operationIdx, index, cellMatchTableData[index]);
                return 0;
            }
        };
        return CellMatchHandle<HandleFill>(offset, shape, cellMatchTableDesc, cellMatchTableData, tagId, funcIdx,
                                           operationIdx);
    }
    return 0;
}

inline uint32_t CellMatchHandleFillEnhanceExec(int cellIndex, uint64_t* cellMatchTableData, uint32_t myOpType,
                                               uint64_t updateTagId, uint32_t updateFuncIdx, uint32_t operationIdx,
                                               const DevCellMatchTableDesc& desc)
{
    DEV_VERBOSE_DEBUG("CellMatchHandleFillEnhanceExec: cell[%d], cellMatchTableData=%p", cellIndex, cellMatchTableData);
    uint64_t cellMemBase = CellMatchCellIndexToMemBase(static_cast<uint64_t>(cellIndex), desc);
    uint64_t meta = cellMatchTableData[cellMemBase];
    uint32_t curActiveOpType = CellMatchGetCurrentOpType(meta);
    uint32_t curActiveOpCount = CellMatchGetCurrentOpCount(meta);
    uint64_t curTagId = CellMatchGetTagId(meta);
    uint32_t targetCount = 0, targetIndex = 0;
    uint32_t maxCount = desc.GetCacheOpMaxCount(myOpType);
    if (maxCount == 0) {
        DEV_VERBOSE_DEBUG("Op type %u not supported in cell[%d], maxCount=0", myOpType, cellIndex);
        return static_cast<uint32_t>(CtrlErr::CELL_MATCH_OP_TYPE_NOT_SUPPORTED);
    }

    if (CellMatchIsMutexOp(myOpType, curActiveOpType)) {
        CellMatchSetCurrentOpType(meta, myOpType);
        CellMatchSetPrevMutexOpType(meta, curActiveOpType);
        CellMatchSetPrevMutexOpCount(meta, curActiveOpCount);
        targetCount = 1;
        targetIndex = 0;
        DEV_VERBOSE_DEBUG("Update mutex op: cell[%d], prev mutex type=%u (count=%u), active=%u (count=1)", cellIndex,
                          curActiveOpType, curActiveOpCount, myOpType);
    } else {
        targetCount = (curTagId != updateTagId) ? 1 : curActiveOpCount + 1;
        if (targetCount <= maxCount) {
            targetIndex = targetCount - 1;
            DEV_VERBOSE_DEBUG("Update multi-concurrent op : cell[%d], active=%u, count=%u -> %u", cellIndex, myOpType,
                              curActiveOpCount, targetCount);
        } else {
            DEV_VERBOSE_DEBUG("Op count not enough for cell[%d], opType=%u, newCount=%u, maxCount=%u", cellIndex,
                              myOpType, targetCount, maxCount);
            return static_cast<uint32_t>(CtrlErr::CELL_MATCH_FILL_OP_NOT_ENOUGH);
        }
    }

    if (curTagId != updateTagId) {
        // set pre mutex op as dirtry data, it's invalid
        CellMatchSetPrevMutexOpType(meta, CELL_MATCH_OP_TYPE_NONE);
        CellMatchSetPrevMutexOpCount(meta, CELL_MATCH_INVALID_OP_COUNT);
    }

    CellMatchSetCurrentOpCount(meta, targetCount);
    CellMatchSetTagId(meta, updateTagId);

    uint64_t taskId = (static_cast<uint64_t>(updateTagId) << CELL_MATCH_META_TAGID_SHIFT32) |
                      MakeTaskID(updateFuncIdx, operationIdx);
    CellMatchAddOpId(cellMatchTableData, cellMemBase, taskId, targetIndex, myOpType, desc);
    DEV_VERBOSE_DEBUG("Added opId to cell[%d]: taskId=0x%lx (Tagid=%lx, funcIdx=%u, opIdx=%u), index=%u, opType=%u",
                      cellIndex, taskId, updateTagId, updateFuncIdx, operationIdx, targetIndex, myOpType);
    cellMatchTableData[cellMemBase] = meta;
    return 0;
}

template <typename... TyArgs>
static uint32_t CellMatchFillEnhance(const uint64_t offset[DEV_SHAPE_DIM_MAX], const uint64_t shape[DEV_SHAPE_DIM_MAX],
                                     uint32_t operationIdx, const DevCellMatchTableDesc& cellMatchTableDesc,
                                     uint32_t opType, TyArgs... args)
{
    if constexpr (sizeof...(args) == 1) {
        UNUSED(opType);
        auto argsTuple = std::make_tuple(args...);
        uint32_t* cellMatchTableData = std::get<0>(argsTuple);
        struct HandleFillFull {
            static inline uint32_t Process(int index, uint32_t* cellMatchTableData, uint32_t operationIdx)
            {
                cellMatchTableData[index] = operationIdx;
                DEV_VERBOSE_DEBUG("cell match fill full, operation %u , cellindex[%d] = operationindex(%u)",
                                  operationIdx, index, operationIdx);
                return 0;
            }
        };
        return CellMatchHandle<HandleFillFull>(offset, shape, cellMatchTableDesc, cellMatchTableData, operationIdx);
    }

    if constexpr (sizeof...(args) == 3) {
        auto argsTuple = std::make_tuple(args...);
        uint64_t* cellMatchTableData = std::get<0>(argsTuple);
        uint64_t tagId = std::get<1>(argsTuple);
        uint32_t updateFuncIdx = std::get<2>(argsTuple);

        struct HandleFillEnhance {
            static inline uint32_t Process(int cellIndex, uint64_t* data, uint64_t tagId, uint32_t funcIdx,
                                           uint32_t opIdx, uint32_t type, const DevCellMatchTableDesc& desc)
            {
                return CellMatchHandleFillEnhanceExec(cellIndex, data, type, tagId, funcIdx, opIdx, desc);
            }
        };

        return CellMatchHandle<HandleFillEnhance>(offset, shape, cellMatchTableDesc, cellMatchTableData, tagId,
                                                  updateFuncIdx, operationIdx, opType, cellMatchTableDesc);
    }
    return 0;
}

template <bool skipExpression, typename... TyArgs>
static uint32_t CellMatchFillIncastOutcast(DevAscendFunction* devFunc, DevAscendFunctionCallOperandUse* operandUseList,
                                           size_t useSize, const uint64_t* runtimeExpressionList,
                                           const DevCellMatchTableDesc& cellMatchTableDesc, TyArgs... args)
{
    if (!IsCellMatchDescFillReady(cellMatchTableDesc)) {
        return 0;
    }

    for (size_t i = 0; i < useSize; i++) {
        auto& use = operandUseList[i];
        uint64_t offset[DEV_SHAPE_DIM_MAX];
        uint64_t validShape[DEV_SHAPE_DIM_MAX];

        bool paramConcrete = GetTensorOffsetAndValidShape<skipExpression>(
            devFunc, offset, validShape, runtimeExpressionList, cellMatchTableDesc,
            cellMatchTableDesc.GetDimensionSize(), use.operationIdx, use.offsetAttrIdx);

        DEV_IF_VERBOSE_DEBUG
        {
            for (int j = 0; j < cellMatchTableDesc.GetDimensionSize(); j++) {
                DEV_VERBOSE_DEBUG("CellMatchFillIncastOutcast, op[%d] -> opType:%u -> dimension[%d] = (offset:%lu "
                                  ", validShape:%lu, cellshape:%d)",
                                  use.operationIdx, static_cast<uint32_t>(use.opType), j, offset[j], validShape[j],
                                  cellMatchTableDesc.cellShape.dim[j]);
            }
        }

        if (paramConcrete) {
            uint32_t errCode = CellMatchFillEnhance(offset, validShape, use.operationIdx, cellMatchTableDesc,
                                                    static_cast<uint32_t>(use.opType), args...);
            if (errCode != 0) {
                return errCode;
            }
        }
    }
    return 0;
}
} // namespace npu::tile_fwk::dynamic
