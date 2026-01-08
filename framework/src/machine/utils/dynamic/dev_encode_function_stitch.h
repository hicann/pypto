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
#include "tilefwk/aicpu_common.h"

namespace npu::tile_fwk::dynamic {
constexpr uint32_t DUPPED_STITCH_SIZE  = 0x10 - (sizeof(void *) / sizeof(uint32_t)) - 0x1;
struct DevAscendFunctionDuppedStitch {
    void InitWithNext(DevAscendFunctionDuppedStitch *next) {
        next_ = next;
        size_ = 0;
    }

    void PushBack(uint32_t taskId) {
        DEV_ASSERT_MSG(size_ < DUPPED_STITCH_SIZE, "Exceed maximum stitch size %u.", DUPPED_STITCH_SIZE);
        taskList_[size_++] = taskId;
    }

    uint32_t Size() const { return size_; }
    DevAscendFunctionDuppedStitch * const &Next() const { return next_; }
    DevAscendFunctionDuppedStitch *&Next() { return next_; }

    uint32_t At(uint32_t idx) const {
        DEV_ASSERT_MSG(idx < size_, "Index %u exceeds stitch size %u.", idx, size_);
        return taskList_[idx];
    }

    void ForEach(const std::function<void(uint32_t id)> &callback) const {
        for (uint32_t i = 0; i < size_; i++) {
            callback(taskList_[i]);
        }
    }

private:
    DevAscendFunctionDuppedStitch *next_;
    uint32_t size_;
    uint32_t taskList_[DUPPED_STITCH_SIZE];
};

struct DevAscendFunctionDuppedStitchList {
    DevAscendFunctionDuppedStitchList() = default;

    bool IsNull() const { return head_ == nullptr; }

    DevAscendFunctionDuppedStitch* const &Head() const { return head_; }
    DevAscendFunctionDuppedStitch* &Head() { return head_; }

    // Low performance, only used in debug
    void ForEach(const std::function<void(uint32_t id)> &callback) const {
        for (auto *p = head_; p != nullptr; p = p->Next()) {
            p->ForEach(callback);
        }
    }

    void PushBack(uint32_t taskId, std::function<DevAscendFunctionDuppedStitch *()> allocate) {
        if (head_ == nullptr || head_->Size() == DUPPED_STITCH_SIZE) {
            auto *newNode = allocate();
            newNode->InitWithNext(head_);
            head_ = newNode;
        }
        head_->PushBack(taskId);
    }

    template<typename T = uint32_t>
    static std::string DumpTask(T id) {
        std::ostringstream oss;
        if constexpr (std::is_same<T, uint64_t>::value) {
            oss << (id >> TASKID_SHIFT32) << "!"; // devicetaskid
        }
        oss << FuncID(static_cast<uint32_t>(id)) << "!" << TaskID(static_cast<uint32_t>(id));
        return oss.str();
    }

    template<typename T = uint32_t>
    static std::string DumpTask(T *idx, int size) {
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

    std::string Dump() const {
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
    DevAscendFunctionDuppedStitch *head_{nullptr};
};
static_assert(sizeof(DevAscendFunctionDuppedStitchList) == sizeof(void *));

struct DevAscendProgramPartialUpdate {
    int slotIndex;

    DevCellMatchTableDesc cellMatchTableDesc;
    DevRelocVector<uint64_t> cellMatchRuntimePartialUpdateTable; // devtaskid | taskid

    bool Empty() const {
        return cellMatchRuntimePartialUpdateTable.size() == 0;
    }
};

template<typename HandleType, typename ...TyArgs>
static void CellMatch5Dimension(const DevCellMatchTableDesc &cellMatchTableDesc, uint64_t* rangeBegin, uint64_t* rangeEnd, TyArgs ... args) {
    int s0 = cellMatchTableDesc.GetStride(1), s1 = cellMatchTableDesc.GetStride(2);
    int s2 = cellMatchTableDesc.GetStride(3), s3 = cellMatchTableDesc.GetStride(4), s4 = 1;
    for (int d0 =  0 + rangeBegin[0] * s0, e0 =  0 + rangeEnd[0] * s0; d0 <= e0; d0 += s0) {
        for (int d1 = d0 + rangeBegin[1] * s1, e1 = d0 + rangeEnd[1] * s1; d1 <= e1; d1 += s1) {
            for (int d2 = d1 + rangeBegin[2] * s2, e2 = d1 + rangeEnd[2] * s2; d2 <= e2; d2 += s2) {
                for (int d3 = d2 + rangeBegin[3] * s3, e3 = d2 + rangeEnd[3] * s3; d3 <= e3; d3 += s3) {
                    for (int d4 = d3 + rangeBegin[4] * s4, e4 = d3 + rangeEnd[4] * s4; d4 <= e4; d4 += s4) {
                        HandleType::Process(d4, args...);
                    }
                }
            }

        }

    }

}

template<typename HandleType, typename ...TyArgs>
static void CellMatch4Dimension(const DevCellMatchTableDesc &cellMatchTableDesc, uint64_t* rangeBegin, uint64_t* rangeEnd, TyArgs ... args) {
    int s0 = cellMatchTableDesc.GetStride(1), s1 = cellMatchTableDesc.GetStride(2);
    int s2 = cellMatchTableDesc.GetStride(3), s3 = 1;
    for (int d0 =  0 + rangeBegin[0] * s0, e0 =  0 + rangeEnd[0] * s0; d0 <= e0; d0 += s0) {
        for (int d1 = d0 + rangeBegin[1] * s1, e1 = d0 + rangeEnd[1] * s1; d1 <= e1; d1 += s1) {
            for (int d2 = d1 + rangeBegin[2] * s2, e2 = d1 + rangeEnd[2] * s2; d2 <= e2; d2 += s2) {
                for (int d3 = d2 + rangeBegin[3] * s3, e3 = d2 + rangeEnd[3] * s3; d3 <= e3; d3 += s3) {
                    HandleType::Process(d3, args...);
                }
            }
        }
    }

}

template<typename HandleType, typename ...TyArgs>
static void CellMatchHandle(const uint64_t offset[DEV_SHAPE_DIM_MAX], const uint64_t shape[DEV_SHAPE_DIM_MAX],
        const DevCellMatchTableDesc &cellMatchTableDesc, TyArgs ... args) {
    uint64_t rangeBegin[DEV_SHAPE_DIM_MAX];
    uint64_t rangeEnd[DEV_SHAPE_DIM_MAX];
    for (int i = 0; i < cellMatchTableDesc.GetDimensionSize(); ++i) {
        auto cellMatchShapeDim = cellMatchTableDesc.GetCellShape(i);
        if(cellMatchShapeDim != 0) {
            rangeBegin[i] = offset[i] / cellMatchShapeDim;
            rangeEnd[i] = (offset[i] + shape[i] - 1) / cellMatchShapeDim;
        } else {
            DEV_ERROR("CellMatchGetIndexRange: cellMatchShapeDim is zero for dimension %d", i);
            DEV_ASSERT(0);
        }
    }
    switch (cellMatchTableDesc.cellShape.dimSize) {
    case 1:
        {
            int s0 = 1;
            for (int d0 =  0 + rangeBegin[0] * s0, e0 =  0 + rangeEnd[0] * s0; d0 <= e0; d0 += s0) {
                HandleType::Process(d0, args...);
            }
        }
        break;
    case DEV_SHAPE_DIM_NUM_2:
        {
            int s0 = cellMatchTableDesc.GetStride(1), s1 = 1;
            for (int d0 =  0 + rangeBegin[0] * s0, e0 =  0 + rangeEnd[0] * s0; d0 <= e0; d0 += s0)
            for (int d1 = d0 + rangeBegin[1] * s1, e1 = d0 + rangeEnd[1] * s1; d1 <= e1; d1 += s1) {
                HandleType::Process(d1, args...);
            }
        }
        break;
    case DEV_SHAPE_DIM_NUM_3:
        {
            int s0 = cellMatchTableDesc.GetStride(1), s1 = cellMatchTableDesc.GetStride(2), s2 = 1;
            for (int d0 =  0 + rangeBegin[0] * s0, e0 =  0 + rangeEnd[0] * s0; d0 <= e0; d0 += s0)
            for (int d1 = d0 + rangeBegin[1] * s1, e1 = d0 + rangeEnd[1] * s1; d1 <= e1; d1 += s1)
            for (int d2 = d1 + rangeBegin[2] * s2, e2 = d1 + rangeEnd[2] * s2; d2 <= e2; d2 += s2) {
                HandleType::Process(d2, args...);
            }
        }
        break;
    case DEV_SHAPE_DIM_NUM_4:
        {
            CellMatch4Dimension<HandleType>(cellMatchTableDesc, rangeBegin, rangeEnd, args...);
        }
        break;
    case DEV_SHAPE_DIM_NUM_5:
        {
            CellMatch5Dimension<HandleType>(cellMatchTableDesc, rangeBegin, rangeEnd, args...);
        }
        break;
    default:
        DEV_ERROR("[Stitch] Too many dimension: %d\n", (int)cellMatchTableDesc.GetDimensionSize());
        break;
    }
}

template<typename ... TyArgs>
static void CellMatchFill(const uint64_t offset[DEV_SHAPE_DIM_MAX], const uint64_t shape[DEV_SHAPE_DIM_MAX],
        uint32_t operationIdx, const DevCellMatchTableDesc &cellMatchTableDesc, TyArgs... args) {
    if constexpr (sizeof...(args) == 1) {
        auto argsTuple = std::make_tuple(args...);
        uint32_t *cellMatchTableData = std::get<0>(argsTuple);
        struct HandleFill {
            static inline void Process(int index, uint32_t *cellMatchTableData, uint32_t operationIdx) {
                cellMatchTableData[index] = operationIdx;
                DEV_VERBOSE_DEBUG("cell match fill, operation %u , cellindex[%d] = operationindex(%u)",
                        operationIdx, index, operationIdx);
            }
        };
        CellMatchHandle<HandleFill>(offset, shape, cellMatchTableDesc, cellMatchTableData, operationIdx);
    }
    if constexpr (sizeof...(args) == 3) {
        auto argsTuple = std::make_tuple(args...);
        uint64_t *cellMatchTableData = std::get<0>(argsTuple);
        uint32_t devTaskId = std::get<1>(argsTuple);
        uint32_t funcIdx = std::get<2>(argsTuple);
        struct HandleFill {
            static inline void Process(int index, uint64_t *cellMatchTableData, uint32_t devTaskId, uint32_t funcIdx, uint32_t operationIdx) {
                cellMatchTableData[index] = (static_cast<uint64_t>(devTaskId) << TASKID_SHIFT32) | MakeTaskID(funcIdx, operationIdx);
                DEV_VERBOSE_DEBUG("cell match fill, devtaskid:%u funcIdx %u operation %u , cellindex[%d] = taskid(%lx)",
                        devTaskId, funcIdx, operationIdx, index, cellMatchTableData[index]);
            }
        };
        CellMatchHandle<HandleFill>(offset, shape, cellMatchTableDesc, cellMatchTableData, devTaskId, funcIdx, operationIdx);
    }
}

template<bool skipExpression>
static bool GetTensorOffsetAndShape(const DevAscendFunction *devFunc, uint64_t offset[DEV_SHAPE_DIM_MAX],
        uint64_t shape[DEV_SHAPE_DIM_MAX], const uint64_t *runtimeExpressionList, int dims, int operationIndex, int operandIndex,
        bool isIOperand = true) {
    auto [offsetSymList, shapeSymList] = devFunc->GetTensorOffsetShapeSymList(operationIndex, operandIndex, isIOperand);

    bool paramConcrete = true;
    for (int i = 0; i < dims; i++) {
        auto value = offsetSymList[i].Value();
        if (offsetSymList[i].IsExpression()) {
            if (skipExpression) {
                paramConcrete = false;
            } else {
                offset[i] = runtimeExpressionList[value];
            }
        } else {
            offset[i] = value;
        }
    }
    for (int i = 0; i < dims; i++) {
        auto value = shapeSymList[i].Value();
        if (shapeSymList[i].IsExpression()) {
            if (skipExpression) {
                paramConcrete = false;
            } else {
                shape[i] = runtimeExpressionList[value];
            }
        } else {
            shape[i] = value;
        }
    }
    return paramConcrete;
}

template<bool skipExpression>
static bool GetTensorRawShape(DevAscendFunction *devFunc, uint64_t rawShape[DEV_SHAPE_DIM_MAX],
        const uint64_t *runtimeExpressionList, int dims, int operationIndex, int operandIndex, bool isIOperand = true) {
    auto &operandInfo = devFunc->GetOperationOperandInfo(operationIndex, operandIndex, isIOperand);
    const SymInt *rawShapeSymList = &(devFunc->GetOperationAttr(operationIndex, operandInfo.staticRawShapeAttrBeginIndex));
    bool paramConcrete = true;
    for (int i = 0; i < dims; i++) {
        auto value = rawShapeSymList[i].Value();
        if (rawShapeSymList[i].IsExpression()) {
            if (skipExpression) {
                paramConcrete = false;
            } else {
                rawShape[i] = runtimeExpressionList[value];
            }
        } else {
            rawShape[i] = value;
        }
    }
    return paramConcrete;
}

template<bool skipExpression, typename ... TyArgs>
static bool CellMatchFillIncastOutcast(DevAscendFunction *devFunc, DevAscendFunctionCallOperandUse *operandUseList,
        size_t useSize, const uint64_t *runtimeExpressionList, bool isIOperand,
        const DevCellMatchTableDesc &cellMatchTableDesc, TyArgs... args) {
    bool allConcrete = true;
    auto validateAndRefreshOffsetShape =
    [&devFunc, &runtimeExpressionList, &cellMatchTableDesc, &isIOperand](
        const uint64_t offset[DEV_SHAPE_DIM_MAX],
        uint64_t shape[DEV_SHAPE_DIM_MAX],
        int operationIndex, int operandIndex) {
        uint64_t rawShape[DEV_SHAPE_DIM_MAX];
        bool paramConcrete = GetTensorRawShape<skipExpression>(devFunc, rawShape, runtimeExpressionList,
            cellMatchTableDesc.GetDimensionSize(), operationIndex, operandIndex, isIOperand);
        if (paramConcrete) {
            for (int j = 0; j < cellMatchTableDesc.GetDimensionSize(); j++) {
                DEV_VERBOSE_DEBUG("cell match fill, operation[%d] -> dimension[%d] = (offset:%lu ,shape:%lu, rawshape:%lu, cellshape:%d)",
                        operationIndex, j, offset[j], shape[j], rawShape[j], cellMatchTableDesc.cellShape.dim[j]);
                if (offset[j] >= rawShape[j]) {
                    DEV_VERBOSE_DEBUG("cell match fill failed, exceed invalid cell");
                    return false;
                } else if (offset[j] + shape[j] > rawShape[j]){
                    shape[j] = rawShape[j] - offset[j];
                }
            }
        }
        return true;
    };

    for (size_t i = 0; i < useSize; i++) {
        auto &use = operandUseList[i];
        uint64_t offset[DEV_SHAPE_DIM_MAX];
        uint64_t shape[DEV_SHAPE_DIM_MAX];
        bool paramConcrete = GetTensorOffsetAndShape<skipExpression>(devFunc, offset, shape, runtimeExpressionList,
            cellMatchTableDesc.GetDimensionSize(), use.operationIdx, use.operandIdx, isIOperand);
        if (paramConcrete) {
            if (!validateAndRefreshOffsetShape(offset, shape, use.operationIdx, use.operandIdx)) {
                continue; // dassemble offset of outoperand maybe exceed the rawshape dimension
            }
            CellMatchFill(offset, shape, use.operationIdx, cellMatchTableDesc, args...);
        }
        allConcrete &= paramConcrete;
    }
    return allConcrete;
}
}