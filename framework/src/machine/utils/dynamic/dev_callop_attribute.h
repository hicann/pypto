/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file dev_callop_attribute.h
 * \brief
 */

#pragma once

#include "machine/utils/dynamic/dev_encode_function.h"
#include "machine/utils/dynamic/dev_encode_types.h"
#include "tilefwk/aicpu_common.h"

namespace npu::tile_fwk::dynamic {
inline bool IsCellMatchDescFillReady(const DevCellMatchTableDesc& cellMatchTableDesc)
{
    int dim = cellMatchTableDesc.GetDimensionSize();
    if (dim <= 0) {
        return false;
    }
    for (int d = 0; d < dim; ++d) {
        if (cellMatchTableDesc.GetCellShape(d) <= 0 || cellMatchTableDesc.GetStrideShape(d) <= 0) {
            return false;
        }
    }
    return true;
}

inline void DumpCellMatchAccessRange(
    int funcKey, int operationIndex, int operandIndex, bool isIOperand,
    const uint64_t offset[DEV_SHAPE_DIM_MAX], const uint64_t validShape[DEV_SHAPE_DIM_MAX],
    const uint64_t rawShape[DEV_SHAPE_DIM_MAX], const DevCellMatchTableDesc& cellMatchTableDesc)
{
    uint64_t dumpOffset[DEV_SHAPE_DIM_MAX] = {0};
    uint64_t dumpValidShape[DEV_SHAPE_DIM_MAX] = {0};
    uint64_t dumpRawShape[DEV_SHAPE_DIM_MAX] = {0};
    uint64_t cellShape[DEV_SHAPE_DIM_MAX] = {0};
    int dims = cellMatchTableDesc.GetDimensionSize();
    int dumpDims = dims < DEV_SHAPE_DIM_MAX ? dims : DEV_SHAPE_DIM_MAX;

    for (int i = 0; i < dumpDims; i++) {
        dumpOffset[i] = offset[i];
        dumpValidShape[i] = validShape[i];
        dumpRawShape[i] = rawShape[i];
        cellShape[i] = static_cast<uint64_t>(cellMatchTableDesc.GetCellShape(i));
    }

    DEV_WARN(
        "[StitchCellRange] funcKey=%d op=%d operand=%d isIOperand=%d dim=%d "
        "offset=[%lu,%lu,%lu,%lu,%lu] validShape=[%lu,%lu,%lu,%lu,%lu] "
        "rawShape=[%lu,%lu,%lu,%lu,%lu] cellShape=[%lu,%lu,%lu,%lu,%lu]",
        funcKey, operationIndex, operandIndex, isIOperand ? 1 : 0, dims,
        dumpOffset[0], dumpOffset[1], dumpOffset[2], dumpOffset[3], dumpOffset[4],
        dumpValidShape[0], dumpValidShape[1], dumpValidShape[2], dumpValidShape[3], dumpValidShape[4],
        dumpRawShape[0], dumpRawShape[1], dumpRawShape[2], dumpRawShape[3], dumpRawShape[4],
        cellShape[0], cellShape[1], cellShape[2], cellShape[3], cellShape[4]);
}

inline bool CheckOffsetAndValidShapeInRawShape(
    uint64_t offset[DEV_SHAPE_DIM_MAX], uint64_t validShape[DEV_SHAPE_DIM_MAX],
    const uint64_t rawShape[DEV_SHAPE_DIM_MAX], int dims)
{
    bool clamped = false;
    for (int i = 0; i < dims; i++) {
        if (validShape[i] == 0) {
            return clamped;
        }
    }
    for (int i = 0; i < dims; i++) {
        if (offset[i] > rawShape[i]) {
            DEV_WARN(
                "#ctrl.stitch.bound: action=offset_out_of_range, offset > rawShape");
            offset[i] = rawShape[i];
            validShape[i] = 0;
            clamped = true;
        } else if (validShape[i] > rawShape[i] - offset[i]) {
            DEV_WARN(
                "#ctrl.stitch.bound: action=validShape_out_of_range, offset + validShape > rawShape");
            validShape[i] = rawShape[i] - offset[i];
            clamped = true;
        }
    }
    return clamped;
}

template <bool skipExpression>
static bool GetTensorOffsetAndShape(
    const DevAscendFunction* devFunc, uint64_t offset[DEV_SHAPE_DIM_MAX], uint64_t shape[DEV_SHAPE_DIM_MAX],
    const uint64_t* runtimeExpressionList, int dims, int operationIndex, int operandIndex, bool isIOperand = true)
{
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

template <bool skipExpression>
static bool GetTensorRawShape(
    const DevAscendFunction* devFunc, uint64_t rawShape[DEV_SHAPE_DIM_MAX],
    const uint64_t* runtimeExpressionList, int dims, int operationIndex, int operandIndex, bool isIOperand = true)
{
    auto& operandInfo = devFunc->GetOperationOperandInfo(operationIndex, operandIndex, isIOperand);
    const SymInt* rawShapeSymList =
        &(devFunc->GetOperationAttr(operationIndex, operandInfo.staticRawShapeAttrBeginIndex));
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

template <bool skipExpression>
static bool GetTensorOffsetAndValidShape(
    const DevAscendFunction* devFunc, uint64_t offset[DEV_SHAPE_DIM_MAX], uint64_t validShape[DEV_SHAPE_DIM_MAX],
    const uint64_t* runtimeExpressionList, const DevCellMatchTableDesc& cellMatchTableDesc, int dims,
    int operationIndex, int operandIndex, bool isIOperand = true)
{
    auto& operandInfo = devFunc->GetOperationOperandInfo(operationIndex, operandIndex, isIOperand);
    const SymInt* offsetSymList =
        &(devFunc->GetOperationAttr(operationIndex, operandInfo.staticOffsetAttrBeginIndex));
    const SymInt* validShapeSymList =
        &(devFunc->GetOperationAttr(operationIndex, operandInfo.staticValidShapeAttrBeginIndex));
    const SymInt* rawShapeSymList =
        &(devFunc->GetOperationAttr(operationIndex, operandInfo.staticRawShapeAttrBeginIndex));

    uint64_t rawShape[DEV_SHAPE_DIM_MAX] = {0};
    bool paramConcrete = true;
    for (int i = 0; i < dims; i++) {
        if (offsetSymList[i].IsExpression()) {
            if (skipExpression) {
                paramConcrete = false;
            } else {
                offset[i] = runtimeExpressionList[offsetSymList[i].Value()];
            }
        } else {
            offset[i] = offsetSymList[i].Value();
        }

        if (validShapeSymList[i].IsExpression()) {
            if (skipExpression) {
                paramConcrete = false;
            } else {
                validShape[i] = runtimeExpressionList[validShapeSymList[i].Value()];
            }
        } else {
            validShape[i] = validShapeSymList[i].Value();
        }

        if (rawShapeSymList[i].IsExpression()) {
            if (skipExpression) {
                paramConcrete = false;
            } else {
                rawShape[i] = runtimeExpressionList[rawShapeSymList[i].Value()];
            }
        } else {
            rawShape[i] = rawShapeSymList[i].Value();
        }
    }

    if (!paramConcrete) {
        return paramConcrete;
    }

    bool clamped = CheckOffsetAndValidShapeInRawShape(offset, validShape, rawShape, dims);
    if (clamped) {
        DumpCellMatchAccessRange(
            devFunc->GetFuncKey(), operationIndex, operandIndex, isIOperand, offset, validShape, rawShape,
            cellMatchTableDesc);
    }
    return paramConcrete;
}
} // namespace npu::tile_fwk::dynamic