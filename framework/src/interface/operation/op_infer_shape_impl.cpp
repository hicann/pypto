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
 * \file op_infer_shape_impl.cpp
 * \brief
 */

#include "op_infer_shape_impl.h"
#include "interface/operation/attr_holder.h"
#include "interface/operation/operation.h"
#include "interface/tensor/symbolic_scalar.h"
#include "interface/utils/common.h"

namespace npu::tile_fwk {
const std::string COPY_OUT_FORCE_INFER_SHAPE = "copy_out_force_infer_shape";

void ElewiseInferFunc(Operation* op,
                      std::vector<std::vector<SymbolicScalar>>& outValidShapes) {
    auto inputNum = op->GetIOperands().size();
    auto shapeDimNum = op->GetIOperands()[0]->GetDynValidShape().size();
    // 将每个输入的同一维shape值填充到一个vector中，便于后续对每一维进行筛选
    std::vector<std::vector<SymbolicScalar>> dimValidShape(shapeDimNum, std::vector<SymbolicScalar>(inputNum, SymbolicScalar()));
    std::vector<std::vector<int64_t>> dimShape(shapeDimNum, std::vector<int64_t>(inputNum, 0));
    for (size_t i = 0; i < op->GetIOperands().size(); ++i) {
        auto iOperand = op->GetInputOperand(i);
        auto validShape = op->GetIOperands()[i]->GetDynValidShape();
        for (size_t dimIdx = 0; dimIdx < validShape.size(); ++dimIdx) {
            dimValidShape[dimIdx][i] = validShape[dimIdx];
        }
        auto shape = op->GetIOperands()[i]->GetShape();
        for (size_t dimIdx = 0; dimIdx < shape.size() && dimIdx < shapeDimNum; ++dimIdx) {
            if (dimIdx == shape.size() - 1 && iOperand->GetProducers().size() == 1 &&
                (*iOperand->GetProducers().begin())->GetOpcode() == Opcode::OP_BRCB) {
                dimShape[dimIdx][i] = 1;
            } else {
                dimShape[dimIdx][i] = shape[dimIdx];
            }
        }
    }
    std::vector<SymbolicScalar> inputValidShape;
    for (size_t i = 0; i < shapeDimNum; ++i) {
        size_t oneDimNum = 0;
        size_t noOneIndex = 0;
        for (size_t j = 0; j < dimShape[i].size(); ++j) {
            if (dimShape[i][j] == 1) {
                oneDimNum++;
            } else {
                noOneIndex = j;
            }
        }
        if (oneDimNum > 0 && oneDimNum < dimShape[i].size()) {
            inputValidShape.push_back(dimValidShape[i][noOneIndex]);
            continue;
        }

        auto flag = false;
        auto minDim = SymbolicScalar();
        for (auto dim : dimValidShape[i]) {
            if (!(dim.IsImmediate())) {
                inputValidShape.push_back(dim);
                flag = true;
                break;
            } else {
                minDim = minDim.ConcreteValid() ? std::min(minDim.Concrete(), dim.Concrete()) : dim.Concrete();
            }
        }
        // 全部都是Immediate值，取用最小的
        if (!flag) {
            inputValidShape.push_back(minDim);
        }
    }

    int64_t mode = 0;
    if (op->GetAttr(OP_ATTR_PREFIX + "cmp_mode", mode) && mode == 1) {
        inputValidShape[inputValidShape.size() - 1] = inputValidShape[inputValidShape.size() - 1] / 8; // 8 bit to 1 byte
    }

    int64_t whereBitMode = 0;
    if (op->GetAttr(OP_ATTR_PREFIX + "whereBitMode", whereBitMode) && whereBitMode == 1) {
        inputValidShape[inputValidShape.size() - 1] = inputValidShape[inputValidShape.size() - 1] * 8;
    }

    for (auto output : op->GetOOperands()) {
        outValidShapes.push_back(inputValidShape);
    }
}

REGISTER_INFER_SHAPE_FUNC(OP_ADD, Opcode::OP_ADD, ElewiseInferFunc);
REGISTER_INFER_SHAPE_FUNC(OP_ADDS, Opcode::OP_ADDS, ElewiseInferFunc);
REGISTER_INFER_SHAPE_FUNC(OP_MULS, Opcode::OP_MULS, ElewiseInferFunc);
REGISTER_INFER_SHAPE_FUNC(OP_S_DIVS, Opcode::OP_S_DIVS, ElewiseInferFunc);
REGISTER_INFER_SHAPE_FUNC(OP_SUB, Opcode::OP_SUB, ElewiseInferFunc);
REGISTER_INFER_SHAPE_FUNC(OP_POW, Opcode::OP_POW, ElewiseInferFunc);
REGISTER_INFER_SHAPE_FUNC(OP_EXP, Opcode::OP_EXP, ElewiseInferFunc);
REGISTER_INFER_SHAPE_FUNC(OP_NEG, Opcode::OP_NEG, ElewiseInferFunc);
REGISTER_INFER_SHAPE_FUNC(OP_RSQRT, Opcode::OP_RSQRT, ElewiseInferFunc);
REGISTER_INFER_SHAPE_FUNC(OP_SQRT, Opcode::OP_SQRT, ElewiseInferFunc);
REGISTER_INFER_SHAPE_FUNC(OP_CEIL, Opcode::OP_CEIL, ElewiseInferFunc);
REGISTER_INFER_SHAPE_FUNC(OP_FLOOR, Opcode::OP_FLOOR, ElewiseInferFunc);
REGISTER_INFER_SHAPE_FUNC(OP_TRUNC, Opcode::OP_TRUNC, ElewiseInferFunc);
REGISTER_INFER_SHAPE_FUNC(OP_BITWISENOT, Opcode::OP_BITWISENOT, ElewiseInferFunc);
REGISTER_INFER_SHAPE_FUNC(OP_ABS, Opcode::OP_ABS, ElewiseInferFunc);
REGISTER_INFER_SHAPE_FUNC(OP_LN, Opcode::OP_LN, ElewiseInferFunc);
REGISTER_INFER_SHAPE_FUNC(OP_HUB, Opcode::OP_HUB, ElewiseInferFunc);
REGISTER_INFER_SHAPE_FUNC(OP_MAXIMUM, Opcode::OP_MAXIMUM, ElewiseInferFunc);
REGISTER_INFER_SHAPE_FUNC(OP_MINIMUM, Opcode::OP_MINIMUM, ElewiseInferFunc);
REGISTER_INFER_SHAPE_FUNC(OP_CAST, Opcode::OP_CAST, ElewiseInferFunc);
REGISTER_INFER_SHAPE_FUNC(OP_MUL, Opcode::OP_MUL, ElewiseInferFunc);
REGISTER_INFER_SHAPE_FUNC(OP_DIV, Opcode::OP_DIV, ElewiseInferFunc);
REGISTER_INFER_SHAPE_FUNC(OP_DIVS, Opcode::OP_DIVS, ElewiseInferFunc);
REGISTER_INFER_SHAPE_FUNC(OP_RECIPROCAL, Opcode::OP_RECIPROCAL, ElewiseInferFunc);
REGISTER_INFER_SHAPE_FUNC(OP_SUBS, Opcode::OP_SUBS, ElewiseInferFunc);
REGISTER_INFER_SHAPE_FUNC(OP_MAXS, Opcode::OP_MAXS, ElewiseInferFunc);
REGISTER_INFER_SHAPE_FUNC(OP_MINS, Opcode::OP_MINS, ElewiseInferFunc);
REGISTER_INFER_SHAPE_FUNC(OP_BITWISEANDS, Opcode::OP_BITWISEANDS, ElewiseInferFunc);
REGISTER_INFER_SHAPE_FUNC(OP_BITWISEORS, Opcode::OP_BITWISEORS, ElewiseInferFunc);
REGISTER_INFER_SHAPE_FUNC(OP_BITWISEXORS, Opcode::OP_BITWISEXORS, ElewiseInferFunc);
REGISTER_INFER_SHAPE_FUNC(OP_S_ADDS, Opcode::OP_S_ADDS, ElewiseInferFunc);
REGISTER_INFER_SHAPE_FUNC(OP_S_SUBS, Opcode::OP_S_SUBS, ElewiseInferFunc);
REGISTER_INFER_SHAPE_FUNC(OP_S_MULS, Opcode::OP_S_MULS, ElewiseInferFunc);
REGISTER_INFER_SHAPE_FUNC(OP_S_MAXS, Opcode::OP_S_MAXS, ElewiseInferFunc);
REGISTER_INFER_SHAPE_FUNC(OP_S_MINS, Opcode::OP_S_MINS, ElewiseInferFunc);
REGISTER_INFER_SHAPE_FUNC(OP_S_ADD, Opcode::OP_S_ADD, ElewiseInferFunc);
REGISTER_INFER_SHAPE_FUNC(OP_S_SUB, Opcode::OP_S_SUB, ElewiseInferFunc);
REGISTER_INFER_SHAPE_FUNC(OP_S_MUL, Opcode::OP_S_MUL, ElewiseInferFunc);
REGISTER_INFER_SHAPE_FUNC(OP_S_DIV, Opcode::OP_S_DIV, ElewiseInferFunc);
REGISTER_INFER_SHAPE_FUNC(OP_S_MAX, Opcode::OP_S_MAX, ElewiseInferFunc);
REGISTER_INFER_SHAPE_FUNC(OP_S_MIN, Opcode::OP_S_MIN, ElewiseInferFunc);
REGISTER_INFER_SHAPE_FUNC(OP_CUM_SUM, Opcode::OP_CUM_SUM, ElewiseInferFunc);
REGISTER_INFER_SHAPE_FUNC(OP_REGISTER_COPY, Opcode::OP_REGISTER_COPY, ElewiseInferFunc);
REGISTER_INFER_SHAPE_FUNC(OP_SCATTER_ELEMENT, Opcode::OP_SCATTER_ELEMENT, ElewiseInferFunc);
REGISTER_INFER_SHAPE_FUNC(OP_SCATTER, Opcode::OP_SCATTER, ElewiseInferFunc);
REGISTER_INFER_SHAPE_FUNC(OP_CMP, Opcode::OP_CMP, ElewiseInferFunc);
REGISTER_INFER_SHAPE_FUNC(OP_CMPS, Opcode::OP_CMPS, ElewiseInferFunc);
REGISTER_INFER_SHAPE_FUNC(OP_MOD, Opcode::OP_MOD, ElewiseInferFunc);
REGISTER_INFER_SHAPE_FUNC(OP_MODS, Opcode::OP_MODS, ElewiseInferFunc);
REGISTER_INFER_SHAPE_FUNC(OP_BITWISERIGHTSHIFT, Opcode::OP_BITWISERIGHTSHIFT, ElewiseInferFunc);
REGISTER_INFER_SHAPE_FUNC(OP_BITWISELEFTSHIFT, Opcode::OP_BITWISELEFTSHIFT, ElewiseInferFunc);
REGISTER_INFER_SHAPE_FUNC(OP_BITWISERIGHTSHIFTS, Opcode::OP_BITWISERIGHTSHIFTS, ElewiseInferFunc);
REGISTER_INFER_SHAPE_FUNC(OP_BITWISELEFTSHIFTS, Opcode::OP_BITWISELEFTSHIFTS, ElewiseInferFunc);
REGISTER_INFER_SHAPE_FUNC(OP_SBITWISERIGHTSHIFT, Opcode::OP_SBITWISERIGHTSHIFT, ElewiseInferFunc);
REGISTER_INFER_SHAPE_FUNC(OP_SBITWISELEFTSHIFT, Opcode::OP_SBITWISELEFTSHIFT, ElewiseInferFunc);
REGISTER_INFER_SHAPE_FUNC(OP_BITWISEAND, Opcode::OP_BITWISEAND, ElewiseInferFunc);
REGISTER_INFER_SHAPE_FUNC(OP_BITWISEOR, Opcode::OP_BITWISEOR, ElewiseInferFunc);
REGISTER_INFER_SHAPE_FUNC(OP_BITWISEXOR, Opcode::OP_BITWISEXOR, ElewiseInferFunc);

void IndexOutCastInferFunc(Operation* op,
                      std::vector<std::vector<SymbolicScalar>>& outValidShapes) {
    std::vector<SymbolicScalar> outValidShape;
    /* 这里取IOperands索引是依据AddOperation中ioprand中的输入的顺序，这里使用的是dst参数的ioprand，即第2个索引 */
    auto inValidShape = op->GetIOperands()[2]->GetDynValidShape();

    for (size_t i = 0; i < inValidShape.size(); ++i) {
        outValidShape.push_back(inValidShape[i]);
    }

    for (auto output : op->GetOOperands()) {
        outValidShapes.push_back(outValidShape);
    }
}
REGISTER_INFER_SHAPE_FUNC(OP_INDEX_OUTCAST, Opcode::OP_INDEX_OUTCAST, IndexOutCastInferFunc);

void GatherElementInferFunc(Operation* op,
                      std::vector<std::vector<SymbolicScalar>>& outValidShapes) {
    std::vector<SymbolicScalar> outValidShape;
    /* 这里取IOperands索引是依据AddOperation中ioprand中的输入的顺序，这里使用的是src参数的ioprand，即第1个索引 */
    auto inValidShape = op->GetIOperands()[1]->GetDynValidShape();

    for (size_t i = 0; i < inValidShape.size(); ++i) {
        outValidShape.push_back(inValidShape[i]);
    }

    for (auto output : op->GetOOperands()) {
        outValidShapes.push_back(outValidShape);
    }
}
REGISTER_INFER_SHAPE_FUNC(OP_GATHER_ELEMENT, Opcode::OP_GATHER_ELEMENT, GatherElementInferFunc);

void IndexAddInferFunc(Operation* op,
                      std::vector<std::vector<SymbolicScalar>>& outValidShapes) {
    std::vector<SymbolicScalar> outValidShape;
    auto inValidShape = op->GetIOperands()[0]->GetDynValidShape();

    for (size_t i = 0; i < inValidShape.size(); ++i) {
        outValidShape.push_back(inValidShape[i]);
    }

    for (auto output : op->GetOOperands()) {
        outValidShapes.push_back(outValidShape);
    }
}
REGISTER_INFER_SHAPE_FUNC(OP_INDEX_ADD, Opcode::OP_INDEX_ADD, IndexAddInferFunc);

void LogicalNotInferFunc(Operation* op,
                        std::vector<std::vector<SymbolicScalar>>& outValidShapes) {
    ElewiseInferFunc(op, outValidShapes);
    outValidShapes.erase(outValidShapes.begin() + 1, outValidShapes.end());
    auto data_type = op->GetIOperands()[0]->Datatype();

    DataType select_dtype;
    if (data_type == DT_FP32 || data_type == DT_BF16) {
        select_dtype = DT_FP32;
    } else {
        select_dtype = DT_FP16;
    }
    constexpr int64_t COUNT_SIZE = 2048;
    constexpr int64_t vcmp_bit_size = COUNT_SIZE / 8;
    constexpr size_t ALIGN_SIZE = 32;

    int64_t total_size = COUNT_SIZE * 2 + COUNT_SIZE * BytesOf(select_dtype) * 2 + vcmp_bit_size + 8;
    total_size = (total_size + ALIGN_SIZE - 1) / ALIGN_SIZE * ALIGN_SIZE;
    int64_t shape = total_size / BytesOf(select_dtype);
    outValidShapes.push_back({shape});
}
REGISTER_INFER_SHAPE_FUNC(OP_LOGICALNOT, Opcode::OP_LOGICALNOT, LogicalNotInferFunc);

void LogicalAndInferFunc(Operation* op,
                        std::vector<std::vector<SymbolicScalar>>& outValidShapes) {
    ElewiseInferFunc(op, outValidShapes);
    outValidShapes.erase(outValidShapes.begin() + 1, outValidShapes.end());
    const int64_t COUNT_SIZE = 64;
    outValidShapes.push_back({COUNT_SIZE * 5 + COUNT_SIZE / 8 + 1});
}
REGISTER_INFER_SHAPE_FUNC(OP_LOGICALAND, Opcode::OP_LOGICALAND, LogicalAndInferFunc);

void ViewTypeInferFunc(Operation* op, std::vector<std::vector<SymbolicScalar>>& outValidShapes) {
    auto inputOperand = op->GetIOperands()[0];
    auto outputOperand = op->GetOOperands()[0];

    auto validShape = inputOperand->GetDynValidShape();
    auto changedDim = validShape[validShape.size() - 1] * BytesOf(inputOperand->Datatype()) / BytesOf(outputOperand->Datatype());
    validShape[validShape.size() - 1] = changedDim;

    outValidShapes.push_back(validShape);
}
REGISTER_INFER_SHAPE_FUNC(OP_VIEW_TYPE, Opcode::OP_VIEW_TYPE, ViewTypeInferFunc);

void IndexPutInferFunc(Operation* op,
                        std::vector<std::vector<SymbolicScalar>>& outValidShapes) {
    outValidShapes.push_back(op->GetIOperands()[0]->GetDynValidShape());
}
REGISTER_INFER_SHAPE_FUNC(OP_INDEX_PUT, Opcode::OP_INDEX_PUT, IndexPutInferFunc);

void PairReduceInferFunc(Operation* op,
                        std::vector<std::vector<SymbolicScalar>>& outValidShapes) {
    auto dimSize = op->GetIOperands()[0]->GetDynValidShape().size();
    std::vector<SymbolicScalar> outValidShape;
    for (size_t i = 0; i < dimSize; i++) {
        outValidShape.push_back(std::max(op->GetIOperands()[0]->GetDynValidShape()[i],
            op->GetIOperands()[1]->GetDynValidShape()[i]));
    }
    for (auto output : op->GetOOperands()) {
        outValidShapes.push_back(outValidShape);
    }
}
REGISTER_INFER_SHAPE_FUNC(OP_PAIRMAX, Opcode::OP_PAIRMAX, PairReduceInferFunc);
REGISTER_INFER_SHAPE_FUNC(OP_PAIRMIN, Opcode::OP_PAIRMIN, PairReduceInferFunc);
REGISTER_INFER_SHAPE_FUNC(OP_PAIRSUM, Opcode::OP_PAIRSUM, PairReduceInferFunc);

// elewise brc infer shape func
void ElewiseBrcInferFunc(Operation* op,
                        std::vector<std::vector<SymbolicScalar>>& outValidShapes) {
    std::vector<SymbolicScalar> outValidShape;

    // elewisebrc dim is immediate and dim is 1, another dim is dst shape dim
    for (size_t i = 0; i < op->GetIOperands()[0]->GetDynValidShape().size(); ++i) {
        auto leftIShapeDim = op->GetIOperands()[0]->GetDynValidShape()[i];
        if (leftIShapeDim.IsImmediate() && leftIShapeDim.Concrete() == 1) {
            outValidShape.push_back(op->GetIOperands()[1]->GetDynValidShape()[i]);
        } else {
            outValidShape.push_back(leftIShapeDim);
        }
    }
    for (auto output : op->GetOOperands()) {
        outValidShapes.push_back(outValidShape);
    }
}
REGISTER_INFER_SHAPE_FUNC(OP_ADD_BRC, Opcode::OP_ADD_BRC, ElewiseBrcInferFunc);
REGISTER_INFER_SHAPE_FUNC(OP_SUB_BRC, Opcode::OP_SUB_BRC, ElewiseBrcInferFunc);
REGISTER_INFER_SHAPE_FUNC(OP_MUL_BRC, Opcode::OP_MUL_BRC, ElewiseBrcInferFunc);
REGISTER_INFER_SHAPE_FUNC(OP_DIV_BRC, Opcode::OP_DIV_BRC, ElewiseBrcInferFunc);
REGISTER_INFER_SHAPE_FUNC(OP_MAX_BRC, Opcode::OP_MAX_BRC, ElewiseBrcInferFunc);

// broadcast infer shape func
void BroadcastInferFunc(Operation* op,
                        std::vector<std::vector<SymbolicScalar>>& outValidShapes) {
    std::vector<SymbolicScalar> outValidShape;
    if (op->GetAttr(OP_ATTR_PREFIX + "validShape", outValidShape)) {
        for (auto output : op->GetOOperands()) {
            outValidShapes.push_back(outValidShape);
        }
        return;
    }
    auto outTensor = op->GetOOperands()[0]; // one in, one out
    // broadcast 1对应的维度采用tileshap
    for (size_t i = 0; i < op->GetIOperands()[0]->GetDynValidShape().size(); ++i) {
        if (op->GetIOperands()[0]->oriShape[i] != 1) {
            outValidShape.push_back(op->GetIOperands()[0]->GetDynValidShape()[i]);
        } else {
            outValidShape.push_back(SymbolicScalar(op->GetOOperands()[0]->GetShape()[i]));
        }
    }
    for (auto output : op->GetOOperands()) {
        outValidShapes.push_back(outValidShape);
    }
}
REGISTER_INFER_SHAPE_FUNC(OP_EXPAND, Opcode::OP_EXPAND, BroadcastInferFunc);

// one hot infer shape func
void OneHotInferFunc(Operation* op, std::vector<std::vector<SymbolicScalar>>& outValidShapes) {
    std::vector<SymbolicScalar> outValidShape(op->GetIOperands()[0]->GetDynValidShape());
    int lastDim = op->GetIntAttribute(OP_ATTR_PREFIX + "numClasses");
    outValidShape.push_back(SymbolicScalar(lastDim));
    outValidShapes.push_back(outValidShape);
}
REGISTER_INFER_SHAPE_FUNC(OP_ONEHOT, Opcode::OP_ONEHOT, OneHotInferFunc);

// Range infer shape func
void RangeInferFunc(Operation *op, std::vector<std::vector<SymbolicScalar>> &outValidShapes) {
    std::vector<SymbolicScalar> outValidShape;
    Element size = op->GetElementAttribute(OP_ATTR_PREFIX + "SIZE");
    outValidShape.push_back(SymbolicScalar(size.GetSignedData()));
    for (auto output : op->GetOOperands()) {
        outValidShapes.push_back(outValidShape);
    }
}
REGISTER_INFER_SHAPE_FUNC(OP_RANGE, Opcode::OP_RANGE, RangeInferFunc);

void LoadInferFunc(Operation *op, std::vector<std::vector<SymbolicScalar>> &outValidShapes) {
    auto iOperands = op->GetIOperands();
    assert(iOperands.size() == NUM2);
    auto offsetValidShape = iOperands[1]->GetDynValidShape();
    for (auto output : op->GetOOperands()) {
        outValidShapes.push_back(offsetValidShape);
    }
}
REGISTER_INFER_SHAPE_FUNC(OP_LOAD, Opcode::OP_LOAD, LoadInferFunc);

// reduce infer shape func
void ReduceInferFunc(Operation* op,
                        std::vector<std::vector<SymbolicScalar>>& outValidShapes) {
    std::vector<std::vector<SymbolicScalar>> inputValidShapes;
    for (auto inputTensor : op->GetIOperands()) {
        inputValidShapes.push_back(inputTensor->GetDynValidShape());
    }
    if (inputValidShapes.empty()) {
        return;
    }
    auto outValidShape = inputValidShapes[0];
    int axis = op->GetIntAttribute(OP_ATTR_PREFIX + "AXIS");
    outValidShape[axis] = SymbolicScalar(1);
    for (auto output : op->GetOOperands()) {
        outValidShapes.push_back(outValidShape);
    }
}
REGISTER_INFER_SHAPE_FUNC(OP_ROWSUMLINE, Opcode::OP_ROWSUMLINE, ReduceInferFunc);
REGISTER_INFER_SHAPE_FUNC(OP_ROWMAXLINE, Opcode::OP_ROWMAXLINE, ReduceInferFunc);
REGISTER_INFER_SHAPE_FUNC(OP_ROWMINLINE, Opcode::OP_ROWMINLINE, ReduceInferFunc);
REGISTER_INFER_SHAPE_FUNC(OP_ROWMAX_SINGLE, Opcode::OP_ROWMAX_SINGLE, ReduceInferFunc);
REGISTER_INFER_SHAPE_FUNC(OP_ROWMIN_SINGLE, Opcode::OP_ROWMIN_SINGLE, ReduceInferFunc);
REGISTER_INFER_SHAPE_FUNC(OP_ROWSUM_SINGLE, Opcode::OP_ROWSUM_SINGLE, ReduceInferFunc);
REGISTER_INFER_SHAPE_FUNC(OP_ROWMAX_COMBINE_AXIS_SINGLE, Opcode::OP_ROWMAX_COMBINE_AXIS_SINGLE, ReduceInferFunc);
REGISTER_INFER_SHAPE_FUNC(OP_ROWSUM_COMBINE_AXIS_SINGLE, Opcode::OP_ROWSUM_COMBINE_AXIS_SINGLE, ReduceInferFunc);

void WhereInferFunc(Operation* op,
                        std::vector<std::vector<SymbolicScalar>>& outValidShapes) {
    ElewiseInferFunc(op, outValidShapes);
}
REGISTER_INFER_SHAPE_FUNC(OP_WHERE_TT, Opcode::OP_WHERE_TT, WhereInferFunc);
REGISTER_INFER_SHAPE_FUNC(OP_WHERE_TS, Opcode::OP_WHERE_TS, WhereInferFunc);
REGISTER_INFER_SHAPE_FUNC(OP_WHERE_ST, Opcode::OP_WHERE_ST, WhereInferFunc);
REGISTER_INFER_SHAPE_FUNC(OP_WHERE_SS, Opcode::OP_WHERE_SS, WhereInferFunc);

// Gather infer shape func
void InferFunc4Gather(Operation* op, std::vector<std::vector<SymbolicScalar>>& outValidShapes) {
    auto iOperands = op->GetIOperands();
    assert(iOperands.size() >= NUM2);
    int axis = op->GetIntAttribute(OP_ATTR_PREFIX + "axis");
    int src0Rank = iOperands[0]->GetShape().size();
    if (axis < 0) {
        axis = axis + src0Rank;
    }
    assert((axis >= 0 && axis < src0Rank) && "InferFunc4Gather, axis is invalid");

    std::vector<std::vector<SymbolicScalar>> inputValidShapes;
    for (auto inputTensor : iOperands) {
        inputValidShapes.push_back(inputTensor->GetDynValidShape());
    }
    // output shape: input0.shape[:aixs] + input1.shape + input0.shape[axis+1:]
    std::vector<SymbolicScalar> outValidShape = inputValidShapes[0];
    outValidShape.erase(outValidShape.begin() + axis);
    outValidShape.insert(outValidShape.begin() + axis, inputValidShapes[1].begin(), inputValidShapes[1].end());

    for (auto output : op->GetOOperands()) {
        outValidShapes.push_back(outValidShape);
    }
}
REGISTER_INFER_SHAPE_FUNC(OP_GATHER_FROM_UB, Opcode::OP_GATHER_FROM_UB, InferFunc4Gather);
REGISTER_INFER_SHAPE_FUNC(OP_GATHER, Opcode::OP_GATHER, InferFunc4Gather);

void InferFuncGatherInL1(Operation *op, std::vector<std::vector<SymbolicScalar>> &outValidShapes) {
    auto iOperands = op->GetIOperands();
    assert(iOperands.size() == 3);
    auto srcValidShape = iOperands[0]->GetDynValidShape();
    auto offsetValidShape = iOperands[1]->GetDynValidShape();
    auto srcStartColumnOffset = op->GetIntAttribute(OpAttributeKey::startOffset);
    ASSERT(op->GetOOperands().size() == 1);
    for (auto output : op->GetOOperands()) {
        outValidShapes.push_back(
            {offsetValidShape[1], std::min(srcValidShape[1] - srcStartColumnOffset, output->GetShape()[1])});
    }
}
REGISTER_INFER_SHAPE_FUNC(OP_GATHER_IN_L1, Opcode::OP_GATHER_IN_L1, InferFuncGatherInL1);
/**
 * 定制，
 * parma [a,b]
 * indices [1,c]
 * axis=-2
 * result [c,b]
 */
void InferFuncGatherInUB(Operation *op, std::vector<std::vector<SymbolicScalar>> &outValidShapes) {
    auto iOperands = op->GetIOperands();
    assert(iOperands.size() == 3);
    auto srcValidShape = iOperands[0]->GetDynValidShape();
    auto indicesValidShape = iOperands[1]->GetDynValidShape();
    ASSERT(op->GetOOperands().size() == 1);
    for (auto output : op->GetOOperands()) {
        outValidShapes.push_back(
            {indicesValidShape[1], srcValidShape[1]});
    }
}
REGISTER_INFER_SHAPE_FUNC(OP_GATHER_IN_UB, Opcode::OP_GATHER_IN_UB, InferFuncGatherInUB);

// matmul infer shape func
void MatmulInferFunc(Operation* op,
                     std::vector<std::vector<SymbolicScalar>>& outValidShapes) {
    std::vector<SymbolicScalar> outValidShape;
    for (auto inputTensor : op->GetIOperands()) {
        auto inputValidShape = inputTensor->GetDynValidShape();
        if (inputTensor->GetMemoryTypeOriginal() == MemoryType::MEM_L0A) {
            outValidShape.push_back(inputValidShape[0]);
        } else if (inputTensor->GetMemoryTypeOriginal() == MemoryType::MEM_L0B) {
            outValidShape.push_back(inputValidShape[1]);
        }
    }

    for (auto output : op->GetOOperands()) {
        outValidShapes.push_back(outValidShape);
    }
}
REGISTER_INFER_SHAPE_FUNC(OP_A_MUL_B, Opcode::OP_A_MUL_B, MatmulInferFunc);
REGISTER_INFER_SHAPE_FUNC(OP_A_MUL_BT, Opcode::OP_A_MUL_BT, MatmulInferFunc);
REGISTER_INFER_SHAPE_FUNC(OP_AT_MUL_B, Opcode::OP_AT_MUL_B, MatmulInferFunc);
REGISTER_INFER_SHAPE_FUNC(OP_AT_MUL_BT, Opcode::OP_AT_MUL_BT, MatmulInferFunc);

void LoadBTFBInferFunc(Operation* op,
                     std::vector<std::vector<SymbolicScalar>>& outValidShapes) {
    for (auto output : op->GetOOperands()) {
        assert(!output->GetDynValidShape().empty());
        outValidShapes.push_back(output->GetDynValidShape());
    }
}

REGISTER_INFER_SHAPE_FUNC(OP_L1_TO_BT, Opcode::OP_L1_TO_BT, LoadBTFBInferFunc);
REGISTER_INFER_SHAPE_FUNC(OP_L1_TO_FIX_QUANT_PRE, Opcode::OP_L1_TO_FIX_QUANT_PRE, LoadBTFBInferFunc);

void MatmulACCInferFunc(Operation* op,
                        std::vector<std::vector<SymbolicScalar>>& outValidShapes) {
    std::vector<SymbolicScalar> outValidShape;
    for (auto inputTensor : op->GetIOperands()) {
        if (inputTensor->GetMemoryTypeOriginal() == MemoryType::MEM_L0C) {
            outValidShape = inputTensor->GetDynValidShape();
            break;
        }
    }

    for (auto output : op->GetOOperands()) {
        outValidShapes.push_back(outValidShape);
    }
}
REGISTER_INFER_SHAPE_FUNC(OP_A_MULACC_B, Opcode::OP_A_MULACC_B, MatmulACCInferFunc);
REGISTER_INFER_SHAPE_FUNC(OP_A_MULACC_BT, Opcode::OP_A_MULACC_BT, MatmulACCInferFunc);

void LoadL0C2L1InferFunc(Operation* op,
                        std::vector<std::vector<SymbolicScalar>>& outValidShapes) {
    auto copyAttr = std::dynamic_pointer_cast<CopyOpAttribute>(op->GetOpAttribute());
    if (copyAttr != nullptr) {
        auto fromValidShape = op->GetIOperands()[0]->GetDynValidShape();
        copyAttr->SetFromDynValidShape(OpImmediate::Specified(fromValidShape));
    } else {
        ALOG_WARN_F("%s[%d] has no copy out attr, set output valid shape same as input.",
            op->GetOpcodeStr().c_str(), op->GetOpMagic());
        outValidShapes.emplace_back(op->GetIOperands()[0]->GetDynValidShape());
        return;
    }
    auto offsets = copyAttr->GetToOffset();
    auto inputShapes = copyAttr->GetToDynValidShape();
    std::vector<SymbolicScalar> outDynShape = op->GetOOperands()[0]->GetDynValidShape();
    if (outDynShape.empty()) {
        outDynShape.resize(op->GetOOperands()[0]->GetShape().size(), SymbolicScalar(0));
    }
    std::vector<SymbolicScalar> outShape;
    for (size_t i = 0; i < inputShapes.size(); i++) {
        auto inputShape = inputShapes[i].GetSpecifiedValue();
        auto offset = offsets[i].GetSpecifiedValue();
        SymbolicScalar actualDim = std::max(outDynShape[i], (inputShape + offset) * (inputShape != 0));
        outShape.emplace_back(actualDim);
    }
    for (auto output : op->GetOOperands()) {
        outValidShapes.emplace_back(outShape);
    }
}
REGISTER_INFER_SHAPE_FUNC(OP_L0C_TO_L1, Opcode::OP_L0C_TO_L1, LoadL0C2L1InferFunc);

void Load2L1InferFunc(Operation* op,
                        std::vector<std::vector<SymbolicScalar>>& outValidShapes) {
    std::vector<std::vector<SymbolicScalar>> inputValidShapes;
    for (auto inputTensor : op->GetIOperands()) {
        inputValidShapes.push_back(inputTensor->GetDynValidShape());
    }
    if (inputValidShapes.empty()) {
        return;
    }

    for (auto output : op->GetOOperands()) {
        outValidShapes.push_back(inputValidShapes[0]);
    }
}
REGISTER_INFER_SHAPE_FUNC(OP_UB_COPY_L1, Opcode::OP_UB_COPY_L1, Load2L1InferFunc);
REGISTER_INFER_SHAPE_FUNC(OP_UB_COPY_ND2NZ, Opcode::OP_UB_COPY_ND2NZ, Load2L1InferFunc);
REGISTER_INFER_SHAPE_FUNC(OP_L0C_COPY_UB, Opcode::OP_L0C_COPY_UB, Load2L1InferFunc);

// MTE infer shape func
template <bool isTrans = false>
void LoadL0InferFunc(Operation *op, std::vector<std::vector<SymbolicScalar>> &outValidShapes)
{
    ASSERT(op != nullptr);
    const std::string L1_TO_L0_OFFSET = OP_ATTR_PREFIX + "l1_to_l0_offset";
    const std::string L1_TO_L0_TILE = OP_ATTR_PREFIX + "l1_to_l0_tile";
    if (op->HasAttr(L1_TO_L0_OFFSET) && op->HasAttr(L1_TO_L0_TILE)) {
        // 大包搬运分支，无法直接从srcValidShape推导至输出dstValidShape，需要获取offset、tile信息
        std::vector<SymbolicScalar> offset;
        std::vector<SymbolicScalar> tile;
        op->GetAttr(L1_TO_L0_OFFSET, offset);
        op->GetAttr(L1_TO_L0_TILE, tile);
        ASSERT(offset.size() == SHAPE_DIM2);
        ASSERT(tile.size() == SHAPE_DIM2);
        ASSERT(!op->GetIOperands().empty() && op->GetIOperands()[0] != nullptr &&
               op->GetIOperands()[0]->GetDynValidShape().size() == SHAPE_DIM2);
        std::vector<SymbolicScalar> srcValidShape = op->GetIOperands()[0]->GetDynValidShape();
        std::vector<SymbolicScalar> dstValidShape = GetViewValidShape(
            srcValidShape, SymbolicScalar::Concrete(offset, 0), offset, SymbolicScalar::Concrete(tile, 0));
        ASSERT(dstValidShape.size() == SHAPE_DIM2);
        if constexpr (isTrans) {
            // L0A始终保持(M, K)，L0B始终保持(K, N)
            std::swap(dstValidShape[0], dstValidShape[1]);
        }
        for (auto output : op->GetOOperands()) {
            outValidShapes.push_back(dstValidShape);
        }
        return;
    }
    // 普通分支，srcValidShape与dstValidShape相同
    std::vector<std::vector<SymbolicScalar>> inputValidShapes;
    for (auto inputTensor : op->GetIOperands()) {
        ASSERT(inputTensor != nullptr);
        inputValidShapes.push_back(inputTensor->GetDynValidShape());
    }
    if (inputValidShapes.empty() || inputValidShapes[0].size() != SHAPE_DIM2) {
        return;
    }
    for (auto output : op->GetOOperands()) {
        if constexpr (isTrans) {
            outValidShapes.push_back({inputValidShapes[0][1], inputValidShapes[0][0]});
        } else {
            outValidShapes.push_back(inputValidShapes[0]);
        }
    }
}

REGISTER_INFER_SHAPE_FUNC(OP_L1_TO_L0A, Opcode::OP_L1_TO_L0A, LoadL0InferFunc<false>);
REGISTER_INFER_SHAPE_FUNC(OP_L1_TO_L0B, Opcode::OP_L1_TO_L0B, LoadL0InferFunc<false>);
REGISTER_INFER_SHAPE_FUNC(OP_L1_TO_L0_AT, Opcode::OP_L1_TO_L0_AT, LoadL0InferFunc<true>);
REGISTER_INFER_SHAPE_FUNC(OP_L1_TO_L0_BT, Opcode::OP_L1_TO_L0_BT, LoadL0InferFunc<true>);

void CopyInInferFunc(Operation* op,
                     std::vector<std::vector<SymbolicScalar>>& outValidShapes) {
    auto copyOpAttribute = std::dynamic_pointer_cast<CopyOpAttribute>(op->GetOpAttribute());
    if (!(op->GetOOperands()[0]->GetDynValidShape().empty())) {
        outValidShapes.push_back(op->GetOOperands()[0]->GetDynValidShape());
        if (copyOpAttribute != nullptr && (copyOpAttribute->GetToDynValidShape()).empty()) {
            auto toDynShape = OpImmediate::Specified(op->GetOOperands()[0]->GetDynValidShape());
            copyOpAttribute->SetToDynValidShape(toDynShape);
        }
        return;
    }

    // 连接incast
    auto toValidShape = copyOpAttribute->GetToDynValidShape();
    std::vector<SymbolicScalar> toValidShapeSym(toValidShape.size());
    OpImmediate::NormalizeValue(toValidShapeSym, 0, toValidShape, 0, false);
    auto toValidShapeValue = SymbolicScalar::Concrete(toValidShapeSym, -1);
    auto tileShape = copyOpAttribute->GetShape();
    std::vector<SymbolicScalar> tileShapeSym(tileShape.size());
    OpImmediate::NormalizeValue(tileShapeSym, 0, tileShape, 0, false);
    if (!toValidShape.empty()) {
        for (auto output : op->GetOOperands()) {
            outValidShapes.push_back(toValidShapeSym);
        }
        return;
    }
    if (!(op->GetOOperands()[0]->GetDynValidShape().empty())) {
        outValidShapes.push_back(op->GetOOperands()[0]->GetDynValidShape());
        auto toDynShape = OpImmediate::Specified(op->GetOOperands()[0]->GetDynValidShape());
        copyOpAttribute->SetToDynValidShape(toDynShape);
        return;
    }
    // 临时空间，固定大小
    if (op->GetIOperands()[0]->GetProducers().empty()) {
        if (toValidShape.empty()) {
            std::vector<SymbolicScalar> toValidShapeVec;
            for (auto dim : copyOpAttribute->GetShape()) {
                toValidShapeVec.push_back(dim.GetSpecifiedValue());
            }
            for (auto output : op->GetOOperands()) {
                outValidShapes.push_back(toValidShapeVec);
            }
        }
        auto toDynShape = OpImmediate::Specified(outValidShapes[0]);
        copyOpAttribute->SetToDynValidShape(toDynShape);
        return;
    }
    // 子图边界, 需要重新推导
    std::vector<std::vector<SymbolicScalar>> inputShapes;
    for (auto inputTensor : op->GetIOperands()) {
        inputShapes.push_back(inputTensor->GetDynValidShape());
    }
    auto offset = copyOpAttribute->GetFromOffset();
    std::vector<SymbolicScalar> oriOffset;
    for (auto offsetValue : offset) {
        oriOffset.push_back(offsetValue.GetSpecifiedValue());
    }
    std::vector<SymbolicScalar> outputShape;
    for (size_t i = 0U; i < inputShapes[0].size(); i++) {
        SymbolicScalar actualDim = std::max(0, std::min((inputShapes[0][i] - oriOffset[i]), tileShapeSym[i]));
        outputShape.push_back(actualDim);
    }
    for (auto output : op->GetOOperands()) {
        outValidShapes.push_back(outputShape);
    }
    // 设置validshape到copyin的toDynvalidshape
    auto toDynShape = OpImmediate::Specified(outputShape);
    copyOpAttribute->SetToDynValidShape(toDynShape);
}
REGISTER_INFER_SHAPE_FUNC(OP_COPY_IN, Opcode::OP_COPY_IN, CopyInInferFunc);

void ShmemGetGm2UBInferFunc(Operation* op,
                      std::vector<std::vector<SymbolicScalar>>& outValidShapes)
{
    auto copyOpAttribute = std::dynamic_pointer_cast<CopyOpAttribute>(op->GetOpAttribute());
    std::vector<SymbolicScalar> toValidShapeSym(copyOpAttribute->GetToDynValidShape().size());
    OpImmediate::NormalizeValue(toValidShapeSym, 0, copyOpAttribute->GetToDynValidShape(), 0, false);
    outValidShapes.push_back(toValidShapeSym);
    outValidShapes.push_back(std::vector<SymbolicScalar>{});
}
REGISTER_INFER_SHAPE_FUNC(OP_SHMEM_GET_GM2UB, Opcode::OP_SHMEM_GET_GM2UB, ShmemGetGm2UBInferFunc);

void CopyOutInferFunc(Operation* op,
                      std::vector<std::vector<SymbolicScalar>>& outValidShapes)
{
    auto copyOpAttribute = std::dynamic_pointer_cast<CopyOpAttribute>(op->GetOpAttribute());
    if (copyOpAttribute != nullptr) {
        copyOpAttribute->SetFromDynValidShape(OpImmediate::Specified(op->GetIOperands()[0]->GetDynValidShape()));
    } else {
        ALOG_WARN_F("Copyout [%d] has no copy out attr.", op->GetOpMagic());
        outValidShapes.push_back(op->GetIOperands()[0]->GetDynValidShape());
        return;
    }

    // 多个tile块copyout到同一个tensor时， 每一个tile都需要推导
    bool needInferShape = false;
    if (!(op->GetOOperands()[0]->GetDynValidShape().empty()) && !op->GetOOperands()[0]->GetAttr(COPY_OUT_FORCE_INFER_SHAPE, needInferShape)) {
        outValidShapes.push_back(op->GetOOperands()[0]->GetDynValidShape());
        return;
    }

    op->GetOOperands()[0]->SetAttr(COPY_OUT_FORCE_INFER_SHAPE, true);

    auto offset = copyOpAttribute->GetToOffset();
    std::vector<SymbolicScalar> oriOffset;
    for (auto offsetValue : offset) {
        oriOffset.push_back(offsetValue.GetSpecifiedValue());
    }

    std::vector<std::vector<SymbolicScalar>> inputShapes;
    std::vector<std::vector<int64_t>> staticInputShapes;

    for (auto inputTensor : op->GetIOperands()) {
        inputShapes.push_back(inputTensor->GetDynValidShape());
        staticInputShapes.push_back(inputTensor->GetShape());
    }
    std::vector<SymbolicScalar> outDynShape = op->GetOOperands()[0]->GetDynValidShape();
    if (outDynShape.empty()) {
        for (size_t i = 0; i < op->GetOOperands()[0]->GetShape().size(); ++i) {
            outDynShape.push_back(SymbolicScalar(0));
        }
    }
    std::vector<SymbolicScalar> outShape;
    for (size_t i = 0U; i < inputShapes[0].size(); i++) {
        SymbolicScalar actualDim;
        if (staticInputShapes[0][i] == op->GetOOperands()[0]->GetShape()[i]) { //src的该维度没有被切分，assmble后该维度大小不变
            actualDim = std::max(SymbolicScalar(0), inputShapes[0][i] + oriOffset[i]);
        } else {
            actualDim = std::max(outDynShape[i], (inputShapes[0][i] + oriOffset[i]) * (inputShapes[0][i] != 0));
        }
        outShape.push_back(actualDim);
    }
    for (auto output : op->GetOOperands()) {
        outValidShapes.push_back(outShape);
    }
}
REGISTER_INFER_SHAPE_FUNC(OP_COPY_OUT, Opcode::OP_COPY_OUT, CopyOutInferFunc);

// MTE infer shape func
void TransposeInferFunc(Operation* op,
    std::vector<std::vector<SymbolicScalar>>& outValidShapes) {
    std::vector<std::vector<SymbolicScalar>> inputValidShapes;
    for (auto inputTensor : op->GetIOperands()) {
        inputValidShapes.push_back(inputTensor->GetDynValidShape());
    }
    if (inputValidShapes.empty()) {
        return;
    }
    for (auto output : op->GetOOperands()) {
        std::vector<SymbolicScalar> res;
        res.insert(res.end(), inputValidShapes[0].begin(), inputValidShapes[0].end());
        auto axises = op->GetVectorIntAttribute<int>(OP_ATTR_PREFIX + "shape");
        size_t index0 = axises[0];
        size_t index1 = axises[1];
        if (index0 < res.size() && index1 < res.size()) {
            std::swap(res[index0], res[index1]);
        }
        outValidShapes.push_back(res);
    }
    if (op->GetOpcode() == Opcode::OP_TRANSPOSE_MOVEIN) {
        auto copyOpAttribute = std::dynamic_pointer_cast<CopyOpAttribute>(op->GetOpAttribute());
        if (copyOpAttribute != nullptr) {
            copyOpAttribute->SetToDynValidShape(OpImmediate::Specified(outValidShapes[0]));
        }
    }
    if (op->GetOpcode() == Opcode::OP_TRANSPOSE_MOVEOUT) {
        auto copyOpAttribute = std::dynamic_pointer_cast<CopyOpAttribute>(op->GetOpAttribute());
        if (copyOpAttribute != nullptr) {
            copyOpAttribute->SetFromDynValidShape(OpImmediate::Specified(outValidShapes[0]));
            std::vector<SymbolicScalar> outDynShape = op->GetOOperands()[0]->GetDynValidShape();
            if (!outDynShape.empty()) {
                auto dynOffset = copyOpAttribute->GetToOffset();
                std::vector<SymbolicScalar> outShape;
                for (size_t i = 0U; i < dynOffset.size(); i++) {
                    SymbolicScalar actualDim = std::max(outDynShape[i],
                        (outValidShapes[0][i] + dynOffset[i].GetSpecifiedValue()) * (outValidShapes[0][i] != 0));
                    outShape.push_back(actualDim);
                }
                outValidShapes[0] = outShape;
            }
        }
    }
}

REGISTER_INFER_SHAPE_FUNC(OP_TRANSPOSE_VNCHWCONV, Opcode::OP_TRANSPOSE_VNCHWCONV, TransposeInferFunc);
REGISTER_INFER_SHAPE_FUNC(OP_TRANSPOSE_MOVEIN, Opcode::OP_TRANSPOSE_MOVEIN, TransposeInferFunc);
REGISTER_INFER_SHAPE_FUNC(OP_TRANSPOSE_MOVEOUT, Opcode::OP_TRANSPOSE_MOVEOUT, TransposeInferFunc);

void ViewInferFunc(Operation* op, std::vector<std::vector<SymbolicScalar>>& outValidShapes) {
    auto viewOpAttribute = std::dynamic_pointer_cast<ViewOpAttribute>(op->GetOpAttribute());
    if (viewOpAttribute == nullptr) {
        ALOG_WARN_F("View [%d] has no view attr.", op->GetOpMagic());
        outValidShapes.push_back(op->GetIOperands()[0]->GetDynValidShape());
        return;
    }
    // view的toDynValidShape是前端已经预设好，直接使用即可
    auto toValidShape = viewOpAttribute->GetToDynValidShape();
    if (!toValidShape.empty()) {
        for (auto output : op->GetOOperands()) {
            outValidShapes.push_back(toValidShape);
        }
    } else {
        auto inputValidShape = op->GetIOperands()[0]->GetDynValidShape();
        if (inputValidShape.empty()) {
            auto shapeImm = OpImmediate::Specified(op->GetIOperands()[0]->GetShape());
            inputValidShape.resize(shapeImm.size());
            OpImmediate::NormalizeValue(inputValidShape, 0, shapeImm, 0, false);
        }
        auto newDynValidShape = GetViewValidShape(inputValidShape, viewOpAttribute->GetFromOffset(),
                                                    viewOpAttribute->GetFromDynOffset(), op->GetOOperands()[0]->oriShape);
        for (auto output : op->GetOOperands()) {
            outValidShapes.push_back(newDynValidShape);
        }
        viewOpAttribute->SetToDynValidShape(newDynValidShape);
    }
}
REGISTER_INFER_SHAPE_FUNC(OP_VIEW, Opcode::OP_VIEW, ViewInferFunc);

void AssembleInferFunc(Operation* op, std::vector<std::vector<SymbolicScalar>>& outValidShapes) {
    auto assembleOpAttribute = std::dynamic_pointer_cast<AssembleOpAttribute>(op->GetOpAttribute());
    if (assembleOpAttribute != nullptr) {
        auto fromValidShape = op->GetIOperands()[0]->GetDynValidShape();
        assembleOpAttribute->SetFromDynValidShape(fromValidShape);
    } else {
        ALOG_WARN_F("Copyout [%d] has no copy out attr.", op->GetOpMagic());
        outValidShapes.push_back(op->GetIOperands()[0]->GetDynValidShape());
        return;
    }
    auto offset = assembleOpAttribute->GetToOffset();
    auto inputShapes = op->GetIOperands()[0]->GetDynValidShape();
    std::vector<SymbolicScalar> outDynShape = op->GetOOperands()[0]->GetDynValidShape();
    if (outDynShape.empty()) {
        for (size_t i = 0; i < op->GetOOperands()[0]->GetShape().size(); ++i) {
            outDynShape.push_back(SymbolicScalar(0));
        }
    }
    std::vector<SymbolicScalar> outShape;
    for (size_t i = 0U; i < inputShapes.size(); i++) {
        SymbolicScalar actualDim = std::max(outDynShape[i], (inputShapes[i] + offset[i]) * (inputShapes[i] != 0));
        outShape.push_back(actualDim);
    }
    for (auto output : op->GetOOperands()) {
        outValidShapes.push_back(outShape);
    }
}
REGISTER_INFER_SHAPE_FUNC(OP_ASSEMBLE, Opcode::OP_ASSEMBLE, AssembleInferFunc);

const std::string TOPK_AXIS = OP_ATTR_PREFIX + "axis";
const std::string TOPK_ORDER = OP_ATTR_PREFIX + "order";
const std::string TOPK_KVALUE = OP_ATTR_PREFIX + "kvalue";
const std::string EXTRACT_MASKMODE = OP_ATTR_PREFIX + "makeMode";
constexpr int32_t blockSize = 32;
constexpr int32_t kFactorSize = 4;
constexpr int32_t NUM3 = 3;
constexpr int32_t kBlockFpNum = 8;

// m,n -> m,4*n align32
void BitSortFunc(Operation *op, std::vector<std::vector<SymbolicScalar>> &outValidShapes) {
    std::vector<std::vector<SymbolicScalar>> inputValidShapes;
    for (auto inputTensor : op->GetIOperands()) {
        inputValidShapes.push_back(inputTensor->GetDynValidShape());
    }
    if (inputValidShapes.empty()) {
        return;
    }
    std::vector<SymbolicScalar> res(inputValidShapes[0]);
    auto topk_axis = op->GetIntAttribute(TOPK_AXIS);
    res[topk_axis] = (res[topk_axis] + blockSize - 1) / blockSize * blockSize;
    res[topk_axis] = res[topk_axis] * NUM2 + inputValidShapes[0][topk_axis];
    outValidShapes.push_back(res);
}

REGISTER_INFER_SHAPE_FUNC(OP_BITSORT, Opcode::OP_BITSORT, BitSortFunc);

// m,4 *n align32byte -> m, 2 * k align8
void MrgSortFunc(Operation *op, std::vector<std::vector<SymbolicScalar>> &outValidShapes) {
    std::vector<std::vector<SymbolicScalar>> inputValidShapes;
    for (auto inputTensor : op->GetIOperands()) {
        inputValidShapes.push_back(inputTensor->GetDynValidShape());
    }
    if (inputValidShapes.empty()) {
        return;
    }
    std::vector<SymbolicScalar> res(inputValidShapes[0]);
    auto topk_axis = op->GetIntAttribute(TOPK_AXIS);
    auto topk_kvalue = op->GetIntAttribute(TOPK_KVALUE);
    SymbolicScalar tmp = (res[topk_axis] + blockSize - 1) / blockSize * blockSize;
    res[topk_axis] = std::min(res[topk_axis] - tmp / NUM3 * NUM2,
     (topk_kvalue + kBlockFpNum - 1) / kBlockFpNum * kBlockFpNum) * NUM2;
    outValidShapes.push_back(res);
}

REGISTER_INFER_SHAPE_FUNC(OP_MRGSORT, Opcode::OP_MRGSORT, MrgSortFunc);

void TiledMrgSortFunc(Operation *op, std::vector<std::vector<SymbolicScalar>> &outValidShapes) {
    std::vector<std::vector<SymbolicScalar>> inputValidShapes;
    for (auto inputTensor : op->GetIOperands()) {
        inputValidShapes.push_back(inputTensor->GetDynValidShape());
    }
    if (inputValidShapes.empty()) {
        return;
    }
    auto outValidShape = inputValidShapes[0];
    for (auto output : op->GetOOperands()) {
        outValidShapes.push_back(outValidShape);
    }
}

REGISTER_INFER_SHAPE_FUNC(OP_TILEDMRGSORT, Opcode::OP_TILEDMRGSORT, TiledMrgSortFunc);

// m, 2 * k align8 -> m, k
void ExtractFunc(Operation *op, std::vector<std::vector<SymbolicScalar>> &outValidShapes) {
    std::vector<std::vector<SymbolicScalar>> inputValidShapes;
    for (auto inputTensor : op->GetIOperands()) {
        inputValidShapes.push_back(inputTensor->GetDynValidShape());
    }
    if (inputValidShapes.empty()) {
        return;
    }
    std::vector<SymbolicScalar> res(inputValidShapes[0]);
    res.back() = op->GetIntAttribute(TOPK_KVALUE);
    outValidShapes.push_back(res);
}

REGISTER_INFER_SHAPE_FUNC(OP_EXTRACT, Opcode::OP_EXTRACT, ExtractFunc);

void VecDupInferFunc(Operation *op, std::vector<std::vector<SymbolicScalar>> &validShapes) {
    std::vector<SymbolicScalar> validShape;
    op->GetAttr(OP_ATTR_PREFIX + "validShape", validShape);
    validShapes.push_back(validShape);
}
REGISTER_INFER_SHAPE_FUNC(OP_VEC_DUP, Opcode::OP_VEC_DUP, VecDupInferFunc);

void ReshapeInferFunc(Operation *op, std::vector<std::vector<SymbolicScalar>> &validShapes) {
    std::vector<SymbolicScalar> validShape;
    if (op->GetAttr(OP_ATTR_PREFIX + "validShape", validShape) && validShape.size() != 0) {
        validShapes.push_back(validShape);
    } else {
        auto dstShape = op->GetOOperands()[0]->GetShape();
        validShapes.push_back(SymbolicScalar::FromConcrete(dstShape));
    }
}
REGISTER_INFER_SHAPE_FUNC(OP_RESHAPE, Opcode::OP_RESHAPE, ReshapeInferFunc);

void BrcbInferFunc(Operation* op, std::vector<std::vector<SymbolicScalar>>& outValidShapes) {
    auto dimSize = op->GetIOperands()[0]->GetDynValidShape().size();
    std::vector<SymbolicScalar> outValidShape;
    for (size_t i = 0; i < dimSize - 1; i++) {
        outValidShape.push_back(op->GetIOperands()[0]->GetDynValidShape()[i]);
    }
    int64_t lastDimShape = blockSize / BytesOf(op->GetIOperands()[0]->Datatype());
    outValidShape.push_back(lastDimShape);
    for (auto output : op->GetOOperands()) {
        outValidShapes.push_back(outValidShape);
    }
}
REGISTER_INFER_SHAPE_FUNC(OP_BRCB, Opcode::OP_BRCB, BrcbInferFunc);
}  // namespace npu::tile_fwk
