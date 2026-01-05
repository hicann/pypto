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
 * \file tensor_transformation.cpp
 * \brief
 */

#include "unary.h"
#include "tensor_transformation.h"
#include "interface/utils/operator_tracer.h"

namespace npu::tile_fwk {

struct ExpandInfo {
    const std::shared_ptr<LogicalTensor> &srcTensor;
    const std::shared_ptr<LogicalTensor> &result;
    std::vector<int64_t> &viewShape;
    std::vector<int64_t> &offset;
    const int expandDim;
    ExpandInfo(const std::shared_ptr<LogicalTensor> &srcTensor0, const std::shared_ptr<LogicalTensor> &result0,
        std::vector<int64_t> &viewShape0, std::vector<int64_t> &offset0, const int expandDim0)
        : srcTensor(srcTensor0), result(result0), viewShape(viewShape0), offset(offset0), expandDim(expandDim0) {}
};

void CheckExpandTensorVaild(const LogicalTensorPtr &operand, const LogicalTensorPtr &result) {
    if (operand->shape.size() != result->shape.size()) {
        ASSERT(false && "Dims not match");
    }

    for (size_t i = 0; i < result->shape.size(); ++i) {
        if (operand->shape[i] != result->shape[i] && operand->shape[i] != 1) {
            ASSERT(0 && "shape not match");
        }
    }
    
    int numExpandAxis = 0;
    for (size_t i = 0; i < result->shape.size(); ++i) {
        if (operand->shape[i] != result->shape[i]) {
            numExpandAxis++;
        }
    }
    ASSERT(numExpandAxis <= 1) << "Only allow to expand one axis";
}

void ExpandTile(Function &function, const struct ExpandInfo &expandInfo) {
    auto resultTile = expandInfo.result->View(function, expandInfo.viewShape, expandInfo.offset);

    std::vector<int64_t> srcShape(expandInfo.srcTensor->shape.size(), 1);
    for (size_t i = 0; i < expandInfo.result->shape.size(); i++) {
        srcShape[i] = std::min(expandInfo.viewShape[i], expandInfo.srcTensor->shape[i]);
    }

    std::vector<int64_t> srcOffset = expandInfo.offset;
    for (size_t j = 0; j < srcOffset.size(); j++) {
        if (expandInfo.srcTensor->shape[j] < expandInfo.result->shape[j]) {
            srcOffset[j] = expandInfo.offset[j] % expandInfo.srcTensor->shape[j];
        }
    }
    auto srcTile = expandInfo.srcTensor->View(function, srcShape, srcOffset);
    auto &newOp = function.AddOperation("TILE_EXPAND", {srcTile}, {resultTile});
    newOp.SetAttribute(OP_ATTR_PREFIX + "EXPANDDIM", expandInfo.expandDim);
    newOp.SetAttribute(OP_ATTR_PREFIX + "validShape", resultTile->GetDynValidShape());
}

void ExpandTile(Function &function, const TileShape &tileShape, int dimIdx, const struct ExpandInfo &expandInfo,
    std::vector<SymbolicScalar> validShape) {
    if (static_cast<size_t>(dimIdx) == expandInfo.result->shape.size()) {
        ExpandTile(function, expandInfo);
        return;
    }
    auto &vecTile = tileShape.GetVecTile();
    for (int i = 0; i < expandInfo.result->shape[dimIdx]; i += vecTile[dimIdx]) {
        expandInfo.offset[dimIdx] = i;
        expandInfo.viewShape[dimIdx] =
            std::min(expandInfo.result->shape[dimIdx] - i, static_cast<int64_t>(vecTile[dimIdx]));
        ExpandTile(function, tileShape, dimIdx + 1, expandInfo, validShape);
    }
}

void Expand(Function &function, const TileShape &tileShape, const LogicalTensorPtr &operand,
    const std::vector<LogicalTensorPtr> &other, const LogicalTensorPtr &result) {
    CheckExpandTensorVaild(operand, result);
    ASSERT(function.GetGraphType() == GraphType::TILE_GRAPH) << "The GetGraphType of function is incorrect";
    std::vector<int64_t> offset(result->shape.size(), 0);
    std::vector<int64_t> viewShape(result->shape.size(), 1);
    std::vector<SymbolicScalar> outValidShape;
    int expandDim = -1;
    for (size_t i = 0; i < result->shape.size(); ++i) {
        if (operand->shape[i] != result->shape[i]) {
            expandDim = i;
            for (auto it : other) {
                if (it != nullptr && it->shape[i] == result->shape[i]) {
                    if (it->GetDynValidShape().empty()) {
                        outValidShape.push_back(it->shape[i]);
                    } else {
                        outValidShape.push_back(it->GetDynValidShape()[i]);
                    }
                    break;
                }
            }
        } else {
            if (operand->GetDynValidShape().empty()) {
                outValidShape.push_back(operand->shape[i]);
            } else {
                outValidShape.push_back(operand->GetDynValidShape()[i]);
            }
        }
    }

    result->UpdateDynValidShape(outValidShape);
    struct ExpandInfo expandInfo(operand, result, viewShape, offset, expandDim);
    ExpandTile(function, tileShape, 0, expandInfo, outValidShape);
}

void ExpandWithResultValidShape(Function &function, const TileShape &tileShape, const LogicalTensorPtr &operand,
    const LogicalTensorPtr &result, const std::vector<SymbolicScalar> resultValidShape) {
    CheckExpandTensorVaild(operand, result);
    ASSERT(function.GetGraphType() == GraphType::TILE_GRAPH) << "The GetGraphType of function is incorrect";
    std::vector<int64_t> offset(result->shape.size(), 0);
    std::vector<int64_t> viewShape(result->shape.size(), 1);
    int expandDim = -1;
    for (size_t i = 0; i < result->shape.size(); ++i) {
        if (operand->shape[i] != result->shape[i]) {
            expandDim = i;
        }
    }
    result->UpdateDynValidShape(resultValidShape);
    struct ExpandInfo expandInfo(operand, result, viewShape, offset, expandDim);
    ExpandTile(function, tileShape, 0, expandInfo, resultValidShape);
}

void TiledExpand(Function &function, const TileShape &tileShape, const LogicalTensorPtr &operand,
    const LogicalTensorPtr &result, const std::vector<SymbolicScalar> &validShape) {
    CheckExpandTensorVaild(operand, result);
    ASSERT(function.GetGraphType() == GraphType::TILE_GRAPH) << "The GetGraphType of function is incorrect";

    std::vector<int64_t> offset(result->shape.size(), 0);
    std::vector<int64_t> viewShape(result->shape.size(), 1);
    int expandDim = -1;
    for (size_t i = 0; i < result->shape.size(); ++i) {
        if (operand->shape[i] != result->shape[i]) {
            expandDim = i;
        }
    }
    result->UpdateDynValidShape(validShape);
    struct ExpandInfo expandInfo(operand, result, viewShape, offset, expandDim);
    ExpandTile(function, tileShape, 0, expandInfo, validShape);
}

Tensor TensorExpandOperation(Function &function, const LogicalTensorPtr &operand, const std::vector<int64_t> &dstShape,
    const std::vector<SymbolicScalar> &validShape) {
    auto result = std::make_shared<LogicalTensor>(function, operand->Datatype(), dstShape, validShape);
    auto &op = function.AddOperation(Opcode::OP_EXPAND, {operand}, {result});

    op.SetAttribute(OP_ATTR_PREFIX + "shape", dstShape);
    op.SetAttribute(OP_ATTR_PREFIX + "validShape", validShape);
    function.UpdateTensorDataUsage(op);
    return result;
}

Tensor TensorJustNeedCopyOperation(Function &function, const LogicalTensorPtr &operand,
    const std::vector<int64_t> &dstShape, const std::vector<SymbolicScalar> &validShape) {
    auto result = std::make_shared<LogicalTensor>(function, operand->Datatype(), dstShape, validShape);
    function.AddOperation(Opcode::OP_REGISTER_COPY, {operand}, {result});
    return result;
}

Tensor Expand(const Tensor &self, const std::vector<int64_t> &dstShape, std::vector<SymbolicScalar> validShape) {
    DECLARE_TRACER();

    ASSERT(self.GetShape().size() == dstShape.size()) << "The shape size of self and dst should be equal";
    int numExpandAxis = 0;
    for (size_t i = 0; i < dstShape.size(); ++i) {
        if (self.GetShape()[i] != dstShape[i]) {
            numExpandAxis++;
        }
    }
    ASSERT(numExpandAxis <= 1) << "Only allow to expand one axis";
    if (validShape.empty()) {
        for (size_t i = 0; i < dstShape.size(); ++i) {
            if (self.GetShape()[i] != dstShape[i]) {
                validShape.emplace_back(dstShape[i]);
            } else {
                validShape.emplace_back(self.GetShape()[i]);
            }
        }
    }
    bool needExpand = false;
    for (size_t i = 0; i < dstShape.size(); ++i) {
        if (self.GetShape()[i] != dstShape[i]) {
            needExpand = true;
        }
    }
    if (needExpand) {
        RETURN_CALL(
            ExpandOperation, *Program::GetInstance().GetCurrentFunction(), self.GetStorage(), dstShape, validShape);
    } else {
        RETURN_CALL(JustNeedCopyOperation, *Program::GetInstance().GetCurrentFunction(), self.GetStorage(), dstShape,
            validShape);
    }
}

enum class TransposeOpType {
    TRANSPOSE_MOVEIN,
    TRANSPOSE_MOVEOUT,
    TRANSPOSE_VNCHWCONV,
};

template <TransposeOpType T>
Opcode GetTransposeOpName() {
#define CASE(X) \
    case TransposeOpType::X: return Opcode::OP_##X
    switch (T) {
        CASE(TRANSPOSE_MOVEIN);
        CASE(TRANSPOSE_MOVEOUT);
        CASE(TRANSPOSE_VNCHWCONV);
        default: ASSERT(false && "unknown transpose op type");
    }
#undef CASE
}

inline void UnalignPadTmpBufTile(std::vector<int64_t> &shape, int blockElem) {
    // tmpbuf按16 8对齐
    auto size = shape.size();
    if (size >= NUM_VALUE_2) {
        shape[size - NUM_VALUE_2] = AlignUp(shape[size - NUM_VALUE_2], (int64_t)VNCHWCONV_REPEAT);
        shape[size - 1] = AlignUp(shape[size - 1], blockElem);
    }
}

template <TransposeOpType T>
void TiledInnerTranspose(Function &function, const TileShape &tileShape, const int cur, Input &input,
    const LogicalTensorPtr &result, const std::vector<int> &shape) {
    int shapeSize = input.tensor.GetShape().size();
    if (cur == shapeSize) {
        auto tile = input.tensor.GetStorage()->View(function, input.tileInfo.shape, input.tileInfo.offset);
        std::vector<int64_t> resultTileShape(input.tileInfo.shape);
        std::swap(resultTileShape[shape[0]], resultTileShape[shape[1]]);
        std::vector<int64_t> resultTileOfs(input.tileInfo.offset);
        std::swap(resultTileOfs[shape[0]], resultTileOfs[shape[1]]);
        auto resultTile = result->View(function, resultTileShape, resultTileOfs);
        if (T == TransposeOpType::TRANSPOSE_MOVEOUT || T == TransposeOpType::TRANSPOSE_MOVEIN) {
            auto &op = function.AddOperation(GetTransposeOpName<T>(), {tile}, {resultTile});
            op.SetAttribute(OP_ATTR_PREFIX + "shape", shape);
        } else {
            std::vector<int64_t> tmpShape(input.tileInfo.shape);
            int64_t blockElem = BLOCK_SIZE / static_cast<int>(BytesOf(tile->Datatype()));
            UnalignPadTmpBufTile(tmpShape, blockElem);
            auto tempTensor = std::make_shared<LogicalTensor>(function, tile->Datatype(), tmpShape);
            tempTensor->dynValidShape_ = SymbolicScalar::FromConcrete(tmpShape);
            auto &op = function.AddOperation(GetTransposeOpName<T>(), {tile}, {resultTile, tempTensor});
            op.SetAttribute(OP_ATTR_PREFIX + "shape", shape);
        }
        return;
    }
    auto &vecTile = tileShape.GetVecTile();
    for (int i = 0; i < input.tensor.GetShape()[cur]; i += vecTile[cur]) {
        input.tileInfo.shape[cur] = std::min(input.tensor.GetShape()[cur] - i, vecTile[cur]);
        input.tileInfo.offset[cur] = i;
        TiledInnerTranspose<T>(function, tileShape, cur + 1, input, result, shape);
    }
}

template <TransposeOpType T>
void TiledInnerTranspose(Function &function, const TileShape &tileShape, const LogicalTensorPtr &operand,
    const LogicalTensorPtr &result, const std::vector<int> &shape) {
    TileInfo tileInfo(result->shape.size(), result->offset.size());
    auto input = Input{operand, tileInfo};
    TiledInnerTranspose<T>(function, tileShape, 0, input, result, shape);
}

void TensorInnerTranspose(
    Function &function, const LogicalTensorPtr &self, const LogicalTensorPtr &result, std::vector<int> perm) {
    if (perm[0] != (int)self->shape.size() - 1 && perm[1] != (int)self->shape.size() - 1) {
        auto &operation = function.AddOperation(Opcode::OP_TRANSPOSE_MOVEOUT, {self}, {result});
        operation.SetAttribute(OP_ATTR_PREFIX + "shape", perm);
        return;
    }

    if (perm[0] == (int)self->shape.size() - 2 && // last 2 dims transpose
        perm[1] == (int)self->shape.size() - 1) {
        auto &operation = function.AddOperation(Opcode::OP_TRANSPOSE_VNCHWCONV, {self}, {result});
        operation.SetAttribute(OP_ATTR_PREFIX + "shape", perm);
        return;
    }

    ASSERT(self->shape.size() == 3 || self->shape.size() == 4) // input should be 3 or 4 dims
        << "Transpose shape should be [A1,T1,A2,T2] or [T1,A2,T2].";

    // [A1,T1,A2,T2] to [A1,A2,T1,T2] or [T1,A2,T2] to [A2,T1,T2]
    auto oldVecTileShapes = TileShape::Current().GetVecTile();
    auto newVecTileShape = oldVecTileShapes;
    std::vector<int64_t> tmpShape(self->shape);
    int dim1 = (tmpShape.size() == 3) ? 0 : 1; // if input is 3 dims, dim1 = 0, otherwise dim1 = 1
    int dim2 = (tmpShape.size() == 3) ? 1 : 2; // if input is 3 dims, dim2 = 1, otherwise dim2 = 2
    std::swap(tmpShape[dim1], tmpShape[dim2]);
    std::swap(newVecTileShape[dim1], newVecTileShape[dim2]);
    auto moveInResult = std::make_shared<LogicalTensor>(function, self->Datatype(), tmpShape);
    auto &inOp = function.AddOperation(Opcode::OP_TRANSPOSE_MOVEIN, {self}, {moveInResult});
    inOp.SetAttribute(OP_ATTR_PREFIX + "shape", std::vector<int>{dim1, dim2});
    TileShape::Current().SetVecTile(newVecTileShape);

    // [A1,A2,T1,T2] to [A1,A2,T2,T1] or [A2,T1,T2] to [A2,T2,T1]
    tmpShape = moveInResult->shape;
    dim1 = (tmpShape.size() == 3) ? 1 : 2; // if input is 3 dims, dim1 = 1, otherwise dim1 = 2
    dim2 = (tmpShape.size() == 3) ? 2 : 3; // if input is 3 dims, dim2 = 2, otherwise dim2 = 3
    std::swap(tmpShape[dim1], tmpShape[dim2]);
    std::swap(newVecTileShape[dim1], newVecTileShape[dim2]);
    auto vnchwconvResult = std::make_shared<LogicalTensor>(function, self->Datatype(), tmpShape);
    auto &convOp = function.AddOperation(Opcode::OP_TRANSPOSE_VNCHWCONV, {moveInResult}, {vnchwconvResult});
    convOp.SetAttribute(OP_ATTR_PREFIX + "shape", std::vector<int>{dim1, dim2});
    TileShape::Current().SetVecTile(newVecTileShape);

    // [A1,A2,T2,T1] to [A1,T2,A2,T1] or [A2,T2,T1] to [T2,A2,T1]
    tmpShape = vnchwconvResult->shape;
    dim1 = (tmpShape.size() == 3) ? 0 : 1; // if input is 3 dims, dim1 = 0, otherwise dim1 = 1
    dim2 = (tmpShape.size() == 3) ? 1 : 2; // if input is 3 dims, dim2 = 1, otherwise dim2 = 2
    std::swap(tmpShape[dim1], tmpShape[dim2]);
    auto &outOp = function.AddOperation(Opcode::OP_TRANSPOSE_MOVEOUT, {vnchwconvResult}, {result});
    outOp.SetAttribute(OP_ATTR_PREFIX + "shape", std::vector<int>{dim1, dim2});
    TileShape::Current().SetVecTile(oldVecTileShapes);
}

bool MergeTransposeAxis(const Tensor &operand, std::vector<int64_t> &inputShape, std::vector<int64_t> &vecTileShape,
    std::vector<SymbolicScalar> &validShape, std::vector<int> &transposeShape) {
    auto oldTransposeShape = transposeShape;
    int64_t pre = 1;
    int64_t mid = 1;
    int64_t after = 1;
    int64_t preTileShape = 1;
    int64_t midTileShape = 1;
    int64_t afterTileShape = 1;
    SymbolicScalar preValidShape = 1;
    SymbolicScalar midValidShape = 1;
    SymbolicScalar afterValidShape = 1;
    int preNum = 0;
    int midNum = 0;
    int afterNum = 0;
    auto oldVecTileShapes = TileShape::Current().GetVecTile();
    auto oldValidShapes = validShape;
    for (int i = 0; i < (int)operand.GetShape().size(); i++) {
        if (i < oldTransposeShape[0]) {
            pre *= operand.GetShape()[i];
            preTileShape *= oldVecTileShapes[i];
            preValidShape = preValidShape * oldValidShapes[i];
            preNum++;
        } else if (i < oldTransposeShape[1] && i > oldTransposeShape[0]) {
            mid *= operand.GetShape()[i];
            midTileShape *= oldVecTileShapes[i];
            midValidShape = midValidShape * oldValidShapes[i];
            midNum++;
        } else if (i > oldTransposeShape[1]) {
            after *= operand.GetShape()[i];
            afterTileShape *= oldVecTileShapes[i];
            afterValidShape = afterValidShape * oldValidShapes[i];
            afterNum++;
        }
    }

    if (preNum <= 1 && midNum <= 1 && afterNum <= 1) {
        return false;
    }
    if (operand.GetShape().size() <= 5 && // tileop支持5维
        oldTransposeShape[0] == (int)operand.GetShape().size() - 2 && // 最后2维转置
        oldTransposeShape[1] == (int)operand.GetShape().size() - 1) {
        return false;
    }

    // [A1,T1,A2,T2,A3]
    validShape.clear();
    if (preNum > 0) {
        inputShape.push_back(pre);
        vecTileShape.push_back(preTileShape);
        validShape.push_back(preValidShape);
        transposeShape[0] -= (preNum - 1);
        transposeShape[1] -= (preNum - 1);
    }
    inputShape.push_back(operand.GetShape()[oldTransposeShape[0]]);
    vecTileShape.push_back(oldVecTileShapes[oldTransposeShape[0]]);
    validShape.push_back(oldValidShapes[oldTransposeShape[0]]);
    if (midNum > 0) {
        inputShape.push_back(mid);
        vecTileShape.push_back(midTileShape);
        validShape.push_back(midValidShape);
        transposeShape[1] -= (midNum - 1);
    }
    inputShape.push_back(operand.GetShape()[oldTransposeShape[1]]);
    vecTileShape.push_back(oldVecTileShapes[oldTransposeShape[1]]);
    validShape.push_back(oldValidShapes[oldTransposeShape[1]]);
    if (afterNum > 0) {
        inputShape.push_back(after);
        vecTileShape.push_back(afterTileShape);
        validShape.push_back(afterValidShape);
    }
    return true;
}

Tensor Transpose(const Tensor &self, std::vector<int> perm) {
    DECLARE_TRACER();
    ASSERT(perm.size() == 2) << "Transpose dim num should be 2."; // perm should be 2 dims
    int shapeSize = self.GetShape().size();
    if (perm[0] < 0) {
        perm[0] += shapeSize;
    }
    if (perm[1] < 0) {
        perm[1] += shapeSize;
    }
    ASSERT(perm[0] < shapeSize && perm[0] >= 0) << "Transpose dim 0 is invalid.";
    ASSERT(perm[1] < shapeSize && perm[1] >= 0) << "Transpose dim 1 is invalid.";

    std::sort(perm.begin(), perm.end());
    if ((self.GetShape()[perm[0]] == 1 && self.GetShape()[perm[1]] == 1) || perm[0] == perm[1]) {
        return self;
    }
    auto oldVecTileShapes = TileShape::Current().GetVecTile();
    ASSERT((int)oldVecTileShapes.size() == shapeSize) << "TileShape dim num should same to input.";
    auto oldValidShapes = self.GetStorage()->GetDynValidShape();
    if (oldValidShapes.empty()) {
        oldValidShapes = SymbolicScalar::FromConcrete(self.GetShape());
    }
    ASSERT((int)oldValidShapes.size() == shapeSize) << "ValidShape dim num should same to input.";

    std::vector<int64_t> newInputShape;
    std::vector<int64_t> newVecTileShape;
    std::vector<int> newTransposeShape = perm;
    std::vector<SymbolicScalar> newValidShape = oldValidShapes;
    std::vector<int64_t> resultShape(self.GetShape());
    std::swap(resultShape[perm[0]], resultShape[perm[1]]);
    if (!MergeTransposeAxis(self, newInputShape, newVecTileShape, newValidShape, newTransposeShape)) {
        Tensor result(self.GetStorage()->Datatype(), resultShape);
        CALL(
            InnerTranspose, *Program::GetInstance().GetCurrentFunction(), self.GetStorage(), result.GetStorage(), perm);
        return result;
    }
    std::swap(oldValidShapes[perm[0]], oldValidShapes[perm[1]]);

    auto tmpInputTensor = Reshape(self, newInputShape, newValidShape);
    TileShape::Current().SetVecTile(newVecTileShape);
    auto tmpOutputTensor = Transpose(tmpInputTensor, newTransposeShape);
    TileShape::Current().SetVecTile(oldVecTileShapes);
    return Reshape(tmpOutputTensor, resultShape, oldValidShapes);
}

void TiledFull(Function &function, const TileShape &tileShape, size_t cur, const Element &value,
    const SymbolicScalar &dynValue, std::vector<int64_t> &shape, const std::vector<SymbolicScalar> &validShape,
    const LogicalTensorPtr &results, TileInfo &resultTileInfo) {
    if (cur == results->shape.size()) {
        auto resultTile = results->View(function, resultTileInfo.shape, resultTileInfo.offset);
        auto lastIndex = resultTile->shape.size() - 1;
        auto bytes = BytesOf(resultTile->Datatype());
        auto paddingIter = BLOCK_PADDING_DIM.find(bytes);
        int paddingValue = 1;
        if (paddingIter != BLOCK_PADDING_DIM.end()) {
            paddingValue = paddingIter->second;
        }
        auto tempTileShape = resultTile->shape;
        tempTileShape[lastIndex] = (tempTileShape[lastIndex] + paddingValue - 1) / paddingValue * paddingValue;
        TileShape::Current().SetVecTile(tempTileShape);
        auto &op = function.AddOperation("TILE_VEC_DUP", {}, {resultTile});

        op.SetAttribute(OpAttributeKey::scalar, value);
        if (dynValue.IsValid()) {
            op.SetAttribute(OpAttributeKey::dynScalar, dynValue);
        }
        op.SetAttribute(OP_ATTR_PREFIX + "shape", resultTileInfo.shape);
        op.SetAttribute(OP_ATTR_PREFIX + "validShape", resultTile->GetDynValidShape());
        return;
    }

    auto &vecTile = tileShape.GetVecTile();
    for (int i = 0; i < results->shape[cur]; i += vecTile[cur]) {
        resultTileInfo.offset[cur] = i;
        resultTileInfo.shape[cur] = std::min(results->shape[cur] - i, vecTile[cur]);
        TiledFull(function, tileShape, cur + 1, value, dynValue, shape, validShape, results, resultTileInfo);
    }
}

void TiledFull(Function &function, const TileShape &tileShape, const Element &value, const SymbolicScalar &dynValue,
    std::vector<int64_t> &shape, const std::vector<SymbolicScalar> &validShape, const LogicalTensorPtr &results) {
    TileInfo resultTileInfo(results->shape.size(), results->offset.size());
    TiledFull(function, tileShape, 0, value, dynValue, shape, validShape, results, resultTileInfo);
}

Tensor TensorFullOperation(Function &function, const Element &src, const SymbolicScalar &dynValue,
    DataType dtype, const std::vector<int64_t> &dstShape, const std::vector<SymbolicScalar> &validShape) {
    auto result = std::make_shared<LogicalTensor>(function, dtype, dstShape, validShape);
    auto &op = function.AddOperation(Opcode::OP_VEC_DUP, {}, {result}); // 输入没有tensor
    op.SetAttribute(OpAttributeKey::scalar, src);
    if (dynValue.IsValid()) {
        op.SetAttribute(OpAttributeKey::dynScalar, dynValue);
    }
    op.SetAttribute(OP_ATTR_PREFIX + "shape", dstShape);
    op.SetAttribute(OP_ATTR_PREFIX + "validShape", validShape);
    function.UpdateTensorDataUsage(op);
    return result;
}

Tensor Full(
    const Element &src, DataType dtype, const std::vector<int64_t> &dstShape, std::vector<SymbolicScalar> validShape) {
    DECLARE_TRACER();
    if (validShape.empty()) {
        for (auto x : dstShape)
            validShape.emplace_back(x);
    }
    RETURN_CALL(FullOperation, *Program::GetInstance().GetCurrentFunction(), src, SymbolicScalar(), dtype,
        dstShape, validShape);
}

Tensor Full(const SymbolicScalar &dynSrc, DataType dtype, const std::vector<int64_t> &dstShape,
    std::vector<SymbolicScalar> validShape) {
    DECLARE_TRACER();
    if (validShape.empty()) {
        for (auto x : dstShape)
            validShape.emplace_back(x);
    }
    RETURN_CALL(FullOperation, *Program::GetInstance().GetCurrentFunction(), Element(dtype, (int64_t)0),
        dynSrc, dtype, dstShape, validShape);
}

template <CastOpType T>
void TiledCastOperation(Function &function, const TileShape &tileShape, const int cur, Input &input,
    const LogicalTensorPtr &result, const CastMode &mode) {
    if (cur == static_cast<int>(input.tensor.GetShape().size())) {
        auto tile = input.tensor.GetStorage()->View(function, input.tileInfo.shape, input.tileInfo.offset);
        auto resultTile = result->View(function, input.tileInfo.shape, input.tileInfo.offset);
        auto &op = function.AddOperation(GetCastOpName<T>(), {tile}, {resultTile});
        op.SetAttribute(OP_ATTR_PREFIX + "mode", mode);
        return;
    }
    auto &vecTile = tileShape.GetVecTile();
    for (int i = 0; i < input.tensor.GetShape()[cur]; i += vecTile[cur]) {
        input.tileInfo.shape[cur] = std::min(input.tensor.GetShape()[cur] - i, vecTile[cur]);
        input.tileInfo.offset[cur] = i;
        TiledCastOperation<T>(function, tileShape, cur + 1, input, result, mode);
    }
}

template <CastOpType T>
void TiledCastOperation(Function &function, const TileShape &tileShape, const LogicalTensorPtr &operand,
    const LogicalTensorPtr &result, const CastMode &mode) {
    ASSERT(operand->shape.size() == operand->offset.size()) << "The shape size of operand and offset should be equal";

    TileInfo tileInfo(result->shape.size(), result->offset.size());
    auto input = Input{operand, tileInfo};
    TiledCastOperation<T>(function, tileShape, 0, input, result, mode);
}

Tensor Cast(const Tensor &self, DataType dstDataType, CastMode mode) {
    DECLARE_TRACER();
    ASSERT(self.GetShape().size() == self.GetStorage()->offset.size()) << "The shape size of self and offset should be equal";
    // Cast to same dType with no mode will do nothing
    if (self.GetStorage()->tensor->datatype == dstDataType && (mode == CAST_NONE || mode == CAST_RINT)) {
        return self;
    }
    RETURN_CALL(CastOperation<CastOpType::CAST>, *Program::GetInstance().GetCurrentFunction(), self.GetStorage(),
        dstDataType, mode);
}

void TensorInnerConcatNew(Function &function, const LogicalTensorPtr &operand, const LogicalTensorPtr &result) {
    function.AddOperation(Opcode::OP_REGISTER_COPY, {operand}, {result});
}

void InnerConcatNew(Function &function, const LogicalTensorPtr &operand, const LogicalTensorPtr &result) {
    CALL(InnerConcatNew, function, operand, result);
}

void CheckCat(const std::vector<Tensor> &tensors, int axis) {
    auto shape = tensors[0].GetShape();
    auto format = tensors[0].Format();
    auto shapeSize = shape.size();
    auto dataType = tensors[0].GetDataType();

    ASSERT(SHAPE_DIM2 <= shapeSize && shapeSize <= SHAPE_DIM4) << "The support dimension must be 2 to 4 dimensions";
    std::vector<DataType> CAT_SUPPORT_DATATYPES = {DataType::DT_FP32, DataType::DT_FP16, DataType::DT_INT32,
        DataType::DT_INT16, DataType::DT_INT8, DataType::DT_BF16};
    ASSERT(
        std::find(CAT_SUPPORT_DATATYPES.begin(), CAT_SUPPORT_DATATYPES.end(), dataType) != CAT_SUPPORT_DATATYPES.end()) << "The datatype is not within the supported range";

    if (axis < 0) {
        axis = shapeSize + axis;
    }
    ASSERT(static_cast<size_t>(axis) < shapeSize) << "The axis should less than shape size";
    for (auto tensor : tensors) {
        ASSERT(tensor.GetShape().size() == shapeSize) << "The shape size of all tensors should be equal";
        ASSERT(tensor.Format() == format) << "The format of all tensors should be equal";
        ASSERT(tensor.GetStorage() != nullptr) << "Each input must not be a null pointer";
        ASSERT(tensor.GetDataType() == dataType) << "The dataType of all tensors should be equal";
    }

    for (auto tensor : tensors) {
        for (int i = 0; static_cast<size_t>(i) < shapeSize; ++i) {
            if (i == axis) {
                continue;
            }
            ASSERT(shape[i] == tensor.GetShape()[i]) << "The shape of all tensors should be equal except at axis";
        }
    }
}

Tensor Cat(const std::vector<Tensor> &tensors, int axis) {
    DECLARE_TRACER();
    CheckCat(tensors, axis);

    auto resultShape = tensors[0].GetShape();
    auto shapeSize = resultShape.size();
    if (axis < 0) {
        axis = shapeSize + axis;
    }
    int axisSize = 0;
    for (auto tensor : tensors) {
        axisSize += tensor.GetShape()[axis];
    }
    resultShape[axis] = axisSize;

    auto format = tensors[0].Format();
    Tensor result(tensors[0].GetDataType(), resultShape, "", format);
    Tensor tmp(tensors[0].GetDataType(), resultShape, "", format);
    auto &function = *Program::GetInstance().GetCurrentFunction();
    std::vector<int64_t> offset(shapeSize, 0);
    for (auto tensor : tensors) {
        auto tmpView = tmp.GetStorage()->View(function, tensor.GetShape(), offset);
        InnerConcatNew(*Program::GetInstance().GetCurrentFunction(), tensor.GetStorage(), tmpView);
        offset[axis] += tensor.GetShape()[axis];
    }
    auto &op = function.AddOperation(Opcode::OP_ASSEMBLE, {tmp.GetStorage()}, {result.GetStorage()});
    op.SetOpAttribute(std::make_shared<AssembleOpAttribute>(std::vector<int64_t>(shapeSize, 0)));

    return result;
}

void MoveOutOperationTileFunc(Function &function, const TileShape &tileShape,
    const std::vector<LogicalTensorPtr> &iOperand, const std::vector<LogicalTensorPtr> &oOperand, const Operation &op) {
    auto shape = op.GetVectorIntAttribute<int>(OP_ATTR_PREFIX + "shape");
    TiledInnerTranspose<TransposeOpType::TRANSPOSE_MOVEOUT>(function, tileShape, iOperand[0], oOperand[0], shape);
}

void MoveInOperationTileFunc(Function &function, const TileShape &tileShape,
    const std::vector<LogicalTensorPtr> &iOperand, const std::vector<LogicalTensorPtr> &oOperand, const Operation &op) {
    auto shape = op.GetVectorIntAttribute<int>(OP_ATTR_PREFIX + "shape");
    TiledInnerTranspose<TransposeOpType::TRANSPOSE_MOVEIN>(function, tileShape, iOperand[0], oOperand[0], shape);
}

void VnchwconvOperationTileFunc(Function &function, const TileShape &tileShape,
    const std::vector<LogicalTensorPtr> &iOperand, const std::vector<LogicalTensorPtr> &oOperand, const Operation &op) {
    auto shape = op.GetVectorIntAttribute<int>(OP_ATTR_PREFIX + "shape");
    TiledInnerTranspose<TransposeOpType::TRANSPOSE_VNCHWCONV>(function, tileShape, iOperand[0], oOperand[0], shape);
}

void ExpandOperationTileFunc(Function &function, const TileShape &tileShape,
    const std::vector<LogicalTensorPtr> &iOperand, const std::vector<LogicalTensorPtr> &oOperand, const Operation &op) {
    UnaryOperationOperandCheck(iOperand, oOperand);
    std::vector<SymbolicScalar> validShape;
    op.GetAttr(OP_ATTR_PREFIX + "validShape", validShape);
    TiledExpand(function, tileShape, iOperand[0], oOperand[0], validShape);
}

inline void CastOperationOperandCheck(
    const std::vector<LogicalTensorPtr> &iOperand, const std::vector<LogicalTensorPtr> &oOperand) {
    ASSERT(iOperand.size() == 1) << "The input operand size should be 1";
    ASSERT(oOperand.size() == 1) << "The output operand size should be 1";
}

void CastOperationTileFunc(Function &function, const TileShape &tileShape,
    const std::vector<LogicalTensorPtr> &iOperand, const std::vector<LogicalTensorPtr> &oOperand, const Operation &op) {
    CastOperationOperandCheck(iOperand, oOperand);
    auto mode = op.GetCastModeAttribute(OP_ATTR_PREFIX + "mode");
    TiledCastOperation<CastOpType::CAST>(function, tileShape, iOperand[0], oOperand[0], mode);
}

void FullOperationTileFunc(Function &function, const TileShape &tileShape,
    [[maybe_unused]] const std::vector<LogicalTensorPtr> &iOperand, const std::vector<LogicalTensorPtr> &oOperand,
    const Operation &op) {
    Element scalar = op.GetElementAttribute(OpAttributeKey::scalar);
    SymbolicScalar dynScalar;
    if (op.HasAttr(OpAttributeKey::dynScalar)) {
        dynScalar = op.GetSymbolicScalarAttribute(OpAttributeKey::dynScalar);
    }
    std::vector<int64_t> shape = op.GetVectorIntAttribute(OP_ATTR_PREFIX + "shape");
    std::vector<SymbolicScalar> validShape;
    op.GetAttr(OP_ATTR_PREFIX + "validShape", validShape);
    TiledFull(function, tileShape, scalar, dynScalar, shape, validShape, oOperand[0]);
}

REGISTER_OPERATION_TILED_FUNC(OP_TRANSPOSE_MOVEOUT, Opcode::OP_TRANSPOSE_MOVEOUT, MoveOutOperationTileFunc);
REGISTER_OPERATION_TILED_FUNC(OP_TRANSPOSE_MOVEIN, Opcode::OP_TRANSPOSE_MOVEIN, MoveInOperationTileFunc);
REGISTER_OPERATION_TILED_FUNC(OP_TRANSPOSE_VNCHWCONV, Opcode::OP_TRANSPOSE_VNCHWCONV, VnchwconvOperationTileFunc);
REGISTER_OPERATION_TILED_FUNC(OP_EXPAND, Opcode::OP_EXPAND, ExpandOperationTileFunc);
REGISTER_OPERATION_TILED_FUNC(OP_CAST, Opcode::OP_CAST, CastOperationTileFunc);
REGISTER_OPERATION_TILED_FUNC(OP_VEC_DUP, Opcode::OP_VEC_DUP, FullOperationTileFunc);

} // namespace npu::tile_fwk
