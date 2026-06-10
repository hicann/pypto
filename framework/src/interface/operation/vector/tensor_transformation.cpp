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
#include <sstream>
#include <string>
#include "tensor_transformation.h"
#include "interface/utils/operator_tracer.h"
#include "tilefwk/error_code.h"
#include "tilefwk/platform.h"

namespace npu::tile_fwk {

struct ExpandInfo {
    const std::shared_ptr<LogicalTensor>& srcTensor;
    const std::shared_ptr<LogicalTensor>& result;
    std::vector<int64_t>& viewShape;
    std::vector<int64_t>& offset;
    const std::vector<int> expandDims;
    ExpandInfo(
        const std::shared_ptr<LogicalTensor>& srcTensor0, const std::shared_ptr<LogicalTensor>& result0,
        std::vector<int64_t>& viewShape0, std::vector<int64_t>& offset0, const std::vector<int> expandDims0)
        : srcTensor(srcTensor0), result(result0), viewShape(viewShape0), offset(offset0), expandDims(expandDims0)
    {}
};

void CheckExpandTensorValid(const LogicalTensorPtr& operand, const LogicalTensorPtr& result)
{
    const auto& operand_shape = operand->shape;
    const auto& result_shape = result->shape;

    if (operand_shape.size() != result_shape.size()) {
        std::ostringstream oss;
        oss << "The number of dimensions must match! "
            << "Operand shape: " << operand_shape.size() << "D (" << operand_shape << ") "
            << "Result shape: " << result_shape.size() << "D (" << result_shape << ")";
        ASSERT(VectorErrorCode::ERR_PARAM_INVALID, false) << oss.str();
    }

    for (size_t i = 0; i < result_shape.size(); ++i) {
        if (operand_shape[i] != result_shape[i] && operand_shape[i] != 1) {
            std::ostringstream oss;
            oss << "The size of tensor a (" << operand_shape[i] << ") must match the size of tensor b ("
                << result_shape[i] << ") at non-singleton dimension " << i << ". "
                << "Operand shape: (" << operand_shape << ") "
                << "Result shape: (" << result_shape << ")";
            ASSERT(VectorErrorCode::ERR_PARAM_INVALID, false) << oss.str();
        }
    }
}

void ExpandTile(Function& function, const struct ExpandInfo& expandInfo)
{
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
    auto& newOp = function.AddOperation("TILE_EXPAND", {srcTile}, {resultTile});
    newOp.SetAttribute(OpAttributeKey::expandDims, expandInfo.expandDims);
    newOp.SetAttribute(OP_ATTR_PREFIX + "validShape", resultTile->GetDynValidShape());
}

void ExpandTile(
    Function& function, const TileShape& tileShape, int dimIdx, const struct ExpandInfo& expandInfo,
    std::vector<SymbolicScalar> validShape)
{
    if (static_cast<size_t>(dimIdx) == expandInfo.result->shape.size()) {
        ExpandTile(function, expandInfo);
        return;
    }
    auto& vecTile = tileShape.GetVecTile();
    for (int i = 0; i < expandInfo.result->shape[dimIdx]; i += vecTile[dimIdx]) {
        expandInfo.offset[dimIdx] = i;
        expandInfo.viewShape[dimIdx] =
            std::min(expandInfo.result->shape[dimIdx] - i, static_cast<int64_t>(vecTile[dimIdx]));
        ExpandTile(function, tileShape, dimIdx + 1, expandInfo, validShape);
    }
}

void Expand(
    Function& function, const TileShape& tileShape, const LogicalTensorPtr& operand,
    const std::vector<LogicalTensorPtr>& other, const LogicalTensorPtr& result)
{
    CheckExpandTensorValid(operand, result);
    ASSERT(VectorErrorCode::ERR_PARAM_INVALID, function.GetGraphType() == GraphType::TILE_GRAPH)
        << "The GetGraphType of function is incorrect";
    std::vector<int64_t> offset(result->shape.size(), 0);
    std::vector<int64_t> viewShape(result->shape.size(), 1);
    std::vector<SymbolicScalar> outValidShape;
    std::vector<int> expandDims;
    for (size_t i = 0; i < result->shape.size(); ++i) {
        if (operand->shape[i] != result->shape[i]) {
            expandDims.push_back(i);
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
    struct ExpandInfo expandInfo(operand, result, viewShape, offset, expandDims);
    ExpandTile(function, tileShape, 0, expandInfo, outValidShape);
}

void ExpandWithResultValidShape(
    Function& function, const TileShape& tileShape, const LogicalTensorPtr& operand, const LogicalTensorPtr& result,
    const std::vector<SymbolicScalar> resultValidShape)
{
    CheckExpandTensorValid(operand, result);
    ASSERT(VectorErrorCode::ERR_PARAM_INVALID, function.GetGraphType() == GraphType::TILE_GRAPH)
        << "The GetGraphType of function is incorrect";
    std::vector<int64_t> offset(result->shape.size(), 0);
    std::vector<int64_t> viewShape(result->shape.size(), 1);
    std::vector<int> expandDims;
    for (size_t i = 0; i < result->shape.size(); ++i) {
        if (operand->shape[i] != result->shape[i]) {
            expandDims.push_back(i);
        }
    }
    result->UpdateDynValidShape(resultValidShape);
    struct ExpandInfo expandInfo(operand, result, viewShape, offset, expandDims);
    ExpandTile(function, tileShape, 0, expandInfo, resultValidShape);
}

void TiledExpand(
    Function& function, const TileShape& tileShape, const LogicalTensorPtr& operand, const LogicalTensorPtr& result,
    const std::vector<SymbolicScalar>& validShape)
{
    CheckExpandTensorValid(operand, result);
    ASSERT(VectorErrorCode::ERR_PARAM_INVALID, function.GetGraphType() == GraphType::TILE_GRAPH)
        << "The GetGraphType of function is incorrect";

    std::vector<int64_t> offset(result->shape.size(), 0);
    std::vector<int64_t> viewShape(result->shape.size(), 1);
    std::vector<int> expandDims;
    for (size_t i = 0; i < result->shape.size(); ++i) {
        if (operand->shape[i] != result->shape[i]) {
            expandDims.push_back(i);
        }
    }
    result->UpdateDynValidShape(validShape);
    struct ExpandInfo expandInfo(operand, result, viewShape, offset, expandDims);
    ExpandTile(function, tileShape, 0, expandInfo, validShape);
}

Tensor TensorExpandOperation(
    Function& function, const LogicalTensorPtr& operand, const std::vector<int64_t>& dstShape,
    const std::vector<SymbolicScalar>& validShape)
{
    auto result = std::make_shared<LogicalTensor>(function, operand->Datatype(), dstShape, validShape);
    auto& op = function.AddOperation(Opcode::OP_EXPAND, {operand}, {result});

    op.SetAttribute(OP_ATTR_PREFIX + "shape", dstShape);
    op.SetAttribute(OP_ATTR_PREFIX + "validShape", validShape);
    function.UpdateTensorDataUsage(op);
    return result;
}

Tensor TensorJustNeedCopyOperation(
    Function& function, const LogicalTensorPtr& operand, const std::vector<int64_t>& dstShape,
    const std::vector<SymbolicScalar>& validShape)
{
    auto result = std::make_shared<LogicalTensor>(function, operand->Datatype(), dstShape, validShape);
    function.AddOperation(Opcode::OP_REGISTER_COPY, {operand}, {result});
    return result;
}

Tensor Expand(const Tensor& self, const std::vector<int64_t>& dstShape, std::vector<SymbolicScalar> validShape)
{
    DECLARE_TRACER();

    std::unordered_set<DataType> supportedTypes = {DT_BF16,  DT_FP32,  DT_FP16,   DT_INT8,   DT_INT16,
                                                   DT_INT32, DT_UINT8, DT_UINT16, DT_UINT32, DT_BOOL};
    CheckTensorDataType(self.GetStorage(), supportedTypes, "EXPAND");
    CheckTensorDimRange(self.GetStorage(), 1, 4, "EXPAND");
    CheckTensorShapeSize(self.GetStorage(), "EXPAND");
    CheckDstShapeSize(dstShape, "EXPAND");
    ASSERT(VectorErrorCode::ERR_PARAM_INVALID, self.GetShape().size() == dstShape.size())
        << "The shape size of self and dst should be equal";
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
        RETURN_CALL(
            JustNeedCopyOperation, *Program::GetInstance().GetCurrentFunction(), self.GetStorage(), dstShape,
            validShape);
    }
}

enum class TransposeOpType {
    TRANSPOSE_MOVEIN,
    TRANSPOSE_MOVEOUT,
    TRANSPOSE_VNCHWCONV,
};

void CheckTransposeAxisCombination(int shapeSize, const std::vector<int>& perm)
{
    if (shapeSize == 4) {
        std::vector<std::pair<int, int>> supported4D = {{0, 2}, {1, 2}, {1, 3}, {2, 3}};
        bool isSupported = false;
        for (const auto& axisPair : supported4D) {
            if (perm[0] == axisPair.first && perm[1] == axisPair.second) {
                isSupported = true;
                break;
            }
        }
        ASSERT(VectorErrorCode::ERR_PARAM_INVALID, isSupported)
            << "4D tensor transpose only supports: (0,2), (1,2), (1,3), (2,3). "
            << "Current dim0=" << perm[0] << ", dim1=" << perm[1] << " is not supported.";
    }
    
    if (shapeSize == 5) {
        ASSERT(VectorErrorCode::ERR_PARAM_INVALID, perm[0] == 3 && perm[1] == 4)
            << "5D tensor transpose only supports: (3,4). "
            << "Current dim0=" << perm[0] << ", dim1=" << perm[1] << " is not supported.";
    }
}

template <TransposeOpType T>
Opcode GetTransposeOpName()
{
#define CASE(X)              \
    case TransposeOpType::X: \
        return Opcode::OP_##X
    switch (T) {
        CASE(TRANSPOSE_MOVEOUT);
        CASE(TRANSPOSE_MOVEIN);
        CASE(TRANSPOSE_VNCHWCONV);
        default:
            ASSERT(VectorErrorCode::ERR_PARAM_INVALID, false) << "unknown transpose op type";
    }
#undef CASE
}

inline void UnalignPadTmpBufTile(std::vector<int64_t>& shape, int blockElem, DataType dtype)
{
    // tmpbuf按16 8对齐
    auto size = shape.size();
    if (size >= NUM_VALUE_2) {
        int64_t alignSize = VNCHWCONV_REPEAT;
        if (dtype == DT_INT8 || dtype == DT_UINT8) {
            alignSize = BLOCK_SIZE; // int8 特判 按 32 对齐
        }
        shape[size - NUM_VALUE_2] = AlignUp(shape[size - NUM_VALUE_2], alignSize);
        shape[size - 1] = AlignUp(shape[size - 1], blockElem);
    }
}

template <TransposeOpType T>
void TiledInnerTranspose(
    Function& function, const TileShape& tileShape, const int cur, Input& input, const LogicalTensorPtr& result,
    const std::vector<int>& shape)
{
    int shapeSize = input.tensor.GetShape().size();
    if (cur == shapeSize) {
        auto tile = input.tensor.GetStorage()->View(function, input.tileInfo.shape, input.tileInfo.offset);
        std::vector<int64_t> resultTileShape(input.tileInfo.shape);
        std::swap(resultTileShape[shape[0]], resultTileShape[shape[1]]);
        std::vector<int64_t> resultTileOfs(input.tileInfo.offset);
        std::swap(resultTileOfs[shape[0]], resultTileOfs[shape[1]]);
        auto resultTile = result->View(function, resultTileShape, resultTileOfs);
        if (T == TransposeOpType::TRANSPOSE_MOVEOUT || T == TransposeOpType::TRANSPOSE_MOVEIN) {
            auto& op = function.AddOperation(GetTransposeOpName<T>(), {tile}, {resultTile});
            op.SetAttribute(OP_ATTR_PREFIX + "shape", shape);
        } else {
            std::vector<int64_t> tmpShape(input.tileInfo.shape);
            int64_t blockElem = BLOCK_SIZE / static_cast<int>(BytesOf(tile->Datatype()));
            UnalignPadTmpBufTile(tmpShape, blockElem, tile->Datatype());
            auto tempTensor = std::make_shared<LogicalTensor>(function, tile->Datatype(), tmpShape);
            tempTensor->dynValidShape_ = SymbolicScalar::FromConcrete(tmpShape);
            auto& op = function.AddOperation(GetTransposeOpName<T>(), {tile}, {resultTile, tempTensor});
            op.SetAttribute(OP_ATTR_PREFIX + "shape", shape);
        }
        return;
    }
    auto& vecTile = tileShape.GetVecTile();
    for (int i = 0; i < input.tensor.GetShape()[cur]; i += vecTile[cur]) {
        input.tileInfo.shape[cur] = std::min(input.tensor.GetShape()[cur] - i, vecTile[cur]);
        input.tileInfo.offset[cur] = i;
        TiledInnerTranspose<T>(function, tileShape, cur + 1, input, result, shape);
    }
}

template <TransposeOpType T>
void TiledInnerTranspose(
    Function& function, const TileShape& tileShape, const LogicalTensorPtr& operand, const LogicalTensorPtr& result,
    const std::vector<int>& shape)
{
    TileInfo tileInfo(result->shape.size(), result->offset.size());
    auto input = Input{operand, tileInfo};
    TiledInnerTranspose<T>(function, tileShape, 0, input, result, shape);
}

void TensorInnerTranspose(
    Function& function, const LogicalTensorPtr& self, const LogicalTensorPtr& result, std::vector<int> perm)
{
    if (perm[0] != (int)self->shape.size() - 1 && perm[1] != (int)self->shape.size() - 1) {
        auto& operation = function.AddOperation(Opcode::OP_TRANSPOSE_MOVEOUT, {self}, {result});
        operation.SetAttribute(OP_ATTR_PREFIX + "shape", perm);
        return;
    }

    if (perm[0] == (int)self->shape.size() - 2 && // last 2 dims transpose
        perm[1] == (int)self->shape.size() - 1) {
        auto& operation = function.AddOperation(Opcode::OP_TRANSPOSE_VNCHWCONV, {self}, {result});
        operation.SetAttribute(OP_ATTR_PREFIX + "shape", perm);
        return;
    }

    ASSERT(
        VectorErrorCode::ERR_PARAM_INVALID,
        self->shape.size() == 3 || self->shape.size() == 4) // input should be 3 or 4 dims
        << "Transpose shape should be [A1,T1,A2,T2] or [T1,A2,T2]";

    // [A1,T1,A2,T2] to [A1,A2,T1,T2] or [T1,A2,T2] to [A2,T1,T2]
    auto oldVecTileShapes = TileShape::Current().GetVecTile();
    auto newVecTileShape = oldVecTileShapes;
    std::vector<int64_t> tmpShape(self->shape);
    int dim1 = (tmpShape.size() == 3) ? 0 : 1; // if input is 3 dims, dim1 = 0, otherwise dim1 = 1
    int dim2 = (tmpShape.size() == 3) ? 1 : 2; // if input is 3 dims, dim2 = 1, otherwise dim2 = 2
    std::swap(tmpShape[dim1], tmpShape[dim2]);
    std::swap(newVecTileShape[dim1], newVecTileShape[dim2]);
    auto outValidShapes = self->GetDynValidShape();
    std::swap(outValidShapes[dim1], outValidShapes[dim2]);
    auto moveInResult = std::make_shared<LogicalTensor>(function, self->Datatype(), tmpShape, outValidShapes);
    auto& inOp = function.AddOperation(Opcode::OP_TRANSPOSE_MOVEIN, {self}, {moveInResult});
    inOp.SetAttribute(OP_ATTR_PREFIX + "shape", std::vector<int>{dim1, dim2});
    TileShape::Current().SetVecTile(newVecTileShape);

    // [A1,A2,T1,T2] to [A1,A2,T2,T1] or [A2,T1,T2] to [A2,T2,T1]
    tmpShape = moveInResult->shape;
    dim1 = (tmpShape.size() == 3) ? 1 : 2; // if input is 3 dims, dim1 = 1, otherwise dim1 = 2
    dim2 = (tmpShape.size() == 3) ? 2 : 3; // if input is 3 dims, dim2 = 2, otherwise dim2 = 3
    std::swap(tmpShape[dim1], tmpShape[dim2]);
    std::swap(newVecTileShape[dim1], newVecTileShape[dim2]);
    std::swap(outValidShapes[dim1], outValidShapes[dim2]);
    auto vnchwconvResult = std::make_shared<LogicalTensor>(function, self->Datatype(), tmpShape, outValidShapes);
    auto& convOp = function.AddOperation(Opcode::OP_TRANSPOSE_VNCHWCONV, {moveInResult}, {vnchwconvResult});
    convOp.SetAttribute(OP_ATTR_PREFIX + "shape", std::vector<int>{dim1, dim2});
    TileShape::Current().SetVecTile(newVecTileShape);

    // [A1,A2,T2,T1] to [A1,T2,A2,T1] or [A2,T2,T1] to [T2,A2,T1]
    tmpShape = vnchwconvResult->shape;
    dim1 = (tmpShape.size() == 3) ? 0 : 1; // if input is 3 dims, dim1 = 0, otherwise dim1 = 1
    dim2 = (tmpShape.size() == 3) ? 1 : 2; // if input is 3 dims, dim2 = 1, otherwise dim2 = 2
    std::swap(tmpShape[dim1], tmpShape[dim2]);
    auto& outOp = function.AddOperation(Opcode::OP_TRANSPOSE_MOVEOUT, {vnchwconvResult}, {result});
    outOp.SetAttribute(OP_ATTR_PREFIX + "shape", std::vector<int>{dim1, dim2});
    TileShape::Current().SetVecTile(oldVecTileShapes);
}

bool MergeTransposeAxis(
    const Tensor& operand, std::vector<int64_t>& inputShape, std::vector<int64_t>& vecTileShape,
    std::vector<SymbolicScalar>& validShape, std::vector<int>& transposeShape)
{
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
    if (operand.GetShape().size() <= 5 &&                             // tileop支持5维
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

Tensor Transpose(const Tensor& self, std::vector<int> perm)
{
    DECLARE_TRACER();
    std::unordered_set<DataType> supportedTypes = {DT_FP16,   DT_BF16, DT_UINT8, DT_INT8,  DT_INT16,
                                                   DT_UINT16, DT_FP32, DT_INT32, DT_UINT32};
    CheckTensorDataType(self.GetStorage(), supportedTypes, "TRANSPOSE");
    CheckTensorDimRange(self.GetStorage(), 1, 5, "TRANSPOSE");
    CheckTensorShapeSize(self.GetStorage(), "TRANSPOSE");
    ASSERT(VectorErrorCode::ERR_PARAM_INVALID, perm.size() == 2)
        << "Transpose dim num should be 2."; // perm should be 2 dims
    int shapeSize = self.GetShape().size();
    if (perm[0] < 0) {
        perm[0] += shapeSize;
    }
    if (perm[1] < 0) {
        perm[1] += shapeSize;
    }
    ASSERT(VectorErrorCode::ERR_PARAM_INVALID, perm[0] < shapeSize && perm[0] >= 0) << "Transpose dim 0 is invalid.";
    ASSERT(VectorErrorCode::ERR_PARAM_INVALID, perm[1] < shapeSize && perm[1] >= 0) << "Transpose dim 1 is invalid.";

    std::sort(perm.begin(), perm.end());
    CheckTransposeAxisCombination(shapeSize, perm);
    if ((self.GetShape()[perm[0]] == 1 && self.GetShape()[perm[1]] == 1) || perm[0] == perm[1]) {
        return self;
    }
    auto oldVecTileShapes = TileShape::Current().GetVecTile();
    ASSERT(VectorErrorCode::ERR_PARAM_INVALID, (int)oldVecTileShapes.size() == shapeSize)
        << "TileShape dim num should same to input.";
    auto oldValidShapes = self.GetStorage()->GetDynValidShape();
    if (oldValidShapes.empty()) {
        oldValidShapes = SymbolicScalar::FromConcrete(self.GetShape());
    }
    ASSERT(VectorErrorCode::ERR_PARAM_INVALID, (int)oldValidShapes.size() == shapeSize)
        << "ValidShape dim num should same to input.";

    std::vector<int64_t> newInputShape;
    std::vector<int64_t> newVecTileShape;
    std::vector<int> newTransposeShape = perm;
    std::vector<SymbolicScalar> newValidShape = oldValidShapes;
    std::swap(oldValidShapes[perm[0]], oldValidShapes[perm[1]]);
    std::vector<int64_t> resultShape(self.GetShape());
    std::swap(resultShape[perm[0]], resultShape[perm[1]]);
    if (!MergeTransposeAxis(self, newInputShape, newVecTileShape, newValidShape, newTransposeShape)) {
        Tensor result(self.GetStorage()->Datatype(), resultShape);
        result.GetStorage()->UpdateDynValidShape(oldValidShapes);
        CALL(
            InnerTranspose, *Program::GetInstance().GetCurrentFunction(), self.GetStorage(), result.GetStorage(), perm);
        return result;
    }

    auto tmpInputTensor = Reshape(self, newInputShape, newValidShape);
    TileShape::Current().SetVecTile(newVecTileShape);
    auto tmpOutputTensor = Transpose(tmpInputTensor, newTransposeShape);
    TileShape::Current().SetVecTile(oldVecTileShapes);
    return Reshape(tmpOutputTensor, resultShape, oldValidShapes);
}

struct TransDataTileInfoPara {
    TileInfo inputTileInfo;
    TileInfo dstTileInfo;
};

struct TransDataPara {
    const LogicalTensorPtr& input;
    const LogicalTensorPtr& dstTensor;
    const std::vector<SymbolicScalar> tileParams;
    const int group;
    int groupIdx;
};

std::shared_ptr<LogicalTensor> transDataPadNC1HWC0(
    Function& function, const std::shared_ptr<LogicalTensor>& inputTile, int64_t C0)
{
    auto inputShape = inputTile->GetShape();
    int64_t N = inputShape[0];
    int64_t C = inputShape[1];
    ASSERT(VectorErrorCode::ERR_PARAM_INVALID, C0 > 0) << "The C0 is not valid !";
    int64_t C1 = (C + C0 - 1) / C0;
    int64_t padC = C1 * C0 - C;
    int64_t H = inputShape[2];
    int64_t W = inputShape[3];

    if (!padC) {
        return inputTile;
    }

    Shape resShape = Shape{N, C1 * C0, H, W};
    auto resValidShape = inputTile->dynValidShape_;
    auto validShapeC = resValidShape[1];
    auto validShapeC1 = (validShapeC + C0 - 1) / C0;
    resValidShape[1] = validShapeC1 * C0;
    auto resTile = std::make_shared<LogicalTensor>(function, inputTile->Datatype(), resShape, resValidShape);
    std::shared_ptr<LogicalTensor> tmpResTile = resTile->View(function, inputTile->GetShape(), {0, 0, 0, 0});
    [[maybe_unused]] auto& copyOp1 = function.AddOperation(Opcode::OP_REGISTER_COPY, {inputTile}, {tmpResTile});
    Shape resultRemainShape = {N, C1 * C0 - C, H, W};
    std::shared_ptr<LogicalTensor> resRemainTile = resTile->View(function, resultRemainShape, {0, C, 0, 0});
    auto padTile = std::make_shared<LogicalTensor>(
        function, inputTile->Datatype(), resultRemainShape, SymbolicScalar::FromConcrete(resultRemainShape));
    auto& vecDupOp = function.AddOperation(Opcode::OP_VEC_DUP, {}, {padTile});
    vecDupOp.SetAttribute(OpAttributeKey::scalar, Element(inputTile->Datatype(), 0));
    vecDupOp.SetAttribute(OP_ATTR_PREFIX + "shape", resultRemainShape);
    vecDupOp.SetAttribute(OP_ATTR_PREFIX + "validShape", SymbolicScalar::FromConcrete(resultRemainShape));
    [[maybe_unused]] auto& copyOp2 = function.AddOperation(Opcode::OP_REGISTER_COPY, {padTile}, {resRemainTile});
    return resTile;
}

std::shared_ptr<LogicalTensor> transDataPadFractalZ(
    Function& function, const std::shared_ptr<LogicalTensor>& inputTile, int64_t C0)
{
    auto inputShape = inputTile->GetShape();
    int64_t N = inputShape[0];
    int64_t C = inputShape[1];
    ASSERT(VectorErrorCode::ERR_PARAM_INVALID, C0 > 0) << "The C0 is not valid !";
    int64_t C1 = (C + C0 - 1) / C0;
    int64_t padC = C1 * C0 - C;
    int64_t H = inputShape[2];
    int64_t W = inputShape[3];
    int64_t N0 = 16;
    int64_t N1 = (N + N0 - 1) / N0;
    int64_t padN = N1 * N0 - N;

    if ((!padC) && (!padN)) {
        return inputTile;
    }

    Shape resShape = Shape{N1 * N0, C1 * C0, H, W};
    auto resValidShape = inputTile->dynValidShape_;
    auto validShapeN = resValidShape[0];
    auto validShapeN1 = (validShapeN + N0 - 1) / N0;
    resValidShape[0] = validShapeN1 * N0;
    auto validShapeC = resValidShape[1];
    auto validShapeC1 = (validShapeC + C0 - 1) / C0;
    resValidShape[1] = validShapeC1 * C0;
    auto resTile = std::make_shared<LogicalTensor>(function, inputTile->Datatype(), resShape, resValidShape);
    std::shared_ptr<LogicalTensor> tmpResTile = resTile->View(function, inputTile->GetShape(), {0, 0, 0, 0});
    [[maybe_unused]] auto& copyOp1 = function.AddOperation(Opcode::OP_REGISTER_COPY, {inputTile}, {tmpResTile});

    if (padC) {
        Shape resultCRemainShape = {N, C1 * C0 - C, H, W};
        std::shared_ptr<LogicalTensor> resCRemainTile = resTile->View(function, resultCRemainShape, {0, C, 0, 0});
        auto padCTile = std::make_shared<LogicalTensor>(
            function, inputTile->Datatype(), resultCRemainShape, SymbolicScalar::FromConcrete(resultCRemainShape));
        auto& vecDupOp = function.AddOperation(Opcode::OP_VEC_DUP, {}, {padCTile});
        vecDupOp.SetAttribute(OpAttributeKey::scalar, Element(inputTile->Datatype(), 0));
        vecDupOp.SetAttribute(OP_ATTR_PREFIX + "shape", resultCRemainShape);
        vecDupOp.SetAttribute(OP_ATTR_PREFIX + "validShape", SymbolicScalar::FromConcrete(resultCRemainShape));
        [[maybe_unused]] auto& copyOp2 = function.AddOperation(Opcode::OP_REGISTER_COPY, {padCTile}, {resCRemainTile});
    }
    if (padN) {
        Shape resultNRemainShape = {N1 * N0 - N, C1 * C0, H, W};
        std::shared_ptr<LogicalTensor> resNRemainTile = resTile->View(function, resultNRemainShape, {N, 0, 0, 0});
        auto padNTile = std::make_shared<LogicalTensor>(
            function, inputTile->Datatype(), resultNRemainShape, SymbolicScalar::FromConcrete(resultNRemainShape));
        auto& vecDupOp = function.AddOperation(Opcode::OP_VEC_DUP, {}, {padNTile});
        vecDupOp.SetAttribute(OpAttributeKey::scalar, Element(inputTile->Datatype(), 0));
        vecDupOp.SetAttribute(OP_ATTR_PREFIX + "shape", resultNRemainShape);
        vecDupOp.SetAttribute(OP_ATTR_PREFIX + "validShape", SymbolicScalar::FromConcrete(resultNRemainShape));
        [[maybe_unused]] auto& copyOp2 = function.AddOperation(Opcode::OP_REGISTER_COPY, {padNTile}, {resNRemainTile});
    }
    return resTile;
}

std::shared_ptr<LogicalTensor> transDataPadFractalZ3D(
    Function& function, const std::shared_ptr<LogicalTensor>& inputTile, int64_t C0)
{
    auto inputShape = inputTile->GetShape();
    int64_t N = inputShape[0];
    int64_t C = inputShape[1];
    int64_t D = inputShape[2];
    ASSERT(VectorErrorCode::ERR_PARAM_INVALID, C0 > 0) << "The C0 is not valid !";
    int64_t C1 = (C + C0 - 1) / C0;
    int64_t padC = C1 * C0 - C;
    int64_t H = inputShape[3];
    int64_t W = inputShape[4];
    int64_t N0 = 16;
    int64_t N1 = (N + N0 - 1) / N0;
    int64_t padN = N1 * N0 - N;

    if ((!padC) && (!padN)) {
        return inputTile;
    }

    Shape resShape = Shape{N1 * N0, C1 * C0, D, H, W};
    auto resValidShape = inputTile->dynValidShape_;
    auto validShapeN = resValidShape[0];
    auto validShapeN1 = (validShapeN + N0 - 1) / N0;
    resValidShape[0] = validShapeN1 * N0;
    auto validShapeC = resValidShape[1];
    auto validShapeC1 = (validShapeC + C0 - 1) / C0;
    resValidShape[1] = validShapeC1 * C0;
    auto resTile = std::make_shared<LogicalTensor>(function, inputTile->Datatype(), resShape, resValidShape);
    std::shared_ptr<LogicalTensor> tmpResTile = resTile->View(function, inputTile->GetShape(), {0, 0, 0, 0, 0});
    [[maybe_unused]] auto& copyOp1 = function.AddOperation(Opcode::OP_REGISTER_COPY, {inputTile}, {tmpResTile});

    if (padC) {
        Shape resultCRemainShape = {N, C1 * C0 - C, D, H, W};
        std::shared_ptr<LogicalTensor> resCRemainTile = resTile->View(function, resultCRemainShape, {0, C, 0, 0, 0});
        auto padCTile = std::make_shared<LogicalTensor>(
            function, inputTile->Datatype(), resultCRemainShape, SymbolicScalar::FromConcrete(resultCRemainShape));
        auto& vecDupOp = function.AddOperation(Opcode::OP_VEC_DUP, {}, {padCTile});
        vecDupOp.SetAttribute(OpAttributeKey::scalar, Element(inputTile->Datatype(), 0));
        vecDupOp.SetAttribute(OP_ATTR_PREFIX + "shape", resultCRemainShape);
        vecDupOp.SetAttribute(OP_ATTR_PREFIX + "validShape", SymbolicScalar::FromConcrete(resultCRemainShape));
        [[maybe_unused]] auto& copyOp2 = function.AddOperation(Opcode::OP_REGISTER_COPY, {padCTile}, {resCRemainTile});
    }
    if (padN) {
        Shape resultNRemainShape = {N1 * N0 - N, C1 * C0, D, H, W};
        std::shared_ptr<LogicalTensor> resNRemainTile = resTile->View(function, resultNRemainShape, {N, 0, 0, 0, 0});
        auto padNTile = std::make_shared<LogicalTensor>(
            function, inputTile->Datatype(), resultNRemainShape, SymbolicScalar::FromConcrete(resultNRemainShape));
        auto& vecDupOp = function.AddOperation(Opcode::OP_VEC_DUP, {}, {padNTile});
        vecDupOp.SetAttribute(OpAttributeKey::scalar, Element(inputTile->Datatype(), 0));
        vecDupOp.SetAttribute(OP_ATTR_PREFIX + "shape", resultNRemainShape);
        vecDupOp.SetAttribute(OP_ATTR_PREFIX + "validShape", SymbolicScalar::FromConcrete(resultNRemainShape));
        [[maybe_unused]] auto& copyOp2 = function.AddOperation(Opcode::OP_REGISTER_COPY, {padNTile}, {resNRemainTile});
    }
    return resTile;
}

std::shared_ptr<LogicalTensor> transDataPadNDC1HWC0(
    Function& function, const std::shared_ptr<LogicalTensor>& inputTile, int64_t C0)
{
    auto inputShape = inputTile->GetShape();
    int64_t N = inputShape[0];
    int64_t D = inputShape[1];
    int64_t C = inputShape[2];
    ASSERT(VectorErrorCode::ERR_PARAM_INVALID, C0 > 0) << "The C0 is not valid !";
    int64_t C1 = (C + C0 - 1) / C0;
    int64_t padC = C1 * C0 - C;
    int64_t H = inputShape[3];
    int64_t W = inputShape[4];

    if (!padC) {
        return inputTile;
    }

    Shape resShape = Shape{N, D, C1 * C0, H, W};
    auto resValidShape = inputTile->dynValidShape_;
    auto validShapeN = resValidShape[0];
    auto validShapeD = resValidShape[1];
    auto validShapeC = resValidShape[2];
    auto validShapeH = resValidShape[3];
    auto validShapeW = resValidShape[4];
    auto validShapeC1 = (validShapeC + C0 - 1) / C0;

    resValidShape = std::vector<SymbolicScalar>{validShapeN, validShapeD, validShapeC1 * C0, validShapeH, validShapeW};
    auto resTile = std::make_shared<LogicalTensor>(function, inputTile->Datatype(), resShape, resValidShape);
    std::shared_ptr<LogicalTensor> tmpResTile = resTile->View(function, inputTile->GetShape(), {0, 0, 0, 0, 0});
    [[maybe_unused]] auto& copyOp1 = function.AddOperation(Opcode::OP_REGISTER_COPY, {inputTile}, {tmpResTile});
    Shape resultRemainShape = {N, D, C1 * C0 - C, H, W};
    std::shared_ptr<LogicalTensor> resRemainTile = resTile->View(function, resultRemainShape, {0, 0, C, 0, 0});
    auto padTile = std::make_shared<LogicalTensor>(
        function, inputTile->Datatype(), resultRemainShape, SymbolicScalar::FromConcrete(resultRemainShape));
    auto& vecDupOp = function.AddOperation(Opcode::OP_VEC_DUP, {}, {padTile});
    vecDupOp.SetAttribute(OpAttributeKey::scalar, Element(inputTile->Datatype(), 0));
    vecDupOp.SetAttribute(OP_ATTR_PREFIX + "shape", resultRemainShape);
    vecDupOp.SetAttribute(OP_ATTR_PREFIX + "validShape", SymbolicScalar::FromConcrete(resultRemainShape));

    [[maybe_unused]] auto& copyOp2 = function.AddOperation(Opcode::OP_REGISTER_COPY, {padTile}, {resRemainTile});
    return resTile;
}

template <TileOpFormat T>
std::shared_ptr<LogicalTensor> transDataPad(
    Function& function, const std::shared_ptr<LogicalTensor>& inputTile, int64_t C0)
{
    switch (T) {
        case TileOpFormat::TILEOP_NC1HWC0:
            return transDataPadNC1HWC0(function, inputTile, C0);
        case TileOpFormat::TILEOP_FRACTAL_Z:
            return transDataPadFractalZ(function, inputTile, C0);
        case TileOpFormat::TILEOP_NDC1HWC0:
            return transDataPadNDC1HWC0(function, inputTile, C0);
        case TileOpFormat::TILEOP_FRACTAL_Z_3D:
            return transDataPadFractalZ3D(function, inputTile, C0);
        default:
            ASSERT(VectorErrorCode::ERR_PARAM_INVALID, false) << "The transDataType is not supported";
    }
    return inputTile;
}

void HandleNC1HWC0Format(
    Function& function, const LogicalTensorPtr& dstTensor, const LogicalTensorPtr& inputTile,
    std::vector<SymbolicScalar>& tileParams, const TransDataTileInfoPara& transDataTileInfoPara)
{
    int64_t C0 = BLOCK_SIZE / BytesOf(inputTile->Datatype());
    auto realInputTile = transDataPad<TileOpFormat::TILEOP_NC1HWC0>(function, inputTile, C0);
    auto realInputShape = realInputTile->GetShape();

    int64_t N = realInputShape[0];
    int64_t C = realInputShape[1];
    int64_t H = realInputShape[2];
    int64_t W = ((realInputShape[3] * BytesOf(inputTile->Datatype()) + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE /
                BytesOf(inputTile->Datatype());
    int64_t C1 = C / C0;
    int64_t shape1 = N * C1 * H * W * C0;
    int64_t yTileSizeElem = BytesOf(inputTile->Datatype()) == 1 ? 32 : 16;
    int64_t shape2 = H * W * ((C0 + yTileSizeElem - 1) / yTileSizeElem * yTileSizeElem);
    std::vector<int64_t> tmpShape = {shape1 + shape2};
    auto tmpTile = std::make_shared<LogicalTensor>(function, inputTile->Datatype(), tmpShape);

    auto& op = function.AddOperation(Opcode::OP_NCHW2NC1HWC0, {realInputTile}, {dstTensor, tmpTile});
    for (int i = 0; i < SHAPE_DIM4; i++) {
        tileParams[i] = SymbolicScalar(transDataTileInfoPara.inputTileInfo.offset[i]);
    }
    op.SetAttribute(OpAttributeKey::transDataOffset, tileParams);
}

void HandleFractalZFormat(
    Function& function, const LogicalTensorPtr& dstTensor, const LogicalTensorPtr& inputTile,
    std::vector<SymbolicScalar>& tileParams, const TransDataTileInfoPara& transDataTileInfoPara,
    const TransDataPara& transDataPara)
{
    int64_t C0 = BLOCK_SIZE / BytesOf(inputTile->Datatype());
    auto realInputTile = transDataPad<TileOpFormat::TILEOP_FRACTAL_Z>(function, inputTile, C0);
    auto realInputShape = realInputTile->GetShape();

    int64_t N = realInputShape[0];
    int64_t C = realInputShape[1];
    int64_t H = realInputShape[2];
    int64_t W = ((realInputShape[3] * BytesOf(inputTile->Datatype()) + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE /
                BytesOf(inputTile->Datatype());
    int64_t N0 = 16;
    int64_t N1 = N / N0;
    int64_t C1 = C / C0;
    int64_t shape1 = N * C1 * H * W * C0;
    int64_t shape2 = C1 * H * W * N1 * N0 * C0;
    int64_t yTileSizeElem = BytesOf(inputTile->Datatype()) == 1 ? 32 : 16;
    int64_t shape3 = H * W * ((C0 + yTileSizeElem - 1) / yTileSizeElem * yTileSizeElem);
    std::vector<int64_t> tmpShape = {shape1 + shape2 + shape3};
    auto tmpTile = std::make_shared<LogicalTensor>(function, inputTile->Datatype(), tmpShape);

    auto& op = function.AddOperation(Opcode::OP_NCHW2Fractal_Z, {realInputTile}, {dstTensor, tmpTile});
    for (int i = 0; i < SHAPE_DIM4; i++) {
        tileParams[i] = SymbolicScalar(transDataTileInfoPara.inputTileInfo.offset[i]);
    }
    tileParams[4] = transDataPara.groupIdx;
    tileParams[5] = transDataPara.group;
    op.SetAttribute(OpAttributeKey::transDataOffset, tileParams);
}

void HandleNDC1HWC0Format(
    Function& function, const LogicalTensorPtr& dstTensor, const LogicalTensorPtr& inputTile,
    std::vector<SymbolicScalar>& tileParams, const TransDataTileInfoPara& transDataTileInfoPara)
{
    int64_t C0 = BLOCK_SIZE / BytesOf(inputTile->Datatype());
    auto realInputTile = transDataPad<TileOpFormat::TILEOP_NDC1HWC0>(function, inputTile, C0);
    auto realInputShape = realInputTile->GetShape();

    int64_t D = realInputShape[1];
    int64_t C = realInputShape[2];
    int64_t H = realInputShape[3];
    int64_t W = ((realInputShape[4] * BytesOf(inputTile->Datatype()) + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE /
                BytesOf(inputTile->Datatype());

    int64_t C1 = C / C0;
    int64_t shape1 = D * C1 * H * W * C0;
    int64_t yTileSizeElem = BytesOf(inputTile->Datatype()) == 1 ? 32 : 16;
    int64_t shape2 = H * W * ((C0 + yTileSizeElem - 1) / yTileSizeElem * yTileSizeElem);
    std::vector<int64_t> tmpShape = {shape1 + shape2};
    auto tmpTile = std::make_shared<LogicalTensor>(function, inputTile->Datatype(), tmpShape);

    auto& op = function.AddOperation(Opcode::OP_NCDHW2NDC1HWC0, {realInputTile}, {dstTensor, tmpTile});
    for (int i = 0; i < SHAPE_DIM5; i++) {
        tileParams[i] = SymbolicScalar(transDataTileInfoPara.inputTileInfo.offset[i]);
    }
    op.SetAttribute(OpAttributeKey::transDataOffset, tileParams);
}

void HandleNCHW5DimFormat(
    Function& function, const LogicalTensorPtr& dstTensor, const LogicalTensorPtr& inputTile,
    std::vector<SymbolicScalar>& tileParams, const TransDataTileInfoPara& transDataTileInfoPara)
{
    int64_t N = inputTile->GetShape()[0];
    int64_t C1 = inputTile->GetShape()[1];
    int64_t H = inputTile->GetShape()[2];
    int64_t W = inputTile->GetShape()[3];
    int64_t C0 = inputTile->GetShape()[4];

    int64_t shape1 = N * C1 * C0 * H * W;
    int64_t yTileSizeElem = BytesOf(inputTile->Datatype()) == 1 ? 32 : 16;
    int64_t shape2 = C0 * ((H * W + yTileSizeElem - 1) / yTileSizeElem * yTileSizeElem);
    std::vector<int64_t> tmpShape = {shape1 + shape2};
    auto tmpTile = std::make_shared<LogicalTensor>(function, inputTile->Datatype(), tmpShape);

    auto& op = function.AddOperation(Opcode::OP_NC1HWC02NCHW, {inputTile}, {dstTensor, tmpTile});
    for (int i = 0; i < SHAPE_DIM5; i++) {
        tileParams[i] = SymbolicScalar(transDataTileInfoPara.inputTileInfo.offset[i]);
    }
    op.SetAttribute(OpAttributeKey::transDataOffset, tileParams);
}

void HandleNCDHW6DimFormat(
    Function& function, const LogicalTensorPtr& dstTensor, const LogicalTensorPtr& inputTile,
    std::vector<SymbolicScalar>& tileParams, const TransDataTileInfoPara& transDataTileInfoPara)
{
    int64_t D = inputTile->GetShape()[1];
    int64_t C1 = inputTile->GetShape()[2];
    int64_t H = inputTile->GetShape()[3];
    int64_t W = inputTile->GetShape()[4];
    int64_t C0 = inputTile->GetShape()[5];

    int64_t shape1 = D * C1 * C0 * H * W;
    int64_t yTileSizeElem = BytesOf(inputTile->Datatype()) == 1 ? 32 : 16;
    int64_t shape2 = C0 * ((H * W + yTileSizeElem - 1) / yTileSizeElem * yTileSizeElem);
    std::vector<int64_t> tmpShape = {shape1 + shape2};
    auto tmpTile = std::make_shared<LogicalTensor>(function, inputTile->Datatype(), tmpShape);
    auto reshapedTile = std::make_shared<LogicalTensor>(function, inputTile->Datatype(), Shape{D, C1, H, W, C0});

    [[maybe_unused]] auto& reshapeOp = function.AddOperation("TILE_RESHAPE", {inputTile}, {reshapedTile});

    auto& op = function.AddOperation(Opcode::OP_NDC1HWC02NCDHW, {reshapedTile}, {dstTensor, tmpTile});
    for (int i = 0; i < SHAPE_DIM6; i++) {
        tileParams[i] = SymbolicScalar(transDataTileInfoPara.inputTileInfo.offset[i]);
    }
    op.SetAttribute(OpAttributeKey::transDataOffset, tileParams);
}

template <TileOpFormat T>
void InnerTransDataND5Dim(
    size_t cur, Function& function, const TileShape& tileShape, const TransDataPara& transDataPara,
    TransDataTileInfoPara& transDataTileInfoPara)
{
    const LogicalTensorPtr& input = transDataPara.input;
    const LogicalTensorPtr& dstTensor = transDataPara.dstTensor;
    std::vector<SymbolicScalar> tileParams = transDataPara.tileParams;
    const int group = transDataPara.group;
    const int groupIdx = transDataPara.groupIdx;
    auto vecTile = tileShape.GetVecTile();
    int inputSize = input->GetShape().size();

    std::unordered_map<int64_t, int64_t> format2InputAxis = {{5, 1}, {6, 2}};
    int64_t inputGroupAxis = format2InputAxis[inputSize];
    ASSERT(VectorErrorCode::ERR_PARAM_INVALID, group > 0) << "The group is not valid !";
    int64_t inputPerGroup = input->GetShape()[inputGroupAxis] / group;

    if (cur == input->GetShape().size()) {
        int64_t offsetSuffix = transDataTileInfoPara.inputTileInfo.offset[inputGroupAxis] % inputPerGroup;
        transDataTileInfoPara.inputTileInfo.offset[inputGroupAxis] = groupIdx * inputPerGroup + offsetSuffix;
        std::shared_ptr<LogicalTensor> inputTile = input->View(
            function, transDataTileInfoPara.inputTileInfo.shape, transDataTileInfoPara.inputTileInfo.offset);

        switch (inputSize) {
            case 5:
                HandleNCHW5DimFormat(function, dstTensor, inputTile, tileParams, transDataTileInfoPara);
                return;
            case 6:
                HandleNCDHW6DimFormat(function, dstTensor, inputTile, tileParams, transDataTileInfoPara);
                return;
            default:
                ASSERT(VectorErrorCode::ERR_PARAM_INVALID, false) << "The transDataType is not supported";
        }
    }

    int64_t tmpTile = vecTile[cur];
    int64_t curShapeLen = cur == static_cast<size_t>(inputGroupAxis) ? inputPerGroup : input->GetShape()[cur];

    for (int i = 0; i < curShapeLen; i += tmpTile) {
        transDataTileInfoPara.inputTileInfo.offset[cur] = i;
        transDataTileInfoPara.inputTileInfo.shape[cur] = std::min(curShapeLen - i, tmpTile);
        InnerTransDataND5Dim<T>(cur + 1, function, tileShape, transDataPara, transDataTileInfoPara);
    }
}

void HandleFractalZ3DFormat(
    Function& function, const LogicalTensorPtr& dstTensor, const LogicalTensorPtr& inputTile,
    std::vector<SymbolicScalar>& tileParams, const TransDataTileInfoPara& transDataTileInfoPara,
    const TransDataPara& transDataPara)
{
    int64_t C0 = BLOCK_SIZE / BytesOf(inputTile->Datatype());
    auto realInputTile = transDataPad<TileOpFormat::TILEOP_FRACTAL_Z_3D>(function, inputTile, C0);
    auto realInputShape = realInputTile->GetShape();

    int64_t N = realInputShape[0];
    int64_t C = realInputShape[1];
    int64_t D = realInputShape[2];
    int64_t H = realInputShape[3];
    int64_t W = ((realInputShape[4] * BytesOf(inputTile->Datatype()) + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE /
                BytesOf(inputTile->Datatype());
    int64_t C1 = C / C0;
    int64_t shape1 = N * C1 * D * H * W * C0;
    int64_t tmp1 = N * C1 * C0 * H * W;
    int64_t yTileSizeElem = BytesOf(inputTile->Datatype()) == 1 ? 32 : 16;
    int64_t tmp2 = H * W * ((C0 + yTileSizeElem - 1) / yTileSizeElem * yTileSizeElem);
    int64_t shape2 = tmp1 + std::max(tmp1, tmp2) * 2;
    std::vector<int64_t> tmpShape = {shape1 + shape2};
    auto tmpTile = std::make_shared<LogicalTensor>(function, inputTile->Datatype(), tmpShape);

    auto& op = function.AddOperation(Opcode::OP_NCDHW2FRACTAL_Z_3D, {realInputTile}, {dstTensor, tmpTile});
    for (int i = 0; i < SHAPE_DIM5; i++) {
        tileParams[i] = SymbolicScalar(transDataTileInfoPara.inputTileInfo.offset[i]);
    }
    tileParams[5] = transDataPara.groupIdx;
    tileParams[6] = transDataPara.group;
    op.SetAttribute(OpAttributeKey::transDataOffset, tileParams);
}

template <TileOpFormat T>
void InnerTransData(
    size_t cur, Function& function, const TileShape& tileShape, const TransDataPara& transDataPara,
    TransDataTileInfoPara& transDataTileInfoPara)
{
    if (T == TileOpFormat::TILEOP_ND) {
        return InnerTransDataND5Dim<T>(cur, function, tileShape, transDataPara, transDataTileInfoPara);
    }

    const LogicalTensorPtr& input = transDataPara.input;
    const LogicalTensorPtr& dstTensor = transDataPara.dstTensor;
    std::vector<SymbolicScalar> tileParams = transDataPara.tileParams;
    const int group = transDataPara.group;
    const int groupIdx = transDataPara.groupIdx;
    auto vecTile = tileShape.GetVecTile();

    int64_t C0 = dstTensor->GetShape().back();
    int64_t N0 = 16;
    std::unordered_map<TileOpFormat, int64_t> format2InputAxis = {
        {TileOpFormat::TILEOP_NC1HWC0, 1},
        {TileOpFormat::TILEOP_FRACTAL_Z, 0},
        {TileOpFormat::TILEOP_NDC1HWC0, 2},
        {TileOpFormat::TILEOP_FRACTAL_Z_3D, 0}};
    int64_t inputGroupAxis = format2InputAxis[T];
    std::unordered_map<TileOpFormat, int64_t> format2OutputAxis = {
        {TileOpFormat::TILEOP_NC1HWC0, 1},
        {TileOpFormat::TILEOP_FRACTAL_Z, 1},
        {TileOpFormat::TILEOP_NDC1HWC0, 2},
        {TileOpFormat::TILEOP_FRACTAL_Z_3D, 1}};
    int64_t outputGroupAxis = format2OutputAxis[T];
    ASSERT(VectorErrorCode::ERR_PARAM_INVALID, group > 0) << "The group is not valid !";
    int64_t inputPerGroup = input->GetShape()[inputGroupAxis] / group;
    int64_t factor = (T == TileOpFormat::TILEOP_FRACTAL_Z || T == TileOpFormat::TILEOP_FRACTAL_Z_3D) ? N0 : C0;
    bool isFractalZ = T == TileOpFormat::TILEOP_FRACTAL_Z || T == TileOpFormat::TILEOP_FRACTAL_Z_3D;
    int64_t dstPerGroup = isFractalZ ? dstTensor->GetShape()[outputGroupAxis] * factor :
                                       dstTensor->GetShape()[outputGroupAxis] / group * factor;

    if (cur == input->GetShape().size()) {
        int64_t offsetSuffix = transDataTileInfoPara.inputTileInfo.offset[inputGroupAxis] % dstPerGroup;
        transDataTileInfoPara.inputTileInfo.offset[inputGroupAxis] = groupIdx * inputPerGroup + offsetSuffix;
        std::shared_ptr<LogicalTensor> inputTile = input->View(
            function, transDataTileInfoPara.inputTileInfo.shape, transDataTileInfoPara.inputTileInfo.offset);
        transDataTileInfoPara.inputTileInfo.offset[inputGroupAxis] = groupIdx * dstPerGroup + offsetSuffix;

        switch (T) {
            case TileOpFormat::TILEOP_NC1HWC0:
                HandleNC1HWC0Format(function, dstTensor, inputTile, tileParams, transDataTileInfoPara);
                return;
            case TileOpFormat::TILEOP_FRACTAL_Z:
                HandleFractalZFormat(function, dstTensor, inputTile, tileParams, transDataTileInfoPara, transDataPara);
                return;
            case TileOpFormat::TILEOP_NDC1HWC0:
                HandleNDC1HWC0Format(function, dstTensor, inputTile, tileParams, transDataTileInfoPara);
                return;
            case TileOpFormat::TILEOP_FRACTAL_Z_3D:
                HandleFractalZ3DFormat(
                    function, dstTensor, inputTile, tileParams, transDataTileInfoPara, transDataPara);
                return;
            default:
                ASSERT(VectorErrorCode::ERR_PARAM_INVALID, false) << "The transDataType is not supported";
        }
    }

    int64_t tmpTile = vecTile[cur];
    int64_t curShapeLen = cur == static_cast<size_t>(inputGroupAxis) ? inputPerGroup : input->GetShape()[cur];

    for (int i = 0; i < curShapeLen; i += tmpTile) {
        transDataTileInfoPara.inputTileInfo.offset[cur] = i;
        transDataTileInfoPara.inputTileInfo.shape[cur] = std::min(curShapeLen - i, tmpTile);
        InnerTransData<T>(cur + 1, function, tileShape, transDataPara, transDataTileInfoPara);
    }
}

template <TileOpFormat T>
void TiledTransData(Function& function, const TileShape& tileShape, TransDataPara& transDataPara)
{
    TransDataTileInfoPara transDataTileInfoPara{
        TileInfo(transDataPara.input->GetShape().size(), transDataPara.input->GetOffset().size()),
        TileInfo(transDataPara.dstTensor->GetShape().size(), transDataPara.dstTensor->GetOffset().size())};
    int group = transDataPara.group;
    for (int i = 0; i < group; i++) {
        transDataPara.groupIdx = i;
        InnerTransData<T>(0, function, tileShape, transDataPara, transDataTileInfoPara);
    }
}

LogicalTensorPtr TransDataNCHW2NC1HWC0(Function& function, const LogicalTensorPtr& self, [[maybe_unused]] const LogicalTensorPtr& output, int group)
{
    Shape resultShape = self->GetShape();
    int64_t C = resultShape[1];
    int64_t C0 = BLOCK_SIZE / BytesOf(self->Datatype());
    ASSERT(VectorErrorCode::ERR_PARAM_INVALID, C0 > 0) << "The C0 is not valid !";
    ASSERT(VectorErrorCode::ERR_PARAM_INVALID, group > 0) << "The group is not valid !";
    int perGroupC = C / group;
    int perGroupC1 = (perGroupC + C0 - 1) / C0;
    int totalC1 = perGroupC1 * group;
    resultShape[1] = totalC1;
    resultShape.push_back(C0);

    VecTile oriVectile = TileShape::Current().GetVecTile();
    ASSERT(VectorErrorCode::ERR_PARAM_INVALID, oriVectile.tile[1] % C0 == 0) << "The tileShape C  is not valid!";

    std::vector<SymbolicScalar> resultValidShape(self->GetDynValidShape());
    SymbolicScalar validShapeC = resultValidShape[1];
    SymbolicScalar perGroupValidShapeC = validShapeC / group;
    SymbolicScalar perGroupValidShapeC1 = (perGroupValidShapeC + C0 - 1) / C0;
    SymbolicScalar totalValidShapeC1 = perGroupValidShapeC1 * group;
    resultValidShape[1] = totalValidShapeC1;
    resultValidShape.push_back(SymbolicScalar(C0));
    auto result = std::make_shared<LogicalTensor>(
        function, self->Datatype(), resultShape, resultValidShape, TileOpFormat::TILEOP_NC1HWC0);

    auto& op = function.AddOperation(Opcode::OP_NCHW2NC1HWC0, {self}, {result});
    std::vector<SymbolicScalar> tileParams = {};
    // n c h w N C H W
    for (auto j : self->GetShape()) {
        (void)j;
        tileParams.push_back(SymbolicScalar(0));
    }
    for (auto j : self->GetShape()) {
        tileParams.push_back(SymbolicScalar(j));
    }
    tileParams[5] = totalC1 * C0;
    op.SetAttribute(OpAttributeKey::transDataOffset, tileParams);
    op.SetAttribute(OP_ATTR_PREFIX + "group", group);
    return result;
}

LogicalTensorPtr TransDataNCHW2Fractal_Z(Function& function, const LogicalTensorPtr& self, [[maybe_unused]] const LogicalTensorPtr& output, int group)
{
    int64_t N = self->GetShape()[0];
    int64_t C = self->GetShape()[1];
    int64_t H = self->GetShape()[2];
    int64_t W = self->GetShape()[3];
    int64_t N0 = 16;
    ASSERT(VectorErrorCode::ERR_PARAM_INVALID, group > 0) << "The group is not valid !";
    int64_t perGroupN = N / group;
    int64_t perGroupN1 = (perGroupN + N0 - 1) / N0;
    int64_t C0 = BLOCK_SIZE / BytesOf(self->Datatype());
    int64_t C1 = (C + C0 - 1) / C0;
    Shape resultShape = {group * C1 * H * W, perGroupN1, N0, C0};
    SymbolicScalar validShapeN = self->GetDynValidShape()[0];
    SymbolicScalar validShapeC = self->GetDynValidShape()[1];
    SymbolicScalar validShapeH = self->GetDynValidShape()[2];
    SymbolicScalar validShapeW = self->GetDynValidShape()[3];
    SymbolicScalar validShapeC1 = validShapeC / C0;
    SymbolicScalar vSPerGroupN1 = (validShapeN / group + N0 - 1) / N0;
    std::vector<SymbolicScalar> resultValidShape = {
        group * validShapeC1 * validShapeH * validShapeW, vSPerGroupN1, N0, C0};

    VecTile oriVectile = TileShape::Current().GetVecTile();
    ASSERT(VectorErrorCode::ERR_PARAM_INVALID, oriVectile.tile[0] % N0 == 0) << "The tileShape N  is not valid!";
    ASSERT(VectorErrorCode::ERR_PARAM_INVALID, oriVectile.tile[1] % C0 == 0) << "The tileShape C  is not valid!";

    auto result = std::make_shared<LogicalTensor>(
        function, self->Datatype(), resultShape, resultValidShape, TileOpFormat::TILEOP_FRACTAL_Z);
    auto& op = function.AddOperation(Opcode::OP_NCHW2Fractal_Z, {self}, {result});
    std::vector<SymbolicScalar> tileParams = {};
    // n c h w idx group N C H W
    for (auto i : self->GetShape()) {
        (void)i;
        tileParams.push_back(SymbolicScalar(0));
    }
    tileParams.push_back(0);
    tileParams.push_back(0);
    for (auto i : self->GetShape()) {
        tileParams.push_back(SymbolicScalar(i));
    }
    tileParams[6] = perGroupN1 * N0;
    tileParams[7] = C1 * C0;
    op.SetAttribute(OpAttributeKey::transDataOffset, tileParams);
    op.SetAttribute(OP_ATTR_PREFIX + "group", group);
    return result;
}

LogicalTensorPtr TransDataNCDHW2NDC1HWC0(Function& function, const LogicalTensorPtr& self, [[maybe_unused]] const LogicalTensorPtr& output, int group)
{
    int64_t N = self->GetShape()[0];
    int64_t C = self->GetShape()[1];
    int64_t D = self->GetShape()[2];
    int64_t H = self->GetShape()[3];
    int64_t W = self->GetShape()[4];
    int64_t C0 = BLOCK_SIZE / BytesOf(self->Datatype());
    ASSERT(VectorErrorCode::ERR_PARAM_INVALID, C0 > 0) << "The C0 is not valid !";
    ASSERT(VectorErrorCode::ERR_PARAM_INVALID, group > 0) << "The group is not valid !";
    int64_t perGroupC = C / group;
    int64_t perGroupC1 = (perGroupC + C0 - 1) / C0;
    int64_t totalC1 = perGroupC1 * group;
    int64_t totalC = totalC1 * C0;
    Shape resultShape = {N, D, totalC1, H, W, C0};

    SymbolicScalar validShapeN = self->GetDynValidShape()[0];
    SymbolicScalar validShapeC = self->GetDynValidShape()[1];
    SymbolicScalar validShapeD = self->GetDynValidShape()[2];
    SymbolicScalar validShapeH = self->GetDynValidShape()[3];
    SymbolicScalar validShapeW = self->GetDynValidShape()[4];
    SymbolicScalar validShapePerGroupC = validShapeC / group;
    SymbolicScalar validShapePerGroupC1 = (validShapePerGroupC + C0 - 1) / C0;
    SymbolicScalar validShapePerTotalC1 = validShapePerGroupC1 * group;

    std::vector<SymbolicScalar> resultValidShape = {validShapeN, validShapeD, validShapePerTotalC1,
                                                    validShapeH, validShapeW, C0};
    auto result = std::make_shared<LogicalTensor>(
        function, self->Datatype(), resultShape, resultValidShape, TileOpFormat::TILEOP_NDC1HWC0);
    auto tmpInput = Permute(function, Tensor(self), {0, 2, 1, 3, 4});

    VecTile oriVectile = TileShape::Current().GetVecTile();
    ASSERT(VectorErrorCode::ERR_PARAM_INVALID, oriVectile.tile[1] % C0 == 0) << "The tileShape C  is not valid!";
    VecTile tmpVectile = TileShape::Current().GetVecTile();
    std::swap(tmpVectile.tile[1], tmpVectile.tile[2]);
    TileShape::Current().SetVecTile(tmpVectile);

    auto& op = function.AddOperation(Opcode::OP_NCDHW2NDC1HWC0, {tmpInput.GetStorage()}, {result});
    std::vector<SymbolicScalar> tileParams = {};
    // n d c h w N D C H W
    for (auto i : tmpInput.GetShape()) {
        (void)i;
        tileParams.push_back(SymbolicScalar(0));
    }
    for (auto i : tmpInput.GetShape()) {
        tileParams.push_back(SymbolicScalar(i));
    }
    tileParams[7] = totalC;
    op.SetAttribute(OpAttributeKey::transDataOffset, tileParams);
    op.SetAttribute(OP_ATTR_PREFIX + "group", group);
    TileShape::Current().SetVecTile(oriVectile);
    return result;
}

LogicalTensorPtr TransDataFRACTAL_Z_3D(Function& function, const LogicalTensorPtr& self, [[maybe_unused]] const LogicalTensorPtr& output, int group)
{
    int64_t N = self->GetShape()[0];
    int64_t C = self->GetShape()[1];
    int64_t D = self->GetShape()[2];
    int64_t H = self->GetShape()[3];
    int64_t W = self->GetShape()[4];
    int64_t N0 = 16;
    ASSERT(VectorErrorCode::ERR_PARAM_INVALID, group > 0) << "The group is not valid !";
    int64_t perGroupN = N / group;
    int64_t perGroupN1 = (perGroupN + N0 - 1) / N0;
    int64_t C0 = BLOCK_SIZE / BytesOf(self->Datatype());
    int64_t C1 = (C + C0 - 1) / C0;
    Shape resultShape = {group * D * C1 * H * W, perGroupN1, N0, C0};

    SymbolicScalar validShapeN = self->GetDynValidShape()[0];
    SymbolicScalar validShapeC = self->GetDynValidShape()[1];
    SymbolicScalar validShapeD = self->GetDynValidShape()[2];
    SymbolicScalar validShapeH = self->GetDynValidShape()[3];
    SymbolicScalar validShapeW = self->GetDynValidShape()[4];
    auto validShapeC1 = (validShapeC - C0 + 1) / C0;
    auto validShapePerGroupN = validShapeN / group;
    auto validShapePerGroupN1 = (validShapePerGroupN + N0 - 1) / N0;
    std::vector<SymbolicScalar> resultValidShape = {
        group * validShapeD * validShapeC1 * validShapeH * validShapeW, validShapePerGroupN1, N0, C0};

    VecTile oriVectile = TileShape::Current().GetVecTile();
    ASSERT(VectorErrorCode::ERR_PARAM_INVALID, oriVectile.tile[0] % N0 == 0) << "The tileShape N  is not valid!";
    ASSERT(VectorErrorCode::ERR_PARAM_INVALID, oriVectile.tile[1] % C0 == 0) << "The tileShape C is not valid!";

    auto result = std::make_shared<LogicalTensor>(
        function, self->Datatype(), resultShape, resultValidShape, TileOpFormat::TILEOP_FRACTAL_Z_3D);

    auto& op = function.AddOperation(Opcode::OP_NCDHW2FRACTAL_Z_3D, {self}, {result});
    std::vector<SymbolicScalar> tileParams = {};
    // n c d h w idx group N C D H W
    for (auto i : self->GetShape()) {
        (void)i;
        tileParams.push_back(SymbolicScalar(0));
    }
    tileParams.push_back(0);
    tileParams.push_back(0);
    for (auto i : self->GetShape()) {
        tileParams.push_back(SymbolicScalar(i));
    }
    tileParams[7] = perGroupN1 * N0;
    tileParams[8] = C1 * C0;
    op.SetAttribute(OpAttributeKey::transDataOffset, tileParams);
    op.SetAttribute(OP_ATTR_PREFIX + "group", group);
    return result;
}

LogicalTensorPtr TransDataNDC1HWC02NCDHW(Function& function, const LogicalTensorPtr& self, const LogicalTensorPtr& output, int group)
{
    int64_t N = self->GetShape()[0];
    int64_t D = self->GetShape()[1];
    int64_t C1 = self->GetShape()[2];
    int64_t H = self->GetShape()[3];
    int64_t W = self->GetShape()[4];
    int64_t C0 = self->GetShape()[5];
    Shape resultShape = {N, D, C1 * C0, H, W};
    ASSERT(VectorErrorCode::ERR_PARAM_INVALID, output->GetShape().size() == SHAPE_DIM6 || C1 * C0 == output->GetShape()[SHAPE_DIM1]) << "Not supported for pad scenarios!";

    SymbolicScalar validShapeN = self->GetDynValidShape()[0];
    SymbolicScalar validShapeD = self->GetDynValidShape()[1];
    SymbolicScalar validShapeC1 = self->GetDynValidShape()[2];
    SymbolicScalar validShapeH = self->GetDynValidShape()[3];
    SymbolicScalar validShapeW = self->GetDynValidShape()[4];
    SymbolicScalar validShapeC0 = self->GetDynValidShape()[5];
    std::vector<SymbolicScalar> resultValidShape = {
        validShapeN, validShapeD, validShapeC1 * C0, validShapeH, validShapeW};

    VecTile oriVectile = TileShape::Current().GetVecTile();
    ASSERT(VectorErrorCode::ERR_PARAM_INVALID, oriVectile.tile[4] % C0 == 0) << "The tileShape W  is not valid!";
    ASSERT(VectorErrorCode::ERR_PARAM_INVALID, oriVectile.tile[5] == C0) << "The tileShape C0 is not valid!";

    auto result =
        std::make_shared<LogicalTensor>(function, self->Datatype(), resultShape, resultValidShape, self->Format());

    auto& op = function.AddOperation(Opcode::OP_NDC1HWC02NCDHW, {self}, {result});
    std::vector<SymbolicScalar> tileParams = {};
    // n d c1 h w c0 N D C1 H W C0
    for (auto i : self->GetShape()) {
        (void)i;
        tileParams.push_back(SymbolicScalar(0));
    }
    for (auto i : self->GetShape()) {
        tileParams.push_back(SymbolicScalar(i));
    }
    op.SetAttribute(OpAttributeKey::transDataOffset, tileParams);
    op.SetAttribute(OP_ATTR_PREFIX + "group", group);
    VecTile tmpVectile = TileShape::Current().GetVecTile();
    tmpVectile.tile[1] *= tmpVectile.tile[5];
    tmpVectile.tile.pop_back();
    TileShape::Current().SetVecTile(tmpVectile);
    auto tmpResult = Permute(function, Tensor(result), {0, 2, 1, 3, 4});
    TileShape::Current().SetVecTile(oriVectile);
    return tmpResult.GetStorage();
}

LogicalTensorPtr TransDataNC1HWC02NCHW(Function& function, const LogicalTensorPtr& self, const LogicalTensorPtr& output, int group)
{
    if (self->GetShape().size() != 5) {
        return TransDataNDC1HWC02NCDHW(function, self, output, group);
    }
    int64_t N = self->GetShape()[0];
    int64_t C1 = self->GetShape()[1];
    int64_t H = self->GetShape()[2];
    int64_t W = self->GetShape()[3];
    int64_t C0 = self->GetShape()[4];
    Shape resultShape = {N, C1 * C0, H, W};
    ASSERT(VectorErrorCode::ERR_PARAM_INVALID, output->GetShape().size() == SHAPE_DIM5 || C1 * C0 == output->GetShape()[1]) << "Not supported for pad scenarios!";
    SymbolicScalar validShapeN = self->GetDynValidShape()[0];
    SymbolicScalar validShapeC1 = self->GetDynValidShape()[1];
    SymbolicScalar validShapeH = self->GetDynValidShape()[2];
    SymbolicScalar validShapeW = self->GetDynValidShape()[3];
    std::vector<SymbolicScalar> resultValidShape = {validShapeN, validShapeC1 * C0, validShapeH, validShapeW};

    VecTile oriVectile = TileShape::Current().GetVecTile();
    ASSERT(VectorErrorCode::ERR_PARAM_INVALID, oriVectile.tile[3] % C0 == 0) << "The tileShape W  is not valid!";
    oriVectile.tile[4] = C0;

    auto result = std::make_shared<LogicalTensor>(
        function, self->Datatype(), resultShape, resultValidShape, TileOpFormat::TILEOP_ND);

    auto& op = function.AddOperation(Opcode::OP_NC1HWC02NCHW, {self}, {result});
    std::vector<SymbolicScalar> tileParams = {};
    // n c1 h W c0 N C1 H W C0
    for (auto i : self->GetShape()) {
        (void)i;
        tileParams.push_back(SymbolicScalar(0));
    }
    for (auto i : self->GetShape()) {
        tileParams.push_back(SymbolicScalar(i));
    }
    op.SetAttribute(OpAttributeKey::transDataOffset, tileParams);
    op.SetAttribute(OP_ATTR_PREFIX + "group", group);
    return result;
}

LogicalTensorPtr TensorTransData(
    Function& function, const LogicalTensorPtr& self, const LogicalTensorPtr& output, TileOpFormat transDataType, int group)
{
    switch (transDataType) {
        case TileOpFormat::TILEOP_NC1HWC0:
            return TransDataNCHW2NC1HWC0(function, self, output, group);
        case TileOpFormat::TILEOP_FRACTAL_Z:
            return TransDataNCHW2Fractal_Z(function, self, output, group);
        case TileOpFormat::TILEOP_NDC1HWC0:
            return TransDataNCDHW2NDC1HWC0(function, self, output, group);
        case TileOpFormat::TILEOP_FRACTAL_Z_3D:
            return TransDataFRACTAL_Z_3D(function, self, output, group);
        case TileOpFormat::TILEOP_ND:
            return TransDataNC1HWC02NCHW(function, self, output, group); // 两种情况，NC1HWC0和NDC1HWC0
        default:
            ASSERT(VectorErrorCode::ERR_PARAM_INVALID, false) << "The transDataType is not supported";
    }
    return TransDataNCHW2NC1HWC0(function, self, output, group);
}

LogicalTensorPtr TransData(
    Function& function, const LogicalTensorPtr& self, const LogicalTensorPtr& output,
    TileOpFormat transDataType, int group)
{
    DECLARE_TRACER();
    std::unordered_set<DataType> supportedTypes = {DT_FP16, DT_BF16, DT_INT8, DT_INT16, DT_FP32, DT_INT32};
    CheckTensorDataType(self, supportedTypes, "TRANSDATA");
    CheckTensorShapeSize(self, "TRANSDATA");
    switch (transDataType) {
        case TileOpFormat::TILEOP_NC1HWC0:
            CheckTensorDimRange(self, SHAPE_DIM4, SHAPE_DIM4, "TRANSDATA NC1HWC0");
            break;
        case TileOpFormat::TILEOP_FRACTAL_Z:
            CheckTensorDimRange(self, SHAPE_DIM4, SHAPE_DIM4, "TRANSDATA FRACTAL_Z");
            break;
        case TileOpFormat::TILEOP_NDC1HWC0:
            CheckTensorDimRange(self, SHAPE_DIM5, SHAPE_DIM5, "TRANSDATA NDC1HWC0");
            break;
        case TileOpFormat::TILEOP_FRACTAL_Z_3D:
            CheckTensorDimRange(self, SHAPE_DIM5, SHAPE_DIM5, "TRANSDATA FRACTAL_Z_3D");
            break;
        case TileOpFormat::TILEOP_ND:
            CheckTensorDimRange(self, SHAPE_DIM5, SHAPE_DIM6, "TRANSDATA ND");
            break;
        default:
            ASSERT(VectorErrorCode::ERR_PARAM_INVALID, false) << "The transDataType is not supported";
    }
    return TensorTransData(function, self, output, transDataType, group);
}

Tensor TransData(const Tensor& self, TileOpFormat transDataType, int group)
{
    DECLARE_TRACER();
    auto& function = *Program::GetInstance().GetCurrentFunction();
    auto tmpTensor = TransData(function, self.GetStorage(), self.GetStorage(), transDataType, group);
    tmpTensor->tensor->format = TileOpFormat::TILEOP_ND;
    return Tensor(tmpTensor);
}

void TiledFull(
    Function& function, const TileShape& tileShape, size_t cur, const Element& value, const SymbolicScalar& dynValue,
    std::vector<int64_t>& shape, const std::vector<SymbolicScalar>& validShape, const LogicalTensorPtr& results,
    TileInfo& resultTileInfo)
{
    if (cur == results->shape.size()) {
        auto resultTile = results->View(function, resultTileInfo.shape, resultTileInfo.offset);
        auto& op = function.AddOperation("TILE_VEC_DUP", {}, {resultTile});
        op.SetAttribute(OpAttributeKey::scalar, value);
        if (dynValue.IsValid()) {
            op.SetAttribute(OpAttributeKey::dynScalar, dynValue);
        }
        op.SetAttribute(OP_ATTR_PREFIX + "shape", resultTileInfo.shape);
        op.SetAttribute(OP_ATTR_PREFIX + "validShape", resultTile->GetDynValidShape());
        return;
    }

    auto& vecTile = tileShape.GetVecTile();
    for (int i = 0; i < results->shape[cur]; i += vecTile[cur]) {
        resultTileInfo.offset[cur] = i;
        resultTileInfo.shape[cur] = std::min(results->shape[cur] - i, vecTile[cur]);
        TiledFull(function, tileShape, cur + 1, value, dynValue, shape, validShape, results, resultTileInfo);
    }
}

void TiledFull(
    Function& function, const TileShape& tileShape, const Element& value, const SymbolicScalar& dynValue,
    std::vector<int64_t>& shape, const std::vector<SymbolicScalar>& validShape, const LogicalTensorPtr& results)
{
    TileInfo resultTileInfo(results->shape.size(), results->offset.size());
    TiledFull(function, tileShape, 0, value, dynValue, shape, validShape, results, resultTileInfo);
}

Tensor TensorFullOperation(
    Function& function, const Element& src, const SymbolicScalar& dynValue, DataType dtype,
    const std::vector<int64_t>& dstShape, const std::vector<SymbolicScalar>& validShape)
{
    auto result = std::make_shared<LogicalTensor>(function, dtype, dstShape, validShape);
    auto& op = function.AddOperation(Opcode::OP_VEC_DUP, {}, {result}); // 输入没有tensor
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
    const Element& src, DataType dtype, const std::vector<int64_t>& dstShape, std::vector<SymbolicScalar> validShape)
{
    DECLARE_TRACER();
    std::unordered_set<DataType> supportedTypes = {DT_FP32,  DT_FP16,  DT_BF16,   DT_INT8,   DT_INT16,
                                                   DT_INT32, DT_UINT8, DT_UINT16, DT_UINT32, DT_BOOL};
    CheckTensorDataType(dtype, supportedTypes, "FULL");
    CheckDstShapeDimRange(dstShape, 1, 4, "FULL");
    CheckDstShapeSize(dstShape, "FULL");
    if (validShape.empty()) {
        for (auto x : dstShape)
            validShape.emplace_back(x);
    }
    RETURN_CALL(
        FullOperation, *Program::GetInstance().GetCurrentFunction(), src, SymbolicScalar(), dtype, dstShape,
        validShape);
}

Tensor Full(
    const SymbolicScalar& dynSrc, DataType dtype, const std::vector<int64_t>& dstShape,
    std::vector<SymbolicScalar> validShape)
{
    DECLARE_TRACER();
    std::unordered_set<DataType> supportedTypes = {DT_FP32,  DT_FP16,  DT_BF16,   DT_INT8,   DT_INT16,
                                                   DT_INT32, DT_UINT8, DT_UINT16, DT_UINT32, DT_BOOL};
    CheckTensorDataType(dtype, supportedTypes, "FULL");
    CheckDstShapeDimRange(dstShape, 1, 4, "FULL");
    CheckDstShapeSize(dstShape, "FULL");
    if (validShape.empty()) {
        for (auto x : dstShape)
            validShape.emplace_back(x);
    }
    RETURN_CALL(
        FullOperation, *Program::GetInstance().GetCurrentFunction(), Element(dtype, (int64_t)0), dynSrc, dtype,
        dstShape, validShape);
}

template <CastOpType T>
void TiledCastOperation(
    Function& function, const TileShape& tileShape, const int cur, Input& input, const LogicalTensorPtr& result,
    const CastMode& mode, const SaturationMode& satmode)
{
    if (cur == static_cast<int>(input.tensor.GetShape().size())) {
        auto tile = input.tensor.GetStorage()->View(function, input.tileInfo.shape, input.tileInfo.offset);
        auto resultTile = result->View(function, input.tileInfo.shape, input.tileInfo.offset);

        DataType srcDtype = tile->Datatype();
        DataType dstDtype = resultTile->Datatype();

        bool needTmpBuffer = false;
        if (((srcDtype == DT_FP32 && dstDtype == DT_INT16) || (srcDtype == DT_FP16 && dstDtype == DT_INT16) ||
             (srcDtype == DT_FP16 && dstDtype == DT_INT8)) &&
            satmode == SaturationMode::OFF) {
            needTmpBuffer = true;
        }

        Operation* op = nullptr;
        if (needTmpBuffer) {
            size_t shapeSize = input.tileInfo.shape.size();
            int64_t shapeW = (shapeSize >= 1) ? input.tileInfo.shape[shapeSize - 1] : 1;
            shapeW = AlignUp(shapeW + ALIGN_SIZE_64, static_cast<int64_t>(BLOCK_SIZE / BytesOf(DT_INT32)));
            std::vector<int64_t> tmpShape = {1, shapeW};
            auto tmpTensor = std::make_shared<LogicalTensor>(function, DT_INT32, tmpShape);
            op = &function.AddOperation(GetCastOpName<T>(), {tile}, {resultTile, tmpTensor});
        } else {
            op = &function.AddOperation(GetCastOpName<T>(), {tile}, {resultTile});
        }
        op->SetAttribute(OP_ATTR_PREFIX + "mode", mode);
        op->SetAttribute(OP_ATTR_PREFIX + "satmode", static_cast<int64_t>(satmode));
        return;
    }
    auto& vecTile = tileShape.GetVecTile();
    for (int i = 0; i < input.tensor.GetShape()[cur]; i += vecTile[cur]) {
        input.tileInfo.shape[cur] = std::min(input.tensor.GetShape()[cur] - i, vecTile[cur]);
        input.tileInfo.offset[cur] = i;
        TiledCastOperation<T>(function, tileShape, cur + 1, input, result, mode, satmode);
    }
}

template <CastOpType T>
void TiledCastOperation(
    Function& function, const TileShape& tileShape, const LogicalTensorPtr& operand, const LogicalTensorPtr& result,
    const CastMode& mode, const SaturationMode& satmode)
{
    ASSERT(VectorErrorCode::ERR_PARAM_INVALID, operand->shape.size() == operand->offset.size())
        << "The shape size of operand and offset should be equal";

    TileInfo tileInfo(result->shape.size(), result->offset.size());
    auto input = Input{operand, tileInfo};
    TiledCastOperation<T>(function, tileShape, 0, input, result, mode, satmode);
}

void CheckCastTypeSupport(DataType srcType, DataType dstType, const std::string& opName)
{
    // 同类型转换始终支持
    if (srcType == dstType) {
        return;
    }

    auto arch = Platform::Instance().GetSoc().GetNPUArch();
    bool isA5Architecture = (arch == NPUArch::DAV_3510);

    if (isA5Architecture) {
        // A5 架构支持的转换
        std::unordered_map<DataType, std::unordered_set<DataType>> a5SupportedConversions = {
            {DT_FP32, {DT_FP16, DT_BF16, DT_INT16, DT_INT32, DT_INT64, DT_FP8E4M3, DT_FP8E5M2, DT_HF8}},
            {DT_FP16, {DT_FP32, DT_INT32, DT_INT16, DT_INT8, DT_UINT8, DT_HF8}},
            {DT_BF16, {DT_FP32, DT_INT32, DT_FP16}},
            {DT_UINT8, {DT_FP16, DT_UINT16}},
            {DT_INT8, {DT_FP16, DT_INT16, DT_INT32}},
            {DT_INT16, {DT_UINT8, DT_FP16, DT_FP32, DT_UINT32, DT_INT32}},
            {DT_INT32, {DT_FP32, DT_INT16, DT_UINT16, DT_INT64, DT_UINT8, DT_FP16}},
            {DT_UINT32, {DT_UINT8, DT_UINT16, DT_INT16}},
            {DT_INT64, {DT_FP32, DT_INT32}},
            {DT_FP8E4M3, {DT_FP32}},
            {DT_FP8E5M2, {DT_FP32}},
            {DT_HF8, {DT_FP32}}};

        if (a5SupportedConversions.count(srcType) == 0 || a5SupportedConversions[srcType].count(dstType) == 0) {
            ASSERT(VectorErrorCode::ERR_PARAM_DTYPE_UNSUPPORTED, false)
                << "A5 architecture does not support cast from " << npu::tile_fwk::DataType2String(srcType) << " to "
                << npu::tile_fwk::DataType2String(dstType) << " in " << opName;
        }
    } else {
        // A2A3 架构支持的转换（其他架构也按A2A3处理）
        std::unordered_map<DataType, std::unordered_set<DataType>> a2a3SupportedConversions = {
            {DT_FP16, {DT_FP32, DT_INT32, DT_INT16, DT_INT8, DT_UINT8, DT_INT4}},
            {DT_BF16, {DT_FP32, DT_INT32}},
            {DT_INT32, {DT_FP32, DT_INT16, DT_INT64, DT_FP16}},
            {DT_FP32, {DT_BF16, DT_FP16, DT_INT16, DT_INT32, DT_INT64}},
            {DT_UINT8, {DT_FP16}},
            {DT_INT8, {DT_FP16}},
            {DT_INT16, {DT_FP32, DT_FP16}},
            {DT_INT64, {DT_FP32, DT_INT32}},
            {DT_INT4, {DT_FP16}}};

        if (a2a3SupportedConversions.count(srcType) == 0 || a2a3SupportedConversions[srcType].count(dstType) == 0) {
            ASSERT(VectorErrorCode::ERR_PARAM_DTYPE_UNSUPPORTED, false)
                << "A2A3 architecture does not support cast from " << npu::tile_fwk::DataType2String(srcType) << " to "
                << npu::tile_fwk::DataType2String(dstType) << " in " << opName;
        }
    }
}

Tensor Cast(const Tensor& self, DataType dstDataType, CastMode mode, SaturationMode satmode)
{
    DECLARE_TRACER();
    CheckCastTypeSupport(self.GetDataType(), dstDataType, "CAST");
    CheckTensorDimRange(self.GetStorage(), 1, 4, "CAST");
    CheckTensorShapeSize(self.GetStorage(), "CAST");
    ASSERT(VectorErrorCode::ERR_PARAM_INVALID, self.GetShape().size() == self.GetStorage()->offset.size())
        << "The shape size of self and offset should be equal";
    // Cast to same dType with no mode will do nothing
    if (self.GetStorage()->tensor->datatype == dstDataType && (mode == CAST_NONE || mode == CAST_RINT)) {
        return self;
    }
    RETURN_CALL(
        CastOperation<CastOpType::CAST>, *Program::GetInstance().GetCurrentFunction(), self.GetStorage(), dstDataType,
        mode, satmode);
}

void TensorInnerConcatNew(Function& function, const LogicalTensorPtr& operand, const LogicalTensorPtr& result)
{
    result->UpdateDynValidShape(operand->GetDynValidShape());
    function.AddOperation(Opcode::OP_REGISTER_COPY, {operand}, {result});
}

void InnerConcatNew(Function& function, const LogicalTensorPtr& operand, const LogicalTensorPtr& result)
{
    CALL(InnerConcatNew, function, operand, result);
}

void CheckCat(const std::vector<Tensor>& tensors, int axis)
{
    std::unordered_set<DataType> supportedTypes = {DT_INT8,   DT_UINT8, DT_INT16, DT_UINT16, DT_INT32,
                                                   DT_UINT32, DT_FP16,  DT_FP32,  DT_BF16};
    CheckTensorDataType(tensors[0].GetStorage(), supportedTypes, "CAT");
    CheckAxisRange(tensors[0], axis);
    std::vector<LogicalTensorPtr> tensorPtrs;
    for (auto tensor : tensors) {
        CheckTensorShapeSize(tensor.GetStorage(), "CAT");
        tensorPtrs.push_back(tensor.GetStorage());
    }
    CheckTensorsDimConsistency(tensorPtrs, "CAT");
    CheckTensorsDataTypeConsistency(tensorPtrs, "CAT");
    CheckTensorsFormatConsistency(tensorPtrs, "CAT");
    auto shape = tensors[0].GetShape();
    for (auto tensor : tensors) {
        for (int i = 0; static_cast<size_t>(i) < tensors[0].GetShape().size(); ++i) {
            if (i == axis) {
                continue;
            }
            ASSERT(VectorErrorCode::ERR_PARAM_INVALID, shape[i] == tensor.GetShape()[i])
                << "The shape of all tensors should be equal except at axis";
        }
    }
}

Tensor Cat(const std::vector<Tensor>& tensors, int axis)
{
    DECLARE_TRACER();
    CheckCat(tensors, axis);

    auto resultShape = tensors[0].GetShape();
    auto shapeSize = resultShape.size();
    CheckAxisRange(tensors[0], axis);
    int axisSize = 0;
    for (auto tensor : tensors) {
        axisSize += tensor.GetShape()[axis];
    }
    resultShape[axis] = axisSize;

    auto format = tensors[0].Format();
    Tensor result(tensors[0].GetDataType(), resultShape, "", format);
    Tensor tmp(tensors[0].GetDataType(), resultShape, "", format);
    auto& function = *Program::GetInstance().GetCurrentFunction();
    std::vector<int64_t> offset(shapeSize, 0);
    for (auto tensor : tensors) {
        auto tmpView = tmp.GetStorage()->View(function, tensor.GetShape(), offset);
        InnerConcatNew(*Program::GetInstance().GetCurrentFunction(), tensor.GetStorage(), tmpView);
        offset[axis] += tensor.GetShape()[axis];
    }
    auto& op = function.AddOperation(Opcode::OP_ASSEMBLE, {tmp.GetStorage()}, {result.GetStorage()});
    op.SetOpAttribute(std::make_shared<AssembleOpAttribute>(std::vector<int64_t>(shapeSize, 0)));

    return result;
}

void MoveOutOperationTileFunc(
    Function& function, const TileShape& tileShape, const std::vector<LogicalTensorPtr>& iOperand,
    const std::vector<LogicalTensorPtr>& oOperand, const Operation& op)
{
    auto shape = op.GetVectorIntAttribute<int>(OP_ATTR_PREFIX + "shape");
    TiledInnerTranspose<TransposeOpType::TRANSPOSE_MOVEOUT>(function, tileShape, iOperand[0], oOperand[0], shape);
}

void MoveInOperationTileFunc(
    Function& function, const TileShape& tileShape, const std::vector<LogicalTensorPtr>& iOperand,
    const std::vector<LogicalTensorPtr>& oOperand, const Operation& op)
{
    auto shape = op.GetVectorIntAttribute<int>(OP_ATTR_PREFIX + "shape");
    TiledInnerTranspose<TransposeOpType::TRANSPOSE_MOVEIN>(function, tileShape, iOperand[0], oOperand[0], shape);
}

void VnchwconvOperationTileFunc(
    Function& function, const TileShape& tileShape, const std::vector<LogicalTensorPtr>& iOperand,
    const std::vector<LogicalTensorPtr>& oOperand, const Operation& op)
{
    auto shape = op.GetVectorIntAttribute<int>(OP_ATTR_PREFIX + "shape");
    TiledInnerTranspose<TransposeOpType::TRANSPOSE_VNCHWCONV>(function, tileShape, iOperand[0], oOperand[0], shape);
}

void ExpandOperationTileFunc(
    Function& function, const TileShape& tileShape, const std::vector<LogicalTensorPtr>& iOperand,
    const std::vector<LogicalTensorPtr>& oOperand, const Operation& op)
{
    UnaryOperationOperandCheck(iOperand, oOperand);
    std::vector<SymbolicScalar> validShape;
    op.GetAttr(OP_ATTR_PREFIX + "validShape", validShape);
    TiledExpand(function, tileShape, iOperand[0], oOperand[0], validShape);
}

inline void CastOperationOperandCheck(
    const std::vector<LogicalTensorPtr>& iOperand, const std::vector<LogicalTensorPtr>& oOperand)
{
    ASSERT(VectorErrorCode::ERR_PARAM_INVALID, iOperand.size() == 1) << "The input operand size should be 1";
    ASSERT(VectorErrorCode::ERR_PARAM_INVALID, oOperand.size() == 1) << "The output operand size should be 1";
}

void CastOperationTileFunc(
    Function& function, const TileShape& tileShape, const std::vector<LogicalTensorPtr>& iOperand,
    const std::vector<LogicalTensorPtr>& oOperand, const Operation& op)
{
    CastOperationOperandCheck(iOperand, oOperand);
    int64_t satmodeValue = 1;
    op.GetAttr(OP_ATTR_PREFIX + "satmode", satmodeValue);
    SaturationMode satmode = static_cast<SaturationMode>(satmodeValue);
    auto mode = op.GetCastModeAttribute(OP_ATTR_PREFIX + "mode");
    TiledCastOperation<CastOpType::CAST>(function, tileShape, iOperand[0], oOperand[0], mode, satmode);
}

void FullOperationTileFunc(
    Function& function, const TileShape& tileShape, [[maybe_unused]] const std::vector<LogicalTensorPtr>& iOperand,
    const std::vector<LogicalTensorPtr>& oOperand, const Operation& op)
{
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

void TransDataTileFunc(
    Function& function, const TileShape& tileShape, const std::vector<LogicalTensorPtr>& iOperand,
    const std::vector<LogicalTensorPtr>& oOperand, const Operation& op)
{
    std::vector<SymbolicScalar> tileParams = op.GetVectorSymbolicScalarAttribute(OpAttributeKey::transDataOffset);
    int group = op.GetIntAttribute(OP_ATTR_PREFIX + "group");
    TransDataPara transDataPara = TransDataPara{iOperand[0], oOperand[0], tileParams, group, 0};
    switch (op.GetOpcode()) {
        case Opcode::OP_NCHW2NC1HWC0:
            TiledTransData<TileOpFormat::TILEOP_NC1HWC0>(function, tileShape, transDataPara);
            break;
        case Opcode::OP_NCHW2Fractal_Z:
            TiledTransData<TileOpFormat::TILEOP_FRACTAL_Z>(function, tileShape, transDataPara);
            break;
        case Opcode::OP_NC1HWC02NCHW:
            TiledTransData<TileOpFormat::TILEOP_ND>(function, tileShape, transDataPara);
            break;
        case Opcode::OP_NCDHW2NDC1HWC0:
            TiledTransData<TileOpFormat::TILEOP_NDC1HWC0>(function, tileShape, transDataPara);
            break;
        case Opcode::OP_NCDHW2FRACTAL_Z_3D:
            TiledTransData<TileOpFormat::TILEOP_FRACTAL_Z_3D>(function, tileShape, transDataPara);
            break;
        case Opcode::OP_NDC1HWC02NCDHW:
            TiledTransData<TileOpFormat::TILEOP_ND>(function, tileShape, transDataPara);
            break;
        default:
            ASSERT(VectorErrorCode::ERR_PARAM_INVALID, false) << "The transDataType is not supported";
    }
}

REGISTER_OPERATION_TILED_FUNC(OP_TRANSPOSE_MOVEOUT, Opcode::OP_TRANSPOSE_MOVEOUT, MoveOutOperationTileFunc);
REGISTER_OPERATION_TILED_FUNC(OP_TRANSPOSE_MOVEIN, Opcode::OP_TRANSPOSE_MOVEIN, MoveInOperationTileFunc);
REGISTER_OPERATION_TILED_FUNC(OP_TRANSPOSE_VNCHWCONV, Opcode::OP_TRANSPOSE_VNCHWCONV, VnchwconvOperationTileFunc);
REGISTER_OPERATION_TILED_FUNC(OP_EXPAND, Opcode::OP_EXPAND, ExpandOperationTileFunc);
REGISTER_OPERATION_TILED_FUNC(OP_CAST, Opcode::OP_CAST, CastOperationTileFunc);
REGISTER_OPERATION_TILED_FUNC(OP_VEC_DUP, Opcode::OP_VEC_DUP, FullOperationTileFunc);
REGISTER_OPERATION_TILED_FUNC(OP_NCHW2NC1HWC0, Opcode::OP_NCHW2NC1HWC0, TransDataTileFunc);
REGISTER_OPERATION_TILED_FUNC(OP_NCHW2Fractal_Z, Opcode::OP_NCHW2Fractal_Z, TransDataTileFunc);
REGISTER_OPERATION_TILED_FUNC(OP_NC1HWC02NCHW, Opcode::OP_NC1HWC02NCHW, TransDataTileFunc);
REGISTER_OPERATION_TILED_FUNC(OP_NCDHW2NDC1HWC0, Opcode::OP_NCDHW2NDC1HWC0, TransDataTileFunc);
REGISTER_OPERATION_TILED_FUNC(OP_NCDHW2FRACTAL_Z_3D, Opcode::OP_NCDHW2FRACTAL_Z_3D, TransDataTileFunc);
REGISTER_OPERATION_TILED_FUNC(OP_NDC1HWC02NCDHW, Opcode::OP_NDC1HWC02NCDHW, TransDataTileFunc);

} // namespace npu::tile_fwk
