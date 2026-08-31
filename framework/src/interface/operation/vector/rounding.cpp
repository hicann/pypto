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
 * \file rounding.cpp
 * \brief
 */

#include "unary_tiled.h"
#include "binary.h"
#include "tensor_transformation.h"
#include "interface/utils/operator_tracer.h"
#include "passes/pass_utils/graph_utils.h"
#include "tilefwk/error_code.h"

namespace npu::tile_fwk {

Tensor Ceil(const Tensor& self)
{
    DECLARE_TRACER();
    CheckTensorFormat(self.GetStorage(), {TileOpFormat::TILEOP_NZ}, "Ceil");

    std::unordered_set<DataType> supportedTypes = {DT_FP16, DT_BF16, DT_INT16, DT_INT32, DT_FP32};
    CheckTensorDataType(self.GetStorage(), supportedTypes, "Ceil");

    if (self.GetDataType() == DataType::DT_INT16 || self.GetDataType() == DataType::DT_INT32) {
        RETURN_CALL(UnaryOperation<UnaryOpType::CEIL>, *Program::GetInstance().GetCurrentFunction(), self.GetStorage());
    }

    auto castSelf = self.GetStorage();
    if (self.GetDataType() != DataType::DT_FP32) {
        castSelf = CALL(CastOperation<CastOpType::CAST>, *Program::GetInstance().GetCurrentFunction(),
                        self.GetStorage(), DataType::DT_FP32, CastMode::CAST_NONE);
    }

    auto ceilResult = CALL(UnaryOperation<UnaryOpType::CEIL>, *Program::GetInstance().GetCurrentFunction(), castSelf);
    if (self.GetDataType() != DataType::DT_FP32) {
        RETURN_CALL(CastOperation<CastOpType::CAST>, *Program::GetInstance().GetCurrentFunction(), ceilResult,
                    self.GetDataType(), CastMode::CAST_NONE);
    }
    return ceilResult;
}

Tensor Floor(const Tensor& self)
{
    DECLARE_TRACER();
    CheckTensorFormat(self.GetStorage(), {TileOpFormat::TILEOP_NZ}, "Floor");

    std::unordered_set<DataType> supportedTypes = {DT_FP16, DT_BF16, DT_INT16, DT_INT32, DT_FP32};
    CheckTensorDataType(self.GetStorage(), supportedTypes, "Floor");

    if (self.GetDataType() == DataType::DT_INT16 || self.GetDataType() == DataType::DT_INT32) {
        RETURN_CALL(UnaryOperation<UnaryOpType::FLOOR>, *Program::GetInstance().GetCurrentFunction(),
                    self.GetStorage());
    }

    auto castSelf = self.GetStorage();
    if (self.GetDataType() != DataType::DT_FP32) {
        castSelf = CALL(CastOperation<CastOpType::CAST>, *Program::GetInstance().GetCurrentFunction(),
                        self.GetStorage(), DataType::DT_FP32, CastMode::CAST_NONE);
    }

    auto floorResult = CALL(UnaryOperation<UnaryOpType::FLOOR>, *Program::GetInstance().GetCurrentFunction(), castSelf);
    if (self.GetDataType() != DataType::DT_FP32) {
        RETURN_CALL(CastOperation<CastOpType::CAST>, *Program::GetInstance().GetCurrentFunction(), floorResult,
                    self.GetDataType(), CastMode::CAST_NONE);
    }
    return floorResult;
}

Tensor Trunc(const Tensor& self)
{
    DECLARE_TRACER();
    CheckTensorFormat(self.GetStorage(), {TileOpFormat::TILEOP_NZ}, "Trunc");

    std::unordered_set<DataType> supportedTypes = {DT_FP16, DT_BF16, DT_INT16, DT_INT32, DT_FP32};
    CheckTensorDataType(self.GetStorage(), supportedTypes, "Trunc");

    if (self.GetDataType() == DataType::DT_INT16 || self.GetDataType() == DataType::DT_INT32) {
        RETURN_CALL(UnaryOperation<UnaryOpType::TRUNC>, *Program::GetInstance().GetCurrentFunction(),
                    self.GetStorage());
    }

    auto castSelf = self.GetStorage();
    if (self.GetDataType() != DataType::DT_FP32) {
        castSelf = CALL(CastOperation<CastOpType::CAST>, *Program::GetInstance().GetCurrentFunction(),
                        self.GetStorage(), DataType::DT_FP32, CastMode::CAST_NONE);
    }

    auto truncResult = CALL(UnaryOperation<UnaryOpType::TRUNC>, *Program::GetInstance().GetCurrentFunction(), castSelf);
    if (self.GetDataType() != DataType::DT_FP32) {
        RETURN_CALL(CastOperation<CastOpType::CAST>, *Program::GetInstance().GetCurrentFunction(), truncResult,
                    self.GetDataType(), CastMode::CAST_NONE);
    }
    return truncResult;
}

void CeilOperationTileFunc(Function& function, const TileShape& tileShape,
                           const std::vector<LogicalTensorPtr>& iOperand, const std::vector<LogicalTensorPtr>& oOperand,
                           [[maybe_unused]] const Operation& op)
{
    UnaryOperationOperandCheck(iOperand, oOperand);
    return TiledUnaryOperation<UnaryOpType::CEIL>(function, tileShape, iOperand[0], oOperand[0]);
}

void FloorOperationTileFunc(Function& function, const TileShape& tileShape,
                            const std::vector<LogicalTensorPtr>& iOperand,
                            const std::vector<LogicalTensorPtr>& oOperand, [[maybe_unused]] const Operation& op)
{
    UnaryOperationOperandCheck(iOperand, oOperand);
    return TiledUnaryOperation<UnaryOpType::FLOOR>(function, tileShape, iOperand[0], oOperand[0]);
}

void TruncOperationTileFunc(Function& function, const TileShape& tileShape,
                            const std::vector<LogicalTensorPtr>& iOperand,
                            const std::vector<LogicalTensorPtr>& oOperand, [[maybe_unused]] const Operation& op)
{
    UnaryOperationOperandCheck(iOperand, oOperand);
    return TiledUnaryOperation<UnaryOpType::TRUNC>(function, tileShape, iOperand[0], oOperand[0]);
}

Tensor TensorRound(Function& function, const LogicalTensorPtr& self, const int& decimals = 0)
{
    auto result = std::make_shared<LogicalTensor>(function, self->Datatype(), self->GetShape(),
                                                  self->GetDynValidShape());
    auto& op = function.AddOperation(Opcode::OP_ROUND, {self}, {result});
    op.SetAttribute(OP_ATTR_PREFIX + "decimals", decimals);
    return result;
}

Tensor Round(const Tensor& self, const int& decimals)
{
    DECLARE_TRACER();
    CheckTensorFormat(self.GetStorage(), {TileOpFormat::TILEOP_NZ}, "Round");

    static const std::unordered_set<DataType> ROUND_A2A3_TYPES = {DT_FP32, DT_FP16, DT_BF16, DT_INT32, DT_INT16};
    static const std::unordered_set<DataType> ROUND_A5_TYPES = {DT_FP32,  DT_FP16,  DT_BF16,
                                                                DT_INT32, DT_INT16, DT_INT64};
    const auto& supportedTypes = GetSupportedDataTypesByArch(ROUND_A2A3_TYPES, ROUND_A5_TYPES);
    CheckTensorDataType(self.GetStorage(), supportedTypes, "ROUND");
    CheckTensorDimRange(self.GetStorage(), 1, NUM_VALUE_4, "ROUND");
    CheckTensorShapeSize(self.GetStorage(), "ROUND");

    RETURN_CALL(Round, *Program::GetInstance().GetCurrentFunction(), self.GetStorage(), decimals);
}

void TiledRound(Function& function, const TileShape& tileShape, size_t cur, Input& input,
                const LogicalTensorPtr& result, const int& decimals = 0)
{
    if (cur == input.tensor.GetShape().size()) {
        auto tile = input.tensor.GetStorage()->View(function, input.tileInfo.shape, input.tileInfo.offset);
        auto resultTile = result->View(function, input.tileInfo.shape, input.tileInfo.offset);
        std::vector<int64_t> srcTileShape(input.tileInfo.shape);
        auto tileShapeLen = srcTileShape.size();
        CHECK(VectorErrorCode::ERR_PARAM_INVALID, SHAPE_DIM1 <= tileShapeLen && tileShapeLen <= SHAPE_DIM4)
            << "Length of tile shape only supports 1~4";
        std::vector<int64_t> tmpShape;
        if (result->Datatype() == DT_FP32) {
            tmpShape = {BLOCK_SIZE / sizeof(float)};
        } else {
            if (srcTileShape.size() == 1) {
                tmpShape.assign(srcTileShape.end() - SHAPE_DIM1, srcTileShape.end());
            } else {
                tmpShape.assign(srcTileShape.end() - SHAPE_DIM2, srcTileShape.end());
            }
            auto alignSize = BLOCK_SIZE / BytesOf(DT_FP32);
            tmpShape[tmpShape.size() - 1] = (tmpShape[tmpShape.size() - 1] + alignSize - 1) / alignSize * alignSize;
        }
        auto tmpTensor = std::make_shared<LogicalTensor>(function, DT_FP32, tmpShape);
        auto& newOp = function.AddOperation(Opcode::OP_ROUND, {tile}, {resultTile, tmpTensor});
        float powDecimals = std::pow(static_cast<float>(NUM_VALUE_10), static_cast<float>(decimals));
        const int32_t maxFp32Len = 38;
        if (decimals > maxFp32Len) {
            powDecimals = INFINITY;
        }
        newOp.SetAttribute(OP_ATTR_PREFIX + "decimals", decimals);
        newOp.SetAttribute(OpAttributeKey::scalar, Element(DataType::DT_FP32, powDecimals));
        return;
    }
    auto& vecTile = tileShape.GetVecTile();
    for (int i = 0; i < input.tensor.GetShape()[cur]; i += vecTile[cur]) {
        input.tileInfo.shape[cur] = std::min(input.tensor.GetShape()[cur] - i, vecTile[cur]);
        input.tileInfo.offset[cur] = i;
        TiledRound(function, tileShape, cur + 1, input, result, decimals);
    }
}

void TiledRound(Function& function, const TileShape& tileShape, const LogicalTensorPtr& operand,
                const LogicalTensorPtr& result, const int& decimals = 0)
{
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, operand->shape.size() == operand->offset.size())
        << "The shape size of operand and offset must be equal";

    TileInfo tileInfo(result->shape.size(), result->offset.size());
    auto input = Input{operand, tileInfo};

    TiledRound(function, tileShape, 0, input, result, decimals);
}

void RoundOperationTileFunc(Function& function, const TileShape& tileShape,
                            const std::vector<LogicalTensorPtr>& iOperand,
                            const std::vector<LogicalTensorPtr>& oOperand, [[maybe_unused]] const Operation& op)
{
    int decimals = op.GetIntAttribute(OP_ATTR_PREFIX + "decimals");
    TiledRound(function, tileShape, iOperand[0], oOperand[0], decimals);
}

REGISTER_OPERATION_TILED_FUNC(OP_CEIL, Opcode::OP_CEIL, CeilOperationTileFunc);
REGISTER_OPERATION_TILED_FUNC(OP_FLOOR, Opcode::OP_FLOOR, FloorOperationTileFunc);
REGISTER_OPERATION_TILED_FUNC(OP_TRUNC, Opcode::OP_TRUNC, TruncOperationTileFunc);
REGISTER_OPERATION_TILED_FUNC(OP_ROUND, Opcode::OP_ROUND, RoundOperationTileFunc);

} // namespace npu::tile_fwk
