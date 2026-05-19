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
 * \file unary.cpp
 * \brief
 */

#include "binary.h"
#include "unary.h"
#include "tensor_transformation.h"
#include "interface/utils/operator_tracer.h"
#include "tilefwk/error_code.h"

namespace npu::tile_fwk {

void UnaryOperationOperandCheck(
    const std::vector<LogicalTensorPtr>& iOperand, const std::vector<LogicalTensorPtr>& oOperand)
{
    ASSERT(VectorErrorCode::ERR_PARAM_INVALID, iOperand.size() == 1) << "The input operand size should be 1";
    ASSERT(VectorErrorCode::ERR_PARAM_INVALID, oOperand.size() == 1) << "The output operand size should be 1";
}

template <UnaryOpType T>
void TiledUnaryOperation(
    Function& function, const TileShape& tileShape, size_t cur, Input& input, const LogicalTensorPtr& result,
    uint32_t workspaceSize = 0, int64_t precisionType = 0)
{
    if (cur == input.tensor.GetShape().size()) {
        auto tile = input.tensor.GetStorage()->View(function, input.tileInfo.shape, input.tileInfo.offset);
        auto resultTile = result->View(function, input.tileInfo.shape, input.tileInfo.offset);
        Operation* op = nullptr;
        if (workspaceSize == 0) {
            op = &function.AddOperation(GetUnaryOpNameCode<T>(), {tile}, {resultTile});
        } else {
            LogicalTensorPtr workspace =
                std::make_shared<LogicalTensor>(function, DT_UINT8, std::vector<int64_t>{workspaceSize});
            op = &function.AddOperation(GetUnaryOpNameCode<T>(), {tile}, {resultTile, workspace});
        }
        if (T == UnaryOpType::EXP || T == UnaryOpType::SQRT || T == UnaryOpType::LN || T == UnaryOpType::RECIPROCAL) {
            op->SetAttribute(OpAttributeKey::precisionType, precisionType);
        }
        return;
    }
    auto& vecTile = tileShape.GetVecTile();
    for (int i = 0; i < input.tensor.GetShape()[cur]; i += vecTile[cur]) {
        input.tileInfo.shape[cur] = std::min(input.tensor.GetShape()[cur] - i, vecTile[cur]);
        input.tileInfo.offset[cur] = i;
        TiledUnaryOperation<T>(function, tileShape, cur + 1, input, result, workspaceSize, precisionType);
    }
}

template <UnaryOpType T>
void TiledUnaryOperation(
    Function& function, const TileShape& tileShape, const LogicalTensorPtr& operand, const LogicalTensorPtr& result,
    int32_t workspaceSize = 0, int64_t precisionType = 0)
{
    ASSERT(VectorErrorCode::ERR_PARAM_INVALID, operand->shape.size() == operand->offset.size())
        << "The shape size of operand and offset must be equal";

    TileInfo tileInfo(result->shape.size(), result->offset.size());
    auto input = Input{operand, tileInfo};
    TiledUnaryOperation<T>(function, tileShape, 0, input, result, workspaceSize, precisionType);
}

Tensor Exp(const Tensor& self, PrecisionType precisionType)
{
    DECLARE_TRACER();
    std::unordered_set<DataType> supportedTypes = {DT_FP16, DT_BF16, DT_FP32};
    CheckTensorDataType(self.GetStorage(), supportedTypes, "Exp");

    auto [result, op] =
        TensorUnaryOperationWithOp<UnaryOpType::EXP>(*Program::GetInstance().GetCurrentFunction(), self.GetStorage());
    op->SetAttribute(OpAttributeKey::precisionType, static_cast<int64_t>(precisionType));
    return Tensor(result);
}

Tensor Ln(const Tensor& operand, PrecisionType precisionType)
{
    DECLARE_TRACER();
    std::unordered_set<DataType> supportedTypes = {DT_FP16, DT_BF16, DT_FP32};
    CheckTensorDataType(operand.GetStorage(), supportedTypes, "Ln");

    auto [result, op] =
        TensorUnaryOperationWithOp<UnaryOpType::LN>(*Program::GetInstance().GetCurrentFunction(), operand.GetStorage());
    op->SetAttribute(OpAttributeKey::precisionType, static_cast<int64_t>(precisionType));
    return Tensor(result);
}

Tensor IsFinite(const Tensor& self)
{
    DECLARE_TRACER();
    std::unordered_set<DataType> supportedTypes = {DT_FP16,  DT_FP32,   DT_BF16,   DT_INT16, DT_INT4,   DT_INT8,
                                                   DT_INT32, DT_UINT16, DT_UINT32, DT_UINT8, DT_UINT64, DT_INT64};
    CheckTensorDataType(self.GetStorage(), supportedTypes, "IsFinite");
    RETURN_CALL(
        UnaryOperation<UnaryOpType::ISFINITE>, *Program::GetInstance().GetCurrentFunction(), self.GetStorage(),
        DT_BOOL);
}

Tensor Rsqrt(const Tensor& self, PrecisionType precisionType)
{
    DECLARE_TRACER();
    std::unordered_set<DataType> supportedTypes = {DT_FP16, DT_BF16, DT_FP32};
    CheckTensorDataType(self.GetStorage(), supportedTypes, "Rsqrt");

    auto castSelf = self.GetStorage();
    if (self.GetDataType() != DataType::DT_FP32) {
        castSelf = CALL(
            CastOperation<CastOpType::CAST>, *Program::GetInstance().GetCurrentFunction(), self.GetStorage(),
            DataType::DT_FP32, CastMode::CAST_NONE);
    }
    auto [sqrtResult, sqrtOp] =
        TensorUnaryOperationWithOp<UnaryOpType::SQRT>(*Program::GetInstance().GetCurrentFunction(), castSelf);
    sqrtOp->SetAttribute(OpAttributeKey::precisionType, static_cast<int64_t>(precisionType));
    auto ones = CALL(
        FullOperation, *Program::GetInstance().GetCurrentFunction(), Element(DataType::DT_FP32, 1.0), SymbolicScalar(),
        DataType::DT_FP32, self.GetShape(), self.GetStorage()->GetDynValidShape());
    auto [divResult, divOp] =
        TensorBinaryOperationWithOp<BinaryOpType::DIV>(*Program::GetInstance().GetCurrentFunction(), ones, sqrtResult);
    divOp->SetAttribute(OpAttributeKey::precisionType, static_cast<int64_t>(precisionType));
    if (self.GetDataType() != DataType::DT_FP32) {
        RETURN_CALL(
            CastOperation<CastOpType::CAST>, *Program::GetInstance().GetCurrentFunction(), divResult,
            self.GetDataType(), CastMode::CAST_NONE);
    }
    return divResult;
}

Tensor Sqrt(const Tensor& self, PrecisionType precisionType)
{
    DECLARE_TRACER();
    std::unordered_set<DataType> supportedTypes = {DT_FP16, DT_BF16, DT_FP32};
    CheckTensorDataType(self.GetStorage(), supportedTypes, "Sqrt");

    auto [result, op] =
        TensorUnaryOperationWithOp<UnaryOpType::SQRT>(*Program::GetInstance().GetCurrentFunction(), self.GetStorage());
    op->SetAttribute(OpAttributeKey::precisionType, static_cast<int64_t>(precisionType));
    return Tensor(result);
}

Tensor Relu(const Tensor& self)
{
    DECLARE_TRACER();
    std::unordered_set<DataType> supportedTypes = {DT_FP16, DT_BF16, DT_FP32};
    CheckTensorDataType(self.GetStorage(), supportedTypes, "Relu");
    RETURN_CALL(UnaryOperation<UnaryOpType::RELU>, *Program::GetInstance().GetCurrentFunction(), self.GetStorage());
}

Tensor Ceil(const Tensor& self)
{
    DECLARE_TRACER();
    std::unordered_set<DataType> supportedTypes = {DT_FP16, DT_BF16, DT_INT16, DT_INT32, DT_FP32};
    CheckTensorDataType(self.GetStorage(), supportedTypes, "Ceil");

    auto castSelf = self.GetStorage();
    if (self.GetDataType() != DataType::DT_FP32) {
        castSelf = CALL(
            CastOperation<CastOpType::CAST>, *Program::GetInstance().GetCurrentFunction(), self.GetStorage(),
            DataType::DT_FP32, CastMode::CAST_NONE);
    }

    auto ceilResult = CALL(UnaryOperation<UnaryOpType::CEIL>, *Program::GetInstance().GetCurrentFunction(), castSelf);
    if (self.GetDataType() != DataType::DT_FP32) {
        RETURN_CALL(
            CastOperation<CastOpType::CAST>, *Program::GetInstance().GetCurrentFunction(), ceilResult,
            self.GetDataType(), CastMode::CAST_NONE);
    }
    return ceilResult;
}

Tensor Floor(const Tensor& self)
{
    DECLARE_TRACER();
    std::unordered_set<DataType> supportedTypes = {DT_FP16, DT_BF16, DT_INT16, DT_INT32, DT_FP32};
    CheckTensorDataType(self.GetStorage(), supportedTypes, "Floor");

    auto castSelf = self.GetStorage();
    if (self.GetDataType() != DataType::DT_FP32) {
        castSelf = CALL(
            CastOperation<CastOpType::CAST>, *Program::GetInstance().GetCurrentFunction(), self.GetStorage(),
            DataType::DT_FP32, CastMode::CAST_NONE);
    }

    auto floorResult = CALL(UnaryOperation<UnaryOpType::FLOOR>, *Program::GetInstance().GetCurrentFunction(), castSelf);
    if (self.GetDataType() != DataType::DT_FP32) {
        RETURN_CALL(
            CastOperation<CastOpType::CAST>, *Program::GetInstance().GetCurrentFunction(), floorResult,
            self.GetDataType(), CastMode::CAST_NONE);
    }
    return floorResult;
}

Tensor Trunc(const Tensor& self)
{
    DECLARE_TRACER();
    std::unordered_set<DataType> supportedTypes = {DT_FP16, DT_BF16, DT_INT16, DT_INT32, DT_FP32};
    CheckTensorDataType(self.GetStorage(), supportedTypes, "Trunc");

    auto castSelf = self.GetStorage();
    if (self.GetDataType() != DataType::DT_FP32) {
        castSelf = CALL(
            CastOperation<CastOpType::CAST>, *Program::GetInstance().GetCurrentFunction(), self.GetStorage(),
            DataType::DT_FP32, CastMode::CAST_NONE);
    }

    auto truncResult = CALL(UnaryOperation<UnaryOpType::TRUNC>, *Program::GetInstance().GetCurrentFunction(), castSelf);
    if (self.GetDataType() != DataType::DT_FP32) {
        RETURN_CALL(
            CastOperation<CastOpType::CAST>, *Program::GetInstance().GetCurrentFunction(), truncResult,
            self.GetDataType(), CastMode::CAST_NONE);
    }
    return truncResult;
}

Tensor BitwiseNot(const Tensor& self)
{
    DECLARE_TRACER();
    if (self.GetDataType() == DT_BOOL) {
        return LogicalNot(self);
    }
    std::unordered_set<DataType> supportedTypes = {DT_INT16, DT_UINT16};
    CheckTensorDataType(self.GetStorage(), supportedTypes, "BitwiseNot");
    RETURN_CALL(
        UnaryOperation<UnaryOpType::BITWISENOT>, *Program::GetInstance().GetCurrentFunction(), self.GetStorage());
}

Tensor Reciprocal(const Tensor& operand, PrecisionType precisionType)
{
    DECLARE_TRACER();
    std::unordered_set<DataType> supportedTypes = {DT_FP16, DT_BF16, DT_FP32};
    CheckTensorDataType(operand.GetStorage(), supportedTypes, "Reciprocal");

    auto [result, op] = TensorUnaryOperationWithOp<UnaryOpType::RECIPROCAL>(
        *Program::GetInstance().GetCurrentFunction(), operand.GetStorage());
    op->SetAttribute(OpAttributeKey::precisionType, static_cast<int64_t>(precisionType));
    return Tensor(result);
}

Tensor Abs(const Tensor& self)
{
    DECLARE_TRACER();
    std::unordered_set<DataType> supportedTypes = {DT_FP16, DT_BF16, DT_FP32};
    CheckTensorDataType(self.GetStorage(), supportedTypes, "Abs");
    RETURN_CALL(UnaryOperation<UnaryOpType::ABS>, *Program::GetInstance().GetCurrentFunction(), self.GetStorage());
}

Tensor Hub(const Tensor& self)
{
    DECLARE_TRACER();
    RETURN_CALL(UnaryOperation<UnaryOpType::HUB>, *Program::GetInstance().GetCurrentFunction(), self.GetStorage());
}

Tensor Duplicate(const Tensor& operand)
{
    DECLARE_TRACER();

    RETURN_CALL(
        UnaryOperation<UnaryOpType::DUPLICATE>, *Program::GetInstance().GetCurrentFunction(), operand.GetStorage());
}

Tensor Sinh(const Tensor& self)
{
    DECLARE_TRACER();
    
    std::unordered_set<DataType> supportedTypes = {DT_FP16, DT_FP32, DT_BF16};
    CheckTensorDataType(self.GetStorage(), supportedTypes, "SINH");
    CheckTensorDimRange(self.GetStorage(), 1, 4, "SINH");
    CheckTensorShapeSize(self.GetStorage(), "SINH");

    auto castSelf = Cast(self, DataType::DT_FP32);
    auto result = CALL(UnaryOperation<UnaryOpType::SINH>, *Program::GetInstance().GetCurrentFunction(), castSelf.GetStorage());
    auto castResult = Cast(result, self.GetDataType());
    return castResult;
}

Tensor Cosh(const Tensor& self)
{
    DECLARE_TRACER();

    std::unordered_set<DataType> supportedTypes = {DT_FP16, DT_FP32, DT_BF16};
    CheckTensorDataType(self.GetStorage(), supportedTypes, "COSH");
    CheckTensorDimRange(self.GetStorage(), 1, 4, "COSH");
    CheckTensorShapeSize(self.GetStorage(), "COSH");

    auto castSelf = Cast(self, DataType::DT_FP32);
    auto result = CALL(UnaryOperation<UnaryOpType::COSH>, *Program::GetInstance().GetCurrentFunction(), castSelf.GetStorage());
    auto castResult = Cast(result, self.GetDataType());
    return castResult;
}

Tensor Sin(const Tensor& self)
{
    DECLARE_TRACER();
    std::unordered_set<DataType> supportedTypes = {DT_FP16, DT_FP32};
    CheckTensorDataType(self.GetStorage(), supportedTypes, "Sin");
    CheckTensorDimRange(self.GetStorage(), 1, 4, "Sin");
    CheckTensorShapeSize(self.GetStorage(), "Sin");

    auto castSelf = Cast(self, DataType::DT_FP32);
    auto result = CALL(UnaryOperation<UnaryOpType::SIN>, *Program::GetInstance().GetCurrentFunction(), castSelf.GetStorage());
    auto castResult = Cast(result, self.GetDataType());
    return castResult;
}

Tensor Cos(const Tensor& self)
{
    DECLARE_TRACER();
    std::unordered_set<DataType> supportedTypes = {DT_FP16, DT_FP32};
    CheckTensorDataType(self.GetStorage(), supportedTypes, "Cos");
    CheckTensorDimRange(self.GetStorage(), 1, 4, "Cos");
    CheckTensorShapeSize(self.GetStorage(), "Cos");

    auto castSelf = Cast(self, DataType::DT_FP32);
    auto result = CALL(UnaryOperation<UnaryOpType::COS>, *Program::GetInstance().GetCurrentFunction(), castSelf.GetStorage());
    auto castResult = Cast(result, self.GetDataType());
    return castResult;
}

void ExpOperationTileFunc(
    Function& function, const TileShape& tileShape, const std::vector<LogicalTensorPtr>& iOperand,
    const std::vector<LogicalTensorPtr>& oOperand, [[maybe_unused]] const Operation& op)
{
    UnaryOperationOperandCheck(iOperand, oOperand);
    int64_t precisionType = static_cast<int64_t>(PrecisionType::INTRINSIC);
    if (op.HasAttr(OpAttributeKey::precisionType)) {
        precisionType = op.GetIntAttribute(OpAttributeKey::precisionType);
    }
    return TiledUnaryOperation<UnaryOpType::EXP>(function, tileShape, iOperand[0], oOperand[0], 0, precisionType);
}

void RsqrtOperationTileFunc(
    Function& function, const TileShape& tileShape, const std::vector<LogicalTensorPtr>& iOperand,
    const std::vector<LogicalTensorPtr>& oOperand, [[maybe_unused]] const Operation& op)
{
    UnaryOperationOperandCheck(iOperand, oOperand);
    return TiledUnaryOperation<UnaryOpType::RSQRT>(function, tileShape, iOperand[0], oOperand[0]);
}

void ReluOperationTileFunc(
    Function& function, const TileShape& tileShape, const std::vector<LogicalTensorPtr>& iOperand,
    const std::vector<LogicalTensorPtr>& oOperand, [[maybe_unused]] const Operation& op)
{
    UnaryOperationOperandCheck(iOperand, oOperand);
    return TiledUnaryOperation<UnaryOpType::RELU>(function, tileShape, iOperand[0], oOperand[0]);
}

void CeilOperationTileFunc(
    Function& function, const TileShape& tileShape, const std::vector<LogicalTensorPtr>& iOperand,
    const std::vector<LogicalTensorPtr>& oOperand, [[maybe_unused]] const Operation& op)
{
    UnaryOperationOperandCheck(iOperand, oOperand);
    return TiledUnaryOperation<UnaryOpType::CEIL>(function, tileShape, iOperand[0], oOperand[0]);
}

void FloorOperationTileFunc(
    Function& function, const TileShape& tileShape, const std::vector<LogicalTensorPtr>& iOperand,
    const std::vector<LogicalTensorPtr>& oOperand, [[maybe_unused]] const Operation& op)
{
    UnaryOperationOperandCheck(iOperand, oOperand);
    return TiledUnaryOperation<UnaryOpType::FLOOR>(function, tileShape, iOperand[0], oOperand[0]);
}

void TruncOperationTileFunc(
    Function& function, const TileShape& tileShape, const std::vector<LogicalTensorPtr>& iOperand,
    const std::vector<LogicalTensorPtr>& oOperand, [[maybe_unused]] const Operation& op)
{
    UnaryOperationOperandCheck(iOperand, oOperand);
    return TiledUnaryOperation<UnaryOpType::TRUNC>(function, tileShape, iOperand[0], oOperand[0]);
}

void SqrtOperationTileFunc(
    Function& function, const TileShape& tileShape, const std::vector<LogicalTensorPtr>& iOperand,
    const std::vector<LogicalTensorPtr>& oOperand, [[maybe_unused]] const Operation& op)
{
    UnaryOperationOperandCheck(iOperand, oOperand);
    int64_t precisionType = static_cast<int64_t>(PrecisionType::INTRINSIC);
    if (op.HasAttr(OpAttributeKey::precisionType)) {
        precisionType = op.GetIntAttribute(OpAttributeKey::precisionType);
    }
    return TiledUnaryOperation<UnaryOpType::SQRT>(function, tileShape, iOperand[0], oOperand[0], 0, precisionType);
}

void BitwiseNotOperationTileFunc(
    Function& function, const TileShape& tileShape, const std::vector<LogicalTensorPtr>& iOperand,
    const std::vector<LogicalTensorPtr>& oOperand, [[maybe_unused]] const Operation& op)
{
    UnaryOperationOperandCheck(iOperand, oOperand);
    return TiledUnaryOperation<UnaryOpType::BITWISENOT>(function, tileShape, iOperand[0], oOperand[0]);
}

void ReciprocalOperationTileFunc(
    Function& function, const TileShape& tileShape, const std::vector<LogicalTensorPtr>& iOperand,
    const std::vector<LogicalTensorPtr>& oOperand, const Operation& op)
{
    UnaryOperationOperandCheck(iOperand, oOperand);
    int64_t precisionType = static_cast<int64_t>(PrecisionType::INTRINSIC);
    if (op.HasAttr(OpAttributeKey::precisionType)) {
        precisionType = op.GetIntAttribute(OpAttributeKey::precisionType);
    }
    return TiledUnaryOperation<UnaryOpType::RECIPROCAL>(
        function, tileShape, iOperand[0], oOperand[0], 0, precisionType);
}

void AbsOperationTileFunc(
    Function& function, const TileShape& tileShape, const std::vector<LogicalTensorPtr>& iOperand,
    const std::vector<LogicalTensorPtr>& oOperand, [[maybe_unused]] const Operation& op)
{
    UnaryOperationOperandCheck(iOperand, oOperand);
    return TiledUnaryOperation<UnaryOpType::ABS>(function, tileShape, iOperand[0], oOperand[0]);
}

void LnOperationTileFunc(
    Function& function, const TileShape& tileShape, const std::vector<LogicalTensorPtr>& iOperand,
    const std::vector<LogicalTensorPtr>& oOperand, [[maybe_unused]] const Operation& op)
{
    UnaryOperationOperandCheck(iOperand, oOperand);
    int64_t precisionType = static_cast<int64_t>(PrecisionType::INTRINSIC);
    if (op.HasAttr(OpAttributeKey::precisionType)) {
        precisionType = op.GetIntAttribute(OpAttributeKey::precisionType);
    }
    return TiledUnaryOperation<UnaryOpType::LN>(function, tileShape, iOperand[0], oOperand[0], 0, precisionType);
}

void IsFiniteOperationTileFunc(
    Function& function, const TileShape& tileShape, const std::vector<LogicalTensorPtr>& iOperand,
    const std::vector<LogicalTensorPtr>& oOperand, [[maybe_unused]] const Operation& op)
{
    UnaryOperationOperandCheck(iOperand, oOperand);
    Shape& shape = TileShape::Current().GetVecTile().tile;
    // tileShape 对应的中间变量结果，类型为 FP16
    uint32_t intermediateBytes = static_cast<int64_t>(BytesOf(DT_FP16)) *
                                 std::accumulate(shape.begin(), shape.end(), 1LL, std::multiplies<int64_t>());
    uint32_t workspaceSize = intermediateBytes;
    return TiledUnaryOperation<UnaryOpType::ISFINITE>(function, tileShape, iOperand[0], oOperand[0], workspaceSize);
}

void HubOperationTileFunc(
    Function& function, const TileShape& tileShape, const std::vector<LogicalTensorPtr>& iOperand,
    const std::vector<LogicalTensorPtr>& oOperand, [[maybe_unused]] const Operation& op)
{
    UnaryOperationOperandCheck(iOperand, oOperand);
    return TiledUnaryOperation<UnaryOpType::HUB>(function, tileShape, iOperand[0], oOperand[0]);
}

void SinhOperationTileFunc(
    Function& function, const TileShape& tileShape, const std::vector<LogicalTensorPtr>& iOperand,
    const std::vector<LogicalTensorPtr>& oOperand, [[maybe_unused]] const Operation& op)
{
    UnaryOperationOperandCheck(iOperand, oOperand);
    auto shape = tileShape.GetVecTile();
    int dim = shape.size();
    auto alignSize = BLOCK_SIZE / BytesOf(DT_FP32);
    std::vector<int64_t> tmpShape = shape.tile;
    tmpShape[dim - 1] = AlignUp(tmpShape[dim - 1], alignSize) * NUM_VALUE_4;
    uint64_t intermediateBytes =
        std::accumulate(tmpShape.begin(), tmpShape.end(), 1LL, std::multiplies<int64_t>()) * BytesOf(DT_FP32);
    return TiledUnaryOperation<UnaryOpType::SINH>(function, tileShape, iOperand[0], oOperand[0], intermediateBytes);
}

void CoshOperationTileFunc(
    Function& function, const TileShape& tileShape, const std::vector<LogicalTensorPtr>& iOperand,
    const std::vector<LogicalTensorPtr>& oOperand, [[maybe_unused]] const Operation& op)
{
    UnaryOperationOperandCheck(iOperand, oOperand);
    auto shape = tileShape.GetVecTile();
    int dim = shape.size();
    auto alignSize = BLOCK_SIZE / BytesOf(DT_FP32);
    std::vector<int64_t> tmpShape = shape.tile;
    tmpShape[dim - 1] = AlignUp(tmpShape[dim - 1], alignSize);
    uint64_t intermediateBytes =
        std::accumulate(tmpShape.begin(), tmpShape.end(), 1LL, std::multiplies<int64_t>()) * BytesOf(DT_FP32);
    return TiledUnaryOperation<UnaryOpType::COSH>(function, tileShape, iOperand[0], oOperand[0], intermediateBytes);
}

void SinOperationTileFunc(
    Function& function, const TileShape& tileShape, const std::vector<LogicalTensorPtr>& iOperand,
    const std::vector<LogicalTensorPtr>& oOperand, [[maybe_unused]] const Operation& op)
{
    UnaryOperationOperandCheck(iOperand, oOperand);
    Shape& shape = TileShape::Current().GetVecTile().tile;
    std::vector<int64_t> tmpShape;
    tmpShape.assign(shape.begin(), shape.end());
    auto alignSize = BLOCK_SIZE / BytesOf(DT_FP32);
    tmpShape[tmpShape.size() - 1] = (tmpShape[tmpShape.size() - 1] + alignSize - 1) / alignSize * alignSize;
    // 3个中间变量
    uint64_t intermediateBytes = static_cast<int64_t>(BytesOf(DT_FP32)) * 3 *
                                std::accumulate(tmpShape.begin(), tmpShape.end(), 1LL, std::multiplies<int64_t>());

    return TiledUnaryOperation<UnaryOpType::SIN>(function, tileShape, iOperand[0], oOperand[0], intermediateBytes);
}

void CosOperationTileFunc(
    Function& function, const TileShape& tileShape, const std::vector<LogicalTensorPtr>& iOperand,
    const std::vector<LogicalTensorPtr>& oOperand, [[maybe_unused]] const Operation& op)
{
    UnaryOperationOperandCheck(iOperand, oOperand);
    Shape& shape = TileShape::Current().GetVecTile().tile;
    std::vector<int64_t> tmpShape;
    tmpShape.assign(shape.begin(), shape.end());
    auto alignSize = BLOCK_SIZE / BytesOf(DT_FP32);
    tmpShape[tmpShape.size() - 1] = (tmpShape[tmpShape.size() - 1] + alignSize - 1) / alignSize * alignSize;
    // 3个中间变量
    uint64_t intermediateBytes = static_cast<int64_t>(BytesOf(DT_FP32)) * 3 *
                                std::accumulate(tmpShape.begin(), tmpShape.end(), 1LL, std::multiplies<int64_t>());

    return TiledUnaryOperation<UnaryOpType::COS>(function, tileShape, iOperand[0], oOperand[0], intermediateBytes);
}

REGISTER_OPERATION_TILED_FUNC(OP_EXP, Opcode::OP_EXP, ExpOperationTileFunc);
REGISTER_OPERATION_TILED_FUNC(OP_RSQRT, Opcode::OP_RSQRT, RsqrtOperationTileFunc);
REGISTER_OPERATION_TILED_FUNC(OP_RELU, Opcode::OP_RELU, ReluOperationTileFunc);
REGISTER_OPERATION_TILED_FUNC(OP_SQRT, Opcode::OP_SQRT, SqrtOperationTileFunc);
REGISTER_OPERATION_TILED_FUNC(OP_CEIL, Opcode::OP_CEIL, CeilOperationTileFunc);
REGISTER_OPERATION_TILED_FUNC(OP_FLOOR, Opcode::OP_FLOOR, FloorOperationTileFunc);
REGISTER_OPERATION_TILED_FUNC(OP_TRUNC, Opcode::OP_TRUNC, TruncOperationTileFunc);
REGISTER_OPERATION_TILED_FUNC(OP_BITWISENOT, Opcode::OP_BITWISENOT, BitwiseNotOperationTileFunc);
REGISTER_OPERATION_TILED_FUNC(OP_RECIPROCAL, Opcode::OP_RECIPROCAL, ReciprocalOperationTileFunc);
REGISTER_OPERATION_TILED_FUNC(OP_ABS, Opcode::OP_ABS, AbsOperationTileFunc);
REGISTER_OPERATION_TILED_FUNC(OP_LN, Opcode::OP_LN, LnOperationTileFunc);
REGISTER_OPERATION_TILED_FUNC(OP_ISFINITE, Opcode::OP_ISFINITE, IsFiniteOperationTileFunc);
REGISTER_OPERATION_TILED_FUNC(OP_HUB, Opcode::OP_HUB, HubOperationTileFunc);
REGISTER_OPERATION_TILED_FUNC(OP_SINH, Opcode::OP_SINH, SinhOperationTileFunc);
REGISTER_OPERATION_TILED_FUNC(OP_COSH, Opcode::OP_COSH, CoshOperationTileFunc);
REGISTER_OPERATION_TILED_FUNC(OP_SIN, Opcode::OP_SIN, SinOperationTileFunc);
REGISTER_OPERATION_TILED_FUNC(OP_COS, Opcode::OP_COS, CosOperationTileFunc);
} // namespace npu::tile_fwk
