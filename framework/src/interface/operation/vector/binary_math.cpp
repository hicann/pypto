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
 * \file binary_math.cpp
 * \brief
 */
#include "tilefwk/data_type.h"
#include "unary.h"
#include "binary.h"
#include "binary_tiled.h"
#include "tensor_transformation.h"
#include "interface/utils/operator_tracer.h"
#include "interface/configs/config_manager.h"
#include "tilefwk/error_code.h"
#include "passes/tile_graph_pass/graph_constraint/axis_combine.h"

namespace npu::tile_fwk {

DataType GetPowRealResultDataType(DataType selfType, DataType otherType)
{
    if (selfType == DT_INT32) {
        return otherType;
    }
    if (otherType == DT_INT32) {
        return selfType;
    }
    if (selfType == DT_BF16) {
        return otherType == DT_FP16 ? DT_FP32 : otherType;
    }
    if (otherType == DT_BF16) {
        return selfType == DT_FP16 ? DT_FP32 : selfType;
    }
    return selfType == DT_FP16 && otherType == DT_FP16 ? DT_FP16 : DT_FP32;
}

DataType GetPowCalcResultDataType(DataType selfType, DataType otherType)
{
    if (selfType == DT_INT32 && otherType == DT_INT32) {
        return DT_INT32;
    }
    return DT_FP32;
}

LogicalTensorPtr CastToResultType(const LogicalTensorPtr& tensor, DataType originType, DataType resultType)
{
    if (originType == resultType) {
        return tensor;
    }
    RETURN_CALL(CastOperation<CastOpType::CAST>, *Program::GetInstance().GetCurrentFunction(), tensor, resultType,
                CastMode::CAST_NONE);
}

void PowCheck(const Tensor& self, const Tensor& other)
{
    std::unordered_set<DataType> supportedTypes = {DT_FP16, DT_BF16, DT_INT32, DT_FP32, DT_INT8, DT_UINT8, DT_INT16};
    CheckTensorDataType(self.GetStorage(), supportedTypes, "POW");
    CheckTensorDimRange(self.GetStorage(), 1, NUM_VALUE_4, "POW");
    CheckTensorShapeSize(self.GetStorage(), "POW");
    CheckTensorShapeSize(other.GetStorage(), "POW");
    CheckTensorsDimConsistency({self.GetStorage(), other.GetStorage()}, "POW");
    CheckTensorsShapeConsistencyOrBroadcast({self.GetStorage(), other.GetStorage()}, "POW");
    CheckTensorsFormatConsistency(self.GetStorage(), other.GetStorage(), "POW");
}

Tensor Pow(const Tensor& self, const Tensor& other, PrecisionType precisionType)
{
    DECLARE_TRACER();
    CheckTensorFormat(self.GetStorage(), {TileOpFormat::TILEOP_NZ}, "Pow");
    CheckTensorFormat(other.GetStorage(), {TileOpFormat::TILEOP_NZ}, "Pow");

    PowCheck(self, other);
    DataType selfType = self.GetDataType();
    DataType otherType = other.GetDataType();
    if (IsInteger(selfType) && IsInteger(otherType)) {
        CheckTensorsDataTypeConsistency(self.GetStorage(), other.GetStorage(), "Pow");
        auto [result, op] = TensorBinaryOperationWithOp<BinaryOpType::POW>(*Program::GetInstance().GetCurrentFunction(),
                                                                           self.GetStorage(), other.GetStorage());
        op->SetAttribute(OpAttributeKey::precisionType, static_cast<int64_t>(PrecisionType::INTRINSIC));
        return result;
    }
    DataType realResultType = GetPowRealResultDataType(selfType, otherType);
    DataType calcResultType = GetPowCalcResultDataType(selfType, otherType);
    auto selfSt = CastToResultType(self.GetStorage(), selfType, calcResultType);
    auto otherSt = CastToResultType(other.GetStorage(), otherType, calcResultType);
    auto [result, op] = TensorBinaryOperationWithOp<BinaryOpType::POW>(*Program::GetInstance().GetCurrentFunction(),
                                                                       selfSt, otherSt);
    op->SetAttribute(OpAttributeKey::precisionType, static_cast<int64_t>(precisionType));
    if (realResultType != calcResultType) {
        RETURN_CALL(CastOperation<CastOpType::CAST>, *Program::GetInstance().GetCurrentFunction(), result,
                    realResultType, CastMode::CAST_NONE);
    }
    return result;
}

Tensor Atan2(const Tensor& y, const Tensor& x)
{
    DECLARE_TRACER();
    CheckTensorFormat(y.GetStorage(), {TileOpFormat::TILEOP_NZ}, "Atan2");
    CheckTensorFormat(x.GetStorage(), {TileOpFormat::TILEOP_NZ}, "Atan2");

    CheckTensorsDataTypeConsistency(y.GetStorage(), x.GetStorage(), "ATAN2");
    std::unordered_set<DataType> supportedTypes = {DT_FP32, DT_FP16, DT_BF16};
    CheckTensorDataType(y.GetStorage(), supportedTypes, "ATAN2");
    auto castY = y.GetStorage();
    auto castX = x.GetStorage();
    DataType dataType = y.GetDataType();
    if (dataType != DataType::DT_FP32) {
        castY = CALL(CastOperation<CastOpType::CAST>, *Program::GetInstance().GetCurrentFunction(), y.GetStorage(),
                     DataType::DT_FP32, CastMode::CAST_NONE);
        castX = CALL(CastOperation<CastOpType::CAST>, *Program::GetInstance().GetCurrentFunction(), x.GetStorage(),
                     DataType::DT_FP32, CastMode::CAST_NONE);
    }
    auto res = CALL(BinaryOperation<BinaryOpType::ATAN2>, *Program::GetInstance().GetCurrentFunction(), castY, castX);
    if (dataType != DataType::DT_FP32) {
        RETURN_CALL(CastOperation<CastOpType::CAST>, *Program::GetInstance().GetCurrentFunction(), res, dataType,
                    CastMode::CAST_NONE);
    }
    return res;
}

Tensor CopySign(const Tensor& self, const Tensor& other)
{
    DECLARE_TRACER();
    CheckTensorFormat(self.GetStorage(), {TileOpFormat::TILEOP_NZ}, "CopySign");
    CheckTensorFormat(other.GetStorage(), {TileOpFormat::TILEOP_NZ}, "CopySign");

    CheckTensorsDataTypeConsistency(self.GetStorage(), other.GetStorage(), "COPYSIGN");
    std::unordered_set<DataType> supportedTypes = {DT_FP16, DT_BF16, DT_INT16, DT_INT32, DT_FP32};
    CheckTensorDataType(self.GetStorage(), supportedTypes, "COPYSIGN");

    DataType selfDType = self.GetDataType();
    DataType otherDType = other.GetDataType();
    Tensor castSelf = self;
    Tensor castOther = other;
    if (selfDType == DT_INT16 || selfDType == DT_INT32) {
        castSelf = CALL(CastOperation<CastOpType::CAST>, *Program::GetInstance().GetCurrentFunction(),
                        self.GetStorage(), DataType::DT_FP32, CastMode::CAST_NONE);
    }
    if (otherDType == DT_INT16 || otherDType == DT_INT32) {
        castOther = CALL(CastOperation<CastOpType::CAST>, *Program::GetInstance().GetCurrentFunction(),
                         other.GetStorage(), DataType::DT_FP32, CastMode::CAST_NONE);
    }
    RETURN_CALL(BinaryOperation<BinaryOpType::COPYSIGN>, *Program::GetInstance().GetCurrentFunction(), castSelf,
                castOther);
}

REGISTER_OPERATION_TILED_FUNC(OP_POW, Opcode::OP_POW, BinaryOperationTileFunc<BinaryOpType::POW>);
REGISTER_OPERATION_TILED_FUNC(OP_COPYSIGN, Opcode::OP_COPYSIGN, BinaryOperationTileFunc<BinaryOpType::COPYSIGN>);
REGISTER_OPERATION_TILED_FUNC(OP_ATAN2, Opcode::OP_ATAN2, BinaryOperationTileFunc<BinaryOpType::ATAN2>);

} // namespace npu::tile_fwk
