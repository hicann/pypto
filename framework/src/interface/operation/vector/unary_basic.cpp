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
 * \file unary_basic.cpp
 * \brief
 */

#include "unary_tiled.h"
#include "binary.h"
#include "tensor_transformation.h"
#include "interface/utils/operator_tracer.h"
#include "passes/pass_utils/graph_utils.h"
#include "tilefwk/error_code.h"

namespace npu::tile_fwk {

Tensor Rsqrt(const Tensor& self, PrecisionType precisionType)
{
    DECLARE_TRACER();
    CheckTensorFormat(self.GetStorage(), {TileOpFormat::TILEOP_NZ}, "Rsqrt");

    std::unordered_set<DataType> supportedTypes = {DT_FP16, DT_BF16, DT_FP32};
    CheckTensorDataType(self.GetStorage(), supportedTypes, "Rsqrt");

    auto castSelf = self.GetStorage();
    if (self.GetDataType() != DataType::DT_FP32) {
        castSelf = CALL(CastOperation<CastOpType::CAST>, *Program::GetInstance().GetCurrentFunction(),
                        self.GetStorage(), DataType::DT_FP32, CastMode::CAST_NONE);
    }
    auto [sqrtResult, sqrtOp] = TensorUnaryOperationWithOp<UnaryOpType::SQRT>(
        *Program::GetInstance().GetCurrentFunction(), castSelf);
    sqrtOp->SetAttribute(OpAttributeKey::precisionType, static_cast<int64_t>(precisionType));
    auto ones = CALL(FullOperation, *Program::GetInstance().GetCurrentFunction(), Element(DataType::DT_FP32, 1.0),
                     SymbolicScalar(), DataType::DT_FP32, self.GetShape(), self.GetStorage()->GetDynValidShape());
    auto [divResult, divOp] = TensorBinaryOperationWithOp<BinaryOpType::DIV>(
        *Program::GetInstance().GetCurrentFunction(), ones, sqrtResult);
    divOp->SetAttribute(OpAttributeKey::precisionType, static_cast<int64_t>(precisionType));
    if (self.GetDataType() != DataType::DT_FP32) {
        RETURN_CALL(CastOperation<CastOpType::CAST>, *Program::GetInstance().GetCurrentFunction(), divResult,
                    self.GetDataType(), CastMode::CAST_NONE);
    }
    return divResult;
}

Tensor Sqrt(const Tensor& self, PrecisionType precisionType)
{
    DECLARE_TRACER();
    CheckTensorFormat(self.GetStorage(), {TileOpFormat::TILEOP_NZ}, "Sqrt");

    std::unordered_set<DataType> supportedTypes = {DT_FP16, DT_BF16, DT_FP32};
    CheckTensorDataType(self.GetStorage(), supportedTypes, "Sqrt");

    auto [result, op] = TensorUnaryOperationWithOp<UnaryOpType::SQRT>(*Program::GetInstance().GetCurrentFunction(),
                                                                      self.GetStorage());
    op->SetAttribute(OpAttributeKey::precisionType, static_cast<int64_t>(precisionType));
    return Tensor(result);
}

Tensor Relu(const Tensor& self)
{
    DECLARE_TRACER();
    CheckTensorFormat(self.GetStorage(), {TileOpFormat::TILEOP_NZ}, "Relu");

    static const std::unordered_set<DataType> RELU_A2A3_TYPES = {DT_FP16, DT_BF16, DT_FP32, DT_INT32, DT_INT16};
    static const std::unordered_set<DataType> RELU_A5_TYPES = {DT_FP16, DT_BF16, DT_FP32, DT_INT32, DT_INT16, DT_INT64};
    const auto& supportedTypes = GetSupportedDataTypesByArch(RELU_A2A3_TYPES, RELU_A5_TYPES);
    CheckTensorDataType(self.GetStorage(), supportedTypes, "Relu");
    RETURN_CALL(UnaryOperation<UnaryOpType::RELU>, *Program::GetInstance().GetCurrentFunction(), self.GetStorage());
}

Tensor BitwiseNot(const Tensor& self)
{
    DECLARE_TRACER();
    CheckTensorFormat(self.GetStorage(), {TileOpFormat::TILEOP_NZ}, "BitwiseNot");

    if (self.GetDataType() == DT_BOOL) {
        return LogicalNot(self);
    }
    static const std::unordered_set<DataType> BITWISE_A2A3_TYPES = {DT_INT16, DT_UINT16, DT_INT8,
                                                                    DT_UINT8, DT_INT32,  DT_UINT32};
    static const std::unordered_set<DataType> BITWISE_A5_TYPES = {DT_INT16, DT_UINT16, DT_INT8,  DT_UINT8,
                                                                  DT_INT32, DT_UINT32, DT_INT64, DT_UINT64};
    const auto& supportedTypes = GetSupportedDataTypesByArch(BITWISE_A2A3_TYPES, BITWISE_A5_TYPES);
    CheckTensorDataType(self.GetStorage(), supportedTypes, "BitwiseNot");
    RETURN_CALL(UnaryOperation<UnaryOpType::BITWISENOT>, *Program::GetInstance().GetCurrentFunction(),
                self.GetStorage());
}

Tensor Reciprocal(const Tensor& operand, PrecisionType precisionType)
{
    DECLARE_TRACER();
    CheckTensorFormat(operand.GetStorage(), {TileOpFormat::TILEOP_NZ}, "Reciprocal");

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
    CheckTensorFormat(self.GetStorage(), {TileOpFormat::TILEOP_NZ}, "Abs");

    static const std::unordered_set<DataType> ABS_A2A3_TYPES = {DT_FP16, DT_BF16, DT_FP32};
    static const std::unordered_set<DataType> ABS_A5_TYPES = {DT_FP16,  DT_BF16,  DT_FP32, DT_INT8,
                                                              DT_INT16, DT_INT32, DT_INT64};
    const auto& supportedTypes = GetSupportedDataTypesByArch(ABS_A2A3_TYPES, ABS_A5_TYPES);
    CheckTensorDataType(self.GetStorage(), supportedTypes, "Abs");
    RETURN_CALL(UnaryOperation<UnaryOpType::ABS>, *Program::GetInstance().GetCurrentFunction(), self.GetStorage());
}

Tensor Hub(const Tensor& self)
{
    DECLARE_TRACER();
    CheckTensorFormat(self.GetStorage(), {TileOpFormat::TILEOP_NZ}, "Hub");

    RETURN_CALL(UnaryOperation<UnaryOpType::HUB>, *Program::GetInstance().GetCurrentFunction(), self.GetStorage());
}

Tensor Neg(const Tensor& self)
{
    DECLARE_TRACER();
    CheckTensorFormat(self.GetStorage(), {TileOpFormat::TILEOP_NZ}, "Neg");

    static const std::unordered_set<DataType> NEG_A2A3_TYPES = {DT_FP16, DT_BF16, DT_INT16, DT_INT32, DT_FP32};
    static const std::unordered_set<DataType> NEG_A5_TYPES = {DT_FP16, DT_BF16, DT_INT16, DT_INT32, DT_FP32, DT_INT64};
    const auto& supportedTypes = GetSupportedDataTypesByArch(NEG_A2A3_TYPES, NEG_A5_TYPES);
    CheckTensorDataType(self.GetStorage(), supportedTypes, "NEG");
    CheckTensorDimRange(self.GetStorage(), 1, NUM_VALUE_4, "NEG");
    CheckTensorShapeSize(self.GetStorage(), "NEG");

    if (IsFloat(self.GetStorage()->Datatype())) {
        RETURN_CALL(BinaryOperationScalar<BinaryOpType::MUL>, *Program::GetInstance().GetCurrentFunction(),
                    self.GetStorage(), Element(self.GetStorage()->Datatype(), -1.0));
    } else {
        RETURN_CALL(BinaryOperationScalar<BinaryOpType::MUL>, *Program::GetInstance().GetCurrentFunction(),
                    self.GetStorage(), Element(self.GetStorage()->Datatype(), -1));
    }
}

void RsqrtOperationTileFunc(Function& function, const TileShape& tileShape,
                            const std::vector<LogicalTensorPtr>& iOperand,
                            const std::vector<LogicalTensorPtr>& oOperand, [[maybe_unused]] const Operation& op)
{
    UnaryOperationOperandCheck(iOperand, oOperand);
    return TiledUnaryOperation<UnaryOpType::RSQRT>(function, tileShape, iOperand[0], oOperand[0]);
}

void ReluOperationTileFunc(Function& function, const TileShape& tileShape,
                           const std::vector<LogicalTensorPtr>& iOperand, const std::vector<LogicalTensorPtr>& oOperand,
                           [[maybe_unused]] const Operation& op)
{
    UnaryOperationOperandCheck(iOperand, oOperand);
    return TiledUnaryOperation<UnaryOpType::RELU>(function, tileShape, iOperand[0], oOperand[0]);
}

void SqrtOperationTileFunc(Function& function, const TileShape& tileShape,
                           const std::vector<LogicalTensorPtr>& iOperand, const std::vector<LogicalTensorPtr>& oOperand,
                           [[maybe_unused]] const Operation& op)
{
    UnaryOperationOperandCheck(iOperand, oOperand);
    int64_t precisionType = static_cast<int64_t>(PrecisionType::INTRINSIC);
    if (op.HasAttr(OpAttributeKey::precisionType)) {
        precisionType = op.GetIntAttribute(OpAttributeKey::precisionType);
    }
    return TiledUnaryOperation<UnaryOpType::SQRT>(function, tileShape, iOperand[0], oOperand[0], 0, precisionType);
}

void BitwiseNotOperationTileFunc(Function& function, const TileShape& tileShape,
                                 const std::vector<LogicalTensorPtr>& iOperand,
                                 const std::vector<LogicalTensorPtr>& oOperand, [[maybe_unused]] const Operation& op)
{
    UnaryOperationOperandCheck(iOperand, oOperand);
    return TiledUnaryOperation<UnaryOpType::BITWISENOT>(function, tileShape, iOperand[0], oOperand[0]);
}

void ReciprocalOperationTileFunc(Function& function, const TileShape& tileShape,
                                 const std::vector<LogicalTensorPtr>& iOperand,
                                 const std::vector<LogicalTensorPtr>& oOperand, const Operation& op)
{
    UnaryOperationOperandCheck(iOperand, oOperand);
    int64_t precisionType = static_cast<int64_t>(PrecisionType::INTRINSIC);
    if (op.HasAttr(OpAttributeKey::precisionType)) {
        precisionType = op.GetIntAttribute(OpAttributeKey::precisionType);
    }
    return TiledUnaryOperation<UnaryOpType::RECIPROCAL>(function, tileShape, iOperand[0], oOperand[0], 0,
                                                        precisionType);
}

void AbsOperationTileFunc(Function& function, const TileShape& tileShape, const std::vector<LogicalTensorPtr>& iOperand,
                          const std::vector<LogicalTensorPtr>& oOperand, [[maybe_unused]] const Operation& op)
{
    UnaryOperationOperandCheck(iOperand, oOperand);
    return TiledUnaryOperation<UnaryOpType::ABS>(function, tileShape, iOperand[0], oOperand[0]);
}

void HubOperationTileFunc(Function& function, const TileShape& tileShape, const std::vector<LogicalTensorPtr>& iOperand,
                          const std::vector<LogicalTensorPtr>& oOperand, [[maybe_unused]] const Operation& op)
{
    UnaryOperationOperandCheck(iOperand, oOperand);
    return TiledUnaryOperation<UnaryOpType::HUB>(function, tileShape, iOperand[0], oOperand[0]);
}

REGISTER_OPERATION_TILED_FUNC(OP_RSQRT, Opcode::OP_RSQRT, RsqrtOperationTileFunc);
REGISTER_OPERATION_TILED_FUNC(OP_RELU, Opcode::OP_RELU, ReluOperationTileFunc);
REGISTER_OPERATION_TILED_FUNC(OP_SQRT, Opcode::OP_SQRT, SqrtOperationTileFunc);
REGISTER_OPERATION_TILED_FUNC(OP_BITWISENOT, Opcode::OP_BITWISENOT, BitwiseNotOperationTileFunc);
REGISTER_OPERATION_TILED_FUNC(OP_RECIPROCAL, Opcode::OP_RECIPROCAL, ReciprocalOperationTileFunc);
REGISTER_OPERATION_TILED_FUNC(OP_ABS, Opcode::OP_ABS, AbsOperationTileFunc);
REGISTER_OPERATION_TILED_FUNC(OP_HUB, Opcode::OP_HUB, HubOperationTileFunc);

} // namespace npu::tile_fwk
