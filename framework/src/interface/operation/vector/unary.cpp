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

namespace npu::tile_fwk {

void UnaryOperationOperandCheck(
    const std::vector<LogicalTensorPtr> &iOperand, const std::vector<LogicalTensorPtr> &oOperand) {
    ASSERT(iOperand.size() == 1) << "The input operand size should be 1";
    ASSERT(oOperand.size() == 1) << "The output operand size should be 1";
}

template <UnaryOpType T>
void TiledUnaryOperation(
    Function &function, const TileShape &tileShape, size_t cur, Input &input, const LogicalTensorPtr &result, uint32_t workspaceSize = 0) {
    if (cur == input.tensor.GetShape().size()) {
        auto tile = input.tensor.GetStorage()->View(function, input.tileInfo.shape, input.tileInfo.offset);
        auto resultTile = result->View(function, input.tileInfo.shape, input.tileInfo.offset);
        if (workspaceSize == 0) {
            function.AddOperation(GetUnaryOpNameCode<T>(), {tile}, {resultTile});
        } else {
            LogicalTensorPtr workspace = std::make_shared<LogicalTensor>(function, DT_UINT8, std::vector<int64_t>{workspaceSize});
            function.AddOperation(GetUnaryOpNameCode<T>(), {tile}, {resultTile, workspace});
        }
        return;
    }
    auto &vecTile = tileShape.GetVecTile();
    for (int i = 0; i < input.tensor.GetShape()[cur]; i += vecTile[cur]) {
        input.tileInfo.shape[cur] = std::min(input.tensor.GetShape()[cur] - i, vecTile[cur]);
        input.tileInfo.offset[cur] = i;
        TiledUnaryOperation<T>(function, tileShape, cur + 1, input, result, workspaceSize);
    }
}

template <UnaryOpType T>
void TiledUnaryOperation(
    Function &function, const TileShape &tileShape, const LogicalTensorPtr &operand, const LogicalTensorPtr &result, int32_t workspaceSize = 0) {
    ASSERT(operand->shape.size() == operand->offset.size()) << "The shape size of operand and offset must be equal";

    TileInfo tileInfo(result->shape.size(), result->offset.size());
    auto input = Input{operand, tileInfo};
    TiledUnaryOperation<T>(function, tileShape, 0, input, result, workspaceSize);
}

Tensor Exp(const Tensor &self) {
    DECLARE_TRACER();

    RETURN_CALL(UnaryOperation<UnaryOpType::EXP>, *Program::GetInstance().GetCurrentFunction(), self.GetStorage());
}

Tensor Ln(const Tensor &operand) {
    DECLARE_TRACER();

    RETURN_CALL(UnaryOperation<UnaryOpType::LN>, *Program::GetInstance().GetCurrentFunction(), operand.GetStorage());
}

Tensor IsFinite(const Tensor &self) {
    DECLARE_TRACER();
    std::vector<DataType> SUPPORT_FLOAT_TYPES = {DT_FP16, DT_FP32, DT_BF16};
    std::vector<DataType> SUPPORT_INT_TYPES = {DT_INT16, DT_INT4, DT_INT8, DT_INT32, DT_UINT16, DT_UINT32, DT_UINT8, DT_UINT64, DT_INT64};

    if (std::find(SUPPORT_INT_TYPES.begin(), SUPPORT_INT_TYPES.end(), self.GetDataType()) !=
           SUPPORT_INT_TYPES.end()) {
        Tensor halfResult = Full(Element(DT_FP16, 1.0f), DT_FP16, self.GetShape(), self.GetValidShape());
        Tensor result = Cast(halfResult, DT_UINT8, CastMode::CAST_NONE);
        return result;
    }

    ASSERT(std::find(SUPPORT_FLOAT_TYPES.begin(), SUPPORT_FLOAT_TYPES.end(), self.GetDataType()) !=
           SUPPORT_FLOAT_TYPES.end()) << "`IsFinite` only supports FP16/BF16/FP32 in float datatypes!";
    RETURN_CALL(UnaryOperation<UnaryOpType::ISFINITE>, *Program::GetInstance().GetCurrentFunction(), self.GetStorage(), DT_BOOL);
}

Tensor Rsqrt(const Tensor &self) {
    DECLARE_TRACER();

    auto castSelf = self.GetStorage();
    if (self.GetDataType() != DataType::DT_FP32) {
        castSelf = CALL(CastOperation<CastOpType::CAST>, *Program::GetInstance().GetCurrentFunction(), self.GetStorage(),
            DataType::DT_FP32, CastMode::CAST_NONE);
    }
    auto sqrtSelf = CALL(UnaryOperation<UnaryOpType::SQRT>, *Program::GetInstance().GetCurrentFunction(), castSelf);
    auto ones = CALL(FullOperation, *Program::GetInstance().GetCurrentFunction(), Element(DataType::DT_FP32, 1.0),
        SymbolicScalar(), DataType::DT_FP32, self.GetShape(), self.GetStorage()->GetDynValidShape());
    auto result = CALL(BinaryOperation<BinaryOpType::DIV>, *Program::GetInstance().GetCurrentFunction(), ones, sqrtSelf);
    if (self.GetDataType() != DataType::DT_FP32) {
        RETURN_CALL(CastOperation<CastOpType::CAST>, *Program::GetInstance().GetCurrentFunction(), result,
            self.GetDataType(), CastMode::CAST_NONE);
    }
    return result;
}

Tensor Sqrt(const Tensor &self) {
    DECLARE_TRACER();

    RETURN_CALL(UnaryOperation<UnaryOpType::SQRT>, *Program::GetInstance().GetCurrentFunction(), self.GetStorage());
}

Tensor Relu(const Tensor &self) {
    DECLARE_TRACER();
    RETURN_CALL(UnaryOperation<UnaryOpType::RELU>, *Program::GetInstance().GetCurrentFunction(), self.GetStorage());
}

Tensor Ceil(const Tensor &self) {
    DECLARE_TRACER();

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

Tensor Floor(const Tensor &self) {
    DECLARE_TRACER();

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

Tensor Trunc(const Tensor &self) {
    DECLARE_TRACER();

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

Tensor BitwiseNot(const Tensor &self) {
    DECLARE_TRACER();
    if (self.GetDataType() == DT_BOOL) {
        return LogicalNot(self);
    }
    RETURN_CALL(UnaryOperation<UnaryOpType::BITWISENOT>, *Program::GetInstance().GetCurrentFunction(), self.GetStorage());
}

Tensor Reciprocal(const Tensor &operand) {
    DECLARE_TRACER();

    RETURN_CALL(
        UnaryOperation<UnaryOpType::RECIPROCAL>, *Program::GetInstance().GetCurrentFunction(), operand.GetStorage());
}

Tensor Abs(const Tensor &self) {
    DECLARE_TRACER();

    RETURN_CALL(UnaryOperation<UnaryOpType::ABS>, *Program::GetInstance().GetCurrentFunction(), self.GetStorage());
}

Tensor Hub(const Tensor &self) {
    DECLARE_TRACER();

    RETURN_CALL(UnaryOperation<UnaryOpType::HUB>, *Program::GetInstance().GetCurrentFunction(), self.GetStorage());
}

Tensor Duplicate(const Tensor &operand) {
    DECLARE_TRACER();

    RETURN_CALL(
        UnaryOperation<UnaryOpType::DUPLICATE>, *Program::GetInstance().GetCurrentFunction(), operand.GetStorage());
}

void ExpOperationTileFunc(Function &function, const TileShape &tileShape, const std::vector<LogicalTensorPtr> &iOperand,
    const std::vector<LogicalTensorPtr> &oOperand, [[maybe_unused]] const Operation &op) {
    UnaryOperationOperandCheck(iOperand, oOperand);
    return TiledUnaryOperation<UnaryOpType::EXP>(function, tileShape, iOperand[0], oOperand[0]);
}

void RsqrtOperationTileFunc(Function &function, const TileShape &tileShape,
    const std::vector<LogicalTensorPtr> &iOperand, const std::vector<LogicalTensorPtr> &oOperand,
    [[maybe_unused]] const Operation &op) {
    UnaryOperationOperandCheck(iOperand, oOperand);
    return TiledUnaryOperation<UnaryOpType::RSQRT>(function, tileShape, iOperand[0], oOperand[0]);
}

void ReluOperationTileFunc(Function &function, const TileShape &tileShape,
    const std::vector<LogicalTensorPtr> &iOperand, const std::vector<LogicalTensorPtr> &oOperand,
    [[maybe_unused]] const Operation &op) {
    UnaryOperationOperandCheck(iOperand, oOperand);
    return TiledUnaryOperation<UnaryOpType::RELU>(function, tileShape, iOperand[0], oOperand[0]);
}

void CeilOperationTileFunc(Function &function, const TileShape &tileShape,
    const std::vector<LogicalTensorPtr> &iOperand, const std::vector<LogicalTensorPtr> &oOperand,
    [[maybe_unused]] const Operation &op) {
    UnaryOperationOperandCheck(iOperand, oOperand);
    return TiledUnaryOperation<UnaryOpType::CEIL>(function, tileShape, iOperand[0], oOperand[0]);
}

void FloorOperationTileFunc(Function &function, const TileShape &tileShape,
    const std::vector<LogicalTensorPtr> &iOperand, const std::vector<LogicalTensorPtr> &oOperand,
    [[maybe_unused]] const Operation &op) {
    UnaryOperationOperandCheck(iOperand, oOperand);
    return TiledUnaryOperation<UnaryOpType::FLOOR>(function, tileShape, iOperand[0], oOperand[0]);
}

void TruncOperationTileFunc(Function &function, const TileShape &tileShape,
    const std::vector<LogicalTensorPtr> &iOperand, const std::vector<LogicalTensorPtr> &oOperand,
    [[maybe_unused]] const Operation &op) {
    UnaryOperationOperandCheck(iOperand, oOperand);
    return TiledUnaryOperation<UnaryOpType::TRUNC>(function, tileShape, iOperand[0], oOperand[0]);
}

void SqrtOperationTileFunc(Function &function, const TileShape &tileShape,
    const std::vector<LogicalTensorPtr> &iOperand, const std::vector<LogicalTensorPtr> &oOperand,
    [[maybe_unused]] const Operation &op) {
    UnaryOperationOperandCheck(iOperand, oOperand);
    return TiledUnaryOperation<UnaryOpType::SQRT>(function, tileShape, iOperand[0], oOperand[0]);
}

void BitwiseNotOperationTileFunc(Function &function, const TileShape &tileShape,
    const std::vector<LogicalTensorPtr> &iOperand, const std::vector<LogicalTensorPtr> &oOperand,
    [[maybe_unused]] const Operation &op) {
    UnaryOperationOperandCheck(iOperand, oOperand);
    return TiledUnaryOperation<UnaryOpType::BITWISENOT>(function, tileShape, iOperand[0], oOperand[0]);
}

void ReciprocalOperationTileFunc(Function &function, const TileShape &tileShape,
    const std::vector<LogicalTensorPtr> &iOperand, const std::vector<LogicalTensorPtr> &oOperand,
    [[maybe_unused]] const Operation &op) {
    UnaryOperationOperandCheck(iOperand, oOperand);
    return TiledUnaryOperation<UnaryOpType::RECIPROCAL>(function, tileShape, iOperand[0], oOperand[0]);
}

void AbsOperationTileFunc(Function &function, const TileShape &tileShape, const std::vector<LogicalTensorPtr> &iOperand,
    const std::vector<LogicalTensorPtr> &oOperand, [[maybe_unused]] const Operation &op) {
    UnaryOperationOperandCheck(iOperand, oOperand);
    return TiledUnaryOperation<UnaryOpType::ABS>(function, tileShape, iOperand[0], oOperand[0]);
}

void LnOperationTileFunc(Function &function, const TileShape &tileShape, const std::vector<LogicalTensorPtr> &iOperand,
    const std::vector<LogicalTensorPtr> &oOperand, [[maybe_unused]] const Operation &op) {
    UnaryOperationOperandCheck(iOperand, oOperand);
    return TiledUnaryOperation<UnaryOpType::LN>(function, tileShape, iOperand[0], oOperand[0]);
}

void IsFiniteOperationTileFunc(Function &function, const TileShape &tileShape, const std::vector<LogicalTensorPtr> &iOperand,
    const std::vector<LogicalTensorPtr> &oOperand, [[maybe_unused]] const Operation &op) {
    UnaryOperationOperandCheck(iOperand, oOperand);
    Shape& shape = TileShape::Current().GetVecTile().tile;
    // tileShape 对应的中间变量结果，类型为 FP16
    uint32_t intermediateBytes = static_cast<int64_t>(BytesOf(DT_FP16)) * std::accumulate(shape.begin(), shape.end(), 1LL, std::multiplies<int64_t>());
    uint32_t workspaceSize = intermediateBytes;
    return TiledUnaryOperation<UnaryOpType::ISFINITE>(function, tileShape, iOperand[0], oOperand[0], workspaceSize);
}

void HubOperationTileFunc(Function &function, const TileShape &tileShape, const std::vector<LogicalTensorPtr> &iOperand,
    const std::vector<LogicalTensorPtr> &oOperand, [[maybe_unused]] const Operation &op) {
    UnaryOperationOperandCheck(iOperand, oOperand);
    return TiledUnaryOperation<UnaryOpType::HUB>(function, tileShape, iOperand[0], oOperand[0]);
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

} // namespace npu::tile_fwk