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
 * \file compare.cpp
 * \brief
 */

#include "binary.h"
#include "tensor_transformation.h"
#include "interface/utils/operator_tracer.h"
#include "passes/pass_utils/graph_utils.h"

namespace npu::tile_fwk {

void TiledCompareOperationImpl(Function &function, const TileShape &tileShape, size_t cur, Input &input1, Input &input2,
    const LogicalTensorPtr &result, TileInfo &resultTileInfo, OpType operation, OutType mode) {
    if (cur == result->shape.size()) {
        auto inputTile1 = input1.tensor.GetStorage()->View(function, input1.tileInfo.shape, input1.tileInfo.offset);
        auto inputTile2 = input2.tensor.GetStorage()->View(function, input2.tileInfo.shape, input2.tileInfo.offset);
        auto resultTile = result->View(function, resultTileInfo.shape, resultTileInfo.offset);

        LogicalTensorPtr convertedTile1 = inputTile1;
        LogicalTensorPtr convertedTile2 = inputTile2;

        if (inputTile1->Datatype() == DT_BF16) {
            convertedTile1 = std::make_shared<LogicalTensor>(function, DT_FP32, inputTile1->GetShape());
            Operation &castOp1 = function.AddOperation(Opcode::OP_CAST, {inputTile1}, {convertedTile1});
            castOp1.SetAttribute(OP_ATTR_PREFIX + "mode", CastMode::CAST_NONE);
            convertedTile2 = std::make_shared<LogicalTensor>(function, DT_FP32, inputTile2->GetShape());
            Operation &castOp2 = function.AddOperation(Opcode::OP_CAST, {inputTile2}, {convertedTile2});
            castOp2.SetAttribute(OP_ATTR_PREFIX + "mode", CastMode::CAST_NONE);
        }

        const int64_t COUNT_MODE_SIZE = 4096;
        size_t element_size = BytesOf(input1.tensor.GetDataType());
        if (inputTile1->Datatype() == DT_BF16) {
            element_size = BytesOf(DT_FP32);
        }
        ASSERT(element_size != 0) << "Element size cannot be zero.";
        int64_t elements_per_chunk = COUNT_MODE_SIZE / element_size;
        int64_t vcmp_bits_size = (elements_per_chunk + 7) / 8;

        const size_t ALIGN_SIZE = 32;

        size_t vcmpBitResult_size = ((vcmp_bits_size + ALIGN_SIZE - 1) / ALIGN_SIZE) * ALIGN_SIZE;
        size_t array_size = elements_per_chunk * element_size;
        size_t aligned_array_size = ((array_size + ALIGN_SIZE - 1) / ALIGN_SIZE) * ALIGN_SIZE;

        size_t total_bytes = vcmpBitResult_size + 3 * aligned_array_size + ALIGN_SIZE;
        std::vector<int64_t> tmp_shape({static_cast<int64_t>(total_bytes)});
        auto tmp_tensor = std::make_shared<LogicalTensor>(function, DT_UINT8, tmp_shape);

        auto &op = function.AddOperation(Opcode::OP_CMP, {convertedTile1, convertedTile2}, {resultTile, tmp_tensor});

        op.SetAttribute(OP_ATTR_PREFIX + "cmp_operation", static_cast<int64_t>(operation));
        op.SetAttribute(OP_ATTR_PREFIX + "cmp_mode", static_cast<int64_t>(mode));
        return;
    }

    auto &vecTile = tileShape.GetVecTile();
    int64_t step = vecTile[cur];

    if (mode == OutType::BIT && cur == result->shape.size() - 1) {
        step = vecTile[cur] / NUM_VALUE_8;
        if (step < 1) step = 1;

        int64_t actualInputStep = step * NUM_VALUE_8;

        for (int i = 0; i < result->shape[cur]; i += step) {
            resultTileInfo.offset[cur] = i;
            resultTileInfo.shape[cur] = std::min(result->shape[cur] - i, step);

            input1.tileInfo.offset[cur] = (i * NUM_VALUE_8) % input1.tensor.GetShape()[cur];
            input1.tileInfo.shape[cur] =
                std::min(input1.tensor.GetShape()[cur] - input1.tileInfo.offset[cur], actualInputStep);
            input2.tileInfo.offset[cur] = (i * NUM_VALUE_8) % input2.tensor.GetShape()[cur];
            input2.tileInfo.shape[cur] =
                std::min(input2.tensor.GetShape()[cur] - input2.tileInfo.offset[cur], actualInputStep);

            TiledCompareOperationImpl(
                function, tileShape, cur + 1, input1, input2, result, resultTileInfo, operation, mode);
        }
    } else {
        for (int i = 0; i < result->shape[cur]; i += step) {
            resultTileInfo.offset[cur] = i;
            resultTileInfo.shape[cur] = std::min(result->shape[cur] - i, step);

            input1.tileInfo.offset[cur] = i % input1.tensor.GetShape()[cur];
            input1.tileInfo.shape[cur] = std::min(input1.tensor.GetShape()[cur] - input1.tileInfo.offset[cur], step);

            input2.tileInfo.offset[cur] = i % input2.tensor.GetShape()[cur];
            input2.tileInfo.shape[cur] = std::min(input2.tensor.GetShape()[cur] - input2.tileInfo.offset[cur], step);

            TiledCompareOperationImpl(
                function, tileShape, cur + 1, input1, input2, result, resultTileInfo, operation, mode);
        }
    }
}

void TiledCompareOperation(Function &function, const TileShape &tileShape, LogicalTensorPtr operand1,
    LogicalTensorPtr operand2, const LogicalTensorPtr &result, OpType operation, OutType mode) {
    auto broadcastOperand = [&](LogicalTensorPtr &operand, LogicalTensorPtr &other) {
        auto dstShape = result->shape;
        if (mode == OutType::BIT) {
            dstShape[dstShape.size() - 1] *= NUM_VALUE_8;
        }
        if (operand->shape == dstShape) {
            return;
        }
        auto expanded = std::make_shared<LogicalTensor>(function, operand->Datatype(), dstShape);
        Expand(function, tileShape, operand, {other}, expanded);
        operand = expanded;
    };
    broadcastOperand(operand1, operand2);
    broadcastOperand(operand2, operand1);

    TileInfo tileInfo1(result->shape.size(), result->offset.size());
    TileInfo tileInfo2(result->shape.size(), result->offset.size());
    TileInfo resultTileInfo(result->shape.size(), result->offset.size());
    auto input1 = Input{operand1, tileInfo1};
    auto input2 = Input{operand2, tileInfo2};

    TiledCompareOperationImpl(function, tileShape, 0, input1, input2, result, resultTileInfo, operation, mode);
}

LogicalTensorPtr TensorCompareOperation(
    Function &function, const Tensor &self, const Tensor &other, OpType operation, OutType mode) {
    auto operandT1 = self.GetStorage();
    auto operandT2 = other.GetStorage();
    if (operandT1->shape.size() != operandT2->shape.size()) {
        std::vector<int> broadCastShape = GetBroadCastShape(operandT1, operandT2);
        operandT1 = BinaryOperationBroadCast(operandT1, broadCastShape);
        operandT2 = BinaryOperationBroadCast(operandT2, broadCastShape);
    }
    std::vector<SymbolicScalar> resultValidShape;
    std::vector<int64_t> resultShape = BinaryOperationResultShape(operandT1, operandT2);
    if (!operandT1->GetDynValidShape().empty() && !operandT2->GetDynValidShape().empty()) {
        for (size_t i = 0; i < resultShape.size(); ++i) {
            if (resultShape[i] == operandT1->shape[i]) {
                resultValidShape.push_back(operandT1->GetDynValidShape()[i]);
            } else {
                resultValidShape.push_back(operandT2->GetDynValidShape()[i]);
            }
        }
    }
    auto resultType = DT_BOOL;
    if (mode == OutType::BIT) {
        resultType = DT_UINT8;
        if (!resultShape.empty() && resultShape.back() % NUM_VALUE_8 != 0) {
            ALOG_ERROR_F("Last dimension must be divisible by 8 in BIT mode");
        }
        if (!resultShape.empty()) {
            resultShape.back() /= NUM_VALUE_8;
            if (!resultValidShape.empty()) {
                resultValidShape.back() = resultValidShape.back() / NUM_VALUE_8;
            }
        }
    }
    auto result = std::make_shared<LogicalTensor>(function, resultType, resultShape, resultValidShape);
    auto &op = function.AddOperation(Opcode::OP_CMP, {operandT1, operandT2}, {result});
    op.SetAttribute(OP_ATTR_PREFIX + "cmp_operation", static_cast<int64_t>(operation));
    op.SetAttribute(OP_ATTR_PREFIX + "cmp_mode", static_cast<int64_t>(mode));
    return result;
}

LogicalTensorPtr TensorCompareOperationScalar(
    Function &function, const Tensor &operand1, const Element &value, OpType operation, OutType mode) {
    DECLARE_TRACER();
    auto operandT1 = operand1.GetStorage();
    std::vector<int64_t> resultShape = operandT1->shape;
    std::vector<SymbolicScalar> resultValidShape = operandT1->GetDynValidShape();
    DataType resultType = DT_BOOL;
    if (mode == OutType::BIT) {
        resultType = DT_UINT8;
        if (!resultShape.empty()) {
            int64_t lastDim = resultShape.back();
            if (lastDim % NUM_VALUE_8 != 0) {
                ALOG_ERROR_F("Last dimension must be divisible by 8 in BIT mode");
            }
            resultShape.back() = lastDim / NUM_VALUE_8;
            if (!resultValidShape.empty()) {
                auto &lastSymDim = resultValidShape.back();
                resultValidShape.back() = lastSymDim / NUM_VALUE_8;
            }
        }
    }
    auto result = std::make_shared<LogicalTensor>(function, resultType, resultShape, resultValidShape);
    auto &op = function.AddOperation(Opcode::OP_CMPS, {operandT1}, {result});
    op.SetAttribute(OpAttributeKey::scalar, value);
    op.SetAttribute(OP_ATTR_PREFIX + "cmp_operation", static_cast<int64_t>(operation));
    op.SetAttribute(OP_ATTR_PREFIX + "cmp_mode", static_cast<int64_t>(mode));

    return result;
}

LogicalTensorPtr TensorCompareOperationScalar(Function& function, const Element& value, const Tensor& operand1,	 
    OpType operation, OutType mode) {	 
    switch(operation) {
        case OpType::LT: operation = OpType::GT; break;
        case OpType::GT: operation = OpType::LT; break;
        case OpType::LE: operation = OpType::GE; break;
        case OpType::GE: operation = OpType::LE; break;
        default: break;
    }
    Element converted_value = value;
    if (value.GetDataType() == DataType::DT_BF16) {
        double val = value.GetFloatData();
        converted_value = Element(DataType::DT_FP32, val);
    }

    return TensorCompareOperationScalar(function, operand1, converted_value, operation, mode);
}

void TiledCmpsOperationImpl(Function &function, const TileShape &tileShape, size_t cur, Input &input,
    const Element &scalar, const LogicalTensorPtr &result, TileInfo &resultTileInfo, OpType operation, OutType mode) {
    if (cur == result->shape.size()) {
        auto inputTile = input.tensor.GetStorage()->View(function, input.tileInfo.shape, input.tileInfo.offset);
        auto resultTile = result->View(function, resultTileInfo.shape, resultTileInfo.offset);

        LogicalTensorPtr convertedTile = inputTile;
        if (inputTile->Datatype() == DT_BF16) {
            convertedTile = std::make_shared<LogicalTensor>(function, DT_FP32, inputTile->GetShape());
            Operation &castOp = function.AddOperation(Opcode::OP_CAST, {inputTile}, {convertedTile});
            castOp.SetAttribute(OP_ATTR_PREFIX + "mode", CastMode::CAST_NONE);
        }

        const int64_t COUNT_MODE_SIZE = 4096;
        size_t element_size = BytesOf(input.tensor.GetDataType());
        if (inputTile->Datatype() == DT_BF16) {
            element_size = BytesOf(DT_FP32);
        }
        ASSERT(element_size != 0) << "Element size cannot be zero.";
        int64_t elements_per_chunk = COUNT_MODE_SIZE / element_size;
        int64_t vcmp_bits_size = (elements_per_chunk + 8 - 1) / 8;

        const size_t ALIGN_SIZE = 32;

        size_t vcmpBitResult_size = ((vcmp_bits_size + ALIGN_SIZE - 1) / ALIGN_SIZE) * ALIGN_SIZE;
        size_t array_size = elements_per_chunk * element_size;
        size_t aligned_array_size = ((array_size + ALIGN_SIZE - 1) / ALIGN_SIZE) * ALIGN_SIZE;

        size_t total_bytes = vcmpBitResult_size + 3 * aligned_array_size + ALIGN_SIZE;
        std::vector<int64_t> tmp_shape({static_cast<int64_t>(total_bytes)});
        auto tmp_tensor = std::make_shared<LogicalTensor>(function, DT_UINT8, tmp_shape);

        auto &op = function.AddOperation(Opcode::OP_CMPS, {convertedTile}, {resultTile, tmp_tensor});

        op.SetAttribute(OP_ATTR_PREFIX + "cmp_operation", static_cast<int64_t>(operation));
        op.SetAttribute(OP_ATTR_PREFIX + "cmp_mode", static_cast<int64_t>(mode));
        op.SetAttribute(OpAttributeKey::scalar, scalar);
        return;
    }

    auto &vecTile = tileShape.GetVecTile();
    int64_t step = vecTile[cur];

    if (mode == OutType::BIT && cur == result->shape.size() - 1) {
        step = vecTile[cur] / NUM_VALUE_8;
        if (step < 1) step = 1;

        int64_t actualInputStep = step * NUM_VALUE_8;

        for (int i = 0; i < result->shape[cur]; i += step) {
            resultTileInfo.offset[cur] = i;
            resultTileInfo.shape[cur] = std::min(result->shape[cur] - i, step);

            input.tileInfo.offset[cur] = (i * NUM_VALUE_8) % input.tensor.GetShape()[cur];
            input.tileInfo.shape[cur] =
                std::min(input.tensor.GetShape()[cur] - input.tileInfo.offset[cur], actualInputStep);

            TiledCmpsOperationImpl(
                function, tileShape, cur + 1, input, scalar, result, resultTileInfo, operation, mode);
        }
    } else {
        for (int i = 0; i < result->shape[cur]; i += step) {
            resultTileInfo.offset[cur] = i;
            resultTileInfo.shape[cur] = std::min(result->shape[cur] - i, step);

            input.tileInfo.offset[cur] = i % input.tensor.GetShape()[cur];
            input.tileInfo.shape[cur] = std::min(input.tensor.GetShape()[cur] - input.tileInfo.offset[cur], step);

            TiledCmpsOperationImpl(
                function, tileShape, cur + 1, input, scalar, result, resultTileInfo, operation, mode);
        }
    }
}

void TiledCmpsOperation(Function &function, const TileShape &tileShape, LogicalTensorPtr operand, const Element &scalar,
    const LogicalTensorPtr &result, OpType operation, OutType mode) {
    TileInfo tileInfo(result->shape.size(), result->offset.size());
    TileInfo resultTileInfo(result->shape.size(), result->offset.size());
    auto input = Input{operand, tileInfo};
    TiledCmpsOperationImpl(function, tileShape, 0, input, scalar, result, resultTileInfo, operation, mode);
}

Tensor Compare(const Tensor &self, const Tensor &other, OpType op, OutType mode) {
    DECLARE_TRACER();
    RETURN_CALL(CompareOperation, *Program::GetInstance().GetCurrentFunction(), self, other, op, mode);
}

Tensor Compare(const Tensor &self, const Element &other, OpType op, OutType mode) {
    DECLARE_TRACER();
    RETURN_CALL(CompareOperationScalar, *Program::GetInstance().GetCurrentFunction(), self, other, op, mode);
}

Tensor Compare(const Element &self, const Tensor &other, OpType op, OutType mode) {
    DECLARE_TRACER();
    RETURN_CALL(CompareOperationScalar, *Program::GetInstance().GetCurrentFunction(), self, other, op, mode);
}

void CompareOperationTileFunc(Function &function, const TileShape &tileShape,
    const std::vector<LogicalTensorPtr> &iOperand, const std::vector<LogicalTensorPtr> &oOperand, const Operation &op) {
    BinaryOperationOperandCheck(iOperand, oOperand);
    auto operation = static_cast<OpType>(op.GetIntAttribute(OP_ATTR_PREFIX + "cmp_operation"));
    auto mode = static_cast<OutType>(op.GetIntAttribute(OP_ATTR_PREFIX + "cmp_mode"));
    TiledCompareOperation(function, tileShape, iOperand[0], iOperand[1], oOperand[0], operation, mode);
}

void CmpsOperationTileFunc(Function &function, const TileShape &tileShape,
    const std::vector<LogicalTensorPtr> &iOperand, const std::vector<LogicalTensorPtr> &oOperand, const Operation &op) {
    auto operation = static_cast<OpType>(op.GetIntAttribute(OP_ATTR_PREFIX + "cmp_operation"));
    auto mode = static_cast<OutType>(op.GetIntAttribute(OP_ATTR_PREFIX + "cmp_mode"));
    TiledCmpsOperation(
        function, tileShape, iOperand[0], op.GetElementAttribute(OpAttributeKey::scalar), oOperand[0], operation, mode);
}

REGISTER_OPERATION_TILED_FUNC(OP_CMP, Opcode::OP_CMP, CompareOperationTileFunc);
REGISTER_OPERATION_TILED_FUNC(OP_CMPS, Opcode::OP_CMPS, CmpsOperationTileFunc);

} // namespace npu::tile_fwk