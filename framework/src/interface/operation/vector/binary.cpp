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
 * \file binary.cpp
 * \brief
 */

#include "binary.h"
#include "tensor_transformation.h"
#include "interface/utils/operator_tracer.h"
#include "interface/configs/config_manager.h"

namespace npu::tile_fwk {

std::vector<int64_t> BinaryOperationResultShape(LogicalTensorPtr operand1, LogicalTensorPtr operand2) {
    std::vector<int64_t> resultShape(operand1->shape.size());
    for (size_t i = 0; i < resultShape.size(); i++) {
        resultShape[i] = std::max(operand1->shape[i], operand2->shape[i]);
    }
    return resultShape;
}

LogicalTensorPtr BinaryOperationBroadCast(const LogicalTensorPtr &operand, const std::vector<int> &broadCastShape) {
    if (operand->shape.size() < broadCastShape.size()) {
        auto broadCastDims = broadCastShape.size() - operand->shape.size();
        std::vector<int64_t> unsqueezeShape(operand->shape);
        unsqueezeShape.insert(unsqueezeShape.begin(), broadCastDims, 1);
        auto tmpOperand = Reshape(operand, unsqueezeShape).GetStorage();
        return tmpOperand;
    }
    return operand;
}

void CheckOperandsValid(const LogicalTensorPtr &operand1, const LogicalTensorPtr &operand2) {
    ASSERT(operand1->shape.size() == operand2->shape.size()) << "The shape size of the two input tensors must be equal";
}

void CheckBinOpOperandsValid(const LogicalTensorPtr &operand1, const LogicalTensorPtr &operand2) {
    CheckOperandsValid(operand1, operand2);
    for (size_t i = 0; i < operand1->shape.size(); ++i) {
        if (operand1->shape[i] != operand2->shape[i] && (operand1->shape[i] != 1 && operand2->shape[i] != 1)) {
            ASSERT(false && "shape not support binary operation");
        }
    }
}

void CheckBinaryInputTensors(const LogicalTensorPtr &tensor1, const LogicalTensorPtr &tensor2, std::string &op) {
    CheckTensorShape(tensor1, op);
    CheckTensorShape(tensor2, op);
    CheckBinOpOperandsValid(tensor1, tensor2);
    if (tensor1->Datatype() != tensor2->Datatype()) {
        ASSERT(false && "The dtype of input tensors are not same.");
    }
    if (tensor1->Format() != tensor2->Format()) {
        ASSERT(false && "The format of input tensors are not same.");
    }
}

void BroadcastOperandTensor(LogicalTensorPtr &operand, LogicalTensorPtr &other, LogicalTensorPtr result,
                                      Function& function, const TileShape& tileShape) {
    auto dstShape = result->shape;
    if (operand->shape == dstShape) {
        return;
    }
    auto expanded = std::make_shared<LogicalTensor>(function, operand->Datatype(), dstShape);
    Expand(function, tileShape, operand, {other}, expanded);
    operand = expanded;
}

void BinaryOperationOperandCheck(
    const std::vector<LogicalTensorPtr> &iOperand, const std::vector<LogicalTensorPtr> &oOperand) {
    constexpr size_t inOpSize = 2;
    constexpr size_t outOpSize = 1;
    ASSERT(iOperand.size() == inOpSize && "iOperand size should be 2");
    ASSERT(oOperand.size() == outOpSize && "oOperand size should be 1");
}

// [m,n] + [m, 1]
bool CallBrcBinOp(LogicalTensorPtr operand1, LogicalTensorPtr operand2) {
    ASSERT(operand1->shape.size() == operand2->shape.size() && "Dims not match");
    size_t shapeSize = operand1->shape.size();
    for (size_t i = 0; i < shapeSize - 1; ++i) {
        if (operand1->shape[i] != operand2->shape[i]) {
            return false;
        }
    }

    return ((operand1->shape[shapeSize - 1] != 1) && (operand2->shape[shapeSize - 1] == 1)) ||
           ((operand1->shape[shapeSize - 1] == 1) && (operand2->shape[shapeSize - 1] != 1));
}

template <BinaryOpType T>
void TiledBinaryOperation(Function &function, const TileShape &tileShape, size_t cur, LogicalInput &input1,
    LogicalInput &input2, const LogicalTensorPtr &result, TileInfo &resultTileInfo, bool withBrc) {
    constexpr size_t shapeSize = 2;
    if (cur == input1.tensor->GetShape().size()) {
        auto inputTile1 = input1.tensor->View(function, input1.tileInfo.shape, input1.tileInfo.offset);
        auto inputTile2 = input2.tensor->View(function, input2.tileInfo.shape, input2.tileInfo.offset);
        auto resultTile = result->View(function, resultTileInfo.shape, resultTileInfo.offset);
        auto opName = GetBinaryOpName<T>();
        if (withBrc) {
            std::vector<int64_t> tmpShape(input1.tileInfo.shape);
            auto alignSize = BLOCK_SIZE / BytesOf(input2.tensor->Datatype());
            tmpShape[input1.tileInfo.shape.size() - 1] = alignSize;
            if (input1.tileInfo.shape.size() == shapeSize) {
                tmpShape[input1.tileInfo.shape.size() - shapeSize] =
                    (tmpShape[input1.tileInfo.shape.size() - shapeSize] + alignSize - 1) / alignSize * alignSize;
            }
            auto tempTensor = std::make_shared<LogicalTensor>(function, input2.tensor->Datatype(), tmpShape);
            function.AddOperation(
                GetBinaryOpNameCode<T, false, true>(), {inputTile1, inputTile2}, {resultTile, tempTensor});
        } else {
            if (opName == "BITWISEXOR" || opName == "COPYSIGN" || opName == "POW") {
                std::vector<int64_t> tmpShape(resultTileInfo.shape);
                auto alignSize = BLOCK_SIZE / BytesOf(result->Datatype());
                tmpShape[resultTileInfo.shape.size() - 1] = 
                    AlignUp(tmpShape[resultTileInfo.shape.size() - 1], alignSize);
                auto tempTensor = std::make_shared<LogicalTensor>(function, result->Datatype(), tmpShape);
                function.AddOperation(
                    GetBinaryOpNameCode<T, false, false>(), {inputTile1, inputTile2}, {resultTile, tempTensor});
            } else {
                function.AddOperation(GetBinaryOpNameCode<T, false, false>(), {inputTile1, inputTile2}, {resultTile});
            }
        }
        return;
    }
    auto &vecTile = tileShape.GetVecTile();
    for (int i = 0; i < result->shape[cur]; i += vecTile[cur]) {
        resultTileInfo.offset[cur] = i;
        resultTileInfo.shape[cur] = std::min(result->shape[cur] - resultTileInfo.offset[cur], vecTile[cur]);
        input1.tileInfo.offset[cur] = i % input1.tensor->GetShape()[cur];
        input1.tileInfo.shape[cur] =
            std::min(input1.tensor->GetShape()[cur] - input1.tileInfo.offset[cur], vecTile[cur]);
        input2.tileInfo.offset[cur] = i % input2.tensor->GetShape()[cur];
        input2.tileInfo.shape[cur] =
            std::min(input2.tensor->GetShape()[cur] - input2.tileInfo.offset[cur], vecTile[cur]);
        TiledBinaryOperation<T>(function, tileShape, cur + 1, input1, input2, result, resultTileInfo, withBrc);
    }
}

template <BinaryOpType T>
void TiledBinaryOperation(Function &function, const TileShape &tileShape, LogicalTensorPtr operand1,
    LogicalTensorPtr operand2, const LogicalTensorPtr &result) {
    CheckBinOpOperandsValid(operand1, operand2);
    bool withBrc = CallBrcBinOp(operand1, operand2) &&
                   (function.paramConfigs_.forceCombineAxis ||
                       function.paramConfigs_.combineAxis);
    // nolast brc will be inline
    if (!withBrc) {
        BroadcastOperandTensor(operand1, operand2, result, function, tileShape);
        BroadcastOperandTensor(operand2, operand1, result, function, tileShape);
    }

    TileInfo tileInfo1(result->shape.size(), result->offset.size());
    TileInfo tileInfo2(result->shape.size(), result->offset.size());
    TileInfo resultTileInfo(result->shape.size(), result->offset.size());
    auto input1 = LogicalInput{operand1, tileInfo1};
    auto input2 = LogicalInput{operand2, tileInfo2};
    // 如果使能了Combine Axis逻辑，需要将withbrc置为false，避免后续走OP_XX_BRC逻辑
    if (function.paramConfigs_.combineAxis) {
        withBrc = false;
    }
    TiledBinaryOperation<T>(function, tileShape, 0, input1, input2, result, resultTileInfo, withBrc);
}

Tensor Add(const Tensor &self, const Tensor &other) {
    DECLARE_TRACER();
    RETURN_CALL(BinaryOperation<BinaryOpType::ADD>, *Program::GetInstance().GetCurrentFunction(), self, other);
}

Tensor Sub(const Tensor &self, const Tensor &other) {
    DECLARE_TRACER();

    RETURN_CALL(BinaryOperation<BinaryOpType::SUB>, *Program::GetInstance().GetCurrentFunction(), self, other);
}

Tensor Mul(const Tensor &self, const Tensor &other) {
    DECLARE_TRACER();

    RETURN_CALL(BinaryOperation<BinaryOpType::MUL>, *Program::GetInstance().GetCurrentFunction(), self, other);
}

Tensor Div(const Tensor &self, const Tensor &other) {
    DECLARE_TRACER();

    RETURN_CALL(BinaryOperation<BinaryOpType::DIV>, *Program::GetInstance().GetCurrentFunction(), self, other);
}

Tensor Fmod(const Tensor &self, const Tensor &other) {
    DECLARE_TRACER();
    RETURN_CALL(BinaryOperation<BinaryOpType::MOD>, *Program::GetInstance().GetCurrentFunction(), self, other);
}

Tensor Maximum(const Tensor &operand1, const Tensor &operand2) {
    DECLARE_TRACER();

    RETURN_CALL(
        BinaryOperation<BinaryOpType::MAXIMUM>, *Program::GetInstance().GetCurrentFunction(), operand1, operand2);
}

Tensor Minimum(const Tensor &operand1, const Tensor &operand2) {
    DECLARE_TRACER();

    RETURN_CALL(
        BinaryOperation<BinaryOpType::MINIMUM>, *Program::GetInstance().GetCurrentFunction(), operand1, operand2);
}

Tensor BitwiseAnd(const Tensor &self, const Tensor &other) {
    DECLARE_TRACER();
    RETURN_CALL(BinaryOperation<BinaryOpType::BITWISEAND>, *Program::GetInstance().GetCurrentFunction(), self, other);
}

Tensor BitwiseOr(const Tensor &self, const Tensor &other) {
    DECLARE_TRACER();
    RETURN_CALL(BinaryOperation<BinaryOpType::BITWISEOR>, *Program::GetInstance().GetCurrentFunction(), self, other);
}

Tensor BitwiseXor(const Tensor &self, const Tensor &other) {
    DECLARE_TRACER();
    RETURN_CALL(BinaryOperation<BinaryOpType::BITWISEXOR>, *Program::GetInstance().GetCurrentFunction(), self, other);
}

Tensor Gcd(const Tensor &self, const Tensor &other) {
    DECLARE_TRACER();
    auto shapeSize = self.GetShape().size();
    auto dataType = self.GetDataType();
    ASSERT(dataType == other.GetDataType()) << "Inputs must have the same dataType.";
    ASSERT(SHAPE_DIM1 <= shapeSize && shapeSize <= SHAPE_DIM5) << "This operation's input only support 1-5 dims";
    std::unordered_set<DataType> GCD_SUPPORT_DATATYPES = {DataType::DT_INT32, DataType::DT_INT16, DataType::DT_INT8, DataType::DT_UINT8};
    ASSERT(GCD_SUPPORT_DATATYPES.count(dataType))<< "This datatype is not supported";
    RETURN_CALL(BinaryOperation<BinaryOpType::GCD>, *Program::GetInstance().GetCurrentFunction(), self, other);
}

Tensor Gcd(const Tensor &self, const Element &other) {
    DECLARE_TRACER();
    auto shapeSize = self.GetShape().size();
    auto dataType = self.GetDataType();
    ASSERT(SHAPE_DIM1 <= shapeSize && shapeSize <= SHAPE_DIM5) << "This operation's input only support 1-5 dims";
    std::unordered_set<DataType> GCD_SUPPORT_DATATYPES = {DataType::DT_INT32, DataType::DT_INT16, DataType::DT_INT8, DataType::DT_UINT8};
    ASSERT(GCD_SUPPORT_DATATYPES.count(dataType))<< "This datatype is not supported";
    RETURN_CALL(BinaryOperationScalar<BinaryOpType::GCD>, *Program::GetInstance().GetCurrentFunction(), self.GetStorage(), other);
}

template <BinaryOpType T>
void TiledBinaryOperationScalar(Function &function, const TileShape &tileShape, size_t cur, LogicalInput &input1,
    Element &value, const LogicalTensorPtr &result, TileInfo &resultTileInfo, bool reverseOperand) {
    if (cur == input1.tensor->GetShape().size()) {
        auto inputTile1 = input1.tensor->View(function, input1.tileInfo.shape, input1.tileInfo.offset);
        auto resultTile = result->View(function, resultTileInfo.shape, resultTileInfo.offset);
        auto opName = GetBinaryOpName<T>();
        if (GetBinaryOpNameCode<T, true, false>() == Opcode::OP_BITWISEXORS) {
            std::vector<int64_t> tmpShape(resultTileInfo.shape);
            auto alignSize = BLOCK_SIZE / BytesOf(input1.tensor->Datatype());
            tmpShape[resultTileInfo.shape.size() - 1] = AlignUp(tmpShape[resultTileInfo.shape.size() - 1], alignSize);
            auto tempTensor = std::make_shared<LogicalTensor>(function, input1.tensor->Datatype(), tmpShape);
            auto &tmpOp = function.AddOperation(
                GetBinaryOpNameCode<T, true, false>(), {inputTile1}, {resultTile, tempTensor});
            tmpOp.SetAttribute(OpAttributeKey::scalar, value);
            tmpOp.SetAttribute(OP_ATTR_PREFIX + "reverseOperand", reverseOperand);
            return;
        }
        // 确认接口
        auto &op = function.AddOperation(GetBinaryOpNameCode<T, true>(), {inputTile1}, {resultTile});
        op.SetAttribute(OpAttributeKey::scalar, value);
        op.SetAttribute(OP_ATTR_PREFIX + "reverseOperand", reverseOperand);
        return;
    }
    auto &vecTile = tileShape.GetVecTile();
    for (int i = 0; i < result->shape[cur]; i += vecTile[cur]) {
        resultTileInfo.offset[cur] = i;
        resultTileInfo.shape[cur] = std::min(result->shape[cur] - resultTileInfo.offset[cur], vecTile[cur]);
        input1.tileInfo.offset[cur] = i % input1.tensor->GetShape()[cur];
        input1.tileInfo.shape[cur] =
            std::min(input1.tensor->GetShape()[cur] - input1.tileInfo.offset[cur], vecTile[cur]);

        TiledBinaryOperationScalar<T>(
            function, tileShape, cur + 1, input1, value, result, resultTileInfo, reverseOperand);
    }
}

template <BinaryOpType T>
void TiledBinaryOperationScalar(Function &function, const TileShape &tileShape, LogicalTensorPtr operand1,
    Element value, const LogicalTensorPtr &result, bool reverseOperand = false) {
    TileInfo tileInfo1(result->shape.size(), result->offset.size());
    TileInfo resultTileInfo(result->shape.size(), result->offset.size());
    auto input1 = LogicalInput{operand1, tileInfo1};
    TiledBinaryOperationScalar<T>(function, tileShape, 0, input1, value, result, resultTileInfo, reverseOperand);
}

Tensor Add(const Tensor &self, const Element &other) {
    DECLARE_TRACER();
    RETURN_CALL(BinaryOperationScalar<BinaryOpType::ADD>, *Program::GetInstance().GetCurrentFunction(),
        self.GetStorage(), other);
}

Tensor Sub(const Tensor &self, const Element &other) {
    DECLARE_TRACER();
    RETURN_CALL(BinaryOperationScalar<BinaryOpType::SUB>, *Program::GetInstance().GetCurrentFunction(),
        self.GetStorage(), other);
}

Tensor Mul(const Tensor &self, const Element &other) {
    DECLARE_TRACER();
    RETURN_CALL(BinaryOperationScalar<BinaryOpType::MUL>, *Program::GetInstance().GetCurrentFunction(),
        self.GetStorage(), other);
}

Tensor Div(const Tensor &self, const Element &other) {
    DECLARE_TRACER();
    RETURN_CALL(BinaryOperationScalar<BinaryOpType::DIV>, *Program::GetInstance().GetCurrentFunction(),
        self.GetStorage(), other);
}

Tensor Fmod(const Tensor &self, const Element &other) {
    DECLARE_TRACER();
    RETURN_CALL(BinaryOperationScalar<BinaryOpType::MOD>, *Program::GetInstance().GetCurrentFunction(),
        self.GetStorage(), other);
}
Tensor BitwiseAnd(const Tensor &self, const Element &other) {
    DECLARE_TRACER();
    RETURN_CALL(BinaryOperationScalar<BinaryOpType::BITWISEAND>, *Program::GetInstance().GetCurrentFunction(),
        self.GetStorage(), other);
}

Tensor BitwiseOr(const Tensor &self, const Element &other) {
    DECLARE_TRACER();
    RETURN_CALL(BinaryOperationScalar<BinaryOpType::BITWISEOR>, *Program::GetInstance().GetCurrentFunction(),
        self.GetStorage(), other);
}

Tensor BitwiseXor(const Tensor &self, const Element &other) {
    DECLARE_TRACER();
    RETURN_CALL(BinaryOperationScalar<BinaryOpType::BITWISEXOR>, *Program::GetInstance().GetCurrentFunction(),
        self.GetStorage(), other);
}

Tensor Maximum(const Tensor &operand1, const Element &operand2) {
    DECLARE_TRACER();
    ASSERT(operand1.GetDataType() == operand2.GetDataType()) << "The datatype of the two input must be equal";
    std::vector<DataType> MAXS_SUPPORT_DATATYPES = {
        DataType::DT_FP32, DataType::DT_FP16, DataType::DT_INT32, DataType::DT_INT16, DataType::DT_BF16};
    ASSERT(std::find(MAXS_SUPPORT_DATATYPES.begin(), MAXS_SUPPORT_DATATYPES.end(), operand1.GetDataType()) !=
           MAXS_SUPPORT_DATATYPES.end())
        << "The datatype is not supported";
    RETURN_CALL(BinaryOperationScalar<BinaryOpType::MAX>, *Program::GetInstance().GetCurrentFunction(),
        operand1.GetStorage(), operand2);
}

Tensor Minimum(const Tensor &operand1, const Element &operand2) {
    DECLARE_TRACER();
    ASSERT(operand1.GetDataType() == operand2.GetDataType()) << "The datatype of the two input must be equal";
    std::vector<DataType> MINS_SUPPORT_DATATYPES = {
        DataType::DT_FP32, DataType::DT_FP16, DataType::DT_INT32, DataType::DT_INT16, DataType::DT_BF16};
    ASSERT(std::find(MINS_SUPPORT_DATATYPES.begin(), MINS_SUPPORT_DATATYPES.end(), operand1.GetDataType()) !=
           MINS_SUPPORT_DATATYPES.end())
        << "The datatype is not supported";
    RETURN_CALL(BinaryOperationScalar<BinaryOpType::MIN>, *Program::GetInstance().GetCurrentFunction(),
        operand1.GetStorage(), operand2);
}

template <BinaryOpType T>
void TiledBinaryOperationAllScalar(Function &function, const TileShape &tileShape, size_t cur, LogicalInput &input1,
    Element &value, const LogicalTensorPtr &result, TileInfo &resultTileInfo, bool reverseOperand) {
    if (cur == input1.tensor->GetShape().size()) {
        auto inputTile1 = input1.tensor->View(function, input1.tileInfo.shape, input1.tileInfo.offset);
        auto resultTile = result->View(function, resultTileInfo.shape, resultTileInfo.offset);
        // 确认接口
        auto &op = function.AddOperation(GetBinaryOpNameCode<T, true>(), {inputTile1}, {resultTile});
        op.SetAttribute(OpAttributeKey::scalar, value);
        op.SetAttribute(OP_ATTR_PREFIX + "reverseOperand", reverseOperand);
        return;
    }
    auto &vecTile = tileShape.GetVecTile();
    for (int i = 0; i < result->shape[cur]; i += vecTile[cur]) {
        resultTileInfo.offset[cur] = i;
        resultTileInfo.shape[cur] = std::min(result->shape[cur] - resultTileInfo.offset[cur], vecTile[cur]);
        input1.tileInfo.offset[cur] = i % input1.tensor->GetShape()[cur];
        input1.tileInfo.shape[cur] =
            std::min(input1.tensor->GetShape()[cur] - input1.tileInfo.offset[cur], vecTile[cur]);

        TiledBinaryOperationScalar<T>(
            function, tileShape, cur + 1, input1, value, result, resultTileInfo, reverseOperand);
    }
}

template <BinaryOpType T>
void TiledBinaryOperationAllScalar(Function &function, const TileShape &tileShape, LogicalTensorPtr operand1,
    Element value, const LogicalTensorPtr &result, bool reverseOperand) {
    TileInfo tileInfo1(result->shape.size(), result->offset.size());
    TileInfo resultTileInfo(result->shape.size(), result->offset.size());
    auto input1 = LogicalInput{operand1, tileInfo1};
    TiledBinaryOperationAllScalar<T>(function, tileShape, 0, input1, value, result, resultTileInfo, reverseOperand);
}

Tensor ScalarAddS(const Tensor &operand, const Element &value, bool reverseOperand) {
    DECLARE_TRACER();

    RETURN_CALL(BinaryOperationAllScalar<BinaryOpType::S_ADD>, *Program::GetInstance().GetCurrentFunction(),
        operand.GetStorage(), value, reverseOperand);
}

Tensor ScalarSubS(const Tensor &operand, const Element &value, bool reverseOperand) {
    DECLARE_TRACER();

    RETURN_CALL(BinaryOperationAllScalar<BinaryOpType::S_SUB>, *Program::GetInstance().GetCurrentFunction(),
        operand.GetStorage(), value, reverseOperand);
}

Tensor ScalarMulS(const Tensor &operand, const Element &value, bool reverseOperand) {
    DECLARE_TRACER();

    RETURN_CALL(BinaryOperationAllScalar<BinaryOpType::S_MUL>, *Program::GetInstance().GetCurrentFunction(),
        operand.GetStorage(), value, reverseOperand);
}

Tensor ScalarDivS(const Tensor &operand, const Element &value, bool reverseOperand) {
    DECLARE_TRACER();

    RETURN_CALL(BinaryOperationAllScalar<BinaryOpType::S_DIV>, *Program::GetInstance().GetCurrentFunction(),
        operand.GetStorage(), value, reverseOperand);
}

Tensor ScalarMaxS(const Tensor &operand, const Element &value, bool reverseOperand) {
    DECLARE_TRACER();

    RETURN_CALL(BinaryOperationAllScalar<BinaryOpType::S_MAX>, *Program::GetInstance().GetCurrentFunction(),
        operand.GetStorage(), value, reverseOperand);
}

template <BinaryOpType T>
void TiledBinaryOperationAllScalar(Function &function, const TileShape &tileShape, size_t cur, LogicalInput &input1,
    LogicalInput &input2, const LogicalTensorPtr &result, TileInfo &resultTileInfo) {
    if (cur == input1.tensor->GetShape().size()) {
        auto inputTile1 = input1.tensor->View(function, input1.tileInfo.shape, input1.tileInfo.offset);
        auto inputTile2 = input2.tensor->View(function, input2.tileInfo.shape, input2.tileInfo.offset);
        auto resultTile = result->View(function, resultTileInfo.shape, resultTileInfo.offset);
        function.AddOperation(GetBinaryOpNameCode<T, false>(), {inputTile1, inputTile2}, {resultTile});
        return;
    }
    auto &vecTile = tileShape.GetVecTile();
    for (int i = 0; i < result->shape[cur]; i += vecTile[cur]) {
        resultTileInfo.offset[cur] = i;
        resultTileInfo.shape[cur] = std::min(result->shape[cur] - resultTileInfo.offset[cur], vecTile[cur]);
        input1.tileInfo.offset[cur] = i % input1.tensor->GetShape()[cur];
        input1.tileInfo.shape[cur] =
            std::min(input1.tensor->GetShape()[cur] - input1.tileInfo.offset[cur], vecTile[cur]);
        input2.tileInfo.offset[cur] = i % input2.tensor->GetShape()[cur];
        input2.tileInfo.shape[cur] =
            std::min(input2.tensor->GetShape()[cur] - input2.tileInfo.offset[cur], vecTile[cur]);
        TiledBinaryOperationAllScalar<T>(function, tileShape, cur + 1, input1, input2, result, resultTileInfo);
    }
}

template <BinaryOpType T>
void TiledBinaryOperationAllScalar(Function &function, const TileShape &tileShape, LogicalTensorPtr operand1,
    LogicalTensorPtr operand2, const LogicalTensorPtr &result) {
    CheckBinOpOperandsValid(operand1, operand2);

    if (operand1->shape != result->shape) {
        auto targetShape = result->shape;
        auto tmp = std::make_shared<LogicalTensor>(function, operand1->Datatype(), targetShape);
        Expand(function, tileShape, operand1, {operand2}, tmp);
        operand1 = tmp;
    }

    if (operand2->shape != result->shape) {
        auto targetShape = result->shape;
        auto tmp = std::make_shared<LogicalTensor>(function, operand2->Datatype(), targetShape);
        Expand(function, tileShape, operand2, {operand1}, tmp);
        operand2 = tmp;
    }

    TileInfo tileInfo1(result->shape.size(), result->offset.size());
    TileInfo tileInfo2(result->shape.size(), result->offset.size());
    TileInfo resultTileInfo(result->shape.size(), result->offset.size());
    auto input1 = LogicalInput{operand1, tileInfo1};
    auto input2 = LogicalInput{operand2, tileInfo2};
    TiledBinaryOperationAllScalar<T>(function, tileShape, 0, input1, input2, result, resultTileInfo);
}

Tensor ScalarAdd(const Tensor &operand1, const Tensor &operand2) {
    DECLARE_TRACER();

    RETURN_CALL(BinaryOperationAllScalar<BinaryOpType::S_ADD>, *Program::GetInstance().GetCurrentFunction(),
        operand1.GetStorage(), operand2.GetStorage());
}
Tensor ScalarSub(const Tensor &operand1, const Tensor &operand2) {
    DECLARE_TRACER();

    RETURN_CALL(BinaryOperationAllScalar<BinaryOpType::S_SUB>, *Program::GetInstance().GetCurrentFunction(),
        operand1.GetStorage(), operand2.GetStorage());
}

Tensor ScalarMul(const Tensor &operand1, const Tensor &operand2) {
    DECLARE_TRACER();

    RETURN_CALL(BinaryOperationAllScalar<BinaryOpType::S_MUL>, *Program::GetInstance().GetCurrentFunction(),
        operand1.GetStorage(), operand2.GetStorage());
}

Tensor ScalarDiv(const Tensor &operand1, const Tensor &operand2) {
    DECLARE_TRACER();

    RETURN_CALL(BinaryOperationAllScalar<BinaryOpType::S_DIV>, *Program::GetInstance().GetCurrentFunction(),
        operand1.GetStorage(), operand2.GetStorage());
}

Tensor ScalarMax(const Tensor &operand1, const Tensor &operand2) {
    DECLARE_TRACER();

    RETURN_CALL(BinaryOperationAllScalar<BinaryOpType::S_MAX>, *Program::GetInstance().GetCurrentFunction(),
        operand1.GetStorage(), operand2.GetStorage());
}

Tensor CopySign(const Tensor &self, const Tensor &other) {
    DECLARE_TRACER();

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
    RETURN_CALL(BinaryOperation<BinaryOpType::COPYSIGN>, *Program::GetInstance().GetCurrentFunction(), castSelf, castOther);
}

// OP_ADD OP_SUB OP_MUL OP_DIV OP_MAX OP_BITWISEAND OP_BITWISEOR OP_BITWISEXOR
template <BinaryOpType T>
void BinaryOperationTileFunc(Function &function, const TileShape &tileShape,
    const std::vector<LogicalTensorPtr> &iOperand, const std::vector<LogicalTensorPtr> &oOperand,
    [[maybe_unused]] const Operation &op) {
    BinaryOperationOperandCheck(iOperand, oOperand);
    TiledBinaryOperation<T>(function, tileShape, iOperand[0], iOperand[1], oOperand[0]);
}

// OP_ADDS OP_SUBS OP_MULS OP_DIVS OP_MAXS OP_MINS OP_BITWISEANDS OP_BITWISEORS OP_BITWISEXORS
template <BinaryOpType T>
void BinaryOperationScalarTileFunc(Function &function, const TileShape &tileShape,
    const std::vector<LogicalTensorPtr> &iOperand, const std::vector<LogicalTensorPtr> &oOperand,
    [[maybe_unused]] const Operation &op) {
    TiledBinaryOperationScalar<T>(
        function, tileShape, iOperand[0], op.GetElementAttribute(OpAttributeKey::scalar), oOperand[0]);
}

// OP_S_ADDS OP_S_SUBS OP_S_MULS OP_S_DIVS OP_S_MAXS
template <BinaryOpType T>
void BinaryOperationAllScalarResTileFunc(Function &function, const TileShape &tileShape,
    const std::vector<LogicalTensorPtr> &iOperand, const std::vector<LogicalTensorPtr> &oOperand,
    [[maybe_unused]] const Operation &op) {
    TiledBinaryOperationAllScalar<T>(function, tileShape, iOperand[0], op.GetElementAttribute(OpAttributeKey::scalar),
        oOperand[0], op.GetBoolAttribute(OP_ATTR_PREFIX + "reverseOperand"));
}

// OP_S_ADD OP_S_SUB OP_S_MUL OP_S_DIV OP_S_MAX
template <BinaryOpType T>
void BinaryOperationAllScalarTileFunc(Function &function, const TileShape &tileShape,
    const std::vector<LogicalTensorPtr> &iOperand, const std::vector<LogicalTensorPtr> &oOperand,
    [[maybe_unused]] const Operation &op) {
    BinaryOperationOperandCheck(iOperand, oOperand);
    TiledBinaryOperationAllScalar<T>(function, tileShape, iOperand[0], iOperand[1], oOperand[0]);
}

REGISTER_OPERATION_TILED_FUNC(OP_ADD, Opcode::OP_ADD, BinaryOperationTileFunc<BinaryOpType::ADD>);
REGISTER_OPERATION_TILED_FUNC(OP_SUB, Opcode::OP_SUB, BinaryOperationTileFunc<BinaryOpType::SUB>);
REGISTER_OPERATION_TILED_FUNC(OP_MUL, Opcode::OP_MUL, BinaryOperationTileFunc<BinaryOpType::MUL>);
REGISTER_OPERATION_TILED_FUNC(OP_DIV, Opcode::OP_DIV, BinaryOperationTileFunc<BinaryOpType::DIV>);
REGISTER_OPERATION_TILED_FUNC(OP_MAXIMUM, Opcode::OP_MAXIMUM, BinaryOperationTileFunc<BinaryOpType::MAXIMUM>);
REGISTER_OPERATION_TILED_FUNC(OP_MINIMUM, Opcode::OP_MINIMUM, BinaryOperationTileFunc<BinaryOpType::MINIMUM>);
REGISTER_OPERATION_TILED_FUNC(OP_POW, Opcode::OP_POW, BinaryOperationTileFunc<BinaryOpType::POW>);
REGISTER_OPERATION_TILED_FUNC(OP_MOD, Opcode::OP_MOD, BinaryOperationTileFunc<BinaryOpType::MOD>);
REGISTER_OPERATION_TILED_FUNC(OP_BITWISEAND, Opcode::OP_BITWISEAND, BinaryOperationTileFunc<BinaryOpType::BITWISEAND>);
REGISTER_OPERATION_TILED_FUNC(OP_BITWISEOR, Opcode::OP_BITWISEOR, BinaryOperationTileFunc<BinaryOpType::BITWISEOR>);
REGISTER_OPERATION_TILED_FUNC(OP_BITWISEXOR, Opcode::OP_BITWISEXOR, BinaryOperationTileFunc<BinaryOpType::BITWISEXOR>);
REGISTER_OPERATION_TILED_FUNC(OP_COPYSIGN, Opcode::OP_COPYSIGN, BinaryOperationTileFunc<BinaryOpType::COPYSIGN>);
REGISTER_OPERATION_TILED_FUNC(OP_GCD, Opcode::OP_GCD, BinaryOperationTileFunc<BinaryOpType::GCD>);

REGISTER_OPERATION_TILED_FUNC(OP_ADDS, Opcode::OP_ADDS, BinaryOperationScalarTileFunc<BinaryOpType::ADD>);
REGISTER_OPERATION_TILED_FUNC(OP_SUBS, Opcode::OP_SUBS, BinaryOperationScalarTileFunc<BinaryOpType::SUB>);
REGISTER_OPERATION_TILED_FUNC(OP_MULS, Opcode::OP_MULS, BinaryOperationScalarTileFunc<BinaryOpType::MUL>);
REGISTER_OPERATION_TILED_FUNC(OP_DIVS, Opcode::OP_DIVS, BinaryOperationScalarTileFunc<BinaryOpType::DIV>);
REGISTER_OPERATION_TILED_FUNC(OP_MAXS, Opcode::OP_MAXS, BinaryOperationScalarTileFunc<BinaryOpType::MAX>);
REGISTER_OPERATION_TILED_FUNC(OP_MINS, Opcode::OP_MINS, BinaryOperationScalarTileFunc<BinaryOpType::MIN>);
REGISTER_OPERATION_TILED_FUNC(OP_MODS, Opcode::OP_MODS, BinaryOperationScalarTileFunc<BinaryOpType::MOD>);
REGISTER_OPERATION_TILED_FUNC(OP_BITWISEANDS, Opcode::OP_BITWISEANDS, BinaryOperationScalarTileFunc<BinaryOpType::BITWISEAND>);
REGISTER_OPERATION_TILED_FUNC(OP_BITWISEORS, Opcode::OP_BITWISEORS, BinaryOperationScalarTileFunc<BinaryOpType::BITWISEOR>);
REGISTER_OPERATION_TILED_FUNC(OP_BITWISEXORS, Opcode::OP_BITWISEXORS, BinaryOperationScalarTileFunc<BinaryOpType::BITWISEXOR>);
REGISTER_OPERATION_TILED_FUNC(OP_GCDS, Opcode::OP_GCDS, BinaryOperationScalarTileFunc<BinaryOpType::GCD>);

REGISTER_OPERATION_TILED_FUNC(OP_S_ADDS, Opcode::OP_S_ADDS, BinaryOperationAllScalarResTileFunc<BinaryOpType::S_ADD>);
REGISTER_OPERATION_TILED_FUNC(OP_S_SUBS, Opcode::OP_S_SUBS, BinaryOperationAllScalarResTileFunc<BinaryOpType::S_SUB>);
REGISTER_OPERATION_TILED_FUNC(OP_S_MULS, Opcode::OP_S_MULS, BinaryOperationAllScalarResTileFunc<BinaryOpType::S_MUL>);
REGISTER_OPERATION_TILED_FUNC(OP_S_DIVS, Opcode::OP_S_DIVS, BinaryOperationAllScalarResTileFunc<BinaryOpType::S_DIV>);
REGISTER_OPERATION_TILED_FUNC(OP_S_MAXS, Opcode::OP_S_MAXS, BinaryOperationAllScalarResTileFunc<BinaryOpType::S_MAX>);
REGISTER_OPERATION_TILED_FUNC(OP_S_MINS, Opcode::OP_S_MINS, BinaryOperationAllScalarResTileFunc<BinaryOpType::S_MIN>);

REGISTER_OPERATION_TILED_FUNC(OP_S_ADD, Opcode::OP_S_ADD, BinaryOperationAllScalarTileFunc<BinaryOpType::S_ADD>);
REGISTER_OPERATION_TILED_FUNC(OP_S_SUB, Opcode::OP_S_SUB, BinaryOperationAllScalarTileFunc<BinaryOpType::S_SUB>);
REGISTER_OPERATION_TILED_FUNC(OP_S_MUL, Opcode::OP_S_MUL, BinaryOperationAllScalarTileFunc<BinaryOpType::S_MUL>);
REGISTER_OPERATION_TILED_FUNC(OP_S_DIV, Opcode::OP_S_DIV, BinaryOperationAllScalarTileFunc<BinaryOpType::S_DIV>);
REGISTER_OPERATION_TILED_FUNC(OP_S_MAX, Opcode::OP_S_MAX, BinaryOperationAllScalarTileFunc<BinaryOpType::S_MAX>);
REGISTER_OPERATION_TILED_FUNC(OP_S_MIN, Opcode::OP_S_MIN, BinaryOperationAllScalarTileFunc<BinaryOpType::S_MIN>);

} // namespace npu::tile_fwk