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
 * \file math.cpp
 * \brief
 */

#include "unary.h"
#include "binary.h"
#include "tensor_transformation.h"
#include "interface/utils/operator_tracer.h"
#include "passes/pass_utils/graph_utils.h"

namespace npu::tile_fwk {

void TiledLogicalNotOperation(
    Function &function, const TileShape &tileShape, size_t cur, Input &input, const LogicalTensorPtr &result) {
    if (cur == input.tensor.GetShape().size()) {
        auto tile = input.tensor.GetStorage()->View(function, input.tileInfo.shape, input.tileInfo.offset);
        auto resultTile = result->View(function, input.tileInfo.shape, input.tileInfo.offset);
        
        constexpr int64_t COUNT_NUM = 2048;
        constexpr int64_t vcmp_bit_size = COUNT_NUM / 8;
        constexpr size_t ALIGN_SIZE = 32;

        DataType select_dtype;
        if (input.tensor.GetDataType() == DT_FP32 || input.tensor.GetDataType() == DT_BF16) {
            select_dtype = DT_FP32;
        } else {
            select_dtype = DT_FP16;
        }
        
        int64_t total_size = COUNT_NUM * 2 + COUNT_NUM * BytesOf(select_dtype) * 2 + vcmp_bit_size + 8;
        total_size = (total_size + ALIGN_SIZE - 1) / ALIGN_SIZE * ALIGN_SIZE;
        std::vector<int64_t> tmpShape({total_size});

        auto tmpTensor = std::make_shared<LogicalTensor>(function, DT_INT8, tmpShape);
        function.AddOperation(Opcode::OP_LOGICALNOT, {tile}, {resultTile, tmpTensor});
        return;
    }

    auto &vecTile = tileShape.GetVecTile();
    for (int i = 0; i < input.tensor.GetShape()[cur]; i += vecTile[cur]) {
        input.tileInfo.shape[cur] = std::min(input.tensor.GetShape()[cur] - i, vecTile[cur]);
        input.tileInfo.offset[cur] = i;
        TiledLogicalNotOperation(function, tileShape, cur + 1, input, result);
    }
}

void TiledLogicalNotOperation(
    Function &function, const TileShape &tileShape, const LogicalTensorPtr &self, const LogicalTensorPtr &result) {
    ASSERT(self->shape.size() == self->offset.size()) << "Shape size and offset size should be equal";

    TileInfo tileInfo(result->shape.size(), result->offset.size());
    auto input = Input{self, tileInfo};
    TiledLogicalNotOperation(function, tileShape, 0, input, result);
}

LogicalTensorPtr TensorLogicalNotOperation(Function &function, LogicalTensorPtr self) {
    auto result = std::make_shared<LogicalTensor>(function, DT_BOOL, self->shape, self->GetDynValidShape());
    function.AddOperation(Opcode::OP_LOGICALNOT, {self}, {result});
    return result;
}

Tensor LogicalNot(const Tensor &self) {
    DECLARE_TRACER();
    bool dtypeIsValid = self.GetDataType() == DT_FP32 || self.GetDataType() == DT_FP16 ||
                        self.GetDataType() == DT_UINT8 || self.GetDataType() == DT_INT8 ||
                        self.GetDataType() == DT_BOOL || self.GetDataType() == DT_BF16;
    if (!dtypeIsValid) {
        std::string errorMessage = "Unsurpported Dtype " + DataType2String(self.GetDataType());
        ASSERT(false) << errorMessage;
    }
    RETURN_CALL(LogicalNotOperation, *Program::GetInstance().GetCurrentFunction(), self.GetStorage());
}

Tensor Neg(const Tensor &self) {
    DECLARE_TRACER();

    if (IsFloat(self.GetStorage()->Datatype())) {
        RETURN_CALL(BinaryOperationScalar<BinaryOpType::MUL>, *Program::GetInstance().GetCurrentFunction(),
            self.GetStorage(), Element(self.GetStorage()->Datatype(), -1.0));
    } else {
        RETURN_CALL(BinaryOperationScalar<BinaryOpType::MUL>, *Program::GetInstance().GetCurrentFunction(),
            self.GetStorage(), Element(self.GetStorage()->Datatype(), -1));
    }
}

Tensor Log(const Tensor &self, LogBaseType base) {
    DECLARE_TRACER();
    ASSERT(base == LogBaseType::LOG_E || base == LogBaseType::LOG_2 || base == LogBaseType::LOG_10)
        << "base is incorrect";
    ASSERT(self.GetStorage()->tensor->datatype == DataType::DT_BF16 ||
           self.GetStorage()->tensor->datatype == DataType::DT_FP16 ||
           self.GetStorage()->tensor->datatype == DataType::DT_FP32)
        << "The datatype is not supported";

    auto operandCast = Tensor(DataType::DT_FP32, self.GetShape());
    if (self.GetStorage()->tensor->datatype == DataType::DT_FP16 || self.GetStorage()->tensor->datatype == DataType::DT_BF16) {
        operandCast = CALL(CastOperation<CastOpType::CAST>, *Program::GetInstance().GetCurrentFunction(),
            self.GetStorage(), DataType::DT_FP32, CastMode::CAST_NONE);
    } else {
        operandCast = self;
    }

    auto resTensor = Tensor(DataType::DT_FP32, self.GetShape());
    resTensor =
        CALL(UnaryOperation<UnaryOpType::LN>, *Program::GetInstance().GetCurrentFunction(), operandCast.GetStorage());

    auto resTensorBeforeCast = Tensor(DataType::DT_FP32, self.GetShape());
    if (base == LogBaseType::LOG_2) {
        resTensorBeforeCast =
            CALL(BinaryOperationScalar<BinaryOpType::DIV>, *Program::GetInstance().GetCurrentFunction(),
                resTensor.GetStorage(), Element(DataType::DT_FP32, std::log(static_cast<float>(NUM_VALUE_2))));
    } else if (base == LogBaseType::LOG_10) {
        resTensorBeforeCast =
            CALL(BinaryOperationScalar<BinaryOpType::DIV>, *Program::GetInstance().GetCurrentFunction(),
                resTensor.GetStorage(), Element(DataType::DT_FP32, std::log(static_cast<float>(NUM_VALUE_10))));
    } else {
        resTensorBeforeCast = resTensor;
    }

    if (self.GetStorage()->tensor->datatype == DataType::DT_FP16) {
        RETURN_CALL(CastOperation<CastOpType::CAST>, *Program::GetInstance().GetCurrentFunction(),
            resTensorBeforeCast.GetStorage(), DataType::DT_FP16, CastMode::CAST_NONE);
    } else if (self.GetStorage()->tensor->datatype == DataType::DT_BF16) {
        RETURN_CALL(CastOperation<CastOpType::CAST>, *Program::GetInstance().GetCurrentFunction(),
            resTensorBeforeCast.GetStorage(), DataType::DT_BF16, CastMode::CAST_NONE);
    }
    return resTensorBeforeCast;
}

LogicalTensorPtr GenAllOneTensor(const Shape &shape, std::vector<SymbolicScalar> validShape, const DataType &dataType) {
    auto result = CALL(FullOperation, *Program::GetInstance().GetCurrentFunction(),
        Element(DataType::DT_FP32, 1.0), SymbolicScalar(), DataType::DT_FP32, shape, validShape);
    if (dataType != DataType::DT_FP32) {
        RETURN_CALL(CastOperation<CastOpType::CAST>, *Program::GetInstance().GetCurrentFunction(), result.GetStorage(),
            dataType, CastMode::CAST_NONE);
    }
    return result.GetStorage();
}

LogicalTensorPtr IntegerPow(const Tensor &self, int32_t intExponent) {
    // 快速幂
    auto result = GenAllOneTensor(self.GetShape(), self.GetStorage()->GetDynValidShape(), self.GetDataType());
    auto current = CALL(BinaryOperation<BinaryOpType::MUL>, *Program::GetInstance().GetCurrentFunction(), self, result);

    while (intExponent != NUM_VALUE_0) {
        if (intExponent % NUM_VALUE_2 != NUM_VALUE_0) {
            result =
                CALL(BinaryOperation<BinaryOpType::MUL>, *Program::GetInstance().GetCurrentFunction(), result, current);
        }
        current =
            CALL(BinaryOperation<BinaryOpType::MUL>, *Program::GetInstance().GetCurrentFunction(), current, current);
        intExponent /= NUM_VALUE_2;
    }
    return result;
}

LogicalTensorPtr GeneralPow(const Tensor &self, double exponent) {
    // 如果指数小于0，先计算a^(-b)，最后再取倒数
    bool expLessThanZero = exponent < NUM_VALUE_0;
    exponent = std::abs(exponent);

    LogicalTensorPtr result;
    int32_t intExponent = static_cast<int32_t>(std::floor(exponent));
    if (exponent - intExponent < NUM_VALUE_EPS) {
        result = IntegerPow(self, intExponent);
    } else {
        auto exponents = CALL(FullOperation, *Program::GetInstance().GetCurrentFunction(), Element(DataType::DT_FP32, exponent),
            SymbolicScalar(), DataType::DT_FP32, self.GetShape(), self.GetStorage()->GetDynValidShape());
        result = CALL(BinaryOperation<BinaryOpType::POW>, *Program::GetInstance().GetCurrentFunction(), self, exponents);
    }

    // 指数小于零，结果取倒数
    if (expLessThanZero) {
        auto oneTensor = GenAllOneTensor(self.GetShape(), self.GetStorage()->GetDynValidShape(), self.GetDataType());
        // 求倒数
        RETURN_CALL(
            BinaryOperation<BinaryOpType::DIV>, *Program::GetInstance().GetCurrentFunction(), oneTensor, result);
    }
    return result;
}

Tensor Pow(const Tensor &self, const Element &other) {
    DECLARE_TRACER();

    double exponent = other.Cast<double>();
    // 指数为0，输出全1
    if (std::abs(exponent) < NUM_VALUE_EPS) {
        return GenAllOneTensor(self.GetShape(), self.GetStorage()->GetDynValidShape(), self.GetDataType());
    }
    Tensor castSelf = self;
    DataType dataType = self.GetDataType();
    if (dataType != DT_FP32) {
        castSelf = CALL(CastOperation<CastOpType::CAST>, *Program::GetInstance().GetCurrentFunction(), self.GetStorage(),
            DataType::DT_FP32, CastMode::CAST_NONE);
    }
    auto result = castSelf.GetStorage();
    if (std::abs(exponent - NUM_VALUE_0_5) < NUM_VALUE_EPS) {
        result = CALL(UnaryOperation<UnaryOpType::SQRT>, *Program::GetInstance().GetCurrentFunction(), result);
    } else if (std::abs(exponent - NUM_VALUE_2) < NUM_VALUE_EPS) {
        result = CALL(BinaryOperation<BinaryOpType::MUL>, *Program::GetInstance().GetCurrentFunction(), result, result);
    } else if (std::abs(exponent - NUM_VALUE_3) < NUM_VALUE_EPS) {
        auto doubleSelf =
            CALL(BinaryOperation<BinaryOpType::MUL>, *Program::GetInstance().GetCurrentFunction(), result, result);
        result = CALL(BinaryOperation<BinaryOpType::MUL>, *Program::GetInstance().GetCurrentFunction(), doubleSelf, result);
    } else {
        result = GeneralPow(result, exponent);
    }
    if (dataType != DT_FP32) {
        RETURN_CALL(CastOperation<CastOpType::CAST>, *Program::GetInstance().GetCurrentFunction(), result,
            dataType, CastMode::CAST_NONE);
    }
    return result;
}

void TiledOneHot(
    Function &function, const TileShape &tileShape, size_t cur, Input &input, Input &output, int numClasses) {
    if (cur == output.tensor.GetShape().size()) {
        auto inputTile = input.tensor.GetStorage()->View(function, input.tileInfo.shape, input.tileInfo.offset);
        auto outputTile = output.tensor.GetStorage()->View(function, output.tileInfo.shape, output.tileInfo.offset);
        auto &newOp = function.AddOperation(Opcode::OP_ONEHOT, {inputTile}, {outputTile});
        newOp.SetAttribute(OP_ATTR_PREFIX + "numClasses", numClasses);
        return;
    }
    auto &vecTile = tileShape.GetVecTile();
    for (int i = 0; i < output.tensor.GetShape()[cur]; i += vecTile[cur]) {
        if (cur < input.tensor.GetShape().size()) {
            input.tileInfo.shape[cur] = std::min(input.tensor.GetShape()[cur] - i, vecTile[cur]);
            input.tileInfo.offset[cur] = i;
        }
        output.tileInfo.shape[cur] = std::min(output.tensor.GetShape()[cur] - i, vecTile[cur]);
        output.tileInfo.offset[cur] = i;
        TiledOneHot(function, tileShape, cur + 1, input, output, numClasses);
    }
}

void TiledOneHot(Function &function, const TileShape &tileShape, const LogicalTensorPtr &self,
    const LogicalTensorPtr &result, int numClasses) {
    ASSERT(self->shape.size() == self->offset.size()) << "Shape size and offset size should be equal";
    ASSERT(numClasses == tileShape.GetVecTile()[result->shape.size() - 1])
        << "The numClasses and last axis of tileshape should be equal";

    TileInfo inputTileInfo(self->shape.size(), self->offset.size());
    TileInfo outputTileInfo(result->shape.size(), result->offset.size());
    auto input = Input{self, inputTileInfo};
    auto output = Input{result, outputTileInfo};
    TiledOneHot(function, tileShape, 0, input, output, numClasses);
}

Tensor TensorOneHot(Function &function, const LogicalTensorPtr &self, int numClasses) {
    Shape shape(self->shape);
    std::vector<SymbolicScalar> validShape(self->dynValidShape_);
    shape.push_back(static_cast<int64_t>(numClasses));
    validShape.push_back(SymbolicScalar(numClasses));
    auto result = std::make_shared<LogicalTensor>(function, DataType::DT_INT64, shape, validShape);
    auto &op = function.AddOperation(Opcode::OP_ONEHOT, {self}, {result});
    op.SetAttribute(OP_ATTR_PREFIX + "numClasses", numClasses);
    function.UpdateTensorDataUsage(op);
    return result;
}

Tensor OneHot(const Tensor &self, int numClasses) {
    DECLARE_TRACER();

    RETURN_CALL(OneHot, *Program::GetInstance().GetCurrentFunction(), self.GetStorage(), numClasses);
}

void TiledLogicalAndOperation(Function& function, const TileShape& tileShape, size_t cur,
        Input& input0, Input& input1, const LogicalTensorPtr& result, TileInfo &resultTileInfo) {
    if (cur == input0.tensor.GetShape().size()) {
        auto tile0 = input0.tensor.GetStorage()->View(function, input0.tileInfo.shape, input0.tileInfo.offset);
        auto tile1 = input1.tensor.GetStorage()->View(function, input1.tileInfo.shape, input1.tileInfo.offset);
        auto resultTile = result->View(function, resultTileInfo.shape, resultTileInfo.offset);

        constexpr size_t ALIGN_SIZE = 32;
        const int64_t element_per_chunk = 64;
        int64_t vcmp_bits_size = (element_per_chunk + 7) / 8;
        size_t float_array_size = element_per_chunk * SHAPE_DIM4;
        size_t half_array_size = element_per_chunk * SHAPE_DIM2;
        size_t vcmpBitResult_size = ((vcmp_bits_size + ALIGN_SIZE - 1) / ALIGN_SIZE) * ALIGN_SIZE;
        size_t aligned_float_array_size = ((float_array_size + ALIGN_SIZE - 1) / ALIGN_SIZE) * ALIGN_SIZE;
        size_t aligned_half_array_size = ((half_array_size + ALIGN_SIZE - 1) / ALIGN_SIZE) * ALIGN_SIZE;
        size_t total_bytes = vcmpBitResult_size + 4 * aligned_float_array_size + aligned_half_array_size + ALIGN_SIZE;
        std::vector<int64_t> tmp_shape({static_cast<int64_t>(total_bytes)});
        auto tmp_tensor = std::make_shared<LogicalTensor>(function, DT_UINT8, tmp_shape);

        function.AddOperation(Opcode::OP_LOGICALAND, {tile0, tile1}, 
                            {resultTile, tmp_tensor});    
        return;
    }

    auto& vecTile = tileShape.GetVecTile();
    for (int i = 0; i < result->shape[cur]; i += vecTile[cur]) {
        resultTileInfo.offset[cur] = i;
        input0.tileInfo.offset[cur] = i % input0.tensor.GetShape()[cur];
        input1.tileInfo.offset[cur] = i % input1.tensor.GetShape()[cur];
        resultTileInfo.shape[cur] = std::min(result->shape[cur] - resultTileInfo.offset[cur], vecTile[cur]);
        input0.tileInfo.shape[cur] = std::min(input0.tensor.GetShape()[cur] - input0.tileInfo.offset[cur], vecTile[cur]);
        input1.tileInfo.shape[cur] = std::min(input1.tensor.GetShape()[cur] - input1.tileInfo.offset[cur], vecTile[cur]);
        TiledLogicalAndOperation(function, tileShape, cur + 1, input0, input1, result, resultTileInfo);
    }
}

void BroadcastOperand(LogicalTensorPtr &operand, LogicalTensorPtr &other, LogicalTensorPtr result,
                                      Function& function, const TileShape& tileShape) {
    auto dstShape = result->shape;
    if (operand->shape == dstShape) {
        return;
    }
    auto expanded = std::make_shared<LogicalTensor>(function, operand->Datatype(), dstShape);
    Expand(function, tileShape, operand, {other}, expanded);
    operand = expanded;
}

void TiledLogicalAndOperation(Function& function, const TileShape& tileShape, LogicalTensorPtr operand0, LogicalTensorPtr operand1, const LogicalTensorPtr& result) {
    BroadcastOperand(operand0, operand1, result, function, tileShape);
    BroadcastOperand(operand1, operand0, result, function, tileShape);

    TileInfo tileInfo0(result->shape.size(), result->offset.size());
    TileInfo tileInfo1(result->shape.size(), result->offset.size());
    TileInfo resultTileInfo(result->shape.size(), result->offset.size());
    auto input0 = Input{operand0, tileInfo0};
    auto input1 = Input{operand1, tileInfo1};
    TiledLogicalAndOperation(function, tileShape, 0, input0, input1, result, resultTileInfo);
}

LogicalTensorPtr TensorLogicalAndOperation(Function& function, const Tensor& self, const Tensor& other) {
    auto operandT0 = self.GetStorage();
    auto operandT1 = other.GetStorage();
    if (operandT0->shape.size() != operandT1->shape.size()) {
        std::vector<int> broadCastShape = GetBroadCastShape(operandT0, operandT1);
        operandT0 = BinaryOperationBroadCast(operandT0, broadCastShape);
        operandT1 = BinaryOperationBroadCast(operandT1, broadCastShape);
    }

    std::vector<SymbolicScalar> resultValidShape;
    std::vector<int64_t> resultShape = BinaryOperationResultShape(operandT0, operandT1);
    if ((!operandT0->GetDynValidShape().empty()) && (!operandT1->GetDynValidShape().empty())) {
        for (size_t i = 0; i < resultShape.size(); ++i) {
            if (resultShape[i] == operandT0->shape[i]) {
                resultValidShape.push_back(operandT0->GetDynValidShape()[i]);
            } else {
                resultValidShape.push_back(operandT1->GetDynValidShape()[i]);
            }
        }
    }

    auto result = std::make_shared<LogicalTensor>(function, DT_BOOL, resultShape, resultValidShape);
    function.AddOperation(Opcode::OP_LOGICALAND, {operandT0, operandT1}, {result});
    return result;
}

Tensor LogicalAnd(const Tensor &self, const Tensor &other) {
    DECLARE_TRACER();
    RETURN_CALL(LogicalAndOperation, *Program::GetInstance().GetCurrentFunction(), self.GetStorage(), other.GetStorage());
}

void LogicNotOperationTileFunc(Function &function, const TileShape &tileShape,
    const std::vector<LogicalTensorPtr> &iOperand, const std::vector<LogicalTensorPtr> &oOperand,
    [[maybe_unused]] const Operation &op) {
    TiledLogicalNotOperation(function, tileShape, iOperand[0], oOperand[0]);
}

void OneHotOperationTileFunc(Function &function, const TileShape &tileShape,
    const std::vector<LogicalTensorPtr> &iOperand, const std::vector<LogicalTensorPtr> &oOperand,
    [[maybe_unused]] const Operation &op) {
    UnaryOperationOperandCheck(iOperand, oOperand);
    int numClasses = op.GetIntAttribute(OP_ATTR_PREFIX + "numClasses");
    TiledOneHot(function, tileShape, iOperand[0], oOperand[0], numClasses);
}

struct CumSumTileInfoPara {
    TileInfo inputTileInfo;
    TileInfo dstTileInfo;
};

struct CumSumPara {
    const LogicalTensorPtr &Input;
    const LogicalTensorPtr &dstTensor;
    const int axis;
    const bool flag;
};

void InnerTiledCumSum(size_t cur, Function &function, const TileShape &tileShape, const CumSumPara &cumSumPara,
    CumSumTileInfoPara &cumSumTileInfo) {
    const LogicalTensorPtr &input = cumSumPara.Input;
    const LogicalTensorPtr &dstTensor = cumSumPara.dstTensor;
    const int axis = cumSumPara.axis;
    const bool flag = cumSumPara.flag;

    if (cur == dstTensor->shape.size()) {
        auto dstTile = dstTensor->View(function, cumSumTileInfo.dstTileInfo.shape, cumSumTileInfo.dstTileInfo.offset);
        auto inputTile = input->View(function, cumSumTileInfo.inputTileInfo.shape, cumSumTileInfo.inputTileInfo.offset);
        auto &op = function.AddOperation(Opcode::OP_CUM_SUM, {inputTile}, {dstTile});
        op.SetAttribute(OP_ATTR_PREFIX + "axis", axis);
        op.SetAttribute(OP_ATTR_PREFIX + "flag", flag);
        return;
    }

    auto &vecTile = tileShape.GetVecTile();
    int64_t tmpTile = vecTile[cur];

    if (static_cast<int>(cur) == axis) {
        tmpTile = input->GetShape()[cur];
    }

    for (int i = 0; i < input->GetShape()[cur]; i += tmpTile) {
        if (static_cast<int>(cur) == axis) {
            cumSumTileInfo.dstTileInfo.offset[cur] = 0;
            cumSumTileInfo.dstTileInfo.shape[cur] = dstTensor->shape[cur];
            cumSumTileInfo.inputTileInfo.offset[cur] = 0;
            cumSumTileInfo.inputTileInfo.shape[cur] = input->shape[cur];
        } else {
            cumSumTileInfo.dstTileInfo.offset[cur] = i;
            cumSumTileInfo.dstTileInfo.shape[cur] = std::min(input->shape[cur] - i, tmpTile);
            cumSumTileInfo.inputTileInfo.offset[cur] = i;
            cumSumTileInfo.inputTileInfo.shape[cur] = std::min(input->shape[cur] - i, tmpTile);
        }
        InnerTiledCumSum(cur + 1, function, tileShape, cumSumPara, cumSumTileInfo);
    }
}

void TiledCumSum(Function &function, const TileShape &tileShape, const CumSumPara &cumSumPara) {
    assert(cumSumPara.Input->GetShape().size() == cumSumPara.Input->GetOffset().size());

    CumSumTileInfoPara cumSumTileInfo{
        TileInfo(cumSumPara.Input->GetShape().size(), cumSumPara.Input->GetOffset().size()),
        TileInfo(cumSumPara.dstTensor->GetShape().size(), cumSumPara.dstTensor->GetOffset().size())};
    InnerTiledCumSum(0, function, tileShape, cumSumPara, cumSumTileInfo);
    return;
}

void TensorCumSum(Function &function, const CumSumPara &cumSumPara) {
    auto &op = function.AddOperation(Opcode::OP_CUM_SUM, {cumSumPara.Input}, {cumSumPara.dstTensor});
    op.SetAttribute(OP_ATTR_PREFIX + "axis", cumSumPara.axis);
    op.SetAttribute(OP_ATTR_PREFIX + "flag", cumSumPara.flag);
    return;
}

Tensor CumSum(const Tensor &input, const int &axis) {
    DECLARE_TRACER();
    auto shapeSize = input.GetShape().size();
    auto dataType = input.GetDataType();

    ASSERT(SHAPE_DIM1 <= shapeSize && shapeSize <= SHAPE_DIM4) << "The shape.size() only support 1~4";
    std::vector<DataType> CUMSUM_SUPPORT_DATATYPES = {DataType::DT_FP32, DataType::DT_INT32, DataType::DT_INT16};
    ASSERT(std::find(CUMSUM_SUPPORT_DATATYPES.begin(), CUMSUM_SUPPORT_DATATYPES.end(), dataType) !=
           CUMSUM_SUPPORT_DATATYPES.end()) << "The datatype is not supported";
    int tmpAxis = axis < 0 ? shapeSize + axis : axis;
    bool flag = input.GetShape().size() == 1 ? true : false;
    if (flag) {
        ASSERT(tmpAxis == 0) << "when input.GetShape().size() is 1, axis must be 0";
    }
    ASSERT(tmpAxis == 0 || static_cast<size_t>(tmpAxis) < shapeSize) << "The tmpAxis should be 0 and less than shape size";

    Tensor result(input.GetDataType(), input.GetShape());
    CALL(CumSum, *Program::GetInstance().GetCurrentFunction(), {input.GetStorage(), result.GetStorage(), tmpAxis, flag});
    result.GetStorage()->UpdateDynValidShape(input.GetStorage()->dynValidShape_);
    return result;
}

void CumSumOperationTileFunc(Function &function, const TileShape &tileShape,
    const std::vector<LogicalTensorPtr> &iOperand, const std::vector<LogicalTensorPtr> &oOperand, const Operation &op) {
    int axis = op.GetIntAttribute(OP_ATTR_PREFIX + "axis");
    bool flag = op.GetBoolAttribute(OP_ATTR_PREFIX + "flag");
    TiledCumSum(function, tileShape, {iOperand[0], oOperand[0], axis, flag});
}

// beginregin: Clip

Tensor Clip(const Tensor &self, const Element &min, const Element &max) {
    ASSERT(self.GetShape().size() >= SHAPE_DIM2 && self.GetShape().size() <= SHAPE_DIM4) << "The shape.size() only support 2~4";
    std::vector<DataType> CLIP_SUPPORT_DATATYPES = {
        DataType::DT_FP32, DataType::DT_FP16, DataType::DT_INT32, DataType::DT_INT16, DataType::DT_BF16};
    ASSERT(std::find(CLIP_SUPPORT_DATATYPES.begin(), CLIP_SUPPORT_DATATYPES.end(), self.GetDataType()) != 
        CLIP_SUPPORT_DATATYPES.end()) << "The datatype is not supported";

    Element min_ = min, max_ = max;

    Tensor result = self;
    if (min_.GetDataType() != DT_BOTTOM) {
        ASSERT(min_.GetDataType() == self.GetDataType()) << "The datatype of inputs should be same";
        result = Maximum(result, min_);
    }
    if (max_.GetDataType() != DT_BOTTOM) {
        ASSERT(max_.GetDataType() == self.GetDataType()) << "The datatype of inputs should be same";
        result = Minimum(result, max_);
    }
    result.GetStorage()->UpdateDynValidShape(self.GetStorage()->GetDynValidShape());
    return result;
}

Tensor Clip(const Tensor &self, const Tensor &min, const Tensor &max) {
    ASSERT(self.GetShape().size() >= SHAPE_DIM2 && self.GetShape().size() <= SHAPE_DIM4) << "The shape.size() only support 2~4";
    std::vector<DataType> CLIP_SUPPORT_DATATYPES = {
        DataType::DT_FP32, DataType::DT_FP16, DataType::DT_INT32, DataType::DT_INT16};
    ASSERT(std::find(CLIP_SUPPORT_DATATYPES.begin(), CLIP_SUPPORT_DATATYPES.end(), self.GetDataType()) !=
        CLIP_SUPPORT_DATATYPES.end()) << "The datatype is not supported";

    Tensor result = self;
    if (min.GetStorage() != nullptr) {
        ASSERT(min.GetDataType() == self.GetDataType()) << "The datatype of inputs should be same";
        std::vector minBroadcastAxes = GetBroadcastAxes(min.GetShape(), self.GetShape());
        ASSERT(minBroadcastAxes.size() <= 1);
        result = Maximum(result, min);
    }
    if (max.GetStorage() != nullptr) {
        std::vector maxBroadcastAxes = GetBroadcastAxes(max.GetShape(), self.GetShape());
        ASSERT(maxBroadcastAxes.size() <= 1);
        result = Minimum(result, max);
    }
    result.GetStorage()->UpdateDynValidShape(self.GetStorage()->GetDynValidShape());
    return result;
}
// endregion: Clip

void LogicAndOperationTileFunc(Function &function, const TileShape &tileShape,
    const std::vector<LogicalTensorPtr> &iOperand, const std::vector<LogicalTensorPtr> &oOperand,
    [[maybe_unused]] const Operation &op) {
    TiledLogicalAndOperation(function, tileShape, iOperand[0], iOperand[1], oOperand[0]);
}

REGISTER_OPERATION_TILED_FUNC(OP_LOGICALNOT, Opcode::OP_LOGICALNOT, LogicNotOperationTileFunc);
REGISTER_OPERATION_TILED_FUNC(OP_ONEHOT, Opcode::OP_ONEHOT, OneHotOperationTileFunc);
REGISTER_OPERATION_TILED_FUNC(OP_LOGICALAND, Opcode::OP_LOGICALAND, LogicAndOperationTileFunc);
REGISTER_OPERATION_TILED_FUNC(OP_CUM_SUM, Opcode::OP_CUM_SUM, CumSumOperationTileFunc);
} // namespace npu::tile_fwk
