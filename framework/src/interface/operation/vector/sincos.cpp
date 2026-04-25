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
 * \file sincos.cpp
 * \brief
 */

#include "tensor_transformation.h"
#include "interface/utils/operator_tracer.h"
#include "passes/pass_utils/graph_utils.h"

namespace npu::tile_fwk {
Tensor TensorTrig(Function& function, const std::string& op, const LogicalTensorPtr& operand1, const LogicalTensorPtr& result)
{
    ASSERT(
        VectorErrorCode::ERR_PARAM_INVALID, op == "SIN" || op == "COS")
        << "Not support op:" << op;
    auto opCode = Opcode::OP_SIN;
    if (op == "SIN") {
        opCode = Opcode::OP_SIN;
    } else {
        opCode = Opcode::OP_COS;
    }
    auto& opNew = function.AddOperation(opCode, {operand1}, {result});
    function.UpdateTensorDataUsage(opNew);
    return result;
}

Tensor Sin(const Tensor& self)
{
    DECLARE_TRACER();
    std::unordered_set<DataType> supportedTypes = {DT_FP16, DT_FP32};
    CheckTensorDataType(self.GetStorage(), supportedTypes, "Sin");
    CheckTensorDimRange(self.GetStorage(), 1, 4, "Sin");

    Tensor castSelf = self;
    if (self.GetDataType() == DataType::DT_FP16) {
        castSelf = Cast(self, DataType::DT_FP32, CastMode::CAST_NONE);
    }
    Tensor result(castSelf.GetStorage()->tensor->datatype, castSelf.GetShape());
    result.GetStorage()->UpdateDynValidShape(castSelf.GetValidShape());
    CALL(Trig, *Program::GetInstance().GetCurrentFunction(), "SIN", castSelf.GetStorage(), result.GetStorage());
    Tensor castResult = result;
    if (self.GetDataType() == DataType::DT_FP16) {
        castResult = Cast(result, DataType::DT_FP16, CastMode::CAST_NONE);
    }
    return castResult;
}

Tensor Cos(const Tensor& self)
{
    DECLARE_TRACER();
    std::unordered_set<DataType> supportedTypes = {DT_FP16, DT_FP32};
    CheckTensorDataType(self.GetStorage(), supportedTypes, "Cos");
    CheckTensorDimRange(self.GetStorage(), 1, 4, "Cos");

    Tensor castSelf = self;
    if (self.GetDataType() == DataType::DT_FP16) {
        castSelf = Cast(self, DataType::DT_FP32, CastMode::CAST_NONE);
    }
    Tensor result(castSelf.GetStorage()->tensor->datatype, castSelf.GetShape());
    result.GetStorage()->UpdateDynValidShape(castSelf.GetValidShape());
    CALL(Trig, *Program::GetInstance().GetCurrentFunction(), "COS", castSelf.GetStorage(), result.GetStorage());
    Tensor castResult = result;
    if (self.GetDataType() == DataType::DT_FP16) {
        castResult = Cast(result, DataType::DT_FP16, CastMode::CAST_NONE);
    }
    return castResult;
}

void TiledTrig(
    Function& function, const TileShape& tileShape, size_t cur, Input& input,
    const LogicalTensorPtr& result, const std::string& op)
{
    ASSERT(
        VectorErrorCode::ERR_PARAM_INVALID, op == "SIN" || op == "COS")
        << "Not support op:" << op;
    auto opCode = Opcode::OP_SIN;
    if (op == "SIN") {
        opCode = Opcode::OP_SIN;
    } else {
        opCode = Opcode::OP_COS;
    }
    if (cur == input.tensor.GetShape().size()) {
        auto tile = input.tensor.GetStorage()->View(function, input.tileInfo.shape, input.tileInfo.offset);
        auto resultTile = result->View(function, input.tileInfo.shape, input.tileInfo.offset);
        std::vector<int64_t> srcTileShape(input.tileInfo.shape);
        std::vector<int64_t> tmpShape;
        std::vector<int64_t> tmpShape2;

        tmpShape.assign(srcTileShape.end() - SHAPE_DIM2, srcTileShape.end());
        tmpShape2.assign(srcTileShape.end() - SHAPE_DIM2, srcTileShape.end());
        auto alignSize = BLOCK_SIZE / BytesOf(DT_FP32);
        tmpShape2[tmpShape2.size() - 1] = (tmpShape2[tmpShape2.size() - 1] + alignSize - 1) / alignSize * alignSize;
        tmpShape[tmpShape.size() - 1] = (tmpShape[tmpShape.size() - 1] + alignSize - 1) / alignSize * alignSize;

        if (input.tensor.GetShape().size() == 1) {
            tmpShape = {tmpShape[1]};
            tmpShape2 = {tmpShape2[1]};
        }

        auto tmpTensor = std::make_shared<LogicalTensor>(function, DT_FP32, tmpShape);
        auto tmpTensorNext = std::make_shared<LogicalTensor>(function, DT_INT32, tmpShape2);
        function.AddOperation(opCode, {tile}, {resultTile, tmpTensor, tmpTensorNext});
        return;
    }
    auto& vecTile = tileShape.GetVecTile();
    for (int i = 0; i < input.tensor.GetShape()[cur]; i += vecTile[cur]) {
        input.tileInfo.shape[cur] = std::min(input.tensor.GetShape()[cur] - i, vecTile[cur]);
        input.tileInfo.offset[cur] = i;
        TiledTrig(function, tileShape, cur + 1, input, result, op);
    }
}

void TiledTrig(
    Function& function, const TileShape& tileShape, LogicalTensorPtr operand1,
    const LogicalTensorPtr& result, const std::string& op)
{
    TileInfo tileInfo(result->shape.size(), result->offset.size());
    auto input = Input{operand1, tileInfo};
    TiledTrig(function, tileShape, 0, input, result, op);
}

void SinOperationTileFunc(
    Function& function, const TileShape& tileShape, const std::vector<LogicalTensorPtr>& iOperand,
    const std::vector<LogicalTensorPtr>& oOperand, [[maybe_unused]] const Operation& op)
{
    TiledTrig(function, tileShape, iOperand[0], oOperand[0], "SIN");
}

void CosOperationTileFunc(
    Function& function, const TileShape& tileShape, const std::vector<LogicalTensorPtr>& iOperand,
    const std::vector<LogicalTensorPtr>& oOperand, [[maybe_unused]] const Operation& op)
{
    TiledTrig(function, tileShape, iOperand[0], oOperand[0], "COS");
}
REGISTER_OPERATION_TILED_FUNC(OP_SIN, Opcode::OP_SIN, SinOperationTileFunc);
REGISTER_OPERATION_TILED_FUNC(OP_COS, Opcode::OP_COS, CosOperationTileFunc);
} // namespace npu::tile_fwk
