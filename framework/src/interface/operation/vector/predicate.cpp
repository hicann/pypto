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
 * \file predicate.cpp
 * \brief
 */

#include "unary_tiled.h"
#include "binary.h"
#include "tensor_transformation.h"
#include "interface/utils/operator_tracer.h"
#include "passes/pass_utils/graph_utils.h"
#include "tilefwk/error_code.h"

namespace npu::tile_fwk {

Tensor IsFinite(const Tensor& self)
{
    DECLARE_TRACER();
    CheckTensorFormat(self.GetStorage(), {TileOpFormat::TILEOP_NZ}, "IsFinite");

    std::unordered_set<DataType> supportedTypes = {DT_FP16,  DT_FP32,   DT_BF16,   DT_INT16, DT_INT4,   DT_INT8,
                                                   DT_INT32, DT_UINT16, DT_UINT32, DT_UINT8, DT_UINT64, DT_INT64};
    CheckTensorDataType(self.GetStorage(), supportedTypes, "IsFinite");
    RETURN_CALL(UnaryOperation<UnaryOpType::ISFINITE>, *Program::GetInstance().GetCurrentFunction(), self.GetStorage(),
                DT_BOOL);
}

Tensor IsNan(const Tensor& self)
{
    DECLARE_TRACER();
    CheckTensorFormat(self.GetStorage(), {TileOpFormat::TILEOP_NZ}, "IsNan");

    std::unordered_set<DataType> supportedTypes = {DT_FP16, DT_FP32, DT_BF16};
    CheckTensorDataType(self.GetStorage(), supportedTypes, "ISNAN");

    RETURN_CALL(UnaryOperation<UnaryOpType::ISNAN>, *Program::GetInstance().GetCurrentFunction(), self.GetStorage(),
                DT_BOOL);
}

void IsFiniteOperationTileFunc(Function& function, const TileShape& tileShape,
                               const std::vector<LogicalTensorPtr>& iOperand,
                               const std::vector<LogicalTensorPtr>& oOperand, [[maybe_unused]] const Operation& op)
{
    UnaryOperationOperandCheck(iOperand, oOperand);
    auto shape = tileShape.GetVecTile().tile;
    // tileShape 对应的中间变量结果，类型为 FP16
    uint32_t intermediateBytes = static_cast<int64_t>(BytesOf(DT_FP16)) *
                                 std::accumulate(shape.begin(), shape.end(), 1LL, std::multiplies<int64_t>());
    uint32_t workspaceSize = intermediateBytes;
    return TiledUnaryOperation<UnaryOpType::ISFINITE>(function, tileShape, iOperand[0], oOperand[0], workspaceSize);
}

void IsNanOperationTileFunc(Function& function, const TileShape& tileShape,
                            const std::vector<LogicalTensorPtr>& iOperand,
                            const std::vector<LogicalTensorPtr>& oOperand, [[maybe_unused]] const Operation& op)
{
    UnaryOperationOperandCheck(iOperand, oOperand);
    auto tmpShape = tileShape.GetVecTile().tile;
    int dim = static_cast<int>(tmpShape.size());
    auto alignSize = BLOCK_SIZE / BytesOf(DT_FP16);
    int64_t tmpW = AlignUp(tmpShape[dim - 1], alignSize);
    int64_t tmpH = (dim >= NUM_VALUE_2) ? tmpShape[dim - NUM_VALUE_2] : 1;

    constexpr int64_t kNumBlocks = NUM_VALUE_3;
    int64_t blockBytes = tmpH * tmpW * BytesOf(DT_FP32);
    uint32_t workspaceSize = kNumBlocks * blockBytes + BLOCK_SIZE;
    return TiledUnaryOperation<UnaryOpType::ISNAN>(function, tileShape, iOperand[0], oOperand[0], workspaceSize);
}

REGISTER_OPERATION_TILED_FUNC(OP_ISFINITE, Opcode::OP_ISFINITE, IsFiniteOperationTileFunc);
REGISTER_OPERATION_TILED_FUNC(OP_ISNAN, Opcode::OP_ISNAN, IsNanOperationTileFunc);

} // namespace npu::tile_fwk
