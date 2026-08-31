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
 * \file unary_tiled.h
 * \brief
 */

#pragma once

#include "unary.h"

namespace npu::tile_fwk {

template <UnaryOpType T>
void TiledUnaryOperation(Function& function, const TileShape& tileShape, size_t cur, Input& input,
                         const LogicalTensorPtr& result, uint32_t workspaceSize = 0, int64_t precisionType = 0)
{
    if (cur == input.tensor.GetShape().size()) {
        auto tile = input.tensor.GetStorage()->View(function, input.tileInfo.shape, input.tileInfo.offset);
        auto resultTile = result->View(function, input.tileInfo.shape, input.tileInfo.offset);
        Operation* op = nullptr;
        if (workspaceSize == 0) {
            op = &function.AddOperation(GetUnaryOpNameCode<T>(), {tile}, {resultTile});
        } else {
            LogicalTensorPtr workspace = std::make_shared<LogicalTensor>(function, DT_UINT8,
                                                                         std::vector<int64_t>{workspaceSize});
            op = &function.AddOperation(GetUnaryOpNameCode<T>(), {tile}, {resultTile, workspace});
        }
        if (T == UnaryOpType::EXP || T == UnaryOpType::SQRT || T == UnaryOpType::LN || T == UnaryOpType::RECIPROCAL) {
            op->SetAttribute(OpAttributeKey::precisionType, precisionType);
        }
        if (T == UnaryOpType::ASIN || T == UnaryOpType::ACOS || T == UnaryOpType::SINH || T == UnaryOpType::ERF ||
            T == UnaryOpType::ASINH || T == UnaryOpType::ATANH || T == UnaryOpType::ISNAN) {
            std::vector<bool> dimMap({true});
            op->SetAttr(OpAttributeKey::rowPad, dimMap);
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
void TiledUnaryOperation(Function& function, const TileShape& tileShape, const LogicalTensorPtr& operand,
                         const LogicalTensorPtr& result, int32_t workspaceSize = 0, int64_t precisionType = 0)
{
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, operand->shape.size() == operand->offset.size())
        << "The shape size of operand and offset must be equal";

    TileInfo tileInfo(result->shape.size(), result->offset.size());
    auto input = Input{operand, tileInfo};
    TiledUnaryOperation<T>(function, tileShape, 0, input, result, workspaceSize, precisionType);
}

} // namespace npu::tile_fwk
