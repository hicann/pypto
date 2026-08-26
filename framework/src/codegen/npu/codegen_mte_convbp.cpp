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
 * \file codegen_mte_convbp.cpp
 * \brief
 */
#include <iterator>
#include <string>

#include "codegen_op_npu.h"
#include "codegen/symbol_mgr/codegen_symbol.h"
#include "codegen/utils/codegen_utils.h"
#include "tilefwk/error_code.h"
#include "securec.h"

namespace npu::tile_fwk {

std::string CodeGenOpNPU::GenMemL1ToL0Load2DConvBpDx() const
{
    std::vector<std::variant<std::string, uint8_t, uint16_t, int, int64_t>> paramList;

    auto tileOpParams = GetTileOpParamsByOrder();
    paramList.insert(paramList.end(), tileOpParams.begin(), tileOpParams.end());

    int64_t hkw = 0;
    int64_t kL0Size = 0;
    int64_t nL0Size = 0;
    int64_t k0Idx = 0;
    int64_t n0Idx = 0;
    GetOpAttr(OpAttributeKey::hwk, hkw);
    GetOpAttr(OpAttributeKey::kL0Size, kL0Size);
    GetOpAttr(OpAttributeKey::nL0Size, nL0Size);
    GetOpAttr(OpAttributeKey::k0Idx, k0Idx);
    GetOpAttr(OpAttributeKey::n0Idx, n0Idx);

    paramList.emplace_back(kL0Size);
    paramList.emplace_back(nL0Size);
    paramList.emplace_back(hkw);
    paramList.emplace_back(k0Idx);
    paramList.emplace_back(n0Idx);

    std::ostringstream oss;
    oss << tileOpName;
    oss << WrapParamByParentheses(paramList) << STMT_END;
    return oss.str();
}

std::string CodeGenOpNPU::GenMemL1CopyInConvBp() const
{
    auto tileOpParams = GetTileOpParamsByOrder();

    int64_t strideH = 1;
    int64_t strideW = 1;
    int64_t skipH = 0;
    int64_t skipW = 0;
    GetOpAttr(OpAttributeKey::strideH, strideH);
    GetOpAttr(OpAttributeKey::strideW, strideW);
    GetOpAttr(OpAttributeKey::skipH, skipH);
    GetOpAttr(OpAttributeKey::skipW, skipW);

    bool isConv3D = false;
    GetOpAttr(Conv::LoadStoreConvOpAttributeKey::isConv3D, isConv3D);

    auto dynOffset = GetOffsetFromAttr(ToUnderlying(MISOIdx::SRC0_IDX));
    size_t expectedDim = isConv3D ? SHAPE_DIM5 : SHAPE_DIM4;
    ASSERT(ConvCodenGenError::CODEGEN_CHECK_DIM_INVALID, dynOffset.size() == expectedDim + 1)
        << "GenMemL1CopyInConvBp offset should be " << (expectedDim + 1) << "-dim!";
    std::vector<SymbolicScalar> srcShapeVec;
    GetOpAttr(OpAttributeKey::srcGmConvValidShape, srcShapeVec);
    ASSERT(ConvCodenGenError::CODEGEN_CHECK_DIM_INVALID, srcShapeVec.size() == expectedDim + 1)
        << "GenMemL1CopyInConvBp shape should be " << (expectedDim + 1) << "-dim!";

    std::vector<std::string> offsetExpr;
    std::vector<std::string> shapeExpr;
    for (size_t i = 0; i < expectedDim; i++) {
        offsetExpr.emplace_back(ToStringHelper(dynOffset[i]));
        shapeExpr.emplace_back(ToStringHelper(srcShapeVec[i]));
    }
    if (!isConv3D) {
        offsetExpr.insert(offsetExpr.begin() + 1, "0");
        shapeExpr.insert(shapeExpr.begin() + 1, "1");
    }

    std::vector<std::string> paramList;
    paramList.emplace_back(tileOpParams[ToUnderlying(MISOIdx::DST_IDX)]);
    paramList.emplace_back(tileOpParams[ToUnderlying(MISOIdx::SRC0_IDX)]);
    paramList.insert(paramList.end(), offsetExpr.begin(), offsetExpr.end());
    paramList.insert(paramList.end(), shapeExpr.begin(), shapeExpr.end());
    paramList.emplace_back(std::to_string(strideH));
    paramList.emplace_back(std::to_string(strideW));
    paramList.emplace_back(std::to_string(skipH));
    paramList.emplace_back(std::to_string(skipW));

    std::ostringstream oss;
    oss << tileOpName << WrapParamByAngleBrackets({std::to_string(isConv3D)});
    oss << WrapParamByParentheses(paramList) << STMT_END;
    return oss.str();
}

std::string CodeGenOpNPU::GenMemL1CopyInConvBpNZ() const
{
    auto tileOpParams = GetTileOpParamsByOrder();

    std::vector<std::string> paramList;
    paramList.emplace_back(tileOpParams[ToUnderlying(MISOIdx::DST_IDX)]);
    paramList.emplace_back(tileOpParams[ToUnderlying(MISOIdx::SRC0_IDX)]);

    auto dynOffset = GetOffsetFromAttr(ToUnderlying(MISOIdx::SRC0_IDX));

    std::vector<std::string> gmOffsetExpr = GenSymbolicArgument(dynOffset);
    for (size_t i = 0; i < SHAPE_DIM4; i++) {
        paramList.emplace_back(gmOffsetExpr[i]);
    }

    std::ostringstream oss;
    oss << tileOpName;
    oss << WrapParamByParentheses(paramList) << STMT_END;
    return oss.str();
}

} // namespace npu::tile_fwk
