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
 * \file codegen_mte_conv.cpp
 * \brief
 */
#include <iterator>
#include <string>

#include "codegen_op_cloudnpu.h"
#include "codegen/symbol_mgr/codegen_symbol.h"
#include "codegen/utils/codegen_utils.h"
#include "interface/utils/conv_error.h"
#include "securec.h"

namespace npu::tile_fwk {
std::string CodeGenOpCloudNPU::GetConvCopyInMode() const
{
    int64_t copyInMode = -1;
    auto ret = GetAttr(Conv::LoadStoreConvOpAttributeKey::copyInMode, copyInMode);
    ASSERT(ConvCodenGenError::CODEGEN_GET_ATTR_FAILED, ret) << "GenMemL1CopyInConv get CopyInMode failed.";
    bool isValidMode =
        copyInMode >= ToUnderlying(Matrix::CopyInMode::ND2NZ) && copyInMode <= ToUnderlying(Matrix::CopyInMode::DN2NZ);
    ASSERT(ConvCodenGenError::CODEGEN_CHECK_ATTR_INVALID, isValidMode)
        << "GenMemL1CopyInConv CopyInMode is invalid: " << copyInMode;
    return CopyInModeToString(static_cast<Matrix::CopyInMode>(copyInMode));
}

std::string CodeGenOpCloudNPU::GenMemL1CopyInConv() const
{
    std::string srcTensor = QueryTileTensorNameByIdx(ToUnderlying(MISOIdx::SRC0_IDX));
    std::string dstTensor = QueryTileTensorNameByIdx(ToUnderlying(MISOIdx::DST_IDX));
    std::string copyInModeStr = GetConvCopyInMode();

    bool isFmap = true, isConv3D = false;
    int64_t offsetN = 0, offsetC = 0, offsetD = 0, offsetH = 0, offsetW = 0;
    int64_t srcShapeN = 0, srcShapeC = 0, srcShapeD = 0, srcShapeH = 0, srcShapeW = 0;
    GetAttr(Conv::LoadStoreConvOpAttributeKey::isFmap, isFmap);
    GetAttr(Conv::LoadStoreConvOpAttributeKey::isConv3D, isConv3D);
    auto dynOffset = offsetFromAttr[ToUnderlying(MISOIdx::SRC0_IDX)];
    auto srcShape = shapeFromAttr[ToUnderlying(MISOIdx::SRC0_IDX)];
    if (isConv3D) {
        ASSERT(ConvCodenGenError::CODEGEN_CHECK_DIM_INVALID, dynOffset.size() == SHAPE_DIM5)
            << "GenMemL1CopyInConv offset should be 5-dim!";
        ASSERT(ConvCodenGenError::CODEGEN_CHECK_DIM_INVALID, srcShape.size() == SHAPE_DIM5)
            << "GenMemL1CopyInConv shape should be 5-dim!";
        offsetN = dynOffset[ID0].Concrete();
        offsetC = dynOffset[ID1].Concrete();
        offsetD = dynOffset[ID2].Concrete();
        offsetH = dynOffset[ID3].Concrete();
        offsetW = dynOffset[ID4].Concrete();
        srcShapeN = srcShape[ID0];
        srcShapeC = srcShape[ID1];
        srcShapeD = srcShape[ID2];
        srcShapeH = srcShape[ID3];
        srcShapeW = srcShape[ID4];
    } else {
        ASSERT(ConvCodenGenError::CODEGEN_CHECK_DIM_INVALID, dynOffset.size() == SHAPE_DIM4)
            << "GenMemL1CopyInConv offset should be 4-dim!";
        ASSERT(ConvCodenGenError::CODEGEN_CHECK_DIM_INVALID, srcShape.size() == SHAPE_DIM4)
            << "GenMemL1CopyInConv shape should be 4-dim!";
        offsetN = dynOffset[ID0].Concrete();
        offsetC = dynOffset[ID1].Concrete();
        offsetH = dynOffset[ID2].Concrete();
        offsetW = dynOffset[ID3].Concrete();
        srcShapeN = srcShape[ID0];
        srcShapeC = srcShape[ID1];
        srcShapeH = srcShape[ID2];
        srcShapeW = srcShape[ID3];
    }

    std::vector<std::string> tileOpParamList = {
        dstTensor,
        srcTensor,
        std::to_string(offsetN),
        std::to_string(offsetC),
        std::to_string(offsetD),
        std::to_string(offsetH),
        std::to_string(offsetW),
        std::to_string(srcShapeN),
        std::to_string(srcShapeC),
        std::to_string(srcShapeD),
        std::to_string(srcShapeH),
        std::to_string(srcShapeW)};

    std::ostringstream oss;
    oss << tileOpName << WrapParamByAngleBrackets({copyInModeStr, std::to_string(isConv3D), std::to_string(isFmap)});
    oss << WrapParamByParentheses(tileOpParamList) << STMT_END;
    return oss.str();
}

std::string CodeGenOpCloudNPU::GetConvCopyOutMode() const
{
    int64_t copyOutMode = -1;
    auto ret = GetAttr(Conv::LoadStoreConvOpAttributeKey::copyOutMode, copyOutMode);
    ASSERT(ConvCodenGenError::CODEGEN_GET_ATTR_FAILED, ret) << "GenMemL1CopyOutConv get CopyOutMode failed.";
    bool isValidMode = copyOutMode == ToUnderlying(Matrix::CopyOutMode::NZ2ND) ||
                       copyOutMode == ToUnderlying(Matrix::CopyOutMode::NZ2NZ) ||
                       copyOutMode == ToUnderlying(Matrix::CopyOutMode::NZ2DN);
    ASSERT(ConvCodenGenError::CODEGEN_CHECK_ATTR_INVALID, isValidMode)
        << "GenMemL1CopyOutConv CopyOutMode is invalid: " << copyOutMode;
    return CopyOutModeToString(static_cast<Matrix::CopyOutMode>(copyOutMode));
}

std::string CodeGenOpCloudNPU::GenMemL1CopyOutConv() const
{
    std::string dstTensor = QueryTileTensorNameByIdx(ToUnderlying(MISOIdx::DST_IDX));
    std::string srcTensor = QueryTileTensorNameByIdx(ToUnderlying(MISOIdx::SRC0_IDX));
    std::string copyOutModeStr = GetConvCopyOutMode();

    bool isConv3D = false;
    int64_t realM = 0, realN = 0;
    int64_t offsetN = 0, offsetC = 0, offsetD = 0, offsetH = 0, offsetW = 0;
    GetAttr(Conv::LoadStoreConvOpAttributeKey::isConv3D, isConv3D);
    auto realShape = shapeFromAttr[ToUnderlying(MISOIdx::DST_IDX)];
    ASSERT(ConvCodenGenError::CODEGEN_CHECK_DIM_INVALID, realShape.size() == SHAPE_DIM2)
        << "GenMemL1CopyOutConv valid shape should be 2-dim!";
    realM = realShape[ID0];
    realN = realShape[ID1];
    auto dynOffset = offsetFromAttr[ToUnderlying(MISOIdx::DST_IDX)];
    if (isConv3D) {
        ASSERT(ConvCodenGenError::CODEGEN_CHECK_DIM_INVALID, dynOffset.size() == SHAPE_DIM5)
            << "GenMemL1CopyOutConv offset should be 5-dim!";
        offsetN = dynOffset[ID0].Concrete();
        offsetC = dynOffset[ID1].Concrete();
        offsetD = dynOffset[ID2].Concrete();
        offsetH = dynOffset[ID3].Concrete();
        offsetW = dynOffset[ID4].Concrete();
    } else {
        ASSERT(ConvCodenGenError::CODEGEN_CHECK_DIM_INVALID, dynOffset.size() == SHAPE_DIM4)
            << "GenMemL1CopyOutConv offset should be 4-dim!";
        offsetN = dynOffset[ID0].Concrete();
        offsetC = dynOffset[ID1].Concrete();
        offsetH = dynOffset[ID2].Concrete();
        offsetW = dynOffset[ID3].Concrete();
    }
    std::vector<std::string> tileOpParamList = {
        dstTensor,
        srcTensor,
        std::to_string(offsetN),
        std::to_string(offsetC),
        std::to_string(offsetD),
        std::to_string(offsetH),
        std::to_string(offsetW),
        std::to_string(realM),
        std::to_string(realN)};

    std::ostringstream oss;
    oss << tileOpName << WrapParamByAngleBrackets({copyOutModeStr, std::to_string(isConv3D)});
    oss << WrapParamByParentheses(tileOpParamList) << STMT_END;
    return oss.str();
}

std::string CodeGenOpCloudNPU::GenMemL1ToL0Load3D() const
{
    std::vector<std::variant<std::string, uint8_t, uint16_t, int, int64_t>> paramList;

    std::string dstVar = QueryTileTensorNameByIdx(ToUnderlying(MISOIdx::DST_IDX));
    std::string srcVar = QueryTileTensorNameByIdx(ToUnderlying(MISOIdx::SRC0_IDX));
    paramList.emplace_back(dstVar);
    paramList.emplace_back(srcVar);

    int64_t mPos = 0, kPos = 0;
    GetAttr(Conv::L12L0ConvOpAttributeKey::postM, mPos);
    GetAttr(Conv::L12L0ConvOpAttributeKey::postK, kPos);
    paramList.emplace_back(mPos);
    paramList.emplace_back(kPos);

    std::vector<int64_t> fmapL1Shape = rawShape[ID1];
    CODEGEN_LOGI("GenMemL1ToL0Load3D %s, fmapL1Shape is %s", tileOpName.c_str(), IntVecToStr(fmapL1Shape).c_str());

    int64_t padLeft = 0, padRight = 0, padTop = 0, padBottom = 0, padValue = 0;
    GetAttr(Conv::L12L0ConvOpAttributeKey::paddingLeft, padLeft);
    GetAttr(Conv::L12L0ConvOpAttributeKey::paddingRight, padRight);
    GetAttr(Conv::L12L0ConvOpAttributeKey::paddingTop, padTop);
    GetAttr(Conv::L12L0ConvOpAttributeKey::paddingBottom, padBottom);
    GetAttr(Conv::L12L0ConvOpAttributeKey::padValue, padValue);
    paramList.emplace_back(padLeft);
    paramList.emplace_back(padRight);
    paramList.emplace_back(padTop);
    paramList.emplace_back(padBottom);
    paramList.emplace_back(padValue);

    int64_t filterH = 0, filterW = 0;
    GetAttr(Conv::L12L0ConvOpAttributeKey::filterH, filterH);
    GetAttr(Conv::L12L0ConvOpAttributeKey::filterW, filterW);
    paramList.emplace_back(filterH);
    paramList.emplace_back(filterW);

    int64_t dilationH = 0, dilationW = 0;
    GetAttr(Conv::L12L0ConvOpAttributeKey::dilationH, dilationH);
    GetAttr(Conv::L12L0ConvOpAttributeKey::dilationW, dilationW);
    paramList.emplace_back(dilationH);
    paramList.emplace_back(dilationW);

    int64_t strideH = 0, strideW = 0;
    GetAttr(Conv::L12L0ConvOpAttributeKey::strideH, strideH);
    GetAttr(Conv::L12L0ConvOpAttributeKey::strideW, strideW);
    paramList.emplace_back(strideH);
    paramList.emplace_back(strideW);

    std::vector<int64_t> fmapL0Shape = rawShape[ID0];
    CODEGEN_LOGI("GenMemL1ToL0Load3D %s, fmapL0Shape is %s", tileOpName.c_str(), IntVecToStr(fmapL0Shape).c_str());
    ASSERT(ConvCodenGenError::CODEGEN_CHECK_DIM_INVALID, fmapL0Shape.size() == SHAPE_DIM2)
        << "GenMemL1ToL0Load3D L0 fmap only support 2-dim!";

    bool isConv3D = false;
    GetAttr(Conv::LoadStoreConvOpAttributeKey::isConv3D, isConv3D);

    std::ostringstream oss;
    oss << tileOpName.c_str() << WrapParamByAngleBrackets({std::to_string(isConv3D)});
    oss << WrapParamByParentheses(paramList) << STMT_END;
    return oss.str();
}

std::string CodeGenOpCloudNPU::GenMemL1ToL0Load2D() const
{
    std::vector<std::variant<std::string, uint16_t, int, int64_t>> paramList;

    std::string dstVar = QueryTileTensorNameByIdx(ToUnderlying(MISOIdx::DST_IDX));
    std::string srcVar = QueryTileTensorNameByIdx(ToUnderlying(MISOIdx::SRC0_IDX));
    paramList.emplace_back(dstVar);
    paramList.emplace_back(srcVar);

    int64_t kPos = 0, nPos = 0;
    GetAttr(Conv::L12L0ConvOpAttributeKey::postK, kPos);
    GetAttr(Conv::L12L0ConvOpAttributeKey::postN, nPos);
    paramList.emplace_back(kPos);
    paramList.emplace_back(nPos);

    std::vector<int64_t> weightL1Shape = rawShape[ID1];
    CODEGEN_LOGI("GenMemL1ToL0Load2D %s, weightL1Shape is %s", tileOpName.c_str(), IntVecToStr(weightL1Shape).c_str());

    std::vector<int64_t> weightL0Shape = rawShape[ID0];
    CODEGEN_LOGI("GenMemL1ToL0Load2D %s, weightL0Shape is %s", tileOpName.c_str(), IntVecToStr(weightL0Shape).c_str());
    ASSERT(ConvCodenGenError::CODEGEN_CHECK_DIM_INVALID, weightL0Shape.size() == SHAPE_DIM2)
        << "GenMemL1ToL0Load2D L0 weight only support 2-dim!";

    std::ostringstream oss;
    oss << tileOpName.c_str() << WrapParamByParentheses(paramList) << STMT_END;
    return oss.str();
}

} // namespace npu::tile_fwk
