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
 * \file codegen_op_litenpu.cpp
 * \brief
 */

#include "codegen_op_litenpu.h"
#include "codegen/npu/litenpu/codegen_litenpu.h"

namespace npu::tile_fwk {

// ensure funcType is static, and isUnderDynamicFunc is false
CodeGenOpLiteNPU::CodeGenOpLiteNPU(const CodeGenOpNPUCtx& ctx) : CodeGenOpNPU(ctx)
{
    InitOpsGenMap();
    forBlkMgr_ = ctx.forBlockManager;
    CodeGenOp::Init(ctx.operation);
    UpdateTileTensorInfo();
    UpdateLoopInfo();
}

TileTensor CodeGenOpLiteNPU::QueryTileTensorByIdx(int paramIdx) const
{
    const int tensorMagic = operandWithMagic[paramIdx];
    const int opMagic = originalOp.GetOpMagic();
    const TileTensor* tileTensor = nullptr;

    tileTensor = sm->QueryTileTensorByMagic(tensorMagic, opMagic);

    if (tileTensor != nullptr) {
        CODEGEN_LOGI("QueryTileTensorByIdx found: %s", tileTensor->ToString().c_str());
        return *tileTensor;
    }

    ASSERT(GenCodeErr::TENSOR_NOT_FOUND, false) << "TileTensor: paramIdx " << paramIdx << ", tensor magic "
                                                << tensorMagic << ", op magic " << opMagic << " is not found !!!";
    static TileTensor emptyTileTensor;
    return emptyTileTensor;
}

std::string CodeGenOpLiteNPU::GenGmParamVar(unsigned gmParamIdx) const
{
    return std::string("RealizedGM") + std::to_string(paramLocation[gmParamIdx]) + ".Addr";
}

TileTensor CodeGenOpLiteNPU::BuildTileTensor(
    int paramIdx, const std::string& usingType, const TileTensorShape& tileTensorShape)
{
    bool isSpillToGm = operand[paramIdx] == SYMBOL_STACK_BASE;

    TileTensor tileTensor;
    tileTensor.isConstant = functionType == FunctionType::STATIC || isMainBlock;
    tileTensor.magic = operandWithMagic[paramIdx];
    tileTensor.isInLoop = tileTensorShape.isInLoop;

    if (tileTensor.isConstant) {
        tileTensor.dim = originShape[paramIdx].size();
    } else {
        tileTensor.dim = dynamicValidShape[paramIdx].size();
    }

    tileTensor.dtype = operandDtype[paramIdx];
    tileTensor.bufType = operandType[paramIdx];

    if (tileTensor.bufType == OperandType::BUF_DDR) {
        tileTensor.bufVar = isSpillToGm ? GenGMAddrExprWithOffset(paramIdx, GM_STACK_BASE) : GenGmParamVar(paramIdx);
    } else {
        tileTensor.bufVar = sm->QueryVarNameByTensorMagic(tileTensor.magic, true);
    }

    tileTensor.usingType = usingType;

    tileTensor.tensorName = sm->GenTensorName(tileTensor.bufType);
    UpdateTileTensorShapeAndStride(paramIdx, tileTensor, isSpillToGm, tileTensorShape);

    tileTensor.localBufOffset = offset[paramIdx];

    return tileTensor;
}

void CodeGenOpLiteNPU::UpdateTileTensorShapeAndStride(
    int paramIdx, TileTensor& tileTensor, [[maybe_unused]] bool isSpillToGm,
    [[maybe_unused]] const TileTensorShape& tileTensorShape)
{
    auto newOriginShape = originShape[paramIdx];
    auto newRawShape = tileTensorShape.rawShape;
    auto newDynValidShape = tileTensorShape.dynamicValidShape;
    CODEGEN_LOGI(
        "newOriginShape is %s, newRawShape is %s, newDynValidShape is %s", IntVecToStr(newOriginShape).c_str(),
        IntVecToStr(newRawShape).c_str(), IntVecToStr(newDynValidShape).c_str());

    tileTensor.rawShape = newRawShape;

    // ---- static ----
    if (functionType == FunctionType::STATIC) {
        for (auto s : newOriginShape) {
            tileTensor.shape.emplace_back(std::to_string(s));
        }
        tileTensor.stride = BuildStride(newRawShape);
        return;
    }
}

std::vector<std::string> CodeGenOpLiteNPU::GetGmOffsetForTileTensor(unsigned gmIdx) const
{
    int dim = static_cast<int>(rawShape[gmIdx].size());

    if (offsetFromAttr[gmIdx][ID0].IsValid()) {
        return GenSymbolicArgument(offsetFromAttr[gmIdx]);
    }

    return GenGetParamMacroPacked(gmIdx, dim, PREFIX_STR_OFFSET);
}

} // namespace npu::tile_fwk
