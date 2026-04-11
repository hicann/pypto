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
 * \file codegen_for_block.cpp
 * \brief
 */

#include "codegen_for_block.h"

#include "interface/tensor/symbolic_scalar.h"
#include "codegen/utils/codegen_utils.h"

namespace npu::tile_fwk {
const std::string DEFAULT_TENSOR_OFFSET = "tileOffsets";

std::string ForNode::Print() const
{
    std::ostringstream os;
    os << "for (";
    PrintInit(os);
    PrintCond(os);
    PrintUpdate(os);
    os << ") {\n";
    return os.str();
}

void ForNode::PrintInit(std::ostringstream& os) const
{
    os << "uint16_t " << loopVar << " = " << SymbolicExpressionTable::BuildExpression(start) << SEMICOLON_BLANK;
}

void ForNode::PrintCond(std::ostringstream& os) const
{
    os << loopVar << " < " << SymbolicExpressionTable::BuildExpression(extent) << SEMICOLON_BLANK;
}

void ForNode::PrintUpdate(std::ostringstream& os) const
{
    if (step.ConcreteValid() && step.Concrete() == 1) {
        os << "++" << loopVar;
    } else {
        os << loopVar << " += " << SymbolicExpressionTable::BuildExpression(step);
    }
}

void ForBlockManager::UpdateAxesList(const std::vector<SymbolicScalar>& axesList)
{
    axesList_ = axesList;
    FillVecWithDummyInHead<SymbolicScalar>(axesList_, MAX_LOOP_DEPTH - axesList.size(), 1);
    CODEGEN_LOGI("axesList_ after fill is : %s, ", IntVecToStr(axesList_).c_str());
    std::vector<std::string> offsetInLoop(axesList_.size(), "");
    for (size_t i = 0; i < axesList_.size(); ++i) {
        std::string loopVar = "idx" + std::to_string(i);
        ForNode forNode{loopVar, 0, axesList_[i], 1};
        forNodes_.push_back(forNode);
        offsetInLoop[i] = axesList_[i].ConcreteValid() && axesList_[i].Concrete() == 1 ? "0" : loopVar;
    }

    allOffsetsInLoop_[offsetInLoop] = DEFAULT_TENSOR_OFFSET;
    defaultOffset_ = offsetInLoop;
    ++offsetCnt_;
}

std::string ForBlockManager::Print() const
{
    std::ostringstream os;
    PrintForHeader(os);
    PrintForBody(os);
    PrintForEnd(os);
    return os.str();
}

void ForBlockManager::PrintForHeader(std::ostringstream& os) const
{
    for (size_t i = 0; i < MAX_LOOP_DEPTH; ++i) {
        PrintIndent(os, i);
        os << forNodes_[i].Print();
    }
}

void ForBlockManager::PrintForBody(std::ostringstream& os) const
{
    PrintOffsetDef(os);
    PrintSetAddrs(os);
    PrintTileOps(os);
}

void ForBlockManager::PrintForEnd(std::ostringstream& os) const
{
    for (size_t i = 0; i < MAX_LOOP_DEPTH; ++i) {
        PrintIndent(os, MAX_LOOP_DEPTH - i - 1);
        os << "}\n";
    }
}

void ForBlockManager::PrintOffsetDef(std::ostringstream& os) const
{
    for (const auto& [offsetInLoop, offsetName] : allOffsetsInLoop_) {
        PrintIndent(os, MAX_LOOP_DEPTH + 1);
        os << "auto " << offsetName << " = "
           << "TileOffset" << WrapParamByParentheses(offsetInLoop) << STMT_END;
    }
}

void ForBlockManager::PrintSetAddrs(std::ostringstream& os) const
{
    for (const auto& [tensor, offset] : tensorOffset_) {
        PrintIndent(os, MAX_LOOP_DEPTH + 1);
        PrintSetAddrSingle(os, tensor, offset);
    }
}

void ForBlockManager::PrintSetAddrSingle(
    std::ostringstream& os, const std::string& tensor, const std::string& offset) const
{
    std::string fullDimTensor;
    fullDimTensor = sm_->QueryTileTensorFullDimByTensorInLoop(tensor);
    os << tensor << ".SetAddr(" << fullDimTensor << ".GetLinearAddr(" << offset << "));\n";
}

void ForBlockManager::PrintTileOps(std::ostringstream& os) const
{
    for (const auto& tileOp : opList_) {
        CODEGEN_LOGI("tileOp is : %s", tileOp.c_str());
        PrintIndent(os, MAX_LOOP_DEPTH + 1);
        os << tileOp;
    }
}

/*
Different tensors use different offsets, generate code like:
for idx0 in axis0 {
  for idx1 in axis1 {
    for idx2 in axis2 {
        auto tileOffsets = TileOffset(idx0, idx1, idx2);
        auto src0tileOffsets = TileOffset(0, 0, idx2);
        auto src1tileOffsets = TileOffset(0, idx1, 0);
        ubTensor_2_low2DimInLoop.SetAddr(ubTensor_2.GetLinearAddr(src1tileOffsets));
        ubTensor_0_low2DimInLoop.SetAddr(ubTensor_0.GetLinearAddr(src0tileOffsets));
        ubTensor_4_low2DimInLoop.SetAddr(ubTensor_4.GetLinearAddr(tileOffsets));
        TAdd<LastUse3Dim<0, 1, 1>>(ubTensor_4_low2DimInLoop, ubTensor_0_low2DimInLoop, ubTensor_2_low2DimInLoop);
    }
  }
}
 */
void ForBlockManager::UpdateTensorOffsetInLoop(
    Opcode opCode, int tensorMagic, int opMagic, const std::string& tensorNameInLoop)
{
    if (SUPPORT_BRC_INLINE.find(opCode) == SUPPORT_BRC_INLINE.end()) {
        tensorOffset_[tensorNameInLoop] = DEFAULT_TENSOR_OFFSET;
        return;
    }

    auto tileTensor = sm_->QueryTileTensorByMagic(tensorMagic, opMagic);
    ASSERT(GenCodeErr::TENSOR_NOT_FOUND, tileTensor != nullptr)
        << "QueryTileTensorByMagic tensor magic is " << tensorMagic << ", op magic is " << opMagic;

    auto rawShape = tileTensor->rawShape;
    FillVecWithDummyInHead<int64_t>(rawShape, SHAPE_DIM5 - rawShape.size(), 1);
    auto newOffset = defaultOffset_;
    for (size_t i = 0; i < MAX_LOOP_DEPTH; ++i) {
        if (rawShape[i] == 1) {
            newOffset[i] = "0";
        }
    }

    std::string offsetName;
    auto iter = allOffsetsInLoop_.find(newOffset);
    if (iter != allOffsetsInLoop_.end()) {
        offsetName = iter->second;
    } else {
        offsetName = DEFAULT_TENSOR_OFFSET + "_" + std::to_string(offsetCnt_);
        allOffsetsInLoop_[newOffset] = offsetName;
        ++offsetCnt_;
    }
    tensorOffset_[tensorNameInLoop] = offsetName;
}

void ForBlockManager::AddTensorInLoopBody(
    const std::string& tensorFullDim, const TileTensor& tileTensor, int opMagic, Opcode opCode)
{
    CODEGEN_LOGI(
        "AddTensorInLoopBody : tensorFullDim: %s, loop tensor: %s,op magic: %d, op code: %s", tensorFullDim.c_str(),
        tileTensor.tensorName.c_str(), opMagic, OpcodeManager::Inst().GetOpcodeStr(opCode).c_str());
    std::string tensorNameInLoop = sm_->AddTileTensor(opMagic, tileTensor);
    sm_->InsertTensorNameInLoopToFullDim(tensorNameInLoop, tensorFullDim);
    UpdateTensorOffsetInLoop(opCode, tileTensor.magic, opMagic, tensorNameInLoop);
}

} // namespace npu::tile_fwk
