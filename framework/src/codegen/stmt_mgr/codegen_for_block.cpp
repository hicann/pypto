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
std::string ForNode::Print() const {
    std::ostringstream os;
    os << "for (";
    PrintInit(os);
    PrintCond(os);
    PrintUpdate(os);
    os << ") {\n";
    return os.str();
}

void ForNode::PrintInit(std::ostringstream &os) const {
    os << "uint16_t " << loopVar << " = " << SymbolicExpressionTable::BuildExpression(start) << SEMICOLON_BLANK;
}

void ForNode::PrintCond(std::ostringstream &os) const {
    os << loopVar << " < " << SymbolicExpressionTable::BuildExpression(extent) << SEMICOLON_BLANK;
}

void ForNode::PrintUpdate(std::ostringstream &os) const {
    if (step.ConcreteValid() && step.Concrete() == 1) {
        os << "++" << loopVar;
    } else {
        os << loopVar << " += " << SymbolicExpressionTable::BuildExpression(step);
    }
}

void ForBlockManager::UpdateAxesList(const std::vector<SymbolicScalar> &axesList) {
    axesList_ = axesList;
    FillIntVecWithDummyInHead<SymbolicScalar>(axesList_, MAX_LOOP_DEPTH - axesList.size(), 1);
    ALOG_INFO_F("axesList_ after fill is : %s, ", IntVecToStr(axesList_).c_str());
    for (size_t i = 0; i < axesList_.size(); ++i) {
        std::string loopVar = "idx" + std::to_string(i);
        ForNode forNode{loopVar, 0, axesList_[i], 1};
        forNodes_.push_back(forNode);
    }
}

std::string ForBlockManager::Print() const {
    std::ostringstream os;
    PrintForHeader(os);
    PrintForBody(os);
    PrintForEnd(os);
    return os.str();
}

void ForBlockManager::PrintForHeader(std::ostringstream &os) const {
    for (size_t i = 0; i < MAX_LOOP_DEPTH; ++i) {
        PrintIndent(os, i);
        os << forNodes_[i].Print();
    }
}

void ForBlockManager::PrintForBody(std::ostringstream &os) const {
    PrintIndent(os, MAX_LOOP_DEPTH + 1);
    PrintOffsetDef(os);
    PrintSetAddrs(os);
    PrintTileOps(os);
}

void ForBlockManager::PrintForEnd(std::ostringstream &os) const {
    for (size_t i = 0; i < MAX_LOOP_DEPTH; ++i) {
        PrintIndent(os, MAX_LOOP_DEPTH - i - 1);
        os << "}\n";
    }
}

void ForBlockManager::PrintOffsetDef(std::ostringstream &os) const {
    os << "auto tileOffsets = TileOffset";
    std::vector<std::string> loopVars;
    for (const auto &forNode : forNodes_) {
        loopVars.emplace_back(forNode.loopVar);
    }
    os << WrapParamByParentheses(loopVars) << STMT_END;
}

void ForBlockManager::PrintSetAddrs(std::ostringstream &os) const {
    for (const auto &tensor : tensorNeedSetAddr_) {
        PrintIndent(os, MAX_LOOP_DEPTH + 1);
        PrintSetAddrSingle(os, tensor);
    }
}

void ForBlockManager::PrintSetAddrSingle(std::ostringstream &os, const std::string &tensor) const {
    std::string fullDimTensor;
    fullDimTensor = sm_->QueryTileTensorFullDimByTensorInLoop(tensor);
    os << tensor << ".SetAddr(" << fullDimTensor << ".GetLinearAddr(tileOffsets));\n";
}

void ForBlockManager::PrintTileOps(std::ostringstream &os) const {
    for (const auto &tileOp : opList_) {
        ALOG_INFO_F("tileOp is : %s", tileOp.c_str());
        PrintIndent(os, MAX_LOOP_DEPTH + 1);
        os << tileOp;
    }
}

} // namespace npu::tile_fwk