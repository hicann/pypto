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
 * \file loopaxes_proc.cpp
 * \brief
 */

#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/utils/common.h"
#include "interface/function/function.h"
#include "interface/tensor/logical_tensor.h"
#include "passes/pass_log/pass_log.h"
#include "passes/pass_interface/pass.h"
#include "loopaxes_proc.h"

#define MODULE_NAME "LoopaxesProc"

namespace npu {
namespace tile_fwk {
Status LoopaxesProc::RunOnFunction(Function &function) {
    bool enableVF = config::GetPassGlobalConfig(KEY_ENABLE_VF, false);
    bool useMarkFor = enableVF || config::GetPassGlobalConfig(KEY_VF_OPT_MARK_FOR, false);
    if (!useMarkFor) {
        return SUCCESS;
    }

    APASS_LOG_INFO_F(
        Elements::Operation, "===============================================================> Start LoopaxesProc.");
    UpdateFuncLoopAxes(function);
    APASS_LOG_INFO_F(
        Elements::Operation, "===============================================================> Finish LoopaxesProc.");
    return SUCCESS;
}

void SetOpLoopEnd(std::shared_ptr<Operation> op) {
    op->SetAttribute(OpAttributeKey::loopGroupEnd, true);
    APASS_LOG_INFO_F(
        Elements::Operation, "Op Code %s, Op[%d] set loopGroup --End--", op->GetOpcodeStr().c_str(), op->GetOpMagic());
}

void LoopaxesProc::ClearStatus() {
    lastGroupIdx = INVALID_LOOP_GROUPID;
    previousOutputMagic = INVALID_LOOP_GROUPID;
    previousLoopAxes.clear();
    if (lastOpInLoop != nullptr) {
        SetOpLoopEnd(lastOpInLoop);
        lastOpInLoop.reset();
    }
}

bool NeedClearStatus(const Operation &op) {
    auto opCode = op.GetOpcode();
    auto iter = SUPPORT_VF_FUSE_OPS.find(opCode);
    if (iter == SUPPORT_VF_FUSE_OPS.end()) {
        return true;
    }

    //  Opcode::OP_EXPAND only support last axis or second last axis in for-loop
    if (opCode == Opcode::OP_EXPAND) {
        std::string axisKey = OP_ATTR_PREFIX + "EXPANDDIM";
        ASSERT(op.HasAttr(axisKey)) << "attr " << axisKey << "not found";
        int64_t expandAxis = op.GetIntAttribute(axisKey);
        int shapeSize = static_cast<int>(op.GetOOperands().front()->GetDynValidShape().size());
        expandAxis += SHAPE_DIM4 - shapeSize;
        return expandAxis == 0 || expandAxis == 1;
    }

    return false;
}

Status LoopaxesProc::UpdateOpLoopAxes(Operation &op) {
    if (SKIP_OPCODE_FOR_CODEGEN.find(op.GetOpcode()) != SKIP_OPCODE_FOR_CODEGEN.end()) {
        APASS_LOG_DEBUG_F(
            Elements::Operation, "Op Code %s, Op[%d] ignore this op", op.GetOpcodeStr().c_str(), op.GetOpMagic());
        return SUCCESS;
    }

    if (NeedClearStatus(op)) {
        ClearStatus();
        return SUCCESS;
    }

    std::vector<SymbolicScalar> loopAxes;
    auto input = op.GetIOperands().front();
    auto output = op.GetOOperands().front();
    auto shape = output->GetDynValidShape();
    if (shape.size() <= NUM2) {
        // 被纳入group的要求维度大于2，否则将其设置为-1
        op.SetAttribute(OpAttributeKey::loopGroup, INVALID_LOOP_GROUPID);
        ClearStatus();
    } else {
        if (op.HasAttr(OpAttributeKey::loopAxes)) {
            loopAxes = op.GetVectorSymbolicScalarAttribute(OpAttributeKey::loopAxes);
        } else {
            for (size_t i = 0UL; i < shape.size() - 2UL; ++i) {
                loopAxes.push_back(shape[i]);
            }
        }
        // 当前节点的loopaxes和group的loopaxes一致，当前节点划入当前的loopaxes
        // 当前节点的loopaxes和group的loopaxes不一致，划入一个新的group起点，进行group
        if (!SameLoopAxes(loopAxes) && previousOutputMagic != input->GetMagic()) {
            lastGroupIdx = groupIdx++;
            previousLoopAxes = loopAxes;
            op.SetAttribute(OpAttributeKey::loopGroupStart, true);
            if (lastOpInLoop != nullptr) {
                SetOpLoopEnd(lastOpInLoop);
            }
            APASS_LOG_INFO_F(Elements::Operation, "Op Code %s, Op[%d] set loopGroup ++Start++",
                op.GetOpcodeStr().c_str(), op.GetOpMagic());
        }
        op.SetAttribute(OpAttributeKey::loopGroup, groupIdx);
        op.SetAttribute(OpAttributeKey::loopAxes, loopAxes);
        lastOpInLoop = op.shared_from_this();
        previousOutputMagic = output->GetMagic();
        APASS_LOG_INFO_F(Elements::Operation, "Op Code %s, Op[%d] groupIdx is %d, loopAxes is %s",
            op.GetOpcodeStr().c_str(), op.GetOpMagic(), groupIdx, IntVecToStr(loopAxes).c_str());
    }
    return SUCCESS;
}

Status LoopaxesProc::UpdateFuncLoopAxes(Function &function) {
    if (function.rootFunc_ == nullptr) {
        return SUCCESS;
    }
    APASS_LOG_DEBUG_F(Elements::Operation, "Function[%s] has rootFunc.", function.GetMagicName().c_str());
    for (auto &subProgram : function.rootFunc_->programs_) {
        groupIdx = INVALID_LOOP_GROUPID;
        lastGroupIdx = groupIdx;
        lastOpInLoop.reset();
        if (subProgram.second == nullptr) {
            APASS_LOG_DEBUG_F(Elements::Operation, "subProgram[%d] of Function[%s] is nullptr.", subProgram.first,
                function.GetMagicName().c_str());
            continue;
        }
        for (auto &op : subProgram.second->Operations(false)) {
            UpdateOpLoopAxes(op);
        }
        if (lastGroupIdx != INVALID_LOOP_GROUPID && lastOpInLoop != nullptr) {
            SetOpLoopEnd(lastOpInLoop);
        }
    }
    return SUCCESS;
}

bool LoopaxesProc::SameLoopAxes(const std::vector<SymbolicScalar> &curLoopAxes) {
    if (curLoopAxes.size() != previousLoopAxes.size()) {
        return false;
    }
    for (size_t i = 0; i < curLoopAxes.size(); ++i) {
        if (SymbolicExpressionTable::BuildExpression(curLoopAxes[i]) !=
            SymbolicExpressionTable::BuildExpression(previousLoopAxes[i])) {
            return false;
        }
    }
    return true;
}
} // namespace tile_fwk
} // namespace npu
