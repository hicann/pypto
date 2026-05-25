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
#include "passes/block_graph_pass/dyn_attr_to_static.h"
#include "tilefwk/error_code.h"

#undef MODULE_NAME
#define MODULE_NAME "LoopaxesProc"

namespace npu {
namespace tile_fwk {

namespace {
constexpr size_t MIN_SHAPE_DIM = 2;

void SetOpLoopEnd(std::shared_ptr<Operation> op)
{
    op->SetAttribute(OpAttributeKey::loopGroupEnd, true);
    APASS_LOG_INFO_F(
        Elements::Operation, "Op Code %s, Op[%d] set loopGroup --End--", op->GetOpcodeStr().c_str(), op->GetOpMagic());
}

void SetOpDynLoopEnd(std::shared_ptr<Operation> op)
{
    op->SetAttribute(OpAttributeKey::dynloopGroupEnd, true);
    APASS_LOG_INFO_F(
        Elements::Operation, "Op Code %s, Op[%d] set dynloopGroup --End--", op->GetOpcodeStr().c_str(),
        op->GetOpMagic());
}

bool NeedClearStatus(const Operation& op)
{
    auto opCode = op.GetOpcode();
    if (SUPPORT_VF_FUSE_OPS.find(opCode) == SUPPORT_VF_FUSE_OPS.end()) {
        APASS_LOG_DEBUG_F(
            Elements::Operation, "%d %s doesn't support VF fuse", op.GetOpMagic(), op.GetOpcodeStr().c_str());
        return true;
    }
    return false;
}

void GetOpLoopAxes(const Operation& op, std::vector<int64_t>& loopAxes, std::vector<SymbolicScalar>& dynloopAxes)
{
    auto output = op.GetOOperands().front();
    auto shape = output->GetShape();
    auto dynShape = output->GetDynValidShape();

    if (op.HasAttr(OpAttributeKey::dynloopAxes)) {
        dynloopAxes = op.GetVectorSymbolicScalarAttribute(OpAttributeKey::dynloopAxes);
    } else {
        for (size_t i = 0; i < dynShape.size() - MIN_SHAPE_DIM; ++i) {
            dynloopAxes.push_back(dynShape[i]);
        }
    }

    if (op.HasAttr(OpAttributeKey::loopAxes)) {
        loopAxes = op.GetVectorIntAttribute(OpAttributeKey::loopAxes);
    } else {
        for (size_t i = 0; i < shape.size() - MIN_SHAPE_DIM; ++i) {
            loopAxes.push_back(shape[i]);
        }
    }
}

void HandleSmallShapeOp(Operation& op)
{
    op.SetAttribute(OpAttributeKey::dynloopGroup, INVALID_LOOP_GROUPID);
    op.SetAttribute(OpAttributeKey::loopGroup, INVALID_LOOP_GROUPID);
}
} // namespace

Status LoopaxesProc::RunOnFunction(Function& function)
{
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

void LoopaxesProc::ClearStatus()
{
    lastGroupIdx = INVALID_LOOP_GROUPID;
    previousOutputMagic = INVALID_LOOP_GROUPID;
    previousLoopAxes.clear();
    if (lastOpInLoop != nullptr) {
        SetOpLoopEnd(lastOpInLoop);
        lastOpInLoop.reset();
    }

    dynLastGroupIdx = INVALID_LOOP_GROUPID;
    dynPreviousOutputMagic = INVALID_LOOP_GROUPID;
    dynPreviousLoopAxes.clear();
    if (dynLastOpInLoop != nullptr) {
        SetOpDynLoopEnd(dynLastOpInLoop);
        dynLastOpInLoop.reset();
    }
}

Status LoopaxesProc::UpdateOpLoopAxes(Operation& op, Function& subFunc)
{
    if (SKIP_OPCODE_FOR_CODEGEN.find(op.GetOpcode()) != SKIP_OPCODE_FOR_CODEGEN.end()) {
        APASS_LOG_DEBUG_F(
            Elements::Operation, "Op Code %s, Op[%d] ignore this op", op.GetOpcodeStr().c_str(), op.GetOpMagic());
        return SUCCESS;
    }

    if (NeedClearStatus(op)) {
        ClearStatus();
        return SUCCESS;
    }

    auto output = op.GetOOperands().front();
    auto shape = output->GetShape();
    auto dynShape = output->GetDynValidShape();

    if (shape.size() != dynShape.size()) {
        APASS_LOG_ERROR_F(
            Elements::Operation, "Op Code %s, Op[%d] output dynShape size != shape size.", op.GetOpcodeStr().c_str(),
            op.GetOpMagic());
        return FAILED;
    }

    if (dynShape.size() <= MIN_SHAPE_DIM) {
        HandleSmallShapeOp(op);
        ClearStatus();
        return SUCCESS;
    }

    std::vector<int64_t> loopAxes;
    std::vector<SymbolicScalar> dynloopAxes;
    GetOpLoopAxes(op, loopAxes, dynloopAxes);

    ProcessDynLoopGroup(op, dynloopAxes, subFunc);
    ProcessStaticLoopGroup(op, loopAxes);

    APASS_LOG_INFO_F(
        Elements::Operation, "Op Code %s, Op[%d] groupIdx=%ld, loopAxes=%s, dynGroupIdx=%ld, dynLoopAxes=%s",
        op.GetOpcodeStr().c_str(), op.GetOpMagic(), groupIdx, IntVecToStr(loopAxes).c_str(), dynGroupIdx,
        IntVecToStr(dynloopAxes).c_str());

    return SUCCESS;
}

void LoopaxesProc::ProcessDynLoopGroup(
    Operation& op, const std::vector<SymbolicScalar>& dynloopAxes, const Function& subFunc)
{
    if (!SameDynLoopAxes(dynloopAxes, subFunc)) {
        dynLastGroupIdx = dynGroupIdx++;
        dynPreviousLoopAxes = dynloopAxes;
        op.SetAttribute(OpAttributeKey::dynloopGroupStart, true);
        if (dynLastOpInLoop != nullptr) {
            SetOpDynLoopEnd(dynLastOpInLoop);
        }
        APASS_LOG_INFO_F(
            Elements::Operation, "Op Code %s, Op[%d] set dynloopGroup ++Start++", op.GetOpcodeStr().c_str(),
            op.GetOpMagic());
    }

    op.SetAttribute(OpAttributeKey::dynloopGroup, dynGroupIdx);
    op.SetAttribute(OpAttributeKey::dynloopAxes, dynloopAxes);
    dynLastOpInLoop = op.shared_from_this();
    dynPreviousOutputMagic = op.GetOOperands().front()->GetMagic();
}

void LoopaxesProc::ProcessStaticLoopGroup(Operation& op, const std::vector<int64_t>& loopAxes)
{
    if (!SameLoopAxes(loopAxes)) {
        lastGroupIdx = groupIdx++;
        previousLoopAxes = loopAxes;
        op.SetAttribute(OpAttributeKey::loopGroupStart, true);
        if (lastOpInLoop != nullptr) {
            SetOpLoopEnd(lastOpInLoop);
        }
        APASS_LOG_INFO_F(
            Elements::Operation, "Op Code %s, Op[%d] set loopGroup ++Start++", op.GetOpcodeStr().c_str(),
            op.GetOpMagic());
    }

    op.SetAttribute(OpAttributeKey::loopGroup, groupIdx);
    op.SetAttribute(OpAttributeKey::loopAxes, loopAxes);
    lastOpInLoop = op.shared_from_this();
    previousOutputMagic = op.GetOOperands().front()->GetMagic();
}

Status LoopaxesProc::UpdateFuncLoopAxes(Function& function)
{
    DynAttrToStatic dyn2Static;
    if (dyn2Static.BuildLeafToCaller(&function) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "Failed to call BuildLeafToCaller.");
        return FAILED;
    }

    for (auto& pair : dyn2Static.leaf2Caller) {
        if (pair.first == nullptr) {
            APASS_LOG_DEBUG_F(Elements::Operation, "subProgram of Function is nullptr.");
            continue;
        }

        ResetGroupState();

        for (auto& op : pair.first->Operations(false)) {
            UpdateOpLoopAxes(op, *pair.first);
        }

        FinalizeLoopGroups();
    }
    return SUCCESS;
}

void LoopaxesProc::ResetGroupState()
{
    groupIdx = INVALID_LOOP_GROUPID;
    lastGroupIdx = groupIdx;
    lastOpInLoop.reset();

    dynGroupIdx = INVALID_LOOP_GROUPID;
    dynLastGroupIdx = dynGroupIdx;
    dynLastOpInLoop.reset();
}

void LoopaxesProc::FinalizeLoopGroups()
{
    if (lastGroupIdx != INVALID_LOOP_GROUPID && lastOpInLoop != nullptr) {
        SetOpLoopEnd(lastOpInLoop);
    }
    if (dynLastGroupIdx != INVALID_LOOP_GROUPID && dynLastOpInLoop != nullptr) {
        SetOpDynLoopEnd(dynLastOpInLoop);
    }
}

bool LoopaxesProc::SameLoopAxes(const std::vector<int64_t>& curLoopAxes)
{
    if (curLoopAxes.size() != previousLoopAxes.size()) {
        return false;
    }
    for (size_t i = 0; i < curLoopAxes.size(); i++) {
        if (curLoopAxes[i] != previousLoopAxes[i]) {
            return false;
        }
    }
    return true;
}

bool LoopaxesProc::SameDynLoopAxes(const std::vector<SymbolicScalar>& curLoopAxes, const Function& subFunc)
{
    if (curLoopAxes.size() != dynPreviousLoopAxes.size()) {
        return false;
    }

    auto dynParamTable = subFunc.GetDynParamTable();
    for (size_t i = 0; i < curLoopAxes.size(); ++i) {
        auto curExpr = SymbolicExpressionTable::BuildExpression(curLoopAxes[i]);
        auto prevExpr = SymbolicExpressionTable::BuildExpression(dynPreviousLoopAxes[i]);

        if (dynParamTable.find(curExpr) != dynParamTable.end() && dynParamTable.find(prevExpr) != dynParamTable.end()) {
            auto curParamInfo = dynParamTable[curExpr];
            auto preParamInfo = dynParamTable[prevExpr];
            if (!curParamInfo.replacedSymbol.empty() && !preParamInfo.replacedSymbol.empty() &&
                curParamInfo.replacedSymbol == preParamInfo.replacedSymbol) {
                APASS_LOG_INFO_F(
                    Elements::Operation, "%s & %s has same replacedSymbol.", curExpr.c_str(), prevExpr.c_str());
                return true;
            }
        }

        if (curExpr != prevExpr) {
            return false;
        }
    }
    return true;
}

} // namespace tile_fwk
} // namespace npu