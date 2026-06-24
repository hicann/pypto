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
#include "passes/pass_utils/dump_function_utils.h"

#undef MODULE_NAME
#define MODULE_NAME "LoopaxesProc"

namespace npu {
namespace tile_fwk {

namespace {
constexpr size_t MIN_SHAPE_DIM = 2;

void SetOpLoopEnd(Operation* op)
{
    op->SetAttribute(OpAttributeKey::loopGroupEnd, true);
    APASS_LOG_INFO_F(
        Elements::Operation, "Op Code %s, Op[%d] set loopGroup --End--", op->GetOpcodeStr().c_str(), op->GetOpMagic());
}

void SetOpDynLoopEnd(Operation* op)
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
    if (lastOpInLoop != nullptr) {
        SetOpLoopEnd(lastOpInLoop);
        lastOpInLoop = nullptr;
    }
    lastGroupIdx = INVALID_LOOP_GROUPID;
    previousOutputMagic = INVALID_LOOP_GROUPID;
    previousLoopAxes.clear();
    sameStaticLoopOpGroup.clear();
    addrStaticConflictIdx.clear();
    addrStaticRecordMap.clear();

    if (dynLastOpInLoop != nullptr) {
        SetOpDynLoopEnd(dynLastOpInLoop);
        dynLastOpInLoop = nullptr;
    }
    dynLastGroupIdx = INVALID_LOOP_GROUPID;
    dynPreviousOutputMagic = INVALID_LOOP_GROUPID;
    dynPreviousLoopAxes.clear();
    sameDynLoopOpGroup.clear();
    addrDynConflictIdx.clear();
    addrDynRecordMap.clear();
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
        CheckAddrOverLap(false, sameDynLoopOpGroup, addrDynConflictIdx, addrDynRecordMap);
        dynLastGroupIdx = dynGroupIdx++;
        dynPreviousLoopAxes = dynloopAxes;
        op.SetAttribute(OpAttributeKey::dynloopGroupStart, true);
        if (dynLastOpInLoop != nullptr) {
            SetOpDynLoopEnd(dynLastOpInLoop);
        }
        APASS_LOG_INFO_F(
            Elements::Operation, "Op Code %s, Op[%d] set dynloopGroup ++Start++", op.GetOpcodeStr().c_str(),
            op.GetOpMagic());
        sameDynLoopOpGroup.clear();
        addrDynConflictIdx.clear();
        addrDynRecordMap.clear();
    }
    sameDynLoopOpGroup.push_back(&op);
    op.SetAttribute(OpAttributeKey::dynloopGroup, dynGroupIdx);
    op.SetAttribute(OpAttributeKey::dynloopAxes, dynloopAxes);
    dynLastOpInLoop = &op;
    dynPreviousOutputMagic = op.GetOOperands().front()->GetMagic();
}

namespace {
struct Interval {
    int l, r;
};

std::vector<Interval> BuildIntervals(const std::set<std::pair<int, int>>& conflicts)
{
    std::vector<Interval> intervals;
    for (const auto& p : conflicts) {
        int a = p.first, b = p.second;
        if (a == b)
            continue;
        if (a > b)
            std::swap(a, b);
        if (a <= b - 1)
            intervals.push_back({a, b - 1});
    }
    return intervals;
}
} // namespace

std::vector<int> FindCuts(const std::set<std::pair<int, int>>& conflicts, int groupSize)
{
    if (groupSize <= 1)
        return {};
    auto intervals = BuildIntervals(conflicts);
    if (intervals.empty())
        return {};

    std::sort(intervals.begin(), intervals.end(), [](const Interval& a, const Interval& b) {
        return a.r < b.r;
    });

    std::vector<int> cuts;
    for (const auto& inv : intervals) {
        bool covered = false;
        for (int cut : cuts) {
            if (cut >= inv.l && cut <= inv.r) {
                covered = true;
                break;
            }
        }
        if (!covered) {
            int newCut = inv.r;
            cuts.push_back(newCut);
        }
    }

    std::sort(cuts.begin(), cuts.end());
    return cuts;
}

void LoopaxesProc::IsOverLap(std::vector<size_t>& addrRange,
                             std::map<int, std::vector<std::vector<size_t>>> &addrRecordMap,
                             std::set<std::pair<int, int>>& addrConflictIdx, int& idx)
{
    for (auto& entry : addrRecordMap) {
        if (entry.first == idx) {
            continue;
        }
        
        for (auto& existingRange : entry.second) {
            bool hasOverlap = !(addrRange[1] <= existingRange[0] || existingRange[1] <= addrRange[0]);
            if (hasOverlap) {
                std::pair<int, int> conflictPair(entry.first, idx);
                addrConflictIdx.insert(conflictPair);
                break;
            }
        }
    }
}

void LoopaxesProc::RecordAddrOverLap(Operation* op, int& idx, std::set<std::pair<int, int>>& addrConflictIdx,
                                     std::map<int, std::vector<std::vector<size_t>>> &addrRecordMap)
{
    auto inputs = op->GetIOperands();
    auto outputs = op->GetOOperands();
    
    std::vector<std::vector<size_t>> currentRanges;
    
    for (auto& input : inputs) {
        std::vector<size_t> addrRange;
        addrRange.push_back(input->memoryrange.start);
        addrRange.push_back(input->memoryrange.end);
        currentRanges.push_back(addrRange);
    }
    
    for (auto& output : outputs) {
        std::vector<size_t> addrRange;
        addrRange.push_back(output->memoryrange.start);
        addrRange.push_back(output->memoryrange.end);
        currentRanges.push_back(addrRange);
    }
    
    if (idx > 0) {
        for (auto& range : currentRanges) {
            IsOverLap(range, addrRecordMap, addrConflictIdx, idx);
        }
    }
    
    addrRecordMap[idx] = currentRanges;
}

void LoopaxesProc::CheckAddrOverLap(bool isStaticLoop, std::vector<Operation*>& sameLoopOpGroup,
                                    std::set<std::pair<int, int>>& addrConflictIdx,
                                    std::map<int, std::vector<std::vector<size_t>>> &addrRecordMap)
{
    if (sameLoopOpGroup.size() != 1) {
        for (int idx = 0; idx < static_cast<int>(sameLoopOpGroup.size()); idx++) {
            APASS_LOG_INFO_F(Elements::Operation, "RecordAddrOverLap %s[%d].",
                sameLoopOpGroup[idx]->GetOpcodeStr().c_str(), sameLoopOpGroup[idx]->GetOpMagic());
            RecordAddrOverLap(sameLoopOpGroup[idx], idx, addrConflictIdx, addrRecordMap);
        }
    }
    if (addrConflictIdx.empty()) {
        return;
    }
    std::vector<int> cutResult;
    int groupSize = static_cast<int>(sameLoopOpGroup.size());
    cutResult = FindCuts(addrConflictIdx, groupSize);
    if (cutResult.empty()) {
        return;
    }
    if (isStaticLoop) {
        ProcessCutStaticGroup(cutResult, sameLoopOpGroup);
    } else {
        ProcessCutDynGroup(cutResult, sameLoopOpGroup);
    }
}

void LoopaxesProc::ProcessCutStaticGroup(std::vector<int>& cutResult, std::vector<Operation*>& sameLoopOpGroup) {
    for (size_t i = 0; i < cutResult.size(); i++) {
        lastGroupIdx = groupIdx++;
        lastOpInLoop1 = sameLoopOpGroup[cutResult[i]];
        if (lastOpInLoop1 != nullptr) {
            SetOpLoopEnd(lastOpInLoop1);
        }
        sameLoopOpGroup[cutResult[i] + 1]->SetAttribute(OpAttributeKey::loopGroupStart, true);
        APASS_LOG_INFO_F(Elements::Operation, "Op Code %s, Op[%d] set loopGroup ++Start++",
            sameLoopOpGroup[cutResult[i] + 1]->GetOpcodeStr().c_str(),
            sameLoopOpGroup[cutResult[i] + 1]->GetOpMagic());
        if (i != cutResult.size() - 1) {
            for (int opIdx = cutResult[i] + 1; opIdx <= cutResult[i + 1]; opIdx++) {
                sameLoopOpGroup[opIdx]->SetAttribute(OpAttributeKey::loopGroup, groupIdx);
            }
        } else {
            for (int opIdx = cutResult[i] + 1; opIdx < static_cast<int>(sameLoopOpGroup.size()); opIdx++) {
                sameLoopOpGroup[opIdx]->SetAttribute(OpAttributeKey::loopGroup, groupIdx);
            }
        }
    }
}

void LoopaxesProc::ProcessCutDynGroup(std::vector<int>& cutResult, std::vector<Operation*>& sameLoopOpGroup) {
    for (size_t i = 0; i < cutResult.size(); i++) {
        dynLastGroupIdx = dynGroupIdx++;
        lastOpInLoop1 = sameLoopOpGroup[cutResult[i]];
        if (lastOpInLoop1 != nullptr) {
            SetOpDynLoopEnd(lastOpInLoop1);
        }
        sameLoopOpGroup[cutResult[i] + 1]->SetAttribute(OpAttributeKey::dynloopGroupStart, true);
        APASS_LOG_INFO_F(Elements::Operation, "Op Code %s, Op[%d] set loopGroup ++Start++",
            sameLoopOpGroup[cutResult[i] + 1]->GetOpcodeStr().c_str(),
            sameLoopOpGroup[cutResult[i] + 1]->GetOpMagic());
        if (i != cutResult.size() - 1) {
            for (int opIdx = cutResult[i] + 1; opIdx <= cutResult[i + 1]; opIdx++) {
                sameLoopOpGroup[opIdx]->SetAttribute(OpAttributeKey::dynloopGroup, dynGroupIdx);
            }
        } else {
            for (int opIdx = cutResult[i] + 1; opIdx < static_cast<int>(sameLoopOpGroup.size()); opIdx++) {
                sameLoopOpGroup[opIdx]->SetAttribute(OpAttributeKey::dynloopGroup, dynGroupIdx);
            }
        }
    }
}

void LoopaxesProc::ProcessStaticLoopGroup(Operation& op, const std::vector<int64_t>& loopAxes)
{
    if (!SameLoopAxes(loopAxes)) {
        CheckAddrOverLap(true, sameStaticLoopOpGroup, addrStaticConflictIdx, addrStaticRecordMap);
        lastGroupIdx = groupIdx++;
        previousLoopAxes = loopAxes;
        op.SetAttribute(OpAttributeKey::loopGroupStart, true);
        if (lastOpInLoop != nullptr) {
            SetOpLoopEnd(lastOpInLoop);
        }
        APASS_LOG_INFO_F(
            Elements::Operation, "Op Code %s, Op[%d] set loopGroup ++Start++", op.GetOpcodeStr().c_str(),
            op.GetOpMagic());
        sameStaticLoopOpGroup.clear();
        addrStaticConflictIdx.clear();
        addrStaticRecordMap.clear();
    }
    sameStaticLoopOpGroup.push_back(&op);
    op.SetAttribute(OpAttributeKey::loopGroup, groupIdx);
    op.SetAttribute(OpAttributeKey::loopAxes, loopAxes);
    lastOpInLoop = &op;
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
    lastOpInLoop = nullptr;

    dynGroupIdx = INVALID_LOOP_GROUPID;
    dynLastGroupIdx = dynGroupIdx;
    dynLastOpInLoop = nullptr;
}

void LoopaxesProc::FinalizeLoopGroups()
{
    CheckAddrOverLap(true, sameStaticLoopOpGroup, addrStaticConflictIdx, addrStaticRecordMap);
    CheckAddrOverLap(false, sameDynLoopOpGroup, addrDynConflictIdx, addrDynRecordMap);
    
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
    bool allReplacedSymbolsMatch = true;
    bool allExprsMatch = true;

    for (size_t i = 0; i < curLoopAxes.size(); ++i) {
        auto curExpr = SymbolicExpressionTable::BuildExpression(curLoopAxes[i]);
        auto prevExpr = SymbolicExpressionTable::BuildExpression(dynPreviousLoopAxes[i]);

        if (dynParamTable.find(curExpr) != dynParamTable.end() && dynParamTable.find(prevExpr) != dynParamTable.end()) {
            auto curParamInfo = dynParamTable[curExpr];
            auto preParamInfo = dynParamTable[prevExpr];
            if (curParamInfo.replacedSymbol.empty() || preParamInfo.replacedSymbol.empty() ||
                curParamInfo.replacedSymbol != preParamInfo.replacedSymbol) {
                allReplacedSymbolsMatch = false;
            }
        } else {
            allReplacedSymbolsMatch = false;
        }

        if (curExpr != prevExpr) {
            allExprsMatch = false;
        }
    }

    return allReplacedSymbolsMatch || allExprsMatch;
}

Status LoopaxesProc::DumpFunctionJson(Function& function, const std::string& logFolder, bool beforeFunction)
{
    DumpFunctionUtils utils;
    return utils.DumpTileFunctionsJson(
        function, logFolder, beforeFunction,
        [this](Function& f, const std::string& folder, bool before) {
            return Pass::DumpFunctionJson(f, folder, before);
        });
}

Status LoopaxesProc::PrintFunction(Function& function, const std::string& logFolder, bool beforeFunction)
{
    DumpFunctionUtils dfUtils;
    return dfUtils.PrintTileFunctions(
        function, logFolder, beforeFunction,
        [this](Function& f, const std::string& folder, bool before) {
            return Pass::PrintFunction(f, folder, before);
        });
}

} // namespace tile_fwk
} // namespace npu