/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "ir/transforms/merge_stmts_pass.h"

#include <algorithm>
#include <unordered_set>

#include "ir/expr.h"
#include "ir/kind_traits.h"

#include "interface/tensor/ir_tensor_op_rebuild.h"
#include "interface/tensor/logical_tensor.h"

namespace pypto::ir {

namespace {

using npu::tile_fwk::LogicalTensor;

struct IfStmtResult {
    IfStmtPtr ifStmt;
    VarExprMap outputCloneMap;
};

struct MergeSegmentResult {
    std::vector<StmtPtr> stmts;
    VarExprMap outputCloneMap;
};

bool IsExternalVar(const std::string& name, const std::vector<std::string>& externalVarNames)
{
    return std::find(externalVarNames.begin(), externalVarNames.end(), name) != externalVarNames.end();
}

VarPtr LookupVarDef(VarPtr var, const VarExprMap& varMap)
{
    if (!var) {
        return var;
    }
    auto it = varMap.find(var);
    if (it != varMap.end()) {
        return std::static_pointer_cast<const Var>(it->second);
    }
    return var;
}

ExprPtr LookupVarInExpr(ExprPtr expr, const VarExprMap& varMap)
{
    if (!expr || varMap.empty()) {
        return expr;
    }
    auto var = std::dynamic_pointer_cast<const Var>(expr);
    auto it = varMap.find(var);
    if (it != varMap.end()) {
        return it->second;
    }
    return expr;
}

VarExprMap BuildYieldVarMap(IfStmtPtr ifStmt, SeqStmtsPtr body)
{
    VarExprMap varMap;
    if (!body || body->stmts_.empty() || ifStmt->returnVars_.empty()) {
        return varMap;
    }
    auto& lastStmt = body->stmts_.back();
    if (!IsA<YieldStmt>(lastStmt)) {
        return varMap;
    }
    auto yieldStmt = As<YieldStmt>(lastStmt);
    for (size_t i = 0; i < ifStmt->returnVars_.size() && i < yieldStmt->value_.size(); ++i) {
        varMap[ifStmt->returnVars_[i]] = yieldStmt->value_[i];
    }
    return varMap;
}

VarExprMap BuildYieldVarMapForForStmt(ForStmtPtr forStmt)
{
    VarExprMap varMap;
    if (!forStmt->body_ || forStmt->body_->stmts_.empty() || forStmt->returnVars_.empty()) {
        return varMap;
    }
    auto& lastStmt = forStmt->body_->stmts_.back();
    std::vector<ExprPtr> yieldValues;
    if (auto c = As<ContinueStmt>(lastStmt)) {
        yieldValues = c->value_;
    } else if (auto y = As<YieldStmt>(lastStmt)) {
        yieldValues = y->value_;
    } else {
        return varMap;
    }
    for (size_t i = 0; i < forStmt->returnVars_.size() && i < yieldValues.size(); ++i) {
        varMap[forStmt->returnVars_[i]] = yieldValues[i];
    }
    return varMap;
}

StmtPtr SubstituteReturnVarUses(StmtPtr stmt, const VarExprMap& varMap);
StmtPtr SubstituteVars(StmtPtr stmt, const VarExprMap& varMap);
void CollectDefinedVars(StmtPtr stmt, std::unordered_set<VarPtr>& defs);
SeqStmtsPtr MergeStmtsIntoIfStmtImpl(SeqStmtsPtr seq, const std::vector<std::string>& externalVarNames);

StmtPtr SubstituteReturnVarUses(TensorOpStmtPtr tensorOp, const VarExprMap& varMap)
{
    std::vector<ExprPtr> newArgs;
    for (auto& arg : tensorOp->args_) {
        newArgs.push_back(LookupVarInExpr(arg, varMap));
    }
    return npu::tile_fwk::RebuildTensorOpStmt(
        tensorOp, tensorOp->result_, tensorOp->result_token_, newArgs, tensorOp->tokens_, tensorOp->span_);
}

StmtPtr SubstituteReturnVarUses(IfStmtPtr ifStmt, const VarExprMap& varMap)
{
    std::vector<StmtPtr> newThenStmts;
    for (auto& s : ifStmt->thenBody_->stmts_) {
        newThenStmts.push_back(SubstituteReturnVarUses(s, varMap));
    }
    auto newThenBody = std::make_shared<SeqStmts>(newThenStmts, ifStmt->span_);
    std::optional<SeqStmtsPtr> newElseBody;
    if (ifStmt->elseBody_) {
        std::vector<StmtPtr> newElseStmts;
        for (auto& s : ifStmt->elseBody_.value()->stmts_) {
            newElseStmts.push_back(SubstituteReturnVarUses(s, varMap));
        }
        newElseBody = std::make_shared<SeqStmts>(newElseStmts, ifStmt->span_);
    }
    return std::make_shared<IfStmt>(ifStmt->condition_, newThenBody, newElseBody, ifStmt->returnVars_, ifStmt->span_);
}

StmtPtr SubstituteReturnVarUses(ForStmtPtr f, const VarExprMap& varMap)
{
    std::vector<StmtPtr> newBodyStmts;
    for (auto& s : f->body_->stmts_) {
        newBodyStmts.push_back(SubstituteReturnVarUses(s, varMap));
    }
    auto newBody = std::make_shared<SeqStmts>(newBodyStmts, f->span_);
    std::vector<IterArgPtr> newIterArgs;
    for (auto& arg : f->iterArgs_) {
        newIterArgs.push_back(std::make_shared<IterArg>(arg->iterVar_, LookupVarInExpr(arg->initValue_, varMap)));
    }
    ExprPtr newStart = LookupVarInExpr(f->start_, varMap);
    ExprPtr newStop = LookupVarInExpr(f->stop_, varMap);
    ExprPtr newStep = LookupVarInExpr(f->step_, varMap);
    return std::make_shared<ForStmt>(
        f->loopVar_, newStart, newStop, newStep, newIterArgs, newBody, f->returnVars_, f->span_, f->attrs_);
}

StmtPtr SubstituteReturnVarUses(StmtPtr stmt, const VarExprMap& varMap)
{
    if (!stmt || varMap.empty()) {
        return stmt;
    }
    if (auto t = As<TensorOpStmt>(stmt)) {
        return SubstituteReturnVarUses(t, varMap);
    }
    if (auto i = As<IfStmt>(stmt)) {
        return SubstituteReturnVarUses(i, varMap);
    }
    if (auto f = As<ForStmt>(stmt)) {
        return SubstituteReturnVarUses(f, varMap);
    }
    return stmt;
}

StmtPtr SubstituteVars(TensorOpStmtPtr t, const VarExprMap& varMap)
{
    std::vector<VarPtr> newResult;
    for (auto& v : t->result_) {
        newResult.push_back(LookupVarDef(v, varMap));
    }
    std::vector<ExprPtr> newArgs;
    for (auto& a : t->args_) {
        newArgs.push_back(LookupVarInExpr(a, varMap));
    }
    return npu::tile_fwk::RebuildTensorOpStmt(
        t, newResult, LookupVarDef(t->result_token_, varMap), newArgs, t->tokens_, t->span_);
}

StmtPtr SubstituteVars(IfStmtPtr ifStmt, const VarExprMap& varMap)
{
    std::vector<StmtPtr> newThenStmts;
    for (auto& s : ifStmt->thenBody_->stmts_) {
        newThenStmts.push_back(SubstituteVars(s, varMap));
    }
    auto newThenBody = std::make_shared<SeqStmts>(newThenStmts, ifStmt->span_);
    std::optional<SeqStmtsPtr> newElseBody;
    if (ifStmt->elseBody_) {
        std::vector<StmtPtr> newElseStmts;
        for (auto& s : ifStmt->elseBody_.value()->stmts_) {
            newElseStmts.push_back(SubstituteVars(s, varMap));
        }
        newElseBody = std::make_shared<SeqStmts>(newElseStmts, ifStmt->span_);
    }
    std::vector<VarPtr> newReturnVars;
    for (auto& v : ifStmt->returnVars_) {
        newReturnVars.push_back(LookupVarDef(v, varMap));
    }
    return std::make_shared<IfStmt>(ifStmt->condition_, newThenBody, newElseBody, newReturnVars, ifStmt->span_);
}

StmtPtr SubstituteVars(ForStmtPtr f, const VarExprMap& varMap)
{
    std::vector<StmtPtr> newBodyStmts;
    for (auto& s : f->body_->stmts_) {
        newBodyStmts.push_back(SubstituteVars(s, varMap));
    }
    auto newBody = std::make_shared<SeqStmts>(newBodyStmts, f->span_);
    std::vector<VarPtr> newReturnVars;
    for (auto& v : f->returnVars_) {
        newReturnVars.push_back(LookupVarDef(v, varMap));
    }
    return std::make_shared<ForStmt>(
        f->loopVar_, f->start_, f->stop_, f->step_, f->iterArgs_, newBody, newReturnVars, f->span_, f->attrs_);
}

StmtPtr SubstituteVars(YieldStmtPtr y, const VarExprMap& varMap)
{
    std::vector<ExprPtr> newValues;
    for (auto& v : y->value_) {
        newValues.push_back(LookupVarInExpr(v, varMap));
    }
    return std::make_shared<YieldStmt>(newValues, y->span_);
}

StmtPtr SubstituteVars(ContinueStmtPtr c, const VarExprMap& varMap)
{
    std::vector<ExprPtr> newValues;
    for (auto& v : c->value_) {
        newValues.push_back(LookupVarInExpr(v, varMap));
    }
    return std::make_shared<ContinueStmt>(newValues, c->span_);
}

StmtPtr SubstituteVars(StmtPtr stmt, const VarExprMap& varMap)
{
    if (!stmt || varMap.empty()) {
        return stmt;
    }
    if (auto t = As<TensorOpStmt>(stmt)) {
        return SubstituteVars(t, varMap);
    }
    if (auto i = As<IfStmt>(stmt)) {
        return SubstituteVars(i, varMap);
    }
    if (auto f = As<ForStmt>(stmt)) {
        return SubstituteVars(f, varMap);
    }
    if (auto seq = As<SeqStmts>(stmt)) {
        std::vector<StmtPtr> newStmts;
        for (auto& s : seq->stmts_) {
            newStmts.push_back(SubstituteVars(s, varMap));
        }
        return std::make_shared<SeqStmts>(newStmts, seq->span_);
    }
    if (auto y = As<YieldStmt>(stmt)) {
        return SubstituteVars(y, varMap);
    }
    if (auto c = As<ContinueStmt>(stmt)) {
        return SubstituteVars(c, varMap);
    }
    return stmt;
}

void CollectDefinedVars(TensorOpStmtPtr t, std::unordered_set<VarPtr>& defs)
{
    for (auto& v : t->result_) {
        if (v) {
            defs.insert(v);
        }
    }
}

void CollectDefinedVars(IfStmtPtr ifStmt, std::unordered_set<VarPtr>& defs)
{
    for (auto& v : ifStmt->returnVars_) {
        if (v) {
            defs.insert(v);
        }
    }
    for (auto& s : ifStmt->thenBody_->stmts_) {
        CollectDefinedVars(s, defs);
    }
    if (ifStmt->elseBody_) {
        for (auto& s : ifStmt->elseBody_.value()->stmts_) {
            CollectDefinedVars(s, defs);
        }
    }
}

void CollectDefinedVars(ForStmtPtr f, std::unordered_set<VarPtr>& defs)
{
    for (auto& v : f->returnVars_) {
        if (v) {
            defs.insert(v);
        }
    }
    for (auto& s : f->body_->stmts_) {
        CollectDefinedVars(s, defs);
    }
}

void CollectDefinedVars(StmtPtr stmt, std::unordered_set<VarPtr>& defs)
{
    if (!stmt) {
        return;
    }
    if (auto t = As<TensorOpStmt>(stmt)) {
        CollectDefinedVars(t, defs);
        return;
    }
    if (auto i = As<IfStmt>(stmt)) {
        CollectDefinedVars(i, defs);
        return;
    }
    if (auto f = As<ForStmt>(stmt)) {
        CollectDefinedVars(f, defs);
        return;
    }
}

std::unordered_set<VarPtr> CollectDefinedVarsFromBody(SeqStmtsPtr body)
{
    std::unordered_set<VarPtr> defs;
    if (!body) {
        return defs;
    }
    for (auto& s : body->stmts_) {
        CollectDefinedVars(s, defs);
    }
    return defs;
}

std::pair<SeqStmtsPtr, std::optional<SeqStmtsPtr>> ResolveBranchConflicts(
    SeqStmtsPtr processedThen, std::optional<SeqStmtsPtr> processedElse,
    const std::vector<std::string>& externalVarNames)
{
    auto thenDefs = CollectDefinedVarsFromBody(processedThen);
    auto elseDefs = CollectDefinedVarsFromBody(processedElse ? processedElse.value() : nullptr);

    std::unordered_set<int> elseRawMagics;
    for (auto& elseVar : elseDefs) {
        auto lt = std::dynamic_pointer_cast<const LogicalTensor>(elseVar);
        if (lt) {
            elseRawMagics.insert(lt->GetRawMagic());
        }
    }

    VarExprMap cloneMap;
    for (auto& v : thenDefs) {
        if (elseDefs.count(v) && !IsExternalVar(v->name_, externalVarNames)) {
            auto lt = std::dynamic_pointer_cast<const LogicalTensor>(v);
            bool sharesRawTensor = lt && elseRawMagics.count(lt->GetRawMagic()) > 0;
            if (lt) {
                cloneMap[v] = lt->Clone(sharesRawTensor);
            } else {
                cloneMap[v] = v->Clone();
            }
        }
    }

    std::optional<SeqStmtsPtr> resolvedElse;
    if (processedElse) {
        auto elseBody = processedElse.value();
        std::vector<StmtPtr> newElseStmts;
        for (auto& s : elseBody->stmts_) {
            newElseStmts.push_back(SubstituteVars(s, cloneMap));
        }
        resolvedElse = std::make_shared<SeqStmts>(newElseStmts, elseBody->span_);
    }

    return {processedThen, resolvedElse};
}

std::vector<VarPtr> CollectOutputVars(const std::vector<StmtPtr>& stmts)
{
    std::vector<VarPtr> outputs;
    for (auto& stmt : stmts) {
        if (auto t = As<TensorOpStmt>(stmt)) {
            for (auto& var : t->result_) {
                if (var) {
                    outputs.push_back(var);
                }
            }
        } else if (auto i = As<IfStmt>(stmt)) {
            for (auto& var : i->returnVars_) {
                if (var) {
                    outputs.push_back(var);
                }
            }
        } else if (auto f = As<ForStmt>(stmt)) {
            for (auto& var : f->returnVars_) {
                if (var) {
                    outputs.push_back(var);
                }
            }
        }
    }
    return outputs;
}

std::vector<StmtPtr> RemoveLastYieldStmt(const std::vector<StmtPtr>& stmts)
{
    std::vector<StmtPtr> result;
    for (size_t i = 0; i < stmts.size(); ++i) {
        if (i == stmts.size() - 1 && IsA<YieldStmt>(stmts[i])) {
            continue;
        }
        result.push_back(stmts[i]);
    }
    return result;
}

SeqStmtsPtr BuildAppendedBranch(
    SeqStmtsPtr branchBody, const std::vector<StmtPtr>& stmts, const VarExprMap& yieldMap,
    const std::vector<VarPtr>& newOutputVars, const std::vector<ExprPtr>& originalYieldValues, Span span)
{
    std::vector<StmtPtr> stmtsWithoutYield;
    if (branchBody) {
        stmtsWithoutYield = RemoveLastYieldStmt(branchBody->stmts_);
    }
    VarExprMap extendedYieldMap = yieldMap;
    for (auto& s : stmts) {
        stmtsWithoutYield.push_back(SubstituteReturnVarUses(s, extendedYieldMap));
        if (auto ifStmt = As<IfStmt>(s)) {
            auto ifYieldMap = BuildYieldVarMap(ifStmt, ifStmt->thenBody_);
            for (auto& [k, v] : ifYieldMap) {
                extendedYieldMap[k] = v;
            }
        } else if (auto forStmt = As<ForStmt>(s)) {
            auto forYieldMap = BuildYieldVarMapForForStmt(forStmt);
            for (auto& [k, v] : forYieldMap) {
                extendedYieldMap[k] = v;
            }
        }
    }
    std::vector<ExprPtr> yieldValues = originalYieldValues;
    for (auto& var : newOutputVars) {
        yieldValues.push_back(var);
    }
    stmtsWithoutYield.push_back(std::make_shared<YieldStmt>(yieldValues, span));
    return std::make_shared<SeqStmts>(stmtsWithoutYield, span);
}

std::vector<ExprPtr> ExtractYieldValues(SeqStmtsPtr body)
{
    if (!body || body->stmts_.empty()) {
        return {};
    }
    auto& lastStmt = body->stmts_.back();
    if (!IsA<YieldStmt>(lastStmt)) {
        return {};
    }
    auto yieldStmt = As<YieldStmt>(lastStmt);
    return yieldStmt->value_;
}

std::pair<std::vector<VarPtr>, VarExprMap> ComputeCloneableReturnVars(
    const std::vector<VarPtr>& vars, const std::vector<std::string>& externalVarNames)
{
    std::vector<VarPtr> result;
    VarExprMap cloneMap;
    for (auto& var : vars) {
        if (IsExternalVar(var->name_, externalVarNames)) {
            result.push_back(var);
        } else {
            auto cloned = var->Clone();
            cloneMap[var] = cloned;
            result.push_back(cloned);
        }
    }
    return {result, cloneMap};
}

IfStmtResult AppendIntoIfStmt(
    IfStmtPtr ifStmt, const std::vector<StmtPtr>& stmts, const std::vector<std::string>& externalVarNames)
{
    auto thenYieldMap = BuildYieldVarMap(ifStmt, ifStmt->thenBody_);
    auto elseYieldMap = BuildYieldVarMap(ifStmt, ifStmt->elseBody_ ? ifStmt->elseBody_.value() : nullptr);
    auto newOutputVars = CollectOutputVars(stmts);
    auto [returnVarsToAdd, outputCloneMap] = ComputeCloneableReturnVars(newOutputVars, externalVarNames);

    auto thenOriginalValues = ExtractYieldValues(ifStmt->thenBody_);
    auto elseOriginalValues = ExtractYieldValues(ifStmt->elseBody_ ? ifStmt->elseBody_.value() : nullptr);

    auto newThenBody =
        BuildAppendedBranch(ifStmt->thenBody_, stmts, thenYieldMap, newOutputVars, thenOriginalValues, ifStmt->span_);
    auto newElseBody = BuildAppendedBranch(
        ifStmt->elseBody_ ? ifStmt->elseBody_.value() : nullptr, stmts, elseYieldMap, newOutputVars, elseOriginalValues,
        ifStmt->span_);

    std::vector<VarPtr> newReturnVars = ifStmt->returnVars_;
    for (auto& var : returnVarsToAdd) {
        newReturnVars.push_back(var);
    }

    auto resultIfStmt =
        std::make_shared<IfStmt>(ifStmt->condition_, newThenBody, newElseBody, newReturnVars, ifStmt->span_);
    return IfStmtResult{resultIfStmt, outputCloneMap};
}

SeqStmtsPtr BuildPrependedBranchBody(
    const std::vector<StmtPtr>& prependStmts, std::optional<SeqStmtsPtr> originalBranch,
    const std::vector<VarPtr>& newOutputVars, const std::vector<ExprPtr>& originalYieldValues, Span span)
{
    std::vector<StmtPtr> stmts;
    VarExprMap accumulatedYieldMap;
    for (auto& s : prependStmts) {
        stmts.push_back(SubstituteReturnVarUses(s, accumulatedYieldMap));
        if (auto ifStmt = As<IfStmt>(s)) {
            auto ifYieldMap = BuildYieldVarMap(ifStmt, ifStmt->thenBody_);
            for (auto& [k, v] : ifYieldMap) {
                accumulatedYieldMap[k] = v;
            }
        } else if (auto forStmt = As<ForStmt>(s)) {
            auto forYieldMap = BuildYieldVarMapForForStmt(forStmt);
            for (auto& [k, v] : forYieldMap) {
                accumulatedYieldMap[k] = v;
            }
        }
    }
    if (originalBranch) {
        for (auto& s : RemoveLastYieldStmt(originalBranch.value()->stmts_)) {
            stmts.push_back(SubstituteReturnVarUses(s, accumulatedYieldMap));
        }
    }
    std::vector<ExprPtr> yieldValues;
    for (auto& var : newOutputVars) {
        yieldValues.push_back(var);
    }
    for (auto& v : originalYieldValues) {
        yieldValues.push_back(v);
    }
    stmts.push_back(std::make_shared<YieldStmt>(yieldValues, span));
    return std::make_shared<SeqStmts>(stmts, span);
}

IfStmtResult PrependIntoIfStmt(
    IfStmtPtr ifStmt, const std::vector<StmtPtr>& stmts, const std::vector<std::string>& externalVarNames)
{
    auto newOutputVars = CollectOutputVars(stmts);
    auto existingVars = std::unordered_set<VarPtr>(ifStmt->returnVars_.begin(), ifStmt->returnVars_.end());
    std::vector<VarPtr> nonExistingVars;
    for (auto& var : newOutputVars) {
        if (!existingVars.count(var)) {
            nonExistingVars.push_back(var);
        }
    }
    auto [prependVars, prependCloneMap] = ComputeCloneableReturnVars(nonExistingVars, externalVarNames);

    auto thenValues = ExtractYieldValues(ifStmt->thenBody_);
    auto elseValues = ExtractYieldValues(ifStmt->elseBody_ ? ifStmt->elseBody_.value() : nullptr);
    auto newThenBody = BuildPrependedBranchBody(stmts, ifStmt->thenBody_, newOutputVars, thenValues, ifStmt->span_);
    auto newElseBody = std::make_optional(
        BuildPrependedBranchBody(stmts, ifStmt->elseBody_, newOutputVars, elseValues, ifStmt->span_));

    std::vector<VarPtr> newReturnVars;
    for (auto& v : prependVars) {
        newReturnVars.push_back(v);
    }
    for (auto& v : ifStmt->returnVars_) {
        newReturnVars.push_back(v);
    }

    auto resultIfStmt =
        std::make_shared<IfStmt>(ifStmt->condition_, newThenBody, newElseBody, newReturnVars, ifStmt->span_);
    return IfStmtResult{resultIfStmt, prependCloneMap};
}

MergeSegmentResult MergeSegment(const std::vector<StmtPtr>& segment, const std::vector<std::string>& externalVarNames)
{
    std::vector<StmtPtr> collectedStmts;
    VarExprMap accumulatedCloneMap;

    for (size_t i = segment.size(); i > 0; --i) {
        auto& stmt = segment[i - 1];
        if (auto ifStmt = As<IfStmt>(stmt)) {
            std::reverse(collectedStmts.begin(), collectedStmts.end());
            auto appendResult = AppendIntoIfStmt(ifStmt, collectedStmts, externalVarNames);
            for (auto& [k, v] : appendResult.outputCloneMap) {
                accumulatedCloneMap[k] = v;
            }
            collectedStmts.clear();
            collectedStmts.push_back(appendResult.ifStmt);
        } else {
            collectedStmts.push_back(stmt);
        }
    }

    std::reverse(collectedStmts.begin(), collectedStmts.end());

    if (!collectedStmts.empty() && IsA<IfStmt>(collectedStmts.back())) {
        auto ifStmt = As<IfStmt>(collectedStmts.back());
        std::vector<StmtPtr> beforeStmts(collectedStmts.begin(), collectedStmts.end() - 1);
        std::vector<StmtPtr> nonForStmts;
        std::vector<StmtPtr> resultStmts;
        for (auto& s : beforeStmts) {
            if (IsA<ForStmt>(s)) {
                resultStmts.insert(resultStmts.end(), nonForStmts.begin(), nonForStmts.end());
                resultStmts.push_back(s);
                nonForStmts.clear();
            } else {
                nonForStmts.push_back(s);
            }
        }
        if (!nonForStmts.empty()) {
            auto prependResult = PrependIntoIfStmt(ifStmt, nonForStmts, externalVarNames);
            for (auto& [k, v] : prependResult.outputCloneMap) {
                accumulatedCloneMap[k] = v;
            }
            resultStmts.push_back(prependResult.ifStmt);
        } else {
            resultStmts.push_back(ifStmt);
        }
        return MergeSegmentResult{resultStmts, accumulatedCloneMap};
    }

    return MergeSegmentResult{collectedStmts, accumulatedCloneMap};
}

std::vector<StmtPtr> RebuildMergedStmtsRecursively(
    const std::vector<StmtPtr>& merged, const std::vector<std::string>& externalVarNames)
{
    std::vector<StmtPtr> finalResult;
    for (auto& stmt : merged) {
        if (auto ifStmt = As<IfStmt>(stmt)) {
            auto processedThen = MergeStmtsIntoIfStmtImpl(ifStmt->thenBody_, externalVarNames);
            std::optional<SeqStmtsPtr> processedElse;
            if (ifStmt->elseBody_) {
                processedElse = MergeStmtsIntoIfStmtImpl(ifStmt->elseBody_.value(), externalVarNames);
            }
            auto [resolvedThen, resolvedElse] = ResolveBranchConflicts(processedThen, processedElse, externalVarNames);
            finalResult.push_back(std::make_shared<IfStmt>(
                ifStmt->condition_, resolvedThen, resolvedElse, ifStmt->returnVars_, ifStmt->span_));
        } else if (auto forStmt = As<ForStmt>(stmt)) {
            auto processedBody = MergeStmtsIntoIfStmtImpl(forStmt->body_, externalVarNames);
            finalResult.push_back(std::make_shared<ForStmt>(
                forStmt->loopVar_, forStmt->start_, forStmt->stop_, forStmt->step_, forStmt->iterArgs_, processedBody,
                forStmt->returnVars_, forStmt->span_, forStmt->attrs_));
        } else {
            finalResult.push_back(stmt);
        }
    }
    return finalResult;
}

SeqStmtsPtr MergeStmtsIntoIfStmtImpl(SeqStmtsPtr seq, const std::vector<std::string>& externalVarNames)
{
    if (!seq) {
        return nullptr;
    }

    std::vector<StmtPtr> currentSegment;
    std::vector<StmtPtr> merged;
    VarExprMap accumulatedCloneMap;

    for (auto& stmt : seq->stmts_) {
        if (IsA<YieldStmt>(stmt) || IsA<ContinueStmt>(stmt)) {
            auto segResult = MergeSegment(currentSegment, externalVarNames);
            for (auto& s : segResult.stmts) {
                merged.push_back(s);
            }
            for (auto& [k, v] : segResult.outputCloneMap) {
                accumulatedCloneMap[k] = v;
            }
            merged.push_back(stmt);
            currentSegment.clear();
        } else {
            currentSegment.push_back(stmt);
        }
    }

    auto segResult = MergeSegment(currentSegment, externalVarNames);
    for (auto& s : segResult.stmts) {
        merged.push_back(s);
    }
    for (auto& [k, v] : segResult.outputCloneMap) {
        accumulatedCloneMap[k] = v;
    }

    if (!accumulatedCloneMap.empty() && !merged.empty() && IsA<YieldStmt>(merged.back())) {
        auto yieldStmt = As<YieldStmt>(merged.back());
        std::vector<ExprPtr> newValues;
        for (auto& v : yieldStmt->value_) {
            newValues.push_back(LookupVarInExpr(v, accumulatedCloneMap));
        }
        merged.back() = std::make_shared<YieldStmt>(newValues, yieldStmt->span_);
    }

    auto finalResult = RebuildMergedStmtsRecursively(merged, externalVarNames);
    return std::make_shared<SeqStmts>(finalResult, seq->span_);
}

} // namespace

SeqStmtsPtr MergeStmtsIntoIfStmt(SeqStmtsPtr seq, const std::vector<std::string>& externalVarNames)
{
    return MergeStmtsIntoIfStmtImpl(seq, externalVarNames);
}

} // namespace pypto::ir
