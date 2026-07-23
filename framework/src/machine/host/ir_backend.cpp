/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file ir_backend.cpp
 * \brief New SCF IR based control flow building functions.
 */

#include "machine/host/ir_backend.h"

#include "ir/kind_traits.h"
#include "tilefwk/pypto_fwk_log.h"
#include "tilefwk/error_code.h"
#include "tilefwk/platform.h"
#include "interface/utils/id_gen.h"
#include "machine/host/expr_generator.h"

namespace npu::tile_fwk {

namespace {

void FindExprFromForStmt(IrBackendContext& ctx, FunctionCache& cache, Linker& linker, const ir::ForStmtPtr& forStmt,
                         Function* dynFunc)
{
    auto iterName = GetLoopVarOriginName(forStmt->loopVar_);
    SymbolicScalar iterSymbol(iterName);
    linker.AddSymbol(iterSymbol);
    Function* loopFunc = IrBuildVirtualLoopFunc(ctx, forStmt.get(), dynFunc);
    linker.AddPrimaryExpressionForLoopBes(loopFunc, ExprPtrToSymbolicScalar(forStmt->start_));
    linker.AddPrimaryExpressionForLoopBes(loopFunc, ExprPtrToSymbolicScalar(forStmt->stop_));
    linker.AddPrimaryExpressionForLoopBes(loopFunc, ExprPtrToSymbolicScalar(forStmt->step_));
    IrParseValueDependDesc(loopFunc, {forStmt->start_, forStmt->stop_, forStmt->step_});
    std::vector<ir::ExprPtr> newCondStack;
    FindExprFromIRStmt(ctx, cache, linker, forStmt->body_, loopFunc, newCondStack);
}

void FindExprFromIfStmt(IrBackendContext& ctx, FunctionCache& cache, Linker& linker, const ir::IfStmtPtr& ifStmt,
                        Function* dynFunc, std::vector<ir::ExprPtr>& condStack)
{
    condStack.push_back(ifStmt->condition_);
    FindExprFromIRStmt(ctx, cache, linker, ifStmt->thenBody_, dynFunc, condStack);
    if (ifStmt->elseBody_) {
        FindExprFromIRStmt(ctx, cache, linker, ifStmt->elseBody_.value(), dynFunc, condStack);
    }
    condStack.pop_back();
}

void FindExprFromTensorOpStmt(FunctionCache& cache, Linker& linker, const ir::StmtPtr& stmt, Function* dynFunc,
                              std::vector<ir::ExprPtr>& condStack)
{
    auto pathFunc = ResolveCalleeFromOpCall(stmt);
    if (pathFunc == nullptr) {
        return;
    }
    for (auto& cond : condStack) {
        linker.AddPrimaryExpressionForLoopPathCond(pathFunc, ExprPtrToSymbolicScalar(cond));
        IrParseValueDependDesc(dynFunc, {cond});
    }
    FindAllExpression(cache, linker, pathFunc);
}

void VisitIfStmtForControlFlow(IrBackendContext& ctx, FunctionCache& cache, Linker& linker,
                               const std::string& sectionName, const ir::IfStmtPtr& ifStmt, Function* dynFunc,
                               std::unordered_map<int, int>& slotIdxMapping,
                               DyndevFunctionAttribute::FunctionGroup& group,
                               std::unordered_map<Function*, Function*>& rootTileDict,
                               std::ostringstream& controlFlowOss, std::ostringstream& expressionOss,
                               std::ostringstream& exprHeaderOss, int indent, const std::string& expName,
                               std::vector<std::string>& exprSrcFiles, ValDependTensorMeta& valDependTensorMeta)
{
    auto cond = SymbolicExpressionTable::BuildExpression(ExprPtrToSymbolicScalar(ifStmt->condition_));
    controlFlowOss << std::setw(indent * TABSIZE) << ' ' << "if (" << cond << ") {\n";
    VisitIRStmtForControlFlow(ctx, cache, linker, sectionName, ifStmt->thenBody_, dynFunc, slotIdxMapping, group,
                              rootTileDict, controlFlowOss, expressionOss, exprHeaderOss, indent + 1, expName,
                              exprSrcFiles, valDependTensorMeta);
    if (ifStmt->elseBody_) {
        controlFlowOss << std::setw(indent * TABSIZE) << ' ' << "} else {\n";
        VisitIRStmtForControlFlow(ctx, cache, linker, sectionName, ifStmt->elseBody_.value(), dynFunc, slotIdxMapping,
                                  group, rootTileDict, controlFlowOss, expressionOss, exprHeaderOss, indent + 1,
                                  expName, exprSrcFiles, valDependTensorMeta);
    }
    controlFlowOss << std::setw(indent * TABSIZE) << ' ' << "}\n";
}

} // namespace

SymbolicScalar ExprPtrToSymbolicScalar(const ir::ExprPtr& expr)
{
    if (!expr) {
        return SymbolicScalar();
    }
    auto rawScalar = std::dynamic_pointer_cast<const RawSymbolicScalar>(expr);
    ASSERT(DevCommonErr::PARAM_CHECK_FAILED, rawScalar != nullptr)
        << "ExprPtr is not a RawSymbolicScalar, cannot convert to SymbolicScalar";
    return SymbolicScalar(std::const_pointer_cast<RawSymbolicScalar>(rawScalar));
}

std::string GetLoopVarOriginName(const ir::VarPtr& loopVar)
{
    auto& ctx = IRContext::Get();
    auto originName = ctx.GetOriginName(loopVar);
    return originName.empty() ? loopVar->name_ : originName;
}

Function* ResolveCalleeFromOpCall(const ir::StmtPtr& stmt)
{
    auto tensorOp = ir::As<ir::TensorOpStmt>(stmt);
    if (tensorOp == nullptr) {
        return nullptr;
    }
    for (auto& [key, value] : tensorOp->attrs_) {
        if (key == "callee") {
            auto calleeName = std::any_cast<std::string>(value);
            return Program::GetInstance().GetFunctionByMagicName(calleeName);
        }
    }
    return nullptr;
}

bool IsOpCallStmt(const ir::StmtPtr& stmt)
{
    auto tensorOp = ir::As<ir::TensorOpStmt>(stmt);
    if (tensorOp == nullptr) {
        return false;
    }
    for (auto& [key, value] : tensorOp->attrs_) {
        (void)value;
        if (key == "callee") {
            return true;
        }
    }
    return false;
}

Function* IrBuildVirtualLoopFunc(IrBackendContext& ctx, const ir::ForStmt* forStmt, Function* dynFunc)
{
    auto currDynFuncAttr = Program::GetInstance().GetCurrentDynamicFunction()->GetDyndevAttribute();
    if (currDynFuncAttr == nullptr) {
        return dynFunc;
    }
    auto it = ctx.forStmtLoopFuncMap.find(forStmt);
    if (it != ctx.forStmtLoopFuncMap.end()) {
        return it->second.get();
    }
    auto loopFuncId = IdGen<IdType::FUNCTION>::Inst().NewId();
    auto iterName = GetLoopVarOriginName(forStmt->loopVar_);
    auto loopFuncMagic = dynFunc->GetRawName() + "_loop_" + iterName + "_" + std::to_string(loopFuncId);
    auto loopFuncName = dynFunc->GetRawName() + "_loop_" + iterName + "_" + std::to_string(loopFuncId);
    auto virtualLoopFunc = std::make_shared<Function>(Program::GetInstance(), loopFuncMagic, loopFuncName, dynFunc);
    virtualLoopFunc->SetFunctionType(FunctionType::DYNAMIC_LOOP);
    virtualLoopFunc->SetGraphType(GraphType::TENSOR_GRAPH);
    virtualLoopFunc->SetUnderDynamicFunction(true);

    auto begin = ExprPtrToSymbolicScalar(forStmt->start_);
    auto end = ExprPtrToSymbolicScalar(forStmt->stop_);
    auto step = ExprPtrToSymbolicScalar(forStmt->step_);
    LoopRange range(begin, end, step);
    bool parallel = forStmt->GetAttr<bool>("parallel", false);
    bool submitBeforeLoop = forStmt->GetAttr<bool>("submit_before_loop", false);
    auto attr = std::make_shared<DynloopFunctionAttribute>(iterName, range, range, submitBeforeLoop, parallel);
    virtualLoopFunc->SetDynloopAttribute(attr);

    virtualLoopFunc->ComputeHash();
    Function* result = virtualLoopFunc.get();
    ctx.forStmtLoopFuncMap[forStmt] = std::move(virtualLoopFunc);
    return result;
}

void IrParseValueDependDesc(Function* func, std::initializer_list<ir::ExprPtr> exprs)
{
    auto currDynFuncAttr = Program::GetInstance().GetCurrentDynamicFunction()->GetDyndevAttribute();
    if (currDynFuncAttr == nullptr) {
        return;
    }
    auto& desc = currDynFuncAttr->valueDependDescDict[func];
    for (auto& expr : exprs) {
        auto ss = ExprPtrToSymbolicScalar(expr);
        std::vector<RawSymbolicScalarPtr> callList = LookupExpressionByOpcode(ss.Raw(), SymbolicOpcode::T_MOP_CALL);
        for (auto& call : callList) {
            auto caller = call->GetExpressionOperandList()[0];
            if (!caller->IsSymbol()) {
                continue;
            }
            std::string name = caller->GetSymbolName();
            if (CallIsGetInputData(name)) {
                desc.getInputDataCount++;
            } else if (CallIsGetTensorData(name)) {
                desc.getTensorDataCount++;
            }
        }
    }
}

void InsertCacheStopForContrlFlow(IrBackendContext& ctx, const ir::ForStmt* forStmt, Function* dynFunc, int indent,
                                  std::ostringstream& controlFlowOss, ValDependTensorMeta& valDependTensorMeta)
{
    auto currDynFuncAttr = Program::GetInstance().GetCurrentDynamicFunction()->GetDyndevAttribute();
    if (currDynFuncAttr == nullptr) {
        return;
    }
    Function* loopFunc = dynFunc;
    auto it = ctx.forStmtLoopFuncMap.find(forStmt);
    if (it != ctx.forStmtLoopFuncMap.end()) {
        loopFunc = it->second.get();
    }
    if (currDynFuncAttr->valueDependDescDict.count(loopFunc) == 0) {
        return;
    }
    auto valueDependDesc = currDynFuncAttr->valueDependDescDict[loopFunc];
    if (valueDependDesc.getInputDataCount + valueDependDesc.getTensorDataCount != 0) {
        controlFlowOss << std::setw(indent * TABSIZE) << ' '
                       << "RUNTIME_RootStitch(RUNTIME_FUNCKEY_CACHESTOP); // force stop cache due to value "
                          "depend in control\n";
        valDependTensorMeta.disableCtrlFlowCache = true;
    }
}

void InsertWaitAicoreStartForControlFlow(const ir::ForStmt* forStmt, int indent, std::ostringstream& controlFlowOss,
                                         ValDependTensorMeta& valDependTensorMeta)
{
    SymbolicScalar startScalar = ExprPtrToSymbolicScalar(forStmt->start_);
    SymbolicScalar stopScalar = ExprPtrToSymbolicScalar(forStmt->stop_);
    SymbolicScalar stepScalar = ExprPtrToSymbolicScalar(forStmt->step_);
    const SymbolicScalar* loopBounds[] = {&startScalar, &stopScalar, &stepScalar};
    for (const SymbolicScalar* boundExpr : loopBounds) {
        if (!boundExpr->IsValid() || boundExpr->IsImmediate()) {
            continue;
        }
        if (SymbolicExpressionTable::CheckExprDependCore(boundExpr->Raw(), valDependTensorMeta.tensorNameToDependCore,
                                                         valDependTensorMeta.valDependMap)) {
            controlFlowOss << std::setw(indent * TABSIZE) << ' ' << "WaitAicoreStart(startArgs);\n";
            break;
        }
    }
}

void FindExprFromIRStmt(IrBackendContext& ctx, FunctionCache& cache, Linker& linker, const ir::StmtPtr& stmt,
                        Function* dynFunc, std::vector<ir::ExprPtr>& condStack)
{
    if (!stmt) {
        return;
    }
    switch (stmt->GetKind()) {
        case ir::ObjectKind::SeqStmts: {
            if (auto seq = ir::As<ir::SeqStmts>(stmt)) {
                for (auto& child : seq->stmts_) {
                    FindExprFromIRStmt(ctx, cache, linker, child, dynFunc, condStack);
                }
            }
            break;
        }
        case ir::ObjectKind::ForStmt: {
            if (auto forStmt = ir::As<ir::ForStmt>(stmt)) {
                FindExprFromForStmt(ctx, cache, linker, forStmt, dynFunc);
            }
            break;
        }
        case ir::ObjectKind::IfStmt: {
            if (auto ifStmt = ir::As<ir::IfStmt>(stmt)) {
                FindExprFromIfStmt(ctx, cache, linker, ifStmt, dynFunc, condStack);
            }
            break;
        }
        case ir::ObjectKind::TensorOpStmt:
            FindExprFromTensorOpStmt(cache, linker, stmt, dynFunc, condStack);
            break;
        default:
            break;
    }
}

void FindAllExpressionFromIR(IrBackendContext& ctx, FunctionCache& cache, Linker& linker, Function* func)
{
    if (func->IsFunctionTypeAndGraphType(FunctionType::DYNAMIC, GraphType::TENSOR_GRAPH)) {
        auto& body = func->body_;
        if (body != nullptr) {
            std::vector<ir::ExprPtr> condStack;
            FindExprFromIRStmt(ctx, cache, linker, body, func, condStack);
        }
        for (auto& callee : GetCalleeList(cache, func)) {
            if (callee->GetFunctionType() != FunctionType::DYNAMIC_LOOP) {
                FindAllExpression(cache, linker, callee);
            }
        }
    }
}

void VisitForStmtForControlFlow(IrBackendContext& ctx, FunctionCache& cache, Linker& linker,
                                const std::string& sectionName, const ir::ForStmtPtr& forStmt, Function* dynFunc,
                                std::unordered_map<int, int>& slotIdxMapping,
                                DyndevFunctionAttribute::FunctionGroup& group,
                                std::unordered_map<Function*, Function*>& rootTileDict,
                                std::ostringstream& controlFlowOss, std::ostringstream& expressionOss,
                                std::ostringstream& exprHeaderOss, int indent, const std::string& expName,
                                std::vector<std::string>& exprSrcFiles, ValDependTensorMeta& valDependTensorMeta)
{
    auto iterBegin = SymbolicExpressionTable::BuildExpression(ExprPtrToSymbolicScalar(forStmt->start_));
    auto iterEnd = SymbolicExpressionTable::BuildExpression(ExprPtrToSymbolicScalar(forStmt->stop_));
    auto iterStep = SymbolicExpressionTable::BuildExpression(ExprPtrToSymbolicScalar(forStmt->step_));
    auto iterSymbolName = GetLoopVarOriginName(forStmt->loopVar_);
    auto iterVar = "VAR_" + iterSymbolName;

    Function* loopFunc = IrBuildVirtualLoopFunc(ctx, forStmt.get(), dynFunc);
    bool submitBeforeLoop = forStmt->GetAttr<bool>("submit_before_loop", false);
    bool parallel = forStmt->GetAttr<bool>("parallel", false);
    bool supportParallelLoop = (config::GetRuntimeOption<uint16_t>(DEVICE_SCHED_PARALLELISM) > 1);
    bool isDav3510 = Platform::Instance().GetSoc().GetNPUArch() == NPUArch::DAV_3510;
    bool needCrossDie = isDav3510 && parallel;

    controlFlowOss << std::setw(indent * TABSIZE) << ' ' << "// hash=" << loopFunc->GetFunctionHash() << "\n";
    if (submitBeforeLoop) {
        controlFlowOss << std::setw(indent * TABSIZE) << ' '
                       << "RUNTIME_RootStitch(RUNTIME_FUNCKEY_LOOP_BARRIER); // force submit before LOOP \n";
    }

    InsertCacheStopForContrlFlow(ctx, forStmt.get(), dynFunc, indent, controlFlowOss, valDependTensorMeta);
    InsertWaitAicoreStartForControlFlow(forStmt.get(), indent, controlFlowOss, valDependTensorMeta);

    controlFlowOss << std::setw(indent * TABSIZE) << ' ' << "LOOP(" << iterVar << ", " << iterBegin << ", " << iterEnd
                   << ", " << iterStep << ") {\n";
    controlFlowOss << std::setw((indent + 1) * TABSIZE) << ' ' << "VALUE_" << iterSymbolName << " = " << iterVar
                   << ";\n";
    if (parallel && supportParallelLoop) {
        controlFlowOss << std::setw((indent + 1) * TABSIZE) << ' '
                       << "RUNTIME_RootStitch(RUNTIME_FUNCKEY_PARALLEL_FOR_BEGIN); // entry parallel for loop \n";
    }
    if (needCrossDie) {
        controlFlowOss << std::setw((indent + 1) * TABSIZE) << ' ' << "RUNTIME_CalcLoopDieId(" << iterSymbolName << ", "
                       << iterVar << ", " << iterEnd << ", " << iterStep << "," << DIE_NUM << ");\n";
    }

    VisitIRStmtForControlFlow(ctx, cache, linker, sectionName, forStmt->body_, dynFunc, slotIdxMapping, group,
                              rootTileDict, controlFlowOss, expressionOss, exprHeaderOss, indent + 1, expName,
                              exprSrcFiles, valDependTensorMeta);

    if (needCrossDie) {
        controlFlowOss << std::setw((indent + 1) * TABSIZE) << ' ' << "RUNTIME_ClearLoopDieId(" << iterSymbolName
                       << ");\n";
    }
    if (parallel && supportParallelLoop) {
        controlFlowOss << std::setw((indent + 1) * TABSIZE) << ' '
                       << "RUNTIME_RootStitch(RUNTIME_FUNCKEY_PARALLEL_FOR_END); // leave parallel for loop \n";
    }
    controlFlowOss << std::setw(indent * TABSIZE) << ' ' << "}\n";
}

void VisitIRStmtForControlFlow(IrBackendContext& ctx, FunctionCache& cache, Linker& linker,
                               const std::string& sectionName, const ir::StmtPtr& stmt, Function* dynFunc,
                               std::unordered_map<int, int>& slotIdxMapping,
                               DyndevFunctionAttribute::FunctionGroup& group,
                               std::unordered_map<Function*, Function*>& rootTileDict,
                               std::ostringstream& controlFlowOss, std::ostringstream& expressionOss,
                               std::ostringstream& exprHeaderOss, int indent, const std::string& expName,
                               std::vector<std::string>& exprSrcFiles, ValDependTensorMeta& valDependTensorMeta)
{
    if (!stmt) {
        return;
    }
    switch (stmt->GetKind()) {
        case ir::ObjectKind::SeqStmts: {
            if (auto seq = ir::As<ir::SeqStmts>(stmt)) {
                for (auto& child : seq->stmts_) {
                    VisitIRStmtForControlFlow(ctx, cache, linker, sectionName, child, dynFunc, slotIdxMapping, group,
                                              rootTileDict, controlFlowOss, expressionOss, exprHeaderOss, indent,
                                              expName, exprSrcFiles, valDependTensorMeta);
                }
            }
            break;
        }
        case ir::ObjectKind::ForStmt: {
            if (auto forStmt = ir::As<ir::ForStmt>(stmt)) {
                VisitForStmtForControlFlow(ctx, cache, linker, sectionName, forStmt, dynFunc, slotIdxMapping, group,
                                           rootTileDict, controlFlowOss, expressionOss, exprHeaderOss, indent, expName,
                                           exprSrcFiles, valDependTensorMeta);
            }
            break;
        }
        case ir::ObjectKind::IfStmt: {
            if (auto ifStmt = ir::As<ir::IfStmt>(stmt)) {
                VisitIfStmtForControlFlow(ctx, cache, linker, sectionName, ifStmt, dynFunc, slotIdxMapping, group,
                                          rootTileDict, controlFlowOss, expressionOss, exprHeaderOss, indent, expName,
                                          exprSrcFiles, valDependTensorMeta);
            }
            break;
        }
        case ir::ObjectKind::TensorOpStmt: {
            if (!IsOpCallStmt(stmt)) {
                return;
            }
            auto pathFunc = ResolveCalleeFromOpCall(stmt);
            if (pathFunc == nullptr) {
                return;
            }
            BuildControlFlow(cache, linker, sectionName, pathFunc, slotIdxMapping, group, rootTileDict, controlFlowOss,
                             expressionOss, exprHeaderOss, indent, expName, exprSrcFiles, valDependTensorMeta);
            break;
        }
        default:
            break;
    }
}

void BuildControlFlowFromIR(IrBackendContext& ctx, FunctionCache& cache, Linker& linker, const std::string& sectionName,
                            Function* func, std::unordered_map<int, int>& slotIdxMapping,
                            DyndevFunctionAttribute::FunctionGroup& group,
                            std::unordered_map<Function*, Function*>& rootTileDict, std::ostringstream& controlFlowOss,
                            std::ostringstream& expressionOss, std::ostringstream& exprHeaderOss, int indent,
                            const std::string& expName, std::vector<std::string>& exprSrcFiles,
                            ValDependTensorMeta& valDependTensorMeta)
{
    ExprBatchGenerator generator(config::GetEmitPath("kernel_aicpu"), 0, 0);
    BuildControlFlowHeader(generator, linker, func, sectionName, expName, controlFlowOss, expressionOss, exprHeaderOss,
                           valDependTensorMeta);
    if (NeedCrossDie(func)) {
        controlFlowOss << std::setw(indent * TABSIZE) << ' ' << "RUNTIME_RootGetDieId(" << 0 << ");\n";
    }

    if (func->body_ != nullptr) {
        VisitIRStmtForControlFlow(ctx, cache, linker, sectionName, func->body_, func, slotIdxMapping, group,
                                  rootTileDict, controlFlowOss, expressionOss, exprHeaderOss, indent + 1, expName,
                                  exprSrcFiles, valDependTensorMeta);
    }

    BuildControlFlowFooter(generator, controlFlowOss, exprHeaderOss, indent);
}

} // namespace npu::tile_fwk
