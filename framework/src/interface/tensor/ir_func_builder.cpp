/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "ir_func_builder.h"

#include "logical_tensor.h"
#include "raw_tensor.h"
#include "interface/function/function.h"
#include "interface/operation/attribute.h"
#include "interface/operation/operation.h"
#include "interface/operation/opcode.h"
#include "interface/program/program.h"
#include "interface/tensor/tensor_slot.h"
#include "tilefwk/tensor.h"
#include "interface/utils/id_gen.h"
#include "interface/configs/config_manager.h"
#include "interface/tensor/irbuilder.h"
#include "interface/tensor/ir_tensor_op_rebuild.h"

#include "ir/expr.h"
#include "ir/kind_traits.h"
#include "ir/type.h"

using namespace pypto;

namespace npu::tile_fwk {

RootFunctionBuilder::RootFunctionBuilder(Function* parentFunc)
    : program_(Program::GetInstance()), parentFunc_(parentFunc) {}

std::shared_ptr<Function> RootFunctionBuilder::Build(const ir::FunctionPtr& irFunc)
{
    InitDynFunc(irFunc);
    auto stmtsWithCall = TransformBody(irFunc->body_);
    dynFunc_->body_ = ir::SeqStmts::Wrap(stmtsWithCall, irFunc->span_);
    FinalizeDynFunc(irFunc);
    return dynFunc_;
}

void RootFunctionBuilder::InitDynFunc(const ir::FunctionPtr& irFunc)
{
    for (const auto& param : irFunc->params_) {
        auto constLT = std::dynamic_pointer_cast<const LogicalTensor>(param);
        ASSERT(constLT) << "RootFunctionBuilder: param is not a LogicalTensor: " << param->name_;
        auto lt = std::const_pointer_cast<LogicalTensor>(constLT);
        logicalParams_.push_back(lt);
    }

    auto funcMagicName = irFunc->name_ + "_" + std::to_string(IdGen<IdType::FUNCTION>::Inst().NewId());
    dynFunc_ = std::make_shared<Function>(program_, funcMagicName, irFunc->name_, parentFunc_);
    dynFunc_->SetFunctionType(FunctionType::DYNAMIC);
    dynFunc_->SetGraphType(GraphType::TENSOR_GRAPH);

    for (auto& param : logicalParams_) {
        dynFunc_->AddOriginIncast(param);
        dynFunc_->inCasts_.push_back(param);
        dynFunc_->GetTensorMap().Insert(param);
    }
}

void RootFunctionBuilder::FinalizeDynFunc(const ir::FunctionPtr& irFunc)
{
    dynFunc_->name_ = irFunc->name_;
    dynFunc_->funcType_ = ir::FunctionType::ORCHESTRATION;
    for (auto& param : logicalParams_) {
        dynFunc_->params_.push_back(std::static_pointer_cast<const ir::Var>(param));
    }
    dynFunc_->ComputeHash();
    BuildDynSlotScope();
}

bool RootFunctionBuilder::IsPureTensorOpSeq(const ir::SeqStmtsPtr& seq)
{
    if (!seq || seq->stmts_.empty()) {
        return false;
    }
    for (auto& stmt : seq->stmts_) {
        if (stmt->GetKind() != ir::ObjectKind::TensorOpStmt) {
            return false;
        }
    }
    return true;
}

std::vector<std::vector<ir::StmtPtr>> RootFunctionBuilder::SplitIntoTensorOpSegments(const ir::SeqStmtsPtr& seq)
{
    std::vector<std::vector<ir::StmtPtr>> segments;
    std::vector<ir::StmtPtr> currentRun;
    for (auto& child : seq->stmts_) {
        if (child->GetKind() == ir::ObjectKind::TensorOpStmt) {
            currentRun.push_back(child);
        } else {
            if (!currentRun.empty()) {
                segments.push_back(currentRun);
                currentRun.clear();
            }
            segments.push_back({child});
        }
    }
    if (!currentRun.empty()) {
        segments.push_back(currentRun);
    }
    return segments;
}

ir::StmtPtr RootFunctionBuilder::ProcessTensorOp(
    std::shared_ptr<Function> pathFunc, const ir::StmtPtr& stmt,
    std::unordered_set<std::shared_ptr<LogicalTensor>>& allInputs,
    std::unordered_set<std::shared_ptr<LogicalTensor>>& allOutputs,
    std::unordered_set<std::shared_ptr<LogicalTensor>>& definedOutputs)
{
    if (stmt->GetKind() != ir::ObjectKind::TensorOpStmt) {
        return stmt;
    }
    auto tensorOpStmt = std::static_pointer_cast<const ir::TensorOpStmt>(stmt);

    LogicalTensors iOperands;
    for (auto& arg : tensorOpStmt->args_) {
        if (!arg) {
            continue;
        }
        iOperands.push_back(std::const_pointer_cast<LogicalTensor>(std::static_pointer_cast<const LogicalTensor>(arg)));
    }

    LogicalTensors oOperands;
    for (auto& result : tensorOpStmt->result_) {
        if (!result) {
            continue;
        }
        auto lt = std::const_pointer_cast<LogicalTensor>(std::static_pointer_cast<const LogicalTensor>(result));
        pathFunc->GetTensorMap().Insert(lt);
        oOperands.push_back(lt);
    }

    for (auto& op : iOperands) {
        allInputs.insert(op);
    }
    for (auto& op : oOperands) {
        allOutputs.insert(op);
        definedOutputs.insert(op);
    }

    ir::StmtPtr opStmt = RebuildTensorOpStmt(
        tensorOpStmt, tensorOpStmt->result_, tensorOpStmt->result_token_, tensorOpStmt->args_, tensorOpStmt->tokens_,
        tensorOpStmt->span_, pathFunc.get());
    if (tensorOpStmt->result_token_) {
        pathFunc->GetVarDependency().AddProducer(tensorOpStmt->result_token_, opStmt);
    }
    for (auto& token : tensorOpStmt->tokens_) {
        pathFunc->GetVarDependency().AddConsumer(token, opStmt);
    }
    return opStmt;
}

void RootFunctionBuilder::ComputeIncast(
    Function& pathFunc,
    const std::unordered_set<std::shared_ptr<LogicalTensor>>& allInputs,
    const std::unordered_set<std::shared_ptr<LogicalTensor>>& definedOutputs)
{
    std::unordered_set<std::shared_ptr<LogicalTensor>> incastPtrs;
    for (auto& input : allInputs) {
        if (definedOutputs.find(input) == definedOutputs.end() && incastPtrs.find(input) == incastPtrs.end()) {
            incastPtrs.insert(input);
            pathFunc.AddOriginIncast(input);
        }
    }
}

void RootFunctionBuilder::ComputeOutcast(
    Function& pathFunc,
    const std::unordered_set<std::shared_ptr<LogicalTensor>>& allOutputs)
{
    std::unordered_set<std::shared_ptr<LogicalTensor>> outcastPtrs;
    for (auto& output : allOutputs) {
        if (outcastPtrs.find(output) == outcastPtrs.end()) {
            bool neededByConsumer = consumedRawMagics_.count(output->GetRawMagic()) > 0;
            bool isFuncOutput = paramRawMagics_.count(output->GetRawMagic()) > 0;
            if (neededByConsumer || isFuncOutput) {
                outcastPtrs.insert(output);
                pathFunc.AddOriginOutcast(output);
            }
        }
    }
}

std::shared_ptr<Function> RootFunctionBuilder::CreatePathFunc(
    const ir::SeqStmtsPtr& seq, const std::string& loopVarName)
{
    auto pathFuncId = IdGen<IdType::FUNCTION>::Inst().NewId();
    std::string pathSuffix =
        loopVarName.empty() ? std::to_string(pathFuncId) : loopVarName + "_" + std::to_string(pathFuncId);
    auto pathMagicName = dynFunc_->GetRawName() + "_path_" + pathSuffix;
    auto pathFunc = std::make_shared<Function>(program_, pathMagicName, pathMagicName, dynFunc_.get());
    pathFunc->SetFunctionType(FunctionType::DYNAMIC_LOOP_PATH);
    pathFunc->SetGraphType(GraphType::TENSOR_GRAPH);
    pathFunc->SetUnderDynamicFunction(true);
    program_.InsertFuncToFunctionMap(pathMagicName, pathFunc);

    std::unordered_set<std::shared_ptr<LogicalTensor>> definedOutputs;
    std::unordered_set<std::shared_ptr<LogicalTensor>> allInputs;
    std::unordered_set<std::shared_ptr<LogicalTensor>> allOutputs;
    for (auto& stmt : seq->stmts_) {
        (void)ProcessTensorOp(pathFunc, stmt, allInputs, allOutputs, definedOutputs);
    }
    ComputeIncast(*pathFunc, allInputs, definedOutputs);
    pathFunc->name_ = pathMagicName;
    pathFunc->funcType_ = ir::FunctionType::IN_CORE;
    return pathFunc;
}

int RootFunctionBuilder::FindOrCreateSlot(
    const std::shared_ptr<LogicalTensor>& lt, const std::shared_ptr<TensorSlotManager>& slotManager, Function* func,
    bool isInput)
{
    int rawMagic = lt->tensor->GetRawMagic();
    int existingSlot = slotManager->LookupSlotIndexByRawMagic(rawMagic);
    if (existingSlot != -1) {
        return existingSlot;
    }

    auto newTensor = std::make_unique<Tensor>(lt);
    if (isInput) {
        slotManager->MarkInput(*newTensor);
    } else {
        slotManager->MarkOutput(*newTensor);
    }

    func->GetSlotTensors().push_back(std::move(newTensor));
    return slotManager->LookupSlotIndexByRawMagic(rawMagic);
}

void RootFunctionBuilder::BuildPathFuncSlotScope(
    Function* pathFunc, const std::shared_ptr<TensorSlotScope>& scope,
    const LogicalTensors& originalIncasts, const LogicalTensors& originalOutcasts)
{
    auto slotManager = program_.GetTensorSlotManager();

    scope->ioslot.incastSlot.resize(pathFunc->GetIncast().size());
    for (size_t idx = 0; idx < pathFunc->GetIncast().size(); idx++) {
        int slotIndex = FindOrCreateSlot(originalIncasts[idx], slotManager, pathFunc, true);
        scope->ioslot.incastSlot[idx] = {slotIndex};
    }

    scope->ioslot.outcastSlot.resize(pathFunc->GetOutcast().size());
    for (size_t idx = 0; idx < pathFunc->GetOutcast().size(); idx++) {
        int slotIndex = FindOrCreateSlot(originalOutcasts[idx], slotManager, pathFunc, false);
        scope->ioslot.outcastSlot[idx] = {slotIndex};
    }

    for (auto& op : pathFunc->Operations()) {
        if (op.GetOpcode() == Opcode::OP_ASSEMBLE || op.GetOpcode() == Opcode::OP_ASSEMBLE_SSA) {
            for (auto& oOperand : op.GetOOperands()) {
                int slotIndex = FindOrCreateSlot(oOperand, slotManager, pathFunc, false);
                scope->constructAssembleSlotList.push_back(slotIndex);
            }
        }
    }
}

void RootFunctionBuilder::BuildDynSlotScope()
{
    auto attr = std::make_shared<DyndevFunctionAttribute>();
    dynFunc_->SetDyndevAttribute(attr);

    auto slotManager = program_.GetTensorSlotManager();
    auto dynScope = std::make_shared<TensorSlotScope>(dynFunc_.get());
    dynFunc_->SetSlotScope(dynScope);
    slotManager->scopeList.push_back(dynScope);

    dynScope->ioslot.incastSlot.resize(logicalParams_.size());
    for (size_t idx = 0; idx < logicalParams_.size(); idx++) {
        int slotIndex = FindOrCreateSlot(logicalParams_[idx], slotManager, dynFunc_.get(), true);
        dynScope->ioslot.incastSlot[idx] = {slotIndex};
        const Tensor* tensor = slotManager->LookupTensorByRawMagic(logicalParams_[idx]->tensor->GetRawMagic());
        attr->startArgsInputTensorList.emplace_back(*tensor);
    }
    attr->startArgsInputLogicalTensorList.resize(attr->startArgsInputTensorList.size());
    for (size_t k = 0; k < attr->startArgsInputTensorList.size(); k++) {
        attr->startArgsInputLogicalTensorList[k] = attr->startArgsInputTensorList[k].get().GetStorage(false);
    }

    auto calleeList = dynFunc_->GetCalleeFunctionList();
    for (auto callee : calleeList) {
        if (callee == nullptr) {
            continue;
        }
        const auto calleeScope = callee->GetSlotScope();
        if (calleeScope == nullptr) {
            continue;
        }
        auto& oSlot = dynScope->originalIocastsSlot.outcastSlot;
        oSlot.insert(oSlot.end(), calleeScope->ioslot.outcastSlot.begin(), calleeScope->ioslot.outcastSlot.end());
    }
    dynScope->ioslot.outcastSlot = dynScope->originalIocastsSlot.outcastSlot;

    dynFunc_->CleanRedundantOutCast();
    dynFunc_->InferParamDirection();
}

bool RootFunctionBuilder::IsPlaceholderCallStmt(const ir::StmtPtr& stmt)
{
    if (stmt->GetKind() != ir::ObjectKind::TensorOpStmt) {
        return false;
    }
    auto tensorOp = std::static_pointer_cast<const ir::TensorOpStmt>(stmt);
    if (tensorOp->opcode_ != "CALL") {
        return false;
    }
    for (auto& attr : tensorOp->attrs_) {
        if (attr.first == "placeholder_funcname") {
            return true;
        }
    }
    return false;
}

std::string RootFunctionBuilder::GetPlaceholderFuncname(const ir::StmtPtr& stmt)
{
    auto tensorOp = std::static_pointer_cast<const ir::TensorOpStmt>(stmt);
    for (auto& [key, value] : tensorOp->attrs_) {
        if (key == "placeholder_funcname") {
            return std::any_cast<std::string>(value);
        }
    }
    return "";
}

std::unordered_set<std::shared_ptr<LogicalTensor>> RootFunctionBuilder::CollectAllOutputs(Function& pathFunc)
{
    std::unordered_set<std::shared_ptr<LogicalTensor>> allOutputs;
    for (auto& op : pathFunc.Operations()) {
        for (auto& oOperand : op.GetOOperands()) {
            allOutputs.insert(oOperand);
        }
    }
    return allOutputs;
}

ir::StmtPtr RootFunctionBuilder::CreatePathFuncAndPlaceholder(
    const ir::SeqStmtsPtr& seq, const std::string& loopVarName)
{
    auto pathFunc = CreatePathFunc(seq, loopVarName);

    for (auto& incast : pathFunc->GetOriginIncast()) {
        consumedRawMagics_.insert(incast->GetRawMagic());
    }

    auto placeholderStmt = std::make_shared<ir::TensorOpStmt>(
        std::vector<ir::VarPtr>{}, nullptr, "CALL",
        std::vector<ir::ExprPtr>{}, std::vector<ir::VarPtr>{},
        std::vector<std::pair<std::string, std::any>>{{"placeholder_funcname", pathFunc->GetMagicName()}},
        seq->span_);

    return placeholderStmt;
}

ir::StmtPtr RootFunctionBuilder::FinalizePathFunc(const ir::StmtPtr& placeholder)
{
    auto funcname = GetPlaceholderFuncname(placeholder);
    auto pathFunc = program_.GetFunctionByMagicName(funcname);
    FE_ASSERT(FeError::NOT_EXIST, pathFunc) << funcname << " is not in functionmap!";

    auto allOutputs = CollectAllOutputs(*pathFunc);
    ComputeOutcast(*pathFunc, allOutputs);

    auto scope = std::make_shared<TensorSlotScope>(pathFunc);
    pathFunc->SetSlotScope(scope);
    program_.GetTensorSlotManager()->scopeList.push_back(scope);

    auto originalIncasts = pathFunc->GetOriginIncast();
    auto originalOutcasts = pathFunc->GetOriginOutcast();

    pathFunc->MakeIncasts(scope);
    pathFunc->MakeOutcasts(scope);

    for (auto& incast : pathFunc->GetIncast()) {
        pathFunc->params_.push_back(std::static_pointer_cast<const ir::Var>(incast));
    }
    for (auto& outcast : pathFunc->GetOutcast()) {
        pathFunc->params_.push_back(std::static_pointer_cast<const ir::Var>(outcast));
    }

    BuildPathFuncSlotScope(pathFunc, scope, originalIncasts, originalOutcasts);

    std::vector<ir::StmtPtr> bodyStmts;
    for (auto& op : pathFunc->Operations(false)) {
        bodyStmts.push_back(std::static_pointer_cast<const ir::Stmt>(op.shared_from_this()));
    }
    pathFunc->body_ = std::make_shared<ir::SeqStmts>(std::move(bodyStmts), placeholder->span_);

    pathFunc->ComputeHash();

    if (config::GetPassDefaultConfig(KEY_PRINT_GRAPH, false)) {
        std::string dumpDir = config::LogTensorGraphFolder();
        std::string baseName = pathFunc->GetRawName();
        pathFunc->DumpJsonFile(dumpDir + "/" + baseName + ".json");
        pathFunc->DumpFile(dumpDir + "/" + baseName + ".tifwkgr");
    }

    program_.GetFunctionCache().Insert(pathFunc->GetFunctionHash(), *pathFunc);

    auto& callOperation = dynFunc_->AddRawOperation(
        Opcode::OP_CALL, originalIncasts, originalOutcasts, placeholder->span_);
    callOperation.SetOpAttribute(pathFunc->CreateCallOpAttribute({}, {}));
    dynFunc_->AppendCalleeMagicName(pathFunc->GetMagicName());
    callOperation.attrs_.emplace_back("callee", pathFunc->GetMagicName());

    return std::static_pointer_cast<const ir::Stmt>(callOperation.shared_from_this());
}

ir::StmtPtr RootFunctionBuilder::TransformStmts(ir::StmtPtr stmt, const std::string& loopVarName)
{
    switch (stmt->GetKind()) {
        case ir::ObjectKind::SeqStmts: {
            auto seq = ir::SeqStmts::AsMut(stmt);
            if (IsPureTensorOpSeq(seq)) {
                auto newStmts = CreatePathFuncAndPlaceholder(seq, loopVarName);
                return std::make_shared<ir::SeqStmts>(std::vector<ir::StmtPtr>{newStmts}, seq->span_);
            }
            auto segments = SplitIntoTensorOpSegments(seq);
            std::vector<ir::StmtPtr> newStmts;
            for (auto& segment : segments) {
                if (!segment.empty() && segment[0]->GetKind() == ir::ObjectKind::TensorOpStmt) {
                    auto segSeq = std::make_shared<ir::SeqStmts>(segment, seq->span_);
                    newStmts.push_back(CreatePathFuncAndPlaceholder(segSeq, loopVarName));
                } else if (!segment.empty()) {
                    newStmts.push_back(TransformStmts(segment[0], loopVarName));
                }
            }
            return std::make_shared<ir::SeqStmts>(newStmts, seq->span_);
        }
        case ir::ObjectKind::ForStmt: {
            auto forStmt = std::static_pointer_cast<const ir::ForStmt>(stmt);
            auto currentLoopVarName = IRContext::Get().GetOriginName(forStmt->loopVar_);
            auto transformedBody = TransformStmts(forStmt->body_, currentLoopVarName);
            return std::make_shared<ir::ForStmt>(
                forStmt->loopVar_, forStmt->start_, forStmt->stop_, forStmt->step_, forStmt->iterArgs_, transformedBody,
                forStmt->returnVars_, forStmt->span_);
        }
        case ir::ObjectKind::IfStmt: {
            auto ifStmt = std::static_pointer_cast<const ir::IfStmt>(stmt);
            auto transformedThen = TransformStmts(ifStmt->thenBody_, loopVarName);
            std::optional<ir::StmtPtr> transformedElse;
            if (ifStmt->elseBody_) {
                transformedElse = TransformStmts(ifStmt->elseBody_.value(), loopVarName);
            }
            return std::make_shared<ir::IfStmt>(
                ifStmt->condition_, transformedThen, transformedElse, ifStmt->returnVars_, ifStmt->span_);
        }
        default:
            return stmt;
    }
}

void RootFunctionBuilder::ReplacePlaceholders(ir::StmtPtr stmt)
{
    if (!stmt) {
        return;
    }

    switch (stmt->GetKind()) {
        case ir::ObjectKind::SeqStmts: {
            auto seq = ir::SeqStmts::AsMut(stmt);
            for (auto& child : seq->stmts_) {
                if (IsPlaceholderCallStmt(child)) {
                    child = FinalizePathFunc(child);
                } else {
                    ReplacePlaceholders(child);
                }
            }
            break;
        }
        case ir::ObjectKind::ForStmt: {
            auto forStmt = std::static_pointer_cast<const ir::ForStmt>(stmt);
            ReplacePlaceholders(forStmt->body_);
            break;
        }
        case ir::ObjectKind::IfStmt: {
            auto ifStmt = std::static_pointer_cast<const ir::IfStmt>(stmt);
            ReplacePlaceholders(ifStmt->thenBody_);
            if (ifStmt->elseBody_) {
                ReplacePlaceholders(ifStmt->elseBody_.value());
            }
            break;
        }
        default:
            break;
    }
}

ir::StmtPtr RootFunctionBuilder::TransformBody(ir::StmtPtr stmt)
{
    if (!stmt) {
        return nullptr;
    }

    paramRawMagics_.clear();
    for (auto& incast : dynFunc_->GetOriginIncast()) {
        paramRawMagics_.insert(incast->GetRawMagic());
    }

    consumedRawMagics_.clear();

    auto irTree = TransformStmts(stmt, "");
    ReplacePlaceholders(irTree);

    return irTree;
}

} // namespace npu::tile_fwk
