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
    : program_(Program::GetInstance()), parentFunc_(parentFunc)
{}

std::shared_ptr<Function> RootFunctionBuilder::Build(const ir::FunctionPtr& irFunc)
{
    InitDynFunc(irFunc);
    auto stmtsWithCall = TransformBody(irFunc->body_);
    dynFunc_->body_ = ir::SeqStmts::Wrap(stmtsWithCall, irFunc->span_);
    FinalizeDynFunc(irFunc);
    DumpFunctionGraph(dynFunc_.get());
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
    ir::Span span = irFunc->span_;
    dynFunc_->SetSpan(span);
    dynFunc_->SetDyndevAttribute(std::make_shared<DyndevFunctionAttribute>());
    program_.SetCurrentDynamicFunction(dynFunc_.get());
}

void RootFunctionBuilder::FinalizeDynFunc(const ir::FunctionPtr& irFunc)
{
    dynFunc_->name_ = irFunc->name_;
    dynFunc_->funcType_ = ir::FunctionType::ORCHESTRATION;

    auto dynScope = std::make_shared<TensorSlotScope>(dynFunc_.get());
    dynFunc_->SetSlotScope(dynScope);
    program_.GetTensorSlotManager()->scopeList.push_back(dynScope);

    auto operationList = dynFunc_->Operations(false).DuplicatedOpList();
    dynFunc_->FillOriginInOutCast(operationList);
    dynFunc_->SetCallOpSlot();

    for (auto& incast : dynFunc_->GetOriginIncast()) {
        dynFunc_->AppendIncast(incast, 0, 0);
    }
    for (auto& outcast : dynFunc_->GetOriginOutcast()) {
        dynFunc_->AppendOutcast(outcast, 0, 0);
    }

    for (auto& param : logicalParams_) {
        dynFunc_->params_.push_back(std::static_pointer_cast<const ir::Var>(param));
    }
    dynFunc_->ComputeHash();
    BuildDynSlotScope(); // CleanRedundantOutCast 会过滤 outCasts_ 中的中间输出
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

ir::StmtPtr RootFunctionBuilder::ProcessTensorOp(std::shared_ptr<Function> pathFunc, const ir::StmtPtr& stmt,
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

    ir::StmtPtr opStmt = RebuildTensorOpStmt(tensorOpStmt, tensorOpStmt->result_, tensorOpStmt->result_token_,
                                             tensorOpStmt->args_, tensorOpStmt->tokens_, tensorOpStmt->span_,
                                             pathFunc.get());
    if (tensorOpStmt->result_token_) {
        pathFunc->GetVarDependency().AddProducer(tensorOpStmt->result_token_, opStmt);
    }
    for (auto& token : tensorOpStmt->tokens_) {
        pathFunc->GetVarDependency().AddConsumer(token, opStmt);
    }
    return opStmt;
}

void RootFunctionBuilder::ComputeIncast(Function& pathFunc,
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

void RootFunctionBuilder::ComputeOutcast(Function& pathFunc,
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

std::shared_ptr<Function> RootFunctionBuilder::CreateHiddenFunc(const ir::SeqStmtsPtr& seq,
                                                                const std::string& loopVarName)
{
    auto pathFuncId = loopNameCounters_[loopVarName]++;
    std::string pathSuffix = loopVarName.empty() ? std::to_string(pathFuncId) :
                                                   loopVarName + "_PATH" + std::to_string(pathFuncId);
    auto hiddenRawName = dynFunc_->GetRawName() + "_" + pathSuffix + "_hiddenfunc";
    auto hiddenMagicName = hiddenRawName + "_" + std::to_string(IdGen<IdType::FUNCTION>::Inst().NewId());
    auto hiddenFunc = std::make_shared<Function>(program_, hiddenMagicName, hiddenRawName, dynFunc_.get());
    hiddenFunc->SetFunctionType(FunctionType::DYNAMIC_LOOP_PATH);
    hiddenFunc->SetGraphType(GraphType::TENSOR_GRAPH);
    hiddenFunc->SetUnderDynamicFunction(true);
    program_.InsertFuncToFunctionMap(hiddenMagicName, hiddenFunc);

    std::unordered_set<std::shared_ptr<LogicalTensor>> definedOutputs;
    std::unordered_set<std::shared_ptr<LogicalTensor>> allInputs;
    std::unordered_set<std::shared_ptr<LogicalTensor>> allOutputs;
    for (auto& stmt : seq->stmts_) {
        (void)ProcessTensorOp(hiddenFunc, stmt, allInputs, allOutputs, definedOutputs);
    }
    ComputeIncast(*hiddenFunc, allInputs, definedOutputs);
    hiddenFunc->name_ = hiddenMagicName;
    hiddenFunc->funcType_ = ir::FunctionType::IN_CORE;
    return hiddenFunc;
}

int RootFunctionBuilder::FindOrCreateSlot(const std::shared_ptr<LogicalTensor>& lt,
                                          const std::shared_ptr<TensorSlotManager>& slotManager, Function* func,
                                          bool isAssembleOut)
{
    int rawMagic = lt->tensor->GetRawMagic();
    int existingSlot = slotManager->LookupSlotIndexByRawMagic(rawMagic);
    if (existingSlot != -1) {
        if (isAssembleOut) {
            auto& slotTensors = func->GetSlotTensors();
            auto it = slotTensors.find(existingSlot);
            if (it != slotTensors.end()) {
                slotManager->TensorWrite(*it->second, SlotProperty::ASSEMBLE_DST);
            }
        }
        return existingSlot;
    }

    auto newTensor = std::make_unique<Tensor>(lt);
    if (isAssembleOut) {
        slotManager->TensorWrite(*newTensor, SlotProperty::ASSEMBLE_DST);
    }

    int tensorSlot = slotManager->LookupSlotIndexByRawMagic(rawMagic);
    func->GetSlotTensors()[tensorSlot] = std::move(newTensor);
    return tensorSlot;
}

void RootFunctionBuilder::BuildPathFuncSlotScope(Function* pathFunc, const std::shared_ptr<TensorSlotScope>& scope,
                                                 const LogicalTensors& originalIncasts,
                                                 const LogicalTensors& originalOutcasts)
{
    auto slotManager = program_.GetTensorSlotManager();

    scope->ioslot.incastSlot.resize(originalIncasts.size());
    for (size_t idx = 0; idx < originalIncasts.size(); idx++) {
        int slotIndex = FindOrCreateSlot(originalIncasts[idx], slotManager, pathFunc);
        scope->ioslot.incastSlot[idx] = {slotIndex};
    }

    scope->ioslot.outcastSlot.resize(originalOutcasts.size());
    for (size_t idx = 0; idx < originalOutcasts.size(); idx++) {
        int slotIndex = FindOrCreateSlot(originalOutcasts[idx], slotManager, pathFunc);
        scope->ioslot.outcastSlot[idx] = {slotIndex};
    }

    std::unordered_set<int> addedSlots;
    for (auto& op : pathFunc->Operations()) {
        if ((op.GetOpcode() == Opcode::OP_ASSEMBLE && op.HasAttr("dassemble")) ||
            op.GetOpcode() == Opcode::OP_ASSEMBLE_SSA || op.GetOpcode() == Opcode::OP_ATOMIC_RMW) {
            for (auto& oOperand : op.GetOOperands()) {
                int slotIndex = FindOrCreateSlot(oOperand, slotManager, pathFunc, true);
                if (paramRawMagics_.count(oOperand->GetRawMagic()) != 0) {
                    continue;
                }
                if (addedSlots.insert(slotIndex).second) {
                    scope->constructAssembleSlotList.push_back(slotIndex);
                }
            }
        }
    }
}

void RootFunctionBuilder::BuildDynSlotScope()
{
    auto attr = dynFunc_->GetDyndevAttribute();
    FE_ASSERT(FeError::INVALID_PTR, attr != nullptr) << "DyndevAttribute is nullptr";

    auto slotManager = program_.GetTensorSlotManager();
    auto& dynScope = dynFunc_->GetSlotScope();

    dynScope->ioslot.incastSlot = dynScope->originalIocastsSlot.incastSlot;
    dynScope->ioslot.outcastSlot = dynScope->originalIocastsSlot.outcastSlot;

    attr->startArgsInputLogicalTensorList.resize(logicalParams_.size());
    for (size_t idx = 0; idx < logicalParams_.size(); idx++) {
        FindOrCreateSlot(logicalParams_[idx], slotManager, dynFunc_.get());
        const Tensor* tensor = slotManager->LookupTensorByRawMagic(logicalParams_[idx]->tensor->GetRawMagic());
        attr->startArgsInputTensorList.emplace_back(*tensor);
        attr->startArgsInputLogicalTensorList[idx] = attr->startArgsInputTensorList.back().get().GetStorage(false);
        slotManager->MarkInput(*tensor);
    }

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

ir::StmtPtr RootFunctionBuilder::CreatePathFuncAndPlaceholder(const ir::SeqStmtsPtr& seq,
                                                              const std::string& loopVarName)
{
    auto pathFunc = CreateHiddenFunc(seq, loopVarName);

    for (auto& incast : pathFunc->GetOriginIncast()) {
        consumedRawMagics_.insert(incast->GetRawMagic());
    }

    auto placeholderStmt = std::make_shared<ir::TensorOpStmt>(
        std::vector<ir::VarPtr>{}, nullptr, "CALL", std::vector<ir::ExprPtr>{}, std::vector<ir::VarPtr>{},
        std::vector<std::pair<std::string, std::any>>{{"placeholder_funcname", pathFunc->GetMagicName()}}, seq->span_);

    return placeholderStmt;
}

void RootFunctionBuilder::CreateAndFinalizePathFunc(Function* pathFunc, Function* hiddenFunc,
                                                    const LogicalTensors& hiddenInArgs,
                                                    const LogicalTensors& hiddenOutArgs, const ir::StmtPtr& placeholder)
{
    // 1. pathFunc 中创建 OP_CALL 指向 hiddenFunc
    auto& callOp = pathFunc->AddRawOperation(Opcode::OP_CALL, hiddenInArgs, hiddenOutArgs, placeholder->span_);
    callOp.SetOpAttribute(hiddenFunc->CreateCallOpAttribute({}, {}));
    pathFunc->AppendCalleeMagicName(hiddenFunc->GetMagicName());
    callOp.attrs_.emplace_back("callee", hiddenFunc->GetMagicName());

    // 2. pathFunc 的 incast/outcast
    for (auto& incast : hiddenInArgs)
        pathFunc->AddOriginIncast(incast);
    for (auto& outcast : hiddenOutArgs)
        pathFunc->AddOriginOutcast(outcast);

    // 3. pathFunc 的 scope + MakeIncasts/MakeOutcasts
    auto pathScope = std::make_shared<TensorSlotScope>(pathFunc);
    pathFunc->SetSlotScope(pathScope);
    program_.GetTensorSlotManager()->scopeList.push_back(pathScope);

    BuildPathFuncSlotScope(pathFunc, pathScope, pathFunc->GetOriginIncast(), pathFunc->GetOriginOutcast());

    LogicalTensors pathInArgs = pathFunc->MakeIncasts(pathScope);
    LogicalTensors pathOutArgs = pathFunc->MakeOutcasts(pathScope);
    for (size_t idx = 0; idx < pathFunc->GetOutcast().size(); idx++) {
        if (pathScope->partialUpdateOutcastDict.count(pathFunc->GetOutcast()[idx])) {
            pathScope->ioslot.partialUpdateOutcastList.push_back(idx);
        }
    }

    // 4. Add valueDepend of hiddenFunc to DynamicFunction
    auto currDynFunc = program_.GetCurrentDynamicFunction();
    FE_ASSERT(FeError::INVALID_PTR, currDynFunc != nullptr) << "CurrentDynamicFunction is nullptr";
    auto dynAttr = currDynFunc->GetDyndevAttribute();
    FE_ASSERT(FeError::INVALID_PTR, dynAttr != nullptr) << "DyndevAttribute is nullptr";
    auto iodescDict = hiddenFunc->GetTensorDataForTensorGraph();
    hiddenFunc->GetTensorDataRefreshIO(iodescDict);
    dynAttr->valueDependDescDict[hiddenFunc] = hiddenFunc->LookupValueDepend();

    for (auto& incast : pathFunc->GetIncast())
        pathFunc->params_.push_back(std::static_pointer_cast<const ir::Var>(incast));
    for (auto& outcast : pathFunc->GetOutcast())
        pathFunc->params_.push_back(std::static_pointer_cast<const ir::Var>(outcast));

    // 5. pathFunc body_ 重建
    std::vector<ir::StmtPtr> pathBodyStmts;
    for (auto& op : pathFunc->Operations(false))
        pathBodyStmts.push_back(std::static_pointer_cast<const ir::Stmt>(op.shared_from_this()));
    pathFunc->body_ = std::make_shared<ir::SeqStmts>(std::move(pathBodyStmts), placeholder->span_);
    pathFunc->ComputeHash();
    DumpFunctionGraph(pathFunc);
    program_.GetFunctionCache().Insert(pathFunc->GetFunctionHash(), *pathFunc);
}

ir::StmtPtr RootFunctionBuilder::FinalizePathFunc(const ir::StmtPtr& placeholder)
{
    auto hiddenFuncName = GetPlaceholderFuncname(placeholder);
    auto hiddenFunc = program_.GetFunctionByMagicName(hiddenFuncName);
    FE_ASSERT(FeError::NOT_EXIST, hiddenFunc) << hiddenFuncName << " is not in functionmap!";

    // 1. 创建 pathFunc（在 SetParent 之前）
    auto pathRawName = hiddenFunc->GetRawName().substr(0, hiddenFunc->GetRawName().find("_hiddenfunc"));
    auto pathMagicName = pathRawName + "_" + std::to_string(IdGen<IdType::FUNCTION>::Inst().NewId());
    auto pathFunc = std::make_shared<Function>(program_, pathMagicName, pathRawName, dynFunc_.get());
    pathFunc->SetFunctionType(FunctionType::DYNAMIC_LOOP_PATH);
    pathFunc->SetGraphType(GraphType::TENSOR_GRAPH);
    pathFunc->SetUnderDynamicFunction(true);
    program_.InsertFuncToFunctionMap(pathMagicName, pathFunc);
    pathFunc->name_ = pathMagicName;
    pathFunc->funcType_ = ir::FunctionType::IN_CORE;

    // 2. hiddenFunc 设为 hidden，parent 改为 pathFunc
    hiddenFunc->SetHiddenFunction(true);
    hiddenFunc->SetParent(pathFunc.get());

    // 3. hiddenFunc 处理（MakeIncasts 的 Parent() = pathFunc）
    auto allOutputs = CollectAllOutputs(*hiddenFunc);
    ComputeOutcast(*hiddenFunc, allOutputs);
    auto hiddenScope = std::make_shared<TensorSlotScope>(hiddenFunc);
    hiddenFunc->SetSlotScope(hiddenScope);
    program_.GetTensorSlotManager()->scopeList.push_back(hiddenScope);

    auto originalIncasts = hiddenFunc->GetOriginIncast();
    auto originalOutcasts = hiddenFunc->GetOriginOutcast();

    BuildPathFuncSlotScope(hiddenFunc, hiddenScope, originalIncasts, originalOutcasts);

    auto hiddenInArgs = hiddenFunc->MakeIncasts(hiddenScope);
    auto hiddenOutArgs = hiddenFunc->MakeOutcasts(hiddenScope);
    for (size_t idx = 0; idx < hiddenFunc->GetOutcast().size(); idx++) {
        if (hiddenScope->partialUpdateOutcastDict.count(hiddenFunc->GetOutcast()[idx])) {
            hiddenScope->ioslot.partialUpdateOutcastList.push_back(idx);
        }
    }

    for (auto& incast : hiddenFunc->GetIncast())
        hiddenFunc->params_.push_back(std::static_pointer_cast<const ir::Var>(incast));
    for (auto& outcast : hiddenFunc->GetOutcast())
        hiddenFunc->params_.push_back(std::static_pointer_cast<const ir::Var>(outcast));

    std::vector<ir::StmtPtr> hiddenBodyStmts;
    for (auto& op : hiddenFunc->Operations(false))
        hiddenBodyStmts.push_back(std::static_pointer_cast<const ir::Stmt>(op.shared_from_this()));
    hiddenFunc->body_ = std::make_shared<ir::SeqStmts>(std::move(hiddenBodyStmts), placeholder->span_);
    hiddenFunc->ComputeHash();
    program_.GetFunctionCache().Insert(hiddenFunc->GetFunctionHash(), *hiddenFunc);
    DumpFunctionGraph(hiddenFunc);

    // 4. pathFunc 处理（独立函数）
    CreateAndFinalizePathFunc(pathFunc.get(), hiddenFunc, hiddenInArgs, hiddenOutArgs, placeholder);
    pathFunc->GetSlotScope()->constructAssembleSlotList = hiddenScope->constructAssembleSlotList;

    // 5. dynFunc 的 CALL op
    auto& callOperation = dynFunc_->AddRawOperation(Opcode::OP_CALL, originalIncasts, originalOutcasts,
                                                    placeholder->span_);
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
            auto config = forStmt->GetAttr<std::shared_ptr<ConfigScope>>("_config_scope");
            ConfigManagerNg::ScopedRestore scoped(config);
            auto currentLoopVarName = IRContext::Get().GetOriginName(forStmt->loopVar_);
            auto stepConst = ir::As<ir::ConstInt>(forStmt->step_);
            int64_t stepValue = stepConst ? stepConst->value_ : 1;
            currentLoopVarName += "_Unroll" + std::to_string(stepValue);
            auto transformedBody = TransformStmts(forStmt->body_, currentLoopVarName);
            return std::make_shared<ir::ForStmt>(forStmt->loopVar_, forStmt->start_, forStmt->stop_, forStmt->step_,
                                                 forStmt->iterArgs_, transformedBody, forStmt->returnVars_,
                                                 forStmt->span_, forStmt->attrs_);
        }
        case ir::ObjectKind::IfStmt: {
            auto ifStmt = std::static_pointer_cast<const ir::IfStmt>(stmt);
            auto transformedThen = TransformStmts(ifStmt->thenBody_, loopVarName);
            std::optional<ir::StmtPtr> transformedElse;
            if (ifStmt->elseBody_) {
                transformedElse = TransformStmts(ifStmt->elseBody_.value(), loopVarName);
            }
            return std::make_shared<ir::IfStmt>(ifStmt->condition_, transformedThen, transformedElse,
                                                ifStmt->returnVars_, ifStmt->span_);
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
    for (auto& param : logicalParams_) {
        paramRawMagics_.insert(param->GetRawMagic());
    }

    consumedRawMagics_.clear();

    auto irTree = TransformStmts(stmt, "");
    ReplacePlaceholders(irTree);

    return irTree;
}

void RootFunctionBuilder::DumpFunctionGraph(Function* func)
{
    if (!config::GetPassDefaultConfig(KEY_PRINT_GRAPH, false)) {
        return;
    }
    std::string dumpDir = config::LogTensorGraphFolder();
    std::string baseName = func->GetRawName();
    func->DumpJsonFile(dumpDir + "/" + baseName + ".json");
    func->DumpFile(dumpDir + "/" + baseName + ".tifwkgr");
}

} // namespace npu::tile_fwk
