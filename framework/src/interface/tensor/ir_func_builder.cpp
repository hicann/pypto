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
#include "interface/operation/opcode.h"
#include "interface/program/program.h"
#include "interface/tensor/tensor_slot.h"
#include "tilefwk/tensor.h"
#include "interface/utils/id_gen.h"
#include "interface/configs/config_manager.h"
#include "interface/tensor/irbuilder.h"

#include "ir/expr.h"
#include "ir/kind_traits.h"
#include "ir/type.h"

using namespace pypto;

namespace npu::tile_fwk {

bool IsPureTensorOpSeq(const ir::SeqStmtsPtr& seq)
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

using StmtSegments = std::vector<std::vector<ir::StmtPtr>>;

static StmtSegments SplitIntoTensorOpSegments(const ir::SeqStmtsPtr& seq)
{
    StmtSegments segments;
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

static void CopyOpAttributes(Operation& operation, const std::vector<std::pair<std::string, std::any>>& attrs)
{
    for (auto& [key, value] : attrs) {
        if (value.type() == typeid(int64_t)) {
            operation.SetAttribute(key, std::any_cast<int64_t>(value));
        } else if (value.type() == typeid(int)) {
            operation.SetAttribute(key, static_cast<int64_t>(std::any_cast<int>(value)));
        } else if (value.type() == typeid(bool)) {
            operation.SetAttribute(key, std::any_cast<bool>(value));
        } else if (value.type() == typeid(std::string)) {
            operation.SetAttribute(key, std::any_cast<std::string>(value));
        } else if (value.type() == typeid(std::vector<int64_t>)) {
            operation.SetAttribute(key, std::any_cast<std::vector<int64_t>>(value));
        } else if (value.type() == typeid(tile_fwk::DataType)) {
            operation.SetAttribute(key, std::any_cast<tile_fwk::DataType>(value));
        }
    }
}

static void ProcessTensorOpIntoPathFunc(
    std::shared_ptr<Function> pathFunc, const ir::StmtPtr& stmt,
    std::unordered_set<std::shared_ptr<LogicalTensor>>& allInputs,
    std::unordered_set<std::shared_ptr<LogicalTensor>>& allOutputs,
    std::unordered_set<std::shared_ptr<LogicalTensor>>& definedOutputs)
{
    if (stmt->GetKind() != ir::ObjectKind::TensorOpStmt) {
        return;
    }
    auto tensorOpStmt = std::static_pointer_cast<const ir::TensorOpStmt>(stmt);
    auto opcode = FindOpcode(tensorOpStmt->opcode_);

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

    auto& operation = pathFunc->AddRawOperation(opcode, iOperands, oOperands, tensorOpStmt->span_);
    CopyOpAttributes(operation, tensorOpStmt->attrs_);

    ir::StmtPtr opStmt = std::static_pointer_cast<const ir::Stmt>(operation.shared_from_this());
    if (tensorOpStmt->result_token_) {
        pathFunc->GetVarDependency().AddProducer(tensorOpStmt->result_token_, opStmt);
    }
    for (auto& token : tensorOpStmt->tokens_) {
        pathFunc->GetVarDependency().AddConsumer(token, opStmt);
    }
}

static void ComputePathFuncInOut(
    Function& pathFunc, const std::unordered_set<std::shared_ptr<LogicalTensor>>& allInputs,
    const std::unordered_set<std::shared_ptr<LogicalTensor>>& allOutputs,
    const std::unordered_set<std::shared_ptr<LogicalTensor>>& definedOutputs,
    const std::unordered_set<std::shared_ptr<LogicalTensor>>& downstreamIncastPtrs,
    const std::unordered_set<std::string>& paramNames)
{
    std::unordered_set<std::shared_ptr<LogicalTensor>> incastPtrs;
    for (auto& input : allInputs) {
        if (definedOutputs.find(input) == definedOutputs.end() && incastPtrs.find(input) == incastPtrs.end()) {
            incastPtrs.insert(input);
            pathFunc.AddOriginIncast(input);
        }
    }

    std::unordered_set<std::shared_ptr<LogicalTensor>> outcastPtrs;
    for (auto& output : allOutputs) {
        if (outcastPtrs.find(output) == outcastPtrs.end()) {
            bool neededByDownstream = downstreamIncastPtrs.find(output) != downstreamIncastPtrs.end();
            bool isFuncOutput = paramNames.find(output->name_) != paramNames.end();
            if (neededByDownstream || isFuncOutput) {
                outcastPtrs.insert(output);
                pathFunc.AddOriginOutcast(output);
            }
        }
    }
}

std::shared_ptr<Function> CreatePathFuncFromSeq(
    const ir::SeqStmtsPtr& seq, Function& dynFunc,
    const std::unordered_set<std::shared_ptr<LogicalTensor>>& downstreamIncastPtrs,
    const std::unordered_set<std::string>& paramNames, const std::string& loopVarName)
{
    auto& program = Program::GetInstance();
    auto pathFuncId = IdGen<IdType::FUNCTION>::Inst().NewId();
    std::string pathSuffix =
        loopVarName.empty() ? std::to_string(pathFuncId) : loopVarName + "_" + std::to_string(pathFuncId);
    auto pathMagicName = dynFunc.GetRawName() + "_path_" + pathSuffix;
    auto pathName = dynFunc.GetRawName() + "_path_" + pathSuffix;
    auto pathFunc = std::make_shared<Function>(program, pathMagicName, pathName, &dynFunc);
    pathFunc->SetFunctionType(FunctionType::DYNAMIC_LOOP_PATH);
    pathFunc->SetGraphType(GraphType::TENSOR_GRAPH);
    pathFunc->SetUnderDynamicFunction(true);
    program.InsertFuncToFunctionMap(pathMagicName, pathFunc);

    std::unordered_set<std::shared_ptr<LogicalTensor>> definedOutputs;
    std::unordered_set<std::shared_ptr<LogicalTensor>> allInputs;
    std::unordered_set<std::shared_ptr<LogicalTensor>> allOutputs;
    for (auto& stmt : seq->stmts_) {
        ProcessTensorOpIntoPathFunc(pathFunc, stmt, allInputs, allOutputs, definedOutputs);
    }
    ComputePathFuncInOut(*pathFunc, allInputs, allOutputs, definedOutputs, downstreamIncastPtrs, paramNames);
    return pathFunc;
}

static int FindOrCreateSlotForLogicalTensor(
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

static void BuildPathFuncSlotScope(
    std::shared_ptr<Function> pathFunc, const std::shared_ptr<TensorSlotScope>& scope,
    const LogicalTensors& inArgumentList, const LogicalTensors& outArgumentList)
{
    auto slotManager = Program::GetInstance().GetTensorSlotManager();

    scope->ioslot.incastSlot.resize(pathFunc->GetIncast().size());
    for (size_t idx = 0; idx < pathFunc->GetIncast().size(); idx++) {
        int slotIndex = FindOrCreateSlotForLogicalTensor(inArgumentList[idx], slotManager, pathFunc.get(), true);
        scope->ioslot.incastSlot[idx] = {slotIndex};
    }

    scope->ioslot.outcastSlot.resize(pathFunc->GetOutcast().size());
    for (size_t idx = 0; idx < pathFunc->GetOutcast().size(); idx++) {
        int slotIndex = FindOrCreateSlotForLogicalTensor(outArgumentList[idx], slotManager, pathFunc.get(), false);
        scope->ioslot.outcastSlot[idx] = {slotIndex};
    }

    for (auto& op : pathFunc->Operations()) {
        if (op.GetOpcode() == Opcode::OP_ASSEMBLE || op.GetOpcode() == Opcode::OP_ASSEMBLE_SSA) {
            for (auto& oOperand : op.GetOOperands()) {
                int slotIndex = FindOrCreateSlotForLogicalTensor(oOperand, slotManager, pathFunc.get(), false);
                scope->constructAssembleSlotList.push_back(slotIndex);
            }
        }
    }
}

void BuildDynFuncSlotScope(std::shared_ptr<Function> dynFunc, const LogicalTensors& params)
{
    auto& program = Program::GetInstance();
    auto attr = std::make_shared<DyndevFunctionAttribute>();
    dynFunc->SetDyndevAttribute(attr);

    auto slotManager = program.GetTensorSlotManager();
    auto dynScope = std::make_shared<TensorSlotScope>(dynFunc.get());
    dynFunc->SetSlotScope(dynScope);
    slotManager->scopeList.push_back(dynScope);

    dynScope->ioslot.incastSlot.resize(params.size());
    for (size_t idx = 0; idx < params.size(); idx++) {
        int slotIndex = FindOrCreateSlotForLogicalTensor(params[idx], slotManager, dynFunc.get(), true);
        dynScope->ioslot.incastSlot[idx] = {slotIndex};
        const Tensor* tensor = slotManager->LookupTensorByRawMagic(params[idx]->tensor->GetRawMagic());
        attr->startArgsInputTensorList.emplace_back(*tensor);
    }
    attr->startArgsInputLogicalTensorList.resize(attr->startArgsInputTensorList.size());
    for (size_t k = 0; k < attr->startArgsInputTensorList.size(); k++) {
        attr->startArgsInputLogicalTensorList[k] = attr->startArgsInputTensorList[k].get().GetStorage(false);
    }

    auto calleeList = dynFunc->GetCalleeFunctionList();
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

    dynFunc->CleanRedundantOutCast();
    dynFunc->InferParamDirection();
}

static ir::StmtPtr CreatePathFuncAndCallStmt(
    const ir::SeqStmtsPtr& seq, Function& dynFunc,
    std::unordered_set<std::shared_ptr<LogicalTensor>>& downstreamIncastPtrs,
    const std::unordered_set<std::string>& paramNames, const std::string& loopVarName)
{
    auto pathFunc = CreatePathFuncFromSeq(seq, dynFunc, downstreamIncastPtrs, paramNames, loopVarName);

    auto originIncasts = pathFunc->GetOriginIncast();
    for (auto& incast : originIncasts) {
        downstreamIncastPtrs.insert(incast);
    }

    auto scope = std::make_shared<TensorSlotScope>(pathFunc.get());
    pathFunc->SetSlotScope(scope);
    Program::GetInstance().GetTensorSlotManager()->scopeList.push_back(scope);

    auto inArgumentList = pathFunc->MakeIncasts(scope);
    auto outArgumentList = pathFunc->MakeOutcasts(scope);

    BuildPathFuncSlotScope(pathFunc, scope, inArgumentList, outArgumentList);

    pathFunc->SortOperations();
    pathFunc->ComputeHash();

    if (config::GetPassDefaultConfig(KEY_PRINT_GRAPH, false)) {
        std::string dumpDir = config::LogTensorGraphFolder();
        std::string baseName = pathFunc->GetRawName();
        pathFunc->DumpJsonFile(dumpDir + "/" + baseName + ".json");
        pathFunc->DumpFile(dumpDir + "/" + baseName + ".tifwkgr");
    }

    auto& program = Program::GetInstance();
    program.GetFunctionCache().Insert(pathFunc->GetFunctionHash(), *pathFunc);

    auto& callOperation = dynFunc.AddRawOperation(Opcode::OP_CALL, inArgumentList, outArgumentList, seq->span_);
    callOperation.SetOpAttribute(pathFunc->CreateCallOpAttribute({}, {}));
    dynFunc.AppendCalleeMagicName(pathFunc->GetMagicName());

    ir::StmtPtr callStmt = std::static_pointer_cast<const ir::Stmt>(callOperation.shared_from_this());
    return callStmt;
}

ir::StmtPtr TransformAndBuildStmts(
    ir::StmtPtr stmt, Function& dynFunc, std::unordered_set<std::shared_ptr<LogicalTensor>>& downstreamIncastPtrs,
    const std::unordered_set<std::string>& paramNames, const std::string& loopVarName)
{
    switch (stmt->GetKind()) {
        case ir::ObjectKind::SeqStmts: {
            auto seq = ir::SeqStmts::AsMut(stmt);
            if (IsPureTensorOpSeq(seq)) {
                return CreatePathFuncAndCallStmt(seq, dynFunc, downstreamIncastPtrs, paramNames, loopVarName);
            }
            auto segments = SplitIntoTensorOpSegments(seq);
            std::vector<ir::StmtPtr> newStmts;
            for (auto it = segments.rbegin(); it != segments.rend(); ++it) {
                if (!it->empty() && it->at(0)->GetKind() == ir::ObjectKind::TensorOpStmt) {
                    auto segSeq = std::make_shared<ir::SeqStmts>(*it, seq->span_);
                    newStmts.push_back(
                        CreatePathFuncAndCallStmt(segSeq, dynFunc, downstreamIncastPtrs, paramNames, loopVarName));
                } else if (!it->empty()) {
                    newStmts.push_back(
                        TransformAndBuildStmts(it->at(0), dynFunc, downstreamIncastPtrs, paramNames, loopVarName));
                }
            }
            std::reverse(newStmts.begin(), newStmts.end());
            return std::make_shared<ir::SeqStmts>(newStmts, seq->span_);
        }
        case ir::ObjectKind::ForStmt: {
            auto forStmt = std::static_pointer_cast<const ir::ForStmt>(stmt);
            auto currentLoopVarName = IRContext::Get().GetOriginName(forStmt->loopVar_);
            auto transformedBody =
                TransformAndBuildStmts(forStmt->body_, dynFunc, downstreamIncastPtrs, paramNames, currentLoopVarName);
            return std::make_shared<ir::ForStmt>(
                forStmt->loopVar_, forStmt->start_, forStmt->stop_, forStmt->step_, forStmt->iterArgs_, transformedBody,
                forStmt->returnVars_, forStmt->span_);
        }
        case ir::ObjectKind::IfStmt: {
            auto ifStmt = std::static_pointer_cast<const ir::IfStmt>(stmt);
            auto transformedThen =
                TransformAndBuildStmts(ifStmt->thenBody_, dynFunc, downstreamIncastPtrs, paramNames, loopVarName);
            std::optional<ir::StmtPtr> transformedElse;
            if (ifStmt->elseBody_) {
                transformedElse = TransformAndBuildStmts(
                    ifStmt->elseBody_.value(), dynFunc, downstreamIncastPtrs, paramNames, loopVarName);
            }
            return std::make_shared<ir::IfStmt>(
                ifStmt->condition_, transformedThen, transformedElse, ifStmt->returnVars_, ifStmt->span_);
        }
        default:
            return stmt;
    }
}

ir::StmtPtr CreateFunctionByStmt(ir::StmtPtr stmt, Function& dynFunc, const std::vector<std::string>& externalVarNames)
{
    if (!stmt) {
        return nullptr;
    }
    std::unordered_set<std::shared_ptr<LogicalTensor>> downstreamIncastPtrs;
    std::unordered_set<std::string> paramNames(externalVarNames.begin(), externalVarNames.end());
    return TransformAndBuildStmts(stmt, dynFunc, downstreamIncastPtrs, paramNames, "");
}

} // namespace npu::tile_fwk