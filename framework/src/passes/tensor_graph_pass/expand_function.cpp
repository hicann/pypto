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
 * \file expand_function.cpp
 * \brief
 */

#include "passes/tensor_graph_pass/expand_function.h"
#include <map>
#include "interface/operation/opcode.h"
#include "interface/function/function.h"
#include "interface/tensor/raw_tensor.h"
#include "interface/tensor/logical_tensor.h"
#include "interface/utils/source_location.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/program/program.h"
#include "interface/operation/operation_impl.h"
#include "interface/configs/config_manager.h"
#include "passes/pass_check/expand_function_checker.h"
#include "passes/statistics/tensor_and_tile_graph_statistic.h"
#include "passes/pass_log/pass_log.h"

#define MODULE_NAME "ExpandFunction"

using namespace npu::tile_fwk;

namespace npu::tile_fwk {
namespace {
bool CheckAssembleNeedCopy(Function &function, const std::shared_ptr<Operation> &op) {
    if (op->GetOpcode() != Opcode::OP_ASSEMBLE) {
        return false;
    }
    if (op->oOperand.empty() || op->iOperand.empty()) {
        return false;
    }
    std::shared_ptr<AssembleOpAttribute> assembleOpAttribute = std::dynamic_pointer_cast<AssembleOpAttribute>(op->GetOpAttribute());
    if (assembleOpAttribute == nullptr) {
        return false;
    }
    if (assembleOpAttribute->GetToDynOffset().empty()) {
        return false;
    }
    auto producers = function.FindProducers(*op);
    if (producers.size() != 1) {
        return false;
    }
    if (((*producers.begin()) == nullptr) || ((*producers.begin())->GetOpcode() != Opcode::OP_RESHAPE)) {
        return false;
    }
    for (size_t i = 1; i < op->oOperand[0]->shape.size(); i++) {
        if (op->oOperand[0]->shape[i] != op->iOperand[0]->shape[i]) {
            APASS_LOG_INFO_F(Elements::Operation, "Assemble op [%d] need to check expansion.",  op->GetOpMagic());
            return true;
        }
    }
    return false;
}

Status UpdateIOOperand(const std::vector<OperationPtr> &tensorOperations) {
    for (auto &op : tensorOperations) {
        // clear consumers and producers
        for (auto &iOperand : op->GetIOperands()) {
            if (iOperand == nullptr) {
                APASS_LOG_ERROR_F(Elements::Operation, "Op:%s[%d] input is null.%s",  op->GetOpcodeStr().c_str(), op->GetOpMagic(), GetFormatBacktrace(*op).c_str());
                return FAILED;
            }
            iOperand->GetConsumers().clear();
            iOperand->GetProducers().clear();
        }
        for (auto &oOperand : op->GetOOperands()) {
            if (oOperand == nullptr) {
                APASS_LOG_ERROR_F(Elements::Operation, "Op:%s[%d] output is null.%s",  op->GetOpcodeStr().c_str(), op->GetOpMagic(), GetFormatBacktrace(*op).c_str());
                return FAILED;
            }
            oOperand->GetConsumers().clear();
            oOperand->GetProducers().clear();
        }
    }
    return SUCCESS;
}

bool NotNeedExpand(Opcode opcode, bool needCopy) {
    return opcode == Opcode::OP_VIEW || (opcode == Opcode::OP_ASSEMBLE && !needCopy) || opcode == Opcode::OP_PAD ||
           opcode == Opcode::OP_NOP;
}

void ProcessForNotExpandOp(Function &function, Operation &op) {
    auto &newOp = function.AddOperation(op.GetOpcode(), op.GetIOperands(), op.GetOOperands());
    newOp.SetOpAttribute(op.GetOpAttribute());
    newOp.CopyAttrFrom(op, OP_EMUOP_PREFIX);
    if (op.HasAttribute(OpAttributeKey::inplaceIdx)) {
        newOp.SetAttribute(OpAttributeKey::inplaceIdx, op.GetIntAttribute(OpAttributeKey::inplaceIdx));
    }
}
}

Status ExpandFunction::PreCheck(Function &function) {
    ExpandFunctionChecker checker;
    return checker.DoPreCheck(function);
}

Status ExpandFunction::PostCheck(Function &function) {
    ExpandFunctionChecker checker;
    return checker.DoPostCheck(function);
}

Status ExpandFunction::RunOnFunction(Function &function) {
    APASS_LOG_INFO_F(Elements::Function, "Start ExpandFunction function [%s].", function.GetRawName().c_str());
    std::ostringstream oss;
    bool verifyResult = true;
    for (auto &op : function.Operations(false)) {
        auto verifyOperationEntry = OpcodeManager::Inst().GetVerifyOperationEntry(op.GetOpcode());
        if (verifyOperationEntry) {
            verifyResult = verifyResult && verifyOperationEntry(function, op, oss);
        }
    }
    if (!verifyResult) {
        APASS_LOG_ERROR_F(Elements::Function, "FUnction[%s] ExpandFunction failed: %s", function.GetRawName().c_str(), oss.str().c_str());
        return FAILED;
    }
    if (Expandfunction(function) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Function, "Function[%s] ExpandFunction failed.", function.GetRawName().c_str());
        return FAILED;
    }
    APASS_LOG_INFO_F(Elements::Function, "End ExpandFunction function [%s].", function.GetRawName().c_str());
    return SUCCESS;
}

Status ExpandFunction::Expandfunction(Function &function) const {
    if (!function.IsGraphType(GraphType::TENSOR_GRAPH)) {
        APASS_LOG_INFO_F(Elements::Function, "Function %s is not static tensor graph, skip expanding.", function.GetRawName().c_str());
        return SUCCESS;
    }
    function.expandFunctionAccelerate = true;
    function.SetGraphType(GraphType::TILE_GRAPH);

    std::vector<OperationPtr> tensorOperations;
    auto operationViewer = function.Operations();
    for (size_t i = 0; i < operationViewer.size(); i++) {
        tensorOperations.emplace_back(operationViewer.operations_[i]);
    }

    function.ResetOperations();
    if (UpdateIOOperand(tensorOperations) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "UpdateIOOperand failed.");
        return FAILED;
    }

    for (auto &op : tensorOperations) {
        if (op == nullptr) {
            APASS_LOG_ERROR_F(Elements::Operation, "Encountered null operation in function.");
            return FAILED;
        }
        if (op->GetOpcode() == Opcode::OP_PRINT) {
            continue;
        }
        SourceLocation::SetLocation(op->GetLocation());
        bool needCopy = CheckAssembleNeedCopy(function, op);
        APASS_LOG_DEBUG_F(Elements::Operation, "Op %s[%d] is needCopy: %d", op->GetOpcodeStr().c_str(), op->GetOpMagic(), needCopy);
        if (NotNeedExpand(op->GetOpcode(), needCopy)) {
            ProcessForNotExpandOp(function, *op);
            continue;
        }
        config::SetSemanticLabel(op->GetSemanticLabel());
        size_t opListPreSize = function.Operations(false).size();
        config::SetPassOption(SG_SET_SCOPE, op->GetScopeId());
        ExpandOperationInto(function, op->GetTileShape(), op->GetOpcode(), op->GetIOperands(), op->GetOOperands(), *op);
        config::SetPassOption(SG_SET_SCOPE, -1);
        auto opListPost = function.Operations(false);
        if (op->GetOpcode() == Opcode::OP_ADDS) {
            for (size_t i = opListPreSize; i < opListPost.size(); i++) {
                auto &newOp = opListPost[i];
                newOp.CopyAttrFrom(*op, OP_EMUOP_PREFIX);
            }
        }
        SourceLocation::ClearLocation();
    }
    function.expandFunctionAccelerate = false;
    return SUCCESS;
}

void ExpandFunction::DoHealthCheckBefore(Function &function, const std::string &folderPath) {
    APASS_LOG_INFO_F(Elements::Operation, "Before ExpandFunction, Health Report: TensorGraph START");
    std::string fileName = GetDumpFilePrefix(function, true);
    HealthCheckTensorGraph(function, folderPath, fileName);
    APASS_LOG_INFO_F(Elements::Operation, "Before ExpandFunction, Health Report: TensorGraph END");
}
} // namespace npu::tile_fwk
