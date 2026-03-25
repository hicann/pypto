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
#include "passes/pass_utils/graph_utils.h"

#define MODULE_NAME "ExpandFunction"

using namespace npu::tile_fwk;

namespace npu::tile_fwk {

// 不需要展开的操作码集合
// 这些操作在展开过程中保持原样，不进行 tile-level 展开
const std::unordered_set<Opcode> ExpandFunction::kNotNeedExpandOps = {
    Opcode::OP_VIEW,
    Opcode::OP_ASSEMBLE,
    Opcode::OP_NOP
};

Status ExpandFunction::ClearIOOperand(const std::vector<OperationPtr> &tensorOperations) const {
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

void ExpandFunction::ProcessForNotExpandOp(Function &function, Operation &op) const {
    auto &newOp = function.AddOperation(op.GetOpcode(), op.GetIOperands(), op.GetOOperands());
    newOp.SetOpAttribute(op.GetOpAttribute());
    newOp.CopyAttrFrom(op, OP_EMUOP_PREFIX);
    if (op.HasAttribute(OpAttributeKey::inplaceIdx)) {
        newOp.SetAttribute(OpAttributeKey::inplaceIdx, op.GetIntAttribute(OpAttributeKey::inplaceIdx));
    }
}

Status ExpandFunction::DefaultEnabledPreCheck(Function &function) {
    ExpandFunctionChecker checker;
    return checker.DoDefaultEnabledPreCheck(function);
}

Status ExpandFunction::PostCheck(Function &function) {
    ExpandFunctionChecker checker;
    return checker.DoPostCheck(function);
}

Status ExpandFunction::RunOnFunction(Function &function) {
    APASS_LOG_INFO_F(Elements::Function, "Start ExpandFunction function [%s].", function.GetRawName().c_str());
    std::ostringstream oss;
    scopeMap_.clear();
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
    APASS_LOG_INFO_F(Elements::Function, "Function operation size is: %zu after expansion.", function.Operations().size());
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
    if (ClearIOOperand(tensorOperations) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "ClearIOOperand failed.");
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
        if (kNotNeedExpandOps.count(op->GetOpcode())) {
            ProcessForNotExpandOp(function, *op);
            continue;
        }
        config::SetSemanticLabel(op->GetSemanticLabel());
        size_t opListPreSize = function.Operations(false).size();
        if (ExpandOperation(function, *op) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "ExpandOperation failed.");
            return FAILED;
        }
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

Status ExpandFunction::ExpandOperation(Function &function, Operation &op) const{
    int scopeIdx = op.GetScopeId();
    if (scopeIdx >= 0) { // scopeIdx < 0 means no need to merge
        scopeMap_[scopeIdx].insert(op.GetCoreType());
        if (!GraphUtils::IsCVMixPlatform() && scopeMap_[scopeIdx].find(CoreType::AIC) != scopeMap_[scopeIdx].end() && scopeMap_[scopeIdx].find(CoreType::AIV) != scopeMap_[scopeIdx].end()) {
            APASS_LOG_ERROR_F(Elements::Function, "Cannot mix cube and vector op on a CV seperate platform in function: %s, please check your setting: sg_set_scope=%d", function.GetRawName().c_str(), scopeIdx);
            return FAILED;
        }
    }
    config::SetPassOption(SG_SET_SCOPE, scopeIdx);
    ExpandOperationInto(function, op.GetTileShape(), op.GetOpcode(), op.GetIOperands(), op.GetOOperands(), op);
    config::SetPassOption(SG_SET_SCOPE, -1);
    return SUCCESS;
}

void ExpandFunction::DoHealthCheckBefore(Function &function, const std::string &folderPath) {
    APASS_LOG_INFO_F(Elements::Operation, "Before ExpandFunction, Health Report: TensorGraph START");
    std::string fileName = GetDumpFilePrefix(function, true);
    HealthCheckTensorGraph(function, folderPath, fileName);
    APASS_LOG_INFO_F(Elements::Operation, "Before ExpandFunction, Health Report: TensorGraph END");
}
} // namespace npu::tile_fwk
