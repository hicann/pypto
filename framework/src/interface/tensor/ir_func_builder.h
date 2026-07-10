/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#pragma once

#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

#include "ir/expr.h"
#include "ir/function.h"
#include "ir/stmt.h"

namespace npu::tile_fwk {

class Function;
class LogicalTensor;
class Program;
class TensorSlotManager;
class TensorSlotScope;
class Tensor;
class Operation;
using LogicalTensorPtr = std::shared_ptr<LogicalTensor>;
using LogicalTensors = std::vector<LogicalTensorPtr>;

class RootFunctionBuilder {
public:
    explicit RootFunctionBuilder(Function* parentFunc);
    std::shared_ptr<Function> Build(const pypto::ir::FunctionPtr& irFunc);

private:
    Program& program_;
    Function* parentFunc_;
    std::shared_ptr<Function> dynFunc_;
    LogicalTensors logicalParams_;
    std::unordered_set<int> consumedRawMagics_;
    std::unordered_set<int> paramRawMagics_;

    void InitDynFunc(const pypto::ir::FunctionPtr& irFunc);
    void FinalizeDynFunc(const pypto::ir::FunctionPtr& irFunc);

    pypto::ir::StmtPtr TransformBody(pypto::ir::StmtPtr stmt);
    pypto::ir::StmtPtr TransformStmts(pypto::ir::StmtPtr stmt, const std::string& loopVarName);
    void ReplacePlaceholders(pypto::ir::StmtPtr stmt);

    pypto::ir::StmtPtr CreatePathFuncAndPlaceholder(
        const pypto::ir::SeqStmtsPtr& seq, const std::string& loopVarName);
    pypto::ir::StmtPtr FinalizePathFunc(const pypto::ir::StmtPtr& placeholder);

    std::shared_ptr<Function> CreatePathFunc(
        const pypto::ir::SeqStmtsPtr& seq, const std::string& loopVarName);
    pypto::ir::StmtPtr ProcessTensorOp(
        std::shared_ptr<Function> pathFunc, const pypto::ir::StmtPtr& stmt,
        std::unordered_set<std::shared_ptr<LogicalTensor>>& allInputs,
        std::unordered_set<std::shared_ptr<LogicalTensor>>& allOutputs,
        std::unordered_set<std::shared_ptr<LogicalTensor>>& definedOutputs);
    void ComputeIncast(
        Function& pathFunc,
        const std::unordered_set<std::shared_ptr<LogicalTensor>>& allInputs,
        const std::unordered_set<std::shared_ptr<LogicalTensor>>& definedOutputs);
    void ComputeOutcast(
        Function& pathFunc,
        const std::unordered_set<std::shared_ptr<LogicalTensor>>& allOutputs);

    void BuildDynSlotScope();
    void BuildPathFuncSlotScope(
        Function* pathFunc, const std::shared_ptr<TensorSlotScope>& scope,
        const LogicalTensors& originalIncasts, const LogicalTensors& originalOutcasts);
    int FindOrCreateSlot(
        const std::shared_ptr<LogicalTensor>& lt,
        const std::shared_ptr<TensorSlotManager>& slotManager,
        Function* func, bool isInput);

    bool IsPureTensorOpSeq(const pypto::ir::SeqStmtsPtr& seq);
    std::vector<std::vector<pypto::ir::StmtPtr>> SplitIntoTensorOpSegments(
        const pypto::ir::SeqStmtsPtr& seq);
    bool IsPlaceholderCallStmt(const pypto::ir::StmtPtr& stmt);
    std::string GetPlaceholderFuncname(const pypto::ir::StmtPtr& stmt);
    std::unordered_set<std::shared_ptr<LogicalTensor>> CollectAllOutputs(Function& pathFunc);
};

} // namespace npu::tile_fwk
