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
 * \file function.h
 * \brief
 */

#pragma once

#include "ir/statement.h"
#include "ir/value.h"
#include "ir/utils.h"
#include "interface/function/function.h"

#include <ostream>
#include <string>
#include <vector>

namespace pto {

// High-level classification of PTO functions.
enum class FunctionKind {
    ControlFlow, // control-flow functions using statement dialect
    DataFlow,    // pure data-flow graphs at tensor/tile level
    Block       // low-level kernels near instruction/memory level
};

// Signature of a function: arguments and results.
// Arguments are Data objects where the name field stores the argument name (e.g. "%A").
struct FunctionSignature {
    FunctionSignature() {}
    FunctionSignature(const std::vector<TensorValuePtr> &args)
    {
        for (auto &arg : args) {
            arguments.emplace_back(arg);
        }
    }

    FunctionSignature(const std::vector<TensorValuePtr> &inputArgs,
                      const std::vector<TensorValuePtr> &outputArgs) 
    {
        for (auto &inArg : inputArgs) {
            arguments.emplace_back(inArg);
        }

        for (auto &outArg : outputArgs) {
            arguments.emplace_back(outArg);
        }
    }
    std::vector<ValuePtr> arguments; // argument types with names stored in Value::name
    std::vector<ValuePtr> results;  // return types
    std::map<std::string, npu::tile_fwk::DynParamInfo> dynParamTable_;

    void SetDynParam(const std::string &name, const npu::tile_fwk::DynParamInfo &dynParam) {
        dynParamTable_[name] = dynParam;
    }

    const npu::tile_fwk::DynParamInfo &GetDynParam(const std::string &name) const {
        return dynParamTable_.at(name);
    }
};

// Minimal container for a PTO function.
// This prototype focuses on structural information and simple printing,
// not on detailed statement/tensor/tile bodies.
class Function : public Object {
public:
    Function(std::string name, FunctionKind kind, FunctionSignature signature);

    ObjectType GetObjectType() const override { return ObjectType::Function; }

    FunctionKind GetKind() const { return kind_; }
    const FunctionSignature& GetSignature() const { return signature_; }

    // Top-level statement sequence forming the function body.
    size_t BodyStmtsNum() const { return compound_->GetStatementsNum(); }
    StatementPtr GetBodyStatement(size_t index) const { return compound_->GetStatement(index); }
    void SetBodyStatement(size_t index, StatementPtr stmt) { compound_->SetStatement(index, stmt); }

    // Scope for Data objects and statements created in this function.
    CompoundStatementPtr GetCompound() { return compound_; }
    const CompoundStatementPtr GetCompound() const { return compound_; }

    // Scope containing function arguments. This scope is the parent of the function body scope.
    CompoundStatementPtr GetInputCompound() { return inputCompound_; }
    const CompoundStatementPtr GetInputCompound() const { return inputCompound_; }

    // Convenience to append a top-level statement.
    void AddStatement(StatementPtr stmt);

    // stack workspace size for function
    int GetStackWorkspaceSize() const { return stackWorkspaceSize_; }
    void SetStackWorkspaceSize(int stackWorkspaceSize) { stackWorkspaceSize_ = stackWorkspaceSize; }

    // Compute hash value for this function using bottom-up approach (always recomputes and updates cachedHash_)
    // Function hash contains Statement hashes, which contain Operation hashes
    uint64_t ComputeHash();

    // Get function hash value
    uint64_t GetFunctionHash() const { return functionHash_; }

    // check if value is from in cast
    bool isFromInCast(const ValuePtr &value) const;
    // check if value is from out cast
    bool isFromOutCast(const ValuePtr &value) const;
    // get index of in cast
    int GetIncastIndex(const ValuePtr &value) const;
    // get index of out cast
    int GetOutcastIndex(const ValuePtr &value) const;

    // Pretty-print a standalone function in PTO-IR-like syntax.
    void Print(std::ostream& os, int indent = 0) const;

protected:
    FunctionKind kind_;
    FunctionSignature signature_;
    CompoundStatementPtr inputCompound_; // Scope holding function arguments (inputs)
    CompoundStatementPtr compound_;  // Scope for Data objects and statements created in this function

private:
    int stackWorkspaceSize_{0};
    uint64_t functionHash_{0};
};

class BlockFunction : public Function {
public:
    BlockFunction(std::string name, FunctionSignature signature) : Function(name, FunctionKind::Block, signature) {}

    int GetProgramId() const { return programId_; }
    void SetProgramId(int programId) { programId_ = programId; }

    void SetDynParam(const std::string &name, const npu::tile_fwk::DynParamInfo &dynParam) {
        signature_.SetDynParam(name, dynParam);
    }

    const npu::tile_fwk::DynParamInfo &GetDynParam(const std::string &name) const {
        return signature_.GetDynParam(name);
    }

    void SetLeafFuncAttribute(const std::shared_ptr<npu::tile_fwk::LeafFuncAttribute> &leafFuncAttr) {
        leafFuncAttr_ = leafFuncAttr;
    }

    const std::shared_ptr<npu::tile_fwk::LeafFuncAttribute> &GetLeafFuncAttribute() const {
        return leafFuncAttr_;
    }

private:
    int programId_;
    std::shared_ptr<npu::tile_fwk::LeafFuncAttribute> leafFuncAttr_;
};

using FunctionPtr = std::shared_ptr<Function>;

// Helper for convenient streaming: std::cout << func;
std::ostream& operator<<(std::ostream& os, const Function& func);

} // namespace pto


