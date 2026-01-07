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

#include <ostream>
#include <string>
#include <vector>

namespace pto {

// High-level classification of PTO functions.
enum class FunctionKind {
    ControlFlow, // control-flow functions using statement dialect
    DataFlow,    // pure data-flow graphs at tensor/tile level
    Kernel       // low-level kernels near instruction/memory level
};

// Signature of a function: arguments and results.
// Arguments are Data objects where the name field stores the argument name (e.g. "%A").
struct FunctionSignature {
    std::vector<ValuePtr> arguments; // argument types with names stored in Value::name
    std::vector<ValuePtr> results;  // return types
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

    // Pretty-print a standalone function in PTO-IR-like syntax.
    void Print(std::ostream& os, int indent = 0) const;

private:
    FunctionKind kind_;
    FunctionSignature signature_;
    CompoundStatementPtr inputCompound_; // Scope holding function arguments (inputs)
    CompoundStatementPtr compound_;  // Scope for Data objects and statements created in this function
};

// Helper for convenient streaming: std::cout << func;
std::ostream& operator<<(std::ostream& os, const Function& func);

} // namespace pto


