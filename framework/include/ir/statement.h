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
 * \file statement.h
 * \brief
 */

#pragma once

#include "ir/object.h"
#include "ir/value.h"
#include "ir/utils.h"
#include "ir/operation.h"
#include "ir/tile_graph.h"

#include <memory>
#include <ostream>
#include <string>
#include <vector>
#include <unordered_map>

namespace pto {

// Kinds of statement nodes supported in the prototype.
enum class StatementKind {
    Compound,
    Op,
    For,
    If,
    Yield,
    Call,
    Return
    // Memory operations (load/store/alloc/...) can be added later.
};

// Base class for all statement nodes.
class Statement : public Object {
public:
    Statement() : Object(ObjectType::Statement) {}
    virtual ~Statement() = default;

    ObjectType GetObjectType() const override { return ObjectType::Statement; }

    virtual StatementKind GetKind() const = 0;

    // Pretty-print with the given indentation (in spaces).
    virtual void Print(std::ostream& os, int indent) const = 0;
};

using StatementPtr = std::shared_ptr<Statement>;

// CompoundStatement class (Scope) for managing Value objects and statements in nested scopes.
// Each scope has a pointer to its parent scope, forming a scope chain.
class CompoundStatement : public Statement {
public:
    // Create a root scope (no parent).
    CompoundStatement() {}

    // Create a child scope with a parent scope.
    explicit CompoundStatement(std::shared_ptr<CompoundStatement> parent) : parent_(parent) {}

    StatementKind GetKind() const override { return StatementKind::Compound; }

    void Print(std::ostream& os, [[maybe_unused]] int indent) const override { os << "Not impl CompundStatement.Print() now!!!" << std::endl; }

    // Get the parent scope (nullptr if this is a root scope).
    std::weak_ptr<CompoundStatement> GetParent() const { return parent_; }
    void SetParent(std::weak_ptr<CompoundStatement> parent) { parent_ = parent; }

    // Get the number of statements in this scope.
    size_t GetStatementsNum() const { return statements_.size(); }
    // Get the statement by index
    StatementPtr GetStatement(size_t index) const { return statements_[index]; }
    // Set the statement by index
    void SetStatement(size_t index, StatementPtr stmt) { statements_[index] = stmt; }
    // Get all statements
    std::vector<StatementPtr> &GetStatements() { return statements_; }

    // Add a Statement to this scope.
    void AddStatement(StatementPtr stmt) { statements_.push_back(std::move(stmt)); }

    // Find a Value object by name in this scope and parent scopes.
    // Returns nullptr if not found.
    ValuePtr FindValue(const std::string& name) const;
    // Romeve a Value in this scope
    void RemoveValue(ValuePtr val);

    // Get all Value objects from ancestor scopes (excluding current scope).
    // Returns a map of variable name -> ValuePtr from parent, grandparent, etc. scopes.
    std::unordered_map<std::string, ValuePtr> GetAncestorValues() const;

    // ===== Environment table management (for SSA variable tracking) =====
    // Environment table: maps variable name (string) to latest SSA ValuePtr
    // This is used to track the latest version of variables across scopes

    // Set a variable in the environment table (by name)
    void SetEnvVar(const std::string& name, ValuePtr value);

    // Get a variable from the environment table (by name), searching up the scope chain
    ValuePtr GetEnvVar(const std::string& name) const;

    // Get the environment table for this scope
    std::unordered_map<std::string, ValuePtr>& GetEnvTable() { return envTable_; }
    const std::unordered_map<std::string, ValuePtr>& GetEnvTable() const { return envTable_; }

private:
    std::weak_ptr<CompoundStatement> parent_{};                         // Pointer to parent scope (nullptr for root)
    std::vector<StatementPtr> statements_;  // Statements in this scope
    std::unordered_map<std::string, ValuePtr> envTable_;  // Environment table: variable name -> latest SSA Value
};

using CompoundStatementPtr = std::shared_ptr<CompoundStatement>;

// A linear basic block of operations with nested statements as children.
class OpStatement : public Statement {
public:
    StatementKind GetKind() const override { return StatementKind::Op; }

    // Free-form textual operations representing concrete IR ops.
    std::vector<OperationPtr>& Operations() { return operations_; }
    const std::vector<OperationPtr>& Operations() const { return operations_; }

    void Print(std::ostream& os, int indent) const override;

private:
    std::vector<OperationPtr> operations_;
};

using OpStatementPtr = std::shared_ptr<OpStatement>;

// A generic scope terminator that returns values to the parent.
class YieldStatement : public Statement {
public:
    StatementKind GetKind() const override { return StatementKind::Yield; }

    std::vector<ValuePtr>& Values() { return values_; }
    const std::vector<ValuePtr>& Values() const { return values_; }

    void Print(std::ostream& os, int indent) const override;

private:
    std::vector<ValuePtr> values_;
};

using YieldStatementPtr = std::shared_ptr<YieldStatement>;

// Loop-carried accumulator argument for statement.for.
struct IterArg {
    ValuePtr initValue;    // Initial value (e.g., a ValuePtr to the initial tensor/scalar)
    ValuePtr value;        // The value representing this iter_arg in loop body
};

// Loop range containing start, end, and step as Scalar values.
class LoopRange {
public:
    LoopRange(ScalarValuePtr start, ScalarValuePtr end, ScalarValuePtr step)
        : start_(std::move(start)), end_(std::move(end)), step_(std::move(step)) {}

    const ScalarValuePtr& GetStart() const { return start_; }
    const ScalarValuePtr& GetEnd() const { return end_; }
    const ScalarValuePtr& GetStep() const { return step_; }

private:
    ScalarValuePtr start_;
    ScalarValuePtr end_;
    ScalarValuePtr step_;
};

// Sequential for loop with induction variable and loop-carried values.
class ForStatement : public Statement {
public:
    ForStatement(ScalarValuePtr iterationVar, ScalarValuePtr start,
                 ScalarValuePtr end, ScalarValuePtr step)
        : iterationVar_(std::move(iterationVar)),
          range_(std::make_shared<LoopRange>(std::move(start), std::move(end), std::move(step))) {
        compound_ = std::make_shared<CompoundStatement>();
    }

    StatementKind GetKind() const override { return StatementKind::For; }

    const ScalarValuePtr& GetIterationVar() const { return iterationVar_; }
    const ScalarValuePtr& GetStart() const { return range_->GetStart(); }
    const ScalarValuePtr& GetEnd() const { return range_->GetEnd(); }
    const ScalarValuePtr& GetStep() const { return range_->GetStep(); }
    const std::shared_ptr<LoopRange>& GetRange() const { return range_; }

    // Optional loop-carried accumulator arguments.
    std::vector<IterArg>& IterArgs() { return iterArgs_; }
    const std::vector<IterArg>& IterArgs() const { return iterArgs_; }

    // Add an iter_arg with the given initial value.
    // The value field will be created and set in ExitForStatement.
    void AddIterArg(ValuePtr initValue) {
        iterArgs_.push_back({std::move(initValue), nullptr});
    }

    // Loop body: sequence of nested statements.
    size_t BodyStmtsNum() const { return compound_->GetStatementsNum(); }
    StatementPtr GetBodyStatement(size_t index) const { return compound_->GetStatement(index); }
    void SetBodyStatement(size_t index, StatementPtr stmt) { compound_->SetStatement(index, stmt); }

    // Loop yield
    std::shared_ptr<YieldStatement> Yield();
    const std::shared_ptr<YieldStatement> Yield() const;

    // create new tensor for the for-statement
    void BuildResult();
    std::vector<ValuePtr>& Results() { return results_; }
    const std::vector<ValuePtr>& Results() const { return results_; }

    // Scope for Data objects and statements created in this loop body.
    CompoundStatementPtr GetCompound() { return compound_; }
    const CompoundStatementPtr GetCompound() const { return compound_; }

    void Print(std::ostream& os, int indent) const override;

private:
    ScalarValuePtr iterationVar_;
    std::shared_ptr<LoopRange> range_;
    std::vector<IterArg> iterArgs_;
    CompoundStatementPtr compound_;  // Scope for Data objects and statements created in this loop body
    std::vector<ValuePtr> results_;  // Result values of the for-statement
};

using ForStatementPtr = std::shared_ptr<ForStatement>;

// A value-producing conditional.
class IfStatement : public Statement {
public:
    explicit IfStatement(ScalarValuePtr condition)
        : condition_(std::move(condition)) {
        thenCompound_ = std::make_shared<CompoundStatement>();
        elseCompound_ = std::make_shared<CompoundStatement>();
    }

    StatementKind GetKind() const override { return StatementKind::If; }

    const ScalarValuePtr GetCondition() const { return condition_; }

    size_t ThenBranchStmtsNum() const { return thenCompound_->GetStatementsNum(); }
    StatementPtr GetThenBranchStatement(size_t index) const { return thenCompound_->GetStatement(index); }
    void SetThenBranchStatement(size_t index, StatementPtr stmt) { thenCompound_->SetStatement(index, stmt); }

    size_t ElseBranchStmtsNum() const { return elseCompound_->GetStatementsNum(); }
    StatementPtr GetElseBranchStatement(size_t index) const { return elseCompound_->GetStatement(index); }
    void SetElseBranchStatement(size_t index, StatementPtr stmt) { elseCompound_->SetStatement(index, stmt); }

    // Scope for Data objects and statements created in the then branch.
    CompoundStatementPtr GetThenCompound() { return thenCompound_; }
    const CompoundStatementPtr GetThenCompound() const { return thenCompound_; }

    // Scope for Data objects and statements created in the else branch.
    CompoundStatementPtr GetElseCompound() { return elseCompound_; }
    const CompoundStatementPtr GetElseCompound() const { return elseCompound_; }

    // Build result tensors for the if-statement based on branch yields.
    // This inspects the terminal YieldStatement in both then/else scopes,
    // verifies that the yielded values have the same Data types, and
    // creates new Tensor results for the if expression.
    void BuildResult();

    // Result values of the if-statement (SSA values).
    std::vector<ValuePtr>& Results() { return results_; }
    const std::vector<ValuePtr>& Results() const { return results_; }

    void Print(std::ostream& os, int indent) const override;

private:
    ScalarValuePtr condition_;
    CompoundStatementPtr thenCompound_;  // Scope for Data objects and statements created in the then branch
    CompoundStatementPtr elseCompound_;  // Scope for Data objects and statements created in the else branch
    std::vector<ValuePtr> results_;  // Result values of the if-statement
};

using IfStatementPtr = std::shared_ptr<IfStatement>;

// Function-level terminator returning final values.
class ReturnStatement : public Statement {
public:
    StatementKind GetKind() const override { return StatementKind::Return; }

    std::vector<ValuePtr>& Values() { return values_; }
    const std::vector<ValuePtr>& Values() const { return values_; }

    void Print(std::ostream& os, int indent) const override;

private:
    std::vector<ValuePtr> values_;
};

using ReturnStatementPtr = std::shared_ptr<ReturnStatement>;

}