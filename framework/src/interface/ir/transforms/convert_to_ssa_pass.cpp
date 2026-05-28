/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <cstddef>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "core/logging.h"
#include "ir/expr.h"
#include "ir/function.h"
#include "ir/kind_traits.h"
#include "ir/span.h"
#include "ir/stmt.h"
#include "ir/transforms/base/mutator.h"
#include "ir/transforms/base/visitor.h"
#include "ir/transforms/passes.h"
#include "ir/transforms/pass_properties.h"
#include "ir/type.h"

namespace pypto {
namespace ir {

namespace {

/**
 * \brief Get the identity key for a variable name
 *
 * Returns the name unchanged. Each variable name is treated as a unique identity.
 * This ensures variables like "tmp_0" and "tmp_1" are treated as distinct variables,
 * not as different versions of the same base variable "tmp".
 *
 * Note: The function name "GetBaseName" is retained for compatibility with existing
 * call sites throughout the SSA converter, but no name normalization is performed.
 */
static std::string GetBaseName(const std::string& name) { return name; }

/**
 * \brief Collects all assigned variable base names in a statement
 *
 * Used to pre-analyze loop bodies to find which outer variables are modified,
 * allowing us to create iter_args before visiting the body.
 */
class AssignmentCollector : public IRVisitor {
    using IRVisitor::VisitExpr_;
    using IRVisitor::VisitStmt_;

public:
    std::set<std::string> assigned_vars;

    void Collect(const StmtPtr& stmt) { VisitStmt(stmt); }

protected:
    void VisitStmt_(const AssignStmtPtr& op) override
    {
        // Extract base name from assignment target
        assigned_vars.insert(GetBaseName(op->var_->name_));
        // Also visit the value in case of nested assignments
        VisitExpr(op->value_);
    }

    void VisitStmt_(const ForStmtPtr& op) override
    {
        // The loop variable is scoped to the loop body and must not be treated as an
        // outer variable assignment candidate for loop-carried analysis. Recurse into
        // the body so we still collect writes to real outer-scope variables.
        VisitStmt(op->body_);
    }

    void VisitStmt_(const WhileStmtPtr& op) override
    {
        // Don't recurse into nested while loops - they handle their own iter_args
        // Visit condition to collect any assignments (though unusual)
        VisitExpr(op->condition_);
        // Visit the body to collect assignments
        VisitStmt(op->body_);
    }

    void VisitStmt_(const IfStmtPtr& op) override
    {
        // Visit both branches
        VisitStmt(op->thenBody_);
        if (op->elseBody_.has_value()) {
            VisitStmt(*op->elseBody_);
        }
    }

    void VisitStmt_(const SeqStmtsPtr& op) override
    {
        for (const auto& s : op->stmts_) {
            VisitStmt(s);
        }
    }
};

/**
 * \brief SSA Converter - Transforms non-SSA IR to SSA form
 *
 * This mutator converts IR with multiple assignments per variable to SSA form by:
 * 1. Renaming variables with version suffixes (x -> x_0, x_1, x_2)
 * 2. Adding phi nodes (return_vars + YieldStmt) for IfStmt control flow
 * 3. Converting loop-modified variables to iter_args + return_vars pattern
 */
class SSAConverter : public IRMutator {
    using IRMutator::VisitExpr_;
    using IRMutator::VisitStmt_;
    using VersionMap = std::unordered_map<std::string, VarPtr>;

public:
    SSAConverter() = default;

    /**
     * \brief Convert a function to SSA form
     */
    FunctionPtr Convert(const FunctionPtr& func)
    {
        // Initialize version counters for parameters
        for (size_t i = 0; i < func->params_.size(); ++i) {
            const auto& var = func->params_[i];
            std::string base_name = GetBaseName(var->name_);
            int version = NextVersion(base_name);
            auto versioned_param = CreateVersionedVar(var, base_name, version);
            current_version_[base_name] = versioned_param;
            new_params_.push_back(versioned_param);
        }

        // Transform the function body
        StmtPtr new_body = nullptr;
        if (func->body_) {
            new_body = VisitStmt(func->body_);
        }

        // Create the new function with versioned parameters
        return std::make_shared<Function>(
            func->name_, new_params_, func->returnTypes_, new_body, func->span_, func->funcType_);
    }

protected:
    // Override expression visitation to replace Var with current version
    ExprPtr VisitExpr_(const VarPtr& op) override
    {
        std::string base_name = GetBaseName(op->name_);
        auto it = current_version_.find(base_name);
        if (it != current_version_.end()) {
            return it->second;
        }
        // Variable not found in current scope - return as-is
        // This can happen for variables that are only defined once
        return op;
    }

    // Override assignment statement to create versioned variables
    StmtPtr VisitStmt_(const AssignStmtPtr& op) override
    {
        // First, visit the RHS expression (uses current versions)
        auto new_value = VisitExpr(op->value_);

        // Create a new versioned variable for LHS
        std::string base_name = GetBaseName(op->var_->name_);
        int version = NextVersion(base_name);
        auto new_var = CreateVersionedVar(op->var_, base_name, version);

        // Update current version mapping
        current_version_[base_name] = new_var;

        return std::make_shared<AssignStmt>(new_var, new_value, op->span_);
    }

    // Override IfStmt to handle phi nodes
    StmtPtr VisitStmt_(const IfStmtPtr& op) override
    {
        auto new_condition = VisitExpr(op->condition_);
        auto versions_before = current_version_;

        StmtPtr new_then;
        auto versions_after_then = VisitThenBranch(op->thenBody_, new_then);

        VersionMap versions_after_else;
        auto new_else = VisitElseBranch(op->elseBody_, versions_before, versions_after_else);
        auto phi_vars = CollectIfPhiVars(versions_before, versions_after_then, versions_after_else);

        // If no variables diverged, just return the updated if statement
        if (phi_vars.empty() && op->returnVars_.empty()) {
            current_version_ = versions_after_then; // Use then branch versions as default
            return std::make_shared<IfStmt>(new_condition, new_then, new_else, std::vector<VarPtr>{}, op->span_);
        }

        // Create return_vars and yields for phi nodes
        std::vector<VarPtr> return_vars;
        std::vector<ExprPtr> then_yields;
        std::vector<ExprPtr> else_yields;

        BuildIfPhiOutputs(
            phi_vars, versions_before, versions_after_then, versions_after_else, op->span_, return_vars, then_yields,
            else_yields);
        AppendExistingIfReturnVars(op->returnVars_, phi_vars, return_vars);

        // Append YieldStmt to branches
        auto then_with_yield = AppendYield(new_then, then_yields, op->span_);
        StmtPtr else_with_yield;
        if (new_else.has_value()) {
            else_with_yield = AppendYield(*new_else, else_yields, op->span_);
        } else {
            // Create an else branch with just the yield
            else_with_yield = std::make_shared<YieldStmt>(else_yields, op->span_);
        }

        return std::make_shared<IfStmt>(
            new_condition, then_with_yield, std::make_optional(else_with_yield), return_vars, op->span_);
    }

    // Override SectionStmt to isolate variable scope
    // Variables defined inside a section (cube/vector) must not leak to sibling sections.
    StmtPtr VisitStmt_(const SectionStmtPtr& op) override
    {
        EnterScope();
        auto new_body = VisitStmt(op->body_);
        ExitScope();
        return std::make_shared<SectionStmt>(op->sectionKind_, new_body, op->span_);
    }

    // Override ForStmt to handle loop-carried variables
    StmtPtr VisitStmt_(const ForStmtPtr& op) override
    {
        // Visit range expressions in outer scope
        auto new_start = VisitExpr(op->start_);
        auto new_stop = VisitExpr(op->stop_);
        auto new_step = VisitExpr(op->step_);

        // Save outer scope versions
        auto versions_before = current_version_;

        auto new_iter_args = VisitIterArgInitializers(op->iterArgs_);

        // PRE-ANALYSIS: Find which outer variables are assigned in the loop body.
        // This allows us to create iter_args BEFORE visiting the body.
        AssignmentCollector collector;
        collector.Collect(op->body_);

        std::string loop_var_base = GetBaseName(op->loopVar_->name_);
        auto loop_carried_vars = CollectLoopCarriedVars(
            collector.assigned_vars, versions_before, op->iterArgs_, std::make_optional(loop_var_base));

        // Create iter_args for loop-carried variables BEFORE visiting the body
        auto return_vars = AppendLoopCarriedIterArgs(loop_carried_vars, versions_before, new_iter_args, op->span_);

        // Enter loop scope
        EnterScope();

        // Create versioned loop variable
        int loop_var_version = NextVersion(loop_var_base);
        auto loop_var_type = SubstituteVarsInType(op->loopVar_->GetType());
        auto new_loop_var = std::make_shared<Var>(
            loop_var_base + "_" + std::to_string(loop_var_version), loop_var_type, op->loopVar_->span_);
        current_version_[loop_var_base] = new_loop_var;

        RegisterIterArgsInCurrentScope(new_iter_args);

        // Visit loop body - now it will correctly reference iter_args
        auto new_body = VisitStmt(op->body_);
        auto versions_after_body = current_version_;

        // Exit loop scope
        ExitScope();

        // Update outer scope to use return_vars for loop-carried variables
        for (size_t i = 0; i < loop_carried_vars.size(); ++i) {
            current_version_[loop_carried_vars[i]] = return_vars[i];
        }

        auto yield_values = CollectYieldValues(new_body, versions_after_body, loop_carried_vars);

        // Copy existing return_vars (from explicit iter_args in original code)
        for (const auto& rv : op->returnVars_) {
            return_vars.push_back(rv);
        }

        // Update body with new yield
        StmtPtr final_body = new_body;
        if (!yield_values.empty()) {
            final_body = ReplaceOrAppendYield(new_body, yield_values, op->span_);
        }

        return std::make_shared<ForStmt>(
            new_loop_var, new_start, new_stop, new_step, new_iter_args, final_body, return_vars, op->span_);
    }

    // Override WhileStmt to handle loop-carried variables
    StmtPtr VisitStmt_(const WhileStmtPtr& op) override
    {
        // Save outer scope versions
        auto versions_before = current_version_;

        auto new_iter_args = VisitIterArgInitializers(op->iterArgs_);

        // PRE-ANALYSIS: Find which outer variables are assigned in the loop body
        AssignmentCollector collector;
        collector.Collect(op->body_);
        // Also collect from condition (though unusual, it's possible)
        collector.Collect(std::make_shared<EvalStmt>(op->condition_, op->span_));

        auto loop_carried_vars =
            CollectLoopCarriedVars(collector.assigned_vars, versions_before, op->iterArgs_, std::nullopt);

        // Create iter_args for loop-carried variables BEFORE visiting the body
        auto new_loop_carried_return_vars =
            AppendLoopCarriedIterArgs(loop_carried_vars, versions_before, new_iter_args, op->span_);

        // Enter loop scope
        EnterScope();

        RegisterIterArgsInCurrentScope(new_iter_args);

        // Visit condition - it will reference iter_args
        auto new_condition = VisitExpr(op->condition_);

        // Visit loop body - now it will correctly reference iter_args
        auto new_body = VisitStmt(op->body_);
        auto versions_after_body = current_version_;

        // Exit loop scope
        ExitScope();

        // Build return_vars in same order as new_iter_args and yield_values:
        // First existing return_vars, then new loop-carried return_vars
        std::vector<VarPtr> return_vars;
        for (const auto& rv : op->returnVars_) {
            return_vars.push_back(rv);
        }
        for (const auto& rv : new_loop_carried_return_vars) {
            return_vars.push_back(rv);
        }

        // Update outer scope to use return_vars for loop-carried variables
        for (size_t i = 0; i < loop_carried_vars.size(); ++i) {
            current_version_[loop_carried_vars[i]] = new_loop_carried_return_vars[i];
        }

        auto yield_values = CollectYieldValues(new_body, versions_after_body, loop_carried_vars);

        // Update body with new yield
        StmtPtr final_body = new_body;
        if (!yield_values.empty()) {
            final_body = ReplaceOrAppendYield(new_body, yield_values, op->span_);
        }

        return std::make_shared<WhileStmt>(new_condition, new_iter_args, final_body, return_vars, op->span_);
    }

private:
    // Version counter per base variable name
    std::unordered_map<std::string, int> version_counter_;

    // Current version of each variable (base_name -> versioned VarPtr)
    std::unordered_map<std::string, VarPtr> current_version_;

    // Scope stack for nested control flow
    std::vector<std::unordered_map<std::string, VarPtr>> scope_stack_;

    // New versioned parameters
    std::vector<VarPtr> new_params_;

    VersionMap VisitThenBranch(const StmtPtr& then_body, StmtPtr& new_then)
    {
        EnterScope();
        new_then = VisitStmt(then_body);
        auto versions_after_then = current_version_;
        ExitScope();
        return versions_after_then;
    }

    std::optional<StmtPtr> VisitElseBranch(
        const std::optional<StmtPtr>& else_body, const VersionMap& versions_before, VersionMap& versions_after_else)
    {
        current_version_ = versions_before;
        if (!else_body.has_value()) {
            versions_after_else = versions_before;
            return std::nullopt;
        }

        EnterScope();
        auto new_else = VisitStmt(*else_body);
        versions_after_else = current_version_;
        ExitScope();
        return new_else;
    }

    void AppendChangedPhiVars(
        const VersionMap& versions_before, const VersionMap& source_versions, const VersionMap& other_versions,
        std::set<std::string>& checked_vars, std::vector<std::string>& phi_vars) const
    {
        for (const auto& [base_name, var] : source_versions) {
            if (checked_vars.count(base_name)) {
                continue;
            }
            checked_vars.insert(base_name);
            auto before_it = versions_before.find(base_name);
            if (before_it == versions_before.end()) {
                continue;
            }
            bool changed_in_source = before_it->second != var;
            auto other_it = other_versions.find(base_name);
            bool changed_in_other = other_it != other_versions.end() && before_it->second != other_it->second;
            if (changed_in_source || changed_in_other) {
                phi_vars.push_back(base_name);
            }
        }
    }

    std::vector<std::string> CollectIfPhiVars(
        const VersionMap& versions_before, const VersionMap& versions_after_then,
        const VersionMap& versions_after_else) const
    {
        std::vector<std::string> phi_vars;
        std::set<std::string> checked_vars;
        AppendChangedPhiVars(versions_before, versions_after_then, versions_after_else, checked_vars, phi_vars);
        AppendChangedPhiVars(versions_before, versions_after_else, VersionMap{}, checked_vars, phi_vars);
        std::sort(phi_vars.begin(), phi_vars.end());
        return phi_vars;
    }

    void BuildIfPhiOutputs(
        const std::vector<std::string>& phi_vars, const VersionMap& versions_before,
        const VersionMap& versions_after_then, const VersionMap& versions_after_else, const Span& span,
        std::vector<VarPtr>& return_vars, std::vector<ExprPtr>& then_yields, std::vector<ExprPtr>& else_yields)
    {
        for (const auto& base_name : phi_vars) {
            VarPtr then_var = versions_after_then.count(base_name) ? versions_after_then.at(base_name) :
                                                                     versions_before.at(base_name);
            VarPtr else_var = versions_after_else.count(base_name) ? versions_after_else.at(base_name) :
                                                                     versions_before.at(base_name);
            int phi_version = NextVersion(base_name);
            auto phi_type = SubstituteVarsInType(then_var->GetType());
            auto phi_var = std::make_shared<Var>(base_name + "_" + std::to_string(phi_version), phi_type, span);
            return_vars.push_back(phi_var);
            then_yields.push_back(then_var);
            else_yields.push_back(else_var);
            current_version_[base_name] = phi_var;
        }
    }

    void AppendExistingIfReturnVars(
        const std::vector<VarPtr>& existing_return_vars, const std::vector<std::string>& phi_vars,
        std::vector<VarPtr>& return_vars)
    {
        for (const auto& existing_rv : existing_return_vars) {
            std::string base_name = GetBaseName(existing_rv->name_);
            if (std::find(phi_vars.begin(), phi_vars.end(), base_name) != phi_vars.end()) {
                continue;
            }
            int rv_version = NextVersion(base_name);
            auto rv_type = SubstituteVarsInType(existing_rv->GetType());
            auto versioned_rv =
                std::make_shared<Var>(base_name + "_" + std::to_string(rv_version), rv_type, existing_rv->span_);
            return_vars.push_back(versioned_rv);
            current_version_[base_name] = versioned_rv;
        }
    }

    std::vector<IterArgPtr> VisitIterArgInitializers(const std::vector<IterArgPtr>& iter_args)
    {
        std::vector<IterArgPtr> new_iter_args;
        for (const auto& iter_arg : iter_args) {
            auto new_init = VisitExpr(iter_arg->initValue_);
            auto ia_type = SubstituteVarsInType(iter_arg->iterVar_->GetType());
            auto new_var = std::make_shared<Var>(iter_arg->iterVar_->name_, ia_type, iter_arg->iterVar_->span_);
            new_iter_args.push_back(std::make_shared<IterArg>(new_var, new_init));
        }
        return new_iter_args;
    }

    std::vector<std::string> CollectLoopCarriedVars(
        const std::set<std::string>& assigned_vars, const std::unordered_map<std::string, VarPtr>& versions_before,
        const std::vector<IterArgPtr>& iter_args, const std::optional<std::string>& skipped_name) const
    {
        std::vector<std::string> loop_carried_vars;
        for (const auto& assigned_name : assigned_vars) {
            if (skipped_name.has_value() && assigned_name == *skipped_name)
                continue;

            bool is_existing_iter_arg = false;
            for (const auto& iter_arg : iter_args) {
                if (GetBaseName(iter_arg->iterVar_->name_) == assigned_name) {
                    is_existing_iter_arg = true;
                    break;
                }
            }
            if (is_existing_iter_arg)
                continue;

            if (versions_before.find(assigned_name) != versions_before.end()) {
                loop_carried_vars.push_back(assigned_name);
            }
        }
        return loop_carried_vars;
    }

    std::vector<VarPtr> AppendLoopCarriedIterArgs(
        const std::vector<std::string>& loop_carried_vars,
        const std::unordered_map<std::string, VarPtr>& versions_before, std::vector<IterArgPtr>& iter_args,
        const Span& span)
    {
        std::vector<VarPtr> return_vars;
        for (const auto& base_name : loop_carried_vars) {
            auto init_var = versions_before.at(base_name);
            int ia_version = NextVersion(base_name);
            auto iter_arg = std::make_shared<IterArg>(
                base_name + "_iter_" + std::to_string(ia_version), init_var->GetType(), init_var, span);
            iter_args.push_back(iter_arg);

            int rv_version = NextVersion(base_name);
            auto rv_type = SubstituteVarsInType(init_var->GetType());
            return_vars.push_back(std::make_shared<Var>(base_name + "_" + std::to_string(rv_version), rv_type, span));
        }
        return return_vars;
    }

    void RegisterIterArgsInCurrentScope(const std::vector<IterArgPtr>& iter_args)
    {
        for (const auto& iter_arg : iter_args) {
            std::string base_name = GetBaseName(iter_arg->iterVar_->name_);
            size_t iter_pos = base_name.find("_iter");
            if (iter_pos != std::string::npos) {
                base_name = base_name.substr(0, iter_pos);
            }
            current_version_[base_name] = iter_arg->iterVar_;
        }
    }

    std::vector<ExprPtr> CollectYieldValues(
        const StmtPtr& body, const std::unordered_map<std::string, VarPtr>& versions_after_body,
        const std::vector<std::string>& loop_carried_vars)
    {
        std::vector<ExprPtr> yield_values;
        if (auto yield_stmt = GetLastYieldStmt(body)) {
            yield_values = yield_stmt->value_;
        }
        for (const auto& base_name : loop_carried_vars) {
            yield_values.push_back(versions_after_body.at(base_name));
        }
        return yield_values;
    }

    /**
     * \brief Get next version number for a base name
     */
    int NextVersion(const std::string& base_name)
    {
        int version = version_counter_[base_name];
        version_counter_[base_name] = version + 1;
        return version;
    }

    /**
     * \brief Substitute Vars in a type's tile_view.validShape using current_version_
     *
     * When ConvertToSSA renames parameters (e.g., M → M_0), the Var references
     * embedded in TileType::tile_view.validShape must also be updated to keep
     * the IR consistent.
     */
    TypePtr SubstituteVarsInType(const TypePtr& type)
    {
        // Handle TileType: substitute Vars in tile_view.validShape
        if (auto tile_type = As<TileType>(type)) {
            if (!tile_type->tileView_.has_value())
                return type;
            const auto& tv = tile_type->tileView_.value();
            if (tv.validShape.empty())
                return type;
            std::vector<ExprPtr> new_valid_shape;
            bool changed = false;
            for (const auto& vs : tv.validShape) {
                auto new_vs = VisitExpr(vs);
                if (new_vs != vs)
                    changed = true;
                new_valid_shape.push_back(new_vs);
            }
            if (!changed)
                return type;
            TileView new_tile_view = tv;
            new_tile_view.validShape = std::move(new_valid_shape);
            return std::make_shared<TileType>(
                tile_type->shape_, tile_type->dtype_, tile_type->memref_, std::make_optional(std::move(new_tile_view)));
        }

        // Handle PtrType: substitute Vars in base_ptr and offset (codegen-level fields)
        if (auto ptr_type = As<PtrType>(type)) {
            if (!ptr_type->base_ptr.has_value())
                return type;
            auto new_base = VisitExpr(*ptr_type->base_ptr);
            ExprPtr new_offset{};
            if (ptr_type->offset.has_value())
                new_offset = VisitExpr(*ptr_type->offset);
            bool changed =
                (new_base != *ptr_type->base_ptr) || (ptr_type->offset.has_value() && new_offset != *ptr_type->offset);
            if (!changed)
                return type;
            auto result = std::make_shared<PtrType>(ptr_type->dtype_);
            result->base_ptr = new_base;
            result->offset = ptr_type->offset.has_value() ? std::make_optional(new_offset) : std::nullopt;
            return result;
        }

        // Handle TensorType: substitute Vars in tensor_view_.ptr (and its PtrType fields)
        if (auto tensor_type = As<TensorType>(type)) {
            if (!tensor_type->tensor_view_.has_value())
                return type;
            const auto& tv = tensor_type->tensor_view_.value();
            if (!tv.ptr.has_value())
                return type;
            auto new_ptr_expr = VisitExpr(*tv.ptr);
            bool ptr_changed = (new_ptr_expr != *tv.ptr);
            if (!ptr_changed)
                return type;
            TensorView new_tv = tv;
            new_tv.ptr = new_ptr_expr;
            return std::make_shared<TensorType>(
                tensor_type->shape_, tensor_type->dtype_, tensor_type->memref_, std::make_optional(std::move(new_tv)));
        }

        return type;
    }

    /**
     * \brief Create a versioned variable from an original variable
     *
     * Also substitutes any Var references embedded in the variable's type
     * (e.g., TileType::tile_view.validShape) using the current version map.
     */
    VarPtr CreateVersionedVar(const VarPtr& original, const std::string& base_name, int version)
    {
        std::string versioned_name = base_name + "_" + std::to_string(version);
        auto type = SubstituteVarsInType(original->GetType());
        return std::make_shared<Var>(versioned_name, type, original->span_);
    }

    /**
     * \brief Enter a new scope
     */
    void EnterScope() { scope_stack_.push_back(current_version_); }

    /**
     * \brief Exit current scope
     */
    void ExitScope()
    {
        if (!scope_stack_.empty()) {
            current_version_ = scope_stack_.back();
            scope_stack_.pop_back();
        }
    }

    /**
     * \brief Append a YieldStmt to a statement
     *
     * When the statement already ends with a YieldStmt (e.g. from LowerBreakContinue),
     * the new phi values are prepended to the existing yield values so that the merged
     * YieldStmt matches the return_vars order: [phi_vars..., existing_rvs...].
     */
    StmtPtr AppendYield(const StmtPtr& stmt, const std::vector<ExprPtr>& values, const Span& span)
    {
        if (values.empty())
            return stmt;

        if (auto seq = As<SeqStmts>(stmt)) {
            if (!seq->stmts_.empty()) {
                if (auto existing_yield = As<YieldStmt>(seq->stmts_.back())) {
                    // Merge: new phi values first, then existing values
                    std::vector<ExprPtr> merged = values;
                    merged.insert(merged.end(), existing_yield->value_.begin(), existing_yield->value_.end());
                    auto merged_yield = std::make_shared<YieldStmt>(merged, span);
                    std::vector<StmtPtr> new_stmts(seq->stmts_.begin(), seq->stmts_.end() - 1);
                    new_stmts.push_back(merged_yield);
                    return std::make_shared<SeqStmts>(new_stmts, seq->span_);
                }
            }
            // No trailing yield — just append
            std::vector<StmtPtr> new_stmts = seq->stmts_;
            new_stmts.push_back(std::make_shared<YieldStmt>(values, span));
            return std::make_shared<SeqStmts>(new_stmts, seq->span_);
        }

        if (auto existing_yield = As<YieldStmt>(stmt)) {
            // Merge: new phi values first, then existing values
            std::vector<ExprPtr> merged = values;
            merged.insert(merged.end(), existing_yield->value_.begin(), existing_yield->value_.end());
            return std::make_shared<YieldStmt>(merged, span);
        }

        // No existing yield — wrap statement and new yield in SeqStmts
        return std::make_shared<SeqStmts>(std::vector<StmtPtr>{stmt, std::make_shared<YieldStmt>(values, span)}, span);
    }

    /**
     * \brief Get the last YieldStmt from a statement (if any)
     */
    YieldStmtPtr GetLastYieldStmt(const StmtPtr& stmt)
    {
        if (auto yield = As<YieldStmt>(stmt)) {
            return yield;
        }
        if (auto seq = As<SeqStmts>(stmt)) {
            if (!seq->stmts_.empty()) {
                return As<YieldStmt>(seq->stmts_.back());
            }
        }
        return nullptr;
    }

    /**
     * \brief Replace or append yield statement
     */
    StmtPtr ReplaceOrAppendYield(const StmtPtr& stmt, const std::vector<ExprPtr>& values, const Span& span)
    {
        auto new_yield = std::make_shared<YieldStmt>(values, span);

        if (auto seq = As<SeqStmts>(stmt)) {
            if (!seq->stmts_.empty() && As<YieldStmt>(seq->stmts_.back())) {
                // Replace last yield
                std::vector<StmtPtr> new_stmts(seq->stmts_.begin(), seq->stmts_.end() - 1);
                new_stmts.push_back(new_yield);
                return std::make_shared<SeqStmts>(new_stmts, seq->span_);
            }
            // Append yield
            std::vector<StmtPtr> new_stmts = seq->stmts_;
            new_stmts.push_back(new_yield);
            return std::make_shared<SeqStmts>(new_stmts, seq->span_);
        }

        if (As<YieldStmt>(stmt)) {
            return new_yield;
        }

        // Wrap single statement and yield in SeqStmts
        return std::make_shared<SeqStmts>(std::vector<StmtPtr>{stmt, new_yield}, span);
    }
};

/**
 * \brief Transform function: Convert a function to SSA form
 */
FunctionPtr TransformConvertToSSA(const FunctionPtr& func)
{
    INTERNAL_CHECK(func) << "ConvertToSSA cannot run on null function";
    SSAConverter converter;
    auto ssa_func = converter.Convert(func);

    return ssa_func;
}

} // namespace

// Factory function
namespace pass {
Pass ConvertToSSA() { return CreateFunctionPass(TransformConvertToSSA, "ConvertToSSA", kConvertToSSAProperties); }
} // namespace pass

} // namespace ir
} // namespace pypto
