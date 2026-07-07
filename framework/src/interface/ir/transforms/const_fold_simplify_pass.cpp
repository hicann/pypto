/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "ir/expr.h"
#include "ir/function.h"
#include "ir/kind_traits.h"
#include "ir/scalar_expr.h"
#include "ir/span.h"
#include "ir/stmt.h"
#include "ir/transforms/base/mutator.h"
#include "ir/transforms/passes.h"
#include "ir/type.h"

namespace pypto {
namespace ir {

// ---------------------------------------------------------------------------
// Helper: try to extract compile-time integer value from an Expr
// ---------------------------------------------------------------------------
static std::optional<int64_t> TryGetConstInt(const ExprPtr& expr)
{
    if (!expr)
        return std::nullopt;
    if (auto ci = As<ConstInt>(expr))
        return ci->value_;
    return std::nullopt;
}

static DataType GetDType(const ExprPtr& expr)
{
    if (auto st = As<ScalarType>(expr->GetType()))
        return st->dtype_;
    return DataType::INDEX;
}

// ---------------------------------------------------------------------------
// Helper: resolve Var through a value map to find the underlying ConstInt
// ---------------------------------------------------------------------------
static std::optional<int64_t> ResolveToConst(
    const ExprPtr& expr, const std::unordered_map<std::string, ExprPtr>& var_vals)
{
    if (!expr)
        return std::nullopt;
    if (auto ci = As<ConstInt>(expr))
        return ci->value_;
    if (auto v = As<Var>(expr)) {
        auto it = var_vals.find(v->name_);
        if (it != var_vals.end())
            return ResolveToConst(it->second, var_vals);
    }
    // Resolve tuple-flavored GetItemExpr → MakeTuple element
    if (auto gi = As<GetItemExpr>(expr); gi && As<TupleType>(gi->value_->GetType())) {
        auto const_idx = As<ConstInt>(gi->slice_);
        if (!const_idx)
            return std::nullopt;
        ExprPtr tuple_expr = gi->value_;
        if (auto tv = As<Var>(tuple_expr)) {
            auto it = var_vals.find(tv->name_);
            if (it != var_vals.end())
                tuple_expr = it->second;
        }
        if (auto mt = As<MakeTuple>(tuple_expr)) {
            int idx = static_cast<int>(const_idx->value_);
            if (idx >= 0 && idx < static_cast<int>(mt->elements_.size())) {
                return ResolveToConst(mt->elements_[idx], var_vals);
            }
        }
    }
    return std::nullopt;
}

static bool IsSameExpr(
    const ExprPtr& a, const ExprPtr& b, const std::unordered_map<std::string, ExprPtr>& var_vals = {})
{
    if (!a || !b)
        return false;
    if (a.get() == b.get())
        return true;
    // Resolve through value map
    auto av = ResolveToConst(a, var_vals);
    auto bv = ResolveToConst(b, var_vals);
    if (av && bv)
        return *av == *bv;
    // Both Var with same name
    auto va = As<Var>(a);
    auto vb = As<Var>(b);
    if (va && vb)
        return va->name_ == vb->name_;
    return false;
}

// ---------------------------------------------------------------------------
// ConstFoldMutator — IRMutator subclass that folds constants and simplifies ifs
// ---------------------------------------------------------------------------
class ConstFoldMutator : public IRMutator {
public:
    using IRMutator::VisitExpr_;
    using IRMutator::VisitStmt_;

    // Track Var → assigned Expr for value resolution
    std::unordered_map<std::string, ExprPtr> var_vals_;

    // ---- Track assignments for value resolution ----
    StmtPtr VisitStmt_(const AssignStmtPtr& op) override
    {
        auto new_val = VisitExpr(op->value_);
        // Record the assigned value (works for ConstInt, MakeTuple, GetItemExpr, etc.)
        var_vals_[op->var_->name_] = new_val;
        if (new_val.get() != op->value_.get())
            return std::make_shared<AssignStmt>(op->var_, new_val, op->span_);
        return op;
    }

    // ---- Constant fold binary expressions ----
    ExprPtr VisitExpr_(const AddPtr& op) override
    {
        auto l = VisitExpr(op->left_);
        auto r = VisitExpr(op->right_);
        auto lv = TryGetConstInt(l), rv = TryGetConstInt(r);
        if (lv && rv)
            return std::make_shared<ConstInt>(*lv + *rv, GetDType(op), op->span_);
        // x + 0 → x
        if (rv && *rv == 0)
            return l;
        if (lv && *lv == 0)
            return r;
        if (l.get() != op->left_.get() || r.get() != op->right_.get())
            return std::make_shared<Add>(l, r, GetDType(op), op->span_);
        return op;
    }

    ExprPtr VisitExpr_(const SubPtr& op) override
    {
        auto l = VisitExpr(op->left_);
        auto r = VisitExpr(op->right_);
        auto lv = TryGetConstInt(l), rv = TryGetConstInt(r);
        if (lv && rv)
            return std::make_shared<ConstInt>(*lv - *rv, GetDType(op), op->span_);
        // x - 0 → x
        if (rv && *rv == 0)
            return l;
        if (l.get() != op->left_.get() || r.get() != op->right_.get())
            return std::make_shared<Sub>(l, r, GetDType(op), op->span_);
        return op;
    }

    ExprPtr VisitExpr_(const MulPtr& op) override
    {
        auto l = VisitExpr(op->left_);
        auto r = VisitExpr(op->right_);
        auto lv = TryGetConstInt(l), rv = TryGetConstInt(r);
        if (lv && rv)
            return std::make_shared<ConstInt>(*lv * *rv, GetDType(op), op->span_);
        // x * 1 → x
        if (rv && *rv == 1)
            return l;
        if (lv && *lv == 1)
            return r;
        // x * 0 → 0
        if ((rv && *rv == 0) || (lv && *lv == 0))
            return std::make_shared<ConstInt>(0, GetDType(op), op->span_);
        if (l.get() != op->left_.get() || r.get() != op->right_.get())
            return std::make_shared<Mul>(l, r, GetDType(op), op->span_);
        return op;
    }

    ExprPtr VisitExpr_(const FloorDivPtr& op) override
    {
        auto l = VisitExpr(op->left_);
        auto r = VisitExpr(op->right_);
        auto lv = TryGetConstInt(l), rv = TryGetConstInt(r);
        if (lv && rv && *rv != 0)
            return std::make_shared<ConstInt>(*lv / *rv, GetDType(op), op->span_);
        if (l.get() != op->left_.get() || r.get() != op->right_.get())
            return std::make_shared<FloorDiv>(l, r, GetDType(op), op->span_);
        return op;
    }

    ExprPtr VisitExpr_(const FloorModPtr& op) override
    {
        auto l = VisitExpr(op->left_);
        auto r = VisitExpr(op->right_);
        auto lv = TryGetConstInt(l), rv = TryGetConstInt(r);
        if (lv && rv && *rv != 0)
            return std::make_shared<ConstInt>(*lv % *rv, GetDType(op), op->span_);
        if (l.get() != op->left_.get() || r.get() != op->right_.get())
            return std::make_shared<FloorMod>(l, r, GetDType(op), op->span_);
        return op;
    }

    // ---- Constant fold comparisons ----
    ExprPtr VisitExpr_(const EqPtr& op) override
    {
        auto l = VisitExpr(op->left_);
        auto r = VisitExpr(op->right_);
        auto lv = TryGetConstInt(l), rv = TryGetConstInt(r);
        if (lv && rv)
            return std::make_shared<ConstInt>(*lv == *rv ? 1 : 0, DataType::BOOL, op->span_);
        if (l.get() != op->left_.get() || r.get() != op->right_.get())
            return std::make_shared<Eq>(l, r, GetDType(op), op->span_);
        return op;
    }

    ExprPtr VisitExpr_(const LtPtr& op) override
    {
        auto l = VisitExpr(op->left_);
        auto r = VisitExpr(op->right_);
        auto lv = TryGetConstInt(l), rv = TryGetConstInt(r);
        if (lv && rv)
            return std::make_shared<ConstInt>(*lv < *rv ? 1 : 0, DataType::BOOL, op->span_);
        if (l.get() != op->left_.get() || r.get() != op->right_.get())
            return std::make_shared<Lt>(l, r, GetDType(op), op->span_);
        return op;
    }

    ExprPtr VisitExpr_(const LePtr& op) override
    {
        auto l = VisitExpr(op->left_);
        auto r = VisitExpr(op->right_);
        auto lv = TryGetConstInt(l), rv = TryGetConstInt(r);
        if (lv && rv)
            return std::make_shared<ConstInt>(*lv <= *rv ? 1 : 0, DataType::BOOL, op->span_);
        if (l.get() != op->left_.get() || r.get() != op->right_.get())
            return std::make_shared<Le>(l, r, GetDType(op), op->span_);
        return op;
    }

    ExprPtr VisitExpr_(const GtPtr& op) override
    {
        auto l = VisitExpr(op->left_);
        auto r = VisitExpr(op->right_);
        auto lv = TryGetConstInt(l), rv = TryGetConstInt(r);
        if (lv && rv)
            return std::make_shared<ConstInt>(*lv > *rv ? 1 : 0, DataType::BOOL, op->span_);
        if (l.get() != op->left_.get() || r.get() != op->right_.get())
            return std::make_shared<Gt>(l, r, GetDType(op), op->span_);
        return op;
    }

    ExprPtr VisitExpr_(const GePtr& op) override
    {
        auto l = VisitExpr(op->left_);
        auto r = VisitExpr(op->right_);
        auto lv = TryGetConstInt(l), rv = TryGetConstInt(r);
        if (lv && rv)
            return std::make_shared<ConstInt>(*lv >= *rv ? 1 : 0, DataType::BOOL, op->span_);
        if (l.get() != op->left_.get() || r.get() != op->right_.get())
            return std::make_shared<Ge>(l, r, GetDType(op), op->span_);
        return op;
    }

    // ---- Constant fold unary Neg ----
    ExprPtr VisitExpr_(const NegPtr& op) override
    {
        auto operand = VisitExpr(op->operand_);
        auto ov = TryGetConstInt(operand);
        if (ov)
            return std::make_shared<ConstInt>(-*ov, GetDType(op), op->span_);
        if (operand.get() != op->operand_.get())
            return std::make_shared<Neg>(operand, GetDType(op), op->span_);
        return op;
    }

    StmtPtr MakeEmptySeq(const Span& span) { return std::make_shared<SeqStmts>(std::vector<StmtPtr>{}, span); }

    std::optional<StmtPtr> SimplifyConstCondition(
        const ExprPtr& cond, const StmtPtr& then_body, const std::optional<StmtPtr>& else_body, const Span& span)
    {
        auto const_value = TryGetConstInt(cond);
        if (!const_value) {
            return std::nullopt;
        }
        if (*const_value != 0) {
            return then_body;
        }
        if (else_body) {
            return *else_body;
        }
        return MakeEmptySeq(span);
    }

    std::vector<ExprPtr> ExtractYieldValues(const StmtPtr& body)
    {
        if (auto yield_stmt = As<YieldStmt>(body)) {
            return yield_stmt->value_;
        }

        auto seq = As<SeqStmts>(body);
        if (!seq) {
            return {};
        }
        if (seq->stmts_.size() == 1) {
            if (auto yield_stmt = As<YieldStmt>(seq->stmts_[0])) {
                return yield_stmt->value_;
            }
        }
        if (!seq->stmts_.empty()) {
            if (auto yield_stmt = As<YieldStmt>(seq->stmts_.back())) {
                return yield_stmt->value_;
            }
        }
        return {};
    }

    bool HaveSameYieldValues(
        const std::vector<ExprPtr>& then_vals, const std::vector<ExprPtr>& else_vals, size_t expected_count)
    {
        if (then_vals.size() != expected_count || else_vals.size() != expected_count) {
            return false;
        }
        for (size_t i = 0; i < then_vals.size(); ++i) {
            if (!IsSameExpr(then_vals[i], else_vals[i], var_vals_)) {
                return false;
            }
        }
        return true;
    }

    StmtPtr MakeAssignSeq(const std::vector<VarPtr>& return_vars, const std::vector<ExprPtr>& values, const Span& span)
    {
        std::vector<StmtPtr> assigns;
        assigns.reserve(return_vars.size());
        for (size_t i = 0; i < return_vars.size(); ++i) {
            assigns.push_back(std::make_shared<AssignStmt>(return_vars[i], values[i], span));
        }
        return std::make_shared<SeqStmts>(std::move(assigns), span);
    }

    std::optional<StmtPtr> TryFoldSameYieldBranches(
        const IfStmtPtr& op, const StmtPtr& then_body, const std::optional<StmtPtr>& else_body)
    {
        if (!else_body || op->returnVars_.empty()) {
            return std::nullopt;
        }

        auto then_vals = ExtractYieldValues(then_body);
        auto else_vals = ExtractYieldValues(*else_body);
        if (!HaveSameYieldValues(then_vals, else_vals, op->returnVars_.size())) {
            return std::nullopt;
        }
        return MakeAssignSeq(op->returnVars_, then_vals, op->span_);
    }

    // ---- If-stmt simplification ----
    StmtPtr VisitStmt_(const IfStmtPtr& op) override
    {
        auto cond = VisitExpr(op->condition_);
        auto then_body = VisitStmt(op->thenBody_);
        std::optional<StmtPtr> else_body;
        if (op->elseBody_)
            else_body = VisitStmt(*op->elseBody_);

        auto const_simplified = SimplifyConstCondition(cond, then_body, else_body, op->span_);
        if (const_simplified) {
            return *const_simplified;
        }

        auto same_yield_folded = TryFoldSameYieldBranches(op, then_body, else_body);
        if (same_yield_folded) {
            return *same_yield_folded;
        }

        // Copy-on-write
        if (cond.get() != op->condition_.get() || then_body.get() != op->thenBody_.get() ||
            (else_body && else_body->get() != op->elseBody_->get())) {
            return std::make_shared<IfStmt>(cond, then_body, else_body, op->returnVars_, op->span_);
        }
        return op;
    }
};

// ---------------------------------------------------------------------------
// Pass factory
// ---------------------------------------------------------------------------
static FunctionPtr TransformConstFoldAndSimplify(const FunctionPtr& func)
{
    if (!func || !func->body_)
        return func;
    ConstFoldMutator mutator;
    auto new_body = mutator.VisitStmt(func->body_);
    if (new_body.get() == func->body_.get())
        return func;
    return std::make_shared<Function>(
        func->name_, func->params_, func->returnTypes_, new_body, func->span_, func->funcType_, func->entry_);
}

namespace pass {

Pass ConstFoldAndSimplify() { return CreateFunctionPass(TransformConstFoldAndSimplify, "ConstFoldAndSimplify"); }

} // namespace pass
} // namespace ir
} // namespace pypto
