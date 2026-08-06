/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * -----------------------------------------------------------------------------------------------------------
 */

#include "stmt_utils.h"

#include <memory>
#include <unordered_set>
#include <vector>

#include "ir/expr.h"
#include "ir/stmt.h"
#include "ir/transforms/base/visitor.h"
#include "interface/operation/operation.h"

namespace pypto {
namespace ir {
namespace utils {

namespace {

using npu::tile_fwk::Operation;

class VarUseCollector : public IRVisitor {
public:
    std::unordered_set<const Var*> var_uses;

    // When set, skip Yield/Break/Continue statements (and their carried-value
    // expressions) at any nesting depth: a terminator only forwards loop
    // carries, it is not a real use of the carried variable.
    explicit VarUseCollector(bool skip_iter_updates = false) : skip_iter_updates_(skip_iter_updates) {}

private:
    using IRVisitor::VisitExpr_;
    using IRVisitor::VisitStmt_;
    void VisitExpr_(const VarPtr& op) override
    {
        var_uses.insert(op.get());
        IRVisitor::VisitExpr_(op);
    }
    void VisitExpr_(const MemRefPtr& op) override
    {
        var_uses.insert(op.get());
        IRVisitor::VisitExpr_(op);
    }
    void VisitStmt_(const YieldStmtPtr& op) override
    {
        if (!skip_iter_updates_) {
            IRVisitor::VisitStmt_(op);
        }
    }
    void VisitStmt_(const BreakStmtPtr& op) override
    {
        if (!skip_iter_updates_) {
            IRVisitor::VisitStmt_(op);
        }
    }
    void VisitStmt_(const ContinueStmtPtr& op) override
    {
        if (!skip_iter_updates_) {
            IRVisitor::VisitStmt_(op);
        }
    }

    void VisitStmt_(const TensorOpStmtPtr& op) override
    {
        IRVisitor::VisitStmt_(op);

        auto operation = std::dynamic_pointer_cast<Operation>(std::const_pointer_cast<TensorOpStmt>(op));
        if (operation) {
            for (auto& attr : operation->GetDynamicAttributeList()) {
                attr.get().GetVarRefs(var_uses);
            }
        }
    }

    bool skip_iter_updates_;
};

} // namespace

std::unordered_set<const Var*> CollectVarUses(const ExprPtr& expr)
{
    VarUseCollector collector;
    if (expr)
        collector.VisitExpr(expr);
    return std::move(collector.var_uses);
}

std::unordered_set<const Var*> CollectStmtVarRefs(const StmtPtr& stmt)
{
    VarUseCollector collector;
    collector.VisitStmt(stmt);
    return std::move(collector.var_uses);
}

std::unordered_set<const Var*> CollectStmtVarRefs(const std::vector<StmtPtr>& stmts, bool skip_iter_updates)
{
    VarUseCollector collector(skip_iter_updates);
    for (const auto& s : stmts) {
        collector.VisitStmt(s);
    }
    return std::move(collector.var_uses);
}

const std::vector<StmtPtr>& FlattenBody(const SeqStmtsPtr& body)
{
    static const std::vector<StmtPtr> empty;
    if (!body)
        return empty;
    return body->stmts_;
}

bool StmtsEqual(const std::vector<StmtPtr>& a, const std::vector<StmtPtr>& b)
{
    if (a.size() != b.size())
        return false;
    for (size_t i = 0; i < a.size(); ++i) {
        if (a[i].get() != b[i].get())
            return false;
    }
    return true;
}

SeqStmtsPtr MakeSeqBody(const std::vector<StmtPtr>& stmts, const Span& span)
{
    return std::make_shared<SeqStmts>(stmts, span);
}
} // namespace utils
} // namespace ir
} // namespace pypto
