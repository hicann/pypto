/*
 * Copyright (c) PyPTO Contributors.
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

namespace pypto {
namespace ir {
namespace utils {

namespace {

class VarUseCollector : public IRVisitor {
public:
    std::unordered_set<const Var*> var_uses;

private:
    using IRVisitor::VisitExpr_;
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
};

}  // namespace

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
    VarUseCollector collector;
    for (const auto& s : stmts) {
        if (skip_iter_updates
            && (std::dynamic_pointer_cast<const YieldStmt>(s)
                || std::dynamic_pointer_cast<const BreakStmt>(s)
                || std::dynamic_pointer_cast<const ContinueStmt>(s)))
            continue;
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
}  // namespace utils
}  // namespace ir
}  // namespace pypto
