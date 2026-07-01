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

#include "ir/transforms/utils/canonicalize.h"

#include <memory>
#include <unordered_set>
#include <utility>
#include <vector>

#include "ir/expr.h"
#include "ir/stmt.h"
#include "stmt_utils.h"

namespace pypto {
namespace ir {

namespace {

using utils::CollectStmtVarRefs;
using utils::StmtsEqual;

std::vector<ExprPtr> MakeReturnVars(const std::vector<ExprPtr>& values, const std::vector<size_t>& kept_indices)
{
    std::vector<ExprPtr> result;
    for (size_t idx : kept_indices) {
        INTERNAL_CHECK(idx < values.size());
        result.push_back(values[idx]);
    }
    return result;
}

void FilterTrailingTerminator(std::vector<StmtPtr>& stmts, const std::vector<size_t>& keptIndices)
{
    auto stmt = stmts.back();
    if (auto yield = std::dynamic_pointer_cast<const YieldStmt>(stmt)) {
        stmts.back() = std::make_shared<YieldStmt>(MakeReturnVars(yield->value_, keptIndices), yield->span_);
    } else if (auto brk = std::dynamic_pointer_cast<const BreakStmt>(stmt)) {
        stmts.back() = std::make_shared<BreakStmt>(MakeReturnVars(brk->value_, keptIndices), brk->span_);
    } else if (auto cont = std::dynamic_pointer_cast<const ContinueStmt>(stmt)) {
        stmts.back() = std::make_shared<ContinueStmt>(MakeReturnVars(cont->value_, keptIndices), cont->span_);
    }
}

template <typename T, typename Rebuild>
StmtPtr CanonicalizeLoopImpl(const T& stmt, const std::unordered_set<const Var*>& afterRefs, Rebuild rebuild)
{
    if (stmt->iterArgs_.empty()) {
        auto body = CanonicalizeSeqStmts(stmt->body_, {});
        if (body == stmt->body_)
            return stmt;
        return rebuild(stmt->iterArgs_, body, stmt->returnVars_);
    }

    auto refs = CollectStmtVarRefs(stmt->body_->stmts_, /*skip_iter_updates=*/true);
    std::vector<size_t> keptIndices;
    for (size_t i = 0; i < stmt->returnVars_.size(); ++i) {
        if (afterRefs.count(stmt->returnVars_[i].get())) {
            keptIndices.push_back(i);
        }
    }
    if (keptIndices.size() == stmt->iterArgs_.size()) {
        return stmt;
    }

    auto body = CanonicalizeSeqStmts(stmt->body_, keptIndices);
    std::vector<IterArgPtr> iterArgs;
    std::vector<VarPtr> returnVars;
    for (size_t i : keptIndices) {
        iterArgs.push_back(stmt->iterArgs_[i]);
        returnVars.push_back(stmt->returnVars_[i]);
    }
    return rebuild(iterArgs, body, returnVars);
}

StmtPtr CanonicalizeIfStmt(IfStmtPtr& ifStmt, const std::unordered_set<const Var*>& suffix)
{
    std::vector<size_t> keptIndices;
    std::vector<VarPtr> returnVars;
    for (size_t i = 0; i < ifStmt->returnVars_.size(); ++i) {
        if (suffix.count(ifStmt->returnVars_[i].get()) > 0) {
            keptIndices.push_back(i);
            returnVars.push_back(ifStmt->returnVars_[i]);
        }
    }

    if (returnVars.size() == ifStmt->returnVars_.size()) {
        return ifStmt;
    }

    std::optional<SeqStmtsPtr> elseBody;
    auto thenBody = CanonicalizeSeqStmts(ifStmt->thenBody_, keptIndices);
    if (ifStmt->elseBody_) {
        elseBody = CanonicalizeSeqStmts(ifStmt->elseBody_.value(), keptIndices);
    }
    // Always rebuild when returnVars changed
    return std::make_shared<const IfStmt>(ifStmt->condition_, thenBody, elseBody, std::move(returnVars), ifStmt->span_);
}

StmtPtr CanonicalizeSectionStmt(SectionStmtPtr& section)
{
    auto newBody = CanonicalizeSeqStmts(section->body_, {});
    if (newBody == section->body_) {
        return section;
    }
    return std::make_shared<const SectionStmt>(section->sectionKind_, newBody, section->span_);
}
} // namespace

SeqStmtsPtr CanonicalizeSeqStmts(const SeqStmtsPtr& seq, const std::vector<size_t>& keptIndices)
{
    if (seq->stmts_.empty())
        return seq;

    auto& stmts = seq->stmts_;

    // the variable used after the block, could only be the returnVars of the block
    // first handle the returnVars in terminate statement
    FilterTrailingTerminator(stmts, keptIndices);

    std::vector<StmtPtr> result(stmts.size());
    std::unordered_set<const Var*> suffix;
    for (size_t idx = stmts.size(); idx-- > 0;) {
        if (auto forStmt = std::dynamic_pointer_cast<const ForStmt>(stmts[idx])) {
            result[idx] = CanonicalizeLoopImpl(
                forStmt, suffix,
                [&forStmt](const std::vector<IterArgPtr>& ia, const SeqStmtsPtr& body, const std::vector<VarPtr>& rv) {
                    return std::make_shared<const ForStmt>(
                        forStmt->loopVar_, forStmt->start_, forStmt->stop_, forStmt->step_, ia, body, rv,
                        forStmt->span_);
                });
        } else if (auto whileStmt = std::dynamic_pointer_cast<const WhileStmt>(stmts[idx])) {
            result[idx] = CanonicalizeLoopImpl(
                whileStmt, suffix,
                [whileStmt](const std::vector<IterArgPtr>& ia, const SeqStmtsPtr& body, const std::vector<VarPtr>& rv) {
                    return std::make_shared<const WhileStmt>(whileStmt->condition_, ia, body, rv, whileStmt->span_);
                });
        } else if (auto if_stmt = std::dynamic_pointer_cast<const IfStmt>(stmts[idx])) {
            result[idx] = CanonicalizeIfStmt(if_stmt, suffix);
        } else if (auto section = std::dynamic_pointer_cast<const SectionStmt>(stmts[idx])) {
            result[idx] = CanonicalizeSectionStmt(section);
        } else {
            result[idx] = stmts[idx];
        }
        auto refs = CollectStmtVarRefs(result[idx]);
        suffix.insert(refs.begin(), refs.end());
    }

    if (StmtsEqual(result, seq->stmts_)) {
        return seq;
    }
    return std::make_shared<SeqStmts>(result, seq->span_);
}

} // namespace ir
} // namespace pypto
