/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "ir/transforms/infer_token_pass.h"
#include "ir/transforms/base/mutator.h"

#include <unordered_set>

#include "ir/expr.h"
#include "ir/kind_traits.h"
#include "ir/type.h"

namespace pypto::ir {

using VarTokenMap = std::unordered_map<VarPtr, VarPtr>;

thread_local int tokenCounter = 0;

class InferTokenMutator : public IRMutator {
public:
    using IRMutator::VisitExpr_;
    using IRMutator::VisitStmt_;

    StmtPtr Apply(StmtPtr stmt)
    {
        producers_.clear();
        pendingUpdates_.clear();
        return VisitStmt(stmt);
    }

private:
    VarTokenMap producers_;
    std::vector<std::pair<VarPtr, VarPtr>> pendingUpdates_;

    VarPtr CreateTokenVar(Span span, const std::string& name = "")
    {
        return std::make_shared<Var>("_" + name + "_token_" + std::to_string(++tokenCounter), GetTokenType(), span);
    }

    VarTokenMap BuildResultTokenMap(SeqStmtsPtr body)
    {
        VarTokenMap map;
        if (!body) {
            return map;
        }
        for (auto& stmt : body->stmts_) {
            if (stmt->GetKind() != ObjectKind::TensorOpStmt) {
                continue;
            }
            auto t = std::static_pointer_cast<const TensorOpStmt>(stmt);
            if (!t->result_token_) {
                continue;
            }
            for (auto& v : t->result_) {
                if (v) {
                    map[v] = t->result_token_;
                }
            }
        }
        return map;
    }

    std::vector<VarPtr> CollectBranchOutputTokens(SeqStmtsPtr body)
    {
        std::vector<VarPtr> tokens;
        if (!body || body->stmts_.empty()) {
            return tokens;
        }
        auto resultTokenMap = BuildResultTokenMap(body);
        auto& lastStmt = body->stmts_.back();
        if (lastStmt->GetKind() != ObjectKind::YieldStmt) {
            return tokens;
        }
        auto yieldStmt = std::static_pointer_cast<const YieldStmt>(lastStmt);
        std::unordered_set<VarPtr> seen;
        for (auto& val : yieldStmt->value_) {
            auto var = std::dynamic_pointer_cast<const Var>(val);
            if (!var) {
                continue;
            }
            auto it = resultTokenMap.find(var);
            if (it != resultTokenMap.end() && !seen.count(it->second)) {
                seen.insert(it->second);
                tokens.push_back(it->second);
            }
        }
        return tokens;
    }

    SeqStmtsPtr ExtendYieldWithTokens(SeqStmtsPtr body, const std::vector<VarPtr>& tokens)
    {
        if (!body || body->stmts_.empty() || tokens.empty()) {
            return body;
        }
        auto& lastStmt = body->stmts_.back();
        if (lastStmt->GetKind() != ObjectKind::YieldStmt) {
            return body;
        }
        auto yieldStmt = std::static_pointer_cast<const YieldStmt>(lastStmt);
        std::vector<ExprPtr> newValues = yieldStmt->value_;
        for (auto& tk : tokens) {
            newValues.push_back(tk);
        }
        std::vector<StmtPtr> newStmts = body->stmts_;
        newStmts.back() = std::make_shared<YieldStmt>(newValues, body->span_);
        return std::make_shared<SeqStmts>(newStmts, body->span_);
    }

    std::vector<size_t> BuildReturnVarToTokenIndex(
        SeqStmtsPtr body, size_t originalReturnVarCount, const std::vector<VarPtr>& branchTokens)
    {
        std::vector<size_t> mapping(originalReturnVarCount, SIZE_MAX);
        if (!body || body->stmts_.empty() || branchTokens.empty()) {
            return mapping;
        }
        auto resultTokenMap = BuildResultTokenMap(body);
        auto& lastStmt = body->stmts_.back();
        if (lastStmt->GetKind() != ObjectKind::YieldStmt) {
            return mapping;
        }
        auto yieldStmt = std::static_pointer_cast<const YieldStmt>(lastStmt);
        for (size_t i = 0; i < originalReturnVarCount && i < yieldStmt->value_.size(); ++i) {
            auto var = std::dynamic_pointer_cast<const Var>(yieldStmt->value_[i]);
            if (!var) {
                continue;
            }
            auto rtIt = resultTokenMap.find(var);
            if (rtIt == resultTokenMap.end()) {
                continue;
            }
            for (size_t j = 0; j < branchTokens.size(); ++j) {
                if (branchTokens[j] == rtIt->second) {
                    mapping[i] = j;
                    break;
                }
            }
        }
        return mapping;
    }

    void UpdateProducers(const std::vector<VarPtr>& results, VarPtr token)
    {
        for (auto& v : results) {
            if (v) {
                producers_[v] = token;
            }
        }
    }

    void QueueUpdate(const VarTokenMap& mappings)
    {
        for (auto& [var, token] : mappings) {
            pendingUpdates_.emplace_back(var, token);
        }
    }

    void FlushUpdates()
    {
        for (auto& update : pendingUpdates_) {
            producers_[update.first] = update.second;
        }
        pendingUpdates_.clear();
    }

    // ========== IRMutator 访问方法 ==========
    StmtPtr VisitStmt_(const SeqStmtsPtr& op) override
    {
        std::vector<StmtPtr> newStmts;
        newStmts.reserve(op->stmts_.size());

        for (auto& stmt : op->stmts_) {
            FlushUpdates();
            StmtPtr newStmt = VisitStmt(stmt);
            newStmts.push_back(newStmt);
        }

        FlushUpdates();
        return std::make_shared<SeqStmts>(newStmts, op->span_);
    }

    StmtPtr VisitStmt_(const TensorOpStmtPtr& op) override
    {
        VarPtr resultToken = op->result_token_ ? op->result_token_ : CreateTokenVar(op->span_, op->opcode_);

        std::vector<VarPtr> newTokens = op->tokens_;
        std::unordered_set<VarPtr> seen(op->tokens_.begin(), op->tokens_.end());

        for (auto& arg : op->args_) {
            auto var = std::dynamic_pointer_cast<const Var>(arg);
            if (!var) {
                continue;
            }
            auto it = producers_.find(var);
            if (it != producers_.end() && !seen.count(it->second)) {
                seen.insert(it->second);
                newTokens.push_back(it->second);
            }
        }

        if (op->opcode_ == "ASSEMBLE" || op->opcode_ == "ASSEMBLE_SSA") {
            for (auto& res : op->result_) {
                if (!res) {
                    continue;
                }
                auto it = producers_.find(res);
                if (it != producers_.end() && !seen.count(it->second)) {
                    seen.insert(it->second);
                    newTokens.push_back(it->second);
                }
            }
        }

        auto newStmt = std::make_shared<TensorOpStmt>(
            op->result_, resultToken, op->opcode_, op->args_, newTokens, op->attrs_, op->span_);

        UpdateProducers(op->result_, resultToken);
        return newStmt;
    }

    StmtPtr VisitStmt_(const IfStmtPtr& op) override
    {
        auto savedProducers = producers_;

        auto processedThen = SeqStmts::AsMut(VisitStmt(op->thenBody_));
        auto thenProducers = producers_;

        producers_ = savedProducers;
        std::optional<SeqStmtsPtr> processedElse;
        if (op->elseBody_) {
            processedElse = SeqStmts::AsMut(VisitStmt(op->elseBody_.value()));
        }

        size_t originalReturnVarCount = op->returnVars_.size();
        auto thenTokens = CollectBranchOutputTokens(processedThen);
        auto elseTokens = processedElse ? CollectBranchOutputTokens(processedElse.value()) : std::vector<VarPtr>{};

        size_t tokenCount = std::max(thenTokens.size(), elseTokens.size());
        std::vector<VarPtr> phiTokens;
        for (size_t i = 0; i < tokenCount; ++i) {
            phiTokens.push_back(CreateTokenVar(op->span_));
        }

        std::vector<VarPtr> thenYieldTokens = thenTokens;
        std::vector<VarPtr> elseYieldTokens = elseTokens;
        while (thenYieldTokens.size() < tokenCount) {
            thenYieldTokens.push_back(CreateTokenVar(op->span_));
        }
        while (elseYieldTokens.size() < tokenCount) {
            elseYieldTokens.push_back(CreateTokenVar(op->span_));
        }

        auto newThenBody = ExtendYieldWithTokens(processedThen, thenYieldTokens);
        std::optional<SeqStmtsPtr> newElseBody;
        if (processedElse) {
            newElseBody = ExtendYieldWithTokens(processedElse.value(), elseYieldTokens);
        }

        std::vector<VarPtr> newReturnVars = op->returnVars_;
        for (auto& phiToken : phiTokens) {
            newReturnVars.push_back(phiToken);
        }

        VarTokenMap returnVarToPhiToken;
        auto thenMapping = BuildReturnVarToTokenIndex(processedThen, originalReturnVarCount, thenTokens);

        for (size_t i = 0; i < originalReturnVarCount; ++i) {
            if (thenMapping[i] != SIZE_MAX && thenMapping[i] < phiTokens.size()) {
                returnVarToPhiToken[op->returnVars_[i]] = phiTokens[thenMapping[i]];
            }
        }

        QueueUpdate(returnVarToPhiToken);

        return std::make_shared<IfStmt>(op->condition_, newThenBody, newElseBody, newReturnVars, op->span_);
    }

    StmtPtr VisitStmt_(const ForStmtPtr& op) override
    {
        auto savedProducers = producers_;

        auto processedBody = SeqStmts::AsMut(VisitStmt(op->body_));

        size_t originalReturnVarCount = op->returnVars_.size();
        auto bodyTokens = CollectBranchOutputTokens(processedBody);

        std::vector<VarPtr> carryTokens;
        for (size_t i = 0; i < bodyTokens.size(); ++i) {
            carryTokens.push_back(CreateTokenVar(op->span_));
        }

        std::vector<VarPtr> newReturnVars = op->returnVars_;
        for (auto& carryToken : carryTokens) {
            newReturnVars.push_back(carryToken);
        }

        VarTokenMap returnVarToCarryToken;
        auto bodyMapping = BuildReturnVarToTokenIndex(processedBody, originalReturnVarCount, bodyTokens);

        for (size_t i = 0; i < originalReturnVarCount; ++i) {
            if (bodyMapping[i] != SIZE_MAX && bodyMapping[i] < carryTokens.size()) {
                returnVarToCarryToken[op->returnVars_[i]] = carryTokens[bodyMapping[i]];
            }
        }

        QueueUpdate(returnVarToCarryToken);
        producers_ = savedProducers;

        return std::make_shared<ForStmt>(
            op->loopVar_, op->start_, op->stop_, op->step_, op->iterArgs_, processedBody, newReturnVars, op->span_);
    }
};

SeqStmtsPtr InferTokenPass(SeqStmtsPtr seq)
{
    InferTokenMutator mutator;
    return SeqStmts::AsMut(mutator.Apply(seq));
}

} // namespace pypto::ir