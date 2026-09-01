/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file pass_token_utils.cpp
 * \brief Token dependency utilities for passes.
 */

#include "pass_token_utils.h"

#include <algorithm>
#include "interface/tensor/irbuilder.h"

namespace npu::tile_fwk {
namespace {

ir::StmtPtr AsStmtPtr(Operation& op) { return std::static_pointer_cast<const ir::Stmt>(op.shared_from_this()); }

void RemoveToken(Operation& op, const ir::VarPtr& token)
{
    if (token == nullptr) {
        return;
    }
    op.tokens_.erase(std::remove(op.tokens_.begin(), op.tokens_.end(), token), op.tokens_.end());
}

void MoveTokenConsumer(Function& function, const ir::VarPtr& token, Operation& oldConsumer, Operation& newConsumer)
{
    if (token == nullptr) {
        return;
    }
    auto& dependency = function.GetVarDependency();
    dependency.RemoveConsumer(token, AsStmtPtr(oldConsumer));
    dependency.AddConsumer(token, AsStmtPtr(newConsumer));
    PassTokenUtils::AddTokenIfAbsent(newConsumer, token);
}

std::vector<ir::StmtPtr> GetConsumers(Function& function, const ir::VarPtr& token)
{
    const auto& consumers = function.GetVarDependency().GetConsumers(token);
    return {consumers.begin(), consumers.end()};
}

ir::VarPtr MoveTokenProducer(Function& function, const ir::VarPtr& oldToken,
                             const std::vector<ir::StmtPtr>& oldConsumers, Operation& newProducer)
{
    if (oldToken == nullptr) {
        return nullptr;
    }
    auto& dependency = function.GetVarDependency();
    ir::VarPtr newToken = nullptr;
    if (!newProducer.result_token_.empty()) {
        newToken = newProducer.result_token_.front();
    } else {
        newToken = IRBuilder().CreateTokenVar(newProducer.GetSpan());
        newProducer.result_token_.push_back(newToken);
    }
    auto newProducerStmt = AsStmtPtr(newProducer);

    dependency.AddProducer(newToken, newProducerStmt);
    for (const auto& consumerStmt : oldConsumers) {
        if (consumerStmt == newProducerStmt) {
            continue;
        }
        dependency.AddConsumer(newToken, consumerStmt);
        auto* consumerOp = static_cast<Operation*>(const_cast<ir::Stmt*>(consumerStmt.get()));
        PassTokenUtils::AddTokenIfAbsent(*consumerOp, newToken);
    }
    return newToken;
}

} // namespace

void PassTokenUtils::AddTokenIfAbsent(Operation& op, const ir::VarPtr& token)
{
    if (token == nullptr || std::find(op.tokens_.begin(), op.tokens_.end(), token) != op.tokens_.end()) {
        return;
    }
    op.tokens_.push_back(token);
}

void PassTokenUtils::MoveTokenDependencyBeforeRemoveOp(Function& function, Operation& op)
{
    auto producers = op.ProducerOps();
    auto consumers = op.ConsumerOps();
    auto inputTokens = op.tokens_;
    for (auto* consumer : consumers) {
        if (consumer == nullptr || consumer == &op) {
            continue;
        }
        for (const auto& token : inputTokens) {
            MoveTokenConsumer(function, token, op, *consumer);
        }
    }
    for (const auto& token : inputTokens) {
        function.GetVarDependency().RemoveConsumer(token, AsStmtPtr(op));
    }
    op.tokens_.clear();

    auto resultTokens = op.result_token_;
    for (const auto& resultToken : resultTokens) {
        auto resultTokenConsumers = GetConsumers(function, resultToken);
        for (auto* producer : producers) {
            if (producer == nullptr || producer == &op) {
                continue;
            }
            MoveTokenProducer(function, resultToken, resultTokenConsumers, *producer);
        }
        for (const auto& consumerStmt : resultTokenConsumers) {
            function.GetVarDependency().RemoveConsumer(resultToken, consumerStmt);
            auto* consumerOp = static_cast<Operation*>(const_cast<ir::Stmt*>(consumerStmt.get()));
            RemoveToken(*consumerOp, resultToken);
        }
        function.GetVarDependency().RemoveProducer(resultToken, AsStmtPtr(op));
        function.GetVarDependency().RemoveVar(resultToken);
    }
    op.result_token_.clear();
}

void PassTokenUtils::AddTokenConsumer(Function& function, const ir::VarPtr& token, Operation& consumerOp)
{
    if (token == nullptr) {
        return;
    }
    function.GetVarDependency().AddConsumer(token, AsStmtPtr(consumerOp));
    AddTokenIfAbsent(consumerOp, token);
}

void PassTokenUtils::CopyTokenDependency(Function& function, Operation& originOp, Operation& copiedOp)
{
    auto& dependency = function.GetVarDependency();
    auto copiedStmt = AsStmtPtr(copiedOp);
    for (const auto& token : copiedOp.tokens_) {
        dependency.RemoveConsumer(token, copiedStmt);
    }
    for (const auto& token : copiedOp.result_token_) {
        if (std::find(originOp.result_token_.begin(), originOp.result_token_.end(), token) !=
            originOp.result_token_.end()) {
            continue;
        }
        auto oldConsumers = GetConsumers(function, token);
        dependency.RemoveProducer(token, copiedStmt);
        for (const auto& consumerStmt : oldConsumers) {
            dependency.RemoveConsumer(token, consumerStmt);
            auto* consumerOp = static_cast<Operation*>(const_cast<ir::Stmt*>(consumerStmt.get()));
            RemoveToken(*consumerOp, token);
        }
    }

    copiedOp.tokens_.clear();

    for (const auto& token : originOp.tokens_) {
        AddTokenIfAbsent(copiedOp, token);
        dependency.AddConsumer(token, copiedStmt);
    }

    std::vector<ir::VarPtr> newResultTokens;
    for (const auto& originToken : originOp.result_token_) {
        auto newToken = IRBuilder().CreateTokenVar(copiedOp.GetSpan());
        newResultTokens.push_back(newToken);
        dependency.AddProducer(newToken, copiedStmt);
        for (const auto& consumerStmt : dependency.GetConsumers(originToken)) {
            dependency.AddConsumer(newToken, consumerStmt);
            auto* consumerOp = static_cast<Operation*>(const_cast<ir::Stmt*>(consumerStmt.get()));
            AddTokenIfAbsent(*consumerOp, newToken);
        }
    }
    copiedOp.result_token_ = std::move(newResultTokens);
}

void PassTokenUtils::MoveResultTokensToProducers(Function& function, const std::vector<Operation*>& sourceOps,
                                                 const std::vector<Operation*>& targetProducerOps,
                                                 const std::unordered_set<ir::StmtPtr>& skippedConsumers)
{
    auto& dependency = function.GetVarDependency();
    for (auto* sourceOp : sourceOps) {
        if (sourceOp == nullptr || sourceOp->result_token_.empty()) {
            continue;
        }
        auto oldTokens = sourceOp->result_token_;
        for (const auto& oldToken : oldTokens) {
            auto oldConsumers = GetConsumers(function, oldToken);
            std::vector<ir::StmtPtr> movedConsumers;
            for (const auto& consumerStmt : oldConsumers) {
                if (skippedConsumers.count(consumerStmt) == 0) {
                    movedConsumers.push_back(consumerStmt);
                }
            }
            for (auto* targetProducer : targetProducerOps) {
                if (targetProducer == nullptr || targetProducer == sourceOp || movedConsumers.empty()) {
                    continue;
                }
                MoveTokenProducer(function, oldToken, movedConsumers, *targetProducer);
            }
            for (const auto& consumerStmt : oldConsumers) {
                dependency.RemoveConsumer(oldToken, consumerStmt);
                auto* consumerOp = static_cast<Operation*>(const_cast<ir::Stmt*>(consumerStmt.get()));
                RemoveToken(*consumerOp, oldToken);
            }
            dependency.RemoveProducer(oldToken, AsStmtPtr(*sourceOp));
            dependency.RemoveVar(oldToken);
        }
        sourceOp->result_token_.clear();
    }
}

void PassTokenUtils::CleanupDeletedTokenDependency(Function& function, const std::vector<Operation*>& deletedOps)
{
    auto& dependency = function.GetVarDependency();
    std::unordered_set<ir::StmtPtr> deletedStmts;
    for (auto* op : deletedOps) {
        if (op == nullptr) {
            continue;
        }
        deletedStmts.insert(AsStmtPtr(*op));
    }
    if (deletedStmts.empty()) {
        return;
    }
    auto dependencies = dependency.GetAllDependencies();
    for (const auto& [token, entry] : dependencies) {
        for (const auto& producer : entry.producers) {
            if (deletedStmts.count(producer) != 0) {
                dependency.RemoveProducer(token, producer);
            }
        }
        for (const auto& consumer : entry.consumers) {
            if (deletedStmts.count(consumer) == 0) {
                continue;
            }
            dependency.RemoveConsumer(token, consumer);
            auto* consumerOp = static_cast<Operation*>(const_cast<ir::Stmt*>(consumer.get()));
            RemoveToken(*consumerOp, token);
        }
        if (dependency.GetProducers(token).empty() && dependency.GetConsumers(token).empty()) {
            dependency.RemoveVar(token);
        }
    }
    for (auto* op : deletedOps) {
        if (op == nullptr) {
            continue;
        }
        op->tokens_.clear();
        op->result_token_.clear();
    }
}

} // namespace npu::tile_fwk
