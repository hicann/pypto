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
 * \file merge_view_assemble_utils.cpp
 * \brief utils of view and assemble operation merging
 */

#include "merge_view_assemble_utils.h"
#include <algorithm>
#include <functional>
#include <limits>
#include <optional>
#include <queue>
#include <unordered_set>
#include "interface/tensor/irbuilder.h"
#include "interface/operation/attribute.h"
#include "passes/pass_utils/dead_operation_eliminate.h"
#include "passes/pass_utils/infer_shape_utils.h"
#include "passes/pass_utils/pass_attr_defs.h"
#include "passes/pass_utils/pass_utils.h"
#include "passes/pass_log/pass_log.h"
#include "tilefwk/tilefwk_op.h"

#define MODULE_NAME "MergeViewAssembleUtils"

namespace npu::tile_fwk {
namespace {
ir::StmtPtr ToStmtPtr(Operation& op) { return std::static_pointer_cast<const ir::Stmt>(op.shared_from_this()); }

Operation* ToOperation(const ir::StmtPtr& stmt) { return static_cast<Operation*>(const_cast<ir::Stmt*>(stmt.get())); }

template <typename T>
void AddUnique(std::vector<T>& values, const T& value)
{
    if (value == nullptr || std::find(values.begin(), values.end(), value) != values.end()) {
        return;
    }
    values.emplace_back(value);
}

void RemoveToken(Operation& op, const ir::VarPtr& token)
{
    op.tokens_.erase(std::remove(op.tokens_.begin(), op.tokens_.end(), token), op.tokens_.end());
}

void AddTokenConsumer(Function& function, const ir::VarPtr& token, Operation& consumer)
{
    if (token == nullptr) {
        return;
    }
    function.GetVarDependency().AddConsumer(token, ToStmtPtr(consumer));
    AddUnique(consumer.tokens_, token);
}

struct TokenSnapshot {
    ir::VarPtr token;
    Operation* producer = nullptr;
    std::vector<Operation*> consumers;
};

std::vector<TokenSnapshot> CollectTokenSnapshots(Function& function, const std::unordered_set<Operation*>& affected)
{
    std::vector<ir::VarPtr> tokens;
    for (auto* op : affected) {
        if (op == nullptr) {
            continue;
        }
        for (const auto& token : op->result_token_) {
            AddUnique(tokens, token);
        }
        for (const auto& token : op->tokens_) {
            AddUnique(tokens, token);
        }
    }
    for (const auto& [token, entry] : function.GetVarDependency().GetAllDependencies()) {
        bool touchesAffected = false;
        for (const auto& producer : entry.producers) {
            if (affected.count(ToOperation(producer)) != 0) {
                touchesAffected = true;
                break;
            }
        }
        if (!touchesAffected) {
            for (const auto& consumer : entry.consumers) {
                if (affected.count(ToOperation(consumer)) != 0) {
                    touchesAffected = true;
                    break;
                }
            }
        }
        if (touchesAffected) {
            AddUnique(tokens, token);
        }
    }

    std::vector<TokenSnapshot> snapshots;
    snapshots.reserve(tokens.size());
    for (const auto& token : tokens) {
        TokenSnapshot snapshot;
        snapshot.token = token;
        const auto& producers = function.GetVarDependency().GetProducers(token);
        if (!producers.empty()) {
            snapshot.producer = ToOperation(*producers.begin());
        } else {
            for (auto* op : affected) {
                if (op != nullptr &&
                    std::find(op->result_token_.begin(), op->result_token_.end(), token) != op->result_token_.end()) {
                    snapshot.producer = op;
                    break;
                }
            }
        }
        for (const auto& consumer : function.GetVarDependency().GetConsumers(token)) {
            AddUnique(snapshot.consumers, ToOperation(consumer));
        }
        for (auto* op : affected) {
            if (op != nullptr && std::find(op->tokens_.begin(), op->tokens_.end(), token) != op->tokens_.end()) {
                AddUnique(snapshot.consumers, op);
            }
        }
        snapshots.emplace_back(std::move(snapshot));
    }
    return snapshots;
}

bool HasTokenDependency(const std::vector<Operation*>& operations)
{
    return std::any_of(operations.begin(), operations.end(), [](const Operation* op) {
        return op != nullptr && (!op->result_token_.empty() || !op->tokens_.empty());
    });
}

bool IsFullTensorAssemble(const Operation& operation)
{
    return operation.GetOpcode() == Opcode::OP_ASSEMBLE && operation.GetIOperands().size() == 1 &&
           operation.GetOOperands().size() == 1 && operation.GetIOperands().front() != nullptr &&
           operation.GetOOperands().front() != nullptr &&
           operation.GetIOperands().front()->GetShape() == operation.GetOOperands().front()->GetShape();
}

void AddDependencyEdge(std::unordered_map<Operation*, std::unordered_set<Operation*>>& adjacency, Operation* producer,
                       Operation* consumer)
{
    if (producer == nullptr || consumer == nullptr || producer == consumer) {
        return;
    }
    adjacency[producer].insert(consumer);
    adjacency.try_emplace(consumer);
}

bool HasDependencyCycle(const std::unordered_map<Operation*, std::unordered_set<Operation*>>& adjacency)
{
    std::unordered_map<Operation*, size_t> indegree;
    indegree.reserve(adjacency.size());
    for (const auto& [op, consumers] : adjacency) {
        indegree.try_emplace(op, 0);
        for (auto* consumer : consumers) {
            ++indegree[consumer];
        }
    }
    std::queue<Operation*> ready;
    for (const auto& [op, degree] : indegree) {
        if (degree == 0) {
            ready.push(op);
        }
    }
    size_t visited = 0;
    while (!ready.empty()) {
        auto* op = ready.front();
        ready.pop();
        ++visited;
        auto iter = adjacency.find(op);
        if (iter == adjacency.end()) {
            continue;
        }
        for (auto* consumer : iter->second) {
            if (--indegree[consumer] == 0) {
                ready.push(consumer);
            }
        }
    }
    return visited != indegree.size();
}

bool WouldCreateCycleAfterContraction(Function& function, const std::unordered_map<Operation*, Operation*>& mapping)
{
    auto mapOp = [&mapping](Operation* op) {
        auto iter = mapping.find(op);
        return iter == mapping.end() ? op : iter->second;
    };
    std::unordered_map<Operation*, std::unordered_set<Operation*>> adjacency;
    for (auto& op : function.Operations(false)) {
        adjacency.try_emplace(mapOp(&op));
        for (const auto& input : op.GetIOperands()) {
            for (auto* producer : input->GetProducers()) {
                if (producer->BelongTo() == &function && !producer->IsDeleted()) {
                    AddDependencyEdge(adjacency, mapOp(producer), mapOp(&op));
                }
            }
        }
    }
    for (const auto& [token, entry] : function.GetVarDependency().GetAllDependencies()) {
        (void)token;
        for (const auto& producerStmt : entry.producers) {
            for (const auto& consumerStmt : entry.consumers) {
                AddDependencyEdge(adjacency, mapOp(ToOperation(producerStmt)), mapOp(ToOperation(consumerStmt)));
            }
        }
    }
    return HasDependencyCycle(adjacency);
}

void CollectLinearTokenDependency(Function& function, const std::vector<Operation*>& chain,
                                  MergeViewAssembleUtils::TokenDependency& tokenDependency)
{
    std::unordered_set<Operation*> chainSet(chain.begin(), chain.end());
    for (const auto& snapshot : CollectTokenSnapshots(function, chainSet)) {
        AddUnique(tokenDependency.touchedTokens, snapshot.token);
        bool producerInChain = chainSet.count(snapshot.producer) != 0;
        bool consumerInChain = std::any_of(snapshot.consumers.begin(), snapshot.consumers.end(),
                                           [&chainSet](Operation* consumer) { return chainSet.count(consumer) != 0; });
        if (!producerInChain && consumerInChain) {
            AddUnique(tokenDependency.inputTokens, snapshot.token);
            continue;
        }
        if (!producerInChain) {
            continue;
        }
        std::vector<Operation*> externalConsumers;
        for (auto* consumer : snapshot.consumers) {
            if (chainSet.count(consumer) == 0) {
                AddUnique(externalConsumers, consumer);
            }
        }
        if (externalConsumers.empty()) {
            continue;
        }
        AddUnique(tokenDependency.resultTokens, snapshot.token);
        for (auto* consumer : externalConsumers) {
            AddUnique(tokenDependency.resultTokenConsumers, ToStmtPtr(*consumer));
        }
    }
}

void ClearLinearTokenDependency(Function& function, const std::vector<Operation*>& chain,
                                const MergeViewAssembleUtils::TokenDependency& tokenDependency)
{
    std::unordered_set<Operation*> chainSet(chain.begin(), chain.end());
    auto snapshots = CollectTokenSnapshots(function, chainSet);
    auto& dependency = function.GetVarDependency();
    for (const auto& snapshot : snapshots) {
        bool producerInChain = chainSet.count(snapshot.producer) != 0;
        bool isResultToken = std::find(tokenDependency.resultTokens.begin(), tokenDependency.resultTokens.end(),
                                       snapshot.token) != tokenDependency.resultTokens.end();
        for (auto* consumer : snapshot.consumers) {
            if (chainSet.count(consumer) != 0 || (producerInChain && !isResultToken)) {
                dependency.RemoveConsumer(snapshot.token, ToStmtPtr(*consumer));
                RemoveToken(*consumer, snapshot.token);
            }
        }
        if (producerInChain && snapshot.producer != nullptr) {
            dependency.RemoveProducer(snapshot.token, ToStmtPtr(*snapshot.producer));
            auto& producerTokens = snapshot.producer->result_token_;
            producerTokens.erase(std::remove(producerTokens.begin(), producerTokens.end(), snapshot.token),
                                 producerTokens.end());
        }
        if (producerInChain && !isResultToken) {
            dependency.RemoveVar(snapshot.token);
        } else if (dependency.GetProducers(snapshot.token).empty() && dependency.GetConsumers(snapshot.token).empty()) {
            dependency.RemoveVar(snapshot.token);
        }
    }
    for (auto* op : chain) {
        op->tokens_.clear();
        op->result_token_.clear();
    }
}

void ApplyLinearTokenDependency(Function& function, Operation& mergedOp,
                                const MergeViewAssembleUtils::TokenDependency& tokenDependency)
{
    for (const auto& token : tokenDependency.inputTokens) {
        AddTokenConsumer(function, token, mergedOp);
    }
    for (const auto& resultToken : tokenDependency.resultTokens) {
        AddUnique(mergedOp.result_token_, resultToken);
        function.GetVarDependency().AddProducer(resultToken, ToStmtPtr(mergedOp));
        for (const auto& consumerStmt : tokenDependency.resultTokenConsumers) {
            AddTokenConsumer(function, resultToken, *ToOperation(consumerStmt));
        }
    }
}

struct RmwModeAttrState {
    bool conflict = false;
    std::optional<AtomicRMWMode> mode;
};

struct AtomicSemanticAttrState {
    bool fromReduceAcc = false;
    bool fromExplicitRmw = false;
};

RmwModeAttrState MergeRmwModeAttr(const RmwModeAttrState& current, const RmwModeAttrState& next)
{
    if (current.conflict || next.conflict) {
        return {true, std::nullopt};
    }
    if (!current.mode.has_value()) {
        return next;
    }
    if (!next.mode.has_value() || current.mode == next.mode) {
        return current;
    }
    return {true, std::nullopt};
}

RmwModeAttrState GetRmwModeAttr(const Operation& op)
{
    RmwModeAttrState rmwModeAttr;
    if (op.HasAttr(RMW_MODE_ATTR_ADD)) {
        rmwModeAttr = MergeRmwModeAttr(rmwModeAttr, {false, AtomicRMWMode::ADD});
    }
    if (op.HasAttr(RMW_MODE_ATTR_MIN)) {
        rmwModeAttr = MergeRmwModeAttr(rmwModeAttr, {false, AtomicRMWMode::MIN});
    }
    if (op.HasAttr(RMW_MODE_ATTR_MAX)) {
        rmwModeAttr = MergeRmwModeAttr(rmwModeAttr, {false, AtomicRMWMode::MAX});
    }
    return rmwModeAttr;
}

RmwModeAttrState GetChainRmwModeAttr(const std::vector<Operation*>& chain)
{
    RmwModeAttrState chainRmwModeAttr;
    for (const auto* op : chain) {
        if (op == nullptr) {
            return {true, std::nullopt};
        }
        chainRmwModeAttr = MergeRmwModeAttr(chainRmwModeAttr, GetRmwModeAttr(*op));
        if (chainRmwModeAttr.conflict) {
            return chainRmwModeAttr;
        }
    }
    return chainRmwModeAttr;
}

bool IsRmwModeAttrCompatible(const std::vector<Operation*>& chain, const Operation& consumer)
{
    auto chainRmwModeAttr = GetChainRmwModeAttr(chain);
    auto consumerRmwModeAttr = GetRmwModeAttr(consumer);
    return !MergeRmwModeAttr(chainRmwModeAttr, consumerRmwModeAttr).conflict;
}

std::string GetRmwModeAttrKey(const RmwModeAttrState& rmwModeAttr)
{
    if (!rmwModeAttr.mode.has_value() || rmwModeAttr.conflict) {
        return "";
    }
    switch (*rmwModeAttr.mode) {
        case AtomicRMWMode::ADD:
            return RMW_MODE_ATTR_ADD;
        case AtomicRMWMode::MIN:
            return RMW_MODE_ATTR_MIN;
        case AtomicRMWMode::MAX:
            return RMW_MODE_ATTR_MAX;
        default:
            return "";
    }
}

bool IsViewLikeOpcode(Opcode opcode) { return opcode == Opcode::OP_VIEW || opcode == Opcode::OP_SLICE; }

bool IsAssembleLikeOpcode(Opcode opcode) { return opcode == Opcode::OP_ASSEMBLE || opcode == Opcode::OP_CONTRACT; }

bool ChainHasOpcode(const std::vector<Operation*>& chain, Opcode opcode)
{
    for (const auto* op : chain) {
        if (op != nullptr && op->GetOpcode() == opcode) {
            return true;
        }
    }
    return false;
}

bool CanMergeViewLikeChain(const std::vector<Operation*>& chain, Opcode nextOpcode)
{
    return !(ChainHasOpcode(chain, Opcode::OP_SLICE) && nextOpcode == Opcode::OP_SLICE);
}

bool CanMergeAssembleLikeChain(const std::vector<Operation*>& chain, Opcode nextOpcode)
{
    return !(ChainHasOpcode(chain, Opcode::OP_CONTRACT) && nextOpcode == Opcode::OP_CONTRACT);
}

Opcode GetMergedViewOpcode(const std::vector<Operation*>& chain)
{
    return ChainHasOpcode(chain, Opcode::OP_SLICE) ? Opcode::OP_SLICE : Opcode::OP_VIEW;
}

Opcode GetMergedAssembleOpcode(const std::vector<Operation*>& chain)
{
    return ChainHasOpcode(chain, Opcode::OP_CONTRACT) ? Opcode::OP_CONTRACT : Opcode::OP_ASSEMBLE;
}

bool WouldCreateProducerGroupCycle(Function& function, const MergeViewAssembleUtils::ProducerGroupFusion& fusion)
{
    if (fusion.downstream == nullptr) {
        return true;
    }
    std::unordered_set<Operation*> groupSet(fusion.producers.begin(), fusion.producers.end());
    std::unordered_map<Operation*, std::unordered_set<Operation*>> adjacency;
    for (auto& op : function.Operations(false)) {
        if (&op == fusion.downstream) {
            continue;
        }
        adjacency.try_emplace(&op);
        for (const auto& input : op.GetIOperands()) {
            for (auto* producer : input->GetProducers()) {
                if (producer == fusion.downstream) {
                    for (auto* groupProducer : fusion.producers) {
                        AddDependencyEdge(adjacency, groupProducer, &op);
                    }
                } else if (producer->BelongTo() == &function && !producer->IsDeleted()) {
                    AddDependencyEdge(adjacency, producer, &op);
                }
            }
        }
    }
    for (const auto& [token, entry] : function.GetVarDependency().GetAllDependencies()) {
        (void)token;
        for (const auto& producerStmt : entry.producers) {
            auto* producer = ToOperation(producerStmt);
            for (const auto& consumerStmt : entry.consumers) {
                auto* consumer = ToOperation(consumerStmt);
                if (producer == fusion.downstream) {
                    if (groupSet.count(consumer) != 0) {
                        return true;
                    }
                    for (auto* groupProducer : fusion.producers) {
                        AddDependencyEdge(adjacency, groupProducer, consumer);
                    }
                } else if (consumer == fusion.downstream) {
                    if (groupSet.count(producer) == 0) {
                        for (auto* groupProducer : fusion.producers) {
                            AddDependencyEdge(adjacency, producer, groupProducer);
                        }
                    }
                } else {
                    AddDependencyEdge(adjacency, producer, consumer);
                }
            }
        }
    }
    return HasDependencyCycle(adjacency);
}

void ResetTokenDependency(Function& function, const ir::VarPtr& token, Operation* producer,
                          const std::vector<Operation*>& consumers)
{
    auto& dependency = function.GetVarDependency();
    dependency.RemoveVar(token);
    if (producer != nullptr) {
        AddUnique(producer->result_token_, token);
        dependency.AddProducer(token, ToStmtPtr(*producer));
    }
    for (auto* consumer : consumers) {
        if (consumer != producer) {
            AddTokenConsumer(function, token, *consumer);
        }
    }
}

ir::VarPtr EnsureResultToken(Function& function, Operation& producer)
{
    if (producer.result_token_.empty()) {
        producer.result_token_.push_back(IRBuilder().CreateTokenVar(producer.GetSpan()));
    }
    auto token = producer.result_token_.front();
    function.GetVarDependency().AddProducer(token, ToStmtPtr(producer));
    return token;
}

bool HasDataPath(Operation* producer, Operation* consumer)
{
    if (producer == nullptr || consumer == nullptr) {
        return false;
    }
    std::vector<Operation*> pending{consumer};
    std::unordered_set<Operation*> visited;
    while (!pending.empty()) {
        auto* current = pending.back();
        pending.pop_back();
        if (!visited.insert(current).second) {
            continue;
        }
        for (auto* predecessor : current->ProducerOps()) {
            if (predecessor == producer) {
                return true;
            }
            if (predecessor != nullptr && predecessor->BelongTo() == consumer->BelongTo() &&
                !predecessor->IsDeleted()) {
                pending.emplace_back(predecessor);
            }
        }
    }
    return false;
}

bool AddDataCoveredTokenConsumers(const MergeViewAssembleUtils::ProducerGroupFusion& fusion,
                                  const std::vector<Operation*>& replacements, Operation* tokenProducer,
                                  std::vector<Operation*>& tokenConsumers)
{
    if (tokenProducer == nullptr || fusion.producers.size() != replacements.size()) {
        return false;
    }
    bool covered = false;
    for (size_t index = 0; index < fusion.producers.size(); ++index) {
        auto* producer = fusion.producers[index];
        if (producer == nullptr) {
            continue;
        }
        if (HasDataPath(tokenProducer, producer)) {
            AddUnique(tokenConsumers, replacements[index]);
            covered = true;
        }
    }
    return covered;
}

void RewriteProducerGroupTokens(Function& function, const MergeViewAssembleUtils::ProducerGroupFusion& fusion,
                                const std::vector<Operation*>& replacements)
{
    std::unordered_set<Operation*> affected(fusion.producers.begin(), fusion.producers.end());
    affected.insert(fusion.downstream);
    auto snapshots = CollectTokenSnapshots(function, affected);
    std::unordered_map<Operation*, Operation*> replacementMap;
    for (size_t index = 0; index < fusion.producers.size(); ++index) {
        replacementMap.emplace(fusion.producers[index], replacements[index]);
    }

    auto& dependency = function.GetVarDependency();
    for (const auto& snapshot : snapshots) {
        for (auto* consumer : snapshot.consumers) {
            dependency.RemoveConsumer(snapshot.token, ToStmtPtr(*consumer));
            RemoveToken(*consumer, snapshot.token);
        }
        if (snapshot.producer != nullptr) {
            dependency.RemoveProducer(snapshot.token, ToStmtPtr(*snapshot.producer));
            auto& producerTokens = snapshot.producer->result_token_;
            producerTokens.erase(std::remove(producerTokens.begin(), producerTokens.end(), snapshot.token),
                                 producerTokens.end());
        }
        dependency.RemoveVar(snapshot.token);
    }
    for (auto* op : affected) {
        op->tokens_.clear();
        op->result_token_.clear();
    }

    std::vector<const TokenSnapshot*> downstreamResultTokens;
    for (const auto& snapshot : snapshots) {
        if (snapshot.producer == fusion.downstream) {
            downstreamResultTokens.emplace_back(&snapshot);
            continue;
        }
        Operation* newProducer = snapshot.producer;
        auto producerIter = replacementMap.find(snapshot.producer);
        if (producerIter != replacementMap.end()) {
            newProducer = producerIter->second;
        }
        std::vector<Operation*> newConsumers;
        for (auto* consumer : snapshot.consumers) {
            if (consumer == fusion.downstream) {
                if (replacementMap.count(snapshot.producer) == 0 &&
                    !AddDataCoveredTokenConsumers(fusion, replacements, snapshot.producer, newConsumers)) {
                    for (auto* replacement : replacements) {
                        AddUnique(newConsumers, replacement);
                    }
                }
                continue;
            }
            auto consumerIter = replacementMap.find(consumer);
            AddUnique(newConsumers, consumerIter == replacementMap.end() ? consumer : consumerIter->second);
        }
        ResetTokenDependency(function, snapshot.token, newProducer, newConsumers);
    }

    for (const auto* snapshot : downstreamResultTokens) {
        for (auto* replacement : replacements) {
            auto resultToken = EnsureResultToken(function, *replacement);
            for (auto* consumer : snapshot->consumers) {
                AddTokenConsumer(function, resultToken, *consumer);
            }
        }
    }

    for (auto* replacement : replacements) {
        auto resultTokens = replacement->result_token_;
        for (const auto& token : resultTokens) {
            if (!function.GetVarDependency().GetConsumers(token).empty()) {
                continue;
            }
            dependency.RemoveProducer(token, ToStmtPtr(*replacement));
            dependency.RemoveVar(token);
            auto& replacementTokens = replacement->result_token_;
            replacementTokens.erase(std::remove(replacementTokens.begin(), replacementTokens.end(), token),
                                    replacementTokens.end());
        }
    }
}

AtomicSemanticAttrState GetChainAtomicSemanticAttr(const std::vector<Operation*>& chain)
{
    AtomicSemanticAttrState attr;
    for (const auto* op : chain) {
        if (op == nullptr) {
            continue;
        }
        attr.fromReduceAcc = attr.fromReduceAcc || op->HasAttr(ATOMIC_FROM_REDUCE_ACC_ATTR);
        attr.fromExplicitRmw = attr.fromExplicitRmw || op->HasAttr(ATOMIC_FROM_EXPLICIT_RMW_ATTR);
    }
    return attr;
}
} // namespace

Status MergeViewAssembleUtils::MergeViewAssemble(Function& function)
{
    MergeViewAssembleUtils MergeViewAssembleUtils;
    Status status = MergeViewAssembleUtils.Process(function);
    return status;
}

Status MergeViewAssembleUtils::Process(Function& function)
{
    Status status = Initialize();
    if (status != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Function, "MergeViewAssembleUtils initialization failed.");
        return status;
    }
    DeadOperationEliminator eliminator;
    eliminator.EliminateOperation(function, false, false);
    status = ProcessOperations(function);
    if (status != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Function, "Processing operations failed.");
        return status;
    }
    status = CleanUp(function);
    if (status != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Function, "Cleanup phase failed.");
        return status;
    }
    return SUCCESS;
}

Status MergeViewAssembleUtils::Initialize()
{
    visitedOp_.clear();
    viewOpToAppend_.clear();
    assembleOpToAppend_.clear();
    consumerCache_.clear();
    tensorConsumerCache_.clear();
    rawTensorVersions_.clear();
    processedGroupTensor_.clear();
    candidateOps_.clear();
    producerGroupFusions_.clear();
    return SUCCESS;
}

const MergeViewAssembleUtils::ConsumerCacheEntry& MergeViewAssembleUtils::BuildTensorConsumerCache(
    Function& function, const LogicalTensorPtr& tensor)
{
    static const ConsumerCacheEntry emptyEntry;
    if (tensor == nullptr) {
        return emptyEntry;
    }
    auto tensorMagic = tensor->GetMagic();
    auto cached = tensorConsumerCache_.find(tensorMagic);
    if (cached != tensorConsumerCache_.end()) {
        return cached->second;
    }

    auto iter = tensorConsumerCache_.emplace(tensorMagic, ConsumerCacheEntry{}).first;
    auto& cacheEntry = iter->second;
    cacheEntry.producerCount = tensor->GetProducers().size();
    cacheEntry.allProducersAreAssemble = cacheEntry.producerCount != 0;
    for (auto* producer : tensor->GetProducers()) {
        if (producer == nullptr || producer->BelongTo() != &function || producer->IsDeleted() ||
            producer->GetOpcode() != Opcode::OP_ASSEMBLE) {
            cacheEntry.allProducersAreAssemble = false;
            break;
        }
    }
    for (auto* consumer : tensor->GetConsumers()) {
        if (consumer == nullptr || consumer->BelongTo() != &function || consumer->IsDeleted()) {
            continue;
        }
        if (IsViewLikeOpcode(consumer->GetOpcode())) {
            cacheEntry.viewConsumers.emplace_back(consumer);
            cacheEntry.hasAssembleChainStopper = true;
        } else if (IsAssembleLikeOpcode(consumer->GetOpcode())) {
            cacheEntry.assembleConsumers.emplace_back(consumer);
            cacheEntry.hasViewChainStopper |= !consumer->result_token_.empty() || !consumer->tokens_.empty() ||
                                              IsFullTensorAssemble(*consumer);
        } else {
            cacheEntry.hasViewChainStopper = true;
            cacheEntry.hasAssembleChainStopper = true;
        }
    }
    return cacheEntry;
}

Status MergeViewAssembleUtils::BuildConsumerCache(Function& function)
{
    auto operations = function.Operations(false);
    consumerCache_.reserve(operations.size());
    tensorConsumerCache_.reserve(operations.size());
    candidateOps_.reserve(operations.size());
    auto recordTensor = [this](const LogicalTensorPtr& tensor) {
        if (tensor == nullptr) {
            return;
        }
        auto& versions = rawTensorVersions_[tensor->GetRawMagic()];
        if (std::none_of(versions.begin(), versions.end(), [&tensor](const LogicalTensorPtr& existing) {
                return existing->GetMagic() == tensor->GetMagic();
            })) {
            versions.emplace_back(tensor);
        }
    };
    for (const auto& incast : function.GetIncast()) {
        recordTensor(incast);
    }
    for (const auto& outcast : function.GetOutcast()) {
        recordTensor(outcast);
    }
    for (auto& operation : operations) {
        for (const auto& input : operation.GetIOperands()) {
            recordTensor(input);
        }
        for (const auto& output : operation.GetOOperands()) {
            recordTensor(output);
        }
        if (!IsViewLikeOpcode(operation.GetOpcode()) && !IsAssembleLikeOpcode(operation.GetOpcode())) {
            continue;
        }
        candidateOps_.emplace_back(&operation);
        if (operation.oOperand.empty()) {
            continue;
        }
        consumerCache_[operation.GetOpMagic()] = &BuildTensorConsumerCache(function, operation.oOperand.front());
    }
    return SUCCESS;
}

Status MergeViewAssembleUtils::DiscoverProducerGroupFusions(Function& function)
{
    for (auto* op : candidateOps_) {
        if (op == nullptr || op->GetOpcode() != Opcode::OP_ASSEMBLE || op->oOperand.empty()) {
            continue;
        }
        const auto& middle = op->oOperand.front();
        if (middle == nullptr || middle->GetProducers().size() <= 1 ||
            processedGroupTensor_.count(middle->GetMagic()) != 0) {
            continue;
        }
        processedGroupTensor_.insert(middle->GetMagic());
        const auto& consumers = BuildTensorConsumerCache(function, middle);
        BuildProducerGroupFusion(function, middle, consumers);
    }
    return SUCCESS;
}

const MergeViewAssembleUtils::ConsumerCacheEntry& MergeViewAssembleUtils::GetConsumers(const Operation& operation) const
{
    static const ConsumerCacheEntry emptyEntry;
    auto iter = consumerCache_.find(operation.GetOpMagic());
    if (iter == consumerCache_.end() || iter->second == nullptr) {
        return emptyEntry;
    }
    return *iter->second;
}

bool MergeViewAssembleUtils::IsFunctionBoundaryTensor(const Function& function, const LogicalTensorPtr& tensor)
{
    auto isSameTensor = [&tensor](const LogicalTensorPtr& boundary) {
        return tensor != nullptr && boundary != nullptr && tensor->GetMagic() == boundary->GetMagic();
    };
    return std::any_of(function.GetIncast().begin(), function.GetIncast().end(), isSameTensor) ||
           std::any_of(function.GetOutcast().begin(), function.GetOutcast().end(), isSameTensor);
}

bool MergeViewAssembleUtils::HasCompleteStaticCoverage(const LogicalTensorPtr& middle,
                                                       const std::vector<Operation*>& producers)
{
    if (middle == nullptr || middle->GetShape().empty() || producers.empty()) {
        return false;
    }
    const auto& targetShape = middle->GetShape();
    struct Region {
        std::vector<int64_t> begin;
        std::vector<int64_t> end;
    };
    std::vector<Region> regions;
    std::vector<std::vector<int64_t>> boundaries(targetShape.size());
    for (size_t dim = 0; dim < targetShape.size(); ++dim) {
        if (targetShape[dim] <= 0) {
            return false;
        }
        boundaries[dim] = {0, targetShape[dim]};
    }
    for (auto* producer : producers) {
        if (producer == nullptr || producer->GetOpcode() != Opcode::OP_ASSEMBLE || producer->iOperand.size() != 1 ||
            producer->oOperand.size() != 1) {
            return false;
        }
        auto attr = std::dynamic_pointer_cast<AssembleOpAttribute>(producer->GetOpAttribute());
        if (attr == nullptr) {
            return false;
        }
        auto offset = attr->GetToOffset();
        const auto& dynOffset = attr->GetToDynOffset();
        if (!dynOffset.empty()) {
            if (dynOffset.size() != targetShape.size() ||
                std::any_of(dynOffset.begin(), dynOffset.end(),
                            [](const SymbolicScalar& value) { return !value.ConcreteValid(); })) {
                return false;
            }
            offset.clear();
            std::transform(dynOffset.begin(), dynOffset.end(), std::back_inserter(offset),
                           [](const SymbolicScalar& value) { return value.Concrete(); });
        }
        const auto& shape = producer->iOperand.front()->GetShape();
        if (offset.size() != targetShape.size() || shape.size() != targetShape.size()) {
            return false;
        }
        Region region{offset, offset};
        for (size_t dim = 0; dim < targetShape.size(); ++dim) {
            if (offset[dim] < 0 || shape[dim] <= 0 || offset[dim] > targetShape[dim] - shape[dim]) {
                return false;
            }
            region.end[dim] += shape[dim];
            boundaries[dim].push_back(region.begin[dim]);
            boundaries[dim].push_back(region.end[dim]);
        }
        regions.emplace_back(std::move(region));
    }
    size_t cellCount = 1;
    for (auto& dimensionBoundaries : boundaries) {
        std::sort(dimensionBoundaries.begin(), dimensionBoundaries.end());
        dimensionBoundaries.erase(std::unique(dimensionBoundaries.begin(), dimensionBoundaries.end()),
                                  dimensionBoundaries.end());
        if (dimensionBoundaries.size() < 2 || cellCount > 100000 / (dimensionBoundaries.size() - 1)) {
            return false;
        }
        cellCount *= dimensionBoundaries.size() - 1;
    }
    std::vector<int64_t> point(targetShape.size(), 0);
    std::function<bool(size_t)> checkCells = [&](size_t dim) {
        if (dim == boundaries.size()) {
            return std::any_of(regions.begin(), regions.end(), [&point](const Region& region) {
                for (size_t index = 0; index < point.size(); ++index) {
                    if (point[index] < region.begin[index] || point[index] >= region.end[index]) {
                        return false;
                    }
                }
                return true;
            });
        }
        for (size_t index = 0; index + 1 < boundaries[dim].size(); ++index) {
            point[dim] = boundaries[dim][index];
            if (!checkCells(dim + 1)) {
                return false;
            }
        }
        return true;
    };
    return checkCells(0);
}

bool MergeViewAssembleUtils::HasSplitVersionContribution(const LogicalTensorPtr& middle,
                                                         const std::vector<Operation*>& currentProducers) const
{
    if (middle == nullptr) {
        return false;
    }
    auto versionsIter = rawTensorVersions_.find(middle->GetRawMagic());
    if (versionsIter == rawTensorVersions_.end() || versionsIter->second.size() <= 1) {
        return false;
    }
    if (HasCompleteStaticCoverage(middle, currentProducers)) {
        return false;
    }
    for (const auto& version : versionsIter->second) {
        if (version == nullptr || version->GetMagic() == middle->GetMagic()) {
            continue;
        }
        for (auto* producer : version->GetProducers()) {
            if (producer != nullptr && !producer->IsDeleted() && producer->GetOpcode() == Opcode::OP_ASSEMBLE) {
                return true;
            }
        }
    }
    return false;
}

bool MergeViewAssembleUtils::BuildProducerGroupFusion(Function& function, const LogicalTensorPtr& middle,
                                                      const ConsumerCacheEntry& consumers)
{
    if (middle == nullptr || consumers.hasAssembleChainStopper || consumers.assembleConsumers.size() != 1 ||
        !consumers.allProducersAreAssemble || IsFunctionBoundaryTensor(function, middle)) {
        return false;
    }
    auto* downstream = consumers.assembleConsumers.front();
    if (downstream == nullptr || visitedOp_.count(downstream->GetOpMagic()) != 0 || downstream->iOperand.size() != 1 ||
        downstream->oOperand.size() != 1 || downstream->iOperand.front()->GetMagic() != middle->GetMagic()) {
        return false;
    }
    std::vector<Operation*> producers;
    for (auto* producer : middle->GetProducers()) {
        if (producer == nullptr || producer->BelongTo() != &function || producer->IsDeleted() ||
            visitedOp_.count(producer->GetOpMagic()) != 0 || producer->GetOpcode() != Opcode::OP_ASSEMBLE ||
            producer->iOperand.size() != 1 || producer->oOperand.size() != 1 ||
            producer->oOperand.front()->GetMagic() != middle->GetMagic()) {
            return false;
        }
        producers.emplace_back(producer);
    }
    if (producers.size() <= 1 || HasSplitVersionContribution(middle, producers) ||
        !HasCompleteStaticCoverage(middle, producers)) {
        return false;
    }

    int effectiveScopeId = downstream->GetScopeId();
    for (auto* producer : producers) {
        int scopeId = producer->GetScopeId();
        if (effectiveScopeId == -1) {
            effectiveScopeId = scopeId;
        } else if (scopeId != -1 && scopeId != effectiveScopeId) {
            return false;
        }
        if (!IsRmwModeAttrCompatible({producer}, *downstream)) {
            return false;
        }
    }

    ProducerGroupFusion fusion;
    fusion.middle = middle;
    fusion.downstream = downstream;
    fusion.producers = producers;
    for (auto* producer : producers) {
        std::vector<Operation*> pair{producer, downstream};
        auto [offset, dynOffset] = CalculateAssembleOffsets(pair, producer->iOperand.front()->offset.size());
        if (offset.empty() && !producer->iOperand.front()->offset.empty()) {
            return false;
        }
        auto rmwModeAttr = GetChainRmwModeAttr(pair);
        if (rmwModeAttr.conflict) {
            return false;
        }
        auto atomicSemanticAttr = GetChainAtomicSemanticAttr(pair);
        fusion.replacements.emplace_back(AssembleOp{producer->iOperand.front(),
                                                    downstream->oOperand.front(),
                                                    offset,
                                                    dynOffset,
                                                    GetFirstSpan(pair),
                                                    GetChainScopeInfo(pair),
                                                    GetRmwModeAttrKey(rmwModeAttr),
                                                    {},
                                                    {},
                                                    atomicSemanticAttr.fromReduceAcc,
                                                    atomicSemanticAttr.fromExplicitRmw});
    }
    std::vector<Operation*> tokenOps = producers;
    tokenOps.emplace_back(downstream);
    if (HasTokenDependency(tokenOps) && WouldCreateProducerGroupCycle(function, fusion)) {
        return false;
    }
    producerGroupFusions_.emplace_back(std::move(fusion));
    for (auto* producer : producers) {
        visitedOp_.insert(producer->GetOpMagic());
    }
    visitedOp_.insert(downstream->GetOpMagic());
    return true;
}

Status MergeViewAssembleUtils::ProcessOperations(Function& function)
{
    function.SortOperations(SortOperationsMode::LIGHTWEIGHT);
    Status status = BuildConsumerCache(function);
    if (status != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Function, "BuildConsumerCache failed.");
        return status;
    }
    status = DiscoverProducerGroupFusions(function);
    if (status != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Function, "DiscoverProducerGroupFusions failed.");
        return status;
    }
    for (auto* op : candidateOps_) {
        if (op == nullptr || op->IsDeleted()) {
            continue;
        }
        if (visitedOp_.count(op->GetOpMagic()) != 0) {
            continue;
        }
        Status processStatus = SUCCESS;
        std::vector<Operation*> chain;
        if (IsViewLikeOpcode(op->GetOpcode())) {
            processStatus = MergeViewChain(function, *op, chain);
        } else if (IsAssembleLikeOpcode(op->GetOpcode())) {
            processStatus = MergeAssembleChain(function, *op, chain);
        }
        if (processStatus != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Function, "ProcessOperations failed.");
            return processStatus;
        }
    }
    status = AppendMergedViewOperations(function);
    if (status != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Function, "AppendMergedViewOperations phase failed.");
        return status;
    }
    status = AppendMergedAssembleOperations(function);
    if (status != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Function, "AppendMergedAssembleOperations phase failed.");
        return FAILED;
    }
    status = AppendProducerGroupFusions(function);
    if (status != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Function, "AppendProducerGroupFusions phase failed.");
        return status;
    }
    return status;
}

Status MergeViewAssembleUtils::AppendMergedViewOperations(Function& function)
{
    /* Process View ops first to avoid View output being cleared in View-Assemble scenarios */
    for (auto& viewOp : viewOpToAppend_) {
        auto attr = std::make_shared<ViewOpAttribute>(viewOp.offset, viewOp.toType, viewOp.dynOffset,
                                                      viewOp.dynValidShape);
        if (!attr) {
            APASS_LOG_ERROR_F(Elements::Function, "Failed to create ViewOpAttribute.");
            return FAILED;
        }
        auto& mergedViewOp = irBuilder_.CreateTensorOpStmt(function, viewOp.opcode, {viewOp.input}, {viewOp.output},
                                                           viewOp.span);
        mergedViewOp.SetScopeInfo(viewOp.scopeInfo);
        mergedViewOp.SetOpAttribute(attr);
        // 继承op_attr_copy_in_mode属性
        if (viewOp.hasCopyInMode) {
            mergedViewOp.SetAttr("op_attr_copy_in_mode", viewOp.copyInModeValue);
        }
        // 继承op_attr_copy_in_l1_padding_mode属性
        if (viewOp.hasL1PaddingMode) {
            mergedViewOp.SetAttr("op_attr_copy_in_l1_padding_mode", viewOp.l1PaddingMode);
        }
        // 继承op_attr_copy_in_l1_k_index属性
        if (viewOp.hasKIndex) {
            mergedViewOp.SetAttr("op_attr_copy_in_l1_k_index", viewOp.kIndex);
        }
        // 继承op_attr_is_gemv属性
        if (viewOp.hasIsGemv) {
            mergedViewOp.SetAttr(OpAttributeKey::isGemv, viewOp.isGemvValue);
        }
        ApplyLinearTokenDependency(function, mergedViewOp, viewOp.tokenDependency);
        viewOp.output->UpdateDynValidShape(viewOp.dynValidShape);
    }
    return SUCCESS;
}

Status MergeViewAssembleUtils::AppendMergedAssembleOperations(Function& function)
{
    for (const auto& assembleOp : assembleOpToAppend_) {
        auto attr = std::make_shared<AssembleOpAttribute>(assembleOp.offset, assembleOp.dynOffset);
        if (!attr) {
            return FAILED;
        }
        auto& mergedAssembleOp = irBuilder_.CreateTensorOpStmt(function, assembleOp.opcode, {assembleOp.input},
                                                               {assembleOp.output}, assembleOp.span);
        mergedAssembleOp.SetScopeInfo(assembleOp.scopeInfo);
        mergedAssembleOp.SetOpAttribute(attr);
        if (!assembleOp.rmwModeAttr.empty()) {
            mergedAssembleOp.SetAttribute(assembleOp.rmwModeAttr, 1L);
        }
        ApplyLinearTokenDependency(function, mergedAssembleOp, assembleOp.tokenDependency);
        if (assembleOp.atomicFromReduceAcc) {
            mergedAssembleOp.SetAttribute(ATOMIC_FROM_REDUCE_ACC_ATTR, true);
        }
        if (assembleOp.atomicFromExplicitRmw) {
            mergedAssembleOp.SetAttribute(ATOMIC_FROM_EXPLICIT_RMW_ATTR, true);
        }
    }
    return SUCCESS;
}

Status MergeViewAssembleUtils::AppendProducerGroupFusions(Function& function)
{
    for (const auto& fusion : producerGroupFusions_) {
        std::vector<Operation*> replacements;
        replacements.reserve(fusion.replacements.size());
        for (const auto& replacement : fusion.replacements) {
            auto attr = std::make_shared<AssembleOpAttribute>(replacement.offset, replacement.dynOffset);
            auto& mergedOp = irBuilder_.CreateTensorOpStmt(function, Opcode::OP_ASSEMBLE, {replacement.input},
                                                           {replacement.output}, replacement.span);
            mergedOp.SetScopeInfo(replacement.scopeInfo);
            mergedOp.SetOpAttribute(attr);
            if (!replacement.rmwModeAttr.empty()) {
                mergedOp.SetAttribute(replacement.rmwModeAttr, 1L);
            }
            if (replacement.atomicFromReduceAcc) {
                mergedOp.SetAttribute(ATOMIC_FROM_REDUCE_ACC_ATTR, true);
            }
            if (replacement.atomicFromExplicitRmw) {
                mergedOp.SetAttribute(ATOMIC_FROM_EXPLICIT_RMW_ATTR, true);
            }
            replacements.emplace_back(&mergedOp);
        }
        RewriteProducerGroupTokens(function, fusion, replacements);
        for (auto* producer : fusion.producers) {
            producer->SetAsDeleted();
        }
        fusion.downstream->SetAsDeleted();
    }
    return SUCCESS;
}

Status MergeViewAssembleUtils::CleanUp(Function& function)
{
    function.EraseOperations(true, false);
    DeadOperationEliminator eliminator;
    eliminator.EliminateOperation(function, false, false);
    function.SortOperations(SortOperationsMode::LIGHTWEIGHT);
    return SUCCESS;
}

ir::Span MergeViewAssembleUtils::GetFirstSpan(const std::vector<Operation*>& chain)
{
    ir::Span firstSpan;
    for (auto* op : chain) {
        auto loc = op->GetSpan();
        if (!loc.IsUnknown()) {
            firstSpan = loc;
            break;
        }
    }
    return firstSpan;
}

Operation::ScopeInfo MergeViewAssembleUtils::GetChainScopeInfo(const std::vector<Operation*>& chain)
{
    for (auto* op : chain) {
        if (op->GetScopeId() != -1) {
            return op->GetScopeInfo();
        }
    }
    return Operation::ScopeInfo();
}

Status MergeViewAssembleUtils::MergeViewChain(Function& function, Operation& operation, std::vector<Operation*>& chain,
                                              int effectiveScopeId)
{
    auto viewOpAttribute = std::dynamic_pointer_cast<ViewOpAttribute>(operation.GetOpAttribute());
    // 1. 初始化操作链
    InitOperationChain(operation, chain);

    int newScopeId = operation.GetScopeId();
    if (effectiveScopeId == -1 && newScopeId != -1) {
        effectiveScopeId = newScopeId;
    }

    // 2. 处理消费者链
    const auto& consumers = GetConsumers(operation);
    bool chainEnd = true;
    Status status = ProcessConsumerChain(function, consumers, chain, chainEnd, effectiveScopeId);
    if (status != SUCCESS) {
        return status;
    }

    // 3. 处理链尾情况
    if (chainEnd && chain.size() > 1) {
        return ProcessChainEnd(function, chain);
    }

    return SUCCESS;
}

void MergeViewAssembleUtils::InitOperationChain(Operation& operation, std::vector<Operation*>& chain)
{
    visitedOp_.insert(operation.opmagic);
    chain.emplace_back(&operation);
}

Status MergeViewAssembleUtils::ProcessConsumerChain(Function& function, const ConsumerCacheEntry& consumers,
                                                    std::vector<Operation*>& chain, bool& chainEnd,
                                                    int effectiveScopeId)
{
    bool hasActiveAssembleConsumer = std::any_of(
        consumers.assembleConsumers.begin(), consumers.assembleConsumers.end(),
        [this](Operation* op) { return op != nullptr && visitedOp_.count(op->GetOpMagic()) == 0; });
    if (consumers.hasViewChainStopper || hasActiveAssembleConsumer || consumers.viewConsumers.empty()) {
        return SUCCESS;
    }
    Operation* currentOp = chain.back();
    auto currentViewAttr = std::dynamic_pointer_cast<ViewOpAttribute>(currentOp->GetOpAttribute());
    if (!currentViewAttr) {
        APASS_LOG_ERROR_F(Elements::Function, "Failed to get current view attribute.");
        return FAILED;
    }
    MemoryType currentMemType = currentViewAttr->GetTo();
    for (auto& op : consumers.viewConsumers) {
        if (!op) {
            return FAILED;
        }
        if (IsViewLikeOpcode(op->GetOpcode())) {
            auto viewOpAttribute = std::dynamic_pointer_cast<ViewOpAttribute>(op->GetOpAttribute());
            if (viewOpAttribute == nullptr) {
                APASS_LOG_ERROR_F(Elements::Function, "View operation has null viewOpAttribute.");
                return FAILED;
            }
            auto memoryTo = viewOpAttribute->GetTo();
            // 根据新的合并原则判断是否可以合并
            bool canMerge = false;
            if (currentMemType == MemoryType::MEM_UNKNOWN || currentMemType == memoryTo) {
                // 1.unknown memType 可以向它之后的view合并 2.相同memType的view可以合并
                canMerge = true;
            }
            canMerge = canMerge && CanMergeViewLikeChain(chain, op->GetOpcode());
            if (canMerge) {
                int consumerScopeId = op->GetScopeId();
                if (effectiveScopeId != -1 && consumerScopeId != -1 && effectiveScopeId != consumerScopeId) {
                    chainEnd = true;
                    continue;
                }
                chainEnd = false;
                Status status = MergeViewChain(function, *op, chain, effectiveScopeId);
                if (status != SUCCESS) {
                    return status;
                }
                chain.pop_back();
            } else {
                chainEnd = true;
            }
        }
    }
    return SUCCESS;
}

Status MergeViewAssembleUtils::ProcessChainEnd(Function& function, std::vector<Operation*>& chain)
{
    // 1. 验证链的有效性
    Operation* startOp = chain.front();
    Operation* endOp = chain.back();
    if (startOp->iOperand.empty()) {
        APASS_LOG_ERROR_F(Elements::Function, "First operation in chain has no input operands.");
        return FAILED;
    }
    if (endOp->oOperand.empty()) {
        APASS_LOG_ERROR_F(Elements::Function, "Last operation in chain has no output operands.");
        return FAILED;
    }
    auto& startTensor = startOp->iOperand.front();
    auto& endTensor = endOp->oOperand.front();
    if (!startTensor) {
        APASS_LOG_ERROR_F(Elements::Function, "Null input tensor found for first operation in chain.");
        return FAILED;
    }
    if (!endTensor) {
        APASS_LOG_ERROR_F(Elements::Function, "Null output tensor found for last operation in chain.");
        return FAILED;
    }
    if (HasTokenDependency(chain)) {
        std::unordered_map<Operation*, Operation*> contraction;
        for (auto* op : chain) {
            contraction.emplace(op, chain.front());
        }
        if (WouldCreateCycleAfterContraction(function, contraction)) {
            return SUCCESS;
        }
    }
    TokenDependency tokenDependency;
    CollectLinearTokenDependency(function, chain, tokenDependency);
    std::vector<int64_t> newOffset;
    std::vector<SymbolicScalar> newDynOffset;
    std::vector<SymbolicScalar> newDynValidShape;
    Status status = CalculateMergedOffsets(chain, newOffset, newDynOffset, newDynValidShape);
    if (status != SUCCESS) {
        return status;
    }
    // 获取链路上第一个非空的span
    ir::Span firstSpan = GetFirstSpan(chain);
    Operation::ScopeInfo chainScopeInfo = GetChainScopeInfo(chain);
    // 记录合并操作
    RecordMergedViewOperation(endOp, startTensor, endTensor, newOffset, newDynOffset, newDynValidShape, firstSpan,
                              chainScopeInfo, GetMergedViewOpcode(chain), tokenDependency);
    ClearLinearTokenDependency(function, chain, tokenDependency);
    // 清理链尾
    endOp->oOperand.clear();
    function.GetTensorMap().Erase(endTensor);
    return SUCCESS;
}

Status MergeViewAssembleUtils::CalculateMergedOffsets(const std::vector<Operation*>& chain,
                                                      std::vector<int64_t>& newOffset,
                                                      std::vector<SymbolicScalar>& newDynOffset,
                                                      std::vector<SymbolicScalar>& newDynValidShape)
{
    for (size_t i = 0; i < chain.size(); ++i) {
        const auto& view = chain[i];
        if (!view) {
            APASS_LOG_ERROR_F(Elements::Function, "Null view operation in chain.");
            return FAILED;
        }
        auto viewOpAttribute = std::dynamic_pointer_cast<ViewOpAttribute>(view->GetOpAttribute());
        if (!viewOpAttribute) {
            APASS_LOG_ERROR_F(Elements::Function, "Failed to get ViewOpAttribute.");
            return FAILED;
        }
        if (i == 0) {
            newOffset = viewOpAttribute->GetFromOffset();
            newDynOffset = viewOpAttribute->GetFromDynOffset();
            if (!viewOpAttribute->GetToDynValidShape().empty()) {
                newDynValidShape = viewOpAttribute->GetToDynValidShape();
            }
            continue;
        }
        auto ret = TensorOffset::Add(newOffset, newDynOffset, viewOpAttribute->GetFromOffset(),
                                     viewOpAttribute->GetFromDynOffset());
        if (!ret.first.empty()) {
            newOffset = ret.first;
            newDynOffset = ret.second;
        }
        if (!viewOpAttribute->GetToDynValidShape().empty()) {
            newDynValidShape = viewOpAttribute->GetToDynValidShape();
            continue;
        }
        newDynValidShape = GetViewValidShape(newDynValidShape, viewOpAttribute->GetFromOffset(),
                                             viewOpAttribute->GetFromDynOffset(), view->GetOOperands()[0]->GetShape());
    }
    return SUCCESS;
}

void MergeViewAssembleUtils::RecordMergedViewOperation(
    Operation* lastViewOp, const std::shared_ptr<LogicalTensor>& startTensor,
    const std::shared_ptr<LogicalTensor>& endTensor, const std::vector<int64_t>& newOffset,
    const std::vector<SymbolicScalar>& newDynOffset, const std::vector<SymbolicScalar>& newDynValidShape,
    const ir::Span& span, const Operation::ScopeInfo& scopeInfo, Opcode opcode, const TokenDependency& tokenDependency)
{
    // 获取最后一个VIEW的属性
    auto lastViewAttr = std::dynamic_pointer_cast<ViewOpAttribute>(lastViewOp->GetOpAttribute());
    if (!lastViewAttr) {
        return;
    }
    // 获取特定的 op_attr_copy_in_mode 属性
    int64_t copyInModeValue = 0;
    bool hasCopyInMode = lastViewOp->GetAttr<int64_t>("op_attr_copy_in_mode", copyInModeValue);
    // 获取特定的 op_attr_copy_in_l1_padding_mode 属性
    int64_t l1PaddingMode = 0;
    bool hasL1PaddingMode = lastViewOp->GetAttr<int64_t>("op_attr_copy_in_l1_padding_mode", l1PaddingMode);
    // 获取特定的 op_attr_copy_in_l1_k_index 属性
    int64_t kIndex = 0;
    bool hasKIndex = lastViewOp->GetAttr<int64_t>("op_attr_copy_in_l1_k_index", kIndex);
    // 获取特定的 op_attr_is_gemv 属性
    int64_t isGemv = 0;
    bool hasIsGemv = lastViewOp->GetAttr<int64_t>(OpAttributeKey::isGemv, isGemv);
    // 清理消费者关系
    endTensor->GetProducers().clear();
    // 记录合并op
    viewOpToAppend_.emplace_back(ViewOp{startTensor, endTensor, newOffset, newDynOffset, newDynValidShape,
                                        lastViewAttr->GetTo(), hasCopyInMode, std::move(copyInModeValue),
                                        hasL1PaddingMode, std::move(l1PaddingMode), hasKIndex, kIndex, hasIsGemv,
                                        std::move(isGemv), span, scopeInfo, opcode, tokenDependency});
}

Status MergeViewAssembleUtils::MergeAssembleChain(Function& function, Operation& operation,
                                                  std::vector<Operation*>& chain, int effectiveScopeId)
{
    // 1. 初始化操作链
    InitAssembleChain(operation, chain);

    int newScopeId = operation.GetScopeId();
    if (effectiveScopeId == -1 && newScopeId != -1) {
        effectiveScopeId = newScopeId;
    }

    // 2. 处理消费者
    const auto& consumers = GetConsumers(operation);
    bool chainEnd = consumers.assembleConsumers.empty() || consumers.hasAssembleChainStopper;
    Status status = ProcessAssembleConsumers(function, consumers, chain, chainEnd, effectiveScopeId);
    if (status != SUCCESS) {
        return status;
    }

    // 3. 处理链尾情况
    if (chainEnd && chain.size() > 1) {
        status = ProcessAssembleChainEnd(function, chain, operation);
        if (status != SUCCESS) {
            return status;
        }
    }

    chain.pop_back();
    return SUCCESS;
}

void MergeViewAssembleUtils::InitAssembleChain(Operation& operation, std::vector<Operation*>& chain)
{
    visitedOp_.insert(operation.opmagic);
    chain.emplace_back(&operation);
}

Status MergeViewAssembleUtils::ProcessAssembleConsumers(Function& function, const ConsumerCacheEntry& consumers,
                                                        std::vector<Operation*>& chain, bool& chainEnd,
                                                        int effectiveScopeId)
{
    if (consumers.hasAssembleChainStopper || consumers.assembleConsumers.empty()) {
        return SUCCESS;
    }
    Operation* currentOp = chain.back();
    if (currentOp == nullptr || currentOp->oOperand.empty() || consumers.producerCount != 1) {
        chainEnd = true;
        return SUCCESS;
    }
    std::vector<Operation*> currentProducers(currentOp->oOperand.front()->GetProducers().begin(),
                                             currentOp->oOperand.front()->GetProducers().end());
    if (HasSplitVersionContribution(currentOp->oOperand.front(), currentProducers)) {
        chainEnd = true;
        return SUCCESS;
    }
    for (auto& op : consumers.assembleConsumers) {
        if (!op) {
            APASS_LOG_ERROR_F(Elements::Function, "Null consumer operation found.");
            return FAILED;
        }
        if (visitedOp_.count(op->GetOpMagic()) != 0 && op->GetIOperands().size() == 1) {
            chainEnd = true;
            continue;
        }
        if (IsAssembleLikeOpcode(op->GetOpcode())) {
            int consumerScopeId = op->GetScopeId();
            if (effectiveScopeId != -1 && consumerScopeId != -1 && effectiveScopeId != consumerScopeId) {
                chainEnd = true;
                continue;
            }
            if (!CanMergeAssembleLikeChain(chain, op->GetOpcode())) {
                chainEnd = true;
                continue;
            }
            if (!IsRmwModeAttrCompatible(chain, *op)) {
                chainEnd = true;
                continue;
            }
            Status status = MergeAssembleChain(function, *op, chain, effectiveScopeId);
            if (status != SUCCESS) {
                APASS_LOG_ERROR_F(Elements::Function, "Run MergeAssembleChain failed.");
                return status;
            }
            continue;
        }
        chainEnd = true;
    }
    return SUCCESS;
}

Status MergeViewAssembleUtils::ProcessAssembleChainEnd(Function& function, std::vector<Operation*>& chain,
                                                       Operation& operation)
{
    (void)operation;
    // 验证链有效性
    if (chain.front()->iOperand.empty() || chain.back()->oOperand.empty()) {
        APASS_LOG_ERROR_F(Elements::Function, "Invalid chain operations.");
        return FAILED;
    }
    auto& startTensor = chain.front()->iOperand.front();
    auto& endTensor = chain.back()->oOperand.front();
    if (!startTensor || !endTensor) {
        APASS_LOG_ERROR_F(Elements::Function, "Null tensor found in chain.");
        return FAILED;
    }
    if (HasTokenDependency(chain)) {
        std::unordered_map<Operation*, Operation*> contraction;
        for (auto* op : chain) {
            contraction.emplace(op, chain.front());
        }
        if (WouldCreateCycleAfterContraction(function, contraction)) {
            return SUCCESS;
        }
    }
    TokenDependency tokenDependency;
    CollectLinearTokenDependency(function, chain, tokenDependency);
    // 计算合并offset
    auto [newOffset, newDynOffset] = CalculateAssembleOffsets(chain, startTensor->offset.size());
    // 获取链路上第一个非空的span
    ir::Span firstSpan = GetFirstSpan(chain);
    Operation::ScopeInfo chainScopeInfo = GetChainScopeInfo(chain);
    RmwModeAttrState rmwModeAttr = GetChainRmwModeAttr(chain);
    if (rmwModeAttr.conflict) {
        APASS_LOG_ERROR_F(Elements::Function, "Assemble chain has conflicting rmw mode attributes.");
        return FAILED;
    }
    AtomicSemanticAttrState atomicSemanticAttr = GetChainAtomicSemanticAttr(chain);
    // 4. 记录并清理
    RecordAssembleOperation(startTensor, endTensor, newOffset, newDynOffset, firstSpan, chainScopeInfo,
                            GetRmwModeAttrKey(rmwModeAttr), GetMergedAssembleOpcode(chain), tokenDependency,
                            atomicSemanticAttr.fromReduceAcc, atomicSemanticAttr.fromExplicitRmw);
    ClearLinearTokenDependency(function, chain, tokenDependency);
    for (auto* op : chain) {
        op->SetAsDeleted();
    }
    function.GetTensorMap().Erase(endTensor);

    return SUCCESS;
}

std::pair<std::vector<int64_t>, std::vector<SymbolicScalar>> MergeViewAssembleUtils::CalculateAssembleOffsets(
    const std::vector<Operation*>& chain, size_t offsetSize)
{
    std::vector<int64_t> newOffset(offsetSize, 0);
    std::vector<SymbolicScalar> newDynOffset;
    for (size_t i = 0; i < chain.size(); ++i) {
        const auto& assemble = chain[i];
        if (!assemble) {
            return {};
        }
        auto assembleOpAttribute = std::dynamic_pointer_cast<AssembleOpAttribute>(assemble->GetOpAttribute());
        if (!assembleOpAttribute) {
            return {};
        }
        if (i == 0) {
            newOffset = assembleOpAttribute->GetToOffset();
            newDynOffset = assembleOpAttribute->GetToDynOffset();
            continue;
        }
        auto ret = TensorOffset::Add(newOffset, newDynOffset, assembleOpAttribute->GetToOffset(),
                                     assembleOpAttribute->GetToDynOffset());
        if (!ret.first.empty()) {
            newOffset = ret.first;
            newDynOffset = ret.second;
        }
    }
    return {newOffset, newDynOffset};
}

void MergeViewAssembleUtils::RecordAssembleOperation(
    const std::shared_ptr<LogicalTensor>& input, const std::shared_ptr<LogicalTensor>& output,
    const std::vector<int64_t>& offset, const std::vector<SymbolicScalar>& dynOffset, const ir::Span& span,
    const Operation::ScopeInfo& scopeInfo, const std::string& rmwModeAttr, Opcode opcode,
    const TokenDependency& tokenDependency, bool atomicFromReduceAcc, bool atomicFromExplicitRmw)
{
    assembleOpToAppend_.emplace_back(AssembleOp{input, output, offset, dynOffset, span, scopeInfo, rmwModeAttr, opcode,
                                                tokenDependency, atomicFromReduceAcc, atomicFromExplicitRmw});
}

} // namespace npu::tile_fwk
