/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "merge_dangling_assemble_output.h"

#include <algorithm>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "interface/function/function.h"
#include "passes/pass_log/pass_log.h"
#include "passes/pass_utils/token_utils.h"

#define MODULE_NAME "MergeDanglingAssembleOutput"

namespace npu::tile_fwk {

std::string MergeDanglingAssembleOutput::BuildLogicalTensorSignature(const LogicalTensor& tensor)
{
    std::ostringstream stream;
    const auto appendVector = [&stream](const auto& values) {
        stream << '[';
        for (size_t index = 0; index < values.size(); ++index) {
            if (index != 0) {
                stream << ',';
            }
            stream << values[index];
        }
        stream << ']';
    };
    stream << DataType2String(tensor.Datatype()) << ':' << static_cast<int>(tensor.Format()) << ':';
    appendVector(tensor.GetShape());
    stream << ':';
    appendVector(tensor.GetOffset());
    stream << ':';
    appendVector(tensor.GetDynOffset());
    stream << ':' << static_cast<int>(tensor.GetMemoryTypeOriginal()) << ':'
           << static_cast<int>(tensor.GetMemoryTypeToBe());
    return stream.str();
}

MergeDanglingAssembleOutput::VersionGroups MergeDanglingAssembleOutput::BuildVersionGroups(Function& function)
{
    std::unordered_map<RawTensor*, VersionGroup> byRawTensor;
    for (auto& op : function.Operations(false)) {
        if (op.GetOpcode() != Opcode::OP_ASSEMBLE || op.GetOutputOperandSize() != 1) {
            continue;
        }
        auto output = op.GetOutputOperand(0);
        if (output == nullptr || output->GetRawTensor() == nullptr) {
            continue;
        }
        byRawTensor[output->GetRawTensor().get()].push_back(
            AssembleVersion{&op, output, BuildLogicalTensorSignature(*output)});
    }

    VersionGroups groups;
    groups.reserve(byRawTensor.size());
    for (auto& [rawTensor, group] : byRawTensor) {
        (void)rawTensor;
        groups.push_back(std::move(group));
    }
    return groups;
}

const MergeDanglingAssembleOutput::AssembleVersion* MergeDanglingAssembleOutput::FindMergeTarget(
    const VersionGroup& group, size_t index)
{
    const AssembleVersion* canonicalSink = nullptr;
    for (size_t next = index + 1; next < group.size(); ++next) {
        if (group[index].output->GetRawTensor() != group[next].output->GetRawTensor() ||
            group[index].signature != group[next].signature) {
            continue;
        }
        canonicalSink = &group[next];
        if (!group[next].output->GetConsumers().empty()) {
            return &group[next];
        }
    }
    return canonicalSink;
}

bool MergeDanglingAssembleOutput::TokenHasConsumer(Function& function, const ir::VarPtr& token)
{
    for (auto& op : function.Operations(false)) {
        for (const auto& consumed : op.tokens_) {
            if (consumed.get() == token.get()) {
                return true;
            }
        }
    }
    return false;
}

void MergeDanglingAssembleOutput::PruneRedundantTokens(Function& function, Operation& producer,
                                                       const LogicalTensorPtr& target)
{
    const auto resultTokens = producer.result_token_;
    if (resultTokens.empty()) {
        return;
    }
    // The merged data edge already orders producer before each target consumer.
    for (const auto& resultToken : resultTokens) {
        for (auto* consumer : target->GetConsumers()) {
            if (consumer == &producer) {
                continue;
            }
            auto& tokens = consumer->tokens_;
            auto newEnd = std::remove_if(tokens.begin(), tokens.end(), [&resultToken](const ir::VarPtr& token) {
                return token.get() == resultToken.get();
            });
            tokens.erase(newEnd, tokens.end());
        }
    }
    auto& producerTokens = producer.result_token_;
    producerTokens.erase(
        std::remove_if(producerTokens.begin(), producerTokens.end(),
                       [this, &function](const ir::VarPtr& token) { return !TokenHasConsumer(function, token); }),
        producerTokens.end());
}

LogicalTensorPtr MergeDanglingAssembleOutput::MergeVersion(Function& function, Operation& producer,
                                                           const LogicalTensorPtr& target)
{
    auto oldOutput = producer.GetOutputOperand(0);
    if (oldOutput == target) {
        return nullptr;
    }
    producer.ReplaceOOperand(0, target);
    PruneRedundantTokens(function, producer, target);
    for (auto& outcast : function.outCasts_) {
        if (outcast == oldOutput) {
            outcast = target;
        }
    }
    return oldOutput;
}

void MergeDanglingAssembleOutput::ReclaimDetachedTensors(Function& function,
                                                         const std::vector<LogicalTensorPtr>& tensors)
{
    const auto& outcasts = function.GetOutcast();
    std::unordered_set<LogicalTensor*> reclaimed;
    for (const auto& tensor : tensors) {
        if (tensor == nullptr || !tensor->GetConsumers().empty() ||
            std::find(outcasts.begin(), outcasts.end(), tensor) != outcasts.end() || !tensor->GetProducers().empty() ||
            !reclaimed.insert(tensor.get()).second) {
            continue;
        }
        if (function.GetTensorMap().GetTensorByMagic(tensor->GetMagic()) == tensor) {
            function.GetTensorMap().Erase(tensor);
        }
    }
}

Status MergeDanglingAssembleOutput::RunOnFunction(Function& function)
{
    APASS_LOG_INFO_F(Elements::Function, "===> Start MergeDanglingAssembleOutput.");

    std::vector<LogicalTensorPtr> detachedTensors;
    for (const auto& group : BuildVersionGroups(function)) {
        for (size_t index = 0; index < group.size(); ++index) {
            const auto& version = group[index];
            // Outcast slots are redirected during merge; only Operation consumers preserve a version snapshot.
            if (!version.output->GetConsumers().empty()) {
                continue;
            }
            const auto* target = FindMergeTarget(group, index);
            if (target == nullptr) {
                continue;
            }
            auto detached = MergeVersion(function, *version.producer, target->output);
            if (detached != nullptr) {
                detachedTensors.push_back(std::move(detached));
            }
        }
    }
    ReclaimDetachedTensors(function, detachedTensors);
    if (TokenUtils::RebuildTokenDependencies(function) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Function, "Failed to rebuild token dependencies after merging versions.");
        return FAILED;
    }

    APASS_LOG_INFO_F(Elements::Function, "===> End MergeDanglingAssembleOutput.");
    return SUCCESS;
}

} // namespace npu::tile_fwk
