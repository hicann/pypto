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
 * \file assign_memory_type.h
 * \brief
 */

#ifndef TILE_FWK_ASSIGN_MEMORY_TYPE_H
#define TILE_FWK_ASSIGN_MEMORY_TYPE_H

#include <queue>
#include <string>
#include <unordered_set>
#include <vector>
#include "passes/pass_interface/pass.h"
#include "interface/operation/opcode.h"
#include "passes/tile_graph_pass/data_path/convert_op_inserter.h"
#include "tilefwk/platform.h"
#include "tilefwk/data_type.h"
#include "passes/pass_check/assign_memory_type_checker.h"
#include "passes/pass_utils/infer_shape_utils.h"

namespace npu::tile_fwk {
class AssignMemoryType : public Pass {
public:
    AssignMemoryType() : Pass("AssignMemoryType") {}
    void SpecialCallInterfaceToBeDeleted(Function& function) { RunOnFunction(function); }

private:
    Status PreCheck(Function& function) override;
    Status PostCheck(Function& function) override;
    Status RunOnFunction(Function& function) override;
    Status InsertConvertOpsAndInferShape(Function& function);

    Status AssignConfirmedMemoryTypes(Function& function);

    Status AssignOpcodeDefinedMemoryTypes(Operation& operation);

    Status AssignMatmulInputRequirements(Operation& operation);

    Status AssignReduceAccInputRequirements(Operation& operation);

    Status AssignViewAttrMemoryType(Operation& operation);

    Status AssignAssembleAttrMemoryType(Operation& operation);

    Status AssignInOutCastMemoryTypes(Function& function);

    Status EnsureAllConsumerRequirementsExist(Function& function);

    Status InferUncertainMemoryTypes(Function& function);

    Status GetFirstInputOutputIfOpcode(
        Operation& operation, Opcode expectedOpcode, const std::string& action, LogicalTensorPtr& input,
        LogicalTensorPtr& output, bool& shouldHandle) const;

    Status InferViewMemoryType(Operation& operation);

    Status InferViewOutputFromRequirement(const LogicalTensorPtr& output, MemoryType& outputOriginal);

    Status InferViewKnownInputOutput(
        Operation& operation, const LogicalTensorPtr& input, MemoryType inputOriginal, MemoryType outputOriginal);

    Status InferViewKnownInputUnknownOutput(
        Operation& operation, const LogicalTensorPtr& input, const LogicalTensorPtr& output, MemoryType inputOriginal);

    bool TryHandleUnalignedView(
        Operation& operation, const LogicalTensorPtr& input, MemoryType inputOriginal, MemoryType outputOriginal);

    bool CanUseDirectViewPath(Operation& operation, MemoryType from, MemoryType to);

    bool TryHandleSpecialDirectMemoryPath(Operation& operation, MemoryType from, MemoryType to, bool& directPath);

    bool IsAdvancedMemoryPath(MemoryType from, MemoryType to) const;

    bool HasParallelDifferentConsumerRequirement(const LogicalTensorPtr& tensor, MemoryType targetType) const;

    bool IsViewFromOffsetAligned(Operation& operation) const;

    bool HasDynOffsetViewAndReshape(Operation& operation, const LogicalTensorPtr& output) const;

    Status InferAssembleMemoryType(
        Function& function, Operation& operation, std::unordered_set<LogicalTensorPtr>& inferredAssembleOutputs);

    Status InferAssembleMemoryType(Operation& operation);

    Status AssignAssembleToOutCastRequirement(Operation& operation);

    Status InferAssembleOutputMemoryType(const LogicalTensorPtr& output);

    bool HasAssembleInputOutputElementCountMismatch(const LogicalTensorPtr& output) const;

    Status TryInferAssembleOutputByTempOriginal(const LogicalTensorPtr& output, MemoryType tempOriginal, bool& handled);

    bool AreAssembleDirectPathsSupported(const LogicalTensorPtr& output, MemoryType targetOriginal);

    bool IsAssembleProducer(Operation* operation) const;

    MemoryType GetAssembleInputType(Operation& operation) const;

    Status ApplyAssembleDirectOutputOriginal(const LogicalTensorPtr& output, MemoryType targetOriginal);

    Status SyncAssembleInputRequirementAndAttr(
        Operation& operation, MemoryType fallbackType, const std::string& reason);

    Status ApplyAssembleDdrOutputWithInputOriginals(
        const LogicalTensorPtr& output, const std::string& originalReason, const std::string& inputReason);

    Status FillAssembleInputRequirementsFromOriginal(const LogicalTensorPtr& output, const std::string& reason);

    Status TryInferAssembleOutputByProducerCandidate(const LogicalTensorPtr& output, bool& handled);

    MemoryType InferAssembleProducerCandidate(const LogicalTensorPtr& output, bool& hasConflict) const;

    Status ApplyAssembleProducerCandidate(const LogicalTensorPtr& output, MemoryType producerCandidate);

    MemoryType InferAssembleTempOriginal(const LogicalTensorPtr& output) const;

    bool CanUseDirectAssemblePath(Operation& operation, MemoryType from, MemoryType to);

    bool IsAssembleToOffsetAligned(Operation& operation, const LogicalTensorPtr& output);

    bool FitsAssembleOutputMemoryLimit(const LogicalTensorPtr& output, MemoryType memoryType) const;

    Status InferReshapeMemoryType(Operation& operation);

    MemoryType GetReshapeInputRequirement(
        Operation& operation, const LogicalTensorPtr& input, MemoryType inputOriginal);

    Status InferReshapeOutputFromRequirement(const LogicalTensorPtr& output, MemoryType& outputOriginal);

    MemoryType InferUniqueRequirementThroughViewConsumers(const LogicalTensorPtr& tensor) const;

    MemoryType InferUniqueRequirementThroughViewConsumers(
        const LogicalTensorPtr& tensor, std::unordered_set<const LogicalTensor*>& visitedTensors) const;

    bool HasRequirementThroughViewConsumers(
        const LogicalTensorPtr& tensor, MemoryType targetRequirement,
        std::unordered_set<const LogicalTensor*>& visitedTensors) const;

    bool CanUseUbForReshape(
        const LogicalTensorPtr& input, const LogicalTensorPtr& output, MemoryType inputRequirement,
        MemoryType outputOriginal) const;

    Status ApplyReshapeMemoryType(
        Operation& operation, const LogicalTensorPtr& input, const LogicalTensorPtr& output, bool isDynamic,
        bool canUseUb);

    Status InferViewTypeMemoryType(Operation& operation);

    Status TryInferViewTypeFromProducerView(
        Operation& operation, const LogicalTensorPtr& input, const LogicalTensorPtr& output, MemoryType targetType,
        bool& handled);

    Status InferViewTypeInput(
        Operation& operation, const LogicalTensorPtr& input, const LogicalTensorPtr& output, MemoryType targetType);

    bool KeepSplitReshapeUb(Operation& operation, const LogicalTensorPtr& input, const LogicalTensorPtr& output);

    bool IsDynamicReshape(Operation& operation, const LogicalTensorPtr& output) const;

    bool FitsTensorInUb(const LogicalTensorPtr& tensor) const;

    Status ApplyOtherSpecialOpcodeRules(Function& function);

    Status HandleNopMemoryType(Operation& operation);

    Status ApplyOversizedLocalBufferFallback(Function& function);

    Status ApplyOversizedLocalBufferFallback(Operation& operation);

    bool ExceedsMemoryLimit(const LogicalTensorPtr& tensor, size_t threshold) const;

    Status ApplyPlatformPathFallbackRules(Function& function);

    Status ResolveMemoryUnknowns(Function& function);

    Status ResolveTensorMemoryUnknowns(const LogicalTensorPtr& tensor);

    Status SyncViewAssembleMemoryAttrs(Function& function);

    Status SyncViewMemoryAttr(Operation& operation);

    Status SyncAssembleMemoryAttr(Operation& operation);

    MemoryType InferOriginalFromRequirements(const LogicalTensorPtr& tensor) const;

    Status SyncTensorToBe(Function& function);

    Status SetOriginalChecked(
        const LogicalTensorPtr& tensor, MemoryType memoryType, const std::string& reason = "unknown",
        bool allowOverride = false);

    void ForceSetOriginal(const LogicalTensorPtr& tensor, MemoryType memoryType, const std::string& reason = "unknown");

    Status SetRequirementChecked(
        const LogicalTensorPtr& tensor, Operation& operation, MemoryType memoryType,
        const std::string& reason = "unknown", bool allowOverride = false);

    void ForceSetRequirement(
        const LogicalTensorPtr& tensor, Operation& operation, MemoryType memoryType,
        const std::string& reason = "unknown");

    void FillUnknownRequirementsWith(const LogicalTensorPtr& tensor, MemoryType memoryType, const char* reason);
    bool AreAllConsumerRequirements(const LogicalTensorPtr& tensor, MemoryType memoryType) const;
    void DowngradeConsumerRequirements(const LogicalTensorPtr& tensor, MemoryType fromType);
    void ProcessL0C2L1SmallToLarge(Function& function);
    void ProcessL0C2L1LargeToSmall(Function& function);
    bool CheckUBTileShape(const LogicalTensorPtr& output);
    bool CheckConsumerViewShapeMultiple(const LogicalTensorPtr& output, const LogicalTensorPtr& input);
    void ProcessL0C2UBSmallToLarge(Function& function);
    void ProcessL0C2UBLargeToSmall(Function& function);
    void ProcessUB2L1SmallToLarge(Function& function);
    void ProcessUB2L1LargeToSmall(Function& function);
    bool ShouldSkipUB2L1SmallToLarge(const LogicalTensorPtr& iOperand, const LogicalTensorPtr& oOperand) const;
    bool IsDimMultiple(const Shape& shape1, const Shape& shape2);
    bool CheckInnerAxisC0Size(const LogicalTensorPtr& input, const LogicalTensorPtr& output) const;
    size_t CalcNZTensorSize(const LogicalTensorPtr& tensor) const;
    int64_t CalcLineOffset(const Shape& shape, const Offset& offset);
    ConvertInserter inserter;
    AssignMemoryTypeChecker checker;
};
static constexpr double UB_THRESHOLD_ASSEMBLE = 0.35;
static constexpr double UB_THRESHOLD_NORMAL = 1.0;
static constexpr double L1_THRESHOLD = 0.5;
static constexpr uint16_t L0C_TILE_SIZE = 16;
} // namespace npu::tile_fwk

#endif // TILE_FWK_ASSIGN_MEMORY_TYPE_H
