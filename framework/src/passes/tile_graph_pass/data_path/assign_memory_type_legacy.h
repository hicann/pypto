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

#ifndef TILE_FWK_ASSIGN_MEMORY_TYPE_LEGACY_H
#define TILE_FWK_ASSIGN_MEMORY_TYPE_LEGACY_H

#include <queue>
#include <string>
#include <unordered_set>
#include <vector>
#include "passes/pass_interface/pass.h"
#include "interface/operation/opcode.h"
#include "passes/tile_graph_pass/data_path/convert_op_inserter_legacy.h"
#include "tilefwk/platform.h"
#include "tilefwk/data_type.h"
#include "passes/pass_check/assign_memory_type_checker.h"
#include "passes/pass_utils/infer_shape_utils.h"

namespace npu::tile_fwk::legacy {
class AssignMemoryType : public Pass {
public:
    AssignMemoryType() : Pass("AssignMemoryType") {}
    Status RunLegacy(Function& function) { return RunOnFunction(function); }

private:
    Status PreCheck(Function& function) override;
    Status PostCheck(Function& function) override;
    Status RunOnFunction(Function& function) override;
    Status InsertConvertOpsAndInferShape(Function& function);

    Status AssignConfirmedMemoryTypes(Function& function);

    Status AssignMatmulInputRequirements(Operation& operation);

    Status AssignViewAttrMemoryType(Operation& operation);

    Status AssignAssembleAttrMemoryType(Operation& operation);

    Status InferUncertainMemoryTypes(Function& function);

    Status InferViewMemoryType(Operation& operation);

    Status InferViewOutputFromRequirement(const LogicalTensorPtr& output, MemoryType& outputOriginal);

    Status InferViewKnownInputOutput(Operation& operation, const LogicalTensorPtr& input, MemoryType inputOriginal,
                                     MemoryType outputOriginal);

    Status InferViewKnownInputUnknownOutput(Operation& operation, const LogicalTensorPtr& input,
                                            const LogicalTensorPtr& output, MemoryType inputOriginal);

    bool TryHandleUnalignedView(Operation& operation, const LogicalTensorPtr& input, MemoryType inputOriginal,
                                MemoryType outputOriginal);

    bool CanUseDirectViewPath(Operation& operation, MemoryType from, MemoryType to);

    bool TryHandleSpecialDirectMemoryPath(Operation& operation, MemoryType from, MemoryType to, bool& directPath);

    bool HasParallelDifferentConsumerRequirement(const LogicalTensorPtr& tensor, MemoryType targetType) const;

    bool IsViewFromOffsetAligned(Operation& operation) const;

    bool HasDynOffsetViewAndReshape(Operation& operation, const LogicalTensorPtr& output) const;

    bool HasTransDataConsumer(const LogicalTensorPtr& tensor) const;

    bool HasPermuteProducerAndTransDataDownstream(const LogicalTensorPtr& input, const LogicalTensorPtr& output) const;

    Status InferAssembleMemoryType(Function& function, Operation& operation,
                                   std::unordered_set<LogicalTensorPtr>& inferredAssembleOutputs);

    Status InferAssembleMemoryType(Operation& operation);

    Status AssignAssembleToOutCastRequirement(Operation& operation);

    Status InferAssembleOutputMemoryType(const LogicalTensorPtr& output);

    bool HasAssembleInputOutputElementCountMismatch(const LogicalTensorPtr& output) const;

    Status TryInferAssembleOutputByTempOriginal(const LogicalTensorPtr& output, MemoryType tempOriginal, bool& handled);

    bool AreAssembleDirectPathsSupported(const LogicalTensorPtr& output, MemoryType targetOriginal);

    Status ApplyAssembleDirectOutputOriginal(const LogicalTensorPtr& output, MemoryType targetOriginal);

    Status SyncAssembleInputRequirementAndAttr(Operation& operation, MemoryType fallbackType,
                                               const std::string& reason);

    Status ApplyAssembleDdrOutputWithInputOriginals(const LogicalTensorPtr& output, const std::string& originalReason,
                                                    const std::string& inputReason);

    Status FillAssembleInputRequirementsFromOriginal(const LogicalTensorPtr& output, const std::string& reason);

    Status TryInferAssembleOutputByProducerCandidate(const LogicalTensorPtr& output, bool& handled);

    MemoryType InferAssembleProducerCandidate(const LogicalTensorPtr& output, bool& hasConflict) const;

    Status ApplyAssembleProducerCandidate(const LogicalTensorPtr& output, MemoryType producerCandidate);

    MemoryType InferAssembleTempOriginal(const LogicalTensorPtr& output) const;

    bool AreAllConsumersL1Views(const LogicalTensorPtr& output) const;

    bool CanUseDirectAssemblePath(Operation& operation, MemoryType from, MemoryType to);

    bool IsAssembleToOffsetAligned(Operation& operation, const LogicalTensorPtr& output);

    Status InferReshapeMemoryType(Operation& operation);

    Status InferReshapeL0C2UBAndUB2L1PatternLiteNPU(Operation& op);

    bool IsReshapeCubeToVecL0C2UBPattern(Operation& op);

    bool IsReshapeVecToCubeUB2L1Pattern(Operation& op);

    bool IsReshapeVecToCubeUB2L1ProducerPattern(const std::set<Operation*, LogicalTensor::CompareOp>& producers);

    bool IsReshapeVecToCubeUB2L1ConsumerPattern(const std::set<Operation*, LogicalTensor::CompareOp>& consumers);

    Status InferViewTypeMemoryType(Operation& operation);

    Status TryInferViewTypeFromProducerView(Operation& operation, const LogicalTensorPtr& input,
                                            const LogicalTensorPtr& output, MemoryType targetType, bool& handled);

    bool KeepSplitReshapeUb(Operation& operation, const LogicalTensorPtr& input, const LogicalTensorPtr& output);

    Status ApplyOversizedLocalBufferFallback(Function& function);

    Status ApplyOversizedLocalBufferFallback(Operation& operation);

    Status DowngradeOversizedViewInputRequirement(Operation& operation);

    Status ApplyPlatformPathFallbackRules(Function& function);

    Status ResolveMemoryUnknowns(Function& function);

    Status SyncViewAssembleMemoryAttrs(Function& function);

    Status SyncViewMemoryAttr(Operation& operation);

    Status SyncAssembleMemoryAttr(Operation& operation);

    bool AreAllConsumerRequirements(const LogicalTensorPtr& tensor, MemoryType memoryType) const;
    void DowngradeConsumerRequirements(const LogicalTensorPtr& tensor, MemoryType fromType);
    void ProcessL0C2L1SmallToLarge(Function& function);
    void ProcessL0C2L1LargeToSmall(Function& function);
    bool CheckConsumerViewShapeMultiple(const LogicalTensorPtr& output, const LogicalTensorPtr& input);
    void ProcessL0C2UBSmallToLarge(Function& function);
    void ProcessL0C2UBLargeToSmall(Function& function);
    void ProcessUB2L1SmallToLarge(Function& function);
    void ProcessUB2L1LargeToSmall(Function& function);
    void ProcessShapeTransportFallback(Function& function);
    bool IsAllowedTransport(const LogicalTensorPtr& prodOut, const LogicalTensorPtr& consIn) const;
    bool ShouldSkipUB2L1SmallToLarge(const LogicalTensorPtr& iOperand, const LogicalTensorPtr& oOperand) const;
    size_t CalcNZTensorSize(const LogicalTensorPtr& tensor) const;
    int64_t CalcLineOffset(const Shape& shape, const Offset& offset) const;
    ConvertInserter inserter;
    AssignMemoryTypeChecker checker;
};
static constexpr double UB_THRESHOLD_ASSEMBLE = 0.35;
static constexpr double UB_THRESHOLD_NORMAL = 1.0;
static constexpr double L1_THRESHOLD = 0.5;
static constexpr uint16_t L0C_TILE_SIZE = 16;
static constexpr uint16_t INT8_ALIGN_SIZE = 32;
} // namespace npu::tile_fwk::legacy

#endif // TILE_FWK_ASSIGN_MEMORY_TYPE_LEGACY_H
