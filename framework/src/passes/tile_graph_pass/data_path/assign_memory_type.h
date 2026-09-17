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

#include <set>
#include <string>
#include <unordered_set>
#include <vector>
#include "passes/pass_interface/pass.h"
#include "interface/operation/opcode.h"
#include "passes/tile_graph_pass/data_path/convert_op_inserter.h"
#include "passes/pass_utils/graph_utils.h"
#include "tilefwk/platform.h"
#include "tilefwk/data_type.h"
#include "passes/pass_check/assign_memory_type_checker.h"
#include "passes/pass_utils/infer_shape_utils.h"

namespace npu::tile_fwk {
class AssignMemoryType : public Pass {
public:
    AssignMemoryType() : Pass("AssignMemoryType") {}

private:
    Status PreCheck(Function& function) override;
    Status PostCheck(Function& function) override;
    Status RunOnFunction(Function& function) override;
    Status InsertConvertOpsAndInferShape(Function& function);

    Status AssignConfirmedMemoryTypes(Function& function);

    Status AssignDynamicShapeOutputMemoryType(Operation& operation);

    Status AssignMatmulInputRequirements(Operation& operation);

    Status AssignViewAttrMemoryType(Operation& operation);

    Status AssignSliceInputRequirement(Operation& operation);

    Status AssignAssembleAttrMemoryType(Operation& operation);

    Status InferUncertainMemoryTypes(Function& function);

    Status InferViewMemoryType(Operation& operation);

    Status InferSliceMemoryType(Operation& operation);

    Status InferContractMemoryType(Operation& operation);

    MemoryType InferOriginalFromOutputRequirements(const LogicalTensorPtr& tensor) const;

    MemoryType InferOriginalFromOutputRequirements(const LogicalTensorPtr& tensor,
                                                   std::unordered_set<const LogicalTensor*>& visitedTensors) const;

    MemoryType InferRequirementFromInputOriginals(const LogicalTensorPtr& tensor) const;

    MemoryType InferRequirementFromInputOriginals(const LogicalTensorPtr& tensor,
                                                  std::unordered_set<const LogicalTensor*>& visitedTensors) const;

    bool CanUseDirectViewPath(Operation& operation, MemoryType from, MemoryType to);

    bool TryHandleSpecialDirectMemoryPath(Operation& operation, MemoryType from, MemoryType to, bool& directPath);

    bool HasParallelDifferentConsumerRequirement(const LogicalTensorPtr& tensor, MemoryType targetType) const;

    bool HasDifferentConsumerRequirement(const LogicalTensorPtr& tensor, MemoryType targetType) const;

    Status InferAssembleMemoryType(Function& function, Operation& operation,
                                   std::unordered_set<LogicalTensorPtr>& inferredAssembleOutputs);

    // 后续有公共方法后替换为通过 rawTensor 查找所有 logicalTensor 的公共接口
    TensorSet GetLogicalTensorsByRawTensor(Function& function, const LogicalTensorPtr& tensor) const;

    void PropagateMemoryTypeToRawTensorSiblings(Function& function, const LogicalTensorPtr& output,
                                                std::unordered_set<LogicalTensorPtr>& inferredAssembleOutputs);

    Status ResolveInconsistentRawTensorMemoryTypes(Function& function);

    Status InferAssembleMemoryType(Operation& operation);

    MemoryType InferParallelAssembleInputRequirement(const LogicalTensorPtr& output) const;

    Status SetParallelAssembleInputRequirements(const LogicalTensorPtr& output, MemoryType memoryType,
                                                const std::string& reason);

    bool CanUseDirectAssemblePath(Operation& operation, MemoryType from, MemoryType to);

    Status IsAssembleToOffsetAligned(Operation& operation, const LogicalTensorPtr& output, bool& aligned);

    Status InferReshapeMemoryType(Operation& operation);

    Status InferReshapeL0C2UBAndUB2L1PatternLiteNPU(Operation& op);

    bool IsReshapeCubeToVecL0C2UBPattern(Operation& op);

    bool IsReshapeVecToCubeUB2L1Pattern(Operation& op);

    bool IsReshapeVecToCubeUB2L1ProducerPattern(const std::set<Operation*, LogicalTensor::CompareOp>& producers);

    bool IsReshapeVecToCubeUB2L1ConsumerPattern(const std::set<Operation*, LogicalTensor::CompareOp>& consumers);

    Status InferViewTypeMemoryType(Operation& operation);

    Status TryInferViewTypeFromProducerSlice(Operation& operation, const LogicalTensorPtr& input,
                                             const LogicalTensorPtr& output, MemoryType targetType, bool& handled);

    Status CanKeepContractProducersInUb(const LogicalTensorPtr& tensor, bool& canKeep);

    Status IsSliceFromOffsetAligned(Operation& sliceOp, const LogicalTensorPtr& input, bool& aligned);

    Status CanKeepSliceConsumersInUb(const LogicalTensorPtr& tensor, bool& canKeep);

    Status HasNonZeroSliceFromOffset(const LogicalTensorPtr& tensor, bool& hasNonZero);

    Status KeepSplitReshapeUb(Operation& operation, const LogicalTensorPtr& input, const LogicalTensorPtr& output,
                              bool& kept);

    Status ApplyOversizedLocalBufferFallback(Function& function);

    Status ApplyOversizedLocalBufferFallback(Operation& operation);

    Status DowngradeOversizedSliceInputRequirement(Operation& operation);

    Status ApplyPlatformPathUpgradeRules(Function& function);

    Status ResolveMemoryUnknowns(Function& function);

    bool ShouldResolveExplicitUnknownRequirementToDdr(const Function& function, const LogicalTensorPtr& tensor) const;

    Status SyncViewAssembleMemoryAttrs(Function& function);

    Status FixViewAssembleSemanticMismatch(Function& function);

    Status SyncViewMemoryAttr(Operation& operation);

    Status SyncAssembleMemoryAttr(Operation& operation);

    Status MarkA5SimtGatherElement(Function& function);

    Status FallbackSameMemoryMoveOps(Function& function);

    Status ResolveUnalignedUbSlices(Function& function);

    Status ProcessL0C2L1SmallToLarge(Function& function);
    Status ProcessL0C2L1LargeToSmall(Function& function);
    bool CheckConsumerSliceShapeMultiple(const LogicalTensorPtr& output, const LogicalTensorPtr& input);
    bool AreAllSliceConsumerShapesPreserved(const LogicalTensorPtr& tensor) const;
    Status ProcessL0C2UBSmallToLarge(Function& function);
    Status ProcessL0C2UBLargeToSmall(Function& function);
    Status ProcessUB2UBContractSlice(Function& function);
    Status ProcessUB2L1SmallToLarge(Function& function);
    Status ProcessUB2L1LargeToSmall(Function& function);
    Status ProcessL1DdrL1(Function& function);
    Status ProcessDdrMultiReshape(Function& function);
    bool ShouldSkipUB2L1SmallToLarge(const LogicalTensorPtr& iOperand, const LogicalTensorPtr& oOperand) const;
    Status TryUpgradeSliceContractPath(Operation& sliceOp, MemoryType sourceType, MemoryType targetType,
                                       const std::string& reason, bool requireMatrixShape, bool checkUbTileShape,
                                       bool checkUb2L1Constraints);
    Status TryUpgradeSingleContractSlicePath(Operation& contractOp, MemoryType sourceType, MemoryType targetType,
                                             const std::string& reason, bool requireMatrixShape, bool checkUbTileShape,
                                             bool checkUb2L1Constraints);
    bool CanUseMiddleTensorForUpgrade(const LogicalTensorPtr& middle, MemoryType targetType) const;
    bool HasOnlyContractProducers(const LogicalTensorPtr& tensor) const;
    bool HasOnlySliceConsumers(const LogicalTensorPtr& tensor) const;
    bool IsSliceOutputTarget(Operation& sliceOp, MemoryType targetType) const;
    Status EnsureSliceOutputTarget(Operation& sliceOp, MemoryType targetType, const std::string& reason);
    Status ApplySliceContractUpgrade(Operation& sliceOp, MemoryType sourceType, MemoryType targetType,
                                     const std::string& reason);
    Status ApplySingleContractSliceUpgrade(Operation& contractOp, MemoryType sourceType, MemoryType targetType,
                                           const std::string& reason);
    bool CanUseL0C2L1UpgradePath(Operation& operation);
    size_t CalcNZTensorSize(const LogicalTensorPtr& tensor) const;
    size_t CalcNzStorageSize(const LogicalTensorPtr& tensor) const;
    size_t CalcNzAlignedSize(const LogicalTensorPtr& tensor) const;
    Status CalcLineOffset(const Shape& shape, const Offset& offset, int64_t& lineOffset) const;
    Status RunOnFunctionLegacy(Function& function);
    ConvertInserter inserter;
    AssignMemoryTypeChecker checker;
};
static constexpr double UB_THRESHOLD_ASSEMBLE = 0.35;
static constexpr double UB_THRESHOLD_NORMAL = 1.0;
static constexpr double L0C_THRESHOLD = 1.0;
static constexpr double L1_THRESHOLD = 0.5;
static constexpr uint16_t L0C_TILE_SIZE = 16;
static constexpr uint16_t INT8_ALIGN_SIZE = 32;
} // namespace npu::tile_fwk

#endif // TILE_FWK_ASSIGN_MEMORY_TYPE_H
