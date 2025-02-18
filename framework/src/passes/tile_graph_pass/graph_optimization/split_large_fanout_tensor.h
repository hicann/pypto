/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file split_large_fanout_tensor.h
 * \brief
 */

#ifndef PASS_SPLIT_LARGE_FANOUT_TENSOR_H_
#define PASS_SPLIT_LARGE_FANOUT_TENSOR_H_

#include "interface/function/function.h"
#include "interface/tensor/logical_tensor.h"
#include "passes/pass_utils/pass_common_defs.h"
#include "passes/pass_utils/dead_operation_eliminate.h"
#include "passes/pass_interface/pass.h"
#include "tilefwk/data_type.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "passes/pass_utils/pass_utils.h"

namespace npu::tile_fwk {
/*
 * SplitLargeFanoutTensor:
 本pass主要用于处理assemble被多个消费者消费时，如果每人都消费的是部分数据，因为assemble导致需要等待全部数据产生，影响并行度的问题
 * 处理原则：
    1. 仅解决OP_ASSEMBLE作为copyOut后接OP_VIEW作为CopyIn的场景
    2. 比较assemble输入与view输出之前的重合关系
    2.1 完全匹配：一到一情况
    2.2 被覆盖：多到一情况
    2.3 覆盖所有：一到多情况
*/
class SplitLargeFanoutTensor : public Pass {
public:
    SplitLargeFanoutTensor() : Pass("SplitLargeFanoutTensor") {}
    ~SplitLargeFanoutTensor() override = default;

private:
    Status RunOnFunction(Function &function) override;
    void EraseRedundantAssembleOp(Function &function);
    void EraseRedundantViewOp(Function &function);
    void RemoveOps(Function &function, std::vector<Operation *> &opList) const;
    void UpdateForRedundantAssemble(Operation &op);
    void UpdateForRedundantView(Operation &op, Operation &consumer);
    int64_t GCD(int64_t x, int64_t y);
    Status LCM(int64_t x, int64_t y, int64_t &lcm);
    Status CalLcmShape(const Shape &toShape, const Shape &fromShape, Shape &lcmShape);
    Status CalGcdShape(const Shape &toShape, const Shape &fromShape, Shape &lcmShape);
    void GenerateOffset(const Shape &maxs, const Shape &steps, 
        Shape &current, std::vector<Shape> &result, size_t dim);
    void CollectLargeTensorToInfo(const LogicalTensorPtr &largeTensor);
    void CollectLargeTensorFromInfo(const LogicalTensorPtr &largeTensor);
    void CollectOverlaps(Function &function, LogicalTensorPtr largeTensor, Shape lcmTileShape, Offset lcmTileOffset,
        std::vector<std::pair<LogicalTensorPtr, Offset>> toTensorInfos,
        std::vector<std::pair<LogicalTensorPtr, Offset>> fromTensorInfos,
        LogicalTensors &overlaps, LogicalTensors &dualOverlaps);
    void CreateOpFor1toM(Function &function, LogicalTensorPtr largeTensor, Shape lcmTileShape, Offset lcmTileOffset,
        LogicalTensors overlaps, LogicalTensors dualOverlaps);
    void CreateOpForMtoM(Function &function, LogicalTensorPtr largeTensor, Shape lcmTileShape, Offset lcmTileOffset,
        LogicalTensors overlaps, LogicalTensors dualOverlaps);
    void MoreSplit(Function &function, LogicalTensorPtr largeTensor, LogicalTensors overlaps, LogicalTensors dualOverlaps);
    void CreateOpForMoreSplit(Function &function, LogicalTensorPtr largeTensor, LogicalTensors overlaps,
        Shape gcdShape, LogicalTensorPtr dualOverlap, std::vector<Shape> gcdTileOffsets, Offset viewOpOffset);
    void CollectLargeTensor(Function &function);
    void SplitLargeTensor(Function &function);
    bool IsBeCovered(Function &function, LogicalTensorPtr largeTensor,
        std::vector<std::pair<LogicalTensorPtr, Offset>> toTensorInfos);
    bool HasDuplicateToTile(std::vector<std::pair<LogicalTensorPtr, Offset>> toTensorInfos);
    void TryToSplitLargeTensor(Function &function, const Shape &lcmShape, const LogicalTensorPtr &largeTensor);
    std::unordered_map<int, std::vector<std::pair<LogicalTensorPtr, Offset>>> toInfoMap;
    std::unordered_map<int, std::vector<std::pair<LogicalTensorPtr, Offset>>> fromInfoMap;
    std::unordered_set<LogicalTensorPtr> largeTensors;
    std::map<LogicalTensorPtr, std::set<Shape>> toShapes;
    std::map<LogicalTensorPtr, std::set<Shape>> fromShapes;
    bool enableMoreSplit = false;
};

struct ShapeComparator {
    bool operator()(const Shape &a, const Shape &b) const { return CommonUtils::Numel(a) < CommonUtils::Numel(b); }
};
} // namespace npu::tile_fwk
#endif // PASS_SPLIT_LARGE_FANOUT_TENSOR_H_