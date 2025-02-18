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
 * \file reschedule_utils.h
 * \brief
 */

#ifndef RESCHEDULE_UTILS_H_
#define RESCHEDULE_UTILS_H_
#include "interface/function/function.h"
#include "interface/tensor/logical_tensor.h"

namespace npu::tile_fwk {
class RescheduleUtils {
public:
    static bool isAllocOp(Operation* op);

    // vector size == 2
    // 对于ingraph outgraph color colornodes等信息可以之后用一个单例管理起来，避免每个pass遍历图去重新收集这些信息，导致编译时间变长
    static std::vector<std::vector<std::vector<int>>> GetInOutGraphs(
        const std::vector<Operation *> &opList, int functionmagic = -1);

    // 按序列获取OP之间的依赖图连边。返回的边，以源op和目标op在opList中的Index表示。
    // 读写关系通过tensor magic来判断。
    static void GetInOutGraphs(const std::vector<Operation *> &opList,
        std::vector<std::vector<int>>& inGraph, std::vector<std::vector<int>>& outGraph, bool hasCtrlEdge=true);

    static void GetInOutGraphs(const std::vector<Operation *> &opList,
        std::unordered_map<Operation*, std::vector<Operation*>>& inGraph,
        std::unordered_map<Operation*, std::vector<Operation*>>& outGraph, bool hasCtrlEdge=true);

    // 按序列获取OP之间的依赖图连边。返回的边，以源op和目标op在opList中的Index表示。
    // 读写关系通过每个tensor的memmap[subGraphId].memId来判断。
    static void GetInOutGraphs(const std::vector<Operation *> &opList,
        std::vector<std::vector<int>>& inGraph, std::vector<std::vector<int>>& outGraph,
        int subGraphId, bool hasCtrlEdge=true);

    static void GetInOutGraphs(const std::vector<Operation *> &opList,
        std::unordered_map<Operation*, std::vector<Operation*>>& inGraph,
        std::unordered_map<Operation*, std::vector<Operation*>>& outGraph,
        int subGraphId, bool hasCtrlEdge=true);

    static PipeType GetOpPipeType(const Operation* op);

    // 计算op的offset无关hash（op name + params + input shape/dtype + output shape/dtype），用于Cycle数校准。
    static unsigned long ComputeOperationHash(const Operation *op);

    static void UpdateTensorConsProd(Function *funcPtr);
    static void ClearInputConsProd(Operation &op, Function *funcPtr, std::unordered_set<LogicalTensorPtr> incastSet);
    static void ClearOutputConsProd(Operation &op, Function *funcPtr, std::unordered_set<LogicalTensorPtr> outcastSet);
    static void FindOtherGraphOp(Operation *op, Function *funcPtr, std::unordered_set<Operation*> &otherGraphOp);
    static void PrintColorNode(Function &func);
};
}  // namespace npu::tile_fwk
#endif