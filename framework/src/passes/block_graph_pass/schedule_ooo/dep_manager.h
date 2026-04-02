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
 * \file dep_manager.h
 * \brief Dependency manager for operation scheduling
 */

#ifndef PASS_DEP_MANAGER_H_
#define PASS_DEP_MANAGER_H_

#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <functional>
#include <string>
#include "interface/operation/operation.h"
#include "interface/utils/common.h"

namespace npu::tile_fwk {

inline bool IsViewOp(const Operation &op) {
    const auto opc = op.GetOpcode();
    return opc == Opcode::OP_VIEW || opc == Opcode::OP_VIEW_TYPE;
}

class DependencyManager {
public:
    void RegisterOp(Operation *op);

    void ClearDependencies();

    static bool IsOpAlloc(Operation *op);

    void AddDependency(Operation *preOp, Operation *postOp);

    void AddAllocDependency(Operation *preOp, Operation *postOp);

    Status InitAllocDependencies(Operation *op, std::unordered_map<int, Operation *> &tensor2AllocOpMap);

    bool RemoveDependency(Operation *preOp, Operation *postOp);

    int InsertSuccessor(Operation *op, Operation *succ);
    int RemoveSuccessor(Operation *op, Operation *succ);
    int InsertPredecessor(Operation *op, Operation *pred);
    int RemovePredecessor(Operation *op, Operation *pred);

    std::unordered_set<Operation *> &GetSuccessors(Operation *op);
    std::unordered_set<Operation *> &GetPredecessors(Operation *op);
    bool HasOp(Operation *op) const;

    std::string PrintOp(Operation *op);

    Operation *SkipViewChain(Operation *start, bool followProducers);

    void FindDependencies(Operation *op, bool needView);
    void InitOpConsumerAndProducer(const std::vector<Operation *> &ops);

    Status InitDependencies(const std::vector<Operation *> &ops, bool needView);

    void PrintDependencies(const std::vector<Operation *> &ops);

private:
    void Clear();

    void HandleScaleOpDependency(Operation *op, MemoryType memType);
    void AddProducerDependencies(Operation *op);
    void AddConsumerDependencies(Operation *op);

    std::unordered_map<Operation *, std::unordered_set<Operation *>> opConsumers;
    std::unordered_map<Operation *, std::unordered_set<Operation *>> opProducers;
    std::unordered_map<Operation *, std::unordered_set<Operation *>> inGraph_;
    std::unordered_map<Operation *, std::unordered_set<Operation *>> outGraph_;
};

} // namespace npu::tile_fwk

#endif