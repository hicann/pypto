/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file infer_shape_utils.cpp
 * \brief 公共的 InferShape 方法实现
 */

#include <unordered_set>
#include "infer_shape_utils.h"
#include "passes/pass_utils/topo_program.h"

namespace npu {
namespace tile_fwk {
Status InferShapeUtils::InferShape(Function& function, const std::vector<Operation*>& targetOps)
{
    std::vector<Operation*> opList;
    std::unordered_set<Operation*> targetOpSet;

    if (targetOps.empty()) {
        opList = function.Operations().DuplicatedOpList();
    } else {
        // 去重并保留顺序，同时构建 targetOpSet
        opList.reserve(targetOps.size());
        for (const auto op : targetOps) {
            if (targetOpSet.insert(op).second) {
                opList.push_back(op);
            }
        }
    }

    // 构建 opMagic -> index 映射
    std::map<int, size_t> opMagic2Idx;
    for (size_t i = 0; i < opList.size(); ++i) {
        opMagic2Idx[opList[i]->GetOpMagic()] = i;
    }

    std::vector<std::vector<size_t>> opOutGraph(opList.size());
    std::vector<std::vector<size_t>> opInGraph(opList.size());
    for (size_t opIdx = 0; opIdx < opList.size(); ++opIdx) {
        const auto& op = opList[opIdx];
        size_t currentIdx = opMagic2Idx[op->GetOpMagic()];

        for (const auto producer : op->ProducerOpsOrdered()) {
            if (targetOpSet.empty() || targetOpSet.find(producer) != targetOpSet.end()) {
                opInGraph[currentIdx].push_back(opMagic2Idx[producer->GetOpMagic()]);
            }
        }
        for (const auto consumer : op->ConsumerOpsOrdered()) {
            if (targetOpSet.empty() || targetOpSet.find(consumer) != targetOpSet.end()) {
                opOutGraph[currentIdx].push_back(opMagic2Idx[consumer->GetOpMagic()]);
            }
        }
    }

    TopoProgramUtils::TopoProgram(opList, opInGraph, opOutGraph);
    return SUCCESS;
}
} // namespace tile_fwk
} // namespace npu
