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
 * \file topo_program.cpp
 * \brief
 */

#include "topo_program.h"

namespace npu {
namespace tile_fwk {
bool NeedInferShape(const Operation *op) {
    if (op->GetOOperands().empty()) {
        return false;
    }
    if (!(op->GetOOperands()[0]->GetDynValidShape().empty()) && op->GetOpcode() != Opcode::OP_ASSEMBLE) {
        return false;
    }
    return true;
}

void TopoProgramUtils::TopoProgram(const std::vector<Operation*>& opList,
                                      const std::vector<std::vector<size_t>>& opInGraph,
                                      const std::vector<std::vector<size_t>>& opOutGraph,
                                      bool isParamIndex){
    std::queue<size_t> procOpQueue;
    std::vector<size_t> inDegree(opList.size(), 0);
    for (size_t j = 0; j < opInGraph.size(); ++j) {
        if (opInGraph[j].empty()) {
            procOpQueue.push(j);
        }
        inDegree[j] = opInGraph[j].size();
    }
    while (!procOpQueue.empty()) {
        auto opIdx = procOpQueue.front();
        procOpQueue.pop();
        for (auto outIdx : opOutGraph[opIdx]) {
            inDegree[outIdx]--;
            if (inDegree[outIdx] == 0) {
                procOpQueue.push(outIdx);
            }
        }
        if (isParamIndex) {
            if (NeedInferShape(opList[opIdx])) {
                InferShapeRegistry::GetInstance().CallInferShapeFunc(opList[opIdx]);
            }
            continue;
        }
        InferShapeRegistry::GetInstance().CallInferShapeFunc(opList[opIdx]);
    }
}
}
}