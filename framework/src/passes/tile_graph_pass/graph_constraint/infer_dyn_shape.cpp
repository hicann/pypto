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
 * \file infer_dyn_shape.cpp
 * \brief
 */

#include <queue>
#include "interface/function/function.h"
#include "infer_dyn_shape.h"
#include "passes/pass_check/infer_dyn_shape_checker.h"
#include "passes/pass_log/pass_log.h"

#define MODULE_NAME "InferDynShape"

namespace npu {
namespace tile_fwk {
Status InferDynShape::PostCheck(Function &function) {
    InferDynShapeChecker checker;
    return checker.DoPostCheck(function);
}

Status InferDynShape::InferShape(Function& function){
    size_t i = 0U;
    std::map<int, size_t> opMagic2Idx;
    std::vector<Operation*> opList = function.Operations().DuplicatedOpList();
    for (const auto op : opList) {
        opMagic2Idx[op->GetOpMagic()] = i;
        i++;
    }
    std::vector<std::vector<size_t>> opInGraph(opList.size());
    std::vector<std::vector<size_t>> opOutGraph(opList.size());
    for (size_t opIdx = 0; opIdx < opList.size(); opIdx++) {
        const auto& op = opList[opIdx];
        for (const auto producer : op->ProducerOpsOrdered()) {
            opInGraph[opMagic2Idx[op->GetOpMagic()]].push_back(opMagic2Idx[producer->GetOpMagic()]);
        }
        for (const auto consumer : op->ConsumerOpsOrdered()) {
            opOutGraph[opMagic2Idx[op->GetOpMagic()]].push_back(opMagic2Idx[consumer->GetOpMagic()]);
        }
    }
    bool isInferIndex = false;
    TopoProgramUtils::TopoProgram(opList, opInGraph, opOutGraph, isInferIndex);
    return SUCCESS;
}

Status InferDynShape::RunOnFunction(Function &function)
{
    // 遍历每一个op，调用对应的infershape函数
    // 遍历顺序，按照入度解依赖
    APASS_LOG_INFO_F(Elements::Function, "===> Start InferDynShape.");
    if (InferShape(function) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Function, "InferShape failed; Please check the InferShape method.");
        return FAILED;
    }
    APASS_LOG_DEBUG_F(Elements::Function, "Dump: %s", function.Dump().c_str());
    APASS_LOG_INFO_F(Elements::Function, "===> End InferDynShape.");
    return SUCCESS;
}
} 
} // namespace npu::tile_fwk