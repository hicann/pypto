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
 * \file pre_graph.cpp
 * \brief
 */

#include "pre_graph.h"
#include "passes/pass_check/pre_graph_checker.h"
#include "passes/pass_log/pass_log.h"

#define MODULE_NAME "PreGraphProcess"

namespace npu::tile_fwk {
void PreGraphProcess::UpdateCopyOpIsCube(Operation &op) const {
    /*
    后续考虑移到InsertCopyOp
    copy_out for producer
    op(copy_in) --> input --> consumerOp(isCube?)
    */
    if (IsCopyIn(op.GetOpcode())) {
        for (const auto &consumerOps : op.ConsumerOps()) {
            if ((consumerOps->HasAttr(OpAttributeKey::isCube)) &&
                (consumerOps->GetSubgraphID() == op.GetSubgraphID())) {
                op.SetAttribute(OpAttributeKey::isCube, consumerOps->GetBoolAttribute(OpAttributeKey::isCube));
                break;
            }
        }
    }
    /*
    copy_in for consumer
    producerOp(isCube?) --> input --> op(copy_out)
    */
    if (IsCopyOut(op.GetOpcode())) {
        for (const auto &producerOps : op.ProducerOps()) {
            if ((producerOps->HasAttr(OpAttributeKey::isCube)) &&
                (producerOps->GetSubgraphID() == op.GetSubgraphID())) {
                op.SetAttribute(OpAttributeKey::isCube, producerOps->GetBoolAttribute(OpAttributeKey::isCube));
                break;
            }
        }
    }
}

Status PreGraphProcess::RunOnFunction(Function &function) {
    APASS_LOG_INFO_F(Elements::Operation, "===> start PreGraph.");
    ColorGraph colorGraph;
    colorGraph.PreColorSort(function);
    auto opList = function.Operations();
    for (auto &op : opList) {
        colorGraph.InitializeTensorColor(op);
        UpdateCopyOpIsCube(op);
    }
    SetBoundary setBoundary;
    setBoundary.SetTensorBoundary(function);
    // Processing Special Ops
    SetCopyAttr setCopyAttr;
    for (auto &op : opList) {
        if (IsCopyOut(op.GetOpcode()) && op.GetOpcode() != Opcode::OP_COPY_OUT) {
            setCopyAttr.ProcessSpecialMTEOperation(op);
        }
        if (IsCopyIn(op.GetOpcode()) && op.GetOpcode() != Opcode::OP_COPY_IN && op.GetOpcode() != Opcode::OP_SHMEM_GET_GM2UB) {
            setCopyAttr.ProcessMoveInOperation(op);
        }
    }
    RemoveRedundantAssemble removeRedundantAssemble;
    removeRedundantAssemble.DeleteRedundantAssemble(function);
    CubeProcess cubeProcess;
    if (cubeProcess.UpdateCubeOp(function) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Function, "Update Cube attr failed.");
        return FAILED;
    }
    APASS_LOG_INFO_F(Elements::Operation, "===> End PreGraph.");
    return SUCCESS;
}

Status PreGraphProcess::PreCheck(Function &function) {
    PreGraphProcessChecker checker;
    return checker.DoPreCheck(function);
}

Status PreGraphProcess::PostCheck(Function &function) {
    PreGraphProcessChecker checker;
    return checker.DoPostCheck(function);
}
} // namespace npu::tile_fwk