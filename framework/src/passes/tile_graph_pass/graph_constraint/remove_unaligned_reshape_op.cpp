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
 * \file remove_unaligned_reshape_op.cpp
 * \brief
 */

#include "remove_unaligned_reshape_op.h"
#include "passes/pass_utils/graph_utils.h"
#include "passes/pass_log/pass_log.h"

#define MODULE_NAME "RemoveUnalignedReshape"

namespace npu::tile_fwk {
/*
before:
    add->reshape(padded)->mul

after:
    add->copyout->reshape->copyin->mul
*/
Status RemoveUnalignedReshape::RunOnFunction(Function &function) {
    APASS_LOG_INFO_F(Elements::Function, "===> Start RemoveUnalignedReshape.");
    ReplaceDynUnalignedReshapeOps(function);
    CollectReshapeOps(function);
    for (auto &a : copyOuts) {
        GraphUtils::CopyDynStatus(a.output, a.input);
        auto &newCopyOut = function.AddRawOperation(Opcode::OP_COPY_OUT, {a.input}, {a.output});
        newCopyOut.SetOpAttribute(std::make_shared<CopyOpAttribute>(a.from, OpImmediate::Specified(a.toOffset),
            OpImmediate::Specified(newCopyOut.iOperand.front()->oriShape),
            OpImmediate::Specified(newCopyOut.oOperand.front()->tensor->GetDynRawShape())));
        auto producerOp = *(a.input->GetProducers().begin());
        newCopyOut.UpdateSubgraphID(producerOp->GetSubgraphID());
        APASS_LOG_INFO_F(Elements::Operation, "ADD OP_COPY_OUT, magic %d ,IOperand tensor magic %d OOperand tensor magic %d.",
            newCopyOut.opmagic, a.input->magic, a.output->magic);
    }
    for (auto &b : copyIns) {
        GraphUtils::CopyDynStatus(b.input, b.output);
        auto &newCopyIn = function.AddRawOperation(Opcode::OP_COPY_IN, {b.input}, {b.output});
        newCopyIn.SetOpAttribute(std::make_shared<CopyOpAttribute>(OpImmediate::Specified(b.fromOffset), b.to,
            OpImmediate::Specified(newCopyIn.oOperand.front()->oriShape),
            OpImmediate::Specified(newCopyIn.iOperand.front()->tensor->GetDynRawShape()),
            OpImmediate::Specified(newCopyIn.iOperand.front()->GetDynValidShape())));
        auto consumerOp = *(b.output->GetConsumers().begin());
        newCopyIn.UpdateSubgraphID(consumerOp->GetSubgraphID());
        APASS_LOG_INFO_F(Elements::Operation, "ADD OP_VIEW, magic %d ,IOperand tensor magic %d OOperand tensor magic %d.",
            newCopyIn.opmagic, b.input->magic, b.output->magic);
    }
    APASS_LOG_INFO_F(Elements::Function, "===> End RemoveUnalignedReshape.");
    return SUCCESS;
}

LogicalTensorPtr RemoveUnalignedReshape::InsertIOTensor(Function &function, Operation &op, std::unordered_map<OverlaprawMagic, std::shared_ptr<RawTensor>> &rawIO, LogicalTensorPtr &ioTensor) {
    if (rawIO.count(ioTensor->tensor->rawmagic) == 0) {
        auto reshapeRawTensor = std::make_shared<RawTensor>(ioTensor->Datatype(),
            ioTensor->tensor->oriRawshape, ioTensor->Format());
        reshapeRawTensor->oriRawshape = reshapeRawTensor->rawshape;
        rawIO.insert({ioTensor->tensor->rawmagic, reshapeRawTensor});
    }
    auto newReshapeIO = std::make_shared<LogicalTensor>(
        function, rawIO[ioTensor->tensor->rawmagic], ioTensor->offset, ioTensor->oriShape);
    newReshapeIO->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    newReshapeIO->subGraphID = op.GetSubgraphID();
    newReshapeIO->isSubGraphBoundary = true;
    function.GetTensorMap().Insert(newReshapeIO);
    return newReshapeIO;
}

bool RemoveUnalignedReshape::CheckUnaligned(Operation &op) {
    int lastIdx;
    for (const auto &input : op.GetIOperands()) {
        if (input != nullptr && input->tensor != nullptr) {
            lastIdx = input->shape.size() - 1;
            if (input->shape.size() == input->tensor->oriRawshape.size() &&
                input->shape.size() == input->tensor->rawshape.size() &&
                input->tensor->oriRawshape[lastIdx] != input->tensor->rawshape[lastIdx]) {
                return true;
            }
        }
    }
    for (const auto &output : op.GetOOperands()) {
        if (output != nullptr && output->tensor != nullptr) {
            lastIdx = output->shape.size() - 1;
            if (output->shape.size() == output->tensor->oriRawshape.size() &&
                output->shape.size() == output->tensor->rawshape.size() &&
                output->tensor->oriRawshape[lastIdx] != output->tensor->rawshape[lastIdx]) {
                return true;
            }
        }
    }
    return false;
}

std::vector<int64_t> FindChangedDims(const std::vector<int64_t>& inputShapes, const std::vector<int64_t>& outputShapes) {
    int inputDimSize = inputShapes.size();
    int outputDimSize = outputShapes.size();

    int left = -1;
    int right = -1;
    std::vector<int64_t> changedInputAxes = {};

    for (int i = 0; i < std::min(inputDimSize, outputDimSize); ++i) {
        if (inputShapes[i] != outputShapes[i] && left == -1) {
            left = i;  // left第一次shape不等的位置
        }

        if (inputShapes[inputDimSize - 1 - i] != outputShapes[outputDimSize - 1 - i] && right == -1) {
            right = inputDimSize - 1 - i;  // right第一次shape不等的位置
        }
    }

    if (left <= right && left != -1 && right != -1) {
        for (int i = left; i <= right; ++i) {
            changedInputAxes.push_back(i);
        }
    }

    return changedInputAxes;
}

void RemoveUnalignedReshape::ReplaceDynUnalignedReshapeOps(Function &function) {
    APASS_LOG_INFO_F(Elements::Function, "===> Start ReplaceDynUnalignedReshapeOps.");
    // 寻找到无法处理的reshape op
    for (auto &op : function.Operations()) {
        if (op.GetOpcode() != Opcode::OP_RESHAPE){
            continue;
        }

        auto input = op.GetIOperands().front();
        auto output = op.GetOOperands().front();
        // only support ub reshape yet
        if ((input->GetMemoryTypeOriginal() != MemoryType::MEM_UB) || (output->GetMemoryTypeOriginal() != MemoryType::MEM_UB)) {
            continue;
        }

        auto inputShapes = input->shape;
        auto outputShapes = output->shape;
        auto changedDims = FindChangedDims(outputShapes, inputShapes);

        auto inDynValidShape = input->GetDynValidShape();
        auto outDynValidShape = output->GetDynValidShape();

        for (const auto &dim : changedDims) {
            if (dim + 1 > (int)outDynValidShape.size()) {
                APASS_LOG_WARN_F(Elements::Operation, "The dynValidShape of output[%d] of op[%d] has no [%d] index.",
                output->GetMagic(), op.GetOpMagic(), dim);
                break;
            } else if (!outDynValidShape[dim].IsImmediate()) {
                op.SetAsDeleted();
                auto tmpWorkSpaceIn = std::make_shared<LogicalTensor>(function, input->Datatype(), input->oriShape, input->Format());
                auto tmpWorkSpaceOut = std::make_shared<LogicalTensor>(function, input->Datatype(), output->oriShape, output->Format());

                tmpWorkSpaceIn->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
                tmpWorkSpaceIn->UpdateDynValidShape(inDynValidShape);
                tmpWorkSpaceOut->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
                tmpWorkSpaceOut->UpdateDynValidShape(outDynValidShape);

                auto &reshapeCopyOutOp = function.AddOperation(Opcode::OP_RESHAPE_COPY_OUT, {input}, {tmpWorkSpaceIn});
                auto &reshapeOp = function.AddOperation(Opcode::OP_RESHAPE, {tmpWorkSpaceIn}, {tmpWorkSpaceOut});
                auto &reshapeCopyInOp = function.AddOperation(Opcode::OP_RESHAPE_COPY_IN, {tmpWorkSpaceOut}, {output});

                reshapeCopyOutOp.UpdateSubgraphID(op.GetSubgraphID());
                reshapeCopyInOp.UpdateSubgraphID(op.GetSubgraphID());
                reshapeOp.UpdateSubgraphID(op.GetSubgraphID());

                reshapeCopyOutOp.SetOpAttribute(std::make_shared<CopyOpAttribute>(
                    MemoryType::MEM_UB, OpImmediate::Specified(std::vector<SymbolicScalar>(input->shape.size(), 0)), OpImmediate::Specified(input->shape),
                    OpImmediate::Specified(input->tensor->GetDynRawShape()), OpImmediate::Specified(input->GetDynValidShape())
                ));

                reshapeCopyInOp.SetOpAttribute(std::make_shared<CopyOpAttribute>(
                    OpImmediate::Specified(std::vector<SymbolicScalar>(output->shape.size(), 0)), MemoryType::MEM_DEVICE_DDR, OpImmediate::Specified(output->shape),
                    OpImmediate::Specified(output->tensor->GetDynRawShape()), OpImmediate::Specified(output->GetDynValidShape())
                ));

                APASS_LOG_INFO_F(Elements::Operation,"Reshape op %d is replaceed by reshapeCopyOutOp %d and reshapeCopyInOp %d.", 
                    op.opmagic, reshapeCopyOutOp.opmagic, reshapeCopyInOp.opmagic);
                break;
            }
        }
    }

    function.EraseOperations(true, false);

    APASS_LOG_INFO_F(Elements::Function, "===> End ReplaceDynUnalignedReshapeOps.");
}

void RemoveUnalignedReshape::CollectReshapeOps(Function &function) {
    for (auto &op : function.Operations()) {
        if (op.GetOpcode() != Opcode::OP_RESHAPE) {
            continue;
        }

        if (!CheckUnaligned(op)) {
            continue;
        }

        auto input = op.GetIOperands().front();
        auto output = op.GetOOperands().front();
        if ((input->GetMemoryTypeOriginal() != MemoryType::MEM_UB) || (output->GetMemoryTypeOriginal() != MemoryType::MEM_UB)) {
            continue;
        }

        // 插入copyout
        LogicalTensorPtr newReshapeInput = InsertIOTensor(function, op, reshapeRawInputs, input);
        copyOuts.emplace_back(CopyOutOpMemUnalign{input->GetMemoryTypeOriginal(), input->offset, input, newReshapeInput});
        op.ReplaceInput(newReshapeInput, input);

        // 插入copyin
        LogicalTensorPtr newReshapeOutput = InsertIOTensor(function, op, reshapeRawOutputs, output);
        copyIns.emplace_back(
            CopyInOpMemUnalign{output->GetMemoryTypeOriginal(), output->offset, newReshapeOutput, output});
        op.ReplaceOutput(newReshapeOutput, output);
        output->tensor->actualRawmagic = -1;
        op.GetOOperands().front()->tensor->actualRawmagic = op.GetIOperands().front()->tensor->GetRawMagic();
    }
}

} // namespace
