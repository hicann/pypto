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
#include "passes/pass_utils/dead_operation_eliminate.h"
#include "passes/pass_log/pass_log.h"

#define MODULE_NAME "RemoveUnalignedReshape"

namespace npu::tile_fwk {
/*
before:
    add->reshape(padded)->mul

after:
    add->copyout->reshape->copyin->mul
*/
Status RemoveUnalignedReshape::RunOnFunction(Function& function)
{
    APASS_LOG_INFO_F(Elements::Function, "===> Start RemoveUnalignedReshape.");
    ReplaceDynUnalignedReshapeOps(function);
    CollectReshapeOps(function);
    for (auto& a : copyOuts) {
        GraphUtils::CopyDynStatus(a.output, a.input);
        auto& newCopyOut = function.AddRawOperation(Opcode::OP_COPY_OUT, {a.input}, {a.output});
        newCopyOut.SetOpAttribute(std::make_shared<CopyOpAttribute>(
            a.from, OpImmediate::Specified(a.toOffset), OpImmediate::Specified(newCopyOut.iOperand.front()->oriShape),
            OpImmediate::Specified(newCopyOut.oOperand.front()->tensor->GetDynRawShape())));
        auto producerOp = *(a.input->GetProducers().begin());
        newCopyOut.UpdateSubgraphID(producerOp->GetSubgraphID());
        APASS_LOG_INFO_F(
            Elements::Operation, "ADD OP_COPY_OUT, magic %d ,IOperand tensor magic %d OOperand tensor magic %d.",
            newCopyOut.opmagic, a.input->magic, a.output->magic);
    }
    for (auto& b : copyIns) {
        GraphUtils::CopyDynStatus(b.input, b.output);
        auto& newCopyIn = function.AddRawOperation(Opcode::OP_COPY_IN, {b.input}, {b.output});
        newCopyIn.SetOpAttribute(std::make_shared<CopyOpAttribute>(
            OpImmediate::Specified(b.fromOffset), b.to, OpImmediate::Specified(newCopyIn.oOperand.front()->oriShape),
            OpImmediate::Specified(newCopyIn.iOperand.front()->tensor->GetDynRawShape()),
            OpImmediate::Specified(newCopyIn.iOperand.front()->GetDynValidShape())));
        auto consumerOp = *(b.output->GetConsumers().begin());
        newCopyIn.UpdateSubgraphID(consumerOp->GetSubgraphID());
        APASS_LOG_INFO_F(
            Elements::Operation, "ADD OP_VIEW, magic %d ,IOperand tensor magic %d OOperand tensor magic %d.",
            newCopyIn.opmagic, b.input->magic, b.output->magic);
    }
    APASS_LOG_INFO_F(Elements::Function, "===> End RemoveUnalignedReshape.");
    return SUCCESS;
}

LogicalTensorPtr RemoveUnalignedReshape::InsertIOTensor(
    Function& function, Operation& op, std::unordered_map<OverlaprawMagic, std::shared_ptr<RawTensor>>& rawIO,
    LogicalTensorPtr& ioTensor)
{
    if (rawIO.count(ioTensor->tensor->rawmagic) == 0) {
        auto reshapeRawTensor =
            std::make_shared<RawTensor>(ioTensor->Datatype(), ioTensor->tensor->oriRawshape, ioTensor->Format());
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

bool RemoveUnalignedReshape::CheckUnaligned(Operation& op)
{
    int lastIdx;
    for (const auto& input : op.GetIOperands()) {
        if (input != nullptr && input->tensor != nullptr) {
            lastIdx = input->shape.size() - 1;
            if (input->shape.size() == input->tensor->oriRawshape.size() &&
                input->shape.size() == input->tensor->rawshape.size() &&
                input->tensor->oriRawshape[lastIdx] != input->tensor->rawshape[lastIdx]) {
                return true;
            }
        }
    }
    for (const auto& output : op.GetOOperands()) {
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

std::vector<int64_t> FindChangedDims(const std::vector<int64_t>& inputShapes, const std::vector<int64_t>& outputShapes)
{
    int inputDimSize = inputShapes.size();
    int outputDimSize = outputShapes.size();

    int left = -1;
    int right = -1;
    std::vector<int64_t> changedInputAxes = {};

    for (int i = 0; i < std::min(inputDimSize, outputDimSize); ++i) {
        if (inputShapes[i] != outputShapes[i] && left == -1) {
            left = i; // left第一次shape不等的位置
        }

        if (inputShapes[inputDimSize - 1 - i] != outputShapes[outputDimSize - 1 - i] && right == -1) {
            right = inputDimSize - 1 - i; // right第一次shape不等的位置
        }
    }

    if (left <= right && left != -1 && right != -1) {
        for (int i = left; i <= right; ++i) {
            changedInputAxes.push_back(i);
        }
    }

    return changedInputAxes;
}
void RemoveUnalignedReshape::ReplaceDynUnalignedReshapeOps(Function& function)
{
    APASS_LOG_INFO_F(Elements::Function, "===> Start ReplaceDynUnalignedReshapeOps.");
    auto opList = function.Operations().DuplicatedOpList();
    for (auto& op : opList) {
        if (op->GetOpcode() != Opcode::OP_RESHAPE || processedReshapeOps.count(op->GetOpMagic())) {
            continue;
        }
        auto input = op->GetIOperands().front();
        auto output = op->GetOOperands().front();
        if (input->GetMemoryTypeOriginal() == MemoryType::MEM_UB &&
            output->GetMemoryTypeOriginal() == MemoryType::MEM_UB) {
            ReplaceDynUnalignedReshapeOpsForUB(function, *op);
        } else if (
            input->GetMemoryTypeOriginal() == MemoryType::MEM_DEVICE_DDR &&
            output->GetMemoryTypeOriginal() == MemoryType::MEM_DEVICE_DDR) {
            ReplaceDynUnalignedReshapeOpsForDDR(function, *op);
        }
    }
    APASS_LOG_INFO_F(Elements::Function, "===> End ReplaceDynUnalignedReshapeOps.");
}

void RemoveUnalignedReshape::ReplaceDynUnalignedReshapeOpsForUB(Function& function, Operation& op)
{
    auto input = op.GetIOperands().front();
    auto output = op.GetOOperands().front();

    auto inputShapes = input->shape;
    auto outputShapes = output->shape;
    auto changedDims = FindChangedDims(outputShapes, inputShapes);

    auto inDynValidShape = input->GetDynValidShape();
    auto outDynValidShape = output->GetDynValidShape();

    for (const auto& dim : changedDims) {
        if ((size_t)dim >= outDynValidShape.size()) {
            APASS_LOG_WARN_F(
                Elements::Operation, "The dynValidShape of output[%d] of op[%d] has no [%ld] index.",
                output->GetMagic(), op.GetOpMagic(), static_cast<long>(dim));
            break;
        } else if (!outDynValidShape[dim].IsImmediate()) {
            op.SetAsDeleted();
            auto tmpWorkSpaceIn =
                std::make_shared<LogicalTensor>(function, input->Datatype(), input->oriShape, input->Format());
            auto tmpWorkSpaceOut =
                std::make_shared<LogicalTensor>(function, input->Datatype(), output->oriShape, output->Format());

            tmpWorkSpaceIn->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
            tmpWorkSpaceIn->UpdateDynValidShape(inDynValidShape);
            tmpWorkSpaceOut->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
            tmpWorkSpaceOut->UpdateDynValidShape(outDynValidShape);

            auto& reshapeCopyOutOp = function.AddOperation(Opcode::OP_RESHAPE_COPY_OUT, {input}, {tmpWorkSpaceIn});
            auto& reshapeOp = function.AddOperation(Opcode::OP_RESHAPE, {tmpWorkSpaceIn}, {tmpWorkSpaceOut});
            auto& reshapeCopyInOp = function.AddOperation(Opcode::OP_RESHAPE_COPY_IN, {tmpWorkSpaceOut}, {output});

            reshapeCopyOutOp.UpdateSubgraphID(op.GetSubgraphID());
            reshapeCopyInOp.UpdateSubgraphID(op.GetSubgraphID());
            reshapeOp.UpdateSubgraphID(op.GetSubgraphID());

            reshapeCopyOutOp.SetOpAttribute(std::make_shared<CopyOpAttribute>(
                MemoryType::MEM_UB, OpImmediate::Specified(std::vector<SymbolicScalar>(input->shape.size(), 0)),
                OpImmediate::Specified(input->shape), OpImmediate::Specified(input->tensor->GetDynRawShape()),
                OpImmediate::Specified(input->GetDynValidShape())));

            reshapeCopyInOp.SetOpAttribute(std::make_shared<CopyOpAttribute>(
                OpImmediate::Specified(std::vector<SymbolicScalar>(output->shape.size(), 0)),
                MemoryType::MEM_DEVICE_DDR, OpImmediate::Specified(output->shape),
                OpImmediate::Specified(output->tensor->GetDynRawShape()),
                OpImmediate::Specified(output->GetDynValidShape())));

            APASS_LOG_INFO_F(
                Elements::Operation, "Reshape op %d is replaceed by reshapeCopyOutOp %d and reshapeCopyInOp %d.",
                op.opmagic, reshapeCopyOutOp.opmagic, reshapeCopyInOp.opmagic);
            processedReshapeOps.insert(op.GetOpMagic());
            function.EraseOperations(true, false);
            break;
        }
    }
}

Operation* RemoveUnalignedReshape::CopyBranchBetweenCopyOut2Reshape(Function& function, std::vector<LogicalTensorPtr>& needToCopyTensors, const int& index) {
    bool canToCopy = false;
    Operation* branchOp = nullptr;
    LogicalTensorPtr curTensor = nullptr;
    LogicalTensorPtr preTensor = nullptr;
    LogicalTensorPtr preCloneTensor = nullptr;
    Operation* preOp = nullptr;
    for (int i = needToCopyTensors.size() - 1; i >= 0; i--) {
        auto tensor = needToCopyTensors[i];
        curTensor = tensor->Clone(function, true);
        if (!canToCopy && tensor->GetConsumers().size() > 1) {
            branchOp = *(std::next(tensor->GetConsumers().begin(), index));
            canToCopy = true;
            branchOp->ReplaceInput(curTensor, tensor);
        }
        if (canToCopy) {
            if (preOp != nullptr) {
                auto& newOp = preOp->CloneOperation(function, preOp->GetIOperands(), preOp->GetOOperands());
                newOp.ReplaceInput(curTensor, tensor);
                newOp.ReplaceOutput(preCloneTensor, preTensor);
            }
            if (!tensor->GetProducers().empty()) {
                preOp = *(tensor->GetProducers().begin());
            }
        }
        preTensor = tensor;
        preCloneTensor = curTensor;
    }
    //此时preOp 为 copyOutop preTensor为CopyOut的输出
    Operation* newCopyOutOp = &(preOp->CloneOperation(function, preOp->GetIOperands(), preOp->GetOOperands()));
    newCopyOutOp->ReplaceOutput(preCloneTensor, preTensor);
    DeadOperationEliminator::EliminateDeadOperation(function);
    return newCopyOutOp;
}

void RemoveUnalignedReshape::ReplaceDynUnalignedReshapeOpsForDDR(Function& function, Operation& op)
{
    auto input = op.GetIOperands().front();
    auto output = op.GetOOperands().front();

    auto inDynValidShape = input->GetDynValidShape();
    auto outDynValidShape = output->GetDynValidShape();

    bool hasNonImmediate = false;
    auto changedDims = FindChangedDims(output->shape, input->shape);
    for (const auto& dim : changedDims) {
        if ((size_t)dim >= outDynValidShape.size()) {
            APASS_LOG_WARN_F(
                Elements::Operation, "The dynValidShape of output[%d] of op[%d] has no [%ld] index.",
                output->GetMagic(), op.GetOpMagic(), static_cast<long>(dim));
            break;
        } else if (!outDynValidShape[dim].IsImmediate()) {
            hasNonImmediate = true;
            break;
        }
    }
    if (hasNonImmediate) {
        std::vector<LogicalTensorPtr> needToCopyTensors;
        int index = -1;
        Operation* copyOutOp = FindAllProducerCopyOuts(input, op, needToCopyTensors, index, op.GetOpMagic());
        if (copyOutOp == nullptr) {
            return;
        }
        if (index != -1) {
            copyOutOp = CopyBranchBetweenCopyOut2Reshape(function, needToCopyTensors, index);
        }
        std::vector<Operation*> copyInOps;
        bool hasViewOrAssemble = false;
        FindAllConsumerCopyIns(output, copyInOps, hasViewOrAssemble);
        if (hasViewOrAssemble) {
            APASS_LOG_WARN_F(
                Elements::Operation,
                "Reshape op %d has view or assemble between reshape and copy in, not supported now.", op.GetOpMagic());
            return;
        }
        if (copyInOps.empty()) {
            APASS_LOG_WARN_F(Elements::Operation, "Cannot find copy_in consumers for reshape op %d.", op.GetOpMagic());
            return;
        }

        ProcessCopyOutOfDDRReshape(function, op, copyOutOp);
        ProcessCopyInOfDDRReshape(function, op, copyInOps);
        APASS_LOG_DEBUG_F(Elements::Operation, "Reshape[%d] on GM had processed successfully.", op.GetOpMagic());
        processedReshapeOps.insert(op.GetOpMagic());
    }
}

void RemoveUnalignedReshape::ProcessCopyOutOfDDRReshape(Function& function, Operation& op, Operation* copyOutOp)
{
    // 当copyout的输入是ub输出为ddr可以直接转化为reshapecopyop
    // 否则需要插copy
    auto copyOutInput = copyOutOp->GetIOperands().front();
    auto copyOutInputMemType = copyOutInput->GetMemoryTypeOriginal();
    auto copyOutOutput = copyOutOp->GetOOperands().front();
    auto copyOutOutputMemType = copyOutOutput->GetMemoryTypeOriginal();
    if (copyOutInputMemType == MemoryType::MEM_UB && copyOutOutputMemType == MemoryType::MEM_DEVICE_DDR) {
        copyOutOp->SetOpCode(Opcode::OP_RESHAPE_COPY_OUT);
    } else if (copyOutInputMemType != MemoryType::MEM_UB && copyOutOutputMemType == MemoryType::MEM_DEVICE_DDR) {
        // copyOutInput(NOTUB) -- COPYOUT -- copyOutOutput(DDR) -- reshape
        // copyOutInput(NOTUB) -- COPYOUT -- copyOutOutput(DDR) -- COPYIN -- newTensor(UB) -- RESHAPECOPYOUT --
        // newTensor2(DDR) -- reshape
        LogicalTensor newTensor(function, copyOutOutput->Datatype(), copyOutOutput->GetShape());
        newTensor.SetMemoryTypeBoth(MemoryType::MEM_UB, true);
        auto newTensorPtr = std::make_shared<LogicalTensor>(std::move(newTensor));
        auto& reshapeCopyInOp = function.AddOperation(Opcode::OP_COPY_IN, {copyOutOutput}, {newTensorPtr});
        reshapeCopyInOp.UpdateSubgraphID(op.GetSubgraphID());
        reshapeCopyInOp.SetOpAttribute(std::make_shared<CopyOpAttribute>(
            OpImmediate::Specified(std::vector<SymbolicScalar>(copyOutOutput->GetShape().size(), 0)),
            MemoryType::MEM_UB, OpImmediate::Specified(copyOutOutput->GetShape()),
            OpImmediate::Specified(copyOutOutput->tensor->GetDynRawShape()),
            OpImmediate::Specified(copyOutOutput->GetDynValidShape())));

        LogicalTensor newTensor2(function, copyOutOutput->Datatype(), copyOutOutput->GetShape());
        newTensor2.SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
        auto newTensor2Ptr = std::make_shared<LogicalTensor>(std::move(newTensor2));
        auto& newCopyOutOp = function.AddOperation(Opcode::OP_RESHAPE_COPY_OUT, {newTensorPtr}, {newTensor2Ptr});
        newCopyOutOp.UpdateSubgraphID(op.GetSubgraphID());
        newCopyOutOp.SetOpAttribute(std::make_shared<CopyOpAttribute>(
            MemoryType::MEM_UB,
            OpImmediate::Specified(std::vector<SymbolicScalar>(copyOutOutput->GetShape().size(), 0)),
            OpImmediate::Specified(copyOutOutput->GetShape()),
            OpImmediate::Specified(copyOutOutput->tensor->GetDynRawShape()),
            OpImmediate::Specified(copyOutOutput->GetDynValidShape())));

        copyOutOutput->RemoveConsumer(&op);
        op.ReplaceInput(newTensor2Ptr, copyOutOutput);
    }
}

void RemoveUnalignedReshape::ProcessCopyInOfDDRReshape(
    Function& function, Operation& op, std::vector<Operation*>& copyInOps)
{
    for (auto* copyInOp : copyInOps) {
        auto copyInInput = copyInOp->GetIOperands().front();
        auto copyInInputMemType = copyInInput->GetMemoryTypeOriginal();
        auto copyInOutput = copyInOp->GetOOperands().front();
        auto copyInOutputMemType = copyInOutput->GetMemoryTypeOriginal();

        bool copyInShapeSame = true;
        if (copyInInput->GetShape().size() != copyInOutput->GetShape().size()) {
            copyInShapeSame = false;
        } else {
            for (size_t i = 0; i < copyInInput->GetShape().size(); i++) {
                if (copyInInput->GetShape()[i] != copyInOutput->GetShape()[i]) {
                    copyInShapeSame = false;
                    break;
                }
            }
        }

        if (copyInInputMemType == MemoryType::MEM_DEVICE_DDR) {
            if (copyInOutputMemType == MemoryType::MEM_UB && copyInShapeSame) {
                copyInOp->SetOpCode(Opcode::OP_RESHAPE_COPY_IN);
            } else if (copyInOutputMemType != MemoryType::MEM_UB || !copyInShapeSame) {
                // reshape -- copyInInput(DDR) -- COPYIN -- copyInOutout(NOTUB)
                // reshape -- copyInInput(DDR) -- RESHAPECOPYIN -- newTensor(UB) -- COPYOUT -- newTensor2(DDR) -- COPYIN
                // --copyInOutout(NOTUB)
                LogicalTensor newTensor(function, copyInInput->Datatype(), copyInInput->GetShape());
                newTensor.SetMemoryTypeBoth(MemoryType::MEM_UB, true);
                auto newTensorPtr = std::make_shared<LogicalTensor>(std::move(newTensor));
                auto& reshapeCopyInOp =
                    function.AddOperation(Opcode::OP_RESHAPE_COPY_IN, {copyInInput}, {newTensorPtr});
                reshapeCopyInOp.UpdateSubgraphID(op.GetSubgraphID());
                reshapeCopyInOp.SetOpAttribute(std::make_shared<CopyOpAttribute>(
                    OpImmediate::Specified(std::vector<SymbolicScalar>(copyInInput->GetShape().size(), 0)),
                    MemoryType::MEM_UB, OpImmediate::Specified(copyInInput->GetShape()),
                    OpImmediate::Specified(copyInInput->tensor->GetDynRawShape()),
                    OpImmediate::Specified(copyInInput->GetDynValidShape())));

                LogicalTensor newTensor2(function, copyInInput->Datatype(), copyInInput->GetShape());
                newTensor2.SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
                auto newTensor2Ptr = std::make_shared<LogicalTensor>(std::move(newTensor2));
                auto& newCopyOutOp = function.AddOperation(Opcode::OP_COPY_OUT, {newTensorPtr}, {newTensor2Ptr});
                newCopyOutOp.UpdateSubgraphID(op.GetSubgraphID());
                newCopyOutOp.SetOpAttribute(std::make_shared<CopyOpAttribute>(
                    MemoryType::MEM_UB,
                    OpImmediate::Specified(std::vector<SymbolicScalar>(copyInInput->GetShape().size(), 0)),
                    OpImmediate::Specified(copyInInput->GetShape()),
                    OpImmediate::Specified(copyInInput->tensor->GetDynRawShape()),
                    OpImmediate::Specified(copyInInput->GetDynValidShape())));

                copyInInput->RemoveConsumer(copyInOp);
                copyInOp->ReplaceInput(newTensor2Ptr, copyInInput);
            }
        }
    }
}

/* 从tensor的生产者列表中递归查找所有OP_COPY_OUT：
 * - 如果遇到OP_COPY_OUT，记录并继续查找（可能还有其他生产者）
 * - 如果遇到OP_VIEW、OP_ASSEMBLE或OP_ASSEMBLE_SSA，递归继续向前追溯
 * - 遇到其他op也继续递归追溯
 */
Operation* RemoveUnalignedReshape::FindAllProducerCopyOuts(
    LogicalTensorPtr tensor, Operation& op, std::vector<LogicalTensorPtr>& needToCopyTensors, int& index,
    const int reshapeMagic)
{
    auto producers = tensor->GetProducers();
    auto consumers = tensor->GetConsumers();
    if (producers.size() != 1) {
        if (producers.size() > 1) {
            APASS_LOG_WARN_F(
                Elements::Operation,
                "Reshape[%d] has multiple input ops, cannot convert copy_out to reshape_copy_out; Consider moving "
                "reshape to UB to avoid possible precision issue.", reshapeMagic);
        }
        return nullptr;
    }
    if (index == -1 && consumers.size() > 1) {
        for (size_t i = 0; i < consumers.size(); i++) {
            auto ioConsumer = *(std::next(consumers.begin(), i));
            if (ioConsumer->GetOpMagic() == op.GetOpMagic()) {
                index = i;
            }
        }
    }
    auto producerOp = *(producers.begin());
    auto opcode = producerOp->GetOpcode();
    if (opcode == Opcode::OP_COPY_OUT) {
        needToCopyTensors.push_back(tensor);
        return producerOp;
    }
    if (opcode == Opcode::OP_RESHAPE_COPY_OUT) {
        return nullptr;
    }

    // 其他类型的op（包括view/assemble或其他op），继续向前追溯
    Operation* copyOutOp = nullptr;
    auto inputOperands = producerOp->GetIOperands();
    if (!inputOperands.empty()) {
        copyOutOp = FindAllProducerCopyOuts(inputOperands.front(), *producerOp,
            needToCopyTensors, index, reshapeMagic);
    }
    if (copyOutOp != nullptr && copyOutOp->GetOpcode() == Opcode::OP_COPY_OUT) {
        needToCopyTensors.push_back(tensor);
    }
    return copyOutOp;
}

/* 从tensor的消费者列表中查找OP_COPY_IN，如果遇到OP_VIEW、OP_ASSEMBLE或OP_ASSEMBLE_SSA，
 * 则标记hasViewOrAssemble为true，表示不支持此类场景。
 */
void RemoveUnalignedReshape::FindAllConsumerCopyIns(
    LogicalTensorPtr tensor, std::vector<Operation*>& copyInOps, bool& hasViewOrAssemble)
{
    auto consumers = tensor->GetConsumers();
    for (auto* consumerOp : consumers) {
        auto opcode = consumerOp->GetOpcode();
        if (opcode == Opcode::OP_COPY_IN) {
            copyInOps.push_back(consumerOp);
        } else if (opcode == Opcode::OP_VIEW || opcode == Opcode::OP_ASSEMBLE || opcode == Opcode::OP_ASSEMBLE_SSA) {
            hasViewOrAssemble = true;
        }
    }
}

void RemoveUnalignedReshape::CollectReshapeOps(Function& function)
{
    for (auto& op : function.Operations()) {
        if (op.GetOpcode() != Opcode::OP_RESHAPE || processedReshapeOps.count(op.GetOpMagic())) {
            continue;
        }

        if (!CheckUnaligned(op)) {
            continue;
        }

        auto input = op.GetIOperands().front();
        auto output = op.GetOOperands().front();
        if ((input->GetMemoryTypeOriginal() != MemoryType::MEM_UB) ||
            (output->GetMemoryTypeOriginal() != MemoryType::MEM_UB)) {
            continue;
        }

        // 插入copyout
        LogicalTensorPtr newReshapeInput = InsertIOTensor(function, op, reshapeRawInputs, input);
        copyOuts.emplace_back(
            CopyOutOpMemUnalign{input->GetMemoryTypeOriginal(), input->offset, input, newReshapeInput});
        op.ReplaceInput(newReshapeInput, input);

        // 插入copyin
        LogicalTensorPtr newReshapeOutput = InsertIOTensor(function, op, reshapeRawOutputs, output);
        copyIns.emplace_back(
            CopyInOpMemUnalign{output->GetMemoryTypeOriginal(), output->offset, newReshapeOutput, output});
        op.ReplaceOutput(newReshapeOutput, output);
        output->tensor->actualRawmagic = -1;
        op.GetOOperands().front()->tensor->actualRawmagic = op.GetIOperands().front()->tensor->GetRawMagic();
        processedReshapeOps.insert(op.GetOpMagic());
    }
}

} // namespace npu::tile_fwk
