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
 * \file auto_cast.cpp
 * \brief
 */

#include "auto_cast.h"
#include "interface/tensor/logical_tensor.h"
#include "passes/pass_check/auto_cast_checker.h"
#include "passes/pass_utils/dead_operation_eliminate.h"

namespace npu {
namespace tile_fwk {

Status AutoCast::GetInOutConnectedTensor(Function &function) {
    inCastConnectedTensors_.clear();
    outCastConnectedTensors_.clear();

    std::vector<std::shared_ptr<LogicalTensor>> inCastConnected(function.inCasts_);
    while (inCastConnected.size() > 0) {
        std::shared_ptr<LogicalTensor> currTensor = inCastConnected.back();
        inCastConnected.pop_back();
        if (inCastConnectedTensors_.count(currTensor->GetMagic()) > 0) {
            continue;
        }
        inCastConnectedTensors_.insert(currTensor->GetMagic());
        for (auto &consumer : currTensor->GetConsumers()) {
            if (consumer->GetOpcode() != Opcode::OP_VIEW) {
                continue;
            }
            for (auto &tensor : consumer->GetOOperands()) {
                inCastConnected.push_back(tensor);
            }
        }
    }

    std::vector<std::shared_ptr<LogicalTensor>> outCastConnected(function.outCasts_);
    while (outCastConnected.size() > 0) {
        std::shared_ptr<LogicalTensor> currTensor = outCastConnected.back();
        outCastConnected.pop_back();
        if (outCastConnectedTensors_.count(currTensor->GetMagic()) > 0) {
            continue;
        }
        outCastConnectedTensors_.insert(currTensor->GetMagic());
        for (auto &producer : currTensor->GetProducers()) {
            if (producer->GetOpcode() != Opcode::OP_ASSEMBLE) {
                continue;
            }
            for (auto &tensor : producer->GetIOperands()) {
                outCastConnected.push_back(tensor);
            }
        }
    }
    return SUCCESS;
}

Status AutoCast::RunOnFunction(Function &function) {
    ALOG_INFO_F("===> Start AutoCast for function [%s].", function.GetRawName().c_str());
    if (GetInOutConnectedTensor(function) != SUCCESS) {
        ALOG_ERROR_F("Failed to get InOutCast-connected tensor.");
        return FAILED;
    }
    if (InsertCast(function) != SUCCESS) {
        ALOG_ERROR_F("Failed to insert CAST for BF16 unsupported Operations.");
        return FAILED;
    }
    if (RemoveRedundantCastChain(function) != SUCCESS) {
        ALOG_ERROR_F("Failed to remove redundant CAST.");
        return FAILED;
    }
    ALOG_INFO_F("===> End AutoCast for function [%s].", function.GetRawName().c_str());
    return SUCCESS;
}

bool AutoCast::SupportBF16(Operation *op) {
    if (Platform::Instance().GetSoc().GetNPUArch() == NPUArch::DAV_3510) {
        if (UNSUPPORT_BF16_ARCH35_OPS.count(op->GetOpcode()) > 0) return false;
    } else {
        if (UNSUPPORT_BF16_OPS.count(op->GetOpcode()) > 0) {
            ALOG_INFO_F("Op[%d] can find in UNSUPPORT_BF16_OPS.", op->GetOpMagic());
            return false;
        }
    }
    return true;
}

void AutoCast::InsertCastOp(Function &function, LogicalTensorPtr src, LogicalTensorPtr tgt, 
                                       const TileShape &tileShape) {
    Operation &newCast = function.AddRawOperation(Opcode::OP_CAST, {src}, {tgt});
    newCast.SetAttribute(OP_ATTR_PREFIX + "mode", CastMode::CAST_NONE);
    newCast.UpdateTileShape(tileShape);
    addedCast_.insert(&newCast);
}

Status AutoCast::InsertCast(Function &function) {
    std::vector<Operation *> opList = function.Operations().DuplicatedOpList();
    std::unordered_map<int, std::shared_ptr<LogicalTensor>> oldMagic2Input;
    for (size_t opIdx = 0; opIdx < opList.size(); opIdx++) {
        Operation *op = opList[opIdx];
        if (SupportBF16(op)) {
            continue;
        }
        auto iOperands = op->GetIOperands();
        std::unordered_set<int> visitedIOp;
        for (auto &iop : iOperands) {
            if (visitedIOp.count(iop->GetMagic()) > 0 || iop->Datatype() != DataType::DT_BF16) {
                continue;
            }
            visitedIOp.insert(iop->GetMagic());
            if (oldMagic2Input.count(iop->GetMagic()) > 0) {
                auto newInput = oldMagic2Input[iop->GetMagic()];
                op->ReplaceInput(newInput, iop);
                continue;
            }
            auto newInput = std::make_shared<LogicalTensor>(function, DataType::DT_FP32, iop->shape, iop->GetDynValidShape(), iop->Format());
            InsertCastOp(function, iop, newInput, op->GetTileShape());
            op->ReplaceInput(newInput, iop);
            oldMagic2Input[iop->GetMagic()] = newInput;
            if (inCastConnectedTensors_.count(iop->GetMagic()) > 0) {
                inCastConnectedTensors_.insert(newInput->GetMagic());
            }
        }
        auto oOperands = op->GetOOperands();
        std::unordered_set<int> visitedOOp;
        for (auto &oop : oOperands) {
            if (visitedOOp.count(oop->GetMagic()) > 0 || oop->Datatype() != DataType::DT_BF16) {
                continue;
            }
            visitedOOp.insert(oop->GetMagic());
            if (oop->Datatype() == DataType::DT_BF16) {
                auto newOutput = std::make_shared<LogicalTensor>(function, DataType::DT_FP32, oop->shape, oop->GetDynValidShape(), oop->Format());
                op->ReplaceOutput(newOutput, oop);
                InsertCastOp(function, newOutput, oop, op->GetTileShape());
                oldMagic2Input[oop->GetMagic()] = newOutput;
                if (outCastConnectedTensors_.count(oop->GetMagic()) > 0) {
                    outCastConnectedTensors_.insert(newOutput->GetMagic());
                }
            }
        }
    }
    return SUCCESS;
}

bool AutoCast::IsLegalCast(DataType ds, DataType dt) {
    const std::set<std::pair<DataType, DataType>> legalCastPair {
        {DataType::DT_FP32, DataType::DT_FP16},
        {DataType::DT_FP16, DataType::DT_FP32},
        {DataType::DT_FP32, DataType::DT_BF16},
        {DataType::DT_BF16, DataType::DT_FP32},
        {DataType::DT_FP32, DataType::DT_BF16},
        {DataType::DT_BF16, DataType::DT_FP32},
        {DataType::DT_FP32, DataType::DT_INT16},
        {DataType::DT_INT16, DataType::DT_FP32},
        {DataType::DT_FP32, DataType::DT_INT32},
        {DataType::DT_INT32, DataType::DT_FP32},
        {DataType::DT_INT32, DataType::DT_FP16},
        {DataType::DT_FP16, DataType::DT_INT8},
        {DataType::DT_INT8, DataType::DT_FP16},
        {DataType::DT_FP32, DataType::DT_FP32},
        {DataType::DT_BF16, DataType::DT_INT32}
    };
    if (legalCastPair.count(std::make_pair(ds, dt)) > 0) {
        return true;
    }
    return false;
}

std::vector<Operation *> AutoCast::GetCastChain(Operation *tailOp)
{
    std::vector<Operation *> tailToHeadChain;
    bool isFront = false;
    Operation *currOp = tailOp;
    while (!isFront) {
        if (currOp->ProducerOps().size() != 1 ||
            (*currOp->ProducerOps().begin())->GetOpcode() != Opcode::OP_CAST ||
            addedCast_.count(*currOp->ProducerOps().begin()) == 0) {
            isFront = true;
            tailToHeadChain.push_back(currOp);
            break;
        }
        tailToHeadChain.push_back(currOp);
        currOp = *(currOp->ProducerOps().begin());
    }
    return tailToHeadChain;
}

Status AutoCast::ShortenChain(Function &function, const std::vector<Operation *> &castChain, Operation *tailOp)
{
    std::shared_ptr<LogicalTensor> tgtTensor = *(tailOp->GetOOperands().begin());
    DataType tgtType = tgtTensor->Datatype();
    bool isTgtOut = (tgtTensor->nodetype == NodeType::OUTCAST);
    bool isTgtOutConnected = (outCastConnectedTensors_.count(tgtTensor->GetMagic()) > 0);
    for (int i = static_cast<int>(castChain.size()) - 1; i >= 0; i--) {
        std::shared_ptr<LogicalTensor> srcTensor = *(castChain[i]->GetIOperands().begin());
        DataType srcType = srcTensor->Datatype();
        bool isSrcIn = (srcTensor->nodetype == NodeType::INCAST);
        bool isSrcOut = (srcTensor->nodetype == NodeType::OUTCAST);
        bool isSrcInConnected = (inCastConnectedTensors_.count(srcTensor->GetMagic()) > 0);
        if (srcType == tgtType && !(isTgtOutConnected && isSrcInConnected)) {
            if (!isTgtOut) {
                auto consumers = tgtTensor->GetConsumers();
                for (auto &consumerOp : consumers) {
                    consumerOp->ReplaceInput(srcTensor, tgtTensor);
                }
                break;
            }
            if (!isSrcIn && !isSrcOut) {
                auto srcProducers = srcTensor->GetProducers();
                auto srcConsumers = srcTensor->GetConsumers();
                auto tgtProducers = tgtTensor->GetProducers();
                for (auto &tgtProducerOp : tgtProducers) {
                    tgtProducerOp->ReplaceOutput(srcTensor, tgtTensor);
                }
                for (auto &srcProducerOp : srcProducers) {
                    srcProducerOp->ReplaceOutput(tgtTensor, srcTensor);
                }
                for (auto &srcConsumerOp : srcConsumers) {
                    srcConsumerOp->ReplaceInput(tgtTensor, srcTensor);
                }
                break;
            }
        }
        if (i != 0 && IsLegalCast(srcType, tgtType)) {
            tgtTensor->RemoveProducer(tailOp);
            auto origTileShape = (*srcTensor->GetConsumers().begin()) -> GetTileShape();
            InsertCastOp(function, srcTensor, tgtTensor, origTileShape);
            break;
        }
    }
    return SUCCESS;
}

Status AutoCast::RemoveRedundantCastChain(Function &function) {
    std::vector<Operation *> opList = function.Operations().DuplicatedOpList();
    for (size_t opIdx = 0; opIdx < opList.size(); opIdx++) {
        Operation *op = opList[opIdx];
        if (op->GetOpcode() != Opcode::OP_CAST || addedCast_.count(op) == 0) {
            continue;
        }
        bool allCast = true;
        for (auto &nextOp : op->ConsumerOps()) {
            if (nextOp->GetOpcode() != Opcode::OP_CAST) {
                allCast = false;
                break;
            }
        }
        if (allCast && op->ConsumerOps().size() > 0) {
            continue;
        }
        std::vector<Operation *> castChain = GetCastChain(op);
        ShortenChain(function, castChain, op);
    }
    return SUCCESS;
}

Status AutoCast::PreCheck(Function &function) {
    AutoCastChecker checker;
    return checker.DoPreCheck(function);
}

Status AutoCast::PostCheck(Function &function) {
    AutoCastChecker checker;
    return checker.DoPostCheck(function);
}
} // namespace tile_fwk
} // namespace npu
