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
 * \file remove_redundant_op.cpp
 * \brief
 */
#include <climits>
#include "remove_redundant_op.h"
#include "passes/pass_check/remove_redundant_op_checker.h"
#include "passes/pass_utils/dead_operation_eliminate.h"
#include "passes/pass_utils/merge_view_assemble_utils.h"
#include "passes/pass_log/pass_log.h"

#define MODULE_NAME "RemoveRedundantOp"

namespace npu {
namespace tile_fwk {
bool EqualInOutShape(const Operation &op) {
    auto in = op.GetIOperands().front();
    auto out = op.GetOOperands().front();
    // 比较memtype
    bool equalMemType = (in->GetMemoryTypeOriginal() == out->GetMemoryTypeOriginal());
    // 比较静态shape
    bool equalShape = (in->GetShape() == out->GetShape());
    return (equalMemType && equalShape);
}

bool EqualInOut(const Operation &op) {
    auto in = op.GetIOperands().front();
    auto out = op.GetOOperands().front();
    bool equalShape = EqualInOutShape(op);
    bool equalDynValidShape = true;
    if (!in->GetDynValidShape().empty() && !out->GetDynValidShape().empty()) {
        auto inDynValidShape = in->GetDynValidShape();
        auto outDynValidShape = out->GetDynValidShape();
        for (size_t i = 0; i < inDynValidShape.size(); i++) {
            if (inDynValidShape[i].Dump() != outDynValidShape[i].Dump()) {
                equalDynValidShape = false;
                break;
            }
        }
    } else if (in->GetDynValidShape().empty() && out->GetDynValidShape().empty()) {
        equalDynValidShape = true;
    } else {
        equalDynValidShape = false;
    }
    return (equalShape && equalDynValidShape);
}

bool RemoveRedundantOp::ProcessRedundantOpWithDynShape(Operation &op) const {
    if (!EqualInOut(op)) {
        APASS_LOG_DEBUG_F(Elements::Operation, "op[%d]'s input and output has unequal shape and dynshape, skip removing.", op.opmagic);
        return false;
    }
    return true;
}

bool RemoveRedundantOp::ProcessRedundantOpWithoutDynShape(Operation &op) const {
    if (!EqualInOutShape(op)) {
        APASS_LOG_DEBUG_F(Elements::Operation, "op[%d]'s input and output has unequal shape, skip removing.", op.opmagic);
        return false;
    }
    if (op.GetOpcode() == Opcode::OP_ASSEMBLE) {
        auto assembleOut = op.GetOOperands().front();
        if (assembleOut->GetProducers().size() > 1) {
            APASS_LOG_DEBUG_F(Elements::Operation, "assembleOut[%d] has more than one producer, skip removing.", assembleOut->GetMagic());
            return false;
        }
    }
    return true;
}

Status RemoveRedundantOp::RemoveDummyOp(Function &function) {
    for (auto &op: function.Operations()) {
        bool canRemove = false;
        if (matchOpcodeWithDynshape.find(op.GetOpcode()) != matchOpcodeWithDynshape.end()) {
            canRemove = ProcessRedundantOpWithDynShape(op);
        } else if (matchOpcodeWithoutDynshape.find(op.GetOpcode()) != matchOpcodeWithoutDynshape.end()) {
            canRemove = ProcessRedundantOpWithoutDynShape(op);
        }
        if (canRemove) {
            operationUpdated = true;
            function.UpdateOperandBeforeRemoveOp(op, false);
        }
    }
    DeadOperationEliminator::EliminateDeadOperation(function);
    return SUCCESS;
}

Status RemoveRedundantOp::RunOnFunction(Function &function) {
    APASS_LOG_INFO_F(Elements::Function, "===> Start RemoveRedundantOp");
    operationUpdated = true;
    iterTime = 0U;
    while (operationUpdated) {
        operationUpdated = false;
        if (RemoveDummyOps(function) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Function, "RemoveDummyOps failed.");
            return FAILED;
        }
        if (RemoveDummyOp(function) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Function, "RemoveDummyOp failed.");
            return FAILED;
        }
        iterTime++;
    }
    Status status = MergeViewAssembleUtils::MergeViewAssemble(function);
    if (status != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Function, "Merge assemble and view failed.");
        return status;
    }
    APASS_LOG_INFO_F(Elements::Function, "===> End RemoveRedundantOp");
    return SUCCESS;
}

Status RemoveRedundantOp::PreCheck(Function &function) {
    RemoveRedundantOpChecker checker;
    return checker.DoPreCheck(function);
}

Status RemoveRedundantOp::PostCheck(Function &function) {
    RemoveRedundantOpChecker checker;
    return checker.DoPostCheck(function);
}

Status RemoveRedundantOp::RemoveDummyOps(Function &function) {
    if (ProcessReshape(function) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Function, "ProcessReshape failed.");
        return FAILED;
    }
    if (iterTime == 0U) {
        if (ProcessViewAssemble(function) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Function, "ProcessViewAssemble failed.");
            return FAILED;
        }
    }
    return SUCCESS;
}

Status RemoveRedundantOp::ProcessViewAssemble(Function &function) {
    for (auto &op : function.Operations()) {
        auto opcode = op.GetOpcode();
        if(opcode != Opcode::OP_VIEW) {
            //跳过非view的op
            continue;
        }
        auto &startTensor = op.iOperand.front();
        auto inputMemtype = startTensor->GetMemoryTypeOriginal();
        auto consumers = op.oOperand.front()->GetConsumers();
        //获取view级联的assemble消费者
        for (const auto &consumer : consumers) {
            if (consumer->GetOpcode() != Opcode::OP_ASSEMBLE) {
                //跳过不是assemble的消费者
                continue;
            }
            auto &endTensor = consumer->oOperand.front();
            if (function.IsFromInCast(startTensor) && function.IsFromOutCast(endTensor)) {
                continue;
            }
            auto outputMemtype = endTensor->GetMemoryTypeOriginal();
            if (inputMemtype != outputMemtype) {
                //跳过view输入和 assemble输出 mem类型不同的场景
                continue;
            }
            if (startTensor->shape == endTensor->shape && startTensor->offset == endTensor->offset ) {
                //case1：view输入和assemble输出tensor shape和offset完全匹配
                //      startTensor(inshape) ---> view1  ---> tempTensor1  --->  assemble1  ---> endTensor(outshape = inshape)
                //                           ---> view2  ---> tempTensor2  --->  assemble2 
                APASS_LOG_DEBUG_F(Elements::Operation, 
                    "CASE1: Process OP_VIEW[%d]'s input and OP_ASSEMBLE[%d]'s output perfectMatch.", op.opmagic, consumer->GetOpMagic());
                ProcessPerfectMatch(function, startTensor, endTensor);
            } else {
                //case2：assemble的输出tensor是view输入tensor的一部分
                //       startTensor(inshape) ---> view1  ---> tempTensor1  --->  assemble1  ---> endTensor(outshape < inshape)
                //                            ---> view2  ---> tempTensor2  --->  assemble2 
                APASS_LOG_DEBUG_F(Elements::Operation, 
                    "CASE2: Process OP_VIEW[%d]'s input is a part of OP_ASSEMBLE[%d]'s output.", op.opmagic, consumer->GetOpMagic());
                GenerateNewView(function, op, startTensor, endTensor);  
            }   
        }    
    }
    DeadOperationEliminator::EliminateDeadOperation(function);
    return SUCCESS;
}

void RemoveRedundantOp::RemoveViewAssembleForOutcast(Function &function, LogicalTensorPtr &startTensor, LogicalTensorPtr &endTensor) {
    bool canRemove;
    std::set<Operation *> removeOp;
    for (auto &startConsumer: startTensor->GetConsumers()) {
        if (startConsumer->GetOpcode() != Opcode::OP_VIEW) {
            continue;
        }
        canRemove = true;
        for (auto &endProducer: startConsumer->ConsumerOps()) {
            if (endProducer->GetOOperands().front() != endTensor || endProducer->GetOpcode() != Opcode::OP_ASSEMBLE) {
                canRemove = false;
            } else {
                removeOp.insert(endProducer);
                operationUpdated = true;
            }
        }
        if (canRemove) {
            removeOp.insert(startConsumer);
        }
    }
    for (auto &op : removeOp) {
        function.UpdateOperandBeforeRemoveOp(*op, false);
        operationUpdated = true;
    }
}

//处理view输入和assemble输出完美匹配场景
void RemoveRedundantOp::ProcessPerfectMatch(Function &function, LogicalTensorPtr &startTensor,LogicalTensorPtr &endTensor) {
    if (!IsValidViewAssemble(startTensor, endTensor)) {
        APASS_LOG_DEBUG_F(Elements::Tensor, "Not valid view-assemble case.");    
        return; 
    }
    //图重连逻辑
    if (endTensor->GetConsumers().size() == 0) {
        RemoveViewAssembleForOutcast(function, startTensor, endTensor);
    } else {
        for (auto &assembleConsumer : endTensor->GetConsumers()) {
            assembleConsumer->iOperand = {startTensor};
            startTensor->AddConsumer(assembleConsumer);
        }
        endTensor->GetConsumers().clear();
        function.GetTensorMap().Erase(endTensor);
        operationUpdated = true;
    }
}

//判断view输入是否非同源
bool RemoveRedundantOp::IsNotSameViewInput(LogicalTensorPtr &startTensor, LogicalTensorPtr &endTensor) const {
    for (auto &assembleOp : endTensor->GetProducers()) {
        if (assembleOp->GetIOperands().empty()) { 
            continue;
        }
        auto &tempTensor = assembleOp->GetIOperands().front();
        auto producers = tempTensor->GetProducers();
        if (producers.empty()) {
            return true;
        } else {
            auto &viewOps = tempTensor->GetProducers(); 
            for (auto &viewOp : viewOps) {
                if (viewOp->GetIOperands().empty()) {
                    continue;
                }
                if (viewOp->GetOpcode() != Opcode::OP_VIEW) {
                    continue;
                }
                auto &viewInTensor = viewOp->GetIOperands().front();
                if (viewInTensor != startTensor) {
                    return true;
                }
            }    
        } 
    }
    return false;
}

//判断assemble数据是否是重排场景
bool RemoveRedundantOp::IsDataReplace(LogicalTensorPtr &endTensor) const {
    for (auto &assembleOp : endTensor->GetProducers()) {
        if (assembleOp->GetIOperands().empty()) {
            continue;
        }
        auto &tempTensor = assembleOp->GetIOperands().front();
        auto producers = tempTensor->GetProducers();
        if (producers.empty()) {
            return true;
        } else {
            auto &viewOps = tempTensor->GetProducers(); 
            for (auto &viewOp : viewOps) {
                if (viewOp->GetIOperands().empty()) {
                    continue;
                }
                if (viewOp->GetOpcode() != Opcode::OP_VIEW) {
                    continue;
                }
                auto viewOpAttribute = dynamic_cast<ViewOpAttribute *>(viewOp->GetOpAttribute().get());
                auto viewOffset = viewOpAttribute->GetFrom();
                auto assembleOpAttribute = dynamic_cast<AssembleOpAttribute *>(assembleOp->GetOpAttribute().get());
                auto assembleOffset = assembleOpAttribute->GetToOffset();
                if (viewOffset != assembleOffset) { //跳过assemble数据重排场景
                    return true;
                }
            }
        }
    }
    return false;
}

bool RemoveRedundantOp::IsValidViewAssemble(LogicalTensorPtr &startTensor, LogicalTensorPtr &endTensor) const {
    //step1：排除view输入非同源场景
    bool isNotSameViewInput = IsNotSameViewInput(startTensor,endTensor); //true表示view的输入非同源
    if (isNotSameViewInput) {
        APASS_LOG_DEBUG_F(Elements::Tensor, 
            "OP_ASSEMBLE'S output endTensor[%d] has different input except startTesnor[%d].", startTensor->magic, endTensor->magic);    
        return false; 
    }
    //step2:排除assemble数据重排场景
    bool isDataRepalce = IsDataReplace(endTensor);  //true表示assemble后数据重排布
    if (isDataRepalce) {
        APASS_LOG_DEBUG_F(Elements::Tensor, 
            "OP_ASSEMBLE'S output endTensor[%d] is repalced comparing with startTesnor[%d].", startTensor->magic, endTensor->magic);
        return false; 
    }
    return true;
}

void RemoveRedundantOp::CalculateViewOffset(Operation &op, LogicalTensorPtr &startTensor, LogicalTensorPtr &endTensor, std::vector<long> &newoffset) {
    for (size_t m = 0; m < op.iOperand[0]->offset.size(); m++) {
        for (auto &comsumerView : startTensor->GetConsumers()) {
            auto opcode = comsumerView->GetOpcode();
            if (opcode != Opcode::OP_VIEW || comsumerView->GetOOperands().empty()) {
                continue;
            }
            auto &tempTensor = comsumerView->GetOOperands().front();
            //检查view输出的消费者，寻找assemble操作
            bool leadsToCurrentEndTesnor = false;
            for (auto &consumerAssemble : tempTensor->GetConsumers()) {
                if (consumerAssemble->GetOpcode() != Opcode::OP_ASSEMBLE) {
                    continue;
                }
                //检查assemble的输出是否是当前的endTensor
                if (!consumerAssemble->GetOOperands().empty() && consumerAssemble->GetOOperands().front() == endTensor) {
                    leadsToCurrentEndTesnor = true;
                    break;
                }
            }
            //如果当前view不经过assemble连接到当前endTensor,跳过不处理
            if (!leadsToCurrentEndTesnor) {
                continue;
            }
            //只处理satrtTensor->view->tempTensor->assemble->endTensor
            auto viewOpAttribute = dynamic_cast<ViewOpAttribute *>(comsumerView->GetOpAttribute().get());
            auto viewOffset = viewOpAttribute->GetFromOffset();
            newoffset[m] = std::min(newoffset[m], viewOffset[m]);
        }
    }
}

void RemoveRedundantOp::GenerateNewView(Function &function, Operation &op, LogicalTensorPtr &startTensor, LogicalTensorPtr &endTensor) {
    //查找最小的offset
    if (!IsValidViewAssemble(startTensor, endTensor)) {
        APASS_LOG_DEBUG_F(Elements::Tensor, "Not valid view-assemble case.");    
        return; 
    }
    std::vector<long> newoffset(op.iOperand[0]->offset.size(),INT_MAX);
    CalculateViewOffset(op, startTensor, endTensor, newoffset);
    //新建一个logical tensor并更新图链接关系:清除endTensor的消费者，清除endTensor，将assemble的消费者连接到newView
    LogicalTensorPtr newViewTensor;
    if (endTensor->GetConsumers().empty()) {
        newViewTensor = endTensor;
        RemoveViewAssembleForOutcast(function, startTensor, endTensor);
    } else {
        std::vector<long> curOffset(endTensor->shape.size(), 0);
        newViewTensor = std::make_shared<LogicalTensor>(function, endTensor->GetRawTensor(), curOffset, endTensor->shape);
        newViewTensor->SetMemoryTypeBoth(endTensor->GetMemoryTypeOriginal());
        for (auto &assembleConsumer : endTensor->GetConsumers()) {
            assembleConsumer->iOperand = {newViewTensor};
            newViewTensor->AddConsumer(assembleConsumer);
        }
        endTensor->GetConsumers().clear();
        function.GetTensorMap().Erase(endTensor);
    }
    //新建一个view op
    auto &newViewOp = function.AddOperation(Opcode::OP_VIEW, {startTensor}, {newViewTensor});
    auto viewAttribute = std::make_shared<ViewOpAttribute>(
        newoffset, SymbolicScalar::FromConcrete(newoffset), newViewTensor->GetDynValidShape());
    newViewOp.SetOpAttribute(viewAttribute);
    operationUpdated = true;
}

Status RemoveRedundantOp::ProcessReshape(Function &function) {
    bool canRemove;
    for (auto &op : function.Operations()) {
        auto opcode = op.GetOpcode();
        if (opcode != Opcode::OP_RESHAPE) {
            //跳过非reshape的op
            continue;
        }
        auto in = op.GetIOperands().front();
        auto out = op.GetOOperands().front();
        canRemove = false;
        if (in->shape == out->shape) {
            APASS_LOG_DEBUG_F(Elements::Operation, "op[%d]'s in->shape == out->shape.", op.GetOpMagic());
            canRemove = true;
        } else if (!op.ConsumerOps().empty()) {
            canRemove = true;
            for (auto &consumerOp : op.ConsumerOps()) {
                if (consumerOp->GetOpcode() != Opcode::OP_RESHAPE) {
                    canRemove = false;
                    continue;
                }
                consumerOp->ReplaceInput(in, out);
            }
            if (canRemove) {
                APASS_LOG_DEBUG_F(Elements::Operation, "All consummers of op [%d] are reshape.", op.GetOpMagic());
            }
        }
        if (canRemove) {
            function.UpdateOperandBeforeRemoveOp(op, false);
            operationUpdated = true;
        }
    }
    DeadOperationEliminator::EliminateDeadOperation(function);
    return SUCCESS;
}
} // namespace tile_fwk
} // namespace npu