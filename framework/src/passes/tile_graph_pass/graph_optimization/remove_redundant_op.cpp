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

Status RemoveRedundantOp::ProcessRedundantOpWithDynShape(Operation &op, Function &function) const {
    if (!EqualInOut(op)) {
        APASS_LOG_DEBUG_F(Elements::Operation, "op[%d]'s input and output has unequal shape and dynshape, skip removing.", op.opmagic);
        return SUCCESS;
    }
    function.UpdateOperandBeforeRemoveOp(op, false);
    return SUCCESS;
}

Status RemoveRedundantOp::ProcessRedundantOpWithoutDynShape(Operation &op, Function &function) const {
    if (!EqualInOutShape(op)) {
        APASS_LOG_DEBUG_F(Elements::Operation, "op[%d]'s input and output has unequal shape, skip removing.", op.opmagic);
        return SUCCESS;
    }
    function.UpdateOperandBeforeRemoveOp(op, false);
    return SUCCESS;
}

Status RemoveRedundantOp::RemoveDummyOp(Function &function) const {
    for (auto &op: function.Operations()) {
        if (matchOpcodeWithDynshape.find(op.GetOpcode()) != matchOpcodeWithDynshape.end()) {
            if (ProcessRedundantOpWithDynShape(op, function) != SUCCESS) {
                APASS_LOG_ERROR_F(Elements::Operation, "ProcessRedundantOp failed.%s", GetFormatBacktrace(op).c_str());
                return FAILED;
            }
        }
        if (matchOpcodeWithoutDynshape.find(op.GetOpcode()) != matchOpcodeWithoutDynshape.end()) {
            if (ProcessRedundantOpWithoutDynShape(op, function) != SUCCESS) {
                APASS_LOG_ERROR_F(Elements::Operation, "ProcessRedundantOpWithoutDynShape failed.%s", GetFormatBacktrace(op).c_str());
                return FAILED;
            }
        }
    }
    DeadOperationEliminator::EliminateDeadOperation(function);
    return SUCCESS;
}

Status RemoveRedundantOp::RunOnFunction(Function &function) {
    APASS_LOG_INFO_F(Elements::Function, "===> Start RemoveRedundantOp");
    if (RemoveViewAssemble(function) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Function, "RemoveDummyExpand failed.");
        return FAILED;
    }
    if (RemoveDummyOp(function) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Function, "RemoveDummyExpand failed.");
        return FAILED;
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

Status RemoveRedundantOp::RemoveViewAssemble(Function &function) const {
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

void RemoveRedundantOp::RemoveViewAssembleForOutcast(Function &function, LogicalTensorPtr &startTensor, LogicalTensorPtr &endTensor) const{
    bool canRemove;
    for (auto &startConsumer: startTensor->GetConsumers()) {
        if (startConsumer->GetOpcode() != Opcode::OP_VIEW) {
            continue;
        }
        canRemove = true;
        for (auto &endProducer: startConsumer->ProducerOps()) {
            if (endProducer->GetOOperands().front() != endTensor || endProducer->GetOpcode() != Opcode::OP_ASSEMBLE) {
                canRemove = false;
            } else {
                function.UpdateOperandBeforeRemoveOp(*endProducer, false);
            }
        }
        if (canRemove) {
            function.UpdateOperandBeforeRemoveOp(*startConsumer, false);
        }
    }
}
    

//处理view输入和assemble输出完美匹配场景
void RemoveRedundantOp::ProcessPerfectMatch(Function &function, LogicalTensorPtr &startTensor,LogicalTensorPtr &endTensor) const{
    //step1：排除view输入非同源场景
    bool isNotSameViewInput = IsNotSameViewInput(startTensor,endTensor); //true表示view的输入非同源
    if (isNotSameViewInput) {
        APASS_LOG_DEBUG_F(Elements::Tensor, 
            "OP_ASSEMBLE'S output endTensor[%d] has different input except startTesnor[%d].", startTensor->magic, endTensor->magic);    
        return; 
    }
    //step2:排除assemble数据重排场景
    bool isDataRepalce = IsDataReplace(endTensor);  //true表示assemble后数据重排布
    if (isDataRepalce) {
        APASS_LOG_DEBUG_F(Elements::Tensor, 
            "OP_ASSEMBLE'S output endTensor[%d] is repalced comparing with startTesnor[%d].", startTensor->magic, endTensor->magic);
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
    }
}

//判断view输入是否非同源
bool RemoveRedundantOp::IsNotSameViewInput(LogicalTensorPtr &startTensor, LogicalTensorPtr &endTensor) const{
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
bool RemoveRedundantOp::IsDataReplace(LogicalTensorPtr &endTensor) const{
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

void RemoveRedundantOp::GenerateNewView(Function &function, Operation &op, LogicalTensorPtr &startTensor, LogicalTensorPtr &endTensor) const {
    //查找最小的offset
    std::vector<long> newoffset(op.iOperand[0]->offset.size(),INT_MAX);
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
            newoffset[m] = std::min(newoffset[m],viewOffset[m]);
        }
    }
    //新建一个logical tensor
    std::shared_ptr<LogicalTensor> input = startTensor;
    std::shared_ptr<RawTensor> newRawTensor = std::make_shared<RawTensor>(input->Datatype(), input->GetShape(), input->Format());;
    std::shared_ptr<LogicalTensor> newViewTensor = std::make_shared<LogicalTensor>(function, newRawTensor, newoffset, endTensor->shape);
    newViewTensor->SetMemoryTypeBoth(endTensor->GetMemoryTypeOriginal());
    //新建一个view op
    auto &newViewOp = function.AddOperation(Opcode::OP_VIEW, {startTensor}, {newViewTensor});
    auto viewAttribute = std::make_shared<ViewOpAttribute>(
        newoffset, SymbolicScalar::FromConcrete(newoffset), newViewTensor->GetDynValidShape());
    newViewOp.SetOpAttribute(viewAttribute);
    //更新图链接关系:清除endTensor的消费者，清除endTensor，将assemble的消费者连接到newView
    for (auto &assembleConsumer : endTensor->GetConsumers()) {
        assembleConsumer->iOperand = {newViewTensor};
        newViewTensor->AddConsumer(assembleConsumer);
    }
    endTensor->GetConsumers().clear();
    function.GetTensorMap().Erase(endTensor);
}
} // namespace tile_fwk
} // namespace npu