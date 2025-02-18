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

using namespace npu::tile_fwk;
namespace npu::tile_fwk {
namespace {
bool EqualShapeInOut(const Operation &op) {
    auto in = op.GetIOperands().front();
    auto out = op.GetOOperands().front();
    // 比较静态shape
    bool equalShape = (in->GetShape() == out->GetShape());
    // 比较动态dynValidShape_
    bool equalDynValidShape = true;
    if (!in->GetDynValidShape().empty() && !out->GetDynValidShape().empty()) {
        auto inDynValidShape = in->GetDynValidShape();
        auto outDynValidShape = out->GetDynValidShape();
        for (size_t i = 0; i < inDynValidShape.size(); i++) {
            // 比较SymbolicScalar dump后的string是否相等
            // 可能是 1.concrete value; 2.symbol; 3.expression
            if (inDynValidShape[i].Dump() == outDynValidShape[i].Dump()) {
                continue;
            } else {
                equalDynValidShape = false;
                break;
            }
        }
    } else if (in->GetDynValidShape().empty() && out->GetDynValidShape().empty()) {
        // 输入和输出同时没有 dynamic valid shape
        equalDynValidShape = true;
    } else {
        // 输入和输出必须同时有 dynamic valid shape
        equalDynValidShape = false;
    }
    return (equalShape && equalDynValidShape);
}

bool AllValidProdView(const Operation &op, const Function &function) {
    bool allProdView = true;
    for (const auto &prod : function.FindProducers(op)) {
        if (prod->GetOpcode() != Opcode::OP_VIEW) {
            allProdView = false;
        }
    }
    if (allProdView) {
        allProdView = false;
        for (const auto &prod : function.FindProducers(op)) {
            auto in = prod->iOperand.front();
            auto out = prod->oOperand.front();
            // 只要存在无法被删除的就表明allprodview为true(标记为无法删除)
            if (!EqualShapeInOut(*prod) || in->GetMemoryTypeOriginal() != out->GetMemoryTypeOriginal()) {
                allProdView = true;
            }
        }
    }
    return allProdView;
}

Status ProcessRegCopy(const Operation &op, const Function &function, bool &needToDelete) {
    auto regCopyIn = op.iOperand.front();
    auto regCopyOut= op.oOperand.front();
    if (regCopyIn->shape == regCopyOut->shape && regCopyIn->GetMemoryTypeOriginal() == regCopyOut->GetMemoryTypeOriginal()) {
        /*
        register copy 输入和输出且memtype相同，无拷贝意义
        view -> ub -> register copy -> ub -> op
        view -> ub  -> op
        */
        auto consumerOps = function.FindConsumers(op);
        /* register copy 一定有后继op*/
        if (consumerOps.empty()) {
            APASS_LOG_ERROR_F(Elements::Operation, 
            "OP_REG_COPY[%d]'s output has no consumer; OP_REG_COPY[%d]'s output must have consumer.%s", op.opmagic, op.opmagic, GetFormatBacktrace(op).c_str());
            return FAILED;
        }
        for (auto &consumerOp : consumerOps) {
            consumerOp->ReplaceInput(regCopyIn, regCopyOut);
        }
        needToDelete = true;
        APASS_LOG_DEBUG_F(Elements::Operation, "Delete Redundant OP_REGISTER_COPY opmagic: %d.", op.opmagic);
    }
    return SUCCESS;
}

Status ProcessAssembleDDR(const Operation &op, const LogicalTensorPtr &assembleIn, const LogicalTensorPtr &assembleOut,
    Function &function, bool &needToDelete) {
    if (AllValidProdView(op, function)) {
        return SUCCESS;
    }
    auto consumerOps = function.FindConsumers(op);
    if (!consumerOps.empty()) {
        for (auto &consumerOp : consumerOps) {
            consumerOp->ReplaceInput(assembleOut, assembleIn);
        }
        needToDelete = true;
        APASS_LOG_DEBUG_F(Elements::Operation, "Delete Redundant OP_ASSEMBLE on DDR opmagic: %d.", op.opmagic);
        return SUCCESS;
    }
    /* DDR --> Assemble --> OUTCAST */
    if (assembleOut->nodetype != NodeType::OUTCAST || !function.IsFromOutCast(assembleOut)) {
        APASS_LOG_ERROR_F(Elements::Operation, 
        "OP_ASSEMBLE[%d]'s output has no consumer but is not outcast; Please check if the OP_ASSEMBLE[%d]'s output is outcast.%s", 
        op.opmagic, op.opmagic, GetFormatBacktrace(op).c_str());
        return FAILED;
    }
    APASS_LOG_DEBUG_F(Elements::Operation, "OP_ASSEMBLE has no consumers, opmagic: %d.", op.opmagic);
    auto childOpsBackup = assembleIn->GetConsumers();
    for (auto &childOp : childOpsBackup) {
        if (childOp->GetOpMagic() == op.GetOpMagic()) {
            continue;
        }
        childOp->ReplaceInput(assembleOut, assembleIn);
        APASS_LOG_DEBUG_F(Elements::Tensor, "Repalce input of %s opmagic: %d, tensor %d --> tensor %d.", 
        childOp->GetOpcodeStr().c_str(), childOp->GetOpMagic(), assembleIn->magic, assembleOut->magic);
    }
    auto producerOps = op.ProducerOps();
    for (auto &producerOp : producerOps) {
        producerOp->ReplaceOutput(assembleOut, assembleIn);
        APASS_LOG_DEBUG_F(Elements::Tensor, "Repalce output of %s opmagic: %d, tensor %d --> tensor %d",
            producerOp->GetOpcodeStr().c_str(), producerOp->GetOpMagic(), assembleIn->magic, assembleOut->magic);
    }
    needToDelete = true;
    APASS_LOG_DEBUG_F(Elements::Operation, "Delete Redundant OP_ASSEMBLE on DDR opmagic: %d.", op.opmagic);
    return SUCCESS;
}

Status ProcessAssembleUB(const Operation &op, const LogicalTensorPtr &ASSEMBLE_in, const LogicalTensorPtr &ASSEMBLE_out,
    Function &function, bool &needToDelete) {
    // 如果输出存在多个producer则意味着是多assemble场景，则assemble不能消除。
    if (ASSEMBLE_out->GetProducers().size() > 1) {
        return SUCCESS;
    }
    /*
    assemble 输入和输出相同，无意义
    */
    auto consumerOps = function.FindConsumers(op);
    /* UB 上的 ASSEMBLE 一定有后继op*/
    if (consumerOps.empty()) {
        APASS_LOG_ERROR_F(Elements::Operation, 
        "OP_ASSEMBLE[%d]'s output is empty; OP_ASSEMBLE[%d]'s output for ub must have consumer.%s", op.opmagic, op.opmagic, GetFormatBacktrace(op).c_str());
        return FAILED;
    }
    for (auto &consumerOp : consumerOps) {
        consumerOp->ReplaceInput(ASSEMBLE_in, ASSEMBLE_out);
    }
    needToDelete = true;
    APASS_LOG_DEBUG_F(Elements::Operation, "Delete Redundant OP_ASSEMBLE on UB opmagic: %d", op.opmagic);
    return SUCCESS;
}

Status ProcessAssemble(const Operation &op, Function &function, bool &needToDelete) {
    auto ASSEMBLE_in = op.iOperand.front();
    auto ASSEMBLE_out = op.oOperand.front();
    if (ASSEMBLE_in->shape != ASSEMBLE_out->shape) {
        return SUCCESS;
    }
    if (ASSEMBLE_in->GetMemoryTypeOriginal() != ASSEMBLE_out->GetMemoryTypeOriginal()) {
        return SUCCESS;
    }
    if (ASSEMBLE_in->GetMemoryTypeOriginal() == MemoryType::MEM_DEVICE_DDR &&
        ProcessAssembleDDR(op, ASSEMBLE_in, ASSEMBLE_out, function, needToDelete)) {
        APASS_LOG_ERROR_F(Elements::Operation, "ProcessAssembleDDR failed.%s", GetFormatBacktrace(op).c_str());
        return FAILED;
    }
    if (ASSEMBLE_in->GetMemoryTypeOriginal() == MemoryType::MEM_UB &&
        ProcessAssembleUB(op, ASSEMBLE_in, ASSEMBLE_out, function, needToDelete)) {
        APASS_LOG_ERROR_F(Elements::Operation, "ProcessAssembleUB failed.%s", GetFormatBacktrace(op).c_str());
        return FAILED;
    }
    return SUCCESS;
}

Status ProcessView(const Operation &op, Function &function, bool &needToDelete) {
    auto in = op.iOperand.front();
    auto out = op.oOperand.front();
    if (EqualShapeInOut(op) && in->GetMemoryTypeOriginal() == out->GetMemoryTypeOriginal()) {
        auto consumerOps = function.FindConsumers(op);
        if (!consumerOps.empty()) {
            for (auto &consumerOp : consumerOps) {
                consumerOp->ReplaceInput(in, out);
            }
        }
        needToDelete = true;
        APASS_LOG_DEBUG_F(Elements::Operation, "Delete Redundant OP_VIEW opmagic: %d", op.opmagic);
    }
    if (out->GetConsumers().size() == 1) {
        auto childOp = *(out->GetConsumers().begin());
        if (childOp->GetOpcode() == Opcode::OP_COMM_WAIT_FLAG) {
            childOp->ReplaceInput(in, out);
            needToDelete = true;
        }
    }
    return SUCCESS;
}

/*
before:
                                            / --> child1
                                            /
inputTensor -> op (OP_EXPAND) -> outputTensor  --> child2
                                            \
                                            \ --> child3

after:
            / --> child1
            /
inputTensor    --> child2
            \
            \ --> child3
*/
Status ProcessExpand(const Operation &op, bool &needToDelete) {
    if (op.GetIOperands().size() != 1 || op.GetOOperands().size() != 1) {
        APASS_LOG_ERROR_F(Elements::Operation, 
        "Expand[%d] has incorrect input or output num; Please check the Expand[%d]'s input/output num.%s", 
        op.opmagic, op.opmagic, GetFormatBacktrace(op).c_str());
        return FAILED;
    }
    needToDelete = false;
    if (EqualShapeInOut(op)) {
        needToDelete = true;
    }
    return SUCCESS;
}
}

Status RemoveRedundantOp::RunOnFunction(Function &function) {
    APASS_LOG_INFO_F(Elements::Function, "===> Start RemoveRedundantOp");
    if (RemoveViewAssemble(function) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Function, "RemoveDummyExpand failed.");
        return FAILED;
    }
    if (DeleteRedundantOps(function) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Function, "DeleteRedundantOps failed.");
        return FAILED;
    }
    if (RemoveDummyExpand(function) != SUCCESS) {
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

Status RemoveRedundantOp::RemoveDummyExpand(Function &function) const {
    std::vector<Operation *> dummyOp;
    bool needToDelete;
    for (auto &op: function.Operations()) {
        if (op.GetOpcode() == Opcode::OP_EXPAND) {
            if (ProcessExpand(op, needToDelete) != SUCCESS) {
                APASS_LOG_ERROR_F(Elements::Operation, "ProcessExpand failed.%s", GetFormatBacktrace(op).c_str());
                return FAILED;
            }
            if (needToDelete) {
                dummyOp.push_back(&op);
                APASS_LOG_INFO_F(Elements::Operation, "Delete OP_EXPAND opmagic: %d.", op.opmagic);
            }
        }
    }
    for (const auto &op : dummyOp) {
        function.UpdateOperandBeforeRemoveOp(*op, false);
    }
    for (auto op : dummyOp) {
        if (op->IsDeleted()) {
            APASS_LOG_ERROR_F(Elements::Operation, 
            "Found invalid op[%d]; Please check the op[%d] is not deleted (RemoveDummyExpand).", op->opmagic, op->opmagic);
            return FAILED;
        }
        op->SetAsDeleted();
    }
    function.EraseOperations(true);
    return SUCCESS;
}

Status RemoveRedundantOp::NeedToDelete(const Operation &op, Function &function, bool &needToDelete) const {
    needToDelete = false;
    auto opcode = op.GetOpcode();
    switch (opcode) {
        case Opcode::OP_REGISTER_COPY: {
            if (ProcessRegCopy(op, function, needToDelete)) {
                APASS_LOG_ERROR_F(Elements::Operation, "ProcessView failed.%s", GetFormatBacktrace(op).c_str());
                return FAILED;
            }
            break;
        }
        case Opcode::OP_ASSEMBLE: {
            if (ProcessAssemble(op, function, needToDelete)) {
                APASS_LOG_ERROR_F(Elements::Operation, "ProcessView failed.%s", GetFormatBacktrace(op).c_str());
                return FAILED;
            }
            break;
        }
        case Opcode::OP_VIEW: {
            if (ProcessView(op, function, needToDelete)) {
                APASS_LOG_ERROR_F(Elements::Operation, "ProcessView failed.%s", GetFormatBacktrace(op).c_str());
                return FAILED;
            }
            break;
        }
        default:
            break;
    }
    return SUCCESS;
}

Status RemoveRedundantOp::DeleteRedundantOps(Function &function) const {
    std::vector<Operation *> redundantOp;
    bool needToDelete;
    for (auto &op : function.Operations()) {
        if (NeedToDelete(op, function, needToDelete) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "NeedToDelete failed.%s", GetFormatBacktrace(op).c_str());
            return FAILED;
        }
        if (needToDelete) {
            redundantOp.push_back(&op);
        }
    }
    for (const auto &op : redundantOp) {
        function.HandleControlOps(*op, redundantOp);
        function.UpdateOperandBeforeRemoveOp(*op, false);
    }
    for (auto op : redundantOp) {
        if (op->IsDeleted()) {
            APASS_LOG_ERROR_F(Elements::Operation, 
            "Found invalid op[%d]; Please check the op[%d] is not deleted (DeleteRedundantOps).", op->opmagic, op->opmagic);
            return FAILED;
        }
        op->SetAsDeleted();
    }
    function.EraseOperations(true);
    return SUCCESS;
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
                ProcessPerfectMatch(function,startTensor,endTensor);
            }else {
                //case2：assemble的输出tensor是view输入tensor的一部分
                //       startTensor(inshape) ---> view1  ---> tempTensor1  --->  assemble1  ---> endTensor(outshape < inshape)
                //                            ---> view2  ---> tempTensor2  --->  assemble2 
                GenerateNewView(function,op,startTensor,endTensor);  
            }   
        }    
    }
    EraseRedundantAssemble(function);
    DeadOperationEliminator::EliminateDeadOperation(function);
    return SUCCESS;
}

//处理view输入和assemble输出完美匹配场景
void RemoveRedundantOp::ProcessPerfectMatch (Function &function,LogicalTensorPtr &startTensor,LogicalTensorPtr &endTensor) const{
    // Skip the CopyOut of OCAST
    if (endTensor->GetConsumers().size() == 0) {
        return;
    }
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
    for (auto &assembleConsumer : endTensor->GetConsumers()) {
        assembleConsumer->iOperand = {startTensor};
        startTensor->AddConsumer(assembleConsumer);
    }
    endTensor->GetConsumers().clear();
    function.GetTensorMap().Erase(endTensor);
}

//判断view输入是否非同源
bool RemoveRedundantOp::IsNotSameViewInput (LogicalTensorPtr &startTensor,LogicalTensorPtr &endTensor) const{
    for (auto &assembleOp : endTensor->GetProducers()) {
        if (assembleOp->GetIOperands().empty()) { 
            continue;
        }
        auto &tempTensor = assembleOp->GetIOperands().front();
        auto producers = tempTensor->GetProducers();
        if (producers.empty()) {
            return true;
        }else {
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
bool RemoveRedundantOp::IsDataReplace (LogicalTensorPtr &endTensor) const{
    for (auto &assembleOp : endTensor->GetProducers()) {
        if (assembleOp->GetIOperands().empty()) {
            continue;
        }
        auto &tempTensor = assembleOp->GetIOperands().front();
        auto producers = tempTensor->GetProducers();
        if (producers.empty()) {
            return true;
        }else {
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

void RemoveRedundantOp::GenerateNewView(Function &function,Operation &op,LogicalTensorPtr &startTensor,LogicalTensorPtr &endTensor) const {
    //查找最小的offset
    std::vector<long> newoffset(op.iOperand[0]->offset.size(),INT_MAX);

    for (size_t m = 0; m < op.iOperand[0]->offset.size(); m++) {
        for (auto &comsumerView : startTensor->GetConsumers()) {
            auto opcode = comsumerView->GetOpcode();
            if (opcode != Opcode::OP_VIEW) {
                continue;
            }
            if (comsumerView->GetOOperands().empty()) { 
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
    MemoryType newTenosrMem = endTensor->GetMemoryTypeOriginal();
    newViewTensor ->SetMemoryTypeBoth(newTenosrMem );
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
// 将输入tensor的producers为空的assemble节点删除
void RemoveRedundantOp::EraseRedundantAssemble(Function &function) const {
    std::vector<Operation *> redundantAssembles;
    for (auto &op : function.Operations()) {
        if (op.GetOpcode() !=  Opcode::OP_ASSEMBLE) {
            continue;
        }

        if (op.iOperand.front()->GetProducers().empty()) {
            redundantAssembles.push_back(&op);
        }
    }
    for (const auto &op : redundantAssembles) {
        function.HandleControlOps(*op, redundantAssembles);
        function.UpdateOperandBeforeRemoveOp(*op, false);
    }
    for (auto op : redundantAssembles) {
        ASSERT(!op->IsDeleted());
        op->SetAsDeleted();
    }
    function.EraseOperations(false);
}
} // namespace npu::tile_fwk