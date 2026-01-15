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
 * \file assign_memory_type.cpp
 * \brief
 */

#include "assign_memory_type.h"

#include "interface/function/function.h"
#include "interface/tensor/logical_tensor.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/program/program.h"
#include "interface/configs/config_manager.h"
#include "passes/pass_check/assign_memory_type_checker.h"
#include "passes/pass_log/pass_log.h"

#define MODULE_NAME "AssignMemoryType"

namespace npu::tile_fwk {

Status AssignMemoryType::RunOnFunction(Function &function) {
    APASS_LOG_INFO_F(Elements::Function, "===> Start AssignMemoryType.");
    for (auto &op : function.Operations()) {
        RunOnOperation(op);
    }
    for (auto &op : function.Operations()) {
        AssignMoveOp(op);
    }
    for (auto &incast : function.inCasts_) {
        /*
        设置INCAST的memory type为DDR
        将INCAST的每个consumer加到其tobeMap中，tobe=DDR
                /--> op1 --> tensor1 -->
        INCAST  ---> op2 --> tensor2 -->
                \--> op3 --> tensor3 -->
        */
        incast->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
        for (const auto &consumerOp : incast->GetConsumers()) {
            inserter.UpdateTensorTobeMap(incast, *consumerOp, MemoryType::MEM_DEVICE_DDR);
        }
    }
    for (auto &outcast : function.outCasts_) {
        /*
        设置OUTCAST的memory type为DDR，因为OCAST没有consumer，所以tobeMap为空
        op --> tensor --> op1 -->\
        op --> tensor --> op2 ---> OCAST
        op --> tensor --> op3 -->/
        */
        outcast->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    }
    AssignMemUnknown(function);
    bool infoBufferSize = false;
    for (auto &op : function.Operations()) {
        AssignSpecialOpMemtype(op, infoBufferSize);
    }
    if (infoBufferSize) {
        const size_t UB_SIZE_THRESHOLD = Platform::Instance().GetDie().GetMemoryLimit(MemoryType::MEM_UB) / 2;
        const size_t L1_SIZE_THRESHOLD = Platform::Instance().GetDie().GetMemoryLimit(MemoryType::MEM_L1) / 2;
        APASS_LOG_INFO_F(Elements::Operation, "UB buffer size threshold %zu, L1 buffer size threshold %zu.",
            UB_SIZE_THRESHOLD, L1_SIZE_THRESHOLD);
    }
    //处理cube级联场景tile等大约束
    ProcesSmallTileToLargeTile(function);
    ProcessLargeTileToSamllTile(function);

    // 插入convert op
    Status insertionStatus = inserter.DoInsertion(function);
    if(insertionStatus != SUCCESS) {return insertionStatus;}
    APASS_LOG_INFO_F(Elements::Function, "===> End AssignMemoryType.");
    return SUCCESS;
}
Status AssignMemoryType::PreCheck(Function &function){
    AssignMemoryTypeChecker checker;
    return checker.DoPreCheck(function);
}

void AssignMemoryType::RunOnOperation(Operation &operation) {
    APASS_LOG_DEBUG_F(Elements::Function, "--- AssignMemoryType::RunOnOperation %s[%d] ---",
        operation.GetOpcodeStr().c_str(), operation.GetOpMagic());
    auto opcode = operation.GetOpcode();
    const auto &inputsMemType = OpcodeManager::Inst().GetInputsMemType(opcode);
    for (size_t i = 0; i < operation.iOperand.size(); ++i) {
        auto &tensor = operation.iOperand[i];
        if (i >= inputsMemType.size()) {
            APASS_LOG_INFO_F(Elements::Operation, "%s[%d] input %d magic %d mem original is NOT Defined in opcode.cpp.",
                operation.GetOpcodeStr().c_str(), operation.GetOpMagic(), i, tensor->magic);
            continue;
        }
        APASS_LOG_DEBUG_F(Elements::Operation, "%s[%d] input %d mem original %s --> %s.", operation.GetOpcodeStr().c_str(),
            operation.GetOpMagic(), tensor->magic, BriefMemoryTypeToString(tensor->GetMemoryTypeOriginal()).c_str(),
            BriefMemoryTypeToString(inputsMemType[i]).c_str());
        if(opcode == Opcode::OP_A_MUL_B || opcode == Opcode::OP_A_MULACC_B) {
            //对A_MUL_B的输入tensor的mem设置做特殊处理
            ProcessAmulBInput(operation, tensor);
            continue;
        }
        tensor->SetMemoryTypeOriginal(inputsMemType[i]); // 如果tensor之前做为oOperand被设置过, 那么这里不生效
        inserter.UpdateTensorTobeMap(tensor, operation, inputsMemType[i]);
    }
    const auto &outputsMemType = OpcodeManager::Inst().GetOutputsMemType(opcode);
    for (size_t i = 0; i < operation.oOperand.size(); ++i) {
        auto &tensor = operation.oOperand[i];
        if (outputsMemType.size() > 0) {
            APASS_LOG_DEBUG_F(Elements::Operation, "%s[%d] output %d mem original %s --> %s.", operation.GetOpcodeStr().c_str(),
                operation.GetOpMagic(), tensor->magic, BriefMemoryTypeToString(tensor->GetMemoryTypeOriginal()).c_str(),
                BriefMemoryTypeToString(outputsMemType[i]).c_str());
            tensor->SetMemoryTypeOriginal(outputsMemType[i]);
            for (const auto &consumerOp : tensor->GetConsumers()) {
                inserter.UpdateTensorTobeMap(tensor, *consumerOp, outputsMemType[i]);
            }
            continue;
        }
        tensor->SetMemoryTypeOriginal(MemoryType::MEM_UNKNOWN);
        for (const auto &consumerOp : tensor->GetConsumers()) {
            APASS_LOG_DEBUG_F(Elements::Operation, "Set for Unknown Op's consumer %s[%d].",
                consumerOp->GetOpcodeStr().c_str(), consumerOp->GetOpMagic());
            inserter.UpdateTensorTobeMap(tensor, *consumerOp, MemoryType::MEM_UNKNOWN);
        }
    }
    if(operation.GetOpcode() == Opcode::OP_VIEW) {
        ProcessViewwithSpecificMem(operation);
    }
    if(operation.GetOpcode() == Opcode::OP_ASSEMBLE) {
        ProcessAssemblewithSpecificMem(operation);
    }
}
void AssignMemoryType::ProcessAmulBInput(Operation &operation, LogicalTensorPtr &tensor) {
    /*
    operation: OP_A_MUL_B or OP_A_MULACC_B
    tensor: an input of OP_A_MUL_B
    */
    auto &producerOps = tensor->GetProducers();
    for(const auto &producerOp : producerOps) {
        auto producerOpcode = producerOp->GetOpcode();
        if (producerOpcode == Opcode::OP_A_MUL_B || producerOpcode == Opcode::OP_A_MULACC_B) {
            tensor->SetMemoryTypeOriginal(MemoryType::MEM_L0C, true);
            inserter.UpdateTensorTobeMap(tensor, operation, MemoryType::MEM_L0C);
            continue;
        } else if (producerOpcode == Opcode::OP_VIEW) {
            auto viewOpAttribute = dynamic_cast<ViewOpAttribute *>(producerOp->GetOpAttribute().get());
            MemoryType attrToType = viewOpAttribute->GetTo();
            tensor->SetMemoryTypeOriginal(attrToType, true);
            inserter.UpdateTensorTobeMap(tensor,operation, attrToType);
            continue;
        }else if (producerOpcode == Opcode::OP_L1_TO_L0A || producerOpcode == Opcode::OP_L1_TO_L0_AT) {
            tensor->SetMemoryTypeOriginal(MemoryType::MEM_L0A, true);
            inserter.UpdateTensorTobeMap(tensor, operation, MemoryType::MEM_L0A);
            continue;
        }else if (producerOpcode == Opcode::OP_L1_TO_L0B || producerOpcode == Opcode::OP_L1_TO_L0_BT) {
            tensor->SetMemoryTypeOriginal(MemoryType::MEM_L0B, true);
            inserter.UpdateTensorTobeMap(tensor, operation, MemoryType::MEM_L0B);
            continue;
        }else{
            inserter.UpdateTensorTobeMap(tensor,operation, MemoryType::MEM_DEVICE_DDR);
        }
    }
}
void AssignMemoryType::ProcessViewwithSpecificMem(Operation &operation) {
    auto viewOpAttribute = dynamic_cast<ViewOpAttribute *>(operation.GetOpAttribute().get());
    MemoryType attrToType = viewOpAttribute->GetTo();
    auto out = operation.GetOOperands().front();
    auto in = operation.iOperand.front();
    if(attrToType == MemoryType::MEM_UNKNOWN) {
        //跳过前端没有指定mem类型的view
        //适配L0C2L1通路，优先选择将view转化为L0C2L1，不满足场景后续转为ddr
        if (in->GetMemoryTypeOriginal() == MemoryType::MEM_L0C && out->GetMemoryTypeOriginal() == MemoryType::MEM_L1) {
            inserter.UpdateTensorTobeMap(in,operation,MemoryType::MEM_L0C);
        }
        return;
    }
    //将view的输出tensor的memory ori和tobe类型设置为view上指定的mem类型
    out->SetMemoryTypeOriginal(attrToType,true); 
    for (auto &consumerOp : out->GetConsumers()) {
        inserter.UpdateTensorTobeMap(out,*consumerOp,attrToType);
    }
    if(attrToType == MemoryType::MEM_L1) {
        auto producerOps = operation.ProducerOps();
        for(const auto &producerOp : producerOps) {
            if(producerOp->GetOpcode() == Opcode::OP_VIEW) {
                in->SetMemoryTypeOriginal(attrToType,true);
                inserter.UpdateTensorTobeMap(in,operation,attrToType);
            }
        }
    }
}
// 适配L0C2L1: assemble的输入来源时l0c且输出预期l1时，不再插convert而是assemble后续转为l0c2l1
void AssignMemoryType::ProcessAssemblewithSpecificMem(Operation &operation) {
    auto input =operation.iOperand.front();
    auto output =operation.oOperand.front();
    if (input->GetMemoryTypeOriginal() != MemoryType::MEM_L0C) {
        return;
    }
    for (const auto &consumerOp : output->GetConsumers()) {
        auto consumerOpAttribute = std::dynamic_pointer_cast<ViewOpAttribute>(consumerOp->GetOpAttribute());
        // 大包搬运场景：assemble后接view且view的toAttr为L1
        // 非大包搬运场景：assemble后的op预期输入为L1
        if (consumerOpAttribute) {
            if (consumerOpAttribute->GetTo() != MemoryType::MEM_L1) {
                return;
            }
        } else {
            const auto &inputsMemType = OpcodeManager::Inst().GetInputsMemType(consumerOp->GetOpcode());
            if (!inputsMemType.empty() && inputsMemType[0] != MemoryType::MEM_L1) {
                return;
            }
        }
    }
    output->SetMemoryTypeOriginal(MemoryType::MEM_L1, true);
    inserter.UpdateTensorTobeMap(input, operation, MemoryType::MEM_L0C);
    for(const auto &consumerOp : output->GetConsumers()) {
        inserter.UpdateTensorTobeMap(output, *consumerOp,MemoryType::MEM_L1);
    }
    APASS_LOG_DEBUG_F(Elements::Operation, "Set assemble Op[%d]'s input[%d] tobeMap as MEM_L0C and output[%d] origin and tobeMap as MEM_L1.",
        operation.GetOpMagic(), input->magic, output->magic);
}

void AssignMemoryType::AssignMemtypeForSplitReshape(Operation &op, const LogicalTensorPtr &input, const LogicalTensorPtr &output) {
    const int UB_SIZE_THRESHOLD = static_cast<int>(Platform::Instance().GetDie().GetMemoryLimit(MemoryType::MEM_UB) * 0.5);
    // 如果reshape前序是Assemble后接View，是SplitReshape处理的Reshape
    auto &producers = input->GetProducers();
    auto &consumers = output->GetConsumers();
    Operation* producer = *producers.begin();
    Operation* consumer = *consumers.begin();
    if (producer != nullptr && consumer != nullptr && producer->GetOpcode() == Opcode::OP_ASSEMBLE && consumer->GetOpcode() == Opcode::OP_VIEW) {
        if (input->GetMemoryTypeOriginal() == MemoryType::MEM_UB && output->GetMemoryTypeOriginal() == MemoryType::MEM_UB && input->GetDataSize() <= UB_SIZE_THRESHOLD) {
            inserter.UpdateTensorTobeMap(input, op, MemoryType::MEM_UB);
            for (const auto &consumerOp : output->GetConsumers()) {
                if (consumerOp->oOperand.front()->GetMemoryTypeOriginal() == MemoryType::MEM_UB) {
                    inserter.UpdateTensorTobeMap(output, *consumerOp, MemoryType::MEM_UB);
                }
            }
        }
    }
}

void AssignMemoryType::AssignOpReshapeMemtype(Operation &op){
    if (op.GetOpcode() == npu::tile_fwk::Opcode::OP_RESHAPE) {
        auto &input = op.iOperand.front();
        auto &output = op.oOperand.front();
        AssignMemtypeForSplitReshape(op, input, output);
        auto inputMemType = inserter.GetMemoryTypeFromTensorTobeMap(input, op);
        if (inputMemType != output->GetMemoryTypeOriginal()) {
            APASS_LOG_DEBUG_F(Elements::Operation, "OP_RESHAPE[%d] input: %s, output: %s.",
                op.opmagic, PrintTensorMem(input).c_str(), PrintTensorMem(output).c_str());
            inserter.UpdateTensorTobeMap(input, op, MemoryType::MEM_DEVICE_DDR);
            output->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, true);
        }
    }
}

void AssignMemoryType::AssignOpViewTypeMemtype(Operation &op){
    if (op.GetOpcode() == npu::tile_fwk::Opcode::OP_VIEW_TYPE) {
        auto &viewTypeIn = op.iOperand.front();
        auto &viewTypeOut = op.oOperand.front();
        auto inputMemType = inserter.GetMemoryTypeFromTensorTobeMap(viewTypeIn, op);
        auto outTobeMem = inserter.GetTobeDefault(viewTypeOut);
        auto prod = *(viewTypeIn->GetProducers().begin());
        if (prod->GetOpcode() == Opcode::OP_VIEW) {
            viewTypeIn->SetMemoryTypeOriginal(viewTypeOut->GetMemoryTypeOriginal(), true);
            inserter.UpdateTensorTobeMap(viewTypeIn, op, viewTypeOut->GetMemoryTypeOriginal());
            return;
        }
        if (inputMemType != viewTypeOut->GetMemoryTypeOriginal()) {
            APASS_LOG_DEBUG_F(Elements::Operation, "OP_RESHAPE[%d] input: %s, output: %s.",
                op.opmagic, PrintTensorMem(viewTypeIn).c_str(), PrintTensorMem(viewTypeOut).c_str());
            inserter.UpdateTensorTobeMap(viewTypeIn, op, MemoryType::MEM_DEVICE_DDR);
            viewTypeOut->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, true);
        }
    }
}

void AssignMemoryType::AssignOpNopMemtype(Operation &op){
    if (op.GetOpcode() == npu::tile_fwk::Opcode::OP_NOP) {
        auto &input = op.iOperand.front();
        auto &output = op.oOperand.front();
        if (input->GetMemoryTypeToBe() != output->GetMemoryTypeOriginal()) {
            input->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
            output->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
        }
        /* no op一定不改变数据类型，也就是noop的后面一定会有view或者assemble*/
        if (output->GetMemoryTypeOriginal() != output->GetMemoryTypeToBe()) {
            output->SetMemoryTypeBoth(output->GetMemoryTypeOriginal(), true);
        }
    }
}

void AssignMemoryType::AssignSpecialOpMemtype(Operation &op, bool &infoBufferSize) {
    AssignOpReshapeMemtype(op);
    AssignOpViewTypeMemtype(op);
    AssignOpNopMemtype(op);
    if (op.GetOpcode() == Opcode::OP_REDUCE_ACC) {
        /*
        reduce acc 输入数量不确定，由Ksplit决定
        每个输入都为DDR
        */
        for (auto &input : op.GetIOperands()) {
            inserter.UpdateTensorTobeMap(input, op, MemoryType::MEM_DEVICE_DDR);
        }
    }

    if (op.GetOpcode() == Opcode::OP_SHMEM_WAIT_UNTIL) {
        /*
        每个输出都为DDR
        before：
        Incast --> View --> Gm --> COMM_WAIT_FLAG --> Gm
        after:
        Incast --> COMM_WAIT_FLAG -->Gm
        */
        for (auto &output : op.GetOOperands()) {
            output->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, true);
        }
    }

    if (op.GetOpcode() == npu::tile_fwk::Opcode::OP_ASSEMBLE) {
        auto &output = op.oOperand.front();
        if (output->GetMemoryTypeOriginal() == npu::tile_fwk::MEM_L1 && output->GetMemoryTypeToBe() == npu::tile_fwk::MEM_DEVICE_DDR) {
            output->SetMemoryTypeBoth(output->GetMemoryTypeOriginal(), true);
        }
        UpdateOverSizedLocalBuffer(op);
        infoBufferSize = true;
    }
}

void AssignMemoryType::UpdateOverSizedLocalBuffer(Operation &operation) {
    const int UB_SIZE_THRESHOLD =
        static_cast<int>(Platform::Instance().GetDie().GetMemoryLimit(MemoryType::MEM_UB) * 0.5);
    const int L1_SIZE_THRESHOLD =
        static_cast<int>(Platform::Instance().GetDie().GetMemoryLimit(MemoryType::MEM_L1) * 0.5);

    auto assembleOut = operation.GetOOperands().front();
    auto memType = assembleOut->GetMemoryTypeOriginal();
    if (((memType == MemoryType::MEM_UB) && (assembleOut->GetDataSize() > UB_SIZE_THRESHOLD)) ||
        ((memType == MemoryType::MEM_L1) && (assembleOut->GetDataSize() > L1_SIZE_THRESHOLD))) {
        assembleOut->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
        APASS_LOG_INFO_F(Elements::Operation, "%s[%d] output %d is oversized, set as MEM_DEVICE_DDR.",
            operation.GetOpcodeStr().c_str(), operation.GetOpMagic(), assembleOut->magic);
    }
}

std::string AssignMemoryType::PrintTensorMem(std::shared_ptr<LogicalTensor> &tensor) const {
    std::ostringstream oss;
    oss << "tensor magic: " << tensor->magic;
    oss << " original: " << BriefMemoryTypeToString(tensor->GetMemoryTypeOriginal());
    oss << ", tobe: " << BriefMemoryTypeToString(tensor->GetMemoryTypeToBe());
    return oss.str();
}

void AssignMemoryType::AssignMoveOp(Operation &operation) {
    auto opcode = operation.GetOpcode();
    switch (opcode) {
        case Opcode::OP_ASSEMBLE: {
            AssignMoveOpForAssemble(operation);
            break;
        }
        case Opcode::OP_VIEW: {
            AssignMoveOpForView(operation);
            break;
        }
        default: 
            break;
    }
}
void AssignMemoryType::AssignMoveOpForAssemble(Operation &operation) {
    /*
    op --> tensor1 --> assemble --> tensor2
    op是常规Op(output mem类型在opcode.cpp中有定义)，tensor1 的 mem original 已经被刷新好
    将tensor2 的mem origianl 刷新为tensor1 的original
    */
    for (size_t i = 0; i < operation.oOperand.size(); ++i) {
        auto &tensor = operation.oOperand[i];
        // Only change original type
        MemoryType fromType = inserter.GetMemoryTypeFromTensorTobeMap(operation.iOperand.front(), operation);
        for (const auto &outputProducer : tensor->GetProducers()) {
            if (fromType != MEM_DEVICE_DDR && outputProducer->iOperand.front()->GetMemoryTypeOriginal() != fromType) {
                fromType = MEM_DEVICE_DDR;
            }
        }
        APASS_LOG_DEBUG_F(Elements::Operation, "%s[%d] output %d mem original %s --> %s.", operation.GetOpcodeStr().c_str(),
            operation.GetOpMagic(), tensor->magic, BriefMemoryTypeToString(tensor->GetMemoryTypeOriginal()).c_str(),
            BriefMemoryTypeToString(fromType).c_str());
        if (operation.iOperand.front()->GetMemoryTypeOriginal() == MemoryType::MEM_L0C &&
            tensor->GetMemoryTypeOriginal() == MemoryType::MEM_L1) {
            continue;
        }
        tensor->SetMemoryTypeOriginal(fromType, true);
        auto assembleOpAttribute = std::dynamic_pointer_cast<AssembleOpAttribute>(operation.GetOpAttribute());
        assembleOpAttribute->SetFromType(fromType);
    }
}
void AssignMemoryType::AssignMoveOpForView(Operation &operation) {
    /*
    tensor1 --> view --> tensor2 --> op
    op是常规Op(output mem类型在opcode.cpp中有定义)，tensor2 的 mem original 已经被刷新好
    将tensor1 的 tobeMap 做更新
    */
    auto viewOpAttribute =dynamic_cast<ViewOpAttribute *>(operation.GetOpAttribute().get());
    MemoryType attrToType = viewOpAttribute->GetTo();
    bool isExplicitMemType = (attrToType != MemoryType::MEM_UNKNOWN);
    if(isExplicitMemType) {
        //跳过前端指定mem类型的view
        return;
    }
    auto outputTensor = operation.GetOOperands().front();
    auto viewOffset = viewOpAttribute->GetFromOffset();
    bool unaligned = ((BytesOf(outputTensor->Datatype()) * viewOffset.back()) % 32 != 0);
    for (size_t i = 0; i < operation.iOperand.size(); ++i) {
        auto &tensor = operation.iOperand[i];
        MemoryType toType = operation.oOperand.front()->GetMemoryTypeOriginal();
        // 当view的offset不对齐，需要插DDR时，不能将view前tensor的MemoryTypeOriginal向view后的tensor传递
        // 避免插DDR后出现original—>DDR->original而DDR->original不存在通路的场景（如original为L0C时）
        if(!unaligned && toType == MemoryType::MEM_UNKNOWN && tensor->GetMemoryTypeOriginal() != MemoryType::MEM_UNKNOWN) {
            //view输出的消费者是assemble或者reshape
            operation.oOperand.front()->SetMemoryTypeOriginal(tensor->GetMemoryTypeOriginal());
            viewOpAttribute->SetToType(tensor->GetMemoryTypeOriginal());
            continue;
        }
        if (tensor->GetMemoryTypeOriginal() == MemoryType::MEM_L0C && outputTensor->GetMemoryTypeOriginal() == MemoryType::MEM_L1) {
            inserter.UpdateTensorTobeMap(tensor, operation, MemoryType::MEM_L0C);
            viewOpAttribute->SetToType(outputTensor->GetMemoryTypeOriginal());
            continue;
        }
        APASS_LOG_DEBUG_F(Elements::Operation, "%s[%d] input %d mem original %s --> %s.", operation.GetOpcodeStr().c_str(),
            operation.GetOpMagic(), tensor->magic, BriefMemoryTypeToString(tensor->GetMemoryTypeOriginal()).c_str(),
            BriefMemoryTypeToString(toType).c_str());
        inserter.UpdateTensorTobeMap(tensor, operation, toType);
        viewOpAttribute->SetToType(toType);
    }
    if(unaligned) {
        auto inputTensor = operation.GetIOperands().front();
        inserter.UpdateTensorTobeMap(inputTensor, operation, MemoryType::MEM_DEVICE_DDR);
    }
}

void AssignMemoryType::AssignMemUnknown(Function &function) {
    std::unordered_set<LogicalTensorPtr> inputOperandVisited;
    std::unordered_set<LogicalTensorPtr> outputOperandVisited;
    for (auto &op : function.Operations()) {
        for (auto &i : op.iOperand) {
            if (inputOperandVisited.count(i) > 0) {
                continue;
            }
            inputOperandVisited.insert(i);
            if (i->GetMemoryTypeOriginal() == MemoryType::MEM_UNKNOWN) {
                MemoryType fromType = MemoryType::MEM_DEVICE_DDR;
                std::map<MemoryType, std::set<Operation *>> localTobeMap = inserter.GetRequiredTobe(i);
                if (localTobeMap.size() == 1 && localTobeMap.begin()->first != MemoryType::MEM_UNKNOWN) {
                    fromType = localTobeMap.begin()->first;
                }
                APASS_LOG_DEBUG_F(Elements::Operation, "%s[%d] iOperand %d mem original is UNKNOWN, force setting as %s.",
                    op.GetOpcodeStr().c_str(), op.GetOpMagic(), i->magic, BriefMemoryTypeToString(fromType).c_str());
                i->SetMemoryTypeOriginal(fromType);
                inserter.UpdateTensorTobeMap(i, op, fromType);
            }
            inserter.UpdateTensorTobeMapUnknown(i, i->GetMemoryTypeOriginal());
        }
        for (auto &o : op.oOperand) {
            if (outputOperandVisited.count(o) > 0) {
                continue;
            }
            outputOperandVisited.insert(o);
            if (o->GetMemoryTypeOriginal() == MemoryType::MEM_UNKNOWN) {
                /*
                说明该op的输出mem type 没有在opcode.cpp总定义，当前有 OP_VIEW, OP_COPY_IN, OP_RESHAPE，并且大概率为级联
                或者图上op的输出数量超过了opcode.cpp中的定义，目前仅有OP_REDUCE_ACC，已有特殊处理
                */
                MemoryType fromType = MemoryType::MEM_DEVICE_DDR;
                std::map<MemoryType, std::set<Operation *>> localTobeMap = inserter.GetRequiredTobe(o);
                if (localTobeMap.size() == 1 && localTobeMap.begin()->first != MemoryType::MEM_UNKNOWN) {
                    fromType = localTobeMap.begin()->first;
                }
                APASS_LOG_DEBUG_F(Elements::Operation, "%s[%d] oOperand %d mem original is UNKNOWN, force setting as %s.",
                    op.GetOpcodeStr().c_str(), op.GetOpMagic(), o->magic, BriefMemoryTypeToString(fromType).c_str());
                o->SetMemoryTypeOriginal(fromType);
                for (const auto &consumerOp : o->GetConsumers()) {
                    inserter.UpdateTensorTobeMap(o, *consumerOp, fromType);
                }
            }
            inserter.UpdateTensorTobeMapUnknown(o, o->GetMemoryTypeOriginal());
        }
    }
}
void AssignMemoryType::ProcesSmallTileToLargeTile(Function &function) {
    //CASE1:处理cube级联场景小搬大
    for (auto &op : function.Operations()) {
        auto opcode = op.GetOpcode();
        if(opcode != Opcode::OP_ASSEMBLE) {
            continue;
        }
        auto oOperand = op.GetOOperands().front();
        auto iOperand = op.GetIOperands().front();
        if(iOperand->GetMemoryTypeOriginal() == MEM_L0C) {
            if (!IsDimMultiple(oOperand->GetShape(), iOperand->GetShape())){
                oOperand->SetMemoryTypeOriginal(MEM_DEVICE_DDR, true);
            }
        }
    }
}
void AssignMemoryType::ProcessLargeTileToSamllTile(Function &function) {
    //CASE2:处理cube级联产经大搬小
    for (auto &op : function.Operations()) {
        auto opcode = op.GetOpcode();
        if(opcode != Opcode::OP_VIEW) {
            continue;
        }
        auto viewOpAttribute =dynamic_cast<ViewOpAttribute *>(op.GetOpAttribute().get());
        MemoryType attrToType = viewOpAttribute->GetTo();
        if(attrToType == MEM_L1) {
            auto iOperand = op.GetIOperands().front();
            auto oOperand = op.GetOOperands().front();
            if(iOperand->GetMemoryTypeOriginal() == MEM_L0C && !IsDimMultiple(iOperand->GetShape(), oOperand->GetShape())) {
                inserter.UpdateTensorTobeMap(iOperand, op, MEM_DEVICE_DDR);
                continue;
            }
            if(iOperand->GetMemoryTypeOriginal() == MEM_UB && oOperand->shape != iOperand->shape) {
                inserter.UpdateTensorTobeMap(iOperand, op, MEM_DEVICE_DDR);
                continue;
            }
        }
    }
}
/*
    @brief 检查第一个矩阵的所有维度是否为第二个矩阵的正整数倍
    @param shape1为第一个矩阵，shape2为第二个矩阵。
    @return 如果第一个矩阵是第二个的正整数倍，则返回true；否则返回false。
*/
bool AssignMemoryType::IsDimMultiple(const Shape &shape1, const Shape &shape2) {
    if (shape1.size() != shape2.size()) {
        return false;
    }
    for (size_t i = 0; i < shape1.size(); ++i) {
        if (shape1[i] <= 0 || shape2[i] <= 0 || shape1[i] % shape2[i] != 0) {
            return false;
        }
    }
    return true;
}
} //namespace npu::tile_fwk