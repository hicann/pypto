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
 * \file assign_memory_type_checker.cpp
 * \brief
 */

#include "assign_memory_type_checker.h"
#include "passes/pass_log/pass_log.h"

#define MODULE_NAME "AssignMemoryType"

namespace npu {
namespace tile_fwk {
Status AssignMemoryTypeChecker::DoPreCheck(Function &function) {
    APASS_LOG_INFO_F(Elements::Function, "===> Start Precheck for AssignMemoryType.");
    auto operations = function.Operations();
    for(auto &operation : operations){
        Operation *op_ptr = &operation;
        //A_MUL_B操作输入tensor生产者校验
        if(operation.GetOpcode() == Opcode::OP_A_MUL_B) {
            Status status = CheckAmulBInputProducers(operation);
            if(status != SUCCESS) {
                return status;
            }
        }
        //创建队列，包含当前操作和嵌套深度
        std::queue<std::pair<Operation *, int>> opQueue;
        std::unordered_set<Operation *> visited;

        opQueue.emplace(op_ptr, 1);
        visited.insert(op_ptr);

        while(!opQueue.empty()){
            auto[currentOp,depth] = opQueue.front();
            opQueue.pop();

            //嵌套深度达到3失败
            if(depth > 3){
                APASS_LOG_WARN_F(Elements::Operation, "Over three view/assemble/reshape operations in sequence, currently reched %d; "
                    "Potential suboptimal allocation around operation %d.", depth, currentOp->GetOpMagic());
                return SUCCESS;
            }
            CheckPattern(currentOp,opQueue,depth,visited);
        }
    }
    return SUCCESS;
}

Status AssignMemoryTypeChecker::CheckAmulBInputProducers(Operation &operation) {
    auto inputs = operation.GetIOperands();
    auto producerOps = operation.ProducerOps();
    for(auto &producerOp : producerOps) {
        auto producerOpcode = producerOp->GetOpcode();
        if(producerOpcode != Opcode::OP_L1_TO_L0A && producerOpcode !=Opcode::OP_L1_TO_L0B && 
           producerOpcode != Opcode::OP_L1_TO_L0_AT && producerOpcode !=Opcode::OP_L1_TO_L0_BT && 
           producerOpcode != Opcode::OP_VIEW && producerOpcode !=Opcode::OP_VEC_DUP) {
            APASS_LOG_ERROR_F(Elements::Operation, "Memory error, %s[%d] has invalid input producer; "
                "Please check input producer %s[%d]. %s", operation.GetOpcodeStr().c_str(), operation.GetOpMagic(),
                producerOp->GetOpcodeStr().c_str(), producerOp->GetOpMagic(), GetFormatBacktrace(operation).c_str());
            return FAILED;
           }
        if(producerOpcode == Opcode::OP_VIEW) {
            auto viewOpAttribute = dynamic_cast<ViewOpAttribute *>(producerOp->GetOpAttribute().get());
            MemoryType attrToType = viewOpAttribute->GetTo();
            if (attrToType != MemoryType::MEM_BT && attrToType != MemoryType::MEM_FIX_QUANT_PRE &&
                attrToType != MemoryType::MEM_L0A && attrToType != MemoryType::MEM_L0B &&
                attrToType != MemoryType::MEM_UNKNOWN) {
                APASS_LOG_ERROR_F(Elements::Operation, "View attribute error, %s[%d] has invalid input OP_VIEW(toType: %d); "
                    "Please check input view %s[%d]. %s", operation.GetOpcodeStr().c_str(), operation.GetOpMagic(), attrToType,
                    producerOp->GetOpcodeStr().c_str(), producerOp->GetOpMagic(), GetFormatBacktrace(operation).c_str());
                return FAILED;
            }
        }
    }
    return SUCCESS;
}

//检查view/assemble/reshape的嵌套深度是否大于等于3
void AssignMemoryTypeChecker::CheckPattern(Operation *operation, std::queue<std::pair<Operation *, int>> &opQueue,
                                            int depth, std::unordered_set<Operation *> &visited) {
    for(auto &tensor : operation->oOperand){
        for(auto &consumerOp : tensor-> GetConsumers()){
            if(consumerOp->GetOpcode() == Opcode::OP_VIEW ||
                consumerOp->GetOpcode() == Opcode::OP_ASSEMBLE ||
                consumerOp->GetOpcode() == Opcode::OP_RESHAPE){
                Operation* consumerOpPtr = consumerOp;
                if(visited.find(consumerOpPtr) == visited.end()){
                    opQueue.emplace(consumerOpPtr, depth + 1);
                    visited.insert(consumerOpPtr);
                    break;
                }
            }
        }
    }
}
} // namespace tile_fwk
} // namespace npu