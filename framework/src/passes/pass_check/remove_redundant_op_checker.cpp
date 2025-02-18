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
 * \file remove_redundant_op_checker.cpp
 * \brief
 */

#include "remove_redundant_op_checker.h"
#include "passes/pass_log/pass_log.h"

#define MODULE_NAME "RemoveRedundantOp"

namespace npu{
namespace tile_fwk {
Status RemoveRedundantOpChecker::PreCheckAssemble(const Operation &op, const LogicalTensorPtr &in) {
    uint32_t assembleRemoveNum = 0;
    uint32_t otherOpNum = 0;
    for (const auto &childOp : in->GetConsumers()) {
        if (childOp->GetOpcode() != Opcode::OP_ASSEMBLE) {
            ++otherOpNum;
            continue;
        }
        auto child_in = op.iOperand.front();
        auto child_out = op.oOperand.front();
        if (!child_out->GetConsumers().empty()) {
            ++otherOpNum;
            continue;
        }
        if (child_in->shape == child_out->shape &&
            child_in->GetMemoryTypeOriginal() == MemoryType::MEM_DEVICE_DDR &&
            child_out->GetMemoryTypeOriginal() == MemoryType::MEM_DEVICE_DDR) {
            ++assembleRemoveNum;
        }
    }
    if (assembleRemoveNum > 1 && otherOpNum > 0) {
        APASS_LOG_ERROR_F(Elements::Operation, 
        "More than one assemble ddr op without consumer; Please check the num of assemble ddr op without consumer.");
        return FAILED;
    }
    return SUCCESS;
}

Status RemoveRedundantOpChecker::PreCheckView(Function &function, const Operation &op, const LogicalTensorPtr &in) {
    auto out = op.oOperand.front();
    if (in->shape == out->shape && op.ConsumerOps().empty() && in->GetConsumers().size() > 1 && function.IsFromOutCast(out)) {
        APASS_LOG_ERROR_F(Elements::Operation, 
        "There is another op consumes the input of a view op[%d] (the output is an outcast) without consumer; Please check view op[%d].%s", 
        op.GetOpMagic(), op.GetOpMagic(), GetFormatBacktrace(op).c_str());
        return FAILED;
    }
    if (function.IsFromOutCast(out)) {
        APASS_LOG_ERROR_F(Elements::Operation, 
        "The output of the op[%d] is an outCast; Please check the type of output from the op[%d].%s", 
        op.GetOpMagic(), op.GetOpMagic(), GetFormatBacktrace(op).c_str());
        return FAILED;
    }
    return SUCCESS;
}

Status RemoveRedundantOpChecker::ProcessPreCheck(Function &function, const Operation &op) {
    if (op.GetOpcode() == Opcode::OP_ASSEMBLE) {
        auto assemble_in = op.iOperand.front();
        if (PreCheckAssemble(op, assemble_in) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "PreCheck for assemble op[%d] failed.%s", op.GetOpMagic(), GetFormatBacktrace(op).c_str());
            return FAILED;
        }
        return SUCCESS;
    }
    if (op.GetOpcode() == Opcode::OP_VIEW) {
        auto view_in = op.iOperand.front();
        if (PreCheckView(function, op, view_in) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "PreCheck for view op[%d] failed.%s", op.GetOpMagic(), GetFormatBacktrace(op).c_str());
            return FAILED;
        }  
    }
    return SUCCESS;
}

Status RemoveRedundantOpChecker::PostCheckAssemble(const Operation &op) {
    auto assemble_in = op.iOperand.front();
    auto assemble_out = op.oOperand.front();
    auto parentOp = *assemble_in->GetProducers().begin();
    if (parentOp == nullptr) {
        APASS_LOG_ERROR_F(Elements::Operation, 
        "The input of assemble [%d] has no producer; Please check the input of assemble [%d] to ensure that it has producers.%s", 
        op.GetOpMagic(), op.GetOpMagic(), GetFormatBacktrace(op).c_str());
        return FAILED;
    }
    if (assemble_in->shape == assemble_out->shape) {
        if (assemble_in->GetMemoryTypeOriginal() == MemoryType::MEM_UB &&
            assemble_out->GetMemoryTypeOriginal() == MemoryType::MEM_UB) {
            APASS_LOG_ERROR_F(Elements::Operation, 
            "PostCheck for assembleUB (both input and output are memorytype UB) op[%d] failed. Input and output has the same shape; Please check assembleUB op[%d].%s", 
            op.GetOpMagic(), op.GetOpMagic(), GetFormatBacktrace(op).c_str());
            return FAILED;
        }
    }
    return SUCCESS;
}

Status RemoveRedundantOpChecker::PostCheckView(const Operation &op) {
    auto view_in = op.iOperand.front();
    auto view_out = op.oOperand.front();
    auto viewOpAttribute = dynamic_cast<ViewOpAttribute *>(op.GetOpAttribute().get());
    if (viewOpAttribute != nullptr && viewOpAttribute->GetToDynValidShape().empty() &&
        view_in->shape == view_out->shape && view_in->GetMemoryTypeOriginal() == view_out->GetMemoryTypeOriginal()) {
        APASS_LOG_ERROR_F(Elements::Operation, 
        "PostCheck for view op[%d] failed: DynValidShape empty, same shape and memory type; Please check view op[%d].%s", 
        op.GetOpMagic(), op.GetOpMagic(), GetFormatBacktrace(op).c_str());
        return FAILED;
    }
    if (view_out->GetConsumers().size() != 1) {
        return SUCCESS;
    }
    auto childOp = *(view_out->GetConsumers().begin());
    if (childOp == nullptr) {
        APASS_LOG_ERROR_F(Elements::Operation, 
        "Found null childOp of op[%d]; Please check if the output consumers of the op[%d] are empty.%s", 
        op.GetOpMagic(), op.GetOpMagic(), GetFormatBacktrace(op).c_str());
        return FAILED;
    }
    if (childOp->GetOpcode() == Opcode::OP_COMM_WAIT_FLAG) {
        APASS_LOG_ERROR_F(Elements::Operation, "View op[%d] has only one commit wait child; Please check view op[%d].%s", 
        op.GetOpMagic(), op.GetOpMagic(), GetFormatBacktrace(op).c_str());
        return FAILED;
    }
    return SUCCESS;
}

Status RemoveRedundantOpChecker::PostCheckRegCopy(const Operation &op) {
    auto regcopy_in = op.iOperand.front();
    auto regcopy_out = op.oOperand.front();
    if (regcopy_in->shape == regcopy_out->shape && regcopy_in->GetMemoryTypeOriginal() == regcopy_out->GetMemoryTypeOriginal()) {
        APASS_LOG_ERROR_F(Elements::Operation, 
        "PostCheck for regcopy op[%d] failed: the shape of input equals to the output; Please check regcopy op[%d].%s", 
        op.GetOpMagic(), op.GetOpMagic(), GetFormatBacktrace(op).c_str());
        return FAILED;
    }
    return SUCCESS;
}

Status RemoveRedundantOpChecker::PostCheckCopyIn(const Operation &op) {
    auto copy_in = op.iOperand.front();
    auto copy_out = op.oOperand.front();
    if (copy_in->shape == copy_out->shape && copy_out->GetMemoryTypeOriginal() == npu::tile_fwk::MEM_L1) {
        bool isRedundant = true;
        for (const auto &producerOp : op.ProducerOps()) {
            if (producerOp == nullptr) {
                APASS_LOG_ERROR_F(Elements::Operation, 
                "Found null producer of op[%d]; Please check the producers of op[%d].%s", op.GetOpMagic(), op.GetOpMagic(), GetFormatBacktrace(op).c_str());
                return FAILED;
            }
            if (producerOp->GetOpcode() != Opcode::OP_VIEW) {
                isRedundant = false;
                break;
            }
        }
        if (isRedundant) {
            APASS_LOG_ERROR_F(Elements::Operation, 
            "PostCheck for copyin op[%d] failed: the producers of the op are view; Please check copyin op[%d].%s", 
            op.GetOpMagic(), op.GetOpMagic(), GetFormatBacktrace(op).c_str());
            return FAILED;
        }
    }
    return SUCCESS;
}

Status RemoveRedundantOpChecker::PostCheckExpand(const Operation &op) {
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
    if (equalShape && equalDynValidShape) {
        APASS_LOG_ERROR_F(Elements::Operation, 
        "PostCheck for expand op[%d] failed: the shape of input equals to the output; Please check expand op[%d].%s", 
        op.GetOpMagic(), op.GetOpMagic(), GetFormatBacktrace(op).c_str());
        return FAILED;
    }
    return SUCCESS;
}

Status RemoveRedundantOpChecker::ProcessPostCheck(const Operation &op) {
    if (op.GetOpcode() == Opcode::OP_ASSEMBLE) {
        if (PostCheckAssemble(op) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "PostCheck for Assemble failed.%s", GetFormatBacktrace(op).c_str());
            return FAILED;
        }
        return SUCCESS;
    }
    if (op.GetOpcode() == Opcode::OP_VIEW) {
        if (PostCheckView(op) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "PostCheck for View failed.%s", GetFormatBacktrace(op).c_str());
            return FAILED;
        }
        return SUCCESS;
    }
    if (op.GetOpcode() == Opcode::OP_REGISTER_COPY) {
        if (PostCheckRegCopy(op) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "PostCheck for RegCopy failed.%s", GetFormatBacktrace(op).c_str());
            return FAILED;
        }
        return SUCCESS;
    }
    if (op.GetOpcode() == Opcode::OP_COPY_IN) {
        if (PostCheckCopyIn(op) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "PostCheck for CopyIn failed.%s", GetFormatBacktrace(op).c_str());
            return FAILED;
        }
        return SUCCESS;
    }
    if (op.GetOpcode() == Opcode::OP_EXPAND) {
        if (PostCheckExpand(op) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "PostCheck for Expand failed.%s", GetFormatBacktrace(op).c_str());
            return FAILED;
        }
    }
    return SUCCESS;
}

Status RemoveRedundantOpChecker::DoPreCheck(Function &function) {
    APASS_LOG_INFO_F(Elements::Operation, "PreCheck for RemoveRedundantOp");
    if (CheckValidOp(function) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Function, "Found invalid op in the function.");
        return FAILED;
    }
    if (CheckOpIOValid(function) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Function, "Found invalid input/output from the function.");
        return FAILED;
    }
    for (const auto &op : function.Operations()) {
        if (ProcessPreCheck(function, op) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "PreCheck for RemoveRedundantOp failed.%s", GetFormatBacktrace(op).c_str());
            return FAILED;
        }
    }
    return SUCCESS;
}

Status RemoveRedundantOpChecker::DoPostCheck(Function &function) {
    APASS_LOG_INFO_F(Elements::Operation, "PostCheck for RemoveRedundantOp");
    if (CheckOpIOValid(function) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Function, "Found invalid input or output in the function.");
        return FAILED;
    }
    for (const auto &op : function.Operations()) {
        if (ProcessPostCheck(op) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "PostCheck for RemoveRedundantOp failed.%s", GetFormatBacktrace(op).c_str());
            return FAILED;
        }
    }
    return SUCCESS;
}
} // namespace tile_fwk
} // namespace npu