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

namespace npu{
namespace tile_fwk {
Status Checker::DoPreCheck(Function &function) {
    (void)function;
    return SUCCESS;
}

Status Checker::DoPostCheck(Function &function) {
    (void)function;
    return SUCCESS;
}

Status Checker::CheckConsumerProducer(const LogicalTensorPtr &tensor) {
    for (const auto &producer : tensor->GetProducers()) {
        if (producer == nullptr) {
            ALOG_ERROR_F("Found null producer in tensor.");
            return FAILED;
        }
    }
    for (const auto &consumer : tensor->GetConsumers()) {
        if (consumer == nullptr) {
            ALOG_ERROR_F("Found null consumer in tensor.");
            return FAILED;
        }
    }
    return SUCCESS;
}

Status Checker::CheckValidOp(Function &function) {
    for (const auto &op : function.Operations().DuplicatedOpList()) {
        if (op == nullptr) {
            ALOG_ERROR_F("Found null op in function.Operations().");
            return FAILED;
        }
    }
    return SUCCESS;
}

Status Checker::CheckOpIOValid(Function &function) {
    for (const auto &op : function.Operations().DuplicatedOpList()) {
        for (const auto &input : op->iOperand) {
            if (input == nullptr) {
                ALOG_ERROR_F("The input of op[%d] is null", op->opmagic);
                return FAILED;
            }
            if (CheckConsumerProducer(input) != SUCCESS) {
                ALOG_ERROR_F("CheckConsumerProducer for op[%d]'s input failed!", op->opmagic);
                return FAILED;
            }
        }
        for (const auto &output : op->oOperand) {
            if (output == nullptr) {
                ALOG_ERROR_F("The output of op[%d] is null", op->opmagic);
                return FAILED;
            }
            if (CheckConsumerProducer(output) != SUCCESS) {
                ALOG_ERROR_F("CheckConsumerProducer for op[%d]'s output failed!", op->opmagic);
                return FAILED;
            }
        }
    }
    return SUCCESS;
}

Status Checker::CheckCompleteness(Function &function) {
    if (function.GetIncast().empty()) {
        ALOG_ERROR_F("The incast of function[%d] is empty.", function.GetFuncMagic());
        return FAILED;
    }
    for (const auto &incast : function.GetIncast()) {
        if (incast == nullptr) {
            ALOG_ERROR_F("The function[%d] contains incast which is null.", function.GetFuncMagic());
            return FAILED;
        }
        if (incast->GetConsumers().empty()) {
            ALOG_WARN_F("The incast[%d] has no consumer.", incast->GetMagic());
            continue;
        }
    }
    if (function.GetOutcast().empty()) {
        ALOG_ERROR_F("The outcast of function[%d] is empty.", function.GetFuncMagic());
        return FAILED;
    }
    for (const auto &outcast : function.GetOutcast()) {
        if (outcast == nullptr) {
            ALOG_ERROR_F("The function[%d] contains outcast which is null.", function.GetFuncMagic());
            return FAILED;
        }
        if (outcast->GetProducers().empty()) {
            ALOG_WARN_F("The outcast[%d] has no producer.", outcast->GetMagic());
            continue;
        }
    }
    return SUCCESS;
}

Status Checker::CheckGraphLoop(Function &function) {
    if (function.GetTotalSubGraphCount() == 0 && !function.OperationLoopCheck()) {
        ALOG_ERROR_F("OperationLoopCheck failed, there is a loop in function[%d].", function.GetFuncMagic());
        return FAILED;
    }
    if (!function.LoopCheck().empty()) {
        ALOG_ERROR_F("Loopcheck failed, there is a loop in function[%d].", function.GetFuncMagic());
        return FAILED;
    }
    return SUCCESS;
}

Status Checker::PublicCheck(Function &function) {
    if (CheckValidOp(function) != SUCCESS) {
        ALOG_ERROR_F("CheckValidOp for function[%d] failed!", function.GetFuncMagic());
        return FAILED;
    }
    if (CheckOpIOValid(function) != SUCCESS) {
        ALOG_ERROR_F("CheckOpIOValid for function[%d] failed!", function.GetFuncMagic());
        return FAILED;
    }
    if (CheckCompleteness(function) != SUCCESS) {
        ALOG_ERROR_F("CheckCompleteness for function[%d] failed!", function.GetFuncMagic());
        return FAILED;
    }
    if (CheckGraphLoop(function) != SUCCESS) {
        ALOG_ERROR_F("CheckGraphLoop for function[%d] failed!", function.GetFuncMagic());
        return FAILED;
    }
    return SUCCESS;
}

inline std::unordered_set<Operation *> GetNeedCheckOps(Function &function, Opcode opcode) {
    std::unordered_set<Operation *> needCheckOps;
    for (const auto &incast : function.GetIncast()) {
        for (auto &consumer : incast->GetConsumers()) {
            if (consumer->GetOpcode() == opcode) {
                needCheckOps.insert(consumer);
            }
        }
    }
    for (const auto &outcast : function.GetOutcast()) {
        for (auto &producer : outcast->GetProducers()) {
            if (producer->GetOpcode() == opcode) {
                needCheckOps.insert(producer);
            }
        }
    }
    return needCheckOps;
}

Status Checker::CheckDynAttrForView(Function &function) {
    std::unordered_set<Operation *> needCheckViewOps = GetNeedCheckOps(function, Opcode::OP_VIEW);
    for (const auto &op : needCheckViewOps) {
        auto viewAttr = std::static_pointer_cast<ViewOpAttribute>(op->GetOpAttribute());
        std::vector<SymbolicScalar> &viewFromDynOffset = viewAttr->GetFromDynOffset();
        if (viewFromDynOffset.empty()) {
            ALOG_ERROR_F("CheckDynAttrForView failed, fromDynOffset_ of op[%d] in function[%d] is empty.", op->GetOpMagic(), function.GetFuncMagic());
            return FAILED;
        }
        std::vector<SymbolicScalar> &viewToDynValidShape = viewAttr->GetToDynValidShape();
        if (viewToDynValidShape.empty()) {
            ALOG_ERROR_F("CheckDynAttrForView failed, toDynValidShape_ of op[%d] in function[%d] is empty.", op->GetOpMagic(), function.GetFuncMagic());
            return FAILED;
        }
    }
    return SUCCESS;
}

Status Checker::CheckToDynOffsetForAssemble(Function &function) {
    std::unordered_set<Operation *> needCheckAssembleOps = GetNeedCheckOps(function, Opcode::OP_ASSEMBLE);
    for (const auto &op : needCheckAssembleOps) {
        auto assembleAttr = std::static_pointer_cast<AssembleOpAttribute>(op->GetOpAttribute());
        std::vector<SymbolicScalar> &assembleToDynOffset = assembleAttr->GetToDynOffset();
        if (assembleToDynOffset.empty()) {
            ALOG_ERROR_F("CheckToDynOffsetForAssemble failed, toDynOffset_ of op[%d] in function[%d] is empty.", op->GetOpMagic(), function.GetFuncMagic());
            return FAILED;
        }
    }
    return SUCCESS;
}
} // namespace tile_fwk
} // namespace npu