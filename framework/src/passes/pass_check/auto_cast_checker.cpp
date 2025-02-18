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
 * \file auto_cast_checker.cpp
 * \brief
 */

#include "auto_cast_checker.h"

namespace npu {
namespace tile_fwk {
Status AutoCastChecker::DoPreCheck(Function &function) {
    ALOG_INFO_F("PreCheck for AutoCast");
    std::vector<Operation *> opList = function.Operations().DuplicatedOpList();
    for (size_t opIdx = 0; opIdx < opList.size(); opIdx++) {
        Operation *op = opList[opIdx];
        if (std::find(CAST_OPS.begin(), CAST_OPS.end(), op->GetOpcode()) == CAST_OPS.end()) {
            continue;
        }
        if (op->GetIOperands().size() != 1) {
            ALOG_ERROR_F("CAST op %d has %d input tensor, which should be 1.",
                         op->GetOpMagic(), static_cast<int>(op->GetIOperands().size()));
            return FAILED;
        }
        if (op->GetOOperands().size() != 1) {
            ALOG_ERROR_F("CAST op %d has %d output tensor, which should be 1.",
                         op->GetOpMagic(), static_cast<int>(op->GetOOperands().size()));
            return FAILED;
        }
    }
    return SUCCESS;
}

Status AutoCastChecker::DoPostCheck(Function &function) {
    ALOG_INFO_F("PostCheck for AutoCast");
    std::vector<Operation *> opList = function.Operations().DuplicatedOpList();
    for (size_t opIdx = 0; opIdx < opList.size(); opIdx++) {
        Operation *op = opList[opIdx];
        if (SupportBF16(op)) {
            continue;
        }
        auto iOperands = op->GetIOperands();
        for (auto &iop : iOperands) {
            if (iop->Datatype() == DataType::DT_BF16) {
                ALOG_ERROR_F("Exist unsupported BF16 compute between op %d and tensor %d",
                             op->GetOpMagic(), iop->GetMagic());
                return FAILED;
            }
        }
        auto oOperands = op->GetOOperands();
        for (auto &oop : oOperands) {
            if (oop->Datatype() == DataType::DT_BF16) {
                ALOG_ERROR_F("Exist unsupported BF16 compute between op %d and tensor %d",
                             op->GetOpMagic(), oop->GetMagic());
                return FAILED;
            }
        }
    }
    return SUCCESS;
}

bool AutoCastChecker::SupportBF16(Operation *op) {
    if (UNSUPPORT_BF16_OPS.count(op->GetOpcode()) > 0) {
        return false;
    }
    return true;
}
} // namespace tile_fwk
} // namespace npu