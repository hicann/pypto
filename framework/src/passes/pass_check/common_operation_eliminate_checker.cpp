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
 * \file common_operation_eliminate_checker.cpp
 * \brief
 */

#include "common_operation_eliminate_checker.h"

namespace npu {
namespace tile_fwk {
Status CommonOperationEliminateChecker::DoPreCheck(Function &function) {
    ALOG_INFO_F("PreCheck for CommonOperationEliminate.");
    for (auto &op : function.Operations().DuplicatedOpList()) {
        if (op->GetOpAttribute() != nullptr) {
            size_t fromOffsetSize = -1;
            if (auto viewOpAttribute = dynamic_cast<ViewOpAttribute*>(op->GetOpAttribute().get())) {
                auto &fromOffset = viewOpAttribute->GetFromOffset();
                fromOffsetSize = fromOffset.size();
            } else if (auto copyOpAttribute = dynamic_cast<CopyOpAttribute*>(op->GetOpAttribute().get())) {
                if (copyOpAttribute->IsCopyOut()) {
                    continue;
                }
                auto [fromOffset, memType] = copyOpAttribute->GetCopyInAttr();
                (void)memType;
                fromOffsetSize = fromOffset.size();
            } else {
                continue;
            }
            auto& ioperands = op->GetIOperands();
            if (ioperands.size() != 1) {
                ALOG_ERROR_F("View or Copy_In Operation %d with not one input operand.", op->GetOpMagic());
                return FAILED;
            }
            if (ioperands.front()->offset.size() != fromOffsetSize) {
                ALOG_ERROR_F("View or Copy_In Operation %d with mismatch input offset shape.", op->GetOpMagic());
                return FAILED;
            }
        }
    }
    return SUCCESS;
}
} // namespace tile_fwk
} // namespace npu