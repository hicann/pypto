/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file split_large_fanout_tensor_checker.cpp
 * \brief
 */

#include "split_large_fanout_tensor_checker.h"
#include "passes/pass_log/pass_log.h"

#define MODULE_NAME "SplitLargeFanoutTensor"

namespace npu {
namespace tile_fwk {
Status SplitLargeFanoutTensorChecker::DoPreCheck(Function &function) {
    APASS_LOG_INFO_F(Elements::Function, "===> Start Precheck for SplitLargeFanoutTensor.");
    if (CheckAssembleOverlap(function) == FAILED) {
        APASS_LOG_ERROR_F(Elements::Function, "Precheck of SplitLargeFanoutTensor failed since overlaps of assemble inputs.");
        return FAILED;
    }
    APASS_LOG_INFO_F(Elements::Function, "End Precheck for SplitLargeFanoutTensor.");
    return SUCCESS;
}

Status SplitLargeFanoutTensorChecker::DoPostCheck(Function &function) {
    APASS_LOG_INFO_F(Elements::Function, "===> Start PostCheck for SplitLargeFanoutTensor.");
    if (CheckAssembleOverlap(function) == FAILED) {
        APASS_LOG_ERROR_F(Elements::Function, "PostCheck of SplitLargeFanoutTensor failed since overlaps of assemble inputs.");
        return FAILED;
    }
    APASS_LOG_INFO_F(Elements::Function, "End PostCheck for SplitLargeFanoutTensor.");
    return SUCCESS;
}
} // namespace tile_fwk
} // namespace npu