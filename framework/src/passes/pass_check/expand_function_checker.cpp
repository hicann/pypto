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
 * \file expand_function_checker.cpp
 * \brief
 */

#include "expand_function_checker.h"
#include "passes/pass_log/pass_log.h"

#define MODULE_NAME "ExpandFunction"

namespace npu {
namespace tile_fwk {
Status ExpandFunctionChecker::DoPreCheck(Function &function) {
    APASS_LOG_INFO_F(Elements::Function, "PreCheck for ExpandFunction.");
    if (!function.OperationLoopCheck()) {
        APASS_LOG_ERROR_F(Elements::Function, "Operation Loop detected before expand function; Please validate the operation input specifications.");
        return FAILED;
    }
    return SUCCESS;
}

Status ExpandFunctionChecker::DoPostCheck(Function &function) {
    APASS_LOG_INFO_F(Elements::Function, "PostCheck for ExpandFunction.");
    if (function.expandFunctionAccelerate != false) {
        APASS_LOG_ERROR_F(Elements::Function, "ExpandFunctionAccelerate should equal to false after ExpandFunction process.");
        return FAILED;
    }
    if (!function.OperationLoopCheck()) {
        APASS_LOG_ERROR_F(Elements::Function, "Operation Loop detected after expand function; Please review the error messages generated during the processing procedure.");
        return FAILED;
    }
    if (CheckDynAttrForView(function) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Function, "CheckDynAttrForView failed.");
        return FAILED;
    }
    if (CheckToDynOffsetForAssemble(function) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Function, "CheckToDynOffsetForAssemble failed.");
        return FAILED;
    }
    return SUCCESS;
}
} // namespace tile_fwk
} // namespace npu