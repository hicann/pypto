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
 * \file codegen_error.h
 * \brief CODEGEN 组件错误分类、场景枚举与错误码常量。
 *        - F6XXXX：CODEGEN 内部已梳理报错（大流程 Category -> 子流程 Scene）。
 */

#pragma once

#include <cstdint>
#include <type_traits>

#include "interface/utils/common.h"

namespace npu::tile_fwk {

// =============================================================================
// 一、大流程：CodeGenErrorCategory（对应 F6Cxxx 中的 C）
// =============================================================================

enum class CodeGenErrorCategory {
    FRAMEWORK = 60000U,         // 0: 框架错误
    OPERATION_ADAPTER = 61000U, // 1: OP初始化错误
    GEN_OP_CODE = 62000U,       // 2: 生成OP代码错误
    COMPILE_CODE = 63000U,       // 3: 编译CCE代码错误
};

// =============================================================================
// 二、子流程：各 Category 下的 Error 枚举（枚举值即错误码 F6xxxx）
// =============================================================================

// Framework error scene
enum class FwkErr : uint32_t {
    PLATFORM_NOT_SUPPORTED = ToUnderlying(CodeGenErrorCategory::FRAMEWORK) + 1U, // 1: 不支持的平台
    INVALID_FUNCTION = ToUnderlying(CodeGenErrorCategory::FRAMEWORK) + 2U,       // 2: 无效的函数
};

// Operation adapter error scene
enum class OperErr : uint32_t {
    ATTRIBUTE_INVALID = ToUnderlying(CodeGenErrorCategory::OPERATION_ADAPTER) + 1U,
    TENSOR_DIM_EXCEEDED = ToUnderlying(CodeGenErrorCategory::OPERATION_ADAPTER) + 2U,
    OPERAND_COUNT_EXCEEDED = ToUnderlying(CodeGenErrorCategory::OPERATION_ADAPTER) + 3U,
    OPERAND_COUNT_NOT_MATCHED = ToUnderlying(CodeGenErrorCategory::OPERATION_ADAPTER) + 4U,
    OPERATION_INIT_FAILED = ToUnderlying(CodeGenErrorCategory::OPERATION_ADAPTER) + 5U,
    OPERAND_TYPE_UNSUPPORTED = ToUnderlying(CodeGenErrorCategory::OPERATION_ADAPTER) + 6U,
};

// Generate operation code error scene
enum class GenCodeErr : uint32_t {
    GEN_OP_CODE_FAILED = ToUnderlying(CodeGenErrorCategory::GEN_OP_CODE) + 1U,
    OP_CODE_UNSUPPORTED = ToUnderlying(CodeGenErrorCategory::GEN_OP_CODE) + 2U,
    PRINT_FAILED = ToUnderlying(CodeGenErrorCategory::GEN_OP_CODE) + 3U,
    PRINT_MODE_ERROR = ToUnderlying(CodeGenErrorCategory::GEN_OP_CODE) + 4U,
    DATA_TYPE_MISMATCHED = ToUnderlying(CodeGenErrorCategory::GEN_OP_CODE) + 5U,
    DATA_TYPE_UNSUPPORTED = ToUnderlying(CodeGenErrorCategory::GEN_OP_CODE) + 6U,
    TENSOR_SHAPE_INVALID = ToUnderlying(CodeGenErrorCategory::GEN_OP_CODE) + 7U,
    TENSOR_SHAPE_MISMATCHED = ToUnderlying(CodeGenErrorCategory::GEN_OP_CODE) + 8U,
    TENSOR_DIM_UNSUPPORTED = ToUnderlying(CodeGenErrorCategory::GEN_OP_CODE) + 9U,
    TENSOR_OFFSET_INVALID = ToUnderlying(CodeGenErrorCategory::GEN_OP_CODE) + 10U,
    TENSOR_MAGIC_CONFLICT = ToUnderlying(CodeGenErrorCategory::GEN_OP_CODE) + 11U,
    PARAM_IDX_INVALID = ToUnderlying(CodeGenErrorCategory::GEN_OP_CODE) + 12U,
    TENSOR_NOT_FOUND = ToUnderlying(CodeGenErrorCategory::GEN_OP_CODE) + 13U,
    SYMBOL_NOT_FOUND = ToUnderlying(CodeGenErrorCategory::GEN_OP_CODE) + 14U,
    PIPE_ID_NOT_FOUND = ToUnderlying(CodeGenErrorCategory::GEN_OP_CODE) + 15U,
};

// Compile CCE error scene
enum class CmpCodeErr : uint32_t {
    COMPILE_CODE_FAILED = ToUnderlying(CodeGenErrorCategory::COMPILE_CODE) + 1U,
    INCLUDE_FILE_NOT_FOUND = ToUnderlying(CodeGenErrorCategory::COMPILE_CODE) + 2U,
    PTO_ISA_NOT_FOUND = ToUnderlying(CodeGenErrorCategory::COMPILE_CODE) + 3U,
    CMD_CHECK_FAILED = ToUnderlying(CodeGenErrorCategory::COMPILE_CODE) + 4U,
    FILE_IO_FAILED = ToUnderlying(CodeGenErrorCategory::COMPILE_CODE) + 5U,
};

} // namespace npu::tile_fwk
