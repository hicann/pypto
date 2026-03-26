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
 * \file function_error.h
 * \brief PyPTO 组件前端错误码常量
 *
 */

#pragma once

#include <cstdint>

namespace npu::tile_fwk {

// =============================================================================
// 一、Function前端错误码（F2XXXX - F3XXXX)
// =============================================================================

// Frontend/Function Error
enum class FError : uint32_t {
    EINTERNAL = 0x21001U, // 内部Error
    INVALID_OPERATION = 0x21002U,     // 不允许的操作
    INVALID_TYPE = 0x21003U,     // 错误的类型
    INVALID_VAL = 0x21004U,    // 无效的值
    INVALID_PTR = 0x21005U,    // 无效的指针
    OUT_OF_RANGE = 0x21006U,    // 参数超出范围
    IS_EXIST = 0x21007U,    // 参数/操作已存在
    NOT_EXIST = 0x21008U,   // 参数/操作不存在

    // File error
    BAD_FD = 0x29001U,  // 错误的文件描述符状态
    INVALID_FILE = 0x29002U, // 无效的文件(内容)

    UNKNOWN = 0x3FFFFU
};

#ifndef FUNCTION_ASSERT
#define FUNCTION_ASSERT_SELECT(_1, _2, NAME, ...) NAME
#define FUNCTION_ASSERT_WITH_UNKNOWN(cond) ASSERT(FError::UNKNOWN, cond)
#define FUNCTION_ASSERT(...) \
    FUNCTION_ASSERT_SELECT(__VA_ARGS__, ASSERT, FUNCTION_ASSERT_WITH_UNKNOWN)(__VA_ARGS__)
#endif

} // namespace npu::tile_fwk
