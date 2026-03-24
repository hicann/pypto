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
 * \file conv_error.h
 * \brief CONV 组件错误分类、场景枚举与错误码常量。
 *        - FC6XXXX-FC8XXX：CONV 内部已梳理报错。
 */

#pragma once
#include <cstdint>
#include "tilefwk/error.h"

namespace npu::tile_fwk {

enum class ConvOperationError : uint32_t {
    // FC61xx: Operation非法拦截类报错
    INPUT_INVALID         = 0xC6101U,
    OVER_BUFFER_LIMIT     = 0xC6102U,
    UNKNOWN               = 0xC6199U
};

enum class ConvExpandFuncError : uint32_t {
    EXPANDFUNC_TENSOR_OP_NULLPTR          = 0xC6201U,
    EXPANDFUNC_TENSOR_ATTR_GET_FAILED     = 0xC6202U,
    EXPANDFUNC_TILE_OP_NULLPTR            = 0xC6203U,
    EXPANDFUNC_PARAMS_INVALID             = 0xC6204U,
    EXPANDFUNC_INNER_STATUS_FAILED        = 0xC6205U,
    UNKNOWN                               = 0xC6299U
};

enum class ConvCodenGenError : uint32_t {
    CODEGEN_GET_ATTR_FAILED               = 0xC6301U,
    CODEGEN_CHECK_ATTR_INVALID            = 0xC6302U,
    CODEGEN_CHECK_DIM_INVALID             = 0xC6303U,
    UNKNOWN                               = 0xC6399U
};

enum class ConvTileOpError : uint32_t {
    TILEOP_TENSOR_FORMAT_FAILED         = 0xC6401U,
    TILEOP_SHAPE_SIZE_FAILED            = 0xC6402U,
    TILEOP_STC_SHAPE_INVALID            = 0xC6403U,
    TILEOP_INDEX_INVALID                = 0xC6404U,
    UNKNOWN                             = 0xC6499U
};

}  // namespace npu::tile_fwk
