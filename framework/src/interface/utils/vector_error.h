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
 * \file vector_error.h
 * \brief VECTOR 组件错误分类、场景枚举与错误码常量。
 *        - FC0XXX-FC2XX：VECTOR
 */

#pragma once
#include <cstdint>

namespace npu::tile_fwk {

enum class VectorErrorCode : uint32_t {
    
    // FC0xxx: 参数错误
    ERR_PARAM_INVALID             = 0xC0000U,  // 参数无效（shape、dtype、format等）
    ERR_PARAM_DTYPE_UNSUPPORTED   = 0xC0001U,  // 不支持的数据类型
    
    // FC1xxx: 配置错误
    ERR_CONFIG_TILE         = 0xC1000U,  // Tile 配置错误
    ERR_CONFIG_ALIGNMENT    = 0xC1001U,  // 对齐错误（16B/32B/64元素）
    
    // FC2xxx: 运行时错误
    ERR_RUNTIME_NULLPTR     = 0xC2000U,  // 空指针错误
    ERR_RUNTIME_LOGIC       = 0xC2001U,  // 逻辑不变量错误
};
}  // namespace npu::tile_fwk
