/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INTERFACE_OPERATION_VECTOR_GATHER_MASK_COMMON_H
#define INTERFACE_OPERATION_VECTOR_GATHER_MASK_COMMON_H

#include <cstdint>

namespace npu::tile_fwk {

enum class GatherMaskPattern : uint8_t {
    P0101 = 1,
    P1010 = 2,
    P0001 = 3,
    P0010 = 4,
    P0100 = 5,
    P1000 = 6,
    P1111 = 7,
};

constexpr bool IsHalfGatherMaskPattern(GatherMaskPattern pattern)
{
    return pattern == GatherMaskPattern::P0101 || pattern == GatherMaskPattern::P1010;
}

constexpr bool IsQuarterGatherMaskPattern(GatherMaskPattern pattern)
{
    return pattern == GatherMaskPattern::P0001 || pattern == GatherMaskPattern::P0010 ||
           pattern == GatherMaskPattern::P0100 || pattern == GatherMaskPattern::P1000;
}

} // namespace npu::tile_fwk

#endif // INTERFACE_OPERATION_VECTOR_GATHER_MASK_COMMON_H
