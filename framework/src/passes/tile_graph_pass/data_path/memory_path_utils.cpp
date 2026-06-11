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
 * \file memory_path_utils.cpp
 * \brief
 */

#include "memory_path_utils.h"

namespace npu::tile_fwk {

bool MemoryPathUtils::IsSpecialDirectMemoryPath(MemoryType from, MemoryType to)
{
    return (from == MemoryType::MEM_L0C && to == MemoryType::MEM_L1) ||
           (from == MemoryType::MEM_L0C && to == MemoryType::MEM_UB) ||
           (from == MemoryType::MEM_UB && to == MemoryType::MEM_L1);
}

bool MemoryPathUtils::IsDifferentKnownRequirement(MemoryType requirement, MemoryType targetType)
{
    return requirement != MemoryType::MEM_UNKNOWN && requirement != targetType;
}

bool MemoryPathUtils::ShouldUseDdrForSpecialPath(
    bool hasParallelDifferentRequirement, MemoryType from, MemoryType to)
{
    return hasParallelDifferentRequirement && IsSpecialDirectMemoryPath(from, to);
}

} // namespace npu::tile_fwk
