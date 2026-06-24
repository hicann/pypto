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
 * \file memory_path_utils.h
 * \brief
 */

#ifndef TILE_FWK_MEMORY_PATH_UTILS_H
#define TILE_FWK_MEMORY_PATH_UTILS_H

#include "tilefwk/tilefwk.h"

namespace npu::tile_fwk {

class MemoryPathUtils {
public:
    static bool IsSpecialDirectMemoryPath(MemoryType from, MemoryType to);

    static bool IsDifferentKnownRequirement(MemoryType requirement, MemoryType targetType);

    static bool ShouldUseDdrForSpecialPath(bool hasParallelDifferentRequirement, MemoryType from, MemoryType to);
};

} // namespace npu::tile_fwk

#endif // TILE_FWK_MEMORY_PATH_UTILS_H
