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
 * \file mem_path_utils.h
 * \brief Queries over the platform memory-path graph: which memType can move to which.
 */

#pragma once

#include "tilefwk/data_type.h"

namespace npu::tile_fwk {

/**
 * @brief Spill-side wrapper over the platform memory-path graph.
 *
 * The graph is built per-arch from platforminfo.ini, so arch differences are already
 * baked in and callers never test NPUArch themselves.
 */
class MemPathUtils {
public:
    /**
     * @brief Whether the platform provides a direct data move from one memory level to another.
     *
     * @param from source memory type.
     * @param to destination memory type.
     * @return true when a direct path exists on the current arch.
     */
    static bool CanMoveTo(MemoryType from, MemoryType to);

    /**
     * @brief Whether data at this memory level can be saved to DDR by a single move.
     *
     * Spill source resolution walks upstream until this holds.
     *
     * @param from memory type holding the data.
     * @return true when a direct move to DDR exists.
     */
    static bool CanSaveToDDR(MemoryType from);

    /**
     * @brief Whether data in DDR can be loaded back into this memory level by a single move.
     *
     * Decides the spill reload strategy: true means allocate a same-memType buffer and
     * retarget consumers; false means the consumers themselves are replaced by the copyin.
     *
     * @param to memory type to load into.
     * @return true when a direct move from DDR exists.
     */
    static bool CanReloadFromDDR(MemoryType to);
};

} // namespace npu::tile_fwk
