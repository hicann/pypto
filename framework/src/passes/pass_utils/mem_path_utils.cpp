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
 * \file mem_path_utils.cpp
 * \brief Queries over the platform memory-path graph
 */

#include "passes/pass_utils/mem_path_utils.h"
#include "tilefwk/platform.h"

namespace npu::tile_fwk {

// 通路图由 platform 初始化时按当前 arch 从 platforminfo.ini 建起(platform.cpp SetMemoryPath),
// 所以机型差异已经落在图里: L1->DDR 在 A2/A3(2201) 有、在 A5(3510) 没有, 这里不必再判机型。
bool MemPathUtils::CanMoveTo(MemoryType from, MemoryType to)
{
    return Platform::Instance().GetDie().HasDirectPath(from, to);
}

bool MemPathUtils::CanSaveToDDR(MemoryType from) { return CanMoveTo(from, MemoryType::MEM_DEVICE_DDR); }

bool MemPathUtils::CanReloadFromDDR(MemoryType to) { return CanMoveTo(MemoryType::MEM_DEVICE_DDR, to); }

} // namespace npu::tile_fwk
