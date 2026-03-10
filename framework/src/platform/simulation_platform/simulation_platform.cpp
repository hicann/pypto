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
 * \file simulation_platform.cpp
 * \brief
 */

#include <climits>
#include <string>
#include <dlfcn.h>
#include "simulation_platform.h"

namespace npu {
namespace tile_fwk {
const std::string PLATFORM_INFO_RELATIVE_PATH = "/configs/A2A3.ini";
const uint32_t PLATFORM_FAILED = 0xFFFFFFFF;
const uint32_t PLATFORM_SUCCESS = 0;

std::string SimulationPlatform::GetCurrentSharedLibPath() {
    std::string currentLibPath;
    Dl_info info;
    if (dladdr(reinterpret_cast<void *>(&GetCurrentSharedLibPath), &info)) {
        currentLibPath = std::string(info.dli_fname);
        int32_t pos = currentLibPath.rfind('/');
        if (pos >= 0) {
            currentLibPath = currentLibPath.substr(0, pos);
        }
    }
    return currentLibPath;
}

bool SimulationPlatform::GetCostModelPlatformRealPath(std::string &realPath) {
    realPath = RealPath(GetCurrentSharedLibPath() + PLATFORM_INFO_RELATIVE_PATH);
    if (realPath.empty()) {
        return false;
    }
    return true;
}
} // namespace tile_fwk
} // namespace npu