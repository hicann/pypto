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
 * \file simulation_platform.h
 * \brief
 */

#pragma once
#ifndef SIMULATION_PLATFORM_H
#define SIMULATION_PLATFORM_H

#include <string>
#include <cstdint>
#include "tilefwk/file.h"

namespace npu {
namespace tile_fwk {
class SimulationPlatform {
public:
    SimulationPlatform() = default;
    ~SimulationPlatform() = default;

    bool GetCostModelPlatformRealPath(std::string &realPath);
private:
    static std::string GetCurrentSharedLibPath();
};
} // namespace tile_fwk
} // namespace npu
#endif