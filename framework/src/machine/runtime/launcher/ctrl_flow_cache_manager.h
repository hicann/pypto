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
 * \file ctrl_flow_cache_manager.h
 * \brief Control flow cache data structure for kernel launch.
 */

#pragma once

#include <cstdint>
#include <vector>

#include "machine/runtime/launcher/device_launcher_types.h"

namespace npu::tile_fwk::dynamic {

struct ControlFlowCache {
    int64_t hash;
    std::vector<DeviceTensorData> inputs;
    uint8_t* devCache{nullptr};

    ControlFlowCache(std::vector<DeviceTensorData>& datas, uint8_t* tcache) : inputs(datas), devCache(tcache)
    {
        hash = Hash(inputs);
    }

    static int64_t Hash(const std::vector<DeviceTensorData>& datas)
    {
        // FNV-1a
        uint64_t h = 14695981039346656037ull;
        for (auto& data : datas) {
            for (auto x : data.GetShape()) {
                h ^= x;
                h *= 1099511628211ull;
            }
        }
        return h;
    }

    static int64_t Hash(const std::vector<std::vector<int64_t>>& shapes)
    {
        // FNV-1a
        uint64_t h = 14695981039346656037ull;
        for (auto& shape : shapes) {
            for (auto x : shape) {
                h ^= x;
                h *= 1099511628211ull;
            }
        }
        return h;
    }
};

} // namespace npu::tile_fwk::dynamic
