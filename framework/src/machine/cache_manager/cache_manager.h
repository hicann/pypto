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
 * \file cache_manager.h
 * \brief
 */

#pragma once

#include <string>
#include <mutex>
#include "machine/host/device_agent_task.h"

namespace npu::tile_fwk {
enum class CacheMode {
    Disable = 0,
    Enable
};
class CacheManager {
public:
    CacheManager(const CacheManager &) = delete;
    CacheManager &operator=(const CacheManager &) = delete;

    static CacheManager &Instance();

    bool Initialize();

    bool MatchBinCache(const std::string &cacheKey) const;

    void SaveTaskFile(const DeviceAgentTask *deviceAgentTask) const;

    bool RecoverTask(const std::string &cacheKey, DeviceAgentTask *deviceAgentTask) const;

    CacheMode GetCacheMode() const { return cacheMode_; }

    bool IsCahceEnable() const { return cacheMode_ == CacheMode::Enable; }
private:
    CacheManager() : isInit_(false), cacheMode_(CacheMode::Disable) {}
    ~CacheManager() {
        isInit_ = false;
        cacheMode_ = CacheMode::Disable;
    }

    bool isInit_;
    CacheMode cacheMode_;
    mutable std::mutex cacheMutex_;
    std::string cacheDirPath_;
};
};