/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once

#include <cstddef>
#include <vector>
#include <unordered_map>
#include "tilefwk/aicpu_common.h"

namespace npu::tile_fwk {
struct DevAscendProgram {
    // shadow definition in `dev_encode.h`, make sure the first 4 members are the same
    DeviceArgs devArgs;
    uint64_t workspaceSize;
    uint64_t l2CacheOffset;
    uint64_t configKey;
};

class AicoreRtManager {
public:
    AicoreRtManager();
    ~AicoreRtManager();
    AicoreRtManager(AicoreRtManager &other) = delete;
    void operator=(const AicoreRtManager &other) = delete;

    static AicoreRtManager &Instance() {
        static AicoreRtManager Inst;
        return Inst;
    }
    int64_t* TileFwkHiddenInput(const std::vector<uint8_t> &op_bin, const uint64_t config_key, const uint32_t block_dim,
                                const uint64_t workspace_size);
    int64_t* TileFwkHiddenInputWithCache(const std::vector<uint8_t> &op_bin, const uint64_t config_key,
                                         const uint32_t block_dim, const uint64_t workspace_size, const int64_t cache_id);

private:
    static bool AllocDevAddr(void **dev_addr, size_t size, std::vector<void *> &allocated_addrs);
    static void BatchFreeDevAddr(std::vector<void *> &allocated_addrs);
    void SaveAllocatedAddrs(const std::vector<void *> &allocated_addrs);
    void AddHiddenInputCache(const int64_t &cache_id, int64_t *hidden_input);
    int64_t* GetHiddenInputCache(const int64_t &cache_id) const;
    static bool GetAicoreRegInfo(const int32_t device_id, std::vector<int64_t> &aic, std::vector<int64_t> &aiv);
    static bool InitDyBinData(const std::vector<int64_t> &aic, const std::vector<int64_t> &aiv,
                              DevAscendProgram *host_args, std::vector<void *> &allocated_addrs);
    std::vector<void *> allocated_addrs_;
    std::unordered_map<int64_t, int64_t *> cache_hidden_input_map_;
};
} // namespace fe
