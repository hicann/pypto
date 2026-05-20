/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file memory_pool.h
 * \brief
 */

#pragma once

#include <map>
#include <unordered_map>
#include <vector>
#include "adapter/api/runtime_define.h"

namespace npu::tile_fwk {
inline constexpr uint32_t ONG_GB_HUGE_PAGE_FLAGS = RT_MEMORY_HBM | RT_MEMORY_POLICY_HUGE1G_PAGE_ONLY;
inline constexpr uint32_t TWO_MB_HUGE_PAGE_FLAGS = RT_MEMORY_HBM | RT_MEMORY_POLICY_HUGE_PAGE_FIRST;

inline uint64_t MemSizeAlign(const uint64_t bytes, const uint32_t aligns = 512U)
{
    const uint64_t alignSize = (aligns == 0U) ? sizeof(uintptr_t) : aligns;
    return (((bytes + alignSize) - 1U) / alignSize) * alignSize;
}

struct MemoryBlock {
    void* base_addr;
    size_t block_size;
    size_t used_size;
    bool is_huge_1g;

    std::map<uintptr_t, size_t> free_map;

    MemoryBlock(void* addr, size_t size, bool is_huge);
    void Init();
    void* Allocate(uint64_t alignSize);
    void Free(void* ptr, size_t size);
};

class DevMemoryPool {
public:
    DevMemoryPool();
    ~DevMemoryPool();

    bool AllocDevAddrInPool(uint8_t** devAddr, uint64_t size);
    void FreeDevAddr(void* ptr);
    bool CheckAllSentinels();
    void DestroyPool();

private:
    static void FreeMemBlock(MemoryBlock* block);
    static void PrintSentinelVal(std::vector<uint64_t>& sentinelVal, uint8_t* sentinelAddr);
    void PutSentinelAddr(uint8_t* baseAddr, uint64_t baseSize);
    bool CheckSentinel(uint8_t* baseAddr, bool remove = true);
    void RecordAllocation(void* ptr, MemoryBlock* block, size_t size);
    MemoryBlock* CreateNewBlock(uint64_t alignSize);
    void DynamicRecycle();
    void PrintPoolStatus() const;

    std::vector<MemoryBlock*> memoryBlocks_;
    std::unordered_map<void*, MemoryBlock*> addrToBlock_;
    std::unordered_map<void*, size_t> allocSizes_;

    bool needMemCheck_{false};
    std::vector<uint64_t> sentinelVec_;
    std::unordered_map<uint8_t*, std::vector<uint8_t*>> sentinelValMap_;
};
} // namespace npu::tile_fwk
