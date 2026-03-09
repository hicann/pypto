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
#include <list>
#include <mutex>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include "interface/utils/log.h"
#include "interface/configs/config_manager.h"

#ifdef BUILD_WITH_CANN
#include "acl/acl.h"
#include "runtime/rt.h"
#include "runtime/rt_preload_task.h"
#endif

namespace npu::tile_fwk {
#ifdef BUILD_WITH_CANN
inline constexpr int RTMALLOC_SUCCESS = 0;
inline constexpr uint32_t ONG_GB_HUGE_PAGE_FLAGS = RT_MEMORY_HBM | RT_MEMORY_POLICY_HUGE1G_PAGE_ONLY;
inline constexpr size_t ONT_GB_SIZE = 1024 * 1024 * 1024;
inline constexpr uint32_t TWO_MB_HUGE_PAGE_FLAGS = RT_MEMORY_HBM | RT_MEMORY_POLICY_HUGE_PAGE_FIRST;

static constexpr uint64_t SENTINEL_VALUE = 0xDEADBEEFDEADBEEF;
static constexpr uint32_t SENTINEL_NUM = 64;
static constexpr uint32_t SENTINEL_MEM_SIZE = 512;

inline uint64_t MemSizeAlign(const uint64_t bytes, const uint32_t aligns = 512U) {
    const uint64_t alignSize = (aligns == 0U) ? sizeof(uintptr_t) : aligns;
    return (((bytes + alignSize) - 1U) / alignSize) * alignSize;
}

struct MemoryBlock {
    void* base_addr;
    size_t block_size;
    size_t used_size;
    bool is_huge_1g;
    
    std::map<uintptr_t, size_t> free_map;
    
    MemoryBlock(void* addr, size_t size, bool is_huge) 
        : base_addr(addr), block_size(size), used_size(0), 
        is_huge_1g(is_huge) {
        Init();
    }

    void Init() {
        if (is_huge_1g) {
            free_map[reinterpret_cast<uintptr_t>(base_addr)] = block_size;
        } else {
            free_map.clear();
        }
    }

    void* Allocate(uint64_t alignSize) {
        if (!is_huge_1g) {
            if (used_size == 0 && block_size >= alignSize) {
                used_size = block_size;
                return base_addr;
            } else {
                MACHINE_LOGE("Logic Error: 2MB block allocation failed. (used_size: %zu, block_size: %zu, req: %lu)", used_size, block_size, alignSize);
                return nullptr;
            }
        }

        for (auto it = free_map.begin(); it != free_map.end(); ++it) {
            uintptr_t chunk_addr = it->first;
            size_t chunk_size = it->second;

            if (chunk_size >= alignSize) {
                void* use_ptr = reinterpret_cast<void*>(chunk_addr);
                size_t remaining = chunk_size - alignSize;

                free_map.erase(it);

                if (remaining > 0) {
                    free_map[chunk_addr + alignSize] = remaining;
                }

                used_size += alignSize;
                MACHINE_LOGI("Allocate in 1GB block: %p, size: %zu, alignSize: %lu", use_ptr, chunk_size, alignSize);
                return use_ptr;
            }
        }
        return nullptr;
    }

    void Free(void* ptr, size_t size) {
        if (!is_huge_1g) {
            MACHINE_LOGE("Logic Error: 2MB block should not call Free()");
            return;
        }

        uintptr_t addr = reinterpret_cast<uintptr_t>(ptr);
        
        free_map[addr] = size;
        used_size -= size;

        auto it = free_map.find(addr);
        if (it == free_map.end()) return;

        auto next_it = std::next(it);
        if (next_it != free_map.end()) {
            if (it->first + it->second == next_it->first) {
                it->second += next_it->second;
                free_map.erase(next_it);
            }
        }

        if (it != free_map.begin()) {
            auto prev_it = std::prev(it);
            if (prev_it->first + prev_it->second == it->first) {
                prev_it->second += it->second;
                free_map.erase(it);
            }
        }
    }
};

class DevMemoryPool {
public:
    DevMemoryPool() {
        needMemCheck_ = (config::GetDebugOption<int64_t>(CFG_RUNTIME_DBEUG_MODE) == CFG_DEBUG_ALL);
        sentinelVec_ = std::vector<uint64_t>(SENTINEL_NUM, SENTINEL_VALUE);
    }
    ~DevMemoryPool() { 
        CheckAllSentinels();
        DestroyPool();
    }

    bool AllocDevAddrInPool(uint8_t **devAddr, uint64_t size) {
        if (size == 0) return false;
        if (devAddr == nullptr) {
            MACHINE_LOGE("devAddr is nullptr");
            return false;
        }
        auto alignSize = MemSizeAlign(size);
        if (needMemCheck_) {
            alignSize += SENTINEL_MEM_SIZE;
        }

        for (auto& block : memoryBlocks_) {
            void* ptr = block->Allocate(alignSize);
            if (ptr != nullptr) {
                *devAddr = static_cast<uint8_t*>(ptr);
                RecordAllocation(ptr, block, alignSize);
                PutSentinelAddr(*devAddr, size);
                return true;
            }
        }
        
        MemoryBlock* newBlock = CreateNewBlock(alignSize);
        if (newBlock != nullptr) {
            void* ptr = newBlock->Allocate(alignSize);
            if (ptr != nullptr) {
                *devAddr = static_cast<uint8_t*>(ptr);
                RecordAllocation(ptr, newBlock, alignSize);
                PutSentinelAddr(*devAddr, size);
                return true;
            }
        }
        
        MACHINE_LOGE("Allocate failed size %lu", size);
        return false;
    }
    
    void FreeDevAddr(void* ptr) {
        if (ptr == nullptr) {
            MACHINE_LOGE("Freeing nullptr");
            return; 
        }
        CheckSentinel(static_cast<uint8_t*>(ptr), true);

        auto it = addrToBlock_.find(ptr);
        if (it == addrToBlock_.end()) {
            MACHINE_LOGE("Freeing unknown pointer: %p", ptr);
            return;
        }

        MemoryBlock* block = it->second;
        size_t size = allocSizes_[ptr];

        if (block->is_huge_1g) {
            block->Free(ptr, size);
        } else {
            MACHINE_LOGI("Directly freeing 2MB block addr %p", block->base_addr);
            FreeMemBlock(block); 
            for (auto vec_it = memoryBlocks_.begin(); vec_it != memoryBlocks_.end(); ++vec_it) {
                if (*vec_it == block) {
                    memoryBlocks_.erase(vec_it);
                    break; 
                }
            }
        }

        addrToBlock_.erase(it);
        allocSizes_.erase(ptr);
    }

    void PutSentinelAddr(uint8_t *baseAddr, uint64_t baseSize) {
        if (needMemCheck_) {
            uint8_t *sentinelAddr = baseAddr + baseSize;
            if (rtMemcpy(sentinelAddr, SENTINEL_MEM_SIZE, sentinelVec_.data(), SENTINEL_MEM_SIZE, RT_MEMCPY_HOST_TO_DEVICE) != 0) {
                MACHINE_LOGW("Memory copy sentinel value failed! Do not check memory.");
                return;
            }
            MACHINE_LOGI("Base addr add %p with sentinelAddr %p.", baseAddr, sentinelAddr);
            sentinelValMap_[baseAddr].push_back(sentinelAddr);
        }
    }

    // Check sentinel values for memory corruption
    bool CheckAllSentinels() {
        if (!needMemCheck_) {
            return true;
        }
        bool allGood = true;
        for (auto &iter : sentinelValMap_) {
            if (!CheckSentinel(iter.first, false)) {
                allGood = false;
            }
        }
        if (!allGood) {
            MACHINE_LOGE("CheckAllSentinels failed.");
        }
        sentinelValMap_.clear();
        return allGood;
    }

    void PrintSentinelVal(std::vector<uint64_t> &sentinelVal, uint8_t *sentinelAddr) {
        std::ostringstream oss;
        uint8_t* byte_ptr = reinterpret_cast<uint8_t*>(sentinelVal.data());
        oss << "Print Sentinel val in hex with ori val[" << std::hex << "0x" << SENTINEL_VALUE << "]" << std::endl;
        ALOG_ERROR_F("%s", oss.str().c_str());
        oss.str("");
        for (uint32_t i = 0; i < SENTINEL_MEM_SIZE; ++i) {
            oss << std::hex << std::setw(2) << std::setfill('0') << (int)byte_ptr[i];
            if ((i + 1) % 16 == 0) {
                oss << std::endl;
            } else {
                oss << " ";
            }
            if ((i + 1) % 64 == 0) {
                MACHINE_LOGE("Sentinel Addr:%p Val:[\n%s]", sentinelAddr + i, oss.str().c_str());
                oss.str("");
            }
        }
    }

    // Check sentinel values for memory corruption
    bool CheckSentinel(uint8_t *baseAddr, bool remove = true) {
        if (!needMemCheck_ || sentinelValMap_.empty()) {
            return true;
        }
        // UT no need check sentinel
        if (baseAddr == reinterpret_cast<uint8_t*>(0x12345678)) {
            return true;
        }
        auto iter = sentinelValMap_.find(baseAddr);
        if (iter == sentinelValMap_.end()) {
            MACHINE_LOGE("Base addr %p not found in map, need check code.", baseAddr);
            return false;
        }
        std::vector<uint64_t> sentinelVal(SENTINEL_NUM, 0);
        bool allGood = true;
        auto &sentinelVec = iter->second;
        for (auto sentinelAddr : sentinelVec) {
            MACHINE_LOGI("Check base:%p sentinelAddr:%p.", baseAddr, sentinelAddr);
            if (rtMemcpy(sentinelVal.data(), SENTINEL_MEM_SIZE, sentinelAddr, SENTINEL_MEM_SIZE, RT_MEMCPY_DEVICE_TO_HOST) != 0) {
                MACHINE_LOGW("Memory copy D2H failed! Do not check memory.");
                break;
            }
            if (memcmp(sentinelVal.data(), sentinelVec_.data(), SENTINEL_MEM_SIZE) != 0) {
                PrintSentinelVal(sentinelVal, sentinelAddr);
                allGood = false;
            }
        }
        if (!allGood) {
            MACHINE_LOGE("BaseAddr:%p check sentinel failed.", baseAddr);
        } else {
            MACHINE_LOGI("BaseAddr:%p check sentinel Ok.", baseAddr);
        }
        if (remove) {
            sentinelValMap_.erase(baseAddr);
        }
        return allGood;
    }

    void DynamicRecycle() {
        auto it = memoryBlocks_.begin();
        while (it != memoryBlocks_.end()) {
            if ((*it)->used_size == 0) {
                MACHINE_LOGI("Recycling empty block %p", (*it)->base_addr);
                FreeMemBlock(*it);
                it = memoryBlocks_.erase(it);
            } else {
                ++it;
            }
        }
    }

    void DestroyPool() {
        for (auto& block : memoryBlocks_) {
            if (block != nullptr) {
                FreeMemBlock(block);
            }
        }
        memoryBlocks_.clear();
        addrToBlock_.clear();
        allocSizes_.clear();
        MACHINE_LOGI("MemPool destroyed, all memory freed");
    }

    void PrintPoolStatus() {
        size_t cnt_1g = 0, cnt_2m = 0;
        size_t total = 0, used = 0;
        
        MACHINE_LOGI("========== [Memory Pool Status] ==========");
        for (size_t i = 0; i < memoryBlocks_.size(); ++i) {
            auto* blk = memoryBlocks_[i];
            if (blk->is_huge_1g) cnt_1g++; else cnt_2m++;
            total += blk->block_size;
            used += blk->used_size;
            
            double rate = blk->block_size ? (double)blk->used_size * 100.0 / blk->block_size : 0;
            MACHINE_LOGI("Block[%lu] %s | Addr: %p | Used: %.1f%% | Fragments: %lu", 
                i, blk->is_huge_1g ? "1G" : "2M", blk->base_addr, rate, blk->free_map.size());
        }
        MACHINE_LOGI("Summary: 1G x %lu, 2M x %lu | Used/Total: %lu/%lu MB", 
            cnt_1g, cnt_2m, used>>20, total>>20);
    }

private:
    void FreeMemBlock(MemoryBlock* block) {
        if (block == nullptr) {
            return;
        }

        if (block->base_addr != nullptr) {
            MACHINE_LOGI("Releasing physical memory: %p (size %lu)", block->base_addr, block->block_size);
            rtFree(block->base_addr);
            block->base_addr = nullptr;
        }
        delete block;
        block = nullptr;
    }

    void RecordAllocation(void* ptr, MemoryBlock* block, size_t size) {
        addrToBlock_[ptr] = block;
        allocSizes_[ptr] = size;
    }

    MemoryBlock* CreateNewBlock(uint64_t alignSize) {
        uint8_t *devAddr = nullptr;
        size_t size1G = ((alignSize - 1) / ONT_GB_SIZE + 1) * ONT_GB_SIZE;
        
        if (rtMalloc((void**)&devAddr, size1G, ONG_GB_HUGE_PAGE_FLAGS, 0) == RTMALLOC_SUCCESS) {
            MemoryBlock* block = new MemoryBlock(devAddr, size1G, true);
            memoryBlocks_.push_back(block);
            return block;
        }

        if (rtMalloc((void**)&devAddr, alignSize, TWO_MB_HUGE_PAGE_FLAGS, 0) == RTMALLOC_SUCCESS) {
            MemoryBlock* block = new MemoryBlock(devAddr, alignSize, false);
            memoryBlocks_.push_back(block);
            return block;
        }

        MACHINE_LOGE("All memory alloc strategies failed");
        return nullptr;
    }

    std::vector<MemoryBlock*> memoryBlocks_;
    std::unordered_map<void*, MemoryBlock*> addrToBlock_;
    std::unordered_map<void*, size_t> allocSizes_;

    bool needMemCheck_{false};
    std::vector<uint64_t> sentinelVec_;
    std::unordered_map<uint8_t*, std::vector<uint8_t*>> sentinelValMap_;
};
#endif
} // namespace npu::tile_fwk
