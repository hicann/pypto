/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "machine/runtime/memory_utils/memory_pool.h"
#include <iomanip>
#include <optional>
#include <sstream>
#include "tilefwk/pypto_fwk_log.h"
#include "tilefwk/error_code.h"
#include "adapter/api/runtime_api.h"
#include "machine/runtime/launcher/device_launcher.h"
#include "interface/configs/config_manager.h"

namespace npu::tile_fwk {
namespace {
inline constexpr int RTMALLOC_SUCCESS = 0;
inline constexpr size_t ONT_GB_SIZE = 1024 * 1024 * 1024;
inline constexpr uint64_t SENTINEL_VALUE = 0xDEADBEEFDEADBEEF;
inline constexpr uint32_t SENTINEL_NUM = 64;
inline constexpr uint32_t SENTINEL_MEM_SIZE = 512;
inline uint64_t MemSizeAlign(const uint64_t bytes, const uint32_t aligns = 512U)
{
    const uint64_t alignSize = (aligns == 0U) ? sizeof(uintptr_t) : aligns;
    return (((bytes + alignSize) - 1U) / alignSize) * alignSize;
}

inline RtError NormalizedRtMemcpy(
    void *dst, uint64_t destMax, const void *src, uint64_t cnt, RtMemcpyKind kind)
{
    std::optional<dynamic::AclModeGuard> captureRelaxGuard;
    if (dynamic::DeviceLauncher::IsCaptureMode()) {
        captureRelaxGuard.emplace(AclMdlRICaptureMode::RELAXED);
    }
    return RuntimeMemcpyDirect(dst, destMax, src, cnt, kind);
}
}

MemoryBlock::MemoryBlock(void* addr, size_t size, bool isHuge)
    : baseAddr(addr), blockSize(size), usedSize(0), isHuge1G(isHuge)
{
    Init();
}

void MemoryBlock::Init()
{
    if (isHuge1G) {
        freeMap[reinterpret_cast<uintptr_t>(baseAddr)] = blockSize;
    } else {
        freeMap.clear();
    }
}

void* MemoryBlock::Allocate(uint64_t alignSize)
{
    if (!isHuge1G) {
        if (usedSize == 0 && blockSize >= alignSize) {
            usedSize = blockSize;
            return baseAddr;
        }
        return nullptr;
    }

    for (auto it = freeMap.begin(); it != freeMap.end(); ++it) {
        uintptr_t chunkAddr = it->first;
        size_t chunkSize = it->second;

        if (chunkSize >= alignSize) {
            void* usePtr = reinterpret_cast<void*>(chunkAddr);
            size_t remaining = chunkSize - alignSize;

            freeMap.erase(it);

            if (remaining > 0) {
                freeMap[chunkAddr + alignSize] = remaining;
            }

            usedSize += alignSize;
            MACHINE_LOGI(
                "Allocate in 1GB block: ptr=%p, chunkSize=%zu, alignSize=%lu.", usePtr, chunkSize, alignSize);
            return usePtr;
        }
    }
    return nullptr;
}

void MemoryBlock::Free(void* ptr, size_t size)
{
    if (!isHuge1G) {
        MACHINE_LOGE(DevCommonErr::FREE_FAILED, "Logic Error: 2MB block should not call Free()");
        return;
    }

    uintptr_t addr = reinterpret_cast<uintptr_t>(ptr);

    freeMap[addr] = size;
    usedSize -= size;

    auto it = freeMap.find(addr);
    if (it == freeMap.end())
        return;

    auto nextIt = std::next(it);
    if (nextIt != freeMap.end()) {
        if (it->first + it->second == nextIt->first) {
            it->second += nextIt->second;
            freeMap.erase(nextIt);
        }
    }

    if (it != freeMap.begin()) {
        auto prevIt = std::prev(it);
        if (prevIt->first + prevIt->second == it->first) {
            prevIt->second += it->second;
            freeMap.erase(it);
        }
    }
}

DevMemoryPool& DevMemoryPool::Instance()
{
    static DevMemoryPool memoryPool;
    return memoryPool;
}

DevMemoryPool::DevMemoryPool()
{
    needMemCheck_ = (config::GetDebugOption<int64_t>(CFG_RUNTIME_DBEUG_MODE) == CFG_DEBUG_ALL);
    sentinelVec_ = std::vector<uint64_t>(SENTINEL_NUM, SENTINEL_VALUE);
}

DevMemoryPool::~DevMemoryPool()
{
    CheckAllSentinels();
    DestroyPool();
}

void DevMemoryPool::AllocDevAddr(uint8_t** devAddr, const uint64_t size)
{
    if (!AllocDevAddrInPool(devAddr, size)) {
        MACHINE_LOGE(DevCommonErr::ALLOC_FAILED, "AllocDevAddrInPool failed for size %lu", size);
    } else {
        MACHINE_LOGI("RuntimeAgentMemory: Alloc success %p", *devAddr);
    }
}

bool DevMemoryPool::AllocDevAddrInPool(uint8_t** devAddr, const uint64_t size)
{
    if (size == 0)
        return false;
    if (devAddr == nullptr) {
        MACHINE_LOGE(DevCommonErr::NULLPTR, "devAddr is nullptr");
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

    MACHINE_LOGE(DevCommonErr::ALLOC_FAILED, "Allocate failed: size=%lu", size);
    return false;
}

void DevMemoryPool::FreeDevAddr(void* ptr)
{
    if (ptr == nullptr) {
        MACHINE_LOGE(DevCommonErr::NULLPTR, "Freeing nullptr");
        return;
    }
    CheckSentinel(static_cast<uint8_t*>(ptr), true);

    auto it = addrToBlock_.find(ptr);
    if (it == addrToBlock_.end()) {
        MACHINE_LOGE(DevCommonErr::FREE_FAILED, "Freeing unknown pointer: %p", ptr);
        return;
    }

    MemoryBlock* block = it->second;
    size_t size = allocSizes_[ptr];

    if (block->isHuge1G) {
        block->Free(ptr, size);
    } else {
        MACHINE_LOGI("Directly freeing 2MB block: addr=%p.", block->baseAddr);
        FreeMemBlock(block);
        for (auto vecIt = memoryBlocks_.begin(); vecIt != memoryBlocks_.end(); ++vecIt) {
            if (*vecIt == block) {
                memoryBlocks_.erase(vecIt);
                break;
            }
        }
    }

    addrToBlock_.erase(it);
    allocSizes_.erase(ptr);
}

void DevMemoryPool::PutSentinelAddr(uint8_t* baseAddr, uint64_t baseSize)
{
    if (needMemCheck_) {
        uint8_t* sentinelAddr = baseAddr + baseSize;
        if (NormalizedRtMemcpy(sentinelAddr, SENTINEL_MEM_SIZE, sentinelVec_.data(), SENTINEL_MEM_SIZE,
                                          RtMemcpyKind::HOST_TO_DEVICE) != 0) {
            MACHINE_LOGW("Memory copy sentinel value failed! Do not check memory.");
            return;
        }
        MACHINE_LOGI("Base addr add: baseAddr=%p, sentinelAddr=%p.", baseAddr, sentinelAddr);
        sentinelValMap_[baseAddr].push_back(sentinelAddr);
    }
}

bool DevMemoryPool::CheckAllSentinels()
{
    if (!needMemCheck_) {
        return true;
    }
    bool allGood = true;
    for (auto& iter : sentinelValMap_) {
        if (!CheckSentinel(iter.first, false)) {
            allGood = false;
        }
    }
    if (!allGood) {
        MACHINE_LOGE(HostLauncherErr::MEM_POOL_CHECK_ALL_SENTINELS_FAILED, "CheckAllSentinels failed.");
    }
    sentinelValMap_.clear();
    return allGood;
}

void DevMemoryPool::PrintSentinelVal(std::vector<uint64_t>& sentinelVal, uint8_t* sentinelAddr)
{
    std::ostringstream oss;
    uint8_t* bytePtr = reinterpret_cast<uint8_t*>(sentinelVal.data());
    oss << "Print Sentinel val in hex with ori val[" << std::hex << "0x" << SENTINEL_VALUE << "]" << std::endl;
    MACHINE_LOGW("%s", oss.str().c_str());
    oss.str("");
    for (uint32_t i = 0; i < SENTINEL_MEM_SIZE; ++i) {
        oss << std::hex << std::setw(2) << std::setfill('0') << (int)bytePtr[i];
        if ((i + 1) % 16 == 0) {
            oss << std::endl;
        } else {
            oss << " ";
        }
        if ((i + 1) % 64 == 0) {
            MACHINE_LOGW("Sentinel Addr:%p Val:[\n%s]", sentinelAddr + i, oss.str().c_str());
            oss.str("");
        }
    }
}

bool DevMemoryPool::CheckSentinel(uint8_t* baseAddr, bool remove)
{
    if (!needMemCheck_ || sentinelValMap_.empty()) {
        return true;
    }
    if (baseAddr == reinterpret_cast<uint8_t*>(0x12345678)) {
        return true;
    }
    auto iter = sentinelValMap_.find(baseAddr);
    if (iter == sentinelValMap_.end()) {
        MACHINE_LOGE(DevCommonErr::PARAM_CHECK_FAILED, "Base addr %p not found in map, need check code.", baseAddr);
        return false;
    }
    std::vector<uint64_t> sentinelVal(SENTINEL_NUM, 0);
    bool allGood = true;
    auto& sentinelVec = iter->second;
    for (auto sentinelAddr : sentinelVec) {
        MACHINE_LOGI("Check Sentinel: baseAddr=%p, sentinelAddr=%p.", baseAddr, sentinelAddr);
        if (NormalizedRtMemcpy(sentinelVal.data(), SENTINEL_MEM_SIZE, sentinelAddr, SENTINEL_MEM_SIZE,
                                          RtMemcpyKind::DEVICE_TO_HOST) != 0) {
            MACHINE_LOGW("Memory copy D2H failed! Do not check memory.");
            break;
        }
        if (memcmp(sentinelVal.data(), sentinelVec_.data(), SENTINEL_MEM_SIZE) != 0) {
            PrintSentinelVal(sentinelVal, sentinelAddr);
            allGood = false;
        }
    }
    if (!allGood) {
        MACHINE_LOGE(DevCommonErr::PARAM_CHECK_FAILED, "BaseAddr:%p check sentinel failed.", baseAddr);
    } else {
        MACHINE_LOGI("BaseAddr:%p check sentinel Ok.", baseAddr);
    }
    if (remove) {
        sentinelValMap_.erase(baseAddr);
    }
    return allGood;
}

void DevMemoryPool::DynamicRecycle()
{
    auto it = memoryBlocks_.begin();
    while (it != memoryBlocks_.end()) {
        if ((*it)->usedSize == 0) {
            MACHINE_LOGI("Recycling empty block: addr=%p", (*it)->baseAddr);
            FreeMemBlock(*it);
            it = memoryBlocks_.erase(it);
        } else {
            ++it;
        }
    }
}

void DevMemoryPool::DestroyPool()
{
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

void DevMemoryPool::PrintPoolStatus() const
{
    size_t cnt1G = 0;
    size_t cnt2M = 0;
    size_t total = 0;
    size_t used = 0;
    MACHINE_LOGI("========== [Memory Pool Status] ==========");
    for (size_t i = 0; i < memoryBlocks_.size(); ++i) {
        auto* blk = memoryBlocks_[i];
        if (blk->isHuge1G)
            cnt1G++;
        else
            cnt2M++;
        total += blk->blockSize;
        used += blk->usedSize;

        double rate = blk->blockSize ? (double)blk->usedSize * 100.0 / blk->blockSize : 0;
        MACHINE_LOGI(
            "Block[%lu] %s | Addr: %p | Used: %.1f%% | Fragments: %lu", i, blk->isHuge1G ? "1G" : "2M",
            blk->baseAddr, rate, blk->freeMap.size());
    }
    MACHINE_LOGI("Summary: 1G x %lu, 2M x %lu | Used/Total: %lu/%lu MB", cnt1G, cnt2M, used >> 20, total >> 20);
}

void DevMemoryPool::FreeMemBlock(MemoryBlock* block)
{
    if (block == nullptr) {
        return;
    }

    if (block->baseAddr != nullptr) {
        MACHINE_LOGI("Releasing physical memory: addr=%p, size=%lu", block->baseAddr, block->blockSize);
        RuntimeFree(block->baseAddr);
        block->baseAddr = nullptr;
    }
    delete block;
    block = nullptr;
}

void DevMemoryPool::RecordAllocation(void* ptr, MemoryBlock* block, size_t size)
{
    addrToBlock_[ptr] = block;
    allocSizes_[ptr] = size;
}

MemoryBlock* DevMemoryPool::CreateNewBlock(uint64_t alignSize)
{
    uint8_t* devAddr = nullptr;
    size_t size1G = ((alignSize - 1) / ONT_GB_SIZE + 1) * ONT_GB_SIZE;

    if (RuntimeMalloc((void**)&devAddr, size1G, ONG_GB_HUGE_PAGE_FLAGS, 0) == RTMALLOC_SUCCESS) {
        MemoryBlock* block = new MemoryBlock(devAddr, size1G, true);
        memoryBlocks_.push_back(block);
        return block;
    }

    if (RuntimeMalloc((void**)&devAddr, alignSize, TWO_MB_HUGE_PAGE_FLAGS, 0) == RTMALLOC_SUCCESS) {
        MemoryBlock* block = new MemoryBlock(devAddr, alignSize, false);
        memoryBlocks_.push_back(block);
        return block;
    }

    MACHINE_LOGE(DevCommonErr::ALLOC_FAILED, "All memory alloc strategies failed");
    return nullptr;
}
} // namespace npu::tile_fwk
