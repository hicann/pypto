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
 * \file seq_ws_allocator.h
 * \brief
 */

#pragma once
#include <cstddef>
#include <cstdint>
#include <algorithm>

#include "machine/utils/device_log.h"
#include "ws_allocator_basics.h"

namespace npu::tile_fwk::dynamic {

enum class WsAicpuSlabMemType : uint8_t {
    DUPPED_FUNC_DATA = 0,
    DYN_FUNC_DATA,
    VEC_STITCHED_LIST,
    DEV_DYN_TASK,
    READY_QUE,
#ifdef SUPPORT_MIX_SUBGRAPH_SCHE
    WRAP_QUEUE,
    WRAP_TASKLIST,
#endif
    COHERENT_SLAB_MEM_TYPE_BUTT, //add new slabmemtype should be above this type

    DUPPED_STITCH, // stitch pool memory
    SLAB_MEM_TYPE_BUTT
};
constexpr int SLAB_ALLOCATOR_MAX_CACHES = 16;

struct StageAllocInfo {
    void* heads[SLAB_ALLOCATOR_MAX_CACHES];
    void* tails[SLAB_ALLOCATOR_MAX_CACHES];
    uint32_t objCnt[SLAB_ALLOCATOR_MAX_CACHES];
};

class SlabWsAllocator {
private:
    struct SlabHeader;
    struct SlabCache {
        uint32_t objSize{0};
        void* freeList{nullptr};
        void* freeListTail{nullptr};
        SlabHeader* activeSlab{nullptr};
        SlabWsAllocator* allocator{nullptr};

        // Per-cache allocation tracking
        void* stageAllocHead{nullptr};
        void* stageAllocTail{nullptr};

        // Cache-level statistics
        uint32_t slabCount{0};
        uint32_t totalObjCount{0};
        uint32_t allocatedObjCount{0};
        uint32_t unPopAllocatedObjCount{0};

        SlabCache() {}
        SlabCache(uint32_t size, SlabWsAllocator* alloc) 
            : objSize(size), freeList(nullptr), activeSlab(nullptr), 
              allocator(alloc), stageAllocHead(nullptr), stageAllocTail(nullptr),
              slabCount(0), totalObjCount(0), allocatedObjCount(0), unPopAllocatedObjCount(0)  {}
    };

    struct SlabHeader {
        SlabCache* cache; // extend
        uint16_t allocatedCount;
        uint16_t totalCount;
    };
public:
    SlabWsAllocator() = default;
    
    void Init(void* baseAddr, uint32_t totalSize, uint32_t alignSize) {
        memBaseaddr_ = static_cast<uint8_t*>(baseAddr);
        totalMemSize_ = totalSize;
        slabAlignSize_ = (((alignSize) + (sizeof(uint64_t)) - 1) & ~((sizeof(uint64_t)) - 1));
        nextFreeSlabAddr_ = memBaseaddr_;
        freeSlabList_ = nullptr;
        numCaches_ = 0;

        totalSlabCount_ = totalMemSize_/slabAlignSize_;
        allocatedSlabCount_ = 0;
        DEV_DEBUG("[SlabWsAllocator]Init SlabWsAllocator: base=%p, size=%u, align=%u.\n",
                  memBaseaddr_, totalMemSize_, slabAlignSize_);
    }

    bool RegistCache(uint32_t type, uint32_t objSize) {
        if (type >= SLAB_ALLOCATOR_MAX_CACHES || objSize == 0) {
            return false;
        }
        
        if (caches_[type].objSize != 0) {
            if (caches_[type].objSize >= objSize) {
                DEV_DEBUG("[SlabWsAllocator]Slab cache exists : objsize = %u, cacheType = %u.\n", objSize, type);
                return true;
            }
            DEV_ERROR("[SlabWsAllocator]Add cache failed type = %u, objsize = %u", type, objSize);
            return false;
        }
        uint32_t realObjSize = (((objSize) + (sizeof(uint64_t)) - 1) & ~((sizeof(uint64_t)) - 1));
        caches_[type] = SlabCache(realObjSize, this);
        numCaches_++;
        DEV_DEBUG("[SlabWsAllocator]Add slab cache : objsize = %u, realobjsize = %u, type = %u.\n", objSize, realObjSize, type);
        return true;
    }

    bool ExistCache(uint32_t cacheType, uint32_t objSize) {
        if (cacheType >= SLAB_ALLOCATOR_MAX_CACHES) {
            return false;
        }
        if (caches_[cacheType].objSize >= objSize) {
            return true;
        }
        return false;
    }

    void AfterAllocSuccess(SlabCache& cache, void* obj, uint32_t objSize) {
        *static_cast<void**>(obj) = nullptr;
        if (!cache.stageAllocHead) {
            cache.stageAllocHead = obj;
        } else {
            *static_cast<void**>(cache.stageAllocTail) = obj;
        }
        cache.stageAllocTail = obj;
        cache.allocatedObjCount++;
        cache.unPopAllocatedObjCount++;
        totalAllocNum_++;
        totalMemReq_ += objSize;
        return;
    }

    void* Alloc(uint32_t cacheType) {
        if (cacheType >= SLAB_ALLOCATOR_MAX_CACHES) {
            return nullptr;
        }
        
        SlabCache& cache = caches_[cacheType];
        uint32_t objSize = cache.objSize;
        if (objSize == 0) {
            return nullptr;
        }

        void* obj = nullptr;
        if (cache.freeList) {
            obj = cache.freeList;
            cache.freeList = *static_cast<void**>(obj);
            if (cache.freeList == nullptr) {
                cache.freeListTail = nullptr;
            }
            DEV_VERBOSE_DEBUG("[SlabWsAllocator]Alloc from slab free list: objsize = %u .\n", objSize);
        } else if (cache.activeSlab && cache.activeSlab->allocatedCount < cache.activeSlab->totalCount) {
            SlabHeader* slab = cache.activeSlab;
            obj = static_cast<uint8_t*>(static_cast<void*>(slab)) + 
                   sizeof(SlabHeader) + slab->allocatedCount * (sizeof(void*) + objSize);
            slab->allocatedCount++;
            DEV_VERBOSE_DEBUG("[SlabWsAllocator]Alloc from active slab: slab = %p, objsize = %u, allocCnt=%u .\n",
                slab, objSize, slab->allocatedCount);
        } else {
            void* slabMem = get_free_slab();
            if (!slabMem) {
                DEV_IF_DEBUG{
                    DumpMemoryStatusWhenAbnormal("alloc null:");
                }

                DEV_DEBUG("[SlabWsAllocator]Alloc memory(%u) not enough : objsize = %u .\n", cacheType, objSize);
                return nullptr; // memory not enough
            }

            SlabHeader* header = new (slabMem) SlabHeader();
            header->cache = &cache;
            header->allocatedCount = 1;
            header->totalCount = (slabAlignSize_ - sizeof(SlabHeader)) / (sizeof(void*) + objSize);
            cache.activeSlab = header;
            obj = static_cast<uint8_t*>(slabMem) + sizeof(SlabHeader);
            
            // Update statistics for new slab
            cache.slabCount++;
            cache.totalObjCount += header->totalCount;
            allocatedSlabCount_++;
            
            DEV_VERBOSE_DEBUG("[SlabWsAllocator]Alloc from new slab: slab = %p, objsize = %u, totalCnt=%u .\n",
                header, objSize, header->totalCount);
        }

        AfterAllocSuccess(cache, obj, objSize);
        DEV_VERBOSE_DEBUG("[SlabWsAllocator]Alloc sucess obj = %p cacheType = %u size = %u .\n", obj, cacheType, objSize);
        return static_cast<uint8_t*>(obj) + sizeof(void*);
    }

    StageAllocInfo PopStageAllocMem(bool keepTail, uint32_t memType) {
        StageAllocInfo info;
        for (uint32_t i = 0; i < SLAB_ALLOCATOR_MAX_CACHES; i++) {
            info.heads[i] = caches_[i].stageAllocHead;
            info.tails[i] = caches_[i].stageAllocTail;
            if (!keepTail || i != memType) {
                // Reset cache tracking
                caches_[i].stageAllocHead = nullptr;
                caches_[i].stageAllocTail = nullptr;
                info.objCnt[i] = caches_[i].allocatedObjCount;
                caches_[i].unPopAllocatedObjCount = 0;
                continue;
            }

            if (caches_[i].stageAllocHead == caches_[i].stageAllocTail) {
                info.heads[i] = nullptr;
                info.tails[i] = nullptr;
                info.objCnt[i] = 0;
            } else {
                void* temp = caches_[i].stageAllocHead;
                if (temp == nullptr) {
                    DEV_ERROR("stageAllocHead is null for cache index %u\n", i);
                }
                DEV_ASSERT(temp != nullptr);
                while (*static_cast<void**>(temp) != caches_[i].stageAllocTail) {
                    temp = *static_cast<void**>(temp);
                }
                if (temp == nullptr) {
                    DEV_ERROR("stageAllocHead is null after loop for cache index %u, stageAllocTail: %p, \n", i, caches_[i].stageAllocTail);
                }
                DEV_ASSERT(temp != nullptr);
                *static_cast<void**>(temp) = nullptr;
                info.tails[i] = temp;
                DEV_VERBOSE_DEBUG("Keep tail not pop %p \n", caches_[i].stageAllocTail);
                caches_[i].stageAllocHead = caches_[i].stageAllocTail;
                info.objCnt[i] = caches_[i].allocatedObjCount - 1;
                caches_[i].unPopAllocatedObjCount = 1;
            }
        }
        
        return info;
    }

    void FreeStageAllocMem(const StageAllocInfo& info) {
        for (int i = 0; i < SLAB_ALLOCATOR_MAX_CACHES; i++) {
            SlabCache& cache = caches_[i];
            cache.allocatedObjCount -= info.objCnt[i];

            DEV_IF_VERBOSE_DEBUG {
                void* temp = info.heads[i];
                while (temp) {
                    DEV_VERBOSE_DEBUG("[SlabWsAllocator]recycle sucess obj = %p cacheType = %d size = %u .\n",
                        temp, i, cache.objSize);
                    temp = *static_cast<void**>(temp);
                }
            }

            /* The recently released ones are placed at the back
               to ensure that newly allocated addresses are from the historical pool. */
            if (cache.freeListTail) {
                *static_cast<void**>(cache.freeListTail) = info.heads[i];
            } else {
                cache.freeList = info.heads[i];
            }
            if (info.tails[i]) {
                cache.freeListTail = info.tails[i];
            }
        }
    }

    /* ==================== DFX Statistics Methods ==================== */
    // Get allocator-level statistics
    struct AllocatorStats {
        uint32_t totalSlabCount;
        uint32_t allocatedSlabCount;
        uint32_t freeSlabCount;
        uint32_t slabSize;
        double usage;
    };

    AllocatorStats GetAllocatorStats() const {
        AllocatorStats stats;
        stats.totalSlabCount = totalSlabCount_;
        stats.allocatedSlabCount = allocatedSlabCount_;
        stats.freeSlabCount = totalSlabCount_ - allocatedSlabCount_;
        stats.slabSize = slabAlignSize_;
        stats.usage = totalSlabCount_ > 0 ? 
            (static_cast<double>(stats.allocatedSlabCount) / stats.totalSlabCount) : 0.0;
        return stats;
    }

    // Get cache-level statistics
    struct CacheStats {
        uint32_t objSize;
        uint32_t slabCount;
        uint32_t totalObjCount;
        uint32_t allocatedObjCount;
        uint32_t freeObjCount;
        double usage;
    };

    CacheStats GetCacheStats(uint32_t cacheType) const {
        if (cacheType >= SLAB_ALLOCATOR_MAX_CACHES) {
            return CacheStats{};
        }
        
        const SlabCache& cache = caches_[cacheType];
        CacheStats stats;
        stats.objSize = cache.objSize;
        stats.slabCount = cache.slabCount;
        stats.totalObjCount = cache.totalObjCount;
        stats.allocatedObjCount = cache.allocatedObjCount;
        stats.freeObjCount = cache.totalObjCount - cache.allocatedObjCount;
        stats.usage = cache.totalObjCount > 0 ? 
            (static_cast<double>(cache.allocatedObjCount) / cache.totalObjCount) : 0.0;
        return stats;
    }

    // Get statistics for all caches
    void GetAllCacheStats(CacheStats stats[SLAB_ALLOCATOR_MAX_CACHES]) const {
        for (int i = 0; i < SLAB_ALLOCATOR_MAX_CACHES; i++) {
            stats[i] = GetCacheStats(i);
        }
    }

    void DumpMemoryStatusWhenAbnormal(const char *title) const {
        DEV_WARN("[SlabWsAllocator]%s\n", title);
        int percent = 100;

        // Dump allocator-level statistics
        AllocatorStats allocStats = GetAllocatorStats();
        DEV_WARN("Slab allocator Stats: BaseMemAddr=%p, TotalSize=%u, TotalSlabs=%u,"
                 "AllocatedSlabs=%u, FreeSlabs=%u, SlabSize=%u, Usage=%.2f%%\n",
                 memBaseaddr_, totalMemSize_, allocStats.totalSlabCount, allocStats.allocatedSlabCount, 
                 allocStats.freeSlabCount, allocStats.slabSize, allocStats.usage * percent);
        
        // Dump cache-level statistics
        for (int i = 0; i < SLAB_ALLOCATOR_MAX_CACHES; i++) {
            if (caches_[i].objSize == 0) continue;

            CacheStats cacheStats = GetCacheStats(i);
            DEV_WARN("Slab cache[%d]: ObjSize=%u, AlloCatedSlabs=%u, TotalObjs=%u,"
                     "AllocatedObjs=%u, FreeObjs=%u, Usage=%.2f%%\n",
                     i, cacheStats.objSize, cacheStats.slabCount, cacheStats.totalObjCount,
                     cacheStats.allocatedObjCount, cacheStats.freeObjCount, cacheStats.usage * percent);
        }
    }

    void DumpMemoryUsage(const char *hint, const char *title) const {
#if DEBUG_MEM_DUMP_LEVEL >= DEBUG_MEM_DUMP_LIGHT
        DEV_MEM_DUMP("%s memory usage (%s)\n", title, hint);
        DEV_MEM_DUMP("            Memory pool size: %10u bytes\n", totalMemSize_);
        DEV_MEM_DUMP("    Total memory requirement: %10u bytes\n", totalMemReq_);
        DEV_MEM_DUMP("      Total allocation count: %10u\n", totalAllocNum_);
#endif // DEBUG_MEM_DUMP_LEVEL >= DEBUG_MEM_DUMP_LIGHT
        (void)title;
        (void)hint;
    }

private:
    void* get_free_slab() {
        if (freeSlabList_) {
            void* slab = freeSlabList_;
            freeSlabList_ = *static_cast<void**>(freeSlabList_);
            return slab;
        }

        if (nextFreeSlabAddr_ + slabAlignSize_ <= memBaseaddr_ + totalMemSize_) {
            void* slab = nextFreeSlabAddr_;
            nextFreeSlabAddr_ += slabAlignSize_;
            return slab;
        }

        return nullptr;
    }

private:
    uint8_t* memBaseaddr_{nullptr};
    uint8_t* nextFreeSlabAddr_{nullptr};
    uint32_t totalMemSize_{0};
    uint32_t slabAlignSize_{0};
    void* freeSlabList_{nullptr};

    SlabCache caches_[SLAB_ALLOCATOR_MAX_CACHES];
    int numCaches_{0};
    
    // Allocator-level statistics
    uint32_t totalSlabCount_{0};
    uint32_t allocatedSlabCount_{0};
    uint32_t totalMemReq_{0};
    uint32_t totalAllocNum_{0};
};

struct WsSlabStageAllocMem {
    std::atomic_bool canFree{false};
    StageAllocInfo generalMetadataStageMem;
    StageAllocInfo stitchStageMem;

    WsSlabStageAllocMem() = default;
    WsSlabStageAllocMem(const WsSlabStageAllocMem& other)
        : canFree(other.canFree.load(std::memory_order_relaxed)),
          generalMetadataStageMem(other.generalMetadataStageMem),
          stitchStageMem(other.stitchStageMem) {}

    WsSlabStageAllocMem& operator=(const WsSlabStageAllocMem& other) {
        if (this != &other) {
            canFree.store(other.canFree.load(std::memory_order_relaxed),
                         std::memory_order_relaxed);
            generalMetadataStageMem = other.generalMetadataStageMem;
            stitchStageMem = other.stitchStageMem;
        }
        return *this;
    }
};
}