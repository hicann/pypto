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
 * \file aikernel_drco_leaf_task_ready_queue.h
 * \brief
 */

#ifndef AIKERNEL_DRCO_LEAF_TASK_READY_QUEUE_H
#define AIKERNEL_DRCO_LEAF_TASK_READY_QUEUE_H

#include "tilefwk/aikernel_tensor.h"

namespace npu::tile_fwk {

/*
 * DRCO = Dependency Resolving by aiCOre
 * DRCU = Dependency Resolving by aiCpU
 */

using LeafTaskId = uint32_t;

struct PerCoreReadyList {
    uint32_t size;
    LeafTaskId taskList[0];
};

struct HubC2VReadyQueue {
    static constexpr uint32_t RING_BUF_SIZE = 6;
    uint32_t head; // 只有C能写
    uint32_t tail;
    uint32_t elems[RING_BUF_SIZE];

#ifndef __TILE_FWK_HOST__
    static inline void Init(HubC2VReadyQueue* addr)
    {
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
        __ssbuf__ uint32_t* ptr = (__ssbuf__ uint32_t*)addr;
        ptr[0] = 0;
        ptr[1] = 0;
        __asm__ volatile("DSB #0");
#else
        (void)addr;
#endif
    }

    // C(AIC)调用：写入一个元素，满则返回false
    static inline bool Push(HubC2VReadyQueue* addr, uint32_t elem)
    {
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
        __ssbuf__ uint32_t* ptr = (__ssbuf__ uint32_t*)addr;
        uint32_t curHead = ptr[0];
        uint32_t curTail = ptr[1];
        uint32_t next = (curHead + 1) % RING_BUF_SIZE;
        if (next == curTail) {
            return false; // 满
        }
        ptr[2 + curHead] = elem; // elems[head]
        ptr[0] = next;           // 更新head
        __asm__ volatile("DSB #0");
        return true;
#else
        (void)addr;
        (void)elem;
        return false;
#endif
    }

    // V(AIV)调用：读出一个元素，空则返回false
    static inline bool Pop(HubC2VReadyQueue* addr, uint32_t& elem)
    {
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
        __ssbuf__ uint32_t* ptr = (__ssbuf__ uint32_t*)addr;
        uint32_t curTail = ptr[1];
        uint32_t curHead = ptr[0];
        if (curTail == curHead) {
            return false; // 空
        }
        elem = ptr[2 + curTail];                // elems[tail]
        ptr[1] = (curTail + 1) % RING_BUF_SIZE; // 更新tail
        __asm__ volatile("DSB #0");
        return true;
#else
        (void)addr;
        (void)elem;
        return false;
#endif
    }

    static inline uint32_t Size(HubC2VReadyQueue* addr)
    {
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
        __ssbuf__ uint32_t* ptr = (__ssbuf__ uint32_t*)addr;
        uint32_t curHead = ptr[0];
        uint32_t curTail = ptr[1];
        return (curHead + RING_BUF_SIZE - curTail) % RING_BUF_SIZE;
#else
        (void)addr;
        return 0;
#endif
    }
#endif
};

/* per core pending queue mechanism can be used in both codr and cudr */
struct PerCorePendingQueue {
    uint32_t head;
    uint32_t tail;
    uint32_t size;
    HubC2VReadyQueue* body; // 指向CV消息通信区域，位于SSBuf区域
    LeafTaskId taskList[0];
#ifdef __TILE_FWK_HOST__
    PerCorePendingQueue() : head(0), tail(0), size(0) {}

    void UnsafeEnqueue(LeafTaskId task) { taskList[size++] = task; }
#endif
};

struct DrcoLocalReadyQueue {
    alignas(64) uint32_t head;
    alignas(64) uint32_t tail;
    uint32_t size;
    LeafTaskId taskList[0];
#ifdef __TILE_FWK_HOST__
    explicit DrcoLocalReadyQueue(uint32_t capacity) : head(0), tail(0), size(capacity) {}
#endif
};

#define DRCO_ENCODE_TASK(task) ((task) + 1)
#define DRCO_DECODE_TASK(task) ((task) - 1)

struct DrcoGlobalReadyQueue {
    alignas(64) uint32_t head;
    alignas(64) uint32_t tail;
    uint32_t size;
    LeafTaskId taskList[0];
#ifdef __TILE_FWK_HOST__
    DrcoGlobalReadyQueue() : head(0), tail(0), size(0) {}

    void UnsafeEnqueue(LeafTaskId task) { taskList[tail++] = DRCO_ENCODE_TASK(task); }
#endif
};

struct DrcoGlobalReadyQueuePtr {
    __gm__ DrcoGlobalReadyQueue* ptr;
};

} // namespace npu::tile_fwk

#endif
