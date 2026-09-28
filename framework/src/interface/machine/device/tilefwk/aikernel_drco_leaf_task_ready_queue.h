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

// SPSC = Single Producer Single Consumer
template <uint64_t bufSize = 6>
struct DrcoSsbufSPSCQueue {
    static constexpr uint32_t RING_BUF_SIZE = bufSize;
    static constexpr uint32_t ELEM_OFFSET = 2;
    /* [0] = head, 只有 producer 写
     * [1] = tail, 只有 consumer 读
     */
    uint32_t elems[2 + RING_BUF_SIZE];

#ifndef __TILE_FWK_HOST__
    static inline void Init(DrcoSsbufSPSCQueue* addr)
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
    static inline bool Push(DrcoSsbufSPSCQueue* addr, uint32_t elem)
    {
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
        __ssbuf__ uint32_t* ptr = (__ssbuf__ uint32_t*)addr;
        uint32_t curHead = ptr[0];
        uint32_t curTail = ptr[1];
        if (curTail + RING_BUF_SIZE <= curHead) {
            return false; // 满
        }
        ptr[ELEM_OFFSET + (curHead % RING_BUF_SIZE)] = elem;
        __asm__ volatile("DSB #0");
        ptr[0] = curHead + 1;
        __asm__ volatile("DSB #0");
        return true;
#else
        (void)addr;
        (void)elem;
        return false;
#endif
    }

    // V(AIV)调用：读出一个元素，空则返回false
    static inline bool Pop(DrcoSsbufSPSCQueue* addr, uint32_t& elem)
    {
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
        __ssbuf__ uint32_t* ptr = (__ssbuf__ uint32_t*)addr;
        uint32_t curTail = ptr[1];
        uint32_t curHead = ptr[0];
        if (curTail == curHead) {
            return false; // 空
        }
        elem = ptr[ELEM_OFFSET + (curTail % RING_BUF_SIZE)];
        ptr[1] = curTail + 1;
        __asm__ volatile("DSB #0");
        return true;
#else
        (void)addr;
        (void)elem;
        return false;
#endif
    }

    static inline uint32_t Size(DrcoSsbufSPSCQueue* addr)
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
using DrcoMixHubC2VReadyQueue = DrcoSsbufSPSCQueue<6>;

// ssbuf 段（__ssbuf__ 专用地址空间，段基址 0 合法）上的 C2V 通信状态：
// 每个 AIC 块两个 SPSC 队列（配对 V0/V1 各消费一个）
struct DrcoSsbufState {
    DrcoMixHubC2VReadyQueue mixHubC2VReadyQueue[2];
};

/* per core pending queue mechanism can be used in both codr and cudr */
struct PerCorePendingQueue {
    uint32_t head;
    uint32_t tail;
    uint32_t size;
    LeafTaskId taskList[0];
#ifdef __TILE_FWK_HOST__
    PerCorePendingQueue() : head(0), tail(0), size(0) {}

    void UnsafeEnqueue(LeafTaskId task) { taskList[size++] = task; }
#endif
};

struct DrcoLocalReadyQueue {
    uint32_t head;
    uint8_t pad[64 - sizeof(uint32_t)];
    uint32_t tail;
    uint8_t pad2[64 - sizeof(uint32_t)];
    uint32_t size;
    LeafTaskId taskList[0];
#ifdef __TILE_FWK_HOST__
    explicit DrcoLocalReadyQueue(uint32_t capacity) : head(0), tail(0), size(capacity) {}
#endif
};

// 组大小 = 组内核数 = 矩阵行列维度。LGS 4→8 四组消融定稿：矩阵 8×8=256B=4 cachelines
// + 槽位 ×4，push 批内逐槽 p50 2.8→1.7us、p90 11.5→6.5us（-40%）
constexpr uint32_t LOCAL_GROUP_SIZE = 8;

// LocalMatrix: 组内 N*N 通信矩阵（N = LOCAL_GROUP_SIZE，与 localReadyQueueArray 的分组一一对应）。
// 数组按类型内本地编号索引：AIC 核（blockIdx < nrValidAic）group = blockIdx / N，
// AIV 核 group = (blockIdx - nrValidAic) / N；行/列 = 本地编号 % N。
// push 依次遍历自己的一行，pop 依次遍历自己的一列，无游标；slot == 0 表示空闲，非 0 为编码后的任务。
// validCoreNum：本组内该类型核数（host 分配时写入）——列 [0, validCoreNum) 均有消费者核，
// push 只写这些列；0 表示本组无该类型核，禁止写入
template <typename T, unsigned N>
struct DrcoLocalReadyMatrixBase {
    enum { Size = N };
    typedef T ElementType;

    uint32_t validCoreNum;
    uint8_t pad[64 - sizeof(uint32_t)];

    T taskList[N][N];
#ifdef __TILE_FWK_HOST__
    explicit DrcoLocalReadyMatrixBase(uint32_t validCoreCount) : validCoreNum(validCoreCount)
    {
        for (uint32_t i = 0; i < N; i++) {
            for (uint32_t j = 0; j < N; j++) {
                taskList[i][j] = 0;
            }
        }
    }
#endif
};
using DrcoLocalReadyMatrix = DrcoLocalReadyMatrixBase<LeafTaskId, LOCAL_GROUP_SIZE>;

#define DRCO_ENCODE_TASK(task) ((task) + 1)
#define DRCO_DECODE_TASK(task) ((task) - 1)

} // namespace npu::tile_fwk

#endif
