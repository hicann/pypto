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
 * \file device_sche_alloc_thread.h
 * \brief
*/

#pragma once

#include <algorithm>
#include "device_common.h"
#include "device_utils.h"
#include "machine/utils/device_log.h"

namespace npu::tile_fwk::dynamic {

constexpr int DAV3510_CPUS_PER_CLUSTER = 2;
constexpr int DAV3510_START_CPU_ID = 3;
constexpr int MAX_CPU_MASK_NUM = 16;

enum ArbitrationLevel : int {
    LEVEL_ARBITRATING = -2, // 仲裁线程已选出，仲裁结果尚未发布
    LEVEL_FAILED = -1,     // 资源完全不足，分配失败
    LEVEL_UNSET = 0,       // 仲裁尚未发布
    LEVEL_CROSS_DIE = 1,   // 跨 die 分配
    LEVEL_SAME_DIE = 2,    // 同 die 分配
    LEVEL_OPTIMAL = 3      // 同 die 同 cluster
};

struct DieMaskInfo {
    int die0MaxCpuid;       // die0 的最大 CPU ID
    uint64_t die0Boundary;  // die0 的边界 mask：(1ULL << (die0MaxCpuid+1)) - 1
    uint64_t die0Mask;      // die0 的可用 CPU mask：cpumask & die0Boundary
    uint64_t die1Mask;      // die1 的可用 CPU mask：cpumask & ~die0Boundary
    int die0Cnt;            // die0 可用 CPU 数量
    int die1Cnt;            // die1 可用 CPU 数量

    DieMaskInfo(int die0Max, uint64_t mask)
    {
        die0MaxCpuid = die0Max;
        die0Boundary = (1ULL << (die0MaxCpuid + 1)) - 1;
        die0Mask = mask & die0Boundary;
        die1Mask = mask & ~die0Boundary;
        die0Cnt = __builtin_popcount(die0Mask);
        die1Cnt = __builtin_popcount(die1Mask);
    }

    int TotalCnt() const
    {
        return die0Cnt + die1Cnt;
    }
};

inline uint64_t GetArbitTimeOutVal()
{
    uint64_t arbitTimeout = TIMEOUT_A5_50US;
    DEV_IF_DEBUG {
        arbitTimeout = TIMEOUT_A5_20MIN;
    }
    DEV_IF_INFO {
        arbitTimeout = TIMEOUT_A5_20MIN;
    }

    return arbitTimeout;
}

inline int WaitForCpuMaskReadyForArbitration(DeviceArgs* devArgs, int targetVal, std::atomic<uint64_t>& mask)
{
    uint64_t arbitTimeout = GetArbitTimeOutVal();
    TIMEOUT_CHECK_INIT(devArgs->archInfo, arbitTimeout);
    (void) timeout_map;

    while (__builtin_popcount(static_cast<uint32_t>(mask.load(std::memory_order_acquire))) < targetVal) {
        __PYPTO_TIMEOUT_CHECK_WARN_EXIT(return DEVICE_MACHINE_ERROR, "#cur alloc cpu failed.");
    }

    return DEVICE_MACHINE_OK;
}

inline int ComputeArbitrationLevel(DeviceArgs* devArgs, std::atomic<uint64_t>& cpumask)
{
    int die0CpuNum = static_cast<int>(devArgs->die0MaxCpuid) - DAV3510_START_CPU_ID + 1;
    int die0MaxCpuNum = ((die0CpuNum + DAV3510_CPUS_PER_CLUSTER - 1) /
        DAV3510_CPUS_PER_CLUSTER) * DAV3510_CPUS_PER_CLUSTER;
    int nrAicpu = static_cast<int>(devArgs->nrAicpu);

    int ret = WaitForCpuMaskReadyForArbitration(devArgs, nrAicpu, cpumask);
    if (ret == DEVICE_MACHINE_OK) {
        return LEVEL_OPTIMAL;
    }

    // 允许部分 die1 的 CPU 未就绪，但 die0 至少有足够的 cluster
    ret = WaitForCpuMaskReadyForArbitration(devArgs, die0MaxCpuNum + DAV3510_CPUS_PER_CLUSTER, cpumask);
    if (ret == DEVICE_MACHINE_OK) {
        return LEVEL_SAME_DIE;
    }

    // 只要满足最小调度需求即可，不考虑拓扑优化
    ret = WaitForCpuMaskReadyForArbitration(devArgs, devArgs->scheCpuNum, cpumask);
    if (ret == DEVICE_MACHINE_OK) {
        return LEVEL_CROSS_DIE;
    }

    return LEVEL_FAILED;
}

inline int WaitForArbitrationLevel(DeviceArgs* devArgs, std::atomic<int>& globalArbitrationLevel)
{
    uint64_t arbitTimeout = GetArbitTimeOutVal();
    TIMEOUT_CHECK_INIT(devArgs->archInfo, arbitTimeout);
    (void) timeout_map;

    int level = globalArbitrationLevel.load(std::memory_order_acquire);
    while (level == LEVEL_UNSET || level == LEVEL_ARBITRATING) {
        __PYPTO_TIMEOUT_CHECK_WARN_EXIT(return LEVEL_FAILED, "#cur alloc cpu arbitration wait failed.");
        level = globalArbitrationLevel.load(std::memory_order_acquire);
    }

    return level;
}

/**
 * 单线程仲裁：第一个到达的线程执行一次三级递进判定并发布全局级别，
 * 其余线程轻量等待后读取同一结果。
 */
inline int PerformArbitrationDav3510(DeviceArgs* devArgs, std::atomic<uint64_t>& cpumask,
    std::atomic<int>& globalArbitrationLevel)
{
    int expected = LEVEL_UNSET;
    if (globalArbitrationLevel.compare_exchange_strong(
        expected, LEVEL_ARBITRATING, std::memory_order_acq_rel, std::memory_order_acquire)) {
        int level = ComputeArbitrationLevel(devArgs, cpumask);
        globalArbitrationLevel.store(level, std::memory_order_release);
        return level;
    }

    return WaitForArbitrationLevel(devArgs, globalArbitrationLevel);
}

inline uint64_t SelectCpusForCluster(uint64_t dieMask, int needNum)
{
    if (needNum <= 0 || dieMask == 0) {
        return 0;
    }

    uint64_t selected = 0;
    int count = 0;

    // 第一阶段：优先选择相邻 CPU 对（cluster）
    // 遍历所有 CPU，查找 pairMask = 0b11 << cpu 的模式
    for (int cpu = 0; cpu < MAX_CPU_MASK_NUM && count < needNum - 1; cpu++) {
        // 3 ：连续 2 位为 1（例如 cpu=3 → pairMask=0b11000 = 3<<3）
        uint64_t pairMask = (3ULL << cpu);
        // 检查 pairMask 是否在 dieMask 中完整出现（cpu 和 cpu+1 都可用）
        if ((dieMask & pairMask) == pairMask) {
            // 将整个 pair 加入 selected
            selected |= pairMask;
            count += DAV3510_CPUS_PER_CLUSTER;  // 增加 2 个 CPU
            cpu++;  // 跳过 cpu+1，避免重复处理
        }
    }

    // 第二阶段：补充单个 CPU（当第一阶段未达到 needNum）
    // 选择剩余的单个可用 CPU（未形成 pair 的）
    for (int cpu = 0; cpu < MAX_CPU_MASK_NUM && count < needNum; cpu++) {
        // 检查：CPU 在 dieMask 中可用，且未被第一阶段选中
        if ((dieMask & (1ULL << cpu)) && !(selected & (1ULL << cpu))) {
            selected |= (1ULL << cpu);
            count++;
        }
    }

    return selected;
}

inline void CalculateDieScheNum(DeviceArgs* devArgs, const DieMaskInfo& info,
    int& die0ScheNum, int& die1ScheNum)
{
    die0ScheNum = 0;
    die1ScheNum = 0;
    int totalCnt = info.die0Cnt + info.die1Cnt;
    
    if (totalCnt > 0) {
        // 计算原始比例：按可用 CPU 数量分配
        int rawDie0ScheNum = static_cast<int>(devArgs->scheCpuNum) * info.die0Cnt / totalCnt;
        
        // 向上取整到 cluster 倍数：确保分配完整 cluster
        // 例如 rawDie0ScheNum=3 → (3+2-1)/2*2 = 4
        if (rawDie0ScheNum % DAV3510_CPUS_PER_CLUSTER != 0) {
            die0ScheNum = ((rawDie0ScheNum + DAV3510_CPUS_PER_CLUSTER - 1) /
                DAV3510_CPUS_PER_CLUSTER) * DAV3510_CPUS_PER_CLUSTER;
        } else {
            die0ScheNum = rawDie0ScheNum;
        }

        // 限制不超过可用数量和请求数量（防止分配超出实际可用）
        die0ScheNum = std::min(die0ScheNum, info.die0Cnt);
        die0ScheNum = std::min(die0ScheNum, static_cast<int>(devArgs->scheCpuNum));
        
        // die1 的分配数量：剩余部分
        die1ScheNum = static_cast<int>(devArgs->scheCpuNum) - die0ScheNum;
    }
}

inline int CalculateThreadIdx(int cpu, int die0ScheNum, uint64_t die0Selected, uint64_t die1Selected,
    const DieMaskInfo& info)
{
    int threadIdx = 0;

    if (cpu <= info.die0MaxCpuid) {
        // die0：统计当前 CPU 在 die0Selected 中的位序
        // mask：从 bit 0 到 cpu 的 mask（包含 cpu）
        uint64_t mask = (1ULL << (cpu + 1)) - 1;
        // tmp：die0Selected 中 cpu 及之前的位
        uint64_t tmp = die0Selected & mask;
        // popcount：统计 tmp 中 1 的数量，得到 threadIdx
        threadIdx = __builtin_popcount(tmp);
    } else {
        // die1：die0ScheNum + 在 die1Selected 中的位序
        uint64_t mask = (1ULL << (cpu + 1)) - 1;
        uint64_t tmp = die1Selected & mask;
        int threadIdxInDie = __builtin_popcount(tmp);
        threadIdx = die0ScheNum + threadIdxInDie;
    }
    
    return threadIdx;
}

/**
 * DAV3510 线程分配主流程（核心实现）
 * 
 * 流程步骤：
 * 1. 非设备模式或 die0MaxCpuid=0：直接递增 threadIdx（无仲裁）
 * 2. 更新 cpumask：标记当前 CPU 就绪
 * 3. 单线程仲裁：第一个到达的线程计算全局级别，其余线程等待并读取
 * 4. LEVEL_FAILED：报错
 * 5. LEVEL_CROSS_DIE：退化策略，直接递增 threadIdx
 * 6. LEVEL_OPTIMAL/SAME_DIE：
 *    - 计算 die 分配数量（按比例）
 *    - 选择 CPU（优先 cluster）
 *    - 计算 threadIdx（逻辑索引）
 * 7. 未选中 CPU：threadIdx=-1（该 CPU 不参与本次调度）
 * 
 */
inline int AllocThreadIdxForDav3510Impl(DeviceArgs* devArgs, int cpu, int& curThreadIdx,
    std::atomic<int>& threadIdx, std::atomic<uint64_t>& cpumask, std::atomic<int>& globalArbitrationLevel)
{
#ifndef __DEVICE__
    curThreadIdx = ++threadIdx;
    return DEVICE_MACHINE_OK;
#endif
    if (devArgs->die0MaxCpuid == 0) {
        curThreadIdx = ++threadIdx;
        return DEVICE_MACHINE_OK;
    }

    cpumask.fetch_or(1ULL << cpu, std::memory_order_release);
    // 执行三级仲裁，确定分配级别
    int level = PerformArbitrationDav3510(devArgs, cpumask, globalArbitrationLevel);
    
    // 处理仲裁失败
    if (level == LEVEL_FAILED) {
        DEV_ERROR(ThreadErr::THREAD_CPU_ALLOC_FAILED, 
            "#sche.thread.arbitration: Currently aicpus resources are insufficient. "
            "Please use the launchSchedAicpuNum frontend interface to reduce the number of aicpus being used.");
        return DEVICE_MACHINE_ERROR;
    }

    if (level == LEVEL_CROSS_DIE) {
        curThreadIdx = ++threadIdx;
        return DEVICE_MACHINE_OK;
    }

    // 构建 DieMaskInfo：分离 die0 和 die1 的可用 CPU
    DieMaskInfo info(static_cast<int>(devArgs->die0MaxCpuid), cpumask.load(std::memory_order_acquire));
    
    // 计算各 die 应分配的 CPU 数量（按比例）
    int die0ScheNum = 0;
    int die1ScheNum = 0;
    CalculateDieScheNum(devArgs, info, die0ScheNum, die1ScheNum);

    // 从各 die 中选择 CPU（优先完整 cluster）
    uint64_t die0Selected = SelectCpusForCluster(info.die0Mask, die0ScheNum);
    uint64_t die1Selected = SelectCpusForCluster(info.die1Mask, die1ScheNum);
    uint64_t selectedMask = die0Selected | die1Selected;
    int actualDie0Count = __builtin_popcount(die0Selected);
    int actualDie1Count = __builtin_popcount(die1Selected);
    if (actualDie0Count < die0ScheNum || actualDie1Count < die1ScheNum) {
        DEV_WARN("#sche.thread.alloc: Insufficient CPUs: die0 requested=%d actual=%d, "
            "die1 requested=%d actual=%d", die0ScheNum, actualDie0Count, die1ScheNum, actualDie1Count);
    }

    if (!(selectedMask & (1ULL << cpu))) {
        curThreadIdx = -1;
        return DEVICE_MACHINE_OK;
    }

    curThreadIdx = CalculateThreadIdx(cpu, die0ScheNum, die0Selected, die1Selected, info);
    threadIdx.store(curThreadIdx, std::memory_order_release);
    DEV_INFO("Thread alloc success: physicalCpu=%d, threadIdx=%d.", cpu, curThreadIdx);
    return DEVICE_MACHINE_OK;
}

} // namespace npu::tile_fwk::dynamic