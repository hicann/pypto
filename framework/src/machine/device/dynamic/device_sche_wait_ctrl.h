/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once

#include "device_common.h"
#include "device_utils.h"
#include "machine/utils/device_log.h"

namespace npu::tile_fwk::dynamic {

enum CtrlWaitLevel : int {
    CTRL_WAIT_ARBITRATING = -2, // SchWait 中
    CTRL_WAIT_FAILED = -1,      // retry 失败
    CTRL_WAIT_UNSET = 0,        // 未决策
    CTRL_WAIT_OK = 1,           // 首次 ScheWait 成功，无需丢弃
    CTRL_WAIT_DROPPED = 2,      // 需丢弃一个线程
};

static inline int ScheWait(const DevAscendProgram* devProg, ArchInfo archInfo)
{
    uint64_t scheWaitTimeout = (archInfo == ArchInfo::DAV_2201) ? TIMEOUT_A2A3_1SEC : TIMEOUT_A5_1SEC;
    TIMEOUT_CHECK_INIT(devProg->devArgs.archInfo, scheWaitTimeout);

    while (unlikely(!devProg->runtimeDataRingBufferInited)) {
        RuntimeYield(0);
        __PYPTO_TIMEOUT_CHECK_WARN_EXIT(return DEVICE_MACHINE_ERROR, "#sche.wait: RingBuffer init.");
    }
    const RuntimeDataRingBufferHead* ringBufferHead = devProg->GetRuntimeDataList();

    while (unlikely(ringBufferHead->Empty())) {
        RuntimeYield(0);
        __PYPTO_TIMEOUT_CHECK_WARN_EXIT(return DEVICE_MACHINE_ERROR, "#sche.wait: RingBuffer data.");
    }
    return DEVICE_MACHINE_OK;
}

// CAS 抢占成功者执行第一次 ScheWait 并发布决策 level；CAS 失败返回 false
static inline bool ResolveCtrlDecision(const DevAscendProgram* devProg, int scheNum,
    std::atomic<int>& ctrlWaitLevel, int& outLevel, ArchInfo archInfo)
{
    int expected = CTRL_WAIT_UNSET;
    if (!ctrlWaitLevel.compare_exchange_strong(expected, CTRL_WAIT_ARBITRATING,
        std::memory_order_acq_rel, std::memory_order_acquire)) {
        return false;
    }
    int firstRet = ScheWait(devProg, archInfo);
    if (firstRet == DEVICE_MACHINE_OK) {
        outLevel = CTRL_WAIT_OK;
    } else if (scheNum == 1) {
        outLevel = CTRL_WAIT_FAILED;
    } else {
        outLevel = CTRL_WAIT_DROPPED;
    }
    ctrlWaitLevel.store(outLevel, std::memory_order_release);
    return true;
}

// 等待第一个 sche 线程 ScheWait 结果
// 超时必须 >= ScheWait 的超时（TIMEOUT_A2A3_1SEC），否则 follower 会在 leader 发布决策前超时退出
static inline int WaitCtrlDecision(ArchInfo archInfo, std::atomic<int>& ctrlWaitLevel, int& outLevel)
{
    uint64_t scheWaitTimeout = TIMEOUT_A2A3_2SEC;
    TIMEOUT_CHECK_INIT(archInfo, scheWaitTimeout);
    int level = ctrlWaitLevel.load(std::memory_order_acquire);
    while (level == CTRL_WAIT_UNSET || level == CTRL_WAIT_ARBITRATING) {
        __PYPTO_TIMEOUT_CHECK_WARN_EXIT(return DEVICE_MACHINE_ERROR, "#cur wait ctrl decision failed.");
        level = ctrlWaitLevel.load(std::memory_order_acquire);
    }
    outLevel = level;
    return DEVICE_MACHINE_OK;
}

// 决策后统一分发：
//   FAILED → ERROR；OK → 直接返回；DROPPED → 减一个 sche，被丢弃者退出，存活者 retry
static inline int ApplyCtrlDecision(const DevAscendProgram* devProg, int& curThreadIdx, int& arbitratedScheNum, 
    int level, ArchInfo archInfo)
{
    if (level == CTRL_WAIT_FAILED) {
        return DEVICE_MACHINE_ERROR;
    }
    if (level == CTRL_WAIT_OK) {
        return DEVICE_MACHINE_OK;
    }
    // CTRL_WAIT_DROPPED：arbitratedScheNum 全局一致地减 1
    int scheNum = arbitratedScheNum;
    arbitratedScheNum -= 1;
    if (curThreadIdx == scheNum) {
        curThreadIdx = -1;
        return DEVICE_MACHINE_OK;
    }
    // 存活线程 retry，等待 ctrl 就绪
    int retryRet = ScheWait(devProg, archInfo);
    if (retryRet != DEVICE_MACHINE_OK) {
        return DEVICE_MACHINE_ERROR;
    }
    return DEVICE_MACHINE_OK;
}

// 对于 Dav2201,第一次等待Ctrl初始化超时丢弃一个Sche线程, 再次等待。
// 对于其他类型, 只等待一次Ctrl, 超时则返回 ERROR
// 通过全局 ctrlWaitLevel 保证所有线程对"是否丢弃/arbitratedScheNum"达成一致：
//   CAS 抢占成功者做第一次 ScheWait 决策并发布；其余线程 spin 等待终态。
//   决策后统一分发：被丢弃线程(curThreadIdx==scheNum)退出不 retry，
//   其余存活线程 retry ScheWait 等待 ctrl 就绪。
static inline int WaitForCtrlDecision(const DevAscendProgram* devProg, ArchInfo archInfo, int& curThreadIdx,
    int& arbitratedScheNum, std::atomic<int>& ctrlWaitLevel)
{
    if (archInfo != ArchInfo::DAV_2201) {
        return ScheWait(devProg, archInfo);
    }

    int level = CTRL_WAIT_UNSET;
    if (!ResolveCtrlDecision(devProg, arbitratedScheNum, ctrlWaitLevel, level, archInfo)) {
        int ret = WaitCtrlDecision(archInfo, ctrlWaitLevel, level);
        if (ret != DEVICE_MACHINE_OK) {
            return ret;
        }
    }
    return ApplyCtrlDecision(devProg, curThreadIdx, arbitratedScheNum, level, archInfo);
}

} // namespace npu::tile_fwk::dynamic

