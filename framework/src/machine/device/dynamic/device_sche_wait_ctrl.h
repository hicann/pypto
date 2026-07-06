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
    CTRL_WAIT_ARBITRATING = -2, // spin 等待中
    CTRL_WAIT_FAILED = -1,      // scheNum==1 时超时
    CTRL_WAIT_UNSET = 0,        // 未决策
    CTRL_WAIT_OK = 1,           // ctrl 已起来，无需丢弃
    CTRL_WAIT_DROPPED = 2,      // 需丢弃一个线程
};

// spin 等待 ctrl 线程起来（ctrlRound >= scheRound），成功返回 true，超时返回 false
static inline bool WaitCtrlRoundReady(ArchInfo archInfo, std::atomic<uint64_t>& ctrlRound, std::atomic<uint64_t>& scheRound)
{
    uint64_t waitTimeout = (archInfo == ArchInfo::DAV_2201) ? TIMEOUT_A2A3_100US : TIMEOUT_A5_50US;
    if (!IsDeviceMode()) {
        waitTimeout = (archInfo == ArchInfo::DAV_2201) ? TIMEOUT_A2A3_1SEC : TIMEOUT_A5_1SEC;
    }
    TIMEOUT_CHECK_INIT(archInfo, waitTimeout);
    while (ctrlRound.load(std::memory_order_acquire) <= scheRound.load(std::memory_order_acquire)) {
        __PYPTO_TIMEOUT_CHECK_WARN_EXIT(return false, "#ctrl.wait: ctrl thread not start.");
    }
    return true;
}

// CAS 抢占成功者 spin 等 ctrl 线程起来（ctrlRound >= scheRound），超时则丢弃
// ctrl 在 EntrySplittedStreamCtrl 第一行 ++ctrlRound_，比 RingBuffer 初始化快得多
static inline bool ResolveCtrlDecision(ArchInfo archInfo, int scheNum, std::atomic<int>& ctrlWaitLevel, int& outLevel,
    std::atomic<uint64_t>& ctrlRound, std::atomic<uint64_t>& scheRound)
{
    int expected = CTRL_WAIT_UNSET;
    if (!ctrlWaitLevel.compare_exchange_strong(expected, CTRL_WAIT_ARBITRATING,
        std::memory_order_acq_rel, std::memory_order_acquire)) {
        return false;
    }
    if (WaitCtrlRoundReady(archInfo, ctrlRound, scheRound)) {
        outLevel = CTRL_WAIT_OK;
    } else {
        outLevel = (scheNum == 1) ? CTRL_WAIT_FAILED : CTRL_WAIT_DROPPED;
    }
    ctrlWaitLevel.store(outLevel, std::memory_order_release);
    return true;
}

// follower spin 等待决策终态 level
static inline int WaitCtrlDecision(ArchInfo archInfo, std::atomic<int>& ctrlWaitLevel, int& outLevel)
{
    uint64_t waitTimeout = (archInfo == ArchInfo::DAV_2201) ? TIMEOUT_A2A3_200US : TIMEOUT_A5_50US;
    if (!IsDeviceMode()) {
        waitTimeout = (archInfo == ArchInfo::DAV_2201) ? TIMEOUT_A2A3_1SEC : TIMEOUT_A5_1SEC;
    }
    TIMEOUT_CHECK_INIT(archInfo, waitTimeout);
    int level = ctrlWaitLevel.load(std::memory_order_acquire);
    while (level == CTRL_WAIT_UNSET || level == CTRL_WAIT_ARBITRATING) {
        __PYPTO_TIMEOUT_CHECK_WARN_EXIT(return DEVICE_MACHINE_ERROR, "Wait for ctrl thread timeout, ctrl wait level: %d", level);
        level = ctrlWaitLevel.load(std::memory_order_acquire);
    }
    outLevel = level;
    return DEVICE_MACHINE_OK;
}

// 决策后统一分发：
//   FAILED → ERROR
//   OK → 直接返回（后续由 device_sche.h 统一 ScheWait）
//   DROPPED → 减一个 sche，被丢弃者退出，存活者直接返回（后续由 device_sche.h 统一 ScheWait）
static inline int ApplyCtrlDecision(int& curThreadIdx, int& arbitratedScheNum, int level)
{
    if (level == CTRL_WAIT_FAILED) {
        DEV_ERROR(SchedErr::WAIT_CTRL_TIMEOUT, "Wait for ctrl thread timeout,"
            "only one sched thread exists, id is %d.", curThreadIdx);
        return DEVICE_MACHINE_ERROR;
    }
    if (level == CTRL_WAIT_OK) {
        return DEVICE_MACHINE_OK;
    }
    // CTRL_WAIT_DROPPED：arbitratedScheNum 全局一致地减 1
    int scheNum = arbitratedScheNum;
    arbitratedScheNum -= 1;
    if (curThreadIdx == scheNum) {
        DEV_INFO("Waiting for ctrl thread timed out, release sched thread %d.", curThreadIdx);
        curThreadIdx = -1;
        return DEVICE_MACHINE_OK;
    }
    DEV_INFO("Thread %d wait ctrl thread timeout, another sched thread released, arbitrated sche num: %d",
        curThreadIdx, arbitratedScheNum);
    return DEVICE_MACHINE_OK;
}

// 对于 Dav2201,通过 ctrlRound/scheRound 判断 ctrl 是否已起来，决定是否丢弃一个线程
// 对于其他类型, 直接返回 DEVICE_MACHINE_OK
static inline int WaitForCtrlDecision(ArchInfo archInfo, int& curThreadIdx, int& arbitratedScheNum, std::atomic<int>& ctrlWaitLevel,
    std::atomic<uint64_t>& ctrlRound, std::atomic<uint64_t>& scheRound)
{
    if (archInfo != ArchInfo::DAV_2201) {
        return DEVICE_MACHINE_OK;
    }

    int level = CTRL_WAIT_UNSET;
    if (!ResolveCtrlDecision(archInfo, arbitratedScheNum, ctrlWaitLevel, level, ctrlRound, scheRound)) {
        int ret = WaitCtrlDecision(archInfo, ctrlWaitLevel, level);
        if (ret != DEVICE_MACHINE_OK) {
            return ret;
        }
    }
    return ApplyCtrlDecision(curThreadIdx, arbitratedScheNum, level);
}
} // namespace npu::tile_fwk::dynamic
