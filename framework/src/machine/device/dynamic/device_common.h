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
 * \file device_machine.h
 * \brief
 */

#pragma once
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include "device_utils.h"
#include "machine/utils/dynamic/spsc_queue.h"
#include "machine/utils/dynamic/dev_workspace.h"
#include "machine/utils/dynamic/device_task.h"
#include "tilefwk/core_func_data.h"
#include "tilefwk/aicpu_common.h"

namespace npu::tile_fwk::dynamic {

inline uint32_t CalcSchAicpuNumByBlockDim(uint32_t blockDim, uint32_t aiCpuNum) {
    uint32_t maxScheCore = aiCpuNum - dynamic::MAX_OTHER_AICPU_NUM;
    if (blockDim > (maxScheCore - 1) * dynamic::MAX_MNG_AICORE_AVG_NUM) {
        return maxScheCore;
    }

    if (blockDim % dynamic::MAX_MNG_AICORE_AVG_NUM == 0) {
        return blockDim / dynamic::MAX_MNG_AICORE_AVG_NUM;
    }

    return blockDim / dynamic::MAX_MNG_AICORE_AVG_NUM + 1;
}

const uint32_t AICORE_TYPE_NUM = 2;
const int DEVICE_MAX_AICPU_NUM = 7;

struct DeviceTaskCtrl {
    int taskType{0};
    uint64_t taskId{0};
    DeviceTask *devTask{nullptr};
    uint64_t initAicFuncNum{0};
    uint64_t initAivFuncNum{0};
    uint64_t finishedAicFunctionCnt{0}; // 所有aicpu处理完成的aic function个数，多线程增加修改
    uint64_t finishedAivFunctionCnt{0}; // 所有aicpu处理完成的aiv function个数，多线程增加修改
    uint64_t finishedAicpuFunctionCnt{0}; // 所有aicpu处理完成的aicpu function个数，多线程增加修改
    uint64_t finishedHubFunctionCnt{0}; // 所有aicpu处理完成的hub function个数，多线程增加修改
    // 这些原子变量跨进程了，不能sche与ctrl间两边同时写
    std::atomic<uint64_t> finishedFunctionCnt{0};
    std::atomic<bool> runFlag{false};
    std::atomic<int> runcnt{0};
    void *ctx{nullptr};
    int retCode{0};
    std::atomic<bool> isAicpuIdle[AICORE_TYPE_NUM][MAX_SCHEDULE_AICPU_NUM];
    bool isFirstDevTask{false};

    inline bool IsNotFree() { return runFlag.load(std::memory_order_acquire); }

    void PutTask(int ret) {
        if (ret != 0)
            retCode = ret;

        // sync point, ensure all aiore_manager threads task finished
        int cnt = runcnt.fetch_sub(1, std::memory_order_acq_rel);
        if (cnt == 1) {
            runFlag.store(false, std::memory_order_release); // set finish
            auto *dynTask = reinterpret_cast<DynDeviceTask*>(devTask);
            dynTask->taskStageAllocMem.canFree.store(true);
        } else {
            // wait finish
            while (runFlag.load(std::memory_order_acquire)) {}
        }
    }
};

#define ALIGN_UP(val, align)            (((val) + (align) - 1) & ~((align) - 1))

const uint64_t DEV_ARGS_SIZE = 1024;  // sizeof(DevStartArgs) is enough, tmp for test GE graph
constexpr uint32_t DEFAULT_QUEUE_SIZE = 64;
const uint64_t DEVICE_TASK_CTRL_SIZE = ALIGN_UP((MAX_DEVICE_TASK_NUM * sizeof(DeviceTaskCtrl)), 512);
const uint64_t DEVICE_TASK_QUEUE_SIZE = sizeof(SPSCQueue<DeviceTaskCtrl *, DEFAULT_QUEUE_SIZE>);
const uint64_t DEVICE_SHM_SIZE = DEV_ARGS_SIZE + DEVICE_TASK_CTRL_SIZE;
} // namespace npu::tile_fwk