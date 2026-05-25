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
 * \file aicpu_task_manager.h
 * \brief
 */

#pragma once

#include <functional>
#include <atomic>
#include <vector>
#include <array>
#include <malloc.h>
#include <queue>

#include "machine/utils/dynamic/dev_workspace.h"
#include "machine/device/distributed/common.h"
#include "machine/device/distributed/shmem_wait_until.h"
#include "machine/utils/machine_ws_intf.h"
#include "interface/operation/opcode.h"
#include "interface/utils/common.h"
#include "tilefwk/error_code.h"

namespace npu::tile_fwk::dynamic {

class AicpuTaskManager {
public:
    enum TaskType {
        SHMEM_WAIT_UNTIL = 0,
        TASK_TYPE_NUM,
    };

    AicpuTaskManager() {};
    ~AicpuTaskManager() {};

    // 每个AICPU都会调用
    inline void TaskEnqueue(uint64_t taskId)
    {
        bool res = readyQueue_->TryEnqueue(taskId);
        DEV_ASSERT(SchedErr::READY_QUEUE_OVERFLOW, res); // fail on queue overflow
    }

    // 仅AICPU_0会调用
    inline void InitDeviceArgs(DeviceArgs* deviceArgs)
    {
        sharedBuffer_ = deviceArgs->sharedBuffer;
        aicNum_ = deviceArgs->nrAic;
        aivNum_ = deviceArgs->nrAiv;
        archInfo_ = deviceArgs->archInfo;
    }

    // 仅AICPU_0会调用
    inline int32_t Init(DynDeviceTask* deviceTask, bool profSwitch)
    {
        curDevTask_ = deviceTask;
        funcDataList_ = reinterpret_cast<DynFuncData*>(&deviceTask->GetDynFuncDataList()->At(0));
        readyQueue_ = reinterpret_cast<ReadyCoreFunctionQueue*>(deviceTask->devTask.readyAicpuFunctionQue);
        shmemWaitUntil_.Init(deviceTask);
        if (profSwitch) {
            KernelArgs* args = (KernelArgs*)(sharedBuffer_ + (aicNum_ + aivNum_) * SHARED_BUFFER_SIZE);
            aicpuTaskStat_ = (Metrics*)(args->shakeBuffer[SHAK_BUF_DFX_DATA_INDEX]);
        }
        return PrepareAicpuTask();
    }

    // 仅AICPU_0会调用
    inline int32_t TaskProcess(uint64_t& taskCount)
    {
        if (readyQueue_->UnsafeAtomicSize() == 0) {
            return DEVICE_MACHINE_OK;
        }
        auto tasksRange = readyQueue_->DequeueAll();
        taskCount = tasksRange.second - tasksRange.first;
        for (auto it = tasksRange.first; it != tasksRange.second; ++it) {
            auto ret = TaskDispatch(*it);
            if (ret != DEVICE_MACHINE_OK) {
                return ret;
            }
        }
        return DEVICE_MACHINE_OK;
    }

    inline int32_t TaskPoll(AiCoreManager* aiCoreManager) { return shmemWaitUntil_.PollCompleted(aiCoreManager); }

    inline bool Finished() { return shmemWaitUntil_.runingTaskQueue_.IsEmpty(); }

    inline int32_t SyncAicpuTaskFinish(AiCoreManager* aiCoreManager)
    {
        TIMEOUT_CHECK_INIT(archInfo_, TIMEOUT_10SEC);
        while (!Finished()) {
            auto ret = TaskPoll(aiCoreManager);
            if (unlikely(ret != DEVICE_MACHINE_OK)) {
                return ret;
            }
            __PYPTO_TIMEOUT_CHECK_EXIT_ONLY(DistributedErrorCode::AICPU_TASK_TIMEOUT,
                return DEVICE_MACHINE_TIMEOUT_SYNC_AICPU_FINISH,
                "#sche.task.end.sync: SyncAicpuTaskFinish.");
        }
        return DEVICE_MACHINE_OK;
    }

private:
    inline TaskType GetTaskType(uint64_t taskId)
    {
        auto funcId = FuncID(taskId);
        auto opIndex = TaskID(taskId);
        auto callList = curDevTask_->dynFuncDataCacheList[funcId].calleeList;
        auto& code = curDevTask_->aicpuLeafBinary[callList[opIndex]].aicpuLeafCode;
        auto taskType = TaskType::TASK_TYPE_NUM;
        switch (code[0]) {
            case static_cast<uint32_t>(Opcode::OP_SHMEM_WAIT_UNTIL):
                taskType = TaskType::SHMEM_WAIT_UNTIL;
                break;
            default:
                break;
        }
        return taskType;
    }

    inline int32_t TaskDispatch(uint64_t taskId)
    {
        int32_t ret = DEVICE_MACHINE_OK;
        auto taskType = GetTaskType(taskId);
        DEV_VERBOSE_DEBUG("Dispatch aicpu task %lu.", taskId);
        if (taskType < TaskType::TASK_TYPE_NUM) {
            TaskStat* taskStat = nullptr;
            if (aicpuTaskStat_ != nullptr) {
                taskStat = &(aicpuTaskStat_->tasks[aicpuTaskStat_->taskCount]);
                ++aicpuTaskStat_->taskCount;
            }
            ret = shmemWaitUntil_.EnqueueOp(taskId, taskStat);
        }
        return ret;
    }

    inline int32_t PrepareAicpuTask()
    {
        for (uint64_t funcId = 0; funcId < curDevTask_->dynFuncDataCacheListSize; ++funcId) {
            auto callList = curDevTask_->dynFuncDataCacheList[funcId].calleeList;
            for (size_t opIndex = 0; opIndex < curDevTask_->dynFuncDataCacheList[funcId].devFunc->GetOperationSize();
                 ++opIndex) {
                auto coreType = curDevTask_->cceBinary[callList[opIndex]].coreType;
                if (unlikely(coreType != static_cast<int>(MachineType::AICPU))) {
                    continue;
                }
                uint32_t taskId = MakeTaskID(funcId, opIndex);
                auto& code = curDevTask_->aicpuLeafBinary[callList[opIndex]].aicpuLeafCode;
                auto ret = shmemWaitUntil_.PrepareTask(taskId, code);
                if (ret != DEVICE_MACHINE_OK) {
                    return ret;
                }
            }
        }
        return DEVICE_MACHINE_OK;
    }

    ReadyCoreFunctionQueue* readyQueue_{nullptr};

    ArchInfo archInfo_{ArchInfo::DAV_2201};
    npu::tile_fwk::Distributed::ShmemWaitUntilImpl shmemWaitUntil_;
    DynDeviceTask* curDevTask_;
    DynFuncData* funcDataList_;
    Metrics* aicpuTaskStat_;
    uint64_t sharedBuffer_;
    uint32_t aicNum_;
    uint32_t aivNum_;
};
} // namespace npu::tile_fwk::dynamic
