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

namespace npu::tile_fwk::dynamic {

class AicpuTaskManager {
public:
    enum TaskType {
        SHMEM_WAIT_UNTIL = 0,
        TASK_TYPE_NUM,
    };
    using InitCallBack = std::function<void(DynDeviceTask *)>;
    using EnqueueOpCallBack = std::function<void(uint64_t, const npu::tile_fwk::dynamic::DevRelocVector<int32_t> &)>;
    using PollCompletedCallBack = std::function<void(std::vector<uint64_t> &)>;

    inline void TaskCallBackRegister() {}

    AicpuTaskManager() {
        TaskCallBackResigter<npu::tile_fwk::Distributed::ShmemWaitUntil>(TaskType::SHMEM_WAIT_UNTIL, shmemWaitUntil_);
    };
    ~AicpuTaskManager() {};

    template <typename T>
    inline void TaskCallBackResigter(TaskType taskType, T &obj) {
        auto index = static_cast<uint32_t>(taskType);
        initCallBack_[index] = std::bind(&T::Init, &obj, std::placeholders::_1);
        enqueueOpCallBack_[index] =
            std::bind(&T::EnqueueOp, &obj, std::placeholders::_1, std::placeholders::_2);
        pollCompletedCallBack_[index] = std::bind(&T::PollCompleted, &obj, std::placeholders::_1);
    }

    // 每个AICPU都会调用
    inline void TaskEnqueue(uint64_t taskId) {
        ReadyQueueLock();
        readyQueue_->elem[readyQueue_->tail] = taskId;
        readyQueue_->tail += 1;
        ReadyQueueUnLock();
    }

    // 仅AICPU_0会调用
    void Init(DynDeviceTask *deviceTask) {
        curDevTask_ = deviceTask;
        funcDataList_ = reinterpret_cast<DynFuncData*>(&deviceTask->GetDynFuncDataList()->At(0));
        readyQueue_ = reinterpret_cast<ReadyCoreFunctionQueue *>(deviceTask->devTask.readyAicpuFunctionQue);
        for (auto &init : initCallBack_) {
            init(deviceTask);
        }
    }

    // 仅AICPU_0会调用
    inline uint64_t  TaskProcess() {
        if (__atomic_load_n(&readyQueue_->tail, __ATOMIC_RELAXED) == __atomic_load_n(&readyQueue_->head, __ATOMIC_RELAXED)) {
            return 0;
        }
        ReadyQueueLock();
        uint64_t taskIdx = readyQueue_->head;
        uint64_t taskCount = readyQueue_->tail - readyQueue_->head;
        readyQueue_->head += taskCount;
        ReadyQueueUnLock();

        for (uint32_t i = 0; i < taskCount; ++i) {
            TaskDispatch(readyQueue_->elem[taskIdx + i]);
        }
        return taskCount;
    }

    inline std::vector<uint64_t> TaskPoll() {
        std::vector<uint64_t> completed;
        for (auto &pollCompleted : pollCompletedCallBack_) {
            pollCompleted(completed);
        }
        return completed;
    }

    inline bool Finished() {
        ReadyQueueLock();
        auto fin = readyQueue_->head == readyQueue_->tail;
        ReadyQueueUnLock();
        return fin;
    }

private:
    inline void ReadyQueueLock() {
        while (!__sync_bool_compare_and_swap(&readyQueue_->lock, 0, 1))
            ;
    }

    inline void ReadyQueueUnLock() {
        while (!__sync_bool_compare_and_swap(&readyQueue_->lock, 1, 0))
            ;
    }

    inline TaskType GetTaskType(uint64_t taskId) {
        auto funcId = FuncID(taskId);
        auto opIndex = TaskID(taskId);
        auto callList = curDevTask_->dynFuncDataCacheList[funcId].calleeList;
        auto &code = curDevTask_->aicpuLeafBinary[callList[opIndex]].aicpuLeafCode;
        auto taskType = TaskType::TASK_TYPE_NUM;
        switch (code[0]) {
            case static_cast<uint32_t>(Opcode::OP_SHMEM_WAIT_UNTIL): taskType = TaskType::SHMEM_WAIT_UNTIL; break;
            default: break;
        }
        return taskType;
    }

    inline void TaskDispatch(uint64_t taskId) {
        auto taskType = GetTaskType(taskId);
        if (taskType < TaskType::TASK_TYPE_NUM) {
            auto enqueueOp = enqueueOpCallBack_[static_cast<uint64_t>(taskType)];
            auto funcId = FuncID(taskId);
            auto opIndex = TaskID(taskId);
            auto callList = curDevTask_->dynFuncDataCacheList[funcId].calleeList;
            auto &code = curDevTask_->aicpuLeafBinary[callList[opIndex]].aicpuLeafCode;
            enqueueOp(taskId, code);
        }
    }

    ReadyCoreFunctionQueue *readyQueue_{nullptr};

    npu::tile_fwk::Distributed::ShmemWaitUntil shmemWaitUntil_;

    std::array<InitCallBack, TaskType::TASK_TYPE_NUM> initCallBack_;
    std::array<EnqueueOpCallBack, TaskType::TASK_TYPE_NUM> enqueueOpCallBack_;
    std::array<PollCompletedCallBack, TaskType::TASK_TYPE_NUM> pollCompletedCallBack_;

    DynDeviceTask *curDevTask_;
    DynFuncData *funcDataList_;
};
} // namespace npu::tile_fwk
