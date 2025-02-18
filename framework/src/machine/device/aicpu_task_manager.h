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
#include "distributed/depend_on.h"
#include "distributed/comm_wait_flag.h"
#include "machine/utils/machine_ws_intf.h"
#include "tilefwk/core_func_data.h"
#include "interface/operation/opcode.h"
#include "interface/utils/common.h"
#include <functional>
#include <atomic>
#include <vector>
#include <array>
#include <malloc.h>

namespace npu::tile_fwk {
constexpr uint32_t AICPU_QUEUE_SIZE = 1024;

class AicpuTaskManager {
public:
    enum TaskType {
        COMM_WAIT_FLAG = 0,
        DEPEND_ON,
        TASK_TYPE_NUM,
    };
    using InitCallBack = std::function<void(DeviceTask *)>;
    using EnqueueOpCallBack = std::function<void(uint64_t, uint64_t *, uint32_t)>;
    using PollCompletedCallBack = std::function<void(std::vector<uint64_t> &)>;

    inline void TaskCallBackRegister() {}

    AicpuTaskManager() {
        TaskCallBackResigter<Distributed::CommWaitFlag>(TaskType::COMM_WAIT_FLAG, commWaitFlag_);
        TaskCallBackResigter<Distributed::DependOn>(TaskType::DEPEND_ON, dependOn_);
    };
    ~AicpuTaskManager() {};

    template <typename T>
    inline void TaskCallBackResigter(TaskType taskType, T &obj) {
        auto index = static_cast<uint32_t>(taskType);
        initCallBack_[index] = std::bind(&T::Init, &obj, std::placeholders::_1);
        enqueueOpCallBack_[index] =
            std::bind(&T::EnqueueOp, &obj, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
        pollCompletedCallBack_[index] = std::bind(&T::PollCompleted, &obj, std::placeholders::_1);
    }

    inline TaskType GetTaskType(uint32_t extType) {
        auto taskType = TaskType::TASK_TYPE_NUM;
        switch (extType) {
            case static_cast<uint32_t>(Opcode::OP_COMM_WAIT_FLAG): taskType = TaskType::COMM_WAIT_FLAG; break;
            case static_cast<uint32_t>(Opcode::OP_DEPEND_ON): taskType = TaskType::DEPEND_ON; break;
            default: break;
        }
        return taskType;
    }

    // 每个AICPU都会调用
    inline void TaskEnqueue(uint64_t taskId) {
        ReadyQueueLock();
        readyQueue_->elem[readyQueue_->tail] = taskId;
        readyQueue_->tail += 1;
        ReadyQueueUnLock();
    }

    // 仅AICPU_0会调用
    void Init(DeviceTask *deviceTask) {
        funcInfo_ = reinterpret_cast<CoreFunctionWsAddr *>(deviceTask->coreFuncData.coreFunctionWsAddr);
        readyQueue_ = reinterpret_cast<StaticReadyCoreFunctionQueue *>(deviceTask->readyAicpuFunctionQue);
        for (auto &init : initCallBack_) {
            init(deviceTask);
        }
    }

    // 仅AICPU_0会调用
    inline uint64_t  TaskProcess() {
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

    inline void TaskDispatch(uint64_t elem) {
        auto topo = reinterpret_cast<CoreFunctionTopo *>(funcInfo_[elem].topoAddr);
        auto taskType = GetTaskType(topo->extType);
        if (taskType < TaskType::TASK_TYPE_NUM) {
            auto enqueueOp = enqueueOpCallBack_[static_cast<uint64_t>(taskType)];
            enqueueOp(elem, reinterpret_cast<uint64_t *>(&topo->depIds[topo->depNum]), topo->extParamNum);
        }
    }

    StaticReadyCoreFunctionQueue *readyQueue_{nullptr};

    Distributed::CommWaitFlag commWaitFlag_;
    Distributed::DependOn dependOn_;

    std::array<InitCallBack, TaskType::TASK_TYPE_NUM> initCallBack_;
    std::array<EnqueueOpCallBack, TaskType::TASK_TYPE_NUM> enqueueOpCallBack_;
    std::array<PollCompletedCallBack, TaskType::TASK_TYPE_NUM> pollCompletedCallBack_;

    CoreFunctionWsAddr *funcInfo_{nullptr};
};
} // namespace npu::tile_fwk
