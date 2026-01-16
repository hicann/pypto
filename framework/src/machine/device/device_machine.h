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

#include <cstdint>

#include "aicore_manager.h"
#include "machine/device/dynamic/device_utils.h"
#include "machine/utils/device_log.h"
#include "machine/utils/machine_ws_intf.h"

#ifndef MAX_DEVICE_TASK_NUM
#define MAX_DEVICE_TASK_NUM 64
#endif

namespace npu::tile_fwk {

class DeviceMachine {
public:
    DeviceMachine() {
        for (uint32_t i = 0; i < MAX_STATIC_SCHEDULE_AICPU_NUM; ++i) {
            aicoreManager_.push_back(std::make_unique<AiCoreManager>(aicpuTaskManager_));
        }
    }

    void init(DeviceArgs *args) {
        DEV_INFO("device machine init .\n");
        if (args->devQueueAddr != 0) {
            serverMode_ = true;
        }

        coreNum_ = args->nrAic + args->nrAiv;
        sharedBuffer_ = args->sharedBuffer;

        if (args->taskType == DEVICE_TASK_TYPE_STATIC) {
            auto devTask = reinterpret_cast<DeviceTask *>(args->taskData);
            DEV_IF_DEBUG {
                dumpTask(args->taskId, devTask);
            }
            auto idx = allocNewTaskCtrl();
            InitTaskCtrl(idx, DEVICE_TASK_TYPE_STATIC, args->taskId, devTask);
            initTaskCtrl = &taskctrl_[idx];
        }
    }

    int allocNewTaskCtrl() {
        while (true) {
            for (int idx = 0; idx < MAX_DEVICE_TASK_NUM; idx++) {
                if (taskctrl_[idx].IsFree()) {
                    return idx;
                }
            }
        }
    }

    void InitTaskCtrl(int idx, int type, uint64_t taskId, void *devTask, void (*finish)(void *) = nullptr) {
        auto taskCtrl = &taskctrl_[idx];
        taskCtrl->taskType = type;
        taskCtrl->devTask = devTask;
        taskCtrl->taskId = taskId;
        taskCtrl->finishedAicFunctionCnt = 0;
        taskCtrl->finishedAivFunctionCnt = 0;
        taskCtrl->finishedFunctionCnt = 0;
        taskCtrl->refcnt = aicoreManager_.size();
        taskCtrl->finishFunc = finish;
       for (auto& eType : taskCtrl->isAicpuIdle) {
            for (auto& e : eType) {
                e.store(true);
            }
        }
    }

    int PushTask(int type, uint64_t taskId, void *devTask, void (*finish)(void *) = nullptr) {
        auto idx = allocNewTaskCtrl();
        InitTaskCtrl(idx, type, taskId, devTask, finish);
        for (auto &m : aicoreManager_) {
            m->PushTask(&taskctrl_[idx]);
        }
        return idx;
    }

    void StopAicoreManager() {
        for (auto &m : aicoreManager_) {
            m->PushTask(nullptr);
        }
    }

    int SyncTask(int idx) {
        while (!taskctrl_[idx].IsFree())
            ;
        return taskctrl_[idx].retCode;
    }

    int SyncTask() {
        int ret = 0;
        for (int idx = 0; idx < MAX_DEVICE_TASK_NUM; idx++) {
            auto rc = SyncTask(idx);
            if (rc != 0)
                ret = rc;
        }
        return ret;
    }

    int Run(int threadIdx, DeviceArgs *args) {
        int ret = 0;
        if (args->nrAic == 0) {
            return npu::tile_fwk::dynamic::DEVICE_MACHINE_ERROR;
        }
        DEV_INFO("thread  %d start .\n", threadIdx);
        if (threadIdx >= npu::tile_fwk::dynamic::START_AICPU_NUM) {
            DEV_INFO("thread start ignore \n");
            return npu::tile_fwk::dynamic::DEVICE_MACHINE_OK;
        }

        ret = aicoreManager_[threadIdx]->Run(threadIdx, args, initTaskCtrl);
        DEV_INFO("thread  %d end , ret = %d \n", threadIdx, ret);
        return ret;
    }

    uint64_t GetMinTaskTime(int threadIdx) {
        return aicoreManager_[threadIdx]->GetTaskStartTime();
    }

    uint64_t GetMaxTaskTime(int threadIdx) {
        return aicoreManager_[threadIdx]->GetTaskEndTime();
    }

    int ExecDyn([[maybe_unused]]uint64_t taskId, [[maybe_unused]]int64_t taskData) {
        return 0;
    }

private:
    void dumpTask(int64_t taskId, DeviceTask *devTask) const {
        DEV_DEBUG("devTask %ld %p\n", taskId, devTask);
        if (devTask == nullptr) {
            return;
        }
        (void)(taskId);
        DEV_DEBUG("devtask { %lu, %lu, %lu, %lu, %lu, %lu, %lu}\n", devTask->coreFunctionCnt,
            devTask->coreFunctionReadyStateAddr, devTask->readyAicCoreFunctionQue, devTask->readyAivCoreFunctionQue,
            devTask->coreFuncData.coreFunctionWsAddr, devTask->coreFuncData.stackWorkSpaceAddr,
            devTask->coreFuncData.stackWorkSpaceSize);

        DEV_DEBUG("===== ready state =====\n");
        auto readyState = reinterpret_cast<CoreFunctionReadyState *>(devTask->coreFunctionReadyStateAddr);
        for (uint64_t i = 0; i < devTask->coreFunctionCnt; i++) {
            DEV_DEBUG("taskId %lu readyCount %ld coreType %lu\n", i, readyState[i].readyCount, readyState[i].coreType);
        }
        (void)(readyState);
        DEV_DEBUG("===== ready aic func =====\n");
        ReadyCoreFunctionQueue *readyFunc =
            reinterpret_cast<ReadyCoreFunctionQueue *>(devTask->readyAicCoreFunctionQue);
        for (uint64_t i = readyFunc->head; i < readyFunc->tail; i++) {
            DEV_DEBUG("taskId %u \n", readyFunc->elem[i]);
        }

        DEV_DEBUG("===== ready aiv func =====\n");
        readyFunc = reinterpret_cast<ReadyCoreFunctionQueue *>(devTask->readyAivCoreFunctionQue);
        for (uint64_t i = readyFunc->head; i < readyFunc->tail; i++) {
            DEV_DEBUG("taskId %u \n", readyFunc->elem[i]);
        }

        auto coreFunc = reinterpret_cast<CoreFunctionWsAddr *>(devTask->coreFuncData.coreFunctionWsAddr);
        DEV_DEBUG("===== core func =====\n");
        for (uint64_t i = 0; i < devTask->coreFunctionCnt; i++) {
            DEV_DEBUG("taskId %lx binAddr %lx invokeEntry %lx topo %lx\n", i, coreFunc[i].functionBinAddr,
                coreFunc[i].invokeEntryAddr, coreFunc[i].topoAddr);
            auto topo = reinterpret_cast<CoreFunctionTopo *>(coreFunc[i].topoAddr);
            DEV_DEBUG("coreType %lu pstId %lu readyCount %ld depNum %lu \n", topo->coreType, topo->psgId,
                topo->readyCount, topo->depNum);
            (void)(topo);
        }
        DEV_DEBUG("===== dev task end =====\n");
    }

private:
    uint64_t sharedBuffer_{0};
    uint64_t coreNum_{0};
    bool serverMode_{false};
    DeviceTaskCtrl taskctrl_[MAX_DEVICE_TASK_NUM];
    DeviceTaskCtrl *initTaskCtrl{nullptr};
    AicpuTaskManager aicpuTaskManager_;
    std::vector<std::unique_ptr<AiCoreManager>> aicoreManager_;
};
} // namespace npu::tile_fwk
