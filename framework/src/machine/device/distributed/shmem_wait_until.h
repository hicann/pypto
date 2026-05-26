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
 * \file shmem_wait_until.h
 * \brief
 */

#ifndef SHMEM_WAIT_UNTIL_H
#define SHMEM_WAIT_UNTIL_H

#include <vector>

#include "common.h"
#include "machine/utils/dynamic/dev_workspace.h"
#include "machine/utils/dynamic/device_task.h"
#include "machine/device/dynamic/device_utils.h"
#include "tilefwk/error_code.h"

namespace npu::tile_fwk::Distributed {
struct SignalTileOp {
    void Init(uint64_t taskId, int32_t* addr, int32_t expectedSum, bool resetSignal)
    {
        taskId_ = taskId;
        addr_ = addr;
        expectedSum_ = expectedSum;
        resetSignal_ = resetSignal;
    }
    bool PollCompleted() const;

    SignalTileOp* next{nullptr};
    uint64_t taskId_{0};
    int32_t* addr_{nullptr};
    int32_t expectedSum_{0};
    bool resetSignal_{false};
    TaskStat* profData_{nullptr};
};

class HashMap {
public:
    void Init()
    {
        (void)memset_s(&taskArray, sizeof(taskArray), 0, sizeof(taskArray));
        (void)memset_s(&hashTable, sizeof(hashTable), 0, sizeof(hashTable));
        taskCount = 0;
    }

    uint32_t Hash(uint32_t taskId) { return taskId & AICPU_TASK_ARRAY_SIZE_MOD; }

    SignalTileOp* CreateTaskData(uint32_t taskId, int32_t* addr, int32_t expectSum, bool resetSignal)
    {
        if (taskCount >= AICPU_TASK_ARRAY_SIZE) {
            DEV_ERROR(
                DistributedErrorCode::AICPU_TASK_NUM_EXCEED_LIMIT,
                "ctrl.task.pre.task.create#: taskCount=%u >= AICPU_TASK_ARRAY_SIZE=%lu", taskCount,
                AICPU_TASK_ARRAY_SIZE - 1);
            return nullptr;
        }
        SignalTileOp* newTask = &taskArray[taskCount];
        newTask->Init(taskId, addr, expectSum, resetSignal);
        taskCount++;
        return newTask;
    }

    int32_t InsertTask(uint32_t taskId, int32_t* addr, int32_t expectSum, bool resetSignal)
    {
        SignalTileOp* newTask = CreateTaskData(taskId, addr, expectSum, resetSignal);
        if (newTask == nullptr) {
            return dynamic::DEVICE_MACHINE_ERROR;
        }
        uint32_t index = Hash(taskId);
        SignalTileOp* current = hashTable[index];
        hashTable[index] = newTask;
        newTask->next = current;
        return dynamic::DEVICE_MACHINE_OK;
    }

    SignalTileOp* FindTask(uint32_t taskId)
    {
        uint32_t index = Hash(taskId);
        SignalTileOp* current = hashTable[index];
        uint32_t loopCount = 0;
        while (current != nullptr && loopCount < AICPU_TASK_ARRAY_SIZE) {
            if (current->taskId_ == taskId) {
                return current;
            }
            current = current->next;
            loopCount++;
        }
        return nullptr;
    }

private:
    SignalTileOp taskArray[AICPU_TASK_ARRAY_SIZE];
    uint32_t taskCount{0};
    SignalTileOp* hashTable[AICPU_TASK_ARRAY_SIZE];
};

class CircularQueue {
public:
    CircularQueue() = default;

    inline int32_t Enqueue(SignalTileOp* task)
    {
        queue_[rear_] = task;
        rear_ = (rear_ + 1) & AICPU_TASK_ARRAY_SIZE_MOD;
        if (rear_ == front_) {
            DEV_ERROR(
                DistributedErrorCode::AICPU_TASK_NUM_EXCEED_LIMIT,
                "ctrl.task.pre.task.enqueue#: SignalTileOp queue is full, capacity = %lu", AICPU_TASK_ARRAY_SIZE - 1);
            return dynamic::DEVICE_MACHINE_ERROR;
        }
        return dynamic::DEVICE_MACHINE_OK;
    }

    inline bool IsEmpty() const { return front_ == rear_; }

    inline int32_t Dequeue()
    {
        if (IsEmpty()) {
            DEV_ERROR(DistributedErrorCode::AICPU_TASK_QUEUE_EMPTY, "sche.task.end.task.dequeue#: Queue is empty.");
            return dynamic::DEVICE_MACHINE_ERROR;
        }
        front_ = (front_ + 1) & AICPU_TASK_ARRAY_SIZE_MOD;
        return dynamic::DEVICE_MACHINE_OK;
    }

    inline const SignalTileOp* operator[](uint16_t index) const { return queue_[index]; }

    inline int32_t Remove(uint16_t index)
    {
        queue_[index] = queue_[front_];
        return Dequeue();
    }

    int32_t PollCompleted(std::function<int32_t(SignalTileOp*)> processor)
    {
        uint16_t current = front_;
        uint16_t end = rear_;
        if (current > end) {
            end += AICPU_TASK_ARRAY_SIZE;
        }
        for (uint16_t i = current; i < end; ++i) {
            uint16_t actualIndex = i & AICPU_TASK_ARRAY_SIZE_MOD;
            SignalTileOp* task = queue_[actualIndex];
            if (task->PollCompleted()) {
                if (task->profData_ != nullptr) {
                    task->profData_->execEnd = dynamic::GetCycles();
                }
                int32_t ret = processor(task);
                if (ret != dynamic::DEVICE_MACHINE_OK) {
                    return ret;
                }
                ret = Remove(actualIndex);
                if (ret != dynamic::DEVICE_MACHINE_OK) {
                    return ret;
                }
            }
        }
        return dynamic::DEVICE_MACHINE_OK;
    }

private:
    SignalTileOp* queue_[AICPU_TASK_ARRAY_SIZE];
    uint16_t front_{0};
    uint16_t rear_{0};
};

class ShmemWaitUntilImpl {
public:
    inline void Init(npu::tile_fwk::dynamic::DynDeviceTask* dynDeviceTask)
    {
        dynDeviceTask_ = dynDeviceTask;
        funcDataList_ = reinterpret_cast<DynFuncData*>(&dynDeviceTask->GetDynFuncDataList()->At(0));
        hcclContextAddr_ = funcDataList_->startArgs->commContexts;
        commGroupNum_ = funcDataList_->startArgs->commGroupNum;
        hashMap_.Init();
    }

    inline int32_t EnqueueOp(uint64_t taskId, TaskStat* taskStat)
    {
        SignalTileOp* task = hashMap_.FindTask(taskId);
        if (task == nullptr) {
            DEV_ERROR(
                DistributedErrorCode::AICPU_TASKID_NOT_IN_MAP, "ctrl.task.pre.task.enqueue#: taskId=%lu not found",
                taskId);
            return dynamic::DEVICE_MACHINE_ERROR;
        }
        if (taskStat != nullptr) {
            task->profData_ = taskStat;
            task->profData_->taskId = static_cast<int32_t>(taskId);
            task->profData_->execStart = dynamic::GetCycles();
        }
        return runingTaskQueue_.Enqueue(task);
    }

    inline int32_t PrepareTask(uint64_t taskId, const npu::tile_fwk::dynamic::DevRelocVector<int32_t>& aicpuCode)
    {
        paramInfo_ = DecodeAicpuCode(aicpuCode);
        TensorInfo info = ShmemWaitUntilImpl::GetTensorInfo(taskId, aicpuCode);
        const int32_t expectedSum = info.expectedSum;
        const bool resetSignal = info.resetSignal;

        uint32_t tileIndex = 0;
        uint32_t viewIndexAccum = 0;

        uint32_t dataDim = paramInfo_.dim;
        uint32_t tileShapeDim = paramInfo_.tileShape.size();
        uint32_t startDim = dataDim - tileShapeDim;

        for (uint32_t dimIdx = 0; dimIdx < tileShapeDim; ++dimIdx) {
            uint32_t curDim = startDim + dimIdx;
            uint32_t viewShape = paramInfo_.viewshapes[dimIdx];
            uint32_t offset = info.offset[curDim];
            uint32_t tileShapeVal = paramInfo_.tileShape[dimIdx];

            uint32_t viewIdx = offset / viewShape;
            uint32_t viewOffset = offset % viewShape;

            uint32_t viewTileIdx = viewOffset / tileShapeVal;

            tileIndex += viewTileIdx * paramInfo_.viewTileStrides[dimIdx];
            viewIndexAccum += viewIdx * paramInfo_.viewIndexStrides[dimIdx];
        }
        tileIndex += viewIndexAccum * paramInfo_.viewTileNum;

        int32_t* addr = reinterpret_cast<int32_t*>(info.rawAddr) +
                        CalcLinearOffset(paramInfo_.totalTileNum, info.offset[OWNER_RANK_ID_INDEX], tileIndex) *
                            paramInfo_.bufferStride;

        DEV_DEBUG(
            "PrepareTask taskId=%lu, baseAddr=0x%lx, actualAddr=0x%lx, ownerRank=%u, actual rawShape=[%lu, %u],"
            "actual offset=[%u, %u], buffer maxTileNum=%lu, bufferStride=%u",
            taskId, info.rawAddr, reinterpret_cast<uint64_t>(addr), info.offset[OWNER_RANK_ID_INDEX],
            GetRankNum(hcclContextAddr_, info.vaddr), paramInfo_.totalTileNum, info.offset[OWNER_RANK_ID_INDEX], tileIndex,
            TileOp::Distributed::DecodeShmemAddrMaxTileNum(info.vaddr), paramInfo_.bufferStride);

        return hashMap_.InsertTask(taskId, addr, expectedSum, resetSignal);
    }

    int32_t PollCompleted(npu::tile_fwk::dynamic::AiCoreManager* aiCoreManager);

    CircularQueue runingTaskQueue_;

private:
    HashMap hashMap_;
    uint32_t signalTileOpCount_{0};

    npu::tile_fwk::dynamic::DynDeviceTask* dynDeviceTask_;
    npu::tile_fwk::DynFuncData* funcDataList_;
    int64_t* hcclContextAddr_;
    uint64_t commGroupNum_{0};
    AicpuParamInfo paramInfo_;

    TensorInfo GetTensorInfo(uint64_t taskId, const npu::tile_fwk::dynamic::DevRelocVector<int32_t>& aicpuCode);
};

} // namespace npu::tile_fwk::Distributed
#endif // SHMEM_WAIT_UNTIL_H
