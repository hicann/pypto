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
#include "interface/utils/distributed_error.h"

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
                AICPU_TASK_ARRAY_SIZE);
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
        while (current != nullptr) {
            if (current->taskId_ == taskId) {
                return current;
            }
            current = current->next;
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
                "ctrl.task.pre.task.enqueue#: SignalTileOp queue_ is full, front=%u, rear=%u", front_, rear_);
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

        int32_t tileCols = (paramInfo_.rawShapeCol + paramInfo_.tileShapeCol - 1) / paramInfo_.tileShapeCol;
        int32_t tileRows = (paramInfo_.rawShapeRow + paramInfo_.tileShapeRow - 1) / paramInfo_.tileShapeRow;
        int32_t tileRow = info.offset[SHMEM_DIM_ROW] / paramInfo_.tileShapeRow;
        int32_t tileCol = info.offset[SHMEM_DIM_COL] / paramInfo_.tileShapeCol;
        int32_t tileIndex = tileRow * tileCols + tileCol;
        int32_t totalTileNum = tileRows * tileCols;

        DEV_DEBUG(
            "ShmemWaitUntilImpl::EnqueueOp logical rawShape=[%u, %u],"
            "logical tile=[%u, %u], logical offset=[%u, %u], ownerRank=%u,"
            "actual rawShape=[%lu, %d], actual offset=[%u, %d], buffer maxTileNum=%lu, bufferStride=%u",
            paramInfo_.rawShapeRow, paramInfo_.rawShapeCol, paramInfo_.tileShapeRow, paramInfo_.tileShapeCol,
            info.offset[SHMEM_DIM_ROW], info.offset[SHMEM_DIM_COL], info.offset[OWNER_RANK_ID_INDEX],
            GetRankNum(hcclContextAddr_, info.vaddr), totalTileNum, info.offset[OWNER_RANK_ID_INDEX], tileIndex,
            TileOp::Distributed::DecodeShmemAddrMaxTileNum(info.vaddr), paramInfo_.bufferStride);

        int32_t* addr =
            reinterpret_cast<int32_t*>(info.rawAddr) +
            CalcLinearOffset(totalTileNum, info.offset[OWNER_RANK_ID_INDEX], tileIndex) * paramInfo_.bufferStride;
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
