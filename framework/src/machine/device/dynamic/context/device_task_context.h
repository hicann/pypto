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
 * \file device_task_context.h
 * \brief
 */

#pragma once

#include "machine/device/dynamic/context/device_stitch_context.h"
#include "machine/utils/dynamic/dev_workspace.h"

namespace npu::tile_fwk::dynamic {
struct DeviceTaskContext {
    void InitAllocator(DevAscendProgram *devProg, DeviceWorkspaceAllocator &workspace,
                       npu::tile_fwk::DevStartArgsBase *startArgs);

    DynDeviceTask *BuildDeviceTaskData(DeviceStitchContext &stitchContext, uint32_t taskId, DevAscendProgram *devProg,
                                       bool withoutTail);

    void ReleaseFinishedTasks(int perfEvtReleaseFinishTask, int perfEvtDeallocateTask);

    void AppendFinishTask(DynDeviceTask *dynTask);

    void ShowStats();

    void UpdateReadyTaskNum(uint64_t cnt) { readyTaskNum += cnt; }
private:
    uint64_t stitchedFuncNum{0};
    uint64_t rootFuncNum{0};
    uint64_t leafFuncNum{0};
    uint64_t readyTaskNum {0};
    uint64_t dynFuncDataSize {0};
    uint64_t leafFuncDataSize {0};
private:
    DevAscendProgram *devProg_{nullptr};
    DeviceWorkspaceAllocator *workspace_{nullptr};
    npu::tile_fwk::DevStartArgsBase *startArgs_{nullptr};
private:
    int BuildReadyQueue(DynDeviceTask *dyntask, DevAscendProgram *devProg);
    void ProcessAivBatchTasks(ReadyCoreFunctionQueue *aivQueue, size_t totalZeroPredAIVBatchEnd,
        const predcount_t *dupPredCountList, size_t funcIndex);
    int InitReadyQueues(DynDeviceTask *dyntask, DevAscendProgram *devProg,
        ReadyCoreFunctionQueue* queue[READY_QUEUE_SIZE]);
    void UpdateDeviceTaskQueueInfo(DynDeviceTask *dyntask, ReadyCoreFunctionQueue *aicpuQueue, ReadyCoreFunctionQueue *aivQueue,
        ReadyCoreFunctionQueue *aicQueue, WrapInfoQueue *wrapQueue, uint32_t *wrapTasklistAddr);
    int BuildDynFuncData(DynDeviceTask *dyntask, uint32_t taskId,
        DevAscendFunctionDupped *stitchedList, uint64_t stitchedSize);

    // mix subgraph schedule
    uint32_t* AllocWrapTasklist(DynDeviceTask *dyntask);
    WrapInfoQueue* AllocWrapQueue(DynDeviceTask *dyntask);
    void ProcessWrapQueue(DynDeviceTask *dyntask, uint32_t wrapId, int funcIndex, size_t opIndex,
        WrapInfoQueue *wrapQueue, uint32_t *wrapTasklistAddr);
    bool IsNeedWrapProcess(DynDeviceTask *dyntask, DevAscendProgram *devProg);

    inline void doResolve(DynDeviceTask *dyntask, int coreType, size_t funcIdx, size_t succIdx, predcount_t *predList) {
        predList[succIdx] -= 1;
        if (predList[succIdx] != 0)
            return;

        if (coreType == static_cast<int>(CoreType::HUB)) {
            ResolveEarlyDepends(dyntask, funcIdx, succIdx);
        } else {
            int32_t* opWrapList = reinterpret_cast<int32_t *>(dyntask->devTask.mixTaskData.opWrapList[funcIdx]);
            if (dyntask->devTask.mixTaskData.wrapIdNum > 0 && opWrapList[succIdx] != -1) {
                ProcessWrapQueue(dyntask, MakeMixWrapID(funcIdx, static_cast<uint32_t>(opWrapList[succIdx])), funcIdx, succIdx,
                    reinterpret_cast<WrapInfoQueue *>(dyntask->devTask.mixTaskData.readyWrapCoreFunctionQue),
                    reinterpret_cast<uint32_t *>(dyntask->devTask.mixTaskData.wrapTasklist));
            } else {
                auto q = dyntask->readyQueue[dyntask->GetReadyQueueIndexByCoreType(static_cast<CoreType>(coreType))];
                q->elem[q->tail++] = MakeTaskID(funcIdx, succIdx);
            }
            readyTaskNum++;
        }
    }

    void ResolveEarlyDepends(DynDeviceTask *dyntask, size_t funcIdx, size_t opIdx);

    void ResolveEarlyDepends(DynDeviceTask *dyntask);

public:
    static void DumpReadyQueue(DynDeviceTask *dynTask, const char *prefix);

    static void DumpDepend(DynDeviceTask *dyntask, DevAscendProgram *devProg, DevStartArgs *startArgs, const char *prefix);

    int BuildDeviceTaskDataAndReadyQueue(DynDeviceTask *dyntask, uint32_t taskId, DevAscendProgram *devProg);
};
}
