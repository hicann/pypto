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
 * \file device_mix_task_context.cpp
 * \brief
 */

 #include "machine/device/dynamic/context/device_task_context.h"

namespace npu::tile_fwk::dynamic {
void DeviceTaskContext::ProcessWrapQueue(DynDeviceTask *dyntask, uint32_t wrapId, int funcIndex, size_t opIndex, WrapInfoQueue *wrapQueue, uint32_t *wrapTasklistAddr) {
    DEV_VERBOSE_DEBUG("add task to wrap queue, wrapId = %u, funcIndex = %d, opIndex = %lu", wrapId, funcIndex, opIndex);
    if (wrapQueue == nullptr || wrapTasklistAddr == nullptr) {
        DEV_VERBOSE_DEBUG("wrapQueue or wrapTasklistAddr = nullptr");
        return;
    }

    for (uint32_t idx = wrapQueue->head; idx < wrapQueue->tail; idx++) {
        if (wrapQueue->elem[idx].wrapId == wrapId) {
            ReadyCoreFunctionQueue* tasklist = &wrapQueue->elem[idx].tasklist;
            tasklist->elem[tasklist->tail++] = MakeTaskID(funcIndex, opIndex);
            return;
        }
    }

    auto opWrapTaskNumList = reinterpret_cast<int32_t*>(dyntask->devTask.mixTaskData.opWrapTaskNumList[funcIndex]);
    auto cceBinary = dyntask->cceBinary;
    auto callList = dyntask->dynFuncDataCacheList[funcIndex].calleeList;

    // add new wrap id to wrapQueue
    WrapInfo *info = &wrapQueue->elem[wrapQueue->tail];
    info->wrapId = wrapId;
    info->aicCoreIdx = 0;
    info->aivCoreIdxZero = 0;
    info->aivCoreIdxOne = 0;
    info->taskCnt = opWrapTaskNumList[opIndex];
    info->mixResourceType = cceBinary[callList[opIndex]].mixResourceType;
    info->tasklist.head = 0;
    info->tasklist.tail = 0;
    info->tasklist.capacity = opWrapTaskNumList[opIndex];
    if (wrapQueue->tail == 0) {
        info->tasklist.elem = wrapTasklistAddr;
    } else {
        WrapInfo *preQueueInfo = &wrapQueue->elem[wrapQueue->tail - 1];
        info->tasklist.elem = preQueueInfo->tasklist.elem + preQueueInfo->tasklist.capacity;
    }
    info->tasklist.elem[info->tasklist.tail++] = MakeTaskID(funcIndex, opIndex);
    wrapQueue->tail++;
}

uint32_t* DeviceTaskContext::AllocWrapTasklist(DynDeviceTask *dyntask) {
    uint32_t size = dyntask->devTask.coreFunctionCnt; // can be optimized by wrapTaskNum
    WsAllocation qalloc = ControlFlowAllocateSlab(devProg_, size, workspace_->SlabAlloc(size, WsAicpuSlabMemType::WRAP_TASKLIST));
    uint32_t *wrapTasklistAddr = qalloc.As<uint32_t>();
    return wrapTasklistAddr;
}

WrapInfoQueue* DeviceTaskContext::AllocWrapQueue(DynDeviceTask *dyntask) {
    uint32_t size = sizeof(WrapInfoQueue) + dyntask->devTask.mixTaskData.wrapIdNum * sizeof(WrapInfo);
    WsAllocation qalloc = ControlFlowAllocateSlab(devProg_, size, workspace_->SlabAlloc(size, WsAicpuSlabMemType::WRAP_QUEUE));
    WrapInfoQueue *q = qalloc.As<WrapInfoQueue>();
    q->head = 0;
    q->tail = 0;
    q->lock = 0;
    q->capacity = dyntask->devTask.mixTaskData.wrapIdNum;
    q->elem = reinterpret_cast<WrapInfo *>(q + 1);
    return q;
}

bool DeviceTaskContext::IsNeedWrapProcess(DynDeviceTask *dyntask, DevAscendProgram *devProg) {
    dyntask->devTask.mixTaskData.wrapIdNum = 0;
    if (devProg->devArgs.archInfo == ArchInfo::DAV_3510) {return false;}
    for (size_t funcIndex = 0; funcIndex < dyntask->dynFuncDataCacheListSize; ++funcIndex) {
        dyntask->devTask.mixTaskData.wrapIdNum += dyntask->dynFuncDataCacheList[funcIndex].devFunc->wrapIdNum_;
    }
    return dyntask->devTask.mixTaskData.wrapIdNum > 0;
}

}