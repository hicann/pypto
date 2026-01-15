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
 * \file device_task_context.cpp
 * \brief
 */

#include "machine/device/dynamic/context/device_task_context.h"

namespace npu::tile_fwk::dynamic {
namespace {
    const uint32_t OP_ATTRS_PRE_NUM = 8;
    const uint32_t OP_ATTRS_OFFSET_PRE_NUM = 4;
    const uint32_t EXPR_TABLE_PRE_NUM = 8;
    const uint32_t RAW_TENSOR_ADDR_MASK = 8;
    const uint32_t CCE_BINARY_MOD = 8;
    const size_t DUP_PRED_COUNT_LOOP_MAX = 8;
    const size_t DUP_PRED_COUNT_PRE_LOOP_CNT = 4;
}

void DeviceTaskContext::InitAllocator(DevAscendProgram *devProg, DeviceWorkspaceAllocator &workspace, npu::tile_fwk::DevStartArgsBase *startArgs) {
    devProg_ = devProg;
    workspace_ = &workspace;
    startArgs_ = startArgs;
}

DynDeviceTask *DeviceTaskContext::BuildDeviceTaskData(DeviceStitchContext &stitchContext, uint32_t taskId, DevAscendProgram *devProg, bool withoutTail) {
    int ret = DEVICE_MACHINE_OK;
    PerfBegin(PERF_EVT_ALLOCATE_TASK);
    DynDeviceTask *dynTask = workspace_->MakeDynDeviceTask();
    ret = stitchContext.MoveTo(dynTask);
    if (unlikely(ret != DEVICE_MACHINE_OK)) {
        return nullptr;
    }
    PerfEnd(PERF_EVT_ALLOCATE_TASK);

    PerfBegin(PERF_EVT_BUILD_TASK_DATA);
    ret = BuildDeviceTaskDataAndReadyQueue(dynTask, taskId, devProg);
    if (unlikely(ret != DEVICE_MACHINE_OK)) {
        return nullptr;
    }
    PerfEnd(PERF_EVT_BUILD_TASK_DATA);

    PerfBegin(PERF_EVT_SLAB_MEM_SUBMIT);
    // cache allocated memory , when task finish will recycle
    dynTask->taskStageAllocMem = workspace_->SlabGetStageAllocMem(withoutTail, WsAicpuSlabMemType::DUPPED_FUNC_DATA);
    workspace_->SlabStageAllocMemSubmmit(&dynTask->taskStageAllocMem);
    PerfEnd(PERF_EVT_SLAB_MEM_SUBMIT);
    return dynTask;
}

void DeviceTaskContext::ReleaseFinishedTasks(int perfEvtReleaseFinishTask, int perfEvtDeallocateTask) {
    (void)perfEvtReleaseFinishTask;
    (void)perfEvtDeallocateTask;
}

void DeviceTaskContext::AppendFinishTask(DynDeviceTask *dynTask) {
    (void)dynTask;
}

void DeviceTaskContext::ShowStats() {
    DEV_ERROR("   Stitched function count: %10lu.", stitchedFuncNum);
    DEV_ERROR("       Root function count: %10lu.", rootFuncNum);
    DEV_ERROR("       Leaf function count: %10lu.", leafFuncNum);
    DEV_ERROR("   Inital ready task count: %10lu.", readyTaskNum);
    DEV_ERROR(" Static function data size: %10lu bytes.", dynFuncDataSize);
    DEV_ERROR("   Leaf function data size: %10lu bytes.", leafFuncDataSize);
}

int DeviceTaskContext::InitReadyQueues(DynDeviceTask *dyntask, DevAscendProgram *devProg,
    ReadyCoreFunctionQueue* queue[READY_QUEUE_SIZE]) {
    uint32_t size = sizeof(ReadyCoreFunctionQueue) + dyntask->devTask.coreFunctionCnt * sizeof(taskid_t);
    if (dyntask->devTask.coreFunctionCnt > devProg->stitchFunctionsize) {
        DEV_ERROR("coreFunctionCnt (%lu) exceeds stitchFunctionsize (%u), cannot build ready queue.",
        dyntask->devTask.coreFunctionCnt, devProg->stitchFunctionsize);
        return DEVICE_MACHINE_ERROR;
    }
    DEV_ASSERT(dyntask->devTask.coreFunctionCnt <= devProg->stitchFunctionsize);

    for (size_t i = 0; i < READY_QUEUE_SIZE; ++i) {
        WsAllocation qalloc = ControlFlowAllocateSlab(devProg_, size, workspace_->SlabAlloc(size, WsAicpuSlabMemType::READY_QUE));
        ReadyCoreFunctionQueue *q = qalloc.As<ReadyCoreFunctionQueue>();
        q->lock = 0;
        q->head = 0;
        q->tail = 0;
        q->capacity = dyntask->devTask.coreFunctionCnt;
        q->elem = reinterpret_cast<taskid_t *>(q + 1);
        queue[i] = q;
        dyntask->readyQueue[i] = q;
    }
    return DEVICE_MACHINE_OK;
}

void DeviceTaskContext::ProcessAivBatchTasks(ReadyCoreFunctionQueue *aivQueue, size_t totalZeroPredAIVBatchEnd,
    const predcount_t *dupPredCountList, size_t funcIndex) {
    uint32v8 one = {1, 1, 1, 1, 1, 1, 1, 1};
    uint32v8 base = {0, 1, 2, 3, 4, 5, 6, 7};
    taskid_t *aivQueueElemList = reinterpret_cast<taskid_t *>(aivQueue->elem);

    for (size_t opIndex = 0; opIndex < totalZeroPredAIVBatchEnd; opIndex += DUP_PRED_COUNT_LOOP_MAX) {
        if (likely((*reinterpret_cast<const uint64_t *>(&dupPredCountList[opIndex]) |
            *reinterpret_cast<const uint64_t *>(&dupPredCountList[opIndex + DUP_PRED_COUNT_PRE_LOOP_CNT])) == 0)) {
            uint32v8 taskidv8 = (one * MakeTaskID(funcIndex, 0)) | (base + static_cast<uint32_t>(opIndex));
#ifdef __x86_64__
            memcpy_s(&aivQueueElemList[aivQueue->tail], sizeof(taskidv8), &taskidv8, sizeof(taskidv8));
#else
            *reinterpret_cast<uint32v8 *>(&aivQueueElemList[aivQueue->tail]) = taskidv8;
#endif
            aivQueue->tail += DUP_PRED_COUNT_LOOP_MAX;
        } else {
            for (size_t idx = 0; idx < DUP_PRED_COUNT_LOOP_MAX; ++idx) {
                if (likely(dupPredCountList[opIndex + idx] == 0)) {
                    aivQueueElemList[aivQueue->tail++] = MakeTaskID(funcIndex, opIndex + idx);
                }
            }
        }
    }
}

void DeviceTaskContext::UpdateDeviceTaskQueueInfo(DynDeviceTask *dyntask, ReadyCoreFunctionQueue *aicpuQueue,
    ReadyCoreFunctionQueue *aivQueue, ReadyCoreFunctionQueue *aicQueue, WrapInfoQueue *wrapQueue, uint32_t *wrapTasklistAddr) {
    dyntask->devTask.readyAivCoreFunctionQue = PtrToValue(aivQueue);
    dyntask->devTask.readyAicCoreFunctionQue = PtrToValue(aicQueue);
    dyntask->devTask.readyAicpuFunctionQue = PtrToValue(aicpuQueue);
    dyntask->devTask.mixTaskData.readyWrapCoreFunctionQue = PtrToValue(wrapQueue);
    dyntask->devTask.mixTaskData.wrapTasklist = PtrToValue(wrapTasklistAddr);
}

int DeviceTaskContext::BuildReadyQueue(DynDeviceTask *dyntask, DevAscendProgram *devProg) {
    PerfBegin(PERF_EVT_READY_QUEUE_IN);

    ReadyCoreFunctionQueue *queue[READY_QUEUE_SIZE];
    if (InitReadyQueues(dyntask, devProg, queue) != DEVICE_MACHINE_OK) {return DEVICE_MACHINE_ERROR;}
    ReadyCoreFunctionQueue *aicpuQueue = queue[DynDeviceTask::GetReadyQueueIndexByCoreType(CoreType::AICPU)];
    ReadyCoreFunctionQueue *aivQueue = queue[DynDeviceTask::GetReadyQueueIndexByCoreType(CoreType::AIV)];
    ReadyCoreFunctionQueue *aicQueue = queue[DynDeviceTask::GetReadyQueueIndexByCoreType(CoreType::AIC)];

    bool isNeedWrap = IsNeedWrapProcess(dyntask, devProg);
    uint32_t *wrapTasklistAddr = isNeedWrap ? AllocWrapTasklist(dyntask) : nullptr;
    WrapInfoQueue *wrapQueue = isNeedWrap ? AllocWrapQueue(dyntask) : nullptr;

    int wrapTaskNum = 0;
    DynFuncDataCache *dynFuncDataCacheList = dyntask->GetDynFuncDataCacheList();
    size_t funcSize = dyntask->dynFuncDataCacheListSize;
    for (size_t funcIndex = 0; funcIndex < funcSize; ++funcIndex) {
        int32_t* opWrapList = reinterpret_cast<int32_t *>(dyntask->devTask.mixTaskData.opWrapList[funcIndex]);
        DevAscendFunctionDuppedData *duppedData = dynFuncDataCacheList->At(funcIndex).duppedData;
        predcount_t *dupPredCountList = &duppedData->GetOperationCurrPredCount(0);
        auto &predInfo = duppedData->GetSource()->GetPredInfo();
        size_t totalZeroPredAIVBatchEnd = isNeedWrap ? 0 : predInfo.totalZeroPredAIV & ~0x7; // wrap doesnt support batch process
        ProcessAivBatchTasks(aivQueue, totalZeroPredAIVBatchEnd, &duppedData->GetOperationCurrPredCount(0), funcIndex);

        for (size_t opIndex = totalZeroPredAIVBatchEnd; opIndex < predInfo.totalZeroPredAIV; ++opIndex) {
            if (likely(dupPredCountList[opIndex] == 0)) {
                if (isNeedWrap && opWrapList[opIndex] != -1) {
                    ProcessWrapQueue(dyntask, MakeMixWrapID(funcIndex, static_cast<uint32_t>(opWrapList[opIndex])),
                        funcIndex, opIndex, wrapQueue, wrapTasklistAddr);
                    wrapTaskNum++;
                } else {aivQueue->elem[aivQueue->tail++] = MakeTaskID(funcIndex, opIndex);}
            }
        }

        // process aic task
        auto aicEnd = predInfo.totalZeroPredAIV + predInfo.totalZeroPredAIC;
        for (size_t opIndex = predInfo.totalZeroPredAIV; opIndex < aicEnd; ++opIndex) {
            if (likely(dupPredCountList[opIndex] == 0)) {
                if (isNeedWrap && opWrapList[opIndex] != -1) {
                    ProcessWrapQueue(dyntask, MakeMixWrapID(funcIndex, static_cast<uint32_t>(opWrapList[opIndex])),
                        funcIndex, opIndex, wrapQueue, wrapTasklistAddr);
                    wrapTaskNum++;
                } else {aicQueue->elem[aicQueue->tail++] = MakeTaskID(funcIndex, opIndex);}
            }
        }

        // process aicpu task
        auto aicpuEnd = predInfo.totalZeroPredAIV + predInfo.totalZeroPredAIC + predInfo.totalZeroPredAicpu;
        for (size_t opIndex = aicEnd; opIndex < aicpuEnd; ++opIndex) {
            if (likely(dupPredCountList[opIndex] == 0)) {aicpuQueue->elem[aicpuQueue->tail++] = MakeTaskID(funcIndex, opIndex);}
        }
    }

    UpdateDeviceTaskQueueInfo(dyntask, aicpuQueue, aivQueue, aicQueue, wrapQueue, wrapTasklistAddr);
    readyTaskNum += static_cast<uint64_t>(aivQueue->tail + aicQueue->tail + aicpuQueue->tail + wrapTaskNum);
    PerfEnd(PERF_EVT_READY_QUEUE_IN);
    return DEVICE_MACHINE_OK;
}

int DeviceTaskContext::BuildDynFuncData(DynDeviceTask *dyntask, uint32_t taskId, DevAscendProgram *devProg,
    DevAscendFunctionDupped *stitchedList, uint64_t stitchedSize) {
    size_t headerSize = sizeof(DynFuncHeader) + stitchedSize * sizeof(DynFuncData);
    auto funcHeader = workspace_->AllocateDynFuncData(headerSize);
    dyntask->dynFuncDataList = funcHeader;
    auto dyndata = &funcHeader->At(0);

    stitchedFuncNum++;

    funcHeader->funcSize = headerSize;
    funcHeader->seqNo = taskId;
    funcHeader->funcNum = stitchedSize;
    funcHeader->cceBinary = reinterpret_cast<DynFuncBin *>(const_cast<DevCceBinary *>(dyntask->cceBinary));
    if (reinterpret_cast<uint64_t>(funcHeader->cceBinary) % CCE_BINARY_MOD != 0) {
        DEV_ERROR("cceBinary address  is not aligned.");
        return DEVICE_MACHINE_ERROR;
    }
    DEV_ASSERT(reinterpret_cast<uint64_t>(funcHeader->cceBinary) % CCE_BINARY_MOD == 0);

    rootFuncNum += stitchedSize;
    for (size_t funcIdx = 0; funcIdx < stitchedSize; ++funcIdx) {
        auto &dupFunc = stitchedList[funcIdx];
        dyndata->opAttrs = reinterpret_cast<uint64_t *>(const_cast<SymInt *>(dupFunc.GetSource()->GetSymoffset(0)));
        dyndata->opAtrrOffsets = dupFunc.GetSource()->GetOpAttrOffsetAddr();
        dyndata->exprNum = dupFunc.GetSource()->expressionList.size();
        dyndata->exprTbl = dupFunc.GetExpressionAddr();
        dyndata->rawTensorAddr = reinterpret_cast<uint64_t *>(&dupFunc.GetIncastAddress(0));
        dyndata->rawTensorDesc = dupFunc.GetSource()->GetRawTensorDesc(0);
        dyndata->startArgs = this->startArgs_;
        dyndata->workspaceAddr = dupFunc.RuntimeWorkspace();
        dyndata->stackWorkSpaceSize = workspace_->StandardStackWorkspacePerCore();
        dyndata->stackWorkSpaceAddr = workspace_->StackWorkspaceAddr();
        dyndata->opAttrSize = dupFunc.GetSource()->GetOpAttrSize();
        dyndata->rawTensorAddrSize = dupFunc.GetSource()->GetIncastSize();
        dyndata->rawTensorDescSize = dupFunc.GetSource()->GetRawTensorDescSize();
        dyndata->commGroupNum = devProg->commGroupNum;
        if (sizeof(dyndata->hcclContext) != sizeof(devProg->hcclContext)) {
            DEV_ERROR("hcclContext size mismatch, dyndata size: %zu, devProg size: %zu",
                      sizeof(dyndata->hcclContext), sizeof(devProg->hcclContext));
            return DEVICE_MACHINE_ERROR;
        }
        DEV_ASSERT(sizeof(dyndata->hcclContext) == sizeof(devProg->hcclContext));
        (void)memcpy_s(dyndata->hcclContext, sizeof(dyndata->hcclContext), devProg->hcclContext, sizeof(devProg->hcclContext));
        if (reinterpret_cast<uint64_t>(dyndata->opAttrs) % OP_ATTRS_PRE_NUM != 0) {
            DEV_ERROR("opAttrs address is not aligned.");
            return DEVICE_MACHINE_ERROR;
        }
        if (reinterpret_cast<uint64_t>(dyndata->opAtrrOffsets) % OP_ATTRS_OFFSET_PRE_NUM != 0) {
            DEV_ERROR("opAtrrOffsets address is not aligned.");
            return DEVICE_MACHINE_ERROR;
        }
        if (reinterpret_cast<uint64_t>(dyndata->exprTbl) % EXPR_TABLE_PRE_NUM != 0) {
            DEV_ERROR("exprTbl address is not aligned.");
            return DEVICE_MACHINE_ERROR;
        }
        if (reinterpret_cast<uint64_t>(dyndata->rawTensorAddr) % RAW_TENSOR_ADDR_MASK != 0) {
            DEV_ERROR("rawTensorAddr address is not aligned.");
            return DEVICE_MACHINE_ERROR;
        }
        DEV_ASSERT(reinterpret_cast<uint64_t>(dyndata->opAttrs) % OP_ATTRS_PRE_NUM == 0);
        DEV_ASSERT(reinterpret_cast<uint64_t>(dyndata->opAtrrOffsets) % OP_ATTRS_OFFSET_PRE_NUM == 0);
        DEV_ASSERT(reinterpret_cast<uint64_t>(dyndata->exprTbl) % EXPR_TABLE_PRE_NUM == 0);
        DEV_ASSERT(reinterpret_cast<uint64_t>(dyndata->rawTensorAddr) % RAW_TENSOR_ADDR_MASK == 0);

        leafFuncDataSize += dupFunc.GetSource()->GetOpAttrSize() * sizeof(SymInt); // opAttrs
        leafFuncDataSize += dupFunc.GetSource()->GetOperationSize() * sizeof(int32_t); // opAttrOffsts;
        leafFuncDataSize += dyndata->exprNum * sizeof(int64_t);
        leafFuncDataSize += dupFunc.GetSource()->GetRawTensorSize() * sizeof(DevRawTensorDesc);

        leafFuncNum += dupFunc.GetSource()->GetOperationSize();
        dupFunc.SetFuncData(dyndata);
        dyndata++;
    }
    dynFuncDataSize += headerSize * sizeof(int64_t);
    return DEVICE_MACHINE_OK;
}

void DeviceTaskContext::ResolveEarlyDepends(DynDeviceTask *dyntask, size_t funcIndex, size_t opIdx) {
    size_t succSize;

    auto cceBinary = dyntask->cceBinary;
    auto func = dyntask->dynFuncDataCacheList[funcIndex].devFunc;
    auto predList = dyntask->dynFuncDataCacheList[funcIndex].predCount;
    auto succList = func->GetOperationDepGraphSuccAddr(opIdx, succSize);
    auto callList = dyntask->dynFuncDataCacheList[funcIndex].calleeList;

    for (size_t index = 0; index < succSize; ++index) {
        auto succIdx = succList[index];
        doResolve(dyntask, cceBinary[callList[succIdx]].coreType, funcIndex, succIdx, predList);
    }

    auto &funcDup = dyntask->stitchedList[funcIndex];
    auto &stitchList = funcDup.GetOperationStitch(opIdx);
    for (auto *node = stitchList.Head(); node != nullptr; node = node->Next()) {
        uint32_t listSize = node->Size();
        for (uint32_t index = 0; index < listSize; ++index) {
            uint32_t id = node->At(index);
            auto succFuncIdx = FuncID(id);
            auto succIdx = TaskID(id);
            predList = dyntask->dynFuncDataCacheList[succFuncIdx].predCount;
            callList = dyntask->dynFuncDataCacheList[succFuncIdx].calleeList;
            doResolve(dyntask, cceBinary[callList[succIdx]].coreType, succFuncIdx, succIdx, predList);
        }
    }
}

void DeviceTaskContext::ResolveEarlyDepends(DynDeviceTask *dyntask) {
    size_t funcSize = dyntask->stitchedList.size();
    for (size_t funcIdx = 0; funcIdx < funcSize; ++funcIdx) {
        auto func = dyntask->dynFuncDataCacheList[funcIdx].devFunc;
        auto predList = dyntask->dynFuncDataCacheList[funcIdx].predCount;
        auto &predInfo = func->GetPredInfo();
        auto opIndex = predInfo.totalZeroPredAIC + predInfo.totalZeroPredAIV + predInfo.totalZeroPredAicpu;
        while (opIndex < predInfo.totalZeroPred) {
            if (predList[opIndex] == 0) {
                ResolveEarlyDepends(dyntask, funcIdx, opIndex);
            }
            opIndex++;
        }
    }
}

void DeviceTaskContext::DumpReadyQueue(DynDeviceTask *dynTask, const char *prefix) {
    DEV_ERROR("%s: coreFunctionCnt: %d", prefix, (int)dynTask->devTask.coreFunctionCnt);
    int aivIndex = DynDeviceTask::GetReadyQueueIndexByCoreType(CoreType::AIV);
    int aicIndex = DynDeviceTask::GetReadyQueueIndexByCoreType(CoreType::AIC);
    int aicpuIndex = DynDeviceTask::GetReadyQueueIndexByCoreType(CoreType::AICPU);

    DEV_ERROR("%s: ready queue aiv: %d-%d", prefix, (int)dynTask->readyQueue[aivIndex]->head, (int)dynTask->readyQueue[aivIndex]->tail);
    for (uint32_t i = dynTask->readyQueue[aivIndex]->head; i < dynTask->readyQueue[aivIndex]->tail; i++) {
        DEV_ERROR("%s: ready queue aiv[%d]: %x", prefix, (int)i, dynTask->readyQueue[aivIndex]->elem[i]);
    }

    DEV_ERROR("%s: ready queue aic: %d-%d", prefix, (int)dynTask->readyQueue[aicIndex]->head, (int)dynTask->readyQueue[aicIndex]->tail);
    for (uint32_t i = dynTask->readyQueue[aicIndex]->head; i < dynTask->readyQueue[aicIndex]->tail; i++) {
        DEV_ERROR("%s: ready queue aic[%d]: %x", prefix, (int)i, dynTask->readyQueue[aicIndex]->elem[i]);
    }

    DEV_ERROR("%s: ready queue aicpu: %d-%d", prefix, (int)dynTask->readyQueue[aicpuIndex]->head, (int)dynTask->readyQueue[aicpuIndex]->tail);
    for (uint32_t i = dynTask->readyQueue[aicpuIndex]->head; i < dynTask->readyQueue[aicpuIndex]->tail; i++) {
        DEV_ERROR("%s: ready queue aicpu[%d]: %x", prefix, (int)i, dynTask->readyQueue[aicpuIndex]->elem[i]);
    }
}
void DeviceTaskContext::DumpDepend(DynDeviceTask *dyntask, DevAscendProgram *devProg, DevStartArgs *startArgs, const char *prefix) {
    (void)devProg;
    (void)startArgs;
    int total = 0;
    for (size_t i = 0; i < READY_QUEUE_SIZE; i++) {
        ReadyCoreFunctionQueue *q = dyntask->readyQueue[i];
        total += q->tail - q->head;
    }
    DEV_ERROR("%s: ready total:%d", prefix, total);
    for (size_t i = 0; i < READY_QUEUE_SIZE; i++) {
        ReadyCoreFunctionQueue *q = dyntask->readyQueue[i];
        for (uint32_t k = q->head; k < q->tail; k++) {
            uint32_t taskId = q->elem[k];
            uint32_t dupIndex = FuncID(taskId);
            uint32_t opIndex = TaskID(taskId);
            DEV_ERROR("%s: ready %d-%d:L(%d,%d,%d)\n",
                prefix,
                (int)i, (int)k,
                (int)dyntask->GetDynFuncDataList()->seqNo, (int)dupIndex, (int)opIndex);
        }
    }
    DEV_ERROR("%s: workspace:%llx", prefix, (unsigned long long)startArgs->contextWorkspaceAddr);
    for (size_t i = 0; i < startArgs->inputTensorSize; i++) {
        DEV_ERROR("%s: input-%d:%llx", prefix, (int)i, (unsigned long long)startArgs->GetInputTensor(i).address);
    }
    for (size_t i = 0; i < startArgs->outputTensorSize; i++) {
        DEV_ERROR("%s: output-%d:%llx", prefix, (int)i, (unsigned long long)startArgs->GetOutputTensor(i).address);
    }
    std::unordered_map<uint64_t, AddressDescriptor> cacheInputOutputDict;
    DevProgramControlFlowCache::RelocBuildInputOutputDesc(cacheInputOutputDict, startArgs);
    RelocRange relocWorkspace(startArgs->contextWorkspaceAddr, 0);

    DynFuncHeader *dynFuncDataList = dyntask->GetDynFuncDataList();
    int deviceIndex = dynFuncDataList->seqNo;
    DynFuncDataCache *dynFuncDataCacheList = dyntask->GetDynFuncDataCacheList();
    for (size_t dupIndex = 0; dupIndex < dynFuncDataList->Size(); dupIndex++) {
        DynFuncData &dynFuncData = dynFuncDataList->At(dupIndex);
        DynFuncDataCache &dynFuncDataCache = dynFuncDataCacheList->At(dupIndex);

        DevAscendFunctionDuppedData *duppedData = dynFuncDataCache.duppedData;

        predcount_t *pred = &duppedData->GetOperationCurrPredCount(0);
        for (size_t opIndex = 0; opIndex < duppedData->GetOperationSize(); opIndex++) {
            DEV_ERROR("%s: L(%d,%d,%d) pred:%d\n",
                prefix,
                (int)deviceIndex, (int)dupIndex, (int)opIndex,
                (int)pred[opIndex]);
        }
        for (size_t stitchIndex = 1; stitchIndex < duppedData->GetStitchSize(); stitchIndex++) {
            DevAscendFunctionDuppedStitchList stitchList = duppedData->GetStitch(stitchIndex);
            stitchList.ForEach([&](int succTaskId){
                uint32_t succDupIndex = FuncID(succTaskId);
                uint32_t succOpIndex = TaskID(succTaskId);
                DEV_ERROR("%s: R(%d,%d).succ-%d: L(%d,%d,%d)\n",
                    prefix, (int)deviceIndex, (int)dupIndex, (int)stitchIndex,
                    (int)deviceIndex, (int)succDupIndex, (int)succOpIndex);
            });
        }
        for (size_t exprIndex = 0; exprIndex < duppedData->GetExpressionSize(); exprIndex++) {
            DEV_ERROR("%s: R(%d,%d).expr-%d: %lld\n",
                prefix, (int)deviceIndex, (int)dupIndex, (int)exprIndex,
                (long long)duppedData->GetExpression(exprIndex));
        }
        for (size_t incastIndex = 0; incastIndex < duppedData->GetIncastSize(); incastIndex++) {
            AddressDescriptor addr = duppedData->GetIncastAddress(incastIndex);
            AddressDescriptor addrDesc = addr;
            DevProgramControlFlowCache::RelocDescToCache(addrDesc, relocWorkspace, cacheInputOutputDict);

            DEV_ERROR("%s: R(%d,%d).incast-%d: 0x%llx - 0x%llx\n",
                prefix, (int)deviceIndex, (int)dupIndex, (int)incastIndex,
                (unsigned long long)addrDesc.GetAddressValue(),
                (unsigned long long)addr.GetAddressValue());
        }
        for (size_t outcastIndex = 0; outcastIndex < duppedData->GetOutcastSize(); outcastIndex++) {
            AddressDescriptor addr = duppedData->GetOutcastAddress(outcastIndex);
            AddressDescriptor addrDesc = addr;
            DevProgramControlFlowCache::RelocDescToCache(addrDesc, relocWorkspace, cacheInputOutputDict);
            DEV_ERROR("%s: R(%d,%d).outcast-%d: 0x%llx - 0x%llx\n",
                prefix, (int)deviceIndex, (int)dupIndex, (int)outcastIndex,
                (unsigned long long)addrDesc.GetAddressValue(),
                (unsigned long long)addr.GetAddressValue());
        }
        DEV_ERROR("%s: R(%d,%d).workspace: 0x%llx\n",
            prefix, (int)deviceIndex, (int)dupIndex, (unsigned long long)duppedData->GetRuntimeWorkspace());
        DEV_ERROR("%s: R(%d,%d).outcastWorkspace: 0x%llx\n",
            prefix, (int)deviceIndex, (int)dupIndex, (unsigned long long)duppedData->GetRuntimeOutcastWorkspace());

        DEV_ERROR("%s: R(%d,%d).opAttrList: 0x%llx\n",
            prefix, (int)deviceIndex, (int)dupIndex, (unsigned long long)dynFuncData.opAttrs);
        DEV_ERROR("%s: R(%d,%d).opAttrList:Dupped: 0x%llx\n",
            prefix, (int)deviceIndex, (int)dupIndex, (unsigned long long)duppedData->GetSource()->GetSymoffset(0));

        DEV_ERROR("%s: R(%d,%d).opAttrOffsetList: 0x%llx\n",
            prefix, (int)deviceIndex, (int)dupIndex, (unsigned long long)dynFuncData.opAtrrOffsets);
        DEV_ERROR("%s: R(%d,%d).opAttrOffsetList:Dupped: 0x%llx\n",
            prefix, (int)deviceIndex, (int)dupIndex, (unsigned long long)duppedData->GetSource()->GetOpAttrOffsetAddr());

        DEV_ERROR("%s: R(%d,%d).exprTbl: 0x%llx\n",
            prefix, (int)deviceIndex, (int)dupIndex, (unsigned long long)dynFuncData.exprTbl);
        DEV_ERROR("%s: R(%d,%d).exprTbl:Dupped: 0x%llx\n",
            prefix, (int)deviceIndex, (int)dupIndex, (unsigned long long)duppedData->GetExpressionAddr());

        DEV_ERROR("%s: R(%d,%d).rawTensorDesc: 0x%llx\n",
            prefix, (int)deviceIndex, (int)dupIndex, (unsigned long long)dynFuncData.rawTensorDesc);
        DEV_ERROR("%s: R(%d,%d).rawTensorDesc:Dupped: 0x%llx\n",
            prefix, (int)deviceIndex, (int)dupIndex, (unsigned long long)duppedData->GetSource()->GetRawTensorDesc(0));

        DEV_ERROR("%s: R(%d,%d).rawTensorDesc: 0x%llx\n",
            prefix, (int)deviceIndex, (int)dupIndex, (unsigned long long)dynFuncData.rawTensorAddr);
        DEV_ERROR("%s: R(%d,%d).rawTensorDesc:Dupped: 0x%llx\n",
            prefix, (int)deviceIndex, (int)dupIndex, (unsigned long long)&duppedData->GetIncastAddress(0));
    }
}

int DeviceTaskContext::BuildDeviceTaskDataAndReadyQueue(DynDeviceTask *dyntask, uint32_t taskId, DevAscendProgram *devProg) {
    int result = DEVICE_MACHINE_OK;
    dyntask->cceBinary = devProg->GetCceBinary(0);
    dyntask->aicpuLeafBinary = devProg->GetAicpuLeafBinary(0);
    DeviceStitchContext::CheckStitch(dyntask);

    DEV_VERBOSE_DEBUG("Build ready queue");
    PerfBegin(PERF_EVT_READY_QUEUE);
    result = BuildReadyQueue(dyntask, devProg);
    if (unlikely(result != DEVICE_MACHINE_OK)) {
        return DEVICE_MACHINE_ERROR;
    }
    PerfEnd(PERF_EVT_READY_QUEUE);

    PerfBegin(PERF_EVT_RESOLVE_EARLY);
    ResolveEarlyDepends(dyntask);
    PerfEnd(PERF_EVT_RESOLVE_EARLY);

    DEV_VERBOSE_DEBUG("Build func data");
    PerfBegin(PERF_EVT_CORE_FUNCDATA);
    result = BuildDynFuncData(dyntask, taskId, devProg, &dyntask->stitchedList[0], dyntask->stitchedList.size());
    if (unlikely(result != DEVICE_MACHINE_OK)) {
        return DEVICE_MACHINE_ERROR;
    }
    PerfEnd(PERF_EVT_CORE_FUNCDATA);
    DEV_INFO("Finish build a new device task");

    DEV_IF_NONDEVICE {
        dyntask->DumpTopo();
    }

#if DEBUG_INFINITE_LIFETIME
    DEV_IF_DEVICE {
        dyntask->DumpTensorAddrInfo(workspace_->DumpTensorWsBaseAddr(), workspace_->DumpTensorWsSize());
    }
#endif
    DEV_IF_VERBOSE_DEBUG {
        dyntask->DumpLeafs();
    }

    DEV_IF_DEBUG {
        int funcIdx = 0;
        for (auto &func : dyntask->stitchedList) {
            DEV_DEBUG_SPLIT("func %d %s.", funcIdx, func.DumpDyn(funcIdx, dyntask->cceBinary).c_str());
            funcIdx++;
            (void)func;
        }
    }
    dyntask->stitchedList.clear();
    return result;
}
}