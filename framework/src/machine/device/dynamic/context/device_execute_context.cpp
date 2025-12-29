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
 * \file device_execute_context.cpp
 * \brief
 */

#include "machine/device/dynamic/context/device_execute_context.h"

#include <cinttypes>

namespace npu::tile_fwk::dynamic {
bool DeviceExecuteContext::DuppedRootCached() {
    if (!controlFlowCacheActivated) {
        return false;
    }
    return duppedRootCount < devProg->controlFlowCache.rootTaskCount;
}

bool DeviceExecuteContext::DuppedRootUpdateAndCachedAllSubmitted() {
    if (!controlFlowCacheActivated) {
        return false;
    }
    duppedRootCount++;
    return duppedRootCount == devProg->controlFlowCache.rootTaskCount;
}

uint64_t DeviceExecuteContext::GetInputShapeDimSize(DeviceExecuteContext *ctx, uint64_t inputIndex) {
    DevTensorData *input = &ctx->args->devTensorList[inputIndex];
    return input->shape.dimSize;
}

uint64_t DeviceExecuteContext::GetInputShapeDim(DeviceExecuteContext *ctx, uint64_t inputIndex, uint64_t n) {
    DevTensorData *input = &ctx->args->devTensorList[inputIndex];
    return input->shape.dim[n];
}

int64_t DeviceExecuteContext::GetInputDataInt32Dim1(DeviceExecuteContext *ctx, uint64_t inputIndex, uint64_t off0) {
    DevTensorData *input = &ctx->args->devTensorList[inputIndex];
    return (reinterpret_cast<int32_t *>(input->address))[off0];
}

int64_t DeviceExecuteContext::GetInputDataInt32Dim2(DeviceExecuteContext *ctx, uint64_t inputIndex, uint64_t off0, uint64_t off1) {
    DevTensorData *input = &ctx->args->devTensorList[inputIndex];
    return (reinterpret_cast<int32_t *>(input->address))[off0 * input->shape.dim[1] + off1];
}

int64_t DeviceExecuteContext::GetInputDataInt32Dim3(DeviceExecuteContext *ctx, uint64_t inputIndex, uint64_t off0, uint64_t off1, uint64_t off2) {
    DevTensorData *input = &ctx->args->devTensorList[inputIndex];
    return (reinterpret_cast<int32_t *>(input->address))[off0 * input->shape.dim[1] * input->shape.dim[2] + off1 * input->shape.dim[2] + off2]; // 2: dim 2
}

int64_t DeviceExecuteContext::GetInputDataInt32Dim4(DeviceExecuteContext *ctx, uint64_t inputIndex, uint64_t off0, uint64_t off1,
    uint64_t off2, uint64_t off3) {
    DevTensorData *input = &ctx->args->devTensorList[inputIndex];
    return (reinterpret_cast<int32_t *>(input->address))[((off0 * input->shape.dim[1] + off1) * input->shape.dim[2] + off2) * input->shape.dim[3] + off3]; // 2: dim 2, 3: dim 3
}

void *DeviceExecuteContext::SymbolHandlerIdToHandler(SymbolHandlerId id) {
    switch (id) {
        case SymbolHandlerId::GetInputShapeDimSize:
            return reinterpret_cast<void *>(GetInputShapeDimSize);
        case SymbolHandlerId::GetInputShapeDim:
            return reinterpret_cast<void *>(GetInputShapeDim);
        case SymbolHandlerId::GetInputDataInt32Dim1:
            return reinterpret_cast<void *>(GetInputDataInt32Dim1);
        case SymbolHandlerId::GetInputDataInt32Dim2:
            return reinterpret_cast<void *>(GetInputDataInt32Dim2);
        case SymbolHandlerId::GetInputDataInt32Dim3:
            return reinterpret_cast<void *>(GetInputDataInt32Dim3);
        case SymbolHandlerId::GetInputDataInt32Dim4:
            return reinterpret_cast<void *>(GetInputDataInt32Dim4);
        default:
            DEV_ERROR("Invalid SymbolHandlerId: %lu", static_cast<uint64_t>(id));
            DEV_ASSERT(0);
            return nullptr;
    }
    return nullptr;
}

int DeviceExecuteContext::RunInit(DevStartArgs *startArgs, PushTaskEntry tPushTask) {
    PerfBegin(PERF_EVT_CONTROL_FLOW_INIT);
    this->pushTask = tPushTask;
    this->args = startArgs;
    this->devProg = startArgs->devProg;

    workspace.Init(startArgs);
    if (devProg->stitchFunctionNumInitial > 0) {
        stitchTaskLoopNumThreshold = std::min<uint16_t>(devProg->stitchFunctionNumInitial, MAX_CACHED_FUNC_NUM);
        DEV_INFO("First stitch task loop num threshold is %u.", stitchTaskLoopNumThreshold);
    }

    slotContext.InitAllocator(workspace, devProg->slotSize);
    slotContext.FillInputOutputSlot(devProg, startArgs);

    stitchContext.Init(devProg, workspace);

    taskContext.InitAllocator(devProg, workspace, startArgs);

    workspace.SetupVector(symbolTable);
    symbolTable.resize(devProg->symbolTable.size());
    for (int index = 0; index < startArgs->GetInputSymbolSize(); ++index) {
        DevInputSymbol &param = startArgs->GetInputSymbol(index);
        int inputSymbolIndex = this->devProg->startArgsInputSymbolIndexList[index];
        symbolTable[inputSymbolIndex] = param.value;
        DEV_INFO("Param %d Symbol Table %d = %ld.", index, inputSymbolIndex, param.value);
    }

    for (size_t index = 0; index < this->devProg->startArgsSymbolHandlerList.size(); ++index) {
        SymbolHandler &symbolHandler = this->devProg->startArgsSymbolHandlerList[index];
        void *handler = SymbolHandlerIdToHandler(symbolHandler.handlerId);
        if (handler == nullptr) {
            return DEVICE_MACHINE_ERROR;
        }
        DEV_ASSERT_MSG(handler, "handler not found.");
        symbolTable[symbolHandler.symIndex] = PtrToValue(handler);
    }

    /* This initialization must only occur after all other AICPU workspace meta memory allocations have completed.
        The remaining portion of AICPU workspace meta memory must support reclamation. */
    workspace.InitMetadataSlabAllocator();

    PerfEnd(PERF_EVT_CONTROL_FLOW_INIT);
    DEV_INFO("Image size is %lu.", devProg->GetSize());
    return DEVICE_MACHINE_OK;
}

DeviceExecuteContext::DeviceExecuteContext(DevStartArgs *startArgs) {
    PerfBegin(PERF_EVT_INIT);
    this->devProg = startArgs->devProg;

    DEV_IF_VERBOSE_DEBUG {
        std::string dump = devProg->Dump(0, true);
        DEV_VERBOSE_DEBUG("[DEVICE] %s.", dump.c_str());
    }

    PerfBegin(PERF_EVT_CONTROL_FLOW_MAPEXE);
    execProg = DeviceExecuteProgram(devProg, reinterpret_cast<AOTBinaryControlFlow::controlFlowEntry>(const_cast<void *>(startArgs->controlFlowEntry)));
    AOTCodePool::GetCodePool().MapExec();
    PerfEnd(PERF_EVT_CONTROL_FLOW_MAPEXE);
    PerfEnd(PERF_EVT_INIT);
}

void DeviceExecuteContext::PushTask(DynDeviceTask *dynTask) {
    pushTask(dynTask, this);
    taskId++;
}

void DeviceExecuteContext::ShowStats() {
    taskContext.ShowStats();
    workspace.DumpMemoryUsage("End ExecDyn");
}

void DeviceExecuteContext::GELaunchRunCached(DevStartArgs *startArgs, PushTaskEntry tPushTask) {
    PerfBegin(PERF_EVT_CONTROL_FLOW_INIT);
    this->pushTask = tPushTask;
    this->args = startArgs;
    this->devProg = startArgs->devProg;
    PerfEnd(PERF_EVT_CONTROL_FLOW_INIT);
    PerfMtTrace(PERF_TRACE_INIT, CTRL_CPU_THREAD_IDX);
    PerfBegin(PERF_EVT_CONTROL_FLOW);
    for (size_t index = 0; index < devProg->controlFlowCache.deviceTaskCount; index++) {
        DynDeviceTask *dynTask = reinterpret_cast<DynDeviceTask *>(devProg->controlFlowCache.deviceTaskCacheList[index].dynTaskBase);
        devProg->controlFlowCache.PredCountDataRestore(dynTask);
        devProg->controlFlowCache.ReadyQueueDataRestore(dynTask);
        taskContext.UpdateReadyTaskNum(dynTask->readyQueueBackup->readyTaskNum);

        PROF_STAGE_BEGIN(PERF_EVT_STAGE_PUSH_TASK, "push.before\n");
        DumpDeviceTask(taskId, dynTask);
        PushTask(dynTask);
        PerfMtTrace(PERF_TRACE_DEV_TASK_BUILD, CTRL_CPU_THREAD_IDX);
        PROF_STAGE_END(PERF_EVT_STAGE_PUSH_TASK, "push.after\n");
    }
    PerfEnd(PERF_EVT_CONTROL_FLOW);
}

int DeviceExecuteContext::RunControlFlow(DevStartArgs *startArgs) {
    PerfBegin(PERF_EVT_CONTROL_FLOW);
    RuntimeCallEntryType runtimeCallList[static_cast<uint32_t>(RuntimeCallStage::T_RUNTIME_CALL_MAX)] = {
        DeviceExecuteRuntimeCallRootAlloc,
        DeviceExecuteRuntimeCallRootStitch,
        DeviceExecuteRuntimeCallLog,
        DeviceExecuteRuntimeCallShmemAllocator,
        DeviceExecuteRuntimeCallSlotMarkNeedAlloc,
    };
    int originalErrorState = this->GetErrorState();
    execProg.controlFlowBinary.CallControlFlow(this, symbolTable.data(), runtimeCallList, startArgs);
    int finalErrorState = this->GetErrorState();
    if (finalErrorState != originalErrorState && finalErrorState != DEVICE_MACHINE_OK) {
        DEV_ERROR("Control flow execution failed with error code: %d", finalErrorState);
        return finalErrorState;
    }
    PerfEnd(PERF_EVT_CONTROL_FLOW);
    return DEVICE_MACHINE_OK;
}

int DeviceExecuteContext::GELaunchFullCacheRunControlFlow(DevStartArgs *startArgs, PushTaskEntry tPushTask) {
    int ret = DEVICE_MACHINE_OK;
    ret = RunInit(startArgs, tPushTask);
    if (unlikely(ret != DEVICE_MACHINE_OK)) {
        return DEVICE_MACHINE_ERROR;
    }
    ret = RunControlFlow(startArgs);
    if (unlikely(ret != DEVICE_MACHINE_OK)) {
        return DEVICE_MACHINE_ERROR;
    }
    return ret;
}

void DeviceExecuteContext::GELaunchFullCache(DevStartArgs *startArgs, PushTaskEntry tPushTask) {
    if (devProg->controlFlowCache.IsActivatedFullCache(startArgs)) {
        DEV_TRACE_DEBUG(CtrlEvent(none(), ControlFlowCacheFullRunCache()));
        GELaunchRunCached(startArgs, tPushTask);
    } else {
        DEV_TRACE_DEBUG(CtrlEvent(none(), ControlFlowCacheFullRunControl()));
        GELaunchFullCacheRunControlFlow(startArgs, tPushTask);
    }
}

int DeviceExecuteContext::GELaunch(DevStartArgs *startArgs, PushTaskEntry tPushTask) {
    int ret = DEVICE_MACHINE_OK;
    if (devProg->controlFlowCache.IsRecording()) {
        devProg->controlFlowCache.InitInputOutput(startArgs);
    }
    ret = GELaunchPartialCache(startArgs, tPushTask);
    if (unlikely(ret != DEVICE_MACHINE_OK)) {
        return DEVICE_MACHINE_ERROR;
    }
    return DEVICE_MACHINE_OK;
}

int DeviceExecuteContext::GELaunchPartialCache(DevStartArgs *startArgs, PushTaskEntry tPushTask) {
    int ret = DEVICE_MACHINE_OK;
    DEV_TRACE_DEBUG(CtrlEvent(none(), Workspace(Range(startArgs->contextWorkspaceAddr, startArgs->contextWorkspaceAddr + startArgs->contextWorkspaceSize))));

    if (devProg->controlFlowCache.IsActivatedPartialCache(startArgs)) {
        controlFlowCacheActivated = true;
        DEV_TRACE_DEBUG(CtrlEvent(none(), ControlFlowCachePartRunCache(devProg->controlFlowCache.deviceTaskCount, devProg->controlFlowCache.rootTaskCount)));
        GELaunchRunCached(startArgs, tPushTask);
    }
DEV_IF_DEVICE {
    uint64_t start = GetCycles();
    while ((startArgs->devProg->devArgs.disableSync == 0) && startArgs->syncFlag != 1) {
        if (GetCycles() - start > HAND_SHAKE_TIMEOUT) {
            DEV_ERROR("Wait sync flag timeout.");
            break;
        }
    }
}
    DEV_TRACE_DEBUG(CtrlEvent(none(), ControlFlowCacheFullRunControl()));
    ret = RunInit(startArgs, tPushTask);
    if (unlikely(ret != DEVICE_MACHINE_OK)) {
        return DEVICE_MACHINE_ERROR;
    }
    ret = RunControlFlow(startArgs);
    if (unlikely(ret != DEVICE_MACHINE_OK)) {
        return DEVICE_MACHINE_ERROR;
    }
    return ret;
}

bool DeviceExecuteContext::AiCoreFree() {
    return false; // extend check point
}

void DeviceExecuteContext::DumpDeviceTask(uint64_t taskId, DynDeviceTask *deviceTask) {
    DEV_IF_VERBOSE_DEBUG {
    } else {
        return;
    }
    for (uint64_t dupIdx = 0; dupIdx < deviceTask->dynFuncDataCacheListSize; dupIdx++) {
        DevAscendFunctionDuppedData *dupped = deviceTask->dynFuncDataCacheList[dupIdx].duppedData;
        DEV_TRACE_DEBUG(REvent(RUid(taskId, dupIdx, dupped->GetSource()->GetRootIndex()), dupped->SchemaGetWorkspace()));
        size_t incastSize = dupped->GetSource()->GetIncastSize();
        DEV_TRACE_DEBUG(REvent(RUid(taskId, dupIdx, dupped->GetSource()->GetRootIndex()), RActIncastCount(incastSize)));
        for (size_t i = 0; i < incastSize; ++i) {
            DEV_TRACE_DEBUG(REvent(RUid(taskId, dupIdx, dupped->GetSource()->GetRootIndex()), RActIncast(i, dupped->SchemaGetIncastRange(i))));
        }

        size_t outcastSize = dupped->GetSource()->GetOutcastSize();
        DEV_TRACE_DEBUG(REvent(RUid(taskId, dupIdx, dupped->GetSource()->GetRootIndex()), RActOutcastCount(outcastSize)));
        for (size_t i = 0; i < outcastSize; ++i) {
            DEV_TRACE_DEBUG(REvent(RUid(taskId, dupIdx, dupped->GetSource()->GetRootIndex()), RActOutcast(i, dupped->SchemaGetOutcastRange(i))));
        }
    }
}

int DeviceExecuteContext::SubmitToAicoreAndRecycleMemory(bool withoutTail, bool isLastTask) {
    int ret = DEVICE_MACHINE_OK;
    DEV_VERBOSE_DEBUG("Submit stitch task.");
    DEV_TRACE_DEBUG(DEvent(taskId, DActSubmit(stitchContext.Size())));
    AutoScopedPerf asp(PERF_EVT_SUBMIT_AICORE);
    if (stitchContext.Empty()) {
        DEV_INFO("Stitch context is empty.");
        return ret;
    }

    PROF_STAGE_BEGIN(PERF_EVT_DECIDE_SLOT_ADDRESS, "slotaddr.before\n");
    ret = stitchContext.DecideSlotAddress(
        slotContext.GetSlotList(), slotContext.GetSlotSize(), slotContext.GetSlotRefCntPool());
    PROF_STAGE_END(PERF_EVT_DECIDE_SLOT_ADDRESS, "slotaddr.after\n");
    if (unlikely(ret != DEVICE_MACHINE_OK)) {
        return DEVICE_MACHINE_ERROR;
    }

    PROF_STAGE_BEGIN(PERF_EVT_DECIDE_INCAST_ADDRESS, "incastaddr.before\n");
    ret = stitchContext.DecideIncastOutcast(taskId);
    PROF_STAGE_END(PERF_EVT_DECIDE_INCAST_ADDRESS, "incastaddr.after\n");
    if (unlikely(ret != DEVICE_MACHINE_OK)) {
        return DEVICE_MACHINE_ERROR;
    }

    DEV_IF_VERBOSE_DEBUG {
            stitchContext.DumpStitchInfo();
#if !DEBUG_INFINITE_LIFETIME
            stitchContext.VerifyStitchedListMemory(*args);
#endif
    }

#if DEBUG_MEM_DUMP_LEVEL >= DEBUG_MEM_DUMP_FULL
    workspace.MarkAsNewStitchWindow();
#endif // DEBUG_MEM_DUMP_LEVEL >= DEBUG_MEM_DUMP_FULL

    PROF_STAGE_BEGIN(PERF_EVT_STAGE_BUILD_TASK, "BuildDeviceTaskData.before\n");
    DynDeviceTask *dynTask = taskContext.BuildDeviceTaskData(stitchContext, taskId, devProg, withoutTail);
    if (dynTask == nullptr) {
        DEV_ERROR("Build device task data failed.");
        return DEVICE_MACHINE_ERROR;
    }
    dynTask->SetLastTask(isLastTask);
    PROF_STAGE_END(PERF_EVT_STAGE_BUILD_TASK, "BuildDeviceTaskData.after\n");

    PROF_STAGE_BEGIN(PERF_EVT_DEALLOCATE_WORKSPACE, "RecycleTensorWorkspace.before\n");
    // Memory recycling
    stitchContext.RecycleTensorWorkspace();

    // Reset stitch context
    stitchContext.Reset();
    slotContext.ClearDirty();
    PROF_STAGE_END(PERF_EVT_DEALLOCATE_WORKSPACE, "RecycleTensorWorkspace.after\n");

    if (devProg->controlFlowCache.IsRecording()) {
        if (!devProg->controlFlowCache.IsRecordingStopped()) {
            devProg->controlFlowCache.PredCountDataBackup(dynTask);
            devProg->controlFlowCache.ReadyQueueDataBackup(dynTask);
            devProg->controlFlowCache.IncastOutcastAddrBackup(dynTask);
            devProg->controlFlowCache.TaskAddrBackupWorkspace(dynTask);
            devProg->controlFlowCache.RuntimeAddrBackup(slotContext.GetSlotList(), &slotContext.GetSlotRefCntPool().At(0), devProg->slotSize, workspace.GetTensorAllocator());
        }
        devProg->controlFlowCache.AppendDeviceTask(dynTask);
    }

    PROF_STAGE_BEGIN(PERF_EVT_STAGE_PUSH_TASK, "push.before\n");
    DumpDeviceTask(taskId, dynTask);
    PushTask(dynTask);
    PROF_STAGE_END(PERF_EVT_STAGE_PUSH_TASK, "push.after\n");
    PerfMtTrace(PERF_TRACE_DEV_TASK_BUILD, CTRL_CPU_THREAD_IDX);
    return ret;
}

schema::RUid DeviceExecuteContext::GetRuid(uint64_t rootKey, bool afterAppend) {
    int64_t dupIndex = stitchContext.Size();
    if (afterAppend) {
        dupIndex -= 1;
    }
    schema::RUid ruid(taskId, dupIndex, rootKey);
    return ruid;
}

int DeviceExecuteContext::ControlFlowCacheStopCache(uint64_t rootKey) {
    int ret = DEVICE_MACHINE_OK;
    ret = SubmitToAicoreAndRecycleMemory(false);
    if (unlikely(ret != DEVICE_MACHINE_OK)) {
        return DEVICE_MACHINE_ERROR;
    }
    devProg->controlFlowCache.StopRecording();
    DEV_INFO("[Stitch Finish]Stop recording ctrl flow cache. rootKey=%" PRIu64 ".", rootKey);
    return ret;
}

void *DeviceExecuteContext::CallRootFunctionAlloc(uint64_t rootKey) {
    int ret = DEVICE_MACHINE_OK;
    DevAscendFunction *devRoot = devProg->GetFunction(rootKey);
    DEV_DEBUG("Slloc one func %lu %p %s.", rootKey, devRoot, devRoot->GetRawName());
    if (stitchContext.Size() == stitchTaskLoopNumThreshold ||
        stitchContext.stitchedCallOpSize() + devRoot->GetOperationSize() > devProg->stitchFunctionsize) {
        ret = SubmitToAicoreAndRecycleMemory(false);
        if (unlikely(ret != DEVICE_MACHINE_OK)) {
            return RUNTIME_FUNCKEY_ERROR;
        }
        auto nextThreshold =
            std::min<uint16_t>(stitchTaskLoopNumThreshold + devProg->stitchFunctionNumStep, MAX_CACHED_FUNC_NUM);
        DEV_INFO("[Stitch Finish]Stitch Limit Exceeded. #task=%zu+1 (limit=%u), #callop=%u+%zu (limit=%u).",
            stitchContext.Size(), stitchTaskLoopNumThreshold,
            stitchContext.stitchedCallOpSize(), devRoot->GetOperationSize(), devProg->stitchFunctionsize);
        stitchTaskLoopNumThreshold = nextThreshold;
    }
    DEV_TRACE_DEBUG(REvent(GetRuid(rootKey), RActDup(devRoot->GetRawName())));

    PROF_STAGE_BEGIN(PERF_EVT_STAGE_DUP_ROOT, "dup.before\n");
    currDevRootDup = workspace.DuplicateRoot(devRoot);
    PROF_STAGE_END(PERF_EVT_STAGE_DUP_ROOT, "dup.after\n");
    return reinterpret_cast<void *>(&currDevRootDup.GetExpression(0));
}

void *DeviceExecuteContext::CallRootFunctionStitch(uint64_t rootKey) {
    int ret = DEVICE_MACHINE_OK;
    DEV_DEBUG("Root stitch %lu.", rootKey);
    if (rootKey == RUNTIME_FUNCKEY_CACHESTOP) {
        if (devProg->controlFlowCache.IsRecording()) {
            ret = ControlFlowCacheStopCache(rootKey);
            if (unlikely(ret != DEVICE_MACHINE_OK)) {
                return RUNTIME_FUNCKEY_ERROR;
            }
            return RUNTIME_FUNCRET_CACHESTOP_RETURN;
        } else {
            return RUNTIME_FUNCRET_CACHESTOP_CONTINUE;
        }
    }
    if (rootKey == RUNTIME_FUNCKEY_FINISH || rootKey == RUNTIME_FUNCKEY_LOOP_BARRIER) {
        ret = SubmitToAicoreAndRecycleMemory(false, rootKey == RUNTIME_FUNCKEY_FINISH ? true : false);
        if (unlikely(ret != DEVICE_MACHINE_OK)) {
            return RUNTIME_FUNCKEY_ERROR;
        }
        DEV_INFO("[Stitch Finish] Finish Signal or Barrier. rootKey=%" PRIu64 " .", rootKey);
        return nullptr;
    }

    DEV_TRACE_DEBUG(REvent(GetRuid(rootKey), currDevRootDup.SchemaGetExpressionTable()));
    // dyn rawshape size depend expresstable calculated
    while (!workspace.TryAllocateFunctionMemory(currDevRootDup, slotContext.GetSlotList())) {
        // Failed to allocate, failed to stitch, submit existing stitched window to aicore and recycle memory
        // If nothing stitched, wait for aicore to finish tasks and release enough memory
        ret = SubmitToAicoreAndRecycleMemory(true);
        if (unlikely(ret != DEVICE_MACHINE_OK)) {
            return RUNTIME_FUNCKEY_ERROR;
        }
        DEV_INFO("[Stitch Finish] Memory Limit Exceeded .");
    }

    if (AiCoreFree()) {
        ret = SubmitToAicoreAndRecycleMemory(false);
        if (unlikely(ret != DEVICE_MACHINE_OK)) {
            return RUNTIME_FUNCKEY_ERROR;
        }
        DEV_INFO("[Stitch Finish] AICore Free .");
    }

    DEV_TRACE_DEBUG(DEvent(taskId, DActStitchStart(GetRuid(rootKey))));
    PROF_STAGE_BEGIN(PERF_EVT_STAGE_STITCH, "stitch.before\n");
    size_t devNextIdx = stitchContext.Size();
    stitchContext.Stitch(slotContext, currDevRootDup, taskId, devNextIdx);

    ret = slotContext.UpdateSlots(currDevRootDup, stitchContext.GetStitchedList(), taskId, devNextIdx);
    if (unlikely(ret != DEVICE_MACHINE_OK)) {
        return RUNTIME_FUNCKEY_ERROR;
    }
    PROF_STAGE_END(PERF_EVT_STAGE_STITCH, "stitch.after\n");
    DEV_TRACE_DEBUG(DEvent(taskId, DActStitchFinish(GetRuid(rootKey, true))));
    return nullptr;
}

void DeviceExecuteContext::MarkSlotNeedAlloc(int slotIndex) {
    DEV_ASSERT_MSG(slotIndex >= 0 && slotIndex < static_cast<int>(slotContext.GetSlotSize()),
        "MarkSlotNeedAlloc: Invalid slot index %d.", slotIndex);
    slotContext.GetSlotList()[slotIndex].isAssembleSlotNeedAlloc = true;
    return;
}

void *DeviceExecuteContext::DeviceExecuteRuntimeCallRootAlloc(void *ctx_, uint64_t rootKey) {
    DeviceExecuteContext *ctx = (DeviceExecuteContext *)ctx_;
    if (ctx == nullptr) {
        DEV_ERROR("invalid ctx.");
        return nullptr;
    }
    PerfBegin(PERF_EVT_ROOT_FUNC);
    void *result = nullptr;
    if (ctx->DuppedRootCached()) {
        result = nullptr;
    } else if (ctx->devProg->controlFlowCache.IsRecording() && ctx->devProg->controlFlowCache.IsRecordingStopped()) {
        result = nullptr;
    } else {
        result = ctx->CallRootFunctionAlloc(rootKey);
        if (result == RUNTIME_FUNCKEY_ERROR) {
            ctx->SetErrorState(DEVICE_MACHINE_ERROR);
        }
    }

    PerfEnd(PERF_EVT_ROOT_FUNC);
    return result;
}

void *DeviceExecuteContext::DeviceExecuteRuntimeCallRootStitch(void *ctx_, uint64_t rootKey) {
    DeviceExecuteContext *ctx = (DeviceExecuteContext *)ctx_;
    if (ctx == nullptr) {
        DEV_ERROR("invalid ctx.");
        return nullptr;
    }
    PerfBegin(PERF_EVT_ROOT_FUNC);
    void *result = nullptr;
    if (ctx->DuppedRootCached()) {
        result = nullptr;
    } else if (ctx->devProg->controlFlowCache.IsRecording() && ctx->devProg->controlFlowCache.IsRecordingStopped()) {
        result = nullptr;
    } else {
        result = ctx->CallRootFunctionStitch(rootKey);
        if (result == RUNTIME_FUNCKEY_ERROR) {
            ctx->SetErrorState(DEVICE_MACHINE_ERROR);
        }
    }
    PerfEnd(PERF_EVT_ROOT_FUNC);
    if (ctx->DuppedRootUpdateAndCachedAllSubmitted()) {
        DEV_TRACE_DEBUG(CtrlEvent(none(), ControlFlowCachePartRunControlContinue()));
        // forcely break device task
        ctx->devProg->controlFlowCache.RuntimeAddrRestore(ctx->slotContext.GetSlotList(), &ctx->slotContext.GetSlotRefCntPool().At(0), ctx->devProg->slotSize, ctx->workspace.GetTensorAllocator());
        ctx->devProg->controlFlowCache.RuntimeAddrRelocWorkspace(0, ctx->args->contextWorkspaceAddr, ctx->args, ctx->slotContext.GetSlotList());
    }
    return result;
}

void *DeviceExecuteContext::DeviceExecuteRuntimeCallLog(void *ctx_, uint64_t value) {
    (void)ctx_;
    DEV_DEBUG("DeviceExecuteRuntimeCallLog -> Value: %lu", value);
#if DEBUG_PLOG
    (void)value;
#endif
    return nullptr;
}

void *DeviceExecuteContext::DeviceExecuteRuntimeCallShmemAllocator(void *ctx_, uint64_t value) {
    (void)ctx_;
    uint64_t groupIndex = (reinterpret_cast<uint64_t*>(value))[0];
    uint64_t memType = (reinterpret_cast<uint64_t*>(value))[1];
    uint64_t size = (reinterpret_cast<uint64_t*>(value))[2];
    static uint64_t offset = 0UL;
    constexpr uint64_t OFFSET_BITS = 58UL;
    constexpr uint64_t GROUP_BITS = 2UL;
    constexpr uint64_t MEMTYPE_BITS = 2UL;
    constexpr uint64_t GROUP_SHIFT = OFFSET_BITS;
    constexpr uint64_t MEMTYPE_SHIFT = GROUP_SHIFT + GROUP_BITS;
    constexpr uint64_t FILL_SHIFT = MEMTYPE_SHIFT + MEMTYPE_BITS;
    uint64_t vaddr = offset | (groupIndex << GROUP_SHIFT) | (memType << MEMTYPE_SHIFT) | (1UL << FILL_SHIFT);
    offset += size;
    return reinterpret_cast<void*>(vaddr);
}

void *DeviceExecuteContext::DeviceExecuteRuntimeCallSlotMarkNeedAlloc(void *ctx_, uint64_t slotIndex) {
    DeviceExecuteContext *ctx = (DeviceExecuteContext *)ctx_;
    ctx->MarkSlotNeedAlloc(slotIndex);
    return nullptr;
}
}
