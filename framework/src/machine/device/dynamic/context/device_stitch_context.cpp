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
 * \file device_stitch_context.cpp
 * \brief
 */

#include "machine/device/dynamic/context/device_stitch_context.h"
#include "machine/device/dynamic/context/dump_device_topo.h"
#include "machine/utils/dynamic/dev_cell_match_mem_layout.h"
#include "machine/utils/dynamic/dev_stitch_dependency_enhanced.h"

namespace npu::tile_fwk::dynamic {
namespace {
struct HandleCellMatchPartial {
    static inline uint32_t Process(int index, uint64_t* cellMatchTableData, uint64_t* matchCount,
                                   DevAscendFunctionDupped* stitchingList, int stitchingSize,
                                   DevAscendFunctionDupped* nextDup, size_t devTaskId, size_t devNextIdx,
                                   int consumerOperationIdx, DeviceWorkspaceAllocator* workspace, int debugSlotIdx)
    {
        uint64_t id = cellMatchTableData[index];
        if (id != AICORE_TASK_INIT && devTaskId == static_cast<uint32_t>(id >> CELL_MATCH_META_TAGID_SHIFT32)) {
            auto funcId = FuncID(static_cast<uint32_t>(id));
            auto producerOperationIdx = TaskID(static_cast<uint32_t>(id));
            DevAscendFunctionDupped& prevDup = stitchingList[funcId];
            (*matchCount)++;
            DEV_VERBOSE_DEBUG("nextindex %lu stitch depend slot table cell[%d] = taskid(%u ! %u),", devNextIdx, index,
                              funcId, producerOperationIdx);
            DeviceStitchContext::HandleOneStitch(
                prevDup, *nextDup, funcId, producerOperationIdx, devNextIdx, consumerOperationIdx, workspace,
                DeviceStitchContext::StitchKind::StitchPartial, debugSlotIdx, static_cast<uint64_t>(devTaskId));
            DeviceStitchContext::CheckStitch(stitchingList, stitchingSize, nextDup);
        }
        return 0;
    }
};

struct HandleCellMatchFull {
    static inline uint32_t Process(int index, uint32_t* cellMatchTableData, uint64_t* matchCount,
                                   DevAscendFunctionDupped* prevDup, DevAscendFunctionDupped* nextDup,
                                   size_t devNextIdx, int consumerOperationIdx, DeviceWorkspaceAllocator* workspace,
                                   int debugSlotIdx, size_t devTaskId, uint32_t preFuncIndex)
    {
        auto producerOperationIdx = cellMatchTableData[index];
        if (producerOperationIdx != static_cast<uint32_t>(-1)) {
            (*matchCount)++;
            DEV_TRACE_DEBUG(
                DEvent(DUid(none()), DActStitchEdge(Producer(LUid(none(), 0, none(), producerOperationIdx, none()),
                                                             none(), none(), debugSlotIdx, none(), none()),
                                                    Consumer(LUid(none(), 0, none(), consumerOperationIdx, none()),
                                                             none(), none(), debugSlotIdx, none(), none()),
                                                    StitchReasonUniqueMatch())));
            DEV_VERBOSE_DEBUG("FullCoverUpdateStitch HandleCellMatchFull handle one stitch [%u] -> [%u!%u]",
                              producerOperationIdx, static_cast<uint32_t>(devNextIdx),
                              static_cast<uint32_t>(consumerOperationIdx));
            DeviceStitchContext::HandleOneStitch(
                *prevDup, *nextDup, preFuncIndex, producerOperationIdx, devNextIdx, consumerOperationIdx, workspace,
                DeviceStitchContext::StitchKind::StitchDefault, debugSlotIdx, static_cast<uint64_t>(devTaskId));
        }
        return 0;
    }
};
} // namespace
void DeviceStitchContext::Init(DevAscendProgram* devProg, DeviceWorkspaceAllocator& workspace)
{
    workspace_ = &workspace;
    workspace_->SetupVector(stitchedList_);
    devProg_ = devProg;

    Reset();
}

void DeviceStitchContext::Reset()
{
    stitchedList_.clear();
    stitchedNum_ = 0;
    stitchReuseContext_.firstDupIdx = 0;
    stitchReuseContext_.lastNonEmptyDupIdx = -1;
}

void DeviceStitchContext::DumpStitchInfo() { DumpStitchInfo(stitchedList_.data(), stitchedList_.size()); }

void DeviceStitchContext::CheckStitch(DevAscendFunctionDupped* stitchedList, int size, DevAscendFunctionDupped* nextDup)
{
    DEV_IF_NONDEVICE
    {
        uint32_t dynPredCount = 0;
        uint32_t dynSuccCount = 0;
        for (int k = 0; k <= size; k++) {
            DevAscendFunctionDupped* dup = nullptr;
            if (k < size) {
                dup = &stitchedList[k];
            } else if (nextDup != nullptr) {
                dup = nextDup;
            } else {
                break;
            }
            auto src = dup->GetSource();
            for (size_t i = 0; i < dup->GetOperationSize(); i++) {
                auto opPredCount = src->GetOperationDepGraphPredCount(i);
                auto opDynPredCount = dup->GetOperationCurrPredCount(i);
                dynPredCount += opDynPredCount - opPredCount;
                auto succStitchList = dup->GetOperationStitch(i);
                for (auto p = succStitchList.Head(); p != nullptr; p = p->Next()) {
                    dynSuccCount += p->Size();
                }
            }
        }
        if (dynPredCount != dynSuccCount) {
            DEV_ERROR(ProgEncodeErr::STITCH_PRED_SUCC_MISMATCH,
                      "#ctrl.task.pre.stitch.check: dynPredCount %u does not match dynSuccCount %u", dynPredCount,
                      dynSuccCount);
        }
        DEV_ASSERT(ProgEncodeErr::STITCH_PRED_SUCC_MISMATCH, dynPredCount == dynSuccCount);
    }
}

void DeviceStitchContext::CheckStitch(DynDeviceTask* dyntask)
{
    DevAscendFunctionDupped* stitchedList = &dyntask->stitchedList[0];
    int stitchedSize = dyntask->stitchedList.size();
    CheckStitch(stitchedList, stitchedSize, nullptr);
}

uint64_t DeviceStitchContext::Stitch(DeviceSlotContext& slotContext, DevAscendFunctionDupped& nextDup, size_t devTaskId,
                                     size_t devNextIdx)
{
    uint64_t count = FastStitch(slotContext.GetSlotList(), slotContext.GetSlotSize(), nextDup, devTaskId, devNextIdx);
    if (stitchedList_.capacity() == 0) {
        /* This stitchedList_ vector can only allocate sufficient space once,
            during a single device task construction process.*/
        stitchedList_.reserve(MAX_STITCH_FUNC_NUM);
    }
    Append(nextDup);
    stitchedCallOpSize_ += (nextDup.GetSource()->GetOperationSize() - nextDup.GetSource()->hubOpCount_);
    return count;
}

void DeviceStitchContext::RecycleTensorWorkspace()
{
    // recycle submitted tasks' workspace memory
    workspace_->RecycleDevFuncWorkspace();
    workspace_->TriggerDelayedRecycle();
}

void DeviceStitchContext::DumpSlotInfo(const char* label, DeviceExecuteSlot* slotList, size_t slotSize)
{
    UNUSED(label);
    UNUSED(slotList);
    UNUSED(slotSize);
    DEV_IF_VERBOSE_DEBUG
    {
        DEV_DEBUG("[DecideSlotAddress] %s.", label);
        for (size_t slotIdx = 0; slotIdx < slotSize; slotIdx++) {
            [[maybe_unused]] const char* extraAttr = "";
            if (slotList[slotIdx].isOutputSlot) {
                extraAttr = " <output>";
            } else if (slotList[slotIdx].isAssembleSlot) {
                extraAttr = " <assemble>";
            }

            if (slotList[slotIdx].rtOutcastIter == ITEM_POOL_INVALID_INDEX) {
                DEV_DEBUG("[DecideSlotAddress]   Slot [%3lu]: <no tensor>%s", slotIdx, extraAttr);
                continue;
            }
            [[maybe_unused]] auto& outcastDesc = workspace_->GetRuntimeOutcastTensor(slotList[slotIdx].rtOutcastIter);
            DEV_DEBUG("[DecideSlotAddress]   Slot [%3lu]: %s%s", slotIdx, outcastDesc.Dump().c_str(), extraAttr);
        }
    }
}

void DeviceStitchContext::DecideSlotAddress(DeviceExecuteSlot* slotList, size_t slotSize)
{
    [[maybe_unused]] static constexpr uint64_t NON_ADDR_MASK = UINT64_C(1) << 62;

    DumpSlotInfo("Update before", slotList, slotSize);
#if !DEBUG_INFINITE_LIFETIME
    for (size_t slotIdx = 0; slotIdx < slotSize; ++slotIdx) {
        auto& slot = slotList[slotIdx];
        if (slot.rtOutcastIter != ITEM_POOL_INVALID_INDEX &&
            workspace_->GetRuntimeOutcastTensor(slot.rtOutcastIter).property ==
                RuntimeTensorMemProperty::DEVTASK_INNER_OUTCAST) {
            workspace_->RuntimeOutcastTensorReplaceAddrWithoutRecycle(slot.rtOutcastIter,
                                                                      workspace_->AllocateBoundaryOutcastSlot(),
                                                                      RuntimeTensorMemProperty::BOUNDARY_OUTCAST);
        }
    }
#endif // !DEBUG_INFINITE_LIFETIME
    DumpSlotInfo("Update after", slotList, slotSize);
}

int DeviceStitchContext::DecideIncastOutcast(uint64_t taskId)
{
    (void)taskId;
    for (size_t funcIndex = 0; funcIndex < stitchedList_.size(); ++funcIndex) {
        auto& dup = stitchedList_[funcIndex];
        // decide incast address
        size_t incastSize = dup.GetSource()->GetIncastSize();
        for (size_t i = 0; i < incastSize; ++i) {
            auto& desc = dup.GetIncastAddress(i);
            DEV_ASSERT(CtrlErr::DEVICE_TASK_BUILD_FAILED, desc.IsRtOutcast());
            ItemPoolIter iter = desc.GetRtOutcastIter();
            uintdevptr_t addr = workspace_->GetRuntimeOutcastTensor(iter).allocation.ptr;
            workspace_->RuntimeOutcastTensorDeref(iter);
            desc = AddressDescriptor::MakeFromAddress(addr);
        }

        // decide outcast address
        size_t outcastSize = dup.GetSource()->GetOutcastSize();
        for (size_t i = 0; i < outcastSize; ++i) {
            auto& desc = dup.GetOutcastAddress(i);
            DEV_ASSERT(CtrlErr::DEVICE_TASK_BUILD_FAILED, desc.IsRtOutcast());
            ItemPoolIter iter = desc.GetRtOutcastIter();
            uintdevptr_t addr = workspace_->GetRuntimeOutcastTensor(iter).Addr();
            workspace_->RuntimeOutcastTensorDeref(iter);
            desc = AddressDescriptor::MakeFromAddress(addr);
        }
    }
    return DEVICE_MACHINE_OK;
}

int DeviceStitchContext::MoveTo(DynDeviceTask* dynTask)
{
    dynTask->stitchedList = std::move(stitchedList_);
    stitchedList_.clear();
    dynTask->devTask.coreFunctionCnt = stitchedCallOpSize_;
    stitchedCallOpSize_ = 0;

    if (dynTask->stitchedList.size() > MAX_STITCH_FUNC_NUM) {
        DEV_ERROR(ProgEncodeErr::STITCH_LIST_TOO_LARGE,
                  "#ctrl.stitch.toomany_root: Stitch list size:%u exceeds maximum allowed cached function number:%zu.",
                  dynTask->stitchedList.size(), MAX_STITCH_FUNC_NUM);
        return DEVICE_MACHINE_ERROR;
    }
    DEV_ASSERT(ProgEncodeErr::STITCH_LIST_TOO_LARGE, dynTask->stitchedList.size() <= MAX_STITCH_FUNC_NUM);
    int size = static_cast<int>(dynTask->stitchedList.size());
    for (int i = 0; i < size; ++i) {
        auto& funcDup = dynTask->stitchedList[i];
        dynTask->dynFuncDataCacheList[i] = {funcDup.GetSource(), &funcDup.GetOperationCurrPredCount(0),
                                            funcDup.GetSource()->GetCalleeIndexAddr(), funcDup.DupDataForDynFuncData()};
        dynTask->devTask.mixTaskData.opWrapList[i] = PtrToValue(funcDup.GetSource()->GetOpWrapListAddr());
    }
    dynTask->dynFuncDataCacheListSize = size;
    return DEVICE_MACHINE_OK;
}

static void ValidateAndDumpStitchEdge(const DevAscendFunctionDupped& producerDup,
                                      const DevAscendFunctionDupped& consumerDup, size_t producerOperationIdx,
                                      size_t consumerIdx, size_t consumerOperationIdx,
                                      DeviceStitchContext::StitchKind debugStitchKind, int debugSlotIdx)
{
    if (producerOperationIdx >= producerDup.GetSource()->GetOperationSize()) {
        DEV_ERROR(ProgEncodeErr::STITCH_HANDLE_INDEX_OUT_OF_RANGE,
                  "#ctrl.task.pre.stitch.handle: producerOperationIdx %zu exceeds the size of GetOperation %zu",
                  producerOperationIdx, producerDup.GetSource()->GetOperationSize());
    }
    if (consumerOperationIdx >= consumerDup.GetSource()->GetOperationSize()) {
        DEV_ERROR(ProgEncodeErr::STITCH_HANDLE_INDEX_OUT_OF_RANGE,
                  "#ctrl.task.pre.stitch.handle: consumerOperationIdx %zu exceeds the size of GetOperation %zu",
                  consumerOperationIdx, consumerDup.GetSource()->GetOperationSize());
    }
    DEV_ASSERT(ProgEncodeErr::STITCH_HANDLE_INDEX_OUT_OF_RANGE,
               producerOperationIdx < producerDup.GetSource()->GetOperationSize());
    DEV_ASSERT(ProgEncodeErr::STITCH_HANDLE_INDEX_OUT_OF_RANGE,
               consumerOperationIdx < consumerDup.GetSource()->GetOperationSize());
    DEV_VERBOSE_DEBUG("[Stitch] slot:%d kind:%s dupIdx:%d funcKey:%d,op:%d -> funcKey:%d,op:%d\n", debugSlotIdx,
                      DeviceStitchContext::GetStitchKindName(debugStitchKind).c_str(), (int)consumerIdx,
                      producerDup.GetSource()->GetFuncKey(), (int)producerOperationIdx,
                      consumerDup.GetSource()->GetFuncKey(), (int)consumerOperationIdx);

    topo_dump::DumpStitchEdge(producerDup, consumerDup, producerOperationIdx, consumerIdx, consumerOperationIdx,
                              debugStitchKind, debugSlotIdx);
}

void DeviceStitchContext::HandleOneStitch(DevAscendFunctionDupped& producerDup, DevAscendFunctionDupped& consumerDup,
                                          DevAscendFunctionDuppedStitchList& producerStitchList,
                                          uint32_t producerFuncIndex, size_t producerOperationIdx, size_t consumerIdx,
                                          size_t consumerOperationIdx, DeviceWorkspaceAllocator* workspace,
                                          StitchKind debugStitchKind, int debugSlotIdx, uint64_t devTaskId)
{
    (void)debugStitchKind;
    (void)debugSlotIdx;
    DEV_VERBOSE_DEBUG("DeviceStitchContext::HandleOneStitch %p stitchlist %p [%u!%d] -> [%u!%u]", &producerDup,
                      &producerStitchList, producerFuncIndex, static_cast<int>(producerOperationIdx),
                      static_cast<uint32_t>(consumerIdx), static_cast<uint32_t>(consumerOperationIdx));

    if (CheckStitchCacheDuplicate(workspace->StitchCacheAddr(), workspace->RootFuncMaxCallOpsize(), producerFuncIndex,
                                  static_cast<uint32_t>(producerOperationIdx), static_cast<uint32_t>(consumerIdx),
                                  consumerOperationIdx, devTaskId)) {
        DEV_VERBOSE_DEBUG("Duplicate stitch ignore.");
        return;
    }

    PushBackTask(producerStitchList, MakeTaskID(consumerIdx, consumerOperationIdx), workspace);
    consumerDup.GetOperationCurrPredCount(consumerOperationIdx)++;

    auto* producerFunc = producerDup.GetSource();
    auto producerIdx = static_cast<uint32_t>(producerOperationIdx);
    producerFunc->ClearTailTask(producerIdx);
    if (producerFunc->ClearDeadEndHub(producerIdx)) {
        producerFunc->PropagateDeadHubClear(producerIdx);
    }

    DEV_IF_NONDEVICE
    {
        ValidateAndDumpStitchEdge(producerDup, consumerDup, producerOperationIdx, consumerIdx, consumerOperationIdx,
                                  debugStitchKind, debugSlotIdx);
    }
}

void DeviceStitchContext::HandleOneStitch(DevAscendFunctionDupped& producerDup, DevAscendFunctionDupped& consumerDup,
                                          uint32_t producerFuncIndex, size_t producerOperationIdx, size_t consumerIdx,
                                          size_t consumerOperationIdx, DeviceWorkspaceAllocator* workspace,
                                          StitchKind debugStitchKind, int debugSlotIdx, uint64_t devTaskId)
{
    auto& producerStitchList = producerDup.GetOperationStitch(producerOperationIdx, false);
    HandleOneStitch(producerDup, consumerDup, producerStitchList, producerFuncIndex, producerOperationIdx, consumerIdx,
                    consumerOperationIdx, workspace, debugStitchKind, debugSlotIdx, devTaskId);
}

uint64_t DeviceStitchContext::PartialUpdateStitchConsumer(DevAscendFunctionDupped& nextDup, size_t devTaskId,
                                                          size_t devNextIdx, DeviceExecuteSlot& slot, int slotIdx,
                                                          DevAscendFunctionIncast& incast)
{
    uint64_t matchCount = 0;
    auto* nextSrc = nextDup.GetSource();
    auto expressionList = &nextDup.GetExpression(0);
    auto& cellMatchTableDesc = slot.partialUpdate->cellMatchTableDesc;
    auto partialUpdateTableData = &slot.partialUpdate->cellMatchRuntimePartialUpdateTable[0];
    size_t tableSize = slot.partialUpdate->cellMatchRuntimePartialUpdateTable.size();

    DEV_VERBOSE_DEBUG(
        "[PartialUpdateStitch] enter slotIdx=%d devTaskId=%lu devNextIdx=%lu consumerFuncKey=%d consumerCount=%zu "
        "tableSize=%zu descDimSize=%d",
        slotIdx, (uint64_t)devTaskId, (uint64_t)devNextIdx, nextSrc->GetFuncKey(), incast.consumerList.size(),
        tableSize, cellMatchTableDesc.GetDimensionSize());
    size_t cellMatchTagId = CellMatchBuildTagId(slot.slotAllocIterId, devTaskId);

    for (size_t n = 0; n < incast.consumerList.size(); n++) {
        auto& consumer = nextSrc->At(incast.consumerList, n);
        uint64_t consumerOffset[DEV_SHAPE_DIM_MAX];
        uint64_t consumerValidShape[DEV_SHAPE_DIM_MAX];
        GetTensorOffsetAndValidShape<false>(nextSrc, consumerOffset, consumerValidShape, expressionList,
                                            cellMatchTableDesc, incast.dim, consumer.operationIdx,
                                            consumer.offsetAttrIdx);

        DEV_IF_VERBOSE_DEBUG
        {
            for (int j = 0; j < cellMatchTableDesc.GetDimensionSize(); j++) {
                DEV_VERBOSE_DEBUG(
                    "PartialUpdateStitchConsumer consumer cell match, operation[%d] -> dimension[%d] = (offset:%lu "
                    ",shape:%lu, "
                    "cellshape:%d)",
                    consumer.operationIdx, j, consumerOffset[j], consumerValidShape[j],
                    cellMatchTableDesc.cellShape.dim[j]);
            }
        }
        topo_dump::DumpConsumerCellAccess(static_cast<uint32_t>(devTaskId), slotIdx, static_cast<uint32_t>(devNextIdx),
                                          *nextSrc, consumer, cellMatchTableDesc, expressionList);

        int consumerOpIdx = consumer.operationIdx;
        if (consumer.wrapTaskHubOpIdx != INVALID_WRAP_TASK_HUB_OP_IDX) {
            DEV_VERBOSE_DEBUG(
                "[PartialUpdateStitch] devTaskId=%lu devNextIdx=%lu, replace consumerOpIdx[%d] witch wrapHubOpIdx[%d]",
                (uint64_t)devTaskId, (uint64_t)devNextIdx, consumerOpIdx, consumer.wrapTaskHubOpIdx);

            consumerOpIdx = consumer.wrapTaskHubOpIdx;
        }
        CellMatchStitchEnhance(consumerOffset, consumerValidShape, cellMatchTableDesc,
                               static_cast<uint32_t>(consumer.opType), partialUpdateTableData, stitchedList_.data(),
                               stitchedList_.size(), &nextDup, cellMatchTagId, devNextIdx, workspace_, consumerOpIdx,
                               slotIdx, &matchCount);
    }

    return matchCount;
}

uint64_t DeviceStitchContext::FullCoverDefaultUpdateStitch(DevAscendFunctionDupped& nextDup, size_t devTaskId,
                                                           size_t devNextIdx, DeviceExecuteSlot& slot, int slotIdx,
                                                           DevAscendFunctionIncast& incast)
{
    uint64_t matchCount = 0;
    DevAscendFunctionDupped& prevDup = stitchedList_[slot.stitchDupIdx];
    auto* prevSrc = prevDup.GetSource();
    auto& outcast = prevSrc->GetOutcast(slot.stitchOutcastIdx);
    auto* nextSrc = nextDup.GetSource();
    auto expressionList = &nextDup.GetExpression(0);
    auto& cellMatchTableDesc = outcast.cellMatchTableDesc;
    size_t tableSize = outcast.cellMatchRuntimeFullUpdateTable.size();
    if (tableSize == 0) {
        return 0;
    }
    auto fullUpdateTableData = &prevSrc->At(outcast.cellMatchRuntimeFullUpdateTable, 0);

    DEV_VERBOSE_DEBUG(
        "[FullCoverDefaultStitch] enter slotIdx=%d devTaskId=%lu devNextIdx=%lu producerFuncKey=%d "
        "consumerFuncKey=%d stitchDupIdx=%u stitchOutcastIdx=%u consumerCount=%zu tableSize=%zu descDimSize=%d",
        slotIdx, (uint64_t)devTaskId, (uint64_t)devNextIdx, prevSrc->GetFuncKey(), nextSrc->GetFuncKey(),
        slot.stitchDupIdx, slot.stitchOutcastIdx, incast.consumerList.size(), tableSize,
        cellMatchTableDesc.GetDimensionSize());

    for (size_t n = 0; n < incast.consumerList.size(); n++) {
        auto& consumer = nextSrc->At(incast.consumerList, n);
        uint64_t fullCoverOffset[DEV_SHAPE_DIM_MAX];
        uint64_t fullCoverValidShape[DEV_SHAPE_DIM_MAX];
        GetTensorOffsetAndValidShape<false>(nextSrc, fullCoverOffset, fullCoverValidShape, expressionList,
                                            cellMatchTableDesc, incast.dim, consumer.operationIdx,
                                            consumer.offsetAttrIdx);
        topo_dump::DumpConsumerCellAccess(static_cast<uint32_t>(devTaskId), slotIdx, static_cast<uint32_t>(devNextIdx),
                                          *nextSrc, consumer, cellMatchTableDesc, expressionList);
        int consumerOpIdx = consumer.operationIdx;
        if (consumer.wrapTaskHubOpIdx != INVALID_WRAP_TASK_HUB_OP_IDX) {
            DEV_VERBOSE_DEBUG("[FullCoverDefaultStitch] devTaskId=%lu devNextIdx=%lu, replace consumerOpIdx[%d] witch "
                              "wrapHubOpIdx[%d]",
                              (uint64_t)devTaskId, (uint64_t)devNextIdx, consumerOpIdx, consumer.wrapTaskHubOpIdx);

            consumerOpIdx = consumer.wrapTaskHubOpIdx;
        }
        CellMatchHandle<HandleCellMatchFull>(fullCoverOffset, fullCoverValidShape, cellMatchTableDesc,
                                             fullUpdateTableData, &matchCount, &prevDup, &nextDup, devNextIdx,
                                             consumerOpIdx, workspace_, slotIdx, devTaskId, slot.stitchDupIdx);
        DeviceStitchContext::CheckStitch(stitchedList_.data(), stitchedList_.size(), &nextDup);
    }
    return matchCount;
}

uint64_t DeviceStitchContext::FullCoverUpdateStitch(DevAscendFunctionDupped& nextDup, size_t devTaskId,
                                                    size_t devNextIdx, DeviceExecuteSlot& slot, int slotIdx,
                                                    DevAscendFunctionIncast& incast)
{
    DevAscendFunctionDupped& prevDup = stitchedList_[slot.stitchDupIdx];
    auto* prevSrc = prevDup.GetSource();
    auto& outcast = prevSrc->GetOutcast(slot.stitchOutcastIdx);
    auto* nextSrc = nextDup.GetSource();
    DEV_VERBOSE_DEBUG("outcast %lu fullcover update stitch\n", (unsigned long)slot.stitchOutcastIdx);
    DEV_VERBOSE_DEBUG("=================FullCoverUpdateStitch %zu %zu===========================\n",
                      outcast.producerConsumerList.size(), incast.consumerList.size());

    // stitchPolicyFullCover hub
    auto producerHubOpIdx = outcast.stitchPolicyFullCoverProducerHubOpIdx;
    if (producerHubOpIdx != -1) {
        auto consumerAllOpIdxList = &nextSrc->At(incast.stitchPolicyFullCoverConsumerAllOpIdxList, 0);
        for (size_t conIndex = 0, conSize = incast.stitchPolicyFullCoverConsumerAllOpIdxList.size(); conIndex < conSize;
             conIndex++) {
            auto& consumerOpIdx = consumerAllOpIdxList[conIndex];
            DEV_VERBOSE_DEBUG("FullCoverUpdateStitch hub handle one stitch [%u!%u] -> [%u!%u]", slot.stitchDupIdx,
                              static_cast<uint32_t>(producerHubOpIdx), static_cast<uint32_t>(devNextIdx),
                              consumerOpIdx);
            DeviceStitchContext::HandleOneStitch(prevDup, nextDup, slot.stitchDupIdx, producerHubOpIdx, devNextIdx,
                                                 consumerOpIdx, workspace_, StitchKind::StitchFullCover, slotIdx,
                                                 static_cast<uint64_t>(devTaskId));
        }
        DeviceStitchContext::CheckStitch(stitchedList_.data(), stitchedList_.size(), &nextDup);
    } else {
        // stitchPolicyFullCover producer
        auto producerList = &prevSrc->At(outcast.stitchPolicyFullCoverProducerList, 0);
        auto consumerAllOpIdxList = &nextSrc->At(incast.stitchPolicyFullCoverConsumerAllOpIdxList, 0);
        for (size_t prodIndex = 0, prodSize = outcast.stitchPolicyFullCoverProducerList.size(); prodIndex < prodSize;
             prodIndex++) {
            auto& producer = producerList[prodIndex];
            auto producerOperationIdx = producer.operationIdx;
            if (producer.opType == CellMatchOpType::READ) {
                continue;
            }
            for (size_t conIndex = 0, conSize = incast.stitchPolicyFullCoverConsumerAllOpIdxList.size();
                 conIndex < conSize; conIndex++) {
                auto& consumerOpIdx = consumerAllOpIdxList[conIndex];
                DEV_VERBOSE_DEBUG("FullCoverUpdateStitch handle one stitch [%u!%u] -> [%u!%u]", slot.stitchDupIdx,
                                  static_cast<uint32_t>(producerOperationIdx), static_cast<uint32_t>(devNextIdx),
                                  consumerOpIdx);
                DeviceStitchContext::HandleOneStitch(prevDup, nextDup, slot.stitchDupIdx, producerOperationIdx,
                                                     devNextIdx, consumerOpIdx, workspace_, StitchKind::StitchFullCover,
                                                     slotIdx, static_cast<uint64_t>(devTaskId));
            }
        }
        DeviceStitchContext::CheckStitch(stitchedList_.data(), stitchedList_.size(), &nextDup);
    }

    return FullCoverDefaultUpdateStitch(nextDup, devTaskId, devNextIdx, slot, slotIdx, incast);
}

uint64_t DeviceStitchContext::PartialUpdateStitchProducer(DevAscendFunctionDupped& nextDup, size_t devTaskId,
                                                          size_t devNextIdx, DeviceExecuteSlot& slot, int slotIdx,
                                                          DevAscendFunctionOutcast& outcast)
{
    uint64_t matchCount = 0;
    auto* nextSrc = nextDup.GetSource();
    auto expressionList = &nextDup.GetExpression(0);
    auto& cellMatchTableDesc = slot.partialUpdate->cellMatchTableDesc;
    if (slot.partialUpdate->Empty()) {
        return matchCount;
    }
    auto partialUpdateTableData = &slot.partialUpdate->cellMatchRuntimePartialUpdateTable[0];
    size_t cellMatchTagId = CellMatchBuildTagId(slot.slotAllocIterId, devTaskId);

    auto processProducerList = [&](auto& producerListRef) {
        auto* producerList = &nextSrc->At(producerListRef, 0);
        for (size_t i = 0; i < producerListRef.size(); i++) {
            auto& producer = producerList[i];
            if (producer.opType == CellMatchOpType::READ) {
                continue;
            }
            uint64_t producerOffset[DEV_SHAPE_DIM_MAX];
            uint64_t producerValidShape[DEV_SHAPE_DIM_MAX];
            GetTensorOffsetAndValidShape<false>(nextSrc, producerOffset, producerValidShape, expressionList,
                                                cellMatchTableDesc, outcast.dim, producer.operationIdx,
                                                producer.offsetAttrIdx);

            DEV_IF_VERBOSE_DEBUG
            {
                for (int k = 0; k < cellMatchTableDesc.GetDimensionSize(); k++) {
                    DEV_VERBOSE_DEBUG(
                        "PartialUpdateStitchProducer cell match, operation[%d] -> dimension[%d] = (offset:%lu "
                        ",validshape:%lu, "
                        "cellshape:%d)",
                        producer.operationIdx, k, producerOffset[k], producerValidShape[k],
                        cellMatchTableDesc.cellShape.dim[k]);
                }
            }

            CellMatchStitchEnhance(producerOffset, producerValidShape, cellMatchTableDesc,
                                   static_cast<uint32_t>(producer.opType), partialUpdateTableData, stitchedList_.data(),
                                   stitchedList_.size(), &nextDup, cellMatchTagId, devNextIdx, workspace_,
                                   producer.operationIdx, slotIdx, &matchCount);
        }
    };

    DEV_VERBOSE_DEBUG("Begin PartialUpdateStitchProducer producer list.");
    processProducerList(outcast.producerConsumerList);

    DEV_VERBOSE_DEBUG("Begin PartialUpdateStitchProducer stitchPolicyFullCoverProducerList list.");
    processProducerList(outcast.stitchPolicyFullCoverProducerList);

    return matchCount;
}

void DeviceStitchContext::ReuseStitch(DevAscendFunctionDupped& nextDup, size_t devNextIdx, size_t devTaskId)
{
    if (nextDup.GetSource()->rootInnerTensorWsMemoryRequirement == 0) {
        // 0 length workspace, no dependency in need
        return;
    }

    uintdevptr_t nextAddrL = nextDup.RuntimeWorkspace();
    uintdevptr_t nextAddrR = nextAddrL + nextDup.GetSource()->rootInnerTensorWsMemoryRequirement;
    auto nextReuseInfo = nextDup.GetRuntimeReuseInfo();
    if (auto& firstDup = stitchedList_[stitchReuseContext_.firstDupIdx];
        firstDup.GetRuntimeReuseInfo().poolResetTimes >= nextReuseInfo.poolResetTimes) {
        return;
    }

    auto needsDependency = [&](uint32_t prevIdx) -> int {
        if (prevIdx >= devNextIdx) {
            // invalid idx
            return INVALID_TOO_AHEAD;
        }

        auto& prevDup = stitchedList_[prevIdx];
        if (prevDup.GetSource()->rootInnerTensorWsMemoryRequirement == 0) {
            // empty workspace
            return SKIP_EMPTY;
        }

        auto prevReuseInfo = prevDup.GetRuntimeReuseInfo();
        if (prevReuseInfo.poolResetTimes + 1 != nextReuseInfo.poolResetTimes) {
            return prevReuseInfo.poolResetTimes >= nextReuseInfo.poolResetTimes ? INVALID_TOO_AHEAD : NO_DEP;
        }

        // proper poolResetTimes
        stitchReuseContext_.lastNonEmptyDupIdx = prevIdx;

        uintdevptr_t prevAddrL = prevDup.RuntimeWorkspace();
        uintdevptr_t prevAddrR = prevAddrL + prevDup.GetSource()->rootInnerTensorWsMemoryRequirement;
        return !(prevAddrR <= nextAddrL || prevAddrL >= nextAddrR) ? NEEDS_DEP : NO_DEP;
    };

    auto skipBefore = [](int result) { return result == NO_DEP || result == SKIP_EMPTY; };
    for (; skipBefore(needsDependency(stitchReuseContext_.firstDupIdx)); stitchReuseContext_.firstDupIdx++) {
    }

    if (needsDependency(stitchReuseContext_.firstDupIdx) == NEEDS_DEP) {
        for (uint32_t prevIdx = stitchReuseContext_.firstDupIdx;; prevIdx++) {
            int res = needsDependency(prevIdx);
            if (res == NO_DEP || res == INVALID_TOO_AHEAD) {
                break;
            }
            if (res != SKIP_EMPTY) {
                auto& prevDup = stitchedList_[prevIdx];
                StitchForWorkspaceReuse(stitchedList_.data(), stitchedList_.size(), prevDup, nextDup, devNextIdx,
                                        workspace_, static_cast<uint64_t>(devTaskId), prevIdx);
                stitchReuseContext_
                    .firstDupIdx = prevIdx; // Risk on time complexity: Duplicated access to empty-workspace funcs
            }
        }
    } else {
        if (stitchReuseContext_.lastNonEmptyDupIdx != -1) {
            auto& prevDup = stitchedList_[stitchReuseContext_.lastNonEmptyDupIdx];
            StitchForWorkspaceReuse(stitchedList_.data(), stitchedList_.size(), prevDup, nextDup, devNextIdx,
                                    workspace_, static_cast<uint64_t>(devTaskId),
                                    stitchReuseContext_.lastNonEmptyDupIdx);
        }
    }
}

uint64_t DeviceStitchContext::FastStitchConsumer(DeviceExecuteSlot* slotList, size_t slotSize,
                                                 DevAscendFunctionDupped& nextDup, size_t devTaskId, size_t devNextIdx)
{
    auto* nextSrc = nextDup.GetSource();
    uint64_t matchCount = 0;
    for (size_t incastIdx = 0; incastIdx < nextSrc->GetIncastSize(); ++incastIdx) {
        auto& incast = nextSrc->GetIncast(incastIdx);
        for (size_t j = 0; j < incast.fromSlotList.size(); ++j) {
            auto slotIdx = nextSrc->At(incast.fromSlotList, j);
            if (slotIdx >= (int)slotSize) {
                DEV_ERROR(ProgEncodeErr::STITCH_HANDLE_INDEX_OUT_OF_RANGE,
                          "#ctrl.stitch.invalid_slot: slotIdx %d is larger than slotSize %zu!.", slotIdx, slotSize);
                continue;
            }
            auto& slot = slotList[slotIdx];
            DEV_VERBOSE_DEBUG("FastStitch slot %d, incastindex %zu, ispartial %d, stitchDupIdx %u", slotIdx, incastIdx,
                              slot.isPartialUpdateStitch, slot.stitchDupIdx);
            if (slot.stitchDupIdx == INVALID_STITCH_IDX) {
                continue;
            }
            if (slot.isPartialUpdateStitch) {
                matchCount = PartialUpdateStitchConsumer(nextDup, devTaskId, devNextIdx, slot, slotIdx, incast);
                continue;
            }
            if (slot.rtOutcastIter == ITEM_POOL_INVALID_INDEX) {
                continue;
            }
            matchCount = FullCoverUpdateStitch(nextDup, devTaskId, devNextIdx, slot, slotIdx, incast);
        }
    }
    return matchCount;
}

uint64_t DeviceStitchContext::FastStitchProducer(DeviceExecuteSlot* slotList, size_t slotSize,
                                                 DevAscendFunctionDupped& nextDup, size_t devTaskId, size_t devNextIdx)
{
    auto* nextSrc = nextDup.GetSource();
    uint64_t matchCount = 0;
    for (size_t outcastIdx = 0; outcastIdx < nextSrc->GetOutcastSize(); ++outcastIdx) {
        auto& outcast = nextSrc->GetOutcast(outcastIdx);
        for (size_t j = 0; j < outcast.toSlotList.size(); ++j) {
            auto slotIdx = nextSrc->At(outcast.toSlotList, j);
            if (slotIdx >= (int)slotSize) {
                DEV_ERROR(ProgEncodeErr::STITCH_HANDLE_INDEX_OUT_OF_RANGE,
                          "#ctrl.stitch.invalid_slot: slotIdx %d is larger than slotSize %zu!.", slotIdx, slotSize);
                continue;
            }
            auto& slot = slotList[slotIdx];
            if (slot.stitchDupIdx == INVALID_STITCH_IDX || !slot.isPartialUpdateStitch) {
                continue;
            }
            DEV_VERBOSE_DEBUG("FastStitch slot %d, outcastindex %zu, ispartial %d, stitchDupIdx %u", slotIdx,
                              outcastIdx, slot.isPartialUpdateStitch, slot.stitchDupIdx);
            matchCount += PartialUpdateStitchProducer(nextDup, devTaskId, devNextIdx, slot, slotIdx, outcast);
        }
    }
    return matchCount;
}

uint64_t DeviceStitchContext::FastStitch(DeviceExecuteSlot* slotList, size_t slotSize, DevAscendFunctionDupped& nextDup,
                                         size_t devTaskId, size_t devNextIdx)
{
    AutoScopedPerf asp(PERF_EVT_FAST_STITCH);
#if !ENABLE_STITCH
    return 0;
#endif
    nextDup.GetSource()->GetFuncidx() = static_cast<int>(devNextIdx);
    if (devNextIdx == 0) {
        return 0;
    }
    uint64_t matchCount = FastStitchConsumer(slotList, slotSize, nextDup, devTaskId, devNextIdx);
    matchCount += FastStitchProducer(slotList, slotSize, nextDup, devTaskId, devNextIdx);
#if !DEBUG_INFINITE_LIFETIME
    ReuseStitch(nextDup, devNextIdx, devTaskId);
#endif
    return matchCount;
}

void DeviceStitchContext::DumpStitchInfo(DevAscendFunctionDupped* stitchedList, int stitchedSize)
{
    int funcId = 0;
    for (int i = 0; i < stitchedSize; i++) {
        auto& funcDup = stitchedList[i];
        for (size_t opIndex = 0; opIndex < funcDup.GetSource()->GetOperationSize(); opIndex++) {
            auto& stitch = funcDup.GetOperationStitch(opIndex);

            std::stringstream oss;
            oss << stitch.Dump();
            DEV_VERBOSE_DEBUG("func %d opIndex %zu stitch list: %p stitchindex:%u %s.", funcId, opIndex, &stitch,
                              funcDup.GetSource()->GetOperationStitchIndex(opIndex), oss.str().c_str());
        }
        funcId++;
    }
}

void DeviceStitchContext::StitchForWorkspaceReuse(DevAscendFunctionDupped* stitchingList, int stitchingSize,
                                                  DevAscendFunctionDupped& prevDup, DevAscendFunctionDupped& currDup,
                                                  size_t devCurrIdx, DeviceWorkspaceAllocator* workspace,
                                                  uint64_t devTaskId, uint32_t preFuncIndex)
{
    // Add dependency between root functions
    auto* prevSrc = prevDup.GetSource();
    auto* currSrc = currDup.GetSource();

    size_t prevNoSuccOpSize = prevSrc->GetNoSuccOpSize();
    size_t currNoPredOpSize = currSrc->GetNoPredOpSize();
    if (unlikely(prevNoSuccOpSize == 0 || currNoPredOpSize == 0)) {
        // Empty root function
        return;
    }

    // Graph has been optimized when encoding, we just put trivial full connection logics here
    for (size_t i = 0; i < prevNoSuccOpSize; ++i) {
        int prevNoSucc = prevSrc->GetNoSuccOpIdx(i);
        auto& stitch = prevDup.GetOperationStitch(prevNoSucc);
        for (size_t j = 0; j < currNoPredOpSize; ++j) {
            int currNoPred = currSrc->GetNoPredOpIdx(j);
            DEV_TRACE_DEBUG(DEvent(DUid(none()), DActStitchEdge(Producer(LUid(none(), 0, none(), prevNoSucc, none()),
                                                                         none(), none(), none(), none(), none()),
                                                                Consumer(LUid(none(), 0, none(), currNoPred, none()),
                                                                         none(), none(), none(), none(), none()),
                                                                StitchReasonWorkspaceReuse())));
            DEV_VERBOSE_DEBUG("StitchForWorkspaceReuse handle one stitch [%u] -> [%u!%u]",
                              static_cast<uint32_t>(prevNoSucc), static_cast<uint32_t>(devCurrIdx),
                              static_cast<uint32_t>(currNoPred));
            DeviceStitchContext::HandleOneStitch(prevDup, currDup, stitch, preFuncIndex, prevNoSucc, devCurrIdx,
                                                 currNoPred, workspace, DeviceStitchContext::StitchKind::StitchReuse,
                                                 -1, devTaskId);
            DeviceStitchContext::CheckStitch(stitchingList, stitchingSize, &currDup);
        }
    }
}
} // namespace npu::tile_fwk::dynamic
