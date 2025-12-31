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
 * \file device_slot_context.cpp
 * \brief
 */

#include "machine/device/dynamic/context/device_slot_context.h"

namespace npu::tile_fwk::dynamic {

void DeviceSlotContext::InitAllocator(DeviceWorkspaceAllocator &workspace, uint64_t slotSize) {
    workspace.SetupVector(slotList_);
    workspace.SetupItemPool(slotRefCntPool_, slotSize);
    workspace_ = &workspace;
    slotList_.resize(slotSize);
}

void DeviceSlotContext::FillInputOutputSlot(DevAscendProgram *devProg, DevStartArgs *args) {
    FillInputOutputSlot(slotList_.data(), slotList_.size(), devProg, args);
}

static void UpdateSlotsForStitch(int slotIdx, DeviceExecuteSlot &slot, DevAscendFunction *devRootSrc,
                                 DevAscendFunctionOutcast &outcast, uint32_t devTaskId, uint32_t devNextIdx,
                                 uint32_t outcastIndex, uint64_t *expressionList) {
    slot.stitchDupIdx = devNextIdx;
    slot.stitchOutcastIdx = outcastIndex;
    UNUSED(slotIdx);

    auto producerList = &devRootSrc->At(outcast.producerList, 0);
    if (slot.isPartialUpdateStitch) {
        auto &cellMatchTableDesc = slot.partialUpdate->cellMatchTableDesc;
        auto tableData = &slot.partialUpdate->cellMatchRuntimePartialUpdateTable[0];
        auto producerSize = outcast.producerList.size();
        if (producerSize != 0) {
            CellMatchFillIncastOutcast<false>(
                    devRootSrc, producerList, producerSize, expressionList, false, cellMatchTableDesc, tableData, devTaskId, devNextIdx);
        } else {
            // maybe is fullcover producer, dassemble full shape
            CellMatchFillIncastOutcast<false>(
                    devRootSrc, &devRootSrc->At(outcast.stitchPolicyFullCoverProducerList, 0), outcast.stitchPolicyFullCoverProducerList.size(),
                    expressionList, false, cellMatchTableDesc, tableData, devTaskId, devNextIdx);
        }

        DEV_VERBOSE_DEBUG("[UpdateSlots]  slot %d CellMatchPartial=%s\n", slotIdx,
            DevAscendFunctionDuppedStitchList::DumpTask<uint64_t>(tableData, slot.partialUpdate->cellMatchRuntimePartialUpdateTable.size()).c_str());
        slot.isPartialUpdateDirty = true;
    } else {
        auto &cellMatchTableDesc = outcast.cellMatchTableDesc;
        auto tableData = &devRootSrc->At(outcast.cellMatchRuntimeFullUpdateTable, 0);
        CellMatchFillIncastOutcast<false>(
                devRootSrc, producerList, outcast.producerList.size(), expressionList, false, cellMatchTableDesc, tableData);
        DEV_VERBOSE_DEBUG("[UpdateSlots] slot %d  CellMatchFull=%s\n", slotIdx,
            DevAscendFunctionDuppedStitchList::DumpTask(tableData, outcast.cellMatchRuntimeFullUpdateTable.size()).c_str());
    }
}

template <WsMemCategory category>
static int UpdateSlotsImpl(DeviceWorkspaceAllocator *workspace, DeviceExecuteSlot *slotList,
    const StitchedList &stitchedList, ItemPool<uint32_t, category> &slotRefCntPool,
    DevAscendFunctionDupped &devRootDup, uint32_t devTaskId, uint32_t devNextIdx) {
    AutoScopedPerf asp(PERF_EVT_UPDATE_SLOT);
    DevAscendFunction *devRootSrc = devRootDup.GetSource();
    size_t outcastSize = devRootSrc->GetOutcastSize();

    std::vector<int64_t> newRefCntIndex;
    newRefCntIndex.resize(outcastSize, itemPoolInvalidIndex);

    // Increase refCnt for linked incasts
    for (size_t index = 0; index < outcastSize; ++index) {
        auto &outcast = devRootSrc->GetOutcast(index);
        auto *rawTensor = devRootSrc->GetOutcastRawTensor(index);
        if (rawTensor->linkedIncastId != -1) {
            auto &incast = devRootSrc->GetIncast(rawTensor->linkedIncastId);
            if (incast.fromSlotList.size() == 0) {
                DEV_ERROR("Incast fromSlotList is empty");
                return DEVICE_MACHINE_ERROR;
            }
            DEV_DEBUG_ASSERT(incast.fromSlotList.size() > 0);
            int slotIndex = devRootSrc->At(incast.fromSlotList, 0);
            auto &slot = slotList[slotIndex];
            int64_t refCntIndex = slot.refCntIndex;
            if (refCntIndex != itemPoolInvalidIndex) {
                slot.RefCntInc(slotRefCntPool, outcast.toSlotList.size());
            }
            newRefCntIndex[index] = refCntIndex;
        }
    }

    // Update slot address
    uint64_t *expressionList = &devRootDup.GetExpression(0);
    for (size_t index = 0; index < outcastSize; ++index) {
        auto &srcDesc = devRootDup.GetOutcastAddress(index);
        auto &outcast = devRootSrc->GetOutcast(index);
        for (size_t i = 0; i < outcast.toSlotList.size(); ++i) {
            int slotIdx = devRootSrc->At(outcast.toSlotList, i);
            auto &slot = slotList[slotIdx];
            UpdateSlotsForStitch(slotIdx, slot, devRootSrc, outcast, devTaskId, devNextIdx, index, expressionList);
            if (!slot.RefCntIsNull() && slot.RefCntDec(slotRefCntPool)) {
                // At this moment only old addresses have available refCnt
                if (slot.desc.IsNullAddress()) {
                    DEV_ERROR("Slot descriptor is null");
                    return DEVICE_MACHINE_ERROR;
                }
                DEV_DEBUG_ASSERT(!slot.desc.IsNullAddress());
                uintdevptr_t freeAddr = slot.desc.IsAddress() ? slot.desc.addr :
                    stitchedList[slot.desc.dupIdx].GetOutcastAddress(slot.desc.outcastIdx).GetAddress();
                workspace->DelayedRecycleSlotMem(freeAddr);
            }

            if (!srcDesc.IsAddress() /* Unroll secondary placeholder */) {
                slot.desc = srcDesc;
            } else {
                slot.desc = AddressDescriptor(devNextIdx, index);
            }
            slot.refCntIndex = newRefCntIndex[index];
            DEV_VERBOSE_DEBUG("[UpdateSlots]  Outcast [%3zu] to slot [%3d], address %s.", index, slotIdx, slot.desc.Dump().c_str());
        }
    }
    return DEVICE_MACHINE_OK;
}

int DeviceSlotContext::UpdateSlots(DevAscendFunctionDupped &devRootDup, const StitchedList &stitchedList,
                                    uint32_t devTaskId, uint32_t devNextIdx) {
    int ret = DEVICE_MACHINE_OK;
    ret = UpdateSlotsImpl(workspace_, slotList_.data(), stitchedList, slotRefCntPool_,
        devRootDup, devTaskId, devNextIdx);
    if (unlikely(ret != DEVICE_MACHINE_OK)) {
        return DEVICE_MACHINE_ERROR;
    }
    return ret;
}

void DeviceSlotContext::FillInputOutputSlot(DeviceExecuteSlot *slotList, size_t slotSize, DevAscendProgram *devProg,
    DevStartArgs *args) {
    DEV_TRACE_DEBUG(CtrlEvent(none(), InputTensorCount(args->GetInputTensorSize())));
    for (int index = 0; index < args->GetInputTensorSize(); ++index) {
        DevTensorData &param = args->GetInputTensor(index);
        int slotIndex = devProg->startArgsInputTensorSlotIndexList[index];
        slotList[slotIndex].desc = AddressDescriptor(param.address);
        // input/output flatten
        slotList[slotIndex].isOutputSlot = true;
        DEV_INFO("Param %d Input Slot %d = %lx.", index, slotIndex, param.address);
        DEV_TRACE_DEBUG(CtrlEvent(none(), InputTensorElement(index, param.address, param.shape.GetSize())));
    }
    DEV_TRACE_DEBUG(CtrlEvent(none(), OutputTensorCount(args->GetOutputTensorSize())));
    for (int index = 0; index < args->GetOutputTensorSize(); ++index) {
        DevTensorData &param = args->GetOutputTensor(index);
        int slotIndex = devProg->startArgsOutputTensorSlotIndexList[index];
        slotList[slotIndex].desc = AddressDescriptor(param.address);
        slotList[slotIndex].isOutputSlot = true;
        DEV_INFO("Param %d Output Slot %d = %lx.", index, slotIndex, param.address);
        DEV_TRACE_DEBUG(CtrlEvent(none(), OutputTensorElement(index, param.address, param.shape.GetSize())));
    }
    for (size_t index = static_cast<size_t>(args->GetOutputTensorSize()); index < devProg->startArgsOutputTensorSlotIndexList.size(); ++index) {
        int outSlot = devProg->startArgsOutputTensorSlotIndexList[index];
        int inSlot = devProg->outputInplaceSlotList[index];
        if (inSlot != -1) {
            slotList[outSlot].desc = slotList[inSlot].desc;
            slotList[outSlot].isOutputSlot = true;
            DEV_VERBOSE_DEBUG("Param %zu Output Slot %d = inSlot %d.", index, outSlot, inSlot);
        }
    }
    for (size_t index = 0; index < devProg->assembleSlotIndexList.size(); ++index) {
        int slotIndex = devProg->assembleSlotIndexList[index];
        slotList[slotIndex].isAssembleSlot = true;
        DEV_VERBOSE_DEBUG("Assemble Slot %d .", slotIndex);
    }
    for (size_t index = 0, ie = devProg->partialUpdateList.size(); index < ie; index++) {
        auto &partialUpdate = devProg->At(devProg->partialUpdateList, index);
        int slotIndex = index;
        if (!partialUpdate.Empty()) {
            slotList[slotIndex].isPartialUpdateStitch = true;
            slotList[slotIndex].partialUpdate = &partialUpdate;
            DEV_VERBOSE_DEBUG("Partial Update Slot %d.\n", slotIndex);
        }
    }
    (void)slotSize;
}

}