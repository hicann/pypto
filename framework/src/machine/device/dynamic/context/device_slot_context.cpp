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

static void UpdateSlotsImpl(DeviceWorkspaceAllocator *workspace, DeviceExecuteSlot *slotList,
                            DevAscendFunctionDupped &devRootDup, uint32_t devTaskId, uint32_t devNextIdx) {
    AutoScopedPerf asp(PERF_EVT_UPDATE_SLOT);
    DevAscendFunction *devRootSrc = devRootDup.GetSource();
    size_t outcastSize = devRootSrc->GetOutcastSize();

    // Update slot address
    uint64_t *expressionList = &devRootDup.GetExpression(0);
    for (size_t i = 0; i < outcastSize; ++i) {
        const auto &outcastDesc = devRootDup.GetOutcastAddress(i);
        auto &outcast = devRootSrc->GetOutcast(i);
        for (size_t j = 0; j < outcast.toSlotList.size(); ++j) {
            int slotIdx = devRootSrc->At(outcast.toSlotList, j);
            auto &slot = slotList[slotIdx];
            UpdateSlotsForStitch(slotIdx, slot, devRootSrc, outcast, devTaskId, devNextIdx, i, expressionList);
            workspace->RuntimeOutcastTensorAssign(slot.rtOutcastIter, outcastDesc.GetRtOutcastIter());
            DEV_VERBOSE_DEBUG("[UpdateSlots]   Outcast [%3zu] to slot [%3d], address %s.", i, slotIdx, outcastDesc.Dump().c_str());
        }
    }
}

void DeviceSlotContext::UpdateSlots(DevAscendFunctionDupped &devRootDup, uint32_t devTaskId, uint32_t devNextIdx) {
    UpdateSlotsImpl(workspace_, slotList_.data(), devRootDup, devTaskId, devNextIdx);
}

void DeviceSlotContext::FillInputOutputSlot(DeviceExecuteSlot *slotList, [[maybe_unused]] size_t slotSize,
                                            DevAscendProgram *devProg, DevStartArgs *args) {
    DEV_TRACE_DEBUG(CtrlEvent(none(), InputTensorCount(args->GetInputTensorSize())));
    for (int index = 0; index < args->GetInputTensorSize(); ++index) {
        DevTensorData &param = args->GetInputTensor(index);
        int slotIndex = devProg->startArgsInputTensorSlotIndexList[index];
        DEV_ASSERT_MSG(slotIndex >= 0 && slotIndex < static_cast<int>(slotSize), "Invalid slot index %d", slotIndex);
        slotList[slotIndex].rtOutcastIter = workspace_->MakeRuntimeOutcastTensor(param.address, RuntimeTensorMemProperty::EXTERNAL);
        // input/output flatten
        slotList[slotIndex].isOutputSlot = true;
        DEV_INFO("Param %d Input Slot %d = %lx.", index, slotIndex, param.address);
        DEV_TRACE_DEBUG(CtrlEvent(none(), InputTensorElement(index, param.address, param.shape.GetSize())));
    }
    DEV_TRACE_DEBUG(CtrlEvent(none(), OutputTensorCount(args->GetOutputTensorSize())));
    for (int index = 0; index < args->GetOutputTensorSize(); ++index) {
        DevTensorData &param = args->GetOutputTensor(index);
        int slotIndex = devProg->startArgsOutputTensorSlotIndexList[index];
        DEV_ASSERT_MSG(slotIndex >= 0 && slotIndex < static_cast<int>(slotSize), "Invalid slot index %d", slotIndex);
        slotList[slotIndex].rtOutcastIter = workspace_->MakeRuntimeOutcastTensor(param.address, RuntimeTensorMemProperty::EXTERNAL);
        slotList[slotIndex].isOutputSlot = true;
        DEV_INFO("Param %d Output Slot %d = %lx.", index, slotIndex, param.address);
        DEV_TRACE_DEBUG(CtrlEvent(none(), OutputTensorElement(index, param.address, param.shape.GetSize())));
    }
    for (size_t index = static_cast<size_t>(args->GetOutputTensorSize()); index < devProg->startArgsOutputTensorSlotIndexList.size(); ++index) {
        int outSlot = devProg->startArgsOutputTensorSlotIndexList[index];
        int inSlot = devProg->outputInplaceSlotList[index];
        if (inSlot != -1) {
            DEV_ASSERT_MSG(outSlot >= 0 && outSlot < static_cast<int>(slotSize), "Invalid slot index %d", outSlot);
            DEV_ASSERT_MSG(inSlot >= 0 && inSlot < static_cast<int>(slotSize), "Invalid slot index %d", inSlot);
            workspace_->RuntimeOutcastTensorAssign(slotList[outSlot].rtOutcastIter, slotList[inSlot].rtOutcastIter);
            slotList[outSlot].isOutputSlot = true;
            DEV_VERBOSE_DEBUG("Param %zu Output Slot %d = inSlot %d.", index, outSlot, inSlot);
        }
    }
    for (size_t index = 0; index < devProg->assembleSlotIndexList.size(); ++index) {
        int slotIndex = devProg->assembleSlotIndexList[index];
        DEV_ASSERT_MSG(slotIndex >= 0 && slotIndex < static_cast<int>(slotSize), "Invalid slot index %d", slotIndex);
        slotList[slotIndex].isAssembleSlot = true;
        DEV_VERBOSE_DEBUG("Assemble Slot %d .", slotIndex);
    }
    for (size_t index = 0, ie = devProg->partialUpdateList.size(); index < ie; index++) {
        auto &partialUpdate = devProg->At(devProg->partialUpdateList, index);
        int slotIndex = index;
        DEV_ASSERT_MSG(slotIndex >= 0 && slotIndex < static_cast<int>(slotSize), "Invalid slot index %d", slotIndex);
        if (!partialUpdate.Empty()) {
            slotList[slotIndex].isPartialUpdateStitch = true;
            slotList[slotIndex].partialUpdate = &partialUpdate;
            DEV_VERBOSE_DEBUG("Partial Update Slot %d.\n", slotIndex);
        }
    }
}

}