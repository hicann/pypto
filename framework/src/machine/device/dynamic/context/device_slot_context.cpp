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
#include "machine/device/dynamic/context/dump_device_topo.h"
#include <algorithm>
#include <unordered_set>
#include <vector>

namespace npu::tile_fwk::dynamic {

static void PrepareRuntimeDynamicPartialUpdateTable(
    DeviceWorkspaceAllocator* workspace, DevAscendProgramPartialUpdate* partialUpdate)
{
    auto& desc = partialUpdate->cellMatchTableDesc;
    const int dim = desc.GetDimensionSize();
    uint64_t tableSize = 1;
    bool launchPrepared = (dim > 0);
    for (int d = 0; d < dim; ++d) {
        int stride = desc.GetStrideShape(d);
        if (stride <= 0) {
            launchPrepared = false;
            break;
        }
        tableSize *= static_cast<uint64_t>(stride);
    }
    DEV_ASSERT_MSG(
        ProgEncodeErr::CELL_MATCH_PARAM_INVALID, launchPrepared,
        "Dynamic cell match launch prepare missing for slot=%d dim=%d", partialUpdate->slotIndex, dim);
    uint64_t slotCellCapacity = workspace->DynamicCellMatchSlotCellCapacity();
    DEV_ASSERT_MSG(
        ProgEncodeErr::CELL_MATCH_PARAM_INVALID, tableSize > 0 && tableSize <= slotCellCapacity,
        "Dynamic cell match table size invalid for slot=%d, tableSize=%lu capacity=%lu",
        partialUpdate->slotIndex, tableSize, slotCellCapacity);

    if (partialUpdate->cellMatchRuntimePartialUpdateTable.Data() == nullptr) {
        return;
    }
    partialUpdate->cellMatchRuntimePartialUpdateTable.HostAssignDataSize(
        reinterpret_cast<uintdevptr_t>(partialUpdate->cellMatchRuntimePartialUpdateTable.Data()), tableSize);
}

void DeviceSlotContext::InitAllocator(DeviceWorkspaceAllocator& workspace, uint64_t slotSize)
{
    workspace.SetupVector(slotList_);
    workspace_ = &workspace;
    slotList_.resize(slotSize);
}

void DeviceSlotContext::FillInputOutputSlot(DevAscendProgram* devProg, DevStartArgs* args)
{
    uint64_t progBegin = reinterpret_cast<uint64_t>(devProg);
    uint64_t progEnd = progBegin + devProg->GetSize();
    for (size_t i = 0, ie = devProg->partialUpdateList.size(); i < ie; ++i) {
        auto& partialUpdate = devProg->At(devProg->partialUpdateList, i);
        uint64_t tablePtr = reinterpret_cast<uint64_t>(partialUpdate.cellMatchRuntimePartialUpdateTable.Data());
        if (tablePtr == 0) {
            continue;
        }
        bool tableInProgramImage = (tablePtr >= progBegin) && (tablePtr < progEnd);
        if (!tableInProgramImage) {
            partialUpdate.cellMatchRuntimePartialUpdateTable.HostAssignDataSize(0, 0);
        }
    }
    FillInputOutputSlot(slotList_.data(), slotList_.size(), devProg, args);
}

static void UpdateSlotsForStitch(
    int slotIdx, DeviceExecuteSlot& slot, DevAscendFunction* devRootSrc, DevAscendFunctionOutcast& outcast,
    uint32_t devTaskId, uint32_t devNextIdx, uint32_t outcastIndex, uint64_t* expressionList)
{
    slot.stitchDupIdx = devNextIdx;
    slot.stitchOutcastIdx = outcastIndex;
    UNUSED(slotIdx);
    topo_dump::DumpProducerCellAccess(devTaskId, slotIdx, devNextIdx, *devRootSrc, outcast, slot, expressionList);

    auto producerList = &devRootSrc->At(outcast.producerList, 0);
    if (slot.isPartialUpdateStitch) {
        if (slot.partialUpdate->cellMatchRuntimePartialUpdateTable.size() == 0) {
            return;
        }
        auto& cellMatchTableDesc = slot.partialUpdate->cellMatchTableDesc;
        auto tableData = &slot.partialUpdate->cellMatchRuntimePartialUpdateTable[0];
        auto producerSize = outcast.producerList.size();
        if (producerSize != 0) {
            CellMatchFillIncastOutcast<false>(
                devRootSrc, producerList, producerSize, expressionList, false, cellMatchTableDesc, tableData, devTaskId,
                devNextIdx);
        } else {
            // maybe is fullcover producer, dassemble full shape
            CellMatchFillIncastOutcast<false>(
                devRootSrc, &devRootSrc->At(outcast.stitchPolicyFullCoverProducerList, 0),
                outcast.stitchPolicyFullCoverProducerList.size(), expressionList, false, cellMatchTableDesc, tableData,
                devTaskId, devNextIdx);
        }

        DEV_VERBOSE_DEBUG(
            "[UpdateSlots]  slot %d CellMatchPartial=%s\n", slotIdx,
            DevAscendFunctionDuppedStitchList::DumpTask<uint64_t>(
                tableData, slot.partialUpdate->cellMatchRuntimePartialUpdateTable.size())
                .c_str());
        slot.isPartialUpdateDirty = true;
    } else {
        auto& cellMatchTableDesc = outcast.cellMatchTableDesc;
        auto tableData = &devRootSrc->At(outcast.cellMatchRuntimeFullUpdateTable, 0);
        CellMatchFillIncastOutcast<false>(
            devRootSrc, producerList, outcast.producerList.size(), expressionList, false, cellMatchTableDesc,
            tableData);
        DEV_VERBOSE_DEBUG(
            "[UpdateSlots] slot %d  CellMatchFull=%s\n", slotIdx,
            DevAscendFunctionDuppedStitchList::DumpTask(tableData, outcast.cellMatchRuntimeFullUpdateTable.size())
                .c_str());
    }
}

static void PrepareRuntimeDynamicPartialUpdateTables(
    DeviceWorkspaceAllocator* workspace, DeviceExecuteSlot* slotList, DevAscendFunctionDupped& devRootDup)
{
    DevAscendFunction* devRootSrc = devRootDup.GetSource();
    std::unordered_set<DevAscendProgramPartialUpdate*> preparedPartials;
    size_t outcastSize = devRootSrc->GetOutcastSize();
    for (size_t i = 0; i < outcastSize; ++i) {
        auto& outcast = devRootSrc->GetOutcast(i);
        for (size_t j = 0; j < outcast.toSlotList.size(); ++j) {
            int slotIdx = devRootSrc->At(outcast.toSlotList, j);
            auto& slot = slotList[slotIdx];
            if (!slot.isPartialUpdateStitch || slot.partialUpdate == nullptr) {
                continue;
            }
            auto* partialUpdate = slot.partialUpdate;
            if (partialUpdate->cellMatchRuntimePartialUpdateTable.size() != 0 ||
                partialUpdate->cellMatchTableDesc.GetDimensionSize() <= 0) {
                continue;
            }
            if (partialUpdate->cellMatchRuntimePartialUpdateTable.Data() == nullptr) {
                continue;
            }
            if (!preparedPartials.insert(partialUpdate).second) {
                continue;
            }
            PrepareRuntimeDynamicPartialUpdateTable(workspace, partialUpdate);
        }
    }
}

static void UpdateSlotsImpl(
    DeviceWorkspaceAllocator* workspace, DeviceExecuteSlot* slotList, DevAscendFunctionDupped& devRootDup,
    uint32_t devTaskId, uint32_t devNextIdx)
{
    AutoScopedPerf asp(PERF_EVT_UPDATE_SLOT);
    DevAscendFunction* devRootSrc = devRootDup.GetSource();
    size_t outcastSize = devRootSrc->GetOutcastSize();

    // Update slot address
    uint64_t* expressionList = &devRootDup.GetExpression(0);
    PrepareRuntimeDynamicPartialUpdateTables(workspace, slotList, devRootDup);
    for (size_t i = 0; i < outcastSize; ++i) {
        const auto& outcastDesc = devRootDup.GetOutcastAddress(i);
        auto& outcast = devRootSrc->GetOutcast(i);
        for (size_t j = 0; j < outcast.toSlotList.size(); ++j) {
            int slotIdx = devRootSrc->At(outcast.toSlotList, j);
            auto& slot = slotList[slotIdx];
            UpdateSlotsForStitch(slotIdx, slot, devRootSrc, outcast, devTaskId, devNextIdx, i, expressionList);
            workspace->RuntimeOutcastTensorAssign(slot.rtOutcastIter, outcastDesc.GetRtOutcastIter());
            DEV_VERBOSE_DEBUG(
                "[UpdateSlots]   Outcast [%3zu] to slot [%3d], address %s.", i, slotIdx, outcastDesc.Dump().c_str());
        }
    }
}

void DeviceSlotContext::UpdateSlots(DevAscendFunctionDupped& devRootDup, uint32_t devTaskId, uint32_t devNextIdx)
{
    UpdateSlotsImpl(workspace_, slotList_.data(), devRootDup, devTaskId, devNextIdx);
}

static void MarkPartialUpdateSlots(DeviceExecuteSlot* slotList, size_t slotSize, DevAscendProgram* devProg)
{
    bool dynamicCellMatchBudgetReady =
        devProg->memBudget.metadata.dynamicCellMatchSlotNum > 0 &&
        devProg->memBudget.metadata.maxDynamicCellMatchTableMem > 0;
    for (size_t index = 0, ie = devProg->partialUpdateList.size(); index < ie; index++) {
        auto& partialUpdate = devProg->At(devProg->partialUpdateList, index);
        int slotIndex = partialUpdate.slotIndex;
        DEV_ASSERT_MSG(
            ProgEncodeErr::STITCH_HANDLE_INDEX_OUT_OF_RANGE, slotIndex >= 0 && slotIndex < static_cast<int>(slotSize),
            "Invalid slot index %d", slotIndex);
        bool hasPartialUpdateTable = !partialUpdate.Empty();
        bool isRuntimeDynamicPartialUpdate =
            dynamicCellMatchBudgetReady &&
            partialUpdate.cellMatchRuntimePartialUpdateTable.size() == 0 &&
            partialUpdate.cellMatchTableDesc.GetDimensionSize() > 0;
        if (hasPartialUpdateTable || isRuntimeDynamicPartialUpdate) {
            slotList[slotIndex].isPartialUpdateStitch = true;
            slotList[slotIndex].partialUpdate = &partialUpdate;
            DEV_VERBOSE_DEBUG("Partial Update Slot %d.\n", slotIndex);
        }
    }
}

static void FillExternalTensorSlot(
    DeviceExecuteSlot* slotList, size_t slotSize, DeviceWorkspaceAllocator* workspace, int slotIndex,
    uint64_t tensorAddr, int tensorIndex, uint64_t tensorSize, bool isInput)
{
    DEV_ASSERT_MSG(
        ProgEncodeErr::STITCH_HANDLE_INDEX_OUT_OF_RANGE, slotIndex >= 0 && slotIndex < static_cast<int>(slotSize),
        "Invalid slot index %d", slotIndex);
    slotList[slotIndex].rtOutcastIter =
        workspace->MakeRuntimeOutcastTensor(WsAllocation(tensorAddr, 0), RuntimeTensorMemProperty::EXTERNAL);
    slotList[slotIndex].isOutputSlot = true;
    DEV_INFO("Param %d %s Slot %d = %lx.", tensorIndex, isInput ? "Input" : "Output", slotIndex, tensorAddr);
    if (isInput) {
        DEV_TRACE_DEBUG(CtrlEvent(none(), InputTensorElement(tensorIndex, tensorAddr, tensorSize)));
    } else {
        DEV_TRACE_DEBUG(CtrlEvent(none(), OutputTensorElement(tensorIndex, tensorAddr, tensorSize)));
    }
}

void DeviceSlotContext::FillInputOutputSlot(
    DeviceExecuteSlot* slotList, [[maybe_unused]] size_t slotSize, DevAscendProgram* devProg, DevStartArgs* args)
{
    DEV_TRACE_DEBUG(CtrlEvent(none(), InputTensorCount(args->GetInputTensorSize())));
    for (int index = 0; index < args->GetInputTensorSize(); ++index) {
        DevTensorData& param = args->GetInputTensor(index);
        int slotIndex = devProg->startArgsInputTensorSlotIndexList[index];
        FillExternalTensorSlot(
            slotList, slotSize, workspace_, slotIndex, param.address, index, param.shape.GetSize(), true);
    }
    DEV_TRACE_DEBUG(CtrlEvent(none(), OutputTensorCount(args->GetOutputTensorSize())));
    for (int index = 0; index < args->GetOutputTensorSize(); ++index) {
        DevTensorData& param = args->GetOutputTensor(index);
        int slotIndex = devProg->startArgsOutputTensorSlotIndexList[index];
        FillExternalTensorSlot(
            slotList, slotSize, workspace_, slotIndex, param.address, index, param.shape.GetSize(), false);
    }
    for (size_t index = static_cast<size_t>(args->GetOutputTensorSize());
         index < devProg->startArgsOutputTensorSlotIndexList.size(); ++index) {
        int outSlot = devProg->startArgsOutputTensorSlotIndexList[index];
        int inSlot = devProg->outputInplaceSlotList[index];
        if (inSlot != -1) {
            DEV_ASSERT_MSG(
                ProgEncodeErr::STITCH_HANDLE_INDEX_OUT_OF_RANGE, outSlot >= 0 && outSlot < static_cast<int>(slotSize),
                "Invalid slot index %d", outSlot);
            DEV_ASSERT_MSG(
                ProgEncodeErr::STITCH_HANDLE_INDEX_OUT_OF_RANGE, inSlot >= 0 && inSlot < static_cast<int>(slotSize),
                "Invalid slot index %d", inSlot);
            workspace_->RuntimeOutcastTensorAssign(slotList[outSlot].rtOutcastIter, slotList[inSlot].rtOutcastIter);
            slotList[outSlot].isOutputSlot = true;
            DEV_VERBOSE_DEBUG("Param %zu Output Slot %d = inSlot %d.", index, outSlot, inSlot);
        }
    }
    for (size_t index = 0; index < devProg->assembleSlotIndexList.size(); ++index) {
        int slotIndex = devProg->assembleSlotIndexList[index];
        DEV_ASSERT_MSG(
            ProgEncodeErr::STITCH_HANDLE_INDEX_OUT_OF_RANGE, slotIndex >= 0 && slotIndex < static_cast<int>(slotSize),
            "Invalid slot index %d", slotIndex);
        slotList[slotIndex].isAssembleSlot = true;
        DEV_VERBOSE_DEBUG("Assemble Slot %d .", slotIndex);
    }
    MarkPartialUpdateSlots(slotList, slotSize, devProg);
}

} // namespace npu::tile_fwk::dynamic
