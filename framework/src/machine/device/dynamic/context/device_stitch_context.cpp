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

namespace npu::tile_fwk::dynamic {
void DeviceStitchContext::Init(DevAscendProgram *devProg, DeviceWorkspaceAllocator &workspace) {
    workspace.SetupVector(stitchedList_);
    workspace_ = &workspace;

    workspace_->SetupVector(slotInfosInDecidingSlotMem_);
    slotInfosInDecidingSlotMem_.resize(devProg->slotSize); // need pre alloc , left memory for slab allocator

    Reset();
}

void DeviceStitchContext::Reset() {
    stitchedList_.clear();
    stitchReuseContext_.firstDupIdx = 0;
    stitchReuseContext_.lastNonEmptyDupIdx = -1;
}

void DeviceStitchContext::DumpStitchInfo() {
    DumpStitchInfo(stitchedList_.data(), stitchedList_.size());
}

void DeviceStitchContext::CheckStitch(DevAscendFunctionDupped *stitchedList, int size, DevAscendFunctionDupped *nextDup) {
    DEV_IF_NONDEVICE {
        uint32_t dynPredCount = 0;
        uint32_t dynSuccCount = 0;
        for (int k = 0; k <= size; k++) {
            DevAscendFunctionDupped *dup = nullptr;
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
            DEV_ERROR("dynPredCount %u does not match dynSuccCount %u", dynPredCount, dynSuccCount);
        }
        DEV_ASSERT(dynPredCount == dynSuccCount);
    }
}

void DeviceStitchContext::CheckStitch(DynDeviceTask *dyntask) {
    DevAscendFunctionDupped *stitchedList = &dyntask->stitchedList[0];
    int stitchedSize = dyntask->stitchedList.size();
    CheckStitch(stitchedList, stitchedSize, nullptr);
}

uint64_t DeviceStitchContext::Stitch(DeviceSlotContext &slotContext, DevAscendFunctionDupped &nextDup, size_t devTaskId,
                                     size_t devNextIdx) {
    uint64_t count = FastStitch(slotContext.GetSlotList(), slotContext.GetSlotSize(), nextDup, devTaskId, devNextIdx);
    if (stitchedList_.capacity() == 0) {
        /* This stitchedList_ vector can only allocate sufficient space once,
            during a single device task construction process.*/
        stitchedList_.reserve(MAX_CACHED_FUNC_NUM);
    }
    Append(nextDup);
    stitchedCallOpSize_ += (nextDup.GetSource()->GetOperationSize() - nextDup.GetSource()->hubOpCount_);
    return count;
}

void DeviceStitchContext::RecycleTensorWorkspace() {
    // recycle submitted tasks' workspace memory
    workspace_->RecycleDevFuncWorkspace();
    workspace_->TriggerDelayedRecycle();
}

void DeviceStitchContext::DumpSlotInfo(const char *label, DeviceExecuteSlot *slotList, size_t slotSize) {
    UNUSED(label);
    UNUSED(slotList);
    UNUSED(slotSize);
    DEV_IF_VERBOSE_DEBUG {
        DEV_DEBUG("[DecideSlotAddress] %s.", label);
        for (size_t slotIdx = 0; slotIdx < slotSize; slotIdx++) {
            [[maybe_unused]] const char *extraAttr = "";
            if (slotList[slotIdx].isOutputSlot) {
                extraAttr = " <output>";
            } else if (slotList[slotIdx].isAssembleSlot) {
                extraAttr = " <assemble>";
            }

            if (slotList[slotIdx].rtOutcastIter == ITEM_POOL_INVALID_INDEX) {
                DEV_DEBUG("[DecideSlotAddress]   Slot [%3lu]: <no tensor>%s", slotIdx, extraAttr);
                continue;
            }
            [[maybe_unused]] auto &outcastDesc = workspace_->GetRuntimeOutcastTensor(slotList[slotIdx].rtOutcastIter);
            DEV_DEBUG("[DecideSlotAddress]   Slot [%3lu]: %s%s",
                slotIdx, outcastDesc.Dump().c_str(), extraAttr);
        }
    }
}

void DeviceStitchContext::DecideSlotAddress(DeviceExecuteSlot *slotList, size_t slotSize) {
    [[maybe_unused]] static constexpr uint64_t NON_ADDR_MASK = UINT64_C(1) << 62;

    DumpSlotInfo("Update before", slotList, slotSize);
#if !DEBUG_INFINITE_LIFETIME
    for (size_t slotIdx = 0; slotIdx < slotSize; ++slotIdx) {
        auto &slot = slotList[slotIdx];
        if (slot.rtOutcastIter != ITEM_POOL_INVALID_INDEX &&
            workspace_->GetRuntimeOutcastTensor(slot.rtOutcastIter).property == RuntimeTensorMemProperty::DEVTASK_INNER_OUTCAST) {
            workspace_->RuntimeOutcastTensorReplaceAddrWithoutRecycle(
                slot.rtOutcastIter, workspace_->AllocateSlot(), RuntimeTensorMemProperty::BOUNDARY_OUTCAST);
        }
    }
#endif // !DEBUG_INFINITE_LIFETIME
    DumpSlotInfo("Update after", slotList, slotSize);
}

int DeviceStitchContext::DecideIncastOutcast(uint64_t taskId) {
    (void)taskId;
    for (size_t funcIndex = 0; funcIndex < stitchedList_.size(); ++funcIndex) {
        auto &dup = stitchedList_[funcIndex];
        // decide incast address
        size_t incastSize = dup.GetSource()->GetIncastSize();
        for (size_t i = 0; i < incastSize; ++i) {
            auto &desc = dup.GetIncastAddress(i);
            DEV_ASSERT(desc.IsRtOutcast());
            ItemPoolIter iter = desc.GetRtOutcastIter();
            uintdevptr_t addr = workspace_->GetRuntimeOutcastTensor(iter).addr;
            workspace_->RuntimeOutcastTensorDeref(iter);
            desc = AddressDescriptor::MakeFromAddress(addr);
        }

        // decide outcast address
        size_t outcastSize = dup.GetSource()->GetOutcastSize();
        for (size_t i = 0; i < outcastSize; ++i) {
            auto &desc = dup.GetOutcastAddress(i);
            DEV_ASSERT(desc.IsRtOutcast());
            ItemPoolIter iter = desc.GetRtOutcastIter();
            uintdevptr_t addr = workspace_->GetRuntimeOutcastTensor(iter).addr;
            workspace_->RuntimeOutcastTensorDeref(iter);
            desc = AddressDescriptor::MakeFromAddress(addr);
        }
    }
    return DEVICE_MACHINE_OK;
}

int DeviceStitchContext::MoveTo(DynDeviceTask *dynTask) {
    dynTask->stitchedList = std::move(stitchedList_);
    stitchedList_.clear();
    dynTask->devTask.coreFunctionCnt = stitchedCallOpSize_;
    stitchedCallOpSize_ = 0;

    if (dynTask->stitchedList.size() > MAX_CACHED_FUNC_NUM) {
        DEV_ERROR("Stitch list size:%u exceeds maximum allowed cached function number:%zu.", dynTask->stitchedList.size(), MAX_CACHED_FUNC_NUM);
        return DEVICE_MACHINE_ERROR;
    }
    DEV_ASSERT(dynTask->stitchedList.size() <= MAX_CACHED_FUNC_NUM);
    int size = static_cast<int>(dynTask->stitchedList.size());
    for (int i = 0; i < size; ++i) {
        auto &funcDup = dynTask->stitchedList[i];
        dynTask->dynFuncDataCacheList[i] = {
            funcDup.GetSource(), &funcDup.GetOperationCurrPredCount(0), funcDup.GetSource()->GetCalleeIndexAddr(), funcDup.DupDataForDynFuncData()};
        dynTask->devTask.mixTaskData.opWrapList[i] = PtrToValue(funcDup.GetSource()->GetOpWrapListAddr());
        dynTask->devTask.mixTaskData.opWrapTaskNumList[i] = PtrToValue(funcDup.GetSource()->GetOpWrapTaskNumListAddr());
    }
    dynTask->dynFuncDataCacheListSize = size;
    return DEVICE_MACHINE_OK;
}

void DeviceStitchContext::HandleOneStitch(
        DevAscendFunctionDupped &producerDup, DevAscendFunctionDupped &consumerDup,
        DevAscendFunctionDuppedStitchList &producerStitchList, size_t producerOperationIdx,
        size_t consumerIdx, size_t consumerOperationIdx,
        DeviceWorkspaceAllocator *workspace,
        StitchKind debugStitchKind, int debugSlotIdx) {
    (void)debugStitchKind;
    (void)debugSlotIdx;

    PushBackTask(producerStitchList, MakeTaskID(consumerIdx, consumerOperationIdx), workspace);
    consumerDup.GetOperationCurrPredCount(consumerOperationIdx)++;

    DEV_IF_NONDEVICE {
        if (producerOperationIdx >= producerDup.GetSource()->GetOperationSize()) {
            DEV_ERROR("producerOperationIdx %zu exceeds the size of GetOperation %zu", producerOperationIdx, producerDup.GetSource()->GetOperationSize());
        }
        if (consumerOperationIdx >= consumerDup.GetSource()->GetOperationSize()) {
            DEV_ERROR("consumerOperationIdx %zu exceeds the size of GetOperation %zu", consumerOperationIdx, consumerDup.GetSource()->GetOperationSize());
        }
        DEV_ASSERT(producerOperationIdx < producerDup.GetSource()->GetOperationSize());
        DEV_ASSERT(consumerOperationIdx < consumerDup.GetSource()->GetOperationSize());
        DEV_VERBOSE_DEBUG("[Stitch] slot:%d kind:%s dupIdx:%d funcKey:%d,op:%d -> funcKey:%d,op:%d\n",
                    debugSlotIdx, GetStitchKindName(debugStitchKind).c_str(), (int)consumerIdx,
                    producerDup.GetSource()->GetFuncKey(), (int)producerOperationIdx,
                    consumerDup.GetSource()->GetFuncKey(), (int)consumerOperationIdx);
    }
}

void DeviceStitchContext::HandleOneStitch(
        DevAscendFunctionDupped &producerDup, DevAscendFunctionDupped &consumerDup,
        size_t producerOperationIdx, size_t consumerIdx, size_t consumerOperationIdx,
        DeviceWorkspaceAllocator *workspace, StitchKind debugStitchKind, int debugSlotIdx) {
    auto &producerStitchList = producerDup.GetOperationStitch(producerOperationIdx, false);
    HandleOneStitch(producerDup, consumerDup, producerStitchList, producerOperationIdx,
        consumerIdx, consumerOperationIdx, workspace, debugStitchKind, debugSlotIdx);
}

uint64_t DeviceStitchContext::PartialUpdateStitch(DevAscendFunctionDupped &nextDup, size_t devTaskId, size_t devNextIdx,
        DeviceExecuteSlot& slot, int slotIdx, DevAscendFunctionIncast& incast) {
    uint64_t matchCount = 0;
    auto *nextSrc = nextDup.GetSource();
    auto expressionList = &nextDup.GetExpression(0);
    auto &cellMatchTableDesc = slot.partialUpdate->cellMatchTableDesc;
    auto partialUpdateTableData = &slot.partialUpdate->cellMatchRuntimePartialUpdateTable[0];
    struct HandleCellMatchPartial {
        static inline void Process(int index, uint64_t *cellMatchTableData, uint64_t *matchCount,
                DevAscendFunctionDupped *stitchingList, int stitchingSize, DevAscendFunctionDupped *nextDup,
                size_t devTaskId, size_t devNextIdx, int consumerOperationIdx,
                DeviceWorkspaceAllocator *workspace,
                int debugSlotIdx) {
            uint64_t id = cellMatchTableData[index];
            if (id != AICORE_TASK_INIT && devTaskId == static_cast<uint32_t>(id >> TASKID_SHIFT32)) {
                auto funcId = FuncID(static_cast<uint32_t>(id));
                auto producerOperationIdx = TaskID(static_cast<uint32_t>(id));
                DevAscendFunctionDupped &prevDup = stitchingList[funcId];
                (*matchCount)++;
                DEV_VERBOSE_DEBUG("nextindex %lu stitch depend slot table cell[%d] = taskid(%u ! %u),", devNextIdx, index, funcId, producerOperationIdx);
                DeviceStitchContext::HandleOneStitch(prevDup, *nextDup, producerOperationIdx, devNextIdx, consumerOperationIdx,
                    workspace, StitchKind::StitchPartial, debugSlotIdx);
                DeviceStitchContext::CheckStitch(stitchingList, stitchingSize, nextDup);
            }
        }
    };
    for (size_t n = 0; n < incast.consumerList.size(); n++) {
        auto &consumer = nextSrc->At(incast.consumerList, n);
        uint64_t consumerOffset[DEV_SHAPE_DIM_MAX];
        uint64_t consumerShape[DEV_SHAPE_DIM_MAX];
        GetTensorOffsetAndShape<false>(
                nextSrc, consumerOffset, consumerShape, expressionList, incast.dim, consumer.operationIdx, consumer.operandIdx, true);

        DEV_IF_VERBOSE_DEBUG {
            for (int j = 0; j < cellMatchTableDesc.GetDimensionSize(); j++) {
                DEV_VERBOSE_DEBUG("PartialUpdateStitch cell match, operation[%d] -> dimension[%d] = (offset:%lu ,shape:%lu, cellshape:%d)",
                        consumer.operationIdx, j, consumerOffset[j], consumerShape[j], cellMatchTableDesc.cellShape.dim[j]);
            }
        }

        CellMatchHandle<HandleCellMatchPartial>(
                consumerOffset, consumerShape, cellMatchTableDesc,
                partialUpdateTableData, &matchCount, stitchedList_.data(), stitchedList_.size(), &nextDup,
                devTaskId, devNextIdx, consumer.operationIdx, workspace_, slotIdx);
    }
    return matchCount;
}

uint64_t DeviceStitchContext::FullCoverDefaultUpdateStitch(DevAscendFunctionDupped &nextDup, size_t devNextIdx,
    DeviceExecuteSlot& slot, int slotIdx, DevAscendFunctionIncast& incast) {
    uint64_t matchCount = 0;
    DevAscendFunctionDupped &prevDup = stitchedList_[slot.stitchDupIdx];
    auto *prevSrc = prevDup.GetSource();
    auto &outcast = prevSrc->GetOutcast(slot.stitchOutcastIdx);
    auto *nextSrc = nextDup.GetSource();
    auto expressionList = &nextDup.GetExpression(0);
    auto &cellMatchTableDesc = outcast.cellMatchTableDesc;
    auto fullUpdateTableData = &prevSrc->At(outcast.cellMatchRuntimeFullUpdateTable, 0);
    struct HandleCellMatchFull {
        static inline void Process(
                int index,
                uint32_t *cellMatchTableData,
                uint64_t *matchCount,
                DevAscendFunctionDupped *prevDup, DevAscendFunctionDupped *nextDup,
                size_t devNextIdx, int consumerOperationIdx,
                DeviceWorkspaceAllocator *workspace,
                int debugSlotIdx) {
            auto producerOperationIdx = cellMatchTableData[index];
            if (producerOperationIdx != static_cast<uint32_t>(-1)) {
                (*matchCount)++;
                DEV_TRACE_DEBUG(DEvent(DUid(none()), DActStitchEdge(
                    Producer(LUid(none(), 0, none(), producerOperationIdx, none()), none(), none(), debugSlotIdx, none(), none()),
                    Consumer(LUid(none(), 0, none(), consumerOperationIdx, none()), none(), none(), debugSlotIdx, none(), none()),
                    StitchReasonUniqueMatch())));
                DeviceStitchContext::HandleOneStitch(*prevDup, *nextDup, producerOperationIdx, devNextIdx, consumerOperationIdx,
                    workspace, StitchKind::StitchDefault, debugSlotIdx);
            }
        }
    };
    for (size_t n = 0; n < incast.consumerList.size(); n++) {
        auto &consumer = nextSrc->At(incast.consumerList, n);
        uint64_t consumerOffset[DEV_SHAPE_DIM_MAX];
        uint64_t consumerShape[DEV_SHAPE_DIM_MAX];
        GetTensorOffsetAndShape<false>(
                nextSrc, consumerOffset, consumerShape, expressionList, incast.dim, consumer.operationIdx, consumer.operandIdx, true);
        CellMatchHandle<HandleCellMatchFull>(
                consumerOffset, consumerShape, cellMatchTableDesc,
                fullUpdateTableData,
                &matchCount,
                &prevDup, &nextDup,
                devNextIdx, consumer.operationIdx,
                workspace_,
                slotIdx);
        DeviceStitchContext::CheckStitch(stitchedList_.data(), stitchedList_.size(), &nextDup);
    }
    return matchCount;
}

uint64_t DeviceStitchContext::FullCoverUpdateStitch(DevAscendFunctionDupped &nextDup, size_t devNextIdx,
    DeviceExecuteSlot& slot, int slotIdx, DevAscendFunctionIncast& incast) {
    DevAscendFunctionDupped &prevDup = stitchedList_[slot.stitchDupIdx];
    auto *prevSrc = prevDup.GetSource();
    auto &outcast = prevSrc->GetOutcast(slot.stitchOutcastIdx);
    auto *nextSrc = nextDup.GetSource();
    DEV_VERBOSE_DEBUG("outcast %lu is %d, cellMatchStaticOutcastTable is %s\n", (unsigned long)slot.stitchOutcastIdx,
        outcast.stitchByAllFullMatch, IntVecToStr(prevDup, outcast.cellMatchStaticOutcastTable).c_str());
    DEV_VERBOSE_DEBUG("=================FullCoverUpdateStitch %zu %zu %zu %zu %d %d===========================\n",
        outcast.producerList.size(), incast.consumerList.size(),
        outcast.cellMatchStaticOutcastTable.size(), incast.cellMatchStaticIncastTable.size(),
        outcast.stitchByAllFullMatch, incast.stitchByAllFullMatch);

    // stitchPolicyFullCover hub
    auto producerHubOpIdx = outcast.stitchPolicyFullCoverProducerHubOpIdx;
    if (producerHubOpIdx != -1) {
        auto consumerAllOpIdxList = &nextSrc->At(incast.stitchPolicyFullCoverConsumerAllOpIdxList, 0);
        for (size_t conIndex = 0, conSize = incast.stitchPolicyFullCoverConsumerAllOpIdxList.size(); conIndex < conSize; conIndex++) {
            auto &consumerOpIdx = consumerAllOpIdxList[conIndex];
            DeviceStitchContext::HandleOneStitch(prevDup, nextDup, producerHubOpIdx, devNextIdx, consumerOpIdx,
                workspace_, StitchKind::StitchFullCover, slotIdx);
        }
        DeviceStitchContext::CheckStitch(stitchedList_.data(), stitchedList_.size(), &nextDup);
    } else {
        // stitchPolicyFullCover producer
        auto producerList = &prevSrc->At(outcast.stitchPolicyFullCoverProducerList, 0);
        auto consumerAllOpIdxList = &nextSrc->At(incast.stitchPolicyFullCoverConsumerAllOpIdxList, 0);
        for (size_t prodIndex = 0, prodSize = outcast.stitchPolicyFullCoverProducerList.size(); prodIndex < prodSize; prodIndex++) {
            auto &producer = producerList[prodIndex];
            auto producerOperationIdx = producer.operationIdx;

            for (size_t conIndex = 0, conSize = incast.stitchPolicyFullCoverConsumerAllOpIdxList.size(); conIndex < conSize; conIndex++) {
                auto &consumerOpIdx = consumerAllOpIdxList[conIndex];
                DeviceStitchContext::HandleOneStitch(prevDup, nextDup, producerOperationIdx, devNextIdx, consumerOpIdx,
                    workspace_, StitchKind::StitchFullCover, slotIdx);
            }
        }
        DeviceStitchContext::CheckStitch(stitchedList_.data(), stitchedList_.size(), &nextDup);
    }

    return FullCoverDefaultUpdateStitch(nextDup, devNextIdx, slot, slotIdx, incast);
}

void DeviceStitchContext::ReuseStitch(DevAscendFunctionDupped &nextDup, size_t devNextIdx) {
    if (nextDup.GetSource()->rootInnerTensorWsMemoryRequirement == 0) {
        // 0 length workspace, no dependency in need
        return;
    }

    uintdevptr_t nextAddrL = nextDup.RuntimeWorkspace();
    uintdevptr_t nextAddrR = nextAddrL + nextDup.GetSource()->rootInnerTensorWsMemoryRequirement;
    auto nextReuseInfo = nextDup.GetRuntimeReuseInfo();
    if (auto &firstDup = stitchedList_[stitchReuseContext_.firstDupIdx];
        firstDup.GetRuntimeReuseInfo().poolResetTimes >= nextReuseInfo.poolResetTimes) {
        return;
    }

    auto needsDependency = [&](uint32_t prevIdx) -> int {
        if (prevIdx >= devNextIdx) {
            // invalid idx
            return INVALID_TOO_AHEAD;
        }

        auto &prevDup = stitchedList_[prevIdx];
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
    for (; skipBefore(needsDependency(stitchReuseContext_.firstDupIdx)); stitchReuseContext_.firstDupIdx++) {}

    if (needsDependency(stitchReuseContext_.firstDupIdx) == NEEDS_DEP) {
        for (uint32_t prevIdx = stitchReuseContext_.firstDupIdx; ; prevIdx++) {
            int res = needsDependency(prevIdx);
            if (res == NO_DEP || res == INVALID_TOO_AHEAD) {
                break;
            }
            if (res != SKIP_EMPTY) {
                auto &prevDup = stitchedList_[prevIdx];
                StitchForWorkspaceReuse(stitchedList_.data(), stitchedList_.size(), prevDup, nextDup, devNextIdx, workspace_);
                stitchReuseContext_.firstDupIdx = prevIdx; // Risk on time complexity: Duplicated access to empty-workspace funcs
            }
        }
    } else {
        if (stitchReuseContext_.lastNonEmptyDupIdx != -1) {
            auto &prevDup = stitchedList_[stitchReuseContext_.lastNonEmptyDupIdx];
            StitchForWorkspaceReuse(stitchedList_.data(), stitchedList_.size(), prevDup, nextDup, devNextIdx, workspace_);
        }
    }
}

uint64_t DeviceStitchContext::FastStitch(DeviceExecuteSlot *slotList, size_t slotSize, DevAscendFunctionDupped &nextDup,
    size_t devTaskId, size_t devNextIdx) {
    AutoScopedPerf asp(PERF_EVT_FAST_STITCH);
#if !ENABLE_STITCH
    return 0;
#endif
    auto *nextSrc = nextDup.GetSource();
    nextDup.GetSource()->GetFuncidx() = static_cast<int>(devNextIdx);
    if (devNextIdx == 0) {
        // The only function, don't need stitch
        return 0;
    }
    uint64_t matchCount = 0;
    for (size_t incastIdx = 0; incastIdx < nextSrc->GetIncastSize(); ++incastIdx) {
        auto &incast = nextSrc->GetIncast(incastIdx);

        for (size_t j = 0; j < incast.fromSlotList.size(); ++j) {
            auto slotIdx = nextSrc->At(incast.fromSlotList, j);
            if (slotIdx >= (int)slotSize) {
                DEV_ERROR("slotIdx %d is larger than slotSize %zu!.", slotIdx, slotSize);
                continue;
            }

            auto &slot = slotList[slotIdx];
            DEV_VERBOSE_DEBUG("FastStitch slot %d, incastindex %zu, ispartial %d, stitchDupIdx %u",
                slotIdx, incastIdx, slot.isPartialUpdateStitch, slot.stitchDupIdx);
            if (slot.stitchDupIdx == INVALID_STITCH_IDX) {
                // Slot never output
                continue;
            }

            if (slot.isPartialUpdateStitch) {
                matchCount = PartialUpdateStitch(nextDup, devTaskId, devNextIdx, slot, slotIdx, incast);
                continue;
            }

            if (slot.rtOutcastIter == ITEM_POOL_INVALID_INDEX) {
                continue;
            }
            DEV_VERBOSE_DEBUG("incast %zu is %d, cellMatchStaticIncastTable is %s\n", incastIdx, incast.stitchByAllFullMatch,
                IntVecToStr(nextDup, incast.cellMatchStaticIncastTable).c_str());
            matchCount = FullCoverUpdateStitch(nextDup, devNextIdx, slot, slotIdx, incast);
        }
    }
#if !DEBUG_INFINITE_LIFETIME
    ReuseStitch(nextDup, devNextIdx);
#endif // !DEBUG_INFINITE_LIFETIME
    return matchCount;
}

void DeviceStitchContext::DumpStitchInfo(DevAscendFunctionDupped *stitchedList, int stitchedSize) {
    int funcId = 0;
    for (int i = 0; i < stitchedSize; i++) {
        auto &funcDup = stitchedList[i];
        for (size_t opIndex = 0; opIndex < funcDup.GetSource()->GetOperationSize(); opIndex++) {
            auto &stitch = funcDup.GetOperationStitch(opIndex);
            if (stitch.IsNull()) {
                continue;
            }
            std::stringstream oss;
            oss << stitch.Dump();
            DEV_VERBOSE_DEBUG("func %d opIndex %zu stitch list: %s.", funcId, opIndex, oss.str().c_str());
        }
        funcId++;
    }
}

void DeviceStitchContext::StitchForWorkspaceReuse(DevAscendFunctionDupped *stitchingList, int stitchingSize,
    DevAscendFunctionDupped &prevDup, DevAscendFunctionDupped &currDup,
    size_t devCurrIdx, DeviceWorkspaceAllocator *workspace) {
    // Add dependency between root functions
    auto *prevSrc = prevDup.GetSource();
    auto *currSrc = currDup.GetSource();

    size_t prevNoSuccOpSize = prevSrc->GetNoSuccOpSize();
    size_t currNoPredOpSize = currSrc->GetNoPredOpSize();
    if (unlikely(prevNoSuccOpSize == 0 || currNoPredOpSize == 0)) {
        // Empty root function
        return;
    }

    // Graph has been optimized when encoding, we just put trivial full connection logics here
    for (size_t i = 0; i < prevNoSuccOpSize; ++i) {
        int prevNoSucc = prevSrc->GetNoSuccOpIdx(i);
        auto &stitch = prevDup.GetOperationStitch(prevNoSucc);
        for (size_t j = 0; j < currNoPredOpSize; ++j) {
            int currNoPred = currSrc->GetNoPredOpIdx(j);
            DEV_TRACE_DEBUG(DEvent(DUid(none()), DActStitchEdge(
                Producer(LUid(none(), 0, none(), prevNoSucc, none()), none(), none(), none(), none(), none()),
                Consumer(LUid(none(), 0, none(), currNoPred, none()), none(), none(), none(), none(), none()),
                StitchReasonWorkspaceReuse())));
            DeviceStitchContext::HandleOneStitch(prevDup, currDup, stitch, prevNoSucc, devCurrIdx, currNoPred, workspace, DeviceStitchContext::StitchKind::StitchReuse, -1);
            DeviceStitchContext::CheckStitch(stitchingList, stitchingSize, &currDup);
        }
    }
}
}