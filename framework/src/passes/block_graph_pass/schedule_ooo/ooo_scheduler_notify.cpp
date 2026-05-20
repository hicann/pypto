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
 * \file ooo_scheduler_notify.cpp
 * \brief Observer notification helpers — all event-construction logic lives here
 *        so scheduler main flows stay focused on scheduling.
 */

#include <numeric>
#include "ooo_scheduler.h"

namespace npu::tile_fwk {

void OoOScheduler::NotifyPipeIssued(PipeType pipeType, int latency)
{
    if (observers_.empty()) return;
    PipeIssuedEvent event{pipeType, latency, clock};
    for (auto* obs : observers_) {
        obs->OnPipeIssued(event);
    }
}

void OoOScheduler::NotifyBufferAllocated(MemoryType memType, int memId)
{
    if (observers_.empty()) return;
    BufferAllocEvent event{memType, memId, localBufferMap_.at(memId)->size, clock};
    for (auto* obs : observers_) {
        obs->OnBufferAllocated(event);
    }
}

void OoOScheduler::NotifyBufferFreed(MemoryType memType, int memId)
{
    if (observers_.empty()) return;
    BufferFreeEvent event{memType, memId, localBufferMap_.at(memId)->size, clock};
    for (auto* obs : observers_) {
        obs->OnBufferFreed(event);
    }
}

void OoOScheduler::NotifySpill(LogicalTensorPtr spillTensor, int spillMemId,
    Operation* spillAllocOp, Operation* spillOp)
{
    if (observers_.empty()) return;

    MemoryType memType = spillTensor->GetMemoryTypeOriginal();
    bool needCopyOut = spillOp->GetOpcodeStr().find("COPY_IN") == std::string::npos;
    uint64_t allocOccupied = 0;
    for (const auto& pair : tensorOccupyMap) {
        if (opIsAllocMap[pair.second]) {
            allocOccupied += localBufferMap_.at(pair.first)->size;
        }
    }
    uint64_t spillCopyoutSize = 0;
    if (needCopyOut) {
        spillCopyoutSize = spillTensor->tensor->GetRawDataSize();
    }
    SpillEvent event{memType, spillMemId,
        localBufferMap_.at(spillMemId)->size,
        spillAllocOp->GetOutputOperand(0)->tensor->GetRawDataSize(),
        spillTensor->GetMagic(), allocOccupied, spillCopyoutSize, clock};
    for (auto* obs : observers_) {
        obs->OnSpill(event);
    }
}

void OoOScheduler::NotifyScheduleEnd(bool success)
{
    if (observers_.empty()) return;
    ScheduleEndEvent event{clock, workspaceOffset, success};
    for (auto* obs : observers_) {
        obs->OnScheduleEnd(event);
    }
}

} // namespace npu::tile_fwk
