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

void OoOScheduler::NotifySpill(const SpillInfo& info, LocalBufferPtr allocBuffer)
{
    if (observers_.empty()) return;

    bool needCopyOut = info.spillOp_->GetOpcodeStr().find("COPY_IN") == std::string::npos;
    uint64_t allocOccupied = 0;
    for (const auto& pair : tensorOccupyMap[allocBuffer->memType]) {
        if (opIsAllocMap[pair.second]) {
            allocOccupied += localBufferMap_.at(pair.first)->size;
        }
    }
    uint64_t spillCopyoutSize = 0;
    if (needCopyOut) {
        auto dtype = info.ddrTensor_->tensor->datatype;
        spillCopyoutSize = std::accumulate(
            info.ddrTensor_->shape.begin(), info.ddrTensor_->shape.end(),
            1, std::multiplies<int64_t>()) * BytesOf(dtype);
    }
    SpillEvent event{allocBuffer->memType, info.spillMemId_,
        localBufferMap_.at(info.spillMemId_)->size, allocBuffer->size,
        info.ddrTensor_->GetMagic(), allocOccupied, spillCopyoutSize, clock};
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
