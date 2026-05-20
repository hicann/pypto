/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file schedule_observer.h
 * \brief Observer interface and event structs for OoO schedule instrumentation.
 *
 * Design:
 *   - Scheduler holds vector<ScheduleObserver*>, dispatches events via Notify().
 *   - When no observers are registered, the for-loop body is never entered — zero overhead.
 *   - Each event is a lightweight POD struct filled at the call site.
 *   - Adding new fields to a struct does NOT change the virtual interface signature.
 *   - Concrete observers (HealthCheck, MemoryTracer, ...) override only the callbacks they need.
 */

#ifndef SCHEDULE_OBSERVER_H
#define SCHEDULE_OBSERVER_H

#include <cstdint>
#include <vector>
#include "interface/utils/common.h"
#include "interface/operation/opcode.h"

namespace npu::tile_fwk {

// ─────────────────────────────────────────────────────────────
//  Event structs — one per instrumentation point
// ─────────────────────────────────────────────────────────────

struct PipeIssuedEvent {
    PipeType pipeType;
    int latency;
    int clock;
};

struct BufferAllocEvent {
    MemoryType memType;
    int memId;
    uint64_t size;
    int clock;
};

struct BufferFreeEvent {
    MemoryType memType;
    int memId;
    uint64_t size;
    int clock;
};

struct SpillEvent {
    MemoryType memType;
    int spillMemId;
    uint64_t spillTensorSize;
    int64_t triggerTensorSize;
    int spillTensorMagic;
    uint64_t allocOccupiedSize;
    uint64_t spillCopyoutSize;
    int clock;
};

struct ScheduleEndEvent {
    int totalCycles;
    int64_t workspaceOffset;
    bool success;
};

// ─────────────────────────────────────────────────────────────
//  Observer interface — all callbacks have empty default bodies
// ─────────────────────────────────────────────────────────────

class ScheduleObserver {
public:
    virtual ~ScheduleObserver() = default;

    virtual void OnPipeIssued(const PipeIssuedEvent&) {}
    virtual void OnBufferAllocated(const BufferAllocEvent&) {}
    virtual void OnBufferFreed(const BufferFreeEvent&) {}
    virtual void OnSpill(const SpillEvent&) {}
    virtual void OnScheduleEnd(const ScheduleEndEvent&) {}
};

} // namespace npu::tile_fwk
#endif // SCHEDULE_OBSERVER_H
