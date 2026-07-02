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
 * \file schedule_main_loop_base.h
 * \brief Template-method base class that owns the shared OoO schedule main loop.
 */

#ifndef PASS_SCHEDULE_MAIN_LOOP_BASE_H
#define PASS_SCHEDULE_MAIN_LOOP_BASE_H

#include <cstdint>
#include "interface/utils/common.h"

namespace npu::tile_fwk {

// Template-method base for the OoO schedule main loop.
// clock / numTotalIssues are now accessed via virtual getters GetClock()/GetNumTotalIssues(),
// allowing derived classes to source them from ScheduleState or local fields.
class ScheduleMainLoopBase {
public:
    ScheduleMainLoopBase() = default;
    virtual ~ScheduleMainLoopBase() = default;

    Status RunMainLoop();

    // clock / numTotalIssues are no longer owned by this base class.
    // Derived classes provide access via virtual getters (typically from ScheduleState).
    virtual int& GetClock() = 0;
    virtual uint64_t& GetNumTotalIssues() = 0;

protected:
    virtual Status PreMainLoop() = 0;
    virtual Status PostMainLoop() = 0;

    virtual Status RetireIssueStage(uint64_t& commitCnt, int& nextCycle) = 0;
    virtual Status BufferAllocStage(uint64_t& commitCnt) = 0;
    virtual Status LaunchIssueStage(int& nextCycle) = 0;
    virtual Status SpillOnBlock() = 0;
};

} // namespace npu::tile_fwk
#endif // PASS_SCHEDULE_MAIN_LOOP_BASE_H
