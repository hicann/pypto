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

// 抽离自 OoOScheduler::ScheduleMainLoop 与 LatencyEstimator::LatencyEstimatorMainLoop。
// 该基类只承载循环骨架与共享的两个状态字段(clock / numTotalIssues),
// 各派生类通过重写下述纯虚钩子来提供具体的 stage 实现。
class ScheduleMainLoopBase {
public:
    ScheduleMainLoopBase() = default;
    virtual ~ScheduleMainLoopBase() = default;

    // 统一入口: 派生类调用此方法运行完整的主循环。
    Status RunMainLoop();

    // 共享状态(派生类与外部均可访问,保持与原 OoOScheduler/LatencyEstimator 行为一致)
    int clock{0};
    uint64_t numTotalIssues{0};

protected:
    // ===== 模板方法钩子 =====
    // 进入主循环之前的初始化(例如更新执行序、激活就绪 issue、设置 numTotalIssues)。
    virtual Status PreMainLoop() = 0;
    // 主循环正常退出后的收尾(例如打印 latency)。
    virtual Status PostMainLoop() = 0;

    virtual Status RetireIssueStage(uint64_t& commitCnt, int& nextCycle) = 0;
    virtual Status BufferAllocStage(uint64_t& commitCnt) = 0;
    virtual Status LaunchIssueStage(int& nextCycle) = 0;
    virtual Status SpillOnBlock() = 0;
};

} // namespace npu::tile_fwk
#endif // PASS_SCHEDULE_MAIN_LOOP_BASE_H
