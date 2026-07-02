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
 * \file schedule_notifier.h
 * \brief Notification dispatcher for schedule instrumentation events.
 *        Holds observer list and broadcasts events. Event construction stays
 *        in the calling code (OoOScheduler / SpillEngine / DualDstEngine).
 */

#ifndef PASS_SCHEDULE_NOTIFIER_H
#define PASS_SCHEDULE_NOTIFIER_H

#include <vector>
#include "passes/statistics/schedule_observer.h"

namespace npu::tile_fwk {

class ScheduleNotifier {
public:
    ScheduleNotifier() {}
    ~ScheduleNotifier() {}

    void AddObserver(ScheduleObserver* observer) { observers_.push_back(observer); }

    std::vector<ScheduleObserver*>& observers() { return observers_; }
    const std::vector<ScheduleObserver*>& observers() const { return observers_; }

    bool HasObservers() const { return !observers_.empty(); }

    void BroadcastOpLaunch(const OpLaunchEvent& event) {
        for (auto* obs : observers_) { obs->OnOpLaunch(event); }
    }

    void BroadcastOpRetire(const OpRetireEvent& event) {
        for (auto* obs : observers_) { obs->OnOpRetire(event); }
    }

    void BroadcastAllocExec(const AllocExecEvent& event) {
        for (auto* obs : observers_) { obs->OnAllocExec(event); }
    }

    void BroadcastSpill(const SpillEvent& event) {
        for (auto* obs : observers_) { obs->OnSpill(event); }
    }

    void BroadcastBufferRearrange(const BufferRearrangeEvent& event) {
        for (auto* obs : observers_) { obs->OnBufferRearrange(event); }
    }

    void BroadcastAllocFail(const AllocFailEvent& event) {
        for (auto* obs : observers_) { obs->OnAllocFail(event); }
    }

    void BroadcastScheduleEnd(const ScheduleEndEvent& event) {
        for (auto* obs : observers_) { obs->OnScheduleEnd(event); }
    }

    void BroadcastInitDDRBuffer(const InitDDRBufferEvent& event) {
        for (auto* obs : observers_) { obs->OnInitDDRBuffer(event); }
    }

    void BroadcastMainLoopBegin() {
        for (auto* obs : observers_) { obs->OnMainLoopBegin(); }
    }

    void BroadcastMainLoopEnd() {
        for (auto* obs : observers_) { obs->OnMainLoopEnd(); }
    }

private:
    std::vector<ScheduleObserver*> observers_;
};

} // namespace npu::tile_fwk
#endif // PASS_SCHEDULE_NOTIFIER_H
