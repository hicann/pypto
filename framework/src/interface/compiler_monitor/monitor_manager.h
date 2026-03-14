/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once

#include <chrono>
#include <map>
#include <unordered_map>
#include <mutex>
#include <string>

namespace npu::tile_fwk {

class MonitorImpl;

class MonitorManager {
public:
    static MonitorManager &Instance();

    void Initialize(bool enable, int interval_sec, int timeout_sec, int total_timeout_sec);
    void Shutdown();

    void StartStage(const std::string &name);
    void EndStage(const std::string &name);
    double GetCurrentStageElapsed(const std::string &name);

    void SetTotalFunctionCount(int n);
    int GetAndIncrementNextFunctionIndex();
    void SetCurrentFunctionIndex(int k);

    void TryEndPrepareStage();

    void NotifyCompilationFinished();

    void SetCompilerMonitorOptions(bool enable, int interval_sec, int timeout_sec, int total_timeout_sec);
    bool IsEnabled() const;
    int GetIntervalSec() const;
    int GetTimeoutSec() const;
    int GetTotalTimeoutSec() const;
    std::string GetCurrentStageName() const;
    std::chrono::steady_clock::time_point GetStageStartTime() const;
    std::chrono::steady_clock::time_point GetTotalStartTime() const;
    int GetTotalFunctionCount() const;
    int GetCurrentFunctionIndex() const;
    std::unordered_map<std::string, double> GetStageElapsedTotals() const;

    void SetStageTimeoutFlag(const std::string &name);
    bool GetStageTimeoutFlag(const std::string &name);

    std::string GetCurrentFunctionName() const;
    void SetCurrentFunctionName(const std::string &name);
    double GetTotalElapsed() const;

    MonitorManager() = default;
    ~MonitorManager();
    MonitorManager(const MonitorManager &) = delete;
    MonitorManager &operator=(const MonitorManager &) = delete;

private:
    void MaybeStartTotalClock();
    void PrintCompilationFinished();

    mutable std::mutex mutex_;
    MonitorImpl *impl_{nullptr};
    bool initialized_{false};
    bool python_stage_ended_{false};

    bool enable_{true};
    bool stage_doing_{false};
    int interval_sec_{60};
    int timeout_sec_{-1};
    int total_timeout_sec_{600};

    std::string current_function_;
    std::string current_stage_;
    std::chrono::steady_clock::time_point total_start_;
    std::chrono::steady_clock::time_point stage_start_;
    std::unordered_map<std::string, double> stage_elapsed_totals_;
    std::map<std::string, bool> stage_timeout_flag_;

    int total_function_count_{0};
    int current_function_index_{0};
    int next_function_index_{1};
    double last_total_elapsed_{0.0};
};

} // namespace npu::tile_fwk
