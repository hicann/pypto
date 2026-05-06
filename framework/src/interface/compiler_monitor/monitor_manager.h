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

#include <atomic>
#include <chrono>
#include <map>
#include <unordered_map>
#include <mutex>
#include <string>
#include <vector>

namespace npu::tile_fwk {
const std::string STAGE_FUNC_TO_BIN = "FuncToBin";

class MonitorImpl;

struct ActiveStageInfo {
    std::string stageName;
    std::chrono::steady_clock::time_point startTime;
    int rootFuncIndex{0};
    std::string rootFuncName;
    int functionIndex{0};
    std::string functionName;
    int rootFuncOpSize{0};
};

class MonitorManager {
public:
    static MonitorManager& Instance();

    void Initialize(bool enable, int interval_sec, int timeout_sec, int total_timeout_sec);
    void Shutdown();

    void StartStage(const std::string& name, int rootFuncIndex = -1, const std::string& rootFuncName = "",
                    int rootFuncOpSize = 0);
    void EndStage(const std::string& name, int rootFuncIndex = -1, const std::string& rootFuncName = "",
                  int rootFuncOpSize = 0);
    double GetCurrentStageElapsed(const std::string& name);

    void SetTotalFunctionCount(int n);
    int GetAndIncrementNextFunctionIndex();
    void SetCurrentFunctionIndex(int k);

    void SwitchStageReset();

    void SetRootFuncCount(int n);
    int PrepareNextRootFunc();
    std::string GetCurrentRootFuncName() const;
    int GetRootFuncCount() const;
    int GetCurrentRootFuncIndex() const;

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

    void SetStageTimeoutFlag(const std::string& name);
    bool GetStageTimeoutFlag(const std::string& name);

    std::string GetCurrentFunctionName() const;
    void SetCurrentFunctionName(const std::string& name);
    int GetCurrentFuncOpSize() const;
    void SetCurrentFuncOpSize(size_t op_size);
    int GetFuncSumOpSize() const;
    void SetFuncSumOpSize(size_t op_size, bool reset = false);
    double GetTotalElapsed() const;
    void PrintCurrentTotalElapsed(std::string str_temp = "");

    std::vector<ActiveStageInfo> GetActiveStages() const;

    int GetProcessingThresholdSec() const;
    void SetProcessingThresholdSec(int sec);
    int GetProgressWidth() const;

    MonitorManager() = default;
    ~MonitorManager();
    MonitorManager(const MonitorManager&) = delete;
    MonitorManager& operator=(const MonitorManager&) = delete;

private:
    void MaybeStartTotalClock();
    void PrintCompilationFinished();
    void EndStageInternal(
        const std::string& name, int rootFuncIndex, const std::string& rootFuncName,
        const std::chrono::steady_clock::time_point& startTime, int rootFuncIndexOriginal, int rootFuncOpSize);

    mutable std::mutex mutex_;
    MonitorImpl* impl_{nullptr};
    bool initialized_{false};
    bool python_stage_ended_{false};

    bool enable_{false};
    bool stage_doing_{false};
    std::atomic<int> interval_sec_{60};
    std::atomic<int> timeout_sec_{-1};
    std::atomic<int> total_timeout_sec_{600};

    std::string current_function_;
    std::string current_stage_;
    std::chrono::steady_clock::time_point total_start_;
    std::chrono::steady_clock::time_point stage_start_;
    std::unordered_map<std::string, double> stage_elapsed_totals_;
    std::map<std::string, bool> stage_timeout_flag_;

    std::vector<ActiveStageInfo> active_stages_;

    int current_func_opsize_{0};
    int func_sum_opsize_{0};
    int total_function_count_{0};
    int current_function_index_{0};
    int next_function_index_{1};
    int root_func_count_{0};
    int current_root_func_index_{0};
    int next_root_func_index_{1};
    std::string current_root_func_;
    double last_total_elapsed_{0.0};
    int processing_threshold_sec_{60};
};

} // namespace npu::tile_fwk
