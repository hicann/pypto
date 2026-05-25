/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <iomanip>
#include <mutex>
#include <string>
#include <iostream>
#include <unistd.h>
#include <thread>

#include "tilefwk/pypto_fwk_log.h"
#include "interface/compiler_monitor/monitor_manager.h"
#include "interface/compiler_monitor/monitor_impl.h"
#include "interface/compiler_monitor/monitor_util.h"

namespace npu::tile_fwk {
namespace {
constexpr int kPassTimeoutBaseOpSize = 200000;
constexpr double kSinglePassTimeoutBaseSec = 20.0;
constexpr double kPassStageTimeoutBaseSec = 90.0;
constexpr double kMillisecondsPerSecond = 1000.0;
constexpr int kPassDurationPrecision = 3;

double CalcPassTimeoutSec(int opSize, double baseTimeoutSec)
{
    if (opSize < 0) {
        return -1.0;
    }
    int normalizedOpSize = std::max(opSize, 1);
    return baseTimeoutSec * static_cast<double>(normalizedOpSize) / static_cast<double>(kPassTimeoutBaseOpSize);
}

std::string FormatPassDuration(double seconds)
{
    if (seconds < 0.0) {
        return "-1";
    }
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(kPassDurationPrecision) << seconds * kMillisecondsPerSecond << "ms";
    return oss.str();
}
} // namespace

MonitorManager::~MonitorManager() { Shutdown(); }

MonitorManager& MonitorManager::Instance()
{
    static MonitorManager instance;
    return instance;
}

double MonitorManager::CalcPassStageTimeoutSec(int op_size)
{
    return CalcPassTimeoutSec(op_size, kPassStageTimeoutBaseSec);
}

std::string MonitorManager::FormatPassDurationForLog(double seconds)
{
    return FormatPassDuration(seconds);
}

void MonitorManager::Initialize(
    bool enable, int interval_sec, double timeout_sec, int total_timeout_sec, bool pass_detail_enable)
{
    std::lock_guard<std::mutex> lock(mutex_);
    this->SetCompilerMonitorOptions(
        enable, interval_sec, timeout_sec, total_timeout_sec, enable && pass_detail_enable);
    if (initialized_) {
        return;
    }
    if (!enable) {
        return;
    }
    impl_ = new MonitorImpl(this);
    current_stage_ = "Prepare";
    total_start_ = std::chrono::steady_clock::now();
    stage_start_ = total_start_;
    next_function_index_ = 1;
    impl_->Start();
    initialized_ = true;
    stage_doing_ = true;
    stage_elapsed_totals_.clear();
    pass_compile_timings_.clear();
    pass_elapsed_totals_.clear();
    last_pass_detail_function_index_ = -1;
    last_pass_detail_function_name_.clear();
    last_pass_detail_strategy_.clear();
    stage_timeout_flag_.clear();
    stage_timeout_flag_["Prepare"] = false;
    stage_timeout_flag_["Pass"] = false;
    stage_timeout_flag_["CodeGen"] = false;
    stage_timeout_flag_[STAGE_FUNC_TO_BIN] = false;
    stage_timeout_flag_["Total"] = false;
    python_stage_ended_ = false;
    stage_elapsed_totals_["Prepare"] = 0.0;
    // Mark that Prepare stage has started (use env var for cross-.so communication)
    (void)setenv("PYPTO_COMPILER_MONITOR_PREPARE_STARTED", "1", 1);
    impl_->StartMonitoring();
}

void MonitorManager::Shutdown()
{
    std::lock_guard<std::mutex> lock(mutex_);
    if (!initialized_ || !enable_) {
        return;
    }
    if (impl_) {
        impl_->Stop();
        delete impl_;
        impl_ = nullptr;
    }
    initialized_ = false;
}

void MonitorManager::MaybeStartTotalClock()
{
    if (current_stage_.empty()) {
        total_start_ = std::chrono::steady_clock::now();
        stage_start_ = total_start_;
    }
}

double MonitorManager::GetCurrentStageElapsed(const std::string& name)
{
    std::lock_guard<std::mutex> lock(mutex_);
    if (!initialized_) {
        return 0.0;
    }
    auto now = std::chrono::steady_clock::now();
    double elapsed = 0.0;
    if (stage_doing_ && name == current_stage_) {
        elapsed = std::chrono::duration<double>(now - stage_start_).count();
    }
    return elapsed;
}

void MonitorManager::SetTotalFunctionCount(int n)
{
    if (!enable_) {
        return;
    }
    MonitorImpl* to_start = nullptr;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        // Check if Prepare stage was started via Python (env var indicates this)
        bool prepare_started = (std::getenv("PYPTO_COMPILER_MONITOR_PREPARE_STARTED") != nullptr);
        if (!initialized_ && enable_ && !prepare_started) {
            // First time initialization (no Prepare stage from Python)
            if (!impl_) {
                impl_ = new MonitorImpl(this);
            }
            current_stage_ = "Prepare";
            total_start_ = std::chrono::steady_clock::now();
            stage_start_ = total_start_;
            stage_elapsed_totals_.clear();
            pass_compile_timings_.clear();
            pass_elapsed_totals_.clear();
            last_pass_detail_function_index_ = -1;
            last_pass_detail_function_name_.clear();
            last_pass_detail_strategy_.clear();
            stage_timeout_flag_.clear();
            python_stage_ended_ = false;
            to_start = impl_;
            initialized_ = true;
        }
        total_function_count_ = n;
        next_function_index_ = 1;
        current_function_index_ = 0;
        (void)setenv("PYPTO_COMPILER_MONITOR_CURRENT", "0", 1); // 进程内唯一，避免多 .so 多单例
    }
    if (to_start != nullptr) {
        to_start->Start();
    }
}

int MonitorManager::GetAndIncrementNextFunctionIndex()
{
    if (!enable_) {
        return 0;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    int k = next_function_index_++;
    return k;
}

void MonitorManager::SetCurrentFunctionIndex(int k)
{
    if (!enable_) {
        return;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    current_function_index_ = k;
    std::string val = std::to_string(k);
    (void)setenv("PYPTO_COMPILER_MONITOR_CURRENT", val.c_str(), 1);
}

void MonitorManager::SetRootFuncCount(int n)
{
    if (!enable_) {
        return;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    root_func_count_ = n;
    next_root_func_index_ = 1;
    current_root_func_index_ = 0;
}

int MonitorManager::PrepareNextRootFunc()
{
    if (!enable_) {
        return 0;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    int k = next_root_func_index_++;
    return k;
}

std::string MonitorManager::GetCurrentRootFuncName() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return current_root_func_;
}

int MonitorManager::GetRootFuncCount() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return root_func_count_;
}

int MonitorManager::GetCurrentRootFuncIndex() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return current_root_func_index_;
}

void MonitorManager::TryEndPrepareStage()
{
    if (!impl_ || !enable_) {
        return;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    impl_->StopMonitoring();
    if (!initialized_ || python_stage_ended_) {
        return;
    }
    python_stage_ended_ = true;
    auto now = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration<double>(now - stage_start_).count();
    stage_elapsed_totals_["Prepare"] += elapsed;

    if (current_stage_ == "Prepare" && enable_) {
        double total_elapsed_prepare =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - total_start_).count();
        std::string msg = "[Compiler Monitor] Stage: " + current_stage_ +
                          "(completed) | Stashed function: " + std::to_string(total_function_count_) +
                          " | Stage elapsed: " + FormatElapsed(elapsed) +
                          " | Total elapsed: " + FormatElapsed(total_elapsed_prepare);
        COMPILER_LOGI(
            "%s. current thread_id:%s pid:%ld ppid:%ld", msg.c_str(),
            []() {
                std::stringstream ss;
                ss << std::this_thread::get_id();
                return ss.str();
            }()
                .c_str(),
            static_cast<long>(getpid()), static_cast<long>(getppid()));
        (void)fprintf(stdout, "%s\n", msg.c_str());
        (void)fflush(stdout);
    }
}

void MonitorManager::NotifyCompilationFinished()
{
    MonitorImpl* impl_to_stop = nullptr;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!initialized_ || !enable_) {
            return;
        }
        PrintCompilationFinished();
        if (impl_) {
            impl_to_stop = impl_;
            impl_ = nullptr;
        }
        initialized_ = false;
        stage_doing_ = false;
    }
    if (impl_to_stop) {
        impl_to_stop->Stop();
        delete impl_to_stop;
    }
}

void MonitorManager::PrintCompilationFinished()
{
    if (enable_) {
        auto now = std::chrono::steady_clock::now();
        double total_elapsed = std::chrono::duration<double>(now - total_start_).count();

        // Calculate total from all stage elapsed totals (sum of all stages)
        // This ensures Total elapsed includes Prepare time from Python side
        double stage_total = 0.0;
        for (const auto& kv : stage_elapsed_totals_) {
            stage_total += kv.second;
        }
        // Use the larger of: clock-based total vs sum of stages
        // (sum of stages may be more accurate when Prepare was tracked via Python)
        if (stage_total > total_elapsed) {
            total_elapsed = stage_total;
        }

        std::string compilation_msg =
            "[Compiler Monitor] Compilation finished " + std::to_string(current_function_index_) + "/" +
            std::to_string(total_function_count_ > 0 ? total_function_count_ : 1) +
            " | Total functions: " + std::to_string(total_function_count_ > 0 ? total_function_count_ : 1);
        (void)fprintf(stdout, "%s\n", compilation_msg.c_str());
        (void)fflush(stdout);
        COMPILER_LOGI("%s", compilation_msg.c_str());

        int n = total_function_count_ > 0 ? total_function_count_ : 1;
        std::ostringstream stage_msg;
        for (const auto& [stage, sec] : stage_elapsed_totals_) {
            if (stage == "Pass" || stage == "CodeGen") {
                stage_msg << " " << ("[" + stage + "]:") << std::fixed << std::setprecision(1) << sec << "s"
                          << " ";
            } else {
                stage_msg << " " << ("[" + stage + "]:") << std::fixed << std::setprecision(1) << sec << "s  (sum over "
                          << n << " functions)\n";
            }
        }
        std::string stageTimingMsg = "[Compiler Monitor] Stage timing (aggregated by stage):" + stage_msg.str();
        COMPILER_LOGI("%s", stageTimingMsg.c_str());
        (void)fprintf(stdout, "%s\n", stageTimingMsg.c_str());
        (void)fflush(stdout);

        std::string final_msg =
            "[Compiler Monitor] Monitoring stopped | Total elapsed: " + FormatElapsed(total_elapsed);
        COMPILER_LOGI("%s", final_msg.c_str());
        (void)fprintf(stdout, "%s\n", final_msg.c_str());
        (void)fflush(stdout);

        // Save to member variable for GetTotalElapsed() to access
        last_total_elapsed_ = total_elapsed;
    }
}

std::string MonitorManager::BuildPassCompileTimingsForFunction(int functionIndex, const std::string& functionName) const
{
    if (pass_compile_timings_.empty()) {
        return "";
    }

    std::vector<PassCompileTiming> timings;
    for (const auto& timing : pass_compile_timings_) {
        if (timing.functionIndex == functionIndex && timing.functionName == functionName) {
            timings.push_back(timing);
        }
    }
    if (timings.empty()) {
        return "";
    }

    std::sort(timings.begin(), timings.end(), [](const PassCompileTiming& lhs, const PassCompileTiming& rhs) {
        if (lhs.strategy != rhs.strategy) {
            return lhs.strategy < rhs.strategy;
        }
        return lhs.passIndex < rhs.passIndex;
    });

    std::ostringstream detailMsg;
    detailMsg << BuildPassFunctionHeaderLocked(functionIndex, functionName);
    std::string lastStrategy;
    for (const auto& timing : timings) {
        if (timing.strategy != lastStrategy) {
            detailMsg << "  |__ [Compiler Monitor] Strategy:[" << timing.strategy << "]\n";
            lastStrategy = timing.strategy;
        }
        double passTimeoutSec = CalcPassTimeoutSec(timing.functionOpSize, kSinglePassTimeoutBaseSec);
        bool passTimeoutEnabled = passTimeoutSec >= 0.0;
        bool passTimedOut = passTimeoutEnabled && timing.elapsedSec > passTimeoutSec;
        detailMsg << "  |__   [" << std::setw(2) << std::setfill('0') << timing.passIndex << std::setfill(' ') << "] "
                  << timing.passIdentifier << " | Pass elapsed: " << FormatPassDuration(timing.elapsedSec);
        if (passTimeoutEnabled) {
            detailMsg << " | Threshold: " << FormatPassDuration(passTimeoutSec);
        }
        detailMsg << (passTimedOut ? " | [** WARNING **] single pass timeout" : "")
                  << (timing.success ? "" : " | Result:[FAILED]") << "\n";
    }

    return detailMsg.str();
}

std::string MonitorManager::BuildPassFunctionHeaderLocked(
    int functionIndex, const std::string& functionName, const std::string& strategy) const
{
    int totalFunctions = total_function_count_ > 0 ? total_function_count_ : 1;
    std::ostringstream oss;
    oss << "  |__ [Compiler Monitor] Function: " << functionIndex << "/" << totalFunctions
        << " | Pass timing detail";
    if (!strategy.empty()) {
        oss << " | Strategy:[" << strategy << "]";
    }
    oss << " | Func:[" << functionName << "]\n";
    return oss.str();
}

std::string MonitorManager::BuildPassProgressLineLocked(
    const std::string& passIdentifier, size_t passIndex, int functionOpSize, const std::string& status,
    double elapsedSec, bool success) const
{
    double passTimeoutSec = CalcPassTimeoutSec(functionOpSize, kSinglePassTimeoutBaseSec);
    bool passTimeoutEnabled = passTimeoutSec >= 0.0;
    bool passTimedOut = passTimeoutEnabled && elapsedSec > passTimeoutSec;

    std::ostringstream oss;
    oss << "  |__ [Compiler Monitor] [" << std::setw(2) << std::setfill('0') << passIndex << std::setfill(' ') << "] "
        << passIdentifier;
    if (status == "running") {
        oss << " start";
    } else {
        oss << " | Pass elapsed: " << FormatPassDuration(elapsedSec);
    }
    if (status != "running" && passTimeoutEnabled) {
        oss << " | Threshold: " << FormatPassDuration(passTimeoutSec);
    }
    if (passTimedOut) {
        oss << " | [** WARNING **] single pass timeout";
    }
    if (!success) {
        oss << " | Result:[FAILED]";
    }
    oss << "\n";
    return oss.str();
}

void MonitorManager::SetCompilerMonitorOptions(
    bool enable, int interval_sec, double timeout_sec, int total_timeout_sec, bool pass_detail_enable)
{
    enable_ = enable;
    pass_detail_enable_ = pass_detail_enable;
    interval_sec_.store((interval_sec > 0) ? interval_sec : 60);
    std::string interval_str = std::to_string(interval_sec_.load());
    (void)setenv("PYPTO_COMPILER_MONITOR_INTERVAL_SEC", interval_str.c_str(), 1);
    timeout_sec_.store((timeout_sec >= -1.0) ? timeout_sec : 0.0);
    std::string timeout_str = std::to_string(timeout_sec_.load());
    (void)setenv("PYPTO_COMPILER_MONITOR_TIMEOUT_SEC", timeout_str.c_str(), 1);
    total_timeout_sec_.store((total_timeout_sec >= 0) ? total_timeout_sec : 600);
    std::string total_timeout_str = std::to_string(total_timeout_sec_.load());
    (void)setenv("PYPTO_COMPILER_MONITOR_TOTAL_TIMEOUT_SEC", total_timeout_str.c_str(), 1);
}

bool MonitorManager::IsEnabled() const { return enable_; }

bool MonitorManager::IsPassDetailEnabled() const { return pass_detail_enable_; }

void MonitorManager::SetStageTimeoutFlag(const std::string& name)
{
    std::lock_guard<std::mutex> lock(mutex_);
    if (stage_timeout_flag_.count(name) > 0) {
        stage_timeout_flag_[name] = true;
    }
}

bool MonitorManager::GetStageTimeoutFlag(const std::string& name)
{
    std::lock_guard<std::mutex> lock(mutex_);
    if (stage_timeout_flag_.count(name) > 0) {
        return stage_timeout_flag_[name];
    }
    return false;
}

void MonitorManager::SetActiveStageWarningPrinted(const std::string& name, int rootFuncIndex)
{
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto& info : active_stages_) {
        if (info.stageName == name && info.rootFuncIndex == rootFuncIndex) {
            info.warningPrinted = true;
            break;
        }
    }
}

int MonitorManager::GetIntervalSec() const { return interval_sec_.load(); }

double MonitorManager::GetTimeoutSec() const { return timeout_sec_.load(); }

int MonitorManager::GetTotalTimeoutSec() const { return total_timeout_sec_.load(); }

std::string MonitorManager::GetCurrentStageName() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return current_stage_;
}

void MonitorManager::SwitchStageReset()
{
    this->SetCurrentFuncOpSize(0);
    this->SetFuncSumOpSize(0, true);
    this->SetCurrentFunctionName("");
}

std::string MonitorManager::GetCurrentFunctionName() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return current_function_;
}

void MonitorManager::SetCurrentFunctionName(const std::string& name)
{
    if (!enable_) {
        return;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    current_function_ = name;
}

int MonitorManager::GetCurrentFuncOpSize() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return current_func_opsize_;
}

void MonitorManager::SetCurrentFuncOpSize(int op_size, bool update_active_stage)
{
    if (!enable_) {
        return;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    current_func_opsize_ = op_size;
    if (!update_active_stage) {
        return;
    }
    for (auto& info : active_stages_) {
        if (info.stageName == "Pass" && info.functionIndex == current_function_index_ &&
            info.functionName == current_function_) {
            info.functionOpSize = op_size;
        }
    }
}

void MonitorManager::PrintCurrentTotalElapsed(std::string str_temp)
{
    if (!enable_) {
        return;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    auto now = std::chrono::steady_clock::now();
    double total_elapsed = std::chrono::duration<double>(now - total_start_).count();
    std::string stage_finish_msg =
        "[Compiler Monitor] " + str_temp + " | Total elapsed: " + FormatElapsed(total_elapsed);
    (void)fprintf(stdout, "%s\n", stage_finish_msg.c_str());
    (void)fflush(stdout);
}

int MonitorManager::GetFuncSumOpSize() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return func_sum_opsize_;
}

void MonitorManager::SetFuncSumOpSize(size_t op_size, bool reset)
{
    if (!enable_) {
        return;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    func_sum_opsize_ += static_cast<int>(op_size);
    if (reset) {
        func_sum_opsize_ = 0;
    }
}

std::chrono::steady_clock::time_point MonitorManager::GetStageStartTime() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return stage_start_;
}

std::chrono::steady_clock::time_point MonitorManager::GetTotalStartTime() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return total_start_;
}

int MonitorManager::GetTotalFunctionCount() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return total_function_count_;
}

int MonitorManager::GetCurrentFunctionIndex() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return current_function_index_;
}

std::unordered_map<std::string, double> MonitorManager::GetStageElapsedTotals() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return stage_elapsed_totals_;
}

void MonitorManager::RecordPassCompileTime(
    const std::string& strategy, const std::string& passIdentifier, size_t passIndex,
    const std::string& functionName, int functionIndex, int functionOpSize, double elapsedSec, bool success)
{
    std::string progressLine;
    bool passTimedOut = false;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!enable_ || !initialized_) {
            return;
        }
        PassCompileTiming timing{
            functionIndex, functionName, functionOpSize, strategy, passIdentifier, passIndex, elapsedSec, success};
        pass_compile_timings_.push_back(timing);
        pass_elapsed_totals_[strategy + "::" + passIdentifier] += elapsedSec;
        if (pass_detail_enable_ && current_stage_ == STAGE_PASS) {
            progressLine = BuildPassProgressLineLocked(
                passIdentifier, passIndex, functionOpSize, "completed", elapsedSec, success);
            double passTimeoutSec = CalcPassTimeoutSec(functionOpSize, kSinglePassTimeoutBaseSec);
            passTimedOut = passTimeoutSec >= 0.0 && elapsedSec > passTimeoutSec;
        }
    }
    if (!progressLine.empty()) {
        std::lock_guard<std::mutex> printLock(pass_detail_print_mutex_);
        (void)fprintf(stdout, "%s", progressLine.c_str());
        (void)fflush(stdout);
        if (passTimedOut) {
            COMPILER_LOGW("%s", progressLine.c_str());
        } else {
            COMPILER_LOGI("%s", progressLine.c_str());
        }
    }
}

std::vector<PassCompileTiming> MonitorManager::GetPassCompileTimings() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return pass_compile_timings_;
}

std::map<std::string, double> MonitorManager::GetPassElapsedTotals() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return pass_elapsed_totals_;
}

void MonitorManager::StartPassCompile(
    const std::string& strategy, const std::string& passIdentifier, size_t passIndex,
    const std::string& functionName, int functionIndex, int functionOpSize)
{
    std::string progressLine;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!enable_ || !initialized_) {
            return;
        }
        current_pass_.active = true;
        current_pass_.functionIndex = functionIndex;
        current_pass_.functionName = functionName;
        current_pass_.functionOpSize = functionOpSize;
        current_pass_.strategy = strategy;
        current_pass_.passIdentifier = passIdentifier;
        current_pass_.passIndex = passIndex;
        current_pass_.startTime = std::chrono::steady_clock::now();
        if (pass_detail_enable_ && current_stage_ == STAGE_PASS) {
            if (last_pass_detail_function_index_ != functionIndex ||
                last_pass_detail_function_name_ != functionName || last_pass_detail_strategy_ != strategy) {
                progressLine += BuildPassFunctionHeaderLocked(functionIndex, functionName, strategy);
                last_pass_detail_function_index_ = functionIndex;
                last_pass_detail_function_name_ = functionName;
                last_pass_detail_strategy_ = strategy;
            }
            progressLine += BuildPassProgressLineLocked(
                passIdentifier, passIndex, functionOpSize, "running", 0.0, true);
        }
    }
    if (!progressLine.empty()) {
        std::lock_guard<std::mutex> printLock(pass_detail_print_mutex_);
        (void)fprintf(stdout, "%s", progressLine.c_str());
        (void)fflush(stdout);
        COMPILER_LOGI("%s", progressLine.c_str());
    }
}

void MonitorManager::EndPassCompile(
    const std::string& strategy, const std::string& passIdentifier, size_t passIndex,
    const std::string& functionName, int functionIndex)
{
    std::lock_guard<std::mutex> lock(mutex_);
    if (!current_pass_.active) {
        return;
    }
    if (current_pass_.strategy == strategy && current_pass_.passIdentifier == passIdentifier &&
        current_pass_.passIndex == passIndex && current_pass_.functionName == functionName &&
        current_pass_.functionIndex == functionIndex) {
        current_pass_ = CurrentPassInfo{};
    }
}

std::string MonitorManager::GetCurrentPassDescription() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    if (!current_pass_.active) {
        return "";
    }
    auto elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - current_pass_.startTime).count();
    std::ostringstream oss;
    oss << " | Current pass:[" << current_pass_.passIdentifier << "]"
        << " | Pass idx:[" << current_pass_.passIndex << "]"
        << " | Pass elapsed:[" << FormatPassDuration(elapsed) << "]"
        << " | Strategy:[" << current_pass_.strategy << "]";
    return oss.str();
}

void MonitorManager::StartStage(const std::string& name, int rootFuncIndex, const std::string& rootFuncName,
                                int rootFuncOpSize)
{
    COMPILER_LOGI("Stage ==[%s]== begin.", name.c_str());
    std::lock_guard<std::mutex> lock(mutex_);
    if (!initialized_ || !impl_ || !enable_) {
        return;
    }

    // 当启动 FuncToBin 时，移除 active_stages_ 中的 CodeGen
    if (name == STAGE_FUNC_TO_BIN) {
        auto it = std::find_if(active_stages_.begin(), active_stages_.end(),
            [](const ActiveStageInfo& info) { return info.stageName == "CodeGen"; });
        if (it != active_stages_.end()) {
            active_stages_.erase(it);
        }
    }

    if (rootFuncIndex == -1) {
        impl_->StartMonitoring();
    }
    MaybeStartTotalClock();
    current_stage_ = name;
    stage_start_ = std::chrono::steady_clock::now();
    stage_doing_ = true;

    ActiveStageInfo info;
    info.stageName = name;
    info.startTime = stage_start_;
    info.functionIndex = current_function_index_;
    info.functionName = current_function_;
    info.functionOpSize = current_func_opsize_;
    info.rootFuncIndex = (rootFuncIndex < 0) ? current_root_func_index_ : rootFuncIndex;
    info.rootFuncName = (rootFuncIndex < 0) ? current_root_func_ : rootFuncName;
    info.rootFuncOpSize = rootFuncOpSize;
    active_stages_.push_back(info);
}

void MonitorManager::EndStage(const std::string& name, int rootFuncIndex, const std::string& rootFuncName,
                              int rootFuncOpSize)
{
    std::lock_guard<std::mutex> lock(mutex_);
    int actualRootFuncIndex = rootFuncIndex;
    std::string actualRootFuncName = rootFuncName;
    auto stageStartTime = stage_start_;
    int actualFunctionIndex = current_function_index_;
    std::string actualFunctionName = current_function_;
    int actualFunctionOpSize = current_func_opsize_;

    auto it = active_stages_.rend();
    if (rootFuncIndex < 0) {
        it = std::find_if(active_stages_.rbegin(), active_stages_.rend(), [&name](const ActiveStageInfo& info) {
            return info.stageName == name;
        });
        if (it != active_stages_.rend()) {
            actualRootFuncIndex = it->rootFuncIndex;
            actualRootFuncName = it->rootFuncName;
            stageStartTime = it->startTime;
            actualFunctionIndex = it->functionIndex;
            actualFunctionName = it->functionName;
            actualFunctionOpSize = it->functionOpSize;
        } else {
            actualRootFuncIndex = current_root_func_index_;
            actualRootFuncName = current_root_func_;
        }
    } else {
        it = std::find_if(active_stages_.rbegin(), active_stages_.rend(), [&](const ActiveStageInfo& info) {
            return info.stageName == name && info.rootFuncIndex == rootFuncIndex;
        });
        if (it != active_stages_.rend()) {
            stageStartTime = it->startTime;
            actualFunctionIndex = it->functionIndex;
            actualFunctionName = it->functionName;
            actualFunctionOpSize = it->functionOpSize;
        }
    }

    if (it != active_stages_.rend()) {
        active_stages_.erase(std::prev(it.base()));
    }
    EndStageInternal(
        name, actualRootFuncIndex, actualRootFuncName, stageStartTime, rootFuncIndex, rootFuncOpSize,
        actualFunctionIndex, actualFunctionName, actualFunctionOpSize);
}

void MonitorManager::EndStageInternal(
    const std::string& name, int rootFuncIndex, const std::string& rootFuncName,
    const std::chrono::steady_clock::time_point& startTime, int rootFuncIndexOriginal, int rootFuncOpSize,
    int functionIndex, const std::string& functionName, int functionOpSize)
{
    if (!initialized_ || !impl_ || !enable_) {
        return;
    }
    bool stageWarningPrinted = false;
    auto timeoutFlagIt = stage_timeout_flag_.find(name);
    if (timeoutFlagIt != stage_timeout_flag_.end()) {
        stageWarningPrinted = timeoutFlagIt->second;
    }
    if (rootFuncIndexOriginal == -1) {
        impl_->StopMonitoring();
    }
    auto now = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration<double>(now - startTime).count();
    if (name != STAGE_FUNC_TO_BIN) {
        stage_elapsed_totals_[name] += elapsed;
    }
    stage_doing_ = false;
    COMPILER_LOGI("Stage ==[%s]== end, sub stage cost %lfs.", name.c_str(), stage_elapsed_totals_[name]);

    double total_elapsed = std::chrono::duration<double>(now - total_start_).count();

    std::string stage_finish_msg;
    if (name == STAGE_FUNC_TO_BIN) {
        int pw = GetProgressWidth();
        stage_finish_msg = "[Compiler Monitor] " + PadLabel("Function(parallel): ") +
                           PadRight(std::to_string(rootFuncIndex) + "/" + std::to_string(root_func_count_), pw) +
                           " | Stage: " + PadStageName("CodeGen[" + name + "]") +
                           "(completed) | Stage elapsed: " + PadElapsed(FormatElapsed(elapsed)) +
                           " | Total elapsed: " + PadElapsed(FormatElapsed(total_elapsed)) + " | Func:[" +
                           rootFuncName + "] Ops: " + std::to_string(rootFuncOpSize);
    } else if (name == "CodeGen") {
        stage_finish_msg = "[Compiler Monitor] Stage: " + name +
                           "(completed) | Stage elapsed: " + PadElapsed(FormatElapsed(elapsed)) +
                           " | Total elapsed: " + PadElapsed(FormatElapsed(total_elapsed));
    } else {
        int pw = GetProgressWidth();
        stage_finish_msg =
            "[Compiler Monitor] " + PadLabel("Function: ") +
            PadRight(std::to_string(functionIndex) + "/" + std::to_string(total_function_count_), pw) +
            " | Stage: " + PadStageName(name) + "(completed) | Stage elapsed: " + PadElapsed(FormatElapsed(elapsed)) +
            " | Total elapsed: " + PadElapsed(FormatElapsed(total_elapsed)) + " | Func:[" + functionName +
            "] Ops: " + std::to_string(functionOpSize);
    }

    (void)fprintf(stdout, "%s\n", stage_finish_msg.c_str());
    (void)fflush(stdout);
    COMPILER_LOGI("%s", stage_finish_msg.c_str());
    if (name == "Pass") {
        double timeoutSec = CalcPassTimeoutSec(functionOpSize, kPassStageTimeoutBaseSec);
        if (timeoutSec >= 0.0 && elapsed > timeoutSec && !stageWarningPrinted) {
            int pw = GetProgressWidth();
            std::string passTimeoutWarnMsg =
                "[Compiler Monitor] | [** WARNING **] Function: " +
                PadRight(std::to_string(functionIndex) + "/" + std::to_string(total_function_count_), pw) +
                " | Stage [Pass] elapsed [" + FormatPassDuration(elapsed) +
                "] exceeded the pass stage time threshold [" + FormatPassDuration(timeoutSec) +
                "] | Func:[" + functionName + "] | Number of op: " + std::to_string(functionOpSize) +
                " | Standard: 200000 ops / 90.0s linear scaled" +
                ", you can terminate the process by pressing Ctrl+C !!!";
            stage_timeout_flag_[name] = true;
            (void)fprintf(stdout, "%s\n", passTimeoutWarnMsg.c_str());
            (void)fflush(stdout);
            COMPILER_LOGW("%s", passTimeoutWarnMsg.c_str());
        }
    }
    if (timeout_sec_.load() < 0.0 || timeout_sec_.load() > 0.0) {
        stage_timeout_flag_[name] = false;
    }
}

std::vector<ActiveStageInfo> MonitorManager::GetActiveStages() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return active_stages_;
}

int MonitorManager::GetProcessingThresholdSec() const { return processing_threshold_sec_; }

void MonitorManager::SetProcessingThresholdSec(int sec) { processing_threshold_sec_ = sec; }

int MonitorManager::GetProgressWidth() const
{
    auto digits = [](int n) { return static_cast<int>(std::to_string(std::max(n, 1)).size()); };
    int maxDigits = std::max(digits(total_function_count_), digits(root_func_count_));
    return 2 * maxDigits + 1;
}

double MonitorManager::GetTotalElapsed() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return last_total_elapsed_;
}

} // namespace npu::tile_fwk
