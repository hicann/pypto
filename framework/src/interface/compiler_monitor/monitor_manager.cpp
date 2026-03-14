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

MonitorManager::~MonitorManager() {
    Shutdown();
}

MonitorManager &MonitorManager::Instance() {
    static MonitorManager instance;
    return instance;
}

void MonitorManager::Initialize(bool enable, int interval_sec, int timeout_sec, int total_timeout_sec) {
    std::lock_guard<std::mutex> lock(mutex_);
    this->SetCompilerMonitorOptions(enable, interval_sec, timeout_sec, total_timeout_sec);
    if (initialized_) {
        return;
    }
    if (!enable_) {
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
    stage_timeout_flag_.clear();
    stage_timeout_flag_["Prepare"] = false;
    stage_timeout_flag_["Pass"] = false;
    stage_timeout_flag_["CodeGen"] = false;
    stage_timeout_flag_["Total"] = false;
    python_stage_ended_ = false;
    stage_elapsed_totals_["Prepare"] = 0.0;
    // Mark that Prepare stage has started (use env var for cross-.so communication)
    (void)setenv("PYPTO_COMPILER_MONITOR_PREPARE_STARTED", "1", 1);
    impl_->StartMonitoring();
}

void MonitorManager::Shutdown() {
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

void MonitorManager::MaybeStartTotalClock() {
    if (current_stage_.empty()) {
        total_start_ = std::chrono::steady_clock::now();
        stage_start_ = total_start_;
    }
}

double MonitorManager::GetCurrentStageElapsed(const std::string &name) {
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

void MonitorManager::SetTotalFunctionCount(int n) {
    if (!enable_) {
        return;
    }
    MonitorImpl *to_start = nullptr;
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

int MonitorManager::GetAndIncrementNextFunctionIndex() {
    if (!enable_) {
        return 0;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    int k = next_function_index_++;
    return k;
}

void MonitorManager::SetCurrentFunctionIndex(int k) {
    if (!enable_) {
        return;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    current_function_index_ = k;
    std::string val = std::to_string(k);
    (void)setenv("PYPTO_COMPILER_MONITOR_CURRENT", val.c_str(), 1);
}

void MonitorManager::TryEndPrepareStage() {
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

void MonitorManager::NotifyCompilationFinished() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!initialized_ || !enable_) {
        return;
    }
    PrintCompilationFinished();
    if (impl_) {
        impl_->Stop();
        delete impl_;
        impl_ = nullptr;
    }
    initialized_ = false;
    stage_doing_ = false;
}

void MonitorManager::PrintCompilationFinished() {
    if (enable_) {
        auto now = std::chrono::steady_clock::now();
        double total_elapsed = std::chrono::duration<double>(now - total_start_).count();

        // Calculate total from all stage elapsed totals (sum of all stages)
        // This ensures Total elapsed includes Prepare time from Python side
        double stage_total = 0.0;
        for (const auto &kv : stage_elapsed_totals_) {
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
            " | Total functions: " + std::to_string(total_function_count_ > 0 ? total_function_count_ : 1) +
            "\n[Compiler Monitor] Stage timing (aggregated by stage):";
        (void)fprintf(stdout, "%s\n", compilation_msg.c_str());
        (void)fflush(stdout);
        COMPILER_LOGI("%s", compilation_msg.c_str());

        int n = total_function_count_ > 0 ? total_function_count_ : 1;
        for (const auto &[stage, sec] : stage_elapsed_totals_) {
            std::ostringstream stage_msg;
            if (stage == "Pass" || stage == "CodeGen") {
                stage_msg << "  " << std::left << std::setw(8) << stage << " " << std::fixed << std::setprecision(1)
                          << sec << "s"
                          << "   (sum over " << n << " functions)\n";
            } else {
                stage_msg << "  " << std::left << std::setw(8) << stage << " " << std::fixed << std::setprecision(1)
                          << sec << "s\n";
            }
            COMPILER_LOGI("%s", stage_msg.str().c_str());
            (void)fprintf(stdout, "%s", stage_msg.str().c_str());
            (void)fflush(stdout);
        }

        std::string final_msg =
            "[Compiler Monitor] Monitoring stopped | Total elapsed: " + FormatElapsed(total_elapsed);
        COMPILER_LOGI("%s", final_msg.c_str());
        (void)fprintf(stdout, "%s\n", final_msg.c_str());
        (void)fflush(stdout);

        // Save to member variable for GetTotalElapsed() to access
        last_total_elapsed_ = total_elapsed;
    }
}

void MonitorManager::SetCompilerMonitorOptions(bool enable, int interval_sec, int timeout_sec, int total_timeout_sec) {
    enable_ = enable;
    interval_sec_ = (interval_sec > 0) ? interval_sec : 60;
    std::string interval_str = std::to_string(interval_sec_);
    (void)setenv("PYPTO_COMPILER_MONITOR_INTERVAL_SEC", interval_str.c_str(), 1);
    timeout_sec_ = (timeout_sec >= -1) ? timeout_sec : 0;
    std::string timeout_str = std::to_string(timeout_sec_);
    (void)setenv("PYPTO_COMPILER_MONITOR_TIMEOUT_SEC", timeout_str.c_str(), 1);
    total_timeout_sec_ = (total_timeout_sec >= 0) ? total_timeout_sec : 600;
    std::string total_timeout_str = std::to_string(total_timeout_sec_);
    (void)setenv("PYPTO_COMPILER_MONITOR_TOTAL_TIMEOUT_SEC", total_timeout_str.c_str(), 1);
}

bool MonitorManager::IsEnabled() const {
    return enable_;
}

void MonitorManager::SetStageTimeoutFlag(const std::string &name) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (stage_timeout_flag_.count(name) > 0) {
        stage_timeout_flag_[name] = true;
    }
}

bool MonitorManager::GetStageTimeoutFlag(const std::string &name) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (stage_timeout_flag_.count(name) > 0) {
        return stage_timeout_flag_[name];
    }
    return false;
}

int MonitorManager::GetIntervalSec() const {
    const char *v = std::getenv("PYPTO_COMPILER_MONITOR_INTERVAL_SEC");
    if (v != nullptr) {
        int env_val = std::atoi(v);
        if (env_val > 0) {
            return env_val;
        }
    }
    return interval_sec_;
}

int MonitorManager::GetTimeoutSec() const {
    const char *v = std::getenv("PYPTO_COMPILER_MONITOR_TIMEOUT_SEC");
    if (v != nullptr) {
        int env_val = std::atoi(v);
        if (env_val > 0) {
            return env_val;
        }
    }
    return timeout_sec_;
}

int MonitorManager::GetTotalTimeoutSec() const {
    const char *v = std::getenv("PYPTO_COMPILER_MONITOR_TOTAL_TIMEOUT_SEC");
    if (v != nullptr) {
        int env_val = std::atoi(v);
        if (env_val > 0) {
            return env_val;
        }
    }
    return total_timeout_sec_;
}

std::string MonitorManager::GetCurrentStageName() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return current_stage_;
}

std::string MonitorManager::GetCurrentFunctionName() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return current_function_;
}

void MonitorManager::SetCurrentFunctionName(const std::string &name) {
    if (!enable_) {
        return;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    current_function_ = name;
}

std::chrono::steady_clock::time_point MonitorManager::GetStageStartTime() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return stage_start_;
}

std::chrono::steady_clock::time_point MonitorManager::GetTotalStartTime() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return total_start_;
}

int MonitorManager::GetTotalFunctionCount() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return total_function_count_;
}

int MonitorManager::GetCurrentFunctionIndex() const {
    const char *v = std::getenv("PYPTO_COMPILER_MONITOR_CURRENT");
    if (v != nullptr) {
        int env_val = std::atoi(v);
        if (env_val >= 0) {
            return env_val;
        }
    }
    std::lock_guard<std::mutex> lock(mutex_);
    return current_function_index_;
}

std::unordered_map<std::string, double> MonitorManager::GetStageElapsedTotals() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return stage_elapsed_totals_;
}

void MonitorManager::StartStage(const std::string &name) {
    COMPILER_LOGI("Stage ==[%s]== begin.", name.c_str());
    std::lock_guard<std::mutex> lock(mutex_);
    if (!initialized_ || !impl_ || !enable_) {
        return;
    }
    impl_->StartMonitoring();
    MaybeStartTotalClock();
    current_stage_ = name;
    stage_start_ = std::chrono::steady_clock::now();
    stage_doing_ = true;
}

void MonitorManager::EndStage(const std::string &name) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!initialized_ || !impl_ || !enable_) {
        return;
    }
    if (this->GetTimeoutSec() != 0) {
        stage_timeout_flag_["Prepare"] = false;
        stage_timeout_flag_["Pass"] = false;
        stage_timeout_flag_["CodeGen"] = false;
    }
    impl_->StopMonitoring();
    auto now = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration<double>(now - stage_start_).count();
    stage_elapsed_totals_[name] += elapsed;
    stage_doing_ = false;
    COMPILER_LOGI("Stage ==[%s]== end, sub stage cost %lfs.", name.c_str(), stage_elapsed_totals_[name]);

    double total_elapsed = std::chrono::duration<double>(now - total_start_).count();

    std::string stage_finish_msg;
    if (name == "CodeGen") {
        stage_finish_msg = "[Compiler Monitor] Stage: " + name +
                           "(completed) | Stage elapsed: " + FormatElapsed(elapsed) +
                           " | Total elapsed: " + FormatElapsed(total_elapsed);
    } else {
        stage_finish_msg = "[Compiler Monitor] Function: " + std::to_string(this->GetCurrentFunctionIndex()) + "/" +
                           std::to_string(total_function_count_) + " | Stage: " + name +
                           "(completed) | Stage elapsed: " + FormatElapsed(elapsed) +
                           " | Total elapsed: " + FormatElapsed(total_elapsed) + " | Func:[" + current_function_ + "]";
    }

    (void)fprintf(stdout, "%s\n", stage_finish_msg.c_str());
    (void)fflush(stdout);
    COMPILER_LOGI("%s", stage_finish_msg.c_str());
}

double MonitorManager::GetTotalElapsed() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return last_total_elapsed_;
}

} // namespace npu::tile_fwk
