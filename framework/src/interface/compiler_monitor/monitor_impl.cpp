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
#include <limits>
#include <sstream>
#include <iostream>
#include <unistd.h>
#include <thread>

#include "tilefwk/pypto_fwk_log.h"
#include "interface/compiler_monitor/monitor_impl.h"
#include "interface/compiler_monitor/monitor_manager.h"
#include "interface/compiler_monitor/monitor_util.h"

namespace npu::tile_fwk {
namespace {
constexpr int kMillisecondsPerSecond = 1000;
constexpr int kPassDetailMonitorIntervalMs = 1;
} // namespace

MonitorImpl::MonitorImpl(MonitorManager* manager) : manager_(manager) {}

MonitorImpl::~MonitorImpl() { Stop(); }

void MonitorImpl::Start()
{
    std::lock_guard<std::mutex> lock(mutex_);
    if (thread_ && thread_->joinable()) {
        return;
    }
    stop_.store(false);
    thread_ = std::make_unique<std::thread>(&MonitorImpl::MonitorLoop, this);
}

void MonitorImpl::Stop()
{
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stop_.store(true);
        stage_start_flag_.store(false);
        cv_.notify_all();
    }
    if (thread_ && thread_->joinable()) {
        thread_->join();
    }
    thread_.reset();
}

bool IsEnabledImmediate(MonitorManager* manager_) { return manager_->IsEnabled(); }

double GetTimeoutSecImmediate(MonitorManager* manager_)
{
    double stage_timeout_sec = manager_->GetTimeoutSec();
    if (stage_timeout_sec <= 0.0) {
        if (stage_timeout_sec >= 0.0) {
            manager_->SetStageTimeoutFlag("Prepare");
            manager_->SetStageTimeoutFlag("Pass");
            manager_->SetStageTimeoutFlag("CodeGen");
            manager_->SetStageTimeoutFlag(STAGE_FUNC_TO_BIN);
        } else {
            stage_timeout_sec = static_cast<double>(std::numeric_limits<int>::max());
        }
    }
    return stage_timeout_sec;
}

int GetTotalTimeoutSecImmediate(MonitorManager* manager_)
{
    int total_timeout_sec = manager_->GetTotalTimeoutSec();
    if (total_timeout_sec <= 0) {
        if (total_timeout_sec < 0) {
            total_timeout_sec = 600;
        } else {
            total_timeout_sec = 0;
            manager_->SetStageTimeoutFlag("Total");
        }
    }
    return total_timeout_sec;
}

int GetIntervalSecImmediate(MonitorManager* manager_)
{
    int interval_sec = manager_->GetIntervalSec();
    if (interval_sec <= 0) {
        interval_sec = 60;
    }
    return interval_sec;
}

void MonitorImpl::StartMonitoring()
{
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stage_start_flag_.store(true);
    }
    cv_.notify_all(); // 唤醒等待的线程
}

void MonitorImpl::StopMonitoring()
{
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stage_start_flag_.store(false);
    }
    cv_.notify_all(); // 唤醒等待的线程
}

void MonitorImpl::PrintTotalTimeOut(double total_elapsed, int total_timeout_sec)
{
    if ((total_elapsed >= total_timeout_sec) && (manager_->GetStageTimeoutFlag("Total") == false)) {
        int current_total_opsize = manager_->GetFuncSumOpSize();
        manager_->SetStageTimeoutFlag("Total");
        std::string warn_msg;
        warn_msg = "[Compiler Monitor] | [== WARNING ==] Total elapsed [" + FormatElapsed(total_elapsed) +
                   "] exceeded the total time threshold [" + FormatElapsed(static_cast<double>(total_timeout_sec)) +
                   "] | Total number of op: " + std::to_string(current_total_opsize) +
                   ", you can terminate the process by pressing Ctrl+C !!!";
        COMPILER_LOGI("%s", warn_msg.c_str());
        (void)fprintf(stdout, "%s\n", warn_msg.c_str());
        (void)fflush(stdout);
    }
}

void MonitorImpl::MonitorLoop()
{
    bool check_enable = IsEnabledImmediate(manager_);
    int print_interval_sec = GetIntervalSecImmediate(manager_);
    double stage_timeout_sec = GetTimeoutSecImmediate(manager_);
    int total_timeout_sec = GetTotalTimeoutSecImmediate(manager_);

    COMPILER_LOGI(
        "[Compiler Monitor] interval_sec=%d, stage_timeout_sec=%.3f, total_timeout_sec=%d, check_enable=%d",
        print_interval_sec, stage_timeout_sec, total_timeout_sec, check_enable);

    int pre_cost = manager_->GetProcessingThresholdSec();
    int check_interval_sec = 1;
    int last_print_time = 0;

    while (!stop_.load()) {
        // 检查 start_flag，如果为 false 则等待
        if (!stage_start_flag_.load()) {
            std::unique_lock<std::mutex> lock(mutex_);
            // 等待直到 start_flag 变为 true 或 stop_ 变为 true
            cv_.wait(lock, [this] { return stage_start_flag_.load() || stop_.load(); });
            if (stage_start_flag_.load()) {
                last_print_time = 0;
            }
            if (stop_.load()) {
                break;
            }
            lock.unlock();
        }

        // 检查是否超过600秒总超时
        auto now = std::chrono::steady_clock::now();
        auto total_start = manager_->GetTotalStartTime();
        auto total_elapsed = std::chrono::duration<double>(now - total_start).count();
        int current_func_opsize = manager_->GetCurrentFuncOpSize();

        // 当总时间超过total_timeout_sec
        PrintTotalTimeOut(total_elapsed, total_timeout_sec);

        auto wait_duration = std::chrono::milliseconds(check_interval_sec * kMillisecondsPerSecond);
        if (manager_->GetCurrentStageName() == STAGE_PASS && manager_->IsPassDetailEnabled()) {
            wait_duration = std::chrono::milliseconds(kPassDetailMonitorIntervalMs);
        }
        std::unique_lock<std::mutex> lock(mutex_);

        // 修改等待条件：检查 stop_ 和 start_flag
        cv_.wait_for(lock, wait_duration, [this] { return stop_.load() || !stage_start_flag_.load(); });
        if (stop_.load()) {
            break;
        }
        lock.unlock();

        // 如果 start_flag 变为 false，则回到循环开头重新等待
        if (!stage_start_flag_.load()) {
            continue;
        }

        if (!check_enable) {
            continue;
        }

        auto activeStages = manager_->GetActiveStages();
        if (activeStages.empty()) {
            continue;
        }

        now = std::chrono::steady_clock::now();
        total_start = manager_->GetTotalStartTime();
        total_elapsed = std::chrono::duration<double>(now - total_start).count();

        PrintTotalTimeOut(total_elapsed, total_timeout_sec);

        std::string warn_msg;
        std::string interval_msg;

        for (const auto& stageInfo : activeStages) {
            total_elapsed = std::chrono::duration<double>(now - total_start).count();
            PrintTotalTimeOut(total_elapsed, total_timeout_sec);
            const std::string& stage = stageInfo.stageName;
            double curr_stage_elapsed = std::chrono::duration<double>(now - stageInfo.startTime).count();
            std::string current_func_name = manager_->GetCurrentFunctionName();
            std::string current_pass_desc = manager_->GetCurrentPassDescription();

            if (stage == STAGE_FUNC_TO_BIN) {
                int total_root_n = manager_->GetRootFuncCount();
                int pw = manager_->GetProgressWidth();
                if (curr_stage_elapsed >= stage_timeout_sec && stageInfo.warningPrinted == false) {
                    manager_->SetActiveStageWarningPrinted(stage, stageInfo.rootFuncIndex);
                    warn_msg =
                        "[Compiler Monitor] | [** WARNING **] RootFunc(parallel): " +
                        PadRight(std::to_string(stageInfo.rootFuncIndex) + "/" + std::to_string(total_root_n), pw) +
                        " | Stage [" + stage + "] elapsed [" + FormatElapsed(curr_stage_elapsed) +
                        "] exceeded the current stage total time threshold [" + FormatElapsed(stage_timeout_sec) +
                        "] | RootFunc:[" + stageInfo.rootFuncName + "] | Number of op: " +
                        std::to_string(stageInfo.rootFuncOpSize) +
                        ", you can terminate the process by pressing Ctrl+C !!!";
                    (void)fprintf(stdout, "%s\n", warn_msg.c_str());
                    (void)fflush(stdout);
                    COMPILER_LOGI("%s", warn_msg.c_str());
                }
                if (curr_stage_elapsed >= pre_cost) {
                    int current_time = static_cast<int>(curr_stage_elapsed);
                    if (current_time >= last_print_time + print_interval_sec) {
                        last_print_time = current_time;
                        interval_msg =
                            "  |__ [Compiler Monitor] RootFunc(parallel): " +
                            PadRight(std::to_string(stageInfo.rootFuncIndex) + "/" + std::to_string(total_root_n), pw) +
                            " | Stage: " + PadStageName("CodeGen" + stage) +
                            "(processing) | Stage elapsed: " + PadElapsed(FormatElapsed(curr_stage_elapsed)) +
                            " | Total elapsed: " + PadElapsed(FormatElapsed(total_elapsed)) + " | RootFunc:[" +
                            stageInfo.rootFuncName + "]";
                        (void)fprintf(stdout, "%s\n", interval_msg.c_str());
                        (void)fflush(stdout);
                        COMPILER_LOGI("%s", interval_msg.c_str());
                    }
                }
                // end for (stage == STAGE_FUNC_TO_BIN)
            } else {
                int current_k = stageInfo.functionIndex;
                int total_n = manager_->GetTotalFunctionCount();
                int pw = manager_->GetProgressWidth();
                auto printPassStageTimeoutWarning = [&]() {
                    if (stage != "Pass") {
                        return;
                    }
                    double passStageTimeoutSec = MonitorManager::CalcPassStageTimeoutSec(stageInfo.functionOpSize);
                    if (passStageTimeoutSec < 0.0 || curr_stage_elapsed < passStageTimeoutSec ||
                        manager_->GetStageTimeoutFlag(stage)) {
                        return;
                    }
                    manager_->SetStageTimeoutFlag(stage);
                    warn_msg = "[Compiler Monitor] | [** WARNING **] Function: " +
                               PadRight(std::to_string(current_k) + "/" + std::to_string(total_n), pw) +
                               " | Stage [Pass] elapsed [" +
                               MonitorManager::FormatPassDurationForLog(curr_stage_elapsed) +
                               "] exceeded the pass stage time threshold [" +
                               MonitorManager::FormatPassDurationForLog(passStageTimeoutSec) + "] | Func:[" +
                               stageInfo.functionName + "] | Number of op: " +
                               std::to_string(stageInfo.functionOpSize) + current_pass_desc +
                               " | Standard: 200000 ops / 90.0s linear scaled" +
                               ", you can terminate the process by pressing Ctrl+C !!!";
                    (void)fprintf(stdout, "%s\n", warn_msg.c_str());
                    (void)fflush(stdout);
                    COMPILER_LOGW("%s", warn_msg.c_str());
                };
                printPassStageTimeoutWarning();
                // pass & codegen
                if (total_n > 1 && current_k > 0) {
                    if (stage != "Pass" && curr_stage_elapsed >= stage_timeout_sec &&
                        manager_->GetStageTimeoutFlag(stage) == false) {
                        manager_->SetStageTimeoutFlag(stage);
                        // for pass
                        warn_msg = "[Compiler Monitor] | [** WARNING **] Function: " +
                                   PadRight(std::to_string(current_k) + "/" + std::to_string(total_n), pw) +
                                   " | Stage [" + stage + "] elapsed [" + FormatElapsed(curr_stage_elapsed) +
                                   "] exceeded the current stage total time threshold [" +
                                   FormatElapsed(stage_timeout_sec) + "] | Func:[" + current_func_name +
                                   "] | Number of op: " + std::to_string(current_func_opsize) + current_pass_desc +
                                   ", you can terminate the process by pressing Ctrl+C !!!";
                        if (stage == "CodeGen") {
                            // for codeGen
                            warn_msg = "[Compiler Monitor] | [** WARNING **] Function: " +
                                       PadRight(std::to_string(current_k) + "/" + std::to_string(total_n), pw) +
                                       " | Stage [" + stage + "] elapsed [" + FormatElapsed(curr_stage_elapsed) +
                                       "] exceeded the current stage total time threshold [" +
                                       FormatElapsed(stage_timeout_sec) + "] | Func:[" +
                                       current_func_name + "] | Number of op: " + std::to_string(current_func_opsize) +
                                       ", you can terminate the process by pressing Ctrl+C !!!";
                        }
                        (void)fprintf(stdout, "%s\n", warn_msg.c_str());
                        (void)fflush(stdout);
                        COMPILER_LOGI("%s", warn_msg.c_str());
                    }
                    // pass & codegen warning

                    if (stage == "Pass") {
                        if (curr_stage_elapsed >= pre_cost) {
                            int current_time = static_cast<int>(curr_stage_elapsed);
                            if (current_time >= last_print_time + print_interval_sec) {
                                last_print_time = current_time;
                                interval_msg =
                                    "  |__ [Compiler Monitor] Function: " +
                                    PadRight(std::to_string(current_k) + "/" + std::to_string(total_n), pw) +
                                    " | Stage: " + PadStageName(stage) +
                                    "(processing) | Stage elapsed: " + PadElapsed(FormatElapsed(curr_stage_elapsed)) +
                                    " | Total elapsed: " + PadElapsed(FormatElapsed(total_elapsed)) + " | Func:[" +
                                    stageInfo.functionName + "]" + current_pass_desc;
                                (void)fprintf(stdout, "%s\n", interval_msg.c_str());
                                (void)fflush(stdout);
                                COMPILER_LOGI("%s", interval_msg.c_str());
                            }
                        }
                    } else {
                        if (curr_stage_elapsed >= pre_cost) {
                            int current_time = static_cast<int>(curr_stage_elapsed);
                            if (current_time >= last_print_time + print_interval_sec) {
                                last_print_time = current_time;
                                interval_msg =
                                    "  |__ [Compiler Monitor] " + PadLabel("Stage: ") + PadStageName(stage) +
                                    "(processing) | Stage elapsed: " + PadElapsed(FormatElapsed(curr_stage_elapsed)) +
                                    " | Total elapsed: " + PadElapsed(FormatElapsed(total_elapsed));
                                (void)fprintf(stdout, "%s\n", interval_msg.c_str());
                                (void)fflush(stdout);
                                COMPILER_LOGI("%s", interval_msg.c_str());
                            }
                        }
                    }
                    // pass & codegen processing
                } else {
                    // for prepare
                    if (stage != "Pass" && curr_stage_elapsed >= stage_timeout_sec &&
                        manager_->GetStageTimeoutFlag(stage) == false) {
                        manager_->SetStageTimeoutFlag(stage);
                        warn_msg = "[Compiler Monitor] | [** WARNING **] " + PadLabel("Stage: ") + "[" + stage +
                                   "] elapsed [" + FormatElapsed(curr_stage_elapsed) +
                                   "] exceeded the current stage total time threshold [" +
                                   FormatElapsed(stage_timeout_sec) +
                                   "] | Number of op: " + std::to_string(current_func_opsize) + current_pass_desc +
                                   ", you can terminate the process by pressing Ctrl+C !!!";
                        (void)fprintf(stdout, "%s\n", warn_msg.c_str());
                        (void)fflush(stdout);
                        COMPILER_LOGI("%s", warn_msg.c_str());
                    }

                    if (curr_stage_elapsed >= pre_cost) {
                        int current_time = static_cast<int>(curr_stage_elapsed);
                        if (current_time >= last_print_time + print_interval_sec) {
                            last_print_time = current_time;
                            interval_msg = "  |__ [Compiler Monitor] " + PadLabel("Stage: ") + PadStageName(stage) +
                                           "(processing) | Stashed function: " + std::to_string(total_n) +
                                           " | Stage elapsed: " + PadElapsed(FormatElapsed(curr_stage_elapsed)) +
                                           " | Total elapsed: " + PadElapsed(FormatElapsed(total_elapsed));
                            (void)fprintf(stdout, "%s\n", interval_msg.c_str());
                            (void)fflush(stdout);
                            COMPILER_LOGI("%s", interval_msg.c_str());
                        }
                    }
                }
            }
        }
    }
}

} // namespace npu::tile_fwk
