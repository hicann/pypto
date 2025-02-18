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
 * \file parallel_execute.cpp
 * \brief
 */

#include "parallel_execute.h"

#include <thread>
#include <mutex>
#include <optional>
#include <deque>

#include "interface/utils/log.h"

namespace npu::tile_fwk {
namespace {
class ThreadSafeTaskQueue {
public:
    explicit ThreadSafeTaskQueue(const std::deque<Task> &v) { q = v; }
    explicit ThreadSafeTaskQueue(std::deque<Task> &&v) { q = std::move(v); }

    std::optional<Task> GetTask() {
        const std::lock_guard<std::mutex> taskLock(m);

        if (q.empty()) {
            return std::nullopt;
        }

        Task e = q.front();
        q.pop_front();

        return e;
    }

private:
    std::deque<Task> q;
    std::mutex m;
};

void TaskRunner(ThreadSafeTaskQueue &taskQueue) {
    while (true) {
        auto taskMaybe = taskQueue.GetTask();
        if (!taskMaybe) {
            break;
        }

        auto &task = taskMaybe.value();
        task();
    }
}
}; // namespace

void ParallelExecuteAndWait(unsigned threadNum, std::deque<Task> tasks) {
    constexpr unsigned threshold = 128;
    if (threadNum > threshold) {
        ALOG_INFO_F("ParallelExecute: threadNum is too large and thus ignored. Reset to %ud.", threshold);
        threadNum = threshold;
    } else if (threadNum == 0) {
        threadNum = 1;
    }

    if (threadNum == 1) {
        // no need to use extra thread if only one thread is needed
        for (auto &task : tasks) {
            task();
        }

        return;
    }

    ThreadSafeTaskQueue taskQueue(std::move(tasks));

    std::vector<std::thread> threadPool;
    for (unsigned i = 0; i < threadNum; i++) {
        threadPool.emplace_back(TaskRunner, std::ref(taskQueue));
    }

    for (auto &tth : threadPool) {
        tth.join();
    }
}
} // namespace npu::tile_fwk
