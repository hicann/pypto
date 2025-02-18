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
 * \file WorkerMachine.h
 * \brief
 */

#pragma once

#include <map>
#include <vector>
#include <queue>
#include "cost_model/simulation/base/Machine.h"
#include "cost_model/simulation/config/WorkerConfig.h"
#include "cost_model/simulation/machine/SwitchFabricMachine.h"

namespace CostModel {

class WorkerTask {
public:
    uint64_t session;
    uint64_t taskId;
    uint64_t layer;

    uint64_t latency;

    MachineType type;
    uint64_t expert;
    uint64_t bindMachineId;

    std::vector<uint64_t> successors;
    std::vector<uint64_t> predecessors;
    uint64_t remainingPredecessors = 0;
};

class WorkerMachine : public Machine {
public:
    std::deque<TaskPack> readyPool;
    std::vector<uint64_t> currentTasks;
    uint64_t currentEnd = 0;
    uint64_t currentLayer;
    MachineType currentType;
    std::shared_ptr<SwitchFabricMachine> busMachine;

    static std::unordered_map<uint64_t, std::shared_ptr<WorkerTask>> taskMap;
    static uint64_t maxLayer;
    static uint64_t bspCount;
    static MachineType bspState;

    static WorkerConfig config;

    void CreateAttnTask(uint64_t taskId, uint64_t session, uint64_t layer, uint64_t bindMachineId);
    void CreateFfnTask(uint64_t taskId, uint64_t session, uint64_t layer);
    void BuildTasks();
    void Process();
    void Dispatch();
    void Execute();
    bool IsBatchReady();
    bool IsBspReady();

    void RunAtEnd();
    void Step() override;
    bool IsTerminate() override;
    void Xfer() override;
    void Build() override;
    void Reset() override;
    std::shared_ptr<SimSys> GetSim() override;
    void Report() override;
    void InitQueueDelay() override;
    void StepQueue() override;
};

}