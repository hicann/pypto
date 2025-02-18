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
 * \file SwitchFabricMachine.h
 * \brief
 */

#pragma once

#include <map>
#include <vector>
#include <random>
#include <list>
#include <queue>

#include "cost_model/simulation/base/Machine.h"
#include "cost_model/simulation/common/ISA.h"
#include "cost_model/simulation/config/ModelConfig.h"
#include "cost_model/simulation/config/SwitchFabricConfig.h"
#include "cost_model/simulation/statistics/TraceLogger.h"

namespace CostModel {

class SwitchFabricMachine : public Machine {
public:
    int currentTask = -1;
    int currentEnd = 0;

    static SwitchFabricConfig config;

    void ForwardPacket(const CommunicationPacket &packet);

    void Process();
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

private:
    void NotifyMachineRelatedPacket(int machineID);
    void ReleaseDependency(int taskID);

    std::map<size_t, uint64_t> machineBusyUntil_;
    std::list<CommunicationPacket> communicationQueue_;
    std::priority_queue<std::pair<uint64_t, int>, std::vector<std::pair<uint64_t, int>>,
                        std::greater<std::pair<uint64_t, int>>>
        processingTasks_;
};

}