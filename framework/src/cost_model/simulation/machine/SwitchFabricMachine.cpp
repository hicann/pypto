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
 * \file SwitchFabricMachine.cpp
 * \brief
 */

#include "cost_model/simulation/machine/SwitchFabricMachine.h"

#include <iostream>
#include <vector>

#include "cost_model/simulation/base/ModelLogger.h"
#include "cost_model/simulation/base/ModelTop.h"
#include "cost_model/simulation/common/ISA.h"

namespace CostModel {

SwitchFabricConfig SwitchFabricMachine::config;

std::shared_ptr<SimSys> SwitchFabricMachine::GetSim()
{
    return sim;
}

void SwitchFabricMachine::Step()
{
    Process();
    RunAtEnd();
}

void SwitchFabricMachine::Process()
{
    // processing
    auto currCycle = GetSim()->GetCycles();

    while (!processingTasks_.empty()) {
        auto [endCycle, taskID] = processingTasks_.top();
        if (endCycle == currCycle) {
            processingTasks_.pop();
            ReleaseDependency(taskID);
        } else {
            break;
        }
    }

    for (auto it = communicationQueue_.begin(); it != communicationQueue_.end();) {
        if (!config.simCommLatency) {
            ReleaseDependency(it->taskId);
            it = communicationQueue_.erase(it);
            continue;
        }

        if (machineBusyUntil_[it->machineIdFrom] > currCycle || machineBusyUntil_[it->machineIdTo] > currCycle) {
            it++;
            continue;
        }

        if (it->latency == 0) {
            ReleaseDependency(it->taskId);
        } else {
            processingTasks_.emplace(currCycle + it->latency, it->taskId);

            machineBusyUntil_[it->machineIdFrom] = currCycle + it->latency;
            machineBusyUntil_[it->machineIdTo] = currCycle + it->latency;
        }
        it = communicationQueue_.erase(it);
    }
}

void SwitchFabricMachine::ReleaseDependency(int taskID)
{
    auto task = WorkerMachine::taskMap[taskID];
    task->remainingPredecessors--;
    if (task->remainingPredecessors == 0) {
        // dispatch
        TaskPack packet;
        packet.taskId = taskID;
        if (task->type == MachineType::ATTN) {
            auto m = std::dynamic_pointer_cast<WorkerMachine>(GetSim()->pidToMachineMp[task->bindMachineId]);
            m->readyPool.push_back(packet);
        } else if (task->type == MachineType::FFN) {
            auto m = std::dynamic_pointer_cast<WorkerMachine>(GetSim()->pidToMachineMp[task->expert]);
            m->readyPool.push_back(packet);
        }
    }
}

void SwitchFabricMachine::ForwardPacket(const CommunicationPacket &packet)
{
    communicationQueue_.push_back(packet);
}

void SwitchFabricMachine::NotifyMachineRelatedPacket(int machineID)
{
    // Do nothing
    machineId = machineID;
}

void SwitchFabricMachine::RunAtEnd()
{
    needTerminate = IsTerminate();
}

void SwitchFabricMachine::Build()
{
    config.OverrideDefaultConfig(&sim->cfgs);
}

bool SwitchFabricMachine::IsTerminate()
{
    return communicationQueue_.empty() && processingTasks_.empty();
}

void SwitchFabricMachine::Reset() {}
void SwitchFabricMachine::Report() {}
void SwitchFabricMachine::Xfer() {}
void SwitchFabricMachine::InitQueueDelay() {}
void SwitchFabricMachine::StepQueue() {}
}