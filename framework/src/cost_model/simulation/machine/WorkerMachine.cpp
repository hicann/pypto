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
 * \file WorkerMachine.cpp
 * \brief
 */

#include "cost_model/simulation/machine/WorkerMachine.h"

#include <random>
#include <cmath>

#include "cost_model/simulation/base/ModelLogger.h"
#include "cost_model/simulation/base/ModelTop.h"

namespace CostModel {

WorkerConfig WorkerMachine::config;

std::unordered_map<uint64_t, std::shared_ptr<WorkerTask>> WorkerMachine::taskMap;
uint64_t WorkerMachine::maxLayer = 0;
uint64_t WorkerMachine::bspCount;
MachineType WorkerMachine::bspState;

static std::default_random_engine &RandomEngine()
{
    static std::default_random_engine rdEngine = [useFixedRandomSeed = WorkerMachine::config.useFixedRandomSeed,
                                                  randomSeed = WorkerMachine::config.randomSeed]() {
        if (useFixedRandomSeed) {
            return std::default_random_engine(randomSeed);
        }
        return std::default_random_engine(std::random_device{}());
    }();
    return rdEngine;
}

constexpr double MOE_NORMAL_DIVISOR = 2.0;
constexpr double MOE_NORMAL_STD_FACTOR = 0.6266; // Precomputed value of std::sqrt(2 * std::log(1.1))

struct LatencyGenerator {
    std::normal_distribution<float> attnNormal;
    std::uniform_int_distribution<int> attnUniform;
    std::normal_distribution<float> ffnNormal;
    std::uniform_int_distribution<int> layerStartUniform;
    std::uniform_int_distribution<int> moeUniform;
    std::normal_distribution<> moeNormal;

    explicit LatencyGenerator(const WorkerConfig &config)
        : attnNormal(config.attnLatencyAvg, config.attnLatencySdv),
          attnUniform(config.attnLatencyMin, config.attnLatencyMax),
          ffnNormal(config.ffnLatencyAvg, config.ffnLatencySdv),
          layerStartUniform(0, config.layerStartVariation),
          moeUniform(0, config.expertNum - 1),
          moeNormal(config.expertNum / MOE_NORMAL_DIVISOR, (config.expertNum / MOE_NORMAL_DIVISOR) / MOE_NORMAL_STD_FACTOR) {}
};

static LatencyGenerator g_latencyGen(WorkerMachine::config);

void WorkerMachine::CreateAttnTask(uint64_t taskId, uint64_t session, uint64_t layer, uint64_t bindMachineId)
{
    uint64_t nextLayer = config.sessionNum * (config.moeNum + 1);
    auto attnTask = std::make_shared<WorkerTask>();
    attnTask->session = session;
    attnTask->taskId = taskId;
    attnTask->type = MachineType::ATTN;
    attnTask->bindMachineId = GetProcessID(MachineType::MIXED, bindMachineId);
    if (layer == 0) {
        if (session < GetSim()->config.workerMachineNumber) {
            attnTask->layer = g_latencyGen.layerStartUniform(RandomEngine());
        } else {
            attnTask->layer = taskMap[session % config.expertNum]->layer;
        }
    } else {
        attnTask->layer = taskMap[taskId - nextLayer]->layer + 1;  // layer always increases
    }
    if (config.isBsp) {
        attnTask->layer = layer;
    }
    if (layer > 0) {
        attnTask->latency = taskMap[taskId - nextLayer]->latency;
        for (uint64_t j = 0; j < config.moeNum; ++j) {
            attnTask->predecessors.push_back(nextLayer * (layer - 1) + session * config.moeNum + j);
        }
        attnTask->remainingPredecessors = config.moeNum;
    } else {
        attnTask->latency = g_latencyGen.attnUniform(RandomEngine());
        attnTask->remainingPredecessors = 0;
    }
    taskMap[taskId] = attnTask;
}

void WorkerMachine::CreateFfnTask(uint64_t taskId, uint64_t session, uint64_t layer)
{
    uint64_t nextLayer = config.sessionNum * (config.moeNum + 1);
    auto attnTask = taskMap[nextLayer * layer + session];
    auto ffnTask = std::make_shared<WorkerTask>();
    ffnTask->session = attnTask->session;
    ffnTask->taskId = taskId;
    ffnTask->layer = attnTask->layer;
    ffnTask->type = MachineType::FFN;
    double expert = g_latencyGen.moeNormal(RandomEngine());
    while (expert >= double(config.expertNum) || expert < 0) {
        expert = g_latencyGen.moeNormal(RandomEngine());
    }
    ffnTask->expert = GetProcessID(MachineType::MIXED, uint64_t(std::floor(expert)));
    ffnTask->latency = config.ffnLatencyAvg;
    attnTask->successors.push_back(taskId);
    ffnTask->predecessors.push_back(attnTask->taskId);
    ffnTask->remainingPredecessors = 1;
    if (layer < config.layerNum - 1) {
        ffnTask->successors.push_back(attnTask->taskId + nextLayer);
    }
    taskMap[taskId] = ffnTask;
}

// Generate attn & ffn tasks and connects them
void WorkerMachine::BuildTasks()
{
    int taskId = 0;
    // layers
    for (uint64_t runLayer = 0; runLayer < config.layerNum; ++runLayer) {
        uint64_t attnMachineId = 0;
        // attention tasks
        for (uint64_t i = 0; i < config.sessionNum; ++i) {
            CreateAttnTask(taskId, i, runLayer, attnMachineId);
            attnMachineId = (attnMachineId + 1) % (GetSim()->config.workerMachineNumber);
            ++taskId;
        }
        // ffn tasks
        for (uint64_t i = 0; i < config.sessionNum; ++i) {
            for (uint64_t j = 0; j < config.moeNum; ++j) {
                CreateFfnTask(taskId, i, runLayer);
                ++taskId;
            }
        }
    }
    // delete tasks that layer >= layerNum
    for (auto it = taskMap.begin(); it != taskMap.end();) {
        if (it->second->layer >= config.layerNum) {
            it = taskMap.erase(it);
        } else {
            auto task = it->second;
            auto newEnd = std::remove_if(task->successors.begin(), task->successors.end(), [](int s) {
                if (taskMap.find(s) == taskMap.end()) {
                    return true;
                }
                return taskMap[s]->layer >= config.layerNum;
            });
            task->successors.erase(newEnd, task->successors.end());
            ++it;
        }
    }
    // add attn polling
    if (config.attnPolling) {
        for (auto it : taskMap) {
            auto task = it.second;
            if (task->type != MachineType::ATTN) {
                continue;
            }
            uint64_t attnMachineId = task->bindMachineId % (static_cast<int>(MachineType::MIXED) * 10000);
            attnMachineId = (attnMachineId + task->layer) % GetSim()->config.workerMachineNumber;
            task->bindMachineId = GetProcessID(MachineType::MIXED, attnMachineId);
        }
    }
    // add ffn polling
    if (config.ffnPolling) {
        for (const auto &it : taskMap) {
            auto task = it.second;
            if (task->type != MachineType::FFN) {
                continue;
            }
            uint64_t expert = task->expert % (static_cast<int>(MachineType::MIXED) * 10000);
            expert = (expert + task->layer) % config.expertNum;
            task->expert = GetProcessID(MachineType::MIXED, expert);
        }
    }
    MLOG_INFO("[Cycle ", GetSim()->GetCycles(), "][WorkerMachine ", machineId, "] buildTasks DONE");
}

std::shared_ptr<SimSys> WorkerMachine::GetSim()
{
    return sim;
}

void WorkerMachine::Step()
{
    Process();
    RunAtEnd();
}

bool TaskCompareById(const TaskPack &a, const TaskPack &b)
{
    return a.taskId < b.taskId;
}

std::unordered_map<uint64_t, uint64_t> g_attnLayerCount;
std::unordered_map<uint64_t, uint64_t> g_ffnLayerCount;

void CountLayers(const std::deque<TaskPack> &readyPool)
{
    g_attnLayerCount.clear();
    g_ffnLayerCount.clear();
    for (auto it : readyPool) {
        auto t = WorkerMachine::taskMap[it.taskId];
        if (t->type == MachineType::ATTN) {
            g_attnLayerCount[t->layer]++;
        } else {
            g_ffnLayerCount[t->layer]++;
        }
    }
}

bool TaskCompareByLayer(const TaskPack &a, const TaskPack &b)
{
    auto taskA = WorkerMachine::taskMap[a.taskId];
    auto taskB = WorkerMachine::taskMap[b.taskId];

    bool isAAttn = (taskA->type == MachineType::ATTN);
    bool isBAttn = (taskB->type == MachineType::ATTN);
    bool aAttnFull = (g_attnLayerCount[taskA->layer] >= WorkerMachine::config.attnBatch);
    bool bAttnFull = (g_attnLayerCount[taskB->layer] >= WorkerMachine::config.attnBatch);
    bool aFfnFull = (g_ffnLayerCount[taskA->layer] >= WorkerMachine::config.ffnBatch);
    bool bFfnFull = (g_ffnLayerCount[taskB->layer] >= WorkerMachine::config.ffnBatch);

    if (taskA->layer == taskB->layer) {
        if (isAAttn) {
            if (isBAttn) {
                return false;                   // 都是 ATTN
            }
            return aAttnFull || !bFfnFull;  // task_a 是 ATTN，task_b 是 FFN
        }
        if (isBAttn) {
            if (bAttnFull) {
                return false;   // task_b 是 ATTN 且已满
            }
            return aFfnFull;  // task_b 是 ATTN 且未满
        }
        return false;  // 都是 FFN
    }

    return taskA->layer < taskB->layer;
}

bool WorkerMachine::IsBspReady()
{
    if (bspState == MachineType::ATTN && bspCount == config.sessionNum) {
        uint64_t readyCnt = 0;
        for (const auto &machine : GetSim()->machines) {
            if (machine->machineType != MachineType::MIXED) {
                continue;
            }
            auto m = std::dynamic_pointer_cast<WorkerMachine>(machine);
            readyCnt += m->readyPool.size();
        }
        if (readyCnt == config.sessionNum * config.moeNum) {
            bspState = MachineType::FFN;
            bspCount = 0;
            return true;  // ready for next ffn
        }
    } else if (bspState == MachineType::FFN && bspCount == config.sessionNum * config.moeNum) {
        uint64_t readyCnt = 0;
        for (const auto &machine : GetSim()->machines) {
            if (machine->machineType != MachineType::MIXED) {
                continue;
            }
            auto m = std::dynamic_pointer_cast<WorkerMachine>(machine);
            readyCnt += m->readyPool.size();
        }
        if (readyCnt == config.sessionNum) {
            bspState = MachineType::ATTN;
            bspCount = 0;
            return true;  // ready for next attn
        }
    }
    if (!readyPool.empty()) {
        std::sort(readyPool.begin(), readyPool.end(), TaskCompareById);
        TaskPack packet = readyPool.front();
        if (taskMap[packet.taskId]->type == bspState) {
            return true;  // tasks left in current bsp
        }
    }
    return false;
}

bool WorkerMachine::IsBatchReady()
{
    uint64_t size = 0;
    uint64_t layer = 0;
    MachineType type = MachineType::UNKNOWN;
    for (auto it : readyPool) {
        auto task = taskMap[it.taskId];
        if (size == 0) {
            type = task->type;
            layer = task->layer;
            size++;
            continue;
        }
        if (type != task->type) {
            break;
        } else if (layer != task->layer) {
            break;
        }
        size++;
    }
    if (size == 0) {
        return true;
    }
    // batch is ready if batch size is met or layer is left behind
    if (type == MachineType::ATTN) {
        if (size >= config.attnBatch) {
            return true;
        } else if (layer < maxLayer - 1) {
            return true;
        }
    } else if (type == MachineType::FFN) {
        if (size >= config.ffnBatch) {
            return true;
        } else if (layer < maxLayer - 1) {
            return true;
        }
    }
    return false;
}

void WorkerMachine::Process()
{
    if (GetSim()->GetCycles() < currentEnd) {
        return;
    }
    Dispatch();
    Execute();
}

void WorkerMachine::Dispatch()
{
    std::normal_distribution<float> latencyNormal(config.commLatencyAvg, config.commLatencySdv);

    for (auto taskId : currentTasks) {
        MLOG_INFO("[Cycle ", GetSim()->GetCycles(), "][WorkerMachine ", machineId, "] complete task ", taskId);
        if (config.isBsp) {
            bspCount++;
        }
        for (auto id : taskMap[taskId]->successors) {
            auto task = taskMap[id];
            CommunicationPacket packet;
            packet.machineIdFrom = this->machineId;
            if (task->type == MachineType::ATTN) {
                packet.machineIdTo = GetSim()->pidToMachineMp[task->bindMachineId]->machineId;
            } else if (task->type == MachineType::FFN) {
                packet.machineIdTo = GetSim()->pidToMachineMp[task->expert]->machineId;
            }
            packet.taskId = id;

            float latency = latencyNormal(RandomEngine());
            if (latency < 1.0) {
                latency = 1.0;
            }
            packet.latency = config.commLatencyAvg;
            busMachine->ForwardPacket(packet);
        }
    }
    if (!currentTasks.empty()) {
        LoggerRecordTaskEnd();
    }
    currentTasks.clear();
}

void WorkerMachine::Execute()
{
    if (config.isBsp && !IsBspReady()) {
        return;
    }
    if (!config.isBsp) {
        CountLayers(readyPool);
        std::sort(readyPool.begin(), readyPool.end(), TaskCompareByLayer);
        if (!IsBatchReady() && GetSim()->GetCycles() < currentEnd + config.layerWaitTime) {
            return;
        }
    }
    // run tasks in batch
    while (!readyPool.empty()) {
        TaskPack packet = readyPool.front();
        auto newTask = taskMap[packet.taskId];
        if (currentTasks.empty()) {
            currentLayer = newTask->layer;
            currentType = newTask->type;
        }
        if (newTask->type != currentType || newTask->layer != currentLayer) {
            break;
        }
        readyPool.pop_front();
        currentTasks.push_back(packet.taskId);
        if (currentType == MachineType::ATTN && currentTasks.size() == config.attnBatch) {
            break;
        }
        if (currentType == MachineType::FFN && currentTasks.size() == config.ffnBatch) {
            break;
        }
    }
    // calc latency, logging
    if (!currentTasks.empty()) {
        if (currentLayer > maxLayer) {
            maxLayer = currentLayer;
        }
        uint64_t latency = 0;
        for (auto t : currentTasks) {
            latency += taskMap[t]->latency;
        }
        int batchSize = int(currentTasks.size());
        currentEnd = GetSim()->GetCycles() + latency / batchSize;
        std::string tasksName;
        for (auto t : currentTasks) {
            tasksName += " Task" + std::to_string(t);
        }
        tasksName = MachineName(currentType) + " BS" + std::to_string(batchSize) + "-L" +
                        std::to_string(currentLayer) + tasksName;
        MLOG_INFO("[Cycle ", GetSim()->GetCycles(), "][WorkerMachine ", machineId, "] begin new tasks", tasksName);
        LoggerRecordTaskStart(tasksName);
    }
}

void WorkerMachine::RunAtEnd()
{
    nextCycles = INT_MAX;
    if (GetSim()->GetCycles() < currentEnd) {
        nextCycles = currentEnd;
        GetSim()->UpdateNextCycles(currentEnd);
    } else if (!readyPool.empty() && GetSim()->GetCycles() < currentEnd + config.layerWaitTime) {
        nextCycles = currentEnd + config.layerWaitTime;
        GetSim()->UpdateNextCycles(currentEnd + config.layerWaitTime);
    }
    needTerminate = IsTerminate();
    if (needTerminate) {
        MLOG_INFO("[Cycle ", GetSim()->GetCycles(), "][WorkerMachine ", machineId, "] needs to terminate!!!");
    }
}

void WorkerMachine::Build()
{
    config.OverrideDefaultConfig(&sim->cfgs);
    if (taskMap.empty()) {
        BuildTasks();
    }
    for (const auto &it : taskMap) {
        uint64_t taskId = it.first;
        auto task = it.second;
        if (task->remainingPredecessors > 0) {
            continue;
        }
        if (task->bindMachineId == machineId) {
            if (task->layer > maxLayer) {
                maxLayer = task->layer;
            }
            TaskPack packet;
            packet.taskId = taskId;
            readyPool.push_back(packet);
        }
    }
    if (config.isBsp) {
        bspCount = 0;
        bspState = MachineType::ATTN;
    }
}

bool WorkerMachine::IsTerminate()
{
    return readyPool.empty() && currentTasks.empty();
}

void WorkerMachine::Reset() {}
void WorkerMachine::Report() {}
void WorkerMachine::Xfer() {}
void WorkerMachine::InitQueueDelay() {}
void WorkerMachine::StepQueue() {}
}