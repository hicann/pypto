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
 * \file DeviceMachine.cpp
 * \brief
 */

#include "cost_model/simulation/machine/DeviceMachine.h"

#include <sstream>
#include <utility>
#include <fstream>
#include <mutex>

#include "nlohmann/json.hpp"
#include "cost_model/simulation/base/ModelLogger.h"
#include "cost_model/simulation/base/ModelTop.h"
#include "cost_model/simulation/common/ISA.h"
#include "cost_model/simulation/value/TileCalculator.h"
#include "interface/function/function.h"
#include "simulation/tools/ParseInput.h"

using Json = nlohmann::json;
using namespace std::string_literals;
using namespace std::chrono_literals;

#define EXPVAL_LOG MLOG_DEBUG

#define TOPO_LOG MLOG_DEBUG

namespace CostModel {

void DeviceMachine::Step()
{
    // device machine is useless in calendar mode
    if (GetSim()->config.calendarMode != static_cast<uint64_t>(CalendarMode::DEVICE)) {
        return;
    }
    RunAtBegin();
    SubmitDeviceTask();
    RunAtEnd();
}

void DeviceMachine::RunAtBegin()
{
    if (!taskBuilded) {
        InitFunctions();
        PrintTopo();
        CalculateTileGolden();
        if (config.replayEnable) {
            BuildReplayInfo();
        }
        taskBuilded = true;
    }

    for (auto &submachine : subMachines) {
        if (submachine->machineType == MachineType::CPU) {
            auto core = std::dynamic_pointer_cast<AICPUMachine>(submachine);
            if (!core->localReadyQueues.Empty()) {
                uint64_t taskId = -1;
                core->localReadyQueues.Dequeue(taskId);
                MLOG_INFO("Dequeued task ID:", taskId);
                PushReadyQueue(taskMap.at(taskId)->machineType, taskId);
            }
        }
    }
}

void DeviceMachine::RunAtEnd()
{
    needTerminate = IsTerminate();
}

void DeviceMachine::RunPVModelDeviceTask()
{
    for (const auto& [taskId, task] : taskMap) {
        auto function = GetSim()->functionCache.GetFunction(task->functionHash);
        GetSim()->pv->Run(taskId, function->pSgId);
    }
    taskMap.clear();
    MLOG_INFO("[Cycle:", GetSim()->GetCycles(), "][Device ", machineId, "] run pvmodel execute tasks", taskMap.size());
}

void DeviceMachine::SubmitDeviceTask()
{
    if (!taskMap.empty()) {
        return;
    }
    if (taskMapQueue.empty()) {
        return;
    }
    SetReplayPreEnd();
    taskMap = std::move(taskMapQueue.front()), taskMapQueue.pop_front();
    if (GetSim()->pvLevel != PVModelLevel::PV_NON) {
        RunPVModelDeviceTask();
        return;
    }
    SetReplayPreStart();
    for (const auto& [taskId, task] : taskMap) {
        currentSeq = task->seqNo;
        if (task->remainingPredecessors == 0) {
            PushReadyQueue(task->machineType, taskId);
        }
    }
    MLOG_WARN("[Cycle:", GetSim()->GetCycles(), "][Device ", machineId, "] submit a new device task to AICPUs, size = ",
              taskMap.size());
}

// Device Init
void DeviceMachine::Build()
{
    MLOG_INFO("DeviceMachine start Building-----");
    config.OverrideDefaultConfig(&sim->cfgs);
    std::string queueId = "DeviceReadyQ";
    readyQueuePid = GetSim()->RegisterQueuePid(queueId);
    GetSim()->GetLogger()->SetProcessName(queueId, readyQueuePid, readyQueuePid);
    readyQueueTotalTid = queueSeq + coreTid;
    GetSim()->GetLogger()->SetThreadName(queueId, readyQueuePid, readyQueueTotalTid);
    queueSeq++;
    for (const auto &machineTypeStr : config.submachineTypes) {
        MachineType mType = ToMachineType(machineTypeStr);
        if (mType != MachineType::UNKNOWN) {
            readyQueues.try_emplace(mType);
            readyQueueTid[mType] = queueSeq + coreTid;
            GetSim()->GetLogger()->SetThreadName((MachineName(mType) + "_ReadyQ"), readyQueuePid, readyQueueTid[mType]);
            queueSeq++;
        }
        if (mType == MachineType::MIXAICORE) {
            cubeVecMix = true;
        }
    }

    stats = std::make_shared<DeviceStats>(GetSim()->GetReporter());
    tileStateGolden = std::make_shared<TileState>();
    tileState = std::make_shared<TileState>();
}

std::shared_ptr<SimSys> DeviceMachine::GetSim()
{
    return sim;
}

void DeviceMachine::Xfer()
{
    StepQueue();
    lastCycles = GetSim()->GetCycles();
    currentHeartModulo = GetSim()->GetCycles() % (GetSim()->config.heartInterval);
    if (currentHeartModulo < lastHeartModulo) {
        MLOG_WARN("@CostModel Heart Cycle:", GetSim()->GetCycles(), ", submit tasks: ", stats->totalSubmitNum);
    }
    lastHeartModulo = currentHeartModulo;
}

void DeviceMachine::Report()
{
    int machineSeq = GetMachineSeq(machineId);
    std::string name = std::to_string(machineSeq);
    stats->Report(name);
}

bool DeviceMachine::IsTerminate()
{
    if (sim->config.calendarMode != static_cast<uint64_t>(CalendarMode::DEVICE)) {
        return true;
    }
    bool readyQueueIsEmpty = std::all_of(
        readyQueues.begin(), readyQueues.end(),
        [](const auto& pair) { return pair.second.empty(); }
    );
    return readyQueueIsEmpty && readySet.empty() && taskMap.empty() && taskMapQueue.empty();
}

void DeviceMachine::InitFunctions()
{
    if (GetSim()->dynamicWorkflow) {
        BuildLeafFunctionTasks();
        return;
    }

    auto functionCache = GetSim()->functionCache.cache;
    auto startFuncHash = GetSim()->startFuncHash;

    if (GetSim()->testSingleFunc) {
        BuildSingleFuncTask();
        return;
    }

    if (config.submitTopo) {
        BuildSubTasksFromTopoJson();
        return;
    }

    if (functionCache[startFuncHash]->topoFromRootFunc) {
        BuildSubtasksFromRootFuncTopo();
        return;
    }
    ASSERT(0 && "Unexpected init functions mode");
}

void DeviceMachine::BuildLeafFunctionTasks() {
    sim->enableExpectValue = false;
    TaskMap taskM;
    auto functionCache = GetSim()->functionCache.cache;
    for (auto &[hash, func] : functionCache) {
        if (func->funcName.find("leaf") == std::string::npos) continue;
        auto subtask = std::make_shared<Task>();
        subtask->status = false;
        subtask->functionHash = hash;
        subtask->functionName = func->funcName;
        subtask->taskId = taskM.size();
        subtask->uniqueKey = subtask->taskId;
        subtask->machineType = func->machineType;
        subtask->remainingPredecessors = 0;
        GetSim()->taskToHash[subtask->taskId] = subtask->functionHash;
        taskM.insert({subtask->taskId, subtask});
    }
    taskMapQueue.push_back(taskM);
    GetSim()->ProcessTaskMap(taskM);
    MLOG_INFO("[Cycle:", GetSim()->GetCycles(), "][DeviceMachine][BuildLeafFunctionTasks] ", "Machine ", machineId,
    " build subtasks done");
}

void DeviceMachine::BuildSubtasksFromRootFuncTopo()
{
    TaskMap taskM;
    auto functionCache = GetSim()->functionCache.cache;
    auto startFuncHash = GetSim()->startFuncHash;
    auto startFunc = functionCache[startFuncHash];
    for (const auto &topoEntry : startFunc->inputTopo) {
        auto subtask = std::make_shared<Task>();
        subtask->status = false;
        subtask->functionHash = topoEntry.calleeHash;
        auto leafFunc = functionCache[subtask->functionHash];
        subtask->functionName = leafFunc->funcName;
        subtask->taskId = topoEntry.eSgId;
        subtask->psgId = leafFunc->pSgId;
        subtask->uniqueKey = subtask->taskId;
        subtask->machineType = leafFunc->machineType;
        subtask->remainingPredecessors = -topoEntry.readyState;
        subtask->fixedLatency = topoEntry.fixedLatency;
        subtask->fixedLatencyVal = topoEntry.fixedLatencyVal;
        if (uint64_t(startFunc->tileOps.size()) > topoEntry.eSgId) {
            subtask->semanticLabel = startFunc->tileOps[topoEntry.eSgId]->semanticLabel;
        }
        GetSim()->taskToHash[subtask->taskId] = subtask->functionHash;
        for (auto &out : topoEntry.outGraph) {
            subtask->successors.push_back(out);
        }
        taskM.insert({subtask->taskId, subtask});
    }
    for (const auto &it : taskM) {
        for (auto &successor : it.second->successors) {
            taskM.at(successor)->predecessors.push_back(it.first);
        }
    }

    for (const auto &it : taskM) {
        MLOG_INFO("Task ID: ", it.second->taskId);
        MLOG_INFO("  Remaing task num: ", it.second->remainingPredecessors);
        for (auto &pre : it.second->predecessors) {
            MLOG_INFO("  Predecessor:", pre);
        }
        for (auto &suc : it.second->successors) {
            MLOG_INFO("  Predecessor:", suc);
        }
    }
    taskMapQueue.push_back(taskM);
    GetSim()->ProcessTaskMap(taskM);

    MLOG_INFO("[Cycle:", GetSim()->GetCycles(), "][DeviceMachine][build_subtasks_from_topo] Machine ", machineId,
              " build subtasks done");
}

void DeviceMachine::BuildSubTasksFromTopoJson()
{
    if (config.replayTaskTimeScaling) {
        BuildLeafFunctionTasks();
    }

    CostModel::ParseInput parser;
    parser.ParseTopoJson(config.submitTopoPath, taskMapQueue);
    MLOG_INFO("[Cycle:", GetSim()->GetCycles(), "][DeviceMachine][BuildSubTasksFromTopoJson] Machine ", machineId,
              " build subtasks done, taskMapQueue size = ", taskMapQueue.size());
    uint64_t cnt = 0;
    for (auto &taskM : taskMapQueue) {
        GetSim()->ProcessTaskMap(taskM, std::to_string(cnt));
        cnt++;
        MLOG_INFO("[Cycle:", GetSim()->GetCycles(), "][DeviceMachine] taskMap Size:", taskM.size());
    }
    return;
}

void DeviceMachine::BuildSingleFuncTask()
{
    auto functionCache = GetSim()->functionCache.cache;
    TaskMap taskM;
    auto subtask = std::make_shared<Task>();
    subtask->status = false;
    subtask->functionHash = GetSim()->singleFuncHash;
    subtask->functionName = functionCache[subtask->functionHash]->funcName;
    subtask->taskId = 1;
    subtask->uniqueKey = subtask->taskId;
    subtask->machineType = functionCache[subtask->functionHash]->machineType;
    subtask->remainingPredecessors = 0;
    GetSim()->taskToHash[subtask->taskId] = subtask->functionHash;
    taskM.insert({subtask->taskId, subtask});
    GetSim()->GetCalendarGenerator()->InitTaskTopoInfo(taskM);
    taskMapQueue.push_back(taskM);
}

void DeviceMachine::PrintFunctionTopo(FunctionPtr func) {
    auto cache = GetSim()->functionCache.cache;
    TOPO_LOG("Function -> " + func->funcName);
    TOPO_LOG("incast:");
    for (const auto &incast : func->incastMagic) {
        TOPO_LOG(func->tileMap[incast]->Dump());
    }

    TOPO_LOG("outcast:");
    for (const auto &outcast : func->outcastMagic) {
        TOPO_LOG(func->tileMap[outcast]->Dump());
    }

    for (const auto &op: func->tileOps) {
        TOPO_LOG(op->opcode);
        TOPO_LOG("incast:");
        for (auto &incast : op->iOperand) {
            TOPO_LOG(incast->Dump());
        }

        TOPO_LOG("outcast:");
        for (auto &outcast : op->oOperand) {
            TOPO_LOG(outcast->Dump());
        }

        if (op->IsCall()) {
            auto invoke = op->operation->GetSubFuncInvokeInfo();
            invoke.PrintInvokeInfo("");
            PrintFunctionTopo(cache[op->calleeHash]);
        }
    }
}

void DeviceMachine::PrintTopo() {
    auto cache = GetSim()->functionCache.cache;
    auto startFuncHash = GetSim()->startFuncHash;
    if (startFuncHash == 0) {
        return;
    }
    auto func = cache[startFuncHash];

    if (func->parentFunction) {
        auto topo = func->parentFunction->topoInfo_;
        for (auto &e : topo.topology_) {
            TOPO_LOG("[TOPO] " + std::to_string(e.esgId) + ", " + std::to_string(e.readyState));
            for (auto &o : e.outGraph) {
                TOPO_LOG("[TOPO] out -> " + std::to_string(o));
            }
        }
    }

    PrintFunctionTopo(func);
}

void DeviceMachine::CalculateFunctionArgTile(FunctionPtr func, std::shared_ptr<TileState> state)
{
    for (auto &incast : func->incastMagic) {
        TileCalculator::Self().CalculateInput(func->tileMap[incast], state);
    }
}

void DeviceMachine::PrintFunctionOutputTile(FunctionPtr func, std::shared_ptr<TileState> state)
{
    for (auto &outcast : func->outcastMagic) {
        auto tile = func->tileMap[outcast];
        auto k = TileState::TileKey(tile->rawMagic, tile->bufType,
                            tile->shape, tile->offset);
        auto value = state->Load(k);
        EXPVAL_LOG("[EXPVAL] outcast -> " + std::to_string(tile->magic) + "-" + k.Dump() + " : " + std::to_string(value));
    }
}

void DeviceMachine::CalculateFunctionTileGolden(FunctionPtr func, std::shared_ptr<TileState> local,
                                                std::shared_ptr<TileState> global, int esgId) {
    auto cache = GetSim()->functionCache.cache;
    for (const auto &op: func->tileOps) {
        if (op->IsCall()) {
            auto callee = cache[op->calleeHash];
            EXPVAL_LOG(std::string("[EXPVAL] function") + " -> [" + callee->funcName + ", " + std::to_string(op->magic) + "] incast");
            for (auto &incast : op->iOperand) {
                auto k = TileState::TileKey(incast->rawMagic, incast->bufType, incast->shape, incast->offset);
                auto value = global->Load(k);
                EXPVAL_LOG("[EXPVAL] incast -> " + std::to_string(incast->magic) + "-" + k.Dump() + " : " + std::to_string(value));
            }
            EXPVAL_LOG("{");

            std::shared_ptr<TileState> l = std::make_shared<TileState>();
            CalculateFunctionTileGolden(callee, l, global, esgId);
            esgId++;

            EXPVAL_LOG("}");

            EXPVAL_LOG(std::string("[EXPVAL] function") + " -> [" + callee->funcName + ", " + std::to_string(op->magic) + "] outcast");
            for (auto &outcast : op->oOperand) {
                auto k = TileState::TileKey(outcast->rawMagic, outcast->bufType, outcast->shape, outcast->offset);
                auto value = global->Load(k);
                EXPVAL_LOG("[EXPVAL] outcast -> " + std::to_string(outcast->magic) + "-" + k.Dump() + " : " + std::to_string(value));
            }

            EXPVAL_LOG("[EXPVAL] ###########################");
        }
        else {
            TileCalculator::Self().Calculate(op, func->invoke[esgId], local, global);
        }
    }
}

void DeviceMachine::CalculateTileGolden() {
    if (!sim->enableExpectValue) {
        return;
    }

    auto cache = sim->functionCache.cache;
    auto startFuncHash = GetSim()->startFuncHash;

    EXPVAL_LOG("[EXPVAL] Function Root Golden");
    TileCalculator::Self().Reset();
    CalculateFunctionArgTile(cache[startFuncHash], tileStateGolden);
    CalculateFunctionTileGolden(cache[startFuncHash], nullptr, tileStateGolden, 0);
    EXPVAL_LOG("[EXPVAL] Function Root Golden Output:");
    PrintFunctionOutputTile(cache[startFuncHash], tileStateGolden);
    EXPVAL_LOG("[EXPVAL] ###########################");

    EXPVAL_LOG("[EXPVAL] Function Root");
    TileCalculator::Self().Reset();
    CalculateFunctionArgTile(cache[startFuncHash], tileState);
}

void DeviceMachine::PushReadyQueue(MachineType mType, uint64_t taskId)
{
    if (config.replayEnable && !replayPreExecute) {
        if (mType == MachineType::HUB) {
            CheckHUBTaskReplayInfo(taskId);
        }
        InsertReadySet(taskId);
        return;
    }
    if (cubeVecMix && mType != MachineType::HUB) {
        mType = MachineType::MIXAICORE;
    }
    readyQueues[mType].push_back(taskId);
    GetSim()->GetLogger()->AddCounterEvent(readyQueuePid, readyQueueTotalTid, CounterType::QUEUE_PUSH);
    GetSim()->GetLogger()->AddCounterEvent(readyQueuePid, readyQueueTid[mType], CounterType::QUEUE_PUSH);
}

uint64_t DeviceMachine::PopReadyQueue(MachineType mType)
{
    if (cubeVecMix && mType != MachineType::HUB) {
        mType = MachineType::MIXAICORE;
    }
    uint64_t taskId = readyQueues[mType].front();
    readyQueues[mType].pop_front();
    GetSim()->GetLogger()->AddCounterEvent(readyQueuePid, readyQueueTotalTid, CounterType::QUEUE_POP);
    GetSim()->GetLogger()->AddCounterEvent(readyQueuePid, readyQueueTid[mType], CounterType::QUEUE_POP);
    return taskId;
}

bool DeviceMachine::EraseReadyQueue(MachineType mType, uint64_t taskId)
{
    if (cubeVecMix && mType != MachineType::HUB) {
        mType = MachineType::MIXAICORE;
    }
    auto it = std::find(readyQueues[mType].begin(), readyQueues[mType].end(), taskId);
    if (it != readyQueues[mType].end()) {
        readyQueues[mType].erase(it);
        GetSim()->GetLogger()->AddCounterEvent(readyQueuePid, readyQueueTotalTid, CounterType::QUEUE_POP);
        GetSim()->GetLogger()->AddCounterEvent(readyQueuePid, readyQueueTid[mType], CounterType::QUEUE_POP);
        return true;
    } else {
        return false;
    }
}

void DeviceMachine::BuildReplayInfo()
{
    ParseInput parser;
    parser.ParseReplayInfoJson(config.replayFile, replayTasksInfoMap);
    // check replay info
    size_t hubMachineId = GetSim()->GetHUBCore()->machineId;
    auto hubIt = replayTasksInfoMap.find(hubMachineId);
    if (hubIt == replayTasksInfoMap.end()) {
        replayTasksInfoMap[hubMachineId] = std::deque<ReplayTaskEntry>();
    }
}

void DeviceMachine::InsertReadySet(uint64_t taskId)
{
    readySet.insert(taskId);
}

void DeviceMachine::CheckHUBTaskReplayInfo(uint64_t taskId)
{
    size_t hubMachineId = GetSim()->GetHUBCore()->machineId;
    auto &replayInfoQ = replayTasksInfoMap[hubMachineId];
    bool found = false;
    for (auto &entry : replayInfoQ) {
        if (entry.taskId == taskId && entry.seqNo == currentSeq) {
            found = true;
            return;
        }
    }
    if (!found) {
        replayInfoQ.push_back(ReplayTaskEntry(currentSeq, taskId, GetSim()->GetCycles(), GetSim()->GetCycles() + 1));
    }
}

void DeviceMachine::EraseReadySet(uint64_t taskId)
{
    readySet.erase(taskId);
}

bool DeviceMachine::IsReady(uint64_t taskId)
{
    auto it = readySet.find(taskId);
    return (it != readySet.end());
}

void DeviceMachine::SetReplayPreStart()
{
    if (!config.replayTaskTimeScaling) {
        return;
    }
    if (!hasPreExecute) {
        replayPreExecute = true;
        replayPreStartTime = GetSim()->GetCycles();
    }
}

void DeviceMachine::SetReplayPreEnd()
{
    if (!replayPreExecute) {
        return;
    }
    replayPreExecute = false;
    hasPreExecute = true;
    GetSim()->ResetCycles(replayPreStartTime);
    GetSim()->ResetStat(false);
    GetSim()->GetLogger()->EraseLogInfo(replayPreStartTime);
    for (auto &subMachine : subMachines) {
        subMachine->Reset();
    }
    EnableScaleTaskExecuteTime();
}

void DeviceMachine::EnableScaleTaskExecuteTime()
{
    for (auto &taskM : taskMapQueue) {
        for (auto &it : taskM) {
            it.second->scaleExecuteTime = true;
        }
    }
}

void DeviceMachine::ScaleTaskExecuteTime(ReplayTaskEntry &replayInfo)
{
    auto &task = taskMap[replayInfo.taskId];
    if (!task->scaleExecuteTime) {
        return;
    }
    task->fixedLatency = true;
    task->printRelativeCycle = true;
    uint64_t realCycle = replayInfo.eCycles - replayInfo.sCycles;
    auto function = GetSim()->functionCache.GetFunction(task->functionHash);

    task->proportion = double(realCycle) / double(function->totalCycles);
    task->fixedLatencyVal = uint64_t(task->proportion * double(function->totalCycles));
}

void DeviceMachine::Reset() {}
void DeviceMachine::InitQueueDelay() {}
void DeviceMachine::StepQueue() {}
}
