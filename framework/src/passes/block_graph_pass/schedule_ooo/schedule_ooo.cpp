/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file schedule_ooo.cpp
 * \brief
 */

#include "schedule_ooo.h"

#include <sstream>

#include "passes/block_graph_pass/schedule_ooo/common/iso_matcher.h"
#include "passes/pass_log/pass_log.h"
#include "passes/pass_utils/dead_operation_eliminate.h"

#ifndef MODULE_NAME
#define MODULE_NAME "OoOSchedule"
#endif

namespace npu::tile_fwk {

bool OoOSchedule::IsAicpuProgram(std::vector<Operation*> opList)
{
    for (auto& op : opList) {
        if (op->GetCoreType() == CoreType::AICPU) {
            return true;
        }
    }
    return false;
}

std::string GetOpInfo(Operation* op)
{
    if (op == nullptr)
        return "nullptr";
    return op->GetOpcodeStr() + "[" + std::to_string(op->GetOpMagic()) + "]";
}

static bool IsMixGraph(const std::vector<Operation*>& opList)
{
    bool hasAIC = false;
    bool hasAIV = false;
    for (auto opPtr : opList) {
        auto coreType = OpcodeManager::Inst().GetCoreType(opPtr->GetOpcode());
        if (coreType == OpCoreType::AIC) {
            hasAIC = true;
        } else if (coreType == OpCoreType::AIV) {
            hasAIV = true;
        }
        if (hasAIC && hasAIV) {
            return true;
        }
    }
    return false;
}

void OoOSchedule::CollectStatistic(OoOScheduleStatistic& oooHealthCheck, Function& function,
                                   std::pair<uint64_t, Function*>& program)
{
    if (passDfxconfigs_.healthCheck) {
        oooHealthCheck.SetOutputPrefix(GetDumpFilePrefix(function, false, program.second, program.first));
        statisticMap_.insert({program.first, oooHealthCheck});
    }
}

Status OoOSchedule::ModifyTaskOplist(std::vector<Operation*>& taskList,
                                     const std::unordered_map<int, Operation*>& allocMap)
{
    std::unordered_set<int> memIds;
    std::unordered_map<int, Operation*> taskAllocMap;
    for (const auto& op : taskList) {
        if (op->GetOpcodeStr().find("ALLOC") != std::string::npos) {
            taskAllocMap[op->GetOutputOperand(0)->memoryrange.memId] = op;
        }
        for (auto& oOperand : op->GetOOperands()) {
            if (oOperand->GetMemoryTypeOriginal() < MemoryType::MEM_DEVICE_DDR) {
                memIds.insert(oOperand->memoryrange.memId);
            }
        }
    }
    for (const auto& memId : memIds) {
        if (taskAllocMap.find(memId) != taskAllocMap.end()) {
            continue;
        }
        APASS_LOG_INFO_F(Elements::Operation, "The alloc op of memId[%d] in other graph", memId);
        auto it = allocMap.find(memId);
        if (it != allocMap.end()) {
            taskList.push_back(it->second);
        } else {
            APASS_LOG_ERROR_F(Elements::Operation, "Cannot find tensor[%d]'s alloc.", memId);
            return FAILED;
        }
    }
    return SUCCESS;
}

Status OoOSchedule::TaskSchedule(std::vector<Operation*>& opList, Function& function, TaskSplitter& splitter)
{
    std::unordered_map<int, Operation*> allocMap;
    for (const auto& op : opList) {
        if (op->GetOpcodeStr().find("ALLOC") != std::string::npos) {
            allocMap[op->GetOutputOperand(0)->memoryrange.memId] = op;
        }
    }
    // 对每个 task 做排序 + 耗时评估
    for (auto& taskNode : splitter.GetTaskGraph().tasks) {
        if (ModifyTaskOplist(taskNode.opList_, allocMap) != SUCCESS) {
            return FAILED;
        }
        OptimizeSort optimizeSort(taskNode.opList_, function);
        if (optimizeSort.SortOps() != SUCCESS) {
            return FAILED;
        }
        LatencyEstimator estimator(optimizeSort.operations, opList);
        if (estimator.LatencyEstimatorMainLoop() != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "LatencyEstimator failed, taskNode[%d].", taskNode.idx);
            return FAILED;
        }
        taskNode.opList_ = std::move(optimizeSort.operations);
        taskNode.latency = estimator.state_.clock;
    }

    // 核间调度
    CoreScheduler coreScheduler;
    coreScheduler.Schedule(splitter.GetTaskGraph(), function.paramConfigs_.oooSchedMode);
    if (splitter.MarkInternalSubgraphID() != SUCCESS) {
        return FAILED;
    }
    return SUCCESS;
}

void OoOSchedule::StableUnique(std::vector<Operation*>& newOpList)
{
    std::unordered_set<Operation*> seen;
    seen.reserve(newOpList.size());

    size_t write = 0;
    for (size_t read = 0; read < newOpList.size(); ++read) {
        if (seen.insert(newOpList[read]).second) {
            newOpList[write++] = newOpList[read];
        }
    }
    newOpList.resize(write);
}

Status OoOSchedule::ConcatTaskOpLists(TaskSplitter& splitter, std::vector<Operation*>& newOpList, Function& function)
{
    auto cyclePairs = splitter.GetCycledTaskNodePairs();
    auto scheduleUnits = BuildScheduleUnits(splitter.GetTaskGraph().tasks, cyclePairs);
    for (auto& unit : scheduleUnits) {
        if (unit.isMerged) {
            OptimizeSort optimizeSort(unit.mergedOps, function);
            if (optimizeSort.SortOps() != SUCCESS) {
                return FAILED;
            }
            unit.mergedOps = optimizeSort.operations;
        }
        newOpList.insert(newOpList.end(), unit.mergedOps.begin(), unit.mergedOps.end());
    }
    StableUnique(newOpList);
    if (ModifyAllocOrder(newOpList) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "ModifyAllocOrder failed.");
        return FAILED;
    }
    return SUCCESS;
}

Status OoOSchedule::DoOoOSchedule(std::vector<Operation*>& opList, Function& function,
                                  std::pair<uint64_t, Function*>& program, int64_t& workeSpaceSize, bool enableDualDst)
{
    OoOScheduler oooSchedule(*program.second);
    oooSchedule.SetEnableDualDst(enableDualDst);
    if (enableDualDst) {
        oooSchedule.SetDualDstPairs(std::move(dualDstPairs_));
        oooSchedule.SetEnableDualDstAllocGuard(true);
    }
    OoOScheduleStatistic oooHealthCheck;
    MemoryTracer oooMemoryTrace;
    if (passDfxconfigs_.healthCheck) {
        oooSchedule.AddObserver(&oooHealthCheck);
    }
    if (passDfxconfigs_.dumpGraph) {
        oooSchedule.AddObserver(&oooMemoryTrace);
    }

    Status schedStat;
    if (Platform::Instance().GetSoc().GetAICToAIVCoreRatio() == SoCAICToAIVCoreRatio::ONE_AIC_TO_ONE_AIV_CORE) {
        schedStat = oooSchedule.Schedule(opList, CORE_INIT_CONFIGS_HARDWARE_ONE_AIV);
    } else if (Platform::Instance().GetSoc().GetNPUArch() != NPUArch::DAV_3510 || !IsMixGraph(opList)) {
        schedStat = oooSchedule.Schedule(opList);
    } else {
        schedStat = oooSchedule.Schedule(opList, CORE_INIT_CONFIGS_HARDWARE_TWO_AIV);
    }

    if (schedStat != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "Schedule failed.");
        if (passDfxconfigs_.dumpGraph) {
            FlushMemoryTraceOnFailure(oooMemoryTrace, function, program);
        }
        return FAILED;
    }
    CollectStatistic(oooHealthCheck, function, program);
    if (passDfxconfigs_.dumpGraph) {
        CollectMemoryTrace(oooMemoryTrace, function, program);
    }
    APASS_LOG_INFO_F(Elements::Operation, "Subgraph[%zu] OOOSchedule end.", program.first);
    program.second->ScheduleBy(oooSchedule.GetNewOperations());
    program.second->RecordOOOSeq();
    RescheduleUtils::UpdateTensorConsProd(program.second);
    workeSpaceSize = std::max(workeSpaceSize, (*program.second).GetStackWorkespaceSize());
    function.SetStackWorkespaceSize(workeSpaceSize);
    return SUCCESS;
}

Status OoOSchedule::Schedule(std::vector<Operation*>& opList, Function& function,
                             std::pair<uint64_t, Function*>& program, int64_t& maxWorkeSpaceSize)
{
    if (CheckAllocOp(opList) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "CheckAllocOp failed!");
        return FAILED;
    }
    bool enableDualDst = false;
    if (Platform::Instance().GetSoc().GetNPUArch() != NPUArch::DAV_3510 || !IsMixGraph(opList)) {
        OptimizeSort sort(opList, *program.second);
        if (sort.SortOps() != SUCCESS) {
            return FAILED;
        }
        opList = sort.operations;
    } else {
        // task划分
        TaskSplitter splitter;
        splitter.SplitGraph(opList);

        // task调度
        if (TaskSchedule(opList, *program.second, splitter) != SUCCESS) {
            return FAILED;
        }

        // 按 task 顺序拼接 opList
        std::vector<Operation*> newOpList;
        if (ConcatTaskOpLists(splitter, newOpList, *program.second) != SUCCESS) {
            return FAILED;
        }
        opList = newOpList;
        enableDualDst = ShouldEnableDualDst(splitter);
    }

    // 乱序调度
    if (DoOoOSchedule(opList, function, program, maxWorkeSpaceSize, enableDualDst) != SUCCESS) {
        return FAILED;
    }
    return SUCCESS;
}

void OoOSchedule::CollectAivTasksByStart(const std::vector<TaskNode>& tasks,
                                         std::unordered_map<int, std::vector<const TaskNode*>>& aiv0ByStart,
                                         std::unordered_map<int, std::vector<const TaskNode*>>& aiv1ByStart)
{
    for (const auto& t : tasks) {
        if (t.opList_.empty())
            continue;
        if (t.targetCoreType == TargetCoreType::AIV0) {
            aiv0ByStart[t.startTime].push_back(&t);
        } else if (t.targetCoreType == TargetCoreType::AIV1) {
            aiv1ByStart[t.startTime].push_back(&t);
        }
    }
}

size_t OoOSchedule::CountNonAllocOps(const std::vector<Operation*>& ops)
{
    size_t count = 0;
    for (auto* op : ops) {
        if (op != nullptr && !IsAllocOpCode(op->GetOpcode()))
            count++;
    }
    return count;
}

bool OoOSchedule::CheckAndRecordDualDstPair(int start, const TaskNode* ta, const TaskNode* tb)
{
    std::unordered_set<Operation*> opsA(ta->opList_.begin(), ta->opList_.end());
    std::unordered_set<Operation*> opsB(tb->opList_.begin(), tb->opList_.end());
    auto entriesA = FindTaskEntryOps(ta->opList_, opsA);
    auto entriesB = FindTaskEntryOps(tb->opList_, opsB);
    auto res = IsoMatchChains(entriesA, entriesB, opsA, opsB);
    size_t nonAllocA = CountNonAllocOps(ta->opList_);
    size_t nonAllocB = CountNonAllocOps(tb->opList_);
    if (!res.rootIsomorphic || res.pairs.size() != nonAllocA || res.pairs.size() != nonAllocB) {
        APASS_LOG_INFO_F(Elements::Operation,
                         "DualDst disabled by isomorphism check: startTime=%d, rootIsomorphic=%d, "
                         "truncated=%zu, matchedNonAlloc=%zu, nonAllocA=%zu, nonAllocB=%zu.",
                         start, static_cast<int>(res.rootIsomorphic), res.truncatedCount, res.pairs.size(), nonAllocA,
                         nonAllocB);
        return false;
    }
    APASS_LOG_INFO_F(Elements::Operation,
                     "DualDst isomorphism check passed: startTime=%d, matchedNonAlloc=%zu, allocPairs=%zu.", start,
                     res.pairs.size(), res.allocPairs.size());
    for (const auto& p : res.allocPairs) {
        auto pairIt = dualDstPairs_.find(p.opA);
        if (pairIt != dualDstPairs_.end()) {
            if (pairIt->second != p.opB) {
                APASS_LOG_INFO_F(Elements::Operation,
                                 "DualDst disabled: duplicated AIV0 alloc maps to different AIV1 alloc, "
                                 "aiv0Alloc=%s, oldAiv1Alloc=%s, newAiv1Alloc=%s.",
                                 GetOpInfo(p.opA).c_str(), GetOpInfo(pairIt->second).c_str(), GetOpInfo(p.opB).c_str());
                return false;
            }
            continue;
        }
        dualDstPairs_[p.opA] = p.opB;
    }
    return true;
}

bool OoOSchedule::ShouldEnableDualDst(TaskSplitter& splitter)
{
    dualDstPairs_.clear();
    const auto& tasks = splitter.GetTaskGraph().tasks;
    std::unordered_map<int, std::vector<const TaskNode*>> aiv0ByStart;
    std::unordered_map<int, std::vector<const TaskNode*>> aiv1ByStart;
    CollectAivTasksByStart(tasks, aiv0ByStart, aiv1ByStart);
    bool hasPair = false;
    for (const auto& [start, aiv0Tasks] : aiv0ByStart) {
        auto it = aiv1ByStart.find(start);
        if (it == aiv1ByStart.end())
            continue;
        hasPair = true;
        for (const TaskNode* ta : aiv0Tasks) {
            for (const TaskNode* tb : it->second) {
                if (!CheckAndRecordDualDstPair(start, ta, tb)) {
                    return false;
                }
            }
        }
    }
    if (hasPair) {
        APASS_LOG_INFO_F(Elements::Operation,
                         "DualDst enabled: all same-startTime AIV0/AIV1 task pairs fully isomorphic.");
    } else {
        dualDstPairs_.clear();
    }
    return hasPair;
}

Status OoOSchedule::BuildMemIdToAllocIdx(const std::vector<Operation*>& opList,
                                         std::unordered_map<uint64_t, size_t>& memIdToAllocIdx)
{
    for (size_t i = 0; i < opList.size(); i++) {
        Operation* op = opList[i];
        if (op == nullptr) {
            APASS_LOG_ERROR_F(Elements::Operation, "ModifyAllocOrder: null op at index %zu.", i);
            return FAILED;
        }
        if (IsAllocOpCode(op->GetOpcode())) {
            uint64_t memId = op->GetOutputOperand(0)->memoryrange.memId;
            memIdToAllocIdx[memId] = i;
        }
    }
    return SUCCESS;
}

bool OoOSchedule::MoveAllocBeforeOp(std::vector<Operation*>& opList, size_t allocIdx, int targetIdx,
                                    std::unordered_map<uint64_t, size_t>& memIdToAllocIdx, uint64_t memId)
{
    if (allocIdx >= opList.size()) {
        APASS_LOG_ERROR_F(Elements::Operation, "ModifyAllocOrder: allocIdx %zu out of range (size %zu).", allocIdx,
                          opList.size());
        return false;
    }
    if (allocIdx > static_cast<size_t>(targetIdx)) {
        Operation* allocOp = opList[allocIdx];
        APASS_LOG_DEBUG_F(Elements::Operation, "Move %s[%d] from index[%zu] to index[%d] before %s[%d]",
                          allocOp->GetOpcodeStr().c_str(), allocOp->GetOpMagic(), allocIdx, targetIdx,
                          opList[targetIdx]->GetOpcodeStr().c_str(), opList[targetIdx]->GetOpMagic());
        opList.erase(opList.begin() + allocIdx);
        opList.insert(opList.begin() + targetIdx, allocOp);

        for (auto& [id, idx] : memIdToAllocIdx) {
            (void)id;
            if (idx >= static_cast<size_t>(targetIdx) && idx < allocIdx) {
                idx++;
            }
        }
        memIdToAllocIdx[memId] = static_cast<size_t>(targetIdx);
    }
    return true;
}

Status OoOSchedule::ModifyAllocOrder(std::vector<Operation*>& opList)
{
    std::unordered_map<uint64_t, size_t> memIdToAllocIdx;
    if (BuildMemIdToAllocIdx(opList, memIdToAllocIdx) != SUCCESS) {
        return FAILED;
    }

    for (int i = static_cast<int>(opList.size()) - 1; i >= 0; i--) {
        Operation* op = opList[i];
        if (IsAllocOpCode(op->GetOpcode())) {
            continue;
        }

        auto outOpd = op->GetOutputOperand(0);
        if (outOpd == nullptr || outOpd->GetMemoryTypeOriginal() == MemoryType::MEM_DEVICE_DDR) {
            continue;
        }

        uint64_t memId = outOpd->memoryrange.memId;
        auto it = memIdToAllocIdx.find(memId);
        if (it == memIdToAllocIdx.end()) {
            continue;
        }

        if (!MoveAllocBeforeOp(opList, it->second, i, memIdToAllocIdx, memId)) {
            return FAILED;
        }
    }
    return SUCCESS;
}

std::vector<ScheduleUnit> OoOSchedule::BuildScheduleUnits(const std::vector<TaskNode>& taskNodeList,
                                                          const std::vector<std::pair<int, int>>& cyclePairs)
{
    std::vector<ScheduleUnit> scheduleUnits;
    std::unordered_set<int> pairedIndices;

    for (const auto& pair : cyclePairs) {
        auto it1 = std::find_if(taskNodeList.begin(), taskNodeList.end(),
                                [&](const TaskNode& n) { return n.idx == pair.first; });
        auto it2 = std::find_if(taskNodeList.begin(), taskNodeList.end(),
                                [&](const TaskNode& n) { return n.idx == pair.second; });
        if (it1 != taskNodeList.end() && it2 != taskNodeList.end()) {
            ScheduleUnit unit;
            unit.mergedOps.insert(unit.mergedOps.end(), it1->opList_.begin(), it1->opList_.end());
            unit.mergedOps.insert(unit.mergedOps.end(), it2->opList_.begin(), it2->opList_.end());
            unit.earliestStartTime = std::min(it1->startTime, it2->startTime);
            unit.isMerged = true;
            pairedIndices.insert(pair.first);
            pairedIndices.insert(pair.second);
            scheduleUnits.push_back(std::move(unit));
        }
    }

    for (const auto& taskNode : taskNodeList) {
        if (pairedIndices.find(taskNode.idx) == pairedIndices.end()) {
            ScheduleUnit unit;
            unit.mergedOps = taskNode.opList_;
            unit.earliestStartTime = taskNode.startTime;
            scheduleUnits.push_back(std::move(unit));
        }
    }

    std::stable_sort(scheduleUnits.begin(), scheduleUnits.end(), [](const ScheduleUnit& a, const ScheduleUnit& b) {
        return a.earliestStartTime < b.earliestStartTime;
    });

    return scheduleUnits;
}

Status OoOSchedule::RunOnFunction(Function& function)
{
    APASS_LOG_INFO_F(Elements::Operation, "=============== START 2CoreSplit ===============");
    int64_t maxWorkeSpaceSize = 0;
    for (auto& program : function.rootFunc_->programs_) {
        auto opList = program.second->Operations(false).DuplicatedOpList();
        oriFunctions.emplace_back(program.second);
        if (IsAicpuProgram(opList)) {
            continue;
        }
        std::pair<uint64_t, Function*> programRef;
        programRef.first = program.first;
        programRef.second = program.second;
        if (Schedule(opList, function, programRef, maxWorkeSpaceSize) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "OoO schedule failed.");
            return FAILED;
        }
        DeadOperationEliminator eliminator;
        eliminator.EliminateOperation(*program.second, false, false);
    }
    for (auto& [programId, tracer] : tracerMap_) {
        (void)programId;
        tracer.Flush(GetPassFolder());
    }
    APASS_LOG_INFO_F(Elements::Operation, "=============== END 2CoreSplit ===============");
    return SUCCESS;
}

void OoOSchedule::DoHealthCheckAfter(Function& function, const std::string& folderPath)
{
    for (auto& [programId, check] : statisticMap_) {
        auto fileName = folderPath + '/' + check.jsonFileName + "_Block_Graph_Health_Report.json";
        auto it = function.rootFunc_->programs_.find(programId);
        if (it != function.rootFunc_->programs_.end()) {
            check.DoHealthCheck(it->second, fileName);
        }
    }
}

Status OoOSchedule::PreCheck(Function& function) { return checker.DoPreCheck(function); }

Status OoOSchedule::PostCheck(Function& function)
{
    checker.SetOriFunctions(oriFunctions);
    return checker.DoPostCheck(function);
}

void OoOSchedule::CollectMemoryTrace(MemoryTracer& tracer, Function& function, std::pair<uint64_t, Function*>& program)
{
    tracer.SetOutputPrefix(GetDumpFilePrefix(function, false, program.second, program.first));
    tracerMap_.emplace(program.first, std::move(tracer));
}

// PostRun 失败时不再执行，这里直接 flush trace；dumpGraph 打开时同步输出。
void OoOSchedule::FlushMemoryTraceOnFailure(MemoryTracer& tracer, Function& function,
                                            std::pair<uint64_t, Function*>& program)
{
    auto prefix = GetDumpFilePrefix(function, false, program.second, program.first);
    tracer.SetOutputPrefix(prefix);
    tracer.Flush(GetPassFolder());
    if (passDfxconfigs_.dumpGraph) {
        program.second->DumpJsonFile(GetPassFolder() + "/" + prefix + ".json");
    }
}

Status OoOSchedule::CheckAllocOp(std::vector<Operation*> list)
{
    std::map<int, Operation*> allocMap;
    for (const auto& op : list) {
        if (IsAllocOpCode(op->GetOpcode())) {
            size_t nonDdrCount = 0;
            for (auto o : op->GetOOperands()) {
                if (o->GetMemoryTypeOriginal() < MemoryType::MEM_DEVICE_DDR) {
                    nonDdrCount++;
                }
            }
            for (auto i : op->GetIOperands()) {
                if (i->GetMemoryTypeOriginal() < MemoryType::MEM_DEVICE_DDR) {
                    nonDdrCount++;
                }
            }
            if (nonDdrCount != 1) {
                APASS_LOG_ERROR_F(Elements::Operation, "%s InOutOperand size not equal to 1.", GetOpInfo(op).c_str());
                return FAILED;
            }
            UpdateAllocMap(op, allocMap);
        }
    }
    for (const auto& op : list) {
        if (!IsAllocOpCode(op->GetOpcode())) {
            UpdateAllocMap(op, allocMap);
        }
    }
    for (auto allocEntry : allocMap) {
        if (!IsAllocOpCode(allocEntry.second->GetOpcode())) {
            APASS_LOG_ERROR_F(Elements::Tensor, "%s Tensor[%d] is missing Alloc.", GetOpInfo(allocEntry.second).c_str(),
                              allocEntry.first);
            return FAILED;
        }
    }
    return SUCCESS;
}

void OoOSchedule::UpdateAllocMap(Operation* op, std::map<int, Operation*>& allocMap)
{
    for (auto outTensor : op->GetOOperands()) {
        if (outTensor->GetMemoryTypeOriginal() >= MemoryType::MEM_DEVICE_DDR) {
            continue;
        }
        int memId = outTensor->memoryrange.memId;
        if (allocMap.find(memId) == allocMap.end()) {
            allocMap[memId] = op;
        }
    }
    for (auto inTensor : op->GetIOperands()) {
        if (inTensor->GetMemoryTypeOriginal() >= MemoryType::MEM_DEVICE_DDR) {
            continue;
        }
        int memId = inTensor->memoryrange.memId;
        if (allocMap.find(memId) == allocMap.end()) {
            allocMap[memId] = op;
        }
    }
}
} // namespace npu::tile_fwk
