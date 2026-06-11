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
 * \file schedule_ooo.cpp
 * \brief
 */

#include "schedule_ooo.h"
#include "passes/pass_log/pass_log.h"
#include "passes/pass_utils/dead_operation_eliminate.h"

#ifndef MODULE_NAME
#define MODULE_NAME "OoOSchedule"
#endif

namespace npu::tile_fwk {

// TODO(dualdst-switch): 临时开关。后续接入 PassConfigs.enable_dual_dst (需要修改
// framework/src/interface/configs/* 下的文件)再删掉这一行,改读 passDfxconfigs_。
// 当前调试时手动改成 true 即可启用 dualdst 融合;Mix 路径才生效。
static constexpr bool kTempEnableDualDst = false;

bool OoOSchedule::IsAicpuProgram(std::vector<Operation*> opList)
{
    for (auto& op : opList) {
        if (op->GetCoreType() == CoreType::AICPU) {
            return true;
        }
    }
    return false;
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

void OoOSchedule::SortTaskList(std::vector<Operation*>& opList, std::vector<Operation*>& taskList)
{
    std::unordered_set<Operation*> taskSet(taskList.begin(), taskList.end());
    std::vector<Operation*> newTaskList;
    newTaskList.reserve(taskList.size());
    for (auto op : opList) {
        if (taskSet.count(op)) {
            newTaskList.push_back(op);
        }
    }
    taskList = std::move(newTaskList);
}

void OoOSchedule::CollectStatistic(
    OoOScheduleStatistic& oooHealthCheck, Function& function, std::pair<uint64_t, Function*>& program)
{
    if (passDfxconfigs_.healthCheck) {
        oooHealthCheck.SetOutputPrefix(GetDumpFilePrefix(function, false, program.second, program.first));
        statisticMap_.insert({program.first, oooHealthCheck});
    }
}

Status OoOSchedule::NonMixSchedule(
    std::vector<Operation*>& opList, Function& function, std::pair<uint64_t, Function*>& program,
    int64_t& maxWorkeSpaceSize)
{
    // 直接对oplist进行GenSpill和mainLoop
    APASS_LOG_INFO_F(Elements::Operation, "=============== START NonMixSchedule ===============");
    OoOScheduler oooSchedule(*program.second);
    OoOScheduleStatistic oooHealthCheck;
    MemoryTracer oooMemoryTrace;
    if (passDfxconfigs_.healthCheck) {
        oooSchedule.AddObserver(&oooHealthCheck);
    }
    if (passDfxconfigs_.dumpGraph) {
        oooSchedule.AddObserver(&oooMemoryTrace);
    }
    if (oooSchedule.Schedule(opList) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "Non-mixGraph schedule failed.");
        if (passDfxconfigs_.dumpGraph) {
            FlushMemoryTraceOnFailure(oooMemoryTrace, function, program);
        }
        return FAILED;
    }
    APASS_LOG_INFO_F(Elements::Operation, "Subgraph[%zu] OOOSchedule end.", program.first);
    // dualdst fuse 在 FuseDualDstPairs 内部以 EraseOperations(false, true) 刷新 opPosition_,
    // 因此这里走默认 needRefresh=false 即可。
    program.second->ScheduleBy(oooSchedule.GetNewOperations());
    program.second->RecordOOOSeq();
    RescheduleUtils::UpdateTensorConsProd(program.second);
    maxWorkeSpaceSize = std::max(maxWorkeSpaceSize, (*program.second).GetStackWorkespaceSize());
    function.SetStackWorkespaceSize(maxWorkeSpaceSize);
    CollectStatistic(oooHealthCheck, function, program);
    if (passDfxconfigs_.dumpGraph) {
        CollectMemoryTrace(oooMemoryTrace, function, program);
    }
    return SUCCESS;
}

Status OoOSchedule::BuildMemIdToAllocIdx(
    const std::vector<Operation*>& opList, std::unordered_map<uint64_t, size_t>& memIdToAllocIdx)
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

bool OoOSchedule::MoveAllocBeforeOp(
    std::vector<Operation*>& opList, size_t allocIdx, int targetIdx,
    std::unordered_map<uint64_t, size_t>& memIdToAllocIdx, uint64_t memId)
{
    if (allocIdx >= opList.size()) {
        APASS_LOG_ERROR_F(
            Elements::Operation, "ModifyAllocOrder: allocIdx %zu out of range (size %zu).", allocIdx, opList.size());
        return false;
    }
    if (allocIdx > static_cast<size_t>(targetIdx)) {
        Operation* allocOp = opList[allocIdx];
        APASS_LOG_DEBUG_F(
            Elements::Operation, "Move %s[%d] from index[%zu] to index[%d] before %s[%d]",
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

std::vector<ScheduleUnit> OoOSchedule::BuildScheduleUnits(
    const std::vector<TaskNode>& taskNodeList, const std::vector<std::pair<int, int>>& cyclePairs,
    std::vector<Operation*>& opList)
{
    std::vector<ScheduleUnit> scheduleUnits;
    std::unordered_set<int> pairedIndices;

    for (const auto& pair : cyclePairs) {
        auto it1 = std::find_if(
            taskNodeList.begin(), taskNodeList.end(), [&](const TaskNode& n) { return n.idx == pair.first; });
        auto it2 = std::find_if(
            taskNodeList.begin(), taskNodeList.end(), [&](const TaskNode& n) { return n.idx == pair.second; });
        if (it1 != taskNodeList.end() && it2 != taskNodeList.end()) {
            ScheduleUnit unit;
            unit.mergedOps.insert(unit.mergedOps.end(), it1->opList_.begin(), it1->opList_.end());
            unit.mergedOps.insert(unit.mergedOps.end(), it2->opList_.begin(), it2->opList_.end());
            SortTaskList(opList, unit.mergedOps);
            unit.earliestStartTime = std::min(it1->startTime, it2->startTime);

            pairedIndices.insert(pair.first);
            pairedIndices.insert(pair.second);
            scheduleUnits.push_back(std::move(unit));
        }
    }

    for (const auto& taskNode : taskNodeList) {
        if (pairedIndices.find(taskNode.idx) == pairedIndices.end()) {
            ScheduleUnit unit;
            unit.mergedOps = taskNode.opList_;
            SortTaskList(opList, unit.mergedOps);
            unit.earliestStartTime = taskNode.startTime;
            scheduleUnits.push_back(std::move(unit));
        }
    }

    std::sort(scheduleUnits.begin(), scheduleUnits.end(), [](const ScheduleUnit& a, const ScheduleUnit& b) {
        return a.earliestStartTime < b.earliestStartTime;
    });

    return scheduleUnits;
}

std::string GetOpInfo(Operation* op)
{
    if (op == nullptr)
        return "nullptr";
    return op->GetOpcodeStr() + "[" + std::to_string(op->GetOpMagic()) + "]";
}

Status OoOSchedule::MixSchedule(
    std::vector<Operation*>& opList, Function& function, std::pair<uint64_t, Function*>& program,
    int64_t& maxWorkeSpaceSize)
{
    APASS_LOG_INFO_F(Elements::Operation, "=============== START MixSchedule ===============");
    TaskSplitter splitter;
    // 对 taskNode.opList_ 进行排序，并返回预估 latency，随后完成 core schedule 与子图合并。
    if (EstimateTaskLatencyAndSchedule(splitter, opList) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "EstimateTaskLatencyAndSchedule failed.");
        return FAILED;
    }
    std::unordered_map<Operation*, CoreLocationType> opCoreMap;
    // 传入 taskNode 序列，对全部 opList 重新拼装并构建 opCoreMap。
    if (BuildMixedScheduleOps(splitter, opList, opCoreMap) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "BuildMixedScheduleOps failed.");
        return FAILED;
    }
    if (ModifyAllocOrder(opList) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "ModifyAllocOrder failed.");
        return FAILED;
    }
    APASS_LOG_DEBUG_F(Elements::Operation, "+++ complete oplist");
    for (size_t i = 0; i < opList.size(); i++) {
        APASS_LOG_DEBUG_F(Elements::Operation, "%zu    op : %s", i, GetOpInfo(opList[i]).c_str());
    }
    OoOScheduler oooSchedule(*program.second);
    oooSchedule.SetEnableDualDst(kTempEnableDualDst);
    OoOScheduleStatistic oooHealthCheck;
    MemoryTracer oooMemoryTrace;
    if (passDfxconfigs_.healthCheck) {
        oooSchedule.AddObserver(&oooHealthCheck);
    }
    if (passDfxconfigs_.dumpGraph) {
        oooSchedule.AddObserver(&oooMemoryTrace);
    }
    if (oooSchedule.Schedule(opList, opCoreMap, CORE_INIT_CONFIGS_HARDWARE_TWO_AIV) != SUCCESS) {
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
    // dualdst fuse 在 FuseDualDstPairs 内部以 EraseOperations(false, true) 刷新 opPosition_,
    // 因此这里走默认 needRefresh=false 即可。
    program.second->ScheduleBy(oooSchedule.GetNewOperations());
    program.second->RecordOOOSeq();
    RescheduleUtils::UpdateTensorConsProd(program.second);
    maxWorkeSpaceSize = std::max(maxWorkeSpaceSize, (*program.second).GetStackWorkespaceSize());
    function.SetStackWorkespaceSize(maxWorkeSpaceSize);
    return SUCCESS;
}

Status OoOSchedule::EstimateTaskLatencyAndSchedule(TaskSplitter& splitter, std::vector<Operation*>& opList)
{
    static const std::unordered_map<TargetCoreType, std::string> targetToString{
        {TargetCoreType::AIC, "AIC"},
        {TargetCoreType::AIV0, "AIV0"},
        {TargetCoreType::AIV1, "AIV1"},
        {TargetCoreType::UNKNOWN, "UNKNOWN"}};

    splitter.SplitGraph(opList);
    for (auto& taskNode : splitter.GetTaskGraph().tasks) {
        if (SortAndLatencyEstimate(opList, taskNode.opList_, taskNode.latency) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "SortAndLatencyEstimate failed, taskNode[%d].", taskNode.idx);
            return FAILED;
        }
    }
    CoreScheduler coreScheduler;
    coreScheduler.Schedule(splitter.GetTaskGraph());
    APASS_LOG_DEBUG_F(Elements::Operation, "============>after schedule");
    for (size_t i = 0; i < splitter.GetTaskGraph().tasks.size(); i++) {
        auto& t = splitter.GetTaskGraph().tasks[i];
        APASS_LOG_DEBUG_F(
            Elements::Operation, "i: %zu =======eval task %d on %s: starttime %d - %d.", i, t.idx,
            targetToString.at(t.targetCoreType).c_str(), t.startTime, t.endTime);
    }
    for (size_t i = 0; i < splitter.GetTaskGraph().tasks.size(); i++) {
        auto& t = splitter.GetTaskGraph().tasks[i];
        APASS_LOG_DEBUG_F(
            Elements::Operation, "i: %zu =======task %d out task %s.", i, t.idx, IntVecToStr(t.outTasks).c_str());
    }
    splitter.MarkInternalSubgraphID();
    return SUCCESS;
}

Status OoOSchedule::BuildMixedScheduleOps(
    TaskSplitter& splitter, std::vector<Operation*>& opList, std::unordered_map<Operation*, CoreLocationType>& opCoreMap)
{
    static const std::unordered_map<TargetCoreType, std::string> targetToString{
        {TargetCoreType::AIC, "AIC"},
        {TargetCoreType::AIV0, "AIV0"},
        {TargetCoreType::AIV1, "AIV1"},
        {TargetCoreType::UNKNOWN, "UNKNOWN"}};
    auto taskNodeList = splitter.GetTaskGraph().tasks;
    std::sort(taskNodeList.begin(), taskNodeList.end(), [](const TaskNode& a, const TaskNode& b) {
        return a.startTime < b.startTime;
    });
    APASS_LOG_DEBUG_F(Elements::Operation, "============>after sort BuildMixedScheduleOps");
    for (size_t i = 0; i < taskNodeList.size(); i++) {
        APASS_LOG_DEBUG_F(
            Elements::Operation, "i: %zu task %d on %s: starttime %d - %d.", i, taskNodeList[i].idx,
            targetToString.at(taskNodeList[i].targetCoreType).c_str(), taskNodeList[i].startTime,
            taskNodeList[i].endTime);
        for (auto op : taskNodeList[i].opList_) {
            APASS_LOG_DEBUG_F(Elements::Operation, "op : %s", GetOpInfo(op).c_str());
        }
    }
    for (const auto& taskNode : taskNodeList) {
        if (UpdateOpCoreMap(taskNode, opCoreMap) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "UpdateOpCoreMap failed.");
            return FAILED;
        }
    }
    // 成环的 TaskNode
    auto cyclePairs = splitter.GetCycledTaskNodePairs();
    auto scheduleUnits = BuildScheduleUnits(taskNodeList, cyclePairs, opList);
    std::vector<Operation*> operations;
    for (const auto& unit : scheduleUnits) {
        operations.insert(operations.end(), unit.mergedOps.begin(), unit.mergedOps.end());
    }
    opList = std::move(operations);
    return SUCCESS;
}

Status OoOSchedule::UpdateOpCoreMap(
    const TaskNode& taskNode, std::unordered_map<Operation*, CoreLocationType>& opCoreMap)
{
    for (auto op : taskNode.opList_) {
        if (taskNode.targetCoreType == TargetCoreType::UNKNOWN) {
            APASS_LOG_ERROR_F(Elements::Operation, "CoreType is not AIC, AIV0 or AIV1");
            return FAILED;
        }
        opCoreMap[op] = targetCoreTypeMap.at(taskNode.targetCoreType);
    }
    return SUCCESS;
}

Status OoOSchedule::SortAndLatencyEstimate(
    std::vector<Operation*>& opList, std::vector<Operation*>& taskOpList, int& latency)
{
    APASS_LOG_INFO_F(Elements::Operation, "=======>start SortAndLatencyEstimate");
    SortTaskList(opList, taskOpList);
    LatencyEstimator latencyEstimator(taskOpList, opList);
    if (latencyEstimator.LatencyEstimatorMainLoop() != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "SortAndLatencyEstimate LatencyEstimatorMainLoop failed.");
        return FAILED;
    }
    latency = latencyEstimator.clock;
    APASS_LOG_INFO_F(Elements::Operation, "=======>end SortAndLatencyEstimate");
    return SUCCESS;
}

Status OoOSchedule::CollectLastUseInfo(Function& function)
{
    lastUseMap_.clear();
    APASS_LOG_INFO_F(Elements::Function, "===> Start CollectLastUseInfo.");
    for (auto& program : function.rootFunc_->programs_) {
        auto opList = program.second->Operations(false);
        for (size_t opIdx = 0; opIdx < opList.size(); opIdx++) {
            Operation* op = &opList[opIdx];
            if (LASTUSE_OPS.find(op->GetOpcode()) != LASTUSE_OPS.end()) {
                int tensorSize = op->GetIOperands().size() + op->GetOOperands().size();
                std::vector<int> initVec(tensorSize, false);
                op->SetAttribute(OpAttributeKey::lastUse, initVec);
            }
            for (size_t inputIdx = 0; inputIdx < op->GetIOperands().size(); inputIdx++) {
                auto inTensor = op->GetInputOperand(inputIdx);
                lastUseMap_[inTensor] = op;
            }
        }
    }
    APASS_LOG_INFO_F(Elements::Function, "===> End CollectLastUseInfo.");
    return SUCCESS;
}

void OoOSchedule::SetLastUseAttributes()
{
    APASS_LOG_INFO_F(Elements::Function, "===> Start SetLastUseAttributes.");
    std::unordered_map<Operation*, std::vector<int>> opInputIdxMap;
    std::unordered_set<Opcode> reduceOp = {
        Opcode::OP_ROWSUM_SINGLE, Opcode::OP_ROWMAX_SINGLE, Opcode::OP_ROWMIN_SINGLE};
    for (auto& entry : lastUseMap_) {
        auto lastUseOp = entry.second;
        auto lastUseTensor = entry.first;
        if (LASTUSE_OPS.find(lastUseOp->GetOpcode()) == LASTUSE_OPS.end()) {
            continue;
        }
        if (opInputIdxMap.find(lastUseOp) == opInputIdxMap.end()) {
            int tensorSize = lastUseOp->GetIOperands().size() + lastUseOp->GetOOperands().size();
            std::vector<int> tensorIdxVec(tensorSize, false);
            int inputIdx = lastUseOp->GetIOperandIndex(lastUseTensor) + lastUseOp->GetOOperands().size();
            if (reduceOp.find(lastUseOp->GetOpcode()) != reduceOp.end() && inputIdx == tensorSize - 1) {
                tensorIdxVec[inputIdx] = false;
            } else {
                tensorIdxVec[inputIdx] = true;
            }
            opInputIdxMap[lastUseOp] = tensorIdxVec;
        } else {
            int inputIdx = lastUseOp->GetIOperandIndex(lastUseTensor) + lastUseOp->GetOOperands().size();
            opInputIdxMap[lastUseOp][inputIdx] = true;
        }
    }
    for (auto& entry : opInputIdxMap) {
        auto op = entry.first;
        if (op->HasAttribute(OpAttributeKey::brcOperand) || op->GetOpcode() == Opcode::OP_EXPAND) {
            APASS_LOG_INFO_F(Elements::Operation, "Skip Process OP_%s[%d] LastUse Attribute",
                op->GetOpcodeStr().c_str(), op->GetOpMagic());
            continue;
        }
        op->SetAttribute(OpAttributeKey::lastUse, opInputIdxMap[op]);
    }
    APASS_LOG_INFO_F(Elements::Function, "===> End SetLastUseAttributes.");
}

Status OoOSchedule::RunOnFunction(Function& function)
{
    APASS_LOG_INFO_F(Elements::Operation, "=============== START 2CoreSplit ===============");
    int64_t maxWorkeSpaceSize = 0;
    for (auto& program : function.rootFunc_->programs_) {
        auto opList = program.second->Operations(false).DuplicatedOpList();
        oriFunctions.emplace_back(program.second);
        // ooo不处理aicpu子图
        if (IsAicpuProgram(opList)) {
            continue;
        }
        OptimizeSort optimizeSort(opList, *program.second);
        if (optimizeSort.SortOps() != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "Global sortOps failed");
            return FAILED;
        }
        // 全局排序的序列
        opList = optimizeSort.operations;
        std::pair<uint64_t, Function*> programRef;
        programRef.first = program.first;
        programRef.second = program.second;
        auto npuArch = Platform::Instance().GetSoc().GetNPUArch();
        bool isMix = IsMixGraph(opList);
        if (npuArch != NPUArch::DAV_3510 || !isMix) {
            if (NonMixSchedule(opList, function, programRef, maxWorkeSpaceSize) != SUCCESS) {
                APASS_LOG_ERROR_F(Elements::Operation, "NonMix OoO schedule failed.");
                return FAILED;
            }
            DeadOperationEliminator eliminator;
            eliminator.EliminateOperation(*program.second, false, false);
            programRef.second = program.second;
            continue;
        }
        if (MixSchedule(opList, function, programRef, maxWorkeSpaceSize) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "Mix OoO schedule failed.");
            return FAILED;
        }
        DeadOperationEliminator eliminator;
        eliminator.EliminateOperation(*program.second, false, false);
        programRef.second = program.second;
    }
    if (CollectLastUseInfo(function) == FAILED) {
        APASS_LOG_ERROR_F(Elements::Function, "Run CollectLastUseInfo Failed.");
        return FAILED;
    }
    SetLastUseAttributes();
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

// PostRun is skipped on FAILED — flush trace inline; mirror dump if dumpGraph is on.
void OoOSchedule::FlushMemoryTraceOnFailure(
    MemoryTracer& tracer, Function& function, std::pair<uint64_t, Function*>& program)
{
    auto prefix = GetDumpFilePrefix(function, false, program.second, program.first);
    tracer.SetOutputPrefix(prefix);
    tracer.Flush(GetPassFolder());
    if (passDfxconfigs_.dumpGraph) {
        program.second->DumpJsonFile(GetPassFolder() + "/" + prefix + ".json");
    }
}
} // namespace npu::tile_fwk
