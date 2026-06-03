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

inline bool IsMixGraph(const std::vector<Operation*>& opList)
{
    bool hasAIC = false;
    bool hasAIV = false;
    for (auto opPtr : opList) {
        if (OpcodeManager::Inst().GetCoreType(opPtr->GetOpcode()) == OpCoreType::AIC) {
            hasAIC = true;
        } else if (OpcodeManager::Inst().GetCoreType(opPtr->GetOpcode()) == OpCoreType::AIV) {
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
    std::vector<Operation*> newTaskList;
    for (auto op : opList) {
        if (std::find(taskList.begin(), taskList.end(), op) != taskList.end()) {
            newTaskList.push_back(op);
        }
    }
    taskList = newTaskList;
}

void OoOSchedule::CollectStatistic(OoOScheduleStatistic& oooHealthCheck,
    Function& function, std::pair<uint64_t, Function*>& program)
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

bool OoOSchedule::IsBoundary(Operation* op)
{
    if (op->GetOpcode() == Opcode::OP_L0C_COPY_UB || op->GetOpcode() == Opcode::OP_L1_COPY_UB ||
        op->GetOpcode() == Opcode::OP_UB_COPY_L1) {
        return true;
    }
    return false;
}

Status OoOSchedule::AdvanceAlloc(std::vector<Operation*>& opList, Operation* op, size_t& index)
{
    APASS_LOG_DEBUG_F(Elements::Operation, "Advance alloc of op: %s[%d]", op->GetOpcodeStr().c_str(), op->GetOpMagic());
    for (auto& preOp : op->GetOutputOperand(0)->GetProducers()) {
        if (preOp->GetOpcodeStr().find("ALLOC") != std::string::npos) {
            auto it = std::find(opList.begin(), opList.end(), preOp);
            if (it == opList.end()) {
                APASS_LOG_ERROR_F(Elements::Operation, "Cannot find the alloc of boundaryop.");
                return FAILED;
            }
            size_t allocIndex = std::distance(opList.begin(), it);
            if (allocIndex > index) {
                APASS_LOG_DEBUG_F(Elements::Operation, "alloc index: %zu, op index: %zu", allocIndex, index);
                std::rotate(opList.begin() + index, opList.begin() + allocIndex, opList.begin() + allocIndex + 1);
                index++;
                return SUCCESS;
            }
        }
    }
    return SUCCESS;
}

Status OoOSchedule::ModifyBoundaryOrder(std::vector<Operation*>& opList)
{
    size_t i = 0;
    while (i < opList.size()) {
        if (IsBoundary(opList[i])) {
            if (AdvanceAlloc(opList, opList[i], i) != SUCCESS) {
                APASS_LOG_ERROR_F(Elements::Operation, "AdvanceAlloc failed.");
                return FAILED;
            }
        }
        i++;
    }
    return SUCCESS;
}

std::vector<ScheduleUnit> OoOSchedule::BuildScheduleUnits(const std::vector<TaskNode> &taskNodeList,
    const std::vector<std::pair<int, int>> &cyclePairs, std::vector<Operation*> &opList)
{
    std::vector<ScheduleUnit> scheduleUnits;
    std::unordered_set<int> pairedIndices;

    for (const auto &pair : cyclePairs) {
        auto it1 = std::find_if(taskNodeList.begin(), taskNodeList.end(),
            [&](const TaskNode& n) { return n.idx == pair.first; });
        auto it2 = std::find_if(taskNodeList.begin(), taskNodeList.end(),
            [&](const TaskNode& n) { return n.idx == pair.second; });
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

    for (const auto &taskNode : taskNodeList) {
        if (pairedIndices.find(taskNode.idx) == pairedIndices.end()) {
            ScheduleUnit unit;
            unit.mergedOps = taskNode.opList_;
            SortTaskList(opList, unit.mergedOps);
            unit.earliestStartTime = taskNode.startTime;
            scheduleUnits.push_back(std::move(unit));
        }
    }

    std::sort(scheduleUnits.begin(), scheduleUnits.end(),
        [](const ScheduleUnit& a, const ScheduleUnit& b) {
            return a.earliestStartTime < b.earliestStartTime;
        });

    return scheduleUnits;
}

Status OoOSchedule::MixSchedule(
    std::vector<Operation*>& opList, Function& function, std::pair<uint64_t, Function*>& program,
    int64_t& maxWorkeSpaceSize)
{
    APASS_LOG_INFO_F(Elements::Operation, "=============== START MixSchedule ===============");
    TaskSpliter spliter;
    // 对 taskNode.opList_ 进行排序，并返回预估 latency，随后完成 core schedule 与子图合并。
    if (EstimateTaskLatencyAndSchedule(spliter, opList) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "EstimateTaskLatencyAndSchedule failed.");
        return FAILED;
    }
    std::unordered_map<Operation*, CoreLocationType> opCoreMap;
    // 传入 taskNode 序列，对全部 opList 重新拼装并构建 opCoreMap。
    if (BuildMixedScheduleOps(spliter, opList, opCoreMap) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "BuildMixedScheduleOps failed.");
        return FAILED;
    }
    if (ModifyBoundaryOrder(opList) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "ModifyBoundaryOrder failed.");
        return FAILED;
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

Status OoOSchedule::EstimateTaskLatencyAndSchedule(TaskSpliter& spliter, std::vector<Operation*>& opList)
{
    static const std::unordered_map<TargetCoreType, std::string> targetToString{
        {TargetCoreType::AIC, "AIC"},
        {TargetCoreType::AIV0, "AIV0"},
        {TargetCoreType::AIV1, "AIV1"},
        {TargetCoreType::UNKNOWN, "UNKNOWN"}};

    spliter.SplitGraph(opList);
    for (auto& taskNode : spliter.GetTaskGraph().tasks) {
        if (SortAndLatencyEstimate(opList, taskNode.opList_, taskNode.latency) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "SortAndLatencyEstimate failed, taskNode[%d].", taskNode.idx);
            return FAILED;
        }
    }
    CoreScheduler coreScheduler;
    coreScheduler.Schedule(spliter.GetTaskGraph(), 10); // BruteForce threshold is 10
    for (auto& taskNode : spliter.GetTaskGraph().tasks) {
        APASS_LOG_INFO_F(Elements::Operation, "eval task %d on %s: %d - %d.", taskNode.idx,
            targetToString.at(taskNode.targetCoreType).c_str(), taskNode.startTime, taskNode.endTime);
    }
    spliter.MarkInternalSubgraphID();
    return SUCCESS;
}

Status OoOSchedule::BuildMixedScheduleOps(TaskSpliter& spliter, std::vector<Operation*>& opList,
    std::unordered_map<Operation*, CoreLocationType>& opCoreMap)
{
    auto taskNodeList = spliter.GetTaskGraph().tasks;
    std::sort(taskNodeList.begin(), taskNodeList.end(), [](const TaskNode& a, const TaskNode& b) {
        return a.startTime < b.startTime;
    });
    for (const auto& taskNode : taskNodeList) {
        if (UpdateOpCoreMap(taskNode, opCoreMap) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "UpdateOpCoreMap failed.");
            return FAILED;
        }
    }
    // 成环的 TaskNode
    auto cyclePairs = spliter.GetCycledTaskNodePairs();
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

Status OoOSchedule::RecordLastUseMemory(Function& function)
{
    APASS_LOG_INFO_F(Elements::Function, "===> Start RecordLastUseMemory.");
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
    std::unordered_map<Operation*, std::vector<int>> opInputIdxMap;
    std::unordered_set<Opcode> reduceOp = {
        Opcode::OP_ROWSUM_SINGLE, Opcode::OP_ROWMAX_SINGLE, Opcode::OP_ROWMIN_SINGLE};
    for (auto& entry : lastUseMap_) {
        auto lastUseOp = entry.second;
        auto lastUseTensor = entry.first;
        if (LASTUSE_OPS.find(lastUseOp->GetOpcode()) == LASTUSE_OPS.end()) {
            continue; // 针对非LASTUSE_OPS中的op不做处理，仅处理LASTUSE_OPS中的op
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
        op->SetAttribute(OpAttributeKey::lastUse, opInputIdxMap[op]);
    }
    APASS_LOG_INFO_F(Elements::Function, "===> End RecordLastUseMemory.");
    return SUCCESS;
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
        if (Platform::Instance().GetSoc().GetNPUArch() != NPUArch::DAV_3510 || !IsMixGraph(opList)) {
            if (NonMixSchedule(opList, function, programRef, maxWorkeSpaceSize) != SUCCESS) {
                APASS_LOG_ERROR_F(Elements::Operation, "NonMix OoO schedule failed.");
                return FAILED;
            }
            DeadOperationEliminator eliminator;
            eliminator.EliminateOperationAndNotSortAfterErase(*program.second);
            programRef.second = program.second;
            continue;
        }
        if (MixSchedule(opList, function, programRef, maxWorkeSpaceSize) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "Mix OoO schedule failed.");
            return FAILED;
        }
        DeadOperationEliminator eliminator;
        eliminator.EliminateOperationAndNotSortAfterErase(*program.second);
        programRef.second = program.second;
    }
    if (RecordLastUseMemory(function) == FAILED) {
        APASS_LOG_ERROR_F(Elements::Function, "Run RecordLastUseMemory Failed.");
        return FAILED;
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

void OoOSchedule::CollectMemoryTrace(MemoryTracer& tracer,
    Function& function, std::pair<uint64_t, Function*>& program)
{
    tracer.SetOutputPrefix(GetDumpFilePrefix(function, false, program.second, program.first));
    tracerMap_.emplace(program.first, std::move(tracer));
}

// PostRun is skipped on FAILED — flush trace inline; mirror dump if dumpGraph is on.
void OoOSchedule::FlushMemoryTraceOnFailure(MemoryTracer& tracer,
    Function& function, std::pair<uint64_t, Function*>& program)
{
    auto prefix = GetDumpFilePrefix(function, false, program.second, program.first);
    tracer.SetOutputPrefix(prefix);
    tracer.Flush(GetPassFolder());
    if (passDfxconfigs_.dumpGraph) {
        program.second->DumpJsonFile(GetPassFolder() + "/" + prefix + ".json");
    }
}
} // namespace npu::tile_fwk
