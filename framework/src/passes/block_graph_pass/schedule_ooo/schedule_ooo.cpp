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

#ifndef MODULE_NAME
#define MODULE_NAME "OoOSchedule"
#endif

namespace npu::tile_fwk {

bool OoOSchedule::IsAicpuProgram(std::vector<Operation *> opList) {
    for (auto &op : opList) {
        if (op->GetCoreType() == CoreType::AICPU) {
            return true;
        }
    }
    return false;
}

inline bool IsMixGraph(const std::vector<Operation*> &opList) {
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

void OoOSchedule::SortTaskList(std::vector<Operation*> &opList, std::vector<Operation*> &taskList) {
    std::vector<Operation*> newTaskList;
    for (auto op : opList) {
        if (std::find(taskList.begin(), taskList.end(), op) != taskList.end()) {
            newTaskList.push_back(op);
        }
    }
    taskList = newTaskList;
}

void OoOSchedule::OoOHealthCheck(OoOScheduler &oooSchedule, Function &function, std::pair<uint64_t, Function*> &program) {
    if (oooSchedule.oooCheck.doHealthCheck) {
        oooSchedule.oooCheck.workspaceOffset = oooSchedule.workspaceOffset;
        oooSchedule.oooCheck.clock = oooSchedule.clock;
        oooSchedule.oooCheck.jsonFileName = GetDumpFilePrefix(function, false, program.second, program.first);
        schedulerMap.insert({program.first, oooSchedule});
    }
}

Status OoOSchedule::NonMixSchedule(std::vector<Operation*> &opList, Function &function,
    std::pair<uint64_t, Function*> &program, int &maxWorkeSpaceSize) {
    // 直接对oplist进行GenSpill和mainLoop
    OoOScheduler oooSchedule(*program.second, combineAxis);
    if (oooSchedule.Schedule(opList) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "Non-mixGraph schedule failed.");
        return FAILED;
    }
    APASS_LOG_INFO_F(Elements::Operation, "Subgraph[%d] OOOSchedule end.", program.first);
    program.second->ScheduleBy(oooSchedule.GetNewOperations());
    program.second->RecordOOOSeq();
    RescheduleUtils::UpdateTensorConsProd(program.second);
    maxWorkeSpaceSize = std::max(maxWorkeSpaceSize, (*program.second).GetStackWorkespaceSize());
    function.SetStackWorkespaceSize(maxWorkeSpaceSize);
    OoOHealthCheck(oooSchedule, function, program);
    return SUCCESS;
}

Status OoOSchedule::MixSchedule(std::vector<Operation*> &opList, Function &function,
    std::pair<uint64_t, Function*> &program, int &maxWorkeSpaceSize) {
    std::unordered_map<TargetCoreType, std::string>  targetToString{{TargetCoreType::AIC, "AIC"}, {TargetCoreType::AIV0, "AIV0"}, {TargetCoreType::AIV1, "AIV1"}, {TargetCoreType::UNKNOWN, "UNKNOWN"}};
    TaskSpliter spliter;
    spliter.SplitGraph(opList);
    for (auto &taskNode : spliter.GetTaskGraph().tasks) {
        // 对taskNode.opList_进行排序，并返回预估的latency
        if (SortAndLatencyEstimate(opList, taskNode.opList_, taskNode.latency) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "SortAndLatencyEstimate failed, taskNode[%d].", taskNode.idx);
            return FAILED;
        }
    }
    CoreScheduler coreScheduler;
    coreScheduler.Schedule(spliter.GetTaskGraph(), 10); // BruteForce threshold is 10
    for (auto &taskNode : spliter.GetTaskGraph().tasks) {
        APASS_LOG_INFO_F(Elements::Operation,  "eval task %d on %s: %d - %d.", taskNode.idx, targetToString[taskNode.targetCoreType].c_str(), taskNode.startTime, taskNode.endTime);
    }
    spliter.MergeTask();
    spliter.MarkInternalSubgraphID();
    // 传入一个taskNode序列 taskNodeList,对全部opList进行schedule
    auto taskNodeList = spliter.GetTaskGraph().tasks;
    std::sort(taskNodeList.begin(), taskNodeList.end(), [](const TaskNode& a, const TaskNode& b) {
        return a.startTime < b.startTime;
    });
    std::vector<Operation*> operations;
    std::unordered_map<Operation*, std::pair<OpCoreType, int>> opCoreMap;
    for (auto& taskNode : taskNodeList) {
        SortTaskList(taskNode.opList_, opList);
        UpdateOpCoreMap(taskNode, opCoreMap);
        operations.insert(operations.end(), taskNode.opList_.begin(), taskNode.opList_.end());
    }
    opList = operations;
    OoOScheduler oooSchedule(*program.second);
    if (oooSchedule.Schedule(opList, opCoreMap, CORE_INIT_CONFIGS_HARDWARE_TWO_AIV) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "Schedule failed.");
        return FAILED;
    }
    OoOHealthCheck(oooSchedule, function, program);
    APASS_LOG_INFO_F(Elements::Operation, "Subgraph[%d] OOOSchedule end.", program.first);
    program.second->ScheduleBy(oooSchedule.GetNewOperations());
    program.second->RecordOOOSeq();
    RescheduleUtils::UpdateTensorConsProd(program.second);
    maxWorkeSpaceSize = std::max(maxWorkeSpaceSize, (*program.second).GetStackWorkespaceSize());
    function.SetStackWorkespaceSize(maxWorkeSpaceSize);
    return SUCCESS;
}

Status OoOSchedule::UpdateOpCoreMap(const TaskNode &taskNode, std::unordered_map<Operation*, std::pair<OpCoreType, int>> &opCoreMap) {
    for (auto op : taskNode.opList_) {
        if (targetCoreTypeMap.find(taskNode.targetCoreType) == targetCoreTypeMap.end()) {
            APASS_LOG_ERROR_F(Elements::Operation, "CoreType is not AIC, AIV0 or AIV1");
            return FAILED;
        }
        opCoreMap[op] = targetCoreTypeMap.at(taskNode.targetCoreType);
    }
    return SUCCESS;
}

Status OoOSchedule::SortAndLatencyEstimate(std::vector<Operation*> &opList, std::vector<Operation*> &taskOpList,
    int &latency) {
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

Status OoOSchedule::RecordLastUseMemory(Function &function) {
    APASS_LOG_INFO_F(Elements::Function, "===> Start RecordLastUseMemory.");
    for (auto &program : function.rootFunc_->programs_) {
        auto opList = program.second->Operations(false);
        for (size_t opIdx = 0; opIdx < opList.size(); opIdx++) {
            Operation *op = &opList[opIdx];
            if (LASTUSE_OPS.find(op->GetOpcode()) == LASTUSE_OPS.end()) {
                APASS_LOG_INFO_F(Elements::Operation, "Op %s[%d] is not in LASTUSE_OPS, skip record last_use Attribute.", op->GetOpcodeStr().c_str(), op->GetOpMagic());
                continue;
            }
            int tensorSize = op->GetIOperands().size() + op->GetOOperands().size();
            std::vector<int> initVec(tensorSize, false);
            op->SetAttribute(OpAttributeKey::lastUse, initVec);
            for (size_t inputIdx = 0; inputIdx < op->GetIOperands().size(); inputIdx++) {
                auto inTensor = op->GetInputOperand(inputIdx);
                lastUseMap_[inTensor] = op;
            }
        }
    }
    std::unordered_map<Operation*, std::vector<int>> opInputIdxMap;
    std::unordered_set<Opcode> reduceOp = {Opcode::OP_ROWSUM_SINGLE, Opcode::OP_ROWMAX_SINGLE, Opcode::OP_ROWMIN_SINGLE};
    for (auto &entry : lastUseMap_) {
        auto lastUseOp = entry.second;
        auto lastUseTensor = entry.first;
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
    for (auto &entry : opInputIdxMap) {
        auto op = entry.first;
        op->SetAttribute(OpAttributeKey::lastUse, opInputIdxMap[op]);
    }
    APASS_LOG_INFO_F(Elements::Function, "===> End RecordLastUseMemory.");
    return SUCCESS;
}

Status OoOSchedule::RunOnFunction(Function &function) {
    combineAxis = function.paramConfigs_.combineAxis;
    forceCombineAxis = function.paramConfigs_.forceCombineAxis;
    APASS_LOG_INFO_F(Elements::Operation, "=============== START 2CoreSplit ===============");
    int maxWorkeSpaceSize = 0;
    for (auto &program : function.rootFunc_->programs_) {
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
            programRef.second = program.second;
            continue;
        }
        if (MixSchedule(opList, function, programRef, maxWorkeSpaceSize) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "Mix OoO schedule failed.");
            return FAILED;
        }
        programRef.second = program.second;
    }
    if (Platform::Instance().GetSoc().GetNPUArch() != NPUArch::DAV_3510) {
        APASS_LOG_INFO_F(Elements::Operation, "=============== END 2CoreSplit ===============");
        return SUCCESS;
    }
    if (RecordLastUseMemory(function) == FAILED) {
        APASS_LOG_ERROR_F(Elements::Function, "Run RecordLastUseMemory Failed.");
        return FAILED;
    }
    APASS_LOG_INFO_F(Elements::Operation, "=============== END 2CoreSplit ===============");
    return SUCCESS;
}

void OoOSchedule::DoHealthCheckAfter(Function &function, const std::string &folderPath) {
    for (auto &scheduler : schedulerMap) {
        auto fileName = folderPath + '/' + scheduler.second.oooCheck.jsonFileName + "_Block_Graph_Health_Report.json";
        auto it = function.rootFunc_->programs_.find(scheduler.first);
        if (it != function.rootFunc_->programs_.end()) {
            auto subFunc = it->second;
            scheduler.second.oooCheck.DoHealthCheck(subFunc, fileName);
        }
    }
}

Status OoOSchedule::PreCheck(Function &function) {
    return checker.DoPreCheck(function);
}

Status OoOSchedule::PostCheck(Function &function) {
    checker.SetOriFunctions(oriFunctions);
    return checker.DoPostCheck(function);
}
} // namespace npu::tile_fwk