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
 * \file ooo_schedule_statistic.cpp
 * \brief
 */

#include "ooo_schedule_statistic.h"
namespace npu {
namespace tile_fwk {

constexpr int32_t percent = 100;
constexpr float decimal = 10000.f; // 保留四位小数

void OoOSchedulerCheck::HealthCheckSpillInfo() {
    int spillIdx = 0;
    Json spill = Json::array();
    for (auto spillInfo : spillInfoVec) {
        Json spillDetails;
        spillDetails["spillEventIdx"] = spillIdx++;
        spillDetails["spillBufferType"] = MemoryTypeToString(spillInfo.spillType);
        spillDetails["bufferCurrentUsage"] = spillInfo.bufferCurrUsage;
        spillDetails["bufferCurrentUsageRate"] = static_cast<float>(spillInfo.bufferCurrUsage) / Platform::Instance().GetDie().GetMemoryLimit(spillInfo.spillType);
        spillDetails["bufferOccupiedByAllocSize"] = spillInfo.allocOccupiedSize;
        spillDetails["spillTensorSize"] = spillInfo.spillTensorSize;
        spillDetails["spillTensorMagic"] = spillInfo.spillTensorMagic;
        spillDetails["triggerTensorSize"] = spillInfo.triggerTensorSize;
        spillDetails["spillCopyoutSize"] = spillInfo.spillCopyoutSize;
        spill.emplace_back(spillDetails);
    }
    report["spillDetails"] = spill;
}

double OoOSchedulerCheck::FormatUsageRate(double value) {
    return std::round(value * decimal) / decimal;
}

Status OoOSchedulerCheck::HealthCheckOoOSchedule() {
    int64_t maxL0ASize = Platform::Instance().GetDie().GetMemoryLimit(MemoryType::MEM_L0A);
    int64_t maxL0BSize = Platform::Instance().GetDie().GetMemoryLimit(MemoryType::MEM_L0B);
    int64_t maxL0CSize = Platform::Instance().GetDie().GetMemoryLimit(MemoryType::MEM_L0C);
    int64_t maxUBSize = Platform::Instance().GetDie().GetMemoryLimit(MemoryType::MEM_UB);
    int64_t maxL1Size = Platform::Instance().GetDie().GetMemoryLimit(MemoryType::MEM_L1);
    if (maxL0ASize == 0 || maxL0BSize == 0 || maxL0CSize == 0 || maxUBSize == 0 || maxL1Size == 0) {
        ALOG_ERROR_F("Max buffer size is 0, HealthCheckOoOSchedule failed!");
        return FAILED;
    }
    // Workspace Info
    report["workspaceOffset"] = workspaceOffset;
    // Execution Info
    report["totalCycles"] = clock;
    // Pipe Usage Rate
    Json pipeUsageRate;
    if (clock == 0) {
        ALOG_ERROR_F("Clock is 0, HealthCheckOoOSchedule failed!");
        return FAILED;
    }
    pipeUsageRate["PIPE_S_Usage_Rate"] = FormatUsageRate(static_cast<double>(pipeUsageCount.at(PipeType::PIPE_S)) / clock * percent);
    pipeUsageRate["PIPE_V_Usage_Rate"] = FormatUsageRate(static_cast<double>(pipeUsageCount.at(PipeType::PIPE_V)) / clock * percent);
    pipeUsageRate["PIPE_M_Usage_Rate"] = FormatUsageRate(static_cast<double>(pipeUsageCount.at(PipeType::PIPE_M)) / clock * percent);
    pipeUsageRate["PIPE_MTE1_Usage_Rate"] = FormatUsageRate(static_cast<double>(pipeUsageCount.at(PipeType::PIPE_MTE1)) / clock * percent);
    pipeUsageRate["PIPE_MTE2_Usage_Rate"] = FormatUsageRate(static_cast<double>(pipeUsageCount.at(PipeType::PIPE_MTE2)) / clock * percent);
    pipeUsageRate["PIPE_MTE3_Usage_Rate"] = FormatUsageRate(static_cast<double>(pipeUsageCount.at(PipeType::PIPE_MTE3)) / clock * percent);
    pipeUsageRate["PIPE_FIX_Usage_Rate"] = FormatUsageRate(static_cast<double>(pipeUsageCount.at(PipeType::PIPE_FIX)) / clock * percent);
    report["pipeUsageRate"] = pipeUsageRate;

    uint64_t maxUsage = 0;
    for (const auto& entry : pipeUsageCount) {
        if (entry.second > maxUsage) {
            maxUsage = entry.second;
        }
    }
    report["theoreticalMinimumCycles"] = maxUsage;
    // Memory Usage
    Json memoryUsage;
    memoryUsage["MEM_UB_Peak_Usage"] = FormatUsageRate(static_cast<double>(bufferMaxUsage.at(MemoryType::MEM_UB)) / maxUBSize * percent);
    memoryUsage["MEM_UB_Average_Usage"] = FormatUsageRate(static_cast<double>(bufferTotalUsage.at(MemoryType::MEM_UB)) / clock / maxUBSize * percent);
    memoryUsage["MEM_L1_Peak_Usage"] = FormatUsageRate(static_cast<double>(bufferMaxUsage.at(MemoryType::MEM_L1)) / maxL1Size * percent);
    memoryUsage["MEM_L1_Average_Usage"] = FormatUsageRate(static_cast<double>(bufferTotalUsage.at(MemoryType::MEM_L1)) / clock / maxL1Size * percent);
    memoryUsage["MEM_L0A_Peak_Usage"] = FormatUsageRate(static_cast<double>(bufferMaxUsage.at(MemoryType::MEM_L0A)) / maxL0ASize * percent);
    memoryUsage["MEM_L0A_Average_Usage"] = FormatUsageRate(static_cast<double>(bufferTotalUsage.at(MemoryType::MEM_L0A)) / clock / maxL0ASize * percent);
    memoryUsage["MEM_L0B_Peak_Usage"] = FormatUsageRate(static_cast<double>(bufferMaxUsage.at(MemoryType::MEM_L0B)) / maxL0BSize * percent);
    memoryUsage["MEM_L0B_Average_Usage"] = FormatUsageRate(static_cast<double>(bufferTotalUsage.at(MemoryType::MEM_L0B)) / clock / maxL0BSize * percent);
    memoryUsage["MEM_L0C_Peak_Usage"] = FormatUsageRate(static_cast<double>(bufferMaxUsage.at(MemoryType::MEM_L0C)) / maxL0CSize * percent);
    memoryUsage["MEM_L0C_Average_Usage"] = FormatUsageRate(static_cast<double>(bufferTotalUsage.at(MemoryType::MEM_L0C)) / clock / maxL0CSize * percent);
    report["memoryUsage"] = memoryUsage;
    // Spill Info
    report["spillCount"] = spillInfoVec.size();
    // Detailed spill information
    HealthCheckSpillInfo();
    return SUCCESS;
}

void OoOSchedulerCheck::HealthCheckBlockGraph(Function *function) {
    report["totalOpCount"] = function->Operations().size();
    auto &tensors = function->GetTensorMap().inverseMap_;
    size_t maxProducers = 0;
    size_t maxConsumers = 0;
    std::vector<int> maxProducersTensors;
    std::vector<int> maxConsumersTensors;
    for (auto &tensor : tensors) {
        maxProducers = std::max(tensor.second->GetProducers().size(), maxProducers);
        maxConsumers = std::max(tensor.second->GetConsumers().size(), maxConsumers);
    }
    for (auto &tensor : tensors) {
        if (tensor.second->GetProducers().size() == maxProducers) {
            maxProducersTensors.emplace_back(tensor.second->GetMagic());
        }
        if (tensor.second->GetConsumers().size() == maxConsumers) {
            maxConsumersTensors.emplace_back(tensor.second->GetMagic());
        }
    }
    size_t maxInputs = 0;
    size_t maxOutputs = 0;
    std::vector<int> maxInputsOps;
    std::vector<int> maxOutputsOps;
    for (auto operation : function->Operations().DuplicatedOpList()) {
        maxInputs = std::max(operation->GetIOperands().size(), maxInputs);
        maxOutputs = std::max(operation->GetOOperands().size(), maxOutputs);
    }
    for (auto operation : function->Operations().DuplicatedOpList()) {
        if (operation->GetIOperands().size() == maxInputs) {
            maxInputsOps.emplace_back(operation->GetOpMagic());
        }
        if (operation->GetOOperands().size() == maxOutputs) {
            maxOutputsOps.emplace_back(operation->GetOpMagic());
        }
    }
    report["maxProducers"] = maxProducers;
    report["maxProducersTensors"] = maxProducersTensors;
    report["maxConsumers"] = maxConsumers;
    report["maxConsumersTensors"] = maxConsumersTensors;
    report["maxInputs"] = maxInputs;
    report["maxInputsOps"] = maxInputsOps;
    report["maxOutputs"] = maxOutputs;
    report["maxOutputsOps"] = maxOutputsOps;
}

Status OoOSchedulerCheck::DoHealthCheck(Function *function, const std::string &fileName) {
    if (HealthCheckOoOSchedule() != SUCCESS) {
        ALOG_ERROR_F("DoHealthCheck failed at HealthCheckOoOSchedule!");
        return FAILED;
    }
    HealthCheckBlockGraph(function);
    std::ofstream file(fileName);
    file << report.dump(1) << std::endl;
    file.close();
    return SUCCESS;
}

} // namespace tile_fwk
} // namespace npu