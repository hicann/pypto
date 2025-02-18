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
 * \file backend.cpp
 * \brief
 */

#include "simulation/backend.h"

#include <cctype>
#include "interface/configs/config_manager.h"
#include "interface/cache/function_cache.h"
#include "interface/machine/host/machine_task.h"

namespace {
const std::string PROGRAM_ENTRY_FUNCTION_NAME = "PROGRAM_ENTRY";
}

namespace npu::tile_fwk {

void CostModelAgent::BuildCostModel()
{
    ALOG_INFO("Init CostModel Simulation.");
    ALOG_INFO("Using Config A2A3.");
    costModel = std::make_shared<CostModel::CostModelInterface>();
    std::vector<std::string> inputArgs = config::GetSimConfig("args", inputArgs);
    int mode = config::GetSimConfig("SIM_MODE", 0);
    int accLevel = config::GetSimConfig("ACCURACY_LEVEL", 2);
    int pvLevel = config::GetSimConfig("PV_LEVEL", 0);
    int logLevel = config::GetSimConfig("LOG_LEVEL", 3);
    int cycleThreshold = config::GetSimConfig("EXECUTE_CYCLE_THRESHOLD", -1);
    std::string jsonPath = config::GetSimConfig("JSON_PATH", "");
    agentJsonPath = config::GetSimConfig("AGENT_JSON_PATH", "");
    auto folder = config::GetAbsoluteTopFolder() + "/" + ("CostModelSimulationOutput");
    config::SetRunDataOption(KEY_SWIM_GRAPH_PATH, folder);
    std::vector<std::string> configs;
    if (!jsonPath.empty()) {
        configs.push_back("-f");
        configs.push_back(jsonPath);
    }
    if (!agentJsonPath.empty()) {
        getFunctionFromJson = true;
    }
    configs.push_back("-m");
    configs.push_back(std::to_string(mode));
    configs.push_back("-o");
    configs.push_back(folder);
    configs.push_back("-a");
    configs.push_back(std::to_string(accLevel));
    configs.push_back("-t");
    configs.push_back(std::to_string(logLevel));
    configs.push_back("-p");
    configs.push_back(std::to_string(pvLevel));
    if (cycleThreshold > 0) {
        configs.push_back("-l");
        configs.push_back(std::to_string(cycleThreshold));
    }
    if (config::GetSimConfig("DRAW_FUNCTION_GRAPH", false)) {
        configs.push_back("-d");
        configs.push_back("true");
    }
    if (inputArgs.size() > 0) {
        configs.push_back("-s");
        for (auto &arg : inputArgs) {
            configs.push_back(arg);
        }
    }
    if (!topoJsonPath.empty()) {
        configs.push_back("-s");
        configs.push_back("Device.submitTopo=true");
        configs.push_back("Device.submitTopoPath=" + topoJsonPath);
    }
    costModel->BuildCostModel(configs);
}

void CostModelAgent::SubmitToCostModel(Function *rootFunc)
{
    if (costModel == nullptr) {
        BuildCostModel();
    }
    if (getFunctionFromJson) {
        GetFunctionFromJson(agentJsonPath);
        rootFunc = Program::GetInstance().GetCurrentFunction()->rootFunc_;
    }
    ALOG_INFO("Submit to CostModel: ", rootFunc->GetMagicName());
    std::vector<npu::tile_fwk::Function *> funcs;
    if (config::GetSimConfig("BUILD_TASK_BASED_TOPO", true)) {
        funcs.push_back(rootFunc);
        costModel->Submit(funcs, true, "");
    } else {
        for (auto &func : Program::GetInstance().GetFunctionMap()) {
            if (func.second->GetMagicName() == PROGRAM_ENTRY_FUNCTION_NAME) {
                continue;
            }
            funcs.push_back(func.second.get());
        }
        costModel->Submit(funcs, false, "root");
    }
}

void CostModelAgent::SubmitLeafFunctionsToCostModel() {
    if (costModel == nullptr) {
        BuildCostModel();
    }
    ALOG_INFO("Submit Leaf Functions to CostModel");
    std::vector<npu::tile_fwk::Function *> funcs;
    for (auto &func : Program::GetInstance().GetFunctionMap()) {
        if (func.second->GetMagicName().find("leaf") == std::string::npos) {
            continue;
        }
        funcs.push_back(func.second.get());
    }
    costModel->Submit(funcs, false, "");
}

Json CostModelAgent::ParseDynTopo(std::string &path)
{
    Json topoJson = Json::array();
    std::ifstream file(path);
    std::string line;
    while (std::getline(file, line)) {
        if (line.empty() || isalpha(line[0])) {
            continue;
        }
        std::vector<uint64_t> fields;
        std::stringstream ss(line);
        std::string item;
        while (std::getline(ss, item, ',')) {
            try {
                uint64_t num = std::stoull(item);
                fields.push_back(num);
            } catch (const std::invalid_argument& e) {
                // ignore
            } catch (const std::out_of_range& e) {
                std::cerr << "Out of range: " << e.what() << std::endl;
            }
        }
        uint64_t seqNo = fields[seqPos];
        uint64_t taskId = fields[taskIdPos];
        Json taskJson;
        taskJson["uniqueKey"] = static_cast<uint64_t>(seqNo) << seqNumOffset | taskId;
        taskJson["seqNo"] = seqNo;
        taskJson["taskId"] = taskId;
        Json successorsJson = Json::array();
        for (size_t i = succStartPos; i < fields.size(); i++) {
            successorsJson.push_back(fields[i]);
        }
        taskJson["successors"] = successorsJson;
        auto coreType = static_cast<npu::tile_fwk::CoreType>(fields[coreTypePos]);
        taskJson["coreType"] = npu::tile_fwk::GetCoreTypeDict().Find(coreType);
        taskJson["rootIndex"] = fields[rootIndexPos];
        taskJson["rootHash"] =  fields[rootHashpos];
        taskJson["leafIndex"] = fields[leafIndexPos];
        taskJson["opmagic"] = fields[opmagicPos];
        taskJson["psgId"] = fields[psgIdPos];
        taskJson["funcHash"] = fields[funcHashPos];
        topoJson.push_back(taskJson);
    }
    return topoJson;
}

void CostModelAgent::SubmitTopo(std::string &path)
{
    Json res = ParseDynTopo(path);
    topoJsonPath = config::LogTopFolder() + "/tmp_topo_json.json";
    std::ofstream file(topoJsonPath);
    file << res.dump(1) << std::endl;
    file.close();
}

uint64_t CostModelAgent::GetLeafFunctionTimeCost(uint64_t hash)
{
    if (costModel == nullptr) {
        return 0;
    }
    auto it = costModel->sim->leafFunctionTime.find(hash);
    if (it != costModel->sim->leafFunctionTime.end()) {
        return it->second;
    }
    return 0;
}

void CostModelAgent::SubmitSingleFuncToCostModel(Function *func)
{
    if (costModel == nullptr) {
        BuildCostModel();
    }
    ALOG_INFO("Submit Single Function to CostModel: ", func->GetMagicName());
    costModel->SubmitSingleFunction(func);
}

void CostModelAgent::RunCostModel()
{
    if (costModel == nullptr) {
        return;
    }
    ALOG_INFO("Start CostModel Run Simulation");
    costModel->Run();
    ALOG_INFO("End CostModel Run Simulation");
}

void CostModelAgent::TerminateCostModel()
{
    if (costModel == nullptr) {
        return;
    }
    costModel->Report();
}

void CostModelAgent::DebugSingleFunc(Function *func)
{
    auto debugFuncName = config::GetSimConfig("DEBUG_SINGLE_FUNCNAME", "");
    for (auto &leafFunc : func->programs_) {
        std::cout << "Func: " << leafFunc.second->GetMagicName() << std::endl;
        if (leafFunc.second->GetMagicName() == debugFuncName) {
            CostModelAgent costModelAgent;
            costModelAgent.SubmitSingleFuncToCostModel(leafFunc.second);
            costModelAgent.RunCostModel();
            costModelAgent.TerminateCostModel();
            std::cout << "match" << std::endl;
        }
    }
}

void CostModelAgent::GetFunctionFromJson(const std::string &jsonPath)
{
    std::ifstream file(jsonPath);
    ASSERT(file.good()) << "Json file: " << jsonPath << " open failed!!!";
    Json jsonData;
    try {
        file >> jsonData;
    } catch (const std::exception &e) {
        ASSERT(false) << "Json file: " << jsonPath << " parsing error: " << e.what();
    }
    Program::GetInstance().LoadJson(jsonData);
}

extern "C" int32_t ExecuteSimulation(const MachineTask *task, FunctionCache &cache)
{
    (void)cache;
    if (!config::GetPlatformConfig("ENABLE_COST_MODEL", true)) {
        return 0;
    }

    config::SetRunDataOption(KEY_RUNTYPE, "simulation");

    CostModelAgent costModelAgent;

    if (config::GetSimConfig("DEBUG_SINGLE_FUNC", false)) {
        costModelAgent.DebugSingleFunc(task->GetFunction()->rootFunc_);
        return 0;
    }

    if (config::GetSimConfig("SIM_MODE", 0) == static_cast<int>(CostModel::SimMode::LEAF_FUNCTION)) {
        costModelAgent.SubmitLeafFunctionsToCostModel();
    } else {
        costModelAgent.SubmitToCostModel(task->GetFunction()->rootFunc_);
    }
    costModelAgent.RunCostModel();
    costModelAgent.TerminateCostModel();
    return 0;
}

} // namespace npu::tile_fwk
