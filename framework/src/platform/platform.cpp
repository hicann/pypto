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
 * \file platform.cpp
 * \brief
 */

#include <fstream>
#include "tilefwk/platform.h"
#include "parser/platform_parser.h"
#include "parser/internal_parser.h"
#include "simulation_platform/simulation_platform.h"

namespace npu::tile_fwk {
const std::string version = "version";
const std::string socVersionInfo = "SoC_version";
const std::string npuArchInfo = "NpuArch";
const std::string shortSocVer = "Short_SoC_version";
const std::string socInfo = "SoCInfo";
const std::string aiCoreCnt = "ai_core_cnt";
const std::string cubeCoreCnt = "cube_core_cnt";
const std::string vectorCoreCnt = "vector_core_cnt";
const std::string aiCpuCnt = "ai_cpu_cnt";
const std::string aiCoreSpec = "AICoreSpec";
const std::string l0aSize = "l0_a_size";
const std::string l0bSize = "l0_b_size";
const std::string l0cSize = "l0_c_size";
const std::string l1Size = "l1_size";
const std::string ubSize = "ub_size";
const std::unordered_map<std::string, NPUArch> npuArchMap = {
    {"1001", NPUArch::DAV_1001},
    {"2201", NPUArch::DAV_2201},
    {"3510", NPUArch::DAV_3510},
};

NPUArch StringToNPUArch(const std::string& npuArch) {
    auto it = npuArchMap.find(npuArch);
    if (it != npuArchMap.end()) {
        FUNCTION_LOGD("Set NpuArch as %s.", npuArch.c_str());
        return it->second;
    }
    return NPUArch::DAV_2201;
}

size_t Core::GetMemorySize(MemoryType type) const {
    auto it = memories_.find(type);
    if (it != memories_.end()) {
        return it->second.size;
    }
    return 0;
}

size_t Die::GetMemoryLimit(MemoryType type) const {
    size_t aic_limit = core_wrap_.GetAICMemorySize(type);
    size_t aiv_limit = core_wrap_.GetAIVMemorySize(type);
    if(aic_limit == 0 && aiv_limit == 0) {
        // ERROR Note
        return 0;
    }
    return aic_limit == 0 ? aiv_limit : aic_limit;
}

bool Die::SetMemoryPath(const std::vector<std::pair<MemoryType, MemoryType>>& dataPaths) {
    for (const auto &pathDesc : dataPaths) {
        if (pathDesc.first != MemoryType::MEM_UNKNOWN && pathDesc.second != MemoryType::MEM_UNKNOWN) {
            memoryGraph_.AddPath(pathDesc.first, pathDesc.second);
        }
    }
    return true;
}

bool Die::FindNearestPath(MemoryType from, MemoryType to, std::vector<MemoryType> &paths) const {
    auto res = memoryGraph_.FindNearestPath(from, to, paths);
    if (res == true) {
        return true;
    }
    paths.clear();
    return false;
}

void SoC::SetNPUArch(const std::string& versionStr) {
    version_ = StringToNPUArch(versionStr);
}

void SoC::SetCoreVersion(const std::unordered_map<std::string, std::string>& ver) {
    for (const auto &pair : ver) {
        if (pair.first == "AIC") {
            GetAICCore().SetVersion(pair.second);
        } else if (pair.first == "AIV") {
            GetAIVCore().SetVersion(pair.second);
        }
    }
}

void SoC::SetCCECVersion(const std::unordered_map<std::string, std::string>& ver) {
    for (const auto &pair : ver) {
        if (pair.first == "AIC") {
            GetAICCore().SetCCECVersion(pair.second);
        } else if (pair.first == "AIV") {
            GetAIVCore().SetCCECVersion(pair.second);
        }
    }
}

std::string SoC::GetCoreVersion(std::string CoreType) {
    if (CoreType == "AIC") {
        return GetAICCore().GetVersion();
    } else if (CoreType == "AIV") {
        return GetAIVCore().GetVersion();
    } else {
        return "UNKNOWN_CORE";
    }
}

std::string SoC::GetCCECVersion(std::string CoreType) {
    if (CoreType == "AIC") {
        return GetAICCore().GetCCECVersion();
    } else if (CoreType == "AIV") {
        return GetAIVCore().GetCCECVersion();
    } else {
        return "UNKNOWN_CORE";
    }
}

size_t SoC::GetAICPUNum() const {
    return ai_cpu_cnt_;
}

void MemoryGraph::AddPath(MemoryType from, MemoryType to) {
    if (from == to) {
        return;
    }
    std::shared_ptr<MemoryNode> fromNode = GetNode(from);
    std::shared_ptr<MemoryNode> toNode = GetNode(to);
    if ((fromNode == nullptr) || (toNode == nullptr)) {
        return;
    }
    fromNode->AddDest(toNode);
}

std::shared_ptr<MemoryNode> MemoryGraph::GetNode(MemoryType type) {
    std::shared_ptr<MemoryNode> node;
    if (nodes.count(type) != 0) {
        node = nodes[type];
        return node;
    }
    node = std::make_shared<MemoryNode>();
    if (node == nullptr) {
        return nullptr;
    }
    node->type = type;
    nodes.insert({type, node});
    return node;
}

void MemoryGraph::DFS(MemoryType target, const std::shared_ptr<MemoryNode> &node, std::vector<MemoryType> &candidate, std::vector<MemoryType> &paths) const {
    for (auto &dest : node->dests) {
        if (std::find(candidate.begin(), candidate.end(), dest) != candidate.end()) {
            continue;
        }
        candidate.push_back(dest);
        if (dest != target) {
            DFS(target, nodes.at(dest), candidate, paths);
            candidate.pop_back();
            continue;
        }
        if ((!paths.empty()) && (paths.size() <= candidate.size())) {
            candidate.pop_back();
            continue;
        }
        paths.clear();
        for (auto &t : candidate) {
            paths.push_back(t);
        }
        candidate.pop_back();
    }
}

bool MemoryGraph::FindNearestPath(MemoryType from, MemoryType to, std::vector<MemoryType> &paths) const {
    if (nodes.count(from) == 0) {
        return false;
    }
    if (nodes.count(to) == 0) {
        return false;
    }
    std::vector<MemoryType> candidate = {from};
    paths.clear();
    const auto it = nodes.find(from);
    DFS(to, it->second, candidate, paths);
    return true;
}

Platform &Platform::Instance() {
    static Platform instance;
    return instance;
}

void Platform::SetMemoryLimit(const PlatformParser &parser) {
    size_t memoryLimit;
    if (parser.GetSizeVal(aiCoreSpec, l0aSize, memoryLimit)) {
        GetAICCore().AddMemory(MemoryInfo(MemoryType::MEM_L0A, memoryLimit));
    }
    if (parser.GetSizeVal(aiCoreSpec, l0bSize, memoryLimit)) {
        GetAICCore().AddMemory(MemoryInfo(MemoryType::MEM_L0B, memoryLimit));
    }
    if (parser.GetSizeVal(aiCoreSpec, l0cSize, memoryLimit)) {
        GetAICCore().AddMemory(MemoryInfo(MemoryType::MEM_L0C, memoryLimit));
    }
    if (parser.GetSizeVal(aiCoreSpec, l1Size, memoryLimit)) {
        GetAIVCore().AddMemory(MemoryInfo(MemoryType::MEM_L1, memoryLimit));
    }
    if (parser.GetSizeVal(aiCoreSpec, ubSize, memoryLimit)) {
        GetAIVCore().AddMemory(MemoryInfo(MemoryType::MEM_UB, memoryLimit));
    }
}

void Platform::LoadPlatformInfo(const PlatformParser &parser) {
    std::string archType;
    std::string shortSocVersion;
    std::unordered_map<std::string, std::string> versionInfo;
    if (parser.GetStringVal(version, npuArchInfo, archType)) {
        GetSoc().SetNPUArch(archType);
    }
    if (parser.GetStringVal(version, shortSocVer, shortSocVersion)) {
        GetSoc().SetShortSocVersion(shortSocVersion);
    }
    if (parser.GetCCECVersion(versionInfo)) {
        GetSoc().SetCCECVersion(versionInfo);
    }
    if (parser.GetCoreVersion(versionInfo)) {
        GetSoc().SetCoreVersion(versionInfo);
    }
    size_t coreNum;
    if (parser.GetSizeVal(socInfo, aiCoreCnt, coreNum)) {
        GetSoc().SetAICoreNum(coreNum);
    }
    if (parser.GetSizeVal(socInfo, cubeCoreCnt, coreNum)) {
        GetSoc().SetAICCoreNum(coreNum);
    }
    if (parser.GetSizeVal(socInfo, vectorCoreCnt, coreNum)) {
        GetSoc().SetAIVCoreNum(coreNum);
    }
    if (parser.GetSizeVal(socInfo, aiCpuCnt, coreNum)) {
        GetSoc().SetAICPUNum(coreNum);
    }
    SetMemoryLimit(parser);
    std::vector<std::pair<MemoryType, MemoryType>> dataPath;
    InternalParser internalParser = InternalParser(archType);
    if (internalParser.LoadInternalInfo()) {
        if (internalParser.GetDataPath(dataPath)) {
            GetDie().SetMemoryPath(dataPath);
        } 
    }
}

void Platform::ObtainPlatformInfo() {
    static bool initialized = false;
    if (initialized) {
        return;
    }

    std::string srcPath;
    std::string socVersion;
    if (CannHostRuntime::Instance().GetSocVersion(socVersion)) {
        srcPath = CannHostRuntime::Instance().GetPlatformFile(socVersion);
    }
    if (srcPath.empty()) {
        FUNCTION_LOGW("Cannot obtain ini from the device, using default ini file.");
        SimulationPlatform simulationPlatform;
        simulationPlatform.GetCostModelPlatformRealPath(srcPath);
    }
    INIParser iniparser;
 	iniparser.Initialize(srcPath);
    LoadPlatformInfo(iniparser);
    initialized = true;
}
}
