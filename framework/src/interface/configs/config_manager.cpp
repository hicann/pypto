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
 * \file config_manager.cpp
 * \brief
 */

#include "config_manager.h"
#include <map>
#include <fstream>
#include <cstdlib>
#include <string>
#include "interface/inner/config.h"
#include "interface/utils/common.h"
#include "interface/utils/log.h"
#include "interface/utils/file_utils.h"
#include <unistd.h>
namespace npu::tile_fwk {

const std::string tilefwkConfigEnvName = "TILEFWK_CONFIG_PATH";
static PassConfigs InternalGetPassConfigs(const nlohmann::json &root, const GlobalPassConfigs *globalConfigs);
static GlobalPassConfigs InternalGetGlobalConfigs(const nlohmann::json &globalCfg);

static const nlohmann::json *GetJsonNode(const nlohmann::json &root, const std::vector<std::string> &keys) {
    auto *curr = &root;
    for (auto &&key : keys) {
        if (auto it = curr->find(key); it != curr->end()) {
            curr = &*it;
        } else {
            return nullptr;
        }
    }
    return curr;
}

static const nlohmann::json *GetJsonChild(const nlohmann::json &root, const std::string &key) {
    return GetJsonNode(root, {key});
}

ConfigManager::ConfigManager() {
    Initialize();
}

ConfigManager &ConfigManager::Instance() {
    static ConfigManager instance;
    return instance;
}

const nlohmann::json *ConfigManager::GetJsonNode(const nlohmann::json &root, const std::vector<std::string> &keys) {
    return ::npu::tile_fwk::GetJsonNode(root, keys);
}

Status ConfigManager::Initialize() {
    if (isInit_) {
        ASLOGI("ConfigManager has been initialized.");
        return SUCCESS;
    }
    /* 环境变量优先生效 */
    std::string jsonFilePath = GetEnvVar(tilefwkConfigEnvName);
    if (jsonFilePath.empty()) {
        jsonFilePath = RealPath(GetCurrentSharedLibPath() + "/configs/tile_fwk_config.json");
    }

    config::SetRunDataOption(KEY_PTO_CONFIG_FILE, jsonFilePath);
    ASLOGI("Start to parse op_json_file %s", jsonFilePath.c_str());
    if (!ReadJsonFile(jsonFilePath, json_)) {
        ASLOGE("ReadJsonFile failed.");
        return FAILED;
    }

#ifdef SRCPATH
    constexpr const char *SRC_PATH = SRCPATH;
    // update Json_ through genJson
    std::string genJsonPath = std::string(SRC_PATH) + "/framework/src/cost_model/simulation/scripts/";
    if (IsPathExist(genJsonPath)) {
        auto files = GetFiles(genJsonPath, "json");
        if (!files.empty()) {
            for (const auto &file : files) {
                std::ifstream genJsonFile(genJsonPath + file);
                nlohmann::json jsonConfig = nlohmann::json::parse(genJsonFile);
                genJsonFile.close();

                if (jsonConfig.contains("global_configs")) {
                    const auto& genGlobal = jsonConfig["global_configs"];
                    if (genGlobal.contains("platform_configs") && !genGlobal["platform_configs"].empty()) {
                        json_["global_configs"]["platform_configs"].update(genGlobal["platform_configs"]);
                    }
                    if (genGlobal.contains("simulation_configs") && !genGlobal["simulation_configs"].empty()) {
                        json_["global_configs"]["simulation_configs"].update(genGlobal["simulation_configs"]);
                    }
                }
            }
        }
    }
#endif

    originJson_ = json_;

    if (auto *node = GetJsonChild(json_, "pass_global_configs")) {
        globalPassConfigs_ = InternalGetGlobalConfigs(*node);
    }

    isInit_ = true;
    return SUCCESS;
}

void ConfigManager::RefreshGlobalPassCfg() {
    if (auto *node = GetJsonChild(json_, "pass_global_configs")) {
        globalPassConfigs_ = InternalGetGlobalConfigs(*node);
    }
}

static std::string CreateLogTopFolder() {
    auto now = std::chrono::high_resolution_clock::now();
    auto time = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    auto us = std::chrono::duration_cast<std::chrono::microseconds>(now.time_since_epoch()).count() % 1000000;

    std::stringstream timestamp;
    timestamp << std::put_time(std::localtime(&time), "%Y%m%d_%H%M%S");
    constexpr int NUM_SIX = 6;
    timestamp << "_" << std::setw(NUM_SIX) << std::setfill('0') << us;

    std::string folderPath = "output";
    bool res = CreateDir(folderPath);
    ASSERT(res) << "Failed to create directory: " << folderPath;
    const char* envDir = std::getenv("TILE_FWK_OUTPUT_DIR");
    if (envDir != nullptr) {
        std::string envStr(envDir);
        if (!envStr.empty()) {
            folderPath = std::move(envStr);
        }
    } else {
        folderPath = folderPath + "/" + "output_" + timestamp.str() + "_" + std::to_string(getpid());
    }
    res = CreateDir(folderPath);
    ASSERT(res) << "Failed to create directory: " << folderPath;
    config::SetRunDataOption(KEY_COMPUTE_GRAPH_PATH, RealPath(folderPath));

    return folderPath;
}

const std::string &ConfigManager::LogTopFolder() {
    if (globalConfigs_.logTopFolder.empty()) {
        globalConfigs_.logTopFolder = CreateLogTopFolder();
    }
    return globalConfigs_.logTopFolder;
}

const std::string &ConfigManager::LogTensorGraphFolder() {
    if (globalConfigs_.logTensorGraphFolder.empty()) {
        globalConfigs_.logTensorGraphFolder = LogTopFolder() + "/TensorGraph";
        CreateDir(globalConfigs_.logTensorGraphFolder);
    }
    return globalConfigs_.logTensorGraphFolder;
}

const std::string &ConfigManager::LogFile() {
    if (globalConfigs_.logFile.empty()) {
        globalConfigs_.logFile = LogTopFolder() + "/run.log";
    }
    return globalConfigs_.logFile;
}

void ConfigManager::ResetLog() {
    globalConfigs_.logTopFolder = CreateLogTopFolder();
    std::string newLogFile = LogTopFolder() + "/run.log";
    LoggerManager::FileLoggerReplace(globalConfigs_.logFile, newLogFile, true);
    globalConfigs_.logFile = std::move(newLogFile);
}

PassConfigs ConfigManager::GetPassConfigs(const std::string &strategy, const std::string &identifier) const {
    auto *node = GetJsonNode(json_, {"strategies", strategy, identifier});
    if (!node) {
        return globalPassConfigs_.defaultPassConfigs;
    }
    return InternalGetPassConfigs(*node, &globalPassConfigs_);
}

void ConfigManager::PassConfigsDebugInfo(
    const std::string &strategy, const std::vector<std::string> &identifiers) const {
    auto *node = GetJsonNode(json_, {"strategies", strategy});
    if (!node) {
        ALOG_INFO("[ConfigManager] Missing custom pass strategy <", strategy, "> configs. ",
                    "You may add your own custom strategy configs in 'tile_fwk_config.json'.");
        return;
    }

    size_t maxLength = 0;
    for (auto &&identifier : identifiers) {
        maxLength = std::max(maxLength, identifier.size());
    }

    ALOG_INFO("[ConfigManager] Strategy <", strategy, "> is found. Custom pass strategy will be used.");
    for (auto &&identifier : identifiers) {
        std::string spaces(maxLength - identifier.size(), ' ');
        if (node->find(identifier) != node->end()) {
            ALOG_INFO("[ConfigManager] Pass instance ", spaces, "<", identifier, "> configs loaded.");
        } else {
            ALOG_INFO("[ConfigManager] Pass instance ", spaces, "<", identifier, "> configs for pass strategy <",
            strategy, "> is missing. You may add your own custom strategy configs in 'tile_fwk_config.json'.");
        }
    }
}

/* Helper Functions */
static std::map<std::string, std::function<void(PassConfigs &, const nlohmann::json &)>> g_assignPassConfigFns = {
    {                 KEY_PRINT_GRAPH,
     [](PassConfigs &configs, const nlohmann::json &node) { configs.printGraph = node.get<bool>(); }},
    {                 KEY_PRINT_PROGRAM,
     [](PassConfigs &configs, const nlohmann::json &node) { configs.printProgram = node.get<bool>(); }},
    {                 KEY_DUMP_GRAPH,
     [](PassConfigs &configs, const nlohmann::json &node) { configs.dumpGraph = node.get<bool>(); }},
    {                 KEY_DUMP_PASS_TIME_COST,
     [](PassConfigs &configs, const nlohmann::json &node) { configs.dumpPassTimeCost = node.get<bool>(); }},
    {                 KEY_PRE_CHECK,
     [](PassConfigs &configs, const nlohmann::json &node) { configs.preCheck = node.get<bool>(); }},
    {                 KEY_POST_CHECK,
     [](PassConfigs &configs, const nlohmann::json &node) { configs.postCheck = node.get<bool>(); }},
    {                 KEY_EXPECTED_VALUE_CHECK,
     [](PassConfigs &configs, const nlohmann::json &node) { configs.expectedValueCheck = node.get<bool>(); }},
    {                 KEY_DISABLE_PASS,
     [](PassConfigs &configs, const nlohmann::json &node) { configs.disablePass = node.get<bool>(); }},
    {                 KEY_HEALTH_CHECK,
     [](PassConfigs &configs, const nlohmann::json &node) { configs.healthCheck = node.get<bool>(); }},
    {                 KEY_RESUME_PARH,
     [](PassConfigs &configs, const nlohmann::json &node) { configs.resumePath = node.get<std::string>(); }},
};

static PassConfigs InternalGetPassConfigs(const nlohmann::json &root, const GlobalPassConfigs *globalConfigs) {
    PassConfigs configs;
    if (globalConfigs != nullptr) {
        if (!globalConfigs->enablePassConfigs) {
            return {};
        }
        configs = globalConfigs->defaultPassConfigs;
    }

    for (auto &&[key, assignFn] : g_assignPassConfigFns) {
        if (auto *node = GetJsonChild(root, key)) {
            assignFn(configs, *node);
        }
    }
    return configs;
}

static std::map<std::string, std::function<void(GlobalPassConfigs &, const nlohmann::json &)>> g_assignGlobalConfigFns = {
    {    "enable_pass_configs",
     [](GlobalPassConfigs &configs, const nlohmann::json &node) { configs.enablePassConfigs = node.get<bool>(); }},
    {   "default_pass_configs",
     [](GlobalPassConfigs &configs, const nlohmann::json &node) { configs.defaultPassConfigs = InternalGetPassConfigs(node, nullptr); }},
};

static GlobalPassConfigs InternalGetGlobalConfigs(const nlohmann::json &globalCfg) {
    GlobalPassConfigs configs;
    for (auto &&[key, assignFn] : g_assignGlobalConfigFns) {
        if (auto *node = GetJsonChild(globalCfg, key)) {
            assignFn(configs, *node);
        }
    }
    return configs;
}
} // namespace npu::tile_fwk
