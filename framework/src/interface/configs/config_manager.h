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
 * \file config_manager.h
 * \brief
 */

#pragma once

#include <nlohmann/json.hpp>
#include <sys/file.h>
#include <string>
#include <unistd.h>
#include <type_traits>
#include <set>
#include "interface/utils/common.h"
#include "interface/utils/log.h"

namespace npu::tile_fwk {
using JsonExpcetion = nlohmann::json::exception;

/* Platform KEYs */
const std::string KEY_STD_LOG_LEVEL = "STD_LOG_LEVEL";
const std::string KEY_FILE_LOG_LEVEL = "FILE_LOG_LEVEL";
const std::string KEY_GRAPH_FILE_TYPE = "GRAPH_FILE_TYPE";
const std::string KEY_GRAPH_ONLY_DOT = "GRAPH_ONLY_DOT";
const std::string KEY_ONLY_TENSOR_GRAPH = "ONLY_TENSOR_GRAPH";
const std::string KEY_ONLY_HOST_COMPILE = "ONLY_HOST_COMPILE";
const std::string KEY_ENABLE_COST_MODEL = "ENABLE_COST_MODEL";
const std::string KEY_ENABLE_DYN_FULL_COST_MODEL = "ENABLE_DYN_FULL_COST_MODEL";
const std::string KEY_ENABLE_AIHAC_BACKEND = "ENABLE_AIHAC_BACKEND";
const std::string KEY_ENABLE_CHECKER = "ENABLE_CHECKER";
const std::string KEY_DUMP_SOURCE_LOCATION = "DUMP_SOURCE_LOCATION";
const std::string KEY_ENABLE_PROF_FUNC = "ENABLE_PROF_FUNC";
const std::string KEY_ENABLE_PROF_AICORE_TIME = "ENABLE_PROF_AICORE_TIME";
const std::string KEY_ENABLE_PROF_AICORE_PMU = "ENABLE_PROF_AICORE_PMU";

/* Host KEYs */
const std::string KEY_STRATEGY = "STRATEGY";
const std::string KEY_ENABLE_BINARY_CACHE = "enable_binary_cache";
const std::string KEY_ONLY_CODEGEN = "only_codegen";

/* Pass KEYs */

const std::string KEY_PRINT_GRAPH = "print_graph";
const std::string KEY_PRINT_PROGRAM = "print_program";
const std::string KEY_DUMP_GRAPH = "dump_graph";
const std::string KEY_DUMP_PASS_TIME_COST = "DUMP_PASS_TIME_COST";
const std::string KEY_PRE_CHECK = "PRE_CHECK";
const std::string KEY_POST_CHECK = "POST_CHECK";
const std::string KEY_EXPECTED_VALUE_CHECK = "EXPECTED_VALUE_CHECK";
const std::string KEY_DISABLE_PASS = "DISABLE_PASS";
const std::string KEY_HEALTH_CHECK = "HEALTH_CHECK";
const std::string KEY_RESUME_PARH = "RESUME_PATH";
const std::string KEY_EXEC_VERIFIER = "EXEC_VERIFIER";
const std::string KEY_SET_SCOPE = "SCOPE";


/* CodeGen KEYs */
const std::string KEY_PARALLEL_COMPILE = "parallel_compile";
const std::string KEY_FIXED_OUTPUT_PATH = "fixed_output_path"; // if true, dump cce to output directory
const std::string KEY_FORCE_OVERWRITE = "force_overwrite"; // if true, don't dump cce when file exists
const std::string KEY_CODEGEN_SUPPORT_TILE_TENSOR = "CODEGEN_SUPPORT_TILE_TENSOR";       // if true, gen code with layout mode
const std::string KEY_CODEGEN_NEED_COMPILE = "CODEGEN_NEED_COMPILE";       // if true, gen code & compile code

enum class DPlatform {
    ASCEND_910B1,
    ASCEND_910B2,
    ASCEND_910B3,
    ASCEND_910B4,
    ASCEND_950PR_9579,
    UNKNOWN_DEVICE,
};

inline DPlatform StringToDpaltform(std::string platform) {
    std::unordered_map<std::string, DPlatform> mappings = {
        {"ASCEND_910B1", DPlatform::ASCEND_910B1},
        {"ASCEND_910B2", DPlatform::ASCEND_910B2},
        {"ASCEND_910B3", DPlatform::ASCEND_910B3},
        {"ASCEND_910B4", DPlatform::ASCEND_910B4},
        {"ASCEND_950PR_9579", DPlatform::ASCEND_950PR_9579},
    };

    if (mappings.count(platform)) {
        return mappings[platform];
    }

    return DPlatform::UNKNOWN_DEVICE;
}

struct PassConfigs {
    bool printGraph{false};
    bool printProgram{false};
    bool dumpGraph{false};
    bool dumpPassTimeCost{false};
    bool preCheck{false};
    bool postCheck{false};
    bool expectedValueCheck{false};
    bool disablePass{false};
    bool healthCheck{false};
    std::string resumePath{""};
};

struct GlobalPassConfigs {
    bool enablePassConfigs{false};
    PassConfigs defaultPassConfigs;
};

template <typename T>
using ConvertedConfigType = std::conditional_t<std::is_constructible_v<std::string, T>, std::string, std::decay_t<T>>;

struct InternalGlobalConfig {
    std::string logTopFolder;
    std::string logTensorGraphFolder;
    std::string logFile;
};

class ConfigManager {
public:
    static ConfigManager &Instance();
    ConfigManager(const ConfigManager &) = delete;
    ConfigManager &operator=(const ConfigManager &) = delete;

    Status Initialize();

    const GlobalPassConfigs &GetGlobalConfigs() const { return globalPassConfigs_; }
    PassConfigs GetPassConfigs(const std::string &strategy, const std::string &identifier) const;
    void PassConfigsDebugInfo(const std::string &strategy, const std::vector<std::string> &identifiers) const;

    const InternalGlobalConfig &GetInternalConfig() const { return globalConfigs_; }
    void SetInternalConfig(const InternalGlobalConfig &globalConfig) { globalConfigs_ = globalConfig; }

    template <typename T>
    auto GetPlatformConfig(const std::string &key, const T &defaultValue) {
        return GetConfig(json_, {"global_configs", "platform_configs", key}, defaultValue);
    }

    template <typename T>
    auto GetHostConfig(const std::string &key, const T &defaultValue) {
        return GetConfig(json_, {"global_configs", "host_configs", key}, defaultValue);
    }

    template <typename T>
    auto GetDeviceConfig(const std::string &key, const T &defaultValue) {
        return GetConfig(json_, {"global_configs", "device_configs", key}, defaultValue);
    }

    template <typename T>
    auto GetCoreConfig(const std::string &key, const T &defaultValue) {
        return GetConfig(json_, {"global_configs", "core_configs", key}, defaultValue);
    }

    template <typename T>
    auto GetSimConfig(const std::string &key, const T &defaultValue) {
        return GetConfig(json_, {"global_configs", "simulation_configs", key}, defaultValue);
    }

    template <typename T>
    auto GetCodeGenConfig(const std::string &key, const T &defaultValue) {
        return GetConfig(json_, {"global_configs", "codegen_configs", key}, defaultValue);
    }

    template <typename T>
    auto GetDistConfig(const std::string &key, const T &defaultValue) {
        return GetConfig(json_, {"global_configs", "distributed_configs", key}, defaultValue);
    }

    template <typename T>
    auto GetPassConfig(
        const std::string &strategy, const std::string &identifier, const std::string &key, const T &defaultValue) {
        return GetConfig(json_, {"strategies", strategy, identifier, key}, defaultValue);
    }

    template <typename T>
    auto GetPassDefaultConfig(const std::string &key, const T &defaultValue) {
        return GetConfig(json_, {"pass_global_configs", "default_pass_configs", key}, defaultValue);
    }

    template <typename T>
    auto SetPassDefaultConfig(const std::string &key, const T &value) {
        SetConfig(json_, {"pass_global_configs", "default_pass_configs", key}, value);
        RefreshGlobalPassCfg();
    }

    template <typename T>
    auto GetPassGlobalConfig(const std::string &key, const T &defaultValue) {
        return GetConfig(json_, {"pass_global_configs", key}, defaultValue);
    }

    template <typename T>
    auto SetPassGlobalConfig(const std::string &key, const T &value) {
        SetConfig(json_, {"pass_global_configs", key}, value);
        RefreshGlobalPassCfg();
    }

    template <typename T>
    void SetPlatformConfig(const std::string &key, const T &value) {
        SetConfig(json_, {"global_configs", "platform_configs", key}, value);
    }

    template <typename T>
    void SetHostConfig(const std::string &key, const T &value) {
        SetConfig(json_, {"global_configs", "host_configs", key}, value);
    }

    template <typename T>
    void SetDeviceConfig(const std::string &key, const T &value) {
        SetConfig(json_, {"global_configs", "device_configs", key}, value);
    }

    template <typename T>
    void SetCoreConfig(const std::string &key, const T &value) {
        SetConfig(json_, {"global_configs", "core_configs", key}, value);
    }

    template <typename T>
    void SetSimConfig(const std::string &key, const T &value) {
        SetConfig(json_, {"global_configs", "simulation_configs", key}, value);
    }

    template <typename T>
    void SetDistConfig(const std::string &key, const T &value) {
        SetConfig(json_, {"global_configs", "distributed_configs", key}, value);
    }

    template <typename T>
    void SetPassConfig(
        const std::string &strategy, const std::string &identifier, const std::string &key, const T &value) {
        SetConfig(json_, {"strategies", strategy, identifier, key}, value);
    }

    template <typename T>
    auto GetOperationConfig(const std::string &key, const T &defaultValue) {
        return GetConfig(json_, {"global_configs", "operation_configs", key}, defaultValue);
    }

    template <typename T>
    void SetOperationConfig(const std::string &key, const T &value) {
        SetConfig(json_, {"global_configs", "operation_configs", key}, value);
    }

    template <typename T>
    void SetCodeGenConfig(const std::string &key, const T &value) {
        SetConfig(json_, {"global_configs", "codegen_configs", key}, value);
    }

    void Reset() { json_ = originJson_; }

    const nlohmann::json &GetJsonData() const { return json_; }
    void SetJsonData(const nlohmann::json &json) { json_ = json; }

    const std::string &LogTopFolder();
    const std::string &LogTensorGraphFolder();
    const std::string &LogFile();
    void ResetLog();

private:
    bool isInit_ = false;
    GlobalPassConfigs globalPassConfigs_;
    InternalGlobalConfig globalConfigs_;
    nlohmann::json json_;
    nlohmann::json originJson_;

    ConfigManager();

    ~ConfigManager() = default;

    static const nlohmann::json *GetJsonNode(const nlohmann::json &root, const std::vector<std::string> &keys);
    void RefreshGlobalPassCfg();

    template <typename T>
    static void SetConfig(nlohmann::json &root, const std::vector<std::string> &keys, const T &value) {
        auto *node = &root;
        for (auto &&key : keys) {
            node = &(*node)[key];
        }
        *node = value;
    }

    template <typename T>
    static ConvertedConfigType<T> GetConfig(
        const nlohmann::json &root, const std::vector<std::string> &keys, const T &defaultValue) {
        if (auto *node = GetJsonNode(root, keys)) {
            return node->get<ConvertedConfigType<T>>();
        }
        return defaultValue;
    }

    template <typename T>
    static auto GetChildConfig(const nlohmann::json &root, const std::string &key, const T &defaultValue) {
        return GetConfig(root, {key}, defaultValue);
    }
};

namespace config {
template <typename T>
auto GetPlatformConfig(const std::string &key, const T &defaultValue) {
    return ConfigManager::Instance().GetPlatformConfig(key, defaultValue);
}

template <typename T>
auto GetHostConfig(const std::string &key, const T &defaultValue) {
    return ConfigManager::Instance().GetHostConfig(key, defaultValue);
}

template <typename T>
auto GetDeviceConfig(const std::string &key, const T &defaultValue) {
    return ConfigManager::Instance().GetDeviceConfig(key, defaultValue);
}

template <typename T>
auto GetCoreConfig(const std::string &key, const T &defaultValue) {
    return ConfigManager::Instance().GetCoreConfig(key, defaultValue);
}

template <typename T>
auto GetPassConfig(
    const std::string &strategy, const std::string &identifier, const std::string &key, const T &defaultValue) {
    return ConfigManager::Instance().GetPassConfig(strategy, identifier, key, defaultValue);
}

template <typename T>
auto GetSimConfig(const std::string &key, const T &defaultValue) {
    return ConfigManager::Instance().GetSimConfig(key, defaultValue);
}

template <typename T>
auto GetPassGlobalConfig(const std::string &key, const T &defaultValue) {
    return ConfigManager::Instance().GetPassGlobalConfig(key, defaultValue);
}

template <typename T>
auto SetPassGlobalConfig(const std::string &key, const T &value) {
     ConfigManager::Instance().SetPassGlobalConfig(key, value);
}

template <typename T>
auto GetPassDefaultConfig(const std::string &key, const T &defaultValue) {
    return ConfigManager::Instance().GetPassDefaultConfig(key, defaultValue);
}

template <typename T>
auto SetPassDefaultConfig(const std::string &key, const T &value) {
     ConfigManager::Instance().SetPassDefaultConfig(key, value);
}

template <typename T>
auto GetCodeGenConfig(const std::string &key, const T &value) {
    return ConfigManager::Instance().GetCodeGenConfig(key, value);
}

template <typename T>
void SetPlatformConfig(const std::string &key, const T &value) {
    ConfigManager::Instance().SetPlatformConfig(key, value);
}

template <typename T>
void SetHostConfig(const std::string &key, const T &value) {
    ConfigManager::Instance().SetHostConfig(key, value);
}

template <typename T>
void SetDeviceConfig(const std::string &key, const T &value) {
    ConfigManager::Instance().SetDeviceConfig(key, value);
}

template <typename T>
void SetCoreConfig(const std::string &key, const T &value) {
    ConfigManager::Instance().SetCoreConfig(key, value);
}

template <typename T>
void SetDistConfig(const std::string &key, const T &value) {
    ConfigManager::Instance().SetDistConfig(key, value);
}

template <typename T>
void SetPassConfig(const std::string &strategy, const std::string &identifier, const std::string &key, const T &value) {
    ConfigManager::Instance().SetPassConfig(strategy, identifier, key, value);
}

template <typename T>
void SetSimConfig(const std::string &key, const T &value) {
    ConfigManager::Instance().SetSimConfig(key, value);
}

template <typename T>
void SetOperationConfig(const std::string &key, const T &value) {
    ConfigManager::Instance().SetOperationConfig(key, value);
}

template <typename T>
void SetCodeGenConfig(const std::string &key, const T &value) {
    ConfigManager::Instance().SetCodeGenConfig(key, value);
}

inline DPlatform GetDevicePlatform() {
    auto platform = ConfigManager::Instance().GetPlatformConfig("DEVICE_PLATFORM", "ASCEND_910B2");
    return StringToDpaltform(platform);
}

inline const std::string GetAbsoluteTopFolder() {
    constexpr size_t size = 1024;
    char cwdBuf[size] = {};
    std::string cwd = getcwd(cwdBuf, size);

    return cwd + "/" + ConfigManager::Instance().LogTopFolder();
}

inline const std::string &LogTopFolder() {
    return ConfigManager::Instance().LogTopFolder();
}

inline const std::string &LogTensorGraphFolder() {
    return ConfigManager::Instance().LogTensorGraphFolder();
}

inline const std::string &LogFile() {
    return ConfigManager::Instance().LogFile();
}

inline Status SetPassStrategy(const std::string s) {
    SetHostConfig(KEY_STRATEGY, s);
    return true;
}

inline std::string GetPassStrategy() {
    return GetHostConfig(KEY_STRATEGY, "OOO");
}

inline bool UseTIG() {
    return GetPassStrategy() == "TIG" || GetPassStrategy() == "PVC2_OOO" || GetPlatformConfig("TEST_IS_TIG", false) ||
           GetPassStrategy() == "DFS_OOO" || GetPassStrategy() == "BFS_DFS_OOO";
}

} // namespace config

} // namespace npu::tile_fwk
