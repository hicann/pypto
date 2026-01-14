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
 * \file config.cpp
 * \brief
 */

#include <variant>
#include <sstream>
#include <shared_mutex>

#include <nlohmann/json.hpp>

#include "interface/inner/config.h"
#include "interface/utils/common.h"
#include "interface/utils/string_utils.h"
#include "interface/utils/file_utils.h"
#include "interface/utils/log.h"
#include "interface/configs/config_manager_ng.h"

namespace npu::tile_fwk {

using ValueType = std::variant<bool, int64_t, std::string, std::vector<int64_t>,
                               std::vector<std::string>, std::map<int64_t, int64_t>>;

using MapType = std::map<int64_t, int64_t>;

static std::map<std::string, ValueType> g_passConfig = {
    {SG_PARALLEL_NUM, 20L},
    {SG_PG_UPPER_BOUND, 10000L},
    {SG_PG_LOWER_BOUND, 512L},
    {CUBE_L1_REUSE_MODE, 0L},
    {CUBE_L1_REUSE_SETTING, std::map<int64_t, int64_t>{}},
    {CUBE_NBUFFER_MODE, 0L},
    {CUBE_NBUFFER_SETTING, std::map<int64_t, int64_t>{}},
    {MG_COPYIN_UPPER_BOUND, 1024 * 1024L},
    {OOO_PRESCHEDULE_METHOD, std::string("PriorDFS")}, // bugs in gcc 9.4
    {VEC_NBUFFER_MODE, 1L},
    {VEC_NBUFFER_SETTING, std::map<int64_t, int64_t>{}},
    {MG_VEC_PARALLEL_LB, 48L},
    {SG_CUBE_PARALLEL_NUM, 24L},
    {PG_SKIP_PARTITION, false},
    {COPYOUT_RESOLVE_COALESCING, 0L},
    {SG_SET_SCOPE, -1L}
};

static std::map<std::string, ValueType> g_runtimeConfig = {
    {DEVICE_SCHED_MODE, 0L},
    {STITCH_FUNCTION_INNER_MEMORY, 10L},
    {STITCH_FUNCTION_OUTCAST_MEMORY, 50L},
    {STITCH_FUNCTION_NUM_INITIAL, 30L},
    {STITCH_FUNCTION_NUM_STEP, 30L}, // Increasing loop number
    {STITCH_CFGCACHE_SIZE, 0L},
    {STITCH_FUNCTION_SIZE, 20000L},
    {CFG_RUN_MODE, CFG_RUN_MODE_NPU}
};

static std::map<std::string, ValueType> g_hostConfig = {
    {ONLY_CODEGEN, false},
};

static std::map<std::string, ValueType> g_codegenConfig = {
    {  SUPPORT_DYNAMIC_ALIGNED, false},
};

static std::map<std::string, ValueType> g_verifyConfig = {
    {KEY_ENABLE_PASS_VERIFY, false},
    {KEY_PASS_VERIFY_SAVE_TENSOR, false},
    {KEY_PASS_VERIFY_SAVE_TENSOR_DIR, std::string("")},
    {KEY_PASS_VERIFY_FILTER, std::vector<std::string>()},
};

static std::map<std::string, ValueType> g_debugConfig = {
    {CFG_COMPILE_DBEUG_MODE, CFG_DEBUG_NONE},
    {CFG_RUNTIME_DBEUG_MODE, CFG_DEBUG_NONE},
};

static std::map<std::string, ValueType> g_globalConfig = {
    {COST_MODEL_ENABLE, false},
};

struct RunDataDir {
    std::string path;
    std::string dName;

    std::string montage() {
        return path + "/" + dName;
    }

    bool empty() {
        return (path.empty() || dName.empty());
    }

    void Reset() {
        path.clear();
        dName.clear();
    }
};

struct ConfigStorage {
    ConfigStorage() { Init(); }

    void Init() {
        printOption.edgeItems = 3; // 3 edge items
        printOption.precision = 4; // 4 float precision
        printOption.threshold = 1000; // 1000 default threshold
        printOption.linewidth = 80; // 80 max line width

        for (auto &[key, val] : g_globalConfig) {
            options[key] = val;
        }

        Reset();
    }

    void Reset() {
        funcType = FunctionType::DYNAMIC;
        semanticLabel = nullptr;
        rundataDir.Reset();
        for (auto &[key, val] : g_passConfig) {
            options["pass." + key] = val;
        }
        for (auto &[key, val] : g_runtimeConfig) {
            options["runtime." + key] = val;
        }
        for (auto &[key, val] : g_hostConfig) {
            options["host." + key] = val;
        }
        for (auto &[key, val] : g_codegenConfig) {
            options["codegen." + key] = val;
        }
        for (auto &[key, val] : g_verifyConfig) {
            options["verify." + key] = val;
        }
        for (auto &[key, val] : g_debugConfig) {
            options["debug." + key] = val;
        }
    }

    FunctionType funcType;
    std::shared_ptr<SemanticLabel> semanticLabel;
    RunDataDir rundataDir;
    std::unordered_map<std::string, ValueType> options;
    PrintOptions printOption;
};

namespace config {

static ConfigStorage g_config;
std::shared_mutex g_rwlock;

void SetBuildStatic(bool isStatic) {
    g_config.funcType = isStatic ? FunctionType::STATIC : FunctionType::DYNAMIC;
}

FunctionType GetFunctionType() {
    return g_config.funcType;
}

void SetSemanticLabel(const std::string &label, const char *filename , int lineno) {
    g_config.semanticLabel = std::make_shared<SemanticLabel>(label, filename, lineno);
}

void SetSemanticLabel(std::shared_ptr<SemanticLabel> label) {
    g_config.semanticLabel = label;
}

std::shared_ptr<SemanticLabel> GetSemanticLabel() {
    return g_config.semanticLabel;
}

bool HasOption(const std::string &key) {
    return g_config.options.find(StringUtils::ToLower(key)) != g_config.options.end();
}

inline void OptionToOss(std::ostringstream &oss, const std::string &key, const ValueType &value) {
    if (std::holds_alternative<bool>(value)) {
        oss << key << ": " << std::boolalpha << std::get<bool>(value);
    } else if (std::holds_alternative<int64_t>(value)) {
        oss << key << ": " << std::get<int64_t>(value);
    } else if (std::holds_alternative<std::string>(value)) {
        oss << key << ": " << std::get<std::string>(value);
    } else if (std::holds_alternative<std::vector<int64_t>>(value)) {
        oss << key << ": " << std::get<std::vector<int64_t>>(value);
    } else if (std::holds_alternative<std::vector<std::string>>(value)) {
        oss << key << ": " << std::get<std::vector<std::string>>(value);
    } else if (std::holds_alternative<MapType>(value)) {
        oss << key << ": ";
        for (auto &[k, v] : std::get<MapType>(value)) {
            oss << "{" << k << ":" << v << "}";
        }
    } else {
        throw std::runtime_error("Config value type not supported: " + key);
    }
}

std::string Dump() {
    std::ostringstream oss;
    auto &printOption = g_config.printOption;

    std::shared_lock lock(g_rwlock);
    oss << "funcType: " << (g_config.funcType == FunctionType::DYNAMIC ? "dynamic" : "static") << std::endl;
    if (g_config.semanticLabel) {
        oss << "sematicLabel: " << g_config.semanticLabel->label << std::endl;
    }
    oss << "printOption.edgeItems: " << printOption.edgeItems << std::endl;
    oss << "printOption.precision: " << printOption.precision << std::endl;
    oss << "printOption.threshold: " << printOption.threshold << std::endl;
    oss << "printOption.linewidth: " << printOption.linewidth << std::endl;

    for (auto &it : g_config.options) {
        OptionToOss(oss, it.first, it.second);
        oss << std::endl;
    }
    return oss.str();
}

bool experimental::IsType(const std::string &key, const std::type_info &type) {
    std::shared_lock lock(g_rwlock);

    auto iter = g_config.options.find(StringUtils::ToLower(key));
    if (iter == g_config.options.end()) {
        return false;
    }
    if (std::holds_alternative<bool>(iter->second)) {
        return type == typeid(bool);
    } else if (std::holds_alternative<int64_t>(iter->second)) {
        return type == typeid(int64_t);
    } else if (std::holds_alternative<std::string>(iter->second)) {
        return type == typeid(std::string);
    } else if (std::holds_alternative<std::vector<int64_t>>(iter->second)) {
        return type == typeid(std::vector<int64_t>);
    } else if (std::holds_alternative<std::vector<std::string>>(iter->second)) {
        return type == typeid(std::vector<std::string>);
    } else if (std::holds_alternative<MapType>(iter->second)) {
        return type == typeid(MapType);
    } else {
        return false;
    }
}

#define DEFINE_GET_OPTION(Type)                                       \
    bool experimental::GetOption(const std::string &key, Type &value) {   \
        std::shared_lock lock(g_rwlock);                              \
        auto iter = g_config.options.find(StringUtils::ToLower(key)); \
        if (iter == g_config.options.end()) {                         \
            return false;                                             \
        }                                                             \
        value = std::get<Type>(iter->second);                         \
        return true;                                                  \
    }

DEFINE_GET_OPTION(bool)
DEFINE_GET_OPTION(int64_t)
DEFINE_GET_OPTION(std::string)
DEFINE_GET_OPTION(std::vector<int64_t>)
DEFINE_GET_OPTION(std::vector<std::string>)
DEFINE_GET_OPTION(MapType)
#undef DEFINE_GET_OPTION

constexpr int LIMIT_DIR_NUM_BEFORE_CREATE = 127;
constexpr const char *PREFIX_RUNDATA = "rundata_";
constexpr const char *ENV_VAR_PYPTO_HOME = "PYPTO_HOME";
constexpr const char *ENV_VAR_HOME = "HOME";
void CreateRunDataDir() {
    std::string envStr = GetEnvVar(ENV_VAR_PYPTO_HOME);
    std::string dir = envStr.empty() ? (GetEnvVar(ENV_VAR_HOME) + "/.pypto") : envStr;
    g_config.rundataDir.path = dir + "/run";
    RemoveOldestDirs(g_config.rundataDir.path, PREFIX_RUNDATA, LIMIT_DIR_NUM_BEFORE_CREATE);
    auto time = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    std::stringstream timestamp;
    timestamp << std::put_time(std::localtime(&time), "%Y%m%d%H%M%S");
    g_config.rundataDir.dName = PREFIX_RUNDATA + timestamp.str();
    bool res = CreateMultiLevelDir(g_config.rundataDir.montage());
    ASSERT(res) << "Failed to create directory: " << g_config.rundataDir.montage();
}

void SetRunDataOption(const std::string &key, const std::string &value) {
    static nlohmann::json j;
    std::shared_lock lock(g_rwlock);
    j[key] = value;
    auto dumpValue = j.dump(2);
    if (g_config.rundataDir.empty()) {
        CreateRunDataDir();
    }
    auto filename = g_config.rundataDir.montage() + "/rundata.json";
    SaveFileSafe(filename, reinterpret_cast<uint8_t*>(dumpValue.data()), dumpValue.size());
}

void experimental::SetOption(const std::string &key, int64_t value) {
    ConfigManagerNg::CurrentScope()->UpdateValue(key, value);
    g_rwlock.lock();
    g_config.options[StringUtils::ToLower(key)] = value;
    g_rwlock.unlock();
}

void experimental::SetOption(const std::string &key, bool value) {
    ConfigManagerNg::CurrentScope()->UpdateValue(key, value);
    g_rwlock.lock();
    g_config.options[StringUtils::ToLower(key)] = value;
    g_rwlock.unlock();
}

void experimental::SetOption(const std::string &key, const char *value) {
    ConfigManagerNg::CurrentScope()->UpdateValue(key, value);
    g_rwlock.lock();
    g_config.options[StringUtils::ToLower(key)] = value;
    g_rwlock.unlock();
}

void experimental::SetOption(const std::string &key, const std::string &value) {
    ConfigManagerNg::CurrentScope()->UpdateValue(key, value);
    g_rwlock.lock();
    g_config.options[StringUtils::ToLower(key)] = value;
    g_rwlock.unlock();
}

void experimental::SetOption(const std::string &key, const std::vector<int64_t> &value) {
    ConfigManagerNg::CurrentScope()->UpdateValue(key, value);
    g_rwlock.lock();
    g_config.options[StringUtils::ToLower(key)] = value;
    g_rwlock.unlock();
}

void experimental::SetOption(const std::string &key, const std::vector<std::string> &value) {
    ConfigManagerNg::CurrentScope()->UpdateValue(key, value);
    g_rwlock.lock();
    g_config.options[StringUtils::ToLower(key)] = value;
    g_rwlock.unlock();
}

void experimental::SetOption(const std::string &key, const std::map<int64_t, int64_t> &value) {
    ConfigManagerNg::CurrentScope()->UpdateValue(key, value);
    g_rwlock.lock();
    g_config.options[StringUtils::ToLower(key)] = value;
    g_rwlock.unlock();
}

void SetPrintOptions(int edgeItems, int precision, int threshold, int linewidth) {
    g_config.printOption.edgeItems = edgeItems;
    g_config.printOption.precision = precision;
    g_config.printOption.threshold = threshold;
    g_config.printOption.linewidth = linewidth;
}

PrintOptions &GetPrintOptions() {
    return g_config.printOption;
}

void Reset() {
    g_rwlock.lock();
    g_config.Reset();
    ConfigManagerNg::CurrentScope()->Clear();
    g_rwlock.unlock();
}

std::unordered_map<std::string, ValueType> GetOptions(){
    return g_config.options;
}

std::shared_ptr<ConfigScope> Duplicate() {
    std::shared_lock lock(g_rwlock);
    auto scopeClone = ConfigManagerNg::CurrentScope();
    return scopeClone;
}

void Restore(std::shared_ptr<ConfigScope> config) {
    g_rwlock.lock();
    ConfigManagerNg::GetInstance().PushScope(config);
    g_rwlock.unlock();
}

template <typename T>
void SetOptionsNg(const std::string &key, const T &value){
    ConfigManagerNg::CurrentScope()->UpdateValue(key, value);
}

template void SetOptionsNg<bool>(const std::string &key, const bool &value);
template void SetOptionsNg<int>(const std::string &key, const int &value);
template void SetOptionsNg<double>(const std::string &key, const double &value);
template void SetOptionsNg<std::string>(const std::string &key, const std::string &value);
template void SetOptionsNg<long>(const std::string &key, const long &value);
template void SetOptionsNg<uint8_t>(const std::string &key, const uint8_t &value);
template void SetOptionsNg<std::map<int, int>>(const std::string &key, const std::map<int, int> &value);
template void SetOptionsNg<std::map<long, long>>(const std::string &key, const std::map<long, long> &value);
template void SetOptionsNg<std::vector<int>>(const std::string &key, const std::vector<int> &value);
template void SetOptionsNg<std::vector<std::string>>(const std::string &key, const std::vector<std::string> &value);

} // namespace config
} // namespace npu::tile_fwk
