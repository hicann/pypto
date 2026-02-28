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
 * \file platform_manager.cpp
 * \brief
 */

#include "machine/platform/platform_manager.h"
#include <fstream>
#include "interface/utils/log.h"
#include "interface/utils/string_utils.h"
#include "interface/utils/file_utils.h"
#include "tilefwk/tilefwk_log.h"

namespace npu::tile_fwk {
#ifdef PROCESSOR_SUBPATH
    constexpr const char *CONFIG_SUBPATH = PROCESSOR_SUBPATH;
#else
    constexpr const char *CONFIG_SUBPATH = "";
#endif
namespace {
const int64_t INVALID_ITEM_VALUE = -1;
const std::string HOME_PATH_ENV_NAME = "ASCEND_HOME_PATH";
const std::string COMPILER_CONFIG_RELATIVE_PATH = "/compiler/data/platform_config/";
const std::string CONFIG_RELATIVE_PATH = "/data/platform_config/";
const std::string AICORE_INTRINSIC_DTYPE_MAP = "AICoreintrinsicDtypeMap";
const std::string VECTORCORE_INTRINSIC_DTYPE_MAP = "VectorCoreintrinsicDtypeMap";
const std::string INTRIC_PREFIX = "Intrinsic_";
const size_t FUNC_POS = 2;
}

const std::map<PlatformManager::PmIntItem,
      std::tuple<std::string, std::string, PlatformManager::PmItemParseFunc>> PlatformManager::kPmIntItemParseFuncMap {
    {PlatformManager::PmIntItem::AICORE_CNT, {"SoCInfo", "ai_core_cnt", &PlatformManager::ParseIntValue}},
    {PlatformManager::PmIntItem::VECCORE_CNT, {"SoCInfo", "vector_core_cnt", &PlatformManager::ParseIntValue}},
    {PlatformManager::PmIntItem::AICPU_CNT, {"SoCInfo", "ai_cpu_cnt", &PlatformManager::ParseIntValue}},
    {PlatformManager::PmIntItem::MEMORY_SIZE, {"SoCInfo", "memory_size", &PlatformManager::ParseIntValue}},
    {PlatformManager::PmIntItem::L2_TYPE, {"SoCInfo", "l2_type", &PlatformManager::ParseIntValue}},
    {PlatformManager::PmIntItem::L2_SIZE, {"SoCInfo", "l2_size", &PlatformManager::ParseIntValue}},
    {PlatformManager::PmIntItem::L2_PAGE_NUM, {"SoCInfo", "l2PageNum", &PlatformManager::ParseIntValue}},
    {PlatformManager::PmIntItem::AICORE_CUBE_FREQ, {"AICoreSpec", "cube_freq", &PlatformManager::ParseIntValue}},
    {PlatformManager::PmIntItem::AICORE_L0A_SIZE, {"AICoreSpec", "l0_a_size", &PlatformManager::ParseIntValue}},
    {PlatformManager::PmIntItem::AICORE_L0B_SIZE, {"AICoreSpec", "l0_b_size", &PlatformManager::ParseIntValue}},
    {PlatformManager::PmIntItem::AICORE_L0C_SIZE, {"AICoreSpec", "l0_c_size", &PlatformManager::ParseIntValue}},
    {PlatformManager::PmIntItem::AICORE_L1_SIZE, {"AICoreSpec", "l1_size", &PlatformManager::ParseIntValue}},
    {PlatformManager::PmIntItem::AICORE_UB_SIZE, {"AICoreSpec", "ub_size", &PlatformManager::ParseIntValue}},
    {PlatformManager::PmIntItem::AICORE_UB_BLOCK_SIZE, {"AICoreSpec", "ubblock_size", &PlatformManager::ParseIntValue}},
    {PlatformManager::PmIntItem::AICORE_UB_BANK_SIZE, {"AICoreSpec", "ubbank_size", &PlatformManager::ParseIntValue}},
    {PlatformManager::PmIntItem::AICORE_UB_BANK_NUM, {"AICoreSpec", "ubbank_num", &PlatformManager::ParseIntValue}},
    {PlatformManager::PmIntItem::AICORE_UB_BANK_GROUP_NUM, {"AICoreSpec", "ubbank_group_num", &PlatformManager::ParseIntValue}},
    {PlatformManager::PmIntItem::AICORE_DDR_RATE, {"AICoreMemoryRates", "ddr_rate", &PlatformManager::ParseIntValue}},
    {PlatformManager::PmIntItem::AICORE_DDR_READ_RATE, {"AICoreMemoryRates", "ddr_read_rate", &PlatformManager::ParseIntValue}},
    {PlatformManager::PmIntItem::AICORE_DDR_WRITE_RATE, {"AICoreMemoryRates", "ddr_write_rate", &PlatformManager::ParseIntValue}},
    {PlatformManager::PmIntItem::AICORE_L2_RATE, {"AICoreMemoryRates", "l2_rate", &PlatformManager::ParseIntValue}},
    {PlatformManager::PmIntItem::AICORE_L2_READ_RATE, {"AICoreMemoryRates", "l2_read_rate", &PlatformManager::ParseIntValue}},
    {PlatformManager::PmIntItem::AICORE_L2_WRITE_RATE, {"AICoreMemoryRates", "l2_write_rate", &PlatformManager::ParseIntValue}}
};

const std::map<PlatformManager::PmStrItem, std::tuple<std::string, std::string>> PlatformManager::kPmStrItemMap {
    {PlatformManager::PmStrItem::SOC_VERSION, {"version", "SoC_version"}},
    {PlatformManager::PmStrItem::SHORT_SOC_VERSION, {"version", "Short_SoC_version"}},
    {PlatformManager::PmStrItem::AIC_VERSION, {"version", "AIC_version"}}
};

PlatformManager::PlatformManager() : isInit_(false) {}

PlatformManager::~PlatformManager() {
    Reset();
}

PlatformManager& PlatformManager::Instance() {
    static PlatformManager platformManager;
    return platformManager;
}

bool PlatformManager::Initialize(const std::string &socVersion) {
    if (isInit_) {
        return true;
    }
    MACHINE_LOGI("Begin to initialize PlatformManager with soc version[%s].", socVersion.c_str());
    if (socVersion.empty()) {
        MACHINE_LOGW("Soc version is empty.");
        return false;
    }
    // get platform file path
    const char *envPath = std::getenv(HOME_PATH_ENV_NAME.c_str());
    if (envPath == nullptr) {
        MACHINE_LOGW("Env[%s] is not existed or empty.", HOME_PATH_ENV_NAME.c_str());
        return false;
    }

    std::string platformConfDir = std::string(envPath) + "/" + std::string(CONFIG_SUBPATH) + CONFIG_RELATIVE_PATH;
    if (RealPath(platformConfDir).empty()) {
        platformConfDir = std::string(envPath) + COMPILER_CONFIG_RELATIVE_PATH;
    }

    std::string platformFile = platformConfDir + socVersion + ".ini";
    if (RealPath(platformFile).empty()) {
        MACHINE_LOGW("Platform file[%s] is not existed.", platformFile.c_str());
        return false;
    }

    std::map<std::string, std::map<std::string, std::string>> contentMap;
    if (!ReadFileContent(platformFile, contentMap)) {
        MACHINE_LOGW("Fail to read platform file[%s].", platformFile.c_str());
        return false;
    }

    ParseStrItem(contentMap);
    ParseIntItem(contentMap);
    ParseInstrDtypeMap(contentMap);
    isInit_ = true;
    platformFile_ = platformFile;
    MACHINE_LOGI("PlatformManager has been initialized successfully with soc version[%s].", socVersion.c_str());
    return true;
}

bool PlatformManager::ReadFileContent(const std::string &filePath,
                                      std::map<std::string, std::map<std::string, std::string>> &contentMap) {
    std::ifstream ifs(filePath);
    if (!ifs.is_open()) {
        return false;
    }

    std::string itemKey;
    std::map<std::string, std::string> itemMap;
    std::string line;
    while (std::getline(ifs, line)) {
        if (line.empty() || line.find('#') == 0) {
            continue;
        }

        if (line.find('[') == 0) {
            if (!itemKey.empty() && !itemMap.empty()) {
                contentMap.emplace(make_pair(itemKey, itemMap));
            }
            itemKey.clear();
            itemMap.clear();

            size_t pos = line.rfind(']');
            if (pos == std::string::npos) {
                continue;
            }
            itemKey = line.substr(1, pos - 1);
            StringUtils::Trim(itemKey);
            continue;
        }

        size_t posEqual = line.find('=');
        if (posEqual == std::string::npos) {
            continue;
        }

        std::string key = line.substr(0, posEqual);
        StringUtils::Trim(key);
        std::string value = line.substr(posEqual + 1, line.length() - posEqual - 1);
        StringUtils::Trim(value);
        if (!key.empty() && !value.empty()) {
            itemMap.emplace(make_pair(key, value));
        }
    }

    if (!itemMap.empty() && !itemKey.empty()) {
        contentMap.emplace(make_pair(itemKey, itemMap));
    }

    ifs.close();
    return true;
}

void PlatformManager::ParseStrItem(const std::map<std::string, std::map<std::string, std::string>> &contentMap) {
    for (const auto &item : kPmStrItemMap) {
        auto iterFirstLayer = contentMap.find(std::get<0>(item.second));
        if (iterFirstLayer == contentMap.end()) {
            continue;
        }
        auto iterSecondLayer = iterFirstLayer->second.find(std::get<1>(item.second));
        if (iterSecondLayer == iterFirstLayer->second.end()) {
            continue;
        }
        pmStrItemArray_[static_cast<size_t>(item.first)] = iterSecondLayer->second;
        MACHINE_LOGI("[%s] [%s] is [%lld]", std::get<0>(item.second).c_str(), std::get<1>(item.second).c_str(),
                     pmStrItemArray_[static_cast<size_t>(item.first)]);
    }
}

void PlatformManager::ParseIntItem(const std::map<std::string, std::map<std::string, std::string>> &contentMap) {
    pmIntItemArray_.fill(INVALID_ITEM_VALUE);
    for (const auto &item : kPmIntItemParseFuncMap) {
        auto iterFirstLayer = contentMap.find(std::get<0>(item.second));
        if (iterFirstLayer == contentMap.end()) {
            continue;
        }
        auto iterSecondLayer = iterFirstLayer->second.find(std::get<1>(item.second));
        if (iterSecondLayer == iterFirstLayer->second.end()) {
            continue;
        }
        pmIntItemArray_[static_cast<size_t>(item.first)] = std::get<FUNC_POS>(item.second)(iterSecondLayer->second);
        MACHINE_LOGI("[%s] [%s] is [%lld]", std::get<0>(item.second).c_str(), std::get<1>(item.second).c_str(),
                     pmIntItemArray_[static_cast<size_t>(item.first)]);
    }
}

void PlatformManager::ParseInstrDtypeMap(std::map<std::string, std::map<std::string, std::string>> &contentMap) {
    auto iter = contentMap.find(AICORE_INTRINSIC_DTYPE_MAP);
    if (iter != contentMap.end()) {
        MappingInstrDtypeMap(iter->second, aiCoreIntrinsicDtypeMap_);
    }
    iter = contentMap.find(VECTORCORE_INTRINSIC_DTYPE_MAP);
    if (iter != contentMap.end()) {
        MappingInstrDtypeMap(iter->second, vectorCoreIntrinsicDtypeMap_);
    }
}

void PlatformManager::MappingInstrDtypeMap(const std::map<std::string, std::string> &contentMap,
                                           std::map<std::string, std::vector<std::string>> &instrDtypeMap) {
    for (const auto &item : contentMap) {
        if (item.second.empty()) {
            continue;
        }
        size_t intrcPos = item.second.find(INTRIC_PREFIX);
        size_t sepPos = item.second.find("|");
        if (intrcPos == std::string::npos || sepPos == std::string::npos) {
            continue;
        }
        instrDtypeMap.emplace(
            item.second.substr(intrcPos + INTRIC_PREFIX.size(), sepPos - intrcPos - INTRIC_PREFIX.size()),
                               StringUtils::Split(item.second.substr(sepPos + 1), ","));
    }
}

void PlatformManager::Finalize() {
    Reset();
}

bool PlatformManager::GetAiCoreIntrinsicDtype(const std::string &intrinsic, std::vector<std::string> &dtypeVec) const {
    if (intrinsic.empty()) {
        return false;
    }
    auto iter = aiCoreIntrinsicDtypeMap_.find(intrinsic);
    if (iter == aiCoreIntrinsicDtypeMap_.end()) {
        return false;
    }
    dtypeVec = iter->second;
    return true;
}

bool PlatformManager::GetVectorCoreIntrinsicDtype(const std::string &intrinsic, std::vector<std::string> &dtypeVec) const {
    if (intrinsic.empty()) {
        return false;
    }
    auto iter = vectorCoreIntrinsicDtypeMap_.find(intrinsic);
    if (iter == vectorCoreIntrinsicDtypeMap_.end()) {
        return false;
    }
    dtypeVec = iter->second;
    return true;
}

void PlatformManager::Reset() {
    pmIntItemArray_.fill(INVALID_ITEM_VALUE);
    pmStrItemArray_.fill("");
    isInit_ = false;
}

int64_t PlatformManager::ParseIntValue(const std::string &value) {
    if (value.empty()) {
        return INVALID_ITEM_VALUE;
    }
    try {
        int64_t num = std::stoll(value);
        return num;
    } catch (const std::invalid_argument& ia) {
        return INVALID_ITEM_VALUE;
    } catch (const std::out_of_range& oor) {
        return INVALID_ITEM_VALUE;
    }
}
}