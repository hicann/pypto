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
 * \file ini_parser.cpp
 * \brief
 */

#include "ini_parser.h"
#include "interface/utils/file_utils.h"
#include "tilefwk/tilefwk_log.h"
namespace npu {
namespace tile_fwk {
const std::string platformConfigEnv = "PLATFORM_CONFIG_PATH";
const std::string version = "version";
const std::string instrinsicMap = "AICoreintrinsicDtypeMap";
const std::string aic = "AIC";
const std::string aiv = "AIV";
const std::string aicVersion = "AIC_version";
const std::string aivVersion = "AIV_version";

inline void trim(std::string& str) {
    size_t lastPos = str.find_last_not_of(" \t\r\n");
    if (lastPos != std::string::npos) {
        str.erase(lastPos + 1);
    } else {
        str.clear();
    }
}

Status INIParser::Initialize(const std::string& iniFilePath) {
    FUNCTION_LOGI("Start to parse ini_file %s.", iniFilePath.c_str());
    if (ReadINIFile(iniFilePath) != SUCCESS) {
        FUNCTION_LOGE("ReadINIFile failed.");
        return FAILED;
    }
    return SUCCESS;
}

Status INIParser::ReadINIFile(const std::string& filepath) {
    data_.clear();
    std::ifstream file(filepath);
    if (!file.is_open()) {
        FUNCTION_LOGE("Failed to open ini file: %s.", filepath.c_str());
        return FAILED;
    }
    std::string line;
    std::string section;
    while (std::getline(file, line)) {
        trim(line);
        if (line.empty()) {
            continue;
        }
        if (line.front() == '[' && line.back() == ']') {
            constexpr size_t kLeftBracketLen  = std::char_traits<char>::length("[");
            constexpr size_t kRightBracketLen = std::char_traits<char>::length("]");
            if (line.size() <= kLeftBracketLen + kRightBracketLen) {
                continue;
            }
            section = line.substr(kLeftBracketLen, line.size() - kLeftBracketLen - kRightBracketLen);
            continue;
        }
        size_t equalPos = line.find('=');
        if (equalPos == std::string::npos) {
            FUNCTION_LOGW("Illegal ini format[%s].", line.c_str());
            continue;
        }
        std::string key = line.substr(0, equalPos);
        std::string value = line.substr(equalPos + 1);
        if (key.empty()) {
            FUNCTION_LOGW("Empty attribute[%s].", line.c_str());
            continue;
        }
        data_[section][key] = value;
    }
    file.close();
    return SUCCESS;
}

Status INIParser::GetStringVal(const std::string& column, const std::string& key, std::string& val) {
    val.clear();
    if (data_.find(column) == data_.end()) {
        FUNCTION_LOGE("Cannot find attr 'version' from the ini file.");
        return FAILED;
    }
    auto value = data_[column];
    if (value.find(key) == value.end()) {
        FUNCTION_LOGW("Cannot find attr '%s' from the [version] tab.", key.c_str());
        return SUCCESS;
    }
    val = value[key];
    return SUCCESS;
}

Status INIParser::GetSizeVal(const std::string& column, const std::string& key, size_t& val) {
    std::string valStr;
    const size_t max_size_t = std::numeric_limits<size_t>::max();
    if (GetStringVal(column, key, valStr) != SUCCESS) {
        FUNCTION_LOGE("GetStringVal FAILED.");
        return FAILED;
    }
    val = 0UL;

    constexpr int    kRadix10    = 10;
    constexpr int    kMaxDigit10 = kRadix10 - 1;

    for (const char &c : valStr) {
        int digit = c - '0';
        if (digit < 0 || digit > kMaxDigit10) {
            FUNCTION_LOGE("Cannot convert string to size_t: %s.", valStr.c_str());
            return FAILED;
        }
        if (val > (max_size_t - digit) / kRadix10) {
            FUNCTION_LOGE("Overflow data: %s.", valStr.c_str());
            return FAILED;
        }
        val = val * kRadix10 + digit;
    }
    return SUCCESS;
}

Status INIParser::GetCCECVersion(std::unordered_map<std::string, std::string>& ccecVersion) {
    ccecVersion.clear();
    if (data_.find(version) == data_.end()) {
        FUNCTION_LOGE("Cannot find attribute 'version' from the ini file.");
        return FAILED;
    }
    auto versionVal = data_[version];
    std::string coreType;
    for (const auto &pair : versionVal) {
        if (FilterCCECVersion(pair.first, coreType)) {
            ccecVersion[coreType] = pair.second;
        }
    }
    return SUCCESS;
}

Status INIParser::GetCoreVersion(std::unordered_map<std::string, std::string>& curVersion) {
    curVersion.clear();
    if (data_.find(version) == data_.end()) {
        FUNCTION_LOGE("Cannot find attribute 'version' from the ini file.");
        return FAILED;
    }
    auto versionVal = data_[version];
    if (versionVal.find(aicVersion) != versionVal.end()) {
        curVersion[aic] = versionVal[aicVersion];
    }
    if (versionVal.find(aivVersion) != versionVal.end()) {
        curVersion[aiv] = versionVal[aivVersion];
    }
    return SUCCESS;
}

Status INIParser::GetDataPath(std::vector<std::vector<std::string>>& dataPath) {
    if (data_.find(instrinsicMap) == data_.end()) {
        FUNCTION_LOGE("Cannot find attribute '%s' from the ini file.", instrinsicMap.c_str());
        return FAILED;
    }
    std::string from;
    std::string to;
    std::string direction;
    std::set<std::string> directions;
    std::vector<std::string> curPath;
    auto instrinsics = data_[instrinsicMap];
    for (const auto &pair : instrinsics) {
        if (FilterDirections(pair.second, direction)) {
            directions.insert(direction);
        }
    }
    for (const auto &rec : directions) {
        if (FilterDataPath(rec, from, to)) {
            curPath = {from, to};
            dataPath.emplace_back(curPath);
        }
    }
    return SUCCESS;
}

bool INIParser::FilterCCECVersion(const std::string& key, std::string &coreType) {
    const std::string prefix = "CCEC_";
    const std::string suffix = "_version";
    const size_t prefixLen = prefix.length();
    const size_t suffixLen = suffix.length();
    if (key.length() >= (prefixLen + suffixLen) &&
        key.substr(0, prefixLen) == prefix &&
        key.substr(key.length() - suffixLen) == suffix) {
        coreType = key.substr(prefixLen, key.length() - prefixLen - suffixLen);
        return true;
    } else {
        return false;
    }
}

bool INIParser::FilterDirections(const std::string& value, std::string &part) {
    const std::string prefix = "Intrinsic_data_move";
    const char middle = '_';
    const char direction = '2';
    const char last = '|';
    size_t lastPos = value.find_last_of(last);
    auto tmpPart = value.substr(0, lastPos);
    if (value.find(prefix) != 0 || tmpPart.find(direction) >= tmpPart.size()) {
        return false;
    }
    size_t middlePos = tmpPart.find_last_of(middle);
    part = value.substr(middlePos + 1, lastPos - middlePos - 1);
    return true;
}

bool INIParser::FilterDataPath(const std::string& part, std::string &from, std::string &to) {
    const std::string direction = "2";
    size_t sepPos = part.find(direction);
    if (sepPos == std::string::npos || sepPos == 0 || sepPos == part.length() - 1) {
        return false;
    }
    from = part.substr(0, sepPos);
    to = part.substr(sepPos + 1);
    return true;
}
}  // namespace tile_fwk
}  // namespace npu