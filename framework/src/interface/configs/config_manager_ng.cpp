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
 * \file config_manager_ng.cpp
 * \brief
 */
#include <string>
#include <map>
#include <typeinfo>
#include <fstream>
#include <sstream>
#include <list>
#include <stack>
#include <mutex>
#include <climits>
#include <utility>

#include <nlohmann/json.hpp>

#include "interface/inner/any.h"
#include "interface/utils/common.h"
#include "interface/utils/log.h"
#include "interface/utils/file_utils.h"
#include "interface/utils/string_utils.h"

#include "config_manager_ng.h"
#include "tilefwk/tile_shape.h"


namespace npu::tile_fwk {

namespace {
    std::mutex mtx;
}

struct TypeInfo {
    TypeInfo() = default;

    void LoadConf(const std::string &path) {
        std::ifstream infile(path);
        ASSERT(infile.is_open()) << "Open file " << path << " failed";
        nlohmann::json jdata;
        infile >> jdata;

        build_type_infos(jdata, "");
    }

    void build_type_infos(const nlohmann::json &jdata, const std::string &prefix) {
        if (jdata.contains("properties")) {
            auto &properties = jdata["properties"];
            for (auto &it : properties.items()) {
                const std::string &key = it.key();
                const nlohmann::json &value = it.value();
                if (prefix.empty()) {
                    build_type_infos(value, key);
                } else {
                    build_type_infos(value, prefix + "." + key);
                }
            }
        } else if (jdata.contains("type")) {
            const std::string &type = jdata["type"];
            if (type == "string") {
                typeInfos.insert({prefix, typeid(std::string)});
            } else if (type == "integer") {
                typeInfos.insert({prefix, typeid(int64_t)});
                int64_t minBound =
                    jdata.contains("minimum") ? jdata["minimum"].get<int64_t>() : INT_MIN;
                int64_t maxBound =
                    jdata.contains("maximum") ? jdata["maximum"].get<int64_t>() : INT_MAX;
                rangeInfos.insert({prefix, {minBound, maxBound}});
            } else if (type == "boolean") {
                typeInfos.insert({prefix, typeid(bool)});
            } else if (type == "array") {
                auto &jitem_type = jdata["items"]["type"];
                if (jitem_type == "string") {
                    typeInfos.insert({prefix, typeid(std::vector<std::string>)});
                } else if (jitem_type == "integer") {
                    typeInfos.insert({prefix, typeid(std::vector<int64_t>)});
                }
            } else if (type == "object") {
                const std::string &typeHints = jdata["typeHints"];
                if (typeHints == "intmap") {
                    typeInfos.insert({prefix, typeid(std::map<int64_t, int64_t>)});
                    int64_t minBound =
                        jdata.contains("key_minimum") ? jdata["key_minimum"].get<int64_t>() : INT_MIN;
                    int64_t maxBound =
                        jdata.contains("key_maximum") ? jdata["key_maximum"].get<int64_t>() : INT_MAX;
                    rangeInfos.insert({prefix + "_key", {minBound, maxBound}});
                    minBound =
                        jdata.contains("value_minimum") ? jdata["value_minimum"].get<int64_t>() : INT_MIN;
                    maxBound =
                        jdata.contains("value_maximum") ? jdata["value_maximum"].get<int64_t>() : INT_MAX;
                    rangeInfos.insert({prefix + "_val", {minBound, maxBound}});
                }
            } else {
                ALOG_ERROR("invalid type: ", type, " at ", prefix);
            }
        } else {
            ALOG_ERROR("type field missing");
        }
    }

    const std::type_info &Type(const std::string &name) const {
        if (typeInfos.find(name) == typeInfos.end()) {
            return typeid(void);
        }
        return typeInfos.at(name);
    }

    std::map<std::string, const std::type_info &> typeInfos;
    std::map<std::string, std::pair<int64_t, int64_t>> rangeInfos;
};

const Any &ConfigScope::GetAnyConfig(const std::string &key) const {
    if (values_.find(key) == values_.end()) {
        if (parent_) {
            return parent_->GetAnyConfig(key);
        }
        throw std::runtime_error("Config " + key + " not found");
    }
    return values_.at(key);
}

bool ConfigScope::HasConfig(const std::string &key) const {
    return values_.find(key) != values_.end() || (parent_ && parent_->HasConfig(key));
}

const std::type_info &ConfigScope::Type(const std::string &key) const {
    return ConfigManagerNg::GetInstance().Type(key);
}

ConfigScope::ConfigScope(ConfigScopePtr parent) : parent_(parent) {
    if (parent_) {
        parent_->children_.push_back(this);
    }
}

TileShape ConfigScope::GenerateTileShape() const {
    std::vector<int64_t> vecTile = GetConfig<std::vector<int64_t>>("vec_tile_shapes");
    CubeTile cubeTile = GetConfig<CubeTile>("cube_tile_shapes");
    DistTile distTile = GetConfig<DistTile>("dist_tile_shapes");
    std::vector<int64_t> matrixSize = GetConfig<std::vector<int64_t>>("matrix_size");
    TileShape tileShape(vecTile, cubeTile, distTile, matrixSize);
    return tileShape;
}

ConfigScope::~ConfigScope() {
    if (parent_) {
        parent_->children_.remove(this);
    }
}

void DumpValue(std::stringstream &os, const std::string &key, const Any &val, const std::string &prefix) {
    os << prefix << key << ": ";
    if (val.Type() == typeid(int64_t)) {
        os << (AnyCast<int64_t>(val));
    } else if (val.Type() == typeid(bool)) {
        os << AnyCast<bool>(val);
    } else if (val.Type() == typeid(std::string)) {
        os << (AnyCast<std::string>(val));
    } else if (val.Type() == typeid(std::vector<int64_t>)) {
        os << (AnyCast<std::vector<int64_t>>(val));
    } else if (val.Type() == typeid(std::vector<std::string>)) {
        os << (AnyCast<std::vector<std::string>>(val));
    } else if (val.Type() == typeid(std::map<int64_t, int64_t>)) {
        os << '{';
        bool is_first = true;
        for (auto &[k, v] : AnyCast<std::map<int64_t, int64_t>>(val)) {
            if (!is_first)
                os << ", ";
            os << "{" << k << ", " << v << "}";
            is_first = false;
        }
        os << '}';
    } else if (val.Type() == typeid(CubeTile)) {
        os << (AnyCast<CubeTile>(val).ToString());
    } else if (val.Type() == typeid(DistTile)) {
        os << (AnyCast<DistTile>(val).ToString());
    } else {
        os << "unknow type: " << val.Type().name();
    }
    os << "\n";
}

void DumpValues(std::stringstream &os, const std::map<std::string, Any> &values,
                const std::string &prefix) {
    for (const auto &[key, val] : values) {
        DumpValue(os, key, val, prefix);
    }
}

void DumpRange(
    std::stringstream &os,
    const std::type_info &type,
    const std::string &key,
    const std::map<std::string, std::pair<int64_t, int64_t>> &rangeInfos) {
    os << "Range: ";
    if (type == typeid(std::map<int64_t, int64_t>)) {
        os << "{[" << rangeInfos.at(key + "_key").first <<
            ", " << rangeInfos.at(key + "_key").second <<
            "], [" << rangeInfos.at(key + "_val").first <<
            ", " << rangeInfos.at(key + "_val").second << "]}";
    } else {
        os << "[" << rangeInfos.at(key).first <<
            ", " << rangeInfos.at(key).second << "]";
    }
}

std::string ConfigScope::ToString() const{
    auto values = GetAllConfig();
    std::stringstream os;
    DumpValues(os, values, "");
    os << "\n";
    return os.str();
}

const std::map<std::string, Any> ConfigScope::GetAllConfig() const{
    std::map<std::string, Any> values;
    auto scope = this;
    while (scope) {
        for (auto &[key, val] : scope->values_) {
            if (!values.count(key)) {
                values[key] = val;
            }
        }
        scope = scope->parent_.get();
    }
    return values;
}

void ConfigScope::AddValue(const std::string &key, Any value) {
    std::lock_guard<std::mutex> lock(mtx);
    values_[key] = value;
}

void ConfigScope::UpdateValueWithAny(const std::string &key, Any value) {
    if (!ConfigManagerNg::GetInstance().IsWithinRange(key, value)) {
        std::stringstream os("Option:");
        std::map<std::string, Any> node;
        node[key] = value;
        DumpValues(os, node, "");
        os << ", its value doesn't within the value range.";
        DumpRange(os, value.Type(), key, ConfigManagerNg::GetInstance().Range());
        os << "\n";
        throw std::runtime_error(os.str().c_str());
    }
    std::stringstream oss;
    DumpValue(oss, key, value, "");
    ALOG_DEBUG_F("Set option successfully: %s ", oss.str().c_str());
    std::lock_guard<std::mutex> lock(mtx);
    values_[key] = value;
}

struct ConfigManagerImpl {
    TypeInfo typeInfo;
    std::stack<ConfigScopePtr> scopes;
    ConfigScopePtr root;

    ConfigManagerImpl() {
        typeInfo.LoadConf(GetConfDir() + "tile_fwk_config_schema.json");
        root = std::make_shared<ConfigScope>(nullptr);
        root->name_ = "default";
        LoadConf();
        InitTileShape();
        scopes.push(root);
        auto global = std::make_shared<ConfigScope>(root);
        global->name_ = "global";
        scopes.push(global);
    }

    void PushScope(ConfigScopePtr scope) {
        // Ensure the provided scope is not null
        ASSERT(scope != nullptr) << "Cannot push a null scope";
        scopes.push(scope);
    }

    inline bool IntervalJudge(const int64_t &stand, const int64_t &lf, const int64_t &rf) const {
        return stand >= lf && stand <=rf;
    }

    bool IsWithinRange(const std::string &properties, const int64_t &value) const {
        return IntervalJudge(value, typeInfo.rangeInfos.at(properties).first, typeInfo.rangeInfos.at(properties).second);
    }

    bool IsWithinRange(const std::string &properties, const std::map<int64_t, int64_t> &value) const {
        auto ins = typeInfo.rangeInfos;
        for (auto &[lf, rf] : value) {
            if (!IntervalJudge(lf, ins.at(properties + "_key").first, ins.at(properties + "_key").second) ||
            !IntervalJudge(rf, ins.at(properties + "_val").first, ins.at(properties + "_val").second)) {
                return false;
            }
        }
        return true;
    }

    void BeginScope(const std::string &name, std::map<std::string, Any> &&values, const char *file, int lino) {
        auto scope = std::make_shared<ConfigScope>(scopes.top());
        scope->values_ = std::move(values);
        scope->begin_file_ = file;
        scope->begin_lino_ = lino;
        scope->name_ = name;
        scopes.push(scope);
    }

    void EndScope(const char *file, int lino) {
        /* at least default and global two levels */
        ASSERT(scopes.size() >= 0x2) << "No scope to pop";
        auto &scope = scopes.top();
        scope->end_file_ = file;
        scope->end_lino_ = lino;
        scopes.pop();
    }

    void SetScope(std::map<std::string, Any> &&values, const char *file, int lino) {
        auto scope = scopes.top();
        if (scope.use_count() > 1) { // clone if shared
            auto oldvalues = scopes.top()->values_;
            auto name = scopes.top()->name_;
            EndScope(file, lino);
            BeginScope(name, std::move(oldvalues), file, lino);
            scope = scopes.top();
        }
        for (auto &it : values) {
            scope->UpdateValueWithAny(it.first, it.second);
        }
    }

    void SetGlobalConfig(std::map<std::string, Any> &&values, const char *file, int lino) {
        if (values.empty()) {
            ALOG_WARN_F("No values provided to set in global config. Locations: %s:%d", file, lino);
            return;
        }
        for (auto &it : values) {
            try {
                root->AddValue(it.first, it.second);
                ALOG_DEBUG_F("Set option successfully. Key: %s", it.first.c_str());
            } catch (const std::exception &e) {
                ALOG_ERROR_F("Failed to set option. Key: %s, Error: %s", it.first.c_str(), e.what());
            }
        }
        ALOG_DEBUG_F("Set locations: %s:%d", file, lino);
    }

    void Dump(std::stringstream &os, ConfigScope *node, const std::string &prefix) {
        if (!node->begin_file_.empty()) {
            os << prefix << "scope_start: " << node->begin_file_ << ":" << node->begin_lino_ << "\n";
        }
        if (!node->end_file_.empty()) {
            os << prefix << "scope_end: " << node->end_file_ << ":" << node->end_lino_ << "\n";
        }
        if (!node->name_.empty()) {
            os << prefix << "scope: " << node->name_ << "\n";
        }
        DumpValues(os, node->values_, prefix);
        os << "\n";
        for (auto child : node->children_) {
            os << prefix << "--------\n";
            Dump(os, child, prefix + ' ');
        }
    }

    std::string GetOptionsTree() {
        std::stringstream os;
        Dump(os, root.get(), "");
        return os.str();
    }

private:
    std::string GetConfDir() { return GetCurrentSharedLibPath() + "/configs/"; }

    void LoadConf(const nlohmann::json &jdata, const std::string &prefix) {
        if (jdata.is_string()) {
            root->AddValue(prefix, jdata.get<std::string>());
        } else if (jdata.is_number()) {
            root->AddValue(prefix, jdata.get<int64_t>());
        } else if (jdata.is_boolean()) {
            root->AddValue(prefix, jdata.get<bool>());
        } else if (typeInfo.Type(prefix) == typeid(std::map<int64_t, int64_t>)) {
            std::map<int64_t, int64_t> mapJson;
            auto arr = jdata.get<std::vector<int64_t>>();
            for (size_t i = 0; i + 1 < arr.size(); i += 2) {
                mapJson[arr[i]] = arr[i + 1];
            }
            root->AddValue(prefix, mapJson);
        } else if (jdata.is_array()) {
            if (typeInfo.Type(prefix) == typeid(std::vector<int64_t>)) {
                root->AddValue(prefix, jdata.get<std::vector<int64_t>>());
            } else {
                root->AddValue(prefix, jdata.get<std::vector<std::string>>());
            }
        } else if (jdata.is_object()) {
            for (auto &it : jdata.items()) {
                const std::string &key = it.key();
                if (prefix.empty()) {
                    LoadConf(it.value(), key);
                } else {
                    LoadConf(it.value(), prefix + "." + key);
                }
            }
        }
    }

    void LoadConf() {
        std::string confPath = GetEnvVar("TILEFWK_CONFIG_PATH");
        if (confPath.empty()) {
            confPath = GetConfDir() + "tile_fwk_config.json";
        }
        std::ifstream ifs(confPath);
        ASSERT(ifs.is_open()) << "Open file " << confPath << " failed";
        nlohmann::json jdata;
        ifs >> jdata;
        LoadConf(jdata, "");
    }

    void InitTileShape() {
        TileShape tileShape;
        tileShape.Reset();
        root->AddValue("cube_tile_shapes", tileShape.GetCubeTile());
        root->AddValue("vec_tile_shapes", tileShape.GetVecTile().tile);
        root->AddValue("matrix_size", tileShape.GetMatrixSize());
        root->AddValue("dist_tile_shapes", tileShape.GetDistTile());
    }
};

void ConfigManagerNg::BeginScope(
    const std::string &name, std::map<std::string, Any> &&values, const char *file, int lino) {
    impl_->BeginScope(name, std::move(values), file, lino);
}

void ConfigManagerNg::EndScope(const char *file, int lino) {
    impl_->EndScope(file, lino);
}

void ConfigManagerNg::SetScope(std::map<std::string, Any> &&values, const char *file, int lino) {
    return impl_->SetScope(std::move(values), file, lino);
}

void ConfigManagerNg::SetGlobalConfig(std::map<std::string, Any> &&values, const char *file, int lino) {
    return impl_->SetGlobalConfig(std::move(values), file, lino);
}

void ConfigManagerNg::PushScope(ConfigScopePtr scope) {
    impl_->PushScope(scope);
}

std::shared_ptr<ConfigScope> ConfigManagerNg::CurrentScope() {
    return GetInstance().impl_->scopes.top();
}

std::shared_ptr<ConfigScope> ConfigManagerNg::GlobalScope() {
    return GetInstance().impl_->root;
}

bool ConfigManagerNg::IsWithinRange(const std::string &properties, Any &value) const {
    try {
        if (value.Type() == typeid(std::map<int64_t, int64_t>)) {
            return impl_->IsWithinRange(properties, AnyCast<std::map<int64_t, int64_t>>(value));
        } else if (value.Type() == typeid(int64_t)) {
            return impl_->IsWithinRange(properties, AnyCast<int64_t>(value));
        }
    } catch (const std::out_of_range &e) {
        ALOG_ERROR_F("key[%s] has been not loaded form tile_fwk_config_schema.json.", properties.c_str());
        return false;
    }
    return true;
}

const std::type_info &ConfigManagerNg::Type(const std::string &key) const {
    return impl_->typeInfo.Type(key);
}

const std::map<std::string, std::pair<int64_t, int64_t>> &ConfigManagerNg::Range() const {
    return impl_->typeInfo.rangeInfos;
}

std::string ConfigManagerNg::GetOptionsTree() {
    return impl_->GetOptionsTree();
}


ConfigManagerNg::ConfigManagerNg() : impl_(std::make_unique<ConfigManagerImpl>()) {
    globalScope = impl_->root;
}

ConfigManagerNg &ConfigManagerNg::GetInstance() {
    static ConfigManagerNg instance;
    return instance;
}

ConfigManagerNg::~ConfigManagerNg() = default;

namespace config{

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

void Reset() {
    ConfigManagerNg::CurrentScope()->Clear();
}


void Restore(std::shared_ptr<ConfigScope> config) {
    ConfigManagerNg::GetInstance().PushScope(config);
}

std::shared_ptr<ConfigScope> Duplicate() {
    return ConfigManagerNg::CurrentScope();
}

}
} // namespace npu::tile_fwk
