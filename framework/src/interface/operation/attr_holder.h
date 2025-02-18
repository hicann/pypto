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
 * \file attr_holder.h
 * \brief
 */

#pragma once
#include <iostream>
#include <map>
#include <string>
#include <any>

#include "tilefwk/symbolic_scalar.h"
#include "interface/inner/any.h"
#include "interface/utils/common.h"
#include "interface/utils/log.h"
#include "interface/utils/string_utils.h"
#include "interface/inner/element.h"

namespace npu::tile_fwk {
const std::string OP_ATTR_PREFIX = "op_attr_";
const std::string OP_EMUOP_PREFIX = "op_emuop_";

class AttrHolder {
protected:
    std::map<std::string, npu::tile_fwk::Any> attributes;

public:
    const std::map<std::string, npu::tile_fwk::Any> &GetAllAttr() const { return attributes; }
    std::map<std::string, npu::tile_fwk::Any> &GetAllAttr() { return attributes; }

    bool HasAttr(const std::string &key) const {
        if (key.empty()) {
            return false;
        }
        return attributes.find(key) != attributes.end();
    }

    // 设置属性值
    template <typename T>
    void SetAttr(const std::string &key, const T &value) {
        static_assert(!std::is_same_v<T, int>);
        static_assert(!std::is_same_v<T, std::vector<int>>);
        attributes[key] = value;
    }

    npu::tile_fwk::Any GetRawAttr(const std::string &key) const {
        auto it = attributes.find(key);
        if (it != attributes.end()) {
            return it->second;
        }
        return npu::tile_fwk::Any();
    }

    template <typename T>
    bool GetAttr(const std::string &key, T &value) const {
        static_assert(!std::is_same_v<T, int>);
        static_assert(!std::is_same_v<T, std::vector<int>>);
        auto it = attributes.find(key);
        if (it != attributes.end()) {
            if (it->second.Type() == typeid(T)) {
                value = npu::tile_fwk::AnyCast<T>(it->second);
            } else {
                std::cout << "Type mismatch: " << it->second.Type().name() << " != " << typeid(T).name() << std::endl;
                return false;
            }
        } else {
            return false;
        }
        return true;
    }

    template <typename T>
    T *GetAttr(const std::string &key) {
        auto it = attributes.find(key);
        if (it != attributes.end() && it->second.Type() == typeid(T)) {
            return AnyCast<T>(&it->second);
        }
        return nullptr;
    }

    // 移除属性
    void RemoveAttr(const std::string &key) {
        auto it = attributes.find(key);
        if (it != attributes.end()) {
            attributes.erase(it);
        } else {
            throw std::out_of_range("Attribute not found: " + key);
        }
    }

    void CopyAttrFrom(const AttrHolder &holder, const std::string &prefix) {
        for (const auto &pair : holder.attributes) {
            if (StringUtils::StartsWith(pair.first, prefix)) {
                attributes[pair.first] = pair.second;
            }
        }
    }

    std::string DumpAttr() const {
        std::ostringstream oss;
        int index = 0;
        for (auto &it : attributes) {
            oss << ((index++ == 0) ? "" : " ");
            oss << "#" << it.first << "{" << DumpAttr(it.first) << "}";
        }
        return oss.str();
    }

    // 打印所有属性
    std::string DumpAttr(const std::string &key) const {
        auto it = attributes.find(key);
        if (it == attributes.end()) {
            return "Invalid attribute key " + key;
        }

        std::string result;
        if (it->second.Type() == typeid(int64_t)) {
            result = std::to_string(npu::tile_fwk::AnyCast<int64_t>(it->second));
        }  else if (it->second.Type() == typeid(float)) {
            result = std::to_string(npu::tile_fwk::AnyCast<float>(it->second));
        } else if (it->second.Type() == typeid(double)) {
            result = std::to_string(npu::tile_fwk::AnyCast<double>(it->second));
        } else if (it->second.Type() == typeid(std::string)) {
            result = npu::tile_fwk::AnyCast<std::string>(it->second);
        } else if (it->second.Type() == typeid(bool)) {
            result = std::to_string(npu::tile_fwk::AnyCast<bool>(it->second));
        } else if (it->second.Type() == typeid(std::vector<int64_t>)){
            result = IntVecToStr(npu::tile_fwk::AnyCast<std::vector<int64_t>>(it->second));
        } else if (it->second.Type() == typeid(Element)) {
            auto tensorElement = npu::tile_fwk::AnyCast<Element>(it->second);
            if (tensorElement.IsSigned()) {
                result = std::to_string(tensorElement.GetSignedData());
            } else if (tensorElement.IsUnsigned()) {
                result = std::to_string(tensorElement.GetUnsignedData());
            } else if (tensorElement.IsFloat()) {
                result = std::to_string(tensorElement.GetFloatData());
            }
        } else if (it->second.Type() == typeid(SymbolicScalar)) {
            auto scalar = npu::tile_fwk::AnyCast<SymbolicScalar>(it->second);
            result = scalar.Dump();
        } else if (it->second.Type() == typeid(std::vector<SymbolicScalar>)) {
            auto scalarList = npu::tile_fwk::AnyCast<std::vector<SymbolicScalar>>(it->second);
            std::ostringstream oss;
            oss << "[";
            for (size_t k = 0; k < scalarList.size(); k++) {
                oss << ((k != 0) ? "," : "") << scalarList[k].Dump();
            }
            oss << "]";
            result = oss.str();
        } else if (it->second.Type() == typeid(std::vector<bool>)) {
            auto scalarList = npu::tile_fwk::AnyCast<std::vector<bool>>(it->second);
            result = IntVecToStr<bool>(scalarList);
        } else {
            result += "unsupported type ";
            result += it->second.Type().name();
        }
        return result;
    }

    nlohmann::json DumpAttrJson() const {
        nlohmann::json attrJson;
        for (const auto &pair : attributes) {
            attrJson[pair.first] = DumpAttr(pair.first);
        }
        return attrJson;
    }

    nlohmann::json DumpAttrJson(const std::string &key) const {
        auto iter = attributes.find(key);
        if (iter != attributes.end()) {
            auto &second = iter->second;
            try {
                if (second.Type() == typeid(int64_t)) {
                    return nlohmann::json(npu::tile_fwk::AnyCast<int64_t>(second));
                } else if (second.Type() == typeid(std::vector<int64_t>)) {
                    return nlohmann::json(npu::tile_fwk::AnyCast<std::vector<int64_t>>(second));
                } else if (second.Type() == typeid(std::vector<float>)) {
                    return nlohmann::json(npu::tile_fwk::AnyCast<std::vector<float>>(second));
                } else if (second.Type() == typeid(std::vector<bool>)) {
                    return nlohmann::json(npu::tile_fwk::AnyCast<std::vector<bool>>(second));
                } else if (second.Type() == typeid(double)) {
                    return nlohmann::json(npu::tile_fwk::AnyCast<double>(second));
                } else if (second.Type() == typeid(float)) {
                    return nlohmann::json(npu::tile_fwk::AnyCast<float>(second));
                } else if (second.Type() == typeid(std::string)) {
                    return nlohmann::json(npu::tile_fwk::AnyCast<std::string>(second));
                } else if (second.Type() == typeid(bool)) {
                    return nlohmann::json(npu::tile_fwk::AnyCast<bool>(second));
                } else if (second.Type() == typeid(Element)) {
                    return ToJson(npu::tile_fwk::AnyCast<Element>(second));
                } else {
                    return nlohmann::json("Unsupported type");
                }
            } catch (const std::bad_any_cast &) {
                std::cout << "Bad any cast" << second.Type().name();
            }
        }
        return nlohmann::json();
    }

    void LoadVecAttr(const std::string &key, const std::vector<nlohmann::json> &vec) {
        if (vec[0].is_string()) {
            std::vector<std::string> strVec;
            for (const auto &j : vec) {
                strVec.emplace_back(j.get<std::string>());
            }
            SetAttr(key, strVec);
        } else if (vec[0].is_number()) {
            if (vec[0].is_number_integer()) {
                std::vector<int64_t> intVec;
                for (const auto &j : vec) {
                    intVec.emplace_back(j.get<int64_t>());
                }
                SetAttr(key, intVec);
            } else {
                std::vector<float> floatVec;
                for (const auto &j : vec) {
                    floatVec.emplace_back(j.get<float>());
                }
                SetAttr(key, floatVec);
            }
        } else if (vec[0].is_boolean()) {
            std::vector<bool> boolVec;
            for (const auto &j : vec) {
                boolVec.emplace_back(j.get<bool>());
            }
            SetAttr(key, boolVec);
        } else {
            return;
        }
    }

    void LoadAttrJson(const std::string &key, const nlohmann::json &attrJson) {
        try {
            if (attrJson.is_array()) {
                // 处理数组
                std::vector<nlohmann::json> vec;
                for (const auto &elem : attrJson) {
                    vec.push_back(elem);
                }
                if (!vec.empty()) {
                    LoadVecAttr(key, vec);
                }
            } else if (attrJson.is_object()) {
                SetAttr(key, parseElement(attrJson));
            } else if (attrJson.is_string()) {
                SetAttr(key, attrJson.get<std::string>());
            } else if (attrJson.is_number()) {
                if (attrJson.is_number_integer()) {
                    SetAttr(key, attrJson.get<int64_t>());
                } else {
                    SetAttr(key, attrJson.get<float>());
                }
            } else if (attrJson.is_boolean()) {
                SetAttr(key, attrJson.get<bool>());
            } else if (attrJson.is_null()) {
                return;
            }
        } catch (...) {
            ALOG_ERROR_F("json parse error");
        }
    }
};
} // namespace npu::tile_fwk
