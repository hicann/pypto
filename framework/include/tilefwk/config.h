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
 * \file config.h
 * \brief
 */

#pragma once
#include <string>
#include <map>
#include <vector>
#include <stdexcept>

namespace npu::tile_fwk {

enum class MachineScheduleConfig {
    /**
     * \brief Default schedule mode: L2CACHE_AFFINITY_SCH(disable) MULTI_CORE_FAIR_SCH(disable)
     */
    DEFAULT_SCH = 0x0,

    /**
     * \brief Dispatch the most recently ready task to maximize cache reuse
     */
    L2CACHE_AFFINITY_SCH = 0x1,

    /**
     * \brief Fair scheduling refers to maintaining as balanced a distribution of tasks across cores as possible,
     *        Enabling this configuration will introduce some additional public scheduling overhead.
     */
    MULTI_CORE_FAIR_SCH = 0x2
};

namespace config {
namespace experimental {
    bool IsType(const std::string &key, const std::type_info &type);
    void SetOption(const std::string &key, bool value);
    void SetOption(const std::string &key, int64_t value);
    void SetOption(const std::string &key, const char *value);
    void SetOption(const std::string &key, const std::string &value);
    void SetOption(const std::string &key, const std::vector<int64_t> &value);
    void SetOption(const std::string &key, const std::vector<std::string> &value);
    void SetOption(const std::string &key, const std::map<int64_t, int64_t> &value);
}

/**
 * \brief Check if option exists
 *
 * \param key config option key
 * \return true if option exists, false otherwise
 */
bool HasOption(const std::string &key);

/**
 * \brief Check if option type is compatible with T
 *
 * \tparam T type to check compatibility with
 * \param key config option key
 * \return true if option type is compatible with T, false otherwise
 */
template <typename T>
bool IsType(const std::string &key) {
    using type = std::decay_t<T>;
    if constexpr (std::is_same_v<type, bool>) {
        return experimental::IsType(key, typeid(bool));
    } else if constexpr (std::is_integral_v<type>) {
        return experimental::IsType(key, typeid(int64_t));
    } else if constexpr (std::is_same_v<type, char *>) {
        return experimental::IsType(key, typeid(std::string));
    } else {
        return experimental::IsType(key, typeid(T));
    }
    return false;
}

/**
 * \brief Set config option value
 *
 * \tparam T type of config option value
 * \param key config option key
 * \param value config option value
 */
template <typename T>
void SetOption(const std::string &key, T &&value) {
    if (!HasOption(key)) {
        throw std::runtime_error("Option " + key + " does not exist");
    }
    if (!IsType<T>(key)) {
        throw std::runtime_error("Option " + key + " bad type");
    }
    using type = std::decay_t<T>;
    if constexpr (std::is_same_v<type, bool>) {
        experimental::SetOption(key, value);
    } else if constexpr (std::is_integral_v<type>) {
        experimental::SetOption(key, static_cast<int64_t>(value));
    } else {
        experimental::SetOption(key, value);
    }
}

inline void SetOption(const std::string &key, const char *value) {
    SetOption(key, std::string(value));
}

/**
 * \brief Set pass options
 *
 * \param key config option key
 *  - pg_upper_bound:
 *      upper bound of schedule cycles for each subgraph
 *      default: 512
 *  - pg_lower_bound:
 *      lower bound of schedule cycles for each subgraph
 *      default: 10000
 * \param value config option value
 */
template <typename T>
void SetPassOption(const std::string &key, const T &value) {
    SetOption("pass." + key, value);
}

/**
 * \brief Set codegen options
 *
 * \param key config option key
 * \param value config option value
 */
template <typename T>
void SetCodeGenOption(const std::string &key, const T &value) {
    SetOption("codegen." + key, value);
}

/**
 * \brief Set runtime options
 *
 * \param key config option key
 * \param value config option value
 */
template <typename T>
void SetRuntimeOption(const std::string &key, const T &value) {
    SetOption("runtime." + key, value);
}

/**
 * \brief Set host options
 *
 * \param key config option key
 * \param value config option value
 */
template <typename T>
void SetHostOption(const std::string &key, const T &value) {
    SetOption("host." + key, value);
}

/**
 * \brief Set host options
 *
 * \param key config option key
 * \param value config option value
 */
template <typename T>
void SetVerifyOption(const std::string &key, const T &value) {
    SetOption("verify." + key, value);
}

/**
 * \brief Set tensor print options
 *
 * \param edgeItems print max items in tensor head and tail
 * \param precision print precision
 * \param threshold threshold to use ...
 * \param linewidth max line width
 */
void SetPrintOptions(int edgeItems, int precision, int threshold, int linewidth);

/**
 * \brief Set the Semantic Label object
 *
 * \param label semantic label
 * \note label will be attached to subsequent operations
 */
void SetSemanticLabel(const std::string &label, const char *filename = __builtin_FILE(),
                      int lineno = __builtin_LINE());

/**
 * \brief Set the Build static function or not
 *
 * \param isStatic true: build static function, false: build dynamic function
 */
void SetBuildStatic(bool isStatic);

/**
 * \brief Dump all config options
 *
 * \return std::string config options string
 */
std::string Dump();

/**
 * \brief Reset config options to default values
 */
void Reset();
}; // namespace config

} // namespace npu::tile_fwk
