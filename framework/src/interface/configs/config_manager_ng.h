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
 * \file config_manager_ng.h
 * \brief
 */
#include <map>
#include <memory>
#include <list>
#include <string>

#include "interface/inner/any.h"
#include "tilefwk/tile_shape.h"

namespace npu::tile_fwk {

class ConfigScope;
struct ConfigManagerImpl;
using ConfigScopePtr = std::shared_ptr<ConfigScope>;

class ConfigScope {
public:
    /**
     * \brief Get the config value with the specific key. throw runtime_error if
     * the key is not found.
     */
    const Any &GetConfig(const std::string &key) const;

    /**
     * \brief Get the typed config value with the specific key.
     *
     */
    template <typename T>
    const T GetConfig(const std::string &key) const {
        return GetConfigAllType<T>(key);
    }

    /**
     * \brief Check if the config with the specific key exists.
     */
    bool HasConfig(const std::string &key) const;

    /**
     * \brief Get pass config (prefix: "pass.")
     */
    template <typename T>
    T GetPassConfig(const std::string &key) const {
        return GetConfigAllType<T>("pass." + key);
    }

    /**
     * \brief Get runtime config (prefix: "runtime.")
     */
    template <typename T>
    T GetRuntimeConfig(const std::string &key) const {
        return GetConfigAllType<T>("runtime." + key);
    }

    /**
     * \brief Get codegen config (prefix: "codegen.")
     */
    template <typename T>
    T GetCodegenConfig(const std::string &key) const {
        return GetConfigAllType<T>("codegen." + key);
    }

    /**
     * \brief Get host config (prefix: "host.")
     */
    template <typename T>
    T GetHostConfig(const std::string &key) const {
        return GetConfigAllType<T>("host." + key);
    }

    /**
     * \brief Get verify config (prefix: "verify.")
     */
    template <typename T>
    T GetVerifyConfig(const std::string &key) const {
        return GetConfigAllType<T>("verify." + key);
    }

    /**
     * \brief Generate a TileShape object from the current configuration scope.
     */
    TileShape GenerateTileShape() const;

    /**
     * \brief Return the type of the config value with the specific key. type void
     * if the key is not found.
     */
    const std::type_info &Type(const std::string &key) const;

    /**
     * \brief Return all configures in current scope
     *
     * \return std::string
     */
    std::string ToString() const;

    /**
     * \brief Add or update a config value for the given key.
     * \param key The config key.
     * \param value The config value to set.
     */
    void AddValue(const std::string &key, Any value);

    /**
     * \brief update a config value for the given key.
     * \param key The config key.
     * \param value The config value to set.
     */
    void UpdateValue(const std::string &key, Any value);

    ConfigScope(ConfigScopePtr parent);
    ~ConfigScope();
private:
    friend struct ConfigManagerImpl;
    std::shared_ptr<ConfigScope> Clone();
    template <typename T>
    T GetConfigAllType(const std::string &key) const {
        if constexpr (std::is_same_v<T, bool>) {
            return AnyCast<bool>(GetConfig(key));
        } else if constexpr (std::is_integral_v<T>) {
            int64_t tmp = AnyCast<int64_t>(GetConfig(key));
            return static_cast<T>(tmp);
        } else {
            return AnyCast<T>(GetConfig(key));
        }
    }

private:
    std::shared_ptr<ConfigScope> parent_;
    std::list<ConfigScope *> children_;
    std::map<std::string, Any> values_;

    std::string name_;
    std::string begin_file_;
    int begin_lino_{0};
    std::string end_file_;
    int end_lino_{0};
};

class ConfigManagerNg {
public:
    /**
     * \brief Begin a new scope with the given config values.
     *
     * \param values
     */
    void BeginScope(const std::string &name, std::map<std::string, Any> &&values,
        const char *file = __builtin_FILE(), int line = __builtin_LINE());

    /**
     * \brief End the current scope.
     *
     */
    void EndScope(const char *file = __builtin_FILE(), int line = __builtin_LINE());

    /**
     * @brief Set the Scope object
     * \brief Scope is not modifiable after it's begin, SetScope is just a syntax sugar for:
     * \code {.c}
     *   auto oldScope = CurrentScope();
     *   EndScope();
     *   BeginScope(oldScope.values + values);
     * \endcode
     *
     * @param values
     */
    void SetScope(std::map<std::string, Any> &&values,
        const char *file = __builtin_FILE(), int line = __builtin_LINE());

    /**
     * @brief Get the Current Scope object
     *
     * @return std::shared_ptr<ConfigScope>
     */
    std::shared_ptr<ConfigScope> CurrentScope() const;

    /**
     * @brief Get the type of the config value with the specific key. type void
     * if the key is not found.
     */
    const std::type_info &Type(const std::string &key) const;

    /**
     * @brief Get the range of the config value with the specific key.
     */
    const std::map<std::string, std::pair<int64_t, int64_t>> &Range() const;

    /**
    * \brief Check if the value is within the specified range.
    */
    bool IsWithinRange(const std::string &properties, Any &value) const;

    static ConfigManagerNg &GetInstance();

    std::string GetOptionsTree();

    ~ConfigManagerNg();

private:
    ConfigManagerNg();
    ConfigManagerNg(const ConfigManagerNg &) = delete;
    ConfigManagerNg &operator=(const ConfigManagerNg &) = delete;

private:
    std::unique_ptr<ConfigManagerImpl> impl_;
};
} // namespace npu::tile_fwk
