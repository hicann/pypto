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

#ifndef CONFIG_MANAGER_NG_H
#define CONFIG_MANAGER_NG_H

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
    const Any &GetAnyConfig(const std::string &key) const;

    /**
     * \brief Returns a map of all configuration key-value pairs.
     */
    const std::map<std::string, Any> GetAllConfig() const;

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
     * \brief Get operation config (prefix: "operation.")
     */
    template <typename T>
    T GetOperationConfig(const std::string &key) const {
        return GetConfigAllType<T>("operation." + key);
    }

    /**
     * \brief Retrieves the CubeTile configuration.
     */
    CubeTile GetCubeTile() const {
        const Any& value = GetAnyConfig("cube_tile_shapes");
        return AnyCast<CubeTile>(value);
    }

    /**
     * \brief Retrieves the VecTile configuration as a VecTile structure.
     */
    VecTile GetVecTile() const {
        const Any& value = GetAnyConfig("vec_tile_shapes");

        return VecTile{AnyCast<std::vector<int64_t>>(value)};
}

    /**
     * \brief Retrieves the matrix size configuration as a vector of integers.
     */
    std::vector<int64_t> GetMatrixSize() const {
        const Any& value = GetAnyConfig("matrix_size");
        return AnyCast<std::vector<int64_t>>(value);
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

    void UpdateValueWithAny(const std::string &key, Any value);

    /**
     * \brief update a config value for the given key.
     * \param key The config key.
     * \param value The config value to set.
     */
    template <typename T>
    void UpdateValue(const std::string &key, T RawValue) {
        Any value = ConvertTtoAny(RawValue);
        UpdateValueWithAny(key, value);
    }

    /**
     * \brief clear the config in Scope
     */
    void Clear() {
        values_.clear();
    }

    template <typename T>
    T GetConfigAllType(const std::string &key) const {
        if constexpr (std::is_same_v<T, bool>) {
            return AnyCast<bool>(GetAnyConfig(key));
        } else if constexpr (std::is_integral_v<T>) {
            int64_t tmp = AnyCast<int64_t>(GetAnyConfig(key));
            return static_cast<T>(tmp);
        } else {
            return AnyCast<T>(GetAnyConfig(key));
        }
    }

    ConfigScope(ConfigScopePtr parent);
    ~ConfigScope();
private:
    friend struct ConfigManagerImpl;
    std::shared_ptr<ConfigScope> Clone();

    std::shared_ptr<ConfigScope> parent_;
    std::list<ConfigScope *> children_;
    std::map<std::string, Any> values_;

    std::string name_;
    std::string begin_file_;
    int begin_lino_{0};
    std::string end_file_;
    int end_lino_{0};

    template <typename T>
    Any ConvertTtoAny(T value) {
        if constexpr (std::is_same_v<T, bool>) {
            return Any(value);
        } else if constexpr (std::is_integral_v<T>) {
            return Any(static_cast<int64_t>(value));
        } else if constexpr (std::is_same_v<T, const char*>) {
            return Any(std::string(value));
        } else {
            return Any(value);
        }
    }
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
    static std::shared_ptr<ConfigScope> CurrentScope();

    /**
     * @brief Get the Global Scope object
     *
     * @return std::shared_ptr<ConfigScope>
     */
    static std::shared_ptr<ConfigScope> GlobalScope();

    /**
     * @brief change the currentScope
     */
    void PushScope(ConfigScopePtr scope);

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

    /**
     * @brief Get Global Config for cpp frontend
     *
     */
    template <typename T>
    static T GetGlobalConfig(const std::string &key) {
        return GetInstance().globalScope->GetConfig<T>("global." + key);
    }

    /**
     * @brief Set Global Config for cpp frontend
     *
     */
    template <typename T>
    static void SetGlobalConfig(const std::string &key, T value) {
        return GetInstance().globalScope->UpdateValue("global." + key, value);
    }

    /**
     * @brief Set Global Config for python frontend
     *
     */
    void SetGlobalConfig(std::map<std::string, Any> &&values, const char *file, int lino);

    ~ConfigManagerNg();

private:
    ConfigManagerNg();
    ConfigManagerNg(const ConfigManagerNg &) = delete;
    ConfigManagerNg &operator=(const ConfigManagerNg &) = delete;

private:
    std::unique_ptr<ConfigManagerImpl> impl_;
    ConfigScopePtr globalScope;
};

namespace config {
#define DEFINE_CONFIG_GROUP(group, prefix)                   \
template <typename T>                                    \
inline T Get##group##Option(const std::string &key) {    \
    return ConfigManagerNg::CurrentScope()->GetConfigAllType<T>(prefix "." + key); \
}

DEFINE_CONFIG_GROUP(CodeGen, "codegen")
DEFINE_CONFIG_GROUP(Pass, "pass")
DEFINE_CONFIG_GROUP(Runtime, "runtime")
DEFINE_CONFIG_GROUP(Host, "host")
DEFINE_CONFIG_GROUP(Verify, "verify")
DEFINE_CONFIG_GROUP(Debug, "debug")
DEFINE_CONFIG_GROUP(Operation, "operation")

} // namespace config

} // namespace npu::tile_fwk
#endif // CONFIG_MANAGER_NG_H
