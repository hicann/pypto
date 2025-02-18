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
 * \file op_registry.h
 * \brief
 */

#pragma once

#include <map>
#include <vector>
#include <cstdint>
#include <memory>
#include <string>

namespace npu::tile_fwk {
using OpImplFunc = void (*)(uint64_t);
class OpImplRegister {
public:
    explicit OpImplRegister(const std::string &opType);
    OpImplRegister(const OpImplRegister &registerData);
    OpImplRegister &operator=(const OpImplRegister &) = delete;
    OpImplRegister &operator=(OpImplRegister &&) = delete;
    ~OpImplRegister();
    void AddImplFunc(const std::map<uint64_t, OpImplFunc> &implFuncMap);
    void AddImplFunc(const uint64_t configKey, const OpImplFunc implFunc);
    std::vector<uint64_t> GetAllConfigKeys() const;
    OpImplFunc GetOpImplFunc(const uint64_t configKey) const;

private:
    std::string opType_;
    std::map<uint64_t, OpImplFunc> implFuncMap_;
};
using OpImplRegisterPtr = std::shared_ptr<OpImplRegister>;

class OpImplRegistry {
public:
    static OpImplRegistry &GetInstance();
    OpImplRegisterPtr CreateOrGetOpRegister(const std::string &opType);
    OpImplFunc GetOpImplFunc(const std::string &opType, const uint64_t configKey) const;
    std::vector<uint64_t> GetAllConfigKeys(const std::string &opType) const;

private:
    OpImplRegistry() {}
    ~OpImplRegistry() {}
    std::map<std::string, OpImplRegisterPtr> opRegisterMap_;
};

class OpImplRegistHelper {
public:
    explicit OpImplRegistHelper(const std::string &opType);
    ~OpImplRegistHelper();
    OpImplRegistHelper &ImplFunc(const std::map<uint64_t, OpImplFunc> &implFuncMap);
    OpImplRegistHelper &ImplFunc(const uint64_t configKey, const OpImplFunc keyToFunc);

private:
    OpImplRegisterPtr opRegister_;
};
} // namespace npu::tile_fwk

#define VAR_UNUSED __attribute__((unused))
#define REGISTER_OP_COUNTER(opType, name, counter) \
    static npu::tile_fwk::OpImplRegistHelper VAR_UNUSED name##counter = npu::tile_fwk::OpImplRegistHelper(#opType)
#define REGISTER_OP_COUNTER_NUMBER(opType, name, counter) REGISTER_OP_COUNTER(opType, name, counter)
#define REGISTER_OP(opType) REGISTER_OP_COUNTER_NUMBER(opType, op_impl_reg_##opType, __COUNTER__)
