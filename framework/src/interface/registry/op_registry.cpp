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
 * \file op_registry.cpp
 * \brief
 */

#include "tilefwk/op_registry.h"
#include "tilefwk/error.h"
#include "interface/utils/op_info_manager.h"

namespace npu::tile_fwk {
OpImplRegister::OpImplRegister(const std::string &opType) : opType_(opType) {}

OpImplRegister::OpImplRegister(const OpImplRegister &registerData) {
    this->opType_ = registerData.opType_;
    this->implFuncMap_ = registerData.implFuncMap_;
}

OpImplRegister::~OpImplRegister() {}

void OpImplRegister::AddImplFunc(const std::map<uint64_t, OpImplFunc> &implFuncMap) {
    for (const auto &iter : implFuncMap) {
      ASSERT((iter.first & SUB_KEY_MASK) == 0) << "Config key only allow use low 52 bit!";
      implFuncMap_.emplace(iter.first, iter.second);
    }
}

void OpImplRegister::AddImplFunc(const uint64_t configKey, const OpImplFunc implFunc) {
    if (implFunc == nullptr) {
        return;
    }
    ASSERT((configKey & SUB_KEY_MASK) == 0) << "Config key only allow use low 52 bit!";
    implFuncMap_.emplace(configKey, implFunc);
}

std::vector<uint64_t> OpImplRegister::GetAllConfigKeys() const {
    std::vector<uint64_t> configKeys;
    for (const auto &item : implFuncMap_) {
        configKeys.emplace_back(item.first);
    }
    return configKeys;
}

OpImplFunc OpImplRegister::GetOpImplFunc(const uint64_t configKey) const {
    auto iter = implFuncMap_.find(configKey);
    return iter == implFuncMap_.end() ? nullptr : iter->second;
}

OpImplRegistry &OpImplRegistry::GetInstance() {
    static OpImplRegistry instance;
    return instance;
}

OpImplRegisterPtr OpImplRegistry::CreateOrGetOpRegister(const std::string &opType) {
    auto iter = opRegisterMap_.find(opType);
    if (iter == opRegisterMap_.end()) {
        OpImplRegisterPtr opRegister = std::make_shared<OpImplRegister>(opType);
        opRegisterMap_.emplace(opType, opRegister);
        return opRegister;
    } else {
        return iter->second;
    }
}

OpImplFunc OpImplRegistry::GetOpImplFunc(const std::string &opType, const uint64_t configKey) const {
    auto iter = opRegisterMap_.find(opType);
    if (iter == opRegisterMap_.end()) {
        return nullptr;
    }
    OpInfoManager::GetInstance().SetOpTilingKey(configKey);
    OpInfoManager::GetInstance().SetOpType(opType);
    return iter->second->GetOpImplFunc(configKey);
}

std::vector<uint64_t> OpImplRegistry::GetAllConfigKeys(const std::string &opType) const {
    std::vector<uint64_t> configKeys;
    auto iter = opRegisterMap_.find(opType);
    if (iter != opRegisterMap_.end()) {
        configKeys = iter->second->GetAllConfigKeys();
    }
    return configKeys;
}

OpImplRegistHelper::OpImplRegistHelper(const std::string &opType) {
    opRegister_ = OpImplRegistry::GetInstance().CreateOrGetOpRegister(opType);
}

OpImplRegistHelper::~OpImplRegistHelper() {}

OpImplRegistHelper &OpImplRegistHelper::ImplFunc(const std::map<uint64_t, OpImplFunc> &implFuncMap) {
    if (opRegister_ != nullptr) {
        opRegister_->AddImplFunc(implFuncMap);
    }
    return *this;
}
OpImplRegistHelper &OpImplRegistHelper::ImplFunc(const uint64_t configKey, const OpImplFunc implFunc) {
    if (opRegister_ != nullptr) {
        opRegister_->AddImplFunc(configKey, implFunc);
    }
    return *this;
}
}
