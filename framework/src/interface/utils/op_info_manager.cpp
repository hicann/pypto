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
 * \file op_info_manager.cpp
 * \brief
 */

#include "op_info_manager.h"

namespace npu::tile_fwk {
OpInfoManager &OpInfoManager::GetInstance() {
  static OpInfoManager instance;
  return instance;
}

// need add check
void OpInfoManager::SetOpTilingKey(uint64_t opTilingKey) {
  opTilingKey_ = (opTilingKey & MAIN_KEY_MASK);
  return;
}

uint64_t OpInfoManager::GetOpTilingKey() const {
  return opTilingKey_;
}

// need check
uint64_t OpInfoManager::GetNewSubTilingKey() {
  std::lock_guard<std::mutex> lock(mtx_);
  subTilingKey_++;
  uint64_t cur_key = (subTilingKey_ << SUB_KEY_OFFSET) | opTilingKey_;
  return cur_key;
}

uint64_t OpInfoManager::GetCurSubTilingKey() const {
  uint64_t cur_key = (subTilingKey_ << SUB_KEY_OFFSET) | opTilingKey_;
  return cur_key;
}

void OpInfoManager::SetOpType(const std::string &opType) {
  opType_ = opType;
  return;
}

const std::string &OpInfoManager::GetOpType() const {
  return opType_;
}

std::vector<char>& OpInfoManager::GetControlBuffer() {
  return controlBuffer_;
}

std::vector<char>& OpInfoManager::GetCustomJson() {
  return customJson_;
}

std::string &OpInfoManager::GetCustomOpJsonPath() {
  return controlFlowSoPath_;
}

void *OpInfoManager::GetControlBinHandle(const std::string &controlJsonPath) {
 if (controlBinHandle_.find(controlJsonPath) != controlBinHandle_.end()) {
    return controlBinHandle_[controlJsonPath];
 }
 return nullptr;
}

void OpInfoManager::SetControlBinHandle(void *controlFlowBindHandle) {
  controlBinHandle_[controlFlowSoPath_] = controlFlowBindHandle;
}

std::string &OpInfoManager::GetOpFuncName() {
  return funcName_;
}
}

