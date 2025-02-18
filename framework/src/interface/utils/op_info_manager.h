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
 * \file op_info_manager.h
 * \brief
 */

#pragma once

#include <map>
#include <vector>
#include <mutex>
#include "tilefwk/tensor.h"

namespace npu::tile_fwk {
constexpr uint64_t MAIN_KEY_MASK = 0xFFFFFFFFFFFFF;
constexpr uint64_t SUB_KEY_OFFSET = 52;
constexpr uint64_t SUB_KEY_MASK = 0xFFF0000000000000;

class OpInfoManager {
public:
    OpInfoManager() = default;
    ~OpInfoManager() = default;
    static OpInfoManager &GetInstance();
    void SetOpTilingKey(uint64_t opTilingKey);
    uint64_t GetOpTilingKey() const;
    uint64_t GetNewSubTilingKey();
    uint64_t GetCurSubTilingKey() const;
    void SetOpType(const std::string &opType);
    const std::string &GetOpType() const;
    bool IsNotFabinCompile();
    std::vector<char>& GetControlBuffer();
    std::vector<char>& GetCustomJson();
    std::string &GetCustomOpJsonPath();
    std::string &GetOpFuncName();
    void *GetControlBinHandle(const std::string &controlJsonPath);
    void SetControlBinHandle(void *controlFlowBindHandle);
private:
  std::mutex mtx_;
  std::string opType_ = "tilefwk";
  uint64_t opTilingKey_{0};
  uint64_t subTilingKey_{0};
  std::vector<char> controlBuffer_ = {'0'};
  std::vector<char> customJson_ = {'0'};
  std::string controlFlowSoPath_;
  std::string funcName_;
  std::map<std::string, void*> controlBinHandle_;
};
}
