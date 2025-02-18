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
 * \file load_aicpu_op.h
 * \brief
 */


#ifndef LOAD_AICPU_OP_H
#define LOAD_AICPU_OP_H
#include <string>
#include <vector>
#include <memory>
#ifdef BUILD_WITH_CANN
#include "runtime/mem.h"
#include "machine/utils/machine_ws_intf.h"
#endif
#include <unordered_map>

namespace npu::tile_fwk {
class LoadAicpuOp
{
private:
    rtFuncHandle funcHandle_;
    void *customBinHandle_ = nullptr;
    std::string builtInOpJsonPath_;
    std::unordered_map<std::string, rtFuncHandle> builtInFuncMap_;
public:
    LoadAicpuOp() = default;
    ~LoadAicpuOp() {};
    int LaunchBuiltInOp(rtStream_t stream, AstKernelArgs *kArgs, const int &aicpuNum, const std::string &funcName);
    int GetBuiltInOpBinHandle();
    int LaunchCustomOp(rtStream_t stream, AstKernelArgs *kArgs, std::string &OpType);
    void CustomAiCpuSoLoad();
    void GenBuiltInOpInfo(const std::string &jsonPath);
    static LoadAicpuOp &GetInstance() {
      static LoadAicpuOp loadCustomAicpuOp;
      return loadCustomAicpuOp;
    }
};

} // namespace
#endif
