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
 * \file PvMemAllocator.cpp
 * \brief
 */

#include "PvMemAllocator.h"

namespace CostModel {
    PvMemAllocator::PvMemAllocator() {
        hbmParaBase_ = 0xffff8000;
        codeBase_ = 0xffffc000;
        argBase_ = 0x30000000;
        workspaceBase_ = 0x80000000;
    }

    uint64_t PvMemAllocator::AllocWorkspace(uint64_t size) {
        uint64_t addr = workspaceBase_;
        workspaceBase_ += size;
        return addr;
    }
    
    uint64_t PvMemAllocator::AllocArg(uint64_t size) {
        uint64_t addr = argBase_;
        argBase_ += ((size + 128 - 1) / 128 * 128);
        return addr;
    }
    
    uint64_t PvMemAllocator::AllocCode(uint64_t size) {
        (void)size;
        uint64_t addr = codeBase_;
        return addr;
    }

} // namespace CostModel
