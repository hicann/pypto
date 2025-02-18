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
 * \file platform.h
 * \brief
 */

#pragma once
#ifndef COSTMODEL_PLATFORM_H
#define COSTMODEL_PLATFORM_H

#include <string>
#include <cstdint>

namespace CostModel {
class CostModelPlatform {
public:
    CostModelPlatform() = default;
    ~CostModelPlatform() = default;

    uint32_t GetCostModelPlatformRealPath(std::string &realPath);

private:
    std::string RealPath(const std::string &path);

    static std::string GetCurrentSharedLibPath();
};
} // namespace CostModel
#endif