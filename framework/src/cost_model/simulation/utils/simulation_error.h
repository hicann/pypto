/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file simulation_error.h
 * \brief SIMULATION 组件错误分类、场景枚举与错误码常量。
 *        - F9XXXX：SIMULATION 内部已梳理报错（大流程 Category -> 子流程 Scene）。
 */

#pragma once

#include <cstdint>

namespace CostModel {

// =============================================================================
// 一、大流程：SimulationErrorCategory
// =============================================================================

enum class SimulationErrorCategory {
    INTERNEL_ERROR      = 90000U, // 0: 内部错误
    EXTERNAL_ERROR      = 91000U, // 1: 外部错误
    FORWARD_SIM          = 92000U,  // 2: 前仿
    POST_SIM           = 93000U, // 3: 后仿
    PRECISION_SIM       = 94000U, // 4: 精度仿真
    UNKNOWN              = 99000U, // 9: 未知/预留
};

// =============================================================================
// 二、子流程：各 Category 下的 ErrorScene 枚举（枚举值即错误码 F9xxxx）
// =============================================================================

enum class InternelErrorScene : uint32_t {
    UNKNOWN                   = 90099U
};

enum class ExternalErrorScene : uint32_t {
    INVALID_CONFIG                = 91001U,
    CONFIG_OUT_OF_RANGE           = 91002U,
    INVALID_CONFIG_NAME           = 91003U,
    PERMISSION_CHECK_ERROR        = 91004U,
    FILE_FORMAT_ERROR             = 91005U,
    FILE_CONTENT_ERROR            = 91006U,
    INVALID_PATH                  = 91007U,
    FILE_OPEN_FAILED              = 91008U,
    PYTHON_CMD_ERROR              = 91009U,
    UNKNOWN                       = 91099U
};

enum class ForwardSimErrorScene : uint32_t {
    INVALID_PIPE_TYPE    = 92001U,
    INVALID_DATA_TYPE    = 92002U,
    DEAD_LOCK            = 92003U,
    FUNC_NOT_SUPPORT     = 92004U,
    UNKNOWN              = 92099U
};

enum class PostSimErrorScene : uint32_t {
    UNKNOWN              = 93099U
};

enum class PrecisionSimErrorScene : uint32_t {
    NO_SO_EXISTS      = 94001U,
    CANN_LOAD_FAILED  = 94002U,
    UNKNOWN           = 94099U
};
}  // namespace npu::tile_fwk
