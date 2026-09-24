/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once

#include <cstdint>
#include <ctime>
#include <string>

namespace npu::tile_fwk {
// Startup only: call before installing handlers that can log. Never call from a log callback.
void InitializeLogTime();

namespace detail {
// Pure calendar conversion; tm_year is the full year and tm_mon is in [1, 12].
void CalLocalTime(std::tm* timeInfo, time_t sec, time_t tzone, int32_t dst);
} // namespace detail

std::string GetCurrentTime();
std::string GetCurrentTimeStr();
} // namespace npu::tile_fwk
