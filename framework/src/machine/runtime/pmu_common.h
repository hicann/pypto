/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file pmu_common.h
 * \brief
 */

#ifndef SRC_MACHINE_RUNTIME_PMU_COMMON_H
#define SRC_MACHINE_RUNTIME_PMU_COMMON_H

#include <vector>
#include <cstdint>
#include "interface/utils/log.h"
#include "machine/utils/machine_ws_intf.h"

namespace npu::tile_fwk {

// pmu event type
constexpr int32_t ARITHMETIC_UTILIZATION = 1;
constexpr int32_t PIPE_UTILIZATION = 2;
constexpr int32_t MEMORY = 4;
constexpr int32_t MEMORY_L0 = 5;
constexpr int32_t RESOURCE_CONFLICT_RATION = 6;
constexpr int32_t MEMORY_UB = 7;
constexpr int32_t L2_CACHE = 8;

constexpr int PMU_EVENT_TYPE_MAX_DAV2201 = 8;
constexpr int PMU_EVENT_TYPE_MAX_DAV3510 = 10;

class PmuCommon {
public:
    static void InitPmuEventType(const ArchInfo& archInfo, std::vector<int64_t>& pmuEvtType);
};

} // namespace npu::tile_fwk

#endif // SRC_MACHINE_RUNTIME_PMU_COMMON_H
