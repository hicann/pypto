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
 * \file device_machine.h
 * \brief
 */

#pragma once
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include "device_utils.h"
#include "machine/utils/dynamic/spsc_queue.h"
#include "machine/utils/dynamic/dev_workspace.h"
#include "machine/utils/dynamic/device_task.h"
#include "tilefwk/core_func_data.h"
#include "tilefwk/aicpu_common.h"

namespace npu::tile_fwk::dynamic {
constexpr uint32_t MAX_MANAGER_AIV_NUM = 72;
const uint32_t READY_ID_FIX_CACHE_NUM = 512;
const uint32_t MAX_DAV_2210_SCHEDULE_AICPU_NUM = 3;
inline uint32_t CalcSchAicpuNumByBlockDim(uint32_t blockDim, uint32_t aiCpuNum, ArchInfo archInfo) {
    uint32_t maxScheCore = aiCpuNum - dynamic::MAX_OTHER_AICPU_NUM;
    if (archInfo == ArchInfo::DAV_2201) {
        maxScheCore = maxScheCore >= MAX_DAV_2210_SCHEDULE_AICPU_NUM ? MAX_DAV_2210_SCHEDULE_AICPU_NUM : maxScheCore;
    }

    if (blockDim > (maxScheCore - 1) * dynamic::MAX_MNG_AICORE_AVG_NUM) {
        return maxScheCore;
    }

    if (blockDim % dynamic::MAX_MNG_AICORE_AVG_NUM == 0) {
        return blockDim / dynamic::MAX_MNG_AICORE_AVG_NUM;
    }

    return blockDim / dynamic::MAX_MNG_AICORE_AVG_NUM + 1;
}

const int DEVICE_MAX_AICPU_NUM = 7;

struct SchduleContext {
    uint64_t waitTaskCnt_[AICORE_TYPE_NUM]{0,0};
    uint32_t corePendReadyCnt_[AICORE_TYPE_NUM]{0,0};
    uint32_t coreRunReadyCnt_[AICORE_TYPE_NUM]{0,0};
    uint32_t runReadyCoreIdx_[AICORE_TYPE_NUM][MAX_MANAGER_AIV_NUM];
    uint32_t lastPendReadyCoreIdx_[AICORE_TYPE_NUM]{0,0};
    uint64_t resolveHubCnt_{0};

    uint32_t readyIds[AICORE_TYPE_NUM][READY_ID_FIX_CACHE_NUM];
    uint32_t readyCount[AICORE_TYPE_NUM]{0,0};
    uint32_t sendCnt_[AICORE_TYPE_NUM]{0,0};
};

} // namespace npu::tile_fwk