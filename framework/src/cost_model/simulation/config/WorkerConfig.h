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
 * \file WorkerConfig.h
 * \brief
 */

// generated from config.toml
#pragma once

#include <cstdint>
#include <string>
#include "cost_model/simulation/base/Config.h"

namespace CostModel {
struct WorkerConfig : public Config {
    WorkerConfig();
    uint64_t sessionNum = 320;
    uint64_t moeNum = 9;
    uint64_t layerNum = 61;
    uint64_t expertNum = 20;
    uint64_t attnBatch = 4;
    uint64_t ffnBatch = 8;
    uint64_t attnLatencyAvg = 100;
    uint64_t attnLatencySdv = 90;
    uint64_t attnLatencyMax = 320;
    uint64_t attnLatencyMin = 20;
    uint64_t ffnLatencyAvg = 20;
    uint64_t ffnLatencySdv = 0;
    uint64_t commLatencyAvg = 0;
    uint64_t commLatencySdv = 0;
    uint64_t layerWaitTime = 5;
    bool isBsp = false;
    bool attnPolling = true;
    bool ffnPolling = true;
    uint64_t layerStartVariation = 0;
    bool useFixedRandomSeed = false;
    uint64_t randomSeed = 16;
};
}