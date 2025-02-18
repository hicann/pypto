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
 * \file ooo_schedule_statistic.h
 * \brief
 */

#ifndef OOO_SCHEDULE_STATISTIC_H
#define OOO_SCHEDULE_STATISTIC_H

#include <nlohmann/json.hpp>
#include "tilefwk/platform.h"
#include "interface/function/function.h"

using Json = nlohmann::json;

namespace npu {
namespace tile_fwk {
class OoOSchedulerCheck {
public:
    Status DoHealthCheck(Function *function, const std::string &fileName);
    void HealthCheckSpillInfo();
    double FormatUsageRate(double value);
    Status HealthCheckOoOSchedule();
    void HealthCheckBlockGraph(Function *function);
    std::string jsonFileName;
    bool doHealthCheck{false};
    int workspaceOffset{0};
    int clock{0};
    uint64_t spillCnt = 0;
    std::unordered_map<PipeType, uint64_t> pipeUsageCount = {
        {PipeType::PIPE_S, 0},
        {PipeType::PIPE_V, 0},
        {PipeType::PIPE_M, 0},
        {PipeType::PIPE_MTE1, 0},
        {PipeType::PIPE_MTE2, 0},
        {PipeType::PIPE_MTE3, 0},
        {PipeType::PIPE_FIX, 0}
    };
    std::unordered_map<MemoryType, uint64_t> bufferTotalUsage = {
        {MemoryType::MEM_UB, 0},
        {MemoryType::MEM_L1, 0},
        {MemoryType::MEM_L0A, 0},
        {MemoryType::MEM_L0B, 0},
        {MemoryType::MEM_L0C, 0}
    };
    std::unordered_map<MemoryType, uint64_t> bufferMaxUsage = {
        {MemoryType::MEM_UB, 0},
        {MemoryType::MEM_L1, 0},
        {MemoryType::MEM_L0A, 0},
        {MemoryType::MEM_L0B, 0},
        {MemoryType::MEM_L0C, 0}
    };
    std::unordered_map<MemoryType, uint64_t> bufferLastUsage = {
        {MemoryType::MEM_UB, 0},
        {MemoryType::MEM_L1, 0},
        {MemoryType::MEM_L0A, 0},
        {MemoryType::MEM_L0B, 0},
        {MemoryType::MEM_L0C, 0}
    };
    std::unordered_map<MemoryType, uint64_t> lastClock = {
        {MemoryType::MEM_UB, 0},
        {MemoryType::MEM_L1, 0},
        {MemoryType::MEM_L0A, 0},
        {MemoryType::MEM_L0B, 0},
        {MemoryType::MEM_L0C, 0}
    };
    struct SpillInfo {
        MemoryType spillType{MemoryType::MEM_UNKNOWN}; //spill buffer类型
        int bufferCurrUsage{0}; // 当前buffer的总使用量
        int spillTensorSize{0}; // spill的tensor的大小
        int triggerTensorSize{0}; // 触发当前spill的tensor的大小
        int allocOccupiedSize{0}; // 当前被alloc占用的buffer大小
        int spillCopyoutSize{0}; // spill到ddr的数据量
        int spillTensorMagic; // spill tensor的magic
    };
    std::vector<SpillInfo> spillInfoVec; // size为spill的次数
    Json report;
};

} // namespace tile_fwk
} // namespace npu
#endif // OOO_SCHEDULE_STATISTIC_H