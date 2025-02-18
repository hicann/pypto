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
 * \file test_comm_wait_flag.cpp
 * \brief
 */

#include <gtest/gtest.h>
#include "machine/device/distributed/comm_wait_flag.h"
#include "tilefwk/core_func_data.h"
#include "tileop/distributed/hccl_context.h"

using namespace npu::tile_fwk;
using namespace npu::tile_fwk::Distributed;

namespace {
TEST(CommWaitFlagTest, CommWaitFlag_EnqueueOp_and_PollCompleted_success) {
    TileOp::HcclCombinOpParam hcclParam;
    hcclParam.rankId = 0;
    constexpr uint32_t rankNum = 4;
    hcclParam.rankNum = rankNum;
    uint8_t winFlag[npu::tile_fwk::Distributed::FLAG_BYTE_SIZE * rankNum] = {0};
    hcclParam.windowsExp[hcclParam.rankId] = reinterpret_cast<uint64_t>(winFlag);

    npu::tile_fwk::CoreFunctionData coreFuncData;
    coreFuncData.hcclContextAddr[hcclParam.rankId] = reinterpret_cast<uint64_t>(&hcclParam);
    coreFuncData.commGroupNum = 1;

    npu::tile_fwk::DeviceTask deviceTask;
    deviceTask.coreFuncData = coreFuncData;

    npu::tile_fwk::Distributed::CommWaitFlag commWaitFlag;
    commWaitFlag.Init(&deviceTask);

    constexpr uint32_t paramSize = 4;
    constexpr uint32_t tileIndex = 0;
    constexpr uint32_t groupIndex = 0;
    constexpr uint32_t rankShape = 2;
    constexpr uint32_t rankOffset = 0;
    uint64_t paramList[paramSize] = {tileIndex, groupIndex, rankShape, rankOffset};
    constexpr uint64_t taskId = 123;
    commWaitFlag.EnqueueOp(taskId, paramList, paramSize);

    for (uint32_t i = 0; i < rankShape; i++) {
        winFlag[i * npu::tile_fwk::Distributed::FLAG_BYTE_SIZE] = 1;
    }
    std::vector<uint64_t> completed;
    commWaitFlag.PollCompleted(completed);
    ASSERT_EQ(completed.size(), 1);
    ASSERT_EQ(completed[0], taskId);
}
} // namespace