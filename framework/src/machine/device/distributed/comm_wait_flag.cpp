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
 * \file comm_wait_flag.cpp
 * \brief
 */

#include "comm_wait_flag.h"

#include <cstddef>
#include <type_traits>
#include <vector>
#include <cstring>
#include <cstdint>

#include "securec.h"

#include "tileop/distributed/hccl_context.h"
#include "machine/utils/device_log.h"
#include "tilefwk/core_func_data.h"
#include "neon_stub.h"


namespace npu::tile_fwk {
namespace Distributed {

constexpr uint32_t NEON_BLOCK_NUM = 16;
constexpr uint32_t NEON_BLOCK_BITS = 8;
constexpr uint32_t LONG_LONG_BITS = 64;
constexpr uint32_t TILE_INDEX_MAX_NUM = 65535;

void FlagPoller::Init(uint32_t rankId, uint32_t rankSize, uint8_t *winFlag)
{
    rankId_ = rankId;
    rankSize_ = rankSize;
    winFlag_ = winFlag;
}

void FlagPoller::EnqueueOp(uint64_t taskId, uint32_t rankShape, uint32_t rankOffset, uint32_t tileIndex)
{
    if ((rankShape < 1) || (tileIndex > TILE_INDEX_MAX_NUM) || (rankOffset + rankShape > TileOp::AICPU_MAX_RANK_NUM)) {
        DEV_ERROR("FlagPoller EnqueueOp failed: tileIndex=%u, rankShape=%u, rankOffset=%u\n", tileIndex, rankShape,
            rankOffset);
        return;
    }

    const uint32_t baseIndex = tileIndex * rankSize_;
    const uint32_t startIndex = baseIndex + rankOffset;
    const uint32_t endIndex = startIndex + rankShape - 1U;
    const uint32_t localIndex = baseIndex + rankId_;
    if (realCount_ <= endIndex) {
        realCount_ = static_cast<uint32_t>(endIndex + 1U);
    }
    if (doneFlag_.size() <= endIndex) {
        doneFlag_.resize(endIndex + 1, false);
    }

    uint32_t opIndex = startIndex / rankShape;
    if (opInfo_.size() <= opIndex) {
        opInfo_.resize(opIndex + 1, OpInfo{0, 0});
    }
    opInfo_[opIndex].taskId = taskId;
    opInfo_[opIndex].todoFlagCount = rankShape;
    rankShape_ = rankShape;
    opCount_++;

    if ((localIndex >= startIndex) && (localIndex <= endIndex)) {
        doneFlag_[localIndex] = true;
        opInfo_[localIndex / rankShape_].todoFlagCount--;
    }
}

void FlagPoller::PollCompleted(std::vector<uint64_t> &completed)
{
    if (opCount_ == 0U) {
        return;
    }
    completed.reserve(realCount_);
    for (size_t flagIndex = 0; flagIndex < realCount_; ++flagIndex) {
        DEV_DEBUG("FlagPoller PollCompleted: flagIndex=%zu, winFlag_ addr=%p, winFlag_=%u.\n",
            flagIndex, winFlag_ + flagIndex * FLAG_BYTE_SIZE, winFlag_[flagIndex * FLAG_BYTE_SIZE]);
        if (
            (!doneFlag_[flagIndex])
            && (winFlag_[flagIndex * FLAG_BYTE_SIZE] == 1)
            && (opInfo_[flagIndex / rankShape_].todoFlagCount != 0) // 等于 0 意味着，可能这个 flag 对应的子图还未执行添加进来，等添加进来后再处理
        ) {
            uint32_t opIndex = static_cast<uint32_t>(flagIndex / rankShape_);
            opInfo_[opIndex].todoFlagCount--;
            if (opInfo_[opIndex].todoFlagCount == 0) {
                completed.emplace_back(opInfo_[opIndex].taskId);
                opCount_--;
            }
            doneFlag_[flagIndex] = true;
        }
    }
}

void CommWaitFlag::Init(DeviceTask *deviceTask)
{
    uint64_t *hcclContextAddr = deviceTask->coreFuncData.hcclContextAddr;
    uint32_t commGroupNum = static_cast<uint32_t>(deviceTask->coreFuncData.commGroupNum);
    if (commGroupNum > DIST_COMM_GROUP_NUM) {
        commGroupNum = 0;
        DEV_ERROR("CommWaitFlag comm group invalid, num=%u\n", commGroupNum);
        return;
    }
    hcclContextAddr_ = hcclContextAddr;
    commGroupNum_ = commGroupNum;
}

bool CommWaitFlag::Prepare(uint32_t groupIndex)
{
    if (inited_[groupIndex]) {
        return true;
    }
    TileOp::HcclCombinOpParam *hcclOpParam = reinterpret_cast<TileOp::HcclCombinOpParam *>(hcclContextAddr_[groupIndex]);
    uint32_t rankId = hcclOpParam->rankId;
    uint32_t rankSize = hcclOpParam->rankNum;
    uint8_t *winFlag = reinterpret_cast<uint8_t *>(hcclOpParam->windowsExp[rankId]);
    if ((rankSize <= 1) || (rankSize > TileOp::AICPU_MAX_RANK_NUM) || (rankId >= rankSize)) {
        DEV_ERROR("CommWaitFlag Prepare failed: groupIndex=%u, rankSize=%u, rankId=%u\n", groupIndex, rankSize, rankId);
        return false;
    }
    flagPoller_[groupIndex].Init(rankId, rankSize, winFlag);
    inited_[groupIndex] = true;
    return true;
}

void CommWaitFlag::EnqueueOp(uint64_t taskId, uint64_t *paramList, uint32_t paramSize)
{
    if (paramSize != 0x4) {
        DEV_ERROR("CommWaitFlag EnqueueOp param size inlvaid: %u\n", paramSize);
        return;
    }
    uint32_t tileIndex = paramList[0x0];
    uint32_t groupIndex = paramList[0x1];
    uint32_t rankShape = paramList[0x2];
    uint32_t rankOffset = paramList[0x3];
    if ((groupIndex >= commGroupNum_) || (hcclContextAddr_[groupIndex] == 0)) {
        DEV_ERROR("CommWaitFlag EnqueueOp param inlvaid: groupIndex=%u\n", groupIndex);
        return;
    }
    if (!Prepare(groupIndex)) {
        return;
    }

    flagPoller_[groupIndex].EnqueueOp(taskId, rankShape, rankOffset, tileIndex);
}

void CommWaitFlag::PollCompleted(std::vector<uint64_t> &completed)
{
    for (uint32_t groupIndex = 0; groupIndex < commGroupNum_; ++groupIndex) {
        if (inited_[groupIndex]) {
            flagPoller_[groupIndex].PollCompleted(completed);
        }
    }
}

} // namespace Distributed
} // namespace npu::tile_fwk
