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
 * \file comm_wait_flag.h
 * \brief
 */

#ifndef COMM_WAIT_FLAG_H
#define COMM_WAIT_FLAG_H

#include "interface/utils/common.h"
#include "tilefwk/core_func_data.h"
#include <cstdint>
#include <vector>
#include <deque>
#include <map>

namespace npu::tile_fwk {
namespace Distributed {
// 以下 ATOMIC_ADD_BLOCK_BYTE_SIZE 和 FLAG_BYTE_SIZE 的定义与 distributed.h 中的定义一致
// AtomicAdd 每次操作 32B 的数据，对同一 32B 的数据进行 AtomicAdd 需要排队
constexpr uint32_t ATOMIC_ADD_BLOCK_BYTE_SIZE = 32;
// 为了消除 AtomicAdd 并发，以 32B 为最小单位，视情况调节每个 flag 占用的字节数
constexpr uint32_t FLAG_BYTE_SIZE = ATOMIC_ADD_BLOCK_BYTE_SIZE * 4;
// 预先设置大小
constexpr uint32_t DONE_FLAG_PRE_SIZE = 1024;

class FlagPoller {
public:
    FlagPoller() : doneFlag_(DONE_FLAG_PRE_SIZE, 0), opInfo_(DONE_FLAG_PRE_SIZE, OpInfo{0, 0}){};
    ~FlagPoller() {};
    void Init(uint32_t rankId, uint32_t rankSize, uint8_t *winFlag);
    void EnqueueOp(uint64_t taskId, uint32_t rankShape, uint32_t rankOffset, uint32_t tileIndex);
    void PollCompleted(std::vector<uint64_t> &completed);

private:
    struct OpInfo {
        uint64_t taskId;
        uint32_t todoFlagCount;
    };
    uint32_t rankId_;
    uint32_t rankSize_;
    uint32_t rankShape_;
    uint8_t *winFlag_{nullptr};
    std::vector<bool> doneFlag_;
    std::vector<OpInfo> opInfo_;
    size_t opCount_{0};
    uint32_t realCount_{0};
};

class CommWaitFlag {
public:
    CommWaitFlag() {};
    ~CommWaitFlag() {};
    void Init(DeviceTask *deviceTask);
    void EnqueueOp(uint64_t taskId, uint64_t *paramList, uint32_t paramSize);
    void PollCompleted(std::vector<uint64_t> &completed);

private:
    bool Prepare(uint32_t groupIndex);
    uint64_t *hcclContextAddr_{nullptr};
    uint32_t commGroupNum_{0};
    FlagPoller flagPoller_[DIST_COMM_GROUP_NUM];
    bool inited_[DIST_COMM_GROUP_NUM]{false};
};

} // namespace Distributed
} // namespace npu::tile_fwk
#endif // COMM_WAIT_FLAG_H
