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
 * \file shmem_wait_until.h
 * \brief
 */

#ifndef SHMEM_WAIT_UNTIL_H
#define SHMEM_WAIT_UNTIL_H

#include <vector>

#include "common.h"
#include "machine/utils/dynamic/dev_workspace.h"
#include "machine/utils/dynamic/device_task.h"

namespace npu::tile_fwk::Distributed {
class SignalTileOp {
public:
    void Init(uint64_t taskId, int32_t* addr, uint32_t endOffset, uint32_t stride, int32_t expectedSum);
    bool PollCompleted(std::vector<uint64_t> &completed);

private:
    uint64_t taskId_;
    int32_t* addr_;
    uint32_t endOffset_;
    uint32_t stride_;
    int32_t expectedSum_;
};

class ShmemWaitUntil {
public:
    void Init(npu::tile_fwk::dynamic::DynDeviceTask *dynDeviceTask);
    void EnqueueOp(uint64_t taskId, const npu::tile_fwk::dynamic::DevRelocVector<int32_t> &aicpuCode);
    void PollCompleted(std::vector<uint64_t> &completed);

private:
    std::vector<SignalTileOp> signalTileOp_{VECTOR_PRE_SIZE};
    std::vector<bool> done_ = std::vector<bool>(VECTOR_PRE_SIZE, false);
    uint32_t signalTileOpCount_{0};

    npu::tile_fwk::dynamic::DynDeviceTask *dynDeviceTask_;
    npu::tile_fwk::DynFuncData *funcDataList_;
    uint64_t *hcclContextAddr_;
    AicpuParamInfo paramInfo_;

    uint64_t GetRawAddr(const uint64_t addr, const uint64_t dstRankId);
    TensorInfo GetTensorInfo(uint64_t taskId, const npu::tile_fwk::dynamic::DevRelocVector<int32_t> &aicpuCode);
};

} // namespace npu::tile_fwk::Distributed
#endif // SHMEM_WAIT_UNTIL_H
