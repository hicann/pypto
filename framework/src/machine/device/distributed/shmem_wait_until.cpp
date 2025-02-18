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
 * \file shmem_wait_until.cpp
 * \brief
 */

#include "shmem_wait_until.h"

#include <cstddef>
#include <type_traits>
#include <vector>
#include <cstring>
#include <cstdint>

#include "securec.h"

#include "machine/utils/dynamic/dev_encode.h"
#include "tileop/distributed/hccl_context.h"
#include "machine/utils/device_log.h"
#include "machine/utils/dynamic/dev_workspace.h"
#include "neon_stub.h"

namespace npu::tile_fwk::Distributed {
void SignalTileOp::Init(uint64_t taskId, int32_t* addr, uint32_t endOffset, uint32_t stride, int32_t expectedSum)
{
    taskId_ = taskId;
    addr_ = addr;
    endOffset_ = endOffset * stride;
    stride_ = stride;
    expectedSum_ = expectedSum;
}

bool SignalTileOp::PollCompleted(std::vector<uint64_t> &completed)
{
    int32_t sum = 0;
    for (uint32_t offset = 0; offset < endOffset_; offset += stride_) {
        sum += addr_[offset];
    }
    if (sum == expectedSum_) {
        completed.emplace_back(taskId_);
        return true;
    }
    return false;
}

void ShmemWaitUntil::Init(npu::tile_fwk::dynamic::DynDeviceTask *dynDeviceTask)
{
    dynDeviceTask_ = dynDeviceTask;
    funcDataList_ = reinterpret_cast<DynFuncData*>(&dynDeviceTask->GetDynFuncDataList()->At(0));
    hcclContextAddr_ = funcDataList_->hcclContext;
}

void ShmemWaitUntil::EnqueueOp(uint64_t taskId, const npu::tile_fwk::dynamic::DevRelocVector<int32_t> &aicpuCode)
{
    paramInfo_ = DecodeAicpuCode(aicpuCode);
    TensorInfo info = ShmemWaitUntil::GetTensorInfo(taskId, aicpuCode);
    const uint32_t offset1 = info.offset[1]; // offset 1
    const uint32_t offset2 = info.offset[2]; // offset 2
    const uint32_t offset3 = info.offset[3]; // offset 3
    const uint32_t shape2 = info.shape[2]; // shape 2
    const uint32_t shape3 = info.shape[3]; // shape 3
    const uint32_t rawShape2 = info.rawShape[2]; // raw shape 2
    const uint32_t rawShape3 = info.rawShape[3]; // raw shape 3
    const int32_t expectedSum = info.expectedSum;
    DEV_DEBUG("ShmemWaitUntil::EnqueueOp offset1=%u, offset2=%u, offset3=%u, shape2=%u, shape3=%u, rawShape2=%u, rawShape3=%u", offset1, offset2, offset3, shape2, shape3, rawShape2, rawShape3);

    int32_t* addr = reinterpret_cast<int32_t*>(info.rawAddr) + offset1 * rawShape2 * rawShape3 + offset2 * rawShape3 + offset3;

    if (signalTileOpCount_ == signalTileOp_.size()) {
        signalTileOp_.resize(signalTileOpCount_ * 2); // 扩容到原本的 2 倍
        done_.resize(signalTileOpCount_ * 2); // 扩容到原本的 2 倍
    }
    signalTileOp_[signalTileOpCount_].Init(taskId, addr, shape2, shape3, expectedSum);
    ++signalTileOpCount_;
}

void ShmemWaitUntil::PollCompleted(std::vector<uint64_t> &completed)
{
    for (uint32_t i = 0; i < signalTileOpCount_; ++i) {
        if (done_[i]) {
            continue;
        }
        if (signalTileOp_[i].PollCompleted(completed)) {
            done_[i] = true;
        }
    }
}

uint64_t ShmemWaitUntil::GetRawAddr(const uint64_t addr, const uint64_t dstRankId)
{
    uint64_t groupIndex = npu::tile_fwk::Distributed::GetVirtualAddrGroupIndex(addr);
    uint64_t offset = npu::tile_fwk::Distributed::GetVirtualAddrOffset(addr);
    auto hcclOpParam = reinterpret_cast<TileOp::HcclCombinOpParam*>(hcclContextAddr_[groupIndex]);
    return hcclOpParam->windowsIn[dstRankId] + offset;
}

TensorInfo ShmemWaitUntil::GetTensorInfo(uint64_t taskId, const npu::tile_fwk::dynamic::DevRelocVector<int32_t> &aicpuCode)
{
    uint32_t funcId = npu::tile_fwk::dynamic::FuncID(taskId);
    uint32_t opIndex = npu::tile_fwk::dynamic::TaskID(taskId);
    auto &funcData = funcDataList_[funcId];
    auto opAttrs = &funcData.opAttrs[funcData.opAtrrOffsets[opIndex]];
    auto expressionTable = funcData.exprTbl;

    int32_t index = aicpuCode[paramInfo_.inIndex + 3]; // ShmemWaitUntil注册registerInfo中ShmemTensor位于第2个输入位，因此dim、offset位于2和3号位
    TensorInfo info;
    info.rawIndex = GetCoa(index, opAttrs, expressionTable);
    ++index; // 跳过 rawIndex
    info.dim = aicpuCode[paramInfo_.inIndex + 2];
    info.offset = GetCoaVector(index, info.dim, opAttrs, expressionTable);
    index += info.dim;
    info.shape = GetCoaVector(index, info.dim, opAttrs, expressionTable);
    index += info.dim;
    info.rawShape = GetCoaVector(index, info.dim, opAttrs, expressionTable);
    index += info.dim;
    info.dynValidShape = GetCoaVector(index, info.dim, opAttrs, expressionTable);
    const uint32_t dstRankId = info.offset[0];

    info.expectedSum = aicpuCode[paramInfo_.attrIndex + 1];
    auto desc = &funcData.rawTensorDesc[info.rawIndex];
    info.rawAddr = ShmemWaitUntil::GetRawAddr(funcData.rawTensorAddr[desc->offsetOrIndex], dstRankId);
    return info;
}
} // namespace npu::tile_fwk::Distributed
