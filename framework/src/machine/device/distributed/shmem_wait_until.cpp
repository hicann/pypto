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

#include "machine/device/dynamic/aicore_manager.h"
#include "tileop/distributed/hccl_context.h"
#include "machine/utils/device_log.h"
#include "machine/utils/dynamic/dev_workspace.h"
#include "neon_stub.h"
#include "machine/device/dynamic/device_utils.h"

namespace npu::tile_fwk::Distributed {

inline bool SignalTileOp::PollCompleted() const
{
    if constexpr (npu::tile_fwk::dynamic::IsDeviceMode()) {
        if (addr_[0] == expectedSum_) {
            if (resetSignal_) {
                addr_[0] = 0;
            }
            return true;
        }
        return false;
    }
    return true;
}

int32_t ShmemWaitUntil::PollCompleted(npu::tile_fwk::dynamic::AiCoreManager *aicoreManager)
{
    return runingTaskQueue_.PollCompleted([&](SignalTileOp* task) {
        if (aicoreManager == nullptr) {
            DEV_ERROR("AicoreManager is nullptr");
            return dynamic::DEVICE_MACHINE_ERROR;
        }
        return aicoreManager->ProcessCompletedAicpuTask(task->taskId_);
    });
}

uint64_t ShmemWaitUntil::GetRawAddr(const uint64_t addr, const uint64_t dstRankId)
{
    uint64_t groupIndex = npu::tile_fwk::Distributed::GetVirtualAddrGroupIndex(addr);
    uint64_t offset = npu::tile_fwk::Distributed::GetVirtualAddrOffset(addr);
    uint64_t memType = npu::tile_fwk::Distributed::GetVirtualAddrMemType(addr);
    auto hcclOpParam = reinterpret_cast<TileOp::HcclCombinOpParam*>(hcclContextAddr_[groupIndex]);
    if (memType == 0) {
        return hcclOpParam->windowsIn[dstRankId] + offset;
    } else {
        return hcclOpParam->windowsExp[dstRankId] + offset;
    }
}

TensorInfo ShmemWaitUntil::GetTensorInfo(uint64_t taskId, const npu::tile_fwk::dynamic::DevRelocVector<int32_t> &aicpuCode)
{
    uint32_t funcId = npu::tile_fwk::FuncID(taskId);
    uint32_t opIndex = npu::tile_fwk::TaskID(taskId);
    auto &funcData = funcDataList_[funcId];
    auto opAttrs = &funcData.opAttrs[funcData.opAtrrOffsets[opIndex]];
    auto expressionTable = funcData.exprTbl;

    int32_t index = aicpuCode[paramInfo_.inIndex + 3]; // ShmemWaitUntil注册registerInfo中ShmemTensor位于第2个输入位，因此dim、offset位于2和3号位
    TensorInfo info;
    info.rawIndex = GetCoa(index, opAttrs, expressionTable);
    ++index; // 跳过 rawIndex
    info.dim = aicpuCode[paramInfo_.inIndex + 2];
    info.offset = GetCoaVector(index, info.dim, opAttrs, expressionTable);
    const uint32_t dstRankId = info.offset[0];

    info.expectedSum = aicpuCode[paramInfo_.attrIndex];
    info.resetSignal = aicpuCode[paramInfo_.attrIndex + 2];
    auto desc = &funcData.rawTensorDesc[info.rawIndex];
    info.rawAddr = ShmemWaitUntil::GetRawAddr(funcData.rawTensorAddr[desc->offsetOrIndex], dstRankId);
    return info;
}
} // namespace npu::tile_fwk::Distributed
