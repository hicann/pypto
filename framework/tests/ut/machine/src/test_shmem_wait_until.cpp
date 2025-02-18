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
 * \file test_shmem_wait_until.cpp
 * \brief
 */

#include <gtest/gtest.h>

#include "machine/device/distributed/common.h"
#include "machine/device/distributed/shmem_wait_until.h"
#include "machine/utils/dynamic/dev_workspace.h"
#include "tileop/distributed/hccl_context.h"

namespace {
void TestShmemWaitUntil(const uint32_t tileOpCount) {
    npu::tile_fwk::Distributed::TensorInfo info;
    info.offset = {0, 1, 0, 0};
    info.shape = {1, 1, 1, 8};
    constexpr uint32_t rankSize = 4;
    constexpr uint32_t rawShape0 = rankSize;
    constexpr uint32_t rawShape1 = rankSize;
    constexpr uint32_t rawShape2 = 4;
    constexpr uint32_t rawShape3 = 8;
    info.rawShape = {rawShape0, rawShape1, rawShape2, rawShape3};
    int32_t rawAddr[rawShape1 * rawShape2 * rawShape3] = {0};
    info.rawAddr = reinterpret_cast<uint64_t>(rawAddr);
    constexpr int32_t value = 1;
    int32_t* addr = rawAddr + info.offset[1] * info.rawShape[2] * info.rawShape[3] + info.offset[2] * info.rawShape[3] + info.offset[3];
    for (uint32_t offset = 0; offset < info.shape[2] * info.shape[3]; offset += info.shape[3]) {
        addr[offset] = value;
    }
    constexpr size_t codeSize = 12;
    auto data = std::make_unique<int32_t[]>(codeSize);
    int32_t initData[codeSize] = {153, 2, 2, 44, 4, 2, 18, 4, 0, 2, 0, 1};
    std::copy(initData, initData + codeSize, data.get());
    npu::tile_fwk::dynamic::DevRelocVector<int32_t> aicpuCode(codeSize, data.get());

    npu::tile_fwk::Distributed::ShmemWaitUntil shmemWaitUntil;
    npu::tile_fwk::dynamic::DeviceWorkspaceAllocator allocator;
    npu::tile_fwk::dynamic::DynDeviceTask task(allocator);

    size_t headerSize = sizeof(npu::tile_fwk::DynFuncHeader);
    size_t dataSize = sizeof(npu::tile_fwk::DynFuncData);
    std::unique_ptr<void, decltype(&free)> buffer(malloc(headerSize + dataSize), free);

    auto* header = new(buffer.get())npu::tile_fwk::DynFuncHeader();
    auto* funcData = new(header + 1)npu::tile_fwk::DynFuncData();
    task.dynFuncDataList = header;
    task.dynFuncDataList[0].seqNo = 1;
    task.dynFuncDataList[0].funcNum = 1;
    task.dynFuncDataList[0].funcSize = 1u;
    task.dynFuncDataList[0].cceBinary = nullptr;

    constexpr size_t exprTblSize = 50;
    auto exprTbl = std::make_unique<uint64_t[]>(exprTblSize);
    funcData->exprTbl = exprTbl.get();

    auto hcclParam = std::make_unique<TileOp::HcclCombinOpParam>();
    hcclParam->rankNum = 0;
    hcclParam->windowsIn[0] = reinterpret_cast<uint64_t>(rawAddr);
    funcData->hcclContext[0] = reinterpret_cast<uint64_t>(hcclParam.get());

    auto rawTensorAddrHolder = std::make_unique<uint64_t[]>(1);
    auto rawTensorDescHolder = std::make_unique<npu::tile_fwk::DevRawTensorDesc[]>(1);
    rawTensorAddrHolder[0] = 0;
    rawTensorDescHolder[0] = {0, 0};
    funcData->rawTensorAddr = rawTensorAddrHolder.get();
    funcData->rawTensorDesc = rawTensorDescHolder.get();

    constexpr size_t opAttrsLength = 17;
    auto opAttrs = std::make_unique<uint64_t[]>(opAttrsLength);
    uint64_t initAttrs[opAttrsLength] = {0, 0, 1, 0, 0, 1, 1, 1, 8, rawShape0, rawShape1, rawShape2, rawShape3, 0, 0, 0, 0};
    std::copy(initAttrs, initAttrs + opAttrsLength, opAttrs.get());
    shmemWaitUntil.Init(&task);

    for (uint32_t taskId = 0; taskId < tileOpCount; ++taskId) {
        auto opAtrrOffsets = std::make_unique<int32_t[]>(taskId + 1);
        opAtrrOffsets[taskId] = 0;

        int opAttrsSize = 1 + opAtrrOffsets[taskId] + opAttrsLength;
        auto opAttrsCopy = std::make_unique<uint64_t[]>(opAttrsSize);
        std::copy(opAttrs.get(), opAttrs.get() + opAttrsLength, opAttrsCopy.get() + opAtrrOffsets[taskId]);

        funcData->opAtrrOffsets = opAtrrOffsets.get();
        funcData->opAttrs = opAttrsCopy.get();

        shmemWaitUntil.EnqueueOp(taskId, aicpuCode);

        std::vector<uint64_t> completed;
        shmemWaitUntil.PollCompleted(completed);
        ASSERT_EQ(completed.size(), 1);
        ASSERT_EQ(completed[0], taskId);
    }
}

TEST(ShmemWaitUntilTest, BasicFunctionality) {
    constexpr int32_t tileOpCount = 1;
    TestShmemWaitUntil(tileOpCount);
}

TEST(ShmemWaitUntilTest, VectorResize) {
    constexpr int32_t tileOpCount = npu::tile_fwk::Distributed::VECTOR_PRE_SIZE + 1;
    TestShmemWaitUntil(tileOpCount);
}

} // namespace