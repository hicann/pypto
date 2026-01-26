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
#include "machine/device/dynamic/aicore_manager.h"

namespace {

auto InitializeTensorInfo(const uint32_t rankSize) {
    npu::tile_fwk::Distributed::TensorInfo info;
    info.offset = {0, 1, 0, 0};
    info.dim = 3;
    info.expectedSum = 0;
    info.resetSignal = false;
    info.rawIndex = 0;
    const uint32_t rawShape0 = rankSize;
    const uint32_t rawShape1 = rankSize;
    const uint32_t rawShape2 = 4;
    const uint32_t rawShape3 = 8;
    std::vector<int32_t> rawAddr(rawShape1 * rawShape2 * rawShape3, 0);
    info.rawAddr = reinterpret_cast<uint64_t>(rawAddr.data());
    int32_t* addr = rawAddr.data() + info.offset[1] * rawShape2 * rawShape3 + info.offset[2] * rawShape3 + info.offset[3];
    addr[0] = 1;
    return std::make_tuple(info, std::move(rawAddr), rawShape0, rawShape1, rawShape2, rawShape3);
}

auto InitializeAicpuCode(const uint32_t rankSize) {
    constexpr size_t codeSize = 17;
    auto data = std::make_unique<int32_t[]>(codeSize);
    uint32_t initData[codeSize] = {153, 2, 2, 44, 4, 2, 18, 4, 0, 2, 1, 0, 4, rankSize, rankSize, 4, 8};
    std::copy(initData, initData + codeSize, data.get());
    npu::tile_fwk::dynamic::DevRelocVector<int32_t> aicpuCode(codeSize, data.get());
    return std::make_tuple(std::move(data), std::move(aicpuCode));
}

auto InitializeTaskData(npu::tile_fwk::dynamic::DynDeviceTask* task) {
    size_t headerSize = sizeof(npu::tile_fwk::DynFuncHeader);
    size_t dataSize = sizeof(npu::tile_fwk::DynFuncData);
    std::unique_ptr<void, decltype(&free)> buffer(malloc(headerSize + dataSize), free);
    auto* header = new(buffer.get())npu::tile_fwk::DynFuncHeader();
    auto* funcData = new(header + 1)npu::tile_fwk::DynFuncData();

    task->dynFuncDataList = header;
    task->dynFuncDataList[0].seqNo = 1;
    task->dynFuncDataList[0].funcNum = 1;
    task->dynFuncDataList[0].funcSize = 1u;
    task->dynFuncDataList[0].cceBinary = nullptr;

    return std::make_tuple(std::move(buffer), funcData);
}

auto ConfigureFuncData(npu::tile_fwk::DynFuncData* funcData, uint64_t rawAddr,
                      const uint32_t rawShape0, const uint32_t rawShape1,
                      const uint32_t rawShape2, const uint32_t rawShape3) {
    constexpr size_t exprTblSize = 50;
    auto exprTbl = std::make_unique<uint64_t[]>(exprTblSize);
    funcData->exprTbl = exprTbl.get();

    auto hcclParam = std::make_unique<TileOp::HcclCombinOpParam>();
    hcclParam->rankNum = 0;
    hcclParam->windowsIn[0] = rawAddr;
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

    return std::make_tuple(std::move(exprTbl), std::move(hcclParam),
                          std::move(rawTensorAddrHolder), std::move(rawTensorDescHolder),
                          std::move(opAttrs));
}

auto InitializeTestEnvironment(const uint32_t rankSize) {
    auto [info, rawAddr, rawShape0, rawShape1, rawShape2, rawShape3] = InitializeTensorInfo(rankSize);
    auto [data, aicpuCode] = InitializeAicpuCode(rankSize);
    auto allocator = std::make_unique<npu::tile_fwk::dynamic::DeviceWorkspaceAllocator>();
    auto task = std::make_unique<npu::tile_fwk::dynamic::DynDeviceTask>(*allocator);
    auto shmemWaitUntil = std::make_unique<npu::tile_fwk::Distributed::ShmemWaitUntil>();
    auto aicpuTaskManager = std::make_unique<npu::tile_fwk::dynamic::AicpuTaskManager>();
    auto aicoreManager = std::make_unique<npu::tile_fwk::dynamic::AiCoreManager>(*aicpuTaskManager);

    auto [buffer, funcData] = InitializeTaskData(task.get());

    auto [exprTbl, hcclParam, rawTensorAddrHolder, rawTensorDescHolder, opAttrs] =
        ConfigureFuncData(funcData, info.rawAddr, rawShape0, rawShape1, rawShape2, rawShape3);
    shmemWaitUntil->Init(task.get());
    return std::make_tuple(std::move(rawAddr), std::move(data), std::move(allocator),
                          std::move(task), std::move(shmemWaitUntil), std::move(buffer),
                          std::move(exprTbl), std::move(hcclParam), std::move(rawTensorAddrHolder),
                          std::move(rawTensorDescHolder), std::move(opAttrs), std::move(aicpuCode),
                          funcData, std::move(aicoreManager));
}

void PrepareTasks(uint32_t tileOpCount, npu::tile_fwk::Distributed::ShmemWaitUntil* shmemWaitUntil,
    const npu::tile_fwk::dynamic::DevRelocVector<int32_t>& aicpuCode, npu::tile_fwk::DynFuncData* funcData,
    uint64_t* opAttrsPtr) {
    constexpr size_t opAttrsLength = 17;
    std::vector<std::unique_ptr<int32_t[]>> opAtrrOffsetsHolder;
    std::vector<std::unique_ptr<uint64_t[]>> opAttrsCopyHolder;
    for (uint32_t taskId = 0; taskId < tileOpCount; ++taskId) {
        auto opAtrrOffsets = std::make_unique<int32_t[]>(taskId + 1);
        opAtrrOffsets[taskId] = 0;

        int opAttrsSize = 1 + opAtrrOffsets[taskId] + opAttrsLength;
        auto opAttrsCopy = std::make_unique<uint64_t[]>(opAttrsSize);
        std::copy(opAttrsPtr, opAttrsPtr + opAttrsLength, opAttrsCopy.get() + opAtrrOffsets[taskId]);

        opAtrrOffsetsHolder.push_back(std::move(opAtrrOffsets));
        opAttrsCopyHolder.push_back(std::move(opAttrsCopy));

        funcData->opAtrrOffsets = opAtrrOffsetsHolder.back().get();
        funcData->opAttrs = opAttrsCopyHolder.back().get();

        shmemWaitUntil->PrepareTask(taskId, aicpuCode);
    }
}

void RunTests(uint32_t tileOpCount, npu::tile_fwk::Distributed::ShmemWaitUntil* shmemWaitUntil,
    npu::tile_fwk::dynamic::AiCoreManager* aicoreManager) {
    TaskStat* taskStat{nullptr};
    for (uint32_t taskId = 0; taskId < tileOpCount; ++taskId) {
        shmemWaitUntil->EnqueueOp(taskId, taskStat);
        shmemWaitUntil->PollCompleted(*aicoreManager);
    }
}

void TestShmemWaitUntil(const uint32_t tileOpCount) {
    const uint32_t rankSize = 4;
    auto [rawAddr, data, allocator, task, shmemWaitUntil, buffer, exprTbl, hcclParam,
          rawTensorAddrHolder, rawTensorDescHolder, opAttrs, aicpuCode, funcData, aicoreManager] = InitializeTestEnvironment(rankSize);

    PrepareTasks(tileOpCount, shmemWaitUntil.get(), aicpuCode, funcData, opAttrs.get());

    RunTests(tileOpCount, shmemWaitUntil.get(), aicoreManager.get());
}

TEST(ShmemWaitUntilTest, BasicFunctionality) {
    constexpr int32_t tileOpCount = 1;
    TestShmemWaitUntil(tileOpCount);
}
} // namespace