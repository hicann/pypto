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
#include "machine/runtime/distributed/hccl_context.h"
#include "machine/device/dynamic/aicore_manager.h"

namespace {

std::vector<int32_t> InitializeShmemSignal(std::vector<uint32_t> shmemSignalRawShape,
    std::vector<uint32_t> shmemSignalOffset, std::vector<uint32_t> tileShape, uint32_t shmemSignalStride,
    int32_t expectedValue)
{
    uint32_t tileRowNum = (shmemSignalRawShape[3] + tileShape[0] - 1) / tileShape[0];
    uint32_t tileColNum = (shmemSignalRawShape[4] + tileShape[1] - 1) / tileShape[1];
    uint32_t tileRowIndex = shmemSignalOffset[3] / tileShape[0];
    uint32_t tileColIndex = shmemSignalOffset[4] / tileShape[1];
    uint32_t totalTileNum = tileRowNum * tileColNum;
    uint32_t tileIndex = tileRowIndex * tileColNum + tileColIndex;
    uint32_t size = std::accumulate(shmemSignalRawShape.begin() + 1, shmemSignalRawShape.end(), 1,
        std::multiplies<uint32_t>());
    std::vector<int32_t> shmemSignal(size, 0);
    uint32_t index = (shmemSignalOffset[1] * shmemSignalRawShape[2] * totalTileNum + shmemSignalOffset[2] *
        totalTileNum + tileIndex) * shmemSignalStride;
    shmemSignal[index] = expectedValue;
    return shmemSignal;
}

auto InitializeAicpuCode(std::vector<uint32_t> shmemSignalRawShape, std::vector<uint32_t> tileShape,
    uint32_t shmemSignalStride, int32_t expectedValue, uint32_t shmemSignalAttrOffset)
{
    std::vector<uint32_t> shmemSignalShape = shmemSignalRawShape;
    constexpr size_t codeSize = 26;
    uint32_t paramSizePerOperand = 2; // 每个 operand 都保存 dim 和 attrOffset，总共 2 个 param
    uint32_t oOperandNum = 1;
    uint32_t iOperandNum = 2;
    uint32_t opcode = -1;
    uint32_t oOperandTotalParamNum = paramSizePerOperand * oOperandNum;
    uint32_t outDim = 2;
    uint32_t outAttrOffset = -1;
    uint32_t iOperandTotalParamNum = paramSizePerOperand * iOperandNum;
    uint32_t predTokenDim = 2;
    uint32_t predTokenAttrOffset = -1;
    uint32_t shmemSignalDim = 5;
    uint32_t shmemSignalShapeNum = shmemSignalDim * 2; // raw shape 存一份，shape 存一份，总共 2 份
    uint32_t attrSize = 5;
    uint32_t resetSignal = 0;
    uint32_t initData[codeSize] = {opcode, oOperandTotalParamNum, outDim, outAttrOffset, iOperandTotalParamNum,
        predTokenDim, predTokenAttrOffset, shmemSignalDim, shmemSignalAttrOffset, shmemSignalShapeNum,
        shmemSignalRawShape[0], shmemSignalRawShape[1], shmemSignalRawShape[2], shmemSignalRawShape[3],
        shmemSignalRawShape[4], shmemSignalShape[0], shmemSignalShape[1], shmemSignalShape[2], shmemSignalShape[3],
        shmemSignalShape[4], attrSize, static_cast<uint32_t>(expectedValue), shmemSignalStride, resetSignal,
        tileShape[0], tileShape[1]};
    auto data = std::make_unique<int32_t[]>(codeSize);
    std::copy(initData, initData + codeSize, data.get());
    npu::tile_fwk::dynamic::DevRelocVector<int32_t> aicpuCode(codeSize, data.get());
    return std::make_tuple(std::move(data), std::move(aicpuCode));
}

auto InitializeTaskData(npu::tile_fwk::dynamic::DynDeviceTask* task) {
    size_t headerSize = sizeof(npu::tile_fwk::DynFuncHeader);
    size_t dataSize = sizeof(npu::tile_fwk::DynFuncData);
    std::unique_ptr<void, decltype(&free)> buffer(malloc(headerSize + dataSize +
        sizeof(npu::tile_fwk::DevStartArgsBase) + sizeof(int64_t)), free);
    auto* header = new(buffer.get())npu::tile_fwk::DynFuncHeader();
    auto* funcData = new(header + 1)npu::tile_fwk::DynFuncData();
    auto* startArgs = new(funcData + 1)npu::tile_fwk::DevStartArgsBase();
    auto* commContext = new(startArgs + 1)int64_t;
    startArgs->commContexts = commContext;
    funcData->startArgs = startArgs;

    task->dynFuncDataList = header;
    task->dynFuncDataList[0].seqNo = 1;
    task->dynFuncDataList[0].funcNum = 1;
    task->dynFuncDataList[0].funcSize = 1u;
    task->dynFuncDataList[0].cceBinary = nullptr;

    return std::make_tuple(std::move(buffer), funcData);
}

auto ConfigureFuncData(npu::tile_fwk::DynFuncData* funcData, uint64_t rawAddr) {
    constexpr size_t exprTblSize = 50;
    auto exprTbl = std::make_unique<uint64_t[]>(exprTblSize);
    funcData->exprTbl = exprTbl.get();

    auto hcclParam = std::make_unique<npu::tile_fwk::HcclCombinOpParam>();
    hcclParam->rankNum = 0;
    hcclParam->windowsIn[0] = rawAddr;

    auto rawTensorAddrHolder = std::make_unique<uint64_t[]>(1);
    auto rawTensorDescHolder = std::make_unique<npu::tile_fwk::DevRawTensorDesc[]>(1);
    rawTensorAddrHolder[0] = 0;
    rawTensorDescHolder[0] = {0, 0};
    funcData->rawTensorAddr = rawTensorAddrHolder.get();
    funcData->rawTensorDesc = rawTensorDescHolder.get();
    funcData->startArgs->commContexts[0] = reinterpret_cast<int64_t>(hcclParam.get());
    funcData->startArgs->commGroupNum = 1;
    constexpr size_t opAttrsLength = 17;
    auto opAttrs = std::make_unique<uint64_t[]>(opAttrsLength);
    std::fill_n(opAttrs.get(), opAttrsLength, 0);

    return std::make_tuple(std::move(exprTbl), std::move(hcclParam),
                          std::move(rawTensorAddrHolder), std::move(rawTensorDescHolder),
                          std::move(opAttrs));
}

auto InitializeTestEnvironment() {
    uint32_t worldSize = 4;
    uint32_t shmemSignalRawShape2 = 1;
    uint32_t shmemSignalRawShape3 = 64;
    uint32_t shmemSignalRawShape4 = 5120;
    std::vector<uint32_t> shmemSignalRawShape{worldSize, worldSize, shmemSignalRawShape2, shmemSignalRawShape3,
        shmemSignalRawShape4};
    std::vector<uint32_t> shmemSignalOffset(shmemSignalRawShape.size());
    std::vector<uint32_t> tileShape{1, shmemSignalRawShape4};
    uint32_t shmemSignalStride = 8;
    int32_t expectedValue = 8;
    std::vector<int32_t> rawAddr = InitializeShmemSignal(shmemSignalRawShape, shmemSignalOffset, tileShape,
        shmemSignalStride, expectedValue);

    uint32_t shmemSignalAttrOffset = 0;
    auto [data, aicpuCode] = InitializeAicpuCode(shmemSignalRawShape, tileShape, shmemSignalStride, expectedValue,
        shmemSignalAttrOffset);

    auto allocator = std::make_unique<npu::tile_fwk::dynamic::DeviceWorkspaceAllocator>();
    auto task = std::make_unique<npu::tile_fwk::dynamic::DynDeviceTask>(*allocator);
    auto shmemWaitUntil = std::make_unique<npu::tile_fwk::Distributed::ShmemWaitUntil>();

    auto [buffer, funcData] = InitializeTaskData(task.get());

    auto [exprTbl, hcclParam, rawTensorAddrHolder, rawTensorDescHolder, opAttrs] =
        ConfigureFuncData(funcData, reinterpret_cast<uint64_t>(rawAddr.data()));
    shmemWaitUntil->Init(task.get());
    return std::make_tuple(std::move(rawAddr), std::move(data), std::move(allocator), std::move(task),
                          std::move(shmemWaitUntil), std::move(buffer), std::move(exprTbl), std::move(hcclParam),
                          std::move(rawTensorAddrHolder), std::move(rawTensorDescHolder), std::move(opAttrs),
                          std::move(aicpuCode), funcData);
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

void RunTests(uint32_t tileOpCount, npu::tile_fwk::Distributed::ShmemWaitUntil* shmemWaitUntil) {
    TaskStat* taskStat{nullptr};
    for (uint32_t taskId = 0; taskId < tileOpCount; ++taskId) {
        shmemWaitUntil->EnqueueOp(taskId, taskStat);
        shmemWaitUntil->PollCompleted(nullptr);
    }
}

void TestShmemWaitUntil(const uint32_t tileOpCount) {
    auto [rawAddr, data, allocator, task, shmemWaitUntil, buffer, exprTbl, hcclParam,
          rawTensorAddrHolder, rawTensorDescHolder, opAttrs, aicpuCode, funcData] = InitializeTestEnvironment();

    PrepareTasks(tileOpCount, shmemWaitUntil.get(), aicpuCode, funcData, opAttrs.get());

    RunTests(tileOpCount, shmemWaitUntil.get());
}

TEST(ShmemWaitUntilTest, BasicFunctionality) {
    constexpr int32_t tileOpCount = 1;
    TestShmemWaitUntil(tileOpCount);
}
} // namespace