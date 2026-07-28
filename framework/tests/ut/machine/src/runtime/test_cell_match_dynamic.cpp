/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_cell_match_dynamic.cpp
 * \brief UT for machine/runtime/launcher/cell_match_dynamic.cpp and .h
 */

#include <gtest/gtest.h>
#include <cstring>
#include <memory>
#include <vector>

#define private public
#define protected public

#include "machine/runtime/launcher/cell_match_dynamic.h"
#include "machine/runtime/launcher/device_launcher_binding.h"
#include "machine/runtime/launcher/aicore_model_launcher.h"
#include "machine/device/dynamic/context/device_stitch_context.h"
#include "machine/device/dynamic/context/device_slot_context.h"
#include "machine/utils/dynamic/dev_encode_function_dupped_data.h"
#include "machine/utils/dynamic/dev_encode_function_stitch.h"
#include "machine/utils/dynamic/dev_cell_match_mem_layout.h"

using namespace npu::tile_fwk;
using namespace npu::tile_fwk::dynamic;

TEST(CellMatchDynamicFuncTest, NullAndEmptyInputs_NoCrash)
{
    std::vector<DevDynamicCellMatchStridePatch> patches;
    PatchHostDynamicCellMatchTableDesc(nullptr, patches);

    uint8_t buf[1024] = {0};
    auto* prog = reinterpret_cast<DevAscendProgram*>(buf);
    PatchHostDynamicCellMatchTableDesc(prog, patches);

    WriteDynamicCellMatchStridePatchesToLaunchArgs(nullptr, patches);

    DyndevFunctionAttribute dynAttr;
    ValidateDynamicCellMatchTableMemBudget(dynAttr, nullptr);

    std::unordered_map<std::string, ScalarImmediateType> symbolDict;
    std::vector<DeviceTensorData> inputs;
    std::vector<DeviceTensorData> outputs;
    Evaluator eval(symbolDict, &inputs, &outputs);

    RefillDynamicMemBudgets(nullptr, dynAttr, eval);

    auto evalPatches = PrepareDynamicCellMatchDescPatches(dynAttr, eval);
    EXPECT_TRUE(evalPatches.empty());

    auto launchPatches = PrepareHostDynamicCellMatchForLaunch(dynAttr, eval, nullptr);
    EXPECT_TRUE(launchPatches.empty());
}

TEST(CellMatchDynamicFuncTest, PatchRuntimeDynamicCellMatchMeta_AllBranches)
{
    AicoreModelMemoryUtils memUtils;
    DeviceKernelArgs nullKArgs{};
    PatchRuntimeDynamicCellMatchMeta(memUtils, nullptr, nullKArgs);

    uint8_t zeroBuf[sizeof(DevAscendProgram)] = {0};
    auto* zeroProg = reinterpret_cast<DevAscendProgram*>(zeroBuf);
    DeviceKernelArgs zeroKArgs{};
    zeroProg->memBudget.metadata.dynamicCellMatch = 0;
    PatchRuntimeDynamicCellMatchMeta(memUtils, zeroProg, zeroKArgs);
    EXPECT_EQ(zeroProg->devArgs.dynamicCellMatchAddr, 0ULL);
    EXPECT_EQ(zeroKArgs.runtimeDynamicCellMatchAddr, 0ULL);

    uint8_t nonZeroBuf[sizeof(DevAscendProgram)] = {0};
    auto* nonZeroProg = reinterpret_cast<DevAscendProgram*>(nonZeroBuf);
    DeviceKernelArgs nonZeroKArgs{};
    nonZeroProg->memBudget.metadata.dynamicCellMatch = 1024;
    PatchRuntimeDynamicCellMatchMeta(memUtils, nonZeroProg, nonZeroKArgs);
    EXPECT_NE(nonZeroProg->devArgs.dynamicCellMatchAddr, 0ULL);
    EXPECT_EQ(nonZeroProg->devArgs.dynamicCellMatchCapacity, 1024ULL);
    EXPECT_EQ(nonZeroKArgs.runtimeDynamicCellMatchAddr, nonZeroProg->devArgs.dynamicCellMatchAddr);
    EXPECT_EQ(nonZeroKArgs.runtimeDynamicCellMatchCapacity, 1024ULL);
}

namespace {

struct MiniDup {
    static constexpr size_t kBuf = 4096;
    std::unique_ptr<uint8_t[]> funcBuf;
    std::unique_ptr<uint8_t[]> dupBuf;
    DevAscendFunction* func{nullptr};
    DevAscendFunctionDuppedData* dupData{nullptr};
    DevAscendFunctionDupped dup;
    size_t cursor{0};

    void Align(size_t a) { cursor = (cursor + a - 1) & ~(a - 1); }

    template <typename T>
    T* Alloc(size_t n = 1)
    {
        Align(alignof(T));
        auto* p = reinterpret_cast<T*>(funcBuf.get() + cursor);
        cursor += sizeof(T) * n;
        EXPECT_LT(cursor, kBuf);
        return p;
    }

    void Build(bool withOutcast)
    {
        funcBuf = std::make_unique<uint8_t[]>(kBuf);
        memset(funcBuf.get(), 0, kBuf);
        cursor = sizeof(DevAscendFunction);
        func = reinterpret_cast<DevAscendFunction*>(funcBuf.get());
        func->funcKey = 1;

        func->operationList_.AssignOffsetSize(cursor, 1);
        auto* op = Alloc<DevAscendOperation>();
        new (op) DevAscendOperation();
        op->stitchIndex = 0;
        op->depGraphPredCount = 0;
        op->depGraphSuccList.AssignOffsetSize(0, 0);

        if (withOutcast) {
            Align(alignof(DevAscendFunctionOutcast));
            func->outcastList.AssignOffsetSize(cursor, 1);
            auto* oc = Alloc<DevAscendFunctionOutcast>();
            new (oc) DevAscendFunctionOutcast();
            oc->stitchPolicyFullCoverProducerHubOpIdx = -1;
            oc->producerConsumerList.AssignOffsetSize(0, 0);
            oc->stitchPolicyFullCoverProducerList.AssignOffsetSize(0, 0);
            oc->toSlotList.AssignOffsetSize(0, 0);
            oc->cellMatchRuntimeFullUpdateTable.AssignOffsetSize(0, 0);
        }

        dupBuf = std::make_unique<uint8_t[]>(kBuf);
        memset(dupBuf.get(), 0, kBuf);
        dupData = reinterpret_cast<DevAscendFunctionDuppedData*>(dupBuf.get());
        uint8_t* p = dupBuf.get() + sizeof(DevAscendFunctionDuppedData);
        dupData->source_ = func;
        dupData->operationList_.size = 1;
        dupData->operationList_.predCountBase = static_cast<uint32_t>(p - dupBuf.get());
        *reinterpret_cast<predcount_t*>(p) = 0;
        p += sizeof(predcount_t);
        dupData->operationList_.stitchBase = static_cast<uint32_t>(p - dupBuf.get());
        dupData->operationList_.stitchCount = 1;
        new (p) DevAscendFunctionDuppedStitchList();
        p += sizeof(DevAscendFunctionDuppedStitchList);
        dupData->expressionList_.base = static_cast<uint32_t>(p - dupBuf.get());
        dupData->expressionList_.size = 1;
        *reinterpret_cast<uint64_t*>(p) = 0;

        WsAllocation a;
        a.ptr = reinterpret_cast<uint64_t>(dupData);
        dup = DevAscendFunctionDupped(a);
    }
};

void BindStitchedList(DeviceStitchContext& ctx, DevAscendFunctionDupped* storage, uint32_t n)
{
    ctx.stitchedList_.dataAllocation_.ptr = reinterpret_cast<uint64_t>(storage);
    ctx.stitchedList_.size_ = n;
    ctx.stitchedList_.capacity_ = n;
}

void UnbindStitchedList(DeviceStitchContext& ctx)
{
    ctx.stitchedList_.dataAllocation_.Invalidate();
    ctx.stitchedList_.size_ = 0;
    ctx.stitchedList_.capacity_ = 0;
}

} // namespace

// FullCoverUpdateStitch → FullCoverPartialProdToFullConsStitch early return (empty lists).
TEST(FullCoverUpdateStitchTest, EmptyLists_EarlyReturn)
{
    MiniDup prev;
    MiniDup next;
    prev.Build(true);
    next.Build(false);

    DevAscendFunctionDupped storage[1] = {prev.dup};
    DeviceStitchContext stitchCtx;
    BindStitchedList(stitchCtx, storage, 1);

    DeviceExecuteSlot slot{};
    slot.stitchDupIdx = 0;
    slot.stitchOutcastIdx = 0;

    DevAscendFunctionIncast incast{};
    incast.consumerList.AssignOffsetSize(0, 0);
    incast.stitchPolicyFullCoverConsumerList.AssignOffsetSize(0, 0);
    incast.stitchPolicyFullCoverConsumerAllOpIdxList.AssignOffsetSize(0, 0);

    EXPECT_EQ(stitchCtx.FullCoverUpdateStitch(next.dup, 0, 1, slot, 0, incast), 0u);
    UnbindStitchedList(stitchCtx);
}

// FullCoverPartialProdToFullConsStitch loop: READ producer is skipped (no HandleOneStitch).
TEST(FullCoverUpdateStitchTest, ReadProducerSkipped)
{
    MiniDup prev;
    MiniDup next;
    prev.Build(true);
    next.Build(false);

    auto& outcast = prev.func->GetOutcast(0);
    auto* prod = prev.Alloc<DevAscendFunctionCallOperandUse>();
    new (prod) DevAscendFunctionCallOperandUse(0, -1, -1, CellMatchOpType::READ);
    outcast.producerConsumerList.AssignOffsetSize(reinterpret_cast<uint8_t*>(prod) - prev.funcBuf.get(), 1);

    auto* cons = next.Alloc<DevAscendFunctionCallOperandUse>();
    new (cons) DevAscendFunctionCallOperandUse(0, -1, -1, CellMatchOpType::READ);

    DevAscendFunctionIncast incast{};
    incast.stitchPolicyFullCoverConsumerList.AssignOffsetSize(reinterpret_cast<uint8_t*>(cons) - next.funcBuf.get(), 1);
    incast.consumerList.AssignOffsetSize(0, 0);
    incast.stitchPolicyFullCoverConsumerAllOpIdxList.AssignOffsetSize(0, 0);

    DevAscendFunctionDupped storage[1] = {prev.dup};
    DeviceStitchContext stitchCtx;
    BindStitchedList(stitchCtx, storage, 1);

    DeviceExecuteSlot slot{};
    slot.stitchDupIdx = 0;
    slot.stitchOutcastIdx = 0;

    EXPECT_EQ(stitchCtx.FullCoverUpdateStitch(next.dup, 0, 1, slot, 0, incast), 0u);
    UnbindStitchedList(stitchCtx);
}

// Covers UpdateSlotsForIncastStitch dual Fill (consumerList + stitchPolicyFullCoverConsumerList).
TEST(UpdateSlotsIncastFillTest, DualConsumerLists)
{
    MiniDup root;
    root.Build(false);

    root.Align(alignof(DevAscendFunctionIncast));
    root.func->incastList.AssignOffsetSize(root.cursor, 1);
    auto* ic = root.Alloc<DevAscendFunctionIncast>();
    new (ic) DevAscendFunctionIncast();

    auto* slotIdx = root.Alloc<int>();
    *slotIdx = 0;
    ic->fromSlotList.AssignOffsetSize(reinterpret_cast<uint8_t*>(slotIdx) - root.funcBuf.get(), 1);

    auto* partialConsumer = root.Alloc<DevAscendFunctionCallOperandUse>();
    new (partialConsumer) DevAscendFunctionCallOperandUse(0, -1, -1, CellMatchOpType::READ);
    ic->consumerList.AssignOffsetSize(reinterpret_cast<uint8_t*>(partialConsumer) - root.funcBuf.get(), 1);

    auto* full = root.Alloc<DevAscendFunctionCallOperandUse>();
    new (full) DevAscendFunctionCallOperandUse(0, -1, -1, CellMatchOpType::READ);
    ic->stitchPolicyFullCoverConsumerList.AssignOffsetSize(reinterpret_cast<uint8_t*>(full) - root.funcBuf.get(), 1);
    ic->stitchPolicyFullCoverConsumerAllOpIdxList.AssignOffsetSize(0, 0);

    DevAscendProgramPartialUpdate partial{};
    partial.slotIndex = 0;
    partial.cellMatchTableDesc.SetCacheOpMaxCount({1, 0, 1});
    uint64_t table[4] = {0};
    partial.cellMatchRuntimePartialUpdateTable = DevRelocVector<uint64_t>(4, table);

    DeviceExecuteSlot slots[1]{};
    slots[0].isPartialUpdateStitch = true;
    slots[0].partialUpdate = &partial;

    DeviceSlotContext slotCtx;
    slotCtx.slotList_.dataAllocation_.ptr = reinterpret_cast<uint64_t>(slots);
    slotCtx.slotList_.size_ = 1;
    slotCtx.slotList_.capacity_ = 1;
    slotCtx.workspace_ = nullptr;

    EXPECT_EQ(slotCtx.UpdateSlots(root.dup, 0, 0, 0), 0u);

    slotCtx.slotList_.dataAllocation_.Invalidate();
    slotCtx.slotList_.size_ = 0;
    slotCtx.slotList_.capacity_ = 0;
}
