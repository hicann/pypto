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
 * \file test_encode.cpp
 * \brief
 */

#include <gtest/gtest.h>
#include <cstdint>

#include "interface/function/rebuildable_attribute.h"
#include "machine/utils/dynamic/dev_encode.h"
#include "machine/utils/dynamic/dev_encode_workspace.h"
#include "machine/utils/dynamic/workspace_budget_calculator.h"
#include "machine/utils/dynamic/dev_workspace.h"
#include "tilefwk/data_type.h"
#include "tilefwk/tilefwk_op.h"
#include "interface/configs/config_manager.h"
#include "interface/program/program.h"

using namespace npu::tile_fwk;
using namespace npu::tile_fwk::dynamic;

class TestDevEncode : public testing::Test {};

namespace {

uint32_t OutcastCacheDepthFromPool(const DevAscendProgram* devProg)
{
    const uint32_t parallelism = devProg->memBudget.tensor.parallelism;
    if (devProg->slotSize == 0 || parallelism == 0) {
        return 0;
    }
    return devProg->runtimeOutcastPoolSize / (devProg->slotSize * parallelism) - 1;
}

} // namespace

TEST_F(TestDevEncode, DevSymShape)
{
    DevSymShape shape;
    shape.SetShape({SymInt(true, 0), SymInt(true, 2), SymInt(2)}); // 4, 8, 2
    uint64_t exprTbl[] = {4, 6, 8};
    uint64_t strides[3] = {0};
    shape.ToStride(strides, exprTbl);
    EXPECT_EQ(strides[0], 16);
    EXPECT_EQ(strides[1], 2);
    EXPECT_EQ(strides[2], 1);
}

TEST_F(TestDevEncode, test_parse_unroll_times_from_name)
{
    EXPECT_EQ(ParseUnrollTimesFromName("TENSOR_main_LoopUnroll2_L0_Unroll1_PATH0_hiddenfunc0_root"), 2u);
    EXPECT_EQ(ParseUnrollTimesFromName("kernel_LoopUnroll512_suffix"), 512u);
    EXPECT_EQ(ParseUnrollTimesFromName("kernel_Unroll64"), 64u);
    EXPECT_EQ(ParseUnrollTimesFromName("foo_LoopUnroll2_bar_Unroll4"), 8u);
    EXPECT_EQ(ParseUnrollTimesFromName("plain_name"), 1u);
}

TEST_F(TestDevEncode, test_dev_encode_program)
{
    config::SetRuntimeOption(STITCH_FUNCTION_MAX_NUM, 64);
    config::SetPlatformConfig(KEY_ENABLE_AIHAC_BACKEND, true);
    TileShape::Current().SetVecTile(32, 32);
    TileShape::Current().SetCubeTile({32, 32}, {32, 32}, {32, 32});
    constexpr int LOOP_COUNT_INNER = 4;
    int s = 32;
    Tensor t0(DT_FP32, {s, s}, "t0");
    Tensor t1(DT_FP32, {s, s}, "t1");
    Tensor t2(DT_FP32, {s, s}, "t2");
    Tensor out(DT_FP32, {LOOP_COUNT_INNER * s, s}, "out");

    // clc
    FUNCTION("main_LoopUnroll2", {t0, t1, t2}, {out})
    {
        LOOP("main_LoopUnroll2_L0", FunctionType::DYNAMIC_LOOP, i, LoopRange(LOOP_COUNT_INNER))
        {
            auto temp = Add(t0, t0);
            SymbolicScalar s_min = std::ternary(i < 2, i, i + 1);

            IF(s_min == i) { temp = Add(temp, t1); }
            ELSE IF(s_min == i + 1) { temp = Add(temp, t2); }
            Assemble(temp, {i * s, 0}, out);
        }
    }

    std::shared_ptr<DyndevFunctionAttribute> funcDynDev =
        Program::GetInstance().GetLastFunction()->GetDyndevAttribute();
    ASSERT_NE(funcDynDev, nullptr);
    DevAscendProgram* devProg = reinterpret_cast<DevAscendProgram*>(funcDynDev->devProgBinary.data());
    ASSERT_NE(devProg, nullptr);
    constexpr uint32_t kConfiguredStitchNumMax = 64;
    EXPECT_EQ(devProg->stitchMaxFunctionNum, kConfiguredStitchNumMax);
    EXPECT_EQ(OutcastCacheDepthFromPool(devProg), kConfiguredStitchNumMax);

    Function* func = Program::GetInstance().GetLastFunction();
    ASSERT_NE(func, nullptr);
    auto* wsAttr = RebuildableAttributeManager::GetInstance().GetAttr<RebuildableWorkspaceDesc>(func);
    ASSERT_NE(wsAttr, nullptr);
    EXPECT_EQ(devProg->memBudget.tensor.rootInnerSpilledMem, wsAttr->desc.maxRootInnerSpilledMem);
    EXPECT_EQ(devProg->memBudget.tensor.devTaskInnerExclusiveOutcasts, wsAttr->desc.maxRootTotalExclusiveOutcastMem);

    devProg->RelocProgram(0, reinterpret_cast<uint64_t>(devProg), true);
    devProg->controlFlowCache.isRecording = false;
    uint64_t contextWorkspaceAddr = devProg->controlFlowCache.contextWorkspaceAddr;
    devProg->controlFlowCache.IncastOutcastAddrReloc(contextWorkspaceAddr, 0, nullptr);
    devProg->controlFlowCache.RuntimeAddrRelocWorkspace(
        contextWorkspaceAddr, 0, nullptr, nullptr, nullptr, devProg->GetParallelism());
    devProg->controlFlowCache.RuntimeAddrRelocProgram(reinterpret_cast<uint64_t>(devProg), 0);
    devProg->controlFlowCache.TaskAddrRelocWorkspace(contextWorkspaceAddr, 0, nullptr);
    devProg->controlFlowCache.TaskAddrRelocProgramAndCtrlCache(
        reinterpret_cast<uint64_t>(devProg), reinterpret_cast<uint64_t>(&devProg->controlFlowCache), 0, 0);
    devProg->controlFlowCache.isActivated = true;

    devProg->Dump(0, true);
    devProg->DumpFile("./dum_dev_program.txt");
    devProg->ResetRerun();
    devProg->RuntimeVerify(0, 0);
    EXPECT_NE(devProg->GetInputTensorSlotIndexList().empty(), true);
    EXPECT_NE(devProg->GetOutputTensorSlotIndexList().empty(), true);
    EXPECT_NE(devProg->GetAssembleTensorSlotIndexList().empty(), true);
    (void)devProg->GetDevControlFlowBinary();
    (void)devProg->GetHostControlFlowBinary();
    (void)devProg->GetExpressionTableBinary();

    DevAscendFunction* devFunc = devProg->GetFunction(0);
    ASSERT_NE(devFunc, nullptr);
    EXPECT_NE(devProg->GetFunctionByRawName(devFunc->GetRawName()), nullptr);
    devFunc->Dump();
    EXPECT_EQ(devFunc->HasValueDepend(), false);
    EXPECT_EQ(devFunc->LookupIncastBySlotIndex(0), 0);
    EXPECT_EQ(devFunc->LookupOutcastBySlotIndex(0), -1);
    std::vector<int> slotIndexList = {0};
    (void)devFunc->LookupIncastBySlotIndexList(slotIndexList);
    (void)devFunc->LookupOutcastBySlotIndexList(slotIndexList);

    DevAscendFunction* devFunc1 = devProg->GetFunction(1);
    if (devFunc1 != nullptr) {
        devFunc->LookupConnectionSlotIndexFrom(devFunc1);
    }

    DevAscendFunctionDuppedData* devFuncDuppedData = devFunc->GetDuppedData();
    ASSERT_NE(devFuncDuppedData, nullptr);
    devFuncDuppedData->source_ = devFunc;
    (void)devFuncDuppedData->Dump();

    devProg->ResetFromLaunch();
}
static DevAscendProgram* BuildAndGetDevProgForExpectedMaxCachedNum()
{
    constexpr int LOOP_COUNT_INNER = 4;
    int s = 32;
    TileShape::Current().SetVecTile(32, 32);
    TileShape::Current().SetCubeTile({32, 32}, {32, 32}, {32, 32});
    Tensor t0(DT_FP32, {s, s}, "t0");
    Tensor t1(DT_FP32, {s, s}, "t1");
    Tensor t2(DT_FP32, {s, s}, "t2");
    Tensor out(DT_FP32, {LOOP_COUNT_INNER * s, s}, "out");
    FUNCTION("stitch_max_cached_num", {t0, t1, t2}, {out})
    {
        LOOP("stitch_max_cached_num_L0", FunctionType::DYNAMIC_LOOP, i, LoopRange(LOOP_COUNT_INNER))
        {
            auto temp = Add(t0, t0);
            SymbolicScalar s_min = std::ternary(i < 2, i, i + 1);
            IF(s_min == i) { temp = Add(temp, t1); }
            ELSE IF(s_min == i + 1) { temp = Add(temp, t2); }
            Assemble(temp, {i * s, 0}, out);
        }
    }
    std::shared_ptr<DyndevFunctionAttribute> funcDynDev =
        Program::GetInstance().GetLastFunction()->GetDyndevAttribute();
    EXPECT_NE(funcDynDev, nullptr);
    if (funcDynDev == nullptr) {
        return nullptr;
    }
    DevAscendProgram* devProg = reinterpret_cast<DevAscendProgram*>(funcDynDev->devProgBinary.data());
    EXPECT_NE(devProg, nullptr);
    return devProg;
}
TEST_F(TestDevEncode, test_max_stitch_function_num)
{
    // case1:
    Program::GetInstance().Reset();
    config::Reset();
    config::SetPlatformConfig(KEY_ENABLE_AIHAC_BACKEND, true);
    config::SetRuntimeOption(STITCH_FUNCTION_MAX_NUM, 256);
    DevAscendProgram* devProg1 = BuildAndGetDevProgForExpectedMaxCachedNum();
    ASSERT_NE(devProg1, nullptr);
    constexpr uint32_t kConfiguredStitchNumMax = 256;
    EXPECT_EQ(devProg1->stitchMaxFunctionNum, kConfiguredStitchNumMax);
    const uint32_t outcastDepth = OutcastCacheDepthFromPool(devProg1);
    EXPECT_EQ(outcastDepth, kConfiguredStitchNumMax);
    const uint32_t expectedPool =
        devProg1->slotSize * (outcastDepth + 1) * devProg1->GetParallelism();
    EXPECT_EQ(devProg1->runtimeOutcastPoolSize, expectedPool);
}

TEST_F(TestDevEncode, test_dev_func_dupped)
{
    DevAscendRawTensor rawTensor;
    std::vector<std::string> lines;
    std::stringstream oss;
    DevAscendFunctionDupped funcDuppped;
    funcDuppped.DumpRawShape(&rawTensor, 0, lines, oss);
}

TEST_F(TestDevEncode, test_init_wrap_info)
{
    Program::GetInstance().Reset();
    config::Reset();
    config::SetPlatformConfig(KEY_ENABLE_AIHAC_BACKEND, true);
    TileShape::Current().SetVecTile(32, 32);
    TileShape::Current().SetCubeTile({32, 32}, {32, 32}, {32, 32});
    int s = 32;
    Tensor t0(DT_FP32, {s, s}, "t0");
    Tensor t1(DT_FP32, {s, s}, "t1");
    Tensor out(DT_FP32, {s, s}, "out");

    FUNCTION("test_wrap_info", {t0, t1}, {out})
    {
        auto temp = Add(t0, t1);
        Assemble(temp, {0, 0}, out);
    }

    std::shared_ptr<DyndevFunctionAttribute> funcDynDev =
        Program::GetInstance().GetLastFunction()->GetDyndevAttribute();
    ASSERT_NE(funcDynDev, nullptr);
    DevAscendProgram* devProg = reinterpret_cast<DevAscendProgram*>(funcDynDev->devProgBinary.data());
    ASSERT_NE(devProg, nullptr);

    devProg->RelocProgram(0, reinterpret_cast<uint64_t>(devProg), true);
    devProg->controlFlowCache.isRecording = false;
    uint64_t contextWorkspaceAddr = devProg->controlFlowCache.contextWorkspaceAddr;
    devProg->controlFlowCache.IncastOutcastAddrReloc(contextWorkspaceAddr, 0, nullptr);
    devProg->controlFlowCache.RuntimeAddrRelocWorkspace(
        contextWorkspaceAddr, 0, nullptr, nullptr, nullptr, devProg->GetParallelism());
    devProg->controlFlowCache.RuntimeAddrRelocProgram(reinterpret_cast<uint64_t>(devProg), 0);
    devProg->controlFlowCache.TaskAddrRelocWorkspace(contextWorkspaceAddr, 0, nullptr);
    devProg->controlFlowCache.TaskAddrRelocProgramAndCtrlCache(
        reinterpret_cast<uint64_t>(devProg), reinterpret_cast<uint64_t>(&devProg->controlFlowCache), 0, 0);
    devProg->controlFlowCache.isActivated = true;

    DevAscendFunction* devFunc = devProg->GetFunction(0);
    ASSERT_NE(devFunc, nullptr);

    EXPECT_GE(devFunc->wrapIdNum_, 0);
}

TEST_F(TestDevEncode, test_workspace_budget_calculator_unroll_ge_depth)
{
    WorkspaceDesc desc;
    desc.maxPerUnitRootInnerSpilledMem = 512;
    desc.maxPerUnitRootTotalExclusiveOutcastMem = 512;
    desc.maxStaticOutcastMem = 200;
    desc.totalExclusiveOutcastSlot = 2;
    desc.totalAssembleOutcastSlot = 1;

    WorkspaceDesc::WorkspacePerRootFunctionDesc lowUnroll;
    lowUnroll.unroll = 1;
    lowUnroll.rootInnerSpilledRawMem = 100;
    lowUnroll.rootTotalExclusiveOutcastRawMem = 50;

    WorkspaceDesc::WorkspacePerRootFunctionDesc highUnroll;
    highUnroll.unroll = 64;
    highUnroll.rootInnerSpilledRawMem = 8000;
    highUnroll.rootTotalExclusiveOutcastRawMem = 4000;

    desc.rootFuncDescList = {lowUnroll, highUnroll};

    WorkspaceDesc wsK1 = desc;
    BuildTensorWorkspaceFromDescriptor(wsK1, 1);
    EXPECT_EQ(wsK1.maxRootInnerSpilledMem, 8000u);
    EXPECT_EQ(wsK1.maxRootTotalExclusiveOutcastMem, 4000u);

    WorkspaceDesc wsK64 = desc;
    BuildTensorWorkspaceFromDescriptor(wsK64, 64);
    EXPECT_EQ(wsK64.maxRootInnerSpilledMem, 512u * 64u);
    EXPECT_EQ(wsK64.maxRootTotalExclusiveOutcastMem, 512u * 64u);

    WorkspaceDesc wsK128 = desc;
    BuildTensorWorkspaceFromDescriptor(wsK128, 128);
    EXPECT_EQ(wsK128.maxRootInnerSpilledMem, 512u * 128u);
    EXPECT_EQ(wsK128.maxRootTotalExclusiveOutcastMem, 512u * 128u);
}

TEST_F(TestDevEncode, test_workspace_budget_calculator_resolve_depth)
{
    WorkspaceDesc desc;
    desc.maxPerUnitRootInnerSpilledMem = 4096;
    desc.maxStaticOutcastMem = 4096;
    desc.totalExclusiveOutcastSlot = 1;

    WorkspaceDesc::WorkspacePerRootFunctionDesc profile;
    profile.unroll = 1;
    profile.rootInnerSpilledRawMem = 4096;
    desc.rootFuncDescList = {profile};

    RuntimeWorkspaceConfig cfg;
    cfg.stitchNumMax = 64;
    cfg.parallelism = 1;
    cfg.aicoreSpilled = 0;
    cfg.debugTotal = 0;

    const StitchDepthConfig depth = ResolveStitchDepthConfig(desc, cfg);
    EXPECT_EQ(depth.kEff, 64u);
    EXPECT_EQ(depth.outcastCacheDepth, 64u);
    EXPECT_EQ(depth.stitchMaxFunctionNum, 64u);
    EXPECT_GT(depth.encodedWorkspaceSize, 0u);
}

TEST_F(TestDevEncode, test_workspace_budget_calculator_outcast_cache_depth)
{
    WorkspaceDesc desc;
    WorkspaceDesc::WorkspacePerRootFunctionDesc unroll512;
    unroll512.unroll = 512;
    WorkspaceDesc::WorkspacePerRootFunctionDesc unroll1;
    unroll1.unroll = 1;
    desc.rootFuncDescList = {unroll512, unroll1};

    EXPECT_EQ(DeriveOutcastCacheDepth(desc, 64, 64), 64u);
    EXPECT_EQ(DeriveOutcastCacheDepth(desc, 8, 64), 8u);
    EXPECT_EQ(DeriveOutcastCacheDepth(desc, 1, 64), 1u);
}

TEST_F(TestDevEncode, test_workspace_flex_io_outcast_skips_assemble_mark)
{
    Program::GetInstance().Reset();
    config::Reset();
    config::SetPlatformConfig(KEY_ENABLE_AIHAC_BACKEND, true);
    config::SetRuntimeOption(STITCH_FUNCTION_MAX_NUM, 128);
    TileShape::Current().SetVecTile(32, 32);
    TileShape::Current().SetCubeTile({32, 32}, {32, 32}, {32, 32});
    constexpr int kLoopCount = 4;
    const int s = 32;
    Tensor t0(DT_FP32, {s, s}, "t0");
    Tensor t1(DT_FP32, {s, s}, "t1");
    Tensor t2(DT_FP32, {s, s}, "t2");
    Tensor out(DT_FP32, {kLoopCount * s, s}, "out");

    FUNCTION("workspace_flex_io_skip", {t0, t1, t2}, {out})
    {
        LOOP("workspace_flex_io_skip_L0", FunctionType::DYNAMIC_LOOP, i, LoopRange(kLoopCount))
        {
            auto temp = Add(t0, t0);
            SymbolicScalar s_min = std::ternary(i < 2, i, i + 1);
            IF(s_min == i) { temp = Add(temp, t1); }
            ELSE IF(s_min == i + 1) { temp = Add(temp, t2); }
            Assemble(temp, {i * s, 0}, out);
        }
    }

    Function* func = Program::GetInstance().GetLastFunction();
    ASSERT_NE(func, nullptr);
    auto dyndev = func->GetDyndevAttribute();
    ASSERT_NE(dyndev, nullptr);
    DevAscendProgram* devProg = reinterpret_cast<DevAscendProgram*>(dyndev->devProgBinary.data());
    ASSERT_NE(devProg, nullptr);

    const uint64_t progAddr = reinterpret_cast<uint64_t>(devProg);
    devProg->RelocProgram(0, progAddr);
    const WorkspaceDesc flex = CollectWorkspaceDesc(
        func, *devProg, dyndev->constructAssembleNeedAllocRuntimeSlots);
    devProg->RelocProgram(progAddr, 0);

    EXPECT_EQ(flex.totalAssembleOutcastSlot, dyndev->inoutLink.assembleSlotIndexList.size());
    EXPECT_LE(flex.totalExclusiveOutcastSlot, devProg->slotSize);
    EXPECT_LT(flex.totalAssembleOutcastSlot, dyndev->inoutLink.inputSlotIndexList.size());

    const uint32_t stitchNumMax = ConfiguredStitchFunctionMaxNum();
    WorkspaceDesc flexWs = flex;
    BuildTensorWorkspaceFromDescriptor(flexWs, stitchNumMax);
    EXPECT_EQ(flexWs.totalAssembleOutcastSlot, dyndev->inoutLink.assembleSlotIndexList.size());
    EXPECT_LT(flexWs.devTaskBoundaryOutcastNum, devProg->slotSize * stitchNumMax);
}

