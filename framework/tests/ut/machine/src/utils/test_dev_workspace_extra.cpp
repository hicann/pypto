/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 */

#include <gtest/gtest.h>
#include <cstring>
#include <vector>

#define private public
#include "machine/utils/dynamic/dev_workspace.h"
#undef private

using namespace npu::tile_fwk::dynamic;

class DevWorkspaceTest : public testing::Test {};

TEST_F(DevWorkspaceTest, Constructor_Default)
{
    DeviceWorkspaceAllocator allocator;
    EXPECT_EQ(allocator.devProg_, nullptr);
}

TEST_F(DevWorkspaceTest, Constructor_WithDevProg)
{
    DevAscendProgram prog{};
    DeviceWorkspaceAllocator allocator(&prog);
    EXPECT_EQ(allocator.devProg_, &prog);
}

TEST_F(DevWorkspaceTest, SwitchWParallelWorkSpace_SetsId)
{
    DeviceWorkspaceAllocator allocator;
    allocator.SwitchWParallelWorkSpace(5);
    EXPECT_EQ(allocator.curParallelWsId, 5u);
}

TEST_F(DevWorkspaceTest, StackWorkspaceAddr_ReturnsBase)
{
    DeviceWorkspaceAllocator allocator;
    allocator.stackWorkspaceBase_ = 0x12345678;
    EXPECT_EQ(allocator.StackWorkspaceAddr(), 0x12345678u);
}

TEST_F(DevWorkspaceTest, StandardStackWorkspacePerCore_ReturnsValue)
{
    DeviceWorkspaceAllocator allocator;
    allocator.standardStackWorkspacePerCore_ = 4096;
    EXPECT_EQ(allocator.StandardStackWorkspacePerCore(), 4096u);
}

TEST_F(DevWorkspaceTest, StitchCacheAddr_ReturnsPointer)
{
    DeviceWorkspaceAllocator allocator;
    allocator.stitchCacheAddr_ = 0xABCD0000;
    EXPECT_EQ(allocator.StitchCacheAddr(), reinterpret_cast<uint64_t*>(0xABCD0000));
}

TEST_F(DevWorkspaceTest, RootFuncMaxCallOpsize_ReturnsValue)
{
    DeviceWorkspaceAllocator allocator;
    allocator.rootFuncMaxCallOpsize_ = 100;
    EXPECT_EQ(allocator.RootFuncMaxCallOpsize(), 100u);
}

TEST_F(DevWorkspaceTest, StitchCacheEpoch_ReturnsLow16Bits)
{
    DevAscendProgram prog{};
    prog.stitchCacheEpoch_ = 0x12345678;
    DeviceWorkspaceAllocator allocator(&prog);
    EXPECT_EQ(allocator.StitchCacheEpoch(), 0x5678u);
}

TEST_F(DevWorkspaceTest, AdvanceStitchCacheEpoch_IncrementsEpoch)
{
    DevAscendProgram prog{};
    prog.stitchCacheEpoch_ = 100;
    prog.memBudget.metadata.stitchCacheSize = 0;
    DeviceWorkspaceAllocator allocator(&prog);
    allocator.stitchCacheAddr_ = 0;
    allocator.AdvanceStitchCacheEpoch();
    EXPECT_EQ(prog.stitchCacheEpoch_, 101u);
}

TEST_F(DevWorkspaceTest, AdvanceStitchCacheEpoch_SkipsZeroLowBits)
{
    DevAscendProgram prog{};
    prog.stitchCacheEpoch_ = 0xFFFF;
    prog.memBudget.metadata.stitchCacheSize = 0;
    DeviceWorkspaceAllocator allocator(&prog);
    allocator.stitchCacheAddr_ = 0;
    allocator.AdvanceStitchCacheEpoch();
    EXPECT_EQ(prog.stitchCacheEpoch_, 0x10001u);
}

TEST_F(DevWorkspaceTest, TENSOR_ADDR_ALIGNMENT_CorrectValue) { EXPECT_EQ(TENSOR_ADDR_ALIGNMENT, 512); }

TEST_F(DevWorkspaceTest, SUBMMIT_TASK_QUE_SIZE_CorrectValue) { EXPECT_EQ(SUBMMIT_TASK_QUE_SIZE, 512u); }

TEST_F(DevWorkspaceTest, ALLOC_NUM_ONE_SLAB_CorrectValue) { EXPECT_EQ(ALLOC_NUM_ONE_SLAB, 4); }
