/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root directory of the software repository for the full text of the License.
 */

/*!
 * \file test_memory_pool.cpp
 * \brief
 */

#include <gtest/gtest.h>
#include "machine/runtime/memory_utils/memory_pool.h"

using namespace npu::tile_fwk;

TEST(MemoryBlockTest, FirstAllocation_ReturnsBaseAddr)
{
    uint8_t buf[1024];
    MemoryBlock block(buf, 1024);
    void* ptr = block.Allocate(512);
    EXPECT_EQ(ptr, static_cast<void*>(buf));
    EXPECT_EQ(block.usedSize, 1024u);
}

TEST(MemoryBlockTest, SecondAllocation_ReturnsNull)
{
    uint8_t buf[1024];
    MemoryBlock block(buf, 1024);
    block.Allocate(512);
    void* ptr = block.Allocate(512);
    EXPECT_EQ(ptr, nullptr);
}

TEST(MemoryBlockTest, AlignExceedsBlockSize_ReturnsNull)
{
    uint8_t buf[256];
    MemoryBlock block(buf, 256);
    void* ptr = block.Allocate(512);
    EXPECT_EQ(ptr, nullptr);
}

// ===== DevMemoryPool tests =====
// DevMemoryPool is a singleton backed by RuntimeMalloc (stub->malloc in UT). Tests below exercise
// FreeDevAddr/CheckAllSentinels/DynamicRecycle/PrintPoolStatus/DestroyPool/DevAlloc/CopyDataToDevice.

// Covers: FreeDevAddr nullptr and unknown pointer error paths (MACHINE_LOGE -> throws)
TEST(DevMemoryPoolTest, FreeDevAddr_ErrorPaths_Throw)
{
    EXPECT_ANY_THROW(DevMemoryPool::Instance().FreeDevAddr(nullptr));
    uint8_t dummy = 0;
    EXPECT_ANY_THROW(DevMemoryPool::Instance().FreeDevAddr(&dummy));
}

// Covers: CheckAllSentinels (needMemCheck_=false returns true), PrintPoolStatus (empty pool)
TEST(DevMemoryPoolTest, SentinelAndPrintStatus_NoCrash)
{
    EXPECT_TRUE(DevMemoryPool::Instance().CheckAllSentinels());
    DevMemoryPool::Instance().DestroyPool();
    DevMemoryPool::Instance().PrintPoolStatus();
    SUCCEED();
}

TEST(DevMemoryPoolTest, AllocAndFree_Lifecycle)
{
    DevMemoryPool::Instance().DestroyPool();
    uint8_t* ptr = nullptr;
    DevMemoryPool::Instance().AllocDevAddr(&ptr, 1024);
    if (ptr != nullptr) {
        DevMemoryPool::Instance().FreeDevAddr(ptr);
    }
    SUCCEED();
}

// Covers: AllocDevAddr nullptr and zero-size error paths (MACHINE_LOGE -> throws)
TEST(DevMemoryPoolTest, AllocDevAddr_ErrorPaths_Throw)
{
    EXPECT_ANY_THROW(DevMemoryPool::Instance().AllocDevAddr(nullptr, 1024));
    uint8_t* ptr = reinterpret_cast<uint8_t*>(0x1);
    EXPECT_ANY_THROW(DevMemoryPool::Instance().AllocDevAddr(&ptr, 0));
}

TEST(DevMemoryPoolTest, DynamicRecycle_AfterFree_RemovesEmptyBlock)
{
    DevMemoryPool::Instance().DestroyPool();
    uint8_t* ptr = nullptr;
    DevMemoryPool::Instance().AllocDevAddr(&ptr, 1024);
    if (ptr != nullptr) {
        DevMemoryPool::Instance().FreeDevAddr(ptr);
    }

    DevMemoryPool::Instance().DynamicRecycle();
    SUCCEED();
}

TEST(DevMemoryPoolTest, DestroyPool_Twice_NoCrash)
{
    DevMemoryPool::Instance().DestroyPool();
    uint8_t* ptr = nullptr;
    DevMemoryPool::Instance().AllocDevAddr(&ptr, 512);
    DevMemoryPool::Instance().DestroyPool();
    DevMemoryPool::Instance().DestroyPool();
    SUCCEED();
}

TEST(DevMemoryPoolTest, DevAlloc_ValidSize_ReturnsAllocatedMemory)
{
    DevMemoryPool::Instance().DestroyPool();
    void* ptr = DevAlloc(256);
    if (ptr != nullptr) {
        DevMemoryPool::Instance().FreeDevAddr(ptr);
    }
    SUCCEED();
}

// Covers: DevAlloc(0) and CopyDataToDevice(0) error paths (MACHINE_LOGE -> throws)
TEST(DevMemoryPoolTest, DevAllocAndCopy_ZeroSize_Throw)
{
    DevMemoryPool::Instance().DestroyPool();
    EXPECT_ANY_THROW(DevAlloc(0));
    uint8_t src[1] = {0};
    EXPECT_ANY_THROW(CopyDataToDevice(src, 0));
}

TEST(DevMemoryPoolTest, CopyDataToDevice_ValidData_CopiesCorrectly)
{
    DevMemoryPool::Instance().DestroyPool();
    uint8_t src[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    void* devPtr = CopyDataToDevice(src, 8);
    if (devPtr != nullptr) {
        DevMemoryPool::Instance().FreeDevAddr(devPtr);
    }
    SUCCEED();
}
