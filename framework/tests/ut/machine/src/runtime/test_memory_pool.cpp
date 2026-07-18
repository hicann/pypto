/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_memory_pool.cpp
 * \brief
 */

#include <gtest/gtest.h>
#include "machine/runtime/memory_utils/memory_pool.h"

using namespace npu::tile_fwk;

TEST(MemoryBlockTest, NonHuge_FirstAllocation_ReturnsBaseAddr)
{
    uint8_t buf[1024];
    MemoryBlock block(buf, 1024, false);
    void* ptr = block.Allocate(512);
    EXPECT_EQ(ptr, static_cast<void*>(buf));
    EXPECT_EQ(block.usedSize, 1024u);
}

TEST(MemoryBlockTest, NonHuge_SecondAllocation_ReturnsNull)
{
    uint8_t buf[1024];
    MemoryBlock block(buf, 1024, false);
    block.Allocate(512);
    void* ptr = block.Allocate(512);
    EXPECT_EQ(ptr, nullptr);
}

TEST(MemoryBlockTest, NonHuge_AlignExceedsBlockSize_ReturnsNull)
{
    uint8_t buf[256];
    MemoryBlock block(buf, 256, false);
    void* ptr = block.Allocate(512);
    EXPECT_EQ(ptr, nullptr);
}

TEST(MemoryBlockTest, NonHuge_Free_ThrowsError)
{
    uint8_t buf[1024];
    MemoryBlock block(buf, 1024, false);
    EXPECT_THROW(block.Free(buf, 512), std::exception);
}

TEST(MemoryBlockTest, Huge_FirstAllocation_ReturnsBaseAddr)
{
    uintptr_t base = 0x1000;
    MemoryBlock block(reinterpret_cast<void*>(base), 4096, true);
    void* ptr = block.Allocate(512);
    EXPECT_EQ(ptr, reinterpret_cast<void*>(base));
}

TEST(MemoryBlockTest, Huge_SecondAllocation_ReturnsNextChunk)
{
    uintptr_t base = 0x1000;
    MemoryBlock block(reinterpret_cast<void*>(base), 4096, true);
    void* ptr1 = block.Allocate(512);
    void* ptr2 = block.Allocate(512);
    EXPECT_EQ(ptr1, reinterpret_cast<void*>(base));
    EXPECT_EQ(ptr2, reinterpret_cast<void*>(base + 512));
}

TEST(MemoryBlockTest, Huge_AllocationExceedsFreeSpace_ReturnsNull)
{
    uintptr_t base = 0x1000;
    MemoryBlock block(reinterpret_cast<void*>(base), 512, true);
    block.Allocate(512);
    void* ptr = block.Allocate(512);
    EXPECT_EQ(ptr, nullptr);
}

TEST(MemoryBlockTest, Huge_FreeAndReallocate)
{
    uintptr_t base = 0x1000;
    MemoryBlock block(reinterpret_cast<void*>(base), 4096, true);
    void* ptr1 = block.Allocate(1024);
    block.Free(ptr1, 1024);
    void* ptr2 = block.Allocate(1024);
    EXPECT_EQ(ptr2, reinterpret_cast<void*>(base));
}

TEST(MemoryBlockTest, Huge_FreeMergesAdjacentChunks)
{
    uintptr_t base = 0x1000;
    MemoryBlock block(reinterpret_cast<void*>(base), 4096, true);
    void* ptr1 = block.Allocate(1024);
    void* ptr2 = block.Allocate(1024);
    block.Free(ptr1, 1024);
    block.Free(ptr2, 1024);
    void* ptr3 = block.Allocate(4096);
    EXPECT_NE(ptr3, nullptr);
}

TEST(MemoryBlockTest, Huge_FreeNonAdjacentChunks_NoMerge)
{
    uintptr_t base = 0x1000;
    MemoryBlock block(reinterpret_cast<void*>(base), 4096, true);
    void* ptr1 = block.Allocate(1024);
    void* ptr2 = block.Allocate(1024);
    void* ptr3 = block.Allocate(1024);
    (void)ptr2;
    block.Free(ptr1, 1024);
    block.Free(ptr3, 1024);
    EXPECT_EQ(block.freeMap.size(), 2u);
}
