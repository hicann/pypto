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
 * \file test_runtime_outcast_tensor.cpp
 * \brief
 */

#include <gtest/gtest.h>

#define private public

#include "machine/utils/dynamic/runtime_outcast_tensor.h"
#include "machine/utils/dynamic/dev_workspace.h"

#if DEBUG_INFINITE_LIFETIME
#error "Unexpectedly turned on DEBUG_INFINITE_LIFETIME in this test file"
#endif

using namespace npu::tile_fwk::dynamic;

class RuntimeOutcastTensorTest : public ::testing::Test {
protected:
    void SetUp() override {
    }

    void TearDown() override {
    }
};

TEST_F(RuntimeOutcastTensorTest, ConstructAndFields) {
    RuntimeOutcastTensor t(0xDEADBEEFull, RuntimeTensorMemProperty::EXTERNAL, 42u);
    EXPECT_EQ(t.addr, static_cast<uintdevptr_t>(0xDEADBEEFull));
    EXPECT_EQ(t.property, RuntimeTensorMemProperty::EXTERNAL);
    EXPECT_EQ(t.refCnt, 42u);
}

TEST_F(RuntimeOutcastTensorTest, DumpFormatWithNonZeroAddr) {
    RuntimeOutcastTensor t(0x1234ABCDull, RuntimeTensorMemProperty::EXTERNAL, 1u);
    std::string s = t.Dump();
    // std::hex outputs lowercase letters, and no leading zeros are added
    EXPECT_EQ(s, std::string("&0x1234abcd, EXTERNAL"));
}

TEST_F(RuntimeOutcastTensorTest, DumpFormatWithZeroAddr) {
    RuntimeOutcastTensor t(0x0ull, RuntimeTensorMemProperty::BOUNDARY_OUTCAST, 0u);
    std::string s = t.Dump();
    EXPECT_EQ(s, std::string("&0x0, BOUNDARY_OUTCAST"));
}

TEST_F(RuntimeOutcastTensorTest, GetRuntimeTensorMemPropertyNameMatchesEnum) {
    EXPECT_STREQ(GetRuntimeTensorMemPropertyName(RuntimeTensorMemProperty::EXTERNAL), "EXTERNAL");
    EXPECT_STREQ(GetRuntimeTensorMemPropertyName(RuntimeTensorMemProperty::DEVTASK_INNER_OUTCAST), "DEVTASK_INNER_OUTCAST");
    EXPECT_STREQ(GetRuntimeTensorMemPropertyName(RuntimeTensorMemProperty::BOUNDARY_OUTCAST), "BOUNDARY_OUTCAST");
}

// Helper to construct and initialize a DeviceWorkspaceAllocator with reasonable
// metadata budgets so `InitAicpuStitchSlabAllocator` won't assert.
static void InitDeviceWorkspaceAllocatorForTest(DeviceWorkspaceAllocator &d, DevAscendProgram &devProg,
                                                std::vector<uint8_t> &workspace) {
    DevStartArgs args;

    // Ensure stitch pool and general metadata are non-zero and large enough
    devProg.memBudget.metadata.general = 1u << 18; // 256KB
    devProg.memBudget.metadata.stitchPool = 1u << 16; // 64KB

    args.deviceRuntimeDataDesc.generalAddr = reinterpret_cast<uint64_t>(workspace.data());
    // Put stitch pool at an offset within the same workspace region
    args.deviceRuntimeDataDesc.stitchPoolAddr = reinterpret_cast<uint64_t>(workspace.data()) +
            devProg.memBudget.metadata.general; // offset 256KB

    devProg.devArgs.nrAic = 1;
    devProg.devArgs.nrAiv = 1;
    devProg.devArgs.nrValidAic = 0;

    args.InitWorkspace(&devProg, workspace.data());

    d.Init(&args);

    std::cerr << "InitDeviceWorkspaceAllocatorForTest finished" << std::endl;
}

TEST_F(RuntimeOutcastTensorTest, DeviceWorkspaceAllocatorBasicOps) {
    // Small workspace and dev program with minimal budgets to initialize allocators
    std::vector<uint8_t> workspace(1u << 20); // 1MB

    DeviceWorkspaceAllocator d;
    DevAscendProgram devProg{};
    devProg.runtimeOutcastPoolSize = 8;
    // Keep tensor budgets small/zero but valid
    devProg.memBudget.tensor.rootInner = 0;
    devProg.memBudget.tensor.devTaskInnerExclusiveOutcasts = 0;
    devProg.memBudget.tensor.maxStaticOutcastMem = 0;
    devProg.memBudget.tensor.maxDynamicAssembleOutcastMem = 0;
    devProg.memBudget.tensor.devTaskBoundaryOutcastNum = 0;

    InitDeviceWorkspaceAllocatorForTest(d, devProg, workspace);

    // Now exercise runtime outcast APIs
    ItemPoolIter a = d.MakeRuntimeOutcastTensor(0xAAull, RuntimeTensorMemProperty::EXTERNAL);
    EXPECT_NE(a, ITEM_POOL_INVALID_INDEX);

    auto &t = d.GetRuntimeOutcastTensor(a);
    EXPECT_EQ(t.addr, static_cast<uintdevptr_t>(0xAAull));
    EXPECT_EQ(t.property, RuntimeTensorMemProperty::EXTERNAL);
    EXPECT_EQ(t.refCnt, 1u);

    d.RuntimeOutcastTensorRef(a);
    EXPECT_EQ(t.refCnt, 2u);

    d.RuntimeOutcastTensorDeref(a);
    EXPECT_EQ(t.refCnt, 1u);

    d.RuntimeOutcastTensorReplaceAddrWithoutRecycle(a, 0xBBull, RuntimeTensorMemProperty::BOUNDARY_OUTCAST);
    EXPECT_EQ(t.addr, static_cast<uintdevptr_t>(0xBBull));
    EXPECT_EQ(t.property, RuntimeTensorMemProperty::BOUNDARY_OUTCAST);

    ItemPoolIter b = d.MakeRuntimeOutcastTensor(0xCCull, RuntimeTensorMemProperty::EXTERNAL);
    d.RuntimeOutcastTensorAssign(b, a);
    EXPECT_EQ(d.GetRuntimeOutcastTensor(b).addr, static_cast<uintdevptr_t>(0xBBull));
    EXPECT_EQ(d.GetRuntimeOutcastTensor(b).refCnt, 2u);
}

TEST_F(RuntimeOutcastTensorTest, DeviceWorkspaceAllocatorSafeRefDerefNoCrash) {
    std::vector<uint8_t> workspace(1u << 20); // 1MB

    DeviceWorkspaceAllocator d;
    DevAscendProgram devProg{};
    devProg.runtimeOutcastPoolSize = 4;

    InitDeviceWorkspaceAllocatorForTest(d, devProg, workspace);

    ItemPoolIter invalid = ITEM_POOL_INVALID_INDEX;
    d.RuntimeOutcastTensorRefSafe(invalid);
    d.RuntimeOutcastTensorDerefSafe(invalid);
    SUCCEED();
}

TEST_F(RuntimeOutcastTensorTest, DerefToZeroReturnsItemToPool) {
    std::vector<uint8_t> workspace(1u << 20); // 1MB

    DeviceWorkspaceAllocator d;
    DevAscendProgram devProg{};
    devProg.runtimeOutcastPoolSize = 4;

    InitDeviceWorkspaceAllocatorForTest(d, devProg, workspace);

    // initial free items should equal pool size
    size_t freeBefore = d.runtimeOutcastTensorPool_.FreeItemNum();
    EXPECT_EQ(freeBefore, devProg.runtimeOutcastPoolSize);

    ItemPoolIter a = d.MakeRuntimeOutcastTensor(0xAAAull, RuntimeTensorMemProperty::EXTERNAL);
    EXPECT_EQ(d.runtimeOutcastTensorPool_.FreeItemNum(), devProg.runtimeOutcastPoolSize - 1);

    // deref once (refCnt -> 0) should destroy and return to pool
    d.RuntimeOutcastTensorDeref(a);
    EXPECT_EQ(d.runtimeOutcastTensorPool_.FreeItemNum(), devProg.runtimeOutcastPoolSize);
}

TEST_F(RuntimeOutcastTensorTest, AssignReleasesPreviousDestination) {
    std::vector<uint8_t> workspace(1u << 20); // 1MB

    DeviceWorkspaceAllocator d;
    DevAscendProgram devProg{};
    devProg.runtimeOutcastPoolSize = 4;

    InitDeviceWorkspaceAllocatorForTest(d, devProg, workspace);

    // allocate two items A and B
    ItemPoolIter A = d.MakeRuntimeOutcastTensor(0xA1ull, RuntimeTensorMemProperty::EXTERNAL);
    ItemPoolIter B = d.MakeRuntimeOutcastTensor(0xB2ull, RuntimeTensorMemProperty::EXTERNAL);
    EXPECT_EQ(d.runtimeOutcastTensorPool_.FreeItemNum(), devProg.runtimeOutcastPoolSize - 2);

    // assign B = A; this should deref old B (destroy it and return to pool) and ref A
    d.RuntimeOutcastTensorAssign(B, A);

    // After assign: one item returned to pool
    EXPECT_EQ(d.runtimeOutcastTensorPool_.FreeItemNum(), devProg.runtimeOutcastPoolSize - 1);

    // The target iterator B now refers to A, so address equals A's address and refCnt is 2
    EXPECT_EQ(d.GetRuntimeOutcastTensor(B).addr, static_cast<uintdevptr_t>(0xA1ull));
    EXPECT_EQ(d.GetRuntimeOutcastTensor(B).refCnt, 2u);
}

TEST_F(RuntimeOutcastTensorTest, SelfAssignDoesNotChangeRefCount) {
    std::vector<uint8_t> workspace(1u << 20); // 1MB

    DeviceWorkspaceAllocator d;
    DevAscendProgram devProg{};
    devProg.runtimeOutcastPoolSize = 4;

    InitDeviceWorkspaceAllocatorForTest(d, devProg, workspace);

    ItemPoolIter a = d.MakeRuntimeOutcastTensor(0xAAull, RuntimeTensorMemProperty::EXTERNAL);
    auto &t = d.GetRuntimeOutcastTensor(a);
    EXPECT_EQ(t.refCnt, 1u);

    // Self assign should not change refCnt (implementation has early return check)
    d.RuntimeOutcastTensorAssign(a, a);
    EXPECT_EQ(t.refCnt, 1u);
    EXPECT_EQ(t.addr, static_cast<uintdevptr_t>(0xAAull));

    // Verify pool state unchanged
    EXPECT_EQ(d.runtimeOutcastTensorPool_.FreeItemNum(), devProg.runtimeOutcastPoolSize - 1);
}

TEST_F(RuntimeOutcastTensorTest, DumpAllPropertyTypes) {
    RuntimeOutcastTensor t1(0x1234ull, RuntimeTensorMemProperty::EXTERNAL, 1u);
    EXPECT_EQ(t1.Dump(), std::string("&0x1234, EXTERNAL"));

    RuntimeOutcastTensor t2(0x5678ull, RuntimeTensorMemProperty::DEVTASK_INNER_OUTCAST, 1u);
    EXPECT_EQ(t2.Dump(), std::string("&0x5678, DEVTASK_INNER_OUTCAST"));

    RuntimeOutcastTensor t3(0x9ABCull, RuntimeTensorMemProperty::BOUNDARY_OUTCAST, 1u);
    EXPECT_EQ(t3.Dump(), std::string("&0x9abc, BOUNDARY_OUTCAST"));
}

TEST_F(RuntimeOutcastTensorTest, BoundaryOutcastDelayedRecycle) {
    std::vector<uint8_t> workspace(1u << 20); // 1MB

    DeviceWorkspaceAllocator d;
    DevAscendProgram devProg{};
    devProg.runtimeOutcastPoolSize = 4;
    // Set boundary outcast budget to allow allocation
    devProg.memBudget.tensor.devTaskBoundaryOutcastNum = 1;
    devProg.memBudget.tensor.maxStaticOutcastMem = 1024;

    InitDeviceWorkspaceAllocatorForTest(d, devProg, workspace);

    // Initially, the delayed recycle list should be empty
    EXPECT_EQ(d.rtBoundaryOutcastToBeFree_.size(), 0u);

    ItemPoolIter a = d.MakeRuntimeOutcastTensor(0x1000ull, RuntimeTensorMemProperty::BOUNDARY_OUTCAST);
    auto &t = d.GetRuntimeOutcastTensor(a);
    EXPECT_EQ(t.property, RuntimeTensorMemProperty::BOUNDARY_OUTCAST);
    EXPECT_EQ(t.addr, static_cast<uintdevptr_t>(0x1000ull));

    // Deref to 0 should add to delayed recycle list (for BOUNDARY_OUTCAST)
    // The item should be returned to pool
    d.RuntimeOutcastTensorDeref(a);
    EXPECT_EQ(d.runtimeOutcastTensorPool_.FreeItemNum(), devProg.runtimeOutcastPoolSize);

    // Verify BOUNDARY_OUTCAST tensor was added to delayed recycle list
    EXPECT_EQ(d.rtBoundaryOutcastToBeFree_.size(), 1u);
    EXPECT_EQ(d.rtBoundaryOutcastToBeFree_[0].addr, static_cast<uintdevptr_t>(0x1000ull));
    EXPECT_EQ(d.rtBoundaryOutcastToBeFree_[0].property, RuntimeTensorMemProperty::BOUNDARY_OUTCAST);

    // Create and destroy another BOUNDARY_OUTCAST tensor
    ItemPoolIter b = d.MakeRuntimeOutcastTensor(0x2000ull, RuntimeTensorMemProperty::BOUNDARY_OUTCAST);
    d.RuntimeOutcastTensorDeref(b);

    // Should have two items in the list now
    EXPECT_EQ(d.rtBoundaryOutcastToBeFree_.size(), 2u);
    EXPECT_EQ(d.rtBoundaryOutcastToBeFree_[1].addr, static_cast<uintdevptr_t>(0x2000ull));
    EXPECT_EQ(d.rtBoundaryOutcastToBeFree_[1].property, RuntimeTensorMemProperty::BOUNDARY_OUTCAST);
}

TEST_F(RuntimeOutcastTensorTest, AllPropertyTypesUsage) {
    std::vector<uint8_t> workspace(1u << 20); // 1MB

    DeviceWorkspaceAllocator d;
    DevAscendProgram devProg{};
    devProg.runtimeOutcastPoolSize = 8;

    InitDeviceWorkspaceAllocatorForTest(d, devProg, workspace);

    // Initially, the delayed recycle list should be empty
    EXPECT_EQ(d.rtBoundaryOutcastToBeFree_.size(), 0u);

    // Test EXTERNAL
    ItemPoolIter ext = d.MakeRuntimeOutcastTensor(0x1000ull, RuntimeTensorMemProperty::EXTERNAL);
    EXPECT_EQ(d.GetRuntimeOutcastTensor(ext).property, RuntimeTensorMemProperty::EXTERNAL);

    // Test DEVTASK_INNER_OUTCAST
    ItemPoolIter inner = d.MakeRuntimeOutcastTensor(0x2000ull, RuntimeTensorMemProperty::DEVTASK_INNER_OUTCAST);
    EXPECT_EQ(d.GetRuntimeOutcastTensor(inner).property, RuntimeTensorMemProperty::DEVTASK_INNER_OUTCAST);

    // Test BOUNDARY_OUTCAST
    ItemPoolIter boundary = d.MakeRuntimeOutcastTensor(0x3000ull, RuntimeTensorMemProperty::BOUNDARY_OUTCAST);
    EXPECT_EQ(d.GetRuntimeOutcastTensor(boundary).property, RuntimeTensorMemProperty::BOUNDARY_OUTCAST);

    // Verify all are tracked correctly
    EXPECT_EQ(d.runtimeOutcastTensorPool_.FreeItemNum(), devProg.runtimeOutcastPoolSize - 3);

    // Deref EXTERNAL - should NOT be added to delayed recycle list
    d.RuntimeOutcastTensorDeref(ext);
    EXPECT_EQ(d.rtBoundaryOutcastToBeFree_.size(), 0u);

    // Deref DEVTASK_INNER_OUTCAST - should NOT be added to delayed recycle list
    d.RuntimeOutcastTensorDeref(inner);
    EXPECT_EQ(d.rtBoundaryOutcastToBeFree_.size(), 0u);

    // Deref BOUNDARY_OUTCAST - SHOULD be added to delayed recycle list
    d.RuntimeOutcastTensorDeref(boundary);
    EXPECT_EQ(d.rtBoundaryOutcastToBeFree_.size(), 1u);
    EXPECT_EQ(d.rtBoundaryOutcastToBeFree_[0].addr, static_cast<uintdevptr_t>(0x3000ull));
    EXPECT_EQ(d.rtBoundaryOutcastToBeFree_[0].property, RuntimeTensorMemProperty::BOUNDARY_OUTCAST);
}

TEST_F(RuntimeOutcastTensorTest, AssignWithInvalidIndex) {
    std::vector<uint8_t> workspace(1u << 20); // 1MB

    DeviceWorkspaceAllocator d;
    DevAscendProgram devProg{};
    devProg.runtimeOutcastPoolSize = 4;

    InitDeviceWorkspaceAllocatorForTest(d, devProg, workspace);

    ItemPoolIter valid = d.MakeRuntimeOutcastTensor(0xAAull, RuntimeTensorMemProperty::EXTERNAL);
    ItemPoolIter invalid = ITEM_POOL_INVALID_INDEX;

    // Assign valid -> invalid should work (deref safe handles invalid)
    ItemPoolIter dst = valid;
    d.RuntimeOutcastTensorAssign(dst, invalid);
    EXPECT_EQ(dst, ITEM_POOL_INVALID_INDEX);

    // Re-create valid tensor
    valid = d.MakeRuntimeOutcastTensor(0xBBull, RuntimeTensorMemProperty::EXTERNAL);

    // Assign invalid -> valid should work
    dst = ITEM_POOL_INVALID_INDEX;
    d.RuntimeOutcastTensorAssign(dst, valid);
    EXPECT_EQ(dst, valid);
    EXPECT_EQ(d.GetRuntimeOutcastTensor(valid).refCnt, 2u); // original + assigned reference

    // Cleanup
    d.RuntimeOutcastTensorDeref(dst);
    d.RuntimeOutcastTensorDeref(valid);
}

TEST_F(RuntimeOutcastTensorTest, MultipleReplaceAddrWithoutRecycle) {
    std::vector<uint8_t> workspace(1u << 20); // 1MB

    DeviceWorkspaceAllocator d;
    DevAscendProgram devProg{};
    devProg.runtimeOutcastPoolSize = 4;

    InitDeviceWorkspaceAllocatorForTest(d, devProg, workspace);

    ItemPoolIter a = d.MakeRuntimeOutcastTensor(0x1000ull, RuntimeTensorMemProperty::EXTERNAL);
    auto &t = d.GetRuntimeOutcastTensor(a);
    EXPECT_EQ(t.addr, static_cast<uintdevptr_t>(0x1000ull));
    EXPECT_EQ(d.rtBoundaryOutcastToBeFree_.size(), 0u);

    // Multiple replacements
    d.RuntimeOutcastTensorReplaceAddrWithoutRecycle(a, 0x2000ull, RuntimeTensorMemProperty::BOUNDARY_OUTCAST);
    EXPECT_EQ(t.addr, static_cast<uintdevptr_t>(0x2000ull));
    EXPECT_EQ(t.property, RuntimeTensorMemProperty::BOUNDARY_OUTCAST);

    d.RuntimeOutcastTensorReplaceAddrWithoutRecycle(a, 0x3000ull, RuntimeTensorMemProperty::DEVTASK_INNER_OUTCAST);
    EXPECT_EQ(t.addr, static_cast<uintdevptr_t>(0x3000ull));
    EXPECT_EQ(t.property, RuntimeTensorMemProperty::DEVTASK_INNER_OUTCAST);

    // Replace back to BOUNDARY_OUTCAST before deref
    d.RuntimeOutcastTensorReplaceAddrWithoutRecycle(a, 0x4000ull, RuntimeTensorMemProperty::BOUNDARY_OUTCAST);
    EXPECT_EQ(t.addr, static_cast<uintdevptr_t>(0x4000ull));
    EXPECT_EQ(t.property, RuntimeTensorMemProperty::BOUNDARY_OUTCAST);

    // RefCnt should remain unchanged
    EXPECT_EQ(t.refCnt, 1u);

    // Deref when property is BOUNDARY_OUTCAST should add to delayed recycle list
    d.RuntimeOutcastTensorDeref(a);
    EXPECT_EQ(d.rtBoundaryOutcastToBeFree_.size(), 1u);
    EXPECT_EQ(d.rtBoundaryOutcastToBeFree_[0].addr, static_cast<uintdevptr_t>(0x4000ull));
    EXPECT_EQ(d.rtBoundaryOutcastToBeFree_[0].property, RuntimeTensorMemProperty::BOUNDARY_OUTCAST);
}

TEST_F(RuntimeOutcastTensorTest, GetRuntimeOutcastTensorPoolBase) {
    std::vector<uint8_t> workspace(1u << 20); // 1MB

    DeviceWorkspaceAllocator d;
    DevAscendProgram devProg{};
    devProg.runtimeOutcastPoolSize = 4;

    InitDeviceWorkspaceAllocatorForTest(d, devProg, workspace);

    auto *poolBase = d.GetRuntimeOutcastTensorPoolBase();
    EXPECT_NE(poolBase, nullptr);

    // Create some tensors and verify we can access them via pool base
    ItemPoolIter a = d.MakeRuntimeOutcastTensor(0xAAull, RuntimeTensorMemProperty::EXTERNAL);
    ItemPoolIter b = d.MakeRuntimeOutcastTensor(0xBBull, RuntimeTensorMemProperty::EXTERNAL);

    // Access via pool base using iterator index
    // Note: This tests the pool base is correctly set up for potential serialization use
    auto &t1 = d.GetRuntimeOutcastTensor(a);
    auto &t2 = d.GetRuntimeOutcastTensor(b);

    EXPECT_EQ(t1.addr, static_cast<uintdevptr_t>(0xAAull));
    EXPECT_EQ(t2.addr, static_cast<uintdevptr_t>(0xBBull));

    // Cleanup
    d.RuntimeOutcastTensorDeref(a);
    d.RuntimeOutcastTensorDeref(b);
}

TEST_F(RuntimeOutcastTensorTest, ComplexUsageSequence) {
    std::vector<uint8_t> workspace(1u << 20); // 1MB

    DeviceWorkspaceAllocator d;
    DevAscendProgram devProg{};
    devProg.runtimeOutcastPoolSize = 8;

    InitDeviceWorkspaceAllocatorForTest(d, devProg, workspace);

    EXPECT_EQ(d.rtBoundaryOutcastToBeFree_.size(), 0u);

    // Create multiple tensors
    ItemPoolIter a = d.MakeRuntimeOutcastTensor(0x1000ull, RuntimeTensorMemProperty::EXTERNAL);
    ItemPoolIter b = d.MakeRuntimeOutcastTensor(0x2000ull, RuntimeTensorMemProperty::EXTERNAL);
    ItemPoolIter c = d.MakeRuntimeOutcastTensor(0x3000ull, RuntimeTensorMemProperty::BOUNDARY_OUTCAST);

    // Ref operations
    d.RuntimeOutcastTensorRef(a);
    d.RuntimeOutcastTensorRef(a);
    EXPECT_EQ(d.GetRuntimeOutcastTensor(a).refCnt, 3u);

    // Replace addr on b
    d.RuntimeOutcastTensorReplaceAddrWithoutRecycle(b, 0x2500ull, RuntimeTensorMemProperty::DEVTASK_INNER_OUTCAST);
    EXPECT_EQ(d.GetRuntimeOutcastTensor(b).addr, static_cast<uintdevptr_t>(0x2500ull));

    // Assign operations - c now points to a (EXTERNAL), so original c (BOUNDARY_OUTCAST) is destroyed
    d.RuntimeOutcastTensorAssign(c, a);
    EXPECT_EQ(d.GetRuntimeOutcastTensor(c).addr, static_cast<uintdevptr_t>(0x1000ull));
    EXPECT_EQ(d.GetRuntimeOutcastTensor(a).refCnt, 4u); // a + 2 refs + c

    // Original c was BOUNDARY_OUTCAST and was destroyed during assign, should be in delayed recycle list
    EXPECT_EQ(d.rtBoundaryOutcastToBeFree_.size(), 1u);
    EXPECT_EQ(d.rtBoundaryOutcastToBeFree_[0].addr, static_cast<uintdevptr_t>(0x3000ull));
    EXPECT_EQ(d.rtBoundaryOutcastToBeFree_[0].property, RuntimeTensorMemProperty::BOUNDARY_OUTCAST);

    // Deref operations
    d.RuntimeOutcastTensorDeref(a);
    EXPECT_EQ(d.GetRuntimeOutcastTensor(a).refCnt, 3u);

    // Cleanup - deref all references
    d.RuntimeOutcastTensorDeref(a);
    d.RuntimeOutcastTensorDeref(a);
    d.RuntimeOutcastTensorDeref(b);
    d.RuntimeOutcastTensorDeref(c);

    // Verify all returned to pool
    EXPECT_EQ(d.runtimeOutcastTensorPool_.FreeItemNum(), devProg.runtimeOutcastPoolSize);

    // List should still have the one BOUNDARY_OUTCAST from earlier
    EXPECT_EQ(d.rtBoundaryOutcastToBeFree_.size(), 1u);
}

TEST_F(RuntimeOutcastTensorTest, PoolExhaustionEdgeCase) {
    std::vector<uint8_t> workspace(1u << 20); // 1MB

    DeviceWorkspaceAllocator d;
    DevAscendProgram devProg{};
    devProg.runtimeOutcastPoolSize = 2; // Small pool

    InitDeviceWorkspaceAllocatorForTest(d, devProg, workspace);

    // Fill up the pool
    ItemPoolIter a = d.MakeRuntimeOutcastTensor(0x1000ull, RuntimeTensorMemProperty::EXTERNAL);
    ItemPoolIter b = d.MakeRuntimeOutcastTensor(0x2000ull, RuntimeTensorMemProperty::EXTERNAL);
    EXPECT_EQ(d.runtimeOutcastTensorPool_.FreeItemNum(), 0u);

    // Free one to make space
    d.RuntimeOutcastTensorDeref(a);
    EXPECT_EQ(d.runtimeOutcastTensorPool_.FreeItemNum(), 1u);

    // Should be able to allocate again
    ItemPoolIter c = d.MakeRuntimeOutcastTensor(0x3000ull, RuntimeTensorMemProperty::EXTERNAL);
    EXPECT_NE(c, ITEM_POOL_INVALID_INDEX);
    EXPECT_EQ(d.runtimeOutcastTensorPool_.FreeItemNum(), 0u);

    // Cleanup
    d.RuntimeOutcastTensorDeref(b);
    d.RuntimeOutcastTensorDeref(c);
}

TEST_F(RuntimeOutcastTensorTest, RefCountMultipleRefsAndDerefs) {
    std::vector<uint8_t> workspace(1u << 20); // 1MB

    DeviceWorkspaceAllocator d;
    DevAscendProgram devProg{};
    devProg.runtimeOutcastPoolSize = 4;

    InitDeviceWorkspaceAllocatorForTest(d, devProg, workspace);

    ItemPoolIter a = d.MakeRuntimeOutcastTensor(0x1000ull, RuntimeTensorMemProperty::EXTERNAL);
    auto &t = d.GetRuntimeOutcastTensor(a);

    // Multiple refs
    for (uint32_t i = 0; i < 10; ++i) {
        d.RuntimeOutcastTensorRef(a);
    }
    EXPECT_EQ(t.refCnt, 11u);

    // Multiple derefs
    for (uint32_t i = 0; i < 10; ++i) {
        d.RuntimeOutcastTensorDeref(a);
    }
    EXPECT_EQ(t.refCnt, 1u);

    // Final deref should return to pool
    d.RuntimeOutcastTensorDeref(a);
    EXPECT_EQ(d.runtimeOutcastTensorPool_.FreeItemNum(), devProg.runtimeOutcastPoolSize);
}

TEST_F(RuntimeOutcastTensorTest, AssignToSelfWithMultipleRefs) {
    std::vector<uint8_t> workspace(1u << 20); // 1MB

    DeviceWorkspaceAllocator d;
    DevAscendProgram devProg{};
    devProg.runtimeOutcastPoolSize = 4;

    InitDeviceWorkspaceAllocatorForTest(d, devProg, workspace);

    ItemPoolIter a = d.MakeRuntimeOutcastTensor(0x1000ull, RuntimeTensorMemProperty::EXTERNAL);
    d.RuntimeOutcastTensorRef(a);
    d.RuntimeOutcastTensorRef(a);
    auto &t = d.GetRuntimeOutcastTensor(a);
    EXPECT_EQ(t.refCnt, 3u);

    // Self assign with multiple refs should not change refCnt
    d.RuntimeOutcastTensorAssign(a, a);
    EXPECT_EQ(t.refCnt, 3u);
    EXPECT_EQ(t.addr, static_cast<uintdevptr_t>(0x1000ull));

    // Cleanup
    d.RuntimeOutcastTensorDeref(a);
    d.RuntimeOutcastTensorDeref(a);
    d.RuntimeOutcastTensorDeref(a);
}

TEST_F(RuntimeOutcastTensorTest, MixedSafeAndUnsafeOperations) {
    std::vector<uint8_t> workspace(1u << 20); // 1MB

    DeviceWorkspaceAllocator d;
    DevAscendProgram devProg{};
    devProg.runtimeOutcastPoolSize = 4;

    InitDeviceWorkspaceAllocatorForTest(d, devProg, workspace);

    ItemPoolIter valid = d.MakeRuntimeOutcastTensor(0x1000ull, RuntimeTensorMemProperty::EXTERNAL);
    ItemPoolIter invalid = ITEM_POOL_INVALID_INDEX;

    // Mix safe and unsafe operations
    d.RuntimeOutcastTensorRef(valid); // unsafe, should work
    d.RuntimeOutcastTensorRefSafe(invalid); // safe, should not crash
    d.RuntimeOutcastTensorDerefSafe(invalid); // safe, should not crash
    d.RuntimeOutcastTensorDeref(valid); // unsafe, should work

    auto &t = d.GetRuntimeOutcastTensor(valid);
    EXPECT_EQ(t.refCnt, 1u);

    // Cleanup
    d.RuntimeOutcastTensorDeref(valid);
}

TEST_F(RuntimeOutcastTensorTest, DelayedRecycleListBehavior) {
    std::vector<uint8_t> workspace(1u << 20); // 1MB

    DeviceWorkspaceAllocator d;
    DevAscendProgram devProg{};
    devProg.runtimeOutcastPoolSize = 8;

    InitDeviceWorkspaceAllocatorForTest(d, devProg, workspace);

    // Initially empty
    EXPECT_EQ(d.rtBoundaryOutcastToBeFree_.size(), 0u);

    // Create different property types and verify only BOUNDARY_OUTCAST is added
    ItemPoolIter ext1 = d.MakeRuntimeOutcastTensor(0x1000ull, RuntimeTensorMemProperty::EXTERNAL);
    ItemPoolIter inner1 = d.MakeRuntimeOutcastTensor(0x2000ull, RuntimeTensorMemProperty::DEVTASK_INNER_OUTCAST);
    ItemPoolIter boundary1 = d.MakeRuntimeOutcastTensor(0x3000ull, RuntimeTensorMemProperty::BOUNDARY_OUTCAST);

    // Destroy EXTERNAL - should NOT be added
    d.RuntimeOutcastTensorDeref(ext1);
    EXPECT_EQ(d.rtBoundaryOutcastToBeFree_.size(), 0u);

    // Destroy DEVTASK_INNER_OUTCAST - should NOT be added
    d.RuntimeOutcastTensorDeref(inner1);
    EXPECT_EQ(d.rtBoundaryOutcastToBeFree_.size(), 0u);

    // Destroy BOUNDARY_OUTCAST - SHOULD be added
    d.RuntimeOutcastTensorDeref(boundary1);
    EXPECT_EQ(d.rtBoundaryOutcastToBeFree_.size(), 1u);
    EXPECT_EQ(d.rtBoundaryOutcastToBeFree_[0].addr, static_cast<uintdevptr_t>(0x3000ull));

    // Create more BOUNDARY_OUTCAST tensors
    ItemPoolIter boundary2 = d.MakeRuntimeOutcastTensor(0x4000ull, RuntimeTensorMemProperty::BOUNDARY_OUTCAST);
    ItemPoolIter boundary3 = d.MakeRuntimeOutcastTensor(0x5000ull, RuntimeTensorMemProperty::BOUNDARY_OUTCAST);

    // Replace property before destroy - if changed to non-BOUNDARY_OUTCAST, should NOT be added
    d.RuntimeOutcastTensorReplaceAddrWithoutRecycle(boundary2, 0x4000ull, RuntimeTensorMemProperty::EXTERNAL);
    d.RuntimeOutcastTensorDeref(boundary2);
    // Should still have only 1 (boundary1), boundary2 was EXTERNAL when destroyed
    EXPECT_EQ(d.rtBoundaryOutcastToBeFree_.size(), 1u);

    // Destroy boundary3 which is still BOUNDARY_OUTCAST
    d.RuntimeOutcastTensorDeref(boundary3);
    // Should now have 2 items
    EXPECT_EQ(d.rtBoundaryOutcastToBeFree_.size(), 2u);
    EXPECT_EQ(d.rtBoundaryOutcastToBeFree_[0].addr, static_cast<uintdevptr_t>(0x3000ull));
    EXPECT_EQ(d.rtBoundaryOutcastToBeFree_[1].addr, static_cast<uintdevptr_t>(0x5000ull));
    EXPECT_EQ(d.rtBoundaryOutcastToBeFree_[1].property, RuntimeTensorMemProperty::BOUNDARY_OUTCAST);

    // Create and replace to BOUNDARY_OUTCAST, then destroy
    ItemPoolIter ext2 = d.MakeRuntimeOutcastTensor(0x6000ull, RuntimeTensorMemProperty::EXTERNAL);
    d.RuntimeOutcastTensorReplaceAddrWithoutRecycle(ext2, 0x6000ull, RuntimeTensorMemProperty::BOUNDARY_OUTCAST);
    d.RuntimeOutcastTensorDeref(ext2);
    // Should now have 3 items
    EXPECT_EQ(d.rtBoundaryOutcastToBeFree_.size(), 3u);
    EXPECT_EQ(d.rtBoundaryOutcastToBeFree_[2].addr, static_cast<uintdevptr_t>(0x6000ull));
    EXPECT_EQ(d.rtBoundaryOutcastToBeFree_[2].property, RuntimeTensorMemProperty::BOUNDARY_OUTCAST);
}


