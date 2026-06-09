/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_op_info_manager.cpp
 * \brief Unit tests for OpInfoManager singleton (tiling keys, op type, control buffer, bin handles)
 */

#include <cstdint>
#include <string>
#include <vector>
#include "gtest/gtest.h"
#include "interface/utils/op_info_manager.h"

using namespace npu::tile_fwk;

class TestOpInfoManager : public testing::Test {
public:
    static void SetUpTestCase() {}
    static void TearDownTestCase() {}
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(TestOpInfoManager, SetAndGetOpTilingKey)
{
    auto& mgr = OpInfoManager::GetInstance();
    uint64_t key = 0x123456789ABCDULL;
    mgr.SetOpTilingKey(key);
    uint64_t result = mgr.GetOpTilingKey();
    EXPECT_EQ(result, key & MAIN_KEY_MASK);
}

TEST_F(TestOpInfoManager, SetOpTilingKeyMasked)
{
    auto& mgr = OpInfoManager::GetInstance();
    uint64_t keyWithSub = 0xFFF123456789ABCDULL;
    mgr.SetOpTilingKey(keyWithSub);
    uint64_t result = mgr.GetOpTilingKey();
    EXPECT_EQ(result, keyWithSub & MAIN_KEY_MASK);
}

TEST_F(TestOpInfoManager, GetNewSubTilingKeyIncrementing)
{
    auto& mgr = OpInfoManager::GetInstance();
    mgr.SetOpTilingKey(100);
    uint64_t k1 = mgr.GetNewSubTilingKey();
    uint64_t k2 = mgr.GetNewSubTilingKey();
    EXPECT_NE(k1, k2);
    uint64_t mainKey = 100 & MAIN_KEY_MASK;
    EXPECT_EQ(k1 & MAIN_KEY_MASK, mainKey);
    EXPECT_EQ(k2 & MAIN_KEY_MASK, mainKey);
}

TEST_F(TestOpInfoManager, GetCurSubTilingKeyNoIncrement)
{
    auto& mgr = OpInfoManager::GetInstance();
    mgr.SetOpTilingKey(200);
    uint64_t k1 = mgr.GetCurSubTilingKey();
    uint64_t k2 = mgr.GetCurSubTilingKey();
    EXPECT_EQ(k1, k2);
}

TEST_F(TestOpInfoManager, SetAndGetOpType)
{
    auto& mgr = OpInfoManager::GetInstance();
    mgr.SetOpType("custom_op");
    EXPECT_EQ(mgr.GetOpType(), "custom_op");
    mgr.SetOpType("tilefwk");
    EXPECT_EQ(mgr.GetOpType(), "tilefwk");
}

TEST_F(TestOpInfoManager, GetControlBufferDefault)
{
    auto& mgr = OpInfoManager::GetInstance();
    auto& buf = mgr.GetControlBuffer();
    EXPECT_EQ(buf.size(), 1u);
    EXPECT_EQ(buf[0], '0');
}

TEST_F(TestOpInfoManager, GetCustomJsonDefault)
{
    auto& mgr = OpInfoManager::GetInstance();
    auto& json = mgr.GetCustomJson();
    EXPECT_EQ(json.size(), 1u);
    EXPECT_EQ(json[0], '0');
}

TEST_F(TestOpInfoManager, GetCustomOpJsonPathDefault)
{
    auto& mgr = OpInfoManager::GetInstance();
    std::string& path = mgr.GetCustomOpJsonPath();
    EXPECT_EQ(path, "");
}

TEST_F(TestOpInfoManager, SetAndGetOpFuncName)
{
    auto& mgr = OpInfoManager::GetInstance();
    mgr.GetOpFuncName() = "my_func";
    EXPECT_EQ(mgr.GetOpFuncName(), "my_func");
    mgr.GetOpFuncName() = "";
}

TEST_F(TestOpInfoManager, GetControlBinHandleNotFound)
{
    auto& mgr = OpInfoManager::GetInstance();
    void* result = mgr.GetControlBinHandle("/nonexistent/path.so");
    EXPECT_EQ(result, nullptr);
}

TEST_F(TestOpInfoManager, SetAndGetControlBinHandle)
{
    auto& mgr = OpInfoManager::GetInstance();
    mgr.GetCustomOpJsonPath() = "/test/path.so";
    int dummy = 0;
    void* handle = &dummy;
    mgr.SetControlBinHandle(handle);
    void* result = mgr.GetControlBinHandle("/test/path.so");
    EXPECT_NE(result, nullptr);
    mgr.GetCustomOpJsonPath() = "";
}

TEST_F(TestOpInfoManager, SubTilingKeyComposition)
{
    auto& mgr = OpInfoManager::GetInstance();
    mgr.SetOpTilingKey(0xABC);
    uint64_t composite = mgr.GetNewSubTilingKey();
    uint64_t mainPart = composite & MAIN_KEY_MASK;
    uint64_t subPart = (composite & SUB_KEY_MASK) >> SUB_KEY_OFFSET;
    EXPECT_EQ(mainPart, 0xABC & MAIN_KEY_MASK);
    EXPECT_GE(subPart, 1u);
}
