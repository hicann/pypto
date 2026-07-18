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
 * \file test_runtime_utils.cpp
 * \brief
 */

#include <gtest/gtest.h>
#include <unistd.h>
#include "machine/runtime/runner/runtime_utils.h"

using namespace npu::tile_fwk;

TEST(RuntimeUtilsTest, AlignSize_DefaultAlign512)
{
    EXPECT_EQ(AlignSize(0), 0u);
    EXPECT_EQ(AlignSize(1), 512u);
    EXPECT_EQ(AlignSize(512), 512u);
    EXPECT_EQ(AlignSize(513), 1024u);
    EXPECT_EQ(AlignSize(1024), 1024u);
}

TEST(RuntimeUtilsTest, AlignSize_CustomAlign)
{
    EXPECT_EQ(AlignSize(1, 64), 64u);
    EXPECT_EQ(AlignSize(64, 64), 64u);
    EXPECT_EQ(AlignSize(65, 64), 128u);
}

TEST(RuntimeUtilsTest, AlignSize_ZeroAlign_FallbackToPointerSize)
{
    auto result = AlignSize(1, 0);
    EXPECT_TRUE(result > 0u);
}

TEST(RuntimeUtilsTest, GetProcessId_FallbackToGetpid)
{
    uint32_t pid = GetProcessId();
    EXPECT_EQ(pid, static_cast<uint32_t>(getpid()));
}

TEST(RuntimeUtilsTest, RegisterKernelBin_EmptyInput_ThrowsError)
{
    std::vector<uint8_t> empty;
    EXPECT_ANY_THROW(RegisterKernelBin(empty));
}
