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
 * \file test_load_aicpu_op.cpp
 * \brief UT for machine/runtime/runner/load_aicpu_op.cpp
 */

#include <gtest/gtest.h>
#define private public
#include "machine/runtime/runner/load_aicpu_op.h"
#undef private
#include "tilefwk/aicpu_common.h"

using namespace npu::tile_fwk;

TEST(LoadAicpuOpTest, AllLaunchOps_ReturnZeroWithoutCann)
{
    DeviceKernelArgs kArgs{};
    EXPECT_EQ(LoadAicpuOp::AicpuKernelLaunch(nullptr, nullptr, &kArgs, 1), 0);

    LoadAicpuOp op;
    EXPECT_EQ(op.LaunchBuiltInOp(nullptr, &kArgs, 1, "PyptoRun"), 0);

    std::string opType = "test";
    EXPECT_EQ(op.LaunchCustomOp(nullptr, &kArgs, opType), 0);

    EXPECT_EQ(op.GetBuiltInOpBinHandle(), 0);

    EXPECT_EQ(op.LaunchPyptoNullOp(nullptr, &kArgs, 1), 0);
    EXPECT_EQ(op.LaunchPyptoNullOp(nullptr, &kArgs, 1), 0);
}

TEST(LoadAicpuOpTest, GenBuiltInOpInfoAndCustomSoLoad_NoCrash)
{
    LoadAicpuOp op;
    op.GenBuiltInOpInfo();
    op.CustomAiCpuSoLoad();
    SUCCEED();
}
