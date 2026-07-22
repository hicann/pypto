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
#include "machine/runtime/launcher/cell_match_dynamic.h"
#include "machine/runtime/launcher/device_launcher_binding.h"
#include "machine/runtime/launcher/aicore_model_launcher.h"

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
