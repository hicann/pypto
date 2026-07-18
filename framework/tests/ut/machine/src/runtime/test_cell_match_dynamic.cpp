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
 * \file test_cell_match_dynamic.cpp
 * \brief UT for machine/runtime/launcher/cell_match_dynamic.cpp
 */

#include <gtest/gtest.h>
#include "machine/runtime/launcher/cell_match_dynamic.h"
#include "machine/runtime/launcher/device_launcher_binding.h"

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

    SUCCEED();
}

TEST(CellMatchDynamicFuncTest, PrepareAndRefill_WithEvaluator)
{
    DyndevFunctionAttribute dynAttr;
    std::unordered_map<std::string, ScalarImmediateType> symbolDict;
    std::vector<DeviceTensorData> inputs;
    std::vector<DeviceTensorData> outputs;
    Evaluator eval(symbolDict, &inputs, &outputs);

    RefillDynamicMemBudgets(nullptr, dynAttr, eval);

    auto patches = PrepareDynamicCellMatchDescPatches(dynAttr, eval);
    EXPECT_TRUE(patches.empty());

    auto patches2 = PrepareHostDynamicCellMatchForLaunch(dynAttr, eval, nullptr);
    EXPECT_TRUE(patches2.empty());
}
