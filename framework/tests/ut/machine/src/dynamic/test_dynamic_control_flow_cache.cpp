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
 * \file test_dynamic_control_flow_cache.cpp
 * \brief
 */

#include "test_machine_common.h"

struct DynamicControlFlowCacheTest : UnitTestBase {};

static DeviceTensorData toTensorData(const std::shared_ptr<LogicalTensor>& t)
{
    return DeviceTensorData(t->Datatype(), nullptr, t->GetShape());
}

TEST_F(DynamicControlFlowCacheTest, KernelReuse)
{
    int tiling = 32;
    TileShape::Current().SetVecTile(tiling, tiling);

    int n = tiling * 4;
    Tensor inputA(DT_INT32, {n, n}, "A");
    Tensor inputB(DT_INT32, {n, n}, "B");
    Tensor output(DT_INT32, {n, n}, "O");

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateConstantTensor<int32_t>(inputA, 1),
        RawTensorData::CreateConstantTensor<int32_t>(inputB, 2),
    });
    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<int32_t>(output, 0),
    });

    Tensor e;
    FUNCTION("main", {inputA, inputB}, {output})
    {
        Tensor sum(DT_INT32, {n, n}, "sum");
        LOOP("L0", FunctionType::DYNAMIC_LOOP, i, LoopRange(GetInputShape(inputA, 0) / tiling))
        {
            LOOP("L1", FunctionType::DYNAMIC_LOOP, j, LoopRange(GetInputShape(inputA, 1) / tiling))
            {
                auto a = View(inputA, {tiling, tiling}, std::vector<SymbolicScalar>({i * tiling, j * tiling}));
                auto b = View(inputB, {tiling, tiling}, std::vector<SymbolicScalar>({i * tiling, j * tiling}));
                Assemble(Add(a, b), {i * tiling, j * tiling}, sum);
            }
        }
        LOOP("X", FunctionType::DYNAMIC_LOOP, _, LoopRange(1))
        {
            (void)_;
            output = Add(sum, sum);
        }
    }
    DeviceLauncherConfig config;
    config.blockdim = 24; // 24:max aicore num
    DevControlFlowCache* ctrlFlowCache = nullptr;
    EmulationMemoryUtils memUtils;
    EXPECT_EQ(0, EmulationLauncher::BuildControlFlowCache(Program::GetInstance().GetLastFunction(), memUtils, {}, {},
                                                          &ctrlFlowCache, config));

    DeviceLauncher::SetDevRunCacheKernelEnable(Program::GetInstance().GetLastFunction(), true);

    for (int k = 0; k < 3; k++) {
        EXPECT_EQ(0,
                  EmulationLauncher::EmulationRunOnce(Program::GetInstance().GetLastFunction(), ctrlFlowCache, config));
    }
}

TEST_F(DynamicControlFlowCacheTest, PartialCache)
{
    // cache at most 3 task
    // every task 4 root func
    config::SetRuntimeOption<int64_t>(STITCH_FUNCTION_MAX_NUM, 0x4);

    int tiling = 32;
    int n = tiling * 4;
    TileShape::Current().SetVecTile(tiling, tiling);

    Tensor inputA(DT_INT32, {n, n}, "A");
    Tensor inputB(DT_INT32, {n, n}, "B");
    Tensor output(DT_INT32, {n, n}, "O");

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateConstantTensor<int32_t>(inputA, 1),
        RawTensorData::CreateConstantTensor<int32_t>(inputB, 2),
    });
    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<int32_t>(output, 0),
    });

    // 17 root func in total
    FUNCTION("main", {inputA, inputB}, {output})
    {
        Tensor sum(DT_INT32, {n, n}, "sum");
        LOOP("L0", FunctionType::DYNAMIC_LOOP, i, LoopRange(GetInputShape(inputA, 0) / tiling))
        {
            LOOP("L1", FunctionType::DYNAMIC_LOOP, j, LoopRange(GetInputShape(inputA, 1) / tiling))
            {
                auto a = View(inputA, {tiling, tiling}, std::vector<SymbolicScalar>({i * tiling, j * tiling}));
                auto b = View(inputB, {tiling, tiling}, std::vector<SymbolicScalar>({i * tiling, j * tiling}));
                Assemble(Add(a, b), {i * tiling, j * tiling}, sum);
            }
        }
        LOOP("Use", FunctionType::DYNAMIC_LOOP, _, LoopRange(1))
        {
            (void)_;
            output = Add(sum, sum);
        }
    }

    std::vector<DeviceTensorData> inputList = {toTensorData(inputA.GetStorage()), toTensorData(inputB.GetStorage())};
    std::vector<DeviceTensorData> outputList = {toTensorData(output.GetStorage())};
    DeviceLauncherConfig config;
    config.blockdim = 24; // 24:max aicore num
    DevControlFlowCache* ctrlFlowCache = nullptr;
    EmulationMemoryUtils memUtils;
    EXPECT_EQ(0, EmulationLauncher::BuildControlFlowCache(Program::GetInstance().GetLastFunction(), memUtils, inputList,
                                                          outputList, &ctrlFlowCache, config));
    DevAscendProgram* devProg = DeviceLauncher::GetDevProg(Program::GetInstance().GetLastFunction());

    EXPECT_EQ(0x5, ctrlFlowCache->deviceTaskCount);
    EXPECT_EQ(0x0, ctrlFlowCache->deviceTaskSkippedCount);

    devProg->RelocProgram(0, (intptr_t)devProg);
    ctrlFlowCache->RelocMetaCache(0, (intptr_t)ctrlFlowCache);
    ctrlFlowCache->TaskAddrRelocProgramAndCtrlCache(0, 0, (intptr_t)devProg, (intptr_t)ctrlFlowCache);

    for (int i = 0; i < 0x4; i++) {
        auto dynTaskBase = ctrlFlowCache->deviceTaskCacheList[i].dynTaskBase;
        EXPECT_EQ(0x4, dynTaskBase->GetDynFuncDataList()->Size());
    }
    auto dynTaskBase = ctrlFlowCache->deviceTaskCacheList[0x4].dynTaskBase;
    EXPECT_EQ(0x1, dynTaskBase->GetDynFuncDataList()->Size());
    ctrlFlowCache->TaskAddrRelocProgramAndCtrlCache((intptr_t)devProg, (intptr_t)ctrlFlowCache, 0, 0);
    devProg->RelocProgram((intptr_t)devProg, 0);
    ctrlFlowCache->RelocMetaCache((intptr_t)ctrlFlowCache, 0);
    EXPECT_EQ(false, ctrlFlowCache->isRelocDataDev);
    EXPECT_EQ(false, ctrlFlowCache->isRelocMetaDev);
    EXPECT_EQ(true, ctrlFlowCache->isActivated);
    DeviceLauncher::SetDevRunCacheKernelEnable(Program::GetInstance().GetLastFunction(), true);

    EXPECT_EQ(0, EmulationLauncher::EmulationRunOnce(Program::GetInstance().GetLastFunction(), ctrlFlowCache, config));
}
