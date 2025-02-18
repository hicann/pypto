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
 * \file test_control_flow.cpp
 * \brief
 */

#include "interface/utils/string_utils.h"

#include "test_machine_common.h"

struct ControlFlowTest : UnitTestBase {};

std::string GetDeclName(const std::string &name) {
    std::vector<std::string> descList = StringUtils::Split(name, "_");
    return descList[1];
}

TEST_F(ControlFlowTest, RunDeviceContext) {
    config::SetRuntimeOption<int64_t>(STITCH_FUNCTION_NUM_INITIAL, 0x4);
    config::SetRuntimeOption<int64_t>(STITCH_FUNCTION_NUM_STEP, 0);

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

    FUNCTION("main", {inputA, inputB}, {output}) {
        LOOP("s0", FunctionType::DYNAMIC_LOOP, _, LoopRange(0x1)) {
            (void)_;
            output = Add(inputA, inputB);
        }
        LOOP("s1", FunctionType::DYNAMIC_LOOP, _, LoopRange(0x40 - 0x1)) {
            (void)_;
            output = Add(output, inputB);
        }
    }

    struct Inspector {
        int count{0};
        std::vector<DevAscendFunction *> rootList;
        static void Entry(void *inspector_, DeviceExecuteContext *execCtx, DynDeviceTask *task) {
            Inspector *inspector = reinterpret_cast<Inspector *>(inspector_);
            (void)execCtx; (void)task;
            inspector->count++;
            DynFuncDataCache *cacheList = task->GetDynFuncDataCacheList();
            for (size_t k = 0; k < task->dynFuncDataCacheListSize; k++) {
                inspector->rootList.push_back(cacheList->At(k).devFunc);
            }
        }
    };
    Inspector inspector;
    PyptoKernelCtrlServerRegisterTaskInspector(Inspector::Entry, &inspector);

    DeviceLauncherConfig config;
    config.blockdim = 24; // 24: max blockdim
    EXPECT_EQ(0, EmulationLauncher::EmulationRunOnce(Program::GetInstance().GetLastFunction(), config));
    EXPECT_EQ(0x10, inspector.count);
    EXPECT_EQ(0x40, inspector.rootList.size());
    EXPECT_EQ("s0", GetDeclName(inspector.rootList[0]->GetRawName()));
    for (size_t k = 1; k < 0x40; k++) {
        EXPECT_EQ("s1", GetDeclName(inspector.rootList[k]->GetRawName()));
    }
}

TEST_F(ControlFlowTest, TestDD) {
    TileShape::Current().SetVecTile(32, 32);
    TileShape::Current().SetCubeTile({32, 32}, {32, 32}, {32, 32});
    int s = 32;
    int n = 8;
    Tensor t0(DT_FP32, {n * s, s}, "t0");  // [32*8, 32]
    Tensor t1(DT_FP32, {s, s}, "t1");  // [32, 32]
    Tensor blockTable{DT_INT32, {n, 1}, "blockTable"};
    Tensor out(DT_FP32, {n * s, s}, "out");

    std::vector<int> tblData;
    for (int i = 0; i < n; i++)
        tblData.push_back(i);

    std::vector<float> golden(n * s * s, 128.0f);

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateConstantTensor<float>(t0, 1.0),
        RawTensorData::CreateConstantTensor<float>(t1, 2.0),
        RawTensorData::CreateTensor<int>(blockTable, tblData),  // value: [0,1,2,...,7]
    });
    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<float>(out, 0.0f),
    });
    ProgramData::GetInstance().AppendGoldens({
        RawTensorData::CreateTensor<float>(out, golden),
    });

    FUNCTION("main", {t0, t1, blockTable}, {out}) {
        LOOP("L0", FunctionType::DYNAMIC_LOOP, i, LoopRange(GetInputShape(t0, 0) / s)) {
            SymbolicScalar idx = GetTensorData(blockTable, {i, 0});
            Tensor t0s = View(t0, {s, s}, {idx * s, 0});

            Tensor qi(DT_FP32, {s, 2*s}, "qi");
            Assemble(t1, {0, 0}, qi);
            Assemble(t0s, {0, s}, qi);

            Tensor ki(DT_FP32, {s, 2*s}, "ki");
            Assemble(t0s, {0, 0}, ki);
            Assemble(t1, {0, s}, ki);

            Tensor t2 = Matrix::Matmul<false, true>(DataType::DT_FP32, qi, ki);
            // conat((t0s + t1, t1)) @ concat (t0s, t1)^T
            Assemble(t2, {idx * s, 0}, out);
        }
    }

    DeviceLauncherConfig config;
    config.blockdim = 25;
    EXPECT_EQ(0, EmulationLauncher::EmulationRunOnce(Program::GetInstance().GetLastFunction(), config));
}

TEST_F(ControlFlowTest, TensorRecycleDestruct) {
    config::SetRuntimeOption<int64_t>(STITCH_FUNCTION_NUM_INITIAL, 100);
    config::SetRuntimeOption<int64_t>(STITCH_FUNCTION_NUM_STEP, 0);

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

    FUNCTION("main", {inputA, inputB}, {output}) {
        LOOP("L0", FunctionType::DYNAMIC_LOOP, k, LoopRange(0x2)) {
            Tensor mid(DT_INT32, {n, n}, "O");
            LOOP("s0", FunctionType::DYNAMIC_LOOP, i, LoopRange(0x4)) {
                LOOP("s1", FunctionType::DYNAMIC_LOOP, j, LoopRange(0x4)) {
                    Tensor t0 = View(inputA, {tiling, tiling}, {i * tiling, j * tiling});
                    Tensor t1 = View(inputB, {tiling, tiling}, {i * tiling, j * tiling});
                    Tensor ts = Add(t0, t1);
                    Assemble(ts, {i * tiling, j * tiling}, mid);
                }
            }
            LOOP("sum", FunctionType::DYNAMIC_LOOP, _, LoopRange(0x1)) {
                (void)_;
                IF (k == 0) {
                    output = Add(mid, Element(DT_INT32, 0));
                } ELSE {
                    output = Add(output, mid);
                }
            }
        }
    }

    struct Inspector {
        std::vector<DynDeviceTask *> taskList;

        static void Entry(void *inspector_, DeviceExecuteContext *execCtx, DynDeviceTask *task) {
            (void)execCtx;
            Inspector *inspector = reinterpret_cast<Inspector *>(inspector_);
            inspector->taskList.push_back(task);
        }
    };
    Inspector inspector;
    PyptoKernelCtrlServerRegisterTaskInspector(Inspector::Entry, &inspector);

    DeviceLauncherConfig config;
    config.blockdim = 25;
    EXPECT_EQ(0, EmulationLauncher::EmulationRunOnce(Program::GetInstance().GetLastFunction(), config));
    EXPECT_EQ(1, inspector.taskList.size());

    DynDeviceTask *task = inspector.taskList[0];
    DynFuncDataCache *cacheList = task->GetDynFuncDataCacheList();
    DevAscendFunctionDuppedData *dup0 = cacheList->At(0).duppedData;
    DevAscendFunctionDuppedData *dup1 = cacheList->At(0x4 * 0x4 + 0x1).duppedData;
    EXPECT_TRUE(dup0->GetOutcastAddress(0).IsAddress());
    EXPECT_TRUE(dup1->GetOutcastAddress(0).IsAddress());
    EXPECT_NE(dup0->GetOutcastAddress(0).GetAddressValue(), dup1->GetOutcastAddress(0).GetAddressValue());
}