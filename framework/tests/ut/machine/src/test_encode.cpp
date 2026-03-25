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
 * \file test_encode.cpp
 * \brief
 */

#include <gtest/gtest.h>
#include <cstdint>

#include "machine/utils/dynamic/dev_encode.h"
#include "tilefwk/data_type.h"
#include "tilefwk/tilefwk_op.h"
#include "interface/configs/config_manager.h"
#include "interface/configs/config_manager.h"
#include "interface/program/program.h"

using namespace npu::tile_fwk;
using namespace npu::tile_fwk::dynamic;

class TestDevEncode : public testing::Test {};

TEST_F(TestDevEncode, DevSymShape) {
    DevSymShape shape;
    shape.SetShape({SymInt(true, 0), SymInt(true, 2), SymInt(2)}); // 4, 8, 2
    uint64_t exprTbl[] = {4, 6, 8};
    uint64_t strides[3] = {0};
    shape.ToStride(strides, exprTbl);
    EXPECT_EQ(strides[0], 16);
    EXPECT_EQ(strides[1], 2);
    EXPECT_EQ(strides[2], 1);
}

TEST_F(TestDevEncode, test_dev_encode_program) {
    config::SetRuntimeOption(STITCH_FUNCTION_MAX_NUM, 64);
    config::SetPlatformConfig(KEY_ENABLE_AIHAC_BACKEND, true);
    TileShape::Current().SetVecTile(32, 32);
    TileShape::Current().SetCubeTile({32, 32}, {32, 32}, {32, 32});
    constexpr int LOOP_COUNT_INNER = 4;
    int s = 32;
    Tensor t0(DT_FP32, {s, s}, "t0");
    Tensor t1(DT_FP32, {s, s}, "t1");
    Tensor t2(DT_FP32, {s, s}, "t2");
    Tensor out(DT_FP32, {LOOP_COUNT_INNER * s, s}, "out");

    //clc
    FUNCTION("main_LoopUnroll2", {t0, t1, t2}, {out}) {
        LOOP("main_LoopUnroll2_L0", FunctionType::DYNAMIC_LOOP, i, LoopRange(LOOP_COUNT_INNER)) {
            auto temp = Add(t0, t0);
            SymbolicScalar s_min = std::ternary(i < 2, i, i + 1);

            IF(s_min == i){
                temp = Add(temp, t1);
            }
            ELSE IF(s_min == i + 1){
                temp = Add(temp, t2);
            }
            Assemble(temp, {i * s, 0}, out);
        }
    }

    std::shared_ptr<DyndevFunctionAttribute> funcDynDev = Program::GetInstance().GetLastFunction()->GetDyndevAttribute();
    ASSERT_NE(funcDynDev, nullptr);
    DevAscendProgram *devProg = reinterpret_cast<DevAscendProgram *>(funcDynDev->devProgBinary.data());
    ASSERT_NE(devProg, nullptr);
    EXPECT_EQ(devProg->stitchFunctionNumInitial, 64);
    devProg->RelocProgram(0, reinterpret_cast<uint64_t>(devProg), true);
    devProg->controlFlowCache.isRecording = false;
    uint64_t contextWorkspaceAddr = devProg->controlFlowCache.contextWorkspaceAddr;
    devProg->controlFlowCache.IncastOutcastAddrReloc(contextWorkspaceAddr, 0, nullptr);
    devProg->controlFlowCache.RuntimeAddrRelocWorkspace(contextWorkspaceAddr, 0, nullptr, nullptr, nullptr);
    devProg->controlFlowCache.RuntimeAddrRelocProgram(reinterpret_cast<uint64_t>(devProg), 0);
    devProg->controlFlowCache.TaskAddrRelocWorkspace(contextWorkspaceAddr, 0, nullptr);
    devProg->controlFlowCache.TaskAddrRelocProgramAndCtrlCache(reinterpret_cast<uint64_t>(devProg), reinterpret_cast<uint64_t>(&devProg->controlFlowCache), 0, 0);
    devProg->controlFlowCache.isActivated = true;

    devProg->Dump(0, true);
    devProg->DumpFile("./dum_dev_program.txt");
    devProg->ResetRerun();
    devProg->RuntimeVerify(0, 0);
    EXPECT_NE(devProg->GetInputTensorSlotIndexList().empty(), true);
    EXPECT_NE(devProg->GetOutputTensorSlotIndexList().empty(), true);
    EXPECT_NE(devProg->GetAssembleTensorSlotIndexList().empty(), true);
    (void)devProg->GetDevControlFlowBinary();
    (void)devProg->GetHostControlFlowBinary();
    (void)devProg->GetExpressionTableBinary();

    DevAscendFunction *devFunc = devProg->GetFunction(0);
    ASSERT_NE(devFunc, nullptr);
    EXPECT_NE(devProg->GetFunctionByRawName(devFunc->GetRawName()), nullptr);
    devFunc->Dump();
    EXPECT_EQ(devFunc->HasValueDepend(), false);
    EXPECT_EQ(devFunc->LookupIncastBySlotIndex(0), 0);
    EXPECT_EQ(devFunc->LookupOutcastBySlotIndex(0), -1);
    std::vector<int> slotIndexList = {0};
    (void)devFunc->LookupIncastBySlotIndexList(slotIndexList);
    (void)devFunc->LookupOutcastBySlotIndexList(slotIndexList);

    DevAscendFunction *devFunc1 = devProg->GetFunction(1);
    if (devFunc1 != nullptr) {
        devFunc->LookupConnectionSlotIndexFrom(devFunc1);
    }

    DevAscendFunctionDuppedData *devFuncDuppedData = devFunc->GetDuppedData();
    ASSERT_NE(devFuncDuppedData, nullptr);
    devFuncDuppedData->source_ = devFunc;
    (void)devFuncDuppedData->Dump();

    devProg->ResetFromLaunch();
}
static DevAscendProgram *BuildAndGetDevProgForExpectedMaxCachedNum()
{
    constexpr int LOOP_COUNT_INNER = 4;
    int s = 32;
    TileShape::Current().SetVecTile(32, 32);
    TileShape::Current().SetCubeTile({32, 32}, {32, 32}, {32, 32});
    Tensor t0(DT_FP32, {s, s}, "t0");
    Tensor t1(DT_FP32, {s, s}, "t1");
    Tensor t2(DT_FP32, {s, s}, "t2");
    Tensor out(DT_FP32, {LOOP_COUNT_INNER * s, s}, "out");
    FUNCTION("stitch_max_cached_num", {t0, t1, t2}, {out}) {
        LOOP("stitch_max_cached_num_L0", FunctionType::DYNAMIC_LOOP, i, LoopRange(LOOP_COUNT_INNER)) {
            auto temp = Add(t0, t0);
            SymbolicScalar s_min = std::ternary(i < 2, i, i + 1);
            IF(s_min == i) {
                temp = Add(temp, t1);
            }
            ELSE IF(s_min == i + 1) {
                temp = Add(temp, t2);
            }
            Assemble(temp, {i * s, 0}, out);
        }
    }
    std::shared_ptr<DyndevFunctionAttribute> funcDynDev =
        Program::GetInstance().GetLastFunction()->GetDyndevAttribute();
    EXPECT_NE(funcDynDev, nullptr);
    if (funcDynDev == nullptr) {
        return nullptr;
    }
    DevAscendProgram *devProg = reinterpret_cast<DevAscendProgram *>(funcDynDev->devProgBinary.data());
    EXPECT_NE(devProg, nullptr);
    return devProg;
}
TEST_F(TestDevEncode, test_max_stitch_function_num) {
    // case1:
    Program::GetInstance().Reset();
    config::Reset();
    config::SetPlatformConfig(KEY_ENABLE_AIHAC_BACKEND, true);
    config::SetRuntimeOption(STITCH_FUNCTION_INNER_MEMORY, 1024);
    config::SetRuntimeOption(STITCH_FUNCTION_OUTCAST_MEMORY, 1024);
    config::SetRuntimeOption(STITCH_FUNCTION_NUM_INITIAL, 256);
    config::SetRuntimeOption(STITCH_FUNCTION_NUM_STEP, 0);
    DevAscendProgram *devProg1 = BuildAndGetDevProgForExpectedMaxCachedNum();
    ASSERT_NE(devProg1, nullptr);
    EXPECT_EQ(devProg1->stitchMaxFunctionNum, 256u);

    // case2:
    Program::GetInstance().Reset();
    config::Reset();
    config::SetPlatformConfig(KEY_ENABLE_AIHAC_BACKEND, true);
    config::SetRuntimeOption(STITCH_FUNCTION_INNER_MEMORY, 1024);
    config::SetRuntimeOption(STITCH_FUNCTION_OUTCAST_MEMORY, 1024);
    config::SetRuntimeOption(STITCH_FUNCTION_NUM_INITIAL, 64);
    config::SetRuntimeOption(STITCH_FUNCTION_NUM_STEP, 0);
    DevAscendProgram *devProg2 = BuildAndGetDevProgForExpectedMaxCachedNum();
    ASSERT_NE(devProg2, nullptr);
    EXPECT_EQ(devProg2->stitchMaxFunctionNum, 128u);

    // case3:
    Program::GetInstance().Reset();
    config::Reset();
    config::SetPlatformConfig(KEY_ENABLE_AIHAC_BACKEND, true);
    config::SetRuntimeOption(STITCH_FUNCTION_INNER_MEMORY, 1024);
    config::SetRuntimeOption(STITCH_FUNCTION_OUTCAST_MEMORY, 1024);
    config::SetRuntimeOption(STITCH_FUNCTION_NUM_INITIAL, 64);
    config::SetRuntimeOption(STITCH_FUNCTION_NUM_STEP, 20);
    DevAscendProgram *devProg3 = BuildAndGetDevProgForExpectedMaxCachedNum();
    ASSERT_NE(devProg3, nullptr);
    EXPECT_EQ(devProg3->stitchMaxFunctionNum, 1024u);

    // case4:
    Program::GetInstance().Reset();
    config::Reset();
    config::SetPlatformConfig(KEY_ENABLE_AIHAC_BACKEND, true);
    config::SetRuntimeOption(STITCH_FUNCTION_INNER_MEMORY, 1024);
    config::SetRuntimeOption(STITCH_FUNCTION_OUTCAST_MEMORY, 1024);
    config::SetRuntimeOption(STITCH_FUNCTION_NUM_INITIAL, 64);
    config::SetRuntimeOption(STITCH_FUNCTION_NUM_STEP, 0);
    config::SetRuntimeOption(STITCH_FUNCTION_MAX_NUM, 512);
    DevAscendProgram *devProg4 = BuildAndGetDevProgForExpectedMaxCachedNum();
    ASSERT_NE(devProg4, nullptr);
    EXPECT_EQ(devProg4->stitchMaxFunctionNum, 512u);

}

TEST_F(TestDevEncode, test_dev_func_dupped) {
    DevAscendRawTensor rawTensor;
    std::vector<std::string> lines;
    std::stringstream oss;
    DevAscendFunctionDupped funcDuppped;
    funcDuppped.DumpRawShape(&rawTensor, 0, lines, oss);
}


