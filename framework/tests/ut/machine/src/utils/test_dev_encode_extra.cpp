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
 * \file test_dev_encode_extra.cpp
 * \brief Additional UT coverage for dev_encode.cpp top-level functions
 */

#include <gtest/gtest.h>
#include "machine/utils/dynamic/dev_encode.h"
#include "machine/utils/dynamic/dev_encode_program.h"
#include "tilefwk/tilefwk.h"
#include "tilefwk/tilefwk_op.h"
#include "interface/inner/tilefwk.h"
#include "interface/configs/config_manager.h"

using namespace npu::tile_fwk;
using namespace npu::tile_fwk::dynamic;

class TestDevEncodeExtra : public testing::Test {
protected:
    void SetUp() override
    {
        Program::GetInstance().Reset();
        config::Reset();
        config::SetHostOption(COMPILE_STAGE, CS_EXECUTE_GRAPH);
        config::SetPlatformConfig(KEY_ENABLE_COST_MODEL, false);
        config::SetRuntimeOption(STITCH_FUNCTION_MAX_NUM, 64);
        config::SetPlatformConfig(KEY_ENABLE_AIHAC_BACKEND, true);
        TileShape::Current().SetVecTile(32, 32);
        TileShape::Current().SetCubeTile({32, 32}, {32, 32}, {32, 32});
    }
    void TearDown() override
    {
        Program::GetInstance().Reset();
        Program::GetInstance().SetLastFunction(nullptr);
        config::Reset();
        config::SetHostOption(COMPILE_STAGE, CS_EXECUTE_GRAPH);
        config::SetPlatformConfig(KEY_ENABLE_AIHAC_BACKEND, true);
    }
};

TEST_F(TestDevEncodeExtra, EncodeDevAscendProgram_NullBase_CalculatesSizeOnly)
{
    constexpr int LOOP_COUNT = 4;
    int s = 32;
    Tensor t0(DT_FP32, {s, s}, "t0");
    Tensor t1(DT_FP32, {s, s}, "t1");
    Tensor out(DT_FP32, {LOOP_COUNT * s, s}, "out");

    FUNCTION("test_encode_prog_sizeonly", {t0, t1}, {out})
    {
        LOOP("test_L0", FunctionType::DYNAMIC_LOOP, i, LoopRange(LOOP_COUNT))
        {
            auto temp = Add(t0, t1);
            Assemble(temp, {i * s, 0}, out);
        }
    }

    auto* func = Program::GetInstance().GetLastFunction();
    ASSERT_NE(func, nullptr);
    uint64_t offset = 0;
    EncodeDevAscendProgram(func, offset, nullptr);
    EXPECT_GT(offset, 0u);
}

TEST_F(TestDevEncodeExtra, EncodeDevAscendProgram_WithBase_EncodesFull)
{
    constexpr int LOOP_COUNT = 4;
    int s = 32;
    Tensor t0(DT_FP32, {s, s}, "t0");
    Tensor t1(DT_FP32, {s, s}, "t1");
    Tensor out(DT_FP32, {LOOP_COUNT * s, s}, "out");

    FUNCTION("test_encode_prog_full", {t0, t1}, {out})
    {
        LOOP("test_L0", FunctionType::DYNAMIC_LOOP, i, LoopRange(LOOP_COUNT))
        {
            auto temp = Add(t0, t1);
            Assemble(temp, {i * s, 0}, out);
        }
    }

    auto* func = Program::GetInstance().GetLastFunction();
    ASSERT_NE(func, nullptr);

    uint64_t sizeOnlyOffset = 0;
    EncodeDevAscendProgram(func, sizeOnlyOffset, nullptr);
    EXPECT_GT(sizeOnlyOffset, 0u);

    std::vector<uint8_t> buffer(sizeOnlyOffset);
    auto* base = reinterpret_cast<DevAscendProgram*>(buffer.data());
    uint64_t offset = 0;
    EncodeDevAscendProgram(func, offset, base);
    EXPECT_EQ(offset, sizeOnlyOffset);
}

TEST_F(TestDevEncodeExtra, EncodeDevAscendProgram_DynamicLoopWithIf)
{
    std::vector<int64_t> shape{32, 32};
    Tensor t0(DT_FP32, shape, "t0");
    Tensor t1(DT_FP32, shape, "t1");
    Tensor out(DT_FP32, {128, 32}, "out");
    constexpr int LOOP_COUNT = 4;
    int s = 32;

    FUNCTION("main_encode_loop", {t0, t1}, {out})
    {
        LOOP("main_L0", FunctionType::DYNAMIC_LOOP, i, LoopRange(LOOP_COUNT))
        {
            auto temp = Add(t0, t0);
            SymbolicScalar sMin = std::ternary(i < 2, i, i + 1);
            IF(sMin == i) { temp = Add(temp, t1); }
            Assemble(temp, {i * s, 0}, out);
        }
    }

    auto* func = Program::GetInstance().GetLastFunction();
    ASSERT_NE(func, nullptr);
    uint64_t offset = 0;
    EncodeDevAscendProgram(func, offset, nullptr);
    EXPECT_GT(offset, 0u);
}
