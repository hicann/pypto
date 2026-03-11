/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_interp_log.cpp
 * \brief Interpreter 日志相关测试用例综合文件
 */

#include <gtest/gtest.h>
#include <vector>
#include <memory>
#include <string>
#include <functional>
#include <cstdio>
#include <unistd.h>

#include "interpreter_log_test_utils.h"
#include "interface/inner/tilefwk.h"
#include "interface/inner/pre_def.h"
#include "interface/configs/config_manager.h"
#include "interface/program/program.h"
#include "interface/function/function.h"
#include "interface/tensor/logical_tensor.h"
#include "interface/operation/operation.h"
#include "interface/interpreter/operation.h"
#include "interface/interpreter/calc.h"

#define private public
#include "utils/host_log/log_manager.h"
#undef private

namespace npu::tile_fwk {
class InterpreterLogTest : public testing::Test {
public:
    void SetUp() override {
        Program::GetInstance().Reset();
        config::Reset();
        ProgramData::GetInstance().Reset();
        if (!calc::IsVerifyEnabled()) {
            GTEST_SKIP() << "Verify not supported skip the verify test";
        }
        TileShape::Current().SetVecTile(32, 32);
        TileShape::Current().SetCubeTile({32, 32}, {32, 32}, {32, 32});
    }

    void TearDown() override {
        config::SetVerifyOption(KEY_ENABLE_PASS_VERIFY, true);
        config::SetVerifyOption(KEY_PASS_VERIFY_SAVE_TENSOR, true);
    }
};

TEST_F(InterpreterLogTest, ReshapeMismatchElementCount) {
    std::string logOutput = CaptureLogFileAndEcho([]() {
        config::SetVerifyOption(KEY_ENABLE_PASS_VERIFY, true);
        config::SetVerifyOption(KEY_PASS_VERIFY_SAVE_TENSOR, true);

        Tensor input(DT_FP32, {256, 1, 128}, "input");
        Tensor output(DT_FP32, {1, 128, 128}, "output");

        ProgramData::GetInstance().AppendInputs({
            RawTensorData::CreateConstantTensor<float>(input, 1.0f),
        });
        ProgramData::GetInstance().AppendOutputs({
            RawTensorData::CreateConstantTensor<float>(output, 0.0f),
        });

        FUNCTION("main", {input}, {output}) {
            LOOP("L0", FunctionType::DYNAMIC_LOOP, i, LoopRange(1)) {
                (void)i;
                TileShape::Current().SetVecTile(128, 128, 128);
                auto t1 = View(input, {128, 1, 128}, {30, 1, 128}, {0, 0, 0});
                auto t2 = Reshape(t1, {128, 128}, {30, 128});
                auto t3 = Reshape(t2, {1, 128, 128}, {1, 30, 128});
                Assemble(t3, {0, 0, 0}, output);
            }
        }
    });

    // 仅校验 verify 日志：不应出现 FAILED
    EXPECT_FALSE(VerifyLogContainsFailed(logOutput))
        << "Expected no FAILED in verify log, captured: " << logOutput;
}

// 测试精度对比失败场景能否正确输出错误日志，并捕获日志进行校验
TEST_F(InterpreterLogTest, PrecisionMismatchErrorLog) {
    bool oriLogStdout = LogManager::Instance().enableStdOut_;
    LogManager::Instance().enableStdOut_ = true;
    std::string logOutput = CaptureStdoutAndEcho([]() {
        config::SetVerifyOption(KEY_ENABLE_PASS_VERIFY, true);
        config::SetVerifyOption(KEY_PASS_VERIFY_SAVE_TENSOR, true);

        int s = 16;
        Tensor input(DT_FP32, {s, s}, "input");
        Tensor output(DT_FP32, {s, s}, "output");

        ProgramData::GetInstance().AppendInputs({
            RawTensorData::CreateConstantTensor<float>(input, 1.0f),
        });
        ProgramData::GetInstance().AppendOutputs({
            RawTensorData::CreateConstantTensor<float>(output, 0.0f),
        });
        // 故意构造与实际输出不一致的 golden，触发精度错误日志
        ProgramData::GetInstance().AppendGoldens({
            RawTensorData::CreateConstantTensor<float>(output, 10.0f),
        });

        FUNCTION("main", {input}, {output}) {
            LOOP("L0", FunctionType::DYNAMIC_LOOP, i, LoopRange(1)) {
                (void)i;
                auto t = View(input, {s, s}, {0, 0});
                Assemble(t, {0, 0}, output);
            }
        }
    });

    // 仅校验 verify 日志：应出现 FAILED（精度校验失败）
    EXPECT_TRUE(VerifyLogContainsFailed(logOutput))
        << "Expected FAILED in verify log for precision mismatch, captured: " << logOutput;
    LogManager::Instance().enableStdOut_ = oriLogStdout;
}

// 测试空 loop (start=0, end=0) 能否触发 interpreter 的 "skip execute due to idx range = 0" 日志
TEST_F(InterpreterLogTest, EmptyLoopStartEndZero) {
    std::string logOutput = CaptureLogFileAndEcho([]() {
        config::SetVerifyOption(KEY_ENABLE_PASS_VERIFY, true);
        config::SetVerifyOption(KEY_PASS_VERIFY_SAVE_TENSOR, true);

        int s = 32;
        Tensor input(DT_FP32, {s, s}, "input");
        Tensor output(DT_FP32, {s, s}, "output");

        ProgramData::GetInstance().AppendInputs({
            RawTensorData::CreateConstantTensor<float>(input, 1.0f),
        });
        ProgramData::GetInstance().AppendOutputs({
            RawTensorData::CreateConstantTensor<float>(output, 0.0f),
        });
        ProgramData::GetInstance().AppendGoldens({
            RawTensorData::CreateConstantTensor<float>(output, 0.0f),
        });

        FUNCTION("main", {input}, {output}) {
            LOOP("L0", FunctionType::DYNAMIC_LOOP, i, LoopRange(0, 0, 1)) {
                (void)i;
                auto t = View(input, {s, s}, {0, 0});
                Assemble(t, {0, 0}, output);
            }
        }
    });

    // 仅校验 verify 日志：不应出现 FAILED
    EXPECT_FALSE(VerifyLogContainsFailed(logOutput))
        << "Expected no FAILED in verify log, captured: " << logOutput;
}

} // namespace npu::tile_fwk
