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
 * \file test_codegen_moe_combine.cpp
 * \brief Unit test for codegen.
 */

#include <gtest/gtest.h>

#include "interface/function/function.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/configs/config_manager.h"
#include "codegen/codegen.h"
#include "codegen/cloudnpu/codegen_cloudnpu.h"
#include "test_codegen_common.h"

namespace npu::tile_fwk::Distributed {

class TestCodegenMoeCombine : public ::testing::Test {
public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}

    void SetUp() override {
        oriEnableAihacBackend = config::GetPlatformConfig(KEY_ENABLE_AIHAC_BACKEND, oriEnableAihacBackend);
        config::SetPlatformConfig(KEY_ENABLE_AIHAC_BACKEND, true);
        Program::GetInstance().Reset();
        config::Reset();
        config::SetHostOption(ONLY_CODEGEN, true);
    }

    void TearDown() override { config::SetPlatformConfig(KEY_ENABLE_AIHAC_BACKEND, oriEnableAihacBackend); }

protected:
    bool oriEnableAihacBackend = false;
};

void TestMoeCombine() {
    const char *group = "hcom123";
    int32_t batchSize = 8;
    int32_t hiddenSize = 5120;
    int32_t totalExpertNum = 160;
    int32_t topK = 8;
    int32_t rankSize = 4;
    int32_t row = std::min(topK * batchSize * rankSize, batchSize * totalExpertNum);
    DataType dType = DT_BF16;

    Tensor in(dType, {row, hiddenSize}, "in");
    Tensor combineInfo(DT_INT32, {row, 3}, "combineInfo");
    Tensor scale(DT_FP32, {batchSize, topK}, "scale");
    Tensor out(dType, {batchSize, hiddenSize}, "out");

    FUNCTION("MoeCombineReceive", {in, combineInfo, scale}, {out}) {
        Distributed::ShmemMoeCombine(in, combineInfo, scale, group, rankSize, totalExpertNum, out);
    }

#if ENABLE_HIDDENLOOP
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "MoeCombine" + SUB_FUNC_SUFFIX + HIDDEN_FUNC_SUFFIX);
#else
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "MoeCombine" + SUB_FUNC_SUFFIX);
#endif
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenCloudNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

TEST_F(TestCodegenMoeCombine, TestMoeCombine) {
    TestMoeCombine();
}
} // namespace npu::tile_fwk::Distributed
