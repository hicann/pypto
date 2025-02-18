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
 * \file test_codegen_dispatch.cpp
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
#include "interface/operation/distributed/distributed_common.h"
#include <vector>
#include <string>

namespace npu::tile_fwk {
namespace Distributed {
class TestCodegenDispatch : public ::testing::Test {
public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}

    void SetUp() override {
        Program::GetInstance().Reset();
        config::Reset();
        config::SetPlatformConfig(KEY_ONLY_HOST_COMPILE, true);
        config::SetPlatformConfig("ENABLE_COST_MODEL", false);
    }

    void TearDown() override {}

protected:
    bool oriEnableAihacBackend = false;
};

void TestMoeDispatch() {
    const char *group = "hcom123";
    DataType dType = DT_BF16;
    int routingExpertNum = 160;
    int topK = 8;
    int batchSize = 8;
    int hiddenSize = 5120;
    int rankSize = 4;

    int32_t expandXRowShape = topK * rankSize < routingExpertNum ?
        static_cast<int32_t>(batchSize) * static_cast<int32_t>(topK) * rankSize :
        static_cast<int32_t>(batchSize) * routingExpertNum;
    
    Shape tokenTensorShape{batchSize, hiddenSize};
    Shape tokenExpertTableShape{batchSize, topK};
    Shape expandXShape{expandXRowShape, hiddenSize};
    Shape validCntShape{routingExpertNum / rankSize};
    Shape combineInfoShape{expandXRowShape, 3};

    Tensor tokenTensor(dType, tokenTensorShape, "tokenTensor");
    Tensor tokenExpertTable(DataType::DT_INT32, tokenExpertTableShape, "tokenExpertTable");
    Tensor validCnt(DataType::DT_INT32, validCntShape, "validCnt");
    Tensor expandX(dType, expandXShape, "expandX");
    Tensor combineInfo(DataType::DT_INT32, combineInfoShape, "combineInfo");

    MoeConfig moeConfig{routingExpertNum, routingExpertNum / rankSize, rankSize};

    FUNCTION("DISPATCH_F", {tokenTensor, tokenExpertTable}, {expandX, validCnt, combineInfo}) {
        Distributed::MoeDispatch(tokenTensor, tokenExpertTable, expandX, validCnt, combineInfo, group, moeConfig);
    }

#if ENABLE_HIDDENLOOP
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "L0" + SUB_FUNC_SUFFIX + HIDDEN_FUNC_SUFFIX);
#else
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "L0" + SUB_FUNC_SUFFIX);
#endif
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenCloudNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

TEST_F(TestCodegenDispatch, TestMoeDispatchMultipyExperts) {
    TestMoeDispatch();
}
} // namespace Distributed
} // namespace npu::tile_fwk
