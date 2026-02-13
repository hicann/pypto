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

namespace npu::tile_fwk::Distributed {
class TestMoeDistributed : public ::testing::Test {
public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}

    void SetUp() override {
        Program::GetInstance().Reset();
        config::Reset();
        config::SetHostOption(COMPILE_STAGE, CS_EXECUTE_GRAPH);
        config::SetPlatformConfig(KEY_ENABLE_COST_MODEL, false);
    }

    void TearDown() override {}

protected:
    bool oriEnableAihacBackend = false;
};

std::string MoeDistributedGetFunctionRawName(const std::string& functionName)
{
    std::string functionRawName = FUNCTION_PREFIX + functionName + SUB_FUNC_SUFFIX;
#if ENABLE_HIDDENLOOP
    functionRawName += HIDDEN_FUNC_SUFFIX;
#endif
    return functionRawName;
}

TEST_F(TestMoeDistributed, MoeDistributedDispatchV2) {
    const char *group = "hcom1";
    DataType dType = DT_BF16;
    int routingExpertNum = 160;
    int topK = 8;
    int batchSize = 8;
    int hiddenSize = 5120;
    int rankSize = 4;

    int32_t expandXRowShape = topK * rankSize < routingExpertNum ?
        static_cast<int32_t>(batchSize) * static_cast<int32_t>(topK) * rankSize :
        static_cast<int32_t>(batchSize) * routingExpertNum;
    
    Shape xShape{batchSize, hiddenSize};
    Shape expertIdsShape{batchSize, topK};
    Shape expandXShape{expandXRowShape, hiddenSize};
    Shape recvCountsShape{1};
    Shape assistInfoForCombineShape{expandXRowShape, 64};
    Shape expertTokenNumsShape{routingExpertNum / rankSize};

    Tensor x(dType, xShape, "x");
    Tensor expertIds(DataType::DT_INT32, expertIdsShape, "expertIds");
    Tensor expertTokenNums(DataType::DT_INT32, expertTokenNumsShape, "expertTokenNums");
    Tensor expandX(dType, expandXShape, "expandX");
    Tensor assistInfoForCombine(DataType::DT_INT32, assistInfoForCombineShape, "assistInfoForCombine");
    Tensor recvCounts(DataType::DT_INT32, recvCountsShape, "recvCounts");


    FUNCTION("DISPATCH_F", {x, expertIds}, {expandX, assistInfoForCombine, expertTokenNums, recvCounts}) {
        Distributed::MoeDistributedDispatchV2(x, expertIds, group,
            rankSize, routingExpertNum, 0, 0, expandX, assistInfoForCombine, expertTokenNums, recvCounts);
    }

    auto functionRawName = MoeDistributedGetFunctionRawName("MoeDistributedDispatchSendData");
    auto function = Program::GetInstance().GetFunctionByRawName(functionRawName);
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenCloudNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

TEST_F(TestMoeDistributed, MoeDistributedDispatch) {
    const char *group = "hcom12";
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
        Distributed::MoeDistributedDispatch(tokenTensor, tokenExpertTable, expandX, validCnt, combineInfo, group, moeConfig);
    }

    auto functionRawName = MoeDistributedGetFunctionRawName("L0");
    auto function = Program::GetInstance().GetFunctionByRawName(functionRawName);
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenCloudNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

void TestMoeDistributedCombineFunc(std::function<void(const Tensor&, const Tensor&, const Tensor&, const Tensor&,
    const char*, uint32_t, uint32_t, uint32_t, uint32_t, Tensor&)> func, std::string loopName)
{
    const char *group = "hcom123";
    int32_t batchSize = 8;
    int32_t hiddenSize = 5120;
    int32_t moeExpertNum = 160;
    int32_t topK = 8;
    int32_t epWorldSize = 4;
    int32_t row = std::min(topK * batchSize * epWorldSize, batchSize * moeExpertNum);
    DataType dType = DT_BF16;

    Tensor expandX(dType, {row, hiddenSize}, "expandX");
    Tensor assistInfoForCombine(DT_INT32, {row, 3}, "assistInfoForCombine");
    Tensor recvCounts(DataType::DT_INT32, {1}, "recvCounts");
    Tensor expertScales(DT_FP32, {batchSize, topK}, "expertScales");
    Tensor out(dType, {batchSize, hiddenSize}, "out");

    FUNCTION("MoeDistributedCombineMain", {expandX, assistInfoForCombine, recvCounts, expertScales}, {out}) {
        func(expandX, assistInfoForCombine, recvCounts, expertScales, group, epWorldSize, moeExpertNum, 0, 0, out);
    }

    auto functionRawName = MoeDistributedGetFunctionRawName(loopName);
    auto function = Program::GetInstance().GetFunctionByRawName(functionRawName);
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenCloudNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

TEST_F(TestMoeDistributed, TestMoeDistributedCombine)
{
    TestMoeDistributedCombineFunc(Distributed::MoeDistributedCombine, "MoeDistributedCombine");
}

TEST_F(TestMoeDistributed, TestMoeDistributedCombineV2)
{
    TestMoeDistributedCombineFunc(Distributed::MoeDistributedCombineV2, "MoeDistributedCombineSend");
}

} // namespace npu::tile_fwk::Distributed
