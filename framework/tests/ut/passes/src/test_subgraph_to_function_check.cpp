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
 * \file test_subgraph_to_function_check.cpp
 * \brief Unit test for SubgraphToFunction preCheck and postCheck.
 */

#include "gtest/gtest.h"
#include <algorithm>
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "tilefwk/data_type.h"
#include "interface/inner/tile_shape.h"
#include "interface/function/function.h"
#include "interface/operation/operation.h"
#include "interface/configs/config_manager.h"
#include "interface/interpreter/raw_tensor_data.h"

using namespace npu::tile_fwk;
using namespace std;

class SubgraphToFunctionCheckTest : public testing::Test {
public:
    void SetUp() override {
        Program::GetInstance().Reset();
        config::Reset();
    }

    void TearDown() override {}
};

TEST_F(SubgraphToFunctionCheckTest, TestPrePostCheck) {
    constexpr int kTileSize = 32;
    constexpr int kVectorSize = 64;
    config::SetPassConfig("PVC2_OOO", "SubgraphToFunction", "PRE_CHECK", true);
    config::SetPassConfig("PVC2_OOO", "SubgraphToFunction", "POST_CHECK", true);
    TileShape::Current().SetVecTile(kTileSize, kTileSize);
    TileShape::Current().SetCubeTile({kTileSize, kTileSize}, {kTileSize, kTileSize}, {kTileSize, kTileSize});

    std::vector<int64_t> shape = {kVectorSize, kVectorSize};
    Tensor a(DT_FP32, shape, "a");
    Tensor b(DT_FP32, shape, "b");
    Tensor c1(DT_FP32, shape, "c1");
    Tensor c2(DT_FP32, shape, "c2");

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateConstantTensor<float>(a, 1.0f),
        RawTensorData::CreateConstantTensor<float>(b, 2.0f),
    });
    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<float>(c1, 0.0f),
        RawTensorData::CreateConstantTensor<float>(c2, 0.0f),
    });

    FUNCTION("SimpleTest", {a, b}, {c1, c2}) {
        Tensor temp1 = Add(a, b);
        temp1 = Mul(temp1, a);
        c1 = Sub(temp1, b);

        Tensor temp2 = Add(a, b);
        temp2 = Mul(temp2, a);
        c2 = Sub(temp2, b);
    }
    auto mainFunc = Program::GetInstance().GetFunctionByMagicName("TENSOR_SimpleTest_2");
    EXPECT_NE(mainFunc, nullptr);
    ALOG_INFO_F("Pre/Post check test completed");
}