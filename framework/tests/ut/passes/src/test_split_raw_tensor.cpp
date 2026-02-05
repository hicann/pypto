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
 * \file test_split_raw_tensor.cpp
 * \brief Unit test for SplitRawTensor pass.
 */

#include <vector>
#include "gtest/gtest.h"
#include "interface/function/function.h"
#include "interface/configs/config_manager.h"
#include "ut_json/ut_json_tool.h"
#include "passes/tile_graph_pass/graph_optimization/split_raw.h"
#include "computational_graph_builder.h"

using namespace npu::tile_fwk;

class SplitLargeLocalRawTest : public testing::Test {
public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}

    void SetUp() override {
        Program::GetInstance().Reset();
        config::Reset();
        config::SetPlatformConfig("ENABLE_COST_MODEL", false);
    }
    void TearDown() override {}
};

void AddTensor(ComputationalGraphBuilder &G, const std::string name, const Shape &shape,
    const Shape &rawShape, const Offset &offset, const MemoryType memType) {
    std::string nameRaw = name + "_raw";
    G.AddTensor(DataType::DT_FP32, rawShape, nameRaw);
    auto tensorRaw = G.GetTensor(nameRaw);
    tensorRaw->SetMemoryTypeBoth(memType, true);
    G.AddTensor(DataType::DT_FP32, shape, name);
    auto tensorLogic = G.GetTensor(name);
    tensorLogic->SetMemoryTypeBoth(memType, true);
    tensorLogic->tensor = tensorRaw->tensor;
    tensorLogic->UpdateOffset(offset);
}

// input -> view1 -> a -> view2 -> b -> assemble1 -> output
TEST_F(SplitLargeLocalRawTest, TestSplitRawTensorCheceker) {
    int NUM_64 = 64;
    int NUM_128 = 128;
    int NUM_256 = 256;
    int NUM_512 = 512;
    std::vector<int64_t> shape1{NUM_64, NUM_64};
    std::vector<int64_t> shape2{NUM_128, NUM_128};
    std::vector<int64_t> shape3{NUM_256, NUM_256};
    std::vector<int64_t> shape4{NUM_512, NUM_512};
    ComputationalGraphBuilder G;

    AddTensor(G, "input", shape3, shape4, shape3, MemoryType::MEM_DEVICE_DDR);
    AddTensor(G, "a", shape2, shape3, shape2, MemoryType::MEM_DEVICE_DDR);
    AddTensor(G, "b", shape1, shape2, shape1, MemoryType::MEM_UB);
    AddTensor(G, "output", shape3, shape4, shape3, MemoryType::MEM_DEVICE_DDR);

    G.AddOp(Opcode::OP_VIEW, {"input"}, {"a"}, "view1");
    G.GetOp("view1")->SetOpAttribute(std::make_shared<ViewOpAttribute>(shape3, MemoryType::MEM_DEVICE_DDR));
    G.AddOp(Opcode::OP_VIEW, {"a"}, {"b"}, "view2");
    G.GetOp("view2")->SetOpAttribute(std::make_shared<ViewOpAttribute>(shape2, MemoryType::MEM_UB));
    G.AddOp(Opcode::OP_ASSEMBLE, {"b"}, {"output"}, "assemble1");
    G.GetOp("assemble1")->SetOpAttribute(std::make_shared<AssembleOpAttribute>(MemoryType::MEM_UB, shape2));
    
    G.SetInCast({"input"});
    G.SetOutCast({"output"});

    Function *function = G.GetFunction();
    npu::tile_fwk::SplitRawTensor splitLargeLocalRawPass;
    EXPECT_EQ(splitLargeLocalRawPass.PreCheck(*function), SUCCESS);
    EXPECT_EQ(splitLargeLocalRawPass.PostCheck(*function), FAILED);
    EXPECT_EQ(splitLargeLocalRawPass.RunOnFunction(*function), SUCCESS);
    EXPECT_EQ(splitLargeLocalRawPass.PostCheck(*function), SUCCESS);
}