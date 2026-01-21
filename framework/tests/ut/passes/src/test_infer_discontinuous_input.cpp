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
 * \file test_memory_reuse.cpp
 * \brief Unit test for RemoveRedundantReshape pass.
 */

#include <gtest/gtest.h>
#include <vector>
#include "tilefwk/tilefwk.h"
#include "interface/function/function.h"
#include "interface/inner/tilefwk.h"
#include "interface/configs/config_manager.h"
#include "computational_graph_builder.h"
#include "passes/tile_graph_pass/graph_optimization/infer_discontinuous_input.h"

namespace npu {
namespace tile_fwk {

class TestInferDiscontinuousInput : public ::testing::Test {
public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}

    void SetUp() override {
        Program::GetInstance().Reset();
        config::Reset();
        config::SetHostOption(COMPILE_STAGE, HOST_COMPILE_END);
        config::SetPlatformConfig(KEY_ENABLE_COST_MODEL, false);
    }
    void TearDown() override {
    }
};

void Construct(ComputationalGraphBuilder &G) {
    G.AddTensor(DataType::DT_FP16, {16, 128}, MemoryType::MEM_DEVICE_DDR, "inputTensor0");
    G.AddTensor(DataType::DT_FP16, {16, 128}, MemoryType::MEM_DEVICE_DDR, "inputTensor1");
    G.AddTensor(DataType::DT_FP16, {16, 128}, MemoryType::MEM_DEVICE_DDR, "inputTensor2");
    G.AddTensor(DataType::DT_FP16, {16, 128}, MemoryType::MEM_DEVICE_DDR, "inputTensor3");
    G.AddTensor(DataType::DT_FP16, {64, 128}, MemoryType::MEM_DEVICE_DDR, "outputTensor");

    G.AddOp(Opcode::OP_ASSEMBLE, {"inputTensor0"}, {"outputTensor"}, "assemble_0");
    auto assemble_0 = G.GetOp("assemble_0");
    auto attrAssemble_0 = std::make_shared<AssembleOpAttribute>(MemoryType::MEM_DEVICE_DDR, std::vector<int64_t>{0, 0});
    assemble_0->SetOpAttribute(attrAssemble_0);
    
    G.AddOp(Opcode::OP_ASSEMBLE, {"inputTensor1"}, {"outputTensor"}, "assemble_1");
    auto assemble_1 = G.GetOp("assemble_1");
    auto attrAssemble_1 = std::make_shared<AssembleOpAttribute>(MemoryType::MEM_DEVICE_DDR, std::vector<int64_t>{16, 0});
    assemble_1->SetOpAttribute(attrAssemble_1);
    
    G.AddOp(Opcode::OP_ASSEMBLE, {"inputTensor2"}, {"outputTensor"}, "assemble_2");
    auto assemble_2 = G.GetOp("assemble_2");
    auto attrAssemble_2 = std::make_shared<AssembleOpAttribute>(MemoryType::MEM_DEVICE_DDR, std::vector<int64_t>{32, 0});
    assemble_2->SetOpAttribute(attrAssemble_2);
    
    G.AddOp(Opcode::OP_ASSEMBLE, {"inputTensor3"}, {"outputTensor"}, "assemble_3");
    auto assemble_3 = G.GetOp("assemble_3");
    auto attrAssemble_3 = std::make_shared<AssembleOpAttribute>(MemoryType::MEM_DEVICE_DDR, std::vector<int64_t>{48, 0});
    assemble_3->SetOpAttribute(attrAssemble_3);
    
    G.SetInCast({});
    G.SetOutCast({"outputTensor"});
}

TEST_F(TestInferDiscontinuousInput, testScenarioWithoutInsert_1) {
    ComputationalGraphBuilder G;
    Function *function = G.GetFunction();
    EXPECT_NE(function, nullptr);
    Construct(G);
    // run pass
    InferDiscontinuousInput inferDiscontinuousInput;
    EXPECT_EQ(inferDiscontinuousInput.Run(*function, "", "", 0), SUCCESS);
    EXPECT_EQ(inferDiscontinuousInput.PostCheck(*function), SUCCESS);

    EXPECT_EQ(function->Operations().size(), 4);
}

TEST_F(TestInferDiscontinuousInput, testScenarioWithoutInsert_2) {
    ComputationalGraphBuilder G;
    Function *function = G.GetFunction();
    EXPECT_NE(function, nullptr);
    Construct(G);

    auto inputTensor2 = G.GetTensor("inputTensor2");
    auto inputTensor3 = G.GetTensor("inputTensor3");
    inputTensor3->tensor = inputTensor2->tensor;
    inputTensor2->tensor->UpdateRawShape({32, 128});
    inputTensor3->UpdateOffset({16, 0});

    // run pass
    InferDiscontinuousInput inferDiscontinuousInput;
    EXPECT_EQ(inferDiscontinuousInput.Run(*function, "", "", 0), SUCCESS);
    EXPECT_EQ(inferDiscontinuousInput.PostCheck(*function), SUCCESS);

    EXPECT_EQ(function->Operations().size(), 4);
}

void check(Function *function, ComputationalGraphBuilder &G) {
    InferDiscontinuousInput inferDiscontinuousInput;
    EXPECT_EQ(inferDiscontinuousInput.PostCheck(*function), FAILED);
    EXPECT_EQ(inferDiscontinuousInput.Run(*function, "", "", 0), SUCCESS);
    EXPECT_EQ(inferDiscontinuousInput.PostCheck(*function), SUCCESS);
    EXPECT_EQ(function->Operations().size(), 12);

    auto inputTensor0 = G.GetTensor("inputTensor0");
    auto inputTensor1 = G.GetTensor("inputTensor1");
    auto inputTensor2 = G.GetTensor("inputTensor2");
    auto inputTensor3 = G.GetTensor("inputTensor3");
    auto viewOp0 = *inputTensor0->GetConsumers().begin();
    EXPECT_EQ(viewOp0->GetOpcode(), Opcode::OP_VIEW);
    auto viewOp1 = *inputTensor1->GetConsumers().begin();
    EXPECT_EQ(viewOp1->GetOpcode(), Opcode::OP_VIEW);
    auto viewOp2 = *inputTensor2->GetConsumers().begin();
    EXPECT_EQ(viewOp2->GetOpcode(), Opcode::OP_VIEW);
    auto viewOp3 = *inputTensor3->GetConsumers().begin();
    EXPECT_EQ(viewOp3->GetOpcode(), Opcode::OP_VIEW);
    auto insertTensor0 = viewOp0->GetOOperands()[0];
    EXPECT_EQ(insertTensor0->GetMemoryTypeOriginal(), MemoryType::MEM_UB);
    auto insertTensor1 = viewOp1->GetOOperands()[0];
    EXPECT_EQ(insertTensor1->GetMemoryTypeOriginal(), MemoryType::MEM_UB);
    auto insertTensor2 = viewOp2->GetOOperands()[0];
    EXPECT_EQ(insertTensor2->GetMemoryTypeOriginal(), MemoryType::MEM_UB);
    auto insertTensor3 = viewOp3->GetOOperands()[0];
    EXPECT_EQ(insertTensor3->GetMemoryTypeOriginal(), MemoryType::MEM_UB);
    auto assembleOp0 = *insertTensor0->GetConsumers().begin();
    EXPECT_EQ(assembleOp0->GetOpcode(), Opcode::OP_ASSEMBLE);
    auto assembleOp1 = *insertTensor1->GetConsumers().begin();
    EXPECT_EQ(assembleOp1->GetOpcode(), Opcode::OP_ASSEMBLE);
    auto assembleOp2 = *insertTensor2->GetConsumers().begin();
    EXPECT_EQ(assembleOp2->GetOpcode(), Opcode::OP_ASSEMBLE);
    auto assembleOp3 = *insertTensor3->GetConsumers().begin();
    EXPECT_EQ(assembleOp3->GetOpcode(), Opcode::OP_ASSEMBLE);
}
TEST_F(TestInferDiscontinuousInput, testScenarioInsert_1) {
    ComputationalGraphBuilder G;
    Function *function = G.GetFunction();
    EXPECT_NE(function, nullptr);
    Construct(G);
    auto inputTensor0 = G.GetTensor("inputTensor0");
    auto inputTensor1 = G.GetTensor("inputTensor1");
    auto inputTensor2 = G.GetTensor("inputTensor2");
    auto inputTensor3 = G.GetTensor("inputTensor3");
    inputTensor3->tensor = inputTensor1->tensor;
    inputTensor1->tensor->UpdateRawShape({32, 128});
    inputTensor3->UpdateOffset({16, 0});

    // run pass
    check(function, G);
}

TEST_F(TestInferDiscontinuousInput, testScenarioInsert_2) {
    ComputationalGraphBuilder G;
    Function *function = G.GetFunction();
    EXPECT_NE(function, nullptr);
    Construct(G);
    auto inputTensor0 = G.GetTensor("inputTensor0");
    auto inputTensor1 = G.GetTensor("inputTensor1");
    auto inputTensor2 = G.GetTensor("inputTensor2");
    auto inputTensor3 = G.GetTensor("inputTensor3");
    inputTensor0->tensor->UpdateRawShape({32, 128});
    inputTensor1->tensor->UpdateRawShape({32, 128});
    inputTensor2->tensor->UpdateRawShape({32, 128});
    inputTensor3->tensor->UpdateRawShape({32, 128});
    // run pass
    check(function, G);
}
} // namespace tile_fwk
} // namespace npu