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
 * \file test_remove_redundant_reshape.cpp
 * \brief Unit test for RemoveRedundantReshape pass.
 */

#include <gtest/gtest.h>
#include "interface/function/function.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "passes/pass_mgr/pass_manager.h"
#include "interface/configs/config_manager.h"
#include <fstream>
#include <vector>
#include <string>

using namespace npu::tile_fwk;

class RemoveRedundantReshapeTest : public ::testing::Test {
public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}

    void SetUp() override {
        Program::GetInstance().Reset();
        config::Reset();
        config::SetPlatformConfig(KEY_ONLY_HOST_COMPILE, true);
        config::SetHostConfig(KEY_STRATEGY, "ReshapeTestStrategy");
        config::SetPlatformConfig("ENABLE_COST_MODEL", false);
    }
    void TearDown() override {}
};

// Verify that Reshape whose output shape is the same as its input shape can be removed and consumer's inputs are updated
TEST_F(RemoveRedundantReshapeTest, TestSameInputOutputShape) {
    // Define the shape of the Tensors
    std::vector<int64_t> shape1{1, 256, 512};
    std::vector<int64_t> shape2{1, 512, 256};

    // Create Tensor
    Tensor in_tensor(DT_FP32, shape1, "in_tensor");
    Tensor out_tensor_A(DT_FP32, shape1, "out_tensor_A");
    Tensor out_tensor_B(DT_FP32, shape2, "out_tensor_B");

    // Initialize PassManager
    PassManager &passManager = PassManager::Instance();
    passManager.RegisterStrategy("ReshapeTestStrategy", {
        { "RemoveRedundantReshape", "RemoveRedundantReshape"},
    });
    ConfigManager::Instance();

    // Create and configure the function
    Function* originFunction = nullptr;
    std::vector<int64_t> originOpmagic;
    FUNCTION("ReshapeFunction") {
        // Add Operations
        out_tensor_A = Reshape(in_tensor, shape1);
        out_tensor_B = Reshape(out_tensor_A, shape2);
        originFunction = Program::GetInstance().GetCurrentFunction();
        ASSERT_NE(originFunction, nullptr) << "当前函数指针为空";
        auto operations = originFunction->Operations();
        for (const auto &op : operations) {
            originOpmagic.emplace_back(op.opmagic);
        }
    }

    // Print and draw the graph
    Function* currentFunction = Program::GetInstance().GetFunctionByRawName("TENSOR_ReshapeFunction");
    printf("Function name is %s\n", currentFunction->GetMagicName().c_str());
    // ================== Verify Pass Effect ==================
    auto updated_operations = currentFunction->Operations();
    EXPECT_EQ(updated_operations.size(), 3) << "3 operations should remain（View + Reshape + Assemble）";
    EXPECT_EQ(updated_operations[0].GetOpcode(), Opcode::OP_VIEW) << "View operation should remain";
    EXPECT_EQ(updated_operations[1].GetOpcode(), Opcode::OP_RESHAPE) << "The second Reshape should remain";
    EXPECT_EQ(updated_operations[1].opmagic, originOpmagic.back()) << "The retained Reshape is the second one";
    EXPECT_EQ(updated_operations[2].GetOpcode(), Opcode::OP_ASSEMBLE) << "Assemble operation should remain";

    // Check Tensor connection relationships
    auto& view_op = updated_operations[0];
    auto& remaining_reshape_op = updated_operations[1];
    auto& assemble_op = updated_operations[2];
    EXPECT_EQ(remaining_reshape_op.GetIOperands()[0], view_op.GetOOperands()[0]) << "The output of View should be connected to the input of the remaining Reshape";
    EXPECT_EQ(assemble_op.GetIOperands()[0], remaining_reshape_op.GetOOperands()[0]) << "The output of the remaining Reshape should be connected to the input of Assemble";
}

TEST_F(RemoveRedundantReshapeTest, TestReshapeChain) {
    // Define Tensor shapes
    std::vector<int64_t> shape1{1, 256, 512};
    std::vector<int64_t> shape2{1, 512, 256};
    std::vector<int64_t> shape3{1, 128, 1024};
    // Create Tensors
    Tensor in_tensor(DT_FP32, shape1, "in_tensor");
    Tensor out_tensor_B(DT_FP32, shape3, "out_tensor_B");

    // Initialize PassManager
    PassManager &passManager = PassManager::Instance();
    passManager.RegisterStrategy("ReshapeTestStrategy", {
        { "RemoveRedundantReshape", "RemoveRedundantReshape"},
    });
    ConfigManager::Instance();

    // Create and configure the function
    Function* originFunction = nullptr;
    std::vector<int64_t> originOpmagic;
    FUNCTION("ReshapeChainFunction") {
        // Add Operations
        Tensor out_tensor_A = Reshape(in_tensor, shape2);
        out_tensor_B = Reshape(out_tensor_A, shape3);
        originFunction = Program::GetInstance().GetCurrentFunction();
        ASSERT_NE(originFunction, nullptr) << "Current function pointer is null";
        auto operations = originFunction->Operations();
        for (const auto &op : operations) {
            originOpmagic.emplace_back(op.opmagic);
        }
    }

    // Print and draw the graph
    Function* currentFunction = Program::GetInstance().GetFunctionByRawName("TENSOR_ReshapeChainFunction");

    // ================== Verify the effect of the Pass ==================
    auto updated_operations = currentFunction->Operations();

    // Verify if the RemoveRedundantReshape Pass removes redundant Reshape operation
    EXPECT_EQ(updated_operations.size(), 3) << "After the Pass, there should be 3 operations (View + Reshape + Assemble)";
    EXPECT_EQ(updated_operations[0].GetOpcode(), Opcode::OP_VIEW) << "View operation should be kept";
    EXPECT_EQ(updated_operations[1].GetOpcode(), Opcode::OP_RESHAPE) << "The second Reshape (valid) should be kept";
    EXPECT_EQ(updated_operations[1].opmagic, originOpmagic.back()) << "The kept Reshape is the second one";
    EXPECT_EQ(updated_operations[2].GetOpcode(), Opcode::OP_ASSEMBLE) << "Assemble operation should be kept";

    // Check the Tensor connection relationship
    auto& view_op = updated_operations[0];
    auto& remaining_reshape_op = updated_operations[1];
    EXPECT_EQ(remaining_reshape_op.GetIOperands()[0], view_op.GetOOperands()[0]) << "The output of View should connect to the input of the second Reshape";

    // ================== Verify Reshape Input and Output Shapes ==================
    auto& reshape_input = remaining_reshape_op.GetIOperands()[0];
    EXPECT_EQ(reshape_input->shape, shape1) << "The input shape of the remaining Reshape operation should be the same as shape1";
    auto& reshape_output = remaining_reshape_op.GetOOperands()[0];
    EXPECT_EQ(reshape_output->shape, shape3) << "The output shape of the remaining Reshape operation should be the same as shape3";
}

TEST_F(RemoveRedundantReshapeTest, TestReplaceInput) {
    // Define Tensor shapes
    std::vector<int64_t> shape1{1, 256, 512};
    std::vector<int64_t> shape2{1, 512, 256};
    std::vector<int64_t> shape3{1, 128, 1024};

    // Create Tensors
    Tensor in_tensor(DT_FP32, shape1, "in_tensor");
    Tensor out_tensor_B(DT_FP32, shape2, "out_tensor_B");
    Tensor out_tensor_C(DT_FP32, shape3, "out_tensor_C");

    // Initialize PassManager
    PassManager &passManager = PassManager::Instance();
    passManager.RegisterStrategy("ReshapeTestStrategy",
        {{"RemoveRedundantReshape", "RemoveRedundantReshape"}}
    );

    TileShape::Current().SetVecTile({1, 64, 64});
    // Create and configure the function
    Function* originFunction = nullptr;
    std::vector<int64_t> reshape_opmagics;
    FUNCTION("ReplaceInputFunction") {
        // Add Operations
        Tensor out_tensor_A = Reshape(in_tensor, shape2);
        out_tensor_B = Reshape(out_tensor_A, shape3);
        Tensor add_tensor(DT_FP32, shape2, "add_tensor");
        out_tensor_C = Add(out_tensor_A, add_tensor);

        originFunction = Program::GetInstance().GetCurrentFunction();
        ASSERT_NE(originFunction, nullptr) << "Current function pointer is null";
        auto operations = originFunction->Operations();
        for (const auto &op : operations) {
            if (op.GetOpcodeStr() == "RESHAPE") {
                reshape_opmagics.push_back(op.opmagic);
            }
        }
    }

    // Print and draw the graph
    Function* currentFunction = Program::GetInstance().GetFunctionByRawName("TENSOR_ReplaceInputFunction");
    auto updated_operations = currentFunction->Operations();

    Operation* first_reshape_op = nullptr;
    for (size_t i = 0; i < updated_operations.size(); i++) {
        auto& op = updated_operations[i];
        if (op.opmagic == reshape_opmagics[0]) {
            first_reshape_op = &op;
        }
        if (op.opmagic == reshape_opmagics[1]) {
            auto& second_reshape_op = op;
            if (first_reshape_op) {
                EXPECT_EQ(second_reshape_op.GetIOperands()[0], first_reshape_op->GetIOperands()[0]) << "The input of the second Reshape operation should be replaced by the input of the first Reshape operation";
                auto& reshape_input = second_reshape_op.GetIOperands()[0];
                EXPECT_EQ(reshape_input->shape, shape1) << "The input shape of the second Reshape operation should be the same as shape1";
            } else {
                std::cerr << "Error: first_reshape_op not found!" << std::endl;
            }
        }
    }

    // Verify if the RemoveRedundantReshape Pass replaces Input of the reshape operation
    EXPECT_EQ(updated_operations.size(), 6) << "After the Pass, there should be 5 operations (View + Reshape + Reshape + Add + Assemble)";
}
