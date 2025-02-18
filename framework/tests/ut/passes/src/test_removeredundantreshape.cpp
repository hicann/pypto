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
 * \file test_expand_function.cpp
 * \brief Unit test for ExpandFunction pass.
 */

#include <gtest/gtest.h>
#include <vector>
#include <string>
#include "interface/function/function.h"
#include "tilefwk/tilefwk.h"
#include "passes/pass_mgr/pass_manager.h"
#include "interface/configs/config_manager.h"
#include "ut_json/ut_json_tool.h"

#define private public
#include "passes/tensor_graph_pass/remove_redundant_reshape.h"

namespace npu {
namespace tile_fwk{
static const size_t kSizeZero = 0UL;
static const size_t kSizeOne = 1UL;
static const size_t kSizeTwelve = 12UL;
static const size_t kSizeThirteen = 13UL;
static const uint16_t kNumZero = 0u;
static const uint16_t kNumOne = 1u;
static const uint16_t kNumTwo = 2u;
static const uint16_t kNumThree = 3u;
static const uint16_t kNumFour = 4u;
static const uint16_t kNumEight = 8u;
static const uint16_t kNumExpFour = 16u;
static const uint16_t kNumExpFive = 32u;
static const uint16_t kNumExpSix = 64u;
static const uint16_t kNumExpSeven = 128u;

class TestRemoveRedundantReshapePass : public ::testing::Test {
public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}

    void SetUp() override {
        Program::GetInstance().Reset();
        config::Reset();
        config::SetPlatformConfig(KEY_ONLY_HOST_COMPILE, true);
        config::SetHostConfig(KEY_STRATEGY, "ExpandFunctionTestStrategy");
        config::SetPlatformConfig("ENABLE_COST_MODEL", false);
    }
    void TearDown() override {}
};

/*
RemoveReshapeChain
inCast{8,16}->reshape->ubTensor1{16,8}->reshape->ubTensor2{32,4}->sqrt->outCast{32,4}
inCast{8,16}->reshape->ubTensor2{32,4}->sqrt->outCast{32,4}
*/
TEST_F(TestRemoveRedundantReshapePass, RemoveRedundantReshapeUTest1) {
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestRemoveRedundantReshape", "TestRemoveRedundantReshape", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    // Prepare the graph
    std::vector<int64_t> shape1 = {kNumEight, kNumExpFour};
    std::vector<int64_t> shape2 = {kNumExpFour, kNumEight};
    std::vector<int64_t> shape3 = {kNumExpFive, kNumFour};
    auto inCast = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape1);
    auto ubTensor1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape2);
    auto ubTensor2 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape3);
    auto outCast = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape3);

    auto &reshape1 = currFunctionPtr->AddOperation(Opcode::OP_RESHAPE, {inCast}, {ubTensor1});
    auto &reshape2 = currFunctionPtr->AddOperation(Opcode::OP_RESHAPE, {ubTensor1}, {ubTensor2});
    auto &sqrt = currFunctionPtr->AddOperation(Opcode::OP_SQRT, {ubTensor2}, {outCast});

    currFunctionPtr->inCasts_.push_back(inCast);
    currFunctionPtr->outCasts_.push_back(outCast);

    RemoveRedundantReshape removeredundantpass;
    auto status = removeredundantpass.RunOnFunction(*currFunctionPtr);
    EXPECT_EQ(status, SUCCESS);

    const auto &operations = currFunctionPtr->Operations();
    uint32_t reshape_num = kNumZero;
    for (auto &op : operations) {
        if (op.GetOpcode() == Opcode::OP_RESHAPE) {
            EXPECT_EQ(reshape2.GetOpMagic(), op.GetOpMagic());
            EXPECT_EQ(reshape2.GetInputOperand(kSizeZero), inCast);
            ++reshape_num;
        } else if (op.GetOpcode() == Opcode::OP_SQRT) {
            EXPECT_EQ(sqrt.GetInputOperandSize(), kSizeOne);
            EXPECT_EQ(sqrt.GetInputOperand(kSizeZero), ubTensor2);
        }
    }
    EXPECT_EQ(operations.Contains(reshape1), false);
    EXPECT_EQ(reshape_num, kNumOne);
}

/*
RemoveSameReshape
inCast{8,16}->reshape->ubTensor{8,16}->sqrt->outCast{8,16}
inCast{8,16}->sqrt->outCast{8,16}
*/
TEST_F(TestRemoveRedundantReshapePass, RemoveRedundantReshapeUTest2) {
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestRemoveRedundantReshape", "TestRemoveRedundantReshape", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    // Prepare the graph
    std::vector<int64_t> shape = {kNumEight, kNumExpFour};
    auto inCast = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    auto ubTensor = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    auto outCast = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);

    currFunctionPtr->AddOperation(Opcode::OP_RESHAPE, {inCast}, {ubTensor});
    auto &sqrt = currFunctionPtr->AddOperation(Opcode::OP_SQRT, {ubTensor}, {outCast});

    currFunctionPtr->inCasts_.push_back(inCast);
    currFunctionPtr->outCasts_.push_back(outCast);

    RemoveRedundantReshape removeredundantpass;
    auto status = removeredundantpass.RunOnFunction(*currFunctionPtr);
    EXPECT_EQ(status, SUCCESS);

    uint32_t reshape_num = kNumZero;
    for (auto &op : currFunctionPtr->Operations()) {
        if (op.GetOpcode() == Opcode::OP_RESHAPE) {
            ++reshape_num;
        } else if (op.GetOpcode() == Opcode::OP_SQRT) {
            EXPECT_EQ(sqrt.GetInputOperandSize(), kSizeOne);
            EXPECT_EQ(sqrt.GetInputOperand(kSizeZero), inCast);
        }
    }
    EXPECT_EQ(reshape_num, kNumZero);
}

/*
RemoveReshapeChainSeveralConsumer(WARNING CASE)
inCast{8,16}->reshape->ubTensor{8,16}->sqrt->outCast1{8,16}
                                    ->exp->outCast2{8,16}
                                    ->reshape->outCast3{16,8}
inCast{8,16}->sqrt->outCast1{8,16}
            ->exp->outCast2{8,16}
*/
TEST_F(TestRemoveRedundantReshapePass, RemoveRedundantReshapeUTest3) {
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestRemoveRedundantReshape", "TestRemoveRedundantReshape", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    // Prepare the graph
    std::vector<int64_t> shape1 = {kNumEight, kNumExpFour};
    std::vector<int64_t> shape2 = {kNumExpFour, kNumEight};
    auto inCast = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape1);
    auto ubTensor = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape1);
    auto outCast1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape1);
    auto outCast2 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape1);
    auto outCast3 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape2);

    currFunctionPtr->AddOperation(Opcode::OP_RESHAPE, {inCast}, {ubTensor});
    auto &sqrt = currFunctionPtr->AddOperation(Opcode::OP_SQRT, {ubTensor}, {outCast1});
    auto &exp = currFunctionPtr->AddOperation(Opcode::OP_EXP, {ubTensor}, {outCast2});
    currFunctionPtr->AddOperation(Opcode::OP_RESHAPE, {ubTensor}, {outCast3});

    currFunctionPtr->inCasts_.push_back(inCast);
    currFunctionPtr->outCasts_.push_back(outCast1);
    currFunctionPtr->outCasts_.push_back(outCast2);
    currFunctionPtr->outCasts_.push_back(outCast3);

    RemoveRedundantReshape removeredundantpass;
    EXPECT_NE(removeredundantpass.PreCheck(*currFunctionPtr), SUCCESS);
    EXPECT_EQ(removeredundantpass.RunOnFunction(*currFunctionPtr), SUCCESS);

    uint32_t reshape_num = kNumZero;
    for (auto &op : currFunctionPtr->Operations()) {
        if (op.GetOpcode() == Opcode::OP_RESHAPE) {
            ++reshape_num;
        } else if (op.GetOpcode() == Opcode::OP_SQRT) {
            EXPECT_EQ(sqrt.GetInputOperandSize(), kSizeOne);
            EXPECT_EQ(sqrt.GetInputOperand(kSizeZero), inCast);
        } else if (op.GetOpcode() == Opcode::OP_EXP) {
            EXPECT_EQ(exp.GetInputOperandSize(), kSizeOne);
            EXPECT_EQ(exp.GetInputOperand(kSizeZero), inCast);
        }
    }
    EXPECT_EQ(reshape_num, kNumZero);
}

/*
RemoveReshapeChainSeveralConsumer
inCast{8,16}->reshape->ubTensor1{16,8}->exp->outCast1{16,8}
                                      ->reshape->ubTensor2{32,4}->sqrt->outCast2{32,4}
inCast{8,16}->reshape->ubTensor1{16,8}->exp->outCast1{16,8}
            ->reshape->ubTensor2{32,4}->sqrt->outCast2{32,4}
*/
TEST_F(TestRemoveRedundantReshapePass, RemoveRedundantReshapeUTest4) {
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestRemoveRedundantReshape", "TestRemoveRedundantReshape", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    // Prepare the graph
    std::vector<int64_t> shape1 = {kNumEight, kNumExpFour};
    std::vector<int64_t> shape2 = {kNumExpFour, kNumEight};
    std::vector<int64_t> shape3 = {kNumExpFive, kNumFour};
    auto inCast = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape1);
    auto ubTensor1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape2);
    auto outCast1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape2);
    auto ubTensor2 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape3);
    auto outCast2 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape3);

    auto &reshape1 = currFunctionPtr->AddOperation(Opcode::OP_RESHAPE, {inCast}, {ubTensor1});
    currFunctionPtr->AddOperation(Opcode::OP_EXP, {ubTensor1}, {outCast1});
    auto &reshape2 = currFunctionPtr->AddOperation(Opcode::OP_RESHAPE, {ubTensor1}, {ubTensor2});
    currFunctionPtr->AddOperation(Opcode::OP_SQRT, {ubTensor2}, {outCast2});

    currFunctionPtr->inCasts_.push_back(inCast);
    currFunctionPtr->outCasts_.push_back(outCast1);
    currFunctionPtr->outCasts_.push_back(outCast2);

    RemoveRedundantReshape removeredundantpass;
    auto status = removeredundantpass.RunOnFunction(*currFunctionPtr);
    EXPECT_EQ(status, SUCCESS);

    uint32_t reshape_num = kNumZero;
    for (auto &op : currFunctionPtr->Operations()) {
        if (op.GetOpcode() == Opcode::OP_RESHAPE) {
            ++reshape_num;
        }
    }
    EXPECT_EQ(reshape1.GetInputOperand(kSizeZero), inCast);
    EXPECT_EQ(reshape2.GetInputOperand(kSizeZero), inCast);
    EXPECT_EQ(reshape_num, kNumTwo);
}

/*
view->reshape->reshape  ->exp       ->reshape   ->reshape   ->assemble
                                                ->assemble
                                    ->assemble
                        ->assemble
             ->exp      ->reshape->assemble
             ->assemble
view->reshape  ->exp        ->reshape   ->assemble
                            ->assemble
                            ->assemble
               ->assemble
    ->reshape  ->exp        ->assemble
               ->assemble
*/
TEST_F(TestRemoveRedundantReshapePass, RemoveRedundantReshapeSTest1) {
    //Define the shape of the Tensors
    std::vector<int64_t> shape1 = {kNumExpSix, kNumExpSix};
    std::vector<int64_t> shape2 = {kNumExpFive, kNumExpSeven};
    std::vector<int64_t> shape3 = {kNumExpSeven, kNumExpFive};

    PassManager &passManager = PassManager::Instance();

    Tensor input(DT_FP32, shape1, "input");
    Tensor reshape1(DT_FP32, shape2, "reshape1");
    Tensor reshape2(DT_FP32, shape3, "reshape2");
    Tensor reshape3(DT_FP32, shape2, "reshape3");
    Tensor output1(DT_FP32, shape3, "output1");
    Tensor exp1(DT_FP32, shape3, "exp1");
    Tensor exp2(DT_FP32, shape2, "exp2");
    Tensor output2(DT_FP32, shape2, "output");

    FUNCTION("STCase1") {
        reshape1 = Reshape(input, shape2);
        reshape2 = Reshape(reshape1, shape3);
        TileShape::Current().SetVecTile({64, 64});
        exp1 = Exp(reshape2);
        reshape3 = Reshape(exp1, shape2);
        output1 = Reshape(reshape3, shape3);
        exp2 = Exp(reshape1);
        output2 = Reshape(exp2, shape2);
    }

    Function* func = Program::GetInstance().GetFunctionByRawName("TENSOR_STCase1");
    EXPECT_EQ(func->Operations().size(), kSizeThirteen);

    passManager.RegisterStrategy("RemoveRedundantReshapeTestStrategy", {
        {   "RemoveRedundantReshape",   "RemoveRedundantReshape"},
    });
    EXPECT_EQ(passManager.RunPass(Program::GetInstance(), *func, "RemoveRedundantReshapeTestStrategy"), SUCCESS);

    // ================== Verify the effect of the Pass ==================
    auto updated_operations = func->Operations();

    int reshape_num = kNumZero;
    EXPECT_EQ(updated_operations.size(), kSizeTwelve);
    for (const auto &op : updated_operations) {
        if (op.GetOpcode() == Opcode::OP_RESHAPE) {
            reshape_num++;
        }
    }
    EXPECT_EQ(reshape_num, kNumThree);
}
}
}