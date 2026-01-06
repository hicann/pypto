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

#define private public
#include "passes/tile_graph_pass/graph_optimization/split_reshape.h"

namespace npu {
namespace tile_fwk{
static const uint32_t kNumZero = 0u;
static const uint32_t kNumOne = 1u;
static const uint32_t kNumTwo = 2u;
static const uint32_t kNumThree = 3u;
static const uint32_t kNumFour = 4u;
static const uint32_t kNumSix = 6u;
static const uint32_t kNumEight = 8u;
static const uint32_t kNumTwelve = 12u;
static const uint32_t kExpFour = 16u;
static const uint32_t kExpFive = 32u;
static const uint32_t kExpSix = 64u;
static const uint32_t kNumNineSix = 96u;
static const uint32_t kExpSeven = 128u;
static const uint32_t kExpEight = 256u;
static const size_t kSizeZero = 0UL;
static const size_t kSizeOne = 1UL;
static const size_t kSizeTwo = 2UL;
static const size_t kSizeFour = 4UL;

class TestSplitReshapePass : public ::testing::Test {
public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}

    void SetUp() override {
        Program::GetInstance().Reset();
        config::Reset();
        config::SetPlatformConfig(KEY_ONLY_HOST_COMPILE, true);
        config::SetHostConfig(KEY_STRATEGY, "SplitReshapeTestStrategy");
        config::SetPlatformConfig("ENABLE_COST_MODEL", false);
    }
    void TearDown() override {}
};

TEST_F(TestSplitReshapePass, TestInit) {
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestReshapeSplit", "TestReshapeSplit", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    SplitReshape pass;
    auto status = pass.Init();
    EXPECT_EQ(status, SUCCESS);

    EXPECT_EQ(pass.AssembleOutToInput.size(), kSizeZero);
    EXPECT_EQ(pass.reshapeSources.size(), kSizeZero);
    EXPECT_EQ(pass.mapOffset.size(), kSizeZero);
    EXPECT_EQ(pass.assembles.size(), kSizeZero);
    EXPECT_EQ(pass.reshapes.size(), kSizeZero);
    EXPECT_EQ(pass.redundantViewops.size(), kSizeZero);
    EXPECT_EQ(pass.reshapeRawOutputs.size(), kSizeZero);
}

TEST_F(TestSplitReshapePass, TestCollectCopyOut) {
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestReshapeSplit", "TestReshapeSplit", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);
    // Prepare the graph

    std::vector<int64_t> shape = {kNumTwo, kNumOne, kNumEight};
    std::vector<int64_t> offset1 = {kNumZero, kNumZero, kNumZero};
    std::vector<int64_t> offset2 = {kNumOne, kNumZero, kNumZero};
    std::vector<int64_t> offset3 = {kNumTwo, kNumZero, kNumZero};
    std::vector<int64_t> shape1 = {kNumOne, kNumOne, kNumEight};
    std::vector<int64_t> shape2 = {kNumThree, kNumOne, kNumEight};
    std::vector<int64_t> shape3 = {kNumThree, kNumEight};

    std::shared_ptr<RawTensor> ddrRawTensor = std::make_shared<RawTensor>(DT_FP32, shape);
    auto input1 = std::make_shared<LogicalTensor>(*currFunctionPtr, ddrRawTensor, offset1, shape1);
    auto input2 = std::make_shared<LogicalTensor>(*currFunctionPtr, ddrRawTensor, offset2, shape1);
    auto input3 = std::make_shared<LogicalTensor>(*currFunctionPtr, ddrRawTensor, offset3, shape1);
    auto copyTensor = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape1);
    auto ubTensor = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape2);
    auto output = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape3);

    auto &assemble_op1 = currFunctionPtr->AddOperation(Opcode::OP_ASSEMBLE, {input1}, {ubTensor});
    auto assemble_Attr1 = std::make_shared<AssembleOpAttribute>(MEM_DEVICE_DDR, offset1);
    assemble_op1.SetOpAttribute(assemble_Attr1);

    auto &assemble_op2 = currFunctionPtr->AddOperation(Opcode::OP_ASSEMBLE, {input2}, {ubTensor});
    auto assemble_Attr2 = std::make_shared<AssembleOpAttribute>(MEM_DEVICE_DDR, offset2);
    assemble_op2.SetOpAttribute(assemble_Attr2);

    currFunctionPtr->AddOperation(Opcode::OP_COPY_OUT, {input3}, {copyTensor});
    
    auto &assemble_op3 = currFunctionPtr->AddOperation(Opcode::OP_ASSEMBLE, {copyTensor}, {ubTensor});
    auto assemble_Attr3 = std::make_shared<AssembleOpAttribute>(MEM_DEVICE_DDR, offset3);
    assemble_op3.SetOpAttribute(assemble_Attr3);

    std::vector<SymbolicScalar> validShape = {SymbolicScalar("a"), kNumEight};
    auto &reshape_op = currFunctionPtr->AddOperation(Opcode::OP_RESHAPE, {ubTensor}, {output});
    reshape_op.SetAttribute(OP_ATTR_PREFIX + "validShape", validShape);

    SplitReshape pass;
    auto status = pass.CollectCopyOut(*currFunctionPtr);
    EXPECT_EQ(status, SUCCESS);

    EXPECT_EQ(pass.reshapeSources.size(), kSizeOne);
    auto iter1 = pass.reshapeSources.find(output->tensor->rawmagic);
    EXPECT_NE(iter1, pass.reshapeSources.end());
    EXPECT_EQ(iter1->second, ubTensor);

    EXPECT_EQ(pass.reshapeDynOutput.size(), kSizeOne);
    auto iter2 = pass.reshapeDynOutput.find(output->tensor->rawmagic);
    EXPECT_NE(iter2, pass.reshapeDynOutput.end());
    for (size_t i = 0; i < kSizeTwo; ++i) {
        EXPECT_EQ(iter2->second[i].Dump(), validShape[i].Dump());
    }

    EXPECT_EQ(pass.AssembleOutToInput.size(), kSizeOne);
    auto iter3 = pass.AssembleOutToInput.find(ubTensor->tensor->rawmagic);
    EXPECT_NE(iter3, pass.AssembleOutToInput.end());
    EXPECT_EQ(iter3->second.size(), kNumTwo);
    EXPECT_EQ(iter3->second.count(input1), kNumOne);
    EXPECT_EQ(iter3->second.count(input2), kNumOne);

    EXPECT_EQ(pass.mapOffset.size(), kSizeTwo);
    EXPECT_EQ(pass.mapOffset.count(copyTensor->magic), kNumZero);
    EXPECT_EQ(pass.mapOffset.count(input1->magic), kNumOne);
    EXPECT_EQ(pass.mapOffset[input1->magic].count(ubTensor->magic), kNumOne);
    EXPECT_EQ(pass.mapOffset[input1->magic][ubTensor->magic], offset1);
    EXPECT_EQ(pass.mapOffset.count(input2->magic), kNumOne);
    EXPECT_EQ(pass.mapOffset[input2->magic].count(ubTensor->magic), kNumOne);
    EXPECT_EQ(pass.mapOffset[input2->magic][ubTensor->magic], offset2);
}

TEST_F(TestSplitReshapePass, TestCheckSplit) {
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestReshapeSplit", "TestReshapeSplit", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);
    // Prepare the graph

    std::vector<int64_t> shape = {kNumTwo, kNumOne, kNumEight};
    std::vector<int64_t> offset1 = {kNumZero, kNumZero, kNumZero};
    std::vector<int64_t> offset2 = {kNumOne, kNumZero, kNumZero};
    std::vector<int64_t> shape1 = {kNumOne, kNumOne, kNumEight};
    std::vector<int64_t> shape2 = {kNumTwo, kNumOne, kNumEight};
    std::vector<int64_t> shape3 = {kNumTwo, kNumEight};

    std::shared_ptr<RawTensor> ddrRawTensor1 = std::make_shared<RawTensor>(DT_FP32, shape);
    std::shared_ptr<RawTensor> ddrRawTensor2 = std::make_shared<RawTensor>(DT_FP32, shape);

    auto case1Input1 = std::make_shared<LogicalTensor>(*currFunctionPtr, ddrRawTensor1, offset1, shape1);
    auto case1Input2 = std::make_shared<LogicalTensor>(*currFunctionPtr, ddrRawTensor1, offset2, shape1);
    auto case1UbTensor = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape2);
    auto case1Output = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape3);
    auto &assemble_op1 = currFunctionPtr->AddOperation(Opcode::OP_ASSEMBLE, {case1Input1}, {case1UbTensor});
    auto assemble_Attr1 = std::make_shared<AssembleOpAttribute>(MEM_DEVICE_DDR, offset1);
    assemble_op1.SetOpAttribute(assemble_Attr1);
    auto &assemble_op2 = currFunctionPtr->AddOperation(Opcode::OP_ASSEMBLE, {case1Input2}, {case1UbTensor});
    auto assemble_Attr2 = std::make_shared<AssembleOpAttribute>(MEM_DEVICE_DDR, offset2);
    assemble_op2.SetOpAttribute(assemble_Attr2);
    currFunctionPtr->AddOperation(Opcode::OP_RESHAPE, {case1UbTensor}, {case1Output});

    auto case2Input = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape2);
    auto case2Output = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape3);
    currFunctionPtr->AddOperation(Opcode::OP_RESHAPE, {case2Input}, {case2Output});

    auto case3Input1 = std::make_shared<LogicalTensor>(*currFunctionPtr, ddrRawTensor1, offset1, shape1);
    auto case3Input2 = std::make_shared<LogicalTensor>(*currFunctionPtr, ddrRawTensor2, offset2, shape1);
    auto case3UbTensor = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape2);
    auto case3Output = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape3);
    auto &assemble_op3 = currFunctionPtr->AddOperation(Opcode::OP_ASSEMBLE, {case3Input1}, {case3UbTensor});
    auto assemble_Attr3 = std::make_shared<AssembleOpAttribute>(MEM_DEVICE_DDR, offset1);
    assemble_op3.SetOpAttribute(assemble_Attr3);
    auto &assemble_op4 = currFunctionPtr->AddOperation(Opcode::OP_ASSEMBLE, {case3Input2}, {case3UbTensor});
    auto assemble_Attr4 = std::make_shared<AssembleOpAttribute>(MEM_DEVICE_DDR, offset2);
    assemble_op4.SetOpAttribute(assemble_Attr4);
    currFunctionPtr->AddOperation(Opcode::OP_RESHAPE, {case3UbTensor}, {case3Output});

    auto case4Input1 = std::make_shared<LogicalTensor>(*currFunctionPtr, ddrRawTensor1, offset1, shape1);
    auto case4UbTensor = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape2);
    auto case4Output = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape3);
    auto &assemble_op5 = currFunctionPtr->AddOperation(Opcode::OP_ASSEMBLE, {case4Input1}, {case4UbTensor});
    auto assemble_Attr5 = std::make_shared<AssembleOpAttribute>(MEM_DEVICE_DDR, offset1);
    assemble_op5.SetOpAttribute(assemble_Attr5);
    currFunctionPtr->AddOperation(Opcode::OP_RESHAPE, {case4UbTensor}, {case4Output});

    SplitReshape pass;
    auto status = pass.CollectCopyOut(*currFunctionPtr);
    EXPECT_EQ(status, SUCCESS);
    EXPECT_EQ(pass.CheckSameRawInput(case1UbTensor), true);
    EXPECT_EQ(pass.CheckSameRawInput(case2Input), true);
    EXPECT_EQ(pass.CheckSameRawInput(case3UbTensor), false);
    EXPECT_EQ(pass.CheckSameRawInput(case4UbTensor), true);
}

TEST_F(TestSplitReshapePass, TestCheckDynStatus) {
    std::vector<int64_t> input;
    std::vector<int64_t> output;
    std::vector<int64_t> alignedShape;
    std::vector<SymbolicScalar> dynOutput;
    SplitReshape pass;

    input = {kNumFour, kNumTwo};
    output = {kNumTwo, kNumFour};
    alignedShape = {kNumTwo, kNumTwo, kNumTwo};
    dynOutput = {SymbolicScalar("a"), kNumFour};
    EXPECT_EQ(pass.CheckDynStatus(alignedShape, input, output, dynOutput), WARNING);

    input = {kNumFour, kNumTwo};
    output = {kNumTwo, kNumFour};
    alignedShape = {kNumTwo, kNumTwo, kNumTwo};
    dynOutput = {kNumTwo, SymbolicScalar("a")};
    EXPECT_EQ(pass.CheckDynStatus(alignedShape, input, output, dynOutput), WARNING);

    input = {kNumFour, kNumTwo};
    output = {kNumTwo, kNumFour};
    alignedShape = {kNumTwo, kNumTwo, kNumTwo};
    dynOutput = {kNumTwo, kNumOne, SymbolicScalar("a")};
    EXPECT_EQ(pass.CheckDynStatus(alignedShape, input, output, dynOutput), FAILED);

    input = {kNumTwo, kNumOne, kNumFour};
    output = {kNumTwo, kNumOne, kNumTwo, kNumTwo};
    alignedShape = {kNumTwo, kNumTwo, kNumTwo};
    dynOutput = {SymbolicScalar("a"), kNumOne, kNumTwo, kNumTwo};
    EXPECT_EQ(pass.CheckDynStatus(alignedShape, input, output, dynOutput), SUCCESS);

    input = {kNumTwo, kNumOne, kNumTwo, kNumTwo};
    output = {kNumTwo, kNumOne, kNumFour};
    alignedShape = {kNumTwo, kNumTwo, kNumTwo};
    dynOutput = {SymbolicScalar("a"), kNumOne, kNumFour};
    EXPECT_EQ(pass.CheckDynStatus(alignedShape, input, output, dynOutput), SUCCESS);

    input = {kNumTwo, kNumOne, kNumTwo, kNumTwo};
    output = {kNumTwo, kNumOne, kNumFour};
    alignedShape = {kNumTwo, kNumTwo, kNumTwo};
    dynOutput = {kNumTwo, kNumOne, SymbolicScalar("a")};
    EXPECT_EQ(pass.CheckDynStatus(alignedShape, input, output, dynOutput), WARNING);

    input = {kNumTwo, kNumOne, kNumTwo, kNumTwo};
    output = {kNumTwo, kNumOne, kNumFour};
    alignedShape = {kNumTwo, kNumTwo, kNumTwo};
    dynOutput = {kNumTwo, SymbolicScalar("a"), kNumFour};
    EXPECT_EQ(pass.CheckDynStatus(alignedShape, input, output, dynOutput), SUCCESS);
}

TEST_F(TestSplitReshapePass, TestShapeAlign) {
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestReshapeSplit", "TestReshapeSplit", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);
    // Prepare the graph
    SplitReshape pass;
    Status status;
    std::vector<int64_t> inputShape;
    std::vector<int64_t> outputShape;
    std::vector<int64_t> alignedShape;
    std::vector<int64_t> expectedShape;

    alignedShape.clear();
    inputShape = {kExpSix, kExpSix};
    outputShape = {kExpFive, kExpSeven};
    expectedShape = {kExpFive, kNumTwo, kExpSix};
    status = pass.ShapeAlign(inputShape, outputShape, alignedShape);
    EXPECT_EQ(status, SUCCESS);
    EXPECT_EQ(alignedShape, expectedShape);

    alignedShape.clear();
    inputShape = {kNumTwo, kExpFive, kExpSix, kExpSeven};
    outputShape = {kExpSix, kExpSix, kExpSeven};
    expectedShape = inputShape;
    status = pass.ShapeAlign(inputShape, outputShape, alignedShape);
    EXPECT_EQ(status, SUCCESS);
    EXPECT_EQ(alignedShape, expectedShape);

    alignedShape.clear();
    inputShape = {kExpSix, kExpSix, kExpSeven};
    outputShape = {kNumTwo, kExpFive, kExpSix, kExpSeven};
    expectedShape = outputShape;
    status = pass.ShapeAlign(inputShape, outputShape, alignedShape);
    EXPECT_EQ(status, SUCCESS);
    EXPECT_EQ(alignedShape, expectedShape);

    inputShape = {kNumNineSix, kNumFour};
    outputShape = {kExpSix, kNumSix};
    status = pass.ShapeAlign(inputShape, outputShape, alignedShape);
    EXPECT_EQ(status, WARNING);
}

TEST_F(TestSplitReshapePass, TestRawToAlign) {
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestReshapeSplit", "TestReshapeSplit", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);
    // Prepare the graph
    SplitReshape pass;
    Status status;
    std::vector<int64_t> rawShape;
    std::vector<int64_t> alignedShape;
    std::vector<int64_t> tileOffset;
    std::vector<int64_t> tileShape;
    std::vector<int64_t> expectOffset;
    std::vector<int64_t> expectShape;
    std::vector<int64_t> newOffset;
    std::vector<int64_t> newShape;

    ReshapeTilePara shapePara;

    rawShape = {kNumEight};
    alignedShape = {kNumTwo, kNumTwo, kNumTwo};
    tileOffset = {kNumTwo};
    tileShape = {kNumTwo};
    shapePara = {rawShape, alignedShape, tileOffset, tileShape};
    status = pass.RawToAlign(shapePara, newOffset, newShape);
    EXPECT_EQ(status, SUCCESS);
    expectOffset = {kNumZero, kNumOne, kNumZero};
    expectShape = {kNumOne, kNumOne, kNumTwo};
    EXPECT_EQ(newShape, expectShape);
    EXPECT_EQ(newOffset, expectOffset);

    rawShape = {kExpFive, kNumTwo, kExpSeven};
    alignedShape = {kExpFive, kNumTwo, kNumFour, kExpFive};
    tileOffset = {kNumZero, kNumZero, kNumZero};
    tileShape = {kNumOne, kNumOne, kExpSix};
    shapePara = {rawShape, alignedShape, tileOffset, tileShape};
    status = pass.RawToAlign(shapePara, newOffset, newShape);
    EXPECT_EQ(status, SUCCESS);
    expectOffset = {kNumZero, kNumZero, kNumZero, kNumZero};
    expectShape = {kNumOne, kNumOne, kNumTwo, kExpFive};
    EXPECT_EQ(newShape, expectShape);
    EXPECT_EQ(newOffset, expectOffset);

    rawShape = {kExpFive, kNumTwo, kExpEight};
    alignedShape = {kExpFive, kNumTwo, kNumEight, kExpFive};
    tileOffset = {kNumZero, kNumOne, kExpSeven};
    tileShape = {kNumOne, kNumOne, kExpSix};
    shapePara = {rawShape, alignedShape, tileOffset, tileShape};
    status = pass.RawToAlign(shapePara, newOffset, newShape);
    EXPECT_EQ(status, SUCCESS);
    expectOffset = {kNumZero, kNumOne, kNumFour, kNumZero};
    expectShape = {kNumOne, kNumOne, kNumTwo, kExpFive};
    EXPECT_EQ(newShape, expectShape);
    EXPECT_EQ(newOffset, expectOffset);

    rawShape = {kExpFive, kNumFour, kExpEight};
    alignedShape = {kExpFive, kNumFour, kNumEight, kExpFive};
    tileOffset = {kNumOne, kNumOne, kExpSeven};
    tileShape = {kNumOne, kNumTwo, kExpSix};
    shapePara = {rawShape, alignedShape, tileOffset, tileShape};
    status = pass.RawToAlign(shapePara, newOffset, newShape);
    EXPECT_EQ(status, SUCCESS);
    expectOffset = {kNumOne, kNumOne, kNumFour, kNumZero};
    expectShape = {kNumOne, kNumTwo, kNumTwo, kExpFive};
    EXPECT_EQ(newShape, expectShape);
    EXPECT_EQ(newOffset, expectOffset);

    rawShape = {kExpFive, kNumFour, kNumNineSix};
    alignedShape = {kExpFive, kNumFour, kNumFour, kExpFive};
    tileOffset = {kNumOne, kNumOne, kExpSeven};
    tileShape = {kNumOne, kNumTwo, kExpSix};
    shapePara = {rawShape, alignedShape, tileOffset, tileShape};
    status = pass.RawToAlign(shapePara, newOffset, newShape);
    EXPECT_EQ(status, WARNING);
}

TEST_F(TestSplitReshapePass, TestAlignToRaw) {
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestReshapeSplit", "TestReshapeSplit", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);
    // Prepare the graph
    SplitReshape pass;
    Status status;
    std::vector<int64_t> alignedShape;
    std::vector<int64_t> rawShape;
    std::vector<int64_t> tileOffset;
    std::vector<int64_t> tileShape;
    std::vector<int64_t> expectOffset;
    std::vector<int64_t> expectShape;
    std::vector<int64_t> newOffset;
    std::vector<int64_t> newShape;

    ReshapeTilePara shapePara;

    rawShape = {kNumSix};
    alignedShape = {kNumTwo, kNumThree};
    tileOffset = {kNumOne, kNumZero};
    tileShape = {kNumOne, kNumThree};
    shapePara = {alignedShape, rawShape, tileOffset, tileShape};
    status = pass.AlignToRaw(shapePara, newOffset, newShape);
    EXPECT_EQ(status, SUCCESS);
    expectOffset = {kNumThree};
    expectShape = {kNumThree};
    EXPECT_EQ(newShape, expectShape);
    EXPECT_EQ(newOffset, expectOffset);

    rawShape = {kExpFive, kNumTwo, kExpSeven};
    alignedShape = {kExpFive, kNumTwo, kNumFour, kExpFive};
    tileOffset = {kNumZero, kNumZero, kNumZero, kNumZero};
    tileShape = {kNumOne, kNumOne, kNumTwo, kExpFive};
    shapePara = {alignedShape, rawShape, tileOffset, tileShape};
    status = pass.AlignToRaw(shapePara, newOffset, newShape);
    EXPECT_EQ(status, SUCCESS);
    expectOffset = {kNumZero, kNumZero, kNumZero};
    expectShape = {kNumOne, kNumOne, kExpSix};
    EXPECT_EQ(newShape, expectShape);
    EXPECT_EQ(newOffset, expectOffset);

    rawShape = {kExpFive, kNumTwo, kExpEight};
    alignedShape = {kExpFive, kNumTwo, kNumEight, kExpFive};
    tileOffset = {kNumZero, kNumOne, kNumFour, kNumZero};
    tileShape = {kNumOne, kNumOne, kNumTwo, kExpFive};
    shapePara = {alignedShape, rawShape, tileOffset, tileShape};
    status = pass.AlignToRaw(shapePara, newOffset, newShape);
    EXPECT_EQ(status, SUCCESS);
    expectOffset = {kNumZero, kNumOne, kExpSeven};
    expectShape = {kNumOne, kNumOne, kExpSix};
    EXPECT_EQ(newShape, expectShape);
    EXPECT_EQ(newOffset, expectOffset);

    rawShape = {kExpFive, kNumFour, kExpEight};
    alignedShape = {kExpFive, kNumFour, kNumEight, kExpFive};
    tileOffset = {kNumOne, kNumOne, kNumFour, kNumZero};
    tileShape = {kNumOne, kNumTwo, kNumTwo, kExpFive};
    shapePara = {alignedShape, rawShape, tileOffset, tileShape};
    status = pass.AlignToRaw(shapePara, newOffset, newShape);
    EXPECT_EQ(status, SUCCESS);
    expectOffset = {kNumOne, kNumOne, kExpSeven};
    expectShape = {kNumOne, kNumTwo, kExpSix};
    EXPECT_EQ(newShape, expectShape);
    EXPECT_EQ(newOffset, expectOffset);
}

TEST_F(TestSplitReshapePass, TestAlignToRawSpecialCase) {
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestReshapeSplit", "TestReshapeSplit", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);
    // Prepare the graph
    SplitReshape pass;
    Status status;
    std::vector<int64_t> alignedShape;
    std::vector<int64_t> rawShape;
    std::vector<int64_t> tileOffset;
    std::vector<int64_t> tileShape;
    std::vector<int64_t> newOffset;
    std::vector<int64_t> newShape;

    ReshapeTilePara shapePara;

    rawShape = {kNumEight};
    alignedShape = {kNumTwo, kNumTwo, kNumTwo};
    tileOffset = {kNumZero, kNumTwo, kNumOne};
    tileShape = {kNumTwo, kNumTwo, kNumTwo};
    shapePara = {alignedShape, rawShape, tileOffset, tileShape};
    status = pass.AlignToRaw(shapePara, newOffset, newShape);
    EXPECT_EQ(status, SUCCESS);
    EXPECT_TRUE(newShape.empty());
    EXPECT_TRUE(newOffset.empty());

    rawShape = {kExpFive, kNumTwo, kExpEight};
    alignedShape = {kExpFive, kNumTwo, kNumEight, kExpFive};
    tileShape = {kNumOne, kNumOne, kNumTwo, kExpFive};
    tileOffset = {kNumZero, kNumOne, kNumZero, kExpFive};
    shapePara = {alignedShape, rawShape, tileOffset, tileShape};
    status = pass.AlignToRaw(shapePara, newOffset, newShape);
    EXPECT_EQ(status, SUCCESS);
    EXPECT_TRUE(newShape.empty());
    EXPECT_TRUE(newOffset.empty());
}

/*
多对一场景
rawShape = {2, 4}
{2, 4} -> assemble -> {2, 4} -> reshape -> {2, 2, 2}
*/
TEST_F(TestSplitReshapePass, TestObtainCopyOutTileBeCovered) {
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestReshapeSplit", "TestReshapeSplit", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);
    // Prepare the graph

    std::vector<int64_t> shape = {kNumTwo, kNumFour};
    std::vector<int64_t> offset1 = {kNumZero, kNumZero};
    std::vector<int64_t> offset2 = {kNumZero, kNumTwo};
    std::vector<int64_t> shape1 = {kNumTwo, kNumTwo};
    std::vector<int64_t> shape2 = {kNumTwo, kNumFour};
    std::vector<int64_t> shape3 = {kNumTwo, kNumTwo, kNumTwo};

    std::shared_ptr<RawTensor> ddrRawTensor = std::make_shared<RawTensor>(DT_FP32, shape);
    auto input1 = std::make_shared<LogicalTensor>(*currFunctionPtr, ddrRawTensor, offset1, shape1);
    auto input2 = std::make_shared<LogicalTensor>(*currFunctionPtr, ddrRawTensor, offset2, shape1);
    auto ubTensor = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape2);
    auto output = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape3);

    auto &assemble_op1 = currFunctionPtr->AddOperation(Opcode::OP_ASSEMBLE, {input1}, {ubTensor});
    auto assemble_Attr1 = std::make_shared<AssembleOpAttribute>(MEM_DEVICE_DDR, offset1);
    assemble_op1.SetOpAttribute(assemble_Attr1);
    auto &assemble_op2 = currFunctionPtr->AddOperation(Opcode::OP_ASSEMBLE, {input2}, {ubTensor});
    auto assemble_Attr2 = std::make_shared<AssembleOpAttribute>(MEM_DEVICE_DDR, offset2);
    assemble_op2.SetOpAttribute(assemble_Attr2);
    currFunctionPtr->AddOperation(Opcode::OP_RESHAPE, {ubTensor}, {output});

    SplitReshape pass;
    LogicalTensors overlaps;
    LogicalTensors newOverlaps;
    std::vector<SymbolicScalar> validShape;
    std::vector<int64_t> newOutputTileOffset = {kNumZero, kNumZero, kNumZero};
    std::vector<int64_t> newOutputTileShape = {kNumTwo, kNumTwo, kNumTwo};
    std::vector<int64_t> alignedShape = {kNumTwo, kNumTwo, kNumTwo};
    auto newOutput = std::make_shared<LogicalTensor>(*currFunctionPtr, output->tensor, newOutputTileOffset, newOutputTileShape, validShape);
    copyOutTilePara copyOutTile = {ubTensor, output, newOutput, alignedShape};
    EXPECT_EQ(pass.CollectCopyOut(*currFunctionPtr), SUCCESS);
    EXPECT_EQ(pass.ObtainCopyOutTile(*currFunctionPtr, copyOutTile, overlaps, newOverlaps), SUCCESS);
    EXPECT_EQ(overlaps.size(), kSizeTwo);
    EXPECT_NE(std::find(overlaps.begin(), overlaps.end(), input1), overlaps.end());
    EXPECT_NE(std::find(overlaps.begin(), overlaps.end(), input2), overlaps.end());
    EXPECT_EQ(newOverlaps.size(), kSizeTwo);
}

/*
一对一场景
rawShape = {2, 2, 2}
{2, 2, 2} -> assemble -> {2, 2, 2} -> reshape -> {4, 2}
*/
TEST_F(TestSplitReshapePass, TestObtainCopyOutTilePerfectlyMatched) {
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestReshapeSplit", "TestReshapeSplit", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);
    // Prepare the graph

    std::vector<int64_t> shape = {kNumTwo, kNumTwo, kNumTwo};
    std::vector<int64_t> offset = {kNumZero, kNumZero, kNumZero};
    std::vector<int64_t> shape1 = {kNumTwo, kNumTwo, kNumTwo};
    std::vector<int64_t> shape2 = {kNumFour, kNumTwo};

    std::shared_ptr<RawTensor> ddrRawTensor = std::make_shared<RawTensor>(DT_FP32, shape);
    auto input = std::make_shared<LogicalTensor>(*currFunctionPtr, ddrRawTensor, offset, shape1);
    auto ubTensor = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape1);
    auto output = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape2);

    auto &assemble_op = currFunctionPtr->AddOperation(Opcode::OP_ASSEMBLE, {input}, {ubTensor});
    auto assemble_Attr = std::make_shared<AssembleOpAttribute>(MEM_DEVICE_DDR, offset);
    assemble_op.SetOpAttribute(assemble_Attr);
    currFunctionPtr->AddOperation(Opcode::OP_RESHAPE, {ubTensor}, {output});

    SplitReshape pass;
    LogicalTensors overlaps;
    LogicalTensors newOverlaps;
    std::vector<SymbolicScalar> validShape;
    std::vector<int64_t> newOutputTileOffset = {kNumZero, kNumZero, kNumZero};
    std::vector<int64_t> newOutputTileShape = {kNumTwo, kNumTwo, kNumTwo};
    std::vector<int64_t> alignedShape = {kNumTwo, kNumTwo, kNumTwo};
    auto newOutput = std::make_shared<LogicalTensor>(*currFunctionPtr, output->tensor, newOutputTileOffset, newOutputTileShape, validShape);
    copyOutTilePara copyOutTile = {ubTensor, output, newOutput, alignedShape};
    EXPECT_EQ(pass.CollectCopyOut(*currFunctionPtr), SUCCESS);
    EXPECT_EQ(pass.ObtainCopyOutTile(*currFunctionPtr, copyOutTile, overlaps, newOverlaps), SUCCESS);
    EXPECT_EQ(overlaps.size(), kSizeOne);
    EXPECT_NE(std::find(overlaps.begin(), overlaps.end(), input), overlaps.end());
    EXPECT_EQ(newOverlaps.size(), kSizeOne);
    EXPECT_EQ(newOverlaps[0]->GetOffset(), newOutputTileOffset);
    EXPECT_EQ(newOverlaps[0]->GetShape(), newOutputTileShape);
}

/*
验证一对一场景下数据的处理
rawShape = {2, 2, 2}
{2, 2, 2}(ddr) -> assemble -> {2, 2, 2}(unknown) -> reshape -> {4, 2}(unknown) -> view -> {4,2}(ddr) -> OP
{2, 2, 2}(ddr) -> assemble -> {2, 2, 2}(unknown) -> reshape(一个ReshapeOp成员) -> {4, 2}(unknown) -> view -> {4,2}(ddr) -> OP
*/
TEST_F(TestSplitReshapePass, TestUpdateForPerfectlyMatch) {
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestReshapeSplit", "TestReshapeSplit", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);
    // Prepare the graph

    std::vector<int64_t> shape = {kNumTwo, kNumTwo, kNumTwo};
    std::vector<int64_t> offset = {kNumZero, kNumZero, kNumZero};
    std::vector<int64_t> shape1 = {kNumTwo, kNumTwo, kNumTwo};
    std::vector<int64_t> shape2 = {kNumFour, kNumTwo};

    std::shared_ptr<RawTensor> ddrRawTensor = std::make_shared<RawTensor>(DT_FP32, shape);
    auto input = std::make_shared<LogicalTensor>(*currFunctionPtr, ddrRawTensor, offset, shape1);
    input->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, false);
    auto ubTensor1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape1);
    ubTensor1->SetMemoryTypeOriginal(MemoryType::MEM_UNKNOWN, false);
    auto ubTensor2 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape2);
    ubTensor2->SetMemoryTypeOriginal(MemoryType::MEM_UNKNOWN, false);
    auto output = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape2);
    output->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, false);

    auto &assemble_op = currFunctionPtr->AddOperation(Opcode::OP_ASSEMBLE, {input}, {ubTensor1});
    auto assemble_Attr = std::make_shared<AssembleOpAttribute>(MEM_DEVICE_DDR, offset);
    assemble_op.SetOpAttribute(assemble_Attr);
    currFunctionPtr->AddOperation(Opcode::OP_RESHAPE, {ubTensor1}, {ubTensor2});
    auto &view_op = currFunctionPtr->AddOperation(Opcode::OP_VIEW, {ubTensor2}, {output});
    std::vector<int64_t> view_offset = {0, 0};
    auto view_Attr = std::make_shared<ViewOpAttribute>(view_offset);
    view_op.SetOpAttribute(view_Attr);

    CalcOverlapPara para;
    std::vector<SymbolicScalar> validShape;
    std::vector<int64_t> newTileOffset = {kNumZero, kNumZero, kNumZero};
    std::vector<int64_t> newTileShape = {kNumTwo, kNumTwo, kNumTwo};
    para.alignedShape = {kNumTwo, kNumTwo, kNumTwo};
    para.overlaps = {input};
    para.newOverlaps = {std::make_shared<LogicalTensor>(*currFunctionPtr, input->tensor, newTileOffset, newTileShape, validShape)};
    para.reshapeSource = ubTensor1;
    para.input = ubTensor2;
    para.output = output;
    para.inputView = ubTensor2;

    SplitReshape pass;
    EXPECT_EQ(pass.CollectCopyOut(*currFunctionPtr), SUCCESS);
    EXPECT_EQ(pass.ProcessOnetoOne(*currFunctionPtr, view_op, para), SUCCESS);
    auto newReshapeOutput = view_op.GetInputOperand(kSizeZero);
    EXPECT_NE(newReshapeOutput, ubTensor2);
    EXPECT_EQ(newReshapeOutput->GetMemoryTypeOriginal(), MemoryType::MEM_UNKNOWN);
    EXPECT_EQ(pass.reshapes.size(), kSizeOne);
    auto reshape = pass.reshapes.begin()->second;
    EXPECT_EQ(pass.assembles.size(), kSizeOne);
    auto assemble = pass.assembles.begin();
    auto newReshapeSource = reshape->input;
    EXPECT_EQ(reshape->output, newReshapeOutput);
    EXPECT_EQ(newReshapeSource->GetMemoryTypeOriginal(), MemoryType::MEM_UNKNOWN);
    auto viewOpAttribute = dynamic_cast<ViewOpAttribute *>(view_op.GetOpAttribute().get());
    EXPECT_EQ(viewOpAttribute->GetFromOffset(), view_offset);
    EXPECT_EQ(assemble->from, MemoryType::MEM_DEVICE_DDR);
    EXPECT_EQ(assemble->toOffset, offset);
    EXPECT_EQ(assemble->input, input);
    EXPECT_EQ(assemble->output, newReshapeSource);
}

/*
验证一对一场景下动态shape数据的处理
rawShape = {2, 2, 2}
{2, 2, 2}(ddr) -> assemble -> {2, 2, 2}(unknown) -> reshape -> {4, 2}(unknown) -> view -> {4,2}(ddr) -> OP
                                                     {4, a}
{2, 2, 2}(ddr) -> assemble -> {2, 2, 2}(unknown) -> reshape(一个ReshapeOp成员) -> {4, 2}(unknown) -> view -> {4,2}(ddr) -> OP
*/
TEST_F(TestSplitReshapePass, TestDynUpdateForPerfectlyMatch) {
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestReshapeSplit", "TestReshapeSplit", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);
    // Prepare the graph

    std::vector<int64_t> shape = {kNumTwo, kNumTwo, kNumTwo};
    std::vector<int64_t> offset = {kNumZero, kNumZero, kNumZero};
    std::vector<int64_t> shape1 = {kNumTwo, kNumTwo, kNumTwo};
    std::vector<int64_t> shape2 = {kNumFour, kNumTwo};
    std::vector<int64_t> view_offset = {kNumZero, kNumZero};
    std::vector<SymbolicScalar> validShape = {kNumFour, SymbolicScalar("a")};

    std::shared_ptr<RawTensor> ddrRawTensor = std::make_shared<RawTensor>(DT_FP32, shape);
    auto input = std::make_shared<LogicalTensor>(*currFunctionPtr, ddrRawTensor, offset, shape1);
    input->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, false);
    auto ubTensor1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape1);
    ubTensor1->SetMemoryTypeOriginal(MemoryType::MEM_UNKNOWN, false);
    auto ubTensor2 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape2);
    ubTensor2->SetMemoryTypeOriginal(MemoryType::MEM_UNKNOWN, false);
    auto output = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape2);
    output->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, false);

    auto &assemble_op = currFunctionPtr->AddOperation(Opcode::OP_ASSEMBLE, {input}, {ubTensor1});
    auto assemble_Attr = std::make_shared<AssembleOpAttribute>(MEM_DEVICE_DDR, offset);
    assemble_op.SetOpAttribute(assemble_Attr);
    currFunctionPtr->AddOperation(Opcode::OP_RESHAPE, {ubTensor1}, {ubTensor2});
    auto &view_op = currFunctionPtr->AddOperation(Opcode::OP_VIEW, {ubTensor2}, {output});
    auto view_Attr = std::make_shared<ViewOpAttribute>(view_offset);
    view_op.SetOpAttribute(view_Attr);

    CalcOverlapPara para;
    std::vector<int64_t> newTileOffset = {kNumZero, kNumZero, kNumZero};
    std::vector<int64_t> newTileShape = {kNumTwo, kNumTwo, kNumTwo};
    para.alignedShape = {kNumTwo, kNumTwo, kNumTwo};
    para.overlaps = {input};
    auto newOverlap = std::make_shared<LogicalTensor>(*currFunctionPtr, input->tensor, newTileOffset, newTileShape);
    para.newOverlaps = {newOverlap};
    para.reshapeSource = ubTensor1;
    para.input = ubTensor2;
    para.output = output;
    para.inputView = ubTensor2;
    auto newValidShape = GetViewValidShape(validShape, view_offset, {}, shape2);
    para.oriViewDynShape = newValidShape;

    SplitReshape pass;
    EXPECT_EQ(pass.CollectCopyOut(*currFunctionPtr), SUCCESS);
    EXPECT_EQ(pass.ProcessOnetoOne(*currFunctionPtr, view_op, para), SUCCESS);
    auto newReshapeOutput = view_op.GetInputOperand(kSizeZero);
    EXPECT_NE(newReshapeOutput, ubTensor2);
    EXPECT_EQ(newReshapeOutput->GetMemoryTypeOriginal(), MemoryType::MEM_UNKNOWN);
    EXPECT_EQ(pass.reshapes.size(), kSizeOne);
    auto reshape = pass.reshapes.begin()->second;
    EXPECT_EQ(pass.assembles.size(), kSizeOne);
    auto assemble = pass.assembles.begin();
    auto newReshapeSource = reshape->input;
    EXPECT_EQ(reshape->output, newReshapeOutput);
    EXPECT_EQ(reshape->dynValidShapes.size(), kSizeOne);
    EXPECT_EQ(reshape->dynValidShapes[0].size(), kSizeTwo);
    std::vector<std::string> expectValidShape = {
        "4",
        "(RUNTIME_GetViewValidShapeDim(a,0,2)*RUNTIME_Ne(RUNTIME_GetViewValidShapeDim(a,0,2), 0))"
    };
    for (size_t i = 0; i < kSizeTwo; ++i) {
        EXPECT_EQ(reshape->dynValidShapes[0][i].Dump(), expectValidShape[i]);
    }
    EXPECT_EQ(newReshapeSource->GetMemoryTypeOriginal(), MemoryType::MEM_UNKNOWN);
    auto viewOpAttribute = dynamic_cast<ViewOpAttribute *>(view_op.GetOpAttribute().get());
    EXPECT_EQ(viewOpAttribute->GetFromOffset(), view_offset);
    EXPECT_EQ(assemble->from, MemoryType::MEM_DEVICE_DDR);
    EXPECT_EQ(assemble->input, input);
    EXPECT_EQ(assemble->output, newReshapeSource);
}

/*
验证一对多场景下数据的处理
rawShape = {2, 2, 2}
{2, 2, 2}(ddr) -> assemble -> {2, 2, 2} -> reshape -> {2, 4}(unknown) -> view -> {2, 2}(ddr)
                                                                      -> view -> {2, 2}(ddr)
{2, 2, 2}(ddr) -> assemble -> {2, 2, 2} -> reshape -> {2, 4}(unknown) -> view -> {2, 2}
                                                                      -> view -> {2, 2}
*/
TEST_F(TestSplitReshapePass, TestUpdateForBeCovered) {
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestReshapeSplit", "TestReshapeSplit", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);
    // Prepare the graph
    std::vector<int64_t> shape1 = {kNumTwo, kNumTwo, kNumTwo};
    std::vector<int64_t> shape2 = {kNumTwo, kNumFour};
    std::vector<int64_t> shape3 = {kNumTwo, kNumTwo};
    std::vector<int64_t> offset1 = {kNumZero, kNumZero, kNumZero};
    std::vector<int64_t> offset2 = {kNumZero, kNumZero};
    std::vector<int64_t> view_offset1 = {kNumZero, kNumZero};
    std::vector<int64_t> view_offset2 = {kNumZero, kNumTwo};

    std::shared_ptr<RawTensor> RawTensor1 = std::make_shared<RawTensor>(DT_FP32, shape1);
    auto input = std::make_shared<LogicalTensor>(*currFunctionPtr, RawTensor1, offset1, shape1);
    input->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, false);
    auto ubTensor1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape1);
    std::shared_ptr<RawTensor> RawTensor2 = std::make_shared<RawTensor>(DT_FP32, shape2);
    auto ubTensor2 = std::make_shared<LogicalTensor>(*currFunctionPtr, RawTensor2, offset2, shape2);
    ubTensor2->SetMemoryTypeOriginal(MemoryType::MEM_UNKNOWN, false);
    auto output1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape3);
    output1->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, false);
    auto output2 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape3);
    output2->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, false);

    auto &assemble_op = currFunctionPtr->AddOperation(Opcode::OP_ASSEMBLE, {input}, {ubTensor1});
    auto assemble_Attr = std::make_shared<AssembleOpAttribute>(MEM_DEVICE_DDR, offset1);
    assemble_op.SetOpAttribute(assemble_Attr);
    currFunctionPtr->AddOperation(Opcode::OP_RESHAPE, {ubTensor1}, {ubTensor2});
    auto &view_op1 = currFunctionPtr->AddOperation(Opcode::OP_VIEW, {ubTensor2}, {output1});
    auto view_Attr1 = std::make_shared<ViewOpAttribute>(view_offset1);
    view_op1.SetOpAttribute(view_Attr1);
    auto &view_op2 = currFunctionPtr->AddOperation(Opcode::OP_VIEW, {ubTensor2}, {output2});
    auto view_Attr2 = std::make_shared<ViewOpAttribute>(view_offset2);
    view_op2.SetOpAttribute(view_Attr2);

    CalcOverlapPara para;
    std::vector<SymbolicScalar> validShape;

    SplitReshape pass;
    std::vector<int64_t> newCopyOutTileOffset = {kNumZero, kNumZero, kNumZero};
    std::vector<int64_t> newCopyOutTileShape = {kNumTwo, kNumTwo, kNumTwo};
    para.alignedShape = {kNumTwo, kNumTwo, kNumTwo};
    para.overlaps = {input};
    para.newOverlaps = {std::make_shared<LogicalTensor>(*currFunctionPtr, input->tensor, newCopyOutTileOffset, newCopyOutTileShape, validShape)};
    para.reshapeSource = ubTensor1;
    para.input = ubTensor2;
    para.output = output1;
    EXPECT_EQ(pass.CollectCopyOut(*currFunctionPtr), SUCCESS);

    std::vector<int64_t> viewOffset = {kNumZero, kNumZero};
    auto inputView = std::make_shared<LogicalTensor>(*currFunctionPtr, ubTensor2->tensor, viewOffset, shape2);
    para.inputView = inputView;
    para.newInputViewTileShape = {kNumTwo, kNumOne, kNumTwo};
    para.newInputViewTileOffset = {kNumZero, kNumZero, kNumZero};
    EXPECT_EQ(pass.ProcessOnetoMulti(*currFunctionPtr, view_op1, para), SUCCESS);
    std::vector<int64_t> view2Offset = {kNumZero, kNumTwo};
    para.newInputViewTileShape = {kNumTwo, kNumOne, kNumTwo};
    para.newInputViewTileOffset = {kNumZero, kNumOne, kNumZero};
    EXPECT_EQ(pass.ProcessOnetoMulti(*currFunctionPtr, view_op2, para), SUCCESS);

    EXPECT_EQ(pass.reshapes.size(), kSizeOne);
    auto newReshape = pass.reshapes.begin()->second;
    auto newReshapeResource = newReshape->input;
    EXPECT_NE(newReshape->output, ubTensor2);
    EXPECT_EQ(newReshape->output->GetMemoryTypeOriginal(), MemoryType::MEM_UNKNOWN);
    EXPECT_EQ(pass.assembles.size(), kSizeOne);
    auto newAssemble = pass.assembles.begin();
    EXPECT_EQ(newAssemble->from, MemoryType::MEM_DEVICE_DDR);
    EXPECT_EQ(newAssemble->toOffset, offset1);
    EXPECT_EQ(newAssemble->input, input);
    EXPECT_EQ(newAssemble->output, newReshapeResource);
    auto viewOpAttribute1 = dynamic_cast<ViewOpAttribute *>(view_op1.GetOpAttribute().get());
    EXPECT_EQ(viewOpAttribute1->GetFromOffset(), view_offset1);
    auto viewOpAttribute2 = dynamic_cast<ViewOpAttribute *>(view_op2.GetOpAttribute().get());
    EXPECT_EQ(viewOpAttribute2->GetFromOffset(), view_offset2);
}

/*
验证一对多场景下动态shape数据的处理
rawShape = {2, 2, 2}
{2, 2, 2}(ddr) -> assemble -> {2, 2, 2} -> reshape -> {2, 4}(unknown) -> view -> {2, 2}(ddr)
                                                                      -> view -> {2, 2}(ddr)
                                            {a, 4}                                                                   {a, 2}/{b, 0}
{2, 2, 2}(ddr) -> assemble -> {2, 2, 2} -> reshape -> {2, 4}(unknown) -> view -> {2, 2}
                                                                      -> view -> {2, 2}
*/
TEST_F(TestSplitReshapePass, TestDynUpdateForBeCovered) {
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestReshapeSplit", "TestReshapeSplit", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);
    // Prepare the graph
    std::vector<int64_t> shape1 = {kNumTwo, kNumTwo, kNumTwo};
    std::vector<int64_t> shape2 = {kNumTwo, kNumFour};
    std::vector<int64_t> shape3 = {kNumTwo, kNumTwo};
    std::vector<int64_t> offset1 = {kNumZero, kNumZero, kNumZero};
    std::vector<int64_t> offset2 = {kNumZero, kNumZero};
    std::vector<int64_t> view_offset1 = {kNumZero, kNumZero};
    std::vector<int64_t> view_offset2 = {kNumZero, kNumTwo};
    std::vector<SymbolicScalar> validShape = {SymbolicScalar("a"), kNumFour};

    std::shared_ptr<RawTensor> RawTensor1 = std::make_shared<RawTensor>(DT_FP32, shape1);
    auto input = std::make_shared<LogicalTensor>(*currFunctionPtr, RawTensor1, offset1, shape1);
    input->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, false);
    auto ubTensor1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape1);
    std::shared_ptr<RawTensor> RawTensor2 = std::make_shared<RawTensor>(DT_FP32, shape2);
    auto ubTensor2 = std::make_shared<LogicalTensor>(*currFunctionPtr, RawTensor2, offset2, shape2);
    ubTensor2->SetMemoryTypeOriginal(MemoryType::MEM_UNKNOWN, false);
    auto output1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape3);
    output1->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, false);
    auto output2 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape3);
    output2->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, false);

    auto &assemble_op = currFunctionPtr->AddOperation(Opcode::OP_ASSEMBLE, {input}, {ubTensor1});
    auto assemble_Attr = std::make_shared<AssembleOpAttribute>(MEM_DEVICE_DDR, offset1);
    assemble_op.SetOpAttribute(assemble_Attr);
    currFunctionPtr->AddOperation(Opcode::OP_RESHAPE, {ubTensor1}, {ubTensor2});
    auto &view_op1 = currFunctionPtr->AddOperation(Opcode::OP_VIEW, {ubTensor2}, {output1});
    auto view_Attr1 = std::make_shared<ViewOpAttribute>(view_offset1);
    view_op1.SetOpAttribute(view_Attr1);
    auto &view_op2 = currFunctionPtr->AddOperation(Opcode::OP_VIEW, {ubTensor2}, {output2});
    auto view_Attr2 = std::make_shared<ViewOpAttribute>(view_offset2);
    view_op2.SetOpAttribute(view_Attr2);

    CalcOverlapPara para;
    SplitReshape pass;
    std::vector<int64_t> newCopyOutTileOffset = {kNumZero, kNumZero, kNumZero};
    std::vector<int64_t> newCopyOutTileShape = {kNumTwo, kNumTwo, kNumTwo};
    para.alignedShape = {kNumTwo, kNumTwo, kNumTwo};
    para.overlaps = {input};
    auto newOverlap = std::make_shared<LogicalTensor>(*currFunctionPtr, input->tensor, newCopyOutTileOffset, newCopyOutTileShape);
    para.newOverlaps = {newOverlap};
    para.reshapeSource = ubTensor1;
    para.input = ubTensor2;
    para.output = output1;
    EXPECT_EQ(pass.CollectCopyOut(*currFunctionPtr), SUCCESS);

    std::vector<int64_t> viewOffset = {kNumZero, kNumZero};
    auto inputView = std::make_shared<LogicalTensor>(*currFunctionPtr, ubTensor2->tensor, viewOffset, shape2);
    para.inputView = inputView;
    para.newInputViewTileShape = {kNumTwo, kNumOne, kNumTwo};
    para.newInputViewTileOffset = {kNumZero, kNumZero, kNumZero};
    auto newValidShape1 = GetViewValidShape(validShape, view_offset1, {}, shape3);
    para.oriViewDynShape = newValidShape1;
    EXPECT_EQ(pass.ProcessOnetoMulti(*currFunctionPtr, view_op1, para), SUCCESS);
    std::vector<int64_t> view2Offset = {kNumZero, kNumTwo};
    para.newInputViewTileShape = {kNumTwo, kNumOne, kNumTwo};
    para.newInputViewTileOffset = {kNumZero, kNumOne, kNumZero};
    auto newValidShape2 = GetViewValidShape(validShape, view_offset1, {}, shape3);
    para.oriViewDynShape = newValidShape2;
    EXPECT_EQ(pass.ProcessOnetoMulti(*currFunctionPtr, view_op2, para), SUCCESS);

    std::vector<SymbolicScalar> expectShape = {SymbolicScalar("a") * 1, kNumFour};
    EXPECT_EQ(pass.reshapes.size(), kSizeOne);
    auto newReshape = pass.reshapes.begin()->second;
    auto newReshapeResource = newReshape->input;
    EXPECT_NE(newReshape->output, ubTensor2);
    auto reshapeOutput = newReshape->output;
    EXPECT_EQ(newReshape->dynValidShapes.size(), kSizeTwo);
    EXPECT_EQ(newReshape->dynValidShapes[0].size(), kSizeTwo);
    EXPECT_EQ(newReshape->dynValidShapes[1].size(), kSizeTwo);
    std::vector<std::string> expectValidShape1 = {
        "(RUNTIME_GetViewValidShapeDim(a,0,2)*RUNTIME_Ne(RUNTIME_GetViewValidShapeDim(a,0,2), 0))",
        "2"
    };
    for (size_t i = 0; i < kSizeTwo; ++i) {
        EXPECT_EQ(newReshape->dynValidShapes[0][i].Dump(), expectValidShape1[i]);
    }
    std::vector<std::string> expectValidShape2 = {
        "(RUNTIME_GetViewValidShapeDim(a,0,2)*RUNTIME_Ne(RUNTIME_GetViewValidShapeDim(a,0,2), 0))",
        "4"
    };
    for (size_t i = 0; i < kSizeTwo; ++i) {
        EXPECT_EQ(newReshape->dynValidShapes[1][i].Dump(), expectValidShape2[i]);
    }
    EXPECT_EQ(reshapeOutput->GetMemoryTypeOriginal(), MemoryType::MEM_UNKNOWN);
    EXPECT_EQ(pass.assembles.size(), kSizeOne);
    auto newAssemble = pass.assembles.begin();
    EXPECT_EQ(newAssemble->from, MemoryType::MEM_DEVICE_DDR);
    EXPECT_EQ(newAssemble->toOffset, offset1);
    EXPECT_EQ(newAssemble->input, input);
    EXPECT_EQ(newAssemble->output, newReshapeResource);
    auto viewOpAttribute1 = dynamic_cast<ViewOpAttribute *>(view_op1.GetOpAttribute().get());
    EXPECT_EQ(viewOpAttribute1->GetFromOffset(), view_offset1);
    auto viewOpAttribute2 = dynamic_cast<ViewOpAttribute *>(view_op2.GetOpAttribute().get());
    EXPECT_EQ(viewOpAttribute2->GetFromOffset(), view_offset2);
}

/*
验证多对一场景下数据的处理
rawShape = {2, 4}
{2, 2}(ddr) -> assemble -> {2, 4}(unknown) -> reshape -> {2, 2, 2}(ddr) -> view -> {2, 2, 2}(ddr)
{2, 2}(ddr) -> assemble ->
{2, 2}(ddr) -> {2, 4}(unknown) -> reshape -> {2, 2, 2}(ddr) -> view -> {2, 2}
{2, 2}(ddr) ->
*/
TEST_F(TestSplitReshapePass, TestUpdateForPerfectlyMatchWithAll) {
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestReshapeSplit", "TestReshapeSplit", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);
    // Prepare the graph
    std::vector<int64_t> shape1 = {kNumTwo, kNumFour};
    std::vector<int64_t> shape2 = {kNumTwo, kNumTwo};
    std::vector<int64_t> shape3 = {kNumTwo, kNumTwo, kNumTwo};
    std::vector<int64_t> offset1 = {kNumZero, kNumZero};
    std::vector<int64_t> offset2 = {kNumZero, kNumTwo};
    std::vector<int64_t> view_offset = {kNumZero, kNumZero, kNumZero};

    std::shared_ptr<RawTensor> RawTensor1 = std::make_shared<RawTensor>(DT_FP32, shape1);
    auto input1 = std::make_shared<LogicalTensor>(*currFunctionPtr, RawTensor1, offset1, shape2);
    input1->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, false);
    auto input2 = std::make_shared<LogicalTensor>(*currFunctionPtr, RawTensor1, offset2, shape2);
    input2->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, false);
    auto ubTensor1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape1);
    ubTensor1->SetMemoryTypeOriginal(MemoryType::MEM_UNKNOWN, false);
    auto ubTensor2 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape3);
    ubTensor2->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, false);
    auto output = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape3);
    output->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, false);

    auto &assemble_op1 = currFunctionPtr->AddOperation(Opcode::OP_ASSEMBLE, {input1}, {ubTensor1});
    auto assemble_Attr1 = std::make_shared<AssembleOpAttribute>(MEM_DEVICE_DDR, offset1);
    assemble_op1.SetOpAttribute(assemble_Attr1);
    auto &assemble_op2 = currFunctionPtr->AddOperation(Opcode::OP_ASSEMBLE, {input2}, {ubTensor1});
    auto assemble_Attr2 = std::make_shared<AssembleOpAttribute>(MEM_DEVICE_DDR, offset2);
    assemble_op2.SetOpAttribute(assemble_Attr2);
    currFunctionPtr->AddOperation(Opcode::OP_RESHAPE, {ubTensor1}, {ubTensor2});
    auto &view_op = currFunctionPtr->AddOperation(Opcode::OP_VIEW, {ubTensor2}, {output});
    auto view_Attr = std::make_shared<ViewOpAttribute>(view_offset);
    view_op.SetOpAttribute(view_Attr);

    CalcOverlapPara para;
    std::vector<SymbolicScalar> validShape;

    SplitReshape pass;
    para.alignedShape = {kNumTwo, kNumTwo, kNumTwo};
    para.overlaps = {input1, input2};
    std::vector<int64_t> newInput1TileOffset = {kNumZero, kNumZero, kNumZero};
    std::vector<int64_t> newInput1TileShape = {kNumTwo, kNumOne, kNumTwo};
    auto newInput1 = std::make_shared<LogicalTensor>(*currFunctionPtr, input1->tensor, newInput1TileOffset, newInput1TileShape, validShape);
    std::vector<int64_t> newInput2TileOffset = {kNumZero, kNumOne, kNumZero};
    std::vector<int64_t> newInput2TileShape = {kNumTwo, kNumOne, kNumTwo};
    auto newInput2 = std::make_shared<LogicalTensor>(*currFunctionPtr, input2->tensor, newInput2TileOffset, newInput2TileShape, validShape);
    para.newOverlaps = {newInput1, newInput2};
    para.reshapeSource = ubTensor1;
    para.input = ubTensor2;
    para.output = output;
    para.newInputViewTileShape = {kNumTwo, kNumTwo, kNumTwo};
    para.newInputViewTileOffset = {kNumZero, kNumZero, kNumZero};
    auto inputView = std::make_shared<LogicalTensor>(*currFunctionPtr, ubTensor2->tensor, view_offset, shape3);
    para.inputView = inputView;
    EXPECT_EQ(pass.CollectCopyOut(*currFunctionPtr), SUCCESS);
    EXPECT_EQ(pass.ProcessMultitoOne(*currFunctionPtr, view_op, para), SUCCESS);
    EXPECT_EQ(pass.redundantViewops.size(), kSizeZero);
    EXPECT_EQ(pass.reshapes.size(), kSizeOne);
    auto reshape = pass.reshapes.begin()->second;
    auto newReshapeSource = reshape->input;
    auto newReshapeOutput = view_op.GetInputOperand(kSizeZero);
    EXPECT_EQ(reshape->output, newReshapeOutput);
    EXPECT_EQ(newReshapeSource->GetMemoryTypeOriginal(), MemoryType::MEM_UNKNOWN);
    EXPECT_EQ(newReshapeOutput->GetMemoryTypeOriginal(), MemoryType::MEM_DEVICE_DDR);
    auto viewOpAttribute = dynamic_cast<ViewOpAttribute *>(view_op.GetOpAttribute().get());
    EXPECT_EQ(viewOpAttribute->GetFromOffset(), inputView->offset);
}

/*
验证多对一场景下动态shape数据的处理
rawShape = {2, 4}
{2, 2}(ddr) -> assemble -> {2, 4}(unknown) -> reshape -> {2, 2, 2}(ddr) -> view -> {2, 2, 2}(ddr)
{2, 2}(ddr) -> assemble ->
                                             {a, 2, 2}
{2, 2}(ddr) -> {2, 4}(unknown) -> reshape -> {2, 2, 2}(ddr) -> view -> {2, 2}
{2, 2}(ddr) ->
*/
TEST_F(TestSplitReshapePass, TestDynUpdateForPerfectlyMatchWithAll) {
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestReshapeSplit", "TestReshapeSplit", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);
    // Prepare the graph
    std::vector<int64_t> shape1 = {kNumTwo, kNumFour};
    std::vector<int64_t> shape2 = {kNumTwo, kNumTwo};
    std::vector<int64_t> shape3 = {kNumTwo, kNumTwo, kNumTwo};
    std::vector<int64_t> offset1 = {kNumZero, kNumZero};
    std::vector<int64_t> offset2 = {kNumZero, kNumTwo};
    std::vector<int64_t> view_offset = {kNumZero, kNumZero, kNumZero};
    std::vector<SymbolicScalar> validShape = {SymbolicScalar("a"), kNumTwo, kNumTwo};

    std::shared_ptr<RawTensor> RawTensor1 = std::make_shared<RawTensor>(DT_FP32, shape1);
    auto input1 = std::make_shared<LogicalTensor>(*currFunctionPtr, RawTensor1, offset1, shape2);
    input1->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, false);
    auto input2 = std::make_shared<LogicalTensor>(*currFunctionPtr, RawTensor1, offset2, shape2);
    input2->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, false);
    auto ubTensor1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape1);
    ubTensor1->SetMemoryTypeOriginal(MemoryType::MEM_UNKNOWN, false);
    auto ubTensor2 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape3);
    ubTensor2->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, false);
    auto output = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape3);
    output->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, false);

    auto &assemble_op1 = currFunctionPtr->AddOperation(Opcode::OP_ASSEMBLE, {input1}, {ubTensor1});
    auto assemble_Attr1 = std::make_shared<AssembleOpAttribute>(MEM_DEVICE_DDR, offset1);
    assemble_op1.SetOpAttribute(assemble_Attr1);
    auto &assemble_op2 = currFunctionPtr->AddOperation(Opcode::OP_ASSEMBLE, {input2}, {ubTensor1});
    auto assemble_Attr2 = std::make_shared<AssembleOpAttribute>(MEM_DEVICE_DDR, offset2);
    assemble_op2.SetOpAttribute(assemble_Attr2);
    currFunctionPtr->AddOperation(Opcode::OP_RESHAPE, {ubTensor1}, {ubTensor2});
    auto &view_op = currFunctionPtr->AddOperation(Opcode::OP_VIEW, {ubTensor2}, {output});
    auto view_Attr = std::make_shared<ViewOpAttribute>(view_offset);
    view_op.SetOpAttribute(view_Attr);

    CalcOverlapPara para;
    SplitReshape pass;
    para.alignedShape = {kNumTwo, kNumTwo, kNumTwo};
    para.overlaps = {input1, input2};
    std::vector<int64_t> newInput1TileOffset = {kNumZero, kNumZero, kNumZero};
    std::vector<int64_t> newInput1TileShape = {kNumTwo, kNumOne, kNumTwo};
    auto newInput1 = std::make_shared<LogicalTensor>(*currFunctionPtr, input1->tensor, newInput1TileOffset, newInput1TileShape);
    std::vector<int64_t> newInput2TileOffset = {kNumZero, kNumOne, kNumZero};
    std::vector<int64_t> newInput2TileShape = {kNumTwo, kNumOne, kNumTwo};
    std::vector<SymbolicScalar> newInput2DynOffset = {SymbolicScalar("b"), kNumOne, kNumZero};
    std::vector<SymbolicScalar> newInput2DynShape = {SymbolicScalar("a"), kNumOne, kNumTwo};
    auto newInput2 = std::make_shared<LogicalTensor>(*currFunctionPtr, input2->tensor, newInput2TileOffset, newInput2TileShape);
    para.newOverlaps = {newInput1, newInput2};
    para.reshapeSource = ubTensor1;
    para.input = ubTensor2;
    para.output = output;
    para.newInputViewTileShape = {kNumTwo, kNumTwo, kNumTwo};
    para.newInputViewTileOffset = {kNumZero, kNumZero, kNumZero};
    auto newValidShape = GetViewValidShape(validShape, view_offset, {}, shape3);
    para.oriViewDynShape = newValidShape;
    auto inputView = std::make_shared<LogicalTensor>(*currFunctionPtr, ubTensor2->tensor, view_offset, shape3);
    para.inputView = inputView;
    EXPECT_EQ(pass.CollectCopyOut(*currFunctionPtr), SUCCESS);
    EXPECT_EQ(pass.ProcessMultitoOne(*currFunctionPtr, view_op, para), SUCCESS);
    EXPECT_EQ(pass.redundantViewops.size(), kSizeZero);
    EXPECT_EQ(pass.reshapes.size(), kSizeOne);
    auto reshape = pass.reshapes.begin()->second;
    EXPECT_EQ(reshape->dynValidShapes.size(), kNumOne);
    EXPECT_EQ(reshape->dynValidShapes[0].size(), kNumThree);
    std::vector<std::string> expectValidShape = {
        "(RUNTIME_GetViewValidShapeDim(a,0,2)*RUNTIME_Ne(RUNTIME_GetViewValidShapeDim(a,0,2), 0))",
        "2",
        "2"
    };
    for (size_t i = 0; i < kNumThree; ++i) {
        EXPECT_EQ(reshape->dynValidShapes[0][i].Dump(), expectValidShape[i]);
    }
    auto newReshapeSource = reshape->input;
    auto newReshapeOutput = view_op.GetInputOperand(kSizeZero);
    EXPECT_EQ(reshape->output, newReshapeOutput);
    EXPECT_EQ(newReshapeSource->GetMemoryTypeOriginal(), MemoryType::MEM_UNKNOWN);
    EXPECT_EQ(newReshapeOutput->GetMemoryTypeOriginal(), MemoryType::MEM_DEVICE_DDR);
    auto viewOpAttribute = dynamic_cast<ViewOpAttribute *>(view_op.GetOpAttribute().get());
    EXPECT_EQ(viewOpAttribute->GetFromOffset(), inputView->offset);
}

void RunPassStra(Function &func, const std::string passName) {
    std::string strategyName = passName + "Strategy";
    PassManager &passManager = PassManager::Instance();
    passManager.RegisterStrategy(strategyName, {
        {passName, passName},
    });
    EXPECT_EQ(passManager.RunPass(Program::GetInstance(), func, strategyName), SUCCESS);
}

struct CheckReshapeStruct {
    const std::vector<int64_t> reshapeInputShape;
    const int reshapeInputProducerSize;
    const bool checkProducer;
    const std::vector<int64_t> reshapeInputOperandShape;
    const std::vector<int64_t> reshapeOutputShape;
    const int reshapeOutputProducerSize;
    const bool checkConsumer;
    const std::vector<int64_t> reshapeOutputOperandShape;
    const uint32_t reshapeOpNum;
};

void CheckOpReshape(Function *func, CheckReshapeStruct expectReshape) {
    int reshapeOp = 0;
    for (auto &op : func->Operations()) {
        if (op.GetOpcode() == Opcode::OP_RESHAPE) {
            EXPECT_EQ(op.GetInputOperandSize(), kSizeOne);
            auto reshapeInput = op.GetInputOperand(kSizeZero);
            EXPECT_NE(reshapeInput, nullptr);
            EXPECT_EQ(reshapeInput->shape, expectReshape.reshapeInputShape);
            EXPECT_EQ(reshapeInput->GetProducers().size(), expectReshape.reshapeInputProducerSize);
            for (const auto &producer : reshapeInput->GetProducers()) {
                EXPECT_EQ(producer->GetOpcode(), Opcode::OP_ASSEMBLE);
                if (expectReshape.checkProducer){
                    EXPECT_EQ(producer->GetInputOperandSize(), kSizeOne);
                    EXPECT_EQ(producer->GetInputOperand(kSizeZero)->shape, expectReshape.reshapeInputOperandShape);
                }
            }
            EXPECT_EQ(op.GetOutputOperandSize(), kSizeOne);
            auto reshapeOutput = op.GetOutputOperand(kSizeZero);
            EXPECT_NE(reshapeOutput, nullptr);
            EXPECT_EQ(reshapeOutput->shape, expectReshape.reshapeOutputShape);
            EXPECT_EQ(reshapeOutput->GetConsumers().size(), expectReshape.reshapeOutputProducerSize);
            for (const auto &consumer : reshapeOutput->GetConsumers()) {
                EXPECT_EQ(consumer->GetOpcode(), Opcode::OP_VIEW);
                if (expectReshape.checkConsumer){
                    EXPECT_EQ(consumer->GetOutputOperandSize(), kSizeOne);
                    EXPECT_EQ(consumer->GetOutputOperand(kSizeZero)->shape, expectReshape.reshapeOutputOperandShape);
                }
            }
            reshapeOp++;
        }
    }
    EXPECT_EQ(reshapeOp, expectReshape.reshapeOpNum);
}

/*
校验一对一场景(使用expandfunction作为前序pass)
1) 用例设置：
{2,2,4} -> exp -> {2,2,4} -> reshape -> {2,2,1,4} -> exp -> {2,2,1,4}
tileshape = {2,2,2,2}
2) expandfunction：
exp -> {2,2,2} -> assemble -> reshape -> {2,2,1,4} -> view -> {2,2,1,2} -> exp -> {2,2,1,2}
exp -> {2,2,2} -> assemble                         -> view -> {2,2,1,2} -> exp -> {2,2,1,2}
3) splitreshape
exp -> {2,2,2} -> assemble -> reshape -> {2,2,1,2} -> view -> {2,2,1,2} -> exp -> {2,2,1,2}
exp -> {2,2,2} -> assemble -> reshape -> {2,2,1,2} -> view -> {2,2,1,2} -> exp -> {2,2,1,2}
*/
TEST_F(TestSplitReshapePass, TestPerfectlyMatchedSTest) {
    //Define the shape of the Tensors
    std::vector<int64_t> origShape = {kNumTwo, kNumTwo, kNumFour};
    std::vector<int64_t> reshapeShape = {kNumTwo, kNumTwo, kNumOne, kNumFour};
    std::vector<int64_t> tiledShape = {kNumTwo, kNumTwo, kNumTwo, kNumTwo};
    std::vector<int64_t> tiledorigShape = {kNumTwo, kNumTwo, kNumTwo};
    std::vector<int64_t> tiledreshapeShape = {kNumTwo, kNumTwo, kNumOne, kNumTwo};

    TileShape::Current().SetVecTile(tiledShape);
    Tensor input(DT_FP32, origShape, "input");
    Tensor output(DT_FP32, reshapeShape, "output");

    FUNCTION("STCase1") {
        Tensor exp = Exp(input);
        Tensor reshape = Reshape(exp, reshapeShape);
        output = Exp(reshape);
    }

    Function* func = Program::GetInstance().GetFunctionByRawName("TENSOR_STCase1");
    
    RunPassStra(*func, "ExpandFunction");
    CheckOpReshape(func, CheckReshapeStruct{origShape, kSizeTwo, false, {}, reshapeShape, kSizeTwo, false, {}, kNumOne});

    RunPassStra(*func, "SplitReshape");
    CheckOpReshape(func, CheckReshapeStruct{tiledorigShape, kSizeOne, false, {}, tiledreshapeShape, kSizeOne, false, {}, kNumTwo});
}

/*
校验一对多场景(使用expandfunction作为前序pass)
1) 用例设置：
{4,2,2} -> exp -> {4,2,2} -> reshape -> {4,4} -> exp -> {4,4}
tileshape = {2,2,2}
2) expandfunction：
exp -> {2,2,2} -> assemble -> reshape -> {4,4} -> view -> {2,2} -> exp -> {2,2}
exp -> {2,2,2} -> assemble                     -> view -> {2,2} -> exp -> {2,2}
                                               -> view -> {2,2} -> exp -> {2,2}
                                               -> view -> {2,2} -> exp -> {2,2}
3) splitreshape
                                               -> view -> {2,2} -> exp -> {2,2}
exp -> {2,2,2} -> assemble -> reshape -> {2,4} -> view -> {2,2} -> exp -> {2,2}
exp -> {2,2,2} -> assemble -> reshape -> {2,4} -> view -> {2,2} -> exp -> {2,2}
                                               -> view -> {2,2} -> exp -> {2,2}
*/
TEST_F(TestSplitReshapePass, TestBeCoveredSTest) {
    //Define the shape of the Tensors
    std::vector<int64_t> origShape = {kNumFour, kNumTwo, kNumTwo};
    std::vector<int64_t> reshapeShape = {kNumFour, kNumFour};
    std::vector<int64_t> tiledShape = {kNumTwo, kNumTwo, kNumTwo};
    std::vector<int64_t> tiledorigShape = {kNumTwo, kNumTwo, kNumTwo};
    std::vector<int64_t> tiledreshapeShape = {kNumTwo, kNumFour};
    std::vector<int64_t> tiledviewShape = {kNumTwo, kNumTwo};

    TileShape::Current().SetVecTile(tiledShape);
    Tensor input(DT_FP32, origShape, "input");
    Tensor output(DT_FP32, reshapeShape, "output");

    FUNCTION("STCase2") {
        Tensor exp = Exp(input);
        Tensor reshape = Reshape(exp, reshapeShape);
        output = Exp(reshape);
    }

    Function* func = Program::GetInstance().GetFunctionByRawName("TENSOR_STCase2");

    RunPassStra(*func, "ExpandFunction");
    CheckOpReshape(func, CheckReshapeStruct{origShape, kSizeTwo, false, {}, reshapeShape, kSizeFour, false, {}, kNumOne});

    RunPassStra(*func, "SplitReshape");
    CheckOpReshape(func, CheckReshapeStruct{tiledorigShape, kSizeOne, false, {}, tiledreshapeShape, kSizeTwo, true, tiledviewShape, kNumTwo});
}

/*
校验多对一场景(使用expandfunction作为前序pass)
1) 用例设置：
{2,4,4} -> exp -> {2,4,4} -> reshape -> {2,4,2,2} -> exp -> {2,4,2,2}
tileshape = {2,2,2,2}
2) expandfunction：
exp -> {2,2,2} -> assemble -> reshape -> {2,4,2,2} -> view -> {2,2,2,2} -> exp -> {2,2,2,2}
exp -> {2,2,2} -> assemble                         -> view -> {2,2,2,2} -> exp -> {2,2,2,2}
exp -> {2,2,2} -> assemble
exp -> {2,2,2} -> assemble
3) splitreshape
exp -> {2,2,2} -> assemble ->
exp -> {2,2,2} -> assemble -> reshape -> {2,2,2,2} -> view -> {2,2,2,2} -> exp -> {2,2,2,2}
exp -> {2,2,2} -> assemble -> reshape -> {2,2,2,2} -> view -> {2,2,2,2} -> exp -> {2,2,2,2}
exp -> {2,2,2} -> assemble ->
*/
TEST_F(TestSplitReshapePass, TestPerfectlyMatchedWithallSTest) {
    //Define the shape of the Tensors
    std::vector<int64_t> origShape = {kNumTwo, kNumFour, kNumFour};
    std::vector<int64_t> reshapeShape = {kNumTwo, kNumFour, kNumTwo, kNumTwo};
    std::vector<int64_t> tiledShape = {kNumTwo, kNumTwo, kNumTwo, kNumTwo};
    std::vector<int64_t> tiledassembleShape = {kNumTwo, kNumTwo, kNumTwo};
    std::vector<int64_t> tiledreshapeShape = {kNumTwo, kNumTwo, kNumFour};
    std::vector<int64_t> tiledviewShape = {kNumTwo, kNumTwo, kNumTwo, kNumTwo};

    TileShape::Current().SetVecTile(tiledShape);
    Tensor input(DT_FP32, origShape, "input");
    Tensor output(DT_FP32, reshapeShape, "output");

    FUNCTION("STCase3") {
        Tensor exp = Exp(input);
        Tensor reshape = Reshape(exp, reshapeShape);
        output = Exp(reshape);
    }

    Function* func = Program::GetInstance().GetFunctionByRawName("TENSOR_STCase3");

    RunPassStra(*func, "ExpandFunction");
    CheckOpReshape(func, CheckReshapeStruct{origShape, kSizeFour, false, {}, reshapeShape, kSizeTwo, false, {}, kNumOne});

    RunPassStra(*func, "SplitReshape");
    CheckOpReshape(func, CheckReshapeStruct{tiledreshapeShape, kSizeTwo, true, tiledassembleShape, tiledviewShape, kSizeOne, true, tiledviewShape, kNumTwo});
}

void CollectOperations(std::shared_ptr<Function> func, std::unordered_map<LogicalTensorPtr, int> &inputsWeight, std::unordered_map<LogicalTensorPtr, Operation*> &newAssembles,
    const uint32_t expectReshapeOp, const uint32_t expectViewOp, int expectAssembleOp){
    int reshapeOp = 0;
    int assembleOp = 0;
    int viewOp = 0;

    for (auto &op : func->Operations().DuplicatedOpList()) {
        if (op->GetOpcode() == Opcode::OP_RESHAPE) {
            reshapeOp++;
        } else if (op->GetOpcode() == Opcode::OP_ASSEMBLE) {
            for (auto [input, weight] : inputsWeight){
                if (op->GetInputOperand(kSizeZero) == input){
                    newAssembles[input] = op;
                    assembleOp += weight;
                }
            }
        } else if (op->GetOpcode() == Opcode::OP_VIEW) {
            viewOp++;
        }
    }
    EXPECT_EQ(reshapeOp, expectReshapeOp);
    EXPECT_EQ(viewOp, expectViewOp);
    EXPECT_EQ(assembleOp, expectAssembleOp);
}

void CheckNewAssembles(std::unordered_map<LogicalTensorPtr, Operation*> &newAssembles, std::unordered_map<LogicalTensorPtr, std::vector<int64_t>> &expectAssembleOffset,
    std::vector<std::string> &expectAssembleDynShape, std::unordered_map<LogicalTensorPtr, std::vector<std::string>> &expectValidShapes,
    std::vector<SymbolicScalar> &dynInputShape, LogicalTensors &reshapeOutputs, const uint32_t reshapeOutputSize = kNumFour) {
    for (auto [input, newAssemble] : newAssembles) {
        EXPECT_NE(newAssemble, nullptr);
        auto assembleDynValidShape = dynamic_cast<AssembleOpAttribute *>(newAssemble->GetOpAttribute().get())->GetFromDynValidShape();
        EXPECT_EQ(assembleDynValidShape.size(), kNumThree);
        for (size_t i = 0; i < kNumThree; ++i) {
            EXPECT_EQ(assembleDynValidShape[i].Dump(), dynInputShape[i].Dump());
        }
        auto assembleOpAttribute = dynamic_cast<AssembleOpAttribute *>(newAssemble->GetOpAttribute().get());
        EXPECT_EQ(assembleOpAttribute->GetToOffset(), expectAssembleOffset[input]);
        auto reshapeSource = newAssemble->GetOutputOperand(kSizeZero);
        std::vector<SymbolicScalar> assembleDynOutput = reshapeSource->GetDynValidShape();
        EXPECT_EQ(assembleDynOutput.size(), kNumThree);
        for (size_t i = 0; i < kNumThree; ++i) {
            EXPECT_EQ(assembleDynOutput[i].Dump(), expectAssembleDynShape[i]);
        }
        auto reshape = *(reshapeSource->GetConsumers().begin());
        auto reshapeOutput = reshape->GetOutputOperand(kSizeZero);
        reshapeOutputs.emplace_back(reshapeOutput);
        std::vector<SymbolicScalar> reshapeAttrValidShape;
        std::vector<SymbolicScalar> reshapeDynOutput = reshapeOutput->GetDynValidShape();
        EXPECT_TRUE(reshape->GetAttr(OP_ATTR_PREFIX + "validShape", reshapeAttrValidShape));
        EXPECT_EQ(reshapeDynOutput.size(), reshapeOutputSize);
        EXPECT_EQ(reshapeAttrValidShape.size(), reshapeOutputSize);
        for (size_t i = 0; i < reshapeOutputSize; ++i) {
            EXPECT_EQ(reshapeDynOutput[i].Dump(), expectValidShapes[input][i]);
            EXPECT_EQ(reshapeAttrValidShape[i].Dump(), expectValidShapes[input][i]);
        }
    }
}

LogicalTensors BuildDynPerfectlyMatchFunc(std::shared_ptr<Function> func){
    std::vector<int64_t> shape1 = {kNumTwo, kNumTwo, kNumTwo};
    std::vector<int64_t> shape2 = {kNumTwo, kNumTwo, kNumFour};
    std::vector<int64_t> shape3 = {kNumTwo, kNumTwo, kNumOne, kNumFour};
    std::vector<int64_t> shape4 = {kNumTwo, kNumTwo, kNumOne, kNumTwo};
    std::vector<int64_t> assembleOffset1 = {kNumZero, kNumZero, kNumZero};
    std::vector<int64_t> assembleOffset2 = {kNumZero, kNumZero, kNumTwo};
    std::vector<int64_t> viewOffset1 = {kNumZero, kNumZero, kNumZero, kNumZero};
    std::vector<int64_t> viewOffset2 = {kNumZero, kNumZero, kNumZero, kNumTwo};
    std::vector<SymbolicScalar> validShape = {SymbolicScalar("a0"), SymbolicScalar("a1"), kNumOne, kNumFour};
    std::vector<SymbolicScalar> dynInputShape = {SymbolicScalar("a0"), SymbolicScalar("a1"), kNumTwo};

    std::shared_ptr<RawTensor> ddrRawTensor1 = std::make_shared<RawTensor>(DT_FP32, shape2);
    std::shared_ptr<RawTensor> ddrRawTensor2 = std::make_shared<RawTensor>(DT_FP32, shape3);
    auto input1 = std::make_shared<LogicalTensor>(*func, ddrRawTensor1, assembleOffset1, shape1, dynInputShape);
    input1->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, false);
    auto input2 = std::make_shared<LogicalTensor>(*func, ddrRawTensor1, assembleOffset2, shape1, dynInputShape);
    input2->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, false);
    auto ubTensor1 = std::make_shared<LogicalTensor>(*func, DT_FP32, shape2);
    ubTensor1->SetMemoryTypeOriginal(MemoryType::MEM_UNKNOWN, false);
    auto ubTensor2 = std::make_shared<LogicalTensor>(*func, DT_FP32, shape3);
    ubTensor2->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, false);
    auto output1 = std::make_shared<LogicalTensor>(*func, ddrRawTensor2, viewOffset1, shape4);
    output1->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, false);
    auto output2 = std::make_shared<LogicalTensor>(*func, ddrRawTensor2, viewOffset2, shape4);
    output2->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, false);

    auto &assemble_op1 = func->AddOperation(Opcode::OP_ASSEMBLE, {input1}, {ubTensor1});
    auto assemble_Attr1 = std::make_shared<AssembleOpAttribute>(MEM_DEVICE_DDR, assembleOffset1);
    assemble_op1.SetOpAttribute(assemble_Attr1);
    auto &assemble_op2 = func->AddOperation(Opcode::OP_ASSEMBLE, {input2}, {ubTensor1});
    auto assemble_Attr2 = std::make_shared<AssembleOpAttribute>(MEM_DEVICE_DDR, assembleOffset2);
    assemble_op2.SetOpAttribute(assemble_Attr2);
    auto &reshape_op = func->AddOperation(Opcode::OP_RESHAPE, {ubTensor1}, {ubTensor2});
    reshape_op.SetAttribute(OP_ATTR_PREFIX + "validShape", validShape);
    auto &view_op1 = func->AddOperation(Opcode::OP_VIEW, {ubTensor2}, {output1});
    auto view_Attr1 = std::make_shared<ViewOpAttribute>(viewOffset1);
    view_op1.SetOpAttribute(view_Attr1);
    auto &view_op2 = func->AddOperation(Opcode::OP_VIEW, {ubTensor2}, {output2});
    auto view_Attr2 = std::make_shared<ViewOpAttribute>(viewOffset2);
    view_op2.SetOpAttribute(view_Attr2);

    func->inCasts_.push_back(input1);
    func->inCasts_.push_back(input2);
    func->outCasts_.push_back(output1);
    func->outCasts_.push_back(output2);
    return {input1, input2};
}

/*
验证一对一场景下动态shape的兜底策略
因为缺乏宏构建策略，手动构造expandfunction的输出构图
1) 用例设置：
                                 {a0,a1,1,4}
{2,2,2} -> assemble -> {2,2,4} -> reshape -> {2,2,1,4} -> view -> {2,2,1,2}
{2,2,2} -> assemble                                    -> view -> {2,2,1,2}
2) splitreshape
                                 {a0,a1,1,2}
{2,2,2} -> assemble -> {2,2,2} -> reshape -> {2,2,1,2} -> view -> {2,2,1,2}
{2,2,2} -> assemble -> {2,2,2} -> reshape -> {2,2,1,2} -> view -> {2,2,1,2}
{a0,a1,2}             {a0,a1,2}              {a0,a1,1,2}
{a0,a1,2}             {a0,a1,2}              {a0,a1,1,2}
*/
TEST_F(TestSplitReshapePass, TestDynPerfectlyMatchSTest) {
    auto func = std::make_shared<Function>(Program::GetInstance(), "TestReshapeSplit", "TestReshapeSplit", nullptr);
    EXPECT_TRUE(func != nullptr);
    std::vector<int64_t> assembleOffset1 = {kNumZero, kNumZero, kNumZero};
    std::vector<int64_t> assembleOffset2 = {kNumZero, kNumZero, kNumTwo};
    std::vector<SymbolicScalar> dynInputShape = {SymbolicScalar("a0"), SymbolicScalar("a1"), kNumTwo};

    auto inputs = BuildDynPerfectlyMatchFunc(func);

    RunPassStra(*func, "SplitReshape");

    std::unordered_map<LogicalTensorPtr, int> inputsWeight = {
        {inputs[0], 1},
        {inputs[1], 10}
    };
    std::unordered_map<LogicalTensorPtr, Operation*> newAssembles = {
        {inputs[0], nullptr},
        {inputs[1], nullptr}
    };
    CollectOperations(func, inputsWeight, newAssembles, kNumTwo, kNumTwo, 11);

    std::unordered_map<LogicalTensorPtr, std::vector<int64_t>> expectAssembleOffset = {
        {inputs[0], assembleOffset1},
        {inputs[1], assembleOffset2}
    };
    LogicalTensors reshapeOutputs;
    std::vector<std::string> expectAssembleDynShape = {
        "RUNTIME_Max(0, (a0*RUNTIME_Ne(a0, 0)))",
        "RUNTIME_Max(0, (a1*RUNTIME_Ne(a1, 0)))",
        "2"
    };
    std::vector<std::string> expectReshapeDynShape = {
        "RUNTIME_Max(0, ((RUNTIME_GetViewValidShapeDim(a0,0,2)*RUNTIME_Ne(RUNTIME_GetViewValidShapeDim(a0,0,2), 0))-0))",
        "RUNTIME_Max(0, ((RUNTIME_GetViewValidShapeDim(a1,0,2)*RUNTIME_Ne(RUNTIME_GetViewValidShapeDim(a1,0,2), 0))-0))",
        "1",
        "2"
    };
    std::unordered_map<LogicalTensorPtr, std::vector<std::string>> expectValidShapes = {
        {inputs[0], expectReshapeDynShape},
        {inputs[1], expectReshapeDynShape}
    };
    CheckNewAssembles(newAssembles, expectAssembleOffset, expectAssembleDynShape, expectValidShapes, dynInputShape, reshapeOutputs, kNumFour);
    EXPECT_NE(reshapeOutputs[0], reshapeOutputs[1]);
    EXPECT_NE(*(reshapeOutputs[0]->GetConsumers().begin()), *(reshapeOutputs[1]->GetConsumers().begin()));
}

LogicalTensors BuildDynBeCoveredFunc(std::shared_ptr<Function> func){
    std::vector<int64_t> shape1 = {kNumTwo, kNumTwo, kNumTwo};
    std::vector<int64_t> shape2 = {kNumTwo, kNumTwo, kNumFour};
    std::vector<int64_t> shape3 = {kNumFour, kNumFour};
    std::vector<int64_t> shape4 = {kNumTwo, kNumTwo};
    std::vector<int64_t> assembleOffset1 = {kNumZero, kNumZero, kNumZero};
    std::vector<int64_t> assembleOffset2 = {kNumZero, kNumZero, kNumTwo};
    std::vector<int64_t> viewOffset1 = {kNumZero, kNumZero};
    std::vector<int64_t> viewOffset2 = {kNumZero, kNumTwo};
    std::vector<int64_t> viewOffset3 = {kNumTwo, kNumZero};
    std::vector<int64_t> viewOffset4 = {kNumTwo, kNumTwo};
    std::vector<SymbolicScalar> validShape = {kNumFour, SymbolicScalar("a")};
    std::vector<SymbolicScalar> dynInputShape = {kNumTwo, kNumTwo, SymbolicScalar("a")};

    auto ubTensor1 = std::make_shared<LogicalTensor>(*func, DT_FP32, shape2);
    ubTensor1->SetMemoryTypeOriginal(MemoryType::MEM_UNKNOWN, false);
    auto ubTensor2 = std::make_shared<LogicalTensor>(*func, DT_FP32, shape3);
    ubTensor2->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, false);
    std::shared_ptr<RawTensor> ddrRawTensor1 = std::make_shared<RawTensor>(DT_FP32, shape2);
    auto input1 = std::make_shared<LogicalTensor>(*func, ddrRawTensor1, assembleOffset1, shape1, dynInputShape);
    input1->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, false);
    auto input2 = std::make_shared<LogicalTensor>(*func, ddrRawTensor1, assembleOffset2, shape1, dynInputShape);
    input2->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, false);
    std::shared_ptr<RawTensor> ddrRawTensor2 = std::make_shared<RawTensor>(DT_FP32, shape3);
    auto output1 = std::make_shared<LogicalTensor>(*func, ddrRawTensor2, viewOffset1, shape4);
    output1->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, false);
    auto output2 = std::make_shared<LogicalTensor>(*func, ddrRawTensor2, viewOffset2, shape4);
    output2->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, false);
    auto output3 = std::make_shared<LogicalTensor>(*func, ddrRawTensor2, viewOffset3, shape4);
    output3->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, false);
    auto output4 = std::make_shared<LogicalTensor>(*func, ddrRawTensor2, viewOffset4, shape4);
    output4->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, false);

    auto &assemble_op1 = func->AddOperation(Opcode::OP_ASSEMBLE, {input1}, {ubTensor1});
    assemble_op1.SetOpAttribute(std::make_shared<AssembleOpAttribute>(MEM_DEVICE_DDR, assembleOffset1));
    auto &assemble_op2 = func->AddOperation(Opcode::OP_ASSEMBLE, {input2}, {ubTensor1});
    assemble_op2.SetOpAttribute(std::make_shared<AssembleOpAttribute>(MEM_DEVICE_DDR, assembleOffset2));
    auto &reshape_op = func->AddOperation(Opcode::OP_RESHAPE, {ubTensor1}, {ubTensor2});
    reshape_op.SetAttribute(OP_ATTR_PREFIX + "validShape", validShape);
    auto &view_op1 = func->AddOperation(Opcode::OP_VIEW, {ubTensor2}, {output1});
    view_op1.SetOpAttribute(std::make_shared<ViewOpAttribute>(viewOffset1));
    auto &view_op2 = func->AddOperation(Opcode::OP_VIEW, {ubTensor2}, {output2});
    view_op2.SetOpAttribute(std::make_shared<ViewOpAttribute>(viewOffset2));
    auto &view_op3 = func->AddOperation(Opcode::OP_VIEW, {ubTensor2}, {output3});
    view_op3.SetOpAttribute(std::make_shared<ViewOpAttribute>(viewOffset3));
    auto &view_op4 = func->AddOperation(Opcode::OP_VIEW, {ubTensor2}, {output4});
    view_op4.SetOpAttribute(std::make_shared<ViewOpAttribute>(viewOffset4));

    func->inCasts_ = {input1, input2};
    func->outCasts_ = {output1, output2, output3, output4};
    return {input1, input2};
}

/*
验证一对多场景下动态shape的兜底策略
因为缺乏宏构建策略，手动构造expandfunction的输出构图
1) 用例设置：
{2,2,2} -> assemble -> {2,2,4} -> reshape -> {4,4} -> view -> {2,2}
{2,2,2} -> assemble                                -> view -> {2,2}
                                                   -> view -> {2,2}
                                                   -> view -> {2,2}
{2,2,a}                            {4,a}
2) splitreshape
                                    {4, Max(0, GetViewValidShapeDim(a,0,2))}
                                                   -> view -> {2,2}
{2,2,2} -> assemble -> {2,2,2} -> reshape -> {4,2} -> view -> {2,2}
{2,2,2} -> assemble -> {2,2,2} -> reshape -> {4,2} -> view -> {2,2}
                                                   -> view -> {2,2}
{2,2,a}                {2,2,a}               {4,a}
                   {4, Max(0, GetViewValidShapeDim(a,2,2))}
*/
TEST_F(TestSplitReshapePass, TestDynBeCoveredSTest) {
    auto func = std::make_shared<Function>(Program::GetInstance(), "TestReshapeSplit", "TestReshapeSplit", nullptr);
    EXPECT_TRUE(func != nullptr);

    std::vector<int64_t> assembleOffset1 = {kNumZero, kNumZero, kNumZero};
    std::vector<int64_t> assembleOffset2 = {kNumZero, kNumZero, kNumTwo};
    std::vector<SymbolicScalar> dynInputShape = {kNumTwo, kNumTwo, SymbolicScalar("a")};

    auto inputs = BuildDynBeCoveredFunc(func);
    RunPassStra(*func, "SplitReshape");

    std::unordered_map<LogicalTensorPtr, int> inputsWeight = {{inputs[0], 1}, {inputs[1], 10}};
    std::unordered_map<LogicalTensorPtr, Operation*> newAssembles = {{inputs[0], nullptr}, {inputs[1], nullptr}};
    CollectOperations(func, inputsWeight, newAssembles, kNumTwo, kNumFour, 11);

    LogicalTensors reshapeOutputs;
    std::vector<std::string> expectAssembleDynShape = {
        "2",
        "2",
        "RUNTIME_Max(0, (a*RUNTIME_Ne(a, 0)))"
    };
    std::vector<std::string> expectValidShape1 = {
        "4",
        "RUNTIME_Max(RUNTIME_Max(0, ((RUNTIME_GetViewValidShapeDim(a,0,2)*RUNTIME_Ne(RUNTIME_GetViewValidShapeDim(a,0,2), 0))-0)), ((RUNTIME_GetViewValidShapeDim(a,0,2)*RUNTIME_Ne(RUNTIME_GetViewValidShapeDim(a,0,2), 0))-0))"
    };
    std::vector<std::string> expectValidShape2 = {
        "4",
        "RUNTIME_Max(RUNTIME_Max(0, (((RUNTIME_GetViewValidShapeDim(a,2,2)+2)*RUNTIME_Ne(RUNTIME_GetViewValidShapeDim(a,2,2), 0))-2)), (((RUNTIME_GetViewValidShapeDim(a,2,2)+2)*RUNTIME_Ne(RUNTIME_GetViewValidShapeDim(a,2,2), 0))-2))"
    };
    std::unordered_map<LogicalTensorPtr, std::vector<int64_t>> expectAssembleOffset = {{inputs[0], assembleOffset1}, {inputs[1], assembleOffset2}};
    std::unordered_map<LogicalTensorPtr, std::vector<std::string>> expectValidShapes = {{inputs[0], expectValidShape1}, {inputs[1], expectValidShape2}};
    CheckNewAssembles(newAssembles, expectAssembleOffset, expectAssembleDynShape, expectValidShapes, dynInputShape, reshapeOutputs, kNumTwo);
    std::vector<int64_t> expectedShape = {kNumFour, kNumTwo};
    EXPECT_EQ(reshapeOutputs[0]->shape, expectedShape);
    EXPECT_EQ(reshapeOutputs[1]->shape, expectedShape);
    EXPECT_NE(reshapeOutputs[0], reshapeOutputs[1]);
    EXPECT_EQ(reshapeOutputs[0]->GetConsumers().size(), kNumTwo);
    EXPECT_EQ(reshapeOutputs[1]->GetConsumers().size(), kNumTwo);
    auto view1 = *(reshapeOutputs[0]->GetConsumers().begin());
    auto view2 = *(++(reshapeOutputs[0]->GetConsumers().begin()));
    auto view3 = *(reshapeOutputs[1]->GetConsumers().begin());
    auto view4 = *(++(reshapeOutputs[1]->GetConsumers().begin()));
    EXPECT_NE(view1, view2);
    EXPECT_NE(view1, view3);
    EXPECT_NE(view1, view4);
}

LogicalTensors BuildDynPerfectlyMatchWithAllFunc(std::shared_ptr<Function> func){
    std::vector<int64_t> shape1 = {kNumTwo, kNumTwo, kNumTwo};
    std::vector<int64_t> shape2 = {kNumTwo, kNumEight, kNumTwo};
    std::vector<int64_t> shape3 = {kNumTwo, kNumFour, kNumTwo, kNumTwo};
    std::vector<int64_t> shape4 = {kNumTwo, kNumTwo, kNumTwo, kNumTwo};
    std::vector<int64_t> assembleOffset1 = {kNumZero, kNumZero, kNumZero};
    std::vector<int64_t> assembleOffset2 = {kNumZero, kNumTwo, kNumZero};
    std::vector<int64_t> assembleOffset3 = {kNumZero, kNumFour, kNumZero};
    std::vector<int64_t> assembleOffset4 = {kNumZero, kNumSix, kNumZero};
    std::vector<int64_t> viewOffset1 = {kNumZero, kNumZero, kNumZero, kNumZero};
    std::vector<int64_t> viewOffset2 = {kNumZero, kNumTwo, kNumZero, kNumZero};
    std::vector<SymbolicScalar> validShape = {kNumTwo, kNumFour, kNumTwo, SymbolicScalar("a")};
    std::vector<SymbolicScalar> dynInputShape = {kNumTwo, kNumTwo, SymbolicScalar("a")};

    std::shared_ptr<RawTensor> ddrRawTensor1 = std::make_shared<RawTensor>(DT_FP32, shape2);
    std::shared_ptr<RawTensor> ddrRawTensor2 = std::make_shared<RawTensor>(DT_FP32, shape3);
    auto input1 = std::make_shared<LogicalTensor>(*func, ddrRawTensor1, assembleOffset1, shape1, dynInputShape);
    input1->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, false);
    auto input2 = std::make_shared<LogicalTensor>(*func, ddrRawTensor1, assembleOffset2, shape1, dynInputShape);
    input2->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, false);
    auto input3 = std::make_shared<LogicalTensor>(*func, ddrRawTensor1, assembleOffset3, shape1, dynInputShape);
    input3->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, false);
    auto input4 = std::make_shared<LogicalTensor>(*func, ddrRawTensor1, assembleOffset4, shape1, dynInputShape);
    input4->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, false);
    auto ubTensor1 = std::make_shared<LogicalTensor>(*func, DT_FP32, shape2);
    ubTensor1->SetMemoryTypeOriginal(MemoryType::MEM_UNKNOWN, false);
    auto ubTensor2 = std::make_shared<LogicalTensor>(*func, DT_FP32, shape3);
    ubTensor2->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, false);
    auto output1 = std::make_shared<LogicalTensor>(*func, ddrRawTensor2, viewOffset1, shape4);
    output1->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, false);
    auto output2 = std::make_shared<LogicalTensor>(*func, ddrRawTensor2, viewOffset2, shape4);
    output2->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, false);

    auto &assemble_op1 = func->AddOperation(Opcode::OP_ASSEMBLE, {input1}, {ubTensor1});
    assemble_op1.SetOpAttribute(std::make_shared<AssembleOpAttribute>(MEM_DEVICE_DDR, assembleOffset1));
    auto &assemble_op2 = func->AddOperation(Opcode::OP_ASSEMBLE, {input2}, {ubTensor1});
    assemble_op2.SetOpAttribute(std::make_shared<AssembleOpAttribute>(MEM_DEVICE_DDR, assembleOffset2));
    auto &assemble_op3 = func->AddOperation(Opcode::OP_ASSEMBLE, {input3}, {ubTensor1});
    assemble_op3.SetOpAttribute(std::make_shared<AssembleOpAttribute>(MEM_DEVICE_DDR, assembleOffset3));
    auto &assemble_op4 = func->AddOperation(Opcode::OP_ASSEMBLE, {input4}, {ubTensor1});
    assemble_op4.SetOpAttribute(std::make_shared<AssembleOpAttribute>(MEM_DEVICE_DDR, assembleOffset4));
    auto &reshape_op = func->AddOperation(Opcode::OP_RESHAPE, {ubTensor1}, {ubTensor2});
    reshape_op.SetAttribute(OP_ATTR_PREFIX + "validShape", validShape);
    auto &view_op1 = func->AddOperation(Opcode::OP_VIEW, {ubTensor2}, {output1});
    view_op1.SetOpAttribute(std::make_shared<ViewOpAttribute>(viewOffset1));
    auto &view_op2 = func->AddOperation(Opcode::OP_VIEW, {ubTensor2}, {output2});
    view_op2.SetOpAttribute(std::make_shared<ViewOpAttribute>(viewOffset2));

    func->inCasts_ = {input1, input2, input3, input4};
    func->outCasts_ = {output1, output2};
    return {input1, input2, input3, input4};
}

/*
验证多对一场景下动态shape的兜底策略
因为缺乏宏构建策略，手动构造expandfunction的输出构图
1) 用例设置：
{2,2,2} -> assemble -> {2,8,2} -> reshape -> {2,4,2,2} -> view -> {2,2,2,2}
{2,2,2} -> assemble                                    -> view -> {2,2,2,2}
{2,2,2} -> assemble
{2,2,2} -> assemble
{2,2,a}                         {2,4,2,a}
2) splitreshape
{2,2,2} -> assemble ->
{2,2,2} -> assemble -> reshape -> {2,2,2,2} -> view -> {2,2,2,2}
{2,2,2} -> assemble -> reshape -> {2,2,2,2} -> view -> {2,2,2,2}
{2,2,2} -> assemble ->
{2,2,a}           {2,4,a}         {2,2,2,a}
                        {2,2,2,Max(0, RUNTIME_GetViewValidShapeDim(a,0,2))}
*/
TEST_F(TestSplitReshapePass, TestDynPerfectlyMatchWithAllSTest) {
    auto func = std::make_shared<Function>(Program::GetInstance(), "TestReshapeSplit", "TestReshapeSplit", nullptr);
    EXPECT_TRUE(func != nullptr);

    std::vector<int64_t> assembleOffset1 = {kNumZero, kNumZero, kNumZero};
    std::vector<int64_t> assembleOffset2 = {kNumZero, kNumTwo, kNumZero};
    std::vector<int64_t> assembleOffset3 = {kNumZero, kNumFour, kNumZero};
    std::vector<int64_t> assembleOffset4 = {kNumZero, kNumSix, kNumZero};
    std::vector<SymbolicScalar> dynInputShape = {kNumTwo, kNumTwo, SymbolicScalar("a")};

    auto inputs = BuildDynPerfectlyMatchWithAllFunc(func);
    RunPassStra(*func, "SplitReshape");

    std::unordered_map<LogicalTensorPtr, int> inputsWeight = {
        {inputs[0], 1}, {inputs[1], 10},
        {inputs[2], 100}, {inputs[3], 1000}
    };
    std::unordered_map<LogicalTensorPtr, Operation*> newAssembles = {
        {inputs[0], nullptr}, {inputs[1], nullptr},
        {inputs[2], nullptr}, {inputs[3], nullptr}
    };
    CollectOperations(func, inputsWeight, newAssembles, kNumTwo, kNumTwo, 1111);

    std::unordered_map<LogicalTensorPtr, std::vector<int64_t>> expectAssembleOffset = {
        {inputs[0], assembleOffset1}, {inputs[1], assembleOffset2},
        {inputs[2], assembleOffset3}, {inputs[3], assembleOffset4}
    };
    LogicalTensors reshapeOutputs;

    std::vector<std::string> expectAssembleDynShape = {
        "2",
        "4",
        "RUNTIME_Max(RUNTIME_Max(0, (a*RUNTIME_Ne(a, 0))), (a*RUNTIME_Ne(a, 0)))"
    };
    std::vector<std::string> expectValidShape = {
        "2",
        "2",
        "2",
        "RUNTIME_Max(0, ((RUNTIME_GetViewValidShapeDim(a,0,2)*RUNTIME_Ne(RUNTIME_GetViewValidShapeDim(a,0,2), 0))-0))"
    };
    std::unordered_map<LogicalTensorPtr, std::vector<std::string>> expectValidShapes = {
        {inputs[0], expectValidShape}, {inputs[1], expectValidShape},
        {inputs[2], expectValidShape}, {inputs[3], expectValidShape}
    };
    CheckNewAssembles(newAssembles, expectAssembleOffset, expectAssembleDynShape, expectValidShapes, dynInputShape, reshapeOutputs, kNumFour);
    EXPECT_NE(reshapeOutputs[0], reshapeOutputs[3]);
    EXPECT_EQ(reshapeOutputs[0]->GetConsumers().size(), kNumOne);
    EXPECT_EQ(reshapeOutputs[3]->GetConsumers().size(), kNumOne);
    EXPECT_NE(*(reshapeOutputs[0]->GetConsumers().begin()), *(reshapeOutputs[3]->GetConsumers().begin()));
}

// check the number of reshape operations
// return the number of total operations
int CheckOpNum(Function* func, const uint32_t expectReshapeNum){
    int reshapeOp = 0;
    int OpNum = 0;
    for (auto &op : func->Operations()) {
        if (op.GetOpcode() == Opcode::OP_RESHAPE) {
            reshapeOp++;
        }
        OpNum++;
    }
    EXPECT_EQ(reshapeOp, expectReshapeNum);
    return OpNum;
}

/*
splitreshape pass不起作用的场景
一对多场景(使用expandfunction作为前序pass)
assemble的tile输入无法被映射为一个完整inputView的tile
1) 用例设置：
{1,1,2,8} -> exp -> {1,1,2,8} -> reshape -> {1,16} -> exp -> {1,16}
tileshape1 = {2,1,2,4}
tileshape2 = {1,4}
2) expandfunction：
exp -> {1,1,2,4} -> assemble -> reshape -> {1,16} -> view -> {1,4} -> exp -> {1,4}
exp -> {1,1,2,4} -> assemble                      -> view -> {1,4} -> exp -> {1,4}
                                                  -> view -> {1,4} -> exp -> {1,4}
                                                  -> view -> {1,4} -> exp -> {1,4}
*/
TEST_F(TestSplitReshapePass, TestExceptionCase1) {
    //Define the shape of the Tensors
    std::vector<int64_t> origShape = {kNumOne, kNumOne, kNumTwo, kNumEight};
    std::vector<int64_t> reshapeShape = {kNumOne, kExpFour};
    std::vector<int64_t> tiledShape1 = {kNumTwo, kNumOne, kNumTwo, kNumFour};
    std::vector<int64_t> tiledShape2 = {kNumOne, kNumFour};

    Tensor input(DT_FP32, origShape, "input");
    Tensor output(DT_FP32, reshapeShape, "output");

    FUNCTION("STCase5") {
        TileShape::Current().SetVecTile(tiledShape1);
        Tensor exp = Exp(input);
        Tensor reshape = Reshape(exp, reshapeShape);
        TileShape::Current().SetVecTile(tiledShape2);
        output = Exp(reshape);
    }

    Function* func = Program::GetInstance().GetFunctionByRawName("TENSOR_STCase5");

    RunPassStra(*func, "ExpandFunction");
    int OpNum = CheckOpNum(func, kNumOne);
    RunPassStra(*func, "SplitReshape");
    int AfterOpNum = CheckOpNum(func, kNumOne);
    EXPECT_EQ(AfterOpNum, OpNum);
}

/*
splitreshape pass不起作用的场景
使用expandfunction作为前序pass
无法计算reshape前后的加细
1) 用例设置：
{64,6} -> exp -> {64,6} -> reshape -> {96,4} -> exp -> {96,4}
tileshape = {32,2}
2) expandfunction：
exp -> {32,2} -> assemble -> {64,6} -> reshape -> {96,4} -> view -> {32,2} -> exp -> {32,2}
exp -> {32,2} -> assemble                                -> view -> {32,2} -> exp -> {32,2}
exp -> {32,2} -> assemble                                -> view -> {32,2} -> exp -> {32,2}
exp -> {32,2} -> assemble                                -> view -> {32,2} -> exp -> {32,2}
exp -> {32,2} -> assemble                                -> view -> {32,2} -> exp -> {32,2}
exp -> {32,2} -> assemble                                -> view -> {32,2} -> exp -> {32,2}
*/
TEST_F(TestSplitReshapePass, TestExceptionCase2) {
    //Define the shape of the Tensors
    std::vector<int64_t> origShape = {kExpSix, kNumSix};
    std::vector<int64_t> reshapeShape = {kNumNineSix, kNumFour};
    std::vector<int64_t> tiledShape = {kExpFive, kNumTwo};

    TileShape::Current().SetVecTile(tiledShape);
    Tensor input(DT_FP32, origShape, "input");
    Tensor output(DT_FP32, reshapeShape, "output");

    FUNCTION("STCase6") {
        Tensor exp = Exp(input);
        Tensor reshape = Reshape(exp, reshapeShape);
        output = Exp(reshape);
    }

    Function* func = Program::GetInstance().GetFunctionByRawName("TENSOR_STCase6");

    RunPassStra(*func, "ExpandFunction");

    CheckOpNum(func, kNumOne);
    RunPassStra(*func, "SplitReshape");
    CheckOpNum(func, kNumOne);
}

/*
splitreshape pass不起作用的场景
多对多的场景，前后的tile即非cover也非covered
1) 用例设置：
{8,8} -> exp -> {8,8} -> reshape -> {16,4} -> exp -> {16,4}
tileshape1 = {2,4}
tileshape1 = {4,2}
2) expandfunction：
exp -> {2,4} -> assemble -> {8,8} -> reshape -> {16,4} -> view -> {4,2} -> exp -> {4,2}
exp -> {2,4} -> assemble                               -> view -> {4,2} -> exp -> {4,2}
exp -> {2,4} -> assemble                               -> view -> {4,2} -> exp -> {4,2}
exp -> {2,4} -> assemble                               -> view -> {4,2} -> exp -> {4,2}
exp -> {2,4} -> assemble                               -> view -> {4,2} -> exp -> {4,2}
exp -> {2,4} -> assemble                               -> view -> {4,2} -> exp -> {4,2}
*/
TEST_F(TestSplitReshapePass, TestExceptionCase3) {
    //Define the shape of the Tensors
    std::vector<int64_t> origShape = {kNumEight, kNumEight};
    std::vector<int64_t> reshapeShape = {kExpFour, kNumFour};
    std::vector<int64_t> tiledShape1 = {kNumTwo, kNumFour};
    std::vector<int64_t> tiledShape2 = {kNumFour, kNumTwo};

    Tensor input(DT_FP32, origShape, "input");
    Tensor output(DT_FP32, reshapeShape, "output");

    FUNCTION("STCase7") {
        TileShape::Current().SetVecTile(tiledShape1);
        Tensor exp = Exp(input);
        Tensor reshape = Reshape(exp, reshapeShape);
        TileShape::Current().SetVecTile(tiledShape2);
        output = Exp(reshape);
    }

    Function* func = Program::GetInstance().GetFunctionByRawName("TENSOR_STCase7");

    RunPassStra(*func, "ExpandFunction");

    CheckOpNum(func, kNumOne);
    RunPassStra(*func, "SplitReshape");
    CheckOpNum(func, kNumOne);
}

/*
splitreshape pass不起作用的场景
动态shape位于变化轴
                                 {2,2,2,a}
{2,2,2} -> assemble -> {2,2,4} -> reshape -> {2,2,2,2} -> view -> {2,2,1,2}
{2,2,2} -> assemble                                    -> view -> {2,2,1,2}
*/
TEST_F(TestSplitReshapePass, TestExceptionCase4) {
    //Define the shape of the Tensors
    auto func = std::make_shared<Function>(Program::GetInstance(), "TestReshapeSplit", "TestReshapeSplit", nullptr);
    EXPECT_TRUE(func != nullptr);

    std::vector<int64_t> shape1 = {kNumTwo, kNumTwo, kNumTwo};
    std::vector<int64_t> shape2 = {kNumTwo, kNumTwo, kNumFour};
    std::vector<int64_t> shape3 = {kNumTwo, kNumTwo, kNumTwo, kNumTwo};
    std::vector<int64_t> shape4 = {kNumTwo, kNumTwo, kNumOne, kNumTwo};
    std::vector<int64_t> assembleOffset1 = {kNumZero, kNumZero, kNumZero};
    std::vector<int64_t> assembleOffset2 = {kNumZero, kNumZero, kNumTwo};
    std::vector<int64_t> viewOffset1 = {kNumZero, kNumZero, kNumZero, kNumZero};
    std::vector<int64_t> viewOffset2 = {kNumZero, kNumZero, kNumOne, kNumZero};
    std::vector<SymbolicScalar> validShape = {kNumTwo, kNumTwo, kNumTwo, SymbolicScalar("a")};

    std::shared_ptr<RawTensor> ddrRawTensor1 = std::make_shared<RawTensor>(DT_FP32, shape2);
    std::shared_ptr<RawTensor> ddrRawTensor2 = std::make_shared<RawTensor>(DT_FP32, shape3);
    auto input1 = std::make_shared<LogicalTensor>(*func, ddrRawTensor1, assembleOffset1, shape1);
    input1->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, false);
    auto input2 = std::make_shared<LogicalTensor>(*func, ddrRawTensor1, assembleOffset2, shape1);
    input2->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, false);
    auto ubTensor1 = std::make_shared<LogicalTensor>(*func, DT_FP32, shape2);
    ubTensor1->SetMemoryTypeOriginal(MemoryType::MEM_UNKNOWN, false);
    auto ubTensor2 = std::make_shared<LogicalTensor>(*func, DT_FP32, shape3);
    ubTensor2->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, false);
    auto output1 = std::make_shared<LogicalTensor>(*func, ddrRawTensor2, viewOffset1, shape4);
    output1->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, false);
    auto output2 = std::make_shared<LogicalTensor>(*func, ddrRawTensor2, viewOffset2, shape4);
    output2->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, false);

    auto &reshape_op = func->AddOperation(Opcode::OP_RESHAPE, {ubTensor1}, {ubTensor2});
    reshape_op.SetAttribute(OP_ATTR_PREFIX + "validShape", validShape);
    auto &view_op1 = func->AddOperation(Opcode::OP_VIEW, {ubTensor2}, {output1});
    view_op1.SetOpAttribute(std::make_shared<ViewOpAttribute>(viewOffset1));
    auto &view_op2 = func->AddOperation(Opcode::OP_VIEW, {ubTensor2}, {output2});
    view_op2.SetOpAttribute(std::make_shared<ViewOpAttribute>(viewOffset2));
    auto &assemble_op1 = func->AddOperation(Opcode::OP_ASSEMBLE, {input1}, {ubTensor1});
    assemble_op1.SetOpAttribute(std::make_shared<AssembleOpAttribute>(MEM_DEVICE_DDR, assembleOffset1));
    auto &assemble_op2 = func->AddOperation(Opcode::OP_ASSEMBLE, {input2}, {ubTensor1});
    assemble_op2.SetOpAttribute(std::make_shared<AssembleOpAttribute>(MEM_DEVICE_DDR, assembleOffset2));

    func->inCasts_.push_back(input1);
    func->inCasts_.push_back(input2);
    func->outCasts_.push_back(output1);
    func->outCasts_.push_back(output2);

    CheckOpNum(func.get(), kNumOne);
    RunPassStra(*func, "SplitReshape");
    CheckOpNum(func.get(), kNumOne);
}

/*
splitreshape pass不起作用的场景
多对一场景(使用expandfunction作为前序pass)
assemble的tile输入无法被映射为一个完整inputView的tile
exp -> {1,4} -> assemble -> {1,16} -> reshape -> {1,1,2,8} -> view -> {1,1,2,4} -> exp -> {1,1,2,4}
exp -> {1,4} -> assemble                                   -> view -> {1,1,2,4} -> exp -> {1,1,2,4}
exp -> {1,4} -> assemble
exp -> {1,4} -> assemble
*/
TEST_F(TestSplitReshapePass, TestExceptionCase5) {
    //Define the shape of the Tensors
    std::vector<int64_t> origShape = {kNumOne, kExpFour};
    std::vector<int64_t> reshapeShape = {kNumOne, kNumOne, kNumTwo, kNumEight};
    std::vector<int64_t> tiledShape1 = {kNumOne, kNumFour};
    std::vector<int64_t> tiledShape2 = {kNumTwo, kNumOne, kNumTwo, kNumFour};
    std::vector<int64_t> tiledreshapeShape = {kNumOne, kNumFour};
    std::vector<int64_t> tiledassembleShape = {kNumOne, kNumOne, kNumOne, kNumFour};
    std::vector<int64_t> tiledviewShape = {kNumOne, kNumOne, kNumTwo, kNumFour};

    Tensor input(DT_FP32, origShape, "input");
    Tensor output(DT_FP32, reshapeShape, "output");

    FUNCTION("STCase8") {
        TileShape::Current().SetVecTile(tiledShape1);
        Tensor exp = Exp(input);
        Tensor reshape = Reshape(exp, reshapeShape);
        TileShape::Current().SetVecTile(tiledShape2);
        output = Exp(reshape);
    }

    Function* func = Program::GetInstance().GetFunctionByRawName("TENSOR_STCase8");

    RunPassStra(*func, "ExpandFunction");
    int OpNum = CheckOpNum(func, kNumOne);
    RunPassStra(*func, "SplitReshape");
    int AfterOpNum = CheckOpNum(func, kNumOne);
    EXPECT_EQ(AfterOpNum, OpNum);
}
}
}