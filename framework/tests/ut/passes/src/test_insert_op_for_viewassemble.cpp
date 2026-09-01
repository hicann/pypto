/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_insert_op_for_viewassemble.cpp
 * \brief Unit test for InsertOpForViewAssemble pass.
 */

#include <gtest/gtest.h>
#include "symbolic_scalar_test_utils.h"
#include <vector>
#include <string>
#include "computational_graph_builder.h"
#include "interface/function/function.h"
#include "tilefwk/tilefwk.h"
#include "ut_json/ut_json_tool.h"
#include "passes/pass_mgr/pass_manager.h"
#include "interface/configs/config_manager.h"

#include "interface/tensor/irbuilder.h"
#define private public
#include "passes/tile_graph_pass/graph_optimization/insert_op_for_viewassemble.h"

namespace npu {
namespace tile_fwk {
static const size_t kSizeZero = 0UL;
static const uint16_t kNumFour = 4u;
static const size_t kSizeEight = 8UL;
static const size_t kSizeTwelve = 12UL;
static const uint16_t kNumExpFour = 16u;
static const uint16_t kNumExpEight = 64u;

class TestInsertCopyPass : public ::testing::Test {
public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}

    void SetUp() override
    {
        Program::GetInstance().Reset();
        config::Reset();
        config::SetHostOption(COMPILE_STAGE, CS_EXECUTE_GRAPH);
        config::SetPlatformConfig(KEY_ENABLE_COST_MODEL, false);
    }
    void TearDown() override {}
};

TEST_F(TestInsertCopyPass, TestNormalCase)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestNormalCase", "TestNormalCase",
                                                      nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);
    /*
               | ------- view --- t0 --- assemble ------- |
             | ------- view --- t1 --- assemble  ---------- |
    inTensor [16, 16]                                         outTensor [16, 16]
             | ------- view --- t2 --- assemble  ---------- |
               | ------- view --- t3 --- assemble ------- |
 */
    // Prepare the graph
    std::vector<int64_t> shape = {kNumExpFour, kNumExpFour};
    std::vector<int64_t> midShape = {kNumExpFour, kNumFour};
    std::vector<int64_t> offset0 = {kSizeZero, kSizeZero};
    std::vector<int64_t> offset1 = {kSizeZero, kNumFour};
    std::vector<int64_t> offset2 = {kSizeZero, kSizeEight};
    std::vector<int64_t> offset3 = {kSizeZero, kSizeTwelve};

    auto inTensor = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    inTensor->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, true);
    auto outTensor = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    outTensor->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, true);
    auto midTensor0 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, midShape, CreateTestConstIntVector(midShape));
    midTensor0->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, true);
    auto midTensor1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, midShape, CreateTestConstIntVector(midShape));
    midTensor1->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, true);
    auto midTensor2 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, midShape, CreateTestConstIntVector(midShape));
    midTensor2->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, true);
    auto midTensor3 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, midShape, CreateTestConstIntVector(midShape));
    midTensor3->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, true);

    auto& viewOp0 = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_VIEW, {inTensor}, {midTensor0});
    viewOp0.SetOpAttribute(std::make_shared<ViewOpAttribute>(offset0, MemoryType::MEM_DEVICE_DDR));
    auto& viewOp1 = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_VIEW, {inTensor}, {midTensor1});
    viewOp1.SetOpAttribute(std::make_shared<ViewOpAttribute>(offset1, MemoryType::MEM_DEVICE_DDR));
    auto& viewOp2 = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_VIEW, {inTensor}, {midTensor2});
    viewOp2.SetOpAttribute(std::make_shared<ViewOpAttribute>(offset2, MemoryType::MEM_DEVICE_DDR));
    auto& viewOp3 = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_VIEW, {inTensor}, {midTensor3});
    viewOp3.SetOpAttribute(std::make_shared<ViewOpAttribute>(offset3, MemoryType::MEM_DEVICE_DDR));

    auto& assOp0 = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_ASSEMBLE, {midTensor0}, {outTensor});
    assOp0.SetOpAttribute(std::make_shared<AssembleOpAttribute>(MemoryType::MEM_DEVICE_DDR, offset0));
    auto& assOp1 = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_ASSEMBLE, {midTensor1}, {outTensor});
    assOp1.SetOpAttribute(std::make_shared<AssembleOpAttribute>(MemoryType::MEM_DEVICE_DDR, offset1));
    auto& assOp2 = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_ASSEMBLE, {midTensor2}, {outTensor});
    assOp2.SetOpAttribute(std::make_shared<AssembleOpAttribute>(MemoryType::MEM_DEVICE_DDR, offset2));
    auto& assOp3 = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_ASSEMBLE, {midTensor3}, {outTensor});
    assOp3.SetOpAttribute(std::make_shared<AssembleOpAttribute>(MemoryType::MEM_DEVICE_DDR, offset3));

    InsertOpForViewAssemble pass;
    EXPECT_EQ(pass.RunOnFunction(*currFunctionPtr), SUCCESS);
    EXPECT_EQ(currFunctionPtr->Operations().size(), kSizeEight);
    EXPECT_EQ(midTensor0->GetMemoryTypeOriginal(), MemoryType::MEM_DEVICE_DDR);
}

TEST_F(TestInsertCopyPass, TestNoEqualSize)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestNormalCase", "TestNormalCase",
                                                      nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);
    /*
               | ------- view --- t0 --- assemble ------- |
             | ------- view --- t1 --- assemble  ---------- |
    inTensor [16, 16]                                         outTensor [16, 64]
             | ------- view --- t2 --- assemble  ---------- |
               | ------- view --- t3 --- assemble ------- |
 */
    // Prepare the graph
    std::vector<int64_t> shape = {kNumExpFour, kNumExpFour};
    std::vector<int64_t> shape1 = {kNumExpFour, kNumExpEight};
    std::vector<int64_t> midShape = {kNumExpFour, kNumFour};
    std::vector<int64_t> offset0 = {kSizeZero, kSizeZero};
    std::vector<int64_t> offset1 = {kSizeZero, kNumFour};
    std::vector<int64_t> offset2 = {kSizeZero, kSizeEight};
    std::vector<int64_t> offset3 = {kSizeZero, kSizeTwelve};

    auto inTensor = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    inTensor->SetMemoryTypeOriginal(MemoryType::MEM_UB, true);
    auto outTensor = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape1, CreateTestConstIntVector(shape1));
    outTensor->SetMemoryTypeOriginal(MemoryType::MEM_UB, true);
    auto midTensor0 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, midShape, CreateTestConstIntVector(midShape));
    midTensor0->SetMemoryTypeOriginal(MemoryType::MEM_UB, true);
    auto midTensor1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, midShape, CreateTestConstIntVector(midShape));
    midTensor1->SetMemoryTypeOriginal(MemoryType::MEM_UB, true);
    auto midTensor2 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, midShape, CreateTestConstIntVector(midShape));
    midTensor2->SetMemoryTypeOriginal(MemoryType::MEM_UB, true);
    auto midTensor3 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, midShape, CreateTestConstIntVector(midShape));
    midTensor3->SetMemoryTypeOriginal(MemoryType::MEM_UB, true);

    auto& viewOp0 = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_VIEW, {inTensor}, {midTensor0});
    viewOp0.SetOpAttribute(std::make_shared<ViewOpAttribute>(offset0, MemoryType::MEM_UB));
    auto& viewOp1 = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_VIEW, {inTensor}, {midTensor1});
    viewOp1.SetOpAttribute(std::make_shared<ViewOpAttribute>(offset1, MemoryType::MEM_UB));
    auto& viewOp2 = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_VIEW, {inTensor}, {midTensor2});
    viewOp2.SetOpAttribute(std::make_shared<ViewOpAttribute>(offset2, MemoryType::MEM_UB));
    auto& viewOp3 = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_VIEW, {inTensor}, {midTensor3});
    viewOp3.SetOpAttribute(std::make_shared<ViewOpAttribute>(offset3, MemoryType::MEM_UB));

    auto& assOp0 = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_ASSEMBLE, {midTensor0}, {outTensor});
    assOp0.SetOpAttribute(std::make_shared<AssembleOpAttribute>(MemoryType::MEM_UB, offset0));
    auto& assOp1 = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_ASSEMBLE, {midTensor1}, {outTensor});
    assOp1.SetOpAttribute(std::make_shared<AssembleOpAttribute>(MemoryType::MEM_UB, offset1));
    auto& assOp2 = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_ASSEMBLE, {midTensor2}, {outTensor});
    assOp2.SetOpAttribute(std::make_shared<AssembleOpAttribute>(MemoryType::MEM_UB, offset2));
    auto& assOp3 = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_ASSEMBLE, {midTensor3}, {outTensor});
    assOp3.SetOpAttribute(std::make_shared<AssembleOpAttribute>(MemoryType::MEM_UB, offset3));

    InsertOpForViewAssemble pass;
    EXPECT_EQ(pass.RunOnFunction(*currFunctionPtr), SUCCESS);
    EXPECT_EQ(currFunctionPtr->Operations().size(), kNumExpFour);
}

TEST_F(TestInsertCopyPass, TestInsert)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestNormalCase", "TestNormalCase",
                                                      nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);
    /*

             | ------- view ---------- t1 ----------- assemble  ---------- |
    inTensor [16, 16]                                                       outTensor [16, 16]
             | ------- view --- t2 --- EXP --- t3 --- assemble  ---------- |

 */
    std::vector<int64_t> shape = {kNumExpFour, kNumExpFour};
    std::vector<int64_t> midShape = {kNumExpFour, kSizeEight};
    std::vector<int64_t> offset0 = {kSizeZero, kSizeZero};
    std::vector<int64_t> offset1 = {kSizeZero, kSizeEight};

    auto inTensor = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    inTensor->SetMemoryTypeOriginal(MemoryType::MEM_UB, true);
    auto outTensor = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    outTensor->SetMemoryTypeOriginal(MemoryType::MEM_UB, true);
    auto midTensor0 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, midShape, CreateTestConstIntVector(midShape));
    midTensor0->SetMemoryTypeOriginal(MemoryType::MEM_UB, true);
    auto midTensor1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, midShape, CreateTestConstIntVector(midShape));
    midTensor1->SetMemoryTypeOriginal(MemoryType::MEM_UB, true);
    auto midTensor2 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, midShape, CreateTestConstIntVector(midShape));
    midTensor2->SetMemoryTypeOriginal(MemoryType::MEM_UB, true);

    auto& viewOp0 = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_VIEW, {inTensor}, {midTensor0});
    viewOp0.SetOpAttribute(std::make_shared<ViewOpAttribute>(offset0, MemoryType::MEM_UB));
    auto& viewOp1 = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_VIEW, {inTensor}, {midTensor1});
    viewOp1.SetOpAttribute(std::make_shared<ViewOpAttribute>(offset1, MemoryType::MEM_UB));
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_EXP, {midTensor1}, {midTensor2});
    auto& assOp0 = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_ASSEMBLE, {midTensor0}, {outTensor});
    assOp0.SetOpAttribute(std::make_shared<AssembleOpAttribute>(MemoryType::MEM_UB, offset0));
    auto& assOp1 = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_ASSEMBLE, {midTensor2}, {outTensor});
    assOp1.SetOpAttribute(std::make_shared<AssembleOpAttribute>(MemoryType::MEM_UB, offset1));

    InsertOpForViewAssemble pass;
    EXPECT_EQ(pass.RunOnFunction(*currFunctionPtr), SUCCESS);
    const int result = 9;
    EXPECT_EQ(currFunctionPtr->Operations().size(), result);
    size_t insertedContractCount = 0;
    size_t insertedSliceCount = 0;
    for (const auto& op : currFunctionPtr->Operations()) {
        if (op.GetOpcode() == config::GetContractOpcode()) {
            insertedContractCount++;
        } else if (op.GetOpcode() == config::GetSliceOpcode()) {
            insertedSliceCount++;
        }
    }
    EXPECT_EQ(insertedContractCount, 2u);
    EXPECT_EQ(insertedSliceCount, 2u);
}

TEST_F(TestInsertCopyPass, TestSharedDdrAssembleInputInsertCopyInOut)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(),
                                                      "TestSharedDdrAssembleInputInsertCopyInOut",
                                                      "TestSharedDdrAssembleInputInsertCopyInOut", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);
    /*
        inTensor [16,16] --- OP_EXP ---> sharedInput [16,16] (DDR, shared)
                                           |                          |
                                      OP_ASSEMBLE                 OP_EXP
                                           |                          |
                                   outTensorA [16,16] (DDR)   outTensorB [16,16] (DDR)

        sharedInput is a DDR tensor produced by OP_EXP and consumed by both an OP_ASSEMBLE
        and an OP_EXP whose output RawTensor differs from the assemble output. This triggers
        InsertCopiesForSharedAssembleInputs, which must insert OP_COPY_IN (DDR->UB) and
        OP_COPY_OUT (UB->DDR) and rewire the assemble input to the COPY_OUT result.
        The producer is OP_EXP (not OP_VIEW) so JudgedViewAssemble does not insert copies.
     */
    std::vector<int64_t> shape = {kNumExpFour, kNumExpFour};
    std::vector<int64_t> offsetZero = {kSizeZero, kSizeZero};

    auto inTensor = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    inTensor->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, true);
    auto sharedInput = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    sharedInput->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, true);
    auto outTensorA = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    outTensorA->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, true);
    auto outTensorB = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    outTensorB->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, true);

    // producer of sharedInput
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_EXP, {inTensor}, {sharedInput});
    auto& assOp = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_ASSEMBLE, {sharedInput}, {outTensorA});
    assOp.SetOpAttribute(std::make_shared<AssembleOpAttribute>(MemoryType::MEM_DEVICE_DDR, offsetZero));
    // another consumer of sharedInput with a different output RawTensor
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_EXP, {sharedInput}, {outTensorB});

    InsertOpForViewAssemble pass;
    EXPECT_EQ(pass.RunOnFunction(*currFunctionPtr), SUCCESS);

    // 3 original ops + OP_COPY_IN + OP_COPY_OUT
    const int result = 5;
    EXPECT_EQ(currFunctionPtr->Operations().size(), result);

    Operation* copyInOp = nullptr;
    Operation* copyOutOp = nullptr;
    Operation* assembleOpFound = nullptr;
    size_t copyInCount = kSizeZero;
    size_t copyOutCount = kSizeZero;
    for (auto& op : currFunctionPtr->Operations()) {
        if (op.GetOpcode() == Opcode::OP_COPY_IN) {
            copyInOp = &op;
            copyInCount++;
        } else if (op.GetOpcode() == Opcode::OP_COPY_OUT) {
            copyOutOp = &op;
            copyOutCount++;
        } else if (op.GetOpcode() == Opcode::OP_ASSEMBLE) {
            assembleOpFound = &op;
        }
    }
    EXPECT_EQ(copyInCount, 1UL);
    EXPECT_EQ(copyOutCount, 1UL);
    ASSERT_TRUE(copyInOp != nullptr);
    ASSERT_TRUE(copyOutOp != nullptr);
    ASSERT_TRUE(assembleOpFound != nullptr);

    // COPY_IN: sharedInput(DDR) -> ubTensor(UB)
    EXPECT_EQ(copyInOp->GetIOperands()[0], sharedInput);
    EXPECT_EQ(copyInOp->GetIOperands()[0]->GetMemoryTypeOriginal(), MemoryType::MEM_DEVICE_DDR);
    EXPECT_EQ(copyInOp->GetOOperands()[0]->GetMemoryTypeOriginal(), MemoryType::MEM_UB);

    // COPY_OUT input comes from COPY_IN output; ubTensor(UB) -> ddrTensor(DDR)
    EXPECT_EQ(copyOutOp->GetIOperands()[0], copyInOp->GetOOperands()[0]);
    EXPECT_EQ(copyOutOp->GetIOperands()[0]->GetMemoryTypeOriginal(), MemoryType::MEM_UB);
    EXPECT_EQ(copyOutOp->GetOOperands()[0]->GetMemoryTypeOriginal(), MemoryType::MEM_DEVICE_DDR);

    // Assemble new input comes from COPY_OUT result, i.e. temp path DDR->UB->DDR
    EXPECT_EQ(assembleOpFound->GetIOperands()[0], copyOutOp->GetOOperands()[0]);
    EXPECT_EQ(assembleOpFound->GetIOperands()[0]->GetMemoryTypeOriginal(), MemoryType::MEM_DEVICE_DDR);

    // sharedInput stays DDR
    EXPECT_EQ(sharedInput->GetMemoryTypeOriginal(), MemoryType::MEM_DEVICE_DDR);
}

TEST_F(TestInsertCopyPass, TestOversizedSharedDdrAssembleInputSkipsCopyInOut)
{
    auto function = std::make_shared<Function>(Program::GetInstance(), "TestOversizedSharedDdrInput",
                                               "TestOversizedSharedDdrInput", nullptr);
    std::vector<int64_t> shape = {128, 2048};
    std::vector<int64_t> offset = {0, 0};
    auto makeDdrTensor = [&shape]() {
        auto tensor = IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
        tensor->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, true);
        return tensor;
    };

    auto input = makeDdrTensor();
    auto sharedInput = makeDdrTensor();
    auto assembleOutput = makeDdrTensor();
    auto otherOutput = makeDdrTensor();
    PassOperationUtils::AddOperation(*function, Opcode::OP_EXP, {input}, {sharedInput});
    auto& assemble = PassOperationUtils::AddOperation(*function, Opcode::OP_ASSEMBLE, {sharedInput}, {assembleOutput});
    assemble.SetOpAttribute(std::make_shared<AssembleOpAttribute>(MemoryType::MEM_DEVICE_DDR, offset));
    PassOperationUtils::AddOperation(*function, Opcode::OP_EXP, {sharedInput}, {otherOutput});

    InsertOpForViewAssemble pass;
    EXPECT_EQ(pass.RunOnFunction(*function), SUCCESS);
    auto operations = function->Operations();
    EXPECT_EQ(operations.size(), 3u);
    EXPECT_EQ(assemble.GetIOperands().front(), sharedInput);
    for (const auto& op : operations) {
        EXPECT_NE(op.GetOpcode(), Opcode::OP_COPY_IN);
        EXPECT_NE(op.GetOpcode(), Opcode::OP_COPY_OUT);
    }
}
} // namespace tile_fwk
} // namespace npu
