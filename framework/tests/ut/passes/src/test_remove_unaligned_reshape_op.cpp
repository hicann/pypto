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
 * \file test_assign_memory_type_unalign.cpp
 * \brief Unit test for RemoveRedundantReshape pass.
 */

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>
#include <fstream>
#include <vector>
#include <string>
#include "interface/function/function.h"
#include "passes/tile_graph_pass/graph_constraint/remove_unaligned_reshape_op.h"
#include "passes/tile_graph_pass/graph_constraint/pad_local_buffer.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "passes/pass_mgr/pass_manager.h"
#include "interface/configs/config_manager.h"

using namespace npu::tile_fwk;

static const uint16_t kNumOne = 1u;

class TestRemoveUnalignedReshapeOp : public ::testing::Test {
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

inline void ConstructGraph1(std::shared_ptr<Function>& currFunctionPtr)
{
    // Prepare the graph
    std::vector<int64_t> shape = {7, 15};
    std::vector<int64_t> reshape_shape = {15, 7};
    std::vector<int64_t> expect_shape = {7, 16};
    auto shapeImme = OpImmediate::Specified(shape);
    auto incast1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    auto ubTensor1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    ubTensor1->SetMemoryTypeBoth(MEM_UB);
    auto ubTensor2 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, reshape_shape);
    ubTensor2->SetMemoryTypeBoth(MEM_UB);
    auto outCast = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, reshape_shape);
    outCast->UpdateDynValidShape({SymbolicScalar("output_0_Dim_0"), SymbolicScalar("output_0_Dim_1")});
    auto& copy_op1 = currFunctionPtr->AddRawOperation(Opcode::OP_COPY_IN, {incast1}, {ubTensor1});
    auto copyin1Attr = std::make_shared<CopyOpAttribute>(
        OpImmediate::Specified({0, 0}), MEM_UB, shapeImme, shapeImme, std::vector<npu::tile_fwk::OpImmediate>());
    std::vector<npu::tile_fwk::OpImmediate> toValidShape = {
        OpImmediate(SymbolicScalar("Input_0_Dim_0")), OpImmediate(SymbolicScalar("Input_0_Dim_1"))};
    copyin1Attr->SetToDynValidShape(toValidShape);
    copy_op1.SetOpAttribute(copyin1Attr);
    auto& reshape_op = currFunctionPtr->AddRawOperation(Opcode::OP_RESHAPE, {ubTensor1}, {ubTensor2});
    (void)reshape_op;
    auto& copy_out_op = currFunctionPtr->AddRawOperation(Opcode::OP_COPY_OUT, {ubTensor2}, {outCast});
    (void)copy_out_op;
    currFunctionPtr->inCasts_.push_back(incast1);
    currFunctionPtr->outCasts_.push_back(outCast);
}

inline void ConstructGraph2(std::shared_ptr<Function>& currFunctionPtr)
{
    // Prepare the graph
    std::vector<int64_t> shape = {8, 16};
    std::vector<int64_t> reshape_shape = {16, 8};
    std::vector<int64_t> expect_shape = {8, 16};
    auto shapeImme = OpImmediate::Specified(shape);
    auto incast1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    auto ubTensor1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    ubTensor1->SetMemoryTypeBoth(MEM_UB);
    auto ubTensor2 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, reshape_shape);
    ubTensor2->SetMemoryTypeBoth(MEM_UB);
    auto outCast = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, reshape_shape);
    outCast->UpdateDynValidShape({SymbolicScalar("output_0_Dim_0"), SymbolicScalar("output_0_Dim_1")});
    auto& copy_op1 = currFunctionPtr->AddRawOperation(Opcode::OP_COPY_IN, {incast1}, {ubTensor1});
    auto copyin1Attr = std::make_shared<CopyOpAttribute>(
        OpImmediate::Specified({0, 0}), MEM_UB, shapeImme, shapeImme, std::vector<npu::tile_fwk::OpImmediate>());
    std::vector<npu::tile_fwk::OpImmediate> toValidShape = {
        OpImmediate(SymbolicScalar("Input_0_Dim_0")), OpImmediate(SymbolicScalar("Input_0_Dim_1"))};
    copyin1Attr->SetToDynValidShape(toValidShape);
    copy_op1.SetOpAttribute(copyin1Attr);
    auto& reshape_op = currFunctionPtr->AddRawOperation(Opcode::OP_RESHAPE, {ubTensor1}, {ubTensor2});
    (void)reshape_op;
    auto& copy_out_op = currFunctionPtr->AddRawOperation(Opcode::OP_COPY_OUT, {ubTensor2}, {outCast});
    (void)copy_out_op;
    currFunctionPtr->inCasts_.push_back(incast1);
    currFunctionPtr->outCasts_.push_back(outCast);
}

inline void ConstructGraph3(std::shared_ptr<Function>& currFunctionPtr)
{
    // Prepare the graph
    std::vector<int64_t> shape = {8, 16};
    std::vector<int64_t> reshape_shape = {16, 8};
    std::vector<int64_t> expect_shape = {8, 16};
    auto shapeImme = OpImmediate::Specified(shape);
    auto incast1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    incast1->SetMemoryTypeBoth(MEM_DEVICE_DDR);
    auto ubTensor1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    ubTensor1->SetMemoryTypeBoth(MEM_UB);
    auto outCast = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, reshape_shape);
    outCast->UpdateDynValidShape({SymbolicScalar("output_0_Dim_0"), SymbolicScalar("output_0_Dim_1")});
    auto& reshape_op = currFunctionPtr->AddRawOperation(Opcode::OP_RESHAPE, {incast1}, {ubTensor1});
    (void)reshape_op;
    auto& copy_out_op = currFunctionPtr->AddRawOperation(Opcode::OP_COPY_OUT, {ubTensor1}, {outCast});
    (void)copy_out_op;
}

inline void ConstructGraph4(std::shared_ptr<Function>& currFunctionPtr)
{
    // Prepare the graph
    std::vector<int64_t> shape = {64, 1};
    std::vector<int64_t> reshape_shape = {1, 64};
    auto shapeImme = OpImmediate::Specified(shape);
    auto incast1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    incast1->SetMemoryTypeBoth(MEM_DEVICE_DDR);
    auto ubTensor1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    ubTensor1->SetMemoryTypeBoth(MEM_UB);
    auto outCast = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, reshape_shape);
    outCast->UpdateDynValidShape({SymbolicScalar("output_0_Dim_0"), SymbolicScalar("output_0_Dim_1")});
    auto& reshape_op = currFunctionPtr->AddRawOperation(Opcode::OP_RESHAPE, {incast1}, {ubTensor1});
    (void)reshape_op;
    auto& copy_out_op = currFunctionPtr->AddRawOperation(Opcode::OP_COPY_OUT, {ubTensor1}, {outCast});
    (void)copy_out_op;
}
/*
before:
    copyin
    [7,15]
     |
    reshape
    [15,7]
     |
    copyout
    [15,7]

after:
    copyin
    [7,15]
      |
    copyout
    [7,15]
      |
    reshape
    [15,7]
      |
    copyin
    [15,8]
      |
    copyout
    [15,7]
*/
TEST_F(TestRemoveUnalignedReshapeOp, reshaped_padded_ub)
{
    auto currFunctionPtr =
        std::make_shared<Function>(Program::GetInstance(), "TestPadLocalBuffer", "TestPadLocalBuffer", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);
    std::vector<int64_t> shape = {7, 15};
    std::vector<int64_t> reshape_shape = {15, 7};
    std::vector<int64_t> expect_shape = {7, 16};
    ConstructGraph1(currFunctionPtr);
    PadLocalBuffer padLocalBufferTest;
    padLocalBufferTest.RunOnFunction(*currFunctionPtr);
    RemoveUnalignedReshape removeUnalignedReshapeOpTest;
    removeUnalignedReshapeOpTest.RunOnFunction(*currFunctionPtr);
    for (auto& op : currFunctionPtr->Operations()) {
        if (op.GetOpcode() == Opcode::OP_RESHAPE) {
            for (auto& out : op.oOperand) {
                if (out->oriShape == reshape_shape) {
                    EXPECT_EQ(out->shape, reshape_shape);
                    EXPECT_EQ(out->tensor->rawshape, reshape_shape);
                }
                EXPECT_EQ(out->GetConsumers().size(), 1);
                auto consumer = *(out->GetConsumers().begin());
                EXPECT_EQ(consumer->GetOpcode(), Opcode::OP_COPY_IN);
            }
            for (auto& in : op.iOperand) {
                EXPECT_EQ(in->GetProducers().size(), 1);
                if (in->oriShape == shape) {
                    EXPECT_EQ(in->shape, shape);
                    EXPECT_EQ(in->tensor->rawshape, shape);
                }
                auto producer = *(in->GetProducers().begin());
                EXPECT_EQ(producer->GetOpcode(), Opcode::OP_COPY_OUT);
            }
        }
    }
}

/*
before:
    copyin
    [8,16]
     |
    reshape
    [16,8]
     |
    copyout
    [16,8]

after:
    copyin
    [8,16]
     |
    reshape
    [16,8]
     |
    copyout
    [16,8]
*/
TEST_F(TestRemoveUnalignedReshapeOp, reshaped_unpadded_ub)
{
    auto currFunctionPtr =
        std::make_shared<Function>(Program::GetInstance(), "TestPadLocalBuffer", "TestPadLocalBuffer", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);
    std::vector<int64_t> shape = {8, 16};
    std::vector<int64_t> reshape_shape = {16, 8};
    std::vector<int64_t> expect_shape = {8, 16};
    ConstructGraph2(currFunctionPtr);
    PadLocalBuffer padLocalBufferTest;
    padLocalBufferTest.RunOnFunction(*currFunctionPtr);
    RemoveUnalignedReshape removeUnalignedReshapeOpTest;
    removeUnalignedReshapeOpTest.RunOnFunction(*currFunctionPtr);
    for (auto& op : currFunctionPtr->Operations()) {
        if (op.GetOpcode() == Opcode::OP_RESHAPE) {
            for (auto& in : op.iOperand) {
                EXPECT_EQ(in->GetProducers().size(), 1);
                auto producer = *(in->GetProducers().begin());
                EXPECT_NE(producer->GetOpcode(), Opcode::OP_COPY_OUT);
                if (in->oriShape == shape) {
                    EXPECT_EQ(in->shape, shape);
                    EXPECT_EQ(in->tensor->rawshape, shape);
                }
            }
            for (auto& out : op.oOperand) {
                auto consumer = *(out->GetConsumers().begin());
                EXPECT_EQ(out->GetConsumers().size(), 1);
                if (out->oriShape == reshape_shape) {
                    EXPECT_EQ(out->shape, reshape_shape);
                    EXPECT_EQ(out->tensor->rawshape, reshape_shape);
                }
                EXPECT_NE(consumer->GetOpcode(), Opcode::OP_COPY_IN);
            }
        }
    }
}

/*
before:
    incast(gm)
    [8,16]
     |
    reshape
    [16,8]
     |
    copyout
    [16,8]

after:
    incast(gm)
    [8,16]
     |
    reshape
    [16,8]
     |
    copyout
    [16,8]
*/
TEST_F(TestRemoveUnalignedReshapeOp, reshaped_unpadded_ub_gm)
{
    auto currFunctionPtr =
        std::make_shared<Function>(Program::GetInstance(), "TestPadLocalBuffer", "TestPadLocalBuffer", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);
    std::vector<int64_t> shape = {8, 16};
    std::vector<int64_t> reshape_shape = {16, 8};
    std::vector<int64_t> expect_shape = {8, 16};
    ConstructGraph3(currFunctionPtr);
    PadLocalBuffer padLocalBufferTest;
    padLocalBufferTest.RunOnFunction(*currFunctionPtr);
    RemoveUnalignedReshape removeUnalignedReshapeOpTest;
    removeUnalignedReshapeOpTest.RunOnFunction(*currFunctionPtr);
    for (auto& op : currFunctionPtr->Operations()) {
        if (op.GetOpcode() == Opcode::OP_RESHAPE) {
            for (auto& out : op.oOperand) {
                EXPECT_EQ(out->GetConsumers().size(), 1);
                auto consumer = *(out->GetConsumers().begin());
                EXPECT_NE(consumer->GetOpcode(), Opcode::OP_COPY_IN);
                if (out->oriShape == reshape_shape) {
                    EXPECT_EQ(out->shape, reshape_shape);
                    EXPECT_EQ(out->tensor->rawshape, reshape_shape);
                }
            }
        }
    }
}

/*
问题用例
before:
    copyin
    [64,1]
     |
    reshape
    [1, 64]
     |
    copyout
    [1, 64]

after:
    copyin
    [64, 1]
      |
    copyout
    [64, 1]
      |
    reshape
    [1, 64]
      |
    copyin
    [1, 64]
      |
    copyout
    [1, 64]
*/
TEST_F(TestRemoveUnalignedReshapeOp, reshaped_unpadded_ub_gm_last_dim_1)
{
    auto currFunctionPtr =
        std::make_shared<Function>(Program::GetInstance(), "TestPadLocalBuffer", "TestPadLocalBuffer", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);
    std::vector<int64_t> shape = {64, 1};
    std::vector<int64_t> reshape_shape = {1, 64};
    ConstructGraph4(currFunctionPtr);
    PadLocalBuffer padLocalBufferTest;
    padLocalBufferTest.RunOnFunction(*currFunctionPtr);
    RemoveUnalignedReshape removeUnalignedReshapeOpTest;
    removeUnalignedReshapeOpTest.RunOnFunction(*currFunctionPtr);
    for (auto& op : currFunctionPtr->Operations()) {
        if (op.GetOpcode() == Opcode::OP_RESHAPE) {
            for (auto& in : op.iOperand) {
                EXPECT_EQ(in->GetProducers().size(), 1);
                auto producer = *(in->GetProducers().begin());
                EXPECT_EQ(producer->GetOpcode(), Opcode::OP_COPY_OUT);
                if (in->oriShape == shape) {
                    EXPECT_EQ(in->shape, shape);
                    EXPECT_EQ(in->tensor->rawshape, shape);
                }
            }
            for (auto& out : op.oOperand) {
                EXPECT_EQ(out->GetConsumers().size(), 1);
                auto consumer = *(out->GetConsumers().begin());
                EXPECT_EQ(consumer->GetOpcode(), Opcode::OP_COPY_IN);
                if (out->oriShape == reshape_shape) {
                    EXPECT_EQ(out->shape, reshape_shape);
                    EXPECT_EQ(out->tensor->rawshape, reshape_shape);
                }
            }
        }
    }
}

// in - COPYIN - COPYOUT - RESHAPE - COPYIN - COPYOUT - out
//                                 - COPYIN - COPYOUT - out

// in - COPYIN - COPYOUT - COPYIN - RESHAPECOPYOUT - RESHAPE - RESHAPECOPYIN - COPYOUT - COPYIN - COPYOUT - out
//                                                           - RESHAPECOPYIN - COPYOUT - COPYIN - COPYOUT - out
TEST_F(TestRemoveUnalignedReshapeOp, TestCopyToReshapeCopyOnL1)
{
    auto currFunctionPtr =
        std::make_shared<Function>(Program::GetInstance(), "TestCopyToReshapeCopy", "TestCopyToReshapeCopy", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    // Prepare the graph
    std::vector<int64_t> inShape = {4, 4};
    std::vector<int64_t> reshapeShape = {2, 8};
    std::vector<int64_t> outShape = {4, 8};
    auto incast = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, inShape);
    auto copyinTensor1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, inShape);
    copyinTensor1->SetMemoryTypeBoth(MemoryType::MEM_L1, true);
    auto copyoutTensor1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, inShape);
    copyoutTensor1->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    std::vector<SymbolicScalar> validShape1 = {SymbolicScalar("Input_0_Dim_0"), SymbolicScalar("Input_0_Dim_1")};
    copyoutTensor1->UpdateDynValidShape(validShape1);
    auto reshapeTensor = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, reshapeShape);
    reshapeTensor->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    std::vector<SymbolicScalar> validShape2 = {SymbolicScalar("Input_1_Dim_0"), SymbolicScalar("Input_1_Dim_1")};
    reshapeTensor->UpdateDynValidShape(validShape2);
    auto copyinTensor2 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, reshapeShape);
    copyinTensor2->SetMemoryTypeBoth(MemoryType::MEM_L1, true);
    auto copyinTensor3 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, reshapeShape);
    copyinTensor3->SetMemoryTypeBoth(MemoryType::MEM_L1, true);
    auto outcast = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, outShape);
    outcast->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);

    currFunctionPtr->AddOperation(Opcode::OP_COPY_IN, {incast}, {copyinTensor1});
    currFunctionPtr->AddOperation(Opcode::OP_COPY_OUT, {copyinTensor1}, {copyoutTensor1});
    currFunctionPtr->AddOperation(Opcode::OP_RESHAPE, {copyoutTensor1}, {reshapeTensor});
    currFunctionPtr->AddOperation(Opcode::OP_COPY_IN, {reshapeTensor}, {copyinTensor2});
    currFunctionPtr->AddOperation(Opcode::OP_COPY_IN, {reshapeTensor}, {copyinTensor3});
    currFunctionPtr->AddOperation(Opcode::OP_COPY_OUT, {copyinTensor2}, {outcast});
    currFunctionPtr->AddOperation(Opcode::OP_COPY_OUT, {copyinTensor3}, {outcast});

    currFunctionPtr->inCasts_.push_back(incast);
    currFunctionPtr->outCasts_.push_back(outcast);

    RemoveUnalignedReshape removeUnalignedReshapeOpTest;
    int curSize = currFunctionPtr->Operations().size();
    EXPECT_EQ(removeUnalignedReshapeOpTest.RunOnFunction(*currFunctionPtr), SUCCESS);
    EXPECT_EQ(currFunctionPtr->Operations().size(), curSize + 6);
}

// in - COPYIN - COPYOUT - RESHAPE - COPYIN - COPYOUT - out
//                                 - COPYIN - COPYOUT - out

// in - COPYIN - RESHAPECOPYOUT - RESHAPE - RESHAPECOPYIN - COPYOUT - out
//                                        - RESHAPECOPYIN - COPYOUT - out
TEST_F(TestRemoveUnalignedReshapeOp, TestCopyToReshapeCopyOnUB)
{
    auto currFunctionPtr =
        std::make_shared<Function>(Program::GetInstance(), "TestCopyToReshapeCopy", "TestCopyToReshapeCopy", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    // Prepare the graph
    std::vector<int64_t> inShape = {4, 4};
    std::vector<int64_t> reshapeShape = {2, 8};
    std::vector<int64_t> outShape = {4, 8};
    auto incast = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, inShape);
    auto copyin_Tensor1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, inShape);
    copyin_Tensor1->SetMemoryTypeBoth(MemoryType::MEM_UB, true);
    auto copyout_Tensor1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, inShape);
    copyout_Tensor1->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    std::vector<SymbolicScalar> validShape1 = {SymbolicScalar("Input_00_Dim_0"), SymbolicScalar("Input_00_Dim_1")};
    copyout_Tensor1->UpdateDynValidShape(validShape1);
    auto reshape_Tensor = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, reshapeShape);
    reshape_Tensor->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    std::vector<SymbolicScalar> validShape2 = {SymbolicScalar("Input_01_Dim_0"), SymbolicScalar("Input_01_Dim_1")};
    reshape_Tensor->UpdateDynValidShape(validShape2);
    auto copyin_Tensor2 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, reshapeShape);
    copyin_Tensor2->SetMemoryTypeBoth(MemoryType::MEM_UB, true);
    auto copyin_Tensor3 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, reshapeShape);
    copyin_Tensor3->SetMemoryTypeBoth(MemoryType::MEM_UB, true);
    auto outcast = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, outShape);
    outcast->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);

    currFunctionPtr->AddOperation(Opcode::OP_COPY_IN, {incast}, {copyin_Tensor1});
    auto& copyoutOp = currFunctionPtr->AddOperation(Opcode::OP_COPY_OUT, {copyin_Tensor1}, {copyout_Tensor1});
    currFunctionPtr->AddOperation(Opcode::OP_RESHAPE, {copyout_Tensor1}, {reshape_Tensor});
    auto& copyinOp1 = currFunctionPtr->AddOperation(Opcode::OP_COPY_IN, {reshape_Tensor}, {copyin_Tensor2});
    auto& copyinOp2 = currFunctionPtr->AddOperation(Opcode::OP_COPY_IN, {reshape_Tensor}, {copyin_Tensor3});
    currFunctionPtr->AddOperation(Opcode::OP_COPY_OUT, {copyin_Tensor2}, {outcast});
    currFunctionPtr->AddOperation(Opcode::OP_COPY_OUT, {copyin_Tensor3}, {outcast});

    currFunctionPtr->inCasts_.push_back(incast);
    currFunctionPtr->outCasts_.push_back(outcast);

    RemoveUnalignedReshape removeUnalignedReshapeOpTest;
    int curOpSize = currFunctionPtr->Operations().size();
    EXPECT_EQ(removeUnalignedReshapeOpTest.RunOnFunction(*currFunctionPtr), SUCCESS);
    EXPECT_EQ(copyoutOp.GetOpcode() == Opcode::OP_RESHAPE_COPY_OUT, true);
    EXPECT_EQ(copyinOp1.GetOpcode() == Opcode::OP_RESHAPE_COPY_IN, true);
    EXPECT_EQ(copyinOp2.GetOpcode() == Opcode::OP_RESHAPE_COPY_IN, true);
    EXPECT_EQ(currFunctionPtr->Operations().size(), curOpSize);
}

// in - COPYIN - COPYOUT - RESHAPE - COPYIN - COPYOUT - out
//    - COPYIN - COPYOUT           - COPYIN - COPYOUT - out

// 不变
TEST_F(TestRemoveUnalignedReshapeOp, TestCopyToReshapeBeforeMultCopyOutOnL1)
{
    auto currFunctionPtr =
        std::make_shared<Function>(Program::GetInstance(), "TestCopyToReshapeCopy", "TestCopyToReshapeCopy", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    // Prepare the graph
    std::vector<int64_t> inShape = {2, 2};
    std::vector<int64_t> copyoutShape = {2, 4};
    std::vector<int64_t> reshapeShape = {4, 2};
    std::vector<int64_t> outShape = {4, 4};
    auto incast = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, inShape);
    auto copyin_Tensor1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, inShape);
    copyin_Tensor1->SetMemoryTypeBoth(MemoryType::MEM_L1, true);
    auto copyout_Tensor1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, copyoutShape);
    copyout_Tensor1->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    std::vector<SymbolicScalar> validShape1 = {SymbolicScalar("Input_00_Dim_0"), SymbolicScalar("Input_00_Dim_1")};
    copyout_Tensor1->UpdateDynValidShape(validShape1);
    auto copyin_Tensor2 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, inShape);
    copyin_Tensor2->SetMemoryTypeBoth(MemoryType::MEM_L1, true);
    auto reshape_Tensor = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, reshapeShape);
    reshape_Tensor->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    std::vector<SymbolicScalar> validShape3 = {SymbolicScalar("Input_02_Dim_0"), SymbolicScalar("Input_02_Dim_1")};
    reshape_Tensor->UpdateDynValidShape(validShape3);
    auto copyin_Tensor3 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, reshapeShape);
    copyin_Tensor3->SetMemoryTypeBoth(MemoryType::MEM_L1, true);
    auto copyin_Tensor4 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, reshapeShape);
    copyin_Tensor4->SetMemoryTypeBoth(MemoryType::MEM_L1, true);
    auto outcast = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, outShape);
    outcast->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);

    currFunctionPtr->AddOperation(Opcode::OP_COPY_IN, {incast}, {copyin_Tensor1});
    currFunctionPtr->AddOperation(Opcode::OP_COPY_IN, {incast}, {copyin_Tensor2});
    currFunctionPtr->AddOperation(Opcode::OP_COPY_OUT, {copyin_Tensor1}, {copyout_Tensor1});
    currFunctionPtr->AddOperation(Opcode::OP_COPY_OUT, {copyin_Tensor2}, {copyout_Tensor1});
    currFunctionPtr->AddOperation(Opcode::OP_RESHAPE, {copyout_Tensor1}, {reshape_Tensor});
    currFunctionPtr->AddOperation(Opcode::OP_COPY_IN, {reshape_Tensor}, {copyin_Tensor3});
    currFunctionPtr->AddOperation(Opcode::OP_COPY_IN, {reshape_Tensor}, {copyin_Tensor4});
    currFunctionPtr->AddOperation(Opcode::OP_COPY_OUT, {copyin_Tensor2}, {outcast});
    currFunctionPtr->AddOperation(Opcode::OP_COPY_OUT, {copyin_Tensor3}, {outcast});

    currFunctionPtr->inCasts_.push_back(incast);
    currFunctionPtr->outCasts_.push_back(outcast);

    RemoveUnalignedReshape removeUnalignedReshapeOpTest;
    int curOpSize = currFunctionPtr->Operations().size();
    EXPECT_EQ(removeUnalignedReshapeOpTest.RunOnFunction(*currFunctionPtr), SUCCESS);
    int expOpSize = curOpSize + 6;
    EXPECT_EQ(currFunctionPtr->Operations().size(), expOpSize);
}

// in - COPYIN - COPYOUT - RESHAPE - COPYIN   - COPYOUT - out1
//                                 - ASSEMBLE - out2

// in - COPYIN - RESHAPECOPYOUT - RESHAPE - RESHAPECOPYIN - COPYOUT - out1
//                                        - RESHAPECOPYIN - COPYOUT - ASSEMBLE - out2
TEST_F(TestRemoveUnalignedReshapeOp, TestCopyToReshapeConsumerAssembleOnUB)
{
    auto currFunctionPtr =
        std::make_shared<Function>(Program::GetInstance(), "TestCopyToReshapeCopy", "TestCopyToReshapeCopy", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    // Prepare the graph
    std::vector<int64_t> inShape = {8, 8};
    std::vector<int64_t> reshapeShape = {4, 16};
    std::vector<int64_t> outShape = {8, 16};
    auto incast = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, inShape);
    auto copyInTensor_1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, inShape);
    copyInTensor_1->SetMemoryTypeBoth(MemoryType::MEM_UB, true);
    auto copyOutTensor_1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, inShape);
    copyOutTensor_1->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    std::vector<SymbolicScalar> validShape_1 = {SymbolicScalar("Input_0_Dim_0"), SymbolicScalar("Input_0_Dim_1")};
    copyOutTensor_1->UpdateDynValidShape(validShape_1);
    auto reshapeTensor = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, reshapeShape);
    reshapeTensor->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    std::vector<SymbolicScalar> validShape_2 = {SymbolicScalar("Input_1_Dim_0"), SymbolicScalar("Input_1_Dim_1")};
    reshapeTensor->UpdateDynValidShape(validShape_2);
    auto copyInTensor_2 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, reshapeShape);
    copyInTensor_2->SetMemoryTypeBoth(MemoryType::MEM_UB, true);
    auto outcast_1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, outShape);
    outcast_1->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    auto outcast_2 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, outShape);
    outcast_2->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);

    currFunctionPtr->AddOperation(Opcode::OP_COPY_IN, {incast}, {copyInTensor_1});
    currFunctionPtr->AddOperation(Opcode::OP_COPY_OUT, {copyInTensor_1}, {copyOutTensor_1});
    currFunctionPtr->AddOperation(Opcode::OP_RESHAPE, {copyOutTensor_1}, {reshapeTensor});
    currFunctionPtr->AddOperation(Opcode::OP_COPY_IN, {reshapeTensor}, {copyInTensor_2});
    currFunctionPtr->AddOperation(Opcode::OP_COPY_OUT, {copyInTensor_2}, {outcast_1});
    currFunctionPtr->AddOperation(Opcode::OP_ASSEMBLE, {reshapeTensor}, {outcast_2});

    currFunctionPtr->inCasts_.push_back(incast);
    currFunctionPtr->outCasts_.push_back(outcast_1);
    currFunctionPtr->outCasts_.push_back(outcast_2);

    RemoveUnalignedReshape removeUnalignedReshapeOpTest;
    int curSize = currFunctionPtr->Operations().size();
    EXPECT_EQ(removeUnalignedReshapeOpTest.RunOnFunction(*currFunctionPtr), SUCCESS);
    EXPECT_EQ(currFunctionPtr->Operations().size(), curSize + 2);
}
// in - COPYIN - COPYOUT - ASSEMBLE - RESHAPE - COPYIN - COPYOUT - out1
//                       - ASSEMBLE - out2
TEST_F(TestRemoveUnalignedReshapeOp, TestBranchBetweenCopyOutReshape1)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestBranchBetweenCopyOutReshape1", "TestBranchBetweenCopyOutReshape1", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    // Prepare the graph
    std::vector<int64_t> inShape = {8, 8};
    std::vector<int64_t> reshapeShape = {4, 16};
    std::vector<int64_t> outShape = {8, 16};
    DataType dataType = DT_FP32;
    auto incast = std::make_shared<LogicalTensor>(*currFunctionPtr, dataType, inShape);
    incast->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    auto copyInTensor_1 = std::make_shared<LogicalTensor>(*currFunctionPtr, dataType, inShape);
    copyInTensor_1->SetMemoryTypeBoth(MemoryType::MEM_UB, true);
    std::vector<SymbolicScalar> validShape_1 = {SymbolicScalar("Input_0_Dim_0"), SymbolicScalar("Input_0_Dim_1")};
    copyInTensor_1->UpdateDynValidShape(validShape_1);
    auto copyOutTensor_1 = std::make_shared<LogicalTensor>(*currFunctionPtr, dataType, inShape);
    copyOutTensor_1->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    auto assembleTensor = std::make_shared<LogicalTensor>(*currFunctionPtr, dataType, inShape);
    assembleTensor->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    std::vector<SymbolicScalar> validShape_2 = {SymbolicScalar("Input_1_Dim_0"), SymbolicScalar("Input_1_Dim_1")};
    assembleTensor->UpdateDynValidShape(validShape_2);
    auto reshapeTensor = std::make_shared<LogicalTensor>(*currFunctionPtr, dataType, reshapeShape);
    reshapeTensor->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    std::vector<SymbolicScalar> validShape_3 = {SymbolicScalar("Input_2_Dim_0"), SymbolicScalar("Input_2_Dim_1")};
    reshapeTensor->UpdateDynValidShape(validShape_3);
    auto copyInTensor_2 = std::make_shared<LogicalTensor>(*currFunctionPtr, dataType, reshapeShape);
    copyInTensor_2->SetMemoryTypeBoth(MemoryType::MEM_UB, true);
    auto copyInTensor_3 = std::make_shared<LogicalTensor>(*currFunctionPtr, dataType, reshapeShape);
    copyInTensor_3->SetMemoryTypeBoth(MemoryType::MEM_UB, true);
    auto outcast_1 = std::make_shared<LogicalTensor>(*currFunctionPtr, dataType, outShape);
    outcast_1->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    auto outcast_2 = std::make_shared<LogicalTensor>(*currFunctionPtr, dataType, outShape);
    outcast_2->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);

    currFunctionPtr->AddOperation(Opcode::OP_COPY_IN, {incast}, {copyInTensor_1});
    auto& copyoutOp = currFunctionPtr->AddOperation(Opcode::OP_COPY_OUT, {copyInTensor_1}, {copyOutTensor_1});
    auto copyoutAttr = std::make_shared<CopyOpAttribute>(MEM_UB, OpImmediate::Specified({0, 0}),
        OpImmediate::Specified(inShape), OpImmediate::Specified(inShape), std::vector<npu::tile_fwk::OpImmediate>());
    copyoutOp.SetOpAttribute(copyoutAttr);
    currFunctionPtr->AddOperation(Opcode::OP_ASSEMBLE, {copyOutTensor_1}, {assembleTensor});
    currFunctionPtr->AddOperation(Opcode::OP_RESHAPE, {assembleTensor}, {reshapeTensor});
    currFunctionPtr->AddOperation(Opcode::OP_COPY_IN, {reshapeTensor}, {copyInTensor_2});
    currFunctionPtr->AddOperation(Opcode::OP_COPY_OUT, {copyInTensor_2}, {outcast_1});
    currFunctionPtr->AddOperation(Opcode::OP_ASSEMBLE, {copyOutTensor_1}, {outcast_2});

    currFunctionPtr->inCasts_.push_back(incast);
    currFunctionPtr->outCasts_.push_back(outcast_1);
    currFunctionPtr->outCasts_.push_back(outcast_2);

    RemoveUnalignedReshape removeUnalignedReshapeOpTest;
    int curSize = currFunctionPtr->Operations().size();
    EXPECT_EQ(removeUnalignedReshapeOpTest.RunOnFunction(*currFunctionPtr), SUCCESS);
    int expSize = curSize + 1;
    EXPECT_EQ(currFunctionPtr->Operations().size(), expSize);
}

/*
before:
    inCast(DDR) - COPYIN(UB) - COPYOUT(DDR) - RESHAPE(DDR) - ADD(DDR) - outCast(DDR)

after:
    inCast(DDR) - COPYIN(UB) - RESHAPECOPYOUT(DDR) - RESHAPE(DDR) - RESHAPECOPYIN(UB) - COPYOUT(DDR) - ADD(DDR) - outCast(DDR)

Test HandleNoCopyInConsumer: reshape output has no COPY_IN consumer, only ADD consumer.
*/
TEST_F(TestRemoveUnalignedReshapeOp, TestHandleNoCopyInConsumerWithAddOp)
{
    auto currFunctionPtr =
        std::make_shared<Function>(Program::GetInstance(), "TestHandleNoCopyInConsumer", "TestHandleNoCopyInConsumer", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    std::vector<int64_t> inShape = {8, 8};
    std::vector<int64_t> reshapeShape = {4, 16};
    std::vector<int64_t> outShape = {4, 16};
    DataType dataType = DT_FP32;

    auto incast = std::make_shared<LogicalTensor>(*currFunctionPtr, dataType, inShape);
    incast->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    auto copyinTensor = std::make_shared<LogicalTensor>(*currFunctionPtr, dataType, inShape);
    copyinTensor->SetMemoryTypeBoth(MemoryType::MEM_UB, true);
    auto copyoutTensor = std::make_shared<LogicalTensor>(*currFunctionPtr, dataType, inShape);
    copyoutTensor->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    std::vector<SymbolicScalar> validShape1 = {SymbolicScalar("Input_0_Dim_0"), SymbolicScalar("Input_0_Dim_1")};
    copyoutTensor->UpdateDynValidShape(validShape1);
    auto reshapeTensor = std::make_shared<LogicalTensor>(*currFunctionPtr, dataType, reshapeShape);
    reshapeTensor->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    std::vector<SymbolicScalar> validShape2 = {SymbolicScalar("Input_1_Dim_0"), SymbolicScalar("Input_1_Dim_1")};
    reshapeTensor->UpdateDynValidShape(validShape2);
    auto addTensor = std::make_shared<LogicalTensor>(*currFunctionPtr, dataType, reshapeShape);
    addTensor->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    auto outcast = std::make_shared<LogicalTensor>(*currFunctionPtr, dataType, outShape);
    outcast->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);

    currFunctionPtr->AddOperation(Opcode::OP_COPY_IN, {incast}, {copyinTensor});
    currFunctionPtr->AddOperation(Opcode::OP_COPY_OUT, {copyinTensor}, {copyoutTensor});
    currFunctionPtr->AddOperation(Opcode::OP_RESHAPE, {copyoutTensor}, {reshapeTensor});
    currFunctionPtr->AddOperation(Opcode::OP_ADD, {reshapeTensor}, {addTensor});
    currFunctionPtr->AddOperation(Opcode::OP_COPY_OUT, {addTensor}, {outcast});

    currFunctionPtr->inCasts_.push_back(incast);
    currFunctionPtr->outCasts_.push_back(outcast);

    RemoveUnalignedReshape removeUnalignedReshapeOpTest;
    int curSize = currFunctionPtr->Operations().size();
    EXPECT_EQ(removeUnalignedReshapeOpTest.RunOnFunction(*currFunctionPtr), SUCCESS);
    int expSize = curSize + 2;
    EXPECT_EQ(currFunctionPtr->Operations().size(), expSize);

    uint32_t reshapeCopyinNum = 0;
    for (auto& op : currFunctionPtr->Operations()) {
        if (op.GetOpcode() == Opcode::OP_RESHAPE_COPY_IN) {
            reshapeCopyinNum++;
        }
    }
    EXPECT_EQ(reshapeCopyinNum, kNumOne);
}

/*
before:
    inCast(DDR) - COPYIN(UB) - COPYOUT(DDR) - RESHAPE(DDR) - ADD(DDR) - outCast1(DDR)
                                            - MUL(DDR) - outCast2(DDR)

after:
    inCast(DDR) - COPYIN(UB) - RESHAPECOPYOUT(DDR) - RESHAPE(DDR) - RESHAPECOPYIN(UB) - COPYOUT(DDR) - ADD(DDR) - outCast1(DDR)
                                                                                                     - MUL(DDR) - outCast2(DDR)

Test HandleNoCopyInConsumer: reshape output has no COPY_IN consumer, has multiple other consumers (ADD, MUL).
*/
TEST_F(TestRemoveUnalignedReshapeOp, TestHandleNoCopyInConsumerWithMultiOps)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestHandleNoCopyInConsumerMulti", "TestHandleNoCopyInConsumerMulti", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    std::vector<int64_t> inShape = {8, 8};
    std::vector<int64_t> reshapeShape = {4, 16};

    auto incast = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, inShape);
    incast->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    auto copyinTensor = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, inShape);
    copyinTensor->SetMemoryTypeBoth(MemoryType::MEM_UB, true);
    auto copyoutTensor = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, inShape);
    copyoutTensor->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    std::vector<SymbolicScalar> validShape1 = {SymbolicScalar("Input_0_Dim_0"), SymbolicScalar("Input_0_Dim_1")};
    copyoutTensor->UpdateDynValidShape(validShape1);
    auto reshapeTensor = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, reshapeShape);
    reshapeTensor->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    std::vector<SymbolicScalar> validShape2 = {SymbolicScalar("Input_1_Dim_0"), SymbolicScalar("Input_1_Dim_1")};
    reshapeTensor->UpdateDynValidShape(validShape2);
    auto addTensor = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, reshapeShape);
    addTensor->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    auto mulTensor = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, reshapeShape);
    mulTensor->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    auto outcast1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, reshapeShape);
    outcast1->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    auto outcast2 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, reshapeShape);
    outcast2->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);

    currFunctionPtr->AddOperation(Opcode::OP_COPY_IN, {incast}, {copyinTensor});
    currFunctionPtr->AddOperation(Opcode::OP_COPY_OUT, {copyinTensor}, {copyoutTensor});
    currFunctionPtr->AddOperation(Opcode::OP_RESHAPE, {copyoutTensor}, {reshapeTensor});
    currFunctionPtr->AddOperation(Opcode::OP_ADD, {reshapeTensor}, {addTensor});
    currFunctionPtr->AddOperation(Opcode::OP_MUL, {reshapeTensor}, {mulTensor});
    currFunctionPtr->AddOperation(Opcode::OP_COPY_OUT, {addTensor}, {outcast1});
    currFunctionPtr->AddOperation(Opcode::OP_COPY_OUT, {mulTensor}, {outcast2});

    currFunctionPtr->inCasts_.push_back(incast);
    currFunctionPtr->outCasts_.push_back(outcast1);
    currFunctionPtr->outCasts_.push_back(outcast2);

    RemoveUnalignedReshape removeUnalignedReshapeOpTest;
    EXPECT_EQ(removeUnalignedReshapeOpTest.RunOnFunction(*currFunctionPtr), SUCCESS);

    uint32_t reshapeCopyinNum = 0;
    uint32_t reshapeCopyoutNum = 0;

    for (auto& op : currFunctionPtr->Operations()) {
        if (op.GetOpcode() == Opcode::OP_RESHAPE_COPY_IN) {
            reshapeCopyinNum++;
        }
        if (op.GetOpcode() == Opcode::OP_RESHAPE_COPY_OUT) {
            reshapeCopyoutNum++;
        }
    }
    EXPECT_EQ(reshapeCopyoutNum, kNumOne);
    EXPECT_EQ(reshapeCopyinNum, kNumOne);
}

// in - COPYIN - COPYOUT - ASSEMBLE - RESHAPE - COPYIN - COPYOUT - out1
//                                  - ASSEMBLE - out2
TEST_F(TestRemoveUnalignedReshapeOp, TestBranchBetweenCopyOutReshape2)
{
    auto currFunctionPtr =
        std::make_shared<Function>(Program::GetInstance(), "TestBranchBetweenCopyOutReshape2", "TestBranchBetweenCopyOutReshape2", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    // Prepare the graph
    std::vector<int64_t> inShape = {8, 8};
    std::vector<int64_t> reshapeShape = {4, 16};
    std::vector<int64_t> outShape = {8, 16};
    DataType dataType32 = DT_FP32;
    auto incast = std::make_shared<LogicalTensor>(*currFunctionPtr, dataType32, inShape);
    incast->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    auto copyInTensor_1 = std::make_shared<LogicalTensor>(*currFunctionPtr, dataType32, inShape);
    copyInTensor_1->SetMemoryTypeBoth(MemoryType::MEM_UB, true);
    auto copyOutTensor_1 = std::make_shared<LogicalTensor>(*currFunctionPtr, dataType32, inShape);
    copyOutTensor_1->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    auto assembleTensor = std::make_shared<LogicalTensor>(*currFunctionPtr, dataType32, inShape);
    assembleTensor->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    std::vector<SymbolicScalar> validShape_1 = {SymbolicScalar("Input_0_Dim_0"), SymbolicScalar("Input_0_Dim_1")};
    assembleTensor->UpdateDynValidShape(validShape_1);
    auto reshapeTensor = std::make_shared<LogicalTensor>(*currFunctionPtr, dataType32, reshapeShape);
    reshapeTensor->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    std::vector<SymbolicScalar> validShape_2 = {SymbolicScalar("Input_1_Dim_0"), SymbolicScalar("Input_1_Dim_1")};
    reshapeTensor->UpdateDynValidShape(validShape_2);
    auto copyInTensor_2 = std::make_shared<LogicalTensor>(*currFunctionPtr, dataType32, reshapeShape);
    copyInTensor_2->SetMemoryTypeBoth(MemoryType::MEM_UB, true);
    auto copyInTensor_3 = std::make_shared<LogicalTensor>(*currFunctionPtr, dataType32, reshapeShape);
    copyInTensor_3->SetMemoryTypeBoth(MemoryType::MEM_UB, true);
    auto outcast_1 = std::make_shared<LogicalTensor>(*currFunctionPtr, dataType32, outShape);
    outcast_1->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    auto outcast_2 = std::make_shared<LogicalTensor>(*currFunctionPtr, dataType32, outShape);
    outcast_2->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);

    currFunctionPtr->AddOperation(Opcode::OP_COPY_IN, {incast}, {copyInTensor_1});
    auto& copyoutOp = currFunctionPtr->AddOperation(Opcode::OP_COPY_OUT, {copyInTensor_1}, {copyOutTensor_1});
    auto copyoutAttr = std::make_shared<CopyOpAttribute>(MEM_UB, OpImmediate::Specified({0, 0}),OpImmediate::Specified(inShape),
        OpImmediate::Specified(inShape), std::vector<npu::tile_fwk::OpImmediate>());
    copyoutOp.SetOpAttribute(copyoutAttr);
    currFunctionPtr->AddOperation(Opcode::OP_ASSEMBLE, {copyOutTensor_1}, {assembleTensor});
    currFunctionPtr->AddOperation(Opcode::OP_RESHAPE, {assembleTensor}, {reshapeTensor});
    currFunctionPtr->AddOperation(Opcode::OP_COPY_IN, {reshapeTensor}, {copyInTensor_2});
    currFunctionPtr->AddOperation(Opcode::OP_COPY_OUT, {copyInTensor_2}, {outcast_1});
    currFunctionPtr->AddOperation(Opcode::OP_ASSEMBLE, {assembleTensor}, {outcast_2});

    currFunctionPtr->inCasts_.push_back(incast);
    currFunctionPtr->outCasts_.push_back(outcast_1);
    currFunctionPtr->outCasts_.push_back(outcast_2);

    RemoveUnalignedReshape removeUnalignedReshapeOpTest;
    int curSize = currFunctionPtr->Operations().size();
    EXPECT_EQ(removeUnalignedReshapeOpTest.RunOnFunction(*currFunctionPtr), SUCCESS);
    int expSize = curSize + 2;
    EXPECT_EQ(currFunctionPtr->Operations().size(), expSize);
}

inline void CheckTestMutiProducerBetweenCopyOutReshape(std::shared_ptr<Function> currFunctionPtr) {
    RemoveUnalignedReshape removeUnalignedReshapeOpTest;
    int curSize = currFunctionPtr->Operations().size();
    EXPECT_EQ(removeUnalignedReshapeOpTest.RunOnFunction(*currFunctionPtr), SUCCESS);
    int expSize = curSize + 1;
    EXPECT_EQ(currFunctionPtr->Operations().size(), expSize);
}
// in1                    - ASSEMBLE \*
// in2 - COPYIN - COPYOUT - ASSEMBLE - RESHAPE - COPYIN - COPYOUT - out1
//                        - ASSEMBLE - out2

TEST_F(TestRemoveUnalignedReshapeOp, TestMutiProducerBetweenCopyOutReshape)
{
    auto currFunctionPtr =
        std::make_shared<Function>(Program::GetInstance(), "TestMutiProducerBetweenCopyOutReshape", "TestMutiProducerBetweenCopyOutReshape", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    // Prepare the graph
    std::vector<int64_t> inShape = {8, 8};
    std::vector<int64_t> reshapeShape = {4, 16};
    std::vector<int64_t> outShape = {8, 16};
    DataType dataTypeFp32 = DT_FP32;
    auto incast_1 = std::make_shared<LogicalTensor>(*currFunctionPtr, dataTypeFp32, inShape);
    incast_1->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    auto incast_2 = std::make_shared<LogicalTensor>(*currFunctionPtr, dataTypeFp32, inShape);
    incast_2->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    auto copyInTensor_1 = std::make_shared<LogicalTensor>(*currFunctionPtr, dataTypeFp32, inShape);
    copyInTensor_1->SetMemoryTypeBoth(MemoryType::MEM_UB, true);
    auto copyOutTensor_1 = std::make_shared<LogicalTensor>(*currFunctionPtr, dataTypeFp32, inShape);
    copyOutTensor_1->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    auto assembleTensor = std::make_shared<LogicalTensor>(*currFunctionPtr, dataTypeFp32, inShape);
    assembleTensor->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    std::vector<SymbolicScalar> validShape_1 = {SymbolicScalar("Input_0_Dim_0"), SymbolicScalar("Input_0_Dim_1")};
    assembleTensor->UpdateDynValidShape(validShape_1);
    auto reshapeTensor = std::make_shared<LogicalTensor>(*currFunctionPtr, dataTypeFp32, reshapeShape);
    reshapeTensor->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    std::vector<SymbolicScalar> validShape_2 = {SymbolicScalar("Input_1_Dim_0"), SymbolicScalar("Input_1_Dim_1")};
    reshapeTensor->UpdateDynValidShape(validShape_2);
    auto copyInTensor_2 = std::make_shared<LogicalTensor>(*currFunctionPtr, dataTypeFp32, reshapeShape);
    copyInTensor_2->SetMemoryTypeBoth(MemoryType::MEM_UB, true);
    auto copyInTensor_3 = std::make_shared<LogicalTensor>(*currFunctionPtr, dataTypeFp32, reshapeShape);
    copyInTensor_3->SetMemoryTypeBoth(MemoryType::MEM_UB, true);
    auto outcast_1 = std::make_shared<LogicalTensor>(*currFunctionPtr, dataTypeFp32, outShape);
    outcast_1->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    auto outcast_2 = std::make_shared<LogicalTensor>(*currFunctionPtr, dataTypeFp32, outShape);
    outcast_2->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);

    currFunctionPtr->AddOperation(Opcode::OP_COPY_IN, {incast_1}, {copyInTensor_1});
    auto& copyoutOp = currFunctionPtr->AddOperation(Opcode::OP_COPY_OUT, {copyInTensor_1}, {copyOutTensor_1});
    auto copyoutAttr = std::make_shared<CopyOpAttribute>(MEM_UB, OpImmediate::Specified({0, 0}),
        OpImmediate::Specified(inShape), OpImmediate::Specified(inShape), std::vector<npu::tile_fwk::OpImmediate>());
    copyoutOp.SetOpAttribute(copyoutAttr);
    currFunctionPtr->AddOperation(Opcode::OP_ASSEMBLE, {copyOutTensor_1}, {assembleTensor});
    currFunctionPtr->AddOperation(Opcode::OP_ASSEMBLE, {incast_2}, {assembleTensor});
    currFunctionPtr->AddOperation(Opcode::OP_RESHAPE, {assembleTensor}, {reshapeTensor});
    currFunctionPtr->AddOperation(Opcode::OP_COPY_IN, {reshapeTensor}, {copyInTensor_2});
    currFunctionPtr->AddOperation(Opcode::OP_COPY_OUT, {copyInTensor_2}, {outcast_1});
    currFunctionPtr->AddOperation(Opcode::OP_ASSEMBLE, {copyOutTensor_1}, {outcast_2});

    currFunctionPtr->inCasts_.push_back(incast_1);
    currFunctionPtr->inCasts_.push_back(incast_2);
    currFunctionPtr->outCasts_.push_back(outcast_1);
    currFunctionPtr->outCasts_.push_back(outcast_2);
    CheckTestMutiProducerBetweenCopyOutReshape(currFunctionPtr);

}
inline void CheckTestMutiConsumersBetweenCopyOutReshape(std::shared_ptr<Function> currFunctionPtr) {
    RemoveUnalignedReshape removeUnalignedReshapeOpTest;
    int curSize = currFunctionPtr->Operations().size();
    EXPECT_EQ(removeUnalignedReshapeOpTest.RunOnFunction(*currFunctionPtr), SUCCESS);
    int expSize = curSize + 2;
    EXPECT_EQ(currFunctionPtr->Operations().size(), expSize);
}

// in - COPYIN - COPYOUT - ASSEMBLE - RESHAPE - COPYIN - COPYOUT - out1
//                                  - ASSEMBLE - out3
//                       - ASSEMBLE - out2

TEST_F(TestRemoveUnalignedReshapeOp, TestMutiConsumersBetweenCopyOutReshape)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestMutiProducerBetweenCopyOutReshape", "TestMutiProducerBetweenCopyOutReshape", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    // Prepare the graph
    std::vector<int64_t> inShape = {8, 8};
    std::vector<int64_t> reshapeShape = {4, 16};
    std::vector<int64_t> outShape = {8, 16};
    DataType dataTypeFP32 = DT_FP32;
    auto incast = std::make_shared<LogicalTensor>(*currFunctionPtr, dataTypeFP32, inShape);
    incast->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    auto copyInTensor_1 = std::make_shared<LogicalTensor>(*currFunctionPtr, dataTypeFP32, inShape);
    copyInTensor_1->SetMemoryTypeBoth(MemoryType::MEM_UB, true);
    auto copyOutTensor_1 = std::make_shared<LogicalTensor>(*currFunctionPtr, dataTypeFP32, inShape);
    copyOutTensor_1->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    auto assembleTensor = std::make_shared<LogicalTensor>(*currFunctionPtr, dataTypeFP32, inShape);
    assembleTensor->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    std::vector<SymbolicScalar> validShape_1 = {SymbolicScalar("Input_0_Dim_0"), SymbolicScalar("Input_0_Dim_1")};
    assembleTensor->UpdateDynValidShape(validShape_1);
    auto reshapeTensor = std::make_shared<LogicalTensor>(*currFunctionPtr, dataTypeFP32, reshapeShape);
    reshapeTensor->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    std::vector<SymbolicScalar> validShape_2 = {SymbolicScalar("Input_1_Dim_0"), SymbolicScalar("Input_1_Dim_1")};
    reshapeTensor->UpdateDynValidShape(validShape_2);
    auto copyInTensor_2 = std::make_shared<LogicalTensor>(*currFunctionPtr, dataTypeFP32, reshapeShape);
    copyInTensor_2->SetMemoryTypeBoth(MemoryType::MEM_UB, true);
    auto copyInTensor_3 = std::make_shared<LogicalTensor>(*currFunctionPtr, dataTypeFP32, reshapeShape);
    copyInTensor_3->SetMemoryTypeBoth(MemoryType::MEM_UB, true);
    auto outcast_1 = std::make_shared<LogicalTensor>(*currFunctionPtr, dataTypeFP32, outShape);
    outcast_1->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    auto outcast_2 = std::make_shared<LogicalTensor>(*currFunctionPtr, dataTypeFP32, outShape);
    outcast_2->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    auto outcast_3 = std::make_shared<LogicalTensor>(*currFunctionPtr, dataTypeFP32, inShape);
    outcast_3->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);

    currFunctionPtr->AddOperation(Opcode::OP_COPY_IN, {incast}, {copyInTensor_1});
    auto& copyout_Op = currFunctionPtr->AddOperation(Opcode::OP_COPY_OUT, {copyInTensor_1}, {copyOutTensor_1});
    auto copyoutAttr = std::make_shared<CopyOpAttribute>(MEM_UB, OpImmediate::Specified({0, 0}),
        OpImmediate::Specified(inShape), OpImmediate::Specified(inShape), std::vector<npu::tile_fwk::OpImmediate>());
    copyout_Op.SetOpAttribute(copyoutAttr);
    currFunctionPtr->AddOperation(Opcode::OP_ASSEMBLE, {copyOutTensor_1}, {assembleTensor});
    currFunctionPtr->AddOperation(Opcode::OP_RESHAPE, {assembleTensor}, {reshapeTensor});
    currFunctionPtr->AddOperation(Opcode::OP_COPY_IN, {reshapeTensor}, {copyInTensor_2});
    currFunctionPtr->AddOperation(Opcode::OP_COPY_OUT, {copyInTensor_2}, {outcast_1});
    currFunctionPtr->AddOperation(Opcode::OP_ASSEMBLE, {copyOutTensor_1}, {outcast_2});
    currFunctionPtr->AddOperation(Opcode::OP_ASSEMBLE, {assembleTensor}, {outcast_3});

    currFunctionPtr->inCasts_.push_back(incast);
    currFunctionPtr->outCasts_.push_back(outcast_1);
    currFunctionPtr->outCasts_.push_back(outcast_2);
    currFunctionPtr->outCasts_.push_back(outcast_3);
    CheckTestMutiConsumersBetweenCopyOutReshape(currFunctionPtr);
}

// in - COPYIN - COPYOUT - RESHAPE - COPYIN - COPYOUT - out
//                                 - COPYIN - COPYOUT - out

// 搬运超过了UB限额，不进行修改
TEST_F(TestRemoveUnalignedReshapeOp, TestCopyToReshapeCopyOnL1OverUB)
{
    auto currFunctionPtr =
        std::make_shared<Function>(Program::GetInstance(), "TestCopyToReshapeCopyOnL1OverUB", "TestCopyToReshapeCopyOnL1OverUB", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    // Prepare the graph
    std::vector<int64_t> inShapeOverUB = {2024, 32};
    std::vector<int64_t> reshapeShapeOverUB = {1024, 64};
    std::vector<int64_t> outShapeOverUB = {2024, 64};
    auto incast = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, inShapeOverUB);
    auto copyinTensor1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, inShapeOverUB);
    copyinTensor1->SetMemoryTypeBoth(MemoryType::MEM_L1, true);
    auto copyoutTensor1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, inShapeOverUB);
    copyoutTensor1->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    std::vector<SymbolicScalar> validShape1 = {SymbolicScalar("Input_0_Dim_0"), SymbolicScalar("Input_0_Dim_1")};
    copyoutTensor1->UpdateDynValidShape(validShape1);
    auto reshapeTensor = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, reshapeShapeOverUB);
    reshapeTensor->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    std::vector<SymbolicScalar> validShape2 = {SymbolicScalar("Input_1_Dim_0"), SymbolicScalar("Input_1_Dim_1")};
    reshapeTensor->UpdateDynValidShape(validShape2);
    auto copyinTensor2 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, reshapeShapeOverUB);
    copyinTensor2->SetMemoryTypeBoth(MemoryType::MEM_L1, true);
    auto copyinTensor3 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, reshapeShapeOverUB);
    copyinTensor3->SetMemoryTypeBoth(MemoryType::MEM_L1, true);
    auto outcast_1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, outShapeOverUB);
    outcast_1->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    auto outcast_2 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, outShapeOverUB);
    outcast_2->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);

    currFunctionPtr->AddOperation(Opcode::OP_COPY_IN, {incast}, {copyinTensor1});
    currFunctionPtr->AddOperation(Opcode::OP_COPY_OUT, {copyinTensor1}, {copyoutTensor1});
    currFunctionPtr->AddOperation(Opcode::OP_RESHAPE, {copyoutTensor1}, {reshapeTensor});
    currFunctionPtr->AddOperation(Opcode::OP_COPY_IN, {reshapeTensor}, {copyinTensor2});
    currFunctionPtr->AddOperation(Opcode::OP_COPY_IN, {reshapeTensor}, {copyinTensor3});
    currFunctionPtr->AddOperation(Opcode::OP_COPY_OUT, {copyinTensor3}, {outcast_1});
    currFunctionPtr->AddOperation(Opcode::OP_COPY_OUT, {copyinTensor2}, {outcast_2});

    currFunctionPtr->inCasts_.push_back(incast);
    currFunctionPtr->outCasts_.push_back(outcast_1);
    currFunctionPtr->outCasts_.push_back(outcast_2);

    RemoveUnalignedReshape removeUnalignedReshapeOpTest;
    int curSize = currFunctionPtr->Operations().size();
    EXPECT_EQ(removeUnalignedReshapeOpTest.RunOnFunction(*currFunctionPtr), SUCCESS);
    EXPECT_EQ(currFunctionPtr->Operations().size(), curSize);
}

TEST_F(TestRemoveUnalignedReshapeOp, TestValidShapeInfer)
{
    auto currFunctionPtr =
        std::make_shared<Function>(Program::GetInstance(), "TestValidShapeInfer", "TestValidShapeInfer", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    auto inCast = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, std::vector<int64_t>{128, 64});
    inCast->SetMemoryTypeOriginal(MemoryType::MEM_UB, false);
    inCast->UpdateDynValidShape({SymbolicScalar(128), SymbolicScalar(64)});

    auto reshapeInput = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, std::vector<int64_t>{128, 64});
    reshapeInput->SetMemoryTypeOriginal(MemoryType::MEM_UB, false);
    reshapeInput->UpdateDynValidShape({SymbolicScalar(128), SymbolicScalar(64)});

    auto reshapeOutput = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, std::vector<int64_t>{64, 128});
    reshapeOutput->SetMemoryTypeOriginal(MemoryType::MEM_UB, false);
    reshapeOutput->UpdateDynValidShape({SymbolicScalar(64), SymbolicScalar(128)});

    auto outCast = std::make_shared<LogicalTensor>(
        *currFunctionPtr, DT_FP32, std::vector<int64_t>{64, 128}, TileOpFormat::TILEOP_ND, "outCast",
        NodeType::OUTCAST);
    outCast->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR, false);

    auto& reshapeOp = currFunctionPtr->AddOperation(Opcode::OP_RESHAPE, {reshapeInput}, {reshapeOutput});
    reshapeOp.UpdateSubgraphID(0);

    auto& copyOutOp = currFunctionPtr->AddOperation(Opcode::OP_COPY_OUT, {reshapeOutput}, {outCast});
    copyOutOp.UpdateSubgraphID(0);

    currFunctionPtr->inCasts_.push_back(inCast);
    currFunctionPtr->outCasts_.push_back(outCast);

    RemoveUnalignedReshape removeUnalignedReshapeOpTest;
    EXPECT_EQ(removeUnalignedReshapeOpTest.RunOnFunction(*currFunctionPtr), SUCCESS);

    for (const auto& op : currFunctionPtr->Operations()) {
        if (op.GetOpcode() == Opcode::OP_RESHAPE_COPY_OUT || op.GetOpcode() == Opcode::OP_RESHAPE_COPY_IN) {
            auto oOperand = op.GetOOperands().front();
            auto dynValidShape = oOperand->GetDynValidShape();
            EXPECT_FALSE(dynValidShape.empty());
        }
    }
}
