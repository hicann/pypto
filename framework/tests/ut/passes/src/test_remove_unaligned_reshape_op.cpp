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

class TestRemoveUnalignedReshapeOp : public ::testing::Test {
public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}

    void SetUp() override {
        Program::GetInstance().Reset();
        config::Reset();
        config::SetPlatformConfig(KEY_ONLY_HOST_COMPILE, true);
        config::SetPlatformConfig(KEY_ENABLE_COST_MODEL, false);
    }
    void TearDown() override {
    }
};

inline void ConstructGraph1(std::shared_ptr<Function> &currFunctionPtr) {
    // Prepare the graph
    std::vector<int64_t> shape = {7, 15};
    std::vector<int64_t> reshape_shape = {15,7};
    std::vector<int64_t> expect_shape = {7, 16};
    auto shapeImme = OpImmediate::Specified(shape);
    auto incast1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    auto ubTensor1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    ubTensor1->SetMemoryTypeBoth(MEM_UB);
    auto ubTensor2 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, reshape_shape);
    ubTensor2->SetMemoryTypeBoth(MEM_UB);
    auto outCast = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, reshape_shape);
    outCast->UpdateDynValidShape({SymbolicScalar("output_0_Dim_0"), SymbolicScalar("output_0_Dim_1")});
    auto &copy_op1 = currFunctionPtr->AddRawOperation(Opcode::OP_COPY_IN, {incast1}, {ubTensor1});
    auto copyin1Attr = std::make_shared<CopyOpAttribute>(OpImmediate::Specified({0, 0}), MEM_UB, shapeImme, shapeImme, std::vector<npu::tile_fwk::OpImmediate>());
    std::vector<npu::tile_fwk::OpImmediate> toValidShape = {OpImmediate(SymbolicScalar("Input_0_Dim_0")), OpImmediate(SymbolicScalar("Input_0_Dim_1"))};
    copyin1Attr->SetToDynValidShape(toValidShape);
    copy_op1.SetOpAttribute(copyin1Attr);
    auto& reshape_op = currFunctionPtr->AddRawOperation(Opcode::OP_RESHAPE, {ubTensor1}, {ubTensor2});
    (void) reshape_op;
    auto& copy_out_op = currFunctionPtr->AddRawOperation(Opcode::OP_COPY_OUT, {ubTensor2}, {outCast});
    (void) copy_out_op;
    currFunctionPtr->inCasts_.push_back(incast1);
    currFunctionPtr->outCasts_.push_back(outCast);
}

inline void ConstructGraph2(std::shared_ptr<Function> &currFunctionPtr) {
    // Prepare the graph
    std::vector<int64_t> shape = {8, 16};
    std::vector<int64_t> reshape_shape = {16,8};
    std::vector<int64_t> expect_shape = {8, 16};
    auto shapeImme = OpImmediate::Specified(shape);
    auto incast1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    auto ubTensor1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    ubTensor1->SetMemoryTypeBoth(MEM_UB);
    auto ubTensor2 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, reshape_shape);
    ubTensor2->SetMemoryTypeBoth(MEM_UB);
    auto outCast = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, reshape_shape);
    outCast->UpdateDynValidShape({SymbolicScalar("output_0_Dim_0"), SymbolicScalar("output_0_Dim_1")});
    auto &copy_op1 = currFunctionPtr->AddRawOperation(Opcode::OP_COPY_IN, {incast1}, {ubTensor1});
    auto copyin1Attr = std::make_shared<CopyOpAttribute>(OpImmediate::Specified({0, 0}), MEM_UB, shapeImme, shapeImme, std::vector<npu::tile_fwk::OpImmediate>());
    std::vector<npu::tile_fwk::OpImmediate> toValidShape = {OpImmediate(SymbolicScalar("Input_0_Dim_0")), OpImmediate(SymbolicScalar("Input_0_Dim_1"))};
    copyin1Attr->SetToDynValidShape(toValidShape);
    copy_op1.SetOpAttribute(copyin1Attr);
    auto& reshape_op = currFunctionPtr->AddRawOperation(Opcode::OP_RESHAPE, {ubTensor1}, {ubTensor2});
    (void) reshape_op;
    auto& copy_out_op = currFunctionPtr->AddRawOperation(Opcode::OP_COPY_OUT, {ubTensor2}, {outCast});
    (void) copy_out_op;
    currFunctionPtr->inCasts_.push_back(incast1);
    currFunctionPtr->outCasts_.push_back(outCast);
}

inline void ConstructGraph3(std::shared_ptr<Function> &currFunctionPtr) {
    // Prepare the graph
    std::vector<int64_t> shape = {8, 16};
    std::vector<int64_t> reshape_shape = {16,8};
    std::vector<int64_t> expect_shape = {8, 16};
    auto shapeImme = OpImmediate::Specified(shape);
    auto incast1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    incast1->SetMemoryTypeBoth(MEM_DEVICE_DDR);
    auto ubTensor1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    ubTensor1->SetMemoryTypeBoth(MEM_UB);
    auto outCast = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, reshape_shape);
    outCast->UpdateDynValidShape({SymbolicScalar("output_0_Dim_0"), SymbolicScalar("output_0_Dim_1")});
    auto& reshape_op = currFunctionPtr->AddRawOperation(Opcode::OP_RESHAPE, {incast1}, {ubTensor1});
    (void) reshape_op;
    auto& copy_out_op = currFunctionPtr->AddRawOperation(Opcode::OP_COPY_OUT, {ubTensor1}, {outCast});
    (void) copy_out_op;
}

inline void ConstructGraph4(std::shared_ptr<Function> &currFunctionPtr) {
    // Prepare the graph
    std::vector<int64_t> shape = {64, 1};
    std::vector<int64_t> reshape_shape = {1,64};
    auto shapeImme = OpImmediate::Specified(shape);
    auto incast1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    incast1->SetMemoryTypeBoth(MEM_DEVICE_DDR);
    auto ubTensor1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    ubTensor1->SetMemoryTypeBoth(MEM_UB);
    auto outCast = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, reshape_shape);
    outCast->UpdateDynValidShape({SymbolicScalar("output_0_Dim_0"), SymbolicScalar("output_0_Dim_1")});
    auto& reshape_op = currFunctionPtr->AddRawOperation(Opcode::OP_RESHAPE, {incast1}, {ubTensor1});
    (void) reshape_op;
    auto& copy_out_op = currFunctionPtr->AddRawOperation(Opcode::OP_COPY_OUT, {ubTensor1}, {outCast});
    (void) copy_out_op;
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
TEST_F(TestRemoveUnalignedReshapeOp, reshaped_padded_ub) {
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestPadLocalBuffer", "TestPadLocalBuffer", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);
    std::vector<int64_t> shape = {7, 15};
    std::vector<int64_t> reshape_shape = {15,7};
    std::vector<int64_t> expect_shape = {7, 16};
    ConstructGraph1(currFunctionPtr);
    PadLocalBuffer padLocalBufferTest;
    padLocalBufferTest.RunOnFunction(*currFunctionPtr);
    RemoveUnalignedReshape removeUnalignedReshapeOpTest;
    removeUnalignedReshapeOpTest.RunOnFunction(*currFunctionPtr);
    for (auto &op : currFunctionPtr->Operations()) {
        if (op.GetOpcode() == Opcode::OP_RESHAPE) {
            for (auto &in : op.iOperand) {
                EXPECT_EQ(in->GetProducers().size(), 1);
                auto producer = *(in->GetProducers().begin());
                EXPECT_EQ(producer->GetOpcode(), Opcode::OP_COPY_OUT);
                if (in->oriShape == shape) {
                    EXPECT_EQ(in->shape, shape);
                    EXPECT_EQ(in->tensor->rawshape, shape);
                    EXPECT_EQ(in->tensor->oriRawshape, shape);
                }
            }
            for (auto &out : op.oOperand) {
                EXPECT_EQ(out->GetConsumers().size(), 1);
                auto consumer = *(out->GetConsumers().begin());
                EXPECT_EQ(consumer->GetOpcode(), Opcode::OP_COPY_IN);
                if (out->oriShape == reshape_shape) {
                    EXPECT_EQ(out->shape, reshape_shape);
                    EXPECT_EQ(out->tensor->rawshape, reshape_shape);
                    EXPECT_EQ(out->tensor->oriRawshape, reshape_shape);
                }
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
TEST_F(TestRemoveUnalignedReshapeOp, reshaped_unpadded_ub) {
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestPadLocalBuffer", "TestPadLocalBuffer", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);
    std::vector<int64_t> shape = {8, 16};
    std::vector<int64_t> reshape_shape = {16,8};
    std::vector<int64_t> expect_shape = {8, 16};
    ConstructGraph2(currFunctionPtr);
    PadLocalBuffer padLocalBufferTest;
    padLocalBufferTest.RunOnFunction(*currFunctionPtr);
    RemoveUnalignedReshape removeUnalignedReshapeOpTest;
    removeUnalignedReshapeOpTest.RunOnFunction(*currFunctionPtr);
    for (auto &op : currFunctionPtr->Operations()) {
        if (op.GetOpcode() == Opcode::OP_RESHAPE) {
            for (auto &in : op.iOperand) {
                EXPECT_EQ(in->GetProducers().size(), 1);
                auto producer = *(in->GetProducers().begin());
                EXPECT_NE(producer->GetOpcode(), Opcode::OP_COPY_OUT);
                if (in->oriShape == shape) {
                    EXPECT_EQ(in->shape, shape);
                    EXPECT_EQ(in->tensor->rawshape, shape);
                    EXPECT_EQ(in->tensor->oriRawshape, shape);
                }
            }
            for (auto &out : op.oOperand) {
                EXPECT_EQ(out->GetConsumers().size(), 1);
                auto consumer = *(out->GetConsumers().begin());
                EXPECT_NE(consumer->GetOpcode(), Opcode::OP_COPY_IN);
                if (out->oriShape == reshape_shape) {
                    EXPECT_EQ(out->shape, reshape_shape);
                    EXPECT_EQ(out->tensor->rawshape, reshape_shape);
                    EXPECT_EQ(out->tensor->oriRawshape, reshape_shape);
                }
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
TEST_F(TestRemoveUnalignedReshapeOp, reshaped_unpadded_ub_gm) {
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestPadLocalBuffer", "TestPadLocalBuffer", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);
    std::vector<int64_t> shape = {8, 16};
    std::vector<int64_t> reshape_shape = {16,8};
    std::vector<int64_t> expect_shape = {8, 16};
    ConstructGraph3(currFunctionPtr);
    PadLocalBuffer padLocalBufferTest;
    padLocalBufferTest.RunOnFunction(*currFunctionPtr);
    RemoveUnalignedReshape removeUnalignedReshapeOpTest;
    removeUnalignedReshapeOpTest.RunOnFunction(*currFunctionPtr);
    for (auto &op : currFunctionPtr->Operations()) {
        if (op.GetOpcode() == Opcode::OP_RESHAPE) {
            for (auto &out : op.oOperand) {
                EXPECT_EQ(out->GetConsumers().size(), 1);
                auto consumer = *(out->GetConsumers().begin());
                EXPECT_NE(consumer->GetOpcode(), Opcode::OP_COPY_IN);
                if (out->oriShape == reshape_shape) {
                    EXPECT_EQ(out->shape, reshape_shape);
                    EXPECT_EQ(out->tensor->rawshape, reshape_shape);
                    EXPECT_EQ(out->tensor->oriRawshape, reshape_shape);
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
TEST_F(TestRemoveUnalignedReshapeOp, reshaped_unpadded_ub_gm_last_dim_1) {
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestPadLocalBuffer", "TestPadLocalBuffer", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);
    std::vector<int64_t> shape = {64, 1};
    std::vector<int64_t> reshape_shape = {1,64};
    ConstructGraph4(currFunctionPtr);
    PadLocalBuffer padLocalBufferTest;
    padLocalBufferTest.RunOnFunction(*currFunctionPtr);
    RemoveUnalignedReshape removeUnalignedReshapeOpTest;
    removeUnalignedReshapeOpTest.RunOnFunction(*currFunctionPtr);
    for (auto &op : currFunctionPtr->Operations()) {
        if (op.GetOpcode() == Opcode::OP_RESHAPE) {
            for (auto &in : op.iOperand) {
                EXPECT_EQ(in->GetProducers().size(), 1);
                auto producer = *(in->GetProducers().begin());
                EXPECT_EQ(producer->GetOpcode(), Opcode::OP_COPY_OUT);
                if (in->oriShape == shape) {
                    EXPECT_EQ(in->shape, shape);
                    EXPECT_EQ(in->tensor->rawshape, shape);
                    EXPECT_EQ(in->tensor->oriRawshape, shape);
                }
            }
            for (auto &out : op.oOperand) {
                EXPECT_EQ(out->GetConsumers().size(), 1);
                auto consumer = *(out->GetConsumers().begin());
                EXPECT_EQ(consumer->GetOpcode(), Opcode::OP_COPY_IN);
                if (out->oriShape == reshape_shape) {
                    EXPECT_EQ(out->shape, reshape_shape);
                    EXPECT_EQ(out->tensor->rawshape, reshape_shape);
                    EXPECT_EQ(out->tensor->oriRawshape, reshape_shape);
                }
            }
        }
    }
}