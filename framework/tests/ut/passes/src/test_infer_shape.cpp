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
 * \file test_infer_shape.cpp
 * \brief
 */

#include "gtest/gtest.h"

#include "interface/tensor/logical_tensor.h"
#include "tilefwk/tilefwk.h"
#include "interface/tensor/irbuilder.h"
#include "passes/pass_utils/pass_operation_utils.h"
#include "symbolic_scalar_test_utils.h"
#include "interface/inner/tilefwk.h"
#include "interface/operation/op_infer_shape_impl.h"
#include "passes/tile_graph_pass/graph_constraint/infer_dyn_shape.h"
#include "interface/operation/attribute.h"
#include "interface/tensor/irbuilder.h"

namespace npu {
namespace tile_fwk {
class InferShapeTest : public testing::Test {
public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}

    void SetUp() override
    {
        Program::GetInstance().Reset();
        config::Reset();
    }

    void TearDown() override {}
};

TEST_F(InferShapeTest, TestAdd)
{
    auto currFunctionPtr =
        std::make_shared<Function>(Program::GetInstance(), "TestAddInferShape", "TestAddInferShape", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    // Prepare the graph
    std::vector<int64_t> shape = {8, 16};
    auto shapeImme = OpImmediate::Specified(shape);
    auto incast1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto incast2 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto ubTensor1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto ubTensor2 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto ubTensor3 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto outCast = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    outCast->UpdateDynValidShape({CreateTestScalarVar("output_0_Dim_0"), CreateTestScalarVar("output_0_Dim_1")});

    auto& copy_op1 = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_COPY_IN, {incast1}, {ubTensor1});
    auto copyin1Attr = std::make_shared<CopyOpAttribute>(
        OpImmediate::Specified({0, 0}), MEM_UB, shapeImme, shapeImme, std::vector<npu::tile_fwk::OpImmediate>());
    std::vector<npu::tile_fwk::OpImmediate> toValidShape = {
        OpImmediate(CreateTestScalarVar("Input_0_Dim_0")), OpImmediate(CreateTestScalarVar("Input_0_Dim_1"))};
    copyin1Attr->SetToDynValidShape(toValidShape);
    copy_op1.SetOpAttribute(copyin1Attr);

    auto& copy_op2 = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_COPY_IN, {incast2}, {ubTensor2});
    auto copyin2Attr = std::make_shared<CopyOpAttribute>(
        OpImmediate::Specified({0, 0}), MEM_UB, shapeImme, shapeImme, std::vector<npu::tile_fwk::OpImmediate>());
    std::vector<npu::tile_fwk::OpImmediate> toValidShape1 = {
        OpImmediate(CreateTestScalarVar("Input_1_Dim_0")), OpImmediate(CreateTestScalarVar("Input_1_Dim_1"))};
    copyin2Attr->SetToDynValidShape(toValidShape1);
    copy_op2.SetOpAttribute(copyin2Attr);

    auto& add_op = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_ADD, {ubTensor1, ubTensor2}, {ubTensor3});
    (void)add_op;
    auto& copy_out_op = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_COPY_OUT, {ubTensor3}, {outCast});
    (void)copy_out_op;

    currFunctionPtr->inCasts_.push_back(incast1);
    currFunctionPtr->inCasts_.push_back(incast2);
    currFunctionPtr->outCasts_.push_back(outCast);

    InferDynShape inferShapeTest;
    inferShapeTest.RunOnFunction(*currFunctionPtr);
    std::cout << currFunctionPtr->Dump() << std::endl;
    EXPECT_EQ(inferShapeTest.PostCheck(*currFunctionPtr), SUCCESS);
}

TEST_F(InferShapeTest, TestAddAlignCase)
{
    auto currFunctionPtr =
        std::make_shared<Function>(Program::GetInstance(), "TestAddInferShape", "TestAddInferShape", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    // Prepare the graph
    std::vector<int64_t> shape = {8, 16};
    auto shapeImme = OpImmediate::Specified(shape);
    auto incast1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto incast2 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto ubTensor1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto ubTensor2 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto ubTensor3 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto outCast = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));

    auto& copy_op1 = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_COPY_IN, {incast1}, {ubTensor1});
    auto copyin1Attr = std::make_shared<CopyOpAttribute>(
        OpImmediate::Specified({0, 0}), MEM_UB, shapeImme, shapeImme, std::vector<npu::tile_fwk::OpImmediate>());
    copy_op1.SetOpAttribute(copyin1Attr);

    auto& copy_op2 = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_COPY_IN, {incast2}, {ubTensor2});
    auto copyin2Attr = std::make_shared<CopyOpAttribute>(
        OpImmediate::Specified({0, 0}), MEM_UB, shapeImme, shapeImme, std::vector<npu::tile_fwk::OpImmediate>());
    copy_op2.SetOpAttribute(copyin2Attr);

    auto& add_op = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_ADD, {ubTensor1, ubTensor2}, {ubTensor3});
    (void)add_op;
    auto& copy_out_op = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_COPY_OUT, {ubTensor3}, {outCast});
    auto copyoutAttr = std::make_shared<CopyOpAttribute>(
        MEM_UB, OpImmediate::Specified({0, 0}), shapeImme, shapeImme, std::vector<npu::tile_fwk::OpImmediate>());
    copy_out_op.SetOpAttribute(copyoutAttr);

    currFunctionPtr->inCasts_.push_back(incast1);
    currFunctionPtr->inCasts_.push_back(incast2);
    currFunctionPtr->outCasts_.push_back(outCast);

    InferDynShape inferShapeTest;
    inferShapeTest.RunOnFunction(*currFunctionPtr);
    std::cout << currFunctionPtr->Dump() << std::endl;
    EXPECT_EQ(inferShapeTest.PostCheck(*currFunctionPtr), SUCCESS);
}

TEST_F(InferShapeTest, TestAddExp)
{
    auto currFunctionPtr =
        std::make_shared<Function>(Program::GetInstance(), "TestAddInferShape", "TestAddInferShape", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    // Prepare the graph
    std::vector<int64_t> shape = {8, 16};
    auto shapeImme = OpImmediate::Specified(shape);
    auto incast1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto incast2 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto ubTensor1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto ubTensor2 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto ubTensor3 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto outCast = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    outCast->UpdateDynValidShape({CreateTestScalarVar("output_0_Dim_0"), CreateTestScalarVar("output_0_Dim_1")});

    auto& copy_op1 = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_COPY_IN, {incast1}, {ubTensor1});
    auto copyin1Attr = std::make_shared<CopyOpAttribute>(
        OpImmediate::Specified({0, 0}), MEM_UB, shapeImme, shapeImme, std::vector<npu::tile_fwk::OpImmediate>());
    std::vector<npu::tile_fwk::OpImmediate> toValidShape = {
        OpImmediate(CreateTestScalarVar("Input_0_Dim_0")), OpImmediate(CreateTestScalarVar("Input_0_Dim_1"))};
    copyin1Attr->SetToDynValidShape(toValidShape);
    copy_op1.SetOpAttribute(copyin1Attr);

    auto& copy_op2 = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_COPY_IN, {incast2}, {ubTensor2});
    auto copyin2Attr = std::make_shared<CopyOpAttribute>(OpImmediate::Specified({0, 0}), MEM_UB, shapeImme, shapeImme);
    std::vector<npu::tile_fwk::OpImmediate> toValidShape1 = {
        OpImmediate(CreateTestScalarVar("Input_1_Dim_0")), OpImmediate(CreateTestScalarVar("Input_1_Dim_1"))};
    copyin2Attr->SetToDynValidShape(toValidShape1);
    copy_op2.SetOpAttribute(copyin2Attr);

    auto& add_op = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_ADD, {ubTensor1, ubTensor2}, {ubTensor3});
    (void)add_op;
    auto tmpCast = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto& copy_out_op = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_COPY_OUT, {ubTensor3}, {tmpCast});
    auto copyout1Attr = std::make_shared<CopyOpAttribute>(
        MEM_UB, OpImmediate::Specified({0, 0}), shapeImme, shapeImme, std::vector<npu::tile_fwk::OpImmediate>());
    copy_out_op.SetOpAttribute(copyout1Attr);

    auto ubTensor4 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto& copy_op3 = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_COPY_IN, {tmpCast}, {ubTensor4});
    auto copyin3Attr = std::make_shared<CopyOpAttribute>(
        OpImmediate::Specified({0, 0}), MEM_UB, shapeImme, shapeImme, std::vector<npu::tile_fwk::OpImmediate>());
    copy_op3.SetOpAttribute(copyin3Attr);

    auto ubTensor5 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto& exp = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_EXP, {ubTensor4}, {ubTensor5});
    (void)exp;

    auto& copy_out_op1 = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_COPY_OUT, {ubTensor5}, {outCast});
    (void)copy_out_op1;

    currFunctionPtr->inCasts_.push_back(incast1);
    currFunctionPtr->inCasts_.push_back(incast2);
    currFunctionPtr->outCasts_.push_back(outCast);

    InferDynShape inferShapeTest;
    inferShapeTest.RunOnFunction(*currFunctionPtr);
    std::cout << currFunctionPtr->Dump() << std::endl;
    EXPECT_EQ(inferShapeTest.PostCheck(*currFunctionPtr), SUCCESS);
}

TEST_F(InferShapeTest, TestReduce)
{
    auto currFunctionPtr =
        std::make_shared<Function>(Program::GetInstance(), "TestReduceInferShape", "TestReduceInferShape", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    // Prepare the graph
    std::vector<int64_t> inshape = {4, 8, 16};
    std::vector<int64_t> outshape = {4, 8, 8};
    auto shapeImme = OpImmediate::Specified(inshape);
    auto incast = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, inshape, CreateTestConstIntVector(inshape));
    auto outcast = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, outshape, CreateTestConstIntVector(outshape));
    auto inTensor = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, inshape, CreateTestConstIntVector(inshape));
    auto outTensor = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, outshape, CreateTestConstIntVector(outshape));

    auto& copyin_op = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_COPY_IN, {incast}, {inTensor});
    auto copyin_Attr = std::make_shared<CopyOpAttribute>(
        OpImmediate::Specified({0, 0}), MEM_UB, shapeImme, shapeImme, std::vector<OpImmediate>());
    std::vector<OpImmediate> toValidShape = {
        OpImmediate(CreateTestScalarVar("Input_0_Dim_0")), OpImmediate(CreateTestScalarVar("Input_0_Dim_1")),
        OpImmediate(CreateTestScalarVar("Input_0_Dim_2"))};
    copyin_Attr->SetToDynValidShape(toValidShape);
    copyin_op.SetOpAttribute(copyin_Attr);

    auto& reduce_op = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_ROWMAX_SINGLE, {inTensor}, {outTensor});
    auto axis = inshape.size() - 1;
    reduce_op.SetAttribute(OP_ATTR_PREFIX + "AXIS", static_cast<int>(axis));
    (void)reduce_op;

    auto& copyout_op = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_COPY_OUT, {outTensor}, {outcast});
    (void)copyout_op;

    currFunctionPtr->inCasts_.push_back(incast);
    currFunctionPtr->outCasts_.push_back(outcast);

    InferDynShape inferShapeTest;
    inferShapeTest.RunOnFunction(*currFunctionPtr);
    std::cout << currFunctionPtr->Dump() << std::endl;
    EXPECT_EQ(inferShapeTest.PostCheck(*currFunctionPtr), SUCCESS);
}

TEST_F(InferShapeTest, TestView)
{
    auto currFunctionPtr =
        std::make_shared<Function>(Program::GetInstance(), "TesViewInferShape", "TesViewInferShape", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    // Prepare the graph
    std::vector<int64_t> shape = {8, 16};
    auto incast = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto outcast = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));

    incast->UpdateDynValidShape({CreateTestScalarVar("input_0_Dim_0"), CreateTestScalarVar("input_0_Dim_1")});
    auto& view_op = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_VIEW, {incast}, {outcast});
    auto view_Attr = std::make_shared<ViewOpAttribute>(
        std::vector<int64_t>(), MEM_UNKNOWN, std::vector<SymbolicScalar>(), std::vector<SymbolicScalar>());
    view_Attr->SetFromOffset(
        std::vector<int64_t>(), {CreateTestScalarVar("Offset_0_Dim_0"), CreateTestScalarVar("Offset_0_Dim_1")});
    view_op.SetOpAttribute(view_Attr);

    currFunctionPtr->inCasts_.push_back(incast);
    currFunctionPtr->outCasts_.push_back(outcast);

    InferDynShape inferShapeTest;
    inferShapeTest.RunOnFunction(*currFunctionPtr);
    std::cout << currFunctionPtr->Dump() << std::endl;
    std::cout << view_op.GetOOperands()[0]->Dump() << std::endl;
    EXPECT_EQ(inferShapeTest.PostCheck(*currFunctionPtr), SUCCESS);
}

TEST_F(InferShapeTest, TestViewAlign)
{
    auto currFunctionPtr =
        std::make_shared<Function>(Program::GetInstance(), "TesViewInferShape", "TesViewInferShape", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    // Prepare the graph
    std::vector<int64_t> shape = {8, 16};
    std::vector<int64_t> offset = {2, 0};
    std::vector<int64_t> viewshape = {8, 4};
    auto incast = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto outcast = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, viewshape, CreateTestConstIntVector(viewshape));

    auto& view_op = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_VIEW, {incast}, {outcast});
    auto view_Attr = std::make_shared<ViewOpAttribute>(
        std::vector<int64_t>(), MEM_UNKNOWN, std::vector<SymbolicScalar>(), std::vector<SymbolicScalar>());
    view_Attr->SetFromOffset(offset, std::vector<SymbolicScalar>());
    view_op.SetOpAttribute(view_Attr);

    currFunctionPtr->inCasts_.push_back(incast);
    currFunctionPtr->outCasts_.push_back(outcast);

    InferDynShape inferShapeTest;
    inferShapeTest.RunOnFunction(*currFunctionPtr);
    std::cout << currFunctionPtr->Dump() << std::endl;
    std::cout << view_op.GetOOperands()[0]->Dump() << std::endl;
    EXPECT_EQ(inferShapeTest.PostCheck(*currFunctionPtr), SUCCESS);
}

TEST_F(InferShapeTest, TestAssemble)
{
    auto currFunctionPtr =
        std::make_shared<Function>(Program::GetInstance(), "TestAssembleInferShape", "TestAssembleInferShape", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    // Prepare the graph
    std::vector<int64_t> shape = {8, 16};
    auto incast = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto outcast = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));

    incast->UpdateDynValidShape({CreateTestScalarVar("input_0_Dim_0"), CreateTestScalarVar("input_0_Dim_1")});
    outcast->UpdateDynValidShape({CreateTestScalarVar("output_0_Dim_0"), CreateTestScalarVar("output_0_Dim_1")});
    auto& assemble_op = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_ASSEMBLE, {incast}, {outcast});
    auto assemble_Attr = std::make_shared<AssembleOpAttribute>(
        MEM_UNKNOWN, std::vector<int64_t>(), std::vector<SymbolicScalar>(), std::vector<SymbolicScalar>());

    auto dynOffset = {CreateTestScalarVar("DynOffset_0_Dim_0"), CreateTestScalarVar("DynOffset_0_Dim_1")};
    assemble_Attr->SetToOffset({2, 2}, dynOffset);
    assemble_op.SetOpAttribute(assemble_Attr);

    currFunctionPtr->inCasts_.push_back(incast);
    currFunctionPtr->outCasts_.push_back(outcast);

    InferDynShape inferShapeTest;
    inferShapeTest.RunOnFunction(*currFunctionPtr);
    std::cout << currFunctionPtr->Dump() << std::endl;
    std::cout << assemble_op.GetOOperands()[0]->Dump() << std::endl;
    EXPECT_EQ(inferShapeTest.PostCheck(*currFunctionPtr), SUCCESS);
}

TEST_F(InferShapeTest, TestFailCopyOut)
{
    auto currFunctionPtr =
        std::make_shared<Function>(Program::GetInstance(), "TestCopyOutInferShape", "TestCopyOutInferShape", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    // Prepare the graph
    std::vector<int64_t> inshape = {8, 16};
    std::vector<int64_t> outshape = {8, 8};
    auto incast = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, inshape, CreateTestConstIntVector(inshape));
    auto outcast = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, outshape, CreateTestConstIntVector(outshape));

    auto& copyout_op = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_COPY_OUT, {incast}, {outcast});
    (void)copyout_op;
    currFunctionPtr->inCasts_.push_back(incast);
    currFunctionPtr->outCasts_.push_back(outcast);

    InferDynShape inferShapeTest;
    inferShapeTest.RunOnFunction(*currFunctionPtr);
    std::cout << currFunctionPtr->Dump() << std::endl;
    EXPECT_EQ(inferShapeTest.PostCheck(*currFunctionPtr), FAILED);
}

TEST_F(InferShapeTest, TestCopyOut)
{
    auto currFunctionPtr =
        std::make_shared<Function>(Program::GetInstance(), "TestCopyOutInferShape", "TestCopyOutInferShape", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    // Prepare the graph
    std::vector<int64_t> inshape = {8, 16};
    std::vector<int64_t> outshape = {8, 16};
    auto toOffsetImme = OpImmediate::Specified({4, 4});
    auto inshapeImme = OpImmediate::Specified(inshape);
    auto incast = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, inshape, CreateTestConstIntVector(inshape));
    auto outcast = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, outshape, CreateTestConstIntVector(outshape));
    incast->UpdateDynValidShape({CreateTestScalarVar("input_0_Dim_0"), CreateTestScalarVar("input_0_Dim_1")});

    auto& copyout_op = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_COPY_OUT, {incast}, {outcast});
    auto copyout_Attr = std::make_shared<CopyOpAttribute>(
        MEM_DEVICE_DDR, toOffsetImme, inshapeImme, inshapeImme, std::vector<OpImmediate>());
    copyout_op.SetOpAttribute(copyout_Attr);

    currFunctionPtr->inCasts_.push_back(incast);
    currFunctionPtr->outCasts_.push_back(outcast);

    InferDynShape inferShapeTest;
    inferShapeTest.RunOnFunction(*currFunctionPtr);
    std::cout << currFunctionPtr->Dump() << std::endl;
    std::cout << copyout_op.GetOOperands()[0]->Dump() << std::endl;
    EXPECT_EQ(inferShapeTest.PostCheck(*currFunctionPtr), SUCCESS);
}

TEST_F(InferShapeTest, TestCopyIn)
{
    auto currFunctionPtr =
        std::make_shared<Function>(Program::GetInstance(), "TestCopyInInferShape", "TestCopyInInferShape", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    // Prepare the graph
    std::vector<int64_t> inshape = {8, 16};
    std::vector<int64_t> outshape = {8, 8};
    auto fromOffsetImme = OpImmediate::Specified({4, 4});
    auto inshapeImme = OpImmediate::Specified(inshape);
    auto incast = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, inshape, CreateTestConstIntVector(inshape));
    auto outcast = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, outshape, CreateTestConstIntVector(outshape));
    incast->UpdateDynValidShape({CreateTestScalarVar("input_0_Dim_0"), CreateTestScalarVar("input_0_Dim_1")});

    auto& copyin_op = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_COPY_IN, {incast}, {outcast});
    auto copyin_Attr = std::make_shared<CopyOpAttribute>(
        fromOffsetImme, MEM_UNKNOWN, inshapeImme, inshapeImme, std::vector<OpImmediate>());
    copyin_op.SetOpAttribute(copyin_Attr);

    currFunctionPtr->inCasts_.push_back(incast);
    currFunctionPtr->outCasts_.push_back(outcast);

    InferDynShape inferShapeTest;
    inferShapeTest.RunOnFunction(*currFunctionPtr);
    std::cout << currFunctionPtr->Dump() << std::endl;
    std::cout << copyin_op.GetOOperands()[0]->Dump() << std::endl;
    EXPECT_EQ(inferShapeTest.PostCheck(*currFunctionPtr), SUCCESS);
}

TEST_F(InferShapeTest, TestReshape)
{
    auto currFunctionPtr =
        std::make_shared<Function>(Program::GetInstance(), "TestReshapeInferShape", "TestReshapeInferShape", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    // Prepare the graph
    std::vector<int64_t> inshape = {8, 16};
    std::vector<int64_t> outshape = {4, 4};
    auto shapeImme = OpImmediate::Specified(inshape);
    auto incast = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, inshape, CreateTestConstIntVector(inshape));
    auto outcast = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, outshape, CreateTestConstIntVector(outshape));
    auto inTensor = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, inshape, CreateTestConstIntVector(inshape));
    auto outTensor = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, outshape, CreateTestConstIntVector(outshape));
    incast->UpdateDynValidShape({CreateTestScalarVar("input_0_Dim_0"), CreateTestScalarVar("input_0_Dim_1")});
    outcast->UpdateDynValidShape({CreateTestScalarVar("output_0_Dim_0"), CreateTestScalarVar("output_0_Dim_1")});

    auto& copyin_op = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_COPY_IN, {incast}, {inTensor});
    auto copyin_Attr = std::make_shared<CopyOpAttribute>(
        OpImmediate::Specified({0, 0}), MEM_UNKNOWN, shapeImme, shapeImme, std::vector<OpImmediate>());

    std::vector<OpImmediate> toValidShape = {
        OpImmediate(CreateTestScalarVar("Input_1_Dim_0")), OpImmediate(CreateTestScalarVar("Input_1_Dim_1"))};
    copyin_Attr->SetToDynValidShape(toValidShape);
    copyin_op.SetOpAttribute(copyin_Attr);

    auto& reshape_op = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_RESHAPE, {inTensor}, {outTensor});
    (void)reshape_op;

    auto& copyout_op = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_COPY_OUT, {outTensor}, {outcast});
    (void)copyout_op;

    currFunctionPtr->inCasts_.push_back(incast);
    currFunctionPtr->outCasts_.push_back(outcast);

    InferDynShape inferShapeTest;
    inferShapeTest.RunOnFunction(*currFunctionPtr);
    std::cout << currFunctionPtr->Dump() << std::endl;
    std::cout << reshape_op.GetOOperands()[0]->Dump() << std::endl;
    EXPECT_EQ(inferShapeTest.PostCheck(*currFunctionPtr), SUCCESS);
}

TEST_F(InferShapeTest, TestSHMEM_LOAD)
{
    auto currFunctionPtr =
        std::make_shared<Function>(Program::GetInstance(), "TestSHMEM_LOAD", "TestSHMEM_LOAD", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    std::vector<int64_t> inshape0 = {1, 1};
    std::vector<int64_t> inshape1 = {1, 1, 8, 16};
    std::vector<int64_t> outshape = {8, 16};
    auto shapeImme = OpImmediate::Specified(outshape);

    auto incast0 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, inshape0, CreateTestConstIntVector(inshape0));
    auto incast1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, inshape1, CreateTestConstIntVector(inshape1));
    auto shmemLoadOut = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, outshape, CreateTestConstIntVector(outshape));
    auto outcast = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, outshape, CreateTestConstIntVector(outshape));

    auto& shmemLoad_op = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_SHMEM_LOAD, {incast0, incast1}, {shmemLoadOut});
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_COPY_OUT, {shmemLoadOut}, {outcast});

    auto shmemLoad_Attr = std::make_shared<CopyOpAttribute>(
        OpImmediate::Specified({0, 0}), MEM_UB, shapeImme, shapeImme, std::vector<OpImmediate>());

    std::vector<OpImmediate> toValidShape = {
        OpImmediate(CreateTestScalarVar("Input_0_Dim_0")), OpImmediate(CreateTestScalarVar("Input_0_Dim_1"))};
    shmemLoad_Attr->SetToDynValidShape(toValidShape);
    shmemLoad_op.SetOpAttribute(shmemLoad_Attr);

    currFunctionPtr->inCasts_.push_back(incast0);
    currFunctionPtr->inCasts_.push_back(incast1);
    currFunctionPtr->outCasts_.push_back(outcast);

    InferDynShape inferShapeTest;
    inferShapeTest.RunOnFunction(*currFunctionPtr);
    std::cout << currFunctionPtr->Dump() << std::endl;
    EXPECT_NE(shmemLoadOut->GetDynValidShape().size(), 0);
    EXPECT_EQ(inferShapeTest.PostCheck(*currFunctionPtr), SUCCESS);
}

TEST_F(InferShapeTest, TestPad)
{
    auto currFunctionPtr =
        std::make_shared<Function>(Program::GetInstance(), "TestPadInferShape", "TestPadInferShape", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    // Prepare the graph
    std::vector<int64_t> inshape = {2, 2};
    std::vector<int64_t> outshape = {3, 4};
    auto shapeImme = OpImmediate::Specified(inshape);
    auto incast = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, inshape, CreateTestConstIntVector(inshape));
    auto outcast = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, outshape, CreateTestConstIntVector(outshape));
    auto inTensor = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, inshape, CreateTestConstIntVector(inshape));
    auto outTensor = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, outshape, CreateTestConstIntVector(outshape));

    auto& copyin_op = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_COPY_IN, {incast}, {inTensor});
    auto copyin_Attr = std::make_shared<CopyOpAttribute>(
        OpImmediate::Specified({0, 0}), MEM_UB, shapeImme, shapeImme, std::vector<OpImmediate>());
    std::vector<OpImmediate> toValidShape = {
        OpImmediate(CreateTestScalarVar("Input_0_Dim_0")), OpImmediate(CreateTestScalarVar("Input_0_Dim_1"))};
    copyin_Attr->SetToDynValidShape(toValidShape);
    copyin_op.SetOpAttribute(copyin_Attr);

    auto& pad_op = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_PAD, {inTensor}, {outTensor});
    pad_op.SetAttribute(OP_ATTR_PREFIX + "pad_right", 2);
    pad_op.SetAttribute(OP_ATTR_PREFIX + "pad_bottom", 1);
    pad_op.SetAttribute(OpAttributeKey::scalar, Element(DT_FP32, 0.0f));

    auto& copyout_op = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_COPY_OUT, {outTensor}, {outcast});
    (void)copyout_op;

    currFunctionPtr->inCasts_.push_back(incast);
    currFunctionPtr->outCasts_.push_back(outcast);

    InferDynShape inferShapeTest;
    inferShapeTest.RunOnFunction(*currFunctionPtr);
    std::cout << currFunctionPtr->Dump() << std::endl;
    EXPECT_EQ(inferShapeTest.PostCheck(*currFunctionPtr), SUCCESS);
}

TEST_F(InferShapeTest, TestFillPad)
{
    auto currFunctionPtr =
        std::make_shared<Function>(Program::GetInstance(), "TestFillPadInferShape", "TestFillPadInferShape", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    // Prepare the graph
    std::vector<int64_t> inshape = {3, 4};
    std::vector<int64_t> outshape = {3, 4};
    auto shapeImme = OpImmediate::Specified(inshape);
    auto incast = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, inshape, CreateTestConstIntVector(inshape));
    auto outcast = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, outshape, CreateTestConstIntVector(outshape));
    auto inTensor = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, inshape, CreateTestConstIntVector(inshape));
    auto outTensor = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, outshape, CreateTestConstIntVector(outshape));

    auto& copyin_op = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_COPY_IN, {incast}, {inTensor});
    auto copyin_Attr = std::make_shared<CopyOpAttribute>(
        OpImmediate::Specified({0, 0}), MEM_UB, shapeImme, shapeImme, std::vector<OpImmediate>());
    std::vector<OpImmediate> toValidShape = {
        OpImmediate(CreateTestScalarVar("Input_0_Dim_0")), OpImmediate(CreateTestScalarVar("Input_0_Dim_1"))};
    copyin_Attr->SetToDynValidShape(toValidShape);
    copyin_op.SetOpAttribute(copyin_Attr);

    auto& fillpad_op = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_FILLPAD, {inTensor}, {outTensor});
    fillpad_op.SetAttribute(OpAttributeKey::scalar, Element(DT_FP32, 0.0f));

    auto& copyout_op = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_COPY_OUT, {outTensor}, {outcast});
    (void)copyout_op;

    currFunctionPtr->inCasts_.push_back(incast);
    currFunctionPtr->outCasts_.push_back(outcast);

    InferDynShape inferShapeTest;
    inferShapeTest.RunOnFunction(*currFunctionPtr);
    std::cout << currFunctionPtr->Dump() << std::endl;
    EXPECT_EQ(inferShapeTest.PostCheck(*currFunctionPtr), SUCCESS);
}

TEST_F(InferShapeTest, TestIndexOutCast)
{
    auto currFunctionPtr =
        std::make_shared<Function>(Program::GetInstance(), "TestIndexOutCast", "TestIndexOutCast", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    std::vector<int64_t> inshape0 = {1, 1};
    std::vector<int64_t> inshape1 = {2, 2};
    std::vector<int64_t> inshape2 = {4, 4};
    std::vector<int64_t> outshape = {4, 4};

    auto incast0 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, inshape0, CreateTestConstIntVector(inshape0));
    incast0->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    auto view0 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, inshape0, CreateTestConstIntVector(inshape0));
    view0->SetMemoryTypeBoth(MemoryType::MEM_UB, true);
    auto incast1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, inshape1, CreateTestConstIntVector(inshape1));
    incast1->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    auto view1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, inshape1, CreateTestConstIntVector(inshape1));
    view1->SetMemoryTypeBoth(MemoryType::MEM_UB, true);
    auto incast2 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, inshape2, CreateTestConstIntVector(inshape2));
    std::vector<SymbolicScalar> validShape = {CreateTestScalarVar("Input_0_Dim_0"), CreateTestScalarVar("Input_0_Dim_1")};
    incast2->UpdateDynValidShape(validShape);
    incast2->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    auto outcast = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, outshape, CreateTestConstIntVector(outshape));
    outcast->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);

    auto& viewOp0 = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_VIEW, {incast0}, {view0});
    auto& viewOp1 = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_VIEW, {incast1}, {view1});
    auto& indexoutcastOp = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_INDEX_OUTCAST, {view0, view1, incast2}, {outcast});

    Offset offsets = {0, 0};
    auto viewOpAttribute0 = std::make_shared<ViewOpAttribute>(offsets);
    auto viewOpAttribute1 = std::make_shared<ViewOpAttribute>(offsets);
    viewOp0.SetOpAttribute(viewOpAttribute0);
    viewOp1.SetOpAttribute(viewOpAttribute1);
    auto indexoutcastOpAttr = std::make_shared<CopyOpAttribute>(
        MemoryType::MEM_DEVICE_DDR, OpImmediate::Specified(offsets), OpImmediate::Specified(inshape1),
        OpImmediate::Specified(incast2->tensor->GetDynRawShape()));
    indexoutcastOp.SetOpAttribute(indexoutcastOpAttr);

    currFunctionPtr->inCasts_.push_back(incast0);
    currFunctionPtr->inCasts_.push_back(incast1);
    currFunctionPtr->inCasts_.push_back(incast2);
    currFunctionPtr->outCasts_.push_back(outcast);

    InferDynShape inferShapeTest;
    inferShapeTest.RunOnFunction(*currFunctionPtr);
    std::cout << currFunctionPtr->Dump() << std::endl;
    EXPECT_NE(outcast->GetDynValidShape().size(), 0);
    EXPECT_EQ(inferShapeTest.PostCheck(*currFunctionPtr), SUCCESS);
    auto indexOutCastOpAttribute = std::dynamic_pointer_cast<CopyOpAttribute>(indexoutcastOp.GetOpAttribute());
    const auto& fromDynValidShape = indexOutCastOpAttribute->GetFromDynValidShape();
    EXPECT_NE(fromDynValidShape.size(), 0U);
}

TEST_F(InferShapeTest, TestPermute)
{
    auto currFunctionPtr =
        std::make_shared<Function>(Program::GetInstance(), "TestPermuteInferShape", "TestPermuteInferShape", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    std::vector<int64_t> inshape = {2, 3, 4};
    std::vector<int64_t> outshape = {3, 2, 4};
    auto shapeImme = OpImmediate::Specified(inshape);
    auto incast = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, inshape, CreateTestConstIntVector(inshape));
    auto outcast = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, outshape, CreateTestConstIntVector(outshape));
    auto inTensor = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, inshape, CreateTestConstIntVector(inshape));
    auto outTensor = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, outshape, CreateTestConstIntVector(outshape));

    auto& copyin_op = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_COPY_IN, {incast}, {inTensor});
    auto copyin_Attr = std::make_shared<CopyOpAttribute>(
        OpImmediate::Specified({0, 0, 0}), MEM_UB, shapeImme, shapeImme, std::vector<OpImmediate>());
    std::vector<OpImmediate> toValidShape = {
        OpImmediate(CreateTestScalarVar("Input_0_Dim_0")), OpImmediate(CreateTestScalarVar("Input_0_Dim_1")),
        OpImmediate(CreateTestScalarVar("Input_0_Dim_2"))};
    copyin_Attr->SetToDynValidShape(toValidShape);
    copyin_op.SetOpAttribute(copyin_Attr);

    auto& permute_op = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_PERMUTE, {inTensor}, {outTensor});
    permute_op.SetAttribute(OpAttributeKey::perm, std::vector<int>{1, 0, 2});

    auto& copyout_op = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_COPY_OUT, {outTensor}, {outcast});
    (void)copyout_op;

    currFunctionPtr->inCasts_.push_back(incast);
    currFunctionPtr->outCasts_.push_back(outcast);

    InferDynShape inferShapeTest;
    inferShapeTest.RunOnFunction(*currFunctionPtr);
    std::cout << currFunctionPtr->Dump() << std::endl;
    EXPECT_EQ(inferShapeTest.PostCheck(*currFunctionPtr), SUCCESS);
}

TEST_F(InferShapeTest, TestPermuteElement)
{
    auto currFunctionPtr = std::make_shared<Function>(
        Program::GetInstance(), "TestPermuteElemInferShape", "TestPermuteElemInferShape", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    std::vector<int64_t> inshape = {2, 3, 4};
    std::vector<int64_t> outshape = {2, 4, 3};
    auto shapeImme = OpImmediate::Specified(inshape);
    auto incast = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, inshape, CreateTestConstIntVector(inshape));
    auto outcast = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, outshape, CreateTestConstIntVector(outshape));
    auto inTensor = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, inshape, CreateTestConstIntVector(inshape));
    auto outTensor = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, outshape, CreateTestConstIntVector(outshape));

    auto& copyin_op = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_COPY_IN, {incast}, {inTensor});
    auto copyin_Attr = std::make_shared<CopyOpAttribute>(
        OpImmediate::Specified({0, 0, 0}), MEM_UB, shapeImme, shapeImme, std::vector<OpImmediate>());
    std::vector<OpImmediate> toValidShape = {
        OpImmediate(CreateTestScalarVar("Input_0_Dim_0")), OpImmediate(CreateTestScalarVar("Input_0_Dim_1")),
        OpImmediate(CreateTestScalarVar("Input_0_Dim_2"))};
    copyin_Attr->SetToDynValidShape(toValidShape);
    copyin_op.SetOpAttribute(copyin_Attr);

    auto& permute_op = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_PERMUTE_ELEMENT, {inTensor}, {outTensor});
    permute_op.SetAttribute(OpAttributeKey::perm, std::vector<int>{0, 2, 1});

    auto& copyout_op = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_COPY_OUT, {outTensor}, {outcast});
    (void)copyout_op;

    currFunctionPtr->inCasts_.push_back(incast);
    currFunctionPtr->outCasts_.push_back(outcast);

    InferDynShape inferShapeTest;
    inferShapeTest.RunOnFunction(*currFunctionPtr);
    std::cout << currFunctionPtr->Dump() << std::endl;
    EXPECT_EQ(inferShapeTest.PostCheck(*currFunctionPtr), SUCCESS);
}

TEST_F(InferShapeTest, TestErfcUnary)
{
    auto currFunctionPtr =
        std::make_shared<Function>(Program::GetInstance(), "TestErfcInferShape", "TestErfcInferShape", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    std::vector<int64_t> shape = {8, 16};
    auto shapeImme = OpImmediate::Specified(shape);
    auto incast = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    auto ubTensor = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    auto ubTensorOut = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    auto outCast = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    outCast->UpdateDynValidShape({CreateTestScalarVar("output_0_Dim_0"), CreateTestScalarVar("output_0_Dim_1")});

    auto& copyInOp = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_COPY_IN, {incast}, {ubTensor});
    auto copyinAttr = std::make_shared<CopyOpAttribute>(
        OpImmediate::Specified({0, 0}), MEM_UB, shapeImme, shapeImme, std::vector<npu::tile_fwk::OpImmediate>());
    std::vector<npu::tile_fwk::OpImmediate> toValidShape = {
        OpImmediate(CreateTestScalarVar("Input_0_Dim_0")), OpImmediate(CreateTestScalarVar("Input_0_Dim_1"))};
    copyinAttr->SetToDynValidShape(toValidShape);
    copyInOp.SetOpAttribute(copyinAttr);

    auto& erfcOp = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_ERFC, {ubTensor}, {ubTensorOut});
    (void)erfcOp;

    auto& copyOutOp = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_COPY_OUT, {ubTensorOut}, {outCast});
    auto copyoutAttr = std::make_shared<CopyOpAttribute>(
        OpImmediate::Specified({0, 0}), MEM_UB, shapeImme, shapeImme, std::vector<npu::tile_fwk::OpImmediate>());
    copyOutOp.SetOpAttribute(copyoutAttr);

    currFunctionPtr->inCasts_.push_back(incast);
    currFunctionPtr->outCasts_.push_back(outCast);

    InferDynShape inferShapeTest;
    inferShapeTest.RunOnFunction(*currFunctionPtr);
    EXPECT_EQ(inferShapeTest.PostCheck(*currFunctionPtr), SUCCESS);
}

} // namespace tile_fwk
} // namespace npu
