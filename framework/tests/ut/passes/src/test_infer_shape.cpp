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
#include "interface/inner/tilefwk.h"
#include "interface/operation/op_infer_shape_impl.h"
#include "passes/tile_graph_pass/graph_constraint/infer_dyn_shape.h"
#include "interface/operation/attribute.h"

namespace npu {
namespace tile_fwk {
class InferShapeTest : public testing::Test {
public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}

    void SetUp() override {
        Program::GetInstance().Reset();
        config::Reset();
    }

    void TearDown() override {}
};

TEST_F(InferShapeTest, TestAdd) {
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(),
                                                      "TestAddInferShape",
                                                      "TestAddInferShape",
                                                      nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    // Prepare the graph
    std::vector<int64_t> shape = {8, 16};
    auto shapeImme = OpImmediate::Specified(shape);
    auto incast1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    auto incast2 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    auto ubTensor1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    auto ubTensor2 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    auto ubTensor3 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    auto outCast = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    outCast->UpdateDynValidShape({SymbolicScalar("output_0_Dim_0"),
                                  SymbolicScalar("output_0_Dim_1")});

    auto &copy_op1 = currFunctionPtr->AddOperation(Opcode::OP_COPY_IN, {incast1}, {ubTensor1});
    auto copyin1Attr = std::make_shared<CopyOpAttribute>(OpImmediate::Specified({0, 0}),
                                                         MEM_UB,
                                                         shapeImme,
                                                         shapeImme,
                                                         std::vector<npu::tile_fwk::OpImmediate>());
    std::vector<npu::tile_fwk::OpImmediate> toValidShape = {OpImmediate(SymbolicScalar("Input_0_Dim_0")),
                                                            OpImmediate(SymbolicScalar("Input_0_Dim_1"))};
    copyin1Attr->SetToDynValidShape(toValidShape);
    copy_op1.SetOpAttribute(copyin1Attr);

    auto &copy_op2 = currFunctionPtr->AddOperation(Opcode::OP_COPY_IN, {incast2}, {ubTensor2});
    auto copyin2Attr = std::make_shared<CopyOpAttribute>(OpImmediate::Specified({0, 0}),
                                                         MEM_UB,
                                                         shapeImme,
                                                         shapeImme,
                                                         std::vector<npu::tile_fwk::OpImmediate>());
    std::vector<npu::tile_fwk::OpImmediate> toValidShape1 = {OpImmediate(SymbolicScalar("Input_1_Dim_0")),
                                                             OpImmediate(SymbolicScalar("Input_1_Dim_1"))};
    copyin2Attr->SetToDynValidShape(toValidShape1);
    copy_op2.SetOpAttribute(copyin2Attr);

    auto& add_op = currFunctionPtr->AddOperation(Opcode::OP_ADD, {ubTensor1, ubTensor2}, {ubTensor3});
    (void) add_op;
    auto& copy_out_op = currFunctionPtr->AddOperation(Opcode::OP_COPY_OUT, {ubTensor3}, {outCast});
    (void) copy_out_op;


    currFunctionPtr->inCasts_.push_back(incast1);
    currFunctionPtr->inCasts_.push_back(incast2);
    currFunctionPtr->outCasts_.push_back(outCast);

    InferDynShape inferShapeTest;
    inferShapeTest.RunOnFunction(*currFunctionPtr);
    std::cout << currFunctionPtr->Dump() << std::endl;
    EXPECT_EQ(inferShapeTest.PostCheck(*currFunctionPtr), SUCCESS);
}

TEST_F(InferShapeTest, TestAddAlignCase) {
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(),
                                                      "TestAddInferShape",
                                                      "TestAddInferShape",
                                                      nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    // Prepare the graph
    std::vector<int64_t> shape = {8, 16};
    auto shapeImme = OpImmediate::Specified(shape);
    auto incast1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    auto incast2 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    auto ubTensor1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    auto ubTensor2 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    auto ubTensor3 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    auto outCast = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);

    auto &copy_op1 = currFunctionPtr->AddOperation(Opcode::OP_COPY_IN, {incast1}, {ubTensor1});
    auto copyin1Attr = std::make_shared<CopyOpAttribute>(OpImmediate::Specified({0, 0}),
                                                         MEM_UB,
                                                         shapeImme,
                                                         shapeImme,
                                                         std::vector<npu::tile_fwk::OpImmediate>());
    copy_op1.SetOpAttribute(copyin1Attr);

    auto &copy_op2 = currFunctionPtr->AddOperation(Opcode::OP_COPY_IN, {incast2}, {ubTensor2});
    auto copyin2Attr = std::make_shared<CopyOpAttribute>(OpImmediate::Specified({0, 0}),
                                                         MEM_UB,
                                                         shapeImme,
                                                         shapeImme,
                                                         std::vector<npu::tile_fwk::OpImmediate>());
    copy_op2.SetOpAttribute(copyin2Attr);

    auto& add_op = currFunctionPtr->AddOperation(Opcode::OP_ADD, {ubTensor1, ubTensor2}, {ubTensor3});
    (void) add_op;
    auto& copy_out_op = currFunctionPtr->AddOperation(Opcode::OP_COPY_OUT, {ubTensor3}, {outCast});
    auto copyoutAttr = std::make_shared<CopyOpAttribute>(MEM_UB,
                                                         OpImmediate::Specified({0, 0}),
                                                         shapeImme,
                                                         shapeImme,
                                                         std::vector<npu::tile_fwk::OpImmediate>());
    copy_out_op.SetOpAttribute(copyoutAttr);


    currFunctionPtr->inCasts_.push_back(incast1);
    currFunctionPtr->inCasts_.push_back(incast2);
    currFunctionPtr->outCasts_.push_back(outCast);

    InferDynShape inferShapeTest;
    inferShapeTest.RunOnFunction(*currFunctionPtr);
    std::cout << currFunctionPtr->Dump() << std::endl;
    EXPECT_EQ(inferShapeTest.PostCheck(*currFunctionPtr), SUCCESS);
}

TEST_F(InferShapeTest, TestAddExp) {
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(),
                                                      "TestAddInferShape",
                                                      "TestAddInferShape",
                                                      nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    // Prepare the graph
    std::vector<int64_t> shape = {8, 16};
    auto shapeImme = OpImmediate::Specified(shape);
    auto incast1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    auto incast2 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    auto ubTensor1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    auto ubTensor2 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    auto ubTensor3 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    auto outCast = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    outCast->UpdateDynValidShape({SymbolicScalar("output_0_Dim_0"),
                                  SymbolicScalar("output_0_Dim_1")});

    auto &copy_op1 = currFunctionPtr->AddOperation(Opcode::OP_COPY_IN, {incast1}, {ubTensor1});
    auto copyin1Attr = std::make_shared<CopyOpAttribute>(OpImmediate::Specified({0, 0}),
                                                         MEM_UB,
                                                         shapeImme,
                                                         shapeImme,
                                                         std::vector<npu::tile_fwk::OpImmediate>());
    std::vector<npu::tile_fwk::OpImmediate> toValidShape = {OpImmediate(SymbolicScalar("Input_0_Dim_0")),
                                                            OpImmediate(SymbolicScalar("Input_0_Dim_1"))};
    copyin1Attr->SetToDynValidShape(toValidShape);
    copy_op1.SetOpAttribute(copyin1Attr);

    auto &copy_op2 = currFunctionPtr->AddOperation(Opcode::OP_COPY_IN, {incast2}, {ubTensor2});
    auto copyin2Attr = std::make_shared<CopyOpAttribute>(OpImmediate::Specified({0, 0}),
                                                         MEM_UB,
                                                         shapeImme,
                                                         shapeImme);
    std::vector<npu::tile_fwk::OpImmediate> toValidShape1 = {OpImmediate(SymbolicScalar("Input_1_Dim_0")),
                                                             OpImmediate(SymbolicScalar("Input_1_Dim_1"))};
    copyin2Attr->SetToDynValidShape(toValidShape1);
    copy_op2.SetOpAttribute(copyin2Attr);

    auto& add_op = currFunctionPtr->AddOperation(Opcode::OP_ADD, {ubTensor1, ubTensor2}, {ubTensor3});
    (void) add_op;
    auto tmpCast = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    auto& copy_out_op = currFunctionPtr->AddOperation(Opcode::OP_COPY_OUT, {ubTensor3}, {tmpCast});
    auto copyout1Attr = std::make_shared<CopyOpAttribute>(MEM_UB,
                                                          OpImmediate::Specified({0, 0}),
                                                          shapeImme,
                                                          shapeImme,
                                                          std::vector<npu::tile_fwk::OpImmediate>());
    copy_out_op.SetOpAttribute(copyout1Attr);

    auto ubTensor4 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    auto &copy_op3 = currFunctionPtr->AddOperation(Opcode::OP_COPY_IN, {tmpCast}, {ubTensor4});
    auto copyin3Attr = std::make_shared<CopyOpAttribute>(OpImmediate::Specified({0, 0}),
                                                         MEM_UB,
                                                         shapeImme,
                                                         shapeImme,
                                                         std::vector<npu::tile_fwk::OpImmediate>());
    copy_op3.SetOpAttribute(copyin3Attr);

    auto ubTensor5 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    auto &exp = currFunctionPtr->AddOperation(Opcode::OP_EXP, {ubTensor4}, {ubTensor5});
    (void) exp;

    auto& copy_out_op1 = currFunctionPtr->AddOperation(Opcode::OP_COPY_OUT, {ubTensor5}, {outCast});
    (void) copy_out_op1;

    currFunctionPtr->inCasts_.push_back(incast1);
    currFunctionPtr->inCasts_.push_back(incast2);
    currFunctionPtr->outCasts_.push_back(outCast);

    InferDynShape inferShapeTest;
    inferShapeTest.RunOnFunction(*currFunctionPtr);
    std::cout << currFunctionPtr->Dump() << std::endl;
    EXPECT_EQ(inferShapeTest.PostCheck(*currFunctionPtr), SUCCESS);
}

TEST_F(InferShapeTest, TestReduce) {
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(),
                                                      "TestReduceInferShape",
                                                      "TestReduceInferShape",
                                                      nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    // Prepare the graph
    std::vector<int64_t> inshape = {4, 8, 16};
    std::vector<int64_t> outshape = {4, 8, 8};
    auto shapeImme = OpImmediate::Specified(inshape);
    auto incast = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, inshape);
    auto outcast = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, outshape);
    auto inTensor = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, inshape);
    auto outTensor = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, outshape);

    auto &copyin_op = currFunctionPtr->AddOperation(Opcode::OP_COPY_IN, {incast}, {inTensor});
    auto copyin_Attr = std::make_shared<CopyOpAttribute>(OpImmediate::Specified({0, 0}),
                                                         MEM_UB,
                                                         shapeImme,
                                                         shapeImme,
                                                         std::vector<OpImmediate>());
    std::vector<OpImmediate> toValidShape = {OpImmediate(SymbolicScalar("Input_0_Dim_0")),
                                             OpImmediate(SymbolicScalar("Input_0_Dim_1")),
                                             OpImmediate(SymbolicScalar("Input_0_Dim_2"))};
    copyin_Attr->SetToDynValidShape(toValidShape);
    copyin_op.SetOpAttribute(copyin_Attr);

    auto &reduce_op = currFunctionPtr->AddOperation(Opcode::OP_ROWMAX_SINGLE, {inTensor}, {outTensor});
    auto axis = inshape.size() - 1;
    reduce_op.SetAttribute(OP_ATTR_PREFIX + "AXIS", static_cast<int>(axis));
    (void) reduce_op;

    auto &copyout_op = currFunctionPtr->AddOperation(Opcode::OP_COPY_OUT, {outTensor}, {outcast});
    (void) copyout_op;

    currFunctionPtr->inCasts_.push_back(incast);
    currFunctionPtr->outCasts_.push_back(outcast);

    InferDynShape inferShapeTest;
    inferShapeTest.RunOnFunction(*currFunctionPtr);
    std::cout << currFunctionPtr->Dump() << std::endl;
    EXPECT_EQ(inferShapeTest.PostCheck(*currFunctionPtr), SUCCESS);
}

TEST_F(InferShapeTest, TestView) {
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(),
                                                      "TesViewInferShape",
                                                      "TesViewInferShape",
                                                      nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    // Prepare the graph
    std::vector<int64_t> shape = {8, 16};
    auto incast = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    auto outcast = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);

    incast->UpdateDynValidShape({SymbolicScalar("input_0_Dim_0"), SymbolicScalar("input_0_Dim_1")});
    auto &view_op = currFunctionPtr->AddOperation(Opcode::OP_VIEW, {incast}, {outcast});
    auto view_Attr = std::make_shared<ViewOpAttribute>(std::vector<int64_t>(),
                                                       MEM_UNKNOWN,
                                                       std::vector<SymbolicScalar>(),
                                                       std::vector<SymbolicScalar>());
    view_Attr->SetFromOffset(std::vector<int64_t>(),
                             {SymbolicScalar("Offset_0_Dim_0"),
                              SymbolicScalar("Offset_0_Dim_1")});
    view_op.SetOpAttribute(view_Attr);

    currFunctionPtr->inCasts_.push_back(incast);
    currFunctionPtr->outCasts_.push_back(outcast);

    InferDynShape inferShapeTest;
    inferShapeTest.RunOnFunction(*currFunctionPtr);
    std::cout << currFunctionPtr->Dump() << std::endl;
    std::cout << view_op.GetOOperands()[0]->Dump() << std::endl;
    EXPECT_EQ(inferShapeTest.PostCheck(*currFunctionPtr), SUCCESS);
}

TEST_F(InferShapeTest, TestViewAlign) {
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(),
                                                      "TesViewInferShape",
                                                      "TesViewInferShape",
                                                      nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    // Prepare the graph
    std::vector<int64_t> shape = {8, 16};
    std::vector<int64_t> offset = {2, 0};
    std::vector<int64_t> viewshape = {8, 4};
    auto incast = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    auto outcast = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, viewshape);

    auto &view_op = currFunctionPtr->AddOperation(Opcode::OP_VIEW, {incast}, {outcast});
    auto view_Attr = std::make_shared<ViewOpAttribute>(std::vector<int64_t>(),
                                                       MEM_UNKNOWN,
                                                       std::vector<SymbolicScalar>(),
                                                       std::vector<SymbolicScalar>());
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

TEST_F(InferShapeTest, TestAssemble) {
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(),
                                                      "TestAssembleInferShape",
                                                      "TestAssembleInferShape",
                                                      nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    // Prepare the graph
    std::vector<int64_t> shape = {8, 16};
    auto incast = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    auto outcast = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);

    incast->UpdateDynValidShape({SymbolicScalar("input_0_Dim_0"), SymbolicScalar("input_0_Dim_1")});
    outcast->UpdateDynValidShape({SymbolicScalar("output_0_Dim_0"), SymbolicScalar("output_0_Dim_1")});
    auto &assemble_op = currFunctionPtr->AddOperation(Opcode::OP_ASSEMBLE, {incast}, {outcast});
    auto assemble_Attr = std::make_shared<AssembleOpAttribute>(MEM_UNKNOWN,
                                                               std::vector<int64_t>(),
                                                               std::vector<SymbolicScalar>(),
                                                               std::vector<SymbolicScalar>());

    auto dynOffset = {SymbolicScalar("DynOffset_0_Dim_0"), SymbolicScalar("DynOffset_0_Dim_1")};
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

TEST_F(InferShapeTest, TestFailCopyOut) {
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(),
                                                      "TestCopyOutInferShape",
                                                      "TestCopyOutInferShape",
                                                      nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    // Prepare the graph
    std::vector<int64_t> inshape = {8, 16};
    std::vector<int64_t> outshape = {8, 8};
    auto incast = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, inshape);
    auto outcast = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, outshape);

    auto &copyout_op = currFunctionPtr->AddOperation(Opcode::OP_COPY_OUT, {incast}, {outcast});
    (void) copyout_op;
    currFunctionPtr->inCasts_.push_back(incast);
    currFunctionPtr->outCasts_.push_back(outcast);

    InferDynShape inferShapeTest;
    inferShapeTest.RunOnFunction(*currFunctionPtr);
    std::cout << currFunctionPtr->Dump() << std::endl;
    EXPECT_EQ(inferShapeTest.PostCheck(*currFunctionPtr), FAILED);
}

TEST_F(InferShapeTest, TestCopyOut) {
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(),
                                                      "TestCopyOutInferShape",
                                                      "TestCopyOutInferShape",
                                                      nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    // Prepare the graph
    std::vector<int64_t> inshape = {8, 16};
    std::vector<int64_t> outshape = {8, 16};
    auto toOffsetImme = OpImmediate::Specified({4, 4});
    auto inshapeImme = OpImmediate::Specified(inshape);
    auto incast = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, inshape);
    auto outcast = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, outshape);
    incast->UpdateDynValidShape({SymbolicScalar("input_0_Dim_0"), SymbolicScalar("input_0_Dim_1")});

    auto &copyout_op = currFunctionPtr->AddOperation(Opcode::OP_COPY_OUT, {incast}, {outcast});
    auto copyout_Attr = std::make_shared<CopyOpAttribute>(MEM_DEVICE_DDR,
                                                          toOffsetImme,
                                                          inshapeImme,
                                                          inshapeImme,
                                                          std::vector<OpImmediate>());
    copyout_op.SetOpAttribute(copyout_Attr);

    currFunctionPtr->inCasts_.push_back(incast);
    currFunctionPtr->outCasts_.push_back(outcast);

    InferDynShape inferShapeTest;
    inferShapeTest.RunOnFunction(*currFunctionPtr);
    std::cout << currFunctionPtr->Dump() << std::endl;
    std::cout << copyout_op.GetOOperands()[0]->Dump() << std::endl;
    EXPECT_EQ(inferShapeTest.PostCheck(*currFunctionPtr), SUCCESS);
}

TEST_F(InferShapeTest, TestCopyIn) {
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(),
                                                      "TestCopyInInferShape",
                                                      "TestCopyInInferShape",
                                                      nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    // Prepare the graph
    std::vector<int64_t> inshape = {8, 16};
    std::vector<int64_t> outshape = {8, 8};
    auto fromOffsetImme = OpImmediate::Specified({4, 4});
    auto inshapeImme = OpImmediate::Specified(inshape);
    auto incast = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, inshape);
    auto outcast = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, outshape);
    incast->UpdateDynValidShape({SymbolicScalar("input_0_Dim_0"), SymbolicScalar("input_0_Dim_1")});

    auto &copyin_op = currFunctionPtr->AddOperation(Opcode::OP_COPY_IN, {incast}, {outcast});
    auto copyin_Attr = std::make_shared<CopyOpAttribute>(fromOffsetImme,
                                                          MEM_UNKNOWN,
                                                          inshapeImme,
                                                          inshapeImme,
                                                          std::vector<OpImmediate>());
    copyin_op.SetOpAttribute(copyin_Attr);

    currFunctionPtr->inCasts_.push_back(incast);
    currFunctionPtr->outCasts_.push_back(outcast);

    InferDynShape inferShapeTest;
    inferShapeTest.RunOnFunction(*currFunctionPtr);
    std::cout << currFunctionPtr->Dump() << std::endl;
    std::cout << copyin_op.GetOOperands()[0]->Dump() << std::endl;
    EXPECT_EQ(inferShapeTest.PostCheck(*currFunctionPtr), SUCCESS);
}

TEST_F(InferShapeTest, TestReshape) {
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(),
                                                      "TestReshapeInferShape",
                                                      "TestReshapeInferShape",
                                                      nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    // Prepare the graph
    std::vector<int64_t> inshape = {8, 16};
    std::vector<int64_t> outshape = {4, 4};
    auto shapeImme = OpImmediate::Specified(inshape);
    auto incast = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, inshape);
    auto outcast = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, outshape);
    auto inTensor = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, inshape);
    auto outTensor = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, outshape);
    incast->UpdateDynValidShape({SymbolicScalar("input_0_Dim_0"), SymbolicScalar("input_0_Dim_1")});
    outcast->UpdateDynValidShape({SymbolicScalar("output_0_Dim_0"), SymbolicScalar("output_0_Dim_1")});

    auto &copyin_op = currFunctionPtr->AddOperation(Opcode::OP_COPY_IN, {incast}, {inTensor});
    auto copyin_Attr = std::make_shared<CopyOpAttribute>(OpImmediate::Specified({0, 0}),
                                                         MEM_UNKNOWN,
                                                         shapeImme,
                                                         shapeImme,
                                                         std::vector<OpImmediate>());

    std::vector<OpImmediate> toValidShape = {OpImmediate(SymbolicScalar("Input_1_Dim_0")),
                                             OpImmediate(SymbolicScalar("Input_1_Dim_1"))};
    copyin_Attr->SetToDynValidShape(toValidShape);
    copyin_op.SetOpAttribute(copyin_Attr);

    auto &reshape_op = currFunctionPtr->AddOperation(Opcode::OP_RESHAPE, {inTensor}, {outTensor});
    (void) reshape_op;

    auto &copyout_op = currFunctionPtr->AddOperation(Opcode::OP_COPY_OUT, {outTensor}, {outcast});
    (void) copyout_op;

    currFunctionPtr->inCasts_.push_back(incast);
    currFunctionPtr->outCasts_.push_back(outcast);

    InferDynShape inferShapeTest;
    inferShapeTest.RunOnFunction(*currFunctionPtr);
    std::cout << currFunctionPtr->Dump() << std::endl;
    std::cout << reshape_op.GetOOperands()[0]->Dump() << std::endl;
    EXPECT_EQ(inferShapeTest.PostCheck(*currFunctionPtr), SUCCESS);
}

TEST_F(InferShapeTest, TestSHMEM_GET_GM2UB) {
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestSHMEM_GET_GM2UB", "TestSHMEM_GET_GM2UB", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    std::vector<int64_t> inshape0 = {1, 1};
    std::vector<int64_t> inshape1 = {1, 1, 8, 16};
    std::vector<int64_t> outshape = {8, 16};
    auto shapeImme = OpImmediate::Specified(outshape);

    auto incast0 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, inshape0);
    auto incast1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, inshape1);
    auto shmemGetGm2UBOut0 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, outshape);
    auto shmemGetGm2UBOut1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, outshape);
    auto outcast = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, outshape);

    auto &shmemGetGm2UB_op = currFunctionPtr->AddOperation(Opcode::OP_SHMEM_GET_GM2UB, {incast0, incast1}, {shmemGetGm2UBOut0, shmemGetGm2UBOut1});
    currFunctionPtr->AddOperation(Opcode::OP_COPY_OUT, {shmemGetGm2UBOut0}, {outcast});
    
    auto shmemGetGm2UB_Attr = std::make_shared<CopyOpAttribute>(OpImmediate::Specified({0, 0}),
                                                         MEM_UB,
                                                         shapeImme,
                                                         shapeImme,
                                                         std::vector<OpImmediate>());

    std::vector<OpImmediate> toValidShape = {OpImmediate(SymbolicScalar("Input_0_Dim_0")),
                                             OpImmediate(SymbolicScalar("Input_0_Dim_1"))};
    shmemGetGm2UB_Attr->SetToDynValidShape(toValidShape);
    shmemGetGm2UB_op.SetOpAttribute(shmemGetGm2UB_Attr);
    
    currFunctionPtr->inCasts_.push_back(incast0);
    currFunctionPtr->inCasts_.push_back(incast1);
    currFunctionPtr->outCasts_.push_back(outcast);

    InferDynShape inferShapeTest;
    inferShapeTest.RunOnFunction(*currFunctionPtr);
    std::cout << currFunctionPtr->Dump() << std::endl;
    EXPECT_NE(shmemGetGm2UBOut1->GetDynValidShape().size(), 0);
    EXPECT_EQ(inferShapeTest.PostCheck(*currFunctionPtr), SUCCESS);
}
}
}