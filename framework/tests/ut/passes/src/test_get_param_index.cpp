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
 * \file test_get_param_index.cpp
 * \brief
 */

#include "gtest/gtest.h"

#include "interface/tensor/logical_tensor.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/operation/op_infer_shape_impl.h"
#include "passes/block_graph_pass/infer_param_index.h"
#include "interface/operation/attribute.h"
#include "interface/function/function.h"

using namespace npu::tile_fwk;

class GetParamIdxTest : public testing::Test {
public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}

    void SetUp() override {
        Program::GetInstance().Reset();
        config::Reset();
    }

    void TearDown() override {}
};

TEST_F(GetParamIdxTest, TestAdd) {
    auto rootFuncPtr = std::make_shared<Function>(Program::GetInstance(), "TestParams", "TestParams", nullptr);
    rootFuncPtr->rootFunc_ = rootFuncPtr.get();
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestAddParams", "TestAddParams", rootFuncPtr.get());
    EXPECT_TRUE(currFunctionPtr != nullptr);
    rootFuncPtr->rootFunc_->programs_.emplace(currFunctionPtr->GetFuncMagic(), currFunctionPtr.get());

    // Prepare the graph
    std::vector<int64_t> shape = {8, 16};
    auto shapeImme = OpImmediate::Specified(shape);
    auto incast1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    auto incast2 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    auto ubTensor1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    ubTensor1->UpdateDynValidShape({SymbolicScalar("S0"), SymbolicScalar("S1")});
    auto ubTensor2 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    ubTensor2->UpdateDynValidShape({SymbolicScalar("Z0"), SymbolicScalar("Z1")});
    auto ubTensor3 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    ubTensor3->UpdateDynValidShape({SymbolicScalar("S0"), SymbolicScalar("Z1")});
    auto outCast = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);

    auto &copy_op1 = currFunctionPtr->AddOperation(Opcode::OP_COPY_IN, {incast1}, {ubTensor1});
    std::vector<npu::tile_fwk::OpImmediate> fromOffset = {OpImmediate::Parameter(1), OpImmediate::Parameter(2)};
    auto copyin1Attr = std::make_shared<CopyOpAttribute>(fromOffset, MEM_UB, shapeImme, shapeImme);
    std::vector<npu::tile_fwk::OpImmediate> toValidShape = {OpImmediate::Parameter(3), OpImmediate::Parameter(4)};
    copyin1Attr->SetToDynValidShape(toValidShape);
    copy_op1.SetIOpAttrOffset(0, 0);
    copy_op1.SetOpAttribute(copyin1Attr);

    auto &copy_op2 = currFunctionPtr->AddOperation(Opcode::OP_COPY_IN, {incast2}, {ubTensor2});
    fromOffset = {OpImmediate::Parameter(6), OpImmediate::Parameter(7)};
    auto copyin2Attr = std::make_shared<CopyOpAttribute>(fromOffset, MEM_UB, shapeImme, shapeImme);
    std::vector<npu::tile_fwk::OpImmediate> toValidShape1 = {OpImmediate::Parameter(8), OpImmediate::Parameter(9)};
    copyin2Attr->SetToDynValidShape(toValidShape1);
    copy_op2.SetIOpAttrOffset(0, 5);
    copy_op2.SetOpAttribute(copyin2Attr);

    auto& add_op = currFunctionPtr->AddOperation(Opcode::OP_ADD, {ubTensor1, ubTensor2}, {ubTensor3});
    (void) add_op;
    auto& copy_out_op = currFunctionPtr->AddOperation(Opcode::OP_COPY_OUT, {ubTensor3}, {outCast});
    copy_out_op.SetOOpAttrOffset(0, 11);

    currFunctionPtr->inCasts_.push_back(incast1);
    currFunctionPtr->inCasts_.push_back(incast2);
    currFunctionPtr->outCasts_.push_back(outCast);

    InferParamIndex getParamIndexTest;
    getParamIndexTest.RunOnFunction(*rootFuncPtr);
    EXPECT_TRUE(true);
}

TEST_F(GetParamIdxTest, TestAddExp) {
    auto rootGraphPtr = std::make_shared<Function>(Program::GetInstance(), "TestParams", "TestParams", nullptr);
    EXPECT_TRUE(rootGraphPtr != nullptr);
    rootGraphPtr->rootFunc_ = rootGraphPtr.get();
    auto subGraphPtr0 = std::make_shared<Function>(Program::GetInstance(), "TestAddParams", "TestAddParams", rootGraphPtr.get());
    auto subGraphPtr1 = std::make_shared<Function>(Program::GetInstance(), "TestExpParams", "TestExpParams", rootGraphPtr.get());
    rootGraphPtr->rootFunc_->programs_.emplace(subGraphPtr0->GetFuncMagic(), subGraphPtr0.get());
    rootGraphPtr->rootFunc_->programs_.emplace(subGraphPtr1->GetFuncMagic(), subGraphPtr1.get());
    // Prepare the graph
    std::vector<int64_t> shape = {8, 16};
    auto shapeImme = OpImmediate::Specified(shape);
    auto incast1 = std::make_shared<LogicalTensor>(*subGraphPtr0, DT_FP32, shape);
    auto incast2 = std::make_shared<LogicalTensor>(*subGraphPtr0, DT_FP32, shape);
    auto ubTensor1 = std::make_shared<LogicalTensor>(*subGraphPtr0, DT_FP32, shape);
    ubTensor1->UpdateDynValidShape({SymbolicScalar("S0"), SymbolicScalar("S1")});
    auto ubTensor2 = std::make_shared<LogicalTensor>(*subGraphPtr0, DT_FP32, shape);
    ubTensor2->UpdateDynValidShape({SymbolicScalar("Z0"), SymbolicScalar("Z1")});
    auto ubTensor3 = std::make_shared<LogicalTensor>(*subGraphPtr0, DT_FP32, shape);
    ubTensor3->UpdateDynValidShape({SymbolicScalar("S0"), SymbolicScalar("Z1")});
    auto outCast = std::make_shared<LogicalTensor>(*subGraphPtr1, DT_FP32, shape);

    auto &copy_op1 = subGraphPtr0->AddOperation(Opcode::OP_COPY_IN, {incast1}, {ubTensor1});
    std::vector<npu::tile_fwk::OpImmediate> fromOffset = {OpImmediate::Parameter(1), OpImmediate::Parameter(2)};
    auto copyin1Attr = std::make_shared<CopyOpAttribute>(fromOffset, MEM_UB, shapeImme, shapeImme);
    std::vector<npu::tile_fwk::OpImmediate> toValidShape = {OpImmediate::Parameter(3), OpImmediate::Parameter(4)};
    copyin1Attr->SetToDynValidShape(toValidShape);
    copy_op1.SetIOpAttrOffset(0, 0);
    copy_op1.SetOpAttribute(copyin1Attr);

    auto &copy_op2 = subGraphPtr0->AddOperation(Opcode::OP_COPY_IN, {incast2}, {ubTensor2});
    fromOffset = {OpImmediate::Parameter(6), OpImmediate::Parameter(7)};
    auto copyin2Attr = std::make_shared<CopyOpAttribute>(fromOffset, MEM_UB, shapeImme, shapeImme);
    std::vector<npu::tile_fwk::OpImmediate> toValidShape1 = {OpImmediate::Parameter(8), OpImmediate::Parameter(9)};
    copyin2Attr->SetToDynValidShape(toValidShape1);
    copy_op2.SetIOpAttrOffset(0, 5);
    copy_op2.SetOpAttribute(copyin2Attr);

    auto& add_op = subGraphPtr0->AddOperation(Opcode::OP_ADD, {ubTensor1, ubTensor2}, {ubTensor3});
    (void) add_op;
    auto tmpCast = std::make_shared<LogicalTensor>(*subGraphPtr0, DT_FP32, shape);
    auto& copy_out_op = subGraphPtr0->AddOperation(Opcode::OP_COPY_OUT, {ubTensor3}, {tmpCast});
    auto copyout1Attr = std::make_shared<CopyOpAttribute>(MEM_UB, OpImmediate::Specified({0, 0}), shapeImme, shapeImme);
    copy_out_op.SetOpAttribute(copyout1Attr);
    copy_out_op.SetOOpAttrOffset(0, 10);

    auto ubTensor4 = std::make_shared<LogicalTensor>(*subGraphPtr1, DT_FP32, shape);
    ubTensor4->UpdateDynValidShape({SymbolicScalar("X0"), SymbolicScalar("X1")});
    auto &copy_op3 = subGraphPtr1->AddOperation(Opcode::OP_COPY_IN, {tmpCast}, {ubTensor4});
    fromOffset = {OpImmediate::Parameter(16), OpImmediate::Parameter(17)};
    auto copyin3Attr = std::make_shared<CopyOpAttribute>(fromOffset, MEM_UB, shapeImme, shapeImme);
    std::vector<npu::tile_fwk::OpImmediate> toValidShape2 = {OpImmediate::Parameter(18), OpImmediate::Parameter(19)};
    copyin3Attr->SetToDynValidShape(toValidShape2);
    copy_op3.SetIOpAttrOffset(0, 15);
    copy_op3.SetOpAttribute(copyin3Attr);

    auto ubTensor5 = std::make_shared<LogicalTensor>(*subGraphPtr1, DT_FP32, shape);
    auto &exp = subGraphPtr1->AddOperation(Opcode::OP_EXP, {ubTensor4}, {ubTensor5});
    (void) exp;

    auto& copy_out_op1 = subGraphPtr1->AddOperation(Opcode::OP_COPY_OUT, {ubTensor5}, {outCast});
    copy_out_op1.SetOOpAttrOffset(0, 11);

    subGraphPtr0->inCasts_.push_back(incast1);
    subGraphPtr0->inCasts_.push_back(incast2);
    subGraphPtr0->outCasts_.push_back(tmpCast);

    subGraphPtr1->inCasts_.push_back(tmpCast);
    subGraphPtr1->outCasts_.push_back(outCast);

    InferParamIndex getParamIndexTest;
    getParamIndexTest.RunOnFunction(*rootGraphPtr);
    EXPECT_TRUE(true);
}
