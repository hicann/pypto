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
#include "tilefwk/tilefwk.h"
#include "interface/function/function.h"
#include "interface/operation/operation.h"
#include "passes/pass_mgr/pass_manager.h"

#define private public
#include "passes/block_graph_pass/loopaxes_proc.h"

namespace npu {
namespace tile_fwk {
static const int kKeepOut = -1;
static const int kNum0 = 0;
static const int kNum1 = 1;
static const int kNum2 = 2;
static const int kNum3 = 3;
static const int kNum4 = 4;
static const int kNum16 = 2;
static const std::vector<int64_t> shape1 = {kNum2, kNum16};
static const std::vector<int64_t> shape2 = {kNum2, kNum2, kNum2, kNum4};
static const std::vector<int64_t> shape3 = {kNum4, kNum2, kNum4};
static const std::vector<int64_t> shape4 = {kNum3, kNum2, kNum2, kNum4};
static const std::vector<SymbolicScalar> symShape1 = {kNum2, kNum16};
static const std::vector<SymbolicScalar> symShape2 = {kNum2, kNum2, kNum2, kNum4};
static const std::vector<SymbolicScalar> symShape3 = {kNum4, kNum2, kNum4};
static const std::vector<SymbolicScalar> symShape4 = {kNum3, kNum2, kNum2, kNum4};
static const std::vector<SymbolicScalar> expectedLoopAxis1 = {kNum2, kNum2};
static const std::vector<SymbolicScalar> expectedLoopAxis2 = {kNum3, kNum2};

class TestLoopaxesProcPass : public ::testing::Test {
public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}

    void SetUp() override {
        Program::GetInstance().Reset();
        config::Reset();
        config::SetPassGlobalConfig(KEY_VF_OPT_MARK_FOR, true);
        config::SetHostOption(COMPILE_STAGE, CS_EXECUTE_GRAPH);
        config::SetHostConfig(KEY_STRATEGY, "ExpandFunctionTestStrategy");
        config::SetPlatformConfig("ENABLE_COST_MODEL", false);
    }
    void TearDown() override {}
};

bool EqualSymShape(const std::vector<SymbolicScalar> &A, const std::vector<SymbolicScalar> &B) {
    if (A.size() != B.size()) {
        return false;
    }
    for (size_t i = 0; i < A.size(); ++i) {
        if (A[i].Dump() != B[i].Dump()) {
            return false;
        }
    }
    return true;
}

TEST_F(TestLoopaxesProcPass, LoopaxesProcUTest1) {
    auto rootFuncPtr =
        std::make_shared<Function>(Program::GetInstance(), "TestLoopaxesProcPass", "TestLoopaxesProcPass", nullptr);
    rootFuncPtr->rootFunc_ = rootFuncPtr.get();
    auto currFunctionPtr = std::make_shared<Function>(
        Program::GetInstance(), "TestLoopaxesProcPassLeaf", "TestLoopaxesProcPassLeaf", nullptr);
    rootFuncPtr->rootFunc_->programs_.emplace(currFunctionPtr->GetFuncMagic(), currFunctionPtr.get());
    rootFuncPtr->SetFunctionType(FunctionType::DYNAMIC_LOOP_PATH);
    rootFuncPtr->SetUnderDynamicFunction(true);

    auto inCast1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape1);
    inCast1->UpdateDynValidShape(symShape1);
    auto inCast2 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape2);
    inCast2->UpdateDynValidShape(symShape2);
    auto ubTensor1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape1);
    ubTensor1->UpdateDynValidShape(symShape1);
    auto ubTensor2 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape2);
    ubTensor2->UpdateDynValidShape(symShape2);
    auto ubTensor3 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape2);
    ubTensor3->UpdateDynValidShape(symShape2);
    auto ubTensor4 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape2);
    ubTensor4->UpdateDynValidShape(symShape2);
    auto ubTensor5 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape2);
    ubTensor5->UpdateDynValidShape(symShape2);
    auto ubTensor6 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape4);
    ubTensor6->UpdateDynValidShape(symShape4);
    auto ubTensor7 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape4);
    ubTensor7->UpdateDynValidShape(symShape4);
    auto ubTensor8 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape4);
    ubTensor8->UpdateDynValidShape(symShape4);
    auto outCast = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape3);
    outCast->UpdateDynValidShape(symShape3);

    auto &expand = currFunctionPtr->AddOperation(Opcode::OP_EXPAND, {inCast2}, {ubTensor2});
    expand.SetAttribute(OP_ATTR_PREFIX + "EXPANDDIM", kNum3);
    currFunctionPtr->AddOperation(npu::tile_fwk::Opcode::OP_BAR_ALL, {inCast1}, {ubTensor2});
    auto &add = currFunctionPtr->AddOperation(Opcode::OP_ADD, {ubTensor2, ubTensor3}, {ubTensor4});
    auto &mul = currFunctionPtr->AddOperation(Opcode::OP_MUL, {ubTensor2, ubTensor4}, {ubTensor5});
    auto &sub = currFunctionPtr->AddOperation(Opcode::OP_SUB, {ubTensor6, ubTensor7}, {ubTensor8});
    currFunctionPtr->AddOperation(Opcode::OP_RESHAPE, {ubTensor5}, {outCast});
    currFunctionPtr->inCasts_.push_back(inCast1);
    currFunctionPtr->inCasts_.push_back(inCast2);
    currFunctionPtr->outCasts_.push_back(outCast);

    LoopaxesProc loopaxesprocpass;
    EXPECT_EQ(loopaxesprocpass.RunOnFunction(*rootFuncPtr), SUCCESS);

    EXPECT_TRUE(expand.HasAttr(OpAttributeKey::loopGroup));
    EXPECT_EQ(expand.GetIntAttribute(OpAttributeKey::loopGroup), kNum0);
    EXPECT_TRUE(expand.HasAttr(OpAttributeKey::loopAxes));
    EXPECT_TRUE(EqualSymShape(expand.GetVectorSymbolicScalarAttribute(OpAttributeKey::loopAxes), expectedLoopAxis1));

    EXPECT_TRUE(add.HasAttr(OpAttributeKey::loopGroup));
    EXPECT_EQ(add.GetIntAttribute(OpAttributeKey::loopGroup), kNum1);
    EXPECT_TRUE(add.HasAttr(OpAttributeKey::loopAxes));
    EXPECT_TRUE(EqualSymShape(add.GetVectorSymbolicScalarAttribute(OpAttributeKey::loopAxes), expectedLoopAxis1));

    EXPECT_TRUE(mul.HasAttr(OpAttributeKey::loopGroup));
    EXPECT_EQ(mul.GetIntAttribute(OpAttributeKey::loopGroup), kNum1);
    EXPECT_TRUE(mul.HasAttr(OpAttributeKey::loopAxes));
    EXPECT_TRUE(EqualSymShape(mul.GetVectorSymbolicScalarAttribute(OpAttributeKey::loopAxes), expectedLoopAxis1));

    EXPECT_TRUE(sub.HasAttr(OpAttributeKey::loopGroup));
    EXPECT_EQ(sub.GetIntAttribute(OpAttributeKey::loopGroup), kNum2);
    EXPECT_TRUE(sub.HasAttr(OpAttributeKey::loopAxes));
    EXPECT_TRUE(EqualSymShape(sub.GetVectorSymbolicScalarAttribute(OpAttributeKey::loopAxes), expectedLoopAxis2));
}
} // namespace tile_fwk
} // namespace npu