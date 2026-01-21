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
 * \file test_codegen_preproc.cpp
 * \brief Unit test for codegen_preproc pass.
 */

#include <gtest/gtest.h>
#include "interface/function/function.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "passes/block_graph_pass/codegen_preproc.h"
#include "passes/tile_graph_pass/graph_constraint/pad_local_buffer.h"
#include "passes/tile_graph_pass/graph_constraint/axis_combine.h"
#include "computational_graph_builder.h"
#include "interface/configs/config_manager.h"
#include "ut_json/ut_json_tool.h"
#include <vector>
#include <string>

namespace npu {
namespace tile_fwk {
constexpr int CP_NUM1 = 1;
constexpr int CP_NUM16 = 16;
constexpr int CP_NUM256 = 256;
const std::vector<bool> AXIS_COMBINED = {true};
const std::string REDUCE_AXIS = OP_ATTR_PREFIX + "AXIS";

class CodegenPreprocTest : public testing::Test {
public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}

    void SetUp() override {
        Program::GetInstance().Reset();
        config::Reset();
        config::SetHostOption(COMPILE_STAGE, HOST_COMPILE_END);
        config::SetPlatformConfig(KEY_ENABLE_COST_MODEL, false);
    }
    void TearDown() override {}
};

TEST_F(CodegenPreprocTest, TestSaveGmTensorParamIdxToOp) {
    auto rootFuncPtr = std::make_shared<Function>(Program::GetInstance(), "TestSaveGmTensorParamIdxToOp", "TestSaveGmTensorParamIdxToOp", nullptr);
    rootFuncPtr->rootFunc_ = rootFuncPtr.get();
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestSaveGmTensorParamIdxToOpLeaf", "TestSaveGmTensorParamIdxToOpLeaf", rootFuncPtr.get());
    EXPECT_TRUE(currFunctionPtr != nullptr);
    rootFuncPtr->rootFunc_->programs_.emplace(currFunctionPtr->GetFuncMagic(), currFunctionPtr.get());
    rootFuncPtr->SetFunctionType(FunctionType::DYNAMIC_LOOP_PATH);
    rootFuncPtr->SetUnderDynamicFunction(true);

    std::vector<int64_t> shape = {CP_NUM16, CP_NUM16};
    auto tensor1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    auto tensor2 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    auto tensor3 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    auto tensor4 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    auto tensor5 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    auto tensor6 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    std::vector<Operation *> opLogPtr;
    auto &copyin1 = currFunctionPtr->AddRawOperation(Opcode::OP_COPY_IN, {tensor1}, {tensor3});
    opLogPtr.emplace_back(&copyin1);
    auto &copyin2 = currFunctionPtr->AddRawOperation(Opcode::OP_COPY_IN, {tensor2}, {tensor4});
    opLogPtr.emplace_back(&copyin2);
    auto &add = currFunctionPtr->AddRawOperation(Opcode::OP_ADD, {tensor3, tensor4}, {tensor5});
    opLogPtr.emplace_back(&add);
    auto &copyout = currFunctionPtr->AddRawOperation(Opcode::OP_COPY_OUT, {tensor5}, {tensor6});
    opLogPtr.emplace_back(&copyout);

    int index{0};
    for (auto op : opLogPtr) {
        if (OpcodeManager::Inst().IsCopyInOrOut(op->GetOpcode())) {
            if (IsCopyIn(op->GetOpcode()))
                op->SetIOpAttrOffset(0, index++);
            else
                op->SetOOpAttrOffset(0, index++);
        }
    }

    CodegenPreproc codegenPreprocPass;
    codegenPreprocPass.SaveGmTensorParamIdxToOp(*rootFuncPtr);

    for (const auto &op : opLogPtr) {
        if (OpcodeManager::Inst().IsCopyInOrOut(op->GetOpcode())) {
            EXPECT_TRUE(op->HasAttr("GmTensorParamIdxInCallFunc"));
        }
    }
}

TEST_F(CodegenPreprocTest, TestForceCombineAxis) {
    auto rootFuncPtr = std::make_shared<Function>(Program::GetInstance(), "TestForceCombineAxis", "TestForceCombineAxis", nullptr);
    rootFuncPtr->rootFunc_ = rootFuncPtr.get();
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestForceCombineAxisLeaf", "TestForceCombineAxisLeaf", rootFuncPtr.get());
    EXPECT_TRUE(currFunctionPtr != nullptr);
    rootFuncPtr->rootFunc_->programs_.emplace(currFunctionPtr->GetFuncMagic(), currFunctionPtr.get());
    rootFuncPtr->SetFunctionType(FunctionType::DYNAMIC_LOOP_PATH);
    rootFuncPtr->SetUnderDynamicFunction(true);

    std::vector<int64_t> shape = {CP_NUM16, CP_NUM16, CP_NUM16};
    auto tensor1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    tensor1->tensor->rawshape = shape;
    auto tensor2 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    tensor2->tensor->rawshape = shape;
    auto tensor3 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    tensor3->oriShape = shape;
    tensor3->tensor->rawshape = shape;
    auto tensor4 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    auto tensor5 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    tensor5->oriShape = shape;
    tensor5->tensor->rawshape = shape;
    auto tensor6 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    tensor6->tensor->rawshape = shape;
    std::vector<Operation *> opLogPtr;
    auto &copyin1 = currFunctionPtr->AddRawOperation(Opcode::OP_COPY_IN, {tensor1}, {tensor3});
    copyin1.SetAttr(OpAttributeKey::outputCombineAxis, AXIS_COMBINED);
    opLogPtr.emplace_back(&copyin1);
    auto &copyin2 = currFunctionPtr->AddRawOperation(Opcode::OP_COPY_IN, {tensor2}, {tensor4});
    copyin2.SetAttr(OpAttributeKey::outputCombineAxis, AXIS_COMBINED);
    opLogPtr.emplace_back(&copyin2);
    auto &add = currFunctionPtr->AddRawOperation(Opcode::OP_ADD, {tensor3, tensor4}, {tensor5});
    add.SetAttr(OpAttributeKey::inputCombineAxis, AXIS_COMBINED);
    add.SetAttr(OpAttributeKey::outputCombineAxis, AXIS_COMBINED);
    opLogPtr.emplace_back(&add);
    auto &copyout = currFunctionPtr->AddRawOperation(Opcode::OP_COPY_OUT, {tensor5}, {tensor6});
    copyout.SetAttr(OpAttributeKey::inputCombineAxis, AXIS_COMBINED);
    opLogPtr.emplace_back(&copyout);

    CodegenPreproc codegenPreprocPass;
    codegenPreprocPass.ForceCombineAxis(*rootFuncPtr);
    bool inputRes{false};
    add.GetAttr(OpAttributeKey::inputCombineAxisDone, inputRes);
    EXPECT_EQ(inputRes, true);
    bool outputRes{false};
    add.GetAttr(OpAttributeKey::outputCombineAxisDone, outputRes);
    EXPECT_EQ(outputRes, true);
    std::vector<int64_t> combinedShape = {CP_NUM16, CP_NUM1, CP_NUM256};
    EXPECT_EQ(tensor3->shape, combinedShape);
    EXPECT_EQ(tensor3->oriShape, combinedShape);
    EXPECT_EQ(tensor3->tensor->rawshape, combinedShape);
    EXPECT_EQ(tensor5->shape, combinedShape);
    EXPECT_EQ(tensor5->oriShape, combinedShape);
    EXPECT_EQ(tensor5->tensor->rawshape, combinedShape);

    bool copyoutRes{false};
    copyout.GetAttr(OpAttributeKey::outputCombineAxisDone, copyoutRes);
    EXPECT_EQ(copyoutRes, true);
    EXPECT_EQ(tensor6->tensor->rawshape, combinedShape);
    bool copyin1Res{false};
    copyin1.GetAttr(OpAttributeKey::inputCombineAxisDone, copyin1Res);
    EXPECT_EQ(copyin1Res, true);
    EXPECT_EQ(tensor1->tensor->rawshape, combinedShape);
    bool copyin2Res{false};
    copyin2.GetAttr(OpAttributeKey::inputCombineAxisDone, copyin2Res);
    EXPECT_EQ(copyin2Res, true);
    EXPECT_EQ(tensor2->tensor->rawshape, combinedShape);
}

TEST_F(CodegenPreprocTest, TestCombineAxis) {
    ComputationalGraphBuilder graph;
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {4, 12, 1}, MemoryType::MEM_UB, "in"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {1, 12, 1}, MemoryType::MEM_UB, "out"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {2, 8}, MemoryType::MEM_UB, "tmp"), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_ROWSUMLINE, {"in"}, {"out", "tmp"}, "sumline", true), true);
    auto sumline = graph.GetOp("sumline");
    sumline->SetAttribute(REDUCE_AXIS, 0);

    auto funcPtr = graph.GetFunction();
    funcPtr->paramConfigs_.combineAxis = true;
    AxisCombine axisCombineTest;
    EXPECT_EQ(axisCombineTest.RunOnFunction(*funcPtr), SUCCESS);
    PadLocalBuffer padLocalBufferTest;
    EXPECT_EQ(padLocalBufferTest.RunOnFunction(*funcPtr), SUCCESS);

    auto rootFuncPtr =
        std::make_shared<Function>(Program::GetInstance(), "TestCombineAxis", "TestCombineAxis", nullptr);
    rootFuncPtr->rootFunc_ = rootFuncPtr.get();
    auto currFunctionPtr = std::make_shared<Function>(
        Program::GetInstance(), "TestCombineAxisLeaf", "TestCombineAxisLeaf", graph.GetFunction());
    EXPECT_TRUE(currFunctionPtr != nullptr);
    rootFuncPtr->rootFunc_->programs_.emplace(currFunctionPtr->GetFuncMagic(), graph.GetFunction());
    rootFuncPtr->SetFunctionType(FunctionType::DYNAMIC_LOOP_PATH);
    rootFuncPtr->SetUnderDynamicFunction(true);
    rootFuncPtr->paramConfigs_.combineAxis = true;

    CodegenPreproc codegenPreprocPass;
    EXPECT_EQ(codegenPreprocPass.RunOnFunction(*rootFuncPtr), SUCCESS);
    // Verify AxisCombine
    auto updatedOperations = rootFuncPtr->Operations();
    int64_t cnt = 0;
    for (const auto &op : updatedOperations) {
        if (op.GetOpcode() == Opcode::OP_BRCB) {
            ++cnt;
        }
    }
    EXPECT_EQ(cnt, 0);
    // Verify PadLocalBuffer
    auto tmp = graph.GetTensor("tmp");
    auto shape = tmp->GetRawTensor()->GetRawShape();
    EXPECT_EQ(shape[shape.size() - 1], CP_NUM16);
    // Verify CodegenPreproc
    sumline = graph.GetOp("sumline");
    std::vector<bool> attr;
    EXPECT_TRUE(sumline->HasAttr(OpAttributeKey::outputCombineAxis));
    sumline->GetAttr(OpAttributeKey::outputCombineAxis, attr);
    EXPECT_EQ(attr, (std::vector<bool>{true, false}));
}

} // namespace tile_fwk
} // namespace npu