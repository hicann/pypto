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

    void SetUp() override
    {
        Program::GetInstance().Reset();
        config::Reset();
        config::SetHostOption(COMPILE_STAGE, CS_EXECUTE_GRAPH);
        config::SetPlatformConfig(KEY_ENABLE_COST_MODEL, false);
    }
    void TearDown() override {}
};

TEST_F(CodegenPreprocTest, TestSaveGmTensorParamIdxToOp)
{
    auto rootFuncPtr = std::make_shared<Function>(
        Program::GetInstance(), "TestSaveGmTensorParamIdxToOp", "TestSaveGmTensorParamIdxToOp", nullptr);
    rootFuncPtr->rootFunc_ = rootFuncPtr.get();
    auto currFunctionPtr = std::make_shared<Function>(
        Program::GetInstance(), "TestSaveGmTensorParamIdxToOpLeaf", "TestSaveGmTensorParamIdxToOpLeaf",
        rootFuncPtr.get());
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
    std::vector<Operation*> opLogPtr;
    auto& copyin1 = currFunctionPtr->AddRawOperation(Opcode::OP_COPY_IN, {tensor1}, {tensor3});
    opLogPtr.emplace_back(&copyin1);
    auto& copyin2 = currFunctionPtr->AddRawOperation(Opcode::OP_COPY_IN, {tensor2}, {tensor4});
    opLogPtr.emplace_back(&copyin2);
    auto& add = currFunctionPtr->AddRawOperation(Opcode::OP_ADD, {tensor3, tensor4}, {tensor5});
    opLogPtr.emplace_back(&add);
    auto& copyout = currFunctionPtr->AddRawOperation(Opcode::OP_COPY_OUT, {tensor5}, {tensor6});
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

    for (const auto& op : opLogPtr) {
        if (OpcodeManager::Inst().IsCopyInOrOut(op->GetOpcode())) {
            EXPECT_TRUE(op->HasAttr("GmTensorParamIdxInCallFunc"));
        }
    }
}

TEST_F(CodegenPreprocTest, TestForceCombineAxis)
{
    auto rootFuncPtr =
        std::make_shared<Function>(Program::GetInstance(), "TestForceCombineAxis", "TestForceCombineAxis", nullptr);
    rootFuncPtr->rootFunc_ = rootFuncPtr.get();
    auto currFunctionPtr = std::make_shared<Function>(
        Program::GetInstance(), "TestForceCombineAxisLeaf", "TestForceCombineAxisLeaf", rootFuncPtr.get());
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
    std::vector<Operation*> opLogPtr;
    auto& copyin1 = currFunctionPtr->AddRawOperation(Opcode::OP_COPY_IN, {tensor1}, {tensor3});
    copyin1.SetAttr(OpAttributeKey::outputCombineAxis, AXIS_COMBINED);
    opLogPtr.emplace_back(&copyin1);
    auto& copyin2 = currFunctionPtr->AddRawOperation(Opcode::OP_COPY_IN, {tensor2}, {tensor4});
    copyin2.SetAttr(OpAttributeKey::outputCombineAxis, AXIS_COMBINED);
    opLogPtr.emplace_back(&copyin2);
    auto& add = currFunctionPtr->AddRawOperation(Opcode::OP_ADD, {tensor3, tensor4}, {tensor5});
    add.SetAttr(OpAttributeKey::inputCombineAxis, AXIS_COMBINED);
    add.SetAttr(OpAttributeKey::outputCombineAxis, AXIS_COMBINED);
    opLogPtr.emplace_back(&add);
    auto& copyout = currFunctionPtr->AddRawOperation(Opcode::OP_COPY_OUT, {tensor5}, {tensor6});
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

TEST_F(CodegenPreprocTest, TestCombineAxisRowSumLine)
{
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
    for (const auto& op : updatedOperations) {
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

TEST_F(CodegenPreprocTest, TestCombineAxisExpand)
{
    ComputationalGraphBuilder graph;
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {128, 1}, MemoryType::MEM_DEVICE_DDR, "in1"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {1, 1}, MemoryType::MEM_UB, "t1"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {64, 1}, MemoryType::MEM_UB, "t2"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {64, 1}, MemoryType::MEM_UB, "t3"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {64, 1}, MemoryType::MEM_UB, "t4"), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_COPY_IN, {"in1"}, {"t1"}, "c1", true), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_EXPAND, {"t1"}, {"t2"}, "expand", true), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_COPY_IN, {"in1"}, {"t3"}, "c2", true), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_SUB, {"t2", "t3"}, {"t4"}, "sub", true), true);
    auto expand = graph.GetOp("expand");
    expand->SetAttribute(OP_ATTR_PREFIX + "EXPANDDIM", 0);

    auto funcPtr = graph.GetFunction();
    funcPtr->paramConfigs_.combineAxis = true;
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
    // Verify CodegenPreproc
    auto afterExpand = graph.GetOp("expand");
    int axis = afterExpand->GetIntAttribute(OP_ATTR_PREFIX + "EXPANDDIM");
    EXPECT_EQ(axis, 1);
}

// 隐式expand
TEST_F(CodegenPreprocTest, TestCombineAxisExpandinline)
{
    ComputationalGraphBuilder graph;
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {1, 1}, MemoryType::MEM_DEVICE_DDR, "in1"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {16, 1}, MemoryType::MEM_DEVICE_DDR, "in2"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {1, 1}, MemoryType::MEM_UB, "t1"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {16, 1}, MemoryType::MEM_UB, "t2"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {16, 1}, MemoryType::MEM_UB, "t3"), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_COPY_IN, {"in1"}, {"t1"}, "c1", true), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_COPY_IN, {"in2"}, {"t2"}, "c2", true), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_SUB, {"t1", "t2"}, {"t3"}, "sub", true), true);
    auto sub = graph.GetOp("sub");
    sub->SetAttribute(OpAttributeKey::brcpIdx, 1);

    auto funcPtr = graph.GetFunction();
    funcPtr->paramConfigs_.combineAxis = true;
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
    for (const auto& op : updatedOperations) {
        if (op.GetOpcode() == Opcode::OP_BRCB) {
            ++cnt;
        }
    }
    EXPECT_EQ(cnt, 0);
    // Verify PadLocalBuffer
    EXPECT_EQ(graph.GetTensor("t1")->GetRawTensor()->GetRawShape(), (Shape{8, 1}));
    EXPECT_EQ(graph.GetTensor("t2")->GetRawTensor()->GetRawShape(), (Shape{16, 1}));
    // Verify CodegenPreproc
    EXPECT_EQ(sub->GetIntAttribute(OpAttributeKey::brcpIdx), 0);
    EXPECT_EQ(sub->GetIntAttribute(OpAttributeKey::brcbIdx), 1);
}

// expand input have multi consumer
TEST_F(CodegenPreprocTest, TestCombineAxisExpand2) 
{
    ComputationalGraphBuilder graph;
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {128, 1}, MemoryType::MEM_DEVICE_DDR, "in1"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {1, 1}, MemoryType::MEM_UB, "t2"), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_COPY_IN, {"in1"}, {"t2"}, "copyin1", true), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {1, 1}, MemoryType::MEM_UB, "t3"), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_EXP, {"t2"}, {"t3"}, "exp", true), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {128, 1}, MemoryType::MEM_UB, "t4"), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_EXPAND, {"t3"}, {"t4"}, "expand", true), true);
    auto expand = graph.GetOp("expand");
    expand->SetAttribute(OP_ATTR_PREFIX + "EXPANDDIM", 0);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {128, 1}, MemoryType::MEM_DEVICE_DDR, "in2"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {1, 1}, MemoryType::MEM_UB, "t22"), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_COPY_IN, {"in2"}, {"t22"}, "copyin12", true), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {1, 1}, MemoryType::MEM_UB, "t32"), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_MUL, {"t3", "t22"}, {"t32"}, "mul1", true), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {1, 1}, MemoryType::MEM_DEVICE_DDR, "out1"), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_COPY_OUT, {"t32"}, {"out1"}, "copyout", true), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {128, 128}, MemoryType::MEM_DEVICE_DDR, "in3"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {128, 128}, MemoryType::MEM_UB, "t23"), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_COPY_IN, {"in3"}, {"t23"}, "copyin13", true), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {128, 128}, MemoryType::MEM_UB, "t33"), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_MUL, {"t23", "t4"}, {"t33"}, "mul2", true), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {128, 128}, MemoryType::MEM_DEVICE_DDR, "out2"), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_COPY_OUT, {"t33"}, {"out2"}, "copyout2", true), true);

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
    // Verify PadLocalBuffer
    EXPECT_EQ(graph.GetTensor("t3")->GetRawTensor()->GetRawShape(), (Shape{8, 1}));
    // Verify CodegenPreproc
    auto afterExpand = graph.GetOp("expand");
    int axis = afterExpand->GetIntAttribute(OP_ATTR_PREFIX + "EXPANDDIM");
    EXPECT_EQ(axis, 1);
    std::vector<bool> inputAttr;
    EXPECT_TRUE(expand->HasAttr(OpAttributeKey::inputCombineAxis));
    expand->GetAttr(OpAttributeKey::inputCombineAxis, inputAttr);
    EXPECT_EQ(inputAttr, (std::vector<bool>{true}));
    std::vector<bool> outAttr;
    EXPECT_TRUE(expand->HasAttr(OpAttributeKey::outputCombineAxis));
    expand->GetAttr(OpAttributeKey::outputCombineAxis, outAttr);
    EXPECT_EQ(outAttr, (std::vector<bool>{true}));
}

TEST_F(CodegenPreprocTest, TestCombineAxis3510)
{
    Platform::Instance().GetSoc().SetNPUArch(NPUArch::DAV_3510);
    ComputationalGraphBuilder graph;
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {128, 1}, MemoryType::MEM_DEVICE_DDR, "in1"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {1, 1}, MemoryType::MEM_UB, "t1"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {64, 1}, MemoryType::MEM_UB, "t2"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {64, 32}, MemoryType::MEM_UB, "t3"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {64, 32}, MemoryType::MEM_UB, "t4"), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_COPY_IN, {"in1"}, {"t1"}, "c1", true), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_EXPAND, {"t1"}, {"t2"}, "expand", true), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_COPY_IN, {"in1"}, {"t3"}, "c2", true), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_SUB, {"t2", "t3"}, {"t4"}, "sub", true), true);
    auto expand = graph.GetOp("expand");
    expand->SetAttribute(OP_ATTR_PREFIX + "EXPANDDIM", 0);

    auto funcPtr = graph.GetFunction();
    funcPtr->paramConfigs_.combineAxis = true;
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
    // Verify CodegenPreproc
    auto afterExpand = graph.GetOp("sub");
    EXPECT_EQ(afterExpand->HasAttr(OpAttributeKey::outputCombineAxis), false);
    Platform::Instance().GetSoc().SetNPUArch(NPUArch::DAV_UNKNOWN);
}

} // namespace tile_fwk
} // namespace npu
