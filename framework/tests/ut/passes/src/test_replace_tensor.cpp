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
 * \file test_replace_tensor.cpp
 * \brief Unit test for ReplaceTensor pass.
 */

#include "gtest/gtest.h"
#include "symbolic_scalar_test_utils.h"
#include "tilefwk/tilefwk_op.h"
#include "interface/function/function.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "passes/pass_mgr/pass_manager.h"
#include "interface/configs/config_manager.h"
#include "ut_json/ut_json_tool.h"
#include "passes/tile_graph_pass/graph_constraint/replace_tensor.h"
#include <fstream>
#include <vector>
#include <string>
#include "computational_graph_builder.h"
#include "interface/tensor/irbuilder.h"

namespace npu {
namespace tile_fwk {
static const uint32_t kNumZero = 0u;
static const uint32_t kNumOne = 1u;
static const uint32_t kNumTwo = 2u;
static const uint32_t kNumThree = 3u;
static const uint32_t kNumFour = 4u;
static const uint32_t kNumSix = 6u;
static const uint32_t kNumEight = 8u;
static const uint32_t kNumTwelve = 12u;
static const uint32_t kNumSixteen = 16u;

class ReplaceTensorTest : public testing::Test {
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

TEST_F(ReplaceTensorTest, TestViewAssemble)
{
    auto currFunctionPtr =
        std::make_shared<Function>(Program::GetInstance(), "TestViewAssemble", "TestViewAssemble", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);
    // Prepare the graph
    std::vector<int64_t> shape = {kNumEight, kNumEight};
    std::vector<int64_t> shape1 = {kNumEight, kNumFour};
    std::vector<int64_t> offset0 = {kNumZero, kNumZero};
    std::vector<int64_t> offset1 = {kNumZero, kNumFour};
    // init RawTensor
    std::shared_ptr<RawTensor> inRawTensor = std::make_shared<RawTensor>(DT_FP32, shape);
    std::shared_ptr<RawTensor> viewRawTensor0 = std::make_shared<RawTensor>(DT_FP32, shape1);
    std::shared_ptr<RawTensor> viewRawTensor1 = std::make_shared<RawTensor>(DT_FP32, shape1);
    std::shared_ptr<RawTensor> assRawTensor0 = std::make_shared<RawTensor>(DT_FP32, shape1);
    std::shared_ptr<RawTensor> assRawTensor1 = std::make_shared<RawTensor>(DT_FP32, shape1);
    std::shared_ptr<RawTensor> outRawTensor = std::make_shared<RawTensor>(DT_FP32, shape);
    // init LogicalTensor
    auto incast = npu::tile_fwk::IRBuilder().CreateTensorVar(inRawTensor, offset0, shape, CreateTestConstIntVector(shape));
    auto viewOut0 = npu::tile_fwk::IRBuilder().CreateTensorVar(viewRawTensor0, offset0, shape1, CreateTestConstIntVector(shape1));
    auto viewOut1 = npu::tile_fwk::IRBuilder().CreateTensorVar(viewRawTensor1, offset1, shape1, CreateTestConstIntVector(shape1));
    auto copyOut0 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape1, CreateTestConstIntVector(shape1));
    auto copyOut1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape1, CreateTestConstIntVector(shape1));
    auto assOut0 = npu::tile_fwk::IRBuilder().CreateTensorVar(assRawTensor0, offset0, shape1, CreateTestConstIntVector(shape1));
    auto assOut1 = npu::tile_fwk::IRBuilder().CreateTensorVar(assRawTensor1, offset1, shape1, CreateTestConstIntVector(shape1));
    auto outcast = npu::tile_fwk::IRBuilder().CreateTensorVar(outRawTensor, offset0, shape, CreateTestConstIntVector(shape));
    /*       Init Graph
                /————> view0 ————> copy ————> assemble \
        incast -                                        - outcast
                \————> view1 ————> copy ————> assemble /
    */
    auto& viewOp0 = currFunctionPtr->AddOperation(Opcode::OP_VIEW, {incast}, {viewOut0});
    auto& viewOp1 = currFunctionPtr->AddOperation(Opcode::OP_VIEW, {incast}, {viewOut1});
    currFunctionPtr->AddOperation(Opcode::OP_COPY_IN, {viewOut0}, {copyOut0});
    currFunctionPtr->AddOperation(Opcode::OP_COPY_IN, {viewOut1}, {copyOut1});
    currFunctionPtr->AddOperation(Opcode::OP_COPY_OUT, {copyOut0}, {assOut0});
    currFunctionPtr->AddOperation(Opcode::OP_COPY_OUT, {copyOut1}, {assOut1});
    auto& assOp0 = currFunctionPtr->AddOperation(Opcode::OP_ASSEMBLE, {assOut0}, {outcast});
    auto& assOp1 = currFunctionPtr->AddOperation(Opcode::OP_ASSEMBLE, {assOut1}, {outcast});
    // Init Attribute
    auto viewAttr0 = std::make_shared<ViewOpAttribute>(offset0);
    auto viewAttr1 = std::make_shared<ViewOpAttribute>(offset1);
    auto assAttr0 = std::make_shared<AssembleOpAttribute>(MEM_DEVICE_DDR, offset0);
    auto assAttr1 = std::make_shared<AssembleOpAttribute>(MEM_DEVICE_DDR, offset1);
    viewOp0.SetOpAttribute(viewAttr0);
    viewOp1.SetOpAttribute(viewAttr1);
    assOp0.SetOpAttribute(assAttr0);
    assOp1.SetOpAttribute(assAttr1);
    // Run the Pass
    ReplaceTensor pass;
    currFunctionPtr->inCasts_.push_back(incast);
    currFunctionPtr->outCasts_.push_back(outcast);
    EXPECT_EQ(pass.RunOnFunction(*currFunctionPtr), SUCCESS);
    EXPECT_EQ(incast->GetRawMagic(), viewOut0->GetRawMagic());
    EXPECT_EQ(incast->GetRawMagic(), viewOut1->GetRawMagic());
    EXPECT_EQ(outcast->GetRawMagic(), assOut0->GetRawMagic());
    EXPECT_EQ(outcast->GetRawMagic(), assOut1->GetRawMagic());
    EXPECT_EQ(pass.PostCheck(*currFunctionPtr), SUCCESS);
}

TEST_F(ReplaceTensorTest, TestReshape)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestReshape", "TestReshape", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);
    // Prepare the graph
    std::vector<int64_t> shape = {kNumEight, kNumEight};
    std::vector<int64_t> shape1 = {kNumOne, kNumEight, kNumEight};
    std::vector<int64_t> offset0 = {kNumZero, kNumZero};
    std::vector<int64_t> offset1 = {kNumZero, kNumZero, kNumZero};
    // init RawTensor
    std::shared_ptr<RawTensor> inRawTensor = std::make_shared<RawTensor>(DT_FP32, shape);
    std::shared_ptr<RawTensor> outRawTensor = std::make_shared<RawTensor>(DT_FP32, shape1);
    // init LogicalTensor
    auto incast = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto reshape0 = npu::tile_fwk::IRBuilder().CreateTensorVar(inRawTensor, offset0, shape, CreateTestConstIntVector(shape));
    auto reshape1 = npu::tile_fwk::IRBuilder().CreateTensorVar(outRawTensor, offset1, shape1, CreateTestConstIntVector(shape1));
    auto outcast = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape1, CreateTestConstIntVector(shape1));
    /* Init Graph
        incast -> CopyIn -> Reshape -> CopyOut -> outCast
    */
    currFunctionPtr->AddOperation(Opcode::OP_COPY_IN, {incast}, {reshape0});
    currFunctionPtr->AddOperation(Opcode::OP_RESHAPE, {reshape0}, {reshape1});
    currFunctionPtr->AddOperation(Opcode::OP_COPY_OUT, {reshape1}, {outcast});
    // Run the Pass
    ReplaceTensor pass;
    currFunctionPtr->inCasts_.push_back(incast);
    currFunctionPtr->outCasts_.push_back(outcast);
    EXPECT_EQ(pass.RunOnFunction(*currFunctionPtr), SUCCESS);
    EXPECT_EQ(reshape0->GetRawMagic(), reshape1->GetRawMagic());
    EXPECT_EQ(pass.PostCheck(*currFunctionPtr), SUCCESS);
}

TEST_F(ReplaceTensorTest, TestIndexOutCast)
{
    auto currFunctionPtr =
        std::make_shared<Function>(Program::GetInstance(), "TestIndexOutCast", "TestIndexOutCast", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);
    // Prepare the graph
    std::vector<int64_t> shape = {kNumEight, kNumEight};
    std::vector<int64_t> offset0 = {kNumZero, kNumZero};
    // init RawTensor
    std::shared_ptr<RawTensor> inRawTensor = std::make_shared<RawTensor>(DT_FP32, shape);
    std::shared_ptr<RawTensor> outRawTensor = std::make_shared<RawTensor>(DT_FP32, shape);
    // init LogicalTensor
    auto inTensor0 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto inTensor1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto inTensor2 = npu::tile_fwk::IRBuilder().CreateTensorVar(inRawTensor, offset0, shape, CreateTestConstIntVector(shape));
    auto outcast = npu::tile_fwk::IRBuilder().CreateTensorVar(outRawTensor, offset0, shape, CreateTestConstIntVector(shape));
    /* Init Graph
        incast -> Index_OutCast -> outCast
    */
    currFunctionPtr->AddOperation(Opcode::OP_INDEX_OUTCAST, {inTensor0, inTensor1, inTensor2}, {outcast});
    // Run the Pass
    ReplaceTensor pass;
    currFunctionPtr->inCasts_.push_back(inTensor2);
    currFunctionPtr->outCasts_.push_back(outcast);
    EXPECT_EQ(pass.RunOnFunction(*currFunctionPtr), SUCCESS);
    EXPECT_NE(inTensor2->GetRawMagic(), outcast->GetRawMagic());
    EXPECT_EQ(pass.PostCheck(*currFunctionPtr), SUCCESS);
}

TEST_F(ReplaceTensorTest, TestViewType)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestViewType", "TestViewType", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);
    // Prepare the graph
    std::vector<int64_t> shape = {kNumEight, kNumTwo};
    std::vector<int64_t> shape1 = {kNumOne, kNumEight};
    std::vector<int64_t> offset0 = {kNumZero, kNumZero};
    std::vector<int64_t> offset1 = {kNumZero, kNumZero};
    // init RawTensor
    std::shared_ptr<RawTensor> inRawTensor = std::make_shared<RawTensor>(DT_INT8, shape);
    std::shared_ptr<RawTensor> outRawTensor = std::make_shared<RawTensor>(DT_FP32, shape1);
    // init LogicalTensor
    auto incast = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_INT8, shape, CreateTestConstIntVector(shape));
    auto viewType0 = npu::tile_fwk::IRBuilder().CreateTensorVar(inRawTensor, offset0, shape, CreateTestConstIntVector(shape));
    auto viewType1 = npu::tile_fwk::IRBuilder().CreateTensorVar(outRawTensor, offset1, shape1, CreateTestConstIntVector(shape1));
    auto outcast = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape1, CreateTestConstIntVector(shape1));
    /* Init Graph
        incast -> CopyIn -> ViewType -> CopyOut -> outCast
    */
    currFunctionPtr->AddOperation(Opcode::OP_COPY_IN, {incast}, {viewType0});
    currFunctionPtr->AddOperation(Opcode::OP_VIEW_TYPE, {viewType0}, {viewType1});
    currFunctionPtr->AddOperation(Opcode::OP_COPY_OUT, {viewType1}, {outcast});
    // Run the Pass
    ReplaceTensor pass;
    currFunctionPtr->inCasts_.push_back(incast);
    currFunctionPtr->outCasts_.push_back(outcast);
    EXPECT_EQ(pass.RunOnFunction(*currFunctionPtr), SUCCESS);
    EXPECT_EQ(viewType0->GetRawMagic(), viewType1->GetRawMagic());
    EXPECT_EQ(pass.PostCheck(*currFunctionPtr), SUCCESS);
}

TEST_F(ReplaceTensorTest, TestHasSameConsecutive_True)
{
    auto currFunctionPtr = std::make_shared<Function>(
        Program::GetInstance(), "TestHasSameConsecutive_True", "TestHasSameConsecutive_True", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    std::vector<int64_t> shape = {kNumEight, kNumEight};

    auto tensor1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto tensor2 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto tensor3 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));

    auto& viewOp1 = currFunctionPtr->AddOperation(Opcode::OP_VIEW, {tensor1}, {tensor2});
    auto& viewOp2 = currFunctionPtr->AddOperation(Opcode::OP_VIEW, {tensor2}, {tensor3});

    // 设置操作连接
    tensor2->AddConsumer(&viewOp2);

    ReplaceTensor pass;
    bool result = pass.HasSameConsecutive(viewOp1);
    EXPECT_TRUE(result);
}

TEST_F(ReplaceTensorTest, TestHasSameConsecutive_False)
{
    auto currFunctionPtr = std::make_shared<Function>(
        Program::GetInstance(), "TestHasSameConsecutive_False", "TestHasSameConsecutive_False", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    std::vector<int64_t> shape = {kNumEight, kNumEight};

    auto tensor1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto tensor2 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto tensor3 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));

    auto& viewOp1 = currFunctionPtr->AddOperation(Opcode::OP_VIEW, {tensor1}, {tensor2});
    auto& assembleOp = currFunctionPtr->AddOperation(Opcode::OP_ASSEMBLE, {tensor2}, {tensor3});

    // 设置操作连接
    tensor2->AddConsumer(&assembleOp);

    ReplaceTensor pass;
    bool result = pass.HasSameConsecutive(viewOp1);
    EXPECT_FALSE(result);
}

TEST_F(ReplaceTensorTest, TestPreCheck_FailNoSubgraphID)
{
    auto currFunctionPtr = std::make_shared<Function>(
        Program::GetInstance(), "TestPreCheck_FailNoSubgraphID", "TestPreCheck_FailNoSubgraphID", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    std::vector<int64_t> shape = {kNumEight, kNumEight};

    auto tensor1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto tensor2 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));

    currFunctionPtr->AddOperation(Opcode::OP_VIEW, {tensor1}, {tensor2});
    // 不设置subgraph ID

    ReplaceTensor pass;
    Status result = pass.PreCheck(*currFunctionPtr);
    EXPECT_EQ(result, FAILED);
}

/*
            /————> copy ————> view ————> viewtype ————> assemble \
    incast -                                                      - outcast
            \————> copy ————> view ————> viewtype ————> assemble /
*/
TEST_F(ReplaceTensorTest, TestBackView)
{
    auto currFunctionPtr =
        std::make_shared<Function>(Program::GetInstance(), "TestViewAssemble", "TestViewAssemble", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);
    // Prepare the graph
    std::vector<int64_t> shape = {kNumEight, kNumEight};
    std::vector<int64_t> shape1 = {kNumEight, kNumFour};
    std::vector<int64_t> offset0 = {kNumZero, kNumZero};
    std::vector<int64_t> offset1 = {kNumZero, kNumFour};
    // init RawTensor
    std::shared_ptr<RawTensor> outRawTensor = std::make_shared<RawTensor>(DT_FP32, shape);
    std::shared_ptr<RawTensor> viewRawTensor0 = std::make_shared<RawTensor>(DT_FP32, shape1);
    std::shared_ptr<RawTensor> viewRawTensor1 = std::make_shared<RawTensor>(DT_FP32, shape1);
    std::shared_ptr<RawTensor> viewTypeRaw0 = std::make_shared<RawTensor>(DT_FP32, shape1);
    std::shared_ptr<RawTensor> viewTypeRaw1 = std::make_shared<RawTensor>(DT_FP32, shape1);
    std::shared_ptr<RawTensor> assRawTensor0 = std::make_shared<RawTensor>(DT_FP32, shape1);
    std::shared_ptr<RawTensor> assRawTensor1 = std::make_shared<RawTensor>(DT_FP32, shape1);
    // init LogicalTensor
    auto incast = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto copy0 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape1, CreateTestConstIntVector(shape1));
    auto copy1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape1, CreateTestConstIntVector(shape1));
    auto viewIn0 = npu::tile_fwk::IRBuilder().CreateTensorVar(viewRawTensor0, offset0, shape1, CreateTestConstIntVector(shape1));
    auto viewIn1 = npu::tile_fwk::IRBuilder().CreateTensorVar(viewRawTensor1, offset0, shape1, CreateTestConstIntVector(shape1));
    auto viewTypeIn0 = npu::tile_fwk::IRBuilder().CreateTensorVar(viewTypeRaw0, offset0, shape1, CreateTestConstIntVector(shape1));
    auto viewTypeIn1 = npu::tile_fwk::IRBuilder().CreateTensorVar(viewTypeRaw1, offset0, shape1, CreateTestConstIntVector(shape1));
    auto assIn0 = npu::tile_fwk::IRBuilder().CreateTensorVar(assRawTensor0, offset0, shape1, CreateTestConstIntVector(shape1));
    auto assIn1 = npu::tile_fwk::IRBuilder().CreateTensorVar(assRawTensor1, offset1, shape1, CreateTestConstIntVector(shape1));
    auto outcast = npu::tile_fwk::IRBuilder().CreateTensorVar(outRawTensor, offset0, shape, CreateTestConstIntVector(shape));
    // Init Graph
    currFunctionPtr->AddOperation(Opcode::OP_COPY_IN, {incast}, {copy0});
    currFunctionPtr->AddOperation(Opcode::OP_COPY_IN, {incast}, {copy1});
    currFunctionPtr->AddOperation(Opcode::OP_COPY_OUT, {copy0}, {viewIn0});
    currFunctionPtr->AddOperation(Opcode::OP_COPY_OUT, {copy1}, {viewIn1});
    auto& viewOp0 = currFunctionPtr->AddOperation(Opcode::OP_VIEW, {viewIn0}, {viewTypeIn0});
    auto& viewOp1 = currFunctionPtr->AddOperation(Opcode::OP_VIEW, {viewIn1}, {viewTypeIn1});
    currFunctionPtr->AddOperation(Opcode::OP_VIEW_TYPE, {viewTypeIn0}, {assIn0});
    currFunctionPtr->AddOperation(Opcode::OP_VIEW_TYPE, {viewTypeIn1}, {assIn1});
    auto& assOp0 = currFunctionPtr->AddOperation(Opcode::OP_ASSEMBLE, {assIn0}, {outcast});
    auto& assOp1 = currFunctionPtr->AddOperation(Opcode::OP_ASSEMBLE, {assIn1}, {outcast});
    // Init Attribute
    auto view_Attr0 = std::make_shared<ViewOpAttribute>(offset0);
    auto view_Attr1 = std::make_shared<ViewOpAttribute>(offset1);
    auto ass_Attr0 = std::make_shared<AssembleOpAttribute>(MEM_DEVICE_DDR, offset0);
    auto ass_Attr1 = std::make_shared<AssembleOpAttribute>(MEM_DEVICE_DDR, offset1);
    viewOp0.SetOpAttribute(view_Attr0);
    viewOp1.SetOpAttribute(view_Attr1);
    assOp0.SetOpAttribute(ass_Attr0);
    assOp1.SetOpAttribute(ass_Attr1);
    // Run the Pass
    ReplaceTensor pass;
    currFunctionPtr->inCasts_.push_back(incast);
    currFunctionPtr->outCasts_.push_back(outcast);
    EXPECT_EQ(pass.RunOnFunction(*currFunctionPtr), SUCCESS);
    EXPECT_EQ(outcast->GetRawMagic(), assIn0->GetRawMagic());
    EXPECT_EQ(outcast->GetRawMagic(), assIn1->GetRawMagic());
    EXPECT_EQ(outcast->GetRawMagic(), viewTypeIn0->GetRawMagic());
    EXPECT_EQ(outcast->GetRawMagic(), viewTypeIn1->GetRawMagic());
    EXPECT_EQ(outcast->GetRawMagic(), viewIn0->GetRawMagic());
    EXPECT_EQ(outcast->GetRawMagic(), viewIn1->GetRawMagic());
    EXPECT_EQ(pass.PostCheck(*currFunctionPtr), SUCCESS);
}

TEST_F(ReplaceTensorTest, TestProcessHubAssembleOp_Success)
{
    auto currFunctionPtr = std::make_shared<Function>(
        Program::GetInstance(), "TestProcessHubAssembleOp_Success", "TestProcessHubAssembleOp_Success", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    // 准备HUB-ASSEMBLE-OUTCAST链
    std::vector<int64_t> shape = {kNumEight, kNumEight};
    std::vector<int64_t> offset = {kNumZero, kNumZero};

    // 创建共享的raw tensor
    std::shared_ptr<RawTensor> rawTensor = std::make_shared<RawTensor>(DT_FP32, shape);

    // 创建HUB操作相关张量
    auto hubInput = npu::tile_fwk::IRBuilder().CreateTensorVar(rawTensor, offset, shape, CreateTestConstIntVector(shape));
    auto hubOutput = npu::tile_fwk::IRBuilder().CreateTensorVar(rawTensor, offset, shape, CreateTestConstIntVector(shape));

    // 创建ASSEMBLE操作相关张量
    auto assembleOutput = npu::tile_fwk::IRBuilder().CreateTensorVar(rawTensor, offset, shape, CreateTestConstIntVector(shape));

    // 设置内存类型
    hubInput->SetMemoryTypeOriginal(MEM_DEVICE_DDR, true);
    hubOutput->SetMemoryTypeOriginal(MEM_DEVICE_DDR, true);
    assembleOutput->SetMemoryTypeOriginal(MEM_DEVICE_DDR, true);

    // 创建操作
    auto& hubOp = currFunctionPtr->AddOperation(Opcode::OP_HUB, {hubInput}, {hubOutput});
    auto& assembleOp = currFunctionPtr->AddOperation(Opcode::OP_ASSEMBLE, {hubOutput}, {assembleOutput});

    // 设置操作连接
    hubOutput->AddConsumer(&assembleOp);

    // 设置ASSEMBLE属性
    auto assembleAttr = std::make_shared<AssembleOpAttribute>(MEM_DEVICE_DDR, offset);
    assembleOp.SetOpAttribute(assembleAttr);

    // 将assembleOutput设置为outcast
    currFunctionPtr->outCasts_.push_back(assembleOutput);

    ReplaceTensor pass;
    pass.ProcessHubAssembleOp(*currFunctionPtr, hubOp, assembleOp, hubInput, hubOutput);

    // 验证hubInput和hubOutput共享了assembleOutput的tensor
    EXPECT_EQ(hubInput->GetRawTensor(), assembleOutput->GetRawTensor());
    EXPECT_EQ(hubOutput->GetRawTensor(), assembleOutput->GetRawTensor());
}

TEST_F(ReplaceTensorTest, TestA_MULACC_B)
{
    auto currFunctionPtr = std::make_shared<Function>(
        Program::GetInstance(), "TestProcessHubAssembleOp_BrokenChain", "TestProcessHubAssembleOp_BrokenChain",
        nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);
    // Prepare the graph
    std::vector<int64_t> mulAccshape = {kNumEight, kNumEight};
    std::vector<int64_t> offset0 = {kNumZero, kNumZero};
    // init RawTensor
    std::shared_ptr<RawTensor> inRawTensor = std::make_shared<RawTensor>(DT_FP32, mulAccshape);
    std::shared_ptr<RawTensor> outRawTensor = std::make_shared<RawTensor>(DT_FP32, mulAccshape);
    // init LogicalTensor
    auto inTensor0 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, mulAccshape, CreateTestConstIntVector(mulAccshape));
    auto inTensor1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, mulAccshape, CreateTestConstIntVector(mulAccshape));
    auto mulAccIn = npu::tile_fwk::IRBuilder().CreateTensorVar(inRawTensor, offset0, mulAccshape, CreateTestConstIntVector(mulAccshape));
    auto mulAccOut = npu::tile_fwk::IRBuilder().CreateTensorVar(outRawTensor, offset0, mulAccshape, CreateTestConstIntVector(mulAccshape));
    auto outTensor = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, mulAccshape, CreateTestConstIntVector(mulAccshape));
    /* Init Graph
        incast -> Index_OutCast -> mulAccOut-> op
    */
    currFunctionPtr->AddOperation(Opcode::OP_A_MULACC_B, {inTensor0, inTensor1, mulAccIn}, {mulAccOut});
    currFunctionPtr->AddOperation(Opcode::OP_COPY_OUT, {mulAccOut}, {outTensor});
    ReplaceTensor pass;
    currFunctionPtr->inCasts_.push_back(mulAccIn);
    currFunctionPtr->outCasts_.push_back(outTensor);
    EXPECT_EQ(pass.RunOnFunction(*currFunctionPtr), SUCCESS);
    EXPECT_EQ(mulAccIn->GetRawMagic(), mulAccOut->GetRawMagic());
    EXPECT_EQ(pass.PostCheck(*currFunctionPtr), SUCCESS);
}

TEST_F(ReplaceTensorTest, TestSameAssembleOut)
{
    auto currFunctionPtr =
        std::make_shared<Function>(Program::GetInstance(), "TestSameAssembleOut", "TestSameAssembleOut", nullptr);
    EXPECT_NE(currFunctionPtr, nullptr);
    // Prepare the graph
    std::vector<int64_t> shape = {kNumEight, kNumEight};
    std::vector<int64_t> shape1 = {kNumEight, kNumFour};
    std::vector<int64_t> offset0 = {kNumZero, kNumZero};
    std::vector<int64_t> offset1 = {kNumZero, kNumFour};
    // init RawTensor
    std::shared_ptr<RawTensor> inRawTensor = std::make_shared<RawTensor>(DT_FP32, shape);
    std::shared_ptr<RawTensor> copyInRawTensor = std::make_shared<RawTensor>(DT_FP32, shape1);
    std::shared_ptr<RawTensor> outRawTensor0 = std::make_shared<RawTensor>(DT_FP32, shape);
    std::shared_ptr<RawTensor> outRawTensor1 = std::make_shared<RawTensor>(DT_FP32, shape);
    // init LogicalTensor
    auto incast = npu::tile_fwk::IRBuilder().CreateTensorVar(inRawTensor, offset0, shape, CreateTestConstIntVector(shape));
    incast->SetMemoryTypeBoth(MEM_DEVICE_DDR, true);
    auto copyInOut = npu::tile_fwk::IRBuilder().CreateTensorVar(copyInRawTensor, offset0, shape1, CreateTestConstIntVector(shape1));
    copyInOut->SetMemoryTypeBoth(MEM_UB, true);
    auto outcast0 = npu::tile_fwk::IRBuilder().CreateTensorVar(outRawTensor0, offset0, shape, CreateTestConstIntVector(shape));
    outcast0->SetMemoryTypeBoth(MEM_DEVICE_DDR, true);
    auto outcast1 = npu::tile_fwk::IRBuilder().CreateTensorVar(outRawTensor1, offset0, shape1, CreateTestConstIntVector(shape1));
    outcast1->SetMemoryTypeBoth(MEM_DEVICE_DDR, true);
    /*       Init Graph
                            /—————> assemble -outcast1
                             /————> assemble \
        incast ————> copyIn -                 - outcast0
                             \————> assemble /
    */
    auto& copyInOp = currFunctionPtr->AddOperation(Opcode::OP_COPY_IN, {incast}, {copyInOut});
    auto& assOp0 = currFunctionPtr->AddOperation(Opcode::OP_ASSEMBLE, {copyInOut}, {outcast0});
    auto& assOp1 = currFunctionPtr->AddOperation(Opcode::OP_ASSEMBLE, {copyInOut}, {outcast0});
    auto& assOp2 = currFunctionPtr->AddOperation(Opcode::OP_ASSEMBLE, {copyInOut}, {outcast1});
    // Init Attribute
    auto copyInAttr = std::make_shared<CopyOpAttribute>(
        OpImmediate::Specified(offset0), MEM_UB, OpImmediate::Specified(shape), OpImmediate::Specified(shape));
    auto assAttr0 = std::make_shared<AssembleOpAttribute>(MEM_UB, offset0);
    auto assAttr1 = std::make_shared<AssembleOpAttribute>(MEM_UB, offset1);
    auto assAttr2 = std::make_shared<AssembleOpAttribute>(MEM_UB, offset0);
    copyInOp.SetOpAttribute(copyInAttr);
    assOp0.SetOpAttribute(assAttr0);
    assOp1.SetOpAttribute(assAttr1);
    assOp2.SetOpAttribute(assAttr2);
    // Run the Pass
    ReplaceTensor pass;
    currFunctionPtr->inCasts_.push_back(incast);
    currFunctionPtr->outCasts_.push_back(outcast0);
    currFunctionPtr->outCasts_.push_back(outcast1);
    int opSumBefore = currFunctionPtr->Operations().size();
    EXPECT_EQ(pass.RunOnFunction(*currFunctionPtr), SUCCESS);
    EXPECT_EQ(currFunctionPtr->Operations().size(), opSumBefore + 6);
    EXPECT_EQ(pass.PostCheck(*currFunctionPtr), SUCCESS);
}

TEST_F(ReplaceTensorTest, TestNotInplaceReshape)
{
    auto currFunctionPtr =
        std::make_shared<Function>(Program::GetInstance(), "TestNotInplaceReshape", "TestNotInplaceReshape", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);
    // Prepare the graph
    std::vector<int64_t> shape = {kNumFour, kNumEight};
    std::vector<int64_t> shape1 = {kNumEight, kNumFour};
    std::vector<int64_t> shape2 = {kNumFour, kNumFour};
    std::vector<int64_t> shape3 = {kNumOne, kNumFour, kNumFour};
    std::vector<int64_t> offset0 = {kNumZero, kNumZero};
    // init RawTensor
    std::shared_ptr<RawTensor> reshapeRawTensor0 = std::make_shared<RawTensor>(DT_FP32, shape);
    std::shared_ptr<RawTensor> viewRawTensor = std::make_shared<RawTensor>(DT_FP32, shape);
    // init LogicalTensor
    auto incast = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto reshapeOut0 = npu::tile_fwk::IRBuilder().CreateTensorVar(reshapeRawTensor0, offset0, shape1, CreateTestConstIntVector(shape1));
    auto viewOut = npu::tile_fwk::IRBuilder().CreateTensorVar(viewRawTensor, offset0, shape2, CreateTestConstIntVector(shape2));
    auto outcast = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape3, CreateTestConstIntVector(shape3));
    /* Init Graph
        incast0 -> Reshape -> View -> Reshape -> outcast
    */
    currFunctionPtr->AddOperation(Opcode::OP_RESHAPE, {incast}, {reshapeOut0});
    auto& viewOp = currFunctionPtr->AddOperation(Opcode::OP_VIEW, {reshapeOut0}, {viewOut});
    currFunctionPtr->AddOperation(Opcode::OP_RESHAPE, {viewOut}, {outcast});

    auto view_Attr = std::make_shared<ViewOpAttribute>(offset0, MEM_DEVICE_DDR);
    viewOp.SetOpAttribute(view_Attr);
    // Run the Pass
    ReplaceTensor pass;
    currFunctionPtr->inCasts_.push_back(incast);
    currFunctionPtr->outCasts_.push_back(outcast);
    EXPECT_EQ(pass.RunOnFunction(*currFunctionPtr), SUCCESS);
    EXPECT_NE(viewOut->GetRawMagic(), outcast->GetRawMagic());
    EXPECT_EQ(pass.PostCheck(*currFunctionPtr), SUCCESS);
}

TEST_F(ReplaceTensorTest, UpdateCopyInAttrAfterBackAssemble)
{
    auto currFunctionPtr = std::make_shared<Function>(
        Program::GetInstance(), "UpdateCopyInAttrAfterBackAssemble", "UpdateCopyInAttrAfterBackAssemble", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    // Prepare the graph
    std::vector<int64_t> inshape = {4, 4};
    std::vector<int64_t> outshape1 = {8, 4};
    std::vector<int64_t> outshape2 = {2, 4};

    auto incast = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, inshape, CreateTestConstIntVector(inshape));
    incast->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    auto copyInout1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, inshape, CreateTestConstIntVector(inshape));
    copyInout1->SetMemoryTypeBoth(MemoryType::MEM_UB, true);
    auto copyOutout1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, inshape, CreateTestConstIntVector(inshape));
    copyOutout1->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    auto copyInout2 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, inshape, CreateTestConstIntVector(inshape));
    copyInout2->SetMemoryTypeBoth(MemoryType::MEM_UB, true);
    auto outcast1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, outshape1, CreateTestConstIntVector(outshape1));
    outcast1->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    auto outcast2 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, outshape2, CreateTestConstIntVector(outshape2));
    outcast2->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    /* Init Graph
        incast -- CopyIn -- copyInout1 -- CopyOut -- copyOutOut1 -- Assemble -- outcast1
                                                                 -- CopyIn   -- copyInout2 -- CopyOut -- outcast2
    */
    currFunctionPtr->AddOperation(Opcode::OP_COPY_IN, {incast}, {copyInout1});
    currFunctionPtr->AddOperation(Opcode::OP_COPY_OUT, {copyInout1}, {copyOutout1});
    auto& assembleOp = currFunctionPtr->AddOperation(Opcode::OP_ASSEMBLE, {copyOutout1}, {outcast1});
    auto& copyInOp = currFunctionPtr->AddOperation(Opcode::OP_COPY_IN, {copyOutout1}, {copyInout2});
    currFunctionPtr->AddOperation(Opcode::OP_COPY_OUT, {copyInout2}, {outcast2});

    Offset assembleToOffset = {4, 0};
    auto assembleOpAttribute =
        std::make_shared<AssembleOpAttribute>(assembleToOffset, CreateTestConstIntVector(assembleToOffset));
    assembleOp.SetOpAttribute(assembleOpAttribute);
    Offset copyIn2FromOffset = {2, 0};
    auto copyInOpAttribute = std::make_shared<CopyOpAttribute>(
        OpImmediate::Specified(copyIn2FromOffset), MEM_UB, OpImmediate::Specified(copyInout2->GetShape()),
        OpImmediate::Specified(copyInout2->tensor->GetDynRawShape()),
        OpImmediate::Specified(copyInout2->GetDynValidShape()));
    copyInOp.SetOpAttribute(copyInOpAttribute);

    currFunctionPtr->inCasts_.push_back(incast);
    currFunctionPtr->outCasts_.push_back(outcast1);
    currFunctionPtr->outCasts_.push_back(outcast2);

    ReplaceTensor replaceTensorPass;
    int opSumBefore = currFunctionPtr->Operations().size();
    replaceTensorPass.RunOnFunction(*currFunctionPtr);
    int opSumExpAfter = opSumBefore + 2;
    EXPECT_EQ(replaceTensorPass.PostCheck(*currFunctionPtr), SUCCESS);
    EXPECT_EQ(currFunctionPtr->Operations().size(), opSumExpAfter);
}

/*
 * 验证场景：SHMEM_WAIT_UNTIL 的输出 tensor 被多个 ASSEMBLE 消费且输出到不同 outcast 时，
 * ReplaceTensor 不插入 COPY_IN/COPY_OUT。
 */
TEST_F(ReplaceTensorTest, TestShmemWaitUntilWithDiffAssembleOut)
{
    auto currFunctionPtr = std::make_shared<Function>(
        Program::GetInstance(), "TestShmemWaitUntilAssemble", "TestShmemWaitUntilAssemble", nullptr);
    EXPECT_NE(currFunctionPtr, nullptr);
    std::vector<int64_t> shape = {kNumEight, kNumEight};
    std::vector<int64_t> shape1 = {kNumEight, kNumFour};
    std::vector<int64_t> offset0 = {kNumZero, kNumZero};
    std::vector<int64_t> offset1 = {kNumZero, kNumFour};
    std::shared_ptr<RawTensor> inRawTensor0 = std::make_shared<RawTensor>(DT_FP32, shape);
    std::shared_ptr<RawTensor> inRawTensor1 = std::make_shared<RawTensor>(DT_FP32, shape);
    std::shared_ptr<RawTensor> shmemRawTensor = std::make_shared<RawTensor>(DT_FP32, shape);
    std::shared_ptr<RawTensor> outRawTensor0 = std::make_shared<RawTensor>(DT_FP32, shape);
    std::shared_ptr<RawTensor> outRawTensor1 = std::make_shared<RawTensor>(DT_FP32, shape);
    auto incast0 = std::make_shared<LogicalTensor>(*currFunctionPtr, inRawTensor0, offset0, shape);
    incast0->SetMemoryTypeBoth(MEM_DEVICE_DDR, true);
    auto incast1 = std::make_shared<LogicalTensor>(*currFunctionPtr, inRawTensor1, offset0, shape);
    incast1->SetMemoryTypeBoth(MEM_DEVICE_DDR, true);
    auto shmemOut = std::make_shared<LogicalTensor>(*currFunctionPtr, shmemRawTensor, offset0, shape);
    shmemOut->SetMemoryTypeBoth(MEM_DEVICE_DDR, true);
    auto outcast0 = std::make_shared<LogicalTensor>(*currFunctionPtr, outRawTensor0, offset0, shape);
    outcast0->SetMemoryTypeBoth(MEM_DEVICE_DDR, true);
    auto outcast1 = std::make_shared<LogicalTensor>(*currFunctionPtr, outRawTensor1, offset1, shape1);
    outcast1->SetMemoryTypeBoth(MEM_DEVICE_DDR, true);
    currFunctionPtr->AddOperation(Opcode::OP_SHMEM_WAIT_UNTIL, {incast0, incast1}, {shmemOut});
    auto& assOp0 = currFunctionPtr->AddOperation(Opcode::OP_ASSEMBLE, {shmemOut}, {outcast0});
    auto& assOp1 = currFunctionPtr->AddOperation(Opcode::OP_ASSEMBLE, {shmemOut}, {outcast1});
    auto assAttr0 = std::make_shared<AssembleOpAttribute>(MEM_DEVICE_DDR, offset0);
    auto assAttr1 = std::make_shared<AssembleOpAttribute>(MEM_DEVICE_DDR, offset1);
    assOp0.SetOpAttribute(assAttr0);
    assOp1.SetOpAttribute(assAttr1);
    currFunctionPtr->inCasts_.push_back(incast0);
    currFunctionPtr->inCasts_.push_back(incast1);
    currFunctionPtr->outCasts_.push_back(outcast0);
    currFunctionPtr->outCasts_.push_back(outcast1);
    ReplaceTensor pass;
    int opSumBefore = currFunctionPtr->Operations().size();
    EXPECT_EQ(pass.RunOnFunction(*currFunctionPtr), SUCCESS);
    EXPECT_EQ(currFunctionPtr->Operations().size(), opSumBefore);
    for (auto& op : currFunctionPtr->Operations()) {
        EXPECT_TRUE(op.GetOpcode() != Opcode::OP_COPY_IN && op.GetOpcode() != Opcode::OP_COPY_OUT);
    }
    EXPECT_EQ(shmemOut->GetRawMagic(), outcast0->GetRawMagic());
    EXPECT_EQ(shmemOut->GetRawMagic(), outcast1->GetRawMagic());
    EXPECT_EQ(pass.PostCheck(*currFunctionPtr), SUCCESS);
}
} // namespace tile_fwk
} // namespace npu
