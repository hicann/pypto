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

namespace npu {
namespace tile_fwk {
static const uint32_t kNumZero = 0u;
static const uint32_t kNumOne = 1u;
static const uint32_t kNumTwo = 2u;
static const uint32_t kNumThree = 3u;
static const uint32_t kNumFour = 4u;
static const uint32_t kNumSix = 6u;
static const uint32_t kNumEight = 8u;
static const uint32_t kNumSixteen = 16u;

class ReplaceTensorTest : public testing::Test {
public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}

    void SetUp() override {
        Program::GetInstance().Reset();
        config::Reset();
    }

    void TearDown() override {}
};

TEST_F(ReplaceTensorTest, TestViewAssemble) {
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestViewAssemble", "TestViewAssemble", nullptr);
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
    auto incast = std::make_shared<LogicalTensor>(*currFunctionPtr, inRawTensor, offset0, shape);
    auto viewOut0 = std::make_shared<LogicalTensor>(*currFunctionPtr, viewRawTensor0, offset0, shape1);
    auto viewOut1 = std::make_shared<LogicalTensor>(*currFunctionPtr, viewRawTensor1, offset1, shape1);
    auto copyOut0 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape1);
    auto copyOut1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape1);
    auto assOut0 = std::make_shared<LogicalTensor>(*currFunctionPtr, assRawTensor0, offset0, shape1);
    auto assOut1 = std::make_shared<LogicalTensor>(*currFunctionPtr, assRawTensor1, offset1, shape1);
    auto outcast = std::make_shared<LogicalTensor>(*currFunctionPtr, outRawTensor, offset0, shape);
    /*       Init Graph
                /————> view0 ————> copy ————> assemble \
        incast -                                        - outcast
                \————> view1 ————> copy ————> assemble /
    */
    auto &viewOp0 = currFunctionPtr->AddOperation(Opcode::OP_VIEW, {incast}, {viewOut0});
    auto &viewOp1 = currFunctionPtr->AddOperation(Opcode::OP_VIEW, {incast}, {viewOut1});
    currFunctionPtr->AddOperation(Opcode::OP_COPY_IN, {viewOut0}, {copyOut0});
    currFunctionPtr->AddOperation(Opcode::OP_COPY_IN, {viewOut1}, {copyOut1});
    currFunctionPtr->AddOperation(Opcode::OP_COPY_OUT, {copyOut0}, {assOut0});
    currFunctionPtr->AddOperation(Opcode::OP_COPY_OUT, {copyOut1}, {assOut1});
    auto &assOp0 = currFunctionPtr->AddOperation(Opcode::OP_ASSEMBLE, {assOut0}, {outcast});
    auto &assOp1 = currFunctionPtr->AddOperation(Opcode::OP_ASSEMBLE, {assOut1}, {outcast});
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

TEST_F(ReplaceTensorTest, TestReshape) {
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
    auto incast = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    auto reshape0 = std::make_shared<LogicalTensor>(*currFunctionPtr, inRawTensor, offset0, shape);
    auto reshape1 = std::make_shared<LogicalTensor>(*currFunctionPtr, outRawTensor, offset1, shape1);
    auto outcast = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape1);
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

TEST_F(ReplaceTensorTest, TestIndexOutCast) {
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestIndexOutCast", "TestIndexOutCast", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);
    // Prepare the graph
    std::vector<int64_t> shape = {kNumEight, kNumEight};
    std::vector<int64_t> offset0 = {kNumZero, kNumZero};
    // init RawTensor
    std::shared_ptr<RawTensor> inRawTensor = std::make_shared<RawTensor>(DT_FP32, shape);
    std::shared_ptr<RawTensor> outRawTensor = std::make_shared<RawTensor>(DT_FP32, shape);
    // init LogicalTensor
    auto inTensor0 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    auto inTensor1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    auto inTensor2 = std::make_shared<LogicalTensor>(*currFunctionPtr, inRawTensor, offset0, shape);
    auto outcast = std::make_shared<LogicalTensor>(*currFunctionPtr, outRawTensor, offset0, shape);
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

TEST_F(ReplaceTensorTest, TestViewType) {
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
    auto incast = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_INT8, shape);
    auto viewType0 = std::make_shared<LogicalTensor>(*currFunctionPtr, inRawTensor, offset0, shape);
    auto viewType1 = std::make_shared<LogicalTensor>(*currFunctionPtr, outRawTensor, offset1, shape1);
    auto outcast = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape1);
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

TEST_F(ReplaceTensorTest, TestComplexMultiBranch) {
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestComplexMultiBranch", "TestComplexMultiBranch", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);
    
    // 准备复杂的多分支图结构
    std::vector<int64_t> shape = {kNumSixteen, kNumSixteen};
    std::vector<int64_t> shapeHalf = {kNumSixteen, kNumEight};
    std::vector<int64_t> offset0 = {kNumZero, kNumZero};
    std::vector<int64_t> offset1 = {kNumZero, kNumEight};
    
    // 初始化RawTensor
    std::shared_ptr<RawTensor> inRawTensor = std::make_shared<RawTensor>(DT_FP32, shape);
    std::shared_ptr<RawTensor> outRawTensor = std::make_shared<RawTensor>(DT_FP32, shape);
    
    auto incast = std::make_shared<LogicalTensor>(*currFunctionPtr, inRawTensor, offset0, shape);
    auto outcast = std::make_shared<LogicalTensor>(*currFunctionPtr, outRawTensor, offset0, shape);
    
    // 创建多个分支，每个分支有不同的操作链
    std::vector<std::shared_ptr<LogicalTensor>> branchOutputs;
    
    // 分支1: VIEW -> VIEW_TYPE -> RESHAPE -> ASSEMBLE
    {
        auto viewOut = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shapeHalf);
        auto viewTypeOut = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP16, shapeHalf);
        auto reshapeOut = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP16, shapeHalf);
        auto assembleOut = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP16, shapeHalf);
        
        auto &viewOp = currFunctionPtr->AddOperation(Opcode::OP_VIEW, {incast}, {viewOut});
        auto &viewTypeOp = currFunctionPtr->AddOperation(Opcode::OP_VIEW_TYPE, {viewOut}, {viewTypeOut});
        (void) viewTypeOp;
        auto &reshapeOp = currFunctionPtr->AddOperation(Opcode::OP_RESHAPE, {viewTypeOut}, {reshapeOut});
        auto &assembleOp = currFunctionPtr->AddOperation(Opcode::OP_ASSEMBLE, {assembleOut}, {outcast});
        
        auto viewAttr = std::make_shared<ViewOpAttribute>(offset0);
        auto assembleAttr = std::make_shared<AssembleOpAttribute>(MEM_DEVICE_DDR, offset0);
        viewOp.SetOpAttribute(viewAttr);
        assembleOp.SetOpAttribute(assembleAttr);
        reshapeOp.SetAttribute(OP_ATTR_PREFIX + "isInplace", true);
    }
    
    // 分支2: VIEW -> INDEX_OUTCAST -> ASSEMBLE
    {
        auto viewOut = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shapeHalf);
        auto indexOut = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shapeHalf);
        auto helperTensor = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shapeHalf);
        auto assembleOut = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shapeHalf);
        
        auto &viewOp = currFunctionPtr->AddOperation(Opcode::OP_VIEW, {incast}, {viewOut});
        auto &indexOp = currFunctionPtr->AddOperation(Opcode::OP_INDEX_OUTCAST, {incast, helperTensor, indexOut}, {indexOut});
        (void) indexOp;
        auto &assembleOp = currFunctionPtr->AddOperation(Opcode::OP_ASSEMBLE, {assembleOut}, {outcast});
        
        auto viewAttr = std::make_shared<ViewOpAttribute>(offset1);
        auto assembleAttr = std::make_shared<AssembleOpAttribute>(MEM_DEVICE_DDR, offset1);
        viewOp.SetOpAttribute(viewAttr);
        assembleOp.SetOpAttribute(assembleAttr);
    }
    
    // 运行Pass
    ReplaceTensor pass;
    currFunctionPtr->inCasts_.push_back(incast);
    currFunctionPtr->outCasts_.push_back(outcast);
    
    EXPECT_EQ(pass.RunOnFunction(*currFunctionPtr), SUCCESS);
    EXPECT_EQ(pass.PostCheck(*currFunctionPtr), SUCCESS);
}

TEST_F(ReplaceTensorTest, TestHasSameConsecutive_True) {
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestHasSameConsecutive_True", 
                                                     "TestHasSameConsecutive_True", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);
    
    std::vector<int64_t> shape = {kNumEight, kNumEight};
    
    auto tensor1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    auto tensor2 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    auto tensor3 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    
    auto &viewOp1 = currFunctionPtr->AddOperation(Opcode::OP_VIEW, {tensor1}, {tensor2});
    auto &viewOp2 = currFunctionPtr->AddOperation(Opcode::OP_VIEW, {tensor2}, {tensor3});
    
    // 设置操作连接
    tensor2->AddConsumer(&viewOp2);
    
    ReplaceTensor pass;
    bool result = pass.HasSameConsecutive(viewOp1);
    EXPECT_TRUE(result);
}

TEST_F(ReplaceTensorTest, TestHasSameConsecutive_False) {
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestHasSameConsecutive_False", 
                                                     "TestHasSameConsecutive_False", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);
    
    std::vector<int64_t> shape = {kNumEight, kNumEight};
    
    auto tensor1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    auto tensor2 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    auto tensor3 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    
    auto &viewOp1 = currFunctionPtr->AddOperation(Opcode::OP_VIEW, {tensor1}, {tensor2});
    auto &assembleOp = currFunctionPtr->AddOperation(Opcode::OP_ASSEMBLE, {tensor2}, {tensor3});
    
    // 设置操作连接
    tensor2->AddConsumer(&assembleOp);
    
    ReplaceTensor pass;
    bool result = pass.HasSameConsecutive(viewOp1);
    EXPECT_FALSE(result);
}

TEST_F(ReplaceTensorTest, TestPreCheck_FailNoSubgraphID) {
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestPreCheck_FailNoSubgraphID", 
                                                     "TestPreCheck_FailNoSubgraphID", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);
    
    std::vector<int64_t> shape = {kNumEight, kNumEight};
    
    auto tensor1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    auto tensor2 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    
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
TEST_F(ReplaceTensorTest, TestBackView) {
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestViewAssemble", "TestViewAssemble", nullptr);
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
    auto incast = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    auto copy0 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape1);
    auto copy1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape1);
    auto viewIn0 = std::make_shared<LogicalTensor>(*currFunctionPtr, viewRawTensor0, offset0, shape1);
    auto viewIn1 = std::make_shared<LogicalTensor>(*currFunctionPtr, viewRawTensor1, offset0, shape1);
    auto viewTypeIn0 = std::make_shared<LogicalTensor>(*currFunctionPtr, viewTypeRaw0, offset0, shape1);
    auto viewTypeIn1 = std::make_shared<LogicalTensor>(*currFunctionPtr, viewTypeRaw1, offset0, shape1);
    auto assIn0 = std::make_shared<LogicalTensor>(*currFunctionPtr, assRawTensor0, offset0, shape1);
    auto assIn1 = std::make_shared<LogicalTensor>(*currFunctionPtr, assRawTensor1, offset1, shape1);
    auto outcast = std::make_shared<LogicalTensor>(*currFunctionPtr, outRawTensor, offset0, shape);
    // Init Graph
    currFunctionPtr->AddOperation(Opcode::OP_COPY_IN, {incast}, {copy0});
    currFunctionPtr->AddOperation(Opcode::OP_COPY_IN, {incast}, {copy1});
    currFunctionPtr->AddOperation(Opcode::OP_COPY_OUT, {copy0}, {viewIn0});
    currFunctionPtr->AddOperation(Opcode::OP_COPY_OUT, {copy1}, {viewIn1});
    auto &viewOp0 = currFunctionPtr->AddOperation(Opcode::OP_VIEW, {viewIn0}, {viewTypeIn0});
    auto &viewOp1 = currFunctionPtr->AddOperation(Opcode::OP_VIEW, {viewIn1}, {viewTypeIn1});
    currFunctionPtr->AddOperation(Opcode::OP_VIEW_TYPE, {viewTypeIn0}, {assIn0});
    currFunctionPtr->AddOperation(Opcode::OP_VIEW_TYPE, {viewTypeIn1}, {assIn1});
    auto &assOp0 = currFunctionPtr->AddOperation(Opcode::OP_ASSEMBLE, {assIn0}, {outcast});
    auto &assOp1 = currFunctionPtr->AddOperation(Opcode::OP_ASSEMBLE, {assIn1}, {outcast});
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

TEST_F(ReplaceTensorTest, TestProcessHubAssembleOp_Success) {
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestProcessHubAssembleOp_Success", 
                                                     "TestProcessHubAssembleOp_Success", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);
    
    // 准备HUB-ASSEMBLE-OUTCAST链
    std::vector<int64_t> shape = {kNumEight, kNumEight};
    std::vector<int64_t> offset = {kNumZero, kNumZero};
    
    // 创建共享的raw tensor
    std::shared_ptr<RawTensor> rawTensor = std::make_shared<RawTensor>(DT_FP32, shape);
    
    // 创建HUB操作相关张量
    auto hubInput = std::make_shared<LogicalTensor>(*currFunctionPtr, rawTensor, offset, shape);
    auto hubOutput = std::make_shared<LogicalTensor>(*currFunctionPtr, rawTensor, offset, shape);
    
    // 创建ASSEMBLE操作相关张量
    auto assembleOutput = std::make_shared<LogicalTensor>(*currFunctionPtr, rawTensor, offset, shape);
    
    // 设置内存类型
    hubInput->SetMemoryTypeOriginal(MEM_DEVICE_DDR, true);
    hubOutput->SetMemoryTypeOriginal(MEM_DEVICE_DDR, true);
    assembleOutput->SetMemoryTypeOriginal(MEM_DEVICE_DDR, true);
    
    // 创建操作
    auto &hubOp = currFunctionPtr->AddOperation(Opcode::OP_HUB, {hubInput}, {hubOutput});
    auto &assembleOp = currFunctionPtr->AddOperation(Opcode::OP_ASSEMBLE, {hubOutput}, {assembleOutput});
    
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

TEST_F(ReplaceTensorTest, TestA_MULACC_B) {
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestProcessHubAssembleOp_BrokenChain", 
                                                     "TestProcessHubAssembleOp_BrokenChain", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);
    // Prepare the graph
    std::vector<int64_t> mulAccshape = {kNumEight, kNumEight};
    std::vector<int64_t> offset0 = {kNumZero, kNumZero};
    // init RawTensor
    std::shared_ptr<RawTensor> inRawTensor = std::make_shared<RawTensor>(DT_FP32, mulAccshape);
    std::shared_ptr<RawTensor> outRawTensor = std::make_shared<RawTensor>(DT_FP32, mulAccshape);
    // init LogicalTensor
    auto inTensor0 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, mulAccshape);
    auto inTensor1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, mulAccshape);
    auto mulAccIn = std::make_shared<LogicalTensor>(*currFunctionPtr, inRawTensor, offset0, mulAccshape);
    auto mulAccOut = std::make_shared<LogicalTensor>(*currFunctionPtr, outRawTensor, offset0, mulAccshape);
    auto outTensor = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, mulAccshape);
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

TEST_F(ReplaceTensorTest, TestSameAssembleOut) {
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestSameAssembleOut", "TestSameAssembleOut", nullptr);
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
    auto incast = std::make_shared<LogicalTensor>(*currFunctionPtr, inRawTensor, offset0, shape);
    incast->SetMemoryTypeBoth(MEM_DEVICE_DDR, true); 
    auto copyInOut = std::make_shared<LogicalTensor>(*currFunctionPtr, copyInRawTensor, offset0, shape1);
    copyInOut->SetMemoryTypeBoth(MEM_UB, true);
    auto outcast0 = std::make_shared<LogicalTensor>(*currFunctionPtr, outRawTensor0, offset0, shape);
    outcast0->SetMemoryTypeBoth(MEM_DEVICE_DDR, true);
    auto outcast1 = std::make_shared<LogicalTensor>(*currFunctionPtr, outRawTensor1, offset0, shape1);
    outcast1->SetMemoryTypeBoth(MEM_DEVICE_DDR, true);
    /*       Init Graph
                            /—————> assemble -outcast1
                             /————> assemble \         
        incast ————> copyIn -                 - outcast0  
                             \————> assemble /          
    */
    auto &copyInOp = currFunctionPtr->AddOperation(Opcode::OP_COPY_IN, {incast}, {copyInOut});
    auto &assOp0 = currFunctionPtr->AddOperation(Opcode::OP_ASSEMBLE, {copyInOut}, {outcast0});
    auto &assOp1 = currFunctionPtr->AddOperation(Opcode::OP_ASSEMBLE, {copyInOut}, {outcast0});
    auto &assOp2 = currFunctionPtr->AddOperation(Opcode::OP_ASSEMBLE, {copyInOut}, {outcast1});
    // Init Attribute
    auto copyInAttr = std::make_shared<CopyOpAttribute>(
        OpImmediate::Specified(offset0),
        MEM_UB,
        OpImmediate::Specified(shape),
        OpImmediate::Specified(shape));
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
    EXPECT_EQ(currFunctionPtr->Operations().size(), opSumBefore + 4);
    EXPECT_EQ(pass.PostCheck(*currFunctionPtr), SUCCESS);
}

TEST_F(ReplaceTensorTest, TestNotInplaceReshape) {
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestNotInplaceReshape", "TestNotInplaceReshape", nullptr);
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
    auto incast = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    auto reshapeOut0 = std::make_shared<LogicalTensor>(*currFunctionPtr, reshapeRawTensor0, offset0, shape1);
    auto viewOut = std::make_shared<LogicalTensor>(*currFunctionPtr, viewRawTensor, offset0, shape2);
    auto outcast = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape3);
    /* Init Graph
        incast0 -> Reshape -> View -> Reshape -> outcast
    */
    currFunctionPtr->AddOperation(Opcode::OP_RESHAPE, {incast}, {reshapeOut0});
    auto &viewOp = currFunctionPtr->AddOperation(Opcode::OP_VIEW, {reshapeOut0}, {viewOut});
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

// ========== 测试用例：InsertAssembleCopy - 整体插入拷贝序列流程 ==========
TEST_F(ReplaceTensorTest, InsertNeedCopy) {
    auto currFunctionPtr =
        std::make_shared<Function>(Program::GetInstance(), "InsertNeedCopy", "InsertNeedCopy", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);
    Program::GetInstance().InsertFuncToFunctionMap("InsertNeedCopy", currFunctionPtr);

    // 创建共享输入tensor (UB内存类型)
    std::vector<int64_t> shape = {32, 64};
    auto sharedInput = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    sharedInput->SetMemoryTypeBoth(MemoryType::MEM_UB, true);

    // 创建多个输出tensor
    auto output1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    output1->SetMemoryTypeBoth(MemoryType::MEM_UB, true);
    auto output2 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    output2->SetMemoryTypeBoth(MemoryType::MEM_UB, true);
    auto output3 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    output3->SetMemoryTypeBoth(MemoryType::MEM_UB, true);

    // 创建多个ASSEMBLE操作，共享同一个输入
    auto &assemble1 = currFunctionPtr->AddRawOperation(Opcode::OP_ASSEMBLE, {sharedInput}, {output1});
    assemble1.SetOpAttribute(std::make_shared<AssembleOpAttribute>(std::vector<int64_t>{0, 0}));
    auto &assemble2 = currFunctionPtr->AddRawOperation(Opcode::OP_ASSEMBLE, {sharedInput}, {output2});
    assemble2.SetOpAttribute(std::make_shared<AssembleOpAttribute>(std::vector<int64_t>{0, 0}));
    auto &assemble3 = currFunctionPtr->AddRawOperation(Opcode::OP_ASSEMBLE, {sharedInput}, {output3});
    assemble3.SetOpAttribute(std::make_shared<AssembleOpAttribute>(std::vector<int64_t>{0, 0}));

    currFunctionPtr->inCasts_.push_back(sharedInput);
    currFunctionPtr->outCasts_.push_back(output1);
    currFunctionPtr->outCasts_.push_back(output2);
    currFunctionPtr->outCasts_.push_back(output3);

    // 调用InsertAssembleCopy
    ReplaceTensor commonOperation;
    commonOperation.InsertNeedCopy(*currFunctionPtr);

    // 验证插入的拷贝序列
    int copyInCount = 0;
    int copyOutCount = 0;
    int assembleCount = 0;
    for (const auto &op : currFunctionPtr->Operations()) {
        if (op.GetOpcode() == Opcode::OP_COPY_IN) {
            copyInCount++;
        } else if (op.GetOpcode() == Opcode::OP_COPY_OUT) {
            copyOutCount++;
        } else if (op.GetOpcode() == Opcode::OP_ASSEMBLE) {
            assembleCount++;
        }
    }

    // 应该为2个ASSEMBLE操作插入拷贝序列（每个需要2个COPY操作）
    EXPECT_EQ(assembleCount, 3) << "Should have 3 ASSEMBLE operations";
    EXPECT_EQ(copyInCount, 2) << "Should insert 2 COPY_IN operations";
    EXPECT_EQ(copyOutCount, 2) << "Should insert 2 COPY_OUT operations";
}

// ========== 测试用例：InsertAssembleCopy - DDR内存类型场景 ==========
TEST_F(ReplaceTensorTest, InsertAssembleCopyDDR) {
    auto currFunctionPtr =
        std::make_shared<Function>(Program::GetInstance(), "InsertAssembleCopyDDR", "InsertAssembleCopyDDR", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);
    Program::GetInstance().InsertFuncToFunctionMap("InsertAssembleCopyDDR", currFunctionPtr);

    // 创建共享输入tensor (DDR内存类型)
    std::vector<int64_t> shape = {32, 64};
    auto sharedInput = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    sharedInput->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);

    // 创建多个输出tensor
    auto output1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    output1->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    auto output2 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    output2->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);

    // 创建多个ASSEMBLE操作，共享同一个输入
    auto &assemble1 = currFunctionPtr->AddRawOperation(Opcode::OP_ASSEMBLE, {sharedInput}, {output1});
    assemble1.SetOpAttribute(std::make_shared<AssembleOpAttribute>(std::vector<int64_t>{0, 0}));
    auto &assemble2 = currFunctionPtr->AddRawOperation(Opcode::OP_ASSEMBLE, {sharedInput}, {output2});
    assemble2.SetOpAttribute(std::make_shared<AssembleOpAttribute>(std::vector<int64_t>{0, 0}));

    currFunctionPtr->inCasts_.push_back(sharedInput);
    currFunctionPtr->outCasts_.push_back(output1);
    currFunctionPtr->outCasts_.push_back(output2);

    // 调用InsertAssembleCopy
    ReplaceTensor commonOperationEliminate;
    commonOperationEliminate.InsertNeedCopy(*currFunctionPtr);

    // 验证插入的拷贝序列
    int copyInNum = 0;
    int copyOutNum = 0;
    int assembleNum = 0;
    for (const auto &op : currFunctionPtr->Operations()) {
        if (op.GetOpcode() == Opcode::OP_COPY_IN) {
            copyInNum++;
        } else if (op.GetOpcode() == Opcode::OP_ASSEMBLE) {
            assembleNum++;
        } else if (op.GetOpcode() == Opcode::OP_COPY_OUT) {
            copyOutNum++;
        }
    }

    // 应该为1个ASSEMBLE操作插入拷贝序列（DDR→UB→DDR）
    EXPECT_EQ(assembleNum, 2) << "Should have 2 ASSEMBLE operations";
    EXPECT_EQ(copyInNum, 1) << "Should insert 1 COPY_IN operation";
    EXPECT_EQ(copyOutNum, 1) << "Should insert 1 COPY_OUT operation";
}

// ========== 测试用例：InsertAssembleCopy - 单个ASSEMBLE不插入拷贝 ==========
TEST_F(ReplaceTensorTest, InsertAssembleCopySingleAssemble) {
    auto currFunctionPtr = std::make_shared<Function>(
        Program::GetInstance(), "InsertAssembleCopySingleAssemble", "InsertAssembleCopySingleAssemble", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);
    Program::GetInstance().InsertFuncToFunctionMap("InsertAssembleCopySingleAssemble", currFunctionPtr);

    // 创建输入tensor
    std::vector<int64_t> shape = {32, 64};
    auto input = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    input->SetMemoryTypeBoth(MemoryType::MEM_UB, true);

    // 创建输出tensor
    auto output = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    output->SetMemoryTypeBoth(MemoryType::MEM_UB, true);

    // 创建单个ASSEMBLE操作
    auto &assemble = currFunctionPtr->AddRawOperation(Opcode::OP_ASSEMBLE, {input}, {output});
    assemble.SetOpAttribute(std::make_shared<AssembleOpAttribute>(std::vector<int64_t>{0, 0}));

    currFunctionPtr->inCasts_.push_back(input);
    currFunctionPtr->outCasts_.push_back(output);

    // 调用InsertAssembleCopy
    ReplaceTensor commonOperationEliminateTest;
    commonOperationEliminateTest.InsertNeedCopy(*currFunctionPtr);

    // 验证没有插入拷贝序列
    int copyInNumBer = 0;
    int copyOutNumBer = 0;
    int assembleInNumBer = 0;
    for (const auto &op : currFunctionPtr->Operations()) {
        if (op.GetOpcode() == Opcode::OP_COPY_IN) {
            copyInNumBer++;
        } else if (op.GetOpcode() == Opcode::OP_COPY_OUT) {
            copyOutNumBer++;
        } else if (op.GetOpcode() == Opcode::OP_ASSEMBLE) {
            assembleInNumBer++;
        }
    }

    // 单个ASSEMBLE不应该插入拷贝序列
    EXPECT_EQ(assembleInNumBer, 1) << "Should have 1 ASSEMBLE operation";
    EXPECT_EQ(copyInNumBer, 0) << "Should not insert COPY_IN operation";
    EXPECT_EQ(copyOutNumBer, 0) << "Should not insert COPY_OUT operation";
}

// ========== 测试用例：InsertNeedCopy - Reshape + ASSEMBLE 插入拷贝 ==========
TEST_F(ReplaceTensorTest, InsertNeedCopyReshapeAssemble) {
    auto currFunctionPtr = std::make_shared<Function>(
        Program::GetInstance(), "InsertNeedCopyReshapeAssemble", "InsertNeedCopyReshapeAssemble", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);
    Program::GetInstance().InsertFuncToFunctionMap("InsertNeedCopyReshapeAssemble", currFunctionPtr);

    // 创建输入tensor
    std::vector<int64_t> shape1 = {64, 64};
    std::vector<int64_t> shape2 = {32, 128};
    auto input = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape1);
    input->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);

    // 创建输出tensor
    auto output = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape2);
    output->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);

    // 创建UB上的tensor
    auto ubTensor1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape1);
    ubTensor1->SetMemoryTypeBoth(MemoryType::MEM_UB, true);
    auto ubTensor2 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape2);
    ubTensor2->SetMemoryTypeBoth(MemoryType::MEM_UB, true);

    // 创建计算图Op操作
    currFunctionPtr->AddRawOperation(Opcode::OP_COPY_IN, {input}, {ubTensor1});
    currFunctionPtr->AddRawOperation(Opcode::OP_RESHAPE, {ubTensor1}, {ubTensor2});
    auto &assemble = currFunctionPtr->AddRawOperation(Opcode::OP_ASSEMBLE, {ubTensor2}, {output});
    assemble.SetOpAttribute(std::make_shared<AssembleOpAttribute>(std::vector<int64_t>{0, 0}));

    currFunctionPtr->inCasts_.push_back(input);
    currFunctionPtr->outCasts_.push_back(output);

    // 调用InsertAssembleCopy
    ReplaceTensor replaceTensor;
    replaceTensor.InsertNeedCopy(*currFunctionPtr);

    // 验证插入拷贝序列
    int copyInNumBer = 0;
    int copyOutNumBer = 0;
    for (const auto &op : currFunctionPtr->Operations()) {
        if (op.GetOpcode() == Opcode::OP_COPY_IN) {
            copyInNumBer++;
        } else if (op.GetOpcode() == Opcode::OP_COPY_OUT) {
            copyOutNumBer++;
        }
    }
    
    EXPECT_EQ(copyInNumBer, kNumTwo) << "Should insert COPY_IN operation";
    EXPECT_EQ(copyOutNumBer, kNumOne) << "Should insert COPY_OUT operation";
}

// ========== 测试用例：InsertNeedCopy - View + Reshape + CopyIn 不插入拷贝 ==========
TEST_F(ReplaceTensorTest, InsertNeedCopyViewReshapeCopyOut) {
    auto currFunctionPtr = std::make_shared<Function>(
        Program::GetInstance(), "InsertNeedCopyViewReshapeCopyOut", "InsertNeedCopyViewReshapeCopyOut", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);
    Program::GetInstance().InsertFuncToFunctionMap("InsertNeedCopyViewReshapeCopyOut", currFunctionPtr);

    // 创建输入tensor
    std::vector<int64_t> shape1 = {16, 64};
    std::vector<int64_t> shape2 = {8, 64};
    std::vector<int64_t> shape3 = {64, 8};
    std::vector<int64_t> offset = {kNumZero, kNumZero};
    auto input = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape1);
    input->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);

    // 创建UB上的tensor
    auto ubTensor1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape2);
    ubTensor1->SetMemoryTypeBoth(MemoryType::MEM_UB, true);
    auto ubTensor2 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape3);
    ubTensor2->SetMemoryTypeBoth(MemoryType::MEM_UB, true);

    // 创建输出tensor
    auto output = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape3);
    output->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);

    // 创建计算图Op操作
    auto &view = currFunctionPtr->AddRawOperation(Opcode::OP_VIEW, {input}, {ubTensor1});
    view.SetOpAttribute(std::make_shared<ViewOpAttribute>(offset));
    currFunctionPtr->AddRawOperation(Opcode::OP_RESHAPE, {ubTensor1}, {ubTensor2});
    currFunctionPtr->AddRawOperation(Opcode::OP_COPY_IN, {ubTensor2}, {output});

    currFunctionPtr->inCasts_.push_back(input);
    currFunctionPtr->outCasts_.push_back(output);

    // 调用InsertAssembleCopy
    ReplaceTensor replaceTensor;
    replaceTensor.InsertNeedCopy(*currFunctionPtr);

    // 验证没有插入拷贝序列
    int copyInNumBer = 0;
    int copyOutNumBer = 0;
    for (const auto &op : currFunctionPtr->Operations()) {
        if (op.GetOpcode() == Opcode::OP_COPY_OUT) {
            copyOutNumBer++;
        } else if (op.GetOpcode() == Opcode::OP_COPY_IN) {
            copyInNumBer++;
        }
    }
    
    EXPECT_EQ(copyInNumBer, kNumOne) << "Should not insert COPY_IN operation";
    EXPECT_EQ(copyOutNumBer, kNumZero) << "Should not insert COPY_OUT operation";
}

TEST_F(ReplaceTensorTest, UpdateCopyInAttrAfterBackAssemble) {
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "UpdateCopyInAttrAfterBackAssemble", "UpdateCopyInAttrAfterBackAssemble", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);
    
    // Prepare the graph
    std::vector<int64_t> inshape = {4, 4};
    std::vector<int64_t> outshape1 = {8, 4};
    std::vector<int64_t> outshape2 = {2, 4};

    auto incast = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, inshape);
    incast->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    auto copyInout1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, inshape);
    copyInout1->SetMemoryTypeBoth(MemoryType::MEM_UB, true);
    auto copyOutout1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, inshape);
    copyOutout1->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    auto copyInout2 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, inshape);
    copyInout2->SetMemoryTypeBoth(MemoryType::MEM_UB, true);
    auto outcast1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, outshape1);
    outcast1->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    auto outcast2 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, outshape2);
    outcast2->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    /* Init Graph
        incast -- CopyIn -- copyInout1 -- CopyOut -- copyOutOut1 -- Assemble -- outcast1
                                                                 -- CopyIn   -- copyInout2 -- CopyOut -- outcast2
    */
    currFunctionPtr->AddOperation(Opcode::OP_COPY_IN, {incast}, {copyInout1});
    currFunctionPtr->AddOperation(Opcode::OP_COPY_OUT, {copyInout1}, {copyOutout1});
    auto &assembleOp = currFunctionPtr->AddOperation(Opcode::OP_ASSEMBLE, {copyOutout1}, {outcast1});
    auto &copyInOp = currFunctionPtr->AddOperation(Opcode::OP_COPY_IN, {copyOutout1}, {copyInout2});
    currFunctionPtr->AddOperation(Opcode::OP_COPY_OUT, {copyInout2}, {outcast2});

    Offset assembleToOffset = {4, 0};
    auto assembleOpAttribute = std::make_shared<AssembleOpAttribute>(assembleToOffset, SymbolicScalar::FromConcrete(assembleToOffset));
    assembleOp.SetOpAttribute(assembleOpAttribute);
    Offset copyIn2FromOffset = {2, 0};
    auto copyInOpAttribute = std::make_shared<CopyOpAttribute>(
        OpImmediate::Specified(copyIn2FromOffset),
        MEM_UB, OpImmediate::Specified(copyInout2->GetShape()),
        OpImmediate::Specified(copyInout2->tensor->GetDynRawShape()),
        OpImmediate::Specified(copyInout2->GetDynValidShape())
        );
    copyInOp.SetOpAttribute(copyInOpAttribute);
    
    currFunctionPtr->inCasts_.push_back(incast);
    currFunctionPtr->outCasts_.push_back(outcast1);
    currFunctionPtr->outCasts_.push_back(outcast2);

    ReplaceTensor replaceTensorPass;
    replaceTensorPass.RunOnFunction(*currFunctionPtr);
    EXPECT_EQ(copyOutout1->GetOffset()[0] == assembleToOffset[0], true);
    EXPECT_EQ(copyOutout1->GetOffset()[1] == assembleToOffset[1], true);
    std::vector<std::string> copyInAttrNewOffset = {"6", "0"} ;
    EXPECT_EQ(copyInOpAttribute->GetFromOffset()[0].Dump(), copyInAttrNewOffset[0]);
    EXPECT_EQ(copyInOpAttribute->GetFromOffset()[1].Dump(), copyInAttrNewOffset[1]);
}
}
}