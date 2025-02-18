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
 * \file test_n_buffer_merge.cpp
 * \brief Unit test for n_buffer_merge pass.
 */

#include <gtest/gtest.h>
#include "interface/function/function.h"
#include "tilefwk/tilefwk.h"
#include "tilefwk/platform.h"
#include "interface/inner/tilefwk.h"
#include "passes/pass_mgr/pass_manager.h"
#include "interface/configs/config_manager.h"
#include "passes/tile_graph_pass/graph_partition/n_buffer_merge.h"
#include <fstream>
#include <vector>
#include <string>
#include "computational_graph_builder.h"

namespace npu {
namespace tile_fwk{

class NBufferMergeTest : public testing::Test {
public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}

    void SetUp() override {
        Program::GetInstance().Reset();
        config::Reset();
        config::SetPlatformConfig(KEY_ONLY_HOST_COMPILE, true);

        config::SetPassOption(VEC_NBUFFER_MODE, 2);
        config::SetPassOption(VEC_NBUFFER_SETTING, std::map<int64_t, int64_t>{{-1, 2}});
        Platform::Instance().ObtainPlatformInfo();
    }

    void TearDown() override {}

    Status TestNBufferMergeWithDifferentVecBufferMap(std::map<int64_t, int64_t> vecNBufferSetting);
};

TEST_F(NBufferMergeTest, TestNBufferMerge) {
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestNBufferMerge", "TestNBufferMerge", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    // Prepare the graph
    constexpr int subGraphID0 = 0;
    constexpr int subGraphID1 = 1;
    std::vector<int64_t> shape = {8, 16};
    auto incast1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    incast1->subGraphID = subGraphID0;
    auto incast2 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    incast2->subGraphID = subGraphID1;
    auto tensor1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    tensor1->subGraphID = subGraphID0;
    auto tensor2 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    tensor2->subGraphID = subGraphID0;
    auto tensor3 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    tensor3->subGraphID = subGraphID1;
    auto tensor4 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    tensor4->subGraphID = subGraphID1;

    auto &copy_op1 = currFunctionPtr->AddOperation(Opcode::OP_COPY_IN, {incast1}, {tensor1});
    copy_op1.UpdateSubgraphID(subGraphID0);
    auto &copy_op2 = currFunctionPtr->AddOperation(Opcode::OP_COPY_IN, {incast2}, {tensor3});
    copy_op2.UpdateSubgraphID(subGraphID1);
    auto &copy_out1 = currFunctionPtr->AddOperation(Opcode::OP_COPY_OUT, {tensor1}, {tensor2});
    copy_out1.UpdateSubgraphID(subGraphID0);
    auto &copy_out2 = currFunctionPtr->AddOperation(Opcode::OP_COPY_OUT, {tensor3}, {tensor4});
    copy_out2.UpdateSubgraphID(subGraphID1);

    currFunctionPtr->inCasts_.push_back(incast1);
    currFunctionPtr->inCasts_.push_back(incast2);
    currFunctionPtr->outCasts_.push_back(tensor2);
    currFunctionPtr->outCasts_.push_back(tensor4);

    // Call the pass
    NBufferMerge nPass;
    Pass &nbufferPass = nPass;
    nbufferPass.PreCheck(*currFunctionPtr);
    nbufferPass.Run(*currFunctionPtr, "", "");
    nbufferPass.PostCheck(*currFunctionPtr);
}
}
}