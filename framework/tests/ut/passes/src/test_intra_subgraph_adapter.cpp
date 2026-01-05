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
 * \file test_intra_subgraph_adapter.cpp
 * \brief Unit test for IntraSubgraphAdapter.
 */

#include <gtest/gtest.h>
#include <vector>
#include "interface/function/function.h"
#include "tilefwk/tilefwk.h"
#include "tilefwk/platform.h"
#include "interface/inner/tilefwk.h"
#include "passes/pass_mgr/pass_manager.h"
#include "interface/configs/config_manager.h"
#include "computational_graph_builder.h"
#include "passes/tile_graph_pass/data_path/intra_subgraph_adapter.h"

namespace npu::tile_fwk {
class IntraSubgraphAdapterTest : public testing::Test {
public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}

    void SetUp() override {
        Program::GetInstance().Reset();
        config::Reset();
        config::SetPlatformConfig(KEY_ONLY_HOST_COMPILE, true);
    }

    void TearDown() override {}
};

TEST_F(IntraSubgraphAdapterTest, TestBoundaryConvert) {
    ComputationalGraphBuilder subGraph;
    std::vector<std::string> tensorNames{"t1", "t2", "t3", "t4"};
    std::vector<MemoryType> tensorMemTypes{MemoryType::MEM_UB, MemoryType::MEM_UB, MemoryType::MEM_L1, MemoryType::MEM_L0A};
    std::vector<Opcode> opCodes{Opcode::OP_ADDS, Opcode::OP_CONVERT, Opcode::OP_L1_TO_L0A};
    std::vector<std::vector<std::string>> ioperands{{"t1"}, {"t2"}, {"t3"}};
    std::vector<std::vector<std::string>> ooperands{{"t2"}, {"t3"}, {"t4"}};
    std::vector<std::string> opNames{"adds", "convert", "L1ToL0A"};
    EXPECT_EQ(subGraph.AddTensors(DataType::DT_FP32, {128, 128}, tensorMemTypes, tensorNames, 0), true);
    EXPECT_EQ(subGraph.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    subGraph.GetOp("adds")->UpdateSubgraphID(0);
    subGraph.GetOp("convert")->UpdateSubgraphID(0);
    subGraph.GetOp("convert")->SetOpAttribute(std::make_shared<ConvertOpAttribute>(MemoryType::MEM_UB, MemoryType::MEM_L1));
    subGraph.GetOp("L1ToL0A")->UpdateSubgraphID(1);
    Function *function = subGraph.GetFunction();
    EXPECT_NE(function, nullptr);
    IntraSubgraphAdapter adapter;
    adapter.RunOnFunction(*function);
    const int opNum = 4;
    EXPECT_EQ(function->Operations().DuplicatedOpList().size(), opNum);
    const int copyOutIdx = 1;
    EXPECT_EQ(function->Operations().DuplicatedOpList()[copyOutIdx]->GetOpcode(), Opcode::OP_COPY_OUT);
    const int viewIdx = 2;
    EXPECT_EQ(function->Operations().DuplicatedOpList()[viewIdx]->GetOpcode(), Opcode::OP_VIEW);
    auto copyOpAttr = dynamic_cast<CopyOpAttribute*>(subGraph.GetOp("convert")->GetOpAttribute().get());
    EXPECT_NE(copyOpAttr, nullptr);
}

TEST_F(IntraSubgraphAdapterTest, TestBoundaryConvertFailed) {
    ComputationalGraphBuilder subGraph;
    std::vector<std::string> tensorNames{"t1", "t2", "t3", "t4"};
    std::vector<MemoryType> tensorMemTypes{MemoryType::MEM_UB, MemoryType::MEM_UB, MemoryType::MEM_L1, MemoryType::MEM_L0B};
    std::vector<Opcode> opCodes{Opcode::OP_ADDS, Opcode::OP_CONVERT, Opcode::OP_L1_TO_L0B};
    std::vector<std::vector<std::string>> ioperands{{"t1"}, {"t2"}, {"t3"}};
    std::vector<std::vector<std::string>> ooperands{{"t2"}, {"t3"}, {"t4"}};
    std::vector<std::string> opNames{"adds", "convert", "L1ToL0B"};
    EXPECT_EQ(subGraph.AddTensors(DataType::DT_FP32, {128, 128}, tensorMemTypes, tensorNames, 0), true);
    EXPECT_EQ(subGraph.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    subGraph.GetOp("adds")->UpdateSubgraphID(0);
    subGraph.GetOp("convert")->UpdateSubgraphID(1);
    subGraph.GetOp("convert")->SetOpAttribute(std::make_shared<ConvertOpAttribute>(MemoryType::MEM_UB, MemoryType::MEM_L1));
    subGraph.GetOp("L1ToL0B")->UpdateSubgraphID(1);
    Function *function = subGraph.GetFunction();
    EXPECT_NE(function, nullptr);
    IntraSubgraphAdapter adapter;
    EXPECT_EQ(adapter.RunOnFunction(*function), FAILED);
}

TEST_F(IntraSubgraphAdapterTest, TestInnerConvert) {
    ComputationalGraphBuilder subGraph;
    std::vector<std::string> tensorNames{"t0", "t1", "t2", "t3"};
    std::vector<MemoryType> tensorMemTypes{MemoryType::MEM_UB, MemoryType::MEM_UB, MemoryType::MEM_L1, MemoryType::MEM_L0A};
    std::vector<Opcode> opCodes{Opcode::OP_ADDS, Opcode::OP_CONVERT, Opcode::OP_L1_TO_L0A};
    std::vector<std::vector<std::string>> ioperands{{"t0"}, {"t1"}, {"t2"}};
    std::vector<std::vector<std::string>> ooperands{{"t1"}, {"t2"}, {"t3"}};
    std::vector<std::string> opNames{"adds", "convert", "L1ToL0A"};
    EXPECT_EQ(subGraph.AddTensors(DataType::DT_FP32, {128, 128}, tensorMemTypes, tensorNames, 0), true);
    EXPECT_EQ(subGraph.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    subGraph.GetOp("adds")->UpdateSubgraphID(0);
    subGraph.GetOp("convert")->UpdateSubgraphID(0);
    subGraph.GetOp("convert")->SetOpAttribute(std::make_shared<ConvertOpAttribute>(MemoryType::MEM_UB, MemoryType::MEM_L1));
    subGraph.GetOp("L1ToL0A")->UpdateSubgraphID(0);
    Function *function = subGraph.GetFunction();
    EXPECT_NE(function, nullptr);
    IntraSubgraphAdapter adapter;
    adapter.RunOnFunction(*function);
    const int opNum = 3;
    EXPECT_EQ(function->Operations().DuplicatedOpList().size(), opNum);
    const int convertIdx = 1;
    EXPECT_EQ(function->Operations().DuplicatedOpList()[convertIdx]->GetOpcode(), Opcode::OP_CONVERT);
}

}