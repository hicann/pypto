/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_axis_combine_marker.cpp
 * \brief Unit test for AxisCombineMarker.
 */

#include <gtest/gtest.h>
#include "interface/function/function.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "passes/tile_graph_pass/graph_constraint/axis_combine_marker.h"
#include "computational_graph_builder.h"
#include "passes/pass_mgr/pass_manager.h"
#include "interface/configs/config_manager.h"
#include <nlohmann/json.hpp>
#include <fstream>
#include <vector>
#include <string>

using namespace npu::tile_fwk;
constexpr size_t K_1 = 1;
constexpr size_t K_2 = 2;
constexpr size_t K_4 = 4;
constexpr size_t K_8 = 8;
constexpr size_t K_16 = 16;
constexpr size_t K_32 = 32;
constexpr size_t K_64 = 64;
constexpr size_t K_128 = 128;

class TestAxisCombineMarker : public ::testing::Test {
public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}

    void SetUp() override {
        Program::GetInstance().Reset();
        config::Reset();
        config::SetHostOption(COMPILE_STAGE, CS_EXECUTE_GRAPH);
        config::SetPlatformConfig(KEY_ENABLE_COST_MODEL, false);
    }
    void TearDown() override {
    }
};

/*
before:
    copyin
    [8,1]
      |
    copyout
    [8,1]

after:
    Tensor should be marked as ENABLE
*/
TEST_F(TestAxisCombineMarker, basic_copyin_enable) {
    ComputationalGraphBuilder graph;
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {8, 1}, MemoryType::MEM_DEVICE_DDR, "t1"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {8, 1}, MemoryType::MEM_UB, "t2"), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_COPY_IN, {"t1"}, {"t2"}, "copy_in", true), true);
    
    auto *rootFuncPtr = graph.GetFunction();
    AxisCombineMarker marker;
    marker.Run(*rootFuncPtr);
    
    // Verify the tensor is marked as ENABLE
    auto t1 = graph.GetTensor("t1");
    auto t2 = graph.GetTensor("t2");
    EXPECT_EQ(marker.IsTensorEnableAxisCombine(t1), false);  // DDR tensor should not be marked
    EXPECT_EQ(marker.IsTensorEnableAxisCombine(t2), true);   // UB tensor with last dim=1 should be enabled
}

/*
before:
    copyin
    [8,16]
      |
    copyout
    [8,16]

after:
    Tensor should be marked as UNKNOWN
*/
TEST_F(TestAxisCombineMarker, basic_copyin_unknown) {
    ComputationalGraphBuilder graph;
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {8, 16}, MemoryType::MEM_DEVICE_DDR, "t1"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {8, 16}, MemoryType::MEM_UB, "t2"), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_COPY_IN, {"t1"}, {"t2"}, "copy_in", true), true);
    
    auto *rootFuncPtr = graph.GetFunction();
    AxisCombineMarker marker;
    marker.Run(*rootFuncPtr);
    
    // Verify the tensor is marked as UNKNOWN
    auto t1 = graph.GetTensor("t1");
    auto t2 = graph.GetTensor("t2");
    EXPECT_EQ(marker.IsTensorEnableAxisCombine(t1), false);  // DDR tensor should not be marked
    EXPECT_EQ(marker.IsTensorEnableAxisCombine(t2), false);   // UB tensor with last dim != 1 should be unknown
}

/*
before:
    copyin
    [8,15]
      |
    view
    [8,1]
      |
    copyout
    [8,1]

after:
    Output tensor should be marked as ENABLE
*/
TEST_F(TestAxisCombineMarker, view_enable) {
    ComputationalGraphBuilder graph;
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {8, 15}, MemoryType::MEM_DEVICE_DDR, "t1"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {8, 15}, MemoryType::MEM_UB, "t2"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {8, 1}, MemoryType::MEM_UB, "t3"), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_COPY_IN, {"t1"}, {"t2"}, "copy_in", true), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_VIEW, {"t2"}, {"t3"}, "view", true), true);
    
    auto *rootFuncPtr = graph.GetFunction();
    AxisCombineMarker marker;
    marker.Run(*rootFuncPtr);
    
    // Verify the tensor is marked as ENABLE
    auto t3 = graph.GetTensor("t3");
    EXPECT_EQ(marker.IsTensorEnableAxisCombine(t3), false);   // View output with last dim=1 should be enabled
}

/*
before:
    copyin
    [8,15]
      |
    view
    [8,16]
      |
    copyout
    [8,16]

after:
    Output tensor should be marked as UNKNOWN
*/
TEST_F(TestAxisCombineMarker, view_unknown) {
    ComputationalGraphBuilder graph;
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {8, 15}, MemoryType::MEM_DEVICE_DDR, "t1"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {8, 15}, MemoryType::MEM_UB, "t2"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {8, 16}, MemoryType::MEM_UB, "t3"), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_COPY_IN, {"t1"}, {"t2"}, "copy_in", true), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_VIEW, {"t2"}, {"t3"}, "view", true), true);
    
    auto *rootFuncPtr = graph.GetFunction();
    AxisCombineMarker marker;
    marker.Run(*rootFuncPtr);
    
    // Verify the tensor is marked as UNKNOWN
    auto t3 = graph.GetTensor("t3");
    EXPECT_EQ(marker.IsTensorEnableAxisCombine(t3), false);   // View output with last dim != 1 should be unknown
}

/*
before:
    copyin
    [8,1]
      |
    assemble
    [8,1]
      |
    copyout
    [8,1]

after:
    Output tensor should be marked as ENABLE
*/
TEST_F(TestAxisCombineMarker, assemble_enable) {
    ComputationalGraphBuilder graph;
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {8, 1}, MemoryType::MEM_DEVICE_DDR, "t1"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {8, 1}, MemoryType::MEM_UB, "t2"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {8, 1}, MemoryType::MEM_UB, "t3"), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_COPY_IN, {"t1"}, {"t2"}, "copy_in", true), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_ASSEMBLE, {"t2"}, {"t3"}, "assemble", true), true);
    
    auto *rootFuncPtr = graph.GetFunction();
    AxisCombineMarker marker;
    marker.Run(*rootFuncPtr);
    
    // Verify the tensor is marked as ENABLE
    auto t3 = graph.GetTensor("t3");
    EXPECT_EQ(marker.IsTensorEnableAxisCombine(t3), true);   // Assemble output with same shape should be enabled
}

/*
before:
    copyin
    [8,1]
      |
    assemble
    [8,16]
      |
    copyout
    [8,16]

after:
    Both tensors should be marked as DISABLE
*/
TEST_F(TestAxisCombineMarker, assemble_disable) {
    ComputationalGraphBuilder graph;
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {8, 1}, MemoryType::MEM_DEVICE_DDR, "t1"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {8, 1}, MemoryType::MEM_UB, "t2"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {8, 16}, MemoryType::MEM_UB, "t3"), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_COPY_IN, {"t1"}, {"t2"}, "copy_in", true), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_ASSEMBLE, {"t2"}, {"t3"}, "assemble", true), true);
    
    auto *rootFuncPtr = graph.GetFunction();
    AxisCombineMarker marker;
    marker.Run(*rootFuncPtr);
    
    // Verify the tensor is marked as DISABLE
    auto t2 = graph.GetTensor("t2");
    auto t3 = graph.GetTensor("t3");
    EXPECT_EQ(marker.IsTensorEnableAxisCombine(t2), false);   // Assemble input should be disabled
    EXPECT_EQ(marker.IsTensorEnableAxisCombine(t3), false);   // Assemble output should be disabled
}

/*
before:
    copyin
    [8,1]
      |
    expand
    [8,1,1]
      |
    copyout
    [8,1,1]

after:
    Output tensor should be marked as ENABLE
*/
TEST_F(TestAxisCombineMarker, expand_non_last_axis) {
    ComputationalGraphBuilder graph;
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {1, 8, 1}, MemoryType::MEM_DEVICE_DDR, "t1"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {1, 8, 1}, MemoryType::MEM_UB, "t2"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {4, 8, 1}, MemoryType::MEM_UB, "t3"), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_COPY_IN, {"t1"}, {"t2"}, "copy_in", true), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_EXPAND, {"t2"}, {"t3"}, "expand", true), true);
    auto expand_op = graph.GetOp("expand");
    expand_op->SetAttribute(OP_ATTR_PREFIX + "EXPANDDIM", 0);  // Expand non-last axis
    
    auto *rootFuncPtr = graph.GetFunction();
    AxisCombineMarker marker;
    marker.Run(*rootFuncPtr);
    
    // Verify the tensor is marked as ENABLE
    auto t3 = graph.GetTensor("t3");
    EXPECT_EQ(marker.IsTensorEnableAxisCombine(t3), true);   // Expand on non-last axis should be enabled
}

/*
before:
    copyin
    [8,1]
      |
    expand
    [8,16]
      |
    copyout
    [8,16]

after:
    Input tensor should be marked as DISABLE, output should be UNKNOWN
*/
TEST_F(TestAxisCombineMarker, expand_last_axis) {
    ComputationalGraphBuilder graph;
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {8, 1}, MemoryType::MEM_DEVICE_DDR, "t1"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {8, 1}, MemoryType::MEM_UB, "t2"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {8, 16}, MemoryType::MEM_UB, "t3"), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_COPY_IN, {"t1"}, {"t2"}, "copy_in", true), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_EXPAND, {"t2"}, {"t3"}, "expand", true), true);
    auto expand_op = graph.GetOp("expand");
    expand_op->SetAttribute(OP_ATTR_PREFIX + "EXPANDDIM", 1);  // Expand last axis
    
    auto *rootFuncPtr = graph.GetFunction();
    AxisCombineMarker marker;
    marker.Run(*rootFuncPtr);
    
    // Verify the tensor is marked correctly
    auto t2 = graph.GetTensor("t2");
    auto t3 = graph.GetTensor("t3");
    EXPECT_EQ(marker.IsTensorEnableAxisCombine(t2), false);   // Input should be disabled
    EXPECT_EQ(marker.IsTensorEnableAxisCombine(t3), false);   // Output should be unknown (not enabled)
}

/*
before:
    copyin
    [8,1,16]
      |
    reduce
    [8,1]
      |
    copyout
    [8,1]

after:
    Output tensor should be marked as ENABLE
*/
TEST_F(TestAxisCombineMarker, reduce_last_axis) {
    ComputationalGraphBuilder graph;
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {8, 1, 16}, MemoryType::MEM_DEVICE_DDR, "t1"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {8, 1, 16}, MemoryType::MEM_UB, "t2"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {8, 1, 1}, MemoryType::MEM_UB, "t3"), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_COPY_IN, {"t1"}, {"t2"}, "copy_in", true), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_ROWMAX_COMBINE_AXIS_SINGLE, {"t2"}, {"t3"}, "reduce", true), true);
    auto reduce_op = graph.GetOp("reduce");
    reduce_op->SetAttribute(OP_ATTR_PREFIX + "AXIS", 2);  // Reduce last axis
    
    auto *rootFuncPtr = graph.GetFunction();
    AxisCombineMarker marker;
    marker.Run(*rootFuncPtr);
    
    // Verify the tensor is marked as ENABLE
    auto t3 = graph.GetTensor("t3");
    EXPECT_EQ(marker.IsTensorEnableAxisCombine(t3), true);   // Reduce on last axis should be enabled
}

/*
before:
    copyin
    [8,1,16]
      |
    reduce
    [8,16]
      |
    copyout
    [8,16]

after:
    Output tensor should be marked as DISABLE
*/
TEST_F(TestAxisCombineMarker, reduce_second_last_axis) {
    ComputationalGraphBuilder graph;
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {8, 8, 16}, MemoryType::MEM_DEVICE_DDR, "t1"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {8, 8, 16}, MemoryType::MEM_UB, "t2"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {8, 1, 16}, MemoryType::MEM_UB, "t3"), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_COPY_IN, {"t1"}, {"t2"}, "copy_in", true), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_ROWMAX_COMBINE_AXIS_SINGLE, {"t2"}, {"t3"}, "reduce", true), true);
    auto reduce_op = graph.GetOp("reduce");
    reduce_op->SetAttribute(OP_ATTR_PREFIX + "AXIS", 1);  // Reduce second last axis
    
    auto *rootFuncPtr = graph.GetFunction();
    AxisCombineMarker marker;
    marker.Run(*rootFuncPtr);
    
    // Verify the tensor is marked as DISABLE
    auto t3 = graph.GetTensor("t3");
    EXPECT_EQ(marker.IsTensorEnableAxisCombine(t3), false);   // Reduce on second last axis should be disabled
}

/*
before:
    copyin     copyin
    [8,1]      [8,1]
      \         /
        add
       [8,1]
         |
      copyout
    [8,1]

after:
    Output tensor should be marked as ENABLE
*/
TEST_F(TestAxisCombineMarker, elewise_enable) {
    ComputationalGraphBuilder graph;
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {8, 1}, MemoryType::MEM_DEVICE_DDR, "t1"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {8, 1}, MemoryType::MEM_DEVICE_DDR, "t2"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {8, 1}, MemoryType::MEM_UB, "t3"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {8, 1}, MemoryType::MEM_UB, "t4"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {8, 1}, MemoryType::MEM_UB, "t5"), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_COPY_IN, {"t1"}, {"t3"}, "copy_in1", true), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_COPY_IN, {"t2"}, {"t4"}, "copy_in2", true), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_ADD, {"t3", "t4"}, {"t5"}, "add", true), true);
    
    auto *rootFuncPtr = graph.GetFunction();
    AxisCombineMarker marker;
    marker.Run(*rootFuncPtr);
    
    // Verify the tensor is marked as ENABLE
    auto t5 = graph.GetTensor("t5");
    EXPECT_EQ(marker.IsTensorEnableAxisCombine(t5), true);   // Elewise output with last dim=1 should be enabled
}

/*
before:
    copyin     copyin
    [8,1]      [8,16]
      \         /
        add
       [8,16]
         |
      copyout
    [8,16]

after:
    Output tensor should be marked as UNKNOWN
*/
TEST_F(TestAxisCombineMarker, elewise_unknown) {
    ComputationalGraphBuilder graph;
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {8, 1}, MemoryType::MEM_DEVICE_DDR, "t1"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {8, 16}, MemoryType::MEM_DEVICE_DDR, "t2"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {8, 1}, MemoryType::MEM_UB, "t3"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {8, 16}, MemoryType::MEM_UB, "t4"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {8, 16}, MemoryType::MEM_UB, "t5"), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_COPY_IN, {"t1"}, {"t3"}, "copy_in1", true), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_COPY_IN, {"t2"}, {"t4"}, "copy_in2", true), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_ADD, {"t3", "t4"}, {"t5"}, "add", true), true);
    
    auto *rootFuncPtr = graph.GetFunction();
    AxisCombineMarker marker;
    marker.Run(*rootFuncPtr);
    
    // Verify the tensor is marked as UNKNOWN
    auto t5 = graph.GetTensor("t5");
    EXPECT_EQ(marker.IsTensorEnableAxisCombine(t5), false);   // Elewise output with last dim != 1 should be unknown
}

/*
before:
    copyin     copyin
    [8,1]      [8,1]
      \         /
        add
       [8,1]
         |
      view
     [8,1]
       |
     copyout
    [8,1]

after:
    All tensors should be marked as ENABLE
*/
TEST_F(TestAxisCombineMarker, complex_graph_enable) {
    ComputationalGraphBuilder graph;
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {8, 1}, MemoryType::MEM_DEVICE_DDR, "t1"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {8, 1}, MemoryType::MEM_DEVICE_DDR, "t2"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {8, 1}, MemoryType::MEM_UB, "t3"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {8, 1}, MemoryType::MEM_UB, "t4"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {8, 1}, MemoryType::MEM_UB, "t5"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {8, 1}, MemoryType::MEM_UB, "t6"), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_COPY_IN, {"t1"}, {"t3"}, "copy_in1", true), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_COPY_IN, {"t2"}, {"t4"}, "copy_in2", true), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_ADD, {"t3", "t4"}, {"t5"}, "add", true), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_VIEW, {"t5"}, {"t6"}, "view", true), true);
    
    auto *rootFuncPtr = graph.GetFunction();
    AxisCombineMarker marker;
    marker.Run(*rootFuncPtr);
    
    // Verify all tensors are marked as ENABLE
    auto t3 = graph.GetTensor("t3");
    auto t4 = graph.GetTensor("t4");
    auto t5 = graph.GetTensor("t5");
    auto t6 = graph.GetTensor("t6");
    EXPECT_EQ(marker.IsTensorEnableAxisCombine(t3), true);
    EXPECT_EQ(marker.IsTensorEnableAxisCombine(t4), true);
    EXPECT_EQ(marker.IsTensorEnableAxisCombine(t5), true);
    EXPECT_EQ(marker.IsTensorEnableAxisCombine(t6), true);
}

/*
before:
    copyin     copyin
    [8,1]      [8,1]
      \         /
        add
       [8,1]
         |
     assemble
     [8,16]
       |
     copyout
    [8,16]

after:
    Tensors before assemble should be ENABLE, after assemble should be DISABLE
*/
TEST_F(TestAxisCombineMarker, complex_graph_disable) {
    ComputationalGraphBuilder graph;
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {8, 1}, MemoryType::MEM_DEVICE_DDR, "t1"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {8, 1}, MemoryType::MEM_DEVICE_DDR, "t2"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {8, 1}, MemoryType::MEM_UB, "t3"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {8, 1}, MemoryType::MEM_UB, "t4"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {8, 1}, MemoryType::MEM_UB, "t5"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {8, 16}, MemoryType::MEM_UB, "t6"), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_COPY_IN, {"t1"}, {"t3"}, "copy_in1", true), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_COPY_IN, {"t2"}, {"t4"}, "copy_in2", true), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_ADD, {"t3", "t4"}, {"t5"}, "add", true), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_ASSEMBLE, {"t5"}, {"t6"}, "assemble", true), true);
    
    auto *rootFuncPtr = graph.GetFunction();
    AxisCombineMarker marker;
    marker.Run(*rootFuncPtr);
    
    // Verify tensors before assemble are ENABLE, after assemble are DISABLE
    auto t3 = graph.GetTensor("t3");
    auto t4 = graph.GetTensor("t4");
    auto t5 = graph.GetTensor("t5");
    auto t6 = graph.GetTensor("t6");
    EXPECT_EQ(marker.IsTensorEnableAxisCombine(t3), false);  // Should be disabled due to assemble
    EXPECT_EQ(marker.IsTensorEnableAxisCombine(t4), false);  // Should be disabled due to assemble
    EXPECT_EQ(marker.IsTensorEnableAxisCombine(t5), false);  // Should be disabled due to assemble
    EXPECT_EQ(marker.IsTensorEnableAxisCombine(t6), false);  // Should be disabled
}

/*
before:
    copyin
    [8,1]
      |
     expand
    [8,1,1]
      |
    reduce
    [8,1]
      |
    copyout
    [8,1]

after:
    All tensors should be marked as ENABLE
*/
TEST_F(TestAxisCombineMarker, expand_reduce_chain) {
    ComputationalGraphBuilder graph;
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {1, 8, 1}, MemoryType::MEM_DEVICE_DDR, "t1"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {1, 8, 1}, MemoryType::MEM_UB, "t2"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {4, 8, 1}, MemoryType::MEM_UB, "t3"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {1, 8, 1}, MemoryType::MEM_UB, "t4"), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_COPY_IN, {"t1"}, {"t2"}, "copy_in", true), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_EXPAND, {"t2"}, {"t3"}, "expand", true), true);
    auto expand_op = graph.GetOp("expand");
    expand_op->SetAttribute(OP_ATTR_PREFIX + "EXPANDDIM", 0);  // Expand non-last axis
    EXPECT_EQ(graph.AddOp(Opcode::OP_ROWMAX_COMBINE_AXIS_SINGLE, {"t3"}, {"t4"}, "reduce", true), true);
    auto reduce_op = graph.GetOp("reduce");
    reduce_op->SetAttribute(OP_ATTR_PREFIX + "AXIS", 0);  // Reduce last axis
    
    auto *rootFuncPtr = graph.GetFunction();
    AxisCombineMarker marker;
    marker.Run(*rootFuncPtr);
    
    // Verify all tensors are marked as ENABLE
    auto t2 = graph.GetTensor("t2");
    auto t3 = graph.GetTensor("t3");
    auto t4 = graph.GetTensor("t4");
    EXPECT_EQ(marker.IsTensorEnableAxisCombine(t2), true);
    EXPECT_EQ(marker.IsTensorEnableAxisCombine(t3), true);
    EXPECT_EQ(marker.IsTensorEnableAxisCombine(t4), true);
}

/*
before:
    copyin
    [8,1]
      |
    reshape
    [8,1]
      |
    copyout
    [8,1]

after:
    Output tensor should be marked as UNKNOWN (reshape is not handled)
*/
TEST_F(TestAxisCombineMarker, unhandled_op) {
    ComputationalGraphBuilder graph;
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {8, 1}, MemoryType::MEM_DEVICE_DDR, "t1"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {8, 1}, MemoryType::MEM_UB, "t2"), true);
    EXPECT_EQ(graph.AddTensor(DataType::DT_FP32, {8, 1}, MemoryType::MEM_UB, "t3"), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_COPY_IN, {"t1"}, {"t2"}, "copy_in", true), true);
    EXPECT_EQ(graph.AddOp(Opcode::OP_RESHAPE, {"t2"}, {"t3"}, "reshape", true), true);
    
    auto *rootFuncPtr = graph.GetFunction();
    AxisCombineMarker marker;
    marker.Run(*rootFuncPtr);
    
    // Verify the tensor is marked as UNKNOWN
    auto t3 = graph.GetTensor("t3");
    EXPECT_EQ(marker.IsTensorEnableAxisCombine(t3), false);   // Unhandled op should be unknown
}