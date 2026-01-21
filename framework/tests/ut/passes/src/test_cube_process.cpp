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
 * \file test_split_k.cpp
 * \brief Unit test for SplitK pass.
 */

#include <fstream>
#include <vector>
#include <string>
#include "gtest/gtest.h"
#include "tilefwk/tilefwk_op.h"
#include "interface/function/function.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/program/program.h"
#include "passes/pass_mgr/pass_manager.h"
#include "interface/configs/config_manager.h"
#include "passes/tile_graph_pass/graph_optimization/split_k.h"
#include "passes/tile_graph_pass/graph_constraint/pre_graph/pre_graph.h"
#include "computational_graph_builder.h"
#include "ut_json/ut_json_tool.h"

using namespace npu::tile_fwk;

namespace npu {
namespace tile_fwk {
class SplitKTest : public testing::Test {
public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}

    void SetUp() override {
        Program::GetInstance().Reset();
        config::Reset();
        config::SetHostOption(COMPILE_STAGE, HOST_COMPILE_END);
        // config::SetHostConfig(KEY_STRATEGY, "SplitKTestStrategy");
        config::SetPlatformConfig(KEY_ENABLE_COST_MODEL, false);
    }

    void SetMatMulAttr(ComputationalGraphBuilder &G, const std::string name,
        bool isAtomic = false, const int nzFormat = 0) {
        auto op = G.GetOp(name);
        if (op == nullptr) {
            return;
        }
        if (isAtomic) {
            op->SetAttribute(ACC_A_MUL_B, 1);
        } else {
            op->SetAttribute(ACC_A_MUL_B, 0);
        }
        op->SetAttribute(MATMUL_NZ_ATTR, nzFormat);
        op->SetAttribute(A_MUL_B_ACT_M, 0L);
        op->SetAttribute(A_MUL_B_ACT_K, 0L);
        op->SetAttribute(A_MUL_B_ACT_N, 0L);
    }

    void SetMatmulMatrixSize(ComputationalGraphBuilder &G, const std::string name, const std::vector<int64_t> &matrixSize) {
        auto op = G.GetOp(name);
        op->SetAttribute(A_MUL_B_ACT_M, matrixSize[0]);
        op->SetAttribute(A_MUL_B_ACT_K, matrixSize[1]);
        op->SetAttribute(A_MUL_B_ACT_N, matrixSize[2]);
    }

    void CheckL0cType(DataType inputAstDtype, DataType outputAstDtype, DataType l0cDtype) {
        ComputationalGraphBuilder G;
        // add tensor
        G.AddTensor(inputAstDtype, {64, 128}, "mat_a");
        auto mat_a = G.GetTensor("mat_a");
        mat_a->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
        G.AddTensor(inputAstDtype, {128, 128}, "mat_b");
        auto mat_b = G.GetTensor("mat_b");
        mat_b->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
        G.AddTensor(outputAstDtype, {64, 128}, "mat_c");
        auto mat_c = G.GetTensor("mat_c");
        mat_c->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
        G.AddTensor(inputAstDtype, {64, 128}, "l1_a");
        auto l1_a = G.GetTensor("l1_a");
        l1_a->SetMemoryTypeBoth(MemoryType::MEM_L1, true);
        G.AddTensor(inputAstDtype, {64, 128}, "l0_a");
        auto l0_a = G.GetTensor("l0_a");
        l0_a->SetMemoryTypeBoth(MemoryType::MEM_L0A, true);
        G.AddTensor(inputAstDtype, {128, 128}, "l1_b");
        auto l1_b = G.GetTensor("l1_b");
        l1_b->SetMemoryTypeBoth(MemoryType::MEM_L1, true);
        G.AddTensor(inputAstDtype, {128, 128}, "l0_b");
        auto l0_b = G.GetTensor("l0_b");
        l0_b->SetMemoryTypeBoth(MemoryType::MEM_L0B, true);
        G.AddTensor(outputAstDtype, {64, 128}, "l0_c");
        auto l0_c = G.GetTensor("l0_c");
        l0_c->SetMemoryTypeBoth(MemoryType::MEM_L0C, true);
        // add op
        G.AddOp(Opcode::OP_COPY_IN, {"mat_a"}, {"l1_a"}, "L1_Copy_In_A");
        G.AddOp(Opcode::OP_COPY_IN, {"mat_b"}, {"l1_b"}, "L1_Copy_In_B");
        G.AddOp(Opcode::OP_L1_TO_L0A, {"l1_a"}, {"l0_a"}, "L1_To_L0A");
        G.AddOp(Opcode::OP_L1_TO_L0B, {"l1_b"}, {"l0_b"}, "L1_To_L0B");
        G.AddOp(Opcode::OP_A_MUL_B, {"l0_a", "l0_b"}, {"l0_c"}, "A_MUL_B");
        SetMatMulAttr(G, "A_MUL_B", false, 0);
        G.AddOp(Opcode::OP_COPY_OUT, {"l0_c"}, {"mat_c"}, "L0C_Copy_out");
        // set incast and outcast
        G.SetInCast({"mat_a", "mat_b"});
        G.SetOutCast({"mat_c"});
        // check before pass
        auto l0cBefore = G.GetTensor("l0_c");
        EXPECT_EQ(l0cBefore->Datatype(), outputAstDtype);
        // run pass
        Function *function = G.GetFunction();
        EXPECT_NE(function, nullptr);
        SplitK passLocal;
        passLocal.Run(*function, "", "", 0);
        CubeProcess cubeProcess;
        cubeProcess.UpdateCubeOp(*function);
        // check after pass
        auto l0cAfter = G.GetTensor("l0_c");
        EXPECT_EQ(l0cAfter->Datatype(), l0cDtype);
    }
    void TearDown() override {}
};

TEST_F(SplitKTest, Test_MM_FP16) {
    CheckL0cType(DataType::DT_FP16, DataType::DT_FP16, DataType::DT_FP32);
}

TEST_F(SplitKTest, Test_MM_BF16) {
    CheckL0cType(DataType::DT_BF16, DataType::DT_BF16, DataType::DT_FP32);
}

TEST_F(SplitKTest, Test_MM_FP32) {
    CheckL0cType(DataType::DT_FP32, DataType::DT_FP32, DataType::DT_FP32);
}

TEST_F(SplitKTest, Test_MM_INT8) {
    CheckL0cType(DataType::DT_INT8, DataType::DT_INT8, DataType::DT_INT32);
}

TEST_F(SplitKTest, Test_MM_INT16) {
    CheckL0cType(DataType::DT_INT16, DataType::DT_INT16, DataType::DT_INT32);
}

TEST_F(SplitKTest, Test_MM_INT32) {
    CheckL0cType(DataType::DT_INT32, DataType::DT_INT32, DataType::DT_INT32);
}

TEST_F(SplitKTest, TestReducAccSplitKOn) {
    ComputationalGraphBuilder G;
    // add tensor
    DataType inputAstDtype = DataType::DT_FP16;
    DataType outputAstDtype = DataType::DT_FP16;
    G.AddTensor(inputAstDtype, {64, 128}, "mat_a");
    auto mat_a = G.GetTensor("mat_a");
    mat_a->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    G.AddTensor(inputAstDtype, {128, 128}, "mat_b");
    auto mat_b = G.GetTensor("mat_b");
    mat_b->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    G.AddTensor(outputAstDtype, {64, 128}, "mat_c_before_reduce_acc_0");
    auto mat_c_before_reduce_acc_0 = G.GetTensor("mat_c_before_reduce_acc_0");
    mat_c_before_reduce_acc_0->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    G.AddTensor(outputAstDtype, {64, 128}, "mat_c_before_reduce_acc_1");
    auto mat_c_before_reduce_acc_1 = G.GetTensor("mat_c_before_reduce_acc_1");
    mat_c_before_reduce_acc_1->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    G.AddTensor(outputAstDtype, {64, 128}, "mat_c_before_reduce_acc_2");
    auto mat_c_before_reduce_acc_2 = G.GetTensor("mat_c_before_reduce_acc_2");
    mat_c_before_reduce_acc_2->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    G.AddTensor(outputAstDtype, {64, 128}, "mat_c_before_reduce_acc_3");
    auto mat_c_before_reduce_acc_3 = G.GetTensor("mat_c_before_reduce_acc_3");
    mat_c_before_reduce_acc_3->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    G.AddTensor(outputAstDtype, {64, 128}, "mat_c_after_reduce_acc");
    auto mat_c_after_reduce_acc = G.GetTensor("mat_c_after_reduce_acc");
    mat_c_after_reduce_acc->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    G.AddTensors(inputAstDtype, {64, 128}, {"l1_a_0", "l1_a_1", "l1_a_2", "l1_a_3"});
    G.AddTensors(inputAstDtype, {128, 128}, {"l1_b_0", "l1_b_1", "l1_b_2", "l1_b_3"});
    auto l1_a_0 = G.GetTensor("l1_a_0");
    l1_a_0->SetMemoryTypeBoth(MemoryType::MEM_L1, true);
    auto l1_a_1 = G.GetTensor("l1_a_1");
    l1_a_1->SetMemoryTypeBoth(MemoryType::MEM_L1, true);
    auto l1_a_2 = G.GetTensor("l1_a_2");
    l1_a_2->SetMemoryTypeBoth(MemoryType::MEM_L1, true);
    auto l1_a_3 = G.GetTensor("l1_a_3");
    l1_a_3->SetMemoryTypeBoth(MemoryType::MEM_L1, true);
    auto l1_b_0 = G.GetTensor("l1_b_0");
    l1_b_0->SetMemoryTypeBoth(MemoryType::MEM_L1, true);
    auto l1_b_1 = G.GetTensor("l1_b_1");
    l1_b_1->SetMemoryTypeBoth(MemoryType::MEM_L1, true);
    auto l1_b_2 = G.GetTensor("l1_b_2");
    l1_b_2->SetMemoryTypeBoth(MemoryType::MEM_L1, true);
    auto l1_b_3 = G.GetTensor("l1_b_3");
    l1_b_3->SetMemoryTypeBoth(MemoryType::MEM_L1, true);
    G.AddTensors(inputAstDtype, {64, 128}, {"l0_a_0", "l0_a_1", "l0_a_2", "l0_a_3"});
    G.AddTensors(inputAstDtype, {128, 128}, {"l0_b_0", "l0_b_1", "l0_b_2", "l0_b_3"});
    G.AddTensors(outputAstDtype, {64, 128}, {"l0_c_0", "l0_c_1", "l0_c_2", "l0_c_3"});
    auto l0_a_0 = G.GetTensor("l0_a_0");
    l0_a_0->SetMemoryTypeBoth(MemoryType::MEM_L0A, true);
    auto l0_a_1 = G.GetTensor("l0_a_1");
    l0_a_1->SetMemoryTypeBoth(MemoryType::MEM_L0A, true);
    auto l0_a_2 = G.GetTensor("l0_a_2");
    l0_a_2->SetMemoryTypeBoth(MemoryType::MEM_L0A, true);
    auto l0_a_3 = G.GetTensor("l0_a_3");
    l0_a_3->SetMemoryTypeBoth(MemoryType::MEM_L0A, true);
    auto l0_b_0 = G.GetTensor("l0_b_0");
    l0_b_0->SetMemoryTypeBoth(MemoryType::MEM_L0B, true);
    auto l0_b_1 = G.GetTensor("l0_b_1");
    l0_b_1->SetMemoryTypeBoth(MemoryType::MEM_L0B, true);
    auto l0_b_2 = G.GetTensor("l0_b_2");
    l0_b_2->SetMemoryTypeBoth(MemoryType::MEM_L0B, true);
    auto l0_b_3 = G.GetTensor("l0_b_3");
    l0_b_3->SetMemoryTypeBoth(MemoryType::MEM_L0B, true);
    auto l0_c_0 = G.GetTensor("l0_c_0");
    l0_c_0->SetMemoryTypeBoth(MemoryType::MEM_L0C, true);
    auto l0_c_1 = G.GetTensor("l0_c_1");
    l0_c_1->SetMemoryTypeBoth(MemoryType::MEM_L0C, true);
    auto l0_c_2 = G.GetTensor("l0_c_2");
    l0_c_2->SetMemoryTypeBoth(MemoryType::MEM_L0C, true);
    auto l0_c_3 = G.GetTensor("l0_c_3");
    l0_c_3->SetMemoryTypeBoth(MemoryType::MEM_L0C, true);
    // add op
    G.AddOp(Opcode::OP_COPY_IN, {"mat_a"}, {"l1_a_0"}, "L1_Copy_In_A_0");
    G.AddOp(Opcode::OP_COPY_IN, {"mat_a"}, {"l1_a_1"}, "L1_Copy_In_A_1");
    G.AddOp(Opcode::OP_COPY_IN, {"mat_a"}, {"l1_a_2"}, "L1_Copy_In_A_2");
    G.AddOp(Opcode::OP_COPY_IN, {"mat_a"}, {"l1_a_3"}, "L1_Copy_In_A_3");
    G.AddOp(Opcode::OP_COPY_IN, {"mat_b"}, {"l1_b_0"}, "L1_Copy_In_B_0");
    G.AddOp(Opcode::OP_COPY_IN, {"mat_b"}, {"l1_b_1"}, "L1_Copy_In_B_1");
    G.AddOp(Opcode::OP_COPY_IN, {"mat_b"}, {"l1_b_2"}, "L1_Copy_In_B_2");
    G.AddOp(Opcode::OP_COPY_IN, {"mat_b"}, {"l1_b_3"}, "L1_Copy_In_B_3");
    G.AddOp(Opcode::OP_L1_TO_L0A, {"l1_a_0"}, {"l0_a_0"}, "L1_To_L0A_0");
    G.AddOp(Opcode::OP_L1_TO_L0A, {"l1_a_1"}, {"l0_a_1"}, "L1_To_L0A_1");
    G.AddOp(Opcode::OP_L1_TO_L0A, {"l1_a_2"}, {"l0_a_2"}, "L1_To_L0A_2");
    G.AddOp(Opcode::OP_L1_TO_L0A, {"l1_a_3"}, {"l0_a_3"}, "L1_To_L0A_3");
    G.AddOp(Opcode::OP_L1_TO_L0B, {"l1_b_0"}, {"l0_b_0"}, "L1_To_L0B_0");
    G.AddOp(Opcode::OP_L1_TO_L0B, {"l1_b_1"}, {"l0_b_1"}, "L1_To_L0B_1");
    G.AddOp(Opcode::OP_L1_TO_L0B, {"l1_b_2"}, {"l0_b_2"}, "L1_To_L0B_2");
    G.AddOp(Opcode::OP_L1_TO_L0B, {"l1_b_3"}, {"l0_b_3"}, "L1_To_L0B_3");
    G.AddOp(Opcode::OP_A_MUL_B, {"l0_a_0", "l0_b_0"}, {"l0_c_0"}, "A_MUL_B_0");
    G.AddOp(Opcode::OP_A_MUL_B, {"l0_a_1", "l0_b_1"}, {"l0_c_1"}, "A_MUL_B_1");
    G.AddOp(Opcode::OP_A_MUL_B, {"l0_a_2", "l0_b_2"}, {"l0_c_2"}, "A_MUL_B_2");
    G.AddOp(Opcode::OP_A_MUL_B, {"l0_a_3", "l0_b_3"}, {"l0_c_3"}, "A_MUL_B_3");
    G.AddOp(Opcode::OP_COPY_OUT, {"l0_c_0"}, {"mat_c_before_reduce_acc_0"}, "L0C_Copy_out_0");
    G.AddOp(Opcode::OP_COPY_OUT, {"l0_c_1"}, {"mat_c_before_reduce_acc_1"}, "L0C_Copy_out_1");
    G.AddOp(Opcode::OP_COPY_OUT, {"l0_c_2"}, {"mat_c_before_reduce_acc_2"}, "L0C_Copy_out_2");
    G.AddOp(Opcode::OP_COPY_OUT, {"l0_c_3"}, {"mat_c_before_reduce_acc_3"}, "L0C_Copy_out_3");
    G.AddOp(Opcode::OP_REDUCE_ACC, {"mat_c_before_reduce_acc_0", "mat_c_before_reduce_acc_1", "mat_c_before_reduce_acc_2", "mat_c_before_reduce_acc_3"}, {"mat_c_after_reduce_acc"}, "Reduce_Acc");
    SetMatMulAttr(G, "A_MUL_B_0", false, 0);
    SetMatMulAttr(G, "A_MUL_B_1", false, 0);
    SetMatMulAttr(G, "A_MUL_B_2", false, 0);
    SetMatMulAttr(G, "A_MUL_B_3", false, 0);
    // set incast and outcast
    G.SetInCast({"mat_a", "mat_b"});
    G.SetOutCast({"mat_c_after_reduce_acc"});
    // check before pass
    Function *function = G.GetFunction();
    EXPECT_NE(function, nullptr);
    int opReduceAccCount = 0;
    for(auto &op : function->Operations()) {
        if(op.GetOpcode() == Opcode::OP_REDUCE_ACC) {
            opReduceAccCount++;
        }
    }
    EXPECT_NE(opReduceAccCount, 0);
    // run pass
    SplitK passLocal;
    passLocal.Run(*function, "", "", 0);
    // check after pass
    opReduceAccCount = 0;
    for(auto &op : function->Operations()) {
        if(op.GetOpcode() == Opcode::OP_REDUCE_ACC) {
            opReduceAccCount++;
        }
    }
    EXPECT_EQ(opReduceAccCount, 0);
}


TEST_F(SplitKTest, TestReducAccSplitKOff) {
    ComputationalGraphBuilder G;
    // add tensor
    DataType inputAstDtype = DataType::DT_FP16;
    DataType outputAstDtype = DataType::DT_FP16;
    G.AddTensor(inputAstDtype, {64, 128}, "mat_a");
    auto mat_a = G.GetTensor("mat_a");
    mat_a->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    G.AddTensor(inputAstDtype, {128, 128}, "mat_b");
    auto mat_b = G.GetTensor("mat_b");
    mat_b->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    G.AddTensor(outputAstDtype, {64, 128}, "mat_c_before_reduce_acc_0");
    auto mat_c_before_reduce_acc_0 = G.GetTensor("mat_c_before_reduce_acc_0");
    mat_c_before_reduce_acc_0->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    G.AddTensor(outputAstDtype, {64, 128}, "mat_c_before_reduce_acc_1");
    auto mat_c_before_reduce_acc_1 = G.GetTensor("mat_c_before_reduce_acc_1");
    mat_c_before_reduce_acc_1->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    G.AddTensor(outputAstDtype, {64, 128}, "mat_c_before_reduce_acc_2");
    auto mat_c_before_reduce_acc_2 = G.GetTensor("mat_c_before_reduce_acc_2");
    mat_c_before_reduce_acc_2->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    G.AddTensor(outputAstDtype, {64, 128}, "mat_c_before_reduce_acc_3");
    auto mat_c_before_reduce_acc_3 = G.GetTensor("mat_c_before_reduce_acc_3");
    mat_c_before_reduce_acc_3->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    G.AddTensors(inputAstDtype, {64, 128}, {"l1_a_0", "l1_a_1", "l1_a_2", "l1_a_3"});
    G.AddTensors(inputAstDtype, {128, 128}, {"l1_b_0", "l1_b_1", "l1_b_2", "l1_b_3"});
    auto l1_a_0 = G.GetTensor("l1_a_0");
    l1_a_0->SetMemoryTypeBoth(MemoryType::MEM_L1, true);
    auto l1_a_1 = G.GetTensor("l1_a_1");
    l1_a_1->SetMemoryTypeBoth(MemoryType::MEM_L1, true);
    auto l1_a_2 = G.GetTensor("l1_a_2");
    l1_a_2->SetMemoryTypeBoth(MemoryType::MEM_L1, true);
    auto l1_a_3 = G.GetTensor("l1_a_3");
    l1_a_3->SetMemoryTypeBoth(MemoryType::MEM_L1, true);
    auto l1_b_0 = G.GetTensor("l1_b_0");
    l1_b_0->SetMemoryTypeBoth(MemoryType::MEM_L1, true);
    auto l1_b_1 = G.GetTensor("l1_b_1");
    l1_b_1->SetMemoryTypeBoth(MemoryType::MEM_L1, true);
    auto l1_b_2 = G.GetTensor("l1_b_2");
    l1_b_2->SetMemoryTypeBoth(MemoryType::MEM_L1, true);
    auto l1_b_3 = G.GetTensor("l1_b_3");
    l1_b_3->SetMemoryTypeBoth(MemoryType::MEM_L1, true);
    G.AddTensors(inputAstDtype, {64, 128}, {"l0_a_0", "l0_a_1", "l0_a_2", "l0_a_3"});
    G.AddTensors(inputAstDtype, {128, 128}, {"l0_b_0", "l0_b_1", "l0_b_2", "l0_b_3"});
    G.AddTensors(outputAstDtype, {64, 128}, {"l0_c_0", "l0_c_1", "l0_c_2", "l0_c_3"});
    auto l0_a_0 = G.GetTensor("l0_a_0");
    l0_a_0->SetMemoryTypeBoth(MemoryType::MEM_L0A, true);
    auto l0_a_1 = G.GetTensor("l0_a_1");
    l0_a_1->SetMemoryTypeBoth(MemoryType::MEM_L0A, true);
    auto l0_a_2 = G.GetTensor("l0_a_2");
    l0_a_2->SetMemoryTypeBoth(MemoryType::MEM_L0A, true);
    auto l0_a_3 = G.GetTensor("l0_a_3");
    l0_a_3->SetMemoryTypeBoth(MemoryType::MEM_L0A, true);
    auto l0_b_0 = G.GetTensor("l0_b_0");
    l0_b_0->SetMemoryTypeBoth(MemoryType::MEM_L0B, true);
    auto l0_b_1 = G.GetTensor("l0_b_1");
    l0_b_1->SetMemoryTypeBoth(MemoryType::MEM_L0B, true);
    auto l0_b_2 = G.GetTensor("l0_b_2");
    l0_b_2->SetMemoryTypeBoth(MemoryType::MEM_L0B, true);
    auto l0_b_3 = G.GetTensor("l0_b_3");
    l0_b_3->SetMemoryTypeBoth(MemoryType::MEM_L0B, true);
    auto l0_c_0 = G.GetTensor("l0_c_0");
    l0_c_0->SetMemoryTypeBoth(MemoryType::MEM_L0C, true);
    auto l0_c_1 = G.GetTensor("l0_c_1");
    l0_c_1->SetMemoryTypeBoth(MemoryType::MEM_L0C, true);
    auto l0_c_2 = G.GetTensor("l0_c_2");
    l0_c_2->SetMemoryTypeBoth(MemoryType::MEM_L0C, true);
    auto l0_c_3 = G.GetTensor("l0_c_3");
    l0_c_3->SetMemoryTypeBoth(MemoryType::MEM_L0C, true);
    // add op
    G.AddOp(Opcode::OP_COPY_IN, {"mat_a"}, {"l1_a_0"}, "L1_Copy_In_A_0");
    G.AddOp(Opcode::OP_COPY_IN, {"mat_a"}, {"l1_a_1"}, "L1_Copy_In_A_1");
    G.AddOp(Opcode::OP_COPY_IN, {"mat_a"}, {"l1_a_2"}, "L1_Copy_In_A_2");
    G.AddOp(Opcode::OP_COPY_IN, {"mat_a"}, {"l1_a_3"}, "L1_Copy_In_A_3");
    G.AddOp(Opcode::OP_COPY_IN, {"mat_b"}, {"l1_b_0"}, "L1_Copy_In_B_0");
    G.AddOp(Opcode::OP_COPY_IN, {"mat_b"}, {"l1_b_1"}, "L1_Copy_In_B_1");
    G.AddOp(Opcode::OP_COPY_IN, {"mat_b"}, {"l1_b_2"}, "L1_Copy_In_B_2");
    G.AddOp(Opcode::OP_COPY_IN, {"mat_b"}, {"l1_b_3"}, "L1_Copy_In_B_3");
    G.AddOp(Opcode::OP_L1_TO_L0A, {"l1_a_0"}, {"l0_a_0"}, "L1_To_L0A_0");
    G.AddOp(Opcode::OP_L1_TO_L0A, {"l1_a_1"}, {"l0_a_1"}, "L1_To_L0A_1");
    G.AddOp(Opcode::OP_L1_TO_L0A, {"l1_a_2"}, {"l0_a_2"}, "L1_To_L0A_2");
    G.AddOp(Opcode::OP_L1_TO_L0A, {"l1_a_3"}, {"l0_a_3"}, "L1_To_L0A_3");
    G.AddOp(Opcode::OP_L1_TO_L0B, {"l1_b_0"}, {"l0_b_0"}, "L1_To_L0B_0");
    G.AddOp(Opcode::OP_L1_TO_L0B, {"l1_b_1"}, {"l0_b_1"}, "L1_To_L0B_1");
    G.AddOp(Opcode::OP_L1_TO_L0B, {"l1_b_2"}, {"l0_b_2"}, "L1_To_L0B_2");
    G.AddOp(Opcode::OP_L1_TO_L0B, {"l1_b_3"}, {"l0_b_3"}, "L1_To_L0B_3");
    G.AddOp(Opcode::OP_A_MUL_B, {"l0_a_0", "l0_b_0"}, {"l0_c_0"}, "A_MUL_B_0");
    G.AddOp(Opcode::OP_A_MUL_B, {"l0_a_1", "l0_b_1"}, {"l0_c_1"}, "A_MUL_B_1");
    G.AddOp(Opcode::OP_A_MUL_B, {"l0_a_2", "l0_b_2"}, {"l0_c_2"}, "A_MUL_B_2");
    G.AddOp(Opcode::OP_A_MUL_B, {"l0_a_3", "l0_b_3"}, {"l0_c_3"}, "A_MUL_B_3");
    G.AddOp(Opcode::OP_COPY_OUT, {"l0_c_0"}, {"mat_c_before_reduce_acc_0"}, "L0C_Copy_out_0");
    G.AddOp(Opcode::OP_COPY_OUT, {"l0_c_1"}, {"mat_c_before_reduce_acc_1"}, "L0C_Copy_out_1");
    G.AddOp(Opcode::OP_COPY_OUT, {"l0_c_2"}, {"mat_c_before_reduce_acc_2"}, "L0C_Copy_out_2");
    G.AddOp(Opcode::OP_COPY_OUT, {"l0_c_3"}, {"mat_c_before_reduce_acc_3"}, "L0C_Copy_out_3");
    SetMatMulAttr(G, "A_MUL_B_0", false, 0);
    SetMatMulAttr(G, "A_MUL_B_1", false, 0);
    SetMatMulAttr(G, "A_MUL_B_2", false, 0);
    SetMatMulAttr(G, "A_MUL_B_3", false, 0);
    // set incast and outcast
    G.SetInCast({"mat_a", "mat_b"});
    G.SetOutCast({"mat_c_before_reduce_acc_0", "mat_c_before_reduce_acc_1", "mat_c_before_reduce_acc_2", "mat_c_before_reduce_acc_3"});
    // check before pass
    Function *function = G.GetFunction();
    EXPECT_NE(function, nullptr);
    int opReduceAccCount = 0;
    int opCountBefore = 0;
    for(auto &op : function->Operations()) {
        opCountBefore++;
        if(op.GetOpcode() == Opcode::OP_REDUCE_ACC) {
            opReduceAccCount++;
        }
    }
    EXPECT_EQ(opReduceAccCount, 0);
    // run pass
    SplitK passLocal;
    passLocal.Run(*function, "", "", 0);
    // check after pass
    opReduceAccCount = 0;
    int opCountAfter = 0;
    for(auto &op : function->Operations()) {
        opCountAfter++;
        if(op.GetOpcode() == Opcode::OP_REDUCE_ACC) {
            opReduceAccCount++;
        }
    }
    EXPECT_EQ(opReduceAccCount, 0);
    EXPECT_EQ(opCountBefore, opCountAfter);
}

TEST_F(SplitKTest, TestReducAccInputLess) {
    ComputationalGraphBuilder G;
    // add tensor
    DataType inputAstDtype = DataType::DT_FP16;
    DataType outputAstDtype = DataType::DT_FP16;
    G.AddTensor(inputAstDtype, {64, 128}, "mat_a");
    auto mat_a = G.GetTensor("mat_a");
    mat_a->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    G.AddTensor(inputAstDtype, {128, 128}, "mat_b");
    auto mat_b = G.GetTensor("mat_b");
    mat_b->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    G.AddTensor(outputAstDtype, {64, 128}, "mat_c_before_reduce_acc");
    auto mat_c_before_reduce_acc = G.GetTensor("mat_c_before_reduce_acc");
    mat_c_before_reduce_acc->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    G.AddTensor(outputAstDtype, {64, 128}, "mat_c_after_reduce_acc");
    auto mat_c_after_reduce_acc = G.GetTensor("mat_c_after_reduce_acc");
    mat_c_after_reduce_acc->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    G.AddTensor(inputAstDtype, {64, 128}, "l1_a");
    auto l1_a = G.GetTensor("l1_a");
    l1_a->SetMemoryTypeBoth(MemoryType::MEM_L1, true);
    G.AddTensor(inputAstDtype, {64, 128}, "l0_a");
    auto l0_a = G.GetTensor("l0_a");
    l0_a->SetMemoryTypeBoth(MemoryType::MEM_L0A, true);
    G.AddTensor(inputAstDtype, {128, 128}, "l1_b");
    auto l1_b = G.GetTensor("l1_b");
    l1_b->SetMemoryTypeBoth(MemoryType::MEM_L1, true);
    G.AddTensor(inputAstDtype, {128, 128}, "l0_b");
    auto l0_b = G.GetTensor("l0_b");
    l0_b->SetMemoryTypeBoth(MemoryType::MEM_L0B, true);
    G.AddTensor(outputAstDtype, {64, 128}, "l0_c");
    auto l0_c = G.GetTensor("l0_c");
    l0_c->SetMemoryTypeBoth(MemoryType::MEM_L0C, true);
    // add op
    G.AddOp(Opcode::OP_COPY_IN, {"mat_a"}, {"l1_a"}, "L1_Copy_In_A");
    G.AddOp(Opcode::OP_COPY_IN, {"mat_b"}, {"l1_b"}, "L1_Copy_In_B");
    G.AddOp(Opcode::OP_L1_TO_L0A, {"l1_a"}, {"l0_a"}, "L1_To_L0A");
    G.AddOp(Opcode::OP_L1_TO_L0B, {"l1_b"}, {"l0_b"}, "L1_To_L0B");
    G.AddOp(Opcode::OP_A_MUL_B, {"l0_a", "l0_b"}, {"l0_c"}, "A_MUL_B");
    G.AddOp(Opcode::OP_COPY_OUT, {"l0_c"}, {"mat_c_before_reduce_acc"}, "L0C_Copy_out");
    G.AddOp(Opcode::OP_REDUCE_ACC, {"mat_c_before_reduce_acc"}, {"mat_c_after_reduce_acc"}, "Reduce_Acc");
    SetMatMulAttr(G, "A_MUL_B", false, 0);
    // set incast and outcast
    G.SetInCast({"mat_a", "mat_b"});
    G.SetOutCast({"mat_c_after_reduce_acc"});
    // check before pass
    Function *function = G.GetFunction();
    EXPECT_NE(function, nullptr);
    int opReduceAccCount = 0;
    for(auto &op : function->Operations()) {
        if(op.GetOpcode() == Opcode::OP_REDUCE_ACC) {
            opReduceAccCount++;
        }
    }
    EXPECT_NE(opReduceAccCount, 0);
    // run pass
    SplitK passLocal;
    Status preCheckResult = passLocal.PreCheck(*function);
    EXPECT_EQ(preCheckResult, SUCCESS);
    passLocal.Run(*function, "", "", 0);
    // check after pass
    opReduceAccCount = 0;
    for(auto &op : function->Operations()) {
        if(op.GetOpcode() == Opcode::OP_REDUCE_ACC) {
            opReduceAccCount++;
        }
    }
    EXPECT_EQ(opReduceAccCount, 0);
}

TEST_F(SplitKTest, TestReducAccOutPutMore) {
    ComputationalGraphBuilder G;
    // add tensor
    DataType inputAstDtype = DataType::DT_FP16;
    DataType outputAstDtype = DataType::DT_FP16;
    G.AddTensor(inputAstDtype, {64, 128}, "mat_a");
    auto mat_a = G.GetTensor("mat_a");
    mat_a->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    G.AddTensor(inputAstDtype, {128, 128}, "mat_b");
    auto mat_b = G.GetTensor("mat_b");
    mat_b->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    G.AddTensor(outputAstDtype, {64, 128}, "mat_c_before_reduce_acc_0");
    auto mat_c_before_reduce_acc_0 = G.GetTensor("mat_c_before_reduce_acc_0");
    mat_c_before_reduce_acc_0->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    G.AddTensor(outputAstDtype, {64, 128}, "mat_c_before_reduce_acc_1");
    auto mat_c_before_reduce_acc_1 = G.GetTensor("mat_c_before_reduce_acc_1");
    mat_c_before_reduce_acc_1->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    G.AddTensor(outputAstDtype, {64, 128}, "mat_c_after_reduce_acc_0");
    auto mat_c_after_reduce_acc_0 = G.GetTensor("mat_c_after_reduce_acc_0");
    mat_c_after_reduce_acc_0->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    G.AddTensor(outputAstDtype, {64, 128}, "mat_c_after_reduce_acc_1");
    auto mat_c_after_reduce_acc_1 = G.GetTensor("mat_c_after_reduce_acc_1");
    mat_c_after_reduce_acc_1->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    G.AddTensor(outputAstDtype, {64, 128}, "mat_c_before_reduce_acc_2");
    auto mat_c_before_reduce_acc_2 = G.GetTensor("mat_c_before_reduce_acc_2");
    mat_c_before_reduce_acc_2->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    G.AddTensor(outputAstDtype, {64, 128}, "mat_c_before_reduce_acc_3");
    auto mat_c_before_reduce_acc_3 = G.GetTensor("mat_c_before_reduce_acc_3");
    mat_c_before_reduce_acc_3->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    G.AddTensors(inputAstDtype, {64, 128}, {"l1_a_0", "l1_a_1", "l1_a_2", "l1_a_3"});
    G.AddTensors(inputAstDtype, {128, 128}, {"l1_b_0", "l1_b_1", "l1_b_2", "l1_b_3"});
    auto l1_a_0 = G.GetTensor("l1_a_0");
    l1_a_0->SetMemoryTypeBoth(MemoryType::MEM_L1, true);
    auto l1_a_1 = G.GetTensor("l1_a_1");
    l1_a_1->SetMemoryTypeBoth(MemoryType::MEM_L1, true);
    auto l1_a_2 = G.GetTensor("l1_a_2");
    l1_a_2->SetMemoryTypeBoth(MemoryType::MEM_L1, true);
    auto l1_a_3 = G.GetTensor("l1_a_3");
    l1_a_3->SetMemoryTypeBoth(MemoryType::MEM_L1, true);
    auto l1_b_0 = G.GetTensor("l1_b_0");
    l1_b_0->SetMemoryTypeBoth(MemoryType::MEM_L1, true);
    auto l1_b_1 = G.GetTensor("l1_b_1");
    l1_b_1->SetMemoryTypeBoth(MemoryType::MEM_L1, true);
    auto l1_b_2 = G.GetTensor("l1_b_2");
    l1_b_2->SetMemoryTypeBoth(MemoryType::MEM_L1, true);
    auto l1_b_3 = G.GetTensor("l1_b_3");
    l1_b_3->SetMemoryTypeBoth(MemoryType::MEM_L1, true);
    G.AddTensors(inputAstDtype, {64, 128}, {"l0_a_0", "l0_a_1", "l0_a_2", "l0_a_3"});
    G.AddTensors(inputAstDtype, {128, 128}, {"l0_b_0", "l0_b_1", "l0_b_2", "l0_b_3"});
    G.AddTensors(outputAstDtype, {64, 128}, {"l0_c_0", "l0_c_1", "l0_c_2", "l0_c_3"});
    auto l0_a_0 = G.GetTensor("l0_a_0");
    l0_a_0->SetMemoryTypeBoth(MemoryType::MEM_L0A, true);
    auto l0_a_1 = G.GetTensor("l0_a_1");
    l0_a_1->SetMemoryTypeBoth(MemoryType::MEM_L0A, true);
    auto l0_a_2 = G.GetTensor("l0_a_2");
    l0_a_2->SetMemoryTypeBoth(MemoryType::MEM_L0A, true);
    auto l0_a_3 = G.GetTensor("l0_a_3");
    l0_a_3->SetMemoryTypeBoth(MemoryType::MEM_L0A, true);
    auto l0_b_0 = G.GetTensor("l0_b_0");
    l0_b_0->SetMemoryTypeBoth(MemoryType::MEM_L0B, true);
    auto l0_b_1 = G.GetTensor("l0_b_1");
    l0_b_1->SetMemoryTypeBoth(MemoryType::MEM_L0B, true);
    auto l0_b_2 = G.GetTensor("l0_b_2");
    l0_b_2->SetMemoryTypeBoth(MemoryType::MEM_L0B, true);
    auto l0_b_3 = G.GetTensor("l0_b_3");
    l0_b_3->SetMemoryTypeBoth(MemoryType::MEM_L0B, true);
    auto l0_c_0 = G.GetTensor("l0_c_0");
    l0_c_0->SetMemoryTypeBoth(MemoryType::MEM_L0C, true);
    auto l0_c_1 = G.GetTensor("l0_c_1");
    l0_c_1->SetMemoryTypeBoth(MemoryType::MEM_L0C, true);
    auto l0_c_2 = G.GetTensor("l0_c_2");
    l0_c_2->SetMemoryTypeBoth(MemoryType::MEM_L0C, true);
    auto l0_c_3 = G.GetTensor("l0_c_3");
    l0_c_3->SetMemoryTypeBoth(MemoryType::MEM_L0C, true);
    // add op
    G.AddOp(Opcode::OP_COPY_IN, {"mat_a"}, {"l1_a_0"}, "L1_Copy_In_A_0");
    G.AddOp(Opcode::OP_COPY_IN, {"mat_a"}, {"l1_a_1"}, "L1_Copy_In_A_1");
    G.AddOp(Opcode::OP_COPY_IN, {"mat_a"}, {"l1_a_2"}, "L1_Copy_In_A_2");
    G.AddOp(Opcode::OP_COPY_IN, {"mat_a"}, {"l1_a_3"}, "L1_Copy_In_A_3");
    G.AddOp(Opcode::OP_COPY_IN, {"mat_b"}, {"l1_b_0"}, "L1_Copy_In_B_0");
    G.AddOp(Opcode::OP_COPY_IN, {"mat_b"}, {"l1_b_1"}, "L1_Copy_In_B_1");
    G.AddOp(Opcode::OP_COPY_IN, {"mat_b"}, {"l1_b_2"}, "L1_Copy_In_B_2");
    G.AddOp(Opcode::OP_COPY_IN, {"mat_b"}, {"l1_b_3"}, "L1_Copy_In_B_3");
    G.AddOp(Opcode::OP_L1_TO_L0A, {"l1_a_0"}, {"l0_a_0"}, "L1_To_L0A_0");
    G.AddOp(Opcode::OP_L1_TO_L0A, {"l1_a_1"}, {"l0_a_1"}, "L1_To_L0A_1");
    G.AddOp(Opcode::OP_L1_TO_L0A, {"l1_a_2"}, {"l0_a_2"}, "L1_To_L0A_2");
    G.AddOp(Opcode::OP_L1_TO_L0A, {"l1_a_3"}, {"l0_a_3"}, "L1_To_L0A_3");
    G.AddOp(Opcode::OP_L1_TO_L0B, {"l1_b_0"}, {"l0_b_0"}, "L1_To_L0B_0");
    G.AddOp(Opcode::OP_L1_TO_L0B, {"l1_b_1"}, {"l0_b_1"}, "L1_To_L0B_1");
    G.AddOp(Opcode::OP_L1_TO_L0B, {"l1_b_2"}, {"l0_b_2"}, "L1_To_L0B_2");
    G.AddOp(Opcode::OP_L1_TO_L0B, {"l1_b_3"}, {"l0_b_3"}, "L1_To_L0B_3");
    G.AddOp(Opcode::OP_A_MUL_B, {"l0_a_0", "l0_b_0"}, {"l0_c_0"}, "A_MUL_B_0");
    G.AddOp(Opcode::OP_A_MUL_B, {"l0_a_1", "l0_b_1"}, {"l0_c_1"}, "A_MUL_B_1");
    G.AddOp(Opcode::OP_A_MUL_B, {"l0_a_2", "l0_b_2"}, {"l0_c_2"}, "A_MUL_B_2");
    G.AddOp(Opcode::OP_A_MUL_B, {"l0_a_3", "l0_b_3"}, {"l0_c_3"}, "A_MUL_B_3");
    G.AddOp(Opcode::OP_COPY_OUT, {"l0_c_0"}, {"mat_c_before_reduce_acc_0"}, "L0C_Copy_out_0");
    G.AddOp(Opcode::OP_COPY_OUT, {"l0_c_1"}, {"mat_c_before_reduce_acc_1"}, "L0C_Copy_out_1");
    G.AddOp(Opcode::OP_COPY_OUT, {"l0_c_2"}, {"mat_c_before_reduce_acc_2"}, "L0C_Copy_out_2");
    G.AddOp(Opcode::OP_COPY_OUT, {"l0_c_3"}, {"mat_c_before_reduce_acc_3"}, "L0C_Copy_out_3");
    G.AddOp(Opcode::OP_REDUCE_ACC, {"mat_c_before_reduce_acc_0", "mat_c_before_reduce_acc_1", "mat_c_before_reduce_acc_2", "mat_c_before_reduce_acc_3"}, {"mat_c_after_reduce_acc_0", "mat_c_after_reduce_acc_1"}, "Reduce_Acc");
    SetMatMulAttr(G, "A_MUL_B_0", false, 0);
    SetMatMulAttr(G, "A_MUL_B_1", false, 0);
    SetMatMulAttr(G, "A_MUL_B_2", false, 0);
    SetMatMulAttr(G, "A_MUL_B_3", false, 0);
    // set incast and outcast
    G.SetInCast({"mat_a", "mat_b"});
    G.SetOutCast({"mat_c_after_reduce_acc_0", "mat_c_after_reduce_acc_1"});
    // check before pass
    Function *function = G.GetFunction();
    EXPECT_NE(function, nullptr);
    int opReduceAccCount = 0;
    for(auto &op : function->Operations()) {
        if(op.GetOpcode() == Opcode::OP_REDUCE_ACC) {
            opReduceAccCount++;
        }
    }
    EXPECT_NE(opReduceAccCount, 0);
    // run pass
    SplitK passLocal;
    Status preCheckResult = passLocal.PreCheck(*function);
    EXPECT_NE(preCheckResult, SUCCESS);
}

TEST_F(SplitKTest, Test_MM_FP16_Atomic_On) {
    int m = 32;
    int n = 512;
    int k = 128;
    std::vector<int64_t> shape_a = {m, k};
    std::vector<int64_t> shape_b = {k, n};
    std::vector<int64_t> shape_c = {m, n};
    DataType inputAstDtype = DataType::DT_FP16;
    DataType outputAstDtype = DataType::DT_FP32;
    config::SetHostConfig(KEY_STRATEGY, "PVC2_OOO");

    PROGRAM("Test_MM_FP16_Atomic_On") {
        Tensor mat_a(inputAstDtype, shape_a, "mat_a");
        Tensor mat_b(inputAstDtype, shape_b, "mat_b");
        Tensor final_out(outputAstDtype, shape_c, "final_out");
        config::SetBuildStatic(true);
        FUNCTION("MM_FP16_Atomic_On", {mat_a, mat_b, final_out}) {
            TileShape::Current().SetCubeTile({32, 32}, {128, 128}, {64, 64}, false, true);
            auto tmpC = Matrix::Matmul(outputAstDtype, mat_a, mat_b, false, false);
            TileShape::Current().SetVecTile(32, 32);
            final_out = Add(tmpC, Element(DataType::DT_FP32, 0.0));
        }
    }
}

TEST_F(SplitKTest, TestAnzBnd) {
    ComputationalGraphBuilder G;
    // add tensor
    DataType inputAstDtype = DataType::DT_FP16;
    DataType outputAstDtype = DataType::DT_FP16;
    G.AddTensor(inputAstDtype, {64, 128}, "mat_a");
    auto mat_a = G.GetTensor("mat_a");
    mat_a->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    G.AddTensor(inputAstDtype, {128, 128}, "mat_b");
    auto mat_b = G.GetTensor("mat_b");
    mat_b->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    G.AddTensor(outputAstDtype, {64, 128}, "mat_c");
    auto mat_c = G.GetTensor("mat_c");
    mat_c->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    G.AddTensor(inputAstDtype, {64, 128}, "l1_a");
    auto l1_a = G.GetTensor("l1_a");
    l1_a->SetMemoryTypeBoth(MemoryType::MEM_L1, true);
    G.AddTensor(inputAstDtype, {64, 128}, "l0_a");
    auto l0_a = G.GetTensor("l0_a");
    l0_a->SetMemoryTypeBoth(MemoryType::MEM_L0A, true);
    G.AddTensor(inputAstDtype, {128, 128}, "l1_b");
    auto l1_b = G.GetTensor("l1_b");
    l1_b->SetMemoryTypeBoth(MemoryType::MEM_L1, true);
    G.AddTensor(inputAstDtype, {128, 128}, "l0_b");
    auto l0_b = G.GetTensor("l0_b");
    l0_b->SetMemoryTypeBoth(MemoryType::MEM_L0B, true);
    G.AddTensor(outputAstDtype, {64, 128}, "l0_c");
    auto l0_c = G.GetTensor("l0_c");
    l0_c->SetMemoryTypeBoth(MemoryType::MEM_L0C, true);
    // add op
    G.AddOp(Opcode::OP_COPY_IN, {"mat_a"}, {"l1_a"}, "L1_Copy_In_A");
    G.AddOp(Opcode::OP_COPY_IN, {"mat_b"}, {"l1_b"}, "L1_Copy_In_B");
    G.AddOp(Opcode::OP_L1_TO_L0A, {"l1_a"}, {"l0_a"}, "L1_To_L0A");
    G.AddOp(Opcode::OP_L1_TO_L0B, {"l1_b"}, {"l0_b"}, "L1_To_L0B");
    G.AddOp(Opcode::OP_A_MUL_B, {"l0_a", "l0_b"}, {"l0_c"}, "A_MUL_B");
    G.AddOp(Opcode::OP_COPY_OUT, {"l0_c"}, {"mat_c"}, "L0C_Copy_out");
    SetMatMulAttr(G, "A_MUL_B", false, 1);
    // set incast and outcast
    G.SetInCast({"mat_a", "mat_b"});
    G.SetOutCast({"mat_c"});
    // run pass
    Function *function = G.GetFunction();
    EXPECT_NE(function, nullptr);
    SplitK passLocal;
    passLocal.Run(*function, "", "", 0);
    CubeProcess cubeProcess;
    cubeProcess.UpdateCubeOp(*function);
    // check after pass
    auto opL1CopyInA = G.GetOp("L1_Copy_In_A");
    EXPECT_NE(opL1CopyInA, nullptr);
    EXPECT_EQ(opL1CopyInA->GetIntAttribute(COPY_IS_NZ), 1);
    auto opL1CopyInB = G.GetOp("L1_Copy_In_B");
    EXPECT_NE(opL1CopyInB, nullptr);
    EXPECT_EQ(opL1CopyInB->GetIntAttribute(COPY_IS_NZ), 0);
}

TEST_F(SplitKTest, TestAnzBndL1) {
    ComputationalGraphBuilder G;
    // add tensor
    DataType inputAstDtype = DataType::DT_FP16;
    DataType outputAstDtype = DataType::DT_FP16;
    G.AddTensor(inputAstDtype, {64, 128}, "mat_a");
    auto mat_a = G.GetTensor("mat_a");
    mat_a->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    G.AddTensor(inputAstDtype, {128, 128}, "mat_b");
    auto mat_b = G.GetTensor("mat_b");
    mat_b->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    G.AddTensor(outputAstDtype, {64, 128}, "mat_c_0");
    auto mat_c_0 = G.GetTensor("mat_c_0");
    mat_c_0->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    G.AddTensor(outputAstDtype, {64, 128}, "mat_c_1");
    auto mat_c_1 = G.GetTensor("mat_c_1");
    mat_c_1->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    G.AddTensor(inputAstDtype, {64, 128}, "l1_a");
    auto l1_a = G.GetTensor("l1_a");
    l1_a->SetMemoryTypeBoth(MemoryType::MEM_L1, true);
    G.AddTensor(inputAstDtype, {128, 128}, "l1_b");
    auto l1_b = G.GetTensor("l1_b");
    l1_b->SetMemoryTypeBoth(MemoryType::MEM_L1, true);
    G.AddTensor(inputAstDtype, {64, 128}, "l1_a_0");
    auto l1_a_0 = G.GetTensor("l1_a_0");
    l1_a_0->SetMemoryTypeBoth(MemoryType::MEM_L1, true);
    G.AddTensor(inputAstDtype, {128, 128}, "l1_b_0");
    auto l1_b_0 = G.GetTensor("l1_b_0");
    l1_b_0->SetMemoryTypeBoth(MemoryType::MEM_L1, true);
    G.AddTensor(inputAstDtype, {64, 128}, "l1_a_1");
    auto l1_a_1 = G.GetTensor("l1_a_1");
    l1_a_1->SetMemoryTypeBoth(MemoryType::MEM_L1, true);
    G.AddTensor(inputAstDtype, {128, 128}, "l1_b_1");
    auto l1_b_1 = G.GetTensor("l1_b_1");
    l1_b_1->SetMemoryTypeBoth(MemoryType::MEM_L1, true);
    G.AddTensor(inputAstDtype, {64, 128}, "l0_a_0");
    auto l0_a_0 = G.GetTensor("l0_a_0");
    l0_a_0->SetMemoryTypeBoth(MemoryType::MEM_L0A, true);
    G.AddTensor(inputAstDtype, {128, 128}, "l0_b_0");
    auto l0_b_0 = G.GetTensor("l0_b_0");
    l0_b_0->SetMemoryTypeBoth(MemoryType::MEM_L0B, true);
    G.AddTensor(outputAstDtype, {64, 128}, "l0_c_0");
    auto l0_c_0 = G.GetTensor("l0_c_0");
    l0_c_0->SetMemoryTypeBoth(MemoryType::MEM_L0C, true);
    G.AddTensor(inputAstDtype, {64, 128}, "l0_a_1");
    auto l0_a_1 = G.GetTensor("l0_a_1");
    l0_a_1->SetMemoryTypeBoth(MemoryType::MEM_L0A, true);
    G.AddTensor(inputAstDtype, {128, 128}, "l0_b_1");
    auto l0_b_1 = G.GetTensor("l0_b_1");
    l0_b_1->SetMemoryTypeBoth(MemoryType::MEM_L0B, true);
    G.AddTensor(outputAstDtype, {64, 128}, "l0_c_1");
    auto l0_c_1 = G.GetTensor("l0_c_1");
    l0_c_1->SetMemoryTypeBoth(MemoryType::MEM_L0C, true);
    // add op
    G.AddOp(Opcode::OP_COPY_IN, {"mat_a"}, {"l1_a"}, "L1_Copy_In_A");
    G.AddOp(Opcode::OP_COPY_IN, {"mat_b"}, {"l1_b"}, "L1_Copy_In_B");
    G.AddOp(Opcode::OP_VIEW, {"l1_a"}, {"l1_a_0","l1_a_1"}, "A_OP_VIEW");
    G.AddOp(Opcode::OP_VIEW, {"l1_b"}, {"l1_b_0","l1_b_1"}, "B_OP_VIEW");
    G.AddOp(Opcode::OP_L1_TO_L0A, {"l1_a_0"}, {"l0_a_0"}, "L1_To_L0A_0");
    G.AddOp(Opcode::OP_L1_TO_L0A, {"l1_a_1"}, {"l0_a_1"}, "L1_To_L0A_1");
    G.AddOp(Opcode::OP_L1_TO_L0B, {"l1_b_0"}, {"l0_b_0"}, "L1_To_L0B_0");
    G.AddOp(Opcode::OP_L1_TO_L0B, {"l1_b_1"}, {"l0_b_1"}, "L1_To_L0B_1");
    G.AddOp(Opcode::OP_A_MUL_B, {"l0_a_0", "l0_b_0"}, {"l0_c_0"}, "A_MUL_B_0");
    G.AddOp(Opcode::OP_A_MUL_B, {"l0_a_1", "l0_b_1"}, {"l0_c_1"}, "A_MUL_B_1");
    G.AddOp(Opcode::OP_COPY_OUT, {"l0_c_0"}, {"mat_c_0"}, "L0C_Copy_out_0");
    G.AddOp(Opcode::OP_COPY_OUT, {"l0_c_1"}, {"mat_c_1"}, "L0C_Copy_out_1");
    SetMatMulAttr(G, "A_MUL_B_0", false, 1);
    SetMatMulAttr(G, "A_MUL_B_1", false, 1);
    // set incast and outcast
    G.SetInCast({"mat_a", "mat_b"});
    G.SetOutCast({"mat_c_0", "mat_c_1"});
    // run pass
    Function *function = G.GetFunction();
    EXPECT_NE(function, nullptr);
    SplitK passLocal;
    passLocal.Run(*function, "", "", 0);
    CubeProcess cubeProcess;
    cubeProcess.UpdateCubeOp(*function);
    // check after pass
    auto opL1CopyInA = G.GetOp("L1_Copy_In_A");
    EXPECT_NE(opL1CopyInA, nullptr);
    EXPECT_EQ(opL1CopyInA->GetIntAttribute(COPY_IS_NZ), 1);
    auto opL1CopyInB = G.GetOp("L1_Copy_In_B");
    EXPECT_NE(opL1CopyInB, nullptr);
    EXPECT_EQ(opL1CopyInB->GetIntAttribute(COPY_IS_NZ), 0);
}

TEST_F(SplitKTest, TestAndBndCnz) {
    ComputationalGraphBuilder G;
    // add tensor
    DataType inputAstDtype = DataType::DT_FP16;
    DataType outputAstDtype = DataType::DT_FP16;
    G.AddTensor(inputAstDtype, {64, 128}, "mat_a");
    auto mat_a = G.GetTensor("mat_a");
    mat_a->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    G.AddTensor(inputAstDtype, {128, 128}, "mat_b");
    auto mat_b = G.GetTensor("mat_b");
    mat_b->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    G.AddTensor(outputAstDtype, {64, 128}, "mat_c_0");
    auto mat_c_0 = G.GetTensor("mat_c_0");
    mat_c_0->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    G.AddTensor(outputAstDtype, {64, 128}, "mat_c_1");
    auto mat_c_1 = G.GetTensor("mat_c_1");
    mat_c_1->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    G.AddTensor(inputAstDtype, {64, 128}, "l1_a");
    auto l1_a = G.GetTensor("l1_a");
    l1_a->SetMemoryTypeBoth(MemoryType::MEM_L1, true);
    G.AddTensor(inputAstDtype, {128, 128}, "l1_b");
    auto l1_b = G.GetTensor("l1_b");
    l1_b->SetMemoryTypeBoth(MemoryType::MEM_L1, true);
    G.AddTensor(inputAstDtype, {64, 128}, "l1_a_0");
    auto l1_a_0 = G.GetTensor("l1_a_0");
    l1_a_0->SetMemoryTypeBoth(MemoryType::MEM_L1, true);
    G.AddTensor(inputAstDtype, {128, 128}, "l1_b_0");
    auto l1_b_0 = G.GetTensor("l1_b_0");
    l1_b_0->SetMemoryTypeBoth(MemoryType::MEM_L1, true);
    G.AddTensor(inputAstDtype, {64, 128}, "l1_a_1");
    auto l1_a_1 = G.GetTensor("l1_a_1");
    l1_a_1->SetMemoryTypeBoth(MemoryType::MEM_L1, true);
    G.AddTensor(inputAstDtype, {128, 128}, "l1_b_1");
    auto l1_b_1 = G.GetTensor("l1_b_1");
    l1_b_1->SetMemoryTypeBoth(MemoryType::MEM_L1, true);
    G.AddTensor(inputAstDtype, {64, 128}, "l0_a_0");
    auto l0_a_0 = G.GetTensor("l0_a_0");
    l0_a_0->SetMemoryTypeBoth(MemoryType::MEM_L0A, true);
    G.AddTensor(inputAstDtype, {128, 128}, "l0_b_0");
    auto l0_b_0 = G.GetTensor("l0_b_0");
    l0_b_0->SetMemoryTypeBoth(MemoryType::MEM_L0B, true);
    G.AddTensor(outputAstDtype, {64, 128}, "l0_c_0");
    auto l0_c_0 = G.GetTensor("l0_c_0");
    l0_c_0->SetMemoryTypeBoth(MemoryType::MEM_L0C, true);
    G.AddTensor(inputAstDtype, {64, 128}, "l0_a_1");
    auto l0_a_1 = G.GetTensor("l0_a_1");
    l0_a_1->SetMemoryTypeBoth(MemoryType::MEM_L0A, true);
    G.AddTensor(inputAstDtype, {128, 128}, "l0_b_1");
    auto l0_b_1 = G.GetTensor("l0_b_1");
    l0_b_1->SetMemoryTypeBoth(MemoryType::MEM_L0B, true);
    G.AddTensor(outputAstDtype, {64, 128}, "l0_c_1");
    auto l0_c_1 = G.GetTensor("l0_c_1");
    l0_c_1->SetMemoryTypeBoth(MemoryType::MEM_L0C, true);
    // add op
    G.AddOp(Opcode::OP_COPY_IN, {"mat_a"}, {"l1_a"}, "L1_Copy_In_A");
    G.AddOp(Opcode::OP_COPY_IN, {"mat_b"}, {"l1_b"}, "L1_Copy_In_B");
    G.AddOp(Opcode::OP_VIEW, {"l1_a"}, {"l1_a_0","l1_a_1"}, "A_OP_VIEW");
    G.AddOp(Opcode::OP_VIEW, {"l1_b"}, {"l1_b_0","l1_b_1"}, "B_OP_VIEW");
    G.AddOp(Opcode::OP_L1_TO_L0A, {"l1_a_0"}, {"l0_a_0"}, "L1_To_L0A_0");
    G.AddOp(Opcode::OP_L1_TO_L0A, {"l1_a_1"}, {"l0_a_1"}, "L1_To_L0A_1");
    G.AddOp(Opcode::OP_L1_TO_L0B, {"l1_b_0"}, {"l0_b_0"}, "L1_To_L0B_0");
    G.AddOp(Opcode::OP_L1_TO_L0B, {"l1_b_1"}, {"l0_b_1"}, "L1_To_L0B_1");
    G.AddOp(Opcode::OP_A_MUL_B, {"l0_a_0", "l0_b_0"}, {"l0_c_0"}, "A_MUL_B_0");
    G.AddOp(Opcode::OP_A_MUL_B, {"l0_a_1", "l0_b_1"}, {"l0_c_1"}, "A_MUL_B_1");
    G.AddOp(Opcode::OP_COPY_OUT, {"l0_c_0"}, {"mat_c_0"}, "L0C_Copy_out_0");
    G.AddOp(Opcode::OP_COPY_OUT, {"l0_c_1"}, {"mat_c_1"}, "L0C_Copy_out_1");
    SetMatMulAttr(G, "A_MUL_B_0", false, 4);
    SetMatmulMatrixSize(G, "A_MUL_B_0", {64, 128, 128});
    SetMatMulAttr(G, "A_MUL_B_1", false, 4);
    SetMatmulMatrixSize(G, "A_MUL_B_1", {64, 128, 128});
    // set incast and outcast
    G.SetInCast({"mat_a", "mat_b"});
    G.SetOutCast({"mat_c_0", "mat_c_1"});
    // run pass
    Function *function = G.GetFunction();
    EXPECT_NE(function, nullptr);
    SplitK passLocal;
    passLocal.Run(*function, "", "", 0);
    CubeProcess cubeProcess;
    cubeProcess.UpdateCubeOp(*function);
    // check after pass
    auto opL1CopyInA = G.GetOp("L1_Copy_In_A");
    EXPECT_NE(opL1CopyInA, nullptr);
    EXPECT_EQ(opL1CopyInA->GetIntAttribute(COPY_IS_NZ), 0);
    auto opL1CopyInB = G.GetOp("L1_Copy_In_B");
    EXPECT_NE(opL1CopyInB, nullptr);
    EXPECT_EQ(opL1CopyInB->GetIntAttribute(COPY_IS_NZ), 0);
    auto opL0cCopyOut0 = G.GetOp("L0C_Copy_out_0");
    EXPECT_NE(opL0cCopyOut0, nullptr);
    EXPECT_EQ(opL0cCopyOut0->GetIntAttribute(COPY_IS_NZ), 1);
    EXPECT_EQ(opL0cCopyOut0->GetIntAttribute(L0C_COPY_OUT_OUTER), 64);
    EXPECT_EQ(opL0cCopyOut0->GetIntAttribute(L0C_COPY_OUT_INNER), 128);
    auto opL0cCopyOut1 = G.GetOp("L0C_Copy_out_1");
    EXPECT_NE(opL0cCopyOut1, nullptr);
    EXPECT_EQ(opL0cCopyOut1->GetIntAttribute(COPY_IS_NZ), 1);
    EXPECT_EQ(opL0cCopyOut1->GetIntAttribute(L0C_COPY_OUT_OUTER), 64);
    EXPECT_EQ(opL0cCopyOut1->GetIntAttribute(L0C_COPY_OUT_INNER), 128);
}

TEST_F(SplitKTest, TestGatherOnL1) {
    ComputationalGraphBuilder G;
    // INCAST mat_a, mat_b, OUTCAST mat_c
    DataType inputAstDtype = DataType::DT_FP16;
    DataType outputAstDtype = DataType::DT_FP16;
    G.AddTensor(inputAstDtype, {1024, 128}, "mat_a");
    auto mat_a = G.GetTensor("mat_a");
    mat_a->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    G.AddTensor(inputAstDtype, {128, 128}, "mat_b");
    auto mat_b = G.GetTensor("mat_b");
    mat_b->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    G.AddTensor(inputAstDtype, {128, 128}, "mat_c");
    auto mat_c = G.GetTensor("mat_c");
    mat_c->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    // paritial mat_a on L1
    G.AddTensor(outputAstDtype, {64, 64}, "mat_a_partial_0");
    auto mat_a_partial_0 = G.GetTensor("mat_a_partial_0");
    mat_a_partial_0->SetMemoryTypeBoth(MemoryType::MEM_L1, true);
    G.AddTensor(outputAstDtype, {64, 64}, "mat_a_partial_1");
    auto mat_a_partial_1 = G.GetTensor("mat_a_partial_1");
    mat_a_partial_1->SetMemoryTypeBoth(MemoryType::MEM_L1, true);
    G.AddTensor(outputAstDtype, {64, 64}, "mat_a_partial_2");
    auto mat_a_partial_2 = G.GetTensor("mat_a_partial_2");
    mat_a_partial_2->SetMemoryTypeBoth(MemoryType::MEM_L1, true);
    G.AddTensor(outputAstDtype, {64, 64}, "mat_a_partial_3");
    auto mat_a_partial_3 = G.GetTensor("mat_a_partial_3");
    mat_a_partial_3->SetMemoryTypeBoth(MemoryType::MEM_L1, true);
    // Assemble on L1
    G.AddTensor(outputAstDtype, {128, 128}, "mat_a_L1");
    auto mat_a_L1 = G.GetTensor("mat_a_L1");
    mat_a_L1->SetMemoryTypeBoth(MemoryType::MEM_L1, true);
    G.AddTensor(outputAstDtype, {128, 128}, "mat_b_L1");
    auto mat_b_L1 = G.GetTensor("mat_b_L1");
    mat_b_L1->SetMemoryTypeBoth(MemoryType::MEM_L1, true);
    // L0
    G.AddTensor(outputAstDtype, {128, 128}, "mat_a_L0");
    auto mat_a_L0 = G.GetTensor("mat_a_L0");
    mat_a_L0->SetMemoryTypeBoth(MemoryType::MEM_L0A, true);
    G.AddTensor(outputAstDtype, {128, 128}, "mat_b_L0");
    auto mat_b_L0 = G.GetTensor("mat_b_L0");
    mat_b_L0->SetMemoryTypeBoth(MemoryType::MEM_L0B, true);
    G.AddTensor(outputAstDtype, {128, 128}, "mat_c_L0");
    auto mat_c_L0 = G.GetTensor("mat_c_L0");
    mat_c_L0->SetMemoryTypeBoth(MemoryType::MEM_L0C, true);

    // add op
    G.AddOp(Opcode::OP_COPY_IN, {"mat_a"}, {"mat_a_partial_0"}, "L1copyInA_0");
    auto L1copyInA_0 = G.GetOp("L1copyInA_0");
    auto attrCopyInA_0 = std::make_shared<CopyOpAttribute>(
        OpImmediate::Specified({256, 0}), MemoryType::MEM_L1,
        OpImmediate::Specified(mat_a->GetShape()), OpImmediate::Specified(mat_a->tensor->GetRawShape()));
    L1copyInA_0->SetOpAttribute(attrCopyInA_0);
    G.AddOp(Opcode::OP_COPY_IN, {"mat_a"}, {"mat_a_partial_1"}, "L1copyInA_1");
    auto L1copyInA_1 = G.GetOp("L1copyInA_1");
    auto attrCopyInA_1 = std::make_shared<CopyOpAttribute>(
        OpImmediate::Specified({256, 64}), MemoryType::MEM_L1,
        OpImmediate::Specified(mat_a->GetShape()), OpImmediate::Specified(mat_a->tensor->GetRawShape()));
    L1copyInA_1->SetOpAttribute(attrCopyInA_1);
    G.AddOp(Opcode::OP_COPY_IN, {"mat_a"}, {"mat_a_partial_2"}, "L1copyInA_2");
    auto L1copyInA_2 = G.GetOp("L1copyInA_2");
    auto attrCopyInA_2 = std::make_shared<CopyOpAttribute>(
        OpImmediate::Specified({512, 0}), MemoryType::MEM_L1,
        OpImmediate::Specified(mat_a->GetShape()), OpImmediate::Specified(mat_a->tensor->GetRawShape()));
    L1copyInA_2->SetOpAttribute(attrCopyInA_2);
    G.AddOp(Opcode::OP_COPY_IN, {"mat_a"}, {"mat_a_partial_3"}, "L1copyInA_3");
    auto L1copyInA_3 = G.GetOp("L1copyInA_3");
    auto attrCopyInA_3 = std::make_shared<CopyOpAttribute>(
        OpImmediate::Specified({512, 64}), MemoryType::MEM_L1,
        OpImmediate::Specified(mat_a->GetShape()), OpImmediate::Specified(mat_a->tensor->GetRawShape()));
    L1copyInA_3->SetOpAttribute(attrCopyInA_3);

    G.AddOp(Opcode::OP_ASSEMBLE, {"mat_a_partial_0"}, {"mat_a_L1"}, "assemble_A_0");
    auto assemble_A_0 = G.GetOp("assemble_A_0");
    auto attrAssemble_0 = std::make_shared<AssembleOpAttribute>(MemoryType::MEM_L1, std::vector<int64_t>{0, 0});
    assemble_A_0->SetOpAttribute(attrAssemble_0);
    G.AddOp(Opcode::OP_ASSEMBLE, {"mat_a_partial_1"}, {"mat_a_L1"}, "assemble_A_1");
    auto assemble_A_1 = G.GetOp("assemble_A_1");
    auto attrAssemble_1 = std::make_shared<AssembleOpAttribute>(MemoryType::MEM_L1, std::vector<int64_t>{0, 64});
    assemble_A_1->SetOpAttribute(attrAssemble_1);
    G.AddOp(Opcode::OP_ASSEMBLE, {"mat_a_partial_2"}, {"mat_a_L1"}, "assemble_A_2");
    auto assemble_A_2 = G.GetOp("assemble_A_2");
    auto attrAssemble_2 = std::make_shared<AssembleOpAttribute>(MemoryType::MEM_L1, std::vector<int64_t>{64, 0});
    assemble_A_2->SetOpAttribute(attrAssemble_2);
    G.AddOp(Opcode::OP_ASSEMBLE, {"mat_a_partial_3"}, {"mat_a_L1"}, "assemble_A_3");
    auto assemble_A_3 = G.GetOp("assemble_A_3");
    auto attrAssemble_3 = std::make_shared<AssembleOpAttribute>(MemoryType::MEM_L1, std::vector<int64_t>{64, 64});
    assemble_A_3->SetOpAttribute(attrAssemble_3);

    G.AddOp(Opcode::OP_COPY_IN, {"mat_b"}, {"mat_b_L1"}, "L1_Copy_In_B");
    auto L1copyInB = G.GetOp("L1_Copy_In_B");
    auto attrCopyInB = std::make_shared<CopyOpAttribute>(
        OpImmediate::Specified({0, 0}), MemoryType::MEM_L1,
        OpImmediate::Specified(mat_b->GetShape()), OpImmediate::Specified(mat_b->tensor->GetRawShape()));
    L1copyInB->SetOpAttribute(attrCopyInB);

    G.AddOp(Opcode::OP_L1_TO_L0A, {"mat_a_L1"}, {"mat_a_L0"}, "L1_To_L0A");
    G.AddOp(Opcode::OP_L1_TO_L0B, {"mat_b_L1"}, {"mat_b_L0"}, "L1_To_L0B");

    G.AddOp(Opcode::OP_A_MUL_B, {"mat_a_L0", "mat_b_L0"}, {"mat_c_L0"}, "A_MUL_B");
    SetMatMulAttr(G, "A_MUL_B", false, 0);

    G.AddOp(Opcode::OP_COPY_OUT, {"mat_c_L0"}, {"mat_c"}, "L0C_Copy_out");
    auto copyOutOp = G.GetOp("L0C_Copy_out");
    auto attrCopyOut = std::make_shared<CopyOpAttribute>(
        OpImmediate::Specified({0, 0}), MemoryType::MEM_L0C,
        OpImmediate::Specified(mat_c->GetShape()), OpImmediate::Specified(mat_c->tensor->GetRawShape()));
    copyOutOp->SetOpAttribute(attrCopyOut);

    // set incast and outcast
    G.SetInCast({"mat_a", "mat_b"});
    G.SetOutCast({"mat_c"});
    // check before pass
    Function *function = G.GetFunction();
    EXPECT_NE(function, nullptr);

    // run pass
    SplitK passLocal;
    Status res = passLocal.Run(*function, "", "", 0);
    CubeProcess cubeProcess;
    cubeProcess.UpdateCubeOp(*function);
    // check after pass
    EXPECT_EQ(res, SUCCESS);
    EXPECT_EQ(mat_c->Datatype(), outputAstDtype);
    EXPECT_EQ(mat_c_L0->Datatype(), DataType::DT_FP32);
}

} // namespace tile_fwk
} // namespace npu