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
 * \file test_split_large_fanout_tensor.cpp
 * \brief Unit test for Split Large Fanout Tensor pass.
 */

#include <vector>
#include <numeric>
#include "gtest/gtest.h"
#include "tilefwk/tilefwk_op.h"
#include "interface/function/function.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "passes/pass_mgr/pass_manager.h"
#include "interface/configs/config_manager.h"
#include "ut_json/ut_json_tool.h"
#include "passes/tile_graph_pass/graph_optimization/split_large_fanout_tensor.h"
#include "computational_graph_builder.h"

using namespace npu::tile_fwk;

namespace npu {
namespace tile_fwk {

class SplitLargeFanoutTensorTest : public testing::Test {
public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}

    void SetUp() override {
        Program::GetInstance().Reset();
        config::Reset();
        config::SetHostOption(COMPILE_STAGE, HOST_COMPILE_END);
        config::SetHostConfig(KEY_STRATEGY, "SplitLargeFanoutTensorTestStrategy");
        config::SetPlatformConfig(KEY_ENABLE_COST_MODEL, false);
    }
    void TearDown() override {}

    std::vector<int64_t> CountViewAssemble(Function &func) {
        std::vector<int64_t> result = {0, 0};
        for (auto &op : func.Operations()) {
            std::cout << op.GetOpcodeStr() << " " << op.GetOpMagic() << std::endl;
            if (op.GetOpcode() == Opcode::OP_VIEW) {
                result[0]++;
            } else if (op.GetOpcode() == Opcode::OP_ASSEMBLE) {
                result[1]++;
            }
        }
        return result;
    }

    // [16, 16] --> View --> [8, 8] --> Sub --> [8, 8] --> Assemble --> [16, 16]
    void TileExpandSub(ComputationalGraphBuilder &G, const int N, const int T) {
        std::vector<int64_t> tileShape{T, T};
        std::vector<SymbolicScalar> dynShape{SymbolicScalar("a"), T};
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                std::vector<int64_t> offset = {i * T, j * T};

                std::string localA = "a_" + std::to_string(i * N + j);
                G.AddTensor(DataType::DT_FP32, tileShape, localA);
                auto tensorA = G.GetTensor(localA);
                tensorA->SetMemoryTypeBoth(MemoryType::MEM_UNKNOWN, true);
                G.AddOp(Opcode::OP_VIEW, {"a"}, {localA}, "View_A_" + std::to_string(i * N + j));
                auto View_A = G.GetOp("View_A_" + std::to_string(i * N + j));
                auto attrA = std::make_shared<ViewOpAttribute>(offset, MemoryType::MEM_UNKNOWN);
                View_A->SetOpAttribute(attrA);

                std::string localB = "b_" + std::to_string(i * N + j);
                G.AddTensor(DataType::DT_FP32, tileShape, localB);
                auto tensorB = G.GetTensor(localB);
                tensorB->SetMemoryTypeBoth(MemoryType::MEM_UNKNOWN, true);
                G.AddOp(Opcode::OP_VIEW, {"b"}, {localB}, "View_B_" + std::to_string(i * N + j));
                auto View_B = G.GetOp("View_B_" + std::to_string(i * N + j));
                auto attrB = std::make_shared<ViewOpAttribute>(offset, MemoryType::MEM_UNKNOWN);
                View_B->SetOpAttribute(attrB);

                std::string localSubOut = "sub_out_" + std::to_string(i * N + j);
                G.AddTensor(DataType::DT_FP32, tileShape, localSubOut);
                G.AddOp(Opcode::OP_SUB, {localA, localB}, {localSubOut}, "Sub_" + std::to_string(i * N + j));
                auto tensorSubOut = G.GetTensor(localSubOut);
                tensorSubOut->SetMemoryTypeBoth(MemoryType::MEM_UNKNOWN, true);
                tensorSubOut->UpdateDynValidShape(dynShape);

                G.AddOp(Opcode::OP_ASSEMBLE, {localSubOut}, {"sub_out"}, "Assemble_" + std::to_string(i * N + j));
                auto attrAssemble = std::make_shared<AssembleOpAttribute>(MemoryType::MEM_UNKNOWN, offset);
                auto assembleOp = G.GetOp("Assemble_" + std::to_string(i * N + j));
                assembleOp->SetOpAttribute(attrAssemble);
            }
        }
    }
    // multiConsumer means a tensor before the large tensor has more than one assemble op
    void BuildGraphForMToM(ComputationalGraphBuilder &G, bool multiConsumer = false) {
        int NUM_64 = 64;
        int NUM_128 = 128;
        int NUM_192 = 192;
        int NUM_256 = 256;
        int NUM_320 = 320;
        std::vector<int64_t> shape1{NUM_256, NUM_256};
        std::vector<int64_t> tiledShape1{NUM_64, NUM_256};
        std::vector<int64_t> shape2{NUM_256, NUM_64};
        std::vector<int64_t> tiledShape2{NUM_64, NUM_64};
        std::vector<int64_t> largeShape{NUM_128, NUM_320};
        std::vector<int64_t> tiledShape3{NUM_128, NUM_128};
        std::vector<int64_t> tiledShape4{NUM_128, NUM_64};

        // InCast a
        std::vector<SymbolicScalar> dynShapeA = {SymbolicScalar("a"), NUM_256};
        G.AddTensor(DataType::DT_FP32, shape1, "a"); // [256, 256]
        auto a = G.GetTensor("a");
        a->UpdateDynValidShape(dynShapeA);
        a->SetMemoryTypeBoth(MemoryType::MEM_UNKNOWN, true);
        // [256, 256] --> View(64, 0) --> [64, 256]
        G.AddTensor(DataType::DT_FP32, tiledShape1, "tiledA"); // [64, 256]
        auto tiledA = G.GetTensor("tiledA");
        tiledA->UpdateDynValidShape(dynShapeA);
        tiledA->SetMemoryTypeBoth(MemoryType::MEM_UNKNOWN, true);
        G.AddOp(Opcode::OP_VIEW, {"a"}, {"tiledA"}, "View_A");
        auto View_A =  G.GetOp("View_A");
        std::vector<int64_t> offsetA = {NUM_64, 0};
        auto attrA = std::make_shared<ViewOpAttribute>(offsetA, MemoryType::MEM_UNKNOWN);
        View_A->SetOpAttribute(attrA);
        // [256, 256] --> View(192, 0) --> [64, 256]
        G.AddTensor(DataType::DT_FP32, tiledShape1, "tiledB"); // [64, 256]
        auto tiledB = G.GetTensor("tiledB");
        tiledB->UpdateDynValidShape(dynShapeA);
        tiledB->SetMemoryTypeBoth(MemoryType::MEM_UNKNOWN, true);
        G.AddOp(Opcode::OP_VIEW, {"a"}, {"tiledB"}, "View_B");
        auto View_B =  G.GetOp("View_B");
        std::vector<int64_t> offsetB = {NUM_192, 0};
        auto attrB = std::make_shared<ViewOpAttribute>(offsetB, MemoryType::MEM_UNKNOWN);
        View_B->SetOpAttribute(attrB);

        // InCast c
        std::vector<SymbolicScalar> dynShapeC = {SymbolicScalar("c"), NUM_64};
        G.AddTensor(DataType::DT_FP32, shape2, "c"); // [256, 64]
        auto c = G.GetTensor("c");
        c->UpdateDynValidShape(dynShapeC);
        c->SetMemoryTypeBoth(MemoryType::MEM_UNKNOWN, true);
        // [256, 64] --> View(64, 0) --> [64, 64]
        G.AddTensor(DataType::DT_FP32, tiledShape2, "tiledC"); // [64, 64]
        auto tiledC = G.GetTensor("tiledC");
        tiledC->UpdateDynValidShape(dynShapeC);
        tiledC->SetMemoryTypeBoth(MemoryType::MEM_UNKNOWN, true);
        G.AddOp(Opcode::OP_VIEW, {"c"}, {"tiledC"}, "View_C");
        auto View_C =  G.GetOp("View_C");
        std::vector<int64_t> offsetC = {NUM_64, 0};
        auto attrC = std::make_shared<ViewOpAttribute>(offsetC, MemoryType::MEM_UNKNOWN);
        View_C->SetOpAttribute(attrC);
        // [256, 64] --> View(192, 0) --> [64, 64]
        G.AddTensor(DataType::DT_FP32, tiledShape2, "tiledD"); // [64, 64]
        auto tiledD = G.GetTensor("tiledD");
        tiledD->UpdateDynValidShape(dynShapeC);
        tiledD->SetMemoryTypeBoth(MemoryType::MEM_UNKNOWN, true);
        G.AddOp(Opcode::OP_VIEW, {"c"}, {"tiledD"}, "View_D");
        auto View_D =  G.GetOp("View_D");
        std::vector<int64_t> offsetD = {NUM_192, 0};
        auto attrD = std::make_shared<ViewOpAttribute>(offsetD, MemoryType::MEM_UNKNOWN);
        View_D->SetOpAttribute(attrD);

        if (multiConsumer) {
            G.AddTensor(DataType::DT_FP32, tiledShape4, "out3");
            auto out3 = G.GetTensor("out3");
            out3->SetMemoryTypeBoth(MemoryType::MEM_UNKNOWN, true);
            G.AddOp(Opcode::OP_VIEW, {"tiledC"}, {"out3"}, "View_C1");
            auto View_C1 =  G.GetOp("View_C1");
            std::vector<int64_t> offsetC1 = {0, 0};
            auto attrC1 = std::make_shared<ViewOpAttribute>(offsetC1, MemoryType::MEM_UNKNOWN);
            View_C1->SetOpAttribute(attrC1);
        }

        // [64, 256][64, 64]
        // [64, 256][64, 64]  Assemble --> [128, 320]
        G.AddTensor(DataType::DT_FP32, largeShape, "largeTensor"); // Assemble --> [128, 320]
        auto largeTensor = G.GetTensor("largeTensor");
        largeTensor->SetMemoryTypeBoth(MemoryType::MEM_UNKNOWN, true);
        G.AddOp(Opcode::OP_ASSEMBLE, {"tiledA"}, {"largeTensor"}, "Assemble_A");
        auto attrAssembleA = std::make_shared<AssembleOpAttribute>(MemoryType::MEM_UNKNOWN, std::vector<int64_t> {0, 0});
        auto assembleA = G.GetOp("Assemble_A");
        assembleA->SetOpAttribute(attrAssembleA);
        G.AddOp(Opcode::OP_ASSEMBLE, {"tiledB"}, {"largeTensor"}, "Assemble_B");
        auto attrAssembleB = std::make_shared<AssembleOpAttribute>(MemoryType::MEM_UNKNOWN, std::vector<int64_t> {NUM_64, 0});
        auto assembleB = G.GetOp("Assemble_B");
        assembleB->SetOpAttribute(attrAssembleB);
        G.AddOp(Opcode::OP_ASSEMBLE, {"tiledC"}, {"largeTensor"}, "Assemble_C");
        auto attrAssembleC = std::make_shared<AssembleOpAttribute>(MemoryType::MEM_UNKNOWN, std::vector<int64_t> {0, NUM_256});
        auto assembleC = G.GetOp("Assemble_C");
        assembleC->SetOpAttribute(attrAssembleC);
        G.AddOp(Opcode::OP_ASSEMBLE, {"tiledD"}, {"largeTensor"}, "Assemble_D");
        auto attrAssembleD = std::make_shared<AssembleOpAttribute>(MemoryType::MEM_UNKNOWN, std::vector<int64_t> {NUM_64, NUM_256});
        auto assembleD = G.GetOp("Assemble_D");
        assembleD->SetOpAttribute(attrAssembleD);

        // output 1: MtoM
        // [128, 320] --> View(0, 0) --> [128, 128]
        G.AddTensor(DataType::DT_FP32, tiledShape3, "tiledView_1");
        auto tiledView_1 = G.GetTensor("tiledView_1");
        tiledView_1->SetMemoryTypeBoth(MemoryType::MEM_UNKNOWN, true);
        G.AddOp(Opcode::OP_VIEW, {"largeTensor"}, {"tiledView_1"}, "View_1");
        auto View_1 =  G.GetOp("View_1");
        std::vector<int64_t> offset1 = {0, 0};
        auto attr1 = std::make_shared<ViewOpAttribute>(offset1, MemoryType::MEM_UNKNOWN);
        View_1->SetOpAttribute(attr1);

        // output 2: MtoM
        // [128, 320] --> View(0, 128) --> [128, 128]
        G.AddTensor(DataType::DT_FP32, tiledShape3, "tiledView_2");
        auto tiledView_2 = G.GetTensor("tiledView_2");
        tiledView_2->SetMemoryTypeBoth(MemoryType::MEM_UNKNOWN, true);
        G.AddOp(Opcode::OP_VIEW, {"largeTensor"}, {"tiledView_2"}, "View_2");
        auto View_2 =  G.GetOp("View_2");
        std::vector<int64_t> offset2 = {0, 0};
        auto attr2 = std::make_shared<ViewOpAttribute>(offset2, MemoryType::MEM_UNKNOWN);
        View_2->SetOpAttribute(attr2);

        // output 3: Mto1
        // [128, 320] --> View(0, 256) --> [128, 64]
        G.AddTensor(DataType::DT_FP32, tiledShape4, "tiledView_3");
        auto tiledView_3 = G.GetTensor("tiledView_3");
        tiledView_3->SetMemoryTypeBoth(MemoryType::MEM_UNKNOWN, true);
        G.AddOp(Opcode::OP_VIEW, {"largeTensor"}, {"tiledView_3"}, "View_3");
        auto View_3 =  G.GetOp("View_3");
        std::vector<int64_t> offset3 = {0, 256};
        auto attr3 = std::make_shared<ViewOpAttribute>(offset3, MemoryType::MEM_UNKNOWN);
        View_3->SetOpAttribute(attr3);

        // output 1 + output 2
        // [128, 128] + [128, 128] --> [128, 128]
        G.AddTensor(DataType::DT_FP32, tiledShape3, "add_out");
        G.AddOp(Opcode::OP_ADD, {"tiledView_1", "tiledView_2"}, {"add_out"}, "Add");
        auto addOut = G.GetTensor("add_out");
        addOut->SetMemoryTypeBoth(MemoryType::MEM_UNKNOWN, true);

        G.AddTensor(DataType::DT_FP32, tiledShape3, "out1");
        auto out1 = G.GetTensor("out1");
        out1->SetMemoryTypeBoth(MemoryType::MEM_UNKNOWN, true);
        G.AddOp(Opcode::OP_ASSEMBLE, {"add_out"}, {"out1"}, "Assemble_1");
        auto attrAssemble1 = std::make_shared<AssembleOpAttribute>(MemoryType::MEM_UNKNOWN, std::vector<int64_t> {0, 0});
        auto assemble1 = G.GetOp("Assemble_1");
        assemble1->SetOpAttribute(attrAssemble1);

        // output 3 exp
        // [128, 64] --> Exp --> [128, 64]
        G.AddTensor(DataType::DT_FP32, tiledShape4, "exp_out");
        G.AddOp(Opcode::OP_EXP, {"tiledView_3"}, {"exp_out"}, "Exp");
        auto expOut = G.GetTensor("exp_out");
        expOut->SetMemoryTypeBoth(MemoryType::MEM_UNKNOWN, true);

        G.AddTensor(DataType::DT_FP32, tiledShape4, "out2");
        auto out2 = G.GetTensor("out2");
        out2->SetMemoryTypeBoth(MemoryType::MEM_UNKNOWN, true);
        G.AddOp(Opcode::OP_ASSEMBLE, {"exp_out"}, {"out2"}, "Assemble_2");
        auto attrAssemble2 = std::make_shared<AssembleOpAttribute>(MemoryType::MEM_UNKNOWN, std::vector<int64_t> {0, 0});
        auto assemble2 = G.GetOp("Assemble_2");
        assemble2->SetOpAttribute(attrAssemble2);

        G.SetInCast({"a", "c"});
        if (multiConsumer){
            G.SetOutCast({"out1", "out2", "out3"});
        } else {
            G.SetOutCast({"out1", "out2"});
        }
    }
};

TEST_F(SplitLargeFanoutTensorTest, TestLCM) {
    int64_t NUM_16 = 16;
    int64_t NUM_2 = 2;
    int64_t NUM_3 = 3;
    int64_t NUM_96 = 96;
    int64_t x = NUM_16 * NUM_2;
    int64_t y = NUM_16 * NUM_3;
    // 单独执行pass
    npu::tile_fwk::SplitLargeFanoutTensor splitLargeFanoutTensor;
    auto gcd = splitLargeFanoutTensor.GCD(x, y);
    int64_t lcm;
    auto status = splitLargeFanoutTensor.LCM(x, y, lcm);
    std::cout << "Run Pass Done." << std::endl;
    EXPECT_EQ(status, SUCCESS);
    EXPECT_EQ(gcd, NUM_16);
    EXPECT_EQ(lcm, NUM_96);
}

TEST_F(SplitLargeFanoutTensorTest, BeCovered_Full) {
    int NUM_2 = 2;
    int NUM_32 = 32;
    int NUM_64 = 64;
    int NUM_128 = 128;
    int NUM_256 = 256;
    int NUM_512 = 512;
    int NUM_576 = 512 + 64;
    std::vector<int64_t> shape0{NUM_128, NUM_64};
    std::vector<int64_t> tiledShape0{NUM_32, NUM_64};

    std::vector<int64_t> shape1{NUM_128, NUM_512};
    std::vector<int64_t> tiledShape1{NUM_32, NUM_512};

    std::vector<int64_t> tiledShape2{NUM_32, NUM_576};
    std::vector<int64_t> tiledShape3{NUM_32, NUM_256};
    ComputationalGraphBuilder G;

    // [128, 512] --> View(64, 0) --> [32, 512]
    std::vector<SymbolicScalar> dynShapeA = {SymbolicScalar("a"), NUM_512};
    G.AddTensor(DataType::DT_FP32, shape1, "a"); // [128, 512]
    auto a = G.GetTensor("a");
    a->UpdateDynValidShape(dynShapeA);
    a->SetMemoryTypeBoth(MemoryType::MEM_UNKNOWN, true);
    G.AddTensor(DataType::DT_FP32, tiledShape1, "tiledA"); // [32, 512]
    auto tiledA = G.GetTensor("tiledA");
    tiledA->UpdateDynValidShape(dynShapeA);
    tiledA->SetMemoryTypeBoth(MemoryType::MEM_UNKNOWN, true);
    G.AddOp(Opcode::OP_VIEW, {"a"}, {"tiledA"}, "View_A");
    auto View_A =  G.GetOp("View_A");
    std::vector<int64_t> offsetA = {NUM_64, 0};
    auto attrA = std::make_shared<ViewOpAttribute>(offsetA, MemoryType::MEM_UNKNOWN);
    View_A->SetOpAttribute(attrA);

    // [128, 64] --> View(64, 0) --> [32, 64]
    std::vector<SymbolicScalar> dynShapeB = {SymbolicScalar("a"), NUM_64};
    G.AddTensor(DataType::DT_FP32, shape0, "b"); // [128, 64]
    auto b = G.GetTensor("b");
    b->UpdateDynValidShape(dynShapeB);
    b->SetMemoryTypeBoth(MemoryType::MEM_UNKNOWN, true);
    G.AddTensor(DataType::DT_FP32, tiledShape0, "tiledB"); // [32, 64]
    auto tiledB = G.GetTensor("tiledB");
    tiledB->UpdateDynValidShape(dynShapeB);
    tiledB->SetMemoryTypeBoth(MemoryType::MEM_UNKNOWN, true);
    G.AddOp(Opcode::OP_VIEW, {"b"}, {"tiledB"}, "View_B");
    auto View_B =  G.GetOp("View_B");
    std::vector<int64_t> offsetB = {NUM_64, 0};
    auto attrB = std::make_shared<ViewOpAttribute>(offsetB, MemoryType::MEM_UNKNOWN);
    View_B->SetOpAttribute(attrB);

    // [32, 512] concat [32, 64] -->  [32, 576]
    G.AddTensor(DataType::DT_FP32, tiledShape2, "tiledConcat"); // [32, 512] + [32, 64] --> [32, 576]
    auto tiledConcat = G.GetTensor("tiledConcat");
    tiledConcat->SetMemoryTypeBoth(MemoryType::MEM_UNKNOWN, true);
    G.AddOp(Opcode::OP_ASSEMBLE, {"tiledA"}, {"tiledConcat"}, "Assemble_A");
    auto attrAssembleA = std::make_shared<AssembleOpAttribute>(MemoryType::MEM_UNKNOWN, std::vector<int64_t> {0, 0});
    auto assembleA = G.GetOp("Assemble_A");
    assembleA->SetOpAttribute(attrAssembleA);
    G.AddOp(Opcode::OP_ASSEMBLE, {"tiledB"}, {"tiledConcat"}, "Assemble_B");
    auto attrAssembleB = std::make_shared<AssembleOpAttribute>(MemoryType::MEM_UNKNOWN, std::vector<int64_t> {0, NUM_512});
    auto assembleB = G.GetOp("Assemble_B");
    assembleB->SetOpAttribute(attrAssembleB);

    // part 1: BeCovered
    // [32, 576] --> View(0, 0) --> [32, 256]
    G.AddTensor(DataType::DT_FP32, tiledShape3, "tiledView_1");
    auto tiledView_1 = G.GetTensor("tiledView_1");
    tiledView_1->SetMemoryTypeBoth(MemoryType::MEM_UNKNOWN, true);
    G.AddOp(Opcode::OP_VIEW, {"tiledConcat"}, {"tiledView_1"}, "View_1");
    auto View_1 =  G.GetOp("View_1");
    std::vector<int64_t> offset1 = {0, 0};
    auto attr1 = std::make_shared<ViewOpAttribute>(offset1, MemoryType::MEM_UNKNOWN);
    View_1->SetOpAttribute(attr1);

    // part 2: BeCovered
    // [32, 576] --> View(0, 256) --> [32, 256]
    G.AddTensor(DataType::DT_FP32, tiledShape3, "tiledView_2");
    auto tiledView_2 = G.GetTensor("tiledView_2");
    tiledView_2->SetMemoryTypeBoth(MemoryType::MEM_UNKNOWN, true);
    G.AddOp(Opcode::OP_VIEW, {"tiledConcat"}, {"tiledView_2"}, "View_2");
    auto View_2 =  G.GetOp("View_2");
    std::vector<int64_t> offset2 = {0, NUM_256};
    auto attr2 = std::make_shared<ViewOpAttribute>(offset2, MemoryType::MEM_UNKNOWN);
    View_2->SetOpAttribute(attr2);

    // part 3: Perfectly Match
    // [32, 576] --> View(0, 512) --> [32, 64]
    G.AddTensor(DataType::DT_FP32, tiledShape0, "tiledView_3");
    auto tiledView_3 = G.GetTensor("tiledView_3");
    tiledView_3->SetMemoryTypeBoth(MemoryType::MEM_UNKNOWN, true);
    G.AddOp(Opcode::OP_VIEW, {"tiledConcat"}, {"tiledView_3"}, "View_3");
    auto View_3 =  G.GetOp("View_3");
    std::vector<int64_t> offset3 = {0, NUM_512};
    auto attr3 = std::make_shared<ViewOpAttribute>(offset3, MemoryType::MEM_UNKNOWN);
    View_3->SetOpAttribute(attr3);

    // part1 + part2
    // [32, 256] + [32, 256] --> [32, 256]
    G.AddTensor(DataType::DT_FP32, tiledShape3, "add_out");
    G.AddOp(Opcode::OP_ADD, {"tiledView_1", "tiledView_2"}, {"add_out"}, "Add");
    auto addOut = G.GetTensor("add_out");
    addOut->SetMemoryTypeBoth(MemoryType::MEM_UNKNOWN, true);

    G.AddTensor(DataType::DT_FP32, tiledShape3, "out1");
    auto out1 = G.GetTensor("out1");
    out1->SetMemoryTypeBoth(MemoryType::MEM_UNKNOWN, true);
    G.AddOp(Opcode::OP_ASSEMBLE, {"add_out"}, {"out1"}, "Assemble_1");
    auto attrAssemble1 = std::make_shared<AssembleOpAttribute>(MemoryType::MEM_UNKNOWN, std::vector<int64_t> {0, 0});
    auto assemble1 = G.GetOp("Assemble_1");
    assemble1->SetOpAttribute(attrAssemble1);

    // part3 exp
    // [32, 64] --> Exp --> [32, 64]
    G.AddTensor(DataType::DT_FP32, tiledShape0, "exp_out");
    G.AddOp(Opcode::OP_EXP, {"tiledView_3"}, {"exp_out"}, "Exp");
    auto expOut = G.GetTensor("exp_out");
    expOut->SetMemoryTypeBoth(MemoryType::MEM_UNKNOWN, true);

    G.AddTensor(DataType::DT_FP32, tiledShape0, "out2");
    auto out2 = G.GetTensor("out2");
    out2->SetMemoryTypeBoth(MemoryType::MEM_UNKNOWN, true);
    G.AddOp(Opcode::OP_ASSEMBLE, {"exp_out"}, {"out2"}, "Assemble_2");
    auto attrAssemble2 = std::make_shared<AssembleOpAttribute>(MemoryType::MEM_UNKNOWN, std::vector<int64_t> {0, 0});
    auto assemble2 = G.GetOp("Assemble_2");
    assemble2->SetOpAttribute(attrAssemble2);

    G.SetInCast({"a", "b"});
    G.SetOutCast({"out1", "out2"});
    Function *function = G.GetFunction();

    // 确认构图完毕
    constexpr int opNumBefore = 11;
    constexpr int viewNumBefore = 5;
    constexpr int assembleNumBefore = 4;
    auto countResultBefore = CountViewAssemble(*function);
    int viewNumCount = countResultBefore[0];
    int assembleNumCount = countResultBefore[1];
    EXPECT_EQ(function->Operations().size(), opNumBefore) << opNumBefore << " operations before pass";
    EXPECT_EQ(viewNumCount, viewNumBefore) << viewNumBefore << " OP_VIEW before pass";
    EXPECT_EQ(assembleNumCount, assembleNumBefore) << assembleNumBefore << " OP_ASSEMBLE before pass";
    std::cout << "Build Graph Done." << std::endl;
    std::unordered_map<int, std::vector<int64_t>> viewOpToUbBfore;
    for (auto &op : function->Operations()) {
        if (op.GetOpcode() != Opcode::OP_VIEW) {
            continue;
        }
        auto viewAttr = dynamic_cast<ViewOpAttribute *>(op.GetOpAttribute().get());
        MemoryType toType = viewAttr->GetTo();
        if (toType != MemoryType::MEM_UNKNOWN) {
            continue;
        }
        viewOpToUbBfore.insert({op.GetOpMagic(), viewAttr->GetFromOffset()});
    }
    /*
    dump graph before Pass
    function->DumpJsonFile(jsonFilePath);
    */
    // 单独执行pass
    npu::tile_fwk::SplitLargeFanoutTensor splitLargeFanoutTensor;
    splitLargeFanoutTensor.PreCheck(*function);
    splitLargeFanoutTensor.RunOnFunction(*function);
    splitLargeFanoutTensor.PostCheck(*function);
    std::cout << "Run Pass Done." << std::endl;
    /*
    dump graph after Pass
    function->DumpJsonFile(jsonFilePath);
    */
    constexpr int opNumAfter = 7;
    constexpr int viewNumAfter = 3;
    constexpr int assembleNumAfter = 2;
    auto countResultAfter = CountViewAssemble(*function);
    viewNumCount = countResultAfter[0];
    assembleNumCount = countResultAfter[1];
    EXPECT_EQ(function->Operations().size(), opNumAfter) << opNumAfter << " operations after pass";
    EXPECT_EQ(viewNumCount, viewNumAfter) << viewNumAfter << " OP_VIEW after pass";
    EXPECT_EQ(assembleNumCount, assembleNumAfter) << assembleNumAfter << " OP_ASSEMBLE after pass";

    for (auto &op : function->Operations()) {
        if (op.GetOpcode() != Opcode::OP_VIEW) {
            continue;
        }
        auto viewAttr = dynamic_cast<ViewOpAttribute *>(op.GetOpAttribute().get());
        MemoryType toType = viewAttr->GetTo();
        if (toType != MemoryType::MEM_UNKNOWN) {
            continue;
        }
        EXPECT_NE(viewOpToUbBfore.find(op.GetOpMagic()), viewOpToUbBfore.end());
        auto oldOffset = viewOpToUbBfore.at(op.GetOpMagic());
        auto newOffset = viewAttr->GetFromOffset();
        auto newDynOffset = viewAttr->GetFromDynOffset();
        EXPECT_EQ(newOffset.size(), NUM_2);
        EXPECT_EQ(newOffset[0], NUM_64);
        EXPECT_EQ(newDynOffset.size(), NUM_2);
        EXPECT_EQ(newDynOffset[0].Concrete(), NUM_64);

        auto input = op.GetIOperands().front();
        auto inputDynShape = input->GetDynValidShape();
        EXPECT_EQ(inputDynShape.size(), NUM_2);
        EXPECT_EQ(inputDynShape[0].Dump(), "a");
    }
}

/*
[64, 256][64, 64]
[64, 256][64, 64] --> [128, 128][128, 128][128, 64]
*/
TEST_F(SplitLargeFanoutTensorTest, MtoM) {
    int NUM_2 = 2;
    ComputationalGraphBuilder G;
    BuildGraphForMToM(G);
    Function *function = G.GetFunction();

    // 确认构图完毕
    constexpr int opNumBefore = 15;
    constexpr int viewNumBefore = 7;
    constexpr int assembleNumBefore = 6;

    auto countResultBefore = CountViewAssemble(*function);
    int viewNumCount = countResultBefore[0];
    int assembleNumCount = countResultBefore[1];
    EXPECT_EQ(function->Operations().size(), opNumBefore) << opNumBefore << " operations before pass";
    EXPECT_EQ(viewNumCount, viewNumBefore) << viewNumBefore << " OP_VIEW before pass";
    EXPECT_EQ(assembleNumCount, assembleNumBefore) << assembleNumBefore << " OP_ASSEMBLE before pass";
    std::cout << "Build Graph Done." << std::endl;
    /*
    dump graph before Pass
    function->DumpJsonFile(jsonFilePath);
    */
    // 单独执行pass
    npu::tile_fwk::SplitLargeFanoutTensor splitLargeFanoutTensor;
    splitLargeFanoutTensor.PreCheck(*function);
    splitLargeFanoutTensor.RunOnFunction(*function);
    splitLargeFanoutTensor.PostCheck(*function);
    std::cout << "Run Pass Done." << std::endl;
    /*
    dump graph after Pass
    function->DumpJsonFile(jsonFilePath);
    */
    auto countResultAfter = CountViewAssemble(*function);
    viewNumCount = countResultAfter[0];
    assembleNumCount = countResultAfter[1];
    EXPECT_EQ(function->Operations().size(), opNumBefore) << opNumBefore << " operations after pass";
    EXPECT_EQ(viewNumCount, viewNumBefore) << viewNumBefore << " OP_VIEW after pass";
    EXPECT_EQ(assembleNumCount, assembleNumBefore) << assembleNumBefore << " OP_ASSEMBLE after pass";

    constexpr int singleViewOpmagic = 10010;
    for (auto &op : function->Operations()) {
        if (op.GetOpMagic() == singleViewOpmagic) {
            auto viewAttr = dynamic_cast<ViewOpAttribute *>(op.GetOpAttribute().get());
            auto offset = viewAttr->GetFromOffset();
            EXPECT_EQ(accumulate(offset.begin(), offset.end(), 0), 0) << "OP_VIEW offset should be all zero";
            auto dynOffset = viewAttr->GetFromDynOffset();
            EXPECT_EQ(dynOffset.size(), 0);
            auto input = op.GetIOperands().front();
            auto inputDynShape = input->GetDynValidShape();
            EXPECT_EQ(inputDynShape.size(), NUM_2);
            EXPECT_EQ(inputDynShape[0].Dump(), "RUNTIME_Max(RUNTIME_Max(0, c), ((c+64)*RUNTIME_Ne(c, 0)))");
        }
    }
}

/*
[64, 256][64, 64]
[64, 256][64, 64] --> [128, 128][128, 128][128, 64]
*/
TEST_F(SplitLargeFanoutTensorTest, MtoMtoMoreSplit) {
    ComputationalGraphBuilder G;
    BuildGraphForMToM(G);
    Function *function = G.GetFunction();

    // 确认构图完毕
    constexpr int opNumBefore = 15;
    constexpr int viewNumBefore = 7;
    constexpr int assembleNumBefore = 6;

    auto countResultBefore = CountViewAssemble(*function);
    int viewNumCount = countResultBefore[0];
    int assembleNumCount = countResultBefore[1];
    EXPECT_EQ(function->Operations().size(), opNumBefore) << opNumBefore << " operations before pass";
    EXPECT_EQ(viewNumCount, viewNumBefore) << viewNumBefore << " OP_VIEW before pass";
    EXPECT_EQ(assembleNumCount, assembleNumBefore) << assembleNumBefore << " OP_ASSEMBLE before pass";
    std::cout << "Build Graph Done." << std::endl;
    /*
    dump graph before Pass
    function->DumpJsonFile(jsonFilePath);
    */
    // 单独执行pass
    npu::tile_fwk::SplitLargeFanoutTensor splitLargeFanoutTensor;
    splitLargeFanoutTensor.enableMoreSplit = true;
    splitLargeFanoutTensor.PreCheck(*function);
    splitLargeFanoutTensor.RunOnFunction(*function);
    splitLargeFanoutTensor.PostCheck(*function);
    std::cout << "Run Pass Done." << std::endl;
    /*
    dump graph after Pass
    function->DumpJsonFile(jsonFilePath);
    */
    constexpr int opNumAfter = 20;
    constexpr int viewNumAfter = 10;
    constexpr int assembleNumAfter = 8;

    auto countResultAfter = CountViewAssemble(*function);
    viewNumCount = countResultAfter[0];
    assembleNumCount = countResultAfter[1];
    EXPECT_EQ(function->Operations().size(), opNumAfter) << opNumAfter << " operations after pass";
    EXPECT_EQ(viewNumCount, viewNumAfter) << viewNumAfter << " OP_VIEW after pass";
    EXPECT_EQ(assembleNumCount, assembleNumAfter) << assembleNumAfter << " OP_ASSEMBLE after pass";
}

TEST_F(SplitLargeFanoutTensorTest, MtoMGetCorrectAssemble) {
    ComputationalGraphBuilder G;
    BuildGraphForMToM(G, true);
    Function *function = G.GetFunction();

    std::cout << "Build Graph Done" << std::endl;
    // 执行pass, 不发生segmentFault即为获取assemble正常
    npu::tile_fwk::SplitLargeFanoutTensor splitLargeFanoutTensor;
    splitLargeFanoutTensor.enableMoreSplit = true;
    EXPECT_EQ(SUCCESS, splitLargeFanoutTensor.PreCheck(*function));
    EXPECT_EQ(SUCCESS, splitLargeFanoutTensor.RunOnFunction(*function));
    EXPECT_EQ(SUCCESS, splitLargeFanoutTensor.PostCheck(*function));
    std::cout << "Run Pass Done" << std::endl;
}

void Build1ToMMultiConsumers(ComputationalGraphBuilder &G){
    int NUM_16 = 16;
    int NUM_32 = 32;

    // 定义所有张量的形状和名称并添加
    std::map<std::string, std::vector<int64_t>> tensors = {
        {"incast0", {NUM_16, NUM_32}}, {"incast1", {NUM_16, NUM_32}},
        {"outcast0", {NUM_16, NUM_16}}, {"outcast1", {NUM_16, NUM_16}},
        {"outcast2", {NUM_16, NUM_16}}, {"outcast3", {NUM_16, NUM_16}},
        {"largeTensor", {NUM_32, NUM_32}}, {"outcast", {NUM_16, NUM_16}}
    };
    for (const auto& [name, shape] : tensors) {
        G.AddTensor(DataType::DT_FP32, shape, name);
        auto tensor = G.GetTensor(name);
        tensor->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    }

    G.AddOp(Opcode::OP_VIEW, {"incast0"}, {"outcast"}, "view");
    auto viewEX = G.GetOp("view");
    Offset offsetEX = {0, 0};
    viewEX->SetOpAttribute(std::make_shared<ViewOpAttribute>(offsetEX, MemoryType::MEM_DEVICE_DDR));

    // 定义所有ASSEMBLE操作并添加
    std::vector<std::tuple<std::string, std::string, std::vector<int64_t>>> assembleOps = {
        {"incast0", "assemble0", {0, 0}}, {"incast1", "assemble1", {16, 0}}
    };
    for (const auto& [input, opName, offset] : assembleOps) {
        G.AddOp(Opcode::OP_ASSEMBLE, {input}, {"largeTensor"}, opName);
        auto assembleOp = G.GetOp(opName);
        assembleOp->SetOpAttribute(std::make_shared<AssembleOpAttribute>(MemoryType::MEM_DEVICE_DDR, offset));
    }

    // 定义所有VIEW操作并添加
    std::vector<std::tuple<std::string, std::string, std::vector<int64_t>>> viewOps = {
        {"outcast0", "view0", {0, 0}}, {"outcast1", "view1", {0, 16}},
        {"outcast2", "view2", {16, 0}}, {"outcast3", "view3", {16, 16}}
    };
    for (const auto& [output, opName, offset] : viewOps) {
        G.AddOp(Opcode::OP_VIEW, {"largeTensor"}, {output}, opName);
        auto viewOp = G.GetOp(opName);
        viewOp->SetOpAttribute(std::make_shared<ViewOpAttribute>(offset, MemoryType::MEM_DEVICE_DDR));
    }

    G.SetInCast({"incast0", "incast1"});
    G.SetOutCast({"outcast0", "outcast1", "outcast2", "outcast3", "outcast"});
}

TEST_F(SplitLargeFanoutTensorTest, 1ToMGetCorrectAssemble) {
    ComputationalGraphBuilder G;
    Build1ToMMultiConsumers(G);
    Function *function = G.GetFunction();

    std::cout << "Build Graph Done." << std::endl;
    // 单独执行pass, 不发生segmentFault即为获取assemble正常
    npu::tile_fwk::SplitLargeFanoutTensor splitLargeFanoutTensor;
    splitLargeFanoutTensor.enableMoreSplit = false;
    EXPECT_EQ(SUCCESS, splitLargeFanoutTensor.PreCheck(*function));
    EXPECT_EQ(SUCCESS, splitLargeFanoutTensor.RunOnFunction(*function));
    EXPECT_EQ(SUCCESS, splitLargeFanoutTensor.PostCheck(*function));
    std::cout << "Run Pass Done." << std::endl;
}

TEST_F(SplitLargeFanoutTensorTest, Unmatched) {
    int N = 2;
    int T = 8;
    std::vector<int64_t> shape0{N * T, N * T};
    std::vector<int64_t> shape1{T, T};
    std::vector<int64_t> shape2{N * T, T};
    std::vector<int64_t> shape3{T / N, N * T};
    ComputationalGraphBuilder G;
    G.AddTensor(DataType::DT_FP32, shape0, "a");
    G.AddTensor(DataType::DT_FP32, shape0, "b");
    G.AddTensor(DataType::DT_FP32, shape0, "sub_out");
    G.AddTensor(DataType::DT_FP32, shape0, "out");
    // [16, 16] --> View --> [8, 8] --> Sub --> [8, 8] --> Assemble --> [16, 16]
    TileExpandSub(G, N, T);
    auto subOut = G.GetTensor("sub_out");
    subOut->SetMemoryTypeBoth(MemoryType::MEM_UNKNOWN, true);

    //  [16, 16] --> View --> [4, 16] --> Exp --> [4, 16]  --> [16, 16]
    for (int i = 0; i < T / N; i++) {
        // View
        G.AddTensor(DataType::DT_FP32, shape3, "viewOut_" + std::to_string(i));
        auto viewOut = G.GetTensor("viewOut_" + std::to_string(i));
        viewOut->SetMemoryTypeBoth(MemoryType::MEM_UNKNOWN, true);
        G.AddOp(Opcode::OP_VIEW, {"sub_out"}, {"viewOut_" + std::to_string(i)}, "View_" + std::to_string(i));
        auto viewOp =  G.GetOp("View_" + std::to_string(i));
        std::vector<int64_t> offset = {i * T / N, 0};
        auto attr = std::make_shared<ViewOpAttribute>(offset, MemoryType::MEM_UNKNOWN);
        viewOp->SetOpAttribute(attr);

        // Exp
        G.AddTensor(DataType::DT_FP32, shape3, "expOut_" + std::to_string(i));
        auto expOut = G.GetTensor("expOut_" + std::to_string(i));
        expOut->SetMemoryTypeBoth(MemoryType::MEM_UNKNOWN, true);
        G.AddOp(Opcode::OP_EXP, {"viewOut_" + std::to_string(i)}, {"expOut_" + std::to_string(i)}, "Exp_" + std::to_string(i));

        // Assemble
        G.AddOp(Opcode::OP_ASSEMBLE, {"expOut_" + std::to_string(i)}, {"out"}, "AssembleFinal_" + std::to_string(i));
        auto attrAssemble = std::make_shared<AssembleOpAttribute>(MemoryType::MEM_UNKNOWN, std::vector<int64_t> {i * T / N, 0});
        auto assembleOp = G.GetOp("AssembleFinal_" + std::to_string(i));
        assembleOp->SetOpAttribute(attrAssemble);
    }

    auto a = G.GetTensor("a");
    a->SetMemoryTypeBoth(MemoryType::MEM_UNKNOWN, true);
    auto b = G.GetTensor("b");
    b->SetMemoryTypeBoth(MemoryType::MEM_UNKNOWN, true);
    auto out = G.GetTensor("out");
    out->SetMemoryTypeBoth(MemoryType::MEM_UNKNOWN, true);

    G.SetInCast({"a", "b"});
    G.SetOutCast({"out"});
    Function *function = G.GetFunction();
    // 确认构图完毕
    constexpr int opNumBefore = 28;
    constexpr int viewNumBefore = 12;
    constexpr int assembleNumBefore = 8;
    auto countResultBefore = CountViewAssemble(*function);
    int viewNumCount = countResultBefore[0];
    int assembleNumCount = countResultBefore[1];
    EXPECT_EQ(function->Operations().size(), opNumBefore) << opNumBefore << " operations before pass";
    EXPECT_EQ(viewNumCount, viewNumBefore) << viewNumBefore << " OP_VIEW before pass";
    EXPECT_EQ(assembleNumCount, assembleNumBefore) << assembleNumBefore << " OP_ASSEMBLE before pass";
    std::cout << "Build Graph Done." << std::endl;

    // 单独执行pass
    npu::tile_fwk::SplitLargeFanoutTensor splitLargeFanoutTensor;
    splitLargeFanoutTensor.PreCheck(*function);
    splitLargeFanoutTensor.RunOnFunction(*function);
    splitLargeFanoutTensor.PostCheck(*function);
    std::cout << "Run Pass Done." << std::endl;

    auto countResultAfter = CountViewAssemble(*function);
    viewNumCount = countResultAfter[0];
    assembleNumCount = countResultAfter[1];
    EXPECT_EQ(function->Operations().size(), opNumBefore) << opNumBefore << " operations after pass";
    EXPECT_EQ(viewNumCount, viewNumBefore) << viewNumBefore << " OP_VIEW after pass";
    EXPECT_EQ(assembleNumCount, assembleNumBefore) << assembleNumBefore << " OP_ASSEMBLE after pass";
}

TEST_F(SplitLargeFanoutTensorTest, PerfectlyMatchWithAll_Full) {
    int NUM_2 = 2;
    int N = 2;
    int T = 8;
    std::vector<int64_t> shape0{N * T, N * T};
    std::vector<int64_t> shape1{T, T};
    std::vector<int64_t> shape2{N * T, T};
    ComputationalGraphBuilder G;
    G.AddTensor(DataType::DT_FP32, shape0, "a");
    G.AddTensor(DataType::DT_FP32, shape0, "b");
    G.AddTensor(DataType::DT_FP32, shape2, "out");
    G.AddTensor(DataType::DT_FP32, shape0, "sub_out");
    // [16, 16] --> View --> [8, 8] --> Sub --> [8, 8] --> Assemble --> [16, 16]
    TileExpandSub(G, N, T);
    auto subOut = G.GetTensor("sub_out");
    subOut->SetMemoryTypeBoth(MemoryType::MEM_UNKNOWN, true);

    // View 获取 [16, 16] 左半边 [16, 8]
    G.AddTensor(DataType::DT_FP32, shape2, "sub_out_left");
    auto subOutLeft = G.GetTensor("sub_out_left");
    subOutLeft->SetMemoryTypeBoth(MemoryType::MEM_UNKNOWN, true);
    G.AddOp(Opcode::OP_VIEW, {"sub_out"}, {"sub_out_left"}, "View_Left");
    auto View_Left =  G.GetOp("View_Left");
    std::vector<int64_t> offsetLeft = {0, 0};
    auto attrLeft = std::make_shared<ViewOpAttribute>(offsetLeft, MemoryType::MEM_UNKNOWN);
    View_Left->SetOpAttribute(attrLeft);

    // View 获取 [16, 16] 右半边 [16, 8]
    G.AddTensor(DataType::DT_FP32, shape2, "sub_out_right");
    auto subOutRight = G.GetTensor("sub_out_right");
    subOutRight->SetMemoryTypeBoth(MemoryType::MEM_UNKNOWN, true);
    G.AddOp(Opcode::OP_VIEW, {"sub_out"}, {"sub_out_right"}, "View_Right");
    auto View_UR =  G.GetOp("View_Right");
    std::vector<int64_t> offsetRight = {0, T};
    auto attrUR = std::make_shared<ViewOpAttribute>(offsetRight, MemoryType::MEM_UNKNOWN);
    View_UR->SetOpAttribute(attrUR);

    // 左半边 [16, 8] + 右半边 [16, 8]
    std::vector<SymbolicScalar> subDynShape = {SymbolicScalar("a"), T};
    G.AddTensor(DataType::DT_FP32, shape2, "add_out");
    G.AddOp(Opcode::OP_ADD, {"sub_out_right", "sub_out_left"}, {"add_out"}, "Add");
    auto addOut = G.GetTensor("add_out");
    addOut->SetMemoryTypeBoth(MemoryType::MEM_UNKNOWN, true);
    addOut->UpdateDynValidShape(subDynShape);

    G.AddOp(Opcode::OP_ASSEMBLE, {"add_out"}, {"out"}, "Assemble_final");
    auto attrAssembleFinal = std::make_shared<AssembleOpAttribute>(MemoryType::MEM_UNKNOWN, std::vector<int64_t> {0, 0});
    auto Op = G.GetOp("Assemble_final");
    Op->SetOpAttribute(attrAssembleFinal);

    auto a = G.GetTensor("a");
    a->SetMemoryTypeBoth(MemoryType::MEM_UNKNOWN, true);
    auto b = G.GetTensor("b");
    b->SetMemoryTypeBoth(MemoryType::MEM_UNKNOWN, true);
    auto out = G.GetTensor("out");
    out->UpdateDynValidShape(subDynShape);

    G.SetInCast({"a", "b"});
    G.SetOutCast({"out"});
    Function *function = G.GetFunction();
    // 确认构图完毕
    constexpr int singleViewOpmagic = 10017;
    constexpr int opNumBefore = 20;
    constexpr int viewNumBefore = 10;
    constexpr int assembleNumBefore = 5;
    auto countResultBefore = CountViewAssemble(*function);
    int viewNumCount = countResultBefore[0];
    int assembleNumCount = countResultBefore[1];
    EXPECT_EQ(function->Operations().size(), opNumBefore) << opNumBefore << " operations before pass";
    EXPECT_EQ(viewNumCount, viewNumBefore) << viewNumBefore << " OP_VIEW before pass";
    EXPECT_EQ(assembleNumCount, assembleNumBefore) << assembleNumBefore << " OP_ASSEMBLE before pass";
    std::cout << "Build Graph Done." << std::endl;
    for (auto &op : function->Operations()) {
        if (op.GetOpMagic() == singleViewOpmagic) {
            auto viewAttr = dynamic_cast<ViewOpAttribute *>(op.GetOpAttribute().get());
            auto offset = viewAttr->GetFromOffset();
            EXPECT_NE(accumulate(offset.begin(), offset.end(), 0), 0) << "OP_VIEW offset should be all zero";
        }
    }
    /*
    dump graph before Pass
    function->DumpJsonFile(jsonFilePath);
    */
    // 单独执行pass
    npu::tile_fwk::SplitLargeFanoutTensor splitLargeFanoutTensor;
    splitLargeFanoutTensor.PreCheck(*function);
    splitLargeFanoutTensor.RunOnFunction(*function);
    splitLargeFanoutTensor.PostCheck(*function);
    std::cout << "Run Pass Done." << std::endl;
    /*
    dump graph after Pass
    function->DumpJsonFile(jsonFilePath);
    */
    auto countResultAfter = CountViewAssemble(*function);
    viewNumCount = countResultAfter[0];
    assembleNumCount = countResultAfter[1];
    EXPECT_EQ(function->Operations().size(), opNumBefore) << opNumBefore << " operations after pass";
    EXPECT_EQ(viewNumCount, viewNumBefore) << viewNumBefore << " OP_VIEW after pass";
    EXPECT_EQ(assembleNumCount, assembleNumBefore) << assembleNumBefore << " OP_ASSEMBLE after pass";
    for (auto &op : function->Operations()) {
        if (op.GetOpMagic() == singleViewOpmagic) {
            auto viewAttr = dynamic_cast<ViewOpAttribute *>(op.GetOpAttribute().get());
            auto offset = viewAttr->GetFromOffset();
            EXPECT_EQ(accumulate(offset.begin(), offset.end(), 0), 0) << "OP_VIEW offset should be all zero";
            auto dynOffset = viewAttr->GetFromDynOffset();
            EXPECT_EQ(dynOffset.size(), 0);
            auto input = op.GetIOperands().front();
            auto inputDynShape = input->GetDynValidShape();
            EXPECT_EQ(inputDynShape.size(), NUM_2);
            EXPECT_EQ(inputDynShape[0].Dump(), "RUNTIME_Max(RUNTIME_Max(0, a), ((a+8)*RUNTIME_Ne(a, 0)))");
        }
    }
}

TEST_F(SplitLargeFanoutTensorTest, PerfectlyMatch_Full) {
    int N = 2;
    int T = 8;
    std::vector<int64_t> shape0{N * T, N * T};
    std::vector<int64_t> shape1{T, T};
    ComputationalGraphBuilder G;
    G.AddTensor(DataType::DT_FP32, shape0, "a");
    G.AddTensor(DataType::DT_FP32, shape0, "b");
    G.AddTensor(DataType::DT_FP32, shape1, "out");
    G.AddTensor(DataType::DT_FP32, shape0, "sub_out");
    // [16, 16] --> View --> [8, 8] --> Sub --> [8, 8] --> Assemble --> [16, 16]
    TileExpandSub(G, N, T);
    auto subOut = G.GetTensor("sub_out");
    subOut->SetMemoryTypeBoth(MemoryType::MEM_UNKNOWN, true);

    // View 获取 [16, 16] 左上角 [8, 8]
    G.AddTensor(DataType::DT_FP32, shape1, "sub_out_upper_right");
    auto subOutUR = G.GetTensor("sub_out_upper_right");
    subOutUR->SetMemoryTypeBoth(MemoryType::MEM_UNKNOWN, true);
    G.AddOp(Opcode::OP_VIEW, {"sub_out"}, {"sub_out_upper_right"}, "View_Upper_Right");
    auto View_UR =  G.GetOp("View_Upper_Right");
    std::vector<int64_t> offsetUR = {0, T};
    auto attrUR = std::make_shared<ViewOpAttribute>(offsetUR, MemoryType::MEM_UNKNOWN);
    View_UR->SetOpAttribute(attrUR);

    // View 获取 [16, 16] 右上角 [8, 8]
    G.AddTensor(DataType::DT_FP32, shape1, "sub_out_lower_left");
    auto subOutLL = G.GetTensor("sub_out_lower_left");
    subOutLL->SetMemoryTypeBoth(MemoryType::MEM_UNKNOWN, true);
    G.AddOp(Opcode::OP_VIEW, {"sub_out"}, {"sub_out_lower_left"}, "View_Lower_Left");
    auto View_LL =  G.GetOp("View_Lower_Left");
    std::vector<int64_t> offsetLL = {T, 0};
    auto attrLL = std::make_shared<ViewOpAttribute>(offsetLL, MemoryType::MEM_UNKNOWN);
    View_LL->SetOpAttribute(attrLL);

    // 左上角 [8, 8] + 右上角 [8, 8] --> [8, 8]
    G.AddTensor(DataType::DT_FP32, shape1, "add_out");
    G.AddOp(Opcode::OP_ADD, {"sub_out_upper_right", "sub_out_lower_left"}, {"add_out"}, "Add");
    auto addOut = G.GetTensor("add_out");
    addOut->SetMemoryTypeBoth(MemoryType::MEM_UNKNOWN, true);

    G.AddOp(Opcode::OP_ASSEMBLE, {"add_out"}, {"out"}, "Assemble_final");
    auto attrAssembleFinal = std::make_shared<AssembleOpAttribute>(MemoryType::MEM_UNKNOWN, std::vector<int64_t> {0, 0});
    auto Op = G.GetOp("Assemble_final");
    Op->SetOpAttribute(attrAssembleFinal);

    auto a = G.GetTensor("a");
    a->SetMemoryTypeBoth(MemoryType::MEM_UNKNOWN, true);
    auto b = G.GetTensor("b");
    b->SetMemoryTypeBoth(MemoryType::MEM_UNKNOWN, true);
    auto out = G.GetTensor("out");
    out->SetMemoryTypeBoth(MemoryType::MEM_UNKNOWN, true);

    G.SetInCast({"a", "b"});
    G.SetOutCast({"out"});
    Function *function = G.GetFunction();
    // 确认构图完毕
    constexpr int opNumBefore = 20;
    constexpr int viewNumBefore = 10;
    constexpr int assembleNumBefore = 5;
    auto countResultBefore = CountViewAssemble(*function);
    int viewNumCount = countResultBefore[0];
    int assembleNumCount = countResultBefore[1];
    EXPECT_EQ(function->Operations().size(), opNumBefore) << opNumBefore << " operations before pass";
    EXPECT_EQ(viewNumCount, viewNumBefore) << viewNumBefore << " OP_VIEW before pass";
    EXPECT_EQ(assembleNumCount, assembleNumBefore) << assembleNumBefore << " OP_ASSEMBLE before pass";
    std::cout << "Build Graph Done." << std::endl;
    /*
    dump graph after Pass
    function->DumpJsonFile(jsonFilePath);
    */
    // 单独执行pass
    npu::tile_fwk::SplitLargeFanoutTensor splitLargeFanoutTensor;
    splitLargeFanoutTensor.PreCheck(*function);
    splitLargeFanoutTensor.RunOnFunction(*function);
    splitLargeFanoutTensor.PostCheck(*function);
    std::cout << "Run Pass Done." << std::endl;

    constexpr int opNumAfter = 10;
    constexpr int viewNumAfter = 6;
    constexpr int assembleNumAfter = 1;
    auto countResultAfter = CountViewAssemble(*function);
    viewNumCount = countResultAfter[0];
    assembleNumCount = countResultAfter[1];
    EXPECT_EQ(function->Operations().size(), opNumAfter) << opNumAfter << " operations after pass";
    EXPECT_EQ(viewNumCount, viewNumAfter) << viewNumAfter << " OP_VIEW after pass";
    EXPECT_EQ(assembleNumCount, assembleNumAfter) << assembleNumAfter << " OP_ASSEMBLE after pass";
}

TEST_F(SplitLargeFanoutTensorTest, PerfectlyMatch_Full_V2) {
    int N = 2;
    int T = 8;
    std::vector<int64_t> shape0{N * T, N * T};
    std::vector<int64_t> shape1{T, T};
    ComputationalGraphBuilder G;
    G.AddTensor(DataType::DT_FP32, shape0, "a");
    G.AddTensor(DataType::DT_FP32, shape0, "b");
    G.AddTensor(DataType::DT_FP32, shape1, "out");
    G.AddTensor(DataType::DT_FP32, shape0, "out2");
    G.AddTensor(DataType::DT_FP32, shape0, "sub_out");
    // [16, 16] --> View --> [8, 8] --> Sub --> [8, 8] --> Assemble --> [16, 16]
    TileExpandSub(G, N, T);
    auto subOut = G.GetTensor("sub_out");
    subOut->SetMemoryTypeBoth(MemoryType::MEM_UNKNOWN, true);

    // [16, 16] --> exp --> [16, 16] --> Assemble --> OCAST
    G.AddTensor(DataType::DT_FP32, shape0, "expOut");
    auto expOut = G.GetTensor("expOut");
    expOut->SetMemoryTypeBoth(MemoryType::MEM_UNKNOWN, true);
    G.AddOp(Opcode::OP_EXP, {"sub_out"}, {"expOut"}, "Exp");
    G.AddOp(Opcode::OP_ASSEMBLE, {"expOut"}, {"out2"}, "Assemble_exp");
    auto attrAssembleExp = std::make_shared<AssembleOpAttribute>(MemoryType::MEM_UNKNOWN, std::vector<int64_t> {0, 0});
    auto assembleExp = G.GetOp("Assemble_exp");
    assembleExp->SetOpAttribute(attrAssembleExp);

    // View 获取 [16, 16] 左上角 [8, 8]
    G.AddTensor(DataType::DT_FP32, shape1, "sub_out_upper_right");
    auto subOutUR = G.GetTensor("sub_out_upper_right");
    subOutUR->SetMemoryTypeBoth(MemoryType::MEM_UNKNOWN, true);
    G.AddOp(Opcode::OP_VIEW, {"sub_out"}, {"sub_out_upper_right"}, "View_Upper_Right");
    auto View_UR =  G.GetOp("View_Upper_Right");
    std::vector<int64_t> offsetUR = {0, T};
    auto attrUR = std::make_shared<ViewOpAttribute>(offsetUR, MemoryType::MEM_UNKNOWN);
    View_UR->SetOpAttribute(attrUR);

    // View 获取 [16, 16] 右上角 [8, 8]
    G.AddTensor(DataType::DT_FP32, shape1, "sub_out_lower_left");
    auto subOutLL = G.GetTensor("sub_out_lower_left");
    subOutLL->SetMemoryTypeBoth(MemoryType::MEM_UNKNOWN, true);
    G.AddOp(Opcode::OP_VIEW, {"sub_out"}, {"sub_out_lower_left"}, "View_Lower_Left");
    auto View_LL =  G.GetOp("View_Lower_Left");
    std::vector<int64_t> offsetLL = {T, 0};
    auto attrLL = std::make_shared<ViewOpAttribute>(offsetLL, MemoryType::MEM_UNKNOWN);
    View_LL->SetOpAttribute(attrLL);

    // 左上角 [8, 8] + 右上角 [8, 8] --> [8, 8]
    G.AddTensor(DataType::DT_FP32, shape1, "add_out");
    G.AddOp(Opcode::OP_ADD, {"sub_out_upper_right", "sub_out_lower_left"}, {"add_out"}, "Add");
    auto addOut = G.GetTensor("add_out");
    addOut->SetMemoryTypeBoth(MemoryType::MEM_UNKNOWN, true);

    G.AddOp(Opcode::OP_ASSEMBLE, {"add_out"}, {"out"}, "Assemble_final");
    auto attrAssembleFinal = std::make_shared<AssembleOpAttribute>(MemoryType::MEM_UNKNOWN, std::vector<int64_t> {0, 0});
    auto Op = G.GetOp("Assemble_final");
    Op->SetOpAttribute(attrAssembleFinal);

    auto a = G.GetTensor("a");
    a->SetMemoryTypeBoth(MemoryType::MEM_UNKNOWN, true);
    auto b = G.GetTensor("b");
    b->SetMemoryTypeBoth(MemoryType::MEM_UNKNOWN, true);
    auto out = G.GetTensor("out");
    out->SetMemoryTypeBoth(MemoryType::MEM_UNKNOWN, true);
    auto out2 = G.GetTensor("out2");
    out2->SetMemoryTypeBoth(MemoryType::MEM_UNKNOWN, true);

    G.SetInCast({"a", "b"});
    G.SetOutCast({"out", "out2"});
    Function *function = G.GetFunction();
    // 确认构图完毕
    constexpr int opNumBefore = 22;
    constexpr int viewNumBefore = 10;
    constexpr int assembleNumBefore = 6;
    auto countResultBefore = CountViewAssemble(*function);
    int viewNumCount = countResultBefore[0];
    int assembleNumCount = countResultBefore[1];
    EXPECT_EQ(function->Operations().size(), opNumBefore) << opNumBefore << " operations before pass";
    EXPECT_EQ(viewNumCount, viewNumBefore) << viewNumBefore << " OP_VIEW before pass";
    EXPECT_EQ(assembleNumCount, assembleNumBefore) << assembleNumBefore << " OP_ASSEMBLE before pass";
    std::cout << "Build Graph Done." << std::endl;
    /*
    dump graph after Pass
    function->DumpJsonFile(jsonFilePath);
    */
    // 单独执行pass
    npu::tile_fwk::SplitLargeFanoutTensor splitLargeFanoutTensor;
    splitLargeFanoutTensor.PreCheck(*function);
    splitLargeFanoutTensor.RunOnFunction(*function);
    splitLargeFanoutTensor.PostCheck(*function);
    std::cout << "Run Pass Done." << std::endl;

    auto countResultAfter = CountViewAssemble(*function);
    viewNumCount = countResultAfter[0];
    assembleNumCount = countResultAfter[1];
    EXPECT_EQ(function->Operations().size(), opNumBefore) << opNumBefore << " operations after pass";
    EXPECT_EQ(viewNumCount, viewNumBefore) << viewNumBefore << " OP_VIEW after pass";
    EXPECT_EQ(assembleNumCount, assembleNumBefore) << assembleNumBefore << " OP_ASSEMBLE after pass";
}

TEST_F(SplitLargeFanoutTensorTest, OneViewOneAssemble) {
    int NUM_32 = 32;
    int NUM_64 = 64;
    int NUM_96 = 96;
    int NUM_128 = 128;
    std::vector<int64_t> shape0{NUM_128, NUM_64};
    std::vector<int64_t> shape1{NUM_32, NUM_64};
    std::vector<int64_t> shape2{NUM_64, NUM_64};
    ComputationalGraphBuilder G;
    G.AddTensor(DataType::DT_FP32, shape0, "a");
    auto a = G.GetTensor("a");
    a->SetMemoryTypeBoth(MemoryType::MEM_UNKNOWN, true);
    G.AddTensor(DataType::DT_FP32, shape2, "out");
    auto out = G.GetTensor("out");
    out->SetMemoryTypeBoth(MemoryType::MEM_UNKNOWN, true);

    // a[128, 64] --> View(0, 0) --> a_ub[128, 64]
    G.AddTensor(DataType::DT_FP32, shape0, "a_ub");
    auto a_ub = G.GetTensor("a_ub");
    a_ub->SetMemoryTypeBoth(MemoryType::MEM_UNKNOWN, true);
    G.AddOp(Opcode::OP_VIEW, {"a"}, {"a_ub"}, "View_To_Ub");
    auto view2ub =  G.GetOp("View_To_Ub");
    std::vector<int64_t> offset2ub = {0, 0};
    auto attr2ub = std::make_shared<ViewOpAttribute>(offset2ub, MemoryType::MEM_UNKNOWN);
    view2ub->SetOpAttribute(attr2ub);

    /*
    a_ub[128, 64] --> View1(32, 0) --> tensor1[32, 64] --> Assemble1(0, 0) -->\
                  \--> View2(96, 0) --> tensor2[32, 64] --> Assemble2(32, 0) --> tensor3[64, 64] --> Exp --> tensor4 --> Assemble --> out
    */
    G.AddTensor(DataType::DT_FP32, shape1, "tensor1");
    auto tensor1 = G.GetTensor("tensor1");
    tensor1->SetMemoryTypeBoth(MemoryType::MEM_UNKNOWN, true);
    G.AddOp(Opcode::OP_VIEW, {"a_ub"}, {"tensor1"}, "View_1");
    auto viewOp1 =  G.GetOp("View_1");
    std::vector<int64_t> offset1 = {NUM_32, 0};
    auto attrView1 = std::make_shared<ViewOpAttribute>(offset1, MemoryType::MEM_UNKNOWN);
    viewOp1->SetOpAttribute(attrView1);

    G.AddTensor(DataType::DT_FP32, shape1, "tensor2");
    auto tensor2 = G.GetTensor("tensor2");
    tensor2->SetMemoryTypeBoth(MemoryType::MEM_UNKNOWN, true);
    G.AddOp(Opcode::OP_VIEW, {"a_ub"}, {"tensor2"}, "View_2");
    auto viewOp2 =  G.GetOp("View_2");
    std::vector<int64_t> offset2 = {NUM_96, 0};
    auto attrView2 = std::make_shared<ViewOpAttribute>(offset2, MemoryType::MEM_UNKNOWN);
    viewOp2->SetOpAttribute(attrView2);

    G.AddTensor(DataType::DT_FP32, shape2, "tensor3");
    auto tensor3 = G.GetTensor("tensor3");
    tensor3->SetMemoryTypeBoth(MemoryType::MEM_UNKNOWN, true);
    G.AddOp(Opcode::OP_ASSEMBLE, {"tensor1"}, {"tensor3"}, "Assemble_1");
    auto attrAssemble1 = std::make_shared<AssembleOpAttribute>(MemoryType::MEM_UNKNOWN, std::vector<int64_t> {0, 0});
    auto assembleOp1 = G.GetOp("Assemble_1");
    assembleOp1->SetOpAttribute(attrAssemble1);
    G.AddOp(Opcode::OP_ASSEMBLE, {"tensor2"}, {"tensor3"}, "Assemble_2");
    auto attrAssemble2 = std::make_shared<AssembleOpAttribute>(MemoryType::MEM_UNKNOWN, std::vector<int64_t> {NUM_32, 0});
    auto assembleOp2 = G.GetOp("Assemble_2");
    assembleOp2->SetOpAttribute(attrAssemble2);

    G.AddTensor(DataType::DT_FP32, shape2, "tensor4");
    auto tensor4 = G.GetTensor("tensor4");
    tensor4->SetMemoryTypeBoth(MemoryType::MEM_UNKNOWN, true);
    G.AddOp(Opcode::OP_EXP, {"tensor3"}, {"tensor4"}, "Exp");
    G.AddOp(Opcode::OP_ASSEMBLE, {"tensor4"}, {"out"}, "Assemble_Final");
    auto attrAssembleFinal = std::make_shared<AssembleOpAttribute>(MemoryType::MEM_UNKNOWN, std::vector<int64_t> {0, 0});
    auto assembleFinal = G.GetOp("Assemble_Final");
    assembleFinal->SetOpAttribute(attrAssembleFinal);

    G.SetInCast({"a"});
    G.SetOutCast({"out"});
    Function *function = G.GetFunction();
    // 确认构图完毕
    constexpr int opNumBefore = 7;
    constexpr int viewNumBefore = 3;
    constexpr int assembleNumBefore = 3;
    auto countResultBefore = CountViewAssemble(*function);
    int viewNumCount = countResultBefore[0];
    int assembleNumCount = countResultBefore[1];
    EXPECT_EQ(function->Operations().size(), opNumBefore) << opNumBefore << " operations before pass";
    EXPECT_EQ(viewNumCount, viewNumBefore) << viewNumBefore << " OP_VIEW before pass";
    EXPECT_EQ(assembleNumCount, assembleNumBefore) << assembleNumBefore << " OP_ASSEMBLE before pass";
    std::cout << "Build Graph Done." << std::endl;
    /*
    dump graph after Pass
    function->DumpJsonFile(jsonFilePath);
    */
    // 单独执行pass
    npu::tile_fwk::SplitLargeFanoutTensor splitLargeFanoutTensor;
    splitLargeFanoutTensor.PreCheck(*function);
    splitLargeFanoutTensor.RunOnFunction(*function);
    splitLargeFanoutTensor.PostCheck(*function);
    std::cout << "Run Pass Done." << std::endl;

    constexpr int opNumAfter = 6;
    constexpr int viewNumAfter = 2;
    constexpr int assembleNumAfter = 3;
    auto countResultAfter = CountViewAssemble(*function);
    viewNumCount = countResultAfter[0];
    assembleNumCount = countResultAfter[1];
    EXPECT_EQ(function->Operations().size(), opNumAfter) << opNumAfter << " operations after pass";
    EXPECT_EQ(viewNumCount, viewNumAfter) << viewNumAfter << " OP_VIEW after pass";
    EXPECT_EQ(assembleNumCount, assembleNumAfter) << assembleNumAfter << " OP_ASSEMBLE after pass";
}

TEST_F(SplitLargeFanoutTensorTest, OneViewMultiAssemble) {
    int NUM_32 = 32;
    int NUM_64 = 64;
    int NUM_128 = 128;
    std::vector<int64_t> shape0{NUM_32, NUM_64};
    std::vector<int64_t> shape1{NUM_64, NUM_64};
    std::vector<int64_t> shape2{NUM_32, NUM_128};
    ComputationalGraphBuilder G;
    G.AddTensor(DataType::DT_FP32, shape1, "a");
    auto a = G.GetTensor("a");
    a->SetMemoryTypeBoth(MemoryType::MEM_UNKNOWN, true);
    G.AddTensor(DataType::DT_FP32, shape1, "out1");
    auto out1 = G.GetTensor("out1");
    out1->SetMemoryTypeBoth(MemoryType::MEM_UNKNOWN, true);
    G.AddTensor(DataType::DT_FP32, shape2, "out2");
    auto out2 = G.GetTensor("out2");
    out2->SetMemoryTypeBoth(MemoryType::MEM_UNKNOWN, true);

    // a[64, 64] --> View(0, 0) --> a_ub[64, 64] --> Muls --> a_ub_new[64, 64]
    G.AddTensor(DataType::DT_FP32, shape1, "a_ub");
    auto a_ub = G.GetTensor("a_ub");
    a_ub->SetMemoryTypeBoth(MemoryType::MEM_UNKNOWN, true);
    G.AddOp(Opcode::OP_VIEW, {"a"}, {"a_ub"}, "View_To_Ub");
    auto view2ub =  G.GetOp("View_To_Ub");
    std::vector<int64_t> offset2ub = {0, 0};
    auto attr2ub = std::make_shared<ViewOpAttribute>(offset2ub, MemoryType::MEM_UNKNOWN);
    view2ub->SetOpAttribute(attr2ub);
    G.AddTensor(DataType::DT_FP32, shape1, "a_ub_new");
    auto a_ub_new = G.GetTensor("a_ub_new");
    a_ub_new->SetMemoryTypeBoth(MemoryType::MEM_UNKNOWN, true);
    G.AddOp(Opcode::OP_MULS, {"a_ub"}, {"a_ub_new"}, "Muls");

    /*
    a_ub_new[64, 64] --> View1(0, 0) --> tensor1[32, 64] --> Assemble1(0, 0) -->\
                  \--> View2(32, 0) --> tensor2[32, 64] --> Assemble2(32, 0) --> tensor3[64, 64] --> Exp --> tensor4 --> AssembleOut1 --> out1
    */
    G.AddTensor(DataType::DT_FP32, shape0, "tensor1");
    auto tensor1 = G.GetTensor("tensor1");
    tensor1->SetMemoryTypeBoth(MemoryType::MEM_UNKNOWN, true);
    G.AddOp(Opcode::OP_VIEW, {"a_ub_new"}, {"tensor1"}, "View_1");
    auto viewOp1 =  G.GetOp("View_1");
    std::vector<int64_t> offset1 = {0, 0};
    auto attrView1 = std::make_shared<ViewOpAttribute>(offset1, MemoryType::MEM_UNKNOWN);
    viewOp1->SetOpAttribute(attrView1);

    G.AddTensor(DataType::DT_FP32, shape0, "tensor2");
    auto tensor2 = G.GetTensor("tensor2");
    tensor2->SetMemoryTypeBoth(MemoryType::MEM_UNKNOWN, true);
    G.AddOp(Opcode::OP_VIEW, {"a_ub_new"}, {"tensor2"}, "View_2");
    auto viewOp2 =  G.GetOp("View_2");
    std::vector<int64_t> offset2 = {NUM_32, 0};
    auto attrView2 = std::make_shared<ViewOpAttribute>(offset2, MemoryType::MEM_UNKNOWN);
    viewOp2->SetOpAttribute(attrView2);

    G.AddTensor(DataType::DT_FP32, shape1, "tensor3");
    auto tensor3 = G.GetTensor("tensor3");
    tensor3->SetMemoryTypeBoth(MemoryType::MEM_UNKNOWN, true);
    G.AddOp(Opcode::OP_ASSEMBLE, {"tensor1"}, {"tensor3"}, "Assemble_1");
    auto attrAssemble1 = std::make_shared<AssembleOpAttribute>(MemoryType::MEM_UNKNOWN, std::vector<int64_t> {0, 0});
    auto assembleOp1 = G.GetOp("Assemble_1");
    assembleOp1->SetOpAttribute(attrAssemble1);
    G.AddOp(Opcode::OP_ASSEMBLE, {"tensor2"}, {"tensor3"}, "Assemble_2");
    auto attrAssemble2 = std::make_shared<AssembleOpAttribute>(MemoryType::MEM_UNKNOWN, std::vector<int64_t> {NUM_32, 0});
    auto assembleOp2 = G.GetOp("Assemble_2");
    assembleOp2->SetOpAttribute(attrAssemble2);

    G.AddTensor(DataType::DT_FP32, shape1, "tensor4");
    auto tensor4 = G.GetTensor("tensor4");
    tensor4->SetMemoryTypeBoth(MemoryType::MEM_UNKNOWN, true);
    G.AddOp(Opcode::OP_EXP, {"tensor3"}, {"tensor4"}, "Exp");
    G.AddOp(Opcode::OP_ASSEMBLE, {"tensor4"}, {"out1"}, "Assemble_Out1");
    auto attrAssembleOut1 = std::make_shared<AssembleOpAttribute>(MemoryType::MEM_UNKNOWN, std::vector<int64_t> {0, 0});
    auto assembleOut1 = G.GetOp("Assemble_Out1");
    assembleOut1->SetOpAttribute(attrAssembleOut1);

    /*
    a_ub_new[64, 64] --> View1(0, 0) --> tensor1[32, 64] --> Assemble3(0, 0) -->\
                  \--> View2(32, 0) --> tensor2[32, 64] --> Assemble4(0, 64) --> tensor5[32, 128] --> Exp --> tensor6 --> AssembleOut2 --> out2
    */
    G.AddTensor(DataType::DT_FP32, shape2, "tensor5");
    auto tensor5 = G.GetTensor("tensor5");
    tensor5->SetMemoryTypeBoth(MemoryType::MEM_UNKNOWN, true);
    G.AddOp(Opcode::OP_ASSEMBLE, {"tensor1"}, {"tensor5"}, "Assemble_3");
    auto attrAssemble3 = std::make_shared<AssembleOpAttribute>(MemoryType::MEM_UNKNOWN, std::vector<int64_t> {0, 0});
    auto assembleOp3 = G.GetOp("Assemble_3");
    assembleOp3->SetOpAttribute(attrAssemble3);
    G.AddOp(Opcode::OP_ASSEMBLE, {"tensor2"}, {"tensor5"}, "Assemble_4");
    auto attrAssemble4 = std::make_shared<AssembleOpAttribute>(MemoryType::MEM_UNKNOWN, std::vector<int64_t> {0, NUM_64});
    auto assembleOp4 = G.GetOp("Assemble_4");
    assembleOp4->SetOpAttribute(attrAssemble4);

    G.AddTensor(DataType::DT_FP32, shape2, "tensor6");
    auto tensor6 = G.GetTensor("tensor6");
    tensor6->SetMemoryTypeBoth(MemoryType::MEM_UNKNOWN, true);
    G.AddOp(Opcode::OP_ABS, {"tensor5"}, {"tensor6"}, "Abs");
    G.AddOp(Opcode::OP_ASSEMBLE, {"tensor6"}, {"out2"}, "Assemble_Out2");
    auto attrAssembleOut2 = std::make_shared<AssembleOpAttribute>(MemoryType::MEM_UNKNOWN, std::vector<int64_t> {0, 0});
    auto assembleOut2 = G.GetOp("Assemble_Out2");
    assembleOut2->SetOpAttribute(attrAssembleOut2);

    G.SetInCast({"a"});
    G.SetOutCast({"out1", "out2"});
    Function *function = G.GetFunction();
    // 确认构图完毕
    constexpr int opNumBefore = 12;
    constexpr int viewNumBefore = 3;
    constexpr int assembleNumBefore = 6;
    auto countResultBefore = CountViewAssemble(*function);
    int viewNumCount = countResultBefore[0];
    int assembleNumCount = countResultBefore[1];
    EXPECT_EQ(function->Operations().size(), opNumBefore) << opNumBefore << " operations before pass";
    EXPECT_EQ(viewNumCount, viewNumBefore) << viewNumBefore << " OP_VIEW before pass";
    EXPECT_EQ(assembleNumCount, assembleNumBefore) << assembleNumBefore << " OP_ASSEMBLE before pass";
    std::cout << "Build Graph Done." << std::endl;
    /*
    dump graph after Pass
    function->DumpJsonFile(jsonFilePath);
    */
    // 单独执行pass
    npu::tile_fwk::SplitLargeFanoutTensor splitLargeFanoutTensor;
    splitLargeFanoutTensor.PreCheck(*function);
    splitLargeFanoutTensor.RunOnFunction(*function);
    splitLargeFanoutTensor.PostCheck(*function);
    std::cout << "Run Pass Done." << std::endl;

    auto countResultAfter = CountViewAssemble(*function);
    viewNumCount = countResultAfter[0];
    assembleNumCount = countResultAfter[1];
    EXPECT_EQ(function->Operations().size(), opNumBefore) << opNumBefore << " operations after pass";
    EXPECT_EQ(viewNumCount, viewNumBefore) << viewNumBefore << " OP_VIEW after pass";
    EXPECT_EQ(assembleNumCount, assembleNumBefore) << assembleNumBefore << " OP_ASSEMBLE after pass";
}

void BuildComplexOverlap(ComputationalGraphBuilder &G){
    int NUM_8 = 8;
    int NUM_16 = 16;
    int NUM_32 = 32;

    // 定义所有张量的形状和名称并添加
    std::map<std::string, std::vector<int64_t>> tensors = {
        {"a", {NUM_8, NUM_32}}, {"b", {NUM_8, NUM_32}}, {"c", {NUM_16, NUM_8}},
        {"d", {NUM_16, NUM_8}}, {"e", {NUM_8, NUM_16}}, {"f", {NUM_8, NUM_16}},
        {"out1", {NUM_8, NUM_16}}, {"out2", {NUM_8, NUM_16}}, {"out3", {NUM_32, NUM_8}},
        {"out4", {NUM_32, NUM_8}}, {"out5", {NUM_16, NUM_8}}, {"out6", {NUM_16, NUM_8}},
        {"largeTensor", {NUM_32, NUM_32}}
    };
    for (const auto& [name, shape] : tensors) {
        G.AddTensor(DataType::DT_FP32, shape, name);
        auto tensor = G.GetTensor(name);
        tensor->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    }

    // 定义所有ASSEMBLE操作并添加
    std::vector<std::tuple<std::string, std::string, std::vector<int64_t>>> assembleOps = {
        {"a", "Assemble_A", {0, 0}}, {"b", "Assemble_B", {24, 0}}, {"c", "Assemble_C", {8, 0}},
        {"d", "Assemble_D", {8, 24}}, {"e", "Assemble_E", {8, 8}}, {"f", "Assemble_F", {16, 8}}
    };
    for (const auto& [input, opName, offset] : assembleOps) {
        G.AddOp(Opcode::OP_ASSEMBLE, {input}, {"largeTensor"}, opName);
        auto assembleOp = G.GetOp(opName);
        assembleOp->SetOpAttribute(std::make_shared<AssembleOpAttribute>(MemoryType::MEM_DEVICE_DDR, offset));
    }

    // 定义所有VIEW操作并添加
    std::vector<std::tuple<std::string, std::vector<int64_t>>> viewOps = {
        {"out1", {0, 8}}, {"out2", {24, 8}}, {"out3", {0, 0}},
        {"out4", {0, 24}}, {"out5", {8, 8}}, {"out6", {8, 16}}
    };
    for (const auto& [output, offset] : viewOps) {
        std::string opName = "View_" + output.substr(3);
        G.AddOp(Opcode::OP_VIEW, {"largeTensor"}, {output}, opName);
        auto viewOp = G.GetOp(opName);
        viewOp->SetOpAttribute(std::make_shared<ViewOpAttribute>(offset, MemoryType::MEM_DEVICE_DDR));
    }

    G.SetInCast({"a", "b", "c", "d", "e", "f"});
    G.SetOutCast({"out1", "out2", "out3", "out4", "out5", "out6"});
}

/*
    input[shape]:       a[8, 32]    b[8, 32]    c[16, 8]    d[16, 8]    e[8, 16]    f[8, 16]
    assemble offset:    [0, 0]      [24, 0]     [8, 0]      [8, 24]     [8, 8]      [16, 8]
    largeTensor[shape]:                         largeTensor[32, 32]
    view offset:        [0, 8]      [24, 8]     [0, 0]      [0, 24]     [8, 8]      [8, 16]
    output[shape]:      out1[8, 16] out2[8, 16] out3[32, 8] out4[32, 8] out5[16, 8] out6[16, 8]
*/
TEST_F(SplitLargeFanoutTensorTest, ComplexOverlap) {
    ComputationalGraphBuilder G;
    BuildComplexOverlap(G);
    Function *function = G.GetFunction();

    std::cout << "Build Graph Done." << std::endl;
    // 单独执行pass
    npu::tile_fwk::SplitLargeFanoutTensor splitLargeFanoutTensor;
    splitLargeFanoutTensor.enableMoreSplit = false;
    splitLargeFanoutTensor.PreCheck(*function);
    splitLargeFanoutTensor.RunOnFunction(*function);
    splitLargeFanoutTensor.PostCheck(*function);
    std::cout << "Run Pass Done." << std::endl;

    // 验证：
    // 拆分后除了两个incast分别各cover一个outcast的场景会被单独拆出以外
    // 其余场景均不会被拆分，依旧走原largeTensor
    std::unordered_map<int, int> recordAssemble;
    std::unordered_map<int, int> recordView;
    for (auto &op: function->Operations()) {
        if (op.GetOpcode() == Opcode::OP_ASSEMBLE) {
            recordAssemble[op.oOperand.front()->GetMagic()]++;
        }
        if (op.GetOpcode() == Opcode::OP_VIEW) {
            recordView[op.iOperand.front()->GetMagic()]++;
        }
    }
    for (auto &[k, v]: recordAssemble) {
        EXPECT_EQ(recordView[k], (v == 1) ? 1 : 4); //6个output，排除两个被input包含的，剩余4个
    }
}

/*
    输入是8个{16， 656}，输出是{16， 512}, {16， 128}各两个。
    其中两个输入与两个输出形成一对一，一个输入对应两个输出的和，其余五个输入未被使用
    拆分后理应得到
    {16， 656} -> view -> {16， 512}
    {16， 656} -> view -> {16， 128}
    {16， 656} -> view -> {16， 512} + {16， 128}
    {16， 656}
    {16， 656}
    {16， 656}
    {16， 656}
    {16， 656}
*/
void BuildPartialInputUnusedGraph(ComputationalGraphBuilder &G){
    G.AddTensor(DataType::DT_FP32, {128, 656}, "largeTensor");
    auto largeTensor = G.GetTensor("largeTensor");
    largeTensor->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);

    std::map<std::string, Shape> logicTensors = {
        {"in1", {16, 656}}, {"in2", {16, 656}}, {"in3", {16, 656}}, {"in4", {16, 656}},
        {"in5", {16, 656}}, {"in6", {16, 656}}, {"in7", {16, 656}}, {"in8", {16, 656}},
        {"out11", {16, 512}}, {"out81", {16, 512}},
        {"out22", {16, 128}}, {"out82", {16, 128}},
        {"out33", {16, 16}}, {"out83", {16, 16}}
    };
    for (const auto& [name, shape] : logicTensors) {
        G.AddTensor(DataType::DT_FP32, shape, name);
        auto logicTensor = G.GetTensor(name);
        logicTensor->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    }
    G.SetInCast({"in1", "in2", "in3", "in4", "in5", "in6", "in7", "in8"});
    G.SetOutCast({"out11", "out22", "out81", "out82"});

    std::vector<std::tuple<std::string, std::string, std::string, Shape>> assembleOps = {
        {"in1", "largeTensor", "assemble1", {0, 0}}, {"in2", "largeTensor", "assemble2", {16, 0}},
        {"in3", "largeTensor", "assemble3", {32, 0}}, {"in4", "largeTensor", "assemble4", {48, 0}},
        {"in5", "largeTensor", "assemble5", {64, 0}}, {"in6", "largeTensor", "assemble6", {80, 0}},
        {"in7", "largeTensor", "assemble7", {96, 0}}, {"in8", "largeTensor", "assemble8", {112, 0}}
    };
    for (const auto& [input, output, opName, offset] : assembleOps) {
        G.AddOp(Opcode::OP_ASSEMBLE, {input}, {output}, opName);
        auto assembleOp = G.GetOp(opName);
        assembleOp->SetOpAttribute(std::make_shared<AssembleOpAttribute>(MemoryType::MEM_DEVICE_DDR, offset));
    }

    std::vector<std::tuple<std::string, std::string, std::string, Shape>> viewOps = {
        {"largeTensor", "out11", "view11", {0, 0}}, {"largeTensor", "out81", "view81", {112, 0}},
        {"largeTensor", "out22", "view22", {16, 512}}, {"largeTensor", "out82", "view82", {112, 512}}
    };
    for (const auto& [input, output, opName, offset] : viewOps) {
        G.AddOp(Opcode::OP_VIEW, {input}, {output}, opName);
        auto viewOp = G.GetOp(opName);
        viewOp->SetOpAttribute(std::make_shared<ViewOpAttribute>(offset, MemoryType::MEM_DEVICE_DDR));
    }
}

TEST_F(SplitLargeFanoutTensorTest, TestPartialInputUnused) {
    ComputationalGraphBuilder G;
    BuildPartialInputUnusedGraph(G);
    Function *function = G.GetFunction();

    std::cout << "Build Graph Done." << std::endl;
    // 单独执行pass
    npu::tile_fwk::SplitLargeFanoutTensor splitLargeFanoutTensor;
    splitLargeFanoutTensor.enableMoreSplit = false;
    splitLargeFanoutTensor.PreCheck(*function);
    splitLargeFanoutTensor.RunOnFunction(*function);
    splitLargeFanoutTensor.PostCheck(*function);
    std::cout << "Run Pass Done." << std::endl;

    const int viewNum = 4;
    const int assembleNum = 0;
    auto countResultAfter = CountViewAssemble(*function);
    EXPECT_EQ(viewNum, countResultAfter[0]) << countResultAfter[0] << "OP_VIEW after pass, should be 4";
    EXPECT_EQ(assembleNum, countResultAfter[1]) << countResultAfter[1] << "OP_ASSEMBLE after pass, should be 0";
}

void BuildOneDim(ComputationalGraphBuilder &G, bool shouldSplit){
    int NUM_1 = 1;
    int NUM_15 = 15;
    int NUM_16 = 16;
    int NUM_32 = 32;
    std::vector<int64_t> shape1{NUM_1};
    std::vector<int64_t> shape15{NUM_15};
    std::vector<int64_t> shape16{NUM_16};
    std::vector<int64_t> shape32{NUM_32};

    G.AddTensor(DataType::DT_FP32, shape1, "a");
    auto a = G.GetTensor("a");
    a->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    G.AddTensor(DataType::DT_FP32, shouldSplit ? shape15 : shape16, "b");
    auto b = G.GetTensor("b");
    b->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    G.AddTensor(DataType::DT_FP32, shouldSplit ? shape16 : shape15, "c");
    auto c = G.GetTensor("c");
    c->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);

    G.AddTensor(DataType::DT_FP32, shape16, "out1");
    auto out1 = G.GetTensor("out1");
    out1->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    G.AddTensor(DataType::DT_FP32, shape16, "out2");
    auto out2 = G.GetTensor("out2");
    out2->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);

    G.AddTensor(DataType::DT_FP32, shape32, "largeTensor");
    auto largeTensor = G.GetTensor("largeTensor");
    largeTensor->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);

    G.AddOp(Opcode::OP_ASSEMBLE, {"a"}, {"largeTensor"}, "Assemble_A");
    auto attrAssembleA = std::make_shared<AssembleOpAttribute>(MemoryType::MEM_DEVICE_DDR, std::vector<int64_t> {0});
    auto assembleA = G.GetOp("Assemble_A");
    assembleA->SetOpAttribute(attrAssembleA);
    G.AddOp(Opcode::OP_ASSEMBLE, {"b"}, {"largeTensor"}, "Assemble_B");
    auto attrAssembleB = std::make_shared<AssembleOpAttribute>(MemoryType::MEM_DEVICE_DDR, std::vector<int64_t> {1});
    auto assembleB = G.GetOp("Assemble_B");
    assembleB->SetOpAttribute(attrAssembleB);
    G.AddOp(Opcode::OP_ASSEMBLE, {"c"}, {"largeTensor"}, "Assemble_C");
    auto attrAssembleC = std::make_shared<AssembleOpAttribute>(MemoryType::MEM_DEVICE_DDR, std::vector<int64_t> {shouldSplit ? 16 : 17});
    auto assembleC = G.GetOp("Assemble_C");
    assembleC->SetOpAttribute(attrAssembleC);

    G.AddOp(Opcode::OP_VIEW, {"largeTensor"}, {"out1"}, "View_1");
    auto View_1 = G.GetOp("View_1");
    auto attr1 = std::make_shared<ViewOpAttribute>(std::vector<int64_t> {0}, MemoryType::MEM_DEVICE_DDR);
    View_1->SetOpAttribute(attr1);
    G.AddOp(Opcode::OP_VIEW, {"largeTensor"}, {"out2"}, "View_2");
    auto View_2 = G.GetOp("View_2");
    auto attr2 = std::make_shared<ViewOpAttribute>(std::vector<int64_t> {16}, MemoryType::MEM_DEVICE_DDR);
    View_2->SetOpAttribute(attr2);

    G.SetInCast({"a", "b", "c"});
    G.SetOutCast({"out1", "out2"});
}

// {1} + {15} + {16} --assemble--> {32} --view--> {16} + {16}
// =>   {1} + {15} --assemble--> {16} --view--> {16}
//      {16} --view--> {16}
TEST_F(SplitLargeFanoutTensorTest, OneDimShouldSplit) {
    ComputationalGraphBuilder G;
    BuildOneDim(G, true);
    Function *function = G.GetFunction();

    std::cout << "Build Graph Done." << std::endl;
    // 单独执行pass
    npu::tile_fwk::SplitLargeFanoutTensor splitLargeFanoutTensor;
    splitLargeFanoutTensor.enableMoreSplit = false;
    splitLargeFanoutTensor.PreCheck(*function);
    splitLargeFanoutTensor.RunOnFunction(*function);
    splitLargeFanoutTensor.PostCheck(*function);
    std::cout << "Run Pass Done." << std::endl;

    // 验证：
    // 依据UT注释展示，共会出现2个view和2个assemble
    auto countResultAfter = CountViewAssemble(*function);
    const int viewAssembleNum = 2;
    EXPECT_EQ(viewAssembleNum, countResultAfter[0]) << countResultAfter[0] << " OP_VIEW after pass, should be 2";
    EXPECT_EQ(viewAssembleNum, countResultAfter[1]) << countResultAfter[1] << " OP_ASSEMBLE after pass, should be 2";
}

// {1} + {16} + {15} --assemble--> {32} --view--> {16} + {16}
// 不进行拆分
TEST_F(SplitLargeFanoutTensorTest, OneDimNotSplit) {
    ComputationalGraphBuilder G;
    BuildOneDim(G, false);
    Function *function = G.GetFunction();

    std::cout << "Build Graph Done." << std::endl;
    auto countResultBefore = CountViewAssemble(*function);
    // 单独执行pass
    npu::tile_fwk::SplitLargeFanoutTensor splitLargeFanoutTensor;
    splitLargeFanoutTensor.enableMoreSplit = false;
    splitLargeFanoutTensor.PreCheck(*function);
    splitLargeFanoutTensor.RunOnFunction(*function);
    splitLargeFanoutTensor.PostCheck(*function);
    std::cout << "Run Pass Done." << std::endl;

    // 验证：pass不会切分，所以前后一致
    auto countResultAfter = CountViewAssemble(*function);
    EXPECT_EQ(countResultBefore[0], countResultAfter[0]) << countResultBefore[0] 
        << "OP_VIEW before pass; " << countResultAfter[0] << " OP_VIEW after pass, should equal.";
    EXPECT_EQ(countResultBefore[1], countResultAfter[1]) << countResultBefore[1] 
        << "OP_ASSEMBLE before pass; " << countResultAfter[1] << " OP_ASSEMBLE after pass, should equal.";
}

// {1} + {2} + {1} + {1} --assemble--> {5} --view--> {3} + {1}
void BuildDiffLcmShape(ComputationalGraphBuilder &G){
    int NUM_1 = 1;
    int NUM_2 = 2;
    int NUM_3 = 3;
    int NUM_5 = 5;

    // 定义所有张量的形状和名称并添加
    std::map<std::string, std::vector<int64_t>> tensors = {
        {"a", {NUM_1}}, {"b", {NUM_2}}, {"c", {NUM_1}}, {"d", {NUM_1}},
        {"out1", {NUM_3}}, {"out2", {NUM_1}},
        {"largeTensor", {NUM_5}}
    };
    for (const auto& [name, shape] : tensors) {
        G.AddTensor(DataType::DT_FP32, shape, name);
        auto tensor = G.GetTensor(name);
        tensor->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    }

    // 定义所有ASSEMBLE操作并添加
    std::vector<std::tuple<std::string, std::string, std::vector<int64_t>>> assembleOps = {
        {"a", "Assemble_A", {0}}, {"b", "Assemble_B", {1}},
        {"c", "Assemble_C", {3}}, {"d", "Assemble_D", {4}}
    };
    for (const auto& [input, opName, offset] : assembleOps) {
        G.AddOp(Opcode::OP_ASSEMBLE, {input}, {"largeTensor"}, opName);
        auto assembleOp = G.GetOp(opName);
        assembleOp->SetOpAttribute(std::make_shared<AssembleOpAttribute>(MemoryType::MEM_DEVICE_DDR, offset));
    }

    // 定义所有VIEW操作并添加
    std::vector<std::tuple<std::string, std::vector<int64_t>>> viewOps = {
        {"out1", {0}}, {"out2", {3}}
    };
    for (const auto& [output, offset] : viewOps) {
        std::string opName = "View_" + output.substr(3);
        G.AddOp(Opcode::OP_VIEW, {"largeTensor"}, {output}, opName);
        auto viewOp = G.GetOp(opName);
        viewOp->SetOpAttribute(std::make_shared<ViewOpAttribute>(offset, MemoryType::MEM_DEVICE_DDR));
    }

    G.SetInCast({"a", "b", "c", "d"});
    G.SetOutCast({"out1", "out2"});
}

// {1} + {2} + {1} + {1} --assemble--> {5} --view--> {3} + {1}
// ==> {1} + {2} --assemble--> {3} --view--> {3}
//     {1} --view--> {1}
// 用于验证小的LcmShape被优先尝试拆分，否则第二组拆分结果会得到：
// {1} + {1} --assemble--> {2} --view--> {1}
// 存在冗余的assemble，不符合预期
TEST_F(SplitLargeFanoutTensorTest, SplitSmallTileFirst) {
    ComputationalGraphBuilder G;
    BuildDiffLcmShape(G);
    Function *function = G.GetFunction();

    std::cout << "Build Graph Done." << std::endl;
    // 单独执行pass
    npu::tile_fwk::SplitLargeFanoutTensor splitLargeFanoutTensor;
    splitLargeFanoutTensor.enableMoreSplit = false;
    splitLargeFanoutTensor.PreCheck(*function);
    splitLargeFanoutTensor.RunOnFunction(*function);
    splitLargeFanoutTensor.PostCheck(*function);
    std::cout << "Run Pass Done." << std::endl;

    // 验证：
    // 依据UT注释展示，共会出现2个view和2个assemble
    auto countResultAfter = CountViewAssemble(*function);
    const int viewAssembleNum = 2;
    EXPECT_EQ(viewAssembleNum, countResultAfter[0]) << countResultAfter[0] << " OP_VIEW after pass, should be 2";
    EXPECT_EQ(viewAssembleNum, countResultAfter[1]) << countResultAfter[1] << " OP_ASSEMBLE after pass, should be 2";
}

// {2} + {2} + {3} --assemble--> {7} --view--> {5}
void BuildCoprimeInputOutput(ComputationalGraphBuilder &G){
    int NUM_2 = 2;
    int NUM_3 = 3;
    int NUM_5 = 5;
    int NUM_7 = 7;

    // 定义所有张量的形状和名称并添加
    std::map<std::string, std::vector<int64_t>> tensors = {
        {"a", {NUM_2}}, {"b", {NUM_2}}, {"c", {NUM_3}},
        {"out1", {NUM_5}},
        {"largeTensor", {NUM_7}}
    };
    for (const auto& [name, shape] : tensors) {
        G.AddTensor(DataType::DT_FP32, shape, name);
        auto tensor = G.GetTensor(name);
        tensor->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    }

    // 定义所有ASSEMBLE操作并添加
    std::vector<std::tuple<std::string, std::string, std::vector<int64_t>>> assembleOps = {
        {"a", "Assemble_A", {0}}, {"b", "Assemble_B", {2}}, {"c", "Assemble_C", {4}}
    };
    for (const auto& [input, opName, offset] : assembleOps) {
        G.AddOp(Opcode::OP_ASSEMBLE, {input}, {"largeTensor"}, opName);
        auto assembleOp = G.GetOp(opName);
        assembleOp->SetOpAttribute(std::make_shared<AssembleOpAttribute>(MemoryType::MEM_DEVICE_DDR, offset));
    }

    // 定义所有VIEW操作并添加
    std::vector<std::tuple<std::string, std::vector<int64_t>>> viewOps = {
        {"out1", {0}}
    };
    for (const auto& [output, offset] : viewOps) {
        std::string opName = "View_" + output.substr(3);
        G.AddOp(Opcode::OP_VIEW, {"largeTensor"}, {output}, opName);
        auto viewOp = G.GetOp(opName);
        viewOp->SetOpAttribute(std::make_shared<ViewOpAttribute>(offset, MemoryType::MEM_DEVICE_DDR));
    }

    G.SetInCast({"a", "b", "c"});
    G.SetOutCast({"out1"});
}

// {2} + {2} + {3} --assemble--> {7} --view--> {5}
// 不进行拆分，不会生成新的lcmTile
TEST_F(SplitLargeFanoutTensorTest, NoSplitLcmLargerThanLargeTensor) {
    ComputationalGraphBuilder G;
    BuildCoprimeInputOutput(G);
    Function *function = G.GetFunction();

    std::cout << "Build Graph Done." << std::endl;
    auto countResultBefore = CountViewAssemble(*function);
    std::vector<int> opMagicBefore;
    for (auto &op : function->Operations()) {
        opMagicBefore.emplace_back(op.GetOpMagic());
    }

    // 单独执行pass
    npu::tile_fwk::SplitLargeFanoutTensor splitLargeFanoutTensor;
    splitLargeFanoutTensor.enableMoreSplit = false;
    splitLargeFanoutTensor.PreCheck(*function);
    splitLargeFanoutTensor.RunOnFunction(*function);
    splitLargeFanoutTensor.PostCheck(*function);
    std::cout << "Run Pass Done." << std::endl;

    // 验证：pass不会切分，所以前后一致
    auto countResultAfter = CountViewAssemble(*function);
    std::vector<int> opMagicAfter;
    for (auto &op : function->Operations()) {
        opMagicAfter.emplace_back(op.GetOpMagic());
    }
    EXPECT_EQ(countResultBefore[0], countResultAfter[0]) << countResultBefore[0] 
        << "OP_VIEW before pass; " << countResultAfter[0] << " OP_VIEW after pass, should equal.";
    EXPECT_EQ(countResultBefore[1], countResultAfter[1]) << countResultBefore[1] 
        << "OP_ASSEMBLE before pass; " << countResultAfter[1] << " OP_ASSEMBLE after pass, should equal.";
    EXPECT_EQ(CommonUtils::ContainerToStr(opMagicBefore), CommonUtils::ContainerToStr(opMagicAfter))
        << "All op magic before pass: " << CommonUtils::ContainerToStr(opMagicBefore)
        << "; All op magic after pass: " << CommonUtils::ContainerToStr(opMagicAfter) << "; Op should not change.";
}
} // namespace tile_fwk
} // namespace npu