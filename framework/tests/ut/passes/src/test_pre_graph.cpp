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
 * \file test_pre_graph.cpp
 * \brief Unit test for PreGraph pass.
 */

#include <fstream>
#include <vector>
#include <string>
#include "gtest/gtest.h"
#include "tilefwk/tilefwk_op.h"
#include "interface/function/function.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "passes/pass_mgr/pass_manager.h"
#include "interface/configs/config_manager.h"
#include "passes/tile_graph_pass/graph_constraint/pre_graph/pre_graph.h"
#include "ut_json/ut_json_tool.h"
#include "computational_graph_builder.h"
#define private public

using namespace npu::tile_fwk;

namespace npu {
namespace tile_fwk {
constexpr int NUM10 = 10;

void PrintGraphInfoPreGraph(Function* func, std::set<int>& tensorMagicWithColorSet) {
    std::cout << "func->Operations().size() = "  << func->Operations().size() << std::endl;
    for (auto &op : func->Operations()) {
        std::cout << "Op:" << op.GetOpMagic() << " " <<  op.GetOpcodeStr() << std::endl;
        std::cout << "input operation:";
        for (const std::shared_ptr<LogicalTensor> &input_tensor : op.GetIOperands()) {
            for (const auto &item_op : input_tensor->GetProducers()) {
                std::cout << "(" << item_op->opmagic << ", " << item_op->GetOpcodeStr() << ") ";
            }
            if (input_tensor->GetMemoryTypeOriginal() == npu::tile_fwk::MemoryType::MEM_DEVICE_DDR) {
                continue;
            }
            int curColor = input_tensor->subGraphID;
            std::cout << "input tensor, cur color is " << curColor << std::endl;
            if (curColor > 0) {
                tensorMagicWithColorSet.insert(input_tensor->magic);
                std::cout << "cur input tensor magic is " << input_tensor->magic << std::endl;
            }
        }
        std::cout << std::endl << "output operation:";
        for (const std::shared_ptr<LogicalTensor> &output_tensor : op.GetOOperands()) {
            for (const auto &item_op : output_tensor->GetConsumers()) {
                std::cout << "(" << item_op->opmagic << ", " << item_op->GetOpcodeStr() << ") ";
            }
            if (output_tensor->GetMemoryTypeOriginal() == npu::tile_fwk::MemoryType::MEM_DEVICE_DDR) {
                continue;
            }
            int curColor = output_tensor->subGraphID;
            std::cout << "output tensor, cur color is " << curColor << std::endl;
            if (curColor > 0) {
                tensorMagicWithColorSet.insert(output_tensor->magic);
                std::cout << "cur output tensor magic is " << output_tensor->magic << std::endl;
            }
        }
        std::cout << std::endl;
    }
}

class PreGraphTest : public testing::Test {
public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}

    void SetUp() override {
        Program::GetInstance().Reset();
        config::Reset();
        config::SetPlatformConfig(KEY_ONLY_HOST_COMPILE, true);
        config::SetHostConfig(KEY_STRATEGY, "PreGraphTestStrategy");
        config::SetPlatformConfig("ENABLE_COST_MODEL", false);
    }
    void TearDown() override {}

    // input[B, N, S] --> Copy_In --> input_ub[1, 1, S] --> Transpose_Datamove --> outputGm, [N, B, S](i, j, 0) --> Assemble --> output[N, B, S]
    void TileExpandTransposeDatamove(ComputationalGraphBuilder &G, const int B, const int N, const int S, bool isInner = false) {
        std::vector<int64_t> tileShape{1, 1, S};
        // 所有transpose_datamove输出partial结果都要指向同一个raw tensor
        G.AddTensor(DataType::DT_FP32, {N, B, S}, "temp_out");
        auto tempOut = G.GetTensor("temp_out");
        tempOut->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
        auto incast = G.GetTensor("input");
        for (int i = 0; i < B; i++) {
            for (int j = 0; j < N; j++) {
                std::vector<int64_t> offset = {i, j, 0};
                std::vector<int64_t> offsetNew = {j, i, 0};
                int subgraphId = i * N + j;

                std::string input_ub = "input_ub_" + std::to_string(subgraphId);
                G.AddTensor(DataType::DT_FP32, tileShape, input_ub);
                auto tensorUb = G.GetTensor(input_ub);
                tensorUb->SetMemoryTypeBoth(MemoryType::MEM_UB, true);
                G.AddOp(Opcode::OP_COPY_IN, {"input"}, {input_ub}, "Ub_Copy_In_" + std::to_string(subgraphId));
                auto copyInOp = G.GetOp("Ub_Copy_In_" + std::to_string(subgraphId));
                auto attrCopyIn = std::make_shared<CopyOpAttribute>(
                    OpImmediate::Specified(offset), MemoryType::MEM_UB,
                    OpImmediate::Specified(incast->GetShape()), OpImmediate::Specified(incast->tensor->GetRawShape()));
                copyInOp->SetOpAttribute(attrCopyIn);
                copyInOp->UpdateSubgraphID(subgraphId);

                std::string outputPartial = "output_ddr_" + std::to_string(subgraphId);
                G.AddTensor(DataType::DT_FP32, tileShape, outputPartial);
                auto outputGm = G.GetTensor(outputPartial);
                outputGm->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
                outputGm->tensor = tempOut->tensor;
                outputGm->UpdateOffset(offsetNew);
                G.AddOp(Opcode::OP_TRANSPOSE_MOVEOUT, {input_ub}, {outputPartial}, "Transpose_Datamove_" + std::to_string(subgraphId));
                auto transposeOp = G.GetOp("Transpose_Datamove_" + std::to_string(subgraphId));
                transposeOp->UpdateSubgraphID(subgraphId);

                G.AddOp(Opcode::OP_ASSEMBLE, {outputPartial}, {"output"}, "Assemble_" + std::to_string(subgraphId));
                auto attrAssemble = std::make_shared<AssembleOpAttribute>(MemoryType::MEM_DEVICE_DDR, offsetNew);
                auto assembleOp = G.GetOp("Assemble_" + std::to_string(subgraphId));
                assembleOp->SetOpAttribute(attrAssemble);
                assembleOp->UpdateSubgraphID(subgraphId);

                if (isInner) {
                    auto outInnerTemp = G.GetTensor("outInnerTemp");
                    G.AddOp(Opcode::OP_ASSEMBLE, {outputPartial}, {"outInnerTemp"}, "Assemble_Inner_" + std::to_string(subgraphId));
                    auto attrAssembleInner = std::make_shared<AssembleOpAttribute>(MemoryType::MEM_DEVICE_DDR, offsetNew);
                    auto assembleOpInner = G.GetOp("Assemble_Inner_" + std::to_string(subgraphId));
                    assembleOpInner->SetOpAttribute(attrAssembleInner);
                    assembleOpInner->UpdateSubgraphID(subgraphId);
                }
            }
        }
    }

    // outInnerTemp[N, B, S] --> Copy_In --> input_ub [1, B, S] --> Exp --> output_ub --> Copy_Out --> output2[N, B, S]
    void TileExpandExp(ComputationalGraphBuilder &G, const int B, const int N, const int S) {
        auto outInnerTemp = G.GetTensor("outInnerTemp");
        auto output2 = G.GetTensor("output2");
        std::vector<int64_t> tileShape{1, B, S};
        for (int i = 0; i < N; i++) {
            std::vector<int64_t> offset = {i, 0, 0};
            int subgraphId = B * N + i;

            std::string input_ub = "input_ub_" + std::to_string(subgraphId);
            G.AddTensor(DataType::DT_FP32, tileShape, input_ub);
            auto tensorUb = G.GetTensor(input_ub);
            tensorUb->SetMemoryTypeBoth(MemoryType::MEM_UB, true);
            G.AddOp(Opcode::OP_COPY_IN, {"outInnerTemp"}, {input_ub}, "Ub_Copy_In_" + std::to_string(subgraphId));
            auto copyInOp = G.GetOp("Ub_Copy_In_" + std::to_string(subgraphId));
            auto attrCopyIn = std::make_shared<CopyOpAttribute>(
                OpImmediate::Specified(offset), MemoryType::MEM_UB,
                OpImmediate::Specified(outInnerTemp->GetShape()), OpImmediate::Specified(outInnerTemp->tensor->GetRawShape()));
            copyInOp->SetOpAttribute(attrCopyIn);
            copyInOp->UpdateSubgraphID(subgraphId);

            std::string outputExpUb = "output_exp_" + std::to_string(subgraphId);
            G.AddTensor(DataType::DT_FP32, tileShape, outputExpUb);
            auto outputGm = G.GetTensor(outputExpUb);
            outputGm->SetMemoryTypeBoth(MemoryType::MEM_UB, true);
            G.AddOp(Opcode::OP_EXP, {input_ub}, {outputExpUb}, "Exp_" + std::to_string(subgraphId));
            auto expOp = G.GetOp("Exp_" + std::to_string(subgraphId));
            expOp->UpdateSubgraphID(subgraphId);

            G.AddOp(Opcode::OP_COPY_OUT, {outputExpUb}, {"output2"}, "Copy_Out_" + std::to_string(subgraphId));
            auto copyOutOp = G.GetOp("Copy_Out_" + std::to_string(subgraphId));
            auto attrCopyOut = std::make_shared<CopyOpAttribute>(
                OpImmediate::Specified(offset), MemoryType::MEM_UB,
                OpImmediate::Specified(output2->GetShape()), OpImmediate::Specified(output2->tensor->GetRawShape()));
            copyOutOp->SetOpAttribute(attrCopyOut);
            copyOutOp->UpdateSubgraphID(subgraphId);
        }
    }

    /*
    [32, 32] --> View --> [16, 16] --> Add --> [16, 16] --> Assemble --> addOutUb[32, 32] -->
                                                    \--> Copy_Out --> out1
    */
    void TileExpandAdd(ComputationalGraphBuilder &G, const int N, const int T) {
        std::vector<int64_t> tileShape{T, T};
        auto a = G.GetTensor("a");
        auto b = G.GetTensor("b");
        auto out1 = G.GetTensor("out1");
        auto addOutUb = G.GetTensor("addOutUb");
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                std::vector<int64_t> offset = {i * T, j * T};
                int idx = i * N + j;

                std::string localA = "a_" + std::to_string(idx);
                G.AddTensor(DataType::DT_FP32, tileShape, localA);
                auto tensorA = G.GetTensor(localA);
                tensorA->SetMemoryTypeBoth(MemoryType::MEM_UB, true);
                G.AddOp(Opcode::OP_COPY_IN, {"a"}, {localA}, "Copy_In_A_" + std::to_string(idx));
                auto copyInA = G.GetOp("Copy_In_A_" + std::to_string(idx));
                auto attrCopyInA = std::make_shared<CopyOpAttribute>(
                    OpImmediate::Specified(offset), MemoryType::MEM_UB,
                    OpImmediate::Specified(a->GetShape()), OpImmediate::Specified(a->tensor->GetRawShape()));
                copyInA->SetOpAttribute(attrCopyInA);
                copyInA->UpdateSubgraphID(0);

                std::string localB = "b_" + std::to_string(idx);
                G.AddTensor(DataType::DT_FP32, tileShape, localB);
                auto tensorB = G.GetTensor(localB);
                tensorB->SetMemoryTypeBoth(MemoryType::MEM_UB, true);
                G.AddOp(Opcode::OP_COPY_IN, {"b"}, {localB}, "Copy_In_B_" + std::to_string(idx));
                auto copyInB = G.GetOp("Copy_In_B_" + std::to_string(idx));
                auto attrCopyInB = std::make_shared<CopyOpAttribute>(
                    OpImmediate::Specified(offset), MemoryType::MEM_UB,
                    OpImmediate::Specified(b->GetShape()), OpImmediate::Specified(b->tensor->GetRawShape()));
                copyInB->SetOpAttribute(attrCopyInB);
                copyInB->UpdateSubgraphID(0);

                std::string localAddOut = "add_out_" + std::to_string(idx);
                G.AddTensor(DataType::DT_FP32, tileShape, localAddOut);
                G.AddOp(Opcode::OP_ADD, {localA, localB}, {localAddOut}, "Add_" + std::to_string(idx));
                auto tensorAddOut = G.GetTensor(localAddOut);
                tensorAddOut->SetMemoryTypeBoth(MemoryType::MEM_UB, true);
                tensorAddOut->tensor = addOutUb->tensor;
                tensorAddOut->UpdateOffset(offset);
                auto addOp = G.GetOp("Add_" + std::to_string(idx));
                addOp->UpdateSubgraphID(0);

                G.AddOp(Opcode::OP_ASSEMBLE, {localAddOut}, {"addOutUb"}, "Assemble_" + std::to_string(idx));
                auto attrAssemble = std::make_shared<AssembleOpAttribute>(MemoryType::MEM_UB, offset);
                auto assembleOp = G.GetOp("Assemble_" + std::to_string(idx));
                assembleOp->SetOpAttribute(attrAssemble);
                assembleOp->UpdateSubgraphID(0);

                G.AddOp(Opcode::OP_COPY_OUT, {localAddOut}, {"out1"}, "Copy_Out_" + std::to_string(idx));
                auto copyOutOp = G.GetOp("Copy_Out_" + std::to_string(idx));
                auto attrCopyOut = std::make_shared<CopyOpAttribute>(
                    OpImmediate::Specified(offset), MemoryType::MEM_UB,
                    OpImmediate::Specified(out1->GetShape()), OpImmediate::Specified(out1->tensor->GetRawShape()));
                copyOutOp->SetOpAttribute(attrCopyOut);
                copyOutOp->UpdateSubgraphID(0);
            }
        }
    }

    int CountAssemble(Function &function) {
        int result = 0;
        for (auto &op : function.Operations()) {
            std::cout << op.GetOpcodeStr() << " " << op.GetOpMagic() << std::endl;
            if (op.GetOpcode() == Opcode::OP_ASSEMBLE) {
                result++;
            }
        }
        return result;
    }
};

TEST_F(PreGraphTest, TestAssemble) {
    PassManager &passManager = PassManager::Instance();
    passManager.RegisterStrategy("PreGraphTestStrategy", {
        {   "RemoveRedundantReshape",   "RemoveRedundantReshape"},
        {      "InferMemoryConflict",      "InferMemoryConflict"},
        {           "ExpandFunction",           "ExpandFunction"},
        {              "DuplicateOp",              "DuplicateOp"},
        {        "MergeViewAssemble",        "MergeViewAssemble"},
        {             "SplitReshape",             "SplitReshape"},
        {           "SplitRawTensor",           "SplitRawTensor"},
        {   "SplitLargeFanoutTensor",   "SplitLargeFanoutTensor"},
        {  "InferDiscontinuousInput",  "InferDiscontinuousInput"},
        {         "AssignMemoryType",         "AssignMemoryType"},
        {        "RemoveRedundantOp",        "RemoveRedundantOp"},
        {                   "SplitK",                   "SplitK"},
        {           "GraphPartition",           "GraphPartition"},
        {             "NBufferMerge",             "NBufferMerge"},
        {     "IntraSubgraphAdapter",     "IntraSubgraphAdapter"},
        {           "GenerateMoveOp",           "GenerateMoveOp"},
        { "CommonOperationEliminate", "CommonOperationEliminate"},
        {       "L1CopyInReuseMerge",       "L1CopyInReuseMerge"},
        {           "PadLocalBuffer",           "PadLocalBuffer"},
        {   "RemoveUnalignedReshape",   "RemoveUnalignedReshape"},
        {           "ReplaceTensor",           "ReplaceTensor"},

    });
    int dim1 = 8;
    int dim2 = 2;
    int dim3 = 64;
    TileShape::Current().SetVecTile(dim1, dim1, dim1, dim1);
    Tensor input(DT_FP32, {1, 384}, "a");
    Tensor res1;
    FUNCTION("TestAssign") {
        TileShape::Current().SetVecTile(1, dim3);
        Tensor res = Exp(input);
        Tensor test = Reshape(res, {2, 1, 1, 192});
        TileShape::Current().SetVecTile(dim2, 1, dim2, dim3);
        res1 = Exp(test);
    }

    std::string jsonFilePath = "./config/pass/json/pre_graph_assemble.json";
    bool dumpJsonFlag = false;
    if (dumpJsonFlag) {
        auto programJson = Program::GetInstance().DumpJson();
        DumpJsonFile(programJson, jsonFilePath);
    }

    // Call the pass
    Function* func = Program::GetInstance().GetFunctionByRawName("TENSOR_TestAssign");
    npu::tile_fwk::PreGraphProcess preGraphPass;
    preGraphPass.PreCheck(*func);
    preGraphPass.RunOnFunction(*func);
    preGraphPass.PostCheck(*func);

    std::set<int> tensorMagicWithColorSet;
    PrintGraphInfoPreGraph(func, tensorMagicWithColorSet);

    // ================== Verify the effect of the Pass ==================
    auto updated_operations = func->Operations();
    int opSize = 30;

    EXPECT_EQ(updated_operations.size(), opSize) << "After the Pass, there should be 30 operations";
    EXPECT_EQ(tensorMagicWithColorSet.size() > 0, true) << "There should be many tensor magic with color";
}

TEST_F(PreGraphTest, TestView) {
config::SetPlatformConfig(KEY_ONLY_HOST_COMPILE, true);

    std::vector<int64_t> shape1{128, 128};
    std::vector<int64_t> shape2{64, 64};
    std::vector<int64_t> shape3{16, 256};

    PassManager &passManager = PassManager::Instance();
    passManager.RegisterStrategy("PreGraphTestStrategy", {
        {   "RemoveRedundantReshape",   "RemoveRedundantReshape"},
        {      "InferMemoryConflict",      "InferMemoryConflict"},
        {           "ExpandFunction",           "ExpandFunction"},
        {              "DuplicateOp",              "DuplicateOp"},
        {        "MergeViewAssemble",        "MergeViewAssemble"},
        {             "SplitReshape",             "SplitReshape"},
        {           "SplitRawTensor",           "SplitRawTensor"},
        {   "SplitLargeFanoutTensor",   "SplitLargeFanoutTensor"},
        {  "InferDiscontinuousInput",  "InferDiscontinuousInput"},
        {         "AssignMemoryType",         "AssignMemoryType"},
        {        "RemoveRedundantOp",        "RemoveRedundantOp"},
        {                   "SplitK",                   "SplitK"},
        {           "GraphPartition",           "GraphPartition"},
        {             "NBufferMerge",             "NBufferMerge"},
        {     "IntraSubgraphAdapter",     "IntraSubgraphAdapter"},
        {           "GenerateMoveOp",           "GenerateMoveOp"},
        { "CommonOperationEliminate", "CommonOperationEliminate"},
        {       "L1CopyInReuseMerge",       "L1CopyInReuseMerge"},
        {           "PadLocalBuffer",           "PadLocalBuffer"},
        {   "RemoveUnalignedReshape",   "RemoveUnalignedReshape"},
        {           "ReplaceTensor",           "ReplaceTensor"},
    });
    ConfigManager::Instance();

    Function* originFunction = nullptr;
    std::vector<int> originOpmagic;

    Tensor in_tensor(DT_FP32, shape1, "in_tensor");
    Tensor in_tensor1(DT_FP32, shape1, "in_tensor1");
    Tensor out_tensor(DT_FP32, shape3, "out_tensor");

    FUNCTION("PreGraphFunction") {
        TileShape::Current().SetVecTile({64, 64});
        auto a = View(in_tensor, shape2, {0,0});
        auto b = View(in_tensor1, shape2, {32,32});
        auto a0 = Add(a, Element(DataType::DT_FP32, 0.0f));
        auto a1 = Reshape(a0, shape3);
        auto b0 = Mul(b, Element(DataType::DT_FP32, 0.1f));
        auto b1 = Reshape(b0, shape3);
        out_tensor = Add(a1, b1);
        originFunction = Program::GetInstance().GetCurrentFunction();
        ASSERT_NE(originFunction, nullptr) << "当前函数指针为空";
        auto operations = originFunction->Operations();
        for (const auto &op : operations) {
            originOpmagic.emplace_back(op.opmagic);
        }
    }

    std::string jsonFilePath = "./config/pass/json/pre_graph_view.json";
    bool dumpJsonFlag = false;
    if (dumpJsonFlag) {
        auto programJson = Program::GetInstance().DumpJson();
        DumpJsonFile(programJson, jsonFilePath);
    }

    // Call the pass
    Function* func = Program::GetInstance().GetFunctionByRawName("TENSOR_PreGraphFunction");
    npu::tile_fwk::PreGraphProcess preGraphPass;
    preGraphPass.PreCheck(*func);
    preGraphPass.RunOnFunction(*func);
    preGraphPass.PostCheck(*func);

    std::set<int> tensorMagicWithColorSet;
    PrintGraphInfoPreGraph(func, tensorMagicWithColorSet);

    // ================== Verify the effect of the Pass ==================
    auto updated_operations = func->Operations();
    int opSize = 24;

    EXPECT_EQ(updated_operations.size(), opSize) << "After the Pass, there should be 24 operations";
    EXPECT_EQ(tensorMagicWithColorSet.size() > 0, true) << "There should be many tensor magic with color";
}

TEST_F(PreGraphTest, TestTransposeDatamove) {
    int B = 3;
    int N = 2;
    int S = 128;
    int NUM_1 = 1;

    std::vector<int64_t> shape0{B, N, S};
    std::vector<int64_t> shape1{N, B, S};
    std::vector<int64_t> tiledShape{NUM_1, NUM_1, S};

    ComputationalGraphBuilder G;
    G.AddTensor(DataType::DT_FP32, shape0, "input");
    G.AddTensor(DataType::DT_FP32, shape1, "output");
    // [16, 16] --> View --> [8, 8] --> Sub --> [8, 8] --> Assemble --> [16, 16]
    TileExpandTransposeDatamove(G, B, N, S);

    auto input = G.GetTensor("input");
    input->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    input->nodetype = NodeType::INCAST;
    auto output = G.GetTensor("output");
    output->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    output->nodetype = NodeType::OUTCAST;

    G.SetInCast({"input"});
    G.SetOutCast({"output"});
    Function *function = G.GetFunction();
    const int SUBGRAPH_NUM = 6;
    function->SetTotalSubGraphCount(SUBGRAPH_NUM);
    // 确认构图完毕
    constexpr int opNumBefore = 18;
    constexpr int assembleNumBefore = 6;
    auto assembleNumCountBefore = CountAssemble(*function);
    EXPECT_EQ(function->Operations().size(), opNumBefore) << opNumBefore << " operations before pass";
    EXPECT_EQ(assembleNumCountBefore, assembleNumBefore) << assembleNumBefore << " OP_ASSEMBLE before pass";
    std::cout << "Build Graph Done." << std::endl;
    /*
    dump graph before Pass
    function->DumpJsonFile(jsonFilePath);
    */
    // 单独执行pass
    npu::tile_fwk::PreGraphProcess preGraph;
    preGraph.PreCheck(*function);
    preGraph.RunOnFunction(*function);
    preGraph.PostCheck(*function);
    std::cout << "Run Pass Done." << std::endl;
    /*
    dump graph after Pass
    function->DumpJsonFile(jsonFilePath);
    */
    constexpr int opNumAfter = 12;
    constexpr int assembleNumafter = 0;
    auto assembleNumCountAfter = CountAssemble(*function);
    EXPECT_EQ(function->Operations().size(), opNumAfter) << opNumAfter << " operations after pass";
    EXPECT_EQ(assembleNumCountAfter, assembleNumafter) << assembleNumafter << " OP_ASSEMBLE before pass";
    EXPECT_EQ(opNumBefore - opNumAfter, assembleNumCountBefore - assembleNumCountAfter) << " only OP_ASSEMBLE should be removed";
}

TEST_F(PreGraphTest, TestTransposeDatamoveExp) {
    int B = 3;
    int N = 2;
    int S = 128;

    std::vector<int64_t> shape0{B, N, S};
    std::vector<int64_t> shape1{N, B, S};

    ComputationalGraphBuilder G;
    G.AddTensor(DataType::DT_FP32, shape0, "input");
    G.AddTensor(DataType::DT_FP32, shape1, "output");
    G.AddTensor(DataType::DT_FP32, shape1, "output2");
    G.AddTensor(DataType::DT_FP32, shape1, "outInnerTemp");
    // [16, 16] --> View --> [8, 8] --> Sub --> [8, 8] --> Assemble --> [16, 16]
    TileExpandTransposeDatamove(G, B, N, S, true);
    TileExpandExp(G, B, N, S);

    auto input = G.GetTensor("input");
    input->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    input->nodetype = NodeType::INCAST;
    auto output = G.GetTensor("output");
    output->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    output->nodetype = NodeType::OUTCAST;
    auto output2 = G.GetTensor("output2");
    output2->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    output2->nodetype = NodeType::OUTCAST;

    auto outInnerTemp = G.GetTensor("outInnerTemp");
    outInnerTemp->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);

    G.SetInCast({"input"});
    G.SetOutCast({"output"});
    Function *function = G.GetFunction();
    const int SUBGRAPH_NUM = 8;
    function->SetTotalSubGraphCount(SUBGRAPH_NUM);
    // 确认构图完毕
    constexpr int opNumBefore = 30;
    constexpr int assembleNumBefore = 12;
    auto assembleNumCountBefore = CountAssemble(*function);
    EXPECT_EQ(function->Operations().size(), opNumBefore) << opNumBefore << " operations before pass";
    EXPECT_EQ(assembleNumCountBefore, assembleNumBefore) << assembleNumBefore << " OP_ASSEMBLE before pass";
    std::cout << "Build Graph Done." << std::endl;
    /*
    dump graph before Pass
    function->DumpJsonFile(jsonFilePath);
    */
    // 单独执行pass
    npu::tile_fwk::PreGraphProcess preGraph;
    preGraph.PreCheck(*function);
    preGraph.RunOnFunction(*function);
    preGraph.PostCheck(*function);
    std::cout << "Run Pass Done." << std::endl;
    /*
    dump graph after Pass
    function->DumpJsonFile(jsonFilePath);
    */
    constexpr int opNumAfter = 30;
    constexpr int assembleNumafter = 12;
    auto assembleNumCountAfter = CountAssemble(*function);
    EXPECT_EQ(function->Operations().size(), opNumAfter) << opNumAfter << " operations after pass";
    EXPECT_EQ(assembleNumCountAfter, assembleNumafter) << assembleNumafter << " OP_ASSEMBLE before pass";
    EXPECT_EQ(opNumBefore - opNumAfter, assembleNumCountBefore - assembleNumCountAfter) << " only OP_ASSEMBLE should be removed";
}


TEST_F(PreGraphTest, TestAddExp) {
    int N = 2;
    int T = 16;
    std::vector<int64_t> shape0{N * T, N * T};
    // std::vector<int64_t> shape1{T, T};
    ComputationalGraphBuilder G;
    G.AddTensor(DataType::DT_FP32, shape0, "a");
    G.AddTensor(DataType::DT_FP32, shape0, "b");
    G.AddTensor(DataType::DT_FP32, shape0, "out1");
    G.AddTensor(DataType::DT_FP32, shape0, "out2");
    G.AddTensor(DataType::DT_FP32, shape0, "addOutUb");
    G.AddTensor(DataType::DT_FP32, shape0, "expOutUb");

    auto a = G.GetTensor("a");
    a->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    auto b = G.GetTensor("b");
    b->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    auto out1 = G.GetTensor("out1");
    out1->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    auto out2 = G.GetTensor("out2");
    out2->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    auto addOutUb = G.GetTensor("addOutUb");
    addOutUb->SetMemoryTypeBoth(MemoryType::MEM_UB, true);
    auto expOutUb = G.GetTensor("expOutUb");
    expOutUb->SetMemoryTypeBoth(MemoryType::MEM_UB, true);
    /*
    [32, 32] --> View --> [16, 16] --> Add --> [16, 16] --> Assemble --> addOutUb[32, 32] --> Exp --> expOutUb[32, 32] --> Copy_Out --> out2
                                                    \--> Copy_Out --> out1
    */
    TileExpandAdd(G, N, T);

    G.AddOp(Opcode::OP_EXP, {"addOutUb"}, {"expOutUb"}, "Exp_Op");
    auto expOp = G.GetOp("Exp_Op");
    expOp->UpdateSubgraphID(0);

    G.AddOp(Opcode::OP_COPY_OUT, {"expOutUb"}, {"out2"}, "Copy_Out_Exp");
    auto copyOutOp = G.GetOp("Copy_Out_Exp");
    auto attrCopyOut = std::make_shared<CopyOpAttribute>(
        OpImmediate::Specified(std::vector<int64_t> {0, 0}), MemoryType::MEM_UB,
        OpImmediate::Specified(out2->GetShape()), OpImmediate::Specified(out2->tensor->GetRawShape()));
    copyOutOp->SetOpAttribute(attrCopyOut);
    copyOutOp->UpdateSubgraphID(0);

    G.SetInCast({"a", "b"});
    G.SetOutCast({"out1", "out2"});
    Function *function = G.GetFunction();
    const int SUBGRAPH_NUM = 1;
    function->SetTotalSubGraphCount(SUBGRAPH_NUM);
    // 确认构图完毕
    constexpr int opNumBefore = 22;
    EXPECT_EQ(function->Operations().size(), opNumBefore) << opNumBefore << " operations before pass";
    std::cout << "Build Graph Done." << std::endl;
    /*
    dump graph before Pass
    function->DumpJsonFile(jsonFilePath);
    */
    // 单独执行pass
    npu::tile_fwk::PreGraphProcess preGraph;
    preGraph.PreCheck(*function);
    preGraph.RunOnFunction(*function);
    preGraph.PostCheck(*function);
    std::cout << "Run Pass Done." << std::endl;
    /*
    dump graph after Pass
    function->DumpJsonFile(jsonFilePath);
    */
    EXPECT_EQ(function->Operations().size(), opNumBefore) << opNumBefore << " operations after pass";
}

TEST_F(PreGraphTest, PreGraphReShapeOnOcast) {
    ComputationalGraphBuilder G;
    // add tensor
    DataType inputAstDtype = DataType::DT_FP16;
    DataType outputAstDtype = DataType::DT_FP16;
    G.AddTensor(inputAstDtype, {64, 8, 16}, "vec_in_rel");
    auto vec_in_rel = G.GetTensor("vec_in_rel");
    vec_in_rel->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    G.AddTensor(inputAstDtype, {64, 8, 16}, "vec_in");
    auto vec_in = G.GetTensor("vec_in");
    vec_in->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    G.AddTensor(outputAstDtype, {64, 128}, "vec_out");
    auto vec_out = G.GetTensor("vec_out");
    vec_out->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    // add op
    G.AddOp(Opcode::OP_VIEW, {"vec_in_rel"}, {"vec_in"}, "VIEW");
    G.AddOp(Opcode::OP_RESHAPE, {"vec_in"}, {"vec_out"}, "RESHAPE");
    // set incast and outcast
    G.SetInCast({"vec_in_rel"});
    G.SetOutCast({"vec_out"});
    // check before pass
    auto inRawMagicBefore = vec_in->GetRawMagic();
    auto outRawMagicBefore = vec_out->GetRawMagic();
    EXPECT_NE(inRawMagicBefore, outRawMagicBefore);
    // run pass
    Function *function = G.GetFunction();
    EXPECT_NE(function, nullptr);
    PreGraphProcess passLocal;
    passLocal.Run(*function, "", "", 0);
    // check after pass
    auto inRawMagicAfter = vec_in->GetRawMagic();
    auto outRawMagicAfter = vec_out->GetRawMagic();
    EXPECT_EQ(inRawMagicAfter, outRawMagicAfter);
    EXPECT_EQ(outRawMagicBefore, outRawMagicAfter);
}

TEST_F(PreGraphTest, TestFixPipeReconnectGraph) {
    Program::GetInstance().Reset();
    config::Reset();
    auto funcPtr = std::make_shared<Function>(Program::GetInstance(), "TestFixPipeReconnectGraph", "TestFixPipeReconnectGraph", nullptr);

    // Build graph
    std::vector<int64_t> shape = {NUM16, NUM16};
    auto tensor0  = std::make_shared<LogicalTensor>(*funcPtr, DT_FP32, shape);
    tensor0->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR);
    auto tensor1 = std::make_shared<LogicalTensor>(*funcPtr, DT_FP32, shape);
    tensor1->SetMemoryTypeBoth(MemoryType::MEM_L1);
    auto tensor2 = std::make_shared<LogicalTensor>(*funcPtr, DT_FP32, shape);
    tensor2->SetMemoryTypeBoth(MemoryType::MEM_FIX_QUANT_PRE);
    auto tensor3 = std::make_shared<LogicalTensor>(*funcPtr, DT_FP32, shape);
    tensor3->SetMemoryTypeBoth(MemoryType::MEM_L0C);
    auto tensor4 = std::make_shared<LogicalTensor>(*funcPtr, DT_FP32, shape);
    tensor4->SetMemoryTypeBoth(MemoryType::MEM_L0C);
    auto tensor5 = std::make_shared<LogicalTensor>(*funcPtr, DT_FP32, shape);
    tensor5->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR);
    auto &copyin = funcPtr->AddRawOperation(Opcode::OP_COPY_IN, {tensor0}, {tensor1});
    (void)copyin;
    auto &l1CopyFB = funcPtr->AddRawOperation(Opcode::OP_L1_TO_FIX_QUANT_PRE, {tensor1}, {tensor2});
    (void)l1CopyFB;
    auto &aMulB = funcPtr->AddRawOperation(Opcode::OP_A_MUL_B, {tensor2}, {tensor3});
    aMulB.SetAttribute(A_MUL_B_SCALE_ATTR, Element(DataType::DT_UINT64, NUM10));
    aMulB.SetAttribute(A_MUL_B_RELU_ATTR, 1);
    auto &aMulAccB = funcPtr->AddRawOperation(Opcode::OP_A_MULACC_B, {tensor3}, {tensor4});
    (void)aMulAccB;
    auto &copyout = funcPtr->AddRawOperation(Opcode::OP_COPY_OUT, {tensor4}, {tensor5});

    // Test Reconnect Graph
    CubeProcess cubeProcess;
    std::vector<Operation *> l0CCopyOuts{};
    cubeProcess.GetL0CCopyOuts(aMulB, l0CCopyOuts);
    EXPECT_EQ(l0CCopyOuts[0], &copyout);
    cubeProcess.ReconnectGraph(aMulB, l0CCopyOuts);
    auto tensor2Consumer = tensor2->GetConsumers().begin();
    EXPECT_EQ(*tensor2Consumer, &copyout);
    auto scaleValue = (copyout.HasAttr(A_MUL_B_SCALE_ATTR)) ? copyout.GetElementAttribute(A_MUL_B_SCALE_ATTR) : Element(DataType::DT_UINT64, 0);
    auto reluType = (copyout.HasAttr(A_MUL_B_RELU_ATTR)) ? copyout.GetIntAttribute(A_MUL_B_RELU_ATTR) : 0;
    EXPECT_EQ(scaleValue, Element(DataType::DT_UINT64, NUM10));
    EXPECT_EQ(reluType, 1);
}
} // namespace tile_fwk
} // namespace npu
#undef private