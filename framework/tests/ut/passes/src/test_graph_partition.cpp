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
 * \file test_graph_partition.cpp
 * \brief Unit test for GraphPartition pass.
 */

#include <fstream>
#include <vector>
#include <string>
#include "gtest/gtest.h"
#include "tilefwk/data_type.h"
#include "tilefwk/tilefwk_op.h"
#include "interface/function/function.h"
#include "passes/tile_graph_pass/graph_partition/iso_partitioner.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "passes/pass_mgr/pass_manager.h"
#include "interface/configs/config_manager.h"
#include "computational_graph_builder.h"

namespace npu {
namespace tile_fwk {

class GraphPartitionTest : public testing::Test {
public:
    static void SetUpTestCase() {}
    static void TearDownTestCase() {}
    void SetUp() override {
        Program::GetInstance().Reset();
        config::Reset();
        config::SetHostOption(COMPILE_STAGE, HOST_COMPILE_END);
        config::SetHostConfig(KEY_STRATEGY, "GraphPartitionTestStrategy");
    }
    void TearDown() override {}
};

void GetPairSumGraph(ComputationalGraphBuilder &G) {
    const int brNum = 4;
    std::vector<int64_t> tileShape{16,16};
    for (int i = 0; i < brNum; i++) {
        std::string br = std::to_string(i);
        std::vector<std::string> tensorNames{"t1" + br, "t2" + br, "t3" + br, "t4" + br};
        std::vector<Opcode> opCodes{Opcode::OP_COPY_IN, Opcode::OP_MULS, Opcode::OP_ADDS};
        std::vector<std::vector<std::string>> ioperands{{"t1" + br}, {"t2" + br}, {"t3" + br}};
        std::vector<std::vector<std::string>> ooperands{{"t2" + br}, {"t3" + br}, {"t4" + br}};
        std::vector<std::string> opNames{"COPY_IN" + br, "MULS" + br, "ADDS" + br};
        EXPECT_EQ(G.AddTensors(DataType::DT_FP32, tileShape, tensorNames), true);
        EXPECT_EQ(G.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    }
    std::vector<std::string> sumTensorNames{"s1", "s2", "s3"};
    std::vector<Opcode> sumOpCodes{Opcode::OP_PAIRSUM, Opcode::OP_PAIRSUM, Opcode::OP_PAIRSUM};
    std::vector<std::vector<std::string>> sumIoperands{{"t40", "t41"}, {"t42", "s1"}, {"t43", "s2"}};
    std::vector<std::vector<std::string>> sumOoperands{{"s1"}, {"s2"}, {"s3"}};
    std::vector<std::string> sumOpNames{"SUM1", "SUM2", "SUM3"};
    EXPECT_EQ(G.AddTensors(DataType::DT_FP32, tileShape, sumTensorNames), true);
    EXPECT_EQ(G.AddOps(sumOpCodes, sumIoperands, sumOoperands, sumOpNames, true), true);
    EXPECT_EQ(G.SetInCast({"t10", "t11", "t12", "t13"}), true);
    EXPECT_EQ(G.SetOutCast({"s3"}), true);
}

TEST_F(GraphPartitionTest, TestBuildOpGraph) {
    ComputationalGraphBuilder G;
    GetPairSumGraph(G);
    Function *function = G.GetFunction();
    const int cycleUB = 10;
    const int parallelTH = 10;
    const int cycleLB = 10;
    const int useNodeHash = false;
    IsoPartitioner partitioner;
    EXPECT_EQ(partitioner.SetParameter(cycleUB, parallelTH, cycleLB, useNodeHash), SUCCESS);
    EXPECT_EQ(partitioner.PartitionGraph(*function), SUCCESS);
    EXPECT_EQ(partitioner.operationInfo_->opList_.size(), function->Operations().size());
    EXPECT_EQ(partitioner.operationInfo_->magic2Idx_.size(), function->Operations().size());
    EXPECT_EQ(partitioner.operationInfo_->inGraph_.size(), function->Operations().size());
    EXPECT_EQ(partitioner.operationInfo_->opHashList_.size(), function->Operations().size());
    EXPECT_EQ(partitioner.operationInfo_->opCoreType_.size(), function->Operations().size());
    const std::vector<std::pair<std::string, int>> inLinkNum{{"COPY_IN0", 0}, {"MULS0", 1}, {"ADDS0", 1},
                                                             {"SUM1", 2}, {"SUM2", 2}, {"SUM3", 2}};
    for (auto &pr: inLinkNum) {
        EXPECT_NE(G.GetOp(pr.first), nullptr);
        int opMagic = G.GetOp(pr.first)->GetOpMagic();
        int opIdx = partitioner.operationInfo_->magic2Idx_[opMagic];
        EXPECT_EQ(partitioner.operationInfo_->inGraph_[opIdx].size(), pr.second);
    }
    const std::vector<std::pair<std::string, int>> outLinkNum{{"COPY_IN0", 1}, {"MULS0", 1}, {"ADDS0", 1},
                                                             {"SUM1", 1}, {"SUM2", 1}, {"SUM3", 0}};
    for (auto &pr: outLinkNum) {
        EXPECT_NE(G.GetOp(pr.first), nullptr);
        int opMagic = G.GetOp(pr.first)->GetOpMagic();
        int opIdx = partitioner.operationInfo_->magic2Idx_[opMagic];
        EXPECT_EQ(partitioner.operationInfo_->outGraph_[opIdx].size(), pr.second);
    }
    int copyIdx0 = partitioner.operationInfo_->magic2Idx_[G.GetOp("COPY_IN0")->GetOpMagic()];
    int copyIdx1 = partitioner.operationInfo_->magic2Idx_[G.GetOp("COPY_IN1")->GetOpMagic()];
    int sumIdx1 = partitioner.operationInfo_->magic2Idx_[G.GetOp("SUM1")->GetOpMagic()];
    int sumIdx2 = partitioner.operationInfo_->magic2Idx_[G.GetOp("SUM2")->GetOpMagic()];
    EXPECT_EQ(partitioner.operationInfo_->opHashList_[copyIdx0], partitioner.operationInfo_->opHashList_[copyIdx1]);
    EXPECT_EQ(partitioner.operationInfo_->opHashList_[sumIdx1], partitioner.operationInfo_->opHashList_[sumIdx2]);
    EXPECT_NE(partitioner.operationInfo_->opHashList_[copyIdx0], partitioner.operationInfo_->opHashList_[sumIdx2]);
    std::unordered_set<uint64_t> copyInHash;
    const int brNum = 4;
    for (int i = 0; i < brNum; i++) {
        EXPECT_NE(G.GetOp("COPY_IN" + std::to_string(i)), nullptr);
        int opMagic = G.GetOp("COPY_IN" + std::to_string(i))->GetOpMagic();
        int opIdx = partitioner.operationInfo_->magic2Idx_[opMagic];
        int nodeIdx = partitioner.superNodeInfo_->op2Node_[opIdx];
        copyInHash.insert(partitioner.superNodeInfo_->nodeHashList_[nodeIdx]);
    }
    const int copyInHashNum = 3;
    EXPECT_EQ(copyInHash.size(), copyInHashNum);
}

void GetReshapeGraph(ComputationalGraphBuilder &G) {
    const int brNum = 4;
    std::vector<int64_t> tileShape{16,16};
    EXPECT_EQ(G.AddTensors(DataType::DT_FP32, tileShape, {"rin", "rout"}), true);
    for (int i = 0; i < brNum; i++) {
        std::string br = std::to_string(i);
        std::vector<std::string> tensorNames{"t1" + br, "t2" + br, "t3" + br};
        std::vector<Opcode> opCodes{Opcode::OP_COPY_IN, Opcode::OP_RESHAPE, Opcode::OP_ASSEMBLE};
        std::vector<std::vector<std::string>> ioperands{{"t1" + br}, {"t2" + br}, {"t3" + br}};
        std::vector<std::vector<std::string>> ooperands{{"t2" + br}, {"t3" + br}, {"rin"}};
        std::vector<std::string> opNames{"COPY_IN" + br, "RESHAPE_IN" + br, "ASSEMBLE" + br};
        EXPECT_EQ(G.AddTensors(DataType::DT_FP32, tileShape, tensorNames), true);
        EXPECT_EQ(G.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    }
    EXPECT_EQ(G.AddOp(Opcode::OP_RESHAPE, {"rin"}, {"rout"}, "MULTI_RESHAPE", true), true);
    for (int i = 0; i < brNum; i++) {
        std::string br = std::to_string(i);
        std::vector<std::string> tensorNames{"b1" + br, "b2" + br, "b3" + br};
        std::vector<Opcode> opCodes{Opcode::OP_VIEW, Opcode::OP_RESHAPE, Opcode::OP_COPY_OUT};
        std::vector<std::vector<std::string>> ioperands{{"rout"}, {"b1" + br}, {"b2" + br}};
        std::vector<std::vector<std::string>> ooperands{{"b1" + br}, {"b2" + br}, {"b3" + br}};
        std::vector<std::string> opNames{"VIEW" + br, "RESHAPE_OUT" + br, "COPY_OUT" + br};
        EXPECT_EQ(G.AddTensors(DataType::DT_FP32, tileShape, tensorNames), true);
        EXPECT_EQ(G.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    }
    EXPECT_EQ(G.SetInCast({"t10", "t11", "t12", "t13"}), true);
    EXPECT_EQ(G.SetOutCast({"b30", "b31", "b32", "b33"}), true);
}

TEST_F(GraphPartitionTest, TestSuperNode) {
    ComputationalGraphBuilder G;
    GetReshapeGraph(G);
    Function *function = G.GetFunction();
    const int cycleUB = 0;
    const int parallelTH = 10;
    const int cycleLB = 10;
    const int useNodeHash = false;
    IsoPartitioner partitioner;
    EXPECT_EQ(partitioner.SetParameter(cycleUB, parallelTH, cycleLB, useNodeHash), SUCCESS);
    EXPECT_EQ(partitioner.PartitionGraph(*function), SUCCESS);
    std::unordered_set<int> frontReshapeNode;
    const int brNum = 4;
    for (int i = 0; i < brNum; i++) {
        EXPECT_NE(G.GetOp("RESHAPE_IN" + std::to_string(i)), nullptr);
        int opMagic = G.GetOp("RESHAPE_IN" + std::to_string(i))->GetOpMagic();
        int opIdx = partitioner.operationInfo_->magic2Idx_[opMagic];
        frontReshapeNode.insert(partitioner.superNodeInfo_->op2Node_[opIdx]);
    }
    EXPECT_EQ(frontReshapeNode.size(), brNum);
    std::unordered_set<int> backReshapeNode;
    for (int i = 0; i < brNum; i++) {
        EXPECT_NE(G.GetOp("RESHAPE_OUT" + std::to_string(i)), nullptr);
        int opMagic = G.GetOp("RESHAPE_OUT" + std::to_string(i))->GetOpMagic();
        int opIdx = partitioner.operationInfo_->magic2Idx_[opMagic];
        backReshapeNode.insert(partitioner.superNodeInfo_->op2Node_[opIdx]);
    }
    EXPECT_EQ(backReshapeNode.size(), brNum);
    const int subGraphNum = 9;
    EXPECT_EQ(function->GetTotalSubGraphCount(), subGraphNum);
}

TEST_F(GraphPartitionTest, TestReduceNodeHash) {
    ComputationalGraphBuilder G;
    GetPairSumGraph(G);
    Function *function = G.GetFunction();
    const int cycleUB = 0;
    const int parallelTH = 10;
    const int cycleLB = 10;
    const int useNodeHash = true;
    IsoPartitioner partitioner;
    EXPECT_EQ(partitioner.SetParameter(cycleUB, parallelTH, cycleLB, useNodeHash), SUCCESS);
    EXPECT_EQ(partitioner.PartitionGraph(*function), SUCCESS);
    std::unordered_set<uint64_t> copyInHash;
    const int brNum = 4;
    for (int i = 0; i < brNum; i++) {
        EXPECT_NE(G.GetOp("COPY_IN" + std::to_string(i)), nullptr);
        int opMagic = G.GetOp("COPY_IN" + std::to_string(i))->GetOpMagic();
        int opIdx = partitioner.operationInfo_->magic2Idx_[opMagic];
        int nodeIdx = partitioner.superNodeInfo_->op2Node_[opIdx];
        copyInHash.insert(partitioner.superNodeInfo_->nodeHashList_[nodeIdx]);
    }
    EXPECT_EQ(copyInHash.size(), 1);
}

void GetCrossGraph(ComputationalGraphBuilder &G) {
    const int brNum = 4;
    std::vector<int64_t> tileShape{16,16};
    for (int i = 0; i < brNum; i++) {
        std::string br = std::to_string(i);
        std::vector<std::string> tensorNames{"t1" + br, "t2" + br, "t3" + br, "t4" + br};
        std::vector<Opcode> opCodes{Opcode::OP_COPY_IN, Opcode::OP_RESHAPE, Opcode::OP_ABS};
        std::vector<std::vector<std::string>> ioperands{{"t1" + br}, {"t2" + br}, {"t3" + br}};
        std::vector<std::vector<std::string>> ooperands{{"t2" + br}, {"t3" + br}, {"t4" + br}};
        std::vector<std::string> opNames{"COPY_IN" + br, "RESHAPE_IN" + br, "ABS" + br};
        EXPECT_EQ(G.AddTensors(DataType::DT_FP32, tileShape, tensorNames), true);
        EXPECT_EQ(G.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    }
    for (int i = 0; i < brNum; i++) {
        std::string br = std::to_string(i);
        std::string sbr = std::to_string((i + 1) % brNum);
        std::vector<std::string> tensorNames{"b1" + br, "b2" + br, "b3" + br};
        std::vector<Opcode> opCodes{Opcode::OP_MUL, Opcode::OP_RESHAPE, Opcode::OP_COPY_OUT};
        std::vector<std::vector<std::string>> ioperands{{"t4" + br, "t4" + sbr}, {"b1" + br}, {"b2" + br}};
        std::vector<std::vector<std::string>> ooperands{{"b1" + br}, {"b2" + br}, {"b3" + br}};
        std::vector<std::string> opNames{"MUL" + br, "RESHAPE_OUT" + br, "COPY_OUT" + br};
        EXPECT_EQ(G.AddTensors(DataType::DT_FP32, tileShape, tensorNames), true);
        EXPECT_EQ(G.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    }
    EXPECT_EQ(G.SetInCast({"t10", "t11", "t12", "t13"}), true);
    EXPECT_EQ(G.SetOutCast({"b30", "b31", "b32", "b33"}), true);
}

TEST_F(GraphPartitionTest, TestBuildIsomorphismGraph) {
    ComputationalGraphBuilder G;
    GetCrossGraph(G);
    Function *function = G.GetFunction();
    const int cycleUB = 100000;
    const int parallelTH = 10;
    const int cycleLB = 100000;
    const int useNodeHash = false;
    IsoPartitioner partitioner;
    EXPECT_EQ(partitioner.SetParameter(cycleUB, parallelTH, cycleLB, useNodeHash), SUCCESS);
    EXPECT_EQ(partitioner.PartitionGraph(*function), SUCCESS);
    const int subGraphNum = 8;
    EXPECT_EQ(function->GetTotalSubGraphCount(), subGraphNum);
}

TEST_F(GraphPartitionTest, TestEmptyGraph) {
    ComputationalGraphBuilder G;
    Function *function = G.GetFunction();
    EXPECT_EQ(function->Operations().size(), 0);
    GraphPartition gpp;
    EXPECT_EQ(gpp.RunOnFunction(*function), SUCCESS);
    EXPECT_EQ(function->GetTotalSubGraphCount(), 0);
}

void GetCubeVectorGraph(ComputationalGraphBuilder &G, int brNum) {
    std::vector<int64_t> tileShape{16,16};
    std::vector<std::string> inTensorNames;
    for (int i = 0; i < brNum; i++) {
        std::string br = std::to_string(i);
        inTensorNames.push_back("ta1" + br);
        std::vector<std::string> tensorNames{"ta1" + br, "ta2" + br, "ta3" + br, "ta4" + br, "ta5" + br, "ta6" + br};
        std::vector<MemoryType> tensorMemType{MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_UB, MemoryType::MEM_UB,
                                              MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_L1, MemoryType::MEM_L0A};
        std::vector<Opcode> opCodes{Opcode::OP_COPY_IN, Opcode::OP_CAST, Opcode::OP_COPY_OUT,
                                    Opcode::OP_COPY_IN, Opcode::OP_L1_TO_L0A};
        std::vector<std::vector<std::string>> ioperands{{"ta1" + br}, {"ta2" + br}, {"ta3" + br},
                                                        {"ta4" + br}, {"ta5" + br}};
        std::vector<std::vector<std::string>> ooperands{{"ta2" + br}, {"ta3" + br}, {"ta4" + br},
                                                        {"ta5" + br}, {"ta6" + br}};
        std::vector<std::string> opNames{"IN_A" + br, "CAST_A" + br, "OUT_A" + br, "IN_L1_A" + br, "L1_TO_L0A" + br};
        EXPECT_EQ(G.AddTensors(DataType::DT_FP32, tileShape, tensorMemType, tensorNames), true);
        EXPECT_EQ(G.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    }
    for (int i = 0; i < brNum; i++) {
        std::string br = std::to_string(i);
        inTensorNames.push_back("tb1" + br);
        std::vector<std::string> tensorNames{"tb1" + br, "tb2" + br, "tb3" + br, "tb4" + br, "tb5" + br, "tb6" + br};
        std::vector<MemoryType> tensorMemType{MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_UB, MemoryType::MEM_UB,
                                              MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_L1, MemoryType::MEM_L0B};
        std::vector<Opcode> opCodes{Opcode::OP_COPY_IN, Opcode::OP_CAST, Opcode::OP_COPY_OUT,
                                    Opcode::OP_COPY_IN, Opcode::OP_L1_TO_L0B};
        std::vector<std::vector<std::string>> ioperands{{"tb1" + br}, {"tb2" + br}, {"tb3" + br},
                                                        {"tb4" + br}, {"tb5" + br}};
        std::vector<std::vector<std::string>> ooperands{{"tb2" + br}, {"tb3" + br}, {"tb4" + br},
                                                        {"tb5" + br}, {"tb6" + br}};
        std::vector<std::string> opNames{"IN_B" + br, "CAST_B" + br, "OUT_B" + br, "IN_L1_B" + br, "L1_TO_L0B" + br};
        EXPECT_EQ(G.AddTensors(DataType::DT_FP32, tileShape, tensorMemType, tensorNames), true);
        EXPECT_EQ(G.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    }
    for (int i = 0; i < brNum; i++) {
        EXPECT_EQ(G.AddTensor(DataType::DT_FP32, tileShape, MemoryType::MEM_L0C, "tc" + std::to_string(i)), true);
    }
    EXPECT_EQ(G.AddTensor(DataType::DT_FP32, tileShape, MemoryType::MEM_DEVICE_DDR, "tout"), true);
    EXPECT_EQ(G.AddOp(Opcode::OP_A_MUL_B, {"ta60", "tb60"}, {"tc0"}, "MUL1", true), true);
    for (int i = 1; i < brNum; i++) {
        std::string br = std::to_string(i);
        std::string lbr = std::to_string(i-1);
        EXPECT_EQ(G.AddOp(Opcode::OP_A_MULACC_B, {"ta6" + br, "tb6" + br, "tc" + lbr}, {"tc" + br}, "MC" + br, true),
                  true);
    }
    EXPECT_EQ(G.AddOp(Opcode::OP_COPY_OUT, {"tc" + std::to_string(brNum - 1)}, {"tout"}, "COPY_OUT_C", true), true);
    EXPECT_EQ(G.SetInCast(inTensorNames), true);
    EXPECT_EQ(G.SetOutCast({"tout"}), true);
}

TEST_F(GraphPartitionTest, TestCVGraph) {
    ComputationalGraphBuilder G;
    const int brNum = 4;
    GetCubeVectorGraph(G, brNum);
    Function *function = G.GetFunction();
    const int cycleUB = 100000;
    const int parallelTH = 1;
    const int cycleLB = 100000;
    const int useNodeHash = false;
    IsoPartitioner partitioner;
    EXPECT_EQ(partitioner.SetParameter(cycleUB, parallelTH, cycleLB, useNodeHash), SUCCESS);
    EXPECT_EQ(partitioner.PartitionGraph(*function), SUCCESS);
    std::unordered_set<std::string> cubeOp{"MUL1", "MC1", "MC2", "MC3", "COPY_OUT_C"};
    for (int i = 0; i < brNum; i++) {
        std::string br = std::to_string(i);
        cubeOp.insert("IN_L1_A" + br);
        cubeOp.insert("IN_L1_B" + br);
        cubeOp.insert("L1_TO_L0A" + br);
        cubeOp.insert("L1_TO_L0B" + br);
    }
    const int subGraphNum = 9;
    EXPECT_EQ(function->GetTotalSubGraphCount(), subGraphNum);
    std::unordered_set<int> subgraphIDs;
    for (auto &opPair : G.operations_) {
        Operation *op = opPair.second;
        EXPECT_NE(op, nullptr);
        if (cubeOp.count(opPair.first) > 0) {
            EXPECT_EQ(op->HasAttr(OpAttributeKey::isCube) && op->GetBoolAttribute(OpAttributeKey::isCube), true);
        } else {
            EXPECT_EQ(op->HasAttr(OpAttributeKey::isCube) && !op->GetBoolAttribute(OpAttributeKey::isCube), true);
        }
        EXPECT_EQ(op->GetSubgraphID() >= 0 && op->GetSubgraphID() < subGraphNum, true);
        subgraphIDs.insert(op->GetSubgraphID());
    }
    EXPECT_EQ(subgraphIDs.size(), subGraphNum);
}

void GetMergeableGraph(ComputationalGraphBuilder &G, int brNum) {
    std::vector<int64_t> tileShape{16,16};
    std::vector<std::string> inCast;
    std::vector<std::string> outCast;
    for (int i = 0; i < brNum; i++) {
        std::string br = std::to_string(i);
        std::vector<std::string> tensorNames{"t1" + br, "t2" + br, "t3" + br, "t4" + br,
                                             "b1" + br, "b2" + br, "b3" + br, "b4" + br};
        std::vector<Opcode> opCodes{Opcode::OP_COPY_IN, Opcode::OP_RESHAPE, Opcode::OP_ABS,
                                    Opcode::OP_COPY_IN, Opcode::OP_RESHAPE, Opcode::OP_ABS};
        std::vector<std::vector<std::string>> ioperands{{"t1" + br}, {"t2" + br}, {"t3" + br},
                                                        {"b1" + br}, {"b2" + br}, {"b3" + br}};
        std::vector<std::vector<std::string>> ooperands{{"t2" + br}, {"t3" + br}, {"t4" + br},
                                                        {"b2" + br}, {"b3" + br}, {"b4" + br}};
        std::vector<std::string> opNames{"COPY_INt" + br, "RESHAPE_INt" + br, "ABSt" + br,
                                         "COPY_INb" + br, "RESHAPE_INb" + br, "ABSb" + br};
        EXPECT_EQ(G.AddTensors(DataType::DT_FP32, tileShape, tensorNames), true);
        EXPECT_EQ(G.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
        inCast.push_back("t1" + br);
        inCast.push_back("b1" + br);
    }
    for (int i = 0; i < brNum; i++) {
        std::string br = std::to_string(i);
        std::vector<std::string> tensorNames{"m1" + br, "m2" + br, "m3" + br};
        std::vector<Opcode> opCodes{Opcode::OP_MUL, Opcode::OP_RESHAPE, Opcode::OP_COPY_OUT};
        std::vector<std::vector<std::string>> ioperands{{"t4" + br, "b4" + br}, {"m1" + br}, {"m2" + br}};
        std::vector<std::vector<std::string>> ooperands{{"m1" + br}, {"m2" + br}, {"m3" + br}};
        std::vector<std::string> opNames{"MUL" + br, "RESHAPE_OUT" + br, "COPY_OUT" + br};
        EXPECT_EQ(G.AddTensors(DataType::DT_FP32, tileShape, tensorNames), true);
        EXPECT_EQ(G.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
        outCast.push_back("m3" + br);
    }
    EXPECT_EQ(G.SetInCast(inCast), true);
    EXPECT_EQ(G.SetOutCast(outCast), true);
}

TEST_F(GraphPartitionTest, TestCycleUpperBound) {
    ComputationalGraphBuilder G;
    const int brNum = 4;
    GetMergeableGraph(G, brNum);
    Function *function = G.GetFunction();
    const int cycleUB = 0;
    const int parallelTH = 1;
    const int cycleLB = 100000;
    const int useNodeHash = false;
    IsoPartitioner partitioner;
    EXPECT_EQ(partitioner.SetParameter(cycleUB, parallelTH, cycleLB, useNodeHash), SUCCESS);
    EXPECT_EQ(partitioner.PartitionGraph(*function), SUCCESS);
    const int subGraphNum = 6 * brNum;
    EXPECT_EQ(function->GetTotalSubGraphCount(), subGraphNum);
}

TEST_F(GraphPartitionTest, TestParallelThreshold) {
    ComputationalGraphBuilder G;
    const int brNum = 4;
    GetMergeableGraph(G, brNum);
    Function *function = G.GetFunction();
    const int cycleUB = 100000;
    const int parallelTH = brNum*2;
    const int cycleLB = 0;
    const int useNodeHash = false;
    IsoPartitioner partitioner;
    EXPECT_EQ(partitioner.SetParameter(cycleUB, parallelTH, cycleLB, useNodeHash), SUCCESS);
    EXPECT_EQ(partitioner.PartitionGraph(*function), SUCCESS);
    const int subGraphNum = 3 * brNum;
    EXPECT_EQ(function->GetTotalSubGraphCount(), subGraphNum);
}

TEST_F(GraphPartitionTest, TestSmallGraphBound) {
    ComputationalGraphBuilder G;
    const int brNum = 4;
    GetMergeableGraph(G, brNum);
    Function *function = G.GetFunction();
    const int cycleUB = 100000;
    const int parallelTH = brNum*2;
    const int cycleLB = 100000;
    const int useNodeHash = false;
    IsoPartitioner partitioner;
    EXPECT_EQ(partitioner.SetParameter(cycleUB, parallelTH, cycleLB, useNodeHash), SUCCESS);
    EXPECT_EQ(partitioner.PartitionGraph(*function), SUCCESS);
    const int subGraphNum = brNum;
    EXPECT_EQ(function->GetTotalSubGraphCount(), subGraphNum);
}

TEST_F(GraphPartitionTest, TestLargeSuperNode) {
    ComputationalGraphBuilder G;
    const int brNum = 5000;
    GetCubeVectorGraph(G, brNum);
    Function *function = G.GetFunction();
    const int cycleUB = 100000;
    const int parallelTH = brNum*2;
    const int cycleLB = 100000;
    const int useNodeHash = false;
    IsoPartitioner partitioner;
    EXPECT_EQ(partitioner.SetParameter(cycleUB, parallelTH, cycleLB, useNodeHash), SUCCESS);
    EXPECT_EQ(partitioner.PartitionGraph(*function), SUCCESS);
}

void GetWideGraph(ComputationalGraphBuilder &G, int brNum) {
    std::vector<int64_t> tileShape{16,16};
    std::vector<std::string> outCast;
    EXPECT_EQ(G.AddTensors(DataType::DT_FP32, tileShape, {"h1", "h2", "h3"}), true);
    EXPECT_EQ(G.AddOp(Opcode::OP_COPY_IN, {"h1"}, {"h2"}, "COPY_IN", true), true);
    EXPECT_EQ(G.AddOp(Opcode::OP_ABS, {"h2"}, {"h3"}, "ABS", true), true);
    for (int i = 0; i < brNum; i++) {
        std::string br = std::to_string(i);
        std::vector<std::string> tensorNames{"t1" + br, "t2" + br, "t3" + br};
        std::vector<Opcode> opCodes{Opcode::OP_EXP, Opcode::OP_RESHAPE, Opcode::OP_COPY_OUT};
        std::vector<std::vector<std::string>> ioperands{{"h3"}, {"t1" + br}, {"t2" + br}};
        std::vector<std::vector<std::string>> ooperands{{"t1" + br}, {"t2" + br}, {"t3" + br}};
        std::vector<std::string> opNames{"EXP" + br, "RESHAPE" + br, "COPY_OUT" + br};
        EXPECT_EQ(G.AddTensors(DataType::DT_FP32, tileShape, tensorNames), true);
        EXPECT_EQ(G.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
        outCast.push_back("t3" + br);
    }
    EXPECT_EQ(G.SetInCast({"h1"}), true);
    EXPECT_EQ(G.SetOutCast(outCast), true);
}

TEST_F(GraphPartitionTest, TestLargeWideGraph) {
    ComputationalGraphBuilder G;
    const int brNum = 5000;
    GetWideGraph(G, brNum);
    Function *function = G.GetFunction();
    const int cycleUB = 100000;
    const int parallelTH = 20;
    const int cycleLB = 100000;
    const int useNodeHash = false;
    IsoPartitioner partitioner;
    EXPECT_EQ(partitioner.SetParameter(cycleUB, parallelTH, cycleLB, useNodeHash), SUCCESS);
    EXPECT_EQ(partitioner.PartitionGraph(*function), SUCCESS);
}

void GetDeepGraph(ComputationalGraphBuilder &G, int brNum) {
    std::vector<int64_t> tileShape{16,16};
    std::vector<std::string> outCast;
    EXPECT_EQ(G.AddTensors(DataType::DT_FP32, tileShape, {"ha", "a0", "hb", "b0"}), true);
    EXPECT_EQ(G.AddOp(Opcode::OP_COPY_IN, {"ha"}, {"a0"}, "COPY_INa", true), true);
    EXPECT_EQ(G.AddOp(Opcode::OP_COPY_IN, {"hb"}, {"b0"}, "COPY_INb", true), true);
    for (int i = 0; i < brNum; i++) {
        std::string br = std::to_string(i);
        std::string nbr = std::to_string(i+1);
        std::vector<std::string> tensorNames{"a" + nbr, "b" + nbr};
        std::vector<Opcode> opCodes{Opcode::OP_ABS, Opcode::OP_ABS};
        std::vector<std::vector<std::string>> ioperands{{"a" + br}, {"b" + br}};
        std::vector<std::vector<std::string>> ooperands{{"a" + nbr}, {"b" + nbr}};
        std::vector<std::string> opNames{"ABSa" + br, "ABSb" + br};
        EXPECT_EQ(G.AddTensors(DataType::DT_FP32, tileShape, tensorNames), true);
        EXPECT_EQ(G.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    }
    std::string tbr = std::to_string(brNum);
    EXPECT_EQ(G.AddTensors(DataType::DT_FP32, tileShape, {"ta", "tb"}), true);
    EXPECT_EQ(G.AddOp(Opcode::OP_COPY_OUT, {"a" + tbr}, {"ta"}, "COPY_OUTa", true), true);
    EXPECT_EQ(G.AddOp(Opcode::OP_COPY_OUT, {"b" + tbr}, {"tb"}, "COPY_OUTb", true), true);
    EXPECT_EQ(G.SetInCast({"ha", "hb"}), true);
    EXPECT_EQ(G.SetOutCast({"ta", "tb"}), true);
}

TEST_F(GraphPartitionTest, TestLargeDeepGraph) {
    ComputationalGraphBuilder G;
    const int brNum = 5000;
    GetDeepGraph(G, brNum);
    Function *function = G.GetFunction();
    const int cycleUB = 100000;
    const int parallelTH = 20;
    const int cycleLB = 100000;
    const int useNodeHash = false;
    IsoPartitioner partitioner;
    EXPECT_EQ(partitioner.SetParameter(cycleUB, parallelTH, cycleLB, useNodeHash), SUCCESS);
    EXPECT_EQ(partitioner.PartitionGraph(*function), SUCCESS);
}

TEST_F(GraphPartitionTest, TestIsomorphismGraph) {
    ComputationalGraphBuilder G;
    std::vector<int64_t> tileShape{16,16};
    EXPECT_EQ(G.AddTensors(DataType::DT_FP32, tileShape, {"h1", "h2", "h3", "h41", "h42", "h5", "h6"}), true);
    EXPECT_EQ(G.AddOp(Opcode::OP_COPY_IN, {"h1"}, {"h2"}, "COPY_IN", true), true);
    EXPECT_EQ(G.AddOp(Opcode::OP_ABS, {"h2"}, {"h3"}, "ABS", true), true);
    EXPECT_EQ(G.AddOp(Opcode::OP_ADDS, {"h3"}, {"h41"}, "ADDS1", true), true);
    EXPECT_EQ(G.AddOp(Opcode::OP_ADDS, {"h3"}, {"h42"}, "ADDS2", true), true);
    EXPECT_EQ(G.AddOp(Opcode::OP_MUL, {"h41", "h42"}, {"h5"}, "MUL", true), true);
    EXPECT_EQ(G.AddOp(Opcode::OP_COPY_OUT, {"h5"}, {"h6"}, "COPY_OUT", true), true);
    EXPECT_EQ(G.SetInCast({"h1"}), true);
    EXPECT_EQ(G.SetOutCast({"h6"}), true);

    Function *function = G.GetFunction();
    const int cycleUB = 100000;
    const int parallelTH = 20;
    const int cycleLB = 100000;
    const int useNodeHash = false;
    IsoPartitioner partitioner;
    EXPECT_EQ(partitioner.SetParameter(cycleUB, parallelTH, cycleLB, useNodeHash), SUCCESS);
    EXPECT_EQ(partitioner.PartitionGraph(*function), SUCCESS);
    const int subGraphNum = 4;
    EXPECT_EQ(function->GetTotalSubGraphCount(), subGraphNum);
}

TEST_F(GraphPartitionTest, TestScopeId) {
    ComputationalGraphBuilder G;
    std::vector<int64_t> tileShape{32,32};
    EXPECT_EQ(G.AddTensors(DataType::DT_FP32, tileShape, {"h1", "h2", "h3", "h41", "h42", "h5", "h6"}), true);
    EXPECT_EQ(G.AddOp(Opcode::OP_COPY_IN, {"h1"}, {"h2"}, "COPY_IN", true), true);
    EXPECT_EQ(G.AddOp(Opcode::OP_ABS, {"h2"}, {"h3"}, "ABS", true), true);
    EXPECT_EQ(G.AddOp(Opcode::OP_ADDS, {"h3"}, {"h41"}, "A1", true), true);
    EXPECT_EQ(G.AddOp(Opcode::OP_ADDS, {"h3"}, {"h42"}, "A2", true), true);
    EXPECT_EQ(G.AddOp(Opcode::OP_MUL, {"h41", "h42"}, {"h5"}, "M", true), true);
    EXPECT_EQ(G.AddOp(Opcode::OP_COPY_OUT, {"h5"}, {"h6"}, "COPY_OUT", true), true);
    EXPECT_EQ(G.SetInCast({"h1"}), true);
    EXPECT_EQ(G.SetOutCast({"h6"}), true);
    G.GetOp("A1")->SetScopeId(1);
    G.GetOp("A2")->SetScopeId(1);
    G.GetOp("M")->SetScopeId(1);
    G.GetOp("COPY_OUT")->SetScopeId(1);

    Function *function = G.GetFunction();
    const int cycleUB = 100000;
    const int parallelTH = 20;
    const int cycleLB = 100000;
    const int useNodeHash = false;
    IsoPartitioner partitioner;
    EXPECT_EQ(partitioner.SetParameter(cycleUB, parallelTH, cycleLB, useNodeHash), SUCCESS);
    EXPECT_EQ(partitioner.PartitionGraph(*function), SUCCESS);
    const int subGraphNum = 2;
    EXPECT_EQ(function->GetTotalSubGraphCount(), subGraphNum);
}

TEST_F(GraphPartitionTest, TestNonIsomorphismGraph) {
    ComputationalGraphBuilder G;
    std::vector<int64_t> tileShape{16,16};
    EXPECT_EQ(G.AddTensors(DataType::DT_FP32, tileShape, {"hin", "h2", "h3", "h41", "h42", "h5", "hout"}), true);
    EXPECT_EQ(G.AddOp(Opcode::OP_COPY_IN, {"hin"}, {"h2"}, "COPY_IN", true), true);
    EXPECT_EQ(G.AddOp(Opcode::OP_ABS, {"h2"}, {"h3"}, "ABS", true), true);
    EXPECT_EQ(G.AddOp(Opcode::OP_ADDS, {"h3"}, {"h41"}, "ADDS1", true), true);
    EXPECT_EQ(G.AddOp(Opcode::OP_MULS, {"h3"}, {"h42"}, "MULS2", true), true);
    EXPECT_EQ(G.AddOp(Opcode::OP_MUL, {"h41", "h42"}, {"h5"}, "MUL", true), true);
    EXPECT_EQ(G.AddOp(Opcode::OP_COPY_OUT, {"h5"}, {"hout"}, "COPY_OUT", true), true);
    EXPECT_EQ(G.SetInCast({"hin"}), true);
    EXPECT_EQ(G.SetOutCast({"hout"}), true);

    Function *function = G.GetFunction();
    const int cycleUB = 100000;
    const int parallelTH = 20;
    const int cycleLB = 100000;
    const int useNodeHash = false;
    IsoPartitioner partitioner;
    EXPECT_EQ(partitioner.SetParameter(cycleUB, parallelTH, cycleLB, useNodeHash), SUCCESS);
    EXPECT_EQ(partitioner.PartitionGraph(*function), SUCCESS);
    const int subGraphNum = 1;
    EXPECT_EQ(function->GetTotalSubGraphCount(), subGraphNum);
}

TEST_F(GraphPartitionTest, TestAvoidSuperNodeLoop) {
    ComputationalGraphBuilder G;
    std::vector<int64_t> tileShape{16,16};
    EXPECT_EQ(G.AddTensors(DataType::DT_FP32, tileShape, {"t1", "t2", "t3", "t4", "t5", "t6", "t7", "t8"}), true);
    std::vector<Opcode> opCodes{Opcode::OP_A_MUL_B, Opcode::OP_A_MUL_B, Opcode::OP_A_MUL_B, Opcode::OP_A_MULACC_B};
    std::vector<std::vector<std::string>> ioperands{{"t1", "t2"}, {"t2", "t3"}, {"t4","t5"}, {"t3", "t6", "t7"}};
    std::vector<std::vector<std::string>> ooperands{{"t3"}, {"t4"}, {"t6"}, {"t8"}};
    std::vector<std::string> opNames{"MUL1", "MUL2", "MUL3", "MULACC"};
    EXPECT_EQ(G.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    EXPECT_EQ(G.SetInCast({"t1", "t2", "t5", "t7"}), true);
    EXPECT_EQ(G.SetOutCast({"t8"}), true);
    Function *function = G.GetFunction();
    GraphPartition gpp;
    EXPECT_EQ(gpp.RunOnFunction(*function), SUCCESS);
    EXPECT_EQ(gpp.PostCheck(*function), SUCCESS);
}

TEST_F(GraphPartitionTest, TestBoundaryConvert) {
    ComputationalGraphBuilder G;
    std::vector<std::string> tensorNames{"t1", "t2", "t3", "t4"};
    std::vector<int64_t> tileShape{16,16};
    std::vector<MemoryType> tensorMemTypes{MemoryType::MEM_UB, MemoryType::MEM_UB, MemoryType::MEM_L1, MemoryType::MEM_L0A};
    std::vector<Opcode> opCodes{Opcode::OP_MULS, Opcode::OP_CONVERT, Opcode::OP_L1_TO_L0A};
    std::vector<std::vector<std::string>> ioperands{{"t1"}, {"t2"}, {"t3"}};
    std::vector<std::vector<std::string>> ooperands{{"t2"}, {"t3"}, {"t4"}};
    std::vector<std::string> opNames{"muls", "convert", "L1ToL0A"};
    EXPECT_EQ(G.AddTensors(DataType::DT_FP32, tileShape, tensorMemTypes, tensorNames, 0), true);
    EXPECT_EQ(G.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    G.GetOp("convert")->SetOpAttribute(std::make_shared<ConvertOpAttribute>(MemoryType::MEM_UB, MemoryType::MEM_L1));
    Function *function = G.GetFunction();
    GraphPartition gpp;
    EXPECT_EQ(gpp.RunOnFunction(*function), SUCCESS);
    EXPECT_EQ(G.GetOp("muls")->GetSubgraphID(), G.GetOp("convert")->GetSubgraphID());
    EXPECT_NE(G.GetOp("L1ToL0A")->GetSubgraphID(), G.GetOp("convert")->GetSubgraphID());
}

} // namespace tile_fwk
} // namespace npu