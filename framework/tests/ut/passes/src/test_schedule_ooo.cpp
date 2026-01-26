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
 * \file test_schedule_ooo.cpp
 * \brief Unit test for OoOSchedule.
 */

#include <gtest/gtest.h>
#include <vector>
#include "interface/function/function.h"
#include "tilefwk/tilefwk.h"
#include "tilefwk/platform.h"
#include "interface/inner/tilefwk.h"
#include "passes/pass_mgr/pass_manager.h"
#include "interface/configs/config_manager.h"
#include "passes/block_graph_pass/schedule_ooo/schedule_ooo.h"
#include "passes/block_graph_pass/schedule_ooo/core_assign.h"
#include "operator/models/deepseek/deepseek_mla.h"
#include "computational_graph_builder.h"

namespace npu::tile_fwk {
constexpr int OOO_NUM2 = 2;
constexpr int OOO_NUM209 = 209;
constexpr int UBPoolSize = 192 * 1024;
std::unordered_map<Opcode, int> preNodePriority = {
            // ALLOC 节点优先级最高，因为一个节点的前序ALLOC节点要在最靠近该节点的地方访问。
            {Opcode::OP_UB_ALLOC, 0}, {Opcode::OP_L1_ALLOC, 0}, {Opcode::OP_L0A_ALLOC, 0}, {Opcode::OP_L0B_ALLOC, 0},
            {Opcode::OP_L0C_ALLOC, 0}, {Opcode::OP_BT_ALLOC, 0}, {Opcode::OP_FIX_ALLOC, 0},
            // 其次是L0级数据搬运Op。
            {Opcode::OP_L1_TO_L0A, 1}, {Opcode::OP_L1_TO_L0B, 1}, {Opcode::OP_L1_TO_L0_AT, 1},
            {Opcode::OP_L1_TO_L0_BT, 1}, {Opcode::OP_L1_TO_FIX, 1}, {Opcode::OP_L1_TO_FIX_QUANT_PRE, 1},
            {Opcode::OP_L1_TO_FIX_RELU_PRE, 1}, {Opcode::OP_L1_TO_FIX_RELU_POST, 1},
            {Opcode::OP_L1_TO_FIX_QUANT_POST, 1}, {Opcode::OP_L1_TO_FIX_ELT_ANTIQ, 1},
            {Opcode::OP_L1_TO_FIX_MTE2_ANTIQ, 1}, {Opcode::OP_L1_TO_BT, 1},
            // 再其次是L1级数据搬运Op。
            {Opcode::OP_COPY_IN, 2}, {Opcode::OP_UB_COPY_IN, 2}, {Opcode::OP_L1_COPY_IN, 2},
            {Opcode::OP_L1_COPY_IN_FRACTAL_Z, 2}, {Opcode::OP_L1_COPY_UB, 2},
            {Opcode::OP_L0C_COPY_UB, 2}, {Opcode::OP_UB_COPY_L1, 2},
            // 最后访问其它计算节点（其它节点默认的优先级为10）。
        };
class ScheduleOoOTest : public ::testing::Test {
public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}

    void SetUp() override {
        Program::GetInstance().Reset();
        Platform::Instance().ObtainPlatformInfo();
    }
    void TearDown() override {}
};

IssueEntryPtr GetIssueEntry(const std::string& name, ComputationalGraphBuilder subGraph, OoOScheduler ooOScheduler) {
    EXPECT_NE(subGraph.GetOp(name), nullptr);
    Operation *op = subGraph.GetOp(name);
    for (auto &issue : ooOScheduler.issueEntries) {
        if (&(issue->tileOp) == op) {
            return issue;
        }
    }
    return nullptr;
}

void SetTensorAttr(LogicalTensorPtr tensor, MemoryType memType, int memId) {
    tensor->SetMemoryTypeOriginal(memType);
    tensor->SetMemoryTypeToBe(memType);
    tensor->memoryrange.memId = memId;
    tensor->UpdateDynValidShape({SymbolicScalar("S0"), SymbolicScalar("S1")});
}

void SetAllocAttr(Operation &alloc, int latency) {
    alloc.UpdateLatency(latency);
}

LogicalTensorPtr CreateTensor(Function &currFunction, DataType dateType, std::vector<int64_t> shape, MemoryType memType, int memId) {
    LogicalTensorPtr tensor = std::make_shared<LogicalTensor>(currFunction, dateType, shape);
    SetTensorAttr(tensor, memType, memId);
    return tensor;
}

Operation &CreateAllocOp(Function &currFunction, LogicalTensorPtr tensor, int latency) {
    Operation &alloc = currFunction.AddOperation(Opcode::OP_UB_ALLOC, {}, LogicalTensors({tensor}));
    SetAllocAttr(alloc, latency);
    return alloc;
}

Operation &CreateCopyOp(Function &currFunction, Opcode opcode, LogicalTensorPtr inTensor, LogicalTensorPtr outTensor, std::vector<int64_t> shape) {
    std::vector<int64_t> offset = {0, 0};
    auto &copy = currFunction.AddOperation(opcode, LogicalTensors({inTensor}), LogicalTensors({outTensor}));
    auto shapeImme = OpImmediate::Specified(shape);
    if (opcode == Opcode::OP_COPY_IN) {
        copy.SetOpAttribute(std::make_shared<CopyOpAttribute>(OpImmediate::Specified(offset), MEM_UB, shapeImme, shapeImme));
    }
    if (opcode == Opcode::OP_COPY_OUT) {
        copy.SetOpAttribute(std::make_shared<CopyOpAttribute>(MEM_UB, OpImmediate::Specified(offset), shapeImme, shapeImme));
    }
    return copy;
}

Operation &CreateAddOp(Function &currFunction, LogicalTensorPtr inTensor1, LogicalTensorPtr inTensor2, LogicalTensorPtr outTensor) {
    auto &add = currFunction.AddOperation(Opcode::OP_ADD, LogicalTensors({inTensor1, inTensor2}), LogicalTensors({outTensor}));
    return add;
}

void ReorderOperations(Function &function) {
    auto opList = function.Operations().DuplicatedOpList();
    std::vector<Operation *> newOperations;
    for (auto &op : opList) {
        if (op->GetOpcodeStr().find("ALLOC") != std::string::npos) {
            newOperations.insert(newOperations.begin(), op);
        } else {
            newOperations.push_back(op);
        }
    }
    function.ScheduleBy(newOperations);
}

TEST_F(ScheduleOoOTest, TestMainScheduleOoO) {
    auto rootFuncPtr = std::make_shared<Function>(Program::GetInstance(), "TestParams", "TestParams", nullptr);
    rootFuncPtr->rootFunc_ = rootFuncPtr.get();
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestOOO", "TestOOO", rootFuncPtr.get());
    currFunctionPtr->paramConfigs_.OoOPreScheduleMethod = "PriorDFS";
    auto emptyOpFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "", "", rootFuncPtr.get());
    EXPECT_TRUE(currFunctionPtr != nullptr);
    EXPECT_TRUE(emptyOpFunctionPtr != nullptr);
    currFunctionPtr->SetGraphType(GraphType::BLOCK_GRAPH);
    emptyOpFunctionPtr->SetGraphType(GraphType::BLOCK_GRAPH);
    rootFuncPtr->rootFunc_->programs_.emplace(currFunctionPtr->GetFuncMagic(), currFunctionPtr.get());
    rootFuncPtr->rootFunc_->programs_.emplace(emptyOpFunctionPtr->GetFuncMagic(), emptyOpFunctionPtr.get());
    std::vector<int64_t> shape = {128, 128};
    auto shapeImme = OpImmediate::Specified(shape);

    auto tensor1 = CreateTensor(*currFunctionPtr, DataType::DT_FP32, shape, MEM_DEVICE_DDR, 0);
    auto tensor2 = CreateTensor(*currFunctionPtr, DataType::DT_FP32, shape, MEM_DEVICE_DDR, 1);
    auto tensor3 = CreateTensor(*currFunctionPtr, DataType::DT_FP32, shape, MEM_UB, 2);
    auto tensor4 = CreateTensor(*currFunctionPtr, DataType::DT_FP32, shape, MEM_UB, 3);
    auto tensor5 = CreateTensor(*currFunctionPtr, DataType::DT_FP32, shape, MEM_UB, 4);
    auto tensor6 = CreateTensor(*currFunctionPtr, DataType::DT_FP32, shape, MEM_UB, 5);
    auto tensor7 = CreateTensor(*currFunctionPtr, DataType::DT_FP32, shape, MEM_DEVICE_DDR, 6);
    auto tensor8 = CreateTensor(*currFunctionPtr, DataType::DT_FP32, shape, MEM_UB, 7);
    auto tensor9 = CreateTensor(*currFunctionPtr, DataType::DT_FP32, shape, MEM_UB, 8);
    auto &alloc1 = CreateAllocOp(*currFunctionPtr, tensor3, 1);
    auto &alloc2 = CreateAllocOp(*currFunctionPtr, tensor4, 1);
    auto &alloc3 = CreateAllocOp(*currFunctionPtr, tensor5, 1);
    auto &alloc4 = CreateAllocOp(*currFunctionPtr, tensor6, 1);
    auto &alloc5 = CreateAllocOp(*currFunctionPtr, tensor8, 1);
    auto &alloc6 = CreateAllocOp(*currFunctionPtr, tensor9, 1);
    auto &copyin1 = CreateCopyOp(*currFunctionPtr, Opcode::OP_COPY_IN, tensor1, tensor3, shape);
    auto &copyin2 = CreateCopyOp(*currFunctionPtr, Opcode::OP_COPY_IN, tensor2, tensor4, shape);
    auto &add1 = CreateAddOp(*currFunctionPtr, tensor3, tensor4, tensor5);
    auto &add2 = CreateAddOp(*currFunctionPtr, tensor3, tensor4, tensor6);
    auto &add3 = CreateAddOp(*currFunctionPtr, tensor6, tensor4, tensor8);
    auto &add4 = CreateAddOp(*currFunctionPtr, tensor8, tensor5, tensor9);
    (void)alloc1, (void)alloc2, (void)alloc3, (void)alloc4, (void)alloc5, (void)alloc6,
    (void)copyin1, (void)copyin2, (void)add1, (void)add2, (void)add3, (void)add4;
    for (auto &program : rootFuncPtr->rootFunc_->programs_) {
        ReorderOperations(*(program.second));
    }
    currFunctionPtr->EndFunction(nullptr);
    emptyOpFunctionPtr->EndFunction(nullptr);
    OoOSchedule oooSchedule;
    EXPECT_EQ(oooSchedule.PreCheck(*rootFuncPtr), SUCCESS);
    oooSchedule.RunOnFunction(*rootFuncPtr);
    EXPECT_EQ(oooSchedule.PostCheck(*rootFuncPtr), SUCCESS);
}

static bool CheckExists(std::unordered_set<int> &issueList, Operation *op,
    std::unordered_map<int, IssueEntryPtr> issueEntryMap) {
    for (auto &issueId : issueList) {
        auto issue = issueEntryMap[issueId];
        if (&(issue->tileOp) == op) {
            return true;
        }
    }
    return false;
}

static bool CheckViewOps(std::vector<Operation*> &viewOps, Operation *op) {
    for (auto viewop : viewOps) {
        if (viewop == op) {
            return true;
        }
    }
    return false;
}

TEST_F(ScheduleOoOTest, TestDependencies) {
    ComputationalGraphBuilder subGraph;
    std::vector<std::string> tensorNames{"t1", "t2", "t3", "t4", "t5", "t6", "t7", "t8"};
    std::vector<MemoryType> tensorMemTypes{MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_UB, MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_UB,
        MemoryType::MEM_UB, MemoryType::MEM_UB, MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_UB};
    std::vector<Opcode> opCodes{Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC,
        Opcode::OP_UB_ALLOC, Opcode::OP_COPY_IN, Opcode::OP_COPY_IN, Opcode::OP_ROWMAX_SINGLE, Opcode::OP_ADD, Opcode::OP_COPY_OUT};
    std::vector<std::vector<std::string>> ioperands{{}, {}, {}, {}, {}, {"t1"}, {"t3"}, {"t2"}, {"t4", "t5"}, {"t5"}};
    std::vector<std::vector<std::string>> ooperands{{"t2"}, {"t4"}, {"t5"}, {"t6"}, {"t8"}, {"t2"}, {"t4"}, {"t5", "t6"}, {"t8"}, {"t7"}};
    std::vector<std::string> opNames{"Alloc1", "Alloc2", "Alloc3", "Alloc4", "Alloc5", "Copyin1", "Copyin2", "RowMax1", "Add1", "Copyout1"};
    EXPECT_EQ(subGraph.AddTensors(DataType::DT_FP32, {16, 16}, tensorMemTypes, tensorNames, 0), true);
    EXPECT_EQ(subGraph.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    Function *function = subGraph.GetFunction();
    EXPECT_NE(function, nullptr);

    OoOScheduler ooOScheduler(*function);
    Status res = ooOScheduler.Init(function->Operations().DuplicatedOpList());
    EXPECT_NE(GetIssueEntry("RowMax1", subGraph, ooOScheduler), nullptr);
    IssueEntryPtr issue = GetIssueEntry("RowMax1", subGraph, ooOScheduler);
    EXPECT_EQ(issue->predecessors.size(), 3);
    EXPECT_TRUE(CheckExists(issue->predecessors, subGraph.GetOp("Alloc4"), ooOScheduler.issueEntryMap));
    EXPECT_EQ(issue->successors.size(), 2);
    EXPECT_TRUE(CheckExists(issue->successors, subGraph.GetOp("Add1"), ooOScheduler.issueEntryMap));
    EXPECT_EQ(res, SUCCESS);
}

TEST_F(ScheduleOoOTest, TestDependenciesView) {
    ComputationalGraphBuilder subGraph;
    std::vector<std::string> tensorNames{"t1", "t2", "t3", "t4", "t5", "t6"};
    std::vector<MemoryType> tensorMemTypes{MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_UB, MemoryType::MEM_UB,
        MemoryType::MEM_UB, MemoryType::MEM_UB, MemoryType::MEM_UB};
    std::vector<Opcode> opCodes{Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_COPY_IN, Opcode::OP_VIEW,
        Opcode::OP_VIEW, Opcode::OP_VIEW, Opcode::OP_ADD};
    std::vector<std::vector<std::string>> ioperands{{}, {}, {"t1"}, {"t2"}, {"t2"}, {"t2"}, {"t3", "t4"}};
    std::vector<std::vector<std::string>> ooperands{{"t2"}, {"t6"}, {"t2"}, {"t3"}, {"t4"}, {"t5"}, {"t6"}};
    std::vector<std::string> opNames{"Alloc1", "Alloc2", "Copyin1", "View1", "View2", "View3", "Add1"};
    EXPECT_EQ(subGraph.AddTensors(DataType::DT_FP32, {16, 16}, tensorMemTypes, tensorNames, 0), true);
    EXPECT_EQ(subGraph.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    Function *function = subGraph.GetFunction();
    EXPECT_NE(function, nullptr);
    std::shared_ptr<LogicalTensor> tensor1 = subGraph.GetTensor("t3");
    tensor1->memoryrange.memId =
        subGraph.GetTensor("t2")->memoryrange.memId;
    std::shared_ptr<LogicalTensor> tensor2 = subGraph.GetTensor("t4");
    tensor2->memoryrange.memId =
        subGraph.GetTensor("t2")->memoryrange.memId;
    std::shared_ptr<LogicalTensor> tensor3 = subGraph.GetTensor("t5");
    tensor3->memoryrange.memId =
        subGraph.GetTensor("t2")->memoryrange.memId;
    OoOScheduler ooOScheduler(*function);
    Status res = ooOScheduler.Init(function->Operations().DuplicatedOpList());
    IssueEntryPtr copyin = GetIssueEntry("Copyin1", subGraph, ooOScheduler);
    EXPECT_NE(copyin, nullptr);
    IssueEntryPtr add = GetIssueEntry("Add1", subGraph, ooOScheduler);
    EXPECT_NE(add, nullptr);
    EXPECT_TRUE(CheckExists(copyin->predecessors, subGraph.GetOp("Alloc1"), ooOScheduler.issueEntryMap));
    EXPECT_TRUE(CheckExists(add->predecessors, subGraph.GetOp("Alloc2"), ooOScheduler.issueEntryMap));
    EXPECT_TRUE(CheckExists(add->predecessors, subGraph.GetOp("Copyin1"), ooOScheduler.issueEntryMap));
    EXPECT_TRUE(CheckViewOps(add->viewOps, subGraph.GetOp("View1")));
    EXPECT_TRUE(CheckViewOps(add->viewOps, subGraph.GetOp("View2")));
    EXPECT_EQ(res, SUCCESS);
}


TEST_F(ScheduleOoOTest, TestDependenciesAssemble) {
    ComputationalGraphBuilder subGraph;
    std::vector<std::string> tensorNames{"t1", "t2", "t3", "t4", "t5", "t6", "t7", "t8"};
    std::vector<MemoryType> tensorMemTypes{MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_DEVICE_DDR,
        MemoryType::MEM_UB, MemoryType::MEM_UB, MemoryType::MEM_UB, MemoryType::MEM_UB, MemoryType::MEM_DEVICE_DDR};
    std::vector<Opcode> opCodes{Opcode::OP_UB_ALLOC, Opcode::OP_SUB, Opcode::OP_SUB, Opcode::OP_SUB, Opcode::OP_ASSEMBLE,
        Opcode::OP_ASSEMBLE, Opcode::OP_ASSEMBLE, Opcode::OP_MUL};
    std::vector<std::vector<std::string>> ioperands{{}, {"t1"}, {"t2"}, {"t3"}, {"t4"}, {"t5"}, {"t6"}, {"t7"}};
    std::vector<std::vector<std::string>> ooperands{{"t4"}, {"t4"}, {"t5"}, {"t6"}, {"t7"}, {"t7"}, {"t7"}, {"t8"}};
    std::vector<std::string> opNames{"Alloc1", "Sub1", "Sub2", "Sub3", "Assemble1", "Assemble2", "Assemble3", "Mul1"};
    EXPECT_EQ(subGraph.AddTensors(DataType::DT_FP32, {16, 16}, tensorMemTypes, tensorNames, 0), true);
    EXPECT_EQ(subGraph.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    Function *function = subGraph.GetFunction();
    EXPECT_NE(function, nullptr);

    for (size_t i = 3; i < tensorNames.size() - 1; i++) {
        EXPECT_NE(subGraph.GetTensor(tensorNames[i]), nullptr);
        std::shared_ptr<LogicalTensor> tensor = subGraph.GetTensor(tensorNames[i]);
        tensor->memoryrange.memId =
            subGraph.GetTensor("t4")->memoryrange.memId;
    }

    OoOScheduler ooOScheduler(*function);
    Status res = ooOScheduler.Init(function->Operations().DuplicatedOpList());
    IssueEntryPtr alloc = GetIssueEntry("Alloc1", subGraph, ooOScheduler);
    EXPECT_NE(alloc, nullptr);
    IssueEntryPtr sub = GetIssueEntry("Sub1", subGraph, ooOScheduler);
    EXPECT_NE(sub, nullptr);
    EXPECT_TRUE(CheckExists(alloc->successors, subGraph.GetOp("Sub3"), ooOScheduler.issueEntryMap));
    EXPECT_TRUE(CheckExists(sub->predecessors, subGraph.GetOp("Alloc1"), ooOScheduler.issueEntryMap));
    EXPECT_TRUE(CheckExists(sub->successors, subGraph.GetOp("Assemble1"), ooOScheduler.issueEntryMap));
    EXPECT_EQ(res, SUCCESS);
}

TEST_F(ScheduleOoOTest, TestDependenciesInplace) {
    ComputationalGraphBuilder subGraph;
    std::vector<std::string> tensorNames{"t1", "t2", "t3", "t4"};
    std::vector<MemoryType> tensorMemTypes{MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_UB, MemoryType::MEM_UB, MemoryType::MEM_DEVICE_DDR};
    std::vector<Opcode> opCodes{Opcode::OP_UB_ALLOC, Opcode::OP_COPY_IN, Opcode::OP_ADD, Opcode::OP_COPY_OUT};
    std::vector<std::vector<std::string>> ioperands{{}, {"t1"}, {"t2"}, {"t3"}};
    std::vector<std::vector<std::string>> ooperands{{"t2"}, {"t2"}, {"t3"}, {"t4"}};
    std::vector<std::string> opNames{"Alloc1", "Copyin1", "Add1", "Copyout1"};
    EXPECT_EQ(subGraph.AddTensors(DataType::DT_FP32, {16, 16}, tensorMemTypes, tensorNames, 0), true);
    EXPECT_EQ(subGraph.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    Function *function = subGraph.GetFunction();
    EXPECT_NE(function, nullptr);

    EXPECT_NE(subGraph.GetTensor("t3"), nullptr);
    std::shared_ptr<LogicalTensor> tensor = subGraph.GetTensor("t3");
    tensor->memoryrange.memId =
        subGraph.GetTensor("t2")->memoryrange.memId;

    OoOScheduler ooOScheduler(*function);
    Status res = ooOScheduler.Init(function->Operations().DuplicatedOpList());
    IssueEntryPtr add = GetIssueEntry("Add1", subGraph, ooOScheduler);
    EXPECT_NE(add, nullptr);
    EXPECT_TRUE(CheckExists(add->successors, subGraph.GetOp("Copyout1"), ooOScheduler.issueEntryMap));
    EXPECT_TRUE(CheckExists(add->predecessors, subGraph.GetOp("Copyin1"), ooOScheduler.issueEntryMap));
    EXPECT_EQ(res, SUCCESS);
}

TEST_F(ScheduleOoOTest, TestDependenciesTrue) {
    ComputationalGraphBuilder subGraph;
    std::vector<std::string> tensorNames{"t1", "t2", "t3", "t4"};
    std::vector<MemoryType> tensorMemTypes{MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_UB, MemoryType::MEM_UB, MemoryType::MEM_DEVICE_DDR};
    std::vector<Opcode> opCodes{Opcode::OP_COPY_IN, Opcode::OP_UB_ALLOC, Opcode::OP_ADD, Opcode::OP_COPY_OUT};
    std::vector<std::vector<std::string>> ioperands{{"t1"}, {}, {"t2"}, {"t3"}};
    std::vector<std::vector<std::string>> ooperands{{"t2"}, {"t2"}, {"t3"}, {"t4"}};
    std::vector<std::string> opNames{"Copyin1", "Alloc1", "Add1", "Copyout1"};
    EXPECT_EQ(subGraph.AddTensors(DataType::DT_FP32, {16, 16}, tensorMemTypes, tensorNames, 0), true);
    EXPECT_EQ(subGraph.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    Function *function = subGraph.GetFunction();
    EXPECT_NE(function, nullptr);

    EXPECT_NE(subGraph.GetTensor("t3"), nullptr);
    std::shared_ptr<LogicalTensor> tensor = subGraph.GetTensor("t3");
    tensor->memoryrange.memId =
        subGraph.GetTensor("t2")->memoryrange.memId;

    OoOScheduler ooOScheduler(*function);
    Status res = ooOScheduler.Init(function->Operations().DuplicatedOpList());
    EXPECT_EQ(res, SUCCESS);
    std::rotate(ooOScheduler.issueEntries.begin(), ooOScheduler.issueEntries.begin() + 1, ooOScheduler.issueEntries.end());
    res = ooOScheduler.InitDependencies();
    EXPECT_EQ(res, SUCCESS);
}

TEST_F(ScheduleOoOTest, TestSpillCopyIn) {
    ComputationalGraphBuilder subGraph;
    std::vector<std::string> tensorNames{"t1", "t2", "t3", "t4", "t5", "t6", "t7", "t8"};
    std::vector<MemoryType> tensorMemTypes{MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_UB,
        MemoryType::MEM_UB, MemoryType::MEM_UB, MemoryType::MEM_UB, MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_UB};
    std::vector<Opcode> opCodes{Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC,
        Opcode::OP_COPY_IN, Opcode::OP_COPY_IN, Opcode::OP_ADD, Opcode::OP_ADD, Opcode::OP_ADD, Opcode::OP_COPY_OUT};
    std::vector<std::vector<std::string>> ioperands{{}, {}, {}, {}, {}, {"t1"}, {"t2"}, {"t3", "t4"}, {"t3", "t5"}, {"t4", "t6"}, {"t8"}};
    std::vector<std::vector<std::string>> ooperands{{"t3"}, {"t4"}, {"t5"}, {"t6"}, {"t8"}, {"t3"}, {"t4"}, {"t5"}, {"t6"}, {"t8"}, {"t7"}};
    std::vector<std::string> opNames{"Alloc1", "Alloc2", "Alloc3", "Alloc4", "Alloc5", "Copyin1", "Copyin2", "Add1", "Add2", "Add3", "Copyout1"};
    EXPECT_EQ(subGraph.AddTensors(DataType::DT_FP32, {128, 128}, tensorMemTypes, tensorNames, 0), true);
    EXPECT_EQ(subGraph.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    Function *function = subGraph.GetFunction();
    EXPECT_NE(function, nullptr);

    OoOScheduler ooOScheduler(*function);
    Status res = ooOScheduler.Init(function->Operations().DuplicatedOpList());
    EXPECT_EQ(res, SUCCESS);
    res = ooOScheduler.SortOps();
    EXPECT_EQ(res, SUCCESS);
    res = ooOScheduler.GenSpillSchedule();
    EXPECT_EQ(res, SUCCESS);
    EXPECT_EQ(ooOScheduler.issueEntries[9]->tileOp.GetOpcodeStr(), "UB_ALLOC");
    EXPECT_EQ(ooOScheduler.issueEntries[10]->tileOp.GetOpcodeStr(), "COPY_IN");
}

TEST_F(ScheduleOoOTest, TestSpill) {
    ComputationalGraphBuilder subGraph;
    std::vector<std::string> tensorNames{"t1", "t2", "t3", "t4", "t5", "t6", "t7", "t8", "t9"};
    std::vector<MemoryType> tensorMemTypes{MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_UB,
        MemoryType::MEM_UB, MemoryType::MEM_UB, MemoryType::MEM_UB, MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_UB, MemoryType::MEM_UB};
    std::vector<Opcode> opCodes{Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC,
        Opcode::OP_COPY_IN, Opcode::OP_COPY_IN, Opcode::OP_ADD, Opcode::OP_ADD, Opcode::OP_ADD, Opcode::OP_ADD, Opcode::OP_COPY_OUT};
    std::vector<std::vector<std::string>> ioperands{{}, {}, {}, {}, {}, {}, {"t1"}, {"t2"}, {"t3", "t4"}, {"t3", "t4"}, {"t4", "t6"}, {"t5", "t8"}, {"t9"}};
    std::vector<std::vector<std::string>> ooperands{{"t3"}, {"t4"}, {"t5"}, {"t6"}, {"t8"}, {"t9"}, {"t3"}, {"t4"}, {"t5"}, {"t6"}, {"t8"}, {"t9"}, {"t7"}};
    std::vector<std::string> opNames{"Alloc1", "Alloc2", "Alloc3", "Alloc4", "Alloc5", "Alloc6", "Copyin1", "Copyin2", "Add1", "Add2", "Add3", "Add4", "Copyout1"};
    EXPECT_EQ(subGraph.AddTensors(DataType::DT_FP32, {128, 128}, tensorMemTypes, tensorNames, 0), true);
    EXPECT_EQ(subGraph.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    Function *function = subGraph.GetFunction();
    EXPECT_NE(function, nullptr);

    OoOScheduler ooOScheduler(*function);
    Status res = ooOScheduler.Init(function->Operations().DuplicatedOpList());
    EXPECT_EQ(res, SUCCESS);
    res = ooOScheduler.SortOps();
    EXPECT_EQ(res, SUCCESS);
    res = ooOScheduler.GenSpillSchedule();
    EXPECT_EQ(res, SUCCESS);
    EXPECT_EQ(ooOScheduler.issueEntries[8]->tileOp.GetOpcodeStr(), "COPY_OUT");
    EXPECT_EQ(ooOScheduler.issueEntries[14]->tileOp.GetOpcodeStr(), "UB_ALLOC");
    EXPECT_EQ(ooOScheduler.issueEntries[15]->tileOp.GetOpcodeStr(), "COPY_IN");
}

TEST_F(ScheduleOoOTest, TestSpillInplace) {
    ComputationalGraphBuilder subGraph;
    std::vector<std::string> tensorNames{"t1", "t2", "t3", "t4", "t5", "t6", "t7", "t8", "t9", "t10", "t11"};
    std::vector<MemoryType> tensorMemTypes{MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_DEVICE_DDR,
        MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_UB, MemoryType::MEM_UB, MemoryType::MEM_UB, MemoryType::MEM_UB, MemoryType::MEM_UB,
        MemoryType::MEM_UB, MemoryType::MEM_UB};
    std::vector<Opcode> opCodes{Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC,
        Opcode::OP_COPY_IN, Opcode::OP_COPY_IN, Opcode::OP_COPY_IN, Opcode::OP_COPY_IN, Opcode::OP_ADD, Opcode::OP_ADD, Opcode::OP_ADD};
    std::vector<std::vector<std::string>> ioperands{{}, {}, {}, {}, {}, {"t1"}, {"t2"}, {"t3"}, {"t4"}, {"t5", "t6"}, {"t7", "t8"}, {"t9", "t10"}};
    std::vector<std::vector<std::string>> ooperands{{"t5"}, {"t6"}, {"t7"}, {"t8"}, {"t9"}, {"t5"}, {"t6"}, {"t7"}, {"t8"}, {"t10"}, {"t9"}, {"t11"}};
    std::vector<std::string> opNames{"Alloc1", "Alloc2", "Alloc3", "Alloc4", "Alloc5", "Copyin1", "Copyin2", "Copyin3", "Copyin4", "Add1", "Add2", "Add3"};
    EXPECT_EQ(subGraph.AddTensors(DataType::DT_FP32, {128, 128}, tensorMemTypes, tensorNames, 0), true);
    EXPECT_EQ(subGraph.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    Function *function = subGraph.GetFunction();
    EXPECT_NE(function, nullptr);

    EXPECT_NE(subGraph.GetTensor("t11"), nullptr);
    std::shared_ptr<LogicalTensor> tensor1 = subGraph.GetTensor("t10");
    tensor1->memoryrange.memId =
        subGraph.GetTensor("t5")->memoryrange.memId;
    std::shared_ptr<LogicalTensor> tensor2 = subGraph.GetTensor("t11");
    tensor2->memoryrange.memId =
        subGraph.GetTensor("t5")->memoryrange.memId;

    OoOScheduler ooOScheduler(*function);
    Status res = ooOScheduler.Init(function->Operations().DuplicatedOpList());
    EXPECT_EQ(res, SUCCESS);
    res = ooOScheduler.SortOps();
    EXPECT_EQ(res, SUCCESS);
    std::rotate(ooOScheduler.issueEntries.begin(), ooOScheduler.issueEntries.begin() + 6, ooOScheduler.issueEntries.begin() + 11);
    res = ooOScheduler.GenSpillSchedule();
    EXPECT_EQ(res, SUCCESS);
    IssueEntryPtr add1 = GetIssueEntry("Add1", subGraph, ooOScheduler);
    EXPECT_NE(add1, nullptr);
    IssueEntryPtr add3 = GetIssueEntry("Add3", subGraph, ooOScheduler);
    EXPECT_NE(add3, nullptr);
    EXPECT_EQ(ooOScheduler.issueEntryMap[(*add1->successors.begin())]->tileOp.GetOpcodeStr(), "COPY_OUT");
    EXPECT_EQ(add3->predecessors.size(), 3);
}

TEST_F(ScheduleOoOTest, TestSpillMultiTensor) {
    ComputationalGraphBuilder subGraph;
    std::vector<std::string> tensorNames{"t1", "t2", "t3", "t4", "t5", "t6", "t7", "t8", "t9", "t10", "t11"};
    std::vector<MemoryType> tensorMemTypes{MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_DEVICE_DDR,
        MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_UB, MemoryType::MEM_UB, MemoryType::MEM_UB, MemoryType::MEM_UB, MemoryType::MEM_UB,
        MemoryType::MEM_UB, MemoryType::MEM_UB};
    std::vector<Opcode> opCodes{Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC,
        Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_COPY_IN, Opcode::OP_COPY_IN, Opcode::OP_COPY_IN, Opcode::OP_COPY_IN,
        Opcode::OP_ADD, Opcode::OP_ADD, Opcode::OP_ADD};
    std::vector<std::vector<std::string>> ioperands{{}, {}, {}, {}, {}, {}, {}, {"t1"}, {"t2"}, {"t3"}, {"t4"}, {"t7", "t8"},
        {"t5", "t6", "t9"}, {"t7", "t8", "t10"}};
    std::vector<std::vector<std::string>> ooperands{{"t5"}, {"t6"}, {"t7"}, {"t8"}, {"t9"}, {"t10"}, {"t11"}, {"t5"}, {"t6"}, {"t7"}, {"t8"},
        {"t9"}, {"t10"}, {"t11"}};
    std::vector<std::string> opNames{"Alloc1", "Alloc2", "Alloc3", "Alloc4", "Alloc5", "Alloc6", "Alloc7", "Copyin1", "Copyin2", "Copyin3", "Copyin4",
        "Add1", "Add2", "Add3"};
    EXPECT_EQ(subGraph.AddTensors(DataType::DT_FP32, {50, 128}, tensorMemTypes, tensorNames, 0), true);
    EXPECT_EQ(subGraph.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    Function *function = subGraph.GetFunction();
    EXPECT_NE(function, nullptr);

    EXPECT_NE(subGraph.GetTensor("t6"), nullptr);
    std::shared_ptr<LogicalTensor> tensor1 = subGraph.GetTensor("t5");
    tensor1->shape = {128, 128};
    tensor1->tensor->rawshape = {128, 128};
    std::shared_ptr<LogicalTensor> tensor2 = subGraph.GetTensor("t6");
    tensor2->shape = {128, 128};
    tensor2->tensor->rawshape = {128, 128};

    OoOScheduler ooOScheduler(*function);
    Status res = ooOScheduler.Init(function->Operations().DuplicatedOpList());
    EXPECT_EQ(res, SUCCESS);
    res = ooOScheduler.SortOps();
    EXPECT_EQ(res, SUCCESS);
    res = ooOScheduler.GenSpillSchedule();
    EXPECT_EQ(res, SUCCESS);
    EXPECT_EQ(ooOScheduler.issueEntries.size(), 21);
}

TEST_F(ScheduleOoOTest, TestSpillView) {
    ComputationalGraphBuilder subGraph;
    std::vector<std::string> tensorNames{"t1", "t2", "t3", "t4", "t5", "t6", "t7", "t8", "t9"};
    std::vector<MemoryType> tensorMemTypes{MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_UB,
        MemoryType::MEM_UB, MemoryType::MEM_UB, MemoryType::MEM_UB, MemoryType::MEM_UB, MemoryType::MEM_UB, MemoryType::MEM_UB};
    std::vector<Opcode> opCodes{Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC,
        Opcode::OP_COPY_IN, Opcode::OP_COPY_IN, Opcode::OP_VIEW, Opcode::OP_VIEW, Opcode::OP_ADD, Opcode::OP_ADD, Opcode::OP_ADD};
    std::vector<std::vector<std::string>> ioperands{{}, {}, {}, {}, {}, {"t1"}, {"t2"}, {"t3"}, {"t3"}, {"t5"}, {"t6"}, {"t4", "t7"}};
    std::vector<std::vector<std::string>> ooperands{{"t3"}, {"t4"}, {"t7"}, {"t8"}, {"t9"}, {"t3"}, {"t4"}, {"t5"}, {"t6"}, {"t7"}, {"t8"}, {"t9"}};
    std::vector<std::string> opNames{"Alloc1", "Alloc2", "Alloc3", "Alloc4", "Alloc5", "Copyin1", "Copyin2", "View1", "View2", "Add1", "Add2", "Add3"};
    EXPECT_EQ(subGraph.AddTensors(DataType::DT_FP32, {128, 128}, tensorMemTypes, tensorNames, 0), true);
    EXPECT_EQ(subGraph.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    Function *function = subGraph.GetFunction();
    EXPECT_NE(function, nullptr);

    EXPECT_NE(subGraph.GetTensor("t6"), nullptr);
    std::shared_ptr<LogicalTensor> tensor1 = subGraph.GetTensor("t5");
    tensor1->memoryrange.memId =
        subGraph.GetTensor("t3")->memoryrange.memId;
    std::shared_ptr<LogicalTensor> tensor2 = subGraph.GetTensor("t6");
    tensor2->memoryrange.memId =
        subGraph.GetTensor("t3")->memoryrange.memId;

    OoOScheduler ooOScheduler(*function);
    Status res = ooOScheduler.Init(function->Operations().DuplicatedOpList());
    EXPECT_EQ(res, SUCCESS);
    res = ooOScheduler.SortOps();
    EXPECT_EQ(res, SUCCESS);
    res = ooOScheduler.GenSpillSchedule();
    EXPECT_EQ(res, SUCCESS);
}

TEST_F(ScheduleOoOTest, TestSpillAssemble) {
    ComputationalGraphBuilder subGraph;
    std::vector<std::string> tensorNames{"t1", "t2", "t3", "t4", "t5", "t6", "t7", "t8", "t9", "t10", "t11", "t12", "t13", "t14"};
    std::vector<MemoryType> tensorMemTypes{MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_DEVICE_DDR,
        MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_UB, MemoryType::MEM_UB, MemoryType::MEM_UB, MemoryType::MEM_UB, MemoryType::MEM_UB,
        MemoryType::MEM_UB, MemoryType::MEM_UB, MemoryType::MEM_UB, MemoryType::MEM_UB, MemoryType::MEM_UB};
    std::vector<Opcode> opCodes{Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_ADD,
        Opcode::OP_ADD, Opcode::OP_ADD, Opcode::OP_ADD, Opcode::OP_ADD, Opcode::OP_ADD, Opcode::OP_SUB, Opcode::OP_SUB, Opcode::OP_ASSEMBLE, Opcode::OP_ASSEMBLE,
        Opcode::OP_ASSEMBLE, Opcode::OP_ASSEMBLE, Opcode::OP_MUL};
    std::vector<std::vector<std::string>> ioperands{{}, {}, {}, {}, {}, {"t1"}, {"t2"}, {"t3"}, {"t4"}, {"t5"}, {"t6"}, {"t7"}, {"t8"}, {"t9"}, {"t10"}, {"t11"}, {"t12"}, {"t13"}};
    std::vector<std::vector<std::string>> ooperands{{"t5"}, {"t6"}, {"t7"}, {"t8"}, {"t12"}, {"t5"}, {"t6"}, {"t7"}, {"t8"}, {"t9"}, {"t10"}, {"t11"}, {"t12"}, {"t13"}, {"t13"}, {"t13"}, {"t13"}, {"t14"}};
    std::vector<std::string> opNames{"Alloc1", "Alloc2", "Alloc3", "Alloc4", "Alloc5", "Copyin1", "Copyin2", "Copyin3", "Copyin4", "Add1", "Add2", "Sub1", "Sub2", "Assemble1", "Assemble2", "Assemble3", "Assemble4", "Mul1"};
    EXPECT_EQ(subGraph.AddTensors(DataType::DT_FP32, {128, 64}, tensorMemTypes, tensorNames, 0), true);
    EXPECT_EQ(subGraph.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    Function *function = subGraph.GetFunction();
    EXPECT_NE(function, nullptr);

    EXPECT_NE(subGraph.GetTensor("t9"), nullptr);
    std::shared_ptr<LogicalTensor> tensor1 = subGraph.GetTensor("t10");
    tensor1->memoryrange.memId =
        subGraph.GetTensor("t9")->memoryrange.memId;
    std::shared_ptr<LogicalTensor> tensor2 = subGraph.GetTensor("t11");
    tensor2->memoryrange.memId =
        subGraph.GetTensor("t9")->memoryrange.memId;
    tensor2->shape= {128, 128};
    tensor2->tensor->rawshape= {128, 128};
    std::shared_ptr<LogicalTensor> tensor3 = subGraph.GetTensor("t12");
    tensor3->memoryrange.memId =
        subGraph.GetTensor("t9")->memoryrange.memId;
    tensor3->shape= {128, 128};
    tensor3->tensor->rawshape= {128, 128};
    std::shared_ptr<LogicalTensor> tensor4 = subGraph.GetTensor("t13");
    tensor4->memoryrange.memId =
        subGraph.GetTensor("t9")->memoryrange.memId;
    tensor4->shape = {128, 256};
    tensor4->tensor->rawshape = {128, 256};
    std::shared_ptr<LogicalTensor> tensor5 = subGraph.GetTensor("t14");
    tensor5->memoryrange.memId =
        subGraph.GetTensor("t9")->memoryrange.memId;
    tensor5->shape = {128, 256};
    tensor5->tensor->rawshape = {128, 256};
    std::shared_ptr<LogicalTensor> tensor6 = subGraph.GetTensor("t7");
    tensor6->shape = {128, 128};
    tensor6->tensor->rawshape = {128, 128};
    std::shared_ptr<LogicalTensor> tensor7 = subGraph.GetTensor("t8");
    tensor7->shape = {128, 128};
    tensor7->tensor->rawshape = {128, 128};
    std::vector<int64_t> offset1 = {0, 0};
    std::vector<int64_t> offset2 = {0, 128};
    std::vector<int64_t> offset3 = {64, 0};
    std::vector<int64_t> offset4 = {64, 128};
    auto assembleAttr = std::make_shared<AssembleOpAttribute>(MemoryType::MEM_UB, offset1);
    auto assemble1 = subGraph.GetOp("Assemble1");
    assemble1->SetOpAttribute(assembleAttr);
    auto assembleAttr2 = std::make_shared<AssembleOpAttribute>(MemoryType::MEM_UB, offset2);
    auto assemble2 = subGraph.GetOp("Assemble2");
    assemble2->SetOpAttribute(assembleAttr2);
    auto assembleAttr3 = std::make_shared<AssembleOpAttribute>(MemoryType::MEM_UB, offset3);
    auto assemble3 = subGraph.GetOp("Assemble3");
    assemble3->SetOpAttribute(assembleAttr3);
    auto assembleAttr4 = std::make_shared<AssembleOpAttribute>(MemoryType::MEM_UB, offset4);
    auto assemble4 = subGraph.GetOp("Assemble4");
    assemble4->SetOpAttribute(assembleAttr4);

    OoOScheduler ooOScheduler(*function);
    Status res = ooOScheduler.Init(function->Operations().DuplicatedOpList());
    EXPECT_EQ(res, SUCCESS);
    res = ooOScheduler.SortOps();
    EXPECT_EQ(res, SUCCESS);
    res = ooOScheduler.GenSpillSchedule();
    EXPECT_EQ(res, SUCCESS);
}

TEST_F(ScheduleOoOTest, TestSpillL0AFailed) {
    ComputationalGraphBuilder subGraph;
    std::vector<std::string> tensorNames{"t1", "t2", "t3", "t4", "t5", "t6", "t7", "t8", "t9", "t10", "t11", "t12"};
    std::vector<MemoryType> tensorMemTypes{MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_L1, MemoryType::MEM_L0A,
        MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_L1, MemoryType::MEM_L0A, MemoryType::MEM_L0C, MemoryType::MEM_L0C, MemoryType::MEM_L0C,
        MemoryType::MEM_L0C, MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_DEVICE_DDR};
    std::vector<Opcode> opCodes{Opcode::OP_L1_ALLOC, Opcode::OP_L1_ALLOC, Opcode::OP_L0A_ALLOC, Opcode::OP_L0A_ALLOC, Opcode::OP_L0C_ALLOC,
        Opcode::OP_L0C_ALLOC, Opcode::OP_COPY_IN, Opcode::OP_COPY_IN, Opcode::OP_L1_TO_L0A, Opcode::OP_L1_TO_L0A, Opcode::OP_A_MUL_B, Opcode::OP_A_MUL_B,
        Opcode::OP_A_MULACC_B, Opcode::OP_A_MULACC_B, Opcode::OP_COPY_OUT, Opcode::OP_COPY_OUT};
    std::vector<std::vector<std::string>> ioperands{{}, {}, {}, {}, {}, {}, {"t1"}, {"t4"}, {"t2"}, {"t5"}, {"t3"}, {"t3"}, {"t6", "t7"}, {"t6", "t8"}, {"t9"}, {"t10"}};
    std::vector<std::vector<std::string>> ooperands{{"t2"}, {"t5"}, {"t3"}, {"t6"}, {"t7"}, {"t8"}, {"t2"}, {"t5"}, {"t3"}, {"t6"}, {"t7"}, {"t8"}, {"t9"}, {"t10"}, {"t11"}, {"t12"}};
    std::vector<std::string> opNames{"Alloc1", "Alloc2", "Alloc3", "Alloc4", "Alloc5", "Alloc6", "Copyin1", "Copyin2",
        "L1toL0A1", "L1toL0A2", "AMulB1", "AMulB2", "AMulaccB1", "AMulaccB2", "Copyout1", "Copyout2"};
    EXPECT_EQ(subGraph.AddTensors(DataType::DT_FP32, {128, 128}, tensorMemTypes, tensorNames, 0), true);
    EXPECT_EQ(subGraph.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    Function *function = subGraph.GetFunction();
    EXPECT_NE(function, nullptr);

    EXPECT_NE(subGraph.GetTensor("t10"), nullptr);
    std::shared_ptr<LogicalTensor> tensor1 = subGraph.GetTensor("t9");
    tensor1->memoryrange.memId =
        subGraph.GetTensor("t7")->memoryrange.memId;
    std::shared_ptr<LogicalTensor> tensor2 = subGraph.GetTensor("t10");
    tensor2->memoryrange.memId =
        subGraph.GetTensor("t8")->memoryrange.memId;

    OoOScheduler ooOScheduler(*function);
    Status res = ooOScheduler.Init(function->Operations().DuplicatedOpList());
    EXPECT_EQ(res, SUCCESS);
    res = ooOScheduler.SortOps();
    EXPECT_EQ(res, SUCCESS);
    res = ooOScheduler.GenSpillSchedule();
    EXPECT_EQ(res, SUCCESS);
}

TEST_F(ScheduleOoOTest, TestSchedule) {
    ComputationalGraphBuilder subGraph;
    std::vector<std::string> tensorNames{"t1", "t2", "t3", "t4", "t5", "t6", "t7", "t8", "t9"};
    std::vector<MemoryType> tensorMemTypes{MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_UB,
        MemoryType::MEM_UB, MemoryType::MEM_UB, MemoryType::MEM_UB, MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_UB, MemoryType::MEM_UB};
    std::vector<Opcode> opCodes{Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC,
        Opcode::OP_COPY_IN, Opcode::OP_COPY_IN, Opcode::OP_ADD, Opcode::OP_ADD, Opcode::OP_ADD, Opcode::OP_ADD, Opcode::OP_COPY_OUT};
    std::vector<std::vector<std::string>> ioperands{{}, {}, {}, {}, {}, {}, {"t1"}, {"t2"}, {"t3", "t4"}, {"t3", "t4"}, {"t4", "t6"}, {"t5", "t8"}, {"t9"}};
    std::vector<std::vector<std::string>> ooperands{{"t3"}, {"t4"}, {"t5"}, {"t6"}, {"t8"}, {"t9"}, {"t3"}, {"t4"}, {"t5"}, {"t6"}, {"t8"}, {"t9"}, {"t7"}};
    std::vector<std::string> opNames{"Alloc1", "Alloc2", "Alloc3", "Alloc4", "Alloc5", "Alloc6", "Copyin1", "Copyin2", "Add1", "Add2", "Add3", "Add4", "Copyout1"};
    EXPECT_EQ(subGraph.AddTensors(DataType::DT_FP32, {64, 64}, tensorMemTypes, tensorNames, 0), true);
    EXPECT_EQ(subGraph.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    Function *function = subGraph.GetFunction();
    EXPECT_NE(function, nullptr);

    OoOScheduler ooOScheduler(*function);
    Status res = ooOScheduler.Schedule(function->Operations().DuplicatedOpList());
    EXPECT_EQ(res, SUCCESS);
    IssueEntryPtr add = GetIssueEntry("Add2", subGraph, ooOScheduler);
    EXPECT_NE(add, nullptr);
    EXPECT_EQ(add->tileOp.oOperand[0]->memoryrange.start, 32768);
    EXPECT_EQ(add->tileOp.oOperand[0]->memoryrange.end, 49152);
}

TEST_F(ScheduleOoOTest, TestScheduleInplace) {
    ComputationalGraphBuilder subGraph;
    std::vector<std::string> tensorNames{"t1", "t2", "t3", "t4", "t5", "t6", "t7", "t8", "t9", "t10", "t11"};
    std::vector<MemoryType> tensorMemTypes{MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_DEVICE_DDR,
        MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_UB, MemoryType::MEM_UB, MemoryType::MEM_UB, MemoryType::MEM_UB, MemoryType::MEM_UB,
        MemoryType::MEM_UB, MemoryType::MEM_UB};
    std::vector<Opcode> opCodes{Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC,
        Opcode::OP_COPY_IN, Opcode::OP_COPY_IN, Opcode::OP_COPY_IN, Opcode::OP_COPY_IN, Opcode::OP_ADD, Opcode::OP_ADD, Opcode::OP_ADD};
    std::vector<std::vector<std::string>> ioperands{{}, {}, {}, {}, {}, {"t1"}, {"t2"}, {"t3"}, {"t4"}, {"t5", "t6"}, {"t7", "t8"}, {"t9", "t10"}};
    std::vector<std::vector<std::string>> ooperands{{"t5"}, {"t6"}, {"t7"}, {"t8"}, {"t9"}, {"t5"}, {"t6"}, {"t7"}, {"t8"}, {"t10"}, {"t9"}, {"t11"}};
    std::vector<std::string> opNames{"Alloc1", "Alloc2", "Alloc3", "Alloc4", "Alloc5", "Copyin1", "Copyin2", "Copyin3", "Copyin4", "Add1", "Add2", "Add3"};
    EXPECT_EQ(subGraph.AddTensors(DataType::DT_FP32, {64, 64}, tensorMemTypes, tensorNames, 0), true);
    EXPECT_EQ(subGraph.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    Function *function = subGraph.GetFunction();
    EXPECT_NE(function, nullptr);

    EXPECT_NE(subGraph.GetTensor("t11"), nullptr);
    std::shared_ptr<LogicalTensor> tensor1 = subGraph.GetTensor("t10");
    tensor1->memoryrange.memId =
        subGraph.GetTensor("t5")->memoryrange.memId;
    std::shared_ptr<LogicalTensor> tensor2 = subGraph.GetTensor("t11");
    tensor2->memoryrange.memId =
        subGraph.GetTensor("t5")->memoryrange.memId;

    OoOScheduler ooOScheduler(*function);
    Status res = ooOScheduler.Schedule(function->Operations().DuplicatedOpList());
    EXPECT_EQ(res, SUCCESS);
    IssueEntryPtr copyin = GetIssueEntry("Copyin1", subGraph, ooOScheduler);
    EXPECT_NE(copyin, nullptr);
    IssueEntryPtr add1 = GetIssueEntry("Add1", subGraph, ooOScheduler);
    EXPECT_NE(add1, nullptr);
    IssueEntryPtr add3 = GetIssueEntry("Add3", subGraph, ooOScheduler);
    EXPECT_NE(add3, nullptr);
    EXPECT_EQ(copyin->tileOp.oOperand[0]->memoryrange.start, 16384);
    EXPECT_EQ(copyin->tileOp.oOperand[0]->memoryrange.end, 32768);
    EXPECT_EQ(add1->tileOp.oOperand[0]->memoryrange.start, 16384);
    EXPECT_EQ(add1->tileOp.oOperand[0]->memoryrange.end, 32768);
    EXPECT_EQ(add3->tileOp.oOperand[0]->memoryrange.start, 16384);
    EXPECT_EQ(add3->tileOp.oOperand[0]->memoryrange.end, 32768);
}

TEST_F(ScheduleOoOTest, TestScheduleView) {
    ComputationalGraphBuilder subGraph;
    std::vector<std::string> tensorNames{"t1", "t2", "t3", "t4", "t5", "t6", "t7", "t8", "t9"};
    std::vector<MemoryType> tensorMemTypes{MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_UB,
        MemoryType::MEM_UB, MemoryType::MEM_UB, MemoryType::MEM_UB, MemoryType::MEM_UB, MemoryType::MEM_UB, MemoryType::MEM_UB};
    std::vector<Opcode> opCodes{Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC,
        Opcode::OP_COPY_IN, Opcode::OP_COPY_IN, Opcode::OP_VIEW, Opcode::OP_VIEW, Opcode::OP_ADD, Opcode::OP_ADD, Opcode::OP_ADD};
    std::vector<std::vector<std::string>> ioperands{{}, {}, {}, {}, {}, {"t1"}, {"t2"}, {"t3"}, {"t3"}, {"t5"}, {"t6"}, {"t4", "t7"}};
    std::vector<std::vector<std::string>> ooperands{{"t3"}, {"t4"}, {"t7"}, {"t8"}, {"t9"}, {"t3"}, {"t4"}, {"t5"}, {"t6"}, {"t7"}, {"t8"}, {"t9"}};
    std::vector<std::string> opNames{"Alloc1", "Alloc2", "Alloc3", "Alloc4", "Alloc5", "Copyin1", "Copyin2", "View1", "View2", "Add1", "Add2", "Add3"};
    EXPECT_EQ(subGraph.AddTensors(DataType::DT_FP32, {64, 64}, tensorMemTypes, tensorNames, 0), true);
    EXPECT_EQ(subGraph.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    Function *function = subGraph.GetFunction();
    EXPECT_NE(function, nullptr);

    EXPECT_NE(subGraph.GetTensor("t6"), nullptr);
    std::shared_ptr<LogicalTensor> tensor1 = subGraph.GetTensor("t5");
    tensor1->memoryrange.memId =
        subGraph.GetTensor("t3")->memoryrange.memId;
    tensor1->shape = {32, 32};
    tensor1->tensor->rawshape = {32, 32};
    std::shared_ptr<LogicalTensor> tensor2 = subGraph.GetTensor("t6");
    tensor2->memoryrange.memId =
        subGraph.GetTensor("t3")->memoryrange.memId;
    tensor2->shape = {32, 32};
    tensor2->tensor->rawshape = {32, 32};

    OoOScheduler ooOScheduler(*function);
    Status res = ooOScheduler.Schedule(function->Operations().DuplicatedOpList());
    EXPECT_EQ(res, SUCCESS);
    IssueEntryPtr copyin = GetIssueEntry("Copyin1", subGraph, ooOScheduler);
    EXPECT_NE(copyin, nullptr);
    EXPECT_EQ(copyin->tileOp.oOperand[0]->memoryrange.start, 0);
    EXPECT_EQ(copyin->tileOp.oOperand[0]->memoryrange.end, 16384);
    EXPECT_EQ(subGraph.GetOp("View1")->GetOutputOperand(0)->memoryrange.start, 0);
    EXPECT_EQ(subGraph.GetOp("View1")->GetOutputOperand(0)->memoryrange.end, 16384);
    EXPECT_EQ(subGraph.GetOp("View2")->GetOutputOperand(0)->memoryrange.start, 0);
    EXPECT_EQ(subGraph.GetOp("View2")->GetOutputOperand(0)->memoryrange.end, 16384);
}

TEST_F(ScheduleOoOTest, TestScheduleAssemble) {
    ComputationalGraphBuilder subGraph;
    std::vector<std::string> tensorNames{"t1", "t2", "t3", "t4", "t5", "t6", "t7", "t8", "t9", "t10", "t11"};
    std::vector<MemoryType> tensorMemTypes{MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_DEVICE_DDR,
        MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_UB, MemoryType::MEM_UB, MemoryType::MEM_UB, MemoryType::MEM_UB, MemoryType::MEM_UB,
        MemoryType::MEM_UB, MemoryType::MEM_UB};
    std::vector<Opcode> opCodes{Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_COPY_IN,
        Opcode::OP_COPY_IN, Opcode::OP_COPY_IN, Opcode::OP_COPY_IN, Opcode::OP_ASSEMBLE, Opcode::OP_ASSEMBLE, Opcode::OP_ADD, Opcode::OP_ADD};
    std::vector<std::vector<std::string>> ioperands{{}, {}, {}, {}, {}, {"t1"}, {"t2"}, {"t3"}, {"t4"}, {"t5"}, {"t6"}, {"t7", "t8"}, {"t9", "t10"}};
    std::vector<std::vector<std::string>> ooperands{{"t5"}, {"t7"}, {"t8"}, {"t10"}, {"t11"}, {"t5"}, {"t6"}, {"t7"}, {"t8"}, {"t9"}, {"t9"}, {"t10"}, {"t11"}};
    std::vector<std::string> opNames{"Alloc1", "Alloc2", "Alloc3", "Alloc4", "Alloc5", "Copyin1", "Copyin2", "Copyin3", "Copyin4", "Assemble1", "Assemble2", "Add1", "Add2"};
    EXPECT_EQ(subGraph.AddTensors(DataType::DT_FP32, {64, 64}, tensorMemTypes, tensorNames, 0), true);
    EXPECT_EQ(subGraph.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    Function *function = subGraph.GetFunction();
    EXPECT_NE(function, nullptr);

    EXPECT_NE(subGraph.GetTensor("t9"), nullptr);
    std::shared_ptr<LogicalTensor> tensor1 = subGraph.GetTensor("t9");
    tensor1->memoryrange.memId =
        subGraph.GetTensor("t5")->memoryrange.memId;
    std::shared_ptr<LogicalTensor> tensor2 = subGraph.GetTensor("t6");
    tensor2->memoryrange.memId =
        subGraph.GetTensor("t5")->memoryrange.memId;
    tensor2->shape = {32, 32};
    tensor2->tensor->rawshape = {32, 32};
    std::shared_ptr<LogicalTensor> tensor3 = subGraph.GetTensor("t5");
    tensor3->shape = {32, 32};
    tensor3->tensor->rawshape = {32, 32};

    OoOScheduler ooOScheduler(*function);
    Status res = ooOScheduler.Schedule(function->Operations().DuplicatedOpList());
    EXPECT_EQ(res, SUCCESS);
    IssueEntryPtr copyin1 = GetIssueEntry("Copyin1", subGraph, ooOScheduler);
    EXPECT_NE(copyin1, nullptr);
    IssueEntryPtr copyin2 = GetIssueEntry("Copyin2", subGraph, ooOScheduler);
    EXPECT_NE(copyin2, nullptr);
    IssueEntryPtr assemble1 = GetIssueEntry("Assemble1", subGraph, ooOScheduler);
    EXPECT_NE(assemble1, nullptr);
    IssueEntryPtr assemble2 = GetIssueEntry("Assemble2", subGraph, ooOScheduler);
    EXPECT_NE(assemble2, nullptr);
    EXPECT_EQ(copyin1->tileOp.oOperand[0]->memoryrange.start, 32768);
    EXPECT_EQ(copyin1->tileOp.oOperand[0]->memoryrange.end, 49152);
    EXPECT_EQ(copyin2->tileOp.oOperand[0]->memoryrange.start, 32768);
    EXPECT_EQ(copyin2->tileOp.oOperand[0]->memoryrange.end, 49152);
    EXPECT_EQ(assemble1->tileOp.oOperand[0]->memoryrange.start, 32768);
    EXPECT_EQ(assemble1->tileOp.oOperand[0]->memoryrange.end, 49152);
    EXPECT_EQ(assemble2->tileOp.oOperand[0]->memoryrange.start, 32768);
    EXPECT_EQ(assemble2->tileOp.oOperand[0]->memoryrange.end, 49152);
}

TEST_F(ScheduleOoOTest, TestScheduleSpillCopyIn) {
    ComputationalGraphBuilder subGraph;
    std::vector<std::string> tensorNames{"t1", "t2", "t3", "t4", "t5", "t6", "t7", "t8"};
    std::vector<MemoryType> tensorMemTypes{MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_UB,
        MemoryType::MEM_UB, MemoryType::MEM_UB, MemoryType::MEM_UB, MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_UB};
    std::vector<Opcode> opCodes{Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC,
        Opcode::OP_COPY_IN, Opcode::OP_COPY_IN, Opcode::OP_ADD, Opcode::OP_ADD, Opcode::OP_ADD, Opcode::OP_COPY_OUT};
    std::vector<std::vector<std::string>> ioperands{{}, {}, {}, {}, {}, {"t1"}, {"t2"}, {"t3", "t4"}, {"t3", "t5"}, {"t4", "t6"}, {"t8"}};
    std::vector<std::vector<std::string>> ooperands{{"t3"}, {"t4"}, {"t5"}, {"t6"}, {"t8"}, {"t3"}, {"t4"}, {"t5"}, {"t6"}, {"t8"}, {"t7"}};
    std::vector<std::string> opNames{"Alloc1", "Alloc2", "Alloc3", "Alloc4", "Alloc5", "Copyin1", "Copyin2", "Add1", "Add2", "Add3", "Copyout1"};
    EXPECT_EQ(subGraph.AddTensors(DataType::DT_FP32, {128, 128}, tensorMemTypes, tensorNames, 0), true);
    EXPECT_EQ(subGraph.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    Function *function = subGraph.GetFunction();
    EXPECT_NE(function, nullptr);

    OoOScheduler ooOScheduler(*function);
    Status res = ooOScheduler.Init(function->Operations().DuplicatedOpList());
    EXPECT_EQ(res, SUCCESS);
    res = ooOScheduler.SortOps();
    EXPECT_EQ(res, SUCCESS);
    res = ooOScheduler.ScheduleMainLoop();
    EXPECT_EQ(res, SUCCESS);
    EXPECT_EQ(ooOScheduler.newOperations_[9]->GetOpcodeStr(), "UB_ALLOC");
    EXPECT_EQ(ooOScheduler.newOperations_[9]->oOperand[0]->memoryrange.start, 131072);
    EXPECT_EQ(ooOScheduler.newOperations_[9]->oOperand[0]->memoryrange.end, 196608);
    EXPECT_EQ(ooOScheduler.newOperations_[10]->GetOpcodeStr(), "COPY_IN");
    EXPECT_EQ(ooOScheduler.newOperations_[10]->oOperand[0]->memoryrange.start, 131072);
    EXPECT_EQ(ooOScheduler.newOperations_[10]->oOperand[0]->memoryrange.end, 196608);
}

TEST_F(ScheduleOoOTest, TestScheduleSpill) {
    ComputationalGraphBuilder subGraph;
    std::vector<std::string> tensorNames{"t1", "t2", "t3", "t4", "t5", "t6", "t7", "t8", "t9"};
    std::vector<MemoryType> tensorMemTypes{MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_UB,
        MemoryType::MEM_UB, MemoryType::MEM_UB, MemoryType::MEM_UB, MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_UB, MemoryType::MEM_UB};
    std::vector<Opcode> opCodes{Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC,
        Opcode::OP_COPY_IN, Opcode::OP_COPY_IN, Opcode::OP_ADD, Opcode::OP_ADD, Opcode::OP_ADD, Opcode::OP_ADD, Opcode::OP_COPY_OUT};
    std::vector<std::vector<std::string>> ioperands{{}, {}, {}, {}, {}, {}, {"t1"}, {"t2"}, {"t3", "t4"}, {"t3", "t4"}, {"t4", "t6"}, {"t5", "t8"}, {"t9"}};
    std::vector<std::vector<std::string>> ooperands{{"t3"}, {"t4"}, {"t5"}, {"t6"}, {"t8"}, {"t9"}, {"t3"}, {"t4"}, {"t5"}, {"t6"}, {"t8"}, {"t9"}, {"t7"}};
    std::vector<std::string> opNames{"Alloc1", "Alloc2", "Alloc3", "Alloc4", "Alloc5", "Alloc6", "Copyin1", "Copyin2", "Add1", "Add2", "Add3", "Add4", "Copyout1"};
    EXPECT_EQ(subGraph.AddTensors(DataType::DT_FP32, {128, 128}, tensorMemTypes, tensorNames, 0), true);
    EXPECT_EQ(subGraph.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    Function *function = subGraph.GetFunction();
    EXPECT_NE(function, nullptr);

    OoOScheduler ooOScheduler(*function);
    Status res = ooOScheduler.Init(function->Operations().DuplicatedOpList());
    EXPECT_EQ(res, SUCCESS);
    res = ooOScheduler.SortOps();
    EXPECT_EQ(res, SUCCESS);
    res = ooOScheduler.ScheduleMainLoop();
    EXPECT_EQ(res, SUCCESS);
    EXPECT_EQ(ooOScheduler.newOperations_[9]->GetOpcodeStr(), "COPY_OUT");
    EXPECT_EQ(ooOScheduler.newOperations_[9]->oOperand[0]->memoryrange.start, 0);
    EXPECT_EQ(ooOScheduler.newOperations_[9]->oOperand[0]->memoryrange.end, 65536);
    EXPECT_EQ(ooOScheduler.newOperations_[14]->GetOpcodeStr(), "UB_ALLOC");
    EXPECT_EQ(ooOScheduler.newOperations_[14]->oOperand[0]->memoryrange.start, 65536);
    EXPECT_EQ(ooOScheduler.newOperations_[14]->oOperand[0]->memoryrange.end, 131072);
    EXPECT_EQ(ooOScheduler.newOperations_[15]->GetOpcodeStr(), "COPY_IN");
    EXPECT_EQ(ooOScheduler.newOperations_[15]->oOperand[0]->memoryrange.start, 65536);
    EXPECT_EQ(ooOScheduler.newOperations_[15]->oOperand[0]->memoryrange.end, 131072);
}

TEST_F(ScheduleOoOTest, TestScheduleSpillInplace) {
    ComputationalGraphBuilder subGraph;
    std::vector<std::string> tensorNames{"t1", "t2", "t3", "t4", "t5", "t6", "t7", "t8", "t9", "t10", "t11"};
    std::vector<MemoryType> tensorMemTypes{MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_DEVICE_DDR,
        MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_UB, MemoryType::MEM_UB, MemoryType::MEM_UB, MemoryType::MEM_UB, MemoryType::MEM_UB,
        MemoryType::MEM_UB, MemoryType::MEM_UB};
    std::vector<Opcode> opCodes{Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC,
        Opcode::OP_COPY_IN, Opcode::OP_COPY_IN, Opcode::OP_COPY_IN, Opcode::OP_COPY_IN, Opcode::OP_ADD, Opcode::OP_ADD, Opcode::OP_ADD};
    std::vector<std::vector<std::string>> ioperands{{}, {}, {}, {}, {}, {"t1"}, {"t2"}, {"t3"}, {"t4"}, {"t5", "t6"}, {"t7", "t8"}, {"t9", "t10"}};
    std::vector<std::vector<std::string>> ooperands{{"t5"}, {"t6"}, {"t7"}, {"t8"}, {"t9"}, {"t5"}, {"t6"}, {"t7"}, {"t8"}, {"t10"}, {"t9"}, {"t11"}};
    std::vector<std::string> opNames{"Alloc1", "Alloc2", "Alloc3", "Alloc4", "Alloc5", "Copyin1", "Copyin2", "Copyin3", "Copyin4", "Add1", "Add2", "Add3"};
    EXPECT_EQ(subGraph.AddTensors(DataType::DT_FP32, {128, 128}, tensorMemTypes, tensorNames, 0), true);
    EXPECT_EQ(subGraph.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    Function *function = subGraph.GetFunction();
    EXPECT_NE(function, nullptr);

    EXPECT_NE(subGraph.GetTensor("t11"), nullptr);
    std::shared_ptr<LogicalTensor> tensor1 = subGraph.GetTensor("t10");
    tensor1->memoryrange.memId =
        subGraph.GetTensor("t5")->memoryrange.memId;
    std::shared_ptr<LogicalTensor> tensor2 = subGraph.GetTensor("t11");
    tensor2->memoryrange.memId =
        subGraph.GetTensor("t5")->memoryrange.memId;

    OoOScheduler ooOScheduler(*function);
    Status res = ooOScheduler.Init(function->Operations().DuplicatedOpList());
    EXPECT_EQ(res, SUCCESS);
    res = ooOScheduler.SortOps();
    EXPECT_EQ(res, SUCCESS);
    std::rotate(ooOScheduler.issueEntries.begin(), ooOScheduler.issueEntries.begin() + 6, ooOScheduler.issueEntries.begin() + 11);
    res = ooOScheduler.ScheduleMainLoop();
    EXPECT_EQ(res, SUCCESS);
    EXPECT_EQ(ooOScheduler.newOperations_[9]->GetOpcodeStr(), "COPY_OUT");
    EXPECT_EQ(ooOScheduler.newOperations_[13]->GetOpcodeStr(), "COPY_IN");
    EXPECT_EQ(ooOScheduler.newOperations_[13]->oOperand[0]->memoryrange.start, 65536);
    EXPECT_EQ(ooOScheduler.newOperations_[13]->oOperand[0]->memoryrange.end, 131072);
    EXPECT_EQ(ooOScheduler.newOperations_[14]->GetOpcodeStr(), "ADD");
    EXPECT_EQ(ooOScheduler.newOperations_[14]->oOperand[0]->memoryrange.start, 65536);
    EXPECT_EQ(ooOScheduler.newOperations_[14]->oOperand[0]->memoryrange.end, 131072);
}

TEST_F(ScheduleOoOTest, TestScheduleSpillView) {
    ComputationalGraphBuilder subGraph;
    std::vector<std::string> tensorNames{"t1", "t2", "t3", "t4", "t5", "t6", "t7", "t8", "t9"};
    std::vector<MemoryType> tensorMemTypes{MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_UB,
        MemoryType::MEM_UB, MemoryType::MEM_UB, MemoryType::MEM_UB, MemoryType::MEM_UB, MemoryType::MEM_UB, MemoryType::MEM_UB};
    std::vector<Opcode> opCodes{Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC,
        Opcode::OP_COPY_IN, Opcode::OP_COPY_IN, Opcode::OP_VIEW, Opcode::OP_VIEW, Opcode::OP_ADD, Opcode::OP_ADD, Opcode::OP_ADD};
    std::vector<std::vector<std::string>> ioperands{{}, {}, {}, {}, {}, {"t1"}, {"t2"}, {"t3"}, {"t3"}, {"t5"}, {"t6"}, {"t4", "t7"}};
    std::vector<std::vector<std::string>> ooperands{{"t3"}, {"t4"}, {"t7"}, {"t8"}, {"t9"}, {"t3"}, {"t4"}, {"t5"}, {"t6"}, {"t7"}, {"t8"}, {"t9"}};
    std::vector<std::string> opNames{"Alloc1", "Alloc2", "Alloc3", "Alloc4", "Alloc5", "Copyin1", "Copyin2", "View1", "View2", "Add1", "Add2", "Add3"};
    EXPECT_EQ(subGraph.AddTensors(DataType::DT_FP32, {128, 128}, tensorMemTypes, tensorNames, 0), true);
    EXPECT_EQ(subGraph.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    Function *function = subGraph.GetFunction();
    EXPECT_NE(function, nullptr);

    EXPECT_NE(subGraph.GetTensor("t6"), nullptr);
    std::shared_ptr<LogicalTensor> tensor1 = subGraph.GetTensor("t5");
    tensor1->memoryrange.memId =
        subGraph.GetTensor("t3")->memoryrange.memId;
    std::shared_ptr<LogicalTensor> tensor2 = subGraph.GetTensor("t6");
    tensor2->memoryrange.memId =
        subGraph.GetTensor("t3")->memoryrange.memId;

    OoOScheduler ooOScheduler(*function);
    Status res = ooOScheduler.Init(function->Operations().DuplicatedOpList());
    EXPECT_EQ(res, SUCCESS);
    res = ooOScheduler.SortOps();
    EXPECT_EQ(res, SUCCESS);
    res = ooOScheduler.ScheduleMainLoop();
    EXPECT_EQ(res, SUCCESS);
}

TEST_F(ScheduleOoOTest, TestScheduleSpillAssemble) {
    ComputationalGraphBuilder subGraph;
    std::vector<std::string> tensorNames{"t1", "t2", "t3", "t4", "t5", "t6", "t7", "t8", "t9", "t10", "t11"};
    std::vector<MemoryType> tensorMemTypes{MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_DEVICE_DDR,
        MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_UB, MemoryType::MEM_UB, MemoryType::MEM_UB, MemoryType::MEM_UB, MemoryType::MEM_UB,
        MemoryType::MEM_UB, MemoryType::MEM_UB};
    std::vector<Opcode> opCodes{Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_COPY_IN,
        Opcode::OP_COPY_IN, Opcode::OP_COPY_IN, Opcode::OP_COPY_IN, Opcode::OP_ASSEMBLE, Opcode::OP_ASSEMBLE, Opcode::OP_ADD, Opcode::OP_ADD};
    std::vector<std::vector<std::string>> ioperands{{}, {}, {}, {}, {}, {"t1"}, {"t2"}, {"t3"}, {"t4"}, {"t5"}, {"t6"}, {"t7", "t8"}, {"t9", "t10"}};
    std::vector<std::vector<std::string>> ooperands{{"t5"}, {"t7"}, {"t8"}, {"t10"}, {"t11"}, {"t5"}, {"t6"}, {"t7"}, {"t8"}, {"t9"}, {"t9"}, {"t10"}, {"t11"}};
    std::vector<std::string> opNames{"Alloc1", "Alloc2", "Alloc3", "Alloc4", "Alloc5", "Copyin1", "Copyin2", "Copyin3", "Copyin4", "Assemble1", "Assemble2", "Add1", "Add2"};
    EXPECT_EQ(subGraph.AddTensors(DataType::DT_FP32, {128, 128}, tensorMemTypes, tensorNames, 0), true);
    EXPECT_EQ(subGraph.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    Function *function = subGraph.GetFunction();
    EXPECT_NE(function, nullptr);

    EXPECT_NE(subGraph.GetTensor("t9"), nullptr);
    std::shared_ptr<LogicalTensor> tensor1 = subGraph.GetTensor("t9");
    tensor1->memoryrange.memId =
        subGraph.GetTensor("t5")->memoryrange.memId;
    std::shared_ptr<LogicalTensor> tensor2 = subGraph.GetTensor("t6");
    tensor2->memoryrange.memId =
        subGraph.GetTensor("t5")->memoryrange.memId;
    std::shared_ptr<LogicalTensor> tensor5 = subGraph.GetTensor("t5");
    std::shared_ptr<LogicalTensor> tensor6 = subGraph.GetTensor("t6");
    tensor5->shape = {64, 128};
    tensor5->tensor->rawshape = {64, 128};
    tensor6->shape = {64, 128};
    tensor6->tensor->rawshape = {64, 128};
    std::vector<int64_t> offset1 = {0, 0};
    std::vector<int64_t> offset2 = {64, 0};
    auto assembleAttr1 = std::make_shared<AssembleOpAttribute>(MemoryType::MEM_UB, offset1);
    auto assemble1 = subGraph.GetOp("Assemble1");
    assemble1->SetOpAttribute(assembleAttr1);
    auto assembleAttr2 = std::make_shared<AssembleOpAttribute>(MemoryType::MEM_UB, offset2);
    auto assemble2 = subGraph.GetOp("Assemble2");
    assemble2->SetOpAttribute(assembleAttr2);

    OoOScheduler ooOScheduler(*function);
    Status res = ooOScheduler.Init(function->Operations().DuplicatedOpList());
    EXPECT_EQ(res, SUCCESS);
    res = ooOScheduler.SortOps();
    EXPECT_EQ(res, SUCCESS);
    res = ooOScheduler.ScheduleMainLoop();
    // 待ScheduleMainloop支持assemble
    EXPECT_EQ(res, FAILED);
}

TEST_F(ScheduleOoOTest, TestScheduleSpillFragFailed) {
    ComputationalGraphBuilder subGraph;
    std::vector<std::string> tensorNames{"t1", "t2", "t3", "t4", "t5", "t6"};
    std::vector<MemoryType> tensorMemTypes{MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_UB,
        MemoryType::MEM_UB, MemoryType::MEM_UB, MemoryType::MEM_UB};
    std::vector<Opcode> opCodes{Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_COPY_IN,
        Opcode::OP_COPY_IN, Opcode::OP_ADD};
    std::vector<std::vector<std::string>> ioperands{{}, {}, {}, {}, {"t1"}, {"t2"}, {"t4", "t5"}};
    std::vector<std::vector<std::string>> ooperands{{"t3"}, {"t4"}, {"t5"}, {"t6"}, {"t3", "t4"}, {"t5"}, {"t6"}};
    std::vector<std::string> opNames{"Alloc1", "Alloc2", "Alloc3", "Alloc4", "Copyin1", "Copyin2", "Add"};
    EXPECT_EQ(subGraph.AddTensors(DataType::DT_FP32, {128, 128}, tensorMemTypes, tensorNames, 0), true);
    EXPECT_EQ(subGraph.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    Function *function = subGraph.GetFunction();
    EXPECT_NE(function, nullptr);

    EXPECT_NE(subGraph.GetTensor("t3"), nullptr);
    std::shared_ptr<LogicalTensor> tensor = subGraph.GetTensor("t3");
    tensor->shape = {32, 32};
    tensor->tensor->rawshape = {32, 32};

    OoOScheduler ooOScheduler(*function);
    Status res = ooOScheduler.Init(function->Operations().DuplicatedOpList());
    EXPECT_EQ(res, SUCCESS);
    res = ooOScheduler.SortOps();
    EXPECT_EQ(res, SUCCESS);
    res = ooOScheduler.ScheduleMainLoop();
    EXPECT_EQ(res, SUCCESS);
}

TEST_F(ScheduleOoOTest, TestEmptyOplist) {
    Function function(Program::GetInstance(), "", "", nullptr);
    std::vector<Operation *> scheduleOpList;
    OoOScheduler ooOScheduler(function);
    Status res = ooOScheduler.Schedule(scheduleOpList);
    EXPECT_EQ(res, SUCCESS);
}

TEST_F(ScheduleOoOTest, TestScheduleReshape) {
    ComputationalGraphBuilder subGraph;
    std::vector<std::string> tensorNames{"t1", "t2"};
    std::vector<MemoryType> tensorMemTypes{MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_DEVICE_DDR};
    std::vector<Opcode> opCodes{Opcode::OP_RESHAPE};
    std::vector<std::vector<std::string>> ioperands{{"t1"}};
    std::vector<std::vector<std::string>> ooperands{{"t2"}};
    std::vector<std::string> opNames{"Reshape1"};
    EXPECT_EQ(subGraph.AddTensors(DataType::DT_FP32, {128, 128}, tensorMemTypes, tensorNames, 0), true);
    EXPECT_EQ(subGraph.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    Function *function = subGraph.GetFunction();
    EXPECT_NE(function, nullptr);

    OoOScheduler ooOScheduler(*function);
    Status res = ooOScheduler.Schedule(function->Operations().DuplicatedOpList());
    EXPECT_EQ(res, SUCCESS);
}

TEST_F(ScheduleOoOTest, TestSingleCopyin1) {
    ComputationalGraphBuilder subGraph;
    std::vector<std::string> tensorNames{"t1", "t2"};
    std::vector<MemoryType> tensorMemTypes{MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_UB};
    std::vector<Opcode> opCodes{Opcode::OP_COPY_IN};
    std::vector<std::vector<std::string>> ioperands{{"t1"}};
    std::vector<std::vector<std::string>> ooperands{{"t2"}};
    std::vector<std::string> opNames{"Copyin1"};
    EXPECT_EQ(subGraph.AddTensors(DataType::DT_FP32, {128, 128}, tensorMemTypes, tensorNames, 0), true);
    EXPECT_EQ(subGraph.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    Function *function = subGraph.GetFunction();
    EXPECT_NE(function, nullptr);

    OoOScheduler ooOScheduler(*function);
    Status res = ooOScheduler.Schedule(function->Operations().DuplicatedOpList());
    EXPECT_EQ(res, FAILED);
}

TEST_F(ScheduleOoOTest, TestSingleCopyin2) {
    ComputationalGraphBuilder subGraph;
    std::vector<std::string> tensorNames{"t1", "t2"};
    std::vector<MemoryType> tensorMemTypes{MemoryType::MEM_UB, MemoryType::MEM_UB};
    std::vector<Opcode> opCodes{Opcode::OP_UB_ALLOC, Opcode::OP_COPY_IN};
    std::vector<std::vector<std::string>> ioperands{{}, {"t1"}};
    std::vector<std::vector<std::string>> ooperands{{"t2"}, {"t2"}};
    std::vector<std::string> opNames{"Alloc1", "Copyin1"};
    EXPECT_EQ(subGraph.AddTensors(DataType::DT_FP32, {128, 128}, tensorMemTypes, tensorNames, 0), true);
    EXPECT_EQ(subGraph.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    Function *function = subGraph.GetFunction();
    EXPECT_NE(function, nullptr);

    OoOScheduler ooOScheduler(*function);
    Status res = ooOScheduler.Schedule(function->Operations().DuplicatedOpList());
    EXPECT_EQ(res, SUCCESS);
}

TEST_F(ScheduleOoOTest, TestDelBufCount) {
    Function function(Program::GetInstance(), "", "", nullptr);
    OoOScheduler oooSchedule(function);
    oooSchedule.DelBufRefCount(-1);
}

TEST_F(ScheduleOoOTest, TestDelBufCount_1) {
    Function function(Program::GetInstance(), "", "", nullptr);
    OoOScheduler oooSchedule(function);
    oooSchedule.bufRefCount[1] = -1;
    oooSchedule.DelBufRefCount(1);
}

TEST_F(ScheduleOoOTest, TestUpdateTensorAttr_DDR) {
    Function function(Program::GetInstance(), "", "", nullptr);
    std::vector<int64_t> shape = {128, 128};
    std::shared_ptr<LogicalTensor> tensor1 = std::make_shared<LogicalTensor>(function, DataType::DT_FP32, shape);
    tensor1->SetMemoryTypeOriginal(MEM_DEVICE_DDR);
    tensor1->SetMemoryTypeToBe(MEM_DEVICE_DDR);
    tensor1->memoryrange.memId = 1;

    std::shared_ptr<LogicalTensor> tensor3 = std::make_shared<LogicalTensor>(function, DataType::DT_FP32, shape);
    tensor3->SetMemoryTypeOriginal(MEM_UB);
    tensor3->SetMemoryTypeToBe(MEM_UB);
    tensor3->memoryrange.memId = 3;

    OoOScheduler oooSchedule(function);
    oooSchedule.UpdateTensorAttr(tensor1, MemoryType::MEM_DEVICE_DDR, tensor3, -1);
}

TEST_F(ScheduleOoOTest, TestUpdateTensorAttr_UB) {
    Function function(Program::GetInstance(), "", "", nullptr);
    std::vector<int64_t> shape = {128, 128};
    std::shared_ptr<LogicalTensor> tensor1 = std::make_shared<LogicalTensor>(function, DataType::DT_FP32, shape);
    tensor1->SetMemoryTypeOriginal(MEM_DEVICE_DDR);
    tensor1->SetMemoryTypeToBe(MEM_DEVICE_DDR);
    tensor1->memoryrange.memId = 1;

    std::shared_ptr<LogicalTensor> tensor3 = std::make_shared<LogicalTensor>(function, DataType::DT_FP32, shape);
    tensor3->SetMemoryTypeOriginal(MEM_UB);
    tensor3->SetMemoryTypeToBe(MEM_UB);
    tensor3->memoryrange.memId = 3;

    OoOScheduler oooSchedule(function);
    oooSchedule.UpdateTensorAttr(tensor1, MemoryType::MEM_UB, tensor3, -1);
}

TEST_F(ScheduleOoOTest, TestGetSpillTensor) {
    Function function(Program::GetInstance(), "", "", nullptr);
    std::vector<Operation *> scheduleOpList;

    std::vector<int64_t> shape = {128, 128};
    std::shared_ptr<LogicalTensor> tensor3 = std::make_shared<LogicalTensor>(function, DataType::DT_FP32, shape);
    tensor3->SetMemoryTypeOriginal(MEM_UB);
    tensor3->SetMemoryTypeToBe(MEM_UB);
    tensor3->memoryrange.memId = 3;

    auto &alloc1 = function.AddOperation(Opcode::OP_UB_ALLOC, {}, std::vector<std::shared_ptr<LogicalTensor>>({tensor3}));
    alloc1.UpdateLatency(1);

    LogicalTensorPtr tensor = nullptr;
    auto allocIssue = std::make_shared<IssueEntry>(alloc1, 1);

    OoOScheduler oooSchedule(function);
    oooSchedule.GetSpillTensor(allocIssue, 1, tensor);
}

TEST_F(ScheduleOoOTest, TestCheckAllocIssue) {
    Function function(Program::GetInstance(), "", "", nullptr);
    std::vector<Operation *> scheduleOpList;

    std::vector<int64_t> shape = {128, 128};
    std::shared_ptr<LogicalTensor> tensor3 = std::make_shared<LogicalTensor>(function, DataType::DT_FP32, shape);
    tensor3->SetMemoryTypeOriginal(MEM_UB);
    tensor3->SetMemoryTypeToBe(MEM_UB);
    tensor3->memoryrange.memId = 3;

    std::shared_ptr<LogicalTensor> tensor2 = std::make_shared<LogicalTensor>(function, DataType::DT_FP32, shape);
    tensor3->SetMemoryTypeOriginal(MEM_UB);
    tensor3->SetMemoryTypeToBe(MEM_UB);
    tensor3->memoryrange.memId = 1;

    auto &alloc1 = function.AddOperation(Opcode::OP_UB_ALLOC, {}, std::vector<std::shared_ptr<LogicalTensor>>({tensor3, tensor2}));
    alloc1.UpdateLatency(1);

    OoOScheduler oooSchedule(function);
    oooSchedule.Init(function.Operations().DuplicatedOpList());
}

TEST_F(ScheduleOoOTest, TestBufferUsage) {
    ComputationalGraphBuilder subGraph;
    std::vector<std::string> tensorNames{"t1", "t2", "t3", "t4", "t5", "t6"};
    std::vector<MemoryType> tensorMemTypes{MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_UB,
        MemoryType::MEM_UB, MemoryType::MEM_UB, MemoryType::MEM_UB};
    std::vector<Opcode> opCodes{Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_COPY_IN,
        Opcode::OP_COPY_IN, Opcode::OP_ADD};
    std::vector<std::vector<std::string>> ioperands{{}, {}, {}, {}, {"t1"}, {"t2"}, {"t4", "t5"}};
    std::vector<std::vector<std::string>> ooperands{{"t3"}, {"t4"}, {"t5"}, {"t6"}, {"t3", "t4"}, {"t5"}, {"t6"}};
    std::vector<std::string> opNames{"Alloc1", "Alloc2", "Alloc3", "Alloc4", "Copyin1", "Copyin2", "Add1"};
    EXPECT_EQ(subGraph.AddTensors(DataType::DT_FP32, {128, 128}, tensorMemTypes, tensorNames, 0), true);
    EXPECT_EQ(subGraph.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    Function *function = subGraph.GetFunction();
    EXPECT_NE(function, nullptr);

    OoOScheduler ooOScheduler(*function);
    ooOScheduler.oooCheck.doHealthCheck = true;
    Status res = ooOScheduler.Init(function->Operations().DuplicatedOpList());
    EXPECT_EQ(res, SUCCESS);
    res = ooOScheduler.SortOps();
    EXPECT_EQ(res, SUCCESS);
    res = ooOScheduler.ScheduleMainLoop();
    EXPECT_EQ(res, SUCCESS);
    std::unordered_map<MemoryType, uint64_t> invalidBufferTotalUsage = {
        {MemoryType::MEM_UB, 0},
        {MemoryType::MEM_L1, 0},
        {MemoryType::MEM_L0A, 0},
        {MemoryType::MEM_L0B, 0},
        {MemoryType::MEM_L0C, 0}
    };
    std::unordered_map<MemoryType, uint64_t> invalidBufferMaxUsage = {
        {MemoryType::MEM_UB, 0},
        {MemoryType::MEM_L1, 0},
        {MemoryType::MEM_L0A, 0},
        {MemoryType::MEM_L0B, 0},
        {MemoryType::MEM_L0C, 0}
    };
    EXPECT_NE(ooOScheduler.oooCheck.bufferTotalUsage, invalidBufferTotalUsage);
    EXPECT_NE(ooOScheduler.oooCheck.bufferMaxUsage, invalidBufferMaxUsage);

    // 增加健康检查校验
    ooOScheduler.oooCheck.clock = 3; // 模拟数据
    res = ooOScheduler.oooCheck.HealthCheckOoOSchedule();
    EXPECT_EQ(res, SUCCESS);
    EXPECT_NE(ooOScheduler.oooCheck.report, nullptr);
}

TEST_F(ScheduleOoOTest, TestScheduleGenSpillInfiniteLoop) {
    ComputationalGraphBuilder subGraph;
    std::vector<std::string> tensorNames{"t1", "t2", "t3", "t4", "t5", "t6", "t7"};
    std::vector<MemoryType> tensorMemTypes{MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_UB,
        MemoryType::MEM_UB, MemoryType::MEM_UB, MemoryType::MEM_UB, MemoryType::MEM_UB};
    std::vector<Opcode> opCodes{Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC,
        Opcode::OP_COPY_IN, Opcode::OP_COPY_IN, Opcode::OP_SUB, Opcode::OP_ADD, Opcode::OP_ADD};
    std::vector<std::vector<std::string>> ioperands{{}, {}, {}, {}, {}, {"t1"}, {"t2"}, {"t3"},{"t4", "t5"}, {"t3", "t6"}};
    std::vector<std::vector<std::string>> ooperands{{"t3"}, {"t4"}, {"t5"}, {"t6"}, {"t7"}, {"t3"}, {"t4"}, {"t5"}, {"t6"}, {"t7"}};
    std::vector<std::string> opNames{"Alloc1", "Alloc2", "Alloc3", "Alloc4", "Alloc5", "Copyin1", "Copyin2", "Sub1", "Add1", "Add2"};
    EXPECT_EQ(subGraph.AddTensors(DataType::DT_FP16, {128, 128}, tensorMemTypes, tensorNames, 0), true);
    EXPECT_EQ(subGraph.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    Function *function = subGraph.GetFunction();
    EXPECT_NE(function, nullptr);

    EXPECT_NE(subGraph.GetTensor("t3"), nullptr);
    std::shared_ptr<LogicalTensor> tensor = subGraph.GetTensor("t3");
    tensor->shape = {80,128};
    tensor->tensor->rawshape = {80,128};

    EXPECT_NE(subGraph.GetTensor("t4"), nullptr);
    std::shared_ptr<LogicalTensor> tensor1 = subGraph.GetTensor("t4");
    tensor1->shape = {176,256};
    tensor1->tensor->rawshape = {176,256};

    EXPECT_NE(subGraph.GetTensor("t5"), nullptr);
    std::shared_ptr<LogicalTensor> tensor2 = subGraph.GetTensor("t5");
    tensor2->shape = {176,256};
    tensor2->tensor->rawshape = {176,256};

    EXPECT_NE(subGraph.GetTensor("t6"), nullptr);
    std::shared_ptr<LogicalTensor> tensor3 = subGraph.GetTensor("t6");
    tensor3->shape = {64, 128};
    tensor3->tensor->rawshape = {64, 128};

    EXPECT_NE(subGraph.GetTensor("t7"), nullptr);
    std::shared_ptr<LogicalTensor> tensor4 = subGraph.GetTensor("t7");
    tensor4->shape = {16, 16};
    tensor4->tensor->rawshape = {16, 16};

    OoOScheduler ooOScheduler(*function);
    Status res = ooOScheduler.Init(function->Operations().DuplicatedOpList());
    EXPECT_EQ(res, SUCCESS);
    res = ooOScheduler.SortOps();
    EXPECT_EQ(res, SUCCESS);
    res = ooOScheduler.GenSpillSchedule();
    EXPECT_EQ(res, SUCCESS);
}

TEST_F(ScheduleOoOTest, TestCheckOpBufferSize) {
    ComputationalGraphBuilder subGraph;
    std::vector<std::string> tensorNames{"t1", "t2", "t3", "t4", "t5", "t6", "t7", "t8"};
    std::vector<MemoryType> tensorMemTypes{MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_UB, MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_UB,
        MemoryType::MEM_UB, MemoryType::MEM_UB, MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_UB};
    std::vector<Opcode> opCodes{Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC,
        Opcode::OP_UB_ALLOC, Opcode::OP_COPY_IN, Opcode::OP_COPY_IN, Opcode::OP_ROWMAX_SINGLE, Opcode::OP_ADD, Opcode::OP_COPY_OUT};
    std::vector<std::vector<std::string>> ioperands{{}, {}, {}, {}, {}, {"t1"}, {"t3"}, {"t2"}, {"t4", "t5"}, {"t5"}};
    std::vector<std::vector<std::string>> ooperands{{"t2"}, {"t4"}, {"t5"}, {"t6"}, {"t8"}, {"t2"}, {"t4"}, {"t5", "t6"}, {"t8"}, {"t7"}};
    std::vector<std::string> opNames{"Alloc1", "Alloc2", "Alloc3", "Alloc4", "Alloc5", "Copyin1", "Copyin2", "RowMax1", "Add1", "Copyout1"};
    EXPECT_EQ(subGraph.AddTensors(DataType::DT_FP32, {144, 128}, tensorMemTypes, tensorNames, 0), true);
    EXPECT_EQ(subGraph.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    Function *function = subGraph.GetFunction();
    EXPECT_NE(function, nullptr);

    OoOScheduler ooOScheduler(*function);
    Status res = ooOScheduler.Init(function->Operations().DuplicatedOpList());
    EXPECT_EQ(res, FAILED);
}

TEST_F(ScheduleOoOTest, TestCheckAllocBufferSize) {
    ComputationalGraphBuilder subGraph;
    std::vector<std::string> tensorNames{"t1", "t2", "t3", "t4", "t5", "t6", "t7", "t8"};
    std::vector<MemoryType> tensorMemTypes{MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_UB, MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_UB,
        MemoryType::MEM_UB, MemoryType::MEM_UB, MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_UB};
    std::vector<Opcode> opCodes{Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC,
        Opcode::OP_UB_ALLOC, Opcode::OP_COPY_IN, Opcode::OP_COPY_IN, Opcode::OP_ROWMAX_SINGLE, Opcode::OP_ADD, Opcode::OP_COPY_OUT};
    std::vector<std::vector<std::string>> ioperands{{}, {}, {}, {}, {}, {"t1"}, {"t3"}, {"t2"}, {"t4", "t5"}, {"t5"}};
    std::vector<std::vector<std::string>> ooperands{{"t2"}, {"t4"}, {"t5"}, {"t6"}, {"t8"}, {"t2"}, {"t4"}, {"t5", "t6"}, {"t8"}, {"t7"}};
    std::vector<std::string> opNames{"Alloc1", "Alloc2", "Alloc3", "Alloc4", "Alloc5", "Copyin1", "Copyin2", "RowMax1", "Add1", "Copyout1"};
    EXPECT_EQ(subGraph.AddTensors(DataType::DT_FP32, {256, 256}, tensorMemTypes, tensorNames, 0), true);
    EXPECT_EQ(subGraph.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    Function *function = subGraph.GetFunction();
    EXPECT_NE(function, nullptr);

    OoOScheduler ooOScheduler(*function);
    Status res = ooOScheduler.Init(function->Operations().DuplicatedOpList());
    EXPECT_EQ(res, FAILED);
}

TEST_F(ScheduleOoOTest, TestOoOMemoryRefactoring) {
    ComputationalGraphBuilder subGraph;
    std::vector<std::string> tensorNames{"t1", "t2", "t3", "t4", "t6", "t7", "t8", "t9", "t10", "t11", "t12", "t13", "t14"};
    std::vector<MemoryType> tensorMemTypes{MemoryType::MEM_L1, MemoryType::MEM_L1, MemoryType::MEM_L0A, MemoryType::MEM_L0B,
        MemoryType::MEM_L1, MemoryType::MEM_L1, MemoryType::MEM_L0A, MemoryType::MEM_L0B,
        MemoryType::MEM_L0C,  MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_L1, MemoryType::MEM_L0A};
    std::vector<std::string> tensorNames2{"t5"};
    std::vector<MemoryType> tensorMemTypes2{MemoryType::MEM_L0C};
    std::vector<Opcode> opCodes{Opcode::OP_L1_ALLOC, Opcode::OP_L1_ALLOC, Opcode::OP_L0A_ALLOC, Opcode::OP_L0B_ALLOC, Opcode::OP_L0C_ALLOC,
        Opcode::OP_L1_ALLOC, Opcode::OP_L1_ALLOC, Opcode::OP_L0A_ALLOC, Opcode::OP_L0B_ALLOC, Opcode::OP_L0C_ALLOC,
        Opcode::OP_L1_TO_L0A, Opcode::OP_L1_TO_L0B, Opcode::OP_A_MUL_B, Opcode::OP_L0C_TO_L1, Opcode::OP_L1_TO_L0A,
        Opcode::OP_L1_TO_L0B, Opcode::OP_A_MUL_B, Opcode::OP_L0C_COPY_OUT, Opcode::OP_L0C_COPY_OUT, Opcode::OP_L1_TO_L0A,
        Opcode::OP_L1_ALLOC, Opcode::OP_L0A_ALLOC};
    std::vector<std::vector<std::string>> ioperands{{}, {}, {}, {}, {}, {}, {}, {}, {}, {},
        {"t1"}, {"t2"}, {"t3", "t4"}, {"t5"}, {"t6"}, {"t7"}, {"t8", "t9"}, {"t10"}, {"t5", "t14"}, {"t13"},
        {}, {}};
    std::vector<std::vector<std::string>> ooperands{{"t1"}, {"t2"}, {"t3"}, {"t4"}, {"t5"}, {"t6"}, {"t7"}, {"t9"}, {"t8"}, {"t10"},
        {"t3"}, {"t4"}, {"t5"}, {"t6"}, {"t8"}, {"t9"}, {"t10"}, {"t11"}, {"t12"}, {"t14"},
        {"t13"}, {"t14"}};
    std::vector<std::string> opNames{"Alloc1", "Alloc2", "Alloc3", "Alloc4", "Alloc5", "Alloc6", "Alloc7", "Alloc8", "Alloc9", "Alloc10",
        "OP_L1_TO_L0A_1", "OP_L1_TO_L0B_1", "OP_A_MUL_B_1", "OP_L0C_TO_L1_1", "OP_L1_TO_L0A_2",
        "OP_L1_TO_L0B_2", "OP_A_MUL_B_2", "OP_L0C_COPY_OUT_1", "OP_L0C_COPY_OUT_2", "OP_L1_TO_L0A_3",
        "Alloc11", "Alloc12"
    };
    EXPECT_EQ(subGraph.AddTensors(DataType::DT_FP32, {16, 16}, tensorMemTypes, tensorNames, 0), true);
    EXPECT_EQ(subGraph.AddTensors(DataType::DT_FP32, {256, 128}, tensorMemTypes2, tensorNames2, 0), true);
    EXPECT_EQ(subGraph.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    Function *function = subGraph.GetFunction();
    EXPECT_NE(function, nullptr);

    OoOScheduler ooOScheduler(*function);
    Status res = ooOScheduler.Init(function->Operations().DuplicatedOpList());
    EXPECT_EQ(res, SUCCESS);

    res = ooOScheduler.PriorDFS(preNodePriority);
    EXPECT_EQ(res, SUCCESS);

    EXPECT_NE(subGraph.GetOp("OP_L0C_COPY_OUT_2"), nullptr);
    Operation *op = subGraph.GetOp("OP_L0C_COPY_OUT_2");
    int idx = 0;
    for (auto &issue : ooOScheduler.issueEntries) {
        if (&(issue->tileOp) == op) {
            break;
        }
        idx++;
    }
    std::rotate(ooOScheduler.issueEntries.begin() + idx, ooOScheduler.issueEntries.begin() + idx + 1, ooOScheduler.issueEntries.end());
    EXPECT_EQ(res, SUCCESS);

    res = ooOScheduler.ExecuteIssue();
    EXPECT_EQ(res, SUCCESS);
}

TEST_F(ScheduleOoOTest, TestOoORollback) {
    ComputationalGraphBuilder subGraph;
    std::vector<std::string> tensorNames{"t1", "t3", "t6", "t8", "t11", "t13", "t16", "t18", "t21", "DDR1", "DDR2", "DDR3", "DDR4", "DDR5", "DDR6", "DDR7"};
    std::vector<MemoryType> tensorMemTypes{MemoryType::MEM_L1, MemoryType::MEM_L1, MemoryType::MEM_L1, MemoryType::MEM_L1, MemoryType::MEM_L1,
    MemoryType::MEM_L1, MemoryType::MEM_L1, MemoryType::MEM_L1, MemoryType::MEM_L1, MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_DEVICE_DDR,
    MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_DEVICE_DDR};
    std::vector<std::string> tensorNames_L0AB{"t2", "t4", "t7", "t9", "t12", "t14", "t17", "t19"};
    std::vector<MemoryType> tensorMemTypes_L0AB{MemoryType::MEM_L0A, MemoryType::MEM_L0B, MemoryType::MEM_L0A, MemoryType::MEM_L0B,
        MemoryType::MEM_L0A, MemoryType::MEM_L0B, MemoryType::MEM_L0A, MemoryType::MEM_L0B};
    std::vector<std::string> tensorNames_L0C{"t5", "t10", "t15", "t20"};
    std::vector<MemoryType> tensorMemTypes_L0C{MemoryType::MEM_L0C, MemoryType::MEM_L0C, MemoryType::MEM_L0C, MemoryType::MEM_L0C};
    std::vector<Opcode> opCodes{Opcode::OP_L1_ALLOC, Opcode::OP_L1_ALLOC, Opcode::OP_L1_ALLOC, Opcode::OP_L1_ALLOC, Opcode::OP_L1_ALLOC, Opcode::OP_L1_ALLOC,
        Opcode::OP_L1_ALLOC, Opcode::OP_L1_ALLOC, Opcode::OP_L1_ALLOC, Opcode::OP_L0A_ALLOC, Opcode::OP_L0A_ALLOC, Opcode::OP_L0A_ALLOC,
        Opcode::OP_L0A_ALLOC, Opcode::OP_L0B_ALLOC, Opcode::OP_L0B_ALLOC, Opcode::OP_L0B_ALLOC, Opcode::OP_L0B_ALLOC, Opcode::OP_L0C_ALLOC,
        Opcode::OP_L0C_ALLOC, Opcode::OP_L0C_ALLOC, Opcode::OP_COPY_IN, Opcode::OP_COPY_IN, Opcode::OP_COPY_IN, Opcode::OP_COPY_IN,
        Opcode::OP_COPY_IN, Opcode::OP_COPY_IN, Opcode::OP_L1_TO_L0A, Opcode::OP_L1_TO_L0A, Opcode::OP_L1_TO_L0A, Opcode::OP_L1_TO_L0A,
        Opcode::OP_L1_TO_L0B, Opcode::OP_L1_TO_L0B, Opcode::OP_L1_TO_L0B, Opcode::OP_L1_TO_L0B, Opcode::OP_L0C_TO_L1, Opcode::OP_L0C_TO_L1,
        Opcode::OP_L0C_TO_L1, Opcode::OP_A_MUL_B, Opcode::OP_A_MUL_B, Opcode::OP_A_MUL_B, Opcode::OP_A_MULACC_B, Opcode::OP_COPY_OUT};
    std::vector<std::vector<std::string>> ioperands{{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {},
        {"DDR1"}, {"DDR2"}, {"DDR3"}, {"DDR4"}, {"DDR5"}, {"DDR6"}, {"t1"}, {"t6"}, {"t11"}, {"t16"}, {"t3"}, {"t8"}, {"t13"}, {"t18"},
        {"t5"}, {"t15"}, {"t20"}, {"t2", "t4"}, {"t7", "t9"}, {"t12", "t14"}, {"t10", "t17", "t19"}, {"t21"}};
    std::vector<std::vector<std::string>> ooperands{{"t1"}, {"t3"}, {"t6"}, {"t8"}, {"t11"}, {"t13"}, {"t16"}, {"t18"}, {"t21"}, {"t2"}, {"t7"},
        {"t12"}, {"t17"}, {"t4"}, {"t9"}, {"t14"}, {"t19"}, {"t5"}, {"t10"}, {"t15"},
        {"t1"}, {"t3"}, {"t8"}, {"t11"}, {"t13"}, {"t18"}, {"t2"}, {"t7"}, {"t12"}, {"t17"}, {"t4"}, {"t9"}, {"t14"}, {"t19"},
        {"t6"}, {"t16"}, {"t21"}, {"t5"}, {"t10"}, {"t15"}, {"t20"}, {"DDR7"}};
    std::vector<std::string> opNames{"L1_Alloc1", "L1_Alloc2", "L1_Alloc3", "L1_Alloc4", "L1_Alloc5", "L1_Alloc6",
        "L1_Alloc7", "L1_Alloc8", "L1_Alloc9", "L0A_Alloc1", "L0A_Alloc2", "L0A_Alloc3", "L0A_Alloc4",
        "L0B_Alloc1", "L0B_Alloc2", "L0B_Alloc3", "L0B_Alloc4", "L0C_Alloc1", "L0C_Alloc2", "L0C_Alloc3",
        "Copyin1", "Copyin2", "Copyin3", "Copyin4", "Copyin5", "Copyin6", "OP_L1_TO_L0A_1", "OP_L1_TO_L0A_2", "OP_L1_TO_L0A_3",
        "OP_L1_TO_L0A_4", "OP_L1_TO_L0B_1", "OP_L1_TO_L0B_2", "OP_L1_TO_L0B_3", "OP_L1_TO_L0B_4",
        "OP_L0C_TO_L1_1", "OP_L0C_TO_L1_2", "OP_L0C_TO_L1_3", "OP_A_MUL_B_1", "OP_A_MUL_B_2", "OP_A_MUL_B_3", "OP_A_MULACC_B", "Copyout"
    };
    EXPECT_EQ(subGraph.AddTensors(DataType::DT_FP32, {16, 16}, tensorMemTypes, tensorNames, 0), true);
    EXPECT_EQ(subGraph.AddTensors(DataType::DT_FP32, {128, 128}, tensorMemTypes_L0AB, tensorNames_L0AB, 0), true);
    EXPECT_EQ(subGraph.AddTensors(DataType::DT_FP32, {128, 256}, tensorMemTypes_L0C, tensorNames_L0C, 0), true);
    EXPECT_EQ(subGraph.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    Function *function = subGraph.GetFunction();
    std::shared_ptr<LogicalTensor> tensor1 = subGraph.GetTensor("t10");
    tensor1->memoryrange.memId =
        subGraph.GetTensor("t20")->memoryrange.memId;
    EXPECT_NE(function, nullptr);
    OoOScheduler ooOScheduler(*function);

    Status res = ooOScheduler.Init(function->Operations().DuplicatedOpList());
    EXPECT_EQ(res, SUCCESS);
    res = ooOScheduler.PriorDFS(preNodePriority);
    EXPECT_EQ(res, SUCCESS);
    res = ooOScheduler.ExecuteIssue();
    EXPECT_EQ(res, SUCCESS);
}

TEST_F(ScheduleOoOTest, TestOoORollbackMix) {
    ComputationalGraphBuilder subGraph;
    std::vector<std::string> tensorNames{"T3", "T1", "T6", "T8", "T11", "T13", "T16", "T18", "T21", "DDR1", "DDR2", "DDR3", "DDR4", "DDR5", "DDR6", "DDR7"};
    std::vector<MemoryType> tensorMemTypes{MemoryType::MEM_L1, MemoryType::MEM_L1, MemoryType::MEM_L1, MemoryType::MEM_L1, MemoryType::MEM_L1,
    MemoryType::MEM_L1, MemoryType::MEM_L1, MemoryType::MEM_L1, MemoryType::MEM_L1, MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_DEVICE_DDR,
    MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_DEVICE_DDR};
    std::vector<std::string> tensorNames_L0AB{"T4", "T2", "T7", "T9", "T12", "T14", "T19", "T17"};
    std::vector<MemoryType> tensorMemTypes_L0AB{MemoryType::MEM_L0B, MemoryType::MEM_L0A, MemoryType::MEM_L0A, MemoryType::MEM_L0B,
        MemoryType::MEM_L0A, MemoryType::MEM_L0B, MemoryType::MEM_L0B, MemoryType::MEM_L0A};
    std::vector<std::string> tensorNames_L0C{"T5", "T10", "T15", "T20"};
    std::vector<MemoryType> tensorMemTypes_L0C{MemoryType::MEM_L0C, MemoryType::MEM_L0C, MemoryType::MEM_L0C, MemoryType::MEM_L0C};
    std::vector<Opcode> opCodes{Opcode::OP_L1_ALLOC, Opcode::OP_L1_ALLOC, Opcode::OP_L1_ALLOC, Opcode::OP_L1_ALLOC, Opcode::OP_L1_ALLOC, Opcode::OP_L1_ALLOC,
        Opcode::OP_L1_ALLOC, Opcode::OP_L1_ALLOC, Opcode::OP_L0A_ALLOC, Opcode::OP_L1_ALLOC, Opcode::OP_L0A_ALLOC, Opcode::OP_L0A_ALLOC,
        Opcode::OP_L0A_ALLOC, Opcode::OP_L0B_ALLOC, Opcode::OP_L0B_ALLOC, Opcode::OP_L0B_ALLOC, Opcode::OP_L0B_ALLOC, Opcode::OP_L0C_ALLOC,
        Opcode::OP_L0C_ALLOC, Opcode::OP_L0C_ALLOC, Opcode::OP_COPY_IN, Opcode::OP_COPY_IN, Opcode::OP_COPY_IN, Opcode::OP_COPY_IN,
        Opcode::OP_COPY_IN, Opcode::OP_COPY_IN, Opcode::OP_L1_TO_L0A, Opcode::OP_L1_TO_L0A, Opcode::OP_L1_TO_L0A, Opcode::OP_L1_TO_L0A,
        Opcode::OP_L1_TO_L0B, Opcode::OP_L1_TO_L0B, Opcode::OP_L1_TO_L0B, Opcode::OP_L1_TO_L0B, Opcode::OP_L0C_TO_L1, Opcode::OP_L0C_TO_L1,
        Opcode::OP_L0C_TO_L1, Opcode::OP_A_MUL_B, Opcode::OP_A_MUL_B, Opcode::OP_A_MUL_B, Opcode::OP_COPY_OUT, Opcode::OP_A_MULACC_B};
    std::vector<std::vector<std::string>> inputoperands{{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {},
        {"DDR2"}, {"DDR1"}, {"DDR3"}, {"DDR4"}, {"DDR5"}, {"DDR6"}, {"T1"}, {"T6"}, {"T11"}, {"T16"}, {"T3"}, {"T8"}, {"T13"}, {"T18"},
        {"T5"}, {"T15"}, {"T20"}, {"T2", "T4"}, {"T7", "T9"}, {"T12", "T14"}, {"T21"}, {"T10", "T17", "T19"}};
    std::vector<std::vector<std::string>> outputoperands{{"T1"}, {"T3"}, {"T6"}, {"T8"}, {"T11"}, {"T13"}, {"T16"}, {"T18"}, {"T2"}, {"T21"}, {"T7"},
        {"T12"}, {"T17"}, {"T4"}, {"T9"}, {"T14"}, {"T19"}, {"T5"}, {"T10"}, {"T15"},
        {"T3"}, {"T1"}, {"T8"}, {"T11"}, {"T13"}, {"T18"}, {"T2"}, {"T7"}, {"T12"}, {"T17"}, {"T4"}, {"T9"}, {"T14"}, {"T19"},
        {"T6"}, {"T16"}, {"T21"}, {"T5"}, {"T10"}, {"T15"}, {"DDR7"}, {"T20"}};
    std::vector<std::string> operationNames{"L1_Alloc1", "L1_Alloc2", "L1_Alloc3", "L1_Alloc4", "L1_Alloc5", "L1_Alloc6",
        "L1_Alloc7", "L1_Alloc8", "L0A_Alloc1", "L1_Alloc9", "L0A_Alloc2", "L0A_Alloc3", "L0A_Alloc4",
        "L0B_Alloc1", "L0B_Alloc2", "L0B_Alloc3", "L0B_Alloc4", "L0C_Alloc1", "L0C_Alloc2", "L0C_Alloc3",
        "Copyin2", "Copyin1", "Copyin3", "Copyin4", "Copyin5", "Copyin6", "OP_L1_TO_L0A_1", "OP_L1_TO_L0A_2", "OP_L1_TO_L0A_3",
        "OP_L1_TO_L0A_4", "OP_L1_TO_L0B_1", "OP_L1_TO_L0B_2", "OP_L1_TO_L0B_3", "OP_L1_TO_L0B_4",
        "OP_L0C_TO_L1_1", "OP_L0C_TO_L1_2", "OP_L0C_TO_L1_3", "OP_A_MUL_B_1", "OP_A_MUL_B_2", "OP_A_MUL_B_3", "Copyout", "OP_A_MULACC_B"
    };
    EXPECT_EQ(subGraph.AddTensors(DataType::DT_FP32, {128, 128}, tensorMemTypes_L0AB, tensorNames_L0AB, 0), true);
    EXPECT_EQ(subGraph.AddTensors(DataType::DT_FP32, {16, 16}, tensorMemTypes, tensorNames, 0), true);
    EXPECT_EQ(subGraph.AddTensors(DataType::DT_FP32, {128, 256}, tensorMemTypes_L0C, tensorNames_L0C, 0), true);
    EXPECT_EQ(subGraph.AddOps(opCodes, inputoperands, outputoperands, operationNames, true), true);
    Function *function = subGraph.GetFunction();
    std::shared_ptr<LogicalTensor> tensor = subGraph.GetTensor("T10");
    tensor->memoryrange.memId =
        subGraph.GetTensor("T20")->memoryrange.memId;
    EXPECT_NE(function, nullptr);

    OptimizeSort optimizeSort(function->Operations().DuplicatedOpList(), *function);
    Status res = optimizeSort.SortOps();
    EXPECT_EQ(res, SUCCESS);
}

TEST_F(ScheduleOoOTest, TestHasEnoughBuffer) {
    ComputationalGraphBuilder subGraph;
    std::vector<std::string> tensorNames{"t1", "t2"};
    std::vector<MemoryType> tensorMemTypes{MemoryType::MEM_UB, MemoryType::MEM_UB};
    std::vector<Opcode> opCodes{Opcode::OP_UB_ALLOC};
    std::vector<std::vector<std::string>> ioperands{{}};
    std::vector<std::vector<std::string>> ooperands{{"t1", "t2"}};
    std::vector<std::string> opNames{"Alloc1"};
    EXPECT_EQ(subGraph.AddTensors(DataType::DT_FP32, {128, 128}, tensorMemTypes, tensorNames, 0), true);
    EXPECT_EQ(subGraph.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    Function *function = subGraph.GetFunction();
    EXPECT_NE(function, nullptr);

    auto op = subGraph.GetOp("Alloc1");
    auto issue = std::make_shared<IssueEntry>(*op, 1);

    OoOScheduler ooOScheduler(*function);
    bool res = ooOScheduler.HasEnoughBuffer(issue, MemoryType::MEM_UB);
    EXPECT_EQ(res, false);
}

TEST_F(ScheduleOoOTest, TestHasEnoughBufferAddMemId) {
    ComputationalGraphBuilder subGraph;
    std::vector<std::string> tensorNames{"t1", "t2"};
    std::vector<MemoryType> tensorMemTypes{MemoryType::MEM_UB, MemoryType::MEM_UB};
    std::vector<Opcode> opCodes{Opcode::OP_UB_ALLOC, Opcode::OP_COPY_IN};
    std::vector<std::vector<std::string>> ioperands{{}, {"t2"}};
    std::vector<std::vector<std::string>> ooperands{{"t1"}, {"t1"}};
    std::vector<std::string> opNames{"Alloc1", "COPY_IN"};
    EXPECT_EQ(subGraph.AddTensors(DataType::DT_FP32, {128, 128}, tensorMemTypes, tensorNames, 0), true);
    EXPECT_EQ(subGraph.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    Function *function = subGraph.GetFunction();
    EXPECT_NE(function, nullptr);
    auto op = subGraph.GetOp("Alloc1");
    auto opCopyIn = subGraph.GetOp("COPY_IN");
    auto tensor1 = subGraph.GetTensor("t1");
    auto tensor2 = subGraph.GetTensor("t2");
    auto issue = std::make_shared<IssueEntry>(*op, 1);
    auto issue2 = std::make_shared<IssueEntry>(*opCopyIn, 2);
    issue->successors.insert(issue2->id);
    issue->tileOp.GetOutputOperand(0)->ClearAllProducers();
    issue->tileOp.GetOutputOperand(0)->AddProducer(*opCopyIn);
    OoOScheduler ooOScheduler(*function);
    ooOScheduler.issueEntryMap[issue2->id] = issue2;
    ooOScheduler.issueEntryMap[issue2->id]->reqMemIds = {1};
    ooOScheduler.InitLocalBuffer(tensor2, 1);
    bool res = ooOScheduler.HasEnoughBuffer(issue, MemoryType::MEM_UB);
    EXPECT_EQ(res, false);
}

TEST_F(ScheduleOoOTest, TestCoreAssign) {
    ComputationalGraphBuilder subGraph;
    std::vector<std::string> tensorNames{"t0", "t1", "t2", "t3", "t4", "t5", "t6", "t7", "t8", "t9", "t10"};
    std::vector<Opcode> opCodes{Opcode::OP_A_MUL_B, Opcode::OP_ADDS, Opcode::OP_ADDS, Opcode::OP_ADDS,
        Opcode::OP_ADDS, Opcode::OP_A_MUL_B, Opcode::OP_ADDS, Opcode::OP_A_MUL_B, Opcode::OP_ADD, Opcode::OP_A_MULACC_B};
    std::vector<std::vector<std::string>> ioperands{{"t0", "t0"}, {"t1"}, {"t1"}, {"t1"}, {"t2", "t2"}, {"t2"}, {"t4"}, {"t5", "t5"}, {"t3", "t6"}, {"t7", "t8", "t9"}};
    std::vector<std::vector<std::string>> ooperands{{"t1"}, {"t2"}, {"t8"}, {"t9"}, {"t3"}, {"t4"}, {"t5"}, {"t6"}, {"t7"}, {"t10"}};
    std::vector<std::string> opNames{"op1", "op2", "op3", "op4", "op5", "op6", "op7", "op8", "op9", "op10"};
    EXPECT_EQ(subGraph.AddTensors(DataType::DT_FP32, {256, 256}, tensorNames), true);
    EXPECT_EQ(subGraph.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    Function *function = subGraph.GetFunction();
    EXPECT_NE(function, nullptr);
    auto opList = function->Operations(false).DuplicatedOpList();
    TaskSpliter spliter;
    spliter.SplitGraph(opList);
    CoreScheduler coreScheduler;
    const int bruteForceThreshold = 10;
    coreScheduler.Schedule(spliter.GetTaskGraph(), bruteForceThreshold);
    const int taskNum = 6;
    EXPECT_EQ(spliter.GetTaskGraph().tasks.size(), taskNum);
    spliter.MergeTask();
    OoOScheduler ooOScheduler(*function);
    spliter.MarkInternalSubgraphID();
    EXPECT_EQ(spliter.GetMergedOperations().size(), opList.size());
}

TEST_F(ScheduleOoOTest, TestLatencyEstimatorMainLoop) {
    // 创建测试数据
    ComputationalGraphBuilder subGraph;

    // 定义测试张量
    std::vector<std::string> tensorNames{"t1", "t2", "t3", "t4", "t5"};
    std::vector<MemoryType> tensorMemTypes{
        MemoryType::MEM_UB, MemoryType::MEM_L0A, MemoryType::MEM_L0B,
        MemoryType::MEM_L0C, MemoryType::MEM_UB
    };

    std::vector<Opcode> opCodes{Opcode::OP_UB_ALLOC, Opcode::OP_L0A_ALLOC, Opcode::OP_L0B_ALLOC, Opcode::OP_A_MUL_B, Opcode::OP_L0C_ALLOC,
        Opcode::OP_UB_ALLOC, Opcode::OP_L0C_COPY_UB, Opcode::OP_UB_COPY_L1};

    std::vector<std::vector<std::string>> ioperands{{}, {}, {}, {"t2", "t3"}, {}, {}, {"t4"},  {"t1"}};

    std::vector<std::vector<std::string>> ooperands{{"t1"}, {"t2"}, {"t3"}, {"t4"}, {"t4"}, {"t5"}, {"t5"}, {"t2"}};

    std::vector<std::string> opNames{
        "UB_ALLOC2", "L0A_Alloc1", "L0B_Alloc1", "Mul1", "L0C_Alloc1",
        "UB_ALLOC1", "OP_L0C_COPY_UB", "OP_UB_COPY_L1"
    };

    // 构建计算图
    EXPECT_EQ(subGraph.AddTensors(DataType::DT_FP32, {16, 16}, tensorMemTypes, tensorNames, 0), true);
    EXPECT_EQ(subGraph.AddOps(opCodes, ioperands, ooperands, opNames, true), true);

    Function *function = subGraph.GetFunction();
    EXPECT_NE(function, nullptr);

    // 创建LatencyEstimator实例
    auto opList = function->Operations(false).DuplicatedOpList();
    auto taskList = opList;
    taskList.erase(taskList.begin());
    int latency = 0;
    OoOSchedule oooSchedule;
    Status res = oooSchedule.SortAndLatencyEstimate(opList, taskList, latency);
    EXPECT_EQ(res, SUCCESS);
}

TEST_F(ScheduleOoOTest, TestMixSchedule) {
    ComputationalGraphBuilder subGraph;
    std::vector<std::string> tensorNames{"T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8"};
    std::vector<MemoryType> tensorMemTypes{MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_UB, MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_UB,
        MemoryType::MEM_UB, MemoryType::MEM_UB, MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_UB};
    std::vector<Opcode> opcodeList{Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC,
        Opcode::OP_UB_ALLOC, Opcode::OP_COPY_IN, Opcode::OP_COPY_IN, Opcode::OP_ROWMAX_SINGLE, Opcode::OP_ADD, Opcode::OP_COPY_OUT};
    std::vector<std::vector<std::string>> inputoperands{{}, {}, {}, {}, {}, {"T1"}, {"T3"}, {"T2"}, {"T4", "T5"}, {"T5"}};
    std::vector<std::vector<std::string>> outputoperands{{"T2"}, {"T4"}, {"T5"}, {"T6"}, {"T8"}, {"T2"}, {"T4"}, {"T5", "T6"}, {"T8"}, {"T7"}};
    std::vector<std::string> opNames{"Alloc1", "Alloc2", "Alloc3", "Alloc4", "Alloc5", "Copyin1", "Copyin2", "RowMax1", "Add1", "Copyout1"};
    EXPECT_EQ(subGraph.AddTensors(DataType::DT_FP32, {32, 32}, tensorMemTypes, tensorNames, 0), true);
    EXPECT_EQ(subGraph.AddOps(opcodeList, inputoperands, outputoperands, opNames, true), true);
    Function *function = subGraph.GetFunction();
    OoOSchedule oooSchedule;
    auto opList = function->Operations(false).DuplicatedOpList();
    std::pair<uint64_t, Function*> functionPair = std::make_pair(0, function);
    int size = 0;
    Status res = oooSchedule.MixSchedule(opList, *function, functionPair, size);
    EXPECT_EQ(res, SUCCESS);
}

TEST_F(ScheduleOoOTest, TestBufferPollRearrange) {
    // 构造bufferPool内存气泡场景
    BufferPool pool;
    pool.memSize_ = UBPoolSize;
    BufferSlice s1(32768, 65536);
    BufferSlice s2(98304, 98304);
    pool.bufferSlices[1] = s1;
    pool.bufferSlices[2] = s2;
    EXPECT_FALSE(pool.CheckBufferSlicesOverlap());

    // 构造子图
    ComputationalGraphBuilder subGraph;
    std::vector<std::string> tensorNames{"t1", "t2"};
    std::vector<MemoryType> tensorMemTypes{MemoryType::MEM_UB, MemoryType::MEM_UB};
    std::vector<Opcode> opCodes{Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC};
    std::vector<std::vector<std::string>> ioperands{{}, {}};
    std::vector<std::vector<std::string>> ooperands{{"t1"}, {"t2"}};
    std::vector<std::string> opNames{"Alloc1", "Alloc2"};
    EXPECT_EQ(subGraph.AddTensors(DataType::DT_FP32, {128, 128}, tensorMemTypes, tensorNames, 0), true);
    EXPECT_EQ(subGraph.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    Function *function = subGraph.GetFunction();
    EXPECT_NE(function, nullptr);

    // 构造issueEntry
    auto alloc1 = subGraph.GetOp("Alloc1");
    auto allocIssue1 = std::make_shared<IssueEntry>(*alloc1, 1);
    auto alloc2 = subGraph.GetOp("Alloc2");
    auto allocIssue2 = std::make_shared<IssueEntry>(*alloc2, 2);

    // 验证重排，排序依据为size从大到小
    OoOScheduler oooSchedule(*function);
    oooSchedule.bufferManagerMap[MemoryType::MEM_UB] = pool;
    oooSchedule.tensorOccupyMap[MemoryType::MEM_UB].emplace(1, allocIssue1);
    oooSchedule.tensorOccupyMap[MemoryType::MEM_UB].emplace(2, allocIssue2);
    EXPECT_EQ(oooSchedule.RearrangeBuffer(MemoryType::MEM_UB), SUCCESS);
    auto &ubPool = oooSchedule.bufferManagerMap[MemoryType::MEM_UB];
    EXPECT_EQ(ubPool.GetBufferSize(1), 65536);
    EXPECT_EQ(ubPool.GetBufferSize(2), 98304);
    EXPECT_EQ(ubPool.GetBufferOffset(1), 98304);
    EXPECT_EQ(ubPool.GetBufferOffset(2), 0);
}

} // namespace npu::tile_fwk