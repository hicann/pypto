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
#define private public
#include "passes/block_graph_pass/schedule_ooo/schedule_ooo.h"
#include "passes/block_graph_pass/schedule_ooo/core_assign.h"
#include "passes/block_graph_pass/schedule_ooo/buffer_rearrange.h"
#include "operator/models/deepseek/deepseek_mla.h"
#include "computational_graph_builder.h"

namespace npu::tile_fwk {
constexpr int OOO_NUM2 = 2;
constexpr int OOO_NUM209 = 209;
constexpr int UBPoolSize = 192 * 1024;
std::unordered_map<Opcode, int> preNodePriority = {
    // ALLOC 节点优先级最高，因为一个节点的前序ALLOC节点要在最靠近该节点的地方访问。
    {Opcode::OP_UB_ALLOC, 0},
    {Opcode::OP_L1_ALLOC, 0},
    {Opcode::OP_L0A_ALLOC, 0},
    {Opcode::OP_L0B_ALLOC, 0},
    {Opcode::OP_L0C_ALLOC, 0},
    {Opcode::OP_BT_ALLOC, 0},
    {Opcode::OP_FIX_ALLOC, 0},
    // 其次是L0级数据搬运Op。
    {Opcode::OP_L1_TO_L0A, 1},
    {Opcode::OP_L1_TO_L0B, 1},
    {Opcode::OP_L1_TO_L0_AT, 1},
    {Opcode::OP_L1_TO_L0_BT, 1},
    {Opcode::OP_L1_TO_FIX, 1},
    {Opcode::OP_L1_TO_FIX_QUANT_PRE, 1},
    {Opcode::OP_L1_TO_FIX_RELU_PRE, 1},
    {Opcode::OP_L1_TO_FIX_RELU_POST, 1},
    {Opcode::OP_L1_TO_FIX_QUANT_POST, 1},
    {Opcode::OP_L1_TO_FIX_ELT_ANTIQ, 1},
    {Opcode::OP_L1_TO_FIX_MTE2_ANTIQ, 1},
    {Opcode::OP_L1_TO_BT, 1},
    // 再其次是L1级数据搬运Op。
    {Opcode::OP_COPY_IN, 2},
    {Opcode::OP_UB_COPY_IN, 2},
    {Opcode::OP_L1_COPY_IN, 2},
    {Opcode::OP_L1_COPY_IN_FRACTAL_Z, 2},
    {Opcode::OP_L1_COPY_UB, 2},
    {Opcode::OP_L0C_COPY_UB, 2},
    {Opcode::OP_UB_COPY_L1, 2},
    // 最后访问其它计算节点（其它节点默认的优先级为10）。
};

class ScheduleOoOTest : public ::testing::Test {
public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}

    void SetUp() override { Program::GetInstance().Reset(); }
    void TearDown() override {}
};

void SetTensorAttr(LogicalTensorPtr tensor, MemoryType memType, int memId) {
    tensor->SetMemoryTypeOriginal(memType);
    tensor->SetMemoryTypeToBe(memType);
    tensor->memoryrange.memId = memId;
    tensor->UpdateDynValidShape({SymbolicScalar("S0"), SymbolicScalar("S1")});
}

void SetAllocAttr(Operation& alloc, int latency) { alloc.UpdateLatency(latency); }

LogicalTensorPtr CreateTensor(
    Function& currFunction, DataType dateType, std::vector<int64_t> shape, MemoryType memType, int memId)
{
    LogicalTensorPtr tensor = std::make_shared<LogicalTensor>(currFunction, dateType, shape);
    SetTensorAttr(tensor, memType, memId);
    return tensor;
}

Operation& CreateAllocOp(Function& currFunction, LogicalTensorPtr tensor, int latency)
{
    Operation& alloc = currFunction.AddOperation(Opcode::OP_UB_ALLOC, {}, LogicalTensors({tensor}));
    SetAllocAttr(alloc, latency);
    return alloc;
}

Operation& CreateCopyOp(
    Function& currFunction, Opcode opcode, LogicalTensorPtr inTensor, LogicalTensorPtr outTensor,
    std::vector<int64_t> shape)
{
    std::vector<int64_t> offset = {0, 0};
    auto& copy = currFunction.AddOperation(opcode, LogicalTensors({inTensor}), LogicalTensors({outTensor}));
    auto shapeImme = OpImmediate::Specified(shape);
    if (opcode == Opcode::OP_COPY_IN) {
        copy.SetOpAttribute(
            std::make_shared<CopyOpAttribute>(OpImmediate::Specified(offset), MEM_UB, shapeImme, shapeImme));
    }
    if (opcode == Opcode::OP_COPY_OUT) {
        copy.SetOpAttribute(
            std::make_shared<CopyOpAttribute>(MEM_UB, OpImmediate::Specified(offset), shapeImme, shapeImme));
    }
    return copy;
}

Operation& CreateAddOp(
    Function& currFunction, LogicalTensorPtr inTensor1, LogicalTensorPtr inTensor2, LogicalTensorPtr outTensor)
{
    auto& add =
        currFunction.AddOperation(Opcode::OP_ADD, LogicalTensors({inTensor1, inTensor2}), LogicalTensors({outTensor}));
    return add;
}

void ReorderOperations(Function& function)
{
    auto opList = function.Operations().DuplicatedOpList();
    std::vector<Operation*> newOperations;
    for (auto& op : opList) {
        if (op->GetOpcodeStr().find("ALLOC") != std::string::npos) {
            newOperations.insert(newOperations.begin(), op);
        } else {
            newOperations.push_back(op);
        }
    }
    function.ScheduleBy(newOperations);
}

TEST_F(ScheduleOoOTest, TestMainScheduleOoO)
{
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
    auto& alloc1 = CreateAllocOp(*currFunctionPtr, tensor3, 1);
    auto& alloc2 = CreateAllocOp(*currFunctionPtr, tensor4, 1);
    auto& alloc3 = CreateAllocOp(*currFunctionPtr, tensor5, 1);
    auto& alloc4 = CreateAllocOp(*currFunctionPtr, tensor6, 1);
    auto& alloc5 = CreateAllocOp(*currFunctionPtr, tensor8, 1);
    auto& alloc6 = CreateAllocOp(*currFunctionPtr, tensor9, 1);
    auto& copyin1 = CreateCopyOp(*currFunctionPtr, Opcode::OP_COPY_IN, tensor1, tensor3, shape);
    auto& copyin2 = CreateCopyOp(*currFunctionPtr, Opcode::OP_COPY_IN, tensor2, tensor4, shape);
    auto& add1 = CreateAddOp(*currFunctionPtr, tensor3, tensor4, tensor5);
    auto& add2 = CreateAddOp(*currFunctionPtr, tensor3, tensor4, tensor6);
    auto& add3 = CreateAddOp(*currFunctionPtr, tensor6, tensor4, tensor8);
    auto& add4 = CreateAddOp(*currFunctionPtr, tensor8, tensor5, tensor9);
    (void)alloc1, (void)alloc2, (void)alloc3, (void)alloc4, (void)alloc5, (void)alloc6, (void)copyin1, (void)copyin2,
        (void)add1, (void)add2, (void)add3, (void)add4;
    for (auto& program : rootFuncPtr->rootFunc_->programs_) {
        ReorderOperations(*(program.second));
    }
    currFunctionPtr->EndFunction(nullptr);
    emptyOpFunctionPtr->EndFunction(nullptr);
    OoOSchedule oooSchedule;
    EXPECT_EQ(oooSchedule.PreCheck(*rootFuncPtr), SUCCESS);
    oooSchedule.RunOnFunction(*rootFuncPtr);
    EXPECT_EQ(oooSchedule.PostCheck(*rootFuncPtr), SUCCESS);
}

static bool CheckViewOps(std::vector<Operation*> &viewOps, Operation *op) {
    for (auto viewop : viewOps) {
        if (viewop == op) {
            return true;
        }
    }
    return false;
}

TEST_F(ScheduleOoOTest, TestDependencies)
{
    ComputationalGraphBuilder subGraph;
    std::vector<std::string> tensorNames{"t1", "t2", "t3", "t4", "t5", "t6", "t7", "t8"};
    std::vector<MemoryType> tensorMemTypes{MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_UB, MemoryType::MEM_DEVICE_DDR,
                                           MemoryType::MEM_UB,         MemoryType::MEM_UB, MemoryType::MEM_UB,
                                           MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_UB};
    std::vector<Opcode> opCodes{Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC,
                                Opcode::OP_UB_ALLOC, Opcode::OP_COPY_IN,  Opcode::OP_COPY_IN,  Opcode::OP_ROWMAX_SINGLE,
                                Opcode::OP_ADD,      Opcode::OP_COPY_OUT};
    std::vector<std::vector<std::string>> ioperands{{}, {}, {}, {}, {}, {"t1"}, {"t3"}, {"t2"}, {"t4", "t5"}, {"t5"}};
    std::vector<std::vector<std::string>> ooperands{{"t2"}, {"t4"}, {"t5"},       {"t6"}, {"t8"},
                                                    {"t2"}, {"t4"}, {"t5", "t6"}, {"t8"}, {"t7"}};
    std::vector<std::string> opNames{"Alloc1",  "Alloc2",  "Alloc3",  "Alloc4", "Alloc5",
                                     "Copyin1", "Copyin2", "RowMax1", "Add1",   "Copyout1"};
    EXPECT_EQ(subGraph.AddTensors(DataType::DT_FP32, {16, 16}, tensorMemTypes, tensorNames, 0), true);
    EXPECT_EQ(subGraph.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    Function* function = subGraph.GetFunction();
    EXPECT_NE(function, nullptr);

    OoOScheduler ooOScheduler(*function);
    Status res = ooOScheduler.Init(function->Operations().DuplicatedOpList());
    Operation* op = subGraph.GetOp("RowMax1");
    EXPECT_NE(op, nullptr);
    EXPECT_EQ(ooOScheduler.GetPredecessors(op).size(), 3);
    EXPECT_TRUE(ooOScheduler.GetPredecessors(op).count(subGraph.GetOp("Alloc4")) > 0);
    EXPECT_EQ(ooOScheduler.GetSuccessors(op).size(), 2);
    EXPECT_TRUE(ooOScheduler.GetSuccessors(op).count(subGraph.GetOp("Add1")) > 0);
    EXPECT_EQ(res, SUCCESS);
}

TEST_F(ScheduleOoOTest, TestDependenciesView)
{
    ComputationalGraphBuilder subGraph;
    std::vector<std::string> tensorNames{"t1", "t2", "t3", "t4", "t5", "t6"};
    std::vector<MemoryType> tensorMemTypes{MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_UB, MemoryType::MEM_UB,
                                           MemoryType::MEM_UB,         MemoryType::MEM_UB, MemoryType::MEM_UB};
    std::vector<Opcode> opCodes{Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_COPY_IN, Opcode::OP_VIEW,
                                Opcode::OP_VIEW,     Opcode::OP_VIEW,     Opcode::OP_ADD};
    std::vector<std::vector<std::string>> ioperands{{}, {}, {"t1"}, {"t2"}, {"t2"}, {"t2"}, {"t3", "t4"}};
    std::vector<std::vector<std::string>> ooperands{{"t2"}, {"t6"}, {"t2"}, {"t3"}, {"t4"}, {"t5"}, {"t6"}};
    std::vector<std::string> opNames{"Alloc1", "Alloc2", "Copyin1", "View1", "View2", "View3", "Add1"};
    EXPECT_EQ(subGraph.AddTensors(DataType::DT_FP32, {16, 16}, tensorMemTypes, tensorNames, 0), true);
    EXPECT_EQ(subGraph.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    Function* function = subGraph.GetFunction();
    EXPECT_NE(function, nullptr);
    std::shared_ptr<LogicalTensor> tensor1 = subGraph.GetTensor("t3");
    tensor1->memoryrange.memId = subGraph.GetTensor("t2")->memoryrange.memId;
    std::shared_ptr<LogicalTensor> tensor2 = subGraph.GetTensor("t4");
    tensor2->memoryrange.memId = subGraph.GetTensor("t2")->memoryrange.memId;
    std::shared_ptr<LogicalTensor> tensor3 = subGraph.GetTensor("t5");
    tensor3->memoryrange.memId = subGraph.GetTensor("t2")->memoryrange.memId;
    OoOScheduler ooOScheduler(*function);
    Status res = ooOScheduler.Init(function->Operations().DuplicatedOpList());
    Operation* copyin = subGraph.GetOp("Copyin1");
    EXPECT_NE(copyin, nullptr);
    Operation* add = subGraph.GetOp("Add1");
    EXPECT_NE(add, nullptr);
    EXPECT_TRUE(ooOScheduler.GetPredecessors(copyin).count(subGraph.GetOp("Alloc1")) > 0);
    EXPECT_TRUE(ooOScheduler.GetPredecessors(add).count(subGraph.GetOp("Alloc2")) > 0);
    EXPECT_TRUE(ooOScheduler.GetPredecessors(add).count(subGraph.GetOp("Copyin1")) > 0);
    EXPECT_TRUE(CheckViewOps(ooOScheduler.GetViewOps(add), subGraph.GetOp("View1")));
    EXPECT_TRUE(CheckViewOps(ooOScheduler.GetViewOps(add), subGraph.GetOp("View2")));
    EXPECT_EQ(res, SUCCESS);
}

TEST_F(ScheduleOoOTest, TestDependenciesAssemble)
{
    ComputationalGraphBuilder subGraph;
    std::vector<std::string> tensorNames{"t1", "t2", "t3", "t4", "t5", "t6", "t7", "t8"};
    std::vector<MemoryType> tensorMemTypes{
        MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_UB,
        MemoryType::MEM_UB,         MemoryType::MEM_UB,         MemoryType::MEM_UB,         MemoryType::MEM_DEVICE_DDR};
    std::vector<Opcode> opCodes{Opcode::OP_UB_ALLOC, Opcode::OP_SUB,      Opcode::OP_SUB,      Opcode::OP_SUB,
                                Opcode::OP_ASSEMBLE, Opcode::OP_ASSEMBLE, Opcode::OP_ASSEMBLE, Opcode::OP_MUL};
    std::vector<std::vector<std::string>> ioperands{{}, {"t1"}, {"t2"}, {"t3"}, {"t4"}, {"t5"}, {"t6"}, {"t7"}};
    std::vector<std::vector<std::string>> ooperands{{"t4"}, {"t4"}, {"t5"}, {"t6"}, {"t7"}, {"t7"}, {"t7"}, {"t8"}};
    std::vector<std::string> opNames{"Alloc1", "Sub1", "Sub2", "Sub3", "Assemble1", "Assemble2", "Assemble3", "Mul1"};
    EXPECT_EQ(subGraph.AddTensors(DataType::DT_FP32, {16, 16}, tensorMemTypes, tensorNames, 0), true);
    EXPECT_EQ(subGraph.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    Function* function = subGraph.GetFunction();
    EXPECT_NE(function, nullptr);

    for (size_t i = 3; i < tensorNames.size() - 1; i++) {
        EXPECT_NE(subGraph.GetTensor(tensorNames[i]), nullptr);
        std::shared_ptr<LogicalTensor> tensor = subGraph.GetTensor(tensorNames[i]);
        tensor->memoryrange.memId = subGraph.GetTensor("t4")->memoryrange.memId;
    }

    OoOScheduler ooOScheduler(*function);
    Status res = ooOScheduler.Init(function->Operations().DuplicatedOpList());
    Operation* alloc = subGraph.GetOp("Alloc1");
    EXPECT_NE(alloc, nullptr);
    Operation* sub = subGraph.GetOp("Sub1");
    EXPECT_NE(sub, nullptr);
    EXPECT_TRUE(ooOScheduler.GetSuccessors(alloc).count(subGraph.GetOp("Sub3")) > 0);
    EXPECT_TRUE(ooOScheduler.GetPredecessors(sub).count(subGraph.GetOp("Alloc1")) > 0);
    EXPECT_TRUE(ooOScheduler.GetSuccessors(sub).count(subGraph.GetOp("Assemble1")) > 0);
    EXPECT_EQ(res, SUCCESS);
}

TEST_F(ScheduleOoOTest, TestDependenciesInplace)
{
    ComputationalGraphBuilder subGraph;
    std::vector<std::string> tensorNames{"t1", "t2", "t3", "t4"};
    std::vector<MemoryType> tensorMemTypes{
        MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_UB, MemoryType::MEM_UB, MemoryType::MEM_DEVICE_DDR};
    std::vector<Opcode> opCodes{Opcode::OP_UB_ALLOC, Opcode::OP_COPY_IN, Opcode::OP_ADD, Opcode::OP_COPY_OUT};
    std::vector<std::vector<std::string>> ioperands{{}, {"t1"}, {"t2"}, {"t3"}};
    std::vector<std::vector<std::string>> ooperands{{"t2"}, {"t2"}, {"t3"}, {"t4"}};
    std::vector<std::string> opNames{"Alloc1", "Copyin1", "Add1", "Copyout1"};
    EXPECT_EQ(subGraph.AddTensors(DataType::DT_FP32, {16, 16}, tensorMemTypes, tensorNames, 0), true);
    EXPECT_EQ(subGraph.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    Function* function = subGraph.GetFunction();
    EXPECT_NE(function, nullptr);

    EXPECT_NE(subGraph.GetTensor("t3"), nullptr);
    std::shared_ptr<LogicalTensor> tensor = subGraph.GetTensor("t3");
    tensor->memoryrange.memId = subGraph.GetTensor("t2")->memoryrange.memId;

    OoOScheduler ooOScheduler(*function);
    Status res = ooOScheduler.Init(function->Operations().DuplicatedOpList());
    Operation* add = subGraph.GetOp("Add1");
    EXPECT_NE(add, nullptr);
    EXPECT_TRUE(ooOScheduler.GetSuccessors(add).count(subGraph.GetOp("Copyout1")) > 0);
    EXPECT_TRUE(ooOScheduler.GetPredecessors(add).count(subGraph.GetOp("Copyin1")) > 0);
    EXPECT_EQ(res, SUCCESS);
}

TEST_F(ScheduleOoOTest, TestDependenciesTrue)
{
    ComputationalGraphBuilder subGraph;
    std::vector<std::string> tensorNames{"t1", "t2", "t3", "t4"};
    std::vector<MemoryType> tensorMemTypes{
        MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_UB, MemoryType::MEM_UB, MemoryType::MEM_DEVICE_DDR};
    std::vector<Opcode> opCodes{Opcode::OP_COPY_IN, Opcode::OP_UB_ALLOC, Opcode::OP_ADD, Opcode::OP_COPY_OUT};
    std::vector<std::vector<std::string>> ioperands{{"t1"}, {}, {"t2"}, {"t3"}};
    std::vector<std::vector<std::string>> ooperands{{"t2"}, {"t2"}, {"t3"}, {"t4"}};
    std::vector<std::string> opNames{"Copyin1", "Alloc1", "Add1", "Copyout1"};
    EXPECT_EQ(subGraph.AddTensors(DataType::DT_FP32, {16, 16}, tensorMemTypes, tensorNames, 0), true);
    EXPECT_EQ(subGraph.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    Function* function = subGraph.GetFunction();
    EXPECT_NE(function, nullptr);

    EXPECT_NE(subGraph.GetTensor("t3"), nullptr);
    std::shared_ptr<LogicalTensor> tensor = subGraph.GetTensor("t3");
    tensor->memoryrange.memId = subGraph.GetTensor("t2")->memoryrange.memId;

    OoOScheduler ooOScheduler(*function);
    Status res = ooOScheduler.Init(function->Operations().DuplicatedOpList());
    EXPECT_EQ(res, SUCCESS);
    std::rotate(ooOScheduler.orderedOps.begin(), ooOScheduler.orderedOps.begin() + 1, ooOScheduler.orderedOps.end());
    res = ooOScheduler.depManager_.InitDependencies(ooOScheduler.orderedOps, false);
    EXPECT_EQ(res, SUCCESS);
}

TEST_F(ScheduleOoOTest, TestSpillCopyIn)
{
    ComputationalGraphBuilder subGraph;
    std::vector<std::string> tensorNames{"t1", "t2", "t3", "t4", "t5", "t6", "t7", "t8"};
    std::vector<MemoryType> tensorMemTypes{MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_UB,
                                           MemoryType::MEM_UB,         MemoryType::MEM_UB,         MemoryType::MEM_UB,
                                           MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_UB};
    std::vector<Opcode> opCodes{Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC,
                                Opcode::OP_UB_ALLOC, Opcode::OP_COPY_IN,  Opcode::OP_COPY_IN,  Opcode::OP_ADD,
                                Opcode::OP_ADD,      Opcode::OP_ADD,      Opcode::OP_COPY_OUT};
    std::vector<std::vector<std::string>> ioperands{{},     {},           {},           {},           {},    {"t1"},
                                                    {"t2"}, {"t3", "t4"}, {"t3", "t5"}, {"t4", "t6"}, {"t8"}};
    std::vector<std::vector<std::string>> ooperands{{"t3"}, {"t4"}, {"t5"}, {"t6"}, {"t8"}, {"t3"},
                                                    {"t4"}, {"t5"}, {"t6"}, {"t8"}, {"t7"}};
    std::vector<std::string> opNames{"Alloc1",  "Alloc2", "Alloc3", "Alloc4", "Alloc5",  "Copyin1",
                                     "Copyin2", "Add1",   "Add2",   "Add3",   "Copyout1"};
    EXPECT_EQ(subGraph.AddTensors(DataType::DT_FP32, {128, 128}, tensorMemTypes, tensorNames, 0), true);
    EXPECT_EQ(subGraph.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    Function* function = subGraph.GetFunction();
    EXPECT_NE(function, nullptr);

    OptimizeSort sort(function->Operations().DuplicatedOpList(), *function);
    Status res = sort.SortOps();
    EXPECT_EQ(res, SUCCESS);
    OoOScheduler ooOScheduler(*function);
    res = ooOScheduler.Init(sort.operations);
    EXPECT_EQ(res, SUCCESS);
    res = ooOScheduler.GenSpillSchedule();
    EXPECT_EQ(res, SUCCESS);
    EXPECT_EQ(ooOScheduler.orderedOps[9]->GetOpcodeStr(), "UB_ALLOC");
    EXPECT_EQ(ooOScheduler.orderedOps[10]->GetOpcodeStr(), "COPY_IN");
}

TEST_F(ScheduleOoOTest, TestSpill)
{
    ComputationalGraphBuilder subGraph;
    std::vector<std::string> tensorNames{"t1", "t2", "t3", "t4", "t5", "t6", "t7", "t8", "t9"};
    std::vector<MemoryType> tensorMemTypes{MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_UB,
                                           MemoryType::MEM_UB,         MemoryType::MEM_UB,         MemoryType::MEM_UB,
                                           MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_UB,         MemoryType::MEM_UB};
    std::vector<Opcode> opCodes{Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC,
                                Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_COPY_IN,  Opcode::OP_COPY_IN,
                                Opcode::OP_ADD,      Opcode::OP_ADD,      Opcode::OP_ADD,      Opcode::OP_ADD,
                                Opcode::OP_COPY_OUT};
    std::vector<std::vector<std::string>> ioperands{
        {}, {}, {}, {}, {}, {}, {"t1"}, {"t2"}, {"t3", "t4"}, {"t3", "t4"}, {"t4", "t6"}, {"t5", "t8"}, {"t9"}};
    std::vector<std::vector<std::string>> ooperands{{"t3"}, {"t4"}, {"t5"}, {"t6"}, {"t8"}, {"t9"}, {"t3"},
                                                    {"t4"}, {"t5"}, {"t6"}, {"t8"}, {"t9"}, {"t7"}};
    std::vector<std::string> opNames{"Alloc1",  "Alloc2", "Alloc3", "Alloc4", "Alloc5", "Alloc6",  "Copyin1",
                                     "Copyin2", "Add1",   "Add2",   "Add3",   "Add4",   "Copyout1"};
    EXPECT_EQ(subGraph.AddTensors(DataType::DT_FP32, {128, 128}, tensorMemTypes, tensorNames, 0), true);
    EXPECT_EQ(subGraph.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    Function* function = subGraph.GetFunction();
    EXPECT_NE(function, nullptr);

    OptimizeSort sort(function->Operations().DuplicatedOpList(), *function);
    Status res = sort.SortOps();
    EXPECT_EQ(res, SUCCESS);
    OoOScheduler ooOScheduler(*function);
    res = ooOScheduler.Init(sort.operations);
    EXPECT_EQ(res, SUCCESS);
    res = ooOScheduler.GenSpillSchedule();
    EXPECT_EQ(res, SUCCESS);
    EXPECT_EQ(ooOScheduler.orderedOps[8]->GetOpcodeStr(), "COPY_OUT");
    EXPECT_EQ(ooOScheduler.orderedOps[14]->GetOpcodeStr(), "UB_ALLOC");
    EXPECT_EQ(ooOScheduler.orderedOps[15]->GetOpcodeStr(), "COPY_IN");
}

TEST_F(ScheduleOoOTest, TestSpillInplace)
{
    ComputationalGraphBuilder subGraph;
    std::vector<std::string> tensorNames{"t1", "t2", "t3", "t4", "t5", "t6", "t7", "t8", "t9", "t10", "t11"};
    std::vector<MemoryType> tensorMemTypes{
        MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_DEVICE_DDR,
        MemoryType::MEM_UB,         MemoryType::MEM_UB,         MemoryType::MEM_UB,         MemoryType::MEM_UB,
        MemoryType::MEM_UB,         MemoryType::MEM_UB,         MemoryType::MEM_UB};
    std::vector<Opcode> opCodes{Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC,
                                Opcode::OP_UB_ALLOC, Opcode::OP_COPY_IN,  Opcode::OP_COPY_IN,  Opcode::OP_COPY_IN,
                                Opcode::OP_COPY_IN,  Opcode::OP_ADD,      Opcode::OP_ADD,      Opcode::OP_ADD};
    std::vector<std::vector<std::string>> ioperands{{},     {},     {},     {},           {},           {"t1"},
                                                    {"t2"}, {"t3"}, {"t4"}, {"t5", "t6"}, {"t7", "t8"}, {"t9", "t10"}};
    std::vector<std::vector<std::string>> ooperands{{"t5"}, {"t6"}, {"t7"}, {"t8"},  {"t9"}, {"t5"},
                                                    {"t6"}, {"t7"}, {"t8"}, {"t10"}, {"t9"}, {"t11"}};
    std::vector<std::string> opNames{"Alloc1",  "Alloc2",  "Alloc3",  "Alloc4", "Alloc5", "Copyin1",
                                     "Copyin2", "Copyin3", "Copyin4", "Add1",   "Add2",   "Add3"};
    EXPECT_EQ(subGraph.AddTensors(DataType::DT_FP32, {128, 128}, tensorMemTypes, tensorNames, 0), true);
    EXPECT_EQ(subGraph.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    Function* function = subGraph.GetFunction();
    EXPECT_NE(function, nullptr);

    EXPECT_NE(subGraph.GetTensor("t11"), nullptr);
    std::shared_ptr<LogicalTensor> tensor1 = subGraph.GetTensor("t10");
    tensor1->memoryrange.memId = subGraph.GetTensor("t5")->memoryrange.memId;
    std::shared_ptr<LogicalTensor> tensor2 = subGraph.GetTensor("t11");
    tensor2->memoryrange.memId = subGraph.GetTensor("t5")->memoryrange.memId;

    OptimizeSort sort(function->Operations().DuplicatedOpList(), *function);
    Status res = sort.SortOps();
    EXPECT_EQ(res, SUCCESS);
    OoOScheduler ooOScheduler(*function);
    res = ooOScheduler.Init(sort.operations);
    EXPECT_EQ(res, SUCCESS);
    std::rotate(ooOScheduler.orderedOps.begin(), ooOScheduler.orderedOps.begin() + 6, ooOScheduler.orderedOps.begin() + 11);
    res = ooOScheduler.GenSpillSchedule();
    EXPECT_EQ(res, SUCCESS);
    Operation* add1 = subGraph.GetOp("Add1");
    EXPECT_NE(add1, nullptr);
    Operation* add3 = subGraph.GetOp("Add3");
    EXPECT_NE(add3, nullptr);
    EXPECT_EQ((*ooOScheduler.GetSuccessors(add1).begin())->GetOpcodeStr(), "COPY_OUT");
    EXPECT_EQ(ooOScheduler.GetPredecessors(add3).size(), 3);
}

TEST_F(ScheduleOoOTest, TestSpillMultiTensor)
{
    ComputationalGraphBuilder subGraph;
    std::vector<std::string> tensorNames{"t1", "t2", "t3", "t4", "t5", "t6", "t7", "t8", "t9", "t10", "t11"};
    std::vector<MemoryType> tensorMemTypes{
        MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_DEVICE_DDR,
        MemoryType::MEM_UB,         MemoryType::MEM_UB,         MemoryType::MEM_UB,         MemoryType::MEM_UB,
        MemoryType::MEM_UB,         MemoryType::MEM_UB,         MemoryType::MEM_UB};
    std::vector<Opcode> opCodes{Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC,
                                Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_COPY_IN,
                                Opcode::OP_COPY_IN,  Opcode::OP_COPY_IN,  Opcode::OP_COPY_IN,  Opcode::OP_ADD,
                                Opcode::OP_ADD,      Opcode::OP_ADD};
    std::vector<std::vector<std::string>> ioperands{{},
                                                    {},
                                                    {},
                                                    {},
                                                    {},
                                                    {},
                                                    {},
                                                    {"t1"},
                                                    {"t2"},
                                                    {"t3"},
                                                    {"t4"},
                                                    {"t7", "t8"},
                                                    {"t5", "t6", "t9"},
                                                    {"t7", "t8", "t10"}};
    std::vector<std::vector<std::string>> ooperands{{"t5"}, {"t6"}, {"t7"}, {"t8"}, {"t9"}, {"t10"}, {"t11"},
                                                    {"t5"}, {"t6"}, {"t7"}, {"t8"}, {"t9"}, {"t10"}, {"t11"}};
    std::vector<std::string> opNames{"Alloc1",  "Alloc2",  "Alloc3",  "Alloc4",  "Alloc5", "Alloc6", "Alloc7",
                                     "Copyin1", "Copyin2", "Copyin3", "Copyin4", "Add1",   "Add2",   "Add3"};
    EXPECT_EQ(subGraph.AddTensors(DataType::DT_FP32, {50, 128}, tensorMemTypes, tensorNames, 0), true);
    EXPECT_EQ(subGraph.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    Function* function = subGraph.GetFunction();
    EXPECT_NE(function, nullptr);

    EXPECT_NE(subGraph.GetTensor("t6"), nullptr);
    std::shared_ptr<LogicalTensor> tensor1 = subGraph.GetTensor("t5");
    tensor1->shape = {128, 128};
    tensor1->tensor->rawshape = {128, 128};
    std::shared_ptr<LogicalTensor> tensor2 = subGraph.GetTensor("t6");
    tensor2->shape = {128, 128};
    tensor2->tensor->rawshape = {128, 128};

    OptimizeSort sort(function->Operations().DuplicatedOpList(), *function);
    Status res = sort.SortOps();
    EXPECT_EQ(res, SUCCESS);
    OoOScheduler ooOScheduler(*function);
    res = ooOScheduler.Init(sort.operations);
    EXPECT_EQ(res, SUCCESS);
    res = ooOScheduler.GenSpillSchedule();
    EXPECT_EQ(res, SUCCESS);
    EXPECT_EQ(ooOScheduler.orderedOps.size(), 23);
}

TEST_F(ScheduleOoOTest, TestSpillView)
{
    ComputationalGraphBuilder subGraph;
    std::vector<std::string> tensorNames{"t1", "t2", "t3", "t4", "t5", "t6", "t7", "t8", "t9"};
    std::vector<MemoryType> tensorMemTypes{MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_UB,
                                           MemoryType::MEM_UB,         MemoryType::MEM_UB,         MemoryType::MEM_UB,
                                           MemoryType::MEM_UB,         MemoryType::MEM_UB,         MemoryType::MEM_UB};
    std::vector<Opcode> opCodes{Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC,
                                Opcode::OP_UB_ALLOC, Opcode::OP_COPY_IN,  Opcode::OP_COPY_IN,  Opcode::OP_VIEW,
                                Opcode::OP_VIEW,     Opcode::OP_ADD,      Opcode::OP_ADD,      Opcode::OP_ADD};
    std::vector<std::vector<std::string>> ioperands{{},     {},     {},     {},     {},     {"t1"},
                                                    {"t2"}, {"t3"}, {"t3"}, {"t5"}, {"t6"}, {"t4", "t7"}};
    std::vector<std::vector<std::string>> ooperands{{"t3"}, {"t4"}, {"t7"}, {"t8"}, {"t9"}, {"t3"},
                                                    {"t4"}, {"t5"}, {"t6"}, {"t7"}, {"t8"}, {"t9"}};
    std::vector<std::string> opNames{"Alloc1",  "Alloc2", "Alloc3", "Alloc4", "Alloc5", "Copyin1",
                                     "Copyin2", "View1",  "View2",  "Add1",   "Add2",   "Add3"};
    EXPECT_EQ(subGraph.AddTensors(DataType::DT_FP32, {128, 128}, tensorMemTypes, tensorNames, 0), true);
    EXPECT_EQ(subGraph.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    Function* function = subGraph.GetFunction();
    EXPECT_NE(function, nullptr);

    EXPECT_NE(subGraph.GetTensor("t6"), nullptr);
    std::shared_ptr<LogicalTensor> tensor1 = subGraph.GetTensor("t5");
    tensor1->memoryrange.memId = subGraph.GetTensor("t3")->memoryrange.memId;
    std::shared_ptr<LogicalTensor> tensor2 = subGraph.GetTensor("t6");
    tensor2->memoryrange.memId = subGraph.GetTensor("t3")->memoryrange.memId;

    OptimizeSort sort(function->Operations().DuplicatedOpList(), *function);
    Status res = sort.SortOps();
    EXPECT_EQ(res, SUCCESS);
    OoOScheduler ooOScheduler(*function);
    res = ooOScheduler.Init(sort.operations);
    EXPECT_EQ(res, SUCCESS);
    res = ooOScheduler.GenSpillSchedule();
    EXPECT_EQ(res, SUCCESS);
}

TEST_F(ScheduleOoOTest, TestSpillAssemble)
{
    ComputationalGraphBuilder subGraph;
    std::vector<std::string> tensorNames{"t1", "t2", "t3",  "t4",  "t5",  "t6",  "t7",
                                         "t8", "t9", "t10", "t11", "t12", "t13", "t14"};
    std::vector<MemoryType> tensorMemTypes{
        MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_DEVICE_DDR,
        MemoryType::MEM_UB,         MemoryType::MEM_UB,         MemoryType::MEM_UB,         MemoryType::MEM_UB,
        MemoryType::MEM_UB,         MemoryType::MEM_UB,         MemoryType::MEM_UB,         MemoryType::MEM_UB,
        MemoryType::MEM_UB,         MemoryType::MEM_UB};
    std::vector<Opcode> opCodes{Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC,
                                Opcode::OP_UB_ALLOC, Opcode::OP_ADD,      Opcode::OP_ADD,      Opcode::OP_ADD,
                                Opcode::OP_ADD,      Opcode::OP_ADD,      Opcode::OP_ADD,      Opcode::OP_SUB,
                                Opcode::OP_SUB,      Opcode::OP_ASSEMBLE, Opcode::OP_ASSEMBLE, Opcode::OP_ASSEMBLE,
                                Opcode::OP_ASSEMBLE, Opcode::OP_MUL};
    std::vector<std::vector<std::string>> ioperands{{},     {},     {},      {},      {},      {"t1"},
                                                    {"t2"}, {"t3"}, {"t4"},  {"t5"},  {"t6"},  {"t7"},
                                                    {"t8"}, {"t9"}, {"t10"}, {"t11"}, {"t12"}, {"t13"}};
    std::vector<std::vector<std::string>> ooperands{{"t5"},  {"t6"},  {"t7"},  {"t8"},  {"t12"}, {"t5"},
                                                    {"t6"},  {"t7"},  {"t8"},  {"t9"},  {"t10"}, {"t11"},
                                                    {"t12"}, {"t13"}, {"t13"}, {"t13"}, {"t13"}, {"t14"}};
    std::vector<std::string> opNames{"Alloc1",  "Alloc2",    "Alloc3",    "Alloc4",    "Alloc5",    "Copyin1",
                                     "Copyin2", "Copyin3",   "Copyin4",   "Add1",      "Add2",      "Sub1",
                                     "Sub2",    "Assemble1", "Assemble2", "Assemble3", "Assemble4", "Mul1"};
    EXPECT_EQ(subGraph.AddTensors(DataType::DT_FP32, {128, 64}, tensorMemTypes, tensorNames, 0), true);
    EXPECT_EQ(subGraph.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    Function* function = subGraph.GetFunction();
    EXPECT_NE(function, nullptr);

    EXPECT_NE(subGraph.GetTensor("t9"), nullptr);
    std::shared_ptr<LogicalTensor> tensor0 = subGraph.GetTensor("t9");
    std::shared_ptr<LogicalTensor> tensor1 = subGraph.GetTensor("t10");
    tensor0->tensor->rawshape = {128, 192};
    tensor1->tensor->rawshape = {128, 192};
    tensor1->memoryrange.memId = subGraph.GetTensor("t9")->memoryrange.memId;
    std::shared_ptr<LogicalTensor> tensor2 = subGraph.GetTensor("t11");
    tensor2->memoryrange.memId = subGraph.GetTensor("t9")->memoryrange.memId;
    tensor2->shape = {128, 128};
    tensor2->tensor->rawshape = {128, 192};
    std::shared_ptr<LogicalTensor> tensor3 = subGraph.GetTensor("t12");
    tensor3->memoryrange.memId = subGraph.GetTensor("t9")->memoryrange.memId;
    tensor3->shape = {128, 128};
    tensor3->tensor->rawshape = {128, 192};
    std::shared_ptr<LogicalTensor> tensor4 = subGraph.GetTensor("t13");
    tensor4->memoryrange.memId = subGraph.GetTensor("t9")->memoryrange.memId;
    tensor4->shape = {128, 192};
    tensor4->tensor->rawshape = {128, 192};
    std::shared_ptr<LogicalTensor> tensor5 = subGraph.GetTensor("t14");
    tensor5->memoryrange.memId = subGraph.GetTensor("t9")->memoryrange.memId;
    tensor5->shape = {128, 192};
    tensor5->tensor->rawshape = {128, 192};
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

    OptimizeSort sort(function->Operations().DuplicatedOpList(), *function);
    Status res = sort.SortOps();
    EXPECT_EQ(res, SUCCESS);
    OoOScheduler ooOScheduler(*function);
    res = ooOScheduler.Init(sort.operations);
    EXPECT_EQ(res, SUCCESS);
    res = ooOScheduler.GenSpillSchedule();
    EXPECT_EQ(res, SUCCESS);
}

TEST_F(ScheduleOoOTest, TestSchedule) {
    ComputationalGraphBuilder subGraph;
    std::vector<std::string> tensorNames{"t1", "t2", "t3", "t4", "t5", "t6", "t7", "t8", "t9"};
    std::vector<MemoryType> tensorMemTypes{MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_UB,
                                           MemoryType::MEM_UB,         MemoryType::MEM_UB,         MemoryType::MEM_UB,
                                           MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_UB,         MemoryType::MEM_UB};
    std::vector<Opcode> opCodes{Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC,
                                Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_COPY_IN,  Opcode::OP_COPY_IN,
                                Opcode::OP_ADD,      Opcode::OP_ADD,      Opcode::OP_ADD,      Opcode::OP_ADD,
                                Opcode::OP_COPY_OUT};
    std::vector<std::vector<std::string>> ioperands{
        {}, {}, {}, {}, {}, {}, {"t1"}, {"t2"}, {"t3", "t4"}, {"t3", "t4"}, {"t4", "t6"}, {"t5", "t8"}, {"t9"}};
    std::vector<std::vector<std::string>> ooperands{{"t3"}, {"t4"}, {"t5"}, {"t6"}, {"t8"}, {"t9"}, {"t3"},
                                                    {"t4"}, {"t5"}, {"t6"}, {"t8"}, {"t9"}, {"t7"}};
    std::vector<std::string> opNames{"Alloc1",  "Alloc2", "Alloc3", "Alloc4", "Alloc5", "Alloc6",  "Copyin1",
                                     "Copyin2", "Add1",   "Add2",   "Add3",   "Add4",   "Copyout1"};
    EXPECT_EQ(subGraph.AddTensors(DataType::DT_FP32, {64, 64}, tensorMemTypes, tensorNames, 0), true);
    EXPECT_EQ(subGraph.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    Function* function = subGraph.GetFunction();
    EXPECT_NE(function, nullptr);

    OptimizeSort sort(function->Operations().DuplicatedOpList(), *function);
    Status res = sort.SortOps();
    EXPECT_EQ(res, SUCCESS);
    OoOScheduler ooOScheduler(*function);
    res = ooOScheduler.Schedule(sort.operations);
    EXPECT_EQ(res, SUCCESS);
    Operation* add = subGraph.GetOp("Add2");
    EXPECT_NE(add, nullptr);
    EXPECT_EQ(add->oOperand[0]->memoryrange.start, 32768);
    EXPECT_EQ(add->oOperand[0]->memoryrange.end, 49152);
}

TEST_F(ScheduleOoOTest, TestScheduleInplace)
{
    ComputationalGraphBuilder subGraph;
    std::vector<std::string> tensorNames{"t1", "t2", "t3", "t4", "t5", "t6", "t7", "t8", "t9", "t10", "t11"};
    std::vector<MemoryType> tensorMemTypes{
        MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_DEVICE_DDR,
        MemoryType::MEM_UB,         MemoryType::MEM_UB,         MemoryType::MEM_UB,         MemoryType::MEM_UB,
        MemoryType::MEM_UB,         MemoryType::MEM_UB,         MemoryType::MEM_UB};
    std::vector<Opcode> opCodes{Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC,
                                Opcode::OP_UB_ALLOC, Opcode::OP_COPY_IN,  Opcode::OP_COPY_IN,  Opcode::OP_COPY_IN,
                                Opcode::OP_COPY_IN,  Opcode::OP_ADD,      Opcode::OP_ADD,      Opcode::OP_ADD};
    std::vector<std::vector<std::string>> ioperands{{},     {},     {},     {},           {},           {"t1"},
                                                    {"t2"}, {"t3"}, {"t4"}, {"t5", "t6"}, {"t7", "t8"}, {"t9", "t10"}};
    std::vector<std::vector<std::string>> ooperands{{"t5"}, {"t6"}, {"t7"}, {"t8"},  {"t9"}, {"t5"},
                                                    {"t6"}, {"t7"}, {"t8"}, {"t10"}, {"t9"}, {"t11"}};
    std::vector<std::string> opNames{"Alloc1",  "Alloc2",  "Alloc3",  "Alloc4", "Alloc5", "Copyin1",
                                     "Copyin2", "Copyin3", "Copyin4", "Add1",   "Add2",   "Add3"};
    EXPECT_EQ(subGraph.AddTensors(DataType::DT_FP32, {64, 64}, tensorMemTypes, tensorNames, 0), true);
    EXPECT_EQ(subGraph.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    Function* function = subGraph.GetFunction();
    EXPECT_NE(function, nullptr);

    EXPECT_NE(subGraph.GetTensor("t11"), nullptr);
    std::shared_ptr<LogicalTensor> tensor1 = subGraph.GetTensor("t10");
    tensor1->memoryrange.memId = subGraph.GetTensor("t5")->memoryrange.memId;
    std::shared_ptr<LogicalTensor> tensor2 = subGraph.GetTensor("t11");
    tensor2->memoryrange.memId = subGraph.GetTensor("t5")->memoryrange.memId;

    OptimizeSort sort(function->Operations().DuplicatedOpList(), *function);
    Status res = sort.SortOps();
    EXPECT_EQ(res, SUCCESS);
    OoOScheduler ooOScheduler(*function);
    res = ooOScheduler.Schedule(sort.operations);
    EXPECT_EQ(res, SUCCESS);
    Operation* copyin = subGraph.GetOp("Copyin1");
    EXPECT_NE(copyin, nullptr);
    Operation* add1 = subGraph.GetOp("Add1");
    EXPECT_NE(add1, nullptr);
    Operation* add3 = subGraph.GetOp("Add3");
    EXPECT_NE(add3, nullptr);
    EXPECT_EQ(copyin->oOperand[0]->memoryrange.start, 49152);
    EXPECT_EQ(copyin->oOperand[0]->memoryrange.end, 65536);
    EXPECT_EQ(add1->oOperand[0]->memoryrange.start, 49152);
    EXPECT_EQ(add1->oOperand[0]->memoryrange.end, 65536);
    EXPECT_EQ(add3->oOperand[0]->memoryrange.start, 49152);
    EXPECT_EQ(add3->oOperand[0]->memoryrange.end, 65536);
}

TEST_F(ScheduleOoOTest, TestScheduleView)
{
    ComputationalGraphBuilder subGraph;
    std::vector<std::string> tensorNames{"t1", "t2", "t3", "t4", "t5", "t6", "t7", "t8", "t9"};
    std::vector<MemoryType> tensorMemTypes{MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_UB,
                                           MemoryType::MEM_UB,         MemoryType::MEM_UB,         MemoryType::MEM_UB,
                                           MemoryType::MEM_UB,         MemoryType::MEM_UB,         MemoryType::MEM_UB};
    std::vector<Opcode> opCodes{Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC,
                                Opcode::OP_UB_ALLOC, Opcode::OP_COPY_IN,  Opcode::OP_COPY_IN,  Opcode::OP_VIEW,
                                Opcode::OP_VIEW,     Opcode::OP_ADD,      Opcode::OP_ADD,      Opcode::OP_ADD};
    std::vector<std::vector<std::string>> ioperands{{},     {},     {},     {},     {},     {"t1"},
                                                    {"t2"}, {"t3"}, {"t3"}, {"t5"}, {"t6"}, {"t4", "t7"}};
    std::vector<std::vector<std::string>> ooperands{{"t3"}, {"t4"}, {"t7"}, {"t8"}, {"t9"}, {"t3"},
                                                    {"t4"}, {"t5"}, {"t6"}, {"t7"}, {"t8"}, {"t9"}};
    std::vector<std::string> opNames{"Alloc1",  "Alloc2", "Alloc3", "Alloc4", "Alloc5", "Copyin1",
                                     "Copyin2", "View1",  "View2",  "Add1",   "Add2",   "Add3"};
    EXPECT_EQ(subGraph.AddTensors(DataType::DT_FP32, {64, 64}, tensorMemTypes, tensorNames, 0), true);
    EXPECT_EQ(subGraph.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    Function* function = subGraph.GetFunction();
    EXPECT_NE(function, nullptr);

    EXPECT_NE(subGraph.GetTensor("t6"), nullptr);
    std::shared_ptr<LogicalTensor> tensor1 = subGraph.GetTensor("t5");
    tensor1->memoryrange.memId = subGraph.GetTensor("t3")->memoryrange.memId;
    tensor1->shape = {32, 32};
    tensor1->tensor->rawshape = {64, 64};
    std::shared_ptr<LogicalTensor> tensor2 = subGraph.GetTensor("t6");
    tensor2->memoryrange.memId = subGraph.GetTensor("t3")->memoryrange.memId;
    tensor2->shape = {32, 32};
    tensor2->tensor->rawshape = {64, 64};

    OptimizeSort sort(function->Operations().DuplicatedOpList(), *function);
    Status res = sort.SortOps();
    EXPECT_EQ(res, SUCCESS);
    OoOScheduler ooOScheduler(*function);
    res = ooOScheduler.Schedule(sort.operations);
    EXPECT_EQ(res, SUCCESS);
    Operation* copyin = subGraph.GetOp("Copyin1");
    EXPECT_NE(copyin, nullptr);
    EXPECT_EQ(copyin->oOperand[0]->memoryrange.start, 0);
    EXPECT_EQ(copyin->oOperand[0]->memoryrange.end, 16384);
    EXPECT_EQ(subGraph.GetOp("View1")->GetOutputOperand(0)->memoryrange.start, 0);
    EXPECT_EQ(subGraph.GetOp("View1")->GetOutputOperand(0)->memoryrange.end, 16384);
    EXPECT_EQ(subGraph.GetOp("View2")->GetOutputOperand(0)->memoryrange.start, 0);
    EXPECT_EQ(subGraph.GetOp("View2")->GetOutputOperand(0)->memoryrange.end, 16384);
}

TEST_F(ScheduleOoOTest, TestScheduleAssemble)
{
    ComputationalGraphBuilder subGraph;
    std::vector<std::string> tensorNames{"t1", "t2", "t3", "t4", "t5", "t6", "t7", "t8", "t9", "t10", "t11"};
    std::vector<MemoryType> tensorMemTypes{
        MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_DEVICE_DDR,
        MemoryType::MEM_UB,         MemoryType::MEM_UB,         MemoryType::MEM_UB,         MemoryType::MEM_UB,
        MemoryType::MEM_UB,         MemoryType::MEM_UB,         MemoryType::MEM_UB};
    std::vector<Opcode> opCodes{Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC,
                                Opcode::OP_UB_ALLOC, Opcode::OP_COPY_IN,  Opcode::OP_COPY_IN,  Opcode::OP_COPY_IN,
                                Opcode::OP_COPY_IN,  Opcode::OP_ASSEMBLE, Opcode::OP_ASSEMBLE, Opcode::OP_ADD,
                                Opcode::OP_ADD};
    std::vector<std::vector<std::string>> ioperands{{},     {},     {},     {},     {},           {"t1"},       {"t2"},
                                                    {"t3"}, {"t4"}, {"t5"}, {"t6"}, {"t7", "t8"}, {"t9", "t10"}};
    std::vector<std::vector<std::string>> ooperands{{"t5"}, {"t7"}, {"t8"}, {"t10"}, {"t11"}, {"t5"}, {"t6"},
                                                    {"t7"}, {"t8"}, {"t9"}, {"t9"},  {"t10"}, {"t11"}};
    std::vector<std::string> opNames{"Alloc1",  "Alloc2",  "Alloc3",    "Alloc4",    "Alloc5", "Copyin1", "Copyin2",
                                     "Copyin3", "Copyin4", "Assemble1", "Assemble2", "Add1",   "Add2"};
    EXPECT_EQ(subGraph.AddTensors(DataType::DT_FP32, {64, 64}, tensorMemTypes, tensorNames, 0), true);
    EXPECT_EQ(subGraph.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    Function* function = subGraph.GetFunction();
    EXPECT_NE(function, nullptr);

    EXPECT_NE(subGraph.GetTensor("t9"), nullptr);
    std::shared_ptr<LogicalTensor> tensor1 = subGraph.GetTensor("t9");
    tensor1->memoryrange.memId = subGraph.GetTensor("t5")->memoryrange.memId;
    std::shared_ptr<LogicalTensor> tensor2 = subGraph.GetTensor("t6");
    tensor2->memoryrange.memId = subGraph.GetTensor("t5")->memoryrange.memId;
    tensor2->shape = {32, 32};
    tensor2->tensor->rawshape = {64, 64};
    std::shared_ptr<LogicalTensor> tensor3 = subGraph.GetTensor("t5");
    tensor3->shape = {32, 32};
    tensor3->tensor->rawshape = {64, 64};

    OptimizeSort sort(function->Operations().DuplicatedOpList(), *function);
    Status res = sort.SortOps();
    EXPECT_EQ(res, SUCCESS);
    OoOScheduler ooOScheduler(*function);
    res = ooOScheduler.Schedule(sort.operations);
    EXPECT_EQ(res, SUCCESS);
    Operation* copyin1 = subGraph.GetOp("Copyin1");
    EXPECT_NE(copyin1, nullptr);
    Operation* copyin2 = subGraph.GetOp("Copyin2");
    EXPECT_NE(copyin2, nullptr);
    Operation* assemble1 = subGraph.GetOp("Assemble1");
    EXPECT_NE(assemble1, nullptr);
    Operation* assemble2 = subGraph.GetOp("Assemble2");
    EXPECT_NE(assemble2, nullptr);
    EXPECT_EQ(copyin1->oOperand[0]->memoryrange.start, 0);
    EXPECT_EQ(copyin1->oOperand[0]->memoryrange.end, 16384);
    EXPECT_EQ(copyin2->oOperand[0]->memoryrange.start, 0);
    EXPECT_EQ(copyin2->oOperand[0]->memoryrange.end, 16384);
    EXPECT_EQ(assemble1->oOperand[0]->memoryrange.start, 0);
    EXPECT_EQ(assemble1->oOperand[0]->memoryrange.end, 16384);
    EXPECT_EQ(assemble2->oOperand[0]->memoryrange.start, 0);
    EXPECT_EQ(assemble2->oOperand[0]->memoryrange.end, 16384);
}

TEST_F(ScheduleOoOTest, TestScheduleSpillCopyIn)
{
    ComputationalGraphBuilder subGraph;
    std::vector<std::string> tensorNames{"t1", "t2", "t3", "t4", "t5", "t6", "t7", "t8"};
    std::vector<MemoryType> tensorMemTypes{MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_UB,
                                           MemoryType::MEM_UB,         MemoryType::MEM_UB,         MemoryType::MEM_UB,
                                           MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_UB};
    std::vector<Opcode> opCodes{Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC,
                                Opcode::OP_UB_ALLOC, Opcode::OP_COPY_IN,  Opcode::OP_COPY_IN,  Opcode::OP_ADD,
                                Opcode::OP_ADD,      Opcode::OP_ADD,      Opcode::OP_COPY_OUT};
    std::vector<std::vector<std::string>> ioperands{{},     {},           {},           {},           {},    {"t1"},
                                                    {"t2"}, {"t3", "t4"}, {"t3", "t5"}, {"t4", "t6"}, {"t8"}};
    std::vector<std::vector<std::string>> ooperands{{"t3"}, {"t4"}, {"t5"}, {"t6"}, {"t8"}, {"t3"},
                                                    {"t4"}, {"t5"}, {"t6"}, {"t8"}, {"t7"}};
    std::vector<std::string> opNames{"Alloc1",  "Alloc2", "Alloc3", "Alloc4", "Alloc5",  "Copyin1",
                                     "Copyin2", "Add1",   "Add2",   "Add3",   "Copyout1"};
    EXPECT_EQ(subGraph.AddTensors(DataType::DT_FP32, {128, 128}, tensorMemTypes, tensorNames, 0), true);
    EXPECT_EQ(subGraph.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    Function* function = subGraph.GetFunction();
    EXPECT_NE(function, nullptr);

    OptimizeSort sort(function->Operations().DuplicatedOpList(), *function);
    Status res = sort.SortOps();
    EXPECT_EQ(res, SUCCESS);
    OoOScheduler ooOScheduler(*function);
    res = ooOScheduler.Init(sort.operations);
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

TEST_F(ScheduleOoOTest, TestScheduleSpill)
{
    ComputationalGraphBuilder subGraph;
    std::vector<std::string> tensorNames{"t1", "t2", "t3", "t4", "t5", "t6", "t7", "t8", "t9"};
    std::vector<MemoryType> tensorMemTypes{MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_UB,
                                           MemoryType::MEM_UB,         MemoryType::MEM_UB,         MemoryType::MEM_UB,
                                           MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_UB,         MemoryType::MEM_UB};
    std::vector<Opcode> opCodes{Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC,
                                Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_COPY_IN,  Opcode::OP_COPY_IN,
                                Opcode::OP_ADD,      Opcode::OP_ADD,      Opcode::OP_ADD,      Opcode::OP_ADD,
                                Opcode::OP_COPY_OUT};
    std::vector<std::vector<std::string>> ioperands{
        {}, {}, {}, {}, {}, {}, {"t1"}, {"t2"}, {"t3", "t4"}, {"t3", "t4"}, {"t4", "t6"}, {"t5", "t8"}, {"t9"}};
    std::vector<std::vector<std::string>> ooperands{{"t3"}, {"t4"}, {"t5"}, {"t6"}, {"t8"}, {"t9"}, {"t3"},
                                                    {"t4"}, {"t5"}, {"t6"}, {"t8"}, {"t9"}, {"t7"}};
    std::vector<std::string> opNames{"Alloc1",  "Alloc2", "Alloc3", "Alloc4", "Alloc5", "Alloc6",  "Copyin1",
                                     "Copyin2", "Add1",   "Add2",   "Add3",   "Add4",   "Copyout1"};
    EXPECT_EQ(subGraph.AddTensors(DataType::DT_FP32, {128, 128}, tensorMemTypes, tensorNames, 0), true);
    EXPECT_EQ(subGraph.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    Function* function = subGraph.GetFunction();
    EXPECT_NE(function, nullptr);

    OptimizeSort sort(function->Operations().DuplicatedOpList(), *function);
    Status res = sort.SortOps();
    EXPECT_EQ(res, SUCCESS);
    OoOScheduler ooOScheduler(*function);
    res = ooOScheduler.Init(sort.operations);
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

TEST_F(ScheduleOoOTest, TestScheduleSpillInplace)
{
    ComputationalGraphBuilder subGraph;
    std::vector<std::string> tensorNames{"t1", "t2", "t3", "t4", "t5", "t6", "t7", "t8", "t9", "t10", "t11"};
    std::vector<MemoryType> tensorMemTypes{
        MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_DEVICE_DDR,
        MemoryType::MEM_UB,         MemoryType::MEM_UB,         MemoryType::MEM_UB,         MemoryType::MEM_UB,
        MemoryType::MEM_UB,         MemoryType::MEM_UB,         MemoryType::MEM_UB};
    std::vector<Opcode> opCodes{Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC,
                                Opcode::OP_UB_ALLOC, Opcode::OP_COPY_IN,  Opcode::OP_COPY_IN,  Opcode::OP_COPY_IN,
                                Opcode::OP_COPY_IN,  Opcode::OP_ADD,      Opcode::OP_ADD,      Opcode::OP_ADD};
    std::vector<std::vector<std::string>> ioperands{{},     {},     {},     {},           {},           {"t1"},
                                                    {"t2"}, {"t3"}, {"t4"}, {"t5", "t6"}, {"t7", "t8"}, {"t9", "t10"}};
    std::vector<std::vector<std::string>> ooperands{{"t5"}, {"t6"}, {"t7"}, {"t8"},  {"t9"}, {"t5"},
                                                    {"t6"}, {"t7"}, {"t8"}, {"t10"}, {"t9"}, {"t11"}};
    std::vector<std::string> opNames{"Alloc1",  "Alloc2",  "Alloc3",  "Alloc4", "Alloc5", "Copyin1",
                                     "Copyin2", "Copyin3", "Copyin4", "Add1",   "Add2",   "Add3"};
    EXPECT_EQ(subGraph.AddTensors(DataType::DT_FP32, {128, 128}, tensorMemTypes, tensorNames, 0), true);
    EXPECT_EQ(subGraph.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    Function* function = subGraph.GetFunction();
    EXPECT_NE(function, nullptr);

    EXPECT_NE(subGraph.GetTensor("t11"), nullptr);
    std::shared_ptr<LogicalTensor> tensor1 = subGraph.GetTensor("t10");
    tensor1->memoryrange.memId = subGraph.GetTensor("t5")->memoryrange.memId;
    std::shared_ptr<LogicalTensor> tensor2 = subGraph.GetTensor("t11");
    tensor2->memoryrange.memId = subGraph.GetTensor("t5")->memoryrange.memId;

    OptimizeSort sort(function->Operations().DuplicatedOpList(), *function);
    Status res = sort.SortOps();
    EXPECT_EQ(res, SUCCESS);
    OoOScheduler ooOScheduler(*function);
    res = ooOScheduler.Init(sort.operations);
    EXPECT_EQ(res, SUCCESS);
    std::rotate(ooOScheduler.orderedOps.begin(), ooOScheduler.orderedOps.begin() + 6, ooOScheduler.orderedOps.begin() + 11);
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

TEST_F(ScheduleOoOTest, TestScheduleSpillView)
{
    ComputationalGraphBuilder subGraph;
    std::vector<std::string> tensorNames{"t1", "t2", "t3", "t4", "t5", "t6", "t7", "t8", "t9"};
    std::vector<MemoryType> tensorMemTypes{MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_UB,
                                           MemoryType::MEM_UB,         MemoryType::MEM_UB,         MemoryType::MEM_UB,
                                           MemoryType::MEM_UB,         MemoryType::MEM_UB,         MemoryType::MEM_UB};
    std::vector<Opcode> opCodes{Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC,
                                Opcode::OP_UB_ALLOC, Opcode::OP_COPY_IN,  Opcode::OP_COPY_IN,  Opcode::OP_VIEW,
                                Opcode::OP_VIEW,     Opcode::OP_ADD,      Opcode::OP_ADD,      Opcode::OP_ADD};
    std::vector<std::vector<std::string>> ioperands{{},     {},     {},     {},     {},     {"t1"},
                                                    {"t2"}, {"t3"}, {"t3"}, {"t5"}, {"t6"}, {"t4", "t7"}};
    std::vector<std::vector<std::string>> ooperands{{"t3"}, {"t4"}, {"t7"}, {"t8"}, {"t9"}, {"t3"},
                                                    {"t4"}, {"t5"}, {"t6"}, {"t7"}, {"t8"}, {"t9"}};
    std::vector<std::string> opNames{"Alloc1",  "Alloc2", "Alloc3", "Alloc4", "Alloc5", "Copyin1",
                                     "Copyin2", "View1",  "View2",  "Add1",   "Add2",   "Add3"};
    EXPECT_EQ(subGraph.AddTensors(DataType::DT_FP32, {128, 128}, tensorMemTypes, tensorNames, 0), true);
    EXPECT_EQ(subGraph.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    Function* function = subGraph.GetFunction();
    EXPECT_NE(function, nullptr);

    EXPECT_NE(subGraph.GetTensor("t6"), nullptr);
    std::shared_ptr<LogicalTensor> tensor1 = subGraph.GetTensor("t5");
    tensor1->memoryrange.memId = subGraph.GetTensor("t3")->memoryrange.memId;
    std::shared_ptr<LogicalTensor> tensor2 = subGraph.GetTensor("t6");
    tensor2->memoryrange.memId = subGraph.GetTensor("t3")->memoryrange.memId;

    OptimizeSort sort(function->Operations().DuplicatedOpList(), *function);
    Status res = sort.SortOps();
    EXPECT_EQ(res, SUCCESS);
    OoOScheduler ooOScheduler(*function);
    ooOScheduler.Init(sort.operations);
    EXPECT_EQ(res, SUCCESS);
    res = ooOScheduler.ScheduleMainLoop();
    EXPECT_EQ(res, SUCCESS);
}

TEST_F(ScheduleOoOTest, TestScheduleSpillAssemble)
{
    ComputationalGraphBuilder subGraph;
    std::vector<std::string> tensorNames{"t1", "t2", "t3", "t4", "t5", "t6", "t7", "t8", "t9", "t10", "t11"};
    std::vector<MemoryType> tensorMemTypes{
        MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_DEVICE_DDR,
        MemoryType::MEM_UB,         MemoryType::MEM_UB,         MemoryType::MEM_UB,         MemoryType::MEM_UB,
        MemoryType::MEM_UB,         MemoryType::MEM_UB,         MemoryType::MEM_UB};
    std::vector<Opcode> opCodes{Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC,
                                Opcode::OP_UB_ALLOC, Opcode::OP_COPY_IN,  Opcode::OP_COPY_IN,  Opcode::OP_COPY_IN,
                                Opcode::OP_COPY_IN,  Opcode::OP_ASSEMBLE, Opcode::OP_ASSEMBLE, Opcode::OP_ADD,
                                Opcode::OP_ADD};
    std::vector<std::vector<std::string>> ioperands{{},     {},     {},     {},     {},           {"t1"},       {"t2"},
                                                    {"t3"}, {"t4"}, {"t5"}, {"t6"}, {"t7", "t8"}, {"t9", "t10"}};
    std::vector<std::vector<std::string>> ooperands{{"t5"}, {"t7"}, {"t8"}, {"t10"}, {"t11"}, {"t5"}, {"t6"},
                                                    {"t7"}, {"t8"}, {"t9"}, {"t9"},  {"t10"}, {"t11"}};
    std::vector<std::string> opNames{"Alloc1",  "Alloc2",  "Alloc3",    "Alloc4",    "Alloc5", "Copyin1", "Copyin2",
                                     "Copyin3", "Copyin4", "Assemble1", "Assemble2", "Add1",   "Add2"};
    EXPECT_EQ(subGraph.AddTensors(DataType::DT_FP32, {128, 128}, tensorMemTypes, tensorNames, 0), true);
    EXPECT_EQ(subGraph.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    Function* function = subGraph.GetFunction();
    EXPECT_NE(function, nullptr);

    EXPECT_NE(subGraph.GetTensor("t9"), nullptr);
    std::shared_ptr<LogicalTensor> tensor1 = subGraph.GetTensor("t9");
    tensor1->memoryrange.memId = subGraph.GetTensor("t5")->memoryrange.memId;
    std::shared_ptr<LogicalTensor> tensor2 = subGraph.GetTensor("t6");
    tensor2->memoryrange.memId = subGraph.GetTensor("t5")->memoryrange.memId;
    std::shared_ptr<LogicalTensor> tensor5 = subGraph.GetTensor("t5");
    std::shared_ptr<LogicalTensor> tensor6 = subGraph.GetTensor("t6");
    tensor5->shape = {64, 128};
    tensor5->tensor->rawshape = {128, 128};
    tensor6->shape = {64, 128};
    tensor6->tensor->rawshape = {128, 128};
    std::vector<int64_t> offset1 = {0, 0};
    std::vector<int64_t> offset2 = {64, 0};
    auto assembleAttr1 = std::make_shared<AssembleOpAttribute>(MemoryType::MEM_UB, offset1);
    auto assemble1 = subGraph.GetOp("Assemble1");
    assemble1->SetOpAttribute(assembleAttr1);
    auto assembleAttr2 = std::make_shared<AssembleOpAttribute>(MemoryType::MEM_UB, offset2);
    auto assemble2 = subGraph.GetOp("Assemble2");
    assemble2->SetOpAttribute(assembleAttr2);

    OptimizeSort sort(function->Operations().DuplicatedOpList(), *function);
    Status res = sort.SortOps();
    EXPECT_EQ(res, SUCCESS);
    OoOScheduler ooOScheduler(*function);
    ooOScheduler.Init(sort.operations);
    EXPECT_EQ(res, SUCCESS);
    res = ooOScheduler.ScheduleMainLoop();
    EXPECT_EQ(res, SUCCESS);
}

TEST_F(ScheduleOoOTest, TestScheduleSpillWithInplaceView)
{
    ComputationalGraphBuilder subGraph;
    std::vector<std::string> tensorNames{"t1", "t2", "t3", "t4", "t5", "t6", "t7", "t8", "t9", "t10", "t11", "t12"};
    std::vector<MemoryType> tensorMemTypes{
        MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_DEVICE_DDR,
        MemoryType::MEM_UB,         MemoryType::MEM_UB,         MemoryType::MEM_UB,         MemoryType::MEM_UB,
        MemoryType::MEM_UB,         MemoryType::MEM_UB,         MemoryType::MEM_UB,         MemoryType::MEM_UB};
    std::vector<Opcode> opCodes{Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC,
                                Opcode::OP_UB_ALLOC, Opcode::OP_COPY_IN,  Opcode::OP_COPY_IN,  Opcode::OP_COPY_IN,
                                Opcode::OP_COPY_IN,  Opcode::OP_ADD,      Opcode::OP_ADD,      Opcode::OP_ADD, 
                                Opcode::OP_VIEW};
    std::vector<std::vector<std::string>> ioperands{{},     {},     {},     {},           {},           {"t1"},
                                                    {"t2"}, {"t3"}, {"t4"}, {"t5", "t6"}, {"t7", "t8"}, {"t9", "t10"}, {"t11"}};
    std::vector<std::vector<std::string>> ooperands{{"t5"}, {"t6"}, {"t7"}, {"t8"},  {"t9"}, {"t5"},
                                                    {"t6"}, {"t7"}, {"t8"}, {"t10"}, {"t9"}, {"t11"}, {"t12"}};
    std::vector<std::string> opNames{"Alloc1",  "Alloc2",  "Alloc3",  "Alloc4", "Alloc5", "Copyin1",
                                     "Copyin2", "Copyin3", "Copyin4", "Add1",   "Add2",   "Add3", "View1"};
    EXPECT_EQ(subGraph.AddTensors(DataType::DT_FP32, {128, 128}, tensorMemTypes, tensorNames, 0), true);
    EXPECT_EQ(subGraph.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    Function* function = subGraph.GetFunction();
    EXPECT_NE(function, nullptr);

    EXPECT_NE(subGraph.GetTensor("t11"), nullptr);
    std::shared_ptr<LogicalTensor> tensor1 = subGraph.GetTensor("t10");
    tensor1->memoryrange.memId = subGraph.GetTensor("t5")->memoryrange.memId;
    std::shared_ptr<LogicalTensor> tensor2 = subGraph.GetTensor("t11");
    tensor2->memoryrange.memId = subGraph.GetTensor("t5")->memoryrange.memId;
    std::shared_ptr<LogicalTensor> tensor3 = subGraph.GetTensor("t12");
    tensor3->memoryrange.memId = subGraph.GetTensor("t5")->memoryrange.memId;

    std::vector<int64_t> offset1 = {0, 0};
    auto viewAttr1 = std::make_shared<ViewOpAttribute>(offset1, MemoryType::MEM_UB);
    auto view1 = subGraph.GetOp("View1");
    view1->SetOpAttribute(viewAttr1);
    auto inputTensor = view1->GetIOperands()[0];
    auto outputTensor = view1->GetOOperands()[0];
    EXPECT_EQ(outputTensor->memoryrange.memId, inputTensor->memoryrange.memId);
    OptimizeSort sort(function->Operations().DuplicatedOpList(), *function);
    Status res = sort.SortOps();
    EXPECT_EQ(res, SUCCESS);
    OoOScheduler ooOScheduler(*function);
    res = ooOScheduler.Init(sort.operations);
    EXPECT_EQ(res, SUCCESS);
    std::rotate(ooOScheduler.orderedOps.begin(), ooOScheduler.orderedOps.begin() + 6, ooOScheduler.orderedOps.begin() + 11);
    res = ooOScheduler.ScheduleMainLoop();
    EXPECT_EQ(res, SUCCESS);
    // The tensor's memId remains the same before and after the view operation following a spill.
    EXPECT_EQ(outputTensor->memoryrange.memId, inputTensor->memoryrange.memId);
}


TEST_F(ScheduleOoOTest, TestScheduleSpillFragFailed)
{
    ComputationalGraphBuilder subGraph;
    std::vector<std::string> tensorNames{"t1", "t2", "t3", "t4", "t5", "t6"};
    std::vector<MemoryType> tensorMemTypes{MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_UB,
                                           MemoryType::MEM_UB,         MemoryType::MEM_UB,         MemoryType::MEM_UB};
    std::vector<Opcode> opCodes{Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC,
                                Opcode::OP_COPY_IN,  Opcode::OP_COPY_IN,  Opcode::OP_ADD};
    std::vector<std::vector<std::string>> ioperands{{}, {}, {}, {}, {"t1"}, {"t2"}, {"t4", "t5"}};
    std::vector<std::vector<std::string>> ooperands{{"t3"}, {"t4"}, {"t5"}, {"t6"}, {"t3", "t4"}, {"t5"}, {"t6"}};
    std::vector<std::string> opNames{"Alloc1", "Alloc2", "Alloc3", "Alloc4", "Copyin1", "Copyin2", "Add"};
    EXPECT_EQ(subGraph.AddTensors(DataType::DT_FP32, {128, 128}, tensorMemTypes, tensorNames, 0), true);
    EXPECT_EQ(subGraph.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    Function* function = subGraph.GetFunction();
    EXPECT_NE(function, nullptr);

    EXPECT_NE(subGraph.GetTensor("t3"), nullptr);
    std::shared_ptr<LogicalTensor> tensor = subGraph.GetTensor("t3");
    tensor->shape = {32, 32};
    tensor->tensor->rawshape = {32, 32};

    OptimizeSort sort(function->Operations().DuplicatedOpList(), *function);
    Status res = sort.SortOps();
    EXPECT_EQ(res, SUCCESS);
    OoOScheduler ooOScheduler(*function);
    ooOScheduler.Init(sort.operations);
    EXPECT_EQ(res, SUCCESS);
    res = ooOScheduler.ScheduleMainLoop();
    EXPECT_EQ(res, SUCCESS);
}

TEST_F(ScheduleOoOTest, TestEmptyOplist)
{
    Function function(Program::GetInstance(), "", "", nullptr);
    std::vector<Operation*> scheduleOpList;
    OoOScheduler ooOScheduler(function);
    Status res = ooOScheduler.Schedule(scheduleOpList);
    EXPECT_EQ(res, SUCCESS);
}

TEST_F(ScheduleOoOTest, TestScheduleReshape)
{
    ComputationalGraphBuilder subGraph;
    std::vector<std::string> tensorNames{"t1", "t2"};
    std::vector<MemoryType> tensorMemTypes{MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_DEVICE_DDR};
    std::vector<Opcode> opCodes{Opcode::OP_RESHAPE};
    std::vector<std::vector<std::string>> ioperands{{"t1"}};
    std::vector<std::vector<std::string>> ooperands{{"t2"}};
    std::vector<std::string> opNames{"Reshape1"};
    EXPECT_EQ(subGraph.AddTensors(DataType::DT_FP32, {128, 128}, tensorMemTypes, tensorNames, 0), true);
    EXPECT_EQ(subGraph.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    Function* function = subGraph.GetFunction();
    EXPECT_NE(function, nullptr);

    OoOScheduler ooOScheduler(*function);
    Status res = ooOScheduler.Schedule(function->Operations().DuplicatedOpList());
    EXPECT_EQ(res, SUCCESS);
}

TEST_F(ScheduleOoOTest, TestSingleCopyin1)
{
    ComputationalGraphBuilder subGraph;
    std::vector<std::string> tensorNames{"t1", "t2"};
    std::vector<MemoryType> tensorMemTypes{MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_UB};
    std::vector<Opcode> opCodes{Opcode::OP_COPY_IN};
    std::vector<std::vector<std::string>> ioperands{{"t1"}};
    std::vector<std::vector<std::string>> ooperands{{"t2"}};
    std::vector<std::string> opNames{"Copyin1"};
    EXPECT_EQ(subGraph.AddTensors(DataType::DT_FP32, {128, 128}, tensorMemTypes, tensorNames, 0), true);
    EXPECT_EQ(subGraph.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    Function* function = subGraph.GetFunction();
    EXPECT_NE(function, nullptr);

    OoOScheduler ooOScheduler(*function);
    Status res = ooOScheduler.Schedule(function->Operations().DuplicatedOpList());
    EXPECT_EQ(res, FAILED);
}

TEST_F(ScheduleOoOTest, TestSingleCopyin2)
{
    ComputationalGraphBuilder subGraph;
    std::vector<std::string> tensorNames{"t1", "t2"};
    std::vector<MemoryType> tensorMemTypes{MemoryType::MEM_UB, MemoryType::MEM_UB};
    std::vector<Opcode> opCodes{Opcode::OP_UB_ALLOC, Opcode::OP_COPY_IN};
    std::vector<std::vector<std::string>> ioperands{{}, {"t1"}};
    std::vector<std::vector<std::string>> ooperands{{"t2"}, {"t2"}};
    std::vector<std::string> opNames{"Alloc1", "Copyin1"};
    EXPECT_EQ(subGraph.AddTensors(DataType::DT_FP32, {128, 128}, tensorMemTypes, tensorNames, 0), true);
    EXPECT_EQ(subGraph.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    Function* function = subGraph.GetFunction();
    EXPECT_NE(function, nullptr);

    OoOScheduler ooOScheduler(*function);
    Status res = ooOScheduler.Schedule(function->Operations().DuplicatedOpList());
    EXPECT_EQ(res, FAILED);
}

TEST_F(ScheduleOoOTest, TestDelBufCount)
{
    Function function(Program::GetInstance(), "", "", nullptr);
    OoOScheduler oooSchedule(function);
    oooSchedule.DelBufRefCount(-1);
}

TEST_F(ScheduleOoOTest, TestDelBufCount_1)
{
    Function function(Program::GetInstance(), "", "", nullptr);
    OoOScheduler oooSchedule(function);
    oooSchedule.bufRefCount_[1] = -1;
    oooSchedule.DelBufRefCount(1);
}

TEST_F(ScheduleOoOTest, TestUpdateTensorAttr_DDR)
{
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

TEST_F(ScheduleOoOTest, TestUpdateTensorAttr_UB)
{
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

TEST_F(ScheduleOoOTest, TestGetSpillTensor)
{
    Function function(Program::GetInstance(), "", "", nullptr);
    std::vector<Operation*> scheduleOpList;

    std::vector<int64_t> shape = {128, 128};
    std::shared_ptr<LogicalTensor> tensor3 = std::make_shared<LogicalTensor>(function, DataType::DT_FP32, shape);
    tensor3->SetMemoryTypeOriginal(MEM_UB);
    tensor3->SetMemoryTypeToBe(MEM_UB);
    tensor3->memoryrange.memId = 3;

    auto& alloc1 =
        function.AddOperation(Opcode::OP_UB_ALLOC, {}, std::vector<std::shared_ptr<LogicalTensor>>({tensor3}));
    alloc1.UpdateLatency(1);

    LogicalTensorPtr tensor = nullptr;

    OoOScheduler oooSchedule(function);
    oooSchedule.GetSpillTensor(&alloc1, 1, tensor);
}

TEST_F(ScheduleOoOTest, TestCheckAllocIssue)
{
    Function function(Program::GetInstance(), "", "", nullptr);
    std::vector<Operation*> scheduleOpList;

    std::vector<int64_t> shape = {128, 128};
    std::shared_ptr<LogicalTensor> tensor3 = std::make_shared<LogicalTensor>(function, DataType::DT_FP32, shape);
    tensor3->SetMemoryTypeOriginal(MEM_UB);
    tensor3->SetMemoryTypeToBe(MEM_UB);
    tensor3->memoryrange.memId = 3;

    std::shared_ptr<LogicalTensor> tensor2 = std::make_shared<LogicalTensor>(function, DataType::DT_FP32, shape);
    tensor3->SetMemoryTypeOriginal(MEM_UB);
    tensor3->SetMemoryTypeToBe(MEM_UB);
    tensor3->memoryrange.memId = 1;

    auto& alloc1 =
        function.AddOperation(Opcode::OP_UB_ALLOC, {}, std::vector<std::shared_ptr<LogicalTensor>>({tensor3, tensor2}));
    alloc1.UpdateLatency(1);

    OoOScheduler oooSchedule(function);
    oooSchedule.Init(function.Operations().DuplicatedOpList());
}

TEST_F(ScheduleOoOTest, TestBufferUsage)
{
    ComputationalGraphBuilder subGraph;
    std::vector<std::string> tensorNames{"t1", "t2", "t3", "t4", "t5", "t6"};
    std::vector<MemoryType> tensorMemTypes{MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_UB,
                                           MemoryType::MEM_UB,         MemoryType::MEM_UB,         MemoryType::MEM_UB};
    std::vector<Opcode> opCodes{Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC,
                                Opcode::OP_COPY_IN,  Opcode::OP_COPY_IN,  Opcode::OP_ADD};
    std::vector<std::vector<std::string>> ioperands{{}, {}, {}, {}, {"t1"}, {"t2"}, {"t4", "t5"}};
    std::vector<std::vector<std::string>> ooperands{{"t3"}, {"t4"}, {"t5"}, {"t6"}, {"t3", "t4"}, {"t5"}, {"t6"}};
    std::vector<std::string> opNames{"Alloc1", "Alloc2", "Alloc3", "Alloc4", "Copyin1", "Copyin2", "Add1"};
    EXPECT_EQ(subGraph.AddTensors(DataType::DT_FP32, {128, 128}, tensorMemTypes, tensorNames, 0), true);
    EXPECT_EQ(subGraph.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    Function* function = subGraph.GetFunction();
    EXPECT_NE(function, nullptr);

    OptimizeSort sort(function->Operations().DuplicatedOpList(), *function);
    Status res = sort.SortOps();
    EXPECT_EQ(res, SUCCESS);
    OoOScheduler ooOScheduler(*function);
    res = ooOScheduler.Init(sort.operations);
    ooOScheduler.oooCheck.doHealthCheck = true;
    EXPECT_EQ(res, SUCCESS);
    res = ooOScheduler.ScheduleMainLoop();
    EXPECT_EQ(res, SUCCESS);
    std::unordered_map<MemoryType, uint64_t> invalidBufferTotalUsage = {
        {MemoryType::MEM_UB, 0},
        {MemoryType::MEM_L1, 0},
        {MemoryType::MEM_L0A, 0},
        {MemoryType::MEM_L0B, 0},
        {MemoryType::MEM_L0C, 0}};
    std::unordered_map<MemoryType, uint64_t> invalidBufferMaxUsage = {
        {MemoryType::MEM_UB, 0},
        {MemoryType::MEM_L1, 0},
        {MemoryType::MEM_L0A, 0},
        {MemoryType::MEM_L0B, 0},
        {MemoryType::MEM_L0C, 0}};
    EXPECT_NE(ooOScheduler.oooCheck.bufferTotalUsage, invalidBufferTotalUsage);
    EXPECT_NE(ooOScheduler.oooCheck.bufferMaxUsage, invalidBufferMaxUsage);

    // 增加健康检查校验
    ooOScheduler.oooCheck.clock = 3; // 模拟数据
    res = ooOScheduler.oooCheck.HealthCheckOoOSchedule();
    EXPECT_EQ(res, SUCCESS);
    EXPECT_NE(ooOScheduler.oooCheck.report, nullptr);
}

TEST_F(ScheduleOoOTest, TestScheduleGenSpillInfiniteLoop)
{
    ComputationalGraphBuilder subGraph;
    std::vector<std::string> tensorNames{"t1", "t2", "t3", "t4", "t5", "t6", "t7"};
    std::vector<MemoryType> tensorMemTypes{MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_UB,
                                           MemoryType::MEM_UB,         MemoryType::MEM_UB,         MemoryType::MEM_UB,
                                           MemoryType::MEM_UB};
    std::vector<Opcode> opCodes{Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC,
                                Opcode::OP_UB_ALLOC, Opcode::OP_COPY_IN,  Opcode::OP_COPY_IN,  Opcode::OP_SUB,
                                Opcode::OP_ADD,      Opcode::OP_ADD};
    std::vector<std::vector<std::string>> ioperands{{},     {},     {},     {},           {},
                                                    {"t1"}, {"t2"}, {"t3"}, {"t4", "t5"}, {"t3", "t6"}};
    std::vector<std::vector<std::string>> ooperands{{"t3"}, {"t4"}, {"t5"}, {"t6"}, {"t7"},
                                                    {"t3"}, {"t4"}, {"t5"}, {"t6"}, {"t7"}};
    std::vector<std::string> opNames{"Alloc1",  "Alloc2",  "Alloc3", "Alloc4", "Alloc5",
                                     "Copyin1", "Copyin2", "Sub1",   "Add1",   "Add2"};
    EXPECT_EQ(subGraph.AddTensors(DataType::DT_FP16, {128, 128}, tensorMemTypes, tensorNames, 0), true);
    EXPECT_EQ(subGraph.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    Function* function = subGraph.GetFunction();
    EXPECT_NE(function, nullptr);

    EXPECT_NE(subGraph.GetTensor("t3"), nullptr);
    std::shared_ptr<LogicalTensor> tensor = subGraph.GetTensor("t3");
    tensor->shape = {80, 128};
    tensor->tensor->rawshape = {80, 128};

    EXPECT_NE(subGraph.GetTensor("t4"), nullptr);
    std::shared_ptr<LogicalTensor> tensor1 = subGraph.GetTensor("t4");
    tensor1->shape = {176, 256};
    tensor1->tensor->rawshape = {176, 256};

    EXPECT_NE(subGraph.GetTensor("t5"), nullptr);
    std::shared_ptr<LogicalTensor> tensor2 = subGraph.GetTensor("t5");
    tensor2->shape = {176, 256};
    tensor2->tensor->rawshape = {176, 256};

    EXPECT_NE(subGraph.GetTensor("t6"), nullptr);
    std::shared_ptr<LogicalTensor> tensor3 = subGraph.GetTensor("t6");
    tensor3->shape = {64, 128};
    tensor3->tensor->rawshape = {64, 128};

    EXPECT_NE(subGraph.GetTensor("t7"), nullptr);
    std::shared_ptr<LogicalTensor> tensor4 = subGraph.GetTensor("t7");
    tensor4->shape = {16, 16};
    tensor4->tensor->rawshape = {16, 16};

    OptimizeSort sort(function->Operations().DuplicatedOpList(), *function);
    Status res = sort.SortOps();
    EXPECT_EQ(res, SUCCESS);
    OoOScheduler ooOScheduler(*function);
    res = ooOScheduler.Init(sort.operations);
    EXPECT_EQ(res, SUCCESS);
    res = ooOScheduler.GenSpillSchedule();
    EXPECT_EQ(res, SUCCESS);
}

TEST_F(ScheduleOoOTest, TestCheckOpBufferSize)
{
    ComputationalGraphBuilder subGraph;
    std::vector<std::string> tensorNames{"t1", "t2", "t3", "t4", "t5", "t6", "t7", "t8"};
    std::vector<MemoryType> tensorMemTypes{MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_UB, MemoryType::MEM_DEVICE_DDR,
                                           MemoryType::MEM_UB,         MemoryType::MEM_UB, MemoryType::MEM_UB,
                                           MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_UB};
    std::vector<Opcode> opCodes{Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC,
                                Opcode::OP_UB_ALLOC, Opcode::OP_COPY_IN,  Opcode::OP_COPY_IN,  Opcode::OP_ROWMAX_SINGLE,
                                Opcode::OP_ADD,      Opcode::OP_COPY_OUT};
    std::vector<std::vector<std::string>> ioperands{{}, {}, {}, {}, {}, {"t1"}, {"t3"}, {"t2"}, {"t4", "t5"}, {"t5"}};
    std::vector<std::vector<std::string>> ooperands{{"t2"}, {"t4"}, {"t5"},       {"t6"}, {"t8"},
                                                    {"t2"}, {"t4"}, {"t5", "t6"}, {"t8"}, {"t7"}};
    std::vector<std::string> opNames{"Alloc1",  "Alloc2",  "Alloc3",  "Alloc4", "Alloc5",
                                     "Copyin1", "Copyin2", "RowMax1", "Add1",   "Copyout1"};
    EXPECT_EQ(subGraph.AddTensors(DataType::DT_FP32, {144, 128}, tensorMemTypes, tensorNames, 0), true);
    EXPECT_EQ(subGraph.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    Function* function = subGraph.GetFunction();
    EXPECT_NE(function, nullptr);

    OoOScheduler ooOScheduler(*function);
    Status res = ooOScheduler.Init(function->Operations().DuplicatedOpList());
    EXPECT_EQ(res, FAILED);
}

TEST_F(ScheduleOoOTest, TestInitLocalBufferFailed)
{
    ComputationalGraphBuilder subGraph;
    std::vector<std::string> tensorNames{"t1", "t2", "t3", "t4", "t5", "t6", "t7", "t8"};
    std::vector<MemoryType> tensorMemTypes{MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_UB, MemoryType::MEM_DEVICE_DDR,
                                           MemoryType::MEM_UB,         MemoryType::MEM_UB, MemoryType::MEM_UB,
                                           MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_UB};
    std::vector<Opcode> opCodes{Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC,
                                Opcode::OP_UB_ALLOC, Opcode::OP_COPY_IN,  Opcode::OP_COPY_IN,  Opcode::OP_ROWMAX_SINGLE,
                                Opcode::OP_ADD,      Opcode::OP_COPY_OUT};
    std::vector<std::vector<std::string>> ioperands{{}, {}, {}, {}, {}, {"t1"}, {"t3"}, {"t2"}, {"t4", "t5"}, {"t5"}};
    std::vector<std::vector<std::string>> ooperands{{"t2"}, {"t4"}, {"t5"},       {"t6"}, {"t8"},
                                                    {"t2"}, {"t4"}, {"t5", "t6"}, {"t8"}, {"t7"}};
    std::vector<std::string> opNames{"Alloc1",  "Alloc2",  "Alloc3",  "Alloc4", "Alloc5",
                                     "Copyin1", "Copyin2", "RowMax1", "Add1",   "Copyout1"};
    EXPECT_EQ(subGraph.AddTensors(DataType::DT_FP32, {62, 69}, tensorMemTypes, tensorNames, 0), true);
    EXPECT_EQ(subGraph.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    Function* function = subGraph.GetFunction();
    EXPECT_NE(function, nullptr);

    OoOScheduler ooOScheduler(*function);
    Status res = ooOScheduler.Init(function->Operations().DuplicatedOpList());
    EXPECT_EQ(res, SUCCESS);
}

TEST_F(ScheduleOoOTest, TestCheckAllocBufferSize)
{
    ComputationalGraphBuilder subGraph;
    std::vector<std::string> tensorNames{"t1", "t2", "t3", "t4", "t5", "t6", "t7", "t8"};
    std::vector<MemoryType> tensorMemTypes{MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_UB, MemoryType::MEM_DEVICE_DDR,
                                           MemoryType::MEM_UB,         MemoryType::MEM_UB, MemoryType::MEM_UB,
                                           MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_UB};
    std::vector<Opcode> opCodes{Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC,
                                Opcode::OP_UB_ALLOC, Opcode::OP_COPY_IN,  Opcode::OP_COPY_IN,  Opcode::OP_ROWMAX_SINGLE,
                                Opcode::OP_ADD,      Opcode::OP_COPY_OUT};
    std::vector<std::vector<std::string>> ioperands{{}, {}, {}, {}, {}, {"t1"}, {"t3"}, {"t2"}, {"t4", "t5"}, {"t5"}};
    std::vector<std::vector<std::string>> ooperands{{"t2"}, {"t4"}, {"t5"},       {"t6"}, {"t8"},
                                                    {"t2"}, {"t4"}, {"t5", "t6"}, {"t8"}, {"t7"}};
    std::vector<std::string> opNames{"Alloc1",  "Alloc2",  "Alloc3",  "Alloc4", "Alloc5",
                                     "Copyin1", "Copyin2", "RowMax1", "Add1",   "Copyout1"};
    EXPECT_EQ(subGraph.AddTensors(DataType::DT_FP32, {256, 256}, tensorMemTypes, tensorNames, 0), true);
    EXPECT_EQ(subGraph.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    Function* function = subGraph.GetFunction();
    EXPECT_NE(function, nullptr);

    OoOScheduler ooOScheduler(*function);
    Status res = ooOScheduler.Init(function->Operations().DuplicatedOpList());
    EXPECT_EQ(res, FAILED);
}

TEST_F(ScheduleOoOTest, TestOoORollbackMix) {
    ComputationalGraphBuilder subGraph;
    std::vector<std::string> tensorNames{"T3",  "T1",   "T6",   "T8",   "T11",  "T13",  "T16",  "T18",
                                         "T21", "DDR1", "DDR2", "DDR3", "DDR4", "DDR5", "DDR6", "DDR7"};
    std::vector<MemoryType> tensorMemTypes{
        MemoryType::MEM_L1,         MemoryType::MEM_L1,         MemoryType::MEM_L1,         MemoryType::MEM_L1,
        MemoryType::MEM_L1,         MemoryType::MEM_L1,         MemoryType::MEM_L1,         MemoryType::MEM_L1,
        MemoryType::MEM_L1,         MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_DEVICE_DDR,
        MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_DEVICE_DDR};
    std::vector<std::string> tensorNames_L0AB{"T4", "T2", "T7", "T9", "T12", "T14", "T19", "T17"};
    std::vector<MemoryType> tensorMemTypes_L0AB{MemoryType::MEM_L0B, MemoryType::MEM_L0A, MemoryType::MEM_L0A,
                                                MemoryType::MEM_L0B, MemoryType::MEM_L0A, MemoryType::MEM_L0B,
                                                MemoryType::MEM_L0B, MemoryType::MEM_L0A};
    std::vector<std::string> tensorNames_L0C{"T5", "T10", "T15", "T20"};
    std::vector<MemoryType> tensorMemTypes_L0C{
        MemoryType::MEM_L0C, MemoryType::MEM_L0C, MemoryType::MEM_L0C, MemoryType::MEM_L0C};
    std::vector<Opcode> opCodes{
        Opcode::OP_L1_ALLOC,  Opcode::OP_L1_ALLOC,  Opcode::OP_L1_ALLOC,  Opcode::OP_L1_ALLOC,  Opcode::OP_L1_ALLOC,
        Opcode::OP_L1_ALLOC,  Opcode::OP_L1_ALLOC,  Opcode::OP_L1_ALLOC,  Opcode::OP_L0A_ALLOC, Opcode::OP_L1_ALLOC,
        Opcode::OP_L0A_ALLOC, Opcode::OP_L0A_ALLOC, Opcode::OP_L0A_ALLOC, Opcode::OP_L0B_ALLOC, Opcode::OP_L0B_ALLOC,
        Opcode::OP_L0B_ALLOC, Opcode::OP_L0B_ALLOC, Opcode::OP_L0C_ALLOC, Opcode::OP_L0C_ALLOC, Opcode::OP_L0C_ALLOC,
        Opcode::OP_COPY_IN,   Opcode::OP_COPY_IN,   Opcode::OP_COPY_IN,   Opcode::OP_COPY_IN,   Opcode::OP_COPY_IN,
        Opcode::OP_COPY_IN,   Opcode::OP_L1_TO_L0A, Opcode::OP_L1_TO_L0A, Opcode::OP_L1_TO_L0A, Opcode::OP_L1_TO_L0A,
        Opcode::OP_L1_TO_L0B, Opcode::OP_L1_TO_L0B, Opcode::OP_L1_TO_L0B, Opcode::OP_L1_TO_L0B, Opcode::OP_L0C_TO_L1,
        Opcode::OP_L0C_TO_L1, Opcode::OP_L0C_TO_L1, Opcode::OP_A_MUL_B,   Opcode::OP_A_MUL_B,   Opcode::OP_A_MUL_B,
        Opcode::OP_COPY_OUT,  Opcode::OP_A_MULACC_B};
    std::vector<std::vector<std::string>> inputoperands{{},           {},
                                                        {},           {},
                                                        {},           {},
                                                        {},           {},
                                                        {},           {},
                                                        {},           {},
                                                        {},           {},
                                                        {},           {},
                                                        {},           {},
                                                        {},           {},
                                                        {"DDR2"},     {"DDR1"},
                                                        {"DDR3"},     {"DDR4"},
                                                        {"DDR5"},     {"DDR6"},
                                                        {"T1"},       {"T6"},
                                                        {"T11"},      {"T16"},
                                                        {"T3"},       {"T8"},
                                                        {"T13"},      {"T18"},
                                                        {"T5"},       {"T15"},
                                                        {"T20"},      {"T2", "T4"},
                                                        {"T7", "T9"}, {"T12", "T14"},
                                                        {"T21"},      {"T10", "T17", "T19"}};
    std::vector<std::vector<std::string>> outputoperands{
        {"T1"},  {"T3"},  {"T6"},  {"T8"},  {"T11"}, {"T13"}, {"T16"}, {"T18"},  {"T2"},  {"T21"}, {"T7"},
        {"T12"}, {"T17"}, {"T4"},  {"T9"},  {"T14"}, {"T19"}, {"T5"},  {"T10"},  {"T15"}, {"T3"},  {"T1"},
        {"T8"},  {"T11"}, {"T13"}, {"T18"}, {"T2"},  {"T7"},  {"T12"}, {"T17"},  {"T4"},  {"T9"},  {"T14"},
        {"T19"}, {"T6"},  {"T16"}, {"T21"}, {"T5"},  {"T10"}, {"T15"}, {"DDR7"}, {"T20"}};
    std::vector<std::string> operationNames{
        "L1_Alloc1",      "L1_Alloc2",      "L1_Alloc3",      "L1_Alloc4",      "L1_Alloc5",      "L1_Alloc6",
        "L1_Alloc7",      "L1_Alloc8",      "L0A_Alloc1",     "L1_Alloc9",      "L0A_Alloc2",     "L0A_Alloc3",
        "L0A_Alloc4",     "L0B_Alloc1",     "L0B_Alloc2",     "L0B_Alloc3",     "L0B_Alloc4",     "L0C_Alloc1",
        "L0C_Alloc2",     "L0C_Alloc3",     "Copyin2",        "Copyin1",        "Copyin3",        "Copyin4",
        "Copyin5",        "Copyin6",        "OP_L1_TO_L0A_1", "OP_L1_TO_L0A_2", "OP_L1_TO_L0A_3", "OP_L1_TO_L0A_4",
        "OP_L1_TO_L0B_1", "OP_L1_TO_L0B_2", "OP_L1_TO_L0B_3", "OP_L1_TO_L0B_4", "OP_L0C_TO_L1_1", "OP_L0C_TO_L1_2",
        "OP_L0C_TO_L1_3", "OP_A_MUL_B_1",   "OP_A_MUL_B_2",   "OP_A_MUL_B_3",   "Copyout",        "OP_A_MULACC_B"};
    EXPECT_EQ(subGraph.AddTensors(DataType::DT_FP32, {128, 128}, tensorMemTypes_L0AB, tensorNames_L0AB, 0), true);
    EXPECT_EQ(subGraph.AddTensors(DataType::DT_FP32, {16, 16}, tensorMemTypes, tensorNames, 0), true);
    EXPECT_EQ(subGraph.AddTensors(DataType::DT_FP32, {128, 256}, tensorMemTypes_L0C, tensorNames_L0C, 0), true);
    EXPECT_EQ(subGraph.AddOps(opCodes, inputoperands, outputoperands, operationNames, true), true);
    Function* function = subGraph.GetFunction();
    std::shared_ptr<LogicalTensor> tensor = subGraph.GetTensor("T10");
    tensor->memoryrange.memId = subGraph.GetTensor("T20")->memoryrange.memId;
    EXPECT_NE(function, nullptr);

    OptimizeSort optimizeSort(function->Operations().DuplicatedOpList(), *function);
    Status res = optimizeSort.SortOps();
    EXPECT_EQ(res, SUCCESS);
}

TEST_F(ScheduleOoOTest, TestHasEnoughBuffer)
{
    ComputationalGraphBuilder subGraph;
    std::vector<std::string> tensorNames{"t1", "t2"};
    std::vector<MemoryType> tensorMemTypes{MemoryType::MEM_UB, MemoryType::MEM_UB};
    std::vector<Opcode> opCodes{Opcode::OP_UB_ALLOC};
    std::vector<std::vector<std::string>> ioperands{{}};
    std::vector<std::vector<std::string>> ooperands{{"t1", "t2"}};
    std::vector<std::string> opNames{"Alloc1"};
    EXPECT_EQ(subGraph.AddTensors(DataType::DT_FP32, {128, 128}, tensorMemTypes, tensorNames, 0), true);
    EXPECT_EQ(subGraph.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    Function* function = subGraph.GetFunction();
    EXPECT_NE(function, nullptr);

    auto op = subGraph.GetOp("Alloc1");

    OoOScheduler ooOScheduler(*function);
    ooOScheduler.orderedOps.push_back(op);
    ooOScheduler.opIsAllocMap[op] = true;
    ooOScheduler.GetSuccessors(op).clear();
    bool res = ooOScheduler.HasEnoughBuffer(op, MemoryType::MEM_UB);
    EXPECT_EQ(res, false);
}

TEST_F(ScheduleOoOTest, TestHasEnoughBufferAddMemId)
{
    ComputationalGraphBuilder subGraph;
    std::vector<std::string> tensorNames{"t1", "t2"};
    std::vector<MemoryType> tensorMemTypes{MemoryType::MEM_UB, MemoryType::MEM_UB};
    std::vector<Opcode> opCodes{Opcode::OP_UB_ALLOC, Opcode::OP_COPY_IN};
    std::vector<std::vector<std::string>> ioperands{{}, {"t2"}};
    std::vector<std::vector<std::string>> ooperands{{"t1"}, {"t1"}};
    std::vector<std::string> opNames{"Alloc1", "COPY_IN"};
    EXPECT_EQ(subGraph.AddTensors(DataType::DT_FP32, {128, 128}, tensorMemTypes, tensorNames, 0), true);
    EXPECT_EQ(subGraph.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    Function* function = subGraph.GetFunction();
    EXPECT_NE(function, nullptr);
    auto op = subGraph.GetOp("Alloc1");
    auto opCopyIn = subGraph.GetOp("COPY_IN");
    auto tensor1 = subGraph.GetTensor("t1");
    auto tensor2 = subGraph.GetTensor("t2");
    op->GetOutputOperand(0)->ClearAllProducers();
    op->GetOutputOperand(0)->AddProducer(*opCopyIn);
    OoOScheduler ooOScheduler(*function);
    ooOScheduler.orderedOps.push_back(op);
    ooOScheduler.orderedOps.push_back(opCopyIn);
    ooOScheduler.opIsAllocMap[op] = true;
    ooOScheduler.GetSuccessors(op).insert(opCopyIn);
    ooOScheduler.opReqMemIdsMap[opCopyIn] = {1};
    EXPECT_EQ(ooOScheduler.InitLocalBuffer(tensor2, 1), SUCCESS);
    bool res = ooOScheduler.HasEnoughBuffer(op, MemoryType::MEM_UB);
    EXPECT_EQ(res, false);
}

TEST_F(ScheduleOoOTest, TestCoreAssign)
{
    ComputationalGraphBuilder subGraph;
    std::vector<std::string> tensorNames{"t0", "t1", "t2", "t3", "t4", "t5", "t6", "t7", "t8", "t9", "t10"};
    std::vector<Opcode> opCodes{Opcode::OP_A_MUL_B, Opcode::OP_ADDS,      Opcode::OP_ADDS, Opcode::OP_ADDS,
                                Opcode::OP_ADDS,    Opcode::OP_A_MUL_B,   Opcode::OP_ADDS, Opcode::OP_A_MUL_B,
                                Opcode::OP_ADD,     Opcode::OP_A_MULACC_B};
    std::vector<std::vector<std::string>> ioperands{
        {"t0", "t0"}, {"t1"}, {"t1"},       {"t1"},       {"t2", "t2"},
        {"t2"},       {"t4"}, {"t5", "t5"}, {"t3", "t6"}, {"t7", "t8", "t9"}};
    std::vector<std::vector<std::string>> ooperands{{"t1"}, {"t2"}, {"t8"}, {"t9"}, {"t3"},
                                                    {"t4"}, {"t5"}, {"t6"}, {"t7"}, {"t10"}};
    std::vector<std::string> opNames{"op1", "op2", "op3", "op4", "op5", "op6", "op7", "op8", "op9", "op10"};
    EXPECT_EQ(subGraph.AddTensors(DataType::DT_FP32, {256, 256}, tensorNames), true);
    EXPECT_EQ(subGraph.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    Function* function = subGraph.GetFunction();
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

TEST_F(ScheduleOoOTest, TestLatencyEstimatorMainLoop)
{
    // 创建测试数据
    ComputationalGraphBuilder subGraph;

    // 定义测试张量
    std::vector<std::string> tensorNames{"t1", "t2", "t3", "t4", "t5"};
    std::vector<MemoryType> tensorMemTypes{
        MemoryType::MEM_UB, MemoryType::MEM_L0A, MemoryType::MEM_L0B, MemoryType::MEM_L0C, MemoryType::MEM_UB};

    std::vector<Opcode> opCodes{Opcode::OP_UB_ALLOC,    Opcode::OP_L0A_ALLOC, Opcode::OP_L0B_ALLOC,
                                Opcode::OP_A_MUL_B,     Opcode::OP_L0C_ALLOC, Opcode::OP_UB_ALLOC,
                                Opcode::OP_L0C_COPY_UB, Opcode::OP_UB_COPY_L1};

    std::vector<std::vector<std::string>> ioperands{{}, {}, {}, {"t2", "t3"}, {}, {}, {"t4"}, {"t1"}};

    std::vector<std::vector<std::string>> ooperands{{"t1"}, {"t2"}, {"t3"}, {"t4"}, {"t4"}, {"t5"}, {"t5"}, {"t2"}};

    std::vector<std::string> opNames{"UB_ALLOC2",  "L0A_Alloc1", "L0B_Alloc1",     "Mul1",
                                     "L0C_Alloc1", "UB_ALLOC1",  "OP_L0C_COPY_UB", "OP_UB_COPY_L1"};

    // 构建计算图
    EXPECT_EQ(subGraph.AddTensors(DataType::DT_FP32, {16, 16}, tensorMemTypes, tensorNames, 0), true);
    EXPECT_EQ(subGraph.AddOps(opCodes, ioperands, ooperands, opNames, true), true);

    Function* function = subGraph.GetFunction();
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

TEST_F(ScheduleOoOTest, TestMixSchedule)
{
    ComputationalGraphBuilder subGraph;
    std::vector<std::string> tensorNames{"T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8"};
    std::vector<MemoryType> tensorMemTypes{MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_UB, MemoryType::MEM_DEVICE_DDR,
                                           MemoryType::MEM_UB,         MemoryType::MEM_UB, MemoryType::MEM_UB,
                                           MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_UB};
    std::vector<Opcode> opcodeList{
        Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC,      Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC,
        Opcode::OP_COPY_IN,  Opcode::OP_COPY_IN,  Opcode::OP_ROWMAX_SINGLE, Opcode::OP_ADD,      Opcode::OP_COPY_OUT};
    std::vector<std::vector<std::string>> inputoperands{{},     {},     {},     {},           {},
                                                        {"T1"}, {"T3"}, {"T2"}, {"T4", "T5"}, {"T5"}};
    std::vector<std::vector<std::string>> outputoperands{{"T2"}, {"T4"}, {"T5"},       {"T6"}, {"T8"},
                                                         {"T2"}, {"T4"}, {"T5", "T6"}, {"T8"}, {"T7"}};
    std::vector<std::string> opNames{"Alloc1",  "Alloc2",  "Alloc3",  "Alloc4", "Alloc5",
                                     "Copyin1", "Copyin2", "RowMax1", "Add1",   "Copyout1"};
    EXPECT_EQ(subGraph.AddTensors(DataType::DT_FP32, {32, 32}, tensorMemTypes, tensorNames, 0), true);
    EXPECT_EQ(subGraph.AddOps(opcodeList, inputoperands, outputoperands, opNames, true), true);
    Function* function = subGraph.GetFunction();
    OoOSchedule oooSchedule;
    auto opList = function->Operations(false).DuplicatedOpList();
    std::pair<uint64_t, Function*> functionPair = std::make_pair(0, function);
    int64_t size = 0;
    Status res = oooSchedule.MixSchedule(opList, *function, functionPair, size);
    EXPECT_EQ(res, SUCCESS);
}

TEST_F(ScheduleOoOTest, TestBufferPollRearrange)
{
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
    std::vector<std::string> tensorNames{"t1", "t2", "t3"};
    std::vector<MemoryType> tensorMemTypes{MemoryType::MEM_UB, MemoryType::MEM_UB, MemoryType::MEM_UB};
    std::vector<Opcode> opCodes{Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC};
    std::vector<std::vector<std::string>> ioperands{{}, {}, {}};
    std::vector<std::vector<std::string>> ooperands{{"t1"}, {"t2"}, {"t3"}};
    std::vector<std::string> opNames{"Alloc1", "Alloc2", "Alloc3"};
    EXPECT_EQ(subGraph.AddTensors(DataType::DT_FP32, {128, 128}, tensorMemTypes, tensorNames, 0), true);
    EXPECT_EQ(subGraph.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    Function* function = subGraph.GetFunction();
    EXPECT_NE(function, nullptr);

    // 构造Operation*
    auto alloc1 = subGraph.GetOp("Alloc1");
    auto alloc2 = subGraph.GetOp("Alloc2");
    auto alloc3 = subGraph.GetOp("Alloc3");

    // 验证重排，排序依据为size从大到小
    OoOScheduler oooSchedule(*function);
    auto corePair = opCoreTypeMap.at(OpCoreType::AIV);
    oooSchedule.opExecOrderMap[alloc1] = 1;
    oooSchedule.opExecOrderMap[alloc2] = 2;
    oooSchedule.opReqMemIdsMap[alloc3] = {3};
    oooSchedule.opExecOrderMap[alloc3] = 0;
    oooSchedule.opCoreLocationMap[alloc3] = corePair;
    oooSchedule.bufferManagerMap[corePair.first][corePair.second][MemoryType::MEM_UB] = pool;
    oooSchedule.tensorOccupyMap[MemoryType::MEM_UB].emplace(1, alloc1);
    oooSchedule.tensorOccupyMap[MemoryType::MEM_UB].emplace(2, nullptr);
    oooSchedule.localBufferMap[3] = std::make_shared<LocalBuffer>(3, 65536, MemoryType::MEM_UB);
    size_t temp = 1;
    EXPECT_EQ(oooSchedule.SpillAllBuffer(alloc3, temp, false, oooSchedule.localBufferMap[3]), FAILED);
    EXPECT_EQ(oooSchedule.RearrangeBuffer(alloc3, MemoryType::MEM_UB, corePair, false), FAILED);
    EXPECT_EQ(oooSchedule.PrintSpillFailedInfo(alloc3, true), FAILED);
    oooSchedule.tensorOccupyMap[MemoryType::MEM_UB][2] = alloc2;
    EXPECT_EQ(oooSchedule.SpillAllBuffer(alloc3, temp, true, oooSchedule.localBufferMap[3]), FAILED);
    oooSchedule.opExecOrderMap[alloc3] = 3;
    EXPECT_EQ(oooSchedule.RearrangeBuffer(alloc3, MemoryType::MEM_UB, corePair, false), SUCCESS);
    auto &ubPool = oooSchedule.bufferManagerMap[corePair.first][corePair.second][MemoryType::MEM_UB];
    EXPECT_EQ(ubPool.GetBufferSize(1), 65536);
    EXPECT_EQ(ubPool.GetBufferSize(2), 98304);
    EXPECT_EQ(ubPool.GetBufferOffset(1), 98304);
    EXPECT_EQ(ubPool.GetBufferOffset(2), 0);
}

TEST_F(ScheduleOoOTest, TestBufferPoolMakeBufferSliceAlreadyAlloc)
{
    BufferPool pool(MemoryType::MEM_UB, 1024);
    auto tensor = std::make_shared<LocalBuffer>(1, 64, MemoryType::MEM_UB);
    BufferSlice slice1(0, 64);
    EXPECT_EQ(pool.MakeBufferSlice(tensor, slice1), SUCCESS);
    BufferSlice slice2(128, 64);
    EXPECT_EQ(pool.MakeBufferSlice(tensor, slice2), FAILED);
}

TEST_F(ScheduleOoOTest, TestBufferPoolAllocateNoFreeSpace)
{
    BufferPool pool(MemoryType::MEM_UB, 256);
    auto tensor1 = std::make_shared<LocalBuffer>(1, 256, MemoryType::MEM_UB);
    EXPECT_EQ(pool.Allocate(tensor1), SUCCESS);
    auto tensor2 = std::make_shared<LocalBuffer>(2, 64, MemoryType::MEM_UB);
    EXPECT_EQ(pool.Allocate(tensor2), FAILED);
}

TEST_F(ScheduleOoOTest, TestBufferRearrangeSingleBubble)
{
    BufferPool pool(MemoryType::MEM_UB, 100);
    auto tensor = std::make_shared<LocalBuffer>(1, 50, MemoryType::MEM_UB);
    BufferSlice s1(0, 50);
    EXPECT_EQ(pool.MakeBufferSlice(tensor, s1), SUCCESS);
    RearrangeScheme scheme = GetRearrangeScheme(pool, 50);
    EXPECT_EQ(scheme.cost, static_cast<size_t>(INT_MAX));
}

TEST_F(ScheduleOoOTest, TestSchedulerAllocTensorMemRangeNonViewOp)
{
    ComputationalGraphBuilder subGraph;
    subGraph.AddTensor(DataType::DT_FP32, {8, 8}, "a");
    subGraph.AddTensor(DataType::DT_FP32, {8, 8}, "b");
    subGraph.AddTensor(DataType::DT_FP32, {8, 8}, "c");
    subGraph.AddOp(Opcode::OP_ADD, {"a", "b"}, {"c"}, "add1");
    Function* function = subGraph.GetFunction();
    OoOScheduler oooSchedule(*function);
    auto addOp = subGraph.GetOp("add1");
    oooSchedule.GetViewOps(addOp).push_back(addOp);
    EXPECT_EQ(oooSchedule.AllocTensorMemRange(addOp), FAILED);
}

TEST_F(ScheduleOoOTest, TestSpillOnBlockFailedAtL0)
{
    // 构造子图
    ComputationalGraphBuilder subGraph;
    std::vector<std::string> tensorNames{"t1", "t2", "t3", "t4"};
    std::vector<MemoryType> tensorMemTypes{
        MemoryType::MEM_L0A, MemoryType::MEM_L0A, MemoryType::MEM_L0B, MemoryType::MEM_L0B};
    std::vector<Opcode> opCodes{Opcode::OP_L1_TO_L0A, Opcode::OP_L0A_ALLOC, Opcode::OP_L1_TO_L0B, Opcode::OP_L0B_ALLOC};
    std::vector<std::vector<std::string>> ioperands{{}, {}, {}, {}};
    std::vector<std::vector<std::string>> ooperands{{"t1"}, {"t2"}, {"t3"}, {"t4"}};
    std::vector<std::string> opNames{"L1toL0A", "AllocL0A", "L1toL0B", "AllocL0B"};
    EXPECT_EQ(subGraph.AddTensors(DataType::DT_FP16, {128, 128}, tensorMemTypes, tensorNames, 0), true);
    EXPECT_EQ(subGraph.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    Function* function = subGraph.GetFunction();
    EXPECT_NE(function, nullptr);
    // 构造Operation和属性
    auto L1toL0A = subGraph.GetOp("L1toL0A");
    auto L1toL0B = subGraph.GetOp("L1toL0B");
    auto AllocL0A = subGraph.GetOp("AllocL0A");
    auto AllocL0B = subGraph.GetOp("AllocL0B");
    // 构造alloc队列、内存气泡场景的localBufferMap、tensorOccupyMap
    OoOScheduler oooSchedule(*function); 
    oooSchedule.SetReqMemIds(AllocL0A, {3});
    oooSchedule.SetReqMemIds(AllocL0B, {4});
    auto corePair = opCoreTypeMap.at(OpCoreType::AIC);
    oooSchedule.allocIssueQueue[corePair.first][corePair.second][MemoryType::MEM_L0A].Insert(AllocL0A);
    oooSchedule.allocIssueQueue[corePair.first][corePair.second][MemoryType::MEM_L0B].Insert(AllocL0B);
    oooSchedule.tensorOccupyMap[MemoryType::MEM_L0A].emplace(1, L1toL0A);
    oooSchedule.tensorOccupyMap[MemoryType::MEM_L0B].emplace(2, L1toL0B);
    oooSchedule.localBufferMap[1] = std::make_shared<LocalBuffer>(1, 32768, MemoryType::MEM_L0A);
    oooSchedule.localBufferMap[2] = std::make_shared<LocalBuffer>(2, 32768, MemoryType::MEM_L0B);
    oooSchedule.localBufferMap[3] = std::make_shared<LocalBuffer>(3, 32768, MemoryType::MEM_L0A);
    oooSchedule.localBufferMap[4] = std::make_shared<LocalBuffer>(4, 32768, MemoryType::MEM_L0B);
    oooSchedule.localBufferMap[1]->start = 512;
    oooSchedule.localBufferMap[1]->end = 33280;
    oooSchedule.localBufferMap[2]->start = 512;
    oooSchedule.localBufferMap[2]->end = 33280;
    // 验证内存气泡导致L0AB卡死
    bool didSpill = false;
    EXPECT_EQ(oooSchedule.SpillOnCoreBlock(corePair.first, corePair.second, didSpill), FAILED);
    EXPECT_EQ(didSpill, false);
}

TEST_F(ScheduleOoOTest, TestOoO1C2V)
{
    ComputationalGraphBuilder subGraph;
    std::vector<std::string> tensorNames{"t1",   "t3",   "t6",   "t7",   "t8",  "t10",
                                         "DDR1", "DDR2", "DDR3", "DDR4", "t11", "t12"};
    std::vector<MemoryType> tensorMemTypes{
        MemoryType::MEM_L1,         MemoryType::MEM_L1,         MemoryType::MEM_UB,         MemoryType::MEM_UB,
        MemoryType::MEM_UB,         MemoryType::MEM_UB,         MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_DEVICE_DDR,
        MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_UB,         MemoryType::MEM_UB};
    std::vector<std::string> tensorNames_L0{"t2", "t4", "t5"};
    std::vector<MemoryType> tensorMemTypes_L0AB{MemoryType::MEM_L0A, MemoryType::MEM_L0B, MemoryType::MEM_L0C};

    std::vector<Opcode> opCodes{
        Opcode::OP_L1_ALLOC, Opcode::OP_L1_ALLOC,  Opcode::OP_L0A_ALLOC,  Opcode::OP_L0B_ALLOC, Opcode::OP_L0C_ALLOC,
        Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC,  Opcode::OP_UB_ALLOC,   Opcode::OP_UB_ALLOC,  Opcode::OP_COPY_IN,
        Opcode::OP_COPY_IN,  Opcode::OP_L1_TO_L0A, Opcode::OP_L1_TO_L0B,  Opcode::OP_A_MUL_B,   Opcode::OP_L0C_COPY_UB,
        Opcode::OP_ADDS,     Opcode::OP_COPY_OUT,  Opcode::OP_L1_COPY_UB, Opcode::OP_ADDS,      Opcode::OP_COPY_OUT,
        Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC,  Opcode::OP_ADDS,       Opcode::OP_UB_COPY_L1};
    std::vector<std::vector<std::string>> ioperands{
        {},     {},           {},     {},     {},     {},     {},     {},      {}, {"DDR1"}, {"DDR2"}, {"t1"},
        {"t3"}, {"t2", "t4"}, {"t5"}, {"t6"}, {"t7"}, {"t3"}, {"t8"}, {"t10"}, {}, {},       {"t11"},  {"t12"}};
    std::vector<std::vector<std::string>> ooperands{
        {"t1"}, {"t3"}, {"t2"}, {"t4"}, {"t5"},   {"t6"}, {"t7"},  {"t8"},   {"t10"}, {"t11"}, {"t3"},  {"t2"},
        {"t4"}, {"t5"}, {"t6"}, {"t7"}, {"DDR3"}, {"t8"}, {"t10"}, {"DDR4"}, {"t11"}, {"t12"}, {"t12"}, {"t1"}};
    std::vector<std::string> opNames{"L1_Alloc1", "L1_Alloc2", "L0A_Alloc1",  "L0B_Alloc1", "L0C_Alloc1", "UB_Alloc1",
                                     "UB_Alloc2", "UB_Alloc3", "UB_Alloc4",   "COPY_IN1",   "COPY_IN2",   "L1_TO_L0A",
                                     "L1_TO_L0B", "A_MUL_B",   "L0C_COPY_UB", "ADDS1",      "COPY_OUT1",  "L1_COPY_UB",
                                     "ADDS2",     "COPY_OUT2", "UB_Alloc5",   "UB_Alloc6",  "ADDS3",      "UB_COPY_L1"};
    EXPECT_EQ(subGraph.AddTensors(DataType::DT_FP32, {16, 16}, tensorMemTypes, tensorNames, 0), true);
    EXPECT_EQ(subGraph.AddTensors(DataType::DT_FP32, {16, 16}, tensorMemTypes_L0AB, tensorNames_L0, 0), true);
    EXPECT_EQ(subGraph.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    subGraph.GetOp("ADDS1")->SetAttribute(OpAttributeKey::isCube, false);
    Function* function = subGraph.GetFunction();
    auto op1 = subGraph.GetOp("ADDS3");
    auto op2 = subGraph.GetOp("ADDS2");
    auto op3 = subGraph.GetOp("ADDS1");
    auto op4 = subGraph.GetOp("L1_TO_L0A");
    OptimizeSort optimizeSort(function->Operations().DuplicatedOpList(), *function);
    Status res = optimizeSort.SortOps();
    EXPECT_EQ(res, SUCCESS);
    auto opList = optimizeSort.operations;
    OoOSchedule oooSchedule;
    std::pair<uint64_t, Function*> functionPair = std::make_pair(0, function);
    int64_t size = 0;
    res = oooSchedule.MixSchedule(opList, *function, functionPair, size);
    EXPECT_EQ(res, SUCCESS);
    EXPECT_EQ(op1->GetInternalSubgraphID(), 1);
    EXPECT_EQ(op2->GetInternalSubgraphID(), 1);
    EXPECT_EQ(op3->GetInternalSubgraphID(), 0);
    EXPECT_EQ(op4->GetInternalSubgraphID(), 2);
}

void SetInternalSubgraphIDAndAIVCore(Operation* op, int id) {
    op->UpdateInternalSubgraphID(id);
    if (id == 0) {
        op->SetAIVCore(AIVCore::AIV0);
    }
}

void SetAttribute(ComputationalGraphBuilder &subGraph, OoOScheduler &oooSchedule, Operation* &ubCopyL1, Operation* &alloc3) {
    Operation* adds = subGraph.GetOp("ADDS");
    ubCopyL1 = subGraph.GetOp("UB_COPY_L1");
    Operation* copyin1 = subGraph.GetOp("COPY_IN1");
    Operation* copyin2 = subGraph.GetOp("COPY_IN2");
    Operation* copyin3 = subGraph.GetOp("COPY_IN3");
    Operation* copyin4 = subGraph.GetOp("COPY_IN4");
    Operation* copyout1 = subGraph.GetOp("COPY_OUT1");
    Operation* copyout2 = subGraph.GetOp("COPY_OUT2");

    Operation* alloc1 = subGraph.GetOp("UB_Alloc2");
    Operation* alloc2 = subGraph.GetOp("L1_Alloc1");
    alloc3 = subGraph.GetOp("L1_Alloc3");
    Operation* alloc4 = subGraph.GetOp("L1_Alloc2");

    Operation* alloc5 = subGraph.GetOp("UB_Alloc1");
    Operation* alloc6 = subGraph.GetOp("L0A_Alloc1");
    Operation* alloc7 = subGraph.GetOp("L0A_Alloc2");

    SetInternalSubgraphIDAndAIVCore(adds, 0);
    SetInternalSubgraphIDAndAIVCore(alloc5, 0);
    SetInternalSubgraphIDAndAIVCore(alloc1, 0);
    SetInternalSubgraphIDAndAIVCore(ubCopyL1, 0);

    SetInternalSubgraphIDAndAIVCore(alloc2, 1);
    SetInternalSubgraphIDAndAIVCore(alloc3, 1);
    SetInternalSubgraphIDAndAIVCore(alloc4, 1);
    SetInternalSubgraphIDAndAIVCore(alloc6, 1);
    SetInternalSubgraphIDAndAIVCore(alloc7, 1);
    SetInternalSubgraphIDAndAIVCore(copyin1, 1);
    SetInternalSubgraphIDAndAIVCore(copyin2, 1);
    SetInternalSubgraphIDAndAIVCore(copyin3, 1);
    SetInternalSubgraphIDAndAIVCore(copyin4, 1);
    SetInternalSubgraphIDAndAIVCore(copyout1, 1);
    SetInternalSubgraphIDAndAIVCore(copyout2, 1);

    oooSchedule.SetIsRetired(alloc5, true);
    oooSchedule.SetIsRetired(adds, true);
    oooSchedule.SetIsRetired(alloc1, true);
    oooSchedule.SetIsRetired(ubCopyL1, true);
    oooSchedule.SetIsRetired(alloc2, true);
    oooSchedule.SetIsRetired(alloc7, true);
    oooSchedule.SetIsRetired(copyin2, true);

    auto localBuffer1 = oooSchedule.localBufferMap[0];
    auto coreAIC = opCoreTypeMap.at(OpCoreType::AIC);
    oooSchedule.bufferManagerMap[coreAIC.first][coreAIC.second][MemoryType::MEM_L1].Allocate(localBuffer1);
    oooSchedule.tensorOccupyMap[MemoryType::MEM_L1].emplace(0, copyin2);
}

void InitSpillInfo(SpillInfo &spillInfo, int memId, Operation* spillOp) {
    spillInfo.spillMemId_ = memId;
    spillInfo.spillOp_ = spillOp;
    spillInfo.spillTensor_ = spillOp->GetOutputOperand(0);
    spillInfo.ddrTensor_ = nullptr;
    spillInfo.isSpecialL1_ = true;
}

static bool CheckOpInSet(std::unordered_set<Operation*> &opSet, Operation* targetOp) {
    return opSet.find(targetOp) != opSet.end();
}

TEST_F(ScheduleOoOTest, TestL1SpillBuffer)
{
    ComputationalGraphBuilder subGraph;
    std::vector<std::string> tensorL1Names{"t1", "t2", "t3"};
    std::vector<MemoryType> tensorL1MemTypes{MemoryType::MEM_L1, MemoryType::MEM_L1, MemoryType::MEM_L1};
    std::vector<std::string> tensorNames{"UB1", "UB2", "L0A1", "L0A2", "DDR1", "DDR2"};
    std::vector<MemoryType> tensorMemTypes{MemoryType::MEM_UB,  MemoryType::MEM_UB,         MemoryType::MEM_L0A,
                                           MemoryType::MEM_L0A, MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_DEVICE_DDR};

    std::vector<Opcode> opCodes{Opcode::OP_L1_ALLOC,   Opcode::OP_L1_ALLOC,  Opcode::OP_L1_ALLOC,  Opcode::OP_UB_ALLOC,
                                Opcode::OP_UB_ALLOC,   Opcode::OP_L0A_ALLOC, Opcode::OP_L0A_ALLOC, Opcode::OP_ADDS,
                                Opcode::OP_UB_COPY_L1, Opcode::OP_COPY_IN,   Opcode::OP_COPY_IN,   Opcode::OP_COPY_IN,
                                Opcode::OP_COPY_IN,    Opcode::OP_COPY_OUT,  Opcode::OP_COPY_OUT};

    std::vector<std::vector<std::string>> ioperands{{},      {},     {},     {},       {},       {},     {},    {"UB1"},
                                                    {"UB2"}, {"t1"}, {"t1"}, {"L0A1"}, {"L0A2"}, {"t2"}, {"t3"}};
    std::vector<std::vector<std::string>> ooperands{{"t1"},   {"t2"},   {"t3"},  {"UB1"},  {"UB2"},
                                                    {"L0A1"}, {"L0A2"}, {"UB2"}, {"t1"},   {"L0A1"},
                                                    {"L0A2"}, {"t2"},   {"t3"},  {"DDR1"}, {"DDR2"}};
    std::vector<std::string> opNames{"L1_Alloc1",  "L1_Alloc2",  "L1_Alloc3", "UB_Alloc1",  "UB_Alloc2",
                                     "L0A_Alloc1", "L0A_Alloc2", "ADDS",      "UB_COPY_L1", "COPY_IN1",
                                     "COPY_IN2",   "COPY_IN3",   "COPY_IN4",  "COPY_OUT1",  "COPY_OUT2"};
    EXPECT_EQ(subGraph.AddTensors(DataType::DT_FP32, {256, 512}, tensorL1MemTypes, tensorL1Names, 0), true);
    EXPECT_EQ(subGraph.AddTensors(DataType::DT_FP32, {16, 16}, tensorMemTypes, tensorNames, 0), true);
    EXPECT_EQ(subGraph.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    Function* function = subGraph.GetFunction();
    OptimizeSort optimizeSort(function->Operations().DuplicatedOpList(), *function);
    Status res = optimizeSort.SortOps();
    EXPECT_EQ(res, SUCCESS);
    auto opList = optimizeSort.operations;
    OoOScheduler oooSchedule(*function);
    res = oooSchedule.Init(opList);
    EXPECT_EQ(res, SUCCESS);
    EXPECT_EQ(oooSchedule.orderedOps[4]->GetOpcodeStr(), "L0A_ALLOC");
    Operation* ubCopyL1 = nullptr;
    Operation* alloc3 = nullptr;
    SetAttribute(subGraph, oooSchedule, ubCopyL1, alloc3);
    auto localBuffer2 = oooSchedule.localBufferMap[2];

    SpillInfo spillInfo;
    InitSpillInfo(spillInfo, 0, ubCopyL1);
    size_t pcIdx = 7;
    res = oooSchedule.SpillBuffer(spillInfo, alloc3, pcIdx, localBuffer2, true);
    EXPECT_EQ(res, SUCCESS);
    EXPECT_EQ(oooSchedule.orderedOps.size(), 18);
    EXPECT_EQ(oooSchedule.orderedOps[4]->GetOpcodeStr(), "COPY_OUT");
    EXPECT_EQ(oooSchedule.orderedOps[4]->GetInternalSubgraphID(), 0);
    EXPECT_EQ(oooSchedule.orderedOps[4]->GetAIVCore(), AIVCore::AIV0);
    EXPECT_EQ(oooSchedule.orderedOps[12]->GetOpcodeStr(), "L1_ALLOC");
    EXPECT_EQ(oooSchedule.orderedOps[13]->GetOpcodeStr(), "COPY_IN");
    EXPECT_EQ(oooSchedule.orderedOps[12]->GetInternalSubgraphID(), 1);
    EXPECT_EQ(oooSchedule.orderedOps[13]->GetAIVCore(), AIVCore::UNSPECIFIED);
    auto attr = std::dynamic_pointer_cast<CopyOpAttribute>(oooSchedule.orderedOps[13]->GetOpAttribute());
    EXPECT_EQ(static_cast<int>(attr->GetFromOffset()[0].GetSpecifiedValue()), 0);
    EXPECT_EQ(static_cast<int>(attr->GetFromOffset()[1].GetSpecifiedValue()), 0);
    EXPECT_TRUE(CheckOpInSet(oooSchedule.GetPredecessors(subGraph.GetOp("COPY_IN1")), oooSchedule.orderedOps[13]));
}

TEST_F(ScheduleOoOTest, TestL1SpillBufferFailed)
{
    ComputationalGraphBuilder subGraph;
    std::vector<std::string> tensorL1{"t1", "t2", "t3"};
    std::vector<MemoryType> tensorL1Types{MemoryType::MEM_L1, MemoryType::MEM_L1, MemoryType::MEM_L1};
    std::vector<std::string> tensorNames{"DDR", "UB2", "L0A1", "L0A2", "DDR1", "DDR2"};
    std::vector<MemoryType> tensorMemTypes{MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_UB,
                                           MemoryType::MEM_L0A,        MemoryType::MEM_L0A,
                                           MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_DEVICE_DDR};

    std::vector<Opcode> opCodes{Opcode::OP_L1_ALLOC,  Opcode::OP_L1_ALLOC,  Opcode::OP_L1_ALLOC, Opcode::OP_UB_ALLOC,
                                Opcode::OP_L0A_ALLOC, Opcode::OP_L0A_ALLOC, Opcode::OP_COPY_IN,  Opcode::OP_UB_COPY_L1,
                                Opcode::OP_COPY_IN,   Opcode::OP_COPY_IN,   Opcode::OP_COPY_IN,  Opcode::OP_COPY_IN,
                                Opcode::OP_COPY_OUT,  Opcode::OP_COPY_OUT};

    std::vector<std::vector<std::string>> ioperands{{},      {},     {},     {},       {},       {},     {"DDR"},
                                                    {"UB2"}, {"t1"}, {"t1"}, {"L0A1"}, {"L0A2"}, {"t2"}, {"t3"}};
    std::vector<std::vector<std::string>> ooperands{{"t1"}, {"t2"},   {"t3"},   {"UB2"}, {"L0A1"}, {"L0A2"}, {"UB2"},
                                                    {"t1"}, {"L0A1"}, {"L0A2"}, {"t2"},  {"t3"},   {"DDR1"}, {"DDR2"}};
    std::vector<std::string> opNames{"L1_Alloc1",  "L1_Alloc2", "L1_Alloc3",  "UB_Alloc2", "L0A_Alloc1",
                                     "L0A_Alloc2", "COPY_IN5",  "UB_COPY_L1", "COPY_IN1",  "COPY_IN2",
                                     "COPY_IN3",   "COPY_IN4",  "COPY_OUT1",  "COPY_OUT2"};
    EXPECT_EQ(subGraph.AddTensors(DataType::DT_FP32, {256, 512}, tensorL1Types, tensorL1, 0), true);
    EXPECT_EQ(subGraph.AddTensors(DataType::DT_FP32, {16, 16}, tensorMemTypes, tensorNames, 0), true);
    EXPECT_EQ(subGraph.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    Function* function = subGraph.GetFunction();
    OptimizeSort ooosort(function->Operations().DuplicatedOpList(), *function);
    Status res = ooosort.SortOps();
    EXPECT_EQ(res, SUCCESS);
    OoOScheduler oooSchedule(*function);
    auto opList = ooosort.operations;
    res = oooSchedule.Init(opList);
    EXPECT_EQ(res, SUCCESS);
    Operation* ubCopyL1 = subGraph.GetOp("UB_COPY_L1");
    Operation* spillCopyout = nullptr;

    SpillInfo spillInfo;
    InitSpillInfo(spillInfo, 0, ubCopyL1);
    int num = 0;
    bool isFinish = false;
    res = oooSchedule.CreateSpecialL1Copyout(spillInfo, spillCopyout, num, isFinish);
    EXPECT_EQ(res, FAILED);
}

void SetAttributeReshape1(ComputationalGraphBuilder &subGraph, OoOScheduler &oooSchedule, Operation* &reshape, Operation* &alloc3) {
    Operation* adds = subGraph.GetOp("ADDS");
    Operation* ubCopyL1 = subGraph.GetOp("UB_COPY_L1");
    reshape = subGraph.GetOp("RESHAPE");
    Operation* copyin1 = subGraph.GetOp("COPY_IN1");
    Operation* copyin2 = subGraph.GetOp("COPY_IN2");
    Operation* copyin3 = subGraph.GetOp("COPY_IN3");
    Operation* copyin4 = subGraph.GetOp("COPY_IN4");
    Operation* copyout1 = subGraph.GetOp("COPY_OUT1");
    Operation* copyout2 = subGraph.GetOp("COPY_OUT2");

    Operation* alloc1 = subGraph.GetOp("L1_Alloc1");
    Operation* alloc2 = subGraph.GetOp("L1_Alloc2");
    alloc3 = subGraph.GetOp("L1_Alloc3");
    Operation* alloc4 = subGraph.GetOp("UB_Alloc1");

    Operation* alloc5 = subGraph.GetOp("UB_Alloc2");
    Operation* alloc6 = subGraph.GetOp("L0A_Alloc1");
    Operation* alloc7 = subGraph.GetOp("L0A_Alloc2");

    SetInternalSubgraphIDAndAIVCore(adds, 0);
    SetInternalSubgraphIDAndAIVCore(alloc4, 0);
    SetInternalSubgraphIDAndAIVCore(alloc5, 0);
    SetInternalSubgraphIDAndAIVCore(ubCopyL1, 0);

    SetInternalSubgraphIDAndAIVCore(copyout1, 1);
    SetInternalSubgraphIDAndAIVCore(copyout2, 1);
    SetInternalSubgraphIDAndAIVCore(alloc1, 1);
    SetInternalSubgraphIDAndAIVCore(alloc2, 1);
    SetInternalSubgraphIDAndAIVCore(alloc3, 1);
    SetInternalSubgraphIDAndAIVCore(alloc6, 1);
    SetInternalSubgraphIDAndAIVCore(alloc7, 1);
    SetInternalSubgraphIDAndAIVCore(reshape, 1);
    SetInternalSubgraphIDAndAIVCore(copyin1, 1);
    SetInternalSubgraphIDAndAIVCore(copyin2, 1);
    SetInternalSubgraphIDAndAIVCore(copyin3, 1);
    SetInternalSubgraphIDAndAIVCore(copyin4, 1);

    oooSchedule.bufRefCount_ = {{4, 0}, {5, 0}, {0, 1}, {7, 1}, {3, 3}, {6, 3}, {2, 3}};
    oooSchedule.SetIsRetired(alloc4, true);
    oooSchedule.SetIsRetired(alloc5, true);
    oooSchedule.SetIsRetired(adds, true);
    oooSchedule.SetIsRetired(alloc1, true);
    oooSchedule.SetIsRetired(ubCopyL1, true);
    oooSchedule.SetIsRetired(reshape, true);
    oooSchedule.SetIsRetired(alloc7, true);
    oooSchedule.SetIsRetired(copyin2, true);

    auto localBuffer1 = oooSchedule.localBufferMap[0];
    auto coreAIC = opCoreTypeMap.at(OpCoreType::AIC);
    oooSchedule.bufferManagerMap[coreAIC.first][coreAIC.second][MemoryType::MEM_L1].Allocate(localBuffer1);
    oooSchedule.tensorOccupyMap[MemoryType::MEM_L1].emplace(0, copyin2);
}

// 场景：UB_COPY_L1-L1-reshape-L1
TEST_F(ScheduleOoOTest, TestL1ReshapeSpillBuffer1)
{
    ComputationalGraphBuilder subGraph;
    std::vector<std::string> tensorL1Names{"t1", "t2", "t3", "t4"};
    std::vector<MemoryType> tensorL1MemTypes{
        MemoryType::MEM_L1, MemoryType::MEM_L1, MemoryType::MEM_L1, MemoryType::MEM_L1};
    std::vector<std::string> tensorNames{"UB1", "UB2", "L0A1", "L0A2", "DDR1", "DDR2"};
    std::vector<MemoryType> tensorMemTypes{MemoryType::MEM_UB,  MemoryType::MEM_UB,         MemoryType::MEM_L0A,
                                           MemoryType::MEM_L0A, MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_DEVICE_DDR};

    std::vector<std::string> opNames{"L1_Alloc1",  "L1_Alloc2", "L1_Alloc3",  "UB_Alloc1", "UB_Alloc2", "L0A_Alloc1",
                                     "L0A_Alloc2", "ADDS",      "UB_COPY_L1", "RESHAPE",   "COPY_IN1",  "COPY_IN2",
                                     "COPY_IN3",   "COPY_IN4",  "COPY_OUT1",  "COPY_OUT2"};
    std::vector<Opcode> opCodes{Opcode::OP_L1_ALLOC,   Opcode::OP_L1_ALLOC,  Opcode::OP_L1_ALLOC,  Opcode::OP_UB_ALLOC,
                                Opcode::OP_UB_ALLOC,   Opcode::OP_L0A_ALLOC, Opcode::OP_L0A_ALLOC, Opcode::OP_ADDS,
                                Opcode::OP_UB_COPY_L1, Opcode::OP_RESHAPE,   Opcode::OP_COPY_IN,   Opcode::OP_COPY_IN,
                                Opcode::OP_COPY_IN,    Opcode::OP_COPY_IN,   Opcode::OP_COPY_OUT,  Opcode::OP_COPY_OUT};

    std::vector<std::vector<std::string>> ioperands{
        {}, {}, {}, {}, {}, {}, {}, {"UB1"}, {"UB2"}, {"t1"}, {"t2"}, {"t2"}, {"L0A1"}, {"L0A2"}, {"t3"}, {"t4"}};
    std::vector<std::vector<std::string>> ooperands{{"t1"},   {"t3"},  {"t4"},   {"UB1"}, {"UB2"},  {"L0A1"},
                                                    {"L0A2"}, {"UB2"}, {"t1"},   {"t2"},  {"L0A1"}, {"L0A2"},
                                                    {"t3"},   {"t4"},  {"DDR1"}, {"DDR2"}};
    EXPECT_EQ(subGraph.AddTensors(DataType::DT_FP32, {256, 512}, tensorL1MemTypes, tensorL1Names, 0), true);
    EXPECT_EQ(subGraph.AddTensors(DataType::DT_FP32, {16, 16}, tensorMemTypes, tensorNames, 0), true);
    EXPECT_EQ(subGraph.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    Function* function = subGraph.GetFunction();
    std::shared_ptr<LogicalTensor> tensor = subGraph.GetTensor("t2");
    tensor->memoryrange.memId = subGraph.GetTensor("t1")->memoryrange.memId;
    OptimizeSort optimizeSort(function->Operations().DuplicatedOpList(), *function);
    Status res = optimizeSort.SortOps();
    EXPECT_EQ(res, SUCCESS);
    OoOScheduler ooOSchedule(*function);
    res = ooOSchedule.Init(optimizeSort.operations);
    EXPECT_EQ(res, SUCCESS);
    EXPECT_EQ(ooOSchedule.orderedOps.size(), 16);
    EXPECT_TRUE(CheckOpInSet(ooOSchedule.GetPredecessors(subGraph.GetOp("COPY_IN1")), ooOSchedule.orderedOps[4]));
    EXPECT_EQ(ooOSchedule.orderedOps[15]->GetOpcodeStr(), "COPY_OUT");
    Operation* reshape = nullptr;
    Operation* alloc3 = nullptr;
    SetAttributeReshape1(subGraph, ooOSchedule, reshape, alloc3);
    EXPECT_EQ(ooOSchedule.bufRefCount_[0], 1);
    auto localBuffer2 = ooOSchedule.localBufferMap[3];

    SpillInfo spillInfo;
    InitSpillInfo(spillInfo, 0, reshape);
    size_t pcIdx = 9;
    res = ooOSchedule.SpillBuffer(spillInfo, alloc3, pcIdx, localBuffer2, true);
    EXPECT_EQ(res, SUCCESS);
    EXPECT_TRUE(CheckOpInSet(ooOSchedule.GetPredecessors(subGraph.GetOp("COPY_IN1")), ooOSchedule.orderedOps[15]));
    EXPECT_EQ(ooOSchedule.bufRefCount_[0], 0);
    EXPECT_EQ(ooOSchedule.orderedOps[15]->GetOpcodeStr(), "RESHAPE");
    EXPECT_EQ(ooOSchedule.orderedOps.size(), 20);
    auto attr = std::dynamic_pointer_cast<CopyOpAttribute>(ooOSchedule.orderedOps[14]->GetOpAttribute());
    EXPECT_EQ(static_cast<int>(attr->GetFromOffset()[0].GetSpecifiedValue()), 0);
    EXPECT_EQ(static_cast<int>(attr->GetFromOffset()[1].GetSpecifiedValue()), 0);
}

void SetAttributeReshape2(ComputationalGraphBuilder &subGraph, OoOScheduler &oooSchedule, Operation* &reshape, Operation* &alloc3) {
    Operation* copyin5 = subGraph.GetOp("COPY_IN5");
    reshape = subGraph.GetOp("RESHAPE");
    Operation* copyin1 = subGraph.GetOp("COPY_IN1");
    Operation* copyout1 = subGraph.GetOp("COPY_OUT1");
    Operation* copyout2 = subGraph.GetOp("COPY_OUT2");
    Operation* copyin2 = subGraph.GetOp("COPY_IN2");
    Operation* copyin3 = subGraph.GetOp("COPY_IN3");
    Operation* copyin4 = subGraph.GetOp("COPY_IN4");
    std::vector<int64_t> offset = {1, 1};
    std::vector<int64_t> shape = {256, 512};
    copyin5->SetOpAttribute(std::make_shared<CopyOpAttribute>(OpImmediate::Specified(offset),
        MemoryType::MEM_L1, OpImmediate::Specified(shape), OpImmediate::Specified(shape)));

    Operation* alloc1 = subGraph.GetOp("L1_Alloc1");
    Operation* alloc2 = subGraph.GetOp("L1_Alloc2");
    alloc3 = subGraph.GetOp("L1_Alloc3");

    Operation* alloc6 = subGraph.GetOp("L0A_Alloc1");
    Operation* alloc7 = subGraph.GetOp("L0A_Alloc2");

    SetInternalSubgraphIDAndAIVCore(alloc1, 1);
    SetInternalSubgraphIDAndAIVCore(copyin5, 1);
    SetInternalSubgraphIDAndAIVCore(alloc2, 1);
    SetInternalSubgraphIDAndAIVCore(alloc3, 1);
    SetInternalSubgraphIDAndAIVCore(alloc6, 1);
    SetInternalSubgraphIDAndAIVCore(alloc7, 1);
    SetInternalSubgraphIDAndAIVCore(copyout1, 1);
    SetInternalSubgraphIDAndAIVCore(reshape, 1);
    SetInternalSubgraphIDAndAIVCore(copyin1, 1);
    SetInternalSubgraphIDAndAIVCore(copyout2, 1);
    SetInternalSubgraphIDAndAIVCore(copyin2, 1);
    SetInternalSubgraphIDAndAIVCore(copyin3, 1);
    SetInternalSubgraphIDAndAIVCore(copyin4, 1);

    oooSchedule.bufRefCount_ = {{0, 1}, {6, 1}, {5, 3}, {3, 3}, {2, 3}};
    oooSchedule.SetIsRetired(alloc1, true);
    oooSchedule.SetIsRetired(copyin5, true);
    oooSchedule.SetIsRetired(reshape, true);
    oooSchedule.SetIsRetired(alloc7, true);
    oooSchedule.SetIsRetired(copyin2, true);

    auto localBuffer = oooSchedule.localBufferMap[0];
    auto coreAIC = opCoreTypeMap.at(OpCoreType::AIC);
    oooSchedule.tensorOccupyMap[MemoryType::MEM_L1].emplace(0, copyin2);
    oooSchedule.bufferManagerMap[coreAIC.first][coreAIC.second][MemoryType::MEM_L1].Allocate(localBuffer);
}

// 场景：copy_in-L1-reshape-L1
TEST_F(ScheduleOoOTest, TestL1ReshapeSpillBuffer2)
{
    ComputationalGraphBuilder subGraph;
    std::vector<std::string> tensorL1Names{"t1", "t2", "t3", "t4"};
    std::vector<MemoryType> tensorL1MemTypes{
        MemoryType::MEM_L1, MemoryType::MEM_L1, MemoryType::MEM_L1, MemoryType::MEM_L1};
    std::vector<std::string> tensorNames{"DDR", "L0A1", "L0A2", "DDR1", "DDR2"};
    std::vector<MemoryType> tensorMemTypes{
        MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_L0A, MemoryType::MEM_L0A, MemoryType::MEM_DEVICE_DDR,
        MemoryType::MEM_DEVICE_DDR};

    std::vector<Opcode> opCodes{Opcode::OP_L1_ALLOC,  Opcode::OP_L1_ALLOC, Opcode::OP_L1_ALLOC, Opcode::OP_L0A_ALLOC,
                                Opcode::OP_L0A_ALLOC, Opcode::OP_COPY_IN,  Opcode::OP_RESHAPE,  Opcode::OP_COPY_IN,
                                Opcode::OP_COPY_IN,   Opcode::OP_COPY_IN,  Opcode::OP_COPY_IN,  Opcode::OP_COPY_OUT,
                                Opcode::OP_COPY_OUT};

    std::vector<std::vector<std::string>> inputOperands{{},     {},     {},       {},       {},     {"DDR"}, {"t1"},
                                                        {"t2"}, {"t2"}, {"L0A1"}, {"L0A2"}, {"t3"}, {"t4"}};
    std::vector<std::vector<std::string>> outputOperands{{"t1"}, {"t3"},   {"t4"},   {"L0A1"}, {"L0A2"},
                                                         {"t1"}, {"t2"},   {"L0A1"}, {"L0A2"}, {"t3"},
                                                         {"t4"}, {"DDR1"}, {"DDR2"}};
    std::vector<std::string> opNames{"L1_Alloc1", "L1_Alloc2", "L1_Alloc3", "L0A_Alloc1", "L0A_Alloc2",
                                     "COPY_IN5",  "RESHAPE",   "COPY_IN1",  "COPY_IN2",   "COPY_IN3",
                                     "COPY_IN4",  "COPY_OUT1", "COPY_OUT2"};
    EXPECT_EQ(subGraph.AddTensors(DataType::DT_FP32, {256, 512}, tensorL1MemTypes, tensorL1Names, 0), true);
    EXPECT_EQ(subGraph.AddTensors(DataType::DT_FP32, {16, 16}, tensorMemTypes, tensorNames, 0), true);
    EXPECT_EQ(subGraph.AddOps(opCodes, inputOperands, outputOperands, opNames, true), true);
    Function* function = subGraph.GetFunction();
    std::shared_ptr<LogicalTensor> tensor0 = subGraph.GetTensor("t2");
    tensor0->memoryrange.memId = subGraph.GetTensor("t1")->memoryrange.memId;
    OptimizeSort optimizeSort(function->Operations().DuplicatedOpList(), *function);
    Status res = optimizeSort.SortOps();
    EXPECT_EQ(res, SUCCESS);
    OoOScheduler oooScheduler(*function);
    res = oooScheduler.Init(optimizeSort.operations);
    EXPECT_EQ(res, SUCCESS);
    EXPECT_EQ(oooScheduler.orderedOps.size(), 13);
    EXPECT_TRUE(CheckOpInSet(oooScheduler.GetPredecessors(subGraph.GetOp("COPY_IN1")), oooScheduler.orderedOps[2]));
    EXPECT_EQ(oooScheduler.orderedOps[10]->GetOpcodeStr(), "L1_ALLOC");
    Operation* reshape = nullptr;
    Operation* alloc3 = nullptr;
    SetAttributeReshape2(subGraph, oooScheduler, reshape, alloc3);
    EXPECT_EQ(oooScheduler.bufRefCount_[0], 1);
    auto localBuffer = oooScheduler.localBufferMap[3];

    SpillInfo spillInfo;
    InitSpillInfo(spillInfo, 0, reshape);
    size_t pcIdx = 5;
    res = oooScheduler.SpillBuffer(spillInfo, alloc3, pcIdx, localBuffer, true);
    EXPECT_EQ(res, SUCCESS);
    EXPECT_TRUE(CheckOpInSet(oooScheduler.GetPredecessors(subGraph.GetOp("COPY_IN1")), oooScheduler.orderedOps[11]));
    EXPECT_EQ(oooScheduler.bufRefCount_[0], 0);
    EXPECT_EQ(oooScheduler.orderedOps[11]->GetOpcodeStr(), "RESHAPE");
    EXPECT_EQ(oooScheduler.orderedOps.size(), 16);
    auto attr = std::dynamic_pointer_cast<CopyOpAttribute>(oooScheduler.orderedOps[10]->GetOpAttribute());
    EXPECT_EQ(static_cast<int>(attr->GetFromOffset()[0].GetSpecifiedValue()), 1);
    EXPECT_EQ(static_cast<int>(attr->GetFromOffset()[1].GetSpecifiedValue()), 1);
}

TEST_F(ScheduleOoOTest, TestL1ReshapeSpillBufferFailed)
{
    ComputationalGraphBuilder subGraph;
    std::vector<std::string> tensorL1Names{"t1", "t2", "t3", "t4"};
    std::vector<MemoryType> tensorL1MemTypes{
        MemoryType::MEM_L1, MemoryType::MEM_L1, MemoryType::MEM_L1, MemoryType::MEM_L1};
    std::vector<std::string> tensorNames{"DDR", "UB2", "L0A1", "L0A2", "DDR1", "DDR2"};
    std::vector<MemoryType> tensorMemTypes{MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_UB,
                                           MemoryType::MEM_L0A,        MemoryType::MEM_L0A,
                                           MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_DEVICE_DDR};

    std::vector<std::string> operationNames{"L1_Alloc1",  "L1_Alloc2", "L1_Alloc3",  "UB_Alloc2", "L0A_Alloc1",
                                            "L0A_Alloc2", "COPY_IN5",  "UB_COPY_L1", "RESHAPE",   "COPY_IN1",
                                            "COPY_IN2",   "COPY_IN3",  "COPY_IN4",   "COPY_OUT1", "COPY_OUT2"};
    std::vector<Opcode> opCodes{Opcode::OP_L1_ALLOC,  Opcode::OP_L1_ALLOC,  Opcode::OP_L1_ALLOC, Opcode::OP_UB_ALLOC,
                                Opcode::OP_L0A_ALLOC, Opcode::OP_L0A_ALLOC, Opcode::OP_COPY_IN,  Opcode::OP_UB_COPY_L1,
                                Opcode::OP_RESHAPE,   Opcode::OP_COPY_IN,   Opcode::OP_COPY_IN,  Opcode::OP_COPY_IN,
                                Opcode::OP_COPY_IN,   Opcode::OP_COPY_OUT,  Opcode::OP_COPY_OUT};

    std::vector<std::vector<std::string>> ioperands{
        {}, {}, {}, {}, {}, {}, {"DDR"}, {"UB2"}, {"t1"}, {"t2"}, {"t2"}, {"L0A1"}, {"L0A2"}, {"t3"}, {"t4"}};
    std::vector<std::vector<std::string>> ooperands{{"t1"},   {"t3"},  {"t4"}, {"UB2"},  {"L0A1"},
                                                    {"L0A2"}, {"UB2"}, {"t1"}, {"t2"},   {"L0A1"},
                                                    {"L0A2"}, {"t3"},  {"t4"}, {"DDR1"}, {"DDR2"}};
    EXPECT_EQ(subGraph.AddTensors(DataType::DT_FP32, {256, 512}, tensorL1MemTypes, tensorL1Names, 0), true);
    EXPECT_EQ(subGraph.AddTensors(DataType::DT_FP32, {16, 16}, tensorMemTypes, tensorNames, 0), true);
    EXPECT_EQ(subGraph.AddOps(opCodes, ioperands, ooperands, operationNames, true), true);
    Function* function = subGraph.GetFunction();
    std::shared_ptr<LogicalTensor> tensor1 = subGraph.GetTensor("t2");
    tensor1->memoryrange.memId = subGraph.GetTensor("t1")->memoryrange.memId;
    OptimizeSort oooSort(function->Operations().DuplicatedOpList(), *function);
    Status res = oooSort.SortOps();
    EXPECT_EQ(res, SUCCESS);
    OoOScheduler oooSchedule(*function);
    res = oooSchedule.Init(oooSort.operations);
    EXPECT_EQ(res, SUCCESS);
    Operation* reshape = subGraph.GetOp("RESHAPE");
    Operation* alloc3 = subGraph.GetOp("L1_Alloc3");
    auto localBuffer0 = oooSchedule.localBufferMap[3];

    SpillInfo spillInfo;
    InitSpillInfo(spillInfo, 0, reshape);
    size_t pcIdx = 7;
    res = oooSchedule.SpillBuffer(spillInfo, alloc3, pcIdx, localBuffer0, true);
    EXPECT_EQ(res, FAILED);
}

TEST_F(ScheduleOoOTest, TestL1SpillBuffeFailed2)
{
    ComputationalGraphBuilder subGraph;
    std::vector<std::string> tensorNames{"t1", "t2", "t3", "t4"};
    std::vector<MemoryType> tensorMemTypes{
        MemoryType::MEM_L1, MemoryType::MEM_L1, MemoryType::MEM_L1, MemoryType::MEM_L1};

    std::vector<std::string> opNames{"L1_Alloc1", "L1_Alloc2", "assemble1", "assemble2", "COPY_IN"};
    std::vector<Opcode> opCodes{
        Opcode::OP_L1_ALLOC, Opcode::OP_L1_ALLOC, Opcode::OP_ASSEMBLE, Opcode::OP_ASSEMBLE, Opcode::OP_COPY_IN};

    std::vector<std::vector<std::string>> ioperands{{}, {}, {"t1"}, {"t2"}, {"t3"}};
    std::vector<std::vector<std::string>> ooperands{{"t1"}, {"t4"}, {"t3"}, {"t3"}, {"t4"}};
    EXPECT_EQ(subGraph.AddTensors(DataType::DT_FP32, {256, 256}, tensorMemTypes, tensorNames, 0), true);
    EXPECT_EQ(subGraph.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    Function* function = subGraph.GetFunction();
    subGraph.GetTensor("t2")->memoryrange.memId = subGraph.GetTensor("t1")->memoryrange.memId;
    subGraph.GetTensor("t3")->memoryrange.memId = subGraph.GetTensor("t1")->memoryrange.memId;
    OptimizeSort sort(function->Operations().DuplicatedOpList(), *function);
    Status res = sort.SortOps();
    EXPECT_EQ(res, SUCCESS);
    OoOScheduler oooSchedule(*function);
    res = oooSchedule.Init(sort.operations);
    EXPECT_EQ(res, SUCCESS);
    Operation* assemble = subGraph.GetOp("assemble1");
    Operation* alloc = subGraph.GetOp("L1_Alloc2");
    auto localBuffer = oooSchedule.localBufferMap[3];

    SpillInfo spillInfo;
    InitSpillInfo(spillInfo, 0, assemble);
    size_t pcIdx = 3;
    res = oooSchedule.SpillBuffer(spillInfo, alloc, pcIdx, localBuffer, true);
    EXPECT_EQ(res, FAILED);
}

TEST_F(ScheduleOoOTest, TestMixGraphAndDAV_3510)
{
    auto rootFuncPtr = std::make_shared<Function>(Program::GetInstance(), "TestParams", "TestParams", nullptr);
    rootFuncPtr->rootFunc_ = rootFuncPtr.get();
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestOOO", "TestOOO", rootFuncPtr.get());
    currFunctionPtr->paramConfigs_.OoOPreScheduleMethod = "PriorDFS";
    EXPECT_TRUE(currFunctionPtr != nullptr);
    currFunctionPtr->SetGraphType(GraphType::BLOCK_GRAPH);
    rootFuncPtr->rootFunc_->programs_.emplace(currFunctionPtr->GetFuncMagic(), currFunctionPtr.get());
    std::vector<int64_t> shape = {16, 16};
    auto shapeImme = OpImmediate::Specified(shape);

    auto tensor0 = CreateTensor(*currFunctionPtr, DataType::DT_FP32, shape, MEM_DEVICE_DDR, 0);
    auto tensor1 = CreateTensor(*currFunctionPtr, DataType::DT_FP32, shape, MEM_DEVICE_DDR, 1);
    auto tensor2 = CreateTensor(*currFunctionPtr, DataType::DT_FP32, shape, MEM_UB, 2);
    auto tensor3 = CreateTensor(*currFunctionPtr, DataType::DT_FP32, shape, MEM_UB, 3);
    auto tensor4 = CreateTensor(*currFunctionPtr, DataType::DT_FP32, shape, MEM_UB, 4);
    auto tensor5 = CreateTensor(*currFunctionPtr, DataType::DT_FP32, shape, MEM_L0A, 5);
    CreateAllocOp(*currFunctionPtr, tensor2, 1);
    CreateAllocOp(*currFunctionPtr, tensor3, 1);
    CreateAllocOp(*currFunctionPtr, tensor4, 1);
    currFunctionPtr->AddOperation(Opcode::OP_L0A_ALLOC, {}, LogicalTensors({tensor5}));
    CreateCopyOp(*currFunctionPtr, Opcode::OP_COPY_IN, tensor0, tensor2, shape);
    CreateCopyOp(*currFunctionPtr, Opcode::OP_COPY_IN, tensor1, tensor3, shape);
    CreateAddOp(*currFunctionPtr, tensor2, tensor3, tensor4);
    currFunctionPtr->AddOperation(Opcode::OP_UB_COPY_L1, LogicalTensors({tensor4}), LogicalTensors({tensor5}));
    for (auto& program : rootFuncPtr->rootFunc_->programs_) {
        ReorderOperations(*(program.second));
    }
    currFunctionPtr->EndFunction(nullptr);
    OoOSchedule oooSchedule;
    Platform::Instance().GetSoc().SetNPUArch(NPUArch::DAV_3510);
    EXPECT_EQ(oooSchedule.RunOnFunction(*rootFuncPtr), SUCCESS);
    Platform::Instance().GetSoc().SetNPUArch(NPUArch::DAV_UNKNOWN);
}

TEST_F(ScheduleOoOTest, TestCreateSpillCopyout)
{
    ComputationalGraphBuilder subGraph;
    std::vector<std::string> tensorNames{"L0A", "L0C", "L1"};
    std::vector<MemoryType> tensorMemTypes{MemoryType::MEM_L0A, MemoryType::MEM_L0C, MemoryType::MEM_L1};

    std::vector<Opcode> opCodes{
        Opcode::OP_COPY_IN, Opcode::OP_L0C_TO_L1, Opcode::OP_L0A_ALLOC, Opcode::OP_L0C_ALLOC, Opcode::OP_L1_ALLOC};
    std::vector<std::string> opNames{"copy_in", "L0C_L1", "L0A_ALLOC", "L0C_ALLOC", "L1_ALLOC"};
    std::vector<std::vector<std::string>> ioperands{{"L0A"}, {"L0C"}, {}, {}, {}};
    std::vector<std::vector<std::string>> ooperands{{"L0C"}, {"L1"}, {"L0A"}, {"L0C"}, {"L1"}};

    EXPECT_EQ(subGraph.AddTensors(DataType::DT_FP32, {16, 16}, tensorMemTypes, tensorNames, 0), true);
    EXPECT_EQ(subGraph.AddOps(opCodes, ioperands, ooperands, opNames, true), true);

    Function* function = subGraph.GetFunction();
    OptimizeSort sort(function->Operations().DuplicatedOpList(), *function);
    Status res = sort.SortOps();
    EXPECT_EQ(res, SUCCESS);
    OoOScheduler oooSchedule(*function);
    res = oooSchedule.Init(sort.operations);
    Operation* l0cCopyL1 = subGraph.GetOp("L0C_L1");
    Operation* copyIn = subGraph.GetOp("copy_in");
    Element scaleValue = Element(DataType::DT_UINT64, 0);
    l0cCopyL1->SetAttribute(OpAttributeKey::scaleValue, scaleValue);
    SpillInfo spillInfo;
    InitSpillInfo(spillInfo, 0, l0cCopyL1);
    Operation* spillCopyout = nullptr;
    res = oooSchedule.CreateSpillCopyout(copyIn, copyIn->GetOutputOperand(0),
    1, spillCopyout, spillInfo);
    EXPECT_EQ(res, SUCCESS);
}

TEST_F(ScheduleOoOTest, TestCopyModeFailed)
{
    ComputationalGraphBuilder subGraph;
    Function* function = subGraph.GetFunction();
    OoOScheduler ooOSchedule(*function);
    auto copyInOp = std::make_shared<Operation>(*function, Opcode::OP_COPY_IN);
    Status res = ooOSchedule.UpdateCopyOutMode(*copyInOp);
    EXPECT_EQ(res, FAILED);
    res = ooOSchedule.UpdateCopyInMode(*copyInOp);
    EXPECT_EQ(res, FAILED);
}

TEST_F(ScheduleOoOTest, TensorMemTypeMismatch)
{
    auto func = std::make_shared<Function>(Program::GetInstance(), "TestMemTypeMismatch", "TestMemTypeMismatch", nullptr);
    std::vector<int64_t> shape = {16, 16};
    
    auto t = std::make_shared<LogicalTensor>(*func, DT_FP32, shape);
    t->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR);
    t->SetMemoryTypeToBe(MemoryType::MEM_UB); // 不一致
    
    func->AddOperation(Opcode::OP_NOP, {t}, {t});
    func->inCasts_.push_back(t);

    OoOScheduleChecker checker;
    bool ok = checker.PreCheckTensorInfo(t);
    EXPECT_FALSE(ok);
}

TEST_F(ScheduleOoOTest, TensorMemIdInvalid)
{
    auto func = std::make_shared<Function>(Program::GetInstance(), "TestMemIdInvalid", "TestMemIdInvalid", nullptr);
    std::vector<int64_t> shape = {16, 16};
    
    auto t = std::make_shared<LogicalTensor>(*func, DT_FP32, shape);
    t->SetMemoryTypeOriginal(MemoryType::MEM_UB);
    t->SetMemoryTypeToBe(MemoryType::MEM_UB);
    t->memoryrange.memId = -1; // 非法
    
    OoOScheduleChecker checker;
    bool ok = checker.PreCheckTensorInfo(t);
    EXPECT_FALSE(ok);
}

TEST_F(ScheduleOoOTest, CallOpNotAllowed)
{
    ComputationalGraphBuilder subGraph;
    std::vector<std::string> tensorNames = {"t1", "t2"};
    std::vector<MemoryType> memTypes = {MEM_DEVICE_DDR, MEM_DEVICE_DDR};
    subGraph.AddTensors(DT_FP32, {16, 16}, memTypes, tensorNames, 0);

    std::vector<Opcode> opCodes = {Opcode::OP_CALL};
    std::vector<std::vector<std::string>> ins = {{"t1"}};
    std::vector<std::vector<std::string>> outs = {{"t2"}};
    std::vector<std::string> opNames = {"CALL_OP"};
    subGraph.AddOps(opCodes, ins, outs, opNames, true);

    Function* function = subGraph.GetFunction();
    OoOScheduleChecker checker;
    bool ret = checker.PreCheckOpInfo(function->Operations().DuplicatedOpList()[0]);
    EXPECT_FALSE(ret);
}

TEST_F(ScheduleOoOTest, ViewMemIdMismatch)
{
    auto func = std::make_shared<Function>(Program::GetInstance(), "ViewMemIdMismatch", "ViewMemIdMismatch", nullptr);
    std::vector<int64_t> shape = {16, 16};
    auto inTensor = std::make_shared<LogicalTensor>(*func, DT_FP32, shape);
    inTensor->SetMemoryTypeOriginal(MemoryType::MEM_UB);
    inTensor->SetMemoryTypeToBe(MemoryType::MEM_UB);
    inTensor->memoryrange.memId = 0;
    auto outTensor = std::make_shared<LogicalTensor>(*func, DT_FP32, shape);
    outTensor->SetMemoryTypeOriginal(MemoryType::MEM_UB);
    outTensor->SetMemoryTypeToBe(MemoryType::MEM_UB);
    outTensor->memoryrange.memId = 1;
    func->AddOperation(Opcode::OP_VIEW, {inTensor}, {outTensor});
    OoOScheduleChecker checker;
    bool ret = checker.PreCheckOpInfo(func->Operations().DuplicatedOpList()[0]);
    EXPECT_FALSE(ret);
}
} // namespace npu::tile_fwk
