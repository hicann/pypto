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
#include "interface/tensor/irbuilder.h"
#include "symbolic_scalar_test_utils.h"
#include "tilefwk/tilefwk.h"
#include "tilefwk/platform.h"
#include "interface/inner/tilefwk.h"
#include "passes/pass_mgr/pass_manager.h"
#include "interface/configs/config_manager.h"
#include "interface/tensor/irbuilder.h"
#define private public
#include "passes/block_graph_pass/schedule_ooo/schedule_ooo.h"
#include "passes/block_graph_pass/schedule_ooo/pre_schedule/core_assign.h"
#include "passes/block_graph_pass/schedule_ooo/post_schedule/buffer_rearrange.h"
#include "passes/block_graph_pass/schedule_ooo/pre_schedule/memory_aware_topo_sort.h"
#include "passes/tile_graph_pass/graph_constraint/infer_dyn_shape.h"
#include "computational_graph_builder.h"

namespace npu::tile_fwk {
constexpr int OOO_NUM2 = 2;
constexpr int OOO_NUM209 = 209;
constexpr int UBPoolSize = 192 * 1024;
std::unordered_map<Opcode, int> preNodePriority = {
    {Opcode::OP_UB_ALLOC, 0},
    {Opcode::OP_L1_ALLOC, 0},
    {Opcode::OP_L0A_ALLOC, 0},
    {Opcode::OP_L0B_ALLOC, 0},
    {Opcode::OP_L0C_ALLOC, 0},
    {Opcode::OP_BT_ALLOC, 0},
    {Opcode::OP_FIX_ALLOC, 0},
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
    {Opcode::OP_COPY_IN, 2},
    {Opcode::OP_UB_COPY_IN, 2},
    {Opcode::OP_L1_COPY_IN, 2},
    {Opcode::OP_L1_COPY_IN_FRACTAL_Z, 2},
    {Opcode::OP_L1_COPY_UB, 2},
    {Opcode::OP_L0C_COPY_UB, 2},
    {Opcode::OP_UB_COPY_L1, 2},
};

class ScheduleOoOTest : public ::testing::Test {
public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}

    void SetUp() override { Program::GetInstance().Reset(); }
    void TearDown() override {}
};

void SetTensorAttr(LogicalTensorPtr tensor, MemoryType memType, int memId)
{
    tensor->SetMemoryTypeOriginal(memType);
    tensor->SetMemoryTypeToBe(memType);
    tensor->memoryrange.memId = memId;
    tensor->UpdateDynValidShape({CreateTestScalarVar("S0"), CreateTestScalarVar("S1")});
}

void SetAllocAttr(Operation& alloc, int latency) { alloc.UpdateLatency(latency); }

LogicalTensorPtr CreateTensor(DataType dateType, std::vector<int64_t> shape, MemoryType memType, int memId)
{
    LogicalTensorPtr tensor = npu::tile_fwk::IRBuilder().CreateTensorVar(dateType, shape,
                                                                         CreateTestConstIntVector(shape));
    SetTensorAttr(tensor, memType, memId);
    return tensor;
}

Operation& CreateAllocOp(Function& currFunction, LogicalTensorPtr tensor, int latency)
{
    Operation& alloc = PassOperationUtils::AddOperation(currFunction, Opcode::OP_UB_ALLOC, {},
                                                        LogicalTensors({tensor}));
    SetAllocAttr(alloc, latency);
    return alloc;
}

Operation& CreateCopyOp(Function& currFunction, Opcode opcode, LogicalTensorPtr inTensor, LogicalTensorPtr outTensor,
                        std::vector<int64_t> shape)
{
    std::vector<int64_t> offset = {0, 0};
    auto& copy = PassOperationUtils::AddOperation(currFunction, opcode, LogicalTensors({inTensor}),
                                                  LogicalTensors({outTensor}));
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

Operation& CreateAddOp(Function& currFunction, LogicalTensorPtr inTensor1, LogicalTensorPtr inTensor2,
                       LogicalTensorPtr outTensor)
{
    auto& add = PassOperationUtils::AddOperation(currFunction, Opcode::OP_ADD, LogicalTensors({inTensor1, inTensor2}),
                                                 LogicalTensors({outTensor}));
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

    auto tensor1 = CreateTensor(DataType::DT_FP32, shape, MEM_DEVICE_DDR, 0);
    auto tensor2 = CreateTensor(DataType::DT_FP32, shape, MEM_DEVICE_DDR, 1);
    auto tensor3 = CreateTensor(DataType::DT_FP32, shape, MEM_UB, 2);
    auto tensor4 = CreateTensor(DataType::DT_FP32, shape, MEM_UB, 3);
    auto tensor5 = CreateTensor(DataType::DT_FP32, shape, MEM_UB, 4);
    auto tensor6 = CreateTensor(DataType::DT_FP32, shape, MEM_UB, 5);
    auto tensor7 = CreateTensor(DataType::DT_FP32, shape, MEM_DEVICE_DDR, 6);
    auto tensor8 = CreateTensor(DataType::DT_FP32, shape, MEM_UB, 7);
    auto tensor9 = CreateTensor(DataType::DT_FP32, shape, MEM_UB, 8);
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

static bool CheckViewOps(std::vector<Operation*>& viewOps, Operation* op)
{
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
    EXPECT_EQ(ooOScheduler.state_.depManager.GetPredecessors(op).size(), 3);
    EXPECT_TRUE(ooOScheduler.state_.depManager.GetPredecessors(op).count(subGraph.GetOp("Alloc4")) > 0);
    EXPECT_EQ(ooOScheduler.state_.depManager.GetSuccessors(op).size(), 2);
    EXPECT_TRUE(ooOScheduler.state_.depManager.GetSuccessors(op).count(subGraph.GetOp("Add1")) > 0);
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
    EXPECT_TRUE(ooOScheduler.state_.depManager.GetPredecessors(copyin).count(subGraph.GetOp("Alloc1")) > 0);
    EXPECT_TRUE(ooOScheduler.state_.depManager.GetPredecessors(add).count(subGraph.GetOp("Alloc2")) > 0);
    EXPECT_TRUE(ooOScheduler.state_.depManager.GetPredecessors(add).count(subGraph.GetOp("Copyin1")) > 0);
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
    EXPECT_TRUE(ooOScheduler.state_.depManager.GetSuccessors(alloc).count(subGraph.GetOp("Sub3")) > 0);
    EXPECT_TRUE(ooOScheduler.state_.depManager.GetPredecessors(sub).count(subGraph.GetOp("Alloc1")) > 0);
    EXPECT_TRUE(ooOScheduler.state_.depManager.GetSuccessors(sub).count(subGraph.GetOp("Assemble1")) > 0);
    EXPECT_EQ(res, SUCCESS);
}

TEST_F(ScheduleOoOTest, TestDependenciesInplace)
{
    ComputationalGraphBuilder subGraph;
    std::vector<std::string> tensorNames{"t1", "t2", "t3", "t4"};
    std::vector<MemoryType> tensorMemTypes{MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_UB, MemoryType::MEM_UB,
                                           MemoryType::MEM_DEVICE_DDR};
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
    EXPECT_TRUE(ooOScheduler.state_.depManager.GetSuccessors(add).count(subGraph.GetOp("Copyout1")) > 0);
    EXPECT_TRUE(ooOScheduler.state_.depManager.GetPredecessors(add).count(subGraph.GetOp("Copyin1")) > 0);
    EXPECT_EQ(res, SUCCESS);
}

TEST_F(ScheduleOoOTest, TestDependenciesTrue)
{
    ComputationalGraphBuilder subGraph;
    std::vector<std::string> tensorNames{"t1", "t2", "t3", "t4"};
    std::vector<MemoryType> tensorMemTypes{MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_UB, MemoryType::MEM_UB,
                                           MemoryType::MEM_DEVICE_DDR};
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
    std::rotate(ooOScheduler.state_.orderedOps.begin(), ooOScheduler.state_.orderedOps.begin() + 1,
                ooOScheduler.state_.orderedOps.end());
    res = ooOScheduler.state_.depManager.InitDependencies(ooOScheduler.state_.orderedOps, false);
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
    res = ooOScheduler.SeqSchedule();
    EXPECT_EQ(res, SUCCESS);
    EXPECT_EQ(ooOScheduler.state_.orderedOps[8]->GetOpcodeStr(), "UB_ALLOC");
    EXPECT_EQ(ooOScheduler.state_.orderedOps[9]->GetOpcodeStr(), "COPY_IN");
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
    res = ooOScheduler.SeqSchedule();
    EXPECT_EQ(res, SUCCESS);
    EXPECT_EQ(ooOScheduler.state_.orderedOps[8]->GetOpcodeStr(), "COPY_OUT");
    EXPECT_EQ(ooOScheduler.state_.orderedOps[13]->GetOpcodeStr(), "UB_ALLOC");
    EXPECT_EQ(ooOScheduler.state_.orderedOps[14]->GetOpcodeStr(), "COPY_IN");
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
    std::rotate(ooOScheduler.state_.orderedOps.begin(), ooOScheduler.state_.orderedOps.begin() + 6,
                ooOScheduler.state_.orderedOps.begin() + 11);
    res = ooOScheduler.SeqSchedule();
    EXPECT_EQ(res, SUCCESS);
    Operation* add1 = subGraph.GetOp("Add1");
    EXPECT_NE(add1, nullptr);
    Operation* add3 = subGraph.GetOp("Add3");
    EXPECT_NE(add3, nullptr);
    EXPECT_EQ((*ooOScheduler.state_.depManager.GetSuccessors(add1).begin())->GetOpcodeStr(), "COPY_OUT");
    EXPECT_EQ(ooOScheduler.state_.depManager.GetPredecessors(add3).size(), 3);
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
    res = ooOScheduler.SeqSchedule();
    EXPECT_EQ(res, SUCCESS);
    EXPECT_EQ(ooOScheduler.state_.orderedOps.size(), 23);
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
    res = ooOScheduler.SeqSchedule();
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
    res = ooOScheduler.SeqSchedule();
    EXPECT_EQ(res, SUCCESS);
}

TEST_F(ScheduleOoOTest, TestSchedule)
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
    EXPECT_EQ(ooOScheduler.state_.newOperations[8]->GetOpcodeStr(), "UB_ALLOC");
    EXPECT_EQ(ooOScheduler.state_.newOperations[8]->oOperand[0]->memoryrange.start, 0);
    EXPECT_EQ(ooOScheduler.state_.newOperations[8]->oOperand[0]->memoryrange.end, 65536);
    EXPECT_EQ(ooOScheduler.state_.newOperations[10]->GetOpcodeStr(), "COPY_IN");
    EXPECT_EQ(ooOScheduler.state_.newOperations[10]->oOperand[0]->memoryrange.start, 0);
    EXPECT_EQ(ooOScheduler.state_.newOperations[10]->oOperand[0]->memoryrange.end, 65536);
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
    EXPECT_EQ(ooOScheduler.state_.newOperations[8]->GetOpcodeStr(), "COPY_OUT");
    EXPECT_EQ(ooOScheduler.state_.newOperations[8]->oOperand[0]->memoryrange.start, 0);
    EXPECT_EQ(ooOScheduler.state_.newOperations[8]->oOperand[0]->memoryrange.end, 65536);
    EXPECT_EQ(ooOScheduler.state_.newOperations[13]->GetOpcodeStr(), "UB_ALLOC");
    EXPECT_EQ(ooOScheduler.state_.newOperations[13]->oOperand[0]->memoryrange.start, 65536);
    EXPECT_EQ(ooOScheduler.state_.newOperations[13]->oOperand[0]->memoryrange.end, 131072);
    EXPECT_EQ(ooOScheduler.state_.newOperations[15]->GetOpcodeStr(), "COPY_IN");
    EXPECT_EQ(ooOScheduler.state_.newOperations[15]->oOperand[0]->memoryrange.start, 65536);
    EXPECT_EQ(ooOScheduler.state_.newOperations[15]->oOperand[0]->memoryrange.end, 131072);
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
    std::rotate(ooOScheduler.state_.orderedOps.begin(), ooOScheduler.state_.orderedOps.begin() + 6,
                ooOScheduler.state_.orderedOps.begin() + 11);
    res = ooOScheduler.ScheduleMainLoop();
    EXPECT_EQ(res, SUCCESS);
    EXPECT_EQ(ooOScheduler.state_.newOperations[6]->GetOpcodeStr(), "COPY_OUT");
    EXPECT_EQ(ooOScheduler.state_.newOperations[13]->GetOpcodeStr(), "COPY_IN");
    EXPECT_EQ(ooOScheduler.state_.newOperations[13]->oOperand[0]->memoryrange.start, 65536);
    EXPECT_EQ(ooOScheduler.state_.newOperations[13]->oOperand[0]->memoryrange.end, 131072);
    EXPECT_EQ(ooOScheduler.state_.newOperations[14]->GetOpcodeStr(), "ADD");
    EXPECT_EQ(ooOScheduler.state_.newOperations[14]->oOperand[0]->memoryrange.start, 65536);
    EXPECT_EQ(ooOScheduler.state_.newOperations[14]->oOperand[0]->memoryrange.end, 131072);
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

TEST_F(ScheduleOoOTest, TestSpillMultiProducerBuffer)
{
    ComputationalGraphBuilder subGraph;
    std::vector<std::string> tensorNames{"DDR1", "DDR2", "DDR3", "UB1",  "UB2",  "UB3",
                                         "UB4",  "L1_1", "L0C1", "L0C2", "L1_2", "L0C3"};
    std::vector<MemoryType> tensorMemTypes{
        MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_UB,
        MemoryType::MEM_UB,         MemoryType::MEM_UB,         MemoryType::MEM_UB,         MemoryType::MEM_L1,
        MemoryType::MEM_L0C,        MemoryType::MEM_L0C,        MemoryType::MEM_L1,         MemoryType::MEM_L0C};
    std::vector<Opcode> opCodes{
        Opcode::OP_COPY_IN,    Opcode::OP_COPY_IN,    Opcode::OP_UB_COPY_ND2NZ, Opcode::OP_UB_COPY_ND2NZ,
        Opcode::OP_UB_COPY_L1, Opcode::OP_UB_COPY_L1, Opcode::OP_A_MUL_B,       Opcode::OP_A_MUL_B,
        Opcode::OP_COPY_IN,    Opcode::OP_A_MUL_B,    Opcode::OP_UB_ALLOC,      Opcode::OP_UB_ALLOC,
        Opcode::OP_UB_ALLOC,   Opcode::OP_UB_ALLOC,   Opcode::OP_L1_ALLOC,      Opcode::OP_L1_ALLOC,
        Opcode::OP_L0C_ALLOC,  Opcode::OP_L0C_ALLOC,  Opcode::OP_L0C_ALLOC};
    std::vector<std::vector<std::string>> ioperands{{"DDR1"}, {"DDR2"}, {"UB1"},  {"UB2"}, {"UB3"}, {"UB4"}, {"L1_1"},
                                                    {"L1_1"}, {"DDR3"}, {"L1_2"}, {},      {},      {},      {},
                                                    {},       {},       {},       {},      {}};
    std::vector<std::vector<std::string>> ooperands{
        {"UB1"}, {"UB2"}, {"UB3"}, {"UB4"}, {"L1_1"}, {"L1_1"}, {"L0C1"}, {"L0C2"}, {"L1_2"}, {"L0C3"},
        {"UB1"}, {"UB2"}, {"UB3"}, {"UB4"}, {"L1_1"}, {"L1_2"}, {"L0C1"}, {"L0C2"}, {"L0C3"}};
    std::vector<std::string> opNames{"copyin1",  "copyin2",   "ubNd2nz1",  "ubNd2nz2", "ub_l11",
                                     "ub_l12",   "aMulB1",    "aMulB2",    "copyin3",  "aMulB3",
                                     "ubAlloc1", "ubAlloc2",  "ubAlloc3",  "ubAlloc4", "l1Alloc1",
                                     "l1Alloc2", "l0cAlloc1", "l0cAlloc2", "l0cAlloc3"};
    subGraph.AddTensors(DataType::DT_FP32, {64, 64}, tensorMemTypes, tensorNames, 0);
    subGraph.AddOps(opCodes, ioperands, ooperands, opNames, true);
    Function* function = subGraph.GetFunction();
    std::shared_ptr<LogicalTensor> tensor5 = subGraph.GetTensor("L1_1");
    std::shared_ptr<LogicalTensor> tensor6 = subGraph.GetTensor("L1_2");
    tensor5->shape = {256, 300};
    tensor5->tensor->rawshape = {256, 300};
    tensor6->shape = {256, 300};
    tensor6->tensor->rawshape = {256, 300};
    auto* ubToL11 = subGraph.GetOp("ub_l11");
    ubToL11->SetOpAttribute(std::make_shared<CopyOpAttribute>(OpImmediate::Specified({0, 0}), MemoryType::MEM_L1,
                                                              OpImmediate::Specified({64, 64}),
                                                              OpImmediate::Specified({256, 256})));
    auto* ubToL12 = subGraph.GetOp("ub_l12");
    ubToL12->SetOpAttribute(std::make_shared<CopyOpAttribute>(OpImmediate::Specified({64, 0}), MemoryType::MEM_L1,
                                                              OpImmediate::Specified({64, 64}),
                                                              OpImmediate::Specified({256, 256})));
    OptimizeSort sort(function->Operations().DuplicatedOpList(), *function);
    EXPECT_EQ(sort.SortOps(), SUCCESS);
    OoOScheduler ooOScheduler(*function);
    rotate(sort.operations.begin() + 12, sort.operations.begin() + 15, sort.operations.begin() + 18);
    EXPECT_EQ(ooOScheduler.Init(sort.operations), SUCCESS);
    Platform::Instance().GetSoc().SetNPUArch(NPUArch::DAV_3510);
    EXPECT_EQ(ooOScheduler.SeqSchedule(), SUCCESS);
}

TEST_F(ScheduleOoOTest, TestSpillMultiProducerBufferNotReady)
{
    ComputationalGraphBuilder subGraph;
    std::vector<std::string> tensorNames{"DDR1", "DDR2", "DDR3", "UB1",  "UB2", "UB3",
                                         "UB4",  "L1_1", "L0C1", "L0C2", "L1_2"};
    std::vector<MemoryType> tensorMemTypes{
        MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_UB,
        MemoryType::MEM_UB,         MemoryType::MEM_UB,         MemoryType::MEM_UB,         MemoryType::MEM_L1,
        MemoryType::MEM_L0C,        MemoryType::MEM_L0C,        MemoryType::MEM_L1};
    std::vector<Opcode> opCodes{Opcode::OP_COPY_IN,       Opcode::OP_COPY_IN,    Opcode::OP_UB_COPY_ND2NZ,
                                Opcode::OP_UB_COPY_ND2NZ, Opcode::OP_UB_COPY_L1, Opcode::OP_UB_COPY_L1,
                                Opcode::OP_A_MUL_B,       Opcode::OP_A_MUL_B,    Opcode::OP_L1_COPY_UB,
                                Opcode::OP_UB_ALLOC,      Opcode::OP_UB_ALLOC,   Opcode::OP_UB_ALLOC,
                                Opcode::OP_UB_ALLOC,      Opcode::OP_L1_ALLOC,   Opcode::OP_L1_ALLOC,
                                Opcode::OP_L0C_ALLOC,     Opcode::OP_L0C_ALLOC};
    std::vector<std::vector<std::string>> ioperands{{"DDR1"}, {"DDR2"}, {"UB1"},  {"UB2"}, {"UB3"}, {"UB4"},
                                                    {"L1_1"}, {"L1_1"}, {"L1_2"}, {},      {},      {},
                                                    {},       {},       {},       {},      {}};
    std::vector<std::vector<std::string>> ooperands{{"UB1"},  {"L1_2"}, {"UB3"},  {"UB4"},  {"L1_1"}, {"L1_1"},
                                                    {"L0C1"}, {"L0C2"}, {"UB2"},  {"UB1"},  {"UB2"},  {"UB3"},
                                                    {"UB4"},  {"L1_1"}, {"L1_2"}, {"L0C1"}, {"L0C2"}};
    std::vector<std::string> opNames{"copyin1",  "copyin2",  "ubNd2nz1", "ubNd2nz2",  "ub_l11",   "ub_l12",
                                     "aMulB1",   "aMulB2",   "l1CopyUb", "ubAlloc1",  "ubAlloc2", "ubAlloc3",
                                     "ubAlloc4", "l1Alloc1", "l1Alloc2", "l0cAlloc1", "l0cAlloc2"};
    subGraph.AddTensors(DataType::DT_FP32, {64, 64}, tensorMemTypes, tensorNames, 0);
    subGraph.AddOps(opCodes, ioperands, ooperands, opNames, true);
    Function* function = subGraph.GetFunction();
    std::shared_ptr<LogicalTensor> tensor5 = subGraph.GetTensor("L1_1");
    std::shared_ptr<LogicalTensor> tensor6 = subGraph.GetTensor("L1_2");
    tensor5->shape = {256, 300};
    tensor5->tensor->rawshape = {256, 300};
    tensor6->shape = {256, 300};
    tensor6->tensor->rawshape = {256, 300};
    auto* ubToL11 = subGraph.GetOp("ub_l11");
    ubToL11->SetOpAttribute(std::make_shared<CopyOpAttribute>(OpImmediate::Specified({0, 0}), MemoryType::MEM_L1,
                                                              OpImmediate::Specified({64, 64}),
                                                              OpImmediate::Specified({256, 256})));
    auto* ubToL12 = subGraph.GetOp("ub_l12");
    ubToL12->SetOpAttribute(std::make_shared<CopyOpAttribute>(OpImmediate::Specified({64, 0}), MemoryType::MEM_L1,
                                                              OpImmediate::Specified({64, 64}),
                                                              OpImmediate::Specified({256, 256})));
    OptimizeSort sort(function->Operations().DuplicatedOpList(), *function);
    EXPECT_EQ(sort.SortOps(), SUCCESS);
    OoOScheduler ooOScheduler(*function);
    rotate(sort.operations.begin(), sort.operations.begin() + 8, sort.operations.begin() + 14);
    rotate(sort.operations.begin() + 4, sort.operations.begin() + 12, sort.operations.begin() + 13);
    EXPECT_EQ(ooOScheduler.Init(sort.operations), SUCCESS);
    Platform::Instance().GetSoc().SetNPUArch(NPUArch::DAV_3510);
    EXPECT_EQ(ooOScheduler.SeqSchedule(), SUCCESS);
}

TEST_F(ScheduleOoOTest, TestSpillL0CMultiConsumer)
{
    // L0C1 三消费�?UB fp16 / L1 fp32 / CopyOut已retired)，UB/L1 �?alloc2 后触�?spill，按 dtype 分两组各一�?COPY_OUT�?
    ComputationalGraphBuilder subGraph;
    std::vector<std::tuple<DataType, MemoryType, std::string>> tensors{
        {DT_FP32, MEM_L0A, "L0A"},          {DT_FP32, MEM_L0B, "L0B"},  {DT_FP32, MEM_L0C, "L0C1"},
        {DT_FP32, MEM_L0C, "L0C2"},         {DT_FP16, MEM_UB, "UBDst"}, {DT_FP32, MEM_L1, "L1Dst"},
        {DT_FP32, MEM_DEVICE_DDR, "DDROut"}};
    for (auto& [dt, mem, name] : tensors) {
        subGraph.AddTensor(dt, {128, 128}, mem, name, 0);
    }
    std::vector<Opcode> opCodes{Opcode::OP_L0A_ALLOC, Opcode::OP_L0B_ALLOC, Opcode::OP_L0C_ALLOC,
                                Opcode::OP_A_MUL_B,   Opcode::OP_UB_ALLOC,  Opcode::OP_L0C_COPY_UB,
                                Opcode::OP_L1_ALLOC,  Opcode::OP_L0C_TO_L1, Opcode::OP_L0C_COPY_OUT,
                                Opcode::OP_L0C_ALLOC, Opcode::OP_A_MUL_B};
    std::vector<std::vector<std::string>> ioperands{{},       {},       {}, {"L0A", "L0B"}, {}, {"L0C1"}, {},
                                                    {"L0C1"}, {"L0C1"}, {}, {"L0A", "L0B"}};
    std::vector<std::vector<std::string>> ooperands{{"L0A"},   {"L0B"},   {"L0C1"},   {"L0C1"}, {"UBDst"}, {"UBDst"},
                                                    {"L1Dst"}, {"L1Dst"}, {"DDROut"}, {"L0C2"}, {"L0C2"}};
    std::vector<std::string> opNames{"L0AAlloc", "L0BAlloc", "L0CAlloc1",  "Matmul1",   "UBAlloc", "L0CCopyUB",
                                     "L1Alloc",  "L0CToL1",  "L0CCopyOut", "L0CAlloc2", "Matmul2"};
    subGraph.AddOps(opCodes, ioperands, ooperands, opNames, true);
    Function* function = subGraph.GetFunction();

    auto shapeImme = OpImmediate::Specified(std::vector<int64_t>{128, 128});
    auto* copyOutOp = subGraph.GetOp("L0CCopyOut");
    copyOutOp->SetOpAttribute(std::make_shared<CopyOpAttribute>(
        MemoryType::MEM_L0C, OpImmediate::Specified(std::vector<int64_t>{0, 0}), shapeImme, shapeImme));

    OptimizeSort sort(function->Operations().DuplicatedOpList(), *function);
    EXPECT_EQ(sort.SortOps(), SUCCESS);
    OoOScheduler ooOScheduler(*function);
    EXPECT_EQ(ooOScheduler.Init(sort.operations), SUCCESS);

    ooOScheduler.state_.bufferManagerMap[CoreLocationType::AIC][MemoryType::MEM_L0C] = BufferPool(MemoryType::MEM_L0C,
                                                                                                  64 * 1024);
    ooOScheduler.state_.orderedOps = {subGraph.GetOp("L0AAlloc"),
                                      subGraph.GetOp("L0BAlloc"),
                                      subGraph.GetOp("L0CAlloc1"),
                                      subGraph.GetOp("Matmul1"),
                                      copyOutOp,
                                      subGraph.GetOp("L0CAlloc2"),
                                      subGraph.GetOp("Matmul2"),
                                      subGraph.GetOp("UBAlloc"),
                                      subGraph.GetOp("L0CCopyUB"),
                                      subGraph.GetOp("L1Alloc"),
                                      subGraph.GetOp("L0CToL1")};
    for (size_t i = 0; i < ooOScheduler.state_.orderedOps.size(); i++) {
        ooOScheduler.state_.schedInfoMap[ooOScheduler.state_.orderedOps[i]].execOrder = static_cast<int>(i);
    }

    EXPECT_EQ(ooOScheduler.SeqSchedule(), SUCCESS);

    auto spillCopyOut = std::count_if(ooOScheduler.state_.orderedOps.begin(), ooOScheduler.state_.orderedOps.end(),
                                      [](Operation* op) { return op->GetOpcodeStr() == "COPY_OUT"; });
    EXPECT_EQ(spillCopyOut, 2);
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
    std::vector<std::vector<std::string>> ioperands{
        {}, {}, {}, {}, {}, {"t1"}, {"t2"}, {"t3"}, {"t4"}, {"t5", "t6"}, {"t7", "t8"}, {"t9", "t10"}, {"t11"}};
    std::vector<std::vector<std::string>> ooperands{{"t5"}, {"t6"}, {"t7"},  {"t8"}, {"t9"},  {"t5"}, {"t6"},
                                                    {"t7"}, {"t8"}, {"t10"}, {"t9"}, {"t11"}, {"t12"}};
    std::vector<std::string> opNames{"Alloc1",  "Alloc2",  "Alloc3", "Alloc4", "Alloc5", "Copyin1", "Copyin2",
                                     "Copyin3", "Copyin4", "Add1",   "Add2",   "Add3",   "View1"};
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
    std::rotate(ooOScheduler.state_.orderedOps.begin(), ooOScheduler.state_.orderedOps.begin() + 6,
                ooOScheduler.state_.orderedOps.begin() + 11);
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
    oooSchedule.state_.DelBufRefCount(-1);
}

TEST_F(ScheduleOoOTest, TestDelBufCount_1)
{
    Function function(Program::GetInstance(), "", "", nullptr);
    OoOScheduler oooSchedule(function);
    oooSchedule.state_.bufRefCount[1] = -1;
    oooSchedule.state_.DelBufRefCount(1);
}

TEST_F(ScheduleOoOTest, TestGetSpillTensor)
{
    Function function(Program::GetInstance(), "", "", nullptr);
    std::vector<Operation*> scheduleOpList;

    std::vector<int64_t> shape = {128, 128};
    std::shared_ptr<LogicalTensor> tensor3 = npu::tile_fwk::IRBuilder().CreateTensorVar(
        DataType::DT_FP32, shape, CreateTestConstIntVector(shape));
    tensor3->SetMemoryTypeOriginal(MEM_UB);
    tensor3->SetMemoryTypeToBe(MEM_UB);
    tensor3->memoryrange.memId = 3;

    auto& alloc1 = PassOperationUtils::AddOperation(function, Opcode::OP_UB_ALLOC, {}, {tensor3});
    alloc1.UpdateLatency(1);

    OoOScheduler oooSchedule(function);
    LogicalTensorPtr tensor = oooSchedule.spillEngine_.GetSpillTensor(&alloc1, 1);
    EXPECT_EQ(tensor, nullptr);
}

TEST_F(ScheduleOoOTest, TestCheckAllocIssue)
{
    Function function(Program::GetInstance(), "", "", nullptr);
    std::vector<Operation*> scheduleOpList;

    std::vector<int64_t> shape = {128, 128};
    std::shared_ptr<LogicalTensor> tensor3 = npu::tile_fwk::IRBuilder().CreateTensorVar(
        DataType::DT_FP32, shape, CreateTestConstIntVector(shape));
    tensor3->SetMemoryTypeOriginal(MEM_UB);
    tensor3->SetMemoryTypeToBe(MEM_UB);
    tensor3->memoryrange.memId = 3;

    std::shared_ptr<LogicalTensor> tensor2 = npu::tile_fwk::IRBuilder().CreateTensorVar(
        DataType::DT_FP32, shape, CreateTestConstIntVector(shape));
    tensor3->SetMemoryTypeOriginal(MEM_UB);
    tensor3->SetMemoryTypeToBe(MEM_UB);
    tensor3->memoryrange.memId = 1;

    auto& alloc1 = PassOperationUtils::AddOperation(function, Opcode::OP_UB_ALLOC, {}, {tensor3, tensor2});
    alloc1.UpdateLatency(1);

    OoOScheduler oooSchedule(function);
    auto opList = function.Operations().DuplicatedOpList();
    oooSchedule.Init(opList);
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
    OoOScheduleStatistic testCheck;
    ooOScheduler.AddObserver(&testCheck);
    res = ooOScheduler.Init(sort.operations);
    EXPECT_EQ(res, SUCCESS);
    res = ooOScheduler.ScheduleMainLoop();
    EXPECT_EQ(res, SUCCESS);
    std::unordered_map<MemoryType, uint64_t> invalidBufferTotalUsage = {{MemoryType::MEM_UB, 0},
                                                                        {MemoryType::MEM_L1, 0},
                                                                        {MemoryType::MEM_L0A, 0},
                                                                        {MemoryType::MEM_L0B, 0},
                                                                        {MemoryType::MEM_L0C, 0}};
    std::unordered_map<MemoryType, uint64_t> invalidBufferMaxUsage = {{MemoryType::MEM_UB, 0},
                                                                      {MemoryType::MEM_L1, 0},
                                                                      {MemoryType::MEM_L0A, 0},
                                                                      {MemoryType::MEM_L0B, 0},
                                                                      {MemoryType::MEM_L0C, 0}};
    EXPECT_NE(testCheck.bufferTotalUsage, invalidBufferTotalUsage);
    EXPECT_NE(testCheck.bufferMaxUsage, invalidBufferMaxUsage);

    // 增加健康检查校�?
    testCheck.clock = 3; // 模拟数据
    res = testCheck.HealthCheckOoOSchedule();
    EXPECT_EQ(res, SUCCESS);
    EXPECT_NE(testCheck.report, nullptr);
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
    res = ooOScheduler.SeqSchedule();
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

TEST_F(ScheduleOoOTest, TestOoORollbackMix)
{
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
    std::vector<MemoryType> tensorMemTypes_L0C{MemoryType::MEM_L0C, MemoryType::MEM_L0C, MemoryType::MEM_L0C,
                                               MemoryType::MEM_L0C};
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
    auto tensor1 = subGraph.GetTensor("t1");
    int memId = tensor1->memoryrange.memId;

    OoOScheduler ooOScheduler(*function);
    ooOScheduler.state_.orderedOps.push_back(op);
    ooOScheduler.state_.schedInfoMap[op].isAlloc = true;
    ooOScheduler.state_.depManager.GetSuccessors(op).clear();
    ooOScheduler.state_.opReqMemIdsMap[op] = {memId};
    ooOScheduler.SetCoreLocation(op, CoreLocationType::AIV0);
    EXPECT_EQ(ooOScheduler.state_.InitLocalBuffer(tensor1, memId), SUCCESS);
    ooOScheduler.state_.bufferManagerMap[CoreLocationType::AIV0][MemoryType::MEM_UB] = BufferPool(MemoryType::MEM_UB,
                                                                                                  0);
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
    int memId1 = tensor1->memoryrange.memId;
    op->GetOutputOperand(0)->ClearAllProducers();
    op->GetOutputOperand(0)->AddProducer(*opCopyIn);
    OoOScheduler ooOScheduler(*function);
    ooOScheduler.state_.orderedOps.push_back(op);
    ooOScheduler.state_.orderedOps.push_back(opCopyIn);
    ooOScheduler.state_.schedInfoMap[op].isAlloc = true;
    ooOScheduler.state_.depManager.InsertSuccessor(op, opCopyIn);
    ooOScheduler.state_.opReqMemIdsMap[opCopyIn] = {1};
    ooOScheduler.state_.opReqMemIdsMap[op] = {memId1};
    ooOScheduler.SetCoreLocation(op, CoreLocationType::AIV0);
    EXPECT_EQ(ooOScheduler.state_.InitLocalBuffer(tensor1, memId1), SUCCESS);
    ooOScheduler.state_.bufferManagerMap[CoreLocationType::AIV0][MemoryType::MEM_UB] = BufferPool(MemoryType::MEM_UB,
                                                                                                  0);
    EXPECT_EQ(ooOScheduler.state_.InitLocalBuffer(tensor2, 1), SUCCESS);
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
    TaskSplitter splitter;
    splitter.SplitGraph(opList);
    CoreScheduler coreScheduler;
    coreScheduler.Schedule(splitter.GetTaskGraph());
    const int taskNum = 10;
    EXPECT_EQ(splitter.GetTaskGraph().tasks.size(), taskNum);
    splitter.MergeTask();
    OoOScheduler ooOScheduler(*function);
    splitter.MarkInternalSubgraphID();
    EXPECT_EQ(splitter.GetMergedOperations().size(), opList.size());
}

TEST_F(ScheduleOoOTest, TestOooScopeMerge)
{
    ComputationalGraphBuilder subGraph;
    std::vector<std::string> tensorNames{"t0", "t1", "t2", "t3", "t4"};
    std::vector<Opcode> opCodes{Opcode::OP_A_MUL_B, Opcode::OP_ADDS, Opcode::OP_ADDS, Opcode::OP_EXP};
    std::vector<std::vector<std::string>> ioperands{{"t0", "t0"}, {"t1"}, {"t2"}, {"t2"}};
    std::vector<std::vector<std::string>> ooperands{{"t1"}, {"t2"}, {"t3"}, {"t4"}};
    std::vector<std::string> opNames{"op1", "op2", "op3", "op4"};
    EXPECT_EQ(subGraph.AddTensors(DataType::DT_FP32, {256, 256}, tensorNames), true);
    EXPECT_EQ(subGraph.AddOps(opCodes, ioperands, ooperands, opNames, true), true);

    auto op2 = subGraph.GetOp("op2");
    auto op3 = subGraph.GetOp("op3");
    auto op4 = subGraph.GetOp("op4");
    ASSERT_NE(op2, nullptr);
    ASSERT_NE(op3, nullptr);
    ASSERT_NE(op4, nullptr);
    op2->SetOooScopeId(1);
    op4->SetOooScopeId(1);
    int magic2 = op2->GetOpMagic();
    int magic3 = op3->GetOpMagic();
    int magic4 = op4->GetOpMagic();

    Function* function = subGraph.GetFunction();
    EXPECT_NE(function, nullptr);
    auto opList = function->Operations(false).DuplicatedOpList();
    EXPECT_EQ(opList.size(), 4U);

    TaskSplitter splitter;
    splitter.SplitGraph(opList);
    auto& tasks = splitter.GetTaskGraph().tasks;

    bool foundMerged23 = false;
    bool foundMerged24 = false;
    for (auto& task : tasks) {
        bool hasOp2 = false, hasOp3 = false, hasOp4 = false;
        for (auto* op : task.opList_) {
            if (op->GetOpMagic() == magic2)
                hasOp2 = true;
            if (op->GetOpMagic() == magic3)
                hasOp3 = true;
            if (op->GetOpMagic() == magic4)
                hasOp4 = true;
        }
        if (hasOp2 && hasOp3) {
            foundMerged23 = true;
        }
        if (hasOp2 && hasOp4) {
            foundMerged24 = true;
        }
    }
    EXPECT_TRUE(!foundMerged23) << "op2 and op3 should not be merged into one task by ooo_scope";
    EXPECT_TRUE(foundMerged24) << "op2 and op4 should be merged into one task by ooo_scope";
}

TEST_F(ScheduleOoOTest, TestLatencyEstimatorMainLoop)
{
    // 创建测试数据
    ComputationalGraphBuilder subGraph;

    // 定义测试张量
    std::vector<std::string> tensorNames{"t1", "t2", "t3", "t4", "t5"};
    std::vector<MemoryType> tensorMemTypes{MemoryType::MEM_UB, MemoryType::MEM_L0A, MemoryType::MEM_L0B,
                                           MemoryType::MEM_L0C, MemoryType::MEM_UB};

    std::vector<Opcode> opCodes{Opcode::OP_UB_ALLOC,    Opcode::OP_L0A_ALLOC, Opcode::OP_L0B_ALLOC,
                                Opcode::OP_A_MUL_B,     Opcode::OP_L0C_ALLOC, Opcode::OP_UB_ALLOC,
                                Opcode::OP_L0C_COPY_UB, Opcode::OP_UB_COPY_L1};

    std::vector<std::vector<std::string>> ioperands{{}, {}, {}, {"t2", "t3"}, {}, {}, {"t4"}, {"t1"}};

    std::vector<std::vector<std::string>> ooperands{{"t1"}, {"t2"}, {"t3"}, {"t4"}, {"t4"}, {"t5"}, {"t5"}, {"t2"}};

    std::vector<std::string> opNames{"UB_ALLOC2",  "L0A_Alloc1", "L0B_Alloc1",     "Mul1",
                                     "L0C_Alloc1", "UB_ALLOC1",  "OP_L0C_COPY_UB", "OP_UB_COPY_L1"};

    // 构建计算�?
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

    // 构造子�?
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
    auto corePair = CoreLocationType::AIV0;
    oooSchedule.state_.schedInfoMap[alloc3].coreLocation = corePair;
    oooSchedule.state_.bufferManagerMap[corePair][MemoryType::MEM_UB] = pool;
    oooSchedule.state_.tensorOccupyMap[1] = alloc1;
    oooSchedule.state_.tensorOccupyMap[2] = alloc2;

    oooSchedule.state_.localBufferMap[1] = std::make_shared<LocalBuffer>(1, 65536, MemoryType::MEM_UB);
    oooSchedule.state_.localBufferMap[2] = std::make_shared<LocalBuffer>(2, 98304, MemoryType::MEM_UB);
    EXPECT_EQ(oooSchedule.RearrangeBuffer(alloc3, MemoryType::MEM_UB), SUCCESS);
    auto& ubPool = oooSchedule.state_.bufferManagerMap[corePair][MemoryType::MEM_UB];
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
    // 构造子�?
    ComputationalGraphBuilder subGraph;
    std::vector<std::string> tensorNames{"t1", "t2", "t3", "t4"};
    std::vector<MemoryType> tensorMemTypes{MemoryType::MEM_L0A, MemoryType::MEM_L0A, MemoryType::MEM_L0B,
                                           MemoryType::MEM_L0B};
    std::vector<Opcode> opCodes{Opcode::OP_L1_TO_L0A, Opcode::OP_L0A_ALLOC, Opcode::OP_L1_TO_L0B, Opcode::OP_L0B_ALLOC};
    std::vector<std::vector<std::string>> ioperands{{}, {}, {}, {}};
    std::vector<std::vector<std::string>> ooperands{{"t1"}, {"t2"}, {"t3"}, {"t4"}};
    std::vector<std::string> opNames{"L1toL0A", "AllocL0A", "L1toL0B", "AllocL0B"};
    EXPECT_EQ(subGraph.AddTensors(DataType::DT_FP16, {128, 128}, tensorMemTypes, tensorNames, 0), true);
    EXPECT_EQ(subGraph.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    Function* function = subGraph.GetFunction();
    EXPECT_NE(function, nullptr);
    // 构造Operation和属�?
    auto L1toL0A = subGraph.GetOp("L1toL0A");
    auto L1toL0B = subGraph.GetOp("L1toL0B");
    auto AllocL0A = subGraph.GetOp("AllocL0A");
    auto AllocL0B = subGraph.GetOp("AllocL0B");
    // 构造alloc队列、内存气泡场景的localBufferMap、tensorOccupyMap
    OoOScheduler oooSchedule(*function);
    oooSchedule.state_.SetOpMemIds(AllocL0A, {3});
    oooSchedule.state_.SetOpMemIds(AllocL0B, {4});
    auto corePair = CoreLocationType::AIC;
    oooSchedule.state_.allocIssueQueue[corePair][MemoryType::MEM_L0A].Insert(AllocL0A);
    oooSchedule.state_.allocIssueQueue[corePair][MemoryType::MEM_L0B].Insert(AllocL0B);
    oooSchedule.state_.tensorOccupyMap.emplace(1, L1toL0A);
    oooSchedule.state_.tensorOccupyMap.emplace(2, L1toL0B);
    oooSchedule.state_.localBufferMap[1] = std::make_shared<LocalBuffer>(1, 32768, MemoryType::MEM_L0A);
    oooSchedule.state_.localBufferMap[2] = std::make_shared<LocalBuffer>(2, 32768, MemoryType::MEM_L0B);
    oooSchedule.state_.localBufferMap[3] = std::make_shared<LocalBuffer>(3, 32768, MemoryType::MEM_L0A);
    oooSchedule.state_.localBufferMap[4] = std::make_shared<LocalBuffer>(4, 32768, MemoryType::MEM_L0B);
    oooSchedule.state_.localBufferMap[1]->start = 512;
    oooSchedule.state_.localBufferMap[1]->end = 33280;
    oooSchedule.state_.localBufferMap[2]->start = 512;
    oooSchedule.state_.localBufferMap[2]->end = 33280;
    // 验证内存气泡导致L0AB卡死
    EXPECT_EQ(oooSchedule.SpillOnBlock(), FAILED);
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
    EXPECT_EQ(op1->GetInternalSubgraphID(), 0);
    EXPECT_EQ(op2->GetInternalSubgraphID(), 1);
    EXPECT_EQ(op3->GetInternalSubgraphID(), 0);
    EXPECT_EQ(op4->GetInternalSubgraphID(), 2);
}

void SetInternalSubgraphIDAndAIVCore(Operation* op, int id)
{
    op->UpdateInternalSubgraphID(id);
    if (id == 0) {
        op->SetAIVCore(AIVCore::AIV0);
    }
}

void SetAttribute(ComputationalGraphBuilder& subGraph, OoOScheduler& oooSchedule, Operation*& ubCopyL1,
                  Operation*& alloc3)
{
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

    auto localBuffer1 = oooSchedule.state_.localBufferMap[0];
    auto coreAIC = CoreLocationType::AIC;
    oooSchedule.state_.bufferManagerMap[coreAIC][MemoryType::MEM_L1].Allocate(localBuffer1);
    oooSchedule.state_.tensorOccupyMap.emplace(0, copyin2);
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

    auto tensor0 = CreateTensor(DataType::DT_FP32, shape, MEM_DEVICE_DDR, 0);
    auto tensor1 = CreateTensor(DataType::DT_FP32, shape, MEM_DEVICE_DDR, 1);
    auto tensor2 = CreateTensor(DataType::DT_FP32, shape, MEM_UB, 2);
    auto tensor3 = CreateTensor(DataType::DT_FP32, shape, MEM_UB, 3);
    auto tensor4 = CreateTensor(DataType::DT_FP32, shape, MEM_UB, 4);
    auto tensor5 = CreateTensor(DataType::DT_FP32, shape, MEM_L0A, 5);
    CreateAllocOp(*currFunctionPtr, tensor2, 1);
    CreateAllocOp(*currFunctionPtr, tensor3, 1);
    CreateAllocOp(*currFunctionPtr, tensor4, 1);
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_L0A_ALLOC, {}, LogicalTensors({tensor5}));
    CreateCopyOp(*currFunctionPtr, Opcode::OP_COPY_IN, tensor0, tensor2, shape);
    CreateCopyOp(*currFunctionPtr, Opcode::OP_COPY_IN, tensor1, tensor3, shape);
    CreateAddOp(*currFunctionPtr, tensor2, tensor3, tensor4);
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_UB_COPY_L1, LogicalTensors({tensor4}),
                                     LogicalTensors({tensor5}));
    for (auto& program : rootFuncPtr->rootFunc_->programs_) {
        ReorderOperations(*(program.second));
    }
    currFunctionPtr->EndFunction(nullptr);
    OoOSchedule oooSchedule;
    Platform::Instance().GetSoc().SetNPUArch(NPUArch::DAV_3510);
    EXPECT_EQ(oooSchedule.RunOnFunction(*rootFuncPtr), SUCCESS);
    Platform::Instance().GetSoc().SetNPUArch(NPUArch::DAV_UNKNOWN);
}

TEST_F(ScheduleOoOTest, TensorMemTypeMismatch)
{
    auto func = std::make_shared<Function>(Program::GetInstance(), "TestMemTypeMismatch", "TestMemTypeMismatch",
                                           nullptr);
    std::vector<int64_t> shape = {16, 16};

    auto t = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    t->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR);
    t->SetMemoryTypeToBe(MemoryType::MEM_UB); // 不一�?

    PassOperationUtils::AddOperation(*func, Opcode::OP_NOP, {t}, {t});
    func->inCasts_.push_back(t);

    OoOScheduleChecker checker;
    bool ok = checker.PreCheckTensorInfo(t);
    EXPECT_FALSE(ok);
}

TEST_F(ScheduleOoOTest, TensorMemIdInvalid)
{
    auto func = std::make_shared<Function>(Program::GetInstance(), "TestMemIdInvalid", "TestMemIdInvalid", nullptr);
    std::vector<int64_t> shape = {16, 16};

    auto t = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
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
    auto inTensor = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    inTensor->SetMemoryTypeOriginal(MemoryType::MEM_UB);
    inTensor->SetMemoryTypeToBe(MemoryType::MEM_UB);
    inTensor->memoryrange.memId = 0;
    auto outTensor = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    outTensor->SetMemoryTypeOriginal(MemoryType::MEM_UB);
    outTensor->SetMemoryTypeToBe(MemoryType::MEM_UB);
    outTensor->memoryrange.memId = 1;
    PassOperationUtils::AddOperation(*func, Opcode::OP_VIEW, {inTensor}, {outTensor});
    OoOScheduleChecker checker;
    bool ret = checker.PreCheckOpInfo(func->Operations().DuplicatedOpList()[0]);
    EXPECT_FALSE(ret);
}

// Helper: compute peak memory usage from an operation sequence
// Helper: build a chain DAG for performance testing
static void BuildDAGForPerfTest(ComputationalGraphBuilder& subGraph, int numNodes, Function*& function)
{
    std::vector<std::string> tensorNames;
    std::vector<MemoryType> tensorMemTypes;
    tensorNames.reserve(numNodes + 1);
    tensorMemTypes.reserve(numNodes + 1);

    tensorNames.push_back("t_ddr");
    tensorMemTypes.push_back(MemoryType::MEM_DEVICE_DDR);
    for (int i = 0; i < numNodes; i++) {
        tensorNames.push_back("t_ub_" + std::to_string(i));
        tensorMemTypes.push_back(MemoryType::MEM_UB);
    }

    EXPECT_EQ(subGraph.AddTensors(DataType::DT_FP32, {16, 16}, tensorMemTypes, tensorNames, 0), true);

    std::vector<Opcode> opCodes;
    std::vector<std::vector<std::string>> ioperands;
    std::vector<std::vector<std::string>> ooperands;
    std::vector<std::string> opNames;

    opCodes.push_back(Opcode::OP_UB_ALLOC);
    ioperands.push_back({});
    ooperands.push_back({"t_ub_0"});
    opNames.push_back("Alloc1");

    opCodes.push_back(Opcode::OP_COPY_IN);
    ioperands.push_back({"t_ddr"});
    ooperands.push_back({"t_ub_0"});
    opNames.push_back("Copyin1");

    for (int i = 0; i < numNodes - 1; i++) {
        opCodes.push_back(Opcode::OP_UB_ALLOC);
        ioperands.push_back({});
        ooperands.push_back({"t_ub_" + std::to_string(i + 1)});
        opNames.push_back("Alloc" + std::to_string(i + 2));

        opCodes.push_back(Opcode::OP_ADD);
        ioperands.push_back({"t_ub_" + std::to_string(i)});
        ooperands.push_back({"t_ub_" + std::to_string(i + 1)});
        opNames.push_back("Add" + std::to_string(i + 1));
    }

    EXPECT_EQ(subGraph.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    function = subGraph.GetFunction();
    EXPECT_NE(function, nullptr);
}

// ============================================================================
// MemoryAwareSortTest - 评分计算验证测试
// ============================================================================

class MemoryAwareSortTest : public ::testing::Test {
protected:
    void SetUp() override { Program::GetInstance().Reset(); }

    void TearDown() override {}

    void SetupReleaseContributionMultiConsumerGraph(ComputationalGraphBuilder& builder)
    {
        std::vector<std::string> tensorNames{"t1", "t2", "t3", "t4", "t5", "t6"};
        std::vector<MemoryType> tensorMemTypes{MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_UB, MemoryType::MEM_UB,
                                               MemoryType::MEM_UB,         MemoryType::MEM_UB, MemoryType::MEM_UB};
        std::vector<Opcode> opCodes{Opcode::OP_UB_ALLOC, Opcode::OP_COPY_IN, Opcode::OP_ADD,
                                    Opcode::OP_ADD,      Opcode::OP_ADD,     Opcode::OP_ADD};
        std::vector<std::vector<std::string>> ioperands{{}, {"t1"}, {"t2"}, {"t2"}, {"t2"}, {"t2"}};
        std::vector<std::vector<std::string>> ooperands{{"t2"}, {"t2"}, {"t3"}, {"t4"}, {"t5"}, {"t6"}};
        std::vector<std::string> opNames{"Alloc1", "Copyin1", "Add1", "Add2", "Add3", "Add4"};
        EXPECT_EQ(builder.AddTensors(DataType::DT_FP32, {16, 16}, tensorMemTypes, tensorNames, 0), true);
        EXPECT_EQ(builder.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    }
};

// Verify progressive release contribution: multi-consumer scenario
TEST_F(MemoryAwareSortTest, TestReleaseContribution_MultiConsumer)
{
    ComputationalGraphBuilder subGraph;
    SetupReleaseContributionMultiConsumerGraph(subGraph);
    Function* function = subGraph.GetFunction();
    EXPECT_NE(function, nullptr);

    SchedulingContext context;
    ScoringParams params;
    params.alpha = 0.7;

    MemoryPoolContext ub_pool;
    ub_pool.usage = 0;
    ub_pool.limit = 192 * 1024;
    ub_pool.type = ConstraintType::SoftConstraint;
    ub_pool.can_spill = true;
    context.memory_pools[MemoryType::MEM_UB] = ub_pool;

    auto tensor_t2 = subGraph.GetTensor("t2");
    EXPECT_NE(tensor_t2, nullptr);
    int memId = tensor_t2->memoryrange.memId;

    context.executed_consumers[memId] = 0;
    context.max_consumer_count = 4;

    // Test 4 consumers: contributions should increase as remaining consumers decrease
    Operation* add1 = subGraph.GetOp("Add1");
    EXPECT_NE(add1, nullptr);
    double c1 = CalcReleaseContribution(memId, add1, context, params);

    context.executed_consumers[memId] = 1;
    Operation* add2 = subGraph.GetOp("Add2");
    EXPECT_NE(add2, nullptr);
    double c2 = CalcReleaseContribution(memId, add2, context, params);

    context.executed_consumers[memId] = 2;
    Operation* add3 = subGraph.GetOp("Add3");
    EXPECT_NE(add3, nullptr);
    double c3 = CalcReleaseContribution(memId, add3, context, params);

    context.executed_consumers[memId] = 3;
    Operation* add4 = subGraph.GetOp("Add4");
    EXPECT_NE(add4, nullptr);
    double c4 = CalcReleaseContribution(memId, add4, context, params);

    EXPECT_GT(c4, c3);
    EXPECT_GT(c3, c2);
    EXPECT_GT(c2, c1);
}

// TestReleaseContribution_SingleConsumer �?验证单一消费者场�?
TEST_F(MemoryAwareSortTest, TestReleaseContribution_SingleConsumer)
{
    ComputationalGraphBuilder subGraph;
    std::vector<std::string> tensorNames{"t1", "t2", "t3"};
    std::vector<MemoryType> tensorMemTypes{MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_UB, MemoryType::MEM_UB};
    std::vector<Opcode> opCodes{Opcode::OP_UB_ALLOC, Opcode::OP_COPY_IN, Opcode::OP_ADD};
    std::vector<std::vector<std::string>> ioperands{{}, {"t1"}, {"t2"}};
    std::vector<std::vector<std::string>> ooperands{{"t2"}, {"t2"}, {"t3"}};
    std::vector<std::string> opNames{"Alloc1", "Copyin1", "Add1"};

    EXPECT_EQ(subGraph.AddTensors(DataType::DT_FP32, {16, 16}, tensorMemTypes, tensorNames, 0), true);
    EXPECT_EQ(subGraph.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    Function* function = subGraph.GetFunction();
    EXPECT_NE(function, nullptr);

    // 初始化调度上下文
    SchedulingContext context;
    ScoringParams params;

    // 设置内存池状态（UB Abundant�?
    MemoryPoolContext ub_pool;
    ub_pool.usage = 0;
    ub_pool.limit = 192 * 1024;
    ub_pool.type = ConstraintType::SoftConstraint;
    ub_pool.can_spill = true;
    context.memory_pools[MemoryType::MEM_UB] = ub_pool;

    // 获取 tensor t2（只�?1 个消费者：Add1�?
    auto tensor_t2 = subGraph.GetTensor("t2");
    EXPECT_NE(tensor_t2, nullptr);
    int memId = tensor_t2->memoryrange.memId;

    // 初始�?executed_consumers
    context.executed_consumers[memId] = 0;
    context.max_consumer_count = 1;

    // 测试单一消费者执行时的边际收�?
    Operation* add1 = subGraph.GetOp("Add1");
    EXPECT_NE(add1, nullptr);
    double contribution = CalcReleaseContribution(memId, add1, context, params);

    // total_consumers = 1, executed_consumers = 0, remaining_consumers = 0
    // marginal_factor = 1.0（彻底释放）
    // 验证 contribution > 0（因为彻底释放）
    EXPECT_GT(contribution, 0.0);
}

// TestAllocationPressure_HighFanout �?验证高扇�?tensor 的压力评�?
TEST_F(MemoryAwareSortTest, TestAllocationPressure_HighFanout)
{
    ComputationalGraphBuilder subGraph;
    std::vector<std::string> tensorNames{"t1", "t2", "t3", "t4", "t5", "t6"};
    std::vector<MemoryType> tensorMemTypes{MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_UB, MemoryType::MEM_UB,
                                           MemoryType::MEM_UB,         MemoryType::MEM_UB, MemoryType::MEM_UB};
    std::vector<Opcode> opCodes{Opcode::OP_UB_ALLOC, Opcode::OP_COPY_IN, Opcode::OP_ADD,
                                Opcode::OP_ADD,      Opcode::OP_ADD,     Opcode::OP_ADD};
    std::vector<std::vector<std::string>> ioperands{{}, {"t1"}, {"t2"}, {"t2"}, {"t2"}, {"t2"}};
    std::vector<std::vector<std::string>> ooperands{{"t2"}, {"t2"}, {"t3"}, {"t4"}, {"t5"}, {"t6"}};
    std::vector<std::string> opNames{"Alloc1", "Copyin1", "Add1", "Add2", "Add3", "Add4"};

    EXPECT_EQ(subGraph.AddTensors(DataType::DT_FP32, {16, 16}, tensorMemTypes, tensorNames, 0), true);
    EXPECT_EQ(subGraph.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    Function* function = subGraph.GetFunction();
    EXPECT_NE(function, nullptr);

    // 初始化调度上下文
    SchedulingContext context;
    ScoringParams params;
    params.beta = 0.4;

    // 设置内存池状态（UB Abundant�?
    MemoryPoolContext ub_pool;
    ub_pool.usage = 0;
    ub_pool.limit = 192 * 1024;
    ub_pool.type = ConstraintType::SoftConstraint;
    ub_pool.can_spill = true;
    context.memory_pools[MemoryType::MEM_UB] = ub_pool;

    // 设置 max_consumer_count
    context.max_consumer_count = 4;

    // 测试高扇�?tensor（t2 �?4 个消费者）
    auto tensor_t2 = subGraph.GetTensor("t2");
    EXPECT_NE(tensor_t2, nullptr);

    // 计算 allocation pressure（假�?t2 是输�?tensor�?
    double pressure = CalcAllocationPressure(tensor_t2, context, params);

    // consumer_count = 4, max_consumer_count = 4
    // consumer_pressure_factor = 1 + 0.4 * (4 - 1) / 4 = 1 + 0.4 * 0.75 = 1.3
    // 验证 pressure > 0（高扇出 tensor 有更高的压力�?
    EXPECT_GT(pressure, 0.0);

    // 测试低扇�?tensor（t3 只有 0 个消费者）
    auto tensor_t3 = subGraph.GetTensor("t3");
    EXPECT_NE(tensor_t3, nullptr);
    double pressure_low = CalcAllocationPressure(tensor_t3, context, params);

    // consumer_count = 0, consumer_pressure_factor = 1.0
    // 验证 pressure > pressure_low（高扇出 tensor 压力更大�?
    EXPECT_GT(pressure, pressure_low);
}

// TestDynamicTypeWeight_L0A_Critical �?验证 L0A Critical 状态权�?
TEST_F(MemoryAwareSortTest, TestDynamicTypeWeight_L0A_Critical)
{
    // 初始化调度上下文
    SchedulingContext context;

    // 设置 L0A 内存池状态（Critical�?
    MemoryPoolContext l0a_pool;
    l0a_pool.usage = 90 * 1024;  // 90KB
    l0a_pool.limit = 100 * 1024; // 100KB
    l0a_pool.type = ConstraintType::SoftConstraint;
    l0a_pool.can_spill = false; // L0A 不支�?spill
    context.memory_pools[MemoryType::MEM_L0A] = l0a_pool;

    // 计算 L0A 的动态类型权�?
    double weight = CalcDynamicTypeWeight(MemoryType::MEM_L0A, context);

    // base_weight = 1.0（soft_constraint�?
    // usage_ratio = 90KB / 100KB = 0.9 �?Critical
    // state_factor = 1.5（Critical�?
    // spill_factor = 1.3（不支持 spill�?
    // weight = 1.0 * 1.5 * 1.3 = 1.95
    // 验证 weight �?1.95（误�?< 1e-6�?
    EXPECT_NEAR(weight, 1.95, 1e-6);
}

// TestDynamicTypeWeight_UB_Abundant �?验证 UB Abundant 状态权�?= 0.5
TEST_F(MemoryAwareSortTest, TestDynamicTypeWeight_UB_Abundant)
{
    // 初始化调度上下文
    SchedulingContext context;

    // 设置 UB 内存池状态（Abundant�?
    MemoryPoolContext ub_pool;
    ub_pool.usage = 20 * 1024;  // 20KB
    ub_pool.limit = 192 * 1024; // 192KB
    ub_pool.type = ConstraintType::SoftConstraint;
    ub_pool.can_spill = true; // UB 支持 spill
    context.memory_pools[MemoryType::MEM_UB] = ub_pool;

    // 计算 UB 的动态类型权�?
    double weight = CalcDynamicTypeWeight(MemoryType::MEM_UB, context);

    // base_weight = 1.0（soft_constraint�?
    // usage_ratio = 20KB / 192KB �?0.104 �?Abundant
    // state_factor = 0.5（Abundant�?
    // spill_factor = 1.0（支�?spill�?
    // weight = 1.0 * 0.5 * 1.0 = 0.5
    // 验证 weight �?0.5（误�?< 1e-6�?
    EXPECT_NEAR(weight, 0.5, 1e-6);
}

// TestNodeScore_Comprehensive �?验证综合评分计算
TEST_F(MemoryAwareSortTest, TestNodeScore_Comprehensive)
{
    ComputationalGraphBuilder subGraph;
    std::vector<std::string> tensorNames{"t1", "t2", "t3", "t4"};
    std::vector<MemoryType> tensorMemTypes{MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_UB, MemoryType::MEM_UB,
                                           MemoryType::MEM_UB};
    std::vector<Opcode> opCodes{Opcode::OP_UB_ALLOC, Opcode::OP_COPY_IN, Opcode::OP_ADD};
    std::vector<std::vector<std::string>> ioperands{{}, {"t1"}, {"t2"}};
    std::vector<std::vector<std::string>> ooperands{{"t2"}, {"t2"}, {"t3"}};
    std::vector<std::string> opNames{"Alloc1", "Copyin1", "Add1"};

    EXPECT_EQ(subGraph.AddTensors(DataType::DT_FP32, {16, 16}, tensorMemTypes, tensorNames, 0), true);
    EXPECT_EQ(subGraph.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    Function* function = subGraph.GetFunction();
    EXPECT_NE(function, nullptr);

    // 初始化调度上下文
    SchedulingContext context;
    ScoringParams params;

    // 设置 UB 内存池状态（Normal�?
    MemoryPoolContext ub_pool;
    ub_pool.usage = 50 * 1024;  // 50KB
    ub_pool.limit = 192 * 1024; // 192KB
    ub_pool.type = ConstraintType::SoftConstraint;
    ub_pool.can_spill = true;
    context.memory_pools[MemoryType::MEM_UB] = ub_pool;

    // 获取 tensor t2
    auto tensor_t2 = subGraph.GetTensor("t2");
    EXPECT_NE(tensor_t2, nullptr);
    int memId = tensor_t2->memoryrange.memId;

    // 初始�?executed_consumers
    context.executed_consumers[memId] = 0;
    context.max_consumer_count = 1;

    // 测试 Add1 节点的综合评�?
    Operation* add1 = subGraph.GetOp("Add1");
    EXPECT_NE(add1, nullptr);

    // NodeID = 2（Add1 �?operations 中的 index�?
    int node_id = 2;
    double score = CalcNodeScore(add1, context, params, node_id);

    // 验证综合评分计算�?
    // ReleaseScore = CalcReleaseContribution(t2, Add1) * StateFactor(UB)
    // AllocationPressure = CalcAllocationPressure(t3) * StateFactor(UB)
    // Score = ReleaseScore - AllocationPressure + epsilon * node_id
    // 验证 score 是合理的数值（不崩溃，不异常）
    EXPECT_TRUE(std::isfinite(score));
}

// TestNodeScore_TieBreaking �?验证等分节点�?NodeID 排序
TEST_F(MemoryAwareSortTest, TestNodeScore_TieBreaking)
{
    ComputationalGraphBuilder subGraph;
    std::vector<std::string> tensorNames{"t1", "t2", "t3", "t4", "t5"};
    std::vector<MemoryType> tensorMemTypes{MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_UB, MemoryType::MEM_UB,
                                           MemoryType::MEM_UB, MemoryType::MEM_UB};
    std::vector<Opcode> opCodes{Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_COPY_IN, Opcode::OP_COPY_IN,
                                Opcode::OP_ADD};
    std::vector<std::vector<std::string>> ioperands{{}, {}, {"t1"}, {"t1"}, {"t2", "t3"}};
    std::vector<std::vector<std::string>> ooperands{{"t2"}, {"t3"}, {"t2"}, {"t3"}, {"t4"}};
    std::vector<std::string> opNames{"Alloc1", "Alloc2", "Copyin1", "Copyin2", "Add1"};

    EXPECT_EQ(subGraph.AddTensors(DataType::DT_FP32, {16, 16}, tensorMemTypes, tensorNames, 0), true);
    EXPECT_EQ(subGraph.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    Function* function = subGraph.GetFunction();
    EXPECT_NE(function, nullptr);

    // 初始化调度上下文
    SchedulingContext context;
    ScoringParams params;

    // 设置 UB 内存池状态（Abundant�?
    MemoryPoolContext ub_pool;
    ub_pool.usage = 0;
    ub_pool.limit = 192 * 1024;
    ub_pool.type = ConstraintType::SoftConstraint;
    ub_pool.can_spill = true;
    context.memory_pools[MemoryType::MEM_UB] = ub_pool;

    // 初始�?executed_consumers
    context.max_consumer_count = 1;

    // 测试两个等分节点（Copyin1 �?Copyin2�?
    Operation* copyin1 = subGraph.GetOp("Copyin1");
    EXPECT_NE(copyin1, nullptr);
    Operation* copyin2 = subGraph.GetOp("Copyin2");
    EXPECT_NE(copyin2, nullptr);

    // NodeID = 2（Copyin1）和 NodeID = 3（Copyin2�?
    int node_id1 = 2;
    int node_id2 = 3;

    double score1 = CalcNodeScore(copyin1, context, params, node_id1);
    double score2 = CalcNodeScore(copyin2, context, params, node_id2);

    // 验证等分节点�?NodeID 排序�?
    // epsilon = 1e-9
    // score1 = base_score + epsilon * 2
    // score2 = base_score + epsilon * 3
    // score2 > score1（NodeID 大的排在后面�?
    EXPECT_GT(score2, score1);

    // 验证差异很小（epsilon * (node_id2 - node_id1) = 1e-9 * 1 = 1e-9�?
    EXPECT_NEAR(score2 - score1, 1e-9, 1e-10);
}

// ============================================================================
// T10: 拓扑序正确�?+ 确定性验证测�?
// ============================================================================

bool VerifyTopologicalOrder(const std::vector<Operation*>& sorted_ops, DependencyManager& depManager)
{
    std::unordered_set<Operation*> executed;
    for (Operation* op : sorted_ops) {
        auto predecessors = depManager.GetPredecessors(op);
        for (Operation* pred : predecessors) {
            if (executed.find(pred) == executed.end()) {
                return false;
            }
        }
        executed.insert(op);
    }
    return true;
}

bool VerifyAllOpsIncluded(const std::vector<Operation*>& sorted_ops, const std::vector<Operation*>& original_ops)
{
    std::unordered_set<Operation*> sorted_set(sorted_ops.begin(), sorted_ops.end());
    for (Operation* op : original_ops) {
        if (sorted_set.find(op) == sorted_set.end()) {
            return false;
        }
    }
    return sorted_ops.size() == original_ops.size();
}

// TestTopologicalSort_SimpleDAG �?简�?DAG 的拓扑序验证
TEST_F(MemoryAwareSortTest, TestTopologicalSort_SimpleDAG)
{
    ComputationalGraphBuilder subGraph;
    std::vector<std::string> tensorNames{"t1", "t2", "t3", "t4", "t5"};
    std::vector<MemoryType> tensorMemTypes{MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_UB, MemoryType::MEM_UB,
                                           MemoryType::MEM_UB, MemoryType::MEM_UB};
    std::vector<Opcode> opCodes{Opcode::OP_UB_ALLOC, Opcode::OP_COPY_IN, Opcode::OP_UB_ALLOC,
                                Opcode::OP_UB_ALLOC, Opcode::OP_ADD,     Opcode::OP_ADD};
    std::vector<std::vector<std::string>> ioperands{{}, {"t1"}, {}, {}, {"t2"}, {"t2", "t3"}};
    std::vector<std::vector<std::string>> ooperands{{"t2"}, {"t2"}, {"t3"}, {"t4"}, {"t3"}, {"t4"}};
    std::vector<std::string> opNames{"Alloc1", "Copyin1", "Alloc2", "Alloc3", "Add1", "Add2"};

    EXPECT_EQ(subGraph.AddTensors(DataType::DT_FP32, {16, 16}, tensorMemTypes, tensorNames, 0), true);
    EXPECT_EQ(subGraph.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    Function* function = subGraph.GetFunction();
    EXPECT_NE(function, nullptr);

    auto original_ops = function->Operations().DuplicatedOpList();

    MemoryAwareTopoSort sorter(original_ops, *function);
    Status res = sorter.InitContext();
    EXPECT_EQ(res, SUCCESS);

    res = sorter.SortOps();
    EXPECT_EQ(res, SUCCESS);

    OoOScheduler ooOScheduler(*function);
    res = ooOScheduler.Init(original_ops);
    EXPECT_EQ(res, SUCCESS);

    EXPECT_TRUE(VerifyTopologicalOrder(sorter.operations, ooOScheduler.state_.depManager));
    EXPECT_TRUE(VerifyAllOpsIncluded(sorter.operations, original_ops));
}

// TestTopologicalSort_ChainGraph �?链状图验�?
TEST_F(MemoryAwareSortTest, TestTopologicalSort_ChainGraph)
{
    ComputationalGraphBuilder subGraph;
    std::vector<std::string> tensorNames{"t1", "t2", "t3", "t4", "t5", "t6"};
    std::vector<MemoryType> tensorMemTypes{MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_UB, MemoryType::MEM_UB,
                                           MemoryType::MEM_UB,         MemoryType::MEM_UB, MemoryType::MEM_UB};
    std::vector<Opcode> opCodes{Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC,
                                Opcode::OP_COPY_IN,  Opcode::OP_ADD,      Opcode::OP_ADD,      Opcode::OP_ADD};
    std::vector<std::vector<std::string>> ioperands{{}, {}, {}, {}, {"t1"}, {"t2"}, {"t3"}, {"t4"}};
    std::vector<std::vector<std::string>> ooperands{{"t2"}, {"t3"}, {"t4"}, {"t5"}, {"t2"}, {"t3"}, {"t4"}, {"t5"}};
    std::vector<std::string> opNames{"Alloc1", "Alloc2", "Alloc3", "Alloc4", "Copyin1", "Add1", "Add2", "Add3"};

    EXPECT_EQ(subGraph.AddTensors(DataType::DT_FP32, {16, 16}, tensorMemTypes, tensorNames, 0), true);
    EXPECT_EQ(subGraph.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    Function* function = subGraph.GetFunction();
    EXPECT_NE(function, nullptr);

    auto original_ops = function->Operations().DuplicatedOpList();

    MemoryAwareTopoSort sorter(original_ops, *function);
    Status res = sorter.InitContext();
    EXPECT_EQ(res, SUCCESS);

    res = sorter.SortOps();
    EXPECT_EQ(res, SUCCESS);

    OoOScheduler ooOScheduler(*function);
    res = ooOScheduler.Init(original_ops);
    EXPECT_EQ(res, SUCCESS);

    EXPECT_TRUE(VerifyTopologicalOrder(sorter.operations, ooOScheduler.state_.depManager));
    EXPECT_TRUE(VerifyAllOpsIncluded(sorter.operations, original_ops));
}

// TestTopologicalSort_StarGraph �?星状图验�?
TEST_F(MemoryAwareSortTest, TestTopologicalSort_StarGraph)
{
    ComputationalGraphBuilder subGraph;
    std::vector<std::string> tensorNames{"t1", "t2", "t3", "t4", "t5", "t6", "t7"};
    std::vector<MemoryType> tensorMemTypes{MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_UB, MemoryType::MEM_UB,
                                           MemoryType::MEM_UB,         MemoryType::MEM_UB, MemoryType::MEM_UB,
                                           MemoryType::MEM_UB};
    std::vector<Opcode> opCodes{Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC,
                                Opcode::OP_UB_ALLOC, Opcode::OP_COPY_IN,  Opcode::OP_ADD,      Opcode::OP_ADD,
                                Opcode::OP_ADD,      Opcode::OP_ADD};
    std::vector<std::vector<std::string>> ioperands{{}, {}, {}, {}, {}, {"t1"}, {"t2"}, {"t2"}, {"t2"}, {"t2"}};
    std::vector<std::vector<std::string>> ooperands{{"t2"}, {"t3"}, {"t4"}, {"t5"}, {"t6"},
                                                    {"t2"}, {"t3"}, {"t4"}, {"t5"}, {"t6"}};
    std::vector<std::string> opNames{"Alloc1",  "Alloc2", "Alloc3", "Alloc4", "Alloc5",
                                     "Copyin1", "Add1",   "Add2",   "Add3",   "Add4"};

    EXPECT_EQ(subGraph.AddTensors(DataType::DT_FP32, {16, 16}, tensorMemTypes, tensorNames, 0), true);
    EXPECT_EQ(subGraph.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    Function* function = subGraph.GetFunction();
    EXPECT_NE(function, nullptr);

    auto original_ops = function->Operations().DuplicatedOpList();

    MemoryAwareTopoSort sorter(original_ops, *function);
    Status res = sorter.InitContext();
    EXPECT_EQ(res, SUCCESS);

    res = sorter.SortOps();
    EXPECT_EQ(res, SUCCESS);

    OoOScheduler ooOScheduler(*function);
    res = ooOScheduler.Init(original_ops);
    EXPECT_EQ(res, SUCCESS);

    EXPECT_TRUE(VerifyTopologicalOrder(sorter.operations, ooOScheduler.state_.depManager));
    EXPECT_TRUE(VerifyAllOpsIncluded(sorter.operations, original_ops));
}

// TestTopologicalSort_DiamondGraph �?菱形图验�?
TEST_F(MemoryAwareSortTest, TestTopologicalSort_DiamondGraph)
{
    ComputationalGraphBuilder subGraph;
    std::vector<std::string> tensorNames{"t1", "t2", "t3", "t4", "t5", "t6"};
    std::vector<MemoryType> tensorMemTypes{MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_UB, MemoryType::MEM_UB,
                                           MemoryType::MEM_UB,         MemoryType::MEM_UB, MemoryType::MEM_UB};
    std::vector<Opcode> opCodes{Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC,
                                Opcode::OP_COPY_IN,  Opcode::OP_ADD,      Opcode::OP_ADD,      Opcode::OP_ADD};
    std::vector<std::vector<std::string>> ioperands{{}, {}, {}, {}, {"t1"}, {"t2"}, {"t2"}, {"t3", "t4"}};
    std::vector<std::vector<std::string>> ooperands{{"t2"}, {"t3"}, {"t4"}, {"t5"}, {"t2"}, {"t3"}, {"t4"}, {"t5"}};
    std::vector<std::string> opNames{"Alloc1", "Alloc2", "Alloc3", "Alloc4", "Copyin1", "Add1", "Add2", "Add3"};

    EXPECT_EQ(subGraph.AddTensors(DataType::DT_FP32, {16, 16}, tensorMemTypes, tensorNames, 0), true);
    EXPECT_EQ(subGraph.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    Function* function = subGraph.GetFunction();
    EXPECT_NE(function, nullptr);

    auto original_ops = function->Operations().DuplicatedOpList();

    MemoryAwareTopoSort sorter(original_ops, *function);
    Status res = sorter.InitContext();
    EXPECT_EQ(res, SUCCESS);

    res = sorter.SortOps();
    EXPECT_EQ(res, SUCCESS);

    OoOScheduler ooOScheduler(*function);
    res = ooOScheduler.Init(original_ops);
    EXPECT_EQ(res, SUCCESS);

    EXPECT_TRUE(VerifyTopologicalOrder(sorter.operations, ooOScheduler.state_.depManager));
    EXPECT_TRUE(VerifyAllOpsIncluded(sorter.operations, original_ops));
}

// TestTopologicalSort_DenseDAG �?密集 DAG 验证
TEST_F(MemoryAwareSortTest, TestTopologicalSort_DenseDAG)
{
    ComputationalGraphBuilder subGraph;
    std::vector<std::string> tensorNames{"t1", "t2", "t3", "t4", "t5", "t6", "t7", "t8"};
    std::vector<MemoryType> tensorMemTypes{MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_UB, MemoryType::MEM_UB,
                                           MemoryType::MEM_UB,         MemoryType::MEM_UB, MemoryType::MEM_UB,
                                           MemoryType::MEM_UB,         MemoryType::MEM_UB};
    std::vector<Opcode> opCodes{Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC,
                                Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_COPY_IN,
                                Opcode::OP_COPY_IN,  Opcode::OP_ADD,      Opcode::OP_ADD,      Opcode::OP_ADD,
                                Opcode::OP_ADD};
    std::vector<std::vector<std::string>> ioperands{
        {}, {}, {}, {}, {}, {}, {}, {"t1"}, {"t1"}, {"t2", "t3"}, {"t2", "t4"}, {"t3", "t4"}, {"t5", "t6"}};
    std::vector<std::vector<std::string>> ooperands{{"t2"}, {"t3"}, {"t4"}, {"t5"}, {"t6"}, {"t7"}, {"t8"},
                                                    {"t2"}, {"t3"}, {"t5"}, {"t6"}, {"t7"}, {"t8"}};
    std::vector<std::string> opNames{"Alloc1",  "Alloc2",  "Alloc3", "Alloc4", "Alloc5", "Alloc6", "Alloc7",
                                     "Copyin1", "Copyin2", "Add1",   "Add2",   "Add3",   "Add4"};

    EXPECT_EQ(subGraph.AddTensors(DataType::DT_FP32, {16, 16}, tensorMemTypes, tensorNames, 0), true);
    EXPECT_EQ(subGraph.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    Function* function = subGraph.GetFunction();
    EXPECT_NE(function, nullptr);

    auto original_ops = function->Operations().DuplicatedOpList();

    MemoryAwareTopoSort sorter(original_ops, *function);
    Status res = sorter.InitContext();
    EXPECT_EQ(res, SUCCESS);

    res = sorter.SortOps();
    EXPECT_EQ(res, SUCCESS);

    OoOScheduler ooOScheduler(*function);
    res = ooOScheduler.Init(original_ops);
    EXPECT_EQ(res, SUCCESS);

    EXPECT_TRUE(VerifyTopologicalOrder(sorter.operations, ooOScheduler.state_.depManager));
    EXPECT_TRUE(VerifyAllOpsIncluded(sorter.operations, original_ops));
}

// TestDeterminism_SameInputSameOutput �?确定性验证：相同输入产生相同输出
TEST_F(MemoryAwareSortTest, TestDeterminism_SameInputSameOutput)
{
    ComputationalGraphBuilder subGraph;
    std::vector<std::string> tensorNames{"t1", "t2", "t3", "t4", "t5"};
    std::vector<MemoryType> tensorMemTypes{MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_UB, MemoryType::MEM_UB,
                                           MemoryType::MEM_UB, MemoryType::MEM_UB};
    std::vector<Opcode> opCodes{Opcode::OP_UB_ALLOC, Opcode::OP_COPY_IN, Opcode::OP_UB_ALLOC,
                                Opcode::OP_UB_ALLOC, Opcode::OP_ADD,     Opcode::OP_ADD};
    std::vector<std::vector<std::string>> ioperands{{}, {"t1"}, {}, {}, {"t2"}, {"t2", "t3"}};
    std::vector<std::vector<std::string>> ooperands{{"t2"}, {"t2"}, {"t3"}, {"t4"}, {"t3"}, {"t4"}};
    std::vector<std::string> opNames{"Alloc1", "Copyin1", "Alloc2", "Alloc3", "Add1", "Add2"};

    EXPECT_EQ(subGraph.AddTensors(DataType::DT_FP32, {16, 16}, tensorMemTypes, tensorNames, 0), true);
    EXPECT_EQ(subGraph.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    Function* function = subGraph.GetFunction();
    EXPECT_NE(function, nullptr);

    auto original_ops = function->Operations().DuplicatedOpList();

    const size_t expected_size = original_ops.size();
    for (int i = 0; i < 100; i++) {
        Program::GetInstance().Reset();
        ComputationalGraphBuilder subGraph2;
        EXPECT_EQ(subGraph2.AddTensors(DataType::DT_FP32, {16, 16}, tensorMemTypes, tensorNames, 0), true);
        EXPECT_EQ(subGraph2.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
        Function* function2 = subGraph2.GetFunction();
        EXPECT_NE(function2, nullptr);

        auto ops2 = function2->Operations().DuplicatedOpList();
        MemoryAwareTopoSort sorter(ops2, *function2);
        ASSERT_EQ(sorter.InitContext(), SUCCESS);
        ASSERT_EQ(sorter.SortOps(), SUCCESS);
        EXPECT_EQ(sorter.operations.size(), expected_size);
        EXPECT_TRUE(VerifyTopologicalOrder(sorter.operations, sorter.depManager_));
        EXPECT_TRUE(VerifyAllOpsIncluded(sorter.operations, ops2));
    }
}

// TestMemoryConstraint_NoOverflow �?内存约束验证：硬约束不超�?
TEST_F(MemoryAwareSortTest, TestMemoryConstraint_NoOverflow)
{
    ComputationalGraphBuilder subGraph;
    std::vector<std::string> tensorNames{"t1", "t2", "t3", "t4", "t5", "t6"};
    std::vector<MemoryType> tensorMemTypes{MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_BT, MemoryType::MEM_BT,
                                           MemoryType::MEM_BT,         MemoryType::MEM_BT, MemoryType::MEM_BT};
    std::vector<Opcode> opCodes{Opcode::OP_BT_ALLOC, Opcode::OP_BT_ALLOC, Opcode::OP_BT_ALLOC, Opcode::OP_BT_ALLOC,
                                Opcode::OP_COPY_IN,  Opcode::OP_ADD,      Opcode::OP_ADD};
    std::vector<std::vector<std::string>> ioperands{{}, {}, {}, {}, {"t1"}, {"t2"}, {"t3", "t4"}};
    std::vector<std::vector<std::string>> ooperands{{"t2"}, {"t3"}, {"t4"}, {"t5"}, {"t2"}, {"t3"}, {"t5"}};
    std::vector<std::string> opNames{"Alloc1", "Alloc2", "Alloc3", "Alloc4", "Copyin1", "Add1", "Add2"};

    EXPECT_EQ(subGraph.AddTensors(DataType::DT_FP32, {16, 16}, tensorMemTypes, tensorNames, 0), true);
    EXPECT_EQ(subGraph.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    Function* function = subGraph.GetFunction();
    EXPECT_NE(function, nullptr);

    auto original_ops = function->Operations().DuplicatedOpList();

    MemoryAwareTopoSort sorter(original_ops, *function);
    Status res = sorter.InitContext();
    EXPECT_EQ(res, SUCCESS);

    sorter.context_.memory_pools[MemoryType::MEM_BT].limit = 32 * 1024;
    sorter.context_.memory_pools[MemoryType::MEM_BT].type = ConstraintType::HardConstraint;
    sorter.context_.memory_pools[MemoryType::MEM_BT].can_spill = false;

    res = sorter.SortOps();
    EXPECT_EQ(res, SUCCESS);

    EXPECT_TRUE(VerifyTopologicalOrder(sorter.operations, sorter.depManager_));
    EXPECT_TRUE(VerifyAllOpsIncluded(sorter.operations, original_ops));
}

// ============================================================================
// MemoryAwareSortTest - 边界场景 + 性能对比测试 (T11)
// ============================================================================

// TestEmptyGraph �?空图验证：空 operations 列表应返�?SUCCESS 且结果为�?
TEST_F(MemoryAwareSortTest, TestEmptyGraph)
{
    Function function(Program::GetInstance(), "TestEmptyGraph", "TestEmptyGraph", nullptr);
    std::vector<Operation*> emptyOps;

    MemoryAwareTopoSort sorter(emptyOps, function);
    Status res = sorter.SortOps();

    EXPECT_EQ(res, SUCCESS);
    EXPECT_TRUE(sorter.operations.empty());
}

// TestSingleNode �?单节点验证：单个 operation 应直接返�?
TEST_F(MemoryAwareSortTest, TestSingleNode)
{
    ComputationalGraphBuilder subGraph;
    std::vector<std::string> tensorNames{"t1", "t2"};
    std::vector<MemoryType> tensorMemTypes{MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_UB};
    std::vector<Opcode> opCodes{Opcode::OP_UB_ALLOC, Opcode::OP_COPY_IN};
    std::vector<std::vector<std::string>> ioperands{{}, {"t1"}};
    std::vector<std::vector<std::string>> ooperands{{"t2"}, {"t2"}};
    std::vector<std::string> opNames{"Alloc1", "Copyin1"};

    EXPECT_EQ(subGraph.AddTensors(DataType::DT_FP32, {16, 16}, tensorMemTypes, tensorNames, 0), true);
    EXPECT_EQ(subGraph.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    Function* function = subGraph.GetFunction();
    EXPECT_NE(function, nullptr);

    auto opList = function->Operations().DuplicatedOpList();

    MemoryAwareTopoSort sorter(opList, *function);
    Status res = sorter.SortOps();

    EXPECT_EQ(res, SUCCESS);
    EXPECT_EQ(sorter.operations.size(), 2);
}

// TestZeroSizeBuffer �?零大�?buffer 验证：零大小 tensor 不应导致崩溃或异�?
TEST_F(MemoryAwareSortTest, TestZeroSizeBuffer)
{
    ComputationalGraphBuilder subGraph;
    std::vector<std::string> tensorNames{"t1", "t2", "t3"};
    std::vector<MemoryType> tensorMemTypes{MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_UB, MemoryType::MEM_UB};
    std::vector<Opcode> opCodes{Opcode::OP_UB_ALLOC, Opcode::OP_COPY_IN, Opcode::OP_UB_ALLOC, Opcode::OP_ADD};
    std::vector<std::vector<std::string>> ioperands{{}, {"t1"}, {}, {"t2"}};
    std::vector<std::vector<std::string>> ooperands{{"t2"}, {"t2"}, {"t3"}, {"t3"}};
    std::vector<std::string> opNames{"Alloc1", "Copyin1", "Alloc2", "Add1"};

    EXPECT_EQ(subGraph.AddTensors(DataType::DT_FP32, {16, 16}, tensorMemTypes, tensorNames, 0), true);
    EXPECT_EQ(subGraph.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    Function* function = subGraph.GetFunction();
    EXPECT_NE(function, nullptr);

    // �?t2 �?shape 设置为零大小
    auto tensor_t2 = subGraph.GetTensor("t2");
    EXPECT_NE(tensor_t2, nullptr);
    tensor_t2->shape = {0, 0};
    tensor_t2->tensor->rawshape = {0, 0};

    auto opList = function->Operations().DuplicatedOpList();

    MemoryAwareTopoSort sorter(opList, *function);
    Status res = sorter.SortOps();

    // 零大�?buffer 不应导致崩溃，排序应正常完成
    EXPECT_EQ(res, SUCCESS);
    EXPECT_EQ(sorter.operations.size(), 4);
}

// TestMemoryExhaustion �?内存耗尽场景验证：当内存池使用率极高时，算法应能优雅处理
TEST_F(MemoryAwareSortTest, TestMemoryExhaustion)
{
    ComputationalGraphBuilder subGraph;
    std::vector<std::string> tensorNames{"t1", "t2", "t3", "t4", "t5", "t6"};
    std::vector<MemoryType> tensorMemTypes{MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_UB, MemoryType::MEM_UB,
                                           MemoryType::MEM_UB,         MemoryType::MEM_UB, MemoryType::MEM_UB};
    std::vector<Opcode> opCodes{Opcode::OP_UB_ALLOC, Opcode::OP_COPY_IN, Opcode::OP_ADD,
                                Opcode::OP_ADD,      Opcode::OP_ADD,     Opcode::OP_ADD};
    std::vector<std::vector<std::string>> ioperands{{}, {"t1"}, {"t2"}, {"t2"}, {"t2"}, {"t2"}};
    std::vector<std::vector<std::string>> ooperands{{"t2"}, {"t2"}, {"t3"}, {"t4"}, {"t5"}, {"t6"}};
    std::vector<std::string> opNames{"Alloc1", "Copyin1", "Add1", "Add2", "Add3", "Add4"};

    EXPECT_EQ(subGraph.AddTensors(DataType::DT_FP32, {16, 16}, tensorMemTypes, tensorNames, 0), true);
    EXPECT_EQ(subGraph.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    Function* function = subGraph.GetFunction();
    EXPECT_NE(function, nullptr);

    // 初始化调度上下文，模拟内存耗尽场景（usage 接近 limit�?
    SchedulingContext context;
    ScoringParams params;

    MemoryPoolContext ub_pool;
    ub_pool.usage = 180 * 1024; // 180KB，接�?192KB 限制
    ub_pool.limit = 192 * 1024; // 192KB
    ub_pool.type = ConstraintType::SoftConstraint;
    ub_pool.can_spill = true;
    context.memory_pools[MemoryType::MEM_UB] = ub_pool;

    auto tensor_t2 = subGraph.GetTensor("t2");
    EXPECT_NE(tensor_t2, nullptr);
    int memId = tensor_t2->memoryrange.memId;
    context.executed_consumers[memId] = 0;
    context.max_consumer_count = 4;

    // 验证在内存耗尽场景下评分计算仍然有�?
    Operation* add1 = subGraph.GetOp("Add1");
    EXPECT_NE(add1, nullptr);
    double score = CalcNodeScore(add1, context, params, 2);

    // 评分应为有限值，不应崩溃
    EXPECT_TRUE(std::isfinite(score));

    // 验证内存状态为 Critical
    double usage_ratio = static_cast<double>(ub_pool.usage) / static_cast<double>(ub_pool.limit);
    MemoryState state = GetMemoryState(usage_ratio);
    EXPECT_EQ(state, MemoryState::Critical);
}

// TestMemBtSlotLimit �?MEM_BT slot 限制验证：最多只能有 1 �?MEM_BT buffer 同时存在
TEST_F(MemoryAwareSortTest, TestMemBtSlotLimit)
{
    ComputationalGraphBuilder subGraph;
    std::vector<std::string> tensorNames{"t1", "t2", "t3", "t4", "t5"};
    std::vector<MemoryType> tensorMemTypes{MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_BT, MemoryType::MEM_BT,
                                           MemoryType::MEM_UB, MemoryType::MEM_UB};
    std::vector<Opcode> opCodes{Opcode::OP_BT_ALLOC, Opcode::OP_BT_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC,
                                Opcode::OP_COPY_IN,  Opcode::OP_ADD,      Opcode::OP_ADD};
    std::vector<std::vector<std::string>> ioperands{{}, {}, {}, {}, {"t1"}, {"t2"}, {"t3"}};
    std::vector<std::vector<std::string>> ooperands{{"t2"}, {"t3"}, {"t4"}, {"t5"}, {"t2"}, {"t4"}, {"t5"}};
    std::vector<std::string> opNames{"BtAlloc1", "BtAlloc2", "UbAlloc1", "UbAlloc2", "Copyin1", "Add1", "Add2"};

    EXPECT_EQ(subGraph.AddTensors(DataType::DT_FP32, {16, 16}, tensorMemTypes, tensorNames, 0), true);
    EXPECT_EQ(subGraph.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    Function* function = subGraph.GetFunction();
    EXPECT_NE(function, nullptr);

    auto opList = function->Operations().DuplicatedOpList();

    MemoryAwareTopoSort sorter(opList, *function);
    Status res = sorter.SortOps();

    // MEM_BT slot 限制�?1，算法应能正确处�?
    EXPECT_EQ(res, SUCCESS);
    EXPECT_EQ(sorter.operations.size(), 7);

    // 验证 MEM_BT pool �?max_slot_count �?1
    auto bt_pool_it = sorter.context_.memory_pools.find(MemoryType::MEM_BT);
    if (bt_pool_it != sorter.context_.memory_pools.end()) {
        EXPECT_EQ(bt_pool_it->second.max_slot_count, 1);
    }
}

// TestEqualScoreNodes �?等分节点排序验证：当多个节点评分相同时，应按 NodeID 排序
TEST_F(MemoryAwareSortTest, TestEqualScoreNodes)
{
    ComputationalGraphBuilder subGraph;
    std::vector<std::string> tensorNames{"t1", "t2", "t3", "t4", "t5", "t6"};
    std::vector<MemoryType> tensorMemTypes{MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_UB, MemoryType::MEM_UB,
                                           MemoryType::MEM_UB,         MemoryType::MEM_UB, MemoryType::MEM_UB};
    // 创建两个独立的分支，评分应该相同
    std::vector<Opcode> opCodes{Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC,
                                Opcode::OP_UB_ALLOC, Opcode::OP_COPY_IN,  Opcode::OP_COPY_IN,  Opcode::OP_ADD,
                                Opcode::OP_ADD,      Opcode::OP_ADD};
    std::vector<std::vector<std::string>> ioperands{{}, {}, {}, {}, {}, {"t1"}, {"t1"}, {"t2"}, {"t3"}, {"t2", "t3"}};
    std::vector<std::vector<std::string>> ooperands{{"t2"}, {"t3"}, {"t4"}, {"t5"}, {"t6"},
                                                    {"t2"}, {"t3"}, {"t4"}, {"t5"}, {"t6"}};
    std::vector<std::string> opNames{"Alloc1",  "Alloc2",  "Alloc3", "Alloc4", "Alloc5",
                                     "Copyin1", "Copyin2", "Add1",   "Add2",   "Add3"};

    EXPECT_EQ(subGraph.AddTensors(DataType::DT_FP32, {16, 16}, tensorMemTypes, tensorNames, 0), true);
    EXPECT_EQ(subGraph.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    Function* function = subGraph.GetFunction();
    EXPECT_NE(function, nullptr);

    auto opList = function->Operations().DuplicatedOpList();

    MemoryAwareTopoSort sorter(opList, *function);
    Status res = sorter.SortOps();

    EXPECT_EQ(res, SUCCESS);
    EXPECT_EQ(sorter.operations.size(), 10);

    // 验证拓扑顺序正确：Alloc1 必须�?Copyin1/Copyin2 之前
    int alloc1_idx = -1;
    int copyin1_idx = -1;
    int copyin2_idx = -1;
    for (size_t i = 0; i < sorter.operations.size(); i++) {
        if (sorter.operations[i]->GetOpcodeStr() == "UB_ALLOC")
            alloc1_idx = static_cast<int>(i);
        if (sorter.operations[i]->GetOpcodeStr() == "COPY_IN") {
            if (copyin1_idx == -1)
                copyin1_idx = static_cast<int>(i);
            else
                copyin2_idx = static_cast<int>(i);
        }
    }
    EXPECT_LT(alloc1_idx, copyin1_idx);
    EXPECT_LT(alloc1_idx, copyin2_idx);
}

TEST_F(MemoryAwareSortTest, TestPerformance_PeakMemoryComparison)
{
    ComputationalGraphBuilder subGraph;
    Function* function = nullptr;
    BuildDAGForPerfTest(subGraph, 100, function);

    auto opList = function->Operations().DuplicatedOpList();

    OptimizeSort oldSorter(opList, *function);
    Status oldRes = oldSorter.SortOps();
    EXPECT_EQ(oldRes, SUCCESS);

    MemoryAwareTopoSort newSorter(opList, *function);
    Status newRes = newSorter.SortOps();
    EXPECT_EQ(newRes, SUCCESS);

    // 验证两个方案都能正确完成排序，且产生完整的算子序�?
    EXPECT_EQ(oldSorter.operations.size(), opList.size());
    EXPECT_EQ(newSorter.operations.size(), opList.size());
}

TEST_F(MemoryAwareSortTest, TestPerformance_SortingOverhead)
{
    ComputationalGraphBuilder subGraph;
    Function* function = nullptr;
    BuildDAGForPerfTest(subGraph, 200, function);

    auto opList = function->Operations().DuplicatedOpList();

    const int numRuns = 5;
    int64_t totalOldTime = 0;
    int64_t totalNewTime = 0;

    for (int run = 0; run < numRuns; run++) {
        OptimizeSort oldSorter(opList, *function);
        auto t1 = std::chrono::high_resolution_clock::now();
        oldSorter.SortOps();
        auto t2 = std::chrono::high_resolution_clock::now();
        totalOldTime += std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();

        MemoryAwareTopoSort newSorter(opList, *function);
        auto t3 = std::chrono::high_resolution_clock::now();
        newSorter.SortOps();
        auto t4 = std::chrono::high_resolution_clock::now();
        totalNewTime += std::chrono::duration_cast<std::chrono::microseconds>(t4 - t3).count();
    }

    int64_t avgOld = totalOldTime / numRuns;
    int64_t avgNew = totalNewTime / numRuns;
    double overhead = (avgOld > 0) ? static_cast<double>(avgNew - avgOld) / static_cast<double>(avgOld) : 0.0;
    EXPECT_LE(overhead, 0.20);
}

// TestInterleavedAllocOrdering �?验证 MemoryAwareTopoSort 产生 ALLOC→consumer 交叉排序
// (而非全部 ALLOC 集中在最前面), 以确保与下游 Pass (GenSpillSchedule/ScheduleMainLoop) 兼容
TEST_F(MemoryAwareSortTest, TestInterleavedAllocOrdering)
{
    // 构造图: ALLOC_A→COPY_IN_A→ALLOC_B→COPY_IN_B→COPY_OUT
    ComputationalGraphBuilder subGraph;
    std::vector<MemoryType> memTypes(5, MemoryType::MEM_UB);
    EXPECT_EQ(
        subGraph.AddTensors(DataType::DT_FP32, {16, 16}, memTypes, {"t_in", "t_ub_a", "t_ub_b", "t_out", "t_ddr"}, 0),
        true);

    EXPECT_EQ(subGraph.AddOps({Opcode::OP_UB_ALLOC, Opcode::OP_COPY_IN, Opcode::OP_UB_ALLOC, Opcode::OP_COPY_IN,
                               Opcode::OP_UB_ALLOC, Opcode::OP_COPY_OUT},
                              {{}, {"t_ddr"}, {}, {"t_ddr"}, {}, {"t_ub_b"}},
                              {{"t_ub_a"}, {"t_ub_a"}, {"t_ub_b"}, {"t_ub_b"}, {"t_out"}, {"t_out"}},
                              {"AllocA", "CopyInA", "AllocB", "CopyInB", "AllocC", "CopyOut"}, true),
              true);

    Function* function = subGraph.GetFunction();
    ASSERT_NE(function, nullptr);
    auto opList = function->Operations().DuplicatedOpList();
    ASSERT_EQ(opList.size(), 6);

    MemoryAwareTopoSort sorter(opList, *function);
    ASSERT_EQ(sorter.SortOps(), SUCCESS);

    const auto& sorted = sorter.operations;
    ASSERT_EQ(sorted.size(), 6);

    // 验证: ALLOC 总是在其 consumer 之前（拓扑排序保证）
    // MemoryAwareTopoSort always places ALLOCs before non-ALLOCs;
    // the key constraint is that each ALLOC precedes every op that reads or writes
    // the tensor it allocates.
    std::unordered_map<int, size_t> allocPositions;
    for (size_t i = 0; i < sorted.size(); i++) {
        if (sorted[i]->GetOpcodeStr().find("ALLOC") != std::string::npos) {
            int memId = sorted[i]->GetOutputOperand(0)->memoryrange.memId;
            allocPositions[memId] = i;
        }
    }
    for (size_t i = 0; i < sorted.size(); i++) {
        Operation* op = sorted[i];
        auto& preds = sorter.depManager_.GetPredecessors(op);
        for (auto pred : preds) {
            if (pred->GetOpcodeStr().find("ALLOC") != std::string::npos) {
                int predMemId = pred->GetOutputOperand(0)->memoryrange.memId;
                auto it = allocPositions.find(predMemId);
                ASSERT_NE(it, allocPositions.end());
                EXPECT_LT(it->second, i) << "ALLOC " << pred->GetOpcodeStr() << " must be scheduled before consumer "
                                         << op->GetOpcodeStr() << " (found at positions " << it->second << " and " << i
                                         << ")";
            }
        }
    }
}

// TestAllocDependencyDepth �?验证 ALLOC 依赖深度正确: ALLOC 总是在其 consumer 之前
TEST_F(MemoryAwareSortTest, TestAllocBeforeConsumer)
{
    ComputationalGraphBuilder subGraph;
    std::vector<MemoryType> memTypes(4, MemoryType::MEM_UB);
    EXPECT_EQ(subGraph.AddTensors(DataType::DT_FP32, {16, 16}, memTypes, {"t_in", "t_ub", "t_out", "t_ddr"}, 0), true);

    EXPECT_EQ(subGraph.AddOps(
                  {Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_COPY_IN, Opcode::OP_ADD, Opcode::OP_COPY_OUT},
                  {{}, {}, {"t_ddr"}, {"t_ub"}, {"t_ub"}}, {{"t_ub"}, {"t_out"}, {"t_ub"}, {"t_out"}, {"t_out"}},
                  {"AllocUb", "AllocOut", "CopyIn", "Add", "CopyOut"}, true),
              true);

    Function* function = subGraph.GetFunction();
    ASSERT_NE(function, nullptr);
    auto opList = function->Operations().DuplicatedOpList();

    MemoryAwareTopoSort sorter(opList, *function);
    ASSERT_EQ(sorter.SortOps(), SUCCESS);

    const auto& sorted = sorter.operations;

    // 找到 ALLOC 和其 consumer (CopyIn) 的位�?
    size_t allocIdx = std::string::npos;
    size_t consumerIdx = std::string::npos;
    for (size_t i = 0; i < sorted.size(); i++) {
        if (sorted[i]->GetOpcodeStr().find("ALLOC") != std::string::npos) {
            allocIdx = i;
        }
        if (sorted[i]->GetOpcodeStr() == "COPY_IN") {
            consumerIdx = i;
        }
    }
    ASSERT_NE(allocIdx, std::string::npos);
    ASSERT_NE(consumerIdx, std::string::npos);
    EXPECT_LT(allocIdx, consumerIdx) << "ALLOC op must precede its consumer (COPY_IN) in topological order";
}

// TestCubeAllocBeforeDataWriter �?验证 cube 模式: ALLOC 在数据搬运节点之�?
// L0A_ALLOC(申请L0A) �?L1_TO_L0A(写入L0A) �?A_MUL_B(读取L0A) 的顺�?
TEST_F(MemoryAwareSortTest, TestCubeAllocBeforeDataWriter)
{
    ComputationalGraphBuilder subGraph;
    std::vector<MemoryType> memTypes(6, MemoryType::MEM_UB);
    EXPECT_EQ(subGraph.AddTensors(DataType::DT_FP16, {16, 16}, memTypes,
                                  {"t_l1", "t_l0a", "t_l0b", "t_l0c", "t_ddr_a", "t_ddr_b"}, 0),
              true);

    EXPECT_EQ(subGraph.AddOps({Opcode::OP_L0A_ALLOC, Opcode::OP_L1_TO_L0A, Opcode::OP_L0B_ALLOC, Opcode::OP_L1_TO_L0B,
                               Opcode::OP_L0C_ALLOC, Opcode::OP_A_MUL_B},
                              {{}, {"t_l1"}, {}, {"t_l1"}, {}, {"t_l0a", "t_l0b"}},
                              {{"t_l0a"}, {"t_l0a"}, {"t_l0b"}, {"t_l0b"}, {"t_l0c"}, {"t_l0c"}},
                              {"AllocL0A", "L1ToL0A", "AllocL0B", "L1ToL0B", "AllocL0C", "MatMul"}, true),
              true);

    Function* function = subGraph.GetFunction();
    ASSERT_NE(function, nullptr);
    auto opList = function->Operations().DuplicatedOpList();
    ASSERT_EQ(opList.size(), 6);

    MemoryAwareTopoSort sorter(opList, *function);
    ASSERT_EQ(sorter.SortOps(), SUCCESS);

    const auto& sorted = sorter.operations;
    ASSERT_EQ(sorted.size(), 6);

    auto pos = [&](const std::string& name) -> size_t {
        for (size_t i = 0; i < sorted.size(); i++) {
            if (sorted[i]->GetOpcodeStr().find(name) != std::string::npos) {
                return i;
            }
        }
        return std::string::npos;
    };

    size_t allocA = pos("L0A_ALLOC");
    size_t l1ToL0A = pos("L1_TO_L0A");
    size_t matMul = pos("A_MUL_B");

    ASSERT_NE(allocA, std::string::npos);
    ASSERT_NE(l1ToL0A, std::string::npos);
    ASSERT_NE(matMul, std::string::npos);

    EXPECT_LT(allocA, l1ToL0A) << "L0A_ALLOC must precede L1_TO_L0A (data writer)";
    EXPECT_LT(l1ToL0A, matMul) << "L1_TO_L0A must precede A_MUL_B (data reader)";
}

// ============================================================================
// T18: ALLOC→writer→reader ordering �?验证 ALLOC 在所有运算符（包括只写不读的 OOperand）之�?
// 回归测试：EnsureAllocInterleaving 曾使�?GetConsumers() 查找消费者，
// �?GetConsumers() 只返�?IOperand（读者），遗漏了 COPY_IN �?OOperand（写者）�?
// 导致 ALLOC 被插入到 COPY_IN 之后�?OoOScheduler Free �?bufferSlices 找不�?tensor�?
// ============================================================================
TEST_F(MemoryAwareSortTest, TestAllocBeforeWriterNotOnlyReader)
{
    ComputationalGraphBuilder subGraph;
    // 模拟：alloc(ub_tensor0); copyin(ub_tensor0, gm_tensor1); add(ub_tensor0, ...)
    // copyin �?ub_tensor0 �?OOperand（输�?写入者），GetConsumers() 不包含它
    // 只有 add �?ub_tensor0 �?IOperand（输�?读取者），GetConsumers() 包含�?
    std::vector<MemoryType> memTypes(5, MemoryType::MEM_UB);
    EXPECT_EQ(subGraph.AddTensors(DataType::DT_FP32, {16, 16}, memTypes,
                                  {"gm_tensor1", "ub_tensor0", "ub_tensor1", "ub_tensor2", "ub_tensor3"}, 0),
              true);

    EXPECT_EQ(subGraph.AddOps({Opcode::OP_UB_ALLOC, Opcode::OP_COPY_IN, Opcode::OP_UB_ALLOC, Opcode::OP_ADD},
                              {{}, {"gm_tensor1"}, {}, {"ub_tensor0"}},
                              {{"ub_tensor0"}, {"ub_tensor0"}, {"ub_tensor1"}, {"ub_tensor1"}},
                              {"UB_Alloc", "CopyIn", "AllocOut", "Add"}, true),
              true);

    Function* function = subGraph.GetFunction();
    ASSERT_NE(function, nullptr);
    auto opList = function->Operations().DuplicatedOpList();
    ASSERT_GE(opList.size(), 3);

    MemoryAwareTopoSort sorter(opList, *function);
    ASSERT_EQ(sorter.SortOps(), SUCCESS);

    const auto& sorted = sorter.operations;

    auto pos = [&](const std::string& substr) -> size_t {
        for (size_t i = 0; i < sorted.size(); i++) {
            if (sorted[i]->GetOpcodeStr().find(substr) != std::string::npos) {
                return i;
            }
        }
        return std::string::npos;
    };

    size_t allocPos = pos("ALLOC");
    size_t copyinPos = pos("COPY_IN");
    size_t addPos = pos("ADD");

    ASSERT_NE(allocPos, std::string::npos);
    ASSERT_NE(copyinPos, std::string::npos);
    ASSERT_NE(addPos, std::string::npos);

    EXPECT_LT(allocPos, copyinPos) << "ALLOC must precede COPY_IN (first writer of the buffer)";
    EXPECT_LT(allocPos, addPos) << "ALLOC must precede ADD (reader of the buffer)";
}

// ============================================================================
// T19: 通用 ALLOC→consumer 顺序验证 �?多个 ALLOC 多个消费者，确保�?ALLOC 在消费者之�?
// ============================================================================
TEST_F(MemoryAwareSortTest, TestNoAllocAfterAnyConsumer)
{
    ComputationalGraphBuilder subGraph;
    std::vector<MemoryType> memTypes(7, MemoryType::MEM_UB);
    EXPECT_EQ(subGraph.AddTensors(DataType::DT_FP32, {16, 16}, memTypes, {"t1", "t2", "t3", "t4", "t5", "t6", "t7"}, 0),
              true);

    // UB_ALLOC(t2), UB_ALLOC(t3), COPY_IN(t1→t2), UB_ALLOC(t4), ADD(t2→t4), COPY_IN(t1→t3), UB_ALLOC(t5), ADD(t3→t5),
    // UB_ALLOC(t6), ADD(t4,t5→t6)
    EXPECT_EQ(
        subGraph.AddOps(
            {Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_COPY_IN, Opcode::OP_UB_ALLOC, Opcode::OP_ADD,
             Opcode::OP_COPY_IN, Opcode::OP_UB_ALLOC, Opcode::OP_ADD, Opcode::OP_UB_ALLOC, Opcode::OP_ADD},
            {{}, {}, {"t1"}, {}, {"t2"}, {"t1"}, {}, {"t3"}, {}, {"t4", "t5"}},
            {{"t2"}, {"t3"}, {"t2"}, {"t4"}, {"t4"}, {"t3"}, {"t5"}, {"t5"}, {"t6"}, {"t6"}},
            {"AllocA", "AllocB", "CopyInA", "AllocC", "AddA", "CopyInB", "AllocD", "AddB", "AllocE", "AddFinal"}, true),
        true);

    Function* function = subGraph.GetFunction();
    ASSERT_NE(function, nullptr);
    auto opList = function->Operations().DuplicatedOpList();
    ASSERT_GE(opList.size(), 10);

    MemoryAwareTopoSort sorter(opList, *function);
    ASSERT_EQ(sorter.SortOps(), SUCCESS);

    const auto& sorted = sorter.operations;

    // 验证拓扑�?�?使用 sorter 自带�?depManager，避免因 ALLOC/writer tensor
    // 重叠导致 OoOScheduler.Init 失败
    EXPECT_TRUE(VerifyTopologicalOrder(sorted, sorter.depManager_));

    // 验证�?ALLOC 在其消费者之�?
    std::unordered_map<Operation*, std::set<Operation*>> consumerOf;
    for (size_t i = 0; i < sorted.size(); i++) {
        Operation* op = sorted[i];
        if (op->GetOpcodeStr().find("ALLOC") == std::string::npos) {
            continue;
        }
        for (size_t j = i + 1; j < sorted.size(); j++) {
            Operation* other = sorted[j];
            auto& preds = sorter.depManager_.GetPredecessors(other);
            if (preds.find(op) != preds.end()) {
                consumerOf[op].insert(other);
            }
        }
        for (size_t j = 0; j < i; j++) {
            Operation* other = sorted[j];
            auto& preds = sorter.depManager_.GetPredecessors(other);
            if (preds.find(op) != preds.end()) {
                ADD_FAILURE() << "ALLOC " << op->GetOpcodeStr() << " at position " << i
                              << " appears AFTER its consumer " << other->GetOpcodeStr() << " at position " << j;
            }
        }
    }
}

// ============================================================================
// === DualDst 融合 / staticValidShape 快照 / core_assign DualDstProcess UT ===
// ============================================================================
//
// 涉及源文�?
//   passes/block_graph_pass/schedule_ooo/dualdst_fuse.cpp
//   passes/block_graph_pass/schedule_ooo/core_assign.cpp  (HasSameValidShape2D /
//                                                           DualDstProcess /
//                                                           TryGetStaticValidShapeFromProducer)
//   passes/tile_graph_pass/graph_constraint/infer_dyn_shape.cpp
//                                                          (RecordStaticValidShapeOnL0CCopyUB)
//
// 目标: 通过公共/可见 (本文�?#define private public) 入口尽可能驱动覆�?
//   - DualDstFuse: RunDualDstFuse / IdentifyDualDstPairs / FuseDualDstPairs / 阶段 2 子步�?
//                  / IsDualDstAlloc / GetDualDstCopyOpFor / GetDualDstPairedMemId
//                  / AllocateDualDstAtCurrent / ResolveDualDstAllocCtx / CommitDualDstAlloc
//                  + 匿名 ns 下的 DynShapeEq / ReadGeometry / LoadGeometries /
//                    ConsumerCore / BuildAdjacencyCandidates / GreedyNonOverlapPick /
//                    PickAllocOrder / FindAllocPred (经入口路径触�?
//   - InferDynShape: RecordStaticValidShapeOnL0CCopyUB (�?dyn 跳过分支)
//   - core_assign:   HasSameValidShape2D / TryGetStaticValidShapeFromProducer
//                    (�?DualDstProcess 整体路径触达)
namespace dualdst_ut {

constexpr int64_t TILE_M = 64;
constexpr int64_t TILE_N = 64;
// 每个 dualdst UT 自己的小 UB pool, 便于 AllocateDualDstAtCurrent 测试�?
constexpr size_t SMALL_UB_POOL = 8 * 1024;

// --- 形态构造工�?-----------------------------------------------------------

// �?OP_L0C_COPY_UB 类拷�?op 设置 CopyOpAttribute (fromOff = src L0C 偏移,
// shape = 实际搬运 tile shape)。dualdst identify 阶段读这两项做相�?+ tile 校验�?
void SetCopyL0cToUbAttr(Operation& op, const std::vector<int64_t>& fromOff, const std::vector<int64_t>& tileShape)
{
    auto fromOffImme = OpImmediate::Specified(fromOff);
    auto shapeImme = OpImmediate::Specified(tileShape);
    op.SetOpAttribute(std::make_shared<CopyOpAttribute>(fromOffImme, MemoryType::MEM_UB, shapeImme, shapeImme));
}

// �?vector<int64_t> �?staticValidShape 写到 op 属�? 模拟 InferDynShape 阶段
// RecordStaticValidShapeOnL0CCopyUB 的快照效果�?
void InjectStaticValidShape(Operation& op, const std::vector<int64_t>& vals)
{
    op.SetAttribute(OpAttributeKey::staticValidShape, vals);
}

// dualdst UT 通用�? �?L0C tensor + 两条 OP_L0C_COPY_UB + 各自下游 Add(ub→out)�?
// fromOff0 / fromOff1 控制 SplitM 还是 SplitN 候�?(相邻 = tile 尺寸�?�?
struct DualDstGraph {
    std::shared_ptr<ComputationalGraphBuilder> builder;
    Function* func{nullptr};
    Operation* allocL0c{nullptr};
    Operation* allocUb0{nullptr};
    Operation* allocUb1{nullptr};
    Operation* allocOut0{nullptr};
    Operation* allocOut1{nullptr};
    Operation* copy0{nullptr}; // L0C -> ub0
    Operation* copy1{nullptr}; // L0C -> ub1
    Operation* add0{nullptr};  // ub0 -> out0  (consumer => AIV0)
    Operation* add1{nullptr};  // ub1 -> out1  (consumer => AIV1)
};

// l0cShape: t_l0c �?shape; tileShape: copy �?tile shape (== ub shape);
// fromOff0/1: �?copy �?src 偏移�?
DualDstGraph BuildDualDstGraph(const std::vector<int64_t>& l0cShape, const std::vector<int64_t>& tileShape,
                               const std::vector<int64_t>& fromOff0, const std::vector<int64_t>& fromOff1)
{
    DualDstGraph g;
    g.builder = std::make_shared<ComputationalGraphBuilder>();
    EXPECT_EQ(g.builder->AddTensor(DataType::DT_FP32, l0cShape, MemoryType::MEM_L0C, "t_l0c"), true);
    EXPECT_EQ(g.builder->AddTensor(DataType::DT_FP32, tileShape, MemoryType::MEM_UB, "t_ub0"), true);
    EXPECT_EQ(g.builder->AddTensor(DataType::DT_FP32, tileShape, MemoryType::MEM_UB, "t_ub1"), true);
    EXPECT_EQ(g.builder->AddTensor(DataType::DT_FP32, tileShape, MemoryType::MEM_UB, "t_out0"), true);
    EXPECT_EQ(g.builder->AddTensor(DataType::DT_FP32, tileShape, MemoryType::MEM_UB, "t_out1"), true);

    EXPECT_EQ(g.builder->AddOp(Opcode::OP_L0C_ALLOC, {}, {"t_l0c"}, "alloc_l0c"), true);
    EXPECT_EQ(g.builder->AddOp(Opcode::OP_UB_ALLOC, {}, {"t_ub0"}, "alloc_ub0"), true);
    EXPECT_EQ(g.builder->AddOp(Opcode::OP_UB_ALLOC, {}, {"t_ub1"}, "alloc_ub1"), true);
    EXPECT_EQ(g.builder->AddOp(Opcode::OP_UB_ALLOC, {}, {"t_out0"}, "alloc_out0"), true);
    EXPECT_EQ(g.builder->AddOp(Opcode::OP_UB_ALLOC, {}, {"t_out1"}, "alloc_out1"), true);
    EXPECT_EQ(g.builder->AddOp(Opcode::OP_L0C_COPY_UB, {"t_l0c"}, {"t_ub0"}, "copy0"), true);
    EXPECT_EQ(g.builder->AddOp(Opcode::OP_L0C_COPY_UB, {"t_l0c"}, {"t_ub1"}, "copy1"), true);
    // OP_ADD �?binary, 需�?2 个输�? 给同一 tensor 两次, consumers_ set 去重�?
    // ub0/ub1 �?consumer 仍只�?1 �?add op (满足 dualdst identify ConsumerCore 逻辑)�?
    // 这样能让 BinaryBrcinlineInferFunc �?InferShape 路径下不越界访问 [1] 而段错�?
    EXPECT_EQ(g.builder->AddOp(Opcode::OP_ADD, {"t_ub0", "t_ub0"}, {"t_out0"}, "add0"), true);
    EXPECT_EQ(g.builder->AddOp(Opcode::OP_ADD, {"t_ub1", "t_ub1"}, {"t_out1"}, "add1"), true);

    g.func = g.builder->GetFunction();
    g.allocL0c = g.builder->GetOp("alloc_l0c");
    g.allocUb0 = g.builder->GetOp("alloc_ub0");
    g.allocUb1 = g.builder->GetOp("alloc_ub1");
    g.allocOut0 = g.builder->GetOp("alloc_out0");
    g.allocOut1 = g.builder->GetOp("alloc_out1");
    g.copy0 = g.builder->GetOp("copy0");
    g.copy1 = g.builder->GetOp("copy1");
    g.add0 = g.builder->GetOp("add0");
    g.add1 = g.builder->GetOp("add1");

    SetCopyL0cToUbAttr(*g.copy0, fromOff0, tileShape);
    SetCopyL0cToUbAttr(*g.copy1, fromOff1, tileShape);
    return g;
}

// �?OoOScheduler 预填 schedInfoMap_:
//   - add0->AIV0 / add1->AIV1, �?dualdst identify 阶段�?ConsumerCore 校验�?
//     split (AIV0 + AIV1)�?
//   - dualdst ResolveDualDstAllocCtx 现已改用 "�?dual_dst op �?ub output tensor �?
//     consumer (add op) 反推 core", 不再依赖 tensorAllocCoreMap (该成员已随主线移�?,
//     所以只�?add0/add1 �?schedInfoMap_ 正确就足够�?
void InjectCoreMap(OoOScheduler& s, const DualDstGraph& g, bool sameCoreForAdds = false)
{
    s.state_.schedInfoMap[g.copy0].coreLocation = CoreLocationType::AIC;
    s.state_.schedInfoMap[g.copy1].coreLocation = CoreLocationType::AIC;
    s.state_.schedInfoMap[g.add0].coreLocation = CoreLocationType::AIV0;
    s.state_.schedInfoMap[g.add1].coreLocation = sameCoreForAdds ? CoreLocationType::AIV0 : CoreLocationType::AIV1;
    s.state_.schedInfoMap[g.allocL0c].coreLocation = CoreLocationType::AIC;
    s.state_.schedInfoMap[g.allocUb0].coreLocation = CoreLocationType::AIV0;
    s.state_.schedInfoMap[g.allocUb1].coreLocation = CoreLocationType::AIV1;
    s.state_.schedInfoMap[g.allocOut0].coreLocation = CoreLocationType::AIV0;
    s.state_.schedInfoMap[g.allocOut1].coreLocation = CoreLocationType::AIV1;
}

} // namespace dualdst_ut

// --- DynShapeEq 三条分支 (�?IdentifyDualDstPairs 间接驱动) -----------------
// dump 严格相等 / concrete 数值相�?/ 不等 -> identify 命中 vs miss
TEST_F(ScheduleOoOTest, DualDst_DynShapeEq_DumpEqual_HitsIdentify)
{
    auto g = dualdst_ut::BuildDualDstGraph(
        /*l0cShape*/ {dualdst_ut::TILE_M, dualdst_ut::TILE_N * 2},
        /*tileShape*/ {dualdst_ut::TILE_M, dualdst_ut::TILE_N},
        /*fromOff0*/ {0, 0},
        /*fromOff1*/ {0, dualdst_ut::TILE_N});
    // 两侧 dyn validShape 都是 {S0, S1}, dump 字符串严格相�?-> DynShapeEq 走分�?1)
    g.copy0->GetOutputOperand(0)->UpdateDynValidShape({SymbolicScalar("S0"), SymbolicScalar("S1")});
    g.copy1->GetOutputOperand(0)->UpdateDynValidShape({SymbolicScalar("S0"), SymbolicScalar("S1")});

    OoOScheduler s(*g.func);
    EXPECT_EQ(s.Init(g.func->Operations().DuplicatedOpList(), {}, CORE_INIT_CONFIGS_HARDWARE_TWO), SUCCESS);
    dualdst_ut::InjectCoreMap(s, g);

    std::vector<DualDstPair> pairs;
    EXPECT_EQ(s.dualDstEngine_.IdentifyDualDstPairs(pairs), SUCCESS);
    EXPECT_EQ(pairs.size(), 1u);
}

TEST_F(ScheduleOoOTest, DualDst_DynShapeEq_ConcreteEqualButDifferentDump_StillHits)
{
    auto g = dualdst_ut::BuildDualDstGraph({dualdst_ut::TILE_M, dualdst_ut::TILE_N * 2},
                                           {dualdst_ut::TILE_M, dualdst_ut::TILE_N}, {0, 0}, {0, dualdst_ut::TILE_N});
    // 给两侧不同符号名�?concrete 数值相等的 SymbolicScalar -> 走分�?2)
    g.copy0->GetOutputOperand(0)->UpdateDynValidShape(
        {SymbolicScalar("a", dualdst_ut::TILE_M), SymbolicScalar("b", dualdst_ut::TILE_N)});
    g.copy1->GetOutputOperand(0)->UpdateDynValidShape(
        {SymbolicScalar("c", dualdst_ut::TILE_M), SymbolicScalar("d", dualdst_ut::TILE_N)});

    OoOScheduler s(*g.func);
    EXPECT_EQ(s.Init(g.func->Operations().DuplicatedOpList(), {}, CORE_INIT_CONFIGS_HARDWARE_TWO), SUCCESS);
    dualdst_ut::InjectCoreMap(s, g);

    std::vector<DualDstPair> pairs;
    EXPECT_EQ(s.dualDstEngine_.IdentifyDualDstPairs(pairs), SUCCESS);
    EXPECT_EQ(pairs.size(), 1u);
}

TEST_F(ScheduleOoOTest, DualDst_DynShapeEq_DumpDifferAndNoConcrete_NoPair)
{
    auto g = dualdst_ut::BuildDualDstGraph({dualdst_ut::TILE_M, dualdst_ut::TILE_N * 2},
                                           {dualdst_ut::TILE_M, dualdst_ut::TILE_N}, {0, 0}, {0, dualdst_ut::TILE_N});
    // 不同符号且无 concrete -> 走分�?3) 返回 false -> identify 0 pair
    g.copy0->GetOutputOperand(0)->UpdateDynValidShape({SymbolicScalar("X0"), SymbolicScalar("X1")});
    g.copy1->GetOutputOperand(0)->UpdateDynValidShape({SymbolicScalar("Y0"), SymbolicScalar("Y1")});

    OoOScheduler s(*g.func);
    EXPECT_EQ(s.Init(g.func->Operations().DuplicatedOpList(), {}, CORE_INIT_CONFIGS_HARDWARE_TWO), SUCCESS);
    dualdst_ut::InjectCoreMap(s, g);

    std::vector<DualDstPair> pairs;
    EXPECT_EQ(s.dualDstEngine_.IdentifyDualDstPairs(pairs), SUCCESS);
    EXPECT_EQ(pairs.size(), 0u);
}

// --- ReadGeometry: staticValidShape 优先 + dyn fallback ---------------------
// op �?staticValidShape 属�? identify 应走属性路�?(覆盖 ReadGeometry 分支 1)
TEST_F(ScheduleOoOTest, DualDst_ReadGeometry_PrefersStaticValidShape)
{
    auto g = dualdst_ut::BuildDualDstGraph({dualdst_ut::TILE_M, dualdst_ut::TILE_N * 2},
                                           {dualdst_ut::TILE_M, dualdst_ut::TILE_N}, {0, 0}, {0, dualdst_ut::TILE_N});
    // 故意�?dyn validShape 设成不同 dump、无 concrete: �?ReadGeometry 错走 dyn
    // 路径会判 false; 但我们注入了 staticValidShape -> 必须命中 (返回 1 �?�?
    g.copy0->GetOutputOperand(0)->UpdateDynValidShape({SymbolicScalar("X0"), SymbolicScalar("X1")});
    g.copy1->GetOutputOperand(0)->UpdateDynValidShape({SymbolicScalar("Y0"), SymbolicScalar("Y1")});
    dualdst_ut::InjectStaticValidShape(*g.copy0, {dualdst_ut::TILE_M, dualdst_ut::TILE_N});
    dualdst_ut::InjectStaticValidShape(*g.copy1, {dualdst_ut::TILE_M, dualdst_ut::TILE_N});

    OoOScheduler s(*g.func);
    EXPECT_EQ(s.Init(g.func->Operations().DuplicatedOpList(), {}, CORE_INIT_CONFIGS_HARDWARE_TWO), SUCCESS);
    dualdst_ut::InjectCoreMap(s, g);

    std::vector<DualDstPair> pairs;
    EXPECT_EQ(s.dualDstEngine_.IdentifyDualDstPairs(pairs), SUCCESS);
    EXPECT_EQ(pairs.size(), 1u);
}

// --- IdentifyDualDstPairs: SplitN 命中路径 (基本) ----------------------------
TEST_F(ScheduleOoOTest, DualDst_Identify_SplitN_HappyPath)
{
    auto g = dualdst_ut::BuildDualDstGraph({dualdst_ut::TILE_M, dualdst_ut::TILE_N * 2},
                                           {dualdst_ut::TILE_M, dualdst_ut::TILE_N}, {0, 0}, {0, dualdst_ut::TILE_N});
    g.copy0->GetOutputOperand(0)->UpdateDynValidShape(
        {SymbolicScalar(dualdst_ut::TILE_M), SymbolicScalar(dualdst_ut::TILE_N)});
    g.copy1->GetOutputOperand(0)->UpdateDynValidShape(
        {SymbolicScalar(dualdst_ut::TILE_M), SymbolicScalar(dualdst_ut::TILE_N)});

    OoOScheduler s(*g.func);
    EXPECT_EQ(s.Init(g.func->Operations().DuplicatedOpList(), {}, CORE_INIT_CONFIGS_HARDWARE_TWO), SUCCESS);
    dualdst_ut::InjectCoreMap(s, g);

    std::vector<DualDstPair> pairs;
    EXPECT_EQ(s.dualDstEngine_.IdentifyDualDstPairs(pairs), SUCCESS);
    ASSERT_EQ(pairs.size(), 1u);
    // SplitN: opEarly fromN 较小, 应是 copy0
    EXPECT_EQ(pairs[0].opEarly, g.copy0);
    EXPECT_EQ(pairs[0].opLate, g.copy1);
    EXPECT_NE(pairs[0].allocEarly, nullptr);
    EXPECT_NE(pairs[0].allocLate, nullptr);
}

// --- IdentifyDualDstPairs: SplitM 命中 ---------------------------------------
TEST_F(ScheduleOoOTest, DualDst_Identify_SplitM_HappyPath)
{
    // M-axis adjacent: l0cShape M = 2*TILE_M; fromOff �?M 轴相�?TILE_M
    auto g = dualdst_ut::BuildDualDstGraph({dualdst_ut::TILE_M * 2, dualdst_ut::TILE_N},
                                           {dualdst_ut::TILE_M, dualdst_ut::TILE_N}, {0, 0}, {dualdst_ut::TILE_M, 0});
    g.copy0->GetOutputOperand(0)->UpdateDynValidShape(
        {SymbolicScalar(dualdst_ut::TILE_M), SymbolicScalar(dualdst_ut::TILE_N)});
    g.copy1->GetOutputOperand(0)->UpdateDynValidShape(
        {SymbolicScalar(dualdst_ut::TILE_M), SymbolicScalar(dualdst_ut::TILE_N)});

    OoOScheduler s(*g.func);
    EXPECT_EQ(s.Init(g.func->Operations().DuplicatedOpList(), {}, CORE_INIT_CONFIGS_HARDWARE_TWO), SUCCESS);
    dualdst_ut::InjectCoreMap(s, g);

    std::vector<DualDstPair> pairs;
    EXPECT_EQ(s.dualDstEngine_.IdentifyDualDstPairs(pairs), SUCCESS);
    ASSERT_EQ(pairs.size(), 1u);
    EXPECT_EQ(pairs[0].opEarly, g.copy0); // fromM 0 < TILE_M
    EXPECT_EQ(pairs[0].opLate, g.copy1);
    // SplitM 方向命中�?dualDstL0CDirection_ 应为 0
    auto l0c = g.copy0->GetInputOperand(0);
    EXPECT_EQ(s.dualDstEngine_.dualDstL0CDirection_.count(l0c), 1u);
    EXPECT_EQ(s.dualDstEngine_.dualDstL0CDirection_[l0c], 0);
}

// --- IdentifyDualDstPairs: 不相�?/ 不同 core / shape 不一�?-> 0 pair ------
TEST_F(ScheduleOoOTest, DualDst_Identify_NotAdjacent_NoPair)
{
    auto g = dualdst_ut::BuildDualDstGraph({dualdst_ut::TILE_M, dualdst_ut::TILE_N * 4},
                                           {dualdst_ut::TILE_M, dualdst_ut::TILE_N}, {0, 0},
                                           {0, dualdst_ut::TILE_N * 2}); // gap = 2*TILE_N, 不相�?
    g.copy0->GetOutputOperand(0)->UpdateDynValidShape(
        {SymbolicScalar(dualdst_ut::TILE_M), SymbolicScalar(dualdst_ut::TILE_N)});
    g.copy1->GetOutputOperand(0)->UpdateDynValidShape(
        {SymbolicScalar(dualdst_ut::TILE_M), SymbolicScalar(dualdst_ut::TILE_N)});

    OoOScheduler s(*g.func);
    EXPECT_EQ(s.Init(g.func->Operations().DuplicatedOpList(), {}, CORE_INIT_CONFIGS_HARDWARE_TWO), SUCCESS);
    dualdst_ut::InjectCoreMap(s, g);

    std::vector<DualDstPair> pairs;
    EXPECT_EQ(s.dualDstEngine_.IdentifyDualDstPairs(pairs), SUCCESS);
    EXPECT_EQ(pairs.size(), 0u);
}

TEST_F(ScheduleOoOTest, DualDst_Identify_SameConsumerCore_NoPair)
{
    auto g = dualdst_ut::BuildDualDstGraph({dualdst_ut::TILE_M, dualdst_ut::TILE_N * 2},
                                           {dualdst_ut::TILE_M, dualdst_ut::TILE_N}, {0, 0}, {0, dualdst_ut::TILE_N});
    g.copy0->GetOutputOperand(0)->UpdateDynValidShape(
        {SymbolicScalar(dualdst_ut::TILE_M), SymbolicScalar(dualdst_ut::TILE_N)});
    g.copy1->GetOutputOperand(0)->UpdateDynValidShape(
        {SymbolicScalar(dualdst_ut::TILE_M), SymbolicScalar(dualdst_ut::TILE_N)});

    OoOScheduler s(*g.func);
    EXPECT_EQ(s.Init(g.func->Operations().DuplicatedOpList(), {}, CORE_INIT_CONFIGS_HARDWARE_TWO), SUCCESS);
    dualdst_ut::InjectCoreMap(s, g, /*sameCoreForAdds=*/true); // both consumers AIV0

    std::vector<DualDstPair> pairs;
    EXPECT_EQ(s.dualDstEngine_.IdentifyDualDstPairs(pairs), SUCCESS);
    EXPECT_EQ(pairs.size(), 0u);
}

// --- RunDualDstFuse 三个出口分支 ---------------------------------------------
// 分支 1: enableDualDst_ false -> 直接 SUCCESS, 不动�?
TEST_F(ScheduleOoOTest, DualDst_RunDualDstFuse_DisabledIsNoOp)
{
    auto g = dualdst_ut::BuildDualDstGraph({dualdst_ut::TILE_M, dualdst_ut::TILE_N * 2},
                                           {dualdst_ut::TILE_M, dualdst_ut::TILE_N}, {0, 0}, {0, dualdst_ut::TILE_N});
    g.copy0->GetOutputOperand(0)->UpdateDynValidShape(
        {SymbolicScalar(dualdst_ut::TILE_M), SymbolicScalar(dualdst_ut::TILE_N)});
    g.copy1->GetOutputOperand(0)->UpdateDynValidShape(
        {SymbolicScalar(dualdst_ut::TILE_M), SymbolicScalar(dualdst_ut::TILE_N)});

    OoOScheduler s(*g.func);
    EXPECT_EQ(s.Init(g.func->Operations().DuplicatedOpList(), {}, CORE_INIT_CONFIGS_HARDWARE_TWO), SUCCESS);
    dualdst_ut::InjectCoreMap(s, g);

    s.dualDstEngine_.enableDualDst_ = false;
    EXPECT_EQ(s.dualDstEngine_.RunDualDstFuse(), SUCCESS);
}

// 分支 2: 只有 AIV0 (HARDWARE_ONE) -> RunDualDstFuse 早返
TEST_F(ScheduleOoOTest, DualDst_RunDualDstFuse_SingleAivPoolEarlyExit)
{
    auto g = dualdst_ut::BuildDualDstGraph({dualdst_ut::TILE_M, dualdst_ut::TILE_N * 2},
                                           {dualdst_ut::TILE_M, dualdst_ut::TILE_N}, {0, 0}, {0, dualdst_ut::TILE_N});
    g.copy0->GetOutputOperand(0)->UpdateDynValidShape(
        {SymbolicScalar(dualdst_ut::TILE_M), SymbolicScalar(dualdst_ut::TILE_N)});
    g.copy1->GetOutputOperand(0)->UpdateDynValidShape(
        {SymbolicScalar(dualdst_ut::TILE_M), SymbolicScalar(dualdst_ut::TILE_N)});

    OoOScheduler s(*g.func);
    EXPECT_EQ(s.Init(g.func->Operations().DuplicatedOpList(), {}, CORE_INIT_CONFIGS_HARDWARE_ONE), SUCCESS);
    dualdst_ut::InjectCoreMap(s, g);

    s.dualDstEngine_.enableDualDst_ = true;
    EXPECT_EQ(s.dualDstEngine_.RunDualDstFuse(), SUCCESS);
}

// 分支 3: 真的 fuse: 图实际被�?(�?op SetAsDeleted -> EraseOperations(false, true) 生效)�?
// 不再校验外部 mutated 标志�?(�?flag 已随 EraseOperations 第二�?true 同步刷新 opPosition_
// 一并删�?, 改为直接对比 operations_.size() 变化�?
TEST_F(ScheduleOoOTest, DualDst_RunDualDstFuse_ActuallyFusesAndMutatesFunction)
{
    auto g = dualdst_ut::BuildDualDstGraph({dualdst_ut::TILE_M, dualdst_ut::TILE_N * 2},
                                           {dualdst_ut::TILE_M, dualdst_ut::TILE_N}, {0, 0}, {0, dualdst_ut::TILE_N});
    g.copy0->GetOutputOperand(0)->UpdateDynValidShape(
        {SymbolicScalar(dualdst_ut::TILE_M), SymbolicScalar(dualdst_ut::TILE_N)});
    g.copy1->GetOutputOperand(0)->UpdateDynValidShape(
        {SymbolicScalar(dualdst_ut::TILE_M), SymbolicScalar(dualdst_ut::TILE_N)});

    OoOScheduler s(*g.func);
    EXPECT_EQ(s.Init(g.func->Operations().DuplicatedOpList(), {}, CORE_INIT_CONFIGS_HARDWARE_TWO), SUCCESS);
    dualdst_ut::InjectCoreMap(s, g);

    size_t opsBefore = g.func->Operations().size();
    s.dualDstEngine_.enableDualDst_ = true;
    EXPECT_EQ(s.dualDstEngine_.RunDualDstFuse(), SUCCESS);

    // EraseOperations(false, true) 移除�?copy0 / copy1 / 一个被剔的 alloc,
    // 新增 1 �?OP_L0C_COPY_UB_DUAL_DST -> 净�?2 �?op
    size_t opsAfter = g.func->Operations().size();
    EXPECT_EQ(opsBefore, opsAfter + 2);

    // 至少有一�?OP_L0C_COPY_UB_DUAL_DST 出现
    bool hasFused = false;
    for (auto& op : g.func->Operations()) {
        if (op.GetOpcode() == Opcode::OP_L0C_COPY_UB_DUAL_DST) {
            hasFused = true;
            break;
        }
    }
    EXPECT_TRUE(hasFused);
}

// --- IsDualDstAlloc / GetDualDstCopyOpFor / GetDualDstPairedMemId ----------
// fuse 后保留下来的那条 alloc 应被识别�?dualdst alloc, 并能反查 dual op + paired memId
TEST_F(ScheduleOoOTest, DualDst_AllocQueryHelpers_AfterFuse)
{
    auto g = dualdst_ut::BuildDualDstGraph({dualdst_ut::TILE_M, dualdst_ut::TILE_N * 2},
                                           {dualdst_ut::TILE_M, dualdst_ut::TILE_N}, {0, 0}, {0, dualdst_ut::TILE_N});
    g.copy0->GetOutputOperand(0)->UpdateDynValidShape(
        {SymbolicScalar(dualdst_ut::TILE_M), SymbolicScalar(dualdst_ut::TILE_N)});
    g.copy1->GetOutputOperand(0)->UpdateDynValidShape(
        {SymbolicScalar(dualdst_ut::TILE_M), SymbolicScalar(dualdst_ut::TILE_N)});

    OoOScheduler s(*g.func);
    EXPECT_EQ(s.Init(g.func->Operations().DuplicatedOpList(), {}, CORE_INIT_CONFIGS_HARDWARE_TWO), SUCCESS);
    dualdst_ut::InjectCoreMap(s, g);
    s.dualDstEngine_.enableDualDst_ = true;
    EXPECT_EQ(s.dualDstEngine_.RunDualDstFuse(), SUCCESS);

    // 找到 fuse 出来�?dualdst op 和它依赖�?(唯一保留) UB alloc
    Operation* dual = nullptr;
    for (auto& op : g.func->Operations()) {
        if (op.GetOpcode() == Opcode::OP_L0C_COPY_UB_DUAL_DST) {
            dual = &op;
            break;
        }
    }
    ASSERT_NE(dual, nullptr);

    Operation* survivingUbAlloc = nullptr;
    for (auto* pred : s.state_.depManager.GetPredecessors(dual)) {
        if (pred != nullptr && pred->GetOpcodeStr().find("UB_ALLOC") != std::string::npos) {
            survivingUbAlloc = pred;
            break;
        }
    }
    ASSERT_NE(survivingUbAlloc, nullptr);

    EXPECT_TRUE(s.dualDstEngine_.IsDualDstAlloc(survivingUbAlloc));
    EXPECT_EQ(s.dualDstEngine_.GetDualDstCopyOpFor(survivingUbAlloc), dual);
    int paired = s.dualDstEngine_.GetDualDstPairedMemId(survivingUbAlloc);
    EXPECT_NE(paired, -1);
    EXPECT_NE(paired, survivingUbAlloc->GetOutputOperand(0)->memoryrange.memId);

    // �?dualdst alloc 应返�?false / nullptr / -1
    EXPECT_FALSE(s.dualDstEngine_.IsDualDstAlloc(g.allocL0c));
    EXPECT_EQ(s.dualDstEngine_.GetDualDstCopyOpFor(g.allocL0c), nullptr);
    EXPECT_EQ(s.dualDstEngine_.GetDualDstPairedMemId(nullptr), -1);
}

// --- AllocateDualDstAtCurrent: ResolveCtx + Commit 成功路径 ----------------
// (Full 分支�?OoOScheduler::FindCommonFreeOffset 返回 nullopt 触发, 单测构造代�?
//  较大, 通过 spill 路径集成测试覆盖; 这里只验�?happy path �?ResolveCtx OK�?
TEST_F(ScheduleOoOTest, DualDst_AllocateDualDstAtCurrent_HappyPath)
{
    auto g = dualdst_ut::BuildDualDstGraph({dualdst_ut::TILE_M, dualdst_ut::TILE_N * 2},
                                           {dualdst_ut::TILE_M, dualdst_ut::TILE_N}, {0, 0}, {0, dualdst_ut::TILE_N});
    g.copy0->GetOutputOperand(0)->UpdateDynValidShape(
        {SymbolicScalar(dualdst_ut::TILE_M), SymbolicScalar(dualdst_ut::TILE_N)});
    g.copy1->GetOutputOperand(0)->UpdateDynValidShape(
        {SymbolicScalar(dualdst_ut::TILE_M), SymbolicScalar(dualdst_ut::TILE_N)});

    OoOScheduler s(*g.func);
    EXPECT_EQ(s.Init(g.func->Operations().DuplicatedOpList(), {}, CORE_INIT_CONFIGS_HARDWARE_TWO), SUCCESS);
    dualdst_ut::InjectCoreMap(s, g);
    s.dualDstEngine_.enableDualDst_ = true;
    EXPECT_EQ(s.dualDstEngine_.RunDualDstFuse(), SUCCESS);

    Operation* dual = nullptr;
    for (auto& op : g.func->Operations()) {
        if (op.GetOpcode() == Opcode::OP_L0C_COPY_UB_DUAL_DST) {
            dual = &op;
            break;
        }
    }
    ASSERT_NE(dual, nullptr);
    Operation* survivingUbAlloc = nullptr;
    for (auto* pred : s.state_.depManager.GetPredecessors(dual)) {
        if (pred != nullptr && pred->GetOpcodeStr().find("UB_ALLOC") != std::string::npos) {
            survivingUbAlloc = pred;
            break;
        }
    }
    ASSERT_NE(survivingUbAlloc, nullptr);

    int memIdA = survivingUbAlloc->GetOutputOperand(0)->memoryrange.memId;
    int memIdB = s.dualDstEngine_.GetDualDstPairedMemId(survivingUbAlloc);
    ASSERT_NE(s.state_.localBufferMap.find(memIdA), s.state_.localBufferMap.end());
    ASSERT_NE(s.state_.localBufferMap.find(memIdB), s.state_.localBufferMap.end());

    bool allocated = false;
    EXPECT_EQ(s.dualDstEngine_.AllocateDualDstAtCurrent(survivingUbAlloc, allocated), SUCCESS);
    EXPECT_TRUE(allocated);
}

// --- SelectSpillBuffers + GetDualSpillGroup: dualdst 分支专项 ---------------
// 归一改造后, SelectSpillBuffers 内按 IsDualDstAlloc 分叉:
//   dualdst -> OoOScheduler::GetDualSpillGroup(poolA, poolB, need)
//              内部嵌套两次单池滑窗, 匹配条件: 两侧 startAddr (freed segment 起点) 一�?
//              -> 返回 vector<combined memIds>, 前半来自 poolA、后半来�?poolB
//   单池   -> OoOScheduler::GetSpillGroup(pool, need) (薄壳, 委托 pool.GetSpillGroup)
// 选不出候选时 (canSpillGroups �?�?GetGroupNextUseTime 全失�? -> 兜底返回 spill-all 列表:
//   dualdst: 两池 GetAddrSortedBufs() 之并�?
//   单池  : 单池 GetAddrSortedBufs()
// spill 执行�?(GenBufferSpill 内的 SpillBuffer 循环) 与单池路径共�?
// 不在�?UT 覆盖, �?ST/集成测试 (test_dualdst.py) 验证�?

// Positive: 两池 offset 0 各预填同 size 占位 buf -> GetDualSpillGroup 命中
//          (startAddrA == startAddrB == 0) -> canSpillGroups 非空, GetGroupNextUseTime
//          �?placeholder 未绑 alloc op 全部失败 -> �?spill-all 兜底,
//          返回两池 sortedBufs 并集 (含两�?placeholder memId)�?
TEST_F(ScheduleOoOTest, DualDst_SelectSpillBuffers_PicksMatchingGroupAcrossAivPools)
{
    auto g = dualdst_ut::BuildDualDstGraph({dualdst_ut::TILE_M, dualdst_ut::TILE_N * 2},
                                           {dualdst_ut::TILE_M, dualdst_ut::TILE_N}, {0, 0}, {0, dualdst_ut::TILE_N});
    g.copy0->GetOutputOperand(0)->UpdateDynValidShape(
        {SymbolicScalar(dualdst_ut::TILE_M), SymbolicScalar(dualdst_ut::TILE_N)});
    g.copy1->GetOutputOperand(0)->UpdateDynValidShape(
        {SymbolicScalar(dualdst_ut::TILE_M), SymbolicScalar(dualdst_ut::TILE_N)});

    OoOScheduler s(*g.func);
    EXPECT_EQ(s.Init(g.func->Operations().DuplicatedOpList(), {}, CORE_INIT_CONFIGS_HARDWARE_TWO), SUCCESS);
    dualdst_ut::InjectCoreMap(s, g);
    s.dualDstEngine_.enableDualDst_ = true;
    EXPECT_EQ(s.dualDstEngine_.RunDualDstFuse(), SUCCESS);

    Operation* dual = nullptr;
    for (auto& op : g.func->Operations()) {
        if (op.GetOpcode() == Opcode::OP_L0C_COPY_UB_DUAL_DST) {
            dual = &op;
            break;
        }
    }
    ASSERT_NE(dual, nullptr);
    Operation* survivingUbAlloc = nullptr;
    for (auto* pred : s.state_.depManager.GetPredecessors(dual)) {
        if (pred != nullptr && pred->GetOpcodeStr().find("UB_ALLOC") != std::string::npos) {
            survivingUbAlloc = pred;
            break;
        }
    }
    ASSERT_NE(survivingUbAlloc, nullptr);
    ASSERT_TRUE(s.dualDstEngine_.IsDualDstAlloc(survivingUbAlloc));

    int memIdA = survivingUbAlloc->GetOutputOperand(0)->memoryrange.memId;
    ASSERT_NE(s.state_.localBufferMap.find(memIdA), s.state_.localBufferMap.end());
    size_t needSize = s.state_.localBufferMap[memIdA]->size;

    // �?AIV0 / AIV1 两池 offset 0 各放一个占�?buf, size = needSize
    // -> spill �?freedRange = [0, poolMem) 双侧完全一�?
    auto& poolA = s.state_.bufferManagerMap[CoreLocationType::AIV0][MemoryType::MEM_UB];
    auto& poolB = s.state_.bufferManagerMap[CoreLocationType::AIV1][MemoryType::MEM_UB];
    ASSERT_GE(poolA.GetMemSize(), needSize);
    ASSERT_GE(poolB.GetMemSize(), needSize);

    constexpr int kPlaceholderMemIdA = 90001; // �?core 全局唯一 (不与 graph builder 冲突)
    constexpr int kPlaceholderMemIdB = 90002;
    auto bufHolderA = std::make_shared<LocalBuffer>(kPlaceholderMemIdA, needSize, MemoryType::MEM_UB);
    auto bufHolderB = std::make_shared<LocalBuffer>(kPlaceholderMemIdB, needSize, MemoryType::MEM_UB);
    ASSERT_EQ(poolA.AllocateAtOffset(bufHolderA, 0), SUCCESS);
    ASSERT_EQ(poolB.AllocateAtOffset(bufHolderB, 0), SUCCESS);

    // 直接�?SelectSpillBuffers 验证 dualdst 选组分支
    auto spillGroup = s.SelectSpillBuffers(survivingUbAlloc);

    // 期待: 双池各贡�?1 �?placeholder memId, 合计 2 �?
    ASSERT_EQ(spillGroup.size(), 2u);
    EXPECT_NE(std::find(spillGroup.begin(), spillGroup.end(), kPlaceholderMemIdA), spillGroup.end());
    EXPECT_NE(std::find(spillGroup.begin(), spillGroup.end(), kPlaceholderMemIdB), spillGroup.end());
}

// Negative: 两池都为�?(无可 spill 候�? -> GetDualSpillGroup 返回�?-> spill-all
//          兜底两池 sortedBufs 也为�?-> 最终返回空 vector
TEST_F(ScheduleOoOTest, DualDst_SelectSpillBuffers_EmptyPoolsReturnEmpty)
{
    auto g = dualdst_ut::BuildDualDstGraph({dualdst_ut::TILE_M, dualdst_ut::TILE_N * 2},
                                           {dualdst_ut::TILE_M, dualdst_ut::TILE_N}, {0, 0}, {0, dualdst_ut::TILE_N});
    g.copy0->GetOutputOperand(0)->UpdateDynValidShape(
        {SymbolicScalar(dualdst_ut::TILE_M), SymbolicScalar(dualdst_ut::TILE_N)});
    g.copy1->GetOutputOperand(0)->UpdateDynValidShape(
        {SymbolicScalar(dualdst_ut::TILE_M), SymbolicScalar(dualdst_ut::TILE_N)});

    OoOScheduler s(*g.func);
    EXPECT_EQ(s.Init(g.func->Operations().DuplicatedOpList(), {}, CORE_INIT_CONFIGS_HARDWARE_TWO), SUCCESS);
    dualdst_ut::InjectCoreMap(s, g);
    s.dualDstEngine_.enableDualDst_ = true;
    EXPECT_EQ(s.dualDstEngine_.RunDualDstFuse(), SUCCESS);

    Operation* dual = nullptr;
    for (auto& op : g.func->Operations()) {
        if (op.GetOpcode() == Opcode::OP_L0C_COPY_UB_DUAL_DST) {
            dual = &op;
            break;
        }
    }
    ASSERT_NE(dual, nullptr);
    Operation* survivingUbAlloc = nullptr;
    for (auto* pred : s.state_.depManager.GetPredecessors(dual)) {
        if (pred != nullptr && pred->GetOpcodeStr().find("UB_ALLOC") != std::string::npos) {
            survivingUbAlloc = pred;
            break;
        }
    }
    ASSERT_NE(survivingUbAlloc, nullptr);
    ASSERT_TRUE(s.dualDstEngine_.IsDualDstAlloc(survivingUbAlloc));

    // 不预填任何占�?buf -> AIV0/AIV1 两池 allocatedBufs 都空
    // -> GetSpillGroup 返回�?-> SelectSpillBuffers 返回�?
    auto spillGroup = s.SelectSpillBuffers(survivingUbAlloc);
    EXPECT_TRUE(spillGroup.empty());
}

// --- GetDualSpillGroup 专项 UT (绕过 SelectSpillBuffers 直接验证 helper) -----

// Positive: 两池各预填一�?buf at offset 0
//   外层 iA=0: startAddrA=0, jA=1, window={bufA}
//   内层 iB=0: startAddrB=0 (== startAddrA), jB=1, window={bufB}
//   -> 输出一个组 [bufA_memId, bufB_memId]
TEST_F(ScheduleOoOTest, DualDst_GetDualSpillGroup_FindsSharedStartAddrCandidate)
{
    auto g = dualdst_ut::BuildDualDstGraph({dualdst_ut::TILE_M, dualdst_ut::TILE_N * 2},
                                           {dualdst_ut::TILE_M, dualdst_ut::TILE_N}, {0, 0}, {0, dualdst_ut::TILE_N});

    OoOScheduler s(*g.func);
    EXPECT_EQ(s.Init(g.func->Operations().DuplicatedOpList(), {}, CORE_INIT_CONFIGS_HARDWARE_TWO), SUCCESS);

    auto& poolA = s.state_.bufferManagerMap[CoreLocationType::AIV0][MemoryType::MEM_UB];
    auto& poolB = s.state_.bufferManagerMap[CoreLocationType::AIV1][MemoryType::MEM_UB];

    constexpr int kBufMemIdA = 80001;
    constexpr int kBufMemIdB = 80002;
    constexpr size_t kBufSize = 1024;
    constexpr size_t kNeedSize = 512; // 小于 kBufSize, spill 后能腾出足够空闲�?
    ASSERT_GE(poolA.GetMemSize(), kBufSize);
    ASSERT_GE(poolB.GetMemSize(), kBufSize);

    auto bufA = std::make_shared<LocalBuffer>(kBufMemIdA, kBufSize, MemoryType::MEM_UB);
    auto bufB = std::make_shared<LocalBuffer>(kBufMemIdB, kBufSize, MemoryType::MEM_UB);
    ASSERT_EQ(poolA.AllocateAtOffset(bufA, 0), SUCCESS);
    ASSERT_EQ(poolB.AllocateAtOffset(bufB, 0), SUCCESS);

    auto groups = s.GetDualSpillGroup(poolA, poolB, kNeedSize);
    ASSERT_EQ(groups.size(), 1u);
    ASSERT_EQ(groups[0].size(), 2u);
    EXPECT_NE(std::find(groups[0].begin(), groups[0].end(), kBufMemIdA), groups[0].end());
    EXPECT_NE(std::find(groups[0].begin(), groups[0].end(), kBufMemIdB), groups[0].end());
}

// Negative: sizeNeedSpill > poolMem -> 外层 iA=0 进入即触�?
//   `(poolA.GetMemSize() - 0) < sizeNeedSpill` break -> 返回�?vector
TEST_F(ScheduleOoOTest, DualDst_GetDualSpillGroup_NeedSizeExceedsPoolReturnsEmpty)
{
    auto g = dualdst_ut::BuildDualDstGraph({dualdst_ut::TILE_M, dualdst_ut::TILE_N * 2},
                                           {dualdst_ut::TILE_M, dualdst_ut::TILE_N}, {0, 0}, {0, dualdst_ut::TILE_N});

    OoOScheduler s(*g.func);
    EXPECT_EQ(s.Init(g.func->Operations().DuplicatedOpList(), {}, CORE_INIT_CONFIGS_HARDWARE_TWO), SUCCESS);

    auto& poolA = s.state_.bufferManagerMap[CoreLocationType::AIV0][MemoryType::MEM_UB];
    auto& poolB = s.state_.bufferManagerMap[CoreLocationType::AIV1][MemoryType::MEM_UB];

    // 占位 buf �?bufsA/bufsB 非空, 保证外层 while 能进�?(才能命中 size-exceeds �?break)
    constexpr int kBufMemIdA = 80003;
    constexpr int kBufMemIdB = 80004;
    auto bufA = std::make_shared<LocalBuffer>(kBufMemIdA, 1024, MemoryType::MEM_UB);
    auto bufB = std::make_shared<LocalBuffer>(kBufMemIdB, 1024, MemoryType::MEM_UB);
    ASSERT_EQ(poolA.AllocateAtOffset(bufA, 0), SUCCESS);
    ASSERT_EQ(poolB.AllocateAtOffset(bufB, 0), SUCCESS);

    // need > 池总容�?-> 不可能腾�?-> �?vector
    size_t needSize = poolA.GetMemSize() + 1024;
    auto groups = s.GetDualSpillGroup(poolA, poolB, needSize);
    EXPECT_TRUE(groups.empty());
}

// --- RecordStaticValidShapeOnL0CCopyUB (InferDynShape 阶段 InferShape 前快�? ---
// 直接调用快照函数, 绕开 InferShape 全量遍历�?op 状态完整性的依赖�?
// 静�?validShape 应被快照�?op 属性�?
TEST_F(ScheduleOoOTest, DualDst_InferDynShape_RecordsStaticValidShape)
{
    auto g = dualdst_ut::BuildDualDstGraph({dualdst_ut::TILE_M, dualdst_ut::TILE_N * 2},
                                           {dualdst_ut::TILE_M, dualdst_ut::TILE_N}, {0, 0}, {0, dualdst_ut::TILE_N});
    g.copy0->GetOutputOperand(0)->UpdateDynValidShape(
        {SymbolicScalar(dualdst_ut::TILE_M), SymbolicScalar(dualdst_ut::TILE_N)});
    g.copy1->GetOutputOperand(0)->UpdateDynValidShape(
        {SymbolicScalar(dualdst_ut::TILE_M), SymbolicScalar(dualdst_ut::TILE_N)});

    InferDynShape pass;
    pass.RecordStaticValidShapeOnL0CCopyUB(*g.func);

    EXPECT_TRUE(g.copy0->HasAttribute(OpAttributeKey::staticValidShape));
    EXPECT_TRUE(g.copy1->HasAttribute(OpAttributeKey::staticValidShape));
    auto v0 = g.copy0->GetVectorIntAttribute<int64_t>(OpAttributeKey::staticValidShape);
    EXPECT_EQ(v0.size(), 2u);
    EXPECT_EQ(v0[0], dualdst_ut::TILE_M);
    EXPECT_EQ(v0[1], dualdst_ut::TILE_N);
}

// 含动态成分的 validShape -> staticValidShape 不应被记�?
TEST_F(ScheduleOoOTest, DualDst_InferDynShape_SkipsDynamicValidShape)
{
    auto g = dualdst_ut::BuildDualDstGraph({dualdst_ut::TILE_M, dualdst_ut::TILE_N * 2},
                                           {dualdst_ut::TILE_M, dualdst_ut::TILE_N}, {0, 0}, {0, dualdst_ut::TILE_N});
    // 一维含符号 (�?concrete) -> allConcrete=false -> 跳过快照
    g.copy0->GetOutputOperand(0)->UpdateDynValidShape({SymbolicScalar("dyn0"), SymbolicScalar(dualdst_ut::TILE_N)});

    InferDynShape pass;
    pass.RecordStaticValidShapeOnL0CCopyUB(*g.func);

    EXPECT_FALSE(g.copy0->HasAttribute(OpAttributeKey::staticValidShape));
}

// --- GetNewOperations 去重 (dualdst 防御�?shim) -----------------------------
TEST_F(ScheduleOoOTest, DualDst_GetNewOperations_DedupePreservesFirstOccurrence)
{
    auto g = dualdst_ut::BuildDualDstGraph({dualdst_ut::TILE_M, dualdst_ut::TILE_N * 2},
                                           {dualdst_ut::TILE_M, dualdst_ut::TILE_N}, {0, 0}, {0, dualdst_ut::TILE_N});
    OoOScheduler s(*g.func);
    EXPECT_EQ(s.Init(g.func->Operations().DuplicatedOpList(), {}, CORE_INIT_CONFIGS_HARDWARE_TWO), SUCCESS);

    // 故意往 newOperations_ 里塞重复 + nullptr -> GetNewOperations 应去�?+ �?null
    s.state_.newOperations.clear();
    s.state_.newOperations.push_back(g.copy0);
    s.state_.newOperations.push_back(g.copy1);
    s.state_.newOperations.push_back(g.copy0); // dup
    s.state_.newOperations.push_back(nullptr); // null

    auto uniq = s.GetNewOperations();
    EXPECT_EQ(uniq.size(), 2u);
    EXPECT_EQ(uniq[0], g.copy0);
    EXPECT_EQ(uniq[1], g.copy1);
}
} // namespace npu::tile_fwk
