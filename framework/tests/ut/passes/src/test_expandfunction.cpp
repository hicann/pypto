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
 * \file test_expand_function.cpp
 * \brief Unit test for ExpandFunction pass.
 */

#include <gtest/gtest.h>
#include <vector>
#include <string>
#include "interface/function/function.h"
#include "tilefwk/tilefwk.h"
#include "ut_json/ut_json_tool.h"
#include "passes/pass_mgr/pass_manager.h"
#include "interface/configs/config_manager.h"

#define private public
#include "interface/operation/operation.h"
#include "passes/tensor_graph_pass/expand_function.h"

namespace npu {
namespace tile_fwk{
static const size_t kSizeZero = 0UL;
static const size_t kSizeThree = 3UL;
static const size_t kSizeEight = 8UL;
static const size_t kSizeEleven = 11UL;
static const uint16_t kNumZero = 0u;
static const uint16_t kNumOne = 1u;
static const uint16_t kNumTwo = 2u;
static const uint16_t kNumThree = 3u;
static const uint16_t kNumFour = 4u;
static const uint16_t kNumEight = 8u;
static const uint16_t kNumForteen = 14u;
static const uint16_t kNumExpFour = 16u;
static const uint16_t kNumTwentyfive = 25u;
static const uint16_t kNumExpFive = 32u;
static const uint16_t kNumExpSix = 64u;
static const uint16_t kNumExpSeven = 128u;

void MakeExpandGrpah(std::shared_ptr<Function> &currFunctionPtr, LogicalTensorPtr& outCast) {
    std::vector<int64_t> shape = {kNumExpSix, kNumExpSix};
    std::vector<int64_t> tile_shape = {kNumExpFive, kNumExpFive};
    auto inCast1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    auto inCast2 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    auto ubTensor = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    outCast = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);

    auto& div_op = currFunctionPtr->AddOperation(Opcode::OP_DIV, {inCast1, inCast2}, {ubTensor});
    auto& assemble_op = currFunctionPtr->AddOperation(Opcode::OP_ASSEMBLE, {ubTensor}, {outCast});
    std::vector<int64_t> toOffset = {kNumZero, kNumZero};
    std::vector<SymbolicScalar> symbol = {SymbolicScalar("sym")};
    auto op_attr = std::make_shared<AssembleOpAttribute>(toOffset, symbol);
    assemble_op.SetOpAttribute(op_attr);
    div_op.tileShape_.SetVecTile(tile_shape);
    assemble_op.tileShape_.SetVecTile(tile_shape);

    currFunctionPtr->inCasts_.push_back(inCast1);
    currFunctionPtr->inCasts_.push_back(inCast2);
    currFunctionPtr->outCasts_.push_back(outCast);
    currFunctionPtr->SetGraphType(GraphType::TENSOR_GRAPH);
}

class TestExpandFunctionPass : public ::testing::Test {
public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}

    void SetUp() override {
        Program::GetInstance().Reset();
        config::Reset();
        config::SetPlatformConfig(KEY_ONLY_HOST_COMPILE, true);
        config::SetHostConfig(KEY_STRATEGY, "ExpandFunctionTestStrategy");
        config::SetPlatformConfig(KEY_ENABLE_COST_MODEL, false);
    }
    void TearDown() override {}
};

/*
TESTExpandFunctionNotTensorGrpah
inCast{8,16}->nop->outCast{8,16}

inCast{8,16}->nop->outCast{8,16}
*/
TEST_F(TestExpandFunctionPass, ExpandFunctionUTest1) {
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestExpandFunction", "TestExpandFunction", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    // Prepare the graph
    std::vector<int64_t> shape = {kNumEight, kNumExpFour};
    auto inCast = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    auto outCast = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);

    auto& nop_op = currFunctionPtr->AddOperation(Opcode::OP_NOP, {inCast}, {outCast});

    currFunctionPtr->inCasts_.push_back(inCast);
    currFunctionPtr->outCasts_.push_back(outCast);
    currFunctionPtr->SetGraphType(GraphType::TILE_GRAPH);

    ExpandFunction expandfunctionpass;
    auto status = expandfunctionpass.RunOnFunction(*currFunctionPtr);
    EXPECT_EQ(status, SUCCESS);
    EXPECT_EQ(currFunctionPtr->GetGraphType(), GraphType::TILE_GRAPH);

    uint32_t nop_num = kNumZero;
    for (auto &op : currFunctionPtr->Operations()) {
        if (op.GetOpcode() == Opcode::OP_NOP) {
            EXPECT_EQ(nop_op.GetOpMagic(), op.GetOpMagic());
            ++nop_num;
        }
    }
    EXPECT_EQ(nop_num, kNumOne);
}

/*
TESTExpandFunctionNOP
inCast{8,16}->pad->ubTensor1{8,16}->nop->ubTensor2{8,16}->view->outCast{8,16}
pad is removed
*/
TEST_F(TestExpandFunctionPass, ExpandFunctionUTest2) {
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestExpandFunction", "TestExpandFunction", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    std::vector<int64_t> shape = {kNumEight, kNumExpFour};
    auto inCast = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    auto ubTensor1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    auto ubTensor2 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    auto outCast = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);

    auto op_attr = std::make_shared<ViewOpAttribute>(std::vector<int64_t>{kNumZero, kNumZero});
    currFunctionPtr->AddOperation(Opcode::OP_PAD, {inCast}, {ubTensor1});
    currFunctionPtr->AddOperation(Opcode::OP_NOP, {ubTensor1}, {ubTensor2});
    currFunctionPtr->AddOperation(Opcode::OP_VIEW, {ubTensor2}, {outCast});
    
    std::shared_ptr<Operation> pad_op, nop_op, view_op;
    for (uint32_t uIndex = 0; uIndex < currFunctionPtr->Operations().size(); ++uIndex){
        auto op = currFunctionPtr->Operations().operations_[uIndex];
        if (op->GetOpcode() == Opcode::OP_PAD) pad_op = op;
        else if (op->GetOpcode() == Opcode::OP_NOP) nop_op = op;
        else if (op->GetOpcode() == Opcode::OP_VIEW) view_op = op;
    }

    view_op->SetOpAttribute(op_attr);
    
    currFunctionPtr->inCasts_.push_back(inCast);
    currFunctionPtr->outCasts_.push_back(outCast);
    currFunctionPtr->SetGraphType(GraphType::TENSOR_GRAPH);

    ExpandFunction expandfunctionpass;
    auto status = expandfunctionpass.RunOnFunction(*currFunctionPtr);
    EXPECT_EQ(status, SUCCESS);
    EXPECT_EQ(currFunctionPtr->GetGraphType(), GraphType::TILE_GRAPH);

    uint32_t view_num = kNumZero, pad_num = kNumZero, nop_num = kNumZero;
    for (auto &op : currFunctionPtr->Operations()) {
        if (op.GetOpcode() == Opcode::OP_VIEW) {
            EXPECT_EQ(op_attr, view_op->GetOpAttribute());
            EXPECT_NE(view_op->GetOpMagic(), op.GetOpMagic()); ++view_num;
        } else if (op.GetOpcode() == Opcode::OP_PAD) {
            EXPECT_TRUE(pad_op->GetOpAttribute() == nullptr);
            EXPECT_NE(pad_op->GetOpMagic(), op.GetOpMagic()); ++pad_num;
        } else if (op.GetOpcode() == Opcode::OP_NOP) {
            EXPECT_NE(nop_op->GetOpMagic(), op.GetOpMagic()); ++nop_num;
        }
    }
    EXPECT_EQ(view_num, kNumOne);
    EXPECT_EQ(pad_num, kNumOne);
    EXPECT_EQ(nop_num, kNumOne);
}

/*
TESTExpandFunctionAssemble
inCast{64,64}->assemble->view->outCast{64,64}
assemble is kept
*/
TEST_F(TestExpandFunctionPass, ExpandFunctionUTest3) {
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestExpandFunction", "TestExpandFunction", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    // Prepare the graph
    std::vector<int64_t> shape = {kNumExpSix, kNumExpSix};
    auto inCast = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    auto outCast = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);

    std::vector<int64_t> toOffset = {kNumZero, kNumZero};
    std::vector<SymbolicScalar> symbol = {SymbolicScalar("sym")};
    auto op_attr = std::make_shared<AssembleOpAttribute>(toOffset, symbol);
    currFunctionPtr->AddOperation(Opcode::OP_ASSEMBLE, {inCast}, {outCast});
    
    std::shared_ptr<Operation> assemble_op;
    for (uint32_t uIndex = 0; uIndex < currFunctionPtr->Operations().size(); ++uIndex){
        if (currFunctionPtr->Operations().operations_[uIndex]->GetOpcode() == Opcode::OP_ASSEMBLE) {
            assemble_op = currFunctionPtr->Operations().operations_[uIndex];
        }
    }

    assemble_op->SetOpAttribute(op_attr);

    currFunctionPtr->inCasts_.push_back(inCast);
    currFunctionPtr->outCasts_.push_back(outCast);
    currFunctionPtr->SetGraphType(GraphType::TENSOR_GRAPH);

    ExpandFunction expandfunctionpass;
    auto status = expandfunctionpass.RunOnFunction(*currFunctionPtr);
    EXPECT_EQ(status, SUCCESS);
    EXPECT_EQ(currFunctionPtr->GetGraphType(), GraphType::TILE_GRAPH);

    uint32_t assemble_num = kNumZero;
    for (auto &op : currFunctionPtr->Operations()) {
        if (op.GetOpcode() == Opcode::OP_ASSEMBLE) {
            EXPECT_EQ(op_attr, op.GetOpAttribute());
            EXPECT_NE(op.GetOpMagic(), assemble_op->GetOpMagic());
            ++assemble_num;
        }
    }
    EXPECT_EQ(assemble_num, kNumOne);
}

/*
TESTExpandFunctionAssemble
inCast1{64,64}->div->ubTensor{64,64}->assemble->outCast{64,64}
inCast2{64,64}->
inCast1{64,64}->view*4->div->ubTensor{64,64}->assemble(*4)->outCast{64,64}
inCast2{64,64}->view*4->
*/
TEST_F(TestExpandFunctionPass, ExpandFunctionUTest4) {
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestExpandFunction", "TestExpandFunction", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);
    std::vector<int64_t> shape = {kNumExpSix, kNumExpSix};
    std::vector<int64_t> tile_shape = {kNumExpFive, kNumExpFive};
    TileShape::Current().SetVecTile(kNumExpFive, kNumExpFive);
    LogicalTensorPtr outCast;
    MakeExpandGrpah(currFunctionPtr, outCast);

    ExpandFunction expandfunctionpass;
    EXPECT_EQ(expandfunctionpass.RunOnFunction(*currFunctionPtr), SUCCESS);
    EXPECT_EQ(currFunctionPtr->GetGraphType(), GraphType::TILE_GRAPH);

    uint32_t div_num = kNumZero;
    uint32_t view_num = kNumZero;
    LogicalTensorPtr assemble_input;
    std::vector<LogicalTensorPtr> additional_assemble;
    for (auto &op : currFunctionPtr->Operations()) {
        if (op.GetOpcode() == Opcode::OP_DIV) {
            EXPECT_EQ(op.GetInputOperand(kSizeZero)->shape, tile_shape);
            EXPECT_EQ(op.GetOutputOperand(kSizeZero)->shape, tile_shape);
            EXPECT_NE(op.GetOutputOperand(kSizeZero), outCast);
            ++div_num;
        } else if (op.GetOpcode() == Opcode::OP_VIEW) {
            ++view_num;
            EXPECT_EQ(op.GetInputOperand(kSizeZero)->shape, shape);
            EXPECT_EQ(op.GetOutputOperand(kSizeZero)->shape, tile_shape);
        } else if (op.GetOpcode() == Opcode::OP_ASSEMBLE) {
            EXPECT_EQ(op.GetOutputOperandSize(), kNumOne);
            if (op.GetOutputOperand(kSizeZero) != outCast) {
                EXPECT_EQ(op.GetOutputOperandSize(), kNumOne);
                additional_assemble.emplace_back(op.GetOutputOperand(kSizeZero));
            } else {
                EXPECT_EQ(op.GetInputOperandSize(), kNumOne);
                assemble_input = op.GetInputOperand(kSizeZero);
            }
        }
    }
    EXPECT_EQ(div_num, kNumFour);
    EXPECT_EQ(view_num, kNumEight);
    EXPECT_EQ(additional_assemble.size(), kNumFour);
    for (size_t i = 0; i < additional_assemble.size(); ++i) {
        EXPECT_EQ(assemble_input, additional_assemble[i]);
    }
}

/*
TESTExpandFunctionAssembleV2
inCast1{32,128}->reshape->ubTensor{64,64}->assemble->outCast{32,128}
inCast1{32,128}->reshape->ubTensor{64,64}->assemble(*4)->outCast{32,128}
*/
TEST_F(TestExpandFunctionPass, ExpandFunctionUTest5) {
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestExpandFunction", "TestExpandFunction", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    // Prepare the graph
    std::vector<int64_t> shape1 = {kNumExpFive, kNumExpSeven};
    std::vector<int64_t> shape2 = {kNumExpSix, kNumExpSix};
    std::vector<int64_t> shape3 = {kNumExpFive, kNumExpSeven};
    auto inCast = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape1);
    auto ubTensor = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape2);
    auto outCast = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape3);

    std::vector<int64_t> toOffset = {kNumZero, kNumZero};
    std::vector<SymbolicScalar> symbol = {SymbolicScalar("sym")};
    auto op_attr = std::make_shared<AssembleOpAttribute>(toOffset, symbol);
    currFunctionPtr->AddOperation(Opcode::OP_RESHAPE, {inCast}, {ubTensor});
    currFunctionPtr->AddOperation(Opcode::OP_ASSEMBLE, {ubTensor}, {outCast});

    std::shared_ptr<Operation> reshape_op;
    std::shared_ptr<Operation> assemble_op;
    for (uint32_t uIndex = 0; uIndex < currFunctionPtr->Operations().size(); ++uIndex){
        if (currFunctionPtr->Operations().operations_[uIndex]->GetOpcode() == Opcode::OP_RESHAPE) {
            reshape_op = currFunctionPtr->Operations().operations_[uIndex];
        }
        if (currFunctionPtr->Operations().operations_[uIndex]->GetOpcode() == Opcode::OP_ASSEMBLE) {
            assemble_op = currFunctionPtr->Operations().operations_[uIndex];
        }
    }

    assemble_op->SetOpAttribute(op_attr);
    reshape_op->tileShape_.SetVecTile({kNumExpFive, kNumExpFive});
    assemble_op->tileShape_.SetVecTile({kNumExpFive, kNumExpFive});

    currFunctionPtr->inCasts_.push_back(inCast);
    currFunctionPtr->outCasts_.push_back(outCast);
    currFunctionPtr->SetGraphType(GraphType::TENSOR_GRAPH);

    ExpandFunction expandfunctionpass;
    auto status = expandfunctionpass.RunOnFunction(*currFunctionPtr);
    EXPECT_EQ(status, SUCCESS);
    EXPECT_EQ(currFunctionPtr->GetGraphType(), GraphType::TILE_GRAPH);

    uint32_t assemble_num = kNumZero;
    for (auto &op : currFunctionPtr->Operations()) {
        if (op.GetOpcode() == Opcode::OP_ASSEMBLE) {
            EXPECT_NE(op.GetOpMagic(), assemble_op->GetOpMagic());
            ++assemble_num;
        }
    }
    EXPECT_EQ(assemble_num, kNumFour);
}

/*
{64, 64} -> exp -> {64, 64}
{64, 64} -> (view) - > exp -> (assemble) - > {64, 64}
{64, 64} -> (view) -> (view {32, 64}) - > exp -> (assemble {32, 64}) -> (assemble) - > {64, 64}
                   -> (view {32, 64}) - > exp -> (assemble {32, 64})
*/
TEST_F(TestExpandFunctionPass, ExpandFunctionSTest1) {
    std::vector<int64_t> shape = {kNumExpSix, kNumExpSix};
    std::vector<int64_t> tile_shape = {kNumExpFive, kNumExpSix};
    PassManager &passManager = PassManager::Instance();
    Tensor input(DT_FP32, shape, "input");
    Tensor output(DT_FP32, shape, "output");
    TileShape::Current().SetVecTile(tile_shape);
    FUNCTION("STCase1") {
        output = Exp(input);
    }

    Function* func = Program::GetInstance().GetFunctionByRawName("TENSOR_STCase1");
    EXPECT_EQ(func->Operations().size(), kSizeThree);
    passManager.RegisterStrategy("ExpandFunctionTestStrategy", {
        {   "ExpandFunction",   PassName::EXPAND_FUNCTION},
    });
    auto ret = passManager.RunPass(Program::GetInstance(), *func, "ExpandFunctionTestStrategy");
    EXPECT_EQ(ret, SUCCESS);

    // ================== Verify the effect of the Pass ==================
    auto updated_operations = func->Operations();
    int exp_num = kNumZero;
    int view_num = kNumZero;
    int assemble_num = kNumZero;
    EXPECT_EQ(updated_operations.size(), kSizeEight);
    for (const auto &op : updated_operations) {
        if (op.GetOpcode() == Opcode::OP_EXP) {
            exp_num++;
            EXPECT_EQ(op.GetInputOperand(0)->shape, tile_shape);
            EXPECT_EQ(op.GetOutputOperand(0)->shape, tile_shape);
        } else if (op.GetOpcode() == Opcode::OP_VIEW) {
            view_num++;
        } else if (op.GetOpcode() == Opcode::OP_ASSEMBLE) {
            assemble_num++;
            if (op.GetInputOperand(kSizeZero)->shape != op.GetOutputOperand(0)->shape) {
                EXPECT_EQ(op.GetInputOperand(kSizeZero)->shape, tile_shape);
                EXPECT_EQ(op.GetOutputOperand(kSizeZero)->shape, shape);
            } else {
                EXPECT_EQ(op.GetInputOperand(kSizeZero)->shape, shape);
                EXPECT_EQ(op.GetOutputOperand(kSizeZero)->shape, shape);
            }
        }
    }
    EXPECT_EQ(view_num, kNumThree);
    EXPECT_EQ(exp_num, kNumTwo);
    EXPECT_EQ(assemble_num, kNumThree);
}

/*
{64, 64} -> exp -> view -> reciprocal
                        -> sqrt -> reshape
{64, 64} -> view -> exp -> view          -> sqrt         -> reshape        ->assemble(end)
                                                         -> assemble(end)
                                         -> reciprocal   -> assemble(end)
                                         -> assemble(end)
                        -> assemble(end)
view -> view(*4) -> exp(*4) -> assemble(*4) ->view  -> view(*4+4)   -> sqrt(*4)         -> assemble(*4)     -> reshape      -> assemble(end)
                                                                                        -> assemble(*4)     -> assemble(*4) -> assemble(end)
                                            ->assemble(end)         -> reciprocal(*4)   -> assemble(*4)     -> assemble(end)
*/
void ConstructGraphST2() {
    std::vector<int64_t> shape = {kNumExpSix, kNumExpSix};
    std::vector<int64_t> view_shape = {kNumExpSeven, kNumExpFive};
    std::vector<int64_t> reshape_shape = {kNumExpFive, kNumExpSeven};
    std::vector<int64_t> tile_shape = {kNumExpFive, kNumExpFive};

    Tensor input(DT_FP32, shape, "input");
    Tensor exp(DT_FP32, shape, "exp");
    Tensor view(DT_FP32, view_shape, "view");
    Tensor output1(DT_FP32, view_shape, "output");
    Tensor sqrt(DT_FP32, view_shape, "sqrt");
    Tensor output2(DT_FP32, reshape_shape, "sqrt");

    FUNCTION("STCase2") {
        TileShape::Current().SetVecTile(tile_shape);
        exp = Exp(input);
        view = View(exp, view_shape, {kNumZero, kNumZero});
        output1 = Reciprocal(view);
        sqrt = Sqrt(view);
        output2 = Reshape(sqrt, reshape_shape);
    }
}

TEST_F(TestExpandFunctionPass, ExpandFunctionSTest2) {
    PassManager &passManager = PassManager::Instance();
    ConstructGraphST2();
    Function* func = Program::GetInstance().GetFunctionByRawName("TENSOR_STCase2");
    passManager.RegisterStrategy("ExpandFunctionTestStrategy", {
        {   "ExpandFunction",   PassName::EXPAND_FUNCTION},
    });
    auto ret = passManager.RunPass(Program::GetInstance(), *func, "ExpandFunctionTestStrategy");
    EXPECT_EQ(ret, SUCCESS);

    // ================== Verify the effect of the Pass ==================
    auto updated_operations = func->Operations();

    int exp_num = kNumZero;
    int sqrt_num = kNumZero;
    int reshape_num = kNumZero;
    int reciprocal_num = kNumZero;
    int view_num = kNumZero;
    int assemble_num = kNumZero;
    for (const auto &op : updated_operations) {
        if (op.GetOpcode() == Opcode::OP_EXP) {
            exp_num++;
        } else if (op.GetOpcode() == Opcode::OP_VIEW) {
            view_num++;
        } else if (op.GetOpcode() == Opcode::OP_ASSEMBLE) {
            assemble_num++;
        } else if (op.GetOpcode() == Opcode::OP_RESHAPE) {
            reshape_num++;
        } else if (op.GetOpcode() == Opcode::OP_SQRT) {
            sqrt_num++;
        } else if (op.GetOpcode() == Opcode::OP_RECIPROCAL) {
            reciprocal_num++;
        }
    }
    // 12个前连接的view + 5个copy和expand的view + 1个开头的view
    EXPECT_EQ(view_num, kNumForteen);
    EXPECT_EQ(exp_num, kNumFour);
    // 12个后链接的assemble
    EXPECT_EQ(assemble_num, kNumTwentyfive);
    // reshape前合入
    EXPECT_EQ(reshape_num, kNumOne);
    EXPECT_EQ(sqrt_num, kNumFour);
    EXPECT_EQ(reciprocal_num, kNumFour);
}

/*
TESTExpandFunctionLoop
inCast{64,64}->assemble->view->outCast{64,64}
             <-assemble<-
loop will be detected
*/
TEST_F(TestExpandFunctionPass, ExpandFunctionUTest6) {
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestExpandFunction", "TestExpandFunction", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    // Prepare the graph
    std::vector<int64_t> shape = {kNumExpSix, kNumExpSix};
    auto inCast = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    auto outCast = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);

    std::vector<int64_t> toOffset = {kNumZero, kNumZero};
    std::vector<SymbolicScalar> symbol = {SymbolicScalar("sym")};
    auto op_attr = std::make_shared<AssembleOpAttribute>(toOffset, symbol);
    auto& assemble_op = currFunctionPtr->AddOperation(Opcode::OP_ASSEMBLE, {inCast}, {outCast});
    assemble_op.SetOpAttribute(op_attr);

    auto& assemble_op_loop = currFunctionPtr->AddOperation(Opcode::OP_ASSEMBLE, {outCast}, {inCast});
    assemble_op_loop.SetOpAttribute(op_attr);

    currFunctionPtr->inCasts_.push_back(inCast);
    currFunctionPtr->outCasts_.push_back(outCast);
    currFunctionPtr->SetGraphType(GraphType::TENSOR_GRAPH);

    ExpandFunction expandfunctionpass;
    EXPECT_EQ(expandfunctionpass.PreCheck(*currFunctionPtr), FAILED);

    currFunctionPtr->SetGraphType(GraphType::TILE_GRAPH);
    EXPECT_EQ(expandfunctionpass.PostCheck(*currFunctionPtr), FAILED);
}
}
}