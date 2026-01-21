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
 * \file test_remove_redundant_op.cpp
 * \brief Unit test for RemoveRedundantOp pass.
 */

#include "gtest/gtest.h"
#include "tilefwk/tilefwk_op.h"
#include "interface/function/function.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "passes/pass_mgr/pass_manager.h"
#include "interface/configs/config_manager.h"
#include "ut_json/ut_json_tool.h"
#include "passes/tile_graph_pass/graph_optimization/remove_redundant_op.h"
#include <fstream>
#include <vector>
#include <string>

using namespace npu::tile_fwk;

void PrintGraphInfoRemoveRedundantOp(Function* func) {
    std::cout << "func->Operations().size() = "  << func->Operations().size() << std::endl;
    for (auto &op : func->Operations()) {
        std::cout << "Op:" << op.GetOpMagic() << " " <<  op.GetOpcodeStr() << std::endl;
        std::cout << "input operation:";
        for (const std::shared_ptr<LogicalTensor> &input_tensor : op.GetIOperands()) {
            for (const auto &item_op : input_tensor->GetProducers()) {
                std::cout << "(" << item_op->opmagic << ", " << item_op->GetOpcodeStr() << ") ";
            }
        }
        std::cout << std::endl << "output operation:";
        for (const std::shared_ptr<LogicalTensor> &output_tensor : op.GetOOperands()) {
            for (const auto &item_op : output_tensor->GetConsumers()) {
                std::cout << "(" << item_op->opmagic << ", " << item_op->GetOpcodeStr() << ") ";
            }
        }
        std::cout << std::endl;
    }
}

void SetUpPassStrategy() {
    PassManager &passManager = PassManager::Instance();
    passManager.RegisterStrategy(
        "RemoveRedundantOpTestStrategy", {
                                             {"RemoveRedundantReshape",  PassName::REMOVE_REDUNDANT_RESHAPE},
                                             {   "InferMemoryConflict",     PassName::INFER_MEMORY_CONFLICT},
                                             {        "ExpandFunction",           PassName::EXPAND_FUNCTION},
                                             {           "DuplicateOp",              PassName::DUPLICATE_OP},
                                             {     "MergeViewAssemble",       PassName::MERGE_VIEW_ASSEMBLE},
                                             {      "AssignMemoryType",        PassName::ASSIGN_MEMORY_TYPE},
                                             {"SplitLargeFanoutTensor", PassName::SPLIT_LARGE_FANOUT_TENSOR},
                                             {          "SplitReshape",             PassName::SPLIT_RESHAPE},
    });
}

class RemoveRedundantOpTest : public testing::Test {
public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}

    void SetUp() override {
        Program::GetInstance().Reset();
        config::Reset();
        config::SetHostOption(COMPILE_STAGE, HOST_COMPILE_END);
        config::SetHostConfig(KEY_STRATEGY, "RemoveRedundantOpTestStrategy");
        config::SetPlatformConfig(KEY_ENABLE_COST_MODEL, false);
    }
    void TearDown() override {}
};

TEST_F(RemoveRedundantOpTest, TestIntermediateOutcast) {
    config::SetHostOption(COMPILE_STAGE, HOST_COMPILE_END);
    int bs = 1;
    int n = 32;
    int d = 128;
    std::vector<int64_t> shape{bs, n, d};
    std::vector<int64_t> resShape{bs, n, d};
    SetUpPassStrategy();
    ConfigManager::Instance();

    Tensor input(DataType::DT_FP32, shape, "input");
    Tensor output(DataType::DT_FP32, resShape, "res");
    Tensor output_add(DataType::DT_FP32, resShape, "res_add");
    config::SetBuildStatic(true);
    FUNCTION("RemoveRedundantOpFunction", {input, output, output_add}) {
        TileShape::Current().SetVecTile(1, 32, 128);
        output = Transpose(input, {0, 1});
        TileShape::Current().SetVecTile(8, 1, 128);
        output_add = Add(output, Element(DataType::DT_FP32, 0.0));
    }

    Function* func = Program::GetInstance().GetFunctionByRawName("TENSOR_RemoveRedundantOpFunction");
    npu::tile_fwk::RemoveRedundantOp removeRedundantOp;
    auto oriOpList = func->Operations(true);
    EXPECT_EQ(oriOpList.size(), 15) << "Before the Pass, there should be 15 operations";
    int ori_view_count = 0;
    int ori_assemble_count = 0;
    for (auto &op : oriOpList) {
        if (op.GetOpcode() == Opcode::OP_VIEW) {
            ori_view_count += 1;
        } else if (op.GetOpcode() == Opcode::OP_ASSEMBLE) {
            ori_assemble_count += 1;
        }
    }
    EXPECT_EQ(ori_view_count, 5) << "There shoule be 5 VIEW op before RemoveRedundantOp";
    EXPECT_EQ(ori_assemble_count, 5) << "There shoule be 5 ASSEMBLE op before RemoveRedundantOp";
    removeRedundantOp.PreCheck(*func);
    removeRedundantOp.RunOnFunction(*func);
    removeRedundantOp.PostCheck(*func);
    PrintGraphInfoRemoveRedundantOp(func);
    // ================== Verify the effect of the Pass ==================
    auto updated_operations = func->Operations(true);
    int opSize = 14;
    EXPECT_EQ(updated_operations.size(), opSize) << "After the Pass, there should be 14 operations, no VIEW be deleted";
    EXPECT_EQ(updated_operations[0].GetOpcode(), Opcode::OP_VIEW) << "The first operation should be VIEW";
    int view_count = 0;
    int assemble_count = 0;

    for (auto &op : updated_operations) {
        if (op.GetOpcode() == Opcode::OP_VIEW) {
            view_count += 1;
        } else if (op.GetOpcode() == Opcode::OP_ASSEMBLE) {
            assemble_count += 1;
        }
    }
    EXPECT_EQ(view_count, 5) << "There shoule be 5 ASSEMBLE op after RemoveRedundantOp";
    EXPECT_EQ(assemble_count, 4) << "There shoule be 5 ASSEMBLE op after RemoveRedundantOp";
}

TEST_F(RemoveRedundantOpTest, TestInternalAssembleView) {
    config::SetHostOption(COMPILE_STAGE, HOST_COMPILE_END);
    int bs = 4;
    int n = 32;
    int d = 128;
    std::vector<int64_t> shape{bs, n, d};
    std::vector<int64_t> resShape{bs, n, d};
    SetUpPassStrategy();
    ConfigManager::Instance();

    Tensor input(DataType::DT_FP32, shape, "input");
    Tensor output(DataType::DT_FP32, resShape, "res");
    config::SetBuildStatic(true);
    FUNCTION("RemoveRedundantOpFunction", {input, output}) {
        TileShape::Current().SetVecTile(1, 32, 128);
        auto tmp = Transpose(input, {0, 1}); // [32, 4, 128]
        TileShape::Current().SetVecTile(8, 1, 64);
        output = Add(tmp, Element(DataType::DT_FP32, 3.0));
    }

    Function* func = Program::GetInstance().GetFunctionByRawName("TENSOR_RemoveRedundantOpFunction");
    npu::tile_fwk::RemoveRedundantOp removeRedundantOp;
    auto oriOpList = func->Operations(true);
    int ori_view_count = 0;
    int ori_assemble_count = 0;
    for (auto &op : oriOpList) {
        if (op.GetOpcode() == Opcode::OP_VIEW) {
            ori_view_count += 1;
        } else if (op.GetOpcode() == Opcode::OP_ASSEMBLE) {
            ori_assemble_count += 1;
        }
    }
    removeRedundantOp.PreCheck(*func);
    removeRedundantOp.RunOnFunction(*func);
    removeRedundantOp.PostCheck(*func);
    PrintGraphInfoRemoveRedundantOp(func);
    // ================== Verify the effect of the Pass ==================
    auto updated_operations = func->Operations(true);
    int view_count = 0;
    int assemble_count = 0;

    for (auto &op : updated_operations) {
        if (op.GetOpcode() == Opcode::OP_VIEW) {
            view_count += 1;
        } else if (op.GetOpcode() == Opcode::OP_ASSEMBLE) {
            assemble_count += 1;
        }
    }
    EXPECT_EQ(updated_operations.size(), oriOpList.size()) << "No op should be removed in RemoveRedundantOp";
    EXPECT_EQ(view_count, ori_view_count) << "No VIEW op should be removed in RemoveRedundantOp";
    EXPECT_EQ(assemble_count, ori_assemble_count) << "No ASSEMBLE op should be removed in RemoveRedundantOp";
}
