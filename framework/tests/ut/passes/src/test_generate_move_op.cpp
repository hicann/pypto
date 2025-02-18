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
 * \file test_generate_move_op.cpp
 * \brief Unit test for Generate Move Op pass.
 */

#include <gtest/gtest.h>
#include "interface/function/function.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "passes/pass_mgr/pass_manager.h"
#include "ut_json/ut_json_tool.h"
#include "passes/tile_graph_pass/data_path/generate_move_op.h"
#include "passes/tile_graph_pass/data_path/convert_op_inserter.h"
#include "interface/configs/config_manager.h"
#include <fstream>
#include <vector>
#include <string>

using namespace npu::tile_fwk;

namespace npu{
namespace tile_fwk {
const int NUM_32 = 32;
const int NUM_64 = 64;
const int NUM_128 = 128;
constexpr float F_3 = 3.0;

class GenerateMoveOpPassTest : public ::testing::Test {
public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}

    void SetUp() override {
        Program::GetInstance().Reset();
        config::Reset();
        config::SetPlatformConfig(KEY_ONLY_HOST_COMPILE, true);
        config::SetHostConfig(KEY_STRATEGY, "GenerateMoveOpPassTestStrategy");
        config::SetPlatformConfig("ENABLE_COST_MODEL", false);
    }
    void TearDown() override {}
};

TEST_F(GenerateMoveOpPassTest, AssembleViewToCopy) {
    PROGRAM("GenerateMoveOpPassTest") {
        std::vector<int64_t> shape1{256, 256};
        std::vector<int64_t> shape2{128, 128};
        TileShape::Current().SetVecTile({128, 128});
        Tensor input_a(DT_FP32, shape1, "input_a");
        Tensor input_b(DT_FP32, shape1, "input_b");
        Tensor output(DT_FP32, shape2, "output");
        PassManager &passManager = PassManager::Instance();
        passManager.RegisterStrategy("GenerateMoveOpPassTestStrategy", {
            {   "RemoveRedundantReshape",   "RemoveRedundantReshape"},
            {      "InferMemoryConflict",      "InferMemoryConflict"},
            {           "ExpandFunction",           "ExpandFunction"},
            {              "DuplicateOp",              "DuplicateOp"},
            {        "MergeViewAssemble",        "MergeViewAssemble"},
            {         "AssignMemoryType",         "AssignMemoryType"},
            {   "SplitLargeFanoutTensor",   "SplitLargeFanoutTensor"},
            {             "SplitReshape",             "SplitReshape"},
            {        "RemoveRedundantOp",        "RemoveRedundantOp"},

        });
        ConfigManager::Instance();

        Function* originFunction = nullptr;
        std::vector<int> originOpmagic;
        config::SetBuildStatic(true);
        FUNCTION("ADD", {input_a, input_b, output}) {
            config::SetPassStrategy("GenerateMoveOpPassTestStrategy");

            auto tmp_a_0 = View(input_a, shape2, {0,0});
            auto tmp_b_1 = View(input_b, shape2, {0,0});

            output = Add(tmp_a_0, tmp_b_1);
        }

        std::string jsonFilePath = "./config/pass/json/generate_move_op_assemble_view_to_copy.json";
        bool dumpJsonFlag = true;
        if (dumpJsonFlag) {
            auto programJson = Program::GetInstance().DumpJson();
            DumpJsonFile(programJson, jsonFilePath);
        }
        Json readData = LoadJsonFile(jsonFilePath);
        Program::GetInstance().LoadJson(readData);

        originFunction = Program::GetInstance().GetFunctionByRawName("TENSOR_ADD");

        ASSERT_NE(originFunction, nullptr) << "当前函数指针为空";
        auto operations = originFunction->Operations();
        for (const auto &op : operations) {
            std::cout << "opmagic: " << op.opmagic << "op type " << op.GetOpcodeStr() << std::endl;
            originOpmagic.emplace_back(op.opmagic);
        }
        GenerateMoveOp generateMoveOp;
        generateMoveOp.RunOnFunction(*originFunction);

        // ================== Verify Pass Effect ==================
        auto updatedOperations = Program::GetInstance().GetFunctionByRawName("TENSOR_ADD")->Operations();
        constexpr int expectedOperations = 4;
        EXPECT_EQ(updatedOperations.size(), expectedOperations) << "4 operations should remain View + Convert + Add + Assemble";
        int assemble_num = 0;
        int view_num = 0;
        int copy_in_num = 0;
        int copy_out_num = 0;
        for (const auto &updatedOperation : updatedOperations) {
            switch (updatedOperation.GetOpcode()){
                case Opcode::OP_ASSEMBLE: {
                    assemble_num++;
                    break;
                }
                case Opcode::OP_VIEW: {
                    view_num++;
                    break;
                }
                case Opcode::OP_COPY_IN: {
                    copy_in_num++;
                    break;
                }
                case Opcode::OP_COPY_OUT: {
                    copy_out_num++;
                    break;
                }
                default: break;
            }
        }
        constexpr int expectedAssemble = 0;
        constexpr int expectedView = 0;
        constexpr int expectedCopyIn = 2;
        constexpr int expectedCopyOut = 1;
        EXPECT_EQ(assemble_num, expectedAssemble) << "0 operations should be OP_ASSEMBLE";
        EXPECT_EQ(view_num, expectedView) << "0 operations should be OP_VIEW";
        EXPECT_EQ(copy_in_num, expectedCopyIn) << "2 operations should be OP_COPY_IN";
        EXPECT_EQ(copy_out_num, expectedCopyOut) << "1 operations should be OP_COPY_OUT";
    }
}

TEST_F(GenerateMoveOpPassTest, ConvertToCopy) {
    PROGRAM("GenerateMoveOpPassTest") {
        std::vector<int64_t> shape1{256, 256};
        std::vector<int64_t> shape2{128, 128};
        TileShape::Current().SetVecTile({128, 128});
        Tensor input_a(DT_FP32, shape1, "input_a");
        Tensor input_b(DT_FP32, shape1, "input_b");
        Tensor output(DT_FP32, shape2, "output");
        PassManager &passManager = PassManager::Instance();
        passManager.RegisterStrategy("GenerateMoveOpPassTestStrategy", {
            {   "RemoveRedundantReshape",   "RemoveRedundantReshape"},
            {      "InferMemoryConflict",      "InferMemoryConflict"},
            {           "ExpandFunction",           "ExpandFunction"},
            {              "DuplicateOp",              "DuplicateOp"},
            {        "MergeViewAssemble",        "MergeViewAssemble"},
            {   "SplitLargeFanoutTensor",   "SplitLargeFanoutTensor"},
            {             "SplitReshape",             "SplitReshape"},
            {         "AssignMemoryType",         "AssignMemoryType"},
            {        "RemoveRedundantOp",        "RemoveRedundantOp"},
            {           "GenerateMoveOp",           "GenerateMoveOp"},
        });
        ConfigManager::Instance();

        std::vector<int> originOpmagic;
        config::SetBuildStatic(true);
        FUNCTION("ADD", {input_a, input_b, output}) {
            config::SetPassStrategy("GenerateMoveOpPassTestStrategy");

            auto tmp_a_0 = View(input_a, shape2, {0,0});
            tmp_a_0.GetStorage()->SetMemoryTypeBoth(MEM_L1, true);
            auto tmp_b_1 = View(input_b, shape2, {0,0});
            tmp_b_1.GetStorage()->SetMemoryTypeBoth(MEM_L1, true);

            output = Add(tmp_a_0, tmp_b_1);
        }

        // ================== Verify Pass Effect ==================
        auto updatedOperations = Program::GetInstance().GetFunctionByRawName("TENSOR_ADD")->Operations();
        constexpr int expectedOperations = 6;
        EXPECT_EQ(updatedOperations.size(), expectedOperations) << "8 operations should remain View + Convert + Add + Assemble";
        int assemble_num = 0;
        int view_num = 0;
        int copy_in_num = 0;
        int copy_out_num = 0;
        for (const auto &updatedOperation : updatedOperations) {
            switch (updatedOperation.GetOpcode()){
                case Opcode::OP_ASSEMBLE: {
                    assemble_num++;
                    break;
                }
                case Opcode::OP_VIEW: {
                    view_num++;
                    break;
                }
                case Opcode::OP_COPY_IN: {
                    copy_in_num++;
                    break;
                }
                case Opcode::OP_COPY_OUT: {
                    copy_out_num++;
                    break;
                }
                default: break;
            }
        }
        constexpr int expectedAssemble = 0;
        constexpr int expectedView = 2;
        constexpr int expectedCopyIn = 2;
        constexpr int expectedCopyOut = 1;
        EXPECT_EQ(assemble_num, expectedAssemble) << "0 operations should be OP_ASSEMBLE";
        EXPECT_EQ(view_num, expectedView) << "0 operations should be OP_VIEW";
        EXPECT_EQ(copy_in_num, expectedCopyIn) << "4 operations should be OP_COPY_IN";
        EXPECT_EQ(copy_out_num, expectedCopyOut) << "3 operations should be OP_COPY_OUT";
    }
}

TEST_F(GenerateMoveOpPassTest, Transpose) {
    PROGRAM("GenerateMoveOpPassTest") {
        std::vector<int64_t> shape{1, 32, 32, 2};
        Tensor a(DT_FP32, shape, "a");
        Tensor a_trans(DT_FP32, shape, "a_trans");

        constexpr int dim0 = 1, dim1 = 16, dim2 = 16, dim3 = 2;
        TileShape::Current().SetVecTile(dim0, dim1, dim2, dim3);

        PassManager &passManager = PassManager::Instance();
        passManager.RegisterStrategy("GenerateMoveOpPassTestStrategy", {
            {   "RemoveRedundantReshape",   "RemoveRedundantReshape"},
            {      "InferMemoryConflict",      "InferMemoryConflict"},
            {           "ExpandFunction",           "ExpandFunction"},
            {              "DuplicateOp",              "DuplicateOp"},
            {        "MergeViewAssemble",        "MergeViewAssemble"},
            {         "AssignMemoryType",         "AssignMemoryType"},
            {   "SplitLargeFanoutTensor",   "SplitLargeFanoutTensor"},
            {             "SplitReshape",             "SplitReshape"},
            {        "RemoveRedundantOp",        "RemoveRedundantOp"},

        });
        ConfigManager::Instance();

        FUNCTION("Tranpose") {
            a_trans = Transpose(a, {1, 2});
        }
        std::string jsonFilePath = "./config/pass/json/generate_move_op_transpose.json";

        bool dumpJsonFlag = true;
        if (dumpJsonFlag) {
            auto programJson = Program::GetInstance().DumpJson();
            DumpJsonFile(programJson, jsonFilePath);
        }
        Json readData = LoadJsonFile(jsonFilePath);
        Program::GetInstance().LoadJson(readData);

        Function* originFunction = Program::GetInstance().GetCurrentFunction();
        GenerateMoveOp generateMoveOp;
        generateMoveOp.RunOnFunction(*originFunction);

        // ================== Verify Pass Effect ==================
        auto updatedOperations = Program::GetInstance().GetFunctionByRawName("TENSOR_Tranpose")->Operations();
        constexpr int expectedOperations = 12;
        EXPECT_EQ(updatedOperations.size(), expectedOperations) << "total 12 operations";
        int assemble_num = 0;
        int view_num = 0;
        int copy_in_num = 0;
        int copy_out_num = 0;
        int transpose_datamove_num = 0;
        for (const auto &updatedOperation : updatedOperations) {
            switch (updatedOperation.GetOpcode()){
                case Opcode::OP_ASSEMBLE: {
                    assemble_num++;
                    break;
                }
                case Opcode::OP_VIEW: {
                    view_num++;
                    break;
                }
                case Opcode::OP_COPY_IN: {
                    copy_in_num++;
                    break;
                }
                case Opcode::OP_COPY_OUT: {
                    copy_out_num++;
                    break;
                }
                case Opcode::OP_TRANSPOSE_MOVEOUT: {
                    transpose_datamove_num++;
                    break;
                }
                default: break;
            }
        }
        constexpr int expectedAssemble = 4;
        constexpr int expectedView = 0;
        constexpr int expectedCopyIn = 4;
        constexpr int expectedCopyOut = 0;
        EXPECT_EQ(assemble_num, expectedAssemble) << "4 operations should be OP_ASSEMBLE";
        EXPECT_EQ(assemble_num, transpose_datamove_num) << "num of OP_ASSEMBLE and OP_TRANSPOSE_MOVEOUT should be equal";
        EXPECT_EQ(view_num, expectedView) << "0 operations should be OP_VIEW";
        EXPECT_EQ(copy_in_num, expectedCopyIn) << "4 operations should be OP_COPY_IN";
        EXPECT_EQ(copy_out_num, expectedCopyOut) << "0 operations should be OP_COPY_OUT";
    }
}

TEST_F(GenerateMoveOpPassTest, ScatterUpdate) {
    PROGRAM("GenerateMoveOpPassTest") {
        int row = 64, col = 32;
        TileShape::Current().SetVecTile(row, col);

        PassManager &passManager = PassManager::Instance();
        passManager.RegisterStrategy("GenerateMoveOpPassTestStrategy", {
            {   "RemoveRedundantReshape",   "RemoveRedundantReshape"},
            {      "InferMemoryConflict",      "InferMemoryConflict"},
            {           "ExpandFunction",           "ExpandFunction"},
            {              "DuplicateOp",              "DuplicateOp"},
            {        "MergeViewAssemble",        "MergeViewAssemble"},
            {         "AssignMemoryType",         "AssignMemoryType"},
            {   "SplitLargeFanoutTensor",   "SplitLargeFanoutTensor"},
            {             "SplitReshape",             "SplitReshape"},
            {        "RemoveRedundantOp",        "RemoveRedundantOp"},

        });
        ConfigManager::Instance();

        int b = 2, s = 64, numExpertsPerTok = 2, h = 128, minus_two = -2;
        Tensor output(DT_FP32, {b*s*numExpertsPerTok, h}, "output");
        Tensor idxs(DT_INT64, {1, b*s*numExpertsPerTok}, "idxs");
        Tensor key_states(DT_FP32, {b*s*numExpertsPerTok, h}, "key_states");
        FUNCTION("ScatterUpdate") {
            output = ScatterUpdate(output, {idxs}, key_states, minus_two);
        }
        std::string jsonFilePath = "./config/pass/json/generate_move_op_scatter_update.json";
        bool dumpJsonFlag = false;
        if (dumpJsonFlag) {
            auto programJson = Program::GetInstance().DumpJson();
            DumpJsonFile(programJson, jsonFilePath);
        }

        Function* originFunction = Program::GetInstance().GetFunctionByRawName("TENSOR_ScatterUpdate");
        GenerateMoveOp generateMoveOp;
        generateMoveOp.RunOnFunction(*originFunction);

        // ================== Verify Pass Effect ==================
        auto updatedOperations = Program::GetInstance().GetFunctionByRawName("TENSOR_ScatterUpdate")->Operations();
        constexpr int expectedOperations = 20;
        EXPECT_EQ(updatedOperations.size(), expectedOperations) << "total 16 operations";
        int assemble_num = 0;
        int view_num = 0;
        int copy_in_num = 0;
        int copy_out_num = 0;
        int index_outcast_num = 0;
        for (const auto &updatedOperation : updatedOperations) {
            switch (updatedOperation.GetOpcode()){
                case Opcode::OP_ASSEMBLE: {
                    assemble_num++;
                    break;
                }
                case Opcode::OP_VIEW: {
                    view_num++;
                    break;
                }
                case Opcode::OP_COPY_IN: {
                    copy_in_num++;
                    break;
                }
                case Opcode::OP_COPY_OUT: {
                    copy_out_num++;
                    break;
                }
                case Opcode::OP_INDEX_OUTCAST: {
                    index_outcast_num++;
                    break;
                }
                default: break;
            }
        }
        constexpr int expectedAssemble = 4;
        constexpr int expectedView = 4;
        constexpr int expectedCopyIn = 8;
        constexpr int expectedCopyOut = 0;
        EXPECT_EQ(assemble_num, expectedAssemble) << "4 operations should be OP_ASSEMBLE";
        EXPECT_EQ(assemble_num, index_outcast_num) << "num of OP_ASSEMBLE and OP_INDEX_OUTCAST should be equal";
        EXPECT_EQ(view_num, expectedView) << "0 operations should be OP_VIEW";
        EXPECT_EQ(copy_in_num, expectedCopyIn) << "8 operations should be OP_COPY_IN";
        EXPECT_EQ(copy_out_num, expectedCopyOut) << "0 operations should be OP_COPY_OUT";
    }
}

TEST_F(GenerateMoveOpPassTest, L1TOL0){
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "L1TOL0", "L1TOL0", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    Program::GetInstance().InsertFuncToFunctionMap("L1TOL0", currFunctionPtr);
    constexpr int opMagic0 = 1001;
    constexpr int opMagic1 = 1002;
    constexpr int opMagic2 = 1003;

    constexpr int tensorMagic0 = 1;
    constexpr int tensorMagic1 = 2;
    constexpr int tensorMagic2 = 3;
    constexpr int tensorMagic3 = 4;
    constexpr int tensorMagic4 = 5;

    // Prepare the graph
    std::vector<int64_t> shape = {8, 16};
    std::vector<int64_t> shape1 = {16, 8};
    std::vector<int64_t> shape2 = {8, 8};
    std::shared_ptr<LogicalTensor> input_a = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    input_a->SetMagic(tensorMagic0);
    input_a->SetMemoryTypeOriginal(MemoryType::MEM_L1);

    std::shared_ptr<LogicalTensor> tmp_a = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    tmp_a->SetMagic(tensorMagic1);
    tmp_a->SetMemoryTypeOriginal(MemoryType::MEM_L0A);

    std::shared_ptr<LogicalTensor> input_b = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape1);
    input_b->SetMagic(tensorMagic2);
    input_b->SetMemoryTypeOriginal(MemoryType::MEM_L1);

    std::shared_ptr<LogicalTensor> tmp_b = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape1);
    tmp_b->SetMagic(tensorMagic3);
    tmp_b->SetMemoryTypeOriginal(MemoryType::MEM_L0B);

    std::shared_ptr<LogicalTensor> output_c = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape2);
    output_c->SetMagic(tensorMagic4);

    auto &convert_op1 = currFunctionPtr->AddRawOperation(Opcode::OP_CONVERT, {input_a}, {tmp_a});
    convert_op1.opmagic = opMagic0;
    convert_op1.SetOpAttribute(std::make_shared<ConvertOpAttribute>(MemoryType::MEM_L1,MemoryType::MEM_L0A));

    auto &convert_op2 = currFunctionPtr->AddRawOperation(Opcode::OP_CONVERT, {input_b}, {tmp_b});
    convert_op2.opmagic = opMagic1;
    convert_op2.SetOpAttribute(std::make_shared<ConvertOpAttribute>(MemoryType::MEM_L1,MemoryType::MEM_L0B));

    auto &matmul_op = currFunctionPtr->AddRawOperation(Opcode::OP_A_MUL_B, {tmp_a,tmp_b}, {output_c});
    matmul_op.opmagic = opMagic2;

    currFunctionPtr->inCasts_.push_back(input_a);
    currFunctionPtr->inCasts_.push_back(input_b);
    currFunctionPtr->outCasts_.push_back(output_c);

    std::stringstream ssBefore;
    ssBefore << "Before_GenerateMoveOp";

    // Call the pass
    ConvertInserter inserter;
    inserter.CreateMoveOpForConvert(convert_op1);
    inserter.CreateMoveOpForConvert(convert_op2);
    GenerateMoveOp generateMoveOp;
    generateMoveOp.PreCheck(*currFunctionPtr);
    generateMoveOp.RunOnFunction(*currFunctionPtr);
    generateMoveOp.PostCheck(*currFunctionPtr);

    std::stringstream ss;
    ss << "After_GenerateMoveOp";

    // Validate the results
    std::cout << "========== op size: " << currFunctionPtr->Operations().size() << std::endl;
    int convert_num = 0;
    int l1tol0a_num = 0;
    int l1tol0B_num = 0;
    for (auto &op : currFunctionPtr->Operations()) {
        std::cout << op.GetOpcodeStr() << " " << op.GetOpMagic() << std::endl;
        for (auto &input : op.GetIOperands()) {
            std::cout << "\t|--- iOperand " << input->magic;
        }
        for (auto &output : op.GetOOperands()) {
            std::cout << "\t|--- oOperand " << output->magic << std::endl;
        }
        if(op.GetOpcode()==Opcode::OP_CONVERT){
            convert_num++;
        }else if(op.GetOpcode()==Opcode::OP_L1_TO_L0A){
            l1tol0a_num++;
        }else if(op.GetOpcode()==Opcode::OP_L1_TO_L0B){
            l1tol0B_num++;
        }
    }
    constexpr int expectedConvert =0;
    constexpr int expectedL1tol0a =1;
    constexpr int expectedL1tol0b =1;
    EXPECT_EQ(convert_num,expectedConvert) << "0 operations shoulde be OP_VIEW.";
    EXPECT_EQ(l1tol0a_num,expectedL1tol0a) << "1 operations shoulde be OP_COPY_IN.";
    EXPECT_EQ(l1tol0B_num,expectedL1tol0b) << "1 operations shoulde be OP_COPY_OUT.";
}
void TransViewTensorWithAttr (std::shared_ptr<Function> &currFunctionPtr) {
    std::shared_ptr<LogicalTensor> view_in1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, std::vector<int64_t>{32,64});
    view_in1 -> SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR);
    std::shared_ptr<LogicalTensor> tensor1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, std::vector<int64_t>{32,64});
    tensor1 -> SetMemoryTypeOriginal(MemoryType::MEM_L1);
    std::shared_ptr<LogicalTensor> view_out1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, std::vector<int64_t>{32,64});
    view_out1 -> SetMemoryTypeOriginal(MemoryType::MEM_BT);

    std::shared_ptr<LogicalTensor> view_in2 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, std::vector<int64_t>{32,64});
    view_in2 -> SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR);
    std::shared_ptr<LogicalTensor> tensor2 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, std::vector<int64_t>{32,64});
    tensor2 -> SetMemoryTypeOriginal(MemoryType::MEM_L1);
    std::shared_ptr<LogicalTensor> view_out2 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, std::vector<int64_t>{32,64});
    view_out2 -> SetMemoryTypeOriginal(MemoryType::MEM_FIX_QUANT_PRE);

    std::shared_ptr<LogicalTensor> output = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, std::vector<int64_t>{32,64});
    output -> SetMemoryTypeOriginal(MemoryType::MEM_L0C);

    auto &view_op1 = currFunctionPtr->AddRawOperation(Opcode::OP_VIEW, {view_in1}, {tensor1});
    auto viewAttribute1 =std::make_shared<ViewOpAttribute>(std::vector<int64_t>{0,0});
    viewAttribute1->SetToType(MemoryType::MEM_L1);
    view_op1.SetOpAttribute(viewAttribute1);
    auto &view_op2 = currFunctionPtr->AddRawOperation(Opcode::OP_VIEW, {tensor1}, {view_out1});
    auto viewAttribute2 =std::make_shared<ViewOpAttribute>(std::vector<int64_t>{0,0});
    viewAttribute2->SetToType(MemoryType::MEM_BT);
    view_op2.SetOpAttribute(viewAttribute2);
    auto &view_op3 = currFunctionPtr->AddRawOperation(Opcode::OP_VIEW, {view_in2}, {tensor2});
    auto viewAttribute3 =std::make_shared<ViewOpAttribute>(std::vector<int64_t>{0,0});
    viewAttribute3->SetToType(MemoryType::MEM_L1);
    view_op3.SetOpAttribute(viewAttribute3);
    auto &view_op4 = currFunctionPtr->AddRawOperation(Opcode::OP_VIEW, {tensor2}, {view_out2});
    auto viewAttribute4 =std::make_shared<ViewOpAttribute>(std::vector<int64_t>{0,0});
    viewAttribute4->SetToType(MemoryType::MEM_FIX_QUANT_PRE);
    view_op4.SetOpAttribute(viewAttribute4);

    currFunctionPtr->AddRawOperation(Opcode::OP_A_MUL_B, {view_out1,view_out2}, {output});

    currFunctionPtr->inCasts_.push_back(view_in1);
    currFunctionPtr->inCasts_.push_back(view_in2);
    currFunctionPtr->outCasts_.push_back(output);
}
TEST_F(GenerateMoveOpPassTest, TransViewWithAttr) {
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TransViewWithAttr", "TransViewWithAttr", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);
    Program::GetInstance().InsertFuncToFunctionMap("TransViewWithAttr", currFunctionPtr);

    TransViewTensorWithAttr(currFunctionPtr);

    std::stringstream ssBefore;
    ssBefore << "Before_GenerateMoveOp";

    // Call the pass
    GenerateMoveOp generateMoveOp;
    generateMoveOp.PreCheck(*currFunctionPtr);
    currFunctionPtr->DumpJsonFile("./config/pass/json/generateMoveOp_TransViewWithAttr_before.json");
    generateMoveOp.RunOnFunction(*currFunctionPtr);
    currFunctionPtr->DumpJsonFile("./config/pass/json/generateMoveOp_TransViewWithAttr_after.json");

    std::stringstream ss;
    ss << "After_GenerateMoveOp";

    // Validate the results
    int copyIn_count_after_pass = 0;
    int l12Bt_count_after_pass = 0;
    int l12Fb_count_after_pass = 0;
    for (auto &op : currFunctionPtr->Operations()) {
        if (op.GetOpcode() == Opcode::OP_COPY_IN) {
           copyIn_count_after_pass++;
        }
        if (op.GetOpcode() == Opcode::OP_L1_TO_BT) {
           l12Bt_count_after_pass++;
        }
        if (op.GetOpcode() == Opcode::OP_L1_TO_FIX_QUANT_PRE) {
           l12Fb_count_after_pass++;
        }
    }
    constexpr int expectedCopyIn =2;
    constexpr int expectedL1toBt =1;
    constexpr int expectedL1toFb =1;
    EXPECT_EQ(copyIn_count_after_pass,expectedCopyIn) << "2 operations shoulde be OP_COPY_IN.";
    EXPECT_EQ(l12Bt_count_after_pass,expectedL1toBt) << "1 operations shoulde be OP_L1_TO_BT.";
    EXPECT_EQ(l12Fb_count_after_pass,expectedL1toFb) << "1 operations shoulde be OP_L1_TO_FIX_QUANT_PRE.";
}
void ViewconnectAssemble (std::shared_ptr<Function> &currFunctionPtr) {
    std::shared_ptr<LogicalTensor> input = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, std::vector<int64_t>{32,64});
    input -> SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR);
    std::shared_ptr<LogicalTensor> tensor1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, std::vector<int64_t>{32,64});
    tensor1 -> SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR);
    std::shared_ptr<LogicalTensor> output = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, std::vector<int64_t>{32,64});
    output -> SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR);

    auto &view_op = currFunctionPtr->AddRawOperation(Opcode::OP_VIEW, {input}, {tensor1});
    auto viewAttribute =std::make_shared<ViewOpAttribute>(std::vector<int64_t>{0,0});
    viewAttribute->SetToType(MemoryType::MEM_DEVICE_DDR);
    view_op.SetOpAttribute(viewAttribute);
    auto &assemble_op = currFunctionPtr->AddRawOperation(Opcode::OP_ASSEMBLE, {tensor1}, {output});
    auto assembleAttribute =std::make_shared<AssembleOpAttribute>(std::vector<int64_t>{0,0});
    assemble_op.SetOpAttribute(assembleAttribute);

    currFunctionPtr->inCasts_.push_back(input);
    currFunctionPtr->outCasts_.push_back(output);
}
TEST_F(GenerateMoveOpPassTest, ViewconnectAssemble) {
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "ViewconnectAssemble", "ViewconnectAssemble", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);
    Program::GetInstance().InsertFuncToFunctionMap("ViewconnectAssemble", currFunctionPtr);

    ViewconnectAssemble(currFunctionPtr);

    std::stringstream ssBefore;
    ssBefore << "Before_GenerateMoveOp";

    // Call the pass
    GenerateMoveOp generateMoveOp;
    generateMoveOp.PreCheck(*currFunctionPtr);
    currFunctionPtr->DumpJsonFile("./config/pass/json/generateMoveOp_ViewconnectAssemble_before.json");
    generateMoveOp.RunOnFunction(*currFunctionPtr);
    currFunctionPtr->DumpJsonFile("./config/pass/json/generateMoveOp_ViewconnectAssemble_after.json");

    std::stringstream ss;
    ss << "After_GenerateMoveOp";

    // Validate the results
    int check_Op_inputsMemType = 0;
    for (auto &op : currFunctionPtr->Operations()) {
        auto consumerOps = op.oOperand[0]->GetConsumers(); 
        for (auto childOp : consumerOps) {
            auto opcode = childOp->GetOpcode();
            const auto &inputsMemType = OpcodeManager::Inst().GetInputsMemType(opcode);
            if (inputsMemType.empty()) {
                check_Op_inputsMemType++;
            }
        }
    }
    constexpr int expectedcheck = 1;
    EXPECT_EQ(check_Op_inputsMemType,expectedcheck) << "1 operation inputsMemType shoulde be OP_COPY_IN.";
}
}
} // namespace npu::tile_fwk