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
 * \file test_assign_memory_type.cpp
 * \brief Unit test for assign_memory_type pass.
 */

#include <fstream>
#include <vector>
#include <gtest/gtest.h>
#include "interface/function/function.h"
#include "tilefwk/tilefwk.h"
#include "tilefwk/platform.h"
#include "interface/inner/tilefwk.h"
#include "interface/configs/config_manager.h"
#include "passes/tile_graph_pass/data_path/assign_memory_type.h"
#include "passes/pass_mgr/pass_manager.h"
#include "computational_graph_builder.h"

using namespace npu::tile_fwk;

namespace npu{
namespace tile_fwk {
const int NUM_32 = 32;
const int NUM_64 = 64;
const int NUM_128 = 128;
const int NUM_256 = 256;
constexpr float F_1 = 1.0;
constexpr float F_3 = 3.0;

class AssignMemoryTypeTest : public testing::Test {
public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}

    void SetUp() override {
        Program::GetInstance().Reset();
        config::Reset();
        config::SetHostOption(COMPILE_STAGE, CS_EXECUTE_GRAPH);
        config::SetPlatformConfig(KEY_ENABLE_COST_MODEL, false);
        config::SetPlatformConfig(KEY_TEST_IS_TIG, true);
        Platform::Instance().ObtainPlatformInfo();
    }
    void TearDown() override {}

    void SetHalfwayStrategy() {
        PassManager &passManager = PassManager::Instance();
        passManager.RegisterStrategy("AssignMemoryTypeTestStrategy", {
            {"RemoveRedundantReshape", PassName::REMOVE_REDUNDANT_RESHAPE},
            {   "InferMemoryConflict",    PassName::INFER_MEMORY_CONFLICT},
            {        "ExpandFunction",          PassName::EXPAND_FUNCTION},
            {           "DuplicateOp",             PassName::DUPLICATE_OP},
            {     "MergeViewAssemble",      PassName::MERGE_VIEW_ASSEMBLE},
        });
        ConfigManager::Instance();
    }

    void CheckConvertOp(const Operation &op, bool verbose = false) {
        /*
        1. 单输入单输出
        2. 输入/输出的tensor mem类型唯一
        3. 输入和输出的tensor的mem类型不同
        */
        EXPECT_EQ(op.GetIOperands().size(), 1) << "OP_CONVERT should have ONLY ONE input.";
        EXPECT_EQ(op.GetOOperands().size(), 1) << "OP_CONVERT should have ONLY ONE input.";
        auto input = op.GetIOperands().front();
        ASSERT_NE(input, nullptr) << "OP_CONVERT input is nullptr";
        auto output = op.GetOOperands().front();
        ASSERT_NE(output, nullptr) << "OP_CONVERT output is nullptr";
        auto inputMemOri = input->GetMemoryTypeOriginal();
        auto inputMemTobe = input->GetMemoryTypeToBe();
        auto outputMemOri = output->GetMemoryTypeOriginal();
        auto outputMemTobe = output->GetMemoryTypeToBe();
        if (verbose) {
            std::cout << "\t|--- iOperand " << input->magic;
            std::cout << ", mem ori: " << BriefMemoryTypeToString(inputMemOri);
            std::cout << ", tobe: " << BriefMemoryTypeToString(inputMemTobe) << std::endl;
            std::cout << "\t|--- oOperand " << output->magic;
            std::cout << ", mem ori: " << BriefMemoryTypeToString(outputMemOri);
            std::cout << ", tobe: " << BriefMemoryTypeToString(outputMemTobe) << std::endl;
        }
        EXPECT_EQ(inputMemOri, inputMemTobe) << "OP_CONVERT input Memory Ori should be the same as Memory Tobe.";
        EXPECT_EQ(outputMemOri, outputMemTobe) << "OP_CONVERT output Memory Ori should be the same as Memory Tobe.";
        EXPECT_NE(inputMemOri, outputMemOri) << "OP_CONVERT input should have different memory type from output.";
    }
};

TEST_F(AssignMemoryTypeTest, AddReshape) {
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TILE_AddReshape", "TILE_AddReshape", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    Program::GetInstance().InsertFuncToFunctionMap("TILE_AddReshape", currFunctionPtr);

    constexpr int opMagic0 = 1001;
    constexpr int opMagic1 = 1002;
    constexpr int opMagic2 = 1003;
    constexpr int opMagic3 = 1004;
    constexpr int opMagic4 = 1005;

    constexpr int tensorMagic0 = 1;
    constexpr int tensorMagic1 = 2;
    constexpr int tensorMagic2 = 3;
    constexpr int tensorMagic3 = 4;
    constexpr int tensorMagic4 = 5;
    constexpr int tensorMagic5 = 6;
    constexpr int tensorMagic6 = 7;
    // Prepare the graph
    std::vector<int64_t> shape = {16, 32};
    std::vector<int64_t> shape1 = {32,16};
    std::shared_ptr<LogicalTensor> input_tensor1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    input_tensor1->SetMemoryTypeBoth(MEM_DEVICE_DDR);
    input_tensor1->SetMagic(tensorMagic0);

    std::shared_ptr<LogicalTensor> input_tensor2 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    input_tensor2->SetMemoryTypeBoth(MEM_UNKNOWN);
    input_tensor2->SetMagic(tensorMagic1);

    std::shared_ptr<LogicalTensor> view_output1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    view_output1->SetMemoryTypeBoth(MEM_UNKNOWN);
    view_output1->SetMagic(tensorMagic6);

    std::shared_ptr<LogicalTensor> view_output2 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    view_output2->SetMemoryTypeBoth(MEM_UNKNOWN);
    view_output2->SetMagic(tensorMagic2);

    std::shared_ptr<LogicalTensor> add_output = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    add_output->SetMemoryTypeBoth(MEM_UNKNOWN);
    add_output->SetMagic(tensorMagic3);

    std::shared_ptr<LogicalTensor> reshape_output = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape1);
    reshape_output->SetMemoryTypeBoth(MEM_UNKNOWN);
    reshape_output->SetMagic(tensorMagic4);

    std::shared_ptr<LogicalTensor> assemble_output = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape1);
    assemble_output->SetMemoryTypeBoth(MEM_UNKNOWN);
    assemble_output->SetMagic(tensorMagic5);

    auto &view_op1 = currFunctionPtr->AddRawOperation(Opcode::OP_VIEW, {input_tensor1}, {view_output1});
    view_op1.SetOpAttribute(std::make_shared<ViewOpAttribute>(std::vector<int64_t>{0, 0}));
    view_op1.opmagic = opMagic0;

    auto &view_op2 = currFunctionPtr->AddRawOperation(Opcode::OP_VIEW, {input_tensor2}, {view_output2});
    view_op2.SetOpAttribute(std::make_shared<ViewOpAttribute>(std::vector<int64_t>{0, 0}));
    view_op2.opmagic = opMagic3;

    auto &add_op = currFunctionPtr->AddRawOperation(Opcode::OP_ADD, {view_output1, view_output2}, {add_output});
    add_op.opmagic = opMagic1;

    auto &reshape_op = currFunctionPtr->AddRawOperation(Opcode::OP_RESHAPE, {add_output}, {reshape_output});
    reshape_op.opmagic = opMagic4;

    auto &assemble_op = currFunctionPtr->AddRawOperation(Opcode::OP_ASSEMBLE, {reshape_output}, {assemble_output});
    assemble_op.SetOpAttribute(std::make_shared<AssembleOpAttribute>(std::vector<int64_t>{0, 0}));
    assemble_op.opmagic = opMagic2;

    currFunctionPtr->inCasts_.push_back(input_tensor1);
    currFunctionPtr->inCasts_.push_back(input_tensor2);
    currFunctionPtr->outCasts_.push_back(assemble_output);

    std::stringstream ssBefore;
    ssBefore << "Before_AssignMemoryType";

    // Call the pass
    AssignMemoryType assignMemoryType;
    assignMemoryType.PreCheck(*currFunctionPtr);
    assignMemoryType.RunOnFunction(*currFunctionPtr);
    assignMemoryType.PostCheck(*currFunctionPtr);

    std::stringstream ss;
    ss << "After_AssignMemoryType";

    // Validate the results, 所有op的输入输出memory类型唯一
    std::cout << "========== op size: " << currFunctionPtr->Operations().size() << std::endl;
    int convertNum = 0;
    for (auto &op : currFunctionPtr->Operations()) {
        std::cout << op.GetOpcodeStr() << " " << op.GetOpMagic() << std::endl;
        for (auto &input : op.GetIOperands()) {
            auto memOri = input->GetMemoryTypeOriginal();
            auto memTobe = input->GetMemoryTypeToBe();
            std::cout << "\t|--- iOperand " << input->magic;
            std::cout << ", mem ori: " << BriefMemoryTypeToString(memOri);
            std::cout << ", tobe: " << BriefMemoryTypeToString(memTobe) << std::endl;
            EXPECT_EQ(memOri, memTobe) << " input Memory Ori should be the same as Memory Tobe";
        }
        for (auto &output : op.GetOOperands()) {
            auto memOri = output->GetMemoryTypeOriginal();
            auto memTobe = output->GetMemoryTypeToBe();
            std::cout << "\t|--- oOperand " << output->magic;
            std::cout << ", mem ori: " << BriefMemoryTypeToString(memOri);
            std::cout << ", tobe: " << BriefMemoryTypeToString(memTobe) << std::endl;
            EXPECT_EQ(memOri, memTobe) << " output Memory Ori should be the same as Memory Tobe";
        }
        if (op.GetOpcode() == Opcode::OP_CONVERT || (op.GetOpcode() == Opcode::OP_ASSEMBLE && op.opmagic != opMagic2)) {
            convertNum++;
            CheckConvertOp(op);
        }
    }
    constexpr int expextedConvertNum = 1;
    EXPECT_EQ(convertNum, expextedConvertNum) << "ONLY ONE OP_CONVERT.";
}

TEST_F(AssignMemoryTypeTest, TestVecToCubeV2) {
    config::SetHostConfig(KEY_STRATEGY, "AssignMemoryTypeTestStrategy");
    std::vector<int64_t> shape0 = {256, 128};
    std::vector<int64_t> shape1 = {128, 64};
    std::vector<int64_t> shape2 = {256, 64};
    PROGRAM("AssignMemoryTest") {
        Tensor input1(DataType::DT_FP32, shape0, "A");
        Tensor input2(DataType::DT_FP32, shape0, "B");
        Tensor weight(DataType::DT_FP32, shape1, "weight");
        Tensor out(DataType::DT_FP32, shape2, "output");
        SetHalfwayStrategy();
        Function* originFunction = nullptr;

        config::SetBuildStatic(true);
        FUNCTION("TestVecToCubeV2", {input1, input2, weight, out}) {
            config::SetPassStrategy("AssignMemoryTypeTestStrategy");
            TileShape::Current().SetVecTile(NUM_128, NUM_128);
            Tensor addRes = Add(input1, input2); // 256 * 128
            TileShape::Current().SetCubeTile({NUM_32, NUM_32}, {NUM_128, NUM_128}, {NUM_64, NUM_64});
            Tensor mmRes = Matrix::Matmul(out.GetDataType(), addRes, weight); // (256 * 128) @ (128 * 64) = (256 * 64)
            TileShape::Current().SetVecTile(NUM_128, NUM_128);
            Tensor sumRes = Sum(addRes, 1, true);
            TileShape::Current().SetVecTile(NUM_64, NUM_64);
            out = Add(mmRes, sumRes);
        }

        originFunction = Program::GetInstance().GetFunctionByRawName("TENSOR_TestVecToCubeV2"); // Tensor_{Function名字}
        ASSERT_NE(originFunction, nullptr) << "当前函数指针为空";
        std::vector<int64_t> beforeMagic;
        for (const auto &op : originFunction->Operations()) {
            if (op.GetOpcode() == Opcode::OP_CONVERT || op.GetOpcode() == Opcode::OP_VIEW ||
                op.GetOpcode() == Opcode::OP_ASSEMBLE) {
                beforeMagic.push_back(op.opmagic);
            }
        }
        // Call the pass
        AssignMemoryType assignMemoryType;
        assignMemoryType.PreCheck(*originFunction);
        assignMemoryType.RunOnFunction(*originFunction);
        assignMemoryType.PostCheck(*originFunction);
        // ================== Verify Pass Effect ==================
        auto updatedOperations = originFunction->Operations();
        int convertNum = 0;
        for (const auto &op : updatedOperations) {
            if (op.GetOpcode() == Opcode::OP_CONVERT || op.GetOpcode() == Opcode::OP_VIEW ||
                op.GetOpcode() == Opcode::OP_ASSEMBLE) {
                if (std::find(beforeMagic.begin(), beforeMagic.end(), op.opmagic) != beforeMagic.end()) {
                    continue;
                }
                convertNum++;
                std::cout << op.GetOpcodeStr() << " " << op.GetOpMagic() << std::endl;
                CheckConvertOp(op, true);
            }
        }
        constexpr int expextedConvertNum = 12;
        EXPECT_EQ(convertNum, expextedConvertNum) << "12 operations should be Convert";
    }
}

TEST_F(AssignMemoryTypeTest, TestCubeToCube) {
    config::SetHostConfig(KEY_STRATEGY, "AssignMemoryTypeTestStrategy");
    std::vector<int64_t> shape0 = {256, 128};
    std::vector<int64_t> shape1 = {128, 64};
    std::vector<int64_t> shape2 = {256, 256};
    PROGRAM("AssignMemoryTest") {
        Tensor inputQ(DataType::DT_BF16, shape0, "Q");
        Tensor inputK(DataType::DT_BF16, shape0, "K");
        Tensor weight(DataType::DT_BF16, shape1, "weight");
        Tensor out(DataType::DT_FP32, shape2, "output");
        SetHalfwayStrategy();
        Function* originFunction = nullptr;

        config::SetBuildStatic(true);
        FUNCTION("TestCubeToCube", {inputQ, inputK, weight, out}) {
            TileShape::Current().SetCubeTile({NUM_128, NUM_128}, {NUM_128, NUM_128}, {NUM_64, NUM_64});
            Tensor qUpdate = Matrix::Matmul(out.GetDataType(), inputQ, weight); // (256 * 128) @ (128 * 64) = (256 * 64)
            TileShape::Current().SetCubeTile({NUM_128, NUM_128}, {NUM_128, NUM_128}, {NUM_64, NUM_64});
            Tensor kUpdate = Matrix::Matmul(out.GetDataType(), inputK, weight); // (256 * 128) @ (128 * 64) = (256 * 64)
            TileShape::Current().SetCubeTile({NUM_128, NUM_128}, {NUM_64, NUM_64}, {NUM_128, NUM_128});
            Tensor QKT = Matrix::Matmul(out.GetDataType(), qUpdate, kUpdate, false, true); // (256 * 64) @ (64 * 256) = (256 * 256)
            TileShape::Current().SetVecTile(NUM_64, NUM_64);
            out = Sub(QKT, Element(DataType::DT_FP32, F_3));
        }

        originFunction = Program::GetInstance().GetFunctionByRawName("TENSOR_TestCubeToCube"); // Tensor_{Function名字}
        ASSERT_NE(originFunction, nullptr) << "当前函数指针为空";
        std::vector<int64_t> beforeMagic;
        for (const auto &op : originFunction->Operations()) {
            if (op.GetOpcode() == Opcode::OP_CONVERT || op.GetOpcode() == Opcode::OP_VIEW ||
                op.GetOpcode() == Opcode::OP_ASSEMBLE) {
                beforeMagic.push_back(op.opmagic);
            }
        }
        // Call the pass
        AssignMemoryType assignMemoryType;
        assignMemoryType.PreCheck(*originFunction);
        assignMemoryType.RunOnFunction(*originFunction);
        assignMemoryType.PostCheck(*originFunction);
        // ================== Verify Pass Effect ==================
        auto opList = originFunction->Operations();
        int convertNum = 0;
        for (const auto &op : opList) {
            if (std::find(beforeMagic.begin(), beforeMagic.end(), op.opmagic) != beforeMagic.end()) {
                continue;
            }
            if (op.GetOpcode() != Opcode::OP_CONVERT) {
                continue;
            }
            std::cout << op.GetOpcodeStr() << " " << op.GetOpMagic() << std::endl;
            CheckConvertOp(op, true);
            convertNum++;
        }
        constexpr int expextedConvertNum = 0;
        EXPECT_EQ(convertNum, expextedConvertNum) << "0 operations should be Convert";
    }
}

TEST_F(AssignMemoryTypeTest, TestCubeToCubeV2) {
    config::SetHostConfig(KEY_STRATEGY, "AssignMemoryTypeTestStrategy");
    std::vector<int64_t> shape0 = {256, 128};
    std::vector<int64_t> shape1 = {128, 64};
    std::vector<int64_t> shape2 = {256, 256};
    PROGRAM("AssignMemoryTest") {
        Tensor inputQ(DataType::DT_FP32, shape0, "Q");
        Tensor inputK(DataType::DT_FP32, shape0, "K");
        Tensor weight(DataType::DT_FP32, shape1, "weight");
        Tensor out(DataType::DT_FP32, shape2, "output");
        SetHalfwayStrategy();
        Function* originFunction = nullptr;

        config::SetBuildStatic(true);
        FUNCTION("TestCubeToCubeV2", {inputQ, inputK, weight, out}) {
            TileShape::Current().SetCubeTile({NUM_128, NUM_128}, {NUM_128, NUM_128}, {NUM_64, NUM_64});
            Tensor qUpdate = Matrix::Matmul(out.GetDataType(), inputQ, weight); // (256 * 128) @ (128 * 64) = (256 * 64)
            TileShape::Current().SetCubeTile({NUM_128, NUM_128}, {NUM_128, NUM_128}, {NUM_64, NUM_64});
            Tensor kUpdate = Matrix::Matmul(out.GetDataType(), inputK, weight); // (256 * 128) @ (128 * 64) = (256 * 64)
            TileShape::Current().SetCubeTile({NUM_128, NUM_128}, {NUM_64, NUM_64}, {NUM_128, NUM_128});
            Tensor QKT = Matrix::Matmul(out.GetDataType(), qUpdate, kUpdate, false, true); // (256 * 64) @ (64 * 256) = (256 * 256)
            TileShape::Current().SetVecTile(NUM_64, NUM_64);
            out = Add(QKT, Element(DataType::DT_FP32, F_1));
        }

        originFunction = Program::GetInstance().GetFunctionByRawName("TENSOR_TestCubeToCubeV2"); // Tensor_{Function名字}
        ASSERT_NE(originFunction, nullptr) << "当前函数指针为空";
        std::vector<int64_t> beforeMagic;
        for (const auto &op : originFunction->Operations()) {
            if (op.GetOpcode() == Opcode::OP_CONVERT || op.GetOpcode() == Opcode::OP_VIEW ||
                op.GetOpcode() == Opcode::OP_ASSEMBLE) {
                beforeMagic.push_back(op.opmagic);
            }
        }
        // Call the pass
        AssignMemoryType assignMemoryType;
        assignMemoryType.PreCheck(*originFunction);
        assignMemoryType.RunOnFunction(*originFunction);
        assignMemoryType.PostCheck(*originFunction);
        // ================== Verify Pass Effect ==================
        auto opList = originFunction->Operations();
        int convertNum = 0;
        for (const auto &op : opList) {
            if (std::find(beforeMagic.begin(), beforeMagic.end(), op.opmagic) != beforeMagic.end()) {
                continue;
            }
            if (op.GetOpcode() != Opcode::OP_CONVERT && op.GetOpcode() != Opcode::OP_VIEW &&
                op.GetOpcode() != Opcode::OP_ASSEMBLE) {
                continue;
            }
            std::cout << op.GetOpcodeStr() << " " << op.GetOpMagic() << std::endl;
            CheckConvertOp(op, true);
            convertNum++;
        }
        constexpr int expextedConvertNum = 32;
        EXPECT_EQ(convertNum, expextedConvertNum) << "32 operations should be Convert";
    }
}

TEST_F(AssignMemoryTypeTest, TestCubeToVec) {
    config::SetHostConfig(KEY_STRATEGY, "AssignMemoryTypeTestStrategy");
    std::vector<int64_t> shape0 = {NUM_256, NUM_128};
    std::vector<int64_t> shape1 = {NUM_128, NUM_64};
    std::vector<int64_t> shape2 = {NUM_256, NUM_64};
    PROGRAM("AssignMemoryTest") {
        Tensor inputA1(DataType::DT_FP32, shape0, "A1");
        Tensor inputB1(DataType::DT_FP32, shape1, "B1");
        Tensor inputA2(DataType::DT_FP32, shape0, "A2");
        Tensor inputB2(DataType::DT_FP32, shape1, "B2");
        Tensor inputV1(DataType::DT_FP32, shape0, "B2");
        Tensor inputV2(DataType::DT_FP32, shape0, "B2");
        Tensor out(DataType::DT_FP32, shape2, "output");
        SetHalfwayStrategy();
        Function* originFunction = nullptr;
        config::SetBuildStatic(true);
        FUNCTION("TestCubeToVec", {inputA1, inputB1, inputA2, inputB2, inputV1, inputV2, out}) {
            TileShape::Current().SetCubeTile({NUM_256, NUM_256}, {NUM_128, NUM_128}, {NUM_64, NUM_64});
            Tensor C1 = Matrix::Matmul(out.GetDataType(), inputA1, inputB1); // (256 * 128) @ (128 * 64) = (256 * 64)
            TileShape::Current().SetCubeTile({NUM_256, NUM_256}, {NUM_128, NUM_128}, {NUM_64, NUM_64});
            Tensor C2 = Matrix::Matmul(out.GetDataType(), inputA2, inputB2); // (256 * 128) @ (128 * 64) = (256 * 64)
            Assemble(C1, {0, 0}, inputV1);
            Assemble(C2, {0, NUM_64}, inputV1);
            TileShape::Current().SetVecTile(NUM_256, NUM_128);
            out = Add(inputV1, inputV2);
        }
        originFunction = Program::GetInstance().GetFunctionByRawName("TENSOR_TestCubeToVec"); // Tensor_{Function名字}
        ASSERT_NE(originFunction, nullptr) << "当前函数指针为空";
        int64_t beforeViewNum = 0;
        for (const auto &op : originFunction->Operations()) {
            if (op.GetOpcode() == Opcode::OP_VIEW) {
                ++beforeViewNum;
            }
        }
        // Call the pass
        AssignMemoryType assignMemoryType;
        assignMemoryType.PreCheck(*originFunction);
        assignMemoryType.RunOnFunction(*originFunction);
        assignMemoryType.PostCheck(*originFunction);
        // ================== Verify Pass Effect ==================
        int64_t afterViewNum = 0;
        for (const auto &op : originFunction->Operations()) {
            if (op.GetOpcode() == Opcode::OP_VIEW) {
                ++afterViewNum;
                auto viewOpAttr = std::dynamic_pointer_cast<ViewOpAttribute>(op.GetOpAttribute());
                EXPECT_TRUE(viewOpAttr->GetTo() == MemoryType::MEM_L1 || viewOpAttr->GetTo() == MemoryType::MEM_UB) << "View to either l1 or ub";
            }
        }
        EXPECT_EQ(afterViewNum, beforeViewNum + 1) << "Should insert one view after assemble and transfter data to DDR before to UB";
    }
}

void GetInvalidPatternGraph(std::shared_ptr<Function> &currFunctionPtr) {
    constexpr int opMagic0 = 1001;
    constexpr int opMagic1 = 1002;
    constexpr int opMagic2 = 1003;
    constexpr int opMagic3 = 1004;
    constexpr int opMagic4 = 1005;
    constexpr int opMagic5 = 1006;
    constexpr int opMagic6 = 1007;
    constexpr int opMagic7 = 1008;

    constexpr int tensorMagic0 = 1;
    constexpr int tensorMagic1 = 2;
    constexpr int tensorMagic2 = 3;
    constexpr int tensorMagic3 = 4;
    constexpr int tensorMagic4 = 5;
    constexpr int tensorMagic5 = 6;
    constexpr int tensorMagic6 = 7;
    constexpr int tensorMagic7 = 8;
    // Prepare the graph
    std::vector<int64_t> shape = {16, 32};
    std::vector<int64_t> shape1 = {32,16};
    std::vector<int64_t> shape2 = {8,32};
    std::vector<int64_t> shape3 = {32,8};
    std::shared_ptr<LogicalTensor> input_cast = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape1);
    input_cast->SetMagic(tensorMagic0);

    std::shared_ptr<LogicalTensor> input_tensor1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    input_tensor1->SetMagic(tensorMagic1);

    std::shared_ptr<LogicalTensor> view_output1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape2);
    view_output1->SetMagic(tensorMagic2);

    std::shared_ptr<LogicalTensor> view_output2 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape2);
    view_output2->SetMagic(tensorMagic3);

    std::shared_ptr<LogicalTensor> reshape_output1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape3);
    reshape_output1->SetMagic(tensorMagic4);

    std::shared_ptr<LogicalTensor> reshape_output2 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape3);
    reshape_output2->SetMagic(tensorMagic5);

    std::shared_ptr<LogicalTensor> assemble_output = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape1);
    assemble_output->SetMagic(tensorMagic6);

    std::shared_ptr<LogicalTensor> output_cast = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape1);
    output_cast->SetMagic(tensorMagic7);

    auto &reshape_op0 = currFunctionPtr->AddRawOperation(Opcode::OP_RESHAPE, {input_cast}, {input_tensor1});
    reshape_op0.opmagic = opMagic0;

    auto &view_op1 = currFunctionPtr->AddRawOperation(Opcode::OP_VIEW, {input_tensor1}, {view_output1});
    view_op1.SetOpAttribute(std::make_shared<ViewOpAttribute>(std::vector<int64_t>{0, 0}));
    view_op1.opmagic = opMagic1;

    auto &view_op2 = currFunctionPtr->AddRawOperation(Opcode::OP_VIEW, {input_tensor1}, {view_output2});
    view_op2.SetOpAttribute(std::make_shared<ViewOpAttribute>(std::vector<int64_t>{8, 0}));
    view_op2.opmagic = opMagic2;

    auto &reshape_op1 = currFunctionPtr->AddRawOperation(Opcode::OP_RESHAPE, {view_output1}, {reshape_output1});
    reshape_op1.opmagic = opMagic3;

    auto &reshape_op2 = currFunctionPtr->AddRawOperation(Opcode::OP_RESHAPE, {view_output2}, {reshape_output2});
    reshape_op2.opmagic = opMagic4;

    auto &assemble_op1 = currFunctionPtr->AddRawOperation(Opcode::OP_ASSEMBLE, {reshape_output1}, {assemble_output});
    assemble_op1.SetOpAttribute(std::make_shared<AssembleOpAttribute>(std::vector<int64_t>{0, 0}));
    assemble_op1.opmagic = opMagic5;

    auto &assemble_op2 = currFunctionPtr->AddRawOperation(Opcode::OP_ASSEMBLE, {reshape_output2}, {assemble_output});
    assemble_op2.SetOpAttribute(std::make_shared<AssembleOpAttribute>(std::vector<int64_t>{8, 0}));
    assemble_op2.opmagic = opMagic6;

    auto &view_op3 = currFunctionPtr->AddRawOperation(Opcode::OP_VIEW, {assemble_output}, {output_cast});
    view_op3.SetOpAttribute(std::make_shared<ViewOpAttribute>(std::vector<int64_t>{0, 0}));
    view_op3.opmagic = opMagic7;

    currFunctionPtr->inCasts_.push_back(input_cast);
    currFunctionPtr->outCasts_.push_back(output_cast);
}
TEST_F(AssignMemoryTypeTest, InValidOpPattern) {
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "InValidOpPattern", "InValidOpPattern", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    Program::GetInstance().InsertFuncToFunctionMap("InValidOpPattern", currFunctionPtr);

    GetInvalidPatternGraph(currFunctionPtr);

    std::stringstream ssBefore;
    ssBefore << "Before_AssignMemoryType";

    // Call the pass
    AssignMemoryType assignMemoryType;
    assignMemoryType.PreCheck(*currFunctionPtr);
    assignMemoryType.RunOnFunction(*currFunctionPtr);
    assignMemoryType.PostCheck(*currFunctionPtr);

    std::stringstream ss;
    ss << "After_AssignMemoryType";

    std::string josnFilePath = "/config/pass/json/assign_mem_type_invalidpattern.json";
    currFunctionPtr->DumpJsonFile(josnFilePath);

    // Validate the results
    std::cout << "========== op size: " << currFunctionPtr->Operations().size() << std::endl;
    for (auto &op : currFunctionPtr->Operations()) {
        std::cout << op.GetOpcodeStr() << " " << op.GetOpMagic() << std::endl;
        for (auto &input : op.GetIOperands()) {
            std::cout << "\t|--- iOperand " << input->magic;
            EXPECT_EQ(input->GetMemoryTypeOriginal(), MemoryType::MEM_DEVICE_DDR) << " Unexpected memory type.";
            EXPECT_EQ(input->GetMemoryTypeOriginal(), input->GetMemoryTypeToBe()) << " iOperand has two memory type.";
        }
        for (auto &output : op.GetOOperands()) {
            std::cout << "\t|--- oOperand " << output->magic << std::endl;
            EXPECT_EQ(output->GetMemoryTypeOriginal(), MemoryType::MEM_DEVICE_DDR) << " Unexpected memory type.";
            EXPECT_EQ(output->GetMemoryTypeOriginal(), output->GetMemoryTypeToBe()) << " oOperand has two memory type.";
        }
    }
}

void GetViewReshapeGraph (std::shared_ptr<Function> &currFunctionPtr) {
    constexpr int opMagic0 = 1001;
    constexpr int opMagic1 = 1002;
    constexpr int opMagic2 = 1003;
    constexpr int opMagic3 = 1004;
    constexpr int opMagic4 = 1005;

    constexpr int tensorMagic0 = 1;
    constexpr int tensorMagic1 = 2;
    constexpr int tensorMagic2 = 3;
    constexpr int tensorMagic3 = 4;
    constexpr int tensorMagic4 = 5;
    constexpr int tensorMagic5 = 6;

    // Prepare the graph
    std::vector<int64_t> shape = {16, 32};
    std::vector<int64_t> shape1 = {32, 16};
    std::vector<int64_t> shape2 = {1, 32};
    std::vector<int64_t> shape3 = {8, 32};
    std::shared_ptr<LogicalTensor> input_cast = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    input_cast->SetMagic(tensorMagic0);

    std::shared_ptr<LogicalTensor> transpose_out = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape1);
    transpose_out->SetMagic(tensorMagic1);

    std::shared_ptr<LogicalTensor> view_output1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape1);
    view_output1->SetMagic(tensorMagic2);

    std::shared_ptr<LogicalTensor> reshape_output = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    reshape_output->SetMagic(tensorMagic3);

    std::shared_ptr<LogicalTensor> view_output2 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape2);
    view_output2->SetMagic(tensorMagic4);

    std::shared_ptr<LogicalTensor> output_cast = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape3);
    output_cast->SetMagic(tensorMagic5);

    auto &transpose_op = currFunctionPtr->AddRawOperation(Opcode::OP_TRANSPOSE_VNCHWCONV, {input_cast}, {transpose_out});
    transpose_op.opmagic = opMagic0;

    auto &view_op1 = currFunctionPtr->AddRawOperation(Opcode::OP_VIEW, {transpose_out}, {view_output1});
    view_op1.SetOpAttribute(std::make_shared<ViewOpAttribute>(std::vector<int64_t>{0, 0}));
    view_op1.opmagic = opMagic1;

    auto &reshape_op = currFunctionPtr->AddRawOperation(Opcode::OP_RESHAPE, {view_output1}, {reshape_output});
    reshape_op.opmagic = opMagic2;

    auto &view_op2 = currFunctionPtr->AddRawOperation(Opcode::OP_VIEW, {reshape_output}, {view_output2});
    view_op2.SetOpAttribute(std::make_shared<ViewOpAttribute>(std::vector<int64_t>{0, 0}));
    view_op2.opmagic = opMagic3;

    auto &expand_op = currFunctionPtr->AddRawOperation(Opcode::OP_EXPAND, {view_output2}, {output_cast});
    expand_op.opmagic = opMagic4;
}
TEST_F(AssignMemoryTypeTest, ViewReshape) {
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "ViewReshape", "ViewReshape", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    Program::GetInstance().InsertFuncToFunctionMap("ViewReshape", currFunctionPtr);

    GetViewReshapeGraph(currFunctionPtr);

    std::stringstream ssBefore;
    ssBefore << "Before_AssignMemoryType";

    // Call the pass
    AssignMemoryType assignMemoryType;
    assignMemoryType.PreCheck(*currFunctionPtr);
    assignMemoryType.RunOnFunction(*currFunctionPtr);
    assignMemoryType.PostCheck(*currFunctionPtr);

    std::stringstream ss;
    ss << "After_AssignMemoryType";

    std::string josnFilePath = "/config/pass/json/assign_mem_type_invalidpattern.json";
    currFunctionPtr->DumpJsonFile(josnFilePath);

    // Validate the results
    std::cout << "========== op size: " << currFunctionPtr->Operations().size() << std::endl;
    for (auto &op : currFunctionPtr->Operations()) {
        std::cout << op.GetOpcodeStr() << " " << op.GetOpMagic() << std::endl;
        for (auto &input : op.GetIOperands()) {
            std::cout << "\t|--- iOperand " << input->magic;
            EXPECT_EQ(input->GetMemoryTypeOriginal(), MemoryType::MEM_UB) << " Unexpected memory type.";
            EXPECT_EQ(input->GetMemoryTypeOriginal(), input->GetMemoryTypeToBe()) << " iOperand has two memory type.";
        }
        for (auto &output : op.GetOOperands()) {
            std::cout << "\t|--- oOperand " << output->magic << std::endl;
            EXPECT_EQ(output->GetMemoryTypeOriginal(), MemoryType::MEM_UB) << " Unexpected memory type.";
            EXPECT_EQ(output->GetMemoryTypeOriginal(), output->GetMemoryTypeToBe()) << " oOperand has two memory type.";
        }
    }
}
void L1DataMoveGraph (std::shared_ptr<Function> &currFunctionPtr) {
    std::shared_ptr<LogicalTensor> input_cast1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, std::vector<int64_t>{32,64});
    std::shared_ptr<LogicalTensor> input_cast2 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, std::vector<int64_t>{64,16});
    std::shared_ptr<LogicalTensor> op_view_L1_out1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, std::vector<int64_t>{32,64});
    std::shared_ptr<LogicalTensor> op_view_L1_out2 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, std::vector<int64_t>{64,16});
    std::shared_ptr<LogicalTensor> view_out1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, std::vector<int64_t>{32,32});
    std::shared_ptr<LogicalTensor> view_out2 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, std::vector<int64_t>{32,32});
    std::shared_ptr<LogicalTensor> view_out3 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, std::vector<int64_t>{32,16});
    std::shared_ptr<LogicalTensor> view_out4 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, std::vector<int64_t>{32,16});
    std::shared_ptr<LogicalTensor> l0a_out1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, std::vector<int64_t>{32,32});
    std::shared_ptr<LogicalTensor> l0a_out2 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, std::vector<int64_t>{32,32});
    std::shared_ptr<LogicalTensor> l0b_out1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, std::vector<int64_t>{32,16});
    std::shared_ptr<LogicalTensor> l0b_out2 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, std::vector<int64_t>{32,16});
    std::shared_ptr<LogicalTensor> a_mul_b_out1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, std::vector<int64_t>{32,16});
    std::shared_ptr<LogicalTensor> a_mul_b_out2 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, std::vector<int64_t>{32,16});
    // std::shared_ptr<LogicalTensor> output_cast = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    auto &view_L1_op1 = currFunctionPtr->AddRawOperation(Opcode::OP_VIEW, {input_cast1}, {op_view_L1_out1});
    std::vector<int> newoffset{0,0};
    auto viewAttribute =std::make_shared<ViewOpAttribute>(std::vector<int64_t>{0,0});
    viewAttribute->SetToType(MemoryType::MEM_L1);
    view_L1_op1.SetOpAttribute(viewAttribute);

    auto &view_L1_op2 = currFunctionPtr->AddRawOperation(Opcode::OP_VIEW, {input_cast2}, {op_view_L1_out2});
    view_L1_op2.SetOpAttribute(viewAttribute);

    auto &view_op1 = currFunctionPtr->AddRawOperation(Opcode::OP_VIEW, {op_view_L1_out1}, {view_out1});
    view_op1.SetOpAttribute(std::make_shared<ViewOpAttribute>(std::vector<int64_t>{0, 0}));
    auto &view_op2 = currFunctionPtr->AddRawOperation(Opcode::OP_VIEW, {op_view_L1_out1}, {view_out2});
    view_op2.SetOpAttribute(std::make_shared<ViewOpAttribute>(std::vector<int64_t>{0, 32}));
    auto &view_op3 = currFunctionPtr->AddRawOperation(Opcode::OP_VIEW, {op_view_L1_out2}, {view_out3});
    view_op3.SetOpAttribute(std::make_shared<ViewOpAttribute>(std::vector<int64_t>{0, 0}));
    auto &view_op4 = currFunctionPtr->AddRawOperation(Opcode::OP_VIEW, {op_view_L1_out2}, {view_out4});
    view_op4.SetOpAttribute(std::make_shared<ViewOpAttribute>(std::vector<int64_t>{32, 0}));

    currFunctionPtr->AddRawOperation(Opcode::OP_L1_TO_L0A, {view_out1}, {l0a_out1});
    currFunctionPtr->AddRawOperation(Opcode::OP_L1_TO_L0A, {view_out2}, {l0a_out2});
    currFunctionPtr->AddRawOperation(Opcode::OP_L1_TO_L0B, {view_out3}, {l0b_out1});
    currFunctionPtr->AddRawOperation(Opcode::OP_L1_TO_L0B, {view_out4}, {l0b_out2});

    currFunctionPtr->AddRawOperation(Opcode::OP_A_MUL_B, {l0a_out1,l0b_out1}, {a_mul_b_out1});
    currFunctionPtr->AddRawOperation(Opcode::OP_A_MUL_B, {l0a_out2, l0b_out2}, {a_mul_b_out2});

    currFunctionPtr->inCasts_.push_back(input_cast1);
    currFunctionPtr->inCasts_.push_back(input_cast2);
    currFunctionPtr->outCasts_.push_back(a_mul_b_out1);
    currFunctionPtr->outCasts_.push_back(a_mul_b_out2);
}
TEST_F(AssignMemoryTypeTest, L1DataMove) {
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "L1DataMove", "L1DataMove", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);
    Program::GetInstance().InsertFuncToFunctionMap("L1DataMove", currFunctionPtr);

    L1DataMoveGraph(currFunctionPtr);

    std::stringstream ssBefore;
    ssBefore << "Before_AssignMemoryType";

    // Call the pass
    AssignMemoryType assignMemoryType;
    assignMemoryType.PreCheck(*currFunctionPtr);
    currFunctionPtr->DumpJsonFile("./config/pass/json/assign_mem_type_L1DataMove_before.json");
    assignMemoryType.RunOnFunction(*currFunctionPtr);
    currFunctionPtr->DumpJsonFile("./config/pass/json/assign_mem_type_L1DataMove_after.json");
    assignMemoryType.PostCheck(*currFunctionPtr);

    std::stringstream ss;
    ss << "After_AssignMemoryType";

    // Validate the results
    std::cout << "========== op size: " << currFunctionPtr->Operations().size() << std::endl;
    for (auto &op : currFunctionPtr->Operations()) {
        if(op.GetOpcode() != Opcode::OP_VIEW) {
            continue;
        }else {
            auto viewOpAttribute = dynamic_cast<ViewOpAttribute *>(op.GetOpAttribute().get());
            auto mem_to = viewOpAttribute->GetTo();
            if(mem_to != MemoryType::MEM_L1) {
                continue;
            }else {
                EXPECT_EQ(op.GetIOperands().size(),1) << "View op has more than one input!";
                EXPECT_EQ(op.GetOOperands().size(),1) << "View op has more than one output!";
                auto input = op.GetIOperands().front();
                auto output = op.GetOOperands().front();
                std::cout << "\t|--- MEM_L1 VIEW iOperand " << input->GetMagic() <<std::endl;
                std::cout << "\t|--- MEM_L1 VIEW oOperand " << output->GetMagic() <<std::endl;
                //EXPECT_EQ(input->GetMemoryTypeToBe(),MemoryType::MEM_L1) << "View op input has unexpected memory type!";
                EXPECT_EQ(output->GetMemoryTypeOriginal(),MemoryType::MEM_L1) << "View op input has unexpected memory type!";
            }
        }
    }
}

void AssignViewTensorWithAttr (std::shared_ptr<Function> &currFunctionPtr) {
    std::shared_ptr<LogicalTensor> view_in1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, std::vector<int64_t>{32,64});
    std::shared_ptr<LogicalTensor> tensor1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, std::vector<int64_t>{32,64});
    std::shared_ptr<LogicalTensor> tensor2 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, std::vector<int64_t>{32,64});
    std::shared_ptr<LogicalTensor> view_out1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, std::vector<int64_t>{32,64});
    std::shared_ptr<LogicalTensor> view_in2 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, std::vector<int64_t>{32,64});
    std::shared_ptr<LogicalTensor> tensor3 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, std::vector<int64_t>{32,64});
    std::shared_ptr<LogicalTensor> tensor4 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, std::vector<int64_t>{32,64});
    std::shared_ptr<LogicalTensor> view_out2 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, std::vector<int64_t>{32,64});
    std::shared_ptr<LogicalTensor> view_in3 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, std::vector<int64_t>{32,64});
    std::shared_ptr<LogicalTensor> tensor5 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, std::vector<int64_t>{32,64});
    std::shared_ptr<LogicalTensor> view_out3 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, std::vector<int64_t>{32,64});
    std::shared_ptr<LogicalTensor> view_in4 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, std::vector<int64_t>{32,64});
    std::shared_ptr<LogicalTensor> tensor6 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, std::vector<int64_t>{32,64});
    std::shared_ptr<LogicalTensor> view_out4 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, std::vector<int64_t>{32,64});
    std::shared_ptr<LogicalTensor> output = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, std::vector<int64_t>{32,64});

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
    auto &view_op5 = currFunctionPtr->AddRawOperation(Opcode::OP_VIEW, {view_in3}, {tensor3});
    auto viewAttribute5 =std::make_shared<ViewOpAttribute>(std::vector<int64_t>{0,0});
    viewAttribute5->SetToType(MemoryType::MEM_L1);
    view_op5.SetOpAttribute(viewAttribute5);
    auto &view_op6 = currFunctionPtr->AddRawOperation(Opcode::OP_VIEW, {tensor3}, {view_out3});
    auto viewAttribute6 =std::make_shared<ViewOpAttribute>(std::vector<int64_t>{0,0});
    viewAttribute6->SetToType(MemoryType::MEM_L0A);
    view_op6.SetOpAttribute(viewAttribute6);
    auto &view_op7 = currFunctionPtr->AddRawOperation(Opcode::OP_VIEW, {view_in4}, {tensor4});
    auto viewAttribute7 =std::make_shared<ViewOpAttribute>(std::vector<int64_t>{0,0});
    viewAttribute7->SetToType(MemoryType::MEM_L1);
    view_op7.SetOpAttribute(viewAttribute7);
    auto &view_op8 = currFunctionPtr->AddRawOperation(Opcode::OP_VIEW, {tensor4}, {view_out4});
    auto viewAttribute8 =std::make_shared<ViewOpAttribute>(std::vector<int64_t>{0,0});
    viewAttribute8->SetToType(MemoryType::MEM_L0B);
    view_op8.SetOpAttribute(viewAttribute7);
    
    currFunctionPtr->AddRawOperation(Opcode::OP_A_MUL_B, {view_out3,view_out4,view_out1,view_out2}, {output});

    currFunctionPtr->inCasts_.push_back(view_in1);
    currFunctionPtr->inCasts_.push_back(view_in2);
    currFunctionPtr->inCasts_.push_back(view_in3);
    currFunctionPtr->inCasts_.push_back(view_in4);
    currFunctionPtr->outCasts_.push_back(output);
}

TEST_F(AssignMemoryTypeTest, TestViewWithAttr) {
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestViewWithAttr", "TestViewWithAttr", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);
    Program::GetInstance().InsertFuncToFunctionMap("TestViewWithAttr", currFunctionPtr);

    AssignViewTensorWithAttr(currFunctionPtr);

    std::stringstream ssBefore;
    ssBefore << "Before_AssignMemoryType";

    // Call the pass
    AssignMemoryType assignMemoryType;
    assignMemoryType.PreCheck(*currFunctionPtr);
    currFunctionPtr->DumpJsonFile("./config/pass/json/assignMemoryType_TestViewWithAttr_before.json");
    assignMemoryType.RunOnFunction(*currFunctionPtr);
    currFunctionPtr->DumpJsonFile("./config/pass/json/assignMemoryType_TestViewWithAttr_after.json");
    assignMemoryType.PostCheck(*currFunctionPtr);

    std::stringstream ss;
    ss << "After_AssignMemoryType";

    // Validate the results
    for (auto &op : currFunctionPtr->Operations()) {
        if (op.GetOpcode() == Opcode::OP_VIEW) {
            auto viewOpAttribute = dynamic_cast<ViewOpAttribute *>(op.GetOpAttribute().get());
            MemoryType attrToType = viewOpAttribute->GetTo();   
            auto output = op.GetOOperands().front();
            auto outputMemOri = output->GetMemoryTypeOriginal();
            auto outputMemTobe = output->GetMemoryTypeToBe();
            std::cout << "\t|--- oOperand " << output->magic;
            std::cout << ", mem ori: " << BriefMemoryTypeToString(outputMemOri);
            std::cout << ", tobe: " << BriefMemoryTypeToString(outputMemTobe) << std::endl;
            EXPECT_EQ(attrToType,outputMemOri);
            EXPECT_EQ(attrToType,outputMemTobe);
        }
    }
}

TEST_F(AssignMemoryTypeTest, TestPostcheckFailWhenTensorMemUnknown) {
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(),
        "TestPostcheckFailWhenTensorMemUnknown", "TestPostcheckFailWhenTensorMemUnknown", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);
    Program::GetInstance().InsertFuncToFunctionMap("TestPostcheckFailWhenTensorMemUnknown", currFunctionPtr);
    AssignViewTensorWithAttr(currFunctionPtr);
    AssignMemoryType assignMemoryType;
    EXPECT_EQ(assignMemoryType.PostCheck(*currFunctionPtr), FAILED);
}

TEST_F(AssignMemoryTypeTest, TestPostcheckFailWhenPathUnreachable) {
    std::vector<int64_t> shape1{NUM_32, NUM_32};
    std::vector<int64_t> shape2{NUM_64, NUM_64};
    std::vector<int64_t> shape3{NUM_128, NUM_128};
    ComputationalGraphBuilder G;
    
    G.AddTensor(DataType::DT_FP32, shape3, "input");
    auto tensorInput = G.GetTensor("input");
    tensorInput->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    G.AddTensor(DataType::DT_FP32, shape2, "a");
    auto tensorA = G.GetTensor("a");
    tensorA->SetMemoryTypeBoth(MemoryType::MEM_L0C, true);
    G.AddTensor(DataType::DT_FP32, shape1, "b");
    auto tensorB= G.GetTensor("b");
    tensorB->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    G.AddTensor(DataType::DT_FP32, shape3, "output");
    auto tensorOutput = G.GetTensor("output");
    tensorOutput->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);

    G.AddOp(Opcode::OP_VIEW, {"input"}, {"a"}, "view1");
    G.GetOp("view1")->SetOpAttribute(std::make_shared<ViewOpAttribute>(shape3, MemoryType::MEM_L0C));
    G.AddOp(Opcode::OP_VIEW, {"a"}, {"b"}, "view2");
    G.GetOp("view2")->SetOpAttribute(std::make_shared<ViewOpAttribute>(shape2, MemoryType::MEM_DEVICE_DDR));
    G.AddOp(Opcode::OP_ASSEMBLE, {"b"}, {"output"}, "assemble1");
    G.GetOp("assemble1")->SetOpAttribute(std::make_shared<AssembleOpAttribute>(MemoryType::MEM_DEVICE_DDR, shape2));
    
    G.SetInCast({"input"});
    G.SetOutCast({"output"});

    Function *function = G.GetFunction();

    AssignMemoryType assignMemoryType;
    EXPECT_EQ(assignMemoryType.PostCheck(*function), FAILED);
}
}
} // namespace npu::tile_fwk