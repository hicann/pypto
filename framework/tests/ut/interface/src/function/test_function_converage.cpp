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
 * \file test_function_converage.cpp
 * \brief
 */

#include "gtest/gtest.h"

#include <iostream>
#include <memory>
#include "interface/configs/config_manager.h"
#define private public
#define protected public
#include "interface/function/function.h"
#undef private
#undef protected
#include "interface/function/function.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"

using namespace npu::tile_fwk;

class FunctionCoverageTest : public testing::Test {
public:
    static void SetUpTestCase() {
        std::cout << "FunctionCoverageTest SetUpTestCase" << std::endl;
        LoggerManager::GetManager().level = LoggerLevel::ERROR;
    }

    static void TearDownTestCase() {
        std::cout << "FunctionCoverageTest TearDownTestCase" << std::endl;
        LoggerManager::GetManager().level = LoggerLevel::ERROR;
    }

    void SetUp() override {
        std::cout << "FunctionCoverageTest SetUp" << std::endl;
        Program::GetInstance().Reset();
    }

    void TearDown() override {
        std::cout << "FunctionCoverageTest TearDown" << std::endl;
    }
};

TEST_F(FunctionCoverageTest, ConverageCase1) {
    config::SetPlatformConfig(KEY_ONLY_HOST_COMPILE, true);

    TileShape::Current().SetVecTile({32, 32});
    std::vector<int64_t> shape{32, 32};
    Tensor input(DT_FP32, shape, "input");
    Tensor output(DT_FP32, shape, "output");
    FUNCTION("ConverageFunc1") {
        Tensor in0 = Reciprocal(input);
        Tensor in1 = Exp(in0);
        Tensor in2 = Sqrt(in1);
        Tensor in3 = Exp(in2);

        Tensor in4 = Exp(in0);
        Tensor in5 = Sqrt(in4);
        Tensor in6 = Exp(in5);
        output = Mul(in3, in6);
    }
    std::cout << "Dump program: " << Program::GetInstance().Dump() << std::endl;

    Function *func = Program::GetInstance().GetFunctionByRawName("TENSOR_ConverageFunc1");
    ASSERT_NE(func, nullptr);
    SubfuncInvokeInfoTy::TensorParamPackTy tensorParam;

    // GetParamIndex
    EXPECT_EQ(func->GetParamIndex(input.GetStorage()->GetRawTensor()), INVALID_IN_OUT_INDEX);
    EXPECT_EQ(func->GetParamIndex(output.GetStorage()->GetRawTensor()), INVALID_IN_OUT_INDEX);
    EXPECT_EQ(func->GetParamIndex(func->GetIncast()[0]->GetRawTensor()), INVALID_IN_OUT_INDEX);
    EXPECT_EQ(func->GetParamIndex(func->GetOutcast()[0]->GetRawTensor()), INVALID_IN_OUT_INDEX);

    std::cout << "===========TensorMagicCheck==========" << std::endl;
    func->TensorMagicCheck();
    std::cout << "===========OperationLoopCheck==========" << std::endl;
    func->OperationLoopCheck("shake it off");

    std::cout << func->DumpJson(true).dump() << std::endl;
    std::cout << func->DumpJson(false).dump() << std::endl;

    Function *currFunc = Program::GetInstance().GetCurrentFunction();
    ASSERT_NE(currFunc, nullptr);
    currFunc->ValidCheck();
}

TEST_F(FunctionCoverageTest, ConverageCase2) {
    config::SetPlatformConfig(KEY_ONLY_HOST_COMPILE, true);

    TileShape::Current().SetVecTile({32, 32});
    std::vector<int64_t> shape{32, 32};
    auto inputPtr = std::make_shared<uint8_t>(1);
    auto outputPtr = std::make_shared<uint8_t>(2);
    Tensor input(DT_FP32, shape, inputPtr.get(), "input");
    Tensor output(DT_FP32, shape, outputPtr.get(), "output");
    config::SetBuildStatic(true);
    FUNCTION("ConverageFunc1", {input, output}) {
        Tensor in0 = Reciprocal(input);
        Tensor in1 = Exp(in0);
        Tensor in2 = Sqrt(in1);
        Tensor in3 = Exp(in2);

        Tensor in4 = Exp(in0);
        Tensor in5 = Sqrt(in4);
        Tensor in6 = Exp(in5);
        output = Mul(in3, in6);
    }
    Function *func = Program::GetInstance().GetFunctionByRawName("TENSOR_ConverageFunc1");
    ASSERT_NE(func, nullptr);

    // GetParamIndex
    EXPECT_EQ(func->GetParamIndex(input.GetStorage()->GetRawTensor()), 0);
    EXPECT_EQ(func->GetParamIndex(output.GetStorage()->GetRawTensor()), 1);
    EXPECT_EQ(func->GetParamIndex(func->GetIncast()[0]->GetRawTensor()), 0);
    EXPECT_EQ(func->GetParamIndex(func->GetOutcast()[0]->GetRawTensor()), 1);
}

TEST_F(FunctionCoverageTest, ConverageCase3) {
    config::SetPlatformConfig(KEY_ONLY_HOST_COMPILE, true);

    TileShape::Current().SetVecTile({32, 32});
    std::vector<int64_t> shape{32, 32};
    auto inputPtr = std::make_shared<uint8_t>(1);
    auto outputPtr = std::make_shared<uint8_t>(2);
    Tensor input(DT_FP32, shape, inputPtr.get(), "input");
    Tensor output(DT_FP32, shape, outputPtr.get(), "output");
    config::SetBuildStatic(true);
    FUNCTION("ConverageFunc") {
        Tensor in0 = Reciprocal(input);
        Tensor in1 = Exp(in0);
        Tensor in2 = Sqrt(in1);
        Tensor in3 = Exp(in2);

        Tensor in4 = Exp(in0);
        Tensor in5 = Sqrt(in4);
        Tensor in6 = Exp(in5);
        output = Mul(in3, in6);
    }
    Function *func = Program::GetInstance().GetFunctionByRawName("TENSOR_ConverageFunc");
    ASSERT_NE(func, nullptr);

    // GetParamIndex
    EXPECT_EQ(func->GetParamIndex(input.GetStorage()->GetRawTensor()), -1);
    EXPECT_EQ(func->GetParamIndex(output.GetStorage()->GetRawTensor()), -1);
    EXPECT_EQ(func->GetParamIndex(func->GetIncast()[0]->GetRawTensor()), -1);
    EXPECT_EQ(func->GetParamIndex(func->GetOutcast()[0]->GetRawTensor()), -1);
}

TEST_F(FunctionCoverageTest, ConverageCase4) {
    config::SetPlatformConfig(KEY_ONLY_HOST_COMPILE, true);
    TileShape::Current().SetVecTile(16, 16);

    std::vector<int64_t> shape{16, 64};
    std::vector<int64_t> childShape{16, 16};
    Tensor a(DataType::DT_FP32, shape, "a");
    Tensor b(DataType::DT_FP32, shape, "b");
    Tensor c(DataType::DT_FP32, shape, "c");

    constexpr int LOOP_END = 4;
    constexpr int CHILD_SHAPE_OFFSET = 16;

    FUNCTION("main", {a, b}, {c}) {
        LOOP("D3", FunctionType::DYNAMIC_LOOP, k, LoopRange(0, LOOP_END)) {
            auto a0 = View(a, childShape, {0, k * CHILD_SHAPE_OFFSET});
            auto b0 = View(b, childShape, {0, k * CHILD_SHAPE_OFFSET});
            auto c0 = Add(a0, b0);
            Assemble(c0, {0, k * CHILD_SHAPE_OFFSET}, c);
            LOOP("D4", FunctionType::DYNAMIC_LOOP, n, LoopRange(0, LOOP_END)) {
                auto d0 = View(a, childShape, {0, n * CHILD_SHAPE_OFFSET});
                auto e0 = View(b, childShape, {0, n * CHILD_SHAPE_OFFSET});
                auto f0 = Add(a0, b0);
                Assemble(c0, {0, k * CHILD_SHAPE_OFFSET}, c);
            }
            auto func = Program::GetInstance().GetCurrentFunction();
            EXPECT_EQ(func->InsertLoopIdxNameList("i"), true);
            EXPECT_EQ(func->InsertLoopIdxNameList("i"), true);
            EXPECT_EQ(func->InsertLoopIdxNameList("k"), false);
            EXPECT_EQ(func->InsertLoopIdxNameList("n"), true);
        }
    }
}

TEST_F(FunctionCoverageTest, TestReuseTensorCase1) {
    config::SetPlatformConfig(KEY_ONLY_HOST_COMPILE, true);
    TileShape::Current().SetVecTile(16, 16);
    std::vector<int64_t> shape{32, 32};
    Tensor input(DT_FP32, shape, "input");
    Tensor output(DT_FP32, shape, "output");
    config::SetBuildStatic(true);
    FUNCTION("R1") {
        Tensor in0 = Exp(input);
        Tensor in1 = Reciprocal(in0);
        Tensor in2 = Exp(in1);
        Tensor in3 = Exp(in2);
        Tensor in4 = Sqrt(in3);
        output = Exp(in4);
    }

    const Function *const_func = Program::GetInstance().GetFunctionByRawName("TENSOR_R1");
    ASSERT_NE(const_func, nullptr);
    Function *func = const_cast<Function *>(const_func);
    ASSERT_NE(func, nullptr);
    std::cout << "=========" << std::endl;
    Operation *re_op = nullptr;
    Operation *sqrt_op = nullptr;
    for (auto &op : func->Operations()) {
        std::cout << "Op:" << op.opmagic<< " " <<  op.GetOpcodeStr() << std::endl;
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
        std::cout << std::endl << std::endl;
        if (op.GetOpcode() == Opcode::OP_RECIPROCAL) {
            re_op = &op;
        }
        if (op.GetOpcode() == Opcode::OP_SQRT) {
            sqrt_op = &op;
        }
    }
    std::cout << "=========" << std::endl;
    ASSERT_NE(re_op, nullptr);
    ASSERT_NE(sqrt_op, nullptr);
    LogicalTensorPtr src_tensor = re_op->GetOutputOperand(0);
    LogicalTensorPtr dst_tensor = sqrt_op->GetOutputOperand(0);
    // EXPECT_EQ(func->TensorReuse(dst_tensor, src_tensor), false);
    // src_tensor->offset = {0, 0};
    // dst_tensor->offset = {0, 0};
    EXPECT_EQ(func->TensorReuse(dst_tensor, src_tensor), true);
    EXPECT_EQ(src_tensor->tensor, dst_tensor->tensor);
    // EXPECT_EQ(sqrt_op->GetInCtrlOperations(), re_op->GetInCtrlOperations());
    EXPECT_EQ(sqrt_op->GetInCtrlOperations().empty(), true);
}

TEST_F(FunctionCoverageTest, TestFunctionHash) {
    config::SetPlatformConfig(KEY_ONLY_HOST_COMPILE, true);
    TileShape::Current().SetVecTile(16, 16);
    std::vector<int64_t> shape{32, 32};
    Tensor input(DT_FP32, shape, "input");
    Tensor output(DT_FP32, shape, "output");
    config::SetBuildStatic(true);
    FUNCTION("R2") {
        Tensor in0 = Exp(input);
        Tensor in1 = Reciprocal(in0);
        Tensor in2 = Exp(in1);
        Tensor in3 = Exp(in2);
        Tensor in4 = Sqrt(in3);
        output = Exp(in4);
    }

    const Function *const_func = Program::GetInstance().GetFunctionByRawName("TENSOR_R2");
    ASSERT_NE(const_func, nullptr);
    Function *func = const_cast<Function *>(const_func);
    ASSERT_NE(func, nullptr);
}
