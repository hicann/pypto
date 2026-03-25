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
 * \file test_execute_calc.cpp
 * \brief
 */

#include <gtest/gtest.h>
#include <limits>

#include "interface/utils/log.h"
#include "interface/interpreter/raw_tensor_data.h"
#include "interface/tensor/float.h"
#include "interface/interpreter/function.h"
#include "interface/interpreter/operation.h"
#include "interface/inner/tilefwk.h"

namespace npu::tile_fwk {
class CalcCommonTest : public testing::Test {
public:
    static void TearDownTestCase() {}

    static void SetUpTestCase() {}

    void SetUp() override {
        Program::GetInstance().Reset();
        config::Reset();
    }

    void TearDown() override {}
};

//测试带有脏数据的Reshape操作
TEST_F(CalcCommonTest, UnalignedReshape) {
    // 创建 Function 和 Operation,构造一个虚拟的ExecuteOperationContext
    auto func = std::make_shared<Function>(Program::GetInstance(), "TestUnalignedReshape", "TestUnalignedReshape", nullptr);
    std::vector<int64_t> inputShape = {2, 2};
    std::vector<int64_t> outputShape = {3, 3};
    auto inputTensor = std::make_shared<LogicalTensor>(*func, DT_FP32, inputShape);
    auto outputTensor = std::make_shared<LogicalTensor>(*func, DT_FP32, outputShape);
    auto &reshapeOp = func->AddOperation(Opcode::OP_RESHAPE, {inputTensor}, {outputTensor});
    Tensor inputTensorData(DT_FP32, outputShape);
    auto inputData = RawTensorData::CreateConstantTensor(inputTensorData, 1.0f);
    auto inputDataView = std::make_shared<LogicalTensorData>(inputData, inputShape, std::vector<int64_t>{0, 0});
    Tensor outputTensorData(DT_FP32, outputShape);
    auto outputData = RawTensorData::CreateConstantTensor(outputTensorData, 1.0f);
    auto outputDataView = std::make_shared<LogicalTensorData>(outputData);
    auto inoutDataPair = std::make_shared<FunctionIODataPair>();
    FunctionFrame frame(func.get(), nullptr, nullptr, inoutDataPair, 0);
    OperationInterpreter opInter;
    std::vector<LogicalTensorDataPtr> ioperandDataViewList = {inputDataView};
    std::vector<LogicalTensorDataPtr> ooperandInplaceDataViewList = {outputDataView};
    ExecuteOperationContext ctx = {
        &frame,
        &opInter,
        &reshapeOp,
        &ioperandDataViewList,
        nullptr,
        &ooperandInplaceDataViewList
    };
    ASSERT_GT(outputDataView->GetSize(), inputDataView->GetSize())
        << "Output size should be greater than input size to trigger the new branch";
    opInter.ExecuteOperation(&ctx);

}

// 测试 Reshape 中当输出 rawTensor 大小大于输入 rawTensor 大小时触发 padding 分支
TEST_F(CalcCommonTest, UnalignedReshapeTriggerPaddingBranch) {
    // 创建 Function 和 Operation, 构造一个虚拟的 ExecuteOperationContext
    auto func = std::make_shared<Function>(Program::GetInstance(),
        "TestUnalignedReshapeTriggerPadding", "TestUnalignedReshapeTriggerPadding", nullptr);

    // 输入逻辑 shape 小，输出逻辑 shape 大
    std::vector<int64_t> inputShape = {2, 2};   // 4 elements
    std::vector<int64_t> outputShape = {3, 3};  // 9 elements

    auto inputTensor = std::make_shared<LogicalTensor>(*func, DT_FP32, inputShape);
    auto outputTensor = std::make_shared<LogicalTensor>(*func, DT_FP32, outputShape);
    auto &reshapeOp = func->AddOperation(Opcode::OP_RESHAPE, {inputTensor}, {outputTensor});

    // 关键点：输入 RawTensor 按 inputShape 创建，输出 RawTensor 按 outputShape 创建
    Tensor inputTensorData(DT_FP32, inputShape);
    auto inputData = RawTensorData::CreateConstantTensor(inputTensorData, 1.0f);
    auto inputDataView = std::make_shared<LogicalTensorData>(
        inputData, inputShape, std::vector<int64_t>{0, 0});

    Tensor outputTensorData(DT_FP32, outputShape);
    auto outputData = RawTensorData::CreateConstantTensor(outputTensorData, 1.0f);
    auto outputDataView = std::make_shared<LogicalTensorData>(outputData);

    auto inoutDataPair = std::make_shared<FunctionIODataPair>();
    FunctionFrame frame(func.get(), nullptr, nullptr, inoutDataPair, 0);
    OperationInterpreter opInter;
    std::vector<LogicalTensorDataPtr> ioperandDataViewList = {inputDataView};
    std::vector<LogicalTensorDataPtr> ooperandInplaceDataViewList = {outputDataView};

    ExecuteOperationContext ctx = {
        &frame,
        &opInter,
        &reshapeOp,
        &ioperandDataViewList,
        nullptr,
        &ooperandInplaceDataViewList
    };

    // 确认 rawTensor 层面输出更大，从而走到 padding 分支
    ASSERT_GT(outputData->GetSize(), inputData->GetSize())
        << "Output raw tensor size should be greater than input raw tensor size to trigger padding branch";

    opInter.ExecuteOperation(&ctx);
}

// 测试 OP_VEC_DUP 在 scalar 为极大 double 时对 FP32 类型输出进行 32 位饱和截断
TEST_F(CalcCommonTest, VecDupClampFp32FromLargeDouble) {
    auto func = std::make_shared<Function>(Program::GetInstance(), "TestVecDupClampFp32",
        "TestVecDupClampFp32", nullptr);

    std::vector<int64_t> outputShape = {2, 2};
    auto outputTensor = std::make_shared<LogicalTensor>(*func, DT_FP32, outputShape);
    auto &vecDupOp = func->AddOperation(Opcode::OP_VEC_DUP, {}, {outputTensor});
    double largeNegDouble = -std::numeric_limits<double>::max();
    Element scalar(DT_FP32, largeNegDouble);
    vecDupOp.SetAttribute(OpAttributeKey::scalar, scalar);
    Tensor outputTensorData(DT_FP32, outputShape);
    auto outputData = RawTensorData::CreateConstantTensor(outputTensorData, 0.0f);
    auto outputDataView = std::make_shared<LogicalTensorData>(outputData);
    auto inoutDataPair = std::make_shared<FunctionIODataPair>();
    FunctionFrame frame(func.get(), nullptr, nullptr, inoutDataPair, 0);
    OperationInterpreter opInter;
    std::vector<LogicalTensorDataPtr> ioperandDataViewList; 
    std::vector<LogicalTensorDataPtr> ooperandInplaceDataViewList = {outputDataView};

    ExecuteOperationContext ctx = {
        &frame,
        &opInter,
        &vecDupOp,
        &ioperandDataViewList,
        nullptr,
        &ooperandInplaceDataViewList
    };

    opInter.ExecuteOperation(&ctx);

    // 期望所有输出元素都被截断为 -FLT_MAX，而不是 -inf
    float expected = -std::numeric_limits<float>::max();
    for (int i = 0; i < outputDataView->GetSize(); ++i) {
        float value = outputDataView->Get<float>(i);
        ASSERT_FLOAT_EQ(value, expected);
    }
}

// 测试 ExecuteOpGatherInL1 中 blocksize 与输入参数、索引和页表的组合是否正确传递到 calc::GatherInL1
TEST_F(CalcCommonTest, ExecuteOpGatherInL1Basic) {
    auto func = std::make_shared<Function>(Program::GetInstance(), "TestGatherInL1",
        "TestGatherInL1", nullptr);

    // params 和 output 为 2 维 tensor，index 和 pageTable 为 1 维 tensor
    std::vector<int64_t> paramsShape = {4, 1};
    std::vector<int64_t> indicesShape = {4, 1};
    std::vector<int64_t> pageTableShape = {2, 1};

    auto paramsTensor = std::make_shared<LogicalTensor>(*func, DT_FP32, paramsShape);
    auto indicesTensor = std::make_shared<LogicalTensor>(*func, DT_INT64, indicesShape);
    auto pageTableTensor = std::make_shared<LogicalTensor>(*func, DT_INT64, pageTableShape);
    auto outputTensor = std::make_shared<LogicalTensor>(*func, DT_FP32, paramsShape);

    auto &gatherOp = func->AddOperation(Opcode::OP_GATHER_IN_L1,
        {paramsTensor, indicesTensor, pageTableTensor},
        {outputTensor});

    int64_t blockSize = 2;
    gatherOp.SetAttribute("op_attr_blocksize", blockSize);

    // 构造输入数据:
    // params: [10, 20, 30, 40]
    // indices: [0, 1, 2, 3]
    // pageTable: [1, 0]
    // GatherInL1 的邏輯會將輸出變換為 [30, 40, 10, 20]
    Tensor paramsTensorData(DT_FP32, paramsShape);
    Tensor indicesTensorData(DT_INT64, indicesShape);
    Tensor pageTableTensorData(DT_INT64, pageTableShape);
    Tensor outputTensorData(DT_FP32, paramsShape);

    std::vector<float> paramsVals = {10.f, 20.f, 30.f, 40.f};
    std::vector<int64_t> indicesVals = {0, 1, 2, 3};
    std::vector<int64_t> pageTableVals = {1, 0};

    auto paramsData = RawTensorData::CreateTensor<float>(paramsTensorData, paramsVals);
    auto indicesData = RawTensorData::CreateTensor<int64_t>(indicesTensorData, indicesVals);
    auto pageTableData = RawTensorData::CreateTensor<int64_t>(pageTableTensorData, pageTableVals);
    auto outputData = RawTensorData::CreateConstantTensor<float>(outputTensorData, 0.f);

    auto paramsView = std::make_shared<LogicalTensorData>(paramsData);
    auto indicesView = std::make_shared<LogicalTensorData>(indicesData);
    auto pageTableView = std::make_shared<LogicalTensorData>(pageTableData);
    auto outputView = std::make_shared<LogicalTensorData>(outputData);

    auto inoutDataPair = std::make_shared<FunctionIODataPair>();
    FunctionFrame frame(func.get(), nullptr, nullptr, inoutDataPair, 0);
    OperationInterpreter opInter;

    std::vector<LogicalTensorDataPtr> ioperandDataViewList = {paramsView, indicesView, pageTableView};
    std::vector<LogicalTensorDataPtr> ooperandInplaceDataViewList = {outputView};

    ExecuteOperationContext ctx = {
        &frame,
        &opInter,
        &gatherOp,
        &ioperandDataViewList,
        nullptr,
        &ooperandInplaceDataViewList
    };

    opInter.ExecuteOperation(&ctx);

    std::vector<float> expected = {30.f, 40.f, 10.f, 20.f};
    ASSERT_EQ(outputView->GetSize(), static_cast<int>(expected.size()));
    for (int i = 0; i < outputView->GetSize(); ++i) {
        float value = outputView->Get<float>(i);
        ASSERT_FLOAT_EQ(value, expected[i]);
    }
}

// 测试 ExecuteOpUnary 以 OP_EXP 为例，验证一元运算路径正确调用 calc::Exp
TEST_F(CalcCommonTest, ExecuteOpUnaryExpBasic) {
    auto func = std::make_shared<Function>(Program::GetInstance(), "TestUnaryExp",
        "TestUnaryExp", nullptr);

    std::vector<int64_t> shape = {4};
    auto inputTensor = std::make_shared<LogicalTensor>(*func, DT_FP32, shape);
    auto outputTensor = std::make_shared<LogicalTensor>(*func, DT_FP32, shape);

    auto &unaryOp = func->AddOperation(Opcode::OP_EXP, {inputTensor}, {outputTensor});

    Tensor inputTensorData(DT_FP32, shape);
    Tensor outputTensorData(DT_FP32, shape);

    std::vector<float> inputVals = {0.f, 1.f, -1.f, 2.f};
    auto inputData = RawTensorData::CreateTensor<float>(inputTensorData, inputVals);
    auto outputData = RawTensorData::CreateConstantTensor<float>(outputTensorData, 0.f);

    auto inputView = std::make_shared<LogicalTensorData>(inputData);
    auto outputView = std::make_shared<LogicalTensorData>(outputData);

    auto inoutDataPair = std::make_shared<FunctionIODataPair>();
    FunctionFrame frame(func.get(), nullptr, nullptr, inoutDataPair, 0);
    OperationInterpreter opInter;

    std::vector<LogicalTensorDataPtr> ioperandDataViewList = {inputView};
    std::vector<LogicalTensorDataPtr> ooperandInplaceDataViewList = {outputView};

    ExecuteOperationContext ctx = {
        &frame,
        &opInter,
        &unaryOp,
        &ioperandDataViewList,
        nullptr,
        &ooperandInplaceDataViewList
    };

    opInter.ExecuteOperation(&ctx);

    ASSERT_EQ(outputView->GetSize(), static_cast<int>(inputVals.size()));
    for (int i = 0; i < outputView->GetSize(); ++i) {
        float value = outputView->Get<float>(i);
        float expected = std::exp(inputVals[i]);
        ASSERT_FLOAT_EQ(value, expected);
    }
}

// 测试 ExecuteOpPad，验证 scalar 属性作为填充值生效
TEST_F(CalcCommonTest, ExecuteOpPadBasic) {
    auto func = std::make_shared<Function>(Program::GetInstance(), "TestPad",
        "TestPad", nullptr);

    // 使用 2 维输入/输出：输入 {2, 2}，输出 {3, 4}，仅支持右侧和底部填充
    std::vector<int64_t> inShape = {2, 2};
    std::vector<int64_t> outShape = {3, 4};

    auto inputTensor = std::make_shared<LogicalTensor>(*func, DT_FP32, inShape);
    auto outputTensor = std::make_shared<LogicalTensor>(*func, DT_FP32, outShape);

    auto &padOp = func->AddOperation(Opcode::OP_PAD, {inputTensor}, {outputTensor});

    // 设置 scalar 属性为 2.0f，期望被用作填充值
    Element scalar(DT_FP32, 2.0f);
    padOp.SetAttribute(OpAttributeKey::scalar, scalar);

    Tensor inputTensorData(DT_FP32, inShape);
    Tensor outputTensorData(DT_FP32, outShape);

    // 按行优先顺序存储 2x2 矩阵：
    // [1, 2]
    // [3, 4]
    std::vector<float> inputVals = {1.f, 2.f, 3.f, 4.f};
    auto inputData = RawTensorData::CreateTensor<float>(inputTensorData, inputVals);
    auto outputData = RawTensorData::CreateConstantTensor<float>(outputTensorData, 0.f);

    auto inputView = std::make_shared<LogicalTensorData>(inputData);
    auto outputView = std::make_shared<LogicalTensorData>(outputData);

    auto inoutDataPair = std::make_shared<FunctionIODataPair>();
    FunctionFrame frame(func.get(), nullptr, nullptr, inoutDataPair, 0);
    OperationInterpreter opInter;

    std::vector<LogicalTensorDataPtr> ioperandDataViewList = {inputView};
    std::vector<LogicalTensorDataPtr> ooperandInplaceDataViewList = {outputView};

    ExecuteOperationContext ctx = {
        &frame,
        &opInter,
        &padOp,
        &ioperandDataViewList,
        nullptr,
        &ooperandInplaceDataViewList
    };

    opInter.ExecuteOperation(&ctx);

    // 输出为 3x4：
    // [1, 2, 2, 2]
    // [3, 4, 2, 2]
    // [2, 2, 2, 2]
    ASSERT_EQ(outputView->GetSize(), 12);
    EXPECT_FLOAT_EQ(outputView->Get<float>(0), 1.f);
    EXPECT_FLOAT_EQ(outputView->Get<float>(1), 2.f);
    EXPECT_FLOAT_EQ(outputView->Get<float>(2), 2.f);
    EXPECT_FLOAT_EQ(outputView->Get<float>(3), 2.f);

    EXPECT_FLOAT_EQ(outputView->Get<float>(4), 3.f);
    EXPECT_FLOAT_EQ(outputView->Get<float>(5), 4.f);
    EXPECT_FLOAT_EQ(outputView->Get<float>(6), 2.f);
    EXPECT_FLOAT_EQ(outputView->Get<float>(7), 2.f);

    EXPECT_FLOAT_EQ(outputView->Get<float>(8), 2.f);
    EXPECT_FLOAT_EQ(outputView->Get<float>(9), 2.f);
    EXPECT_FLOAT_EQ(outputView->Get<float>(10), 2.f);
    EXPECT_FLOAT_EQ(outputView->Get<float>(11), 2.f);
}

// 测试 ExecuteOpOneHot，验证 numClasses 属性和 one_hot 结果
TEST_F(CalcCommonTest, ExecuteOpOneHotBasic) {
    auto func = std::make_shared<Function>(Program::GetInstance(), "TestOneHot",
        "TestOneHot", nullptr);

    // 输入 shape 为 {3}，输出 shape 为 {3, 4}，numClasses = 4
    std::vector<int64_t> inShape = {3};
    std::vector<int64_t> outShape = {3, 4};

    auto inputTensor = std::make_shared<LogicalTensor>(*func, DT_INT32, inShape);
    auto outputTensor = std::make_shared<LogicalTensor>(*func, DT_INT32, outShape);

    auto &oneHotOp = func->AddOperation(Opcode::OP_ONEHOT, {inputTensor}, {outputTensor});

    int numClasses = 4;
    oneHotOp.SetAttribute(OP_ATTR_PREFIX + "numClasses", numClasses);

    Tensor inputTensorData(DT_INT32, inShape);
    Tensor outputTensorData(DT_INT32, outShape);

    // indices: [0, 1, 3]
    std::vector<int32_t> inputVals = {0, 1, 3};
    auto inputData = RawTensorData::CreateTensor<int32_t>(inputTensorData, inputVals);
    auto outputData = RawTensorData::CreateConstantTensor<int32_t>(outputTensorData, 0);

    auto inputView = std::make_shared<LogicalTensorData>(inputData);
    auto outputView = std::make_shared<LogicalTensorData>(outputData);

    auto inoutDataPair = std::make_shared<FunctionIODataPair>();
    FunctionFrame frame(func.get(), nullptr, nullptr, inoutDataPair, 0);
    OperationInterpreter opInter;

    std::vector<LogicalTensorDataPtr> ioperandDataViewList = {inputView};
    std::vector<LogicalTensorDataPtr> ooperandInplaceDataViewList = {outputView};

    ExecuteOperationContext ctx = {
        &frame,
        &opInter,
        &oneHotOp,
        &ioperandDataViewList,
        nullptr,
        &ooperandInplaceDataViewList
    };

    opInter.ExecuteOperation(&ctx);

    // 期望输出为 3x4：
    // [1, 0, 0, 0]
    // [0, 1, 0, 0]
    // [0, 0, 0, 1]
    ASSERT_EQ(outputView->GetSize(), 12);
    EXPECT_EQ(outputView->Get<int32_t>(0), 1);
    EXPECT_EQ(outputView->Get<int32_t>(1), 0);
    EXPECT_EQ(outputView->Get<int32_t>(2), 0);
    EXPECT_EQ(outputView->Get<int32_t>(3), 0);

    EXPECT_EQ(outputView->Get<int32_t>(4), 0);
    EXPECT_EQ(outputView->Get<int32_t>(5), 1);
    EXPECT_EQ(outputView->Get<int32_t>(6), 0);
    EXPECT_EQ(outputView->Get<int32_t>(7), 0);

    EXPECT_EQ(outputView->Get<int32_t>(8), 0);
    EXPECT_EQ(outputView->Get<int32_t>(9), 0);
    EXPECT_EQ(outputView->Get<int32_t>(10), 0);
    EXPECT_EQ(outputView->Get<int32_t>(11), 1);
}

// 测试 ExecuteOpTransposeMoveOut，在无 CopyOpAttribute 时走默认转置分支
TEST_F(CalcCommonTest, ExecuteOpTransposeMoveOutBasic) {
    auto func = std::make_shared<Function>(Program::GetInstance(), "TestTransposeMoveOut",
        "TestTransposeMoveOut", nullptr);

    // 输入 2x3，输出 3x2，转置轴为 (0, 1)
    std::vector<int64_t> inShape = {2, 3};
    std::vector<int64_t> outShape = {3, 2};

    auto inputTensor = std::make_shared<LogicalTensor>(*func, DT_FP32, inShape);
    auto outputTensor = std::make_shared<LogicalTensor>(*func, DT_FP32, outShape);

    auto &transposeOp = func->AddOperation(Opcode::OP_TRANSPOSE_MOVEOUT, {inputTensor}, {outputTensor});

    // 设置转置轴属性
    std::vector<int64_t> axes = {0, 1};
    transposeOp.SetAttribute(OP_ATTR_PREFIX + "shape", axes);

    Tensor inputTensorData(DT_FP32, inShape);
    Tensor outputTensorData(DT_FP32, outShape);

    // 2x3 矩阵：
    // [1, 2, 3]
    // [4, 5, 6]
    std::vector<float> inputVals = {1.f, 2.f, 3.f, 4.f, 5.f, 6.f};
    auto inputData = RawTensorData::CreateTensor<float>(inputTensorData, inputVals);
    auto outputData = RawTensorData::CreateConstantTensor<float>(outputTensorData, 0.f);

    auto inputView = std::make_shared<LogicalTensorData>(inputData);
    auto outputView = std::make_shared<LogicalTensorData>(outputData);

    auto inoutDataPair = std::make_shared<FunctionIODataPair>();
    FunctionFrame frame(func.get(), nullptr, nullptr, inoutDataPair, 0);
    OperationInterpreter opInter;

    std::vector<LogicalTensorDataPtr> ioperandDataViewList = {inputView};
    std::vector<LogicalTensorDataPtr> ooperandInplaceDataViewList = {outputView};

    ExecuteOperationContext ctx = {
        &frame,
        &opInter,
        &transposeOp,
        &ioperandDataViewList,
        nullptr,
        &ooperandInplaceDataViewList
    };

    opInter.ExecuteOperation(&ctx);

    // 期望输出为 3x2：
    // [1, 4]
    // [2, 5]
    // [3, 6]
    ASSERT_EQ(outputView->GetSize(), 6);
    EXPECT_FLOAT_EQ(outputView->Get<float>(0), 1.f);
    EXPECT_FLOAT_EQ(outputView->Get<float>(1), 4.f);
    EXPECT_FLOAT_EQ(outputView->Get<float>(2), 2.f);
    EXPECT_FLOAT_EQ(outputView->Get<float>(3), 5.f);
    EXPECT_FLOAT_EQ(outputView->Get<float>(4), 3.f);
    EXPECT_FLOAT_EQ(outputView->Get<float>(5), 6.f);
}

// 测试 ExecuteOpTranspose，验证通过 OP_TRANSPOSE_VNCHWCONV 的转置行为
TEST_F(CalcCommonTest, ExecuteOpTransposeBasic) {
    auto func = std::make_shared<Function>(Program::GetInstance(), "TestTranspose",
        "TestTranspose", nullptr);

    // 输入 2x3，输出 3x2，转置轴为 (0, 1)
    std::vector<int64_t> inShape = {2, 3};
    std::vector<int64_t> outShape = {3, 2};

    auto inputTensor = std::make_shared<LogicalTensor>(*func, DT_FP32, inShape);
    auto outputTensor = std::make_shared<LogicalTensor>(*func, DT_FP32, outShape);

    auto &transposeOp = func->AddOperation(Opcode::OP_TRANSPOSE_VNCHWCONV, {inputTensor}, {outputTensor});

    // 设置转置轴属性
    std::vector<int64_t> axes = {0, 1};
    transposeOp.SetAttribute(OP_ATTR_PREFIX + "shape", axes);

    Tensor inputTensorData(DT_FP32, inShape);
    Tensor outputTensorData(DT_FP32, outShape);

    // 2x3 矩阵：
    // [1, 2, 3]
    // [4, 5, 6]
    std::vector<float> inputVals = {1.f, 2.f, 3.f, 4.f, 5.f, 6.f};
    auto inputData = RawTensorData::CreateTensor<float>(inputTensorData, inputVals);
    auto outputData = RawTensorData::CreateConstantTensor<float>(outputTensorData, 0.f);

    auto inputView = std::make_shared<LogicalTensorData>(inputData);
    auto outputView = std::make_shared<LogicalTensorData>(outputData);

    auto inoutDataPair = std::make_shared<FunctionIODataPair>();
    FunctionFrame frame(func.get(), nullptr, nullptr, inoutDataPair, 0);
    OperationInterpreter opInter;

    std::vector<LogicalTensorDataPtr> ioperandDataViewList = {inputView};
    std::vector<LogicalTensorDataPtr> ooperandInplaceDataViewList = {outputView};

    ExecuteOperationContext ctx = {
        &frame,
        &opInter,
        &transposeOp,
        &ioperandDataViewList,
        nullptr,
        &ooperandInplaceDataViewList
    };

    opInter.ExecuteOperation(&ctx);

    // 期望输出为 3x2：
    // [1, 4]
    // [2, 5]
    // [3, 6]
    ASSERT_EQ(outputView->GetSize(), 6);
    EXPECT_FLOAT_EQ(outputView->Get<float>(0), 1.f);
    EXPECT_FLOAT_EQ(outputView->Get<float>(1), 4.f);
    EXPECT_FLOAT_EQ(outputView->Get<float>(2), 2.f);
    EXPECT_FLOAT_EQ(outputView->Get<float>(3), 5.f);
    EXPECT_FLOAT_EQ(outputView->Get<float>(4), 3.f);
    EXPECT_FLOAT_EQ(outputView->Get<float>(5), 6.f);
}

// 测试 ExecuteOpLogicalNot，验证布尔取反
TEST_F(CalcCommonTest, ExecuteOpLogicalNotBasic) {
    auto func = std::make_shared<Function>(Program::GetInstance(), "TestLogicalNot",
        "TestLogicalNot", nullptr);

    std::vector<int64_t> shape = {4};
    auto inputTensor = std::make_shared<LogicalTensor>(*func, DT_BOOL, shape);
    auto outputTensor = std::make_shared<LogicalTensor>(*func, DT_BOOL, shape);

    auto &logicalNotOp = func->AddOperation(Opcode::OP_LOGICALNOT, {inputTensor}, {outputTensor});

    Tensor inputTensorData(DT_BOOL, shape);
    Tensor outputTensorData(DT_BOOL, shape);

    // 输入: [true, false, true, false]
    std::vector<uint8_t> inputVals = {1, 0, 1, 0};
    auto inputData = RawTensorData::CreateTensor<uint8_t>(inputTensorData, inputVals);
    auto outputData = RawTensorData::CreateConstantTensor<uint8_t>(outputTensorData, 0);

    auto inputView = std::make_shared<LogicalTensorData>(inputData);
    auto outputView = std::make_shared<LogicalTensorData>(outputData);

    auto inoutDataPair = std::make_shared<FunctionIODataPair>();
    FunctionFrame frame(func.get(), nullptr, nullptr, inoutDataPair, 0);
    OperationInterpreter opInter;

    std::vector<LogicalTensorDataPtr> ioperandDataViewList = {inputView};
    std::vector<LogicalTensorDataPtr> ooperandInplaceDataViewList = {outputView};

    ExecuteOperationContext ctx = {
        &frame,
        &opInter,
        &logicalNotOp,
        &ioperandDataViewList,
        nullptr,
        &ooperandInplaceDataViewList
    };

    opInter.ExecuteOperation(&ctx);

    ASSERT_EQ(outputView->GetSize(), 4);
    EXPECT_EQ(outputView->Get<bool>(0), false);
    EXPECT_EQ(outputView->Get<bool>(1), true);
    EXPECT_EQ(outputView->Get<bool>(2), false);
    EXPECT_EQ(outputView->Get<bool>(3), true);
}

// 测试 ExecuteOpLogicalAnd，验证布尔与
TEST_F(CalcCommonTest, ExecuteOpLogicalAndBasic) {
    auto func = std::make_shared<Function>(Program::GetInstance(), "TestLogicalAnd",
        "TestLogicalAnd", nullptr);

    std::vector<int64_t> shape = {4};
    auto lhsTensor = std::make_shared<LogicalTensor>(*func, DT_BOOL, shape);
    auto rhsTensor = std::make_shared<LogicalTensor>(*func, DT_BOOL, shape);
    auto outputTensor = std::make_shared<LogicalTensor>(*func, DT_BOOL, shape);

    auto &logicalAndOp = func->AddOperation(Opcode::OP_LOGICALAND, {lhsTensor, rhsTensor}, {outputTensor});

    Tensor lhsTensorData(DT_BOOL, shape);
    Tensor rhsTensorData(DT_BOOL, shape);
    Tensor outputTensorData(DT_BOOL, shape);

    // lhs: [true, true, false, false]
    // rhs: [true, false, true, false]
    std::vector<uint8_t> lhsVals = {1, 1, 0, 0};
    std::vector<uint8_t> rhsVals = {1, 0, 1, 0};
    auto lhsData = RawTensorData::CreateTensor<uint8_t>(lhsTensorData, lhsVals);
    auto rhsData = RawTensorData::CreateTensor<uint8_t>(rhsTensorData, rhsVals);
    auto outputData = RawTensorData::CreateConstantTensor<uint8_t>(outputTensorData, 0);

    auto lhsView = std::make_shared<LogicalTensorData>(lhsData);
    auto rhsView = std::make_shared<LogicalTensorData>(rhsData);
    auto outputView = std::make_shared<LogicalTensorData>(outputData);

    auto inoutDataPair = std::make_shared<FunctionIODataPair>();
    FunctionFrame frame(func.get(), nullptr, nullptr, inoutDataPair, 0);
    OperationInterpreter opInter;

    std::vector<LogicalTensorDataPtr> ioperandDataViewList = {lhsView, rhsView};
    std::vector<LogicalTensorDataPtr> ooperandInplaceDataViewList = {outputView};

    ExecuteOperationContext ctx = {
        &frame,
        &opInter,
        &logicalAndOp,
        &ioperandDataViewList,
        nullptr,
        &ooperandInplaceDataViewList
    };

    opInter.ExecuteOperation(&ctx);

    // 期望: [true && true, true && false, false && true, false && false]
    // 即 [true, false, false, false]
    ASSERT_EQ(outputView->GetSize(), 4);
    EXPECT_EQ(outputView->Get<bool>(0), true);
    EXPECT_EQ(outputView->Get<bool>(1), false);
    EXPECT_EQ(outputView->Get<bool>(2), false);
    EXPECT_EQ(outputView->Get<bool>(3), false);
}

// 测试 ExecuteOpLog1p，验证 calc::Log1p 行为
TEST_F(CalcCommonTest, ExecuteOpLog1pBasic) {
    auto func = std::make_shared<Function>(Program::GetInstance(), "TestLog1p",
        "TestLog1p", nullptr);

    std::vector<int64_t> shape = {4};
    auto inputTensor = std::make_shared<LogicalTensor>(*func, DT_FP32, shape);
    auto outputTensor = std::make_shared<LogicalTensor>(*func, DT_FP32, shape);

    auto &log1pOp = func->AddOperation(Opcode::OP_LOG1P, {inputTensor}, {outputTensor});

    Tensor inputTensorData(DT_FP32, shape);
    Tensor outputTensorData(DT_FP32, shape);

    std::vector<float> inputVals = {0.f, 1.f, 2.f, -0.5f};
    auto inputData = RawTensorData::CreateTensor<float>(inputTensorData, inputVals);
    auto outputData = RawTensorData::CreateConstantTensor<float>(outputTensorData, 0.f);

    auto inputView = std::make_shared<LogicalTensorData>(inputData);
    auto outputView = std::make_shared<LogicalTensorData>(outputData);

    auto inoutDataPair = std::make_shared<FunctionIODataPair>();
    FunctionFrame frame(func.get(), nullptr, nullptr, inoutDataPair, 0);
    OperationInterpreter opInter;

    std::vector<LogicalTensorDataPtr> ioperandDataViewList = {inputView};
    std::vector<LogicalTensorDataPtr> ooperandInplaceDataViewList = {outputView};

    ExecuteOperationContext ctx = {
        &frame,
        &opInter,
        &log1pOp,
        &ioperandDataViewList,
        nullptr,
        &ooperandInplaceDataViewList
    };

    opInter.ExecuteOperation(&ctx);

    ASSERT_EQ(outputView->GetSize(), static_cast<int>(inputVals.size()));
    for (int i = 0; i < outputView->GetSize(); ++i) {
        float value = outputView->Get<float>(i);
        float expected = std::log1pf(inputVals[i]);
        ASSERT_FLOAT_EQ(value, expected);
    }
}

// 测试 ExecuteOpCumSum，验证按指定 axis 进行前缀和
TEST_F(CalcCommonTest, ExecuteOpCumSumBasic) {
    auto func = std::make_shared<Function>(Program::GetInstance(), "TestCumSum",
        "TestCumSum", nullptr);

    std::vector<int64_t> shape = {2, 3};
    auto inputTensor = std::make_shared<LogicalTensor>(*func, DT_FP32, shape);
    auto outputTensor = std::make_shared<LogicalTensor>(*func, DT_FP32, shape);

    auto &cumsumOp = func->AddOperation(Opcode::OP_CUM_SUM, {inputTensor}, {outputTensor});
    int axis = 1;
    cumsumOp.SetAttribute(OP_ATTR_PREFIX + "axis", axis);

    Tensor inputTensorData(DT_FP32, shape);
    Tensor outputTensorData(DT_FP32, shape);

    // 2x3:
    // [1, 2, 3]
    // [4, 5, 6]
    std::vector<float> inputVals = {1.f, 2.f, 3.f, 4.f, 5.f, 6.f};
    auto inputData = RawTensorData::CreateTensor<float>(inputTensorData, inputVals);
    auto outputData = RawTensorData::CreateConstantTensor<float>(outputTensorData, 0.f);

    auto inputView = std::make_shared<LogicalTensorData>(inputData);
    auto outputView = std::make_shared<LogicalTensorData>(outputData);

    auto inoutDataPair = std::make_shared<FunctionIODataPair>();
    FunctionFrame frame(func.get(), nullptr, nullptr, inoutDataPair, 0);
    OperationInterpreter opInter;

    std::vector<LogicalTensorDataPtr> ioperandDataViewList = {inputView};
    std::vector<LogicalTensorDataPtr> ooperandInplaceDataViewList = {outputView};

    ExecuteOperationContext ctx = {
        &frame,
        &opInter,
        &cumsumOp,
        &ioperandDataViewList,
        nullptr,
        &ooperandInplaceDataViewList
    };

    opInter.ExecuteOperation(&ctx);

    // 期望沿 axis=1 逐行做前缀和：
    // [1, 3, 6]
    // [4, 9, 15]
    ASSERT_EQ(outputView->GetSize(), 6);
    EXPECT_FLOAT_EQ(outputView->Get<float>(0), 1.f);
    EXPECT_FLOAT_EQ(outputView->Get<float>(1), 3.f);
    EXPECT_FLOAT_EQ(outputView->Get<float>(2), 6.f);
    EXPECT_FLOAT_EQ(outputView->Get<float>(3), 4.f);
    EXPECT_FLOAT_EQ(outputView->Get<float>(4), 9.f);
    EXPECT_FLOAT_EQ(outputView->Get<float>(5), 15.f);
}

// 测试 ExecuteOpIndexPut，验证简单 index_put 行为（不累加）
TEST_F(CalcCommonTest, ExecuteOpIndexPutBasic) {
    auto func = std::make_shared<Function>(Program::GetInstance(), "TestIndexPut",
        "TestIndexPut", nullptr);

    // self: 1x4，values: 1 元素，indices: 2 个一维索引：行索引 {0}，列索引 {1}
    std::vector<int64_t> selfShape = {1, 4};
    std::vector<int64_t> valuesShape = {1};
    std::vector<int64_t> indexShape = {1};

    auto selfTensor = std::make_shared<LogicalTensor>(*func, DT_FP32, selfShape);
    auto valuesTensor = std::make_shared<LogicalTensor>(*func, DT_FP32, valuesShape);
    auto rowIndexTensor = std::make_shared<LogicalTensor>(*func, DT_INT64, indexShape);
    auto colIndexTensor = std::make_shared<LogicalTensor>(*func, DT_INT64, indexShape);
    auto outputTensor = std::make_shared<LogicalTensor>(*func, DT_FP32, selfShape);

    // 传入两个 index tensor，对应 2 个维度
    auto &indexPutOp = func->AddOperation(Opcode::OP_INDEX_PUT,
        {selfTensor, valuesTensor, rowIndexTensor, colIndexTensor}, {outputTensor});
    indexPutOp.SetAttribute(OpAttributeKey::accumulate, false);

    Tensor selfDataTensor(DT_FP32, selfShape);
    Tensor valuesDataTensor(DT_FP32, valuesShape);
    Tensor rowIndexDataTensor(DT_INT64, indexShape);
    Tensor colIndexDataTensor(DT_INT64, indexShape);
    Tensor outputDataTensor(DT_FP32, selfShape);

    // self: [1, 2, 3, 4], values: [10], indices: [1] -> 位置 1 被赋值为 10
    std::vector<float> selfVals = {1.f, 2.f, 3.f, 4.f};
    std::vector<float> valuesVals = {10.f};
    // 行索引 0，列索引 1
    std::vector<int64_t> rowIndexVals = {0};
    std::vector<int64_t> colIndexVals = {1};

    auto selfData = RawTensorData::CreateTensor<float>(selfDataTensor, selfVals);
    auto valuesData = RawTensorData::CreateTensor<float>(valuesDataTensor, valuesVals);
    auto rowIndexData = RawTensorData::CreateTensor<int64_t>(rowIndexDataTensor, rowIndexVals);
    auto colIndexData = RawTensorData::CreateTensor<int64_t>(colIndexDataTensor, colIndexVals);
    auto outputData = RawTensorData::CreateConstantTensor<float>(outputDataTensor, 0.f);

    auto selfView = std::make_shared<LogicalTensorData>(selfData);
    auto valuesView = std::make_shared<LogicalTensorData>(valuesData);
    auto rowIndexView = std::make_shared<LogicalTensorData>(rowIndexData);
    auto colIndexView = std::make_shared<LogicalTensorData>(colIndexData);
    auto outputView = std::make_shared<LogicalTensorData>(outputData);

    auto inoutDataPair = std::make_shared<FunctionIODataPair>();
    FunctionFrame frame(func.get(), nullptr, nullptr, inoutDataPair, 0);
    OperationInterpreter opInter;

    std::vector<LogicalTensorDataPtr> ioperandDataViewList = {selfView, valuesView, rowIndexView, colIndexView};
    std::vector<LogicalTensorDataPtr> ooperandInplaceDataViewList = {outputView};

    ExecuteOperationContext ctx = {
        &frame,
        &opInter,
        &indexPutOp,
        &ioperandDataViewList,
        nullptr,
        &ooperandInplaceDataViewList
    };

    opInter.ExecuteOperation(&ctx);

    // 期望: [1, 10, 3, 4]
    ASSERT_EQ(outputView->GetSize(), 4);
    EXPECT_FLOAT_EQ(outputView->Get<float>(0), 1.f);
    EXPECT_FLOAT_EQ(outputView->Get<float>(1), 10.f);
    EXPECT_FLOAT_EQ(outputView->Get<float>(2), 3.f);
    EXPECT_FLOAT_EQ(outputView->Get<float>(3), 4.f);
}

// 测试 ExecuteOpTopkExtract，验证从打包的 [value, index, ...] 中提取 top-k 值
TEST_F(CalcCommonTest, ExecuteOpTopkExtractValueBasic) {
    auto func = std::make_shared<Function>(Program::GetInstance(), "TestTopkExtractValue",
        "TestTopkExtractValue", nullptr);

    // 输入 shape: {1, 6}，数据按 [v0, i0, v1, i1, v2, i2] 打包
    std::vector<int64_t> inShape = {1, 6};
    std::vector<int64_t> outShape = {1, 2}; // k = 2

    auto inputTensor = std::make_shared<LogicalTensor>(*func, DT_FP32, inShape);
    auto outputTensor = std::make_shared<LogicalTensor>(*func, DT_FP32, outShape);

    auto &topkOp = func->AddOperation(Opcode::OP_TOPK_EXTRACT, {inputTensor}, {outputTensor});
    int k = 2;
    int isIndex = 0; // 提取 value
    topkOp.SetAttribute(OP_ATTR_PREFIX + "k", k);
    topkOp.SetAttribute(OP_ATTR_PREFIX + "is_index", isIndex);

    Tensor inputTensorData(DT_FP32, inShape);
    Tensor outputTensorData(DT_FP32, outShape);

    // v: [0.5, 1.5, 2.5], i: [1, 3, 5] -> TopkExtract 按顺序提取前 k 对的 value，这里应为 [0.5, 1.5]
    std::vector<float> packedVals = {0.5f, 1.f, 1.5f, 3.f, 2.5f, 5.f};
    auto inputData = RawTensorData::CreateTensor<float>(inputTensorData, packedVals);
    auto outputData = RawTensorData::CreateConstantTensor<float>(outputTensorData, 0.f);

    auto inputView = std::make_shared<LogicalTensorData>(inputData);
    auto outputView = std::make_shared<LogicalTensorData>(outputData);

    auto inoutDataPair = std::make_shared<FunctionIODataPair>();
    FunctionFrame frame(func.get(), nullptr, nullptr, inoutDataPair, 0);
    OperationInterpreter opInter;

    std::vector<LogicalTensorDataPtr> ioperandDataViewList = {inputView};
    std::vector<LogicalTensorDataPtr> ooperandInplaceDataViewList = {outputView};

    ExecuteOperationContext ctx = {
        &frame,
        &opInter,
        &topkOp,
        &ioperandDataViewList,
        nullptr,
        &ooperandInplaceDataViewList
    };

    opInter.ExecuteOperation(&ctx);

    ASSERT_EQ(outputView->GetSize(), 2);
    EXPECT_FLOAT_EQ(outputView->Get<float>(0), 0.5f);
    EXPECT_FLOAT_EQ(outputView->Get<float>(1), 1.5f);
}

// 测试 ExecuteOpTopkExtract，提取 indices
TEST_F(CalcCommonTest, ExecuteOpTopkExtractIndexBasic) {
    auto func = std::make_shared<Function>(Program::GetInstance(), "TestTopkExtractIndex",
        "TestTopkExtractIndex", nullptr);

    std::vector<int64_t> inShape = {1, 6};
    std::vector<int64_t> outShape = {1, 2}; // k = 2

    auto inputTensor = std::make_shared<LogicalTensor>(*func, DT_FP32, inShape);
    auto outputTensor = std::make_shared<LogicalTensor>(*func, DT_INT32, outShape);

    auto &topkOp = func->AddOperation(Opcode::OP_TOPK_EXTRACT, {inputTensor}, {outputTensor});
    int k = 2;
    int isIndex = 1; // 提取 index
    topkOp.SetAttribute(OP_ATTR_PREFIX + "k", k);
    topkOp.SetAttribute(OP_ATTR_PREFIX + "is_index", isIndex);

    Tensor inputTensorData(DT_FP32, inShape);
    Tensor outputTensorData(DT_INT32, outShape);

    // 同样的 packed 数据
    std::vector<float> packedVals = {0.5f, 1.f, 1.5f, 3.f, 2.5f, 5.f};
    auto inputData = RawTensorData::CreateTensor<float>(inputTensorData, packedVals);
    auto outputData = RawTensorData::CreateConstantTensor<int32_t>(outputTensorData, 0);

    auto inputView = std::make_shared<LogicalTensorData>(inputData);
    auto outputView = std::make_shared<LogicalTensorData>(outputData);

    auto inoutDataPair = std::make_shared<FunctionIODataPair>();
    FunctionFrame frame(func.get(), nullptr, nullptr, inoutDataPair, 0);
    OperationInterpreter opInter;

    std::vector<LogicalTensorDataPtr> ioperandDataViewList = {inputView};
    std::vector<LogicalTensorDataPtr> ooperandInplaceDataViewList = {outputView};

    ExecuteOperationContext ctx = {
        &frame,
        &opInter,
        &topkOp,
        &ioperandDataViewList,
        nullptr,
        &ooperandInplaceDataViewList
    };

    opInter.ExecuteOperation(&ctx);

    ASSERT_EQ(outputView->GetSize(), 2);
    EXPECT_EQ(outputView->Get<int32_t>(0), 1);
    // TopkExtract 按 [v0, i0, v1, i1, ...] 顺序，从前 k 对中提取索引，这里应为 [i0, i1] = [1, 3]
    EXPECT_EQ(outputView->Get<int32_t>(1), 3);
}

// 测试 ExecuteOpReduceAcc，验证多输入累加
TEST_F(CalcCommonTest, ExecuteOpReduceAccBasic) {
    auto func = std::make_shared<Function>(Program::GetInstance(), "TestReduceAcc",
        "TestReduceAcc", nullptr);

    std::vector<int64_t> shape = {3};
    auto t0 = std::make_shared<LogicalTensor>(*func, DT_FP32, shape);
    auto t1 = std::make_shared<LogicalTensor>(*func, DT_FP32, shape);
    auto t2 = std::make_shared<LogicalTensor>(*func, DT_FP32, shape);
    auto outputTensor = std::make_shared<LogicalTensor>(*func, DT_FP32, shape);

    auto &reduceAccOp = func->AddOperation(Opcode::OP_REDUCE_ACC, {t0, t1, t2}, {outputTensor});

    Tensor data0(DT_FP32, shape);
    Tensor data1(DT_FP32, shape);
    Tensor data2(DT_FP32, shape);
    Tensor outData(DT_FP32, shape);

    std::vector<float> v0 = {1.f, 2.f, 3.f};
    std::vector<float> v1 = {4.f, 5.f, 6.f};
    std::vector<float> v2 = {7.f, 8.f, 9.f};

    auto d0 = RawTensorData::CreateTensor<float>(data0, v0);
    auto d1 = RawTensorData::CreateTensor<float>(data1, v1);
    auto d2 = RawTensorData::CreateTensor<float>(data2, v2);
    auto dout = RawTensorData::CreateConstantTensor<float>(outData, 0.f);

    auto v0View = std::make_shared<LogicalTensorData>(d0);
    auto v1View = std::make_shared<LogicalTensorData>(d1);
    auto v2View = std::make_shared<LogicalTensorData>(d2);
    auto outView = std::make_shared<LogicalTensorData>(dout);

    auto inoutDataPair = std::make_shared<FunctionIODataPair>();
    FunctionFrame frame(func.get(), nullptr, nullptr, inoutDataPair, 0);
    OperationInterpreter opInter;

    std::vector<LogicalTensorDataPtr> ioperandDataViewList = {v0View, v1View, v2View};
    std::vector<LogicalTensorDataPtr> ooperandInplaceDataViewList = {outView};

    ExecuteOperationContext ctx = {
        &frame,
        &opInter,
        &reduceAccOp,
        &ioperandDataViewList,
        nullptr,
        &ooperandInplaceDataViewList
    };

    opInter.ExecuteOperation(&ctx);

    // 期望逐元素相加: [1+4+7, 2+5+8, 3+6+9] = [12, 15, 18]
    ASSERT_EQ(outView->GetSize(), 3);
    EXPECT_FLOAT_EQ(outView->Get<float>(0), 12.f);
    EXPECT_FLOAT_EQ(outView->Get<float>(1), 15.f);
    EXPECT_FLOAT_EQ(outView->Get<float>(2), 18.f);
}
} // namespace npu::tile_fwk
