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

//測試帶有髒數據的Reshape操作
TEST_F(CalcCommonTest, UnalignedReshape) {
    // 创建 Function 和 Operation,構造一個虛擬的ExecuteOperationContext
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
} // namespace npu::tile_fwk
