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
 * \file test_dynamic_gen_gated_score.cpp
 * \brief
 */
#include "operator/models/nsa/gen_gated_score.h"
#include "interface/interpreter/raw_tensor_data.h"
#include "interface/tensor/float.h"
#include "test_dev_func_runner.h"
#include "test_suite_stest_ops.h"
#include "tilefwk/tensor.h"

using namespace npu::tile_fwk;

class GenGatedScore : public npu::tile_fwk::stest::TestSuite_STest_Ops_Aihac {};

template <typename T>
static std::shared_ptr<RawTensorData> CreateTensorData(Tensor tensor, std::string fileName) {
    auto shape = tensor.GetShape();
    int capacity = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<>());
    std::vector<T> values(capacity, 0);
    readInput<T>(GetGoldenDir() + fileName, values);
    return RawTensorData::CreateTensor<T>(tensor, values);
}

template <typename T>
static std::vector<T> GetGoldenVec(std::vector<int64_t> shape, std::string fileName) {
    uint64_t capacity = std::accumulate(shape.begin(), shape.end(), uint64_t{1}, std::multiplies<uint64_t>());
    std::vector<T> golden(capacity, 0);
    readInput<T>(GetGoldenDir() + fileName, golden);
    return golden;
}

template<typename T = npu::tile_fwk::float16>
void GenGatedScoreEntryPrefill(const std::vector<int64_t>& bnsh) {
    config::SetHostOption(ONLY_CODEGEN, true);

    int64_t b = bnsh[0];
    int64_t n = bnsh[1];
    int64_t s = bnsh[2];
    int64_t h = bnsh[3];

    DataType dType = GetAstDtype<T>();

    std::vector<int64_t> xShape = {b, s, h};
    std::vector<int64_t> w1Shape = {h, h * 4};
    std::vector<int64_t> w2Shape = {h * 4, n * 3};
    std::vector<int64_t> gatingScoreShape = {b, s, 3, n};

    Tensor x(dType, xShape, "x");
    Tensor w1(dType, w1Shape, "w1");
    Tensor w2(dType, w2Shape, "w2");
    Tensor gatingScore(dType, gatingScoreShape, "gatingScore");

    auto goldenData = GetGoldenVec<T>(gatingScoreShape, "/gatingscore.bin");
    auto xData = CreateTensorData<T>(x, "/x.bin");
    auto w1Data = CreateTensorData<T>(w1, "/w1.bin");
    auto w2Data = CreateTensorData<T>(w2, "/w2.bin");
    auto gatingScoreData = RawTensorData::CreateConstantTensor<T>(gatingScore, 0.0);
    std::vector<RawTensorDataPtr> inputDataList = {xData, w1Data, w2Data};
    std::vector<RawTensorDataPtr> outputDataList = {gatingScoreData};

    GenGatedScoreFuncPrefill(x, w1, w2, gatingScore);

#ifdef BUILD_WITH_CANN
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction(), inputDataList, outputDataList);
    std::cout << "====== GateScore ======" << std::endl;
    EXPECT_TRUE(resultCmp<T>(goldenData, (T *)gatingScoreData->data(), 0.005f));
#endif
}

TEST_F(GenGatedScore, gated_score_fp16_s8k_prefill) {
    std::vector<int64_t> bnsh = {4, 128, 8192, 7168};
    GenGatedScoreEntryPrefill<float16>(bnsh);
}