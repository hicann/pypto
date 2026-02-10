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
 * \file test_moe_combine.cpp
 * \brief
 */

#include "distributed_op_test_common.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/configs/config_manager.h"
#include "tilefwk/data_type.h"
#include "test_dev_func_runner.h"

namespace npu::tile_fwk::Distributed {

template<typename T>
void TestMoeDistributedCombine(OpTestParam& testParam, std::string& goldenDir)
{
    constexpr size_t paramsSize = 6;
    auto [batchSize, hiddenSize, moeExpertNum, topK, dtype_num, useV2] = GetParams<paramsSize>(goldenDir  + "/params.bin");

    DataType dType = GetDataTypeNum(dtype_num);

    int64_t row = std::min(topK * batchSize * testParam.rankSize, batchSize * moeExpertNum);
    Shape inShape{row, hiddenSize};
    Shape combineInfoShape{row, 3};
    Shape recvCountsShape{1};
    Shape scaleShape{batchSize, topK};
    Shape outShape{batchSize, hiddenSize};

    Tensor expandX(dType, inShape, "expandX");
    Tensor assistInfoForCombine(DataType::DT_INT32, combineInfoShape, "assistInfoForCombine");
    Tensor recvCounts(DataType::DT_INT32, recvCountsShape, "recvCounts");
    Tensor expertScales(DataType::DT_FP32, scaleShape, "expertScales");
    Tensor out(dType, outShape, "out");

    std::string dispatchPath = goldenDir + "/dispatch";
    std::vector<T> expandXPtr = ReadToVector<T>(
        dispatchPath + "/y_rank_" + std::to_string(testParam.rankId) + ".bin", inShape);
    std::vector<int32_t> assistInfoForCombinePtr = ReadToVector<int32_t>(
        dispatchPath + "/combine_info_rank_" + std::to_string(testParam.rankId) + ".bin", combineInfoShape);
    std::vector<int32_t> recvCountsPtr = ReadToVector<int32_t>(
        dispatchPath + "/recv_counts_rank_" + std::to_string(testParam.rankId) + ".bin", recvCountsShape);
    std::vector<float> expertScalesPtr = ReadToVector<float>(
        dispatchPath + "/scale_rank_" + std::to_string(testParam.rankId) + ".bin", scaleShape);

    using CombineFunc = std::function<void(const Tensor&, const Tensor&, const Tensor&, const Tensor&, const char*,
        uint32_t, uint32_t, uint32_t, uint32_t, Tensor&)>;
    CombineFunc func = (useV2 == 1) ? MoeDistributedCombineV2 : MoeDistributedCombine;

    FUNCTION("MoeDistributedCombineMain", {expandX, assistInfoForCombine, recvCounts, expertScales}, {out}) {
        func(expandX, assistInfoForCombine, recvCounts, expertScales, testParam.group,
            testParam.rankSize, moeExpertNum, 0, 0, out);
    }

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateTensor<T>(expandX, expandXPtr),
        RawTensorData::CreateTensor<int32_t>(assistInfoForCombine, assistInfoForCombinePtr),
        RawTensorData::CreateTensor<int32_t>(recvCounts, recvCountsPtr),
        RawTensorData::CreateTensor<float>(expertScales, expertScalesPtr)
    });
    ProgramData::GetInstance().AppendOutputs({RawTensorData::CreateTensorZero(out)});

    DeviceLauncherConfig config;
    config.runModel = false;
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction(), config);

    int64_t outEleNum = outShape[0] * outShape[1];
    auto outPtr = ProgramData::GetInstance().GetOutputData(0)->GetDevPtr();

    EXPECT_TRUE(CompareWithGolden<uint8_t*>(dType, goldenDir + "/out_rank_", outEleNum, outPtr, testParam));
}

template void TestMoeDistributedCombine<int32_t>(OpTestParam& testParam, std::string& goldenDir);
template void TestMoeDistributedCombine<float>(OpTestParam& testParam, std::string& goldenDir);
template void TestMoeDistributedCombine<float16>(OpTestParam& testParam, std::string& goldenDir);
template void TestMoeDistributedCombine<bfloat16>(OpTestParam& testParam, std::string& goldenDir);

} // namespace tile_fwk::Distributed