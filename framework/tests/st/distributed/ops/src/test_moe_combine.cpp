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

#include "distributed_op_test_suite.h"
#include "distributed_op_test_common.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/configs/config_manager.h"
#include "tilefwk/data_type.h"
#include "test_dev_func_runner.h"

namespace npu::tile_fwk::Distributed {

void TestShmemMoeCombine(OpTestParam& testParam)
{
    constexpr size_t paramsSize = 5;
    auto [batchSize, hiddenSize, totalExpertNum, topK, dtype_num] =
        GetParams<paramsSize>(GetGoldenDir() + "/params.bin");

    DataType dType = GetDataTypeNum(dtype_num);

    int64_t row = std::min(topK * batchSize * testParam.rankSize, batchSize * totalExpertNum);
    Shape inShape{row, hiddenSize};
    Shape combineInfoShape{row, 3};
    Shape scaleShape{batchSize, topK};
    Shape outShape{batchSize, hiddenSize};

    Tensor in(dType, inShape, "in");
    Tensor combineInfo(DataType::DT_INT32, combineInfoShape, "combineInfo");
    Tensor scale(DataType::DT_FP32, scaleShape, "scale");
    Tensor out(dType, outShape, "out");

    using T = npu::tile_fwk::bfloat16;
    std::string dispatchPath = GetGoldenDir() + "/dispatch";
    std::vector<T> inPtr = ReadToVector<T>(
        dispatchPath + "/y_rank_" + std::to_string(testParam.rankId) + ".bin", inShape);
    std::vector<int32_t> combineInfoPtr = ReadToVector<int32_t>(
        dispatchPath + "/combine_info_rank_" + std::to_string(testParam.rankId) + ".bin", combineInfoShape);
    std::vector<float> scalePtr = ReadToVector<float>(
        dispatchPath + "/scale_rank_" + std::to_string(testParam.rankId) + ".bin", scaleShape);

    FUNCTION("Moe_Combine", {in, combineInfo, scale}, {out}) {
        ShmemMoeCombine(in, combineInfo, scale, testParam.group, testParam.rankSize, totalExpertNum, out);
    }

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateTensor<T>(in, inPtr),
        RawTensorData::CreateTensor<int32_t>(combineInfo, combineInfoPtr),
        RawTensorData::CreateTensor<float>(scale, scalePtr)
    });
    ProgramData::GetInstance().AppendOutputs({RawTensorData::CreateTensorZero(out)});

    auto dynAttr = Program::GetInstance().GetLastFunction()->GetDyndevAttribute();
    auto hcclContext = GetHcclContext(dynAttr->commGroupNames);
    DeviceLauncherConfig config;
    config.runModel = false;
    config.hcclContext = hcclContext;
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction(), config);

    int64_t outEleNum = outShape[0] * outShape[1];
    auto outPtr = ProgramData::GetInstance().GetOutputData(0)->GetDevPtr();
    if (batchSize == 256) { // bs=256 暂时不支持零误差一致
        EXPECT_TRUE(CompareWithGolden<uint8_t*>(dType, "/y_rank_", outEleNum, outPtr, testParam));
    } else {
        EXPECT_TRUE(CompareWithGolden<uint8_t*>(dType, "/y_rank_", outEleNum, outPtr, testParam, 0));
    }
}

} // namespace tile_fwk::Distributed