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
 * \file test_moe_dispatch.cpp
 * \brief
 */

#include "distributed_op_test_common.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/configs/config_manager.h"
#include "tilefwk/data_type.h"
#include "test_dev_func_runner.h"

namespace npu::tile_fwk::Distributed {

void TestShmemMoeDispatch(OpTestParam &testParam)
{
    constexpr size_t paramsSize = 5;
    auto [batchSize, hiddenSize, routedNum, topK, typeNum] = GetParams<paramsSize>(GetGoldenDir() + "/params.bin");
    DataType dType = GetDataTypeNum(typeNum);
    int32_t totalExpertNum = routedNum;
    int32_t expertNumPerRank = totalExpertNum / testParam.rankSize;
    Shape tokenTensorShape{batchSize, hiddenSize};
    Shape tokenExpertTableShape{batchSize, topK};
    int32_t expandXRowShape = std::min(static_cast<int32_t>(batchSize) *
        static_cast<int32_t>(topK) * testParam.rankSize, static_cast<int32_t>(batchSize) * totalExpertNum);
    Shape expandXShape{expandXRowShape, hiddenSize};
    Shape validCntShape{expertNumPerRank};
    Shape combineInfoShape{expandXRowShape, 3};
    Tensor tokenTensor(dType, tokenTensorShape, "tokenTensor");
    Tensor tokenExpertTable(DataType::DT_INT32, tokenExpertTableShape, "tokenExpertTable");
    Tensor validCnt(DataType::DT_INT32, validCntShape, "validCnt");
    Tensor expandX(dType, expandXShape, "expandX");
    Tensor combineInfo(DataType::DT_INT32, combineInfoShape, "combineInfo");
    int64_t expandXEleNum = expandXShape[0] * expandXShape[1];
    int64_t validCntEleNum = validCntShape[0];
    int64_t combineInfoEleNum = combineInfoShape[0] * combineInfoShape[1];

    using T = npu::tile_fwk::bfloat16;

    std::string xPath = GetGoldenDir() + "/x_rank_" + std::to_string(testParam.rankId) + ".bin";
    std::vector<T> tokenTensorPtr = ReadToVector<T>(xPath, tokenTensorShape);
    std::string expertIdsPath = GetGoldenDir() + "/expert_ids_rank_" + std::to_string(testParam.rankId) + ".bin";
    std::vector<int32_t> tokenExpertTablePtr = ReadToVector<int32_t>(expertIdsPath, tokenExpertTableShape);

    MoeConfig moeConfig{routedNum, expertNumPerRank, testParam.rankSize};
    FUNCTION("MoeDispatch", {tokenTensor, tokenExpertTable}, {expandX, validCnt, combineInfo}) {
        Distributed::MoeDispatch(tokenTensor, tokenExpertTable, expandX, validCnt, combineInfo, testParam.group, moeConfig);
    }

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateTensor<T>(tokenTensor, tokenTensorPtr),
        RawTensorData::CreateTensor<int32_t>(tokenExpertTable, tokenExpertTablePtr),
    });
    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateTensorZero(expandX),
        RawTensorData::CreateTensorZero(validCnt),
        RawTensorData::CreateTensor(combineInfo, std::vector<int32_t>(combineInfoEleNum, -1))
    });

    DeviceLauncherConfig config;
    config.runModel = false;
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction(), config);

    auto expandXOutPut = ProgramData::GetInstance().GetOutputData(0);
    EXPECT_TRUE(CompareWithGolden<uint8_t *>(dType, "/y_rank_", expandXEleNum, expandXOutPut->GetDevPtr(), testParam));
    auto validCntOutPut = ProgramData::GetInstance().GetOutputData(1);
    EXPECT_TRUE(CompareWithGolden<uint8_t *>(DataType::DT_INT32, "/valid_count_rank_", validCntEleNum, validCntOutPut->GetDevPtr(), testParam));
    auto combineInfoOutPut = ProgramData::GetInstance().GetOutputData(2);
    EXPECT_TRUE(CompareWithGolden<uint8_t *>(DataType::DT_INT32, "/combine_info_rank_", combineInfoEleNum, combineInfoOutPut->GetDevPtr(), testParam));
}

} // namespace npu::tile_fwk::Distributed
