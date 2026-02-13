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
template<typename T>
void TestShmemMoeDispatch(OpTestParam& testParam, std::string& goldenDir)
{
    constexpr size_t paramsSize = 5;
    auto [batchSize, hiddenSize, routedNum, topK, typeNum] = GetParams<paramsSize>(goldenDir + "/params.bin");
    DataType dType = GetDataTypeNum(typeNum);
    int32_t expertNumPerRank = routedNum / testParam.rankSize;
    Shape tokenTensorShape{batchSize, hiddenSize};
    Shape tokenExpertTableShape{batchSize, topK};
    int32_t expandXRowShape = std::min(static_cast<int32_t>(batchSize) *
        static_cast<int32_t>(topK) * testParam.rankSize, static_cast<int32_t>(batchSize) * routedNum);
    Shape expandXShape{expandXRowShape, hiddenSize};
    Shape combineInfoShape{expandXRowShape, 64};
    Tensor tokenTensor(dType, tokenTensorShape, "tokenTensor");
    Tensor tokenExpertTable(DataType::DT_INT32, tokenExpertTableShape, "tokenExpertTable");
    Tensor expertTokenNums(DataType::DT_INT32, {expertNumPerRank}, "expertTokenNums");
    Tensor expandX(dType, expandXShape, "expandX");
    Tensor combineInfo(DataType::DT_INT32, combineInfoShape, "combineInfo");
    Tensor recvCounts(DataType::DT_INT32, {1}, "recvCounts");
    int64_t expandXEleNum = expandXShape[0] * expandXShape[1];
    int64_t combineInfoEleNum = combineInfoShape[0] * combineInfoShape[1];
    std::string xPath = goldenDir+ "/x_rank_" + std::to_string(testParam.rankId) + ".bin";
    std::vector<T> tokenTensorPtr = ReadToVector<T>(xPath, tokenTensorShape);
    std::string expertIdsPath = goldenDir + "/expert_ids_rank_" + std::to_string(testParam.rankId) + ".bin";
    std::vector<int32_t> tokenExpertTablePtr = ReadToVector<int32_t>(expertIdsPath, tokenExpertTableShape);
    FUNCTION("MoeDispatch", {tokenTensor, tokenExpertTable}, {expandX, expertTokenNums, combineInfo, recvCounts}) {
        Distributed::MoeDistributedDispatchV2(tokenTensor, tokenExpertTable, testParam.group,
            static_cast<uint32_t>(testParam.rankSize), routedNum, 0, 0, expandX, combineInfo, expertTokenNums, recvCounts);
    }
    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateTensor<T>(tokenTensor, tokenTensorPtr),
        RawTensorData::CreateTensor<int32_t>(tokenExpertTable, tokenExpertTablePtr),
    });
    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateTensorZero(expandX),
        RawTensorData::CreateTensorZero(expertTokenNums),
        RawTensorData::CreateTensorZero(combineInfo),
        RawTensorData::CreateTensorZero(recvCounts),
    });
    DeviceLauncherConfig config;
    config.runModel = false;
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction(), config);
    auto expandXOutPut = ProgramData::GetInstance().GetOutputData(0);
    EXPECT_TRUE(CompareWithGolden<uint8_t *>(dType, goldenDir + "/y_rank_", expandXEleNum, expandXOutPut->GetDevPtr(), testParam));
    auto expertTokenNumsOutPut = ProgramData::GetInstance().GetOutputData(1);
    EXPECT_TRUE(CompareWithGolden<uint8_t *>(DataType::DT_INT32, goldenDir + "/valid_count_rank_", expertNumPerRank, expertTokenNumsOutPut->GetDevPtr(), testParam));
    auto combineInfoOutPut = ProgramData::GetInstance().GetOutputData(2);
    EXPECT_TRUE(CompareWithGolden<uint8_t *>(DataType::DT_INT32, goldenDir + "/combine_info_rank_", combineInfoEleNum, combineInfoOutPut->GetDevPtr(), testParam));
    auto recvCountsOutPut = ProgramData::GetInstance().GetOutputData(3);
    EXPECT_TRUE(CompareWithGolden<uint8_t *>(DataType::DT_INT32, goldenDir + "/recv_counts_rank_", 1, recvCountsOutPut->GetDevPtr(), testParam));
}

template void TestShmemMoeDispatch<int32_t>(OpTestParam& testParam, std::string& goldenDir);
template void TestShmemMoeDispatch<float>(OpTestParam& testParam, std::string& goldenDir);
template void TestShmemMoeDispatch<float16>(OpTestParam& testParam, std::string& goldenDir);
template void TestShmemMoeDispatch<bfloat16>(OpTestParam& testParam, std::string& goldenDir);

} // namespace npu::tile_fwk::Distributed
