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
 * \file test_allgather_attn_post_reducescatter.cpp
 * \brief
 */

#include "distributed_op_test_common.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/configs/config_manager.h"
#include "tilefwk/data_type.h"
#include "test_dev_func_runner.h"

namespace npu::tile_fwk {
namespace Distributed {
std::tuple<Tensor, Tensor, Tensor, Tensor> InitializeTestData(OpTestParam &testParam) {
    constexpr size_t paramsSize = 7;
    auto [b, s, n, kvLoraRank, vHeadDim, h, typeNum] = GetParams<paramsSize>(GetGoldenDir() + "/params.bin");
    DataType dtype = GetDataTypeNum(typeNum);

    Shape agInShape = {b * n * s / testParam.rankSize, kvLoraRank};
    Shape wLoraShape = {n, kvLoraRank, vHeadDim};
    Shape wOutShape = {n * vHeadDim, h};
    Shape outShape = {b * s / testParam.rankSize, h};

    Tensor agIn(dtype, agInShape, "agIn");
    Tensor wLora(dtype, wLoraShape, "wLora");
    Tensor wOut(dtype, wOutShape, "wOut");
    Tensor out(dtype, outShape, "out");

    std::vector<bfloat16> agInPtr =
        ReadToVector<bfloat16>(GetGoldenDir() + "/ag_in_rank_" + std::to_string(testParam.rankId) + ".bin", agInShape);
    std::vector<bfloat16> wLoraPtr =
        ReadToVector<bfloat16>(GetGoldenDir() + "/w_lora_rank_" + std::to_string(testParam.rankId) + ".bin", wLoraShape);
    std::vector<bfloat16> wOutPtr =
        ReadToVector<bfloat16>(GetGoldenDir() + "/w_out_rank_" + std::to_string(testParam.rankId) + ".bin", wOutShape);

    ProgramData::GetInstance().AppendInputs({RawTensorData::CreateTensor<bfloat16>(agIn, agInPtr)});
    ProgramData::GetInstance().AppendInputs({RawTensorData::CreateTensor<bfloat16>(wLora, wLoraPtr)});
    ProgramData::GetInstance().AppendInputs({RawTensorData::CreateTensor<bfloat16>(wOut, wOutPtr)});
    ProgramData::GetInstance().AppendOutputs({RawTensorData::CreateTensorZero(out)});
    return {agIn, wLora, wOut, out};
}

void TestAllGatherAttentionPostReducescatter(OpTestParam &testParam) {
    constexpr size_t paramsSize = 7;
    auto [b, s, n, kvLoraRank, vHeadDim, h, typeNum] = GetParams<paramsSize>(GetGoldenDir() + "/params.bin");
    DataType dtype = GetDataTypeNum(typeNum);
    auto [agIn, wLora, wOut, out] = InitializeTestData(testParam);

    FUNCTION("ALLGATHER_ATTNPOST_REDUCESCATTER", {agIn, wLora, wOut}, {out}) {
        Tensor agOut(dtype, {b * n * s, kvLoraRank}, "agOut");
        LOOP("ALLGATHER", FunctionType::DYNAMIC_LOOP, unusedDynRankId, LoopRange(1)) {
            (void) unusedDynRankId;
            TileShape::Current().SetVecTile({64, kvLoraRank});
            AllGather(agIn, agIn, testParam.group, static_cast<uint32_t>(testParam.rankSize), agOut);
        }
        Tensor attnOut(dtype, {b * s, h}, "attnOut");
        LOOP("ATTNPOST", FunctionType::DYNAMIC_LOOP, batchId, LoopRange(1)) {
            (void) batchId;
            TileShape::Current().SetVecTile({4, 16, 1, kvLoraRank});
            Tensor attnIn = Reshape(agOut, {b, n, s, kvLoraRank});
            TileShape::Current().SetVecTile({4, 16, 1, kvLoraRank});
            Tensor attnRes0 = Transpose(attnIn, {1, 2});
            TileShape::Current().SetVecTile({4, 1, 32, std::min(512, kvLoraRank)});
            Tensor attnRes1 = Reshape(attnRes0, {b * s, n, kvLoraRank});
            TileShape::Current().SetVecTile({4, 16, kvLoraRank});
            Tensor t2Res = Transpose(attnRes1, {0, 1});
            TileShape::Current().SetCubeTile({16, 16}, {256, 256}, {128, 128});
            // {n, b * s, kvLoraRank} @ {n, kvLoraRank, vHeadDim} = {n, b * s, vHeadDim}
            Tensor fp32Bmm4Res =  Matrix::BatchMatmul(DataType::DT_FP32, t2Res, wLora);
            Tensor bmm4Res = Cast(fp32Bmm4Res, dtype);
            TileShape::Current().SetVecTile({32, 4, vHeadDim}); // 必须切，但是尾轴不能切
            Tensor t3Res = Transpose(bmm4Res, {0, 1}); // [bs,n,vHeadDim]
            TileShape::Current().SetVecTile({4, 32, vHeadDim});
            Tensor r2Res = Reshape(t3Res, {b * s, n * vHeadDim});
            TileShape::Current().SetCubeTile({16, 16}, {256, 256}, {128, 128});
            // {b * s, n * vHeadDim} @ {n * vHeadDim, h} = {b * s, h}
            attnOut = Matrix::Matmul<false, false>(dtype, r2Res, wOut);
        }
        LOOP("REDUCESCATTER", FunctionType::DYNAMIC_LOOP, unusedIndex, LoopRange(1)) {
            (void) unusedIndex;
            Tensor predToken(DT_INT32, {1, 1}, "predToken");
            TileShape::Current().SetVecTile({16, h});
            Distributed::ReduceScatter(predToken, attnOut, testParam.group, static_cast<uint32_t>(testParam.rankSize),
                DistReduceType::DIST_REDUCE_ADD, out);
        }
    }
    auto dynAttr = Program::GetInstance().GetLastFunction()->GetDyndevAttribute();
    auto hcclContext = GetHcclContext(dynAttr->commGroupNames);
    DeviceLauncherConfig config;
    config.runModel = false;
    config.hcclContext = hcclContext;
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction(), config);

    auto output = ProgramData::GetInstance().GetOutputData(0);
    int32_t outSize = b * s / testParam.rankSize * h;
    EXPECT_TRUE(CompareWithGolden<uint8_t*>(dtype, "/rs_out_rank_", outSize, output->GetDevPtr(), testParam, 0.1f));
}

} // namespace Distributed
} // namespace npu::tile_fwk