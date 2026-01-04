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
 * \file test_allreduce.cpp
 * \brief
 */

#include "distributed_op_test_suite.h"
#include "distributed_op_test_common.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/configs/config_manager.h"
#include "tilefwk/data_type.h"
#include "test_dev_func_runner.h"

namespace npu::tile_fwk {
namespace Distributed {

template<typename T, bool useTwoShot>
void TestShmemAllReduce(OpTestParam &testParam)
{
    constexpr size_t paramsSize = 3;
    auto [row, col, typeNum] = GetParams<paramsSize>(GetGoldenDir() + "/params.bin");
    DataType dType = GetDataTypeNum(typeNum);

    int32_t outSize = row * col;

    Shape shape{row, col};
    Tensor in(dType, shape, "in");
    Tensor out(dType, shape, "out");

    std::vector<T> inPtr = ReadToVector<T>(
        GetGoldenDir() + "/input_rank_" + std::to_string(testParam.rankId) + ".bin", {row, col});

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateTensor<T>(in, inPtr),
    });
    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateTensorZero(out),
    });
    int32_t tileNum1 = 1;
    int32_t tileNum2 = 1;
    int32_t scatterRow = useTwoShot ? row / testParam.rankSize : row;
    FUNCTION("ALLREDUCE", {in}, {out}) {
        TileShape::Current().SetDistTile(
            {scatterRow / tileNum1, tileNum1, scatterRow % tileNum1},
            {col / tileNum2, tileNum2, col % tileNum2},
            {1, testParam.rankSize, 0});
        if (useTwoShot) {
            TwoShotShmemAllReduce(in, in, testParam.group, out);
        } else {
            OneShotShmemAllReduce(in, in, testParam.group, out);
        }
    }
    auto dynAttr = Program::GetInstance().GetLastFunction()->GetDyndevAttribute();
    auto hcclContext = GetHcclContext(dynAttr->commGroupNames);
    DeviceLauncherConfig config;
    config.runModel = false;
    config.hcclContext = hcclContext;
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction(), config);

    auto output = ProgramData::GetInstance().GetOutputData(0);
    EXPECT_TRUE(CompareWithGolden<uint8_t*>(dType, "/output_rank_", outSize, output->GetDevPtr(), testParam));
}
template void TestShmemAllReduce<int32_t, true>(OpTestParam &testParam);
template void TestShmemAllReduce<int32_t, false>(OpTestParam &testParam);
template void TestShmemAllReduce<float, true>(OpTestParam &testParam);
template void TestShmemAllReduce<float, false>(OpTestParam &testParam);
template void TestShmemAllReduce<float16, true>(OpTestParam &testParam);
template void TestShmemAllReduce<float16, false>(OpTestParam &testParam);
template void TestShmemAllReduce<bfloat16, true>(OpTestParam &testParam);
template void TestShmemAllReduce<bfloat16, false>(OpTestParam &testParam);

} // namespace Distributed
} // namespace npu::tile_fwk