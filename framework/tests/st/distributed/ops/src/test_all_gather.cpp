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
 * \file test_all_gather.cpp
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

template<typename T>
void TestDynAllGather(OpTestParam &testParam)
{
    constexpr size_t paramsSize = 3;
    auto [M, N, typeNum] = GetParams<paramsSize>(GetGoldenDir() + "/params.bin");

    DataType dType = GetDataTypeNum(typeNum);

    int32_t outSize = M * N * testParam.rankSize;

    Shape shape{M, N};
    Shape outShape{testParam.rankSize * M, N};
    Tensor in(dType, shape, "in");
    Tensor barrierDummy(DT_INT32, {1, 1}, "barrierDummy");
    Tensor out(dType, outShape, "out");

    std::vector<T> inPtr = ReadToVector<T>(GetGoldenDir() + "/input_rank_" + std::to_string(testParam.rankId) + ".bin", shape);

    int32_t tileNum1 = 8;
    int32_t tileNum2 = 8;
    FUNCTION("ALLGATHER", {in, barrierDummy}, {out}) {
        TileShape::Current().SetDistTile(
            {M / tileNum1, tileNum1, M % tileNum1},
            {N / tileNum2, tileNum2, N % tileNum2},
            {1, testParam.rankSize, 0});
        ShmemAllGather(in, barrierDummy, testParam.group, out);
    }

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateTensor<T>(in, inPtr),
        RawTensorData::CreateTensorZero(barrierDummy)
    });
    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateTensorZero(out)
    });

    auto dynAttr = Program::GetInstance().GetLastFunction()->GetDyndevAttribute();
    auto hcclContext = GetHcclContext(dynAttr->commGroupNames);
    DeviceLauncherConfig config;
    config.runModel = false;
    config.hcclContext = hcclContext;
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction(), config);

    auto outPtr = ProgramData::GetInstance().GetOutputData(0)->GetDevPtr();
    EXPECT_TRUE(CompareWithGolden<uint8_t*>(dType, "/output_rank_", outSize, outPtr, testParam));
}
template void TestDynAllGather<int32_t>(OpTestParam &testParam);
template void TestDynAllGather<float>(OpTestParam &testParam);
template void TestDynAllGather<float16>(OpTestParam &testParam);
template void TestDynAllGather<bfloat16>(OpTestParam &testParam);

} // namespace Distributed
} // namespace npu::tile_fwk
