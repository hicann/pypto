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
 * \file test_reduce_scatter.cpp
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

template<typename T>
void TestShmemReduceScatter(OpTestParam &testParam)
{
    constexpr size_t paramsSize = 5;
    auto [row, col, typeNum, tileRow, tileCol] = GetParams<paramsSize>(GetGoldenDir() + "/params.bin");
    int rowOut = row / testParam.rankSize;
    DataType dType = GetDataTypeNum(typeNum);
    Tensor in(dType, {row, col}, "in");
    Tensor out(dType, {rowOut, col}, "out");

    std::vector<T> inData = ReadToVector<T>(
        GetGoldenDir() + "/input_rank_" + std::to_string(testParam.rankId) + ".bin", {row, col});

    FUNCTION("ShmemReduceScatter", {in}, {out}) {
        LOOP("LOOP", FunctionType::DYNAMIC_LOOP, idx, LoopRange(1)) {
            (void)idx;
            TileShape::Current().SetVecTile({tileRow, tileCol});
            Tensor predToken(DT_INT32, {1, 1}, "predToken");
            ReduceScatter(predToken, in, testParam.group, static_cast<uint32_t>(testParam.rankSize),
                npu::tile_fwk::Distributed::DistReduceType::DIST_REDUCE_ADD, out);
        }
    }

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateTensor<T>(in, inData),
    });
    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<T>(out, 0),
    });

    auto dynAttr = Program::GetInstance().GetLastFunction()->GetDyndevAttribute();
    auto hcclContext = GetHcclContext(dynAttr->commGroupNames);
    DeviceLauncherConfig config;
    config.runModel = false;
    config.hcclContext = hcclContext;
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction(), config);

    auto outPut = ProgramData::GetInstance().GetOutputData(0);
    EXPECT_TRUE(CompareWithGolden<uint8_t*>(dType, "/output_rank_", rowOut * col, outPut->GetDevPtr(), testParam));
}

template void TestShmemReduceScatter<int32_t>(OpTestParam &testParam);
template void TestShmemReduceScatter<float>(OpTestParam &testParam);
template void TestShmemReduceScatter<float16>(OpTestParam &testParam);
template void TestShmemReduceScatter<bfloat16>(OpTestParam &testParam);
} // namespace Distributed
} // namespace npu::tile_fwk