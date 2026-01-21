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

#include "distributed_op_test_common.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/configs/config_manager.h"
#include "tilefwk/data_type.h"
#include "test_dev_func_runner.h"

namespace npu::tile_fwk {
namespace Distributed {

template<typename T>
void TestAllGather(OpTestParam& testParam)
{
    constexpr size_t paramsSize = 5;
    auto [row, col, typeNum, tileRow, tileCol] = GetParams<paramsSize>(GetGoldenDir() + "/params.bin");

    DataType dType = GetDataTypeNum(typeNum);

    int32_t outSize = row * col * testParam.rankSize;

    Shape shape{row, col};
    Shape outShape{testParam.rankSize * row, col};
    Tensor in(dType, shape, "in");
    Tensor out(dType, outShape, "out");

    std::vector<T> inPtr = ReadToVector<T>(GetGoldenDir() + "/input_rank_" + std::to_string(testParam.rankId) + ".bin", shape);
    
    Shape shmemDataShape{testParam.rankSize, row, col};
    FUNCTION("ALLGATHER", {in}, {out}) {
        TileShape::Current().SetVecTile({tileRow, tileCol});
        Tensor shmemData;
        Tensor shmemSignal;
        LOOP("CreateShmemTensor", FunctionType::DYNAMIC_LOOP, index, LoopRange(1)) {
            (void)index;
            CreateShmemData(testParam.group, testParam.rankSize, dType, shmemDataShape, shmemData);
            CreateShmemSignal(testParam.group, shmemData, shmemSignal);
        }
        AllGather(in, in, testParam.group, shmemData, shmemSignal, out);
    }

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateTensor<T>(in, inPtr)
    });
    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateTensorZero(out)
    });

    RunTest();
    auto outPtr = ProgramData::GetInstance().GetOutputData(0)->GetDevPtr();
    EXPECT_TRUE(CompareWithGolden<uint8_t*>(dType, "/output_rank_", outSize, outPtr, testParam));

}

template void TestAllGather<int32_t>(OpTestParam& testParam);
template void TestAllGather<float>(OpTestParam& testParam);
template void TestAllGather<float16>(OpTestParam& testParam);
template void TestAllGather<bfloat16>(OpTestParam& testParam);

} // namespace Distributed
} // namespace npu::tile_fwk