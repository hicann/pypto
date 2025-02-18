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
 * \file test_allgather_maymul_reducescatter.cpp
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

void TestDynAllGatherMatmulReducescatter(OpTestParam &testParam)
{
    constexpr size_t paramsSize = 3;
    auto [row, col, typeNum] = GetParams<paramsSize>(GetGoldenDir() + "/params.bin");
    DataType dType = GetDataTypeNum(typeNum);

    Shape shape{row, col};
    Tensor in(dType, shape, "in");

    std::vector<float> inPtr = ReadToVector<float>(GetGoldenDir() + "/input_rank_" +
        std::to_string(testParam.rankId) + ".bin", shape);

    Shape agShape{row * testParam.rankSize, col};

    Shape outShape{row * testParam.rankSize, col};
    Tensor out(dType, outShape, "out");

    ProgramData::GetInstance().AppendInputs({RawTensorData::CreateTensor<float>(in, inPtr)});
    ProgramData::GetInstance().AppendOutputs({RawTensorData::CreateTensorZero(out)});

    FUNCTION("ALLGATHER_and_ALLGATHER", {in}, {out}) {
        Tensor allGatherOut(dType, agShape, "allGatherOut");
        Tensor allGatherMatmul(dType, agShape, "allGatherMatmul");
        Tensor reduceScatterOut(dType, shape, "reduceScatterOut");

        LOOP("ALLGATHER1", FunctionType::DYNAMIC_LOOP, unusedDynRankId, LoopRange(0, 1, 1)) {
            (void)unusedDynRankId;
            TileShape::Current().SetDistTile({row, 1, 0}, {col, 1, 0}, {1, testParam.rankSize, 0});
            auto barrierDummy = Barrier(in, testParam.group);
            ShmemAllGather(in, barrierDummy, testParam.group, allGatherOut);
        }
        TileShape::Current().SetVecTile({256, 128});
        LOOP("ADD", FunctionType::DYNAMIC_LOOP, unusedDynRankId, LoopRange(0, 1, 1)) {
            (void)unusedDynRankId;
            allGatherMatmul = npu::tile_fwk::Add(allGatherOut, allGatherOut);
        }
        LOOP("REDUCESCATTER", FunctionType::DYNAMIC_LOOP, unusedIndex, LoopRange(1)) {
            (void)unusedIndex;
            TileShape::Current().SetDistTile({row, 1, 0}, {col, 1, 0}, {1, testParam.rankSize, 0});
            Distributed::ShmemReduceScatter(allGatherMatmul, testParam.group, DistReduceType::DIST_REDUCE_ADD,
                reduceScatterOut);
        }
        LOOP("ALLGATHER2", FunctionType::DYNAMIC_LOOP, unusedDynRankId, LoopRange(0, 1, 1)) {
           (void)unusedDynRankId;
            TileShape::Current().SetDistTile({row, 1, 0}, {col, 1, 0}, {1, testParam.rankSize, 0});
            auto barrierDummy = Barrier(reduceScatterOut, testParam.group);
            ShmemAllGather(reduceScatterOut, barrierDummy, testParam.group, out);
        }
    }
    auto dynAttr = Program::GetInstance().GetLastFunction()->GetDyndevAttribute();
    auto hcclContext = GetHcclContext(dynAttr->commGroupNames);
    DeviceLauncherConfig config;
    config.runModel = false;
    config.hcclContext = hcclContext;
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction(), config);

    auto output = ProgramData::GetInstance().GetOutputData(0);
    int32_t outSize = row * testParam.rankSize * col;
    EXPECT_TRUE(CompareWithGolden<uint8_t*>(dType, "/double_allgather_rank_", outSize, output->GetDevPtr(), testParam));
}

} // namespace Distributed
} // namespace npu::tile_fwk