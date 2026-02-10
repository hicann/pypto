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
 * \file test_allreduce_add_allreduce.cpp
 * \brief
 */

#include "distributed_op_test_suite.h"
#include "distributed_op_test_common.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/configs/config_manager.h"
#include "tilefwk/data_type.h"
#include "test_dev_func_runner.h"
#include "tilefwk/symbolic_distributed.h"

namespace npu::tile_fwk::Distributed {

void LoopAllReduce1(const Tensor& in, Tensor& allReduceOut, const OpTestParam& testParam, int32_t row, int32_t col)
{
    LOOP("AllReduce1", FunctionType::DYNAMIC_LOOP, allReduce1Index, LoopRange(0, 1, 1)) {
        (void)allReduce1Index;
        Tensor shmemData;
        Tensor shmemSignal;
        DataType shmemDataType = in.GetDataType();
        Shape shmemDataShape {1, row, col};
        if ((shmemDataType == DT_BF16) || (shmemDataType == DT_FP16)) {
            shmemDataType = DT_FP32;
        }
        LOOP("CreateShmemTensor", FunctionType::DYNAMIC_LOOP, index, LoopRange(1)) {
            (void)index;
            CreateShmemData(testParam.group, testParam.rankSize, shmemDataType, shmemDataShape, shmemData);
            CreateShmemSignal(testParam.group, shmemData, shmemSignal);
        }
        TileShape::Current().SetVecTile(row, col);
        OneShotAllReduce(in, in, testParam.group, shmemData, shmemSignal, allReduceOut);
    }
}

void LoopAdd(const Tensor& allReduceOut, Tensor& addOut)
{
    LOOP("Add", FunctionType::DYNAMIC_LOOP, index, LoopRange(0, 1, 1)) {
        (void)index;
        TileShape::Current().SetVecTile({128, 256});
        addOut = npu::tile_fwk::Add(allReduceOut, allReduceOut);
    }
}

void LoopCreateShmemTensor(const Tensor& addOut, Tensor& shmemBarrier1ShmemSignal, Tensor& shmemBarrier2ShmemSignal,
    Tensor& allReduce2ShmemData, Tensor& allReduce2ShmemSignal, const OpTestParam& testParam, int32_t row, int32_t col)
{
    LOOP("CreateShmemTensor", FunctionType::DYNAMIC_LOOP, index, LoopRange(1)) {
        (void)index;
        uint32_t worldSize = static_cast<uint32_t>(testParam.rankSize);
        CreateShmemData(testParam.group, worldSize, DT_INT32, Shape{1, 1, 1, 8}, shmemBarrier1ShmemSignal, 1);
        CreateShmemData(testParam.group, worldSize, DT_INT32, Shape{1, 1, 1, 8}, shmemBarrier2ShmemSignal, 1);
        TileShape::Current().SetVecTile(row, col);
        Shape allReduce2ShmemDataShape = {1, addOut.GetShape(0), addOut.GetShape(1)};
        DataType allReduce2ShmemDataType = addOut.GetDataType();
        if ((allReduce2ShmemDataType == DT_BF16) || (allReduce2ShmemDataType == DT_FP16)) {
            allReduce2ShmemDataType = DT_FP32;
        }
        CreateShmemData(testParam.group, worldSize, allReduce2ShmemDataType, allReduce2ShmemDataShape, allReduce2ShmemData);
        CreateShmemSignal(testParam.group, allReduce2ShmemData, allReduce2ShmemSignal);
    }
}

void LoopAllReduce2(const Tensor& addOut, Tensor& shmemBarrier1ShmemSignal, Tensor& shmemBarrier2ShmemSignal,
    Tensor& allReduce2ShmemData, Tensor& allReduce2ShmemSignal, Tensor& out, const OpTestParam& testParam, int32_t row,
    int32_t col, std::string group)
{
    LOOP("AllReduce2", FunctionType::DYNAMIC_LOOP, allReduce2Index, LoopRange(0, 1, 1)) {
        (void)allReduce2Index;

        Tensor memSetOut(DT_INT32, {addOut.GetShape(0), addOut.GetShape(1)}, "memSetOut");

        SymbolicScalar thisRank = GetHcclRankId(group);
        TileShape::Current().SetVecTile({1, 8});
        auto barrier1Out = ShmemBarrier(addOut, shmemBarrier1ShmemSignal, testParam.group, static_cast<uint32_t>(testParam.rankSize));
        TileShape::Current().SetVecTile(row, col);
        auto allReduce2ShmemDataTile = View(allReduce2ShmemData, {1, 1, addOut.GetShape(0), addOut.GetShape(1)},
            std::vector<SymbolicScalar>{thisRank, 0, 0, 0});
        auto memSetDataOut = ShmemDataSet(barrier1Out, allReduce2ShmemDataTile);
        auto allReduce2ShmemSignalTile = View(allReduce2ShmemSignal, {1, 1, 1, addOut.GetShape(0), addOut.GetShape(1)},
            std::vector<SymbolicScalar>{thisRank, 0, 0, 0, 0});
        auto memSetSignalOut = ShmemSignalSet(barrier1Out, allReduce2ShmemSignalTile);
        memSetOut = Nop({memSetDataOut, memSetSignalOut});
        TileShape::Current().SetVecTile({1, 8});
        auto barrier2Out = ShmemBarrier(memSetOut, shmemBarrier2ShmemSignal, testParam.group, static_cast<uint32_t>(testParam.rankSize));
        TileShape::Current().SetVecTile(row, col);
        OneShotAllReduce(barrier2Out, addOut, testParam.group, allReduce2ShmemData, allReduce2ShmemSignal, out);
    }
}

void FuncAllReduceAddAllReduce(const Tensor& in, Tensor& out, const OpTestParam& testParam, int32_t row, int32_t col)
{
    FUNCTION("AllReduceAddAllReduce", {in}, {out}) {
        Tensor allReduceOut(in.GetDataType(), in.GetShape(), "allReduceOut");
        Tensor addOut(in.GetDataType(), in.GetShape(), "addOut");
        LoopAllReduce1(in, allReduceOut, testParam, row, col);
        LoopAdd(allReduceOut, addOut);
        Tensor shmemBarrier1ShmemSignal;
        Tensor shmemBarrier2ShmemSignal;
        Tensor allReduce2ShmemData;
        Tensor allReduce2ShmemSignal;
        LoopCreateShmemTensor(addOut, shmemBarrier1ShmemSignal, shmemBarrier2ShmemSignal, allReduce2ShmemData,
            allReduce2ShmemSignal, testParam, row, col);
        LoopAllReduce2(addOut, shmemBarrier1ShmemSignal, shmemBarrier2ShmemSignal, allReduce2ShmemData,
            allReduce2ShmemSignal, out, testParam, row, col, std::string(testParam.group));
    };
}

template<typename T>
void TestAllReduceAddAllReduce(OpTestParam &testParam, std::string& goldenDir)
{
    constexpr size_t paramsSize = 3;
    auto [row, col, typeNum] = GetParams<paramsSize>(goldenDir + "/params.bin");

    Shape shape{row, col};
    DataType dType = GetDataTypeNum(typeNum);
    Tensor in(dType, shape, "in");
    Tensor out(dType, shape, "out");

    std::vector<T> inPtr = ReadToVector<T>(goldenDir +"/input_rank_" + std::to_string(testParam.rankId) + ".bin",
        shape);

    ProgramData::GetInstance().AppendInputs({RawTensorData::CreateTensor<T>(in, inPtr)});
    ProgramData::GetInstance().AppendOutputs({RawTensorData::CreateTensorZero(out)});

    FuncAllReduceAddAllReduce(in, out, testParam, row, col);

    RunTest();
    auto output = ProgramData::GetInstance().GetOutputData(0);
    int32_t outSize = row * col;
    EXPECT_TRUE(CompareWithGolden<uint8_t*>(dType, goldenDir + "/out_rank_", outSize, output->GetDevPtr(), testParam));
}

template void TestAllReduceAddAllReduce<int32_t>(OpTestParam& testParam, std::string& goldenDir);
template void TestAllReduceAddAllReduce<float>(OpTestParam& testParam, std::string& goldenDir);
template void TestAllReduceAddAllReduce<float16>(OpTestParam& testParam, std::string& goldenDir);
template void TestAllReduceAddAllReduce<bfloat16>(OpTestParam& testParam, std::string& goldenDir);


} // namespace npu::tile_fwk::Distributed