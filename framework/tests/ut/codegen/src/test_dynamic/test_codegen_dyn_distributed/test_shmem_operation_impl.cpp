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
 * \file test_shmem_operation_impl.cpp
 * \brief Unit test for codegen.
 */

#include <vector>
#include <string>

#include <gtest/gtest.h>

#include "interface/function/function.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/configs/config_manager.h"
#include "codegen/codegen.h"
#include "codegen/cloudnpu/codegen_cloudnpu.h"
#include "test_codegen_common.h"
#include "tilefwk/tilefwk_op.h"

namespace npu::tile_fwk::Distributed {

class TestDistributedShmemImpl : public ::testing::Test {
private:
    void CreateShmemTensors(const char* group, uint32_t worldSize, const DataType shmemDataType,
        const Shape& shmemDataShape, Tensor& shmemData, Tensor& shmemSignal)
    {
        LOOP("CreateShmemTensor", FunctionType::DYNAMIC_LOOP, index, LoopRange(1)) {
            (void)index;
            CreateShmemData(group, worldSize, shmemDataType, shmemDataShape, shmemData);
            CreateShmemSignal(group, shmemData, shmemSignal);
        }
    }

    DataType GetType(const Tensor& in)
    {
        DataType shmemDataType = in.GetDataType();
        if ((shmemDataType == DT_BF16) || (shmemDataType == DT_FP16)) {
            shmemDataType = DT_FP32;
        }
        return shmemDataType;
    }

public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}

    void SetUp() override
    {
        Program::GetInstance().Reset();
        config::Reset();
        config::SetHostOption(COMPILE_STAGE, HOST_COMPILE_END);
        config::SetPlatformConfig(KEY_ENABLE_COST_MODEL, false);
    }

    void TearDown() override {}
};

std::string GetFunctionRawName(const std::string& functionName)
{
    std::string functionRawName = FUNCTION_PREFIX + functionName + SUB_FUNC_SUFFIX;
#if ENABLE_HIDDENLOOP
    functionRawName += HIDDEN_FUNC_SUFFIX;
#endif
    return functionRawName;
}

TEST_F(TestDistributedShmemImpl, TestAllGather)
{
    const char *group = "hcom123";
    uint32_t worldSize = 4;
    Tensor in(DT_FP16, {16, 32}, "in");
    Tensor out(DT_FP16, {64, 32}, "out");
    Shape shmemDataShape{worldSize, 16, 32};
    FUNCTION("ALLGATHER", {in}, {out}) {
        TileShape::Current().SetVecTile({16, 32});
        Tensor shmemData;
        Tensor shmemSignal;
        CreateShmemTensors(group, worldSize, DT_FP16, shmemDataShape, shmemData, shmemSignal);
        AllGather(in, in, group, shmemData, shmemSignal, out);
    }

    std::string functionRawName = GetFunctionRawName("CreateShmemTensor");
    auto function = Program::GetInstance().GetFunctionByRawName(functionRawName);
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenCloudNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

TEST_F(TestDistributedShmemImpl, TestReduceScatter)
{
    const char *group = "hcom123";
    uint32_t worldSize = 4;
    Tensor in(DT_FP16, {64, 256}, "in");
    Tensor out(DT_FP16, {16, 256}, "out");
    Shape shmemDataShape = {1, 64 / 4, 256};
    FUNCTION("REDUCESCATTER", {in}, {out}) {
        TileShape::Current().SetVecTile({64, 256});
        Tensor shmemData;
        Tensor shmemSignal;
        DataType shmemDataType = GetType(in);
        CreateShmemTensors(group, worldSize, shmemDataType, shmemDataShape, shmemData, shmemSignal);
        ReduceScatter(in, in, group, shmemData, shmemSignal, DistReduceType::DIST_REDUCE_ADD, out);
    }

    std::string functionRawName = GetFunctionRawName("CreateShmemTensor");
    auto function = Program::GetInstance().GetFunctionByRawName(functionRawName);
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenCloudNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

TEST_F(TestDistributedShmemImpl, TestTwoShotAllReduce)
{
    const char *group = "hcom123";
    uint32_t worldSize = 4;
    Tensor in(DT_FP16, {64, 256}, "in");
    Tensor out(DT_FP16, {64, 256}, "out");
    Shape shmemDataShape = {worldSize, 64 / 4, 256};
    FUNCTION("ALLREDUCE", {in}, {out}) {
        TileShape::Current().SetVecTile({64, 256});
        Tensor shmemData;
        Tensor shmemSignal;
        DataType shmemDataType = GetType(in);
        CreateShmemTensors(group, worldSize, shmemDataType, shmemDataShape, shmemData, shmemSignal);
        TwoShotAllReduce(in, in, group, shmemData, shmemSignal, out);
    }

    std::string functionRawName = GetFunctionRawName("CreateShmemTensor");
    auto function = Program::GetInstance().GetFunctionByRawName(functionRawName);
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenCloudNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

TEST_F(TestDistributedShmemImpl, TestOneShotAllReduce)
{
    const char *group = "hcom123";

    uint32_t worldSize = 4;
    Tensor in(DT_FP16, {64, 256}, "in");
    Tensor out(DT_FP16, {64, 256}, "out");
     Shape shmemDataShape = {1, 64, 256};
    FUNCTION("ALLREDUCE", {in}, {out}) {
        TileShape::Current().SetVecTile({64, 256});
        Tensor shmemData;
        Tensor shmemSignal;
        DataType shmemDataType = GetType(in);
        CreateShmemTensors(group, worldSize, shmemDataType, shmemDataShape, shmemData, shmemSignal);
        OneShotAllReduce(in, in, group, shmemData, shmemSignal, out);
    }

    std::string functionRawName = GetFunctionRawName("CreateShmemTensor");
    auto function = Program::GetInstance().GetFunctionByRawName(functionRawName);
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenCloudNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

TEST_F(TestDistributedShmemImpl, TestShmemDataSet)
{
    Tensor predToken(DT_INT32, {1, 1}, "predToken");
    Tensor in(DT_BF16, {4, 1, 256, 102400}, "in");
    Tensor out(DT_INT32, {1, 1}, "out");

    std::string functionName = "ShmemDataSet";
    FUNCTION(functionName + "Main", {in}, {out}) {
        LOOP(functionName, FunctionType::DYNAMIC_LOOP, index, LoopRange(1)) {
            (void)index;
            out = ShmemDataSet(predToken, in);
        }
    }

    std::string functionRawName = GetFunctionRawName(functionName);
    auto function = Program::GetInstance().GetFunctionByRawName(functionRawName);
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenCloudNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

TEST_F(TestDistributedShmemImpl, TestShmemSignalSet)
{
    int64_t worldSize = 4;
    Tensor predToken(DT_INT32, {1, 1}, "predToken");
    Tensor in(DT_BF16, {worldSize, worldSize, 1, 256, 102400}, "in");
    Tensor out(DT_INT32, {1, 1}, "out");

    std::string functionName = "ShmemSignalSet";
    FUNCTION(functionName + "Main", {in}, {out}) {
        LOOP(functionName, FunctionType::DYNAMIC_LOOP, index, LoopRange(1)) {
            (void)index;
            out = ShmemSignalSet(predToken, in);
        }
    }

    std::string functionRawName = GetFunctionRawName(functionName);
    auto function = Program::GetInstance().GetFunctionByRawName(functionRawName);
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenCloudNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

TEST_F(TestDistributedShmemImpl, TestShmemBarrier)
{
    const char* group = "hcom123";
    uint32_t worldSize = 4;
    int64_t row = 16;
    int64_t col = 32;
    Tensor shmemSignal(DT_INT32, {worldSize, worldSize, 1, row, col}, "in");
    Tensor out(DT_INT32, {1, 1}, "out");
    std::string functionName = "ShmemBarrier";
    FUNCTION(functionName + "Main", {shmemSignal}, {out}) {
        TileShape::Current().SetVecTile({row, col});
        Tensor predToken(DT_INT32, {1, 1}, "predToken");
        LOOP(functionName, FunctionType::DYNAMIC_LOOP, index, LoopRange(1)) {
            (void) index;
            out = ShmemBarrier(predToken, shmemSignal, group, worldSize);
        }
    }

    std::string functionRawName = GetFunctionRawName(functionName);
    auto function = Program::GetInstance().GetFunctionByRawName(functionRawName);
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenCloudNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

TEST_F(TestDistributedShmemImpl, TestShmemGetGm2Ub)
{
    int64_t row = 4;
    int64_t col = 64;
    Tensor dummy(DT_INT32, {1, 1}, "dummy");
    Tensor shmemData(DT_INT32, {1, 1, row, col}, "in");
    Tensor out(DT_INT32, {row, col}, "out");
    std::string functionName = "ShmemGetGm2Ub";
    FUNCTION(functionName + "Main", {dummy, shmemData}, {out}) {
        TileShape::Current().SetVecTile({row, col});
        LOOP(functionName, FunctionType::DYNAMIC_LOOP, index, LoopRange(1)) {
            (void) index;
            out = ShmemGetGm2Ub(dummy, shmemData);
        }
    }

    std::string functionRawName = GetFunctionRawName(functionName);
    auto function = Program::GetInstance().GetFunctionByRawName(functionRawName);
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenCloudNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

}
