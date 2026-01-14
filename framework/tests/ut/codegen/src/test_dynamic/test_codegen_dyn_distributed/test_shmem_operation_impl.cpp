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

namespace npu::tile_fwk::Distributed {

class TestDistributedShmemImpl : public ::testing::Test {
public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}

    void SetUp() override
    {
        Program::GetInstance().Reset();
        config::Reset();
        config::SetPlatformConfig(KEY_ONLY_HOST_COMPILE, true);
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

TEST_F(TestDistributedShmemImpl, TestShmemAllGather)
{
    const char *group = "hcom123";
    uint32_t worldSize = 4;
    Tensor in(DT_FP16, {16, 32}, "in");
    Tensor out(DT_FP16, {64, 32}, "out");
    FUNCTION("ALLGATHER", {in}, {out}) {
        TileShape::Current().SetVecTile({16, 32});
        Tensor predToken(DT_INT32, {1, 1}, "predToken");
        AllGather(predToken, in, group, worldSize, out);
    }

    std::string functionRawName = GetFunctionRawName("L0");
    auto function = Program::GetInstance().GetFunctionByRawName(functionRawName);
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenCloudNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

TEST_F(TestDistributedShmemImpl, TestShmemReduceScatter)
{
    const char *group = "hcom123";

    uint32_t worldSize = 4;
    Tensor in(DT_FP16, {64, 256}, "in");
    Tensor out(DT_FP16, {16, 256}, "out");
    FUNCTION("REDUCESCATTER", {in}, {out}) {
        TileShape::Current().SetVecTile({64, 256});
        Tensor predToken(DT_INT32, {1, 1}, "predToken");
        ReduceScatter(predToken, in, group, worldSize, DistReduceType::DIST_REDUCE_ADD, out);
    }

    std::string functionRawName = GetFunctionRawName("RS");
    auto function = Program::GetInstance().GetFunctionByRawName(functionRawName);
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenCloudNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

TEST_F(TestDistributedShmemImpl, TestTwoShotShmemAllReduce)
{
    const char *group = "hcom123";

    uint32_t worldSize = 4;
    Tensor in(DT_FP16, {64, 256}, "in");
    Tensor out(DT_FP16, {64, 256}, "out");
    FUNCTION("ALLREDUCE", {in}, {out}) {
        TileShape::Current().SetVecTile({64, 256});
        Tensor predToken(DT_INT32, {1, 1}, "predToken");
        TwoShotAllReduce(predToken, in, group, worldSize, out);
    }

    std::string functionRawName = GetFunctionRawName("TwoShotAllReduce");
    auto function = Program::GetInstance().GetFunctionByRawName(functionRawName);
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenCloudNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

TEST_F(TestDistributedShmemImpl, TestOneShotShmemAllReduce)
{
    const char *group = "hcom123";

    uint32_t worldSize = 4;
    Tensor in(DT_FP16, {64, 256}, "in");
    Tensor out(DT_FP16, {64, 256}, "out");
    FUNCTION("ALLREDUCE", {in}, {out}) {
        TileShape::Current().SetVecTile({64, 256});
        Tensor predToken(DT_INT32, {1, 1}, "predToken");
        OneShotAllReduce(predToken, in, group, worldSize, out);
    }

    std::string functionRawName = GetFunctionRawName("OneShotAllReduce");
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

TEST_F(TestDistributedShmemImpl, TestPredTokenView1D)
{
    const char* group = "hcom123";
    uint32_t worldSize = 4;
    int64_t row = 16;
    int64_t col = 32;
    Tensor in(DT_FP16, {row, col}, "in");
    Tensor out(DT_FP16, {row * worldSize, col}, "out");
    FUNCTION("ALLGATHER", {in}, {out}) {
        TileShape::Current().SetVecTile({4, 8});
        Tensor predToken(DT_INT32, {2, 9}, "predToken");
        AllGather(predToken, in, group, worldSize, out);
    }

    std::string functionRawName = GetFunctionRawName("L0");
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
            ShmemBarrier(predToken, shmemSignal, group, worldSize, out);
        }
    }

    std::string functionRawName = GetFunctionRawName(functionName);
    auto function = Program::GetInstance().GetFunctionByRawName(functionRawName);
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenCloudNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

}
