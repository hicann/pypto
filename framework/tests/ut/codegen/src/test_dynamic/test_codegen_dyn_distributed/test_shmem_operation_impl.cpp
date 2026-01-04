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
        config::SetPlatformConfig("ENABLE_COST_MODEL", false);
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

    Tensor in(DT_FP16, {16, 32}, "in");
    Tensor out(DT_FP16, {64, 32}, "out");
    FUNCTION("ALLGATHER", {in}, {out}) {
        TileShape::Current().SetDistTile(
            {16, 1, 0}, {32, 1, 0}, {1, 4, 0});
        Tensor barrierDummy(DT_INT32, {1, 1}, "barrierDummy");
        ShmemAllGather(in, barrierDummy, group, out);
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

    int32_t ranksize = 4;
    Tensor in(DT_FP16, {64, 256}, "in");
    Tensor out(DT_FP16, {16, 256}, "out");
    FUNCTION("REDUCESCATTER", {in}, {out}) {
        TileShape::Current().SetDistTile(
            {64 / ranksize, 1, 0}, {256, 1, 0}, {1, ranksize, 0});
        ShmemReduceScatter(in, group, DistReduceType::DIST_REDUCE_ADD, out);
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

    int32_t ranksize = 4;
    Tensor in(DT_FP16, {64, 256}, "in");
    Tensor out(DT_FP16, {64, 256}, "out");
    Tensor predToken(DT_INT32, {1, 1}, "predToken");
    FUNCTION("ALLREDUCE", {in, predToken}, {out}) {
        TileShape::Current().SetDistTile(
            {64 / ranksize, 1, 0}, {256, 1, 0}, {1, ranksize, 0});
        TwoShotShmemAllReduce(predToken, in, group, out);
    }

    std::string functionRawName = GetFunctionRawName("TowShotAllReduce");
    auto function = Program::GetInstance().GetFunctionByRawName(functionRawName);
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenCloudNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

TEST_F(TestDistributedShmemImpl, TestOneShotShmemAllReduce)
{
    const char *group = "hcom123";

    int32_t ranksize = 4;
    Tensor in(DT_FP16, {64, 256}, "in");
    Tensor out(DT_FP16, {64, 256}, "out");
    Tensor predToken(DT_INT32, {1, 1}, "predToken");
    FUNCTION("ALLREDUCE", {in, predToken}, {out}) {
        TileShape::Current().SetDistTile(
            {64, 1, 0}, {256, 1, 0}, {1, ranksize, 0});
        OneShotShmemAllReduce(predToken, in, group, out);
    }

    std::string functionRawName = GetFunctionRawName("OneShotAllReduce");
    auto function = Program::GetInstance().GetFunctionByRawName(functionRawName);
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenCloudNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

TEST_F(TestDistributedShmemImpl, TestShmemSet)
{
    Tensor predToken(DT_INT32, {1, 1}, "predToken");
    Tensor in(DT_BF16, {4, 1, 256, 102400}, "in");
    Tensor out(DT_INT32, {1, 1}, "out");

    std::string functionName = "ShmemSet";
    FUNCTION(functionName + "Main", {in}, {out}) {
        LOOP(functionName, FunctionType::DYNAMIC_LOOP, index, LoopRange(1)) {
            (void)index;
            out = ShmemSet(predToken, in);
        }
    }

    std::string functionRawName = GetFunctionRawName(functionName);
    auto function = Program::GetInstance().GetFunctionByRawName(functionRawName);
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenCloudNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

}
