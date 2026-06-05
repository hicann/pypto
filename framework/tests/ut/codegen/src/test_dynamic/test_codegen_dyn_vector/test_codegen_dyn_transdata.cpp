/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_codegen_dyn_tri.cpp
 * \brief Unit test for codegen.
 */

#include "gtest/gtest.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/configs/config_manager.h"
#include "interface/operation/operation.h"
#include "tilefwk/data_type.h"
#include "codegen/npu/cloudnpu/codegen_cloudnpu.h"
#include "test_codegen_common.h"
#include "test_codegen_utils.h"

namespace npu::tile_fwk {

class TestCodegenDynTransData : public CodegenTestBase {
public:
    TestCodegenDynTransData() : CodegenTestBase({.compileStage = CS_CODEGEN_INSTRUCTION}) {}

    static void TearDownTestCase() { config::SetCodeGenConfig(KEY_CODEGEN_SUPPORT_TILE_TENSOR, true); }
};

TEST_F(TestCodegenDynTransData, TestTransData2)
{
    config::SetCodeGenConfig(KEY_CODEGEN_SUPPORT_TILE_TENSOR, true);
    constexpr const int N = 1;
    constexpr const int C = 7;
    constexpr const int H = 1;
    constexpr const int W = 8;
    std::vector<int64_t> shape = {N, C, H, W};
    TileShape::Current().SetVecTile({N, 8, H, W});

    Tensor inputSrc(DT_FP32, shape, "input");
    Tensor output(DT_FP32, {1, 1, 1, 8, 8}, "output");
    ConfigManager::Instance();
    std::string funcName = "TRANSDATA";
    FUNCTION(funcName, {inputSrc, output})
    {
        LOOP(funcName, FunctionType::DYNAMIC_LOOP, i, LoopRange(1))
        {
            (void)i;
            output = TransData(inputSrc, TileOpFormat(2), 1);
        }
    }
    auto function =
        Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + funcName + SUB_FUNC_SUFFIX + HIDDEN_FUNC_SUFFIX);
    std::string res = GenCodeByFunction(*function);
    std::string expect =
        R"!!!(TTransDataNCHW2NC1HWC0<(int)(1), (int)(8), (int)(1), (int)(8)>(gmTensor_3, Coord5Dim((RUNTIME_COA_GET_PARAM_OFFSET(5, 18, 0)), (RUNTIME_COA_GET_PARAM_OFFSET(5, 18, 1)), (RUNTIME_COA_GET_PARAM_OFFSET(5, 18, 2)), (RUNTIME_COA_GET_PARAM_OFFSET(5, 18, 3)), (RUNTIME_COA_GET_PARAM_OFFSET(5, 18, 4))), ubTensor_4, ubTensor_5, (int)(0), (int)(0), (int)(0), (int)(0));)!!!";
    CheckStringExist(expect, res);
}

TEST_F(TestCodegenDynTransData, TestTransData0_5)
{
    config::SetCodeGenConfig(KEY_CODEGEN_SUPPORT_TILE_TENSOR, true);
    constexpr const int N = 1;
    constexpr const int C1 = 1;
    constexpr const int H = 1;
    constexpr const int W = 8;
    constexpr const int C0 = 8;
    std::vector<int64_t> shape = {N, C1, H, W, C0};
    TileShape::Current().SetVecTile({N, C1, H, W, C0});

    Tensor inputSrc(DT_FP32, shape, "input");
    Tensor output(DT_FP32, {1, 8, 1, 8}, "output");
    ConfigManager::Instance();
    std::string funcName = "TRANSDATA";
    FUNCTION(funcName, {inputSrc, output})
    {
        LOOP(funcName, FunctionType::DYNAMIC_LOOP, i, LoopRange(1))
        {
            (void)i;
            output = TransData(inputSrc, TileOpFormat(0), 1);
        }
    }
    auto function =
        Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + funcName + SUB_FUNC_SUFFIX + HIDDEN_FUNC_SUFFIX);
    std::string res = GenCodeByFunction(*function);
    std::string expect =
        R"!!!(TTransDataNC1HWC02NCHW<(int)(1), (int)(1), (int)(1), (int)(8), (int)(8)>(gmTensor_2, Coord4Dim((RUNTIME_COA_GET_PARAM_OFFSET(4, 22, 0)), (RUNTIME_COA_GET_PARAM_OFFSET(4, 22, 1)), (RUNTIME_COA_GET_PARAM_OFFSET(4, 22, 2)), (RUNTIME_COA_GET_PARAM_OFFSET(4, 22, 3))), ubTensor_3, ubTensor_0, (int)(0), (int)(0), (int)(0), (int)(0), (int)(0));
)!!!";
    CheckStringExist(expect, res);
}

TEST_F(TestCodegenDynTransData, TestTransData4)
{
    config::SetCodeGenConfig(KEY_CODEGEN_SUPPORT_TILE_TENSOR, true);
    constexpr const int N = 15;
    constexpr const int C = 15;
    constexpr const int H = 2;
    constexpr const int W = 16;
    std::vector<int64_t> shape = {N, C, H, W};
    TileShape::Current().SetVecTile({16, 16, 2, 16});

    Tensor inputSrc(DT_FP16, shape, "input");
    Tensor output(DT_FP16, {32, 1, 16, 16}, "output");
    ConfigManager::Instance();
    std::string funcName = "TRANSDATA";
    FUNCTION(funcName, {inputSrc, output})
    {
        LOOP(funcName, FunctionType::DYNAMIC_LOOP, i, LoopRange(1))
        {
            (void)i;
            output = TransData(inputSrc, TileOpFormat(4), 1);
        }
    }
    auto function =
        Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + funcName + SUB_FUNC_SUFFIX + HIDDEN_FUNC_SUFFIX);
    std::string res = GenCodeByFunction(*function);
    std::string expect =
        R"!!!(TTransDataNCHW2Fractal_Z<(int)(16), (int)(16), (int)(2), (int)(16)>(gmTensor_4, Coord4Dim((RUNTIME_COA_GET_PARAM_OFFSET(4, 18, 0)), (RUNTIME_COA_GET_PARAM_OFFSET(4, 18, 1)), (RUNTIME_COA_GET_PARAM_OFFSET(4, 18, 2)), (RUNTIME_COA_GET_PARAM_OFFSET(4, 18, 3))), ubTensor_5, ubTensor_6, (int)(0), (int)(0), (int)(0), (int)(0), (int)(0), (int)(1));
)!!!";
    CheckStringExist(expect, res);
}

TEST_F(TestCodegenDynTransData, TestTransData0_6)
{
    config::SetCodeGenConfig(KEY_CODEGEN_SUPPORT_TILE_TENSOR, true);
    constexpr const int N = 1;
    constexpr const int D = 1;
    constexpr const int C1 = 1;
    constexpr const int H = 1;
    constexpr const int W = 8;
    constexpr const int C0 = 8;
    std::vector<int64_t> shape = {N, D, C1, H, W, C0};
    TileShape::Current().SetVecTile({N, D, C1, H, W, C0});

    Tensor inputSrc(DT_FP32, shape, "input");
    Tensor output(DT_FP32, {1, 8, 1, 1, 8}, "output");
    ConfigManager::Instance();
    std::string funcName = "TRANSDATA";
    FUNCTION(funcName, {inputSrc, output})
    {
        LOOP(funcName, FunctionType::DYNAMIC_LOOP, i, LoopRange(1))
        {
            (void)i;
            output = TransData(inputSrc, TileOpFormat(0), 1);
        }
    }
    auto function =
        Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + funcName + SUB_FUNC_SUFFIX + HIDDEN_FUNC_SUFFIX);
    std::string res = GenCodeByFunction(*function);
    std::string expect =
        R"!!!(TTransDataNDC1HWC02NCDHW<(int)(1), (int)(1), (int)(1), (int)(1), (int)(8), (int)(8)>(gmTensor_2, Coord5Dim((RUNTIME_COA_GET_PARAM_OFFSET(5, 26, 0)), (RUNTIME_COA_GET_PARAM_OFFSET(5, 26, 1)), (RUNTIME_COA_GET_PARAM_OFFSET(5, 26, 2)), (RUNTIME_COA_GET_PARAM_OFFSET(5, 26, 3)), (RUNTIME_COA_GET_PARAM_OFFSET(5, 26, 4))), ubTensor_3, ubTensor_0, (int)(0), (int)(0), (int)(0), (int)(0), (int)(0), (int)(0));
)!!!";
    CheckStringExist(expect, res);
}

TEST_F(TestCodegenDynTransData, TestTransData3)
{
    config::SetCodeGenConfig(KEY_CODEGEN_SUPPORT_TILE_TENSOR, true);
    constexpr const int N = 1;
    constexpr const int C = 7;
    constexpr const int D = 1;
    constexpr const int H = 1;
    constexpr const int W = 8;
    std::vector<int64_t> shape = {N, C, D, H, W};
    TileShape::Current().SetVecTile({1, 8, 1, 1, 8});

    Tensor inputSrc(DT_FP32, shape, "input");
    Tensor output(DT_FP32, {1, 1, 1, 1, 8, 8}, "output");
    ConfigManager::Instance();
    std::string funcName = "TRANSDATA";
    FUNCTION(funcName, {inputSrc, output})
    {
        LOOP(funcName, FunctionType::DYNAMIC_LOOP, i, LoopRange(1))
        {
            (void)i;
            output = TransData(inputSrc, TileOpFormat(3), 1);
        }
    }
    auto function =
        Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + funcName + SUB_FUNC_SUFFIX + HIDDEN_FUNC_SUFFIX);
    std::string res = GenCodeByFunction(*function);
    std::string expect =
        R"!!!(TTransDataNCDHW2NDC1HWC0<(int)(1), (int)(1), (int)(8), (int)(1), (int)(8)>(gmTensor_3, Coord6Dim((RUNTIME_COA_GET_PARAM_OFFSET(6, 22, 0)), (RUNTIME_COA_GET_PARAM_OFFSET(6, 22, 1)), (RUNTIME_COA_GET_PARAM_OFFSET(6, 22, 2)), (RUNTIME_COA_GET_PARAM_OFFSET(6, 22, 3)), (RUNTIME_COA_GET_PARAM_OFFSET(6, 22, 4)), (RUNTIME_COA_GET_PARAM_OFFSET(6, 22, 5))), ubTensor_4, ubTensor_5, (int)(0), (int)(0), (int)(0), (int)(0), (int)(0));
)!!!";
    CheckStringExist(expect, res);
}

TEST_F(TestCodegenDynTransData, TestTransData5)
{
    config::SetCodeGenConfig(KEY_CODEGEN_SUPPORT_TILE_TENSOR, true);
    constexpr const int N = 15;
    constexpr const int C = 7;
    constexpr const int D = 1;
    constexpr const int H = 1;
    constexpr const int W = 8;
    std::vector<int64_t> shape = {N, C, D, H, W};
    TileShape::Current().SetVecTile({16, 8, 1, 1, 8});

    Tensor inputSrc(DT_FP32, shape, "input");
    Tensor output(DT_FP32, {8, 1, 16, 8}, "output");
    ConfigManager::Instance();
    std::string funcName = "TRANSDATA";
    FUNCTION(funcName, {inputSrc, output})
    {
        LOOP(funcName, FunctionType::DYNAMIC_LOOP, i, LoopRange(1))
        {
            (void)i;
            output = TransData(inputSrc, TileOpFormat(5), 1);
        }
    }
    auto function =
        Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + funcName + SUB_FUNC_SUFFIX + HIDDEN_FUNC_SUFFIX);
    std::string res = GenCodeByFunction(*function);
    std::string expect =
        R"!!!(TTransDataNCDHW2FRACTAL_Z_3D<(int)(16), (int)(8), (int)(1), (int)(1), (int)(8)>(gmTensor_4, Coord4Dim((RUNTIME_COA_GET_PARAM_OFFSET(4, 22, 0)), (RUNTIME_COA_GET_PARAM_OFFSET(4, 22, 1)), (RUNTIME_COA_GET_PARAM_OFFSET(4, 22, 2)), (RUNTIME_COA_GET_PARAM_OFFSET(4, 22, 3))), ubTensor_5, ubTensor_6, (int)(0), (int)(0), (int)(0), (int)(0), (int)(0), (int)(0), (int)(1));
)!!!";
    CheckStringExist(expect, res);
}
} // namespace npu::tile_fwk
