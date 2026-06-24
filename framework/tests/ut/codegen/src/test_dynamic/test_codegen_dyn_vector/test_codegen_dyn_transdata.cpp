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

    static void TearDownTestCase() {}

    struct TransDataTestCase {
        std::string caseName;
        std::vector<int64_t> inputShape;
        std::vector<int64_t> vecTile;
        DataType dtype;
        std::vector<int64_t> outputShape;
        int tileOpFormat;
        std::vector<SymbolicScalar> validShape;
        std::string expect;
    };

    void RunTransDataTest(const TransDataTestCase& tc)
    {
        TileShape::Current().SetVecTile(tc.vecTile);

        Tensor inputSrc(tc.dtype, tc.inputShape, "input");
        Tensor output(tc.dtype, tc.outputShape, "output");
        ConfigManager::Instance();
        FUNCTION(tc.caseName, {inputSrc, output})
        {
            LOOP(tc.caseName, FunctionType::DYNAMIC_LOOP, i, LoopRange(1))
            {
                (void)i;
                output = TransData(inputSrc, TileOpFormat(tc.tileOpFormat), tc.outputShape, tc.validShape, 1);
            }
        }
        auto function = Program::GetInstance().GetFunctionByRawName(
            FUNCTION_PREFIX + tc.caseName + SUB_FUNC_SUFFIX + HIDDEN_FUNC_SUFFIX);
        std::string res = GenCodeByFunction(*function);
        CheckStringExist(tc.expect, res);
    }
};

TEST_F(TestCodegenDynTransData, TestTransData2)
{
    RunTransDataTest(
        {.caseName = "TestTransData2",
         .inputShape = {1, 7, 1, 8},
         .vecTile = {1, 8, 1, 8},
         .dtype = DT_FP32,
         .outputShape = {1, 1, 1, 8, 8},
         .tileOpFormat = 2,
         .validShape = {1, 1, 1, 8, 8},
         .expect =
             R"!!!(TTransDataNCHW2NC1HWC0<(int)(1), (int)(8), (int)(1), (int)(8)>(gmTensor_3, Coord5Dim((RUNTIME_COA_GET_PARAM_OFFSET(5, 18, 0)), (RUNTIME_COA_GET_PARAM_OFFSET(5, 18, 1)), (RUNTIME_COA_GET_PARAM_OFFSET(5, 18, 2)), (RUNTIME_COA_GET_PARAM_OFFSET(5, 18, 3)), (RUNTIME_COA_GET_PARAM_OFFSET(5, 18, 4))), ubTensor_4, ubTensor_5, (int)(0), (int)(0), (int)(0), (int)(0));)!!!"});
}

TEST_F(TestCodegenDynTransData, TestTransData4)
{
    RunTransDataTest(
        {.caseName = "TestTransData4",
         .inputShape = {15, 15, 2, 16},
         .vecTile = {16, 16, 2, 16},
         .dtype = DT_FP16,
         .outputShape = {32, 1, 16, 16},
         .tileOpFormat = 4,
         .validShape = {32, 1, 16, 16},
         .expect =
             R"!!!(TTransDataNCHW2Fractal_Z<(int)(16), (int)(16), (int)(2), (int)(16)>(gmTensor_4, Coord4Dim((RUNTIME_COA_GET_PARAM_OFFSET(4, 18, 0)), (RUNTIME_COA_GET_PARAM_OFFSET(4, 18, 1)), (RUNTIME_COA_GET_PARAM_OFFSET(4, 18, 2)), (RUNTIME_COA_GET_PARAM_OFFSET(4, 18, 3))), ubTensor_5, ubTensor_6, (int)(0), (int)(0), (int)(0), (int)(0), (int)(0), (int)(1));
)!!!"});
}

TEST_F(TestCodegenDynTransData, TestTransData3)
{
    RunTransDataTest(
        {.caseName = "TestTransData3",
         .inputShape = {1, 7, 1, 1, 8},
         .vecTile = {1, 8, 1, 1, 8},
         .dtype = DT_FP32,
         .outputShape = {1, 1, 1, 1, 8, 8},
         .tileOpFormat = 3,
         .validShape = {1, 1, 1, 1, 8, 8},
         .expect =
             R"!!!(TTransDataNCDHW2NDC1HWC0<(int)(1), (int)(1), (int)(8), (int)(1), (int)(8)>(gmTensor_3, Coord6Dim((RUNTIME_COA_GET_PARAM_OFFSET(6, 22, 0)), (RUNTIME_COA_GET_PARAM_OFFSET(6, 22, 1)), (RUNTIME_COA_GET_PARAM_OFFSET(6, 22, 2)), (RUNTIME_COA_GET_PARAM_OFFSET(6, 22, 3)), (RUNTIME_COA_GET_PARAM_OFFSET(6, 22, 4)), (RUNTIME_COA_GET_PARAM_OFFSET(6, 22, 5))), ubTensor_4, ubTensor_5, (int)(0), (int)(0), (int)(0), (int)(0), (int)(0));
)!!!"});
}

TEST_F(TestCodegenDynTransData, TestTransData5)
{
    RunTransDataTest(
        {.caseName = "TestTransData5",
         .inputShape = {15, 7, 1, 1, 8},
         .vecTile = {16, 8, 1, 1, 8},
         .dtype = DT_FP32,
         .outputShape = {8, 1, 16, 8},
         .tileOpFormat = 5,
         .validShape = {8, 1, 16, 8},
         .expect =
             R"!!!(TTransDataNCDHW2FRACTAL_Z_3D<(int)(16), (int)(8), (int)(1), (int)(1), (int)(8)>(gmTensor_4, Coord4Dim((RUNTIME_COA_GET_PARAM_OFFSET(4, 22, 0)), (RUNTIME_COA_GET_PARAM_OFFSET(4, 22, 1)), (RUNTIME_COA_GET_PARAM_OFFSET(4, 22, 2)), (RUNTIME_COA_GET_PARAM_OFFSET(4, 22, 3))), ubTensor_5, ubTensor_6, (int)(0), (int)(0), (int)(0), (int)(0), (int)(0), (int)(0), (int)(1));
)!!!"});
}

TEST_F(TestCodegenDynTransData, TestTransData0_3)
{
    RunTransDataTest(
        {.caseName = "TestTransData0_3",
         .inputShape = {1, 1, 1, 8, 8},
         .vecTile = {1, 1, 1, 8, 8},
         .dtype = DT_FP32,
         .outputShape = {1, 7, 1, 8},
         .tileOpFormat = 0,
         .validShape = {1, 7, 1, 8},
         .expect =
             R"!!!(TTransDataNC1HWC02NCHW<(int)(1), (int)(7), (int)(1), (int)(8)>(gmTensor_2, Coord4Dim((RUNTIME_COA_GET_PARAM_OFFSET(4, 22, 0)), (RUNTIME_COA_GET_PARAM_OFFSET(4, 22, 1)), (RUNTIME_COA_GET_PARAM_OFFSET(4, 22, 2)), (RUNTIME_COA_GET_PARAM_OFFSET(4, 22, 3))), ubTensor_3, ubTensor_0, (int)(0), (int)(0), (int)(0), (int)(0), (int)(0), (int)(0), (int)(1), (int)(1));)!!!"});
}

TEST_F(TestCodegenDynTransData, TestTransData0_6)
{
    RunTransDataTest(
        {.caseName = "TestTransData0_6",
         .inputShape = {1, 1, 1, 1, 8, 8},
         .vecTile = {1, 1, 1, 1, 8, 8},
         .dtype = DT_FP32,
         .outputShape = {1, 7, 1, 1, 8},
         .tileOpFormat = 0,
         .validShape = {1, 7, 1, 1, 8},
         .expect =
             R"!!!(TTransDataNDC1HWC02NCDHW<(int)(1), (int)(1), (int)(7), (int)(1), (int)(8)>(gmTensor_2, Coord5Dim((RUNTIME_COA_GET_PARAM_OFFSET(5, 47, 0)), (RUNTIME_COA_GET_PARAM_OFFSET(5, 47, 1)), (RUNTIME_COA_GET_PARAM_OFFSET(5, 47, 2)), (RUNTIME_COA_GET_PARAM_OFFSET(5, 47, 3)), (RUNTIME_COA_GET_PARAM_OFFSET(5, 47, 4))), ubTensor_3, ubTensor_0, (int)(0), (int)(0), (int)(0), (int)(0), (int)(0), (int)(0), (int)(0), (int)(1), (int)(1));
)!!!"});
}
} // namespace npu::tile_fwk
