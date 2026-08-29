/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_matmul_operation.cpp
 * \brief
 */

#include "gtest/gtest.h"
#include "tilefwk/tilefwk.h"
#include "tilefwk/platform.h"
#include "interface/inner/tilefwk.h"
#include "interface/operation/operation.h"

using namespace npu::tile_fwk;

class MatmulOperationTest : public testing::Test {
public:
    void SetUp() override
    {
        Program::GetInstance().Reset();
        config::Reset();
        config::SetHostOption(COMPILE_STAGE, CS_EXECUTE_GRAPH);
    }

    void TearDown() override
    {
        // MX cases switch the NPU arch; restore the default so other tests are not affected.
        Platform::Instance().GetSoc().SetNPUArch(NPUArch::DAV_UNKNOWN);
        Platform::Instance().ReloadMemoryPaths("2201");
    }
};

TEST_F(MatmulOperationTest, Test_Matmul_Bias)
{
    TileShape::Current().SetCubeTile({128, 128}, {128, 128}, {128, 128});
    Tensor matA(DT_BF16, {128, 256}, "matA");
    Tensor matB(DT_BF16, {256, 128}, "matB");
    Tensor matBias(DT_FP32, {1, 128}, "biasA");
    Tensor result;
    Matrix::MatmulExtendParam extendParam;
    extendParam.biasTensor = matBias;
    FUNCTION("TestMatmulBias") { result = Matrix::Matmul(DT_FP32, matA, matB, extendParam, false, false, false); }
}

TEST_F(MatmulOperationTest, Test_MatmulMX_Bias)
{
    Platform::Instance().GetSoc().SetNPUArch(NPUArch::DAV_3510);
    Platform::Instance().ReloadMemoryPaths("3510");
    TileShape::Current().SetCubeTile({128, 128}, {128, 128}, {128, 128});
    Tensor matA(DT_FP8E5M2, {128, 256}, "matA");
    Tensor matB(DT_FP8E5M2, {256, 128}, "matB");
    Tensor scaleA(DT_FP8E8M0, {128, 4, 2}, "scaleA");
    Tensor scaleB(DT_FP8E8M0, {4, 128, 2}, "scaleB");
    Tensor matBias(DT_BF16, {1, 128}, "biasA");
    Tensor result;
    Matrix::MatmulExtendParam extendParam;
    extendParam.biasTensor = matBias;
    FUNCTION("TestMatmulMXBias")
    {
        result = Matrix::MatmulMX(DT_FP32, matA, scaleA, matB, scaleB, extendParam, false, false, false, false, false);
    }
}

TEST_F(MatmulOperationTest, Test_BatchMatmul_Bias_3D)
{
    TileShape::Current().SetCubeTile({128, 128}, {128, 128}, {128, 128});
    Tensor matA(DT_BF16, {2, 128, 256}, "matA");
    Tensor matB(DT_BF16, {2, 256, 128}, "matB");
    Tensor matBias(DT_FP32, {1, 128}, "biasA");
    Tensor result;
    Matrix::MatmulExtendParam extendParam;
    extendParam.biasTensor = matBias;
    FUNCTION("TestBatchMatmulBiasWith3Dim")
    {
        result = Matrix::BatchMatmul(DT_FP32, matA, matB, extendParam, false, false, false);
    }
}

TEST_F(MatmulOperationTest, Test_BatchMatmul_Bias_4D)
{
    TileShape::Current().SetCubeTile({128, 128}, {128, 128}, {128, 128});
    Tensor matA(DT_BF16, {1, 2, 128, 256}, "matA");
    Tensor matB(DT_BF16, {2, 2, 256, 128}, "matB");
    Tensor matBias(DT_FP32, {1, 128}, "biasA");
    Tensor result;
    Matrix::MatmulExtendParam extendParam;
    extendParam.biasTensor = matBias;
    FUNCTION("TestBatchMatmulBiasWith4Dim")
    {
        result = Matrix::BatchMatmul(DT_FP32, matA, matB, extendParam, false, false, false);
    }
}

TEST_F(MatmulOperationTest, Test_BatchMatmul_fixpipe_4D)
{
    TileShape::Current().SetCubeTile({128, 128}, {128, 128}, {128, 128});
    Tensor matA(DT_INT8, {1, 2, 128, 256}, "matA");
    Tensor matB(DT_INT8, {2, 2, 256, 128}, "matB");
    Tensor matfixpipe(DT_UINT64, {1, 128}, "fixpipeA");
    Tensor result;
    Matrix::MatmulExtendParam extendParam;
    extendParam.scaleTensor = matfixpipe;
    FUNCTION("TestBatchMatmulFixpipeWith4Dim")
    {
        result = Matrix::BatchMatmul(DT_FP16, matA, matB, extendParam, false, false, false);
    }
}

TEST_F(MatmulOperationTest, Test_BatchMatmul_fixpipe_3D)
{
    TileShape::Current().SetCubeTile({128, 128}, {128, 128}, {128, 128});
    Tensor matA(DT_INT8, {2, 128, 256}, "matA");
    Tensor matB(DT_INT8, {2, 256, 128}, "matB");
    Tensor matfixpipe(DT_UINT64, {1, 128}, "fixpipeA");
    Tensor result;
    Matrix::MatmulExtendParam extendParam;
    extendParam.scaleTensor = matfixpipe;
    FUNCTION("TestBatchMatmulFixpipeWith3Dim")
    {
        result = Matrix::BatchMatmul(DT_FP16, matA, matB, extendParam, false, false, false);
    }
}

TEST_F(MatmulOperationTest, Test_BatchMatmulMX_Bias_3D)
{
    Platform::Instance().GetSoc().SetNPUArch(NPUArch::DAV_3510);
    Platform::Instance().ReloadMemoryPaths("3510");
    TileShape::Current().SetCubeTile({128, 128}, {128, 128}, {128, 128});
    Tensor matA(DT_FP8E5M2, {2, 128, 256}, "matA");
    Tensor matB(DT_FP8E5M2, {1, 256, 128}, "matB");
    Tensor scaleA(DT_FP8E8M0, {2, 128, 4, 2}, "scaleA");
    Tensor scaleB(DT_FP8E8M0, {1, 4, 128, 2}, "scaleB");
    Tensor matBias(DT_BF16, {1, 128}, "biasA");
    Tensor result;
    Matrix::MatmulExtendParam extendParam;
    extendParam.biasTensor = matBias;
    FUNCTION("TestBatchMatmulMXBias3D")
    {
        result = Matrix::BatchMatmulMX(DT_FP32, matA, scaleA, matB, scaleB, extendParam, false, false, false, false,
                                       false);
    }
}

TEST_F(MatmulOperationTest, Test_BatchMatmulMX_Bias_4D)
{
    Platform::Instance().GetSoc().SetNPUArch(NPUArch::DAV_3510);
    Platform::Instance().ReloadMemoryPaths("3510");
    TileShape::Current().SetCubeTile({128, 128}, {128, 128}, {128, 128});
    Tensor matA(DT_FP8E5M2, {2, 1, 128, 256}, "matA");
    Tensor matB(DT_FP8E5M2, {1, 3, 256, 128}, "matB");
    Tensor scaleA(DT_FP8E8M0, {2, 1, 128, 4, 2}, "scaleA");
    Tensor scaleB(DT_FP8E8M0, {1, 3, 4, 128, 2}, "scaleB");
    Tensor matBias(DT_BF16, {1, 128}, "biasA");
    Tensor result;
    Matrix::MatmulExtendParam extendParam;
    extendParam.biasTensor = matBias;
    FUNCTION("TestBatchMatmulMXBias4D")
    {
        result = Matrix::BatchMatmulMX(DT_FP32, matA, scaleA, matB, scaleB, extendParam, false, false, false, false,
                                       false);
    }
}

TEST_F(MatmulOperationTest, Test_BatchMatmulGCC_3D)
{
    TileShape::Current().SetCubeTile({128, 128}, {128, 128}, {128, 128}, true);
    Tensor matA(DT_FP16, {2, 128, 256}, "matA");
    Tensor matB(DT_FP16, {2, 256, 128}, "matB");
    Tensor result;
    FUNCTION("TestBatchMatmulGCC3D") { result = Matrix::BatchMatmul(DT_FP32, matA, matB, false, false, false); }
}

TEST_F(MatmulOperationTest, Test_BatchMatmulGCC_4D)
{
    TileShape::Current().SetCubeTile({128, 128}, {128, 128}, {128, 128}, true);
    Tensor matA(DT_INT8, {2, 2, 128, 256}, "matA");
    Tensor matB(DT_INT8, {2, 2, 256, 128}, "matB");
    Tensor result;
    FUNCTION("TestBatchMatmulGCC4D") { result = Matrix::BatchMatmul(DT_INT32, matA, matB, false, false, false); }
}

TEST_F(MatmulOperationTest, Test_Matmul_SFA)
{
    TileShape::Current().SetCubeTile({128, 128}, {128, 128}, {128, 128});
    Tensor matA(DT_BF16, {128, 576}, "matA");
    Tensor matB(DT_BF16, {576, 2048}, "matB");
    Tensor result;
    FUNCTION("TestMatmulSFA") { result = Matrix::Matmul(DT_FP32, matA, matB, false, false, false); }
}

TEST_F(MatmulOperationTest, Test_Matmul_SFA_T)
{
    TileShape::Current().SetCubeTile({128, 128}, {128, 128}, {128, 128});
    Tensor matA(DT_BF16, {576, 128}, "matA");
    Tensor matB(DT_BF16, {576, 2048}, "matB");
    Tensor result;
    FUNCTION("TestMatmulSFAT") { result = Matrix::Matmul(DT_FP32, matA, matB, true, false, false); }
}
