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
 * \file test_onboard_bmm.cpp
 * \brief
 */

#include "test_suite_stest_ops.h"
#include "test_dev_func_runner.h"

using namespace npu::tile_fwk;

class OnBoardTest : public npu::tile_fwk::stest::TestSuite_STest_Ops_Aihac {};

template<typename T>
int64_t GetShapeCapacity(const vector<int64_t>& shape) {
    int64_t capacity = sizeof(T);
    for (auto dim: shape) {
        capacity = capacity * dim;
    }
    return capacity;
}

template<typename InputT, typename OnputT, bool transpose = false>
void TestBatchMatmul3D(std::vector<int64_t> shape_a, std::vector<int64_t>shape_b ,string dataPath) {
    aclInit(nullptr);
    rtSetDevice(GetDeviceIdByEnvVar());
    assert(shape_a.size() == 3 && shape_b.size() == 3);

    int bs = std::max(shape_a[0], shape_b[0]);
    int m = shape_a[1];
    int n = transpose ? shape_b[1] : shape_b[2];
    void *mat_a_ptr = readToDev<InputT>(dataPath + "/mat_a.bin", GetShapeCapacity<InputT>(shape_a));
    void *mat_b_ptr = readToDev<InputT>(dataPath + "/mat_b.bin", GetShapeCapacity<InputT>(shape_b));

    std::vector<int64_t> shape_c = {bs, m, n};
    const int capacity_mat_c = bs * m * n;
    uint32_t outputSize = capacity_mat_c * sizeof(OnputT);
    uint8_t* mat_c_ptr = allocDevAddr(outputSize);

    auto InputAstDtype = GetAstDtype<InputT>();
    auto OutputAstDtype = GetAstDtype<OnputT>();

    PROGRAM("BATCHMATMUL") {
        Tensor matA(InputAstDtype, shape_a, (uint8_t *)mat_a_ptr, "MatA");
        Tensor matB(InputAstDtype, shape_b, (uint8_t *)mat_b_ptr, "MatB");
        Tensor matC(OutputAstDtype, shape_c, mat_c_ptr, "MatC");
        config::SetBuildStatic(true);
        FUNCTION("BATCHMATMUL", {matA, matB, matC}) {
            matC = npu::tile_fwk::Matrix::BatchMatmul(OutputAstDtype, matA, matB, false, transpose);
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());
    std::vector<OnputT> res(capacity_mat_c);
    machine::GetRA()->CopyFromTensor((uint8_t *)res.data(), mat_c_ptr, outputSize);
    std::vector<OnputT> golden(capacity_mat_c);
    readInput(dataPath + "/mat_c.bin", golden);
    int ret = resultCmp(golden, res, 0.001f);
    EXPECT_EQ(ret, true);
}

template<typename InputT, typename OnputT, bool transpose=false>
void TestBatchMatmul4D(std::vector<int64_t> shape_a, std::vector<int64_t>shape_b ,string dataPath) {
    aclInit(nullptr);
    rtSetDevice(GetDeviceIdByEnvVar());
    assert(shape_a.size() == 4 && shape_b.size() == 4);
    int bs1 = std::max(shape_a[0], shape_b[0]);
    int bs2 = std::max(shape_a[1], shape_b[1]);
    int m = shape_a[2];
    int n = transpose ? shape_b[2] : shape_b[3];
    void *mat_a_ptr = readToDev<InputT>(dataPath + "/mat_a.bin", GetShapeCapacity<InputT>(shape_a));
    void *mat_b_ptr = readToDev<InputT>(dataPath + "/mat_b.bin", GetShapeCapacity<InputT>(shape_b));

    std::vector<int64_t> shape_c = {bs1, bs2, m, n};
    const int capacity_mat_c = bs1 * bs2 * m * n;
    uint32_t outputSize = capacity_mat_c * sizeof(OnputT);
    uint8_t* mat_c_ptr = allocDevAddr(outputSize);

    auto InputAstDtype = GetAstDtype<InputT>();
    auto OutputAstDtype = GetAstDtype<OnputT>();

    PROGRAM("BATCHMATMUL") {
        Tensor matA(InputAstDtype, shape_a, (uint8_t *)mat_a_ptr, "MatA");
        Tensor matB(InputAstDtype, shape_b, (uint8_t *)mat_b_ptr, "MatB");
        Tensor matC(OutputAstDtype, shape_c, mat_c_ptr, "MatC");
        config::SetBuildStatic(true);
        FUNCTION("BATCHMATMUL", {matA, matB, matC}) {
            matC = npu::tile_fwk::Matrix::BatchMatmul(OutputAstDtype, matA, matB, false, transpose);
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());
    std::vector<OnputT> res(capacity_mat_c);
    machine::GetRA()->CopyFromTensor((uint8_t *)res.data(), mat_c_ptr, outputSize);
    std::vector<OnputT> golden(capacity_mat_c);
    readInput(dataPath + "/mat_c.bin", golden);
    int ret = resultCmp(golden, res, 0.001f);
    EXPECT_EQ(ret, true);
}

TEST_F(OnBoardTest, test_BMM_Simple) {
    TileShape::Current().SetCubeTile({32, 32}, {128, 128}, {128, 128});
    std::vector<int64_t> mat_a_shape = {2, 32, 128};
    std::vector<int64_t> mat_b_shape = {2, 128, 512};
    TestBatchMatmul3D<npu::tile_fwk::float16, float>(mat_a_shape, mat_b_shape, GetGoldenDir());
}

TEST_F(OnBoardTest, test_BMM_Simple_FP32_BT) {
    TileShape::Current().SetCubeTile({16, 16}, {32, 32}, {16, 16});
    std::vector<int64_t> mat_a_shape = {1, 16, 32};
    std::vector<int64_t> mat_b_shape = {1, 32, 16};
    TestBatchMatmul3D<float, float, true>(mat_a_shape, mat_b_shape, GetGoldenDir());
}

TEST_F(OnBoardTest, test_BMM_Simple_FP32) {
    TileShape::Current().SetCubeTile({16, 16}, {32, 32}, {32, 32});
    std::vector<int64_t> mat_a_shape = {1, 32, 32};
    std::vector<int64_t> mat_b_shape = {1, 32, 32};
    TestBatchMatmul3D<float, float, false>(mat_a_shape, mat_b_shape, GetGoldenDir());
}

TEST_F(OnBoardTest, test_BMM_Simple_FP32_256_256) {
    TileShape::Current().SetCubeTile({32, 32}, {32, 32}, {32, 32});
    std::vector<int64_t> mat_a_shape = {1, 256, 256};
    std::vector<int64_t> mat_b_shape = {1, 256, 256};
    TestBatchMatmul3D<float, float, false>(mat_a_shape, mat_b_shape, GetGoldenDir());
}

TEST_F(OnBoardTest, test_BMM_UNALIGN_2_1024_32) {
    TileShape::Current().SetCubeTile({16, 16}, {128, 128}, {32, 32});
    std::vector<int64_t>  mat_a_shape = {2, 1, 1024};
    std::vector<int64_t>  mat_b_shape = {2, 1024, 32};
    TestBatchMatmul3D<npu::tile_fwk::float16, float>(mat_a_shape, mat_b_shape, GetGoldenDir());
}

TEST_F(OnBoardTest, test_BMM_UNALIGN_32_4_128_512) {
    TileShape::Current().SetCubeTile({16, 16}, {128, 128}, {128, 128});
    std::vector<int64_t>  mat_a_shape = {32, 4, 128};
    std::vector<int64_t>  mat_b_shape = {32, 128, 512};
    TestBatchMatmul3D<npu::tile_fwk::float16, npu::tile_fwk::float16>(mat_a_shape, mat_b_shape, GetGoldenDir());
}

TEST_F(OnBoardTest, test_BMM_BF16) {
    TileShape::Current().SetCubeTile({64, 64}, {64, 64}, {64, 64});
    std::vector<int64_t> mat_a_shape = {2, 64, 64};
    std::vector<int64_t> mat_b_shape = {2, 64, 64};
    TestBatchMatmul3D<npu::tile_fwk::bfloat16, npu::tile_fwk::bfloat16>(mat_a_shape, mat_b_shape, GetGoldenDir());
}

TEST_F(OnBoardTest, test_BMM_post) {
    TileShape::Current().SetCubeTile({32, 32}, {128, 128}, {128, 128});
    std::vector<int64_t>    mat_a_shape = {32, 32, 512};
    std::vector<int64_t>    mat_b_shape = {32, 512, 128};
    TestBatchMatmul3D<npu::tile_fwk::float16, npu::tile_fwk::float16>(mat_a_shape, mat_b_shape, GetGoldenDir());
}

TEST_F(OnBoardTest, test_BMM_3D_Brc) {
    TileShape::Current().SetCubeTile({16, 16}, {128, 128}, {128, 128});
    std::vector<int64_t>    mat_a_shape = {2, 16, 256};
    std::vector<int64_t>    mat_b_shape = {1, 256, 128};
    TestBatchMatmul3D<npu::tile_fwk::float16, float>(mat_a_shape, mat_b_shape, GetGoldenDir());
}

TEST_F(OnBoardTest, test_BMM_3D_Brc_Transpose) {
    TileShape::Current().SetCubeTile({16, 16}, {64, 64}, {32, 32});
    std::vector<int64_t>    mat_a_shape = {2, 64, 256};
    std::vector<int64_t>    mat_b_shape = {1, 128, 256};
    TestBatchMatmul3D<npu::tile_fwk::float16, float, true>(mat_a_shape, mat_b_shape, GetGoldenDir());
}

TEST_F(OnBoardTest, test_BMM_4D_Brc) {
    TileShape::Current().SetCubeTile({32, 32}, {128, 128}, {64, 64});
    std::vector<int64_t>    mat_a_shape = {2, 2, 128, 128};
    std::vector<int64_t>    mat_b_shape = {2, 1, 128, 64};
    TestBatchMatmul4D<npu::tile_fwk::float16, float>(mat_a_shape, mat_b_shape, GetGoldenDir());
}

TEST_F(OnBoardTest, test_BMM_4D_Brc_Transpose) {
    TileShape::Current().SetCubeTile({64, 64}, {64, 64}, {32, 32});
    std::vector<int64_t>    mat_a_shape = {2, 2, 128, 128};
    std::vector<int64_t>    mat_b_shape = {2, 1, 64, 128};
    TestBatchMatmul4D<npu::tile_fwk::float16, float, true>(mat_a_shape, mat_b_shape, GetGoldenDir());
}

TEST_F(OnBoardTest, test_BMM_9_16_7168_2048) {
    TileShape::Current().SetCubeTile({32, 32}, {128, 128}, {128, 128});
    std::vector<int64_t>  mat_a_shape = {1, 16, 7168};
    std::vector<int64_t>  mat_b_shape = {9, 7168, 2048};
    TestBatchMatmul3D<npu::tile_fwk::float16, npu::tile_fwk::float16, true>(mat_a_shape, mat_b_shape, GetGoldenDir());
}

template<typename InputT, typename OnputT, bool transpose = false>
void TestBatchMatmulA8W8O32(std::vector<int64_t> shape_a_in, std::vector<int64_t> shape_b_in) {
    aclInit(nullptr);
    rtSetDevice(GetDeviceIdByEnvVar());
    int bs = shape_a_in[0];
    int m = shape_a_in[1];
    int k = shape_a_in[2];
    int n = transpose ? shape_b_in[1] : shape_b_in[2];
    const int capacity_mat_a = bs * m * k;
    const int capacity_mat_b = bs * k * n;
    const int capacity_mat_c = bs * m * n;

    std::vector<int64_t> shape_a = {bs, m, k};
    std::vector<int64_t> shape_b = {bs, k, n};
    std::vector<int64_t> shape_c = {bs, m, n};
    void *mat_a_ptr = readToDev<__uint16_t>(GetGoldenDir() + "/mat_a.bin", capacity_mat_a);
    void *mat_b_ptr = readToDev<__uint16_t>(GetGoldenDir() + "/mat_b.bin", capacity_mat_b);
    assert(mat_a_ptr != nullptr && mat_b_ptr != nullptr);
    uint32_t outputSize = capacity_mat_c * sizeof(float);
    uint8_t* mat_c_ptr = allocDevAddr(outputSize);

    PROGRAM("BATCHMATMUL") {
        config::Reset();
        Tensor matA(DataType::DT_INT8, shape_a, (uint8_t *)mat_a_ptr, "MatA");
        Tensor matB(DataType::DT_INT8, shape_b, (uint8_t *)mat_b_ptr, "MatB");
        Tensor matC(DataType::DT_INT32, shape_c, mat_c_ptr, "MatC");
        config::SetBuildStatic(true);
        FUNCTION("BATCHMATMUL", {matA, matB, matC}) {
            matC = npu::tile_fwk::Matrix::BatchMatmul(DataType::DT_INT32, matA, matB, false, false);
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());

    std::vector<int32_t> res(capacity_mat_c);
    machine::GetRA()->CopyFromTensor((uint8_t *)res.data(), mat_c_ptr, outputSize);

    std::vector<int32_t> golden(capacity_mat_c);
    readInput(GetGoldenDir() + "/mat_c.bin", golden);

    int ret = resultCmp(golden, res, 0.001f);
    EXPECT_EQ(ret, true);
}

template<typename InputT, typename OnputT, bool transpose = false>
void TestBatchMatmulA8W8O32ACC(std::vector<int64_t> shape_a_in, std::vector<int64_t> shape_b_in) {
    aclInit(nullptr);
    rtSetDevice(GetDeviceIdByEnvVar());
    int bs = shape_a_in[0];
    int m = shape_a_in[1];
    int k = shape_a_in[2];
    int n = transpose ? shape_b_in[1] : shape_b_in[2];
    const int capacity_mat_a = bs * m * k;
    const int capacity_mat_b = bs * k * n;
    const int capacity_mat_c = bs * m * n;

    std::vector<int64_t> shape_a = {bs, m, k};
    std::vector<int64_t> shape_b = {bs, k, n};
    std::vector<int64_t> shape_c = {bs, m, n};
    void *mat_a_ptr = readToDev<__uint16_t>(GetGoldenDir() + "/mat_a.bin", capacity_mat_a);
    void *mat_b_ptr = readToDev<__uint16_t>(GetGoldenDir() + "/mat_b.bin", capacity_mat_b);
    assert(mat_a_ptr != nullptr && mat_b_ptr != nullptr);
    uint32_t outputSize = capacity_mat_c * sizeof(float);
    uint8_t* mat_c_ptr = allocDevAddr(outputSize);

    PROGRAM("BATCHMATMUL") {
        config::Reset();
        Tensor matA(DataType::DT_INT8, shape_a, (uint8_t *)mat_a_ptr, "MatA");
        Tensor matB(DataType::DT_INT8, shape_b, (uint8_t *)mat_b_ptr, "MatB");
        Tensor matC(DataType::DT_INT32, shape_c, mat_c_ptr, "MatC");

        config::SetBuildStatic(true);
        FUNCTION("BATCHMATMUL", {matA, matB, matC}) {
            matC = npu::tile_fwk::Matrix::Matmul(DataType::DT_INT32, matA, matB, false, false);
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());

    std::vector<int32_t> res(capacity_mat_c);
    machine::GetRA()->CopyFromTensor((uint8_t *)res.data(), mat_c_ptr, outputSize);

    std::vector<int32_t> golden(capacity_mat_c);
    readInput(GetGoldenDir() + "/mat_c.bin", golden);

    int ret = resultCmp(golden, res, 0.001f);
    EXPECT_EQ(ret, true);
}


TEST_F(OnBoardTest, test_BMM_Simple_A8W8O32) {
    TileShape::Current().SetCubeTile({32, 32}, {32, 32}, {32, 32});
    std::vector<int64_t>  mat_a_shape = {1, 32, 64};
    std::vector<int64_t>  mat_b_shape = {1, 64, 64};
    TestBatchMatmulA8W8O32<int8_t, int32_t, false>(mat_a_shape, mat_b_shape);
}

TEST_F(OnBoardTest, test_BMM_Simple_A8W8O32_4_4_64_64) {
    TileShape::Current().SetCubeTile({32, 32}, {32, 32}, {32, 32});
    std::vector<int64_t>  mat_a_shape = {4, 4, 64};
    std::vector<int64_t>  mat_b_shape = {4, 64, 64};
    TestBatchMatmulA8W8O32<int8_t, int32_t, false>(mat_a_shape, mat_b_shape);
}

TEST_F(OnBoardTest, test_BMM_Simple_A8W8O32_1_4_4096_7168) {
    TileShape::Current().SetCubeTile({32, 32}, {32, 32}, {32, 32});
    std::vector<int64_t>  mat_a_shape = {1, 4, 4096};
    std::vector<int64_t>  mat_b_shape = {1, 4096, 7168};
    TestBatchMatmulA8W8O32<int8_t, int32_t, false>(mat_a_shape, mat_b_shape);
}
