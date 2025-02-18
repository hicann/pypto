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
 * \file test_nz_datafmt.cpp
 * \brief
 */

#include "test_suite_stest_ops.h"
#include "test_dev_func_runner.h"

using namespace npu::tile_fwk;

class OnBoardTest : public npu::tile_fwk::stest::TestSuite_STest_Ops_Aihac {};

template <DataType inputType, DataType outputType, bool IsANZ = false, bool IsBNZ = false>
void TestNZFormat(int bs, int m, int k, int n) {
    aclInit(nullptr);
    rtSetDevice(GetDeviceIdByEnvVar());

    const int capacity_mat_a = bs * m * k;
    const int capacity_mat_b = bs * k * n;
    const int capacity_mat_c = bs * m * n;

    std::vector<int64_t> shape_a = {m, k};
    std::vector<int64_t> shape_b = { k, n};
    std::vector<int64_t> shape_c = { m, n};

    void *mat_a_ptr = readToDev<__uint16_t>(GetGoldenDir() + "/mat_a.bin", capacity_mat_a);
    void *mat_b_ptr = readToDev<__uint16_t>(GetGoldenDir() + "/mat_b.bin", capacity_mat_b);
    assert(mat_a_ptr != nullptr && mat_b_ptr != nullptr);
    uint32_t outputSize = capacity_mat_c * sizeof(float);
    uint8_t* mat_c_ptr = allocDevAddr(outputSize);

    PROGRAM("BATCHMATMUL") {
        config::Reset();
        TileShape::Current().SetCubeTile({32, 32}, {32, 32}, {32, 32});
        auto afmt = IsANZ ? TileOpFormat::TILEOP_NZ : TileOpFormat::TILEOP_ND;
        auto bfmt = IsBNZ ? TileOpFormat::TILEOP_NZ : TileOpFormat::TILEOP_ND;
        Tensor matA(inputType, shape_a, (uint8_t *)mat_a_ptr, "MatA", afmt);
        Tensor matB(inputType, shape_b, (uint8_t *)mat_b_ptr, "MatB", bfmt);
        Tensor matC(outputType, shape_c, mat_c_ptr, "MatC");
        config::SetBuildStatic(true);
        FUNCTION("BATCHMATMUL", {matA, matB, matC}) {
            matC = npu::tile_fwk::Matrix::Matmul(outputType, matA, matB);
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());
    std::vector<float> res(capacity_mat_c);
    machine::GetRA()->CopyFromTensor((uint8_t *)res.data(), mat_c_ptr, outputSize);

    std::vector<float> golden(capacity_mat_c);
    readInput(GetGoldenDir() + "/mat_c.bin", golden);

    int ret = resultCmp(golden, res, 0.001f);
    EXPECT_EQ(ret, true);
}

TEST_F(OnBoardTest, test_BMM_NZ_1_32_32_32) {
    TestNZFormat<DataType::DT_FP16, DataType::DT_FP32, true, false>(1, 32, 32, 32);
}

TEST_F(OnBoardTest, test_BMM_NZ_1_32_32_320) {
    TestNZFormat<DataType::DT_FP16, DataType::DT_FP32, true, false>(1, 32, 32, 320);
}

TEST_F(OnBoardTest, test_BMM_NZ_1_64_32_16) {
    TestNZFormat<DataType::DT_FP16, DataType::DT_FP32, true, false>(1, 64, 16, 32);
}

TEST_F(OnBoardTest, test_BMM_NZ_1_128_128_128) {
    TestNZFormat<DataType::DT_FP16, DataType::DT_FP32, false, true>(1, 128, 128, 128);
}

TEST_F(OnBoardTest, test_BMM_NZ_1_32_7168_576) {
    TestNZFormat<DataType::DT_FP16, DataType::DT_FP32, false, true>(1, 32, 7168, 576);
}

TEST_F(OnBoardTest, test_BMM_NZ_1_32_7168_1536) {
    TestNZFormat<DataType::DT_FP16, DataType::DT_FP32, false, true>(1, 32, 7168, 1536);
}

TEST_F(OnBoardTest, test_BMM_NZ_1_64_32_16_A8W8O32) {
    TestNZFormat<DataType::DT_INT8, DataType::DT_INT32, true, false>(1, 64, 16, 32);
}

TEST_F(OnBoardTest, test_BMM_NZ_1_128_128_128_A8W8O32) {
    TestNZFormat<DataType::DT_INT8, DataType::DT_INT32, false, true>(1, 128, 128, 128);
}

template <DataType inputType, DataType outputType, bool IsANZ = false, bool IsBNZ = false, bool isTransB = false>
void TestNZFormatBatch(int bs, int m, int k, int n) {
    aclInit(nullptr);
    rtSetDevice(GetDeviceIdByEnvVar());

    const int capacity_mat_a = bs * m * k;
    const int capacity_mat_b = bs * k * n;
    const int capacity_mat_c = bs * m * n;
    std::vector<int64_t> batch_shape_a = {bs*m, k};
    auto nLen = isTransB ? bs * n : bs * k;
    auto kLen = isTransB ? k : n;
    std::vector<int64_t> batch_shape_b = {nLen, kLen};
    std::vector<int64_t> batch_shape_c = {bs*m, n};

    std::vector<int64_t> shape_a = {m, k};
    nLen = isTransB ? n : k;
    std::vector<int64_t> shape_b = {nLen, kLen};
    std::vector<int64_t> shape_c = { m, n};

    void *mat_a_ptr = readToDev<__uint16_t>(GetGoldenDir() + "/mat_a.bin", capacity_mat_a);
    void *mat_b_ptr = readToDev<__uint16_t>(GetGoldenDir() + "/mat_b.bin", capacity_mat_b);
    assert(mat_a_ptr != nullptr && mat_b_ptr != nullptr);
    uint32_t outputSize = capacity_mat_c * sizeof(float);
    uint8_t* mat_c_ptr = allocDevAddr(outputSize);

    PROGRAM("BATCHMATMUL") {
        config::Reset();
        TileShape::Current().SetCubeTile({32, 32}, {32, 32}, {32, 32});
        auto afmt = IsANZ ? TileOpFormat::TILEOP_NZ : TileOpFormat::TILEOP_ND;
        auto bfmt = IsBNZ ? TileOpFormat::TILEOP_NZ : TileOpFormat::TILEOP_ND;
        Tensor matA(inputType, batch_shape_a, (uint8_t *)mat_a_ptr, "MatA", afmt);
        Tensor matB(inputType, batch_shape_b, (uint8_t *)mat_b_ptr, "MatB", bfmt);
        Tensor matC(outputType, batch_shape_c, mat_c_ptr, "MatC");
        std::vector<Tensor> matrixVec;
        config::SetBuildStatic(true);
        FUNCTION("BATCHMATMUL", {matA, matB, matC}) {
            std::vector<std::pair<Tensor, std::vector<int64_t>>> assembleVec;
            for (size_t index = 0; index < (size_t)bs; ++index) {
                auto inputA = View(matA, {m, k}, {(int)index*m, 0});
                auto inputB = isTransB ? View(matB, {n, k}, {(int)index*n, 0}) : View(matB, {k, n}, {(int)index*k, 0});
                TileShape::Current().SetMatrixSize({m, k, n});
                auto outTensor = npu::tile_fwk::Matrix::Matmul<false, isTransB>(outputType, inputA, inputB);
                std::vector<int64_t> pairSecond = {(int)index * m, 0};
                auto pair = std::make_pair(outTensor, pairSecond);
                assembleVec.emplace_back(pair);
            }
            matC = Assemble(assembleVec);
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());
    std::vector<float> res(capacity_mat_c);
    machine::GetRA()->CopyFromTensor((uint8_t *)res.data(), mat_c_ptr, outputSize);

    std::vector<float> golden(capacity_mat_c);
    readInput(GetGoldenDir() + "/mat_c.bin", golden);

    int ret = resultCmp(golden, res, 0.001f);
    EXPECT_EQ(ret, true);
}

TEST_F(OnBoardTest, test_BMM_NZ_1_128_256_128_Batch) {
    TestNZFormatBatch<DataType::DT_FP16, DataType::DT_FP32, false, true>(2, 128, 128, 256);
}

TEST_F(OnBoardTest, test_BMMT_NZ_1_128_256_128_Batch) {
    TestNZFormatBatch<DataType::DT_FP16, DataType::DT_FP32, false, true, true>(2, 128, 128, 256);
}

template <DataType inputType, DataType outputType, bool IsANZ = false, bool IsBNZ = false>
void TestNZFormatACC(int bs, int m, int k, int n) {
    aclInit(nullptr);
    rtSetDevice(GetDeviceIdByEnvVar());

    const int capacity_mat_a = bs * m * k;
    const int capacity_mat_b = bs * k * n;
    const int capacity_mat_c = bs * m * n;

    std::vector<int64_t> shape_a = {m, k};
    std::vector<int64_t> shape_b = { k, n};
    std::vector<int64_t> shape_c = { m, n};

    void *mat_a_ptr = readToDev<__uint16_t>(GetGoldenDir() + "/mat_a.bin", capacity_mat_a);
    void *mat_b_ptr = readToDev<__uint16_t>(GetGoldenDir() + "/mat_b.bin", capacity_mat_b);
    assert(mat_a_ptr != nullptr && mat_b_ptr != nullptr);
    uint32_t outputSize = capacity_mat_c * sizeof(float);
    uint8_t* mat_c_ptr = allocDevAddr(outputSize);
    auto kSplit = 4;
    auto kSplitSize = k / kSplit;
    auto afmt = IsANZ ? TileOpFormat::TILEOP_NZ : TileOpFormat::TILEOP_ND;
    auto bfmt = IsBNZ ? TileOpFormat::TILEOP_NZ : TileOpFormat::TILEOP_ND;
    Tensor mat_a(inputType, shape_a, (uint8_t *)mat_a_ptr, "MatA", afmt);
    Tensor mat_b(inputType, shape_b, (uint8_t *)mat_b_ptr, "MatB", bfmt);
    Tensor mat_c(outputType, shape_c, mat_c_ptr, "MatC");

    config::SetBuildStatic(true);
    FUNCTION("Matmul_T", {mat_a, mat_b, mat_c}) {
        TileShape::Current().SetVecTile(64, 64);
        Tensor tmpC(outputType, shape_c, "tmp_c");
        tmpC = Mul(tmpC, Element(DataType::DT_FP32, 0.0f));
        std::vector<Tensor> matmulResult;
        for (int ki = 0; ki < kSplit; ki++) {
            auto input_mk = View(mat_a, {m, kSplitSize}, {0, ki * kSplitSize});
            auto input_kn = View(mat_b, {kSplitSize, n}, {ki * kSplitSize, 0});
            TileShape::Current().SetCubeTile({16, 16}, {128, 128}, {128, 128});
            auto tmpC1 = Matrix::Matmul<false, false>(outputType, input_mk, input_kn, tmpC);
            matmulResult.emplace_back(tmpC1);
        }
        TileShape::Current().SetVecTile(16, 128);
        tmpC = npu::tile_fwk::Reduce(matmulResult, ReduceMode::ATOMIC_ADD);
        TileShape::Current().SetVecTile(16, 128);
        mat_c = Add(tmpC, Element(DataType::DT_FP32, 0.0));
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());

    std::vector<float> res(capacity_mat_c);
    machine::GetRA()->CopyFromTensor((uint8_t *)res.data(), mat_c_ptr, outputSize);

    std::vector<float> golden(capacity_mat_c);
    readInput(GetGoldenDir() + "/mat_c.bin", golden);

    int ret = resultCmp(golden, res, 0.001f);
    EXPECT_EQ(ret, true);
}

TEST_F(OnBoardTest, test_BMM_NZ_1_4_4096_7168_ACC) {
    TileShape::Current().SetMatrixSize({4, 4096, 7168});
    TestNZFormatACC<DataType::DT_FP16, DataType::DT_FP32, false, true>(1, 4, 4096, 7168);
}

TEST_F(OnBoardTest, test_BMM_NZ_1_128_128_128_ACC) {
    TileShape::Current().SetMatrixSize({128, 128, 128});
    TestNZFormatACC<DataType::DT_FP16, DataType::DT_FP32, false, true>(1, 128, 128, 128);
}

TEST_F(OnBoardTest, test_BMM_NZ_1_32_2048_7168_ACC) {
    TestNZFormatACC<DataType::DT_FP16, DataType::DT_FP32, false, true>(1, 32, 7168, 2048);
}
