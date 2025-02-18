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
 * \file test_transpose.cpp
 * \brief
 */

#include "test_suite_stest_ops.h"
#include "test_dev_func_runner.h"

namespace {
int capacity;
}; // namespace

void TransposePre(uint8_t** out_ptr, uint64_t* outsize) {
    aclInit(nullptr);
    rtSetDevice(GetDeviceIdByEnvVar());
    *outsize = capacity * sizeof(float);
    *out_ptr = allocDevAddr(*outsize);
}

void TransposePost(uint8_t* outputGmAddr, uint64_t outputSize) {
    std::vector<float> golden(capacity);
    std::vector<float> res(capacity);
    machine::GetRA()->CopyFromTensor((uint8_t *)res.data(), (uint8_t *)outputGmAddr, outputSize);
    readInput(GetGoldenDir() + "/res.bin", golden);
    int ret = resultCmp(golden, res, 0.001f, 64);
    EXPECT_EQ(ret, true);
}

class TransposeTest : public npu::tile_fwk::stest::TestSuite_STest_Ops_Aihac {};

TEST_F(TransposeTest, TestTranspose_BNSD_BSND) {
    int b = 2;
    int n = 32;
    int s = 16;
    int d = 16;
    capacity = b * n * s * d;
    std::vector<int64_t> shape{b, n, s, d};
    std::vector<int64_t> resShape{b, s, n, d};
    uint8_t* out_ptr = nullptr;
    uint64_t outSize = 0;
    TransposePre(&out_ptr, &outSize);
    PROGRAM("Transpose") {
        TileShape::Current().SetVecTile(1, 16, 16, 16);
        void *input_ptr = readToDev(GetGoldenDir() + "/input.bin", capacity);
        Tensor input(DataType::DT_FP32, shape, (uint8_t *)input_ptr, "input");
        Tensor output(DataType::DT_FP32, resShape, out_ptr, "res");
        config::SetBuildStatic(true);
        FUNCTION("BNSD_BSND", {input, output}) {
            output = Transpose(input, {1, 2});
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());
    TransposePost(out_ptr, outSize);
}

TEST_F(TransposeTest, TestTranspose_ABC_BAC) {
    int bs = 128;
    int n = 2;
    int d = 128;
    capacity = bs * n * d;
    std::vector<int64_t> shape{bs, n, d};
    std::vector<int64_t> resShape{n, bs, d};
    uint8_t* out_ptr = nullptr;
    uint64_t outSize = 0;
    TransposePre(&out_ptr, &outSize);
    PROGRAM("Transpose") {
        TileShape::Current().SetVecTile(32, 1, 128);
        void *input_ptr = readToDev(GetGoldenDir() + "/input.bin", capacity);
        Tensor input(DataType::DT_FP32, shape, (uint8_t *)input_ptr, "input");
        Tensor output(DataType::DT_FP32, resShape, out_ptr, "res");
        config::SetBuildStatic(true);
        FUNCTION("ABC_BAC", {input, output}) {
            output = Transpose(input, {0, 1});
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());
    TransposePost(out_ptr, outSize);
}

TEST_F(TransposeTest, TestTranspose_ABCD_ABDC_1_2_16_31) {
    int a = 1;
    int b = 2;
    int c = 16;
    int d = 31;
    capacity = a * b * c * d;
    std::vector<int64_t> shape{a,b,c,d};
    std::vector<int64_t> resShape{a,b,d,c};
    uint8_t* out_ptr = nullptr;
    uint64_t outSize = 0;
    TransposePre(&out_ptr, &outSize);
    cout << "outSize: " << outSize << endl;
    PROGRAM("Transpose") {
        TileShape::Current().SetVecTile(1, 2, 16, 31);
        void *input_ptr = readToDev(GetGoldenDir() + "/input.bin", capacity);
        Tensor input(DT_FP32, shape, (uint8_t *)input_ptr, "input");
        Tensor output(DT_FP32, resShape, out_ptr, "res");
        config::SetBuildStatic(true);
        FUNCTION("ABC_BAC", {input, output}) {
            output = Transpose(input, {2, 3});
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());
    TransposePost(out_ptr, outSize);
}

TEST_F(TransposeTest, TestTranspose_BNSD2_BNS2D_small) {
    int b = 2;
    int n = 4;
    int s = 32;
    int d = 64;
    capacity = b * n * s * d;
    std::vector<int64_t> shape{b, n, s, d / 2, 2};
    std::vector<int64_t> resShape{b, n, s, 2, d / 2};
    uint8_t* out_ptr = nullptr;
    uint64_t outSize = 0;
    TransposePre(&out_ptr, &outSize);
    PROGRAM("Transpose") {
        TileShape::Current().SetVecTile(1, 2, 32, 32, 2);
        void *input_ptr = readToDev(GetGoldenDir() + "/input.bin", capacity);
        Tensor input(DataType::DT_FP32, shape, (uint8_t *)input_ptr, "input");
        Tensor output(DataType::DT_FP32, resShape, out_ptr, "res");
        config::SetBuildStatic(true);
        FUNCTION("BNSD2_BNS2D_small", {input, output}) {
            output = Transpose(input, {3, 4});
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());
    TransposePost(out_ptr, outSize);
}

TEST_F(TransposeTest, TestTranspose_BNDS_BNSD) {
    int b = 2;
    int n = 32;
    int s = 32;
    int d = 32;
    capacity = b * n * s * d;
    std::vector<int64_t> shape{b, n, s, d};
    std::vector<int64_t> resShape{b, n, d, s};
    uint8_t* out_ptr = nullptr;
    uint64_t outSize = 0;
    TransposePre(&out_ptr, &outSize);
    PROGRAM("Transpose") {
        TileShape::Current().SetVecTile(1, 16, 16, 16);
        void *input_ptr = readToDev(GetGoldenDir() + "/input.bin", capacity);
        Tensor input(DataType::DT_FP32, shape, (uint8_t *)input_ptr, "input");
        Tensor output(DataType::DT_FP32, resShape, out_ptr, "res");
        config::SetBuildStatic(true);
        FUNCTION("BNDS_BNSD", {input, output}) {
            output = Transpose(input, {2, 3});
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());
    TransposePost(out_ptr, outSize);
}

TEST_F(TransposeTest, TestTranspose_AB_BA) {
    int b = 1;
    int n = 1;
    int s = 32;
    int d = 64;
    capacity = b * n * s * d;
    std::vector<int64_t> shape{s, d};
    std::vector<int64_t> resShape{d, s};
    uint8_t* out_ptr = nullptr;
    uint64_t outSize = 0;
    TransposePre(&out_ptr, &outSize);
    PROGRAM("Transpose") {
        TileShape::Current().SetVecTile(32, 64);
        void *input_ptr = readToDev(GetGoldenDir() + "/input.bin", capacity);
        Tensor input(DataType::DT_FP32, shape, (uint8_t *)input_ptr, "input");
        Tensor output(DataType::DT_FP32, resShape, out_ptr, "res");
        config::SetBuildStatic(true);
        FUNCTION("AB_BA", {input, output}) {
            output = Transpose(input, {0, 1});
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());
    TransposePost(out_ptr, outSize);
}

TEST_F(TransposeTest, TestTranspose_ROPE_5D) {
    int b = 4;
    int n = 64;
    int s = 1;
    int d = 64;
    std::vector<int64_t> shape{b, n, s, d / 2, 2};
    std::vector<int64_t> resShape{b, n, s, 2, d / 2};
    capacity = b * n * s * d;
    uint8_t* out_ptr = nullptr;
    uint64_t outSize = 0;
    TransposePre(&out_ptr, &outSize);
    PROGRAM("Transpose") {
        TileShape::Current().SetVecTile(1, 2, 1, 32, 2);
        void *input_ptr = readToDev(GetGoldenDir() + "/input.bin", capacity);
        Tensor input(DataType::DT_FP32, shape, (uint8_t *)input_ptr, "input");
        Tensor output(DataType::DT_FP32, resShape, out_ptr, "res");
        config::SetBuildStatic(true);
        FUNCTION("ROPE_5D", {input, output}) {
            output = Transpose(input, {3, 4});
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());
    TransposePost(out_ptr, outSize);
}

TEST_F(TransposeTest, TestTranspose_MLA_3D_0) {
    int bs = 32;
    int n = 32;
    int d = 64;
    std::vector<int64_t> shape{bs, n, d};
    std::vector<int64_t> resShape{bs, n, d};
    capacity = bs * n * d;
    uint8_t* out_ptr = nullptr;
    uint64_t outSize = 0;
    TransposePre(&out_ptr, &outSize);
    PROGRAM("Transpose") {
        TileShape::Current().SetVecTile(4, 4, 64);
        void *input_ptr = readToDev(GetGoldenDir() + "/input.bin", capacity);
        Tensor input(DataType::DT_FP32, shape, (uint8_t *)input_ptr, "input");
        Tensor output(DataType::DT_FP32, resShape, out_ptr, "res");
        config::SetBuildStatic(true);
        FUNCTION("MLA_3D_0", {input, output}) {
            output = Transpose(input, {0, 1});
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());
    TransposePost(out_ptr, outSize);
}

TEST_F(TransposeTest, TestTranspose_MLA_3D_1) {
    int bs = 32;
    int n = 32;
    int d = 512;
    std::vector<int64_t> shape{bs, n, d};
    std::vector<int64_t> resShape{bs, n, d};
    capacity = bs * n * d;
    uint8_t* out_ptr = nullptr;
    uint64_t outSize = 0;
    TransposePre(&out_ptr, &outSize);
    PROGRAM("Transpose") {
        TileShape::Current().SetVecTile(2, 2, 512);
        void *input_ptr = readToDev(GetGoldenDir() + "/input.bin", capacity);
        Tensor input(DataType::DT_FP32, shape, (uint8_t *)input_ptr, "input");
        Tensor output(DataType::DT_FP32, resShape, out_ptr, "res");
        config::SetBuildStatic(true);
        FUNCTION("MLA_3D_1", {input, output}) {
            output = Transpose(input, {0, 1});
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());
    TransposePost(out_ptr, outSize);
}

TEST_F(TransposeTest, TestTranspose_MLA_4D_0) {
    int b = 32;
    int n = 1;
    int s = 32;
    int d = 512;
    std::vector<int64_t> shape{b, n, s, d};
    std::vector<int64_t> resShape{b, s, n, d};
    capacity = b * n * s * d;
    uint8_t* out_ptr = nullptr;
    uint64_t outSize = 0;
    TransposePre(&out_ptr, &outSize);
    PROGRAM("Transpose") {
        TileShape::Current().SetVecTile(1, 1, 32, 512);
        void *input_ptr = readToDev(GetGoldenDir() + "/input.bin", capacity);
        Tensor input(DataType::DT_FP32, shape, (uint8_t *)input_ptr, "input");
        Tensor output(DataType::DT_FP32, resShape, out_ptr, "res");
        config::SetBuildStatic(true);
        FUNCTION("MLA_4D_0", {input, output}) {
            output = Transpose(input, {1, 2});
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());
    TransposePost(out_ptr, outSize);
}

TEST_F(TransposeTest, TestTranspose_MLA_4D_1) {
    int b = 32;
    int n = 1;
    int s = 32;
    int d = 64;
    std::vector<int64_t> shape{b, n, s, d};
    std::vector<int64_t> resShape{b, s, n, d};
    capacity = b * n * s * d;
    uint8_t* out_ptr = nullptr;
    uint64_t outSize = 0;
    TransposePre(&out_ptr, &outSize);
    PROGRAM("Transpose") {
        TileShape::Current().SetVecTile(1, 1, 32, 64);
        void *input_ptr = readToDev(GetGoldenDir() + "/input.bin", capacity);
        Tensor input(DataType::DT_FP32, shape, (uint8_t *)input_ptr, "input");
        Tensor output(DataType::DT_FP32, resShape, out_ptr, "res");
        config::SetBuildStatic(true);
        FUNCTION("MLA_4D_1", {input, output}) {
            output = Transpose(input, {1, 2});
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());
    TransposePost(out_ptr, outSize);
}

TEST_F(TransposeTest, TestTranspose_MLA_4D_3) {
    int b = 32;
    int n = 1;
    int s = 1;
    int d = 64;
    std::vector<int64_t> shape{b, n, s, d};
    std::vector<int64_t> resShape{b, s, n, d};
    capacity = b * n * s * d;
    uint8_t* out_ptr = nullptr;
    uint64_t outSize = 0;
    TransposePre(&out_ptr, &outSize);
    PROGRAM("Transpose") {
        TileShape::Current().SetVecTile(1, 1, 32, 64);
        void *input_ptr = readToDev(GetGoldenDir() + "/input.bin", capacity);
        Tensor input(DataType::DT_FP32, shape, (uint8_t *)input_ptr, "input");
        Tensor output(DataType::DT_FP32, resShape, out_ptr, "res");
        config::SetBuildStatic(true);
        FUNCTION("MLA_4D_3", {input, output}) {
            output = Add(input, input);
            output = Transpose(output, {1, 2});
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());
    TransposePost(out_ptr, outSize);
}

TEST_F(TransposeTest, TestTranspose_MLA_4D_4) {
    int b = 32;
    int n = 1;
    int s = 1;
    int d = 512;
    std::vector<int64_t> shape{b, n, s, d};
    std::vector<int64_t> resShape{b, s, n, d};
    capacity = b * n * s * d;
    uint8_t* out_ptr = nullptr;
    uint64_t outSize = 0;
    TransposePre(&out_ptr, &outSize);
    PROGRAM("Transpose") {
        TileShape::Current().SetVecTile(1, 1, 32, 512);
        void *input_ptr = readToDev(GetGoldenDir() + "/input.bin", capacity);
        Tensor input(DataType::DT_FP32, shape, (uint8_t *)input_ptr, "input");
        Tensor output(DataType::DT_FP32, resShape, out_ptr, "res");
        config::SetBuildStatic(true);
        FUNCTION("MLA_4D_4", {input, output}) {
            output = Add(input, input);
            output = Transpose(output, {1, 2});
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());
    TransposePost(out_ptr, outSize);
}

TEST_F(TransposeTest, TestTranspose_MLA_4D_5) {
    int b = 32;
    int n = 1;
    int s = 256;
    int d = 512 + 64;
    std::vector<int64_t> shape{b, n, s, d};
    std::vector<int64_t> resShape{b, s, n, d};
    capacity = b * n * s * d;
    uint8_t* out_ptr = nullptr;
    uint64_t outSize = 0;
    TransposePre(&out_ptr, &outSize);
    PROGRAM("Transpose") {
        TileShape::Current().SetVecTile(1, 1, 16, d);
        void *input_ptr = readToDev(GetGoldenDir() + "/input.bin", capacity);
        Tensor input(DataType::DT_FP32, shape, (uint8_t *)input_ptr, "input");
        Tensor output(DataType::DT_FP32, resShape, out_ptr, "res");
        config::SetBuildStatic(true);
        FUNCTION("MLA_4D_5", {input, output}) {
            output = Transpose(input, {2, 3});
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());
    TransposePost(out_ptr, outSize);
}

TEST_F(TransposeTest, TestTranspose_MLA_4D_50) {
    int b = 1;
    int n = 1;
    int s = 128;
    int d = 128;
    std::vector<int64_t> shape{b, n, s, d};
    std::vector<int64_t> resShape{b, s, n, d};
    capacity = b * n * s * d;
    uint8_t* out_ptr = nullptr;
    uint64_t outSize = 0;
    TransposePre(&out_ptr, &outSize);
    PROGRAM("Transpose") {
        TileShape::Current().SetVecTile(1, 1, s, d);
        void *input_ptr = readToDev(GetGoldenDir() + "/input.bin", capacity);
        Tensor input(DataType::DT_FP32, shape, (uint8_t *)input_ptr, "input");
        Tensor output(DataType::DT_FP32, resShape, out_ptr, "res");
        config::SetBuildStatic(true);
        FUNCTION("MLA_4D_50", {input, output}) {
            output = Transpose(input, {2, 3});
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());
    TransposePost(out_ptr, outSize);
}

TEST_F(TransposeTest, TestTranspose_MLA_4D_6) {
    int b = 32;
    int n = 32;
    int s = 1;
    int d = 512;
    std::vector<int64_t> shape{b, n, s, d};
    std::vector<int64_t> resShape{b, s, n, d};
    capacity = b * n * s * d;
    uint8_t* out_ptr = nullptr;
    uint64_t outSize = 0;
    TransposePre(&out_ptr, &outSize);
    PROGRAM("Transpose") {
        //  1,1,1,512 ok
        TileShape::Current().SetVecTile(4, 4, 1, 512);
        void *input_ptr = readToDev(GetGoldenDir() + "/input.bin", capacity);
        Tensor input(DataType::DT_FP32, shape, (uint8_t *)input_ptr, "input");
        Tensor output(DataType::DT_FP32, resShape, out_ptr, "res");
        config::SetBuildStatic(true);
        FUNCTION("MLA_4D_6", {input, output}) {
            output = Transpose(input, {1, 2});
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());
    TransposePost(out_ptr, outSize);
}

TEST_F(TransposeTest, TestTranspose_MLA_3D_2) {
    int bs = 32;
    int n = 32;
    int d = 128;
    std::vector<int64_t> shape{bs, n, d};
    std::vector<int64_t> resShape{bs, n, d};
    capacity = bs * n * d;
    uint8_t* out_ptr = nullptr;
    uint64_t outSize = 0;
    TransposePre(&out_ptr, &outSize);
    PROGRAM("Transpose") {
        TileShape::Current().SetVecTile(2, 2, 128);
        void *input_ptr = readToDev(GetGoldenDir() + "/input.bin", capacity);
        Tensor input(DataType::DT_FP32, shape, (uint8_t *)input_ptr, "input");
        Tensor output(DataType::DT_FP32, resShape, out_ptr, "res");
        config::SetBuildStatic(true);
        FUNCTION("MLA_3D_2", {input, output}) {
            output = Transpose(input, {0, 1});
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());
    TransposePost(out_ptr, outSize);
}

TEST_F(TransposeTest, TestTranspose_MLA_4D_7) {
    int b = 32;
    int n = 128;
    int s = 1;
    int d = 512;
    std::vector<int64_t> shape{b, n, s, d};
    std::vector<int64_t> resShape{n, b, n, d};
    capacity = b * n * s * d;
    uint8_t* out_ptr = nullptr;
    uint64_t outSize = 0;
    TransposePre(&out_ptr, &outSize);
    PROGRAM("Transpose") {
        TileShape::Current().SetVecTile(4, 8, s, d);
        void *input_ptr = readToDev(GetGoldenDir() + "/input.bin", capacity);
        Tensor input(DT_FP32, shape, (uint8_t *)input_ptr, "input");
        Tensor output(DT_FP32, resShape, out_ptr, "res");
        config::SetBuildStatic(true);
        FUNCTION("MLA_4D_1", {input, output}) {
            auto tmp = Transpose(input, {0, 1});
            output = Add(tmp, Element(DataType::DT_FP32, 0.0));
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());
    TransposePost(out_ptr, outSize);
}

TEST_F(TransposeTest, Test_Datamove_Nonalign_Dim4) {
    int b = 32;
    int n = 1;
    int s = 32;
    int d = 437;
    std::vector<int64_t> shape{b, n, s, d};
    std::vector<int64_t> resShape{b, s, n, d};
    capacity = b * n * s * d;
    uint8_t* out_ptr = nullptr;
    uint64_t outSize = 0;
    TransposePre(&out_ptr, &outSize);
    PROGRAM("Transpose") {
        TileShape::Current().SetVecTile(1, 1, 32, 512);
        void *input_ptr = readToDev(GetGoldenDir() + "/input.bin", capacity);
        Tensor input(DataType::DT_FP32, shape, (uint8_t *)input_ptr, "input");
        Tensor output(DataType::DT_FP32, resShape, out_ptr, "res");
        config::SetBuildStatic(true);
        FUNCTION("MLA_4D_0", {input, output}) {
            output = Transpose(input, {1, 2});
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());
    TransposePost(out_ptr, outSize);
}

TEST_F(TransposeTest, Test_Datamove_Nonalign_Dim3) {
    int n = 1;
    int s = 32;
    int d = 437;
    std::vector<int64_t> shape{n, s, d};
    std::vector<int64_t> resShape{s, n, d};
    capacity = n * s * d;
    uint8_t* out_ptr = nullptr;
    uint64_t outSize = 0;
    TransposePre(&out_ptr, &outSize);
    PROGRAM("Transpose") {
        TileShape::Current().SetVecTile(1, 32, 512);
        void *input_ptr = readToDev(GetGoldenDir() + "/input.bin", capacity);
        Tensor input(DataType::DT_FP32, shape, (uint8_t *)input_ptr, "input");
        Tensor output(DataType::DT_FP32, resShape, out_ptr, "res");
        config::SetBuildStatic(true);
        FUNCTION("MLA_4D_0", {input, output}) {
            output = Transpose(input, {0, 1});
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());
    TransposePost(out_ptr, outSize);
}

TEST_F(TransposeTest, Test_AB_BA_32_768) {
    int a = 32;
    int b = 768;
    std::vector<int64_t> shape{a,b};
    std::vector<int64_t> resShape{b,a};
    capacity = a * b ;
    uint8_t* out_ptr = nullptr;
    uint64_t outSize = 0;
    TransposePre(&out_ptr, &outSize);
    PROGRAM("Transpose") {
        TileShape::Current().SetVecTile(a, b);
        void *input_ptr = readToDev(GetGoldenDir() + "/input.bin", capacity);
        Tensor input(DataType::DT_FP32, shape, (uint8_t *)input_ptr, "input");
        Tensor output(DataType::DT_FP32, resShape, out_ptr, "res");
        config::SetBuildStatic(true);
        FUNCTION("MLA_4D_0", {input, output}) {
            output = Transpose(input, {0, 1});
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());
    TransposePost(out_ptr, outSize);
}

TEST_F(TransposeTest, Test_AB_BA_768_32) {
    int a = 768;
    int b = 32;
    std::vector<int64_t> shape{a,b};
    std::vector<int64_t> resShape{b,a};
    capacity = a * b ;
    uint8_t* out_ptr = nullptr;
    uint64_t outSize = 0;
    TransposePre(&out_ptr, &outSize);
    PROGRAM("Transpose") {
        TileShape::Current().SetVecTile(a, b);
        void *input_ptr = readToDev(GetGoldenDir() + "/input.bin", capacity);
        Tensor input(DataType::DT_FP32, shape, (uint8_t *)input_ptr, "input");
        Tensor output(DataType::DT_FP32, resShape, out_ptr, "res");
        config::SetBuildStatic(true);
        FUNCTION("MLA_4D_0", {input, output}) {
            output = Transpose(input, {0, 1});
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());
    TransposePost(out_ptr, outSize);
}

TEST_F(TransposeTest, Test_AB_BA_128_511) {
    int a = 128;
    int b = 511;
    std::vector<int64_t> shape{a,b};
    std::vector<int64_t> resShape{b,a};
    capacity = a * b ;
    uint8_t* out_ptr = nullptr;
    uint64_t outSize = 0;
    TransposePre(&out_ptr, &outSize);
    PROGRAM("Transpose") {
        TileShape::Current().SetVecTile(NUM16, b);
        void *input_ptr = readToDev(GetGoldenDir() + "/input.bin", capacity);
        Tensor input(DataType::DT_FP32, shape, (uint8_t *)input_ptr, "input");
        Tensor output(DataType::DT_FP32, resShape, out_ptr, "res");
        config::SetBuildStatic(true);
        FUNCTION("MLA_4D_0", {input, output}) {
            output = Transpose(input, {0, 1});
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());
    TransposePost(out_ptr, outSize);
}

TEST_F(TransposeTest, Test_AB_BA_128_255) {
    int a = 128;
    int b = 255;
    std::vector<int64_t> shape{a,b};
    std::vector<int64_t> resShape{b,a};
    capacity = a * b ;
    uint8_t* out_ptr = nullptr;
    uint64_t outSize = 0;
    TransposePre(&out_ptr, &outSize);
    PROGRAM("Transpose") {
        TileShape::Current().SetVecTile(NUM16, b);
        void *input_ptr = readToDev(GetGoldenDir() + "/input.bin", capacity);
        Tensor input(DataType::DT_FP32, shape, (uint8_t *)input_ptr, "input");
        Tensor output(DataType::DT_FP32, resShape, out_ptr, "res");
        config::SetBuildStatic(true);
        FUNCTION("MLA_4D_0", {input, output}) {
            output = Transpose(input, {0, 1});
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());
    TransposePost(out_ptr, outSize);
}


TEST_F(TransposeTest, Test_AB_BA_128_63) {
    int a = 128;
    int b = 63;
    std::vector<int64_t> shape{a,b};
    std::vector<int64_t> resShape{b,a};
    capacity = a * b ;
    uint8_t* out_ptr = nullptr;
    uint64_t outSize = 0;
    TransposePre(&out_ptr, &outSize);
    PROGRAM("Transpose") {
        TileShape::Current().SetVecTile(NUM16, b);
        void *input_ptr = readToDev(GetGoldenDir() + "/input.bin", capacity);
        Tensor input(DataType::DT_FP32, shape, (uint8_t *)input_ptr, "input");
        Tensor output(DataType::DT_FP32, resShape, out_ptr, "res");
        config::SetBuildStatic(true);
        FUNCTION("MLA_4D_0", {input, output}) {
            output = Transpose(input, {0, 1});
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());
    TransposePost(out_ptr, outSize);
}

TEST_F(TransposeTest, Test_AB_BA_1_128_511) {
    int a = 128;
    int b = 511;
    std::vector<int64_t> shape{1,a,b};
    std::vector<int64_t> resShape{1,b,a};
    capacity = a * b ;
    uint8_t* out_ptr = nullptr;
    uint64_t outSize = 0;
    TransposePre(&out_ptr, &outSize);
    PROGRAM("Transpose") {
        TileShape::Current().SetVecTile(1, NUM2, b);
        void *input_ptr = readToDev(GetGoldenDir() + "/input.bin", capacity);
        Tensor input(DataType::DT_FP32, shape, (uint8_t *)input_ptr, "input");
        Tensor output(DataType::DT_FP32, resShape, out_ptr, "res");
        config::SetBuildStatic(true);
        FUNCTION("MLA_4D_0", {input, output}) {
            output = Transpose(input, {1, 2});
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());
    TransposePost(out_ptr, outSize);
}

TEST_F(TransposeTest, Test_AB_BA_1_128_255) {
    int a = 128;
    int b = 255;
    std::vector<int64_t> shape{1,a,b};
    std::vector<int64_t> resShape{1,b,a};
    capacity = a * b ;
    uint8_t* out_ptr = nullptr;
    uint64_t outSize = 0;
    TransposePre(&out_ptr, &outSize);
    PROGRAM("Transpose") {
        TileShape::Current().SetVecTile(1, NUM16, b);
        void *input_ptr = readToDev(GetGoldenDir() + "/input.bin", capacity);
        Tensor input(DataType::DT_FP32, shape, (uint8_t *)input_ptr, "input");
        Tensor output(DataType::DT_FP32, resShape, out_ptr, "res");
        config::SetBuildStatic(true);
        FUNCTION("MLA_4D_0", {input, output}) {
            output = Transpose(input, {1,2});
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());
    TransposePost(out_ptr, outSize);
}


TEST_F(TransposeTest, Test_AB_BA_1_128_63) {
    int a = 128;
    int b = 63;
    std::vector<int64_t> shape{1,a,b};
    std::vector<int64_t> resShape{1,b,a};
    capacity = a * b ;
    uint8_t* out_ptr = nullptr;
    uint64_t outSize = 0;
    TransposePre(&out_ptr, &outSize);
    PROGRAM("Transpose") {
        TileShape::Current().SetVecTile(1, NUM16, b);
        void *input_ptr = readToDev(GetGoldenDir() + "/input.bin", capacity);
        Tensor input(DataType::DT_FP32, shape, (uint8_t *)input_ptr, "input");
        Tensor output(DataType::DT_FP32, resShape, out_ptr, "res");
        config::SetBuildStatic(true);
        FUNCTION("MLA_4D_0", {input, output}) {
            output = Transpose(input, {1,2});
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());
    TransposePost(out_ptr, outSize);
}


TEST_F(TransposeTest, TestTranspose_abcd_bacd_2_128_3_32) {
    int b = 2;
    int n = 128;
    int s = 3;
    int d = 32;
    std::vector<int64_t> shape{b, n, s, d};
    std::vector<int64_t> resShape{b, s, n, d};
    capacity = b * n * s * d;
    uint8_t* out_ptr = nullptr;
    uint64_t outSize = 0;
    TransposePre(&out_ptr, &outSize);
    PROGRAM("Transpose") {
        TileShape::Current().SetVecTile(1, 1, s, d);
        void *input_ptr = readToDev(GetGoldenDir() + "/input.bin", capacity);
        Tensor input(DataType::DT_FP32, shape, (uint8_t *)input_ptr, "input");
        Tensor output(DataType::DT_FP32, resShape, out_ptr, "res");
        config::SetBuildStatic(true);
        FUNCTION("MLA_4D_0", {input, output}) {
            output = Transpose(input, {0, 1});
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());
    TransposePost(out_ptr, outSize);;
}
