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
 * \file test_quant.cpp
 * \brief
 */

#include "test_suite_stest_ops.h"
#include "operator/models/llama/llama_def.h"
#include "operator/models/deepseek/deepseek_spec.h"
#include "test_dev_func_runner.h"

namespace {
int capacity;
int INT8_MAX_VALUE = 127;
constexpr float F_127 = 127.0;

void QuantPre(uint8_t **out_ptr, uint64_t *outsize) {
    aclInit(nullptr);
    rtSetDevice(GetDeviceIdByEnvVar());
    *outsize = capacity * sizeof(float);
    *out_ptr = allocDevAddr(*outsize);
}

void QuantPost(uint8_t *outputGmAddr, uint64_t outputSize) {
    std::vector<float> golden(capacity);
    std::vector<float> res(capacity);
    machine::GetRA()->CopyFromTensor((uint8_t *) res.data(), (uint8_t *) outputGmAddr, outputSize);
    readInput(GetGoldenDir() + "/res.bin", golden);
    int ret = resultCmp(golden, res, 0.001f, 64);
    EXPECT_EQ(ret, true);
}
} // namespace

using namespace npu::tile_fwk;

class QuantTest : public npu::tile_fwk::stest::TestSuite_STest_Ops_Aihac {};

TEST_F(QuantTest, Test_quant) {
    config::SetPlatformConfig(KEY_ONLY_HOST_COMPILE, true);
    std::vector<int64_t> vecTileShape = {128, 128};
    int b = 2; // 32
    int s = 1; // 1, optimize set_tile
    int h = std::get<int>(deepseekConfig1["hiddenSize"]);
    std::cout << "Test_deepseekAttention  b,s,h: " << b << ", " << s << ", " << h << std::endl;

    Tensor input = Tensor(DataType::DT_FP16, {b * s, h}, "input");
    Tensor res;

    TileShape::Current().SetCubeTile({std::min(128, s), std::min(128, s)}, {256, 256}, {64, 64});
    TileShape::Current().SetVecTile(vecTileShape[0], vecTileShape[1]); // for Assemble

    FUNCTION("A") {
        res = std::get<0>(Quant(input));
    }

    std::cout << Program::GetInstance().Dump() << std::endl;
}

TEST_F(QuantTest, Test_ScalarDivS) {
    std::vector<int64_t> vecTileShape = {128, 128};
    int b = 2; // 32
    int s = 1; // 1, optimize set_tile
    std::vector<int64_t> shape{b * s, 1};
    capacity = b * s;
    uint8_t *out_ptr = nullptr;
    uint64_t outSize = 0;
    QuantPre(&out_ptr, &outSize);
    PROGRAM("Quant") {
        TileShape::Current().SetVecTile(vecTileShape[0], vecTileShape[1]);
        void *input_ptr = readToDev(GetGoldenDir() + "/input.bin", capacity);
        Tensor input(DataType::DT_FP32, shape, (uint8_t *) input_ptr, "input");
        Tensor output(DataType::DT_FP32, shape, out_ptr, "res");
        config::SetBuildStatic(true);
        FUNCTION("ScalarDivS", {input, output}) {
            output = ScalarDivS(input, Element(DataType::DT_FP32, static_cast<double>(INT8_MAX_VALUE)),
                                true);
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());
    QuantPost(out_ptr, outSize);
}

TEST_F(QuantTest, Test_ScalarAddS) {
    std::vector<int64_t> vecTileShape = {128, 128};
    int b = 2; // 32
    int s = 1; // 1, optimize set_tile
    std::vector<int64_t> shape{b * s, 1};
    capacity = b * s;
    uint8_t *out_ptr = nullptr;
    uint64_t outSize = 0;
    QuantPre(&out_ptr, &outSize);
    PROGRAM("Quant") {
        TileShape::Current().SetVecTile(vecTileShape[0], vecTileShape[1]);
        void *input_ptr = readToDev(GetGoldenDir() + "/input.bin", capacity);
        Tensor input(DataType::DT_FP32, shape, (uint8_t *) input_ptr, "input");
        Tensor output(DataType::DT_FP32, shape, out_ptr, "res");
        config::SetBuildStatic(true);
        FUNCTION("ScalarAddS", {input, output}) {
            output = ScalarAddS(input, Element(DataType::DT_FP32, static_cast<double>(INT8_MAX_VALUE)),
                                true);
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());
    QuantPost(out_ptr, outSize);
}

TEST_F(QuantTest, Test_ScalarSubS) {
    std::vector<int64_t> vecTileShape = {128, 128};
    int b = 2; // 32
    int s = 1; // 1, optimize set_tile
    std::vector<int64_t> shape{b * s, 1};
    capacity = b * s;
    uint8_t *out_ptr = nullptr;
    uint64_t outSize = 0;
    QuantPre(&out_ptr, &outSize);
    PROGRAM("Quant") {
        TileShape::Current().SetVecTile(vecTileShape[0], vecTileShape[1]);
        void *input_ptr = readToDev(GetGoldenDir() + "/input.bin", capacity);
        Tensor input(DataType::DT_FP32, shape, (uint8_t *) input_ptr, "input");
        Tensor output(DataType::DT_FP32, shape, out_ptr, "res");
        config::SetBuildStatic(true);
        FUNCTION("ScalarSubS", {input, output}) {
            output = ScalarSubS(input, Element(DataType::DT_FP32, static_cast<double>(INT8_MAX_VALUE)),
                                true);
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());
    QuantPost(out_ptr, outSize);
}

TEST_F(QuantTest, Test_ScalarMulS) {
    std::vector<int64_t> vecTileShape = {128, 128};
    int b = 2; // 32
    int s = 1; // 1, optimize set_tile
    std::vector<int64_t> shape{b * s, 1};
    capacity = b * s;
    uint8_t *out_ptr = nullptr;
    uint64_t outSize = 0;
    QuantPre(&out_ptr, &outSize);
    PROGRAM("Quant") {
        TileShape::Current().SetVecTile(vecTileShape[0], vecTileShape[1]);
        void *input_ptr = readToDev(GetGoldenDir() + "/input.bin", capacity);
        Tensor input(DataType::DT_FP32, shape, (uint8_t *) input_ptr, "input");
        Tensor output(DataType::DT_FP32, shape, out_ptr, "res");
        config::SetBuildStatic(true);
        FUNCTION("ScalarMulS", {input, output}) {
            output = ScalarMulS(input, Element(DataType::DT_FP32, static_cast<double>(INT8_MAX_VALUE)),
                                true);
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());
    QuantPost(out_ptr, outSize);
}

TEST_F(QuantTest, Test_ScalarMaxS) {
    std::vector<int64_t> vecTileShape = {128, 128};
    int b = 2; // 32
    int s = 1; // 1, optimize set_tile
    std::vector<int64_t> shape{b * s, 1};
    capacity = b * s;
    uint8_t *out_ptr = nullptr;
    uint64_t outSize = 0;
    QuantPre(&out_ptr, &outSize);
    PROGRAM("Quant") {
        TileShape::Current().SetVecTile(vecTileShape[0], vecTileShape[1]);
        void *input_ptr = readToDev(GetGoldenDir() + "/input.bin", capacity);
        Tensor input(DataType::DT_FP32, shape, (uint8_t *) input_ptr, "input");
        Tensor output(DataType::DT_FP32, shape, out_ptr, "res");
        config::SetBuildStatic(true);
        FUNCTION("ScalarMaxS", {input, output}) {
            output = ScalarMaxS(input, Element(DataType::DT_FP32, static_cast<double>(INT8_MAX_VALUE-1)),
                                true);
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());
    QuantPost(out_ptr, outSize);
}

TEST_F(QuantTest, Test_ScalarOp) {
    std::vector<int64_t> vecTileShape = {128, 128};
    int b = 2; // 32
    int s = 1; // 1, optimize set_tile
    std::vector<int64_t> shape{b * s, 35};
    capacity = b * s;
    uint8_t *out_ptr = nullptr;
    uint64_t outSize = 0;
    QuantPre(&out_ptr, &outSize);
    PROGRAM("Quant") {
        TileShape::Current().SetVecTile(vecTileShape[0], vecTileShape[1]);
        void *input_ptr = readToDev(GetGoldenDir() + "/input.bin", capacity);
        Tensor input(DataType::DT_FP32, shape, (uint8_t *) input_ptr, "input");
        Tensor output(DataType::DT_FP32, shape, out_ptr, "res");
        config::SetBuildStatic(true);
        FUNCTION("ScalarAddS", {input, output}) {
            auto output_a = ScalarAddS(input, Element(DataType::DT_FP32, F_127), true);
            auto output_b = ScalarSubS(output_a, Element(DataType::DT_FP32, F_127),
                true);
            auto output_c = ScalarMulS(output_b, Element(DataType::DT_FP32, F_127),
                true);
            auto output_d = ScalarDivS(output_c, Element(DataType::DT_FP32, F_127),
                true);
            output = ScalarMaxS(output_d, Element(DataType::DT_FP32, F_127), true);
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());
    QuantPost(out_ptr, outSize);
}
