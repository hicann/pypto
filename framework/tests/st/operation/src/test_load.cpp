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
 * \file test_load.cpp
 * \brief
 */

#include <gtest/gtest.h>
#include <random>
#include "interface/tensor/float.h"
#include "tilefwk/data_type.h"
#include "tilefwk/symbolic_scalar.h"
#include "interface/program/program.h"
#include "test_suite_stest_ops.h"
#include "interface/interpreter/raw_tensor_data.h"
#include "test_dev_func_runner.h"

using namespace npu::tile_fwk;
using namespace npu::tile_fwk::dynamic;

class LoadTest : public npu::tile_fwk::stest::TestSuite_STest_Ops_Aihac {
    void SetUp() override {
        TestSuite_STest_Ops_Aihac::SetUp();
        config::SetHostOption(COMPILE_STAGE, GEN_KERNEL_CODE);
        rtSetDevice(GetDeviceIdByEnvVar());
    }
    void TearDown() override {
        config::SetHostOption(COMPILE_STAGE, 0);
        TestSuite_STest_Ops_Aihac::TearDown();
    }
};

TEST_F(LoadTest, test) {
    random_device seed;
    ranlux48 engine(seed());
    uniform_real_distribution<float> floatRandom(-100, 100);

    constexpr int64_t SRC0 = 2048;
    constexpr int64_t SRC1 = 1280;
    constexpr int64_t DST0 = 256;
    constexpr int64_t DST1 = 512;

    auto TotalSize = [](const std::vector<int64_t> &shapes) {
        size_t res = 1;
        for (auto v : shapes) {
            res *= v;
        }
        return res;
    };

    auto Simu = [](const std::vector<float16> &src, const std::vector<int32_t> &offsets) {
        std::vector<float16> dst(offsets.size());
        for (size_t i = 0; i < dst.size(); i++) {
            ASSERT(static_cast<size_t>(offsets[i]) < src.size());
            dst[i] = src[offsets[i]];
        }
        return dst;
    };

    Shape srcShapes{SRC0, SRC1};
    Shape dstShapes{DST0, DST1};
    Shape offsetsShapes{DST0, DST1};
    ASSERT_TRUE(offsetsShapes == dstShapes);

    Tensor src(DT_FP16, srcShapes, "src");
    Tensor offsets(DT_INT32, offsetsShapes, "offsets");
    Tensor dst(DT_FP16, dstShapes, "dst");

    std::vector<float16> srcData(TotalSize(srcShapes));
    for (auto &v : srcData) {
        v = floatRandom(engine);
    }
    uniform_int_distribution<int32_t> intRandom(0, TotalSize(srcShapes) - 1);
    std::vector<int32_t> offsetsData(TotalSize(offsetsShapes));
    for (auto &v : offsetsData) {
        v = intRandom(engine);
    }
    auto golden = Simu(srcData, offsetsData);
    std::cout << "simu finished" << std::endl;

    FUNCTION("test", {src, offsets}, {dst}) {
        LOOP("LOOP", FunctionType::DYNAMIC_LOOP, sIdx, LoopRange(0, 1, 1)) {
            (void)sIdx;
            TileShape::Current().SetVecTile(128, 128);
            dst = Load(src, offsets);
        }
    }
    std::cout << "compile finished" << std::endl;

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateTensor<float16>(src, srcData),
        RawTensorData::CreateTensor<int32_t>(offsets, offsetsData),
    });
    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<float16>(dst, 0),
    });

    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());
    auto out = npu::tile_fwk::ProgramData::GetInstance().GetOutputData(0);
    int maxErrorPrintNum = 50;
    int curErrorPrintNum = 0;
    float eps = 1e-6f;
    for (size_t i = 0; i < golden.size(); i++) {
        auto actual = ((float16 *)out->data())[i];
        auto expect = golden[i];
        if (fabs(actual - expect) > eps && curErrorPrintNum < maxErrorPrintNum) {
            std::cout << i << ": output: " << actual << "; expect: " << expect << std::endl;
            curErrorPrintNum++;
        }
    }
    EXPECT_TRUE(resultCmp(golden, (float16 *)out->data(), eps));
}