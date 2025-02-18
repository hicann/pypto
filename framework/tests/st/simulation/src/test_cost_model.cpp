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
 * \file test_cost_model.cpp
 * \brief
 */

#include "gtest/gtest.h"
#include <dlfcn.h>

#include "interface/configs/config_manager.h"
#include "operator/models/llama/llama_def.h"
#include "cost_model/simulation/common/CommonType.h"
#include "test_common.h"
#include "test_cost_model.h"
#include "test_dev_func_runner.h"

using namespace npu::tile_fwk;
namespace CostModel {
class CostModelTest : public testing::Test {
public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}

    void SetUp() override {
        oriEnableCostModel = config::GetPlatformConfig(KEY_ENABLE_COST_MODEL, oriEnableCostModel);
        config::SetPlatformConfig(KEY_ENABLE_COST_MODEL, true);

        oriEnableAihacBackend = config::GetPlatformConfig(KEY_ENABLE_AIHAC_BACKEND, oriEnableAihacBackend);
        config::SetPlatformConfig(KEY_ENABLE_AIHAC_BACKEND, true);

        oriEnableBinaryCache = config::GetPlatformConfig(KEY_ENABLE_BINARY_CACHE, oriEnableBinaryCache);
        config::SetPlatformConfig(KEY_ENABLE_BINARY_CACHE, false);
        Program::GetInstance().Reset();
        rtSetDevice(GetDeviceIdByEnvVar());
    }

    void TearDown() override {
        config::SetPlatformConfig(KEY_ENABLE_COST_MODEL, oriEnableCostModel);
        config::SetPlatformConfig(KEY_ENABLE_AIHAC_BACKEND, oriEnableAihacBackend);
        config::SetPlatformConfig(KEY_ENABLE_BINARY_CACHE, oriEnableBinaryCache);
    }

    void EnablePVModel(int level)
    {
        CostModel::PvData::Instance().Enable();
        CostModel::SoftMemory::Instance().Enable();
        oriPvLevel = config::GetSimConfig("PV_LEVEL", 0);
        config::SetSimConfig("PV_LEVEL", level);
    }

    void ResetPVModelConfig()
    {
        config::SetSimConfig("PV_LEVEL", oriPvLevel);
    }

protected:
    bool oriEnableAihacBackend = false;
    bool oriEnableCostModel = false;
    bool oriEnableBinaryCache = false;
    int oriPvLevel = 0;
};

template<typename InputT, typename OnputT>
void TestMatmulTrans(int m, int k, int n, string dataPath) {
    std::vector<int64_t> shape_a = {m, k};
    std::vector<int64_t> shape_b = {n, k};
    std::vector<int64_t> shape_c = {m, n};
    const int capacity_a = m * k;
    const int capacity_b = k * n;
    const int capacity_c = m * n;

    aclInit(nullptr);
    rtSetDevice(GetDeviceIdByEnvVar());
    uint64_t outputSize = capacity_c * sizeof(OnputT);
    uint8_t* c_ptr = allocDevAddr(outputSize);
    auto InputDtype = GetAstDtype<InputT>();
    auto OutputDtype = GetAstDtype<OnputT>();


    std::cout << "####:" << dataPath << "/a.bin" << std::endl;

    PROGRAM("Matmul") {
        void *a_ptr = readToDev<InputT>(dataPath + "/a.bin", capacity_a);
        void *b_ptr = readToDev<InputT>(dataPath + "/b.bin", capacity_b);

        Tensor mat_a(InputDtype, shape_a, (uint8_t *)a_ptr, "mat_a");
        Tensor mat_b(InputDtype, shape_b, (uint8_t *)b_ptr, "mat_b");
        Tensor mat_c(OutputDtype, shape_c, c_ptr, "mat_c");

        npu::tile_fwk::config::SetBuildStatic(true);
        FUNCTION("Matmul_T", {mat_a, mat_b, mat_c}) {
            mat_c = npu::tile_fwk::Matrix::Matmul<false, true>(OutputDtype, mat_a, mat_b);  // result dtype
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());
    std::vector<OnputT> dev_res(capacity_c);
    std::vector<OnputT> golden(capacity_c);
    machine::GetRA()->CopyFromTensor((uint8_t *)dev_res.data(), c_ptr, outputSize);
    readInput(dataPath + "/c_golden.bin", golden);
    int ret = resultCmp(golden, dev_res, 0.001f);
    EXPECT_EQ(ret, true);
}

TEST_F(CostModelTest, test_mm_float32_64_64_64_bt) {
    int level = static_cast<int>(CostModel::PVModelLevel::PV_EXECUTE);
    EnablePVModel(level);
    TileShape::Current().SetCubeTile({32, 32}, {32, 32}, {32, 32});
    TestMatmulTrans<npu::tile_fwk::float16, float>(64, 64, 64, GetGoldenDir());
    ResetPVModelConfig();
}


class CostModelDynTest : public testing::Test {
public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}

    void SetUp() override {
        cacheEnable = config::GetHostConfig(KEY_ENABLE_BINARY_CACHE, false);
        config::SetHostConfig(KEY_ENABLE_BINARY_CACHE, false);
        oriEnableAihacBackend = config::GetPlatformConfig(KEY_ENABLE_AIHAC_BACKEND, oriEnableAihacBackend);
        config::SetPlatformConfig(KEY_ENABLE_AIHAC_BACKEND, true);
        Program::GetInstance().Reset();
        constexpr int level = 2;
        EnablePVModel(level);
        rtSetDevice(GetDeviceIdByEnvVar());
    }

    void TearDown() override {
        config::SetHostConfig(KEY_ENABLE_BINARY_CACHE, cacheEnable);
        config::SetPlatformConfig(KEY_ENABLE_AIHAC_BACKEND, oriEnableAihacBackend);
        ResetPVModelConfig();
    }

    void EnablePVModel(int level)
    {
        oriPvLevel = config::GetSimConfig("PV_LEVEL", 0);
        config::SetSimConfig("PV_LEVEL", level);
    }

    void ResetPVModelConfig()
    {
        config::SetSimConfig("PV_LEVEL", oriPvLevel);
    }

protected:
    bool oriEnableAihacBackend = false;
    int oriPvLevel = 0;
    bool cacheEnable = false;
};

void CostModelTestLoopViewAssemble(const Tensor &t0, const Tensor &t1, const Tensor &blockTable, Tensor &out, int s) {
    FUNCTION("main", {t0, t1, blockTable}, {out}) {
        LOOP("L0", FunctionType::DYNAMIC_LOOP, i, LoopRange(GetInputShape(t0, 0) / s)) {
            SymbolicScalar idx = GetTensorData(blockTable, {i, 0});
            Tensor t0s = View(t0, {s, s}, {idx * s, 0});

            Tensor qi(DT_FP32, {s, 2*s}, "qi");
            Assemble(t1, {0, 0}, qi);
            Assemble(t0s, {0, s}, qi);

            Tensor ki(DT_FP32, {s, 2*s}, "ki");
            Assemble(t0s, {0, 0}, ki);
            Assemble(t1, {0, s}, ki);

            Tensor t2 = Matrix::Matmul<false, true>(DataType::DT_FP32, qi, ki);
            // conat((t0s + t1, t1)) @ concat (t0s, t1)^T
            Assemble(t2, {idx * s, 0}, out);
        }
    }
}

TEST_F(CostModelDynTest, TestDD) {
    config::SetHostOption(ONLY_CODEGEN, true);
    constexpr int tilingX = 32;
    constexpr int tilingY = 32;
    TileShape::Current().SetVecTile(tilingX, tilingY);
    constexpr int tilingM = 32;
    constexpr int tilingN = 32;
    constexpr int tilingK = 32;
    TileShape::Current().SetCubeTile({tilingM, tilingM}, {tilingN, tilingN}, {tilingK, tilingK});
    std::vector<uint8_t> devProgBinary;

    int s = 32;
    int n = 8;
    Tensor t0(DT_FP32, {n * s, s}, "t0");  // [32*8, 32]
    Tensor t1(DT_FP32, {s, s}, "t1");  // [32, 32]
    Tensor blockTable{
        DT_INT32, {n, 1},
         "blockTable"
    };
    Tensor out(DT_FP32, {n * s, s}, "out");
    CostModelTestLoopViewAssemble(t0, t1, blockTable, out, s);

    std::vector<int> tblData;
    for (int i = 0; i < n; i++)
        tblData.push_back(i);

    ProgramData::GetInstance().AppendInputs({
        RawTensorData::CreateConstantTensor<float>(t0, 1.0),
        RawTensorData::CreateConstantTensor<float>(t1, 2.0),
        RawTensorData::CreateTensor<int>(blockTable, tblData),  // value: [0,1,2,...,7]
    });
    ProgramData::GetInstance().AppendOutputs({
        RawTensorData::CreateConstantTensor<float>(out, 0.0f),
    });

    auto func = Program::GetInstance().GetLastFunction();
#ifdef BUILD_WITH_CANN
    CostModelDynFuncRunner::Run(func);
    std::vector<float> golden(n * s * s, 0.0f);
    auto outs = npu::tile_fwk::ProgramData::GetInstance().GetOutputData(0);
    EXPECT_TRUE(resultCmp(golden, (float *)outs->data(), 0.001f));
#endif
}

}
