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
 * \file test_dynamic_dsa_selected_attention.cpp
 * \brief
 */
 
#include <gtest/gtest.h>
#include <cstdint>
#include "tilefwk/data_type.h"
#include "test_data_prepare.h"
#include "test_suite_stest_ops.h"
#include "interface/interpreter/raw_tensor_data.h"
#include "operator/models/deepseek_v3.2_exp/sparse_flash_attention.h"
#include "test_dev_func_runner.h"
 
using namespace npu::tile_fwk;
using namespace npu::tile_fwk::dynamic;
class DynamicSparseFlashAttnDSASTest : public npu::tile_fwk::stest::TestSuite_STest_Ops_Aihac {};
 
template <typename T = npu::tile_fwk::float16>
void TestSa(SaTileShapeConfig& tileConfig) {
    config::SetHostOption(ONLY_CODEGEN, true);
 
    DataType dType = DT_FP32;
    if (std::is_same<T, npu::tile_fwk::float16>::value) {
        dType = DT_FP16;
    } else if (std::is_same<T, npu::tile_fwk::bfloat16>::value) {
        dType = DT_BF16;
    } else {
        dType = DT_FP32;
    }
 
    int paramsSize = 8;
    std::vector<int> input_param(paramsSize);
    readInput<int>(GetGoldenDir() + "/input_param.bin", input_param);
    int b = input_param[0];
    int sq = input_param[1];
    int nq = input_param[2];
    int nkv = input_param[3];
    int dn = input_param[4];
    int dr = input_param[5];
    int smax = input_param[6];
    int topk = smax;
 
    float softmaxScale = static_cast<float>(1.0 / sqrtf((dn + dr)));
 
    std::cout << "====input param==== b sq nq nkv dn dr smax: " << b << " " << sq << " " << nq << " " << nkv << " " << dn << " " << dr << " " << smax << std::endl;
 
    std::vector<int64_t> qNopeShape = {b * sq * nq, dn};
    std::vector<int64_t> qRopeShape = {b * sq * nq, dr};
    std::vector<int64_t> kSlcShape = {b * sq * smax, dn + dr};
    std::vector<int64_t> vSlcShape = {b * sq * smax, dn};
    std::vector<int64_t> actSeqsShape = {b};
    std::vector<int64_t> saOutShape = {b, sq, nq, dn};
 
    auto qNope = CreateTensorAndData<T>(qNopeShape, dType, "qNope", "/q_nope.bin", {0});
    auto qRope = CreateTensorAndData<T>(qRopeShape, dType, "qRope", "/q_rope.bin", {0});
    auto kSlc = CreateTensorAndData<T>(kSlcShape, dType, "kSlc", "/k_slc.bin", {0});
    auto vSlc = CreateTensorAndData<T>(vSlcShape, dType, "vSlc", "/v_slc.bin", {0});
    auto actSeqs = CreateTensorAndData<int32_t>(actSeqsShape, DT_INT32, "actSeqs", "/actual_seq.bin", {0});
 
    SymbolicScalar batchSizeSym = GetInputShape(actSeqs.tensor, 0); // b
    SymbolicScalar s1N2GSym = GetInputShape(qNope.tensor, 0) / batchSizeSym; // s1n2
    SymbolicScalar s1Sym = s1N2GSym / nq; // s1
 
    Tensor saOut(dType, {batchSizeSym, s1Sym, nq, dn}, "saOut");
    RawTensorDataPtr saOutData = RawTensorData::CreateConstantTensorData<T>(saOutShape, dType, 0);
 
    SparseFlashAttention(qNope.tensor, qRope.tensor, kSlc.tensor, vSlc.tensor, actSeqs.tensor, nq, nkv, softmaxScale, topk, saOut, tileConfig);
 
    // 读数据
    int saOutSize = std::accumulate(saOutShape.begin(), saOutShape.end(), 1, std::multiplies<>());
    std::vector<T> golden(saOutSize, 0);
    readInput(GetGoldenDir() + "/atten_out.bin", golden);
 
    ProgramData::GetInstance().AppendInputs({
        qNope.dataPtr,
        qRope.dataPtr,
        kSlc.dataPtr,
        vSlc.dataPtr,
        actSeqs.dataPtr
    });
    ProgramData::GetInstance().AppendOutputs({
        saOutData
    });
 
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());
    auto outs = npu::tile_fwk::ProgramData::GetInstance().GetOutputData(0);
    EXPECT_TRUE(resultCmp(golden, (T *)outs->data(), 0.0005f));
}
 
TEST_F(DynamicSparseFlashAttnDSASTest, dsa_slc_attn_bf16_b48_s1) {
    SaTileShapeConfig tileConfig;
    const int gTile = 128; // for gLoop split
    const int sTile = 2048; // for s2Loop split
    tileConfig.gTile = gTile;
    tileConfig.sKvTile = sTile;
    tileConfig.c1TileShape = {gTile, gTile, 64, 64, 256, 256}; // (n1, dn+dr) @ (s2Tile, dn+dr) -> (n1, s2Tile)
    tileConfig.v1TileShape = {16, 256}; // (n1, s2Tile)
    tileConfig.c2TileShape = {gTile, gTile, 128, 128, 128, 128}; // (n1, s2Tile) @ (s2Tile, dn) -> (n1, d)
    tileConfig.v2TileShape = {64, 128}; // (n1, d)
 
    TestSa<npu::tile_fwk::bfloat16>(tileConfig);
}
 
TEST_F(DynamicSparseFlashAttnDSASTest, dsa_slc_attn_bf16_b32_s2) {
    SaTileShapeConfig tileConfig;
    const int gTile = 128; // for gLoop split
    const int sTile = 1024; // for s2Loop split
    tileConfig.gTile = gTile;
    tileConfig.sKvTile = sTile;
    tileConfig.c1TileShape = {gTile, gTile, 64, 64, 256, 256}; // (n1, dn+dr) @ (s2Tile, dn+dr) -> (n1, s2Tile)
    tileConfig.v1TileShape = {16, 256}; // (n1, s2Tile)
    tileConfig.c2TileShape = {gTile, gTile, 128, 128, 128, 128}; // (n1, s2Tile) @ (s2Tile, dn) -> (n1, d)
    tileConfig.v2TileShape = {64, 128}; // (n1, d)
 
    TestSa<npu::tile_fwk::bfloat16>(tileConfig);
}