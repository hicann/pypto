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
 * \file test_nsa_slc_attention.cpp
 * \brief
 */

#include "gtest/gtest.h"
#include "tilefwk/tilefwk_op.h"

#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"

#include "interface/tensor/logical_tensor.h"
#include "interface/tensor/raw_tensor.h"
#include "interface/interpreter/raw_tensor_data.h"
#include "operator/models/nsa/slc_attn.h"
#include "interface/configs/config_manager.h"
#include "interface/tensor/float.h"

using namespace npu::tile_fwk;

class SlcAttnUtest : public testing::Test {
public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}

    void SetUp() override {
        Program::GetInstance().Reset();
        config::Reset();
        config::SetPlatformConfig(KEY_ONLY_HOST_COMPILE, true);
        config::SetPlatformConfig("ENABLE_COST_MODEL", false);
    }

    void TearDown() override {}
};

struct SaConfig {
    bool manualUnroll{false};
    int maxUnrollTimes{1};
    bool onlyBatchLoop{false};
    bool isNzFormat{false};
};

template <typename T = npu::tile_fwk::float16>
void TestSaUT(std::vector<int> &input_param, SaTileShapeConfig& tileConfig, SaConfig config) {

    DataType dType = DT_FP32;
    if (std::is_same<T, npu::tile_fwk::float16>::value) {
        dType = DT_FP16;
    } else if (std::is_same<T, npu::tile_fwk::bfloat16>::value) {
        dType = DT_BF16;
    } else {
        dType = DT_FP32;
    }

    int b = input_param[0];
    int sq = input_param[1];
    int nq = input_param[2];
    int nkv = input_param[3];
    int dn = input_param[4];
    int dr = input_param[5];
    int smax = input_param[6];
    float softmaxScale = static_cast<float>(1.0 / sqrtf((dn + dr)));

    TileOpFormat kvFormat = config.isNzFormat ? TileOpFormat::TILEOP_NZ : TileOpFormat::TILEOP_ND;

    std::vector<int64_t> qNopeShape = {b * sq * nq, dn};
    std::vector<int64_t> qRopeShape = {b * sq * nq, dr};
    std::vector<int64_t> kSlcShape = {b * sq * nkv * smax, dn + dr};
    std::vector<int64_t> vSlcShape = {b * sq * nkv * smax, dn};
    std::vector<int64_t> saOutShape = {b, sq, nq, dn};

    Tensor actSeqs(DT_INT32, {b, sq}, "actSeqs");
    Tensor qNope(dType, qNopeShape, "qNope");
    Tensor qRope(dType, qRopeShape, "qRope");
    Tensor kSlc(dType, kSlcShape, "kSlc", kvFormat);
    Tensor vSlc(dType, vSlcShape, "vSlc", kvFormat);
    Tensor saOut(DT_FP32, saOutShape, "saOut");

    SlcAttn(qNope, qRope, kSlc, vSlc, actSeqs, nq, nkv, softmaxScale, saOut, tileConfig);
}

TEST_F(SlcAttnUtest, slc_attn_fp16) {
    SaTileShapeConfig tileConfig;
    const int gTile = 128; // for gLoop split
    const int sTile = 1024; // for s2Loop split
    tileConfig.gTile = gTile;
    tileConfig.sKvTile = sTile;
    tileConfig.c1TileShape = {gTile, gTile, 64, 64, 128, 128}; // (n1, dn+dr) @ (s2Tile, dn+dr) -> (n1, s2Tile)
    tileConfig.v1TileShape = {16, 256}; // (n1, s2Tile)
    tileConfig.c2TileShape = {gTile, gTile, 64, 64, 128, 128}; // (n1, s2Tile) @ (s2Tile, dn) -> (n1, d)
    tileConfig.v2TileShape = {16, 256}; // (n1, d)

    std::vector<int> input_param = {32, 1, 128, 1, 512, 64, 1024};
    SaConfig config;
    TestSaUT<npu::tile_fwk::float16>(input_param, tileConfig, config);
}
