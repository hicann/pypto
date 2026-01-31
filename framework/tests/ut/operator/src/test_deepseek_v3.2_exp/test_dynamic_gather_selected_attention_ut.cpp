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
 * \file test_dynamic_gather_selected_attention_ut.cpp
 * \brief
 */

#include <cstdint>
#include "tilefwk/tilefwk_op.h"
#include "tilefwk/tilefwk.h"
#include "test_cost_macro.h"
#include "interface/inner/tilefwk.h"
#include "interface/tensor/logical_tensor.h"
#include "interface/tensor/raw_tensor.h"
#include "interface/interpreter/raw_tensor_data.h"
#include "operator/models/deepseek_v3.2_exp/gather_selected_attention.h"

using namespace npu::tile_fwk;

class DynamicGatherSlcFlashAttnUtest : public testing::Test {
public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}

    void SetUp() override { Program::GetInstance().Reset(); }

    void TearDown() override {}
};

SaTileShapeConfig GetDefaultSaTileShapeConfig(const int gTile, const int sTile) {
    SaTileShapeConfig tileConfig;
    tileConfig.gTile = gTile; // for gLoop split
    tileConfig.sKvTile = sTile; // for s2Loop split
    tileConfig.c1TileShape = {gTile, gTile, 64, 64, 256, 256}; // (n1, dn+dr) @ (s2Tile, dn+dr) -> (n1, s2Tile)
    tileConfig.v1TileShape = {16, 256}; // (n1, s2Tile)
    tileConfig.c2TileShape = {gTile, gTile, 128, 128, 128, 128}; // (n1, s2Tile) @ (s2Tile, dn) -> (n1, d)
    tileConfig.v2TileShape = {64, 128}; // (n1, d)
    return tileConfig;
}

void TestSaUT(const std::vector<int64_t> &input_param, SaTileShapeConfig& tileConfig) {
    int b = input_param.at(0);
    int sq = input_param.at(1);
    int nq = input_param.at(2);
    int nKv = input_param.at(3);
    int maxKVSeq = input_param.at(4);
    int dn = input_param.at(5);
    int dr = input_param.at(6);
    int blockNum = input_param.at(7);
    int blockSize = input_param.at(8);
    int topk = input_param.at(9);
    int isKnQuant = input_param.at(10);
    int nQ = nq;
    DataType dType = DT_BF16;
    DataType knDType = DT_BF16;
    if(isKnQuant) knDType = DT_INT8;
    int maxBlockNumPerBatch = CeilDiv(maxKVSeq, blockSize);

    float softmaxScale = static_cast<float>(1.0 / sqrtf((dn + dr)));
    std::vector<int64_t> qNopeShape = {b * sq * nq, dn};
    std::vector<int64_t> qRopeShape = {b * sq * nq, dr};
    std::vector<int64_t> knShape = {blockNum * blockSize, dn};
    std::vector<int64_t> krShape = {blockNum * blockSize, dr};
    std::vector<int64_t> knScalesShape = {blockNum * blockSize, 4};
    std::vector<int64_t> topKIndciesShape = {b * sq, nKv * topk};
    std::vector<int64_t> blockTableShape = {b, maxBlockNumPerBatch};
    std::vector<int64_t> actSeqsShape = {b};
    std::vector<int64_t> saOutShape = {b, sq, nq, dn};

    Tensor qNope(dType, qNopeShape, "qNope");
    Tensor qRope(dType, qRopeShape, "qRope");
    Tensor kNope2D(knDType, knShape, "kNope2D");
    Tensor kRope2D(dType, krShape, "kRope2D");
    Tensor kNopeScales(DT_FP32, knScalesShape, "kNopeScales");
    Tensor topKIndcies(DT_INT32, topKIndciesShape, "topKIndcies");
    Tensor blockTable(DT_INT32, blockTableShape, "blockTable");
    Tensor kvSlcActSeqs(DT_INT32, actSeqsShape, "kvSlcActSeqs");
    Tensor attentionOut(dType, saOutShape, "attentionOut");
    SelectedAttentionV2(
        qNope, qRope, kNope2D, kRope2D, kNopeScales, topKIndcies, blockTable, kvSlcActSeqs, nQ, nKv, softmaxScale,
        topk, blockSize, maxBlockNumPerBatch, attentionOut, tileConfig
    );
}

TEST_F(DynamicGatherSlcFlashAttnUtest, dsa_gather_slc_attn_bf16_b32_s4) {
    SaTileShapeConfig tileConfig = GetDefaultSaTileShapeConfig(128, 2048);
    std::vector<int64_t> input_param = {32, 4, 128, 1, 8192, 512, 64, 32, 128, 2048, 0};
    TestSaUT(input_param, tileConfig);
}

TEST_F_WITH_COST(DynamicGatherSlcFlashAttnUtest, dsa_gather_slc_attn_bf16_b32_s4_int8, 60) {
    SaTileShapeConfig tileConfig = GetDefaultSaTileShapeConfig(128, 2048);
    std::vector<int64_t> input_param = {32, 4, 128, 1, 8192, 512, 64, 32, 128, 2048, 1};
    TestSaUT(input_param, tileConfig);
}