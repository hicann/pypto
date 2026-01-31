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
 * \file test_dynamic_kv_slc.cpp
 * \brief
 */
#include <gtest/gtest.h>
#include "operator/models/deepseek/gen_kv_slc.h"
#include "interface/tensor/float.h"
#include "interface/configs/config_manager.h"

using namespace npu::tile_fwk;
class DynamicKvSlcUtTest : public testing::Test {
public:
    void SetUp() override {
        oriEnableAihacBackend = config::GetPlatformConfig(KEY_ENABLE_AIHAC_BACKEND, oriEnableAihacBackend);
        config::SetPlatformConfig(KEY_ENABLE_AIHAC_BACKEND, true);
        Program::GetInstance().Reset();
        config::Reset();
    }

    void TearDown() override { config::SetPlatformConfig(KEY_ENABLE_AIHAC_BACKEND, oriEnableAihacBackend);}
protected:
    bool oriEnableAihacBackend = false;
};

TEST_F(DynamicKvSlcUtTest, test_kv_slc) { // b_n_s_s2_h_q_lora_rank
    KvSlcTileShapeConfig tileConfig;
    const int nTile = 32;
    tileConfig.v0TileShape = {nTile, 32};

    std::vector<int> input_param = {1, 1, 1, 64, 64, 1, 2, 4, 16, 32};
    int b = input_param[0];
    int s = input_param[1];
    int n2 = input_param[2];
    int kv_lora_rank = input_param[3];
    int rope_dim = input_param[4];
    int front = input_param[5];
    int near = input_param[6];
    int topK = input_param[7];
    int l_prime = input_param[8];
    int blockSize = input_param[9];

    int blockNum = 0;
    std::vector<int> seq(1024, 1);
    for (auto s_item : seq) {
        blockNum += CeilDiv(s_item, blockSize);
    }
    int maxSeqAllBatch = *(std::max_element(seq.begin(), seq.end()));
    int maxBlockNumPerBatch = CeilDiv(maxSeqAllBatch, blockSize);

    Tensor topk_tensor(DT_INT32, {b, s, topK - front - near}, "topk_tensor");
    Tensor topk_tensor_shape(DT_INT32, {b, s}, "topk_tensor_shape");
    Tensor kvNopeCache(DT_FP16, {int(blockNum * blockSize), n2 * kv_lora_rank}, "kNopeCache");
    Tensor kRopeCache(DT_FP16, {int(blockNum * blockSize), n2 * rope_dim}, "vNopeCache");
    Tensor kvActSeqs(DT_INT32, {b}, "kvActSeqs");
    Tensor blockTable(DT_INT32, {b, maxBlockNumPerBatch}, "blockTable");
    Tensor k_slcOut(DT_FP16, {b * s * n2 * topK * l_prime, rope_dim + kv_lora_rank}, "k_slcOut");
    Tensor v_slcOut(DT_FP16, {b * s * n2 * topK * l_prime, kv_lora_rank}, "v_slcOut");
    Tensor kvSlcActSeqs(DT_INT32, {b, s}, "kvSlcActSeqs");
    GenKvSlc(topk_tensor, topk_tensor_shape, kvNopeCache, kRopeCache, kvActSeqs, front, near, topK, l_prime,
            n2, blockTable, blockSize, k_slcOut, v_slcOut, kvSlcActSeqs, tileConfig);
}