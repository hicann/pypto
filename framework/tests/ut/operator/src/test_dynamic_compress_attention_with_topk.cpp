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
 * \file test_dynamic_compress_attention_with_topk.cpp
 * \brief
 */

#include <gtest/gtest.h>
#include "interface/tensor/float.h"
#include "tilefwk/data_type.h"
#include "interface/function/function.h"
#include "tilefwk/tilefwk_op.h"
#include "tilefwk/tilefwk.h"
#include "interface/configs/config_manager.h"
#include "interface/interpreter/raw_tensor_data.h"
#include "operator/models/nsa/compress_attention_with_topk.h"

using namespace npu::tile_fwk;

class CmpAttnTopk : public testing::Test {
    void SetUp() override { config::SetHostOption(ONLY_CODEGEN, true); }
};

template <typename T = npu::tile_fwk::bfloat16>
void TestCmpAttnTopk(CmpAttnTopkTile &tileConfig, std::vector<int> input_param, std::vector<int> actSeqLen) {
    config::SetHostOption(ONLY_CODEGEN, true);

    DataType dType = DT_FP32;
    if (std::is_same<T, npu::tile_fwk::bfloat16>::value) {
        dType = DT_BF16;
    } else if (std::is_same<T, npu::tile_fwk::float16>::value) {
        dType = DT_FP16;
    } else {
        dType = DT_FP32;
    }

    const int b = input_param[0];
    const int s1 = input_param[1];
    const int n1 = input_param[2];
    const int dn = input_param[3];
    const int dr = input_param[4];
    const int n2 = input_param[5];
    const int blockSize = input_param[6];
    const int cmpBlockSize = input_param[7];
    const int cmpStride = input_param[8];
    const int slcBlockSize = input_param[9];
    const int topk = input_param[10];
    const int front = input_param[11];
    const int near = input_param[12];
    const float softmaxScale = static_cast<float>(1.0 / sqrtf((dn + dr)));

    DataType qType = dType;
    DataType kType = dType;

    std::vector<int> actCmpSeq;
    for (auto curSeq : actSeqLen) {
        auto curCmpSeq = (curSeq - cmpBlockSize) / cmpStride + 1;
        actCmpSeq.emplace_back(curCmpSeq);
    }

    int cmpBlockNum = 0;
    for (auto s : actCmpSeq) {
        cmpBlockNum += CeilDiv(s, blockSize);
    }
    int maxCmpSeq = *(std::max_element(actCmpSeq.begin(), actCmpSeq.end()));
    int maxCmpBlockNum = CeilDiv(maxCmpSeq, blockSize);

    const int slcSize = slcBlockSize / cmpStride;
    const int blockSlcNum = blockSize / slcSize;
    (void)blockSlcNum;

    // Construct input tensors
    Tensor qNope(qType, {b * s1 * n1, dn}, "qNope");
    Tensor qRope(qType, {b * s1 * n1, dr}, "qRope");
    Tensor cmpKvCache(kType, {cmpBlockNum, blockSize, n2, dn}, "cmpKvCache");
    Tensor cmpKrCache(kType, {cmpBlockNum, blockSize, n2, dr}, "cmpKrCache");
    Tensor cmpBlockTable(DT_INT32, {b, maxCmpBlockNum}, "cmpBlockTable");
    Tensor actSeq(DT_INT32, {b}, "actSeq");
    Tensor auxTensor(DT_FP32, {slcBlockSize / cmpStride + cmpBlockSize / cmpStride - 1, n1}, "auxTensor"); // (5, n1)

    // Construct out tensors
    Tensor cmpAttn(DT_FP32, {b, s1, n1, dn}, "cmpAttnOut");
    Tensor topkRes(DT_INT32, {b, s1, topk}, "topkRes");

    // Read goldens
    std::vector<float> attnGolden(b * s1 * n1 * dn, 0.0);
    std::vector<int32_t> topkGolden(b * s1 * topk, 0);

    FUNCTION("CompressAttentionWithTopK",
        {qNope, qRope, cmpKvCache, cmpKrCache, cmpBlockTable, actSeq, auxTensor}, {cmpAttn, topkRes}) {
        CompressAttentionWithTopK(qNope, qRope, cmpKvCache, cmpKrCache, cmpBlockTable, actSeq, auxTensor, cmpAttn,
            topkRes, blockSize, cmpBlockSize, cmpStride, slcBlockSize, softmaxScale, n1, topk, front, near, tileConfig);
    }
}

void CommonTestConfig() {

    CmpAttnTopkTile config;
    config.topkTile = {1, 1, 128};
    config.cmpTile.c1Tile = {128, 128, 128, 128, 128, 128};
    config.cmpTile.v1Tile = {128, 128};
    config.cmpTile.c2Tile = {128, 128, 128, 128, 128, 128};
    config.cmpTile.v2Tile = {128, 64};

    std::vector<int> inputParam = {2, 1, 128, 512, 64, 1, 128, 32, 16, 64, 16, 1, 2};
    std::vector<int> actSeqLen = {8192, 8192};

    TestCmpAttnTopk<npu::tile_fwk::bfloat16>(config, inputParam, actSeqLen);
}

TEST_F(CmpAttnTopk, cmp_attn_with_topk_singleop_bf16) {
    CommonTestConfig();
}