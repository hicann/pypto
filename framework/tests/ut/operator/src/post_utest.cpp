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
 * \file post_utest.cpp
 * \brief
 */

#include "gtest/gtest.h"
#include "tilefwk/tilefwk_op.h"

#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/tensor/logical_tensor.h"
#include "interface/tensor/raw_tensor.h"
#include "interface/interpreter/raw_tensor_data.h"
#include "operator/models/nsa/attention_post.h"
#include "interface/configs/config_manager.h"
#include "interface/tensor/float.h"

using namespace npu::tile_fwk;

class AttentionPostUTest : public testing::Test {
public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}

    void SetUp() override {
        Program::GetInstance().Reset();
    }

    void TearDown() override {}
};

struct TestPostParams {
    int b;
    int n;
    int s;
    int h;
    int kvLoraRank;
    int vHeadDim;
};

template <typename T = npu::tile_fwk::float16, bool nz = true, typename wUvDtype = int8_t, bool isSmoothWuv = false,
    typename wODtype = int8_t, bool isSmoothWo = false>
void TestAttentionPostUt(const TestPostParams &params, const PostTileConfig &tileConfig) {
    int b = params.b;
    int n = params.n;
    int s = params.s;
    int h = params.h;
    int kvLoraRank = params.kvLoraRank;
    int vHeadDim = params.vHeadDim;

    DataType dType = (std::is_same<T, npu::tile_fwk::float16>::value) ? DT_FP16 : DT_BF16;
    bool isQuantWUv = std::is_same<wUvDtype, int8_t>::value;
    bool isQuantWo = std::is_same<wODtype, int8_t>::value;

    std::vector<int64_t> xShape = {b, s, n, kvLoraRank};
    std::vector<int64_t> wUvShape = {n, kvLoraRank, vHeadDim};
    std::vector<int64_t> wUvScaleShape = {n, 1, vHeadDim};
    std::vector<int64_t> smoothWUvShape = {1, kvLoraRank};
    std::vector<int64_t> woShape = {n * vHeadDim, h};
    std::vector<int64_t> woScaleShape = {1, h};
    std::vector<int64_t> smoothWoShape = {1, n * vHeadDim};
    std::vector<int64_t> outShape = {b, s, h};

    TileOpFormat weightFormat = nz ? TileOpFormat::TILEOP_NZ : TileOpFormat::TILEOP_ND;
    Tensor x(dType, xShape, "x");
    Tensor wUv(isQuantWUv ? DT_INT8 : dType, wUvShape, "wUv");
    Tensor wUvScale;
    Tensor smoothWUv;
    Tensor wo(isQuantWo ? DT_INT8 : dType, woShape, "wo", weightFormat);
    Tensor woScale;
    Tensor smoothWo;
    Tensor postOut(dType, outShape, "postOut");

    if(isQuantWUv) {
        Tensor scale(DT_FP32, wUvScaleShape, "wUvScale");
        wUvScale =scale;
        if(isSmoothWuv) {
            Tensor smooth(DT_FP32, smoothWUvShape, "smoothWUv");
            smoothWUv =smooth;
        }
    }
    if (isQuantWo) {
        Tensor scale(DT_FP32, woScaleShape, "woScale");
        woScale = scale;
        if (isSmoothWo) {
            Tensor smooth(DT_FP32, smoothWoShape, "smoothWo");
            smoothWo = smooth;
        }
    }

    PostTensors postTensors{wUv, wo, wUvScale, smoothWUv, woScale, smoothWo};
    AttentionPostStandalone(x, postTensors, tileConfig, postOut);
}

TEST_F(AttentionPostUTest, b32_s1_nz_fp16_quant) {
    // b, n, s, h, kvLoraRank, vHeadDim
    TestPostParams params = {32, 128, 1, 7168, 512, 128};
    PostTileConfig tileConfig = {16, 1};

    TestAttentionPostUt<npu::tile_fwk::float16, true, npu::tile_fwk::float16, false, int8_t, true>(params, tileConfig);
}