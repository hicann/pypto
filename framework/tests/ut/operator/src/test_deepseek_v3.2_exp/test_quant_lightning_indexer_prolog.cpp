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
\file test_quant_lightning_indexer_prolog.cpp
\brief
*/
#include <gtest/gtest.h>
#include "tilefwk/tilefwk_op.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "operator/models/deepseek_v3.2_exp/quant_lightning_indexer_prolog.h"

using namespace npu::tile_fwk;

class DynamicQuantLightningIndexerPrologUtest : public testing::Test {
public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}

    void SetUp() override {
        Program::GetInstance().Reset();
        config::SetCodeGenConfig(KEY_CODEGEN_SUPPORT_TILE_TENSOR, false);
    }

    void TearDown() override {
        config::SetCodeGenConfig(KEY_CODEGEN_SUPPORT_TILE_TENSOR, true);
    }
};
namespace {

TEST_F(DynamicQuantLightningIndexerPrologUtest, b4_s1_2_s2_64k) {
    QuantIndexerConfigs configs;
    configs.qLinear = {16, 16, 128, 128, 128, 128};
    configs.qHd = {16, 16, 128, 128, 128, 128};
    configs.kLinear = {16, 16, 128, 128, 128, 128};
    configs.wLinear = {16, 16, 128, 128, 128, 128};

    config::SetHostOption(ONLY_CODEGEN, true);

    int64_t s2 = 64 * 1024;
    int64_t b = 4;
    int64_t s1 = 2;
    int64_t t = b * s1;
    int64_t h = 7168;
    int64_t qLoraRank = 1536;
    int64_t headDim = 128;
    int64_t headNum = 64;
    int64_t ropeHeadDim = 64;
    int64_t blockSize = 128;
    int64_t blockNum = s2 * b / blockSize;
    int64_t nKV = 1;
    constexpr int64_t nzFirstDim = 16;
    constexpr int64_t b16C0Dim = 16;
    constexpr int64_t b8C0Dim = 32;

    DataType dType = DT_BF16;
    TileOpFormat weightFormat = TileOpFormat::TILEOP_NZ;
    Tensor x(dType, {t, h}, "x");
    Tensor qNorm(DT_INT8, {t, qLoraRank}, "qNorm");
    Tensor qNormScale(DT_FP32, {t, 1}, "qNormScale");
    Tensor wQb(
        DT_INT8, {headNum * headDim / b8C0Dim, qLoraRank / nzFirstDim, nzFirstDim, b8C0Dim}, "wQb", weightFormat);
    Tensor wQbScale(DT_FP32, {headNum * headDim, 1}, "wQbScale");
    Tensor wk(dType, {headDim / b16C0Dim, h / nzFirstDim, nzFirstDim, b16C0Dim}, "wk", weightFormat);
    Tensor wProj(dType, {headNum / b16C0Dim, h / nzFirstDim, nzFirstDim, b16C0Dim}, "wProj", weightFormat);
    Tensor lnGammaK(dType, {headDim}, "lnGammaK");
    Tensor lnBetaK(dType, {headDim}, "lnBetaK");
    Tensor cosIdxRope(dType, {t, ropeHeadDim}, "cosIdxRope");
    Tensor sinIdxRope(dType, {t, ropeHeadDim}, "sinIdxRope");
    Tensor hadamardQ(dType, {headDim, headDim}, "hadamardQ");
    Tensor hadamardK(dType, {headDim, headDim}, "hadamardK");
    Tensor kCache(DT_INT8, {blockNum, blockSize, nKV, headDim}, "kCache");
    Tensor kCacheScale(DT_FP16, {blockNum, blockSize, nKV, 1}, "kCacheScale");
    Tensor kCacheIndex(DT_INT64, {t}, "kCacheIndex");

    QuantIndexerPrologInput staticInput{x, qNorm, qNormScale, wQb, wQbScale, wk, wProj, lnGammaK, lnBetaK, cosIdxRope,
        sinIdxRope, hadamardQ, hadamardK, kCache, kCacheScale, kCacheIndex};

    // outputs
    Tensor qInt8(DT_INT8, {t, headNum, headDim}, "qInt8");
    Tensor qScale(DT_FP16, {t, headNum, 1}, "qScale");
    Tensor kInt8(DT_INT8, {blockNum, blockSize, nKV, headDim}, "kInt8");
    Tensor kScale(DT_FP16, {blockNum, blockSize, nKV, 1}, "kScale");
    Tensor weights(DT_FP16, {t, headNum}, "weights");

    QuantIndexerPrologOutput staticOutput{qInt8, qScale, kInt8, kScale, weights};

    QuantIndexerPrologAttr attrs;
    attrs.eps = 1e-6f;
    attrs.layeroutKey = "PA_BSND";
    attrs.layeroutQuery = "TND";

    QuantLightningIndexerProlog(staticInput, staticOutput, attrs, configs);
}
} // namespace