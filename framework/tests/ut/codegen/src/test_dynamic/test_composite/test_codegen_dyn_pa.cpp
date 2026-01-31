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
 * \file test_codegen_dyn_pa.cpp
 * \brief Unit test for codegen.
 */

#include "gtest/gtest.h"

#include "codegen/codegen.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/machine/host/host_machine.h"
#include "operator/models/llama/llama_def.h"
#include "operator/models/deepseek/page_attention.h"
#include "interface/configs/config_manager.h"
#include "codegen/cloudnpu/codegen_cloudnpu.h"

namespace npu::tile_fwk {

class TestCodegenDynPa : public ::testing::Test {
public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}

    void SetUp() override {
        Program::GetInstance().Reset();
        config::Reset();
        config::SetHostOption(COMPILE_STAGE, CS_EXECUTE_GRAPH);
        config::SetPlatformConfig(KEY_ENABLE_COST_MODEL, false);
    }

    void TearDown() override {}
};

void testPa(PaTileShapeConfig &tileConfig, int maxUnrollTimes = 1) {

    int b = 4;
    int sq = 1;
    int nq = 32;
    int nk = 1;
    int dn = 512;
    int dr = 64;
    int blockSize = 128;
    float softmaxScale = static_cast<float>(1.0 / sqrtf((dn + dr)));

    std::vector<int> seq(b, 256);

    int blockNum = 0;
    for (auto s : seq) {
        blockNum += CeilDiv(s, blockSize);
    }
    // blockTable: (b, maxBlockNumPerBatch)
    int maxSeqAllBatch = *(std::max_element(seq.begin(), seq.end()));
    int maxBlockNumPerBatch = CeilDiv(maxSeqAllBatch, blockSize);

    Tensor qNope(DT_BF16, {b * nq * sq, dn}, "qNope");
    Tensor kNopeCache(DT_BF16, {int(blockNum * blockSize), nk * dn}, "kNopeCache");
    Tensor vNopeCache(DT_BF16, {int(blockNum * blockSize), nk * dn}, "vNopeCache");
    Tensor qRope(DT_BF16, {b * nq * sq, nk * dr}, "qRope");
    Tensor kRopeCache(DT_BF16, {int(blockNum * blockSize), nk * dr}, "kRope");
    Tensor blockTable(DT_INT32, {b, maxBlockNumPerBatch}, "blockTable");
    Tensor actSeqs(DT_INT32, {b}, "actSeqs");
    Tensor paOut(DT_FP32, {b * nq * sq, dn}, "paOut");

    PageAttention(qNope, kNopeCache, vNopeCache, qRope, kRopeCache, blockTable, actSeqs, blockSize, softmaxScale, paOut,
        tileConfig, maxUnrollTimes);

    for (auto &ele : Program::GetInstance().GetFunctionMap()) {
        bool isRootExist = ele.second.get()->rootFunc_ != nullptr;
        if (isRootExist) {
            npu::tile_fwk::CodeGenCtx ctx;
            npu::tile_fwk::CodeGenCloudNPU codeGen(ctx);
            codeGen.GenCode(*ele.second.get(), {});
        }
    }
}

TEST_F(TestCodegenDynPa, PaHighThroughputDviewLargeDynamicValidShape) {
    PaTileShapeConfig tileConfig;
    const int nTile = 128;
    tileConfig.headNumQTile = nTile;
    tileConfig.v0TileShape = {nTile, 64};
    tileConfig.c1TileShape = {nTile, nTile, 64, 64, 256, 512};
    tileConfig.v1TileShape = {nTile / 8, 256};
    tileConfig.c2TileShape = {nTile, nTile, 64, 512, 256, 256};
    tileConfig.v2TileShape = {nTile / 8, 256};
    testPa(tileConfig);
}
} // namespace npu::tile_fwk