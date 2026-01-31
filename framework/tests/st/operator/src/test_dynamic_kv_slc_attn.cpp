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
 * \file test_dynamic_kv_slc_attn.cpp
 * \brief
 */

#include "test_dev_func_runner.h"
#include "test_common.h"
#include "test_suite_stest_ops.h"
#include "operator/models/nsa/nsa_selected_attention.h"
#include "operator/models/nsa/dynamic_nsa_v1.h"

using namespace npu::tile_fwk;
using namespace npu::tile_fwk::dynamic;
class DynamicKvSATest : public npu::tile_fwk::stest::TestSuite_STest_Ops_Aihac {};

void SetKvSAPreConfig() {
}

template <typename T>
static std::vector<T> getGoldenVec(std::vector<int64_t> shape, std::string fileName) {
    int capacity = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<>());
    std::vector<T> golden(capacity, 0);
    readInput<T>(GetGoldenDir() + fileName, golden);
    return golden;
}

template <typename T>
static std::shared_ptr<RawTensorData> CreateTensorData(Tensor tensor, std::string fileName) {
    auto shape = tensor.GetShape();
    int capacity = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<>());
    std::vector<T> values(capacity, 0);
    readInput<T>(GetGoldenDir() + fileName, values);
    return RawTensorData::CreateTensor<T>(tensor, values);
}

template <typename T = npu::tile_fwk::float16>
void TestKvSlcAttn(const NSAV1SimpleParams &params, SATileShapeConfig& saTileConfig) {
    SetInterpreterConfig();
    SetKvSAPreConfig();

    int b = params.b;
    int s1 = params.s1;
    int n1 = params.n1;
    int n2 = params.n2;
    int v_dim = params.kv_lora_rank;
    int dn = v_dim;
    int dr = params.rope_dim;
    float softmaxScale = static_cast<float>(1.0 / sqrtf((dn + dr)));
    int blockSize = params.blockSize;
    int cmpBlockSize = params.cmpBlockSize;
    int slcBlockSize = params.slcBlockSize;
    int front = params.front;
    int near = params.near;
    int topk = params.topk;
    int smax = params.topk * params.slcBlockSize;

    std::vector<int> kvCacheActSeqVec(b);
    readInput<int>(GetGoldenDir() + "/kv_cache_actual_seq_len.bin", kvCacheActSeqVec);
    int blockNum = 0;
    for (auto seqItem : kvCacheActSeqVec) {
        blockNum += CeilDiv(seqItem, blockSize);
    }
    int maxSeqAllBatch = *(std::max_element(kvCacheActSeqVec.begin(), kvCacheActSeqVec.end()));
    int maxBlockNumPerBatch = CeilDiv(maxSeqAllBatch, blockSize);

    DataType dType = (std::is_same<T, npu::tile_fwk::float16>::value) ? DT_FP16 : DT_BF16;

    // 1. 设置shape
    std::vector<int64_t> topkIndicesShape = {b, s1, topk - front - near};
    std::vector<int64_t> topkTensorShapeShape = {b, s1};
    std::vector<int64_t> kvNopeCacheShape = {int(blockNum * blockSize), n2 * dn};
    std::vector<int64_t> kRopeCacheShape = {int(blockNum * blockSize), n2 * dr};
    std::vector<int64_t> kvCacheActSeqShape = {b};
    std::vector<int64_t> blockTableShape = {b, maxBlockNumPerBatch};
    std::vector<int64_t> slcActSeqsShape = {b, s1};

    std::vector<int64_t> qNopeShape = {b * s1 * n1, dn};
    std::vector<int64_t> qRopeShape = {b * s1 * n1, dr};
    std::vector<int64_t> kSlcShape = {b * s1 * n2 * smax, dn + dr};
    std::vector<int64_t> vSlcShape = {b * s1 * n2 * smax, dn};

    std::vector<int64_t> shape_selAtten = {b, s1, n1, v_dim};

    // 2. 构造tensor
    Tensor topkIndices(DT_INT32, topkIndicesShape, "topkTensor");
    Tensor topkTensorShape(DT_INT32, topkTensorShapeShape, "topkTensorShape");
    Tensor kvNopeCache(dType, kvNopeCacheShape, "kNopeCache");
    Tensor kRopeCache(dType, kRopeCacheShape, "kRopeCache");
    Tensor kvCacheActSeq(DT_INT32, kvCacheActSeqShape, "kvCacheActSeq");
    Tensor blockTable(DT_INT32, blockTableShape, "blockTable");
    Tensor slcActSeqs(DT_INT32, slcActSeqsShape, "slcActSeqs");

    Tensor qNope(dType, qNopeShape, "qNope");
    Tensor qRope(dType, qRopeShape, "qRope");
    Tensor kSlc(dType, kSlcShape, "kSlc" );
    Tensor vSlc(dType, vSlcShape, "vSlc");

    Tensor kvSlcActSeqsMidOut(DT_INT32, slcActSeqsShape, "kvSlcActSeqsMidOut");
    Tensor attenOut(DT_FP32, shape_selAtten, "attenOut");

    // 3. 为输入填充数据
    auto topkIndicesData = CreateTensorData<int32_t>(topkIndices, "/topk_tensor.bin");
    auto topkTensorShapeData = CreateTensorData<int32_t>(topkTensorShape, "/topk_tensor_shape.bin");
    auto kvNopeCacheData = CreateTensorData<T>(kvNopeCache, "/kv_nope_cache.bin");
    auto kRopeCacheData = CreateTensorData<T>(kRopeCache, "/k_rope_cache.bin");
    auto kvCacheActSeqData = CreateTensorData<int32_t>(kvCacheActSeq, "/kv_cache_actual_seq_len.bin");
    auto blockTableData = CreateTensorData<int32_t>(blockTable, "/block_table.bin");

    auto qNopeData = CreateTensorData<T>(qNope, "/q_nope.bin");
    auto qRopeData = CreateTensorData<T>(qRope, "/q_rope.bin");
    auto slcActSeqsData = CreateTensorData<int32_t>(slcActSeqs, "/kv_slc_actual_seqs.bin");

    auto kvSlcActSeqsMidOutZeroData = RawTensorData::CreateConstantTensor<int32_t>(kvSlcActSeqsMidOut, 0.0);
    auto kSlcOutData = RawTensorData::CreateConstantTensor<T>(kSlc, 0.0);
    auto vSlcOutData = RawTensorData::CreateConstantTensor<T>(vSlc, 0.0);
    auto attenOutZeroData = RawTensorData::CreateConstantTensor<float>(attenOut, 0.0);

    std::vector<int32_t> kvSlcActSeqMidOutGolden = getGoldenVec<int32_t>(slcActSeqsShape, "/kv_slc_actual_seqs.bin");
    std::vector<T> kSlcOutGolden = getGoldenVec<T>(kSlcShape, "/kv_slc_out.bin");
    std::vector<T> vSlcOutGolden = getGoldenVec<T>(vSlcShape, "/kr_slc_out.bin");
    std::vector<float> attenOutGolden = getGoldenVec<float>(shape_selAtten, "/slc_attn_out.bin");

    ProgramData::GetInstance().AppendInputs({
        topkIndicesData, kvNopeCacheData, kRopeCacheData, kvCacheActSeqData,
        blockTableData, qNopeData, qRopeData,
    });

    ProgramData::GetInstance().AppendOutputs({
        attenOutZeroData,
    });

    ProgramData::GetInstance().AppendGoldens({
        RawTensorData::CreateTensor<float>(attenOut, attenOutGolden),
    });

    // 4. 计算接口
    SelectedAttention(topkIndices, kvNopeCache, kRopeCache, kvCacheActSeq, blockTable,
        qNope, qRope, attenOut,
        n1, n2, softmaxScale, front, near, topk, blockSize, cmpBlockSize, slcBlockSize,
        saTileConfig);

#ifndef AC_ENABLE_FRAMEWORK_WITHOUT_CANN
    // 5. 更新输入输出list
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction(),
        {topkIndicesData, kvNopeCacheData, kRopeCacheData, kvCacheActSeqData, blockTableData, // genkvSlc
         qNopeData, qRopeData// slcAtten
        }, // input list
        {attenOutZeroData}); // output list

    std::cout << "slcAttnOut ====== " << std::endl;
    EXPECT_TRUE(resultCmp<float>(attenOutGolden, (float *)attenOutZeroData->data(), 0.0005f));
#endif
}

TEST_F(DynamicKvSATest, kv_slc_attn_b48_s1_fp16_perf) {
    NSAV1SimpleParams params = NSAV1SimpleParams::getDecodeParams();

    int paramsSize = 7;
    std::vector<int> inputParams(paramsSize);
    readInput<int>(GetGoldenDir() + "/input_params.bin", inputParams); // 在golden中保存了变化的参数，便于调试
    params.b = inputParams[0]; // 16
    params.s1 = inputParams[1];
    params.s2 = inputParams[2];
    params.n1 = inputParams[3];
    params.n2 = inputParams[4];

    SATileShapeConfig saTileConfig;
    saTileConfig.kvSlcV0TileShape = {64, 256}; // slcBlockSize=64
    const int gTile = 128; // for gLoop split
    const int sTile = 1024; // for s2Loop split
    saTileConfig.gTile = gTile;
    saTileConfig.sKvTile = sTile;
    saTileConfig.c1TileShape = {gTile, gTile, 64, 64, 256, 256}; // (n1, dn+dr) @ (s2Tile, dn+dr) -> (n1, s2Tile)
    saTileConfig.v1TileShape = {16, 256}; // (n1, s2Tile)
    saTileConfig.c2TileShape = {gTile, gTile, 128, 128, 128, 128}; // (n1, s2Tile) @ (s2Tile, dn) -> (n1, d)
    saTileConfig.v2TileShape = {64, 128}; // (n1, d)

    TestKvSlcAttn<npu::tile_fwk::float16>(params, saTileConfig);
}

TEST_F(DynamicKvSATest, kv_slc_attn_b32_s2_bf16_perf) {
    NSAV1SimpleParams params = NSAV1SimpleParams::getDecodeParams();

    int paramsSize = 7;
    std::vector<int> inputParams(paramsSize);
    readInput<int>(GetGoldenDir() + "/input_params.bin", inputParams); // 在golden中保存了变化的参数，便于调试
    params.b = inputParams[0]; // 32
    params.s1 = inputParams[1];
    params.s2 = inputParams[2];
    params.n1 = inputParams[3];
    params.n2 = inputParams[4];

    SATileShapeConfig saTileConfig;
    saTileConfig.kvSlcV0TileShape = {64, 256}; // slcBlockSize=64
    const int gTile = 128; // for gLoop split
    const int sTile = 1024; // for s2Loop split
    saTileConfig.gTile = gTile;
    saTileConfig.sKvTile = sTile;
    saTileConfig.c1TileShape = {gTile, gTile, 64, 64, 256, 256}; // (n1, dn+dr) @ (s2Tile, dn+dr) -> (n1, s2Tile)
    saTileConfig.v1TileShape = {16, 256}; // (n1, s2Tile)
    saTileConfig.c2TileShape = {gTile, gTile, 128, 128, 128, 128}; // (n1, s2Tile) @ (s2Tile, dn) -> (n1, d)
    saTileConfig.v2TileShape = {64, 128}; // (n1, d)

    TestKvSlcAttn<npu::tile_fwk::bfloat16>(params, saTileConfig);
}

TEST_F(DynamicKvSATest, kv_slc_attn_b2_s1_fp16) { // 2batch, 128K, 带尾块场景, kvCacheActSeq=[128*1024, 64*1024+35], kvSlcActSeq=[16*64, 15*64+35]
    NSAV1SimpleParams params = NSAV1SimpleParams::getDecodeParams();

    int paramsSize = 7;
    std::vector<int> inputParams(paramsSize);
    readInput<int>(GetGoldenDir() + "/input_params.bin", inputParams); // 在golden中保存了变化的参数，便于调试
    params.b = inputParams[0]; // 16
    params.s1 = inputParams[1];
    params.s2 = inputParams[2];
    params.n1 = inputParams[3];
    params.n2 = inputParams[4];

    SATileShapeConfig saTileConfig;
    saTileConfig.kvSlcV0TileShape = {64, 256}; // slcBlockSize=64
    const int gTile = 128; // for gLoop split
    const int sTile = 1024; // for s2Loop split
    saTileConfig.gTile = gTile;
    saTileConfig.sKvTile = sTile;
    saTileConfig.c1TileShape = {gTile, gTile, 64, 64, 256, 256}; // (n1, dn+dr) @ (s2Tile, dn+dr) -> (n1, s2Tile)
    saTileConfig.v1TileShape = {16, 256}; // (n1, s2Tile)
    saTileConfig.c2TileShape = {gTile, gTile, 128, 128, 128, 128}; // (n1, s2Tile) @ (s2Tile, dn) -> (n1, d)
    saTileConfig.v2TileShape = {64, 128}; // (n1, d)

    TestKvSlcAttn<npu::tile_fwk::float16>(params, saTileConfig);
}
