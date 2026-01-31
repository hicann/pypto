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
 * \file test_dynamic_attention.cpp
 * \brief
 */

#include "test_dev_func_runner.h"
#include "test_suite_stest_ops.h"
#include "operator/models/deepseek/attention.h"

using namespace npu::tile_fwk;
using namespace npu::tile_fwk::dynamic;
class DynamicAttention : public npu::tile_fwk::stest::TestSuite_STest_Ops_Aihac {};

namespace {
template <typename T = npu::tile_fwk::float16, typename wDtype = int8_t, bool splitK = false, bool nz = false, bool usePrefetch = false>
void TestDynamicAttention(std::vector<int> &params, PaTileShapeConfig &paTileConfig, string dataPath,
        uint64_t timeThreshold, bool isQuant = false, bool isSmooth = false) {
    (void) timeThreshold;

    config::SetRuntimeOption(DEVICE_SCHED_MODE, static_cast<uint8_t>(MachineScheduleConfig::L2CACHE_AFFINITY_SCH));
    config::SetRuntimeOption(STITCH_FUNCTION_NUM_INITIAL, 128);
    std::string cacheMode = "PA_NZ";

    int b = params[0];
    int s = params[1];
    int s2 = params[2];
    int n = params[3];
    int h = params[4];
    int qLoraRank = params[5];
    int qkNopeHeadDim = params[6];
    int qkRopeHeadDim = params[7];
    int kvLoraRank = params[8];

    int vHeadDim = params[9];
    int blockSize =params[10];
    int q_head_dim = qkNopeHeadDim + qkRopeHeadDim;

    std::vector<int> atcSeqs(b);
    readInput<int>(dataPath + "/actual_seq_len.bin", atcSeqs);

    int blockNum = 0;
    for (auto seq : atcSeqs) {
        blockNum += CeilDiv(seq, blockSize);
    }

    float softmaxScale = static_cast<float>(1.0 / sqrtf((kvLoraRank + qkRopeHeadDim)));
    // blockTable: (b, maxBlockNumPerBatch)
    int maxSeqAllBatch = *(std::max_element(atcSeqs.begin(), atcSeqs.end()));
    int maxBlockNumPerBatch = CeilDiv(maxSeqAllBatch, blockSize);

    DataType dType = DT_FP32;
    if (std::is_same<T, npu::tile_fwk::float16>::value) {
        dType = DT_FP16;
    } else if (std::is_same<T, npu::tile_fwk::bfloat16>::value) {
        dType = DT_BF16;
    } else {
        dType = DT_FP32;
    }

    DataType dTypeQuantIn = isQuant ? DT_INT8 : dType;

    std::vector<int64_t> x_shape = {b, s, h};
    std::vector<int64_t> w_qa_shape = {h, qLoraRank};
    std::vector<int64_t> w_qb_shape = {qLoraRank, n * q_head_dim};
    std::vector<int64_t> w_kv_a_shape = {h, kvLoraRank + qkRopeHeadDim};
    std::vector<int64_t> w_kv_b_k_shape = {n, qkNopeHeadDim, kvLoraRank};
    std::vector<int64_t> cos_shape = {b, s, qkRopeHeadDim};
    std::vector<int64_t> gamma_cq_shape = {qLoraRank};
    std::vector<int64_t> gamma_ckv_shape = {kvLoraRank};
    std::vector<int64_t> kv_len_shape = {b, s};
    std::vector<int64_t> kv_cache_shape = {b, 1, s2, kvLoraRank};
    std::vector<int64_t> kr_cache_shape = {b, 1, s2, qkRopeHeadDim};
    if (cacheMode != "BNSD") {
        kv_cache_shape = {blockNum, blockSize, 1, kvLoraRank};
        kr_cache_shape = {blockNum, blockSize, 1, qkRopeHeadDim};
    }
    std::vector<int64_t> w_qb_scale_shape = {1, n * q_head_dim};
    std::vector<int64_t> smooth_cq_shape{1, qLoraRank};
    // pa
    std::vector<int64_t> blockTableShape = {b, 1, s2, qkRopeHeadDim};
    // output
    std::vector<int64_t> q_out_shape = {b, s, n, kvLoraRank};
    std::vector<int64_t> q_rope_out_shape = {b, s, n, qkRopeHeadDim};
    std::vector<int64_t> kv_cache_out_shape = {b, 1, s2, kvLoraRank};
    std::vector<int64_t> kr_cache_out_shape = {b, 1, s2, qkRopeHeadDim};
    std::vector<int64_t> fake_out_shape = {b, s, kvLoraRank + qkRopeHeadDim};
    std::vector<int64_t> fake_out_shape1 = {n, b * s, qkNopeHeadDim};

    int capacity_x = std::accumulate(x_shape.begin(), x_shape.end(), 1, std::multiplies<>());
    int wDqCapacity = std::accumulate(w_qa_shape.begin(), w_qa_shape.end(), 1, std::multiplies<>());
    int wUqQrCapacity = std::accumulate(w_qb_shape.begin(), w_qb_shape.end(), 1, std::multiplies<>());
    int wDkvKrCapacity = std::accumulate(w_kv_a_shape.begin(), w_kv_a_shape.end(), 1, std::multiplies<>());
    int wUkCapacity = std::accumulate(w_kv_b_k_shape.begin(), w_kv_b_k_shape.end(), 1, std::multiplies<>());
    int capacity_cos = std::accumulate(cos_shape.begin(), cos_shape.end(), 1, std::multiplies<>());
    int capacity_gamma_cq = std::accumulate(gamma_cq_shape.begin(), gamma_cq_shape.end(), 1, std::multiplies<>());
    int capacity_gamma_ckv = std::accumulate(gamma_ckv_shape.begin(), gamma_ckv_shape.end(), 1, std::multiplies<>());
    int capacity_kv_len = std::accumulate(kv_len_shape.begin(), kv_len_shape.end(), 1, std::multiplies<>());
    int capacity_kv_cache = std::accumulate(kv_cache_shape.begin(), kv_cache_shape.end(), 1, std::multiplies<>());
    int capacity_kr_cache = std::accumulate(kr_cache_shape.begin(), kr_cache_shape.end(), 1, std::multiplies<>());
    int capacity_w_qb_scale = std::accumulate(w_qb_scale_shape.begin(), w_qb_scale_shape.end(), 1, std::multiplies<>());
    int capacity_smooth_cq = std::accumulate(smooth_cq_shape.begin(), smooth_cq_shape.end(), 1, std::multiplies<>());
    // output
    int capacity_q_out = std::accumulate(q_out_shape.begin(), q_out_shape.end(), 1, std::multiplies<>());
    int capacity_q_rope_out = std::accumulate(q_rope_out_shape.begin(), q_rope_out_shape.end(), 1, std::multiplies<>());
    int capacity_fake_out = std::accumulate(fake_out_shape.begin(), fake_out_shape.end(), 1, std::multiplies<>());
    int capacity_fake_out1 = std::accumulate(fake_out_shape1.begin(), fake_out_shape1.end(), 1, std::multiplies<>());

    TileOpFormat weightFormat = nz ? TileOpFormat::TILEOP_NZ : TileOpFormat::TILEOP_ND;
    TileOpFormat paFormat = cacheMode == "PA_NZ" ? TileOpFormat::TILEOP_NZ : TileOpFormat::TILEOP_ND;

    // mla_prolog
    Tensor x(dType, x_shape, "x");
    Tensor wDq(dType, w_qa_shape, "wDq", weightFormat);
    Tensor wUqQr(dTypeQuantIn, w_qb_shape, "wUqQr", weightFormat);
    if constexpr (usePrefetch) {
        wDq.SetCachePolicy(CachePolicy::PREFETCH, true);
        wUqQr.SetCachePolicy(CachePolicy::PREFETCH, true);
    }
    Tensor wDkvKr(dType, w_kv_a_shape, "wDkvKr", weightFormat);
    Tensor wUk(dType, w_kv_b_k_shape, "wUk", weightFormat);
    Tensor gamma_cq(dType, gamma_cq_shape, "gamma_cq");
    Tensor gamma_ckv(dType, gamma_ckv_shape, "gamma_ckv");
    Tensor cos(dType, cos_shape, "cos");
    Tensor sin(dType, cos_shape, "sin");
    Tensor kv_len(DT_INT64, kv_len_shape, "kv_len"); // int64
    Tensor kv_cache(dType, kv_cache_shape, "kv_cache", paFormat);
    Tensor kr_cache(dType, kr_cache_shape, "kr_cache", paFormat);
    Tensor w_qb_scale;
    Tensor smooth_cq;
    if (isQuant) {
        w_qb_scale = Tensor(DT_FP32, w_qb_scale_shape, "w_qb_scale");
        if (isSmooth) {
            smooth_cq = Tensor(DT_FP32, smooth_cq_shape, "smooth_cq");
        }
    }

    Tensor output_q(dType, {b*s*n, kvLoraRank}, "output_q");
    Tensor output_q_rope(dType, {b*s*n, qkRopeHeadDim}, "output_q_rope");
    Tensor output_kv_cache(dType, {b * 1 * s2, kvLoraRank}, "output_kv_cache", paFormat);
    Tensor output_kr_cache(dType, {b * 1 * s2, qkRopeHeadDim}, "output_kr_cache", paFormat);

    // pa
    Tensor blockTable(DT_INT32, {b, maxBlockNumPerBatch}, "blockTable");
    Tensor actSeqs(DT_INT32, {b}, "actSeqs");
    // pa output
    Tensor paOut(DT_FP32, {b * n * s, kvLoraRank}, "paOut");
    // post
    Tensor weightUV(dType, {n, kvLoraRank, vHeadDim}, "weightUV");
    weightUV.SetCachePolicy(CachePolicy::NONE_CACHEABLE, true);
    Tensor weightO(DT_INT8, {n * vHeadDim, h}, "weightO", weightFormat); // NZ
    weightO.SetCachePolicy(CachePolicy::NONE_CACHEABLE, true);
    Tensor weightOScaleW(DT_FP32, {1, h}, "weightOScaleW");
    weightOScaleW.SetCachePolicy(CachePolicy::NONE_CACHEABLE, true);
    // output
    Tensor postOut(dType, x_shape, "postOut");
    postOut.SetCachePolicy(CachePolicy::NONE_CACHEABLE, true);

    int tileB = b;
    RoPETileShapeConfigNew ropeConfig {
        {tileB, 1, 64}, // (b,s,d)
        {tileB, 1, 1, 64}, // Q (b,s,n,d)
        {tileB, 1, 1, 64}, // K (b,s,1,d)
        {tileB, 1, 1, 32, 2} // (b,s,n,d//2,2)
    };

    std::vector<T> xValue(capacity_x, 0);
    std::vector<T> wDqValue(wDqCapacity, 0);
    std::vector<wDtype> wUqQrValue(wUqQrCapacity, 0);
    std::vector<T> wUkValue(wUkCapacity, 0);
    std::vector<T> wDkvKrValue(wDkvKrCapacity, 0);
    std::vector<T> gammaCqValue(capacity_gamma_cq, 0);
    std::vector<T> gammaCkvValue(capacity_gamma_ckv, 0);
    std::vector<T> sinValue(capacity_cos, 0);
    std::vector<T> cosValue(capacity_cos, 0);
    std::vector<int64_t> kvLenValue(capacity_kv_len, 0);
    std::vector<T> kvCacheValue(capacity_kv_cache, 0);
    std::vector<T> krCacheValue(capacity_kr_cache, 0);
    std::vector<float> wQbScaleValue(capacity_w_qb_scale, 0);
    std::vector<float> smoothCqValue(capacity_smooth_cq, 0);
    //pa
    std::vector<int32_t> blockTableValue(b*maxBlockNumPerBatch, 0);
    std::vector<int32_t> actSeqsValue(b, s2);
    //post
    std::vector<T> weightUVValue(n*kvLoraRank*vHeadDim, 0);
    std::vector<int8_t> weightOValue(n * vHeadDim*h, 0);
    std::vector<float> weightOScaleWValue(h, 0);

    // read data
    readInput<T>(dataPath + "/x.bin", xValue);
    readInput<T>(dataPath + "/wDq.bin", wDqValue);
    readInput<wDtype>(dataPath + "/wUqQr.bin", wUqQrValue);
    readInput<T>(dataPath + "/wUk.bin", wUkValue);
    readInput<T>(dataPath + "/wDkvKr.bin", wDkvKrValue);
    readInput<T>(dataPath + "/gamma_cq.bin", gammaCqValue);
    readInput<T>(dataPath + "/gamma_ckv.bin", gammaCkvValue);
    readInput<T>(dataPath + "/sin.bin", sinValue);
    readInput<T>(dataPath + "/cos.bin", cosValue);
    readInput<int64_t>(dataPath + "/kv_len.bin", kvLenValue);
    readInput<T>(dataPath + "/kv_cache.bin", kvCacheValue);
    readInput<T>(dataPath + "/kr_cache.bin", krCacheValue);
    if (isQuant) {
        readInput<float>(dataPath + "/w_qb_scale.bin", wQbScaleValue);
        if (isSmooth) {
            readInput<float>(dataPath + "/smooth_cq.bin", smoothCqValue);
        }
    }
    // pa
    readInput<int32_t>(dataPath + "/block_table.bin", blockTableValue);
    readInput<int32_t>(dataPath + "/actual_seq_len.bin", actSeqsValue);
    // post
    readInput<T>(dataPath + "/w_uv.bin", weightUVValue);
    readInput<int8_t>(dataPath + "/w_o.bin", weightOValue); // NZ
    readInput<float>(dataPath + "/w_o_scale_w.bin", weightOScaleWValue);

     // golden
    std::vector<T> q_golden(capacity_q_out, 0);
    std::vector<T> q_rope_golden(capacity_q_rope_out, 0);
    std::vector<T> kv_cache_golden(capacity_kv_cache, 0);
    std::vector<T> kr_cache_golden(capacity_kr_cache, 0);
    std::vector<T> golden5(capacity_fake_out, 0);
    std::vector<T> golden6(capacity_fake_out1, 0);

    std::vector<float> atten_out_golden(b * n * s*kvLoraRank, 0);
    std::vector<T> attn_output_golden(capacity_x, 0);

    readInput<T>(dataPath + "/q_golden.bin", q_golden);
    readInput<T>(dataPath + "/q_rope_golden.bin", q_rope_golden);
    readInput<T>(dataPath + "/kv_cache_golden.bin", kv_cache_golden);
    readInput<T>(dataPath + "/kr_cache_golden.bin", kr_cache_golden);

    readInput<float>(dataPath + "/atten_out.bin", atten_out_golden); // pa out
    readInput<T>(dataPath + "/attn_output.bin", attn_output_golden);    // attention out

    auto xData = RawTensorData::CreateTensor<T>(x, xValue);
    auto wDqData = RawTensorData::CreateTensor<T>(wDq, wDqValue);
    auto wUqQrData = RawTensorData::CreateTensor<wDtype>(wUqQr, wUqQrValue);
    auto wUkData = RawTensorData::CreateTensor<T>(wUk, wUkValue);
    auto wDkvKrData = RawTensorData::CreateTensor<T>(wDkvKr, wDkvKrValue);
    auto gammaCqData = RawTensorData::CreateTensor<T>(gamma_cq, gammaCqValue);
    auto gammaCkvData = RawTensorData::CreateTensor<T>(gamma_ckv, gammaCkvValue);
    auto cosData = RawTensorData::CreateTensor<T>(cos, cosValue);
    auto sinData = RawTensorData::CreateTensor<T>(sin, sinValue);
    auto kvLenData = RawTensorData::CreateTensor<int64_t>(kv_len, kvLenValue);
    auto kvCacheData = RawTensorData::CreateTensor<T>(kv_cache, kvCacheValue);
    auto krCacheData = RawTensorData::CreateTensor<T>(kr_cache, krCacheValue);
    std::shared_ptr<RawTensorData> wQbScaleData;
    std::shared_ptr<RawTensorData> smoothCqData;
    if (isQuant) {
        wQbScaleData = RawTensorData::CreateTensor<float>(w_qb_scale, wQbScaleValue);
        if (isSmooth) {
            smoothCqData = RawTensorData::CreateTensor<float>(smooth_cq, smoothCqValue);
        }
    }

    auto outputQData = RawTensorData::CreateConstantTensor<T>(output_q, 0.0);
    auto outputQRopeData = RawTensorData::CreateConstantTensor<T>(output_q_rope, 0.0);
    // pa
    auto blockTableData = RawTensorData::CreateTensor<int32_t>(blockTable, blockTableValue);
    auto actSeqsData = RawTensorData::CreateTensor<int32_t>(actSeqs, actSeqsValue);
    auto paOutData = RawTensorData::CreateConstantTensor<float>(paOut, 0.0);
    // post
    auto weightUVData = RawTensorData::CreateTensor<T>(weightUV, weightUVValue);
    auto weightOData = RawTensorData::CreateTensor<int8_t>(weightO, weightOValue);
    auto weightOScaleWData = RawTensorData::CreateTensor<float>(weightOScaleW, weightOScaleWValue);
    // output
    auto postOutData = RawTensorData::CreateConstantTensor<T>(postOut, 0.0);

    MlaQuantInputs quantInputs;
    if (isQuant) {
        quantInputs.dequantScaleWUqQr = w_qb_scale;
        if (isSmooth) {
            quantInputs.smoothScalesCq = smooth_cq;
        }
    }
    Attention(x, wDq, wUqQr, wUk, wDkvKr, gamma_cq, gamma_ckv, sin, cos, kv_len, kv_cache, kr_cache,
            output_q, output_q_rope, output_kv_cache, output_kr_cache, quantInputs, ropeConfig, /*---*/
            blockTable, actSeqs, paOut, blockSize, softmaxScale, paTileConfig, /*---*/
            weightUV, weightO, weightOScaleW, postOut, 1e-5f, 1e-5f, cacheMode);

#ifdef BUILD_WITH_CANN
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction(),
        {xData, wDqData, wUqQrData, wUkData, wDkvKrData, gammaCqData, gammaCkvData, sinData, cosData, kvLenData,
         kvCacheData, krCacheData, wQbScaleData, smoothCqData,
         blockTableData, actSeqsData, weightUVData, weightOData, weightOScaleWData},
        {postOutData});

    std::cout << "====== kvCacheData out: " << std::endl;
    EXPECT_TRUE(resultCmp<T>(kv_cache_golden, (T *)kvCacheData->data(), 0.001f));
    std::cout << "====== krCacheData out: " << std::endl;
    EXPECT_TRUE(resultCmp<T>(kr_cache_golden, (T *)krCacheData->data(), 0.001f));
    std::cout << "====== postOutData out: " << std::endl;
    EXPECT_TRUE(resultCmp<T>(attn_output_golden, (T *)postOutData->data(), 0.03f, 0, 1000, false, true, 0));
#endif
}

TEST_F(DynamicAttention, dynamic_attention_low) { // b_n_s_s2_h_q_lora_rank
    int b = 4;
    int s = 1;
    int s2 = 256;
    int h = 7168;   //7168
    int n = 32;
    int qLoraRank = 1536;   //1536
    int qkNopeHeadDim = 128;
    int qkRopeHeadDim = 64;
    int kvLoraRank = 512;
    int vHeadDim = 128;
    int blockSize = 256;
    std::vector<int> params = {b, s, s2, n, h, qLoraRank, qkNopeHeadDim, qkRopeHeadDim, kvLoraRank, vHeadDim, blockSize};

    const bool splitK = false;
    const bool nz = false;

    PaTileShapeConfig tileConfig;
    const int nTile = 32;
    tileConfig.headNumQTile = nTile;
    tileConfig.v0TileShape = {nTile, 64};
    tileConfig.c1TileShape = {nTile, nTile, 64, 64, 128, 128};
    tileConfig.v1TileShape = {nTile, 64};
    tileConfig.c2TileShape = {nTile, nTile, 64, 64, 128, 128};
    tileConfig.v2TileShape = {nTile, 64};

    TestDynamicAttention<npu::tile_fwk::float16, npu::tile_fwk::float16, splitK, nz>(params, tileConfig, GetGoldenDir(), 10000, false);
}

TEST_F(DynamicAttention, dynamic_attention_high) { // b_n_s_s2_h_q_lora_rank
    int b = 32;
    int s = 1;
    int s2 = 4096;
    int h = 7168;
    int n = 128;
    int qLoraRank = 1536;
    int qkNopeHeadDim = 128;
    int qkRopeHeadDim = 64;
    int kvLoraRank = 512;
    int vHeadDim = 128;
    int blockSize = 4096;
    std::vector<int> params = {b, s, s2, n, h, qLoraRank, qkNopeHeadDim, qkRopeHeadDim, kvLoraRank, vHeadDim, blockSize};

    const bool splitK = false;
    const bool nz = false;

    PaTileShapeConfig paTileConfig;
    const int nTile = 128;
    paTileConfig.headNumQTile = nTile;

    paTileConfig.c1TileShape = {nTile, nTile, 128, 256, 128, 256};
    paTileConfig.v1TileShape = {16, 256};
    paTileConfig.c2TileShape = {nTile, nTile, 256, 256, 128, 128};
    paTileConfig.v2TileShape = {16, 256};

    TestDynamicAttention<npu::tile_fwk::float16, npu::tile_fwk::float16, splitK, nz>(params, paTileConfig, GetGoldenDir(), 10000, false);
}

TEST_F(DynamicAttention, low_latency_quant_smooth_nz) { // b_n_s_s2_h_q_lora_rank
    int b = 4;
    int s = 1;
    int s2 = 256;
    int h = 7168;
    int n = 32;
    int qLoraRank = 1536;
    int qkNopeHeadDim = 128;
    int qkRopeHeadDim = 64;
    int kvLoraRank = 512;
    int vHeadDim = 128;
    int blockSize = 256;
    std::vector<int> params = {b, s, s2, n, h, qLoraRank, qkNopeHeadDim, qkRopeHeadDim, kvLoraRank, vHeadDim, blockSize};

    const bool splitK = false;
    const bool nz = true;
    const bool isQuant = true;
    const bool isSmooth = true;

    PaTileShapeConfig paTileConfig;
    const int nTile = 32;
    paTileConfig.headNumQTile = nTile;
    paTileConfig.v0TileShape = {nTile, 64};
    paTileConfig.c1TileShape = {nTile, nTile, 64, 64, 128, 128};
    paTileConfig.v1TileShape = {nTile, 64};
    paTileConfig.c2TileShape = {nTile, nTile, 64, 64, 128, 128};
    paTileConfig.v2TileShape = {nTile, 64};

    TestDynamicAttention<npu::tile_fwk::float16, int8_t, splitK, nz>(params, paTileConfig, GetGoldenDir(), 10000,
            isQuant, isSmooth);
}

TEST_F(DynamicAttention, high_throughput_quant_smooth_nz) { // b_n_s_s2_h_q_lora_rank
    int b = 32;
    int s = 1;
    int s2 = 4096;
    int h = 7168;
    int n = 128;
    int qLoraRank = 1536;
    int qkNopeHeadDim = 128;
    int qkRopeHeadDim = 64;
    int kvLoraRank = 512;
    int vHeadDim = 128;
    int blockSize = 4096;
    std::vector<int> params = {b, s, s2, n, h, qLoraRank, qkNopeHeadDim, qkRopeHeadDim, kvLoraRank, vHeadDim, blockSize};

    const bool splitK = false;
    const bool nz = true;
    const bool isQuant = true;
    const bool isSmooth = true;

    PaTileShapeConfig paTileConfig;
    const int nTile = 128;
    paTileConfig.headNumQTile = nTile;

    paTileConfig.c1TileShape = {nTile, nTile, 128, 256, 128, 256};
    paTileConfig.v1TileShape = {16, 256};
    paTileConfig.c2TileShape = {nTile, nTile, 256, 256, 128, 128};
    paTileConfig.v2TileShape = {16, 256};

    // Set Stitching window optimization
    config::SetRuntimeOption<int>(STITCH_FUNCTION_INNER_MEMORY, 11);
    config::SetRuntimeOption<int>(STITCH_FUNCTION_OUTCAST_MEMORY, 32);

    TestDynamicAttention<npu::tile_fwk::float16, int8_t, splitK, nz>(params, paTileConfig, GetGoldenDir(), 10000,
            isQuant, isSmooth);
}

} // namespace
