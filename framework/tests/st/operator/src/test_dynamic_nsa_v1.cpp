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
 * \file test_dynamic_nsa.cpp
 * \brief
 */

#include <functional>
#include <vector>
#include "operator/models/nsa/attention_post.h"
#include "test_dev_func_runner.h"
#include "test_common.h"
#include "test_data_prepare.h"
#include "test_suite_stest_ops.h"
#include "operator/models/nsa/dynamic_nsa_v1.h"
#include "operator/models/nsa/fused_compress_kv_select.h"

using namespace npu::tile_fwk;
using namespace npu::tile_fwk::dynamic;
class DynamicNSATest : public npu::tile_fwk::stest::TestSuite_STest_Ops_Aihac {};

void SetPreConfig() {
    config::SetHostOption(COMPILE_STAGE, GEN_KERNEL_CODE);
}

template <typename T>
static std::vector<T> getGoldenVec(std::vector<int64_t> shape, std::string fileName) {
    int capacity = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<>());
    std::vector<T> golden(capacity, 0);
    readInput<T>(GetGoldenDir() + fileName, golden);
    return golden;
}

template <typename T = npu::tile_fwk::float16, typename wDtype = int8_t, bool isSmooth = false, bool nz = false,
    bool ci = false, bool debug = false>
void TestNsa(const NSAV1SimpleParams &params, const MlaTileConfig &prologConfig,
    WinAttenTileShapeConfig &winAttntileConfig, SATileShapeConfig &saTileConfig,
    PostTileConfig &postConfig, CmpAttnTile &cmpTileConfig, float precision, std::string cacheMode = "PA_BSND") {
    (void)precision;
    SetPreConfig();

    float eps = params.eps;
    int b = params.b;
    int s1 = params.s1;
    int s2 = params.s2;
    int n1 = params.n1;
    int n2 = params.n2;
    int h = params.h;
    int v_dim = params.kv_lora_rank;
    int qLoraRank = params.q_lora_rank;
    int qkNopeHeadDim = params.qk_nope_head_dim;
    int qkRopeHeadDim = params.qk_rope_head_dim;
    int qHeadDim = qkNopeHeadDim + qkRopeHeadDim;
    int smax = params.topk * params.slcBlockSize;
    int dn = v_dim;
    int dr = params.rope_dim;
    float softmaxScale = static_cast<float>(1.0 / sqrtf((dn + dr)));
    int blockSize = params.blockSize;
    int winSize = params.winSize;
    int slcBlockSize = params.slcBlockSize;
    int front = params.front;
    int near = params.near;
    int topk = params.topk;

    std::vector<int> kvCacheActSeqVec(b);
    readInput<int>(GetGoldenDir() + "/kv_cache_actual_seq_len.bin", kvCacheActSeqVec);
    int blockNum = 0;
    for (auto seqItem : kvCacheActSeqVec) {
        blockNum += CeilDiv(seqItem, blockSize);
    }
    std::cout << "========= blockNum " << blockNum << std::endl;
    int maxSeqAllBatch = *(std::max_element(kvCacheActSeqVec.begin(), kvCacheActSeqVec.end()));
    int maxBlockNumPerBatch = CeilDiv(maxSeqAllBatch, blockSize);

    int vHeadDim = params.vHeadDim;
    DataType dType = (std::is_same<T, npu::tile_fwk::float16>::value) ? DT_FP16 : DT_BF16;
    bool isQuant = std::is_same<wDtype, int8_t>::value;
    DataType dTypeQuant = isQuant ? DT_INT8 : dType;

    // 1. 设置shape
    // MlaProlog
    std::vector<int64_t> xShape = {b, s1, h};
    std::vector<int64_t> wDqShape = {h, qLoraRank};
    std::vector<int64_t> wUqQrShape = {qLoraRank, n1 * qHeadDim};
    std::vector<int64_t> wDkvKrShape = {h, v_dim + qkRopeHeadDim};
    std::vector<int64_t> wUkShape = {n1, qkNopeHeadDim, v_dim};
    std::vector<int64_t> cosShape = {b, s1, qkRopeHeadDim};
    std::vector<int64_t> gammaCqShape = {qLoraRank};
    std::vector<int64_t> gammaCkvShape = {v_dim};
    std::vector<int64_t> kvLenShape = {b, s1};
    std::vector<int64_t> kvCacheShape = {b, n2, s2, v_dim};
    std::vector<int64_t> krCacheShape = {b, n2, s2, qkRopeHeadDim};
    std::vector<int64_t> kvCacheOutShape = {b, n2, s2, v_dim};
    std::vector<int64_t> krCacheOutShape = {b, n2, s2, qkRopeHeadDim};
    if (cacheMode != "BNSD") {
        int blockNum2 = b * (s2 / blockSize);
        std::cout << "========= blockNum2 " << blockNum2 << std::endl;
        kvCacheShape = {blockNum, blockSize, n2, v_dim};
        krCacheShape = {blockNum, blockSize, n2, qkRopeHeadDim};
        kvCacheOutShape = {blockNum * blockSize, n2 * v_dim};
        krCacheOutShape = {blockNum * blockSize, n2 * qkRopeHeadDim};
    }
    std::vector<int64_t> wQbScaleShape = {1, n1 * qHeadDim};
    std::vector<int64_t> smoothCqShape{1, qLoraRank};
    std::vector<int64_t> qOutShape = {b, s1, n1, v_dim};
    std::vector<int64_t> qRopeOutShape = {b, s1, n1, qkRopeHeadDim};

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

    std::vector<int64_t> gateW1Shape = {h, 4 * h};
    std::vector<int64_t> gateW2Shape = {4 * h, 3 * n1};
    std::vector<int64_t> gateSimW1Shape = {h, 3 * n1};
    // std::vector<int64_t> gatingScoreShape = {b, s1, n1, 3};

    std::vector<int64_t> shape_cmpAtten = {b, s1, n1, v_dim};
    std::vector<int64_t> shape_selAtten = {b, s1, n1, v_dim};
    std::vector<int64_t> shape_winAtten = {b, s1, n1, v_dim};
    std::vector<int64_t> shape_attentionOut = {b, s1, n1, v_dim};

    // post: shape
    std::vector<int64_t> wUvShape = {n1, v_dim, vHeadDim};
    std::vector<int64_t> wUvScaleShape = {n1, 1, vHeadDim};
    std::vector<int64_t> smoothWUvShape = {1, v_dim};
    std::vector<int64_t> woShape = {n1 * vHeadDim, h};
    std::vector<int64_t> woScaleShape = {1, h};
    std::vector<int64_t> smoothWoShape = {1, n1 * vHeadDim};
    std::vector<int64_t> outShape = {b, s1, h};

    // 2. 构造tensor
    // MlaProlog
    Tensor x(dType, xShape, "x");
    TileOpFormat weightFormat = nz ? TileOpFormat::TILEOP_NZ : TileOpFormat::TILEOP_ND;
    Tensor wDq(dType, wDqShape, "wDq", weightFormat);
    Tensor wUqQr(dTypeQuant, wUqQrShape, "wUqQr", weightFormat);
    const bool usePrefetch = true;
    if constexpr (usePrefetch) {
        wDq.SetCachePolicy(CachePolicy::PREFETCH, true);
        wUqQr.SetCachePolicy(CachePolicy::PREFETCH, true);
    }
    Tensor wDkvKr(dType, wDkvKrShape, "wDkvKr", weightFormat);
    Tensor wUk(dType, wUkShape, "wUk", weightFormat);
    Tensor gammaCq(dType, gammaCqShape, "gammaCq");
    Tensor gammaCkv(dType, gammaCkvShape, "gammaCkv");
    Tensor cos(dType, cosShape, "cos");
    Tensor sin(dType, cosShape, "sin");
    Tensor cacheIndex(DT_INT64, kvLenShape, "cacheIndex"); // int64
    Tensor kvCache(dType, kvCacheShape, "kvCache");
    Tensor krCache(dType, krCacheShape, "krCache");
    Tensor wQbScale(DT_FP32, wQbScaleShape, "wQbScale");
    Tensor smoothCq(DT_FP32, smoothCqShape, "smoothCq");
    // MlaProlog output
    Tensor outputKvCache(dType, kvCacheOutShape, "outputKvCache");
    Tensor outputKrCache(dType, krCacheOutShape, "outputKrCache");
    // Tensor outputQ(dType, qOutShape, "outputQ");
    // Tensor outputQRope(dType, qRopeOutShape, "outputQRope");

    Tensor topkIndices(DT_INT32, topkIndicesShape, "topkTensor");
    Tensor topkTensorShape(DT_INT32, topkTensorShapeShape, "topkTensorShape");
    Tensor kvNopeCache(dType, kvNopeCacheShape, "kNopeCache");
    Tensor kRopeCache(dType, kRopeCacheShape, "vNopeCache");
    Tensor kvCacheActSeq(DT_INT32, kvCacheActSeqShape, "kvCacheActSeq");
    Tensor blockTable(DT_INT32, blockTableShape, "blockTable");

    Tensor slcActSeqs(DT_INT32, slcActSeqsShape, "slcActSeqs");
    // Tensor qNope(dType, qNopeShape, "qNope");
    // Tensor qRope(dType, qRopeShape, "qRope");
    Tensor kSlc(dType, kSlcShape, "kSlc");
    Tensor vSlc(dType, vSlcShape, "vSlc");

    Tensor gateW1(dType, gateW1Shape, "gateW1");
    Tensor gateW2(dType, gateW2Shape, "gateW2");
    Tensor gateSimW1(dType, gateSimW1Shape, "gateSimW1");
    // Tensor gatingScore(dType, gatingScoreShape, "gatingScore");

    Tensor cmpAtten(dType, shape_cmpAtten, "cmpAtten");
    Tensor slcAttn(DT_FP32, shape_selAtten, "selAtten"); // fp32输入
    // Tensor winAtten(DT_FP32, shape_winAtten, "winAtten");

    Tensor kvSlcActSeqsMidOut(DT_INT32, slcActSeqsShape, "kvSlcActSeqsMidOut");
    Tensor attenOut(dType, shape_attentionOut, "attenOut");

    // post: Tensor
    Tensor wUv(dType, wUvShape, "wUv");
    Tensor wo(dTypeQuant, woShape, "wo", weightFormat);
    Tensor postOut(dType, outShape, "postOut");

    // 3. 为输入填充数据
    auto xData = CreateTensorData<T>(x, "/x.bin");
    auto wDqData = CreateTensorData<T>(wDq, "/wDq.bin");
    auto wUqQrData = CreateTensorData<wDtype>(wUqQr, "/wUqQr.bin");
    auto wUkData = CreateTensorData<T>(wUk, "/wUk.bin");
    auto wDkvKrData = CreateTensorData<T>(wDkvKr, "/wDkvKr.bin");
    auto gammaCqData = CreateTensorData<T>(gammaCq, "/gamma_cq.bin");
    auto gammaCkvData = CreateTensorData<T>(gammaCkv, "/gamma_ckv.bin");
    auto cosData = CreateTensorData<T>(cos, "/cos.bin");
    auto sinData = CreateTensorData<T>(sin, "/sin.bin");
    auto kvLenData = CreateTensorData<int64_t>(cacheIndex, "/kv_len.bin");
    auto kvCacheData = CreateTensorData<T>(kvCache, "/kv_cache.bin");
    auto krCacheData = CreateTensorData<T>(krCache, "/kr_cache.bin");
    auto outKvCacheData = CreateTensorData<T>(outputKvCache, "/kv_cache.bin");
    auto outKrCacheData = CreateTensorData<T>(outputKrCache, "/kr_cache.bin");
    // auto outputQData = RawTensorData::CreateConstantTensor<T>(outputQ, 0.0);
    // auto outputQRopeData = RawTensorData::CreateConstantTensor<T>(outputQRope, 0.0);

    auto topkIndicesData = CreateTensorData<int32_t>(topkIndices, "/topk_tensor.bin");
    auto topkTensorShapeData = CreateTensorData<int32_t>(topkTensorShape, "/topk_tensor_shape.bin");
    // auto kvNopeCacheData = CreateTensorData<T>(kvNopeCache, "/kv_nope_cache.bin");
    // auto kRopeCacheData = CreateTensorData<T>(kRopeCache, "/k_rope_cache.bin");
    auto kvCacheActSeqData = CreateTensorData<int32_t>(kvCacheActSeq, "/kv_cache_actual_seq_len.bin");
    auto blockTableData = CreateTensorData<int32_t>(blockTable, "/block_table.bin");

    auto slcActSeqsData = CreateTensorData<int32_t>(slcActSeqs, "/kv_slc_actual_seqs.bin");
    // auto qNopeData = CreateTensorData<T>(qNope, "/q_nope.bin");
    // auto qRopeData = CreateTensorData<T>(qRope, "/q_rope.bin");
    auto kSlcData = CreateTensorData<T>(kSlc, "/k_slc.bin");
    auto vSlcData = CreateTensorData<T>(vSlc, "/v_slc.bin");

    // auto xData = CreateTensorData<T>(x, "/x.bin");
    auto gateW1Data = CreateTensorData<T>(gateW1, "/gate_w1.bin");
    auto gateW2Data = CreateTensorData<T>(gateW2, "/gate_w2.bin");
    auto gateSimW1Data = CreateTensorData<T>(gateSimW1, "/gate_sim_w1.bin");

    auto cmpAttenData = CreateTensorData<T>(cmpAtten, "/cmp_atten.bin");
    // auto selAttenData = CreateTensorData<T>(selAtten, "/sel_atten.bin");
    // auto winAttenData = CreateTensorData<T>(winAtten, "/win_atten.bin");

    // post: data
    auto wUvData = CreateTensorData<T>(wUv, "/w_uv.bin");
    auto woData = CreateTensorData<wDtype>(wo, "/w_o.bin");

    // auto gatingScoreZeroData = RawTensorData::CreateConstantTensor<T>(gatingScore, 0.0);
    auto kvSlcActSeqsMidOutZeroData = RawTensorData::CreateConstantTensor<int32_t>(kvSlcActSeqsMidOut, 0.0);
    auto kSlcZeroData = RawTensorData::CreateConstantTensor<T>(kSlc, 0.0);
    auto vSlcZeroData = RawTensorData::CreateConstantTensor<T>(vSlc, 0.0);
    // auto winAttenData = RawTensorData::CreateConstantTensor<float>(winAtten, 0.0);
    auto slcAttnZeroData = RawTensorData::CreateConstantTensor<float>(slcAttn, 0.0);
    auto attenOutZeroData = RawTensorData::CreateConstantTensor<T>(attenOut, 0.0);
    auto outputData = RawTensorData::CreateConstantTensor<T>(postOut, 0.0);

    // auto qNopeData = RawTensorData::CreateConstantTensor<T>(qNope, 0.0);
    // auto qRopeData = RawTensorData::CreateConstantTensor<T>(qRope, 0.0);

    // MlaProlog output golden
    std::vector<T> golden1 = getGoldenVec<T>(qOutShape, "/q_golden.bin");
    std::vector<T> golden2 = getGoldenVec<T>(qRopeOutShape, "/q_rope_golden.bin");
    std::vector<T> golden3 = getGoldenVec<T>(kvCacheOutShape, "/kv_cache_golden.bin");
    std::vector<T> golden4 = getGoldenVec<T>(krCacheOutShape, "/kr_cache_golden.bin");
    // std::vector<T> gatingScoreGolden = getGoldenVec<T>(gatingScoreShape, "/gating_score.bin");
    std::vector<int32_t> kvSlcActSeqMidOutGolden = getGoldenVec<int32_t>(slcActSeqsShape, "/kv_slc_actual_seqs.bin");
    std::vector<T> kSlcOutGolden = getGoldenVec<T>(kSlcShape, "/kv_slc_out.bin");
    std::vector<T> vSlcOutGolden = getGoldenVec<T>(vSlcShape, "/kr_slc_out.bin");
    std::vector<float> winAttnGolden = getGoldenVec<float>(shape_winAtten, "/winAttn.bin");
    std::vector<float> slcAttnOutGolden = getGoldenVec<float>(shape_selAtten, "/sel_atten.bin");
    std::vector<T> attenOutGolden = getGoldenVec<T>(shape_attentionOut, "/attention_out.bin");
    // Post output golden
    std::vector<T> postGolden = getGoldenVec<T>(outShape, "/golden_output.bin");

    int paramsSize = 10;
    std::vector<int> input_param(paramsSize);
    readInput<int>(GetGoldenDir() + "/input_param_compress.bin", input_param);

    const int b_v2 = input_param[0];
    s1 = input_param[1];
    n1 = input_param[2];
    const int dq = input_param[3];
    s2 = input_param[4];
    n2 = input_param[5];
    const int dv = input_param[6];
    blockSize = input_param[7];
    const int cmpBlockSize = input_param[8];
    const int cmpStride = input_param[9];
    dr = dq - dv;
    softmaxScale = static_cast<float>(1.0 / sqrtf((dq)));

    DataType qType = dType;
    DataType kType = dType;

    // Read actSeqLen_v2
    std::vector<int> actSeq(b_v2);
    readInput<int>(GetGoldenDir() + "/act_seq_compress.bin", actSeq);
    //    int blockNum = 0;
    for (auto s : actSeq) {
        blockNum += CeilDiv(s, blockSize);
    }
    // blockTable_v2: (b_v2, maxBlockNum)
    int maxBlockNum = CeilDiv(s2, blockSize);

    // Read actCmpSeqLen_v2
    std::vector<int> actCmpSeq(b_v2);
    readInput<int>(GetGoldenDir() + "/act_cmp_seq_compress.bin", actCmpSeq);
    int cmpBlockNum = 0;
    for (auto s : actCmpSeq) {
        cmpBlockNum += CeilDiv(s, blockSize);
    }
    // cmpBlockTable_v2: (b_v2, maxCmpBlockNum)
    int maxCmpSeq = *(std::max_element(actCmpSeq.begin(), actCmpSeq.end()));
    int maxCmpBlockNum = CeilDiv(maxCmpSeq, blockSize);

    // Construct input tensors
    Tensor qNope_v2(qType, {b_v2 * s1 * n1, dv}, "qNope_v2");
    Tensor qRope_v2(qType, {b_v2 * s1 * n1, dr}, "qRope_v2");
    Tensor kvCache_v2(kType, {blockNum * blockSize, n2 * dv}, "kvCache_v2");
    Tensor krCache_v2(kType, {blockNum * blockSize, n2 * dr}, "krCache_v2");
    Tensor cmpKvCache_v2(kType, {cmpBlockNum * blockSize, n2 * dv}, "cmpKvCache_v2");
    Tensor cmpKrCache_v2(kType, {cmpBlockNum * blockSize, n2 * dr}, "cmpKrCache_v2");
    Tensor blockTable_v2(DT_INT32, {b_v2, maxBlockNum}, "blockTable_v2");
    Tensor cmpBlockTable_v2(DT_INT32, {b_v2, maxCmpBlockNum}, "cmpBlockTable_v2");
    Tensor actSeqLen_v2(DT_INT32, {b_v2}, "actSeqLen_v2");
    Tensor actCmpSeqLen_v2(DT_INT32, {b_v2}, "actCmpSeqLen_v2");
    Tensor mlpWk1_v2(kType, {cmpBlockSize * dq, 2 * cmpBlockSize * dq}, "mlpWk1_v2");
    Tensor mlpWk2_v2(kType, {2 * cmpBlockSize * dq, dq}, "mlpWk2_v2");
    Tensor mlpCos_v2(kType, {b_v2, cmpBlockSize, dr}, "mlpCos_v2");
    Tensor mlpSin_v2(kType, {b_v2, cmpBlockSize, dr}, "mlpSin_v2");
    Tensor cmpAttn(DT_FP32, {b_v2 * s1 * n1, dv}, "cmpAttnOut");
    Tensor cmpAttn16(DT_FP16, {b, s1, n1, dv}, "cmpAttnOut16");
    Tensor cmpSoftmax(DT_FP32, {b_v2 * s1 * n1, maxCmpSeq}, "cmpSoftmax");
    Tensor fullK(kType, {maxBlockNum * blockSize, n2, dq}, "fullK");
    Tensor cmpK(DT_FP32, {b_v2, maxCmpSeq, n2, dq}, "cmpK");
    Tensor firstRope(qType, {maxCmpSeq, cmpBlockSize, n2, dr}, "firstRope");
    Tensor firstRopeInput(qType, {maxCmpSeq, cmpBlockSize, dr}, "firstRopeInput");
    Tensor topkRes(DT_INT32, {b, s1, 16}, "topkRes");
    int a = (((int((s2 - 32) / 16)) + 1) + 3) / 4;
    std::cout << "xxxxxxxxxxxxxxxxxx  s2:" << s2 << ", a: " << a << std::endl;
    Tensor topkInput(DT_FP32, {b, a}, "topkInput");

    std::vector<uint32_t> topkIndicesGolden = getGoldenVec<uint32_t>({b, s1, 16}, "/topk_full.bin");
    std::vector<float> topkInputGolden = getGoldenVec<float>({b, s1, 128}, "/topk_input.bin");

    // Read goldens
    std::vector<float> attnGolden(b_v2 * s1 * n1 * dv, 0.0);
    std::vector<T> attn16Golden(b * s1 * n1 * dv, 0.0);
    std::vector<float> softmaxGolden(b_v2 * s1 * n1 * maxCmpSeq, 0.0);
    std::vector<T> fullKGolden(maxBlockNum * blockSize * n2 * dq, 0.0);
    std::vector<float> kCmpGolden(b_v2 * maxCmpSeq * n2 * dq, 0.0);
    std::vector<T> firstRopeGolden(maxCmpSeq * cmpBlockSize * dr, 0.0);
    std::vector<T> firstRopeInputGolden(maxCmpSeq * cmpBlockSize * dr, 0.0);

    readInput(GetGoldenDir() + "/cmp_attn_compress.bin", attnGolden);
    readInput(GetGoldenDir() + "/cmp_attn16_compress.bin", attn16Golden);
    readInput(GetGoldenDir() + "/cmp_softmax_compress.bin", softmaxGolden);
    readInput(GetGoldenDir() + "/k_tensor_out_compress.bin", fullKGolden);
    readInput(GetGoldenDir() + "/k_cmp_out_compress.bin", kCmpGolden);
    readInput(GetGoldenDir() + "/first_rope_compress.bin", firstRopeGolden);
    readInput(GetGoldenDir() + "/first_rope_input_compress.bin", firstRopeInputGolden);

    auto qNopeData_v3 = CreateTensorData<T>(qNope_v2, "/q_nope_compress.bin");
    auto qRopeData_v3 = CreateTensorData<T>(qRope_v2, "/q_rope_compress.bin");
    auto kvCacheData_v3 = CreateTensorData<T>(kvCache_v2, "/kv_cache_compress.bin");
    auto krCacheData_v3 = CreateTensorData<T>(krCache_v2, "/kr_cache_compress.bin");
    auto cmpKvCacheData_v3 = CreateTensorData<T>(cmpKvCache_v2, "/cmp_kv_cache_compress.bin");
    auto cmpKrCacheData_v3 = CreateTensorData<T>(cmpKrCache_v2, "/cmp_kr_cache_compress.bin");
    auto blockTableData_v3 = CreateTensorData<int32_t>(blockTable_v2, "/block_table_compress.bin");
    auto cmpBlockTableData_v3 = CreateTensorData<int32_t>(cmpBlockTable_v2, "/cmp_block_table_compress.bin");
    auto actSeqData_v3 = CreateTensorData<int32_t>(actSeqLen_v2, "/act_seq_compress.bin");
    auto actCmpSeqData_v3 = CreateTensorData<int32_t>(actCmpSeqLen_v2, "/act_cmp_seq_compress.bin");

    auto wk1Data_v3 = CreateTensorData<T>(mlpWk1_v2, "/mlp_wk1_compress.bin");
    auto wk2Data_v3 = CreateTensorData<T>(mlpWk2_v2, "/mlp_wk2_compress.bin");
    auto cosData_v3 = CreateTensorData<T>(mlpCos_v2, "/mlp_cos_compress.bin");
    auto sinData_v3 = CreateTensorData<T>(mlpSin_v2, "/mlp_sin_compress.bin");

    auto cmpAttn_v3 = RawTensorData::CreateConstantTensor<float>(cmpAttn, 0.0f);
    auto cmpAttn16_v3 = RawTensorData::CreateConstantTensor<T>(cmpAttn16, 0.0f);
    auto cmpSoftmax_v3 = RawTensorData::CreateConstantTensor<float>(cmpSoftmax, 0.0f);
    auto fullK_v3 = RawTensorData::CreateConstantTensor<T>(fullK, 0.0f);
    auto cmpK_v3 = RawTensorData::CreateConstantTensor<float>(cmpK, 0.0f);
    auto firstRope_v3 = RawTensorData::CreateConstantTensor<T>(firstRope, 0.0f);
    auto firstRopeInput_v3 = RawTensorData::CreateConstantTensor<T>(firstRopeInput, 0.0f);
    //    auto topkRes_v3 = RawTensorData::CreateConstantTensor<float>(topkRes, 0.0f);
    auto topkRes_v3 = RawTensorData::CreateConstantTensor<uint32_t>(topkRes, 0.0f);
    auto topkInputData = RawTensorData::CreateConstantTensor<float>(topkInput, 0.0f);

    std::vector<RawTensorDataPtr> outputDataList = {/*outKvCacheData, outKrCacheData,*/
        outputData, cmpAttn_v3, cmpSoftmax_v3, fullK_v3, cmpK_v3, topkRes_v3, topkInputData};
    std::vector<RawTensorDataPtr> inputDataList = {xData, wDqData, wUqQrData, wUkData, wDkvKrData, gammaCqData,
        gammaCkvData, sinData, cosData, kvLenData, kvCacheData, krCacheData, cmpKvCacheData_v3, cmpKrCacheData_v3,
        blockTableData, cmpBlockTableData_v3, actSeqData_v3, actCmpSeqData_v3, wk1Data_v3, wk2Data_v3, cosData_v3,
        sinData_v3};
    MlaQuantInputs quantInputs;
    if (isQuant) {
        auto wQbScaleData = CreateTensorData<float>(wQbScale, "/w_qb_scale.bin");
        inputDataList.emplace_back(wQbScaleData);
        quantInputs.dequantScaleWUqQr = wQbScale;
        if (isSmooth) {
            auto smoothCqData = CreateTensorData<float>(smoothCq, "/smooth_cq.bin");
            inputDataList.emplace_back(smoothCqData);
            quantInputs.smoothScalesCq = smoothCq;
        }
    } else {
        inputDataList.emplace_back(nullptr); // quantInputs.dequantScaleWUqQr
        inputDataList.emplace_back(nullptr); // quantInputs.smoothScalesCq
    }
    std::vector<RawTensorDataPtr> tmpInputDataList = {
        topkIndicesData, /*kvNopeCacheData, kRopeCacheData,*/ kvCacheActSeqData, // genkvSlc
        /*qNopeData, qRopeData, slcActSeqsData,*/                                                     // slcAtten
        /*xData, */ gateW1Data, gateW2Data, gateSimW1Data,                                            // gatedScore
        cmpAttenData, /*winAttenData,*/                                                               // genAttn
        wUvData, woData,                                                                              // post
    };
    inputDataList.insert(inputDataList.end(), tmpInputDataList.begin(), tmpInputDataList.end());

    QuantTensorWithData wUvQuant{
        false, false, wUvScaleShape, smoothWUvShape, "wUvScale", "smoothWUv", "/w_uv_scale.bin", "/smooth_w_uv.bin"};
    CreateQuantTensorAndData(wUvQuant);
    inputDataList.emplace_back(wUvQuant.scale.dataPtr);
    inputDataList.emplace_back(wUvQuant.smooth.dataPtr);
    QuantTensorWithData wOQuant{
        isQuant, isSmooth, woScaleShape, smoothWoShape, "woScale", "smoothWo", "/w_o_scale.bin", "/smooth_w_o.bin"};
    CreateQuantTensorAndData(wOQuant);
    inputDataList.emplace_back(wOQuant.scale.dataPtr);
    inputDataList.emplace_back(wOQuant.smooth.dataPtr);

    PostTensors postTensors{
        wUv, wo, wUvQuant.scale.tensor, wUvQuant.smooth.tensor, wOQuant.scale.tensor, wOQuant.smooth.tensor};
    // 4. 计算接口
    DynamicNsa(x, wDq, wUqQr, wUk, wDkvKr, gammaCq, gammaCkv, sin, cos, cacheIndex, kvCache, krCache, quantInputs,
        prologConfig, eps, eps, cacheMode, topkIndices, /*kvNopeCache, kRopeCache,*/ kvCacheActSeq,
        blockTable, front, near, topk, slcBlockSize, blockSize,     // genKvSlc
        /*qNope, qRope, slcActSeqs,*/ softmaxScale, saTileConfig,   // slcAttn
        /*x, */ gateW1, gateW2, gateSimW1, GateMode::standard,      // gatedscore
        cmpAtten, winSize, winAttntileConfig,                       // gen win
        postTensors, postConfig,                                    // post
        outputKvCache, outputKrCache, postOut, cmpKvCache_v2, cmpKrCache_v2, cmpBlockTable_v2, actSeqLen_v2,
        actCmpSeqLen_v2, mlpWk1_v2, mlpWk2_v2, mlpCos_v2, mlpSin_v2, cmpAttn, cmpSoftmax, fullK, cmpK, firstRope,
        firstRopeInput, topkRes, topkInput, cmpBlockSize, cmpStride, cmpTileConfig, debug);

#ifdef BUILD_WITH_CANN
    // 5. 更新输入输出list
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction(), inputDataList, outputDataList); // output list
    if constexpr (!ci) {
        std::cout << "MlaProlog kv ====== " << std::endl;
        EXPECT_TRUE(resultCmp<T>(golden3, (T *)outKvCacheData->data(), 0.003f));
        std::cout << "MlaProlog kr ====== " << std::endl;
        EXPECT_TRUE(resultCmp<T>(golden4, (T *)outKrCacheData->data(), 0.003f));

        std::cout << "========================fullKCast==============================" << std::endl;
        EXPECT_TRUE(resultCmp(fullKGolden, (T *)fullK_v3->data(), 0.05f, 16));
        std::cout << "=======================ropeOut===============================" << std::endl;
        EXPECT_TRUE(resultCmp<T>(firstRopeGolden, (T *)firstRope_v3->data(), 0.05f, 16));
        std::cout << "========================kCmpCast==============================" << std::endl;
        EXPECT_TRUE(resultCmp(kCmpGolden, (float *)cmpK_v3->data(), 0.05f, 16));
        std::cout << "=======================softmaxOut===============================" << std::endl;
        EXPECT_TRUE(resultCmp(softmaxGolden, (float *)cmpSoftmax_v3->data(), 0.05f, 16));
        std::cout << "=======================attnOut===============================" << std::endl;
        EXPECT_TRUE(resultCmp(attnGolden, (float *)cmpAttn_v3->data(), 0.05f, 16));
        std::cout << "=======================attn16Out===============================" << std::endl;
        EXPECT_TRUE(resultCmp(attn16Golden, (T *)cmpAttn16_v3->data(), 0.005f, 100));

        std::cout << "=======================topkInput===============================" << std::endl;
        EXPECT_TRUE(resultCmp(
            topkInputGolden, (float *)topkInputData->data(), 0.008f, 0, 16, false, false, topkInputGolden.size()));

        std::cout << "=======================topkRes===============================" << std::endl;
        //    EXPECT_TRUE(resultCmp(topkIndicesGolden, (float *)topkRes_v3->data(), 0.05f, 0, 16, false, false, 20));
        EXPECT_TRUE(resultCmp(
            topkIndicesGolden, (uint32_t *)topkRes_v3->data(), 0.008f, 0, 16, false, false, topkIndicesGolden.size()));

        // std::cout << "attenOut ====== " << std::endl;
        // EXPECT_TRUE(resultCmp<T>(attenOutGolden, (T *)attenOutZeroData->data(), 0.05f));
        std::cout << "post out ====== " << std::endl;
        EXPECT_TRUE(resultCmp<T>(postGolden, (T *)outputData->data(), 0.05f));
    }
    std::cout << "post out ====== print" << std::endl;
    EXPECT_TRUE(resultCmp<T>(postGolden, (T *)outputData->data(), 0.13f, int(0.05 * postGolden.size()), 1000, false, false, 128));
#endif
}

TEST_F(DynamicNSATest, nsa_b_16_s1_1_s2_8192_h_7168_fp16_quant) {
    NSAV1SimpleParams params = NSAV1SimpleParams::getDecodeParams();

    int paramsSize = 7;
    std::vector<int> inputParams(paramsSize);
    readInput<int>(GetGoldenDir() + "/input_params.bin", inputParams); // 在golden中保存了变化的参数，便于调试
    params.b = inputParams[0];                                         // 16
    params.s1 = inputParams[1];
    params.s2 = inputParams[2];
    params.n1 = inputParams[3];
    params.n2 = inputParams[4];
    int isQuant = inputParams[5];
    int isSmooth = inputParams[6];
    std::cout << "===========nsa_b_16_s1_1_s2_8192_h_7168_fp16_quant: isQuant: " << isQuant
              << ", isSmooth: " << isSmooth << std::endl;

    SATileShapeConfig saTileConfig;
    saTileConfig.kvSlcV0TileShape = {64, 256}; // slcBlockSize=64
    const int gTile = 128;  // for gLoop split
    const int sTile = 1024; // for s2Loop split
    saTileConfig.gTile = gTile;
    saTileConfig.sKvTile = sTile;
    saTileConfig.c1TileShape = {gTile, gTile, 64, 64, 128, 128}; // (n1, dn+dr) @ (s2Tile, dn+dr) -> (n1, s2Tile)
    saTileConfig.v1TileShape = {16, 256};                        // (n1, s2Tile)
    saTileConfig.c2TileShape = {gTile, gTile, 64, 64, 128, 128}; // (n1, s2Tile) @ (s2Tile, dn) -> (n1, d)
    saTileConfig.v2TileShape = {16, 256};                        // (n1, d)

    WinAttenTileShapeConfig winAttnTileConfig;
    const int gTileSize = NUM_128; // for gLoop split
    winAttnTileConfig.gTile = gTileSize;
    winAttnTileConfig.vNopeTileShape = {NUM_16, NUM_256};
    winAttnTileConfig.vRopeTileShape = {NUM_128, NUM_64};
    winAttnTileConfig.outTileShape = {NUM_16, NUM_256};
    winAttnTileConfig.c1TileShape = {
        gTileSize, gTileSize, NUM_64, NUM_64, NUM_128, NUM_128}; // (n1, dN+dR) @ (winSize, dN+dR) -> (n1, s2Tile)
    winAttnTileConfig.v1TileShape = {NUM_16, NUM_256};           // (n1, s2Tile)
    winAttnTileConfig.c2TileShape = {
        gTileSize, gTileSize, NUM_128, NUM_128, NUM_128, NUM_128}; // (n1, winSize) @ (winSize, dN) -> (n1, d)
    winAttnTileConfig.v2TileShape = {NUM_16, NUM_256};           // (n1, d)

    int tileB = 16;
    if (params.b == 24) {
        tileB = 24;
    }
    PostTileConfig postConfig = {tileB, 1};
    MlaTileConfig prologConfig = {tileB, 1};

    std::string cacheMode = "PA_BSND";

    CmpAttnTile config;
    // Block concat tile
    config.castTile = {128, 64}; // {blockSize, n2 * d}
    // MlpRope
    config.mlpRopeTile.twoDim = {64, 64};           // (cmpBlockSize, n2*dk)
    config.mlpRopeTile.threeDim = {1, 64, 64};      // (1, cmpBlockSize, dk)
    config.mlpRopeTile.fourDim = {1, 64, 1, 64};    // (1, cmpBlockSize, n2, dK) * (1, cmpBlockSize, 1, dK) & RotateHalf
    config.mlpRopeTile.fiveDim = {1, 64, 1, 64, 2}; // (1, cmpBlockSize, n2, dk / 2, 2)
    // MlpCmp
    config.mlpCmpTile.transTileShape = {32, 1, 192};              // (cmpBlockSize, n2, d)
    config.mlpCmpTile.c1TileShape = {16, 16, 128, 128, 128, 128}; // (n2, 2 * cmpBlockSize * d)
    config.mlpCmpTile.v1TileShape = {1, 128};                     // (n2, 2 * cmpBlockSize * d)
    config.mlpCmpTile.c2TileShape = {16, 16, 128, 128, 128, 128}; // // (n2, d)
    config.mlpCmpTile.v2TileShape = {1, 1, 128};                  // (1, n2, d)
    // CmpAttn
    config.attnTile.c1TileShape = {16, 16, 128, 128, 128, 128}; // (g, effSeq)
    config.attnTile.v1TileShape = {16, 128};                    // (g, effSeq)
    config.attnTile.c2TileShape = {16, 16, 128, 128, 128, 128}; // (g, dN)

    if (isQuant == 1) {
        if (isSmooth == 1) {
            TestNsa<npu::tile_fwk::float16, int8_t, true>(params, prologConfig, winAttnTileConfig, saTileConfig,
                postConfig, config, 0.06f, cacheMode);
        } else {
            TestNsa<npu::tile_fwk::float16, int8_t, false>(params, prologConfig, winAttnTileConfig, saTileConfig,
                postConfig, config, 0.06f, cacheMode);
        }
    } else {
        TestNsa<npu::tile_fwk::float16, npu::tile_fwk::float16, false>(params, prologConfig, winAttnTileConfig,
            saTileConfig, postConfig, config, 0.02f, cacheMode);
    }
}

template <bool ci = false, bool debug = false>
void test_common(NSAV1SimpleParams params) {
    int paramsSize = 7;
    config::SetPassConfig("PVC2_OOO", "SplitReshape", KEY_DISABLE_PASS, true);
    std::vector<int> inputParams(paramsSize);
    readInput<int>(GetGoldenDir() + "/input_params.bin", inputParams); // 在golden中保存了变化的参数，便于调试
    params.b = inputParams[0];                                         // 16
    params.s1 = inputParams[1];
    params.s2 = inputParams[2];
    params.n1 = inputParams[3];
    params.n2 = inputParams[4];
    int isQuant = inputParams[5];
    int isSmooth = inputParams[6];
    std::cout << "===========nsa_b_16_s1_1_s2_8192_h_7168_fp16: isQuant: " << isQuant << ", isSmooth: " << isSmooth
              << std::endl;

    SATileShapeConfig saTileConfig;
    saTileConfig.kvSlcV0TileShape = {64, 256}; // slcBlockSize=64
    const int gTile = 128;  // for gLoop split
    const int sTile = 1024; // for s2Loop split
    saTileConfig.gTile = gTile;
    saTileConfig.sKvTile = sTile;
    saTileConfig.c1TileShape = {gTile, gTile, 64, 64, 128, 128}; // (n1, dn+dr) @ (s2Tile, dn+dr) -> (n1, s2Tile)
    saTileConfig.v1TileShape = {16, 256};                        // (n1, s2Tile)
    saTileConfig.c2TileShape = {gTile, gTile, 64, 64, 128, 128}; // (n1, s2Tile) @ (s2Tile, dn) -> (n1, d)
    saTileConfig.v2TileShape = {16, 256};                        // (n1, d)

    WinAttenTileShapeConfig winAttnTileConfig;
    const int gTileSize = NUM_128; // for gLoop split
    winAttnTileConfig.gTile = gTileSize;
    winAttnTileConfig.vNopeTileShape = {NUM_16, NUM_256};
    winAttnTileConfig.vRopeTileShape = {NUM_128, NUM_64};
    winAttnTileConfig.outTileShape = {NUM_16, NUM_256};
    winAttnTileConfig.c1TileShape = {
        gTileSize, gTileSize, NUM_64, NUM_64, NUM_128, NUM_128}; // (n1, dN+dR) @ (winSize, dN+dR) -> (n1, s2Tile)
    winAttnTileConfig.v1TileShape = {NUM_16, NUM_256};           // (n1, s2Tile)
    winAttnTileConfig.c2TileShape = {
        gTileSize, gTileSize, NUM_128, NUM_128, NUM_128, NUM_128}; // (n1, winSize) @ (winSize, dN) -> (n1, d)
    winAttnTileConfig.v2TileShape = {NUM_16, NUM_256};           // (n1, d)

    int tileB = 16;
    if (params.b == 24) {
        tileB = 24;
    }
    PostTileConfig postConfig = {tileB, 1};
    MlaTileConfig prologConfig = {tileB, 1};

    CmpAttnTile config;
    // Block concat tile
    config.castTile = {128, 64}; // {blockSize, n2 * d}
    // MlpRope
    config.mlpRopeTile.twoDim = {64, 64};           // (cmpBlockSize, n2*dk)
    config.mlpRopeTile.threeDim = {1, 64, 64};      // (1, cmpBlockSize, dk)
    config.mlpRopeTile.fourDim = {1, 64, 1, 64};    // (1, cmpBlockSize, n2, dK) * (1, cmpBlockSize, 1, dK) & RotateHalf
    config.mlpRopeTile.fiveDim = {1, 64, 1, 64, 2}; // (1, cmpBlockSize, n2, dk / 2, 2)
    // MlpCmp
    config.mlpCmpTile.transTileShape = {32, 1, 192};              // (cmpBlockSize, n2, d)
    config.mlpCmpTile.c1TileShape = {16, 16, 128, 128, 128, 128}; // (n2, 2 * cmpBlockSize * d)
    config.mlpCmpTile.v1TileShape = {1, 128};                     // (n2, 2 * cmpBlockSize * d)
    config.mlpCmpTile.c2TileShape = {16, 16, 128, 128, 128, 128}; // // (n2, d)
    config.mlpCmpTile.v2TileShape = {1, 1, 128};                  // (1, n2, d)
    // CmpAttn
    config.attnTile.c1TileShape = {16, 16, 128, 128, 128, 128}; // (g, effSeq)
    config.attnTile.v1TileShape = {16, 128};                    // (g, effSeq)
    config.attnTile.c2TileShape = {16, 16, 128, 128, 128, 128}; // (g, dN)

    std::string cacheMode = "PA_BSND";
    if (isQuant == 1) {
        if (isSmooth == 1) {
            TestNsa<npu::tile_fwk::float16, int8_t, true, false, ci, debug>(params, prologConfig, winAttnTileConfig,
                saTileConfig, postConfig, config, 0.06f, cacheMode);
        } else {
            TestNsa<npu::tile_fwk::float16, int8_t, false, false, ci, debug>(params, prologConfig, winAttnTileConfig,
                saTileConfig, postConfig, config, 0.06f, cacheMode);
        }
    } else {
        TestNsa<npu::tile_fwk::float16, npu::tile_fwk::float16, false, false, ci, debug>(params, prologConfig,
            winAttnTileConfig, saTileConfig, postConfig, config, 0.02f, cacheMode);
    }
}

TEST_F(DynamicNSATest, nsa_b_16_s1_1_s2_8192_h_7168_fp16) {
    NSAV1SimpleParams params = NSAV1SimpleParams::getDecodeParams();
    test_common(params);
}

TEST_F(DynamicNSATest, s2_1024) {
    NSAV1SimpleParams params = NSAV1SimpleParams::getDecodeParams();
    test_common(params);
}

TEST_F(DynamicNSATest, s2_2048) {
    NSAV1SimpleParams params = NSAV1SimpleParams::getDecodeParams();
    test_common(params);
}

TEST_F(DynamicNSATest, s2_8192) {
    NSAV1SimpleParams params = NSAV1SimpleParams::getDecodeParams();
    test_common(params);
}

TEST_F(DynamicNSATest, s2_4096) {
    NSAV1SimpleParams params = NSAV1SimpleParams::getDecodeParams();
    test_common(params);
}

TEST_F(DynamicNSATest, mini) {
    NSAV1SimpleParams params = NSAV1SimpleParams::getDecodeParams();
    test_common<true>(params);
}

TEST_F(DynamicNSATest, mini_debug) {
    NSAV1SimpleParams params = NSAV1SimpleParams::getDecodeParams();
    test_common<true, true>(params);
}
