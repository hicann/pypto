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
 * \file test_onboard_mla_prolog_v2.cpp
 * \brief
 */

#include "test_suite_stest_ops.h"
#include "operator/models/deepseek/mla_prolog.h"
#include "test_dev_func_runner.h"

using namespace npu::tile_fwk;

class MlaPrologV2OnBoardTest : public npu::tile_fwk::stest::TestSuite_STest_Ops_Aihac {};

template <typename T = npu::tile_fwk::float16, bool splitReduceLastDim = true, bool splitK = false, bool nz = false, bool usePrefetch = false>
void TestMlaPrologV2(std::vector<int> &params, string dataPath, bool isQuant = false, bool hasSmooth = false,
        int blockSize = 128, std::string cacheMode = "BNSD") {
    // b, s, s2, n, h, qLoraRank, qkNopeHeadDim, qkRopeHeadDim, kvLoraRank, vHeadDim
    int b = params[0];
    int s = params[1];
    int s2 = params[2];
    int n = params[3];
    int h = params[4];
    int qLoraRank = params[5];
    int qkNopeHeadDim = params[6];
    int qkRopeHeadDim = params[7];
    int kvLoraRank = params[8];
    int q_head_dim = qkNopeHeadDim + qkRopeHeadDim;

    DataType dType = DataType::DT_FP32;
    if (std::is_same<T, npu::tile_fwk::float16>::value) {
        dType = DataType::DT_FP16;
    } else if (std::is_same<T, npu::tile_fwk::bfloat16>::value) {
        dType = DataType::DT_BF16;
    } else {
        dType = DataType::DT_FP32;
    }

    DataType dTypeQuantIn = isQuant ? DataType::DT_INT8 : dType;
    // typedef float outDtype;
    typedef T outDtype;
    typedef int8_t wDtype;

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
    // output
    std::vector<int64_t> q_out_shape = {b, s, n, kvLoraRank};
    std::vector<int64_t> q_rope_out_shape = {b, s, n, qkRopeHeadDim};
    if (cacheMode == "PA_BSND") {
        int blockNum = b * (s2 / blockSize);
        kv_cache_shape = {blockNum, blockSize, 1, kvLoraRank};
        kr_cache_shape = {blockNum, blockSize, 1, qkRopeHeadDim};
    }

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
    // output
    int capacity_q_out = std::accumulate(q_out_shape.begin(), q_out_shape.end(), 1, std::multiplies<>());
    int capacity_q_rope_out = std::accumulate(q_rope_out_shape.begin(), q_rope_out_shape.end(), 1, std::multiplies<>());
    int capacity_kv_out = std::accumulate(kv_cache_shape.begin(), kv_cache_shape.end(), 1, std::multiplies<>());
    int capacity_kr_out = std::accumulate(kr_cache_shape.begin(), kr_cache_shape.end(), 1, std::multiplies<>());

    std::vector<int64_t> w_qb_scale_shape = {1, n * q_head_dim};
    int capacity_w_qb_scale = std::accumulate(w_qb_scale_shape.begin(), w_qb_scale_shape.end(), 1, std::multiplies<>());

    std::vector<int64_t> smooth_cq_shape = {1, qLoraRank};
    int capacity_smooth_cq = std::accumulate(smooth_cq_shape.begin(), smooth_cq_shape.end(), 1, std::multiplies<>());

    aclInit(nullptr);
    rtSetDevice(GetDeviceIdByEnvVar());
    uint64_t outputSize0 = capacity_q_out * sizeof(T);
    uint64_t outputSize1 = capacity_q_rope_out * sizeof(T);
    uint64_t outputSize2 = capacity_kv_out * sizeof(T);
    uint64_t outputSize3 = capacity_kr_out * sizeof(T);
    uint8_t* q_out_ptr = allocDevAddr(outputSize0);
    uint8_t* q_rope_out_ptr = allocDevAddr(outputSize1);
    void *kv_cache_ptr = readToDev<T>(dataPath + "/kv_cache.bin", capacity_kv_cache);
    void *kr_cache_ptr = readToDev<T>(dataPath + "/kr_cache.bin", capacity_kr_cache);


    ConfigManager::Instance();
    PROGRAM("MlaProlog") {
        void *x_ptr = readToDev<T>(dataPath + "/x.bin", capacity_x);
    void *wDqPtr = readToDev<T>(dataPath + "/wDq.bin", wDqCapacity);
    void *wUqQrPtr = isQuant ? readToDev<wDtype>(dataPath + "/wUqQr.bin", wUqQrCapacity) :
                               readToDev<T>(dataPath + "/wUqQr.bin", wUqQrCapacity);
    void *wDkvKrPtr = readToDev<T>(dataPath + "/wDkvKr.bin", wDkvKrCapacity);
    void *wUkPtr = readToDev<T>(dataPath + "/wUk.bin", wUkCapacity);
        void *gamma_cq_ptr = readToDev<T>(dataPath + "/gamma_cq.bin", capacity_gamma_cq);
        void *gamma_ckv_ptr = readToDev<T>(dataPath + "/gamma_ckv.bin", capacity_gamma_ckv);
        void *cos_ptr = readToDev<T>(dataPath + "/cos.bin", capacity_cos);
        void *sin_ptr = readToDev<T>(dataPath + "/sin.bin", capacity_cos);
        void *kv_len_ptr = readToDev<int64_t>(dataPath + "/kv_len.bin", capacity_kv_len);  // int64

        Tensor x(dType, x_shape, (uint8_t *)x_ptr, "x");
        TileOpFormat weightFormat = nz ? TileOpFormat::TILEOP_NZ : TileOpFormat::TILEOP_ND;
        Tensor wDq(dType, w_qa_shape, (uint8_t *)wDqPtr, "wDq", weightFormat);
        Tensor wUqQr(dTypeQuantIn, w_qb_shape, (uint8_t *)wUqQrPtr, "wUqQr", weightFormat);
        if constexpr (usePrefetch) {
            wDq.SetCachePolicy(CachePolicy::PREFETCH, true);
            wUqQr.SetCachePolicy(CachePolicy::PREFETCH, true);
        }
        Tensor wDkvKr(dType, w_kv_a_shape, (uint8_t *)wDkvKrPtr, "wDkvKr", weightFormat);
        Tensor wUk(dType, w_kv_b_k_shape, (uint8_t *)wUkPtr, "wUk", weightFormat);
        Tensor gammaCq(dType, gamma_cq_shape, (uint8_t *)gamma_cq_ptr, "gammaCq");
        Tensor gammaCkv(dType, gamma_ckv_shape, (uint8_t *)gamma_ckv_ptr, "gammaCkv");
        Tensor cos(dType, cos_shape, (uint8_t *)cos_ptr, "cos");
        Tensor sin(dType, cos_shape, (uint8_t *)sin_ptr, "sin");
        Tensor kv_len(DT_INT64, kv_len_shape, (uint8_t *)kv_len_ptr, "kv_len");  // int64
        Tensor kv_cache(dType, kv_cache_shape, (uint8_t *)kv_cache_ptr, "kv_cache");
        Tensor kr_cache(dType, kr_cache_shape, (uint8_t *)kr_cache_ptr, "kr_cache");
        // output
        Tensor output_q(dType, q_out_shape, q_out_ptr, "output_q");
        Tensor output_q_rope(dType, q_rope_out_shape, q_rope_out_ptr, "output_q_rope");

        RoPETileShapeConfigNew ropeConfig{
            {b, 1, 64}, // (b,s,d)
            {b, 1, 1, 64}, // Q (b,s,n,d)
            {b, 1, 1, 64}, // K (b,s,1,d)
            {b, 1, 1, 32, 2} // (b,s,n,d//2,2)
        };

        MlaQuantInputs quantInputs;

        if (isQuant) {
            void *w_qb_scale_ptr = readToDev<float>(dataPath + "/w_qb_scale.bin", capacity_w_qb_scale);
            Tensor w_qb_scale = Tensor(DataType::DT_FP32, w_qb_scale_shape, (uint8_t *)w_qb_scale_ptr, "w_qb_scale");
            quantInputs.dequantScaleWUqQr = w_qb_scale;
            void *smooth_cq_ptr = readToDev<float>(dataPath + "/smooth_cq.bin", capacity_smooth_cq);
            Tensor smooth_cq = Tensor(DT_FP32, smooth_cq_shape, (uint8_t *)smooth_cq_ptr, "smooth_cq");
            if (hasSmooth) {
                quantInputs.smoothScalesCq = smooth_cq;
                smooth_cq.SetCachePolicy(CachePolicy::PREFETCH, true);
            }
            config::SetBuildStatic(true);
            FUNCTION("MlaProlog_T", {x, wDq, wUqQr, w_qb_scale, smooth_cq, wUk, wDkvKr, gammaCq, gammaCkv, sin, cos, kv_len, kv_cache, kr_cache, output_q, output_q_rope}) {
                MlaProlog(x, wDq, wUqQr, wUk, wDkvKr, gammaCq, gammaCkv, sin, cos, kv_len, kv_cache, kr_cache,
                    quantInputs, ropeConfig, output_q, output_q_rope, kv_cache, kr_cache, 1e-5f, 1e-5f, cacheMode, splitReduceLastDim,  splitK);
            };
        } else {
            config::SetBuildStatic(true);
            FUNCTION("MlaProlog_T", {x, wDq, wUqQr, wUk, wDkvKr, gammaCq, gammaCkv, sin, cos, kv_len, kv_cache, kr_cache, output_q, output_q_rope}) {
                MlaProlog(x, wDq, wUqQr, wUk, wDkvKr, gammaCq, gammaCkv, sin, cos, kv_len, kv_cache, kr_cache,
                    quantInputs, ropeConfig, output_q, output_q_rope, kv_cache, kr_cache, 1e-5f, 1e-5f, cacheMode, splitReduceLastDim,  splitK);
            };
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());

    std::vector<outDtype> q_golden(capacity_q_out);
    std::vector<outDtype> q_npu(capacity_q_out);
    std::vector<outDtype> q_rope_golden(capacity_q_rope_out);
    std::vector<outDtype> q_rope_npu(capacity_q_rope_out);
    std::vector<T> kv_golden(capacity_kv_out);
    std::vector<T> kv_npu(capacity_kv_out);
    std::vector<T> kr_golden(capacity_kr_out);
    std::vector<T> kr_npu(capacity_kr_out);

    readInput<outDtype>(dataPath + "/q_golden.bin", q_golden);
    readInput<outDtype>(dataPath + "/q_rope_golden.bin", q_rope_golden);
    readInput<T>(dataPath + "/kv_cache_golden.bin", kv_golden);
    readInput<T>(dataPath + "/kr_cache_golden.bin", kr_golden);
    machine::GetRA()->CopyFromTensor((uint8_t *)q_npu.data(), (uint8_t *)q_out_ptr, outputSize0);
    machine::GetRA()->CopyFromTensor((uint8_t *)q_rope_npu.data(), (uint8_t *)q_rope_out_ptr, outputSize1);
    machine::GetRA()->CopyFromTensor((uint8_t *)kv_npu.data(), (uint8_t *)kv_cache_ptr, outputSize2);
    machine::GetRA()->CopyFromTensor((uint8_t *)kr_npu.data(), (uint8_t *)kr_cache_ptr, outputSize3);

    std::cout << "\n====== resultCmp: output q start" << std::endl;
    int ret0 = resultCmp<outDtype>(q_golden, q_npu, 0.008f);
    EXPECT_EQ(ret0, true);

    std::cout << "\n====== resultCmp: output q_rope start" << std::endl;
    int ret1 = resultCmp<outDtype>(q_rope_golden, q_rope_npu, 0.005f);
    EXPECT_EQ(ret1, true);

    std::cout << "\n====== resultCmp: output kv_cache start" << std::endl;
    int ret2 = resultCmp<T>(kv_golden, kv_npu, 0.003f);
    EXPECT_EQ(ret2, true);

    std::cout << "\n====== resultCmp: output kr_cache start" << std::endl;
    int ret3 = resultCmp<T>(kr_golden, kr_npu, 0.003f);
    EXPECT_EQ(ret3, true);
}

//using IfaTestParam = std::unordered_map<std::string, int>;
//
//static IfaTestParam hightThroughputParams = {
//    {"b", 32},
//    {"nq", 128},
//    {"s2", 4096},
//    {"block_size", 256}
//};

//static IfaTileShapeConfig hightThroughputTileParams {
//    256, // block size
//    32,  // nTile
//    {128, 128}, // v1 tile
//    {32, 32, 64, 64, 128, 128}, // c1 tile
//    {128, 128}, // v1 tile
//    {32, 32, 128, 128, 128, 128}, // c2 tile
//    {128, 128}, // v2 tile
//};

void readAttentionBlockTableFromFile(const std::string& filename, int rows, int cols, std::vector<std::vector<int>> & blockTable) {
    std::ifstream inFile(filename, std::ios::binary);
    if (!inFile) {
        std::cerr << "Error opening file for reading!" << std::endl;
        return;
    }

    for (int i = 0; i < rows; ++i) {
        inFile.read(reinterpret_cast<char*>(blockTable[i].data()), cols * sizeof(int));
    }

    inFile.close();
    return;
}

Tensor CastTranspoeReshape(const Tensor &input, bool isTranspose, std::vector<int64_t>& shape, std::vector<int64_t>& tileShape1, std::vector<int64_t>& tileShape2) {
    TileShape::Current().SetVecTile(tileShape1);
    auto input32 = Cast(input, DT_FP32);
    auto transTensor = isTranspose ? Transpose(input32, {1, 2}) : input32;
    TileShape::Current().SetVecTile(tileShape2);
    auto bfTensor = Cast(transTensor, DT_BF16);
    return Reshape(bfTensor, shape);
}

template <typename T = npu::tile_fwk::bfloat16, bool DEBUG = false>
void TestAttentionV2(std::vector<int> &params, string dataPath, IfaTileShapeConfig tileConfig, bool isQuant = true) {
    // b, s, s2, n, h, q_lora_rank, qk_nope_head_dim, qk_rope_head_dim, kv_lora_rank, v_head_dim
    int b = params[0];
    int s = params[1];
    int s2 = params[2];
    int n = params[3];
    int h = params[4];
    int q_lora_rank = params[5];
    int qk_nope_head_dim = params[6];
    int qk_rope_head_dim = params[7];
    int kv_lora_rank = params[8];
    int v_head_dim = params[9];
    int q_head_dim = qk_nope_head_dim + qk_rope_head_dim;

    const int blockSize = tileConfig.blockSize;
    std::vector<int> actSeqs(b, s2);
    const float softmaxScale = 0.8f;

    // 输出size
    int outCap = b * 1 * n * kv_lora_rank;
    uint64_t outputSize = outCap * sizeof(float);
    uint8_t *outPtr = allocDevAddr(outputSize);

    // 根据Per Batch实际的sequence构造blockNum，blockNum >= Sum(blockNumPerBatch)，此处选取相等场景
    int blockNum = 0;
    for (auto actSeq : actSeqs) {
        blockNum += CeilDiv(actSeq, blockSize);
    }

    typedef int8_t T_INT8;
    typedef float T_FLOAT;

    DataType dType = DT_FP32;
    if (std::is_same<T, npu::tile_fwk::float16>::value) {
        dType = DT_FP16;
    } else if (std::is_same<T, npu::tile_fwk::bfloat16>::value) {
        dType = DT_BF16;
    } else {
        dType = DT_FP32;
    }

    DataType dTypeQuantIn = isQuant ? DT_INT8 : dType;
    // typedef float outDtype;
    typedef T outDtype;
    typedef int8_t wDtype;

    std::vector<int64_t> x_shape = {b, s, h};
    std::vector<int64_t> w_qa_shape = {h, q_lora_rank};
    std::vector<int64_t> w_qb_shape = {q_lora_rank, n * q_head_dim};
    std::vector<int64_t> w_kv_a_shape = {h, kv_lora_rank + qk_rope_head_dim};
    std::vector<int64_t> w_kv_b_k_shape = {n, qk_nope_head_dim, kv_lora_rank};
    std::vector<int64_t> cos_shape = {b, s, qk_rope_head_dim};
    std::vector<int64_t> gamma_cq_shape = {q_lora_rank};
    std::vector<int64_t> gamma_ckv_shape = {kv_lora_rank};
    std::vector<int64_t> kv_len_shape = {b, s};
    std::vector<int64_t> kv_cache_shape = {b, 1, s2, kv_lora_rank};
    std::vector<int64_t> kr_cache_shape = {b, 1, s2, qk_rope_head_dim};
    // output
    std::vector<int64_t> q_out_shape = {b, s, n, kv_lora_rank};
    std::vector<int64_t> q_rope_out_shape = {b, s, n, qk_rope_head_dim};
    std::vector<int64_t> kv_cache_out_shape = {b, 1, s2, kv_lora_rank};
    std::vector<int64_t> kr_cache_out_shape = {b, 1, s2, qk_rope_head_dim};

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
    // output
    int capacity_q_out = std::accumulate(q_out_shape.begin(), q_out_shape.end(), 1, std::multiplies<>());
    int capacity_q_rope_out = std::accumulate(q_rope_out_shape.begin(), q_rope_out_shape.end(), 1, std::multiplies<>());
    int capacity_kv_out = std::accumulate(kv_cache_out_shape.begin(), kv_cache_out_shape.end(), 1, std::multiplies<>());
    int capacity_kr_out = std::accumulate(kr_cache_out_shape.begin(), kr_cache_out_shape.end(), 1, std::multiplies<>());

    std::vector<int64_t> w_qb_scale_shape;
    int capacity_w_qb_scale;
    if (isQuant) {
        w_qb_scale_shape = {1, n * q_head_dim};
        capacity_w_qb_scale = std::accumulate(w_qb_scale_shape.begin(), w_qb_scale_shape.end(), 1, std::multiplies<>());
    }

    aclInit(nullptr);
    rtSetDevice(GetDeviceIdByEnvVar());
    uint64_t outputSize0 = capacity_q_out * sizeof(T);
    uint64_t outputSize1 = capacity_q_rope_out * sizeof(T);
    uint64_t outputSize2 = capacity_kv_out * sizeof(T);
    uint64_t outputSize3 = capacity_kr_out * sizeof(T);
    uint8_t *q_out_ptr = allocDevAddr(outputSize0);
    uint8_t *q_rope_out_ptr = allocDevAddr(outputSize1);
    void *kv_cache_ptr = readToDev<T>(dataPath + "/kv_cache.bin", capacity_kv_cache);
    void *kr_cache_ptr = readToDev<T>(dataPath + "/kr_cache.bin", capacity_kr_cache);

    uint8_t *q_nope_bnsd_ptr = allocDevAddr(outputSize0);
    uint8_t *q_rope_bnsd_ptr = allocDevAddr(outputSize1);
    uint8_t *k_nope_bnsd_ptr = allocDevAddr(capacity_kv_cache * sizeof(T));
    uint8_t *k_rope_bnsd_ptr = allocDevAddr(capacity_kr_cache * sizeof(T));
    uint8_t *v_nope_bnsd_ptr = allocDevAddr(capacity_kv_cache * sizeof(T));


    DataType dTypeInt8 = DataType::DT_INT8;
    DataType dTypeFp32 = DataType::DT_FP32;

    int dtypeSize = BytesOf(dType);

    int inputSize = b * n * s * kv_lora_rank;
    int wUvSize = n * kv_lora_rank * v_head_dim;
    int wUvScaleWSize = n * 1 * v_head_dim;
    int wOSize = n * v_head_dim * h;
    int wOScaleWSize = 1 * h;
    int postOutputSize = b * s * h;
    int t1Size = b * s * n * kv_lora_rank;

    uint64_t outputByteSize = postOutputSize * dtypeSize;
    uint8_t *out_ptr = allocDevAddr(outputByteSize);

    ConfigManager::Instance();
    PROGRAM("MlaProlog") {
        void *x_ptr = readToDev<T>(dataPath + "/x.bin", capacity_x);
        void *wDqPtr = readToDev<T>(dataPath + "/wDq.bin", wDqCapacity);
        void *wUqQrPtr = isQuant ? readToDev<wDtype>(dataPath + "/wUqQr.bin", wUqQrCapacity) :
                                   readToDev<T>(dataPath + "/wUqQr.bin", wUqQrCapacity);
        void *wDkvKrPtr = readToDev<T>(dataPath + "/wDkvKr.bin", wDkvKrCapacity);
        void *wUkPtr = readToDev<T>(dataPath + "/wUk.bin", wUkCapacity);
        void *gamma_cq_ptr = readToDev<T>(dataPath + "/gamma_cq.bin", capacity_gamma_cq);
        void *gamma_ckv_ptr = readToDev<T>(dataPath + "/gamma_ckv.bin", capacity_gamma_ckv);
        void *cos_ptr = readToDev<T>(dataPath + "/cos.bin", capacity_cos);
        void *sin_ptr = readToDev<T>(dataPath + "/sin.bin", capacity_cos);
        void *kv_len_ptr = readToDev<int64_t>(dataPath + "/kv_len.bin", capacity_kv_len); // int64

        Tensor x(dType, x_shape, (uint8_t *)x_ptr, "x");
        TileOpFormat weightFormat = TileOpFormat::TILEOP_ND;
        Tensor wDq(dType, w_qa_shape, (uint8_t *)wDqPtr, "wDq", weightFormat);
        Tensor wUqQr(dTypeQuantIn, w_qb_shape, (uint8_t *)wUqQrPtr, "wUqQr", weightFormat);
        Tensor wDkvKr(dType, w_kv_a_shape, (uint8_t *)wDkvKrPtr, "wDkvKr", weightFormat);
        Tensor wUk(dType, w_kv_b_k_shape, (uint8_t *)wUkPtr, "wUk", weightFormat);
        Tensor gamma_cq(dType, gamma_cq_shape, (uint8_t *)gamma_cq_ptr, "gamma_cq");
        Tensor gamma_ckv(dType, gamma_ckv_shape, (uint8_t *)gamma_ckv_ptr, "gamma_ckv");
        Tensor cos(dType, cos_shape, (uint8_t *)cos_ptr, "cos");
        Tensor sin(dType, cos_shape, (uint8_t *)sin_ptr, "sin");
        Tensor kv_len(DT_INT64, kv_len_shape, (uint8_t *)kv_len_ptr, "kv_len"); // int64
        Tensor kv_cache(dType, kv_cache_shape, (uint8_t *)kv_cache_ptr, "kv_cache");
        Tensor kr_cache(dType, kr_cache_shape, (uint8_t *)kr_cache_ptr, "kr_cache");
        // output
        //        Tensor output_q(dType, q_out_shape, q_out_ptr, "output_q");
        Tensor output_q_rope(dType, q_rope_out_shape, q_rope_out_ptr, "output_q_rope");

        RoPETileShapeConfigNew ropeConfig{
            {32, 1, 64}, // (b,s,d)
            {1, 1, 32, 64}, // Q (b,s,n,d)
            {32, 1, 1, 64}, // K (b,s,1,d)
            {32, 1, 1, 32, 2}  // (b,s,n,d//2,2)
        };

        MlaQuantInputs quantInputs;

        // 读数据
        void *qNopeData = readToDev<npu::tile_fwk::bfloat16>(GetGoldenDir() + "/q_nope.bin", outputSize0);
        void *qRopeData = readToDev<npu::tile_fwk::bfloat16>(GetGoldenDir() + "/q_rope.bin", outputSize1);
        void *kNopeCacheData = readToDev<npu::tile_fwk::bfloat16>(GetGoldenDir() + "/k_cache_nope.bin", capacity_kv_cache);
        void *kRopeCacheData = readToDev<npu::tile_fwk::bfloat16>(GetGoldenDir() + "/k_cache_rope.bin", capacity_kr_cache);
        void *vNopeCacheData = readToDev<npu::tile_fwk::bfloat16>(GetGoldenDir() + "/v_cache.bin", capacity_kv_cache);

        std::vector<int64_t> qNopeShape = {b * s * n, kv_lora_rank};
        std::vector<int64_t> qRopeShape = {b * s * n, qk_rope_head_dim};
        std::vector<int64_t> kNopeShape = {blockNum * blockSize * 1, kv_lora_rank};
        std::vector<int64_t> kRopeShape = {blockNum * blockSize * 1, qk_rope_head_dim};
        std::vector<int64_t> vNopeShape = {blockNum * blockSize * 1, kv_lora_rank};

        Tensor qNope(DT_BF16, qNopeShape, (uint8_t *)qNopeData, "qNope");
        Tensor qRope(DT_BF16, qRopeShape, (uint8_t *)qRopeData, "qRope");
        Tensor kNopeCache(DT_BF16, kNopeShape, (uint8_t *)kNopeCacheData, "kNopeCache");
        Tensor kRopeCache(DT_BF16, kRopeShape, (uint8_t *)kRopeCacheData, "kRope");
        Tensor vNopeCache(DT_BF16, vNopeShape, (uint8_t *)vNopeCacheData, "vNopeCache");

        // blockTable: (b, maxBlockNumPerBatch)
        int maxSeqAllBatch = *(std::max_element(actSeqs.begin(), actSeqs.end()));
        int maxBlockNumPerBatch = CeilDiv(maxSeqAllBatch, blockSize);
        std::vector<std::vector<int>> blockTable(b, std::vector<int>(maxBlockNumPerBatch, 0));
        readAttentionBlockTableFromFile(GetGoldenDir() + "/block_table.bin", b, maxBlockNumPerBatch, blockTable);

        for (const auto &row : blockTable) {
            for (int num : row) {
                std::cout << num << " ";
            }
            std::cout << std::endl;
        }

        Tensor attentionOut(DT_FP32, {b * s * n, kv_lora_rank}, outPtr, "attentionOut");

        std::vector<int64_t> inputShape = {b, n, s, kv_lora_rank};
        std::vector<int64_t> wUvShape = {n, kv_lora_rank, v_head_dim};
        std::vector<int64_t> wUvScaleWShape = {n, 1, v_head_dim};
        std::vector<int64_t> wOShape = {n * v_head_dim, h};
        std::vector<int64_t> wOScaleWShape = {1, h};
        std::vector<int64_t> outputShapeT = {b, s, h};
        std::vector<int64_t> t1Shape = {b, s, n, kv_lora_rank};
        void *input_ptr = readToDev<T>(GetGoldenDir() + "/input.bin", inputSize);
        void *w_uv_ptr = readToDev<T>(GetGoldenDir() + "/w_uv.bin", wUvSize);
        void *w_uv_scale_w_ptr = readToDev<T_FLOAT>(GetGoldenDir() + "/w_uv_scale_w.bin", wUvScaleWSize);
        void *w_o_ptr = readToDev<T_INT8>(GetGoldenDir() + "/w_o.bin", wOSize);
        void *w_o_scale_w_ptr = readToDev<T_FLOAT>(GetGoldenDir() + "/w_o_scale_w.bin", wOScaleWSize);

        void *t1_ptr = readToDev<T>(GetGoldenDir() + "/t1.bin", t1Size);

        Tensor input_i(dType, inputShape, (uint8_t *)input_ptr, "A");
        Tensor w_uv_i(dType, wUvShape, (uint8_t *)w_uv_ptr, "B");
        Tensor w_uv_scale_w_i(dTypeFp32, wUvScaleWShape, (uint8_t *)w_uv_scale_w_ptr, "B1");
        Tensor w_o_i(dTypeInt8, wOShape, (uint8_t *)w_o_ptr, "C");
        Tensor w_o_scale_w_i(dTypeFp32, wOScaleWShape, (uint8_t *)w_o_scale_w_ptr, "C1");
        Tensor outputT(dType, outputShapeT, out_ptr, "D1");
        Tensor t1_i(dType, t1Shape, (uint8_t *)t1_ptr, "E");

        if (isQuant) {
            void *w_qb_scale_ptr = readToDev<float>(dataPath + "/w_qb_scale.bin", capacity_w_qb_scale);
            Tensor w_qb_scale = Tensor(DT_FP32, w_qb_scale_shape, (uint8_t *)w_qb_scale_ptr, "w_qb_scale");
            quantInputs.dequantScaleWUqQr = w_qb_scale;

            config::SetBuildStatic(true);
            FUNCTION("MlaProlog_T",
                {x, wDq, wUqQr, w_qb_scale, wUk, wDkvKr, gamma_cq, gamma_ckv, sin, cos, kv_len, kv_cache, kr_cache,
                    output_q_rope, qNope, kNopeCache, vNopeCache, qRope, kRopeCache, attentionOut, input_i, w_uv_i,
                    w_uv_scale_w_i, w_o_i, w_o_scale_w_i, outputT}) {
                Tensor output_q(dType, q_out_shape, q_out_ptr, "output_q");
                Tensor q_nope_bnsd(DT_BF16, qNopeShape, (uint8_t *)q_nope_bnsd_ptr, "q_nope_bnsd");
                Tensor q_rope_bnsd(DT_BF16, qRopeShape, (uint8_t *)q_rope_bnsd_ptr, "q_rope_bnsd");
                Tensor k_nope_bnsd(DT_BF16, kNopeShape, (uint8_t *)k_nope_bnsd_ptr, "k_nope_bnsd");
                Tensor k_rope_bnsd(DT_BF16, kRopeShape, (uint8_t *)k_rope_bnsd_ptr, "k_rope_bnsd");
                Tensor v_nope_bnsd(DT_BF16, vNopeShape, (uint8_t *)v_nope_bnsd_ptr, "v_nope_bnsd");

                MlaProlog(x, wDq, wUqQr, wUk, wDkvKr, gamma_cq, gamma_ckv, sin, cos, kv_len, kv_cache, kr_cache,
                    quantInputs, ropeConfig, output_q, output_q_rope, kv_cache, kr_cache);

                std::vector<int64_t> tileShape1 = {1, 1, 8, kv_lora_rank};
                std::vector<int64_t> tileShape2 = {1, 8, 1, kv_lora_rank};

                q_nope_bnsd = CastTranspoeReshape(output_q, true, qNopeShape, tileShape1, tileShape2);

                tileShape1 = {1, 1, 8, qk_rope_head_dim};
                tileShape2 = {1, 8, 1, qk_rope_head_dim};
                q_rope_bnsd = CastTranspoeReshape(output_q_rope, true, qRopeShape, tileShape1, tileShape2);

                tileShape1 = {2, 1, 16, kv_lora_rank};
                tileShape2 = {2, 1, 16, kv_lora_rank};
                k_nope_bnsd = CastTranspoeReshape(kv_cache, false, kNopeShape, tileShape1, tileShape2);
                k_rope_bnsd = CastTranspoeReshape(kr_cache, false, kRopeShape, tileShape1, tileShape2);
                v_nope_bnsd = CastTranspoeReshape(kv_cache, false, vNopeShape, tileShape1, tileShape2);

                IncreFlashAttention(q_nope_bnsd, k_nope_bnsd, v_nope_bnsd, q_rope_bnsd, k_rope_bnsd, blockTable,
                    actSeqs, softmaxScale, attentionOut, tileConfig);
                //                IncreFlashAttention(qNope, kNopeCache, vNopeCache, qRope, kRopeCache, blockTable,
                //                actSeqs, softmaxScale, attentionOut, tileConfig);

                TileShape::Current().SetVecTile({4, 16, 1, kv_lora_rank});
                Tensor atten_res0 = Transpose(Cast(Reshape(attentionOut, inputShape), DT_BF16), {1, 2});
                //                        Tensor atten_res0 = Transpose(input_i, {1, 2});
                TileShape::Current().SetVecTile({4, 1, 32, kv_lora_rank});
                Tensor atten_res1 = Reshape(atten_res0, {b * s, n, kv_lora_rank});
                TileShape::Current().SetVecTile({4, 16, kv_lora_rank});
                Tensor t2_res = Transpose(atten_res1, {0, 1});

                TileShape::Current().SetCubeTile({16, 16}, {std::min(256, kv_lora_rank), std::min(256, kv_lora_rank)},
                    {std::min(128, v_head_dim), std::min(128, v_head_dim)}); // M 16对齐
                // 所有子图申请的UB空间总和可能大于192K，所以tileShape不能太大（1、ooo
                // pass申请UB空间的方式不合理，在重构；2、RMS里面有repeattimes写死的64，可能会导致tileShape太大）
                // [n,bs,kv_lora_rank] * [n, kv_lora_rank, v_head_dim] = [n,bs,v_head_dim]
                Tensor bmm4_res = Matrix::BatchMatmul(dType, t2_res, w_uv_i);

                TileShape::Current().SetVecTile(32, 4, v_head_dim); // 必须切，但是尾轴不能切
                Tensor t3_res = Transpose(bmm4_res, {0, 1}); // [bs,n,v_head_dim]

                TileShape::Current().SetVecTile({4, 32, v_head_dim});
                Tensor r2_res = Reshape(t3_res, {b * s, n * v_head_dim});

                TileShape::Current().SetCubeTile({16, 16},
                    {std::min(256, n * v_head_dim), std::min(256, n * v_head_dim)},
                    {std::min(128, h), std::min(128, h)});
                TileShape::Current().SetVecTile(4, 1024);
                Tensor bmm5_res = npu::tile_fwk::Matrix::QuantMM(r2_res, w_o_i, w_o_scale_w_i);

                TileShape::Current().SetVecTile({4, std::min(8192, h)});
                outputT = Reshape(bmm5_res, {b, s, h});
            };

            //            FUNCTION("MlaProlog_T", FunctionType::STATIC, {x, w_qa, w_qb, w_kv_b_k, w_kv_a,
            //                                                                           gamma_cq, gamma_ckv, sin, cos,
            //                                                                           kv_len, kv_cache, kr_cache,
            //                                                                           output_q, output_q_rope}) {
            //                MlaProlog(x, w_qa, w_qb, w_kv_b_k, w_kv_a, gamma_cq, gamma_ckv, sin, cos, kv_len,
            //                kv_cache, kr_cache,
            //                    quantInputs, ropeConfig, output_q, output_q_rope, kv_cache, kr_cache);
            //            };
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());

    std::vector<outDtype> q_golden(capacity_q_out);
    std::vector<outDtype> q_npu(capacity_q_out);
    std::vector<outDtype> q_rope_golden(capacity_q_rope_out);
    std::vector<outDtype> q_rope_npu(capacity_q_rope_out);
    std::vector<T> kv_golden(capacity_kv_out);
    std::vector<T> kv_npu(capacity_kv_out);
    std::vector<T> kr_golden(capacity_kr_out);
    std::vector<T> kr_npu(capacity_kr_out);

    readInput<outDtype>(dataPath + "/q_golden.bin", q_golden);
    readInput<outDtype>(dataPath + "/q_rope_golden.bin", q_rope_golden);
    readInput<T>(dataPath + "/kv_cache_golden.bin", kv_golden);
    readInput<T>(dataPath + "/kr_cache_golden.bin", kr_golden);
    machine::GetRA()->CopyFromTensor((uint8_t *)q_npu.data(), (uint8_t *)q_out_ptr, outputSize0);
    machine::GetRA()->CopyFromTensor((uint8_t *)q_rope_npu.data(), (uint8_t *)q_rope_out_ptr, outputSize1);
    machine::GetRA()->CopyFromTensor((uint8_t *)kv_npu.data(), (uint8_t *)kv_cache_ptr, outputSize2);
    machine::GetRA()->CopyFromTensor((uint8_t *)kr_npu.data(), (uint8_t *)kr_cache_ptr, outputSize3);

    std::cout << "\n====== resultCmp: output q_rope start" << std::endl;
    int ret1 = resultCmp<outDtype>(q_rope_golden, q_rope_npu, 0.005f, 16);
    EXPECT_EQ(ret1, true);

    std::cout << "\n====== resultCmp: output kv_cache start" << std::endl;
    int ret2 = resultCmp<T>(kv_golden, kv_npu, 0.003f, 16);
    EXPECT_EQ(ret2, true);

    std::cout << "\n====== resultCmp: output kr_cache start" << std::endl;
    int ret3 = resultCmp<T>(kr_golden, kr_npu, 0.003f, 16);
    EXPECT_EQ(ret3, true);

    std::cout << "\n====== resultCmp: output attentionOut start" << std::endl;
    std::vector<float> golden(outCap);
    std::vector<float> res(outCap);
    machine::GetRA()->CopyFromTensor((uint8_t *)res.data(), (uint8_t *)outPtr, outputSize);
    readInput(GetGoldenDir() + "/atten_out.bin", golden);
    int ret = resultCmp(golden, res, 0.004f, 16);
    EXPECT_EQ(ret, true);

    std::vector<T> postGolden(postOutputSize);
    std::vector<T> postRes(postOutputSize);
    machine::GetRA()->CopyFromTensor((uint8_t *)res.data(), (uint8_t *)out_ptr, outputByteSize);
    readInput(GetGoldenDir() + "/attn_output.bin", golden);
    ret = resultCmp(golden, res, 0.1f, 650);
    EXPECT_EQ(ret, true);

    if constexpr (DEBUG) {
        std::vector<outDtype> q_golden_trans(capacity_q_out);
        std::vector<outDtype> q_npu_trans(capacity_q_out);
        std::vector<outDtype> q_rope_golden_trans(capacity_q_rope_out);
        std::vector<outDtype> q_rope_npu_trans(capacity_q_rope_out);
        std::vector<T> kv_golden_trans(capacity_kv_out);
        std::vector<T> kv_npu_trans(capacity_kv_out);
        std::vector<T> kr_golden_trans(capacity_kr_out);
        std::vector<T> kr_npu_trans(capacity_kr_out);

        readInput<outDtype>(dataPath + "/a_q_no.bin", q_golden_trans);
        readInput<outDtype>(dataPath + "/a_q_ro.bin", q_rope_golden_trans);
        readInput<T>(dataPath + "/a_kv_no.bin", kv_golden_trans);
        readInput<T>(dataPath + "/a_kv_ro.bin", kr_golden_trans);
        machine::GetRA()->CopyFromTensor((uint8_t *)q_npu_trans.data(), (uint8_t *)q_nope_bnsd_ptr, outputSize0);
        machine::GetRA()->CopyFromTensor((uint8_t *)q_rope_npu_trans.data(), (uint8_t *)q_rope_bnsd_ptr, outputSize1);
        machine::GetRA()->CopyFromTensor((uint8_t *)kv_npu_trans.data(), (uint8_t *)k_nope_bnsd_ptr, outputSize2);
        machine::GetRA()->CopyFromTensor((uint8_t *)kr_npu_trans.data(), (uint8_t *)k_rope_bnsd_ptr, outputSize3);

        std::cout << "\n====== resultCmp: output a_q_no start" << std::endl;
        int ret4 = resultCmp<outDtype>(q_golden_trans, q_npu_trans, 0.008f, 16);
        EXPECT_EQ(ret4, true);

        std::cout << "\n====== resultCmp: output a_q_ro start" << std::endl;
        int ret5 = resultCmp<outDtype>(q_rope_golden_trans, q_rope_npu_trans, 0.005f, 16);
        EXPECT_EQ(ret5, true);

        std::cout << "\n====== resultCmp: output a_kv_no start" << std::endl;
        int ret6 = resultCmp<T>(kv_golden_trans, kv_npu_trans, 0.003f, 16);
        EXPECT_EQ(ret6, true);

        std::cout << "\n====== resultCmp: output a_kv_ro start" << std::endl;
        int ret7 = resultCmp<T>(kr_golden_trans, kr_npu_trans, 0.003f, 16);
        EXPECT_EQ(ret7, true);
    }
}

TEST_F(MlaPrologV2OnBoardTest, attentionV2_bf16_4_1024_1024_32_256) {  // b_n_s_s2_h_q_lora_rank, bfloat16
    int b = 4;
    int s = 1;
    int s2 = 256;
    int h = 256;
    int n = 32;
    int q_lora_rank = 256;
    int qk_nope_head_dim = 128;
    int qk_rope_head_dim = 64;
    int kv_lora_rank = 512;
    int v_head_dim = 128;

    std::vector<int> params = {b, s, s2, n, h, q_lora_rank, qk_nope_head_dim, qk_rope_head_dim,
        kv_lora_rank, v_head_dim};
    TestAttentionV2(params, GetGoldenDir(), hightThroughputTileParams);
}

TEST_F(MlaPrologV2OnBoardTest, attentionV2_bf16_low) {  // b_n_s_s2_h_q_lora_rank, bfloat16
    int b = 4;
    int s = 1;
    int s2 = 256;
    int h = 7168;
    int n = 32;
    int q_lora_rank = 1536;
    int qk_nope_head_dim = 128;
    int qk_rope_head_dim = 64;
    int kv_lora_rank = 512;
    int v_head_dim = 128;

    std::vector<int> params = {b, s, s2, n, h, q_lora_rank, qk_nope_head_dim, qk_rope_head_dim,
        kv_lora_rank, v_head_dim};
    TestAttentionV2(params, GetGoldenDir(), lowLatencyTileParams);
}


TEST_F(MlaPrologV2OnBoardTest, test_MlaPrologV2_float16_32_2_1_256_256_512) {  // b_n_s_s2_h_q_lora_rank
    int b = 32;
    int s = 1;
    int s2 = 256;
    int h = 256;
    int n = 2;
    int qLoraRank = 512;
    int qkNopeHeadDim = 128;
    int qkRopeHeadDim = 64;
    int kvLoraRank = 512;
    int vHeadDim = 128;

    std::vector<int> params = {b, s, s2, n, h, qLoraRank, qkNopeHeadDim, qkRopeHeadDim,
                               kvLoraRank, vHeadDim};
    TestMlaPrologV2<npu::tile_fwk::float16>(params, GetGoldenDir());
}

TEST_F(MlaPrologV2OnBoardTest, test_MlaPrologV2_float16_32_32_1_256_7168_1536) {  // b_n_s_s2_h_q_lora_rank
    int b = 32;
    int s = 1;
    int s2 = 256;
    int h = 7168;
    int n = 32;
    int qLoraRank = 1536;
    int qkNopeHeadDim = 128;
    int qkRopeHeadDim = 64;
    int kvLoraRank = 512;
    int vHeadDim = 128;

    std::vector<int> params = {b, s, s2, n, h, qLoraRank, qkNopeHeadDim, qkRopeHeadDim,
                               kvLoraRank, vHeadDim};
    TestMlaPrologV2<npu::tile_fwk::float16>(params, GetGoldenDir());
}

TEST_F(MlaPrologV2OnBoardTest, test_MlaPrologV2_bfloat16_32_32_1_256_7168_1536) {  // b_n_s_s2_h_q_lora_rank, bfloat16
    int b = 32;
    int s = 1;
    int s2 = 256;
    int h = 7168;
    int n = 32;
    int qLoraRank = 1536;
    int qkNopeHeadDim = 128;
    int qkRopeHeadDim = 64;
    int kvLoraRank = 512;
    int vHeadDim = 128;

    std::vector<int> params = {b, s, s2, n, h, qLoraRank, qkNopeHeadDim, qkRopeHeadDim,
                               kvLoraRank, vHeadDim};
    TestMlaPrologV2<npu::tile_fwk::bfloat16>(params, GetGoldenDir());
}

TEST_F(MlaPrologV2OnBoardTest, test_MlaPrologV2_float16_32_128_1_4096_7168_1536) {  // b_n_s_s2_h_q_lora_rank
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

    std::vector<int> params = {b, s, s2, n, h, qLoraRank, qkNopeHeadDim, qkRopeHeadDim,
                               kvLoraRank, vHeadDim};
    TestMlaPrologV2<npu::tile_fwk::float16>(params, GetGoldenDir());
}

TEST_F(MlaPrologV2OnBoardTest, test_MlaPrologV2_bfloat16_32_128_1_256_7168_1536) {  // b_n_s_s2_h_q_lora_rank, bfloat16
    int b = 32;
    int s = 1;
    int s2 = 256;
    int h = 7168;
    int n = 128;
    int qLoraRank = 1536;
    int qkNopeHeadDim = 128;
    int qkRopeHeadDim = 64;
    int kvLoraRank = 512;
    int vHeadDim = 128;

    std::vector<int> params = {b, s, s2, n, h, qLoraRank, qkNopeHeadDim, qkRopeHeadDim,
                               kvLoraRank, vHeadDim};
    TestMlaPrologV2<npu::tile_fwk::bfloat16>(params, GetGoldenDir());
}

TEST_F(MlaPrologV2OnBoardTest, test_MlaPrologV2_bfloat16_32_128_1_4096_7168_1536) {  // b_n_s_s2_h_q_lora_rank, bfloat16
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

    std::vector<int> params = {b, s, s2, n, h, qLoraRank, qkNopeHeadDim, qkRopeHeadDim,
                               kvLoraRank, vHeadDim};
    TestMlaPrologV2<npu::tile_fwk::bfloat16>(params, GetGoldenDir());
}

TEST_F(MlaPrologV2OnBoardTest, test_MlaPrologV2_bfloat16_4_32_1_256_7168_1536) {  // b_n_s_s2_h_q_lora_rank, bfloat16
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

    std::vector<int> params = {b, s, s2, n, h, qLoraRank, qkNopeHeadDim, qkRopeHeadDim,
                               kvLoraRank, vHeadDim};
    TestMlaPrologV2<npu::tile_fwk::bfloat16>(params, GetGoldenDir());
}

TEST_F(MlaPrologV2OnBoardTest, test_MlaPrologV2_bfloat16_4_32_1_4096_7168_1536) {  // b_n_s_s2_h_q_lora_rank, bfloat16
    int b = 4;
    int s = 1;
    int s2 = 4096;
    int h = 7168;
    int n = 32;
    int qLoraRank = 1536;
    int qkNopeHeadDim = 128;
    int qkRopeHeadDim = 64;
    int kvLoraRank = 512;
    int vHeadDim = 128;

    std::vector<int> params = {b, s, s2, n, h, qLoraRank, qkNopeHeadDim, qkRopeHeadDim,
                               kvLoraRank, vHeadDim};
    TestMlaPrologV2<npu::tile_fwk::bfloat16>(params, GetGoldenDir());
}

TEST_F(MlaPrologV2OnBoardTest, test_MlaPrologV2_bfloat16_4_128_1_256_7168_1536) {  // b_n_s_s2_h_q_lora_rank, bfloat16
    int b = 4;
    int s = 1;
    int s2 = 256;
    int h = 7168;
    int n = 128;
    int qLoraRank = 1536;
    int qkNopeHeadDim = 128;
    int qkRopeHeadDim = 64;
    int kvLoraRank = 512;
    int vHeadDim = 128;

    std::vector<int> params = {b, s, s2, n, h, qLoraRank, qkNopeHeadDim, qkRopeHeadDim,
                               kvLoraRank, vHeadDim};
    TestMlaPrologV2<npu::tile_fwk::bfloat16>(params, GetGoldenDir());
}

TEST_F(MlaPrologV2OnBoardTest, test_MlaPrologV2_bfloat16_4_128_1_4096_7168_1536) {  // b_n_s_s2_h_q_lora_rank, bfloat16
    int b = 4;
    int s = 1;
    int s2 = 4096;
    int h = 7168;
    int n = 128;
    int qLoraRank = 1536;
    int qkNopeHeadDim = 128;
    int qkRopeHeadDim = 64;
    int kvLoraRank = 512;
    int vHeadDim = 128;

    std::vector<int> params = {b, s, s2, n, h, qLoraRank, qkNopeHeadDim, qkRopeHeadDim,
                               kvLoraRank, vHeadDim};
    TestMlaPrologV2<npu::tile_fwk::bfloat16>(params, GetGoldenDir());
}

TEST_F(MlaPrologV2OnBoardTest, test_mla_bf16_low_quant_smooth) {  // b_n_s_s2_h_q_lora_rank
    config::SetPassOption(CUBE_L1_REUSE_MODE, 4);
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

    std::vector<int> params = {b, s, s2, n, h, qLoraRank, qkNopeHeadDim, qkRopeHeadDim,
                               kvLoraRank, vHeadDim};
    TestMlaPrologV2<npu::tile_fwk::bfloat16>(params, GetGoldenDir(), true);
}

TEST_F(MlaPrologV2OnBoardTest, test_mla_bf16_high_quant_smooth) {  // b_n_s_s2_h_q_lora_rank
    const int pg_upper_bound = 20000;
    const int vec_nbuffer_mode = 2;
    config::SetPassOption(VEC_NBUFFER_MODE, vec_nbuffer_mode);
    config::SetPassOption(VEC_NBUFFER_SETTING,  std::map<int64_t, int64_t>{{-1, 2}});
    config::SetPassOption(SG_PG_UPPER_BOUND, pg_upper_bound);
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

    std::vector<int> params = {b, s, s2, n, h, qLoraRank, qkNopeHeadDim, qkRopeHeadDim,
                               kvLoraRank, vHeadDim};
    TestMlaPrologV2<npu::tile_fwk::bfloat16>(params, GetGoldenDir(), true);
}

TEST_F(MlaPrologV2OnBoardTest, test_MlaPrologV2_float16_4_32_1_256_7168_1536_quant) {  // b_n_s_s2_h_q_lora_rank
    config::SetPassOption(CUBE_L1_REUSE_MODE, 4);
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

    std::vector<int> params = {b, s, s2, n, h, qLoraRank, qkNopeHeadDim, qkRopeHeadDim,
                               kvLoraRank, vHeadDim};
    TestMlaPrologV2<npu::tile_fwk::float16>(params, GetGoldenDir(), true);
}

TEST_F(MlaPrologV2OnBoardTest, test_MlaPrologV2_float16_32_32_1_256_7168_1536_quant) {  // b_n_s_s2_h_q_lora_rank
    int b = 32;
    int s = 1;
    int s2 = 256;
    int h = 7168;
    int n = 32;
    int qLoraRank = 1536;
    int qkNopeHeadDim = 128;
    int qkRopeHeadDim = 64;
    int kvLoraRank = 512;
    int vHeadDim = 128;

    std::vector<int> params = {b, s, s2, n, h, qLoraRank, qkNopeHeadDim, qkRopeHeadDim,
                               kvLoraRank, vHeadDim};
    TestMlaPrologV2<npu::tile_fwk::float16>(params, GetGoldenDir(), true);
}

TEST_F(MlaPrologV2OnBoardTest, test_mla_fp16_high_quant_smooth) {  // b_n_s_s2_h_q_lora_rank
    config::SetPassOption(VEC_NBUFFER_MODE, 1);
    config::SetPassOption(CUBE_L1_REUSE_MODE, 4);
    config::SetPassOption(CUBE_NBUFFER_SETTING, std::map<int64_t, int64_t>{{3,4}});
    config::SetPassOption(MG_COPYIN_UPPER_BOUND, 2*1024*1024);
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
    const bool splitReduceLastDim = false;
    const bool splitK = false;
    const bool nz = true;
    const bool usePrefetch = true;
    std::vector<int> params = {b, s, s2, n, h, qLoraRank, qkNopeHeadDim, qkRopeHeadDim,
        kvLoraRank, vHeadDim};
    TestMlaPrologV2<npu::tile_fwk::float16,splitReduceLastDim,splitK,nz,usePrefetch>(params, GetGoldenDir(), true, true);
}

TEST_F(MlaPrologV2OnBoardTest, test_mla_fp16_low_quant_smooth_pa_bsnd) {  // b_n_s_s2_h_q_lora_rank
    config::SetPassOption(CUBE_L1_REUSE_MODE, 4);
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

    int blockSize = 128;
    std::string cacheMode = "PA_BSND";

    const bool splitReduceLastDim = false;
    const bool splitK = false;
    const bool nz = false;
    const bool usePrefetch = false;

    std::vector<int> params = {b, s, s2, n, h, qLoraRank, qkNopeHeadDim, qkRopeHeadDim,
                               kvLoraRank, vHeadDim};
    TestMlaPrologV2<npu::tile_fwk::float16, splitReduceLastDim, splitK, nz, usePrefetch>(params, GetGoldenDir(), true, true,
                                                                               blockSize, cacheMode);
}

TEST_F(MlaPrologV2OnBoardTest, test_mla_fp16_high_quant_smooth_pa_bsnd) {  // b_n_s_s2_h_q_lora_rank
    const int vec_nbuffer_mode = 2;
    config::SetPassOption(VEC_NBUFFER_MODE, vec_nbuffer_mode);
    config::SetPassOption(VEC_NBUFFER_SETTING,  std::map<int64_t, int64_t>{{-1, 2}});
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

    int blockSize = 128;
    std::string cacheMode = "PA_BSND";

    const bool splitReduceLastDim = false;
    const bool splitK = false;
    const bool nz = false;
    const bool usePrefetch = false;

    std::vector<int> params = {b, s, s2, n, h, qLoraRank, qkNopeHeadDim, qkRopeHeadDim,
                               kvLoraRank, vHeadDim};
    TestMlaPrologV2<npu::tile_fwk::float16, splitReduceLastDim, splitK, nz, usePrefetch>(params, GetGoldenDir(), true, true,
                                                                               blockSize, cacheMode);
}

TEST_F(MlaPrologV2OnBoardTest, test_mla_fp16_low_quant_smooth_nz_pa_bsnd) {  // b_n_s_s2_h_q_lora_rank
    config::SetPassOption(CUBE_L1_REUSE_MODE, 4);
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

    int blockSize = 128;
    std::string cacheMode = "PA_BSND";

    const bool splitReduceLastDim = false;
    const bool splitK = false;
    const bool nz = true;
    const bool usePrefetch = false;

    std::vector<int> params = {b, s, s2, n, h, qLoraRank, qkNopeHeadDim, qkRopeHeadDim,
                               kvLoraRank, vHeadDim};
    TestMlaPrologV2<npu::tile_fwk::float16, splitReduceLastDim, splitK, nz, usePrefetch>(params, GetGoldenDir(), true, true,
                                                                               blockSize, cacheMode);
}

TEST_F(MlaPrologV2OnBoardTest, test_mla_bf16_high_quant_smooth_nz_pa_bsnd) {  // b_n_s_s2_h_q_lora_rank
    config::SetPassOption(VEC_NBUFFER_MODE, 2);
    config::SetPassOption(VEC_NBUFFER_SETTING,  std::map<int64_t, int64_t>{{-1, 2}});
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

    int blockSize = 128;
    std::string cacheMode = "PA_BSND";

    const bool splitReduceLastDim = false;
    const bool splitK = false;
    const bool nz = true;
    const bool usePrefetch = false;

    std::vector<int> params = {b, s, s2, n, h, qLoraRank, qkNopeHeadDim, qkRopeHeadDim,
                               kvLoraRank, vHeadDim};
    TestMlaPrologV2<npu::tile_fwk::bfloat16, splitReduceLastDim, splitK, nz, usePrefetch>(params, GetGoldenDir(), true, true,
                                                                                 blockSize, cacheMode);
}

TEST_F(MlaPrologV2OnBoardTest, test_mla_bf16_high48_quant_smooth_nz_pa_bsnd) {  // b_n_s_s2_h_q_lora_rank
    config::SetPassOption(VEC_NBUFFER_MODE, 2);
    config::SetPassOption(VEC_NBUFFER_SETTING,  std::map<int64_t, int64_t>{{-1, 2}});
    int b = 48;
    int s = 1;
    int s2 = 4096;
    int h = 7168;
    int n = 128;
    int qLoraRank = 1536;
    int qkNopeHeadDim = 128;
    int qkRopeHeadDim = 64;
    int kvLoraRank = 512;
    int vHeadDim = 128;

    int blockSize = 128;
    std::string cacheMode = "PA_BSND";

    const bool splitReduceLastDim = false;
    const bool splitK = false;
    const bool nz = true;
    const bool usePrefetch = false;

    std::vector<int> params = {b, s, s2, n, h, qLoraRank, qkNopeHeadDim, qkRopeHeadDim,
                               kvLoraRank, vHeadDim};
    TestMlaPrologV2<npu::tile_fwk::bfloat16, splitReduceLastDim, splitK, nz, usePrefetch>(params, GetGoldenDir(), true, true,
                                                                                 blockSize, cacheMode);
}
