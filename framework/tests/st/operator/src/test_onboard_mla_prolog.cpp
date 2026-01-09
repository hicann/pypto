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
 * \file test_onboard_mla_prolog.cpp
 * \brief
 */

#include "test_suite_stest_ops.h"
#include "operator/models/deepseek/deepseek_mla.h"
#include "test_dev_func_runner.h"

using namespace npu::tile_fwk;

class MlaPrologOnBoardTest : public npu::tile_fwk::stest::TestSuite_STest_Ops_Aihac {};

template <typename T = npu::tile_fwk::float16>
void TestMlaProlog(std::vector<int> &params, string dataPath, bool isQuant = false, uint64_t allTimeThreshold = 0, uint64_t opTimeThreshold = 0) {
    // b, s, s2, n, h, qLoraRank, qkNopeHeadDim, qkRopeHeadDim, kvLoraRank, vHeadDim
    auto start_all = std::chrono::high_resolution_clock::now();
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
    std::vector<int64_t> position_ids_shape = {b, s};
    std::vector<int64_t> cos_shape = {s, qkRopeHeadDim};
    std::vector<int64_t> past_key_states_shape = {b, 1, s2, kvLoraRank + qkRopeHeadDim};
    std::vector<int64_t> kv_len_shape = {1, 1};
    // output
    std::vector<int64_t> q_shape = {b, n, s, kvLoraRank + qkRopeHeadDim};
    std::vector<int64_t> kv_shape = {b, 1, s2, kvLoraRank + qkRopeHeadDim};

    int capacity_x = std::accumulate(x_shape.begin(), x_shape.end(), 1, std::multiplies<>());
    int capacity_w_qa = std::accumulate(w_qa_shape.begin(), w_qa_shape.end(), 1, std::multiplies<>());
    int capacity_w_qb = std::accumulate(w_qb_shape.begin(), w_qb_shape.end(), 1, std::multiplies<>());
    int capacity_w_kv_a = std::accumulate(w_kv_a_shape.begin(), w_kv_a_shape.end(), 1, std::multiplies<>());
    int capacity_w_kv_b_k = std::accumulate(w_kv_b_k_shape.begin(), w_kv_b_k_shape.end(), 1, std::multiplies<>());
    int capacity_position_ids = std::accumulate(position_ids_shape.begin(), position_ids_shape.end(), 1,
                                                std::multiplies<>());
    int capacity_cos = std::accumulate(cos_shape.begin(), cos_shape.end(), 1, std::multiplies<>());
    int capacity_past_key_states = std::accumulate(past_key_states_shape.begin(), past_key_states_shape.end(), 1,
                                                   std::multiplies<>());
    int capacity_kv_len = std::accumulate(kv_len_shape.begin(), kv_len_shape.end(), 1, std::multiplies<>());
    // output
    int capacity_q = std::accumulate(q_shape.begin(), q_shape.end(), 1, std::multiplies<>());
    int capacity_kv = std::accumulate(kv_shape.begin(), kv_shape.end(), 1, std::multiplies<>());

    std::vector<int64_t> w_qb_scale_shape;
    int capacity_w_qb_scale;
    if (isQuant) {
        w_qb_scale_shape = {1, n * q_head_dim};
        capacity_w_qb_scale = std::accumulate(w_qb_scale_shape.begin(), w_qb_scale_shape.end(), 1, std::multiplies<>());
    }

    aclInit(nullptr);
    rtSetDevice(GetDeviceIdByEnvVar());
    uint64_t outputSize0 = capacity_q * sizeof(T);
    uint64_t outputSize1 = capacity_kv * sizeof(T);
    uint8_t* q_out_ptr = allocDevAddr(outputSize0);
    void *past_key_states_ptr = readToDev<T>(dataPath + "/past_key_states.bin", capacity_past_key_states);


    ConfigManager::Instance();
    auto start_op = std::chrono::high_resolution_clock::now();
    PROGRAM("MlaProlog") {
        void *x_ptr = readToDev<T>(dataPath + "/x.bin", capacity_x);
        void *w_qa_ptr = readToDev<T>(dataPath + "/w_qa.bin", capacity_w_qa);
        void *w_qb_ptr = isQuant ? readToDev<wDtype>(dataPath + "/w_qb.bin", capacity_w_qb) :
                                    readToDev<T>(dataPath + "/w_qb.bin", capacity_w_qb);
        void *w_kv_a_ptr = readToDev<T>(dataPath + "/w_kv_a.bin", capacity_w_kv_a);
        void *w_kv_b_k_ptr = readToDev<T>(dataPath + "/w_kv_b_k.bin", capacity_w_kv_b_k);
        void *position_ids_ptr = readToDev<int32_t>(dataPath + "/position_ids.bin", capacity_position_ids);
        void *cos_ptr = readToDev<T>(dataPath + "/cos.bin", capacity_cos);
        void *sin_ptr = readToDev<T>(dataPath + "/sin.bin", capacity_cos);
        void *kv_len_ptr = readToDev<int32_t>(dataPath + "/kv_len.bin", capacity_kv_len);

        Tensor x(dType, x_shape, (uint8_t *)x_ptr, "x");
        Tensor w_qa(dType, w_qa_shape, (uint8_t *)w_qa_ptr, "w_qa");
        Tensor w_qb(dTypeQuantIn, w_qb_shape, (uint8_t *)w_qb_ptr, "w_qb");
        Tensor w_kv_a(dType, w_kv_a_shape, (uint8_t *)w_kv_a_ptr, "w_kv_a");
        Tensor w_kv_b_k(dType, w_kv_b_k_shape, (uint8_t *)w_kv_b_k_ptr, "w_kv_b_k");
        Tensor position_ids(DataType::DT_INT32, position_ids_shape, (uint8_t *)position_ids_ptr, "position_ids");
        Tensor cos(dType, cos_shape, (uint8_t *)cos_ptr, "cos");
        Tensor sin(dType, cos_shape, (uint8_t *)sin_ptr, "sin");
        Tensor past_key_states(dType, past_key_states_shape, (uint8_t *)past_key_states_ptr, "past_key_states");
        Tensor kv_len(DataType::DT_INT32, kv_len_shape, (uint8_t *)kv_len_ptr, "kv_len");
        // output
        Tensor output_q(dType, q_shape, q_out_ptr, "output_q");

        AttentionW aw;
        aw.qAProjW = w_qa;
        aw.qBProjW = w_qb;
        aw.kvAProjWithMqaW = w_kv_a;
        aw.kvBProjWK = w_kv_b_k;
        Tensor kvBProjWV;  // not used in MlaProlog
        Tensor oProjW;       // not used in MlaProlog
        aw.kvBProjWV = kvBProjWV;
        aw.oProjW = oProjW;

        if (isQuant) {
            void *w_qb_scale_ptr = readToDev<float>(dataPath + "/w_qb_scale.bin", capacity_w_qb_scale);
            Tensor w_qb_scale = Tensor(DataType::DT_FP32, w_qb_scale_shape, (uint8_t *)w_qb_scale_ptr, "w_qb_scale");
            aw.qBProjWScale = w_qb_scale;

            std::tuple<Tensor, Tensor> res;
            DeepseekAttention Attention(g_deepseekConfig, aw, 1);

            RoPETileShapeConfig ropeTileConfig{
                {32, 64}, // for cos/sin->cast, [s,d]
                {1, 32, 64}, // for gather,unsqueeze, [b,s,d]
                {1, 32, 1, 64}, // [b,n,s,d]
                {1, 32, 1, 64, 64} // for transpose, [b,n,s,d/2,2]
            };

            config::SetBuildStatic(true);
            FUNCTION("MlaProlog_T", {x, w_qa, w_qb, w_qb_scale, w_kv_a, w_kv_b_k, position_ids,
                    cos, sin, past_key_states, kv_len, output_q}) {
                auto q_kv = Attention.MlaPrologFoward(x, position_ids, cos, sin, kv_len, past_key_states, ropeTileConfig, isQuant);
                output_q = q_kv[0];
                past_key_states = q_kv[1];
            }
        } else {
            std::tuple<Tensor, Tensor> res;
            DeepseekAttention Attention(g_deepseekConfig, aw, 1);

            RoPETileShapeConfig ropeTileConfig{
                {32, 64}, // for cos/sin->cast, [s,d]
                {1, 32, 64}, // for gather,unsqueeze, [b,s,d]
                {1, 32, 1, 64}, // [b,n,s,d]
                {1, 32, 1, 64, 64} // for transpose, [b,n,s,d/2,2]
            };

            config::SetBuildStatic(true);
            FUNCTION("MlaProlog_T", {x, w_qa, w_qb, w_kv_a, w_kv_b_k, position_ids,
                    cos, sin, past_key_states, kv_len, output_q}) {
                auto q_kv = Attention.MlaPrologFoward(x, position_ids, cos, sin, kv_len, past_key_states, ropeTileConfig, isQuant);
                output_q = q_kv[0];
                past_key_states = q_kv[1];
            }
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());
    auto end_op = std::chrono::high_resolution_clock::now();
    std::vector<outDtype> q_golden(capacity_q);
    std::vector<outDtype> q_npu(capacity_q);
    std::vector<T> kv_golden(capacity_kv);
    std::vector<T> kv_npu(capacity_kv);

    readInput<outDtype>(dataPath + "/q_golden.bin", q_golden);
    readInput<T>(dataPath + "/kv_golden.bin", kv_golden);
    machine::GetRA()->CopyFromTensor((uint8_t *)q_npu.data(), (uint8_t *)q_out_ptr, outputSize0);
    machine::GetRA()->CopyFromTensor((uint8_t *)kv_npu.data(), (uint8_t *)past_key_states_ptr, outputSize1);

    std::cout << "\n====== resultCmp: output q start" << std::endl;
    int ret0 = resultCmp<outDtype>(q_golden, q_npu, 0.005f);
    EXPECT_EQ(ret0, true);
    std::cout << "\n====== resultCmp: output kv start" << std::endl;
    int ret1 = resultCmp<T>(kv_golden, kv_npu, 0.003f);
    EXPECT_EQ(ret1, true);
    auto end_all = std::chrono::high_resolution_clock::now();
    auto duration_all = std::chrono::duration_cast<std::chrono::microseconds>(end_all - start_all).count();
    auto duration_op = std::chrono::duration_cast<std::chrono::microseconds>(end_op - start_op).count();
    if (allTimeThreshold != 0) {
        EXPECT_LT(duration_all, allTimeThreshold * 1.1);
    }
    if (opTimeThreshold != 0) {
        EXPECT_LT(duration_op, opTimeThreshold * 1.05);
    }
}

void readBlockTable(const std::string& filename, int rows, int cols, std::vector<std::vector<int>> & blockTable) {
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

template <typename T = npu::tile_fwk::float16>
void AstToFile(std::string name,vector<T> datas){
    std::ofstream outFile(name + ".bin", std::ios::binary);
    T* arr = datas.data();

    if (outFile.is_open()) {
        // 写入数组数据
        outFile.write(reinterpret_cast<const char*>(arr), datas.size() * sizeof(T));
        // 关闭文件
        outFile.close();
        std::cout << "数据已成功写入二进制文件。" << std::endl;
    } else {
        std::cerr << "无法打开文件。" << std::endl;
    }
}


template <typename T = npu::tile_fwk::float16>
void Attention(std::vector<int> &params, string dataPath, bool isQuant = false, bool fuse = true, bool usePost = true,
    bool skipReshape = true) {
    int b = params[0];
    int s1 = params[1];
    int s2 = params[2];
    int nq = params[3];
    int h = params[4];
    int qLoraRank = params[5];
    int qkNopeHeadDim = params[6];
    int qkRopeHeadDim = params[7];
    int kvLoraRank = params[8];
    int q_head_dim = qkNopeHeadDim + qkRopeHeadDim;
    int vHeadDim = params[9];
    const int nkv = 1;

    DataType dType = DT_FP32;
    if (std::is_same<T, npu::tile_fwk::float16>::value) {
        dType = DT_FP16;
    } else if (std::is_same<T, npu::tile_fwk::bfloat16>::value) {
        dType = DT_BF16;
    } else {
        dType = DT_FP32;
    }
    int dtypeSize = BytesOf(dType);

    std::vector<int> actSeqs(b, s2);
    const int blockSize = 256;
    const int nTile = nq;
    const float softmaxScale = 0.8f;
    IfaTileShapeConfig tileConfig;
    tileConfig.blockSize = blockSize;
    tileConfig.headNumQTile = nTile;
    tileConfig.v0TileShape = {32, 128};
    tileConfig.c1TileShape = {32, 32, 64, 64, 64, 64};
    tileConfig.v1TileShape = {32, 128};
    tileConfig.c2TileShape = {32, 32, 64, 64, 64, 64};
    tileConfig.v2TileShape = {32, 128};

    // 输出size
    int outCap = b * 1 * nq * kvLoraRank;
    uint64_t outputSize = outCap * sizeof(float);
    uint8_t *outPtr = allocDevAddr(outputSize);

    // 根据Per Batch实际的sequence构造blockNum，blockNum >= Sum(blockNumPerBatch)，此处选取相等场景
    int blockNum = 0;
    for (auto s : actSeqs) {
        blockNum += CeilDiv(s, blockSize);
    }

    // 输入size
    int qNopeSize = b * s1 * nq * kvLoraRank;
    int qRopeSize = b * s1 * nq * qkRopeHeadDim;

    // B B H
    int kNopeCacheSize = blockNum * blockSize * nkv * kvLoraRank;
    int kRopeCacheSize = blockNum * blockSize * nkv * qkRopeHeadDim;
    int vNopeCacheSize = blockNum * blockSize * nkv * kvLoraRank;

    DataType dTypeQuantIn = isQuant ? DT_INT8 : dType;
    // typedef float outDtype;
    typedef T outDtype;
    typedef int8_t wDtype;

    std::vector<int64_t> x_shape = {b, s1, h};
    std::vector<int64_t> w_qa_shape = {h, qLoraRank};
    std::vector<int64_t> w_qb_shape = {qLoraRank, nq * q_head_dim};
    std::vector<int64_t> w_kv_a_shape = {h, kvLoraRank + qkRopeHeadDim};
    std::vector<int64_t> w_kv_b_k_shape = {nq, qkNopeHeadDim, kvLoraRank};
    std::vector<int64_t> position_ids_shape = {b, s1};
    std::vector<int64_t> cos_shape = {s1, qkRopeHeadDim};
    std::vector<int64_t> past_key_states_shape = {b, 1, s2, kvLoraRank + qkRopeHeadDim};
    std::vector<int64_t> kv_len_shape = {b, s1};;
    // output
    std::vector<int64_t> q_shape = {b, nq, s1, kvLoraRank + qkRopeHeadDim};
    std::vector<int64_t> kv_shape = {b, 1, s2, kvLoraRank + qkRopeHeadDim};

    int capacity_x = std::accumulate(x_shape.begin(), x_shape.end(), 1, std::multiplies<>());
    int capacity_w_qa = std::accumulate(w_qa_shape.begin(), w_qa_shape.end(), 1, std::multiplies<>());
    int capacity_w_qb = std::accumulate(w_qb_shape.begin(), w_qb_shape.end(), 1, std::multiplies<>());
    int capacity_w_kv_a = std::accumulate(w_kv_a_shape.begin(), w_kv_a_shape.end(), 1, std::multiplies<>());
    int capacity_w_kv_b_k = std::accumulate(w_kv_b_k_shape.begin(), w_kv_b_k_shape.end(), 1, std::multiplies<>());
    int capacity_position_ids = std::accumulate(position_ids_shape.begin(), position_ids_shape.end(), 1,
        std::multiplies<>());
    int capacity_cos = std::accumulate(cos_shape.begin(), cos_shape.end(), 1, std::multiplies<>());
    int capacity_past_key_states = std::accumulate(past_key_states_shape.begin(), past_key_states_shape.end(), 1,
        std::multiplies<>());
    int capacity_kv_len = std::accumulate(kv_len_shape.begin(), kv_len_shape.end(), 1, std::multiplies<>());
    // output
    int capacity_q = std::accumulate(q_shape.begin(), q_shape.end(), 1, std::multiplies<>());
    int capacity_kv = std::accumulate(kv_shape.begin(), kv_shape.end(), 1, std::multiplies<>());


    int inputSize = b*nq*s1*kvLoraRank;
    int wUvSize = nq*kvLoraRank*vHeadDim;
    int wOSize = nq*vHeadDim*h;
    int postOutputSize = b*s1*h;
    int t1Size = b*s1*nq*kvLoraRank;

    uint64_t outputByteSize = postOutputSize * dtypeSize;
    uint8_t* out_ptr = allocDevAddr(outputByteSize);

    std::vector<int64_t> w_qb_scale_shape;
    int capacity_w_qb_scale;
    if (isQuant) {
        w_qb_scale_shape = {1, nq * q_head_dim};
        capacity_w_qb_scale = std::accumulate(w_qb_scale_shape.begin(), w_qb_scale_shape.end(), 1, std::multiplies<>());
    }

    int q0_capacity = b * s1 * nq*kvLoraRank;
    int q1_capacity = b * s1 * nq* qkRopeHeadDim;
    int k0_capacity = blockNum * blockSize * nkv* kvLoraRank;
    int k1_capacity = blockNum * blockSize * nkv* qkRopeHeadDim;
    int v0_capacity = blockNum * blockSize * nkv* kvLoraRank;

    int q0_size = q0_capacity* sizeof(T);
    int q1_size = q1_capacity* sizeof(T);
    int k0_size = k0_capacity* sizeof(T);
    int k1_size = k1_capacity* sizeof(T);
    int v0_size = v0_capacity* sizeof(T);

    uint8_t* q0_ptr = allocDevAddr(q0_size);
    uint8_t* q1_ptr = allocDevAddr(q1_size);
    uint8_t* k0_ptr = allocDevAddr(k0_size);
    uint8_t* k1_ptr = allocDevAddr(k1_size);
    uint8_t* v0_ptr = allocDevAddr(v0_size);

    aclInit(nullptr);
    rtSetDevice(GetDeviceIdByEnvVar());
    uint64_t outputSize0 = capacity_q * sizeof(T);
    uint64_t outputSize1 = capacity_kv * sizeof(T);
    uint8_t* q_out_ptr = allocDevAddr(outputSize0);
    void *past_key_states_ptr = readToDev<T>(dataPath + "/past_key_states.bin", capacity_past_key_states);


    ConfigManager::Instance();
    PROGRAM("MlaProlog") {
        void *x_ptr = readToDev<T>(dataPath + "/x.bin", capacity_x);
        void *w_qa_ptr = readToDev<T>(dataPath + "/w_qa.bin", capacity_w_qa);
        void *w_qb_ptr = isQuant ? readToDev<wDtype>(dataPath + "/w_qb.bin", capacity_w_qb) :
                                   readToDev<T>(dataPath + "/w_qb.bin", capacity_w_qb);
        void *w_kv_a_ptr = readToDev<T>(dataPath + "/w_kv_a.bin", capacity_w_kv_a);
        void *w_kv_b_k_ptr = readToDev<T>(dataPath + "/w_kv_b_k.bin", capacity_w_kv_b_k);
        void *position_ids_ptr = readToDev<int32_t>(dataPath + "/position_ids.bin", capacity_position_ids);
        void *cos_ptr = readToDev<T>(dataPath + "/cos.bin", capacity_cos);
        void *sin_ptr = readToDev<T>(dataPath + "/sin.bin", capacity_cos);
        void *kv_len_ptr = readToDev<int64_t>(dataPath + "/kv_len.bin", capacity_kv_len);

        Tensor x(dType, x_shape, (uint8_t *)x_ptr, "x");
        Tensor w_qa(dType, w_qa_shape, (uint8_t *)w_qa_ptr, "w_qa");
        Tensor w_qb(dTypeQuantIn, w_qb_shape, (uint8_t *)w_qb_ptr, "w_qb");
        Tensor w_kv_a(dType, w_kv_a_shape, (uint8_t *)w_kv_a_ptr, "w_kv_a");
        Tensor w_kv_b_k(dType, w_kv_b_k_shape, (uint8_t *)w_kv_b_k_ptr, "w_kv_b_k");
        Tensor position_ids(DT_INT32, position_ids_shape, (uint8_t *)position_ids_ptr, "position_ids");
        Tensor cos(dType, cos_shape, (uint8_t *)cos_ptr, "cos");
        Tensor sin(dType, cos_shape, (uint8_t *)sin_ptr, "sin");
        Tensor past_key_states(dType, past_key_states_shape, (uint8_t *)past_key_states_ptr, "past_key_states");
        Tensor kv_len(DT_INT64, kv_len_shape, (uint8_t *)kv_len_ptr, "kv_len");
        // output
        Tensor output_q(dType, q_shape, q_out_ptr, "output_q");

        AttentionW aw;
        aw.qAProjW = w_qa;
        aw.qBProjW = w_qb;
        aw.kvAProjWithMqaW = w_kv_a;
        aw.kvBProjWK = w_kv_b_k;
        Tensor kvBProjWV;  // not used in MlaProlog
        Tensor oProjW;       // not used in MlaProlog
        aw.kvBProjWV = kvBProjWV;
        aw.oProjW = oProjW;

        // 读数据
        void *qNopeData = readToDev(GetGoldenDir() + "/q_nope.bin", qNopeSize);
        void *qRopeData = readToDev(GetGoldenDir() + "/q_rope.bin", qRopeSize);
        void *kNopeCacheData = readToDev(GetGoldenDir() + "/k_cache_nope.bin", kNopeCacheSize);
        void *kRopeCacheData = readToDev(GetGoldenDir() + "/k_cache_rope.bin", kRopeCacheSize);
        void *vNopeCacheData = readToDev(GetGoldenDir() + "/v_cache.bin", vNopeCacheSize);

        Tensor qNope(DT_BF16, {b * s1 * nq, kvLoraRank}, (uint8_t *)qNopeData, "qNope");
        Tensor qRope(DT_BF16, {b * s1 * nq, qkRopeHeadDim}, (uint8_t *)qRopeData, "qRope");
        Tensor kNopeCache(
            DT_BF16, {blockNum * blockSize * nkv, kvLoraRank}, (uint8_t *)kNopeCacheData, "kNopeCache");
        Tensor kRopeCache(
            DT_BF16, {blockNum * blockSize * nkv, qkRopeHeadDim}, (uint8_t *)kRopeCacheData, "kRope");
        Tensor vNopeCache(
            DT_BF16, {blockNum * blockSize * nkv, kvLoraRank}, (uint8_t *)vNopeCacheData, "vNopeCache");

        Tensor q0(DT_BF16, {b * s1 * nq, kvLoraRank}, q0_ptr, "q0");
        Tensor q1(DT_BF16, {b * s1 * nq, qkRopeHeadDim}, q1_ptr, "q1");
        Tensor k0(DT_BF16, {blockNum * blockSize * nkv, kvLoraRank}, k0_ptr, "k0");
        Tensor k1(DT_BF16, {blockNum * blockSize * nkv, qkRopeHeadDim}, k1_ptr, "k1");
        Tensor v0(DT_BF16, {blockNum * blockSize * nkv, kvLoraRank}, v0_ptr, "v0");

        // blockTable: (b, maxBlockNumPerBatch)
        int maxSeqAllBatch = *(std::max_element(actSeqs.begin(), actSeqs.end()));
        int maxBlockNumPerBatch = CeilDiv(maxSeqAllBatch, blockSize);
        std::vector<std::vector<int>> blockTable(b, std::vector<int>(maxBlockNumPerBatch, 0));
        readBlockTable(GetGoldenDir() + "/block_table.bin", b, maxBlockNumPerBatch, blockTable);

        for (const auto& row : blockTable) {
            for (int num : row) {
                std::cout << num << " ";
            }
            std::cout << std::endl;
        }

        Tensor attentionOut(DT_FP32, {b * s1 * nq, kvLoraRank}, outPtr, "attentionOut");

        std::vector<int64_t> inputShape = {b,nq,s1,kvLoraRank};
        std::vector<int64_t> wUvShape = {nq,kvLoraRank,vHeadDim};
        std::vector<int64_t> wOShape = {nq*vHeadDim,h};
        std::vector<int64_t> outputShapeT = {b, s1, h};
        std::vector<int64_t> t1Shape = {b, s1, nq, kvLoraRank};
        void *input_ptr = readToDev<T>(GetGoldenDir() + "/input.bin", inputSize);
        void *w_uv_ptr = readToDev<T>(GetGoldenDir() + "/w_uv.bin", wUvSize);
        void *w_o_ptr = readToDev<T>(GetGoldenDir() + "/w_o.bin", wOSize);

        void *t1_ptr = readToDev<T>(GetGoldenDir() + "/t1.bin", t1Size);

        Tensor input_i(dType, inputShape, (uint8_t *)input_ptr, "A");
        Tensor w_uv_i(dType, wUvShape, (uint8_t *)w_uv_ptr, "B");
        Tensor w_o_i(dType, wOShape, (uint8_t *)w_o_ptr, "C");
        Tensor outputT(dType, outputShapeT, out_ptr, "D1");
        Tensor t1_i(dType, t1Shape, (uint8_t *)t1_ptr, "E");

        if (isQuant) {
            void *w_qb_scale_ptr = readToDev<float>(dataPath + "/w_qb_scale.bin", capacity_w_qb_scale);
            Tensor w_qb_scale = Tensor(DT_FP32, w_qb_scale_shape, (uint8_t *)w_qb_scale_ptr, "w_qb_scale");
            aw.qBProjWScale = w_qb_scale;

            std::tuple<Tensor, Tensor> res;
            DeepseekAttention Attention(g_deepseekConfig, aw, 1);

            RoPETileShapeConfig ropeTileConfig{
                {32, 64}, // for cos/sin->cast, [s1,d]
                {1, 32, 64}, // for gather,unsqueeze, [b,s1,d]
                {1, 32, 1, 64}, // [b,nq,s1,d]
                {1, 32, 1, 64, 64} // for transpose, [b,nq,s1,d/2,2]
            };

            config::SetBuildStatic(true);
            FUNCTION("MlaProlog_T", {x, w_qa, w_qb, w_qb_scale, w_kv_a, w_kv_b_k, position_ids,
                                                                           cos, sin, past_key_states, kv_len, output_q}) {
                auto q_kv = Attention.MlaPrologFoward(x, position_ids, cos, sin, kv_len, past_key_states, ropeTileConfig, isQuant);
                output_q = q_kv[0];
                past_key_states = q_kv[1];
            }
        } else {
            std::tuple<Tensor, Tensor> res;
            DeepseekAttention Attention(g_deepseekConfig, aw, 1);

            RoPETileShapeConfig ropeTileConfig{
                {32, 64}, // for cos/sin->cast, [s1,d]
                {1, 32, 64}, // for gather,unsqueeze, [b,s1,d]
                {1, 32, 1, 64}, // [b,nq,s1,d]
                {1, 32, 1, 64, 64} // for transpose, [b,nq,s1,d/2,2]
            };

            config::SetBuildStatic(true);
            FUNCTION("MlaProlog_T",
                {x, w_qa, w_qb, w_kv_a, w_kv_b_k, position_ids, cos, sin, past_key_states, kv_len, output_q,
                    q0, q1, k0, k1, v0,
                    attentionOut,
                    qNope, kNopeCache, vNopeCache, qRope, kRopeCache,
                    input_i, t1_i, w_uv_i, w_o_i, outputT}) {
                if (skipReshape){
                    config::SetPassConfig("PVC2_OOO", "SplitReshape", KEY_DISABLE_PASS, true);
                }

                auto q_kv = Attention.MlaPrologFoward(
                    x, position_ids, cos, sin, kv_len, past_key_states, ropeTileConfig, isQuant);
                output_q = q_kv[0];
                past_key_states = q_kv[1];
                q0 = q_kv[2];
                q1 = q_kv[3];

                auto tmp_k0 = View(past_key_states, {b, 1, s2, kvLoraRank}, {0, 0, 0, 0});
                auto tmp32_k0 = Cast(tmp_k0, DT_FP32);
                k0 = Cast(tmp32_k0, DT_BF16);

                auto tmp_k1 = View(past_key_states, {b, 1, s2, qkRopeHeadDim}, {0, 0, 0, kvLoraRank});
                auto tmp32_k1 = Cast(tmp_k1, DT_FP32);
                k1 = Cast(tmp32_k1, DT_BF16);

                auto tmp_v0 = View(past_key_states, {b, 1, s2, kvLoraRank}, {0, 0, 0, 0});
                auto tmp32_v0 = Cast(tmp_v0, DT_FP32);
                v0 = Cast(tmp32_v0, DT_BF16);

                TileShape::Current().SetVecTile({2, 16, 1, qkRopeHeadDim});
                auto q0_1 = Reshape(q0, {b * s1 * nq, kvLoraRank});
                auto q1_1 = Reshape(q1, {b * s1 * nq, qkRopeHeadDim});
                TileShape::Current().SetVecTile({2, 2, 128, qkRopeHeadDim});
                auto k0_1 = Reshape(k0, {blockNum * blockSize * nkv, kvLoraRank});
                auto k1_1 = Reshape(k1, {blockNum * blockSize * nkv, qkRopeHeadDim});
                auto v0_1 = Reshape(v0, {blockNum * blockSize * nkv, kvLoraRank});

                if (fuse) {
                    IncreFlashAttention(
                        q0_1, k0_1, v0_1, q1_1, k1_1, blockTable, actSeqs, softmaxScale, attentionOut, tileConfig);
                } else {
                    IncreFlashAttention(qNope, kNopeCache, vNopeCache, qRope, kRopeCache, blockTable, actSeqs,
                        softmaxScale, attentionOut, tileConfig);
                }
               if (usePost) {
                   TileShape::Current().SetVecTile({16, kvLoraRank});
                   auto tempAttentionOut = Reshape(attentionOut, inputShape);
                   TileShape::Current().SetVecTile({2, 16, 1, kvLoraRank});
                   auto castOut = Cast(tempAttentionOut, DataType::DT_BF16);
                   Tensor atten_res0 = Transpose(castOut, {1, 2});
                   TileShape::Current().SetVecTile({4, 1, 32, kvLoraRank});
                   Tensor atten_res1 = Reshape(atten_res0, {b * s1, nq, kvLoraRank});
                   TileShape::Current().SetVecTile({2, 16, kvLoraRank});
                   Tensor t2_res = Transpose(atten_res1, {0, 1});

                   TileShape::Current().SetCubeTile({16, 16}, {std::min(256, kvLoraRank), std::min(256, kvLoraRank)},
                       {std::min(128, vHeadDim), std::min(128, vHeadDim)}); // M 16对齐
                   TileShape::Current().SetVecTile(8, 4,
                       vHeadDim); // 所有子图申请的UB空间总和可能大于192K，所以tileShape不能太大（1、ooo
                                  // pass申请UB空间的方式不合理，在重构；2、RMS里面有repeattimes写死的64，可能会导致tileShape太大）
                   // [n,bs,kvLoraRank] * [n, kvLoraRank, vHeadDim] = [n,bs,vHeadDim]
                   Tensor bmm4_res = Matrix::BatchMatmul(dType, t2_res, w_uv_i);

                   // 原 TileShape::Current().SetVecTile(8, 16, vHeadDim); //
                   // 1、因为cast不支持跳写，这个Transpose必须与上面的tileshape后两维一致； 2、Transpose尾轴不能切
                   Tensor t3_res = Transpose(bmm4_res, {0, 1}); // [bs,n,vHeadDim]

                   TileShape::Current().SetVecTile({4, 32, vHeadDim});
                   Tensor r2_res = Reshape(t3_res, {b * s1, nq * vHeadDim});

                   // [b,s, n*vHeadDim] @ [n*vHeadDim, h] = [b,s,h]
                   TileShape::Current().SetCubeTile({16, 16},
                       {std::min(256, nq * vHeadDim), std::min(256, nq * vHeadDim)},
                       {std::min(128, h), std::min(128, h)});
                   Tensor bmm5_res = Matrix::Matmul<false, false>(dType, r2_res, w_o_i);

                   TileShape::Current().SetVecTile({4, std::min(8192, h)});
                   outputT = Reshape(bmm5_res, {b, s1, h});
               }
            }
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());

    std::vector<outDtype> q_golden(capacity_q);
    std::vector<outDtype> q_npu(capacity_q);
    std::vector<T> kv_golden(capacity_kv);
    std::vector<T> kv_npu(capacity_kv);

    readInput<outDtype>(dataPath + "/q_golden.bin", q_golden);
    readInput<T>(dataPath + "/kv_golden.bin", kv_golden);
    machine::GetRA()->CopyFromTensor((uint8_t *)q_npu.data(), (uint8_t *)q_out_ptr, outputSize0);
    machine::GetRA()->CopyFromTensor((uint8_t *)kv_npu.data(), (uint8_t *)past_key_states_ptr, outputSize1);

    std::cout << "\n====== resultCmp: output q start" << std::endl;
    int ret0 = resultCmp<outDtype>(q_golden, q_npu, 0.01f, 16);
    EXPECT_EQ(ret0, true);
    std::cout << "\n====== resultCmp: output kv start" << std::endl;
    int ret1 = resultCmp<T>(kv_golden, kv_npu, 0.003f, 16);
    EXPECT_EQ(ret1, true);

   std::vector<outDtype> q0_golden(q0_capacity);
   std::vector<outDtype> q0_npu(q0_capacity);
   readInput<outDtype>(dataPath + "/q0_golden.bin", q0_golden);
   machine::GetRA()->CopyFromTensor((uint8_t *)q0_npu.data(), (uint8_t *)q0_ptr, q0_size);
   std::cout << "\n====== resultCmp: output q0 start" << std::endl;
   int q0_ret = resultCmp<outDtype>(q0_golden, q0_npu, 0.01f, 16);
   EXPECT_EQ(q0_ret, true);

   std::vector<outDtype> q1_golden(q1_capacity);
   std::vector<outDtype> q1_npu(q1_capacity);
   readInput<outDtype>(dataPath + "/q1_golden.bin", q1_golden);
   machine::GetRA()->CopyFromTensor((uint8_t *)q1_npu.data(), (uint8_t *)q1_ptr, q1_size);
   std::cout << "\n====== resultCmp: output q1 start" << std::endl;
   int q1_ret = resultCmp<outDtype>(q1_golden, q1_npu, 0.005f, 16);
   EXPECT_EQ(q1_ret, true);

   std::vector<outDtype> k0_golden(k0_capacity);
   std::vector<outDtype> k0_npu(k0_capacity);
   readInput<outDtype>(dataPath + "/k0_golden.bin", k0_golden);
   machine::GetRA()->CopyFromTensor((uint8_t *)k0_npu.data(), (uint8_t *)k0_ptr, k0_size);
   std::cout << "\n====== resultCmp: output k0 start" << std::endl;
   int k0_ret = resultCmp<outDtype>(k0_golden, k0_npu, 0.005f, 16);
   EXPECT_EQ(k0_ret, true);

   std::vector<outDtype> k1_golden(k1_capacity);
   std::vector<outDtype> k1_npu(k1_capacity);
   readInput<outDtype>(dataPath + "/k1_golden.bin", k1_golden);
   machine::GetRA()->CopyFromTensor((uint8_t *)k1_npu.data(), (uint8_t *)k1_ptr, k1_size);
   std::cout << "\n====== resultCmp: output k1 start" << std::endl;
   int k1_ret = resultCmp<outDtype>(k1_golden, k1_npu, 0.005f, 16);
   EXPECT_EQ(k1_ret, true);


   std::vector<outDtype> v0_golden(v0_capacity);
   std::vector<outDtype> v0_npu(v0_capacity);
   readInput<outDtype>(dataPath + "/v0_golden.bin", v0_golden);
   machine::GetRA()->CopyFromTensor((uint8_t *)v0_npu.data(), (uint8_t *)v0_ptr, v0_size);
   std::cout << "\n====== resultCmp: output v0 start" << std::endl;
   int v0_ret = resultCmp<outDtype>(v0_golden, v0_npu, 0.005f, 16);
   EXPECT_EQ(v0_ret, true);

   std::cout << "\n====== resultCmp: output fa start" << std::endl;
   std::vector<float> golden(outCap);
   std::vector<float> res(outCap);
   machine::GetRA()->CopyFromTensor((uint8_t *)res.data(), (uint8_t *)outPtr, outputSize);
   readInput(GetGoldenDir() + "/atten_out.bin", golden);
   int ret2 = resultCmp(golden, res, 0.05f, 64);
   EXPECT_EQ(ret2, true);

   if(usePost){
       std::cout << "\n====== resultCmp: output post start" << std::endl;
       std::vector<T> postGolden(postOutputSize);
       std::vector<T> postRes(postOutputSize);
       machine::GetRA()->CopyFromTensor((uint8_t *)postRes.data(), (uint8_t *)out_ptr, outputByteSize);
       readInput<T>(GetGoldenDir() + "/attn_output.bin", postGolden);
       int ret = resultCmp<T>(postGolden, postRes, 0.08f, 500);
       EXPECT_EQ(ret, true);
   }
   if (skipReshape){
       config::SetPassConfig("PVC2_OOO", "SplitReshape", KEY_DISABLE_PASS, false);
   }
}

template <typename T = npu::tile_fwk::float16>
void attention_high(std::vector<int> &params, string dataPath, bool isQuant = false, bool fuse = true, bool usePost = true,
    bool skipReshape = true) {
    int b = params[0];
    int s1 = params[1];
    int s2 = params[2];
    int nq = params[3];
    int h = params[4];
    int q_lora_rank = params[5];
    int qk_nope_head_dim = params[6];
    int qk_rope_head_dim = params[7];
    int kv_lora_rank = params[8];
    int q_head_dim = qk_nope_head_dim + qk_rope_head_dim;
    int v_head_dim = params[9];
    const int nkv = 1;

    DataType dType = DT_FP32;
    if (std::is_same<T, npu::tile_fwk::float16>::value) {
        dType = DT_FP16;
    } else if (std::is_same<T, npu::tile_fwk::bfloat16>::value) {
        dType = DT_BF16;
    } else {
        dType = DT_FP32;
    }
    int dtypeSize = BytesOf(dType);

    std::vector<int> actSeqs(b, s2);
    const int blockSize = 256;
    const int nTile = nq;
    const float softmaxScale = 0.8f;
    IfaTileShapeConfig tileConfig;
    tileConfig.blockSize = blockSize;
    tileConfig.headNumQTile = nTile;
    tileConfig.v0TileShape = {32, 128};
    tileConfig.c1TileShape = {32, 32, 64, 64, 64, 64};
    tileConfig.v1TileShape = {32, 128};
    tileConfig.c2TileShape = {32, 32, 64, 64, 64, 64};
    tileConfig.v2TileShape = {32, 128};

    // 输出size
    int outCap = b * 1 * nq * kv_lora_rank;
    uint64_t outputSize = outCap * sizeof(float);
    uint8_t *outPtr = allocDevAddr(outputSize);

    // 根据Per Batch实际的sequence构造blockNum，blockNum >= Sum(blockNumPerBatch)，此处选取相等场景
    int blockNum = 0;
    for (auto s : actSeqs) {
        blockNum += CeilDiv(s, blockSize);
    }

    // 输入size
    int qNopeSize = b * s1 * nq * kv_lora_rank;
    int qRopeSize = b * s1 * nq * qk_rope_head_dim;

    // B B H
    int kNopeCacheSize = blockNum * blockSize * nkv * kv_lora_rank;
    int kRopeCacheSize = blockNum * blockSize * nkv * qk_rope_head_dim;
    int vNopeCacheSize = blockNum * blockSize * nkv * kv_lora_rank;

    DataType dTypeQuantIn = isQuant ? DT_INT8 : dType;
    // typedef float outDtype;
    typedef T outDtype;
    typedef int8_t wDtype;

    std::vector<int64_t> x_shape = {b, s1, h};
    std::vector<int64_t> w_qa_shape = {h, q_lora_rank};
    std::vector<int64_t> w_qb_shape = {q_lora_rank, nq * q_head_dim};
    std::vector<int64_t> w_kv_a_shape = {h, kv_lora_rank + qk_rope_head_dim};
    std::vector<int64_t> w_kv_b_k_shape = {nq, qk_nope_head_dim, kv_lora_rank};
    std::vector<int64_t> position_ids_shape = {b, s1};
    std::vector<int64_t> cos_shape = {s1, qk_rope_head_dim};
    std::vector<int64_t> past_key_states_shape = {b, 1, s2, kv_lora_rank + qk_rope_head_dim};
    std::vector<int64_t> kv_len_shape = {b, s1};
    // output
    std::vector<int64_t> q_shape = {b, nq, s1, kv_lora_rank + qk_rope_head_dim};
    std::vector<int64_t> kv_shape = {b, 1, s2, kv_lora_rank + qk_rope_head_dim};

    int capacity_x = std::accumulate(x_shape.begin(), x_shape.end(), 1, std::multiplies<>());
    int capacity_w_qa = std::accumulate(w_qa_shape.begin(), w_qa_shape.end(), 1, std::multiplies<>());
    int capacity_w_qb = std::accumulate(w_qb_shape.begin(), w_qb_shape.end(), 1, std::multiplies<>());
    int capacity_w_kv_a = std::accumulate(w_kv_a_shape.begin(), w_kv_a_shape.end(), 1, std::multiplies<>());
    int capacity_w_kv_b_k = std::accumulate(w_kv_b_k_shape.begin(), w_kv_b_k_shape.end(), 1, std::multiplies<>());
    int capacity_position_ids = std::accumulate(position_ids_shape.begin(), position_ids_shape.end(), 1,
        std::multiplies<>());
    int capacity_cos = std::accumulate(cos_shape.begin(), cos_shape.end(), 1, std::multiplies<>());
    int capacity_past_key_states = std::accumulate(past_key_states_shape.begin(), past_key_states_shape.end(), 1,
        std::multiplies<>());
    int capacity_kv_len = std::accumulate(kv_len_shape.begin(), kv_len_shape.end(), 1, std::multiplies<>());
    // output
    int capacity_q = std::accumulate(q_shape.begin(), q_shape.end(), 1, std::multiplies<>());
    int capacity_kv = std::accumulate(kv_shape.begin(), kv_shape.end(), 1, std::multiplies<>());


    int inputSize = b*nq*s1*kv_lora_rank;
    int wUvSize = nq*kv_lora_rank*v_head_dim;
    int wOSize = nq*v_head_dim*h;
    int postOutputSize = b*s1*h;
    int t1Size = b*s1*nq*kv_lora_rank;

    uint64_t outputByteSize = postOutputSize * dtypeSize;
    uint8_t* out_ptr = allocDevAddr(outputByteSize);

    std::vector<int64_t> w_qb_scale_shape;
    int capacity_w_qb_scale;
    if (isQuant) {
        w_qb_scale_shape = {1, nq * q_head_dim};
        capacity_w_qb_scale = std::accumulate(w_qb_scale_shape.begin(), w_qb_scale_shape.end(), 1, std::multiplies<>());
    }

    int q0_capacity = b * s1 * nq*kv_lora_rank;
    int q1_capacity = b * s1 * nq* qk_rope_head_dim;
    int k0_capacity = blockNum * blockSize * nkv* kv_lora_rank;
    int k1_capacity = blockNum * blockSize * nkv* qk_rope_head_dim;
    int v0_capacity = blockNum * blockSize * nkv* kv_lora_rank;

    int q0_size = q0_capacity* sizeof(T);
    int q1_size = q1_capacity* sizeof(T);
    int k0_size = k0_capacity* sizeof(T);
    int k1_size = k1_capacity* sizeof(T);
    int v0_size = v0_capacity* sizeof(T);

    uint8_t* q0_ptr = allocDevAddr(q0_size);
    uint8_t* q1_ptr = allocDevAddr(q1_size);
    uint8_t* k0_ptr = allocDevAddr(k0_size);
    uint8_t* k1_ptr = allocDevAddr(k1_size);
    uint8_t* v0_ptr = allocDevAddr(v0_size);

    aclInit(nullptr);
    rtSetDevice(GetDeviceIdByEnvVar());
    uint64_t outputSize0 = capacity_q * sizeof(T);
    uint64_t outputSize1 = capacity_kv * sizeof(T);
    uint8_t* q_out_ptr = allocDevAddr(outputSize0);
    void *past_key_states_ptr = readToDev<T>(dataPath + "/past_key_states.bin", capacity_past_key_states);


    ConfigManager::Instance();
    PROGRAM("MlaProlog") {
        void *x_ptr = readToDev<T>(dataPath + "/x.bin", capacity_x);
        void *w_qa_ptr = readToDev<T>(dataPath + "/w_qa.bin", capacity_w_qa);
        void *w_qb_ptr = isQuant ? readToDev<wDtype>(dataPath + "/w_qb.bin", capacity_w_qb) :
                                   readToDev<T>(dataPath + "/w_qb.bin", capacity_w_qb);
        void *w_kv_a_ptr = readToDev<T>(dataPath + "/w_kv_a.bin", capacity_w_kv_a);
        void *w_kv_b_k_ptr = readToDev<T>(dataPath + "/w_kv_b_k.bin", capacity_w_kv_b_k);
        void *position_ids_ptr = readToDev<int32_t>(dataPath + "/position_ids.bin", capacity_position_ids);
        void *cos_ptr = readToDev<T>(dataPath + "/cos.bin", capacity_cos);
        void *sin_ptr = readToDev<T>(dataPath + "/sin.bin", capacity_cos);
        void *kv_len_ptr = readToDev<int64_t>(dataPath + "/kv_len.bin", capacity_kv_len);

        Tensor x(dType, x_shape, (uint8_t *)x_ptr, "x");
        Tensor w_qa(dType, w_qa_shape, (uint8_t *)w_qa_ptr, "w_qa");
        Tensor w_qb(dTypeQuantIn, w_qb_shape, (uint8_t *)w_qb_ptr, "w_qb");
        Tensor w_kv_a(dType, w_kv_a_shape, (uint8_t *)w_kv_a_ptr, "w_kv_a");
        Tensor w_kv_b_k(dType, w_kv_b_k_shape, (uint8_t *)w_kv_b_k_ptr, "w_kv_b_k");
        Tensor position_ids(DT_INT32, position_ids_shape, (uint8_t *)position_ids_ptr, "position_ids");
        Tensor cos(dType, cos_shape, (uint8_t *)cos_ptr, "cos");
        Tensor sin(dType, cos_shape, (uint8_t *)sin_ptr, "sin");
        Tensor past_key_states(dType, past_key_states_shape, (uint8_t *)past_key_states_ptr, "past_key_states");
        Tensor kv_len(DT_INT64, kv_len_shape, (uint8_t *)kv_len_ptr, "kv_len");
        // output
        Tensor output_q(dType, q_shape, q_out_ptr, "output_q");

        AttentionW aw;
        aw.qAProjW = w_qa;
        aw.qBProjW = w_qb;
        aw.kvAProjWithMqaW = w_kv_a;
        aw.kvBProjWK = w_kv_b_k;
        Tensor kvBProjWV;  // not used in MlaProlog
        Tensor oProj2;       // not used in MlaProlog
        aw.kvBProjWV = kvBProjWV;
        aw.oProjW = oProj2;

        // 读数据
        void *qNopeData = readToDev(GetGoldenDir() + "/q_nope.bin", qNopeSize);
        void *qRopeData = readToDev(GetGoldenDir() + "/q_rope.bin", qRopeSize);
        void *kNopeCacheData = readToDev(GetGoldenDir() + "/k_cache_nope.bin", kNopeCacheSize);
        void *kRopeCacheData = readToDev(GetGoldenDir() + "/k_cache_rope.bin", kRopeCacheSize);
        void *vNopeCacheData = readToDev(GetGoldenDir() + "/v_cache.bin", vNopeCacheSize);


        Tensor qNope(DT_BF16, {b * s1 * nq, kv_lora_rank}, (uint8_t *)qNopeData, "qNope");
        Tensor qRope(DT_BF16, {b * s1 * nq, qk_rope_head_dim}, (uint8_t *)qRopeData, "qRope");
        Tensor kNopeCache(
            DT_BF16, {blockNum * blockSize * nkv, kv_lora_rank}, (uint8_t *)kNopeCacheData, "kNopeCache");
        Tensor kRopeCache(
            DT_BF16, {blockNum * blockSize * nkv, qk_rope_head_dim}, (uint8_t *)kRopeCacheData, "kRope");
        Tensor vNopeCache(
            DT_BF16, {blockNum * blockSize * nkv, kv_lora_rank}, (uint8_t *)vNopeCacheData, "vNopeCache");

        Tensor q0(DT_BF16, {b * s1 * nq, kv_lora_rank}, q0_ptr, "q0");
        Tensor q1(DT_BF16, {b * s1 * nq, qk_rope_head_dim}, q1_ptr, "q1");
        Tensor k0(DT_BF16, {blockNum * blockSize * nkv, kv_lora_rank}, k0_ptr, "k0");
        Tensor k1(DT_BF16, {blockNum * blockSize * nkv, qk_rope_head_dim}, k1_ptr, "k1");
        Tensor v0(DT_BF16, {blockNum * blockSize * nkv, kv_lora_rank}, v0_ptr, "v0");


        // blockTable: (b, maxBlockNumPerBatch)
        int maxSeqAllBatch = *(std::max_element(actSeqs.begin(), actSeqs.end()));
        int maxBlockNumPerBatch = CeilDiv(maxSeqAllBatch, blockSize);
        std::vector<std::vector<int>> blockTable(b, std::vector<int>(maxBlockNumPerBatch, 0));
        readBlockTable(GetGoldenDir() + "/block_table.bin", b, maxBlockNumPerBatch, blockTable);

        for (const auto& row : blockTable) {
            for (int num : row) {
                std::cout << num << " ";
            }
            std::cout << std::endl;
        }

        Tensor attentionOut(DT_FP32, {b * s1 * nq, kv_lora_rank}, outPtr, "attentionOut");

        std::vector<int64_t> inputShape = {b,nq,s1,kv_lora_rank};
        std::vector<int64_t> wUvShape = {nq,kv_lora_rank,v_head_dim};
        std::vector<int64_t> wOShape = {nq*v_head_dim,h};
        std::vector<int64_t> outputShapeT = {b, s1, h};
        std::vector<int64_t> t1Shape = {b, s1, nq, kv_lora_rank};
        void *input_ptr = readToDev<T>(GetGoldenDir() + "/input.bin", inputSize);
        void *w_uv_ptr = readToDev<T>(GetGoldenDir() + "/w_uv.bin", wUvSize);
        void *w_o_ptr = readToDev<T>(GetGoldenDir() + "/w_o.bin", wOSize);

        void *t1_ptr = readToDev<T>(GetGoldenDir() + "/t1.bin", t1Size);

        Tensor input_i(dType, inputShape, (uint8_t *)input_ptr, "A");
        Tensor w_uv_i(dType, wUvShape, (uint8_t *)w_uv_ptr, "B");
        Tensor w_o_i(dType, wOShape, (uint8_t *)w_o_ptr, "C");
        Tensor outputT(dType, outputShapeT, out_ptr, "D1");
        Tensor t1_i(dType, t1Shape, (uint8_t *)t1_ptr, "E");

        if (isQuant) {
            void *w_qb_scale_ptr = readToDev<float>(dataPath + "/w_qb_scale.bin", capacity_w_qb_scale);
            Tensor w_qb_scale = Tensor(DT_FP32, w_qb_scale_shape, (uint8_t *)w_qb_scale_ptr, "w_qb_scale");
            aw.qBProjWScale = w_qb_scale;

            std::tuple<Tensor, Tensor> res;
            DeepseekAttention attention(g_deepseekConfig, aw, 1);

            RoPETileShapeConfig ropeTileConfig{
                {32, 64}, // for cos/sin->cast, [s1,d]
                {1, 32, 64}, // for gather,unsqueeze, [b,s1,d]
                {1, 32, 1, 64}, // [b,nq,s1,d]
                {1, 32, 1, 64, 64} // for transpose, [b,nq,s1,d/2,2]
            };

            config::SetBuildStatic(true);
            FUNCTION("MlaProlog_T", {x, w_qa, w_qb, w_qb_scale, w_kv_a, w_kv_b_k, position_ids,
                                                                           cos, sin, past_key_states, kv_len, output_q}) {
                auto q_kv = attention.MlaPrologFoward(x, position_ids, cos, sin, kv_len, past_key_states, ropeTileConfig, isQuant);
                output_q = q_kv[0];
                past_key_states = q_kv[1];
            }
        } else {
            std::tuple<Tensor, Tensor> res;
            DeepseekAttention attention(g_deepseekConfig, aw, 1);

            RoPETileShapeConfig ropeTileConfig{
                {32, 64}, // for cos/sin->cast, [s1,d]
                {1, 32, 64}, // for gather,unsqueeze, [b,s1,d]
                {1, 32, 1, 64}, // [b,nq,s1,d]
                {1, 32, 1, 64, 64} // for transpose, [b,nq,s1,d/2,2]
            };

            config::SetBuildStatic(true);
            FUNCTION("MlaProlog_T",
                {x, w_qa, w_qb, w_kv_a, w_kv_b_k, position_ids, cos, sin, past_key_states, kv_len, output_q,
                    q0, q1, k0, k1, v0,
                    attentionOut,
                    qNope, kNopeCache, vNopeCache, qRope, kRopeCache,
                    input_i, t1_i, w_uv_i, w_o_i, outputT}) {
                if (skipReshape){
                    config::SetPassConfig("PVC2_OOO", "SplitReshape", KEY_DISABLE_PASS, true);
                }

                auto q_kv = attention.MlaPrologFoward(
                    x, position_ids, cos, sin, kv_len, past_key_states, ropeTileConfig, isQuant);
                output_q = q_kv[0];
                past_key_states = q_kv[1];
                q0 = q_kv[2];
                q1 = q_kv[3];

                auto tmp_k0 = View(past_key_states, {b, 1, s2, kv_lora_rank}, {0, 0, 0, 0});
                auto tmp32_k0 = Cast(tmp_k0, DT_FP32);
                k0 = Cast(tmp32_k0, DT_BF16);

                auto tmp_k1 = View(past_key_states, {b, 1, s2, qk_rope_head_dim}, {0, 0, 0, kv_lora_rank});
                auto tmp32_k1 = Cast(tmp_k1, DT_FP32);
                k1 = Cast(tmp32_k1, DT_BF16);

                auto tmp_v0 = View(past_key_states, {b, 1, s2, kv_lora_rank}, {0, 0, 0, 0});
                auto tmp32_v0 = Cast(tmp_v0, DT_FP32);
                v0 = Cast(tmp32_v0, DT_BF16);

                TileShape::Current().SetVecTile({2, 16, 1, qk_rope_head_dim});
                auto q0_1 = Reshape(q0, {b * s1 * nq, kv_lora_rank});
                auto q1_1 = Reshape(q1, {b * s1 * nq, qk_rope_head_dim});
                TileShape::Current().SetVecTile({2, 2, 128, qk_rope_head_dim});
                auto k0_1 = Reshape(k0, {blockNum * blockSize * nkv, kv_lora_rank});
                auto k1_1 = Reshape(k1, {blockNum * blockSize * nkv, qk_rope_head_dim});
                auto v0_1 = Reshape(v0, {blockNum * blockSize * nkv, kv_lora_rank});

                if (fuse) {
                    IncreFlashAttention(
                        q0_1, k0_1, v0_1, q1_1, k1_1, blockTable, actSeqs, softmaxScale, attentionOut, hightThroughputTileParams);
                } else {
                    IncreFlashAttention(qNope, kNopeCache, vNopeCache, qRope, kRopeCache, blockTable, actSeqs,
                        softmaxScale, attentionOut, tileConfig);
                }
                if (usePost) {
                    int B = b;
                    int S = s1;
                    int N = nq;
                    int H = h;
                    TileShape::Current().SetVecTile({4, 16, 1, kv_lora_rank});
                    Tensor atten_res0 = Transpose(input_i, {1, 2});
                    TileShape::Current().SetVecTile({4, 1, 32, kv_lora_rank});
                    Tensor atten_res1 = Reshape(atten_res0, {B * S, N, kv_lora_rank});
                    TileShape::Current().SetVecTile({4, 16, kv_lora_rank});
                    Tensor t2_res = Transpose(atten_res1, {0, 1});

                    TileShape::Current().SetCubeTile({32, 32},
                        {std::min(256, kv_lora_rank), std::min(256, kv_lora_rank)},
                        {std::min(128, v_head_dim), std::min(128, v_head_dim)}); // M 16对齐
                    // [n,bs,kv_lora_rank] * [n, kv_lora_rank, v_head_dim] = [n,bs,v_head_dim]
                    Tensor bmm4_res = Matrix::BatchMatmul(dType, t2_res, w_uv_i);

                    TileShape::Current().SetVecTile(32, 4, v_head_dim); // 必须切，但是尾轴不能切
                    Tensor t3_res = Transpose(bmm4_res, {0, 1}); // [bs,n,v_head_dim]

                    TileShape::Current().SetVecTile({4, 32, v_head_dim});
                    Tensor r2_res = Reshape(t3_res, {B * S, N*v_head_dim});

                    // [b,s, n*v_head_dim] @ [n*v_head_dim, h] = [b,s,h]
                    TileShape::Current().SetCubeTile({32, 32},
                        {std::min(256, N * v_head_dim), std::min(256, N * v_head_dim)},
                        {std::min(128, H), std::min(128, H)});
                    Tensor bmm5_res = Matrix::Matmul<false, false>(dType, r2_res, w_o_i);

                    TileShape::Current().SetVecTile({32, std::min(2048, H)});
                    outputT = Reshape(bmm5_res, {B, S, H});
                }
            }
        }
    }
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction());

    std::vector<outDtype> q_golden(capacity_q);
    std::vector<outDtype> q_npu(capacity_q);
    std::vector<T> kv_golden(capacity_kv);
    std::vector<T> kv_npu(capacity_kv);

    readInput<outDtype>(dataPath + "/q_golden.bin", q_golden);
    readInput<T>(dataPath + "/kv_golden.bin", kv_golden);
    machine::GetRA()->CopyFromTensor((uint8_t *)q_npu.data(), (uint8_t *)q_out_ptr, outputSize0);
    machine::GetRA()->CopyFromTensor((uint8_t *)kv_npu.data(), (uint8_t *)past_key_states_ptr, outputSize1);

    std::cout << "\n====== resultCmp: output q start" << std::endl;
    int ret0 = resultCmp<outDtype>(q_golden, q_npu, 0.005f, 16);
    EXPECT_EQ(ret0, true);
    std::cout << "\n====== resultCmp: output kv start" << std::endl;
    int ret1 = resultCmp<T>(kv_golden, kv_npu, 0.003f, 16);
    EXPECT_EQ(ret1, true);

    std::vector<outDtype> q0_golden(q0_capacity);
    std::vector<outDtype> q0_npu(q0_capacity);
    readInput<outDtype>(dataPath + "/q0_golden.bin", q0_golden);
    machine::GetRA()->CopyFromTensor((uint8_t *)q0_npu.data(), (uint8_t *)q0_ptr, q0_size);
    std::cout << "\n====== resultCmp: output q0 start" << std::endl;
    int q0_ret = resultCmp<outDtype>(q0_golden, q0_npu, 0.005f, 16);
    EXPECT_EQ(q0_ret, true);

    std::vector<outDtype> q1_golden(q1_capacity);
    std::vector<outDtype> q1_npu(q1_capacity);
    readInput<outDtype>(dataPath + "/q1_golden.bin", q1_golden);
    machine::GetRA()->CopyFromTensor((uint8_t *)q1_npu.data(), (uint8_t *)q1_ptr, q1_size);
    std::cout << "\n====== resultCmp: output q1 start" << std::endl;
    int q1_ret = resultCmp<outDtype>(q1_golden, q1_npu, 0.005f, 16);
    EXPECT_EQ(q1_ret, true);

    std::vector<outDtype> k0_golden(k0_capacity);
    std::vector<outDtype> k0_npu(k0_capacity);
    readInput<outDtype>(dataPath + "/k0_golden.bin", k0_golden);
    machine::GetRA()->CopyFromTensor((uint8_t *)k0_npu.data(), (uint8_t *)k0_ptr, k0_size);
    std::cout << "\n====== resultCmp: output k0 start" << std::endl;
    int k0_ret = resultCmp<outDtype>(k0_golden, k0_npu, 0.005f, 16);
    EXPECT_EQ(k0_ret, true);

    std::vector<outDtype> k1_golden(k1_capacity);
    std::vector<outDtype> k1_npu(k1_capacity);
    readInput<outDtype>(dataPath + "/k1_golden.bin", k1_golden);
    machine::GetRA()->CopyFromTensor((uint8_t *)k1_npu.data(), (uint8_t *)k1_ptr, k1_size);
    std::cout << "\n====== resultCmp: output k1 start" << std::endl;
    int k1_ret = resultCmp<outDtype>(k1_golden, k1_npu, 0.005f, 16);
    EXPECT_EQ(k1_ret, true);


    std::vector<outDtype> v0_golden(v0_capacity);
    std::vector<outDtype> v0_npu(v0_capacity);
    readInput<outDtype>(dataPath + "/v0_golden.bin", v0_golden);
    machine::GetRA()->CopyFromTensor((uint8_t *)v0_npu.data(), (uint8_t *)v0_ptr, v0_size);
    std::cout << "\n====== resultCmp: output v0 start" << std::endl;
    int v0_ret = resultCmp<outDtype>(v0_golden, v0_npu, 0.005f, 16);
    EXPECT_EQ(v0_ret, true);

    std::cout << "\n====== resultCmp: output fa start" << std::endl;
    std::vector<float> golden(outCap);
    std::vector<float> res(outCap);
    machine::GetRA()->CopyFromTensor((uint8_t *)res.data(), (uint8_t *)outPtr, outputSize);
    readInput(GetGoldenDir() + "/atten_out.bin", golden);
    int ret2 = resultCmp(golden, res, 0.01f, 16);
    EXPECT_EQ(ret2, true);

    if(usePost){
        std::cout << "\n====== resultCmp: output post start" << std::endl;
        std::vector<T> postGolden(postOutputSize);
        std::vector<T> postRes(postOutputSize);
        machine::GetRA()->CopyFromTensor((uint8_t *)postRes.data(), (uint8_t *)out_ptr, outputByteSize);
        readInput<T>(GetGoldenDir() + "/attn_output.bin", postGolden);
        int ret = resultCmp<T>(postGolden, postRes, 0.02f, 500);
        EXPECT_EQ(ret, true);
    }
    if (skipReshape){
        config::SetPassConfig("PVC2_OOO", "SplitReshape", KEY_DISABLE_PASS, false);
    }
}

TEST_F(MlaPrologOnBoardTest, test_MlaProlog_float16_32_2_1_256_256_512) {  // b_n_s_s2_h_q_lora_rank
    int& h = std::get<int>(g_deepseekConfig["hiddenSize"]);
    int& n = std::get<int>(g_deepseekConfig["numAttentionHeads"]);
    int& qLoraRank = std::get<int>(g_deepseekConfig["qLoraRank"]);
    int& qkRopeHeadDim = std::get<int>(g_deepseekConfig["qkRopeHeadDim"]);
    int& kvLoraRank = std::get<int>(g_deepseekConfig["kvLoraRank"]);
    int& vHeadDim = std::get<int>(g_deepseekConfig["vHeadDim"]);
    int& qkNopeHeadDim = std::get<int>(g_deepseekConfig["qkNopeHeadDim"]);

    int b = 32;
    int s = 1;
    int s2 = 256;
    h = 256;  //
    n = 2;  //
    qLoraRank = 512;  //
    qkNopeHeadDim = 128;
    qkRopeHeadDim = 64;
    kvLoraRank = 512;
    vHeadDim = 128;

    std::vector<int> params = {b, s, s2, n, h, qLoraRank, qkNopeHeadDim, qkRopeHeadDim,
                               kvLoraRank, vHeadDim};
    TestMlaProlog<npu::tile_fwk::float16>(params, GetGoldenDir());
}

TEST_F(MlaPrologOnBoardTest, test_MlaProlog_float16_32_32_1_256_1024_1536) {  // b_n_s_s2_h_q_lora_rank
    int& h = std::get<int>(g_deepseekConfig["hiddenSize"]);
    int& n = std::get<int>(g_deepseekConfig["numAttentionHeads"]);
    int& qLoraRank = std::get<int>(g_deepseekConfig["qLoraRank"]);
    int& qkRopeHeadDim = std::get<int>(g_deepseekConfig["qkRopeHeadDim"]);
    int& kvLoraRank = std::get<int>(g_deepseekConfig["kvLoraRank"]);
    int& vHeadDim = std::get<int>(g_deepseekConfig["vHeadDim"]);
    int& qkNopeHeadDim = std::get<int>(g_deepseekConfig["qkNopeHeadDim"]);

    int b = 32;
    int s = 1;
    int s2 = 256;
    h = 1024;  //
    n = 32;
    qLoraRank = 1536;
    qkNopeHeadDim = 128;
    qkRopeHeadDim = 64;
    kvLoraRank = 512;
    vHeadDim = 128;

    std::vector<int> params = {b, s, s2, n, h, qLoraRank, qkNopeHeadDim, qkRopeHeadDim,
                               kvLoraRank, vHeadDim};
    TestMlaProlog<npu::tile_fwk::float16>(params, GetGoldenDir());
}

TEST_F(MlaPrologOnBoardTest, test_MlaProlog_bfloat16_32_32_1_256_1024_1536) {  // b_n_s_s2_h_q_lora_rank, bfloat16
    int& h = std::get<int>(g_deepseekConfig["hiddenSize"]);
    int& n = std::get<int>(g_deepseekConfig["numAttentionHeads"]);
    int& qLoraRank = std::get<int>(g_deepseekConfig["qLoraRank"]);
    int& qkRopeHeadDim = std::get<int>(g_deepseekConfig["qkRopeHeadDim"]);
    int& kvLoraRank = std::get<int>(g_deepseekConfig["kvLoraRank"]);
    int& vHeadDim = std::get<int>(g_deepseekConfig["vHeadDim"]);
    int& qkNopeHeadDim = std::get<int>(g_deepseekConfig["qkNopeHeadDim"]);

    int b = 32;
    int s = 1;
    int s2 = 256;
    h = 1024;  //
    n = 32;
    qLoraRank = 1536;
    qkNopeHeadDim = 128;
    qkRopeHeadDim = 64;
    kvLoraRank = 512;
    vHeadDim = 128;

    std::vector<int> params = {b, s, s2, n, h, qLoraRank, qkNopeHeadDim, qkRopeHeadDim,
                               kvLoraRank, vHeadDim};
    TestMlaProlog<npu::tile_fwk::bfloat16>(params, GetGoldenDir());
}

TEST_F(MlaPrologOnBoardTest, test_MlaProlog_float16_32_32_1_256_7168_1536) {  // b_n_s_s2_h_q_lora_rank
    int& h = std::get<int>(g_deepseekConfig["hiddenSize"]);
    int& n = std::get<int>(g_deepseekConfig["numAttentionHeads"]);
    int& qLoraRank = std::get<int>(g_deepseekConfig["qLoraRank"]);
    int& qkRopeHeadDim = std::get<int>(g_deepseekConfig["qkRopeHeadDim"]);
    int& kvLoraRank = std::get<int>(g_deepseekConfig["kvLoraRank"]);
    int& vHeadDim = std::get<int>(g_deepseekConfig["vHeadDim"]);
    int& qkNopeHeadDim = std::get<int>(g_deepseekConfig["qkNopeHeadDim"]);

    int b = 32;
    int s = 1;
    int s2 = 256;
    h = 7168;
    n = 32;
    qLoraRank = 1536;
    qkNopeHeadDim = 128;
    qkRopeHeadDim = 64;
    kvLoraRank = 512;
    vHeadDim = 128;

    std::vector<int> params = {b, s, s2, n, h, qLoraRank, qkNopeHeadDim, qkRopeHeadDim,
                               kvLoraRank, vHeadDim};
    TestMlaProlog<npu::tile_fwk::float16>(params, GetGoldenDir());
}

TEST_F(MlaPrologOnBoardTest, test_MlaProlog_bfloat16_32_32_1_256_7168_1536) {  // b_n_s_s2_h_q_lora_rank, bfloat16
    int& h = std::get<int>(g_deepseekConfig["hiddenSize"]);
    int& n = std::get<int>(g_deepseekConfig["numAttentionHeads"]);
    int& qLoraRank = std::get<int>(g_deepseekConfig["qLoraRank"]);
    int& qkRopeHeadDim = std::get<int>(g_deepseekConfig["qkRopeHeadDim"]);
    int& kvLoraRank = std::get<int>(g_deepseekConfig["kvLoraRank"]);
    int& vHeadDim = std::get<int>(g_deepseekConfig["vHeadDim"]);
    int& qkNopeHeadDim = std::get<int>(g_deepseekConfig["qkNopeHeadDim"]);

    int b = 32;
    int s = 1;
    int s2 = 256;
    h = 7168;
    n = 32;
    qLoraRank = 1536;
    qkNopeHeadDim = 128;
    qkRopeHeadDim = 64;
    kvLoraRank = 512;
    vHeadDim = 128;

    std::vector<int> params = {b, s, s2, n, h, qLoraRank, qkNopeHeadDim, qkRopeHeadDim,
                               kvLoraRank, vHeadDim};
    TestMlaProlog<npu::tile_fwk::bfloat16>(params, GetGoldenDir());
}

TEST_F(MlaPrologOnBoardTest, test_MlaProlog_float16_32_128_1_4096_7168_1536) {  // b_n_s_s2_h_q_lora_rank
    int& h = std::get<int>(g_deepseekConfig["hiddenSize"]);
    int& n = std::get<int>(g_deepseekConfig["numAttentionHeads"]);
    int& qLoraRank = std::get<int>(g_deepseekConfig["qLoraRank"]);
    int& qkRopeHeadDim = std::get<int>(g_deepseekConfig["qkRopeHeadDim"]);
    int& kvLoraRank = std::get<int>(g_deepseekConfig["kvLoraRank"]);
    int& vHeadDim = std::get<int>(g_deepseekConfig["vHeadDim"]);
    int& qkNopeHeadDim = std::get<int>(g_deepseekConfig["qkNopeHeadDim"]);

    int b = 32;
    int s = 1;
    int s2 = 4096;
    h = 7168;
    n = 128;
    qLoraRank = 1536;
    qkNopeHeadDim = 128;
    qkRopeHeadDim = 64;
    kvLoraRank = 512;
    vHeadDim = 128;

    std::vector<int> params = {b, s, s2, n, h, qLoraRank, qkNopeHeadDim, qkRopeHeadDim,
                               kvLoraRank, vHeadDim};
    TestMlaProlog<npu::tile_fwk::float16>(params, GetGoldenDir());
}

TEST_F(MlaPrologOnBoardTest, test_MlaProlog_bfloat16_32_128_1_256_7168_1536) {  // b_n_s_s2_h_q_lora_rank, bfloat16
    int& h = std::get<int>(g_deepseekConfig["hiddenSize"]);
    int& n = std::get<int>(g_deepseekConfig["numAttentionHeads"]);
    int& qLoraRank = std::get<int>(g_deepseekConfig["qLoraRank"]);
    int& qkRopeHeadDim = std::get<int>(g_deepseekConfig["qkRopeHeadDim"]);
    int& kvLoraRank = std::get<int>(g_deepseekConfig["kvLoraRank"]);
    int& vHeadDim = std::get<int>(g_deepseekConfig["vHeadDim"]);
    int& qkNopeHeadDim = std::get<int>(g_deepseekConfig["qkNopeHeadDim"]);

    int b = 32;
    int s = 1;
    int s2 = 256;
    h = 7168;
    n = 128;
    qLoraRank = 1536;
    qkNopeHeadDim = 128;
    qkRopeHeadDim = 64;
    kvLoraRank = 512;
    vHeadDim = 128;

    std::vector<int> params = {b, s, s2, n, h, qLoraRank, qkNopeHeadDim, qkRopeHeadDim,
                               kvLoraRank, vHeadDim};
    TestMlaProlog<npu::tile_fwk::bfloat16>(params, GetGoldenDir());
}

TEST_F(MlaPrologOnBoardTest, test_MlaProlog_bfloat16_32_128_1_4096_7168_1536) {  // b_n_s_s2_h_q_lora_rank, bfloat16
    int& h = std::get<int>(g_deepseekConfig["hiddenSize"]);
    int& n = std::get<int>(g_deepseekConfig["numAttentionHeads"]);
    int& qLoraRank = std::get<int>(g_deepseekConfig["qLoraRank"]);
    int& qkRopeHeadDim = std::get<int>(g_deepseekConfig["qkRopeHeadDim"]);
    int& kvLoraRank = std::get<int>(g_deepseekConfig["kvLoraRank"]);
    int& vHeadDim = std::get<int>(g_deepseekConfig["vHeadDim"]);
    int& qkNopeHeadDim = std::get<int>(g_deepseekConfig["qkNopeHeadDim"]);

    int b = 32;
    int s = 1;
    int s2 = 4096;
    h = 7168;
    n = 128;
    qLoraRank = 1536;
    qkNopeHeadDim = 128;
    qkRopeHeadDim = 64;
    kvLoraRank = 512;
    vHeadDim = 128;

    std::vector<int> params = {b, s, s2, n, h, qLoraRank, qkNopeHeadDim, qkRopeHeadDim,
                               kvLoraRank, vHeadDim};
    TestMlaProlog<npu::tile_fwk::bfloat16>(params, GetGoldenDir());
}

TEST_F(MlaPrologOnBoardTest, test_MlaProlog_bfloat16_4_32_1_256_7168_1536) {  // b_n_s_s2_h_q_lora_rank, bfloat16
    int& h = std::get<int>(g_deepseekConfig["hiddenSize"]);
    int& n = std::get<int>(g_deepseekConfig["numAttentionHeads"]);
    int& qLoraRank = std::get<int>(g_deepseekConfig["qLoraRank"]);
    int& qkRopeHeadDim = std::get<int>(g_deepseekConfig["qkRopeHeadDim"]);
    int& kvLoraRank = std::get<int>(g_deepseekConfig["kvLoraRank"]);
    int& vHeadDim = std::get<int>(g_deepseekConfig["vHeadDim"]);
    int& qkNopeHeadDim = std::get<int>(g_deepseekConfig["qkNopeHeadDim"]);

    int b = 4;
    int s = 1;
    int s2 = 256;
    h = 7168;
    n = 32;
    qLoraRank = 1536;
    qkNopeHeadDim = 128;
    qkRopeHeadDim = 64;
    kvLoraRank = 512;
    vHeadDim = 128;

    std::vector<int> params = {b, s, s2, n, h, qLoraRank, qkNopeHeadDim, qkRopeHeadDim,
                               kvLoraRank, vHeadDim};
    TestMlaProlog<npu::tile_fwk::bfloat16>(params, GetGoldenDir());
}

TEST_F(MlaPrologOnBoardTest, test_MlaProlog_bfloat16_4_32_1_4096_7168_1536) {  // b_n_s_s2_h_q_lora_rank, bfloat16
    int& h = std::get<int>(g_deepseekConfig["hiddenSize"]);
    int& n = std::get<int>(g_deepseekConfig["numAttentionHeads"]);
    int& qLoraRank = std::get<int>(g_deepseekConfig["qLoraRank"]);
    int& qkRopeHeadDim = std::get<int>(g_deepseekConfig["qkRopeHeadDim"]);
    int& kvLoraRank = std::get<int>(g_deepseekConfig["kvLoraRank"]);
    int& vHeadDim = std::get<int>(g_deepseekConfig["vHeadDim"]);
    int& qkNopeHeadDim = std::get<int>(g_deepseekConfig["qkNopeHeadDim"]);

    int b = 4;
    int s = 1;
    int s2 = 4096;
    h = 7168;
    n = 32;
    qLoraRank = 1536;
    qkNopeHeadDim = 128;
    qkRopeHeadDim = 64;
    kvLoraRank = 512;
    vHeadDim = 128;

    std::vector<int> params = {b, s, s2, n, h, qLoraRank, qkNopeHeadDim, qkRopeHeadDim,
                               kvLoraRank, vHeadDim};
    TestMlaProlog<npu::tile_fwk::bfloat16>(params, GetGoldenDir());
}

TEST_F(MlaPrologOnBoardTest, test_MlaProlog_bfloat16_4_128_1_256_7168_1536) {  // b_n_s_s2_h_q_lora_rank, bfloat16
    int& h = std::get<int>(g_deepseekConfig["hiddenSize"]);
    int& n = std::get<int>(g_deepseekConfig["numAttentionHeads"]);
    int& qLoraRank = std::get<int>(g_deepseekConfig["qLoraRank"]);
    int& qkRopeHeadDim = std::get<int>(g_deepseekConfig["qkRopeHeadDim"]);
    int& kvLoraRank = std::get<int>(g_deepseekConfig["kvLoraRank"]);
    int& vHeadDim = std::get<int>(g_deepseekConfig["vHeadDim"]);
    int& qkNopeHeadDim = std::get<int>(g_deepseekConfig["qkNopeHeadDim"]);

    int b = 4;
    int s = 1;
    int s2 = 256;
    h = 7168;
    n = 128;
    qLoraRank = 1536;
    qkNopeHeadDim = 128;
    qkRopeHeadDim = 64;
    kvLoraRank = 512;
    vHeadDim = 128;

    std::vector<int> params = {b, s, s2, n, h, qLoraRank, qkNopeHeadDim, qkRopeHeadDim,
                               kvLoraRank, vHeadDim};
    TestMlaProlog<npu::tile_fwk::bfloat16>(params, GetGoldenDir());
}

TEST_F(MlaPrologOnBoardTest, test_MlaProlog_bfloat16_4_128_1_4096_7168_1536) {  // b_n_s_s2_h_q_lora_rank, bfloat16
    int& h = std::get<int>(g_deepseekConfig["hiddenSize"]);
    int& n = std::get<int>(g_deepseekConfig["numAttentionHeads"]);
    int& qLoraRank = std::get<int>(g_deepseekConfig["qLoraRank"]);
    int& qkRopeHeadDim = std::get<int>(g_deepseekConfig["qkRopeHeadDim"]);
    int& kvLoraRank = std::get<int>(g_deepseekConfig["kvLoraRank"]);
    int& vHeadDim = std::get<int>(g_deepseekConfig["vHeadDim"]);
    int& qkNopeHeadDim = std::get<int>(g_deepseekConfig["qkNopeHeadDim"]);

    int b = 4;
    int s = 1;
    int s2 = 4096;
    h = 7168;
    n = 128;
    qLoraRank = 1536;
    qkNopeHeadDim = 128;
    qkRopeHeadDim = 64;
    kvLoraRank = 512;
    vHeadDim = 128;

    std::vector<int> params = {b, s, s2, n, h, qLoraRank, qkNopeHeadDim, qkRopeHeadDim,
                               kvLoraRank, vHeadDim};
    TestMlaProlog<npu::tile_fwk::bfloat16>(params, GetGoldenDir());
}

TEST_F(MlaPrologOnBoardTest, test_MlaProlog_float16_32_2_1_256_256_512_quant) {  // b_n_s_s2_h_q_lora_rank
    int& h = std::get<int>(g_deepseekConfig["hiddenSize"]);
    int& n = std::get<int>(g_deepseekConfig["numAttentionHeads"]);
    int& qLoraRank = std::get<int>(g_deepseekConfig["qLoraRank"]);
    int& qkRopeHeadDim = std::get<int>(g_deepseekConfig["qkRopeHeadDim"]);
    int& kvLoraRank = std::get<int>(g_deepseekConfig["kvLoraRank"]);
    int& vHeadDim = std::get<int>(g_deepseekConfig["vHeadDim"]);
    int& qkNopeHeadDim = std::get<int>(g_deepseekConfig["qkNopeHeadDim"]);

    int b = 32;
    int s = 1;
    int s2 = 256;
    h = 256;  //
    n = 2;  //
    qLoraRank = 512;  //
    qkNopeHeadDim = 128;
    qkRopeHeadDim = 64;
    kvLoraRank = 512;
    vHeadDim = 128;

    std::vector<int> params = {b, s, s2, n, h, qLoraRank, qkNopeHeadDim, qkRopeHeadDim,
                               kvLoraRank, vHeadDim};
    TestMlaProlog<npu::tile_fwk::float16>(params, GetGoldenDir(), true);
}

TEST_F(MlaPrologOnBoardTest, test_MlaProlog_float16_32_32_1_256_7168_1536_quant) {  // b_n_s_s2_h_q_lora_rank
    int& h = std::get<int>(g_deepseekConfig["hiddenSize"]);
    int& n = std::get<int>(g_deepseekConfig["numAttentionHeads"]);
    int& qLoraRank = std::get<int>(g_deepseekConfig["qLoraRank"]);
    int& qkRopeHeadDim = std::get<int>(g_deepseekConfig["qkRopeHeadDim"]);
    int& kvLoraRank = std::get<int>(g_deepseekConfig["kvLoraRank"]);
    int& vHeadDim = std::get<int>(g_deepseekConfig["vHeadDim"]);
    int& qkNopeHeadDim = std::get<int>(g_deepseekConfig["qkNopeHeadDim"]);

    int b = 32;
    int s = 1;
    int s2 = 256;
    h = 7168;
    n = 32;
    qLoraRank = 1536;
    qkNopeHeadDim = 128;
    qkRopeHeadDim = 64;
    kvLoraRank = 512;
    vHeadDim = 128;

    std::vector<int> params = {b, s, s2, n, h, qLoraRank, qkNopeHeadDim, qkRopeHeadDim,
                               kvLoraRank, vHeadDim};
    TestMlaProlog<npu::tile_fwk::float16>(params, GetGoldenDir(), true);
}

TEST_F(MlaPrologOnBoardTest, test_MlaProlog_bfloat16_4_32_1_256_7168_1536_quant) {  // b_n_s_s2_h_q_lora_rank, bfloat16
    int& h = std::get<int>(g_deepseekConfig["hiddenSize"]);
    int& n = std::get<int>(g_deepseekConfig["numAttentionHeads"]);
    int& qLoraRank = std::get<int>(g_deepseekConfig["qLoraRank"]);
    int& qkRopeHeadDim = std::get<int>(g_deepseekConfig["qkRopeHeadDim"]);
    int& kvLoraRank = std::get<int>(g_deepseekConfig["kvLoraRank"]);
    int& vHeadDim = std::get<int>(g_deepseekConfig["vHeadDim"]);
    int& qkNopeHeadDim = std::get<int>(g_deepseekConfig["qkNopeHeadDim"]);

    int b = 4;
    int s = 1;
    int s2 = 256;
    h = 7168;
    n = 32;
    qLoraRank = 1536;
    qkNopeHeadDim = 128;
    qkRopeHeadDim = 64;
    kvLoraRank = 512;
    vHeadDim = 128;

    std::vector<int> params = {b, s, s2, n, h, qLoraRank, qkNopeHeadDim, qkRopeHeadDim,
                               kvLoraRank, vHeadDim};
    TestMlaProlog<npu::tile_fwk::bfloat16>(params, GetGoldenDir(), true);
}

TEST_F(MlaPrologOnBoardTest, test_MlaProlog_float16_32_128_1_4096_7168_1536_quant) {  // b_n_s_s2_h_q_lora_rank
    // config::SetPassOption(NBUFFER_NUM, 2);
    config::SetPassOption(VEC_NBUFFER_MODE, 1);
    config::SetPassOption(CUBE_L1_REUSE_MODE, 4);
    int& h = std::get<int>(g_deepseekConfig["hiddenSize"]);
    int& n = std::get<int>(g_deepseekConfig["numAttentionHeads"]);
    int& qLoraRank = std::get<int>(g_deepseekConfig["qLoraRank"]);
    int& qkRopeHeadDim = std::get<int>(g_deepseekConfig["qkRopeHeadDim"]);
    int& kvLoraRank = std::get<int>(g_deepseekConfig["kvLoraRank"]);
    int& vHeadDim = std::get<int>(g_deepseekConfig["vHeadDim"]);
    int& qkNopeHeadDim = std::get<int>(g_deepseekConfig["qkNopeHeadDim"]);

    int b = 32;
    int s = 1;
    int s2 = 4096;
    h = 7168;
    n = 128;
    qLoraRank = 1536;
    qkNopeHeadDim = 128;
    qkRopeHeadDim = 64;
    kvLoraRank = 512;
    vHeadDim = 128;

    std::vector<int> params = {b, s, s2, n, h, qLoraRank, qkNopeHeadDim, qkRopeHeadDim,
                               kvLoraRank, vHeadDim};
    TestMlaProlog<npu::tile_fwk::float16>(params, GetGoldenDir(), true);
}

TEST_F(MlaPrologOnBoardTest, test_MlaProlog_float16_2_32_1_4096_7168_1536) {  // b_n_s_s2_h_q_lora_rank
    config::SetPassOption(VEC_NBUFFER_MODE, 2);
    config::SetPassOption(VEC_NBUFFER_SETTING,  std::map<int64_t, int64_t>{{-1, 2}});
    int& h = std::get<int>(g_deepseekConfig["hiddenSize"]);
    int& n = std::get<int>(g_deepseekConfig["numAttentionHeads"]);
    int& qLoraRank = std::get<int>(g_deepseekConfig["qLoraRank"]);
    int& qkRopeHeadDim = std::get<int>(g_deepseekConfig["qkRopeHeadDim"]);
    int& kvLoraRank = std::get<int>(g_deepseekConfig["kvLoraRank"]);
    int& vHeadDim = std::get<int>(g_deepseekConfig["vHeadDim"]);
    int& qkNopeHeadDim = std::get<int>(g_deepseekConfig["qkNopeHeadDim"]);

    int b = 4;
    int s = 1;
    int s2 = 4096;
    h = 7168;
    n = 32;
    qLoraRank = 1536;
    qkNopeHeadDim = 128;
    qkRopeHeadDim = 64;
    kvLoraRank = 512;
    vHeadDim = 128;

    std::vector<int> params = {b, s, s2, n, h, qLoraRank, qkNopeHeadDim, qkRopeHeadDim,
        kvLoraRank, vHeadDim};
    Attention<npu::tile_fwk::bfloat16>(params, GetGoldenDir(), false);
}

TEST_F(MlaPrologOnBoardTest, attention_bf16_test) {  // b_n_s_s2_h_q_lora_rank
    config::SetPassOption(VEC_NBUFFER_MODE, 2);
    config::SetPassOption(VEC_NBUFFER_SETTING,  std::map<int64_t, int64_t>{{-1, 2}});
    int& h = std::get<int>(g_deepseekConfig["hiddenSize"]);
    int& n = std::get<int>(g_deepseekConfig["numAttentionHeads"]);
    int& qLoraRank = std::get<int>(g_deepseekConfig["qLoraRank"]);
    int& qkRopeHeadDim = std::get<int>(g_deepseekConfig["qkRopeHeadDim"]);
    int& kvLoraRank = std::get<int>(g_deepseekConfig["kvLoraRank"]);
    int& vHeadDim = std::get<int>(g_deepseekConfig["vHeadDim"]);
    int& qkNopeHeadDim = std::get<int>(g_deepseekConfig["qkNopeHeadDim"]);

    int b = 4;
    int s = 1;
    int s2 = 256;
    h = 128;
    n = 32;
    qLoraRank = 128;
    qkNopeHeadDim = 128;
    qkRopeHeadDim = 64;
    kvLoraRank = 512;
    vHeadDim = 128;

    std::vector<int> params = {b, s, s2, n, h, qLoraRank, qkNopeHeadDim, qkRopeHeadDim,
        kvLoraRank, vHeadDim};
    Attention<npu::tile_fwk::bfloat16>(params, GetGoldenDir(), false);
}

TEST_F(MlaPrologOnBoardTest, attention_bf16_high) {  // b_n_s_s2_h_q_lora_rank
    config::SetPassOption(VEC_NBUFFER_MODE, 2);
    config::SetPassOption(VEC_NBUFFER_SETTING,  std::map<int64_t, int64_t>{{-1, 2}});
    int& h = std::get<int>(g_deepseekConfig["hiddenSize"]);
    int& n = std::get<int>(g_deepseekConfig["numAttentionHeads"]);
    int& qLoraRank = std::get<int>(g_deepseekConfig["qLoraRank"]);
    int& qkRopeHeadDim = std::get<int>(g_deepseekConfig["qkRopeHeadDim"]);
    int& kvLoraRank = std::get<int>(g_deepseekConfig["kvLoraRank"]);
    int& vHeadDim = std::get<int>(g_deepseekConfig["vHeadDim"]);
    int& qkNopeHeadDim = std::get<int>(g_deepseekConfig["qkNopeHeadDim"]);

    int b = 32;
    int s = 1;
    int s2 = 4096;
    h = 7168;
    n = 128;
    qLoraRank = 1536;
    qkNopeHeadDim = 128;
    qkRopeHeadDim = 64;
    kvLoraRank = 512;
    vHeadDim = 128;

    std::vector<int> params = {b, s, s2, n, h, qLoraRank, qkNopeHeadDim, qkRopeHeadDim,
        kvLoraRank, vHeadDim};
    attention_high<npu::tile_fwk::bfloat16>(params, GetGoldenDir(), false);
}

TEST_F(MlaPrologOnBoardTest, attention_bf16_low) {  // b_n_s_s2_h_q_lora_rank
    config::SetPassOption(VEC_NBUFFER_MODE, 2);
    config::SetPassOption(VEC_NBUFFER_SETTING,  std::map<int64_t, int64_t>{{-1, 2}});
    int& h = std::get<int>(g_deepseekConfig["hiddenSize"]);
    int& n = std::get<int>(g_deepseekConfig["numAttentionHeads"]);
    int& qLoraRank = std::get<int>(g_deepseekConfig["qLoraRank"]);
    int& qkRopeHeadDim = std::get<int>(g_deepseekConfig["qkRopeHeadDim"]);
    int& kvLoraRank = std::get<int>(g_deepseekConfig["kvLoraRank"]);
    int& vHeadDim = std::get<int>(g_deepseekConfig["vHeadDim"]);
    int& qkNopeHeadDim = std::get<int>(g_deepseekConfig["qkNopeHeadDim"]);

    int b = 4;
    int s = 1;
    int s2 = 256;
    h = 7168;
    n = 32;
    qLoraRank = 1536;
    qkNopeHeadDim = 128;
    qkRopeHeadDim = 64;
    kvLoraRank = 512;
    vHeadDim = 128;

    std::vector<int> params = {b, s, s2, n, h, qLoraRank, qkNopeHeadDim, qkRopeHeadDim,
        kvLoraRank, vHeadDim};
    Attention<npu::tile_fwk::bfloat16>(params, GetGoldenDir(), false);
}

//test_MlaProlog_float16_2_32_1_256_256_256
TEST_F(MlaPrologOnBoardTest, attention_bf16_4_1024_1024_32_256) {  // b_n_s_s2_h_q_lora_rank
    config::SetPassOption(VEC_NBUFFER_MODE, 2);
    config::SetPassOption(VEC_NBUFFER_SETTING,  std::map<int64_t, int64_t>{{-1, 2}});
    int& h = std::get<int>(g_deepseekConfig["hiddenSize"]);
    int& n = std::get<int>(g_deepseekConfig["numAttentionHeads"]);
    int& qLoraRank = std::get<int>(g_deepseekConfig["qLoraRank"]);
    int& qkRopeHeadDim = std::get<int>(g_deepseekConfig["qkRopeHeadDim"]);
    int& kvLoraRank = std::get<int>(g_deepseekConfig["kvLoraRank"]);
    int& vHeadDim = std::get<int>(g_deepseekConfig["vHeadDim"]);
    int& qkNopeHeadDim = std::get<int>(g_deepseekConfig["qkNopeHeadDim"]);

    int b = 4;
    int s = 1;
    int s2 = 1024;
    h = 1024;
    n = 32;
    qLoraRank = 256;
    qkNopeHeadDim = 128;
    qkRopeHeadDim = 64;
    kvLoraRank = 512;
    vHeadDim = 128;
    std::vector<int> params = {b, s, s2, n, h, qLoraRank, qkNopeHeadDim, qkRopeHeadDim,
        kvLoraRank, vHeadDim};
    Attention<npu::tile_fwk::bfloat16>(params, GetGoldenDir(), false);
}
