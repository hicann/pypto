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
 * \file test_graph_only.cpp
 * \brief
 */


#include "gtest/gtest.h"

#include "interface/tensor/logical_tensor.h"
#include "interface/configs/config_manager.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "operator/models/llama/llama_def.h"
#include "operator/models/deepseek/deepseek_mla.h"
#include "operator/models/deepseek/mla_prolog.h"
#include "operator/models/deepseek/deepseek_spec.h"
#include "tilefwk/data_type.h"

using namespace npu::tile_fwk;

class GraphTest : public testing::Test {
public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}

    void SetUp() override {
        Program::GetInstance().Reset();
        config::Reset();
        config::SetHostOption(COMPILE_STAGE, HOST_COMPILE_END);
        config::SetHostConfig(KEY_STRATEGY, "PVC2_OOO");
        config::SetPlatformConfig(KEY_ENABLE_COST_MODEL, false);
        config::SetSimConfig(KEY_BUILD_TASK_BASED_TOPO, false);
    }

    void TearDown() override {}
};

void RunLLamaLayerGraph(const AttentionDims &dimsCfg) {
    int b = dimsCfg.b;
    int n = dimsCfg.n;
    int s = dimsCfg.s;
    int d = dimsCfg.d;
    PROGRAM("LLAMALAYER") {
        Tensor H(DataType::DT_FP32, {b * s, n * d}, "H");
        Tensor AW(DataType::DT_FP16, {n * d, n * d * 3}, "AW");
        Tensor DW(DataType::DT_FP16, {n * d, n * d}, "DW");
        Tensor FW(DataType::DT_FP16, {n * d, n * d * 3}, "FW");
        Tensor Res(DT_FP32, {b * s, n * d}, "Res");
        FUNCTION("LLAMA") {
            Res = LlamaLayer(H, AW, DW, FW, dimsCfg, SMALL_DFS_VEC_CFG, DFS_CUBE_CFG);
        }
    }
}

TEST_F(GraphTest, llama_1_1_128_128)
{
    AttentionDims dimsCfg = {1, 1, 128, 128, DFT_SINGLE_M, DFT_SINGLE_N};
    RunLLamaLayerGraph(dimsCfg);
}

TEST_F(GraphTest, llama_1_1_256_128)
{
    AttentionDims dimsCfg = {1, 1, 256, 128, DFT_SINGLE_M, DFT_SINGLE_N};
    RunLLamaLayerGraph(dimsCfg);
}

TEST_F(GraphTest, llama_1_1_512_128)
{
    AttentionDims dimsCfg = {1, 1, 512, 128, DFT_SINGLE_M, DFT_SINGLE_N};
    RunLLamaLayerGraph(dimsCfg);
}

TEST_F(GraphTest, llama_1_1_256_128_mix)
{
    config::SetPassConfig("PVC2_OOO", "PreGraphProcess", KEY_PRE_CHECK, false);
    config::SetPassConfig("PVC2_OOO", "PreGraphProcess", KEY_POST_CHECK, false);
    AttentionDims dimsCfg = {1, 1, 256, 128, DFT_SINGLE_M, DFT_SINGLE_N};
    RunLLamaLayerGraph(dimsCfg);
}

TEST_F(GraphTest, llama_1_1_1024_128)
{
    AttentionDims dimsCfg = {1, 1, 1024, 128, DFT_SINGLE_M, DFT_SINGLE_N};
    RunLLamaLayerGraph(dimsCfg);
}

TEST_F(GraphTest, deepseek_qkvPre) {
        int b = 2;
    int s = 128;
    int h = std::get<int>(deepseekConfig1["hiddenSize"]);
    int num_heads = std::get<int>(deepseekConfig1["numAttentionHeads"]);
    int qLoraRank = std::get<int>(deepseekConfig1["qLoraRank"]);
    int qkRopeHeadDim = std::get<int>(deepseekConfig1["qkRopeHeadDim"]);
    int kvLoraRank = std::get<int>(deepseekConfig1["kvLoraRank"]);
    int vHeadDim = std::get<int>(deepseekConfig1["vHeadDim"]);
    int qkNopeHeadDim = std::get<int>(deepseekConfig1["qkNopeHeadDim"]);
    int q_head_dim = qkNopeHeadDim + qkRopeHeadDim;
    ALOG_INFO("Test_qkvPre  b,s,h: ", b, ", ", s, ", ", h);

    Tensor hidden_states = Tensor(DT_BF16, {b, s, h}, "hidden_states");

    AttentionW aw;
    aw.qAProjW = Tensor(DT_BF16, {h, qLoraRank}, "qAProjW");
    aw.qBProjW = Tensor(DT_BF16, {qLoraRank, num_heads * q_head_dim}, "qBProjW");
    aw.kvAProjWithMqaW = Tensor(DT_BF16, {h, kvLoraRank + qkRopeHeadDim}, "kvAProjWithMqaW");
    aw.kvBProjWK = Tensor(DT_BF16, {num_heads, qkNopeHeadDim, kvLoraRank}, "kvBProjWK");
    aw.kvBProjWV = Tensor(DT_BF16, {num_heads, kvLoraRank, vHeadDim}, "kvBProjWV");
    aw.oProjW = Tensor(DT_BF16, {num_heads * vHeadDim, h}, "oProjW");

    std::tuple<Tensor, Tensor> res;
    DeepseekAttention Attention(deepseekConfig1, aw, 1);

    FUNCTION("A") {
        res = Attention.QkvPre(hidden_states);
    }
}

TEST_F(GraphTest, TestAttentionPost) {
    config::SetHostOption(COMPILE_STAGE, HOST_COMPILE_END);
    int b = 1;
    int n = 2;
    int s = 128;
    int d = 512;
    int v_head =128;
    int h = 256;
    std::vector<int64_t> inShape = {b, n, s, d}; // (b, n, s, d)
    Tensor attnPostIn(DT_FP32, inShape, "attnPostIn");
    Tensor kvBProjWV(DT_FP32, {n, d, v_head}, "kvBProjWV");
    Tensor oProjW(DT_FP32, {n * v_head, h}, "oProjW");
    Tensor atten_output;
    ConfigManager::Instance();
    FUNCTION("AttentionPost") {
        int new_b = attnPostIn.GetShape()[0];
        int new_n = attnPostIn.GetShape()[1];
        int new_s = attnPostIn.GetShape()[2];
        DataType dType = attnPostIn.GetStorage()->Datatype();
        TileShape::Current().SetVecTile({1, 1, 32, d});
        Tensor atten_res1 = Reshape(Transpose(attnPostIn, {1, 2}), {new_b * new_s, new_n, d});
        TileShape::Current().SetVecTile({32, 1, d});
        Tensor atten_res2 = Transpose(atten_res1, {0, 1});
        // [n,bs,kvLoraRank] * [n, kvLoraRank, vHeadDim] = [n,bs,vHeadDim]
        TileShape::Current().SetVecTile(128, 128);
        TileShape::Current().SetCubeTile({32, 32}, {128, 128}, {128, 128});
        Tensor mm7_res = Matrix::BatchMatmul(dType, atten_res2, kvBProjWV);
        // Tensor mm7_res = Matrix::BatchMatmul(dType, atten_res2, kvBProjWV);
        TileShape::Current().SetVecTile({1, 128, 128});
        Tensor mm7_res1 = Transpose(mm7_res, {0, 1});
        Tensor mm7_res2 = Reshape(mm7_res1, {new_b, new_s, new_n * v_head});

        // [b,s, n*vHeadDim] @ [n*vHeadDim, H] = [b,s,h]
        Tensor attn_out_w = Unsqueeze(oProjW, 0);
        atten_output = Matrix::BatchMatmul(dType, mm7_res2, attn_out_w);
    }
}

TEST_F(GraphTest, Test_deepseekAttention_s_1) {
    config::SetHostOption(COMPILE_STAGE, HOST_COMPILE_END);

    int b = 2; //  32
    int s = 1;
    int s2 = 512;
    int h = std::get<int>(deepseekConfig1["hiddenSize"]); // 256
    int num_heads = std::get<int>(deepseekConfig1["numAttentionHeads"]);
    int qLoraRank = std::get<int>(deepseekConfig1["qLoraRank"]);
    int qkRopeHeadDim = std::get<int>(deepseekConfig1["qkRopeHeadDim"]); // 64
    int kvLoraRank = std::get<int>(deepseekConfig1["kvLoraRank"]);         // 512
    int vHeadDim = std::get<int>(deepseekConfig1["vHeadDim"]);
    int qkNopeHeadDim = std::get<int>(deepseekConfig1["qkNopeHeadDim"]);
    int q_head_dim = qkNopeHeadDim + qkRopeHeadDim;
    Tensor hidden_states = Tensor(DT_BF16, {b, s, h}, "hidden_states");
    Tensor atten_mask = Tensor(DT_FP32, {b, 1, s, s2}, "atten_mask");
    Tensor position_ids = Tensor(DT_INT32, {b, s}, "position_ids");
    Tensor cos = Tensor(DT_BF16, {s, qkRopeHeadDim}, "cos");
    Tensor sin = Tensor(DT_BF16, {s, qkRopeHeadDim}, "sin");
    Tensor kv_len = Tensor(DT_INT32, {1, 1}, "kv_len");
    Tensor past_key_states = Tensor(DT_BF16, {b, 1, s2, kvLoraRank + qkRopeHeadDim}, "past_key_states");

    AttentionW aw;
    aw.qAProjW = Tensor(DT_BF16, {h, qLoraRank}, "qAProjW");
    aw.qBProjW = Tensor(DT_BF16, {qLoraRank, num_heads * q_head_dim}, "qBProjW");
    aw.kvAProjWithMqaW = Tensor(DT_BF16, {h, kvLoraRank + qkRopeHeadDim}, "kvAProjWithMqaW");
    aw.kvBProjWK = Tensor(DT_BF16, {num_heads, qkNopeHeadDim, kvLoraRank}, "kvBProjWK");
    aw.kvBProjWV = Tensor(DT_BF16, {num_heads, kvLoraRank, vHeadDim}, "kvBProjWV");
    aw.oProjW = Tensor(DT_BF16, {num_heads * vHeadDim, h}, "oProjW");

    RoPETileShapeConfig ropeTileConfig{
        {32, 32},
        {1, 32, 32},
        {1, 1, 32, 32},
        {1, 1, 32, 32, 2}
    };

    Tensor res;
    DeepseekAttention deepseekAttention(deepseekConfig1, aw, 1);
    ConfigManager::Instance();
    FUNCTION("A") {
        res = deepseekAttention.Forward(
            hidden_states, atten_mask, position_ids, cos, sin, kv_len, past_key_states, ropeTileConfig);
    }
}

TEST_F(GraphTest, Test_deepseekAttention_pre) {
    config::SetHostOption(COMPILE_STAGE, HOST_COMPILE_END);

    int b = 2; //  32
    int s = 1;
    int s2 = 256;
    int h = std::get<int>(deepseekConfig1["hiddenSize"]); // 256
    int num_heads = std::get<int>(deepseekConfig1["numAttentionHeads"]);
    int qLoraRank = std::get<int>(deepseekConfig1["qLoraRank"]);
    int qkRopeHeadDim = std::get<int>(deepseekConfig1["qkRopeHeadDim"]); // 64
    int kvLoraRank = std::get<int>(deepseekConfig1["kvLoraRank"]);         // 512
    int vHeadDim = std::get<int>(deepseekConfig1["vHeadDim"]);
    int qkNopeHeadDim = std::get<int>(deepseekConfig1["qkNopeHeadDim"]);
    int q_head_dim = qkNopeHeadDim + qkRopeHeadDim;
    Tensor hidden_states = Tensor(DT_BF16, {b, s, h}, "hidden_states");
    Tensor atten_mask = Tensor(DT_FP32, {b, 1, s, s2}, "atten_mask");
    Tensor position_ids = Tensor(DT_INT32, {b, s}, "position_ids");
    Tensor cos = Tensor(DT_BF16, {s, qkRopeHeadDim}, "cos");
    Tensor sin = Tensor(DT_BF16, {s, qkRopeHeadDim}, "sin");
    Tensor kv_len = Tensor(DT_INT32, {1, 1}, "kv_len");
    Tensor past_key_states = Tensor(DT_BF16, {b, 1, s2, kvLoraRank + qkRopeHeadDim}, "past_key_states");

    AttentionW aw;
    aw.qAProjW = Tensor(DT_BF16, {h, qLoraRank}, "qAProjW");
    aw.qBProjW = Tensor(DT_BF16, {qLoraRank, num_heads * q_head_dim}, "qBProjW");
    aw.kvAProjWithMqaW = Tensor(DT_BF16, {h, kvLoraRank + qkRopeHeadDim}, "kvAProjWithMqaW");
    aw.kvBProjWK = Tensor(DT_BF16, {num_heads, qkNopeHeadDim, kvLoraRank}, "kvBProjWK");
    aw.kvBProjWV = Tensor(DT_BF16, {num_heads, kvLoraRank, vHeadDim}, "kvBProjWV");
    aw.oProjW = Tensor(DT_BF16, {num_heads * vHeadDim, h}, "oProjW");

    RoPETileShapeConfig ropeTileConfig{
        {32, 32},
        {1, 32, 32},
        {1, 1, 32, 32},
        {1, 1, 32, 32, 2}
    };

    std::tuple<Tensor, Tensor> res;
    DeepseekAttention deepseekAttention(deepseekConfig1, aw, 1);
    ConfigManager::Instance();
    FUNCTION("A") {
        res = deepseekAttention.AtentionPreForward(
            hidden_states, atten_mask, position_ids, cos, sin, kv_len, past_key_states, ropeTileConfig);
    }
}



TEST_F(GraphTest, test_operation_rope_subgraph_deepseekv3_bf16) {

    RoPETileShapeConfig ropeTileConfig{
        {64, 64}, // for cos/sin->cast
        {1, 64, 64}, // for gather,unsqueeze
        {1, 64, 1, 64},
        {1, 64, 1, 32, 2} // for transpose
    };

    int B = 1;
    int N = 32;                // N=32
    int S = 1;                 // IFA S=1 S=1024
    int qkRopeHeadDim = 64; // qkRopeHeadDim = 64

    std::vector<int64_t> qPeShape{B, S, N, qkRopeHeadDim};
    std::vector<int64_t> kPeShape{B, S, qkRopeHeadDim};
    std::vector<int64_t> idsShape{B, S};
    std::vector<int64_t> cosShape{S, qkRopeHeadDim};

    std::vector<int64_t> qEmbedShape{B, N, S, qkRopeHeadDim};
    std::vector<int64_t> kEmbedShape{B, 1, S, qkRopeHeadDim};

    Tensor qPe(DT_BF16, qPeShape, "qPe");
    Tensor kPe(DT_BF16, kPeShape, "kPe");
    Tensor cos(DT_BF16, cosShape, "cos");
    Tensor sin(DT_BF16, cosShape, "sin");
    Tensor positionIds(DT_INT32, idsShape, "positionIds");
    Tensor qEmbed(DT_BF16, qEmbedShape, "qEmbed");
    Tensor kEmbed(DT_BF16, kEmbedShape, "kEmbed");

    ConfigManager::Instance();
    FUNCTION("RoPE") {
        TileShape::Current().SetVecTile({1, 1, 64, 64});
        auto qPeTrans = Transpose(qPe, {1, 2}); // [b,s,n,d]->[b,n,s,d]

        int b = kPe.GetShape()[0];
        int s = kPe.GetShape()[1];
        int d = kPe.GetShape()[2];
        // 以下两步Reshape+Transpose可以优化成1个reshape：auto kPeReshape = Reshape(kPe, {b, 1, s, d}); //
        // [b,s,d]->[b,1,s,d]
        auto kPeReshape = Reshape(kPe, {b, 1, s, d}); // [b,s,d]->[b,1,s,d]
        // auto kPeTrans = Transpose(Reshape(kPe, {b, s, 1, d}), {1, 2}); // [b,s,d]->[b,s,1,d]->[b,1,s,d]
        ApplyRotaryPosEmb(qPeTrans, kPeReshape, cos, sin, positionIds, qEmbed, kEmbed, 1, ropeTileConfig);
    }
}

// inputType: 0-fp16, 1-bf16, 2-fp32
template <bool splitReduceLastDim = true, bool splitK = false, bool nz= false>
void TestMlaPrologV2(std::vector<int> &params, int inputType, bool isQuant = false) {
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

    DataType dType = DT_FP32;
    if (inputType == 0) {
        dType = DT_FP16;
    } else if (inputType == 1) {
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
    // output
    std::vector<int64_t> q_out_shape = {b, s, n, kvLoraRank};
    std::vector<int64_t> q_rope_out_shape = {b, s, n, qkRopeHeadDim};
    std::vector<int64_t> kv_cache_out_shape = {b, 1, s2, kvLoraRank};
    std::vector<int64_t> kr_cache_out_shape = {b, 1, s2, qkRopeHeadDim};

    std::vector<int64_t> w_qb_scale_shape;
    if (isQuant) {
        w_qb_scale_shape = {1, n * q_head_dim};
    }


    ConfigManager::Instance();
    PROGRAM("MlaProlog") {
        Tensor x(dType, x_shape, "x");
        Tensor w_qa(dType, w_qa_shape, "w_qa");
        Tensor w_qb(dTypeQuantIn, w_qb_shape, "w_qb");
        Tensor w_kv_a(dType, w_kv_a_shape, "w_kv_a");
        Tensor w_kv_b_k(dType, w_kv_b_k_shape, "w_kv_b_k");
        Tensor gamma_cq(dType, gamma_cq_shape, "gamma_cq");
        Tensor gamma_ckv(dType, gamma_ckv_shape, "gamma_ckv");
        Tensor cos(dType, cos_shape, "cos");
        Tensor sin(dType, cos_shape, "sin");
        Tensor kv_len(DT_INT64, kv_len_shape, "kv_len");  // int64
        Tensor kv_cache(dType, kv_cache_shape, "kv_cache");
        Tensor kr_cache(dType, kr_cache_shape, "kr_cache");
        // output
        Tensor output_q(dType, q_out_shape, "output_q");
        Tensor output_q_rope(dType, q_rope_out_shape, "output_q_rope");

        RoPETileShapeConfigNew ropeConfig{
            {32, 1, 64}, // (b,s,d)
            {1, 1, 32, 64}, // Q (b,s,n,d)
            {32, 1, 1, 64}, // K (b,s,1,d)
            {32, 1, 1, 32, 2} // (b,s,n,d//2,2)
        };

        MlaQuantInputs quantInputs;

        if (isQuant) {
            Tensor w_qb_scale = Tensor(DT_FP32, w_qb_scale_shape, "w_qb_scale");
            quantInputs.dequantScaleWUqQr = w_qb_scale;

            config::SetBuildStatic(true);
            FUNCTION("MlaProlog_T", {x, w_qa, w_qb, w_qb_scale, w_kv_b_k, w_kv_a,
                gamma_cq, gamma_ckv, sin, cos, kv_len, kv_cache, kr_cache, output_q, output_q_rope}) {
                MlaProlog(x, w_qa, w_qb, w_kv_b_k, w_kv_a, gamma_cq, gamma_ckv, sin, cos, kv_len, kv_cache, kr_cache,
                    quantInputs, ropeConfig, output_q, output_q_rope, kv_cache, kr_cache, 1e-5f, 1e-5f,  "BNSD", splitReduceLastDim,  splitK);
            };
        } else {
            config::SetBuildStatic(true);
            FUNCTION("MlaProlog_T", {x, w_qa, w_qb, w_kv_b_k, w_kv_a,
                gamma_cq, gamma_ckv, sin, cos, kv_len, kv_cache, kr_cache, output_q, output_q_rope}) {
                MlaProlog(x, w_qa, w_qb, w_kv_b_k, w_kv_a, gamma_cq, gamma_ckv, sin, cos, kv_len, kv_cache, kr_cache,
                    quantInputs, ropeConfig, output_q, output_q_rope, kv_cache, kr_cache, 1e-5f, 1e-5f,  "BNSD", splitReduceLastDim,  splitK);
            };
        }
    }
}

TEST_F(GraphTest, Test_MlaPrologV2_bfloat16_4_32_1_256_7168_1536) {  // b_n_s_s2_h_q_lora_rank, bfloat16
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
    TestMlaPrologV2(params, 1);
}

TEST_F(GraphTest, Test_MlaPrologV2_bfloat16_4_32_1_256_7168_1536_splitnz) {  // b_n_s_s2_h_q_lora_rank, bfloat16
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
    TestMlaPrologV2<true,true>(params, 1);
}

TEST_F(GraphTest, test_mla_bf16_low_quant_smooth) {  // b_n_s_s2_h_q_lora_rank, bfloat16
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
    TestMlaPrologV2(params, 1, true);
}

void TestMlaProlog(std::vector<int> &params) {
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

    DataType dType = DT_BF16;

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


    PROGRAM("MlaProlog") {
        Tensor x(dType, x_shape, "x");
        Tensor w_qa(dType, w_qa_shape, "w_qa");
        Tensor w_qb(dType, w_qb_shape, "w_qb");
        Tensor w_kv_a(dType, w_kv_a_shape, "w_kv_a");
        Tensor w_kv_b_k(dType, w_kv_b_k_shape, "w_kv_b_k");
        Tensor position_ids(DT_INT32, position_ids_shape, "position_ids");
        Tensor cos(dType, cos_shape, "cos");
        Tensor sin(dType, cos_shape, "sin");
        Tensor past_key_states(dType, past_key_states_shape, "past_key_states");
        Tensor kv_len(DT_INT32, kv_len_shape, "kv_len");
        // output
        Tensor output_q(dType, q_shape, "output_q");

        AttentionW aw;
        aw.qAProjW = w_qa;
        aw.qBProjW = w_qb;
        aw.kvAProjWithMqaW = w_kv_a;
        aw.kvBProjWK = w_kv_b_k;
        Tensor kvBProjWV;  // not used in MlaProlog
        Tensor oProjW;       // not used in MlaProlog
        aw.kvBProjWV = kvBProjWV;
        aw.oProjW = oProjW;

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
            auto q_kv = Attention.MlaPrologFoward(x, position_ids, cos, sin, kv_len, past_key_states, ropeTileConfig);
            output_q = q_kv[0];
            past_key_states = q_kv[1];
        }
    }
}

TEST_F(GraphTest, test_attention_bf16_4_1024_1024_32_256) {  // b_n_s_s2_h_q_lora_rank
    config::SetPassOption(VEC_NBUFFER_MODE, 2);
    config::SetPassOption(VEC_NBUFFER_SETTING, std::map<int64_t, int64_t>{{-1, 2}});
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
    TestMlaProlog(params);
}

void TestLoopTailBlock(const Tensor &t0, const Tensor &blockTable, Tensor &out, int s) {
    int blockSize = 64;

    FUNCTION("main", {t0, blockTable}, {out}) {
        LOOP("L0", FunctionType::DYNAMIC_LOOP, i, LoopRange(GetInputShape(t0, 0) / s)) {
            SymbolicScalar size = GetTensorData(blockTable, {i, 0});
            Tensor t0s = View(t0, {s, s}, {size, s}, {blockSize * i, 0});
            Tensor t1s = View(t0, {s/2, s}, {size, s}, {blockSize * i, 0});
            Tensor t1 = Add(t1s, t1s);
            Assemble(t1, {blockSize * i, 0}, out);
        }
    }
}

TEST_F(GraphTest, TestTailBlock) {
    TileShape::Current().SetVecTile(32, 32);
    TileShape::Current().SetCubeTile({32, 32}, {32, 32}, {32, 32});
    int s = 64;
    int n = 8;
    Tensor t0(DT_FP32, {n * s, s}, "t0");  // [32*8, 32]
    Tensor blockTable{DT_INT32, {n, 1}, "blockTable"};
    Tensor out(DT_FP32, {n * s, s}, "out");
    TestLoopTailBlock(t0, blockTable, out, s);
}

TEST_F(GraphTest, TestTranspose_MLA_3D_2_add) {
    int bs = 8;
    int n = 32;
    int d = 128;
    std::vector<int64_t> shape{bs, n, d};
    std::vector<int64_t> resShape{n, bs, d};
    PROGRAM("Transpose") {
        Tensor input(DataType::DT_FP32, shape, "input");
        Tensor output(DataType::DT_FP32, resShape, "res");
        config::SetBuildStatic(true);
        FUNCTION("MLA_3D_2", {input, output}) {
            TileShape::Current().SetVecTile(NUM_2, NUM_2, NUM_128);
            auto tmp = Transpose(input, {0, 1});
            TileShape::Current().SetVecTile(NUM_8, NUM_8, NUM_128);
            output = Add(tmp, Element(DataType::DT_FP32, 0.0));
        }
    }
}

TEST_F(GraphTest, TestTranspose_MLA_3D_2_reshape) {
    int bs = 8;
    int n = 32;
    int d = 128;
    std::vector<int64_t> shape{bs, n, d};
    std::vector<int64_t> transposeShape{n, bs, d};
    std::vector<int64_t> resShape{n, bs * d};

    PROGRAM("Transpose") {
        Tensor input(DataType::DT_FP32, shape, "input");
        Tensor output1(DataType::DT_FP32, transposeShape, "res1");
        Tensor output2(DataType::DT_FP32, resShape, "res2");
        config::SetBuildStatic(true);
        FUNCTION("MLA_3D_2", {input, output1, output2}) {
            TileShape::Current().SetVecTile(NUM_2, NUM_2, NUM_128);
            output1 = Transpose(input, {0, 1}); // [8, 32, 128] --> [32, 8, 128]
            TileShape::Current().SetVecTile(NUM_8, NUM_8, NUM_128);
            output2 = Reshape(output1, resShape); // [32, 8, 128] --> [32, 1024]
        }
    }
}