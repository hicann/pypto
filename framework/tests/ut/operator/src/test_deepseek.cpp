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
 * \file test_deepseek.cpp
 * \brief
 */

#include "tilefwk/tilefwk_op.h"
#include "test_cost_macro.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "operator/models/deepseek/deepseek_mla.h"
#include "interface/configs/config_manager.h"
#include "operator/models/llama/llama_def.h"
#include "operator/models/deepseek/deepseek_spec.h"
#include "operator/models/deepseek/page_attention.h"
#include "interface/interpreter/raw_tensor_data.h"
#include "operator/models/deepseek/dynamic_mla.h"

using namespace npu::tile_fwk;

constexpr float F_127 = 127.0;

class FunctionTest : public testing::Test {
public:
    static void SetUpTestCase() {
        config::SetCodeGenConfig(KEY_CODEGEN_SUPPORT_TILE_TENSOR, false);
    }

    static void TearDownTestCase() {
        config::SetCodeGenConfig(KEY_CODEGEN_SUPPORT_TILE_TENSOR, true);
    }

    void SetUp() override { Program::GetInstance().Reset(); }

    void TearDown() override {}
};

void pre() {
    config::SetHostOption(ONLY_CODEGEN, true);
}

TEST_F(FunctionTest, TestAddTensorFunctionDim4) {
    std::vector<int64_t> shape{2,2,32,32};

    Tensor a(DT_FP32, shape, "a");
    Tensor b(DT_FP32, shape, "b");
    Tensor c;
    config::SetPlatformConfig(KEY_ONLY_HOST_COMPILE, true);

    TileShape::Current().SetVecTile(1, 1, 16, 16);

    FUNCTION("A") {
        c = Add(a, b);
    }

    // Program::GetInstance().GraphCheck();

    ALOG_INFO(Program::GetInstance().Dump());
}

TEST_F(FunctionTest, TestAddTensorFunctionDim2) {
    std::vector<int64_t> shape{16,16};

    Tensor a(DT_FP32, shape, "a");
    Tensor b(DT_FP32, shape, "b");
    Tensor c;
    config::SetPlatformConfig(KEY_ONLY_HOST_COMPILE, true);

    TileShape::Current().SetVecTile(8, 16);

    FUNCTION("A") {
        c = Add(a, b);
    }

    // Program::GetInstance().GraphCheck();

    ALOG_INFO(Program::GetInstance().Dump());
}

TEST_F(FunctionTest, TestOperationRopeV2Deepseekv3B32) {
    int B = 32;
    int N = 128;                // N=128
    int S = 1;                 // IFA S=1 S=1024
    int qkRopeHeadDim = 64; // qkRopeHeadDim = 64

    std::vector<int64_t> qPeShape{B, S, N, qkRopeHeadDim};
    std::vector<int64_t> kPeShape{B, S, 1, qkRopeHeadDim};
    std::vector<int64_t> cosSinShape{B, S, qkRopeHeadDim};
    std::vector<int64_t> qEmbedShape{B, S, N, qkRopeHeadDim};
    std::vector<int64_t> kEmbedShape{B, S, 1, qkRopeHeadDim};

    Tensor q(DT_FP32, qPeShape, "q");
    Tensor k(DT_FP32, kPeShape, "k");
    Tensor cos(DT_FP32, cosSinShape, "cos");
    Tensor sin(DT_FP32, cosSinShape, "sin");
    Tensor qEmbed(DT_FP32, qEmbedShape, "qEmbed");
    Tensor kEmbed(DT_FP32, kEmbedShape, "kEmbed");

    config::SetPlatformConfig(KEY_ONLY_HOST_COMPILE, true);
    RoPETileShapeConfigNew ropeTileConfig{
        {32, 1, 64}, // (b,s,d)
        {1, 1, 32, 64}, // Q (b,s,n,d)
        {32, 1, 1, 64}, // K (b,s,1,d)
        {32, 1, 1, 32, 2} // (b,s,n,d//2,2)
    };
    FUNCTION("A") {
        ApplyRotaryPosEmbV2(q, k, cos, sin, qEmbed, kEmbed, 2, ropeTileConfig);
    }

    ALOG_INFO(Program::GetInstance().Dump());
}

TEST_F(FunctionTest, test_fa_new) {

    AttentionDims atDims = {1, 1, 128, 128, DFT_SINGLE_M, DFT_SINGLE_N};
    int b = atDims.b;
    int n = atDims.n;
    int s = atDims.s;
    int d = atDims.d;
    int dim0 = b * n * s; // 1024
    int dim1 = d;         // 128
    int capacity = dim0 * dim1;
    int capacity_reduce = dim0 * 1;
    std::vector<int64_t> shape = {dim0, dim1};
    std::vector<int64_t> shape_reduce = {dim0, 1};
    std::vector<uint16_t> q_data(capacity);
    std::vector<uint16_t> k_data(capacity);
    std::vector<uint16_t> v_data(capacity);
    std::vector<float> res_golden_data(capacity);
    std::vector<float> max_golden_data(capacity_reduce);
    std::vector<float> sum_golden_data(capacity_reduce);

    config::SetPlatformConfig(KEY_ONLY_HOST_COMPILE, true);
    Tensor Q(DataType::DT_FP16, shape, "Q");
    Tensor K(DataType::DT_FP16, shape, "K");
    Tensor V(DataType::DT_FP16, shape, "V");
    Tensor M(DataType::DT_FP32, shape_reduce, "M");
    Tensor L(DataType::DT_FP32, shape_reduce, "L");
    Tensor Res(DT_FP32, shape, "Res");
    config::SetBuildStatic(true);
    FUNCTION("FA", {Q, K, V, M, L, Res}) {
        TileShape::Current().SetVecTile({16, 128});
        TileShape::Current().SetCubeTile({128, 128}, {128, 128}, {128, 128});
        Res = FlashAttentionNew(Q, K, V, M, L, atDims);
    }
}

TEST_F(FunctionTest, TestSubTensorFunctionDim2) {
    std::vector<int64_t> shape{64,64};

    Tensor a(DT_FP32, shape, "a");
    Tensor b(DT_FP32, shape, "b");
    Tensor c;
    config::SetPlatformConfig(KEY_ONLY_HOST_COMPILE, true);

    TileShape::Current().SetVecTile(32, 32);

    FUNCTION("A") {
        c = Sub(a, b);
    }

    // Program::GetInstance().GraphCheck();

    ALOG_INFO(Program::GetInstance().Dump());
}

TEST_F(FunctionTest, TestMulTensorFunctionDim2) {
    std::vector<int64_t> shape{64,64};

    Tensor a(DT_FP32, shape, "a");
    Tensor b(DT_FP32, shape, "b");
    Tensor c;
    config::SetPlatformConfig(KEY_ONLY_HOST_COMPILE, true);

    TileShape::Current().SetVecTile(32, 32);

    FUNCTION("A") {
        c = Mul(a, b);
    }

    // Program::GetInstance().GraphCheck();

    ALOG_INFO(Program::GetInstance().Dump());
}

TEST_F(FunctionTest, TestDivTensorFunctionDim2) {
    std::vector<int64_t> shape{64,64};

    Tensor a(DT_FP32, shape, "a");
    Tensor b(DT_FP32, shape, "b");
    Tensor c;
    config::SetPlatformConfig(KEY_ONLY_HOST_COMPILE, true);

    TileShape::Current().SetVecTile(32, 32);

    FUNCTION("A") {
        c = Div(a, b);
    }

    // Program::GetInstance().GraphCheck();

    ALOG_INFO(Program::GetInstance().Dump());
}

TEST_F(FunctionTest, TestAddScalarFunctionDim2) {
    std::vector<int64_t> shape{64,64};

    Tensor a(DT_FP32, shape, "a");
    Element value(DataType::DT_FP32, 1.5);
    Tensor d;

    config::SetPlatformConfig(KEY_ONLY_HOST_COMPILE, true);

    TileShape::Current().SetVecTile(32, 32);

    FUNCTION("A") {
        d = Add(a, value);
    }

    // Program::GetInstance().GraphCheck();

    ALOG_INFO(Program::GetInstance().Dump());
}

TEST_F(FunctionTest, TestAddScalarFunctionDim3) {
    std::vector<int64_t> shape{64,64,64};

    Tensor a(DT_FP32, shape, "a");
    Element value(DataType::DT_FP32, 1.5);
    Tensor d;

    config::SetPlatformConfig(KEY_ONLY_HOST_COMPILE, true);

    TileShape::Current().SetVecTile(32, 32, 32);

    FUNCTION("A") {
        d = Add(a, value);
    }

    // Program::GetInstance().GraphCheck();

    ALOG_INFO(Program::GetInstance().Dump());
}

TEST_F(FunctionTest, TestSubScalarFunctionDim2) {
    std::vector<int64_t> shape{64,64};

    Tensor a(DT_FP32, shape, "a");
    Element value(DataType::DT_FP32, 1.5);
    Tensor d;

    config::SetPlatformConfig(KEY_ONLY_HOST_COMPILE, true);

    TileShape::Current().SetVecTile(32, 32);

    FUNCTION("A") {
        d = Sub(a, value);
    }

    // Program::GetInstance().GraphCheck();

    ALOG_INFO(Program::GetInstance().Dump());
}

TEST_F(FunctionTest, TestMulScalarFunctionDim2) {
    std::vector<int64_t> shape{64,64};

    Tensor a(DT_FP32, shape, "a");
    Element value(DataType::DT_FP32, 1.5);
    Tensor d;

    config::SetPlatformConfig(KEY_ONLY_HOST_COMPILE, true);

    TileShape::Current().SetVecTile(32, 32);

    FUNCTION("A") {
        d = Mul(a, value);
    }

    // Program::GetInstance().GraphCheck();

    ALOG_INFO(Program::GetInstance().Dump());
}

TEST_F(FunctionTest, TestDivScalarFunctionDim2) {
    std::vector<int64_t> shape{64,64};

    Tensor a(DT_FP32, shape, "a");
    Element value(DataType::DT_FP32, 1.5);
    Tensor d;

    config::SetPlatformConfig(KEY_ONLY_HOST_COMPILE, true);

    TileShape::Current().SetVecTile(32, 32);

    FUNCTION("A") {
        d = Div(a, value);
    }

    // Program::GetInstance().GraphCheck();

    ALOG_INFO(Program::GetInstance().Dump());
}

TEST_F(FunctionTest, TestExpTensorFunctionDim2) {
    std::vector<int64_t> shape{64,64};

    Tensor a(DT_FP32, shape, "a");
    Tensor c;
    config::SetPlatformConfig(KEY_ONLY_HOST_COMPILE, true);

    TileShape::Current().SetVecTile(32, 32);

    FUNCTION("A") {
        c = Exp(a);
    }

    ALOG_INFO(Program::GetInstance().Dump());
}

TEST_F(FunctionTest, TestSin) {
    TileShape::Current().SetVecTile({1, 1, 4, 4});
    config::SetPlatformConfig(KEY_ONLY_HOST_COMPILE, true);

    std::vector<int64_t> shape1 = {1, 2, 8, 8};
    Tensor input(DT_FP16, shape1, "input");
    Tensor res;

    FUNCTION("Sin") {
        res = Sin(input);
    }
    // Program::GetInstance().GraphCheck();

    ALOG_INFO(Program::GetInstance().Dump());
}

TEST_F(FunctionTest, TestCos) {
    TileShape::Current().SetVecTile({1, 1, 4, 4});
    config::SetPlatformConfig(KEY_ONLY_HOST_COMPILE, true);

    std::vector<int64_t> shape1 = {1, 2, 8, 8};
    Tensor input(DT_FP32, shape1, "input");
    Tensor res;

    FUNCTION("Cos") {
        res = Cos(input);
    }
    // Program::GetInstance().GraphCheck();

    ALOG_INFO(Program::GetInstance().Dump());
}

TEST_F(FunctionTest, TestGatherAxis0Indices2_1) {
    TileShape::Current().SetVecTile(32, 128);
    // TileShape::Current().SetVecTile(1, 32, 64);
    // tile graph
    config::SetPlatformConfig(KEY_ONLY_HOST_COMPILE, true);

    std::vector<int64_t> shape1 = {16, 1024};
    // std::vector<int64_t> shape1 = {16, 256};
    // std::vector<int64_t> shape1 = {16, 128};
    std::vector<int64_t> shape2 = {64};
    // std::vector<int64_t> resShape = {1, 64, 128};
    int axis = 0;
    Tensor params(DT_FP32, shape1, "params");
    Tensor indices(DT_INT32, shape2, "indices");
    Tensor res;

    FUNCTION("A") {
        res = Gather(params, indices, axis);
    }

    ALOG_INFO(Program::GetInstance().Dump());
}

TEST_F(FunctionTest, TestGatherAxis1Indices2_1) {
    TileShape::Current().SetVecTile(32, 128);
    // TileShape::Current().SetVecTile(1, 32, 64);
    // tile graph
    config::SetPlatformConfig(KEY_ONLY_HOST_COMPILE, true);

    std::vector<int64_t> shape1 = {1024, 16};
    // std::vector<int64_t> shape1 = {16, 256};
    // std::vector<int64_t> shape1 = {16, 128};
    std::vector<int64_t> shape2 = {64};
    // std::vector<int64_t> resShape = {1, 64, 128};
    int axis = 1;
    Tensor params(DT_FP32, shape1, "params");
    Tensor indices(DT_INT32, shape2, "indices");
    Tensor res;

    FUNCTION("A") {
        res = Gather(params, indices, axis);
    }

    ALOG_INFO(Program::GetInstance().Dump());
}

TEST_F(FunctionTest, TestGatherAxis3Indices4_2) {
    TileShape::Current().SetVecTile(4, 3, 8, 8, 8);
    config::SetPlatformConfig(KEY_ONLY_HOST_COMPILE, true);

    std::vector<int64_t> shape1 = {8, 8, 17, 20};
    std::vector<int64_t> shape2 = {32, 15};
    // std::vector<int64_t> resShape = {64, 512};
    int axis = 3;
    Tensor params(DT_FP32, shape1, "params");
    Tensor indices(DT_INT32, shape2, "indices");
    Tensor res;

    FUNCTION("A") {
        res = Gather(params, indices, axis);
    }

    ALOG_INFO(Program::GetInstance().Dump());
}

TEST_F(FunctionTest, TestGatherElementAxis1Indices2) {
    TileShape::Current().SetVecTile(8, 64);
    config::SetPlatformConfig(KEY_ONLY_HOST_COMPILE, true);

    std::vector<int64_t> shape1 = {32, 512};
    std::vector<int64_t> shape2 = {16,64};
    // std::vector<int64_t> resShape = {16,64};
    int axis = 1;
    Tensor params(DT_FP32, shape1, "params");
    Tensor indices(DT_INT32, shape2, "indices");
    Tensor res;

    FUNCTION("A") {
        res = GatherElements(params, indices, axis);
    }
    // Program::GetInstance().GraphCheck();
    //

    ALOG_INFO(Program::GetInstance().Dump());
}

TEST_F(FunctionTest, TestGatherElementAxis0Indices2) {
    TileShape::Current().SetVecTile(16, 32);
    config::SetPlatformConfig(KEY_ONLY_HOST_COMPILE, true);

    std::vector<int64_t> shape1 = {32, 512};
    std::vector<int64_t> shape2 = {16,64};
    // std::vector<int64_t> resShape = {16,64};
    int axis = 0;
    Tensor params(DT_FP32, shape1, "params");
    Tensor indices(DT_INT32, shape2, "indices");
    Tensor res;

    FUNCTION("A") {
        res = GatherElements(params, indices, axis);
    }
    // Program::GetInstance().GraphCheck();
    //

    ALOG_INFO(Program::GetInstance().Dump());
}

TEST_F(FunctionTest, TestScatter_) {
    int b = 2, s = 512, nRoutedExperts = 256, numExpertsPerTok = 8;
    TileShape::Current().SetVecTile(128, nRoutedExperts);

    Tensor cnts(DT_FP32, {b * s, nRoutedExperts}, "cnts");
    Tensor topk_ids(DT_INT32, {b * s, numExpertsPerTok}, "topk_ids");

    Tensor res;

    config::SetPlatformConfig(KEY_ONLY_HOST_COMPILE, true);

    FUNCTION("A") {
        res = Scatter_(cnts, topk_ids, Element(DataType::DT_FP32, 1.0), 1); // (b*s, nRoutedExperts)
    }
    ALOG_INFO(Program::GetInstance().Dump());
}

TEST_F(FunctionTest, TestScatterUpdate2) {
    int b = 2, s = 1, s2 = 512, kvLoraRank = 512, qkRopeHeadDim = 64;
    Tensor kv_len(DT_INT32, {1, 1}, "kv_len");
    Tensor past_key_states(DT_FP32, {b, 1, s2, kvLoraRank + qkRopeHeadDim}, "past_key_states");
    Tensor compressed_kv(DT_FP32, {b, s, kvLoraRank}, "past_key_states");
    Tensor k_pe_rope(DT_FP32, {b, 1, s, qkRopeHeadDim}, "k_pe_rope"); // (b,1,s,qkRopeHeadDim)
    Tensor res;
    Tensor key_states(DT_FP32, {b,1,s, kvLoraRank + qkRopeHeadDim}, "past_key_states");
    Tensor past_key_states_new(DT_FP32, {b,1,s2, kvLoraRank + qkRopeHeadDim}, "past_key_states_new");
    config::SetPlatformConfig(KEY_ONLY_HOST_COMPILE, true);

    FUNCTION("A") {
        TileShape::Current().SetVecTile(1, 1, 256, 128);
        past_key_states_new = ScatterUpdate(past_key_states, kv_len, key_states, -2);
    }
}

TEST_F(FunctionTest, testRowSumSingle) {
    config::SetPlatformConfig(KEY_ONLY_HOST_COMPILE, true);

    TileShape::Current().SetVecTile(1, 1, 32, 32);
    std::vector<int64_t> tshape = {2, 2, 64, 64};

    // TileShape::Current().SetVecTile(32, 32);
    // std::vector<int64_t> tshape = {64, 64};

    Tensor T(DT_FP32, tshape, "T");
    Tensor c, d;

    FUNCTION("A") {
        d = npu::tile_fwk::Sum(T, -1, true);
        c = npu::tile_fwk::Sum(T, -2, true);
    }
    // Program::GetInstance().GraphCheck();

    ALOG_INFO(Program::GetInstance().Dump());
}

TEST_F(FunctionTest, testRowMaxSingle) {
    config::SetPlatformConfig(KEY_ONLY_HOST_COMPILE, true);

    TileShape::Current().SetVecTile(1, 1, 32, 32);
    std::vector<int64_t> tshape = {1, 4, 64, 64};

    Tensor T(DT_FP32, tshape, "T");
    Tensor d;

    FUNCTION("A") {
        d = npu::tile_fwk::Amax(T);
    }
    // Program::GetInstance().GraphCheck();

    ALOG_INFO(Program::GetInstance().Dump());
}

TEST_F(FunctionTest, testSoftmax) {
    config::SetPlatformConfig(KEY_ONLY_HOST_COMPILE, true);

    TileShape::Current().SetVecTile(1, 1, 32, 32);
    std::vector<int64_t> tshape = {2, 2, 64, 64};

    Tensor T(DT_FP32, tshape, "T");
    Tensor d;

    FUNCTION("A") {
        d = SoftmaxNew(T);
    }
    // Program::GetInstance().GraphCheck();

    ALOG_INFO(Program::GetInstance().Dump());
}

TEST_F(FunctionTest, TestRoPE) {
    config::SetPlatformConfig(KEY_ONLY_HOST_COMPILE, true);

    RoPETileShapeConfig ropeTileConfig{
        {64, 64}, // for cos/sin->cast
        {1, 64, 64}, // for gather,unsqueeze
        {1, 64, 1, 64},
        {1, 64, 1, 32, 2} // for transpose
    };

    int b = 2;
    int n = 32;
    int s = 1;
    int d = 64;

    std::vector<int64_t> qShape{b, n, s, d};
    std::vector<int64_t> kShape{b, 1, s, d};
    std::vector<int64_t> idsShape{b, s};
    std::vector<int64_t> cosShape{s, d};

    Tensor q(DT_BF16, qShape, "q");
    Tensor k(DT_BF16, kShape, "k");
    Tensor cos(DT_BF16, cosShape, "cos");
    Tensor sin(DT_BF16, cosShape, "sin");
    Tensor positionIds(DT_INT32, idsShape, "positionIds");
    Tensor qEmbed(DT_BF16, qShape, "qEmbed");
    Tensor kEmbed(DT_BF16, kShape, "kEmbed");

    ConfigManager::Instance();
    FUNCTION("A") {
        ApplyRotaryPosEmb(q, k, cos, sin, positionIds, qEmbed, kEmbed, 1, ropeTileConfig);
    }

    ALOG_INFO(Program::GetInstance().Dump());
}

TEST_F(FunctionTest, TestRoPEDeepseekV3) {
    config::SetPlatformConfig(KEY_ONLY_HOST_COMPILE, true);

    RoPETileShapeConfig ropeTileConfig{
        {64, 64}, // for cos/sin->cast
        {1, 64, 64}, // for gather,unsqueeze
        {1, 64, 1, 64},
        {1, 64, 1, 32, 2} // for transpose
    };

    int B = 2;
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

    ALOG_INFO(Program::GetInstance().Dump());
}

TEST_F(FunctionTest, testRmsNormNewMultiDims) {
    config::SetPlatformConfig(KEY_ONLY_HOST_COMPILE, true);
    TileShape::Current().SetVecTile(1, 1, 8, 8);
    // Create some tensors (these would be created from elsewhere in your code)
    std::vector<int64_t> tshape = {2, 2, 24, 24};

    Tensor T(DT_FP32, tshape, "T");

    FUNCTION("Function_BLOCK1") {
        T = RmsNorm(T);
    }
    // Program::GetInstance().GraphCheck();
    ALOG_INFO(Program::GetInstance().Dump());
}

TEST_F(FunctionTest, TestConcat) {
    TileShape::Current().SetVecTile(16, 6, 6, 8);
    config::SetPlatformConfig(KEY_ONLY_HOST_COMPILE, true);

    std::vector<int64_t> shape1 = {10, 10, 10, 10};
    std::vector<int64_t> shape2 = {20, 10, 10, 10};
    int axis = 0;
    Tensor params1(DT_FP32, shape1, "params1");
    Tensor params2(DT_FP32, shape2, "params2");
    Tensor res;

    FUNCTION("A") {
        res = Cat(std::vector<Tensor>{params1, params2}, axis);
    }
    //Program::GetInstance().GraphCheck();

    ALOG_INFO(Program::GetInstance().Dump());
}

static std::map<std::string, std::variant<bool, int, float, std::string>> attnPostConfig = {
    {        "hiddenSize", 512},
    {       "kvLoraRank",  512},
    {"numAttentionHeads",    32},
    {         "vHeadDim",  128}
};

TEST_F(FunctionTest, TestAttentionPost) {
    config::SetPlatformConfig(KEY_ONLY_HOST_COMPILE, true);
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
        int f_b = attnPostIn.GetShape()[0];
        int f_n = attnPostIn.GetShape()[1];
        int f_s = attnPostIn.GetShape()[2];
        DataType dType = attnPostIn.GetStorage()->Datatype();
        TileShape::Current().SetVecTile({1, 1, 32, d});
        Tensor atten_res1 = Reshape(Transpose(attnPostIn, {1, 2}), {f_b * f_s, f_n, d});
        TileShape::Current().SetVecTile({32, 1, d});
        Tensor atten_res2 = Transpose(atten_res1, {0, 1});
        // [n,bs,kvLoraRank] * [n, kvLoraRank, vHeadDim] = [n,bs,vHeadDim]
        TileShape::Current().SetVecTile(128, 128);
        TileShape::Current().SetCubeTile({32, 32}, {128, 128}, {128, 128});
        Tensor mm7_res = Matrix::BatchMatmul(dType, atten_res2, kvBProjWV);
        // Tensor mm7_res = Matrix::BatchMatmul(dType, atten_res2, kvBProjWV);
        TileShape::Current().SetVecTile({1, 128, 128});
        Tensor mm7_res1 = Transpose(mm7_res, {0, 1});
        Tensor mm7_res2 = Reshape(mm7_res1, {f_b, f_s, f_n * v_head});

        // [b,s, n*vHeadDim] @ [n*vHeadDim, H] = [b,s,h]
        Tensor attn_out_w = Unsqueeze(oProjW, 0);
        atten_output = Matrix::BatchMatmul(dType, mm7_res2, attn_out_w);
    }
    ALOG_INFO(Program::GetInstance().Dump());
}

TEST_F(FunctionTest, Test_qkvPre) {
    config::SetPlatformConfig(KEY_ONLY_HOST_COMPILE, true);
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
    std::cout << "Test_qkvPre  b,s,h: " << b << ", " << s << ", " << h << std::endl;

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
    ConfigManager::Instance();
    FUNCTION("A") {
        res = Attention.QkvPre(hidden_states);
    }

    ALOG_INFO(Program::GetInstance().Dump());
}

TEST_F(FunctionTest, Test_qkvPre2) {
    config::SetPlatformConfig(KEY_ONLY_HOST_COMPILE, true);

    int& h = std::get<int>(g_deepseekConfig["hiddenSize"]);
    int& num_heads = std::get<int>(g_deepseekConfig["numAttentionHeads"]);
    int& qLoraRank = std::get<int>(g_deepseekConfig["qLoraRank"]);
    int& qkRopeHeadDim = std::get<int>(g_deepseekConfig["qkRopeHeadDim"]);
    int& kvLoraRank = std::get<int>(g_deepseekConfig["kvLoraRank"]);
    int& vHeadDim = std::get<int>(g_deepseekConfig["vHeadDim"]);
    int& qkNopeHeadDim = std::get<int>(g_deepseekConfig["qkNopeHeadDim"]);

    int b = 2;
    int s = 1;
    h = 256;
    num_heads = 2;
    qLoraRank = 512;
    qkNopeHeadDim = 128;
    qkRopeHeadDim = 64;
    kvLoraRank = 512;
    vHeadDim = 128;
    int q_head_dim = qkNopeHeadDim + qkRopeHeadDim;

    Tensor hidden_states = Tensor(DT_BF16, {b, s, h}, "hidden_states");  // [2,1,256]

    AttentionW aw;
    aw.qAProjW = Tensor(DT_BF16, {h, qLoraRank}, "qAProjW");  // [256,512]
    aw.qBProjW = Tensor(DT_BF16, {qLoraRank, num_heads * q_head_dim}, "qBProjW");  // [512,2*192]
    // [256,576]
    aw.kvAProjWithMqaW = Tensor(DT_BF16, {h, kvLoraRank + qkRopeHeadDim}, "kvAProjWithMqaW");
    aw.kvBProjWK = Tensor(DT_BF16, {num_heads, qkNopeHeadDim, kvLoraRank}, "kvBProjWK");
    aw.kvBProjWV = Tensor(DT_BF16, {num_heads, kvLoraRank, vHeadDim}, "kvBProjWV");
    aw.oProjW = Tensor(DT_BF16, {num_heads * vHeadDim, h}, "oProjW");

    std::vector<Tensor> res;
    DeepseekAttention Attention(deepseekConfig1, aw, 1);

    FUNCTION("Test_qkvPre2") {
        res = Attention.QkvPre2(hidden_states);
    }

    ALOG_INFO(Program::GetInstance().Dump());
}

TEST_F(FunctionTest, Test_deepseekAttention_pre) {
    config::SetPlatformConfig(KEY_ONLY_HOST_COMPILE, true);

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
    std::cout << "Test_deepseekAttention  b,s,h: " << b << ", " << s << ", " << h << std::endl;
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
    ALOG_INFO(Program::GetInstance().Dump());
}

TEST_F(FunctionTest, TestBMMtest) {
    config::SetPlatformConfig(KEY_ONLY_HOST_COMPILE, true);
    std::vector<int64_t> shape_a{2, 1, 256};
    std::vector<int64_t> shape_b{1, 256, 512};
    Tensor a(DT_FP16, shape_a, "a");
    Tensor b(DT_FP16, shape_b, "b");
    Tensor c;
    TileShape::Current().SetCubeTile({std::min(128, 1), std::min(128, 1)}, {128, 128}, {64, 64});
    FUNCTION("BMM") {
        c = npu::tile_fwk::Matrix::BatchMatmul(DT_FP16, a, b, false, false);
    }

    ALOG_INFO(Program::GetInstance().Dump());
}

TEST_F(FunctionTest, TestBMMtest2) {
    config::SetPlatformConfig(KEY_ONLY_HOST_COMPILE, true);
    std::vector<int64_t> shape_a{2, 1, 256};
    std::vector<int64_t> shape_b{1, 512, 256};
    Tensor a(DT_FP16, shape_a, "a");
    Tensor b(DT_FP16, shape_b, "b");
    Tensor c;
    TileShape::Current().SetCubeTile({std::min(128, 1), std::min(128, 1)}, {128, 128}, {64, 64});
    FUNCTION("BMM") {
        c = npu::tile_fwk::Matrix::BatchMatmul(DT_FP16, a, b, false, true);
    }

    ALOG_INFO(Program::GetInstance().Dump());
}

TEST_F(FunctionTest, Test_deepseekMoEGate) {
    config::SetPlatformConfig(KEY_ONLY_HOST_COMPILE, true);

    int b = 2; //  32
    int s = 1; //  1, optimize set_tile
    int h = std::get<int>(deepseekConfig1["hiddenSize"]);
    std::cout << "Test_deepseekAttention  b,s,h: " << b << ", " << s << ", " << h << std::endl;
    Tensor hidden_states = Tensor(DT_FP32, {b*s, h}, "hidden_states");

    Tensor topk_idx, topk_weight;
    MoEGate deepseekMoEGate(deepseekConfig1);

    TileShape::Current().SetCubeTile({std::min(128, s), std::min(128, s)}, {256, 256}, {64, 64});
    TileShape::Current().SetVecTile(128, 64); // for Assemble

    FUNCTION("A") {
        auto res = deepseekMoEGate.Forward(hidden_states);
        topk_idx = std::get<0>(res);
        topk_weight = std::get<1>(res);
    }
    // Program::GetInstance().GraphCheck();

    ALOG_INFO(Program::GetInstance().Dump());
}

TEST_F(FunctionTest, Test_deepseekMoEInfer) {
    config::SetPlatformConfig(KEY_ONLY_HOST_COMPILE, true);

    int b = 2;   //  32
    int s = 1; //  1, optimize set_tile
    int h = std::get<int>(deepseekConfig1["hiddenSize"]);
    int numExpertsPerTok = std::get<int>(deepseekConfig1["numExpertsPerTok"]);
    std::cout << "Test_deepseekAttention  b,s,h: " << b << ", " << s << ", " << h << std::endl;
    Tensor hidden_states = Tensor(DT_FP32, {b*s, h}, "hidden_states");
    Tensor topk_idx = Tensor(DT_INT32, {b*s, numExpertsPerTok}, "topk_idx");
    Tensor topk_weight =Tensor(DT_FP32, {b*s, numExpertsPerTok}, "topk_weight");
    DeepseekV2MoE deepseekMoEInfer(deepseekConfig1);

    Tensor res;

    TileShape::Current().SetCubeTile({std::min(128, s), std::min(128, s)}, {256, 256}, {64, 64});
    TileShape::Current().SetVecTile(128, 256); // for Assemble

    FUNCTION("A") {
        res = deepseekMoEInfer.MoeInfer(hidden_states, topk_idx, topk_weight);
    }
    // Program::GetInstance().GraphCheck();

    ALOG_INFO(Program::GetInstance().Dump());
}

TEST_F(FunctionTest, Test_deepseekMoEInfer_singleout) {
    config::SetPlatformConfig(KEY_ONLY_HOST_COMPILE, true);

    int32_t nRoutedExperts = 256;
    int b = 4;   // 32
    int s = 1; // 1, optimize set_tile
    int h = 256;
    int numExpertsPerTok = std::get<int>(deepseekConfig1["numExpertsPerTok"]); // 8

    DeepseekV2MoE deepseekMoEInfer(deepseekConfig1);

    std::vector<int64_t> hiddenStatesShape = { b* s, h };
    std::vector<int64_t> topKShape = { b* s, numExpertsPerTok };

    Tensor ffnWeight1(DT_FP16, {h, h * 3}, "ffnWeight1");
    Tensor ffnWeight2(DT_FP16, {h, h * 3}, "ffnWeight2");
    Tensor ffnWeight3(DT_FP16, {h, h * 3}, "ffnWeight3");

    Tensor finalout(DT_FP32, { b*s, h }, "finalout");

    TileShape::Current().SetCubeTile({64, 64}, {64, 64}, {64, 64});
    TileShape::Current().SetVecTile(64, nRoutedExperts); // for Assemble

    Tensor hiddenStates = Tensor(DT_FP32, hiddenStatesShape, "hiddenStates");
    Tensor topkIdx = Tensor(DT_INT32, topKShape, "topkIdx");
    Tensor topkWeight = Tensor(DT_FP32, topKShape, "topkWeight");

    FUNCTION("MOE_INFER_F") {
            finalout = deepseekMoEInfer.MoeInfer(hiddenStates, topkIdx, topkWeight, ffnWeight1, ffnWeight2, ffnWeight3, nRoutedExperts);
    }
}

TEST_F(FunctionTest, test_deepseekMoEInfer_singleout_2) {
    config::SetPlatformConfig(KEY_ONLY_HOST_COMPILE, true);

    int32_t nRoutedExperts = 256;
    int b = 4; // 32
    int s = 1;  // 1, optimize set_tile
    int h = std::get<int>(deepseekConfig1["hiddenSize"]);
    int numExpertsPerTok = std::get<int>(deepseekConfig1["numExpertsPerTok"]); // 8

    DeepseekV2MoE deepseekMoEInfer(deepseekConfig1);

    std::vector<int64_t> hiddenStatesShape = {b * s, h};
    std::vector<int64_t> topKShape = {b * s, numExpertsPerTok};
    std::vector<int64_t> resShape = {b * s, numExpertsPerTok};

    Tensor ffnWeight1(DataType::DT_FP16, {h, h * 3}, "ffnWeight1");
    Tensor ffnWeight2(DataType::DT_FP16, {h, h * 3}, "ffnWeight2");
    Tensor ffnWeight3(DataType::DT_FP16, {h, h * 3}, "ffnWeight3");

    Tensor outs(DataType::DT_FP32, {b * s * numExpertsPerTok, h}, "outs");
    Tensor sortedTokens(DataType::DT_FP32, {b * s * numExpertsPerTok, h}, "sortedTokens");
    Tensor idxs(DataType::DT_INT32, {b * s * numExpertsPerTok}, "idxs");

    Tensor finalout(DataType::DT_FP32, { b*s, h }, "finalout");

    TileShape::Current().SetCubeTile({std::min(128, b * s), std::min(128, b * s)}, {64, 64}, {64, 64});
    TileShape::Current().SetVecTile(64, nRoutedExperts); // for Assemble

    Tensor hiddenStates = Tensor(DataType::DT_FP32, hiddenStatesShape, "hiddenStates");
    Tensor topkIdx = Tensor(DataType::DT_INT32, topKShape, "topkIdx");
    Tensor topkWeight = Tensor(DataType::DT_FP32, topKShape, "topkWeight");

    FUNCTION("MOE_INFER_F") {
        finalout = deepseekMoEInfer.MoeInfer(hiddenStates, topkIdx, topkWeight, ffnWeight1, ffnWeight2,
            ffnWeight3, idxs, sortedTokens, outs, nRoutedExperts);
    }
}

TEST_F(FunctionTest, test_deepseekMoEInfer_singleout_singlemlp) {
    config::SetPlatformConfig(KEY_ONLY_HOST_COMPILE, true);

    int32_t nRoutedExperts = 256;
    int b = 4;
    int s = 1;
    int h = std::get<int>(deepseekConfig1["hiddenSize"]);
    int weightN = 2048;
    int numExpertsPerTok = std::get<int>(deepseekConfig1["numExpertsPerTok"]); // 8

    DeepseekV2MoE deepseekMoEInfer(deepseekConfig1);

    std::vector<int64_t> hiddenStatesShape = { b* s, h };
    std::vector<int64_t> topKShape = { b* s, numExpertsPerTok };
    std::vector<int64_t> resShape = { b* s, numExpertsPerTok };

    Tensor ffnWeight1(DT_FP16, {h, weightN}, "ffnWeight1");
    Tensor ffnWeight2(DT_FP16, {h, weightN}, "ffnWeight2");
    Tensor ffnWeight3(DT_FP16, {h, weightN},"ffnWeight3");

    Tensor finalout(DT_FP32, { b*s, h }, "finalout");

    TileShape::Current().SetCubeTile({64, 64}, {64, 64}, {64, 64});
    TileShape::Current().SetVecTile(64, nRoutedExperts); // for Assemble

    Tensor hiddenStates = Tensor(DT_FP32, hiddenStatesShape, "hiddenStates");
    Tensor topkIdx = Tensor(DT_INT32, topKShape, "topkIdx");
    Tensor topkWeight = Tensor(DT_FP32, topKShape, "topkWeight");

    FUNCTION("MOE_INFER_F") {
            finalout = deepseekMoEInfer.MoeInferSingleMlp(hiddenStates, topkIdx, topkWeight, ffnWeight1, ffnWeight2, ffnWeight3, nRoutedExperts);
    }
}

TEST_F(FunctionTest, test_deepseekMoEInfer_singleout_singlemlp_withquant) {
    config::SetPlatformConfig(KEY_ONLY_HOST_COMPILE, true);

    int32_t nRoutedExperts = 256;
    int b = 4;
    int s = 1;
    int h = std::get<int>(deepseekConfig1["hiddenSize"]);
    int weightN = 2048;
    int numExpertsPerTok = std::get<int>(deepseekConfig1["numExpertsPerTok"]); // 8

    DeepseekV2MoE deepseekMoEInfer(deepseekConfig1);

    std::vector<int64_t> hiddenStatesShape = { b* s, h };
    std::vector<int64_t> topKShape = { b* s, numExpertsPerTok };
    std::vector<int64_t> resShape = { b* s, numExpertsPerTok };

    Tensor ffnWeight1(DT_INT8, {h, weightN}, "ffnWeight1", TileOpFormat::TILEOP_NZ);
    Tensor ffnWeight2(DT_INT8, {h, weightN}, "ffnWeight2", TileOpFormat::TILEOP_NZ);
    Tensor ffnWeight3(DT_INT8, {h, weightN}, "ffnWeight3", TileOpFormat::TILEOP_NZ);
    Tensor ffnwight1Scale(DT_FP32, {1, weightN}, "ffnwight1Scale");
    Tensor ffnwight2Scale(DT_FP32, {1, weightN}, "ffnwight2Scale");
    Tensor ffnwight3Scale(DT_FP32, {h, 1}, "ffnwight3Scale");

    Tensor finalout(DT_FP32, { b*s, h }, "finalout");

    TileShape::Current().SetCubeTile({64, 64}, {64, 64}, {64, 64});
    TileShape::Current().SetVecTile(64, nRoutedExperts); // for Assemble

    Tensor hiddenStates = Tensor(DT_FP32, hiddenStatesShape, "hiddenStates");
    Tensor topkIdx = Tensor(DT_INT32, topKShape, "topkIdx");
    Tensor topkWeight = Tensor(DT_FP32, topKShape, "topkWeight");

    FUNCTION("MOE_INFER_F") {
            finalout = deepseekMoEInfer.MoeInferSingleMlpQuant(hiddenStates, topkIdx, topkWeight, ffnWeight1, ffnWeight2, ffnWeight3,  ffnwight1Scale, ffnwight2Scale, ffnwight3Scale, nRoutedExperts);
    }
}

TEST_F(FunctionTest, Test_quant) {
    config::SetPlatformConfig(KEY_ONLY_HOST_COMPILE, true);

    std::vector<int64_t> vecTileShape  = {128, 128};
    int b = 32;  // 32
    int s = 1;  // 1, optimize set_tile
    int h = 7168;
    std::cout << "Test_deepseekAttention  b,s,h: " << b << ", " << s << ", " << h << std::endl;

    Tensor input = Tensor(DT_FP16, {b, s, h}, "input");
    Tensor res;

    TileShape::Current().SetCubeTile({std::min(128, s), std::min(128, s)}, {256, 256}, {64, 64});
    TileShape::Current().SetVecTile(1, vecTileShape[0], vecTileShape[1]); // for Assemble

    FUNCTION("A") {
        res = std::get<0>(Quant(input));
    }

    ALOG_INFO(Program::GetInstance().Dump());
}

TEST_F(FunctionTest, Test_ScalarOp) {
    config::SetPlatformConfig(KEY_ONLY_HOST_COMPILE, true);

    std::vector<int64_t> shape = {128, 32};
    TileShape::Current().SetVecTile({128, 32});
    Tensor input_a(DT_FP32, shape, "A");
    auto output = Tensor(DT_FP32, shape, "res"); // std::make_tuple(Tensor(DT_FP32, shape, "res"), Tensor(DT_FP32, shape, "resDics"));
    config::SetBuildStatic(true);
    FUNCTION("ScalarAddS") {
        auto a = ScalarAddS(input_a, Element(DataType::DT_FP32, F_127), true);
        auto b = ScalarSubS(a, Element(DataType::DT_FP32, F_127), true);
        auto c = ScalarMulS(b, Element(DataType::DT_FP32, F_127), true);
        auto d = ScalarDivS(c, Element(DataType::DT_FP32, F_127), true);
        output = ScalarMaxS(d, Element(DataType::DT_FP32, F_127), true);
    }

    ALOG_INFO(Program::GetInstance().Dump());
}

TEST_F(FunctionTest, TestPad) {
    config::SetPlatformConfig(KEY_ONLY_HOST_COMPILE, true);
    std::vector<int64_t> shape{8, 16};
    std::vector<int64_t> newShape{8, 24};
    Tensor a(DT_FP32, shape, "a");
    Tensor b;
    TileShape::Current().SetVecTile(8, 8);

    config::SetBuildStatic(true);
    FUNCTION("Pad") {
        b = Pad(a, newShape);
    }
    ALOG_INFO(Program::GetInstance().Dump());
}

TEST_F_WITH_COST(FunctionTest, Test_quantMM, 95) {
    config::SetPlatformConfig(KEY_ONLY_HOST_COMPILE, true);

    std::vector<int64_t> vecTileShape  = {32, 512};
    int m = 128;
    int k = 16384;
    int n = 7168;

    Tensor inputA = Tensor(DT_BF16, {m, k}, "inputA");
    Tensor inputW = Tensor(DT_INT8, {k, n}, "inputW");
    Tensor inputScaleW = Tensor(DT_FP32, {1, n}, "inputScaleW");
    Tensor res;

    TileShape::Current().SetCubeTile({32, 32}, {128, 128}, {128, 128});
    TileShape::Current().SetVecTile(32, 64); // for Assemble

    FUNCTION("A") {
        res = npu::tile_fwk::Matrix::QuantMM(inputA, inputW, inputScaleW);
    }
}

TEST_F(FunctionTest, TestRmsNorm) {

    std::vector<int64_t> shapea{8, 16};
    std::vector<int64_t> shapeb{16};
    Tensor a(DT_FP32, shapea, "a");
    Tensor b(DT_FP32, shapeb, "b");
    Tensor c;
    TileShape::Current().SetVecTile(8, 8);
    FUNCTION("RmsNorm") {
        c = RmsNorm(a, b, 1e-5f);
    }
    ALOG_INFO(Program::GetInstance().Dump());
}

TEST_F(FunctionTest, dynamic_pa_low_lantency) {
    config::SetPlatformConfig(KEY_ONLY_HOST_COMPILE, true);
    std::vector<int64_t> input_param = {4, 1, 32, 1, 512, 64, 128, 32};
    int b = input_param[0];
    int sq = input_param[1];
    int nq = input_param[2];
    int nk = input_param[3];
    int dn = input_param[4];
    int dr = input_param[5];
    int blockSize = input_param[6];
    int nTile = input_param[7];
    float softmaxScale = static_cast<float>(1.0 / sqrtf((dn + dr)));

    PaTileShapeConfig tileConfig;
    tileConfig.headNumQTile = nTile;
    tileConfig.v0TileShape = {nTile, 64};
    tileConfig.c1TileShape = {nTile, nTile, 64, 64, blockSize, blockSize};
    tileConfig.v1TileShape = {nTile, 64};
    tileConfig.c2TileShape = {nTile, nTile, 64, 64, blockSize, blockSize};
    tileConfig.v2TileShape = {nTile, 64};

    std::vector<int> seq(b, 256);

    int blockNum = 0;
    for (auto s : seq) {
        blockNum += ((s + (blockSize - 1)) / blockSize);
    }
    // blockTable: (b, maxBlockNumPerBatch)
    int maxSeqAllBatch = *(std::max_element(seq.begin(), seq.end()));
    int maxBlockNumPerBatch = ((maxSeqAllBatch + (blockSize - 1)) / blockSize);

    Tensor qNope(DT_BF16, {b * nq * sq, dn}, "qNope");
    Tensor kNopeCache(DT_BF16, {int(blockNum * blockSize), nk * dn}, "kNopeCache");
    Tensor vNopeCache(DT_BF16, {int(blockNum * blockSize), nk * dn}, "vNopeCache");
    Tensor qRope(DT_BF16, {b * nq * sq, nk * dr}, "qRope");
    Tensor kRopeCache(DT_BF16, {int(blockNum * blockSize), nk * dr}, "kRope");
    Tensor blockTable(DT_INT32, {b, maxBlockNumPerBatch}, "blockTable");
    Tensor actSeqs(DT_INT32, {b}, "actSeqs");
    Tensor paOut(DT_FP32, {b * nq * sq, dn}, "paOut");

    PageAttention(qNope, kNopeCache, vNopeCache, qRope, kRopeCache, blockTable, actSeqs, blockSize, softmaxScale, paOut,
        tileConfig);

    auto mainFunc = Program::GetInstance().GetFunctionByMagicName("TENSOR_main_2");
    EXPECT_NE(mainFunc, nullptr);
    EXPECT_EQ(mainFunc->GetCalleeFunctionList().size(), 1);
    auto loopFunc1 = mainFunc->GetCalleeFunctionList().front();
    EXPECT_NE(loopFunc1, nullptr);
    EXPECT_EQ(loopFunc1->GetFunctionType(), FunctionType::DYNAMIC_LOOP);
    EXPECT_EQ(loopFunc1->GetGraphType(), GraphType::TENSOR_GRAPH);
    EXPECT_EQ(loopFunc1->GetCalleeFunctionList().size(), 1);
    auto loopPathFunc1 = loopFunc1->GetCalleeFunctionList().front();
    EXPECT_NE(loopPathFunc1, nullptr);
    EXPECT_EQ(loopPathFunc1->GetFunctionType(), FunctionType::DYNAMIC_LOOP_PATH);
    EXPECT_EQ(loopPathFunc1->GetCalleeFunctionList().size(), 1);
    auto loopFunc2 = loopPathFunc1->GetCalleeFunctionList().front();
    EXPECT_NE(loopFunc2, nullptr);
    EXPECT_EQ(loopFunc2->GetFunctionType(), FunctionType::DYNAMIC_LOOP);
    EXPECT_EQ(loopFunc2->GetCalleeFunctionList().size(), 1);
    auto loopPathFunc2 = loopFunc2->GetCalleeFunctionList().front();
    EXPECT_NE(loopPathFunc2, nullptr);
    EXPECT_EQ(loopPathFunc2->GetFunctionType(), FunctionType::DYNAMIC_LOOP_PATH);
    EXPECT_EQ(loopPathFunc2->GetCalleeFunctionList().size(), 1);
    auto loopFunc3 = loopPathFunc2->GetCalleeFunctionList().front();
    EXPECT_NE(loopFunc3, nullptr);
    EXPECT_EQ(loopFunc3->GetFunctionType(), FunctionType::DYNAMIC_LOOP);
#if ENABLE_HIDDENLOOP
    EXPECT_EQ(loopFunc3->GetCalleeFunctionList().size(), 1);
    auto loopPathFunc3 = loopFunc3->GetCalleeFunctionList().front();
    EXPECT_NE(loopPathFunc3, nullptr);
    EXPECT_EQ(loopPathFunc3->GetFunctionType(), FunctionType::DYNAMIC_LOOP_PATH);
    EXPECT_EQ(loopPathFunc3->GetCalleeFunctionList().size(), 1);
    auto loopFunc4 = loopPathFunc3->GetCalleeFunctionList().front();
    EXPECT_NE(loopFunc4, nullptr);
    EXPECT_EQ(loopFunc4->GetFunctionType(), FunctionType::DYNAMIC_LOOP);
    EXPECT_EQ(loopFunc4->GetCalleeFunctionList().size(), 4);
    for (auto loopPathFunc4 : loopFunc4->GetCalleeFunctionList()) {
        EXPECT_NE(loopPathFunc4, nullptr);
        EXPECT_EQ(loopPathFunc4->GetFunctionType(), FunctionType::DYNAMIC_LOOP_PATH);
        EXPECT_EQ(loopPathFunc4->GetGraphType(), GraphType::TENSOR_GRAPH);
        EXPECT_EQ(loopPathFunc4->GetCalleeFunctionList().size(), 1);
        auto hiddenLoopPathFunc4 = loopPathFunc4->GetCalleeFunctionList().front();
        EXPECT_EQ(hiddenLoopPathFunc4->GetFunctionType(), FunctionType::DYNAMIC_LOOP_PATH);
        EXPECT_EQ(hiddenLoopPathFunc4->GetGraphType(), GraphType::TILE_GRAPH);
    }
#else
    EXPECT_EQ(loopFunc3->GetCalleeFunctionList().size(), 4);
    for (auto loopPathFunc3 : loopFunc3->GetCalleeFunctionList()) {
        EXPECT_NE(loopPathFunc3, nullptr);
        EXPECT_EQ(loopPathFunc3->GetFunctionType(), FunctionType::DYNAMIC_LOOP_PATH);
        EXPECT_EQ(loopPathFunc3->GetGraphType(), GraphType::TILE_GRAPH);
    }
#endif
}


template <typename T = npu::tile_fwk::float16, typename wDtype = int8_t, bool splitK = false, bool nz = true,
    bool isSmooth = true, bool usePrefetch = true>
void TestMlaPrologV2(const SimpleParams &params) {
    pre();

    int b = params.b;
    int s = params.s;
    int s2 = params.s2;
    int n = params.n;
    int h = params.h;
    int qLoraRank = params.q_lora_rank;
    int qkNopeHeadDim = params.qk_nope_head_dim;
    int qkRopeHeadDim = params.qk_rope_head_dim;
    int kvLoraRank = params.kv_lora_rank;
    int q_head_dim = params.q_head_dim;

    DataType dType = (std::is_same<T, npu::tile_fwk::float16>::value) ? DT_FP16 : DT_BF16;
    bool isQuant = std::is_same<wDtype, int8_t>::value;
    DataType dTypeQuant = isQuant ? DT_INT8 : dType;

    std::vector<int64_t> x_shape = {b, s, h};
    std::vector<int64_t> wDqShape = {h, qLoraRank};
    std::vector<int64_t> wUqQrShape = {qLoraRank, n * q_head_dim};
    std::vector<int64_t> wDkvKrShape = {h, kvLoraRank + qkRopeHeadDim};
    std::vector<int64_t> wUkShape = {n, qkNopeHeadDim, kvLoraRank};
    std::vector<int64_t> cos_shape = {b, s, qkRopeHeadDim};
    std::vector<int64_t> gamma_cq_shape = {qLoraRank};
    std::vector<int64_t> gamma_ckv_shape = {kvLoraRank};
    std::vector<int64_t> kv_len_shape = {b, s};
    std::vector<int64_t> kv_cache_shape = {b, 1, s2, kvLoraRank};
    std::vector<int64_t> kr_cache_shape = {b, 1, s2, qkRopeHeadDim};
    if (params.cacheMode == "PA_BSND") {
        int blockNum = b * (s2 / params.blockSize);
        kv_cache_shape = {blockNum, params.blockSize, 1, kvLoraRank};
        kr_cache_shape = {blockNum, params.blockSize, 1, qkRopeHeadDim};
    }
    std::vector<int64_t> w_qb_scale_shape = {1, n * q_head_dim};
    std::vector<int64_t> smooth_cq_shape{1, qLoraRank};
    // output
    std::vector<int64_t> q_out_shape = {b, s, n, kvLoraRank};
    std::vector<int64_t> q_rope_out_shape = {b, s, n, qkRopeHeadDim};
    std::vector<int64_t> kv_cache_out_shape = {b, 1, s2, kvLoraRank};
    std::vector<int64_t> kr_cache_out_shape = {b, 1, s2, qkRopeHeadDim};

    Tensor x(dType, x_shape, "x");
    TileOpFormat weightFormat = nz ? TileOpFormat::TILEOP_NZ : TileOpFormat::TILEOP_ND;
    Tensor wDq(dType, wDqShape, "wDq", weightFormat);
    Tensor wUqQr(dTypeQuant, wUqQrShape, "wUqQr", weightFormat);
    if constexpr (usePrefetch) {
        wDq.SetCachePolicy(CachePolicy::PREFETCH, true);
        wUqQr.SetCachePolicy(CachePolicy::PREFETCH, true);
    }
    Tensor wDkvKr(dType, wDkvKrShape, "wDkvKr", weightFormat);
    Tensor wUk(dType, wUkShape, "wUk", weightFormat);
    Tensor gamma_cq(dType, gamma_cq_shape, "gamma_cq");
    Tensor gamma_ckv(dType, gamma_ckv_shape, "gamma_ckv");
    Tensor cos(dType, cos_shape, "cos");
    Tensor sin(dType, cos_shape, "sin");
    Tensor kv_len(DT_INT64, kv_len_shape, "kv_len"); // int64
    Tensor kv_cache(dType, kv_cache_shape, "kv_cache");
    Tensor kr_cache(dType, kr_cache_shape, "kr_cache");
    Tensor w_qb_scale(DT_FP32, w_qb_scale_shape, "w_qb_scale");
    Tensor smooth_cq(DT_FP32, smooth_cq_shape, "smooth_cq");

    // output
    Tensor output_kv_cache(dType, kv_cache_shape, "output_kv_cache");
    Tensor output_kr_cache(dType, kr_cache_shape, "output_kr_cache");
    Tensor output_q(dType, q_out_shape, "output_q");
    Tensor output_q_rope(dType, q_rope_out_shape, "output_q_rope");

    RoPETileShapeConfigNew ropeConfig{
        {b, 1, 64}, // (b,s,d)
        {b, 1, 1, 64}, // Q (b,s,n,d)
        {b, 1, 1, 64}, // K (b,s,1,d)
        {b, 1, 1, 32, 2}  // (b,s,n,d//2,2)
    };

    MlaQuantInputs quantInputs;
    if (isQuant) {
        quantInputs.dequantScaleWUqQr = w_qb_scale;
        if (isSmooth) {
            quantInputs.smoothScalesCq = smooth_cq;
        }
    }
    config::SetPassConfig("PVC2_OOO", "InferMemoryConflict", KEY_DISABLE_PASS, true);
    MlaProlog(x, wDq, wUqQr, wUk, wDkvKr, gamma_cq, gamma_ckv, sin, cos, kv_len, kv_cache, kr_cache, quantInputs,
        ropeConfig, output_q, output_q_rope, output_kv_cache, output_kr_cache, 1e-5f, 1e-5f, params.cacheMode, splitK,
        isSmooth);
}

TEST_F(FunctionTest, low) {
    TestMlaPrologV2<npu::tile_fwk::float16>(SimpleParams::getLowParams());
}
TEST_F(FunctionTest, low_PAND) {
    config::SetPlatformConfig(KEY_ONLY_HOST_COMPILE, true);
    npu::tile_fwk::SimpleParams params = SimpleParams::getLowParams();
    params.cacheMode = "PA_BSND";
    TestMlaPrologV2<npu::tile_fwk::float16, int8_t, true>(params);
}

TEST_F(FunctionTest, dynamic_page_attention_adds) {
    config::SetHostOption(ONLY_CODEGEN, true);

    config::SetPlatformConfig(KEY_ENABLE_AIHAC_BACKEND, true);
    std::vector<uint8_t> devProgBinary;

    std::vector<int> input_param = {4, 1, 32, 1, 512, 64, 128, 32};
    int b = input_param[0];
    int sq = input_param[1];
    int nq = input_param[2];
    int nk = input_param[3];
    int dn = input_param[4];
    int dr = input_param[5];
    int blockSize = input_param[6];
    int nTile = input_param[7];
    float softmaxScale = static_cast<float>(1.0 / sqrtf((dn + dr)));

    PaTileShapeConfig tileConfig;
    tileConfig.headNumQTile = nTile;
    tileConfig.v0TileShape = {nTile, 64};
    tileConfig.c1TileShape = {nTile, nTile, 64, 64, blockSize, blockSize};
    tileConfig.v1TileShape = {nTile, 64};
    tileConfig.c2TileShape = {nTile, nTile, 64, 64, blockSize, blockSize};
    tileConfig.v2TileShape = {nTile, 64};
    std::vector<int> seq(b, 256);

    int blockNum = 0;
    for (auto s : seq) {
        blockNum += ((s + (blockSize - 1)) / blockSize);
    }
    int maxSeqAllBatch = *(std::max_element(seq.begin(), seq.end()));
    int maxBlockNumPerBatch = ((maxSeqAllBatch + (blockSize - 1)) / blockSize);

    Tensor qNope(DT_BF16, {b * nq * sq, dn}, "qNope");
    Tensor kNopeCache(DT_BF16, {int(blockNum * blockSize), nk * dn}, "kNopeCache");
    Tensor vNopeCache(DT_BF16, {int(blockNum * blockSize), nk * dn}, "vNopeCache");
    Tensor qRope(DT_BF16, {b * nq * sq, nk * dr}, "qRope");
    Tensor kRopeCache(DT_BF16, {int(blockNum * blockSize), nk * dr}, "kRope");
    Tensor blockTable(DT_INT32, {b, maxBlockNumPerBatch}, "blockTable");
    Tensor actSeqs(DT_INT32, {b}, "actSeqs");
    Tensor paOut(DT_FP32, {b * nq * sq, dn}, "paOut");
    Tensor postOut(DT_FP32, {b * nq * sq, dn}, "postOut");

    int maxUnrollTimes = 1;
    PageAttentionAddS(qNope, kNopeCache, vNopeCache, qRope, kRopeCache, blockTable, actSeqs, blockSize, softmaxScale, paOut, postOut,
        tileConfig, maxUnrollTimes);
}

TEST_F(FunctionTest, dynamic_page_attention_adds_single_single_out) {
    config::SetHostOption(ONLY_CODEGEN, true);

    config::SetPlatformConfig(KEY_ENABLE_AIHAC_BACKEND, true);
    std::vector<uint8_t> devProgBinary;

    std::vector<int> input_param = {4, 1, 32, 1, 512, 64, 128, 32};
    int b = input_param[0];
    int sq = input_param[1];
    int nq = input_param[2];
    int nk = input_param[3];
    int dn = input_param[4];
    int dr = input_param[5];
    int blockSize = input_param[6];
    int nTile = input_param[7];
    float softmaxScale = static_cast<float>(1.0 / sqrtf((dn + dr)));

    PaTileShapeConfig tileConfig;
    tileConfig.headNumQTile = nTile;
    tileConfig.v0TileShape = {nTile, 64};
    tileConfig.c1TileShape = {nTile, nTile, 64, 64, blockSize, blockSize};
    tileConfig.v1TileShape = {nTile, 64};
    tileConfig.c2TileShape = {nTile, nTile, 64, 64, blockSize, blockSize};
    tileConfig.v2TileShape = {nTile, 64};
    std::vector<int> seq(b, 256);

    int blockNum = 0;
    for (auto s : seq) {
        blockNum += ((s + (blockSize - 1)) / blockSize);
    }
    int maxSeqAllBatch = *(std::max_element(seq.begin(), seq.end()));
    int maxBlockNumPerBatch = ((maxSeqAllBatch + (blockSize - 1)) / blockSize);

    Tensor qNope(DT_BF16, {b * nq * sq, dn}, "qNope");
    Tensor kNopeCache(DT_BF16, {int(blockNum * blockSize), nk * dn}, "kNopeCache");
    Tensor vNopeCache(DT_BF16, {int(blockNum * blockSize), nk * dn}, "vNopeCache");
    Tensor qRope(DT_BF16, {b * nq * sq, nk * dr}, "qRope");
    Tensor kRopeCache(DT_BF16, {int(blockNum * blockSize), nk * dr}, "kRope");
    Tensor blockTable(DT_INT32, {b, maxBlockNumPerBatch}, "blockTable");
    Tensor actSeqs(DT_INT32, {b}, "actSeqs");
    Tensor paOut(DT_FP32, {b * nq * sq, dn}, "paOut");
    Tensor postOut(DT_FP32, {b * nq * sq, dn}, "postOut");

    int maxUnrollTimes = 1;
    PageAttentionAddSSingleOutput(qNope, kNopeCache, vNopeCache, qRope, kRopeCache, blockTable, actSeqs, blockSize, softmaxScale, paOut, postOut,
        tileConfig, maxUnrollTimes);
}
