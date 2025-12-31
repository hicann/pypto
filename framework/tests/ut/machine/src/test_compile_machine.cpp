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
 * \file test_compile_machine.cpp
 * \brief
 */

#include <gtest/gtest.h>
#include "acl/acl.h"
#include "runtime/rt.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "tilefwk/data_type.h"
#include "interface/operation/operation.h"
#include "interface/configs/config_manager.h"
#include "interface/tensor/float.h"
#include "operator/models/deepseek/deepseek_mla.h"
#include "machine/runtime/machine_agent.h"
#include "machine/host/backend.h"

using namespace npu::tile_fwk;

class HostMachineCompileTest : public testing::Test {
  public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}

    void SetUp() override {
        Program::GetInstance().Reset();
    }

    void TearDown() override {}
};

template <typename T = npu::tile_fwk::float16>
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

    DataType dType = DataType::DT_FP32;
    if (std::is_same<T, npu::tile_fwk::float16>::value) {
        dType = DataType::DT_FP16;
    } else if (std::is_same<T, npu::tile_fwk::bfloat16>::value) {
        dType = DataType::DT_BF16;
    } else {
        dType = DataType::DT_FP32;
    }

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

    aclInit(nullptr);
    rtSetDevice(0);


    PROGRAM("MlaProlog") {
        Tensor x(dType, x_shape, "x");
        Tensor w_qa(dType, w_qa_shape, "w_qa");
        w_qa.SetCachePolicy(CachePolicy::PREFETCH, true);
        Tensor w_qb(dType, w_qb_shape, "w_qb");
        w_qb.SetCachePolicy(CachePolicy::PREFETCH, true);
        Tensor w_kv_a(dType, w_kv_a_shape, "w_kv_a");
        Tensor w_kv_b_k(dType, w_kv_b_k_shape, "w_kv_b_k");
        Tensor position_ids(DataType::DT_INT32, position_ids_shape, "position_ids");
        Tensor cos(dType, cos_shape, "cos");
        Tensor sin(dType, cos_shape, "sin");
        Tensor past_key_states(dType, past_key_states_shape, "past_key_states");
        Tensor kv_len(DataType::DT_INT32, kv_len_shape, "kv_len");
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